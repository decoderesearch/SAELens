#!/usr/bin/env python3
"""Scale and Pareto Benchmark Suite for PhaseSAE vs Top-K SAE.

Evaluates the scaling performance and Pareto frontier of PhaseMultiplexedSAE (P=4, P=8)
against baseline TopKSAE across sparsity regimes K in [8, 16, 24, 32] on target models
including tiny-stories-1M (d_in=64) and gpt2-small (d_in=768, Layer 8 hook_resid_post).

Tracks:
  - Downstream Cross-Entropy (CE) loss recovery score (% recovered)
  - Kullback-Leibler (KL) divergence score
  - Explained variance (R^2)
  - MSE reconstruction error
  - Empirical L0 sparsity
  - Peak memory bandwidth / parameter streaming footprint (1/P)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformer_lens import HookedTransformer

from sae_lens.evals import EvalConfig, run_evals
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.training.activation_scaler import ActivationScaler
from tests.helpers import (
    NEEL_NANDA_C4_10K_DATASET,
    TINYSTORIES_MODEL,
    build_phase_multiplexed_runner_cfg,
    build_topk_runner_cfg,
)

# Supported Model Specifications
MODEL_SPECS = {
    "tiny-stories-1M": {
        "model_name": TINYSTORIES_MODEL,
        "d_in": 64,
        "hook_name": "blocks.0.hook_resid_post",
        "dataset_path": NEEL_NANDA_C4_10K_DATASET,
        "default_context_size": 32,
        "default_training_tokens": 16000,
        "store_batch_size_prompts": 4,
        "train_batch_size_tokens": 32,
        "model_batch_size": 2,
        "n_batches_in_buffer": 4,
    },
    "gpt2-small": {
        "model_name": "gpt2-small",
        "d_in": 768,
        "hook_name": "blocks.8.hook_resid_post",
        "dataset_path": NEEL_NANDA_C4_10K_DATASET,
        "default_context_size": 64,
        "default_training_tokens": 32000,
        "store_batch_size_prompts": 4,
        "train_batch_size_tokens": 32,
        "model_batch_size": 2,
        "n_batches_in_buffer": 4,
    },
}

_MODEL_CACHE: Dict[str, HookedTransformer] = {}


def get_cached_model(model_name: str, device: str = "cpu") -> HookedTransformer:
    """Load and cache the base transformer model."""
    cache_key = f"{model_name}_{device}"
    if cache_key not in _MODEL_CACHE:
        print(f"Loading base model '{model_name}' on device '{device}'...")
        model = HookedTransformer.from_pretrained(model_name, device=device)
        _MODEL_CACHE[cache_key] = model
    return _MODEL_CACHE[cache_key]


def compute_streaming_memory_footprint(
    d_in: int, d_sae: int, num_phases: int, bytes_per_param: int = 4
) -> Dict[str, Any]:
    """Compute active parameter streaming slice and memory bandwidth footprint.

    Monolithic Top-K (P=1) requires full dictionary materialization in memory bandwidth.
    PhaseMultiplexedSAE (P micro-ticks) partitions parameters into P sequential tiles,
    yielding a 1/P parameter streaming footprint per micro-tick.
    """
    total_encoder_decoder_params = 2 * d_in * d_sae
    total_dict_bytes = total_encoder_decoder_params * bytes_per_param

    streaming_slice_params = total_encoder_decoder_params // num_phases
    streaming_slice_bytes = streaming_slice_params * bytes_per_param
    streaming_slice_mb = streaming_slice_bytes / (1024 * 1024)

    return {
        "num_phases": num_phases,
        "bandwidth_footprint_fraction": 1.0 / num_phases,
        "bandwidth_reduction_factor": float(num_phases),
        "total_dict_params": total_encoder_decoder_params,
        "total_dict_bytes": total_dict_bytes,
        "total_dict_mb": total_dict_bytes / (1024 * 1024),
        "peak_streaming_params_per_tick": streaming_slice_params,
        "peak_streaming_bytes_per_tick": streaming_slice_bytes,
        "peak_stream_param_slice_mb": streaming_slice_mb,
    }


def to_python_scalar(val: Any) -> Any:
    """Recursively convert PyTorch tensors and numpy types to JSON-serializable Python scalars."""
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        return val.tolist()
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, dict):
        return {k: to_python_scalar(v) for k, v in val.items()}
    if isinstance(val, list):
        return [to_python_scalar(v) for v in val]
    return val


def train_and_eval_configuration(
    model: HookedTransformer,
    model_spec: Dict[str, Any],
    architecture: str,  # "topk" or "phase_multiplexed"
    target_k: int,
    num_phases: int,  # 1 for TopK, 4 or 8 for PhaseSAE
    d_sae: int,
    training_tokens: int,
    eval_batches: int,
    device: str,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Train a single SAE configuration and evaluate all downstream Pareto metrics."""
    d_in = model_spec["d_in"]
    model_name = model_spec["model_name"]
    hook_name = model_spec["hook_name"]
    dataset_path = model_spec["dataset_path"]
    context_size = model_spec.get("default_context_size", 32)
    store_batch_size = model_spec.get("store_batch_size_prompts", 4)
    train_batch_size = model_spec.get("train_batch_size_tokens", 32)
    model_batch_size = model_spec.get("model_batch_size", 2)
    n_batches_in_buffer = model_spec.get("n_batches_in_buffer", 4)

    run_id = f"{model_name}_{architecture}_p{num_phases}_k{target_k}_{int(time.time())}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    common_cfg = dict(
        d_in=d_in,
        d_sae=d_sae,
        training_tokens=training_tokens,
        store_batch_size_prompts=store_batch_size,
        train_batch_size_tokens=train_batch_size,
        model_batch_size=model_batch_size,
        context_size=context_size,
        n_batches_in_buffer=n_batches_in_buffer,
        dataset_path=dataset_path,
        hook_name=hook_name,
        model_name=model_name,
        device=device,
        act_store_device=device,
        seed=seed,
        n_checkpoints=0,
        exclude_special_tokens=True,
        save_final_checkpoint=False,
        output_path=str(run_dir),
        verbose=False,
    )

    if architecture == "topk":
        arch_label = "Top-K Baseline (Monolithic)"
        runner_cfg = build_topk_runner_cfg(
            k=target_k,
            **common_cfg,
        )
    elif architecture == "phase_multiplexed":
        arch_label = f"PhaseMultiplexedSAE (P={num_phases})"
        k_per_phase = target_k // num_phases
        if k_per_phase < 1:
            raise ValueError(
                f"Target K ({target_k}) must be >= num_phases ({num_phases})"
            )
        runner_cfg = build_phase_multiplexed_runner_cfg(
            num_phases=num_phases,
            k_per_phase=k_per_phase,
            **common_cfg,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    print(
        f"\n--> Training {arch_label} | Target K={target_k} | "
        f"d_in={d_in} | d_sae={d_sae} | Tokens={training_tokens}..."
    )
    t_train_start = time.time()
    runner = LanguageModelSAETrainingRunner(runner_cfg, override_model=model)
    sae = runner.run()
    train_time_s = time.time() - t_train_start
    print(f"    [Done] Training completed in {train_time_s:.2f}s")

    print(
        f"--> Running Evals via sae_lens.evals (batches={eval_batches}, CE loss, KL, R^2, MSE)..."
    )
    t_eval_start = time.time()
    eval_cfg = EvalConfig(
        n_eval_reconstruction_batches=eval_batches,
        compute_ce_loss=True,
        compute_kl=True,
        compute_variance_metrics=True,
        compute_sparsity_metrics=True,
        compute_featurewise_density_statistics=False,
    )

    metrics_raw, _ = run_evals(
        sae,
        runner.activations_store,
        model,
        activation_scaler=ActivationScaler(),
        eval_config=eval_cfg,
    )
    eval_time_s = time.time() - t_eval_start
    print(f"    [Done] Evaluations completed in {eval_time_s:.2f}s")

    # Extract key Pareto frontier metrics
    ce_loss_score = float(
        metrics_raw.get("model_performance_preservation", {}).get("ce_loss_score", 0.0)
    )
    kl_div_score = float(
        metrics_raw.get("model_behavior_preservation", {}).get("kl_div_score", 0.0)
    )
    explained_variance = float(
        metrics_raw.get("reconstruction_quality", {}).get("explained_variance", 0.0)
    )
    mse = float(metrics_raw.get("reconstruction_quality", {}).get("mse", 0.0))
    l0 = float(metrics_raw.get("sparsity", {}).get("l0", float(target_k)))
    l1 = float(metrics_raw.get("sparsity", {}).get("l1", 0.0))

    # Streaming footprint metrics
    footprint = compute_streaming_memory_footprint(d_in, d_sae, num_phases)

    result_record = {
        "run_id": run_id,
        "model_name": model_name,
        "hook_name": hook_name,
        "architecture": architecture,
        "arch_display": arch_label,
        "num_phases": num_phases,
        "target_k": target_k,
        "k_per_phase": target_k // num_phases if num_phases > 1 else target_k,
        "d_in": d_in,
        "d_sae": d_sae,
        "training_tokens": training_tokens,
        "train_time_s": train_time_s,
        "eval_time_s": eval_time_s,
        # Downstream & Representation Quality
        "ce_loss_score": ce_loss_score,
        "ce_loss_pct": ce_loss_score * 100.0,
        "kl_div_score": kl_div_score,
        "explained_variance": explained_variance,
        "mse": mse,
        "l0": l0,
        "l1": l1,
        # Parameter Streaming & Memory Bandwidth
        "peak_stream_param_slice_mb": footprint["peak_stream_param_slice_mb"],
        "bandwidth_footprint_fraction": footprint["bandwidth_footprint_fraction"],
        "bandwidth_reduction_factor": footprint["bandwidth_reduction_factor"],
        "total_dict_mb": footprint["total_dict_mb"],
        # Raw metrics snapshot
        "raw_metrics": to_python_scalar(metrics_raw),
    }

    print(
        f"    Results: CE Loss Score={ce_loss_score*100:.2f}% | "
        f"KL Div={kl_div_score:.4f} | R^2={explained_variance:.4f} | "
        f"MSE={mse:.4f} | L0={l0:.1f} | Stream Slice={footprint['peak_stream_param_slice_mb']:.4f} MB ({100.0/num_phases:.1f}%)"
    )

    return result_record


def generate_pareto_markdown_table(results: List[Dict[str, Any]]) -> str:
    """Format benchmark results into a clean markdown table comparing the Pareto frontier."""
    lines = []
    lines.append(
        "| Architecture | Total K | L0 (Empirical) | CE Loss Recov (%) | KL Div Score | Expl. Variance ($R^2$) | MSE Recon | Bandwidth (1/P) | Param Footprint |"
    )
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in results:
        arch_tag = (
            "Top-K Baseline"
            if r["architecture"] == "topk"
            else f"PhaseSAE (P={r['num_phases']})"
        )
        ce_pct = f"{r['ce_loss_pct']:+.2f}%"
        kl_score = f"{r['kl_div_score']:.4f}"
        ev_score = f"{r['explained_variance']:.4f}"
        mse_score = f"{r['mse']:.4f}"
        l0_score = f"{r['l0']:.2f}"
        bw_frac = f"{r['bandwidth_footprint_fraction']*100:.1f}%"
        footprint_mb = f"{r['peak_stream_param_slice_mb']:.4f} MB"

        lines.append(
            f"| **{arch_tag}** | {r['target_k']} | {l0_score} | {ce_pct} | {kl_score} | {ev_score} | {mse_score} | {bw_frac} | {footprint_mb} |"
        )

    return "\n".join(lines)


def run_benchmark_suite(
    model_name: str = "tiny-stories-1M",
    k_values: Optional[List[int]] = None,
    phases: Optional[List[int]] = None,
    include_baseline: bool = True,
    training_tokens: Optional[int] = None,
    eval_batches: int = 10,
    expansion_factor: int = 4,
    device: str = "auto",
    seed: int = 42,
    output_json: str = "artifacts/pareto_sweep_results.json",
) -> Dict[str, Any]:
    """Execute complete scale and Pareto benchmark sweep across architectures and K values."""
    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"Unsupported model: {model_name}. Available: {list(MODEL_SPECS.keys())}"
        )

    if k_values is None:
        k_values = [8, 16, 24]
    if phases is None:
        phases = [4, 8]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_spec = MODEL_SPECS[model_name]
    if training_tokens is None:
        training_tokens = model_spec.get("default_training_tokens", 16000)

    d_in = model_spec["d_in"]
    d_sae = d_in * expansion_factor

    print("=" * 80)
    print("  PHASESAE FLAGSHIP SCALING & PARETO BENCHMARK SUITE")
    print("=" * 80)
    print(
        f"Target Model      : {model_name} (d_in={d_in}, Hook={model_spec['hook_name']})"
    )
    print(f"Dictionary Width  : d_sae={d_sae} ({expansion_factor}x expansion)")
    print(f"Sparsity Grid (K) : {k_values}")
    print(f"Phase Tilings (P) : {phases}")
    print(f"Include Baseline  : {include_baseline} (Top-K)")
    print(f"Training Tokens   : {training_tokens}")
    print(f"Eval Batches      : {eval_batches}")
    print(f"Device            : {device}")
    print(f"Output Artifact   : {output_json}")
    print("=" * 80)

    # Load model
    base_model = get_cached_model(model_name, device=device)

    # Setup directories
    out_json_path = Path(output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_json_path.parent / "pareto_runs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    suite_start = time.time()
    all_results: List[Dict[str, Any]] = []

    # Sweep across K values
    for k in k_values:
        print(
            f"\n====================== SPARSITY REGIME K = {k} ======================"
        )

        # 1. Baseline TopK
        if include_baseline:
            try:
                res_topk = train_and_eval_configuration(
                    model=base_model,
                    model_spec=model_spec,
                    architecture="topk",
                    target_k=k,
                    num_phases=1,
                    d_sae=d_sae,
                    training_tokens=training_tokens,
                    eval_batches=eval_batches,
                    device=device,
                    seed=seed,
                    output_dir=temp_dir,
                )
                all_results.append(res_topk)
            except Exception as e:
                print(f"ERROR running Top-K for K={k}: {e}", file=sys.stderr)
                raise

        # 2. PhaseMultiplexed configurations (e.g. P=4, P=8)
        for p in phases:
            if k % p != 0:
                print(
                    f"Skipping PhaseSAE P={p} for K={k} because K is not divisible by P."
                )
                continue

            try:
                res_phase = train_and_eval_configuration(
                    model=base_model,
                    model_spec=model_spec,
                    architecture="phase_multiplexed",
                    target_k=k,
                    num_phases=p,
                    d_sae=d_sae,
                    training_tokens=training_tokens,
                    eval_batches=eval_batches,
                    device=device,
                    seed=seed,
                    output_dir=temp_dir,
                )
                all_results.append(res_phase)
            except Exception as e:
                print(f"ERROR running PhaseSAE P={p} for K={k}: {e}", file=sys.stderr)
                raise

    suite_duration_s = time.time() - suite_start

    # Build final suite payload
    suite_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_name": model_name,
            "d_in": d_in,
            "d_sae": d_sae,
            "expansion_factor": expansion_factor,
            "hook_name": model_spec["hook_name"],
            "dataset_path": model_spec["dataset_path"],
            "k_values": k_values,
            "phases": phases,
            "training_tokens": training_tokens,
            "eval_batches": eval_batches,
            "device": device,
            "seed": seed,
            "suite_duration_s": suite_duration_s,
        },
        "results": all_results,
    }

    # Save to JSON
    with open(out_json_path, "w") as f:
        json.dump(suite_payload, f, indent=2)
    print(f"\n[Saved] Raw metrics successfully saved to: {out_json_path}")

    # Generate Markdown Table
    md_table = generate_pareto_markdown_table(all_results)
    print("\n" + "=" * 80)
    print("  PARETO FRONTIER COMPARISON TABLE")
    print("=" * 80)
    print(md_table)
    print("=" * 80)

    return suite_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flagship Scaling & Pareto Benchmark Runner for PhaseSAE vs TopK"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny-stories-1M",
        choices=list(MODEL_SPECS.keys()),
        help="Target transformer model (tiny-stories-1M or gpt2-small)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[8, 16, 24],
        help="Sparsity sweep total K values (default: 8 16 24)",
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=[4, 8],
        help="Number of sequential phase tiles P for PhaseSAE (default: 4 8)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline Top-K SAE runs",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="Training tokens per SAE (default: from model spec)",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=10,
        help="Reconstruction & downstream evaluation batches (default: 10)",
    )
    parser.add_argument(
        "--expansion-factor",
        type=int,
        default=4,
        help="SAE expansion factor d_sae / d_in (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on ('cpu', 'cuda', or 'auto')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/pareto_sweep_results.json",
        help="Path to save output JSON metrics",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark_suite(
        model_name=args.model,
        k_values=args.k_values,
        phases=args.phases,
        include_baseline=not args.no_baseline,
        training_tokens=args.tokens,
        eval_batches=args.eval_batches,
        expansion_factor=args.expansion_factor,
        device=args.device,
        seed=args.seed,
        output_json=args.output_json,
    )
