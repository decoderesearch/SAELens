#!/usr/bin/env python
"""Pareto Sweep: PhaseSAE vs TopK across sparsity budgets K ∈ {8, 16, 24}.

Trains both architectures on TinyStories-1M (blocks.0.hook_resid_post) for each K,
evaluates downstream CE loss recovery, explained variance, MSE, and L0 via sae_lens.evals.

Usage:
    .venv/bin/python scripts/run_pareto_sweep.py
"""

import json
import time
from pathlib import Path
import torch

from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.training.activation_scaler import ActivationScaler
from tests.helpers import (
    TINYSTORIES_MODEL,
    NEEL_NANDA_C4_10K_DATASET,
    build_phase_multiplexed_runner_cfg,
    build_topk_runner_cfg,
    load_model_cached,
)

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = ARTIFACT_DIR / "pareto_sweep_results.json"

K_VALUES = [8, 16, 24]
NUM_PHASES = 4
TRAINING_TOKENS = 16000

def main():
    print("=" * 75)
    print("  PHASESAE VS TOP-K PARETO SWEEP BENCHMARK (K ∈ {8, 16, 24})")
    print("=" * 75)

    print("\n1. Loading base model: TinyStories-1M (CPU)...")
    model = load_model_cached(TINYSTORIES_MODEL)
    d_in = model.cfg.d_model  # 64
    d_sae = 256  # 4x expansion

    common_cfg = dict(
        d_in=d_in,
        d_sae=d_sae,
        training_tokens=TRAINING_TOKENS,
        store_batch_size_prompts=4,
        train_batch_size_tokens=32,
        model_batch_size=2,
        context_size=32,
        n_batches_in_buffer=4,
        dataset_path=NEEL_NANDA_C4_10K_DATASET,
        hook_name="blocks.0.hook_resid_post",
        model_name=TINYSTORIES_MODEL,
        n_checkpoints=0,
        exclude_special_tokens=True,
        save_final_checkpoint=False,
    )

    eval_cfg = EvalConfig(
        n_eval_reconstruction_batches=10,
        compute_ce_loss=True,
        compute_kl=True,
        compute_variance_metrics=True,
        compute_sparsity_metrics=True,
        compute_featurewise_density_statistics=True,
    )

    results = {}

    for k in K_VALUES:
        print(f"\n{'='*75}")
        print(f"  SWEEP POINT: K = {k}")
        print(f"{'='*75}")
        results[str(k)] = {}

        # ── 1. Top-K Baseline ──
        print(f"\n[K={k}] Training Baseline Top-K...")
        topk_runner_cfg = build_topk_runner_cfg(
            k=k,
            output_path=f"artifacts/pareto_sweep/topk_k{k}",
            **common_cfg,
        )
        t0 = time.time()
        topk_runner = LanguageModelSAETrainingRunner(topk_runner_cfg, override_model=model)
        topk_sae = topk_runner.run()
        topk_time = time.time() - t0

        print(f"[K={k}] Evaluating Top-K...")
        topk_metrics, _ = run_evals(
            topk_sae,
            topk_runner.activations_store,
            model,
            activation_scaler=ActivationScaler(),
            eval_config=eval_cfg,
        )

        topk_ce = float(topk_metrics["model_performance_preservation"].get("ce_loss_score", 0.0))
        topk_r2 = float(topk_metrics["reconstruction_quality"].get("explained_variance", 0.0))
        topk_mse = float(topk_metrics["reconstruction_quality"].get("mse", 0.0))
        topk_l0 = float(topk_metrics["sparsity"].get("l0", float(k)))
        topk_peak_mb = (d_in * d_sae * 4) / (1024 * 1024)

        results[str(k)]["topk"] = {
            "ce_loss_recovery": topk_ce,
            "explained_variance": topk_r2,
            "mse": topk_mse,
            "l0": topk_l0,
            "peak_stream_param_slice_mb": topk_peak_mb,
            "train_time_s": topk_time,
        }

        # ── 2. PhaseSAE ──
        k_per_phase = max(1, k // NUM_PHASES)
        actual_total_k = k_per_phase * NUM_PHASES
        print(f"\n[K={k}] Training PhaseSAE (P={NUM_PHASES}, k_pp={k_per_phase}, total_k={actual_total_k})...")
        phase_runner_cfg = build_phase_multiplexed_runner_cfg(
            num_phases=NUM_PHASES,
            k_per_phase=k_per_phase,
            output_path=f"artifacts/pareto_sweep/phase_k{k}",
            **common_cfg,
        )
        t0 = time.time()
        phase_runner = LanguageModelSAETrainingRunner(phase_runner_cfg, override_model=model)
        phase_sae = phase_runner.run()
        phase_time = time.time() - t0

        print(f"[K={k}] Evaluating PhaseSAE...")
        phase_metrics, _ = run_evals(
            phase_sae,
            phase_runner.activations_store,
            model,
            activation_scaler=ActivationScaler(),
            eval_config=eval_cfg,
        )

        phase_ce = float(phase_metrics["model_performance_preservation"].get("ce_loss_score", 0.0))
        phase_r2 = float(phase_metrics["reconstruction_quality"].get("explained_variance", 0.0))
        phase_mse = float(phase_metrics["reconstruction_quality"].get("mse", 0.0))
        phase_l0 = float(phase_metrics["sparsity"].get("l0", float(actual_total_k)))
        phase_peak_mb = (d_in * (d_sae // NUM_PHASES) * 4) / (1024 * 1024)

        results[str(k)]["phasesae"] = {
            "ce_loss_recovery": phase_ce,
            "explained_variance": phase_r2,
            "mse": phase_mse,
            "l0": phase_l0,
            "k_per_phase": k_per_phase,
            "num_phases": NUM_PHASES,
            "peak_stream_param_slice_mb": phase_peak_mb,
            "train_time_s": phase_time,
        }

        # ── 3. Delta ──
        ce_delta = (phase_ce - topk_ce) / (abs(topk_ce) + 1e-9) * 100
        r2_delta = (phase_r2 - topk_r2) / (abs(topk_r2) + 1e-9) * 100
        mse_delta = (phase_mse - topk_mse) / (topk_mse + 1e-9) * 100
        bw_factor = topk_peak_mb / phase_peak_mb

        results[str(k)]["delta"] = {
            "ce_recovery_rel_gain_pct": float(ce_delta),
            "r2_rel_gain_pct": float(r2_delta),
            "mse_rel_pct_change": float(mse_delta),
            "bandwidth_reduction_factor": float(bw_factor),
        }

        print(f"\n--> K={k} Comparison:")
        print(f"    Top-K:    CE Rec={topk_ce*100:.2f}%  R²={topk_r2:.4f}  MSE={topk_mse:.4f}  Peak={topk_peak_mb:.4f}MB")
        print(f"    PhaseSAE: CE Rec={phase_ce*100:.2f}%  R²={phase_r2:.4f}  MSE={phase_mse:.4f}  Peak={phase_peak_mb:.4f}MB")
        print(f"    Δ: CE Rec={ce_delta:+.1f}%  R²={r2_delta:+.1f}%  BW={bw_factor:.1f}x reduction")

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all sweep results → {OUT_FILE}")

    print("\n" + "="*85)
    print(f"{'Target K':>8} | {'Architecture':>12} | {'CE Rec (%)':>11} | {'R²':>7} | {'MSE':>8} | {'Peak MB':>8} | {'BW Red.':>8}")
    print("-"*85)
    for k in K_VALUES:
        r = results[str(k)]
        t = r["topk"]
        p = r["phasesae"]
        print(f"{k:>8} | {'Top-K':>12} | {t['ce_loss_recovery']*100:>10.2f}% | {t['explained_variance']:>7.4f} | {t['mse']:>8.4f} | {t['peak_stream_param_slice_mb']:>8.4f} | {'1.0x':>8}")
        bw_str = f"{r['delta']['bandwidth_reduction_factor']:.1f}x"
        print(f"{k:>8} | {'PhaseSAE':>12} | {p['ce_loss_recovery']*100:>10.2f}% | {p['explained_variance']:>7.4f} | {p['mse']:>8.4f} | {p['peak_stream_param_slice_mb']:>8.4f} | {bw_str:>8}")
        print("-"*85)
    print("="*85)

if __name__ == "__main__":
    main()
