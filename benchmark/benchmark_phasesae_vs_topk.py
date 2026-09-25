import json
import time

from sae_lens.evals import EvalConfig, run_evals
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.training.activation_scaler import ActivationScaler
from tests.helpers import (
    NEEL_NANDA_C4_10K_DATASET,
    TINYSTORIES_MODEL,
    build_phase_multiplexed_runner_cfg,
    build_topk_runner_cfg,
    load_model_cached,
)


def run_benchmark():
    print("=" * 70)
    print("  PHASESAE VS TOP-K EMPIRICAL BENCHMARK (DOWNSTREAM EVALS)")
    print("=" * 70)

    print("\n1. Loading base model: TinyStories-1M (CPU)...")
    model = load_model_cached(TINYSTORIES_MODEL)
    d_in = model.cfg.d_model  # 64
    d_sae = 256  # 4x expansion
    k_total = 16
    training_tokens = 16000

    # Common training parameters
    common_cfg = dict(
        d_in=d_in,
        d_sae=d_sae,
        training_tokens=training_tokens,
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

    # 2. Train Baseline: TopK SAE
    print(
        f"\n2. Training Baseline Top-K SAE (d_sae={d_sae}, k={k_total}, tokens={training_tokens})..."
    )
    topk_runner_cfg = build_topk_runner_cfg(
        k=k_total,
        output_path="artifacts/eval_topk",
        **common_cfg,
    )
    t0 = time.time()
    topk_runner = LanguageModelSAETrainingRunner(topk_runner_cfg, override_model=model)
    topk_sae = topk_runner.run()
    topk_train_time = time.time() - t0
    print(f"   Top-K Training completed in {topk_train_time:.1f}s")

    # 3. Train PhaseSAE: 4 Phases x 4 Latents = 16 Total
    print(
        f"\n3. Training PhaseSAE (d_sae={d_sae}, P=4, k_phase=4, total_k={k_total}, tokens={training_tokens})..."
    )
    phase_runner_cfg = build_phase_multiplexed_runner_cfg(
        num_phases=4,
        k_per_phase=4,
        output_path="artifacts/eval_phasesae",
        **common_cfg,
    )
    t0 = time.time()
    phase_runner = LanguageModelSAETrainingRunner(
        phase_runner_cfg, override_model=model
    )
    phase_sae = phase_runner.run()
    phase_train_time = time.time() - t0
    print(f"   PhaseSAE Training completed in {phase_train_time:.1f}s")

    # 4. Run Downstream Evals via sae_lens.evals
    print(
        "\n4. Running Downstream Evals (CE Loss Recovery, KL Divergence, Explained Variance)..."
    )
    eval_cfg = EvalConfig(
        n_eval_reconstruction_batches=10,
        compute_ce_loss=True,
        compute_kl=True,
        compute_variance_metrics=True,
        compute_sparsity_metrics=True,
        compute_featurewise_density_statistics=True,
    )

    print("   Evaluating Top-K Baseline...")
    topk_metrics, _ = run_evals(
        topk_sae,
        topk_runner.activations_store,
        model,
        activation_scaler=ActivationScaler(),
        eval_config=eval_cfg,
    )

    print("   Evaluating PhaseSAE...")
    phase_metrics, _ = run_evals(
        phase_sae,
        phase_runner.activations_store,
        model,
        activation_scaler=ActivationScaler(),
        eval_config=eval_cfg,
    )

    # 5. Extract Metrics
    results = {
        "model": TINYSTORIES_MODEL,
        "dataset": NEEL_NANDA_C4_10K_DATASET,
        "d_in": d_in,
        "d_sae": d_sae,
        "target_k": k_total,
        "training_tokens": training_tokens,
        "topk": {
            "ce_loss_score": topk_metrics["model_performance_preservation"].get(
                "ce_loss_score", 0.0
            ),
            "kl_div_score": topk_metrics["model_behavior_preservation"].get(
                "kl_div_score", 0.0
            ),
            "explained_variance": topk_metrics["reconstruction_quality"].get(
                "explained_variance", 0.0
            ),
            "mse": topk_metrics["reconstruction_quality"].get("mse", 0.0),
            "l0": topk_metrics["sparsity"].get("l0", 0.0),
            "training_time_s": topk_train_time,
            "peak_stream_param_slice_mb": (d_in * d_sae * 4) / (1024 * 1024),
        },
        "phase_sae": {
            "num_phases": 4,
            "k_per_phase": 4,
            "ce_loss_score": phase_metrics["model_performance_preservation"].get(
                "ce_loss_score", 0.0
            ),
            "kl_div_score": phase_metrics["model_behavior_preservation"].get(
                "kl_div_score", 0.0
            ),
            "explained_variance": phase_metrics["reconstruction_quality"].get(
                "explained_variance", 0.0
            ),
            "mse": phase_metrics["reconstruction_quality"].get("mse", 0.0),
            "l0": phase_metrics["sparsity"].get("l0", 0.0),
            "training_time_s": phase_train_time,
            "peak_stream_param_slice_mb": (d_in * (d_sae // 4) * 4) / (1024 * 1024),
        },
    }

    with open("artifacts/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Top-K (Baseline)':<18} | {'PhaseSAE (P=4)':<18}")
    print("-" * 75)
    print(
        f"{'CE Loss Score (% Recovered)':<35} | {results['topk']['ce_loss_score']*100:>16.2f}% | {results['phase_sae']['ce_loss_score']*100:>16.2f}%"
    )
    print(
        f"{'KL Divergence Score':<35} | {results['topk']['kl_div_score']:>17.4f} | {results['phase_sae']['kl_div_score']:>17.4f}"
    )
    print(
        f"{'Explained Variance (R^2)':<35} | {results['topk']['explained_variance']:>17.4f} | {results['phase_sae']['explained_variance']:>17.4f}"
    )
    print(
        f"{'MSE Reconstruction Error':<35} | {results['topk']['mse']:>17.4f} | {results['phase_sae']['mse']:>17.4f}"
    )
    print(
        f"{'L0 (Active Latents)':<35} | {results['topk']['l0']:>17.2f} | {results['phase_sae']['l0']:>17.2f}"
    )
    print(
        f"{'Peak Stream Param Footprint':<35} | {results['topk']['peak_stream_param_slice_mb']:>14.2f} MB | {results['phase_sae']['peak_stream_param_slice_mb']:>14.2f} MB (1/4)"
    )
    print("=" * 70)
    print(
        "\nBenchmark successfully completed and saved to artifacts/benchmark_results.json"
    )


if __name__ == "__main__":
    run_benchmark()
