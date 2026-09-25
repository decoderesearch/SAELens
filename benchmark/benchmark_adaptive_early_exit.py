#!/usr/bin/env python
"""Adaptive Early-Exit (Anytime SAE) Benchmark.

Evaluates PhaseSAE with per-token dynamic early exit across residual error thresholds:
  exit_threshold ∈ [None, 0.75, 0.70, 0.65, 0.60]

Simulates autoregressive inference (B=1 per-token early termination):
  - Average phases evaluated per token (effective compute / latency scaling)
  - Reconstruction R² and MSE
  - Effective active L0
  - Effective compute & bandwidth speedup relative to full P ticks

Usage:
    .venv/bin/python scripts/benchmark_adaptive_early_exit.py
"""

import json
import time
from pathlib import Path

import torch

from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from tests.helpers import (
    NEEL_NANDA_C4_10K_DATASET,
    TINYSTORIES_MODEL,
    build_phase_multiplexed_runner_cfg,
    load_model_cached,
)

MODEL_NAME = TINYSTORIES_MODEL
HOOK_POINT = "blocks.0.hook_resid_post"
N_TOKENS_EVAL = 5000
D_IN = 64
D_SAE = 256
NUM_PHASES = 4
K_PER_PHASE = 4
TRAINING_TOKENS = 16000

EXIT_THRESHOLDS = [None, 0.75, 0.70, 0.65, 0.60]

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = ARTIFACT_DIR / "adaptive_early_exit_results.json"


def get_eval_activations(model, n_tokens):
    from datasets import load_dataset

    dataset = load_dataset(NEEL_NANDA_C4_10K_DATASET, split="train", streaming=True)
    all_acts = []
    for ex in dataset:
        text = ex["text"]
        tokens = model.to_tokens(text[:256])
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=HOOK_POINT)
        acts = cache[HOOK_POINT].squeeze(0)
        all_acts.append(acts)
        if sum(a.shape[0] for a in all_acts) >= n_tokens:
            break
    return torch.cat(all_acts, dim=0)[:n_tokens].cpu()


def main():
    print("=" * 75)
    print("  PHASESAE ADAPTIVE EARLY-EXIT (ANYTIME SAE) BENCHMARK")
    print("=" * 75)

    print("\n1. Loading TinyStories-1M …")
    model = load_model_cached(MODEL_NAME)

    print("\n2. Training PhaseSAE via LanguageModelSAETrainingRunner …")
    runner_cfg = build_phase_multiplexed_runner_cfg(
        d_in=D_IN,
        d_sae=D_SAE,
        num_phases=NUM_PHASES,
        k_per_phase=K_PER_PHASE,
        training_tokens=TRAINING_TOKENS,
        store_batch_size_prompts=4,
        train_batch_size_tokens=32,
        model_batch_size=2,
        context_size=32,
        n_batches_in_buffer=4,
        dataset_path=NEEL_NANDA_C4_10K_DATASET,
        hook_name=HOOK_POINT,
        model_name=MODEL_NAME,
        output_path="artifacts/eval_early_exit/phase_sae",
        n_checkpoints=0,
        exclude_special_tokens=True,
        save_final_checkpoint=False,
    )
    t0 = time.time()
    runner = LanguageModelSAETrainingRunner(runner_cfg, override_model=model)
    sae = runner.run()
    print(f"   PhaseSAE training finished in {time.time() - t0:.1f}s")

    print(
        f"\n3. Collecting {N_TOKENS_EVAL} activations for anytime early-exit evaluation …"
    )
    acts = get_eval_activations(model, N_TOKENS_EVAL)
    ss_tot = (acts - acts.mean(0, keepdim=True)).pow(2).sum().item()

    sae_in = sae.process_sae_in(acts)
    initial_norm = sae_in.norm(dim=-1)

    # Collect per-phase outputs
    ticks = list(sae.stream_phase_ticks(acts))
    # Each tick: (p, acts_p, recon_p, cur_residual)

    results = {}
    print("\n" + "=" * 80)
    print(
        f"{'Exit Thresh':>12} | {'R²':>7} | {'MSE':>8} | {'Avg Phases':>10} | {'Avg L0':>8} | {'Speedup':>8}"
    )
    print("-" * 80)

    base_phases = float(NUM_PHASES)
    N = acts.shape[0]

    for thresh in EXIT_THRESHOLDS:
        label = "None (Full)" if thresh is None else f"{thresh:.2f}"

        with torch.no_grad():
            token_active = torch.ones(N, dtype=torch.bool)
            phases_per_token = torch.zeros(N)
            final_recon = torch.zeros_like(sae_in)
            final_acts = torch.zeros(N, D_SAE)
            m = D_SAE // NUM_PHASES

            for p, acts_p, recon_p, cur_residual in ticks:
                # Accumulate for active tokens
                mask = token_active.unsqueeze(-1)
                final_recon += recon_p * mask.float()
                final_acts[:, p * m : (p + 1) * m] = acts_p * mask.float()
                phases_per_token[token_active] += 1

                if thresh is not None:
                    res_ratio = cur_residual.norm(dim=-1) / (initial_norm + 1e-8)
                    exited_now = token_active & (res_ratio < thresh)
                    token_active = token_active & (~exited_now)

            if sae.cfg.apply_b_dec_to_input:
                final_recon += sae.b_dec

            ss_res = (acts - final_recon).pow(2).sum().item()
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            mse = (acts - final_recon).pow(2).mean().item()

            avg_phases = float(phases_per_token.mean().item())
            avg_l0 = float((final_acts > 0).float().sum(dim=-1).mean().item())
            speedup = base_phases / max(avg_phases, 1e-6)

        thresh_key = "full" if thresh is None else str(thresh)
        results[thresh_key] = {
            "exit_threshold": thresh,
            "r2": float(r2),
            "mse": float(mse),
            "avg_phases_evaluated": float(avg_phases),
            "avg_l0": float(avg_l0),
            "effective_compute_speedup": float(speedup),
        }

        print(
            f"{label:>12} | {r2:>7.4f} | {mse:>8.5f} | {avg_phases:>10.2f} | {avg_l0:>8.2f} | {speedup:>7.2f}x"
        )

    print("=" * 80)

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved adaptive early-exit benchmark results → {OUT_FILE}")


if __name__ == "__main__":
    main()
