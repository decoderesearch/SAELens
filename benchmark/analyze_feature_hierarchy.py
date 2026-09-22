#!/usr/bin/env python
"""Feature Specialization Audit for PhaseSAE.

Trains PhaseSAE on TinyStories-1M via LanguageModelSAETrainingRunner, then computes:
  (A) Per-feature firing frequency → Gini coefficient per phase
  (B) Cross-phase cosine similarity matrix between decoder columns
  (C) Top-10 max-activating token contexts for Phase 0 vs Phase (P-1)

Usage:
    .venv/bin/python scripts/analyze_feature_hierarchy.py
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
N_TOKENS_AUDIT = 10_000
D_IN = 64
D_SAE = 256
NUM_PHASES = 4
K_PER_PHASE = 4
TRAINING_TOKENS = 16000
TOP_N_TOKENS = 10

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = ARTIFACT_DIR / "feature_specialization_report.json"


def get_activations_with_tokens(model, n_tokens):
    """Returns (acts, token_strings) for audit use."""
    from datasets import load_dataset

    dataset = load_dataset(NEEL_NANDA_C4_10K_DATASET, split="train", streaming=True)
    all_acts, all_tok_strs = [], []
    for ex in dataset:
        text = ex["text"]
        tokens = model.to_tokens(text[:256])
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=HOOK_POINT)
        acts = cache[HOOK_POINT].squeeze(0)
        tok_strs = [model.to_string(t.unsqueeze(0)) for t in tokens.squeeze(0)]
        all_acts.append(acts)
        all_tok_strs.extend(tok_strs)
        if sum(a.shape[0] for a in all_acts) >= n_tokens:
            break
    acts_cat = torch.cat(all_acts, dim=0)[:n_tokens].cpu()
    tok_strs = all_tok_strs[:n_tokens]
    return acts_cat, tok_strs


def gini_coefficient(freqs: torch.Tensor) -> float:
    """Gini coefficient of a non-negative frequency vector."""
    f = freqs.sort().values.float()
    n = f.shape[0]
    if n == 0 or f.sum() == 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    return float(((2 * idx - n - 1) * f).sum() / (n * f.sum()))


def cross_phase_cosine_sim(W_dec: torch.Tensor, num_phases: int, m: int) -> list:
    """Compute P×P cross-phase cosine similarity matrices between decoder tiles."""
    tiles = [W_dec[p * m : (p + 1) * m, :] for p in range(num_phases)]
    result = []
    for i in range(num_phases):
        row = []
        ni = tiles[i] / (tiles[i].norm(dim=-1, keepdim=True) + 1e-8)
        for j in range(num_phases):
            nj = tiles[j] / (tiles[j].norm(dim=-1, keepdim=True) + 1e-8)
            sim_matrix = (ni @ nj.T).abs()
            if i == j:
                mask = ~torch.eye(m, dtype=torch.bool)
                mean_sim = sim_matrix[mask].mean().item()
            else:
                mean_sim = sim_matrix.mean().item()
            row.append(float(mean_sim))
        result.append(row)
    return result


def top_activating_tokens(
    feature_acts_all: torch.Tensor,
    tok_strs: list,
    phase_idx: int,
    m: int,
    top_n: int = 10,
):
    start = phase_idx * m
    end = (phase_idx + 1) * m
    phase_acts = feature_acts_all[:, start:end]
    results = {}
    for feat_local in range(m):
        acts_feat = phase_acts[:, feat_local]
        topk_vals, topk_idx = acts_feat.topk(min(top_n, len(tok_strs)))
        results[feat_local] = {
            "top_tokens": [tok_strs[i] for i in topk_idx.tolist()],
            "top_activations": [float(v) for v in topk_vals.tolist()],
            "firing_rate": float((acts_feat > 0).float().mean().item()),
        }
    return results


def main():
    print("=" * 75)
    print("  PHASESAE FEATURE SPECIALIZATION & HIERARCHY AUDIT")
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
        output_path="artifacts/feature_audit/phase_sae",
        n_checkpoints=0,
        exclude_special_tokens=True,
        save_final_checkpoint=False,
    )
    t0 = time.time()
    runner = LanguageModelSAETrainingRunner(runner_cfg, override_model=model)
    sae = runner.run()
    print(f"   Training completed in {time.time() - t0:.1f}s")

    print(f"\n3. Collecting {N_TOKENS_AUDIT} audit activations with token alignments …")
    audit_acts, tok_strs = get_activations_with_tokens(model, N_TOKENS_AUDIT)
    print(f"   Audit activations shape: {audit_acts.shape}")

    print("\n4. Running activations through PhaseSAE …")
    m = D_SAE // NUM_PHASES
    with torch.no_grad():
        feature_acts_all = sae.encode(audit_acts)

    # ── A: Gini per phase ──
    print("\n── [Metric A: Firing Frequency & Gini Coefficient per Phase] ──")
    gini_per_phase = {}
    for p in range(NUM_PHASES):
        phase_acts = feature_acts_all[:, p * m : (p + 1) * m]
        firing_counts = (phase_acts > 0).float().sum(0)
        g = gini_coefficient(firing_counts)
        mean_fr = float((phase_acts > 0).float().mean().item())
        gini_per_phase[p] = {"gini": g, "mean_firing_rate": mean_fr}
        print(f"   Phase {p}: Gini = {g:.4f} | Mean Firing Rate = {mean_fr:.4f}")

    # ── B: Cross-phase cosine similarity ──
    print("\n── [Metric B: Cross-Phase Cosine Orthogonality Matrix (P×P)] ──")
    W_dec = sae.W_dec.data.detach().cpu()
    cross_phase_sim = cross_phase_cosine_sim(W_dec, NUM_PHASES, m)
    for i, row in enumerate(cross_phase_sim):
        print(f"   Phase {i} vs [0..3]: " + "  ".join(f"{v:.4f}" for v in row))

    # ── C: Top activating tokens Phase 0 vs Phase P-1 ──
    print(
        f"\n── [Metric C: Feature Specialization (Phase 0 vs Phase {NUM_PHASES-1})] ──"
    )
    top_phase0 = top_activating_tokens(feature_acts_all, tok_strs, 0, m, TOP_N_TOKENS)
    top_phaseLast = top_activating_tokens(
        feature_acts_all, tok_strs, NUM_PHASES - 1, m, TOP_N_TOKENS
    )

    print(f"   Phase 0 Feat 0: tokens = {top_phase0[0]['top_tokens'][:6]}")
    print(
        f"   Phase {NUM_PHASES-1} Feat 0: tokens = {top_phaseLast[0]['top_tokens'][:6]}"
    )

    phase0_mean_fr = float(
        torch.tensor([top_phase0[f]["firing_rate"] for f in top_phase0]).mean()
    )
    phaseLast_mean_fr = float(
        torch.tensor([top_phaseLast[f]["firing_rate"] for f in top_phaseLast]).mean()
    )

    report = {
        "model": MODEL_NAME,
        "hook_point": HOOK_POINT,
        "num_phases": NUM_PHASES,
        "k_per_phase": K_PER_PHASE,
        "d_sae": D_SAE,
        "n_audit_tokens": N_TOKENS_AUDIT,
        "A_gini_per_phase": {str(k): v for k, v in gini_per_phase.items()},
        "B_cross_phase_cosine_sim": cross_phase_sim,
        "C_top_tokens_phase0": {str(k): v for k, v in list(top_phase0.items())[:5]},
        "C_top_tokens_phase_last": {
            str(k): v for k, v in list(top_phaseLast.items())[:5]
        },
        "summary": {
            "phase0_mean_firing_rate": phase0_mean_fr,
            f"phase{NUM_PHASES-1}_mean_firing_rate": phaseLast_mean_fr,
            "mean_gini": float(
                sum(v["gini"] for v in gini_per_phase.values()) / NUM_PHASES
            ),
            "cross_phase_off_diag_mean_cos_sim": float(
                sum(
                    cross_phase_sim[i][j]
                    for i in range(NUM_PHASES)
                    for j in range(NUM_PHASES)
                    if i != j
                )
                / (NUM_PHASES * (NUM_PHASES - 1))
            ),
            "within_phase_mean_cos_sim": float(
                sum(cross_phase_sim[i][i] for i in range(NUM_PHASES)) / NUM_PHASES
            ),
        },
    }

    with open(OUT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved feature specialization audit report → {OUT_FILE}")


if __name__ == "__main__":
    main()
