import argparse
import os
from typing import Any, Callable

import torch
import torch._inductor.config
import triton  # type: ignore
from tabulate import tabulate

from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAE
from tests.helpers import (
    build_topk_sae_training_cfg,
)

torch._inductor.config.coordinate_descent_tuning = True

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--shape",
    type=int,
    nargs=3,
    default=[1024, 1024, 1024 * 16],
    help="Shape of the input tensor (seq_len, d_in, d_sae)",
)
parser.add_argument("--k", type=int, default=100, help="Number of topk elements")
args = parser.parse_args()

device = args.device

os.environ["TOKENIZERS_PARALLELISM"] = "false"

d_in = args.shape[1]
d_sae = args.shape[2]
k = args.k
seq_len = args.shape[0]

cfg_sparse = build_topk_sae_training_cfg(
    d_in=d_in,
    d_sae=d_sae,
    k=k,
    device=device,
    use_sparse_activations=True,
)
cfg_dense = build_topk_sae_training_cfg(
    d_in=d_in,
    d_sae=d_sae,
    k=k,
    device=device,
    use_sparse_activations=False,
)

sae_sparse = TopKTrainingSAE(cfg_sparse)
sae_dense = TopKTrainingSAE(cfg_dense)

dead_neuron_mask = None  # torch.randn(d_sae, device = device) > 0.1
input_acts = torch.randn(seq_len, d_in, device=device)
input_var = (input_acts - input_acts.mean(0)).pow(2).sum()

step_input = TrainStepInput(
    sae_in=input_acts,
    dead_neuron_mask=dead_neuron_mask,
    coefficients={},
    n_training_steps=0,
)


def encode_proj(sae: TopKTrainingSAE, input_acts: torch.Tensor) -> torch.Tensor:
    sae_in = sae.process_sae_in(input_acts)
    return sae.hook_sae_acts_pre(sae_in @ sae.W_enc + sae.b_enc)


def topk_activation(sae: TopKTrainingSAE, hidden_pre: torch.Tensor) -> torch.Tensor:
    return sae.activation_fn(hidden_pre)


def decode_step(sae: TopKTrainingSAE, feature_acts: torch.Tensor) -> torch.Tensor:
    return sae.decode(feature_acts)


def loss_computation(
    sae: TopKTrainingSAE, sae_out: torch.Tensor, sae_in: torch.Tensor
) -> torch.Tensor:
    # Calculate MSE loss
    per_item_mse_loss = sae.mse_loss_fn(sae_out, sae_in)
    return per_item_mse_loss.sum(dim=-1).mean()


def triton_bench(fn: Callable[[], Any]) -> float:
    # note that the warmup and rep params here are in ms, not iterations
    return triton.testing.do_bench(fn, warmup=1000, rep=2000)  # type: ignore


def benchmark_sae(sae: TopKTrainingSAE) -> dict[str, float]:
    results = {}
    results["encode_proj"] = triton_bench(lambda: encode_proj(sae, input_acts))
    hidden_pre = encode_proj(sae, input_acts)
    results["topk_activation"] = triton_bench(lambda: topk_activation(sae, hidden_pre))
    feature_acts = topk_activation(sae, hidden_pre)
    results["decode_step"] = triton_bench(lambda: decode_step(sae, feature_acts))
    sae_out = decode_step(sae, feature_acts)
    results["loss_computation"] = triton_bench(
        lambda: loss_computation(sae, sae_out, input_acts)
    )
    results["full_forward_pass"] = triton_bench(
        lambda: sae.training_forward_pass(step_input)
    )
    results["other"] = 2 * results["full_forward_pass"] - sum(results.values())  # type: ignore
    return results


if __name__ == "__main__":
    print("This may take a while (5 mins). Go grab a coffee!")
    results_sparse = benchmark_sae(sae_sparse)
    results_dense = benchmark_sae(sae_dense)

    # Pretty print results table with metrics as columns
    headers = [
        "Implementation",
        "Encode",
        "TopK",
        "Decode",
        "Loss Calc",
        "Full Fwd",
        "Other",
    ]

    metric_keys = results_sparse.keys()

    table_data = [
        ["Sparse"] + [f"{results_sparse[key]:.3f}" for key in metric_keys],
        ["Dense"] + [f"{results_dense[key]:.3f}" for key in metric_keys],
    ]
    print("Metric: Latency (ms)")
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
