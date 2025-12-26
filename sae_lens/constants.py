import torch

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}

# Reverse mapping from torch.dtype to canonical string format
DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


SPARSITY_FILENAME = "sparsity.safetensors"
SAE_WEIGHTS_FILENAME = "sae_weights.safetensors"
SAE_CFG_FILENAME = "cfg.json"
RUNNER_CFG_FILENAME = "runner_cfg.json"
SPARSIFY_WEIGHTS_FILENAME = "sae.safetensors"
TRAINER_STATE_FILENAME = "trainer_state.pt"
ACTIVATIONS_STORE_STATE_FILENAME = "activations_store_state.safetensors"
ACTIVATION_SCALER_CFG_FILENAME = "activation_scaler.json"
