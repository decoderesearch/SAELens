import inspect
import os
import sys
import time

import torch
import yaml

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import CacheActivationsRunnerConfig
from sae_lens.constants import DTYPE_MAP

if len(sys.argv) > 1:
    job_name = sys.argv[1]
    print(f"Cache Activations Job Name: {job_name}")
else:
    raise ValueError("Error: One argument required - the Cache Activations Job Name")

if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    device = "mps"
    torch.mps.empty_cache()
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load the yaml file as config
# load only the keys that are in CacheActivationsRunnerConfig
# TODO: this is a hacky way of importing
with open(f"./jobs/cache_acts/{job_name}/cache_acts.yml") as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)

    config_params = inspect.signature(CacheActivationsRunnerConfig).parameters
    filtered_data = {k: v for k, v in config_yaml.items() if k in config_params}
    config = CacheActivationsRunnerConfig(**filtered_data)

    if type(config.dtype) != torch.dtype:
        config.dtype = DTYPE_MAP[config.dtype]  # type: ignore
    config.device = device

if config is None:
    raise ValueError("Error: The config is not loaded.")

print(f"Total Training Tokens: {config.training_tokens}")

# This is set by Ansible
new_cached_activations_path = config.new_cached_activations_path
if new_cached_activations_path is None:
    raise ValueError("Error: The new_cached_activations_path is not set.")

# Check directory to make sure we aren't overwriting unintentionally
if os.path.exists(new_cached_activations_path):
    raise ValueError(
        f"Error: The new_cached_activations_path ({new_cached_activations_path}) is not empty."
    )

start_time = time.time()

runner = CacheActivationsRunner(config)

print("-" * 50)
print(runner.__str__())
print("-" * 50)
runner.run()


end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print(
    f"{config.training_tokens / ((end_time - start_time)*10**6):.2f} Million Tokens / Second"
)
