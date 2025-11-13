"""Base classes for Sparse Autoencoders (SAEs)."""

import copy
import json
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
)

import einops
import torch
from numpy.typing import NDArray
from safetensors.torch import load_file, save_file
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint
from typing_extensions import deprecated, overload, override

from sae_lens import __version__
from sae_lens.constants import (
    DTYPE_MAP,
    SAE_CFG_FILENAME,
    SAE_WEIGHTS_FILENAME,
)
from sae_lens.util import filter_valid_dataclass_fields

if TYPE_CHECKING:
    from sae_lens.config import LanguageModelSAERunnerConfig

from sae_lens.loading.pretrained_sae_loaders import (
    NAMED_PRETRAINED_SAE_LOADERS,
    PretrainedSaeDiskLoader,
    PretrainedSaeHuggingfaceLoader,
    get_conversion_loader_name,
    handle_config_defaulting,
    sae_lens_disk_loader,
)
from sae_lens.loading.pretrained_saes_directory import (
    get_config_overrides,
    get_norm_scaling_factor,
    get_pretrained_saes_directory,
    get_repo_id_and_folder_name,
)
from sae_lens.registry import get_sae_class, get_sae_training_class

T_SAE_CONFIG = TypeVar("T_SAE_CONFIG", bound="SAEConfig")
T_TRAINING_SAE_CONFIG = TypeVar("T_TRAINING_SAE_CONFIG", bound="TrainingSAEConfig")
T_SAE = TypeVar("T_SAE", bound="SAE")  # type: ignore
T_TRAINING_SAE = TypeVar("T_TRAINING_SAE", bound="TrainingSAE")  # type: ignore


class SAEMetadata:
    """Core metadata about how this SAE should be used, if known."""

    def __init__(self, **kwargs: Any):
        # Set default version fields with their current behavior
        self.sae_lens_version = kwargs.pop("sae_lens_version", __version__)
        self.sae_lens_training_version = kwargs.pop(
            "sae_lens_training_version", __version__
        )

        # Set all other attributes dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> None:
        """Return None for any missing attribute (like defaultdict)"""
        return

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting any attribute"""
        super().__setattr__(name, value)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access: metadata['key'] - returns None for missing keys"""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment: metadata['key'] = value"""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator: 'key' in metadata"""
        # Only return True if the attribute was explicitly set (not just defaulting to None)
        return key in self.__dict__

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style get with default"""
        value = getattr(self, key)
        # If the attribute wasn't explicitly set and we got None from __getattr__,
        # use the provided default instead
        if key not in self.__dict__ and value is None:
            return default
        return value

    def keys(self):
        """Return all explicitly set attribute names"""
        return self.__dict__.keys()

    def values(self):
        """Return all explicitly set attribute values"""
        return self.__dict__.values()

    def items(self):
        """Return all explicitly set attribute name-value pairs"""
        return self.__dict__.items()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SAEMetadata":
        """Create from dictionary"""
        return cls(**data)

    def __repr__(self) -> str:
        return f"SAEMetadata({self.__dict__})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SAEMetadata):
            return False
        return self.__dict__ == other.__dict__

    def __deepcopy__(self, memo: dict[int, Any]) -> "SAEMetadata":
        """Support for deep copying"""

        return SAEMetadata(**copy.deepcopy(self.__dict__, memo))

    def __getstate__(self) -> dict[str, Any]:
        """Support for pickling"""
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Support for unpickling"""
        self.__dict__.update(state)


@dataclass
class SAEConfig(ABC):
    """Base configuration for SAE models."""

    d_in: int
    d_sae: int
    dtype: str = "float32"
    device: str = "cpu"
    apply_b_dec_to_input: bool = True
    normalize_activations: Literal["none", "expected_average_only_in", "layer_norm"] = (
        "none"  # none, expected_average_only_in (Anthropic April Update)
    )
    reshape_activations: Literal["none", "hook_z"] = "none"
    metadata: SAEMetadata = field(default_factory=SAEMetadata)

    @classmethod
    @abstractmethod
    def architecture(cls) -> str: ...

    def to_dict(self) -> dict[str, Any]:
        res = {field.name: getattr(self, field.name) for field in fields(self)}
        res["metadata"] = self.metadata.to_dict()
        res["architecture"] = self.architecture()
        return res

    @classmethod
    def from_dict(cls: type[T_SAE_CONFIG], config_dict: dict[str, Any]) -> T_SAE_CONFIG:
        cfg_class = get_sae_class(config_dict["architecture"])[1]
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cfg_class)
        res = cfg_class(**filtered_config_dict)
        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])
        if not isinstance(res, cls):
            raise ValueError(
                f"SAE config class {cls} does not match dict config class {type(res)}"
            )
        return res

    def __post_init__(self):
        if self.normalize_activations not in [
            "none",
            "expected_average_only_in",
            "constant_norm_rescale",
            "layer_norm",
        ]:
            raise ValueError(
                f"normalize_activations must be none, expected_average_only_in, layer_norm, or constant_norm_rescale. Got {self.normalize_activations}"
            )


@dataclass
class TrainStepOutput:
    """Output from a training step."""

    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    hidden_pre: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    losses: dict[str, torch.Tensor]
    # any extra metrics to log can be added here
    metrics: dict[str, torch.Tensor | float | int] = field(default_factory=dict)


@dataclass
class TrainStepInput:
    """Input to a training step."""

    sae_in: torch.Tensor
    coefficients: dict[str, float]
    dead_neuron_mask: torch.Tensor | None
    n_training_steps: int


class TrainCoefficientConfig(NamedTuple):
    value: float
    warm_up_steps: int


class SAE(HookedRootModule, Generic[T_SAE_CONFIG], ABC):
    """Abstract base class for all SAE architectures."""

    cfg: T_SAE_CONFIG
    dtype: torch.dtype
    device: torch.device
    use_error_term: bool

    # For type checking only - don't provide default values
    # These will be initialized by subclasses
    W_enc: nn.Parameter
    W_dec: nn.Parameter
    b_dec: nn.Parameter

    def __init__(self, cfg: T_SAE_CONFIG, use_error_term: bool = False):
        """Initialize the SAE."""
        super().__init__()

        self.cfg = cfg

        if cfg.metadata and cfg.metadata.model_from_pretrained_kwargs:
            warnings.warn(
                "\nThis SAE has non-empty model_from_pretrained_kwargs. "
                "\nFor optimal performance, load the model like so:\n"
                "model = HookedSAETransformer.from_pretrained_no_processing(..., **cfg.model_from_pretrained_kwargs)",
                category=UserWarning,
                stacklevel=1,
            )

        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)
        self.use_error_term = use_error_term

        # Set up activation function
        self.activation_fn = self.get_activation_fn()

        # Initialize weights
        self.initialize_weights()

        # Set up hooks
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_output = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()

        # handle hook_z reshaping if needed.
        if self.cfg.reshape_activations == "hook_z":
            self.turn_on_forward_pass_hook_z_reshaping()
        else:
            self.turn_off_forward_pass_hook_z_reshaping()

        # Set up activation normalization
        self._setup_activation_normalization()

        self.setup()  # Required for HookedRootModule

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float):
        self.W_enc.data *= scaling_factor  # type: ignore
        self.W_dec.data /= scaling_factor  # type: ignore
        self.b_dec.data /= scaling_factor  # type: ignore
        self.cfg.normalize_activations = "none"

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the activation function specified in config."""
        return nn.ReLU()

    def _setup_activation_normalization(self):
        """Set up activation normalization functions based on config."""
        if self.cfg.normalize_activations == "constant_norm_rescale":

            def run_time_activation_norm_fn_in(x: torch.Tensor) -> torch.Tensor:
                self.x_norm_coeff = (self.cfg.d_in**0.5) / x.norm(dim=-1, keepdim=True)
                return x * self.x_norm_coeff

            def run_time_activation_norm_fn_out(x: torch.Tensor) -> torch.Tensor:
                x = x / self.x_norm_coeff  # type: ignore
                del self.x_norm_coeff
                return x

            self.run_time_activation_norm_fn_in = run_time_activation_norm_fn_in
            self.run_time_activation_norm_fn_out = run_time_activation_norm_fn_out

        elif self.cfg.normalize_activations == "layer_norm":
            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_ln_in(
                x: torch.Tensor, eps: float = 1e-5
            ) -> torch.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                x = x - mu
                std = x.std(dim=-1, keepdim=True)
                x = x / (std + eps)
                self.ln_mu = mu
                self.ln_std = std
                return x

            def run_time_activation_ln_out(
                x: torch.Tensor,
                eps: float = 1e-5,  # noqa: ARG001
            ) -> torch.Tensor:
                return x * self.ln_std + self.ln_mu  # type: ignore

            self.run_time_activation_norm_fn_in = run_time_activation_ln_in
            self.run_time_activation_norm_fn_out = run_time_activation_ln_out
        else:
            self.run_time_activation_norm_fn_in = lambda x: x
            self.run_time_activation_norm_fn_out = lambda x: x

    def initialize_weights(self):
        """Initialize model weights."""
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        w_dec_data = torch.empty(
            self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_data)
        self.W_dec = nn.Parameter(w_dec_data)

        w_enc_data = self.W_dec.data.T.clone().detach().contiguous()
        self.W_enc = nn.Parameter(w_enc_data)

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to feature space."""
        pass

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode feature activations back to input space."""
        pass

    def turn_on_forward_pass_hook_z_reshaping(self):
        if (
            self.cfg.metadata.hook_name is not None
            and not self.cfg.metadata.hook_name.endswith("_z")
        ):
            raise ValueError("This method should only be called for hook_z SAEs.")

        # print(f"Turning on hook_z reshaping for {self.cfg.hook_name}")

        def reshape_fn_in(x: torch.Tensor):
            # print(f"reshape_fn_in input shape: {x.shape}")
            self.d_head = x.shape[-1]
            # print(f"Setting d_head to: {self.d_head}")
            self.reshape_fn_in = lambda x: einops.rearrange(
                x, "... n_heads d_head -> ... (n_heads d_head)"
            )
            return einops.rearrange(x, "... n_heads d_head -> ... (n_heads d_head)")

        self.reshape_fn_in = reshape_fn_in
        self.reshape_fn_out = lambda x, d_head: einops.rearrange(
            x, "... (n_heads d_head) -> ... n_heads d_head", d_head=d_head
        )
        self.hook_z_reshaping_mode = True
        # print(f"hook_z reshaping turned on, self.d_head={getattr(self, 'd_head', None)}")

    def turn_off_forward_pass_hook_z_reshaping(self):
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x  # noqa: ARG005
        self.d_head = None
        self.hook_z_reshaping_mode = False

    @overload
    def to(
        self: T_SAE,
        device: torch.device | str | None = ...,
        dtype: torch.dtype | None = ...,
        non_blocking: bool = ...,
    ) -> T_SAE: ...

    @overload
    def to(self: T_SAE, dtype: torch.dtype, non_blocking: bool = ...) -> T_SAE: ...

    @overload
    def to(self: T_SAE, tensor: torch.Tensor, non_blocking: bool = ...) -> T_SAE: ...

    def to(self: T_SAE, *args: Any, **kwargs: Any) -> T_SAE:  # type: ignore
        device_arg = None
        dtype_arg = None

        # Check args
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device_arg = arg
            elif isinstance(arg, torch.dtype):
                dtype_arg = arg
            elif isinstance(arg, torch.Tensor):
                device_arg = arg.device
                dtype_arg = arg.dtype

        # Check kwargs
        device_arg = kwargs.get("device", device_arg)
        dtype_arg = kwargs.get("dtype", dtype_arg)

        # Update device in config if provided
        if device_arg is not None:
            # Convert device to torch.device if it's a string
            device = (
                torch.device(device_arg) if isinstance(device_arg, str) else device_arg
            )

            # Update the cfg.device
            self.cfg.device = str(device)

            # Update the device property
            self.device = device

        # Update dtype in config if provided
        if dtype_arg is not None:
            # Update the cfg.dtype
            self.cfg.dtype = str(dtype_arg)

            # Update the dtype property
            self.dtype = dtype_arg

        return super().to(*args, **kwargs)

    def process_sae_in(self, sae_in: torch.Tensor) -> torch.Tensor:
        sae_in = sae_in.to(self.dtype)
        sae_in = self.reshape_fn_in(sae_in)

        sae_in = self.hook_sae_input(sae_in)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)

        # Here's where the error happens
        bias_term = self.b_dec * self.cfg.apply_b_dec_to_input

        return sae_in - bias_term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SAE."""
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        if self.use_error_term:
            with torch.no_grad():
                # Recompute without hooks for true error term
                with _disable_hooks(self):
                    feature_acts_clean = self.encode(x)
                    x_reconstruct_clean = self.decode(feature_acts_clean)
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            sae_out = sae_out + sae_error

        return self.hook_sae_output(sae_out)

    # overwrite this in subclasses to modify the state_dict in-place before saving
    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        pass

    # overwrite this in subclasses to modify the state_dict in-place after loading
    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        pass

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Fold decoder norms into encoder."""
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T

        # Only update b_enc if it exists (standard/jumprelu architectures)
        if hasattr(self, "b_enc") and isinstance(self.b_enc, nn.Parameter):
            self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()

    def get_name(self):
        """Generate a name for this SAE."""
        return f"sae_{self.cfg.metadata.model_name}_{self.cfg.metadata.hook_name}_{self.cfg.d_sae}"

    def save_model(self, path: str | Path) -> tuple[Path, Path]:
        """Save model weights and config to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Generate the weights
        state_dict = self.state_dict()  # Use internal SAE state dict
        self.process_state_dict_for_saving(state_dict)
        model_weights_path = path / SAE_WEIGHTS_FILENAME
        save_file(state_dict, model_weights_path)

        # Save the config
        config = self.cfg.to_dict()
        cfg_path = path / SAE_CFG_FILENAME
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        return model_weights_path, cfg_path

    # Class methods for loading models
    @classmethod
    @deprecated("Use load_from_disk instead")
    def load_from_pretrained(
        cls: type[T_SAE],
        path: str | Path,
        device: str = "cpu",
        dtype: str | None = None,
    ) -> T_SAE:
        return cls.load_from_disk(path, device=device, dtype=dtype)

    @classmethod
    def load_from_disk(
        cls: type[T_SAE],
        path: str | Path,
        device: str = "cpu",
        dtype: str | None = None,
        converter: PretrainedSaeDiskLoader = sae_lens_disk_loader,
    ) -> T_SAE:
        overrides = {"dtype": dtype} if dtype is not None else None
        cfg_dict, state_dict = converter(path, device, cfg_overrides=overrides)
        cfg_dict = handle_config_defaulting(cfg_dict)
        sae_config_cls = cls.get_sae_config_class_for_architecture(
            cfg_dict["architecture"]
        )
        sae_cfg = sae_config_cls.from_dict(cfg_dict)
        sae_cls = cls.get_sae_class_for_architecture(sae_cfg.architecture())
        sae = sae_cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)
        return sae

    @classmethod
    def from_pretrained(
        cls: type[T_SAE],
        release: str,
        sae_id: str,
        device: str = "cpu",
        force_download: bool = False,
        converter: PretrainedSaeHuggingfaceLoader | None = None,
    ) -> T_SAE:
        """
        Load a pretrained SAE from the Hugging Face model hub.

        Args:
            release: The release name. This will be mapped to a huggingface repo id based on the pretrained_saes.yaml file.
            id: The id of the SAE to load. This will be mapped to a path in the huggingface repo.
            device: The device to load the SAE on.
        """
        return cls.from_pretrained_with_cfg_and_sparsity(
            release, sae_id, device, force_download, converter=converter
        )[0]

    @classmethod
    def from_pretrained_with_cfg_and_sparsity(
        cls: type[T_SAE],
        release: str,
        sae_id: str,
        device: str = "cpu",
        force_download: bool = False,
        converter: PretrainedSaeHuggingfaceLoader | None = None,
    ) -> tuple[T_SAE, dict[str, Any], torch.Tensor | None]:
        """
        Load a pretrained SAE from the Hugging Face model hub, along with its config dict and sparsity, if present.
        In SAELens <= 5.x.x, this was called SAE.from_pretrained().

        Args:
            release: The release name. This will be mapped to a huggingface repo id based on the pretrained_saes.yaml file.
            id: The id of the SAE to load. This will be mapped to a path in the huggingface repo.
            device: The device to load the SAE on.
        """

        # get sae directory
        sae_directory = get_pretrained_saes_directory()

        # Validate release and sae_id
        if release not in sae_directory:
            if "/" not in release:
                raise ValueError(
                    f"Release {release} not found in pretrained SAEs directory, and is not a valid huggingface repo."
                )
        elif sae_id not in sae_directory[release].saes_map:
            # Handle special cases like Gemma Scope
            if (
                "gemma-scope" in release
                and "canonical" not in release
                and f"{release}-canonical" in sae_directory
            ):
                canonical_ids = list(
                    sae_directory[release + "-canonical"].saes_map.keys()
                )
                # Shorten the lengthy string of valid IDs
                if len(canonical_ids) > 5:
                    str_canonical_ids = str(canonical_ids[:5])[:-1] + ", ...]"
                else:
                    str_canonical_ids = str(canonical_ids)
                value_suffix = f" If you don't want to specify an L0 value, consider using release {release}-canonical which has valid IDs {str_canonical_ids}"
            else:
                value_suffix = ""

            valid_ids = list(sae_directory[release].saes_map.keys())
            # Shorten the lengthy string of valid IDs
            if len(valid_ids) > 5:
                str_valid_ids = str(valid_ids[:5])[:-1] + ", ...]"
            else:
                str_valid_ids = str(valid_ids)

            raise ValueError(
                f"ID {sae_id} not found in release {release}. Valid IDs are {str_valid_ids}."
                + value_suffix
            )

        conversion_loader = (
            converter
            or NAMED_PRETRAINED_SAE_LOADERS[get_conversion_loader_name(release)]
        )
        repo_id, folder_name = get_repo_id_and_folder_name(release, sae_id)
        config_overrides = get_config_overrides(release, sae_id)
        config_overrides["device"] = device

        # Load config and weights
        cfg_dict, state_dict, log_sparsities = conversion_loader(
            repo_id=repo_id,
            folder_name=folder_name,
            device=device,
            force_download=force_download,
            cfg_overrides=config_overrides,
        )
        cfg_dict = handle_config_defaulting(cfg_dict)

        # Create SAE with appropriate architecture
        sae_config_cls = cls.get_sae_config_class_for_architecture(
            cfg_dict["architecture"]
        )
        sae_cfg = sae_config_cls.from_dict(cfg_dict)
        sae_cls = cls.get_sae_class_for_architecture(sae_cfg.architecture())
        sae = sae_cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)

        # Apply normalization if needed
        if cfg_dict.get("normalize_activations") == "expected_average_only_in":
            norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
            if norm_scaling_factor is not None:
                sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
                cfg_dict["normalize_activations"] = "none"
            else:
                warnings.warn(
                    f"norm_scaling_factor not found for {release} and {sae_id}, but normalize_activations is 'expected_average_only_in'. Skipping normalization folding."
                )

        return sae, cfg_dict, log_sparsities

    @classmethod
    def from_dict(cls: type[T_SAE], config_dict: dict[str, Any]) -> T_SAE:
        """Create an SAE from a config dictionary."""
        sae_cls = cls.get_sae_class_for_architecture(config_dict["architecture"])
        sae_config_cls = cls.get_sae_config_class_for_architecture(
            config_dict["architecture"]
        )
        return sae_cls(sae_config_cls.from_dict(config_dict))

    @classmethod
    def get_sae_class_for_architecture(
        cls: type[T_SAE], architecture: str
    ) -> type[T_SAE]:
        """Get the SAE class for a given architecture."""
        sae_cls, _ = get_sae_class(architecture)
        if not issubclass(sae_cls, cls):
            raise ValueError(
                f"Loaded SAE is not of type {cls.__name__}. Use {sae_cls.__name__} instead"
            )
        return sae_cls

    # in the future, this can be used to load different config classes for different architectures
    @classmethod
    def get_sae_config_class_for_architecture(
        cls,
        architecture: str,  # noqa: ARG003
    ) -> type[SAEConfig]:
        return SAEConfig

    ### Methods to support deprecated usage of SAE.from_pretrained() ###

    def __getitem__(self, index: int) -> Any:
        """
        Support indexing for backward compatibility with tuple unpacking.
        DEPRECATED: SAE.from_pretrained() no longer returns a tuple.
        Use SAE.from_pretrained_with_cfg_and_sparsity() instead.
        """
        warnings.warn(
            "Indexing SAE objects is deprecated. SAE.from_pretrained() now returns "
            "only the SAE object. Use SAE.from_pretrained_with_cfg_and_sparsity() "
            "to get the config dict and sparsity as well.",
            DeprecationWarning,
            stacklevel=2,
        )

        if index == 0:
            return self
        if index == 1:
            return self.cfg.to_dict()
        if index == 2:
            return None
        raise IndexError(f"SAE tuple index {index} out of range")

    def __iter__(self):
        """
        Support unpacking for backward compatibility with tuple unpacking.
        DEPRECATED: SAE.from_pretrained() no longer returns a tuple.
        Use SAE.from_pretrained_with_cfg_and_sparsity() instead.
        """
        warnings.warn(
            "Unpacking SAE objects is deprecated. SAE.from_pretrained() now returns "
            "only the SAE object. Use SAE.from_pretrained_with_cfg_and_sparsity() "
            "to get the config dict and sparsity as well.",
            DeprecationWarning,
            stacklevel=2,
        )

        yield self
        yield self.cfg.to_dict()
        yield None

    def __len__(self) -> int:
        """
        Support len() for backward compatibility with tuple unpacking.
        DEPRECATED: SAE.from_pretrained() no longer returns a tuple.
        Use SAE.from_pretrained_with_cfg_and_sparsity() instead.
        """
        warnings.warn(
            "Getting length of SAE objects is deprecated. SAE.from_pretrained() now returns "
            "only the SAE object. Use SAE.from_pretrained_with_cfg_and_sparsity() "
            "to get the config dict and sparsity as well.",
            DeprecationWarning,
            stacklevel=2,
        )

        return 3


@dataclass(kw_only=True)
class TrainingSAEConfig(SAEConfig, ABC):
    # https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    # 0.1 corresponds to the "heuristic" initialization, use None to disable
    decoder_init_norm: float | None = 0.1

    @classmethod
    @abstractmethod
    def architecture(cls) -> str: ...

    @classmethod
    def from_sae_runner_config(
        cls: type[T_TRAINING_SAE_CONFIG],
        cfg: "LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]",
    ) -> T_TRAINING_SAE_CONFIG:
        metadata = SAEMetadata(
            model_name=cfg.model_name,
            hook_name=cfg.hook_name,
            hook_head_index=cfg.hook_head_index,
            context_size=cfg.context_size,
            prepend_bos=cfg.prepend_bos,
            seqpos_slice=cfg.seqpos_slice,
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs or {},
        )
        if not isinstance(cfg.sae, cls):
            raise ValueError(
                f"SAE config class {cls} does not match SAE runner config class {type(cfg.sae)}"
            )
        return replace(cfg.sae, metadata=metadata)

    @classmethod
    def from_dict(
        cls: type[T_TRAINING_SAE_CONFIG], config_dict: dict[str, Any]
    ) -> T_TRAINING_SAE_CONFIG:
        cfg_class = cls
        if "architecture" in config_dict:
            cfg_class = get_sae_training_class(config_dict["architecture"])[1]
        if not issubclass(cfg_class, cls):
            raise ValueError(
                f"SAE config class {cls} does not match dict config class {type(cfg_class)}"
            )
        # remove any keys that are not in the dataclass
        # since we sometimes enhance the config with the whole LM runner config
        valid_config_dict = filter_valid_dataclass_fields(config_dict, cfg_class)
        if "metadata" in config_dict:
            valid_config_dict["metadata"] = SAEMetadata(**config_dict["metadata"])
        return cfg_class(**valid_config_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            **asdict(self),
            "metadata": self.metadata.to_dict(),
            "architecture": self.architecture(),
        }

    def get_inference_config_class(self) -> type[SAEConfig]:
        """
        Get the architecture for inference.
        """
        return get_sae_class(self.architecture())[1]

    # this needs to exist so we can initialize the parent sae cfg without the training specific
    # parameters. Maybe there's a cleaner way to do this
    def get_inference_sae_cfg_dict(self) -> dict[str, Any]:
        """
        Creates a dictionary containing attributes corresponding to the fields
        defined in the base SAEConfig class.
        """
        base_sae_cfg_class = self.get_inference_config_class()
        base_config_field_names = {f.name for f in fields(base_sae_cfg_class)}
        result_dict = {
            field_name: getattr(self, field_name)
            for field_name in base_config_field_names
        }
        result_dict["architecture"] = base_sae_cfg_class.architecture()
        result_dict["metadata"] = self.metadata.to_dict()
        return result_dict


class TrainingSAE(SAE[T_TRAINING_SAE_CONFIG], ABC):
    """Abstract base class for training versions of SAEs."""

    def __init__(self, cfg: T_TRAINING_SAE_CONFIG, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

        # Turn off hook_z reshaping for training mode - the activation store
        # is expected to handle reshaping before passing data to the SAE
        self.turn_off_forward_pass_hook_z_reshaping()
        self.mse_loss_fn = mse_loss

    @abstractmethod
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]: ...

    @abstractmethod
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode with access to pre-activation values for training."""
        ...

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        For inference, just encode without returning hidden_pre.
        (training_forward_pass calls encode_with_hidden_pre).
        """
        feature_acts, _ = self.encode_with_hidden_pre(x)
        return feature_acts

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decodes feature activations back into input space,
        applying optional finetuning scale, hooking, out normalization, etc.
        """
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def initialize_weights(self):
        super().initialize_weights()
        if self.cfg.decoder_init_norm is not None:
            with torch.no_grad():
                self.W_dec.data /= self.W_dec.norm(dim=-1, keepdim=True)
                self.W_dec.data *= self.cfg.decoder_init_norm
            self.W_enc.data = self.W_dec.data.T.clone().detach().contiguous()

    @abstractmethod
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Calculate architecture-specific auxiliary loss terms."""
        ...

    def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> TrainStepOutput:
        """Forward pass during training."""
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        # Calculate MSE loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # Calculate architecture-specific auxiliary losses
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        # Total loss is MSE plus all auxiliary losses
        total_loss = mse_loss

        # Create losses dictionary with mse_loss
        losses = {"mse_loss": mse_loss}

        # Add architecture-specific losses to the dictionary
        # Make sure aux_losses is a dictionary with string keys and tensor values
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)

        # Sum all losses for total_loss
        if isinstance(aux_losses, dict):
            for loss_value in aux_losses.values():
                total_loss = total_loss + loss_value
        else:
            # Handle case where aux_losses is a tensor
            total_loss = total_loss + aux_losses

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )

    def save_inference_model(self, path: str | Path) -> tuple[Path, Path]:
        """Save inference version of model weights and config to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Generate the weights
        state_dict = self.state_dict()  # Use internal SAE state dict
        self.process_state_dict_for_saving_inference(state_dict)
        model_weights_path = path / SAE_WEIGHTS_FILENAME
        save_file(state_dict, model_weights_path)

        # Save the config
        config = self.cfg.get_inference_sae_cfg_dict()
        cfg_path = path / SAE_CFG_FILENAME
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        return model_weights_path, cfg_path

    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        """
        Process the state dict for saving the inference model.
        This is a hook that can be overridden to change how the state dict is processed for the inference model.
        """
        return self.process_state_dict_for_saving(state_dict)

    @torch.no_grad()
    def log_histograms(self) -> dict[str, NDArray[Any]]:
        """Log histograms of the weights and biases."""
        W_dec_norm_dist = self.W_dec.detach().float().norm(dim=1).cpu().numpy()
        return {
            "weights/W_dec_norms": W_dec_norm_dist,
        }

    @classmethod
    def get_sae_class_for_architecture(
        cls: type[T_TRAINING_SAE], architecture: str
    ) -> type[T_TRAINING_SAE]:
        """Get the SAE class for a given architecture."""
        sae_cls, _ = get_sae_training_class(architecture)
        if not issubclass(sae_cls, cls):
            raise ValueError(
                f"Loaded SAE is not of type {cls.__name__}. Use {sae_cls.__name__} instead"
            )
        return sae_cls

    # in the future, this can be used to load different config classes for different architectures
    @classmethod
    def get_sae_config_class_for_architecture(
        cls,
        architecture: str,  # noqa: ARG003
    ) -> type[TrainingSAEConfig]:
        return get_sae_training_class(architecture)[1]

    def load_weights_from_checkpoint(self, checkpoint_path: Path | str) -> None:
        checkpoint_path = Path(checkpoint_path)
        state_dict = load_file(checkpoint_path / SAE_WEIGHTS_FILENAME)
        self.process_state_dict_for_loading(state_dict)
        self.load_state_dict(state_dict)


_blank_hook = nn.Identity()


@contextmanager
def _disable_hooks(sae: SAE[Any]):
    """
    Temporarily disable hooks for the SAE. Swaps out all the hooks with a fake modules that does nothing.
    """
    try:
        for hook_name in sae.hook_dict:
            setattr(sae, hook_name, _blank_hook)
        yield
    finally:
        for hook_name, hook in sae.hook_dict.items():
            setattr(sae, hook_name, hook)


def mse_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(preds, target, reduction="none")
