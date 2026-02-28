import io
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import einops
import torch
from datasets import Array2D, Dataset, Features, Sequence, Value
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformer_lens.HookedTransformer import HookedRootModule

from sae_lens import logger
from sae_lens.config import CacheActivationsRunnerConfig
from sae_lens.load_model import load_model
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.util import str_to_dtype

# Directory names for temporary operations during caching
_TMP_SHARDS_DIR = ".tmp_shards"
_SHUFFLED_DIR = ".shuffled"
_BACKUP_SUFFIX = ".backup"


def _mk_activations_store(
    model: HookedRootModule,
    cfg: CacheActivationsRunnerConfig,
    override_dataset: Dataset | None = None,
) -> ActivationsStore:
    """
    Internal method used in CacheActivationsRunner. Used to create a cached dataset
    from a ActivationsStore.
    """
    device = torch.device("cpu")  # since we're saving to disk
    return ActivationsStore(
        model=model,
        dataset=override_dataset or cfg.dataset_path,
        streaming=cfg.streaming,
        hook_name=cfg.hook_name,
        hook_head_index=None,
        context_size=cfg.context_size,
        d_in=cfg.d_in,
        n_batches_in_buffer=cfg.n_batches_in_buffer,
        total_training_tokens=cfg.training_tokens,
        store_batch_size_prompts=cfg.model_batch_size,
        train_batch_size_tokens=-1,
        prepend_bos=cfg.prepend_bos,
        normalize_activations="none",
        device=device,
        dtype=cfg.dtype,
        cached_activations_path=None,
        model_kwargs=cfg.model_kwargs,
        autocast_lm=cfg.autocast_lm,
        dataset_trust_remote_code=cfg.dataset_trust_remote_code,
        seqpos_slice=cfg.seqpos_slice,
        disable_concat_sequences=cfg.disable_concat_sequences,
        sequence_separator_token=cfg.sequence_separator_token,
    )


class CacheActivationsRunner:
    def __init__(
        self,
        cfg: CacheActivationsRunnerConfig,
        override_dataset: Dataset | None = None,
    ):
        self.cfg = cfg
        self.model: HookedRootModule = load_model(
            model_class_name=self.cfg.model_class_name,
            model_name=self.cfg.model_name,
            device=self.cfg.device,
            model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
        )
        if self.cfg.compile_llm:
            self.model = torch.compile(self.model, mode=self.cfg.llm_compilation_mode)  # type: ignore
        self.activations_store = _mk_activations_store(
            self.model,
            self.cfg,
            override_dataset=override_dataset,
        )
        self.context_size = self._get_sliced_context_size(
            self.cfg.context_size, self.cfg.seqpos_slice
        )
        features_dict: dict[str, Array2D | Sequence] = {
            hook_name: Array2D(
                shape=(self.context_size, self.cfg.d_in), dtype=self.cfg.dtype
            )
            for hook_name in [self.cfg.hook_name]
        }
        features_dict["token_ids"] = Sequence(  # type: ignore
            Value(dtype="int32"), length=self.context_size
        )
        self.features = Features(features_dict)

    def __str__(self):
        """
        Print the number of tokens to be cached.
        Print the number of buffers, and the number of tokens per buffer.
        Print the disk space required to store the activations.

        """

        bytes_per_token = (
            self.cfg.d_in * self.cfg.dtype.itemsize
            if isinstance(self.cfg.dtype, torch.dtype)
            else str_to_dtype(self.cfg.dtype).itemsize
        )
        total_training_tokens = self.cfg.n_seq_in_dataset * self.context_size
        total_disk_space_gb = total_training_tokens * bytes_per_token / 10**9

        return (
            f"Activation Cache Runner:\n"
            f"Total training tokens: {total_training_tokens}\n"
            f"Number of buffers: {self.cfg.n_buffers}\n"
            f"Tokens per buffer: {self.cfg.n_tokens_in_buffer}\n"
            f"Disk space required: {total_disk_space_gb:.2f} GB\n"
            f"Configuration:\n"
            f"{self.cfg}"
        )

    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without rewriting data.

        Each of the shards must be of the same format, aka the full dataset must be able to
        be recreated like so:

        ```
        ds = concatenate_datasets(
            [Dataset.load_from_disk(str(shard_dir)) for shard_dir in sorted(source_dir.iterdir())]
        )

        ```

        Sharded dataset format:
        ```
        source_dir/
            shard_00000/
                dataset_info.json
                state.json
                data-00000-of-00002.arrow
                data-00001-of-00002.arrow
            shard_00001/
                dataset_info.json
                state.json
                data-00000-of-00001.arrow
        ```

        And flattens them into the format:

        ```
        output_dir/
            dataset_info.json
            state.json
            data-00000-of-00003.arrow
            data-00001-of-00003.arrow
            data-00002-of-00003.arrow
        ```

        allowing the dataset to be loaded like so:

        ```
        ds = datasets.load_from_disk(output_dir)
        ```

        Args:
            source_dir: Directory containing the sharded datasets
            output_dir: Directory to consolidate the shards into
            copy_files: If True, copy files; if False, move them and delete source_dir
        """
        first_shard_dir_name = "shard_00000"  # shard_{i:05d}

        if not source_dir.exists() or not source_dir.is_dir():
            raise NotADirectoryError(
                f"source_dir is not an existing directory: {source_dir}"
            )

        if not output_dir.exists() or not output_dir.is_dir():
            raise NotADirectoryError(
                f"output_dir is not an existing directory: {output_dir}"
            )

        other_items = [p for p in output_dir.iterdir() if p.name != _TMP_SHARDS_DIR]
        if other_items:
            raise FileExistsError(
                f"output_dir must be empty (besides {_TMP_SHARDS_DIR}). Found: {other_items}"
            )

        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        # Move dataset_info.json from any shard (all the same)
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue

            # state.json contains arrow filenames
            state = json.loads((shard_dir / "state.json").read_text())

            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(list(source_dir.iterdir())):05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,  # temporary
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        # fingerprint is generated from dataset.__getstate__ (not including _fingerprint)
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:  # cleanup source dir
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(output_dir)

    @torch.no_grad()
    def run(self) -> Dataset:
        activation_save_path = self.cfg.new_cached_activations_path
        assert activation_save_path is not None

        ### Paths setup
        final_cached_activation_path = Path(activation_save_path)
        final_cached_activation_path.mkdir(exist_ok=True, parents=True)
        if any(final_cached_activation_path.iterdir()):
            raise Exception(
                f"Activations directory ({final_cached_activation_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
            )

        tmp_cached_activation_path = final_cached_activation_path / _TMP_SHARDS_DIR
        tmp_cached_activation_path.mkdir(exist_ok=False, parents=False)

        ### Create temporary sharded datasets

        logger.info(f"Started caching activations for {self.cfg.dataset_path}")

        for i in tqdm(range(self.cfg.n_buffers), desc="Caching activations"):
            try:
                # Accumulate n_batches_in_buffer batches into one shard
                buffers: list[tuple[torch.Tensor, torch.Tensor | None]] = []
                for _ in range(self.cfg.n_batches_in_buffer):
                    buffers.append(self.activations_store.get_raw_llm_batch())
                # Concatenate all batches
                acts = torch.cat([b[0] for b in buffers], dim=0)
                token_ids: torch.Tensor | None = None
                if buffers[0][1] is not None:
                    # All batches have token_ids if the first one does
                    token_ids = torch.cat([b[1] for b in buffers], dim=0)  # type: ignore[arg-type]
                shard = self._create_shard((acts, token_ids))
                shard.save_to_disk(
                    f"{tmp_cached_activation_path}/shard_{i:05d}", num_shards=1
                )
                del buffers, acts, token_ids, shard
            except StopIteration:
                logger.warning(
                    f"Warning: Ran out of samples while filling the buffer at batch {i} before reaching {self.cfg.n_buffers} batches."
                )
                break

        ### Concatenate shards and push to Huggingface Hub

        dataset = self._consolidate_shards(
            tmp_cached_activation_path, final_cached_activation_path, copy_files=False
        )

        if self.cfg.shuffle:
            # shuffle_across_sequences: shuffle individual activations globally,
            # treating the entire dataset as a flat array of (total_tokens, d_in).
            # This breaks up sequential patterns within sequences while keeping
            # token_ids paired with their corresponding activations.
            if self.cfg.shuffle_across_sequences:
                logger.info("Shuffling across sequences...")
                dataset.set_format("torch")
                hook_name = self.cfg.hook_name

                # Load all data and flatten
                # With torch format, [:] returns tensors directly
                all_data = dataset[:]
                acts = all_data[hook_name]  # (n_seq, context_size, d_in)
                token_ids = all_data["token_ids"]  # (n_seq, context_size)
                n_seq = acts.shape[0]

                acts_flat = einops.rearrange(
                    acts, "n_seq context_size d_in -> (n_seq context_size) d_in"
                )
                token_ids_flat = einops.rearrange(
                    token_ids, "n_seq context_size -> (n_seq context_size)"
                )

                # Shuffle globally with the same permutation for both
                generator = torch.Generator().manual_seed(self.cfg.seed)
                perm = torch.randperm(acts_flat.shape[0], generator=generator)
                acts_flat = acts_flat[perm]
                token_ids_flat = token_ids_flat[perm]

                # Reshape back to sequences
                acts_shuffled = einops.rearrange(
                    acts_flat,
                    "(n_seq context_size) d_in -> n_seq context_size d_in",
                    n_seq=n_seq,
                    context_size=self.context_size,
                )
                token_ids_shuffled = einops.rearrange(
                    token_ids_flat,
                    "(n_seq context_size) -> n_seq context_size",
                    n_seq=n_seq,
                    context_size=self.context_size,
                )

                # Create new dataset from shuffled data
                dataset = Dataset.from_dict(
                    {
                        hook_name: acts_shuffled,
                        "token_ids": token_ids_shuffled.to(torch.int32),
                    },
                    features=self.features,
                )
            else:
                # Sequence-level shuffle only: shuffle the order of sequences (rows)
                # Skip if shuffle_across_sequences was used since global shuffle is stronger
                logger.info("Shuffling sequences...")
                dataset = dataset.shuffle(seed=self.cfg.seed)

            # Save the shuffled dataset back to disk using atomic rename with backup
            # to prevent data loss if the process crashes mid-operation.
            # Note: shuffled_path must be a sibling (not child) of final_cached_activation_path
            # so that renaming the parent doesn't invalidate the shuffled path.
            shuffled_path = final_cached_activation_path.parent / (
                final_cached_activation_path.name + _SHUFFLED_DIR
            )
            backup_path = final_cached_activation_path.parent / (
                final_cached_activation_path.name + _BACKUP_SUFFIX
            )

            dataset.save_to_disk(str(shuffled_path))

            # Atomic swap: rename original to backup, then shuffled to original
            try:
                final_cached_activation_path.rename(backup_path)
                shuffled_path.rename(final_cached_activation_path)
                # Success - remove backup
                shutil.rmtree(backup_path)
            except Exception:
                # Rollback: restore from backup if it exists
                if backup_path.exists() and not final_cached_activation_path.exists():
                    backup_path.rename(final_cached_activation_path)
                # Clean up shuffled path if it still exists
                if shuffled_path.exists():
                    shutil.rmtree(shuffled_path)
                raise

            # Reload the dataset from the new location
            dataset = Dataset.load_from_disk(str(final_cached_activation_path))

        if self.cfg.hf_repo_id:
            logger.info("Pushing to Huggingface Hub...")
            dataset.push_to_hub(
                repo_id=self.cfg.hf_repo_id,
                num_shards=self.cfg.hf_num_shards,
                private=self.cfg.hf_is_private_repo,
                revision=self.cfg.hf_revision,
            )

            meta_io = io.BytesIO()
            meta_contents = json.dumps(
                asdict(self.cfg), indent=2, ensure_ascii=False
            ).encode("utf-8")
            meta_io.write(meta_contents)
            meta_io.seek(0)

            api = HfApi()
            api.upload_file(
                path_or_fileobj=meta_io,
                path_in_repo="cache_activations_runner_cfg.json",
                repo_id=self.cfg.hf_repo_id,
                repo_type="dataset",
                commit_message="Add cache_activations_runner metadata",
            )

        return dataset

    def _create_shard(
        self,
        buffer: tuple[
            torch.Tensor,  # shape: (bs context_size) d_in
            torch.Tensor | None,  # shape: (bs context_size) or None
        ],
    ) -> Dataset:
        hook_names = [self.cfg.hook_name]
        acts, token_ids = buffer

        acts = einops.rearrange(
            acts,
            "(bs context_size) d_in -> bs context_size d_in",
            bs=self.cfg.n_seq_in_buffer,
            context_size=self.context_size,
            d_in=self.cfg.d_in,
        )
        shard_dict: dict[str, object] = {
            hook_name: act_batch
            for hook_name, act_batch in zip(hook_names, [acts], strict=True)
        }

        if token_ids is not None:
            token_ids = einops.rearrange(
                token_ids,
                "(bs context_size) -> bs context_size",
                bs=self.cfg.n_seq_in_buffer,
                context_size=self.context_size,
            )
            shard_dict["token_ids"] = token_ids.to(torch.int32)
        return Dataset.from_dict(
            shard_dict,
            features=self.features,
        )

    @staticmethod
    def _get_sliced_context_size(
        context_size: int, seqpos_slice: tuple[int | None, ...] | None
    ) -> int:
        if seqpos_slice is not None:
            context_size = len(range(context_size)[slice(*seqpos_slice)])
        return context_size
