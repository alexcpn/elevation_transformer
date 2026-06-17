import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
import os
from datasets import load_dataset #huggingface datasets

from torch.utils.data import IterableDataset, get_worker_info

from pathloss_transformer import TARGET_MEAN, TARGET_STD

class PathLossDataset(IterableDataset):
    def __init__(
        self,
        parquet_dir,
        file_list=None,
        seq_length=750,
        split="train",
        max_samples=None,
        shuffle=True,
        split_mod=100,
        val_mods=(0,),
        test_mods=(1,),
    ):
        """
        Dataset that loads directly from parquet files using Hugging Face datasets in STREAMING mode.
        This avoids creating large cache files on disk.
        
        Args:
            parquet_dir: Directory containing parquet files
            file_list: Optional list of specific parquet file paths.
                       If None, uses all parquet files in parquet_dir.
            seq_length: Fixed sequence length for elevation (pad/truncate)
            split: "train", "val", or "test" - modular index split applied to both HF and local parquet
            max_samples: Optional limit on number of samples (for quick benchmarking)
        """
        self.seq_length = seq_length
        self.split = split
        self.max_samples = max_samples
        files = None
        
        # Load parquet files
        if file_list is not None:
            files = file_list
        elif parquet_dir is not None:
            files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        
        if files is None or len(files) == 0:
            print(f"No parquet files found. Loading from HF dataset (split={split})")
            full_ds = load_dataset("alexcpn/longely_rice_model", split="train", streaming=True)
            self.n_files = 4741  # approximate file count for full HF dataset (~26.7M samples)
        else:
            print(f"Loading parquet files from {parquet_dir}")
            full_ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
            self.n_files = len(files)

        # Apply train/val/test split using modular arithmetic on the (original-order) index.
        # Applied to BOTH the HF and the local-parquet streams so val/test never leak into the
        # train set. The split is deterministic per row (files are sorted), so separate
        # PathLossDataset instances for train/val/test stay mutually disjoint.
        # Default: split_mod=100, val_mods=(0,), test_mods=(1,) -> 98% train / 1% val / 1% test.
        val_mods_set = set(val_mods)
        test_mods_set = set(test_mods)
        if split == "train":
            self.dataset = full_ds.filter(
                lambda x, idx: (idx % split_mod) not in val_mods_set and (idx % split_mod) not in test_mods_set,
                with_indices=True,
            )
        elif split == "val":
            self.dataset = full_ds.filter(lambda x, idx: (idx % split_mod) in val_mods_set, with_indices=True)
        else:  # test
            self.dataset = full_ds.filter(lambda x, idx: (idx % split_mod) in test_mods_set, with_indices=True)
        
        # Buffer size for shuffling (local randomness without loading the whole dataset).
        self.shuffle_buffer_size = 10000
        self.shuffle = shuffle

        # SOURCE-LEVEL SHUFFLE (before take): the HF stream is ordered (e.g. by file /
        # difficulty - the first ~20k rows max out at ~201 dB while the full data reaches
        # ~318 dB). Taking the first max_samples therefore grabs a biased low-loss slice,
        # and a full pass feeds easy->hard batches (loss spikes / forgetting). Shuffling here
        # also randomizes the file/shard order, so each split draws a representative sample.
        # Applied AFTER the split filter so train/val/test stay disjoint (the split is keyed
        # on the original stream index, independent of this shuffle).
        if self.shuffle:
            self.dataset = self.dataset.shuffle(seed=42, buffer_size=self.shuffle_buffer_size)

        # Apply max_samples limit if specified (now over the shuffled stream)
        if max_samples is not None:
            self.dataset = self.dataset.take(max_samples)
            print(f"Limiting dataset to {max_samples} samples")

    def __iter__(self):
        worker_info = get_worker_info()
        ds = self.dataset

        # Shard the dataset across workers
        if worker_info is not None:
            # Note: For streaming datasets, shard() interleaves examples or files.
            # With Parquet files, it's generally efficient enough.
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        
        # Apply shuffling (with a buffer) if enabled
        # We shuffle *after* sharding to ensure each worker shuffles its own stream
        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size, seed=42)
        
        for row in ds:
            # Elevation: pad/truncate to seq_length, build mask
            elev = row['elevation_profile_m']
            
            # Skip only truly empty profiles (Transformer yields NaN if mask is all True)
            # Shorter profiles are OK - they get padded and masked
            if len(elev) == 0:
                continue

            elev_len = min(len(elev), self.seq_length)

            elevation = torch.zeros(self.seq_length, dtype=torch.float32)
            
            # Convert to tensor and Normalize
            # Normalization is critical for Transformer stability, especially with fp16
            elev_tensor = torch.tensor(elev[:elev_len], dtype=torch.float32)
            # Per-sample normalization (Instance Normalization style)
            mean = elev_tensor.mean()
            std = elev_tensor.std()
            elev_tensor = (elev_tensor - mean) / (std + 1e-6)  # epsilon prevents div-by-zero
            elevation[:elev_len] = elev_tensor

            # Mask: True = padded (ignored by attention)
            # In PyTorch MultiheadAttention, True in key_padding_mask means "ignore this position"
            mask = torch.zeros(self.seq_length, dtype=torch.bool)
            mask[elev_len:] = True

            # Construct features tensor
            # Order: ['distance_to_ap_m', 'center_freq_mhz', 'receiver_ht_m', 'accesspoint_ht_m',
            #         'elev_mean', 'elev_std']
            # NORMALIZE features to prevent fp16 overflow (values up to 200k cause NaN in mixed precision)
            # The per-sample elevation mean/std are discarded by instance normalization above, but
            # absolute terrain height (mean) and roughness (std, ~ITM's terrain irregularity Δh) are
            # physically relevant to path loss, so feed them back as scalar features (scaled to ~O(1)).
            feat_vals = [
                np.log10(row['distance_to_ap_m']),     # path loss is logarithmic in distance
                np.log10(row['center_freq_mhz']),      # path loss is logarithmic in frequency
                row['receiver_ht_m'] / 100.0,          # 100m -> 1.0
                row['accesspoint_ht_m'] / 100.0,       # 100m -> 1.0
                float(mean) / 1000.0,                  # absolute elevation level (km)
                float(std) / 1000.0                    # terrain roughness (km)
            ]
            features = torch.tensor(feat_vals, dtype=torch.float32)

            # Target: normalize dB to ~zero-mean/unit-scale so SmoothL1 gets a usable
            # gradient (raw 105-202 dB targets stall training). Denormalize with
            # denormalize_target() before reporting/comparing predictions in dB.
            target = torch.tensor(
                (row['itm_loss_db'] - TARGET_MEAN) / TARGET_STD, dtype=torch.float32
            )

            yield features, elevation, target, mask

    def __len__(self):
        # We cannot easily know the exact length in streaming mode without scanning.
        # Returning an estimate based on file count if needed, or raising NotImplementedError.
        # Ideally, we return a rough count so the progress bar works (even if inaccurate).
        # Based on previous logs: ~26.7M samples for 4741 files -> ~5650 samples/file
        estimated_samples = self.n_files * 5650
        return estimated_samples
