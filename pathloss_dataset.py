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
        skip_samples=0,
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
        
        if files is not None and len(files) == 0:
            # A local source (parquet_dir/file_list) was explicitly requested but nothing was
            # found. Fail loudly instead of silently streaming/downloading from Hugging Face.
            raise FileNotFoundError(
                f"No '*.parquet' files found for the requested local source "
                f"(parquet_dir={parquet_dir!r}, file_list given={file_list is not None}). "
                f"Refusing to fall back to HF streaming. Check the path, or omit --input-dir to stream."
            )

        if files is None:
            print(f"No local dir given. Streaming from HF dataset (split={split})")
            full_ds = load_dataset("alexcpn/longely_rice_model", split="train", streaming=True)
            self.n_files = 4741  # approximate file count for full HF dataset (~26.7M samples)
        else:
            print(f"Loading {len(files)} parquet files locally (no streaming) from {parquet_dir}")
            full_ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
            self.n_files = len(files)

        # Exact row count from parquet footers (reads metadata only - cheap even for
        # thousands of files), so __len__/total_steps reflect the real dataset instead of a
        # hardcoded rows-per-file guess. Only possible for local files; HF streaming stays an
        # estimate.
        self._exact_total_rows = None
        if files is not None:
            import pyarrow.parquet as pq
            self._exact_total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)

        # Apply train/val/test split using modular arithmetic on the (original-order) index.
        # Applied to BOTH the HF and the local-parquet streams so val/test never leak into the
        # train set. The split is deterministic per row (files are sorted), so separate
        # PathLossDataset instances for train/val/test stay mutually disjoint.
        # Default: split_mod=100, val_mods=(0,), test_mods=(1,) -> 98% train / 1% val / 1% test.
        val_mods_set = set(val_mods)
        test_mods_set = set(test_mods)
        # Fraction of rows this split keeps, mirroring the filters below. Used by __len__.
        if split == "train":
            self._split_fraction = (split_mod - len(val_mods_set | test_mods_set)) / split_mod
        elif split == "val":
            self._split_fraction = len(val_mods_set) / split_mod
        else:  # test
            self._split_fraction = len(test_mods_set) / split_mod
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
        # Kept modest (x num_workers buffers live at once, each holding full elevation arrays)
        # to avoid the OOM killer SIGKILL-ing DataLoader workers on RAM-limited pods. File order
        # is already randomized by the source shuffle, so a small row buffer mixes fine.
        self.shuffle_buffer_size = 2000
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

        # RESUME: skip the already-trained samples so a restart continues where it stopped.
        # This is correct ONLY because the shuffle seed is fixed (seed=42) and NUM_WORKERS<=1,
        # so the stream order is identical across runs. Applied AFTER the shuffle so it skips the
        # exact ordered samples the previous run consumed. NOTE: .skip() streams and discards
        # those rows (no model compute), so the first batch after a resume can take a while.
        # Small caveat: empty-elevation rows dropped in __iter__ aren't counted as steps, so a
        # handful of already-seen samples near the boundary may repeat - negligible, never lost.
        self.skip_samples = skip_samples
        if skip_samples:
            self.dataset = self.dataset.skip(skip_samples)
            print(f"RESUME: skipping first {skip_samples:,} samples of the deterministic stream "
                  f"({split}); streaming past them now, first batch will be slow.")

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

        # Shuffling is already applied once at the source in __init__ (randomizes file/shard
        # order + a row buffer). We do NOT shuffle again here: a second buffer only doubled
        # the time-to-first-batch (and memory) without meaningfully improving randomness.

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
        # Local files: exact total row count (from parquet footers) scaled by this split's
        # fraction. HF streaming: fall back to a per-file estimate (~6818 rows/file observed).
        # Note this can't subtract rows skipped for empty elevation profiles (see __iter__),
        # so it's a tight upper bound rather than an exact count.
        if self._exact_total_rows is not None:
            n = int(self._exact_total_rows * self._split_fraction)
        else:
            n = int(self.n_files * 6818 * self._split_fraction)
        if self.max_samples is not None:
            n = min(n, self.max_samples)
        return n
