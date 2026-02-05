import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
import os
from datasets import load_dataset #huggingface datasets

from torch.utils.data import IterableDataset, get_worker_info

class PathLossDataset(IterableDataset):
    def __init__(self, parquet_dir, file_list=None, seq_length=750, split="train", max_samples=None):
        """
        Dataset that loads directly from parquet files using Hugging Face datasets in STREAMING mode.
        This avoids creating large cache files on disk.
        
        Args:
            parquet_dir: Directory containing parquet files
            file_list: Optional list of specific parquet file paths.
                       If None, uses all parquet files in parquet_dir.
            seq_length: Fixed sequence length for elevation (pad/truncate)
            split: "train" (80%), "val" (10%), or "test" (10%) - only used for HF dataset
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
            
            # Apply split using modular arithmetic on indices
            # Train: indices where idx % 10 < 8  (80%)
            # Val:   indices where idx % 10 == 8 (10%)
            # Test:  indices where idx % 10 == 9 (10%)
            if split == "train":
                self.dataset = full_ds.filter(lambda x, idx: idx % 10 < 8, with_indices=True)
            elif split == "val":
                self.dataset = full_ds.filter(lambda x, idx: idx % 10 == 8, with_indices=True)
            else:  # test
                self.dataset = full_ds.filter(lambda x, idx: idx % 10 == 9, with_indices=True)
            
            self.n_files = 1  # approximate
        else:
            print(f"Loading parquet files from {parquet_dir}")   
            self.dataset = load_dataset("parquet", data_files=files, split="train", streaming=True)
            self.n_files = len(files)
        
        # Apply max_samples limit if specified
        if max_samples is not None:
            self.dataset = self.dataset.take(max_samples)
            print(f"Limiting dataset to {max_samples} samples")
        
        # Set a buffer size for shuffling. 
        # This provides local randomness without loading the whole dataset.
        self.shuffle_buffer_size = 10000

    def __iter__(self):
        worker_info = get_worker_info()
        ds = self.dataset

        # Shard the dataset across workers
        if worker_info is not None:
            # Note: For streaming datasets, shard() interleaves examples or files.
            # With Parquet files, it's generally efficient enough.
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        
        # Apply shuffling (with a buffer)
        # We shuffle *after* sharding to ensure each worker shuffles its own stream
        ds = ds.shuffle(buffer_size=self.shuffle_buffer_size, seed=42)
        
        for row in ds:
            # Elevation: pad/truncate to seq_length, build mask
            elev = row['elevation_profile_m']
            
            # Skip empty profiles and where it is not written properly; proper sequnces should have length
            # 750 (seq_length)
            if len(elev) < self.seq_length:
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
            # Order: ['distance_to_ap_m', 'center_freq_mhz', 'receiver_ht_m', 'accesspoint_ht_m']
            # NORMALIZE features to prevent fp16 overflow (values up to 200k cause NaN in mixed precision)
            feat_vals = [
                row['distance_to_ap_m'] / 100000.0,    # 100km -> 1.0
                row['center_freq_mhz'] / 10000.0,      # 10GHz -> 1.0  
                row['receiver_ht_m'] / 100.0,          # 100m -> 1.0
                row['accesspoint_ht_m'] / 100.0        # 100m -> 1.0
            ]
            features = torch.tensor(feat_vals, dtype=torch.float32)

            # Target (dB)
            target = torch.tensor(row['itm_loss_db'], dtype=torch.float32)

            yield features, elevation, target, mask

    def __len__(self):
        # We cannot easily know the exact length in streaming mode without scanning.
        # Returning an estimate based on file count if needed, or raising NotImplementedError.
        # Ideally, we return a rough count so the progress bar works (even if inaccurate).
        # Based on previous logs: ~26.7M samples for 4741 files -> ~5650 samples/file
        estimated_samples = self.n_files * 5650
        return estimated_samples

