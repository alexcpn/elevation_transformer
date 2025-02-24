from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import torch
import pandas as pd
import glob
import os
import numpy as np

class PathLossDataset(Dataset):
    def __init__(self, parquet_files, max_len=765):
        self.parquet_files = parquet_files
        self.max_len = max_len
        self.file_lengths = []  # Store the length of each file
        self.cumulative_lengths = [] #Store the cumulative length of all previous files.
        self._calculate_lengths() #calculate the lengths.

    def _calculate_lengths(self):
        cumulative_len = 0
        for file in self.parquet_files:
            df = pd.read_parquet(file)
            length = len(df)
            self.file_lengths.append(length)
            cumulative_len += length
            self.cumulative_lengths.append(cumulative_len)
       
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        # Find which file the index belongs to
        file_index = 0
        for i, cumulative_len in enumerate(self.cumulative_lengths):
            if idx < cumulative_len:
                file_index = i
                break

        # Calculate the index within the selected file
        if file_index == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[file_index - 1]

        # Load the relevant Parquet file
        df = pd.read_parquet(self.parquet_files[file_index])
        row = df.iloc[local_idx]
        extra_features = torch.tensor([
            row['dEP_FSRx_m']/1e3,
            row['center_freq'] / 1e3,
            row['receiver_ht_m'],
            row['accesspoint_ht_m']
        ], dtype=torch.float32)
         
        elevation_data = torch.tensor(row['elevation_data'], dtype=torch.float32)
        seq_len = elevation_data.shape[0]
        if seq_len < self.max_len:
            last_value = elevation_data[-1]
            # Create a padding tensor by repeating the last value
            padding = last_value.repeat(self.max_len - seq_len)
            elevation_data = torch.cat([elevation_data, padding])
        elif seq_len > self.max_len:
            elevation_data = elevation_data[:self.max_len] #truncate if longer.

        path_loss = torch.tensor(row['path_loss'], dtype=torch.float32)
        
          # 4. Create CORRECT mask (True = ignore)
        padding_mask = torch.zeros(self.max_len, dtype=torch.bool)  # Default: don't mask
        padding_mask[seq_len:] = True  # ← Mask padded positions
        return extra_features, elevation_data, path_loss,padding_mask