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
            row['dEP_FSRx_m'],
            row['center_freq'],
            row['receiver_ht_m'],
            row['accesspoint_ht_m']
        ], dtype=torch.float32)
         
        elevation_data = torch.tensor(row['elevation_data'], dtype=torch.float32)
        seq_len = elevation_data.shape[0]

        if seq_len < self.max_len:
            padding_len = self.max_len - seq_len
            padding = torch.full((padding_len,), float('-inf'), dtype=torch.float32)
            elevation_data = torch.cat([elevation_data, padding])
        elif seq_len > self.max_len:
            elevation_data = elevation_data[:self.max_len] #truncate if longer.

        path_loss = torch.tensor(row['path_loss'], dtype=torch.float32)

        return extra_features, elevation_data, path_loss

# Define the folder containing Parquet files
INPUT_DIR = "loss_parquet_files"
# Get a list of all Parquet files in the folder (sorted for consistency)
parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
parquet_files = parquet_files[:10]  # Limit to 10 files for testing
nfiles = len(parquet_files)
print(f"Number of parquet_files= {nfiles}")
# Compute split index
train_ratio=0.8
BATCH_SIZE = 25  # 5 GB for 

dataset = PathLossDataset(parquet_files)
dataset_len = len(dataset)
train_ratio = 0.8
train_len = int(dataset_len * train_ratio)
val_len = dataset_len - train_len

indices = list(range(dataset_len))
np.random.shuffle(indices)  # Shuffle the indices

train_indices = indices[:train_len]
val_indices = indices[train_len:]

BATCH_SIZE = 25

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4, pin_memory=True)

print(f"Train loader length: {len(train_loader)}")
print(f"Val loader length: {len(val_loader)}")
# train_loader = DataLoader(PathLossDataset(parquet_files),
#                           batch_size=BATCH_SIZE, shuffle=True,
#                           num_workers=4, pin_memory=True)
# Test the DataLoader
for i, (extra_features, elevation_data, path_loss) in enumerate(train_loader):
    print(f"Batch {i}: Extra features shape: {extra_features.shape}, Elevation data shape: {elevation_data.shape}, Path loss shape: {path_loss.shape}")
    if i == 2:
        break
    
for i, (extra_features, elevation_data, path_loss) in enumerate(val_loader):
    print(f"Batch {i}: Extra features shape: {extra_features.shape}, Elevation data shape: {elevation_data.shape}, Path loss shape: {path_loss.shape}")
    if i == 2:
        break