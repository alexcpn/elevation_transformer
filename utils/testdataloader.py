from torch.utils.data import  DataLoader,SubsetRandomSampler
import glob
import os
import numpy as np
#add parent directory to path
import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from dataset import PathLossDataset


# Define the folder containing Parquet files
INPUT_DIR = "./itm_loss"
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
for i, (input_features, elevation_data, path_loss,_) in enumerate(train_loader):
    print(f"Batch {i}: Extra features shape: {input_features.shape}, Elevation data shape: {elevation_data.shape}, Path loss shape: {path_loss.shape}")
    input_features =(input_features - input_features.mean(dim=1, keepdim=True)) / (input_features.std(dim=1, keepdim=True) + 1e-6)
    elevation_data = (elevation_data - elevation_data.mean(dim=1, keepdim=True)) / (elevation_data.std(dim=1, keepdim=True) + 1e-6)
    path_loss = (path_loss - path_loss.mean()) / (path_loss.std() + 1e-6)
        
    print(f"Extra features: {input_features[:1]}")
    #print(f"Elevation data: {elevation_data[:1]}")
    print(f"Path loss: {path_loss}")
    if i == 2:
        break
    
# for i, (extra_features, elevation_data, path_loss,_) in enumerate(val_loader):
#     print(f"Batch {i}: Extra features shape: {extra_features.shape}, Elevation data shape: {elevation_data.shape}, Path loss shape: {path_loss.shape}")
#     if i == 2:
#         break