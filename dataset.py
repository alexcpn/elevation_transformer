import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class PathLossDataset(Dataset):
    def __init__(self, data_dir="./processed_data", split="train", val_ratio=0.1, seed=42):
        self.data_dir = data_dir
        
        # Load stats
        with open(os.path.join(data_dir, "stats.json"), "r") as f:
            self.stats = json.load(f)
            
        self.total_samples = self.stats["total_samples"]
        
        # Load Memmaps (Read-only)
        self.features = np.memmap(os.path.join(data_dir, "features.npy"), dtype='float32', mode='r', shape=(self.total_samples, 4))
        self.elevation = np.memmap(os.path.join(data_dir, "elevation.npy"), dtype='float32', mode='r', shape=(self.total_samples, 765))
        self.targets = np.memmap(os.path.join(data_dir, "targets.npy"), dtype='float32', mode='r', shape=(self.total_samples,))
        self.masks = np.memmap(os.path.join(data_dir, "masks.npy"), dtype='bool', mode='r', shape=(self.total_samples, 765))
        
        # Split indices
        indices = np.arange(self.total_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        val_size = int(self.total_samples * val_ratio)
        if split == "train":
            self.indices = indices[val_size:]
        else:
            self.indices = indices[:val_size]
            
        # Pre-compute normalization constants as tensors for speed
        self.feat_mean = torch.tensor(self.stats["features_mean"], dtype=torch.float32)
        self.feat_std = torch.tensor(self.stats["features_std"], dtype=torch.float32)
        self.elev_mean = torch.tensor(self.stats["elevation_mean"], dtype=torch.float32)
        self.elev_std = torch.tensor(self.stats["elevation_std"], dtype=torch.float32)
        self.target_mean = torch.tensor(self.stats["target_mean"], dtype=torch.float32)
        self.target_std = torch.tensor(self.stats["target_std"], dtype=torch.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map logical index to physical index
        real_idx = self.indices[idx]
        
        # Read from memmap
        # Copy is important to avoid negative strides or memory issues with torch
        features = torch.from_numpy(self.features[real_idx].copy())
        elevation = torch.from_numpy(self.elevation[real_idx].copy())
        target = torch.tensor(self.targets[real_idx], dtype=torch.float32)
        mask = torch.from_numpy(self.masks[real_idx].copy())
        
        # Normalize
        features = (features - self.feat_mean) / (self.feat_std + 1e-6)
        elevation = (elevation - self.elev_mean) / (self.elev_std + 1e-6)
        # target = (target - self.target_mean) / (self.target_std + 1e-6) # Optional: normalize target?
        
        # Note: The original code expected target shape (1,) or scalar. 
        # Let's return scalar for now, or (1,) if needed.
        # Original code: target_labels = (B, 1) or (B,)
        
        return features, elevation, target, mask