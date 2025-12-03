from torch.utils.data import  DataLoader,SubsetRandomSampler
import glob
import os
import numpy as np
import pandas as pd
import torch
import json

# Define the folder containing Parquet files
INPUT_DIR = "/data/itm_loss"
# Get a list of all Parquet files in the folder (sorted for consistency)
parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
#parquet_files = parquet_files[:1]  # Limit to 10 files for testing
nfiles = len(parquet_files)
print(f"Number of parquet_files= {nfiles}")
# Compute split index
count = 0
for file in parquet_files:
    count += 1
    print(f"Processing file: {file} ({count}/{nfiles})")
    try:
        df = pd.read_parquet(file)
        row = df.iloc[0]
        # #    Convert elevation_data from string (if needed)
        if isinstance(row["elevation_data"], str):
            elevation_data_list = json.loads(row["elevation_data"])  # Convert from JSON string to list
            df["elevation_data"] = df["elevation_data"].apply(json.loads)
            
        # Generate new file name with "_p" suffix
        new_file = file.replace(".parquet", "_p.parquet")

        # Save compressed version
        df.to_parquet(new_file, engine="pyarrow", compression="zstd", index=False)
        # Delete original file after processing
        os.remove(file)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        continue
    # #Read    
    # extra_features = torch.tensor([
    #     row['dEP_FSRx_m']/1e3,
    #     row['center_freq'] ,
    #     row['receiver_ht_m'],
    #     row['accesspoint_ht_m']
    # ], dtype=torch.float32)
    # elevation_data_list = row["elevation_data"]  # Already a list
    # print(f"Extra features: {extra_features}")
    # elevation_data = torch.tensor(elevation_data_list, dtype=torch.float32)
    # print(f"Extra features: {elevation_data[:10]}")
    
    