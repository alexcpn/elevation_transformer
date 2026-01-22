"""
Analyze parquet files to get dataset statistics for white paper
"""
import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

INPUT_DIR = "./itm_loss"

# Get all parquet files
parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
nfiles = len(parquet_files)
print(f"Number of parquet files: {nfiles}")

# Initialize statistics collectors
total_rows = 0
all_frequencies = []
all_distances = []
all_path_loss = []
all_receiver_ht = []
all_ap_ht = []
all_elevation_lengths = []

print("Analyzing parquet files...")
for i, file in enumerate(tqdm(parquet_files)):
    try:
        df = pd.read_parquet(file, engine='pyarrow')
        total_rows += len(df)

        # Collect statistics
        if 'center_freq' in df.columns:
            all_frequencies.extend(df['center_freq'].tolist())
        if 'dEP_FSRx_m' in df.columns:
            all_distances.extend(df['dEP_FSRx_m'].tolist())
        if 'path_loss' in df.columns:
            all_path_loss.extend(df['path_loss'].tolist())
        if 'receiver_ht_m' in df.columns:
            all_receiver_ht.extend(df['receiver_ht_m'].tolist())
        if 'accesspoint_ht_m' in df.columns:
            all_ap_ht.extend(df['accesspoint_ht_m'].tolist())
        if 'elevation_data' in df.columns:
            for elev in df['elevation_data']:
                all_elevation_lengths.append(len(elev) if isinstance(elev, (list, np.ndarray)) else 0)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Convert to numpy arrays
all_frequencies = np.array(all_frequencies)
all_distances = np.array(all_distances)
all_path_loss = np.array(all_path_loss)
all_receiver_ht = np.array(all_receiver_ht)
all_ap_ht = np.array(all_ap_ht)
all_elevation_lengths = np.array(all_elevation_lengths)

# Print results
print("\n" + "=" * 60)
print("DATASET STATISTICS FOR WHITE PAPER")
print("=" * 60)

print(f"\n1. TOTAL SAMPLES")
print(f"   Files: {nfiles}")
print(f"   Total rows: {total_rows:,}")
print(f"   Avg rows per file: {total_rows/nfiles:.1f}")

print(f"\n2. FREQUENCY (center_freq)")
print(f"   Min:  {all_frequencies.min()/1e6:.2f} MHz")
print(f"   Max:  {all_frequencies.max()/1e6:.2f} MHz")
print(f"   Mean: {all_frequencies.mean()/1e6:.2f} MHz")
print(f"   Unique values: {len(np.unique(all_frequencies))}")
if len(np.unique(all_frequencies)) < 20:
    print(f"   Values: {sorted(np.unique(all_frequencies)/1e6)} MHz")

print(f"\n3. DISTANCE (dEP_FSRx_m)")
print(f"   Min:  {all_distances.min()/1000:.2f} km")
print(f"   Max:  {all_distances.max()/1000:.2f} km")
print(f"   Mean: {all_distances.mean()/1000:.2f} km")
print(f"   Std:  {all_distances.std()/1000:.2f} km")

print(f"\n4. PATH LOSS (target)")
print(f"   Min:  {all_path_loss.min():.2f} dB")
print(f"   Max:  {all_path_loss.max():.2f} dB")
print(f"   Mean: {all_path_loss.mean():.2f} dB")
print(f"   Std:  {all_path_loss.std():.2f} dB")

print(f"\n5. RECEIVER HEIGHT")
print(f"   Min:  {all_receiver_ht.min():.2f} m")
print(f"   Max:  {all_receiver_ht.max():.2f} m")
print(f"   Mean: {all_receiver_ht.mean():.2f} m")
if len(np.unique(all_receiver_ht)) < 20:
    print(f"   Unique values: {sorted(np.unique(all_receiver_ht))} m")

print(f"\n6. ACCESS POINT HEIGHT")
print(f"   Min:  {all_ap_ht.min():.2f} m")
print(f"   Max:  {all_ap_ht.max():.2f} m")
print(f"   Mean: {all_ap_ht.mean():.2f} m")
if len(np.unique(all_ap_ht)) < 20:
    print(f"   Unique values: {sorted(np.unique(all_ap_ht))} m")

print(f"\n7. ELEVATION PROFILE LENGTH")
print(f"   Min:  {all_elevation_lengths.min()}")
print(f"   Max:  {all_elevation_lengths.max()}")
print(f"   Mean: {all_elevation_lengths.mean():.1f}")

# Summary for abstract
print("\n" + "=" * 60)
print("COPY-PASTE FOR ABSTRACT")
print("=" * 60)
freq_min = all_frequencies.min()/1e9
freq_max = all_frequencies.max()/1e9
dist_min = all_distances.min()/1000
dist_max = all_distances.max()/1000

print(f"""
Dataset: {total_rows:,} ITM-generated path loss samples
Frequency range: {freq_min:.2f} - {freq_max:.2f} GHz
Distance range: {dist_min:.1f} - {dist_max:.1f} km
Antenna heights: {all_ap_ht.min():.0f}-{all_ap_ht.max():.0f} m (AP), {all_receiver_ht.min():.0f}-{all_receiver_ht.max():.0f} m (Rx)
Path loss range: {all_path_loss.min():.0f} - {all_path_loss.max():.0f} dB
""")

# Save to JSON for reference
import json
stats = {
    "total_files": nfiles,
    "total_samples": total_rows,
    "frequency_min_mhz": float(all_frequencies.min()/1e6),
    "frequency_max_mhz": float(all_frequencies.max()/1e6),
    "distance_min_km": float(all_distances.min()/1000),
    "distance_max_km": float(all_distances.max()/1000),
    "path_loss_min_db": float(all_path_loss.min()),
    "path_loss_max_db": float(all_path_loss.max()),
    "path_loss_mean_db": float(all_path_loss.mean()),
    "path_loss_std_db": float(all_path_loss.std()),
    "receiver_ht_min_m": float(all_receiver_ht.min()),
    "receiver_ht_max_m": float(all_receiver_ht.max()),
    "ap_ht_min_m": float(all_ap_ht.min()),
    "ap_ht_max_m": float(all_ap_ht.max()),
}

with open("./logs/dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("Stats saved to ./logs/dataset_stats.json")
