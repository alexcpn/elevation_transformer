"""
Analyze parquet files to get dataset statistics for white paper
Uses PathLossDataset to ensure consistency with training data.
"""
import sys
import os

# Add parent directory to path to import dataset
sys.path.append("..")    # Assumes we are in /ssd/pathlossTransformer/utils
sys.path.append("/ssd/pathlossTransformer") # Absolute path backup

import numpy as np
import json
from tqdm import tqdm
from dataset import PathLossDataset

# CONFIG
INPUT_DIR = "/data/itm_loss"
SAMPLE_LIMIT = 500000  # Analyze this many samples to save time (set to None for all)

def main():
    print(f"Initializing dataset from {INPUT_DIR}...")
    # Initialize dataset (this sets up the streaming, but doesn't load everything)
    ds = PathLossDataset(INPUT_DIR)
    
    # We access the underlying HuggingFace IterableDataset to get RAW values
    # (The PathLossDataset.__iter__ yields tensors and normalized elevation)
    # We want raw stats for the paper.
    raw_dataset = ds.dataset
    
    print(f"Dataset initialized. Streaming samples (Limit: {SAMPLE_LIMIT})...")

    # Initialize statistics collectors
    all_frequencies = []
    all_distances = []
    all_path_loss = []
    all_receiver_ht = []
    all_ap_ht = []
    all_elevation_lengths = []
    
    count = 0
    
    # Iterate through the streaming dataset
    try:
        for row in tqdm(raw_dataset):
            # Row keys: ['distance_to_ap_m', 'center_freq_mhz', 'receiver_ht_m', 'accesspoint_ht_m', 'elevation_profile_m', 'itm_loss_db']
            
            # Check for missing keys just in case, though schema should be fixed
            if 'center_freq_mhz' in row:
                all_frequencies.append(row['center_freq_mhz'])
            
            if 'distance_to_ap_m' in row:
                all_distances.append(row['distance_to_ap_m'])
                
            if 'itm_loss_db' in row:
                all_path_loss.append(row['itm_loss_db'])
                
            if 'receiver_ht_m' in row:
                all_receiver_ht.append(row['receiver_ht_m'])
                
            if 'accesspoint_ht_m' in row:
                all_ap_ht.append(row['accesspoint_ht_m'])
                
            if 'elevation_profile_m' in row:
                elev = row['elevation_profile_m']
                all_elevation_lengths.append(len(elev) if elev is not None else 0)

            count += 1
            if SAMPLE_LIMIT and count >= SAMPLE_LIMIT:
                print(f"Reached limit of {SAMPLE_LIMIT} samples.")
                break
                
    except Exception as e:
        print(f"Error during iteration: {e}")
        import traceback
        traceback.print_exc()

    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    all_frequencies = np.array(all_frequencies)
    all_distances = np.array(all_distances)
    all_path_loss = np.array(all_path_loss)
    all_receiver_ht = np.array(all_receiver_ht)
    all_ap_ht = np.array(all_ap_ht)
    all_elevation_lengths = np.array(all_elevation_lengths)
    
    n_samples = len(all_path_loss)

    # Print results
    print("\n" + "=" * 60)
    print("DATASET STATISTICS FOR WHITE PAPER")
    print("=" * 60)

    print(f"\n1. SAMPLES ANALYZED")
    print(f"   Total samples: {n_samples:,}")

    if n_samples == 0:
        print("No samples found!")
        return

    print(f"\n2. FREQUENCY (center_freq_mhz)")
    print(f"   Min:  {all_frequencies.min():.2f} MHz")
    print(f"   Max:  {all_frequencies.max():.2f} MHz")
    print(f"   Mean: {all_frequencies.mean():.2f} MHz")
    print(f"   Unique values: {len(np.unique(all_frequencies))}")
    if len(np.unique(all_frequencies)) < 20:
        print(f"   Values: {sorted(np.unique(all_frequencies))} MHz")

    print(f"\n3. DISTANCE (distance_to_ap_m)")
    print(f"   Min:  {all_distances.min()/1000:.2f} km")
    print(f"   Max:  {all_distances.max()/1000:.2f} km")
    print(f"   Mean: {all_distances.mean()/1000:.2f} km")
    print(f"   Std:  {all_distances.std()/1000:.2f} km")

    print(f"\n4. PATH LOSS (itm_loss_db)")
    print(f"   Min:  {all_path_loss.min():.2f} dB")
    print(f"   Max:  {all_path_loss.max():.2f} dB")
    print(f"   Mean: {all_path_loss.mean():.2f} dB")
    print(f"   Std:  {all_path_loss.std():.2f} dB")
    
    # Check for NaN/Inf
    nans = np.isnan(all_path_loss).sum()
    infs = np.isinf(all_path_loss).sum()
    if nans > 0 or infs > 0:
        print(f"   WARNING: Found {nans} NaNs and {infs} Infs in Path Loss!")

    print(f"\n5. RECEIVER HEIGHT (receiver_ht_m)")
    print(f"   Min:  {all_receiver_ht.min():.2f} m")
    print(f"   Max:  {all_receiver_ht.max():.2f} m")
    print(f"   Mean: {all_receiver_ht.mean():.2f} m")
    
    print(f"\n6. ACCESS POINT HEIGHT (accesspoint_ht_m)")
    print(f"   Min:  {all_ap_ht.min():.2f} m")
    print(f"   Max:  {all_ap_ht.max():.2f} m")
    print(f"   Mean: {all_ap_ht.mean():.2f} m")

    print(f"\n7. ELEVATION PROFILE LENGTH")
    print(f"   Min:  {all_elevation_lengths.min()}")
    print(f"   Max:  {all_elevation_lengths.max()}")
    print(f"   Mean: {all_elevation_lengths.mean():.1f}")
    
    # Elevation Distribution Stats? (Optional, skipping for speed)

    # Summary for abstract
    print("\n" + "=" * 60)
    print("COPY-PASTE FOR ABSTRACT")
    print("=" * 60)
    freq_min_ghz = all_frequencies.min()/1000
    freq_max_ghz = all_frequencies.max()/1000
    dist_min_km = all_distances.min()/1000
    dist_max_km = all_distances.max()/1000

    print(f"""
Dataset: {n_samples:,} sampled ITM-generated path loss data points
Frequency range: {freq_min_ghz:.2f} - {freq_max_ghz:.2f} GHz
Distance range: {dist_min_km:.1f} - {dist_max_km:.1f} km
Antenna heights: {all_ap_ht.min():.0f}-{all_ap_ht.max():.0f} m (AP), {all_receiver_ht.min():.0f}-{all_receiver_ht.max():.0f} m (Rx)
Path loss range: {all_path_loss.min():.0f} - {all_path_loss.max():.0f} dB
    """)

    # Save to JSON for reference
    os.makedirs("./logs", exist_ok=True)
    stats = {
        "samples_analyzed": count,
        "frequency_min_mhz": float(all_frequencies.min()),
        "frequency_max_mhz": float(all_frequencies.max()),
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

    with open("./itm_loss_test/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Stats saved to ./itm_loss_test/dataset_stats.json")

if __name__ == "__main__":
    main()
