"""
Pick one random sample, run it through the transformer and through native ITM,
and compare the two path-loss predictions.

Usage:
    uv run python compare_one.py
    uv run python compare_one.py --data_dir itm_loss_test --weights weights/model_weights20260617145437.pth
    uv run python compare_one.py --seed 123

Note: the TARGET_MEAN/TARGET_STD constants in pathloss_transformer.py must match
those used when the chosen weights were trained, otherwise the denormalized dB
prediction will be off.
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from pathloss_transformer import create_model, load_weights, TARGET_MEAN, TARGET_STD  # noqa: E402
from itm import itmlosswinnf as itmloss  # noqa: E402

SEQ_LENGTH = 750  # must match PathLossDataset(seq_length=...) used in training


def parse_args():
    p = argparse.ArgumentParser(description="Compare model vs ITM on one random sample")
    p.add_argument("--data_dir", default="itm_loss_test", help="Directory with parquet files")
    p.add_argument("--weights", default=None, help="Model weights .pth (default: newest in weights/)")
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    p.add_argument("--target-mean", type=float, default=TARGET_MEAN,
                   help="Target denorm mean; MUST match the value used to train --weights")
    p.add_argument("--target-std", type=float, default=TARGET_STD,
                   help="Target denorm std; MUST match the value used to train --weights")
    return p.parse_args()


def newest_weights():
    files = glob.glob(os.path.join(REPO_ROOT, "weights", "*.pth"))
    if not files:
        raise FileNotFoundError("No weights found in weights/. Pass --weights.")
    return max(files, key=os.path.getmtime)


def pick_random_row(data_dir, rng):
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    df = pd.read_parquet(rng.choice(files))
    return df.iloc[rng.randint(0, len(df) - 1)]


def preprocess(row):
    """Replicate PathLossDataset preprocessing for a single row (see pathloss_dataset.py)."""
    elev = np.asarray(row["elevation_profile_m"], dtype=np.float32)
    elev_len = min(len(elev), SEQ_LENGTH)

    elevation = torch.zeros(SEQ_LENGTH, dtype=torch.float32)
    elev_t = torch.tensor(elev[:elev_len], dtype=torch.float32)
    mean = elev_t.mean()
    std = elev_t.std()
    elevation[:elev_len] = (elev_t - mean) / (std + 1e-6)  # instance normalization

    mask = torch.zeros(SEQ_LENGTH, dtype=torch.bool)
    mask[elev_len:] = True  # True = padded position to ignore

    features = torch.tensor(
        [
            np.log10(row["distance_to_ap_m"]),
            np.log10(row["center_freq_mhz"]),
            row["receiver_ht_m"] / 100.0,
            row["accesspoint_ht_m"] / 100.0,
            float(mean) / 1000.0,
            float(std) / 1000.0,
        ],
        dtype=torch.float32,
    )
    # add batch dimension
    return features.unsqueeze(0), elevation.unsqueeze(0), mask.unsqueeze(0)


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    rng = np.random.RandomState(seed)

    weights = args.weights or newest_weights()
    row = pick_random_row(args.data_dir, rng)

    # --- Model prediction ---
    model = create_model()
    model = load_weights(model, weights)
    model.eval()
    features, elevation, mask = preprocess(row)
    with torch.no_grad():
        pred_norm = model(features, elevation, mask=mask)
    model_db = float(pred_norm.item() * args.target_std + args.target_mean)

    # --- Native ITM prediction (live call) ---
    elev_list = np.asarray(row["elevation_profile_m"], dtype=np.float64).tolist()
    itm_db = float(
        itmloss.ITMLossWinnf().getItmLoss(
            float(row["center_freq_mhz"]),
            float(row["accesspoint_ht_m"]),
            float(row["receiver_ht_m"]),
            elev_list,
            float(row["distance_to_ap_m"]),
        )
    )

    stored_db = float(row["itm_loss_db"])  # ITM value precomputed in the dataset

    print("=" * 56)
    print(f"Random sample (seed={seed}) from {args.data_dir}")
    print(f"Weights: {os.path.basename(weights)}")
    print("-" * 56)
    print(f"  distance        : {row['distance_to_ap_m'] / 1000.0:10.3f} km")
    print(f"  frequency       : {row['center_freq_mhz']:10.1f} MHz")
    print(f"  AP height       : {row['accesspoint_ht_m']:10.2f} m")
    print(f"  receiver height : {row['receiver_ht_m']:10.2f} m")
    print(f"  elevation points: {len(row['elevation_profile_m']):10d}")
    print("-" * 56)
    print(f"  Model path loss : {model_db:10.2f} dB")
    print(f"  ITM  path loss  : {itm_db:10.2f} dB   (live)")
    print(f"  ITM  (stored)   : {stored_db:10.2f} dB   (dataset itm_loss_db)")
    print("-" * 56)
    print(f"  Model - ITM     : {model_db - itm_db:+10.2f} dB  (error)")
    print("=" * 56)


if __name__ == "__main__":
    main()
