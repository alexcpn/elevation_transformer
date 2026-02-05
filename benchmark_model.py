import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import glob
import math
import json
import time
import argparse
import logging as log
from datetime import datetime
from pathloss_transformer import create_model, load_weights
import random
from torch.utils.data import DataLoader
from pathloss_dataset import PathLossDataset

# Configure logging
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S')

# Enable TF32 for faster matrix multiplication on Ampere+ GPUs
torch.set_float32_matmul_precision('high')
NUM_WORKERS = 4

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark PathLoss Transformer")
    parser.add_argument("--weights", type=str, default="./weights/model_weights20260204165247.pth", help="Path to model weights file")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for inference")
    parser.add_argument("--data_dir", type=str, default="itm_loss", help="Directory containing parquet files")
    args = parser.parse_args()

    # Find latest weights if not specified
    weights_path = args.weights
    if not weights_path:
        log.info(f"No weights specified")
        return

    # Load Model
    log.info("Loading model...")
    model = create_model()
    load_weights(model, weights_path)
    model.to('cuda')
    model.eval()
    
    # Optimize model with torch.compile
    if hasattr(torch, 'compile'):
        log.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    else:
        log.info("torch.compile not available, skipping compilation.")

    # Prepare Data
    INPUT_DIR = "/data/itm_loss/"
    parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    random.seed(42)  # For reproducibility
    random.shuffle(parquet_files)  # Shuffle to ensure train/val have similar distributions
    nfiles = len(parquet_files)
    log.info(f"Number of parquet files: {nfiles}")

    # Train/val split by file
    train_ratio = 0.99
    split_idx = int(nfiles * train_ratio)
    parquet_files_train = parquet_files[:split_idx]
    parquet_files_valid = parquet_files[split_idx:]
    log.info(f"Train files: {len(parquet_files_train)}")
    log.info(f"Validation files: {len(parquet_files_valid)}")
        
    log.info(f"Found {nfiles} total files. Using {len(parquet_files_valid)} for validation.")
    #val_dataset = PathLossDataset(INPUT_DIR, file_list=parquet_files_valid[:10])
    val_dataset = PathLossDataset(None, split="val", max_samples=1000)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    # Benchmarking
    all_predictions = []
    all_targets = []
    
    log.info("Running inference on validation set...")
    start_time = time.time()
    
    with torch.no_grad():
       for step, (input_features, elevation, path_loss, mask) in enumerate(val_loader, args.batch_size):
   
            input_features = input_features.to('cuda')
            elevation_data = elevation.to('cuda')
            target_labels = path_loss.to('cuda')
            # Forward pass
            logits = model(input_features, elevation_data)
            # Store results
            preds_norm = logits.cpu().numpy()
            targets_norm = target_labels.cpu().numpy()
            
            all_predictions.extend(preds_norm)
            all_targets.extend(targets_norm)


    end_time = time.time()
    total_time = end_time - start_time
    
    # Compute Metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    mae = np.mean(np.abs(all_predictions - all_targets))
    errors = np.abs(all_predictions - all_targets)
    p50_error = np.percentile(errors, 50)
    p90_error = np.percentile(errors, 90)
    p95_error = np.percentile(errors, 95)
    
    samples_per_second = len(all_predictions) / total_time
    
    log.info("=" * 60)
    log.info("BENCHMARK RESULTS")
    log.info("=" * 60)
    log.info(f"Model: {weights_path}")
    log.info(f"Samples: {len(all_predictions)}")
    log.info(f"RMSE: {rmse:.2f}")
    log.info(f"MAE: {mae:.2f}")
    log.info(f"Median Error: {p50_error:.2f}")
    log.info(f"90th Percentile: {p90_error:.2f}")
    log.info(f"Throughput: {samples_per_second:.0f} samples/sec")
    log.info("=" * 60)

    # Save results
    results = {
        "model": weights_path,
        "rmse": float(rmse),
        "mae": float(mae),
        "median_error": float(p50_error),
        "p90_error": float(p90_error),
        "p95_error": float(p95_error),
        "throughput": float(samples_per_second)
    }
    
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
