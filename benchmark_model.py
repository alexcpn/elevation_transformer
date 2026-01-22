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
from pathloss_transformer import create_model, process_batch, load_weights, TARGET_MEAN, TARGET_STD

# Configure logging
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S')

# Enable TF32 for faster matrix multiplication on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# ============================================================
# DATA LOADING UTILS
# ============================================================

def batch_loader(parquet_files, batch_size):
    # 1) Compute total rows across all files
    total_rows = 0
    for file in parquet_files:
        tmp = pd.read_parquet(file, engine='pyarrow', columns=['path_loss'])
        total_rows += len(tmp)
    num_steps = math.ceil(total_rows / batch_size)
    log.info(f"Total Steps = {num_steps}")
    
    steps = 0
    for start_idx in range(0, len(parquet_files), batch_size):
        chunk_files = parquet_files[start_idx:start_idx + batch_size]
        df_list = [pd.read_parquet(file, engine='pyarrow') for file in chunk_files]
        if not df_list:
            continue
        df = pd.concat(df_list, ignore_index=True)
        
        for row_start in range(0, len(df), batch_size):
            batch_df = df.iloc[row_start:row_start + batch_size]
            if len(batch_df) == 0:
                break
            steps += 1
            yield batch_df, steps, num_steps

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark PathLoss Transformer")
    parser.add_argument("--weights", type=str, help="Path to model weights file")
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
    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    nfiles = len(parquet_files)
    train_ratio = .1 # Using 10% for validation/benchmarking as per previous edit
    split_idx = int(nfiles * train_ratio)
    parquet_files_valid = parquet_files[:split_idx]
    
    log.info(f"Found {nfiles} total files. Using {len(parquet_files_valid)} for validation.")

    # Benchmarking
    all_predictions = []
    all_targets = []
    all_predictions_db = []
    all_targets_db = []
    
    log.info("Running inference on validation set...")
    start_time = time.time()
    
    with torch.no_grad():
        for df, step, total_steps in batch_loader(parquet_files_valid, args.batch_size):
            input_features, elevation_data, path_loss = process_batch(df)

            input_features = input_features.to('cuda')
            elevation_data = elevation_data.to('cuda')
            target_labels = path_loss.to('cuda')

            # Forward pass
            logits = model(input_features, elevation_data)

            # Store results
            preds_norm = logits.squeeze(1).cpu().numpy()
            targets_norm = target_labels.cpu().numpy()
            
            all_predictions.extend(preds_norm)
            all_targets.extend(targets_norm)
            
            # Denormalize
            preds_db = preds_norm * TARGET_STD + TARGET_MEAN
            targets_db = targets_norm * TARGET_STD + TARGET_MEAN
            
            all_predictions_db.extend(preds_db)
            all_targets_db.extend(targets_db)
            
            if step % 100 == 0:
                log.info(f"Processed step {step}/{total_steps}")

    end_time = time.time()
    total_time = end_time - start_time
    
    # Compute Metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_predictions_db = np.array(all_predictions_db)
    all_targets_db = np.array(all_targets_db)

    rmse_db = np.sqrt(np.mean((all_predictions_db - all_targets_db) ** 2))
    mae_db = np.mean(np.abs(all_predictions_db - all_targets_db))
    errors_db = np.abs(all_predictions_db - all_targets_db)
    p50_error = np.percentile(errors_db, 50)
    p90_error = np.percentile(errors_db, 90)
    p95_error = np.percentile(errors_db, 95)
    
    samples_per_second = len(all_predictions_db) / total_time
    
    log.info("=" * 60)
    log.info("BENCHMARK RESULTS")
    log.info("=" * 60)
    log.info(f"Model: {weights_path}")
    log.info(f"Samples: {len(all_predictions_db)}")
    log.info(f"RMSE (dB): {rmse_db:.2f}")
    log.info(f"MAE (dB): {mae_db:.2f}")
    log.info(f"Median Error (dB): {p50_error:.2f}")
    log.info(f"90th Percentile (dB): {p90_error:.2f}")
    log.info(f"Throughput: {samples_per_second:.0f} samples/sec")
    log.info("=" * 60)

    # Save results
    results = {
        "model": weights_path,
        "rmse_db": float(rmse_db),
        "mae_db": float(mae_db),
        "median_error_db": float(p50_error),
        "p90_error_db": float(p90_error),
        "p95_error_db": float(p95_error),
        "throughput": float(samples_per_second)
    }
    
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
