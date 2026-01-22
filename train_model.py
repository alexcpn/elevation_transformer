"""
Using torch.MultiHeadAttention instead of custom implementation
"""

# !pip install datasets
# !pip install --upgrade sentencepiece

# configure logging
import torch.nn.functional as F
import torch.nn as nn
import torch
import sentencepiece as spm
from datasets import load_dataset
import math
import numpy as np
import logging as log
import os
import gc
import pandas as pd
import glob
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from pathloss_transformer import create_model, process_batch, TARGET_MEAN, TARGET_STD, load_weights

# ============================================================
# CONFIGURATION
# ============================================================

# Set to None to train from scratch, or path to weights file to resume
RESUME_FROM_WEIGHTS = "./weights/model_weights20260122140404.pth"  # Set to None to start fresh

# High loss threshold for diagnostic logging
HIGH_LOSS_THRESHOLD = 0.5  # Log batches with loss above this value

datetimesatmp = datetime.now().strftime("%Y%m%d%H%M%S")

outfile = f"./logs/pl_{datetimesatmp}_.log"
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler(),
                ],
                force=True,
                )

# Enable TF32 for faster matrix multiplication on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# Enable mixed precision training for memory efficiency
USE_AMP = True  # Automatic Mixed Precision (fp16)

loss_log_file_base = f"./logs/loss_log_{datetimesatmp}_.npy"
loss_log_file = f"./logs/loss_log_{datetimesatmp}_.npy.npz"

# Initialize loss log
if not os.path.exists(loss_log_file):
    np.savez_compressed(loss_log_file, loss=np.array([]))
    log.info(f"Created path loss file {loss_log_file}")

# Load the small dataset for training our tiny language model
# get the dataframe in batches
def batch_loader(parquet_files,batch_size):

      # 1) Compute total rows across all files
    total_rows = 0
    for file in parquet_files:
        tmp = pd.read_parquet(file, engine='pyarrow', columns=['path_loss'])
        total_rows += len(tmp)
    num_steps = math.ceil(total_rows / batch_size)
    log.info(f"Total Steps = {num_steps}")
    # dEP_FSRx_m  center_freq  receiver_ht_m  accesspoint_ht_m  elevation_data   path_loss
    # Iterate in increments of `batch_size`.
    # The last chunk may be smaller than `batch_size`, but it will still be yielded.
    steps =0
    for start_idx in range(0, len(parquet_files), batch_size):
        # Slice out the chunk of files for this batch
        chunk_files = parquet_files[start_idx:start_idx + batch_size]
        # Read and concatenate them
        df_list = [pd.read_parquet(file, engine='pyarrow') for file in chunk_files]
        df = pd.concat(df_list, ignore_index=True)
       # Ensure we yield exactly `batch_size` rows per step
        for row_start in range(0, len(df), batch_size):
            batch_df = df.iloc[row_start:row_start + batch_size]  # Slice out the required batch
            
            if len(batch_df) == 0:  # If there are no more full batches left, stop
                break
            steps += 1
            yield batch_df, steps, num_steps


read_seq_length = 768

# Create model from shared definition
model = create_model()

# Define the loss function
loss_function = nn.SmoothL1Loss() # since we have just regression

# SGD is unstable and hence we use this
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Mixed precision scaler
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

# Place all in GPU
model.to('cuda')

# Load pre-trained weights if specified
if RESUME_FROM_WEIGHTS and os.path.exists(RESUME_FROM_WEIGHTS):
    log.info(f"Loading weights from: {RESUME_FROM_WEIGHTS}")
    model = load_weights(model, RESUME_FROM_WEIGHTS)
    log.info("Weights loaded successfully. Resuming training.")
elif RESUME_FROM_WEIGHTS:
    log.warning(f"Weights file not found: {RESUME_FROM_WEIGHTS}. Training from scratch.")

# Optimize model with torch.compile
# Note: torch.compile can cause issues with MultiheadAttention backward pass
# Use dynamic=True to handle varying batch sizes, or disable for training
ENABLE_COMPILE = False  # Set to True to enable compilation (may cause stride errors)

if ENABLE_COMPILE and hasattr(torch, 'compile'):
    log.info("Compiling model with torch.compile (dynamic=True)...")
    # dynamic=True helps with varying tensor shapes
    # fullgraph=False allows graph breaks which improves compatibility
    model = torch.compile(model, dynamic=True, fullgraph=False)
else:
    log.info("Running without torch.compile (more stable for training).")

# NO NEED TO EXECUTE THIS AGAIN ( this need A100, )
log.info("Training model...")

BATCH_SIZE = 64  # Reduced for memory without torch.compile. With AMP (fp16), can try 96-128. 
model.train()
loss_value_list = []

# Define the folder containing Parquet files
INPUT_DIR = "itm_loss"
# Get a list of all Parquet files in the folder (sorted for consistency)
parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
nfiles = len(parquet_files)
print(f"Number of parquet_files= {nfiles}")
# Compute split index
train_ratio=0.8
split_idx = int(nfiles * train_ratio)
parquet_files_train= parquet_files[:split_idx] 
parquet_files_valid= parquet_files[split_idx:] 
print(f"Train files= {len(parquet_files_train)}")
print(f"Validation files= {len(parquet_files_valid)}")


for epoch in range(1):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for df,step,total_steps in batch_loader(parquet_files_train, BATCH_SIZE):
        dEP_FSRx_m = df['dEP_FSRx_m'] # this will have BATCH_SIZE rows of floats
        center_freq = df['center_freq']
        receiver_ht_m = df['receiver_ht_m']
        accesspoint_ht_m = df['accesspoint_ht_m']
        elevation_data  = df['elevation_data'] # this is a list of max read_seq_length(765)
        path_loss = df['path_loss'] #this is the target label
        input_features, elevation_data, path_loss = process_batch(df,read_seq_length)
  
        # Move to GPU
        input_features = input_features.to('cuda')
        elevation_data = elevation_data.to('cuda')
        target_labels = path_loss.to('cuda')

        # Mixed precision forward pass
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            logits = model(input_features, elevation_data)
            loss = loss_function(logits.squeeze(1), target_labels)

        optimizer.zero_grad()

        # Scaled backward pass for mixed precision
        scaler.scale(loss).backward()

        # Clip gradients (unscale first for proper clipping)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        num_batches += 1

        # Diagnostic logging for high-loss batches
        if loss.item() > HIGH_LOSS_THRESHOLD:
            log.warning(f"HIGH LOSS batch at step {step}: loss={loss.item():.4f}")
            log.warning(f"  Distance (m): min={df['dEP_FSRx_m'].min():.0f}, max={df['dEP_FSRx_m'].max():.0f}, mean={df['dEP_FSRx_m'].mean():.0f}")
            log.warning(f"  Path loss (dB): min={df['path_loss'].min():.1f}, max={df['path_loss'].max():.1f}, mean={df['path_loss'].mean():.1f}")
            log.warning(f"  Frequency (MHz): min={df['center_freq'].min():.0f}, max={df['center_freq'].max():.0f}")
            log.warning(f"  RX height (m): min={df['receiver_ht_m'].min():.1f}, max={df['receiver_ht_m'].max():.1f}")
            log.warning(f"  TX height (m): min={df['accesspoint_ht_m'].min():.1f}, max={df['accesspoint_ht_m'].max():.1f}")
            # Log prediction vs target for worst samples in batch
            with torch.no_grad():
                preds = logits.squeeze(1).cpu().numpy()
                targets = target_labels.cpu().numpy()
                errors = np.abs(preds - targets)
                worst_idx = np.argsort(errors)[-3:]  # Top 3 worst
                log.warning(f"  Worst predictions (normalized): pred={preds[worst_idx]}, target={targets[worst_idx]}, error={errors[worst_idx]}")
                # Denormalize for interpretability
                preds_db = preds[worst_idx] * TARGET_STD + TARGET_MEAN
                targets_db = targets[worst_idx] * TARGET_STD + TARGET_MEAN
                log.warning(f"  Worst predictions (dB): pred={preds_db}, target={targets_db}")

        if epoch == 0 and step == 1:
            log.info(f"Input Features Shape: {input_features.shape}")  # Expected: (BATCH_SIZE, 4)
            log.info(f"Elevation Data Shape: {elevation_data.shape}")  # Expected: (BATCH_SIZE, read_seq_length)
            log.info(f"Path Loss Shape: {path_loss.shape}")  # Expected: (BATCH_SIZE,)
            log.info(f"logits.shape {logits.shape}")
        # print training progress occasionally
        loss_value_list.append((epoch, step, loss.item()))
        if step % 100 == 0:
            log.info("[Epoch=%d | Step=%d/%d] loss=%.4f",
                     epoch+1, step, total_steps,loss.item())
            data = np.load(loss_log_file,allow_pickle=True)
            loss_history = []
            if "loss" in data:
                # Convert to list for appending
                loss_history = data["loss"].tolist()
            loss_list = loss_history + loss_value_list
            np.savez_compressed(
                loss_log_file, loss=np.array(loss_list, dtype=object))
            loss_value_list = []
        del  input_features,elevation_data,target_labels 
        gc.collect()
        torch.cuda.empty_cache()
    avg_epoch_loss = epoch_loss / num_batches
    log.info("---------Epoch %02d | Average Loss: %.4f", epoch+1, avg_epoch_loss)
    # do a validation loss

    model.eval()
    validation_loss = 0
    num_valid_batches = 0
    overestimation_count = 0 # like false positive
    underestimation_count = 0 # like false negative
    with torch.no_grad():
        for df,step,total_steps in batch_loader(parquet_files_valid, BATCH_SIZE):
            dEP_FSRx_m = df['dEP_FSRx_m'] # this will have BATCH_SIZE rows of floats
            center_freq = df['center_freq']
            receiver_ht_m = df['receiver_ht_m']
            accesspoint_ht_m = df['accesspoint_ht_m']
            elevation_data  = df['elevation_data'] # this is a list of max read_seq_length(765)
            path_loss = df['path_loss'] #this is the target label
            input_features, elevation_data, path_loss = process_batch(df,read_seq_length)

            # Move to GPU
            input_features = input_features.to('cuda')
            elevation_data = elevation_data.to('cuda')
            target_labels = path_loss.to('cuda')

            # Mixed precision inference
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                logits = model(input_features, elevation_data)
                loss = loss_function(logits.squeeze(1), target_labels)

            num_valid_batches += 1
            validation_loss += loss.item()
            # Calculate overestimation and underestimation counts
            predictions = logits.squeeze(1).float().cpu().detach().numpy()  # .float() to ensure fp32 for numpy
            targets = target_labels.cpu().detach().numpy()
            diff = predictions - targets
            overestimation_count += np.sum(diff > 0.2)
            underestimation_count += np.sum(diff < 0.2)
    avg_valid_loss = validation_loss / num_valid_batches
    
    log.info("---------Epoch %02d | Average Validation : %.4f", epoch+1, avg_valid_loss)
    log.info(f"---------Epoch %02d | Overestimation Count (like False Positive): {overestimation_count}", epoch+1)
    log.info(f"---------Epoch %02d | Underestimation Count (like False Negative): {underestimation_count}", epoch+1)

"""# Use the trained model to predict"""

# save the model weights
save_path = f"./weights/model_weights{datetimesatmp}.pth"
torch.save(model.state_dict(), save_path)
log.info(f"Model weights saved at {save_path}")

log.info("Training Over")

# ============================================================
# BENCHMARKING SECTION - For White Paper Metrics
# ============================================================
log.info("=" * 60)
log.info("BENCHMARKING - Computing metrics for white paper")
log.info("=" * 60)


model.eval()
all_predictions = []
all_targets = []
all_predictions_db = []
all_targets_db = []

# Collect predictions on validation set
log.info("Computing accuracy metrics on validation set...")
with torch.no_grad():
    for df, step, total_steps in batch_loader(parquet_files_valid, BATCH_SIZE):
        input_features, elevation_data, path_loss = process_batch(df, read_seq_length)

        input_features = input_features.to('cuda')
        elevation_data = elevation_data.to('cuda')
        target_labels = path_loss.to('cuda')

        logits = model(input_features, elevation_data)

        # Store normalized predictions and targets
        preds_norm = logits.squeeze(1).cpu().numpy()
        targets_norm = target_labels.cpu().numpy()

        all_predictions.extend(preds_norm)
        all_targets.extend(targets_norm)

        # Denormalize to dB for real-world metrics
        preds_db = preds_norm * TARGET_STD + TARGET_MEAN
        targets_db = targets_norm * TARGET_STD + TARGET_MEAN

        all_predictions_db.extend(preds_db)
        all_targets_db.extend(targets_db)

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
all_predictions_db = np.array(all_predictions_db)
all_targets_db = np.array(all_targets_db)

# Compute accuracy metrics
rmse_normalized = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
mae_normalized = np.mean(np.abs(all_predictions - all_targets))

rmse_db = np.sqrt(np.mean((all_predictions_db - all_targets_db) ** 2))
mae_db = np.mean(np.abs(all_predictions_db - all_targets_db))

# Percentile errors
errors_db = np.abs(all_predictions_db - all_targets_db)
p50_error = np.percentile(errors_db, 50)
p90_error = np.percentile(errors_db, 90)
p95_error = np.percentile(errors_db, 95)

log.info("-" * 40)
log.info("ACCURACY METRICS (on validation set)")
log.info("-" * 40)
log.info(f"  Samples evaluated: {len(all_predictions_db)}")
log.info(f"  RMSE (normalized): {rmse_normalized:.4f}")
log.info(f"  MAE (normalized):  {mae_normalized:.4f}")
log.info(f"  RMSE (dB):         {rmse_db:.2f} dB")
log.info(f"  MAE (dB):          {mae_db:.2f} dB")
log.info(f"  Median error:      {p50_error:.2f} dB")
log.info(f"  90th percentile:   {p90_error:.2f} dB")
log.info(f"  95th percentile:   {p95_error:.2f} dB")

# Speed benchmark
log.info("-" * 40)
log.info("SPEED BENCHMARK")
log.info("-" * 40)

import time

# Prepare a batch for timing
sample_df = next(batch_loader(parquet_files_valid, BATCH_SIZE))[0]
input_features, elevation_data, path_loss = process_batch(sample_df, read_seq_length)
input_features = input_features.to('cuda')
elevation_data = elevation_data.to('cuda')

# Optionally compile model for inference benchmarking (more stable than during training)
if hasattr(torch, 'compile') and not ENABLE_COMPILE:
    log.info("Compiling model for inference benchmark...")
    try:
        model = torch.compile(model, dynamic=True, fullgraph=False)
    except Exception as e:
        log.warning(f"torch.compile failed for inference: {e}")

# Warm-up GPU
with torch.no_grad():
    for _ in range(10):
        logits = model(input_features, elevation_data)

# Time model inference
torch.cuda.synchronize()
num_runs = 100
start_time = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        logits = model(input_features, elevation_data)
        torch.cuda.synchronize()
end_time = time.time()

total_time = end_time - start_time
time_per_batch = total_time / num_runs
samples_per_second = (BATCH_SIZE * num_runs) / total_time
time_per_sample_us = (total_time / (BATCH_SIZE * num_runs)) * 1e6

log.info(f"  Batch size: {BATCH_SIZE}")
log.info(f"  Runs: {num_runs}")
log.info(f"  Total time: {total_time:.3f} s")
log.info(f"  Time per batch: {time_per_batch*1000:.2f} ms")
log.info(f"  Time per sample: {time_per_sample_us:.1f} µs")
log.info(f"  Throughput: {samples_per_second:.0f} samples/second")

# Summary for white paper
log.info("=" * 60)
log.info("SUMMARY FOR WHITE PAPER")
log.info("=" * 60)
log.info(f"  Dataset: {nfiles} terrain profiles")
log.info(f"  Train/Val split: {len(parquet_files_train)}/{len(parquet_files_valid)}")
log.info(f"  Model accuracy: RMSE = {rmse_db:.2f} dB, MAE = {mae_db:.2f} dB")
log.info(f"  Inference speed: {samples_per_second:.0f} predictions/second")
log.info(f"  Time per prediction: {time_per_sample_us:.1f} µs")
log.info("=" * 60)

# Save benchmark results to file
benchmark_results = {
    "dataset_size": nfiles,
    "train_files": len(parquet_files_train),
    "val_files": len(parquet_files_valid),
    "val_samples": len(all_predictions_db),
    "rmse_db": float(rmse_db),
    "mae_db": float(mae_db),
    "median_error_db": float(p50_error),
    "p90_error_db": float(p90_error),
    "p95_error_db": float(p95_error),
    "samples_per_second": float(samples_per_second),
    "time_per_sample_us": float(time_per_sample_us),
    "batch_size": BATCH_SIZE,
}

import json
benchmark_file = f"./logs/benchmark_{datetimesatmp}.json"
with open(benchmark_file, "w") as f:
    json.dump(benchmark_results, f, indent=2)
log.info(f"Benchmark results saved to {benchmark_file}")
