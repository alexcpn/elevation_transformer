"""
Path loss transformer training script.
Uses PathLossDataset with DataLoader for proper shuffling and masking.
Only elevation data is normalized.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging as log
import os
import glob
import math
import json
import time
import random
from datetime import datetime
from torch.utils.data import DataLoader

from pathloss_transformer import create_model, load_weights
from pathloss_dataset import PathLossDataset

# ============================================================
# CONFIGURATION
# ============================================================

# Set to None to train from scratch, or path to weights file to resume
RESUME_FROM_WEIGHTS = None

# Weighted loss: upweight hard examples
USE_WEIGHTED_LOSS = False  # Disabled - was causing loss spikes
WEIGHT_SCALE = 0.3

BATCH_SIZE = 64
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

datetimestamp = datetime.now().strftime("%Y%m%d%H%M%S")

outfile = f"./logs/pl_{datetimestamp}_.log"
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
USE_AMP = True

loss_log_file = f"./logs/loss_log_{datetimestamp}_.npy.npz"
if not os.path.exists(loss_log_file):
    np.savez_compressed(loss_log_file, loss=np.array([]))
    log.info(f"Created loss log file {loss_log_file}")

# create the necessary directories if they do not exist
os.makedirs("logs", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# ============================================================
# DATA SETUP
# ============================================================

INPUT_DIR = "/data/itm_loss/"
parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
random.seed(42)  # For reproducibility
random.shuffle(parquet_files)  # Shuffle to ensure train/val have similar distributions
nfiles = len(parquet_files)
log.info(f"Number of parquet files: {nfiles}")

# Train/val split by file
train_ratio = 0.8
split_idx = int(nfiles * train_ratio)
parquet_files_train = parquet_files[:split_idx]
parquet_files_valid = parquet_files[split_idx:]
log.info(f"Train files: {len(parquet_files_train)}")
log.info(f"Validation files: {len(parquet_files_valid)}")

# Create datasets
log.info("Loading training data...")
train_dataset = PathLossDataset(INPUT_DIR, file_list=parquet_files_train)
log.info(f"Training samples: {len(train_dataset)}")

log.info("Loading validation data...")
val_dataset = PathLossDataset(INPUT_DIR, file_list=parquet_files_valid)
log.info(f"Validation samples: {len(val_dataset)}")

# Create data loaders
# Note: For IterableDataset, shuffle must be False in DataLoader (shuffling is done internally)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

try:
    total_steps = len(train_loader)
except:
    # Fallback if __len__ is not available or reliable
    total_steps = len(train_dataset) // BATCH_SIZE

log.info(f"Total training steps per epoch (estimated): {total_steps}")

# ============================================================
# MODEL SETUP
# ============================================================

model = create_model()
loss_function = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

model.to('cuda')

# Load pre-trained weights if specified
if RESUME_FROM_WEIGHTS and os.path.exists(RESUME_FROM_WEIGHTS):
    log.info(f"Loading weights from: {RESUME_FROM_WEIGHTS}")
    model = load_weights(model, RESUME_FROM_WEIGHTS)
    log.info("Weights loaded successfully. Resuming training.")
elif RESUME_FROM_WEIGHTS:
    log.warning(f"Weights file not found: {RESUME_FROM_WEIGHTS}. Training from scratch.")

log.info("Running without torch.compile (more stable for training).")

# ============================================================
# TRAINING
# ============================================================

log.info("Training model...")
loss_value_list = []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for step, (features, elevation, targets, mask) in enumerate(train_loader, 1):
        # Move to GPU
        features = features.cuda(non_blocking=True)
        elevation = elevation.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        # Mixed precision forward pass
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            preds = model(features, elevation, mask=mask)
            loss = loss_function(preds, targets)

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

       
        if epoch == 0 and step == 1:
            log.info(f"Features shape: {features.shape}")
            log.info(f"Elevation shape: {elevation.shape}")
            log.info(f"Targets shape: {targets.shape}")
            log.info(f"Mask shape: {mask.shape}")
            log.info(f"Preds shape: {preds.shape}")

        loss_value_list.append((epoch, step, loss.item()))
        if step % 100 == 0:
            log.info("[Epoch=%d | Step=%d/%d] loss=%.4f",
                     epoch+1, step, total_steps, loss.item())
            # Save loss log
            data = np.load(loss_log_file, allow_pickle=True)
            loss_history = data["loss"].tolist() if "loss" in data else []
            loss_list = loss_history + loss_value_list
            np.savez_compressed(loss_log_file, loss=np.array(loss_list, dtype=object))
            loss_value_list = []

    avg_epoch_loss = epoch_loss / num_batches
    log.info("---------Epoch %02d | Average Loss: %.4f", epoch+1, avg_epoch_loss)

    # ============================================================
    # VALIDATION
    # ============================================================

    model.eval()
    validation_loss = 0.0
    num_valid_batches = 0
    overestimation_count = 0
    underestimation_count = 0

    with torch.no_grad():
        for features, elevation, targets, mask in val_loader:
            features = features.cuda(non_blocking=True)
            elevation = elevation.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                preds = model(features, elevation, mask=mask)
                loss = loss_function(preds, targets)

            num_valid_batches += 1
            validation_loss += loss.item()

            preds_np = preds.float().cpu().numpy()
            targets_np = targets.cpu().numpy()
            diff = preds_np - targets_np
            overestimation_count += np.sum(diff > 3.0)   # 3 dB threshold
            underestimation_count += np.sum(diff < -3.0)

    avg_valid_loss = validation_loss / num_valid_batches
    log.info("---------Epoch %02d | Average Validation: %.4f", epoch+1, avg_valid_loss)
    log.info(f"---------Epoch {epoch+1:02d} | Overestimation (>3dB): {overestimation_count}")
    log.info(f"---------Epoch {epoch+1:02d} | Underestimation (<-3dB): {underestimation_count}")

# ============================================================
# SAVE MODEL
# ============================================================

save_path = f"./weights/model_weights{datetimestamp}.pth"
torch.save(model.state_dict(), save_path)
log.info(f"Model weights saved at {save_path}")
log.info("Training Over")

# ============================================================
# BENCHMARKING SECTION
# ============================================================
log.info("=" * 60)
log.info("BENCHMARKING - Computing metrics")
log.info("=" * 60)

model.eval()
all_predictions_db = []
all_targets_db = []

log.info("Computing accuracy metrics on validation set...")
with torch.no_grad():
    for features, elevation, targets, mask in val_loader:
        features = features.cuda(non_blocking=True)
        elevation = elevation.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            preds = model(features, elevation, mask=mask)

        # Predictions and targets are already in dB
        all_predictions_db.extend(preds.float().cpu().numpy())
        all_targets_db.extend(targets.numpy())

all_predictions_db = np.array(all_predictions_db)
all_targets_db = np.array(all_targets_db)

rmse_db = np.sqrt(np.mean((all_predictions_db - all_targets_db) ** 2))
mae_db = np.mean(np.abs(all_predictions_db - all_targets_db))

errors_db = np.abs(all_predictions_db - all_targets_db)
p50_error = np.percentile(errors_db, 50)
p90_error = np.percentile(errors_db, 90)
p95_error = np.percentile(errors_db, 95)

log.info("-" * 40)
log.info("ACCURACY METRICS (on validation set)")
log.info("-" * 40)
log.info(f"  Samples evaluated: {len(all_predictions_db)}")
log.info(f"  RMSE (dB):         {rmse_db:.2f} dB")
log.info(f"  MAE (dB):          {mae_db:.2f} dB")
log.info(f"  Median error:      {p50_error:.2f} dB")
log.info(f"  90th percentile:   {p90_error:.2f} dB")
log.info(f"  95th percentile:   {p95_error:.2f} dB")

# Speed benchmark
log.info("-" * 40)
log.info("SPEED BENCHMARK")
log.info("-" * 40)

# Get a sample batch for timing
sample_batch = next(iter(val_loader))
sample_features = sample_batch[0].cuda()
sample_elevation = sample_batch[1].cuda()
sample_mask = sample_batch[3].cuda()

# Warm-up GPU
with torch.no_grad():
    for _ in range(10):
        _ = model(sample_features, sample_elevation, mask=sample_mask)

# Time model inference
torch.cuda.synchronize()
num_runs = 100
start_time = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(sample_features, sample_elevation, mask=sample_mask)
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
log.info(f"  Time per sample: {time_per_sample_us:.1f} us")
log.info(f"  Throughput: {samples_per_second:.0f} samples/second")

# Summary
log.info("=" * 60)
log.info("SUMMARY")
log.info("=" * 60)
log.info(f"  Dataset: {nfiles} parquet files")
log.info(f"  Train/Val split: {len(parquet_files_train)}/{len(parquet_files_valid)}")
log.info(f"  Model accuracy: RMSE = {rmse_db:.2f} dB, MAE = {mae_db:.2f} dB")
log.info(f"  Inference speed: {samples_per_second:.0f} predictions/second")
log.info(f"  Time per prediction: {time_per_sample_us:.1f} us")
log.info("=" * 60)

# Save benchmark results
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

benchmark_file = f"./logs/benchmark_{datetimestamp}.json"
with open(benchmark_file, "w") as f:
    json.dump(benchmark_results, f, indent=2)
log.info(f"Benchmark results saved to {benchmark_file}")
