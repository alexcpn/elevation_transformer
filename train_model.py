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
from itertools import chain

from pathloss_transformer import create_model, load_weights
from pathloss_dataset import PathLossDataset

# ============================================================
# CONFIGURATION
# ============================================================

# Set to None to train from scratch, or path to weights file to resume
RESUME_FROM_WEIGHTS = "weights/model_weights20260204165247.pth"

# Weighted loss: upweight hard examples
USE_WEIGHTED_LOSS = False  # Disabled - was causing loss spikes
WEIGHT_SCALE = 0.3

BATCH_SIZE = 64 # for a 6 GB GPU Mempory 64 batch size is fine; Increase it if you have more GPU memory (e.g. 128 for 24 GB GPU), or decrease if you have less memory (e.g. 32 for 4 GB GPU)
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
DROP_LAST = True  # Set False to allow a smaller final batch
LIMIT_TRAIN_SAMPLES = 1000  # Set to None for full training, or a number to limit samples
LIMIT_VAL_SAMPLES = 250000  # Set to None for full validation, or a number to limit (~1% of full dataset)

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


def validate_model(model, val_loader, loss_function, USE_AMP, limit_batches=None):
    model.eval()
    validation_loss = 0.0
    num_valid_batches = 0
    total_val_samples = 0
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
            total_val_samples += targets.size(0)

            preds_np = preds.float().cpu().numpy()
            targets_np = targets.cpu().numpy()
            diff = preds_np - targets_np
            overestimation_count += np.sum(diff > 3.0)   # 3 dB threshold
            underestimation_count += np.sum(diff < -3.0)

            if limit_batches is not None and num_valid_batches >= limit_batches:
                break

    avg_valid_loss = validation_loss / num_valid_batches if num_valid_batches > 0 else 0.0

    return avg_valid_loss, overestimation_count, underestimation_count,total_val_samples

# ============================================================
# DATA SETUP
# ============================================================

# INPUT_DIR = "/data/itm_loss/"
# parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
# random.seed(42)  # For reproducibility
# random.shuffle(parquet_files)  # Shuffle to ensure train/val have similar distributions
# nfiles = len(parquet_files)
# log.info(f"Number of parquet files: {nfiles}")

# # Train/val split by file
# train_ratio = 0.8
# split_idx = int(nfiles * train_ratio)
# parquet_files_train = parquet_files[:split_idx]
# parquet_files_valid = parquet_files[split_idx:]
# log.info(f"Train files: {len(parquet_files_train)}")
# log.info(f"Validation files: {len(parquet_files_valid)}")

# Create datasets from local parquet files
# log.info("Loading training data...")
# train_dataset = PathLossDataset(INPUT_DIR, file_list=parquet_files_train)
# log.info(f"Training samples: {len(train_dataset)}")

# log.info("Loading validation data...")
# val_dataset = PathLossDataset(INPUT_DIR, file_list=parquet_files_valid)
# log.info(f"Validation samples: {len(val_dataset)}")

# Create datasets from Huggingface datasets
train_dataset = PathLossDataset(None, split="train", max_samples=LIMIT_TRAIN_SAMPLES)
val_dataset = PathLossDataset(None, split="val", max_samples=LIMIT_VAL_SAMPLES)

# Helper for rebuilding loaders (useful for fallback if multi-worker streaming stalls)
def build_train_loader(num_workers):
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=DROP_LAST)

# Create data loaders
# Note: For IterableDataset, shuffle must be False in DataLoader (shuffling is done internally)
train_loader = build_train_loader(NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# Calculate estimated steps from dataset stats
estimated_samples = len(train_dataset)
if LIMIT_TRAIN_SAMPLES is not None:
    estimated_samples = min(estimated_samples, LIMIT_TRAIN_SAMPLES)
if DROP_LAST:
    total_steps = estimated_samples // BATCH_SIZE
else:
    total_steps = math.ceil(estimated_samples / BATCH_SIZE)
if total_steps == 0:
    log.warning("Estimated steps per epoch is 0. Check BATCH_SIZE (%d), LIMIT_TRAIN_SAMPLES (%s), and DROP_LAST.",
                BATCH_SIZE, LIMIT_TRAIN_SAMPLES)

log.info(f"Estimated training samples: {estimated_samples}")
log.info(f"Total training steps per epoch (estimated): {total_steps}")

# ============================================================
# MODEL SETUP
# ============================================================

model = create_model()
loss_function = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

model.to('cuda')

# Load pre-trained weights if specified
if RESUME_FROM_WEIGHTS and os.path.exists(RESUME_FROM_WEIGHTS):
    log.info(f"Loading weights from: {RESUME_FROM_WEIGHTS}")
    model = load_weights(model, RESUME_FROM_WEIGHTS)
    log.info("Weights loaded successfully. Resuming training.")
elif RESUME_FROM_WEIGHTS:
    log.warning(f"Weights file not found: {RESUME_FROM_WEIGHTS}. Training from scratch.")

#orch.compile()
log.info("Running without torch.compile (more stable for training).")

# ============================================================
# TRAINING
# ============================================================

log.info("Training model...")
loss_value_list = []
def iter_train_batches(loader):
    train_iter = iter(loader)
    first_batch = next(train_iter, None)
    if first_batch is None:
        return None
    return chain([first_batch], train_iter)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    train_iter = iter_train_batches(train_loader)
    if train_iter is None and train_loader.num_workers > 0:
        log.warning("Train loader produced 0 batches with num_workers=%d. Retrying with num_workers=0.",
                    train_loader.num_workers)
        train_loader = build_train_loader(0)
        train_iter = iter_train_batches(train_loader)
    if train_iter is None:
        log.error("Train loader is empty. Check dataset availability, split, LIMIT_TRAIN_SAMPLES, and DROP_LAST.")
        break

    for step, (features, elevation, targets, mask) in enumerate(train_iter, 1):
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

        print(f"step {step}/{total_steps} - loss: {loss.item():.4f}")
        if epoch == 0 and step == 1:
            log.info(f"Features shape: {features.shape}")
            log.info(f"Elevation shape: {elevation.shape}")
            log.info(f"Targets shape: {targets.shape}")
            log.info(f"Mask shape: {mask.shape}")
            log.info(f"Preds shape: {preds.shape}")

        loss_value_list.append((epoch, step, loss.item()))
        if step % 10 == 0:
            total_samples_processed = num_batches * BATCH_SIZE
            log.info("[Epoch=%d | Step=%d/%d | Samples=%d] loss=%.4f",
                     epoch+1, step, total_steps, total_samples_processed, loss.item())
            # Save loss log
            data = np.load(loss_log_file, allow_pickle=True)
            loss_history = data["loss"].tolist() if "loss" in data else []
            loss_list = loss_history + loss_value_list
            np.savez_compressed(loss_log_file, loss=np.array(loss_list, dtype=object))
            loss_value_list = []
            avg_epoch_loss = epoch_loss / num_batches
            log.info("---------Epoch %02d | Training samples: %d | Average Loss: %.4f", epoch+1, total_samples_processed, avg_epoch_loss)
        

        if step % 500 == 0:
            avg_valid_loss, overestimation_count, underestimation_count, total_val_samples = validate_model(
                model, val_loader, loss_function, USE_AMP, limit_batches=5
            )
            log.info("Epoch=%d | Step=%d/%d |---Validation | Validation samples: %d", epoch+1, step, total_val_samples)
            log.info("---------Validation | Average Validation: %.4f", avg_valid_loss)
            log.info(f"---------Validation | Overestimation (>3dB): {overestimation_count}")
            log.info(f"---------Validation | Underestimation (<-3dB): {underestimation_count}")

        # Check if we've hit the sample limit
        if LIMIT_TRAIN_SAMPLES is not None and num_batches * BATCH_SIZE >= LIMIT_TRAIN_SAMPLES:
            log.info(f"Reached training sample limit ({LIMIT_TRAIN_SAMPLES}), stopping epoch early.")
            break
    log.info(f"Epoch {epoch+1} completed. Average Training Loss: {epoch_loss / num_batches:.4f}")
 
log.info("Training completed. Starting final validation...")
# ============================================================
avg_valid_loss, overestimation_count, underestimation_count, total_val_samples = validate_model(
    model, val_loader, loss_function, USE_AMP, limit_batches=None
)
log.info("Validation | Validation samples: %d", total_val_samples)
log.info("Validation | Average Validation: %.4f", avg_valid_loss)
log.info(f"Validation | Overestimation (>3dB): {overestimation_count}")
log.info(f"Validation | Underestimation (<-3dB): {underestimation_count}")


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
# (Speed benchmark code removed; see benchmark_model.py)


# Summary
log.info("=" * 60)
log.info("SUMMARY")
log.info("=" * 60)
log.info(f"  Dataset: {getattr(train_dataset, 'n_files', 'N/A')} files (approx)")
log.info(f"  Train/Val split: 80% / 10%")
log.info(f"  Model accuracy: RMSE = {rmse_db:.2f} dB, MAE = {mae_db:.2f} dB")

log.info("=" * 60)

# Save benchmark results
benchmark_results = {
    "dataset_size": getattr(train_dataset, 'n_files', 'N/A'),
    "train_files": "80%",
    "val_files": "10%",
    "val_samples": len(all_predictions_db),
    "rmse_db": float(rmse_db),
    "mae_db": float(mae_db),
    "median_error_db": float(p50_error),
    "p90_error_db": float(p90_error),
    "p95_error_db": float(p95_error),

    "batch_size": BATCH_SIZE,
}

benchmark_file = f"./logs/benchmark_{datetimestamp}.json"
with open(benchmark_file, "w") as f:
    json.dump(benchmark_results, f, indent=2)
log.info(f"Benchmark results saved to {benchmark_file}")
