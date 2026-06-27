"""
Path loss transformer training script.
Uses PathLossDataset with DataLoader for proper shuffling and masking.
Elevation and target (ITM loss dB) are normalized; distance/frequency are log10-scaled.

How to run
----------
  # Train from scratch, streaming the dataset from Hugging Face (default):
  python3 train_model.py

  # Override batch size (e.g. on a larger runpod GPU):
  python3 train_model.py --batch-size 128

  # Train from local parquet files instead of streaming:
  python3 train_model.py --input-dir /data/itm_loss

  # Combined:
  python3 train_model.py --input-dir /data/itm_loss --batch-size 128

Download the dataset locally (optional, avoids re-streaming over the network each epoch)
----------------------------------------------------------------------------------------
  
  # Download every parquet shard into /data/itm_loss (~tens of GB; pick any local dir):
  hf download \
      alexcpn/longely_rice_model \
      --repo-type dataset \
      --local-dir /data/itm_loss \
      --include "*.parquet"

  # Then point training at it:
  python3 train_model.py --input-dir /data/itm_loss

Options:
  --batch-size INT   Training/validation batch size (default: BATCH_SIZE below, 64)
  --input-dir  STR   Local parquet directory. Default None => stream from
                     Hugging Face (alexcpn/longely_rice_model).

Other settings (epochs, learning rate, sample limits, resume weights) are the
constants in the CONFIGURATION section below.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging as log
import os
import math
import json
from datetime import datetime
from torch.utils.data import DataLoader
from itertools import chain
from collections import deque

from pathloss_transformer import create_model, load_weights, denormalize_target, TARGET_STD
from pathloss_dataset import PathLossDataset
# Remove HugingFace Dataset sreaming logs

import logging
from datasets.utils.logging import disable_progress_bar, set_verbosity_error

disable_progress_bar()
set_verbosity_error()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


# ============================================================
# CONFIGURATION
# ============================================================

# Set to None to train from scratch, or path to weights file to resume
RESUME_FROM_WEIGHTS = "weights/model_weights20260204165247.pth" # trained in my laptio
RESUME_FROM_WEIGHTS = "weights/model_weights20260205140023.pth" #trained in Runpod from above
RESUME_FROM_WEIGHTS = None # Train from scratch
# NOTE: --resume-weights on the command line overrides this; applied just after argparse below.

# Weighted loss: upweight hard examples
USE_WEIGHTED_LOSS = False  # Disabled - was causing loss spikes
WEIGHT_SCALE = 0.3

BATCH_SIZE = 64 # for a 6 GB GPU Mempory 64 batch size is fine; Increase it if you have more GPU memory (e.g. 128 for 24 GB GPU), or decrease if you have less memory (e.g. 32 for 4 GB GPU)

# Allow overriding batch size from the command line (handy on runpod), defaulting to BATCH_SIZE above
parser = argparse.ArgumentParser(description="Train the path loss transformer")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"Training/validation batch size (default: {BATCH_SIZE})")
parser.add_argument("--input-dir", type=str, default=None,
                    help="Directory of local parquet files. Default None = stream from "
                         "Hugging Face (alexcpn/longely_rice_model)")
parser.add_argument("--resume-step", type=int, default=0,
                    help="Resume after this many completed training steps: skips "
                         "resume_step*batch_size samples in the deterministic (seed=42) stream "
                         "and continues the step counter. Pair with the matching checkpoint in "
                         "RESUME_FROM_WEIGHTS. Requires NUM_WORKERS<=1 for the skip to line up.")
parser.add_argument("--loss", type=str, default="smoothl1", choices=["smoothl1", "mse"],
                    help="Regression loss. Default smoothl1 (robust, chosen for the high "
                         "dynamic range). 'mse' directly optimizes RMSE but is outlier-sensitive "
                         "- prefer a fresh run when comparing, not a mid-resume switch.")
parser.add_argument("--resume-weights", type=str, default=None,
                    help="Path to a checkpoint to resume from; overrides the RESUME_FROM_WEIGHTS "
                         "set in the script. Accepts both full-state and legacy raw-state_dict files.")
args = parser.parse_args()
BATCH_SIZE = args.batch_size
INPUT_DIR = args.input_dir
RESUME_STEP = args.resume_step
LOSS_NAME = args.loss
# --resume-weights overrides the RESUME_FROM_WEIGHTS set above (RESUME_FROM_WEIGHTS is defined
# earlier in the file, so this assignment is the effective one when the flag is passed).
if args.resume_weights is not None:
    RESUME_FROM_WEIGHTS = args.resume_weights
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
# NOTE: keep at 1 (or 0). With NUM_WORKERS>1 the manual ds.shard() in
# PathLossDataset.__iter__ only delivers one worker's shard, so a "full epoch" silently
# trains on ~1/NUM_WORKERS of the data (e.g. 4 workers -> ~1/4). TODO: fix the multi-worker
# shard path before raising this for data-loading throughput.
NUM_WORKERS = 1
DROP_LAST = True  # Set False to allow a smaller final batch
CHECKPOINT_EVERY = 1000  # Save a (single, rotated) checkpoint to the current dir every N steps

# Early stopping: exit once the loss is very small and stable. Uses a moving average over the
# last EARLY_STOP_PATIENCE steps (not a single noisy batch). The loss is normalized SmoothL1;
# very roughly RMSE(dB) ~= sqrt(2*loss)*TARGET_STD, so 0.001 ~ 1.5 dB. Set to None to disable.
EARLY_STOP_LOSS = 0.001
EARLY_STOP_PATIENCE = 100

# total rows: 32314577
LIMIT_TRAIN_SAMPLES = None  # Full training over the whole dataset (source-shuffled, unbiased)
LIMIT_VAL_SAMPLES = None  # Set to None for full validation, or a number to limit (~1% of full dataset)

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
            # preds/targets are normalized; scale the difference back to dB for the 3 dB threshold
            diff = (preds_np - targets_np) * TARGET_STD
            overestimation_count += np.sum(diff > 3.0)   # 3 dB threshold
            underestimation_count += np.sum(diff < -3.0)

            if limit_batches is not None and num_valid_batches >= limit_batches:
                break

    avg_valid_loss = validation_loss / num_valid_batches if num_valid_batches > 0 else 0.0

    return avg_valid_loss, overestimation_count, underestimation_count,total_val_samples

# ============================================================
# DATA SETUP
# # ============================================================


# INPUT_DIR comes from --input-dir (default None = stream from Hugging Face).
#
# If you want to download the dataset from Huggingface, uncomment the following lines and run them in your terminal. Make sure you have the `huggingface_hub` package installed and are logged in with `huggingface-cli login`.
# huggingface-cli download \
#   alexcpn/longely_rice_model \
#   --repo-type dataset \
#   --local-dir /data/itm_loss \
#   --include "*.parquet"
# then run:  python3 train_model.py --input-dir /data/itm_loss
log.info(f"INPUT_DIR: {INPUT_DIR if INPUT_DIR else 'None (Hugging Face streaming)'}")
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
train_dataset = PathLossDataset(INPUT_DIR, split="train", max_samples=LIMIT_TRAIN_SAMPLES,
                                skip_samples=RESUME_STEP * BATCH_SIZE)
val_dataset = PathLossDataset(INPUT_DIR, split="val", max_samples=LIMIT_VAL_SAMPLES)

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
loss_function = nn.MSELoss() if LOSS_NAME == "mse" else nn.SmoothL1Loss()
log.info(f"Loss function: {LOSS_NAME}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

# Cosine LR decay over the full epoch (eta_min keeps a small floor so late steps still learn).
# Resumable: its state is saved in the checkpoint and restored below; for legacy checkpoints we
# fast-forward it to the resume step so the LR matches where training continues.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(total_steps, 1), eta_min=LEARNING_RATE * 0.05
)

model.to('cuda')

# Load pre-trained weights if specified. Supports two checkpoint formats:
#   - full-state dict {model, optimizer, scaler, scheduler, step}  (new, fully resumable)
#   - raw model state_dict                                          (legacy, weights only)
if RESUME_FROM_WEIGHTS and os.path.exists(RESUME_FROM_WEIGHTS):
    log.info(f"Loading checkpoint: {RESUME_FROM_WEIGHTS}")
    ckpt = torch.load(RESUME_FROM_WEIGHTS, weights_only=False)
    restored_scheduler = False
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            restored_scheduler = True
        log.info(f"Full training state restored (checkpoint step={ckpt.get('step', '?')}).")
    else:
        # Legacy raw state_dict: model only; optimizer/scaler/scheduler start fresh.
        model.load_state_dict(ckpt)
        log.info("Legacy weights-only checkpoint loaded (no optimizer/scheduler state).")
    # Align the LR schedule to the resume point when it wasn't restored from the checkpoint
    # (legacy checkpoint, or a checkpoint saved before the scheduler existed).
    if RESUME_STEP > 0 and not restored_scheduler:
        for _ in range(min(RESUME_STEP, total_steps)):
            scheduler.step()
        log.info(f"Fast-forwarded LR scheduler to step {RESUME_STEP} "
                 f"(lr={scheduler.get_last_lr()[0]:.2e}).")
    log.info("Resuming training.")
elif RESUME_FROM_WEIGHTS:
    log.warning(f"Weights file not found: {RESUME_FROM_WEIGHTS}. Training from scratch.")


def save_checkpoint(path, step):
    """Atomically save a fully-resumable checkpoint (model + optimizer + scaler + scheduler)."""
    tmp_path = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic rename on same filesystem

#orch.compile()
log.info("Running without torch.compile (more stable for training).")

# ============================================================
# TRAINING
# ============================================================

log.info("Training model...")
loss_value_list = []
step = RESUME_STEP  # ensure defined even if the training loop yields no batches
def iter_train_batches(loader):
    train_iter = iter(loader)
    first_batch = next(train_iter, None)
    if first_batch is None:
        return None
    return chain([first_batch], train_iter)

stop_training = False
recent_losses = deque(maxlen=EARLY_STOP_PATIENCE)  # moving window for early stopping

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

    for local_step, (features, elevation, targets, mask) in enumerate(train_iter, 1):
        # Global step continues from where a resumed run left off (RESUME_STEP), so the
        # displayed step / checkpoint counters stay aligned with the full-epoch total_steps.
        step = RESUME_STEP + local_step
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
        scheduler.step()  # per-step cosine LR decay (resumable via checkpoint)

        epoch_loss += loss.item()
        num_batches += 1

        print(f"step {step}/{total_steps} - loss: {loss.item():.4f}")
        if epoch == 0 and local_step == 1:
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
            log.info("Epoch=%d | Step=%d/%d |---Validation | Validation samples: %d", epoch+1, step, total_steps, total_val_samples)
            log.info("---------Validation | Average Validation: %.4f", avg_valid_loss)
            log.info(f"---------Validation | Overestimation (>3dB): {overestimation_count}")
            log.info(f"---------Validation | Underestimation (<-3dB): {underestimation_count}")

        # Periodic checkpoint to the current directory (crash safety on long runs).
        # Keeps only ONE rotated file (overwrites each time). Atomic: write to a temp file then
        # os.replace, so a crash mid-save can't corrupt the existing good checkpoint.
        # Resume from this via RESUME_FROM_WEIGHTS. Saved in CWD as requested.
        if CHECKPOINT_EVERY and step % CHECKPOINT_EVERY == 0:
            latest_path = f"./weights/model_weights{datetimestamp}_latest.pth"
            save_checkpoint(latest_path, step)
            log.info(f"Checkpoint saved (rotated): {latest_path} @ step {step}")

        # Early stopping: exit when the moving-average loss is very small (stable, not a single
        # lucky batch). Saves a final rotated checkpoint before leaving the loop.
        if EARLY_STOP_LOSS is not None:
            recent_losses.append(loss.item())
            if len(recent_losses) == recent_losses.maxlen:
                avg_recent = sum(recent_losses) / len(recent_losses)
                if avg_recent < EARLY_STOP_LOSS:
                    log.info("Early stop: avg loss over last %d steps = %.6f < %g (step %d).",
                             EARLY_STOP_PATIENCE, avg_recent, EARLY_STOP_LOSS, step)
                    latest_path = f"./weights/model_weights{datetimestamp}_latest.pth"
                    save_checkpoint(latest_path, step)
                    log.info(f"Checkpoint saved (rotated): {latest_path} @ step {step}")
                    stop_training = True
                    break

        # Check if we've hit the sample limit
        if LIMIT_TRAIN_SAMPLES is not None and num_batches * BATCH_SIZE >= LIMIT_TRAIN_SAMPLES:
            log.info(f"Reached training sample limit ({LIMIT_TRAIN_SAMPLES}), stopping epoch early.")
            break
    log.info(f"Epoch {epoch+1} completed. Average Training Loss: {epoch_loss / num_batches:.4f}")
    if stop_training:
        break
 
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
torch.save(model.state_dict(), save_path)  # raw state_dict: for inference/benchmark/load_weights
log.info(f"Model weights saved at {save_path}")
# Also write a fully-resumable checkpoint so training can be continued from the end if needed.
resume_path = f"./weights/model_weights{datetimestamp}_resume.pth"
save_checkpoint(resume_path, step)
log.info(f"Resumable checkpoint saved at {resume_path} @ step {step}")
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

        # Model/targets are in normalized units -> denormalize back to dB for metrics
        all_predictions_db.extend(denormalize_target(preds.float().cpu().numpy()))
        all_targets_db.extend(denormalize_target(targets.numpy()))

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
log.info(f"Loss log file: {loss_log_file}")
