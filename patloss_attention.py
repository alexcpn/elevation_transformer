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

loss_log_file_base = f"./logs/loss_log_{datetimesatmp}_.npy"
loss_log_file = f"./logs/loss_log_{datetimesatmp}_.npy.npz"

# Initialize loss log
if not os.path.exists(loss_log_file):
    np.savez_compressed(loss_log_file, loss=np.array([]))
    log.info(f"Created path loss file {loss_log_file}")

# Target normalization constants (critical for training stability)
# from  analyze_parquet.py
TARGET_MEAN = 218.0                                                                                                                                                     
TARGET_STD = 31.0    

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


def process_batch(df, read_seq_length):
    """
    Convert batch data from Pandas DataFrame to PyTorch tensors.
    Handles variable-length elevation_data by padding/truncating.
    Normalizes features and target for stable training.
    """
    # Feature normalization constants (from analyze_parquet.py)
    FEAT_MEAN = torch.tensor([135920.0, 6300.0, 41.0, 89.0])
    FEAT_STD = torch.tensor([46380.0, 100.0, 150.0, 35.0])

    # Elevation normalization (from analyze_parquet.py)
    ELEV_MEAN = 805.0
    ELEV_STD = 736.0

    # Convert scalar float columns to tensors
    dEP_FSRx_m = torch.tensor(df['dEP_FSRx_m'].values, dtype=torch.float32)
    center_freq = torch.tensor(df['center_freq'].values, dtype=torch.float32)
    receiver_ht_m = torch.tensor(df['receiver_ht_m'].values, dtype=torch.float32)
    accesspoint_ht_m = torch.tensor(df['accesspoint_ht_m'].values, dtype=torch.float32)

    # Convert target labels to tensor and NORMALIZE
    path_loss = torch.tensor(df['path_loss'].values, dtype=torch.float32)
    path_loss = (path_loss - TARGET_MEAN) / (TARGET_STD + 1e-6)

    # Process elevation_data (handling variable sequence lengths)
    elevation_data_list = df['elevation_data'].tolist()  # Convert Series to list of lists

    # Ensure each sequence is exactly `read_seq_length`
    elevation_tensors = []
    for seq in elevation_data_list:
        seq = torch.tensor(seq, dtype=torch.float32)  # Convert to tensor
        if len(seq) > read_seq_length:
            seq = seq[:read_seq_length]  # Truncate if longer
        elif len(seq) < read_seq_length:
            seq = torch.cat([seq, torch.zeros(read_seq_length - len(seq))])  # Pad if shorter
        elevation_tensors.append(seq)

    # Stack to get (batch_size, read_seq_length)
    elevation_data = torch.stack(elevation_tensors)

    # Normalize elevation data
    elevation_data = (elevation_data - ELEV_MEAN) / (ELEV_STD + 1e-6)

    # Stack all other tensors into a single tensor for batch processing
    input_features = torch.stack([dEP_FSRx_m, center_freq, receiver_ht_m, accesspoint_ht_m], dim=1)

    # Normalize input features
    input_features = (input_features - FEAT_MEAN) / (FEAT_STD + 1e-6)

    return input_features, elevation_data, path_loss

# we need to add positional encoding to the input_ids
# Positional encoding is a way to provide the model with information about the position of each token in the sequence.
# This is important because the model has no inherent sense of order in the tokens, since it only sees them as embeddings.
# generated by LLM
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=768):
        super().__init__()

        # Learnable projection: elevation (1 dim) -> d_model dimensions
        self.elevation_projection = nn.Linear(1, d_model)

        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

        # Register as buffer (so it's automatically moved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len) - raw elevation values
        Returns: (batch_size, seq_len, d_model) - embedded elevation + positional encoding
        """
        seq_len = x.size(1)
        # Project elevation to d_model dimensions (learnable embedding)
        x = x.unsqueeze(-1)  # (B, seq_len, 1)
        x = self.elevation_projection(x)  # (B, seq_len, d_model) - learned representation
        # Add positional encoding
        return x + self.pe[:, :seq_len, :]  # (B, seq_len, d_model)
# Instead of doing Multi head sequentially like previous, lets do it in parallel

vocab_size = 2000
d_k = 64  # attention size
read_seq_length = 768
d_model = 512  # embediding size
final_size = 1 # we just want one value
num_heads = 8  
extra_feature_size = 4
# add in the embdeiing part from previous layer
pos_encoding = PositionalEncoding(d_model,read_seq_length)
extra_features_embedding = nn.Linear(extra_feature_size, d_model)
multihead_attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1,batch_first=True)
prediction_layer1 = nn.Linear(d_model, vocab_size)
layer_norm1 = nn.LayerNorm(vocab_size)
prediction_layer2 = nn.Linear(vocab_size, final_size)
layer_norm2 = nn.LayerNorm(final_size)  # last dimension is the vocab size
# Define the loss function
loss_function = nn.SmoothL1Loss() # since we have just regression
# We'll combine these into a simple pipeline
model = nn.ModuleList([pos_encoding,
                      multihead_attention, extra_features_embedding,
                      layer_norm1, layer_norm2,
                      prediction_layer1, prediction_layer2])
# SGD is unstable and hence we use this
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# with higher learning loss is Nan

# Place all in GPU
pos_encoding.to('cuda')
multihead_attention.to('cuda')
extra_features_embedding.to('cuda')
layer_norm1.to('cuda')
layer_norm2.to('cuda')
prediction_layer1.to('cuda')
prediction_layer2.to('cuda')
model.to('cuda')
# NO NEED TO EXECUTE THIS AGAIN ( this need A100, )
log.info("Training model...")

BATCH_SIZE = 30  # 5 GB for 
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


for epoch in range(2):
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

        # project to higher dim
        extra_features_tokens = extra_features_embedding(input_features)  # (B, 512)
        pos_embedded_tokens = pos_encoding(elevation_data)
        score,_ = multihead_attention(pos_embedded_tokens,pos_embedded_tokens,pos_embedded_tokens)
        hidden1 = score + pos_embedded_tokens
        pooled_score = hidden1.mean(dim=1)  # (B, 512)

        # Combine elevation features with extra features (NOW USED!)
        combined = pooled_score + extra_features_tokens  # (B, 512)

        hidden2 = prediction_layer1(combined)  # through linear layer
        hidden2 = layer_norm1(hidden2)
        hidden2 = F.relu(hidden2)  # Add activation
        logits = prediction_layer2(hidden2)
        loss = loss_function(
            logits.squeeze(1),
            target_labels
        )
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        num_batches += 1

        # Clip gradients to avoid erratic jumps
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch == 0 and step == 1:
            log.info(f"Input Features Shape: {input_features.shape}")  # Expected: (BATCH_SIZE, 4)
            log.info(f"Elevation Data Shape: {elevation_data.shape}")  # Expected: (BATCH_SIZE, read_seq_length)
            log.info(f"Path Loss Shape: {path_loss.shape}")  # Expected: (BATCH_SIZE,)
            log.info(f"extra_features_tokens.shape {extra_features_tokens.shape}") #([20, 512])
            log.info(f"Elevation Data Shape: {elevation_data.shape}") # ([20, 768])
            log.info(f"pos_embedded_tokens.shape {pos_embedded_tokens.shape}") #[20, 768, 512])
            log.info(f"score.shape {score.shape}") # [20, 768, 512]
            log.info(f"hidden1.shape {hidden1.shape}")
            log.info(f"pooled_score.shape {pooled_score.shape}")
            log.info(f"hidden2.shape {hidden2.shape}")
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

        # project to higher dim
        extra_features_tokens = extra_features_embedding(input_features)  # (B, 512)
        pos_embedded_tokens = pos_encoding(elevation_data)
        score,_ = multihead_attention(pos_embedded_tokens,pos_embedded_tokens,pos_embedded_tokens)
        hidden1 = score + pos_embedded_tokens
        pooled_score = hidden1.mean(dim=1)  # (B, 512)

        # Combine elevation features with extra features
        combined = pooled_score + extra_features_tokens  # (B, 512)

        hidden2 = prediction_layer1(combined)
        hidden2 = layer_norm1(hidden2)
        hidden2 = F.relu(hidden2)
        logits = prediction_layer2(hidden2)
        loss = loss_function(
            logits.squeeze(1),
            target_labels
        )
        num_valid_batches += 1
        validation_loss += loss.item()
        # Calculate overestimation and underestimation counts
        predictions = logits.squeeze(1).cpu().detach().numpy()
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

        # Forward pass
        extra_features_tokens = extra_features_embedding(input_features)
        pos_embedded_tokens = pos_encoding(elevation_data)
        score, _ = multihead_attention(pos_embedded_tokens, pos_embedded_tokens, pos_embedded_tokens)
        hidden1 = score + pos_embedded_tokens
        pooled_score = hidden1.mean(dim=1)
        combined = pooled_score + extra_features_tokens
        hidden2 = prediction_layer1(combined)
        hidden2 = layer_norm1(hidden2)
        hidden2 = F.relu(hidden2)
        logits = prediction_layer2(hidden2)

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

# Warm-up GPU
with torch.no_grad():
    for _ in range(10):
        extra_features_tokens = extra_features_embedding(input_features)
        pos_embedded_tokens = pos_encoding(elevation_data)
        score, _ = multihead_attention(pos_embedded_tokens, pos_embedded_tokens, pos_embedded_tokens)
        hidden1 = score + pos_embedded_tokens
        pooled_score = hidden1.mean(dim=1)
        combined = pooled_score + extra_features_tokens
        hidden2 = prediction_layer1(combined)
        hidden2 = layer_norm1(hidden2)
        hidden2 = F.relu(hidden2)
        logits = prediction_layer2(hidden2)

# Time model inference
torch.cuda.synchronize()
num_runs = 100
start_time = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        extra_features_tokens = extra_features_embedding(input_features)
        pos_embedded_tokens = pos_encoding(elevation_data)
        score, _ = multihead_attention(pos_embedded_tokens, pos_embedded_tokens, pos_embedded_tokens)
        hidden1 = score + pos_embedded_tokens
        pooled_score = hidden1.mean(dim=1)
        combined = pooled_score + extra_features_tokens
        hidden2 = prediction_layer1(combined)
        hidden2 = layer_norm1(hidden2)
        hidden2 = F.relu(hidden2)
        logits = prediction_layer2(hidden2)
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

