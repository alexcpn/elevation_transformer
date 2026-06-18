import torch
import numpy as np
import os
import glob
import json
import time
import argparse
import logging as log
from datetime import datetime
from pathloss_transformer import create_model, load_weights, denormalize_target
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
BATCH_SIZE = 64

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark PathLoss Transformer")
    parser.add_argument("--weights", type=str, default="weights/model_weights20260205140023.pth", help="Path to model weights file")
    parser.add_argument("--batch_size", type=int, default=384, help="Batch size for inference")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory of local parquet files. Default None = stream from "
                             "Hugging Face (alexcpn/longely_rice_model)")
    parser.add_argument("--max-samples", type=int, default=20000, help="Number of validation samples to evaluate")
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

    # Prepare validation data: local parquet dir if --input-dir given, else stream from HF.
    # The val split (modular index) is applied inside PathLossDataset for both paths.
    log.info(f"Validation source: {args.input_dir if args.input_dir else 'None (Hugging Face streaming)'}")
    val_dataset = PathLossDataset(args.input_dir, split="val", max_samples=args.max_samples)
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
            # Model/targets are in normalized units -> denormalize back to dB for metrics
            preds_db = denormalize_target(logits.cpu().numpy())
            targets_db = denormalize_target(target_labels.cpu().numpy())

            all_predictions.extend(preds_db)
            all_targets.extend(targets_db)


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
    log.info("Timing forward pass only on preloaded CUDA tensors (excludes CPU->GPU transfer).")
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



    # Summary    
    log.info(f"  Batch size: {BATCH_SIZE}")
    log.info(f"  Runs: {num_runs}")
    log.info(f"  Total time: {total_time:.3f} s")
    log.info(f"  Time per batch: {time_per_batch*1000:.2f} ms")
    log.info(f"  Time per sample: {time_per_sample_us:.1f} us")
    log.info(f"  Throughput: {samples_per_second:.0f} samples/second")
    log.info(f"  Inference speed: {samples_per_second:.0f} predictions/second")
    log.info(f"  Time per prediction: {time_per_sample_us:.1f} us")

if __name__ == "__main__":
    main()
