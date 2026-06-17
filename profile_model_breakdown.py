import argparse
import glob
import json
import logging as log
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from pathloss_transformer import create_model, load_weights


log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile CPU->GPU transfer time versus model forward time"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/model_weights20260205140023.pth",
        help="Path to model weights file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="itm_loss_test",
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size to profile",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=750,
        help="Pad/truncate elevation sequences to this length",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of timed runs",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=10,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin CPU tensors before transfer to CUDA",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile for this profiling run",
    )
    return parser.parse_args()


def build_batch(data_dir, batch_size, seq_length):
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    columns = [
        "distance_to_ap_m",
        "center_freq_mhz",
        "receiver_ht_m",
        "accesspoint_ht_m",
        "elevation_profile_m",
    ]

    feature_rows = []
    elevation_rows = []
    mask_rows = []

    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path, columns=columns)
        for row in df.itertuples(index=False):
            elev = torch.as_tensor(row.elevation_profile_m, dtype=torch.float32)
            elev_len = min(len(elev), seq_length)

            elevation = torch.zeros(seq_length, dtype=torch.float32)
            mask = torch.zeros(seq_length, dtype=torch.bool)
            mask[elev_len:] = True

            mean = 0.0
            std = 0.0
            if elev_len > 0:
                cropped = elev[:elev_len]
                mean = cropped.mean()
                std = cropped.std()
                elevation[:elev_len] = (cropped - mean) / (std + 1e-6)

            features = torch.tensor(
                [
                    np.log10(float(row.distance_to_ap_m)),  # path loss is logarithmic in distance
                    np.log10(float(row.center_freq_mhz)),   # path loss is logarithmic in frequency
                    float(row.receiver_ht_m) / 100.0,
                    float(row.accesspoint_ht_m) / 100.0,
                    float(mean) / 1000.0,                   # absolute elevation level (km)
                    float(std) / 1000.0,                    # terrain roughness (km)
                ],
                dtype=torch.float32,
            )

            feature_rows.append(features)
            elevation_rows.append(elevation)
            mask_rows.append(mask)

            if len(feature_rows) >= batch_size:
                return (
                    torch.stack(feature_rows, dim=0),
                    torch.stack(elevation_rows, dim=0),
                    torch.stack(mask_rows, dim=0),
                )

    raise RuntimeError(
        f"Only found {len(feature_rows)} samples in {data_dir}, fewer than batch_size={batch_size}"
    )


def tensor_bytes(*tensors):
    return sum(t.nelement() * t.element_size() for t in tensors)


def format_us_per_sample(total_s, runs, batch_size):
    return (total_s / (runs * batch_size)) * 1e6


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script on the GPU machine.")

    log.info("Building CPU batch from parquet data...")
    features_cpu, elevation_cpu, mask_cpu = build_batch(
        args.data_dir, args.batch_size, args.seq_length
    )

    if args.pin_memory:
        features_cpu = features_cpu.pin_memory()
        elevation_cpu = elevation_cpu.pin_memory()
        mask_cpu = mask_cpu.pin_memory()

    batch_bytes = tensor_bytes(features_cpu, elevation_cpu, mask_cpu)
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Sequence length: {args.seq_length}")
    log.info(f"Tensor payload: {batch_bytes / 1024:.1f} KiB")
    log.info(f"Pinned memory: {args.pin_memory}")

    log.info("Loading model...")
    model = create_model()
    load_weights(model, args.weights)
    model = model.to("cuda")
    model.eval()

    if hasattr(torch, "compile") and not args.no_compile:
        log.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    non_blocking = args.pin_memory

    for _ in range(args.warmup_runs):
        features_gpu = features_cpu.to("cuda", non_blocking=non_blocking)
        elevation_gpu = elevation_cpu.to("cuda", non_blocking=non_blocking)
        mask_gpu = mask_cpu.to("cuda", non_blocking=non_blocking)
        with torch.no_grad():
            _ = model(features_gpu, elevation_gpu, mask=mask_gpu)
    torch.cuda.synchronize()

    log.info("Profiling CPU->GPU copy only...")
    start_time = time.perf_counter()
    for _ in range(args.runs):
        features_gpu = features_cpu.to("cuda", non_blocking=non_blocking)
        elevation_gpu = elevation_cpu.to("cuda", non_blocking=non_blocking)
        mask_gpu = mask_cpu.to("cuda", non_blocking=non_blocking)
        torch.cuda.synchronize()
    copy_total_s = time.perf_counter() - start_time

    features_gpu = features_cpu.to("cuda", non_blocking=non_blocking)
    elevation_gpu = elevation_cpu.to("cuda", non_blocking=non_blocking)
    mask_gpu = mask_cpu.to("cuda", non_blocking=non_blocking)
    torch.cuda.synchronize()

    log.info("Profiling model forward only...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.runs):
            _ = model(features_gpu, elevation_gpu, mask=mask_gpu)
            torch.cuda.synchronize()
    forward_total_s = time.perf_counter() - start_time

    log.info("Profiling copy + forward end-to-end...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.runs):
            features_gpu = features_cpu.to("cuda", non_blocking=non_blocking)
            elevation_gpu = elevation_cpu.to("cuda", non_blocking=non_blocking)
            mask_gpu = mask_cpu.to("cuda", non_blocking=non_blocking)
            _ = model(features_gpu, elevation_gpu, mask=mask_gpu)
            torch.cuda.synchronize()
    end_to_end_total_s = time.perf_counter() - start_time

    copy_us = format_us_per_sample(copy_total_s, args.runs, args.batch_size)
    forward_us = format_us_per_sample(forward_total_s, args.runs, args.batch_size)
    end_to_end_us = format_us_per_sample(end_to_end_total_s, args.runs, args.batch_size)

    copy_fraction = copy_total_s / end_to_end_total_s if end_to_end_total_s else 0.0
    forward_fraction = forward_total_s / end_to_end_total_s if end_to_end_total_s else 0.0

    log.info("-" * 40)
    log.info("BREAKDOWN")
    log.info("-" * 40)
    log.info(f"  Copy only total: {copy_total_s:.3f} s")
    log.info(f"  Copy only per sample: {copy_us:.1f} us")
    log.info(f"  Forward only total: {forward_total_s:.3f} s")
    log.info(f"  Forward only per sample: {forward_us:.1f} us")
    log.info(f"  End-to-end total: {end_to_end_total_s:.3f} s")
    log.info(f"  End-to-end per sample: {end_to_end_us:.1f} us")
    log.info(f"  Copy share of end-to-end: {copy_fraction * 100:.1f}%")
    log.info(f"  Forward share of end-to-end: {forward_fraction * 100:.1f}%")

    results = {
        "weights": os.path.abspath(args.weights),
        "data_dir": os.path.abspath(args.data_dir),
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "pin_memory": args.pin_memory,
        "compiled": hasattr(torch, "compile") and not args.no_compile,
        "batch_kib": batch_bytes / 1024.0,
        "copy_only_total_s": copy_total_s,
        "copy_only_per_sample_us": copy_us,
        "forward_only_total_s": forward_total_s,
        "forward_only_per_sample_us": forward_us,
        "end_to_end_total_s": end_to_end_total_s,
        "end_to_end_per_sample_us": end_to_end_us,
        "copy_fraction": copy_fraction,
        "forward_fraction": forward_fraction,
    }

    output_file = f"profile_model_breakdown_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    log.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
