import argparse
from dataclasses import dataclass
from datetime import datetime
import glob
import json
import logging as log
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from itm import itmlosswinnf as itmloss  # noqa: E402


log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


@dataclass
class ITMSample:
    frequency_mhz: float
    accesspoint_ht_m: float
    receiver_ht_m: float
    distance_m: float
    elevation_profile_m: list[float]
    target_loss_db: float


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark direct ITM inference")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="itm_loss_test",
        help="Directory containing parquet files with raw ITM samples",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to evaluate for accuracy metrics",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of serial ITM calls grouped into one timed batch",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of timed speed benchmark runs",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=2,
        help="Number of untimed warm-up runs before measuring speed",
    )
    return parser.parse_args()


def load_samples(data_dir, max_samples):
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    log.info(f"Loading direct ITM benchmark data from {data_dir}")

    columns = [
        "center_freq_mhz",
        "accesspoint_ht_m",
        "receiver_ht_m",
        "distance_to_ap_m",
        "itm_loss_db",
        "elevation_profile_m",
    ]

    samples = []
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path, columns=columns)
        for row in df.itertuples(index=False):
            samples.append(
                ITMSample(
                    frequency_mhz=float(row.center_freq_mhz),
                    accesspoint_ht_m=float(row.accesspoint_ht_m),
                    receiver_ht_m=float(row.receiver_ht_m),
                    distance_m=float(row.distance_to_ap_m),
                    elevation_profile_m=np.asarray(row.elevation_profile_m, dtype=np.float64).tolist(),
                    target_loss_db=float(row.itm_loss_db),
                )
            )
            if max_samples and len(samples) >= max_samples:
                log.info(f"Loaded {len(samples)} samples from {len(parquet_files)} parquet files")
                return samples

    log.info(f"Loaded {len(samples)} samples from {len(parquet_files)} parquet files")
    return samples


def run_itm_batch(itm_engine, batch_samples):
    predictions = []
    for sample in batch_samples:
        predictions.append(
            itm_engine.getItmLoss(
                sample.frequency_mhz,
                sample.accesspoint_ht_m,
                sample.receiver_ht_m,
                sample.elevation_profile_m,
                sample.distance_m,
            )
        )
    return predictions


def main():
    args = parse_args()
    samples = load_samples(args.data_dir, args.max_samples)
    if not samples:
        raise RuntimeError("No benchmark samples loaded")

    batch_size = min(args.batch_size, len(samples))
    timed_batch = samples[:batch_size]
    itm_engine = itmloss.ITMLossWinnf()

    log.info("Running direct ITM accuracy benchmark...")
    start_time = time.perf_counter()
    predictions = run_itm_batch(itm_engine, samples)
    end_time = time.perf_counter()

    targets = np.array([sample.target_loss_db for sample in samples], dtype=np.float64)
    predictions = np.array(predictions, dtype=np.float64)
    accuracy_total_time = end_time - start_time

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    errors = np.abs(predictions - targets)
    p50_error = np.percentile(errors, 50)
    p90_error = np.percentile(errors, 90)
    p95_error = np.percentile(errors, 95)
    accuracy_samples_per_second = len(predictions) / accuracy_total_time

    log.info("=" * 60)
    log.info("BENCHMARK RESULTS")
    log.info("=" * 60)
    log.info("Engine: direct ITM (itm.ITMLossWinnf.getItmLoss)")
    log.info(f"Samples: {len(predictions)}")
    log.info(f"RMSE: {rmse:.6f}")
    log.info(f"MAE: {mae:.6f}")
    log.info(f"Median Error: {p50_error:.6f}")
    log.info(f"90th Percentile: {p90_error:.6f}")
    log.info(f"95th Percentile: {p95_error:.6f}")
    log.info(f"Throughput: {accuracy_samples_per_second:.0f} samples/sec")
    log.info("=" * 60)

    log.info("-" * 40)
    log.info("SPEED BENCHMARK")
    log.info("-" * 40)

    for _ in range(args.warmup_runs):
        run_itm_batch(itm_engine, timed_batch)

    start_time = time.perf_counter()
    for _ in range(args.runs):
        run_itm_batch(itm_engine, timed_batch)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    time_per_batch = total_time / args.runs
    samples_per_second = (batch_size * args.runs) / total_time
    time_per_sample_us = (total_time / (batch_size * args.runs)) * 1e6

    log.info(f"  Batch size: {batch_size}")
    log.info(f"  Runs: {args.runs}")
    log.info(f"  Total time: {total_time:.3f} s")
    log.info(f"  Time per batch: {time_per_batch * 1000:.2f} ms")
    log.info(f"  Time per sample: {time_per_sample_us:.1f} us")
    log.info(f"  Throughput: {samples_per_second:.0f} samples/second")
    log.info(f"  Inference speed: {samples_per_second:.0f} predictions/second")
    log.info(f"  Time per prediction: {time_per_sample_us:.1f} us")

    results = {
        "engine": "direct ITM (itm.ITMLossWinnf.getItmLoss)",
        "data_dir": os.path.abspath(args.data_dir),
        "samples": int(len(predictions)),
        "rmse": float(rmse),
        "mae": float(mae),
        "median_error": float(p50_error),
        "p90_error": float(p90_error),
        "p95_error": float(p95_error),
        "accuracy_throughput": float(accuracy_samples_per_second),
        "batch_size": int(batch_size),
        "runs": int(args.runs),
        "warmup_runs": int(args.warmup_runs),
        "speed_total_time_s": float(total_time),
        "speed_time_per_batch_ms": float(time_per_batch * 1000.0),
        "speed_time_per_sample_us": float(time_per_sample_us),
        "speed_throughput": float(samples_per_second),
    }

    output_file = f"benchmark_itm_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    log.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
