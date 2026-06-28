#!/usr/bin/env python3
"""Upload trained model weights (and training artifacts) to Weights & Biases as a
versioned Artifact.

Prerequisites (run once, interactively):
    pip install wandb
    wandb login                 # paste key from https://wandb.ai/authorize

Typical usage
-------------
Upload the full-run weights with metrics read from the benchmark JSON:

    python3 upload_weights_wandb.py \
        --run-dir /data/elevation_transformer/weights/june282026_full_run \
        --project elevation_transformer

Or point at individual files explicitly:

    python3 upload_weights_wandb.py \
        --weights /data/.../model_weights20260627023536.pth \
        --resume  /data/.../model_weights20260627023536_resume.pth \
        --loss-log /data/.../loss_log_20260627023536_.npy.npz \
        --metrics-json benchmark_20260627023536.json

The weights file is uploaded under the canonical name ``model_inference.pth`` so a
consumer can always pull the same name regardless of the original timestamp:

    run = wandb.init()
    path = run.use_artifact("<project>/<artifact-name>:latest").download()
"""
import argparse
import glob
import json
import os
import sys


def find_one(run_dir, patterns):
    """Return the single newest file in run_dir matching any glob pattern, or None."""
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(run_dir, pat)))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_metrics(metrics_json):
    """Read a benchmark_*.json (as written by train_model.py / benchmark_model.py)
    and map it to artifact metadata. Returns {} if no file given or readable."""
    if not metrics_json or not os.path.exists(metrics_json):
        return {}
    with open(metrics_json) as f:
        data = json.load(f)
    # Accept both the benchmark_model.py keys and the train_model.py summary keys.
    key_map = {
        "rmse": "rmse_db", "rmse_db": "rmse_db",
        "mae": "mae_db", "mae_db": "mae_db",
        "median_error": "median_db", "median_error_db": "median_db",
        "p90_error": "p90_db", "p90_error_db": "p90_db",
        "p95_error": "p95_db", "p95_error_db": "p95_db",
        "val_samples": "val_samples", "batch_size": "batch_size",
        "throughput": "throughput_samples_per_s",
    }
    return {dst: data[src] for src, dst in key_map.items() if src in data}


def main():
    p = argparse.ArgumentParser(description="Upload model weights to W&B as an Artifact")
    p.add_argument("--run-dir", help="Directory holding the run's weights/logs; "
                                     "files are auto-discovered unless overridden")
    p.add_argument("--weights", help="Raw state_dict file for inference (overrides auto-discovery)")
    p.add_argument("--resume", help="Full resume checkpoint (model+optimizer+scheduler)")
    p.add_argument("--loss-log", help="Loss-log .npz file")
    p.add_argument("--train-log", help="Training .log file")
    p.add_argument("--metrics-json", help="benchmark_*.json to attach as artifact metadata")
    p.add_argument("--project", default="elevation_transformer", help="W&B project")
    p.add_argument("--entity", default=None, help="W&B entity (team/user); default = your default")
    p.add_argument("--artifact-name", default="pathloss_transformer", help="Artifact name")
    p.add_argument("--run-name", default=None, help="W&B run name")
    p.add_argument("--alias", action="append", default=[],
                   help="Extra alias(es) for this artifact version, e.g. --alias full-epoch")
    p.add_argument("--dry-run", action="store_true", help="Print what would be uploaded and exit")
    args = p.parse_args()

    # Resolve files: explicit flags win, else auto-discover within --run-dir.
    rd = args.run_dir
    weights = args.weights or (find_one(rd, ["model_weights*[0-9].pth"]) if rd else None)
    resume = args.resume or (find_one(rd, ["*_resume.pth"]) if rd else None)
    loss_log = args.loss_log or (find_one(rd, ["loss_log_*.npz", "loss_log_*.npy.npz"]) if rd else None)
    train_log = args.train_log or (find_one(rd, ["pl_*.log", "*.log"]) if rd else None)
    metrics_json = args.metrics_json or (find_one(rd, ["benchmark_*.json"]) if rd else None)

    if not weights:
        sys.exit("ERROR: no inference weights found. Pass --weights or a --run-dir "
                 "containing a model_weights*.pth file.")

    metadata = load_metrics(metrics_json)

    files = [(weights, "model_inference.pth")]
    if resume:
        files.append((resume, "checkpoint_resume.pth"))
    if loss_log:
        files.append((loss_log, "loss_log.npz"))
    if train_log:
        files.append((train_log, "training.log"))

    print("Artifact :", args.artifact_name, "(type=model)")
    print("Project  :", (args.entity + "/" if args.entity else "") + args.project)
    print("Metadata :", json.dumps(metadata) if metadata else "(none)")
    print("Files    :")
    for src, name in files:
        print(f"  {name:<22} <- {src} ({os.path.getsize(src) / 1e6:.1f} MB)")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    import wandb  # imported late so --dry-run works without wandb installed

    run = wandb.init(project=args.project, entity=args.entity,
                     job_type="upload-weights", name=args.run_name)
    art = wandb.Artifact(name=args.artifact_name, type="model", metadata=metadata)
    for src, name in files:
        art.add_file(src, name=name)
    aliases = ["latest"] + args.alias
    run.log_artifact(art, aliases=aliases)
    run.finish()
    print(f"\nUploaded. Pull with:\n"
          f'  run.use_artifact("{args.project}/{args.artifact_name}:latest").download()')


if __name__ == "__main__":
    main()
