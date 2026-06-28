#!/usr/bin/env python3
"""Upload trained model weights (and training artifacts) to a Hugging Face model repo.

Prerequisites (run once, interactively):
    pip install huggingface_hub
    huggingface-cli login        # paste a write token from https://huggingface.co/settings/tokens

Typical usage
-------------
    python3 upload_weights_hf.py \
        --run-dir /data/elevation_transformer/weights/june282026_full_run \
        --repo-id alexcpn/elevation_transformer

Files are auto-discovered from --run-dir; the raw state_dict keeps its original
timestamped name (provenance) and is *also* uploaded as ``model_inference.pth`` so a
consumer can always pull the same canonical name:

    from huggingface_hub import hf_hub_download
    path = hf_hub_download("alexcpn/elevation_transformer", "model_inference.pth")
"""
import argparse
import glob
import os
import sys


def find_one(run_dir, patterns):
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(run_dir, pat)))
    return max(matches, key=os.path.getmtime) if matches else None


def main():
    p = argparse.ArgumentParser(description="Upload model weights to a Hugging Face repo")
    p.add_argument("--run-dir", help="Directory holding the run's weights/logs (auto-discovered)")
    p.add_argument("--weights", help="Raw state_dict for inference (overrides auto-discovery)")
    p.add_argument("--resume", help="Full resume checkpoint (model+optimizer+scheduler)")
    p.add_argument("--loss-log", help="Loss-log .npz file")
    p.add_argument("--metrics-json", help="benchmark_*.json to upload alongside the weights")
    p.add_argument("--repo-id", default="alexcpn/elevation_transformer", help="HF repo id")
    p.add_argument("--subdir", default="", help="Optional path prefix inside the repo, e.g. 'weights'")
    p.add_argument("--commit-message", default=None)
    p.add_argument("--no-canonical", action="store_true",
                   help="Do not also upload the inference weights as model_inference.pth")
    p.add_argument("--dry-run", action="store_true", help="Print what would be uploaded and exit")
    args = p.parse_args()

    rd = args.run_dir
    weights = args.weights or (find_one(rd, ["model_weights*[0-9].pth"]) if rd else None)
    resume = args.resume or (find_one(rd, ["*_resume.pth"]) if rd else None)
    loss_log = args.loss_log or (find_one(rd, ["loss_log_*.npz", "loss_log_*.npy.npz"]) if rd else None)
    metrics_json = args.metrics_json or (find_one(rd, ["benchmark_*.json"]) if rd else None)

    if not weights:
        sys.exit("ERROR: no inference weights found. Pass --weights or a --run-dir "
                 "containing a model_weights*.pth file.")

    def dst(name):
        return f"{args.subdir.rstrip('/')}/{name}" if args.subdir else name

    # (local_path, path_in_repo)
    uploads = [(weights, dst(os.path.basename(weights)))]
    if not args.no_canonical:
        uploads.append((weights, dst("model_inference.pth")))
    if resume:
        uploads.append((resume, dst(os.path.basename(resume))))
    if loss_log:
        uploads.append((loss_log, dst(os.path.basename(loss_log))))
    if metrics_json:
        uploads.append((metrics_json, dst(os.path.basename(metrics_json))))

    print("Repo  :", args.repo_id)
    print("Files :")
    for src, path_in_repo in uploads:
        print(f"  {path_in_repo:<34} <- {src} ({os.path.getsize(src) / 1e6:.1f} MB)")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi  # late import so --dry-run needs no install
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    msg = args.commit_message or f"Add full-run weights ({os.path.basename(weights)})"
    # De-duplicate identical local files (weights uploaded twice would otherwise re-hash).
    for src, path_in_repo in uploads:
        api.upload_file(path_or_fileobj=src, path_in_repo=path_in_repo,
                        repo_id=args.repo_id, repo_type="model", commit_message=msg)
        print(f"  uploaded {path_in_repo}")
    print(f"\nDone: https://huggingface.co/{args.repo_id}/tree/main")


if __name__ == "__main__":
    main()
