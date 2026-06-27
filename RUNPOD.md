# Running on RunPod

## Quick Setup with uv

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies from pyproject.toml
uv pip install -e .

# Or install specific packages
uv pip install datasets pandas 
```

## Alternative: pip install

```bash
pip install datasets pandas 
```

## GPU Compatibility

### If you hit: "CUDA error: no kernel image is available for execution on the device"

Check GPU + torch build compatibility:

```bash
nvidia-smi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("archs", torch.cuda.get_arch_list())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
PY
```

### Reinstall torch for your GPU

**CUDA 12.8 build (for RTX 5090 / Blackwell sm_120):**

```bash
uv pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**CUDA 11.8 build:**

```bash
uv pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> Note: RTX 5090 (sm_120) requires CUDA 12.8+ builds. PyTorch staff explicitly call out CUDA 12.8+ for Blackwell GPUs.

## Training

### Download the dataset locally

Download every parquet shard into `/data/itm_loss` (~tens of GB; pick any local
directory with enough disk space):

```bash
hf download \
  alexcpn/longely_rice_model \
  --repo-type dataset \
  --local-dir /data/itm_loss \
  --include "*.parquet"
```

Then point training at it:

```bash
python train_model.py --batch-size 320 --input-dir /workspace/itm_loss/ \
    --resume-weights ./weights/model_weights20260626XXXXXX_latest.pth \
    --resume-step 27000
```

### Run over SSH with tmux

Use `tmux` when you want to reconnect and watch progress interactively:

```bash
cd /workspace/elevation_transformer
tmux new -s elevation-train
python3 train_model.py --input-dir /data/itm_loss --batch-size 64
```

Detach without stopping training with `Ctrl-b`, then `d`.

Reconnect later:

```bash
tmux attach -t elevation-train
```

Useful session commands:

```bash
tmux ls
tmux kill-session -t elevation-train
```

### Run over SSH with nohup

Use `nohup` when you just want training to keep running after SSH disconnects:

```bash
cd /workspace/elevation_transformer
nohup python3 train_model.py --input-dir /data/itm_loss --batch-size 64 > train.log 2>&1 &
```

Check progress:

```bash
tail -f train.log
```

Check whether it is still running:

```bash
ps -ef | grep train_model.py | grep -v grep
```

Stop it if needed:

```bash
pkill -f "python3 train_model.py"
```
