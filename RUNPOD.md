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

```bash
# Run training
python train.py

# View training loss
python utils/loss_viewer.py
```
