
To run in runport

pip install datasets
pip install pandas

python -m pip install -U filelock huggingface_hub datasets

# If you hit: "CUDA error: no kernel image is available for execution on the device"
# Check GPU + torch build compatibility
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("archs", torch.cuda.get_arch_list())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
PY

# Reinstall a torch build that supports your GPU (pick one that matches your driver)
# CUDA 12.8 build

You’re on an RTX 5090 (sm_120). Your current build is torch 2.1.0+cu118, which doesn’t include sm_120 kernels, so CUDA throws “no kernel image.” The fix is to install a PyTorch build compiled for CUDA 12.8+ (Blackwell support). PyTorch staff explicitly call out CUDA 12.8+ for Blackwell GPUs. (discuss.pytorch.org)


python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128



# CUDA 11.8 build
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
