import torch
import sys

def check_cuda_installation():
    if not torch.cuda.is_available():
        import subprocess
        print("CUDA is not available. Installing PyTorch with CUDA support...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "-y"])
        if sys.platform.startswith('win'):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
        print("Installation completed. Please restart the script.")
        sys.exit(1)

check_cuda_installation()
num_gpus = torch.cuda.device_count()
print(f"Number of CUDA GPUs available: {num_gpus}")

if num_gpus > 0:
    # Print information for each GPU
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu.name}")
else:
    print("No CUDA GPUs available")

# Try to create a tensor on GPU to verify CUDA is working
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tensor = torch.zeros(1).to(device)
    print(f"Successfully created tensor on {device}")
except RuntimeError as e:
    print(f"Failed to use CUDA: {e}")
    
    