# 📦 IPFS Accelerate Python - Installation Troubleshooting Guide

This comprehensive guide helps you install and configure IPFS Accelerate Python across different environments, with solutions for common issues and optimization tips.

## 🚀 Quick Start Installation

### Minimal Installation (Recommended for Testing)
```bash
# Basic functionality with core dependencies only
pip install ipfs-accelerate-py[minimal]

# Or install from source
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip install -e .[minimal]
```

### Full Installation (Production Ready)
```bash
# Complete installation with all optional dependencies
pip install ipfs-accelerate-py[full]

# For development with testing tools
pip install ipfs-accelerate-py[full,testing]
```

### Custom Installation Groups
```bash
# Core ML functionality
pip install ipfs-accelerate-py[core,ml]

# Web server capabilities
pip install ipfs-accelerate-py[minimal,web]

# Visualization and analysis tools
pip install ipfs-accelerate-py[core,viz]

# WebNN and WebGPU support
pip install ipfs-accelerate-py[webnn,webgpu]
```

## 🔧 Installation Modes by Use Case

### 1. Development and Testing
```bash
# Lightweight setup for development
pip install -e .[minimal,testing]

# Verify installation
python -c "import ipfs_accelerate_py; print('✅ Installation successful!')"
python examples/basic_usage.py --quick
```

### 2. Production Deployment
```bash
# Complete production setup
pip install ipfs-accelerate-py[full]

# Production validation
python utils/production_validation.py --level production
python utils/advanced_benchmarking.py --quick
```

### 3. Machine Learning Focus
```bash
# ML-optimized installation
pip install ipfs-accelerate-py[core,ml]

# Test ML capabilities
python examples/model_optimization.py --model bert-tiny
```

### 4. Web/Browser Integration
```bash
# Web-focused installation
pip install ipfs-accelerate-py[minimal,web,webnn,webgpu]

# Test web capabilities
python examples/demo_webnn_webgpu.py
```

### 5. Edge/Mobile Deployment
```bash
# Lightweight edge deployment
pip install ipfs-accelerate-py[minimal]

# Test edge compatibility
python utils/hardware_detection.py --mobile-check
```

## 🛠️ Platform-Specific Installation

### 🐧 Linux (Ubuntu/Debian)

#### System Dependencies
```bash
# Update package lists
sudo apt update

# Essential build tools
sudo apt install -y build-essential python3-dev python3-pip

# Optional GPU support
sudo apt install -y nvidia-cuda-toolkit  # For NVIDIA GPUs

# Optional Intel optimization
sudo apt install -y intel-mkl intel-mkl-dev  # For Intel CPUs
```

#### Installation Commands
```bash
# Standard installation
pip3 install ipfs-accelerate-py[full]

# Development installation
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip3 install -e .[full,testing]
```

#### Common Linux Issues

**Issue: `gcc` not found**
```bash
sudo apt install build-essential
```

**Issue: Python headers missing**
```bash
sudo apt install python3-dev python3-distutils
```

**Issue: CUDA not detected**
```bash
# Install NVIDIA drivers and CUDA toolkit
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
nvidia-smi  # Verify installation
```

### 🍎 macOS

#### Prerequisites
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.8+ if needed
brew install python@3.11
```

#### Installation Commands
```bash
# Standard installation
pip3 install ipfs-accelerate-py[full]

# For Apple Silicon optimization
pip3 install ipfs-accelerate-py[full] --extra-index-url https://download.pytorch.org/whl/cpu
```

#### Apple Silicon Specific

**MPS (Metal Performance Shaders) Support:**
```python
# Verify MPS availability
python3 -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"
```

**Common macOS Issues:**

**Issue: `clang` compilation errors**
```bash
# Ensure Xcode tools are installed
sudo xcode-select --reset
xcode-select --install
```

**Issue: Permission denied in `/usr/local`**
```bash
# Use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install ipfs-accelerate-py[full]
```

### 🪟 Windows

#### Prerequisites
```powershell
# Install Python 3.8+ from Microsoft Store or python.org
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### Installation Commands
```powershell
# Standard installation
pip install ipfs-accelerate-py[full]

# Development installation
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip install -e .[full,testing]
```

#### CUDA on Windows
```powershell
# Install CUDA toolkit from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvidia-smi
nvcc --version
```

#### Common Windows Issues

**Issue: `Microsoft Visual C++ 14.0 is required`**
```
Download and install Microsoft C++ Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**Issue: `torch` installation fails**
```powershell
# Install PyTorch with specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue: Long path names**
```powershell
# Enable long paths in Windows
# Run as administrator:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## 🌐 Specialized Environments

### 🐳 Docker Installation

#### Basic Docker Image
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install IPFS Accelerate Python
RUN pip install ipfs-accelerate-py[full]

# Copy application code
COPY . /app
WORKDIR /app

# Run application
CMD ["python", "examples/basic_usage.py"]
```

#### GPU-Enabled Docker
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install with GPU support
RUN pip install ipfs-accelerate-py[full]

# Test GPU availability
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### ☁️ Cloud Platform Installation

#### Google Colab
```python
# Install in Colab notebook
!pip install ipfs-accelerate-py[full]

# Test installation
import ipfs_accelerate_py
from utils.hardware_detection import HardwareDetector

detector = HardwareDetector()
print(f"Available hardware: {detector.get_available_hardware()}")
```

#### AWS EC2
```bash
# On EC2 instance
sudo yum update -y
sudo yum install -y python3 python3-pip git

# For GPU instances
sudo yum install -y nvidia-docker2

# Install package
pip3 install ipfs-accelerate-py[full] --user
```

#### Azure ML
```python
# In Azure ML environment
import subprocess
subprocess.run(["pip", "install", "ipfs-accelerate-py[full]"])

# Verify installation
from ipfs_accelerate_py import HardwareDetector
detector = HardwareDetector()
```

### 📱 Edge/Embedded Devices

#### Raspberry Pi
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dev tools
sudo apt install -y python3-dev python3-pip

# Lightweight installation for Pi
pip3 install ipfs-accelerate-py[minimal] --no-cache-dir

# Test installation
python3 -c "from hardware_detection import HardwareDetector; print('✅ Pi installation successful!')"
```

#### NVIDIA Jetson
```bash
# Install JetPack SDK first
# Then install Python package
pip3 install ipfs-accelerate-py[full]

# Verify GPU support
python3 -c "import torch; print(f'CUDA on Jetson: {torch.cuda.is_available()}')"
```

## 🔍 Verification and Testing

### Basic Installation Verification
```bash
# Quick verification script
python -c "
try:
    import ipfs_accelerate_py
    from hardware_detection import HardwareDetector
    
    detector = HardwareDetector()
    hardware = detector.get_available_hardware()
    
    print('✅ IPFS Accelerate Python installed successfully!')
    print(f'📊 Available hardware: {hardware}')
    print(f'🎯 Recommended: {detector.get_best_available_hardware()}')
    
except Exception as e:
    print(f'❌ Installation issue: {e}')
    print('💡 Try: pip install ipfs-accelerate-py[minimal]')
"
```

### Comprehensive System Test
```bash
# Run comprehensive system test
python examples/basic_usage.py --comprehensive

# Production validation
python utils/production_validation.py --level basic

# Quick benchmark
python utils/advanced_benchmarking.py --quick
```

### Dependency Verification
```bash
# Check dependency status
python utils/safe_imports.py

# Verify core functionality
python -c "
from utils.safe_imports import print_dependency_status
print_dependency_status()
"
```

## 🚨 Common Issues and Solutions

### Issue: `ImportError: No module named 'torch'`

**Solution 1: Install PyTorch**
```bash
# CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Solution 2: Use minimal installation**
```bash
# Install without heavy ML dependencies
pip install ipfs-accelerate-py[minimal]
```

### Issue: `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
# Install transformers
pip install transformers>=4.46

# Or use full installation
pip install ipfs-accelerate-py[full]
```

### Issue: Memory errors during installation

**Solution:**
```bash
# Use no-cache flag
pip install ipfs-accelerate-py[minimal] --no-cache-dir

# Or install in smaller chunks
pip install numpy duckdb aiohttp
pip install ipfs-accelerate-py[minimal]
```

### Issue: `gcc` or compilation errors

**Linux:**
```bash
sudo apt install build-essential python3-dev
```

**macOS:**
```bash
xcode-select --install
```

**Windows:**
```
Install Microsoft Visual C++ Build Tools
```

### Issue: SSL/Certificate errors

**Solution:**
```bash
# Update certificates
pip install --upgrade certifi

# Use trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org ipfs-accelerate-py[minimal]
```

### Issue: Permission denied errors

**Solution:**
```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

pip install ipfs-accelerate-py[full]
```

### Issue: `CUDA out of memory`

**Solution:**
```python
# Use CPU fallback
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Or limit GPU memory
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5)
```

## ⚡ Performance Optimization

### CPU Optimization
```bash
# Install with Intel MKL support
pip install mkl mkl-devel

# Set threading for optimal CPU performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### GPU Optimization
```bash
# Install optimized PyTorch version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB')"
```

### Memory Optimization
```python
# Configure for low-memory systems
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

## 📊 Monitoring Installation Health

### Automated Health Check
```bash
# Create health check script
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
import sys
import subprocess

def check_health():
    checks = []
    
    # Check Python version
    py_version = sys.version_info
    checks.append(("Python version", py_version >= (3, 8), f"{py_version.major}.{py_version.minor}"))
    
    # Check package installation
    try:
        import ipfs_accelerate_py
        checks.append(("IPFS Accelerate Python", True, "Installed"))
    except ImportError:
        checks.append(("IPFS Accelerate Python", False, "Not installed"))
    
    # Check hardware detection
    try:
        from hardware_detection import HardwareDetector
        detector = HardwareDetector()
        hardware = detector.get_available_hardware()
        checks.append(("Hardware Detection", True, f"{len(hardware)} platforms"))
    except Exception as e:
        checks.append(("Hardware Detection", False, str(e)))
    
    # Print results
    print("\n🔍 Installation Health Check")
    print("=" * 40)
    for check, status, details in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {check}: {details}")
    
    overall_health = all(status for _, status, _ in checks)
    print(f"\n🎯 Overall Health: {'✅ GOOD' if overall_health else '⚠️ NEEDS ATTENTION'}")
    
    return overall_health

if __name__ == "__main__":
    check_health()
EOF

python health_check.py
```

### Continuous Monitoring
```bash
# Add to cron for regular health checks
echo "0 */6 * * * cd /path/to/project && python health_check.py" | crontab -
```

## 🎯 Advanced Configuration

### Environment Variables
```bash
# Performance tuning
export IPFS_ACCELERATE_WORKERS=4
export IPFS_ACCELERATE_CACHE_SIZE=1000
export IPFS_ACCELERATE_LOG_LEVEL=INFO

# Hardware preferences
export IPFS_ACCELERATE_PREFERRED_HARDWARE=cuda
export IPFS_ACCELERATE_FALLBACK_HARDWARE=cpu

# Memory limits
export IPFS_ACCELERATE_MAX_MEMORY_GB=8
export IPFS_ACCELERATE_ENABLE_SWAP=false
```

### Configuration File
```python
# config.py - Advanced configuration
IPFS_ACCELERATE_CONFIG = {
    "hardware": {
        "preferred_order": ["cuda", "mps", "cpu"],
        "memory_limit_gb": 8,
        "enable_quantization": True
    },
    "performance": {
        "max_workers": 4,
        "cache_size": 1000,
        "enable_profiling": False
    },
    "logging": {
        "level": "INFO",
        "enable_file_logging": True,
        "log_file": "/var/log/ipfs_accelerate.log"
    }
}
```

## 🆘 Getting Help

### Self-Diagnosis
```bash
# Run diagnostic script
python utils/production_validation.py --diagnostic

# Check logs
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from hardware_detection import HardwareDetector
HardwareDetector()
"
```

### Support Resources
- 📚 **Documentation**: [GitHub Wiki](https://github.com/endomorphosis/ipfs_accelerate_py/wiki)
- 🐛 **Issues**: [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
- 📧 **Email**: Create an issue for direct support

### Reporting Issues
When reporting issues, include:
1. **System Information**: OS, Python version, hardware
2. **Installation Method**: pip, source, docker
3. **Error Messages**: Full stack traces
4. **Diagnostic Output**: `python utils/production_validation.py --diagnostic`

```bash
# Generate diagnostic report
python -c "
import sys, platform
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'Architecture: {platform.machine()}')

try:
    import ipfs_accelerate_py
    print('✅ Package installed')
    
    from hardware_detection import HardwareDetector
    detector = HardwareDetector()
    print(f'Hardware: {detector.get_available_hardware()}')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

---

## 📋 Installation Checklist

Before reporting issues, please verify:

- [ ] Python 3.8+ installed
- [ ] Latest pip version (`pip install --upgrade pip`)
- [ ] Virtual environment activated (recommended)
- [ ] System dependencies installed (build tools, etc.)
- [ ] Sufficient disk space (>2GB for full installation)
- [ ] Stable internet connection for downloads
- [ ] Hardware drivers updated (GPU, if applicable)
- [ ] Health check passes (`python health_check.py`)

---

*This guide is continuously updated. For the latest information, visit the [GitHub repository](https://github.com/endomorphosis/ipfs_accelerate_py).*