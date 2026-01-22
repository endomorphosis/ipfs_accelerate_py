# IPFS Accelerate Python - Installation & Setup Guide

This guide provides comprehensive installation and setup instructions for the IPFS Accelerate Python framework.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Hardware Setup](#hardware-setup)
- [IPFS Setup](#ipfs-setup)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Development Setup](#development-setup)

## System Requirements

### Operating System Support

- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+, Fedora 30+
- **macOS**: macOS 10.15+ (Intel and Apple Silicon)
- **Windows**: Windows 10+ (64-bit)

### Python Requirements

- **Python Version**: 3.8 or higher
- **Architecture**: x86_64, ARM64 (Apple Silicon), ARM (Raspberry Pi)

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 10 GB free space
- **Network**: Broadband internet connection

#### Recommended Requirements
- **CPU**: 4+ cores, 3.0+ GHz
- **RAM**: 8+ GB
- **Storage**: 50+ GB SSD
- **GPU**: NVIDIA GPU with CUDA support (optional)

### Optional Hardware Acceleration

- **NVIDIA CUDA**: CUDA 11.0+ with compatible GPU
- **AMD ROCm**: ROCm 5.0+ with compatible GPU
- **Intel OpenVINO**: OpenVINO 2023.1+ runtime
- **Apple Silicon**: macOS with M1/M2/M3 processor
- **Qualcomm**: Snapdragon devices with NPU support

## Installation Methods

### Method 1: PyPI Installation (Recommended)

#### Basic Installation

```bash
# Install from PyPI
pip install ipfs_accelerate_py

# Verify installation
python -c "import ipfs_accelerate_py; print('Installation successful')"
```

#### Installation with Hardware Support

```bash
# For WebNN/WebGPU browser support
pip install ipfs_accelerate_py[webnn]

# For visualization tools
pip install ipfs_accelerate_py[viz]

# For complete installation with all features
pip install ipfs_accelerate_py[all]
```

#### Hardware-Specific Installations

```bash
# For NVIDIA CUDA support
pip install ipfs_accelerate_py[cuda]

# For Intel OpenVINO support
pip install ipfs_accelerate_py[openvino]

# For AMD ROCm support
pip install ipfs_accelerate_py[rocm]

# For development tools
pip install ipfs_accelerate_py[dev]
```

### Method 2: Source Installation

#### Clone Repository

```bash
# Clone the repository
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install in development mode
pip install -e .

# Or install with all features
pip install -e ".[all]"
```

#### Build from Source

```bash
# Install build dependencies
pip install build wheel

# Build the package
python -m build

# Install the built package
pip install dist/ipfs_accelerate_py-*.whl
```

### Method 3: Container Installation

#### Docker

```dockerfile
# Dockerfile example
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install IPFS Accelerate Python
RUN pip install ipfs_accelerate_py[all]

# Set working directory
WORKDIR /app

# Copy your application
COPY . /app/

# Run your application
CMD ["python", "your_app.py"]
```

```bash
# Build and run Docker container
docker build -t ipfs-accelerate-app .
docker run -it ipfs-accelerate-app
```

#### Conda

```bash
# Create conda environment
conda create -n ipfs-accelerate python=3.10
conda activate ipfs-accelerate

# Install via pip (conda package coming soon)
pip install ipfs_accelerate_py[all]
```

## Hardware Setup

### NVIDIA CUDA Setup

#### Linux/WSL

```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-525

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvidia-smi
nvcc --version
```

#### Windows

1. Install NVIDIA GPU drivers from [NVIDIA website](https://www.nvidia.com/drivers)
2. Install CUDA Toolkit from [NVIDIA CUDA website](https://developer.nvidia.com/cuda-downloads)
3. Add CUDA to PATH in environment variables

#### Verification

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

### Intel OpenVINO Setup

#### Linux

```bash
# Download OpenVINO
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/linux/l_openvino_toolkit_ubuntu20_2023.1.0.12185.47b736f28c6_x86_64.tgz

# Extract and install
tar -xzf l_openvino_toolkit_ubuntu20_2023.1.0.12185.47b736f28c6_x86_64.tgz
cd l_openvino_toolkit_ubuntu20_2023.1.0.12185.47b736f28c6_x86_64
sudo ./install.sh

# Setup environment
source /opt/intel/openvino_2023/setupvars.sh
```

#### Windows

1. Download OpenVINO from [Intel website](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
2. Run the installer
3. Add OpenVINO to PATH

#### Verification

```python
try:
    from openvino.runtime import Core
    ie = Core()
    print("OpenVINO available")
    print(f"Available devices: {ie.available_devices}")
except ImportError:
    print("OpenVINO not available")
```

### AMD ROCm Setup

#### Linux (Ubuntu/RHEL)

```bash
# Add ROCm repository
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.4.3/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm
sudo apt update
sudo apt install rocm-dkms rocm-libs miopen-hip

# Add user to render group
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# Reboot required
sudo reboot
```

#### Verification

```bash
# Check ROCm installation
rocm-smi
rocminfo
```

### Apple Silicon (MPS) Setup

Apple Silicon support is built into macOS and PyTorch:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

No additional setup required for Apple Silicon Macs.

## IPFS Setup

### Option 1: Official IPFS (Kubo)

#### Linux/macOS

```bash
# Download and install IPFS
wget https://dist.ipfs.tech/kubo/v0.21.0/kubo_v0.21.0_linux-amd64.tar.gz
tar -xzf kubo_v0.21.0_linux-amd64.tar.gz
cd kubo
sudo bash install.sh

# Initialize IPFS
ipfs init

# Start IPFS daemon
ipfs daemon
```

#### Windows

1. Download IPFS from [IPFS Downloads](https://dist.ipfs.tech/#kubo)
2. Extract to a folder (e.g., `C:\ipfs`)
3. Add to PATH environment variable
4. Initialize: `ipfs init`
5. Start daemon: `ipfs daemon`

### Option 2: Docker IPFS

```bash
# Run IPFS in Docker
docker run -d --name ipfs-node \
  -p 4001:4001 -p 5001:5001 -p 8080:8080 \
  ipfs/kubo:latest

# Check IPFS status
curl http://localhost:5001/api/v0/version
```

### IPFS Configuration

```bash
# Configure IPFS for better performance
ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8080
ipfs config Addresses.API /ip4/0.0.0.0/tcp/5001
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["PUT", "POST"]'

# Enable experimental features
ipfs config --json Experimental.FilestoreEnabled true
ipfs config --json Experimental.UrlstoreEnabled true

# Restart daemon
ipfs shutdown
ipfs daemon &
```

### Verification

```bash
# Test IPFS connectivity
curl http://localhost:5001/api/v0/version

# Test gateway
curl http://localhost:8080/ipfs/QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn
```

## Configuration

### Configuration File Setup

Create a configuration file in your home directory:

```bash
# Create configuration directory
mkdir -p ~/.ipfs_accelerate

# Create configuration file
cat > ~/.ipfs_accelerate/config.json << 'EOF'
{
    "hardware": {
        "prefer_cuda": true,
        "allow_openvino": true,
        "allow_mps": true,
        "precision": "fp16",
        "mixed_precision": true,
        "max_memory": "8GB"
    },
    "ipfs": {
        "gateway": "http://localhost:8080/ipfs/",
        "local_node": "http://localhost:5001",
        "timeout": 30,
        "retry_count": 3
    },
    "performance": {
        "cache_size": "2GB",
        "parallel_requests": 4,
        "enable_prefetch": true
    },
    "logging": {
        "level": "INFO",
        "file": "~/.ipfs_accelerate/ipfs_accelerate.log"
    }
}
EOF
```

### Environment Variables

Set up environment variables for automatic configuration:

```bash
# Add to ~/.bashrc or ~/.zshrc
export IPFS_ACCELERATE_HARDWARE_PREFER_CUDA=true
export IPFS_ACCELERATE_IPFS_GATEWAY="http://localhost:8080/ipfs/"
export IPFS_ACCELERATE_IPFS_LOCAL_NODE="http://localhost:5001"
export IPFS_ACCELERATE_LOG_LEVEL=INFO

# Apply changes
source ~/.bashrc
```

### Project Configuration

Create a project-specific configuration file:

```bash
# In your project directory
cat > ipfs_accelerate.json << 'EOF'
{
    "project": {
        "name": "My ML Project",
        "version": "1.0.0",
        "description": "Machine learning inference project"
    },
    "models": {
        "bert-base-uncased": {
            "cache_dir": "./models/bert",
            "precision": "fp16"
        }
    },
    "hardware": {
        "batch_size": 8,
        "precision": "fp16"
    }
}
EOF
```

## Verification

### Basic Functionality Test

```python
#!/usr/bin/env python3
"""
Basic functionality test for IPFS Accelerate Python
"""

import asyncio
from ipfs_accelerate_py import ipfs_accelerate_py

def test_basic_functionality():
    """Test basic framework functionality."""
    print("Testing IPFS Accelerate Python...")
    
    # Initialize framework
    try:
        accelerator = ipfs_accelerate_py({}, {})
        print("✓ Framework initialization successful")
    except Exception as e:
        print(f"✗ Framework initialization failed: {e}")
        return False
    
    # Test hardware detection
    try:
        if hasattr(accelerator, 'hardware_detection'):
            hardware_info = accelerator.hardware_detection.detect_all_hardware()
            print(f"✓ Hardware detection successful")
            print(f"  Available hardware: {list(hardware_info.keys())}")
        else:
            print("⚠ Hardware detection not available")
    except Exception as e:
        print(f"✗ Hardware detection failed: {e}")
    
    # Test basic inference
    try:
        result = accelerator.process(
            model="bert-base-uncased",
            input_data={"input_ids": [101, 2054, 2003, 102]},
            endpoint_type="text_embedding"
        )
        print("✓ Basic inference successful")
    except Exception as e:
        print(f"✗ Basic inference failed: {e}")
    
    print("Basic functionality test completed!")
    return True

async def test_async_functionality():
    """Test asynchronous functionality."""
    print("\nTesting async functionality...")
    
    try:
        accelerator = ipfs_accelerate_py({}, {})
        
        result = await accelerator.process_async(
            model="bert-base-uncased",
            input_data={"input_ids": [101, 2054, 2003, 102]},
            endpoint_type="text_embedding"
        )
        print("✓ Async inference successful")
    except Exception as e:
        print(f"✗ Async inference failed: {e}")

if __name__ == "__main__":
    # Run basic tests
    test_basic_functionality()
    
    # Run async tests
    asyncio.run(test_async_functionality())
```

Save this as `test_installation.py` and run:

```bash
python test_installation.py
```

### Hardware Verification Test

```python
#!/usr/bin/env python3
"""
Hardware verification test
"""

from ipfs_accelerate_py import ipfs_accelerate_py

def test_hardware():
    """Test hardware acceleration capabilities."""
    print("Hardware Verification Test")
    print("=" * 30)
    
    accelerator = ipfs_accelerate_py({}, {})
    
    if hasattr(accelerator, 'hardware_detection'):
        hardware_info = accelerator.hardware_detection.detect_all_hardware()
        
        for hardware_type, info in hardware_info.items():
            status = "✓ Available" if info.get("available", False) else "✗ Not available"
            print(f"{hardware_type.upper()}: {status}")
            
            if info.get("available", False):
                for key, value in info.items():
                    if key != "available":
                        print(f"  {key}: {value}")
    else:
        print("Hardware detection not available")

if __name__ == "__main__":
    test_hardware()
```

### IPFS Connectivity Test

```python
#!/usr/bin/env python3
"""
IPFS connectivity test
"""

import asyncio
from ipfs_accelerate_py import ipfs_accelerate_py

async def test_ipfs():
    """Test IPFS connectivity."""
    print("IPFS Connectivity Test")
    print("=" * 25)
    
    config = {
        "ipfs": {
            "gateway": "http://localhost:8080/ipfs/",
            "local_node": "http://localhost:5001"
        }
    }
    
    accelerator = ipfs_accelerate_py(config, {})
    
    try:
        # Test basic IPFS operations
        test_data = b"Hello, IPFS!"
        cid = await accelerator.store_to_ipfs(test_data)
        print(f"✓ Stored data to IPFS: {cid}")
        
        retrieved_data = await accelerator.query_ipfs(cid)
        if retrieved_data == test_data:
            print("✓ Data retrieval successful")
        else:
            print("✗ Data retrieval failed: content mismatch")
            
    except Exception as e:
        print(f"✗ IPFS test failed: {e}")
        print("  Make sure IPFS daemon is running:")
        print("  ipfs daemon")

if __name__ == "__main__":
    asyncio.run(test_ipfs())
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'ipfs_accelerate_py'

```bash
# Check if package is installed
pip list | grep ipfs-accelerate

# If not installed, install it
pip install ipfs_accelerate_py

# Check Python path
python -c "import sys; print(sys.path)"
```

#### CUDA not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### IPFS connection failed

```bash
# Check if IPFS daemon is running
ps aux | grep ipfs

# Start IPFS daemon
ipfs daemon

# Check IPFS API
curl http://localhost:5001/api/v0/version
```

#### Permission denied errors

```bash
# Fix permissions
sudo chown -R $USER:$USER ~/.ipfs
sudo chown -R $USER:$USER ~/.ipfs_accelerate

# Add user to appropriate groups (Linux)
sudo usermod -a -G docker $USER
sudo usermod -a -G render $USER  # For ROCm
```

#### Memory errors

```bash
# Reduce batch size or model size
export IPFS_ACCELERATE_HARDWARE_MAX_MEMORY="4GB"

# Enable memory optimization
export IPFS_ACCELERATE_PERFORMANCE_ENABLE_MEMORY_OPTIMIZATION=true
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from ipfs_accelerate_py import ipfs_accelerate_py

# Enable debug mode in configuration
config = {
    "logging": {
        "level": "DEBUG",
        "enable_performance_logging": True
    }
}

accelerator = ipfs_accelerate_py(config, {})
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
2. Review the [documentation](../README.md)
3. Join our [community discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
4. Contact support with debug logs and system information

## Development Setup

### Development Dependencies

```bash
# Clone the repository
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/

# Run linting
flake8 ipfs_accelerate_py/
black ipfs_accelerate_py/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

For detailed contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Next Steps

After successful installation and setup:

1. Read the [Usage Guide](USAGE.md) for detailed usage instructions
2. Check out the [Examples](../examples/README.md) for practical examples
3. Review the [API Reference](API.md) for complete API documentation
4. Learn about [Hardware Optimization](HARDWARE.md) for your specific setup
5. Explore [IPFS Integration](IPFS.md) for distributed inference

Welcome to IPFS Accelerate Python! 🚀