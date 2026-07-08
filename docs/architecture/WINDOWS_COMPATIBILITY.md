# Windows Compatibility Guide

This document provides Windows-specific installation and usage guidance for the IPFS Accelerate Python package.

## Python 3.12 Compatibility

The IPFS Accelerate Python package now supports Python 3.12 on Windows with the following considerations:

### System Requirements

- Python 3.8+ (including 3.12)
- Windows 10/11 (64-bit recommended)
- At least 4GB RAM (8GB+ recommended for larger models)
- GPU support (optional but recommended):
  - NVIDIA GPU with CUDA 11.8+ 
  - Intel GPU with Intel Extension for PyTorch
  - DirectML for AMD/Intel integrated graphics

### Installation on Windows

#### Option 1: Using pip (Recommended)
```bash
pip install ipfs_accelerate_py
```

#### Option 2: Development Installation
```bash
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip install -e .
```

### Windows-Specific Dependencies

Some dependencies may require additional setup on Windows:

1. **Visual Studio Build Tools**: Required for compiling native extensions
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Select "C++ build tools" workload

2. **PyTorch**: For GPU acceleration
   ```bash
   # For NVIDIA GPUs
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Long Path Support**: Enable long path support for Windows:
   - Run as administrator: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`
   - Or through Group Policy

### CLI Usage on Windows

The CLI tool supports Windows paths and provides enhanced error handling:

```cmd
# Basic usage
python -m ipfs_accelerate_py.ipfs_cli infer --model bert-base-uncased

# With Windows paths
python -m ipfs_accelerate_py.ipfs_cli infer --model "C:\Models\my-model" --output "C:\Results\output.json"

# Fast mode for performance
python -m ipfs_accelerate_py.ipfs_cli infer --model bert-base-uncased --fast

# Local mode (no IPFS networking)
python -m ipfs_accelerate_py.ipfs_cli infer --model bert-base-uncased --local
```

### Performance Considerations

#### GPU Acceleration
- **NVIDIA GPUs**: Use CUDA backend for best performance
- **Intel GPUs**: Enable Intel Extension for PyTorch
- **AMD GPUs**: Use DirectML backend (limited support)

#### Memory Management
- Windows memory management differs from Linux
- Use smaller batch sizes if encountering memory errors
- Enable virtual memory if needed for large models

#### Path Handling
- Use forward slashes or raw strings for paths: `r"C:\path\to\model"`
- Avoid spaces in model names when possible
- Use quotes around paths with spaces

### Web Interface

The web interface now supports modern Windows browsers:

```python
from ipfs_accelerate_py import ipfs_accelerate_py

# Initialize with Windows-specific config
config = {
    "web_interface": {
        "host": "localhost",
        "port": 8080,
        "browser": "edge"  # or "chrome", "firefox"
    }
}
accelerator = ipfs_accelerate_py({}, config)
```

### Troubleshooting

#### Common Issues

1. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'ipfs_accelerate_py'
   ```
   **Solution**: Ensure package is installed: `pip install ipfs_accelerate_py`

2. **Permission Errors**:
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   **Solution**: Run PowerShell as Administrator or check folder permissions

3. **Path Too Long Errors**:
   ```
   OSError: [Errno 36] File name too long
   ```
   **Solution**: Enable long path support or use shorter paths

4. **GPU Not Detected**:
   ```
   No CUDA devices found
   ```
   **Solutions**:
   - Update GPU drivers
   - Install CUDA Toolkit
   - Verify PyTorch CUDA installation

#### Environment Variables

Set these environment variables for optimal performance:

```cmd
# For CUDA
set CUDA_VISIBLE_DEVICES=0
set TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# For Intel Extension
set INTEL_EXTENSION_FOR_PYTORCH=1

# For debugging
set IPFS_ACCELERATE_DEBUG=1
set IPFS_ACCELERATE_LOG_LEVEL=DEBUG
```

### Testing on Windows

Run the compatibility tests:

```cmd
# Basic compatibility check
python -m ipfs_accelerate_py.test.compatibility_check_fixed

# Full test suite
python -m ipfs_accelerate_py.test.test_ipfs_accelerate_simple_fixed --verbose

# CLI tests
python -m ipfs_accelerate_py.ipfs_cli test --verbose
```

### Known Limitations

1. **WebGPU Support**: Limited on Windows, use WebNN or DirectML instead
2. **Container Support**: Docker Desktop required for containerized workloads
3. **File Locking**: Some antivirus software may interfere with model caching
4. **Network Firewalls**: Corporate firewalls may block IPFS networking

### Support

For Windows-specific issues:

1. Check the [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
2. Review Windows compatibility tests in `/test/` directory
3. Enable debug logging for detailed error information
4. Consider using WSL2 for Linux-like environment

### Version History

- **v0.0.45**: Added Python 3.12 support and Windows compatibility
- **Future**: Enhanced DirectML support, better WebNN integration