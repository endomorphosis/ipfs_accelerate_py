# 🚀 IPFS Accelerate Python - Installation Guide

This comprehensive guide covers installation options, troubleshooting, and getting started with IPFS Accelerate Python.

## 📦 Installation Options

### Quick Install (Recommended)
```bash
pip install ipfs_accelerate_py
```

### Development Install
```bash
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip install -e .
```

### Minimal Install (Testing/CI)
```bash
pip install ipfs_accelerate_py[minimal]
```

### Full Install (All Features)
```bash
pip install ipfs_accelerate_py[all]
```

## 🎯 Installation Modes by Use Case

### For Testing and Development
If you're contributing to the project or running tests:

```bash
# Clone and install in development mode
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip install -e .[testing]

# Run tests to verify installation
python run_all_tests.py
```

### For Production Deployment
For production environments with specific hardware:

```bash
# CPU-only environments
pip install ipfs_accelerate_py[minimal]

# GPU-enabled environments
pip install ipfs_accelerate_py[full]

# Web/Browser environments
pip install ipfs_accelerate_py[webnn]
```

### For Research and Analysis
For data analysis and visualization:

```bash
pip install ipfs_accelerate_py[viz]
```

## ⚡ Quick Start

After installation, verify everything works:

```python
from ipfs_accelerate_py import HardwareDetector

# Detect available hardware
detector = HardwareDetector()
available = detector.get_available_hardware()
print(f"Available hardware: {available}")

# Get best hardware for your setup
best = detector.get_best_available_hardware()
print(f"Best hardware: {best}")
```

## 🔧 Troubleshooting Common Issues

### Issue: ImportError with torch/transformers

**Problem**: `ImportError: No module named 'torch'` or similar for heavy dependencies.

**Solution**:
```bash
# The package includes graceful fallbacks
# For full ML functionality, install PyTorch:
pip install torch torchvision

# For transformers support:
pip install transformers>=4.46
```

**Alternative**: Use CPU-only mode which doesn't require GPU packages:
```python
# This works without torch installed
from ipfs_accelerate_py import HardwareDetector
detector = HardwareDetector()
# Will detect CPU and other available hardware
```

### Issue: CUDA not detected

**Problem**: CUDA-enabled GPU not being detected despite having NVIDIA hardware.

**Diagnosis**:
```python
from ipfs_accelerate_py import HardwareDetector
detector = HardwareDetector()
cuda_info = detector._detect_cuda()
print(f"CUDA detection result: {cuda_info}")
```

**Solutions**:
1. **Install PyTorch with CUDA support**:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Check NVIDIA drivers**:
   ```bash
   nvidia-smi  # Should show GPU information
   ```

3. **Verify CUDA installation**:
   ```bash
   nvcc --version  # Should show CUDA compiler version
   ```

### Issue: MPS not detected (Apple Silicon)

**Problem**: Apple Silicon GPU (Metal Performance Shaders) not being detected.

**Solutions**:
1. **Update macOS**: MPS requires macOS 12.3+
2. **Install PyTorch with MPS support**:
   ```bash
   pip install torch torchvision
   ```

3. **Test MPS availability**:
   ```python
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

### Issue: WebNN/WebGPU not detected

**Problem**: Browser-based AI acceleration not working.

**Solutions**:
1. **Install WebNN dependencies**:
   ```bash
   pip install ipfs_accelerate_py[webnn]
   ```

2. **Enable browser flags**:
   - Chrome: `--enable-webnn --enable-webgpu`
   - Edge: `--enable-webnn --enable-webgpu`

3. **Test in supported browsers**:
   - Chrome 106+
   - Edge 106+
   - Safari Technology Preview

### Issue: Permission denied errors

**Problem**: `PermissionError` during installation or package access.

**Solutions**:
1. **Use virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install ipfs_accelerate_py
   ```

2. **User installation**:
   ```bash
   pip install --user ipfs_accelerate_py
   ```

3. **System-wide installation** (use with caution):
   ```bash
   sudo pip install ipfs_accelerate_py
   ```

### Issue: Memory errors during model loading

**Problem**: `OutOfMemoryError` when loading large models.

**Solutions**:
1. **Check available memory**:
   ```python
   from ipfs_accelerate_py import HardwareDetector
   detector = HardwareDetector()
   memory_info = detector.get_memory_info()
   print(f"Available memory: {memory_info}")
   ```

2. **Use smaller models**:
   ```python
   # Instead of large models, use efficient alternatives
   model_name = "prajjwal1/bert-tiny"  # 4MB instead of 440MB
   ```

3. **Enable model quantization**:
   ```python
   # This is planned for future releases
   # Currently use native PyTorch quantization
   ```

### Issue: Slow model inference

**Problem**: Model inference is slower than expected.

**Diagnosis**:
```python
from ipfs_accelerate_py import HardwareDetector
detector = HardwareDetector()
performance = detector.benchmark_hardware()
print(f"Hardware performance: {performance}")
```

**Solutions**:
1. **Use optimal hardware**:
   ```python
   best_hardware = detector.get_best_available_hardware()
   print(f"Switch to: {best_hardware}")
   ```

2. **Check hardware utilization**:
   ```bash
   # For NVIDIA GPUs
   watch nvidia-smi
   
   # For system resources
   htop
   ```

3. **Model optimization** (planned feature):
   ```python
   # Future release will include
   # optimizer = ModelOptimizer()
   # optimized_model = optimizer.optimize_for_hardware(model, hardware)
   ```

### Issue: Tests failing

**Problem**: Test suite not passing completely.

**Diagnosis**:
```bash
# Run individual test suites
python test_smoke_basic.py      # Basic functionality (2s)
python test_comprehensive.py   # Comprehensive tests (8s)
python test_integration.py     # Integration tests (6s)

# Run with verbose output
python -m pytest test_*.py -v
```

**Solutions**:
1. **Missing dependencies**:
   ```bash
   pip install pytest>=8.0.0 pytest-timeout>=2.4.0
   ```

2. **Environment issues**:
   ```bash
   # Clean environment
   pip install --upgrade pip setuptools
   pip install -e .[testing]
   ```

3. **Hardware-specific issues**:
   ```python
   # Tests include mocking - should work on any system
   # If specific hardware tests fail, that's expected behavior
   ```

## 🌐 Platform-Specific Installation

### Linux (Ubuntu/Debian)
```bash
# System dependencies
sudo apt update
sudo apt install python3-dev python3-pip git

# Optional: CUDA support
sudo apt install nvidia-cuda-toolkit

# Install package
pip install ipfs_accelerate_py
```

### Linux (CentOS/RHEL)
```bash
# System dependencies
sudo yum install python3-devel python3-pip git

# Install package
pip install ipfs_accelerate_py
```

### macOS
```bash
# Using Homebrew
brew install python git

# Install package
pip install ipfs_accelerate_py
```

### Windows
```powershell
# Using chocolatey
choco install python git

# Or download from python.org
# Install package
pip install ipfs_accelerate_py
```

## 🧪 Testing Your Installation

### Basic Test
```python
#!/usr/bin/env python3
from ipfs_accelerate_py import HardwareDetector

def test_installation():
    """Test basic functionality."""
    try:
        detector = HardwareDetector()
        hardware = detector.get_available_hardware()
        
        print("✅ Installation successful!")
        print(f"Available hardware: {hardware}")
        
        if hardware.get('cpu'):
            print("✅ CPU detection working")
        if hardware.get('cuda'):
            print("✅ CUDA detection working")
        if hardware.get('mps'):
            print("✅ MPS (Apple Silicon) detection working")
            
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

if __name__ == "__main__":
    test_installation()
```

### Complete Test Suite
```bash
# Run all tests (should complete in ~15 seconds)
python run_all_tests.py

# Expected output:
# 🎉 ALL TESTS PASSED! 32/32 tests in 14.5s
```

## 📈 Performance Expectations

### Test Execution Times
- **Smoke tests**: ~2 seconds (6 tests)
- **Comprehensive tests**: ~8 seconds (16 tests)
- **Integration tests**: ~6 seconds (10 tests)
- **Total test suite**: ~15 seconds (32 tests)

### Hardware Detection Times
- **CPU detection**: ~0.1 seconds
- **GPU detection**: ~0.3 seconds
- **All hardware scan**: ~0.5 seconds
- **Cached results**: ~0.01 seconds

## 🚀 Next Steps

After successful installation:

1. **Read the documentation**:
   - `TESTING_README.md` - Testing infrastructure
   - `README.md` - General usage guide
   - `IMPLEMENTATION_PLAN.md` - Future features

2. **Explore examples**:
   - `examples/` directory
   - Test files for usage patterns

3. **Contribute**:
   - Check `IMPLEMENTATION_PLAN.md` for contribution opportunities
   - Run tests before submitting PRs
   - Follow existing code patterns

## 🆘 Getting Help

If you're still having issues after following this guide:

1. **Check existing issues**: [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
2. **Run diagnostics**:
   ```python
   from ipfs_accelerate_py import HardwareDetector
   detector = HardwareDetector()
   diagnostics = detector.run_diagnostics()
   print(diagnostics)
   ```
3. **Create a new issue** with:
   - Your operating system and Python version
   - Complete error messages
   - Output of the diagnostic script above

## 📋 Installation Checklist

- [ ] Python 3.8+ installed
- [ ] pip updated to latest version
- [ ] Virtual environment created (recommended)
- [ ] Package installed successfully
- [ ] Basic test passes
- [ ] Hardware detection working for your setup
- [ ] Test suite passes (if contributing)

---

*This installation guide covers the most common scenarios. For advanced configurations or enterprise deployments, please refer to the main documentation or contact the maintainers.*