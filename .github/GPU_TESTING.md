# GPU Testing with Self-Hosted Runners

This document explains how to run GPU-enabled tests using self-hosted runners.

## Overview

The CI/CD pipeline supports two testing modes:

### 1. Lightweight Testing (Default - GitHub-hosted runners)
- Uses `testing` Docker stage
- Includes only minimal dependencies (`[minimal,testing,mcp]`)
- ~1GB image size
- Runs on GitHub-hosted runners (`ubuntu-latest`)
- **No GPU/CUDA support**
- Ideal for basic CI tests, imports, and unit tests

### 2. GPU Testing (Manual - Self-hosted runners recommended)
- Uses `testing-gpu` Docker stage
- Includes full ML stack with PyTorch, CUDA, Transformers
- ~6GB image size
- Requires manual workflow trigger with `build_gpu_images: true`
- **Supports GPU/CUDA for AI inference testing**
- Best for self-hosted runners with GPUs

## Using Self-Hosted Runners for GPU Tests

### Prerequisites

1. **Self-hosted runner with GPU**
   - NVIDIA GPU with CUDA support
   - Docker installed
   - NVIDIA Container Toolkit installed
   - Runner registered with your repository

2. **Runner label**
   - Add label `gpu` to your self-hosted runner
   - Or use default `self-hosted` label

### Running GPU Tests

#### Option 1: Manual Workflow Dispatch (Recommended)

1. Go to Actions → AMD64 CI/CD Pipeline (or Multi-Architecture CI/CD Pipeline)
2. Click "Run workflow"
3. Set the following inputs:
   - `build_gpu_images`: ✅ true
   - `use_self_hosted`: ✅ true (to use self-hosted runners)
4. Click "Run workflow"

#### Option 2: Modify Workflow File

Change the GPU test job to always use self-hosted runners:

```yaml
test-gpu-support:
  runs-on: self-hosted  # Force self-hosted runner
  # ... rest of job
```

### What Gets Tested

The GPU tests will:

1. ✅ Build the `testing-gpu` Docker image with full ML stack
2. ✅ Verify PyTorch installation and CUDA availability
3. ✅ Verify Transformers installation
4. ✅ Run GPU-aware tests (tests marked with `gpu`, `cuda`, or `inference` keywords)
5. ✅ Detect if GPU is available and use it if present

### Example Test Output

On self-hosted GPU runner:
```
✅ GPU image import successful
✅ PyTorch available: 2.1.0
✅ CUDA available: True
✅ CUDA version: 12.1
✅ GPU devices: 1
✅ Transformers available: 4.46.0
✅ GPU detected, enabling GPU access in container
```

On GitHub-hosted runner (no GPU):
```
✅ GPU image import successful
✅ PyTorch available: 2.1.0
✅ CUDA available: False
✅ Transformers available: 4.46.0
ℹ️ No GPU detected, running CPU-only tests
```

## Disk Space Requirements

| Stage | Size | Recommended Runner |
|-------|------|-------------------|
| `testing` | ~1GB | GitHub-hosted (14GB available) |
| `testing-gpu` | ~6GB | Self-hosted (20GB+ recommended) |
| `development` | ~6GB | Self-hosted (20GB+ recommended) |
| `production` | ~2GB | GitHub-hosted or Self-hosted |

## GitHub-Hosted vs Self-Hosted Runners

### GitHub-Hosted Runners
- ❌ No GPU support
- ❌ Limited disk space (~14GB)
- ✅ Free for public repos
- ✅ Pre-configured environment
- ✅ Good for basic CI tests

### Self-Hosted Runners
- ✅ GPU support available
- ✅ More disk space (configurable)
- ✅ Faster builds (can cache Docker layers locally)
- ❌ Requires setup and maintenance
- ❌ Must manage your own infrastructure

## Troubleshooting

### "No space left on device" on GitHub-hosted runners
- ✅ **Fixed**: Use `testing` stage (not `testing-gpu`)
- The lightweight testing stage is designed to fit within GitHub runner limits

### "No space left on device" on self-hosted runners
- Check available disk space: `df -h`
- Clean up Docker: `docker system prune -af --volumes`
- Increase disk allocation for your runner

### GPU not detected in tests
- Verify NVIDIA drivers: `nvidia-smi`
- Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi`
- Check workflow uses `use_self_hosted: true` input

### Tests skip GPU tests
- Tests must be marked with pytest markers: `@pytest.mark.gpu` or use keywords in test name
- Or tests will run with `-k "gpu or cuda or inference"` filter

## Contributing

When adding GPU-specific tests:

1. Mark tests with appropriate keywords or markers:
```python
@pytest.mark.gpu
def test_model_inference_gpu():
    # GPU test code
    pass

# Or use naming convention
def test_cuda_memory_allocation():
    # Test will be picked up by -k "cuda"
    pass
```

2. Make tests conditional on GPU availability:
```python
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_gpu_inference():
    # Test code
    pass
```

## Questions?

If you have questions about GPU testing setup, please:
1. Check this documentation
2. Review the workflow files in `.github/workflows/`
3. Open an issue with the `ci/cd` label
