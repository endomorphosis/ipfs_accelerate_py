# Updated Model Implementations with CUDA and MPS Support

This directory contains updated model test implementations with full hardware support across all platforms, including CUDA and MPS (Apple Silicon).

## Current Implementations

| Model | File | CUDA Support | MPS Support | Status |
|-------|------|-------------|-------------|--------|
| BERT | [test_hf_bert.py](test_hf_bert.py) | ✅ | ✅ | Complete |
| T5 | [test_hf_t5.py](test_hf_t5.py) | ✅ | ✅ | Complete |

## Implementation Features

These implementations include:

1. **Complete Platform Support**:
   - CPU: General-purpose CPU execution
   - CUDA: NVIDIA GPU acceleration
   - OpenVINO: Intel hardware acceleration
   - MPS: Apple Silicon GPU acceleration
   - ROCm: AMD GPU acceleration 
   - WebNN: Browser-based neural network API
   - WebGPU: Browser-based GPU API

2. **Hardware-Aware Execution**:
   - Automatic hardware detection
   - Proper device placement for tensors
   - Graceful fallback to available hardware
   - Comprehensive error handling

3. **Standardized Implementation Pattern**:
   - Platform-specific initialization methods (`init_cuda`, `init_mps`, etc.)
   - Platform-specific handler creation (`create_cuda_handler`, `create_mps_handler`, etc.)
   - Consistent testing approach across platforms

## Usage

These files can be used in two ways:

1. **Direct Usage**:
   ```python
   from test_hf_bert import BERTTestBase
   
   # Create test instance
   test = BERTTestBase(model_id="bert-base-uncased")
   
   # Run test on all platforms
   results = test.test()
   
   # Or run on a specific platform
   cuda_results = test.run_test("cuda")
   ```

2. **Using the Runner Script**:
   ```bash
   # Test BERT on CUDA
   python run_hardware_tests.py --models bert --platforms cuda
   
   # Test T5 on MPS (Apple Silicon)
   python run_hardware_tests.py --models t5 --platforms mps
   
   # Test both models on all platforms
   python run_hardware_tests.py --models bert t5 --platforms all
   ```

## Implementation Pattern for Other Models

These files serve as templates for implementing support for other models. To implement support for a new model:

1. Copy the pattern of platform-specific initialization methods
2. Implement the appropriate handler creation methods
3. Ensure proper tensor placement on the target device
4. Add appropriate error handling and fallback mechanisms

See [PHASE16_COMPLETION_TASKS.md](../PHASE16_COMPLETION_TASKS.md) for the full list of models requiring implementation.

## Testing

To test these implementations, run:

```bash
cd ..  # Move to the parent directory
./run_phase16_fixes.sh
```

This script will test the implementations on all available hardware platforms and generate test reports in the `hardware_test_results` directory.