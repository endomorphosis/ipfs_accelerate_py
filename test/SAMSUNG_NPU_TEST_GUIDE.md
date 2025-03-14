# Samsung NPU Support Testing Guide

This guide provides instructions for testing the Samsung Neural Processing Unit (NPU) support implementation in the IPFS Accelerate Python Framework.

## Prerequisites

Before running any tests, you'll need to install the required dependencies:

```bash
# Install all Samsung NPU support dependencies
pip install -r requirements_samsung.txt

# Or for minimal testing (ultra-minimal test only)
pip install numpy>=1.20.0
```

The `requirements_samsung.txt` file includes:
- Core dependencies: numpy
- Database dependencies: duckdb, pandas
- API dependencies: fastapi, uvicorn, pydantic
- Visualization: matplotlib, plotly (optional)

## Basic Tests

We provide several test scripts with increasing levels of complexity:

### 1. Ultra-Minimal Test

This test verifies only the most basic functionality of the Samsung NPU support:

```bash
# Run with simulation mode (no actual hardware required)
TEST_SAMSUNG_CHIPSET=exynos_2400 python test_minimal_samsung.py
```

### 2. Basic Samsung NPU Test

This test provides a more comprehensive verification of core functionality:

```bash
# Run with simulation mode
python test_samsung_npu_basic.py --simulation

# Test only standalone detection
python test_samsung_npu_basic.py --simulation --standalone

# Test only centralized hardware detection integration
python test_samsung_npu_basic.py --simulation --centralized

# Test only model compatibility analysis
python test_samsung_npu_basic.py --simulation --compatibility
```

### 3. Comprehensive Test Suite

This runs the full test suite, including extensive unit tests:

```bash
# Run the comprehensive test suite
python test_samsung_support.py
```

### 4. Hardware Comparison Tool

Test the hardware comparison capabilities with Samsung NPU:

```bash
# Generate hardware capability report (text format)
python test_samsung_npu_comparison.py report --samsung-simulation --format text

# Test model compatibility
python test_samsung_npu_comparison.py compatibility --model models/bert-base-uncased.onnx --samsung-simulation

# Run performance benchmark comparison
python test_samsung_npu_comparison.py benchmark --model models/bert-base-uncased.onnx --hardware samsung qualcomm cpu --samsung-simulation
```

### 5. Mobile NPU Comparison

Compare Samsung NPU with other mobile NPU hardware:

```bash
# Run the full comparison
python test_mobile_npu_comparison.py

# Output results in JSON format
python test_mobile_npu_comparison.py --json

# Test only model compatibility
python test_mobile_npu_comparison.py --models-only
```

## Integration Tests

### Integration with Centralized Hardware Detection

Test the integration with the centralized hardware detection system:

```bash
# Import HardwareManager from centralized hardware detection
from centralized_hardware_detection.hardware_detection import HardwareManager

# Create hardware manager
hw_manager = HardwareManager()

# Get capabilities and check for Samsung NPU
capabilities = hw_manager.get_capabilities()
has_samsung_npu = capabilities.get("samsung_npu", False)
is_simulation = capabilities.get("samsung_npu_simulation", False)

print(f"Samsung NPU detected: {has_samsung_npu}, Simulation mode: {is_simulation}")

# Check model compatibility
model_name = "bert-base-uncased"
compatibility = hw_manager.get_model_hardware_compatibility(model_name)
is_model_compatible = compatibility.get("samsung_npu", False)

print(f"Model {model_name} compatible with Samsung NPU: {is_model_compatible}")
```

## Troubleshooting

If you encounter issues running the tests:

1. **Missing packages**: Ensure all required packages are installed
   - For ultra-minimal test: Only `numpy` is required
   - For basic test: `numpy`, `duckdb`, `pandas` are required
   - For comprehensive tests: All dependencies in `requirements_samsung.txt` are needed
   
2. **Simulation mode**: Make sure to set the `TEST_SAMSUNG_CHIPSET` environment variable for simulation mode

3. **Import errors**: Verify your Python path includes the project root directory

4. **Test failure by test type**:
   - If ultra-minimal test fails but other tests work: Could be an issue with the core Samsung NPU classes
   - If centralized detection fails but standalone detection works: Check the centralized hardware detection integration
   - If only model compatibility tests fail: Check the database dependencies
   
5. **Dependency issues**:
   - `ImportError` for `duckdb`: Install with `pip install duckdb>=0.8.0`
   - `ImportError` for `pandas`: Install with `pip install pandas>=1.3.0`
   - `ImportError` for `fastapi`: Install with `pip install fastapi>=0.95.0`
   - For visualization errors: Install optional dependencies with `pip install matplotlib>=3.5.0 plotly>=5.5.0`

## Real Hardware Testing

When testing with actual Samsung NPU hardware:

1. Remove any `TEST_SAMSUNG_CHIPSET` environment variable settings
2. Ensure the Samsung Neural SDK is installed and configured
3. Set the `SAMSUNG_SDK_PATH` environment variable to point to your Samsung ONE SDK installation

```bash
# Example for real hardware testing
unset TEST_SAMSUNG_CHIPSET
export SAMSUNG_SDK_PATH=/opt/samsung/one-sdk
python test_samsung_npu_basic.py
```