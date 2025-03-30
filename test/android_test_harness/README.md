# Android Test Harness

**Date: April 2025**  
**Status: Phase 2 (Alpha) Implementation**

## Overview

The Android Test Harness is a comprehensive framework for testing, benchmarking, and analyzing machine learning models on Android devices as part of the IPFS Accelerate Python Framework. It provides tools for model deployment, execution, performance measurement, thermal monitoring, and result reporting.

## Features

- **Android Device Management**: Connect to and manage Android devices via ADB
- **Model Deployment**: Push and prepare models for execution on Android devices
- **Real Model Execution**: Execute ONNX and TFLite models with hardware acceleration
- **Hardware Acceleration**: Support for CPU, GPU, NPU, and specialized accelerators (Qualcomm DSP, Samsung NPU, MediaTek APU)
- **Performance Metrics**: Collect detailed performance metrics including latency, throughput, and memory usage
- **Thermal Monitoring**: Monitor and analyze thermal conditions during model execution
- **Battery Impact Analysis**: Measure and report battery impact of model execution
- **Benchmark Database Integration**: Store and retrieve benchmark results from the DuckDB benchmark database
- **Report Generation**: Generate comprehensive performance and thermal reports

## Components

### Core Components

1. **AndroidDevice**: Manages connection and communication with Android devices via ADB
2. **AndroidModelRunner**: Handles model deployment and execution on Android devices
3. **AndroidModelExecutor**: Implements real model execution with hardware acceleration
4. **AndroidThermalMonitor**: Monitors and analyzes thermal conditions during model execution
5. **AndroidTestHarness**: Main class orchestrating the testing process

### Additional Components

- **android_thermal_analysis.py**: Comprehensive thermal analysis tools
- **database_integration.py**: Integration with benchmark database
- **cross_platform_analysis.py**: Cross-platform performance analysis
- **real_execution_example.py**: Example usage of real model execution

## Installation

The Android Test Harness is part of the IPFS Accelerate Python Framework. To use it, you need:

1. Python 3.8+
2. Android Debug Bridge (ADB)
3. Android device with developer mode enabled

```bash
# Clone the repository
git clone https://github.com/yourusername/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from test.android_test_harness.android_test_harness import AndroidTestHarness

# Initialize test harness
harness = AndroidTestHarness(device_serial="your_device_serial")

# Connect to device
harness.connect_to_device()

# Run benchmark
results = harness.run_benchmark(
    model_path="/path/to/your/model.onnx",
    model_name="my_model",
    model_type="onnx",
    batch_sizes=[1, 2, 4],
    accelerators=["auto", "cpu", "gpu"],
    iterations=50,
    save_to_db=True
)

# Generate report
report = harness.generate_report(results_data=results)
```

### Example Script

For a complete working example, see `real_execution_example.py`:

```bash
python test/android_test_harness/real_execution_example.py \
  --model /path/to/your/model.onnx \
  --name "My Model" \
  --serial your_device_serial \
  --analysis both \
  --batch-sizes 1,2,4 \
  --accelerators auto,cpu,gpu \
  --iterations 50 \
  --duration 300 \
  --db-path /path/to/benchmark.db \
  --output-dir ./android_results
```

## Command Line Interface

The Android Test Harness provides a command line interface for basic operations:

```bash
# Connect to a device
python -m test.android_test_harness.android_test_harness connect --serial your_device_serial

# Run a benchmark
python -m test.android_test_harness.android_test_harness benchmark \
  --model /path/to/your/model.onnx \
  --name "My Model" \
  --type onnx \
  --serial your_device_serial \
  --db-path /path/to/benchmark.db \
  --batch-sizes 1,2,4 \
  --accelerators auto,cpu,gpu \
  --iterations 50

# Generate a report
python -m test.android_test_harness.android_test_harness report \
  --results /path/to/results.json \
  --format markdown \
  --output report.md
```

## Accelerator Types

The Android Test Harness supports various hardware accelerators:

- **auto**: Automatically select the best available accelerator
- **cpu**: CPU execution
- **gpu**: GPU execution
- **npu**: Neural Processing Unit (generic)
- **dsp**: Digital Signal Processor (Qualcomm)
- **qnn**: Qualcomm Neural Network SDK
- **apu**: AI Processing Unit (MediaTek)

## Model Formats

Supported model formats:

- **onnx**: ONNX models
- **tflite**: TensorFlow Lite models
- **tflite_quantized**: Quantized TensorFlow Lite models
- **qnn**: Qualcomm Neural Network format (for Snapdragon devices)

## Database Integration

The Android Test Harness integrates with the benchmark database to store and retrieve benchmark results. To use this feature, provide a database path to the `AndroidTestHarness` constructor:

```python
harness = AndroidTestHarness(device_serial="your_device_serial", db_path="/path/to/benchmark.db")
```

## Thermal Monitoring

The Android Test Harness includes comprehensive thermal monitoring capabilities to analyze thermal behavior during model execution:

```python
from test.android_test_harness.android_thermal_monitor import AndroidThermalMonitor

# Initialize thermal monitor
thermal_monitor = AndroidThermalMonitor(device)

# Start monitoring
thermal_monitor.start_monitoring()

# ... run model ...

# Get thermal report
report = thermal_monitor.get_thermal_report()

# Stop monitoring
thermal_monitor.stop_monitoring()
```

## Real Model Execution

For real model execution using hardware acceleration, use the `AndroidModelExecutor`:

```python
from test.android_test_harness.android_model_executor import AndroidModelExecutor, ModelFormat, AcceleratorType

# Initialize model executor
executor = AndroidModelExecutor(device)

# Prepare model
remote_path = executor.prepare_model(
    model_path="/path/to/your/model.onnx",
    model_format=ModelFormat.ONNX,
    optimize_for_device=True
)

# Execute model
results = executor.execute_model(
    model_path=remote_path,
    model_format=ModelFormat.ONNX,
    accelerator=AcceleratorType.GPU,
    iterations=50,
    batch_size=1
)
```

## Implementation Status

The Android Test Harness implementation is currently in Phase 2 (Alpha) according to the Mobile Edge Support Expansion Plan:

- ✅ Basic Android device management
- ✅ Model deployment
- ✅ Thermal monitoring
- ✅ Performance metrics collection
- ✅ Benchmark database integration
- ✅ Report generation
- ✅ Real model execution framework
- 🔄 ONNX Runtime execution (Simulated in current version)
- 🔄 TFLite execution (Simulated in current version)
- 🔄 Advanced hardware acceleration (Under development)
- 🔄 iOS support (Planned)

## Future Work

Future development will focus on:

1. Implementing actual ONNX and TFLite runtime execution
2. Improving hardware acceleration support for specific platforms
3. Adding iOS support
4. Enhancing CI/CD integration
5. Building a web-based dashboard for visualization

## Documentation

For more detailed documentation, see:

- [MOBILE_EDGE_EXPANSION_PLAN.md](../MOBILE_EDGE_EXPANSION_PLAN.md): Overall plan for mobile/edge support
- [MOBILE_EDGE_SUPPORT_GUIDE.md](../MOBILE_EDGE_SUPPORT_GUIDE.md): Detailed guide for mobile/edge support
- [SAMSUNG_NPU_SUPPORT_GUIDE.md](../SAMSUNG_NPU_SUPPORT_GUIDE.md): Guide for Samsung NPU support
- [QUALCOMM_INTEGRATION_GUIDE.md](../QUALCOMM_INTEGRATION_GUIDE.md): Guide for Qualcomm integration

## Contributing

Contributions to the Android Test Harness are welcome. Please follow the standard contribution process for the IPFS Accelerate Python Framework.

## License

This project is licensed under the same license as the IPFS Accelerate Python Framework.