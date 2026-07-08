# iOS Test Harness

**Date: April 2025**  
**Status: Phase 2 (Alpha) Implementation**

## Overview

The iOS Test Harness is a framework for testing, benchmarking, and analyzing machine learning models on iOS devices. It provides tools for model deployment, execution, performance measurement, thermal monitoring, and result reporting, working as the iOS counterpart to the Android Test Harness within the IPFS Accelerate Python Framework.

## Features

- **iOS Device Management**: Connect to and manage iOS devices via USB
- **Core ML Integration**: Execute models using Apple's Core ML framework
- **ONNX Conversion**: Convert ONNX models to Core ML format
- **Neural Engine Support**: Utilize Apple Neural Engine hardware acceleration
- **Performance Metrics**: Collect detailed performance metrics including latency, throughput, memory usage
- **Thermal Monitoring**: Monitor device temperature during model execution
- **Battery Impact Analysis**: Measure and analyze battery consumption
- **Benchmark Database Integration**: Store and retrieve results from the benchmark database
- **Report Generation**: Generate comprehensive performance reports in markdown and HTML formats

## Requirements

- **macOS**: The iOS Test Harness requires macOS to work with iOS devices
- **Python 3.8+**: Compatible with Python 3.8 and newer
- **Xcode**: Xcode with command line tools for iOS device connectivity
- **iOS Debugging Tools**: libimobiledevice suite for device communication
- **iOS Device**: iOS 13.0+ device with Developer Mode enabled

## Installation

```bash
# Install dependencies
brew install libimobiledevice
pip install onnx onnx-coreml

# Clone the repository
git clone https://github.com/your-org/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install the Python package
pip install -e .
```

## Usage

### Basic Usage

```python
from test.ios_test_harness import IOSTestHarness

# Initialize the test harness
harness = IOSTestHarness()

# Connect to an iOS device
harness.connect_to_device()

# Run a benchmark
results = harness.run_benchmark(
    model_path="/path/to/model.mlmodel",
    model_name="my_model",
    model_type="coreml",
    batch_sizes=[1, 2, 4],
    iterations=50
)

# Generate a report
report = harness.generate_report(results_data=results, report_format="markdown")
```

### Command Line Usage

```bash
# Connect to an iOS device
python -m test.ios_test_harness.ios_test_harness connect --verbose

# Run a benchmark
python -m test.ios_test_harness.ios_test_harness benchmark \
  --model path/to/model.mlmodel \
  --name "My Model" \
  --type coreml \
  --iterations 50 \
  --batch-sizes 1,2,4 \
  --output-dir ./ios_results

# Generate a report
python -m test.ios_test_harness.ios_test_harness report \
  --results path/to/results.json \
  --format html \
  --output report.html
```

## Core Components

### IOSDevice

This class manages connection and communication with iOS devices via USB. It provides functionality for:

- Device detection and connection
- Device information retrieval
- Battery and thermal data collection
- File transfer between host and device
- App installation and management

### IOSModelRunner

This class handles model deployment and execution on iOS devices, supporting:

- Core ML model execution
- ONNX model conversion to Core ML
- Remote execution on the device
- Performance metrics collection

### IOSTestHarness

The main class that orchestrates the entire testing process:

- Device management
- Benchmark execution with multiple configurations
- Database integration
- Report generation
- Metrics collection and analysis

## Model Formats

The iOS Test Harness supports the following model formats:

- **Core ML (.mlmodel)**: Native Apple machine learning format
- **ONNX (.onnx)**: Open Neural Network Exchange format (automatically converted to Core ML)

## Compute Units

Core ML models can target different compute units on iOS devices:

- **CPU**: Central Processing Unit for general computation
- **GPU**: Graphics Processing Unit for parallel computation
- **Neural Engine**: Apple's dedicated neural processing hardware (A12 Bionic and newer)
- **All**: Automatic selection of the best compute unit based on the model

## Database Integration

The iOS Test Harness integrates with the benchmark database system, allowing:

- Storage of benchmark results
- Cross-platform performance comparisons
- Historical performance tracking
- Advanced queries and analytics

```python
# Store results in database
harness = IOSTestHarness(db_path="/path/to/benchmark.db")
results = harness.run_benchmark(
    model_path="/path/to/model.mlmodel",
    model_name="my_model",
    save_to_db=True
)
```

## Implementation Status

The iOS Test Harness is currently in Phase 2 (Alpha) implementation according to the Mobile Edge Support Expansion Plan:

- ✅ Basic iOS device management
- ✅ Model deployment framework
- ✅ Performance metrics collection
- ✅ Thermal monitoring
- ✅ Battery impact analysis
- ✅ Report generation
- ✅ Database integration
- 🔄 ONNX to Core ML conversion
- 🔄 Neural Engine optimization
- 🔄 Real device validation
- 🔄 CI/CD integration

## Limitations

- Currently requires macOS for iOS device communication
- Some features are simulated in the alpha implementation
- Requires a benchmark app to be installed on the iOS device (not included)

## Future Work

Future development will focus on:

1. Real device validation and testing
2. Implementing the benchmark app for iOS
3. CI/CD integration for automated testing
4. Enhanced Neural Engine optimization
5. Support for more complex model architectures
6. Federation with the Android Test Harness for cross-platform analysis

## Documentation

For more detailed documentation, see:

- [MOBILE_EDGE_EXPANSION_PLAN.md](../MOBILE_EDGE_EXPANSION_PLAN.md): Overall plan for mobile/edge support
- [MOBILE_EDGE_SUPPORT_GUIDE.md](../MOBILE_EDGE_SUPPORT_GUIDE.md): Detailed guide for mobile/edge support
- [BATTERY_IMPACT_ANALYSIS.md](../BATTERY_IMPACT_ANALYSIS.md): Methodology for battery impact analysis

## License

This project is licensed under the same license as the IPFS Accelerate Python Framework.