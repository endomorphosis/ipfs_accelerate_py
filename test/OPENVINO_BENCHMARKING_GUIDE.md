# OpenVINO Benchmarking Guide

This guide explains how to use the enhanced OpenVINO benchmarking features within the IPFS Accelerate Python framework. The benchmarking system builds on the enhanced OpenVINO backend and provides comprehensive performance analysis across different precision formats and hardware configurations.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Benchmark Features](#benchmark-features)
- [Basic Usage](#basic-usage)
- [Advanced Usage](#advanced-usage)
- [INT8 Quantization](#int8-quantization)
- [Database Integration](#database-integration)
- [Report Generation](#report-generation)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The OpenVINO benchmarking system enables comprehensive performance testing of deep learning models across different Intel hardware platforms (CPU, GPU, VPU) with support for multiple precision formats (FP32, FP16, INT8). It integrates with the DuckDB database for result storage and provides detailed performance metrics and analysis.

Key capabilities include:
- Benchmarking models with different precision formats (FP32, FP16, INT8)
- Testing across various batch sizes
- Integration with optimum.intel for HuggingFace models
- INT8 quantization with calibration data
- Detailed performance metrics (latency, throughput, memory usage)
- Database integration for result storage
- Comprehensive report generation

## Prerequisites

1. Install the required dependencies:

```bash
# Install OpenVINO Runtime
pip install openvino>=2023.0.0

# Install optimum.intel for enhanced HuggingFace integration
pip install optimum[openvino]

# Install additional dependencies
pip install transformers torch
```

2. Ensure you have access to the database:

```bash
# Set the database path environment variable
export BENCHMARK_DB_PATH="./benchmark_db.duckdb"
```

## Benchmark Features

### Precision Formats

The benchmarking system supports multiple precision formats:

- **FP32 (float32)**: Full precision floating-point, suitable for high-accuracy requirements
- **FP16 (float16)**: Half precision floating-point, offers good balance between accuracy and performance
- **INT8 (integer 8-bit)**: Quantized 8-bit integer precision, provides significant performance improvements with some accuracy trade-offs

### Metrics Collected

For each benchmark, the following metrics are collected:

- **Latency**: Time to perform inference (milliseconds)
  - Average latency
  - Minimum latency
  - Maximum latency
  - P50, P95, P99 percentiles
- **Throughput**: Number of inferences per second (items/sec)
- **Memory Usage**: Memory consumption during inference (MB)
- **Load Time**: Time to load the model (seconds)

### Model Types Supported

The system can benchmark different types of models:

- **Text Models**: BERT, T5, DistilBERT, etc.
- **Vision Models**: ViT, ResNet, DETR, etc.
- **Audio Models**: Whisper, Wav2Vec2, etc.
- **Multimodal Models**: CLIP, FLAVA, etc.

## Basic Usage

### Using the Shell Script

The simplest way to run benchmarks is using the provided shell script:

```bash
# Run benchmarks for text models with default settings
./run_openvino_benchmarks.sh

# Run benchmarks for a specific model
./run_openvino_benchmarks.sh --models bert-base-uncased

# Run benchmarks for a specific model family
./run_openvino_benchmarks.sh --family vision

# Specify precision formats
./run_openvino_benchmarks.sh --precision FP32,FP16

# Specify batch sizes
./run_openvino_benchmarks.sh --batch-sizes 1,4,16

# Generate a report
./run_openvino_benchmarks.sh --report 1
```

### Using the Python Script Directly

You can also use the Python script directly for more control:

```bash
# Run benchmarks for a specific model
python benchmark_openvino.py --model bert-base-uncased

# Run benchmarks for a model family
python benchmark_openvino.py --model-family text

# Specify precision formats and batch sizes
python benchmark_openvino.py --model bert-base-uncased --precision FP32,FP16 --batch-sizes 1,2,4,8,16

# Generate a report
python benchmark_openvino.py --model bert-base-uncased --report --report-format markdown
```

## Advanced Usage

### Testing Multiple Models

You can test multiple models in a single run:

```bash
# Using the shell script
./run_openvino_benchmarks.sh --models "bert-base-uncased,t5-small,distilbert-base-uncased"

# Using the Python script
python benchmark_openvino.py --model bert-base-uncased,t5-small,distilbert-base-uncased
```

### Testing Multiple Device Types

If your system has multiple OpenVINO-compatible devices (e.g., CPU and Intel GPU), you can test across them:

```bash
# Using the shell script
./run_openvino_benchmarks.sh --device "CPU,GPU"

# Using the Python script
python benchmark_openvino.py --device CPU,GPU
```

### Configuring Iterations

You can specify the number of iterations for more reliable metrics:

```bash
# Using the shell script
./run_openvino_benchmarks.sh --iterations 20

# Using the Python script
python benchmark_openvino.py --iterations 20
```

### Dry Run Mode

To test your configuration without running actual benchmarks:

```bash
# Using the shell script
./run_openvino_benchmarks.sh --dry-run

# Using the Python script
python benchmark_openvino.py --dry-run
```

## INT8 Quantization

INT8 quantization can significantly improve performance with some trade-offs in accuracy. The benchmarking system supports advanced INT8 quantization with calibration data.

### Basic INT8 Benchmarking

```bash
# Using the shell script
./run_openvino_benchmarks.sh --precision INT8

# Using the Python script
python benchmark_openvino.py --precision INT8
```

### Configuring Calibration Samples

For more accurate INT8 quantization, you can specify the number of calibration samples:

```bash
# Using the Python script
python benchmark_openvino.py --precision INT8 --calibration-samples 20
```

## Database Integration

The benchmarking system integrates with DuckDB for storing and analyzing results.

### Storing Results in Database

By default, all benchmark results are stored in the database specified by the `BENCHMARK_DB_PATH` environment variable or provided as a command-line argument.

```bash
# Using the shell script with a specific database path
./run_openvino_benchmarks.sh --db-path ./custom_benchmark.duckdb

# Using the Python script with a specific database path
python benchmark_openvino.py --db-path ./custom_benchmark.duckdb
```

### Disabling Database Integration

If you don't want to store results in the database:

```bash
# Using the Python script
python benchmark_openvino.py --no-db
```

## Report Generation

The benchmarking system can generate comprehensive reports in different formats.

### Markdown Reports

```bash
# Using the shell script
./run_openvino_benchmarks.sh --report 1 --report-format markdown

# Using the Python script
python benchmark_openvino.py --report --report-format markdown
```

### HTML Reports

```bash
# Using the shell script
./run_openvino_benchmarks.sh --report 1 --report-format html

# Using the Python script
python benchmark_openvino.py --report --report-format html
```

### JSON Reports

```bash
# Using the shell script
./run_openvino_benchmarks.sh --report 1 --report-format json

# Using the Python script
python benchmark_openvino.py --report --report-format json
```

### Sample Report Content

Reports include:
- Summary of benchmark results by precision
- Detailed results for each model-device-precision-batch combination
- Latency and throughput metrics
- Memory usage

## Best Practices

### Selecting Models

- **Representative Models**: Choose models that represent your real workload
- **Various Sizes**: Include both small and large models
- **Different Types**: Include different model architectures (transformers, CNNs, etc.)

### Precision Selection

- **FP32**: Use for baseline accuracy and when precision is critical
- **FP16**: Good balance between performance and accuracy, use as default
- **INT8**: Use for maximum performance when some accuracy loss is acceptable

### Batch Size Selection

- **Batch Size 1**: Test for real-time inference scenarios
- **Multiple Batch Sizes**: Test with 1, 2, 4, 8, 16 to understand scaling behavior
- **Application-Specific**: Use batch sizes that match your application's needs

### Iterations

- **Minimum 10 Iterations**: Ensures stable metrics
- **More for Variance Analysis**: 20+ iterations helps identify performance variability
- **Warmup**: The system automatically runs a warmup iteration before measurement

## Troubleshooting

### Model Loading Issues

If you encounter model loading issues:

1. Check if the model is available locally or can be downloaded from HuggingFace
2. Verify the model supports OpenVINO conversion
3. Try with a smaller model first

```bash
# Test with a smaller model
python benchmark_openvino.py --model prajjwal1/bert-tiny
```

### Memory Issues

If you encounter memory issues:

1. Reduce batch sizes
2. Try with lower precision (FP16 or INT8)
3. Use a smaller model

```bash
# Reduce batch size and use lower precision
python benchmark_openvino.py --model bert-base-uncased --batch-sizes 1,2 --precision FP16
```

### Device Issues

If specific devices are not working:

1. Verify the device is available:
```bash
# Check available devices
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

2. Update OpenVINO drivers if necessary
3. Fall back to CPU device

### INT8 Quantization Issues

If INT8 quantization fails:

1. Try with more calibration samples
2. Verify the model architecture supports quantization
3. Fall back to FP16 precision

```bash
# Try with more calibration samples
python benchmark_openvino.py --model bert-base-uncased --precision INT8 --calibration-samples 30
```

## Conclusion

The OpenVINO benchmarking system provides a comprehensive way to measure and analyze the performance of deep learning models across different Intel hardware platforms and precision formats. With its integration with the database system and detailed reporting capabilities, it enables informed decision-making for deployment scenarios.

For any issues or suggestions, please report them through the project's issue tracker.

---

## Example Commands Reference

### Basic Benchmarking

```bash
# Basic text model benchmarking
python benchmark_openvino.py --model bert-base-uncased
./run_openvino_benchmarks.sh --models bert-base-uncased

# Basic vision model benchmarking
python benchmark_openvino.py --model-family vision
./run_openvino_benchmarks.sh --family vision

# Basic audio model benchmarking
python benchmark_openvino.py --model-family audio
./run_openvino_benchmarks.sh --family audio
```

### Comprehensive Benchmarking

```bash
# Comprehensive benchmarking across multiple precision formats
python benchmark_openvino.py --model bert-base-uncased,t5-small --precision FP32,FP16,INT8 --batch-sizes 1,2,4,8,16 --report

# Comprehensive benchmarking with results stored in database
python benchmark_openvino.py --model-family text --precision FP32,FP16,INT8 --batch-sizes 1,4,16 --db-path ./benchmark_db.duckdb --report

# Comparative benchmarking across different devices
python benchmark_openvino.py --model bert-base-uncased --device CPU,GPU --precision FP32,FP16 --batch-sizes 1,8 --report
```

### Report Generation

```bash
# Generate markdown report
python benchmark_openvino.py --model bert-base-uncased --report --report-format markdown --report-file benchmark_report.md

# Generate HTML report
python benchmark_openvino.py --model bert-base-uncased --report --report-format html --report-file benchmark_report.html

# Generate JSON report
python benchmark_openvino.py --model bert-base-uncased --report --report-format json --report-file benchmark_report.json
```

### Advanced INT8 Quantization

```bash
# Advanced INT8 quantization with custom calibration samples
python benchmark_openvino.py --model bert-base-uncased --precision INT8 --calibration-samples 20

# Compare INT8 vs FP16 vs FP32
python benchmark_openvino.py --model bert-base-uncased --precision FP32,FP16,INT8 --report
```