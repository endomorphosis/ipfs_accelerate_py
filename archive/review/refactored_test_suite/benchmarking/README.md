# IPFS Accelerate Benchmarking System

This directory contains the performance benchmarking system for the IPFS Accelerate Python framework. The system provides standardized performance measurement across all supported hardware backends and model architectures.

## Features

- Standardized benchmarking across 6 hardware backends (CPU, CUDA, ROCm, MPS, OpenVINO, QNN)
- Support for all 11 model architectures in the framework
- Comprehensive metrics collection (latency, throughput, memory usage)
- Database storage for long-term tracking and analysis
- Visualization and reporting tools for hardware comparison

## Components

- **run_hardware_benchmark.py**: Core benchmarking script for running benchmarks on specific models and hardware
- **setup_benchmark_db.py**: Script to set up the benchmark database schema
- **visualize_benchmarks.py**: Visualization tools for benchmark results
- **batch_benchmark.py**: Script for running benchmarks on multiple models in batch
- **integration_benchmark.py**: Integration with Distributed Testing Framework

## Usage

### Setting Up the Database

Before using the benchmarking system, set up the database schema:

```bash
python setup_benchmark_db.py --db-path benchmark_db.duckdb
```

To overwrite an existing database:

```bash
python setup_benchmark_db.py --db-path benchmark_db.duckdb --overwrite
```

### Running Benchmarks

Run a benchmark for a specific model on a specific device:

```bash
python run_hardware_benchmark.py --model-id bert-base-uncased --device cuda --save
```

Run with custom parameters:

```bash
python run_hardware_benchmark.py \
  --model-id gpt2 \
  --device cuda \
  --precision float16 \
  --batch-sizes 1,2,4,8 \
  --sequence-lengths 128,256,512 \
  --iterations 20 \
  --save \
  --output-dir benchmark_results
```

### Visualizing Results

Visualize results from a JSON benchmark file:

```bash
python visualize_benchmarks.py --json-path benchmark_results/gpt2_cuda_float16_20250410_120000.json
```

Compare hardware performance for a specific model:

```bash
python visualize_benchmarks.py --db-path benchmark_db.duckdb --model-id bert-base-uncased
```

Generate architecture comparison visualization:

```bash
python visualize_benchmarks.py --db-path benchmark_db.duckdb --architecture-comparison
```

## Hardware Support

The benchmarking system supports the following hardware backends:

1. **CPU**: Universal fallback support for all models
2. **CUDA**: Optimized support for NVIDIA GPUs
3. **ROCm**: Support for AMD GPUs
4. **MPS**: Support for Apple Silicon (M1/M2/M3)
5. **OpenVINO**: Support for Intel CPUs, GPUs, and VPUs
6. **QNN**: Support for Qualcomm Neural Network devices

## Metrics

The following metrics are collected:

- **Latency**: Average, median, min, max, and 90th percentile latency in milliseconds
- **Throughput**: Samples processed per second
- **Memory Usage**: Peak and average memory usage in megabytes
- **First Token Latency**: Time to generate the first token (for generative models)
- **Load Time**: Time to load the model into memory

## Database Schema

The benchmark database uses the following schema:

- **benchmark_runs**: Metadata about benchmark runs
- **benchmark_results**: Detailed performance data
- **hardware_info**: Information about hardware used in benchmarks
- **raw_benchmark_data**: Raw benchmark data for detailed analysis

## Integration with Distributed Testing

The benchmarking system integrates with the Distributed Testing Framework to enable efficient execution of benchmarks across multiple hardware platforms in parallel.

## Adding Support for New Hardware

To add support for a new hardware backend:

1. Extend the `hardware_detection.py` module to detect and initialize the new hardware
2. Add hardware-specific model loading code to `ModelBenchmark` class
3. Implement hardware-specific metrics collection
4. Add the new hardware to the visualization color scheme

## Visualizations

The system generates the following visualizations:

1. **Latency vs Batch Size**: Plot of latency as batch size increases
2. **Throughput vs Batch Size**: Plot of throughput as batch size increases
3. **Memory Usage vs Batch Size**: Plot of memory usage as batch size increases
4. **Hardware Comparison**: Comparison of performance across hardware backends
5. **Architecture Comparison**: Comparison of performance across model architectures

## Requirements

- Python 3.7+
- DuckDB
- PyTorch
- Transformers
- Matplotlib
- Pandas
- NumPy

## Next Steps

- Implement power consumption metrics
- Add support for WebNN and WebGPU backends
- Enhance visualization of temporal performance trends
- Implement distributed benchmark execution
- Add regression detection for performance regressions