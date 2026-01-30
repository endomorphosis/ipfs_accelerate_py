# Skillset Benchmark Implementation

This directory contains benchmark implementations for measuring the performance of skillset implementations in the IPFS Accelerate Python framework.

## Overview

The skillset benchmark system provides a way to measure and compare the performance of different model implementations across various hardware backends. It focuses on two key metrics:

1. **Initialization Performance**: Measures how quickly models can be loaded and initialized
2. **Throughput Performance**: Measures how many models can be concurrently loaded in parallel

## Benchmark Types

### Inference Benchmark

The `SkillsetInferenceBenchmark` class measures the initialization time of a skillset implementation across different batch sizes. It provides detailed metrics on:

- Module import time
- Class instantiation time
- Model initialization time for various batch sizes
- Statistical metrics (mean, standard deviation)

### Throughput Benchmark

The `SkillsetThroughputBenchmark` class extends the inference benchmark to measure throughput performance when multiple model instances are initialized concurrently. It provides:

- Concurrent execution timing
- Throughput in models per second
- Speedup compared to sequential execution

## Hardware Support

The benchmark system supports multiple hardware backends:

- CPU: Standard CPU processing
- CUDA: NVIDIA GPU acceleration
- ROCm: AMD GPU acceleration
- OpenVINO: Intel's neural network optimization framework
- MPS: Apple's Metal Performance Shaders for Apple Silicon
- QNN: Qualcomm Neural Network processing

## Usage

### Generating Benchmark Files

To generate benchmark files for skillset implementations:

```bash
python generate_skillset_benchmarks.py --skillset-dir ../ipfs_accelerate_py/worker/skillset --output-dir data/benchmarks/skillset
```

### Running Individual Benchmarks

To run a benchmark for a specific model:

```bash
# For inference benchmark
python data/benchmarks/skillset/benchmark_bert.py --type inference --hardware cpu

# For throughput benchmark
python data/benchmarks/skillset/benchmark_bert.py --type throughput --hardware cpu --concurrent-workers 4
```

### Running All Benchmarks

To run benchmarks for all skillset implementations:

```bash
python run_all_skillset_benchmarks.py --hardware cpu --type both --report
```

Command-line options:
- `--skillset-dir`: Directory containing skillset implementations
- `--output-dir`: Directory to write benchmark results
- `--benchmark-dir`: Directory to write benchmark files
- `--hardware`: Hardware backend to use (cpu, cuda, rocm, openvino, mps, qnn)
- `--type`: Type of benchmark to run (inference, throughput, both)
- `--concurrent-workers`: Number of concurrent workers for throughput benchmarks
- `--batch-sizes`: Comma-separated list of batch sizes to test
- `--runs`: Number of measurement runs
- `--model`: Specific model to benchmark (can be specified multiple times)
- `--generate-only`: Only generate benchmark files, don't run benchmarks
- `--report`: Generate HTML reports for benchmark results

## Results

Benchmark results are saved as JSON files in the specified output directory. If the `--report` option is used, HTML reports with visualizations are also generated.

### Inference Benchmark Results

Inference benchmark results include:
- Module import time
- Class instantiation time
- Initialization times for each batch size
- Statistical metrics (mean, standard deviation)

### Throughput Benchmark Results

Throughput benchmark results include:
- Concurrent execution time
- Throughput in models per second
- Speedup compared to sequential execution

## Integration with the Benchmark Framework

The skillset benchmark system is integrated with the IPFS Accelerate benchmark framework using the registry pattern. Benchmark classes are registered with metadata that allows them to be discovered and executed by the benchmark runner.

## Implementation Notes

- The benchmark system uses reflection to dynamically load and test skillset implementations
- Warmup runs are performed to ensure more accurate measurements
- Statistical analysis is performed on benchmark results
- HTML reports with charts are generated for easy visualization of results
- Concurrent execution is tested using Python's ThreadPoolExecutor