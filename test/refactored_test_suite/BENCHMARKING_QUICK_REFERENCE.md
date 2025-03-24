# Performance Benchmarking Quick Reference Guide

This guide provides a quick reference for using the performance benchmarking system in the IPFS Accelerate Python framework.

## Setup

Before using the benchmarking system, set up the database schema:

```bash
# Set up the benchmark database
python -m benchmarking.setup_benchmark_db --db-path benchmark_db.duckdb

# Or, to overwrite an existing database
python -m benchmarking.setup_benchmark_db --db-path benchmark_db.duckdb --overwrite
```

## Running Single Model Benchmarks

```bash
# Basic benchmark on CPU
python -m benchmarking.run_hardware_benchmark --model-id bert-base-uncased --device cpu

# Benchmark on CUDA with custom settings
python -m benchmarking.run_hardware_benchmark \
  --model-id gpt2 \
  --device cuda \
  --precision float16 \
  --batch-sizes 1,2,4,8 \
  --sequence-lengths 128,256,512 \
  --iterations 20 \
  --save
```

## Batch Benchmarking

```bash
# Benchmark multiple models in batch
python -m benchmarking.batch_benchmark \
  --models bert-base-uncased,gpt2,t5-small \
  --devices cpu,cuda \
  --output-dir benchmark_results

# Or use a model list file
python -m benchmarking.batch_benchmark \
  --model-list benchmarking/model_list.txt \
  --devices cpu,cuda \
  --db-path benchmark_db.duckdb
```

## Visualizing Results

```bash
# Visualize a single benchmark result
python -m benchmarking.visualize_benchmarks \
  --json-path benchmark_results/bert-base-uncased_cuda_float32_20250410_120000.json

# Compare hardware performance for a model
python -m benchmarking.visualize_benchmarks \
  --db-path benchmark_db.duckdb \
  --model-id bert-base-uncased

# Generate architecture comparison
python -m benchmarking.visualize_benchmarks \
  --db-path benchmark_db.duckdb \
  --architecture-comparison
```

## Database Queries

Use DuckDB to query the benchmark database directly:

```python
import duckdb

# Connect to database
con = duckdb.connect("benchmark_db.duckdb")

# View benchmark runs
con.execute("SELECT * FROM benchmark_runs").fetchdf()

# Compare hardware performance
con.execute("""
SELECT 
    r.model_id, r.device, r.architecture_type, 
    b.latency_mean_ms, b.throughput_samples_per_sec, b.memory_usage_mb
FROM benchmark_runs r
JOIN benchmark_results b ON r.id = b.run_id
WHERE r.model_id = 'bert-base-uncased'
  AND b.batch_size = 1
  AND b.sequence_length = 128
ORDER BY b.throughput_samples_per_sec DESC
""").fetchdf()
```

## Integration with Test Suite

Run benchmarks through the comprehensive test suite:

```bash
# Benchmark a specific model
python run_comprehensive_test_suite.py \
  --benchmark \
  --benchmark-model bert-base-uncased \
  --benchmark-device cuda

# Batch benchmark multiple models
python run_comprehensive_test_suite.py \
  --batch-benchmark \
  --batch-benchmark-models bert-base-uncased,gpt2,t5-small \
  --batch-benchmark-devices cpu,cuda \
  --db-path benchmark_db.duckdb
```

## Hardware Support

The benchmarking system supports all 6 hardware backends:

- **CPU**: Universal fallback support for all models
- **CUDA**: Optimized support for NVIDIA GPUs
- **ROCm**: Support for AMD GPUs
- **MPS**: Support for Apple Silicon (M1/M2/M3)
- **OpenVINO**: Support for Intel CPUs, GPUs, and VPUs
- **QNN**: Support for Qualcomm Neural Network devices

## Metrics Collected

The benchmarking system collects the following metrics:

- **Latency**: Average, median, min, max, and 90th percentile latency in milliseconds
- **Throughput**: Samples processed per second
- **Memory Usage**: Peak and average memory usage in megabytes
- **First Token Latency**: Time to generate the first token (for generative models)
- **Load Time**: Time to load the model into memory

## Next Steps

For more detailed documentation, see:
- [benchmarking/README.md](benchmarking/README.md) - Complete benchmarking documentation
- [PERFORMANCE_BENCHMARKING_PLAN.md](PERFORMANCE_BENCHMARKING_PLAN.md) - Implementation plan and roadmap