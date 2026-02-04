# Skillset Benchmarks

This directory contains benchmark implementations for the IPFS Accelerate skillset modules.

## Available Benchmarks

### Skillset Inference Benchmark

This benchmark measures the initialization time for skillset models across different hardware devices and batch sizes.

Key metrics:
- Import time
- Instantiation time
- Model initialization time
- Batch size scaling

### Skillset Throughput Benchmark

This benchmark measures the throughput of concurrent skillset model initialization.

Key metrics:
- Concurrent model loading throughput
- Speedup over sequential execution
- Resource sharing efficiency

## Usage

You can run these benchmarks using the following command:

```bash
python run_skillset_benchmark.py --type inference --hardware cpu --model bert
```

### Common Options

- `--type`: Type of benchmark to run (`inference` or `throughput`)
- `--hardware`: Hardware to benchmark on (`cpu`, `cuda`, `rocm`, `openvino`, `mps`, `qnn`)
- `--model`: Model to benchmark. Use `all` to benchmark all models
- `--batch-sizes`: Comma-separated list of batch sizes (default: "1,2,4,8")

### Additional Options

- `--random-sample`: Use random sample of models when `all` is specified
- `--sample-size`: Number of models to sample when using random sampling (default: 10)
- `--concurrent-models`: Number of concurrent models for throughput benchmark (default: 3)
- `--warmup-runs`: Number of warmup runs (default: 2)
- `--measurement-runs`: Number of measurement runs (default: 5)
- `--output-dir`: Directory for benchmark results (default: "benchmark_results")
- `--report`: Generate HTML report of benchmark results

## Example Commands

### Benchmark a specific model on CPU

```bash
python run_skillset_benchmark.py --type inference --hardware cpu --model bert
```

### Benchmark all models with random sampling

```bash
python run_skillset_benchmark.py --type inference --hardware cpu --model all --random-sample --sample-size 5
```

### Run throughput benchmark for multiple concurrent models

```bash
python run_skillset_benchmark.py --type throughput --hardware cpu --model all --random-sample --sample-size 5 --concurrent-models 3
```

### Generate HTML report

```bash
python run_skillset_benchmark.py --type inference --hardware cpu --model bert --report
```

## Integration with Benchmark Framework

These benchmarks are integrated with the unified benchmark framework and can be run through the standard `BenchmarkRunner` interface.

```python
from benchmark_core import BenchmarkRunner

runner = BenchmarkRunner()
result = runner.execute("skillset_inference_benchmark", {
    "hardware": "cpu",
    "model": "bert",
    "batch_sizes": [1, 2, 4, 8]
})
```