# Benchmark Generators

This directory contains tools for generating benchmarks for AI models across different hardware platforms.

## Purpose

These benchmark generators enable:

1. Consistent cross-platform performance testing
2. Comparison of hardware acceleration options
3. Performance analysis with visualizations
4. Integration with database reporting

## Available Generators

- **benchmark_generator.py**: Core benchmark generation logic
- Additional utilities for specific benchmark types

## Usage

```python
from generators.benchmark_generators.benchmark_generator import BenchmarkGenerator

# Initialize the generator
generator = BenchmarkGenerator(db_path="benchmark_db.duckdb")

# Generate a benchmark for a specific model and hardware
benchmark = generator.generate_benchmark(
    model_name="bert-base-uncased",
    hardware="cuda",
    batch_sizes=[1, 2, 4, 8, 16]
)

# Run the benchmark
results = benchmark.run()

# Store results in database
generator.store_results(results)
```

## Integration

These generators integrate with:

- DuckDB database for result storage and analysis
- Visualization tools for performance reporting
- Template system for hardware-aware benchmark generation