# Refactored Benchmark Suite

This directory contains the implementation of a unified benchmark system for the IPFS Accelerate project. The goal of this refactoring is to create a single entry point for all benchmark operations, standardize result reporting, and simplify CI/CD integration.

## Directory Structure

- `benchmark_refactoring_plan.md`: Comprehensive plan for the benchmark refactoring
- `benchmark_ast_analyzer.py`: Script to analyze the AST of benchmark code
- `BENCHMARK_FASTAPI_DASHBOARD.md`: Comprehensive documentation for the API server and dashboard
- `benchmark_api_server.py`: FastAPI server implementation for benchmark operations
- `benchmark_api_client.py`: Python client for interacting with the API server
- `benchmark_dashboard.py`: Interactive dashboard for visualizing benchmark results
- `run_benchmark_api_server.sh`: Shell script for starting the API server
- `run_benchmark_dashboard.sh`: Shell script for starting the dashboard
- `benchmark_integration_example.py`: Example script demonstrating API integration
- `benchmark_core/`: Core framework implementation
  - `__init__.py`: Package exports
  - `registry.py`: Benchmark registry system
  - `base.py`: Base class for all benchmarks
  - `hardware.py`: Hardware detection and management
  - `results.py`: Result collection and storage
  - `runner.py`: Main entry point for running benchmarks
  - `db_integration.py`: Database integration for result storage

## Core Architecture

The refactored benchmark system is built around these key components:

1. **BenchmarkRegistry**: Central repository of all benchmark implementations
2. **BenchmarkBase**: Base class for all benchmark implementations
3. **BenchmarkRunner**: Unified entry point for executing benchmarks
4. **ResultsCollector**: Standardized result collection and storage
5. **HardwareManager**: Hardware detection and management

## Using the Framework

### Defining a New Benchmark

```python
from benchmark_core import BenchmarkBase, BenchmarkRegistry

@BenchmarkRegistry.register(
    name="model_inference",
    category="inference",
    models=["bert", "vit"],
    hardware=["cpu", "cuda", "webgpu"]
)
class ModelInferenceBenchmark(BenchmarkBase):
    """Benchmark for model inference performance."""
    
    def setup(self):
        """Set up benchmark environment."""
        # Load model, prepare data, etc.
        
    def execute(self):
        """Execute the benchmark."""
        # Run inference and collect metrics
        
    def process_results(self, raw_results):
        """Process raw benchmark results."""
        # Calculate metrics and return standardized results
```

### Running a Benchmark

```python
from benchmark_core import BenchmarkRunner

# Create runner
runner = BenchmarkRunner()

# Run a single benchmark
result = runner.execute("model_inference", {
    "hardware": "cuda",
    "model": "bert-base-uncased",
    "batch_size": 16
})

# Save results
results_path = runner.save_results()

# Generate report
report_path = runner.generate_report()
```

### Running a Benchmark Suite

```python
from benchmark_core import BenchmarkRunner

# Create runner
runner = BenchmarkRunner()

# Define suite configuration
suite = {
    "name": "Model Performance Suite",
    "description": "Performance benchmarks for various models",
    "benchmarks": [
        {
            "name": "model_inference",
            "params": {
                "hardware": "cuda",
                "model": "bert-base-uncased",
                "batch_size": 16
            }
        },
        {
            "name": "model_inference",
            "params": {
                "hardware": "cuda",
                "model": "vit-base-patch16-224",
                "batch_size": 8
            }
        }
    ]
}

# Run suite
suite_result = runner.execute_suite(suite)
```

### CI/CD Integration

The framework provides a CLI entry point for CI/CD integration:

```bash
python -m benchmark_core.runner --benchmark model_inference --hardware cuda --params '{"model": "bert-base-uncased", "batch_size": 16}'
```

Or with a suite configuration file:

```bash
python -m benchmark_core.runner --suite suites/performance_suite.json --compare previous_results.json
```

## AST Analysis

The `benchmark_ast_analyzer.py` script analyzes the AST of benchmark code to extract information about classes, methods, dependencies, and complexity. This analysis informs the refactoring process.

To run the analyzer:

```bash
python benchmark_ast_analyzer.py --root /path/to/benchmark/code --output analysis_results
```

The analyzer generates the following reports:

- `class_hierarchy.json`: Hierarchy of benchmark classes
- `function_signatures.json`: Signatures of benchmark functions
- `dependencies.json`: Dependencies between benchmark modules
- `complexity.json`: Code complexity metrics
- `summary.json`: Summary of analysis results
- `summary.md`: Markdown version of the summary report

## API Server and Interactive Dashboard

The benchmark framework now includes a comprehensive API server and interactive dashboard for benchmark control and visualization.

### FastAPI Server

To start the benchmark API server:

```bash
# Start with default settings
./run_benchmark_api_server.sh

# Start with custom settings
./run_benchmark_api_server.sh --port 8888 --db-path /path/to/benchmarks.duckdb --results-dir /path/to/results
```

The server provides these RESTful endpoints:

- `POST /api/benchmark/run` - Start a benchmark run with specified parameters
- `GET /api/benchmark/status/{run_id}` - Get status of a running benchmark
- `GET /api/benchmark/results/{run_id}` - Get results of a completed benchmark
- `GET /api/benchmark/models` - List available models for benchmarking
- `GET /api/benchmark/hardware` - List available hardware platforms
- `GET /api/benchmark/reports` - List available benchmark reports
- `GET /api/benchmark/query` - Query benchmark results with optional filters
- `WebSocket /api/benchmark/ws/{run_id}` - Real-time benchmark progress tracking

### Interactive Dashboard

To start the interactive dashboard for visualization:

```bash
# Start with default settings
./run_benchmark_dashboard.sh

# Start with custom settings
./run_benchmark_dashboard.sh --port 8050 --api-url http://localhost:8000 --db-path /path/to/benchmarks.duckdb
```

The dashboard provides the following features:

1. **Overview Tab**: High-level metrics and performance comparisons
2. **Comparison Tab**: Detailed performance comparisons with heatmap visualization
3. **Live Runs Tab**: Monitor and control active benchmark runs
4. **Reports Tab**: Access benchmark reports and run custom SQL queries

### Running an Example

To run an example benchmark and see the complete workflow:

```bash
# First, start the API server
./run_benchmark_api_server.sh

# In another terminal, run the example script
python benchmark_integration_example.py

# Start the dashboard to visualize results
./run_benchmark_dashboard.sh
```

For comprehensive documentation, see [BENCHMARK_FASTAPI_DASHBOARD.md](./BENCHMARK_FASTAPI_DASHBOARD.md).

## Design Principles

1. **Single Responsibility**: Each component has a clear, focused responsibility
2. **Extensibility**: Easy to add new benchmark types and storage backends
3. **Standardization**: Consistent interface and result format across benchmarks
4. **Discoverability**: Registry pattern for finding available benchmarks
5. **CI/CD Integration**: Simple command-line interface for automation