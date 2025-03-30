# Unified Benchmark Framework

## Project Overview

This project implements a comprehensive benchmark framework for the IPFS Accelerate project, providing a standardized way to measure and track performance across different hardware, models, and configurations.

## Key Components

### Core Framework

- `benchmark_core/registry.py`: Central registry for all benchmark implementations
- `benchmark_core/base.py`: Base class for all benchmark implementations
- `benchmark_core/hardware.py`: Hardware detection and management
- `benchmark_core/results.py`: Result collection, storage, and comparison
- `benchmark_core/runner.py`: Main entry point for executing benchmarks

### Analysis Tools

- `benchmark_ast_analyzer.py`: Script for analyzing the AST of benchmark code to inform refactoring decisions

### Examples

- `examples/model_benchmark.py`: Example model benchmark implementation
- `examples/performance_suite.json`: Example benchmark suite configuration
- `examples/ci_benchmark.py`: Example CI/CD integration script
- `examples/benchmark_workflow.yml`: Example GitHub Actions workflow configuration

## Architecture Design

The benchmark framework is built on these key design principles:

1. **Registry Pattern**: Benchmark implementations register themselves with metadata, enabling discovery and filtering
2. **Hardware Abstraction**: Common interface for different hardware backends (CPU, CUDA, MPS, WebGPU, etc.)
3. **Standardized Results**: Consistent format for benchmark results to enable comparison and trend analysis
4. **CI/CD Integration**: Simple command-line interface for automation in CI/CD pipelines
5. **Result Comparison**: Built-in support for comparing results against a baseline to detect regressions

## Benchmark Types

The framework supports various benchmark types:

1. **Model Benchmarks**: Performance benchmarks for ML models
   - Latency
   - Throughput
   - Memory usage

2. **Hardware Benchmarks**: Performance benchmarks for specific hardware features
   - Compute capabilities
   - Memory bandwidth
   - Hardware-specific optimizations

3. **Resource Pool Benchmarks**: Performance benchmarks for resource management
   - Concurrent model execution
   - Memory sharing
   - Error recovery

## Migration Strategy

The migration to the unified benchmark framework follows this approach:

1. **Analysis**: Analyze existing benchmark code to understand patterns and requirements
2. **Core Implementation**: Build the core framework components
3. **Gradual Migration**: Migrate existing benchmark code to the new framework
4. **CI/CD Integration**: Update CI/CD pipelines to use the new framework

## Directory Structure

```
refactored_benchmark_suite/
├── benchmark_refactoring_plan.md  # Comprehensive refactoring plan
├── benchmark_ast_analyzer.py      # AST analysis script
├── OVERVIEW.md                    # This file
├── README.md                      # Framework documentation
├── benchmark_core/                # Core framework implementation
│   ├── __init__.py
│   ├── registry.py
│   ├── base.py
│   ├── hardware.py
│   ├── results.py
│   └── runner.py
└── examples/                      # Example implementations
    ├── model_benchmark.py
    ├── performance_suite.json
    ├── ci_benchmark.py
    └── benchmark_workflow.yml
```

## Usage

### Defining a New Benchmark

```python
from benchmark_core import BenchmarkBase, BenchmarkRegistry

@BenchmarkRegistry.register(
    name="my_benchmark",
    category="inference",
    hardware=["cpu", "cuda"]
)
class MyBenchmark(BenchmarkBase):
    def setup(self):
        # Prepare benchmark environment
        
    def execute(self):
        # Execute benchmark and collect metrics
        
    def process_results(self, raw_results):
        # Process and format results
```

### Running a Benchmark

```python
from benchmark_core import BenchmarkRunner

runner = BenchmarkRunner()
result = runner.execute("my_benchmark", {"hardware": "cpu"})
runner.save_results()
```

### Running a Benchmark Suite

```bash
python ci_benchmark.py --suite performance_suite.json --output-dir ./results
```

## CI/CD Integration

The framework integrates with CI/CD pipelines to monitor performance:

1. **Daily Benchmarks**: Scheduled benchmarks to track performance over time
2. **PR Validation**: Performance validation for pull requests
3. **Regression Detection**: Alert on performance regressions
4. **Historical Tracking**: Store and analyze performance trends

## API Server and Dashboard Integration (Completed)

The benchmark framework now includes a comprehensive API server and interactive dashboard for visualization:

### FastAPI Server

- `benchmark_api_server.py`: RESTful API server with WebSocket support
- `run_benchmark_api_server.sh`: Launcher script for the API server
- API endpoints for benchmark operations, status monitoring, and results retrieval
- Real-time progress tracking via WebSockets
- DuckDB integration for result storage and querying

### Interactive Dashboard

- `benchmark_dashboard.py`: Web-based dashboard for visualization and analysis
- `run_benchmark_dashboard.sh`: Launcher script for the dashboard
- Interactive visualization of benchmark results with filtering and comparison
- Real-time monitoring of active benchmark runs
- Custom SQL query interface for advanced analysis

### Key Features

1. **API Integration**: RESTful endpoints for programmatic interaction
2. **WebSocket Support**: Real-time progress updates during benchmark execution
3. **Interactive Visualizations**: Heatmaps, charts, and tables for result analysis
4. **Resource-Aware Scheduling**: Optimized hardware utilization during benchmarking
5. **Database Integration**: Efficient storage and querying of benchmark results
6. **Electron Integration**: Designed for integration with desktop applications

For detailed documentation, see:
- `BENCHMARK_FASTAPI_DASHBOARD.md`: Comprehensive guide to the API server and dashboard

## Directory Structure (Updated)

```
refactored_benchmark_suite/
├── benchmark_refactoring_plan.md  # Comprehensive refactoring plan
├── benchmark_ast_analyzer.py      # AST analysis script
├── OVERVIEW.md                    # This file
├── README.md                      # Framework documentation
├── BENCHMARK_FASTAPI_DASHBOARD.md # API server and dashboard documentation
├── benchmark_api_server.py        # FastAPI server implementation
├── benchmark_api_client.py        # Client library for API interaction
├── benchmark_dashboard.py         # Interactive dashboard implementation
├── run_benchmark_api_server.sh    # Launcher script for API server
├── run_benchmark_dashboard.sh     # Launcher script for dashboard
├── benchmark_integration_example.py # Example script for API integration
├── benchmark_core/                # Core framework implementation
│   ├── __init__.py
│   ├── registry.py
│   ├── base.py
│   ├── hardware.py
│   ├── results.py
│   ├── runner.py
│   └── db_integration.py         # Database integration for results
└── examples/                      # Example implementations
    ├── model_benchmark.py
    ├── performance_suite.json
    ├── ci_benchmark.py
    └── benchmark_workflow.yml
```

## All Implementation Completed

All planned features of the benchmark framework have now been successfully implemented:

1. ✅ Core Framework Implementation: Base classes, registry, and runner
2. ✅ AST Analysis: Comprehensive analysis of existing benchmark code
3. ✅ Migration Strategy: Gradual migration of existing benchmark code
4. ✅ CI/CD Integration: Updated pipelines for automation
5. ✅ Database Integration: DuckDB integration for result storage
6. ✅ API Server: FastAPI implementation with WebSocket support
7. ✅ Interactive Dashboard: Web-based visualization and analysis

The benchmark framework is now fully functional and ready for production use. It provides a comprehensive solution for benchmark execution, result storage, analysis, and visualization.