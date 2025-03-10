# DuckDB Benchmark API

This module provides a comprehensive system for storing, querying, and analyzing benchmark results for model-hardware combinations.

## Overview

The DuckDB Benchmark API replaces the previous JSON-based approach with a robust database-centric system that offers:

- **Database-First Storage**: All benchmark results stored directly in DuckDB
- **Incremental Benchmarking**: Only run missing or outdated benchmarks
- **Simulation Awareness**: Track which results are from real hardware vs. simulation
- **Comprehensive Reporting**: Generate reports and visualizations from the database
- **API Access**: Programmatic and REST API interfaces for all operations
- **Legacy Migration**: Tools to migrate existing JSON data to the database

This system ensures consistent, reliable storage and analysis of benchmark results across all model-hardware combinations.

## Directory Structure

```
duckdb_api/
├── core/                     # Core database functionality
│   ├── benchmark_db_api.py   # API interface for storing and querying benchmark data
│   ├── run_benchmark_with_db.py # Benchmark runner with database integration
│   └── ...
├── migration/                # Migration tools for JSON to database
│   ├── benchmark_db_converter.py # Converts JSON files to database format
│   └── ...
├── schema/                   # Database schema definitions
│   ├── create_benchmark_schema.py # Creates the database schema
│   └── ...
├── utils/                    # Utility functions for database operations
│   ├── benchmark_db_maintenance.py # Database maintenance utilities
│   ├── run_incremental_benchmarks.py # Smart incremental benchmark runner
│   ├── simulation_detection.py # Simulation detection and reporting
│   └── ...
└── visualization/            # Result visualization tools
    ├── benchmark_db_query.py # Database query and reporting tool
    ├── benchmark_visualizer.py # Visualization and chart generation
    └── ...
```

## Key Components

### Core API (`core/benchmark_db_api.py`)

The Core API provides programmatic and REST API access to the benchmark database:

```python
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI

# Initialize API
api = BenchmarkDBAPI(db_path="./benchmark_db.duckdb")

# Store performance result
result_id = api.store_performance_result(
    model_name="bert-base-uncased",
    hardware_type="cuda",
    batch_size=4,
    throughput=120.5,
    latency_avg=8.2
)

# Query performance metrics
df = api.get_performance_metrics(model_name="bert-base-uncased", hardware_type="cuda")
```

Start the REST API server:

```bash
python -m duckdb_api.core.benchmark_db_api --serve --host 0.0.0.0 --port 8000
```

### Benchmark Runner (`core/run_benchmark_with_db.py`)

The Benchmark Runner executes benchmarks and stores results directly in the database:

```python
from duckdb_api.core.run_benchmark_with_db import BenchmarkRunner

# Initialize runner
runner = BenchmarkRunner(db_path="./benchmark_db.duckdb")

# Run a single benchmark
result = runner.run_single_benchmark(
    model_name="bert-base-uncased",
    hardware_type="cuda",
    batch_size=4,
    sequence_length=128,
    use_simulation=False
)

# Store result in database
success = runner.store_benchmark_result(result)

# Run multiple benchmarks
summary = runner.run_benchmarks(
    model_names=["bert-base-uncased", "t5-small"],
    hardware_types=["cpu", "cuda"],
    batch_sizes=[1, 4, 16]
)
```

Command-line usage:

```bash
# Run benchmarks for specific models and hardware
python -m duckdb_api.core.run_benchmark_with_db --model bert-base-uncased --hardware cpu,cuda --batch-sizes 1,4,16

# Enable simulation for unavailable hardware
python -m duckdb_api.core.run_benchmark_with_db --model bert-base-uncased --hardware rocm --force-simulation

# Specify database path and output JSON file
python -m duckdb_api.core.run_benchmark_with_db --model bert-base-uncased --hardware cuda --db-path ./custom_benchmark.duckdb --output-json results.json
```

### Migration Tools (`migration/benchmark_db_converter.py`)

Convert existing JSON benchmark files to the database format:

```bash
python -m duckdb_api.migration.benchmark_db_converter --input-dir ./benchmark_results --output-db ./benchmark_db.duckdb
```

Consolidate data from multiple directories:

```bash
python -m duckdb_api.migration.benchmark_db_converter --consolidate --categories performance hardware compatibility
```

### Schema Management (`schema/create_benchmark_schema.py`)

Create or update the database schema:

```bash
python -m duckdb_api.schema.create_benchmark_schema --output ./benchmark_db.duckdb
```

Generate sample data for testing:

```bash
python -m duckdb_api.schema.create_benchmark_schema --output ./benchmark_db.duckdb --sample-data
```

### Utilities

#### Database Maintenance (`utils/benchmark_db_maintenance.py`)

Maintain the database:

```bash
# Check database integrity
python -m duckdb_api.utils.benchmark_db_maintenance --db-path ./benchmark_db.duckdb --check-integrity

# Optimize database
python -m duckdb_api.utils.benchmark_db_maintenance --db-path ./benchmark_db.duckdb --optimize-db

# Create backup
python -m duckdb_api.utils.benchmark_db_maintenance --db-path ./benchmark_db.duckdb --backup --backup-compress
```

#### Incremental Benchmarks (`utils/run_incremental_benchmarks.py`)

Run benchmarks incrementally, focusing only on missing or outdated benchmarks:

```bash
# Run missing benchmarks
python -m duckdb_api.utils.run_incremental_benchmarks --missing-only

# Run outdated benchmarks (older than 14 days)
python -m duckdb_api.utils.run_incremental_benchmarks --refresh-older-than 14

# Only run priority combinations
python -m duckdb_api.utils.run_incremental_benchmarks --priority-only

# Specify models and hardware
python -m duckdb_api.utils.run_incremental_benchmarks --models bert,t5,vit --hardware cpu,cuda
```

#### Simulation Detection (`utils/simulation_detection.py`)

Detect, mark, and validate simulated benchmark results:

```bash
# Update database schema for simulation tracking
python -m duckdb_api.utils.simulation_detection --update-schema

# Mark hardware as simulated
python -m duckdb_api.utils.simulation_detection --mark-hardware cuda --reason "Hardware not available"

# Generate simulation report
python -m duckdb_api.utils.simulation_detection --report --output simulation_report.json

# Get simulated results
python -m duckdb_api.utils.simulation_detection --get-simulated --output simulated.csv

# Run benchmarks with simulation awareness
python -m duckdb_api.utils.simulation_detection --run-benchmarks benchmark_configs.csv
```

The `run-benchmarks` feature reads a CSV file with benchmark configurations and runs them:

```csv
model_name,hardware_type,batch_size
bert-base-uncased,cpu,1
bert-base-uncased,cpu,4
bert-base-uncased,cuda,1
bert-base-uncased,cuda,4
```

### Visualization Tools

#### Database Querying (`visualization/benchmark_db_query.py`)

Query the database and generate reports:

```bash
# Execute SQL query
python -m duckdb_api.visualization.benchmark_db_query --sql "SELECT * FROM performance_results" --format csv --output output.csv

# Generate performance report
python -m duckdb_api.visualization.benchmark_db_query --report performance --format html --output report.html

# Compare hardware platforms for a model
python -m duckdb_api.visualization.benchmark_db_query --model bert-base-uncased --metric throughput --compare-hardware

# Compare models on a hardware platform
python -m duckdb_api.visualization.benchmark_db_query --hardware cuda --metric throughput --compare-models
```

#### Chart Generation (`visualization/benchmark_visualizer.py`)

Create visual charts for benchmark data:

```bash
# Hardware comparison chart
python -m duckdb_api.visualization.benchmark_visualizer --chart-type hardware-comparison --model bert-base-uncased --metric throughput --output hardware_comparison.png

# Model comparison chart
python -m duckdb_api.visualization.benchmark_visualizer --chart-type model-comparison --hardware cuda --metric throughput --output model_comparison.png

# Batch size comparison chart
python -m duckdb_api.visualization.benchmark_visualizer --chart-type batch-size-comparison --model bert-base-uncased --hardware cuda --metric throughput --output batch_comparison.png

# Hardware heatmap
python -m duckdb_api.visualization.benchmark_visualizer --chart-type heatmap --metric throughput --output heatmap.png

# Create comprehensive dashboard
python -m duckdb_api.visualization.benchmark_visualizer --chart-type dashboard --output dashboard_dir
```

## Database Schema

The benchmark database uses the following main tables:

- `models`: Information about AI models (name, family, parameters, etc.)
- `hardware_platforms`: Information about hardware platforms (type, device name, memory, etc.)
- `test_runs`: Information about test runs (name, type, git commit, start/end time, etc.)
- `performance_results`: Performance benchmark results (latency, throughput, memory usage, simulation status)
- `hardware_compatibility`: Hardware compatibility test results (compatibility status, errors, workarounds)
- `integration_test_results`: Integration test results (status, execution time, error messages)
- `hardware_availability_log`: Hardware availability tracking (detection timestamp, availability status, reason)

Key enhancements in the schema:

1. **Simulation Tracking Columns**:
   - `is_simulated`: Boolean flag indicating if the result is from a simulation
   - `simulation_reason`: Text description of why simulation was used

2. **Hardware Availability Logging**:
   - Tracks when hardware was available or unavailable
   - Records the reason for unavailability
   - Maintains history of hardware detection

3. **Comprehensive Metadata**:
   - All tables include metadata columns for additional information
   - Creation timestamps for tracking result age
   - JSON metadata fields for extensibility

## Integration with Existing Codebase

The benchmark database system integrates with the codebase in the following ways:

1. **Test Runners**: Test runners use the `BenchmarkRunner` to execute benchmarks and store results directly in the database, with full simulation awareness.
2. **Incremental Benchmarking**: The `IncrementalBenchmarkRunner` identifies missing or outdated benchmarks and uses `run_benchmark_with_db.py` to run only what's needed.
3. **Reporting Tools**: Reporting tools query the database using `benchmark_db_query.py` to generate comprehensive reports and visualizations.
4. **Legacy Migration**: Legacy JSON files can be migrated to the database using `benchmark_db_converter.py`.
5. **CI/CD Integration**: GitHub Actions workflows use `run_benchmark_with_db.py` to store test results directly in the database.
6. **Simulation Tracking**: The system tracks which results were collected on simulated vs real hardware using `simulation_detection.py`.
7. **Hardware Availability**: Hardware availability is logged in the database, enabling transparent result validation.

## Usage Examples

### Running Benchmarks with Database Integration

```python
from duckdb_api.core.run_benchmark_with_db import BenchmarkRunner

# Initialize the benchmark runner
runner = BenchmarkRunner(db_path="./benchmark_db.duckdb")

# Run comprehensive benchmarks
summary = runner.run_benchmarks(
    model_names=["bert-base-uncased", "t5-small", "vit-base"],
    hardware_types=["cpu", "cuda", "rocm"],
    batch_sizes=[1, 4, 16],
    sequence_length=128
)

print(f"Completed {summary['total']} benchmarks")
print(f"Successful: {summary['successful']}, Failed: {summary['failed']}")

# Run a single benchmark with simulation awareness
result = runner.run_single_benchmark(
    model_name="bert-base-uncased", 
    hardware_type="cuda", 
    batch_size=8,
    use_simulation=True  # Enable simulation if hardware not available
)

# Manually store a result if needed
runner.store_benchmark_result(result)
```

### Storing Results with the API

```python
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI

api = BenchmarkDBAPI()

# Store performance result
result_id = api.store_performance_result({
    "model_name": "bert-base-uncased",
    "hardware_type": "cuda",
    "device_name": "NVIDIA RTX 4090",
    "batch_size": 4,
    "precision": "fp16",
    "throughput": 162.5,
    "latency_avg": 24.6,
    "memory_peak": 2100.0
})
```

### Querying Benchmark Results

```python
import pandas as pd
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI

api = BenchmarkDBAPI()

# Get performance metrics for a model on different hardware
df = api.get_performance_metrics(model_name="bert-base-uncased")

# Compare different hardware platforms
comparison = api.get_performance_comparison(model_name="bert-base-uncased", metric="throughput")

# Custom SQL query
sql = """
SELECT 
    m.model_name, 
    hp.hardware_type, 
    AVG(pr.throughput_items_per_second) as avg_throughput
FROM 
    performance_results pr
JOIN 
    models m ON pr.model_id = m.model_id
JOIN 
    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
GROUP BY 
    m.model_name, hp.hardware_type
ORDER BY 
    avg_throughput DESC
"""
results = api.query(sql)
```

### Creating Visualizations

```python
from duckdb_api.visualization.benchmark_visualizer import BenchmarkVisualizer

visualizer = BenchmarkVisualizer()

# Create hardware comparison chart
visualizer.create_hardware_comparison_chart(
    model_name="bert-base-uncased",
    metric="throughput",
    output_file="hardware_comparison.png"
)

# Create model comparison chart
visualizer.create_model_comparison_chart(
    hardware_type="cuda",
    model_names=["bert-base-uncased", "t5-small", "vit-base"],
    metric="throughput",
    output_file="model_comparison.png"
)

# Create batch size comparison chart
visualizer.create_batch_size_comparison_chart(
    model_name="bert-base-uncased",
    hardware_type="cuda",
    metric="throughput",
    output_file="batch_size_comparison.png"
)

# Create hardware heatmap
visualizer.create_hardware_heatmap(
    model_names=["bert-base-uncased", "t5-small", "vit-base"],
    hardware_types=["cpu", "cuda", "rocm"],
    metric="throughput",
    output_file="heatmap.png"
)

# Create comprehensive dashboard
visualizer.create_dashboard(
    output_dir="dashboard",
    metrics=["throughput", "latency", "memory"],
    model_names=["bert-base-uncased", "t5-small", "vit-base"],
    hardware_types=["cpu", "cuda", "rocm"]
)
```

### Database Maintenance

```python
from duckdb_api.utils.benchmark_db_maintenance import BenchmarkDBMaintenance

maintenance = BenchmarkDBMaintenance()

# Check database integrity
maintenance.check_integrity()

# Optimize database
maintenance.optimize_db()

# Create backup
backup_path = maintenance.backup_db(compress=True)

# Get database statistics
stats = maintenance.get_database_stats()
```

### Simulation Detection

```python
from duckdb_api.utils.simulation_detection import SimulationDetection
import pandas as pd

detector = SimulationDetection()

# Mark hardware as simulated
detector.mark_hardware_as_simulated(
    hardware_type="cuda",
    reason="Hardware not available on CI server"
)

# Generate simulation report
report = detector.generate_simulation_report()

# Get all simulated results
simulated_df = detector.get_simulated_results()

# Run benchmarks with simulation awareness
benchmarks_df = pd.DataFrame([
    {"model_name": "bert-base-uncased", "hardware_type": "cpu", "batch_size": 1},
    {"model_name": "bert-base-uncased", "hardware_type": "cpu", "batch_size": 4},
    {"model_name": "bert-base-uncased", "hardware_type": "cuda", "batch_size": 1},
    {"model_name": "bert-base-uncased", "hardware_type": "cuda", "batch_size": 4}
])
success = detector.run_benchmarks(benchmarks_df)
```

### Incremental Benchmarking

The incremental benchmarking system identifies which benchmarks are missing or outdated and only runs those:

```python
from duckdb_api.utils.run_incremental_benchmarks import IncrementalBenchmarkRunner

# Initialize with database path
runner = IncrementalBenchmarkRunner(db_path="./benchmark_db.duckdb")

# Identify missing benchmarks
missing_df = runner.identify_missing_benchmarks(
    models=["bert-base-uncased", "t5-small"],
    hardware=["cpu", "cuda"],
    batch_sizes=[1, 4, 16]
)
print(f"Found {len(missing_df)} missing benchmark configurations")

# Identify outdated benchmarks
outdated_df = runner.identify_outdated_benchmarks(
    models=["bert-base-uncased", "t5-small"],
    hardware=["cpu", "cuda"],
    batch_sizes=[1, 4, 16],
    older_than_days=30
)
print(f"Found {len(outdated_df)} outdated benchmark configurations")

# Identify priority benchmarks for key models and hardware
priority_df = runner.identify_priority_benchmarks(
    priority_models=["bert-base-uncased", "t5-small", "vit-base"],
    priority_hardware=["cpu", "cuda", "openvino"]
)
print(f"Found {len(priority_df)} priority benchmark configurations")

# Run only the benchmarks that are needed
# This uses the run_benchmark_with_db.py module internally
success = runner.run_benchmarks(missing_df)  # or outdated_df or priority_df

# Real-world usage: Run all missing and outdated benchmarks
combined_df = pd.concat([missing_df, outdated_df]).drop_duplicates()
success = runner.run_benchmarks(combined_df)
```

Command-line usage:

```bash
# Run missing benchmarks for specific models and hardware
python -m duckdb_api.utils.run_incremental_benchmarks --models bert-base-uncased,t5-small --hardware cpu,cuda --missing-only

# Run outdated benchmarks (older than 14 days)
python -m duckdb_api.utils.run_incremental_benchmarks --refresh-older-than 14

# Run priority model-hardware combinations only
python -m duckdb_api.utils.run_incremental_benchmarks --priority-only

# Dry run (identify benchmarks but don't run them)
python -m duckdb_api.utils.run_incremental_benchmarks --dry-run --output missing_benchmarks.csv
```

## Best Practices

When working with the DuckDB Benchmark API, follow these best practices:

1. **Use Database for All Results Storage**:
   - Always store benchmark results directly in the database
   - Avoid creating new JSON result files
   - Use `run_benchmark_with_db.py` for running benchmarks

2. **Run Benchmarks Incrementally**:
   - Use `run_incremental_benchmarks.py` instead of re-running all benchmarks
   - This saves time and resources by only running what's needed
   - Query the database first to determine what benchmarks are missing or outdated

3. **Be Transparent About Simulation**:
   - Always track if results are from simulated hardware
   - When hardware isn't available, use the `--force-simulation` flag
   - Use `simulation_detection.py` to mark hardware as simulated with reasons

4. **Use Proper Database Paths**:
   - Set the `BENCHMARK_DB_PATH` environment variable for consistent database access
   - Or explicitly specify `--db-path` in all commands
   - Use absolute paths to avoid confusion

5. **Maintain the Database**:
   - Regularly run maintenance operations with `benchmark_db_maintenance.py`
   - Back up the database before major operations
   - Use `--check-integrity` to verify database health

6. **Integrate with CI/CD**:
   - Store CI/CD test results directly in the database
   - Use simulation flags when running in CI environments
   - Verify simulation status when generating reports

7. **Leverage the API for Integration**:
   - Use the REST API for integrating with other tools
   - The API provides a stable interface for database access
   - Enable CORS for web dashboard integration