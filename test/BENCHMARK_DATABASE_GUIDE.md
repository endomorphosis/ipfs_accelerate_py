# Benchmark Database System Guide

## Overview

The Benchmark Database System is a comprehensive solution for storing, querying, and analyzing benchmark results in a structured and efficient manner. This system replaces the previous approach of storing results in individual JSON files, providing better performance, data consistency, and analytical capabilities.

The database system uses DuckDB as the underlying storage engine, with a Parquet-compatible format that allows for efficient querying and storage of benchmark data. This guide explains how to use the various components of the system to manage your benchmark data.

## System Components

The Benchmark Database System consists of the following components:

1. **Benchmark DB Converter** (`benchmark_db_converter.py`): Converts JSON files to the database format
2. **Benchmark DB API** (`benchmark_db_api.py`): Programmatic and REST API for storing and querying results
3. **Benchmark DB Query** (`benchmark_db_query.py`): Command-line tool for querying and reporting
4. **Benchmark DB Updater** (`benchmark_db_updater.py`): Updates the database with new results
5. **Benchmark DB Maintenance** (`benchmark_db_maintenance.py`): Maintenance tasks like optimization and cleanup
6. **Schema Definition** (`scripts/create_benchmark_schema.py`): Defines the database schema

## Getting Started

### Setting Up the Database

To create a new benchmark database:

```bash
# Create database with schema
python test/scripts/create_benchmark_schema.py --output ./benchmark_db.duckdb --sample-data

# Or use the converter to create from existing JSON files
python test/benchmark_db_converter.py --consolidate --categories performance hardware compatibility
```

### Converting Existing Data

To convert existing JSON files to the database:

```bash
# Convert files from a specific directory
python test/benchmark_db_converter.py --input-dir ./archived_test_results --output-db ./benchmark_db.duckdb

# Consolidate data from multiple directories
python test/benchmark_db_converter.py --consolidate --directories ./archived_test_results ./performance_results
```

### Storing New Results

To store new benchmark results directly:

```bash
# Start the API server
python test/benchmark_db_api.py --serve

# Programmatic usage
from benchmark_db_api import BenchmarkDBAPI
api = BenchmarkDBAPI()
api.store_performance_result(model_name="bert-base-uncased", hardware_type="cuda", throughput=123.4, latency_avg=10.5)
```

### Querying the Database

To query the database:

```bash
# Execute a SQL query
python test/benchmark_db_query.py --sql "SELECT model, hardware, AVG(throughput) FROM benchmark_performance GROUP BY model, hardware"

# Generate a report
python test/benchmark_db_query.py --report performance --format html --output benchmark_report.html

# Compare hardware platforms for a specific model
python test/benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware
```

### Updating the Database

To update the database with new results:

```bash
# Update from a single file
python test/benchmark_db_updater.py --input-file ./new_results.json

# Scan a directory for new files
python test/benchmark_db_updater.py --scan-dir ./new_results --incremental

# Process auto-store files from test runners
python test/benchmark_db_updater.py --auto-store
```

### Maintaining the Database

To perform maintenance tasks:

```bash
# Validate database structure and integrity
python test/benchmark_db_maintenance.py --validate

# Optimize the database
python test/benchmark_db_maintenance.py --optimize

# Create a backup
python test/benchmark_db_maintenance.py --backup

# Clean up old JSON files
python test/benchmark_db_maintenance.py --clean-json --older-than 30
```

## Database Schema

The benchmark database uses a structured schema with the following key tables:

### Dimension Tables

1. **hardware_platforms**: Stores information about hardware platforms
   - hardware_id (Primary Key)
   - hardware_type (CPU, CUDA, ROCm, MPS, OpenVINO, WebNN, WebGPU)
   - device_name
   - platform
   - memory_gb
   - other hardware specifications

2. **models**: Stores information about models
   - model_id (Primary Key)
   - model_name
   - model_family
   - modality
   - parameters_million
   - other model specifications

3. **test_runs**: Stores information about test runs
   - run_id (Primary Key)
   - test_name
   - test_type
   - started_at
   - completed_at
   - execution_time_seconds
   - success

### Data Tables

1. **performance_results**: Stores performance benchmark results
   - result_id (Primary Key)
   - run_id (Foreign Key)
   - model_id (Foreign Key)
   - hardware_id (Foreign Key)
   - test_case
   - batch_size
   - precision
   - throughput_items_per_second
   - average_latency_ms
   - memory_peak_mb
   - metrics (JSON)

2. **hardware_compatibility**: Stores hardware compatibility results
   - compatibility_id (Primary Key)
   - run_id (Foreign Key)
   - model_id (Foreign Key)
   - hardware_id (Foreign Key)
   - is_compatible
   - detection_success
   - initialization_success
   - error_message
   - error_type

3. **integration_test_results**: Stores integration test results
   - test_result_id (Primary Key)
   - run_id (Foreign Key)
   - test_module
   - test_class
   - test_name
   - status
   - hardware_id (Foreign Key)
   - model_id (Foreign Key)
   - error_message

## API Reference

### Benchmark DB API

The Benchmark DB API provides a programmable interface for storing and querying benchmark data.

#### REST API Endpoints

When running the API server (`benchmark_db_api.py --serve`), the following endpoints are available:

- `GET /`: API information
- `GET /health`: Health check
- `POST /performance`: Store performance result
- `GET /performance`: Get performance results
- `POST /compatibility`: Store compatibility result
- `GET /compatibility`: Get compatibility results
- `POST /integration`: Store integration test result
- `GET /integration`: Get integration test results
- `POST /query`: Execute a custom SQL query
- `GET /hardware`: Get hardware list
- `GET /models`: Get model list

#### Python API

```python
from benchmark_db_api import BenchmarkDBAPI

# Initialize API
api = BenchmarkDBAPI(db_path="./benchmark_db.duckdb")

# Store performance result
api.store_performance_result(
    model_name="bert-base-uncased",
    hardware_type="cuda",
    device_name="NVIDIA A100",
    batch_size=32,
    precision="fp16",
    throughput=250.5,
    latency_avg=4.2,
    memory_peak=2048.0
)

# Get performance metrics
df = api.get_performance_metrics(
    model_name="bert-base-uncased",
    hardware_type="cuda"
)

# Execute custom query
df = api.query(
    "SELECT model_name, hardware_type, AVG(throughput_items_per_second) FROM performance_results GROUP BY model_name, hardware_type"
)
```

## Integration with Testing Framework

The Benchmark Database System integrates with the existing testing framework to automatically store and analyze test results:

### Storing Test Results

Test runners can automatically store results in the database:

```python
from benchmark_db_api import BenchmarkDBAPI

# Run your test
result = run_performance_test(model_name="bert-base-uncased", hardware="cuda")

# Store the result
api = BenchmarkDBAPI()
api.store_performance_result(
    model_name=result["model"],
    hardware_type=result["hardware"],
    throughput=result["throughput"],
    latency_avg=result["latency"]
)
```

### Auto-Store Mode

For automatic storage of test results:

1. Configure test runners to save results to a designated directory
2. Run the updater in auto-store mode to process the results:

```bash
python test/benchmark_db_updater.py --auto-store --auto-store-dir ./auto_store_results
```

### CI/CD Integration

For integration with CI/CD pipelines:

1. Set up tests to generate result files in CI environment
2. Use the API or updater to store results in the database
3. Generate reports for comparison with historical results

```bash
# Example CI workflow
python test/run_model_benchmarks.py --output-json ci_benchmark_results.json
python test/benchmark_db_updater.py --input-file ci_benchmark_results.json
python test/benchmark_db_query.py --report performance --format md --output benchmark_report.md
```

## Best Practices

### Data Organization

- Use consistent naming conventions for models and hardware
- Include appropriate metadata with test results
- Maintain a consistent schema across different test types

### Performance Optimization

- Periodically run the optimizer to maintain performance
- Index frequently queried columns
- Clean up old and redundant data

### Data Migration

- Use the incremental mode when updating large datasets
- Validate the database after major updates
- Create backups before large-scale operations

### Query Performance

- Use views for common queries
- Limit result sets for large queries
- Prefilter data before complex aggregations

## Troubleshooting

### Common Issues

1. **Database Not Found**: Ensure the database file exists at the specified path
2. **Schema Mismatch**: Use the schema validation tools to check for inconsistencies
3. **Slow Queries**: Run the optimizer and check for inefficient query patterns
4. **Missing Data**: Check if the data was properly converted from JSON

### Error Codes

- **DB001**: Database connection failure
- **DB002**: Schema validation error
- **DB003**: Data conversion error
- **DB004**: Query execution error
- **DB005**: Database maintenance error

### Getting Help

For more information, refer to the following resources:

- [DuckDB Documentation](https://duckdb.org/docs/)
- [Benchmark DB API Documentation](./API_DOCUMENTATION.md)
- [Testing Framework Documentation](./TESTING_FRAMEWORK_README.md)

## Future Enhancements

The Benchmark Database System will continue to evolve with the following planned enhancements:

1. **Advanced Analytics**: Integration with data analysis tools like Pandas and NumPy
2. **Visualization Dashboard**: Interactive web dashboard for exploring benchmark results
3. **Predictive Modeling**: ML-based prediction of performance for untested configurations
4. **Distributed Storage**: Support for distributed database deployments
5. **Real-time Monitoring**: Real-time monitoring of benchmark results and trends