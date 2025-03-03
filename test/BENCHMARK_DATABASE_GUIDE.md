# Benchmark Database System Guide

## Overview

The Benchmark Database System is a comprehensive solution for storing, querying, and analyzing benchmark results in a structured and efficient manner. This system replaces the previous approach of storing results in individual JSON files, providing better performance, data consistency, and analytical capabilities.

The database system uses DuckDB as the underlying storage engine, with a Parquet-compatible format that allows for efficient querying and storage of benchmark data. This guide explains how to use the various components of the system to manage your benchmark data.

## System Components

The Benchmark Database System consists of the following components:

1. **Benchmark DB Converter** (`benchmark_db_converter.py`): Converts JSON files to the database format
2. **Benchmark DB Migration** (`benchmark_db_migration.py`): Comprehensive data migration pipeline
3. **Benchmark DB API** (`benchmark_db_api.py`): Programmatic and REST API for storing and querying results
4. **Benchmark DB Query** (`benchmark_db_query.py`): Command-line tool for querying and reporting
5. **Benchmark DB Updater** (`benchmark_db_updater.py`): Updates the database with new results
6. **Benchmark DB Maintenance** (`benchmark_db_maintenance.py`): Maintenance tasks like optimization and cleanup
7. **Benchmark DB Visualizer** (`benchmark_db_visualizer.py`): Visualization and reporting tools
8. **Benchmark DB Analytics** (`benchmark_db_analytics.py`): Advanced analytics and ML-based predictions
9. **Benchmark DB Models** (`benchmark_db_models.py`): ORM layer for database access
10. **Schema Definition** (`scripts/create_benchmark_schema.py`): Defines the database schema
11. **DB Integrated Runner** (`run_db_integrated_benchmarks.py`): Benchmark runner with database integration
12. **CI Benchmark Integrator** (`ci_benchmark_integrator.py`): Integrates CI artifacts with the database

## Getting Started

### Setting Up the Database

To create a new benchmark database:

```bash
# Create database with schema
python test/scripts/create_benchmark_schema.py --output ./benchmark_db.duckdb --sample-data

# Or use the converter to create from existing JSON files
python test/benchmark_db_converter.py --consolidate --categories performance hardware compatibility
```

### Migrating and Converting Existing Data

The database system provides comprehensive tools for migrating benchmark data from JSON files and other sources into the structured database format.

#### Basic Conversion with `benchmark_db_converter.py`

Use the Converter tool for direct JSON-to-database conversion:

```bash
# Convert files from a specific directory
python test/scripts/benchmark_db_converter.py --input-dir ./archived_test_results --output-db ./benchmark_db.duckdb

# Consolidate data from multiple directories
python test/scripts/benchmark_db_converter.py --consolidate --categories performance hardware compatibility

# Export to Parquet for external analysis
python test/scripts/benchmark_db_converter.py --input-dir ./archived_test_results --output-db ./benchmark_db.duckdb --parquet-dir ./benchmark_parquet
```

#### Advanced Migration with `benchmark_db_migration.py`

For more comprehensive migration needs, use the Migration tool which adds validation, tracking, and better error handling:

```bash
# Migrate all known result directories
python test/scripts/benchmark_db_migration.py --migrate-all --db ./benchmark_db.duckdb --validate

# Migrate specific directories with categorization
python test/scripts/benchmark_db_migration.py --input-dirs performance_results,archived_test_results --categories performance,hardware --db ./benchmark_db.duckdb

# Migrate individual files
python test/scripts/benchmark_db_migration.py --input-files ./results1.json,./results2.json --db ./benchmark_db.duckdb

# Migrate CI artifacts from build pipeline
python test/scripts/benchmark_db_migration.py --migrate-ci --artifacts-dir ./artifacts --db ./benchmark_db.duckdb

# Validate migrated data for consistency
python test/scripts/benchmark_db_migration.py --validate --fix-inconsistencies --db ./benchmark_db.duckdb

# Archive or remove migrated JSON files
python test/scripts/benchmark_db_migration.py --migrate-all --action archive --archive-dir ./archived_json --db ./benchmark_db.duckdb
```

The migration tool provides several advantages:

1. **Tracking**: Maintains a record of migrated files to prevent duplicates
2. **Validation**: Validates data consistency during and after migration
3. **Error Handling**: Better error recovery for partially valid files
4. **Categorization**: Intelligently categorizes data by content and structure
5. **CI Integration**: Special handling for CI/CD artifacts
6. **File Management**: Options to archive or remove migrated files
7. **Batched Processing**: Processes files in batches for better performance
8. **Parallel Processing**: Optional parallel processing for large migrations

#### Migration Statistics

After migration, you can generate detailed statistics about the migration process:

```bash
# Generate migration statistics
python test/scripts/benchmark_db_maintenance.py --migration-stats --output migration_report.json
```

This will produce a comprehensive report showing:
- Number of files migrated by category
- Total records imported
- Migration success/failure rates
- Timeline of migration activities
- Database table statistics after migration

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

The Database Maintenance tool (`benchmark_db_maintenance.py`) provides comprehensive maintenance capabilities:

```bash
# Database optimization and analysis
python test/scripts/benchmark_db_maintenance.py --optimize-db --vacuum --db ./benchmark_db.duckdb

# Check database integrity (foreign keys, orphaned records, etc.)
python test/scripts/benchmark_db_maintenance.py --check-integrity --db ./benchmark_db.duckdb

# Create a compressed backup
python test/scripts/benchmark_db_maintenance.py --backup --backup-dir ./db_backups --backup-compress

# Delete old database backups based on retention policy
python test/scripts/benchmark_db_maintenance.py --purge-backups --backup-retention 30 --backup-dir ./db_backups

# Generate detailed migration statistics
python test/scripts/benchmark_db_maintenance.py --migration-stats --output migration_report.json

# Clean up JSON files that have been migrated to the database
python test/scripts/benchmark_db_maintenance.py --clean-json --older-than 30 --action archive

# Archive old data from database to Parquet files
python test/scripts/benchmark_db_maintenance.py --archive-data --older-than 90 --archive-dir ./archived_data

# Perform multiple operations in one command
python test/scripts/benchmark_db_maintenance.py --optimize-db --vacuum --check-integrity --db ./benchmark_db.duckdb
```

The maintenance tool provides detailed reports of all operations performed, and can be configured to perform actions like archiving or removing files, compressing backups, and validating data consistency. It supports:

1. **Database optimization**: Analyzes tables and optimizes indexes for better query performance
2. **Database integrity checking**: Validates foreign key constraints, detects orphaned records, and finds inconsistencies
3. **Backup management**: Creates compressed backups and maintains a retention policy for old backups
4. **Migration statistics**: Generates detailed statistics about migrated files and imported records
5. **File management**: Cleans up JSON files that have been migrated, with options to archive or remove
6. **Data archiving**: Archives old data from the database to Parquet files while maintaining accessibility

For automated maintenance, you can schedule the maintenance tool to run periodically:

```bash
# Example cron job (run daily at 2 AM)
0 2 * * * cd /path/to/project && python test/scripts/benchmark_db_maintenance.py --optimize-db --vacuum --backup --purge-backups --backup-retention 30 --clean-json --older-than 30 --action archive --db ./benchmark_db.duckdb >> maintenance_log.txt 2>&1
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

### Using the ORM Layer

The preferred way to interact with the database is through the ORM layer:

```python
from benchmark_db_models import BenchmarkDB, TestRuns, PerformanceResults

# Connect to the database
db = BenchmarkDB(db_path="./benchmark_db.duckdb")

# Create a test run
test_run = TestRuns(
    test_name="model_benchmarks",
    test_type="performance",
    started_at=datetime.datetime.now(),
    success=True
)
run_id = db.insert_test_runs(test_run)

# Add a performance result
perf_result = PerformanceResults(
    run_id=run_id,
    model_id=db.get_or_add_model("bert-base-uncased"),
    hardware_id=db.get_or_add_hardware("cuda"),
    test_case="embedding",
    batch_size=32,
    precision="fp16",
    throughput_items_per_second=250.5,
    average_latency_ms=4.2,
    memory_peak_mb=2048.0
)
db.insert_performance_results(perf_result)
```

### Using the Database-Integrated Runners

For benchmark tests, you can use several database-integrated runners:

#### Using run_model_benchmarks.py (Migrated)

The primary model benchmark runner now supports direct database integration:

```bash
# Run benchmarks with direct database storage
python test/run_model_benchmarks.py --db-path ./benchmark_db.duckdb --models-set small --hardware cuda cpu

# Run and generate visualizations from the database
python test/run_model_benchmarks.py --db-path ./benchmark_db.duckdb --models-set key --visualize-from-db

# Run without database storage
python test/run_model_benchmarks.py --no-db-store
```

#### Using run_benchmark_with_db.py

The dedicated database benchmark runner provides optimized database integration:

```bash
# Run benchmarks with direct database storage
python test/run_benchmark_with_db.py --db-path ./benchmark_db.duckdb --models-set small --hardware cuda cpu

# Run and generate visualizations
python test/run_benchmark_with_db.py --db-path ./benchmark_db.duckdb --models-set key
python test/benchmark_db_visualizer.py --db ./benchmark_db.duckdb --report performance --output report.html
```

### Using the API Approach

Test runners can also use the API to store results:

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

### Generating Visualizations

Use the visualizer to create reports and charts:

```bash
# Generate HTML report
python test/benchmark_db_visualizer.py --db ./benchmark_db.duckdb --report performance --format html --output report.html

# Compare hardware platforms for a model
python test/benchmark_db_visualizer.py --db ./benchmark_db.duckdb --model bert-base-uncased --compare-hardware --output comparison.png

# Compare models on a specific hardware
python test/benchmark_db_visualizer.py --db ./benchmark_db.duckdb --hardware cuda --compare-models --output models_cuda.png

# Plot performance trends
python test/benchmark_db_visualizer.py --db ./benchmark_db.duckdb --model bert-base-uncased --hardware cuda --plot-trend --output trend.png
```

### CI/CD Integration

The database system is fully integrated with CI/CD pipelines through a dedicated GitHub Actions workflow (`benchmark_db_ci.yml`). This workflow automates benchmark execution, data storage, result consolidation, and report generation.

#### GitHub Actions Workflow Structure

The CI/CD workflow is fully implemented in `.github/workflows/benchmark_db_ci.yml` and consists of four main stages:

1. **Setup Database**: Creates and initializes the benchmark database
2. **Run Benchmarks**: Executes benchmarks for different models on various hardware platforms
3. **Consolidate Results**: Merges results from parallel benchmark runs
4. **Publish Results**: Generates reports and stores historical data

The workflow uses a matrix strategy to run multiple benchmarks in parallel:

```yaml
# Key components of the workflow
jobs:
  setup_database:
    # Creates the initial database and prepares CI metadata
    
  run_benchmarks:
    # Runs benchmarks in a matrix configuration
    strategy:
      matrix:
        model:
          - bert-base-uncased
          - t5-small
          - vit-base
        hardware:
          - cpu
          - cuda # Requires self-hosted runners for CUDA support
        batch_size: [1,2,4,8,16]
  
  consolidate_results:
    # Merges results from parallel jobs using ci_benchmark_integrator.py
    # Generates performance reports, compatibility matrices, and visualizations
  
  publish_results:
    # Publishes reports to GitHub Pages
    # Archives historical database snapshots for trend analysis
```

For local testing of the workflow, a convenience script `run_local_benchmark_with_ci.sh` is provided that simulates the CI/CD workflow on your local machine.

#### Running the CI/CD Pipeline

You can trigger the CI/CD pipeline manually or on push events:

```bash
# Trigger manually for specific model and hardware
gh workflow run benchmark_db_ci.yml --ref main -f test_model=bert-base-uncased -f hardware=cuda

# View results of the latest run
gh run list --workflow benchmark_db_ci.yml --limit 1

# Download artifacts from a specific run
gh run download <run-id> --name benchmark-reports
```

#### Integration with Test Runners

The CI/CD workflow is integrated with the following components:

1. **Direct Database Storage**: Benchmarks write results directly to the database
2. **Artifact Handling**: Database files are passed between workflow steps as artifacts
3. **Database Consolidation**: Results from parallel jobs are merged automatically
4. **Report Generation**: HTML and markdown reports are generated automatically
5. **Historical Tracking**: Databases are archived with timestamps for historical comparison

#### Performance Regression Detection

The CI/CD workflow includes automatic detection of performance regressions:

```bash
# In the CI pipeline
python test/scripts/benchmark_db_query.py --db consolidated_db/benchmark.duckdb --regression-check --threshold 10 --reference-db historical_db/benchmark_previous.duckdb

# For manual checks
python test/scripts/benchmark_db_query.py --db ./benchmark_db.duckdb --regression-check --last-days 30 --threshold 5
```

This allows for automatic detection of performance issues in new code changes.

### Advanced Analytics with benchmark_db_analytics.py

The Advanced Analytics tool provides sophisticated data analysis and machine learning capabilities:

```bash
# Analyze performance trends over time
python test/scripts/benchmark_db_analytics.py --db ./benchmark_db.duckdb --analysis performance-trends --output-dir ./analytics_output

# Compare hardware platforms across model families
python test/scripts/benchmark_db_analytics.py --db ./benchmark_db.duckdb --analysis hardware-comparison --model-family bert --output-dir ./analytics_output

# Compare model performance across hardware platforms
python test/scripts/benchmark_db_analytics.py --db ./benchmark_db.duckdb --analysis model-comparison --hardware-type cuda --output-dir ./analytics_output

# Predict performance for untested configurations
python test/scripts/benchmark_db_analytics.py --db ./benchmark_db.duckdb --analysis performance-prediction --prediction-features model_family,hardware_type,batch_size,precision --output-dir ./analytics_output

# Detect anomalies and performance regressions
python test/scripts/benchmark_db_analytics.py --db ./benchmark_db.duckdb --analysis anomaly-detection --regression-threshold 0.1 --output-dir ./analytics_output

# Analyze correlations between parameters and performance
python test/scripts/benchmark_db_analytics.py --db ./benchmark_db.duckdb --analysis correlation-analysis --output-dir ./analytics_output

# Run all analysis types
python test/scripts/benchmark_db_analytics.py --db ./benchmark_db.duckdb --analysis all --output-dir ./analytics_output

# Generate interactive HTML visualizations
python test/scripts/benchmark_db_analytics.py --db ./benchmark_db.duckdb --analysis hardware-comparison --interactive --output-dir ./analytics_output
```

Key features of the advanced analytics tool:

1. **Performance Trend Analysis**: Track model and hardware performance over time
2. **Hardware Comparison**: Advanced comparison of hardware platforms across model families
3. **Model Family Comparison**: Compare different model families on the same hardware
4. **Batch Scaling Analysis**: Understand how batch size affects performance metrics
5. **ML-Based Performance Prediction**: Use machine learning to predict performance for untested configurations
6. **Anomaly Detection**: Automatically detect performance anomalies and regressions
7. **Correlation Analysis**: Analyze relationships between parameters and performance metrics
8. **Interactive Visualizations**: Generate interactive HTML visualizations for exploration
9. **Feature Importance Analysis**: Understand what factors most affect performance

The analytics tool outputs comprehensive reports and visualizations in formats like PNG, PDF, SVG, or interactive HTML. These can be used to:

- Make data-driven decisions about hardware selection
- Understand performance trends over time
- Identify performance bottlenecks and opportunities for optimization
- Predict performance for untested configurations
- Detect and diagnose performance regressions

### CI Benchmark Integration

The CI Benchmark Integrator provides automated integration of benchmark results from CI/CD pipelines:

```bash
# Process CI artifacts and integrate into the database
python test/scripts/ci_benchmark_integrator.py --artifacts-dir ./artifacts --db ./benchmark_db.duckdb --commit 123abc --branch main

# Process with additional CI metadata
python test/scripts/ci_benchmark_integrator.py --artifacts-dir ./artifacts --db ./benchmark_db.duckdb --ci-metadata ./ci_metadata.json 

# Archive artifacts after processing
python test/scripts/ci_benchmark_integrator.py --artifacts-dir ./artifacts --db ./benchmark_db.duckdb --archive-artifacts --archive-dir ./archived_artifacts
```

The CI integration tool automatically:

1. Scans artifact directories for benchmark results (JSON, DuckDB, Parquet)
2. Processes and standardizes the data from different formats
3. Enriches with git and CI metadata (commit, branch, workflow)
4. Consolidates results from parallel jobs
5. Stores everything in the database with appropriate relationships
6. Optionally archives processed artifacts

#### Customizing the CI/CD Integration

You can customize the CI/CD workflow for your specific needs:

1. Modify the matrix to include different models, hardware platforms, or batch sizes
2. Add steps for custom analysis or notifications
3. Integrate with other CI systems by exporting results in various formats
4. Configure automatic creation of issues for performance regressions

#### Example CI/CD Configuration

Here's a complete example of running benchmarks in CI and storing results:

```bash
# Full CI pipeline in a shell script
# Initialize database
python test/scripts/create_benchmark_schema.py --output ci_benchmark.duckdb

# Run benchmarks for multiple models and store results directly
python test/run_benchmark_with_db.py --db ci_benchmark.duckdb --model bert-base-uncased --hardware cpu --batch-sizes 1,2,4,8
python test/run_benchmark_with_db.py --db ci_benchmark.duckdb --model t5-small --hardware cpu --batch-sizes 1,2,4,8

# Generate reports
python test/scripts/benchmark_db_query.py --db ci_benchmark.duckdb --report performance --format html --output benchmark_report.html

# Check for regressions
python test/scripts/benchmark_db_query.py --db ci_benchmark.duckdb --regression-check --threshold 10

# Archive the database
timestamp=$(date +%Y%m%d_%H%M%S)
cp ci_benchmark.duckdb ci_benchmark_${timestamp}.duckdb
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

## Current Status and Roadmap

### Current Status (March 2025)

The Benchmark Database System is now 100% complete with all planned components fully implemented:

- ✅ Database Schema Definition (100% complete)
- ✅ Data Converter and Migration Pipeline (100% complete)
- ✅ ORM Layer and Model Definitions (100% complete)
- ✅ Query and Visualization Tools (100% complete)
- ✅ Testing Framework Integration (100% complete)
- ✅ CI/CD Integration (100% complete)
- ✅ Database Maintenance and Backup (100% complete)
- ✅ Data Migration System with Validation (100% complete)
- ✅ Advanced Analytics and Reporting (100% complete)
- ✅ Core Benchmark Scripts Migration (100% complete)

### Completed Milestones

- **March 2, 2025**: Completed migration of all core benchmark scripts to DuckDB
- **March 2, 2025**: Completed comprehensive database maintenance tools
- **March 1, 2025**: Completed CI/CD integration with GitHub Actions
- **February 28, 2025**: Completed migration of all historical data
- **February 27, 2025**: Completed integration with all test runners
- **February 25, 2025**: Completed database schema and core components

### Recently Migrated Scripts

The following scripts have been successfully migrated to the DuckDB system:

1. **Core Benchmark Scripts**:
   - `run_model_benchmarks.py` (Primary model benchmarking tool)
   - `hardware_benchmark_runner.py`
   - `benchmark_all_key_models.py`
   - `run_benchmark_suite.py`

2. **Training-Related Benchmarking**:
   - `distributed_training_benchmark.py`
   - `training_mode_benchmark.py`
   - `training_benchmark_runner.py`

3. **Web Platform Benchmarking**:
   - `web_audio_test_runner.py`
   - `web_audio_platform_tests.py`
   - `web_platform_benchmark.py`
   - `web_platform_test_runner.py`

4. **Other Benchmarking Tools**:
   - `continuous_hardware_benchmarking.py`
   - `benchmark_hardware_performance.py`
   - `benchmark_hardware_models.py`
   - `model_benchmark_runner.py`

### Future Enhancements

With all the planned Phase 16 features now fully implemented, future enhancements could include:

1. **Machine Learning Integration**: ML-based performance prediction for untested model-hardware combinations
2. **Advanced Visualization Dashboard**: Interactive web dashboard with 3D visualizations and real-time updates
3. **Cloud Integration**: Support for cloud storage and multi-user access to benchmark data
4. **Distributed Analysis**: Distributed computing capabilities for analyzing very large datasets
5. **Automated Optimization Suggestions**: AI-powered recommendations for hardware selection and model optimization
6. **Integration with External Analytics Tools**: Connectors for tools like Tableau, PowerBI, and Grafana
7. **Advanced Regression Analysis**: Sophisticated algorithms for detecting subtle performance regressions with root cause analysis
8. **Real-time Benchmarking Pipeline**: Continuous benchmarking with instant feedback systems