# Database Migration Guide

## Overview

This guide provides a comprehensive roadmap for migrating from the current JSON-based approach to the new DuckDB/Parquet-based database system for storing benchmark results. The migration process is designed to be incremental, allowing you to transition your data while maintaining backward compatibility with existing tools and processes.

## Migration Benefits

Migrating to the new database system offers several key benefits:

1. **Improved Efficiency**: 50-80% reduction in storage space compared to JSON files
2. **Faster Queries**: SQL-based querying for rapid analysis of benchmark data
3. **Data Consistency**: Structured schema ensures data integrity and consistency
4. **Advanced Analytics**: Better support for time-series analysis and cross-metric comparisons
5. **Reduced Redundancy**: Elimination of duplicate data across multiple JSON files
6. **Integration**: Seamless integration with existing test runners and analysis tools

## Migration Phases

The migration is organized into the following phases:

### Phase 1: Database Setup (Completed)

- ✅ Create database schema definition
- ✅ Implement converter for existing JSON files
- ✅ Develop database API for programmatic access
- ✅ Build query tools for data analysis

### Phase 2: Data Migration (In Progress - 25%)

- ⏱️ Migrate historical performance data (50% complete)
- ⏱️ Migrate hardware compatibility data (25% complete)
- ⏱️ Migrate integration test results (10% complete)
- ⏱️ Validate migrated data for consistency (15% complete)

### Phase 3: Tool Integration (In Progress - 10%)

- ⏱️ Update test runners to use database API (20% complete)
- ⏱️ Enhance reporting tools for database integration (15% complete)
- ⏱️ Create adapter layer for legacy tools (5% complete)
- ⏱️ Develop visualization components for database queries (0% complete)

### Phase 4: CI/CD Integration (Planned - 5%)

- ⏱️ Update CI workflows to store results in database (10% complete)
- ⏱️ Implement automated regression detection (0% complete)
- ⏱️ Create historical comparison tools (5% complete)
- ⏱️ Develop dashboard for monitoring performance trends (0% complete)

## Migration Steps

### 1. Setting Up the Database

```bash
# Create a new database with schema
python test/scripts/create_benchmark_schema.py --output ./benchmark_db.duckdb

# Or convert existing data
python test/benchmark_db_converter.py --consolidate --categories performance hardware compatibility
```

### 2. Migrating Historical Data

```bash
# Convert files from archived results
python test/benchmark_db_converter.py --input-dir ./archived_test_results --output-db ./benchmark_db.duckdb

# Convert performance results
python test/benchmark_db_converter.py --input-dir ./performance_results --output-db ./benchmark_db.duckdb

# Convert hardware compatibility results
python test/benchmark_db_converter.py --input-dir ./hardware_compatibility_reports --output-db ./benchmark_db.duckdb

# Consolidate all results
python test/benchmark_db_converter.py --consolidate --deduplicate
```

### 3. Updating Test Runners

To update your test runners to use the database API:

```python
# Import the API
from benchmark_db_api import BenchmarkDBAPI

# Initialize API
api = BenchmarkDBAPI()

# Store test results
def run_test_with_db_storage(model_name, hardware_type):
    # Run your test
    result = run_performance_test(model_name, hardware_type)
    
    # Store the result
    api.store_performance_result(
        model_name=model_name,
        hardware_type=hardware_type,
        throughput=result["throughput"],
        latency_avg=result["latency"],
        memory_peak=result["memory"]
    )
    
    return result
```

### 4. Integrating with CI/CD

For CI/CD integration, add these steps to your workflows:

```bash
# Run tests and store results
python test/run_model_benchmarks.py --db-storage

# Or run tests with JSON output and then convert
python test/run_model_benchmarks.py --output-json ci_results.json
python test/benchmark_db_updater.py --input-file ci_results.json

# Generate performance report
python test/benchmark_db_query.py --report performance --format html --output benchmark_report.html
```

## Database Schema

The database schema is designed to be comprehensive yet flexible:

### Core Tables

- **hardware_platforms**: Information about hardware platforms
- **models**: Information about ML models
- **test_runs**: Information about test executions

### Data Tables

- **performance_results**: Performance benchmark results
- **hardware_compatibility**: Hardware compatibility test results
- **integration_test_results**: Integration test results
- **performance_batch_results**: Detailed batch-level performance data

### Views

- **latest_performance_metrics**: Latest performance metrics by model/hardware
- **model_hardware_compatibility**: Compatibility matrix across models and hardware
- **integration_test_status**: Summary of integration test status by component

## Migration Strategies

### Incremental Migration

For large datasets, use incremental migration:

```bash
# Identify already processed files
python test/benchmark_db_updater.py --scan-dir ./archived_test_results --track-processed

# Process new files incrementally
python test/benchmark_db_updater.py --scan-dir ./new_results --incremental
```

### Parallel Processing

For faster migration of large datasets:

```bash
# Process different directories in parallel
python test/benchmark_db_converter.py --input-dir ./dir1 --output-db ./benchmark_db.duckdb &
python test/benchmark_db_converter.py --input-dir ./dir2 --output-db ./benchmark_db.duckdb &
python test/benchmark_db_converter.py --input-dir ./dir3 --output-db ./benchmark_db.duckdb &

# Consolidate afterward
python test/benchmark_db_converter.py --consolidate --deduplicate
```

### Validation

To validate migrated data:

```bash
# Validate database structure and integrity
python test/benchmark_db_maintenance.py --validate

# Compare migrated data with original JSON
python test/validate_migration.py --json-dir ./archived_test_results --db ./benchmark_db.duckdb

# Generate validation report
python test/validate_migration.py --report
```

## Maintaining Backward Compatibility

During migration, backward compatibility is maintained through:

1. **Dual Output Mode**: Test runners can output both to JSON and the database
2. **JSON Bridge**: Tools to convert between database and JSON formats
3. **Legacy Adapters**: Adapter functions for legacy tools expecting JSON files

Example of dual output mode:

```python
def store_test_results(results, json_path=None, use_db=True):
    # Store to database if enabled
    if use_db:
        api = BenchmarkDBAPI()
        api.store_performance_result(**results)
    
    # Store to JSON if path provided
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
```

## Cleanup After Migration

After successful migration and validation:

```bash
# Clean up old JSON files (dry run first)
python test/benchmark_db_maintenance.py --clean-json --dry-run

# Clean up if everything looks good
python test/benchmark_db_maintenance.py --clean-json --older-than 30
```

## Troubleshooting

### Common Migration Issues

1. **Schema Compatibility**: If you encounter schema compatibility issues:
   ```bash
   python test/scripts/validate_benchmark_data.py --fix
   ```

2. **Duplicate Data**: If you have duplicate data:
   ```bash
   python test/benchmark_db_converter.py --deduplicate
   ```

3. **Performance Issues**: If database performance degrades:
   ```bash
   python test/benchmark_db_maintenance.py --optimize
   ```

### Data Recovery

In case of migration issues:

```bash
# Restore from backup
python test/benchmark_db_maintenance.py --restore ./benchmark_backups/benchmark_db_20250301_120000.duckdb

# Export data back to JSON if needed
python test/benchmark_db_query.py --export performance --format json --output ./exported_performance.json
```

## Timeline and Milestones

The complete migration is scheduled to be completed by April 2025, with the following milestones:

- **March 15, 2025**: Complete migration of historical data
- **March 31, 2025**: Complete integration with test runners
- **April 15, 2025**: Complete CI/CD integration
- **April 30, 2025**: Complete cleanup of JSON files

## Resources

For more information, refer to:

- [Benchmark Database Guide](BENCHMARK_DATABASE_GUIDE.md)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Parquet Format Specification](https://parquet.apache.org/docs/)