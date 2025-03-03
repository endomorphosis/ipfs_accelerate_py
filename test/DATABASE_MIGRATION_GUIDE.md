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

### Phase 2: Data Migration (Completed - 100%)

- ✅ Migrate historical performance data (100% complete)
- ✅ Migrate hardware compatibility data (100% complete)
- ✅ Migrate integration test results (100% complete)
- ✅ Validate migrated data for consistency (100% complete)
- ✅ Implement comprehensive data migration tool (100% complete)

### Phase 3: Tool Integration (Completed - 100%)

- ✅ Create ORM layer for database access (100% complete)
- ✅ Create basic query and reporting tools (100% complete)
- ✅ Implement data visualization components (100% complete)
- ✅ Update test runners to use database API (100% complete)
- ✅ Create adapter layer for legacy tools (100% complete)
- ✅ Implement database maintenance utilities (100% complete)

### Phase 4: CI/CD Integration (Completed - 100%)

- ✅ Update CI workflows to store results in database (100% complete)
- ✅ Implement automated regression detection (100% complete)
- ✅ Create historical comparison tools (100% complete)
- ✅ Develop dashboard for monitoring performance trends (100% complete)
- ✅ Create database backup and retention systems (100% complete)
- ✅ Implement GitHub Actions workflow for CI benchmarking (100% complete)
- ✅ Add CI/CD pipeline for historical performance tracking (100% complete)

## Migration Steps

### 1. Setting Up the Database

```bash
# Create a new database with schema
python test/scripts/create_benchmark_schema.py --output ./benchmark_db.duckdb

# Create a database with sample data for testing
python test/scripts/create_benchmark_schema.py --output ./benchmark_db.duckdb --sample-data

# Or convert existing data
python test/benchmark_db_converter.py --consolidate --directories performance_results archived_test_results --output-db ./benchmark_db.duckdb
```

### 2. Migrating Historical Data

Use the comprehensive migration tool for a complete migration with validation:

```bash
# Migrate all known result directories with validation
python test/scripts/benchmark_db_migration.py --migrate-all --db ./benchmark_db.duckdb --validate

# Migrate specific categories of data
python test/scripts/benchmark_db_migration.py --categories performance,hardware --db ./benchmark_db.duckdb

# Migrate CI artifacts from GitHub Actions
python test/scripts/benchmark_db_migration.py --migrate-ci --artifacts-dir ./artifacts --db ./benchmark_db.duckdb

# Archive processed files after migration
python test/scripts/benchmark_db_migration.py --migrate-all --action archive --archive-dir ./archived_json --db ./benchmark_db.duckdb

# For legacy migration approach:
python test/scripts/benchmark_db_converter.py --input-dir ./archived_test_results --output-db ./benchmark_db.duckdb
python test/scripts/benchmark_db_converter.py --input-dir ./performance_results --output-parquet-dir ./benchmark_parquet
```

### 3. Querying the Database

The enhanced query tools provide comprehensive capabilities for data analysis and visualization:

```bash
# Execute SQL queries on the database
python test/scripts/benchmark_db_query.py --sql "SELECT model_name, hardware_type, AVG(throughput_items_per_second) FROM performance_results JOIN models USING(model_id) JOIN hardware_platforms USING(hardware_id) GROUP BY model_name, hardware_type"

# Generate comprehensive HTML reports
python test/scripts/benchmark_db_query.py --report performance --format html --output performance_report.html
python test/scripts/benchmark_db_query.py --report hardware --format html --output hardware_report.html
python test/scripts/benchmark_db_query.py --report compatibility --format html --output compatibility_matrix.html

# Compare hardware platforms for a specific model
python test/scripts/benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware --output bert_hardware_comparison.png

# Compare models on a specific hardware platform
python test/scripts/benchmark_db_query.py --hardware cuda --metric throughput --compare-models --output cuda_model_comparison.png

# Plot performance trends over time
python test/scripts/benchmark_db_query.py --trend performance --model bert-base-uncased --hardware cuda --metric throughput --format chart

# Check for performance regressions
python test/scripts/benchmark_db_query.py --regression-check --threshold 10 --last-days 30

# Export data to various formats
python test/scripts/benchmark_db_query.py --sql "SELECT * FROM performance_results" --format csv --output performance_data.csv
```

### 4. Running Benchmarks with Database Integration

For direct database storage during benchmark runs:

```bash
# Run benchmarks with direct database storage
python test/run_benchmark_with_db.py --model bert-base-uncased --hardware cuda --batch-sizes 1,2,4,8,16

# Run benchmarks with advanced options
python test/run_benchmark_with_db.py --model t5-small --hardware cpu --precision fp32,fp16 --iterations 100 --warmup 10
```

### 4. Integrating with CI/CD

The database system now has a fully automated GitHub Actions workflow for CI/CD integration, implemented in `.github/workflows/benchmark_db_ci.yml`. To use this workflow:

```bash
# Run the CI/CD workflow manually via GitHub CLI
gh workflow run benchmark_db_ci.yml --ref main -f test_model=bert-base-uncased -f hardware=cpu -f batch_size=1,2,4,8

# Check the results of the latest workflow run
gh run list --workflow benchmark_db_ci.yml --limit 1

# Download workflow artifacts
gh run download <run-id> --name benchmark-reports
```

For local testing of the CI pipeline, you can use:

```bash
# Run local benchmarks with CI integration (simulates CI workflow locally)
./test/run_local_benchmark_with_ci.sh --model bert-base-uncased --hardware cpu --simulate

# Or use the individual commands for more control:
# Run benchmarks directly with database storage
python test/run_benchmark_with_db.py --model bert-base-uncased --hardware cpu --batch-sizes 1,2,4,8

# Generate performance report
python test/scripts/benchmark_db_query.py --report performance --format html --output benchmark_report.html

# Compare with previous run results
python test/scripts/benchmark_db_query.py --model bert-base-uncased --metric throughput --plot-trend --output trend.png
```

For full documentation on the CI/CD integration, see [BENCHMARK_CI_INTEGRATION.md](BENCHMARK_CI_INTEGRATION.md).

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

The complete migration has been successfully completed in March 2025, ahead of the originally planned schedule:

- ✅ **February 28, 2025**: Complete migration of historical data
- ✅ **March 1, 2025**: Complete integration with test runners
- ✅ **March 2, 2025**: Complete CI/CD integration
- ⏱️ **March 15, 2025**: Complete cleanup of JSON files (currently in progress)

## Resources

For more information, refer to:

- [Benchmark Database Guide](BENCHMARK_DATABASE_GUIDE.md)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Parquet Format Specification](https://parquet.apache.org/docs/)