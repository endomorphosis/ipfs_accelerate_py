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
python test/scripts/create_new_database.py --db ./benchmark_db.duckdb --force

# Create a database with sample data
python test/scripts/create_new_database.py --db ./benchmark_db.duckdb --force

# Or convert existing data
python test/benchmark_db_converter.py --consolidate --categories performance hardware compatibility --output-db ./benchmark_db.duckdb
```

### 2. Migrating Historical Data

To migrate historical data, use the converter tool which has been enhanced to properly handle various data formats:

```bash
# Import data from specific directories
python test/benchmark_db_converter.py --input-dir ./performance_results --output-db ./benchmark_db.duckdb

# Consolidate data from multiple directories
python test/benchmark_db_converter.py --consolidate --categories performance hardware compatibility --output-db ./benchmark_db.duckdb

# Export to Parquet format for external analysis
python test/benchmark_db_converter.py --input-dir ./performance_results --output-parquet-dir ./benchmark_parquet

# Fix database issues when needed
python test/scripts/benchmark_db_fix.py --fix-all --db ./benchmark_db.duckdb

# Fix specific issues like timestamp errors
python test/scripts/benchmark_db_fix.py --fix-timestamps --db ./benchmark_db.duckdb
```

### 3. Querying the Database

The updated query tools provide capabilities for data analysis and visualization, and now work with both old and new schema formats:

```bash
# Execute SQL queries on the database
python test/benchmark_db_query.py --sql "SELECT * FROM performance_results" --format csv --output performance_data.csv --db ./benchmark_db.duckdb

# Generate HTML reports
python test/benchmark_db_query.py --report performance --format html --output benchmark_report.html --db ./benchmark_db.duckdb

# Compare hardware platforms for a specific model
python test/benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware --output hardware_comparison.png --db ./benchmark_db.duckdb

# Compare models on a specific hardware platform
python test/benchmark_db_query.py --hardware cuda --metric throughput --compare-models --db ./benchmark_db.duckdb

# Show tabular data directly in the terminal
python test/benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware --db ./benchmark_db.duckdb
```

### 4. Running Benchmarks with Database Integration

The database-integrated benchmark runner has been updated to handle various DuckDB versions:

```bash
# Run benchmarks with direct database storage
python test/run_benchmark_with_db.py --model bert-base-uncased --hardware cuda --batch-sizes 1,2,4 --db ./benchmark_db.duckdb

# Run simulated benchmarks for testing
python test/run_benchmark_with_db.py --model bert-base-uncased --hardware cpu --batch-sizes 1 --db ./benchmark_db.duckdb --simulate

# Run multiple hardware tests
python test/run_benchmark_with_db.py --model bert-base-uncased --hardware cuda --hardware cpu --batch-sizes 1 --db ./benchmark_db.duckdb
```

### 5. Database Maintenance

The database maintenance utilities have been improved to handle various issues:

```bash
# Validate database structure and integrity
python test/benchmark_db_maintenance.py --validate --db ./benchmark_db.duckdb

# Optimize database performance
python test/benchmark_db_maintenance.py --optimize --db ./benchmark_db.duckdb

# Create a backup of the database
python test/benchmark_db_maintenance.py --backup --backup-dir ./benchmark_backups --db ./benchmark_db.duckdb

# Generate a maintenance report
python test/benchmark_db_maintenance.py --report --report-file maintenance_report.json --db ./benchmark_db.duckdb
```

For serious issues that require fixing database structure:

```bash
# Create a completely new database with proper schema
python test/scripts/create_new_database.py --db ./benchmark_db_fixed.duckdb --force

# Apply fixes to an existing database
python test/scripts/benchmark_db_fix.py --fix-all --db ./benchmark_db.duckdb

# Fix only specific issues
python test/scripts/benchmark_db_fix.py --fix-timestamps --db ./benchmark_db.duckdb
python test/scripts/benchmark_db_fix.py --fix-web-platform --db ./benchmark_db.duckdb
```

For full documentation on database maintenance, see [BENCHMARK_DATABASE_GUIDE.md](BENCHMARK_DATABASE_GUIDE.md).

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

The migration is on schedule with major milestones completed in March 2025:

- ✅ **March 2, 2025**: Fix database access and integration issues
- ✅ **March 3, 2025**: Create robust database schema and fix timestamp handling
- ✅ **March 3, 2025**: Update converter for proper data handling
- ✅ **March 3, 2025**: Fix benchmark runner compatibility
- ✅ **March 3, 2025**: Update query tools to handle both old and new formats
- ⏱️ **March 15, 2025**: Complete cleanup of JSON files (in progress)

## Resources

For more information, refer to:

- [Benchmark Database Guide](BENCHMARK_DATABASE_GUIDE.md)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Parquet Format Specification](https://parquet.apache.org/docs/)