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

To migrate historical data, use one of the migration tools that can handle various data formats:

#### General Benchmark Data Migration

For general benchmark data, use the converter tool:

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

#### IPFS Test Results Migration

For IPFS-specific test results, use the specialized migration tool:

```bash
# Migrate IPFS test results from specific directories
python test/migrate_ipfs_test_results.py --input-dirs ./test_results ./archived_results

# Migrate and archive original files
python test/migrate_ipfs_test_results.py --input-dirs ./test_results --archive

# Migrate, archive, and generate a report
python test/migrate_ipfs_test_results.py --input-dirs ./test_results --archive --report

# Create an archive package of all processed files
python test/migrate_ipfs_test_results.py --input-dirs ./test_results --create-archive-package

# Migrate, archive, and delete original files after successful migration
python test/migrate_ipfs_test_results.py --input-dirs ./test_results --delete
```

For testing the IPFS migration tool:

```bash
# Create sample data and test migration
python test/test_ipfs_migration.py --all

# Create sample data only
python test/test_ipfs_migration.py --create-samples

# Run migration on existing sample data
python test/test_ipfs_migration.py --migrate

# Validate migration results
python test/test_ipfs_migration.py --validate
```

See [DATA_MIGRATION_README.md](DATA_MIGRATION_README.md) for complete documentation on the IPFS migration tool.

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

### Comprehensive Testing Tables

- **model_architecture_coverage**: Information about all 300+ HuggingFace architectures
- **hardware_compatibility_matrix**: Compatibility status across architectures and hardware
- **generator_improvements**: Tracking of generator improvements and their impact
- **architecture_metadata**: Details about each architecture including parameters, tasks, etc.
- **comprehensive_test_results**: Results from testing all architectures across hardware

### Views

- **latest_performance_metrics**: Latest performance metrics by model/hardware
- **model_hardware_compatibility**: Compatibility matrix across models and hardware
- **integration_test_status**: Summary of integration test status by component
- **comprehensive_coverage_summary**: Coverage percentage by model category and hardware
- **architecture_hardware_support**: Support status of each architecture on each hardware platform

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

## Database Storage Transition Complete

As of March 5, 2025, all benchmark tools have been fully migrated to use the DuckDB database system:

1. **Complete Migration**: All JSON files have been migrated to the database system
2. **JSON Deprecated**: JSON output has been officially deprecated and disabled by default
3. **Database-Only Storage**: All benchmark tools now store results directly in the database
4. **Archived Files**: Legacy JSON files have been archived for reference

> **Important Update (March 6, 2025)**: The transition phase is complete. JSON file generation has been deprecated and disabled by default (DEPRECATE_JSON_OUTPUT=1). All tools now use the database for storage and retrieval. The archived JSON files remain available for reference but are no longer actively used.
>
> **Critical Implementation Note**: No new files should be written to the `benchmark_results` directory. All benchmark results must be written directly to the DuckDB database. Any code that still writes JSON files to this directory should be updated to use the database API. See the "Writing Results to Database Instead of JSON Files" section in [BENCHMARK_DATABASE_GUIDE.md](BENCHMARK_DATABASE_GUIDE.md) for implementation details.

Example of database-only storage (current implementation):

```python
def store_test_results(results, json_path=None, use_db=True):
    # Store to database (now the default behavior)
    api = BenchmarkDBAPI()
    api.store_performance_result(**results)
    
    # JSON output is deprecated and disabled by default
    # Only used if explicitly requested and JSON deprecation is disabled
    if json_path and not os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes"):
        logger.warning("Using deprecated JSON output. Consider using database-only storage.")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
```

## Archiving Completed

The JSON file cleanup and archiving process has been completed as of March 5, 2025:

```bash
# All JSON files have been archived into compressed tar.gz files:
archived_json_files/
  - api_check_results_20250305.tar.gz
  - archived_test_results_20250305.tar.gz
  - benchmark_results_20250305.tar.gz
  - critical_model_results_20250305.tar.gz
  - hardware_fix_results_20250305.tar.gz

# The following categories of files have been archived:
# 1. JSON benchmark and test result files (now in DuckDB)
# 2. Backup (.bak) files from previous development iterations
# 3. Files in /archived_* directories
# 4. Legacy scripts replaced by the DuckDB-based system
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

## Migrating Comprehensive HuggingFace Testing Data

The database migration system has been extended to handle data from comprehensive HuggingFace model testing:

```bash
# Migrate comprehensive test results to the database
python test/benchmark_db_migration.py --migrate-comprehensive --db ./benchmark_db.duckdb

# Migrate specific model categories
python test/benchmark_db_migration.py --migrate-comprehensive --categories text_encoders,vision_models --db ./benchmark_db.duckdb

# Extract and store architecture metadata during migration
python test/benchmark_db_migration.py --migrate-comprehensive --extract-architecture-metadata --db ./benchmark_db.duckdb

# Analyze model-architecture coverage after migration
python test/benchmark_db_query.py --db ./benchmark_db.duckdb --report comprehensive-coverage --format html --output coverage_report.html
```

### Comprehensive Testing Data Structure

The database stores detailed information about all 300+ HuggingFace model architectures:

1. **Architecture Information**:
   - Architecture name and category
   - Model count and popularity metrics
   - Parameter counts and computational complexity
   - Task compatibility and special requirements

2. **Hardware Compatibility**:
   - Status on each hardware platform (compatible, mock, incompatible)
   - Implementation details and error categories
   - Performance scores and memory requirements
   - Optimization opportunities and limitations

3. **Generator Improvements**:
   - Changes made to test generators
   - Impact on coverage and compatibility
   - Error resolution patterns and statistics
   - Code patterns and reusable components

### Validation and Reporting

After migrating comprehensive test data:

```bash
# Validate architecture coverage data
python test/benchmark_db_maintenance.py --validate-architecture-coverage --db ./benchmark_db.duckdb

# Generate hardware compatibility matrix
python test/benchmark_db_query.py --db ./benchmark_db.duckdb --comprehensive-matrix --format html --output matrix.html

# Create improvement plan based on coverage gaps
python test/benchmark_db_query.py --db ./benchmark_db.duckdb --generate-improvement-plan --output plan.md
```

## Timeline and Milestones

The migration is on schedule with all major milestones completed in March 2025:

- ✅ **March 2, 2025**: Fix database access and integration issues
- ✅ **March 3, 2025**: Create robust database schema and fix timestamp handling
- ✅ **March 3, 2025**: Update converter for proper data handling
- ✅ **March 3, 2025**: Fix benchmark runner compatibility
- ✅ **March 3, 2025**: Update query tools to handle both old and new formats
- ✅ **March 5, 2025**: Extend schema for comprehensive HuggingFace model testing
- ✅ **March 5, 2025**: Migrate comprehensive test results to database
- ✅ **March 5, 2025**: Complete cleanup and archiving of JSON files

## Resources

For more information, refer to:

- [Benchmark Database Guide](BENCHMARK_DATABASE_GUIDE.md)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Parquet Format Specification](https://parquet.apache.org/docs/)