# JSON to DuckDB Migration Guide

## Overview

As of March 2025, we have fully deprecated JSON output in the IPFS Accelerate Python Framework, and all benchmark scripts now use DuckDB as the primary storage format. This guide will help you transition to the new database-driven approach.

## Key Changes

1. **Default Behavior Changed**: All scripts now use `DEPRECATE_JSON_OUTPUT=1` by default
2. **Environment Variable**: Use `BENCHMARK_DB_PATH` to set the database location
3. **Command Line Arguments**: Scripts now use `--db-path` instead of output file arguments
4. **Scripts Updated**: All benchmark scripts have been updated to write directly to DuckDB

## Transitioning from JSON to DuckDB

### Setting Up Your Environment

Add this to your `.bashrc` or `.zshrc` file:

```bash
# Set the benchmark database path
export BENCHMARK_DB_PATH=/path/to/your/benchmark_db.duckdb
```

### Running Benchmarks

Instead of:
```bash
python generators/benchmark_generators/run_model_benchmarks.py --output-dir ./results --models bert
```

Now use:
```bash
python generators/benchmark_generators/run_model_benchmarks.py --models bert
```

The results will be automatically stored in the database specified by `BENCHMARK_DB_PATH`.

### Converting Existing JSON Files

If you have existing JSON files you'd like to migrate to the database:

```bash
python test/scripts/benchmark_db_converter.py --input-dir ./your_json_results
```

### Querying Results

To query results from the database:

```bash
# Get performance reports
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report performance --format html --output report.html

# Run SQL queries
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "SELECT * FROM performance_results LIMIT 10"

# Compare hardware platforms
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware
```

## Temporarily Reverting to JSON (Not Recommended)

If you need to temporarily revert to the old JSON-based output for backward compatibility:

```bash
# Set environment variable to 0
export DEPRECATE_JSON_OUTPUT=0

# Run with JSON output explicitly
python generators/benchmark_generators/run_model_benchmarks.py --models bert --output-dir ./json_results
```

Note that this approach is not recommended and is only provided for transition purposes.

## Advantages of the Database Approach

- **Reduced Storage**: 50-80% reduction in storage space compared to individual JSON files
- **Faster Queries**: 5-20x faster queries for complex analyses
- **Data Integrity**: Foreign key constraints ensure data consistency
- **Advanced Analytics**: SQL-based querying with join support
- **Visualization**: Integrated tools for data visualization

## Available Tools

- `duckdb_api/core/benchmark_db_api.py`: Programmatic API for storing and querying results
- `duckdb_api/core/benchmark_db_query.py`: Command-line tool for querying and reporting
- `benchmark_db_converter.py`: Converts JSON files to database format
- `duckdb_api/core/benchmark_db_maintenance.py`: Maintenance tasks for the database
- `benchmark_db_migration.py`: Comprehensive data migration tool

## Common Issues

### Database Not Found

If you see errors like "Database not found":

```bash
# Check your environment variable
echo $BENCHMARK_DB_PATH

# Create a new database if needed
python test/scripts/create_benchmark_schema.py --sample-data
```

### Missing Tables

If you see errors about missing tables:

```bash
# Recreate the database schema
python test/scripts/create_benchmark_schema.py --db-path $BENCHMARK_DB_PATH
```

### Performance Issues

If the database becomes slow:

```bash
# Optimize the database
python test/scripts/duckdb_api/core/benchmark_db_maintenance.py --optimize-db --vacuum
```

## Further Information

For more details, see:
- [BENCHMARK_DATABASE_GUIDE.md](BENCHMARK_DATABASE_GUIDE.md)
- [DATABASE_MIGRATION_GUIDE.md](DATABASE_MIGRATION_GUIDE.md)
- [PHASE16_DATABASE_IMPLEMENTATION.md](PHASE16_DATABASE_IMPLEMENTATION.md)