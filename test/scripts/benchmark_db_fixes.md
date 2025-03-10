# Benchmark Database Fixes

The database system for storing benchmark results has been fixed with the following updates:

## 1. New Database Schema Script

The script `test/scripts/create_new_database.py` creates a brand new database with the correct schema, properly handling timestamps and all required tables.

Usage:
```bash
# Create a new database
python test/scripts/create_new_database.py --db ./benchmark_db.duckdb --force
```

## 2. Fix for DuckDB Compatibility Issues

The `run_benchmark_with_db.py` script has been fixed to handle different versions of DuckDB with varying parameter requirements.

## 3. Database Maintenance Tool

Updated the `benchmark_db_maintenance.py` utility to validate, optimize, and backup database files.

## 4. Migration Script

The `benchmark_db_fix.py` script can be used to fix issues with existing databases, including timestamp type errors and table structure problems.

## Next Steps and Best Practices

1. Use the create_new_database.py script to generate a fresh, properly structured database.
2. Use run_benchmark_with_db.py to populate the database with benchmark results.
3. Run regular validation and maintenance using benchmark_db_maintenance.py.
4. Consider backing up the database before major operations.

## Common Errors and Fixes

### Timestamp Type Errors

The most common issue is a type mismatch between INTEGER and TIMESTAMP. This is fixed in the new database schema.

### Views Don't Exist

Older versions of DuckDB had issues with the SHOW VIEWS command. The new schema handles this more gracefully.

### Unable to Transform Python Value

This error occurs when trying to directly insert pandas DataFrames into DuckDB. The updated converter handles this by properly preparing the data formats.

## Complete Fix Sequence

For a full reset and fresh start:

```bash
# Create backup of existing database
cp benchmark_db.duckdb benchmark_db.duckdb.bak

# Create fresh database with proper schema
python test/scripts/create_new_database.py --db ./benchmark_db.duckdb --force

# Run a simple benchmark to test
python test/run_benchmark_with_db.py --model bert-base-uncased --hardware cpu --batch-sizes 1 --simulate

# Validate database
python duckdb_api/core/benchmark_db_maintenance.py --validate --db ./benchmark_db.duckdb
```