# Benchmark Database Tools

This directory contains tools for maintaining, querying, and analyzing benchmark data for the IPFS Accelerate Python Framework as part of the Phase 16 database implementation.

## Overview

The tools use DuckDB and Parquet files to efficiently store, query, and analyze benchmark results from various tests. This provides significant advantages over the previous JSON-based approach:

- **Storage Efficiency**: 50-80% reduced storage requirements compared to JSON
- **Query Performance**: Fast SQL-based analytics on benchmark data
- **Programmability**: Python API and REST endpoints for programmatic access
- **Visualization**: Built-in charting and dashboard capabilities
- **Integration**: Seamless integration with existing test infrastructure

## Tools

### 1. Schema Creation

`create_benchmark_schema.py` - Defines and creates the database schema for storing benchmark results.

```bash
# Create the initial schema
./create_benchmark_schema.py --output ./benchmark_db.duckdb

# Generate sample data for testing
./create_benchmark_schema.py --output ./benchmark_db.duckdb --sample-data

# Force recreate existing tables
./create_benchmark_schema.py --output ./benchmark_db.duckdb --force
```

### 2. Data Conversion

`benchmark_db_converter.py` - Converts existing JSON test result files to the DuckDB/Parquet format.

```bash
# Convert all JSON files in default directories
./benchmark_db_converter.py --output-db ./benchmark_db.duckdb

# Convert files from a specific directory with category filtering
./benchmark_db_converter.py --input-dir ./performance_results --categories performance --output-db ./benchmark_db.duckdb

# Export to Parquet files as well as DuckDB
./benchmark_db_converter.py --output-db ./benchmark_db.duckdb --parquet-dir ./benchmark_parquet

# Dry run to see what would be converted without making changes
./benchmark_db_converter.py --output-db ./benchmark_db.duckdb --dry-run --verbose
```

### 3. Database Querying

`duckdb_api/core/benchmark_db_query.py` - Queries the database and generates reports from benchmark results.

```bash
# Execute a custom SQL query
./duckdb_api/core/benchmark_db_query.py --db ./benchmark_db.duckdb --sql "SELECT * FROM models LIMIT 10"

# Generate a performance report for a specific model family
./duckdb_api/core/benchmark_db_query.py --db ./benchmark_db.duckdb --report performance --family bert

# Generate a hardware compatibility matrix
./duckdb_api/core/benchmark_db_query.py --db ./benchmark_db.duckdb --compatibility-matrix

# Compare throughput across hardware platforms for a specific model
./duckdb_api/core/benchmark_db_query.py --db ./benchmark_db.duckdb --model bert-base-uncased --metric throughput --compare-hardware

# Export report to HTML
./duckdb_api/core/benchmark_db_query.py --db ./benchmark_db.duckdb --report summary --format html --output report.html

# Generate a chart comparing model performance
./duckdb_api/core/benchmark_db_query.py --db ./benchmark_db.duckdb --model bert-base-uncased --metric throughput --compare-hardware --format chart --output chart.png
```

### 4. Database Maintenance

`duckdb_api/core/benchmark_db_maintenance.py` - Performs maintenance operations on the database and related files.

```bash
# Clean up JSON files that have been migrated to the database (older than 30 days)
./duckdb_api/core/benchmark_db_maintenance.py --db ./benchmark_db.duckdb --clean-json --older-than 30

# Archive old JSON files instead of deleting
./duckdb_api/core/benchmark_db_maintenance.py --db ./benchmark_db.duckdb --clean-json --archive-data --archive-dir ./archived_json

# Optimize database tables and indexes
./duckdb_api/core/benchmark_db_maintenance.py --db ./benchmark_db.duckdb --optimize-db

# Reclaim space with VACUUM operation
./duckdb_api/core/benchmark_db_maintenance.py --db ./benchmark_db.duckdb --vacuum

# Perform a dry run to see what would happen
./duckdb_api/core/benchmark_db_maintenance.py --db ./benchmark_db.duckdb --clean-json --dry-run --verbose
```

### 5. API Server

`duckdb_api/core/benchmark_db_api.py` - Provides a REST API and web dashboard for accessing benchmark data.

```bash
# Start the API server
./duckdb_api/core/benchmark_db_api.py --db ./benchmark_db.duckdb --serve

# Start server on a different host/port
./duckdb_api/core/benchmark_db_api.py --db ./benchmark_db.duckdb --host 0.0.0.0 --port 8080 --serve

# Enable auto-reload for development
./duckdb_api/core/benchmark_db_api.py --db ./benchmark_db.duckdb --serve --reload --verbose
```

After starting the server, access:
- API documentation: http://localhost:8000/docs
- Interactive dashboard: http://localhost:8000/dashboard

## Installation

Install dependencies:

```bash
pip install -r requirements_db.txt
```

## Migration Process

The migration process from JSON files to the database involves the following steps:

1. **Create the schema**: Run `create_benchmark_schema.py` to set up the database structure
2. **Convert existing data**: Run `benchmark_db_converter.py` to migrate JSON files
3. **Verify data**: Use `duckdb_api/core/benchmark_db_query.py` to check that data was migrated correctly
4. **Update test runners**: Modify existing test runners to write directly to the database
5. **Clean up old files**: Use `duckdb_api/core/benchmark_db_maintenance.py` to remove or archive old JSON files

## Data Model

The database schema is organized around these core tables:

- **hardware_platforms**: Hardware device information
- **models**: Model metadata and properties
- **test_runs**: Test execution metadata
- **performance_results**: Performance benchmark measurements
- **hardware_compatibility**: Hardware-model compatibility results
- **integration_test_results**: Integration test outcomes
- **integration_test_assertions**: Test assertions and validations

For a complete schema reference, see comments in `create_benchmark_schema.py`.

## Python API Usage

The database can also be accessed programmatically:

```python
import duckdb
import pandas as pd

# Connect to the database
conn = duckdb.connect('./benchmark_db.duckdb')

# Query data
df = conn.execute("""
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
""").fetchdf()

# Analyze and visualize
pivot_df = df.pivot(index='model_name', columns='hardware_type', values='avg_throughput')
pivot_df.plot(kind='bar', figsize=(12, 6))
```