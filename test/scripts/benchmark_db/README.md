# Benchmark Database System

This directory contains the scripts for the Benchmark Database System, which provides efficient storage, querying, and analysis of benchmark results for the IPFS Accelerate Python Framework.

## Overview

The Benchmark Database System replaces the previous JSON-based approach with a DuckDB/Parquet database, offering improved storage efficiency, query performance, and data analysis capabilities. It is a core component of Phase 16 of the project, focusing on advanced hardware benchmarking and database consolidation.

## Components

### Core Components

- **`benchmark_db_api.py`**: RESTful API for storing and querying benchmark data
- **`benchmark_db_converter.py`**: Tool for converting JSON files to the database format
- **`benchmark_db_query.py`**: Query and reporting interface for benchmark data
- **`benchmark_db_updater.py`**: Updates the database with new benchmark results
- **`benchmark_db_maintenance.py`**: Maintenance tools for the database
- **`benchmark_db_migration.py`**: Migration utilities for transitioning from JSON to the database

### Installation

To install the dependencies for the Benchmark Database System:

```bash
pip install -r ../requirements_db.txt
```

## Usage

### Setting Up

1. Create the database schema:
   ```bash
   python create_benchmark_schema.py --output ./benchmark_db.duckdb --sample-data
   ```

2. Convert existing JSON files to the database:
   ```bash
   python benchmark_db_converter.py --input-dir ../../archived_test_results --output-db ./benchmark_db.duckdb
   ```

### Storing Data

1. Start the API server:
   ```bash
   python benchmark_db_api.py --serve
   ```

2. Use the API to store benchmark results:
   ```bash
   curl -X POST "http://localhost:8000/performance" \
        -H "Content-Type: application/json" \
        -d '{"model_name": "bert-base-uncased", "hardware_type": "cuda", "throughput": 123.4, "latency_avg": 10.5}'
   ```

### Querying Data

1. Execute a SQL query:
   ```bash
   python benchmark_db_query.py --sql "SELECT model, hardware, AVG(throughput) FROM benchmark_performance GROUP BY model, hardware"
   ```

2. Generate a report:
   ```bash
   python benchmark_db_query.py --report performance --format html --output benchmark_report.html
   ```

3. Compare hardware platforms for a specific model:
   ```bash
   python benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware
   ```

### Maintaining the Database

1. Validate the database structure:
   ```bash
   python benchmark_db_maintenance.py --validate
   ```

2. Optimize the database:
   ```bash
   python benchmark_db_maintenance.py --optimize
   ```

3. Clean up old JSON files:
   ```bash
   python benchmark_db_maintenance.py --clean-json --older-than 30
   ```

## Integration with Test Framework

The Benchmark Database System integrates with the existing test framework through:

1. **Direct API Calls**: Test runners can call the API directly to store results
2. **Auto-Store Mode**: Test runners can save results to a designated directory for automatic processing
3. **Dual Output**: Support for both database storage and traditional JSON output
4. **Adapters**: Legacy adapters for backward compatibility

## Documentation

For detailed documentation, see:

- [Benchmark Database Guide](../../BENCHMARK_DATABASE_GUIDE.md)
- [Database Migration Guide](../../DATABASE_MIGRATION_GUIDE.md)
- [Phase 16 Database Implementation](../../PHASE16_DATABASE_IMPLEMENTATION.md)

## Development

To contribute to the Benchmark Database System:

1. Set up a development environment:
   ```bash
   pip install -r ../requirements_db.txt
   ```

2. Run tests:
   ```bash
   python -m unittest discover -s test_benchmark_database
   ```

3. Generate Python ORM models for the database:
   ```bash
   python create_db_models.py --output benchmark_db_models.py
   ```