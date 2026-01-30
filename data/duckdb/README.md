# IPFS Accelerate Python - DuckDB API

This directory contains all database-related code for the IPFS Accelerate Python framework, moved from the `/test` directory as part of the March 2025 reorganization.

## Directory Structure

- **core/**: Core database functionality
  - Database connection management
  - Query execution
  - Basic CRUD operations
  - Schema management
  
- **migration/**: Migration tools for JSON to database
  - JSON file converters
  - Data migration utilities
  - Schema migration tools
  - Backward compatibility
  
- **schema/**: Database schema definitions
  - Table definitions
  - Index configurations
  - Schema validation
  - Schema upgrade scripts
  
- **utils/**: Utility functions for database operations
  - Helper functions
  - Database backup and restore
  - Database integrity checks
  - Performance optimizations
  
- **visualization/**: Result visualization tools
  - Chart generation
  - Report generation
  - Interactive dashboards
  - Performance comparison tools

## Key Files

- **benchmark_db_api.py**: Main API for database operations
- **benchmark_db_query.py**: Tools for querying the benchmark database
- **benchmark_timing_report.py**: Generates comprehensive timing reports
- **benchmark_db_visualizer.py**: Creates visualizations from benchmark results
- **run_comprehensive_benchmarks.py**: Script for running benchmarks across models and hardware
- **benchmark_db_maintenance.py**: Database maintenance and optimization
- **migrate_all_json_files.py**: Migrates legacy JSON files to the database

## Usage Examples

### Benchmark Execution

```bash
# Run comprehensive benchmarks with the orchestration script
python duckdb_api/run_comprehensive_benchmarks.py

# Run benchmarks for specific models and hardware
python duckdb_api/run_comprehensive_benchmarks.py --models bert,t5,vit --hardware cpu,cuda

# Run with specific batch sizes
python duckdb_api/run_comprehensive_benchmarks.py --batch-sizes 1,4,16 --report-format markdown
```

### Database Queries

```bash
# Query the database with SQL
python duckdb_api/benchmark_db_query.py --sql "SELECT * FROM performance_results LIMIT 10"

# Generate reports
python duckdb_api/benchmark_db_query.py --report performance --format html --output report.html

# Compare hardware platforms
python duckdb_api/benchmark_db_query.py --model bert-base-uncased --compare-hardware --output comparison.png
```

### Report Generation

```bash
# Generate comprehensive benchmark timing report
python duckdb_api/benchmark_timing_report.py --generate --format html --output report.html

# Generate with different formats
python duckdb_api/benchmark_timing_report.py --generate --format markdown --output report.md

# Launch interactive dashboard
python duckdb_api/benchmark_timing_report.py --interactive
```

### Database Maintenance

```bash
# Migrate JSON files to database
python duckdb_api/migration/migrate_all_json_files.py --db-path ./benchmark_db.duckdb --archive

# Optimize the database
python duckdb_api/core/benchmark_db_maintenance.py --optimize-db --vacuum

# Create database backup
python duckdb_api/core/benchmark_db_maintenance.py --backup --backup-dir ./db_backups
```

## Related Documentation

- [Benchmark Database Guide](../test/BENCHMARK_DATABASE_GUIDE.md)
- [Database Migration Guide](../test/DATABASE_MIGRATION_GUIDE.md)
- [Benchmark Timing Report Guide](../test/BENCHMARK_TIMING_REPORT_GUIDE.md)
- [Benchmark JSON Deprecation Guide](../test/BENCHMARK_JSON_DEPRECATION_GUIDE.md)
- [Migration Guide](../test/MIGRATION_GUIDE.md)
