# DuckDB API Directory Structure

## Overview

The `duckdb_api` directory contains all database-related functionality for the IPFS Accelerate project, organized into subdirectories by functionality.

## Directory Layout

```
duckdb_api/
├── core/                     # Core database functionality
│   ├── benchmark_db_api.py
│   ├── benchmark_db_query.py
│   ├── benchmark_db_maintenance.py
│   ├── benchmark_db_fix.py
│   └── benchmark_with_db_integration.py
├── migration/                # Migration utilities
│   ├── benchmark_db_converter.py
│   ├── benchmark_db_migration.py
│   └── migrate_all_json_files.py
├── schema/                   # Schema management
│   ├── check_db_schema.py
│   ├── create_hardware_model_benchmark_database.py
│   ├── fix_benchmark_db_schema.py
│   ├── update_benchmark_db_schema.py
│   └── update_db_schema_for_simulation.py
├── utils/                    # Utilities
│   ├── cleanup_stale_reports.py
│   ├── benchmark_db_updater.py
│   ├── qnn_simulation_helper.py
│   ├── simulation_analysis.py
│   └── test_simulation_detection.py
└── visualization/            # Visualization
    ├── benchmark_db_visualizer.py
    ├── view_benchmark_data.py
    ├── view_benchmark_results.py
    └── view_performance_data.py
```

## Key Files

### Core API
- `core/benchmark_db_api.py`: REST API for database
- `core/benchmark_db_query.py`: Query interface
- `core/benchmark_db_maintenance.py`: Database maintenance
- `core/benchmark_with_db_integration.py`: Integration with benchmarks

### Migration
- `migration/benchmark_db_converter.py`: Convert JSON to DuckDB
- `migration/benchmark_db_migration.py`: Migrate between schema versions
- `migration/migrate_all_json_files.py`: Migrate all JSON files

### Schema Management
- `schema/check_db_schema.py`: Validate schema
- `schema/update_benchmark_db_schema.py`: Update schema
- `schema/update_db_schema_for_simulation.py`: Add simulation flags

### Utilities
- `utils/cleanup_stale_reports.py`: Clean up old reports
- `utils/qnn_simulation_helper.py`: QNN simulation utilities
- `utils/simulation_analysis.py`: Analyze simulation results

### Visualization
- `visualization/view_benchmark_results.py`: View benchmark results
- `visualization/benchmark_db_visualizer.py`: Visualize database data
- `visualization/view_performance_data.py`: View performance metrics
