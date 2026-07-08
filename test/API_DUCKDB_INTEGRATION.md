# API DuckDB Integration

This document provides an overview of the DuckDB integration for the API components of the IPFS Accelerate project. These integrations enable persistent storage, historical tracking, and advanced querying for the Test Suite and Generator APIs.

## Overview

The DuckDB integration provides the following capabilities across API components:

1. **Persistent Storage**: Store run and task information in a robust database
2. **Historical Data**: Maintain a history of operations for analysis
3. **Performance Metrics**: Track and analyze performance across models
4. **Advanced Querying**: SQL-based querying for insights and reporting
5. **RESTful API**: Consistent API access to database information

## Components

The DuckDB integration is implemented for the following API components:

### Test Suite API

- **Database Path**: `/test/refactored_test_suite/database/`
- **Documentation**: `/test/refactored_test_suite/database/README.md`
- **Default DB File**: `./data/test_runs.duckdb`
- **API Base URL**: `http://localhost:8000/api/test/db/`

### Generator API

- **Database Path**: `/test/refactored_generator_suite/database/`
- **Documentation**: `/test/GENERATOR_DUCKDB_INTEGRATION.md`
- **Default DB File**: `./data/generator_tasks.duckdb`
- **API Base URL**: `http://localhost:8001/api/generator/db/`

## Common Architecture

Both integrations follow a consistent three-layer architecture:

1. **Database Handler Layer**: Direct database operations (db_handler.py)
2. **Integration Layer**: Bridge between API and database (db_integration.py)
3. **API Endpoints Layer**: FastAPI endpoints for database access (api_endpoints.py)

## Database Schema

### Test Suite Schema

1. **test_runs**: Main table for test run information
2. **test_run_results**: Detailed test results
3. **test_run_steps**: Detailed progress tracking
4. **test_run_metrics**: Performance and resource metrics
5. **test_run_tags**: Custom tags for test runs
6. **batch_test_runs**: Batch test run information
7. **test_run_batch**: Relations between test runs and batches
8. **test_run_artifacts**: Files produced during testing

### Generator Schema

1. **generator_tasks**: Main table for task information
2. **generator_task_steps**: Detailed progress tracking
3. **generator_task_metrics**: Performance and resource metrics
4. **generator_task_tags**: Custom tags for tasks
5. **generator_batch_tasks**: Batch task information

## API Endpoint Patterns

Both integrations provide consistent API endpoint patterns:

- `GET /<component>/db/runs` or `/<component>/db/tasks`: List history
- `GET /<component>/db/batches`: List batch history
- `GET /<component>/db/models/stats`: Get model statistics
- `GET /<component>/db/run/{id}` or `/<component>/db/task/{id}`: Get details
- `GET /<component>/db/batch/{id}`: Get batch details
- `GET /<component>/db/performance`: Get performance report
- `POST /<component>/db/export`: Export database to JSON
- `DELETE /<component>/db/run/{id}` or `/<component>/db/task/{id}`: Delete record
- `DELETE /<component>/db/batch/{id}`: Delete batch

## Running Servers with Database Integration

### Test Suite API

```bash
python -m test.refactored_test_suite.api.test_api_server --db-path ./data/test_runs.duckdb
```

### Generator API

```bash
python -m test.refactored_generator_suite.generator_api_server --db-path ./data/generator_tasks.duckdb
```

## Testing

Test scripts are provided to verify the database integration:

- **Test Suite**: `python test_suite_db_integration.py`
- **Generator**: `python test_generator_db_integration.py`

## API Usage Examples

### Get Recent Test Runs

```bash
curl http://localhost:8000/api/test/db/runs?limit=10
```

### Get Model Statistics

```bash
curl http://localhost:8001/api/generator/db/models/stats
```

### Get Performance Report

```bash
curl http://localhost:8000/api/test/db/performance?days=30
```

### Export Database

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"filename": "test_export.json"}' \
  http://localhost:8000/api/test/db/export
```

## Requirements

- **DuckDB**: `pip install duckdb`
- **FastAPI**: `pip install fastapi uvicorn`

## Integration with Unified API Server

The database integrations are automatically included when the Test Suite and Generator APIs are integrated with the Unified API Server. The Unified API Server forwards database-related requests to the appropriate component API.

## Conclusion

The DuckDB integration provides a robust foundation for tracking, analyzing, and optimizing operations across the IPFS Accelerate project. It enables historical data analysis, performance monitoring, and advanced reporting capabilities for both the Test Suite and Generator APIs.