# Test Suite Database Integration

This module provides DuckDB integration for the Test Suite API, allowing for persistent storage of test runs, history, performance metrics, and batch operations.

## Components

The database integration consists of the following components:

- **db_handler.py**: Core database handler for DuckDB operations
- **db_integration.py**: Integration layer between the Test Suite API and database
- **api_endpoints.py**: FastAPI endpoints for accessing database data

## Features

- **Persistent Storage**: Records of all test runs, their parameters, and results
- **Performance Metrics**: Collect and analyze test performance across models
- **Historical Data**: Maintain a history of test runs for analysis
- **Batch Operations**: Track and manage batch test runs
- **Query Capabilities**: Advanced SQL querying on test history

## Database Schema

The database consists of the following tables:

1. **test_runs**: Main table for test run information
   - run_id, model_name, hardware, status, etc.

2. **test_run_results**: Detailed test results
   - tests_passed, tests_failed, tests_skipped, etc.

3. **test_run_steps**: Detailed progress tracking
   - step_name, status, progress, timestamps, etc.

4. **test_run_metrics**: Performance and resource metrics
   - metric_name, metric_value, metric_unit, etc.

5. **test_run_tags**: Custom tags for test runs
   - tag_name, tag_value, etc.

6. **batch_test_runs**: Batch test run information
   - batch_id, description, run_count, etc.

7. **test_run_batch**: Relation between test runs and batches
   - run_id, batch_id, etc.

8. **test_run_artifacts**: Files produced during testing
   - artifact_type, artifact_path, description, etc.

## API Endpoints

The database integration adds the following endpoints to the Test Suite API:

- `GET /api/test/db/runs`: List test run history
- `GET /api/test/db/batches`: List batch test run history
- `GET /api/test/db/models/stats`: Get model statistics
- `GET /api/test/db/hardware/stats`: Get hardware statistics
- `GET /api/test/db/run/{run_id}`: Get test run details
- `GET /api/test/db/batch/{batch_id}`: Get batch details
- `GET /api/test/db/performance`: Get performance report
- `POST /api/test/db/search`: Search test runs
- `POST /api/test/db/export`: Export database to JSON
- `DELETE /api/test/db/run/{run_id}`: Delete a test run
- `DELETE /api/test/db/batch/{batch_id}`: Delete a batch

## Usage

### Configuration

The database path can be configured when starting the Test API server:

```bash
python test_api_server.py --db-path /path/to/database.duckdb
```

If not specified, it defaults to `./data/test_runs.duckdb`.

### Querying Data

You can query test run history and statistics through the API endpoints:

```bash
# Get test run history
curl http://localhost:8000/api/test/db/runs?limit=10

# Get model statistics
curl http://localhost:8000/api/test/db/models/stats

# Get test run details
curl http://localhost:8000/api/test/db/run/RUN_ID
```

### Exporting Data

You can export the database to a JSON file:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"filename": "test_export.json"}' \
  http://localhost:8000/api/test/db/export
```

## Integration with Test API

The database integration is automatically used by the Test API server if available. All test runs, steps, and metrics are automatically tracked and stored in the database.

## Testing

A test script is provided to verify the database integration:

```bash
python test_suite_db_integration.py
```

This script tests both the database handler and the integration layer.

## Requirements

- DuckDB: `pip install duckdb`
- FastAPI: `pip install fastapi uvicorn`