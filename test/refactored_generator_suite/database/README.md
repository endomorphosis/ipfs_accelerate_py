# Generator Database Integration

This module provides DuckDB integration for the Generator API, allowing for persistent storage of generator tasks, history, performance metrics, and batch operations.

## Components

The database integration consists of the following components:

- **db_handler.py**: Core database handler for DuckDB operations
- **db_integration.py**: Integration layer between the Generator API and database
- **api_endpoints.py**: FastAPI endpoints for accessing database data

## Features

- Persistent storage of generator tasks and their status
- Tracking of detailed progress for each task
- Performance metrics collection and analysis
- Batch task management
- Historical data querying
- Export/import capabilities

## Database Schema

The database consists of the following tables:

1. **generator_tasks**: Main table for task information
   - task_id, model_name, hardware, status, etc.

2. **generator_task_steps**: Detailed progress tracking
   - step_name, status, progress, timestamps, etc.

3. **generator_task_metrics**: Performance and resource metrics
   - metric_name, metric_value, metric_unit, etc.

4. **generator_task_tags**: Custom tags for tasks
   - tag_name, tag_value, etc.

5. **generator_batch_tasks**: Batch task information
   - batch_id, description, task_count, etc.

## API Endpoints

The database is accessible through the following API endpoints:

- `/api/generator/db/tasks`: List task history
- `/api/generator/db/batches`: List batch task history
- `/api/generator/db/models/stats`: Get model statistics
- `/api/generator/db/task/{task_id}`: Get task details
- `/api/generator/db/batch/{batch_id}`: Get batch details
- `/api/generator/db/performance`: Get performance report
- `/api/generator/db/export`: Export database to JSON

## Usage

### Configuration

The database path can be configured when starting the Generator API server:

```bash
python generator_api_server.py --db-path /path/to/database.duckdb
```

If not specified, it defaults to `./data/generator_tasks.duckdb`.

### Querying Data

You can query task history and statistics through the API endpoints:

```bash
# Get task history
curl http://localhost:8001/api/generator/db/tasks?limit=10

# Get model statistics
curl http://localhost:8001/api/generator/db/models/stats

# Get task details
curl http://localhost:8001/api/generator/db/task/TASK_ID
```

### Exporting Data

You can export the database to a JSON file:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"filename": "generator_export.json"}' \
  http://localhost:8001/api/generator/db/export
```

## Integration with Generator API

The database integration is automatically used by the Generator API server if available. All tasks, steps, and metrics are automatically tracked and stored in the database.

## Requirements

- DuckDB: `pip install duckdb`
- FastAPI: `pip install fastapi uvicorn`