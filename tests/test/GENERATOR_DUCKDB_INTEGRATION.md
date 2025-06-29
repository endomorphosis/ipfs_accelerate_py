# Generator DuckDB Integration

This document provides details on the DuckDB integration for the Generator API, which enables tracking and querying of generator tasks, performance metrics, and usage histories.

## Overview

The Generator DuckDB Integration provides:

1. **Persistent Storage**: Records of all generation tasks, their parameters, and results
2. **Performance Metrics**: Collect and analyze generation performance across models
3. **Historical Data**: Maintain a history of task execution for analysis
4. **Batch Operations**: Track and manage batch generation operations
5. **Query Capabilities**: Advanced SQL querying on generation history

## Implementation Components

The implementation consists of three main components:

1. **db_handler.py** - Core database handler that manages the DuckDB connection, schema creation, and basic CRUD operations
2. **db_integration.py** - Integration layer that bridges between the Generator API and the database
3. **api_endpoints.py** - FastAPI endpoints that expose the database functionality through the API

## Database Schema

The DuckDB database uses a relational schema with the following tables:

- **generator_tasks** - Main table for generation tasks
  - task_id, model_name, hardware, status, etc.
- **generator_task_steps** - Steps within each generation task
  - task_id, step_name, status, progress, etc.
- **generator_task_metrics** - Performance metrics for tasks
  - task_id, metric_name, metric_value, etc.
- **generator_task_tags** - Custom tags for tasks
  - task_id, tag_name, tag_value
- **generator_batch_tasks** - Information about batch operations
  - batch_id, description, task_count, etc.

## Database API Endpoints

The database integration adds the following endpoints to the Generator API:

- `GET /api/generator/db/tasks` - List task history
- `GET /api/generator/db/batches` - List batch task history
- `GET /api/generator/db/models/stats` - Get model statistics
- `GET /api/generator/db/task/{task_id}` - Get task details
- `GET /api/generator/db/batch/{batch_id}` - Get batch details
- `GET /api/generator/db/performance` - Get performance report
- `POST /api/generator/db/export` - Export database to JSON
- `DELETE /api/generator/db/task/{task_id}` - Delete a task
- `DELETE /api/generator/db/batch/{batch_id}` - Delete a batch

## Example Queries

### Get Recent Tasks

```bash
curl http://localhost:8001/api/generator/db/tasks?limit=10
```

### Get Model Statistics

```bash
curl http://localhost:8001/api/generator/db/models/stats
```

### Get Performance Report

```bash
curl http://localhost:8001/api/generator/db/performance?days=30
```

## Configuration

The database path can be configured when starting the Generator API server:

```bash
python generator_api_server.py --db-path /path/to/database.duckdb
```

If not specified, the database will be created at `./data/generator_tasks.duckdb` relative to the Generator API server script.

## Implementation Details

### Task Tracking Flow

1. When a task is created, `track_task_start` is called to save the initial task data
2. During execution, `track_task_update` is called to update progress
3. Upon completion, `track_task_completion` is called to save the final results

### Batch Operation Tracking

1. When a batch operation is created, `track_batch_start` is called
2. For each task in the batch, the database is updated with the batch ID
3. When all tasks are complete, `track_batch_completion` is called

### Performance Metrics

The database stores various performance metrics:

- Task duration: Time taken to generate the model
- Resource usage: Memory, CPU, etc.
- Error counts: Number of failures or retries
- Step durations: Time taken for each step in the generation process

## Usage Examples

### JavaScript WebSocket Client

```javascript
// Connect to real-time updates for a task
const ws = new WebSocket(`ws://localhost:8001/api/generator/ws/${taskId}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress * 100}%`);
  console.log(`Current step: ${data.current_step}`);
  updateProgressBar(data.progress);
};
```

### Python Client

```python
import requests

# Get model statistics
response = requests.get("http://localhost:8001/api/generator/db/models/stats")
stats = response.json()

for model in stats:
    print(f"Model: {model['model_name']}")
    print(f"  Total tasks: {model['total_tasks']}")
    print(f"  Completed tasks: {model['completed_tasks']}")
    print(f"  Average duration: {model['avg_duration']:.2f} seconds")
```

### Export Database

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"filename": "generator_export.json"}' \
  http://localhost:8001/api/generator/db/export
```

## Testing

A test script is provided to verify the database integration:

```bash
python test_generator_db_integration.py
```

This script tests both the database handler and the integration layer.

## Future Enhancements

1. Add visualization endpoints for performance data
2. Implement advanced filtering and search capabilities
3. Add user and authorization tracking
4. Integrate with the Unified API Server

## Conclusion

The DuckDB integration provides a powerful foundation for tracking, analyzing, and optimizing generator tasks. It enables historical data analysis, performance monitoring, and advanced reporting capabilities for the Generator API.