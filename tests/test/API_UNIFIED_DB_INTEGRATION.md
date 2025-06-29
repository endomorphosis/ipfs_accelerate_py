# Unified API Server with DuckDB Database Integration

This document describes the enhanced Unified API Server with DuckDB database integration for the IPFS Accelerate Python Framework. The implementation provides a centralized gateway for accessing all component APIs and their respective database operations with built-in authentication, cross-component querying, and unified data views.

**Status: COMPLETED (100%)**

The Unified API Server now fully supports:
- Routing of all component API requests
- Component-specific database operations
- Cross-component database operations with unified views
- API key authentication for database operations
- WebSocket forwarding for real-time updates

## Architecture Overview

The implementation consists of three main layers:

1. **Unified API Server (Gateway)**
   - Central entry point for all API and database operations
   - Handles authentication with API key validation
   - Routes requests to appropriate component APIs
   - Provides cross-component aggregation endpoints

2. **Component Database APIs**
   - Component-specific database APIs (Test Suite, Generator, Benchmark)
   - Each component exposes its own database endpoints
   - Standardized interface across all components

3. **Database Handlers and Integration Layers**
   - Each component has its own database handler and integration layer
   - Database handlers provide direct database operations
   - Integration layers connect APIs to databases

## Authentication and Security

All database operations require authentication using an API key. The API key is provided via the `X-API-Key` HTTP header. Currently, the implementation only validates the presence of an API key, but in production, this would validate against a secure store.

```bash
# Example curl request with API key
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/db/overview
```

## Unified API Server Endpoints

### Component Database Routes

The Unified API Server provides routes to access each component's database API:

- **Test Suite Database API**: `/api/db/test/{path}`
- **Generator Database API**: `/api/db/generator/{path}`
- **Benchmark Database API**: `/api/db/benchmark/{path}`

These routes forward requests to the respective component API with authentication.

### Cross-Component Database Operations

The Unified API Server provides cross-component database operations for aggregated views:

#### Database Overview

```
GET /api/db/overview
```

Returns a unified overview of all database components with statistics, including:
- Test statistics from the Test Suite API
- Generator statistics from the Generator API
- Benchmark statistics from the Benchmark API

#### Model Unified Data

```
GET /api/db/model/{model_name}
```

Returns unified data for a specific model across all components, including:
- Recent test runs for the model
- Recent generator tasks for the model
- Recent benchmark runs for the model
- Aggregate statistics and success rates

#### Unified Search

```
POST /api/db/search?query={query}&limit={limit}
```

Searches across all database components for the given query, returning combined results from:
- Test Suite database
- Generator database
- Benchmark database

## Component-Specific Database APIs

Each component has its own database API with similar endpoint structures:

### Test Suite Database API

- `GET /api/test/db/runs` - Get test run history
- `GET /api/test/db/batches` - Get batch test run history
- `GET /api/test/db/models/stats` - Get model statistics
- `GET /api/test/db/hardware/stats` - Get hardware statistics
- `GET /api/test/db/run/{run_id}` - Get test run details
- `GET /api/test/db/batch/{batch_id}` - Get batch details
- `GET /api/test/db/performance` - Get performance report
- `POST /api/test/db/search` - Search test runs
- `POST /api/test/db/export` - Export database to JSON
- `DELETE /api/test/db/run/{run_id}` - Delete a test run
- `DELETE /api/test/db/batch/{batch_id}` - Delete a batch

### Generator Database API

- `GET /api/generator/db/tasks` - Get generation task history
- `GET /api/generator/db/batches` - Get batch task history
- `GET /api/generator/db/models/stats` - Get model statistics
- `GET /api/generator/db/task/{task_id}` - Get task details
- `GET /api/generator/db/batch/{batch_id}` - Get batch details
- `GET /api/generator/db/performance` - Get performance report
- `POST /api/generator/db/export` - Export database to JSON
- `DELETE /api/generator/db/task/{task_id}` - Delete a task
- `DELETE /api/generator/db/batch/{batch_id}` - Delete a batch

### Benchmark Database API

- `GET /api/benchmark/db/runs` - Get benchmark run history
- `GET /api/benchmark/db/batches` - Get batch run history
- `GET /api/benchmark/db/models/stats` - Get model statistics
- `GET /api/benchmark/db/hardware/stats` - Get hardware statistics
- `GET /api/benchmark/db/run/{run_id}` - Get run details
- `GET /api/benchmark/db/batch/{batch_id}` - Get batch details
- `GET /api/benchmark/db/performance` - Get performance metrics
- `POST /api/benchmark/db/export` - Export database to JSON
- `DELETE /api/benchmark/db/run/{run_id}` - Delete a benchmark run
- `DELETE /api/benchmark/db/batch/{batch_id}` - Delete a batch

## Starting the Unified API Server

To start the Unified API Server with database integration:

```bash
cd /path/to/ipfs_accelerate_py/test
python unified_api_server.py --gateway-port 8080 --test-api-port 8000 --generator-api-port 8001 --benchmark-api-port 8002
```

### Command-line Options

- `--config` - Path to JSON configuration file
- `--gateway-port` - Port for the API gateway (default: 8080)
- `--test-api-port` - Port for the Test API (default: 8000)
- `--generator-api-port` - Port for the Generator API (default: 8001)
- `--benchmark-api-port` - Port for the Benchmark API (default: 8002)

### Configuration File

You can provide a JSON configuration file with the `--config` option:

```json
{
  "test_api": {
    "enabled": true,
    "port": 8000,
    "host": "0.0.0.0",
    "module": "refactored_test_suite.integration.test_api_integration",
    "args": ["--server", "--db-path", "/path/to/test_db.duckdb"]
  },
  "generator_api": {
    "enabled": true,
    "port": 8001,
    "host": "0.0.0.0",
    "module": "refactored_generator_suite.generator_api_server",
    "args": ["--db-path", "/path/to/generator_db.duckdb"]
  },
  "benchmark_api": {
    "enabled": true,
    "port": 8002,
    "host": "0.0.0.0",
    "module": "refactored_benchmark_suite.benchmark_api_server",
    "args": ["--db-path", "/path/to/benchmark_db.duckdb"]
  },
  "gateway": {
    "enabled": true,
    "port": 8080,
    "host": "0.0.0.0"
  }
}
```

## Example Usage

### Python Client Example

```python
import requests

# API key for authentication
headers = {
    "X-API-Key": "your-api-key"
}

# Base URL of the Unified API Server
base_url = "http://localhost:8080"

# Get model unified data
model_name = "bert-base-uncased"
response = requests.get(
    f"{base_url}/api/db/model/{model_name}", 
    headers=headers
)
unified_data = response.json()
print(f"Model: {model_name}")
print(f"Total test runs: {unified_data['overview']['total_test_runs']}")
print(f"Total generator tasks: {unified_data['overview']['total_generator_tasks']}")
print(f"Total benchmark runs: {unified_data['overview']['total_benchmark_runs']}")

# Search across all databases
response = requests.post(
    f"{base_url}/api/db/search",
    params={"query": "bert-base", "limit": 10},
    headers=headers
)
search_results = response.json()
print(f"Search results: {search_results['result_counts']['total']} total matches")
```

### JavaScript/Fetch Example

```javascript
// API key for authentication
const headers = {
    "X-API-Key": "your-api-key"
};

// Base URL of the Unified API Server
const baseUrl = "http://localhost:8080";

// Get database overview
fetch(`${baseUrl}/api/db/overview`, {
    method: 'GET',
    headers: headers
})
.then(response => response.json())
.then(data => {
    console.log("Database Overview:");
    console.log("Test Stats:", data.test_stats);
    console.log("Generator Stats:", data.generator_stats);
    console.log("Benchmark Stats:", data.benchmark_stats);
})
.catch(error => console.error("Error:", error));

// Search across all databases
fetch(`${baseUrl}/api/db/search?query=bert-base&limit=10`, {
    method: 'POST',
    headers: headers
})
.then(response => response.json())
.then(data => {
    console.log("Search Results:");
    console.log(`Total Matches: ${data.result_counts.total}`);
    console.log(`Test Results: ${data.result_counts.test}`);
    console.log(`Generator Results: ${data.result_counts.generator}`);
    console.log(`Benchmark Results: ${data.result_counts.benchmark}`);
})
.catch(error => console.error("Error:", error));
```

## Database Schema

Each component has its own database schema designed for its specific needs:

### Test Suite Database Schema

- `test_runs` - Main table for test runs
- `test_run_steps` - Details of each step in a test run
- `test_run_metrics` - Performance metrics for test runs
- `test_batch_runs` - Information about batch test runs
- `test_run_tags` - Tags associated with test runs

### Generator Database Schema

- `generator_tasks` - Main table for generator tasks
- `generator_task_steps` - Details of each step in a task
- `generator_task_metrics` - Performance metrics for tasks
- `generator_batch_tasks` - Information about batch tasks
- `generator_task_tags` - Tags for generator tasks

### Benchmark Database Schema

- `benchmark_runs` - Main table for benchmark runs
- `benchmark_metrics` - Detailed metrics for benchmarks
- `benchmark_batch_runs` - Information about batch benchmark runs
- `benchmark_hardware_metrics` - Hardware-specific metrics
- `benchmark_model_metrics` - Model-specific metrics

## Implementation Status

The Unified API Server with DuckDB database integration is now fully implemented and ready for use. This implementation completes the planned work for the Unified API Server, which was previously at 30% completion, bringing it to 100% completion.

Key features implemented:
- Routing of database operations to appropriate component APIs
- Authentication framework with API key validation
- Cross-component database operations
- Unified data views across all components
- Error handling and enhanced logging
- Concurrent data fetching from multiple components
- Aggregate statistics and metrics across components
- Component-specific database endpoint forwarding

This implementation completes the Unified API Server milestone from the API Refactoring Initiative described in CLAUDE.md. The server now provides a centralized access point for all IPFS Accelerate framework components with comprehensive database operations support.

## Further Work (Optional)

While the current implementation is complete, there are several areas for potential enhancement:

1. **Advanced Authentication** - Implement JWT-based authentication with role-based access control
2. **Caching Layer** - Add Redis caching for frequently accessed database queries
3. **Data Visualization API** - Endpoints for generating charts and visualizations
4. **Streaming Database Updates** - WebSocket endpoints for real-time database updates
5. **Database Migration API** - Endpoints for managing database migrations and schema updates

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure you're providing the `X-API-Key` header with all database requests
   - Check that the API key is correctly formatted

2. **Connection Errors**
   - Verify that all component APIs are running
   - Check that the ports specified match the actual running ports

3. **Missing Database Files**
   - Ensure all database files exist at the specified paths
   - Check that the paths provided have appropriate permissions

### Logging

The Unified API Server logs to both the console and a log file:
- Console output shows real-time logs
- `unified_api_server.log` file contains detailed logs

To increase log verbosity, modify the logging level in the server:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("unified_api_server.log")
    ]
)
```