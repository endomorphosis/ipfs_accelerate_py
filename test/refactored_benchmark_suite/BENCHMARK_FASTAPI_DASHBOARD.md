# Benchmark System with FastAPI and Dashboard Integration

This document provides comprehensive information about the FastAPI and Dashboard integration for the refactored benchmark system.

## System Architecture

The benchmark system is composed of three main components:

1. **FastAPI Server** (`benchmark_api_server.py`): Provides RESTful endpoints and WebSocket support for benchmark operations
2. **API Client** (`benchmark_api_client.py`): Client library for interacting with the API server
3. **Interactive Dashboard** (`benchmark_dashboard.py`): Web-based visualization and analysis interface

### FastAPI Server

The FastAPI server provides the following features:

- RESTful endpoints for benchmark operations
- WebSocket support for real-time progress monitoring
- Background task execution for non-blocking benchmark runs
- Integration with DuckDB for result storage
- Hardware detection and model discovery

### Interactive Dashboard

The interactive dashboard offers:

- Real-time visualization of benchmark results
- Performance comparison across hardware and models
- Live monitoring of active benchmark runs
- Custom SQL query support for advanced analysis
- Report management and access

## Getting Started

### Prerequisites

```bash
# Install required dependencies
pip install fastapi uvicorn pydantic duckdb websocket-client dash dash-bootstrap-components plotly pandas
```

### Starting the FastAPI Server

```bash
# Start with default settings
./run_benchmark_api_server.sh

# With custom settings
./run_benchmark_api_server.sh --port 8888 --db-path /path/to/benchmarks.duckdb --results-dir /path/to/results
```

### Starting the Dashboard

```bash
# Start with default settings
./run_benchmark_dashboard.sh

# With custom settings
./run_benchmark_dashboard.sh --port 8050 --api-url http://localhost:8000 --db-path /path/to/benchmarks.duckdb
```

### Running a Complete Example

```bash
# First, start the API server
./run_benchmark_api_server.sh

# In another terminal, run the example script
python benchmark_integration_example.py

# Start the dashboard to visualize results
./run_benchmark_dashboard.sh
```

## API Reference

### RESTful Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/benchmark/run` | POST | Start a benchmark run |
| `/api/benchmark/status/{run_id}` | GET | Get status of a benchmark run |
| `/api/benchmark/results/{run_id}` | GET | Get results of a completed benchmark |
| `/api/benchmark/models` | GET | List available models |
| `/api/benchmark/hardware` | GET | List available hardware platforms |
| `/api/benchmark/reports` | GET | List available benchmark reports |
| `/api/benchmark/query` | GET | Query benchmark results with filters |

### WebSocket Endpoint

| Endpoint | Description |
|----------|-------------|
| `/api/benchmark/ws/{run_id}` | Real-time updates for a benchmark run |

### API Client Example

```python
from benchmark_api_client import BenchmarkAPIClient

# Create client
client = BenchmarkAPIClient("http://localhost:8000")

# Start a benchmark
run_data = client.start_benchmark(
    priority="high",
    hardware=["cpu"],
    models=["bert", "gpt2"],
    batch_sizes=[1, 8],
    precision="fp32"
)

run_id = run_data["run_id"]

# Monitor progress
def progress_callback(status):
    print(f"Progress: {status['progress']:.1%}")
    print(f"Step: {status['current_step']}")

client.monitor_progress(run_id, progress_callback)

# Get results
results = client.get_results(run_id)
```

## Dashboard Features

### Overview Tab

- Hardware comparison charts
- Top-performing models by hardware
- Batch size scaling visualization

### Comparison Tab

- Performance heatmap across model families and hardware
- Detailed results table with filtering and sorting
- Export capabilities

### Live Runs Tab

- Real-time progress tracking
- Start new benchmark runs
- Configure benchmark parameters

### Reports Tab

- Available benchmark reports
- Custom SQL query interface
- Direct access to report files

## Integration with Electron

The FastAPI server is designed for seamless integration with Electron containers:

```javascript
// Example JavaScript code for Electron
const ws = new WebSocket('ws://localhost:8000/api/benchmark/ws/YOUR_RUN_ID');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress);
  updateProgressBar(data.progress);
};

// Start a benchmark run
fetch('http://localhost:8000/api/benchmark/run', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    priority: 'high',
    hardware: ['cpu'],
    models: ['bert', 'gpt2'],
    batch_sizes: [1, 8]
  })
}).then(response => response.json())
  .then(data => {
    console.log('Run ID:', data.run_id);
    // Connect WebSocket for real-time updates
    const ws = new WebSocket(`ws://localhost:8000/api/benchmark/ws/${data.run_id}`);
  });
```

## DuckDB Integration

The benchmark system stores results in a DuckDB database with the following schema:

- `models`: Model information
- `hardware_platforms`: Hardware platform information
- `test_runs`: Test run information
- `performance_results`: Performance metrics
- `hardware_compatibility`: Hardware compatibility data

Example query:

```sql
-- Get average throughput by model family and hardware
SELECT 
    m.model_family,
    h.hardware_type,
    AVG(pr.throughput_items_per_second) as avg_throughput
FROM 
    performance_results pr
JOIN 
    models m ON pr.model_id = m.model_id
JOIN 
    hardware_platforms h ON pr.hardware_id = h.hardware_id
GROUP BY 
    m.model_family, h.hardware_type
ORDER BY 
    avg_throughput DESC
```

## Advanced Features

### WebSocket Integration

The benchmark system uses WebSockets for real-time progress updates:

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"Progress: {data['progress']:.1%}")
    print(f"Current step: {data['current_step']}")
    print(f"Models: {data['completed_models']}/{data['total_models']}")

ws = websocket.WebSocketApp(
    "ws://localhost:8000/api/benchmark/ws/YOUR_RUN_ID",
    on_message=on_message
)
ws.run_forever()
```

### Resource-Aware Scheduling

The benchmark system includes resource-aware scheduling for optimal hardware utilization:

- Hardware detection and capability assessment
- Priority-based scheduling
- Incremental benchmarking for efficiency

### Progressive Complexity Mode

Progressive complexity mode starts with simpler benchmarks and gradually increases complexity:

```python
run_data = client.start_benchmark(
    priority="high",
    hardware=["cpu"],
    models=["bert", "gpt2"],
    batch_sizes=[1, 8, 16, 32, 64],
    precision="fp32",
    progressive_mode=True
)
```

## Extending the System

### Adding Custom Benchmarks

1. Create a benchmark implementation in `benchmarks/custom/`
2. Register the benchmark with the registry

### Creating Custom Visualizations

Extend the dashboard with custom visualizations:

1. Add a new tab in `benchmark_dashboard.py`
2. Create visualization components
3. Connect to data sources with callbacks

## Troubleshooting

### Common Issues

- **API server not starting**: Check FastAPI and Uvicorn installation
- **Dashboard not connecting**: Verify API URL and CORS settings
- **WebSocket connection failing**: Check if the server is running and the run ID is valid
- **DuckDB connection error**: Verify the database path and permissions

### Logging

Both the API server and dashboard generate log files:

- `benchmark_api_server.log`
- `benchmark_dashboard.log`