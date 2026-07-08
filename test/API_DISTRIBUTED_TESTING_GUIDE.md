# API Distributed Testing Framework Guide

This guide provides a comprehensive overview of the API Distributed Testing Framework, a system for testing and benchmarking multiple API providers (OpenAI, Claude, Groq, etc.) in a distributed environment.

## Architecture Overview

The API Distributed Testing Framework uses a coordinator-worker architecture to enable scalable and distributed testing of API providers.

```
┌───────────────────┐           ┌───────────────────┐
│                   │           │                   │
│     Coordinator   │◄─────────►│    Worker Node 1  │
│      Server       │           │                   │
│                   │           └───────────────────┘
└───────┬───────────┘
        │                       ┌───────────────────┐
        │                       │                   │
        └──────────────────────►│    Worker Node 2  │
        │                       │                   │
        │                       └───────────────────┘
        │
        │                       ┌───────────────────┐
        │                       │                   │
        └──────────────────────►│    Worker Node N  │
                                │                   │
                                └───────────────────┘
```

The framework consists of the following key components:

1. **Coordinator Server**: Manages task distribution, collects results, and provides centralized coordination.
2. **Worker Nodes**: Execute API tests and report results back to the coordinator.
3. **API Testing Interface**: Provides a unified interface for testing different API providers.
4. **Anomaly Detection**: Identifies anomalies in API performance metrics.
5. **Predictive Analytics**: Forecasts future API performance trends.
6. **Monitoring Dashboard**: Visualizes test results and performance metrics.

## Key Components

### Coordinator Server

The coordinator server (`run_api_coordinator_server.py`) is responsible for:

- Managing worker node registration and heartbeats
- Distributing API test tasks to appropriate worker nodes
- Collecting and aggregating test results
- Detecting anomalies in API performance metrics
- Generating performance reports and visualizations
- Predicting future API performance trends

### Worker Nodes

Worker nodes (`run_api_worker_node.py`) are responsible for:

- Registering with the coordinator server
- Executing API test tasks assigned by the coordinator
- Reporting test results back to the coordinator
- Automatically detecting supported API providers
- Managing API key security and authentication

### API Testing Interface

The unified API testing interface (`api_unified_testing_interface.py`) provides:

- A common interface for testing different API providers
- Support for multiple test types (latency, throughput, reliability, cost efficiency)
- Standardized metrics collection and reporting
- Integration with the distributed testing framework

### End-to-End Example

The end-to-end example (`run_end_to_end_api_distributed_test.py`) demonstrates:

- A complete workflow of the API Distributed Testing Framework
- Simulated environment for testing without actual API calls
- Multiple API providers (OpenAI, Claude, Groq)
- Different test types (latency, throughput, reliability, cost efficiency)
- Performance metrics collection and visualization
- Multiple execution modes (simulation, multiprocess, distributed)

## Setup and Installation

### Prerequisites

- Python 3.8 or later
- Network connectivity between coordinator and worker nodes
- API keys for supported providers (optional)

### Coordinator Server Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the coordinator server:
   ```bash
   python run_api_coordinator_server.py --host 0.0.0.0 --port 5555
   ```

3. Optional flags:
   - `--disable-anomaly-detection`: Disable anomaly detection
   - `--disable-predictive-analytics`: Disable predictive analytics
   - `--disable-dashboard`: Disable the monitoring dashboard
   - `--dashboard-port PORT`: Set dashboard port (default: 8080)
   - `--results-dir DIR`: Set results directory

### Worker Node Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set API keys as environment variables:
   ```bash
   export OPENAI_API_KEY=your_openai_key
   export ANTHROPIC_API_KEY=your_claude_key
   export GROQ_API_KEY=your_groq_key
   ```

3. Start a worker node:
   ```bash
   python run_api_worker_node.py --coordinator http://coordinator-host:5555
   ```

4. Optional flags:
   - `--worker-id ID`: Set custom worker ID
   - `--tags TAGS`: Set comma-separated tags for worker selection
   - `--disable-anomaly-detection`: Disable anomaly detection
   - `--disable-predictive-analytics`: Disable predictive analytics
   - `--results-dir DIR`: Set results directory

### End-to-End Example

The end-to-end example provides a simple way to test the framework without setting up a full coordinator-worker infrastructure:

```bash
python run_end_to_end_api_distributed_test.py
```

This runs a simulated environment with:
- A simulated coordinator
- Multiple simulated worker nodes
- Simulated API providers (OpenAI, Claude, Groq)
- A basic test suite (latency, throughput, reliability, cost efficiency)

Optional flags:
- `--mode MODE`: Set execution mode (simulation, multiprocess, distributed)
- `--test-suite SUITE`: Set test suite (basic, comprehensive, stress)
- `--apis APIS`: Set comma-separated list of API providers
- `--num-workers N`: Set number of worker nodes
- `--output-dir DIR`: Set output directory

## Usage Guide

### Running API Tests

The framework supports several test types:

1. **Latency Tests**: Measure response time for API requests
2. **Throughput Tests**: Measure requests per second the API can handle
3. **Reliability Tests**: Measure success rate of API requests
4. **Cost Efficiency Tests**: Measure tokens per dollar and other cost metrics

### Creating API Tests with the Coordinator

You can create API tests using the coordinator's API:

```python
from api_unified_testing_interface import APIDistributedTesting

# Connect to coordinator
api_testing = APIDistributedTesting(coordinator_url="http://coordinator-host:5555")

# Run a latency test for OpenAI
test_id = api_testing.run_distributed_test(
    api_type="openai",
    test_type="latency",
    parameters={
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "iterations": 10
    },
    num_workers=2
)

# Wait for and get results
result = api_testing.get_test_results(test_id)
```

### Comparing Multiple API Providers

You can compare multiple API providers:

```python
# Compare OpenAI, Claude, and Groq
comparison_id = api_testing.compare_apis(
    api_types=["openai", "claude", "groq"],
    test_type="throughput",
    parameters={
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "duration": 10,
        "concurrent_requests": 5
    }
)

# Get comparison results
results = api_testing.get_comparison_results(comparison_id)
```

### Running the End-to-End Example

The end-to-end example is the easiest way to get started:

```bash
# Run basic test suite
python run_end_to_end_api_distributed_test.py

# Run comprehensive test suite
python run_end_to_end_api_distributed_test.py --test-suite comprehensive

# Run stress test suite
python run_end_to_end_api_distributed_test.py --test-suite stress

# Test specific APIs
python run_end_to_end_api_distributed_test.py --apis openai,claude
```

The example generates a summary of the test results and rankings for each API provider, as well as a detailed JSON file with all results.

## Advanced Features

### Anomaly Detection

The framework includes anomaly detection (`api_anomaly_detection.py`) that can identify:

- Latency spikes
- Throughput drops
- Success rate decreases
- Cost anomalies

Anomalies are detected using multiple algorithms:

- Z-score analysis
- Moving average deviation
- Trend detection
- Seasonality-aware anomaly detection

### Predictive Analytics

The framework includes predictive analytics (`api_predictive_analytics.py`) that can:

- Forecast future API performance metrics
- Predict potential future anomalies
- Provide cost optimization recommendations
- Analyze long-term API performance trends

### Monitoring Dashboard

The monitoring dashboard (`api_monitoring_dashboard.py`) provides:

- Real-time visualization of API performance metrics
- Historical trend analysis
- Anomaly highlighting and alerts
- Comparative visualizations across API providers

## Integration Examples

### Integrating with GitHub Actions

```yaml
name: API Performance Testing

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  api_testing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run API tests
        run: |
          python run_end_to_end_api_distributed_test.py --test-suite comprehensive
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

### Integrating with Monitoring Systems

The framework can integrate with external monitoring systems:

```python
# Configure notification rules
notification_config = {
    "email": {
        "enabled": True,
        "recipients": ["alerts@example.com"],
        "min_severity": "MEDIUM"
    },
    "slack": {
        "enabled": True,
        "webhook_url": "https://hooks.slack.com/services/...",
        "channel": "#api-alerts",
        "min_severity": "HIGH"
    }
}

# Start coordinator with notifications
coordinator = APICoordinatorServer(
    notification_config=notification_config
)
```

## FAQ

**Q: How do I add support for a new API provider?**

A: Create a new backend class in `api_unified_testing_interface.py` that inherits from `APIBackend` and implements the required methods.

**Q: How many worker nodes should I deploy?**

A: It depends on your testing needs. For basic testing, 3-5 worker nodes should be sufficient. For high-throughput testing, you may need 10 or more.

**Q: Are there rate limits enforced by the framework?**

A: The framework respects rate limits set by API providers and implements appropriate backoff strategies. You can configure additional rate limits in the worker nodes.

**Q: Can I run tests without actual API keys?**

A: Yes, you can use the simulation mode in the end-to-end example to test the framework without actual API calls.

**Q: How do I interpret anomaly detection results?**

A: Anomalies are reported with severity levels (LOW, MEDIUM, HIGH) and confidence scores. Focus on anomalies with HIGH severity and confidence > 0.8.

## Troubleshooting

### Common Issues

1. **Connection issues between coordinator and workers**:
   - Check network connectivity
   - Verify firewall settings
   - Ensure coordinator host/port are correctly configured

2. **API key errors**:
   - Verify environment variables are correctly set
   - Check API key validity and permissions
   - Ensure API provider is available and not experiencing downtime

3. **Worker nodes not receiving tasks**:
   - Check worker registration status
   - Verify worker capabilities match task requirements
   - Check coordinator logs for task distribution issues

### Logging

Both coordinator and worker nodes use Python's logging module:

```bash
# Enable debug logging
python run_api_coordinator_server.py --log-level debug

# Save logs to file
python run_api_worker_node.py --log-file worker.log
```

## Next Steps

1. Implement the multiprocess and distributed modes in the end-to-end example
2. Add support for more API providers
3. Enhance visualization capabilities
4. Implement a web-based dashboard for monitoring

## Contributing

Contributions to the API Distributed Testing Framework are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.