# Web Dashboard for Result Aggregator

The Web Dashboard provides a user-friendly interface to visualize and interact with the Result Aggregator data. It includes interactive visualizations, real-time updates, and a comprehensive API for accessing test results.

## Features

- **Interactive Dashboards**: Visualize test results, performance trends, and anomalies
- **Real-time Monitoring**: Monitor cluster health, worker nodes, and task execution in real-time
- **WebSocket Support**: True real-time updates via WebSocket with automatic fallback to polling
- **REST API**: Comprehensive API for accessing and querying results data
- **Advanced Filtering**: Filter results by various criteria such as test type, status, worker ID, etc.
- **Authentication**: Basic user authentication to secure access to the dashboard
- **Responsive Design**: Works on desktop and mobile devices

## Dashboard Components

- **Main Dashboard**: The primary dashboard for accessing test results and performance data
- **Real-time Monitoring Dashboard**: A specialized dashboard for monitoring cluster health, worker nodes, and tasks in real-time
- **Results Page**: Detailed view of test results with filtering and sorting
- **Trends Page**: Visualization of performance trends over time
- **Anomalies Page**: Detection and visualization of performance anomalies
- **Reports Page**: Generate and view comprehensive reports

## Requirements

- Python 3.6+
- Required Python packages:
  - Flask
  - Flask-CORS
  - Flask-SocketIO (for WebSocket real-time updates)
  - DuckDB
  - Pandas
  - NumPy
  - Matplotlib (for visualization)
  - Scikit-learn (for anomaly detection)

Install the required packages:

```bash
pip install flask flask-cors flask-socketio duckdb pandas numpy matplotlib scikit-learn
```

## Running the Dashboard

Use the provided run script to start the dashboard:

```bash
python run_web_dashboard.py [--port PORT] [--db-path DB_PATH] [--debug] [--update-interval SECONDS]
```

Options:
- `--port PORT`: Port to run the web server on (default: 8050)
- `--db-path DB_PATH`: Path to DuckDB database (default: ./test_results.duckdb)
- `--debug`: Run in debug mode
- `--enable-ml`: Enable machine learning features (default: enabled)
- `--enable-visualization`: Enable visualization features (default: enabled)
- `--update-interval SECONDS`: Interval in seconds for WebSocket real-time monitoring updates (default: 5)

## Accessing the Dashboard

Once the server is running, access the dashboard at:

```
http://localhost:8050
```

For the real-time monitoring dashboard, use:

```
http://localhost:8050/monitoring
```

Use the default credentials to log in:
- Username: `admin` or `user`
- Password: `admin_password` or `user_password`

For more information about the real-time monitoring dashboard, see [REAL_TIME_MONITORING_DASHBOARD.md](REAL_TIME_MONITORING_DASHBOARD.md).

## Integrating with Existing Systems

### Coordinator Integration

The dashboard integrates with the Distributed Testing Coordinator through either the `IntegratedAnalysisSystem` (recommended) or the legacy `ResultAggregatorIntegration` class. This integration provides real-time processing of test results, anomaly detection, and notification capabilities.

#### Method 1: Using the IntegratedAnalysisSystem (Recommended)

```python
from coordinator import DistributedTestingCoordinator
from result_aggregator.integrated_analysis_system import IntegratedAnalysisSystem
from datetime import timedelta

# Initialize the coordinator
coordinator = DistributedTestingCoordinator(db_path='./test_db.duckdb')

# Initialize the integrated analysis system
analysis_system = IntegratedAnalysisSystem(
    db_path='./test_db.duckdb',
    enable_ml=True,
    enable_visualization=True,
    enable_real_time_analysis=True,
    analysis_interval=timedelta(minutes=5)
)

# Register with coordinator
analysis_system.register_with_coordinator(coordinator)

# Register notification handler (optional)
def notification_handler(notification):
    print(f"Notification: {notification['type']} - {notification['severity']}")
    print(f"Message: {notification['message']}")
    
analysis_system.register_notification_handler(notification_handler)
```

#### Method 2: Using the Legacy ResultAggregatorIntegration

```python
from coordinator import DistributedTestingCoordinator
from result_aggregator.coordinator_integration import ResultAggregatorIntegration

# Initialize the coordinator
coordinator = DistributedTestingCoordinator(db_path='./test_db.duckdb')

# Initialize the result aggregator integration
integration = ResultAggregatorIntegration(
    coordinator=coordinator,
    enable_ml=True,
    enable_visualization=True,
    enable_real_time_analysis=True,
    enable_notifications=True
)

# Register with coordinator
integration.register_with_coordinator()
```

### Web Dashboard Integration

The web dashboard can be integrated with existing systems by using the provided REST API endpoints. This allows for programmatic access to test results and visualizations.

## API Endpoints

The following API endpoints are available:

### Results

- `GET /api/results`: Get test results with pagination and filtering
  - Query Parameters:
    - `limit`: Maximum number of results to return
    - `offset`: Offset for pagination
    - `test_type`: Filter by test type
    - `status`: Filter by status
    - `worker_id`: Filter by worker ID
    - `start_time`: Filter by start time
    - `end_time`: Filter by end time

- `GET /api/result/<result_id>`: Get a specific test result by ID

### Aggregated Results

- `GET /api/aggregated`: Get aggregated test results
  - Query Parameters:
    - `aggregation_type`: Type of aggregation (mean, median, min, max, etc.)
    - `group_by`: Fields to group by (comma-separated)
    - `metrics`: Metrics to aggregate (comma-separated)
    - `test_type`: Filter by test type
    - `status`: Filter by status
    - `start_time`: Filter by start time
    - `end_time`: Filter by end time

### Performance Trends

- `GET /api/trends`: Get performance trends
  - Query Parameters:
    - `metrics`: Metrics to analyze (comma-separated)
    - `test_type`: Filter by test type
    - `start_time`: Filter by start time
    - `end_time`: Filter by end time
    - `window_size`: Window size for moving average

### Real-time Monitoring

- `GET /api/monitoring/cluster`: Get cluster status metrics
  - Returns:
    - Active worker count
    - Total task count
    - Success rate
    - Cluster health score and status
    - Trend data for key metrics

- `GET /api/monitoring/workers`: Get worker node information
  - Returns:
    - Worker ID and status
    - Health status (healthy, warning, critical, unknown)
    - Current CPU and memory usage
    - Tasks completed
    - Success rate
    - Available hardware types

- `GET /api/monitoring/tasks`: Get task queue data
  - Query Parameters:
    - `status`: Filter by task status (all, pending, running, completed, failed)
  - Returns:
    - Task ID and type
    - Status and priority
    - Assigned worker ID (if any)

- `GET /api/monitoring/resources`: Get resource usage trends
  - Returns:
    - CPU usage data (average and maximum)
    - Memory usage data (average and maximum)
    - Time labels for x-axis

- `GET /api/monitoring/hardware`: Get hardware availability data
  - Returns:
    - Available hardware counts by type
    - Total hardware counts by type
    - Hardware type labels

- `GET /api/monitoring/network`: Get network topology data
  - Returns:
    - Node data (coordinator and workers)
    - Link data showing connections and strengths
    - Status information for each node

For detailed information about the monitoring API endpoints, see [REAL_TIME_MONITORING_DASHBOARD.md](REAL_TIME_MONITORING_DASHBOARD.md#api-endpoints).

### Performance Analysis

- `GET /api/performance/regression`: Detect performance regression for metrics
  - Query Parameters:
    - `metric`: Name of the metric to analyze (omit for all key metrics)
    - `baseline_period`: Period for baseline (e.g., "7d" for 7 days)
    - `comparison_period`: Period for comparison (e.g., "1d" for 1 day)
    - `test_type`: Filter by test type

- `GET /api/performance/hardware`: Compare performance across different hardware profiles
  - Query Parameters:
    - `metrics`: Metrics to compare (comma-separated, omit for all key metrics)
    - `test_type`: Filter by test type
    - `time_period`: Time period for analysis (e.g., "30d" for 30 days)

- `GET /api/performance/efficiency`: Analyze resource efficiency metrics
  - Query Parameters:
    - `test_type`: Filter by test type
    - `time_period`: Time period for analysis (e.g., "30d" for 30 days)

- `GET /api/performance/time`: Analyze performance over time with advanced regression analysis
  - Query Parameters:
    - `metric`: Metric to analyze (required)
    - `grouping`: Time grouping (day, week, month)
    - `test_type`: Filter by test type
    - `time_period`: Time period for analysis (e.g., "90d" for 90 days)

- `GET /api/performance/report`: Generate a comprehensive performance report
  - Query Parameters:
    - `report_type`: Type of report (comprehensive, regression, hardware_comparison, efficiency, time_analysis)
    - `format`: Report format (json, markdown, html)
    - `test_type`: Filter by test type
    - `time_period`: Time period for analysis (e.g., "30d" for 30 days)

### Anomalies

- `GET /api/anomalies`: Get detected anomalies
  - Query Parameters:
    - `test_type`: Filter by test type
    - `start_time`: Filter by start time
    - `end_time`: Filter by end time

### Reports

- `GET /api/report`: Generate an analysis report
  - Query Parameters:
    - `report_type`: Type of report (performance, anomaly, summary)
    - `format`: Report format (json, markdown, html)
    - `test_type`: Filter by test type
    - `start_time`: Filter by start time
    - `end_time`: Filter by end time

### Visualizations

- `GET /api/visualizations/performance`: Generate a performance visualization
- `GET /api/visualizations/trends`: Generate a trend analysis visualization
- `GET /api/visualizations/anomalies`: Generate an anomaly dashboard
- `GET /api/visualizations/summary`: Generate a summary dashboard

### Notifications

- `GET /api/notifications`: Get recent notifications
  - Query Parameters:
    - `count`: Number of notifications to return

## Performance Analysis Features

The web dashboard includes advanced performance analysis capabilities that help you understand test performance across different hardware platforms, detect regressions, analyze resource efficiency, and visualize performance trends over time.

### Performance Regression Detection

The dashboard can automatically detect performance regressions by comparing recent test results with a baseline period. It applies statistical analysis to determine if the changes are significant and classifies regressions by severity.

Key features:
- Statistical significance testing to reduce false positives
- Severity classification (critical, major, minor)
- Detailed comparison of baseline and current metrics
- Support for analyzing multiple metrics simultaneously

### Hardware Performance Comparison

Compare test performance across different hardware profiles to determine which hardware is best suited for different types of tests. The analysis includes:

- Per-metric hardware comparison with statistical analysis
- Overall hardware scoring based on relative performance
- Best hardware recommendations by metric
- Comprehensive hardware performance matrix

### Resource Efficiency Analysis

Analyze resource efficiency metrics to optimize resource utilization:

- Throughput per memory unit analysis
- Throughput per power unit analysis
- Execution time per memory unit efficiency
- Configuration-specific efficiency metrics
- Identification of most efficient setups for different workloads

### Time-Based Performance Analysis

Advanced time series analysis of performance metrics with:

- Linear and polynomial regression modeling
- Trend detection and direction analysis
- Statistical validation of trends
- Performance forecasting
- Detailed time series visualization

### Comprehensive Performance Reports

Generate detailed performance reports that combine multiple analysis types:

- Multiple report formats (markdown, HTML, JSON)
- Combined analysis in a single comprehensive report
- Customizable report sections
- Support for filtering by test type and time period

## Example Usage

### Generating a Performance Report

```bash
curl "http://localhost:8050/api/performance/report?report_type=comprehensive&format=markdown&time_period=30d"
```

### Detecting Performance Regression

```bash
curl "http://localhost:8050/api/performance/regression?metric=throughput&baseline_period=14d&comparison_period=7d"
```

### Comparing Hardware Performance

```bash
curl "http://localhost:8050/api/performance/hardware?metrics=throughput,latency_ms,memory_usage_mb&time_period=30d"
```

### Analyzing Resource Efficiency

```bash
curl "http://localhost:8050/api/performance/efficiency?time_period=30d"
```

### Analyzing Performance Trends Over Time

```bash
curl "http://localhost:8050/api/performance/time?metric=throughput&grouping=day&time_period=90d"
```

### Fetching Recent Results

```bash
curl "http://localhost:8050/api/results?limit=10&test_type=benchmark&status=completed"
```

### Analyzing Basic Performance Trends

```bash
curl "http://localhost:8050/api/trends?metrics=throughput,latency&test_type=benchmark"
```

## Customizing the Dashboard

The dashboard can be customized by modifying the following files:

- `templates/layout.html`: Main layout template
- `templates/index.html`: Dashboard home page
- `templates/results.html`: Test results page
- `static/css/dashboard.css`: CSS styles

## Security Considerations

This implementation includes basic authentication but should be enhanced for production use:

1. Use HTTPS for secure communication
2. Implement proper user management and authorization
3. Protect sensitive data and API endpoints
4. Consider using a production-ready web server like Gunicorn or uWSGI

## End-to-End Integration Testing

An end-to-end integration test is provided to validate the complete flow from task creation to visualization in the dashboard. This test sets up a full distributed testing environment with a coordinator and simulated workers, generates test data, and verifies dashboard functionality.

### Running the End-to-End Test

Use the provided script to run the end-to-end test:

```bash
./run_e2e_web_dashboard_test.sh [OPTIONS]
```

Options:
- `--db-path PATH`: Path to DuckDB database (default: ./e2e_test_results.duckdb)
- `--coordinator-port PORT`: Port for coordinator (default: 8081)
- `--dashboard-port PORT`: Port for web dashboard (default: 8050)
- `--num-workers NUM`: Number of simulated workers (default: 5)
- `--num-tasks NUM`: Number of simulated tasks (default: 50)
- `--generate-anomalies`: Generate anomalous test results
- `--generate-trends`: Generate performance trends
- `--quick`: Run a quick test with fewer workers and tasks
- `--debug`: Enable debug mode
- `--open-browser`: Open web browser when dashboard is ready

### Example Usage

To run a comprehensive test that includes anomalies and trends:

```bash
./run_e2e_web_dashboard_test.sh --generate-anomalies --generate-trends --open-browser
```

For a quick test during development:

```bash
./run_e2e_web_dashboard_test.sh --quick --debug
```

### Test Process

The end-to-end test performs the following steps:

1. Initializes a coordinator instance with the IntegratedAnalysisSystem
2. Starts the web dashboard as a separate process
3. Creates simulated workers with various hardware capabilities
4. Generates and executes test tasks across different hardware profiles
5. Simulates successful completions and failures
6. Optionally generates anomalous results and performance trends
7. Runs comprehensive analysis including workload distribution, failure patterns, and circuit breaker evaluation
8. Generates summary, performance, and anomaly reports along with visualizations
9. Opens the dashboard in a web browser (if requested)

### Generated Reports

During the test, the following reports are generated in the `reports` directory:

- `summary_report.md`: Overview of test results
- `performance_report.md`: Detailed performance analysis
- `anomaly_report.md`: Analysis of detected anomalies (if any)

## Troubleshooting

### Dashboard won't start

- Check if all required dependencies are installed
- Verify the DuckDB database path is correct
- Ensure the port is available and not in use by another application

### Visualizations not showing

- Verify that Matplotlib and/or Plotly are installed
- Check server logs for visualization-related errors
- Ensure the `--enable-visualization` flag is set when starting the dashboard

### No real-time updates

- Verify that Flask-SocketIO is installed
- Check for WebSocket connection errors in the browser console
- Ensure the server is properly configured to handle WebSocket connections
- Check if the WebSocket indicator shows "WebSocket Connected" in the monitoring dashboard
- If using polling fallback, verify that auto-refresh is enabled and the interval is appropriate
- Check the dashboard logs for WebSocket connection and messaging errors
- Verify your browser supports WebSocket connections
- Try running with a different update interval using the `--update-interval` parameter

### End-to-End Test Failures

- Ensure that both coordinator and dashboard ports are available
- Check permissions for writing to the database and reports directory
- Verify that all dependencies are installed correctly
- Check the e2e_integration_test.log file for detailed error information