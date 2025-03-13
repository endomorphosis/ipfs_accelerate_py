# Distributed Testing Dashboard

The Distributed Testing Dashboard provides a comprehensive visualization and monitoring system for the IPFS Accelerate Python Framework's distributed testing infrastructure. It enables real-time monitoring, interactive exploration of test results, and deep insights into performance trends across different hardware platforms and model configurations.

## Overview

The dashboard system consists of three main components:

1. **Visualization Engine** - Creates statistical visualizations and charts from test data
2. **Dashboard Generator** - Builds interactive HTML dashboards from aggregated results
3. **Dashboard Server** - Serves dashboards via HTTP with REST API and WebSocket support

This integrated system works with the Result Aggregator to provide intuitive visualizations of complex test data, enabling easier identification of performance trends, regressions, and optimization opportunities.

## Features

### Visualization Engine (`visualization.py`)

- Statistical visualization creation for performance metrics
- Support for multiple visualization types:
  - Time-series analysis charts
  - Dimension comparison charts (hardware, model, etc.)
  - Regression analysis visualizations
  - Correlation analysis charts
  - Performance heatmaps
- Theme customization (light/dark mode)
- Static image generation with Matplotlib
- Optional interactive visualizations with Plotly
- Visualization API for automated chart generation

### Dashboard Generator (`dashboard_generator.py`)

- Interactive HTML dashboard creation
- Multiple visualization sections:
  - Performance trends
  - Dimension analysis
  - Regression detection
  - Test details
  - Worker details
- Tabbed interface for different views
- Responsive design for different devices
- Color-coded performance indicators
- Automatic refresh options
- Customizable layout and content

### Dashboard Server (`dashboard_server.py`)

- Web server for hosting interactive dashboards
- REST API for programmatic access to results
- WebSocket support for real-time updates
- Template-based rendering with Jinja2
- Static file serving for visualizations
- API caching for improved performance
- Configurable server settings
- Browser compatibility for all major browsers

## Usage Examples

### Starting the Dashboard Server

```python
from duckdb_api.distributed_testing.dashboard.dashboard_server import DashboardServer
from duckdb_api.distributed_testing.result_aggregator.service import ResultAggregatorService

# Create result aggregator
result_aggregator = ResultAggregatorService(db_manager=my_db_manager)

# Create and start dashboard server
server = DashboardServer(
    host="localhost",
    port=8081,
    result_aggregator=result_aggregator,
    output_dir="./dashboards"
)

# Start server asynchronously
server_thread = server.start_async()
```

### Creating a Custom Visualization

```python
from duckdb_api.distributed_testing.dashboard.visualization import VisualizationEngine

# Create visualization engine
viz_engine = VisualizationEngine(output_dir="./visualizations")

# Create time series visualization
time_series_data = {
    "model1": [(datetime(2025, 3, 1), 156.7), (datetime(2025, 3, 2), 158.2)],
    "model2": [(datetime(2025, 3, 1), 142.3), (datetime(2025, 3, 2), 145.1)]
}

viz_path = viz_engine.create_visualization(
    "time_series",
    {
        "time_series": time_series_data,
        "metric": "throughput",
        "title": "Model Throughput Comparison"
    }
)
```

### Generating a Dashboard

```python
from duckdb_api.distributed_testing.dashboard.dashboard_generator import DashboardGenerator

# Create dashboard generator
dashboard_gen = DashboardGenerator(
    result_aggregator=result_aggregator,
    output_dir="./dashboards"
)

# Generate a comprehensive dashboard
dashboard_path = dashboard_gen.generate_dashboard()

# Generate a specific report
regression_report = dashboard_gen.generate_report("regression")
```

## API Reference

### VisualizationEngine

The `VisualizationEngine` creates various types of visualizations from test result data.

#### Key Methods

- `create_visualization(visualization_type, data, output_path=None)`: Creates a specific visualization
- `create_performance_dashboard(data=None, output_path=None)`: Creates a comprehensive performance dashboard
- `create_regression_report(data=None, output_path=None)`: Creates a regression analysis report
- `configure(config_updates)`: Updates the visualization engine configuration

#### Supported Visualization Types

- `time_series`: Time-based performance trends
- `dimension_comparison`: Comparisons across different dimensions (hardware, model, etc.)
- `regression_analysis`: Performance regression visualizations
- `correlation`: Correlation analysis between different metrics
- `heatmap`: Multi-dimensional performance heatmaps

### DashboardGenerator

The `DashboardGenerator` builds interactive HTML dashboards from test result data.

#### Key Methods

- `generate_dashboard(data=None, output_path=None)`: Generates a comprehensive dashboard
- `generate_report(report_type, data=None, output_path=None)`: Generates a specific report
- `configure(config_updates)`: Updates the dashboard generator configuration

#### Available Report Types

- `performance`: Comprehensive performance dashboard
- `regression`: Regression analysis report
- `dimension`: Dimension comparison report
- `worker`: Worker performance report
- `test`: Test details report

### DashboardServer

The `DashboardServer` provides a web interface for accessing dashboards and test results.

#### Key Methods

- `start()`: Starts the dashboard server
- `start_async()`: Starts the server in a separate thread
- `stop()`: Stops the server
- `broadcast(message)`: Broadcasts a message to all WebSocket connections
- `configure(config_updates)`: Updates the server configuration

#### API Endpoints

- `GET /`: Main index page
- `GET /dashboard`: Main dashboard view
- `GET /report/{report_type}`: Specific report view
- `GET /visualization/{viz_type}`: Specific visualization
- `GET /api/status`: Overall status information
- `GET /api/tests`: Test result data
- `GET /api/workers`: Worker information
- `GET /api/regressions`: Regression data
- `GET /api/dimensions`: Dimension analysis data
- `GET /api/performance`: Performance history data
- `GET /ws`: WebSocket endpoint for real-time updates

## Installation and Dependencies

The dashboard system depends on several Python packages:

```
aiohttp>=3.8.0          # Web server and WebSocket support
jinja2>=3.0.0           # HTML template rendering
matplotlib>=3.6.0       # Static visualization generation
plotly>=5.10.0          # Optional: Interactive visualizations
pandas>=1.4.0           # Data manipulation
numpy>=1.21.0           # Numerical operations
```

## Integration with Other Components

The dashboard system integrates closely with the following components:

- **Result Aggregator Service**: Provides aggregated test results for visualization
- **DuckDB API**: Supplies raw performance data from the benchmark database
- **Distributed Testing Framework**: Coordinates test execution and result collection
- **Performance Trend Analyzer**: Performs trend analysis for visualization

## Browser Compatibility

The dashboard is designed to work with all major browsers:

- **Chrome/Chromium**: Full support for all features
- **Firefox**: Full support with optimized WebSocket performance
- **Edge**: Full support with enhanced WebNN visualization
- **Safari**: Full support with some WebGL performance limitations

## Customization

The dashboard appearance and behavior can be customized through configuration:

```python
server.configure({
    "theme": "dark",             # Dashboard visual theme (light, dark)
    "auto_refresh": 60,          # Auto-refresh interval in seconds (0 to disable)
    "max_items_per_page": 50,    # Maximum items per page in data tables
    "default_report_type": "performance",  # Default report type
    "api_cache_time": 10         # API response cache time in seconds
})
```

## Advanced Usage

### Creating Custom Dashboard Templates

Custom dashboard templates can be placed in the `templates` directory:

```
dashboards/
└── templates/
    ├── index.html              # Main index template
    ├── dashboard.html          # Dashboard template
    └── report.html             # Report template
```

### Extending with Custom Visualizations

You can extend the visualization engine with custom visualization types:

```python
# Create a custom visualization creator
def create_custom_visualization(data, output_path):
    # Implementation...
    return output_path

# Register the custom visualization
viz_engine._create_custom_visualization = create_custom_visualization

# Use the custom visualization
viz_path = viz_engine.create_visualization("custom", data)
```

### Implementing Custom API Endpoints

The dashboard server can be extended with custom API endpoints:

```python
async def handle_custom_api(request):
    # Implementation...
    return web.json_response(data)

# Add custom route to the server
server.app.router.add_get("/api/custom", handle_custom_api)
```

## Troubleshooting

### Common Issues

- **Missing Visualizations**: Ensure the output directory exists and is writable
- **API Data Not Updating**: Check that the cache time is appropriate for your use case
- **WebSocket Connection Errors**: Verify that the server is running and accessible
- **Visualization Libraries Missing**: Install optional dependencies for enhanced visualizations

### Logging

The dashboard components provide detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned enhancements for future releases:

- Interactive 3D visualizations for multi-dimensional data
- Machine learning-based anomaly detection in visualizations
- Enhanced real-time collaboration features
- Mobile app integration with push notifications
- Additional export formats (PDF, PPT, interactive HTML)