# Distributed Testing Dashboard

This directory contains the Monitoring Dashboard for the Distributed Testing Framework, including the Error Visualization system.

## Components

### Core Components

- `monitoring_dashboard.py`: The main dashboard implementation
- `monitoring_dashboard_routes.py`: HTTP and WebSocket route handlers
- `dashboard_server.py`: Standalone dashboard server
- `websocket_handlers.py`: WebSocket communication handlers

### Error Visualization

- `error_visualization_integration.py`: Error visualization integration component
- `static/sounds/`: Sound files for error notifications
- `templates/error_visualization.html`: Error visualization dashboard template

## Getting Started

### Running the Dashboard

Run the dashboard with error visualization enabled:

```bash
python run_monitoring_dashboard_with_error_visualization.py
```

Advanced options:

```bash
python run_monitoring_dashboard_with_error_visualization.py \
    --host 0.0.0.0 \
    --port 8080 \
    --db-path ./benchmark_db.duckdb \
    --theme dark
```

### Accessing the Dashboard

Access the dashboard in your web browser:

- Main dashboard: `http://localhost:8080/`
- Error visualization: `http://localhost:8080/error-visualization`

## Testing

Run all error visualization tests:

```bash
python run_error_visualization_tests.py
```

Run specific test types:

```bash
# Test sound notification system
python run_error_visualization_tests.py --type sound

# Test system-critical sound notifications specifically
python run_error_visualization_tests.py --test-system-critical

# Test severity detection
python run_error_visualization_tests.py --type severity

# Test WebSocket integration
python run_error_visualization_tests.py --type websocket

# Generate test reports
python run_error_visualization_tests.py --report --report-format html
```

Run the interactive sound demo:

```bash
# Open the sound demo in a browser
firefox ./dashboard/static/sounds/sound_demo.html

# Open the enhanced error notification demo
firefox ./dashboard/static/sounds/error_notification_demo.html
```

Run the system-critical error demo:

```bash
# From the dashboard/static/sounds directory
cd dashboard/static/sounds

# Test system-critical errors with default settings
python test_system_critical_demo.py

# Test with custom settings (more errors, slower interval)
python test_system_critical_demo.py --count 5 --interval 3
```

## Features

### Error Visualization

- Real-time error monitoring with WebSocket updates
- Hierarchical sound notifications based on error severity (system-critical, critical, warning, info)
- Enhanced system-critical alerts with distinctive rising frequency pattern (880Hz → 1046.5Hz → 1318.5Hz)
- Accelerating pulse rate (4Hz to 16Hz) for urgent attention to highest-priority infrastructure failures
- Specialized visual treatment for system-critical errors with continuous pulsing animation
- Persistent desktop notifications for system-critical alerts
- Error pattern detection and analysis
- Worker status monitoring
- Hardware error analysis
- Interactive charts and filtering

### Dashboard Features

- Task execution monitoring
- Worker node status tracking
- Result aggregation
- Performance analytics
- Error reporting and visualization
- Theme customization (light/dark)

## Documentation

For detailed documentation, please refer to:

- `ERROR_VISUALIZATION_GUIDE.md`: Comprehensive guide to the error visualization system
- `ERROR_VISUALIZATION_IMPLEMENTATION_GUIDE.md`: Implementation guide for integrating error reporting
- `ERROR_VISUALIZATION_STATUS.md`: Current status and features of the error visualization system

## Project Structure

```
dashboard/
├── __init__.py
├── dashboard_generator.py
├── dashboard_server.py
├── error_visualization_integration.py
├── monitoring_dashboard.py
├── monitoring_dashboard_e2e_integration.py
├── monitoring_dashboard_result_aggregator_integration.py
├── monitoring_dashboard_routes.py
├── monitoring_dashboard_visualization_integration.py
├── README.md
├── static/
│   └── sounds/
│       ├── error-system-critical.mp3
│       ├── error-critical.mp3
│       ├── error-warning.mp3
│       ├── error-info.mp3
│       ├── error-notification.mp3
│       ├── error-notification.txt
│       ├── generate_sound_files.py
│       ├── test_sound_files.py
│       ├── test_sound_notification_integration.py
│       ├── test_error_notification_system.py
│       ├── sound_demo.html
│       └── error_notification_demo.html
├── templates/
│   ├── dashboard_management.html
│   ├── e2e_test_monitoring.html
│   ├── e2e_test_results.html
│   ├── e2e_test_results_disabled.html
│   ├── error_visualization.html
│   ├── performance_analytics.html
│   ├── results.html
│   └── sidebar.html
├── tests/
│   ├── __init__.py
│   ├── test_dashboard_integration.py
│   └── test_monitoring_dashboard.py
├── visualization.py
└── websocket_handlers.py
```