# Test Dashboard for HuggingFace Models

This document describes the test dashboard for HuggingFace models, providing comprehensive visualizations for test results, hardware compatibility, and performance metrics.

## Overview

The test dashboard provides interactive and static visualizations for model testing data, hardware compatibility, and distributed testing results. It integrates with the broader testing framework to provide insights into test coverage, performance, and compatibility across different hardware platforms.

## Key Features

1. **Comprehensive Visualizations**:
   - Model coverage by architecture
   - Test success rates and inference types
   - Hardware compatibility matrix
   - Performance comparison across hardware platforms
   - Memory usage analysis
   - Distributed testing results
   - Worker performance metrics

2. **Multiple Dashboard Formats**:
   - Static HTML dashboard for sharing and documentation
   - Interactive Dash dashboard for real-time exploration
   - JSON data export for integration with other tools

3. **Data Integration**:
   - Collects test results from multiple sources
   - Integrates with DuckDB for historical data analysis
   - Processes distributed testing results
   - Analyzes hardware compatibility data

## Usage

### Generating Dashboard Data

```bash
# Generate dashboard data with default settings
python create_test_dashboard.py

# Generate dashboard data with specific sources
python create_test_dashboard.py --results-dir collected_results --dist-dir distributed_results --hardware-db hardware_compatibility_matrix.duckdb

# Generate data for the last 7 days
python create_test_dashboard.py --days 7
```

### Creating Static Dashboard

```bash
# Generate static HTML dashboard with default settings
python create_test_dashboard.py --static

# Specify output directory
python create_test_dashboard.py --static --output-dir dashboard_static
```

### Launching Interactive Dashboard

```bash
# Launch interactive dashboard server
python create_test_dashboard.py --interactive

# Specify port
python create_test_dashboard.py --interactive --port 8080
```

## Dashboard Components

### 1. Overview Page

The overview page provides a high-level summary of testing status:

- **Test Statistics**: Count of test results, model architectures, hardware platforms
- **Model Coverage Chart**: Pie chart showing test distribution by architecture
- **Success Rate Chart**: Stacked bar chart of success vs. failure rates by architecture
- **Mock vs. Real Inference**: Distribution of mock objects vs. real inference tests
- **Hardware Compatibility Matrix**: Heatmap of compatibility across model types and hardware platforms

### 2. Test Results Page

The test results page provides detailed test results and trends:

- **Performance Trend Chart**: Line chart of inference time trends by architecture
- **Test Results Table**: Detailed table of individual test results with filtering
- **Architecture Distribution**: Distribution of tests across different model architectures
- **Success Rate Metrics**: Detailed metrics on test success rates

### 3. Hardware Compatibility Page

The hardware compatibility page analyzes performance across hardware platforms:

- **Hardware Detection Cards**: Summary of available hardware platforms
- **Hardware Performance Comparison**: Bar chart comparing inference times across platforms
- **Performance by Model Type**: Grouped bar chart of performance by model and hardware
- **Memory Usage Chart**: Bar chart of memory usage by model type and hardware
- **Optimal Hardware Recommendations**: Recommendations for best hardware by model type

### 4. Distributed Testing Page

The distributed testing page provides insights into distributed test execution:

- **Distributed Test Runs Table**: Summary of distributed test runs
- **Worker Performance Chart**: Bar chart of worker execution times
- **Hardware Distribution Chart**: Pie chart of task distribution across hardware
- **Success Rate Analysis**: Analysis of test success rates in distributed mode
- **Performance Metrics**: Execution time and resource utilization metrics

## Data Sources

The dashboard integrates data from multiple sources:

1. **Test Results** (`collected_results/`):
   - Individual test results in JSON format
   - Contains model information, success status, execution time
   - Tracks mock vs. real inference status
   - Includes timestamp and environment details

2. **Hardware Compatibility Data** (`hardware_compatibility_matrix.duckdb`):
   - Comprehensive hardware compatibility matrix
   - Performance metrics across hardware platforms
   - Memory usage and execution time statistics
   - Hardware detection results

3. **Distributed Testing Results** (`distributed_results/`):
   - Distributed test run summaries
   - Worker execution details
   - Task distribution across hardware
   - Performance metrics by worker

## Customization

The dashboard can be customized in several ways:

1. **Data Sources**:
   - Specify alternative data directories with command-line arguments
   - Integrate with different database sources
   - Filter data by date range

2. **Visualization Options**:
   - Customize chart types and metrics
   - Add additional visualizations
   - Modify layout and styling

3. **Export Options**:
   - Export charts as HTML, PNG, or PDF
   - Export data in JSON or CSV format
   - Integrate with external reporting tools

## Dependencies

The dashboard requires the following dependencies:

- **Plotly**: For chart generation
- **Dash**: For interactive dashboard (optional)
- **Pandas**: For data processing
- **DuckDB**: For database integration (optional)

Install dependencies with:
```bash
pip install plotly dash pandas duckdb
```

## Integration with CI/CD

The dashboard can be integrated with CI/CD pipelines:

1. **GitHub Actions**:
   - Generate dashboard as part of CI workflow
   - Publish dashboard as GitHub Pages artifact
   - Track performance trends across commits

2. **GitLab CI**:
   - Generate dashboard during pipeline execution
   - Store dashboard as artifacts
   - Deploy dashboard to static hosting

3. **Jenkins**:
   - Generate dashboard as post-build step
   - Publish dashboard as Jenkins artifact
   - Monitor performance trends across builds

## Future Enhancements

Planned enhancements for the dashboard:

1. **Real-time Data Updates**:
   - Live updating of test results
   - WebSocket integration for real-time monitoring
   - Push notifications for test failures

2. **Advanced Analytics**:
   - Statistical analysis of performance patterns
   - Anomaly detection for performance regression
   - Trend forecasting for resource planning

3. **Expanded Visualizations**:
   - 3D visualization of multi-dimensional performance data
   - Network diagrams of distributed test execution
   - Timeline visualization of test history

4. **Mobile-Friendly Interface**:
   - Responsive design for mobile devices
   - Mobile-specific optimizations
   - Progressive web app capabilities

## Conclusion

The test dashboard provides a comprehensive visualization solution for HuggingFace model testing, hardware compatibility analysis, and distributed testing results. It enables data-driven decision making for hardware selection, optimization strategies, and test coverage planning, supporting the broader goals of the testing framework.

For more information, see the related documentation:
- [DISTRIBUTED_TESTING_README.md](DISTRIBUTED_TESTING_README.md)
- [HARDWARE_COMPATIBILITY_README.md](HARDWARE_COMPATIBILITY_README.md)
- [TEST_IMPLEMENTATION_SUMMARY.md](TEST_IMPLEMENTATION_SUMMARY.md)