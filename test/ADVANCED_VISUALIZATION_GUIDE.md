# Advanced Visualization Guide

## Overview

The Advanced Visualization System provides comprehensive visualization capabilities for the Predictive Performance System, enabling sophisticated analysis and presentation of model performance data across different hardware platforms, batch sizes, and precision formats.

Implemented in May 2025, this system offers both interactive (HTML/Plotly) and static (PNG/PDF) visualizations, making it suitable for both interactive exploration and report generation.

## Key Features

- **3D Visualizations**: Explore multi-dimensional relationships between performance metrics
- **Interactive Dashboards**: Filter and analyze performance metrics with interactive controls
- **Time-Series Visualizations**: Track performance metrics over time with trend detection
- **Power Efficiency Analysis**: Visualize performance relative to power consumption with efficiency contours
- **Dimension Reduction Visualizations**: Analyze feature importance through PCA or t-SNE
- **Confidence Visualization**: Display prediction uncertainties with confidence intervals
- **Batch Visualization Generation**: Generate multiple visualization types with a single command
- **Visualization Reports**: Combine multiple visualizations into comprehensive HTML reports

## Usage Guide

### Basic Usage

```python
# Import the visualization module
from predictive_performance.visualization import AdvancedVisualization, create_visualization_report

# Initialize visualization system (default: interactive=True)
vis = AdvancedVisualization(
    output_dir="./visualizations",
    interactive=True  # Set to False for static images
)

# Create a simple 3D visualization
vis.create_3d_visualization(
    data=performance_data,  # DataFrame or path to CSV/JSON
    x_metric="batch_size",
    y_metric="throughput", 
    z_metric="memory_usage",
    color_metric="hardware",
    title="3D Performance Visualization"
)

# Generate a performance dashboard
vis.create_performance_dashboard(
    data=performance_data,
    metrics=["throughput", "latency_mean", "memory_usage"],
    groupby=["model_name", "hardware"],
    title="Performance Dashboard"
)

# Create a time-series visualization
vis.create_time_series_visualization(
    data=performance_data,
    time_column="timestamp",
    metric="throughput", 
    groupby=["model_name", "hardware"],
    title="Performance Over Time"
)

# Generate a complete set of visualizations
visualization_files = vis.create_batch_visualizations(
    data=performance_data,
    metrics=["throughput", "latency_mean", "memory_usage"],
    groupby=["model_category", "hardware"],
    include_3d=True,
    include_time_series=True,
    include_power_efficiency=True,
    include_dimension_reduction=True,
    include_confidence=True
)

# Create a comprehensive report
report_path = create_visualization_report(
    visualization_files=visualization_files,
    title="Performance Visualization Report",
    output_file="visualization_report.html",
    output_dir="./visualizations"
)
```

### Running the Demo Script

```bash
# Run basic demo with default settings
python run_visualization_demo.py --demo

# Run with advanced visualization features
python run_visualization_demo.py --demo --advanced-vis

# Use predictions from the performance prediction system
python run_visualization_demo.py --generate --advanced-vis

# Use your own data
python run_visualization_demo.py --data performance_data.json --advanced-vis

# Specify output directory
python run_visualization_demo.py --demo --output-dir ./custom_output
```

## Visualization Types

### 3D Visualizations

3D visualizations allow exploring the relationships between three metrics simultaneously, with additional dimensions represented through color and size.

```python
# Create a 3D visualization
vis.create_3d_visualization(
    data=df,
    x_metric="batch_size",
    y_metric="throughput",
    z_metric="memory_usage",
    color_metric="hardware",
    size_metric="confidence",
    title="3D Performance Visualization"
)
```

Key features:
- **Interactive Rotation**: In HTML output, rotate and zoom to explore the 3D space
- **Color Coding**: Differentiate categories (e.g., hardware platforms)
- **Size Mapping**: Represent additional metrics through point size
- **Hover Information**: Display detailed information on hover in interactive mode

### Performance Dashboards

Performance dashboards provide comparative visualizations of metrics across different groupings.

```python
# Create a performance dashboard
vis.create_performance_dashboard(
    data=df,
    metrics=["throughput", "latency_mean", "memory_usage"],
    groupby=["model_category", "hardware"],
    title="Performance Dashboard"
)
```

Key features:
- **Multi-Metric Views**: Compare multiple metrics in a single dashboard
- **Grouped Comparisons**: Compare performance across different categories
- **Interactive Filtering**: Filter data points in interactive mode
- **Sortable Metrics**: Sort by different metrics in interactive mode

### Time-Series Visualizations

Time-series visualizations show how performance metrics change over time, with trend detection and anomaly highlighting.

```python
# Create a time-series visualization
vis.create_time_series_visualization(
    data=df,
    time_column="timestamp",
    metric="throughput",
    groupby=["model_name", "hardware"],
    include_trend=True,
    window_size=5,
    title="Performance Over Time"
)
```

Key features:
- **Trend Lines**: Show moving averages to identify trends
- **Multi-Series Comparison**: Compare time series for different categories
- **Zoom and Pan**: Zoom into specific time ranges in interactive mode
- **Anomaly Highlighting**: Identify outlier points that deviate from trends

### Power Efficiency Visualizations

Power efficiency visualizations show the relationship between performance and power consumption, with efficiency contours.

```python
# Create a power efficiency visualization
vis.create_power_efficiency_visualization(
    data=df,
    performance_metric="throughput",
    power_metric="power_consumption",
    groupby=["model_category", "hardware"],
    title="Power Efficiency Analysis"
)
```

Key features:
- **Efficiency Contours**: Show lines of constant efficiency
- **Scatter Plot**: Position points by power and performance
- **Quadrant Analysis**: Identify high-performance, low-power configurations
- **Efficiency Metrics**: Calculate and display efficiency metrics

### Dimension Reduction Visualizations

Dimension reduction visualizations use PCA or t-SNE to analyze feature importance and relationships.

```python
# Create a dimension reduction visualization
vis.create_dimension_reduction_visualization(
    data=df,
    features=["batch_size", "memory_usage", "power_consumption", "latency_mean"],
    target="throughput",
    method="pca",  # or "tsne"
    n_components=2,
    groupby="model_category",
    title="Feature Importance Visualization"
)
```

Key features:
- **PCA Analysis**: Show principal components and feature loadings
- **t-SNE Visualization**: Reveal complex non-linear relationships
- **Feature Importance**: Identify which features have the most impact
- **Cluster Detection**: Identify natural clusters in the data

### Prediction Confidence Visualizations

Confidence visualizations show prediction uncertainties with confidence intervals and reliability indicators.

```python
# Create a prediction confidence visualization
vis.create_prediction_confidence_visualization(
    data=df,
    metric="throughput",
    confidence_column="confidence",
    groupby=["model_category", "hardware"],
    title="Prediction Confidence Visualization"
)
```

Key features:
- **Confidence Intervals**: Show prediction ranges
- **Visual Uncertainty**: Represent prediction certainty through opacity
- **Error Bars**: Display upper and lower bounds
- **Confidence Scoring**: Color-code based on prediction reliability

## Batch Visualization Generation

The `create_batch_visualizations` function generates a complete set of visualizations for a dataset.

```python
# Generate a complete set of visualizations
visualization_files = vis.create_batch_visualizations(
    data=df,
    metrics=["throughput", "latency_mean", "memory_usage"],
    groupby=["model_category", "hardware"],
    include_3d=True,
    include_time_series=True,
    include_power_efficiency=True,
    include_dimension_reduction=True,
    include_confidence=True
)
```

This function returns a dictionary mapping visualization types to lists of output file paths.

## Visualization Reports

The `create_visualization_report` function combines multiple visualizations into a comprehensive HTML report.

```python
# Create a comprehensive report
report_path = create_visualization_report(
    visualization_files=visualization_files,
    title="Performance Visualization Report",
    output_file="visualization_report.html",
    output_dir="./visualizations"
)
```

The report includes all visualizations with appropriate grouping, making it easy to navigate and explore the results.

## Customization Options

The `AdvancedVisualization` class provides several customization options:

- **Style**: Set visualization style (default: "whitegrid")
- **Context**: Set context for visualizations (default: "paper")
- **Palette**: Set color palette (default: "viridis")
- **Figure Size**: Set default figure size (default: (12, 8))
- **DPI**: Set DPI for static images (default: 100)
- **Output Format**: Set default output format for static images (default: "png")
- **Output Directory**: Set directory for output files (default: "./visualizations")
- **Interactive**: Whether to create interactive visualizations (default: True)

Example:
```python
# Customize visualization settings
vis = AdvancedVisualization(
    style="darkgrid",
    context="talk",
    palette="Set2",
    figure_size=(16, 10),
    dpi=150,
    output_format="svg",
    output_dir="./custom_visualizations",
    interactive=True
)
```

## Integration with Predictive Performance System

The visualization system is fully integrated with the Predictive Performance System, allowing direct visualization of predictions and benchmarks.

```python
# Import the prediction module
from predictive_performance.predict import PerformancePredictor

# Make predictions for various configurations
predictor = PerformancePredictor()
predictions = []

# Generate predictions for different configurations
for model_name in ["bert-base-uncased", "t5-small", "vit-base"]:
    for hardware in ["cpu", "cuda", "webgpu"]:
        for batch_size in [1, 8, 32]:
            prediction = predictor.predict(
                model_name=model_name,
                model_type=model_name.split("-")[0],
                hardware_platform=hardware,
                batch_size=batch_size
            )
            predictions.append(prediction)

# Convert predictions to DataFrame
import pandas as pd
df = pd.DataFrame(predictions)

# Visualize predictions
vis = AdvancedVisualization()
visualization_files = vis.create_batch_visualizations(df)
```

For automated visualization of predictions, use the `run_visualization_demo.py` script with the `--generate` flag.

## Best Practices

- **Interactive vs. Static**: Use interactive visualizations for exploration and static images for reports
- **Data Preparation**: Ensure timestamp data is in ISO format for time-series visualizations
- **Feature Selection**: Choose meaningful metrics and groupings for visualizations
- **Report Generation**: Use the `create_visualization_report` function to combine visualizations
- **Customization**: Adjust visualization style and context based on your needs
- **Integration**: Combine with the Predictive Performance System for end-to-end workflows

## Conclusion

The Advanced Visualization System provides powerful tools for visualizing and analyzing model performance data. By combining interactive and static visualizations with comprehensive reporting capabilities, it enables deep insights into performance patterns across different hardware platforms, batch sizes, and model architectures.