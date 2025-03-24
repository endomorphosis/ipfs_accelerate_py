# Advanced Visualization Guide

## Overview

The Advanced Visualization System provides comprehensive visualization capabilities for the IPFS Accelerate Framework, enabling sophisticated analysis and presentation of model performance data across different hardware platforms, batch sizes, and precision formats.

This system offers both interactive (HTML/Plotly) and static (PNG/PDF) visualizations, making it suitable for both interactive exploration and report generation.

## New Features (March 2025 Update)

The Advanced Visualization System has been enhanced with the following new features:

- **Interactive 3D Performance Visualizations**: Explore relationships between multiple performance metrics in 3D space with interactive controls
- **Dynamic Hardware Comparison Heatmaps**: Compare performance across hardware platforms and model families with interactive filtering
- **Power Efficiency Visualization**: Analyze the efficiency of different hardware platforms and models with interactive filters
- **Animated Time-Series Visualizations**: Track performance metrics over time with animation controls and trend analysis 
- **Customizable Dashboards**: Create and save customizable dashboards with multiple visualization components
- **Monitoring Dashboard Integration**: Seamless integration with the Distributed Testing Framework's Monitoring Dashboard
- **Customizable Themes**: Light and dark themes with configurable color schemes
- **Enhanced Interactivity**: Dropdown filters, camera controls, and advanced tooltips

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

# Test specific visualization types
python test_advanced_visualization.py --viz-type 3d
python test_advanced_visualization.py --viz-type heatmap
python test_advanced_visualization.py --viz-type power
python test_advanced_visualization.py --viz-type time-series

# Run all visualization types
python test_advanced_visualization.py --viz-type all

# Customize time-series visualization
python test_advanced_visualization.py --viz-type time-series --db-path ./benchmark_db.duckdb --output-dir ./visualizations --no-open

# Create comprehensive visualization report with all types
python generate_comprehensive_visualization_report.py --include-all --output-dir ./reports
```

#### Advanced Command-line Options for Time-Series Visualization

The time-series visualization supports additional command line options through a specialized script:

```bash
# Run time-series visualization with custom settings
python run_timeseries_visualization.py --metric throughput_items_per_second --dimensions model_family,hardware_type --time-range 90 --interval day

# Create comparative time-series visualization
python run_timeseries_visualization.py --comparative --metrics throughput_items_per_second,average_latency_ms --dimensions hardware_type

# Generate animation export
python run_timeseries_visualization.py --export-animation --format mp4 --output animation.mp4

# Create visual dashboard with multiple time-series views
python run_timeseries_visualization.py --dashboard --metrics throughput_items_per_second,memory_peak_mb,energy_consumption_joules --output dashboard.html
```

## Visualization Types

### 3D Visualizations

The system provides two implementations for 3D visualizations:

#### Interactive 3D Visualization Component

Our newly implemented `Visualization3D` component provides advanced interactive 3D visualizations for exploring complex relationships between performance metrics:

```python
from duckdb_api.visualization.advanced_visualization import Visualization3D

# Create a 3D visualization component
viz = Visualization3D(theme="light")

# Create a 3D scatter plot comparing throughput, memory, and latency
viz.create_3d_visualization(
    x_metric="throughput",
    y_metric="memory",
    z_metric="latency",
    color_by="hardware_type",
    model_families=["Text", "Vision"],
    hardware_types=["CPU", "GPU", "WebGPU"],
    output_path="3d_scatter.html",
    title="Hardware Performance Comparison: Throughput vs Memory vs Latency"
)
```

This component offers several visualization modes:

1. **3D Scatter Plot**: Basic 3D scatter plot for exploring relationships between metrics
   ```python
   viz.create_3d_visualization(
       x_metric="throughput",
       y_metric="memory",
       z_metric="latency",
       color_by="hardware_type",
       output_path="3d_scatter.html"
   )
   ```

2. **3D Surface Plot**: Surface plot for visualizing continuous relationships
   ```python
   viz.create_3d_visualization(
       x_metric="throughput",
       y_metric="batch_size",
       z_metric="latency",
       show_surface=True,
       surface_contours=True,
       output_path="3d_surface.html"
   )
   ```

3. **3D Clustered Plot**: Automatically cluster data points to identify patterns
   ```python
   viz.create_3d_visualization(
       x_metric="throughput",
       y_metric="memory",
       z_metric="latency",
       cluster_points=True,
       num_clusters=4,
       show_cluster_centroids=True,
       output_path="3d_clustered.html"
   )
   ```

4. **Animated 3D Plot**: Animate the plot over a dimension (e.g., batch size)
   ```python
   viz.create_3d_visualization(
       x_metric="throughput",
       y_metric="memory",
       z_metric="latency",
       enable_animation=True,
       animation_frame="batch_size",
       output_path="3d_animated.html"
   )
   ```

5. **3D Plot with Regression Plane**: Add a regression plane to identify trends
   ```python
   viz.create_3d_visualization(
       x_metric="throughput",
       y_metric="memory",
       z_metric="latency",
       regression_plane=True,
       output_path="3d_regression.html"
   )
   ```

Key features:
- **Interactive 3D Exploration**: Rotate, zoom, and pan to explore data from any angle
- **Multiple Visualization Types**: Choose from scatter, surface, or clustered visualizations
- **Customizable Appearance**: Control colors, marker sizes, opacity, and more
- **Detailed Hover Information**: See comprehensive details when hovering over points
- **Statistical Analysis**: Analyze trends with regression planes and clustering
- **Animation Support**: Create animated visualizations to explore additional dimensions
- **Auto-Rotation**: Enable auto-rotation for presentations
- **Wall Projections**: Show projections on walls for better perspective
- **Flexible Output**: Export to HTML (interactive) or PNG (static)

#### Legacy 3D Visualization

The system also includes the original 3D visualization implementation:

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
# Create a basic time-series visualization
vis.create_time_series_visualization(
    data=df,
    time_column="timestamp",
    metric="throughput",
    groupby=["model_name", "hardware"],
    include_trend=True,
    window_size=5,
    title="Performance Over Time"
)

# Create an animated time-series visualization with interactive controls
vis.create_animated_time_series_visualization(
    data=df,
    metric="throughput_items_per_second",
    dimensions=["model_family", "hardware_type"],
    time_range=90,  # Show last 90 days
    time_interval="day",  # Aggregate by day
    include_trend=True,
    window_size=5,
    title="Performance Trends Over Time"
)
```

Key features:
- **Trend Lines**: Show moving averages to identify trends
- **Multi-Series Comparison**: Compare time series for different categories
- **Zoom and Pan**: Zoom into specific time ranges in interactive mode
- **Anomaly Highlighting**: Identify outlier points that deviate from trends
- **Animation Controls**: Play, pause, and step through time with animation controls
- **Interactive Time Slider**: Jump to specific points in time
- **Progressive Visualization**: Watch how metrics evolve over time
- **Configurable Time Intervals**: Aggregate by hour, day, week, or month

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
- **Animation Controls**: For time-series animations, use window_size parameter to adjust trend smoothness
- **Dimensional Analysis**: Compare performance across different dimensions to identify patterns
- **Performance Monitoring**: Create regular time-series visualizations for ongoing monitoring
- **Benchmark Comparison**: Use animated visualizations to compare performance across benchmarks

## New Visualization Components (June 2025)

The new visualization components introduced in June 2025 provide enhanced interactive capabilities using Plotly. These components are implemented in the `advanced_visualization.py` module.

### Animated Time-Series Performance Visualization

The Animated Time-Series Performance Visualization allows tracking performance metrics over time with interactive animation controls and trend analysis.

#### Overview

```python
from duckdb_api.visualization.advanced_visualization import AnimatedTimeSeriesVisualization

# Create a visualization component
viz = AnimatedTimeSeriesVisualization(theme="light")

# Create a basic animated time series visualization
viz.create_animated_time_series(
    metric="throughput",
    dimensions=["hardware_type"],
    time_interval="day",
    time_range=90,  # Last 90 days of data
    title="Throughput Over Time by Hardware Type"
)

# Advanced usage with trend analysis and anomaly detection
viz.create_animated_time_series(
    metric="latency",
    dimensions=["model_family", "hardware_type"],
    time_range=180,  # Last 6 months of data
    time_interval="week",  # Aggregate by week
    show_trend=True,
    trend_window=5,  # Window size for moving average
    show_anomalies=True,  # Detect and highlight anomalies
    anomaly_threshold=2.0,  # Z-score threshold for anomaly detection
    title="Latency Trends with Anomaly Detection",
    filters={
        "batch_size": [1, 4, 16, 32],  # Only specific batch sizes
        "model_family": ["Text", "Vision"]  # Only specific model families
    }
)

# Add event markers to correlate with performance changes
viz.create_animated_time_series(
    metric="throughput",
    dimensions=["model_family"],
    time_interval="day",
    events=[
        {"date": "2025-05-15", "label": "Framework v2.1 Release", "color": "green"},
        {"date": "2025-06-01", "label": "Hardware Upgrade", "color": "blue"},
        {"date": "2025-06-10", "label": "Config Change", "color": "orange"}
    ],
    title="Performance Impact of System Changes"
)
```

#### Key Features

- **Interactive Animation**: Play, pause, and step through time using animation controls
- **Time Slider**: Interactive timeline slider for jumping to specific points in time
- **Progressive Display**: See how data accumulates over time (or disable to show only current data)
- **Trend Analysis**: Visualize moving average trend lines with configurable window size
- **Anomaly Detection**: Automatically identify and highlight data points that deviate from normal patterns
- **Event Markers**: Add vertical markers for significant events to correlate with performance changes
- **Dimension Grouping**: Group by multiple dimensions (model family, hardware type, etc.)
- **Time Aggregation**: Aggregate data by hour, day, week, or month with configurable intervals
- **Animation Controls**: 
  - Play/pause controls for automatic playback
  - Step forward/backward buttons for frame-by-frame analysis
  - Speed selector (0.5x, 1x, 2x, 5x) for adjusting animation pace
- **Detailed Tooltips**: Rich hover information showing all relevant data for each point
- **Animation Export**: Export animations as MP4 or GIF files for presentations and sharing
- **Adaptive Output**: Generates interactive HTML with Plotly or static PNG with Matplotlib based on available libraries

#### Animation Controls and Interaction

The animation controls provide a rich interactive experience:

1. **Basic Controls**:
   - **Play Button**: Start the animation to see how metrics change over time
   - **Pause Button**: Stop the animation at any point for detailed examination
   - **Step Forward/Backward**: Move one time step at a time for precise analysis

2. **Timeline Slider**:
   - Interactive slider showing all time points
   - Click or drag to jump to specific dates
   - Shows current date during animation

3. **Speed Controls**:
   - 0.5x: Slow motion for detailed examination
   - 1x: Normal playback speed
   - 2x: Accelerated playback for faster review
   - 5x: High-speed playback for quick overview

#### Progressive vs. Non-Progressive Mode

The visualization supports two display modes:

1. **Progressive Mode** (default): 
   - Shows all data accumulated up to the current date
   - Allows seeing how trends develop over time
   - Shows the full history at each point

2. **Non-Progressive Mode**:
   - Shows only data for the current date
   - Useful for isolating specific time periods
   - Best for comparing discrete time points

To switch modes:

```python
# Enable progressive display (default)
viz.create_animated_time_series(
    metric="throughput",
    progressive_display=True
)

# Disable progressive display
viz.create_animated_time_series(
    metric="throughput",
    progressive_display=False
)
```

#### Event Correlation

The visualization supports adding event markers to correlate performance changes with system events:

```python
# Add event markers to timeline
viz.create_animated_time_series(
    metric="throughput",
    events=[
        {"date": "2025-05-15", "label": "Framework v2.1 Release", "color": "green"},
        {"date": "2025-06-01", "label": "Hardware Upgrade", "color": "blue"},
        {"date": "2025-06-10", "label": "Config Change", "color": "orange", "effect": "negative"}
    ]
)
```

Event markers create vertical lines on the timeline with annotations, making it easy to see how specific events correlate with performance changes. Events can also include an optional `effect` parameter to indicate whether the expected effect is positive or negative.

#### Trend Analysis and Anomaly Detection

The component provides built-in trend analysis and anomaly detection:

```python
# Enable trend analysis and anomaly detection
viz.create_animated_time_series(
    metric="throughput",
    show_trend=True,
    trend_window=7,  # 7-day moving average
    show_anomalies=True,
    anomaly_threshold=2.5  # Z-score threshold
)
```

- **Trend Analysis**: Shows moving average trend lines to identify patterns
- **Anomaly Detection**: Automatically highlights data points that deviate significantly from normal patterns (based on Z-score)

#### Time Interval Aggregation

Data can be aggregated at different time intervals:

```python
# Hourly aggregation
viz.create_animated_time_series(
    metric="throughput",
    time_interval="hour"
)

# Daily aggregation (default)
viz.create_animated_time_series(
    metric="throughput",
    time_interval="day"
)

# Weekly aggregation
viz.create_animated_time_series(
    metric="throughput",
    time_interval="week"
)

# Monthly aggregation
viz.create_animated_time_series(
    metric="throughput",
    time_interval="month"
)
```

#### Multi-Dimensional Filtering and Grouping

The visualization supports filtering and grouping by multiple dimensions:

```python
# Group by hardware type
viz.create_animated_time_series(
    metric="throughput",
    dimensions=["hardware_type"]
)

# Group by multiple dimensions
viz.create_animated_time_series(
    metric="throughput",
    dimensions=["model_family", "hardware_type"]
)

# Apply filters to specific dimensions
viz.create_animated_time_series(
    metric="throughput",
    dimensions=["model_family"],
    filters={
        "hardware_type": ["GPU", "WebGPU"],  # Only GPU and WebGPU hardware
        "batch_size": [1, 4, 16]  # Only specific batch sizes
    }
)
```

#### Animation Export

The component supports exporting animations to different formats:

```python
# Create and display an animated visualization
viz.create_animated_time_series(
    metric="throughput",
    dimensions=["hardware_type"],
    output_path="animated_throughput.html"
)

# Export the animation as MP4
viz.export_animation(output_format="mp4", output_path="animated_throughput.mp4")

# Export the animation as GIF
viz.export_animation(output_format="gif", output_path="animated_throughput.gif", fps=15)
```

Export options include:
- **MP4**: High-quality video format, good for presentations and documentation
- **GIF**: Widely compatible animated format, good for sharing and embedding in documents
- **HTML**: Interactive web format with full animation controls (default)

#### Command-Line Usage

The visualization component can be accessed through a command-line interface:

```bash
# Basic usage
python test_animated_time_series.py

# Specify days of data to generate
python test_animated_time_series.py --days 180

# Change time interval
python test_animated_time_series.py --interval week

# Export animations
python test_animated_time_series.py --export-mp4 --export-gif

# Specify output directory
python test_animated_time_series.py --output-dir ./custom_visualizations
```

#### Use Cases

This visualization is particularly useful for:

1. **Performance Monitoring**:
   - Track performance metrics over time to identify trends
   - Detect performance regressions early
   - Observe how performance evolves with system changes

2. **Comparative Analysis**:
   - Compare performance across different hardware platforms
   - Analyze how different model families perform over time
   - See how various configurations impact performance trends

3. **Event Impact Analysis**:
   - Correlate performance changes with system events
   - Visualize the impact of optimizations or configuration changes
   - Understand how external factors affect performance

4. **Anomaly Investigation**:
   - Automatically detect unusual performance patterns
   - Identify outliers that require further investigation
   - Find temporal patterns in performance anomalies

5. **Demonstration and Presentation**:
   - Create dynamic visualizations for presentations
   - Export animations for documentation and reports
   - Demonstrate performance improvements to stakeholders

#### Interactive vs. Static Implementation

The component provides two implementation paths:

1. **Interactive (Plotly)**:
   - Full animation controls with play, pause, step buttons
   - Interactive timeline slider
   - Speed controls
   - HTML output with embedded JavaScript

2. **Static (Matplotlib)**:
   - Static image outputs when Plotly is unavailable
   - Animation export to MP4/GIF for non-interactive viewing
   - PNG output for reports and documentation

The system automatically selects the best implementation based on available libraries.

#### Complete Example

```python
from duckdb_api.visualization.advanced_visualization import AnimatedTimeSeriesVisualization

# Create a visualization component
viz = AnimatedTimeSeriesVisualization(theme="dark")

# Create a comprehensive animated time series visualization
viz.create_animated_time_series(
    metric="throughput",
    dimensions=["model_family", "hardware_type"],
    time_range=90,
    time_interval="day",
    show_trend=True,
    trend_window=7,
    show_anomalies=True,
    anomaly_threshold=2.0,
    progressive_display=True,
    show_timeline_slider=True,
    control_buttons=True,
    speed_selector=True,
    events=[
        {"date": "2025-05-15", "label": "Framework v2.1 Release", "color": "green"},
        {"date": "2025-06-01", "label": "Hardware Upgrade", "color": "blue"}
    ],
    filters={
        "batch_size": [1, 4, 16],
        "model_family": ["Text", "Vision"]
    },
    title="Comprehensive Performance Visualization",
    output_path="comprehensive_animation.html"
)

# Export the animation in multiple formats
viz.export_animation(output_format="mp4", output_path="comprehensive_animation.mp4")
viz.export_animation(output_format="gif", output_path="comprehensive_animation.gif")
```

This comprehensive example showcases all major features of the Animated Time Series Visualization component, creating a rich interactive visualization for exploring performance trends over time.

### Interactive 3D Performance Visualization

The Interactive 3D Performance Visualization allows exploring the relationships between three performance metrics in a fully interactive 3D space.

```python
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
from duckdb_api.visualization.advanced_visualization import AdvancedVisualizationSystem

# Initialize database API and visualization system
db_api = BenchmarkDBAPI(db_path="./benchmark_db.duckdb")
viz = AdvancedVisualizationSystem(db_api=db_api, output_dir="./visualizations")

# Create a 3D performance visualization
viz.create_3d_performance_visualization(
    metrics=["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
    dimensions=["model_family", "hardware_type", "batch_size", "precision"],
    title="3D Performance Visualization"
)
```

Key features:
- **Multiple Metrics**: Visualize three performance metrics along the x, y, and z axes
- **Color Coding**: Points are color-coded by a dimension such as model family or hardware type
- **Interactive Camera Controls**: Rotate, pan, and zoom with the mouse
- **Detailed Tooltips**: Hover over points to see detailed information
- **Camera Position Controls**: Quickly switch between different views with the camera buttons
- **HTML Output**: Interactive visualizations are saved as HTML files with embedded JavaScript

### Dynamic Hardware Comparison Heatmap

The Dynamic Hardware Comparison Heatmap visualizes performance metrics across hardware platforms and model families in a structured format.

```python
# Create hardware comparison heatmap
viz.create_hardware_comparison_heatmap(
    metric="throughput",
    model_families=["transformers", "vision", "audio"],
    hardware_types=["nvidia_a100", "amd_mi250", "intel_arc", "apple_m2"],
    batch_size=1,
    title="Hardware Comparison by Model Family"
)
```

Key features:
- **Family-Based Organization**: Models are grouped by family for more structured comparison
- **Color-Coded Metrics**: Performance metrics are color-coded for easy visual comparison
- **Simulated Result Markers**: Clearly identifies simulated results with markers
- **Interactive Controls**: Toggle simulated result markers on/off
- **Metric-Aware Colormaps**: Uses appropriate colormaps for different metrics (higher or lower is better)
- **Multi-Family Subplots**: Creates a separate subplot for each model family

### Power Efficiency Visualization

The Power Efficiency Visualization analyzes the relationship between throughput and energy consumption.

```python
# Create power efficiency visualization
viz.create_power_efficiency_visualization(
    hardware_types=["nvidia_a100", "amd_mi250", "intel_arc"],
    model_families=["transformers", "vision", "audio"],
    batch_sizes=[1, 8, 32],
    title="Power Efficiency Visualization"
)
```

Key features:
- **Efficiency Metric**: Calculates and visualizes throughput per joule
- **Efficiency Contours**: Shows lines of constant efficiency
- **Multi-Dimensional Data**: Points sized by latency and colored by efficiency
- **Interactive Filtering**: Filter by model family and hardware type with dropdowns
- **Hover Details**: Comprehensive information on hover including all performance metrics
- **Dynamic Filtering**: Interactive controls for exploring different subsets of the data

### Usage with Command-Line Interface

The new visualization components can be accessed through a command-line interface:

```bash
# Create a 3D visualization
python test_advanced_visualization.py --viz-type 3d --db-path ./benchmark_db.duckdb --output-dir ./visualizations

# Create a hardware comparison heatmap
python test_advanced_visualization.py --viz-type heatmap --db-path ./benchmark_db.duckdb --output-dir ./visualizations

# Create a power efficiency visualization
python test_advanced_visualization.py --viz-type power --db-path ./benchmark_db.duckdb --output-dir ./visualizations

# Create an animated time-series visualization
python test_advanced_visualization.py --viz-type time-series --db-path ./benchmark_db.duckdb --output-dir ./visualizations

# Create all visualization types
python test_advanced_visualization.py --viz-type all --db-path ./benchmark_db.duckdb --output-dir ./visualizations
```

### Configuration Options

The Advanced Visualization System can be configured with various options:

```python
viz.configure({
    "theme": "dark",             # 'light' or 'dark'
    "color_palette": "plasma",   # 'viridis', 'plasma', 'inferno', etc.
    "default_width": 1200,       # Width in pixels
    "default_height": 900,       # Height in pixels
    "auto_open": True,           # Automatically open visualizations in browser
    "include_annotations": True, # Include annotations on charts
    "animation_duration": 1500,  # Animation duration in milliseconds
    "include_controls": True,    # Include interactive controls
    "save_data": True,           # Save data alongside visualizations
})
```

### Dependencies

The new visualization components require the following Python packages:

```bash
pip install plotly pandas scikit-learn ipywidgets
```

You can install all dependencies using the provided setup script:

```bash
# Make the script executable if needed
chmod +x setup_advanced_visualization.sh

# Run the setup script
./setup_advanced_visualization.sh
```

### Integration with Existing Code

The Advanced Visualization System integrates seamlessly with the existing DuckDB database architecture:

```python
# Import the necessary modules
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
from duckdb_api.visualization.advanced_visualization import AdvancedVisualizationSystem

# Initialize the database API with your benchmark database
db_api = BenchmarkDBAPI(db_path="./benchmark_db.duckdb")

# Initialize the visualization system
viz = AdvancedVisualizationSystem(db_api=db_api, output_dir="./visualizations")

# Create visualizations using the database data
viz.create_3d_performance_visualization()
```

### Extending the Visualization System

You can extend the Advanced Visualization System with custom visualizations by subclassing the `AdvancedVisualizationSystem` class:

```python
class CustomVisualizationSystem(AdvancedVisualizationSystem):
    def create_custom_visualization(self, **kwargs):
        """Create a custom visualization."""
        # Implementation here
        pass
```

### Troubleshooting

Common issues and solutions:

1. **No data appears in visualizations**
   - Check that your database contains valid benchmark data
   - Verify that the metrics you're requesting exist in the database
   - Try using different filter criteria
   - For time-series visualizations, ensure sufficient date range coverage

2. **Import errors**
   - Ensure all dependencies are installed using the setup script
   - Verify that the Python path includes the parent directory
   - Run `pip install -r advanced_visualization_requirements.txt` to install all dependencies

3. **Browser doesn't open automatically**
   - Set `auto_open=False` in the configuration and manually open the HTML files
   - Check that you have a default browser configured in your system

4. **Visualizations take too long to generate**
   - Consider filtering your data to reduce the number of points
   - Set `include_controls=False` to simplify the visualizations
   - For time-series animations, reduce the number of time buckets with a larger time_interval

5. **Animation controls not working**
   - Ensure you have the latest version of Plotly installed (v5.18.0+)
   - Check browser console for JavaScript errors
   - Try using Chrome or Firefox for best animation support
   - Verify that animation_duration in config is set to a reasonable value (500-2000ms)

6. **Database connectivity issues**
   - Verify database path exists and is accessible
   - Check database permissions
   - Try running a simple query first to verify connectivity

## Advanced Time-Series Visualization Techniques

The animated time-series visualization component supports several advanced techniques for extracting deeper insights from temporal performance data:

### Comparative Analysis

To perform comparative analysis across different dimensions:

```python
# Compare performance across hardware platforms
viz.create_animated_time_series_visualization(
    metric="throughput_items_per_second",
    dimensions=["hardware_type"],
    comparative_dimension="model_family",
    time_range=90,
    normalized=True,  # Normalize values for fair comparison
    title="Relative Performance by Hardware Platform"
)
```

### Anomaly Detection

For automatic detection and highlighting of performance anomalies:

```python
# Enable anomaly detection
viz.create_animated_time_series_visualization(
    metric="average_latency_ms",
    dimensions=["model_name", "hardware_type"],
    time_range=90,
    detect_anomalies=True,
    anomaly_threshold=2.5,  # Z-score threshold for anomaly detection
    title="Latency Anomalies Over Time"
)
```

### Enhanced Regression Visualization

For statistical analysis and visualization of performance regressions with advanced features like confidence intervals, trend lines, and interactive controls:

```python
from duckdb_api.distributed_testing.dashboard.regression_visualization import RegressionVisualization
from duckdb_api.distributed_testing.dashboard.regression_detection import RegressionDetector

# Initialize the components
detector = RegressionDetector()
visualizer = RegressionVisualization(output_dir="./visualizations/regression")

# Detect regressions in time series data
time_series_data = {
    "timestamps": [...],  # List of timestamps
    "values": [...]       # Corresponding metric values
}
regressions = detector.detect_regressions(time_series_data, "latency_ms")

# Create enhanced visualization with statistical features
figure = visualizer.create_interactive_regression_figure(
    time_series_data=time_series_data,
    regressions=regressions,
    metric="latency_ms",
    title="Latency Regression Analysis",
    include_confidence_intervals=True,
    include_trend_lines=True,
    include_annotations=True
)

# Export the visualization
visualizer.export_regression_visualization(
    figure_dict=figure,
    output_path="./latency_regression_analysis.html",
    format="html"
)

# Generate comprehensive report
visualizer.create_regression_summary_report(
    metrics_data={"latency_ms": time_series_data},
    regressions_by_metric={"latency_ms": regressions},
    output_path="./regression_report.html",
    include_plots=True
)
```

#### Dashboard Integration

When used with the EnhancedVisualizationDashboard, you can control visualization options through the UI:

```python
from duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard import EnhancedVisualizationDashboard

# Create and configure the dashboard
dashboard = EnhancedVisualizationDashboard(
    db_path="benchmark_db.duckdb",
    output_dir="./visualizations/dashboard",
    theme="dark",
    enable_regression_detection=True,
    enhanced_visualization=True
)

# Start the dashboard (provides UI controls for all regression visualization options)
await dashboard.start()
```

Key features available in the UI:
- Toggle confidence intervals to visualize statistical uncertainty
- Toggle trend lines to identify slope changes before/after regressions
- Toggle annotations for detailed statistical information at change points
- Select export formats (HTML, PNG, SVG, PDF, JSON)
- Generate comprehensive reports with statistical insights
- Synchronized theme between dashboard and visualizations

### Event Correlation

To correlate performance changes with system events:

```python
# Add event markers to timeline
viz.create_animated_time_series_visualization(
    metric="throughput_items_per_second",
    dimensions=["model_family"],
    time_range=90,
    events=[
        {"date": "2025-05-15", "label": "Framework v2.1 Release", "color": "green"},
        {"date": "2025-06-01", "label": "Hardware Upgrade", "color": "blue"},
        {"date": "2025-06-10", "label": "Config Change", "color": "orange"}
    ],
    title="Performance Impact of System Changes"
)
```

## Customizable Dashboard System

The new Customizable Dashboard System provides a flexible way to combine multiple visualizations in a single interface, allowing for comprehensive analysis of performance data across different metrics and dimensions. Dashboards can be easily created, saved, customized, and shared, making it simple to build standardized views for different analysis needs.

### Creating Dashboards

```python
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
from duckdb_api.visualization.advanced_visualization import AdvancedVisualizationSystem

# Initialize database API and visualization system
db_api = BenchmarkDBAPI(db_path="./benchmark_db.duckdb")
viz = AdvancedVisualizationSystem(db_api=db_api, output_dir="./visualizations")

# Create a dashboard from a template
dashboard_path = viz.create_dashboard(
    dashboard_name="hardware_overview",
    template="hardware_comparison",
    title="Hardware Performance Overview",
    description="Comparison of performance metrics across different hardware platforms"
)

# Create a custom dashboard with specific components
dashboard_path = viz.create_dashboard(
    dashboard_name="custom_analysis",
    title="Custom Performance Analysis",
    description="Analysis of performance trends and comparisons",
    components=[
        {
            "type": "3d",
            "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
            "dimensions": ["model_family", "hardware_type"],
            "title": "3D Performance Visualization",
            "width": 1,
            "height": 1
        },
        {
            "type": "time-series",
            "metric": "throughput_items_per_second",
            "dimensions": ["model_family", "hardware_type"],
            "include_trend": True,
            "window_size": 3,
            "title": "Performance Trends Over Time",
            "width": 2,
            "height": 1
        }
    ],
    columns=2,
    row_height=500
)
```

### Managing Dashboards

```python
# List all saved dashboards
dashboards = viz.list_dashboards()

# Get a specific dashboard configuration
dashboard_config = viz.get_dashboard("hardware_overview")

# Update an existing dashboard
updated_path = viz.update_dashboard(
    dashboard_name="hardware_overview",
    title="Updated Hardware Overview",
    columns=3
)

# Add a component to a dashboard
viz.add_component_to_dashboard(
    dashboard_name="hardware_overview",
    component_type="power",
    component_config={
        "title": "Power Efficiency Analysis",
        "hardware_types": ["nvidia_a100", "amd_mi250", "intel_arc"]
    },
    width=2,
    height=1
)

# Remove a component from a dashboard
viz.remove_component_from_dashboard(
    dashboard_name="hardware_overview",
    component_index=1
)

# Export a dashboard to different formats
viz.export_dashboard(
    dashboard_name="hardware_overview",
    format="pdf",
    output_path="./reports/hardware_overview.pdf"
)
```

### Dashboard Templates

The system includes several predefined templates for common dashboard layouts:

1. **Overview Template**: General overview of performance metrics across models and hardware
   - 3D visualization showing throughput, latency, and memory usage
   - Hardware comparison heatmap for throughput
   - Time-series visualization showing performance trends
   - Ideal for getting a high-level overview of system performance

2. **Hardware Comparison Template**: Detailed comparison of hardware platforms
   - Hardware comparison heatmap spanning the full width
   - Power efficiency visualization
   - Time-series visualization filtered by hardware type
   - Optimized for comparing performance across different hardware platforms

3. **Model Analysis Template**: Detailed analysis of model performance
   - 3D visualization of performance metrics
   - Time-series visualization filtered by model family
   - Hardware comparison heatmap for latency
   - Best for deep-diving into specific model family performance characteristics

4. **Empty Template**: A blank template for fully custom dashboards
   - Starts with zero components
   - Allows complete customization of layout and components
   - Perfect for specialized dashboard needs

Templates can be used as starting points and then customized further, or you can build custom dashboards from scratch with specific component configurations.

### Dashboard Components

Each dashboard can include multiple visualization components of different types:

1. **3D Performance Visualization**:
   - Shows the relationship between three performance metrics in 3D space
   - Interactive rotation and zoom for exploring data relationships
   - Color-coded by dimension (e.g., model family, hardware type)
   - Configuration options include metrics, dimensions, and title

2. **Hardware Comparison Heatmap**:
   - Shows performance metrics across hardware platforms and model families
   - Color intensity indicates metric values for easy comparison
   - Special markers for simulated vs. real results
   - Configuration options include metric, model families, hardware types, and title

3. **Power Efficiency Visualization**:
   - Plots throughput vs. energy consumption with efficiency contours
   - Point size represents latency for multi-dimensional analysis
   - Interactive tooltips with detailed metrics
   - Configuration options include hardware types, model families, batch sizes, and title

4. **Animated Time-Series Visualization**:
   - Shows performance metrics over time with animation controls
   - Includes trend lines for identifying patterns
   - Interactive time slider for exploring specific time periods
   - Configuration options include metric, dimensions, time range, trend settings, and title

### Component Layout and Sizing

Dashboard components can be sized and arranged with a flexible grid system:

- **Width**: Components can span multiple columns (specified by the "width" parameter)
- **Height**: Components can have custom heights (specified by the "height" parameter)
- **Grid Layout**: Components are arranged in a responsive grid based on their width and height
- **Columns**: The dashboard grid can have any number of columns (default: 2)
- **Row Height**: The height of each grid row can be customized (default: 500px)

This flexible layout system allows for a wide variety of dashboard designs, from simple single-column layouts to complex multi-column arrangements.

### Command-Line Usage

#### Using the test script

```bash
# Create a dashboard from a template
python test_advanced_visualization.py --viz-type dashboard --dashboard-template overview

# Create a custom dashboard with all visualization types
python test_advanced_visualization.py --viz-type dashboard

# Create a dashboard with a specific database and output directory
python test_advanced_visualization.py --viz-type dashboard --db-path ./benchmark_db.duckdb --output-dir ./dashboards
```

#### Using the Dedicated Dashboard Script

For more advanced dashboard management, use the dedicated `run_customizable_dashboard.py` script:

```bash
# Create a dashboard from a template
./run_customizable_dashboard.py create --name hardware_overview --template hardware_comparison

# Create a custom dashboard
./run_customizable_dashboard.py create --name custom_dashboard --columns 3 --theme dark

# List all saved dashboards
./run_customizable_dashboard.py list

# Export a dashboard to PDF
./run_customizable_dashboard.py export --name hardware_overview --format pdf

# Delete a dashboard
./run_customizable_dashboard.py delete --name old_dashboard

# Show help
./run_customizable_dashboard.py --help
```

This script provides a more comprehensive interface for managing dashboards, including listing, exporting, and deleting dashboards.

### Dashboard Management Features

The dashboard system provides a complete set of management features:

1. **Dashboard Creation**:
   - Create from templates or custom component configurations
   - Set custom titles, descriptions, and layouts
   - Choose light or dark theme

2. **Dashboard Listing**:
   - View all saved dashboards with metadata
   - See creation and modification dates
   - View number of components in each dashboard

3. **Dashboard Updating**:
   - Change dashboard title, description, or layout
   - Add or remove visualization components
   - Modify component configurations

4. **Dashboard Export**:
   - Export to HTML for interactive web viewing
   - Export to PDF for reports and documentation
   - Export to PNG for presentations and screenshots

5. **Dashboard Component Management**:
   - Add individual components with specific configurations
   - Remove components that are no longer needed
   - Reposition components within the grid layout

6. **Dashboard Persistence**:
   - Dashboards are automatically saved to disk
   - Configurations can be loaded in future sessions
   - Dashboard configurations are stored in JSON format for easy sharing

7. **Theme Customization**:
   - Light mode for standard viewing and printing
   - Dark mode for reduced eye strain in low-light environments
   - Custom color palettes for different visualization types

### Best Practices for Dashboard Design

When creating dashboards, follow these best practices for optimal results:

1. **Purpose-Driven Design**: Design each dashboard with a specific analytical purpose in mind
   - Create focused dashboards that answer specific questions
   - Avoid including too many unrelated visualizations

2. **Component Placement**: Organize components logically
   - Place related visualizations near each other
   - Consider information flow from top to bottom and left to right

3. **Size Appropriately**: Size components based on importance
   - Use larger sizes for primary visualizations
   - Use smaller sizes for supporting visualizations

4. **Consistent Filtering**: Use consistent dimensions across components
   - Filter by the same model families or hardware types when possible
   - Create visual connections between different metrics

5. **Template Usage**: Use templates as starting points
   - Start with a template that matches your analysis goal
   - Customize components as needed rather than starting from scratch

6. **Export for Sharing**: Use appropriate export formats
   - Use HTML for interactive sharing with technical users
   - Use PDF for formal reports and documentation
   - Use PNG for presentations and quick sharing

## Example Dashboard Use Cases

### 1. Hardware Platform Evaluation

Create a dashboard for evaluating and comparing hardware platforms:

```python
viz.create_dashboard(
    dashboard_name="hardware_evaluation",
    title="Hardware Platform Evaluation Dashboard",
    description="Comprehensive comparison of hardware platforms for model inference",
    components=[
        {
            "type": "heatmap",
            "metric": "throughput",
            "title": "Throughput Comparison by Hardware Platform",
            "width": 2,
            "height": 1
        },
        {
            "type": "power",
            "title": "Power Efficiency Analysis",
            "hardware_types": ["nvidia_a100", "amd_mi250", "intel_arc", "apple_m2"],
            "width": 1,
            "height": 1
        },
        {
            "type": "time-series",
            "metric": "throughput_items_per_second",
            "dimensions": ["hardware_type"],
            "title": "Throughput Trends by Hardware Platform",
            "width": 1,
            "height": 1
        }
    ]
)
```

This dashboard provides a comprehensive view of hardware platform performance characteristics, with focus on throughput, power efficiency, and performance trends over time.

### 2. Model Family Performance Analysis

Create a dashboard for analyzing performance across model families:

```python
viz.create_dashboard(
    dashboard_name="model_performance",
    title="Model Family Performance Analysis",
    description="Analysis of performance metrics across different model families",
    components=[
        {
            "type": "3d",
            "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
            "dimensions": ["model_family"],
            "title": "3D Performance Visualization by Model Family",
            "width": 2,
            "height": 1
        },
        {
            "type": "time-series",
            "metric": "throughput_items_per_second",
            "dimensions": ["model_family"],
            "include_trend": True,
            "title": "Throughput Trends by Model Family",
            "width": 2,
            "height": 1
        }
    ]
)
```

This dashboard focuses on comparing different model families (transformers, vision, audio) across multiple performance dimensions with both static and time-series views.

### 3. Optimization Impact Dashboard

Create a dashboard for tracking the impact of system optimizations:

```python
viz.create_dashboard(
    dashboard_name="optimization_impact",
    title="Optimization Impact Dashboard",
    description="Track performance changes resulting from system optimizations",
    components=[
        {
            "type": "time-series",
            "metric": "throughput_items_per_second",
            "dimensions": ["model_family", "hardware_type"],
            "include_trend": True,
            "title": "Throughput Changes Over Time",
            "width": 2,
            "height": 1,
            "events": [
                {"date": "2025-03-15", "label": "Optimization Phase 1", "color": "green"},
                {"date": "2025-04-10", "label": "Optimization Phase 2", "color": "blue"},
                {"date": "2025-05-05", "label": "Optimization Phase 3", "color": "purple"}
            ]
        },
        {
            "type": "heatmap",
            "metric": "throughput",
            "title": "Current Throughput by Model and Hardware",
            "width": 2,
            "height": 1
        }
    ]
)
```

This dashboard is designed to track and visualize the impact of system optimizations over time, with event markers for key optimization phases and current performance status.

## Export Capabilities

The Advanced Visualization System now includes comprehensive export capabilities for all visualization types, enabling users to export visualizations in various formats for reports, presentations, and sharing.

### Supported Export Formats

| Format | Description | Support |
|--------|-------------|---------|
| HTML | Interactive HTML with embedded JavaScript | All visualization types |
| PNG | Static image in PNG format | All visualization types |
| PDF | Portable Document Format | All visualization types |
| SVG | Scalable Vector Graphics | All visualization types |
| JSON | JSON representation of the visualization | All visualization types |
| CSV | CSV export of the underlying data | All visualization types |
| MP4 | Video format for animations | Time-series animations |
| GIF | Animated GIF format | Time-series animations |

### Using the Export Functionality

Export capabilities are integrated directly into the Advanced Visualization System API, making it easy to export any visualization:

```python
# Export a 3D visualization
exports = viz.export_3d_visualization(
    visualization_data=result,
    formats=['html', 'png', 'pdf'],
    visualization_id="hardware_performance_3d"
)

# Export a heatmap visualization
exports = viz.export_heatmap_visualization(
    visualization_data=result,
    formats=['html', 'png', 'pdf'],
    visualization_id="hardware_comparison"
)

# Export a power efficiency visualization
exports = viz.export_power_visualization(
    visualization_data=result,
    formats=['html', 'png', 'pdf'],
    visualization_id="power_efficiency"
)

# Export a time-series visualization with animations
exports = viz.export_time_series_visualization(
    visualization_data=result,
    formats=['html', 'png', 'pdf', 'mp4', 'gif'],
    visualization_id="performance_trends"
)

# Export a dashboard
dashboard_path = viz.export_dashboard(
    dashboard_name="performance_dashboard",
    format="pdf"
)
```

### Optimized Animation Export

For time-series animations, you can use the specialized animation export function with optimized settings:

```python
# Export an optimized animation in MP4 format
animation_path = viz.export_animated_time_series(
    visualization_data=result,
    format="mp4",
    visualization_id="performance_animation",
    settings={
        "width": 1920,  # HD resolution
        "height": 1080,
        "fps": 30,      # High frame rate
        "duration": 15000  # 15 seconds
    }
)
```

### Export Manager API

For more advanced export management, you can use the Export Manager API:

```python
# Export all visualizations in a batch
exports = viz.export_all_visualizations(
    visualizations={
        "3d_visualization": result_3d,
        "heatmap_visualization": result_heatmap,
        "power_visualization": result_power,
        "time_series_visualization": result_time_series
    },
    formats={
        '3d': ['html', 'png', 'pdf'],
        'heatmap': ['html', 'png', 'pdf'],
        'power': ['html', 'png', 'pdf'],
        'time-series': ['html', 'png', 'pdf', 'mp4', 'gif']
    },
    create_index=True,
    title="All Visualization Types"
)

# Generate a comprehensive export report
report_path = viz.generate_export_report(
    title="Visualization Export Report",
    description="Comprehensive report of all exported visualizations"
)

# Configure default export settings
viz.configure_export_settings({
    "width": 1200,
    "height": 800,
    "scale": 2,
    "include_plotlyjs": True,
    "include_mathjax": False,
    "full_html": True,
    "fps": 30,
    "duration": 10000
})
```

### Command-Line Export Tool

The Advanced Visualization System includes a command-line tool for exporting visualizations:

```bash
# Export a 3D visualization
python run_export_visualization.py export --viz-type 3d --formats html,png,pdf

# Export a heatmap visualization with specific metrics and dimensions
python run_export_visualization.py export --viz-type heatmap --metrics throughput --dimensions "transformer,vision" "nvidia_a100,amd_mi250,intel_arc"

# Export a power efficiency visualization
python run_export_visualization.py export --viz-type power --formats html,png,pdf

# Export a time-series visualization with animations
python run_export_visualization.py export --viz-type time-series --formats html,png,pdf,mp4,gif

# Export a dashboard
python run_export_visualization.py export --viz-type dashboard --name performance_dashboard --format pdf

# Export all visualization types
python run_export_visualization.py export-all --formats html,png,pdf,mp4,gif

# Export an optimized animation
python run_export_visualization.py export-animation --format mp4 --width 1920 --height 1080 --fps 30 --duration 15
```

### Setting Up Export Dependencies

To use all export capabilities, you need to install the required dependencies:

```bash
# Make the script executable
chmod +x setup_export_visualization.sh

# Run the setup script
./setup_export_visualization.sh
```

This script will install all required Python packages and system dependencies for exporting visualizations in all supported formats.

### Export Best Practices

1. **Choose the Right Format**: Use HTML for interactive exploration, PNG/PDF for reports, and MP4/GIF for animations.
2. **Resolution and Quality**: For high-quality exports, increase the width, height, and scale settings.
3. **Animation Settings**: For smooth animations, use higher fps (30) and appropriate frame duration (50ms).
4. **File Size Management**: For web sharing, MP4 is more efficient than GIF for animations.
5. **Template Usage**: Use export templates for standardized reporting formats across your organization.
6. **Batch Processing**: Use the `export_all_visualizations` method for efficiently exporting multiple visualizations.
7. **Session Management**: Save export metadata for tracking and reproducing exports.

## Customizable Dashboard System

The Customizable Dashboard System provides a flexible way to combine multiple visualizations in a single interface, allowing for comprehensive analysis of performance data across different metrics and dimensions. Dashboards can be easily created, saved, customized, and shared, making it simple to build standardized views for different analysis needs.

### Creating Dashboards

```python
from duckdb_api.visualization.advanced_visualization import CustomizableDashboard

# Initialize dashboard system
dashboard = CustomizableDashboard(theme="light", output_dir="./dashboards")

# Create a dashboard from a template
dashboard_path = dashboard.create_dashboard(
    dashboard_name="hardware_overview",
    template="hardware_comparison",
    title="Hardware Performance Overview",
    description="Comparison of performance metrics across different hardware platforms"
)

# Create a custom dashboard with specific components
dashboard_path = dashboard.create_dashboard(
    dashboard_name="custom_analysis",
    title="Custom Performance Analysis",
    description="Analysis of performance trends and comparisons",
    components=[
        {
            "type": "3d",
            "config": {
                "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                "dimensions": ["model_family", "hardware_type"],
                "title": "3D Performance Visualization"
            },
            "width": 1,
            "height": 1
        },
        {
            "type": "animated-time-series",
            "config": {
                "metric": "throughput_items_per_second",
                "dimensions": ["model_family", "hardware_type"],
                "time_range": 90,
                "title": "Performance Trends Over Time"
            },
            "width": 2,
            "height": 1
        }
    ],
    columns=2,
    row_height=500
)
```

### Managing Dashboards

```python
# List all saved dashboards
dashboards = dashboard.list_dashboards()

# Get a specific dashboard configuration
dashboard_config = dashboard.get_dashboard("hardware_overview")

# Update an existing dashboard
updated_path = dashboard.update_dashboard(
    dashboard_name="hardware_overview",
    title="Updated Hardware Overview",
    columns=3
)

# Add a component to a dashboard
dashboard.add_component_to_dashboard(
    dashboard_name="hardware_overview",
    component_type="heatmap",
    component_config={
        "metric": "memory_peak_mb",
        "title": "Memory Usage Comparison"
    },
    width=2,
    height=1
)

# Remove a component from a dashboard
dashboard.remove_component_from_dashboard(
    dashboard_name="hardware_overview",
    component_index=1
)

# Export a dashboard to different formats
dashboard.export_dashboard(
    dashboard_name="hardware_overview",
    format="pdf",
    output_path="./reports/hardware_overview.pdf"
)

# Delete a dashboard
dashboard.delete_dashboard("hardware_overview")
```

### Dashboard Templates

The system includes several predefined templates for common dashboard layouts:

1. **Overview Template**: General overview of performance metrics across models and hardware
   - 3D visualization showing throughput, latency, and memory usage
   - Hardware comparison heatmap for throughput
   - Animated time-series visualization showing performance trends
   - Ideal for getting a high-level overview of system performance

2. **Hardware Comparison Template**: Detailed comparison of hardware platforms
   - Hardware comparison heatmap for throughput spanning the full width
   - Hardware comparison heatmap for latency
   - Animated time-series visualization filtered by hardware type
   - Optimized for comparing performance across different hardware platforms

3. **Model Analysis Template**: Detailed analysis of model performance
   - 3D visualization of performance metrics
   - Animated time-series visualization filtered by model family
   - Hardware comparison heatmap for latency
   - Best for deep-diving into specific model family performance characteristics

4. **Empty Template**: A blank template for fully custom dashboards
   - Starts with zero components
   - Allows complete customization of layout and components
   - Perfect for specialized dashboard needs

Templates can be used as starting points and then customized further, or you can build custom dashboards from scratch with specific component configurations.

### Dashboard Components

Each dashboard can include multiple visualization components of different types:

1. **3D Performance Visualization**:
   - Shows the relationship between three performance metrics in 3D space
   - Interactive rotation and zoom for exploring data relationships
   - Color-coded by dimension (e.g., model family, hardware type)
   - Configuration options include metrics, dimensions, and title

2. **Hardware Comparison Heatmap**:
   - Shows performance metrics across hardware platforms and model families
   - Color intensity indicates metric values for easy comparison
   - Special markers for simulated vs. real results
   - Configuration options include metric, model families, hardware types, and title

3. **Time-Series Visualization**:
   - Shows performance metrics over time
   - Includes trend lines for identifying patterns
   - Configuration options include metric, dimensions, time range, and title

4. **Animated Time-Series Visualization**:
   - Shows performance metrics over time with animation controls
   - Includes trend lines for identifying patterns
   - Interactive time slider for exploring specific time periods
   - Configuration options include metric, dimensions, time range, trend settings, and title

### Component Layout and Sizing

Dashboard components can be sized and arranged with a flexible grid system:

- **Width**: Components can span multiple columns (specified by the "width" parameter)
- **Height**: Components can have custom heights (specified by the "height" parameter)
- **Grid Layout**: Components are arranged in a responsive grid based on their width and height
- **Columns**: The dashboard grid can have any number of columns (default: 2)
- **Row Height**: The height of each grid row can be customized (default: 500px)

This flexible layout system allows for a wide variety of dashboard designs, from simple single-column layouts to complex multi-column arrangements.

### Command-Line Usage

The `run_customizable_dashboard.py` script provides a convenient command-line interface for working with the dashboard system:

```bash
# Create a dashboard from a template
python run_customizable_dashboard.py create --name hardware_overview --template hardware_comparison --open

# Create a custom dashboard with all visualization types
python run_customizable_dashboard.py create --name custom_dashboard --include-all --open

# Create a dashboard with specific visualization types
python run_customizable_dashboard.py create --name specific_dashboard --include-3d --include-animated --open

# List all saved dashboards
python run_customizable_dashboard.py list

# List available templates and components
python run_customizable_dashboard.py templates

# Add a component to a dashboard
python run_customizable_dashboard.py add --name hardware_overview --type heatmap --config-file heatmap_config.json --open

# Update a dashboard
python run_customizable_dashboard.py update --name hardware_overview --title "Updated Hardware Overview" --columns 3 --open

# Export a dashboard to PDF
python run_customizable_dashboard.py export --name hardware_overview --format pdf

# Delete a dashboard
python run_customizable_dashboard.py delete --name old_dashboard --force
```

### Dashboard Management Features

The dashboard system provides a complete set of management features:

1. **Dashboard Creation**:
   - Create from templates or custom component configurations
   - Set custom titles, descriptions, and layouts
   - Choose light or dark theme

2. **Dashboard Listing**:
   - View all saved dashboards with metadata
   - See creation and modification dates
   - View number of components in each dashboard

3. **Dashboard Updating**:
   - Change dashboard title, description, or layout
   - Add or remove visualization components
   - Modify component configurations

4. **Dashboard Export**:
   - Export to HTML for interactive web viewing
   - Export to PDF for reports and documentation
   - Export to PNG for presentations and screenshots

5. **Dashboard Component Management**:
   - Add individual components with specific configurations
   - Remove components that are no longer needed
   - Reposition components within the grid layout

6. **Dashboard Persistence**:
   - Dashboards are automatically saved to disk
   - Configurations can be loaded in future sessions
   - Dashboard configurations are stored in JSON format for easy sharing

7. **Theme Customization**:
   - Light mode for standard viewing and printing
   - Dark mode for reduced eye strain in low-light environments
   - Custom color palettes for different visualization types

### Testing the Dashboard System

The dashboard system includes a comprehensive test script that demonstrates all features:

```bash
# Test dashboard creation with a specific template
python test_customizable_dashboard.py --template overview

# Test dashboard creation with all templates
python test_customizable_dashboard.py --template all

# Test dashboard management features
python test_customizable_dashboard.py --management-test

# Test with a specific output directory
python test_customizable_dashboard.py --output-dir ./my_dashboards

# Test without opening dashboards in browser
python test_customizable_dashboard.py --template all --no-open
```

### Best Practices for Dashboard Design

When creating dashboards, follow these best practices for optimal results:

1. **Purpose-Driven Design**: Design each dashboard with a specific analytical purpose in mind
   - Create focused dashboards that answer specific questions
   - Avoid including too many unrelated visualizations

2. **Component Placement**: Organize components logically
   - Place related visualizations near each other
   - Consider information flow from top to bottom and left to right

3. **Size Appropriately**: Size components based on importance
   - Use larger sizes for primary visualizations
   - Use smaller sizes for supporting visualizations

4. **Consistent Filtering**: Use consistent dimensions across components
   - Filter by the same model families or hardware types when possible
   - Create visual connections between different metrics

5. **Template Usage**: Use templates as starting points
   - Start with a template that matches your analysis goal
   - Customize components as needed rather than starting from scratch

6. **Export for Sharing**: Use appropriate export formats
   - Use HTML for interactive sharing with technical users
   - Use PDF for formal reports and documentation
   - Use PNG for presentations and quick sharing

### Example Use Cases

#### Hardware Evaluation Dashboard

Create a dashboard for evaluating and comparing hardware platforms:

```python
dashboard = CustomizableDashboard(output_dir="./dashboards")
dashboard_path = dashboard.create_dashboard(
    dashboard_name="hardware_evaluation",
    template="hardware_comparison",
    title="Hardware Platform Evaluation",
    description="Comprehensive evaluation of hardware platforms for model inference"
)
```

This dashboard focuses on comparing hardware platforms across multiple performance metrics, with visualizations for throughput, latency, and performance trends.

#### Model Family Analysis Dashboard

Create a dashboard for analyzing performance across model families:

```python
dashboard = CustomizableDashboard(output_dir="./dashboards")
dashboard_path = dashboard.create_dashboard(
    dashboard_name="model_analysis",
    template="model_analysis",
    title="Model Family Performance Analysis",
    description="Analysis of performance metrics across different model families"
)
```

This dashboard focuses on comparing different model families (transformers, vision, audio) across multiple performance dimensions, with both static and time-series views.

#### Optimization Impact Dashboard

Create a dashboard for tracking the impact of system optimizations:

```python
dashboard = CustomizableDashboard(output_dir="./dashboards")

# Create a dashboard with an animated time-series visualization showing optimization impacts
dashboard_path = dashboard.create_dashboard(
    dashboard_name="optimization_impact",
    title="Optimization Impact Dashboard",
    description="Track performance changes resulting from system optimizations",
    components=[
        {
            "type": "animated-time-series",
            "config": {
                "metric": "throughput_items_per_second",
                "dimensions": ["model_family", "hardware_type"],
                "time_range": 90,
                "events": [
                    {"date": "2025-03-15", "label": "Optimization Phase 1", "color": "green"},
                    {"date": "2025-04-10", "label": "Optimization Phase 2", "color": "blue"},
                    {"date": "2025-05-05", "label": "Optimization Phase 3", "color": "purple"}
                ],
                "title": "Throughput Changes Over Time"
            },
            "width": 2,
            "height": 1
        },
        {
            "type": "heatmap",
            "config": {
                "metric": "throughput_items_per_second",
                "title": "Current Throughput by Model and Hardware"
            },
            "width": 2,
            "height": 1
        }
    ]
)
```

This dashboard is designed to track and visualize the impact of system optimizations over time, with event markers for key optimization phases and current performance status.

#### Memory Usage Monitoring Dashboard

Create a dashboard focused on memory usage across different models and hardware platforms:

```python
dashboard = CustomizableDashboard(output_dir="./dashboards")
dashboard_path = dashboard.create_dashboard(
    dashboard_name="memory_monitoring",
    title="Memory Usage Monitoring Dashboard",
    description="Comprehensive analysis of memory usage across models and hardware",
    components=[
        {
            "type": "3d",
            "config": {
                "metrics": ["memory_peak_mb", "batch_size", "model_size_mb"],
                "dimensions": ["model_family", "hardware_type"],
                "title": "Memory Usage in 3D Space"
            },
            "width": 2,
            "height": 1
        },
        {
            "type": "heatmap",
            "config": {
                "metric": "memory_peak_mb",
                "title": "Peak Memory Usage Comparison"
            },
            "width": 2,
            "height": 1
        },
        {
            "type": "animated-time-series",
            "config": {
                "metric": "memory_peak_mb",
                "dimensions": ["model_family", "hardware_type"],
                "time_range": 90,
                "title": "Memory Usage Trends Over Time"
            },
            "width": 2,
            "height": 1
        }
    ],
    columns=2,
    row_height=500
)
```

This dashboard provides a comprehensive view of memory usage across different models and hardware platforms, with 3D visualization, comparative heatmap, and time-series trend analysis.

#### Multi-Model Inference Dashboard

Create a dashboard for analyzing performance of multi-model inference scenarios with tensor sharing:

```python
dashboard = CustomizableDashboard(output_dir="./dashboards")

# Create a dashboard for multi-model inference with tensor sharing
dashboard_path = dashboard.create_dashboard(
    dashboard_name="multi_model_inference",
    title="Multi-Model Inference Dashboard",
    description="Performance analysis of multi-model inference with tensor sharing",
    components=[
        {
            "type": "heatmap",
            "config": {
                "metric": "throughput_items_per_second",
                "model_families": ["transformers+vision", "audio+transformers", "vision+audio"],
                "title": "Multi-Model Throughput Comparison"
            },
            "width": 2,
            "height": 1
        },
        {
            "type": "3d",
            "config": {
                "metrics": ["throughput_items_per_second", "memory_reduction_percent", "latency_improvement_percent"],
                "dimensions": ["model_combination", "hardware_type"],
                "title": "Tensor Sharing Benefits"
            },
            "width": 2,
            "height": 1
        },
        {
            "type": "animated-time-series",
            "config": {
                "metric": "memory_reduction_percent",
                "dimensions": ["model_combination", "tensor_sharing_type"],
                "time_range": 90,
                "title": "Memory Reduction Trends with Tensor Sharing"
            },
            "width": 2,
            "height": 1
        }
    ],
    columns=2,
    row_height=500
)
```

This dashboard focuses on analyzing the performance benefits of multi-model inference with tensor sharing, showing throughput comparison, 3D visualization of benefits, and time-series trends of memory reduction.

### Advanced Dashboard Customization

Beyond the basic dashboard creation and management features, the Customizable Dashboard System offers several advanced customization options.

#### Custom Layout Grid

You can customize the layout grid to create more complex dashboard layouts:

```python
dashboard = CustomizableDashboard(output_dir="./dashboards")

# Create a dashboard with a custom layout grid
dashboard_path = dashboard.create_dashboard(
    dashboard_name="custom_layout",
    title="Custom Layout Dashboard",
    description="Dashboard with custom layout grid",
    components=[
        # Large 3D visualization spanning the first two columns and rows
        {
            "type": "3d",
            "config": {
                "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                "dimensions": ["model_family", "hardware_type"],
                "title": "3D Performance Visualization"
            },
            "width": 2,
            "height": 2
        },
        # Heatmap in the third column, first row
        {
            "type": "heatmap",
            "config": {
                "metric": "throughput_items_per_second",
                "title": "Throughput Heatmap"
            },
            "width": 1,
            "height": 1
        },
        # Heatmap in the third column, second row
        {
            "type": "heatmap",
            "config": {
                "metric": "average_latency_ms",
                "title": "Latency Heatmap"
            },
            "width": 1,
            "height": 1
        },
        # Time-series visualization spanning the full width of the third row
        {
            "type": "animated-time-series",
            "config": {
                "metric": "throughput_items_per_second",
                "dimensions": ["model_family", "hardware_type"],
                "time_range": 90,
                "title": "Performance Trends Over Time"
            },
            "width": 3,
            "height": 1
        }
    ],
    columns=3,  # 3-column grid
    row_height=400  # Smaller row height
)
```

This creates a dashboard with a 3x3 grid layout, with components of varying sizes arranged in a custom pattern.

#### Dynamic Component Configuration

You can create dashboards with dynamically configured components based on data analysis:

```python
import pandas as pd
from duckdb_api.visualization.advanced_visualization import CustomizableDashboard

# Load and analyze performance data
data = pd.read_csv("performance_data.csv")

# Find top performing hardware platforms
top_hardware = data.groupby("hardware_type")["throughput_items_per_second"].mean().nlargest(3).index.tolist()

# Find most memory-intensive model families
memory_intensive_models = data.groupby("model_family")["memory_peak_mb"].mean().nlargest(3).index.tolist()

# Create dashboard with dynamically configured components
dashboard = CustomizableDashboard(output_dir="./dashboards")
dashboard_path = dashboard.create_dashboard(
    dashboard_name="dynamic_analysis",
    title="Dynamic Performance Analysis",
    description="Automatically configured dashboard based on data analysis",
    components=[
        {
            "type": "heatmap",
            "config": {
                "metric": "throughput_items_per_second",
                "hardware_types": top_hardware,  # Dynamically selected hardware
                "title": "Top Hardware Throughput Comparison"
            },
            "width": 2,
            "height": 1
        },
        {
            "type": "heatmap",
            "config": {
                "metric": "memory_peak_mb",
                "model_families": memory_intensive_models,  # Dynamically selected models
                "title": "Memory-Intensive Models Comparison"
            },
            "width": 2,
            "height": 1
        }
    ],
    columns=2,
    row_height=500
)
```

This creates a dashboard with components that are dynamically configured based on data analysis, focusing on top-performing hardware platforms and memory-intensive model families.

#### Interactive Dashboard Generation Script

For ease of use, you can create a script that generates dashboards with interactive prompts:

```python
# interactive_dashboard_generator.py
import argparse
import os
from duckdb_api.visualization.advanced_visualization import CustomizableDashboard

def main():
    parser = argparse.ArgumentParser(description="Interactive Dashboard Generator")
    parser.add_argument("--output-dir", default="./dashboards", help="Output directory")
    parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Dashboard theme")
    args = parser.parse_args()
    
    # Create dashboard instance
    dashboard = CustomizableDashboard(theme=args.theme, output_dir=args.output_dir)
    
    # Interactive prompts
    print("Interactive Dashboard Generator")
    print("-" * 50)
    
    # Get dashboard name and title
    dashboard_name = input("Enter dashboard name: ")
    title = input("Enter dashboard title: ")
    description = input("Enter dashboard description: ")
    
    # Choose template or custom
    use_template = input("Use a template? (y/n): ").lower() == 'y'
    
    if use_template:
        # Show available templates
        templates = dashboard.list_available_templates()
        print("\nAvailable templates:")
        for i, (name, details) in enumerate(templates.items()):
            print(f"{i+1}. {name}: {details.get('title', '')}")
        
        # Select template
        template_idx = int(input("Select template number: ")) - 1
        template_name = list(templates.keys())[template_idx]
        
        # Create dashboard from template
        dashboard_path = dashboard.create_dashboard(
            dashboard_name=dashboard_name,
            template=template_name,
            title=title,
            description=description
        )
    else:
        # Custom dashboard
        components = []
        
        # Add components
        while True:
            add_component = input("Add a component? (y/n): ").lower() == 'y'
            if not add_component:
                break
            
            # Show available component types
            comp_types = dashboard.list_available_components()
            print("\nAvailable component types:")
            for i, (name, desc) in enumerate(comp_types.items()):
                print(f"{i+1}. {name}: {desc}")
            
            # Select component type
            comp_idx = int(input("Select component type number: ")) - 1
            comp_type = list(comp_types.keys())[comp_idx]
            
            # Configure component
            config = {}
            config["title"] = input("Component title: ")
            
            # Basic configuration based on component type
            if comp_type == "3d":
                config["metrics"] = input("Metrics (comma-separated): ").split(",")
                config["dimensions"] = input("Dimensions (comma-separated): ").split(",")
            elif comp_type in ["heatmap", "time-series", "animated-time-series"]:
                config["metric"] = input("Metric: ")
                dimensions = input("Dimensions (comma-separated): ")
                if dimensions:
                    config["dimensions"] = dimensions.split(",")
            
            # Component size
            width = int(input("Component width (columns): "))
            height = int(input("Component height (rows): "))
            
            # Add component to list
            components.append({
                "type": comp_type,
                "config": config,
                "width": width,
                "height": height
            })
        
        # Get layout settings
        columns = int(input("Number of columns in the grid: "))
        row_height = int(input("Row height in pixels: "))
        
        # Create custom dashboard
        dashboard_path = dashboard.create_dashboard(
            dashboard_name=dashboard_name,
            title=title,
            description=description,
            components=components,
            columns=columns,
            row_height=row_height
        )
    
    print(f"\nDashboard created successfully: {dashboard_path}")
    
    # Open in browser
    open_browser = input("Open dashboard in browser? (y/n): ").lower() == 'y'
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")

if __name__ == "__main__":
    main()
```

Save this script as `interactive_dashboard_generator.py` and run it to create dashboards interactively:

```bash
python interactive_dashboard_generator.py --theme dark
```

This script provides an interactive command-line interface for creating dashboards without having to write code, making it accessible to a wider range of users.

## Monitoring Dashboard Integration

The Advanced Visualization System now integrates seamlessly with the Distributed Testing Framework's Monitoring Dashboard, allowing you to embed custom visualization dashboards within the monitoring interface.

### Running the Monitoring Dashboard with Visualization Integration

```bash
# Start the monitoring dashboard with visualization integration enabled
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration

# Use a specific dashboard directory
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --dashboard-dir ./custom_dashboards

# Run with additional integrations
python -m duckdb_api.distributed_testing.run_monitoring_dashboard --enable-visualization-integration --enable-e2e-test-integration --enable-performance-analytics
```

### Integration API

The integration between the Advanced Visualization System and the Monitoring Dashboard is facilitated by the `VisualizationDashboardIntegration` class:

```python
from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_visualization_integration import VisualizationDashboardIntegration

# Create the integration component
viz_integration = VisualizationDashboardIntegration(
    dashboard_dir="./dashboards",
    integration_dir="./dashboards/monitor_integration"
)

# Create an embedded dashboard for the overview page
dashboard_details = viz_integration.create_embedded_dashboard(
    name="overview_dashboard",
    page="index",
    template="overview",
    title="System Overview Dashboard",
    description="Overview of system performance metrics",
    position="below"  # Can be "above", "below", or "tab"
)

# Generate a dashboard from performance data
dashboard_path = viz_integration.generate_dashboard_from_performance_data(
    performance_data=analytics_data,
    name="performance_dashboard",
    title="Performance Analytics Dashboard"
)

# Get HTML for embedding the dashboard in a web page
iframe_html = viz_integration.get_dashboard_iframe_html(
    name="overview_dashboard",
    width="100%",
    height="600px"
)

# Update an embedded dashboard
viz_integration.update_embedded_dashboard(
    name="overview_dashboard",
    title="Updated Overview Dashboard",
    description="Updated description",
    position="above",
    page="results"
)

# Remove an embedded dashboard
viz_integration.remove_embedded_dashboard("overview_dashboard")

# Get a list of available templates
templates = viz_integration.list_available_templates()

# Get a list of available components
components = viz_integration.list_available_components()

# Export a dashboard to a different format
viz_integration.export_embedded_dashboard(
    name="overview_dashboard",
    format="html"
)
```

### Dashboard Management UI

The Monitoring Dashboard includes a comprehensive Dashboard Management UI that allows you to:

1. **List Dashboards**: View all embedded dashboards with details
2. **Create Dashboards**: Create new dashboards from templates or with specific components
3. **Update Dashboards**: Update dashboard properties and position
4. **Remove Dashboards**: Remove dashboards from the monitoring interface
5. **View Templates**: Explore available dashboard templates
6. **View Components**: Explore available visualization components

To access the Dashboard Management UI, navigate to:
```
http://<monitoring-dashboard-host>:<port>/dashboards
```

### Automatic Dashboard Generation

The integration automatically generates dashboards from monitoring data when appropriate:

1. **Results Dashboards**: Generated from result aggregator data on the Results page
2. **Performance Analytics Dashboards**: Generated from performance analytics data
3. **E2E Test Results Dashboards**: Generated from end-to-end test results

These dashboards are automatically generated when no embedded dashboards exist for the specific page, providing immediate visualization of monitoring data without manual configuration.

### Integration Testing

The integration between the Advanced Visualization System and the Monitoring Dashboard includes comprehensive test coverage:

```bash
# Run all integration tests
python -m duckdb_api.distributed_testing.tests.run_visualization_dashboard_tests

# Run with verbose output
python -m duckdb_api.distributed_testing.tests.run_visualization_dashboard_tests --verbose

# Run a specific test pattern
python -m duckdb_api.distributed_testing.tests.run_visualization_dashboard_tests --test create_embedded_dashboard
```

The test suite includes:

1. Core integration tests for the VisualizationDashboardIntegration class
2. Web server integration tests for file serving and embedding
3. Monitoring dashboard tests for initialization and integration

For more details on testing, see [MONITORING_DASHBOARD_INTEGRATION_GUIDE.md](MONITORING_DASHBOARD_INTEGRATION_GUIDE.md#testing-the-integration).

## Conclusion

The Advanced Visualization System provides powerful tools for visualizing, analyzing, and exporting model performance data. By combining interactive and static visualizations with comprehensive reporting and export capabilities, it enables deep insights into performance patterns across different hardware platforms, batch sizes, and model architectures.

The system now includes seven key components:

1. **Interactive 3D Visualizations**: Explore multi-dimensional performance data with interactive 3D plots
2. **Dynamic Hardware Comparison Heatmaps**: Compare performance across hardware platforms and model families
3. **Power Efficiency Visualization Tools**: Analyze the relationship between performance and energy consumption
4. **Animated Time-Series Visualizations**: Track performance metrics over time with interactive animations
5. **Customizable Dashboard System**: Combine multiple visualizations in personalized layouts
6. **Comprehensive Export System**: Export visualizations in various formats for reporting and sharing
7. **Monitoring Dashboard Integration**: Embed visualizations within the Distributed Testing Framework's Monitoring Dashboard

The newly added Monitoring Dashboard integration enhances the Advanced Visualization System by allowing seamless integration with the Distributed Testing Framework, providing a unified interface for monitoring and analyzing performance data. This integration enables users to create, manage, and view customized visualization dashboards directly within the monitoring interface, streamlining the workflow for performance analysis and monitoring.

The completion of this system marks an important milestone in the IPFS Accelerate Framework's capabilities, providing a comprehensive set of tools for exploring, understanding, and communicating performance data across the entire project ecosystem. By enabling data-driven decision making through intuitive visualizations, the Advanced Visualization System will help drive continuous improvement in model performance across all supported hardware platforms.