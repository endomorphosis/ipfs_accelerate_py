# Enhanced Visualization System

This document describes the enhanced visualization capabilities added to the benchmark suite, particularly focusing on how these visualizations leverage the hardware-aware metrics.

## Overview

The visualization system has been enhanced to:

1. **Display detailed hardware-specific metrics**: Visualize hardware-specific performance data like latency percentiles, detailed memory usage, and FLOPs breakdowns
2. **Support multi-hardware comparison**: Compare performance across different hardware platforms
3. **Show batch size scaling**: Visualize how performance scales with increasing batch sizes
4. **Provide detailed breakdowns**: Show detailed performance breakdowns where available

## Key Enhancements

### Latency Visualization

The latency visualization has been enhanced to include percentile information (p90, p95, p99), providing a more complete picture of the latency distribution:

```python
# Generate latency comparison with percentiles
results.plot_latency_comparison(include_percentiles=True)
```

- **Bar Chart**: Mean latency across batch sizes
- **Percentile Lines**: p90, p95, and p99 latency percentiles
- **Hardware Comparison**: Side-by-side comparison of different hardware platforms

### Memory Usage Visualization

Memory visualization now provides a detailed breakdown of different memory metrics:

```python
# Generate detailed memory usage visualization
results.plot_memory_usage(detailed=True)
```

- **Total Memory Usage**: Overall memory usage across batch sizes
- **Peak Memory**: Peak GPU memory (for CUDA devices)
- **Allocated Memory**: Memory allocated by tensors
- **Reserved Memory**: Memory reserved by CUDA allocator
- **CPU Memory**: Host memory usage

### FLOPs Visualization

A new visualization for FLOPs (Floating Point Operations) has been added:

```python
# Generate FLOPs comparison with detailed breakdown
results.plot_flops_comparison(detailed=True)
```

- **Total FLOPs**: Bar chart showing FLOPs across batch sizes
- **FLOPs Breakdown**: Pie chart showing the breakdown of FLOPs by component (for transformer models: attention vs. linear layers, etc.)
- **Hardware Comparison**: FLOPs comparison across hardware platforms

## Integration with Hardware Abstraction

The visualization system leverages the hardware abstraction layer by:

1. **Hardware-specific metrics**: Extracting and visualizing hardware-specific metrics
2. **Device-aware aggregation**: Properly aggregating metrics for each hardware platform
3. **Intelligent fallbacks**: Showing alternative visualizations when certain metrics aren't available for a platform

## Usage Examples

### Basic Usage

```python
# Run benchmark
benchmark = ModelBenchmark(
    model_id="bert-base-uncased",
    hardware=["cpu", "cuda"],
    metrics=["latency", "throughput", "memory", "flops"]
)
results = benchmark.run()

# Generate visualizations
results.plot_latency_comparison(include_percentiles=True)
results.plot_throughput_scaling()
results.plot_memory_usage(detailed=True)
results.plot_flops_comparison(detailed=True)
```

### Low-Level API

For more control, you can use the visualization functions directly:

```python
from visualizers.plots import (
    plot_latency_comparison,
    plot_throughput_scaling,
    plot_memory_usage,
    plot_flops_comparison
)

# Customize output paths
latency_plot = plot_latency_comparison(results, output_path="custom_latency.png", include_percentiles=True)
memory_plot = plot_memory_usage(results, output_path="custom_memory.png", detailed=True)
```

## Dashboard Integration

The enhanced metrics are now fully integrated with the interactive dashboard, providing a comprehensive view of benchmark results:

```python
from visualizers.dashboard import generate_dashboard

# Generate interactive dashboard with all enhanced metrics
dashboard_path = generate_dashboard(results, output_dir="dashboard")
```

### Dashboard Features

The enhanced dashboard now includes:

- **Flexible Chart Types**: Choose between bar charts, line charts, and scatter plots
- **Metric Selection**: Select from an expanded list of metrics including:
  - Latency (mean, p90, p95, p99)
  - Throughput (items/sec)
  - Memory (total, peak, allocated, reserved, CPU)
  - FLOPs and GFLOPs
- **Specialized Visualizations**:
  - **Latency Percentiles**: Dedicated chart showing mean, p90, p95, p99 latency
  - **Memory Breakdown**: Stacked bar chart showing allocated, reserved, and CPU memory with peak memory overlay
  - **Batch Size Scaling**: Visualize how metrics scale with increasing batch size
- **Detailed Results Table**: Comprehensive table with all hardware-specific metrics
- **Interactive Filtering**: Filter by model, hardware platform, and metric type

## Implementation Details

### Latency Percentiles

The enhanced `LatencyMetric` now records and calculates:
- Mean latency
- Median latency
- Minimum and maximum latency
- Standard deviation
- p90, p95, and p99 percentiles

### Memory Tracking

The hardware-aware `MemoryMetric` now tracks:
- Peak memory usage
- Memory allocated by tensors
- Memory reserved by the allocator
- Memory growth during inference
- CPU memory usage

### FLOPs Estimation

The `FLOPsMetric` now provides:
- Total FLOPs count
- Model type detection (transformer, CNN, other)
- FLOPs breakdown by component
- Hardware-specific FLOPs estimation when possible

## Next Steps

With the dashboard and visualization enhancements now implemented, future work could include:

1. **Enhanced Export Options**: Add support for exporting visualizations in multiple formats (SVG, PDF)
2. **Notebook Integration**: Create Jupyter widget for interactive exploration within notebooks
3. **Advanced Statistical Analysis**: Add statistical significance testing between hardware platforms
4. **Historical Tracking**: Implement version tracking to compare performance across benchmark runs
5. **A/B Testing**: Support direct comparison between different model versions
6. **Hardware Efficiency Metrics**: Add visualizations for hardware utilization efficiency (FLOPs/watt, etc.)
7. **Customizable Dashboards**: Allow users to customize which charts/metrics are displayed
8. **3D Visualizations**: Create 3D visualizations for exploring relationships between multiple metrics