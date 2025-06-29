# Benchmark Visualization Guide

**Date: March 7, 2025**  
**Version: 1.0**

This guide focuses specifically on visualizing benchmark results from the IPFS Accelerate Python Framework. Learn how to generate insightful visualizations from benchmark data stored in the DuckDB database to compare model performance across different hardware platforms, configurations, and over time.

## Table of Contents

1. [Introduction](#introduction)
2. [Benchmark Database Structure](#benchmark-database-structure)
3. [Basic Benchmark Visualizations](#basic-benchmark-visualizations)
4. [Advanced Benchmark Visualizations](#advanced-benchmark-visualizations)
5. [Automated Reporting](#automated-reporting)
6. [Custom Benchmark Visualization](#custom-benchmark-visualization)
7. [Integration with CI/CD](#integration-with-cicd)
8. [Best Practices](#best-practices)

## Introduction

The IPFS Accelerate Python Framework's benchmark database contains valuable data about model performance across different hardware platforms, configurations, and over time. This guide will help you transform that data into meaningful visualizations that can inform decisions about hardware selection, optimization strategies, and model deployment.

Key benefits of benchmark visualization:
- Compare performance across hardware platforms
- Identify optimization opportunities
- Track performance improvements over time
- Make data-driven decisions about model deployment
- Share insights with stakeholders

## Benchmark Database Structure

The benchmark database uses a DuckDB schema optimized for performance analysis:

```sql
-- Main performance results table
CREATE TABLE performance_results (
    result_id INTEGER PRIMARY KEY,
    model_id INTEGER,
    hardware_id INTEGER,
    batch_size INTEGER,
    sequence_length INTEGER,
    precision VARCHAR,
    throughput_items_per_second FLOAT,
    latency_ms FLOAT,
    memory_mb FLOAT,
    power_watts FLOAT,
    timestamp TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
);

-- Models table
CREATE TABLE models (
    model_id INTEGER PRIMARY KEY,
    model_name VARCHAR,
    model_type VARCHAR,
    model_family VARCHAR,
    model_size VARCHAR
);

-- Hardware platforms table
CREATE TABLE hardware_platforms (
    hardware_id INTEGER PRIMARY KEY,
    hardware_type VARCHAR,
    device_name VARCHAR,
    compute_units INTEGER,
    memory_capacity FLOAT
);
```

## Basic Benchmark Visualizations

### Performance Comparison Chart

Compare model performance across hardware platforms:

```python
from fixed_web_platform.benchmark_visualization import BenchmarkVisualizer

# Create benchmark visualizer
visualizer = BenchmarkVisualizer(db_path="./benchmark_db.duckdb")

# Generate performance comparison chart
visualizer.compare_performance(
    model_name="bert-base-uncased",
    metric="throughput_items_per_second",
    hardware_platforms=["cuda", "rocm", "mps", "webgpu", "qualcomm"],
    batch_size=1,
    output_path="bert_performance_comparison.html"
)
```

Example output:

![Performance Comparison Chart](https://example.com/performance_comparison.png)

### Memory Usage Comparison

Compare memory usage across hardware platforms:

```python
# Generate memory usage comparison chart
visualizer.compare_performance(
    model_name="bert-base-uncased",
    metric="memory_mb",
    hardware_platforms=["cuda", "rocm", "mps", "webgpu", "qualcomm"],
    batch_size=1,
    output_path="bert_memory_comparison.html"
)
```

### Batch Size Scaling

Visualize how throughput scales with batch size:

```python
# Generate batch size scaling chart
visualizer.batch_size_scaling(
    model_name="bert-base-uncased",
    hardware_platform="cuda",
    batch_sizes=[1, 2, 4, 8, 16, 32],
    metric="throughput_items_per_second",
    output_path="bert_batch_scaling_cuda.html"
)
```

Example output:

![Batch Size Scaling Chart](https://example.com/batch_scaling.png)

### Model Size Comparison

Compare performance across different model sizes:

```python
# Generate model size comparison chart
visualizer.compare_model_sizes(
    model_family="bert",
    model_sizes=["tiny", "mini", "small", "base", "large"],
    hardware_platform="cuda",
    metric="throughput_items_per_second",
    batch_size=1,
    output_path="bert_size_comparison_cuda.html"
)
```

## Advanced Benchmark Visualizations

### Performance Heatmap

Create a heatmap showing performance across model types and hardware platforms:

```python
# Generate performance heatmap
visualizer.create_heatmap(
    model_families=["text", "vision", "audio", "multimodal"],
    hardware_platforms=["cuda", "rocm", "mps", "webgpu", "qualcomm"],
    metric="throughput_items_per_second",
    normalize=True,  # Normalize values for better comparison
    output_path="performance_heatmap.html"
)
```

Example output:

![Performance Heatmap](https://example.com/performance_heatmap.png)

### Precision Impact Analysis

Analyze the impact of precision on performance and memory usage:

```python
# Generate precision impact chart
visualizer.analyze_precision_impact(
    model_name="llama-7b",
    hardware_platform="cuda",
    precisions=["fp32", "fp16", "int8", "int4"],
    metrics=["throughput_items_per_second", "memory_mb"],
    output_path="llama_precision_impact.html"
)
```

### Multi-Metric Radar Chart

Create a radar chart comparing multiple metrics across hardware platforms:

```python
# Generate multi-metric radar chart
visualizer.create_radar_chart(
    model_name="bert-base-uncased",
    hardware_platforms=["cuda", "rocm", "mps", "webgpu", "qualcomm"],
    metrics=[
        "throughput_items_per_second",
        "latency_ms",
        "memory_mb",
        "power_watts"
    ],
    normalize=True,
    output_path="bert_radar_chart.html"
)
```

Example output:

![Multi-Metric Radar Chart](https://example.com/radar_chart.png)

### Performance-Memory Tradeoff

Visualize the tradeoff between performance and memory usage:

```python
# Generate performance-memory tradeoff chart
visualizer.create_scatter_plot(
    model_names=["bert-tiny", "bert-mini", "bert-small", "bert-base", "bert-large"],
    hardware_platform="cuda",
    x_metric="memory_mb",
    y_metric="throughput_items_per_second",
    color_by="model_size",
    output_path="bert_performance_memory_tradeoff.html"
)
```

## Automated Reporting

### Generate Comprehensive Benchmark Report

Create a complete benchmark report with multiple visualizations:

```python
from fixed_web_platform.benchmark_reporting import BenchmarkReporter

# Create benchmark reporter
reporter = BenchmarkReporter(db_path="./benchmark_db.duckdb")

# Generate comprehensive report
reporter.generate_report(
    title="BERT Model Benchmark Report",
    model_name="bert-base-uncased",
    hardware_platforms=["cuda", "rocm", "mps", "webgpu", "qualcomm"],
    metrics=["throughput_items_per_second", "latency_ms", "memory_mb", "power_watts"],
    batch_sizes=[1, 4, 16],
    include_executive_summary=True,
    include_recommendations=True,
    output_path="bert_benchmark_report.html"
)
```

### Generate Periodic Benchmark Reports

Schedule regular benchmark reports:

```python
# Create scheduled reporter
from fixed_web_platform.benchmark_reporting import ScheduledReporter

# Create scheduled reporter
scheduled_reporter = ScheduledReporter(db_path="./benchmark_db.duckdb")

# Schedule weekly report
scheduled_reporter.schedule_report(
    report_config={
        "title": "Weekly Performance Report",
        "model_families": ["text", "vision", "audio"],
        "hardware_platforms": ["cuda", "webgpu"],
        "metrics": ["throughput_items_per_second", "memory_mb"],
        "output_path": "reports/weekly_performance_report_{date}.html"
    },
    frequency="weekly",
    weekday="monday",
    time="09:00"
)
```

### Command-Line Reporting

Generate reports from the command line:

```bash
# Generate benchmark report for specific model
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report performance --model bert-base-uncased --format html --output reports/bert_report.html

# Generate comparative hardware report
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report hardware_comparison --models bert-base-uncased,t5-small,vit-base --format html --output reports/hardware_comparison.html

# Generate trend report for specific model
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report performance_trend --model bert-base-uncased --hardware cuda --time-range "last_3_months" --format html --output reports/bert_trend_report.html
```

## Custom Benchmark Visualization

### Using SQL Queries for Custom Visualizations

Create custom visualizations using SQL queries:

```python
# Get performance data with custom SQL
from fixed_web_platform.benchmark_visualization import SQLBenchmarkVisualizer

# Create SQL benchmark visualizer
sql_viz = SQLBenchmarkVisualizer(db_path="./benchmark_db.duckdb")

# Generate custom visualization
sql_viz.visualize_query(
    query="""
    SELECT m.model_name, h.hardware_type, AVG(p.throughput_items_per_second) as avg_throughput, 
           STDDEV(p.throughput_items_per_second) as std_throughput
    FROM performance_results p
    JOIN models m ON p.model_id = m.model_id
    JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
    WHERE m.model_family = 'bert' AND p.batch_size = 1
    GROUP BY m.model_name, h.hardware_type
    ORDER BY avg_throughput DESC
    """,
    chart_type="bar",
    x_column="model_name",
    y_column="avg_throughput",
    error_column="std_throughput",
    color_column="hardware_type",
    title="BERT Family Performance Comparison",
    output_path="custom_bert_comparison.html"
)
```

### Integration with External BI Tools

Export benchmark data for use with external BI tools:

```python
from fixed_web_platform.benchmark_export import BenchmarkExporter

# Create benchmark exporter
exporter = BenchmarkExporter(db_path="./benchmark_db.duckdb")

# Export data for Power BI
exporter.export_to_powerbi(
    query="""
    SELECT m.model_name, m.model_family, h.hardware_type, p.batch_size, 
           p.throughput_items_per_second, p.latency_ms, p.memory_mb, p.timestamp
    FROM performance_results p
    JOIN models m ON p.model_id = m.model_id
    JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
    WHERE p.timestamp >= DATE_SUB(NOW(), INTERVAL 3 MONTH)
    """,
    output_path="exports/benchmark_data_powerbi.pbix"
)

# Export data for Tableau
exporter.export_to_tableau(
    query="""
    SELECT m.model_name, m.model_family, h.hardware_type, p.batch_size, 
           p.throughput_items_per_second, p.latency_ms, p.memory_mb, p.timestamp
    FROM performance_results p
    JOIN models m ON p.model_id = m.model_id
    JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
    WHERE p.timestamp >= DATE_SUB(NOW(), INTERVAL 3 MONTH)
    """,
    output_path="exports/benchmark_data_tableau.hyper"
)
```

## Integration with CI/CD

### Automated Benchmark Visualization in CI/CD

Integrate benchmark visualization into CI/CD pipelines:

```yaml
# GitHub Actions workflow example
name: Benchmark Visualization

on:
  workflow_dispatch:
  schedule:
    - cron: '0 0 * * 1'  # Run every Monday

jobs:
  generate-visualizations:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v2
        
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          
      - name: Generate benchmark visualizations
        run: |
          python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report weekly_performance --format html --output reports/weekly_performance.html
          python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report hardware_comparison --format html --output reports/hardware_comparison.html
          
      - name: Upload visualization artifacts
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-visualizations
          path: reports/*.html
```

### Performance Regression Detection

Automatically detect performance regressions:

```python
from fixed_web_platform.benchmark_analysis import RegressionDetector

# Create regression detector
detector = RegressionDetector(db_path="./benchmark_db.duckdb")

# Detect performance regressions
regressions = detector.detect_regressions(
    model_name="bert-base-uncased",
    hardware_platform="cuda",
    metric="throughput_items_per_second",
    threshold_percent=5,  # 5% regression threshold
    time_window="last_30_days"
)

# Generate regression report
if regressions:
    detector.generate_regression_report(
        regressions=regressions,
        output_path="reports/regression_report.html"
    )
    
    # Send alert
    detector.send_regression_alert(
        regressions=regressions,
        notification_method="email",
        recipients=["team@example.com"]
    )
```

## Best Practices

### Benchmark Visualization Best Practices

1. **Focus on Key Metrics**:
   - Throughput (items/second) for overall performance
   - Latency (ms) for response time sensitivity
   - Memory usage (MB) for resource constraints
   - Power consumption (watts) for edge/mobile deployment

2. **Provide Context**:
   - Include baseline comparisons
   - Show relative improvement percentages
   - Add reference points (e.g., real-time threshold)

3. **Use Appropriate Scales**:
   - Log scale for wide-ranging values
   - Consistent scales for direct comparisons
   - Zero-baseline for bar charts

4. **Include Statistical Information**:
   - Error bars for variance
   - Min/max/mean/median values
   - Confidence intervals for predictions

5. **Optimize for Different Audiences**:
   - Executive summaries for high-level stakeholders
   - Detailed technical views for engineers
   - Comparative views for decision-makers

### Common Benchmark Visualization Patterns

| Pattern | When to Use |
|---------|-------------|
| Bar Chart Comparison | Compare performance across discrete categories (hardware types, model sizes) |
| Line Chart | Show scaling behavior (batch size impact) or trends over time |
| Heatmap | Visualize performance across multiple dimensions (model types × hardware platforms) |
| Scatter Plot | Show relationships between metrics (throughput vs. memory usage) |
| Radar Chart | Compare multiple metrics across hardware platforms |
| Box Plot | Show distribution of performance results with outliers |

## Related Documentation

- [Visualization Guide](VISUALIZATION_GUIDE.md)
- [Benchmark Database Guide](../BENCHMARK_DATABASE_GUIDE.md)
- [Performance Dashboard Specification](../PERFORMANCE_DASHBOARD_SPECIFICATION.md)
- [Hardware Benchmarking Guide](../HARDWARE_BENCHMARKING_GUIDE.md)
- [CI/CD Integration Guide](CI_CD_INTEGRATION_GUIDE.md)