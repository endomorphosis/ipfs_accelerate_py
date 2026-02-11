# Cross-Platform Mobile Analysis Tool Guide

This guide explains how to use the Cross-Platform Mobile Analysis Tool to compare and analyze model performance across Android and iOS platforms.

## Overview

The Cross-Platform Mobile Analysis Tool provides comprehensive capabilities for analyzing and comparing the performance of ML models between Android and iOS devices. It leverages benchmark data collected from both platforms to generate insights, visualizations, and optimization recommendations.

## Features

- Cross-platform performance comparison (Android vs iOS)
- Model performance analysis across different mobile hardware
- Battery impact comparison and analysis
- Thermal behavior analysis
- Hardware compatibility scoring
- Optimization recommendations for mobile deployment
- Report generation in multiple formats (Markdown, HTML)
- Performance visualizations

## Prerequisites

- Python 3.7+
- DuckDB database with Android and iOS benchmark results
- Matplotlib (for visualizations)
- NumPy (for data analysis)

## Installation

The tool is included as part of the IPFS Accelerate Python Framework. No additional installation steps are required beyond the framework's dependencies.

## Usage

### Command-Line Interface

The tool provides a command-line interface for various operations:

#### Compare Performance Across Platforms

```bash
python test/cross_platform_analysis.py compare --db-path /path/to/benchmark.duckdb [--model MODEL_NAME] [--output REPORT_FILE] [--format markdown|html]
```

This generates a comprehensive report comparing performance metrics between Android and iOS platforms. Optionally filter by model name.

#### Generate Performance Visualization

```bash
python test/cross_platform_analysis.py visualize --db-path /path/to/benchmark.duckdb --output visualization.png [--model MODEL_NAME]
```

This generates performance visualization charts comparing Android and iOS metrics. Optionally filter by model name.

#### Analyze Cross-Platform Performance

```bash
python test/cross_platform_analysis.py analyze --db-path /path/to/benchmark.duckdb [--model MODEL_NAME] [--output ANALYSIS_FILE]
```

This performs a detailed analysis of cross-platform performance and outputs the results in JSON format. Optionally filter by model name.

### Programmatic API

The tool can also be used programmatically:

```python
from test.cross_platform_analysis import CrossPlatformAnalyzer

# Initialize analyzer
analyzer = CrossPlatformAnalyzer("/path/to/benchmark.duckdb")

# Analyze performance
analysis = analyzer.analyze_cross_platform_performance(model_name="bert-base-uncased")

# Generate report
report = analyzer.generate_comparison_report(model_name="bert-base-uncased", format="markdown")

# Generate visualization
viz_path = analyzer.generate_visualization(analysis, output_path="performance_comparison.png")
```

## Data Requirements

The tool requires benchmark data for both Android and iOS platforms in the DuckDB database. The database should contain the following tables:

- `models`: Model information
- `android_benchmark_results`: Android benchmark results
- `ios_benchmark_results`: iOS benchmark results

The database schema should match the one used by the Android and iOS test harnesses.

## Report Contents

The generated reports include:

1. **Summary**: Overview of platform support and performance metrics
2. **Performance Comparison**: Detailed comparison of throughput, latency, battery impact
3. **Model-Specific Analysis**: Performance metrics for each model across platforms
4. **Optimization Recommendations**: Suggestions for platform selection and optimizations
5. **Implementation Guidelines**: Platform-specific implementation recommendations
6. **Visualizations**: Performance comparison charts

## Visualization Types

The tool generates the following visualizations:

- **Throughput Comparison**: Bar chart comparing throughput across platforms
- **Throughput Ratio**: Bar chart showing the iOS/Android throughput ratio
- **Performance Score**: Visualization of combined performance metrics

## Integration with Test Harnesses

The Cross-Platform Analysis Tool works with data from both the Android and iOS Test Harnesses. After running benchmarks on both platforms, use this tool to analyze and compare the results.

1. Run Android benchmarks using the Android Test Harness
2. Run iOS benchmarks using the iOS Test Harness
3. Use Cross-Platform Analysis Tool to compare results

## Examples

### Basic Comparison

```bash
python test/cross_platform_analysis.py compare --db-path data/benchmarks/benchmark.duckdb --output cross_platform_report.md
```

### Model-Specific Analysis

```bash
python test/cross_platform_analysis.py analyze --db-path data/benchmarks/benchmark.duckdb --model bert-base-uncased --output bert_analysis.json
```

### HTML Report with Visualization

```bash
python test/cross_platform_analysis.py compare --db-path data/benchmarks/benchmark.duckdb --model vit-base-patch16 --output vit_report.html --format html
```

## Troubleshooting

- **No data available**: Ensure benchmark tests have been run on both Android and iOS platforms
- **Visualization errors**: Check that matplotlib is installed correctly
- **Database connection errors**: Verify the database path is correct and the database exists

## Conclusion

The Cross-Platform Mobile Analysis Tool provides valuable insights for mobile ML deployment by comparing performance across Android and iOS platforms. Use these insights to make informed decisions about platform selection, optimization strategies, and hardware requirements for your mobile ML applications.