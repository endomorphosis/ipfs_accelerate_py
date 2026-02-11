# Benchmark Database Query Guide

## Overview

This guide explains how to use the fixed benchmark database query tool for the IPFS Accelerate Python Framework. The tool provides a robust interface for querying the benchmark database and generating reports.

## Installation

The tool is included in the test directory of the IPFS Accelerate Python Framework:

```bash
/home/barberb/ipfs_accelerate_py/duckdb_api/core/benchmark_db_query.py
```

## Requirements

- Python 3.8+
- DuckDB
- Pandas
- Matplotlib
- Seaborn
- Tabulate

## Basic Usage

### Set Database Path

You can set the database path using the `--db` parameter or the `BENCHMARK_DB_PATH` environment variable:

```bash
# Set via environment variable (recommended)
export BENCHMARK_DB_PATH=./benchmark_db.duckdb

# Or specify directly in the command
python duckdb_api/core/benchmark_db_query.py --db ./benchmark_db.duckdb --report summary
```

### Generate Reports

The tool can generate several types of reports:

1. **Summary Report**
   ```bash
   python duckdb_api/core/benchmark_db_query.py --report summary --format markdown --output benchmark_summary.md
   ```

2. **Hardware Report**
   ```bash
   python duckdb_api/core/benchmark_db_query.py --report hardware --format markdown --output hardware_report.md
   ```

3. **Performance Report**
   ```bash
   python duckdb_api/core/benchmark_db_query.py --report performance --format html --output benchmark_report.html
   ```

### Generate Hardware Compatibility Matrix

```bash
python duckdb_api/core/benchmark_db_query.py --compatibility-matrix --format markdown --output compatibility_matrix.md
```

### Model-Specific Analysis

1. **Show Data for a Specific Model**
   ```bash
   python duckdb_api/core/benchmark_db_query.py --model bert-base-uncased
   ```

2. **Compare Hardware Platforms for a Model**
   ```bash
   python duckdb_api/core/benchmark_db_query.py --model bert-base-uncased --compare-hardware
   ```

3. **Generate Performance Comparison Chart**
   ```bash
   python duckdb_api/core/benchmark_db_query.py --model bert-base-uncased --compare-hardware --metric throughput --format chart --output bert_throughput.png
   ```

### Hardware-Specific Analysis

```bash
python duckdb_api/core/benchmark_db_query.py --hardware cuda
```

### Custom SQL Queries

```bash
python duckdb_api/core/benchmark_db_query.py --sql "SELECT m.model_name, h.hardware_type, AVG(p.throughput_items_per_second) AS avg_throughput FROM performance_results p JOIN models m ON p.model_id = m.model_id JOIN hardware_platforms h ON p.hardware_id = h.hardware_id GROUP BY m.model_name, h.hardware_type ORDER BY avg_throughput DESC"
```

## Output Formats

The tool supports multiple output formats:

- **markdown**: Markdown table format
- **html**: HTML format with styling
- **csv**: Comma-separated values
- **json**: JSON format
- **chart**: Generated chart image (for comparison operations)

Example:
```bash
python duckdb_api/core/benchmark_db_query.py --report summary --format html --output summary.html
```

## Examples

### Generate a Summary Report

```bash
python duckdb_api/core/benchmark_db_query.py --report summary --format markdown --output benchmark_summary.md
```

This generates a summary report that includes:
- Count of models, hardware platforms, and benchmark results
- Latest test date
- Average performance by hardware

### Generate a Hardware Compatibility Matrix

```bash
python duckdb_api/core/benchmark_db_query.py --compatibility-matrix --format markdown --output compatibility_matrix.md
```

This generates a matrix showing which models have been tested on different hardware platforms.

### Compare Hardware Performance for BERT

```bash
python duckdb_api/core/benchmark_db_query.py --model bert-base-uncased --compare-hardware --metric throughput --format chart --output bert_throughput.png
```

This generates a chart comparing throughput for bert-base-uncased across different hardware platforms.

### Run a Custom SQL Query

```bash
python duckdb_api/core/benchmark_db_query.py --sql "SELECT m.model_name, h.hardware_type, p.batch_size, p.average_latency_ms, p.throughput_items_per_second FROM performance_results p JOIN models m ON p.model_id = m.model_id JOIN hardware_platforms h ON p.hardware_id = h.hardware_id ORDER BY p.throughput_items_per_second DESC LIMIT 10"
```

This runs a custom SQL query to find the top 10 fastest benchmark results.

## Advanced Features

### Trend Analysis (Coming Soon)

```bash
python duckdb_api/core/benchmark_db_query.py --trend performance --model bert-base-uncased --hardware cuda --metric throughput
```

This will generate a trend analysis showing how performance has changed over time (feature in development).

### WebGPU Report (Coming Soon)

```bash
python duckdb_api/core/benchmark_db_query.py --report webgpu --format html --output webgpu_report.html
```

This will generate a specialized report for WebGPU performance (feature in development).

## Conclusion

The fixed benchmark database query tool provides a robust interface for querying the benchmark database and generating reports. It is the recommended tool for interacting with the benchmark database in the IPFS Accelerate Python Framework.