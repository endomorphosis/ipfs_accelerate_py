# WebNN/WebGPU Database Integration Guide

## Introduction

This guide explains how to use the database integration features for WebNN and WebGPU benchmarking with resource pool integration. The May 2025 update enhances the comprehensive DuckDB integration with support for concurrent model execution metrics, resource pool performance tracking, and real hardware validation.

## Key Features

- **Database-First Storage**: All benchmark results are stored directly in DuckDB (no JSON intermediates)
- **Resource Pool Metrics**: Track connection pooling efficiency and concurrent model execution
- **Browser-Specific Metrics**: Track performance across Chrome, Firefox, Edge, and Safari
- **Precision Level Tracking**: Compare 2-bit, 3-bit, 4-bit, 8-bit, 16-bit, and 32-bit precision
- **Real Hardware Validation**: Enhanced tracking to distinguish between genuine hardware acceleration and simulation
- **IPFS Acceleration Metrics**: Track IPFS content delivery and acceleration factors
- **Enhanced Analysis Capabilities**: Advanced SQL queries with statistical performance analysis
- **Visualization Integration**: Connect with interactive visualization tools through the database

## Getting Started

### Running Benchmarks with Database Integration

To run benchmarks and store results in the database, use the new resource pool benchmarking tools:

```bash
# Run resource pool benchmarks with WebGPU acceleration
python duckdb_api/core/benchmark_webnn_webgpu_resource_pool.py --model bert-base-uncased --platform webgpu --db-path ./benchmark_db.duckdb

# Benchmark concurrent model execution with 3 models
python duckdb_api/core/benchmark_webnn_webgpu_resource_pool.py --concurrent-models 3 --models bert-base-uncased,whisper-tiny,vit-base --db-path ./benchmark_db.duckdb

# Test Firefox audio optimizations for Whisper
python duckdb_api/core/benchmark_webnn_webgpu_resource_pool.py --browser firefox --model whisper-tiny --platform webgpu --optimize-audio --db-path ./benchmark_db.duckdb

# Run comprehensive benchmarks across all browsers and platforms with real hardware validation
python duckdb_api/core/benchmark_webnn_webgpu_resource_pool.py --comprehensive --db-path ./benchmark_db.duckdb
```

For real hardware validation and testing:

```bash
# Test real WebGPU implementation with hardware validation
python generators/models/test_ipfs_accelerate_with_real_webnn_webgpu.py --browser chrome --platform webgpu --model bert-base-uncased --db-path ./benchmark_db.duckdb

# Test Firefox with audio optimizations and real hardware validation
python generators/models/test_ipfs_accelerate_with_real_webnn_webgpu.py --browser firefox --model whisper-tiny --optimize-audio --db-path ./benchmark_db.duckdb

# Test Edge WebNN with real hardware validation
python generators/models/test_ipfs_accelerate_with_real_webnn_webgpu.py --browser edge --platform webnn --model bert-base-uncased --db-path ./benchmark_db.duckdb

# Run comprehensive hardware validation tests
python generators/models/test_ipfs_accelerate_with_real_webnn_webgpu.py --comprehensive --db-path ./benchmark_db.duckdb
```

Still compatible with legacy methods:

```bash
# Run using standard WebNN/WebGPU tests
python run_comprehensive_webnn_webgpu_tests.py --browser chrome --platform webgpu --model bert --db-path ./benchmark_db.duckdb

# Set database path through environment variable
export BENCHMARK_DB_PATH=./benchmarks/webnn_webgpu_results.duckdb
python run_comprehensive_webnn_webgpu_tests.py --browser chrome --platform webgpu --model bert --db-only
```

### Exploring Results in the Database

Once your benchmark results are stored in the database, you can analyze them using SQL queries:

```bash
# Use the duckdb_api/core/benchmark_db_query.py tool to analyze results
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "SELECT * FROM resource_pool_benchmarks" --db-path ./benchmark_db.duckdb

# Compare concurrent model execution
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "
  SELECT 
    model_name,
    platform,
    browser,
    concurrent_models,
    AVG(inference_time_ms) as avg_inference_time,
    AVG(ipfs_time_ms) as avg_ipfs_time,
    AVG(acceleration_factor) as avg_acceleration
  FROM 
    resource_pool_benchmarks
  WHERE 
    concurrent_models > 1
  GROUP BY 
    model_name, platform, browser, concurrent_models
  ORDER BY 
    concurrent_models DESC, avg_acceleration DESC
" --db-path ./benchmark_db.duckdb --format markdown

# Compare Firefox audio optimizations for Whisper models
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "
  SELECT 
    browser,
    browser_optimizations,
    AVG(latency_ms) as avg_latency,
    AVG(throughput_items_per_sec) as avg_throughput,
    AVG(acceleration_factor) as avg_acceleration
  FROM 
    resource_pool_benchmarks
  WHERE 
    model_name LIKE '%whisper%'
    AND platform = 'webgpu'
  GROUP BY 
    browser, browser_optimizations
  ORDER BY 
    avg_throughput DESC
" --db-path ./benchmark_db.duckdb --format markdown

# Compare real hardware vs. simulation performance
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "
  SELECT 
    platform,
    browser,
    is_real_implementation,
    COUNT(*) as benchmark_count,
    AVG(latency_ms) as avg_latency,
    AVG(throughput_items_per_sec) as avg_throughput,
    AVG(acceleration_factor) as avg_acceleration
  FROM 
    resource_pool_benchmarks
  GROUP BY 
    platform, browser, is_real_implementation
  ORDER BY 
    platform, browser, is_real_implementation
" --db-path ./benchmark_db.duckdb --format markdown

# Analyze resource pool connections and model initialization
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "
  SELECT 
    concurrent_models,
    AVG(init_time_ms) as avg_init_time,
    AVG(inference_time_ms) as avg_inference_time,
    AVG(ipfs_time_ms) as avg_ipfs_time,
    AVG(init_time_ms / concurrent_models) as avg_init_per_model
  FROM 
    resource_pool_benchmarks
  GROUP BY 
    concurrent_models
  ORDER BY 
    concurrent_models
" --db-path ./benchmark_db.duckdb --format markdown
```

### Generating Reports from Database

You can generate reports directly from the database:

```bash
# Generate a WebNN/WebGPU comparison report
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report webnn_webgpu_comparison --format html --output comparison_report.html

# Generate browser comparison report
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report browser_comparison --format html --output browser_report.html

# Generate precision level comparison report
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report precision_comparison --format html --output precision_report.html

# Generate model compatibility report
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --report model_compatibility --format markdown --output compatibility.md
```

## Database Schema

The database schema for WebNN/WebGPU benchmarks includes:

### Performance Results Table

Stores the actual benchmark performance metrics:

```sql
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'performance_results'
```

Key fields:
- `result_id`: Unique identifier
- `model_id`: Reference to the model being tested
- `hardware_id`: Reference to hardware platform (includes browser info)
- `average_latency_ms`: Average inference latency in milliseconds
- `throughput_items_per_second`: Throughput in items per second
- `memory_peak_mb`: Peak memory usage in MB
- `precision`: Precision level used (e.g., "4bit", "8bit")
- `metrics`: JSON field with additional metrics, including simulation status and browser

### Hardware Platforms Table

Stores information about hardware platforms, including WebNN and WebGPU details:

```sql
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'hardware_platforms'
```

Key fields:
- `hardware_id`: Unique identifier
- `hardware_type`: Type of hardware (e.g., "webgpu_chrome", "webnn_edge")
- `device_name`: Device name, including browser information
- `platform`: Platform name (WebNN or WebGPU)

### Hardware Compatibility Table

Tracks which models are compatible with which hardware platforms:

```sql
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'hardware_compatibility'
```

Key fields:
- `compatibility_id`: Unique identifier
- `model_id`: Reference to model
- `hardware_id`: Reference to hardware platform
- `is_compatible`: Boolean indicating compatibility
- `is_simulation`: Boolean indicating whether compatibility is via simulation

## Best Practices

1. **Use Consistent Hardware Types**: Always use consistent hardware_type strings (e.g., "webgpu_chrome", "webnn_edge") for accurate comparisons.

2. **Track Simulation Status**: Always record whether results are from real hardware or simulation for accurate analysis.

3. **Include Browser Version**: When possible, include browser version information for more detailed analysis.

4. **Store Raw Results**: For important benchmarks, store raw results (not just averages) for statistical analysis.

5. **Regular Database Backups**: Regularly back up your benchmark database to prevent data loss.

## Troubleshooting

- **Connection Errors**: Ensure the database path is valid and the directory exists.
- **Missing Dependencies**: Make sure DuckDB is installed (`pip install duckdb`).
- **Permission Issues**: Check that you have write permission to the database file.
- **Schema Issues**: If you encounter schema errors, check that your database was properly initialized.

## Advanced Topics

### Extending the Schema

If you need to track additional metrics:

1. Update the database schema to include new columns:
```python
conn.execute("ALTER TABLE performance_results ADD COLUMN new_metric FLOAT")
```

2. Modify the `store_benchmark_result` function to include the new metrics.

### Custom Reports

Create custom reports by defining new functions in duckdb_api/core/benchmark_db_query.py:

```python
def generate_custom_report(db_path, format="html"):
    """Generate a custom report from benchmark data."""
    # Connect to database
    db = BenchmarkDBAPI(db_path)
    
    # Query data
    data = db.query("SELECT * FROM performance_results WHERE ...")
    
    # Generate report
    if format == "html":
        # Generate HTML report
        pass
    elif format == "markdown":
        # Generate Markdown report
        pass
    
    return report_path
```

### Integration with Visualization Tools

For advanced visualization, export data to formats compatible with visualization tools:

```bash
# Export to CSV for Excel/Tableau
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "SELECT ... FROM performance_results" --format csv --output data.csv

# Export to JSON for JavaScript visualizations
python duckdb_api/core/duckdb_api/core/benchmark_db_query.py --sql "SELECT ... FROM performance_results" --format json --output data.json
```

## Conclusion

The database integration for WebNN/WebGPU benchmarking provides a robust foundation for storing, analyzing, and visualizing benchmark results. By leveraging the power of DuckDB, you can perform complex analyses and generate insightful reports on the performance of various models across different browsers, precision levels, and optimization settings.