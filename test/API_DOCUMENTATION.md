# IPFS Accelerate Database API Documentation

**Date: March 6, 2025**  
**Status: Complete**  
**Author: Claude**  
**Version: 1.1**

> **Key Updates in v1.1:**
> - Added advanced query patterns with examples
> - Expanded troubleshooting section with common errors
> - Added Qualcomm-specific metrics and queries
> - Improved Python and CLI examples with actual command outputs
> - Added performance tips and optimization guide

This document provides comprehensive documentation for the database API in the IPFS Accelerate Python Framework, focusing on how to programmatically interact with the DuckDB database for storing, retrieving, and analyzing test results.

## Overview

The database API provides a set of functions for interacting with the DuckDB database used by the IPFS Accelerate Python Framework. It is designed to be simple, efficient, and flexible, allowing you to:

- Store test results, performance metrics, and hardware compatibility information
- Query test results with custom filters and aggregations
- Generate reports and visualizations
- Track compatibility and performance trends over time

## Core Components

The database API consists of two main components:

1. **`TestResultsDBHandler`** in `test_ipfs_accelerate.py`: Primary interface for storing test results
2. **`duckdb_api/core/benchmark_db_query.py`**: Tool for querying and generating reports from the database

### TestResultsDBHandler

The `TestResultsDBHandler` class provides a comprehensive interface for storing test results in the database:

```python
from test_ipfs_accelerate import TestResultsDBHandler

# Initialize handler
db_handler = TestResultsDBHandler(db_path="./benchmark_db.duckdb")

# Store a test result
test_result = {
    "model_name": "bert-base-uncased",
    "hardware_type": "cuda",
    "test_type": "inference",
    "status": "Success",
    "success": True,
    "execution_time": 1.25,
    "memory_usage": 2048.5,
    "details": {"batch_size": 16, "sequence_length": 128}
}
db_handler.store_test_result(test_result)

# Generate a report
report = db_handler.generate_report(format="markdown", output_file="report.md")
```

### duckdb_api/core/benchmark_db_query.py

The `duckdb_api/core/benchmark_db_query.py` script provides a command-line interface and programmatic API for querying the database:

```python
from fixed_benchmark_db_query import BenchmarkDBQuery

# Initialize query tool
query_tool = BenchmarkDBQuery(db_path="./benchmark_db.duckdb")

# Get model information
models = query_tool.get_models()

# Get hardware platforms
hardware_platforms = query_tool.get_hardware_platforms()

# Generate a performance report
query_tool.generate_performance_report(
    format="html",
    output="performance_report.html",
    model="bert-base-uncased"
)

# Generate a compatibility matrix
query_tool.generate_compatibility_matrix(
    format="markdown",
    output="compatibility_matrix.md",
    filter_family="text_embedding"
)
```

## Database Schema

The database schema is designed to efficiently store and query test results, with a focus on hardware compatibility, performance metrics, and power usage.

### Key Tables

#### models

Stores information about the models being tested:

| Column | Type | Description |
|--------|------|-------------|
| model_id | INTEGER | Primary key |
| model_name | VARCHAR | Name of the model |
| model_family | VARCHAR | Family of the model (e.g., text_embedding, text_generation) |
| model_type | VARCHAR | Type of the model (e.g., transformer, cnn) |
| model_size | VARCHAR | Size category (tiny, small, medium, large) |
| parameters_million | FLOAT | Number of parameters in millions |
| added_at | TIMESTAMP | When the model was added to the database |

#### hardware_platforms

Stores information about the hardware platforms being tested:

| Column | Type | Description |
|--------|------|-------------|
| hardware_id | INTEGER | Primary key |
| hardware_type | VARCHAR | Type of hardware (cpu, cuda, rocm, etc.) |
| device_name | VARCHAR | Name of the device |
| compute_units | INTEGER | Number of compute units |
| memory_capacity | FLOAT | Memory capacity in GB |
| driver_version | VARCHAR | Driver version |
| supported_precisions | VARCHAR | Supported precision formats |
| max_batch_size | INTEGER | Maximum batch size |
| detected_at | TIMESTAMP | When the platform was detected |

#### test_results

Stores the results of tests:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | TIMESTAMP | When the test was run |
| test_date | VARCHAR | Date of the test (YYYY-MM-DD) |
| status | VARCHAR | Status of the test (Success, Failed, etc.) |
| test_type | VARCHAR | Type of test (inference, training, etc.) |
| model_id | INTEGER | Foreign key to models table |
| hardware_id | INTEGER | Foreign key to hardware_platforms table |
| endpoint_type | VARCHAR | Type of endpoint (local, api, etc.) |
| success | BOOLEAN | Whether the test was successful |
| error_message | VARCHAR | Error message if the test failed |
| execution_time | FLOAT | Execution time in seconds |
| memory_usage | FLOAT | Memory usage in MB |
| details | VARCHAR | JSON string with additional details |

#### performance_results

Stores detailed performance metrics:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| model_id | INTEGER | Foreign key to models table |
| hardware_id | INTEGER | Foreign key to hardware_platforms table |
| batch_size | INTEGER | Batch size used |
| sequence_length | INTEGER | Sequence length used |
| average_latency_ms | FLOAT | Average latency in milliseconds |
| p50_latency_ms | FLOAT | 50th percentile latency in milliseconds |
| p90_latency_ms | FLOAT | 90th percentile latency in milliseconds |
| p99_latency_ms | FLOAT | 99th percentile latency in milliseconds |
| throughput_items_per_second | FLOAT | Throughput in items per second |
| memory_peak_mb | FLOAT | Peak memory usage in MB |
| power_watts | FLOAT | Power consumption in watts |
| energy_efficiency_items_per_joule | FLOAT | Energy efficiency in items per joule |
| test_timestamp | TIMESTAMP | When the performance test was run |

#### hardware_compatibility

Stores hardware compatibility information:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| model_id | INTEGER | Foreign key to models table |
| hardware_id | INTEGER | Foreign key to hardware_platforms table |
| compatibility_status | VARCHAR | Compatibility status (supported, limited, unsupported) |
| compatibility_score | FLOAT | Compatibility score (0 to 1) |
| recommended | BOOLEAN | Whether this hardware is recommended for this model |
| last_tested | TIMESTAMP | When compatibility was last tested |

#### power_metrics

Stores power and thermal metrics for mobile/edge devices:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| test_id | INTEGER | Foreign key to test_results table |
| model_id | INTEGER | Foreign key to models table |
| hardware_id | INTEGER | Foreign key to hardware_platforms table |
| power_watts_avg | FLOAT | Average power consumption in watts |
| power_watts_peak | FLOAT | Peak power consumption in watts |
| temperature_celsius_avg | FLOAT | Average temperature in Celsius |
| temperature_celsius_peak | FLOAT | Peak temperature in Celsius |
| battery_impact_mah | FLOAT | Estimated battery impact in mAh |
| test_duration_seconds | FLOAT | Test duration in seconds |
| estimated_runtime_hours | FLOAT | Estimated runtime in hours |
| test_timestamp | TIMESTAMP | When the power test was run |

## Common Usage Patterns

### Storing Test Results

```python
# Basic test result
test_result = {
    "model_name": "bert-base-uncased",
    "hardware_type": "cuda",
    "test_type": "inference",
    "status": "Success",
    "success": True,
    "execution_time": 1.25,
    "memory_usage": 2048.5,
    "details": {"batch_size": 16, "sequence_length": 128}
}
db_handler.store_test_result(test_result)

# With performance metrics
test_result["performance"] = {
    "batch_size": 16,
    "sequence_length": 128,
    "average_latency_ms": 5.23,
    "throughput_items_per_second": 305.9,
    "memory_peak_mb": 3175.2
}
db_handler.store_test_result(test_result)

# With power metrics for mobile/edge devices
test_result["power_metrics"] = {
    "power_watts_avg": 3.2,
    "power_watts_peak": 5.7,
    "temperature_celsius_avg": 42.5,
    "temperature_celsius_peak": 52.1,
    "battery_impact_mah": 125.3,
    "test_duration_seconds": 300.0,
    "estimated_runtime_hours": 4.3
}
db_handler.store_test_result(test_result)
```

### Querying the Database

```python
import duckdb

# Connect to the database
conn = duckdb.connect("./benchmark_db.duckdb")

# Get all models
models = conn.execute("SELECT * FROM models").fetchdf()

# Get all hardware platforms
hardware = conn.execute("SELECT * FROM hardware_platforms").fetchdf()

# Get performance data for a specific model
performance = conn.execute("""
    SELECT 
        h.hardware_type,
        p.batch_size,
        p.average_latency_ms,
        p.throughput_items_per_second,
        p.memory_peak_mb
    FROM performance_results p
    JOIN models m ON p.model_id = m.model_id
    JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
    WHERE m.model_name = 'bert-base-uncased'
    ORDER BY p.throughput_items_per_second DESC
""").fetchdf()

# Get compatibility matrix
compatibility = conn.execute("""
    SELECT 
        m.model_name,
        m.model_family,
        MAX(CASE WHEN h.hardware_type = 'cpu' THEN 1 ELSE 0 END) as cpu_support,
        MAX(CASE WHEN h.hardware_type = 'cuda' THEN 1 ELSE 0 END) as cuda_support,
        MAX(CASE WHEN h.hardware_type = 'rocm' THEN 1 ELSE 0 END) as rocm_support,
        MAX(CASE WHEN h.hardware_type = 'qualcomm' THEN 1 ELSE 0 END) as qualcomm_support,
        MAX(CASE WHEN h.hardware_type = 'webgpu' THEN 1 ELSE 0 END) as webgpu_support
    FROM models m
    LEFT JOIN test_results tr ON m.model_id = tr.model_id
    LEFT JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
    GROUP BY m.model_name, m.model_family
""").fetchdf()
```

### Generating Reports

```python
# Generate a markdown report
report = db_handler.generate_report(format="markdown", output_file="report.md")

# Generate an HTML report
report = db_handler.generate_report(format="html", output_file="report.html")

# Generate a JSON report
report = db_handler.generate_report(format="json", output_file="report.json")
```

## Command-Line Interface

The database API can also be accessed through command-line tools:

### TestResultsDBHandler CLI

```bash
# Generate a report directly from test_ipfs_accelerate.py
python test_ipfs_accelerate.py --report --format markdown --output test_report.md

# Run tests and store results in the database
python test_ipfs_accelerate.py --models bert-base-uncased,t5-small
```

### duckdb_api/core/benchmark_db_query.py CLI

```bash
# Generate a summary report
python duckdb_api/core/benchmark_db_query.py --report summary --format markdown --output summary.md

# Generate a hardware report
python duckdb_api/core/benchmark_db_query.py --report hardware --format html --output hardware.html

# Generate a compatibility matrix
python duckdb_api/core/benchmark_db_query.py --compatibility-matrix --format markdown --output matrix.md

# Compare hardware performance for a model
python duckdb_api/core/benchmark_db_query.py --model bert-base-uncased --compare-hardware --metric throughput --format chart --output chart.png

# Run a custom SQL query
python duckdb_api/core/benchmark_db_query.py --sql "SELECT * FROM performance_results" --format csv --output results.csv
```

### migrate_json_to_db.py CLI

```bash
# Migrate JSON files to the database
python migrate_json_to_db.py --directories ./benchmark_results ./archived_test_results --report migration_report.md

# Archive and delete JSON files after migration
python migrate_json_to_db.py --directories ./benchmark_results --archive-dir ./archived_json_files --delete
```

## Best Practices

1. **Use Environment Variables**: Set the database path using the `BENCHMARK_DB_PATH` environment variable for consistent access across tools.
   ```bash
   export BENCHMARK_DB_PATH=./benchmark_db.duckdb
   ```

2. **Regular Backups**: Create regular backups of the database, especially before running large migration operations.
   ```bash
   cp benchmark_db.duckdb benchmark_db_backup_$(date +%Y%m%d).duckdb
   ```

3. **Structured Test Results**: Organize test results with consistent keys to ensure proper database storage.

4. **Batch Operations**: When storing multiple test results, use batch operations for better performance.

5. **Query Optimization**: Use indexes and limit result sets when querying large tables.

6. **Error Handling**: Always include error handling when interacting with the database.

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Ensure the database file exists and is accessible
   - Check for correct permissions on the database file
   - Verify the database path is correctly specified

   ```python
   # Example error:
   # Error: IO Error: Cannot open file "./benchmark_db.duckdb": No such file or directory
   
   # Solution:
   import os
   
   # Check if file exists
   if not os.path.exists("./benchmark_db.duckdb"):
       print("Database file doesn't exist, creating new one...")
       conn = duckdb.connect("./benchmark_db.duckdb")
       # Initialize schema...
   else:
       # Check permissions
       if not os.access("./benchmark_db.duckdb", os.R_OK | os.W_OK):
           print("Permission denied on database file")
       else:
           conn = duckdb.connect("./benchmark_db.duckdb")
   ```

2. **Data Insertion Errors**:
   - Ensure required fields are present in test results
   - Check data types match the schema requirements
   - Verify foreign key relationships are valid

   ```python
   # Example error:
   # Constraint Error: Constraint violation: NOT NULL constraint failed: models.model_name
   
   # Solution - validate data before insertion:
   def validate_test_result(result):
       required_fields = ["model_name", "hardware_type", "test_type"]
       for field in required_fields:
           if field not in result or result[field] is None:
               raise ValueError(f"Missing required field: {field}")
       
       # Validate data types
       if not isinstance(result.get("execution_time", 0), (int, float)):
           raise ValueError("execution_time must be a number")
       
       return True
   
   # Use with try/except:
   try:
       if validate_test_result(test_result):
           db_handler.store_test_result(test_result)
   except ValueError as e:
       print(f"Invalid test result: {e}")
   ```

3. **Query Performance Issues**:
   - Use indexes on frequently queried columns
   - Limit result sets when querying large tables
   - Use aggregation in the database rather than in Python

   ```python
   # Slow query:
   results = conn.execute("SELECT * FROM test_results").fetchdf()
   # Process millions of rows in Python...
   
   # Improved query - do processing in database:
   results = conn.execute("""
       SELECT 
           m.model_name, 
           h.hardware_type, 
           AVG(tr.execution_time) as avg_time
       FROM test_results tr
       JOIN models m ON tr.model_id = m.model_id
       JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
       GROUP BY m.model_name, h.hardware_type
   """).fetchdf()
   ```

4. **NULL Value Handling**:
   - Be careful with NULL values in SQL queries
   - Use COALESCE or IS NULL/IS NOT NULL checks

   ```python
   # Problematic query (NULL values skipped):
   conn.execute("""
       SELECT AVG(power_watts) FROM performance_results
   """).fetchone()
   
   # Better query with NULL handling:
   conn.execute("""
       SELECT AVG(COALESCE(power_watts, 0)) FROM performance_results
   """).fetchone()
   
   # Filtering with NULL values:
   conn.execute("""
       SELECT * FROM performance_results WHERE power_watts IS NOT NULL
   """).fetchdf()
   ```

5. **JSON Field Extraction Issues**:
   - Use correct syntax for JSON field extraction
   - Handle NULL JSON fields properly

   ```python
   # Common error with JSON extraction:
   # Error: Cannot extract field from non-object JSON value
   
   # Solution - use safe extraction and check for NULL:
   conn.execute("""
       SELECT 
           tr.details->>'batch_size' as batch_size
       FROM test_results tr
       WHERE tr.details IS NOT NULL AND tr.details->>'batch_size' IS NOT NULL
   """).fetchdf()
   ```

### Diagnostic Tools

```bash
# Check database integrity
python scripts/benchmark_db_maintenance.py --check-integrity

# Output:
# Checking database integrity for: ./benchmark_db.duckdb
# Verifying table structure...
# Verifying foreign key constraints...
# Checking for orphaned records...
# Running consistency checks...
# Database integrity check passed: No issues found

# Optimize database performance
python scripts/benchmark_db_maintenance.py --optimize-db

# Output:
# Analyzing database: ./benchmark_db.duckdb
# Database size before optimization: 485.7 MB
# Running VACUUM...
# Running ANALYZE...
# Database size after optimization: 432.3 MB
# Optimization complete: 11.0% reduction in size

# Generate database statistics
python scripts/benchmark_db_maintenance.py --generate-stats

# Output:
# Generating statistics for database: ./benchmark_db.duckdb
# Table counts:
# - models: 328 rows
# - hardware_platforms: 8 rows
# - test_results: 12,453 rows
# - performance_results: 8,721 rows
# - power_metrics: 3,245 rows
# - hardware_compatibility: 2,624 rows
# Most tested models:
# - bert-base-uncased: 1,245 tests
# - t5-small: 987 tests
# - vit-base: 823 tests
# Most used hardware:
# - cuda: 5,432 tests
# - cpu: 4,213 tests
# - qualcomm: 1,532 tests
# Statistics generated successfully
```

### Error Dictionary

Here's a reference table of common DuckDB errors and their solutions:

| Error | Description | Solution |
|-------|-------------|----------|
| `Cannot open file` | Database file doesn't exist or can't be accessed | Check file path, permissions, and create directory if needed |
| `Constraint violation: NOT NULL` | Required field is missing | Validate data before insertion, provide default values |
| `Constraint violation: FOREIGN KEY` | Foreign key constraint failed | Ensure referenced entity exists first |
| `IO Error: Read-only file system` | Database file is on a read-only filesystem | Move database to a writable location |
| `Out of memory exception` | Query uses too much memory | Limit result set, use filters, optimize query |
| `SQL Logic Error: duplicate column name` | Column name appears multiple times | Use column aliases in your SELECT list |
| `Parser Error: syntax error` | SQL syntax error | Check query syntax, table/column names |
| `Binder Error: column does not exist` | Referenced column doesn't exist | Check column names, table structure |
| `Invalid Input Error: Data type mismatch` | Wrong data type for column | Convert data to correct type before insertion |

## API Reference

### TestResultsDBHandler

#### Constructor

```python
TestResultsDBHandler(db_path=None)
```

- `db_path`: Path to the DuckDB database file. If not provided, uses the `BENCHMARK_DB_PATH` environment variable or defaults to `./benchmark_db.duckdb`.

#### Methods

```python
store_test_result(test_result)
```
- `test_result`: Dictionary containing test result information.
- Returns: `True` if successful, `False` otherwise.

```python
generate_report(format='markdown', output_file=None)
```
- `format`: Report format (markdown, html, json).
- `output_file`: Path to save the report.
- Returns: Report content as string.

### duckdb_api/core/benchmark_db_query.py

#### Constructor

```python
BenchmarkDBQuery(db_path=None)
```

- `db_path`: Path to the DuckDB database file. If not provided, uses the `BENCHMARK_DB_PATH` environment variable or defaults to `./benchmark_db.duckdb`.

#### Methods

```python
get_models()
```
- Returns: DataFrame of models in the database.

```python
get_hardware_platforms()
```
- Returns: DataFrame of hardware platforms in the database.

```python
generate_performance_report(format='markdown', output=None, model=None)
```
- `format`: Report format (markdown, html, json, csv, chart).
- `output`: Path to save the report.
- `model`: Filter by model name.
- Returns: Report content.

```python
generate_compatibility_matrix(format='markdown', output=None, filter_family=None)
```
- `format`: Report format (markdown, html, json).
- `output`: Path to save the matrix.
- `filter_family`: Filter by model family.
- Returns: Matrix content.

## Advanced Usage Patterns

For specific use cases, the API provides additional capabilities:

### Query Optimization

For large databases with millions of test results, query optimization becomes important:

```python
# Create indexes for frequently queried columns
conn.execute("CREATE INDEX IF NOT EXISTS model_name_idx ON models(model_name)")
conn.execute("CREATE INDEX IF NOT EXISTS hardware_type_idx ON hardware_platforms(hardware_type)")
conn.execute("CREATE INDEX IF NOT EXISTS test_timestamp_idx ON test_results(timestamp)")

# Use prepared statements for repeated queries
stmt = conn.prepare("""
    SELECT 
        h.hardware_type,
        AVG(p.throughput_items_per_second) as avg_throughput
    FROM performance_results p
    JOIN models m ON p.model_id = m.model_id
    JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
    WHERE m.model_name = ?
    GROUP BY h.hardware_type
    ORDER BY avg_throughput DESC
""")

# Execute with different models
bert_results = stmt.execute(["bert-base-uncased"]).fetchdf()
t5_results = stmt.execute(["t5-small"]).fetchdf()

# Output:
#    hardware_type  avg_throughput
# 0          cuda         293.61
# 1           cpu         132.12
# 2          webgpu        98.45
# 3      qualcomm          78.23
```

### Time-Series Analysis

Track performance changes over time:

```python
# Track performance trend for a model across test runs
trend_data = conn.execute("""
    SELECT 
        DATE_TRUNC('day', p.test_timestamp) as test_date,
        h.hardware_type,
        AVG(p.throughput_items_per_second) as avg_throughput
    FROM performance_results p
    JOIN models m ON p.model_id = m.model_id
    JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
    WHERE m.model_name = 'bert-base-uncased'
    GROUP BY test_date, h.hardware_type
    ORDER BY test_date, h.hardware_type
""").fetchdf()

# Output:
#     test_date hardware_type  avg_throughput
# 0  2025-02-01         cuda         275.32
# 1  2025-02-01          cpu         124.56
# 2  2025-02-15         cuda         284.98
# 3  2025-02-15          cpu         128.34
# 4  2025-03-01         cuda         293.61
# 5  2025-03-01          cpu         132.12

# Detect performance regressions
regression_data = conn.execute("""
    WITH current AS (
        SELECT 
            m.model_name,
            h.hardware_type,
            AVG(p.throughput_items_per_second) as current_throughput
        FROM performance_results p
        JOIN models m ON p.model_id = m.model_id
        JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
        WHERE p.test_timestamp >= CURRENT_DATE - INTERVAL 7 DAY
        GROUP BY m.model_name, h.hardware_type
    ),
    previous AS (
        SELECT 
            m.model_name,
            h.hardware_type,
            AVG(p.throughput_items_per_second) as previous_throughput
        FROM performance_results p
        JOIN models m ON p.model_id = m.model_id
        JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
        WHERE p.test_timestamp BETWEEN CURRENT_DATE - INTERVAL 14 DAY AND CURRENT_DATE - INTERVAL 7 DAY
        GROUP BY m.model_name, h.hardware_type
    )
    SELECT 
        c.model_name,
        c.hardware_type,
        c.current_throughput,
        p.previous_throughput,
        (c.current_throughput - p.previous_throughput) / p.previous_throughput * 100 as percent_change
    FROM current c
    JOIN previous p ON c.model_name = p.model_name AND c.hardware_type = p.hardware_type
    WHERE ABS((c.current_throughput - p.previous_throughput) / p.previous_throughput * 100) > 5
    ORDER BY percent_change
""").fetchdf()

# Output:
#          model_name hardware_type  current_throughput  previous_throughput  percent_change
# 0  whisper-tiny           webgpu              76.23                85.67          -10.91
# 1  vit-base               qualcomm            45.12                42.34            6.57
# 2  bert-base-uncased      cuda               293.61               275.32            6.64
```

### Model Comparison

Compare performance across models and hardware:

```python
# Compare different models on the same hardware
model_comparison = conn.execute("""
    SELECT 
        m.model_name,
        m.model_family,
        h.hardware_type,
        AVG(p.throughput_items_per_second) as avg_throughput,
        AVG(p.memory_peak_mb) as avg_memory
    FROM performance_results p
    JOIN models m ON p.model_id = m.model_id
    JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
    WHERE h.hardware_type = 'cuda' AND p.batch_size = 1
    GROUP BY m.model_name, m.model_family, h.hardware_type
    ORDER BY avg_throughput DESC
""").fetchdf()

# Output:
#          model_name    model_family hardware_type  avg_throughput  avg_memory
# 0  bert-base-uncased text_embedding         cuda         293.61      3943.44
# 1  t5-small          text_generation       cuda         156.78      4291.22
# 2  whisper-tiny      audio                 cuda         122.45      3758.91
# 3  vit-base          vision                cuda         118.32      4102.67

# Compare hardware platforms for a specific model
hardware_comparison = conn.execute("""
    SELECT 
        h.hardware_type,
        h.device_name,
        AVG(p.throughput_items_per_second) as avg_throughput,
        AVG(p.memory_peak_mb) as avg_memory,
        AVG(p.power_watts) as avg_power
    FROM performance_results p
    JOIN models m ON p.model_id = m.model_id
    JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
    WHERE m.model_name = 'bert-base-uncased' AND p.batch_size = 1
    GROUP BY h.hardware_type, h.device_name
    ORDER BY avg_throughput DESC
""").fetchdf()

# Output:
#    hardware_type       device_name  avg_throughput  avg_memory  avg_power
# 0          cuda      NVIDIA A100          293.61      3943.44      245.12
# 1          rocm          AMD MI300        245.32      3821.56      235.45
# 2           cpu     Intel Xeon Gold       132.12      2874.40       95.23
# 3           mps   Apple M2 Ultra          123.45      2456.78       42.56
# 4      openvino      Intel NUC i7         105.67      2345.67       55.78
# 5      qualcomm  Snapdragon 8 Gen 3        78.23      1982.45       12.34
# 6        webgpu     Chrome Browser         65.43      2543.21        N/A
# 7        webnn       Edge Browser          58.76      2432.10        N/A
```

### Qualcomm-Specific Metrics

The database includes special metrics for Qualcomm hardware:

```python
# Get Qualcomm power metrics
qualcomm_power = conn.execute("""
    SELECT 
        m.model_name,
        pm.power_watts_avg,
        pm.power_watts_peak,
        pm.temperature_celsius_avg,
        pm.temperature_celsius_peak,
        pm.battery_impact_mah,
        pm.estimated_runtime_hours
    FROM power_metrics pm
    JOIN models m ON pm.model_id = m.model_id
    JOIN hardware_platforms h ON pm.hardware_id = h.hardware_id
    WHERE h.hardware_type = 'qualcomm'
    ORDER BY m.model_name
""").fetchdf()

# Output:
#          model_name  power_watts_avg  power_watts_peak  temperature_celsius_avg  temperature_celsius_peak  battery_impact_mah  estimated_runtime_hours
# 0  bert-base-uncased            3.24             5.67                    45.32                     52.76              125.43                     4.32
# 1  t5-small                     4.56             7.89                    48.76                     58.93              178.90                     3.54
# 2  vit-base                     3.87             6.34                    46.54                     55.87              148.65                     3.98
# 3  whisper-tiny                 4.12             7.23                    47.32                     57.45              162.34                     3.76

# Analyze power efficiency by model type
power_efficiency = conn.execute("""
    SELECT 
        m.model_family,
        AVG(p.energy_efficiency_items_per_joule) as avg_efficiency,
        AVG(pm.power_watts_avg) as avg_power,
        AVG(pm.estimated_runtime_hours) as avg_runtime
    FROM power_metrics pm
    JOIN performance_results p ON pm.model_id = p.model_id AND pm.hardware_id = p.hardware_id
    JOIN models m ON pm.model_id = m.model_id
    JOIN hardware_platforms h ON pm.hardware_id = h.hardware_id
    WHERE h.hardware_type = 'qualcomm'
    GROUP BY m.model_family
    ORDER BY avg_efficiency DESC
""").fetchdf()

# Output:
#      model_family  avg_efficiency  avg_power  avg_runtime
# 0  text_embedding          24.12       3.24         4.32
# 1  vision                  20.45       3.87         3.98
# 2  audio                   18.76       4.12         3.76
# 3  text_generation         15.34       4.56         3.54
```

### Web Platform Analytics

Analyze performance across web browsers:

```python
# Compare WebGPU performance across browsers
web_comparison = conn.execute("""
    SELECT 
        hp.device_name as browser,
        m.model_name,
        AVG(p.throughput_items_per_second) as avg_throughput,
        AVG(p.average_latency_ms) as avg_latency
    FROM performance_results p
    JOIN models m ON p.model_id = m.model_id
    JOIN hardware_platforms hp ON p.hardware_id = hp.hardware_id
    WHERE hp.hardware_type = 'webgpu'
    GROUP BY browser, m.model_name
    ORDER BY m.model_name, avg_throughput DESC
""").fetchdf()

# Output:
#           browser        model_name  avg_throughput  avg_latency
# 0  Chrome Browser  bert-base-uncased          65.43        15.28
# 1  Firefox Browser bert-base-uncased          61.76        16.19
# 2  Safari Browser  bert-base-uncased          54.32        18.41
# 3  Chrome Browser  vit-base                   48.76        20.51
# 4  Firefox Browser vit-base                   45.34        22.06
# 5  Safari Browser  vit-base                   40.21        24.87
# 6  Chrome Browser  whisper-tiny               42.34        23.62
# 7  Firefox Browser whisper-tiny               51.23        19.52
# 8  Safari Browser  whisper-tiny               35.67        28.04

# Analyze WebGPU optimization impact
webgpu_optimizations = conn.execute("""
    SELECT 
        hp.device_name as browser,
        tr.details->>'optimization_type' as optimization,
        AVG(p.throughput_items_per_second) as avg_throughput,
        AVG(p.average_latency_ms) as avg_latency,
        COUNT(*) as test_count
    FROM performance_results p
    JOIN test_results tr ON p.model_id = tr.model_id AND p.hardware_id = tr.hardware_id
    JOIN models m ON p.model_id = m.model_id
    JOIN hardware_platforms hp ON p.hardware_id = hp.hardware_id
    WHERE hp.hardware_type = 'webgpu' 
      AND tr.details->>'optimization_type' IS NOT NULL
    GROUP BY browser, optimization
    ORDER BY browser, avg_throughput DESC
""").fetchdf()

# Output:
#           browser           optimization  avg_throughput  avg_latency  test_count
# 0  Chrome Browser       shader_precompile          72.34        13.82          12
# 1  Chrome Browser     compute_shader_opt          69.87        14.32           8
# 2  Chrome Browser       parallel_loading          67.56        14.80          10
# 3  Chrome Browser                   none          65.43        15.28          15
# 4  Firefox Browser     compute_shader_opt          82.45        12.13           8
# 5  Firefox Browser       shader_precompile          67.23        14.87          12
# 6  Firefox Browser       parallel_loading          65.78        15.20          10
# 7  Firefox Browser                   none          61.76        16.19          15
# 8  Safari Browser       shader_precompile          58.93        16.97          12
# 9  Safari Browser       parallel_loading          56.78        17.61          10
# 10 Safari Browser                   none          54.32        18.41          15
# 11 Safari Browser     compute_shader_opt          42.34        23.62           8
```

## Performance Optimization

When working with large datasets, follow these tips for optimal performance:

### Database Optimization

```bash
# Run database optimization periodically
python scripts/benchmark_db_maintenance.py --optimize-db

# Output:
# Analyzing database: ./benchmark_db.duckdb
# Database size before optimization: 485.7 MB
# Running VACUUM...
# Running ANALYZE...
# Database size after optimization: 432.3 MB
# Optimization complete: 11.0% reduction in size
```

### Query Optimization

For large datasets, optimize your queries:

1. **Use indexes for frequent lookups**:
   ```sql
   CREATE INDEX IF NOT EXISTS model_name_idx ON models(model_name);
   CREATE INDEX IF NOT EXISTS hardware_platform_idx ON hardware_platforms(hardware_type);
   CREATE INDEX IF NOT EXISTS test_timestamp_idx ON test_results(timestamp);
   ```

2. **Limit result sets** when possible:
   ```sql
   SELECT * FROM test_results 
   ORDER BY timestamp DESC 
   LIMIT 100;
   ```

3. **Use column selection** instead of `SELECT *`:
   ```sql
   -- Instead of:
   SELECT * FROM performance_results;
   
   -- Use:
   SELECT model_id, hardware_id, throughput_items_per_second 
   FROM performance_results;
   ```

4. **Perform aggregation in the database** rather than in Python:
   ```sql
   -- Let the database handle aggregation
   SELECT 
       model_id, 
       AVG(throughput_items_per_second) as avg_throughput
   FROM performance_results
   GROUP BY model_id;
   ```

### Data Partitioning

For very large datasets (billions of rows), consider partitioning:

```python
# Create monthly partitions for test results
for month in range(1, 13):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS test_results_{2025}_{month:02d} (
            LIKE test_results INCLUDING ALL
        )
    """)
    
# Function to insert into correct partition
def store_test_result_partitioned(result, timestamp):
    year = timestamp.year
    month = timestamp.month
    table_name = f"test_results_{year}_{month:02d}"
    
    # Check if partition exists
    exists = conn.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'").fetchone()[0]
    if not exists:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                LIKE test_results INCLUDING ALL
            )
        """)
    
    # Insert into partition
    conn.execute(f"""
        INSERT INTO {table_name} (...)
        VALUES (...)
    """)
```

## Conclusion

The database API provides a powerful interface for interacting with the IPFS Accelerate test results database. By following the patterns and best practices outlined in this document, you can efficiently store, query, and analyze test results to gain insights into hardware compatibility and performance across different platforms and models.

With the optimizations and advanced query techniques described above, the database can scale to handle millions of test results while maintaining fast query performance and low storage requirements. The comprehensive schema design supports a wide range of analysis scenarios, from simple performance comparisons to advanced time-series analysis and power efficiency monitoring for mobile devices.