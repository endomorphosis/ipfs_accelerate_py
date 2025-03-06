# IPFS Accelerate Testing with DuckDB Integration

## Overview

The test system for IPFS Accelerate Python has been enhanced with DuckDB integration, allowing test results to be efficiently stored and analyzed in a structured database rather than relying on JSON files. This document provides details on how to use the database integration with the test framework.

## Implementation Details

### Database Integration Architecture

The implementation integrates IPFS Accelerate testing with DuckDB through the following components:

1. **TestResultsDBHandler**: A class that manages database connections and operations
2. **Schema Definition**: A structured database schema for test results and metrics
3. **Result Storage Logic**: Code that converts test results to database records
4. **Query Tools**: Utilities for extracting and analyzing test data

### Test Integration

The test_ipfs_accelerate.py script has been updated to:
1. Connect to a DuckDB database specified by environment variable
2. Create necessary tables if they don't exist
3. Store test results directly in the database
4. Generate reports from database data

### Schema Structure

The database includes the following key tables:

- `ipfs_test_results`: Stores basic test results and status
- `ipfs_performance_metrics`: Contains detailed performance measurements
- `ipfs_backend_operations`: Logs container and backend operations
- `ipfs_configuration_tests`: Tracks configuration management tests
- `ipfs_hardware_compatibility`: Records hardware compatibility information

## Running Tests with Database Integration

### Basic Usage

```bash
# Set the database path environment variable
export BENCHMARK_DB_PATH=./benchmark_db.duckdb

# Run the test script with database integration
python test_ipfs_accelerate.py

# Run with specific options
python test_ipfs_accelerate.py --db-path ./custom_db.duckdb --report
```

### Environment Variables

The following environment variables control database integration behavior:

- `BENCHMARK_DB_PATH`: Path to the DuckDB database file (default: ./benchmark_db.duckdb)
- `DEPRECATE_JSON_OUTPUT`: Set to 1 to disable JSON output (default: 0)
- `DISABLE_DB_STORAGE`: Set to 1 to disable database storage (default: 0)

### Command Line Options

The script supports several database-related command-line options:

```bash
# Specify database path directly
python test_ipfs_accelerate.py --db-path ./custom_db.duckdb

# Generate a report from the database
python test_ipfs_accelerate.py --report --format markdown --output test_report.md

# Run tests without storing results in the database
python test_ipfs_accelerate.py --no-db-store

# Run tests with both JSON and database output
python test_ipfs_accelerate.py --json-output --db-path ./benchmark_db.duckdb
```

## Database Code Implementation

The TestResultsDBHandler implementation manages all database operations:

```python
class TestResultsDBHandler:
    def __init__(self, db_path=None):
        # Use environment variable or default path
        self.db_path = db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        self._connect()
        
    def _connect(self):
        self.con = duckdb.connect(self.db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create all necessary tables if they don't exist"""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ipfs_test_results (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                test_name VARCHAR,
                status VARCHAR,
                execution_time FLOAT,
                error_message VARCHAR,
                details JSON
            )
        """)
        
        # Create other tables...
        
    def store_test_result(self, test_result):
        """Store a test result in the database"""
        self.con.execute("""
            INSERT INTO ipfs_test_results (
                timestamp, test_name, status, execution_time, error_message, details
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, ?, ?, ?
            )
        """, (
            test_result["name"],
            test_result["status"],
            test_result.get("execution_time", 0),
            test_result.get("error", ""),
            json.dumps(test_result.get("details", {}))
        ))
        
    def get_test_results(self, limit=100):
        """Retrieve test results from the database"""
        return self.con.execute("""
            SELECT * FROM ipfs_test_results
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
    def generate_report(self, format="markdown"):
        """Generate a report from test results"""
        # Implementation details...
```

The main test script integrates with this handler:

```python
# Initialize database handler
db_handler = None
if not args.no_db_store:
    try:
        import duckdb
        db_handler = TestResultsDBHandler(args.db_path)
    except ImportError:
        print("DuckDB not installed. Database storage disabled.")
        
# Run tests and store results
for test in tests:
    result = run_test(test)
    
    # Store in database if enabled
    if db_handler and not args.no_db_store:
        db_handler.store_test_result(result)
    
    # Store as JSON if enabled
    if not os.environ.get("DEPRECATE_JSON_OUTPUT", "0") == "1" or args.json_output:
        save_json_result(result)
```

## Analyzing Results

### Basic SQL Queries

You can analyze test results using SQL queries:

```bash
# Open DuckDB CLI
duckdb ./benchmark_db.duckdb

# Run a query
SELECT test_name, status, execution_time 
FROM ipfs_test_results 
ORDER BY timestamp DESC 
LIMIT 10;
```

### Using Python

You can also analyze results programmatically:

```python
import duckdb

# Connect to database
con = duckdb.connect('./benchmark_db.duckdb')

# Get average execution time by test
results = con.execute("""
    SELECT 
        test_name, 
        COUNT(*) as runs,
        AVG(execution_time) as avg_time,
        MIN(execution_time) as min_time,
        MAX(execution_time) as max_time
    FROM ipfs_test_results
    GROUP BY test_name
    ORDER BY avg_time DESC
""").fetchdf()

print(results)
```

### Using Report Generator

The script includes a report generator:

```bash
# Generate a markdown report
python test_ipfs_accelerate.py --report --format markdown --output test_report.md

# Generate an HTML report
python test_ipfs_accelerate.py --report --format html --output test_report.html
```

## Example Queries

Here are some useful queries for analyzing test results:

### Basic Test Status

```sql
-- Get status counts for each test
SELECT 
    test_name, 
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
    SUM(CASE WHEN status = 'failure' THEN 1 ELSE 0 END) as failures,
    ROUND(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate
FROM ipfs_test_results
GROUP BY test_name
ORDER BY success_rate DESC
```

### Performance Analysis

```sql
-- Analyze performance by test type
SELECT 
    test_name,
    AVG(execution_time) as avg_time,
    MIN(execution_time) as min_time,
    MAX(execution_time) as max_time,
    STDDEV(execution_time) as std_dev
FROM ipfs_test_results
WHERE status = 'success'
GROUP BY test_name
ORDER BY avg_time DESC
```

### Error Analysis

```sql
-- Find common error messages
SELECT 
    error_message,
    COUNT(*) as occurrences,
    array_agg(test_name) as affected_tests
FROM ipfs_test_results
WHERE status = 'failure' AND error_message != ''
GROUP BY error_message
ORDER BY occurrences DESC
```

## Migration from JSON

If you have existing JSON test results, you can migrate them to the database:

```bash
# Run the migration script
python migrate_test_results.py --input-dir ./test_results --db-path ./benchmark_db.duckdb
```

## Future Enhancements

1. **Real-time Dashboard**: Develop a web dashboard for monitoring test results
2. **Historical Trend Analysis**: Implement tools for analyzing performance over time
3. **Automated Regression Detection**: Create alerts for performance regressions
4. **CI/CD Integration**: Automatically store test results from CI/CD pipelines
5. **Cross-Test Analysis**: Enable comparison across different test types and configurations

## Resources

For more information on DuckDB integration, see:

- [BENCHMARK_DATABASE_GUIDE.md](./BENCHMARK_DATABASE_GUIDE.md) - Detailed database schema documentation
- [DATABASE_MIGRATION_GUIDE.md](./DATABASE_MIGRATION_GUIDE.md) - Guide for migrating from JSON to DuckDB
- [IPFS_ACCELERATE_SUMMARY.md](./IPFS_ACCELERATE_SUMMARY.md) - Overview of the IPFS Accelerate package