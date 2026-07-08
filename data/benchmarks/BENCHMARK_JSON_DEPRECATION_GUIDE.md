# Benchmark Results Storage: DuckDB Migration Guide

## IMPORTANT: JSON Output is Deprecated

As of March 6, 2025, all benchmark results should be written directly to the DuckDB database instead of JSON files in the `benchmark_results` directory. The environment variable `DEPRECATE_JSON_OUTPUT=1` is now set as the default for all scripts.

## Why This Change?

The migration from JSON files to DuckDB provides several key benefits:

1. **Structured Data Schema**: Consistent data format with proper relationships
2. **Efficient Storage**: 50-80% reduction in disk space compared to JSON
3. **Fast Queries**: SQL-based querying for rapid analysis
4. **Data Consistency**: Enforced schema prevents inconsistent data formats
5. **Advanced Analysis**: Support for complex queries and aggregations
6. **Visualization**: Integrated tools for generating charts and reports
7. **CI/CD Integration**: Seamless integration with testing pipelines

## How to Comply with This Change

### For Script Users

When running benchmark scripts:

```bash
# Set the database path environment variable
export BENCHMARK_DB_PATH=./benchmark_db.duckdb

# Run benchmarks (results stored directly in database)
python test/benchmark_all_key_models.py --db-only

# If you need JSON output files (for backward compatibility only):
# Files should be saved to benchmark_results directory (not a new folder)
# and overwrite existing files each time
python test/benchmark_all_key_models.py --output-dir ./benchmark_results

# Files will be automatically cleaned up from the repository after successful runs
```

### For Developers

When writing code that deals with benchmark results:

```python
# RECOMMENDED: Store directly in database
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
api = BenchmarkDBAPI()
api.store_performance_result(
    model_name="bert-base-uncased",
    hardware_type="cuda",
    throughput=125.7,
    latency_avg=8.2
)

# NOT RECOMMENDED: Writing to JSON files (deprecated)
# Replace this pattern:
# with open("benchmark_results/result.json", "w") as f:
#     json.dump(results, f, indent=2)
```

### For Existing Code

When updating existing code:

1. Add the environment variable check for JSON deprecation:
   ```python
   # Always deprecate JSON output in favor of DuckDB
   DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")
   ```

2. Wrap any JSON writing code in a conditional:
   ```python
   if not DEPRECATE_JSON_OUTPUT:
       # Legacy JSON writing code (will not execute by default)
       with open(results_file, 'w') as f:
           json.dump(self.results, f, indent=2)
   else:
       logger.info("JSON output is deprecated. Results stored directly in database.")
   ```

3. Add database storage code:
   ```python
   try:
       from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
       db_api = BenchmarkDBAPI()
       db_api.store_performance_result(**results)
   except ImportError:
       logger.warning("benchmark_db_api not available. Results may not be stored properly.")
   ```

## Finding Legacy JSON Output

To find any remaining code that writes to JSON files:

```bash
# Look for code that writes to benchmark_results directory
grep -r "benchmark_results.*json.dump" --include="*.py" ./

# Look for code that writes JSON without using the database
grep -r "json.dump" --include="*.py" ./ | grep -v "DEPRECATE_JSON_OUTPUT"
```

## Migration Status

The JSON deprecation script (`deprecate_json_benchmarks.py`) has been used to update the following critical files:

- benchmark_all_key_models.py
- run_model_benchmarks.py
- hardware_benchmark_runner.py
- benchmark_db_api.py
- web_platform_benchmark.py
- web_platform_testing.py
- test_webgpu_ultra_low_precision.py
- run_benchmark_with_db.py
- benchmark_hardware_performance.py
- model_benchmark_runner.py
- training_benchmark_runner.py
- distributed_training_benchmark.py
- continuous_hardware_benchmarking.py

If you maintain or work with any of these files or other benchmark-related scripts, please ensure your code follows the guidelines in this document.

## Documentation Updates

The following documentation files now reflect this change:

1. [BENCHMARK_DATABASE_GUIDE.md](/home/barberb/ipfs_accelerate_py/test/BENCHMARK_DATABASE_GUIDE.md) - Complete guide to database usage
2. [DATABASE_MIGRATION_GUIDE.md](/home/barberb/ipfs_accelerate_py/test/DATABASE_MIGRATION_GUIDE.md) - Details on the migration process
3. [README.md](/home/barberb/ipfs_accelerate_py/test/README.md) - Important notice about benchmark storage
4. [CLAUDE.md](/home/barberb/ipfs_accelerate_py/test/CLAUDE.md) - Updated benchmark command examples

## Support

If you have questions about the database system or need help migrating your code, refer to:

- [BENCHMARK_DATABASE_GUIDE.md](/home/barberb/ipfs_accelerate_py/test/BENCHMARK_DATABASE_GUIDE.md) - Comprehensive database documentation
- [DATABASE_MIGRATION_GUIDE.md](/home/barberb/ipfs_accelerate_py/test/DATABASE_MIGRATION_GUIDE.md) - Migration process details