# Benchmark Timing Report Guide Updates

## Overview

This file documents the updates made to the BENCHMARK_TIMING_REPORT_GUIDE.md to reflect the reorganization of the codebase.

## Path Updates

The following path changes were made to reflect the new directory structure:

### Scripts moved to duckdb_api/ directory

1. **run_comprehensive_benchmarks.py**
   - `python test/run_comprehensive_benchmarks.py` → `python duckdb_api/run_comprehensive_benchmarks.py`

2. **run_benchmark_timing_report.py**
   - `python run_benchmark_timing_report.py` → `python duckdb_api/run_benchmark_timing_report.py`

3. **query_benchmark_timings.py**
   - `python query_benchmark_timings.py` → `python duckdb_api/query_benchmark_timings.py`

4. **benchmark_db_query.py**
   - `python benchmark_db_query.py` → `python duckdb_api/benchmark_db_query.py`

## Script Usage Examples Updated

All examples throughout the BENCHMARK_TIMING_REPORT_GUIDE.md have been updated to reflect the new directory structure:

```bash
# Old path
python test/run_comprehensive_benchmarks.py --models bert,t5,vit --hardware cpu,cuda

# New path
python duckdb_api/run_comprehensive_benchmarks.py --models bert,t5,vit --hardware cpu,cuda
```

## GitHub Actions Workflow Example

The GitHub Actions workflow example has been updated to use the new path:

```yaml
# Old path in workflow
python test/run_benchmark_timing_report.py --generate --format html --output benchmark_report.html

# New path in workflow
python duckdb_api/run_benchmark_timing_report.py --generate --format html --output benchmark_report.html
```

## Next Steps

1. Continue updating remaining documentation to reflect the new directory structure
2. Ensure all Python import statements in the relocated files are updated
3. Update any CI/CD pipelines to reference the new file locations
4. Consider creating a README in the duckdb_api/ directory to explain its purpose and contents