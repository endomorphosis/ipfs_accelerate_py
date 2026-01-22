# Archived Files Reference Guide

## Overview

This document provides a reference for all files and directories that have been archived as of March 5, 2025. As part of the database migration and project cleanup efforts, we've archived several categories of files that are no longer actively used but are retained for historical reference.

## Archiving Status

The archiving process has been completed with the following results:

- **Process completion date**: March 5, 2025
- **Migration status**: 100% complete
- **Database conversion**: All JSON files migrated to DuckDB
- **Archive location**: `/home/barberb/ipfs_accelerate_py/test/archived_json_files/`

## Archived File Categories

### 1. JSON Benchmark and Test Result Files

All JSON benchmark and test result files have been migrated to the DuckDB database system and archived:

```
archived_json_files/
  - api_check_results_20250305.tar.gz
  - archived_test_results_20250305.tar.gz
  - benchmark_results_20250305.tar.gz
  - critical_model_results_20250305.tar.gz
  - hardware_fix_results_20250305.tar.gz
```

These archives contain:
- Performance test results
- Hardware compatibility test results
- Integration test results
- API implementation checks
- Model functionality tests

### 2. Backup Files and Legacy Versions

These files represent previous iterations of development that are no longer needed:

- Backup (.bak) files for major script files:
  - `generators/test_generators/merged_test_generator.py.bak*`
  - `fixed_generators/test_generators/merged_test_generator.py.bak*`
  - `generators/skill_generators/integrated_skillset_generator.py.bak*`
  - `fix_generator_hardware_support.py.bak*`
  - `template_generators/hardware/hardware_detection.py.bak*`
  - All `.bak_json_dep` files (created during JSON deprecation)

- Old versions of generators that have been replaced by improved versions:
  - Legacy versions of various test generators
  - Superseded implementation files

### 3. Archived Directories

The following directories already contain archived files:

- `/archived_cuda_fixes/`: Old CUDA detection fixes
- `/archived_md_files/`: Superseded documentation
- `/archived_reports/`: Old API implementation reports
- `/archived_scripts/`: Replaced scripts
- `/archived_test_files/`: Old test outputs
- `/archived_test_results/`: Legacy test results

### 4. Deprecated Legacy Scripts

These scripts have been explicitly deprecated and replaced:

- `benchmark_database.py` → Replaced by `duckdb_api/core/benchmark_db_api.py`
- `benchmark_query.py` → Replaced by `duckdb_api/core/benchmark_db_query.py`
- `test_model_benchmarks.py` → Replaced by integrated tests

## Accessing Archived Files

If you need to access archived files:

1. For tar.gz archives:
   ```bash
   tar -xzf archived_json_files/benchmark_results_20250305.tar.gz -C /tmp/extract_dir
   ```

2. For individually archived files:
   - Check the relevant `/archived_*` directory

## Impact on Current Development

The archiving of these files has no impact on current development:

- All data is preserved in the DuckDB database system
- Current tools use `DEPRECATE_JSON_OUTPUT=1` by default
- All benchmark and test runners have been updated to use the database
- Complete documentation is available in:
  - `DATABASE_MIGRATION_GUIDE.md`
  - `BENCHMARK_DATABASE_GUIDE.md`
  - `DATABASE_MIGRATION_STATUS.md`

## Future Cleanup Considerations

For future cleanup operations, consider:

1. Consolidating the `/archived_*` directories further
2. Compressing individual backup (.bak) files
3. Setting up automatic cleanup of older archives beyond a retention period

## Conclusion

The archiving of deprecated files marks the successful completion of the database migration project. All relevant data has been migrated to DuckDB, and the older files have been preserved for reference while reducing clutter in the active codebase.