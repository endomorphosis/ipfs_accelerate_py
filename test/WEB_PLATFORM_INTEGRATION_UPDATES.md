# Web Platform Integration Guide Updates

## Overview

This file documents the updates made to the WEB_PLATFORM_INTEGRATION_GUIDE.md to reflect the recent reorganization of the codebase.

## Path Updates

The following path changes were made to reflect the new directory structure:

### Python scripts moved to scripts/generators/
- `python scripts/generators/test_scripts/generators/merged_test_generator.py` → `python scripts/generators/scripts/generators/test_scripts/generators/merged_test_generator.py`
- `python fix_test_generator.py` → `python scripts/generators/fix_test_generator.py`

### Python scripts moved to duckdb_api/
- `python run_model_benchmarks.py` → `python duckdb_api/run_model_benchmarks.py`

## Affected Commands

The following commands were updated:

1. Generator commands:
   ```bash
   # Old
   python scripts/generators/test_scripts/generators/merged_test_generator.py --generate bert --web-platform webnn --real-implementation
   
   # New
   python scripts/generators/scripts/generators/test_scripts/generators/merged_test_generator.py --generate bert --web-platform webnn --real-implementation
   ```

2. Database-related commands:
   ```bash
   # Old
   ./run_web_platform_tests.sh python scripts/generators/run_model_benchmarks.py --hardware webnn
   
   # New
   ./run_web_platform_tests.sh python duckdb_api/run_model_benchmarks.py --hardware webnn
   ```

3. Fix commands:
   ```bash
   # Old
   python fix_test_generator.py --fix-all
   
   # New
   python scripts/generators/fix_test_generator.py --fix-all
   ```

## Next Steps

1. Verify that all paths in the guide have been updated correctly
2. Test the commands to ensure they work with the new paths
3. Update additional documentation files that reference these paths
4. Consider creating a README file in the scripts/generators/ directory to explain its purpose and contents
5. Create similar documentation updates for other affected files