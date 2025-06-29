# Code Reorganization Migration Guide - March 9, 2025

## Overview

This guide documents the major code reorganization that took place in March 2025, moving files from the `/test/` directory into more organized subdirectories. The reorganization aims to improve code maintainability, make the directory structure more intuitive, and separate different functional components of the framework.

## Directory Structure Changes

### Before Reorganization

Prior to March 2025, most code was housed in the `/test/` directory, which contained a mix of different types of files:

```
/test/
├── benchmark_*.py          # Benchmark-related files
├── generators/test_generators/merged_test_generator.py # Test generator
├── fixed_generators/test_generators/merged_test_generator.py # Fixed test generator
├── generators/skill_generators/integrated_skillset_generator.py # Skillset generator
└── ... (many other files)  # Various test, benchmark, and utility files
```

### After Reorganization

After the March 2025 reorganization, files were moved to the following directory structure:

```
/
├── generators/             # All generator-related code (216 files)
│   ├── benchmark_generators/ # Benchmark generation tools
│   ├── models/             # Model implementations and skill files
│   ├── runners/            # Test runner scripts
│   ├── skill_generators/   # Skill generation tools
│   ├── template_generators/ # Template generation utilities
│   ├── templates/          # Template files and template system
│   ├── test_generators/    # Test generation tools
│   └── utils/              # Utility functions
├── duckdb_api/             # All database-related code (83 files)
│   ├── core/               # Core database functionality
│   ├── migration/          # Migration tools for JSON to database
│   ├── schema/             # Database schema definitions
│   ├── utils/              # Utility functions for database operations
│   └── visualization/      # Result visualization tools
├── fixed_web_platform/     # Web platform implementation components
├── predictive_performance/ # ML-based performance prediction system
└── test/                   # Remaining test files and documentation
```

## Key Components Moved

### Generators (moved to `/generators/`)

- **Test Generators**:
  - `generators/test_generators/merged_test_generator.py` → `generators/generators/test_generators/merged_test_generator.py`
  - `fixed_generators/test_generators/merged_test_generator.py` → `generators/test_generators/fixed_generators/test_generators/merged_test_generator.py`
  - `generators/test_generators/simple_test_generator.py` → `generators/test_generators/generators/test_generators/simple_test_generator.py`
  - `generate_modality_tests.py` → `generators/generate_modality_tests.py`

- **Skill Generators**:
  - `generators/skill_generators/integrated_skillset_generator.py` → `generators/models/generators/skill_generators/integrated_skillset_generator.py`
  - `skillset_generator.py` → `generators/skill_generators/skillset_generator.py`

- **Template System**:
  - `template_database.py` → `generators/templates/template_database.py`
  - `template_validator.py` → `generators/templates/template_validator.py`
  - `template_inheritance_system.py` → `generators/templates/template_inheritance_system.py`

- **Web Platform Generators**:
  - `run_real_web_benchmarks.py` → `generators/run_real_web_benchmarks.py`
  - `run_real_webgpu_webnn.py` → `generators/run_real_webgpu_webnn.py`
  - `check_browser_webnn_webgpu.py` → `generators/check_browser_webnn_webgpu.py`

### Database Tools (moved to `/duckdb_api/`)

- **Core Database Functions**:
  - `duckdb_api/core/benchmark_db_api.py` → `duckdb_api/core/duckdb_api/core/benchmark_db_api.py`
  - `benchmark_db_updater.py` → `duckdb_api/core/benchmark_db_updater.py`
  - `duckdb_api/core/benchmark_db_query.py` → `duckdb_api/core/duckdb_api/core/benchmark_db_query.py`
  - `duckdb_api/core/benchmark_db_maintenance.py` → `duckdb_api/core/duckdb_api/core/benchmark_db_maintenance.py`

- **Visualization**:
  - `benchmark_db_visualizer.py` → `duckdb_api/visualization/benchmark_db_visualizer.py`
  - `benchmark_timing_report.py` → `duckdb_api/visualization/benchmark_timing_report.py`

- **Schema Management**:
  - `create_benchmark_schema.py` → `duckdb_api/schema/create_benchmark_schema.py`
  - `update_db_schema_for_simulation.py` → `duckdb_api/schema/update_db_schema_for_simulation.py`

- **Migration Tools**:
  - `benchmark_db_converter.py` → `duckdb_api/migration/benchmark_db_converter.py`
  - `benchmark_db_migration.py` → `duckdb_api/migration/benchmark_db_migration.py`
  - `migrate_all_json_files.py` → `duckdb_api/migration/migrate_all_json_files.py`

## Updating Your Code

### Import Statements

When importing modules, you'll need to update your import statements to reflect the new directory structure:

```python
# Old imports
from merged_test_generator import MergedTestGenerator
from duckdb_api.core.benchmark_db_query import query_database

# New imports
from generators.merged_test_generator import MergedTestGenerator
from duckdb_api.core.benchmark_db_query import query_database
```

### Running Files

When running files, update your commands to use the new paths:

```bash
# Old commands
python generators/generators/test_generators/merged_test_generator.py --generate bert

# New commands
python generators/generators/test_generators/merged_test_generator.py --generate bert
```

### Backward Compatibility

To maintain backward compatibility, import redirection has been implemented in some modules:

```python
# In test/generators/test_generators/merged_test_generator.py
import sys
import warnings

warnings.warn(
    "This module has been moved to generators/generators/test_generators/merged_test_generator.py. "
    "Please update your imports.",
    DeprecationWarning
)

# Import from the new location
from generators.merged_test_generator import *
```

This allows old imports to continue working, but they will generate deprecation warnings.

## Documentation Updates

All documentation has been updated to reflect the new directory structure. If you find any references to old paths, please report them or submit a pull request to update them.

Key documentation files that have been updated include:

1. README.md
2. CLAUDE.md
3. WEB_PLATFORM_INTEGRATION_GUIDE.md
4. BENCHMARK_TIMING_REPORT_GUIDE.md
5. DATABASE_MIGRATION_GUIDE.md

## Future Migration Plans

As noted in CLAUDE.md, there are upcoming migrations planned:

> **UPCOMING MIGRATION (Q2-Q3 2025):**
> 
> All WebGPU/WebNN implementations will be moved from `/fixed_web_platform/` to a dedicated `ipfs_accelerate_js` folder once all tests pass. This migration will create a clearer separation between JavaScript-based components and Python-based components.

## Issues and Support

If you encounter any issues with the reorganized codebase:

1. Check this migration guide for information on where files have been moved
2. Look for deprecation warnings that indicate new import locations
3. Refer to updated documentation for examples of using the new directory structure
4. Create an issue if you find missing or incorrectly moved files

## Conclusion

This reorganization significantly improves the maintainability and organization of the codebase by properly separating different components into logical directories. While it may require some initial updates to your code, the benefits in terms of organization and clarity are substantial.

The migration was completed successfully on March 9, 2025, with 299 files moved and all import paths updated.