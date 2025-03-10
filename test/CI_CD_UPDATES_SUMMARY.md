# CI/CD Integration with New Directory Structure

## Overview

This document provides guidance on how to update and use the CI/CD workflows with the new directory structure that has been implemented as part of the March 2025 reorganization. The new structure moves files from the `/test` directory into dedicated packages for better organization and maintainability.

## New Directory Structure

The codebase has been reorganized with the following top-level structure:

- **generators/** - Generator code (formerly in test/)
  - **generators/test_generators/** - Test generation tools
  - **generators/utils/** - Utility functions
  - **generators/models/** - Model implementations
  - **generators/hardware/** - Hardware detection tools
  - Other generator subdirectories

- **duckdb_api/** - Database code (formerly in test/)
  - **duckdb_api/core/** - Core database functionality
  - **duckdb_api/migration/** - Migration tools
  - **duckdb_api/schema/** - Schema definitions
  - **duckdb_api/visualization/** - Visualization tools
  - Other database-related subdirectories

## Updated CI/CD Workflows

The primary CI/CD workflows have been updated to use the new directory structure. Here are the key changes:

### 1. `test_and_benchmark.yml`

This workflow runs tests and benchmarks on a regular schedule and on various triggers.

**Updated file paths**:
```yaml
# Old
python test/models/test_ipfs_accelerate.py --models $MODELS --endpoints $HARDWARE

# New
python generators/models/test_ipfs_accelerate.py --models $MODELS --endpoints $HARDWARE
```

```yaml
# Old
python test/benchmark_db_query.py --format markdown --output compatibility_matrix.md

# New
python duckdb_api/visualization/generate_compatibility_matrix.py --format markdown --output compatibility_matrix.md
```

### 2. `benchmark_db_ci.yml`

This workflow runs benchmarks and stores results in the database.

**Updated file paths**:
```yaml
# Old
python test/run_benchmark_with_db.py --db $DB_FILE --model ${{ matrix.model }}

# New
python duckdb_api/core/run_benchmark_with_db.py --db $DB_FILE --model ${{ matrix.model }}
```

```yaml
# Old
python test/ci_benchmark_integrator.py --artifacts-dir ./artifacts

# New
python duckdb_api/scripts/ci_benchmark_integrator.py --artifacts-dir ./artifacts
```

## Triggering Workflows

The workflows have been updated to trigger based on changes in the new directory structure:

```yaml
on:
  push:
    branches: [ main ]
    paths:
      - 'generators/**'
      - 'duckdb_api/**'
      - 'fixed_web_platform/**'
      - '.github/workflows/test_and_benchmark.yml'
```

## Environment Variables

Environment variables have been updated to reflect the new structure:

```yaml
env:
  BENCHMARK_DB_PATH: ./benchmark_db.duckdb
  DEPRECATE_JSON_OUTPUT: 1
  PYTHONPATH: ${{ github.workspace }}
```

The `PYTHONPATH` environment variable ensures that imports from the new package structure work correctly.

## Dependencies Installation

Dependencies are now installed from updated requirements files:

```yaml
pip install -r duckdb_api/scripts/requirements_db.txt
pip install -r requirements.txt
```

## Running CI/CD Tasks Locally

To run CI/CD tasks locally with the new directory structure:

1. **Run Tests**:
```bash
# Single model test
python generators/models/test_ipfs_accelerate.py --models BAAI/bge-small-en-v1.5 --endpoints cpu

# Multiple models test
python generators/models/test_ipfs_accelerate.py --models BAAI/bge-small-en-v1.5,prajjwal1/bert-tiny --endpoints cpu,cuda
```

2. **Run Benchmarks**:
```bash
# With database integration
python duckdb_api/core/run_benchmark_with_db.py --models BAAI/bge-small-en-v1.5 --hardware cpu

# Generate reports
python duckdb_api/visualization/generate_compatibility_matrix.py --format markdown --output compatibility_matrix.md
```

3. **Generate Reports**:
```bash
# Generate test report
python generators/models/test_ipfs_accelerate.py --report --format markdown --output test_report.md

# Generate database report
python duckdb_api/core/benchmark_db_query.py --report performance --format html --output performance_report.html
```

## GitHub Actions Manual Triggers

The workflows support manual triggers with the following inputs:

1. **test_and_benchmark.yml**:
   - `models`: Comma-separated list of models to test (default: "BAAI/bge-small-en-v1.5,prajjwal1/bert-tiny")
   - `hardware`: Comma-separated list of hardware to test (default: "cpu,cuda")
   - `report_format`: Report format (options: markdown, html, json)

2. **benchmark_db_ci.yml**:
   - `test_model`: Model to test (default: "all")
   - `hardware`: Hardware to test on (options: cpu, cuda, all)
   - `batch_size`: Batch sizes to test (default: "1,2,4,8")

To manually trigger a workflow:
1. Go to the "Actions" tab in the GitHub repository
2. Select the workflow you want to run
3. Click "Run workflow" and enter the desired parameters
4. Click "Run workflow" to start the job

## Adding New CI/CD Features

When adding new CI/CD features, follow these guidelines:

1. **File Paths**: Use the new directory structure for all file paths
2. **Imports**: Use absolute imports with the new package structure
3. **Environment Variables**: Set `PYTHONPATH` to ensure imports work correctly
4. **Requirements**: Update dependency installation to use the appropriate requirements files

Example for adding a new CI job:

```yaml
run_new_feature:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r duckdb_api/scripts/requirements_db.txt
        pip install -r requirements.txt
        
    - name: Run new feature
      run: |
        python generators/new_feature/run_feature.py --option value
```

## Troubleshooting

If you encounter issues with the CI/CD workflows after the directory restructuring:

1. **Import Errors**: Check that the import paths use the new package structure
2. **File Not Found**: Ensure files are referenced using the new paths
3. **Environment Variables**: Verify that `PYTHONPATH` is set correctly
4. **Testing Locally**: Test the workflow steps locally before pushing

If a file is still in the old location, you can use the migration scripts:

```bash
# Move a file to the new structure
python test/move_files_to_packages.py --file old_path.py --type [generator|database]

# Update imports
python test/update_imports.py
```

## Additional Resources

- See [MIGRATION_REPORT.md](MIGRATION_REPORT.md) for details on the migration process
- See [CLAUDE.md](CLAUDE.md) for the full directory structure and organization update

---

Generated: March 10, 2025