# End-to-End Testing Framework: User Guide

This guide provides practical instructions for using the End-to-End Testing Framework to test AI models across different hardware platforms.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Working with Hardware](#working-with-hardware)
5. [Result Validation](#result-validation)
6. [Database Integration](#database-integration)
7. [Documentation Generation](#documentation-generation)
8. [CI/CD Integration](#ci-cd-integration)
9. [Maintenance Tasks](#maintenance-tasks)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

Before using the framework, ensure you have the following installed:

```bash
# Install required packages
pip install torch duckdb selenium gitpython

# Optional packages for specific hardware platforms
pip install openvino  # For OpenVINO support
```

### Running Your First Test

To run a simple test for a model on CPU hardware:

```bash
python run_e2e_tests.py --model bert-base-uncased --hardware cpu
```

This will:
- Generate skill, test, and benchmark files for the model
- Run the test and collect results
- Compare results with expected results (creating baseline if none exist)
- Store results in the collected_results directory

## Basic Usage

### Testing Models

```bash
# Test a specific model
python run_e2e_tests.py --model bert-base-uncased --hardware cpu

# Test multiple models
python run_e2e_tests.py --model bert-base-uncased,t5-small --hardware cpu

# Test all models in a family
python run_e2e_tests.py --model-family text-embedding --hardware cpu

# Test all supported models
python run_e2e_tests.py --all-models --hardware cpu
```

### Hardware Options

```bash
# Test on specific hardware
python run_e2e_tests.py --model bert-base-uncased --hardware cuda

# Test on multiple hardware platforms
python run_e2e_tests.py --model bert-base-uncased --hardware cpu,cuda,openvino

# Test on priority hardware
python run_e2e_tests.py --model bert-base-uncased --priority-hardware

# Test on all supported hardware
python run_e2e_tests.py --model bert-base-uncased --all-hardware
```

### Common Options

```bash
# Enable verbose logging
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --verbose

# Update expected results
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --update-expected

# Generate documentation
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --generate-docs

# Keep temporary files for debugging
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --keep-temp
```

## Advanced Usage

### Distributed Testing

Use multiple worker threads for faster testing:

```bash
# Use 4 worker threads
python run_e2e_tests.py --all-models --priority-hardware --workers 4
```

### Custom Tolerance

Set custom tolerance for numeric comparisons:

```bash
# Set 5% tolerance
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --tolerance 0.05
```

### Force Simulation

Force simulation mode for all hardware:

```bash
# Test in simulation mode
python run_e2e_tests.py --model bert-base-uncased --hardware cuda --force-simulation
```

### Export Reports

Export test reports in different formats:

```bash
# Export JSON report
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --export-report json

# Export Markdown report
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --export-report md

# Export HTML report
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --export-report html
```

## Working with Hardware

### Hardware Detection

The framework automatically detects available hardware platforms. To check which hardware is available:

```bash
python run_e2e_tests.py --detect-hardware
```

### Simulation Mode

When hardware is not available, the framework falls back to simulation mode. To identify which tests are running in simulation mode, check the logs or use the `--verbose` flag:

```bash
python run_e2e_tests.py --model bert-base-uncased --all-hardware --verbose
```

The logs will indicate which hardware platforms are being simulated:

```
INFO: cpu detected as real hardware
WARNING: cuda will be simulated as the real hardware is not detected
```

### Hardware-Specific Notes

#### CUDA
- Requires NVIDIA GPU and appropriate drivers
- The framework checks for CUDA availability via PyTorch

#### ROCm
- Requires AMD GPU and appropriate drivers
- The framework checks for ROCm availability via PyTorch

#### MPS (Apple Metal Performance Shaders)
- Requires Apple Silicon (M1/M2/M3) hardware
- The framework checks for MPS availability via PyTorch

#### OpenVINO
- Supports Intel hardware acceleration
- Detects available devices using the OpenVINO Runtime API

#### QNN (Qualcomm Neural Networks)
- Supports Qualcomm Snapdragon hardware
- Requires the QNN SDK to be installed

#### WebNN/WebGPU
- Requires a browser with WebNN/WebGPU support
- Detection uses Selenium to launch a headless browser

## Result Validation

### Comparing Results

The framework compares test results with expected results using the ResultComparer:

- Numeric values are compared with a configurable tolerance (default 10%)
- Tensors are compared with relative and absolute tolerances
- Large arrays can use statistical comparison

### Updating Expected Results

When you make intentional changes to model behavior, update the expected results:

```bash
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --update-expected
```

### Inspecting Differences

When tests fail due to result differences, examine the differences in the test report:

```bash
cat generators/collected_results/bert-base-uncased/cpu/20250310_120000/comparison.json
```

You can also check the failure status file:

```bash
cat generators/collected_results/bert-base-uncased/cpu/20250310_120000/failure.status
```

## Database Integration

### Enabling Database Storage

The framework can store test results in a DuckDB database:

```bash
# Specify database path
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --db-path ./benchmark_db.duckdb
```

### Disabling Database Storage

If you only want file-based storage:

```bash
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --no-db
```

### Querying the Database

Use the DuckDB API to query the database:

```python
import duckdb

# Connect to the database
conn = duckdb.connect('./benchmark_db.duckdb')

# Query test results
results = conn.execute("""
    SELECT model_name, hardware_type, success, test_date 
    FROM test_results 
    ORDER BY test_date DESC
    LIMIT 10
""").fetchall()

# Print results
for row in results:
    print(row)
```

### Database Schema

The database schema includes the following tables:

- `test_results`: Stores test results with comprehensive metadata
- `hardware_capabilities`: Tracks hardware capabilities and driver versions
- `model_conversion_metrics`: Tracks model conversion statistics
- `performance_comparison`: Stores performance comparisons across hardware

## Documentation Generation

### Generating Documentation

Generate documentation for model implementations:

```bash
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --generate-docs
```

This creates Markdown documentation in the `generators/model_documentation` directory.

### Documentation Content

The generated documentation includes:

- Model overview
- Skill implementation details
- Test implementation
- Benchmark implementation
- Expected results
- Hardware-specific notes
- Performance characteristics

### Custom Documentation Path

Specify a custom output directory for documentation:

```bash
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --generate-docs --doc-dir ./docs
```

## CI/CD Integration

### Enabling CI/CD Mode

When running in CI/CD environments, use the `--ci-integration` flag:

```bash
python run_e2e_tests.py --all-models --priority-hardware --ci-integration
```

This enables:
- Status badges for pull requests
- Automatic test summaries
- Enhanced error reporting
- CI environment detection

### GitHub Actions Example

```yaml
name: E2E Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  e2e_tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        with:
          fetch-depth: 0
          
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install duckdb
          
      - name: Run end-to-end tests
        run: |
          python generators/runners/end_to_end/run_e2e_tests.py --all-models --priority-hardware --ci-integration
          
      - name: Generate test summary
        run: |
          python generators/runners/end_to_end/run_e2e_tests.py --generate-report --format markdown
```

## Maintenance Tasks

### Cleaning Old Results

To clean up old test results:

```bash
python run_e2e_tests.py --clean-old-results --days 14
```

This removes collected results older than 14 days.

### Database Maintenance

Optimize the database periodically:

```bash
python duckdb_api/benchmark_db_maintenance.py --optimize-db --vacuum
```

Create database backups:

```bash
python duckdb_api/benchmark_db_maintenance.py --backup --backup-dir ./db_backups
```

### Bulk Testing

To run tests for all models and hardware, possibly over a weekend or overnight:

```bash
python run_e2e_tests.py --all-models --all-hardware --workers 8
```

## Troubleshooting

### Common Issues and Solutions

1. **Test fails due to result differences**:
   ```bash
   # Check the differences in the comparison file
   cat generators/collected_results/bert-base-uncased/cpu/20250310_120000/comparison.json
   
   # If the differences are expected, update the expected results
   python run_e2e_tests.py --model bert-base-uncased --hardware cpu --update-expected
   ```

2. **Database connection issues**:
   ```bash
   # Check if DuckDB is installed
   pip install duckdb==0.9.2
   
   # Try with a new database file
   python run_e2e_tests.py --model bert-base-uncased --hardware cpu --db-path ./new_benchmark_db.duckdb
   ```

3. **Hardware detection failures**:
   ```bash
   # Run with verbose logging to see detection details
   python run_e2e_tests.py --model bert-base-uncased --hardware cuda --verbose
   
   # Force simulation mode if hardware is problematic
   python run_e2e_tests.py --model bert-base-uncased --hardware cuda --force-simulation
   ```

4. **Out of memory errors**:
   ```bash
   # Test with smaller batch sizes or smaller models
   python run_e2e_tests.py --model bert-tiny --hardware cuda
   
   # Test one model at a time instead of distributed testing
   python run_e2e_tests.py --model bert-base-uncased --hardware cuda --workers 1
   ```

5. **Documentation generation fails**:
   ```bash
   # Check that all required files exist
   ls generators/runners/end_to_end/
   
   # Run with verbose logging
   python run_e2e_tests.py --model bert-base-uncased --hardware cpu --generate-docs --verbose
   ```

### Getting Help

If you encounter issues not covered in this guide:

1. Check the detailed documentation in `DOCUMENTATION.md`
2. Run with `--verbose` for detailed logging
3. Examine error logs in the collected_results directory
4. Check for database error logs
5. Contact the infrastructure team for assistance

---

This user guide provides a practical overview of the End-to-End Testing Framework. For more detailed technical information, refer to the `DOCUMENTATION.md` file.