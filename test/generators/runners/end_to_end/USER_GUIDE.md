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

### Using the Unified Component Tester (Recommended)

The Unified Component Tester is the recommended way to run tests, as it provides enhanced functionality:

```bash
# Test a specific model
python unified_component_tester.py --model bert-base-uncased --hardware cpu

# Test multiple models
python unified_component_tester.py --model bert-base-uncased,t5-small --hardware cpu

# Test all models in a family
python unified_component_tester.py --model-family text-embedding --hardware cpu

# Test all supported models
python unified_component_tester.py --all-models --hardware cpu

# Run tests in parallel with multiple workers
python unified_component_tester.py --all-models --hardware cpu --max-workers 4
```

### Using the Legacy Test Runner

For backward compatibility, the original end-to-end testing framework is still available:

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

With the Unified Component Tester:

```bash
# Test on specific hardware
python unified_component_tester.py --model bert-base-uncased --hardware cuda

# Test on multiple hardware platforms
python unified_component_tester.py --model bert-base-uncased --hardware cpu,cuda,openvino

# Test on priority hardware (cpu, cuda, openvino, webgpu)
python unified_component_tester.py --model bert-base-uncased --priority-hardware

# Test on all supported hardware
python unified_component_tester.py --model bert-base-uncased --all-hardware
```

With the legacy runner:

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

With the Unified Component Tester:

```bash
# Enable verbose logging
python unified_component_tester.py --model bert-base-uncased --hardware cpu --verbose

# Update expected results
python unified_component_tester.py --model bert-base-uncased --hardware cpu --update-expected

# Generate documentation
python unified_component_tester.py --model bert-base-uncased --hardware cpu --generate-docs

# Keep temporary files for debugging
python unified_component_tester.py --model bert-base-uncased --hardware cpu --keep-temp

# Run a quick test with minimal validation
python unified_component_tester.py --model bert-base-uncased --hardware cpu --quick-test

# Set a custom tolerance for numeric comparisons
python unified_component_tester.py --model bert-base-uncased --hardware cpu --tolerance 0.05
```

With the legacy runner:

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

Use multiple worker processes for parallel testing:

```bash
# With the Unified Component Tester (recommended)
# Use 4 worker processes
python unified_component_tester.py --all-models --priority-hardware --max-workers 4

# Test specific families in parallel
python unified_component_tester.py --model-family text-embedding,vision --hardware cpu,cuda --max-workers 4

# With the legacy runner
# Use 4 worker threads
python run_e2e_tests.py --all-models --priority-hardware --workers 4
```

### Custom Tolerance

Set custom tolerance for numeric comparisons:

```bash
# With the Unified Component Tester
# Set 5% tolerance
python unified_component_tester.py --model bert-base-uncased --hardware cpu --tolerance 0.05

# With the legacy runner
# Set 5% tolerance
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --tolerance 0.05
```

### Force Simulation

Force simulation mode for all hardware:

```bash
# With the Unified Component Tester
# Test in simulation mode
python unified_component_tester.py --model bert-base-uncased --hardware cuda --force-simulation

# With the legacy runner
# Test in simulation mode
python run_e2e_tests.py --model bert-base-uncased --hardware cuda --force-simulation
```

### Export Reports

Export test reports in different formats:

```bash
# With the Unified Component Tester
# Export a summary report after running tests
python unified_component_tester.py --model bert-base-uncased --hardware cpu --generate-report

# With the legacy runner
# Export JSON report
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --export-report json

# Export Markdown report
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --export-report md

# Export HTML report
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --export-report html
```

### Testing the Tester

The unified component tester includes comprehensive test suites and validation tools:

```bash
# Run the test suite for the unified component tester
python test_unified_component_tester.py

# Run tests for a specific model family
python test_unified_component_tester.py --model-family vision

# Run comprehensive tests (all model families and hardware platforms)
python test_unified_component_tester.py --comprehensive

# Use the shell script to run all tests
./run_unified_component_tests.sh

# Run realistic tests (takes longer)
./run_unified_component_tests.sh --realistic

# Run CI-optimized tests
python ci_unified_component_test.py
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
cat scripts/generators/collected_results/bert-base-uncased/cpu/20250310_120000/comparison.json
```

You can also check the failure status file:

```bash
cat scripts/generators/collected_results/bert-base-uncased/cpu/20250310_120000/failure.status
```

## Database Integration

### Enabling Database Storage

The framework can store test results in a DuckDB database:

```bash
# With the Unified Component Tester
# Specify database path
python unified_component_tester.py --model bert-base-uncased --hardware cpu --db-path ./benchmark_db.duckdb

# With the legacy runner
# Specify database path
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --db-path ./benchmark_db.duckdb
```

### Disabling Database Storage

If you only want file-based storage:

```bash
# With the Unified Component Tester
python unified_component_tester.py --model bert-base-uncased --hardware cpu --no-db

# With the legacy runner
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
# With the Unified Component Tester (enhanced documentation)
python unified_component_tester.py --model bert-base-uncased --hardware cpu --generate-docs

# With the legacy runner
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --generate-docs
```

This creates Markdown documentation in the `scripts/generators/model_documentation` directory. The unified component tester generates enhanced documentation with model-family specific content and hardware-specific optimization information.

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

For CI/CD environments, you have two options:

#### Using the Unified Component Tester (Recommended)

```bash
# Use the dedicated CI test script
python ci_unified_component_test.py

# OR run the unified component tester with CI integration flag
python unified_component_tester.py --all-models --priority-hardware --ci-integration
```

#### Using the Legacy Runner

```bash
python run_e2e_tests.py --all-models --priority-hardware --ci-integration
```

These approaches enable:
- Status badges for pull requests
- Automatic test summaries
- Enhanced error reporting
- CI environment detection
- Exit code handling for test status

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
          pip install duckdb==0.9.2
          
      # Using the Unified Component Tester (recommended)
      - name: Run CI tests
        run: |
          python scripts/generators/runners/end_to_end/ci_unified_component_test.py
          
      # Or run the unified component tester directly
      - name: Run end-to-end tests
        run: |
          python scripts/generators/runners/end_to_end/unified_component_tester.py --all-models --priority-hardware --ci-integration
          
      # Or use the legacy runner (not recommended for new projects)
      # - name: Run legacy end-to-end tests
      #   run: |
      #     python scripts/generators/runners/end_to_end/run_e2e_tests.py --all-models --priority-hardware --ci-integration
          
      - name: Generate test summary
        run: |
          python scripts/generators/runners/end_to_end/unified_component_tester.py --generate-report --format markdown --output test_summary.md
          
      - name: Archive test results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: |
            scripts/generators/collected_results/
            benchmark_db.duckdb
            test_summary.md
```

## Maintenance Tasks

### Cleaning Old Results

To clean up old test results:

```bash
# With the Unified Component Tester
python unified_component_tester.py --clean-old-results --days 14

# With the legacy runner
python run_e2e_tests.py --clean-old-results --days 14
```

This removes collected results older than 14 days to save disk space.

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
# With the Unified Component Tester (recommended for large workloads)
python unified_component_tester.py --all-models --all-hardware --max-workers 8

# With the legacy runner
python run_e2e_tests.py --all-models --all-hardware --workers 8
```

The unified component tester provides better handling of large workloads with more robust error recovery and parallel execution.

## Visualization Dashboard and Integrated Reports System

The framework provides comprehensive visualization and reporting capabilities through an interactive web dashboard and an integrated reports system.

### Visualization Dashboard

The Visualization Dashboard provides an interactive web interface for exploring test results and performance metrics.

#### Starting the Dashboard

```bash
# Start the dashboard with default settings
python visualization_dashboard.py

# Specify a custom port and database path
python visualization_dashboard.py --port 8050 --db-path ./benchmark_db.duckdb

# Run in development mode with auto-reloading
python visualization_dashboard.py --debug
```

Once started, open your web browser and navigate to `http://localhost:8050` (or the custom port you specified).

#### Dashboard Features

The dashboard is organized into five main tabs:

1. **Overview**: Provides a high-level summary of test results, including success rates and distribution across models and hardware platforms.

2. **Performance Analysis**: Allows detailed analysis of performance metrics (throughput, latency, memory usage) for specific models and hardware combinations.

3. **Hardware Comparison**: Enables side-by-side comparison of different hardware platforms, with visualizations to identify optimal hardware for each model type.

4. **Time Series Analysis**: Shows performance trends over time, with statistical analysis to identify significant changes and potential regressions.

5. **Simulation Validation**: Validates the accuracy of hardware simulations by comparing performance metrics between simulated and real hardware.

#### Dashboard Requirements

To use the dashboard, you'll need to install the following dependencies:

```bash
pip install dash dash-bootstrap-components plotly pandas numpy duckdb
```

### Integrated Visualization and Reports System

The Integrated Visualization and Reports System combines the Visualization Dashboard with the Enhanced CI/CD Reports Generator into a unified interface that provides both interactive exploration and comprehensive reporting.

#### Basic Usage

```bash
# Start the dashboard only
python integrated_visualization_reports.py --dashboard

# Generate reports only
python integrated_visualization_reports.py --reports

# Start dashboard and generate reports (combines both features)
python integrated_visualization_reports.py --dashboard --reports

# Specify database path and automatically open browser
python integrated_visualization_reports.py --dashboard --db-path ./benchmark_db.duckdb --open-browser

# Export dashboard visualizations for offline viewing
python integrated_visualization_reports.py --dashboard-export
```

#### Report Generation Options

The integrated system supports generating various specialized report types:

```bash
# Generate simulation validation report (validates simulation accuracy)
python integrated_visualization_reports.py --reports --simulation-validation

# Generate cross-hardware comparison report (compares performance across hardware)
python integrated_visualization_reports.py --reports --cross-hardware-comparison

# Generate a combined report with multiple analyses in one document
python integrated_visualization_reports.py --reports --combined-report

# Generate historical trend analysis over a specific time period
python integrated_visualization_reports.py --reports --historical --days 30

# Generate CI/CD status badges for dashboards
python integrated_visualization_reports.py --reports --badge-only

# Generate reports with enhanced visualizations
python integrated_visualization_reports.py --reports --include-visualizations

# Export metrics to CSV for further analysis
python integrated_visualization_reports.py --reports --export-metrics

# Generate report with simulation highlighting
python integrated_visualization_reports.py --reports --highlight-simulation

# Set a specific tolerance for simulation validation comparisons
python integrated_visualization_reports.py --reports --simulation-validation --tolerance 0.15

# Specify output format for reports
python integrated_visualization_reports.py --reports --format markdown
```

#### Dashboard Options

The integrated system provides enhanced dashboard options:

```bash
# Start dashboard on a custom port and host
python integrated_visualization_reports.py --dashboard --dashboard-port 8080 --dashboard-host 0.0.0.0

# Enable debug mode for development with hot reloading
python integrated_visualization_reports.py --dashboard --debug

# Configure output directory for reports and exports
python integrated_visualization_reports.py --reports --output-dir ./my_reports
```

#### Integrated Workflows

The system is designed to support common workflows with convenient command combinations:

```bash
# Run a complete analysis: start dashboard, generate reports, and open browser
python integrated_visualization_reports.py --dashboard --reports --combined-report --open-browser

# CI pipeline integration with badge generation and GitHub Pages support
python integrated_visualization_reports.py --reports --badge-only --github-pages --ci

# Performance analysis focus: hardware comparison and simulation validation
python integrated_visualization_reports.py --reports --cross-hardware-comparison --simulation-validation

# Export dashboard, generate reports, with enhanced visualizations
python integrated_visualization_reports.py --dashboard-export --reports --include-visualizations

# Detailed simulation analysis with custom tolerance and highlighting
python integrated_visualization_reports.py --reports --simulation-validation --tolerance 0.10 --highlight-simulation

# Advanced tracking over extended period with metrics export
python integrated_visualization_reports.py --reports --historical --days 90 --export-metrics
```

#### Key Benefits

The integrated system provides several advantages over using the standalone components:

1. **Unified Interface**: Consistent command-line interface for both the dashboard and reports
   - Single entry point for all visualization and reporting needs
   - Compatible argument structure for both components
   - Simplified configuration for database connections

2. **Process Management**: Robust dashboard process handling
   - Automated startup and monitoring of the dashboard process
   - Graceful handling of keyboard interrupts and process termination
   - Proper cleanup of resources when shutting down

3. **Consistent Database Access**: Same database connection used across all components
   - Ensures dashboard and reports use the same dataset
   - Unified database configuration
   - Consistent performance metrics across visualizations and reports

4. **Coordinated Analysis**: Better integration between visualizations and reports
   - Generate reports based on the same data viewed in the dashboard
   - Consistent filtering and selection criteria
   - Integrated simulation validation across both components

5. **Enhanced User Experience**: Streamlined workflows
   - Browser integration to automatically open the dashboard
   - Export capabilities for offline sharing of visualizations
   - Unified option handling for both components
   - Consistent styling and formatting across all outputs

6. **CI/CD Integration**: Optimized for continuous integration environments
   - Badge generation for status dashboards
   - GitHub Pages support for report publishing
   - Export formats suitable for CI artifacts
   - Consistent exit codes for CI pipeline integration

#### Implementation Architecture

The `IntegratedSystem` class:
- Coordinates the Visualization Dashboard and Enhanced CI/CD Reports Generator
- Manages process lifecycle for the dashboard
- Provides a unified command-line interface
- Handles resource management and cleanup
- Ensures consistent database configuration
- Supports both interactive and CI/CD usage patterns

The system is designed with a modular architecture that allows for:
- Independent operation of each component
- Combined execution for integrated workflows
- Consistent configuration across components
- Extensibility for future enhancements
- Clear separation of concerns while maintaining integration

For comprehensive documentation, see:
- [VISUALIZATION_DASHBOARD_README.md](VISUALIZATION_DASHBOARD_README.md) for detailed dashboard information.
- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) for a thorough explanation of the system architecture and component interactions.
- [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md) for solutions to common issues with the integrated system.

## Troubleshooting

### Common Issues and Solutions

1. **Test fails due to result differences**:
   ```bash
   # Check the differences in the comparison file
   cat scripts/generators/collected_results/bert-base-uncased/cpu/20250310_120000/comparison.json
   
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
   ls scripts/generators/runners/end_to_end/
   
   # Run with verbose logging
   python run_e2e_tests.py --model bert-base-uncased --hardware cpu --generate-docs --verbose
   ```

### Unified Component Tester Specific Issues

1. **Test Suite Failures**:
   ```bash
   # Check which tests are failing
   python test_unified_component_tester.py --verbose
   
   # Run tests for a specific model family or hardware
   python test_unified_component_tester.py --model-family vision
   python test_unified_component_tester.py --hardware cpu
   ```

2. **Parallel Execution Issues**:
   ```bash
   # Reduce worker count
   python unified_component_tester.py --model bert-base-uncased --hardware cpu --max-workers 1
   
   # Enable verbose logging
   python unified_component_tester.py --model bert-base-uncased --hardware cpu --verbose
   ```

3. **Template Issues**:
   ```bash
   # Check template database
   python template_database.py --list-templates
   
   # Validate templates
   python template_validation.py --validate-all
   ```

### Getting Help

If you encounter issues not covered in this guide:

1. **Consult the dedicated documentation**:
   - For detailed system architecture: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
   - For troubleshooting the integrated system: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
   - These guides provide comprehensive information on system components, interactions, data flow, and solutions for common issues like dashboard process management, database connectivity, report generation, visualization rendering, and browser integration.

2. Check the detailed documentation:
   - For the unified component tester: `UNIFIED_COMPONENT_TESTER.md`
   - For the legacy framework: `DOCUMENTATION.md`
   
3. Run the appropriate tool with `--verbose` for detailed logging:
   ```bash
   python unified_component_tester.py --model bert-base-uncased --hardware cpu --verbose
   ```
   
4. Examine error logs in the collected_results directory
   
5. Run the test suite to check for issues with the tester itself:
   ```bash
   python test_unified_component_tester.py
   ```
   
6. Check for database error logs
   
7. Contact the infrastructure team for assistance

---

This user guide provides a practical overview of the End-to-End Testing Framework. For more detailed technical information, refer to:

- For the unified component tester: [UNIFIED_COMPONENT_TESTER.md](UNIFIED_COMPONENT_TESTER.md)
- For the overall testing framework: [DOCUMENTATION.md](DOCUMENTATION.md)
- For system architecture: [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
- For the template system: [TEMPLATE_SYSTEM_GUIDE.md](TEMPLATE_SYSTEM_GUIDE.md)
- For integration details: [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)
- For troubleshooting: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)

The unified component tester is the recommended approach for all new testing as it provides enhanced functionality, better error handling, and comprehensive model family and hardware support.