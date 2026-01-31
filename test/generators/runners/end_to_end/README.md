# End-to-End Testing Framework

## Overview

The End-to-End Testing Framework is a comprehensive solution for testing model implementations across different hardware platforms. Rather than testing individual components separately, this framework focuses on:

1. Generating all components (skill, test, benchmark) together
2. Testing components as a unified whole
3. Comparing results against expected outputs
4. Generating comprehensive documentation
5. Tracking simulation vs. real hardware execution
6. Storing comprehensive test results in DuckDB database

This approach ensures that models work correctly across the entire pipeline and helps identify issues at the generator level rather than in individual files.

## Component Testing Frameworks

### Unified Component Tester

The **Unified Component Tester** is our latest enhanced implementation that provides a complete solution for generating and testing model components across different hardware platforms. It builds on the integrated component tester with:

1. **Complete Component Integration**: Generates, tests, and evaluates all components together as a cohesive unit
2. **Enhanced Result Organization**: Creates robust expected/collected results structure with historical tracking
3. **Advanced Documentation Generation**: Produces detailed markdown documentation with model-family and hardware-specific content
4. **Comprehensive Testing**: Validates components across all model families and hardware platforms
5. **Template-Driven Development**: Uses a robust template system for maintenance efficiency
6. **Parallel Execution**: Supports multi-worker execution for faster testing
7. **Database Integration**: Stores results in DuckDB with simulation status tracking

```bash
# Test a model with the unified component tester
python unified_component_tester.py --model bert-base-uncased --hardware cpu

# Run a comprehensive test across model families
python unified_component_tester.py --model-family text-embedding --hardware cpu,cuda

# Run tests in parallel with documentation
python unified_component_tester.py --all-models --priority-hardware --max-workers 4 --generate-docs
```

[Learn more about the Unified Component Tester](./unified_component_tester.py)

### Integrated Component Testing Framework (Previous Version)

The **Integrated Component Testing Framework** was our initial implementation that addresses the prioritized tasks from our roadmap:

1. **Joint Component Generation**: Generates skill, test, and benchmark components together for every model
2. **Expected/Collected Results Organization**: Creates structured directories for verification
3. **Comprehensive Documentation**: Generates Markdown documentation of HuggingFace class skills
4. **Generator-Focused Fixes**: Focuses on fixing generators rather than individual files
5. **Template-Driven Approach**: Implements a template-driven workflow for maintenance efficiency

[Learn more about the Integrated Component Testing Framework](./INTEGRATED_COMPONENT_TESTING.md)

## Key Features

- **Joint Component Generation**: Creates skill, test, and benchmark files together
- **Cross-Platform Testing**: Tests models across different hardware platforms with robust hardware detection
- **Advanced Result Validation**: Compares actual results with expected results using statistical methods and configurable tolerances
- **Automatic Documentation**: Generates comprehensive Markdown documentation for each implementation
- **Template-Driven Approach**: Focuses fixes on generators rather than individual files
- **Test Result Persistence**: Stores test results in both file system and DuckDB database
- **Hardware Simulation Awareness**: Detects and tracks when hardware is being simulated
- **Enhanced Metadata**: Captures detailed environment, platform, and git information
- **CI/CD Integration**: Special handling for continuous integration environments
- **Distributed Testing Support**: Parallel execution with worker threads

## Usage

### Unified Component Tester

```bash
# Test a single model on a single hardware platform
python unified_component_tester.py --model bert-base-uncased --hardware cpu

# Test with documentation generation
python unified_component_tester.py --model bert-base-uncased --hardware cpu --generate-docs

# Update expected results
python unified_component_tester.py --model bert-base-uncased --hardware cpu --update-expected

# Enable verbose logging
python unified_component_tester.py --model bert-base-uncased --hardware cpu --verbose

# Run comprehensive tests
python unified_component_tester.py --model-family text-embedding --hardware cpu,cuda,webgpu --generate-docs --max-workers 4

# Run a quick test with minimal validation
python unified_component_tester.py --model bert-base-uncased --hardware cpu --quick-test

# Run tests and store results in custom database
python unified_component_tester.py --model bert-base-uncased --hardware cpu --db-path ./my_test_db.duckdb

# Run tests for all models on priority hardware platforms
python unified_component_tester.py --all-models --priority-hardware --generate-docs

# Clean up old test results
python unified_component_tester.py --clean-old-results --days 14
```

### Run Unified Component Tests Script

For a comprehensive test of the unified component tester itself:

```bash
# Run basic tests
./run_unified_component_tests.sh

# Run with realistic tests (takes longer)
./run_unified_component_tests.sh --realistic
```

### CI Testing

For continuous integration environments:

```bash
# Run the CI-optimized tests
python ci_unified_component_test.py
```

### Legacy End-to-End Tests

The original end-to-end testing framework is still available:

```bash
# Test a single model on a single hardware platform
python run_e2e_tests.py --model bert-base-uncased --hardware cpu

# Test with documentation generation
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --generate-docs

# Test all models in a family
python run_e2e_tests.py --model-family text-embedding --hardware cpu

# Test all supported models
python run_e2e_tests.py --all-models --priority-hardware
```

## Directory Structure

```
scripts/generators/
├── expected_results/        # Expected outputs for regression testing
│   ├── bert-base-uncased/
│   │   ├── cpu/
│   │   │   └── expected_result.json
│   │   └── ...
│   └── ...
├── collected_results/       # Actual test results with timestamps
│   ├── bert-base-uncased/
│   │   ├── cpu/
│   │   │   └── 20250310_120000/
│   │   └── ...
│   └── summary/             # Summary reports from test runs
├── model_documentation/     # Generated documentation
│   ├── bert-base-uncased/
│   │   ├── cpu_implementation.md
│   │   └── ...
│   └── ...
└── runners/
    └── end_to_end/          # End-to-end testing framework scripts
        ├── run_e2e_tests.py             # Main script for running tests
        ├── template_validation.py       # Validation and comparison logic
        ├── model_documentation_generator.py  # Documentation generator
        └── simple_utils.py              # Utility functions
```

## Command Line Options

```
usage: run_e2e_tests.py [-h] [--model MODEL] [--model-family MODEL_FAMILY]
                        [--all-models] [--hardware HARDWARE]
                        [--priority-hardware] [--all-hardware] [--quick-test]
                        [--update-expected] [--generate-docs] [--keep-temp]
                        [--clean-old-results] [--days DAYS]
                        [--clean-failures] [--verbose] [--db-path DB_PATH]
                        [--no-db] [--workers WORKERS] [--tolerance TOLERANCE]
                        [--force-simulation] [--ci-integration]
                        [--export-report EXPORT_REPORT]

End-to-End Testing Framework for IPFS Accelerate

options:
  -h, --help            show this help message and exit
  --model MODEL         Specific model to test
  --model-family MODEL_FAMILY
                        Model family to test (e.g., text-embedding, vision)
  --all-models          Test all supported models
  --hardware HARDWARE   Hardware platforms to test, comma-separated (e.g.,
                        cpu,cuda,webgpu)
  --priority-hardware   Test on priority hardware platforms (cpu, cuda,
                        openvino, webgpu)
  --all-hardware        Test on all supported hardware platforms
  --quick-test          Run a quick test with minimal validation
  --update-expected     Update expected results with current test results
  --generate-docs       Generate markdown documentation for models
  --keep-temp           Keep temporary directories after tests
  --clean-old-results   Clean up old collected results
  --days DAYS           Number of days to keep results when cleaning (default:
                        14)
  --clean-failures      Clean failed test results too
  --verbose             Enable verbose logging
  --db-path DB_PATH     Path to the DuckDB database for storing results
  --no-db               Disable database storage of results
  --workers WORKERS     Number of worker threads for distributed testing
  --tolerance TOLERANCE Set custom tolerance for numeric comparisons (e.g., 0.05 for 5%)
  --force-simulation    Force simulation mode for all hardware platforms
  --ci-integration      Enable CI/CD-specific optimizations and reporting
  --export-report EXPORT_REPORT
                        Export test report to specified file format (json, md, html)
```

## Workflow

1. **Test Execution**:
   - The framework detects available hardware platforms and tracks simulation status
   - It generates skill, test, and benchmark files based on templates
   - It runs tests with real hardware or in simulation mode and collects results
   - It compares results with expected results using advanced validation
   - It generates comprehensive documentation if requested
   - It stores results in both file system and DuckDB database

2. **Result Management**:
   - Expected results are stored in the `expected_results` directory
   - Actual results are stored in timestamped directories under `collected_results`
   - Rich metadata is captured including environment, platform, and git information
   - Performance metrics are tracked for trend analysis
   - Summary reports are generated for each test run
   - Test results are stored in DuckDB database with simulation status tracking

3. **Distributed Execution**:
   - Tests can be distributed across multiple worker threads
   - Tasks are allocated efficiently to maximize resource utilization
   - Results are aggregated into a single comprehensive report
   - Each worker handles a subset of the model/hardware combinations

4. **Database Integration**:
   - Test results are stored in DuckDB database with rich metadata
   - SQL queries can be used to analyze test results trends
   - Performance metrics are tracked over time for regression detection
   - Simulation status is tracked to distinguish between real and simulated tests

5. **Maintenance**:
   - When models or hardware change, update expected results
   - Fix issues in the generators rather than in individual files
   - Clean up old test results periodically
   - Monitor database size and performance

## Adding New Models or Hardware

To add a new model or hardware platform to the testing framework:

1. Update the `MODEL_FAMILY_MAP` in `run_e2e_tests.py` to include the new model
2. Update the `SUPPORTED_HARDWARE` list to include the new hardware platform
3. Run tests with the `--update-expected` flag to generate expected results
4. Run tests normally to verify the implementation works correctly

## Best Practices

1. **Always update expected results after significant changes**:
   ```bash
   python run_e2e_tests.py --model your-model --hardware your-hardware --update-expected
   ```

2. **Generate documentation to understand implementation**:
   ```bash
   python run_e2e_tests.py --model your-model --hardware your-hardware --generate-docs
   ```

3. **Clean up old results periodically**:
   ```bash
   python run_e2e_tests.py --clean-old-results --days 14
   ```

4. **Focus fixes on generators rather than individual files**

## Integration with CI/CD

This framework is fully integrated with CI/CD pipelines to:

1. Test all models and hardware platforms automatically
2. Update expected results when approved by reviewers
3. Generate documentation as part of the build process
4. Alert on test failures or regressions
5. Track performance metrics over time
6. Store results in the database with CI metadata
7. Generate badges and reports for pull requests

Sample CI/CD configuration:

```yaml
jobs:
  e2e_tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        with:
          fetch-depth: 0  # Get full history for git information
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          
      - name: Set up DuckDB
        run: |
          pip install duckdb==0.9.2
          
      - name: Run end-to-end tests
        run: |
          python scripts/generators/runners/end_to_end/run_e2e_tests.py --all-models --priority-hardware --ci-integration --db-path ./benchmark_db.duckdb
          
      - name: Archive test results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: |
            scripts/generators/collected_results/
            benchmark_db.duckdb
            
      - name: Generate test summary
        run: |
          python scripts/generators/runners/end_to_end/run_e2e_tests.py --generate-report --format markdown --output test_summary.md
          
      - name: Update PR with test results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v5
        with:
          script: |
            const fs = require('fs');
            const summary = fs.readFileSync('test_summary.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: summary
            });
```

The CI/CD integration includes:

1. **Status Badges**: Add test status badges to pull requests and README
2. **Performance Tracking**: Monitor performance trends over time
3. **Simulation Awareness**: Clear indicators when hardware is simulated
4. **Comprehensive Reporting**: Detailed reports with test results and metrics
5. **Database Integration**: Store results in DuckDB with CI environment metadata
6. **Pull Request Comments**: Automatic test summary comments on pull requests

## Completed Features and Implementation Status

The Unified Component Tester successfully implements all the required features for the Improved End-to-End Testing Framework as prioritized in CLAUDE.md:

✅ **Generation and testing of all components together for every model**
- Fully implemented in the `UnifiedComponentTester.generate_components()` method
- Ensures skill, test, and benchmark files are created and tested as a unit
- Validates all components work correctly together
- Focuses on fixing generators rather than individual files

✅ **Creation of "expected_results" and "collected_results" folders for verification**
- Creates organized directory structure for results with timestamps
- Supports comparison between expected and actual results
- Enables regression testing with configurable tolerance
- Stores comprehensive metadata with results

✅ **Markdown documentation of HuggingFace class skills to compare with templates**
- Generates detailed documentation with model-family specific content
- Includes hardware-specific optimization information
- Captures API details, usage examples, and benchmark results
- Supports enhanced visualization with ASCII charts

✅ **Focus on fixing generators rather than individual test files**
- Template-driven approach to component generation
- Robust error handling and fallback mechanisms
- Validation of components against expected structure
- Consistent code generation across the system

✅ **Template-driven approach for maintenance efficiency**
- Support for model-family-specific templates
- Hardware-specific template customization
- Cross-platform compatibility built into templates
- Advanced variable substitution with transformations

Additional enhancements:

✅ **Database Integration**
- Complete integration with DuckDB for result storage
- Schema support for comprehensive metrics
- Simulation detection and tracking
- Historical performance analysis

✅ **Parallel Execution**
- Multi-worker support for distributed testing
- Efficient resource utilization with worker pools
- Centralized result aggregation
- Configurable worker count via `--max-workers`

✅ **CI/CD Integration**
- Specialized test script for CI environments
- Quick test mode for faster validation
- Exit code handling for test status
- Comprehensive test summary output

✅ **Comprehensive Testing**
- Support for all model families (text-embedding, text-generation, vision, audio, multimodal)
- Support for all hardware platforms (cpu, cuda, rocm, mps, openvino, qnn, webnn, webgpu)
- Automated test suite for the framework itself
- Batch testing with configurable options

## Troubleshooting

1. **Tests are failing but the implementation looks correct**:
   - Check if expected results need updating: `--update-expected`
   - Examine the differences in the test report
   - Adjust the tolerance level if precision differences are expected: `--tolerance 0.05`
   - Check if hardware is being simulated when it shouldn't be

2. **Documentation is not generating correctly**:
   - Check that all components are being generated correctly
   - Verify the model and hardware names are correct
   - Examine the template_validation.py logs for errors
   - Ensure the ModelDocGenerator has access to the required files

3. **Database integration issues**:
   - Verify the DuckDB installation with: `pip install duckdb==0.9.2`
   - Check database path permissions: `--db-path ./benchmark_db.duckdb`
   - Examine the db_error.log files in the results directory
   - Try using `--no-db` to disable database integration temporarily

4. **Visualization Dashboard and Integrated Reports System issues**:
   - Consult the comprehensive [TROUBLESHOOTING_GUIDE.md](./TROUBLESHOOTING_GUIDE.md) for detailed solutions to:
     - Dashboard process management issues
     - Database connectivity problems
     - Report generation failures
     - Visualization rendering issues
     - Browser integration problems
     - Combined workflow challenges
     - CI/CD integration issues

5. **Hardware detection issues**:
   - Ensure required libraries are installed for hardware detection
   - Check hardware-specific environment variables and drivers
   - Use `--verbose` to see detailed hardware detection logs
   - Use `--force-simulation` if hardware detection is problematic

5. **Clean old results**:
   - If the collected_results directory is getting too large, use `--clean-old-results`
   - For database maintenance, use the duckdb_api utilities
   - Archive old results with `--days 14 --clean-failures`

6. **Performance issues with distributed testing**:
   - Adjust worker count for your system: `--workers 4`
   - Monitor system resource usage during testing
   - Reduce batch size or use smaller model variants for faster testing
   - Run tests for specific hardware or models to reduce workload

## Visualization Dashboard

The framework includes a comprehensive visualization dashboard for exploring test results and performance metrics:

```bash
# Start the dashboard with default settings
python visualization_dashboard.py

# Customize port and database path
python visualization_dashboard.py --port 8050 --db-path ./benchmark_db.duckdb

# Run in development mode
python visualization_dashboard.py --debug
```

The dashboard provides:
- Real-time monitoring of test results and performance
- Interactive visualizations for comparing hardware platforms
- Time series analysis with statistical testing
- Simulation validation visualization
- Customizable filtering and views

To install dashboard dependencies:
```bash
pip install -r dashboard_requirements.txt
```

### Integrated Visualization and Reports System

For enhanced functionality, you can use the integrated system that combines the dashboard with the CI/CD reporting tools:

```bash
# Start the dashboard only
python integrated_visualization_reports.py --dashboard

# Generate reports only
python integrated_visualization_reports.py --reports

# Start dashboard and generate reports
python integrated_visualization_reports.py --dashboard --reports

# Specify database path and automatically open browser
python integrated_visualization_reports.py --dashboard --db-path ./benchmark_db.duckdb --open-browser

# Generate specific report types
python integrated_visualization_reports.py --reports --simulation-validation

# Export dashboard visualizations for offline viewing
python integrated_visualization_reports.py --dashboard-export
```

The integrated system provides:
- Unified command-line interface for dashboard and reports
- Consistent database access across all components
- Report generation based on live dashboard data
- Easy-to-use commands for common scenarios
- Support for both interactive exploration and CI/CD integration

For detailed documentation on the visualization dashboard and integrated system, see:
- [VISUALIZATION_DASHBOARD_README.md](./VISUALIZATION_DASHBOARD_README.md)
- [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)
- [TROUBLESHOOTING_GUIDE.md](./TROUBLESHOOTING_GUIDE.md)
- [INTEGRATION_SUMMARY.md](./INTEGRATION_SUMMARY.md)

## Next Steps

For further enhancement of the unified component tester:

1. **Expanded Visualization Dashboard**: Add more advanced analytics and visualization types
2. **Performance Prediction**: Implement ML-based performance prediction for untested configurations
3. **Anomaly Detection**: Add automated detection of performance anomalies and regressions
4. **Distributed Testing Integration**: Complete integration with the Distributed Testing Framework
5. **Mobile-Friendly Interface**: Optimize dashboard for mobile and tablet devices

## Contact

If you have questions or need support with the unified component testing framework, please contact the infrastructure team.