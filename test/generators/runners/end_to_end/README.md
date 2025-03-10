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

### Basic Usage

```bash
# Test a single model on a single hardware platform
python run_e2e_tests.py --model bert-base-uncased --hardware cpu

# Test with documentation generation
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --generate-docs

# Update expected results
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --update-expected

# Enable verbose logging
python run_e2e_tests.py --model bert-base-uncased --hardware cpu --verbose
```

### Advanced Usage

```bash
# Test a model on multiple hardware platforms
python run_e2e_tests.py --model bert-base-uncased --hardware cpu,cuda,webgpu

# Test all models in a family
python run_e2e_tests.py --model-family text-embedding --hardware cpu

# Test all supported models
python run_e2e_tests.py --all-models --priority-hardware

# Clean up old test results
python run_e2e_tests.py --clean-old-results --days 14
```

## Directory Structure

```
generators/
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
          python generators/runners/end_to_end/run_e2e_tests.py --all-models --priority-hardware --ci-integration --db-path ./benchmark_db.duckdb
          
      - name: Archive test results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: |
            generators/collected_results/
            benchmark_db.duckdb
            
      - name: Generate test summary
        run: |
          python generators/runners/end_to_end/run_e2e_tests.py --generate-report --format markdown --output test_summary.md
          
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

4. **Hardware detection issues**:
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

## Contact

If you have questions or need support with the end-to-end testing framework, please contact the infrastructure team.