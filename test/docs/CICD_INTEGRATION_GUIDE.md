# CI/CD Integration for Test Results

**Date: March 7, 2025**  
**Version: 1.0**  
**Status: Active**

This guide documents the CI/CD integration implemented for the IPFS Accelerate Python Framework for automatically running tests, storing results in the database, generating reports, and detecting performance regressions.

## Overview

The CI/CD integration system automates the following tasks:

1. Running tests on multiple models and hardware platforms
2. Storing test results directly in the DuckDB database
3. Generating comprehensive reports (compatibility matrix, performance reports)
4. Detecting performance regressions automatically
5. Creating GitHub issues for significant regressions
6. Publishing reports to GitHub Pages
7. Archiving historical data for trend analysis

The system is implemented using GitHub Actions workflows that run on schedule, on push to main, on pull requests, or can be triggered manually.

## Workflows

### Test Results Integration Workflow

The primary workflow for test results integration is `test_results_integration.yml`. This workflow:

- Runs daily at 1 AM UTC
- Runs on push to main branch for relevant files
- Runs on pull requests for relevant files
- Can be manually triggered with customizable parameters

The workflow consists of the following jobs:

1. **setup_database**: Initializes the DuckDB database for storing test results
   - Creates a new DuckDB database with the proper schema
   - Generates a unique run ID for tracking this execution
   - Creates CI metadata with information about the current run
   - Uploads the database as an artifact for other jobs to use

2. **run_tests**: Runs tests for specified models and hardware platforms
   - Creates a matrix of models and hardware platforms to test
   - Downloads the database from the setup job
   - Runs tests using `test_ipfs_accelerate.py` with direct database storage
   - Verifies that test results were properly stored in the database
   - Generates test reports in markdown format
   - Uploads the updated database and reports as artifacts

3. **run_web_tests**: Optionally runs web platform tests (WebNN, WebGPU)
   - Runs only when the `include_web` parameter is set to true
   - Sets up Node.js environment for web platform testing
   - Runs web platform tests using `web_platform_test_runner.py`
   - Stores results directly in the database
   - Uploads the updated database as an artifact

4. **consolidate_results**: Processes all test results and generates reports
   - Downloads all artifacts from previous jobs
   - Uses `ci_benchmark_integrator.py` to consolidate results into a single database
   - Generates a compatibility matrix using the consolidated data
   - Creates performance reports in HTML format
   - Detects performance regressions using `benchmark_regression_detector.py`
   - Creates GitHub issues for significant regressions when running on schedule
   - Uploads the consolidated database and reports as artifacts

5. **publish_reports**: Publishes reports to GitHub Pages
   - Runs only on schedule or on push to main
   - Downloads the reports from the consolidate job
   - Creates an index HTML file for navigation
   - Deploys all reports to GitHub Pages
   - Provides a permanent URL for viewing the reports

6. **archive_database**: Archives the database for historical data retention
   - Creates a date-stamped copy of the database
   - Uploads the archived database as an artifact
   - Retains historical data for 90 days

### Compatibility Matrix Update Workflow

The `update_compatibility_matrix.yml` workflow:

- Runs weekly on Sunday at midnight UTC
- Can be manually triggered with customizable parameters:
  - `all_models`: Whether to include all models (not just key models)
  - `output_formats`: Output formats for the matrix (markdown, HTML, JSON)
  
The workflow performs the following tasks:
- Generates an updated compatibility matrix using `generate_enhanced_compatibility_matrix.py`
- Detects if there are changes to the matrix compared to the previous version
- Creates a pull request with the updated matrix if changes are detected
- Uploads the matrix files as artifacts

Key features:
- Automatic PR creation with descriptive title and body
- Support for multiple output formats (markdown, HTML, JSON)
- Option to include all models or just key models
- Integration with the DuckDB database for data-driven matrix generation

### Benchmark Database CI Workflow

The `benchmark_db_ci.yml` workflow:

- Runs on schedule (daily at midnight)
- Runs on push to main branch for relevant files
- Runs on pull requests for relevant files
- Can be manually triggered with customizable parameters:
  - `test_model`: The model to benchmark (default: all)
  - `hardware`: The hardware to benchmark on (cpu, cuda, all)
  - `batch_size`: Batch sizes to test (default: 1,2,4,8)

The workflow consists of the following jobs:

1. **setup_database**: Creates a new database for the benchmark run
   - Generates a unique run ID for tracking
   - Creates a database with the schema and sample data
   - Uploads the database as an artifact

2. **run_benchmarks**: Runs benchmarks for specified models and hardware
   - Creates a matrix of models and hardware to benchmark
   - Runs benchmarks using `run_benchmark_with_db.py`
   - Stores results directly in the database
   - Uploads the updated database as an artifact

3. **consolidate_results**: Processes benchmark results and generates reports
   - Uses `ci_benchmark_integrator.py` to consolidate results
   - Generates performance reports, compatibility matrix, and hardware comparison charts
   - Detects performance regressions using `benchmark_regression_detector.py`
   - Creates GitHub issues for severe regressions
   - Integrates with the hardware model predictor to validate predictions
   - Uploads the consolidated database and reports as artifacts

4. **publish_results**: Publishes results to GitHub Pages
   - Creates an index HTML file for navigation
   - Deploys all reports to GitHub Pages
   - Archives the database for historical data retention

## Integration with GitHub Pages

All CI/CD workflows are integrated with GitHub Pages to provide a permanent, accessible location for viewing reports:

1. **Report Types Published**:
   - Performance reports showing benchmark results across hardware platforms
   - Compatibility matrix showing model compatibility with different hardware
   - Regression reports highlighting performance regressions
   - Hardware comparison charts visualizing relative performance

2. **Access URL**: Reports are published to:
   `https://{organization}.github.io/{repository}/benchmark-reports/`

3. **Report Organization**:
   - Reports are organized by date with a central index page
   - The index page provides links to the latest reports
   - Historical reports are retained for trend analysis

4. **Automated Publishing**:
   - Reports are automatically published when workflows run on schedule or on push to main
   - Pull request runs generate reports but do not publish them
   - Manual workflow runs can optionally publish reports

## Usage

### Running Manually

To run the test results integration workflow manually:

1. Go to the Actions tab in the GitHub repository
2. Select "Test Results Integration" workflow
3. Click "Run workflow"
4. Configure the parameters:
   - **models**: Comma-separated list of models to test (e.g., "prajjwal1/bert-tiny,BAAI/bge-small-en-v1.5")
   - **hardware**: Comma-separated list of hardware platforms (e.g., "cpu,cuda")
   - **include_web**: Whether to include web platform tests (WebNN, WebGPU)
5. Click "Run workflow"

### Scheduled Runs

The workflow runs automatically:
- **Daily**: The test results integration workflow runs daily at 1 AM UTC
- **Weekly**: The compatibility matrix update workflow runs weekly on Sunday at midnight UTC

### Viewing Reports

Reports are published to GitHub Pages and can be accessed at:
`https://{organization}.github.io/{repository}/benchmark-reports/`

The following reports are available:
- Compatibility Matrix
- Performance Report
- Regression Analysis Report

## Performance Regression Detection

The system automatically detects performance regressions by:

1. Comparing current benchmark results with historical data
2. Identifying statistically significant degradations in performance metrics
3. Calculating severity based on the magnitude of the degradation
4. Creating GitHub issues for high and medium severity regressions

Regression detection parameters:
- **threshold**: Default is 0.1 (10% degradation)
- **window**: Default is 5 (compares with the last 5 runs)
- **metrics**: Throughput, latency, and memory usage

The regression detection system works as follows:

1. **Data Collection**: For each model-hardware-batch size combination, the system collects:
   - Current run performance metrics (throughput, latency, memory usage)
   - Historical performance metrics from previous runs (up to the specified window size)

2. **Statistical Analysis**:
   - Calculates mean and standard deviation of historical performance metrics
   - Computes the change ratio between current and historical mean
   - Calculates z-score to measure statistical significance
   - Ignores regressions with low statistical significance (z-score < 2)

3. **Severity Classification**:
   - **High**: Performance degradation > 20%
   - **Medium**: Performance degradation between 10-20%
   - **Low**: Performance degradation < 10% (not reported)

4. **Report Generation**:
   - Creates HTML and markdown reports with detailed regression information
   - Includes model, hardware, batch size, metric, change percentage, and severity
   - Provides statistical significance information

5. **GitHub Issue Creation** (for scheduled runs):
   - Creates GitHub issues for medium and high severity regressions
   - Includes detailed regression information and links to reports
   - Adds appropriate labels for automated tracking

Example regression report:
```
## Performance Regression Report

**Date:** 2025-03-07 10:15:22
**Total Regressions:** 2

### Summary of Regressions
| Model | Hardware | Batch Size | Metric | Change | Severity |
|-------|----------|------------|--------|--------|----------|
| bert-base-uncased | cuda | 8 | throughput | -15.34% | MEDIUM |
| t5-small | cuda | 16 | memory_peak | +22.67% | HIGH |

### Detailed Regression Information

#### Regression #1: bert-base-uncased on cuda
- **Metric:** throughput
- **Batch Size:** 8
- **Current Value:** 256.4321
- **Historical Mean:** 302.8756
- **Change:** -15.34%
- **Statistical Significance (Z-Score):** 3.45
- **Severity:** MEDIUM
- **Run ID:** 20250307101522
- **Timestamp:** 2025-03-07T10:15:22Z

#### Regression #2: t5-small on cuda
- **Metric:** memory_peak
- **Batch Size:** 16
- **Current Value:** 2.4563
- **Historical Mean:** 2.0025
- **Change:** +22.67%
- **Statistical Significance (Z-Score):** 4.21
- **Severity:** HIGH
- **Run ID:** 20250307101522
- **Timestamp:** 2025-03-07T10:15:22Z
```

## Database Integration

All test results are stored directly in the DuckDB database, which offers:

1. Efficient storage and querying capabilities
2. Historical data preservation
3. Comprehensive data analysis capabilities
4. Integration with visualization tools

The database schema includes:
- `test_results`: Stores basic test execution data
- `performance_results`: Stores performance metrics
- `hardware_compatibility`: Tracks hardware compatibility status
- `model_metadata`: Stores information about tested models

## Extending the CI/CD System

### Adding New Test Types

To add a new test type to the CI/CD system:

1. Update the test script to support database integration
2. Add a new step in the workflow YAML file
3. Update the report generation to include the new test type

### Custom Report Generation

To customize report generation:

1. Modify the scripts in the `test/scripts/` directory
2. Update the workflow steps that generate reports

### Local Testing

To test the CI/CD integration locally:

```bash
# Set environment variables
export BENCHMARK_DB_PATH=./benchmark_db_local.duckdb
export DEPRECATE_JSON_OUTPUT=1

# Run tests with database integration
python test/test_ipfs_accelerate.py --models prajjwal1/bert-tiny --endpoints cpu --db-path $BENCHMARK_DB_PATH

# Generate reports
python test/generate_compatibility_matrix.py --db-path $BENCHMARK_DB_PATH --format markdown --output compatibility_matrix.md
python test/scripts/benchmark_db_query.py --db $BENCHMARK_DB_PATH --report performance --format html --output performance_report.html

# Check for regressions
python test/scripts/benchmark_regression_detector.py --db $BENCHMARK_DB_PATH --threshold 0.1
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure all required Python packages are installed
2. **Database connection errors**: Check database path and permissions
3. **Incomplete results**: Verify that tests completed successfully
4. **Report generation failures**: Check log output for specific errors

### Viewing Workflow Logs

To diagnose issues:

1. Go to the Actions tab in the GitHub repository
2. Select the workflow run
3. Expand the job that failed
4. Review the logs for error messages

## Best Practices for CI/CD Integration

When working with the CI/CD integration system, follow these best practices:

1. **Test Changes Locally First**
   - Run tests locally with database integration before pushing changes
   - Use `export BENCHMARK_DB_PATH=./benchmark_db_local.duckdb` to set up local database
   - Verify that your changes don't cause performance regressions

2. **Review Workflow Runs**
   - Check workflow runs after pushing changes or creating PRs
   - Review test results and performance metrics
   - Address any failures or performance regressions

3. **Control Workflow Resources**
   - When running manually, limit the scope of tests to save resources
   - For exploratory testing, specify a small set of models and hardware platforms
   - Use the full test suite only when necessary

4. **Interpret Regression Reports Carefully**
   - Investigate all high-severity regressions
   - Consider the context of medium-severity regressions
   - Look for patterns across multiple models and hardware platforms
   - Check if regressions are related to recent code changes

5. **Manage GitHub Pages Content**
   - Archive or delete outdated reports 
   - Maintain a clean, organized structure for report pages
   - Update index pages to highlight the most relevant reports

## Configuring Scheduled Runs

The CI/CD integration system includes scheduled runs that automatically execute tests and benchmarks on a regular basis. The following schedules are configured:

1. **Daily Test Integration** (`test_results_integration.yml`)
   - Runs at 1 AM UTC daily
   - Tests a subset of models across all hardware platforms
   - Generates daily test reports and compatibility matrices
   - Checks for performance regressions compared to historical data

2. **Weekly Compatibility Matrix Update** (`update_compatibility_matrix.yml`)
   - Runs at midnight UTC on Sundays
   - Generates a comprehensive compatibility matrix for all models
   - Creates a PR with the updated matrix if changes are detected

To modify these schedules:

1. Edit the `cron` expression in the workflow YAML file
2. Use standard cron syntax: `minute hour day-of-month month day-of-week`
3. Consider time zones and resource availability when scheduling
4. Avoid scheduling multiple resource-intensive workflows at the same time

Example custom schedule (run at 3:30 PM UTC on weekdays):
```yaml
schedule:
  - cron: '30 15 * * 1-5'  # At 15:30 UTC, Monday through Friday
```

## Future Enhancements

The CI/CD integration system will be enhanced with the following features in the future:

1. **Advanced Dashboard Integration**
   - Interactive dashboard for test results and benchmarks
   - Trend visualization for performance metrics
   - Filtering and comparison tools for analysis

2. **Extended Test Coverage**
   - Mobile and edge device testing integration
   - Browser-specific test suite for web platforms
   - Cross-platform compatibility validation

3. **Enhanced Regression Analysis**
   - Machine learning-based regression detection
   - Root cause analysis for performance regressions
   - Predictive regression prevention

4. **Integration with Hardware-Aware Model Selection API**
   - Automated hardware recommendation validation
   - Performance prediction accuracy tracking
   - Optimization suggestion system

These enhancements will be implemented as part of the next development phase, focusing on the hardware-aware model selection API and interactive performance dashboard.

## Conclusion

The CI/CD integration system provides a robust framework for automated testing, result storage, report generation, and regression detection. By automating these processes, we can:

1. **Maintain Quality Standards**: Ensure consistent performance across all hardware platforms
2. **Detect Issues Early**: Identify and address performance regressions quickly
3. **Track Performance Trends**: Monitor performance improvements and regressions over time
4. **Document Compatibility**: Maintain up-to-date compatibility matrices for all models
5. **Support Decision Making**: Provide data-driven insights for hardware selection and optimization

This system forms the foundation for data-driven development and optimization of the IPFS Accelerate framework, enabling continuous improvement of hardware compatibility and performance.