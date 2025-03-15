# CI/CD Integration for Simulation Validation Framework

This document explains the CI/CD integration for the Simulation Accuracy and Validation Framework. The integration ensures that validation results are automatically processed, analyzed, and published with every code change or on-demand.

## CI/CD Workflow Overview

The automated workflow consists of three main jobs:

1. **Validation**: Runs the simulation validation framework against different hardware profiles and generates validation results
2. **Analysis**: Processes the validation results to detect issues, analyze coverage, and generate summaries
3. **Dashboard**: Builds an interactive dashboard for visualizing the validation results and publishes it to GitHub Pages

![CI/CD Integration Diagram](../../../docs/images/simulation_validation_ci_diagram.png)

## GitHub Actions Workflow

The GitHub Actions workflow is defined in the `.github/workflows/simulation_validation_ci.yml` file. It triggers on:

- Pushes to the `main` branch affecting the simulation validation code
- Pull requests targeting the `main` branch affecting the simulation validation code
- Manual workflow dispatch with customizable parameters

### Manual Workflow Dispatch

The workflow supports manual triggering with the following parameters:

- **Test Type**: Type of test to run (all, unit, e2e, calibration, drift, visualization)
- **Hardware Profile**: Hardware profile to validate (all, cpu, gpu, webgpu)

These parameters allow targeted validation runs for specific testing needs.

### Validation Job

The validation job performs the following steps:

1. Checks out the repository code
2. Sets up Python environment and installs dependencies
3. Runs the validation tests based on the specified parameters
4. Generates validation reports in multiple formats
5. Uploads test results and validation output as artifacts

```yaml
validate-simulation-framework:
  runs-on: ubuntu-latest
  outputs:
    test_results_path: ${{ steps.run_tests.outputs.test_results_path }}
    validation_timestamp: ${{ steps.timestamp.outputs.timestamp }}
    validation_run_id: ${{ steps.runid.outputs.run_id }}
  steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    # Install dependencies
    # Run tests and validation
    # Generate validation report
    # Upload artifacts
```

### Analysis Job

The analysis job processes the validation results to extract insights:

1. Downloads the test results and validation output artifacts
2. Analyzes test coverage from the coverage XML file
3. Analyses validation results to extract key metrics and patterns
4. Detects validation issues and classifies them by severity
5. Creates GitHub issues for high-severity validation problems

```yaml
analyze-results:
  needs: validate-simulation-framework
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    
    # Download artifacts
    # Analyze test coverage
    # Analyze validation results
    # Detect validation issues
    # Create GitHub issues if needed
```

### Dashboard Job

The dashboard job builds an interactive visualization dashboard and publishes it:

1. Generates a comprehensive dashboard from the validation results
2. Creates an index page with links to various reports
3. Deploys the dashboard to GitHub Pages for easy access

```yaml
build-dashboard:
  needs: [validate-simulation-framework, analyze-results]
  runs-on: ubuntu-latest
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  steps:
    - uses: actions/checkout@v3
    
    # Download artifacts
    # Generate dashboard
    # Create index file
    # Deploy to GitHub Pages
```

## Components Used in CI/CD Integration

### 1. Validation Test Runner

The framework includes a test runner script (`run_e2e_tests.py`) that executes validation tests for specified hardware profiles and collects the results.

```bash
python -m duckdb_api.simulation_validation.run_e2e_tests \
  --hardware-profile gpu \
  --run-id 20250714123456 \
  --output-dir validation_output \
  --html-report
```

### 2. Test Coverage Analyzer

The test coverage analyzer (`analyze_test_coverage.py`) processes the coverage data from pytest-cov and generates comprehensive reports.

```bash
python -m duckdb_api.simulation_validation.analyze_test_coverage \
  --coverage-file test_results/coverage.xml \
  --output-format markdown
```

Key features:
- Overall coverage percentage calculation
- Component-level coverage breakdown
- Identification of low-coverage modules
- Multiple report formats (text, markdown, HTML, JSON)

### 3. Validation Results Analyzer

The validation results analyzer (`analyze_validation_results.py`) processes validation results to extract insights.

```bash
python -m duckdb_api.simulation_validation.analyze_validation_results \
  --results-dir validation_output \
  --output-format markdown
```

Key features:
- Overall validation accuracy metrics
- Hardware-model pair performance analysis
- Best and worst performing pairs identification
- Metric-level analysis across hardware profiles

### 4. Validation Issue Detector

The validation issue detector (`detect_validation_issues.py`) analyzes validation results to identify potential problems.

```bash
python -m duckdb_api.simulation_validation.detect_validation_issues \
  --results-dir validation_output \
  --threshold 0.1 \
  --output-format markdown
```

Key features:
- Detection of high-error hardware-model pairs
- Identification of metrics with consistently high errors
- Analysis of unusual error distributions
- Severity classification (high, medium, low)
- Detailed recommendations for addressing each issue

### 5. Dashboard Generator

The dashboard generator (`generate_dashboard.py`) creates an interactive visualization dashboard from validation results.

```bash
python -m duckdb_api.simulation_validation.visualization.generate_dashboard \
  --input-dir validation_output \
  --output-dir dashboard \
  --run-id 20250714123456 \
  --interactive
```

Key features:
- Comprehensive validation overview
- Interactive charts and visualizations
- Hardware comparison heatmaps
- Metric performance charts
- Drift detection visualizations
- Validation history tracking

## GitHub Pages Integration

The workflow automatically deploys the generated dashboard to GitHub Pages, making it accessible at:

```
https://{owner}.github.io/{repo}/simulation-validation/
```

The dashboard includes:
- Validation Report: Comprehensive validation results
- Calibration Report: Calibration effectiveness metrics
- Drift Detection Report: Drift analysis results
- Visualization Gallery: Collection of interactive visualizations
- Hardware Profiles: Performance by hardware profile
- Performance Analysis: Detailed performance metrics

## GitHub Issue Creation

For high-severity validation issues, the workflow automatically creates a GitHub issue with:
- Detailed description of each issue
- Severity classification
- Affected hardware-model pairs
- Specific metrics with problems
- Recommendations for addressing the issues

## Setting Up CI/CD Integration

To enable the CI/CD integration:

1. Ensure the GitHub Actions workflow file is in the correct location:
   ```
   .github/workflows/simulation_validation_ci.yml
   ```

2. Install the required Python dependencies:
   ```
   pip install -r duckdb_api/simulation_validation/requirements.txt
   ```

3. Set up GitHub Pages for your repository:
   - Go to repository settings
   - Navigate to Pages section
   - Select the gh-pages branch as the source

4. Grant the necessary permissions for issue creation:
   - Go to repository settings
   - Navigate to Actions > General
   - Under "Workflow permissions", select "Read and write permissions"

## Running the CI/CD Workflow Manually

To manually trigger the CI/CD workflow:

1. Go to the repository on GitHub
2. Navigate to the Actions tab
3. Select "Simulation Accuracy and Validation CI" from the workflows list
4. Click "Run workflow"
5. Configure the parameters:
   - Test Type: Select the type of test to run
   - Hardware Profile: Select the hardware profile to validate
6. Click "Run workflow" to start the process

## Interpreting CI/CD Results

### GitHub Actions Output

The workflow provides detailed step summaries for each job:
- Test coverage results
- Validation summary statistics
- Detected issues with severity classifications
- Links to generated reports and dashboards

### GitHub Pages Dashboard

The deployed dashboard provides an interactive interface for exploring validation results:
- Navigate between different visualization types
- Filter results by hardware, model, or metric
- View detailed statistics and performance metrics
- Track validation history over time

### GitHub Issues

Created issues contain structured information about validation problems:
- Issue title indicates the type of problem and timestamp
- Issue body contains detailed breakdown of each problem
- Labels categorize issues by type and severity
- Recommendations provide guidance for addressing issues

## Extending the CI/CD Integration

### Adding New Hardware Profiles

To add support for new hardware profiles:

1. Update the workflow dispatch inputs in the workflow file
2. Add the new hardware profile to the test data generators
3. Extend the reporting components to include the new profile

### Adding Custom Validation Protocols

To implement new validation protocols:

1. Add the protocol to the validation methodology
2. Create test cases for the new protocol
3. Update the test runner to support the protocol
4. Extend the reporting to include protocol-specific metrics

### Customizing Issue Creation

To customize the GitHub issue creation:

1. Modify the issue detector's severity classification logic
2. Update the issue reporting format
3. Customize the labels and categories used for issues

## Troubleshooting CI/CD Integration

### Common Issues

1. **Missing Dependencies**
   - Ensure all required packages are listed in requirements.txt
   - Check for platform-specific dependencies

2. **Failed Test Execution**
   - Check test logs for specific errors
   - Verify test environment setup
   - Look for missing test data or configuration

3. **Dashboard Deployment Failures**
   - Verify GitHub Pages setup
   - Check file paths and references in the generated HTML
   - Ensure interactive visualizations are properly configured

4. **Issue Creation Problems**
   - Check workflow permissions for issue creation
   - Verify JSON parsing in the issue creation script
   - Ensure issue templates are valid

### Getting Help

For assistance with CI/CD integration:
- Check the documentation in this repository
- Report issues in the GitHub repository
- Contact the Simulation Validation Framework team

## Conclusion

The CI/CD integration for the Simulation Accuracy and Validation Framework provides a robust system for automatically validating, analyzing, and reporting on simulation accuracy. It ensures that validation results are consistently processed and made accessible through interactive dashboards, helping to maintain high-quality simulation outputs.

The automated workflow streamlines the validation process, reduces manual effort, and provides early detection of simulation accuracy issues, contributing to the overall reliability of the framework.