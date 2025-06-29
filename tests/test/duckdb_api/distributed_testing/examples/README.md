# Distributed Testing Framework - CI/CD Integration Examples

This directory contains example configuration files and code samples for integrating the Distributed Testing Framework with various CI/CD systems.

## GitHub Actions Integration

The `github_workflow.yml` file demonstrates how to set up GitHub Actions to run distributed tests:

- **Workflow Triggers**: Run on push, pull request, or manual dispatch
- **Test Discovery**: Automatically find and analyze test files
- **Hardware Routing**: Route tests to appropriate workers based on hardware requirements
- **Result Collection**: Aggregate and report test results
- **Artifact Generation**: Generate comprehensive test reports and artifacts

To use this example:
1. Copy `github_workflow.yml` to your repository's `.github/workflows/distributed-testing.yml`
2. Configure coordinator URL and API key as GitHub secrets
3. Customize test patterns and other parameters as needed

## GitLab CI Integration

The `gitlab-ci.yml` file demonstrates how to set up GitLab CI to run distributed tests:

- **Pipeline Stages**: Set up proper pipeline stages for test preparation, execution, and reporting
- **Variable Management**: Manage coordinator URL and API key via GitLab CI/CD variables
- **Job Definitions**: Define test jobs with appropriate settings
- **Artifact Configuration**: Configure artifact collection and expiration
- **Custom Runner Tags**: Route tests to specific runners with hardware capabilities

To use this example:
1. Copy `gitlab-ci.yml` to your repository's `.gitlab-ci.yml`
2. Configure coordinator URL and API key as GitLab CI/CD variables
3. Update test patterns and other parameters as needed

## Jenkins Integration

Two example files are provided for Jenkins integration:

### Basic Jenkinsfile

The `Jenkinsfile` demonstrates a basic Jenkins pipeline for distributed testing:

- **Docker-Based Execution**: Run in a Python Docker container
- **Parameter Management**: Define parameters for coordinator URL, API key, etc.
- **Test Execution**: Run distributed tests via the CI/CD integration module
- **Artifact Collection**: Archive test reports and visualizations

### Enhanced Jenkinsfile

The `enhanced_jenkinsfile` demonstrates a more advanced Jenkins pipeline:

- **Dynamic Agent Selection**: Select appropriate agent based on hardware requirements
- **Local Testing Environment**: Optional setup of a local coordinator and workers
- **Parallel Testing**: Execute different types of tests in parallel
- **Comprehensive Reporting**: Generate detailed reports and visualizations
- **Test Type Selection**: Choose specific types of tests to run
- **Advanced Cleanup**: Proper cleanup of resources after testing

To use these examples:
1. Copy either `Jenkinsfile` or `enhanced_jenkinsfile` to your repository's `Jenkinsfile`
2. Configure coordinator URL and API key as Jenkins credentials
3. Update agent labels, test patterns, and other parameters as needed

## Additional Examples

- `cross_platform_worker_example.py`: Demonstrates how to use the Cross-Platform Worker Support module
- `error_handling_demo.py`: Shows how to handle different types of errors in a CI/CD context
- `generate_and_submit_tests.py`: Example script for generating and submitting tests
- `dashboard_example.py`: Demonstrates how to use the dashboard in a CI/CD workflow
- `high_availability_cluster.sh`: Script for setting up a high-availability coordinator cluster
- `performance_analysis.sh`: Script for analyzing performance results

## Usage Notes

1. **Security**: Always store API keys and coordinator URLs securely in your CI/CD system's secrets or credentials store.
2. **Timeout Management**: Set appropriate timeouts based on your test execution times.
3. **Hardware Requirements**: Ensure your CI/CD runners have access to the required hardware for your tests.
4. **Artifact Retention**: Configure appropriate retention periods for test artifacts.
5. **Resource Management**: Be mindful of resource consumption during testing, especially for long-running or resource-intensive tests.

## Integration with Manual Testing

These examples can be adapted for manual testing scenarios:

```bash
# Run tests with local coordinator and workers
./high_availability_cluster.sh --local --workers 2

# Generate and submit tests from a manual process
python generate_and_submit_tests.py --test-dir ./tests --output-dir ./results

# View test results in the dashboard
python dashboard_example.py --port 8080
```

## Documentation

For comprehensive documentation on CI/CD integration, refer to the following:

- [CI/CD Integration Guide](../CI_CD_INTEGRATION_GUIDE.md): Detailed guide for all CI/CD integrations
- [CI/CD Integration Summary](../CI_CD_INTEGRATION_SUMMARY.md): Summary of implementation status and features
- [Distributed Testing Design](../DISTRIBUTED_TESTING_DESIGN.md): Overall design and architecture of the framework