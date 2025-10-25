# CI/CD Integration Guide for Distributed Testing Framework

This guide explains how to integrate the Distributed Testing Framework's end-to-end tests with your CI/CD pipeline. The framework provides comprehensive testing of the coordinator, dynamic resource manager, performance trend analyzer, and workers as a complete system.

## Table of Contents

- [Overview](#overview)
- [GitHub Actions Integration](#github-actions-integration)
- [Workflows](#workflows)
- [Test Types](#test-types)
- [Configuration](#configuration)
- [Visualizations](#visualizations)
- [Badges](#badges)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

The CI/CD integration for the Distributed Testing Framework provides:

1. **Automated Testing**: Runs tests on push, pull request, and schedule
2. **Comprehensive Coverage**: Tests different components and their interactions
3. **Resource Scaling Validation**: Validates dynamic scaling behavior
4. **Performance Metrics**: Collects and analyzes performance data
5. **Visualizations**: Generates graphs and charts for test results
6. **Status Badges**: Provides test status badges for your repository
7. **Coverage Reports**: Generates test coverage reports

## GitHub Actions Integration

The framework includes GitHub Actions workflows that can be added to your repository. The main workflow file is `distributed-testing-e2e.yml`, which defines the jobs, steps, and triggers for running the tests.

### Installation

1. Ensure the `.github/workflows/distributed-testing-e2e.yml` file is in your repository
2. Configure the workflow according to your needs (see [Configuration](#configuration))
3. Push the changes to your repository

### Self-Hosted Runner Setup

**IMPORTANT**: If using self-hosted runners for testing, you must add the runner user to the docker group:

```bash
sudo usermod -aG docker <runner-user>
```

After adding the user to the docker group, either log out and back in, or restart the runner service:

```bash
sudo systemctl restart actions-runner
```

For complete self-hosted runner setup instructions, including hardware-specific configurations, see [SELF_HOSTED_RUNNER_SETUP.md](../../docs/SELF_HOSTED_RUNNER_SETUP.md).

## Workflows

The CI/CD integration includes the following workflows:

### `distributed-testing-e2e.yml`

Runs end-to-end integration tests for the Distributed Testing Framework.

**Triggers**:
- Push to `main` branch affecting files in `test/distributed_testing/**`
- Pull requests to `main` branch affecting files in `test/distributed_testing/**`
- Manual trigger via GitHub Actions UI

**Jobs**:
1. **prepare**: Sets up the test matrix
2. **test**: Runs the tests based on the test matrix
3. **report**: Generates reports and visualizations

## Test Types

The CI/CD integration supports the following test types:

1. **End-to-End Tests (e2e)**: Tests the complete system with all components
2. **Component Tests**: Tests individual components in isolation
3. **Integration Tests**: Tests interactions between specific components

You can configure which test types to run using the `test_type` input parameter.

## Configuration

### Workflow Inputs

When manually triggering the workflow, you can provide the following inputs:

- **test_type**: Type of tests to run (e2e, component, integration, all)
- **test_filter**: Test filter pattern (e.g., `TestE2EIntegratedSystem*`)
- **hardware**: Hardware to test on (comma-separated: cpu,cuda,webgpu)
- **timeout**: Test timeout in seconds
- **visualize**: Whether to generate visualization reports

### Environment Variables

You can configure the following environment variables in your GitHub repository:

- **COORDINATOR_URL**: URL of a remote coordinator to test against (optional)
- **COORDINATOR_API_KEY**: API key for authentication (optional)

If these are not provided, the workflow will start a local coordinator and worker for testing.

### Example: Manually Triggering Tests

To run only the end-to-end tests with a longer timeout:

1. Go to your repository on GitHub
2. Click "Actions"
3. Select "Distributed Testing E2E Integration Tests"
4. Click "Run workflow"
5. Set:
   - **test_type**: e2e
   - **timeout**: 3600
   - **visualize**: true
6. Click "Run workflow"

## Visualizations

The CI/CD integration generates several visualizations to help analyze test results:

1. **Performance Summary**: Shows execution time statistics for tests
2. **Component Interaction Graph**: Visualizes interactions between components
3. **Resource Usage**: Shows CPU, memory, task, and worker counts over time
4. **Scaling Analysis**: Visualizes dynamic resource scaling behavior
5. **Test Results Summary**: Shows pass rates and execution times by test type

These visualizations are saved as image files and included in an HTML dashboard.

### Accessing Visualizations

After a workflow run completes:

1. Go to the workflow run on GitHub Actions
2. Click on the "Artifacts" section
3. Download the "test-visualizations" artifact
4. Extract the archive and open `dashboard.html` in a web browser

## Badges

The CI/CD integration generates status badges for your repository:

- **Combined Tests**: Overall status of all tests
- **E2E Tests**: Status of end-to-end tests
- **Component Tests**: Status of component tests
- **Integration Tests**: Status of integration tests

These badges are automatically updated and committed to your repository.

### Adding Badges to READMEs

To add badges to your README files, include the following markdown:

```markdown
![Combined Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/distributed_testing/.github/badges/combined-status.json)
![E2E Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/distributed_testing/.github/badges/e2e-status.json)
![Component Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/distributed_testing/.github/badges/component-status.json)
![Integration Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/distributed_testing/.github/badges/integration-status.json)
```

Replace `username/repo` with your GitHub username and repository name.

## Troubleshooting

### Common Issues

1. **Tests fail with "Timeout exceeded"**:
   - Increase the timeout value in the workflow dispatch
   - Consider optimizing your tests for performance

2. **Worker connection issues**:
   - Check network configuration
   - Verify the coordinator URL is correct
   - Ensure the API key is correctly set

3. **Missing or incomplete test results**:
   - Check that your tests properly report results
   - Ensure your tests include required metadata

### Debugging

For detailed debugging:

1. Check the workflow logs on GitHub Actions
2. Download test artifacts for detailed error messages
3. Run the tests locally to diagnose issues

## Best Practices

1. **Run Tests Regularly**: Schedule tests to run regularly to catch regressions
2. **Isolate Tests**: Make tests independent to avoid interference
3. **Use Mocks**: Mock external services to avoid dependencies
4. **Include Metadata**: Ensure tests include proper metadata for visualizations
5. **Use Temp Directories**: Use temporary directories for test files
6. **Cleanup Resources**: Properly clean up resources in tearDown methods
7. **Monitor Performance**: Watch for performance regressions over time
8. **Keep Tests Fast**: Optimize tests for quick execution in CI
9. **Include Assertions**: Add detailed assertions for clear failure messages
10. **Update Expected Results**: Update expected results when behavior changes

By following this guide, you can effectively integrate the Distributed Testing Framework with your CI/CD pipeline, ensuring the reliability and performance of your system.