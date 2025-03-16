# CI/CD Integration for Distributed Testing Framework

This document provides an overview of the CI/CD integration features available in the Distributed Testing Framework. These features allow test results to be reported to various CI/CD systems in a standardized way, with support for different output formats, artifact management, and PR/MR comments.

## Overview

The CI/CD integration system provides:

1. **Standardized Interfaces**: A common interface for all CI providers (GitHub, GitLab, Jenkins, etc.)
2. **Multiple Output Formats**: Support for Markdown, HTML, and JSON report formats
3. **Artifact Management**: Collection, storage, and uploading of test artifacts
4. **PR/MR Comments**: Automatic commenting on Pull Requests or Merge Requests
5. **Batch Task Processing**: Integration with the coordinator's batch task processing system
6. **Performance Metrics**: Tracking and reporting of performance metrics
7. **Comprehensive Reporting**: Detailed test result reports with pass/fail information, metrics, and more

## Architecture

The CI/CD integration system is built around these key components:

1. **`CIProviderInterface`**: Abstract base class defining the standard interface for all CI providers
2. **Provider Implementations**: Concrete implementations for various CI systems (GitHub, GitLab, Jenkins, etc.)
3. **`TestRunResult`**: Standardized representation of test run results
4. **`TestResultFormatter`**: Formats test results for different output formats
5. **`TestResultReporter`**: Reports test results to CI systems and generates reports
6. **`CIProviderFactory`**: Factory for creating CI provider instances

## Available CI Providers

The following CI providers are available:

| Provider | Class | Description |
|----------|-------|-------------|
| GitHub | `GitHubClient` | GitHub Actions and Checks API |
| GitLab | `GitLabClient` | GitLab CI/CD API |
| Jenkins | `JenkinsClient` | Jenkins API |
| Azure DevOps | `AzureDevOpsClient` | Azure DevOps API |
| CircleCI | `CircleCIClient` | CircleCI API |
| Travis CI | `TravisClient` | Travis CI API |
| TeamCity | `TeamCityClient` | TeamCity API |
| Bitbucket | `BitbucketClient` | Bitbucket Pipelines API |
| Local | `LocalCIProvider` | Local mode for testing |

## Usage

### Basic Usage with TestResultReporter

```python
from distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers

# Register all providers
register_all_providers()

# Create a CI provider
ci_config = {
    "token": "YOUR_GITHUB_TOKEN",
    "repository": "your-username/your-repo"
}
ci_provider = await CIProviderFactory.create_provider("github", ci_config)

# Create a test result reporter
reporter = TestResultReporter(
    ci_provider=ci_provider,
    report_dir="./reports",
    artifact_dir="./artifacts"
)

# Create a test result
test_result = TestRunResult(
    test_run_id="test-123",
    status="success",
    total_tests=42,
    passed_tests=40,
    failed_tests=1,
    skipped_tests=1,
    duration_seconds=125.7
)

# Add metadata
test_result.metadata = {
    "performance_metrics": {
        "average_throughput": 124.5,
        "average_latency_ms": 8.7
    }
}

# Generate reports
report_files = await reporter.report_test_result(
    test_result,
    formats=["markdown", "html", "json"]
)

# Collect and upload artifacts
artifacts = await reporter.collect_and_upload_artifacts(
    test_result.test_run_id,
    ["./artifacts/*.json", "./artifacts/*.log"]
)
```

### Integration with Coordinator

The CI/CD integration system integrates seamlessly with the coordinator's batch task processing system:

```python
# Create coordinator with batch processing
coordinator = DistributedTestingCoordinator(
    db_path="./coordinator.db",
    enable_batch_processing=True,
    batch_size_limit=5,
    model_grouping=True,
    hardware_grouping=True
)

# Create tasks and submit them to the coordinator
# ... (see examples for details)

# Create CI provider and reporter
# ... (see examples for details)

# Report results
# ... (see examples for details)
```

## Example Scripts

Several example scripts are provided to demonstrate the CI/CD integration features:

1. **`github_ci_integration_example.py`**: Example using GitHub CI integration
2. **`gitlab_ci_integration_example.py`**: Example using GitLab CI integration
3. **`generic_ci_integration_example.py`**: Generic example that works with any CI provider
4. **`ci_coordinator_batch_example.py`**: Example demonstrating integration with coordinator batch processing
5. **`worker_auto_discovery_with_ci.py`**: Example showing worker auto-discovery with CI/CD integration

### Running the Examples

```bash
# GitHub example (requires GITHUB_TOKEN environment variable)
python examples/github_ci_integration_example.py

# GitLab example (requires GITLAB_TOKEN environment variable)
python examples/gitlab_ci_integration_example.py

# Generic example with auto-detection of CI environment
python examples/generic_ci_integration_example.py --auto-detect

# Generic example with explicit provider and config file
python examples/generic_ci_integration_example.py --ci-provider github --config ci_config_example.json

# Coordinator batch processing example
python examples/ci_coordinator_batch_example.py --ci-provider local

# Worker auto-discovery example with 3 simulated workers
python examples/worker_auto_discovery_with_ci.py --workers 3 --ci-provider local
```

## Advanced Features

### Worker Auto-Discovery

The CI/CD integration system can be combined with the worker auto-discovery feature to create a fully automated distributed testing environment:

```python
# Create coordinator with worker auto-discovery
coordinator = DistributedTestingCoordinator(
    db_path="./coordinator.db",
    worker_auto_discovery=True,      # Enable worker auto-discovery
    discovery_interval=5,            # Check for new workers every 5 seconds
    auto_register_workers=True,      # Allow workers to auto-register
    enable_batch_processing=True     # Enable batch processing for efficiency
)

# Workers will automatically register with their hardware capabilities
worker = Worker(
    coordinator_url="http://coordinator-host:8080",
    worker_id="worker-1",
    capabilities=detect_hardware_capabilities(),  # Auto-detect capabilities
    auto_register=True                           # Enable auto-registration
)

# Tasks will be automatically assigned to workers with matching capabilities
# Results will be reported to CI systems as before
```

This enables:

1. **Dynamic Cluster Scaling**: Workers can join and leave the cluster at any time
2. **Hardware-Aware Task Assignment**: Tasks are assigned to workers with matching hardware capabilities
3. **Automated Reporting**: Test results are automatically collected and reported to CI systems
4. **Comprehensive Metrics**: Detailed metrics on worker utilization, task performance, and more

For a complete example, see `examples/worker_auto_discovery_with_ci.py`.

## Configuration

A sample configuration file (`ci_config_example.json`) is provided that shows the configuration options for all supported CI providers.

For most CI systems, the configuration can be picked up from environment variables if running in the actual CI environment.

### GitHub Configuration

```json
{
  "token": "YOUR_GITHUB_TOKEN",
  "repository": "your-username/your-repo",
  "commit_sha": "abcdef123456789",
  "pr_number": "42"
}
```

### GitLab Configuration

```json
{
  "token": "YOUR_GITLAB_TOKEN",
  "project_id": "12345678",
  "commit_sha": "abcdef7890123456",
  "mr_iid": "24"
}
```

## Auto-Detection of CI Environment

The system can auto-detect the CI environment based on environment variables set by the CI system. This allows for seamless integration with existing CI pipelines.

```python
provider_type, provider_config = detect_ci_environment()
if provider_type:
    ci_provider = await CIProviderFactory.create_provider(provider_type, provider_config)
```

## Report Formats

### Markdown Format

The Markdown format provides a clean, readable report suitable for GitHub, GitLab, and other systems that support Markdown.

Example:
```markdown
# Test Run Report: test-123

**Status:** SUCCESS

**Summary:**
- Total Tests: 42
- Passed: 40 (95.2%)
- Failed: 1 (2.4%)
- Skipped: 1 (2.4%)
- Duration: 2m 5.70s

## Performance Metrics

| Metric | Value |
|--------|-------|
| Average Throughput | 124.50 |
| Average Latency Ms | 8.70 |

## Test Details

### Failed Tests

| Test | Error | Duration |
|------|-------|----------|
| test_large_batch | CUDA out of memory | 3.20s |

### Passed Tests: 40

*Report generated on 2025-03-15 14:30:00*
```

### HTML Format

The HTML format provides a visually rich report with styling and progress bars, suitable for standalone viewing.

### JSON Format

The JSON format provides a machine-readable representation of the test results, suitable for programmatic processing.

## Developing New CI Provider Implementations

To implement a new CI provider:

1. Create a new class that inherits from `CIProviderInterface`
2. Implement all the required methods (see `api_interface.py` for details)
3. Register the provider with the factory in `register_providers.py`

Example:

```python
class MyCIProvider(CIProviderInterface):
    # Implement required methods
    # ...

# Register the provider
CIProviderFactory.register_provider("my-ci", MyCIProvider)
```

## Best Practices

1. **Use Environment Variables**: For security, use environment variables for tokens and other sensitive information
2. **Provide Meaningful Reports**: Include performance metrics and other relevant information in your reports
3. **Use Batch Processing**: When possible, use the coordinator's batch processing capabilities to optimize task execution
4. **Collect Artifacts**: Collect and upload relevant artifacts for later analysis
5. **Handle Failures Gracefully**: Ensure that your CI integration can handle failures and partial results

## Troubleshooting

Common issues and solutions:

1. **Authentication Failures**: Ensure your token has the necessary permissions
2. **Missing Repository**: Ensure the repository is correctly specified
3. **Connection Errors**: Check network connectivity and API endpoint URLs
4. **Rate Limiting**: Be aware of rate limits imposed by CI providers

## Future Enhancements

Planned future enhancements include:

1. **Enhanced Visualization**: More advanced visualization of test results
2. **Advanced Metrics**: More sophisticated performance metrics and analysis
3. **Historical Comparisons**: Comparison of test results with historical runs
4. **Extended Provider Support**: Support for additional CI providers
5. **Webhooks Integration**: Support for webhook-based integration with CI systems