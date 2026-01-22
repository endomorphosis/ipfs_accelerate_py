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
7. **`artifact_handler.py`**: Centralized artifact management across providers
8. **Artifact URL Retrieval**: Cross-provider mechanism to access artifact URLs

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

# Retrieve artifact URLs for inclusion in reports or notifications
artifact_urls = {}
for artifact_name in artifacts:
    url = await ci_provider.get_artifact_url(test_result.test_run_id, artifact_name)
    if url:
        artifact_urls[artifact_name] = url
        
# Use the URLs in reports, notifications, or dashboards
for name, url in artifact_urls.items():
    print(f"Artifact '{name}' available at: {url}")
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
6. **`reporter_artifact_url_example.py`**: Example demonstrating integration of TestResultReporter with artifact URL retrieval

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

## Artifact URL Retrieval System

The CI/CD integration system provides a comprehensive artifact URL retrieval system that works across all supported CI providers. This feature enables downstream components to access artifacts without needing provider-specific code.

### Key Features

- **Universal Interface**: The same method works across all providers
- **URL Caching**: Minimizes API calls by caching URLs
- **Error Handling**: Robust error handling with appropriate logging
- **Fallback Mechanisms**: Multiple URL resolution strategies
- **Simulation Support**: Works even with simulated test runs
- **Bulk URL Retrieval**: Efficient batch retrieval of multiple artifact URLs
- **Reporter Integration**: Automatic inclusion of artifact URLs in test reports
- **Parallel Processing**: Asynchronous processing for improved performance
- **Cross-Provider Compatibility**: Consistent behavior across all CI providers

### Enhanced TestResultReporter with Artifact URL Integration

The `TestResultReporter` class now fully integrates with the artifact URL retrieval system, enabling automatic inclusion of artifact URLs in test reports, PR comments, and dashboards.

#### Key Integration Components

1. **Enhanced `collect_and_upload_artifacts` Method**: Automatically retrieves URLs for uploaded artifacts
2. **New `get_artifact_urls` Method**: Efficiently retrieves multiple artifact URLs in parallel
3. **Updated `report_test_result` Method**: Includes artifact URLs in reports and PR comments

```python
# Create a reporter
reporter = TestResultReporter(
    ci_provider=ci_provider,
    report_dir="./reports",
    artifact_dir="./artifacts"
)

# Collect and upload artifacts with automatic URL retrieval
artifacts = await reporter.collect_and_upload_artifacts(
    test_run_id="test-123",
    artifact_patterns=["./results/*.json", "./logs/*.log"]
)

# Add artifacts to test result metadata
test_result.metadata["artifacts"] = artifacts

# Generate reports with artifact URLs included
report_files = await reporter.report_test_result(
    test_result,
    formats=["markdown", "html", "json"]
)

# Bulk retrieve multiple artifact URLs in a single operation
artifact_urls = await reporter.get_artifact_urls(
    test_run_id="test-123",
    artifact_names=["test_results.json", "performance_metrics.csv", "test_log.txt"]
)
```

#### Bulk URL Retrieval Implementation

The `get_artifact_urls` method retrieves multiple URLs in parallel using asyncio tasks:

```python
async def get_artifact_urls(self, test_run_id: str, artifact_names: List[str]) -> Dict[str, Optional[str]]:
    """
    Retrieve URLs for multiple artifacts in bulk.
    
    This method efficiently retrieves URLs for multiple artifacts in a single operation,
    which is more efficient than retrieving them one by one.
    
    Args:
        test_run_id: Test run ID
        artifact_names: List of artifact names
        
    Returns:
        Dictionary mapping artifact names to their URLs (or None if not found)
    """
    if not self.ci_provider or not hasattr(self.ci_provider, 'get_artifact_url'):
        logger.warning("CI provider doesn't support get_artifact_url method")
        return {name: None for name in artifact_names}
    
    # Create tasks for retrieving URLs in parallel
    tasks = []
    for name in artifact_names:
        task = asyncio.create_task(self.ci_provider.get_artifact_url(test_run_id, name))
        tasks.append((name, task))
    
    # Wait for all tasks to complete
    urls = {}
    for name, task in tasks:
        try:
            url = await task
            urls[name] = url
        except Exception as e:
            logger.error(f"Error retrieving artifact URL for {name}: {str(e)}")
            urls[name] = None
    
    return urls
```

#### Integration Benefits

1. **Automatic URL Retrieval**: URLs for uploaded artifacts are automatically retrieved and included in reports
2. **Efficient Batch Processing**: Multiple URLs are retrieved in parallel for better performance
3. **Rich Reporting**: URLs are properly formatted in all output formats (Markdown, HTML, JSON)
4. **PR Comment Enhancement**: PR comments include direct links to artifacts
5. **Report Artifact URLs**: Report artifacts themselves have accessible URLs included in metadata
6. **Graceful Degradation**: Robust error handling ensures failures don't disrupt the testing process
7. **URL Caching**: Minimizes redundant API calls by caching previous URL retrievals
8. **Universal Interface**: Works the same way across all supported CI providers

### Report Formats with Artifact URLs

#### Markdown Reports

```markdown
# Test Run Report: test-123

**Status:** SUCCESS

**Summary:**
- Total Tests: 10
- Passed: 9 (90.0%)
- Failed: 1 (10.0%)
- Skipped: 0 (0.0%)
- Duration: 45.60s

## Artifacts

- [Test Results JSON](https://github.com/owner/repo/actions/runs/123/artifacts/456) (2.3 KB)
- [Performance Metrics CSV](https://github.com/owner/repo/actions/runs/123/artifacts/457) (1.5 KB)
- [Test Log](https://github.com/owner/repo/actions/runs/123/artifacts/458) (5.7 KB)
```

#### HTML Reports

```html
<h2>Artifacts</h2>
<ul>
  <li><a href="https://github.com/owner/repo/actions/runs/123/artifacts/456">Test Results JSON</a> (2.3 KB)</li>
  <li><a href="https://github.com/owner/repo/actions/runs/123/artifacts/457">Performance Metrics CSV</a> (1.5 KB)</li>
  <li><a href="https://github.com/owner/repo/actions/runs/123/artifacts/458">Test Log</a> (5.7 KB)</li>
</ul>
```

#### PR Comments

```markdown
## Test Run Results: test-123

**Status:** SUCCESS

### Artifacts
- [Test Results JSON](https://github.com/owner/repo/actions/runs/123/artifacts/456) (2.3 KB)
- [Performance Metrics CSV](https://github.com/owner/repo/actions/runs/123/artifacts/457) (1.5 KB)
- [Test Log](https://github.com/owner/repo/actions/runs/123/artifacts/458) (5.7 KB)
```

### Implementation Details

Each CI provider implements the `get_artifact_url` method following this signature:

```python
async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
    """
    Get the URL for a test run artifact.
    
    Args:
        test_run_id: Test run ID
        artifact_name: Name of artifact
        
    Returns:
        URL to the artifact or None if not found
    """
```

#### Provider-Specific URL Patterns

Different CI providers use different URL patterns and APIs for artifacts:

| Provider | URL Mechanism | URL Pattern Example |
|----------|---------------|---------------------|
| GitHub | GitHub API | `https://github.com/owner/repo/suites/{id}/artifacts/{artifact_id}` |
| GitLab | GitLab Jobs Artifacts | `https://gitlab.com/api/v4/projects/{project_id}/jobs/{job_id}/artifacts/{path}` |
| Jenkins | Jenkins Artifacts | `https://jenkins.example.com/job/{job_name}/{build_id}/artifact/{path}` |
| CircleCI | CircleCI Artifacts API | `https://circleci.com/api/v2/project/{project_slug}/{job_number}/artifacts/{path}` |
| Azure DevOps | Test Attachments API | `https://dev.azure.com/{org}/{project}/_apis/test/runs/{run_id}/attachments/{id}` |
| TeamCity | TeamCity Artifacts API | `https://teamcity.example.com/app/rest/builds/id:{build_id}/artifacts/content/{path}` |
| Travis CI | Custom storage (e.g. S3) | `https://s3.amazonaws.com/travis-artifacts/{repo}/{build_id}/{artifact_name}` |
| Bitbucket | Bitbucket Downloads API | `https://bitbucket.org/{workspace}/{repo}/downloads/{path}` |

### Usage Example

```python
from distributed_testing.ci.api_interface import CIProviderFactory
from distributed_testing.ci.register_providers import register_all_providers

# Register all providers
register_all_providers()

# Create a provider
provider = await CIProviderFactory.create_provider("github", {
    "token": "YOUR_GITHUB_TOKEN",
    "repository": "owner/repo"
})

# Upload an artifact
await provider.upload_artifact(
    test_run_id="test-123",
    artifact_path="./results.json",
    artifact_name="test_results.json"
)

# Get the artifact URL
url = await provider.get_artifact_url(
    test_run_id="test-123",
    artifact_name="test_results.json"
)

if url:
    print(f"Artifact URL: {url}")
    
    # URL can be used in reports, notifications, etc.
    report_html = f"""
    <html>
    <body>
        <h1>Test Report</h1>
        <p>Test run completed successfully.</p>
        <p>Artifacts:</p>
        <ul>
            <li><a href="{url}">Test Results JSON</a></li>
        </ul>
    </body>
    </html>
    """
```

### Complete Example

A complete example demonstrating all features of the integration is available:

```
distributed_testing/examples/reporter_artifact_url_example.py
```

This example demonstrates:
1. Creating a test result
2. Generating reports in multiple formats
3. Uploading and collecting artifacts
4. Retrieving artifact URLs
5. Including artifact URLs in test reports and PR comments

To run the example:

```bash
# Run with mock setup (no actual CI provider)
python distributed_testing/examples/reporter_artifact_url_example.py

# Run with a specific CI provider
python distributed_testing/examples/reporter_artifact_url_example.py \
    --provider github \
    --token YOUR_TOKEN \
    --repository owner/repo
```

### Integration with Artifact Discovery

The artifact URL retrieval system integrates with the artifact discovery system:

```python
from distributed_testing.ci.artifact_discovery import discover_artifacts

# Discover artifacts from a test run
artifacts = await discover_artifacts(
    test_run_id="test-123",
    provider="github",
    config={
        "token": "YOUR_GITHUB_TOKEN",
        "repository": "owner/repo"
    }
)

# Get URLs for all discovered artifacts
for artifact in artifacts:
    url = await provider.get_artifact_url(
        test_run_id="test-123",
        artifact_name=artifact["name"]
    )
    
    if url:
        artifact["url"] = url
```

### Testing the Artifact URL Retrieval System

A dedicated test script is available for testing the artifact URL retrieval system:

```bash
# Run the artifact URL retrieval test
python distributed_testing/test_artifact_url_retrieval.py

# Run tests for a specific provider
python distributed_testing/test_artifact_url_retrieval.py --provider github

# Run tests with real CI providers (requires auth tokens)
python distributed_testing/test_artifact_url_retrieval.py --real-providers --config ci_config.json
```

For detailed documentation, see the comprehensive [ARTIFACT_URL_RETRIEVAL_GUIDE.md](ARTIFACT_URL_RETRIEVAL_GUIDE.md).

## Future Enhancements

Planned future enhancements include:

1. **Enhanced Visualization**: More advanced visualization of test results
2. **Advanced Metrics**: More sophisticated performance metrics and analysis
3. **Historical Comparisons**: Comparison of test results with historical runs
4. **Extended Provider Support**: Support for additional CI providers
5. **Webhooks Integration**: Support for webhook-based integration with CI systems
6. **Enhanced Artifact Discovery**: More advanced artifact discovery capabilities
7. **Bulk URL Retrieval**: Optimized batch retrieval of multiple artifact URLs
8. **URL Validation and Health Checks**: Validate artifact URLs and monitor availability