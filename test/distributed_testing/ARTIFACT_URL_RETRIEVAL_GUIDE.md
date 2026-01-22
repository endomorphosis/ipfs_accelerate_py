# Artifact URL Retrieval Guide

This guide provides comprehensive documentation for the artifact URL retrieval system implemented across all CI providers in the Distributed Testing Framework.

## Overview

The artifact URL retrieval system provides a standardized way to access artifact URLs across different CI platforms, even when the underlying storage mechanisms vary significantly. This feature enables downstream components to retrieve artifact URLs without needing provider-specific code, facilitating the creation of rich reports, notifications, and dashboards that include links to test artifacts.

## Key Features

- **Universal Interface**: The same method works across all CI providers
- **Provider-Specific Logic**: Each provider implements URL retrieval according to its own artifact storage mechanisms
- **URL Caching**: Minimizes API calls by caching URLs for future requests
- **Robust Error Handling**: Comprehensive error handling with appropriate logging
- **Fallback Mechanisms**: Multiple URL resolution strategies when primary methods fail
- **Simulation Support**: Works with simulated test runs for development and testing

## Architecture

The artifact URL retrieval system is implemented as part of the CI/CD integration framework. Each CI provider implements the `get_artifact_url` method defined in the `CIProviderInterface` abstract base class.

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

## Provider Implementations

Each CI provider implements `get_artifact_url` with provider-specific logic:

### GitHub

- **URL Pattern**: `https://github.com/owner/repo/suites/{id}/artifacts/{artifact_id}`
- **API Endpoint**: GitHub Actions Artifacts API
- **Resolution Strategy**: 
  1. Check URL cache
  2. Query artifacts API with test run ID
  3. Retrieve artifact URL for specific artifact name

### GitLab

- **URL Pattern**: `https://gitlab.com/api/v4/projects/{project_id}/jobs/{job_id}/artifacts/{path}`
- **API Endpoint**: GitLab Jobs Artifacts API
- **Resolution Strategy**:
  1. Check URL cache
  2. Get job ID from test run ID
  3. Retrieve artifact URL using job ID and artifact name

### Jenkins

- **URL Pattern**: `https://jenkins.example.com/job/{job_name}/{build_id}/artifact/{path}`
- **API Endpoint**: Jenkins Artifacts API
- **Resolution Strategy**:
  1. Check URL cache
  2. Extract job name and build ID from test run ID
  3. Construct artifact URL using Jenkins artifact path pattern

### CircleCI

- **URL Pattern**: `https://circleci.com/api/v2/project/{project_slug}/{job_number}/artifacts/{path}`
- **API Endpoint**: CircleCI Artifacts API
- **Resolution Strategy**:
  1. Check URL cache
  2. Get project slug and job number from test run ID
  3. Retrieve artifact list for job
  4. Find artifact URL by name

### Azure DevOps

- **URL Pattern**: `https://dev.azure.com/{org}/{project}/_apis/test/runs/{run_id}/attachments/{id}`
- **API Endpoint**: Azure Test Attachments API
- **Resolution Strategy**:
  1. Check URL cache
  2. Get test run ID
  3. Retrieve attachment list for test run
  4. Find attachment ID by name
  5. Construct attachment URL

### TeamCity

- **URL Pattern**: `https://teamcity.example.com/app/rest/builds/id:{build_id}/artifacts/content/{path}`
- **API Endpoint**: TeamCity Artifacts API
- **Resolution Strategy**:
  1. Check URL cache
  2. Extract build ID from test run ID
  3. Get build details to retrieve build type ID
  4. Retrieve artifact list for build
  5. Find artifact URL by name

### Travis CI

- **URL Pattern**: `https://s3.amazonaws.com/travis-artifacts/{repo}/{build_id}/{artifact_name}`
- **Storage Mechanism**: Custom storage (typically S3)
- **Resolution Strategy**:
  1. Check URL cache
  2. Extract repository and build ID from test run ID
  3. Construct artifact URL using S3 pattern

### Bitbucket

- **URL Pattern**: `https://bitbucket.org/{workspace}/{repo}/downloads/{path}`
- **API Endpoint**: Bitbucket Downloads API
- **Resolution Strategy**:
  1. Check URL cache
  2. Extract workspace, repository, and report ID from test run ID
  3. Retrieve downloads list
  4. Find download URL by name
  5. Fallback to constructed URL if API search fails

## Implementation Details

Each provider implementation includes the following common elements:

### URL Caching

All providers implement an efficient URL caching mechanism to minimize API calls:

```python
# Check if URL is cached
if hasattr(self, "_artifact_urls") and test_run_id in self._artifact_urls and artifact_name in self._artifact_urls[test_run_id]:
    logger.info(f"Using cached artifact URL for {artifact_name}")
    return self._artifact_urls[test_run_id][artifact_name]

# [API calls to fetch URL]

# Cache the URL for future use
if not hasattr(self, "_artifact_urls"):
    self._artifact_urls = {}

if test_run_id not in self._artifact_urls:
    self._artifact_urls[test_run_id] = {}

self._artifact_urls[test_run_id][artifact_name] = artifact_url
```

### Error Handling

All implementations include robust error handling:

```python
try:
    # [API calls to fetch URL]
except Exception as e:
    logger.error(f"Exception getting artifact URL: {str(e)}")
    return None
```

### Simulation Support

All providers gracefully handle simulated test runs:

```python
if test_run_id.startswith("simulated-"):
    logger.warning(f"Cannot get artifact URL for simulated test run {test_run_id}")
    return None
```

### Fallback Mechanisms

Most providers implement fallback mechanisms when primary URL resolution methods fail:

```python
# Try primary method
if primary_method_succeeded:
    return primary_url

# Fallback if primary method fails
logger.warning("Primary method failed, using fallback approach")
fallback_url = construct_fallback_url()
return fallback_url
```

## Usage Examples

### Basic Usage

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
result = await provider.upload_artifact(
    test_run_id="test-123",
    artifact_path="./results.json",
    artifact_name="test_results.json"
)

# Get the artifact URL
if result:
    url = await provider.get_artifact_url(
        test_run_id="test-123",
        artifact_name="test_results.json"
    )
    
    if url:
        print(f"Artifact URL: {url}")
```

### Integration with Test Result Reporting

```python
from distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers

# Register all providers
register_all_providers()

# Create a CI provider
ci_provider = await CIProviderFactory.create_provider("github", {
    "token": "YOUR_GITHUB_TOKEN",
    "repository": "owner/repo"
})

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

# Retrieve artifact URLs for inclusion in reports
artifact_urls = {}
for artifact_name in artifacts:
    url = await ci_provider.get_artifact_url(test_result.test_run_id, artifact_name)
    if url:
        artifact_urls[artifact_name] = url

# Use artifact URLs in a PR comment
if artifact_urls:
    comment = "## Test Results\n\n"
    comment += f"Test run {test_result.test_run_id} completed with status: {test_result.status}\n\n"
    comment += "### Artifacts:\n\n"
    
    for name, url in artifact_urls.items():
        comment += f"- [{name}]({url})\n"
    
    await ci_provider.add_pr_comment("42", comment)
```

### Integration with Artifact Handler

```python
from distributed_testing.ci.artifact_handler import get_artifact_handler
from distributed_testing.ci.register_providers import register_all_providers
from distributed_testing.ci.api_interface import CIProviderFactory

# Register all providers
register_all_providers()

# Create provider
provider = await CIProviderFactory.create_provider("github", {
    "token": "YOUR_GITHUB_TOKEN",
    "repository": "owner/repo"
})

# Get artifact handler
artifact_handler = get_artifact_handler()
artifact_handler.register_provider("github", provider)

# Upload artifact
success, metadata = await artifact_handler.upload_artifact(
    source_path="./test_results.json",
    artifact_name="test_results.json",
    artifact_type="report",
    test_run_id="test-123",
    provider_name="github"
)

if success:
    # Get artifact URL
    url = await provider.get_artifact_url("test-123", "test_results.json")
    
    if url:
        # Store URL in metadata for future reference
        metadata.update({"url": url})
        artifact_handler.update_artifact_metadata("test-123", "test_results.json", metadata)
        
        # URL can now be used in reports, notifications, etc.
        print(f"Artifact available at: {url}")
```

## Testing

A dedicated test script is available to validate the artifact URL retrieval functionality across all CI providers:

```bash
# Run all tests
python distributed_testing/test_artifact_url_retrieval.py

# Run tests for a specific provider
python distributed_testing/test_artifact_url_retrieval.py --provider github

# Run tests with mock implementations
python distributed_testing/test_artifact_url_retrieval.py --mock-only

# Run with real CI configurations (requires auth tokens)
python distributed_testing/test_artifact_url_retrieval.py --real-providers --config ci_config.json
```

## Common Troubleshooting

### URL Not Found

If `get_artifact_url` returns `None`:

1. Verify the artifact was successfully uploaded (check `upload_artifact` return value)
2. Ensure the test run ID and artifact name match what was used during upload
3. Check if the CI provider's API is accessible and responding
4. Verify authentication credentials are valid
5. For some providers, ensure the artifact is still available (check retention policies)

### API Rate Limiting

Some CI providers may rate-limit API calls:

1. Implement suitable retry mechanisms with exponential backoff
2. Use the built-in URL caching to minimize API calls
3. Consider batching URL retrievals when possible

### URL Format Issues

If the URL format is unexpected:

1. Check CI provider documentation for changes to artifact URL patterns
2. Verify URL construction logic in the provider implementation
3. Test the URL directly to ensure it's accessible

## Best Practices

1. **Use URL Caching**: Always rely on the built-in URL caching to minimize API calls
2. **Handle Absent URLs**: Always check if `get_artifact_url` returns `None` and handle appropriately
3. **Secure Credentials**: Ensure CI provider credentials are securely stored and managed
4. **Include URLs in Reports**: Include artifact URLs in test reports for easy access
5. **Metadata Storage**: Store URLs as part of artifact metadata for future reference
6. **Error Handling**: Implement proper error handling for URL retrieval failures

## Integration with TestResultReporter

The artifact URL retrieval system is now fully integrated with the `TestResultReporter` class, enabling automatic inclusion of artifact URLs in test reports, PR comments, and dashboards.

### Key Features

- **Automatic URL Retrieval**: URLs for uploaded artifacts are automatically retrieved and included in reports
- **Bulk URL Retrieval**: Efficient batch retrieval of multiple artifact URLs using asynchronous parallel processing
- **Report Formatting**: URLs are properly formatted in Markdown, HTML, and JSON reports
- **PR Comments with URLs**: PR comments include direct links to artifacts
- **Parallel Processing**: URLs are retrieved in parallel for better performance
- **URL Caching**: Intelligent caching minimizes redundant API calls
- **Graceful Degradation**: Robust error handling with fallback mechanisms when URL retrieval fails
- **Cross-Provider Compatibility**: Works consistently across all supported CI providers

### Key Implementation Components

The TestResultReporter integration relies on three main components:

1. **`get_artifact_urls` Method**: A new method that efficiently retrieves multiple artifact URLs in parallel
2. **Enhanced `collect_and_upload_artifacts`**: Updated to automatically retrieve URLs for uploaded artifacts
3. **Updated `report_test_result`**: Enhanced to include artifact URLs in generated reports and PR comments

#### The `get_artifact_urls` Method

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

### Integration Workflow

The complete integration workflow follows these steps:

1. **Artifact Collection and Upload**:
   ```python
   # Collect and upload artifacts
   artifacts = await reporter.collect_and_upload_artifacts(
       test_run_id="test-123",
       artifact_patterns=["./results/*.json", "./logs/*.log"]
   )
   ```

2. **Automatic URL Retrieval**:
   The `collect_and_upload_artifacts` method now automatically retrieves URLs for each successfully uploaded artifact using the CI provider's `get_artifact_url` method.

3. **URL Addition to Metadata**:
   URLs are added to the artifact metadata and returned as part of the artifacts list:
   ```python
   # Artifacts list now contains URLs
   for artifact in artifacts:
       print(f"Artifact: {artifact['name']}, URL: {artifact.get('url', 'No URL')}")
   ```

4. **Report Generation with URLs**:
   ```python
   # Add artifacts to test result metadata
   test_result.metadata["artifacts"] = artifacts
   
   # Generate reports with artifact URLs included
   report_files = await reporter.report_test_result(
       test_result,
       formats=["markdown", "html", "json"]
   )
   ```

5. **Automatic URL Inclusion**:
   URLs are automatically included in all generated reports:
   - Markdown reports show URLs as clickable links in an Artifacts section
   - HTML reports show URLs as hyperlinks in the Artifacts section
   - JSON reports include URLs in the artifacts section of the metadata

### Usage Example

```python
from distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers

# Register all providers
register_all_providers()

# Create a CI provider
provider = await CIProviderFactory.create_provider("github", {
    "token": "YOUR_TOKEN",
    "repository": "owner/repo"
})

# Create a test result reporter
reporter = TestResultReporter(
    ci_provider=provider,
    report_dir="./reports",
    artifact_dir="./artifacts"
)

# Create a test result
test_result = TestRunResult(
    test_run_id="test-123",
    status="success",
    total_tests=10,
    passed_tests=9,
    failed_tests=1,
    skipped_tests=0,
    duration_seconds=45.6,
    metadata={
        "pr_number": "123",
        "performance_metrics": {
            "average_throughput": 125.4,
            "average_latency_ms": 7.9
        }
    }
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
```

### Bulk URL Retrieval

The `TestResultReporter` class now includes a `get_artifact_urls` method for efficient bulk retrieval of artifact URLs:

```python
# Retrieve multiple artifact URLs in a single operation
artifact_urls = await reporter.get_artifact_urls(
    test_run_id="test-123",
    artifact_names=["test_results.json", "performance_metrics.csv", "test_log.txt"]
)

# Use the URLs
for name, url in artifact_urls.items():
    if url:
        print(f"Artifact {name} is available at: {url}")
```

### Enhanced Reports with URLs

#### Markdown Report Example

```markdown
# Test Run Report: test-123

**Status:** SUCCESS

**Summary:**
- Total Tests: 10
- Passed: 9 (90.0%)
- Failed: 1 (10.0%)
- Skipped: 0 (0.0%)
- Duration: 45.60s

## Performance Metrics

| Metric | Value |
|--------|-------|
| Average Throughput | 125.40 |
| Average Latency Ms | 7.90 |

## Artifacts

- [Test Results JSON](https://github.com/owner/repo/actions/runs/123/artifacts/456) (2.3 KB)
- [Performance Metrics CSV](https://github.com/owner/repo/actions/runs/123/artifacts/457) (1.5 KB)
- [Test Log](https://github.com/owner/repo/actions/runs/123/artifacts/458) (5.7 KB)

*Report generated on 2025-03-16 10:30:45*
```

#### HTML Report Artifacts Section

```html
<h2>Artifacts</h2>
<ul>
  <li><a href="https://github.com/owner/repo/actions/runs/123/artifacts/456">Test Results JSON</a> (2.3 KB)</li>
  <li><a href="https://github.com/owner/repo/actions/runs/123/artifacts/457">Performance Metrics CSV</a> (1.5 KB)</li>
  <li><a href="https://github.com/owner/repo/actions/runs/123/artifacts/458">Test Log</a> (5.7 KB)</li>
</ul>
```

### PR Comments with Artifact URLs

The `report_test_result` method automatically includes artifact URLs in PR comments when a PR number is provided in the test result metadata:

```python
# Test result with PR number
test_result.metadata["pr_number"] = "123"

# Generate reports and add PR comment with artifact URLs
await reporter.report_test_result(test_result)
```

This will create a PR comment like:

```markdown
## Test Run Results: test-123

**Status:** SUCCESS

**Summary:**
- Total Tests: 10
- Passed: 9 (90.0%)
- Failed: 1 (10.0%)
- Skipped: 0 (0.0%)
- Duration: 45.60s

### Artifacts
- [Test Results JSON](https://github.com/owner/repo/actions/runs/123/artifacts/456) (2.3 KB)
- [Performance Metrics CSV](https://github.com/owner/repo/actions/runs/123/artifacts/457) (1.5 KB)
- [Test Log](https://github.com/owner/repo/actions/runs/123/artifacts/458) (5.7 KB)
```

### Complete Example

A complete example demonstrating all features of the integration is available in the following file:

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
    
# Run with provider-specific configuration file
python distributed_testing/examples/reporter_artifact_url_example.py \
    --provider github \
    --config provider_config.json
```

### Implementation Details

The integration with `TestResultReporter` handles several edge cases and optimizations:

1. **Provider Not Supporting URL Retrieval**: If the CI provider doesn't support the `get_artifact_url` method, the reporter gracefully handles this by logging a warning and continuing without URLs.

2. **URL Retrieval Failures**: If URL retrieval fails for some artifacts, the reporter logs the errors and continues with the available URLs, maintaining smooth operation.

3. **Parallel Processing**: URL retrieval operations are performed in parallel using asyncio tasks, significantly improving performance when retrieving multiple URLs.

4. **Report URLs**: The reports themselves are uploaded as artifacts, and their URLs are included in the test result metadata, enabling easy access to reports.

5. **Fallback Mechanisms**: When a URL cannot be retrieved, the reporter uses fallback mechanisms including placeholder URLs when appropriate.

6. **Report Formatting**: URLs are formatted appropriately in each report format, ensuring proper display and accessibility.

7. **Cross-Provider Consistency**: The integration works consistently across all supported CI providers despite their different URL patterns and retrieval mechanisms.

## Integration with Distributed Testing Framework

The artifact URL retrieval system is fully integrated with the Distributed Testing Framework, enabling seamless artifact access across all components of the framework.

### Key Integration Points

1. **Coordinator Integration**:
   - Artifact URLs included in task results sent to the coordinator
   - Dashboard displays incorporate clickable artifact links
   - Task metadata includes artifact URLs for downstream components
   - Batch task processing system preserves artifact URLs

2. **Worker Integration**:
   - Workers can report test results with artifact URLs
   - Artifact URLs accessible to workers through task metadata
   - Multi-worker environments benefit from URL-based artifact sharing
   - Worker auto-discovery preserves artifact URL functionality

3. **Result Aggregation**:
   - Aggregated results maintain artifact URLs
   - Reports generated from aggregated results include clickable links
   - Historical result storage preserves artifact URLs

4. **Dashboard Integration**:
   - Dashboard views display artifact URLs as clickable links
   - Interactive dashboards provide direct artifact access
   - Trend analysis preserves artifact URL accessibility

### Implementation Details

The integration has been implemented through these key components:

1. **Coordinator Integration**:
   ```python
   # Coordinator processes test results with artifact URLs
   test_result = TestRunResult(
       test_run_id="test-123",
       status="success",
       total_tests=10,
       passed_tests=9,
       failed_tests=1,
       skipped_tests=0,
       duration_seconds=15.5,
       metadata={
           "artifacts": [
               {
                   "name": "test_results.json",
                   "path": "/artifacts/test_results.json",
                   "size_bytes": 1024,
                   "url": "https://github.com/owner/repo/actions/runs/123/artifacts/456"
               }
           ]
       }
   )
   
   # Send test result to coordinator
   await coordinator.process_test_result(test_result)
   
   # Artifact URLs are preserved in task metadata
   task = await coordinator.get_task(task_id)
   artifact_urls = [a["url"] for a in task["result_metadata"]["artifacts"]]
   ```

2. **Dashboard Integration**:
   ```python
   # Generate dashboard with artifact URLs
   dashboard_items = await coordinator.get_dashboard_items(limit=10)
   
   for item in dashboard_items:
       if "artifacts" in item["result_metadata"]:
           artifact_urls = [(a["name"], a["url"]) for a in item["result_metadata"]["artifacts"]]
           # Use artifact URLs in dashboard display
   ```

3. **Worker Integration**:
   ```python
   # Worker retrieves task with artifact URLs
   task = await worker.get_task()
   
   # Access artifact URLs from task metadata
   artifact_urls = {}
   if "artifacts" in task["metadata"]:
       for artifact in task["metadata"]["artifacts"]:
           artifact_urls[artifact["name"]] = artifact["url"]
   ```

### DTF Integration Example

A complete example demonstrating integration with the Distributed Testing Framework is available in the following file:

```
distributed_testing/examples/enhanced_reporter_artifact_url_example.py
```

This example demonstrates:
1. Creating a coordinator with batch processing
2. Registering and updating tasks
3. Collecting and uploading artifacts with automatic URL retrieval
4. Sending test results with artifact URLs to the coordinator
5. Retrieving task details with artifact URLs
6. Generating dashboard reports with artifact URLs

To run the example:

```bash
# Run with DTF integration
python distributed_testing/examples/enhanced_reporter_artifact_url_example.py
```

The output will include sections demonstrating integration with the Distributed Testing Framework, showing how artifact URLs are preserved throughout the entire testing pipeline.

### Implementation Benefits

The integration provides several benefits:

1. **Simplified Artifact Access**: Artifacts are easily accessible through URLs
2. **Improved Reporting**: Reports include clickable links to artifacts
3. **Cross-Component Access**: Artifacts can be accessed by any component through URLs
4. **Efficient Distribution**: URL-based access enables efficient artifact distribution
5. **Centralized Storage**: Artifacts can be stored centrally and accessed via URLs
6. **Security Improvements**: URL-based access can include security features
7. **Performance Benefits**: Parallel URL retrieval improves performance
8. **Reduced API Calls**: Smart caching reduces API calls to CI providers
9. **Enhanced Visibility**: Artifacts are more visible in reports and dashboards

## Comprehensive Testing

A comprehensive test suite verifies the integration between the artifact URL retrieval system and the Distributed Testing Framework:

```
distributed_testing/test_reporter_artifact_integration.py
```

This test suite includes:
1. Tests for basic URL retrieval across all CI providers
2. Tests for automatic URL retrieval in `collect_and_upload_artifacts`
3. Tests for artifact URL inclusion in test reports
4. Tests for PR comment integration with artifact URLs
5. Performance testing for parallel URL retrieval
6. Edge case testing for various scenarios
7. Integration testing with the Distributed Testing Framework coordinator

To run the test suite:

```bash
# Run the full test suite
python distributed_testing/test_reporter_artifact_integration.py

# Run with specific test focus
python distributed_testing/test_reporter_artifact_integration.py --focus dtf-integration
```

## URL Validation

The artifact URL retrieval system now includes a comprehensive URL validation system for verifying that artifact URLs remain accessible. This feature provides valuable health monitoring and validation capabilities.

### URL Validation Integration

The URL validation system is integrated with TestResultReporter:

1. **URL Validation in get_artifact_urls**:
   ```python
   # Validate URLs during retrieval
   urls = await reporter.get_artifact_urls(
      test_run_id="test-123",
      artifact_names=["artifact1.json", "artifact2.log"],
      validate=True  # Enable validation
   )
   ```

2. **URL Validation in collect_and_upload_artifacts**:
   ```python
   # Collect artifacts with URL validation
   artifacts = await reporter.collect_and_upload_artifacts(
      test_run_id="test-123",
      artifact_patterns=["./artifacts/*.json"],
      validate_urls=True,  # Enable validation
      include_health_info=True  # Include health metrics
   )
   ```

3. **Artifact Metadata with Validation**:
   ```python
   # Example artifact with validation info
   {
      "name": "test_results.json",
      "path": "/path/to/artifact.json",
      "size_bytes": 1024,
      "url": "https://github.com/owner/repo/actions/runs/123/artifacts/456",
      "url_validated": True,
      "url_valid": True,
      "url_health": {
         "url": "https://github.com/owner/repo/actions/runs/123/artifacts/456",
         "is_valid": True,
         "last_checked": 1647408123.45,
         "status_code": 200,
         "availability": 100.0
      }
   }
   ```

### Core Validation Features

The URL validator provides the following core features:

1. **Validation**: Verify that URLs are accessible using HTTP HEAD requests
2. **Parallel Processing**: Validate multiple URLs in parallel for efficiency
3. **Caching**: Cache validation results to minimize external requests
4. **Health Monitoring**: Track URL health over time with periodic checks
5. **Health Reporting**: Generate health reports in multiple formats
6. **Graceful Degradation**: Proper error handling and fallback mechanisms

### Usage Examples

See the comprehensive [ENHANCED_ARTIFACT_URL_RETRIEVAL.md](docs/ENHANCED_ARTIFACT_URL_RETRIEVAL.md) for detailed documentation on the URL validation system, including usage examples, integration patterns, and health monitoring capabilities.

For a complete example implementation, see `distributed_testing/examples/enhanced_reporter_artifact_url_example.py`.

## Future Enhancements

Planned enhancements to the artifact URL retrieval system include:

1. **URL Signing**: Support for signed URLs with expiration for secure access
2. **Extended Provider Support**: Support for additional CI providers 
3. **Artifact Discovery**: More advanced artifact discovery capabilities
4. **Configuration Options**: More fine-grained control over URL inclusion
5. **URL Metrics**: Add metrics on URL access patterns and performance
6. **Advanced Caching Strategies**: Implement more sophisticated caching strategies for URL retrieval
7. **URL Health Dashboard**: Create a dashboard for monitoring URL health and accessibility
8. **Artifact Lifecycle Management**: Add lifecycle management for artifacts and their URLs

## Conclusion

The artifact URL retrieval system provides a robust, standardized way to access artifact URLs across different CI platforms. By implementing this functionality across all CI providers and integrating it with the Distributed Testing Framework, the system enables rich reporting, notification, and visualization features that include direct links to test artifacts, enhancing the overall user experience of the framework.

The integration with the Distributed Testing Framework ensures that artifact URLs are preserved throughout the entire testing pipeline, from test execution to result reporting and analysis, creating a seamless experience for both developers and test system administrators.