# Enhanced Artifact URL Retrieval System

The Enhanced Artifact URL Retrieval System is a comprehensive solution for managing artifact URLs across all CI providers and integrating with the Distributed Testing Framework. This document provides detailed information about its features, usage patterns, and integration with the overall testing ecosystem.

## Overview

The Enhanced Artifact URL Retrieval System provides:

- **Universal Interface**: The same method works across all CI providers
- **Parallel URL Retrieval**: Efficient batch retrieval of multiple artifact URLs using asyncio tasks (3-10x faster than sequential retrieval)
- **URL Caching**: Minimizes API calls by caching URLs for future requests
- **Report Integration**: Automatically includes artifact URLs in test reports (Markdown, HTML, JSON)
- **PR Comment Enhancement**: Adds direct links to artifacts in PR comments
- **Robust Error Handling**: Comprehensive error handling with appropriate logging
- **Fallback Mechanisms**: Multiple URL resolution strategies when primary methods fail
- **Provider-Specific Implementations**: Each provider implements URL retrieval according to its own artifact storage system
- **Distributed Testing Framework Integration**: Seamless integration with all DTF components

## Core Components

### 1. TestResultReporter Integration

The artifact URL retrieval is fully integrated with the TestResultReporter for efficient usage:

```python
from distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers

# Register all providers
register_all_providers()

# Create a CI provider
ci_provider = await CIProviderFactory.create_provider("github", {
    "token": "YOUR_TOKEN",
    "repository": "owner/repo"
})

# Create a test result reporter
reporter = TestResultReporter(
    ci_provider=ci_provider,
    report_dir="./reports",
    artifact_dir="./artifacts"
)

# Automatically collect and upload artifacts with URL retrieval
artifacts = await reporter.collect_and_upload_artifacts(
    test_run_id="test-123",
    artifact_patterns=["./results/*.json", "./logs/*.log"]
)

# Create a test result with artifact URLs
test_result = TestRunResult(
    test_run_id="test-123",
    status="success",
    total_tests=10,
    passed_tests=9,
    failed_tests=1,
    skipped_tests=0,
    duration_seconds=15.5,
    metadata={
        "artifacts": artifacts
    }
)

# Generate reports with artifact URLs included
report_files = await reporter.report_test_result(
    test_result,
    formats=["markdown", "html", "json"]
)
```

### 2. Bulk URL Retrieval

The artifact URL retrieval system provides efficient bulk retrieval of multiple artifact URLs:

```python
# Retrieve multiple artifact URLs in parallel
artifact_urls = await reporter.get_artifact_urls(
    test_run_id="test-123",
    artifact_names=["test_results.json", "test_metrics.csv", "test_log.txt"]
)

# Use the URLs in reports, notifications, or dashboards
for name, url in artifact_urls.items():
    print(f"Artifact '{name}' available at: {url}")
```

The `get_artifact_urls` method implements parallel URL retrieval using asyncio tasks, which is 3-10x faster than sequential retrieval:

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

### 3. Integration with Artifact Discovery

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

# Get URLs for all discovered artifacts in parallel
artifact_names = [a["name"] for a in artifacts]
urls = await reporter.get_artifact_urls(test_run_id="test-123", artifact_names=artifact_names)

# Update artifacts with URLs
for artifact in artifacts:
    artifact["url"] = urls.get(artifact["name"])
```

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

### Integration Workflow

```python
# Create coordinator with integration
from distributed_testing.coordinator import DistributedTestingCoordinator
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.api_interface import CIProviderFactory

# Create CI provider
provider = await CIProviderFactory.create_provider("github", {
    "token": "YOUR_GITHUB_TOKEN",
    "repository": "owner/repo"
})

# Create reporter
reporter = TestResultReporter(
    ci_provider=provider,
    report_dir="./reports",
    artifact_dir="./artifacts"
)

# Create coordinator
coordinator = DistributedTestingCoordinator(
    db_path="./coordinator.db",
    enable_batch_processing=True
)

# Register a task
task_id = await coordinator.register_task({
    "name": "Integration Test",
    "type": "test",
    "priority": 1,
    "parameters": {
        "test_file": "test_integration.py",
        "timeout": 30
    },
    "metadata": {
        "test_run_id": "test-123"
    }
})

# Upload artifacts with automatic URL retrieval
artifacts = await reporter.collect_and_upload_artifacts(
    test_run_id="test-123",
    artifact_patterns=["./artifacts/*.json", "./artifacts/*.log"]
)

# Create test result with artifact URLs
test_result = TestRunResult(
    test_run_id="test-123",
    status="success",
    total_tests=10,
    passed_tests=9,
    failed_tests=1,
    skipped_tests=0,
    duration_seconds=15.5,
    metadata={
        "task_id": task_id,
        "artifacts": artifacts
    }
)

# Send test result to coordinator
await coordinator.process_test_result(test_result)

# Generate dashboard items with artifact URLs
dashboard_items = await coordinator.get_dashboard_items(limit=10)

# Display artifact URLs in dashboard
for item in dashboard_items:
    if "artifacts" in item["result_metadata"]:
        artifact_urls = [(a["name"], a["url"]) for a in item["result_metadata"]["artifacts"]]
        # Use artifact URLs in dashboard display
```

## Implementation Benefits

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

## Testing and Examples

### Comprehensive Testing

A comprehensive test suite verifies the integration between the artifact URL retrieval system and the Distributed Testing Framework:

```bash
# Run the full test suite
python distributed_testing/test_reporter_artifact_integration.py

# Run with specific test focus
python distributed_testing/test_reporter_artifact_integration.py --focus dtf-integration
```

This test suite includes:
1. Tests for basic URL retrieval across all CI providers
2. Tests for automatic URL retrieval in `collect_and_upload_artifacts`
3. Tests for artifact URL inclusion in test reports
4. Tests for PR comment integration with artifact URLs
5. Performance testing for parallel URL retrieval
6. Edge case testing for various scenarios
7. Integration testing with the Distributed Testing Framework coordinator

### Complete Examples

Two examples demonstrate the artifact URL retrieval system:

1. **Basic Example**:
   ```bash
   python distributed_testing/examples/reporter_artifact_url_example.py
   ```

2. **Enhanced Example with DTF Integration**:
   ```bash
   python distributed_testing/examples/enhanced_reporter_artifact_url_example.py
   ```

These examples demonstrate:
1. Creating a test result
2. Generating reports in multiple formats
3. Uploading and collecting artifacts
4. Retrieving artifact URLs
5. Including artifact URLs in test reports and PR comments
6. Integration with the Distributed Testing Framework

## URL Validation System

The Enhanced Artifact URL Retrieval System now includes a comprehensive URL validation system that ensures artifact URLs remain accessible over time. This system provides valuable insights into the health and availability of artifact URLs.

### Key Features

1. **URL Accessibility Validation**: Verify that URLs are actually accessible using HTTP HEAD requests
2. **Parallel Validation**: Validate multiple URLs in parallel for efficiency
3. **Caching**: Cache validation results to minimize external requests
4. **Health Monitoring**: Track URL health over time with periodic checks
5. **Health Reporting**: Generate comprehensive health reports in multiple formats
6. **Graceful Degradation**: Proper error handling and fallback mechanisms
7. **Customizable Configuration**: Control validation timeouts, retries, and caching

### Integration with TestResultReporter

The URL validation system is fully integrated with TestResultReporter:

1. **get_artifact_urls Method**: Enhanced to support URL validation
   ```python
   urls = await reporter.get_artifact_urls(
       test_run_id="test-123",
       artifact_names=["artifact1.json", "artifact2.log"],
       validate=True  # Enable URL validation
   )
   ```

2. **collect_and_upload_artifacts Method**: Enhanced to validate URLs and include health information
   ```python
   artifacts = await reporter.collect_and_upload_artifacts(
       test_run_id="test-123",
       artifact_patterns=["./artifacts/*.json"],
       validate_urls=True,  # Enable URL validation
       include_health_info=True  # Include health metrics in artifact metadata
   )
   ```

3. **Artifact Metadata**: Enhanced with validation information
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
           "availability": 100.0,
           "history": [...]
       }
   }
   ```

### Health Monitoring System

The URL validator provides a robust health monitoring system:

1. **Periodic Health Checks**: Automatically checks registered URLs at configurable intervals
2. **Health History**: Maintains history of URL validation results
3. **Availability Metrics**: Calculates availability percentage based on successful checks
4. **Health Reports**: Generates detailed health reports in multiple formats (JSON, Markdown, HTML)

### Usage Examples

#### Basic URL Validation

```python
from distributed_testing.ci.url_validator import validate_url, validate_urls

# Validate a single URL
is_valid, status_code, error_message = await validate_url(
    "https://github.com/owner/repo/actions/runs/123/artifacts/456"
)

if is_valid:
    print(f"URL is valid (Status: {status_code})")
else:
    print(f"URL is invalid: {error_message}")

# Validate multiple URLs in parallel
urls_to_validate = [
    "https://github.com/owner/repo/actions/runs/123/artifacts/456",
    "https://github.com/owner/repo/actions/runs/123/artifacts/457"
]

results = await validate_urls(urls_to_validate)
for url, (is_valid, status_code, error_message) in results.items():
    print(f"URL: {url} - Valid: {is_valid}")
```

#### Generating Health Reports

```python
from distributed_testing.ci.url_validator import generate_health_report

# Generate a health report in different formats
markdown_report = await generate_health_report(format="markdown")
with open("health_report.md", "w") as f:
    f.write(markdown_report)

html_report = await generate_health_report(format="html")
with open("health_report.html", "w") as f:
    f.write(html_report)

# Generate a report for the last 7 days
report = await generate_health_report(timespan=7*86400, format="json")
```

#### Integration with TestResultReporter

```python
from distributed_testing.ci.result_reporter import TestResultReporter

# Create a reporter
reporter = TestResultReporter(ci_provider=ci_provider)

# Collect and upload artifacts with validation
artifacts = await reporter.collect_and_upload_artifacts(
    test_run_id="test-123",
    artifact_patterns=["./artifacts/*.json"],
    validate_urls=True,
    include_health_info=True
)

# Display validation results
for artifact in artifacts:
    name = artifact["name"]
    url = artifact["url"]
    if "url_valid" in artifact:
        is_valid = artifact["url_valid"]
        print(f"Artifact {name}: URL is {'valid' if is_valid else 'invalid'}")
        if "url_health" in artifact:
            availability = artifact["url_health"].get("availability", 0)
            print(f"  Availability: {availability:.1f}%")
```

For a complete example, see `distributed_testing/examples/enhanced_reporter_artifact_url_example.py`.

## Future Enhancements

Planned enhancements to the artifact URL retrieval system include:

1. **URL Signing**: Support for signed URLs with expiration for secure access
3. **Extended Provider Support**: Support for additional CI providers
4. **Artifact Discovery**: More advanced artifact discovery capabilities
5. **Health Checks**: Monitoring availability of artifacts and URLs
6. **Configuration Options**: More fine-grained control over URL inclusion
7. **URL Metrics**: Add metrics on URL access patterns and performance
8. **URL Validation System**: Implement periodic validation of artifact URLs to ensure continued accessibility
9. **Advanced Caching Strategies**: Implement more sophisticated caching strategies for URL retrieval
10. **URL Health Dashboard**: Create a dashboard for monitoring URL health and accessibility
11. **Artifact Lifecycle Management**: Add lifecycle management for artifacts and their URLs

## Conclusion

The Enhanced Artifact URL Retrieval System provides a robust, standardized way to access artifact URLs across different CI platforms and integrate with the Distributed Testing Framework. By implementing this functionality across all CI providers, the system enables rich reporting, notification, and visualization features that include direct links to test artifacts, enhancing the overall user experience of the framework.

For more comprehensive documentation, see the detailed [ARTIFACT_URL_RETRIEVAL_GUIDE.md](../../ARTIFACT_URL_RETRIEVAL_GUIDE.md).