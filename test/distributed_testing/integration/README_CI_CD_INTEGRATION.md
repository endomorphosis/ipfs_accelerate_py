# CI/CD Integration Plugin for Distributed Testing Framework

## Overview

The CI/CD Integration Plugin provides a comprehensive integration with popular CI/CD systems, enabling seamless reporting, artifact management, and test result analysis. This plugin serves as the bridge between the Distributed Testing Framework and various continuous integration platforms, standardizing the interaction and enhancing the capabilities across all platforms.

## Supported CI/CD Systems

- GitHub Actions
- Jenkins
- GitLab CI
- Azure DevOps
- CircleCI
- Travis CI
- Bitbucket Pipelines
- TeamCity
- Local environment (with file-based storage)

## Key Features

- **Standardized API**: Unified interface for interacting with all CI/CD systems
- **Automatic Environment Detection**: Detects CI environment and configures automatically
- **Test Run Management**: Creates and manages test runs in CI systems
- **Status Updates**: Provides real-time status updates to CI systems
- **Result Reporting**: Generates comprehensive reports in multiple formats (JSON, XML, HTML, Markdown)
- **Artifact Management**: Uploads and organizes test artifacts with categorization
- **Pull Request Integration**: Automatic comments on pull requests with test results
- **Advanced Failure Analysis**: Detailed analysis of test failures with classification
- **Test History Tracking**: Records test run history for trend analysis
- **Performance Metrics**: Tracks and analyzes performance metrics over time
- **Customizable Notifications**: Configurable notification templates and channels
- **Dashboard Integration**: Visualizes test results in a dashboard
- **Fault Tolerance**: Retry mechanism for API calls with exponential backoff

## Configuration Options

The plugin supports extensive configuration options to customize its behavior:

```python
config = {
    # CI System Configuration
    "ci_system": "auto",  # auto, github, jenkins, gitlab, azure, circle, travis, bitbucket, teamcity
    "api_url": None,      # CI system API URL
    "api_token": None,    # API token for authentication
    
    # Update Configuration
    "update_interval": 60,                    # Status update interval in seconds
    "update_on_completion_only": False,       # Only update status on completion
    "enable_status_updates": True,            # Enable CI status updates
    "status_update_format": "detailed",       # minimal, basic, detailed
    
    # PR Integration
    "enable_pr_comments": True,               # Enable PR comments
    "pr_comment_on_failure_only": False,      # Only comment on PR if tests fail
    "pr_comment_template": "default",         # default, minimal, detailed
    "pr_update_existing_comments": True,      # Update existing comments instead of creating new ones
    
    # Artifact Management
    "enable_artifacts": True,                 # Enable artifact upload
    "artifact_dir": "distributed_test_results", # Directory for storing artifacts
    "artifact_retention_days": 30,            # Number of days to retain artifacts
    "artifact_categories": ["reports", "logs", "data", "metrics"], # Categories of artifacts
    "artifact_compression": True,             # Compress artifacts before upload
    
    # Result Reporting
    "result_format": "all",                   # junit, json, html, markdown, all
    "include_system_info": True,              # Include system information in reports
    "include_failure_analysis": True,         # Include failure analysis in reports
    "include_performance_metrics": True,      # Include performance metrics in reports
    
    # History Tracking
    "enable_history_tracking": True,          # Enable test history tracking
    "history_retention_days": 90,             # Number of days to retain history
    "track_performance_trends": True,         # Track performance trends
    
    # Notifications
    "enable_notifications": False,            # Enable notifications
    "notification_channels": [],              # email, slack, teams, discord
    "notification_on_failure_only": True,     # Only send notifications on failure
    
    # Advanced Options
    "enable_failure_analysis": True,          # Enable failure analysis
    "failure_analysis_depth": "detailed",     # basic, detailed, comprehensive
    "dashboard_integration": False,           # Enable dashboard integration
    "dashboard_url": None,                    # Dashboard URL
    "detailed_logging": False,                # Enable detailed logging
    "retry_failed_api_calls": True,           # Retry failed API calls
    "max_retries": 3,                         # Maximum number of retries
    "retry_delay_seconds": 5                  # Delay between retries
}
```

## API Reference

The plugin provides a standardized API for interacting with CI/CD systems:

### Test Run Management

```python
# Create a test run
test_run = await ci_client.create_test_run({
    "name": "Distributed Test Run",
    "build_id": "build-123",
    "commit_sha": "abcdef123456",
    "branch": "main",
    "pr_number": "42",
    "system_info": {
        "python_version": "3.9.7",
        "platform": "Linux-5.15.0-x86_64",
        "hostname": "ci-worker-01"
    }
})

# Update a test run
success = await ci_client.update_test_run(
    test_run["id"],
    {
        "status": "completed",
        "summary": {
            "total_tasks": 10,
            "task_statuses": {"completed": 9, "failed": 1},
            "duration": 120.5
        },
        "end_time": datetime.now().isoformat()
    }
)
```

### Artifacts

```python
# Upload an artifact
success = await ci_client.upload_artifact(
    test_run["id"],
    "test_report.html",
    "Test Report"
)

# Download an artifact
success = await ci_client.download_artifact(
    test_run["id"],
    "Test Report",
    "downloaded_report.html"
)
```

### Pull Request Integration

```python
# Add a comment to a pull request
success = await ci_client.add_pr_comment(
    "42",
    "## Test Results\n\n**Status**: PASS\n\n**Duration**: 120.5 seconds\n\n**Total Tests**: 10\n\n**Failed Tests**: 1"
)
```

### Status Updates

```python
# Set a status on a commit
success = await ci_client.set_status(
    "abcdef123456",
    "success",
    "distributed-tests",
    "All tests passed",
    "https://ci.example.com/results/123"
)
```

### Test History and Performance Analysis

```python
# Get test run history
history = await ci_client.get_test_history(
    limit=10,
    branch="main"
)

# Record a performance metric
success = await ci_client.record_performance_metric(
    test_run["id"],
    task_id="task-123",
    metric_name="execution_time",
    metric_value=1.23,
    unit="seconds"
)

# Get performance metrics
metrics = await ci_client.get_performance_metrics(
    test_run_id=test_run["id"],
    metric_name="execution_time"
)

# Analyze performance trends
trends = await ci_client.analyze_performance_trends(
    metric_name="execution_time",
    grouping="branch",
    timeframe="1w"
)
```

## Environment Detection

The plugin automatically detects the CI environment from environment variables:

- **GitHub Actions**: Detects `GITHUB_ACTIONS=true` and extracts repository, build ID, commit SHA, branch, etc.
- **Jenkins**: Detects `JENKINS_URL` and extracts job name, build ID, build URL, branch, etc.
- **GitLab CI**: Detects `GITLAB_CI=true` and extracts project path, job ID, commit SHA, branch, etc.
- **Azure DevOps**: Detects `TF_BUILD=True` and extracts project, build ID, repository, commit SHA, branch, etc.
- **CircleCI**: Detects `CIRCLECI=true` and extracts project, build number, commit SHA, branch, etc.
- **Travis CI**: Detects `TRAVIS=true` and extracts repository, build ID, commit SHA, branch, etc.
- **Bitbucket Pipelines**: Detects `BITBUCKET_BUILD_NUMBER` and extracts repository, build ID, commit SHA, branch, etc.
- **TeamCity**: Detects `TEAMCITY_VERSION` and extracts project name, build ID, build number, etc.

## History Database

The plugin uses a SQLite database to store test history and performance metrics, enabling:

- Test run history tracking with details about each run
- Task execution tracking for individual tasks within runs
- Performance metric recording and trend analysis
- Automatic cleanup of old history data based on retention settings

## Standardized API Architecture

The plugin uses a standardized API architecture with the following components:

1. **Environment Detection**: Automatically detects the CI environment
2. **CI Client Factory**: Creates the appropriate client for the detected environment
3. **Standardized Client**: Wraps platform-specific clients with a unified interface
4. **Capability Detection**: Determines which features are supported by each CI system
5. **Retry Mechanism**: Implements retry logic with exponential backoff for API calls
6. **History Tracking**: Records test history in a local database for persistence
7. **Performance Analysis**: Analyzes performance metrics and trends over time

## Usage Examples

### Basic Integration

```python
from distributed_testing.plugin_architecture import PluginType
from distributed_testing.coordinator import DistributedTestingCoordinator

# Create coordinator with plugin support
coordinator = DistributedTestingCoordinator(
    db_path="benchmark_db.duckdb",
    enable_plugins=True
)

# Start coordinator
await coordinator.start()

# Get CI/CD integration plugin
ci_plugin = coordinator.plugin_manager.get_plugins_by_type(PluginType.INTEGRATION)["CICDIntegration-1.0.0"]

# Check CI status
ci_status = ci_plugin.get_ci_status()
print(f"CI System: {ci_status['ci_system']}")
print(f"Test Run: {ci_status['test_run_id']}")
print(f"Status: {ci_status['test_run_status']}")
```

### Advanced Performance Trend Analysis

```python
# Get performance trends for execution time
trends = await ci_plugin.analyze_performance_trends(
    metric_name="execution_time",
    grouping="branch",
    timeframe="1w",
    limit=10
)

# Print results
print(f"Performance Trends for {trends['metric_name']} ({trends['unit']})")
print(f"Overall average: {trends['overall']['avg']:.2f} {trends['unit']}")
print("By branch:")
for group in trends['groups']:
    print(f"  {group['name']}: {group['avg']:.2f} {trends['unit']} (min: {group['min']:.2f}, max: {group['max']:.2f})")
```

## Error Handling

The plugin implements comprehensive error handling with:

- Detailed error messages for all API calls
- Automatic retry with exponential backoff for transient errors
- Fallback mechanisms for unavailable features
- Detailed logging for debugging
- Graceful degradation for missing capabilities

## Extending Support for New CI Systems

To add support for a new CI system:

1. Create a new client implementation in `distributed_testing/ci/`
2. Implement the required methods (create_test_run, update_test_run, etc.)
3. Add detection logic to `_detect_ci_environment()`
4. Add token retrieval logic to `_get_ci_token()`
5. Add client creation logic to `_create_ci_client()`

## Conclusion

The CI/CD Integration Plugin provides a comprehensive solution for integrating the Distributed Testing Framework with various CI/CD systems. By abstracting away the differences between these systems and providing a standardized API, it enables seamless reporting, artifact management, and test result analysis across all platforms.