# CI/CD Client Modules

This directory contains client modules for integrating with CI/CD systems. These clients are used by the CI/CD Integration plugin to report test results, update build status, add PR comments, and manage artifacts.

## Standardized Integration

The CI/CD clients implement a standardized interface defined in `api_interface.py`, ensuring consistent behavior across different CI providers. Key features include:

- **Common Interface**: All providers implement the same interface for test reporting, artifact handling, and PR comments
- **Factory Pattern**: CI providers can be created dynamically via the `CIProviderFactory`
- **Standardized Artifact Handling**: All providers support a consistent approach for managing artifacts
- **Centralized Error Handling**: Common error handling for all providers

## Available Clients

### GitHub Client

**File**: `github_client.py`

Client for interacting with GitHub's API using the Checks API to report test results and status.

Key features:
- Creating and updating check runs
- Adding PR comments
- Support for partial artifact management

### GitLab Client

**File**: `gitlab_client.py`

Client for interacting with GitLab's API using commit status and merge request APIs.

Key features:
- Creating and updating commit status
- Adding merge request comments
- Integration with GitLab CI pipeline jobs

### Jenkins Client

**File**: `jenkins_client.py`

Client for interacting with Jenkins using build description and test reporting APIs.

Key features:
- Updating build descriptions with test results
- Associating test runs with Jenkins jobs
- Basic artifact management support

### Azure DevOps Client

**File**: `azure_client.py`

Client for interacting with Azure DevOps API for test run management and reporting.

Key features:
- Creating and updating test runs in Azure Test Plans
- Adding PR comments to Azure Repos
- Basic artifact management

## Usage

These clients are typically used by the CI/CD Integration plugin, which automatically selects and initializes the appropriate client based on the CI environment:

```python
from distributed_testing.ci import GitHubClient, GitLabClient, JenkinsClient, AzureClient

# Create and use GitHub client directly if needed
client = GitHubClient(
    token="github_token",
    repository="owner/repo"
)

# Create test run
test_run = await client.create_test_run({
    "name": "My Test Run",
    "commit_sha": "abc123"
})

# Update test run with results
await client.update_test_run(
    test_run["id"],
    {
        "status": "completed",
        "summary": {
            "total_tasks": 10,
            "task_statuses": {
                "completed": 8,
                "failed": 2
            },
            "duration": 120.5
        }
    }
)

# Add PR comment
await client.add_pr_comment(
    "42",
    "## Test Results\n\n- Total: 10\n- Passed: 8\n- Failed: 2"
)
```

## Artifact Handling

The CI/CD clients now support standardized artifact handling through the new `artifact_handler.py` module. This module provides a consistent way to handle artifacts across different CI providers and offers:

- **Centralized Storage**: Local storage for artifacts with comprehensive metadata tracking
- **Efficient Metadata**: File size, content hash, and other metadata automatically tracked
- **Provider-Independent**: Same API regardless of the underlying CI system
- **Batch Operations**: Upload multiple artifacts at once
- **Failure Handling**: Graceful fallbacks when CI provider uploads fail
- **Artifact URL Retrieval**: Universal mechanism to retrieve artifact URLs across all CI providers

### Artifact Upload and Management

Example usage for basic artifact management:

```python
from distributed_testing.ci.artifact_handler import get_artifact_handler
from distributed_testing.ci.register_providers import register_all_providers
from distributed_testing.ci.api_interface import CIProviderFactory

# Register all providers
register_all_providers()

# Create a provider
provider = await CIProviderFactory.create_provider("github", {
    "token": "github_token",
    "repository": "owner/repo"
})

# Get singleton artifact handler
artifact_handler = get_artifact_handler()

# Register provider with handler
artifact_handler.register_provider("github", provider)

# Upload artifact
success, metadata = await artifact_handler.upload_artifact(
    source_path="./test_results.json",
    artifact_name="test_results.json",
    artifact_type="report",
    test_run_id="test-123",
    provider_name="github"
)

# Get artifacts for a test run
artifacts = artifact_handler.get_artifacts_for_test_run("test-123")

# Get artifact by name
report = artifact_handler.get_artifact_by_name("test-123", "test_results.json")

# Purge artifacts
await artifact_handler.purge_artifacts_for_test_run("test-123")
```

### Artifact URL Retrieval

All CI providers now implement the `get_artifact_url` method that retrieves URLs for artifacts uploaded to the CI system. This method provides a standardized way to access artifacts across different CI platforms, even when the underlying storage mechanisms vary:

```python
from distributed_testing.ci.register_providers import register_all_providers
from distributed_testing.ci.api_interface import CIProviderFactory

# Register all providers
register_all_providers()

# Create a provider
provider = await CIProviderFactory.create_provider("github", {
    "token": "github_token",
    "repository": "owner/repo"
})

# Upload an artifact
result = await provider.upload_artifact(
    test_run_id="test-123",
    artifact_path="./test_results.json",
    artifact_name="test_results.json"
)

if result:
    # Retrieve the artifact URL
    url = await provider.get_artifact_url(
        test_run_id="test-123",
        artifact_name="test_results.json"
    )
    
    if url:
        print(f"Artifact URL: {url}")
        # URL can be used to download or access the artifact directly
```

#### Provider-Specific Implementations

Each CI provider implements `get_artifact_url` with provider-specific logic:

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

#### Implementation Features

The `get_artifact_url` implementations include:

- **URL Caching**: URLs are cached to minimize API calls
- **Error Handling**: Robust error handling with appropriate logging
- **Fallback Mechanisms**: Alternative URL resolution strategies when primary methods fail
- **Simulation Support**: Graceful handling of simulated test runs

#### Example Integration with Artifact Handler

```python
from distributed_testing.ci.artifact_handler import get_artifact_handler
from distributed_testing.ci.register_providers import register_all_providers
from distributed_testing.ci.api_interface import CIProviderFactory

# Register all providers
register_all_providers()

# Create provider
provider = await CIProviderFactory.create_provider("github", {
    "token": "github_token",
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

### Testing and Demos

To run tests and demos for the artifact handling system:

```bash
# Run all tests
python run_ci_provider_tests.py

# Run tests only
python run_ci_provider_tests.py --test-only

# Run demo with GitHub
python run_ci_provider_tests.py --demo-only --provider github --token YOUR_TOKEN --repository owner/repo

# Test artifact handling directly
python ci/test_artifact_handling.py

# Run artifact handling demo
python run_test_artifact_handling.py --provider github --token YOUR_TOKEN --repository owner/repo

# Test artifact URL retrieval specifically
python distributed_testing/test_artifact_url_retrieval.py
```

For more details on how these clients are used as part of the CI/CD Integration plugin, see [../integration/README.md](../integration/README.md).

## Hardware Monitoring CI Integration

The CI/CD clients are also used by the hardware monitoring system's CI integration, which provides:

- **GitHub Actions Workflows**: Automated test execution with GitHub Actions
- **Multi-Channel Notification System**: Notifications via Email, Slack, and GitHub
- **Status Badge Generation**: SVG badges showing current test status
- **Local CI Simulation**: Script for testing CI workflow locally

### Notification System

The hardware monitoring system includes a notification system that sends alerts when tests fail:

```python
from distributed_testing.ci_notification import send_notifications, load_config

# Load notification configuration
config = load_config("notification_config.json")

# Send notifications about test results
success = send_notifications({
    "test_status": "failure",
    "test_report": "./test_report.html",
    "channels": ["github", "slack", "email"]
}, config)
```

### Status Badge Generator

The status badge generator creates SVG badges showing the current test status:

```python
from distributed_testing.generate_status_badge import generate_badge_svg, get_test_status

# Get test status from database
status, passing_runs, total_runs = get_test_status("./test_metrics.duckdb")

# Generate badge
badge_svg = generate_badge_svg("tests", status)

# Write badge to file
with open("test_status.svg", "w") as f:
    f.write(badge_svg)
```

### CI Simulation

The hardware monitoring system includes a script for simulating the CI environment locally:

```bash
# Run basic CI tests
./run_hardware_monitoring_ci_tests.sh

# Run with badge generation and notifications
./run_hardware_monitoring_ci_tests.sh --mode full --generate-badge --send-notifications
```

For more details on the hardware monitoring CI integration, see:
- [../README_CI_INTEGRATION.md](../README_CI_INTEGRATION.md) - Quick guide to CI integration features
- [../CI_INTEGRATION_SUMMARY.md](../CI_INTEGRATION_SUMMARY.md) - Detailed implementation summary