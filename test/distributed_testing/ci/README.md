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

Example usage:

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
```

For more details on how these clients are used as part of the CI/CD Integration plugin, see [../integration/README.md](../integration/README.md).