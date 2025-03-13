# CI/CD Client Modules

This directory contains client modules for integrating with CI/CD systems. These clients are used by the CI/CD Integration plugin to report test results, update build status, add PR comments, and manage artifacts.

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

For more details on how these clients are used as part of the CI/CD Integration plugin, see [../integration/README.md](../integration/README.md).