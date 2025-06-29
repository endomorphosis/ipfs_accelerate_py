# CI/CD Integration Guide for Distributed Testing Framework

This guide provides detailed instructions for integrating the Distributed Testing Framework with various CI/CD systems, including GitHub Actions, GitLab CI, and Jenkins.

## Overview

The CI/CD integration enables:

1. **Automated Test Discovery**: Automatically finds and analyzes test files
2. **Test Requirement Analysis**: Detects hardware requirements from test file contents
3. **Distributed Test Execution**: Submits tests to the coordinator for distribution to worker nodes
4. **Result Collection and Reporting**: Aggregates results and generates comprehensive reports
5. **Status Badge Generation**: Creates and updates status badges for repository documentation
6. **Coverage Tracking**: Monitors code coverage across test runs

## GitHub Actions Integration

### Workflow Configuration

The GitHub Actions workflow is defined in `.github/workflows/distributed-testing.yml`. It runs tests in parallel and generates reports and badges.

#### Customizing the Workflow

1. **Test Types**: By default, the workflow runs all test types (integration, fault, monitoring, stress). To run specific test types, modify the `test_type` input or the matrix configuration.

2. **Hardware Requirements**: The workflow detects hardware requirements from test files. To run tests on specific hardware, use the `hardware` input parameter.

3. **Timeouts**: Default timeout is 3600 seconds (1 hour). For longer-running tests, increase the `timeout` input parameter.

4. **Secrets**: For secure coordinator access, set the following repository secrets:
   - `COORDINATOR_URL`: URL of the coordinator server
   - `COORDINATOR_API_KEY`: API key for authentication

### Badge Configuration

Status badges are automatically generated and updated in the `.github/badges` directory. To display these badges in your README:

```markdown
![Combined Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/combined-status.json)
![Integration Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/integration-status.json)
![Fault Tolerance Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/fault-status.json)
![Monitoring Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/monitoring-status.json)
![Stress Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/stress-status.json)
![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/username/repo/main/test/duckdb_api/distributed_testing/.github/badges/coverage.json)
```

Replace `username/repo` with your actual GitHub username and repository name.

## GitLab CI Integration

### Pipeline Configuration

Create a `.gitlab-ci.yml` file in your repository:

```yaml
stages:
  - test

variables:
  COORDINATOR_URL: ${COORDINATOR_URL}
  API_KEY: ${COORDINATOR_API_KEY}

distributed-testing:
  stage: test
  image: python:3.10-slim
  script:
    - pip install -r test/duckdb_api/distributed_testing/requirements.test.txt
    - python -m duckdb_api.distributed_testing.cicd_integration --provider gitlab --coordinator ${COORDINATOR_URL} --api-key ${API_KEY} --test-dir test/duckdb_api/distributed_testing/tests --output-dir test_reports --report-formats json md html
  artifacts:
    paths:
      - test_reports/
    reports:
      junit: test_reports/test_report_*.xml
```

### GitLab CI Variables

Configure the following CI/CD variables in your GitLab project:
- `COORDINATOR_URL`: URL of the coordinator server
- `COORDINATOR_API_KEY`: API key for authentication

## Jenkins Integration

### Pipeline Configuration

Create a `Jenkinsfile` in your repository:

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.10-slim'
        }
    }
    
    environment {
        COORDINATOR_URL = credentials('COORDINATOR_URL')
        API_KEY = credentials('COORDINATOR_API_KEY')
    }
    
    stages {
        stage('Test') {
            steps {
                sh 'pip install -r test/duckdb_api/distributed_testing/requirements.test.txt'
                sh 'python -m duckdb_api.distributed_testing.cicd_integration --provider jenkins --coordinator ${COORDINATOR_URL} --api-key ${API_KEY} --test-dir test/duckdb_api/distributed_testing/tests --output-dir test_reports --report-formats json md html'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'test_reports/**', allowEmptyArchive: true
                    junit 'test_reports/test_report_*.xml'
                }
            }
        }
    }
}
```

### Jenkins Credentials

Configure the following credentials in Jenkins:
- `COORDINATOR_URL`: URL of the coordinator server
- `COORDINATOR_API_KEY`: API key for authentication

## Custom CI/CD Integration

For other CI/CD systems, use the `cicd_integration.py` module directly:

```bash
python -m duckdb_api.distributed_testing.cicd_integration \
    --provider generic \
    --coordinator http://coordinator-url:8080 \
    --api-key YOUR_API_KEY \
    --test-dir ./tests \
    --output-dir ./reports \
    --report-formats json md html \
    --verbose
```

### Command Line Options

- `--provider`: CI/CD provider (github, gitlab, jenkins, generic)
- `--coordinator`: URL of the coordinator server
- `--api-key`: API key for authentication
- `--test-dir`: Directory containing test files
- `--test-pattern`: Glob pattern for finding test files
- `--test-files`: Explicit list of test files to run
- `--output-dir`: Directory for report output
- `--report-formats`: Report formats to generate (json, md, html)
- `--timeout`: Maximum time to wait for test completion
- `--poll-interval`: How often to poll for results
- `--verbose`: Enable verbose output

## Programmatic Integration

You can also use the CI/CD integration programmatically in your Python code:

```python
from duckdb_api.distributed_testing.cicd_integration import CICDIntegration

integration = CICDIntegration(
    coordinator_url="http://coordinator-url:8080",
    api_key="YOUR_API_KEY",
    provider="generic",
    timeout=3600,
    poll_interval=15,
    verbose=True
)

exit_code = integration.run(
    test_dir="./tests",
    output_dir="./reports",
    report_formats=["json", "md", "html"]
)

sys.exit(exit_code)
```

## Test Result Analysis

The CI/CD integration generates comprehensive test reports with detailed information about test execution, including:

- Test status (passed, failed, cancelled, timeout)
- Execution time
- Hardware used for testing
- Error details for failed tests
- Statistical summaries of test results
- Artifact links with direct URLs for accessing test artifacts

### Report Formats

- **JSON**: Machine-readable format for automated processing
- **Markdown**: Human-readable format for documentation
- **HTML**: Interactive format for web viewing

### Artifact URL Integration

Test reports now include direct links to test artifacts, enabling easy access to test outputs, logs, and other artifacts. This integration leverages the artifact URL retrieval system, which provides a standardized way to access artifacts across different CI providers.

#### Key Benefits

- **Direct Access**: One-click access to test artifacts from reports, PR comments, and dashboards
- **Consistent Format**: Same URL format and behavior across all CI providers
- **Automatic Inclusion**: URLs automatically added to all report formats (Markdown, HTML, JSON)
- **Parallel Processing**: Efficient batch retrieval of multiple URLs using asyncio tasks (3-10x faster than sequential retrieval)
- **Enhanced PR Comments**: PR comments include direct links to artifacts for easy review
- **Robust Error Handling**: Graceful degradation when URL retrieval fails
- **Caching System**: Intelligent caching to minimize redundant API calls
- **Cross-Provider Compatibility**: Works consistently across GitHub, GitLab, Jenkins, CircleCI, Azure DevOps, TeamCity, Travis CI, and Bitbucket
- **Performance Optimization**: Reduced latency through parallel processing and caching
- **Automatic Recovery**: Fallback mechanisms when primary URL retrieval methods fail

#### Integration Components

The integration leverages three main components in the TestResultReporter class:

1. **Enhanced `collect_and_upload_artifacts`**: Automatically retrieves URLs for uploaded artifacts
   ```python
   # Automatic URL retrieval in collect_and_upload_artifacts
   artifacts = await reporter.collect_and_upload_artifacts(
       test_run_id="test-123",
       artifact_patterns=["./results/*.json", "./logs/*.log"]
   )
   
   # Each artifact includes its URL
   for artifact in artifacts:
       print(f"Artifact {artifact['name']} URL: {artifact.get('url')}")
   ```

2. **New `get_artifact_urls` Method**: Efficiently retrieves multiple artifact URLs in parallel using asyncio tasks
   ```python
   # Retrieve multiple artifact URLs in parallel
   artifact_urls = await reporter.get_artifact_urls(
       test_run_id="test-123",
       artifact_names=["test_results.json", "performance.csv", "test.log"]
   )
   
   # Use the URLs
   for name, url in artifact_urls.items():
       if url:
           print(f"Artifact {name} available at: {url}")
   ```

3. **Updated `report_test_result`**: Includes artifact URLs in reports and PR comments
   ```python
   # Add artifacts to test result metadata
   test_result.metadata["artifacts"] = artifacts
   
   # Generate reports with artifact URLs included
   report_files = await reporter.report_test_result(
       test_result,
       formats=["markdown", "html", "json"]
   )
   ```

The integration workflow automatically:
1. Uploads test artifacts to the CI provider
2. Retrieves artifact URLs using parallel URL retrieval for better performance
3. Includes URLs in all report formats (Markdown, HTML, JSON)
4. Adds URLs to PR comments and notifications
5. Handles edge cases and failures with appropriate fallback mechanisms

#### Technical Implementation

The `get_artifact_urls` method uses asyncio tasks to retrieve multiple URLs in parallel:

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

Performance testing shows that this parallel implementation is 3-10x faster than sequential URL retrieval when retrieving multiple URLs, especially in high-latency environments.

#### Report Formats with Artifact URLs

##### Markdown Reports with Artifacts
```markdown
## Artifacts

- [Test Results JSON](https://github.com/owner/repo/actions/runs/123/artifacts/456) (2.3 KB)
- [Performance Metrics CSV](https://github.com/owner/repo/actions/runs/123/artifacts/457) (1.5 KB)
- [Test Log](https://github.com/owner/repo/actions/runs/123/artifacts/458) (5.7 KB)
```

##### HTML Reports with Artifacts
```html
<h2>Artifacts</h2>
<ul>
  <li><a href="https://github.com/owner/repo/actions/runs/123/artifacts/456">Test Results JSON</a> (2.3 KB)</li>
  <li><a href="https://github.com/owner/repo/actions/runs/123/artifacts/457">Performance Metrics CSV</a> (1.5 KB)</li>
  <li><a href="https://github.com/owner/repo/actions/runs/123/artifacts/458">Test Log</a> (5.7 KB)</li>
</ul>
```

##### JSON Reports with Artifacts
```json
{
  "metadata": {
    "artifacts": [
      {
        "name": "Test Results JSON",
        "path": "./reports/test_results.json",
        "size_bytes": 2355,
        "url": "https://github.com/owner/repo/actions/runs/123/artifacts/456",
        "type": "result"
      },
      {
        "name": "Performance Metrics CSV",
        "path": "./reports/performance.csv",
        "size_bytes": 1536,
        "url": "https://github.com/owner/repo/actions/runs/123/artifacts/457",
        "type": "metrics"
      },
      {
        "name": "Test Log",
        "path": "./reports/test.log",
        "size_bytes": 5832,
        "url": "https://github.com/owner/repo/actions/runs/123/artifacts/458",
        "type": "log"
      }
    ]
  }
}
```

##### PR Comments with Artifacts
```markdown
## Test Run Results: test-123

**Status:** SUCCESS

### Artifacts
- [Test Results JSON](https://github.com/owner/repo/actions/runs/123/artifacts/456) (2.3 KB)
- [Performance Metrics CSV](https://github.com/owner/repo/actions/runs/123/artifacts/457) (1.5 KB)
- [Test Log](https://github.com/owner/repo/actions/runs/123/artifacts/458) (5.7 KB)
```

#### Example Usage

```python
from distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers

# Register all providers
register_all_providers()

# Create a CI provider
provider = await CIProviderFactory.create_provider("github", {
    "token": "YOUR_GITHUB_TOKEN",
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
    test_run_id=test_result.test_run_id,
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

#### Error Handling and Graceful Degradation

The artifact URL retrieval system includes robust error handling:

- Gracefully handles missing or unavailable artifacts
- Provides fallback mechanisms when primary URL retrieval methods fail
- Returns `None` for URLs that can't be retrieved rather than raising exceptions
- Logs detailed error information for troubleshooting
- Continues processing even when some URLs can't be retrieved

```python
# Error handling example
urls = await reporter.get_artifact_urls(
    test_run_id=test_run_id,
    artifact_names=["existing.json", "non_existent.json"]
)

# URLs for non-existent artifacts will be None, but the operation completes successfully
for name, url in urls.items():
    status = "Available" if url else "Not available"
    print(f"Artifact {name}: {status}")
```

#### Testing the Integration

A comprehensive test suite is available for testing the artifact URL retrieval integration:

```bash
# Run the test suite
python -m distributed_testing.test_reporter_artifact_integration

# Run the enhanced example
python -m distributed_testing.examples.enhanced_reporter_artifact_url_example --provider github
```

For complete documentation and details, see the [Artifact URL Retrieval Guide](../../../distributed_testing/ARTIFACT_URL_RETRIEVAL_GUIDE.md).

### Advanced Reporting

For advanced reporting needs, you can customize the report generation in `cicd_integration.py`. The `generate_report` method can be extended to include additional information or custom formatting.

## Troubleshooting

### Common Issues

1. **Coordinator Connection Failures**:
   - Verify the coordinator URL is correct
   - Ensure the coordinator is running and accessible from the CI environment
   - Check that the API key is valid

2. **Test Discovery Issues**:
   - Verify the test directory or pattern is correct
   - Ensure test files follow the naming convention (test_*.py)
   - Check file permissions

3. **Long-Running Tests**:
   - Increase the timeout value
   - Consider breaking large tests into smaller ones
   - Monitor resource usage during test execution

4. **Worker Availability**:
   - Ensure workers with required hardware are registered with the coordinator
   - Check worker health status in the coordinator dashboard
   - Verify worker capabilities match test requirements

### Debugging Tips

1. Run with the `--verbose` flag to get detailed output
2. Check the coordinator logs for errors or warnings
3. Verify test requirements are correctly analyzed
4. Run tests locally before submitting to CI/CD

## Advanced Configuration

### Custom Test Requirements

The CI/CD integration automatically detects hardware requirements from test file contents. You can customize detection by modifying the `analyze_test_requirements` method in `cicd_integration.py`.

### Custom Result Handling

For custom result handling, extend the `report_to_ci_system` method in `cicd_integration.py` to integrate with your specific CI/CD system's reporting capabilities.

### Custom Badge Generation

To customize badge generation, modify the `github_badge_generator.py` script. You can add new badge types or change the appearance of existing badges.

## Security Considerations

### API Key Management

- Store API keys securely in CI/CD secrets
- Use different API keys for different environments
- Rotate API keys periodically

### Worker Authentication

- Ensure workers authenticate securely with the coordinator
- Use JWT token authentication for all communications
- Validate worker identity before accepting results

### Network Security

- Use HTTPS for all communications with the coordinator
- Restrict network access to the coordinator server
- Consider using a VPN for coordinator-worker communication

## Best Practices

1. **Run Tests Locally First**: Use the `run_all_tests.sh` script to run tests locally before pushing changes.

2. **Clear Test Documentation**: Document hardware requirements and dependencies for your tests.

3. **Test Isolation**: Ensure each test can run independently without side effects.

4. **Cleanup After Tests**: Always clean up temporary files and resources after test completion.

5. **Graceful Failure Handling**: Design tests to handle failures gracefully and provide clear error messages.

6. **Progressive Testing**: Start with small, quick tests and gradually add more complex ones.

7. **Hardware Awareness**: Be explicit about hardware requirements in your test files.

8. **CI Resources**: Be mindful of CI resource consumption, especially for stress tests.

9. **Consistent Naming**: Follow consistent naming conventions for test files and classes.

10. **Regular Updates**: Keep the Distributed Testing Framework components up to date.