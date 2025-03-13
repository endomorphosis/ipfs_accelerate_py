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

### Report Formats

- **JSON**: Machine-readable format for automated processing
- **Markdown**: Human-readable format for documentation
- **HTML**: Interactive format for web viewing

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