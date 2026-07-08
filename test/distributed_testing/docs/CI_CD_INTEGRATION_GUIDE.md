# CI/CD Integration Guide for Distributed Testing Framework

This comprehensive guide explains how to integrate the Distributed Testing Framework with various CI/CD systems, including GitHub Actions, GitLab CI, Jenkins, Azure DevOps, CircleCI, Travis CI, Bitbucket Pipelines, and TeamCity.

## Table of Contents

1. [Overview](#overview)
2. [Standardized API Architecture](#standardized-api-architecture)
3. [Supported CI/CD Systems](#supported-cicd-systems)
4. [Enhanced Artifact Handling](#enhanced-artifact-handling)
5. [Integration Options](#integration-options)
6. [Plugin-Based Integration](#plugin-based-integration)
7. [Command-Line Integration](#command-line-integration)
8. [API-Based Integration](#api-based-integration)
9. [Configuration Options](#configuration-options)
10. [Performance History Tracking](#performance-history-tracking)
11. [Advanced Usage](#advanced-usage)
12. [Troubleshooting](#troubleshooting)
13. [API Reference](#api-reference)

## Overview

The Distributed Testing Framework provides seamless integration with popular CI/CD systems, enabling:

- **Test Execution**: Run distributed tests as part of CI/CD pipelines
- **Status Reporting**: Report test results back to CI/CD systems
- **Artifact Management**: Upload and organize test artifacts with categorization
- **PR Feedback**: Add comments to pull requests with test results
- **Build Status Updates**: Update build status based on test results
- **Environment Detection**: Automatically detect the CI/CD environment
- **Test History Tracking**: Record and analyze test run history over time
- **Performance Analysis**: Track and analyze performance metrics for optimization
- **Failure Analysis**: Detailed analysis of test failures with classification
- **Dashboard Integration**: Visualize test results in a comprehensive dashboard

The integration is designed to be flexible, allowing you to choose the approach that best fits your workflow and providing enhanced reliability with comprehensive retry mechanisms and fault tolerance features.

## Standardized API Architecture

The framework provides a comprehensive standardized API architecture for CI/CD integrations, ensuring consistent behavior across different providers while maintaining platform-specific optimizations. This makes it easy to switch between CI/CD systems or add support for new ones.

The architecture consists of several key components:

1. **CIProviderInterface**: Abstract base class that defines the interface all CI providers must implement
2. **CI Provider Factory**: Factory pattern that creates and manages CI provider implementations
3. **TestRunResult**: Standardized representation of test results across different CI systems
4. **Environment Detection System**: Automatically detects and configures for the CI/CD environment
5. **Capability Management**: Detects and tracks which features are supported by each CI system
6. **Retry Mechanism**: Implements exponential backoff and retry logic for API calls
7. **History Tracking System**: Records test history in a local database for persistence
8. **Performance Analysis Engine**: Analyzes metrics and trends across test runs

This architecture is defined in the `distributed_testing/ci` package and includes:

- `CIProviderInterface`: Abstract base class defining the contract for all CI providers
- `CIProviderFactory`: Factory class for creating provider instances by platform
- `TestRunResult`: Standardized representation of test results across platforms
- `PerformanceMetric`: Representation of performance data for analysis

All CI providers now fully implement the standardized `CIProviderInterface`, ensuring consistent behavior across different CI/CD systems.

### Interface Design Principles

The standardized API was designed with these key principles in mind:

1. **Consistency**: Provide a consistent experience regardless of the underlying CI/CD system
2. **Extensibility**: Make it easy to add support for new CI/CD systems
3. **Separation of Concerns**: Decouple the framework from specific CI/CD implementations
4. **Resilience**: Provide graceful fallbacks when CI/CD system features are unavailable
5. **Authentication Flexibility**: Support various authentication mechanisms based on provider needs

### Key Components in Detail

#### CIProviderInterface

The `CIProviderInterface` defines the contract that all CI provider implementations must follow:

```python
class CIProviderInterface(abc.ABC):
    @abc.abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with provider-specific configuration."""
        pass
        
    @abc.abstractmethod
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new test run in the CI system."""
        pass
        
    # ... other required methods
```

The following methods are required for all implementations:

| Method | Purpose | Common Parameters | Return Value |
|--------|---------|-------------------|--------------|
| `initialize` | Set up the provider with config | `token`, `repository`, `api_url` | Boolean success |
| `create_test_run` | Create new test run | `name`, `build_id`, `commit_sha` | Test run dict |
| `update_test_run` | Update existing test run | `test_run_id`, status data | Boolean success |
| `add_pr_comment` | Add comment to PR | `pr_number`, `comment` text | Boolean success |
| `upload_artifact` | Upload test artifact | `test_run_id`, file path and name | Boolean success |
| `get_test_run_status` | Get current status | `test_run_id` | Status dict |
| `set_build_status` | Set build status | `status`, `description` | Boolean success |
| `close` | Clean up resources | None | None |

#### TestRunResult

The `TestRunResult` class provides a standardized way to represent test results across different CI systems:

```python
result = TestRunResult(
    test_run_id="1234",
    status="completed",
    total_tests=10,
    passed_tests=8,
    failed_tests=2,
    skipped_tests=0,
    duration_seconds=25.5,
    metadata={"commit": "abc123", "branch": "main"}
)

# Convert to dict for serialization
result_dict = result.to_dict()

# Create from dict (e.g., when deserializing)
restored_result = TestRunResult.from_dict(result_dict)
```

#### CIProviderFactory

The `CIProviderFactory` simplifies provider creation and management:

```python
# Register a provider
CIProviderFactory.register_provider("circleci", CircleCIProvider)

# Create provider instance
provider = await CIProviderFactory.create_provider(
    "github", 
    {"token": "token123", "repository": "owner/repo"}
)

# Get list of available providers
providers = CIProviderFactory.get_available_providers()
```

### CI Client Modules and Authentication

The framework provides specialized client modules for each supported CI/CD system in the `distributed_testing/ci` directory. All of these clients implement the `CIProviderInterface` abstract base class:

- **github_client.py**: `GitHubClient` implements integration with GitHub API using Checks API and Issues API
- **gitlab_client.py**: `GitLabClient` implements integration with GitLab API using Commit Status and Merge Requests API
- **jenkins_client.py**: `JenkinsClient` implements integration with Jenkins using Build Description and Test Results API
- **azure_client.py**: `AzureDevOpsClient` implements integration with Azure DevOps using Test Plans API and Pull Requests API
- Additional providers for CircleCI, Bitbucket, TeamCity and Travis CI

All clients implement the same standardized interface for consistent usage across different CI/CD systems, making it easy to switch between CI/CD systems or add support for new ones.

The API supports different authentication mechanisms based on provider requirements:

- **Token-based Authentication**: Used by GitHub, GitLab (via `token` config parameter)
- **Username/Token Authentication**: Used by Jenkins (via `user` and `token` parameters)
- **PAT Authentication**: Used by Azure DevOps (via `token` parameter)
- **API Key Authentication**: Used by other systems with similar authentication requirements

For security, tokens should be provided via environment variables or secure storage:

```python
# Recommended approach
import os
token = os.environ.get("GITHUB_TOKEN")
provider = await CIProviderFactory.create_provider(
    "github", 
    {"token": token, "repository": "owner/repo"}
)
```

### Error Handling

The API provides consistent error handling across providers:

- All methods return `boolean` success indicators or rich result objects
- Exceptions include provider-specific details but follow consistent patterns
- Providers implement graceful degradation for unavailable features
- Detailed error information is logged for troubleshooting

Error types you may encounter:

- `AuthenticationError`: Issues with credentials
- `ResourceNotFoundError`: Referenced resource doesn't exist
- `PermissionError`: Insufficient permissions
- `RateLimitError`: API rate limits exceeded
- `ConnectionError`: Network-related issues
- `ConfigurationError`: Missing or invalid configuration

Example error handling:

```python
try:
    success = await provider.add_pr_comment("123", "Test results")
    if not success:
        logger.warning("Failed to add PR comment")
except Exception as e:
    logger.error(f"Error during PR comment: {e}")
    # Implement fallback if critical
```

### Integration with Plugin Architecture

The CI/CD API interface integrates seamlessly with the framework's plugin architecture:

```python
from distributed_testing.plugin_architecture import Plugin, PluginType, HookType
from distributed_testing.ci import CIProviderFactory

class MyCIIntegrationPlugin(Plugin):
    async def initialize(self, coordinator):
        # Create CI provider using factory
        self.provider = await CIProviderFactory.create_provider(
            "github", self.config
        )
        
        # Register for test completion hook
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        
        return True
        
    async def on_task_completed(self, task_id, result):
        # Report results to CI system
        await self.provider.update_test_run(self.test_run_id, {
            "status": "completed",
            "summary": self._convert_result_to_summary(result)
        })
```

## Supported CI/CD Systems

The framework currently supports the following CI/CD systems:

| CI/CD System | Provider ID | Features | Capabilities | Authentication Method |
|--------------|-------------|----------|--------------|----------------------|
| GitHub Actions | `github` | Check runs, PR comments, status updates, artifacts | Full API support, PR integration, Check runs API | Token-based |
| GitLab CI | `gitlab` | Pipelines, merge request comments, job artifacts | MR comments, CI pipeline integration, job artifacts | Token-based |
| Jenkins | `jenkins` | Build status, test reports, artifacts | Build integration, test reporting, artifact archiving | Username/Token |
| Azure DevOps | `azure` | Test runs, PR comments, artifacts | Test run integration, PR integration, artifact upload | PAT-based |
| CircleCI | `circleci` | Workflows, artifacts, status updates | Workflow integration, artifact upload, status checks | Token-based |
| Travis CI | `travis` | Build status, PR comments, artifacts | Build integration, PR comments, artifact management | Token-based |
| Bitbucket Pipelines | `bitbucket` | Pipelines, PR comments, artifacts | Pipeline integration, PR comments, artifact upload | Username/App Password |
| TeamCity | `teamcity` | Build status, test reports, artifacts | Build integration, test reporting, artifact archiving | Username/Password |
| Local Mode | `local` | File-based storage, history tracking | Local storage, history tracking, trend analysis | None |

## Enhanced Artifact Handling

The framework now includes an Enhanced Artifact Handling System for comprehensive management, discovery, and analysis of artifacts across different CI/CD providers.

This system provides:

- **Standardized Metadata**: Uniform metadata extraction across all CI providers
- **Content Classification**: Automatic detection of artifact types and content
- **Efficient Retrieval**: Smart caching system for fast artifact access
- **Trend Analysis**: Tools for analyzing metrics trends across test runs
- **Discovery**: Powerful search capabilities for finding artifacts by criteria
- **Comparison**: Tools for comparing artifacts between versions

### Key Components

The Enhanced Artifact Handling System consists of three main components:

1. **Artifact Metadata** (`ArtifactMetadata`): Extracts and manages metadata about artifacts
2. **Artifact Discovery** (`ArtifactDiscovery`): Provides search and analysis capabilities 
3. **Artifact Retriever** (`ArtifactRetriever`): Retrieves and caches artifacts efficiently

### Artifact Metadata

The `ArtifactMetadata` class provides comprehensive metadata extraction:

```python
# Create artifact metadata with automatic type detection
metadata = ArtifactMetadata(
    artifact_name="test_report.json",
    artifact_path="/path/to/report.json",
    test_run_id="test-123",
    provider_name="github"
)

# Add custom labels for categorization
metadata.add_label("performance")
metadata.add_label("regression-test")

# Add custom metadata
metadata.add_metadata("version", "1.0")
metadata.add_metadata("platform", "linux")

# Validate artifact (checks file exists and hash matches)
if metadata.validate():
    print("Artifact valid!")
```

### Artifact Discovery

The `ArtifactDiscovery` class provides powerful search and analysis tools:

```python
# Discover artifacts matching criteria
matching_artifacts = ArtifactDiscovery.discover_artifacts(
    artifacts=all_artifacts,
    artifact_type="performance_report",
    labels=["regression-test"],
    metadata_query={"platform": "linux"},
    content_query={"metrics.throughput": 1250.5}
)

# Group artifacts by type
grouped_artifacts = ArtifactDiscovery.group_artifacts_by_type(all_artifacts)

# Find latest artifact of a specific type
latest_perf_report = ArtifactDiscovery.find_latest_artifact(
    artifacts=all_artifacts,
    artifact_type="performance_report"
)

# Extract metrics from multiple artifacts for analysis
metrics = ArtifactDiscovery.extract_metrics_from_artifacts(
    artifacts=perf_artifacts,
    metric_names=["throughput", "latency", "memory_usage"]
)
```

### Artifact Retriever

The `ArtifactRetriever` efficiently retrieves and caches artifacts from CI providers:

```python
# Create retriever with custom settings
retriever = ArtifactRetriever(
    cache_dir="./artifact_cache",
    max_cache_size_mb=1024,
    max_cache_age_days=7
)

# Register CI providers
retriever.register_provider("github", github_client)
retriever.register_provider("gitlab", gitlab_client)

# Retrieve artifact with caching
artifact_path, metadata = await retriever.retrieve_artifact(
    test_run_id="test-123",
    artifact_name="performance_report.json",
    provider_name="github",
    use_cache=True
)

# Batch retrieve multiple artifacts in parallel
artifacts_to_retrieve = [
    {"test_run_id": "test-123", "artifact_name": "logs.txt", "provider_name": "github"},
    {"test_run_id": "test-123", "artifact_name": "metrics.json", "provider_name": "github"},
    {"test_run_id": "test-456", "artifact_name": "report.json", "provider_name": "gitlab"}
]

results = await retriever.retrieve_artifacts_batch(artifacts_to_retrieve)

# Compare artifacts between versions
comparison = await retriever.compare_artifacts(
    artifact1={"test_run_id": "test-123", "artifact_name": "report.json", "provider_name": "github"},
    artifact2={"test_run_id": "test-456", "artifact_name": "report.json", "provider_name": "github"}
)
```

### CI Provider-Specific Implementations

The system provides specialized artifact handling for each CI provider:

- **GitHub**: Uses GitHub Gists API for text artifacts and Releases API for binary artifacts
- **GitLab**: Uses Repository Files API with dedicated branches for storing artifacts
- **Jenkins**: Uses Jenkins Artifact Storage API with direct artifact uploading
- **Azure DevOps**: Uses Test Attachment API for test-associated artifacts

### Initialization with Providers

```python
from distributed_testing.ci.register_providers import initialize_artifact_systems

# Initialize with providers
providers = await initialize_artifact_systems(
    provider_configs={
        "github": {"token": "github_token", "repository": "owner/repo"},
        "gitlab": {"token": "gitlab_token", "project": "group/project"}
    },
    artifact_handler_storage_dir="./artifacts",
    artifact_retriever_cache_dir="./artifact_cache"
)

# Access registered providers
github_provider = providers["github"]
gitlab_provider = providers["gitlab"]

# Systems are pre-configured with providers
artifact_handler = get_artifact_handler()
artifact_retriever = get_artifact_retriever()
```

For more details, see the [Enhanced Artifact Handling System documentation](ENHANCED_ARTIFACT_HANDLING.md).

## Integration Options

There are three ways to integrate the framework with CI/CD systems:

1. **Plugin-Based Integration**: Use the `CICDIntegrationPlugin` for seamless integration within the framework
2. **Command-Line Integration**: Use the `cicd_integration.py` script for command-line integration
3. **API-Based Integration**: Use the CI provider classes directly for custom integration

## Plugin-Based Integration

The easiest way to integrate is using the `CICDIntegrationPlugin` that comes with the framework. The latest implementation now supports all major CI/CD systems with a unified interface and specialized client modules:

```python
from distributed_testing.plugin_architecture import PluginType
from distributed_testing.integration.ci_cd_integration_plugin import CICDIntegrationPlugin

# Create and configure the CI/CD Integration plugin
ci_plugin = CICDIntegrationPlugin()

# Configure plugin (or let it auto-detect the CI environment)
ci_plugin.configure({
    "ci_system": "github",  # Options: github, gitlab, jenkins, azure, circleci, travis, bitbucket, teamcity, auto
    "api_token": "YOUR_TOKEN",  # Or username/password for systems that require it
    "repository": "owner/repo",
    "update_interval": 30,
    "enable_pr_comments": True,
    "enable_artifacts": True,
    "artifact_dir": "test_artifacts",
    "result_format": "all",  # Options: junit, json, html, all
    "enable_history_tracking": True,  # Track test performance over time
    "track_performance_trends": True,  # Analyze performance trends
    "retry_attempts": 3  # Retry API calls on failure
})

# Initialize coordinator with plugin support
coordinator = DistributedTestingCoordinator(
    db_path="benchmark_db.duckdb",
    enable_plugins=True,
    plugin_dirs=["distributed_testing/integration"]
)

# Start coordinator
await coordinator.start()

# Load plugin
await coordinator.plugin_manager.load_plugin(ci_plugin)

# Get CI status
ci_status = ci_plugin.get_ci_status()
print(f"CI System: {ci_status['ci_system']}")
print(f"Test Run: {ci_status['test_run_id']}")
print(f"Status: {ci_status['test_run_status']}")

# Run tests - plugin will automatically report results
await coordinator.submit_task({"type": "test", "config": {"test_file": "test_example.py"}})
```

The plugin uses specialized client implementations for each CI/CD system located in the `distributed_testing/ci` directory. These clients handle the specifics of interacting with each system while providing a consistent interface to the plugin. All client implementations follow the standardized `CIProviderInterface` and are managed through the `CIProviderFactory`, making it easy to switch between different CI/CD systems or add support for new ones.

## Command-Line Integration

For command-line integration, use the `ci_integration_runner.py` script in the `distributed_testing/integration` package:

```bash
# Run tests from GitHub Actions
python -m distributed_testing.integration.ci_integration_runner \
    --provider github \
    --coordinator http://coordinator-url:8080 \
    --api-key YOUR_API_KEY \
    --test-dir ./tests \
    --report-formats json,xml,html

# Run tests from GitLab CI
python -m distributed_testing.integration.ci_integration_runner \
    --provider gitlab \
    --coordinator http://coordinator-url:8080 \
    --api-key YOUR_API_KEY \
    --test-pattern "test_*.py" \
    --enable-pr-comments

# Run tests from Jenkins
python -m distributed_testing.integration.ci_integration_runner \
    --provider jenkins \
    --coordinator http://coordinator-url:8080 \
    --api-key YOUR_API_KEY \
    --test-files test_file1.py test_file2.py \
    --user jenkins_user \
    --token jenkins_token
    
# Run tests from CircleCI
python -m distributed_testing.integration.ci_integration_runner \
    --provider circleci \
    --coordinator http://coordinator-url:8080 \
    --api-key YOUR_API_KEY \
    --test-dir ./tests \
    --token $CIRCLE_TOKEN \
    --project-slug ${CIRCLE_PROJECT_USERNAME}/${CIRCLE_PROJECT_REPONAME}

# Run tests from Bitbucket Pipelines
python -m distributed_testing.integration.ci_integration_runner \
    --provider bitbucket \
    --coordinator http://coordinator-url:8080 \
    --api-key YOUR_API_KEY \
    --test-dir ./tests \
    --username $BITBUCKET_USER \
    --app-password $BITBUCKET_APP_PASSWORD \
    --workspace $BITBUCKET_WORKSPACE \
    --repository $BITBUCKET_REPO_SLUG
    
# Run tests from TeamCity
python -m distributed_testing.integration.ci_integration_runner \
    --provider teamcity \
    --coordinator http://coordinator-url:8080 \
    --api-key YOUR_API_KEY \
    --test-dir ./tests \
    --url $TEAMCITY_URL \
    --username $TEAMCITY_USER \
    --password $TEAMCITY_PASSWORD \
    --build-id $TEAMCITY_BUILD_ID

# Run tests from Travis CI
python -m distributed_testing.integration.ci_integration_runner \
    --provider travis \
    --coordinator http://coordinator-url:8080 \
    --api-key YOUR_API_KEY \
    --test-dir ./tests \
    --token $TRAVIS_TOKEN \
    --repository $TRAVIS_REPO_SLUG

# Run the example CI integration
python -m distributed_testing.run_test_ci_integration \
    --ci-system github \
    --repository user/repo \
    --api-token $GITHUB_TOKEN \
    --update-interval 5
```

The command-line interface supports the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | CI provider (github, gitlab, jenkins, azure, auto) | auto |
| `--coordinator` | Coordinator URL | http://localhost:8080 |
| `--api-key` | API key for coordinator | None |
| `--test-dir` | Directory containing test files | ./tests |
| `--test-pattern` | Glob pattern for test files | test_*.py |
| `--test-files` | Specific test files to run | None |
| `--report-formats` | Report formats (json, xml, html, all) | xml |
| `--output-dir` | Directory for output files | ./test_results |
| `--timeout` | Timeout in seconds | 3600 |
| `--enable-pr-comments` | Enable PR comments | False |
| `--detailed-logging` | Enable detailed logging | False |

### CI/CD Configuration Examples

#### GitHub Actions

```yaml
name: Distributed Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  distributed-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run distributed tests with new CI/CD integration
      run: |
        python -m distributed_testing.integration.ci_integration_runner \
          --provider github \
          --coordinator ${{ secrets.COORDINATOR_URL }} \
          --api-key ${{ secrets.API_KEY }} \
          --test-dir ./tests \
          --report-formats json,xml,html \
          --enable-pr-comments
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    - name: Upload test reports
      uses: actions/upload-artifact@v3
      with:
        name: test-reports
        path: test_results/
```

#### GitLab CI

```yaml
distributed-tests:
  stage: test
  image: python:3.10
  before_script:
    - pip install -r requirements.txt
  script:
    - python -m distributed_testing.integration.ci_integration_runner \
        --provider gitlab \
        --coordinator http://coordinator-url:8080 \
        --api-key $COORDINATOR_API_KEY \
        --test-dir ./tests \
        --report-formats json,xml,html \
        --enable-pr-comments
  variables:
    GITLAB_TOKEN: $CI_JOB_TOKEN
  artifacts:
    paths:
      - test_results/
```

#### Jenkins Pipeline

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.10'
        }
    }
    stages {
        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh '''
                python -m distributed_testing.integration.ci_integration_runner \
                    --provider jenkins \
                    --coordinator ${COORDINATOR_URL} \
                    --api-key ${API_KEY} \
                    --test-dir ./tests \
                    --report-formats json,xml,html \
                    --user ${JENKINS_USER} \
                    --token ${JENKINS_TOKEN}
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'test_results/*', fingerprint: true
                    junit 'test_results/*.xml'
                }
            }
        }
    }
    
    environment {
        JENKINS_USER = credentials('jenkins-user')
        JENKINS_TOKEN = credentials('jenkins-token')
        COORDINATOR_URL = 'http://coordinator-url:8080'
        API_KEY = credentials('coordinator-api-key')
    }
}
```

#### Advanced Jenkins Pipeline with Matrix Testing

```groovy
pipeline {
    agent none
    
    parameters {
        booleanParam(name: 'RUN_UNIT_TESTS', defaultValue: true, description: 'Run unit tests')
        booleanParam(name: 'RUN_INTEGRATION_TESTS', defaultValue: true, description: 'Run integration tests')
        choice(name: 'TEST_ENVIRONMENT', choices: ['dev', 'staging', 'prod'], description: 'Test environment')
    }
    
    stages {
        stage('Matrix Tests') {
            matrix {
                agent {
                    docker {
                        image 'python:${PYTHON_VERSION}'
                    }
                }
                axes {
                    axis {
                        name 'PYTHON_VERSION'
                        values '3.8', '3.9', '3.10'
                    }
                    axis {
                        name 'TEST_TYPE'
                        values 'unit', 'integration'
                    }
                }
                
                when {
                    anyOf {
                        expression { params.RUN_UNIT_TESTS && TEST_TYPE == 'unit' }
                        expression { params.RUN_INTEGRATION_TESTS && TEST_TYPE == 'integration' }
                    }
                }
                
                stages {
                    stage('Setup') {
                        steps {
                            sh 'pip install -r requirements.txt'
                        }
                    }
                    stage('Test') {
                        steps {
                            sh """
                            python -m distributed_testing.integration.ci_integration_runner \\
                                --provider jenkins \\
                                --coordinator ${COORDINATOR_URL} \\
                                --api-key ${API_KEY} \\
                                --test-dir ./tests/${TEST_TYPE} \\
                                --output-dir test_results/${TEST_TYPE}-python${PYTHON_VERSION} \\
                                --report-formats junit,json,html \\
                                --test-environment ${params.TEST_ENVIRONMENT} \\
                                --user ${JENKINS_USER} \\
                                --token ${JENKINS_TOKEN}
                            """
                        }
                        post {
                            always {
                                archiveArtifacts artifacts: "test_results/${TEST_TYPE}-python${PYTHON_VERSION}/*", fingerprint: true
                                junit "test_results/${TEST_TYPE}-python${PYTHON_VERSION}/*.xml"
                            }
                        }
                    }
                }
            }
        }
        
        stage('Report Aggregation') {
            agent {
                docker {
                    image 'python:3.10'
                }
            }
            steps {
                sh """
                python -m distributed_testing.integration.ci_integration_runner \\
                    --provider jenkins \\
                    --coordinator ${COORDINATOR_URL} \\
                    --api-key ${API_KEY} \\
                    --aggregate-reports \\
                    --report-dir test_results \\
                    --output-dir test_reports \\
                    --user ${JENKINS_USER} \\
                    --token ${JENKINS_TOKEN}
                """
            }
            post {
                always {
                    archiveArtifacts artifacts: "test_reports/*", fingerprint: true
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'test_reports',
                        reportFiles: 'aggregate_report.html',
                        reportName: 'Aggregated Test Report'
                    ])
                }
            }
        }
    }
    
    environment {
        JENKINS_USER = credentials('jenkins-user')
        JENKINS_TOKEN = credentials('jenkins-token')
        COORDINATOR_URL = 'http://coordinator-url:8080'
        API_KEY = credentials('coordinator-api-key')
    }
    
    post {
        success {
            echo 'All tests passed!'
        }
        failure {
            echo 'Tests failed!'
        }
    }
}
```

#### Azure DevOps Pipeline

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
      - main
      - feature/*

pool:
  vmImage: 'ubuntu-latest'

variables:
  PYTHON_VERSION: '3.10'
  COORDINATOR_URL: 'http://coordinator-url:8080'

stages:
  - stage: Test
    jobs:
      - job: RunDistributedTests
        timeoutInMinutes: 60
        variables:
          - group: test-credentials
          
        steps:
          - task: UsePythonVersion@0
            inputs:
              versionSpec: '$(PYTHON_VERSION)'
              addToPath: true
            displayName: 'Set up Python $(PYTHON_VERSION)'
            
          - script: |
              python -m pip install --upgrade pip
              pip install -r requirements.txt
            displayName: 'Install dependencies'
            
          - script: |
              python -m distributed_testing.integration.ci_integration_runner \
                --provider azure \
                --coordinator $(COORDINATOR_URL) \
                --api-key $(COORDINATOR_API_KEY) \
                --test-dir ./tests \
                --output-dir $(Build.ArtifactStagingDirectory)/test_results \
                --report-formats junit,xml,html \
                --enable-pr-comments
            displayName: 'Run distributed tests'
            env:
              AZURE_DEVOPS_TOKEN: $(System.AccessToken)
            
          - task: PublishTestResults@2
            inputs:
              testResultsFormat: 'JUnit'
              testResultsFiles: '$(Build.ArtifactStagingDirectory)/test_results/*.xml'
              mergeTestResults: true
              testRunTitle: 'Distributed Tests'
            displayName: 'Publish test results'
            
          - task: PublishBuildArtifacts@1
            inputs:
              PathtoPublish: '$(Build.ArtifactStagingDirectory)/test_results'
              ArtifactName: 'test-results'
              publishLocation: 'Container'
            displayName: 'Publish test artifacts'
```

#### Advanced Azure DevOps Pipeline with Matrix Testing and Pull Request Integration

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
      - main
      - feature/*

pr:
  branches:
    include:
      - main

pool:
  vmImage: 'ubuntu-latest'

variables:
  - name: COORDINATOR_URL
    value: 'http://coordinator-url:8080'
  - name: ENABLE_PR_COMMENTS
    value: ${{ eq(variables['Build.Reason'], 'PullRequest') }}

stages:
  - stage: Test
    jobs:
      - job: MatrixTesting
        strategy:
          matrix:
            Python38_Unit:
              PYTHON_VERSION: '3.8'
              TEST_TYPE: 'unit'
            Python38_Integration:
              PYTHON_VERSION: '3.8'
              TEST_TYPE: 'integration'
            Python39_Unit:
              PYTHON_VERSION: '3.9'
              TEST_TYPE: 'unit'
            Python39_Integration:
              PYTHON_VERSION: '3.9'
              TEST_TYPE: 'integration'
            Python310_Unit:
              PYTHON_VERSION: '3.10'
              TEST_TYPE: 'unit'
            Python310_Integration:
              PYTHON_VERSION: '3.10'
              TEST_TYPE: 'integration'
              
        timeoutInMinutes: 60
        variables:
          - group: test-credentials
          
        steps:
          - task: UsePythonVersion@0
            inputs:
              versionSpec: '$(PYTHON_VERSION)'
              addToPath: true
            displayName: 'Set up Python $(PYTHON_VERSION)'
            
          - script: |
              python -m pip install --upgrade pip
              pip install -r requirements.txt
            displayName: 'Install dependencies'
            
          - script: |
              python -m distributed_testing.integration.ci_integration_runner \
                --provider azure \
                --coordinator $(COORDINATOR_URL) \
                --api-key $(COORDINATOR_API_KEY) \
                --test-dir ./tests/$(TEST_TYPE) \
                --output-dir $(Build.ArtifactStagingDirectory)/test_results/$(TEST_TYPE)-python$(PYTHON_VERSION) \
                --report-formats junit,xml,html \
                --enable-pr-comments $(ENABLE_PR_COMMENTS) \
                --pr-number $(System.PullRequest.PullRequestNumber)
            displayName: 'Run $(TEST_TYPE) tests on Python $(PYTHON_VERSION)'
            condition: succeeded()
            env:
              AZURE_DEVOPS_TOKEN: $(System.AccessToken)
            
          - task: PublishTestResults@2
            inputs:
              testResultsFormat: 'JUnit'
              testResultsFiles: '$(Build.ArtifactStagingDirectory)/test_results/$(TEST_TYPE)-python$(PYTHON_VERSION)/*.xml'
              testRunTitle: '$(TEST_TYPE) Tests - Python $(PYTHON_VERSION)'
            displayName: 'Publish test results'
            condition: succeededOrFailed()
            
          - task: PublishBuildArtifacts@1
            inputs:
              PathtoPublish: '$(Build.ArtifactStagingDirectory)/test_results/$(TEST_TYPE)-python$(PYTHON_VERSION)'
              ArtifactName: 'test-results-$(TEST_TYPE)-python$(PYTHON_VERSION)'
              publishLocation: 'Container'
            displayName: 'Publish test artifacts'
            condition: succeededOrFailed()
      
      - job: AggregateResults
        dependsOn: MatrixTesting
        condition: succeededOrFailed()
        steps:
          - task: DownloadBuildArtifacts@0
            inputs:
              buildType: 'current'
              downloadType: 'specific'
              itemPattern: 'test-results-*/**'
              downloadPath: '$(System.ArtifactsDirectory)'
            displayName: 'Download test artifacts'
            
          - script: |
              mkdir -p $(Build.ArtifactStagingDirectory)/aggregated_results
              python -m distributed_testing.integration.ci_integration_runner \
                --provider azure \
                --coordinator $(COORDINATOR_URL) \
                --api-key $(COORDINATOR_API_KEY) \
                --aggregate-reports \
                --report-dir $(System.ArtifactsDirectory) \
                --output-dir $(Build.ArtifactStagingDirectory)/aggregated_results
            displayName: 'Aggregate test results'
            env:
              AZURE_DEVOPS_TOKEN: $(System.AccessToken)
            
          - task: PublishBuildArtifacts@1
            inputs:
              PathtoPublish: '$(Build.ArtifactStagingDirectory)/aggregated_results'
              ArtifactName: 'aggregated-test-results'
              publishLocation: 'Container'
            displayName: 'Publish aggregated results'
            
          - task: PublishHtmlReport@1
            inputs:
              reportPath: '$(Build.ArtifactStagingDirectory)/aggregated_results/summary.html'
              tabName: 'Aggregated Test Report'
            displayName: 'Publish HTML report'
```

## API-Based Integration

For custom integration, use the CI provider classes directly:

```python
import anyio
import os
from distributed_testing.ci import CIProviderFactory

async def main():
    # Get available providers
    available_providers = CIProviderFactory.get_available_providers()
    print(f"Available providers: {available_providers}")  # github, gitlab, jenkins, azure, circleci, bitbucket, teamcity, travis
    
    # Choose a provider based on environment or configuration
    # Example configurations for different providers:
    
    configs = {
        "github": {
            "token": os.environ.get("GITHUB_TOKEN"),
            "repository": "owner/repo",
            "commit_sha": "1234567890abcdef"
        },
        "gitlab": {
            "token": os.environ.get("GITLAB_TOKEN"),
            "project_id": "12345",
            "ref": "main"
        },
        "jenkins": {
            "url": "https://jenkins.example.com/",
            "user": os.environ.get("JENKINS_USER"),
            "token": os.environ.get("JENKINS_TOKEN"),
            "job_name": "test-job",
            "build_id": "123"
        },
        "azure": {
            "org_url": "https://dev.azure.com/organization",
            "project": "project-name",
            "token": os.environ.get("AZURE_TOKEN"),
            "build_id": "456"
        },
        "circleci": {
            "token": os.environ.get("CIRCLE_TOKEN"),
            "project_slug": "github/owner/repo",
            "build_num": os.environ.get("CIRCLE_BUILD_NUM")
        },
        "bitbucket": {
            "username": os.environ.get("BB_USERNAME"),
            "app_password": os.environ.get("BB_APP_PASSWORD"),
            "workspace": "workspace-name",
            "repository": "repo-name",
            "commit_hash": "abcdef1234567890"
        },
        "teamcity": {
            "url": "https://teamcity.example.com",
            "username": os.environ.get("TC_USERNAME"),
            "password": os.environ.get("TC_PASSWORD"),
            "build_id": "123"
        },
        "travis": {
            "token": os.environ.get("TRAVIS_TOKEN"),
            "repository": "owner/repo",
            "build_id": os.environ.get("TRAVIS_BUILD_ID")
        }
    }
    
    # Choose provider based on environment
    provider_type = os.environ.get("CI_PROVIDER", "github")
    
    # Create CI provider using factory
    provider = await CIProviderFactory.create_provider(
        provider_type,
        configs[provider_type]
    )
    
    # Create a test run
    test_run = await provider.create_test_run({
        "name": "Example Test Run",
        "build_id": "12345"
    })
    
    # Update test run status
    await provider.update_test_run(
        test_run["id"],
        {
            "status": "running",
            "summary": {
                "total_tasks": 10,
                "task_statuses": {
                    "completed": 5,
                    "running": 3,
                    "failed": 2
                },
                "duration": 15.2
            }
        }
    )
    
    # Add PR comment if supported by the provider
    if hasattr(provider, "add_pr_comment"):
        await provider.add_pr_comment(
            "123",
            "## Test Results\n\n8/10 tests passed in 25.5 seconds"
        )
    
    # Upload artifact
    await provider.upload_artifact(
        test_run["id"],
        "test_results.json",
        "Test Results"
    )
    
    # Record performance metrics for analysis
    await provider.record_performance_metric(
        test_run_id=test_run["id"],
        task_id="task-123",
        metric_name="execution_time",
        metric_value=0.75,
        unit="seconds"
    )
    
    # Analyze performance trends
    trends = await provider.analyze_performance_trends(
        metric_name="execution_time",
        grouping="task_type"
    )
    print(f"Performance trends: {trends}")
    
    # Complete the test run
    await provider.update_test_run(
        test_run["id"],
        {
            "status": "completed",
            "summary": {
                "total_tasks": 10,
                "task_statuses": {
                    "completed": 8,
                    "failed": 2
                },
                "duration": 25.5
            }
        }
    )
    
    # Clean up
    await provider.close()

anyio.run(main)
```

## Configuration Options

The CI/CD integration plugin supports the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `ci_system` | CI system to use (auto, github, jenkins, gitlab, azure) | auto |
| `api_token` | API token for authentication | None |
| `repository` | Repository name/path | None |
| `project` | Project ID/name (for GitLab/Azure) | None |
| `commit_sha` | Commit SHA for status updates | auto-detected |
| `branch` | Branch name | auto-detected |
| `pr_number` | Pull request number | auto-detected |
| `update_interval` | Status update interval in seconds | 60 |
| `update_on_completion_only` | Only update status on completion | false |
| `artifact_dir` | Directory for storing artifacts | test_results |
| `result_format` | Result format (junit, json, html, all) | junit |
| `enable_status_updates` | Enable CI status updates | true |
| `enable_pr_comments` | Enable PR comments | true |
| `enable_artifacts` | Enable artifact upload | true |
| `detailed_logging` | Enable detailed logging | false |
       error TEXT,
       FOREIGN KEY (run_id) REFERENCES test_runs (id)
   )
   ```

3. **performance_metrics**: Stores performance metrics for tasks and test runs
   ```sql
   CREATE TABLE performance_metrics (
       id TEXT PRIMARY KEY,
       run_id TEXT,
       task_id TEXT,
       metric_name TEXT,
       metric_value REAL,
       unit TEXT,
       recorded_at TEXT,
       FOREIGN KEY (run_id) REFERENCES test_runs (id),
       FOREIGN KEY (task_id) REFERENCES test_tasks (id)
   )
   ```

### Recording Performance Metrics

Performance metrics can be recorded using the `record_performance_metric` method:

```python
await ci_client.record_performance_metric(
    test_run_id="test-run-123",
    task_id="task-456",
    metric_name="execution_time",
    metric_value=1.23,
    unit="seconds"
)
```

Common metrics to record include:
- **execution_time**: How long a task took to execute
- **memory_usage**: Memory usage during task execution
- **cpu_usage**: CPU utilization during task execution
- **network_io**: Network I/O during task execution
- **disk_io**: Disk I/O during task execution
- **accuracy**: Model accuracy or test precision metrics

### Analyzing Performance Trends

The `analyze_performance_trends` method provides powerful analysis of performance trends:

```python
trends = await ci_client.analyze_performance_trends(
    metric_name="execution_time",
    grouping="branch",  # branch, commit, task_type
    timeframe="1w",     # 1d, 1w, 1m, all
    limit=5
)
```

This returns a comprehensive analysis including:
- **Overall statistics**: Average, min, max, and standard deviation across all data
- **Group statistics**: Performance broken down by the specified grouping
- **Trend direction**: Whether performance is improving or degrading
- **Statistical significance**: Whether the trend is statistically significant

### Querying Test History

The test history can be queried using the `get_test_history` method:

```python
history = await ci_client.get_test_history(
    limit=10,
    branch="main",
    commit_sha="1234567890abcdef"
)
```

This returns a list of test runs with their associated metadata, making it easy to track test performance over time and identify trends or regressions.

## Advanced Usage

### JavaScript SDK Integration

The CI/CD integration also works with the new JavaScript SDK (`ipfs_accelerate_js`) for WebNN/WebGPU implementations:

```javascript
// In your JavaScript SDK code
import { TestReporter } from 'ipfs_accelerate_js/core/testing';

// Initialize test reporter with CI integration
const reporter = new TestReporter({
  ciSystem: 'github', 
  testRunId: process.env.TEST_RUN_ID,
  apiToken: process.env.GITHUB_TOKEN,
  repository: process.env.GITHUB_REPOSITORY
});

// Report test results from JavaScript SDK
async function runWebGPUTests() {
  try {
    // Run your WebGPU tests
    const results = await webgpuTests.run();
    
    // Report results back to CI system
    await reporter.reportResults({
      totalTests: results.total,
      passedTests: results.passed,
      failedTests: results.failed,
      skippedTests: results.skipped,
      duration: results.duration
    });
    
    return results.failed === 0;
  } catch (error) {
    await reporter.reportFailure(error.message);
    return false;
  }
}
```

You can trigger the JavaScript tests from your Python CI/CD integration:

```python
# Run JavaScript SDK tests with CI integration
from distributed_testing.cicd_integration import CICDIntegration
from distributed_testing.js_sdk_runner import JavaScriptSDKRunner

integration = CICDIntegration(
    coordinator_url="http://coordinator-url:8080",
    api_key="YOUR_API_KEY",
    provider="github"
)

# Create test run
test_run_id = integration.create_test_run()

# Run JavaScript SDK tests
js_runner = JavaScriptSDKRunner()
results = js_runner.run_tests(
    test_dir="./js_tests",
    env={
        "TEST_RUN_ID": test_run_id,
        "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN"),
        "GITHUB_REPOSITORY": os.environ.get("GITHUB_REPOSITORY")
    }
)

# Update test results
integration.update_test_results(test_run_id, results)
```

### Custom CI Provider Implementation

To add support for a new CI system, implement the `CIProviderInterface`:

```python
from distributed_testing.ci.api_interface import CIProviderInterface

class MyCIProvider(CIProviderInterface):
    """Custom CI provider implementation."""
    
    async def initialize(self, config):
        # Initialize provider with configuration
        return True
    
    async def create_test_run(self, test_run_data):
        # Create a test run
        return {"id": "test-run-id", "status": "running"}
    
    # Implement other required methods...
    
    async def close(self):
        # Clean up resources
        pass

# Register provider with factory
from distributed_testing.ci import CIProviderFactory
CIProviderFactory.register_provider("my-ci", MyCIProvider)
```

### Environment-Dependent Configuration

For multi-environment CI setup:

```python
import os
from distributed_testing.ci import CIProviderFactory

async def create_ci_provider():
    # Detect environment
    if os.environ.get("GITHUB_ACTIONS") == "true":
        provider_type = "github"
        config = {
            "token": os.environ.get("GITHUB_TOKEN"),
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "commit_sha": os.environ.get("GITHUB_SHA")
        }
    elif os.environ.get("GITLAB_CI") == "true":
        provider_type = "gitlab"
        config = {
            "token": os.environ.get("GITLAB_TOKEN"),
            "project": os.environ.get("CI_PROJECT_PATH"),
            "commit_sha": os.environ.get("CI_COMMIT_SHA")
        }
    else:
        provider_type = "generic"
        config = {}
    
    # Create provider
    return await CIProviderFactory.create_provider(provider_type, config)
```

## Troubleshooting

### Common Issues and Solutions

#### Authentication Failures

**Symptom**: "Authentication failed" or "Unauthorized" errors when connecting to CI/CD systems

**Common error messages**:
- GitHub: `401 Unauthorized: Bad credentials`
- GitLab: `HTTP Error 401: Unauthorized`
- Jenkins: `401 Client Error: Unauthorized for url`
- Azure DevOps: `TF400813: The user '***' is not authorized to access this resource`

**Solutions**:
- Check that the API token is correct and has the required permissions
  ```bash
  # Test GitHub token
  curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user
  
  # Test GitLab token
  curl -H "PRIVATE-TOKEN: YOUR_TOKEN" https://gitlab.com/api/v4/user
  ```
- Ensure the token is provided in the correct format (Bearer, Basic, PAT formats vary)
- Verify that the token has not expired (GitHub tokens, PATs have expiration)
- Check for proper scopes:
  - GitHub: Need `repo` scope for private repositories
  - GitLab: Need `api` scope for API access
  - Azure: Need proper PAT permissions for test runs
- Validate environment variables are correctly set and accessible

**Code example for token debugging**:
```python
import os
import requests

def test_github_token(token=None):
    """Test if a GitHub token is valid."""
    token = token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("No token provided or found in GITHUB_TOKEN environment variable")
        return False
        
    headers = {"Authorization": f"token {token}"}
    response = requests.get("https://api.github.com/user", headers=headers)
    
    if response.status_code == 200:
        user_data = response.json()
        print(f"Token is valid for user: {user_data['login']}")
        return True
    else:
        print(f"Token validation failed: {response.status_code} - {response.text}")
        return False
```

#### Missing Test Results

**Symptom**: Tests run but results are not reported to CI system

**Diagnostic steps**:
1. Confirm test run creation:
   ```bash
   # Find logs related to test run creation
   grep "Created test run" distributed_testing.log
   ```

2. Verify test result reporting:
   ```bash
   # Check for update attempts
   grep "Updating test run" distributed_testing.log
   
   # Look for errors during update
   grep "Error updating" distributed_testing.log
   ```

**Solutions**:
- Check that the CI provider is correctly configured
- Verify that the coordinator is running and accessible with network check:
  ```bash
  curl -I http://coordinator-url:8080/health
  ```
- Ensure that the test run ID is correctly passed between components
- Check permissions to create and update test runs
- Verify report formats match what the CI system expects
- Examine path configurations for artifact uploads
- For JUnit reports, verify XML format is valid:
  ```bash
  xmllint --noout path/to/report.xml
  ```

#### PR Comments Not Appearing

**Symptom**: Test results are not posted as PR comments

**Diagnostic checklist**:
- Is `enable_pr_comments` set to `true`?
- Is PR number correctly identified? (Check logs for "PR number detected: XXX")
- Does the token have permission to post comments? 
- For GitHub, is the repo using default PR template that might conflict?

**Solutions**:
- Check that `enable_pr_comments` is set to `true`
- Verify that the API token has permission to post comments
  - GitHub: Needs `repo` or `pull_request` scope
  - GitLab: Needs `api` scope
  - Azure DevOps: Needs proper PAT permissions
- Ensure that the PR number is correctly identified:
  ```python
  # For GitHub
  from distributed_testing.ci import GitHubClient
  
  # Test PR commenting
  async def test_pr_comment(token, repo, pr_number):
      client = GitHubClient()
      await client.initialize({"token": token, "repository": repo})
      result = await client.add_pr_comment(pr_number, "Test comment")
      print(f"Comment result: {result}")
  ```
- Verify the PR is still open (can't comment on merged/closed PRs in some systems)

#### Execution Timeout Issues

**Symptom**: Tests timeout or fail due to execution time constraints

**Solutions**:
- Increase timeouts in the distributed testing configuration:
  ```bash
  python -m distributed_testing.cicd_integration \
      --provider github \
      --coordinator http://coordinator-url:8080 \
      --api-key $GITHUB_TOKEN \
      --test-dir ./tests \
      --timeout 7200  # Increase to 2 hours
  ```
- Configure CI/CD system timeouts:
  - GitHub Actions: Use `timeout-minutes` in workflow
  - GitLab CI: Set `timeout` in job configuration
  - Azure DevOps: Set `timeoutInMinutes` for job
- Implement test sharding to split tests across multiple jobs
- Use a keep-alive mechanism for long-running tests

#### Inconsistent Results Across CI Systems

**Symptom**: Tests pass in one CI system but fail in another

**Diagnostic approach**:
1. Compare environment variables between systems
2. Check for platform-specific dependencies
3. Verify test execution order differences
4. Compare resource allocations (memory, CPU)

**Solutions**:
- Use containerization (Docker) to ensure consistent environments
- Set explicit environment variables for all systems
- Implement platform detection and conditional test execution
- Configure resource limits consistently across systems
- Use matrix testing to identify environment-specific issues

### Advanced Troubleshooting

#### Diagnostic Decision Tree

Use this decision tree to diagnose common issues:

```
Is the test running in CI/CD system?
├── NO → Check CI/CD configuration and triggers
├── YES → Is the coordinator accessible?
    ├── NO → Check network connectivity and firewall rules
    ├── YES → Are authentication credentials valid?
        ├── NO → Update/rotate credentials and verify permissions
        ├── YES → Are tests executing correctly?
            ├── NO → Check test code and dependencies
            ├── YES → Are results being reported?
                ├── NO → Check result collection and reporting configuration
                ├── YES → Are results visible in CI/CD system?
                    ├── NO → Check CI/CD system configuration and permissions
                    └── YES → Problem resolved!
```

#### Common Error Messages and Solutions

| Error Message | System | Cause | Solution |
|---------------|--------|-------|----------|
| `401 Unauthorized` | GitHub | Invalid or expired token | Generate new token with required scopes |
| `404 Not Found` for repository | GitHub | Repository doesn't exist or no access | Check repo name and permissions |
| `TF400813: Resource not authorized` | Azure | Insufficient permissions | Update PAT with correct scopes |
| `Could not find the test run` | All | Test run ID incorrect or deleted | Verify ID persistence between steps |
| `JUnit XML format error` | All | Invalid XML in test results | Validate XML format before submission |
| `Rate limit exceeded` | GitHub | Too many API requests | Implement backoff strategy or increase limits |
| `Socket timeout` | All | Network issues or service unavailable | Check connectivity and service status |
| `No such file or directory` | All | Missing report files | Verify file paths and generation |

#### Logging and Diagnostics

Enable comprehensive logging for troubleshooting:

```bash
# Enable detailed logging
python -m distributed_testing.cicd_integration \
    --provider github \
    --coordinator http://coordinator-url:8080 \
    --api-key $GITHUB_TOKEN \
    --test-dir ./tests \
    --detailed-logging true \
    --log-level debug \
    --log-file ci_integration.log

# View logs with filtering
grep "ERROR" ci_integration.log
grep "API request" ci_integration.log | grep -i failed
```

For API request inspection, use the debug proxy approach:

```python
# In your environment or test script
os.environ["DISTRIBUTED_TESTING_DEBUG_PROXY"] = "1"
os.environ["DISTRIBUTED_TESTING_DEBUG_PROXY_PORT"] = "8888"

# Start the debug proxy before running tests
from distributed_testing.debug import start_proxy_server
start_proxy_server()

# Then run your tests with requests going through the debug proxy
```

### Debugging Tips

1. **Enable Detailed Logging**: Set `detailed_logging` to `true` and configure `log-level` to `debug` for more verbose logs

2. **Analyze HTTP Requests**:
   - Use the debug proxy feature to capture all HTTP requests
   - Inspect request/response headers and bodies
   - Compare successful vs. failed requests

3. **Environment Validation**:
   - Print environment variables to ensure they are available and correct
   ```python
   import os
   for key in ["GITHUB_TOKEN", "GITHUB_REPOSITORY", "GITHUB_SHA"]:
       value = os.environ.get(key, "NOT SET")
       masked_value = value[:4] + "***" if len(value) > 4 else value
       print(f"{key}: {masked_value}")
   ```

4. **Component Isolation**:
   - Test CI provider independently from test execution
   - Use standalone scripts to verify individual functions
   - Test local execution before CI system integration

5. **CI System Diagnostics**:
   - Review CI system logs for errors or warnings
   - Check API rate limits and quotas
   - Verify CI system service status

6. **Interactive Debugging**:
   - Add steps to export state between actions for inspection
   - Use debug jobs with command-line access
   - Add verbose status reporting steps

## API Reference

### CIProviderInterface

```python
class CIProviderInterface(abc.ABC):
    @abc.abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the CI provider with configuration."""
    
    @abc.abstractmethod
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new test run in the CI system."""
    
    @abc.abstractmethod
    async def update_test_run(self, test_run_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a test run in the CI system."""
    
    @abc.abstractmethod
    async def add_pr_comment(self, pr_number: str, comment: str) -> bool:
        """Add a comment to a pull request."""
    
    @abc.abstractmethod
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """Upload an artifact for a test run."""
    
    @abc.abstractmethod
    async def get_test_run_status(self, test_run_id: str) -> Dict[str, Any]:
        """Get the status of a test run."""
    
    @abc.abstractmethod
    async def set_build_status(self, status: str, description: str) -> bool:
        """Set the build status in the CI system."""
    
    @abc.abstractmethod
    async def close(self) -> None:
        """Close the CI provider and clean up resources."""
```

### TestRunResult

```python
class TestRunResult:
    def __init__(
        self,
        test_run_id: str,
        status: str,
        total_tests: int,
        passed_tests: int,
        failed_tests: int,
        skipped_tests: int,
        duration_seconds: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a test run result."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestRunResult':
        """Create from dictionary."""
```

### CIProviderFactory

```python
class CIProviderFactory:
    @classmethod
    def register_provider(cls, provider_type: str, provider_class: type) -> None:
        """Register a CI provider class."""
    
    @classmethod
    async def create_provider(cls, provider_type: str, config: Dict[str, Any]) -> CIProviderInterface:
        """Create a CI provider instance."""
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider types."""
```