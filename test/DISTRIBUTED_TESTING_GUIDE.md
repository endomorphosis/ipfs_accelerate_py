# Distributed Testing Framework User Guide

## Introduction

The Distributed Testing Framework enables parallel execution of benchmarks and tests across multiple machines with heterogeneous hardware. This guide provides detailed instructions for setting up, configuring, and using the framework in different scenarios.

### New Features (March 2025)

The Distributed Testing Framework has been enhanced with the following features:

1. **Coordinator Redundancy and Failover**: Multiple coordinator instances can now operate in a high-availability configuration with automatic leader election and state synchronization.

2. **Performance Trend Analysis**: Long-term performance trend tracking with statistical analysis, anomaly detection, and predictive forecasting for worker and task performance.

3. **Advanced Fault Tolerance**: Enhanced recovery mechanisms for worker and coordinator failures with automatic state recovery.

4. **Visualization System**: Visual representations of performance metrics and trends with interactive charts and reports.

5. **Time-Series Data Storage**: Historical performance data storage with efficient querying and aggregation capabilities.

6. **CI/CD Integration**: Seamless integration with GitHub Actions, GitLab CI, and Jenkins for automated testing and reporting with intelligent test discovery and requirement analysis.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Network connectivity between coordinator and worker machines
- DuckDB (optional but recommended for result storage)
- Websockets library for Python
- Required hardware drivers on worker machines

### Installation

The Distributed Testing Framework is integrated into the IPFS Accelerate Python project. No separate installation is required, but you need to ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Components Overview

The Distributed Testing Framework consists of several components:

1. **Coordinator Server**: Central server that distributes tasks and collects results
2. **Worker Nodes**: Machines that execute tasks and report results
3. **Dashboard Server**: Web interface for monitoring and managing the system
4. **Test Runner**: Command-line tool for running tests and submitting tasks
5. **Auto Recovery System**: Manages coordinator redundancy and failover for high availability
6. **Health Monitor**: Tracks health status of workers and coordinators, triggering automatic recovery actions
7. **Performance Trend Analyzer**: Tracks, analyzes and visualizes performance metrics over time
8. **Load Balancer**: Efficiently distributes workload across workers based on capabilities and performance history
9. **CI/CD Integration**: Enables automated testing within CI/CD pipelines with test discovery and requirement analysis

## Basic Usage

### Running the Coordinator Server

The coordinator server is the central component that manages workers and distributes tasks:

```bash
# Start with default settings
python duckdb_api/distributed_testing/coordinator.py

# Start with specific host and port
python duckdb_api/distributed_testing/coordinator.py --host 0.0.0.0 --port 8080

# Start with database integration
python duckdb_api/distributed_testing/coordinator.py --host 0.0.0.0 --port 8080 --db-path ./benchmark_db.duckdb

# Generate a worker API key
python duckdb_api/distributed_testing/coordinator.py --generate-worker-key --security-config ./security_config.json
```

Common options:
- `--host`: Host to bind the server to (default: localhost)
- `--port`: Port to bind the server to (default: 8080)
- `--db-path`: Path to DuckDB database for result storage
- `--heartbeat-timeout`: Timeout in seconds for worker heartbeats (default: 60)
- `--security-config`: Path to security configuration file
- `--generate-worker-key`: Generate an API key for worker authentication

### Running Worker Nodes

Worker nodes execute tasks assigned by the coordinator:

```bash
# Start a worker node
python duckdb_api/distributed_testing/worker.py --coordinator http://coordinator-host:8080 --api-key YOUR_API_KEY

# Start with a specific worker ID
python duckdb_api/distributed_testing/worker.py --coordinator http://coordinator-host:8080 --api-key YOUR_API_KEY --worker-id worker1

# Start with a specific working directory
python duckdb_api/distributed_testing/worker.py --coordinator http://coordinator-host:8080 --api-key YOUR_API_KEY --work-dir ./worker_tasks

# Start with verbose logging
python duckdb_api/distributed_testing/worker.py --coordinator http://coordinator-host:8080 --api-key YOUR_API_KEY --verbose
```

Common options:
- `--coordinator`: URL of the coordinator server (required)
- `--api-key`: API key for authentication (required)
- `--worker-id`: Worker ID (generated if not provided)
- `--work-dir`: Working directory for tasks
- `--reconnect-interval`: Interval in seconds between reconnection attempts (default: 5)
- `--heartbeat-interval`: Interval in seconds between heartbeats (default: 30)
- `--verbose`: Enable verbose logging

### Running the Dashboard

The dashboard provides a web interface for monitoring and managing the system:

```bash
# Start the dashboard server
python duckdb_api/distributed_testing/dashboard_server.py --coordinator-url http://coordinator-host:8080

# Start with specific host and port
python duckdb_api/distributed_testing/dashboard_server.py --host 0.0.0.0 --port 8081 --coordinator-url http://coordinator-host:8080

# Start and automatically open in browser
python duckdb_api/distributed_testing/dashboard_server.py --coordinator-url http://coordinator-host:8080 --auto-open
```

Common options:
- `--host`: Host to bind the server to (default: localhost)
- `--port`: Port to bind the server to (default: 8081)
- `--coordinator-url`: URL of the coordinator server (required)
- `--auto-open`: Automatically open the dashboard in a web browser

### Integrated Testing Environment

For easy testing, you can use the `run_test.py` script to start all components:

```bash
# Start all components (coordinator, workers, dashboard)
python duckdb_api/distributed_testing/run_test.py --mode all

# Start with specific host and ports
python duckdb_api/distributed_testing/run_test.py --mode all --host localhost --port 8080 --dashboard-port 8081

# Start with more workers
python duckdb_api/distributed_testing/run_test.py --mode all --worker-count 4

# Start with database integration
python duckdb_api/distributed_testing/run_test.py --mode all --db-path ./benchmark_db.duckdb
```

Common options:
- `--mode`: Operation mode (all, coordinator, worker, client, dashboard)
- `--host`: Host to bind servers to (default: localhost)
- `--port`: Port for coordinator (default: 8080)
- `--dashboard-port`: Port for dashboard (default: 8081)
- `--db-path`: Path to DuckDB database
- `--worker-count`: Number of worker nodes to start (default: 2)
- `--dashboard-auto-open`: Automatically open dashboard in web browser

## Submitting Tasks

You can submit tasks to the coordinator using the `run_test.py` script in client mode:

```bash
# Submit a test task
python duckdb_api/distributed_testing/run_test.py --mode client --coordinator http://coordinator-host:8080 --test-file /path/to/test_file.py

# Submit with arguments
python duckdb_api/distributed_testing/run_test.py --mode client --coordinator http://coordinator-host:8080 --test-file /path/to/test_file.py --test-args "--verbose --batch-size 4"

# Submit with priority (lower is higher priority)
python duckdb_api/distributed_testing/run_test.py --mode client --coordinator http://coordinator-host:8080 --test-file /path/to/test_file.py --priority 1

# Submit and wait for completion
python duckdb_api/distributed_testing/run_test.py --mode client --coordinator http://coordinator-host:8080 --test-file /path/to/test_file.py --timeout 600
```

Common options:
- `--coordinator`: URL of the coordinator server (required)
- `--test-file`: Path to the test file (required)
- `--test-args`: Arguments for the test
- `--priority`: Priority of the task (lower is higher priority, default: 5)
- `--timeout`: Timeout in seconds (default: 600)

## Task Format

Tasks are defined in JSON format. Here's an example of a benchmark task:

```json
{
  "type": "benchmark",
  "priority": 1,
  "config": {
    "model": "bert-base-uncased",
    "batch_sizes": [1, 2, 4, 8, 16],
    "precision": "fp16",
    "iterations": 100
  },
  "requirements": {
    "hardware": ["cuda"],
    "min_memory_gb": 8,
    "min_cuda_compute": 7.5
  },
  "timeout_seconds": 1800,
  "retry_policy": {
    "max_retries": 3,
    "retry_delay_seconds": 60
  }
}
```

Key fields:
- `type`: Type of task (benchmark, test, command)
- `priority`: Priority of the task (lower is higher priority)
- `config`: Configuration specific to the task type
- `requirements`: Hardware requirements for the task
- `timeout_seconds`: Maximum execution time in seconds
- `retry_policy`: Configuration for task retries

## Advanced Usage

### Task Dependencies

You can define dependencies between tasks:

```json
{
  "type": "benchmark",
  "priority": 1,
  "config": {
    "model": "bert-base-uncased",
    "batch_sizes": [1, 2, 4, 8, 16]
  },
  "dependencies": ["prepare-model-task-123"]
}
```

The task will only be executed after all dependencies have completed successfully.

### Hardware Requirements

Specify hardware requirements for optimal task assignment:

```json
{
  "type": "benchmark",
  "requirements": {
    "hardware": ["cuda", "rocm"],  // Either CUDA or ROCm
    "min_memory_gb": 8,            // At least 8GB memory
    "min_cuda_compute": 7.5,       // For CUDA GPUs
    "browser": "firefox",          // Specific browser
    "device_type": "mobile"        // Device type
  }
}
```

The task scheduler will match these requirements against worker capabilities.

### Task Types

The framework supports different task types:

1. **Benchmark Tasks**: For performance benchmarking
   ```json
   {
     "type": "benchmark",
     "config": {
       "model": "bert-base-uncased",
       "batch_sizes": [1, 2, 4, 8, 16],
       "precision": "fp16",
       "iterations": 100
     }
   }
   ```

2. **Test Tasks**: For running test files
   ```json
   {
     "type": "test",
     "config": {
       "test_file": "/path/to/test_file.py",
       "test_args": ["--verbose", "--batch-size", "4"]
     }
   }
   ```

3. **Command Tasks**: For running arbitrary commands
   ```json
   {
     "type": "command",
     "config": {
       "command": "python -m benchmark_script.py --model bert"
     }
   }
   ```

## Dashboard Features

The web dashboard provides several features for monitoring and managing the system:

1. **System Summary**: Overview of workers and tasks
2. **Worker Status**: Real-time status of worker nodes
3. **Task Status**: Status of tasks in queue, running, and completed
4. **Performance Metrics**: Statistics about worker and task performance
5. **Alerts**: System alerts and notifications

## Security

The framework includes several security features:

1. **API Key Authentication**: Worker nodes authenticate using API keys
2. **JWT Tokens**: Secure communication with JSON Web Tokens
3. **Role-Based Access**: Different permissions for different roles
4. **Secure Configuration**: Configuration for security settings

### Generating API Keys

```bash
# Generate a worker API key
python duckdb_api/distributed_testing/coordinator.py --generate-worker-key --security-config ./security_config.json
```

This will create a security configuration file with an API key for worker authentication.

## Troubleshooting

### Common Issues

1. **Connection Errors**:
   - Verify network connectivity between coordinator and workers
   - Check firewall settings to ensure ports are open
   - Verify correct coordinator URL in worker configuration

2. **Authentication Failures**:
   - Ensure the API key is correct and not expired
   - Check security configuration file for proper format
   - Regenerate API key if necessary

3. **Task Execution Failures**:
   - Check worker logs for error messages
   - Verify task requirements match worker capabilities
   - Check for environment issues on worker nodes

### Logging

Enable verbose logging for more detailed information:

```bash
# Enable verbose logging for coordinator
python duckdb_api/distributed_testing/coordinator.py --verbose

# Enable verbose logging for worker
python duckdb_api/distributed_testing/worker.py --verbose
```

Logs include important information for diagnosing issues.

## Performance Optimization

### Worker Optimization

1. **Hardware Matching**: Ensure workers have appropriate hardware for tasks
2. **Worker Capacity**: Configure maximum concurrent tasks per worker
3. **Resource Limits**: Set appropriate resource limits for workers

### Task Optimization

1. **Task Granularity**: Divide large tasks into smaller ones
2. **Task Dependencies**: Use dependencies for complex workflows
3. **Priority Settings**: Assign appropriate priorities to tasks

## Integration with Existing Systems

The Distributed Testing Framework can be integrated with existing systems:

1. **API Integration**: Use the coordinator API for custom integrations
2. **Database Integration**: Use DuckDB for result storage and analysis
3. **Monitoring Integration**: Connect to external monitoring systems

## Implementation Status

| Component | Status | Features |
|-----------|--------|----------|
| Core Infrastructure | ✅ 100% | Task distribution, worker management, database integration |
| Security | ✅ 100% | JWT authentication, API keys, role-based access control |
| Intelligent Task Distribution | ✅ 100% | Hardware-aware routing, requirements matching |
| Adaptive Load Balancing | ✅ 100% | Dynamic reassessment, performance-based balancing |
| Health Monitoring | ✅ 100% | Worker health tracking, auto-recovery, anomaly detection |
| Dashboard | ✅ 100% | Real-time monitoring, visualizations, system management |
| Coordinator Redundancy | ✅ 100% | Leader election, state replication, automatic failover |
| Performance Trend Analysis | ✅ 100% | Time-series tracking, statistical analysis, anomaly detection |
| CI/CD Integration | ✅ 100% | GitHub Actions, GitLab CI, Jenkins integration |

All components have now been fully implemented as of March 15, 2025.

## New Features (March 2025)

### Coordinator Redundancy and Failover

The Auto Recovery System enables high availability for coordinators by allowing multiple coordinator instances to work together:

```bash
# Run coordinator with auto recovery enabled
python duckdb_api/distributed_testing/coordinator.py --auto-recovery --coordinator-id coordinator1

# Run a second coordinator as part of the same cluster
python duckdb_api/distributed_testing/coordinator.py --auto-recovery --coordinator-id coordinator2 --coordinator-addresses http://coordinator1-host:8080
```

Configuration options:
- `--auto-recovery`: Enable auto recovery system
- `--coordinator-id`: Unique identifier for this coordinator instance
- `--coordinator-addresses`: Comma-separated list of other coordinator addresses
- `--failover-enabled`: Enable automatic failover (default: true)
- `--auto-leader-election`: Enable automatic leader election (default: true)

Features:
- Automatic leader election using distributed consensus
- State replication between coordinator instances
- Automatic failover when the leader fails
- Task state persistence and recovery
- Worker reassignment during failover

### Performance Trend Analysis

The Performance Trend Analyzer provides long-term tracking and analysis of performance metrics:

```bash
# Enable performance trend analyzer on coordinator
python duckdb_api/distributed_testing/coordinator.py --performance-analyzer --visualization-path ./performance_visualizations

# Generate a performance report
python duckdb_api/distributed_testing/performance_trend_analyzer.py --report --output performance_report.html

# Generate visualizations for a specific worker
python duckdb_api/distributed_testing/performance_trend_analyzer.py --visualize --entity-id worker1 --entity-type worker --output worker1_trends.html

# Export performance data
python duckdb_api/distributed_testing/performance_trend_analyzer.py --export --format json --output performance_data.json
```

Features:
- Time-series tracking of performance metrics
- Statistical trend analysis with significance testing
- Anomaly detection based on z-scores
- Performance prediction and forecasting
- Visualization of performance trends
- Integration with the dashboard for real-time monitoring

Available metrics include:
- Execution time trends
- Success rate changes
- Resource utilization patterns
- Worker specialization effectiveness
- Task throughput performance

### Advanced Fault Tolerance

The enhanced Health Monitor provides advanced fault tolerance features:

```bash
# Configure health monitor with enhanced settings
python duckdb_api/distributed_testing/coordinator.py --advanced-health-monitoring --recovery-attempts 5 --recovery-cooldown 1200
```

Features:
- Intelligent error classification and categorization
- Advanced recovery strategies based on error type
- Automatic worker quarantine for consistently failing nodes
- Performance anomaly detection and resolution
- Proactive health checks and preventive actions
- Hardware-specific recovery strategies

### Using Multiple Coordinators

To take advantage of coordinator redundancy:

1. Start the first coordinator as the initial leader:
   ```bash
   python duckdb_api/distributed_testing/coordinator.py --auto-recovery --coordinator-id coordinator1
   ```

2. Start additional coordinators in the same cluster:
   ```bash
   python duckdb_api/distributed_testing/coordinator.py --auto-recovery --coordinator-id coordinator2 --coordinator-addresses http://coordinator1-host:8080
   ```

3. Workers can connect to any coordinator in the cluster:
   ```bash
   python duckdb_api/distributed_testing/worker.py --coordinator http://coordinator1-host:8080,http://coordinator2-host:8080 --api-key YOUR_API_KEY
   ```

4. Monitor the cluster status:
   ```bash
   python duckdb_api/distributed_testing/dashboard_server.py --coordinator-url http://coordinator1-host:8080 --show-cluster-status
   ```

If the leader coordinator fails, the system will automatically elect a new leader and redirect worker connections.

## Performance History Visualization

The Performance Trend Analyzer can generate various visualizations:

- **Time Series Charts**: Performance metrics over time
- **Trend Analysis Charts**: Linear regression and forecasting
- **Anomaly Highlighting**: Visual representation of detected anomalies
- **Hardware Comparison Charts**: Performance across different hardware types
- **Task Type Comparison Charts**: Performance for different task types

Example:

```bash
# Generate comprehensive performance visualization
python duckdb_api/distributed_testing/performance_trend_analyzer.py --visualize-all --days 30 --output-dir ./visualizations
```

## CI/CD Integration

The Distributed Testing Framework provides seamless integration with popular CI/CD systems, enabling automated testing as part of your continuous integration workflows. The integration includes intelligent test discovery, hardware requirement analysis, and comprehensive result reporting.

### Using CI/CD Integration

The framework includes a dedicated module for CI/CD integration:

```bash
# GitHub Actions integration
python -m duckdb_api.distributed_testing.cicd_integration \
  --provider github \
  --coordinator http://coordinator-url:8080 \
  --api-key YOUR_API_KEY \
  --test-dir ./tests \
  --output-dir ./test_reports \
  --report-formats json md html \
  --verbose

# GitLab CI integration
python -m duckdb_api.distributed_testing.cicd_integration \
  --provider gitlab \
  --coordinator http://coordinator-url:8080 \
  --api-key YOUR_API_KEY \
  --test-pattern "tests/**/*_test.py" \
  --output-dir ./test_reports

# Jenkins integration
python -m duckdb_api.distributed_testing.cicd_integration \
  --provider jenkins \
  --coordinator http://coordinator-url:8080 \
  --api-key YOUR_API_KEY \
  --test-files test_file1.py test_file2.py \
  --output-dir ./test_reports
```

Common options:
- `--provider`: CI/CD provider (github, gitlab, jenkins, generic)
- `--coordinator`: URL of the coordinator server (required)
- `--api-key`: API key for authentication (required)
- `--test-dir`: Directory to search for tests (mutually exclusive with other test options)
- `--test-pattern`: Glob pattern for test files (mutually exclusive with other test options)
- `--test-files`: Explicit list of test files (mutually exclusive with other test options)
- `--output-dir`: Directory to write reports (defaults to current directory)
- `--report-formats`: Report formats to generate (json, md, html)
- `--timeout`: Maximum time to wait for test completion in seconds (default: 3600)
- `--poll-interval`: How often to poll for results in seconds (default: 15)
- `--verbose`: Enable verbose output

### Features

The CI/CD integration provides the following features:

1. **Automatic Test Discovery**: Discover test files based on directories, patterns, or explicit lists
2. **Intelligent Requirement Analysis**: Analyze test files to determine hardware, browser, and memory requirements
3. **Test Submission and Monitoring**: Submit tests to the coordinator and monitor execution
4. **Comprehensive Reporting**: Generate detailed reports in multiple formats (JSON, Markdown, HTML)
5. **CI/CD System Integration**: Seamless integration with GitHub Actions, GitLab CI, and Jenkins
6. **Appropriate Exit Codes**: Return appropriate exit codes for CI/CD pipeline success/failure
7. **Environment Variable Awareness**: Detect CI/CD-specific environment variables for context

### Example Workflow Files

The framework includes example workflow files for each supported CI/CD system:

#### GitHub Actions

Create a `.github/workflows/distributed-testing.yml` file:

```yaml
name: Distributed Testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      test_pattern:
        description: 'Test pattern to run'
        required: false
        default: ''
      hardware:
        description: 'Hardware to test on'
        required: false
        default: 'cpu'

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
      
      - name: Run distributed tests
        run: |
          python -m duckdb_api.distributed_testing.cicd_integration \
            --provider github \
            --coordinator ${{ secrets.COORDINATOR_URL }} \
            --api-key ${{ secrets.COORDINATOR_API_KEY }} \
            --test-pattern "${{ github.event.inputs.test_pattern || 'tests/**/*_test.py' }}" \
            --output-dir ./test_reports \
            --report-formats json md html
      
      - name: Upload test reports
        uses: actions/upload-artifact@v3
        with:
          name: test-reports
          path: ./test_reports
```

#### GitLab CI

Create a `.gitlab-ci.yml` file:

```yaml
stages:
  - test

variables:
  COORDINATOR_URL: "${COORDINATOR_URL}"
  API_KEY: "${COORDINATOR_API_KEY}"
  TEST_PATTERN: "tests/**/*_test.py"

distributed-tests:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements.txt
    - python -m duckdb_api.distributed_testing.cicd_integration \
        --provider gitlab \
        --coordinator ${COORDINATOR_URL} \
        --api-key ${API_KEY} \
        --test-pattern ${TEST_PATTERN} \
        --output-dir ./test_reports \
        --report-formats json md html
  artifacts:
    paths:
      - test_reports/
    expire_in: 1 week
```

#### Jenkins

Create a `Jenkinsfile`:

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.10'
        }
    }
    
    parameters {
        string(name: 'COORDINATOR_URL', defaultValue: 'http://coordinator-url:8080', description: 'URL of the coordinator server')
        password(name: 'API_KEY', description: 'API key for coordinator authentication')
        string(name: 'TEST_PATTERN', defaultValue: 'tests/**/*_test.py', description: 'Test pattern to run')
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Run Distributed Tests') {
            steps {
                sh '''
                python -m duckdb_api.distributed_testing.cicd_integration \
                  --provider jenkins \
                  --coordinator ${COORDINATOR_URL} \
                  --api-key ${API_KEY} \
                  --test-pattern "${TEST_PATTERN}" \
                  --output-dir ./test_reports \
                  --report-formats json md html \
                  --verbose
                '''
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'test_reports/**', allowEmptyArchive: true
            publishHTML(target: [
                allowMissing: true,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'test_reports',
                reportFiles: '*.html',
                reportName: 'Test Reports'
            ])
        }
    }
}
```

### Report Generation

The CI/CD integration generates comprehensive reports in multiple formats:

1. **JSON Reports**: Structured data for programmatic analysis
2. **Markdown Reports**: Human-readable documentation-style reports
3. **HTML Reports**: Interactive visualizations for web display

Reports include:
- Basic information about the run (timestamp, provider, build ID, etc.)
- Summary statistics (total tasks, completion status)
- Detailed results with status, duration, hardware, and error details

Example report structure:
```
# Distributed Testing Report

- **Timestamp:** 20250315_120000
- **Provider:** github
- **Build ID:** 12345678
- **Repository:** organization/repository
- **Branch:** main
- **Commit:** abcdef1234567890

## Summary

- **Total Tasks:** 25
- **Completed:** 23
- **Failed:** 1
- **Cancelled:** 0
- **Timeout:** 1

## Detailed Results

| Task ID | Test File | Status | Duration | Hardware | Details |
|---------|-----------|--------|----------|----------|--------|
| task1 | test_bert.py | COMPLETED | 125.3s | cuda | ✅ Success |
| task2 | test_vit.py | COMPLETED | 87.1s | cuda | ✅ Success |
| task3 | test_whisper.py | FAILED | 45.2s | cuda | ❌ AssertionError: Expected value 0.95, got 0.85 |
| task4 | test_llama.py | TIMEOUT | N/A | cuda | ⏱️ Test timed out |
...
```

### Advanced Configuration

For advanced CI/CD integration scenarios, consider the following configurations:

1. **Matrix Testing**: Set up tests across multiple hardware types or configurations
2. **Scheduled Testing**: Run tests on a regular schedule for performance tracking
3. **Selective Testing**: Run only tests affected by changes
4. **Hardware-Specific Branches**: Dedicated branches for hardware-specific testing

## Conclusion

The Distributed Testing Framework provides a powerful and flexible system for parallel test execution across multiple machines. With the new features for high availability, performance analysis, and advanced fault tolerance, it offers a robust solution for large-scale testing needs. By following this guide, you can set up and use the framework effectively for your testing requirements.