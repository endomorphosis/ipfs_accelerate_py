# Integration and Extensibility Guide

This document provides detailed information about the Integration and Extensibility features of the Distributed Testing Framework. These features enable the framework to integrate with external systems and be extended through plugins.

## Table of Contents

1. [Plugin Architecture](#plugin-architecture)
2. [CI/CD Integration](#ci-cd-integration)
3. [External System Integrations](#external-system-integrations)
4. [API Standardization](#api-standardization)
5. [Custom Scheduler Support](#custom-scheduler-support)
6. [Examples and Tutorials](#examples-and-tutorials)

## Plugin Architecture

The plugin architecture allows the framework to be extended with custom functionality without modifying the core codebase.

### Plugin Interface

All plugins implement the base `Plugin` interface:

```python
class Plugin:
    def initialize(self, coordinator: Coordinator) -> bool:
        """Initialize the plugin with the coordinator instance."""
        pass

    def shutdown(self) -> bool:
        """Perform cleanup operations when shutting down."""
        pass

    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the plugin with the provided configuration."""
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of this plugin."""
        pass
```

### Plugin Types

The framework supports several types of plugins:

| Plugin Type | Description | Interface |
|-------------|-------------|-----------|
| Task Processor | Custom task execution logic | `TaskProcessorPlugin` |
| Scheduler | Alternative scheduling algorithms | `SchedulerPlugin` |
| Results Handler | Custom result processing | `ResultsHandlerPlugin` |
| Resource Manager | Specialized hardware management | `ResourceManagerPlugin` |
| Notification | Custom notification mechanisms | `NotificationPlugin` |

### Plugin Discovery

Plugins can be discovered through several mechanisms:

- Entry point registration
- Directory scanning
- Manual registration

Example configuration:

```yaml
plugins:
  directories:
    - /path/to/plugins
  entry_points:
    - distributed_testing.plugins
  enabled:
    - MyCustomTaskProcessor
    - JiraIntegration
    - SlackNotifier
```

### Creating a Custom Plugin

Here's a simple example of creating a custom results handler plugin:

```python
from distributed_testing.plugins import ResultsHandlerPlugin

class CSVResultsHandler(ResultsHandlerPlugin):
    def initialize(self, coordinator):
        self.coordinator = coordinator
        self.output_file = None
        return True
        
    def configure(self, config):
        self.output_file = config.get("output_file", "results.csv")
        return True
        
    def handle_result(self, task_id, result):
        # Write result to CSV file
        with open(self.output_file, "a") as f:
            # Format result as CSV and write
            pass
        return True
        
    def get_capabilities(self):
        return {
            "formats": ["csv"],
            "supports_streaming": True
        }
```

## CI/CD Integration

The framework can integrate with various CI/CD systems to enable automated testing within CI/CD pipelines.

### GitHub Actions Integration

The framework provides a GitHub Actions workflow for running distributed tests:

```yaml
name: Distributed Testing

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
    
    - name: Set up coordinator
      run: |
        python -m distributed_testing.coordinator --db-path ./coordinator.duckdb --daemon
        
    - name: Run workers
      run: |
        python -m distributed_testing.worker --coordinator http://localhost:8080 --worker-id worker-1 &
        python -m distributed_testing.worker --coordinator http://localhost:8080 --worker-id worker-2 &
        
    - name: Submit tests
      run: |
        python -m distributed_testing.client submit-task --file tests/test_suite.json
        
    - name: Wait for completion
      run: |
        python -m distributed_testing.client wait-completion
        
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: ./test-results
```

### Jenkins Integration

For Jenkins, the framework provides a Pipeline script:

```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'python -m distributed_testing.coordinator --db-path ./coordinator.duckdb --daemon'
                sh 'python -m distributed_testing.worker --coordinator http://localhost:8080 --worker-id worker-1 &'
            }
        }
        
        stage('Run Tests') {
            steps {
                sh 'python -m distributed_testing.client submit-task --file tests/test_suite.json'
                sh 'python -m distributed_testing.client wait-completion'
            }
        }
        
        stage('Process Results') {
            steps {
                sh 'python -m distributed_testing.client export-results --format junit --output test-results.xml'
            }
        }
    }
    
    post {
        always {
            junit 'test-results.xml'
        }
    }
}
```

### GitLab CI Integration

For GitLab CI, the framework provides a `.gitlab-ci.yml` configuration:

```yaml
stages:
  - setup
  - test
  - process_results

setup:
  stage: setup
  script:
    - python -m distributed_testing.coordinator --db-path ./coordinator.duckdb --daemon
    - python -m distributed_testing.worker --coordinator http://localhost:8080 --worker-id worker-1 &

run_tests:
  stage: test
  script:
    - python -m distributed_testing.client submit-task --file tests/test_suite.json
    - python -m distributed_testing.client wait-completion

process_results:
  stage: process_results
  script:
    - python -m distributed_testing.client export-results --format gitlab --output test-results.json
  artifacts:
    reports:
      junit: test-results.xml
```

## External System Integrations

The framework can integrate with various external systems to provide additional functionality.

### JIRA Integration

The JIRA integration allows the framework to create and update issues based on test results:

```python
from distributed_testing.integrations import JiraIntegration

# Initialize the JIRA integration
jira = JiraIntegration(
    url="https://your-jira-instance.com",
    username="your-username",
    api_token="your-api-token",
    project_key="TEST"
)

# Create an issue for a failed test
issue = jira.create_issue(
    summary="Test failure: TestCase123",
    description="The test failed with error: Connection timeout",
    issue_type="Bug",
    labels=["automated-test", "critical"]
)

# Update an existing issue
jira.update_issue(
    issue.key,
    fields={
        "status": "In Progress",
        "assignee": {"name": "john.doe"}
    }
)

# Close an issue
jira.close_issue(issue.key, resolution="Fixed")
```

### Slack/MS Teams Integration

The framework can send notifications to Slack or MS Teams:

```python
from distributed_testing.integrations import SlackIntegration, TeamsIntegration

# Initialize the Slack integration
slack = SlackIntegration(webhook_url="https://hooks.slack.com/services/...")

# Send a message to Slack
slack.send_message(
    channel="#testing",
    text="Test suite completed with 95% success rate",
    attachments=[
        {
            "title": "Test Results",
            "text": "10 tests passed, 1 test failed",
            "color": "warning"
        }
    ]
)

# Initialize the MS Teams integration
teams = TeamsIntegration(webhook_url="https://outlook.office.com/webhook/...")

# Send a message to MS Teams
teams.send_message(
    title="Test Suite Completed",
    text="Test suite completed with 95% success rate",
    sections=[
        {
            "title": "Test Results",
            "text": "10 tests passed, 1 test failed"
        }
    ],
    color="warning"
)
```

### Prometheus Integration

The framework can export metrics to Prometheus:

```python
from distributed_testing.integrations import PrometheusIntegration

# Initialize the Prometheus integration
prometheus = PrometheusIntegration(
    port=9090,
    metrics_path="/metrics"
)

# Register metrics
task_counter = prometheus.register_counter(
    name="distributed_testing_tasks_total",
    description="Total number of tasks processed",
    labels=["status"]
)

task_duration = prometheus.register_histogram(
    name="distributed_testing_task_duration_seconds",
    description="Task execution time in seconds",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

# Update metrics
task_counter.labels(status="success").inc()
task_counter.labels(status="failure").inc()

with task_duration.time():
    # Execute a task and measure its duration
    execute_task()
```

## API Standardization

The framework provides a standardized API for interacting with the system.

### RESTful API

The RESTful API follows consistent patterns:

- `GET /api/v1/tasks`: List all tasks
- `POST /api/v1/tasks`: Submit a new task
- `GET /api/v1/tasks/{task_id}`: Get task details
- `PUT /api/v1/tasks/{task_id}`: Update a task
- `DELETE /api/v1/tasks/{task_id}`: Delete a task

Example request:

```bash
curl -X POST http://localhost:8080/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "type": "benchmark",
    "model": "bert-base-uncased",
    "hardware": "cuda",
    "batch_sizes": [1, 2, 4, 8, 16],
    "priority": 1
  }'
```

### GraphQL API

The GraphQL API provides a flexible way to query the system:

```graphql
query {
  tasks(status: "running") {
    id, type, status, priority, created_at, updated_at, worker {
      id, status, available_hardware
    }, result {
      status, start_time, end_time, metrics
    }
  }
}
```

Example using Python:

```python
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

# Select your transport with a defined URL
transport = AIOHTTPTransport(url="http://localhost:8080/graphql")

# Create a GraphQL client
client = Client(transport=transport, fetch_schema_from_transport=True)

# Execute the query
query = gql("""
query {
  tasks(status: "running") {
    id, type, status, priority, created_at, updated_at, worker {
      id, status, available_hardware
    }
  }
}
""")

result = client.execute(query)
```

## Custom Scheduler Support

The framework allows for custom scheduling algorithms through the scheduler plugin interface.

### Scheduler Interface

Custom schedulers implement the `SchedulerPlugin` interface:

```python
from distributed_testing.plugins import SchedulerPlugin

class FairShareScheduler(SchedulerPlugin):
    def initialize(self, coordinator):
        self.coordinator = coordinator
        self.user_shares = {}
        return True
        
    def configure(self, config):
        self.user_shares = config.get("user_shares", {})
        self.default_share = config.get("default_share", 1)
        return True
        
    def select_task(self, available_tasks, available_workers):
        # Implement fair share scheduling algorithm
        user_usage = self._calculate_user_usage()
        
        # Sort tasks by user share ratio
        tasks_by_priority = sorted(
            available_tasks,
            key=lambda task: user_usage.get(task.user, 0) / self.user_shares.get(task.user, self.default_share)
        )
        
        # Find the highest priority task that can be executed
        for task in tasks_by_priority:
            for worker in available_workers:
                if self._can_execute(task, worker):
                    return task, worker
        
        return None, None
        
    def _calculate_user_usage(self):
        # Calculate current resource usage by user
        pass
        
    def _can_execute(self, task, worker):
        # Check if the worker can execute the task
        pass
```

### Resource Representation

Tasks and workers are represented with standard resource descriptions:

```python
task_resources = {
    "cpu": 2,
    "memory": 4096,
    "gpu_memory": 8192,
    "hardware": ["cuda"]
}

worker_resources = {
    "cpu": 8,
    "memory": 16384,
    "gpu_memory": 16384,
    "hardware": ["cuda", "cpu"]
}
```

### Constraint Expression

The framework supports constraint expressions for scheduling decisions:

```python
constraints = [
    "task.hardware in worker.hardware",
    "task.cpu <= worker.available_cpu",
    "task.memory <= worker.available_memory",
    "task.gpu_memory <= worker.available_gpu_memory"
]
```

## Examples and Tutorials

### Creating a Simple Plugin

```python
from distributed_testing.plugins import NotificationPlugin

class EmailNotifier(NotificationPlugin):
    def initialize(self, coordinator):
        self.coordinator = coordinator
        return True
        
    def configure(self, config):
        self.smtp_server = config["smtp_server"]
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config["username"]
        self.password = config["password"]
        self.from_address = config["from_address"]
        self.recipients = config["recipients"]
        return True
        
    def notify(self, message, severity="info"):
        # Send an email notification
        import smtplib
        from email.mime.text import MIMEText
        
        msg = MIMEText(message)
        msg["Subject"] = f"[{severity.upper()}] Distributed Testing Framework"
        msg["From"] = self.from_address
        msg["To"] = ", ".join(self.recipients)
        
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            
        return True
```

### Using the GraphQL API

```python
import anyio
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

async def main():
    # Set up the transport
    transport = AIOHTTPTransport(url="http://localhost:8080/graphql")
    
    # Create the client
    client = Client(transport=transport, fetch_schema_from_transport=True)
    
    # Submit a new task
    mutation = gql("""
    mutation($input: TaskInput!) {
      createTask(input: $input) {
        id, type, status
      }
    }
    """)
    
    params = {
        "input": {
            "type": "benchmark",
            "model": "bert-base-uncased",
            "hardware": "cuda",
            "batch_sizes": [1, 2, 4, 8, 16]
        }
    }
    
    result = await client.execute_async(mutation, variable_values=params)
    task_id = result["createTask"]["id"]
    
    # Poll for task completion
    query = gql("""
    query($id: ID!) {
      task(id: $id) {
        id, status, result {
          status, metrics
        }
      }
    }
    """)
    
    while True:
        result = await client.execute_async(query, variable_values={"id": task_id})
        status = result["task"]["status"]
        
        if status in ["completed", "failed"]:
            print(f"Task {task_id} {status}")
            if status == "completed":
                print(f"Metrics: {result['task']['result']['metrics']}")
            break
            
        print(f"Task {task_id} status: {status}")
        await anyio.sleep(5)

# Run the example
anyio.run(main)
```

### Creating a CI/CD Integration

```python
#!/usr/bin/env python
# ci_runner.py

import argparse
import os
import sys
import time
import json
import requests

def main():
    parser = argparse.ArgumentParser(description="Run distributed tests in CI")
    parser.add_argument("--coordinator", default="http://localhost:8080", help="Coordinator URL")
    parser.add_argument("--test-file", required=True, help="Test file to run")
    parser.add_argument("--output-dir", default="./test-results", help="Output directory")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument("--format", choices=["json", "junit", "html"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    # Read the test file
    with open(args.test_file, "r") as f:
        tests = json.load(f)
        
    # Submit the tests
    task_ids = []
    for test in tests:
        response = requests.post(
            f"{args.coordinator}/api/v1/tasks",
            json=test
        )
        response.raise_for_status()
        task_ids.append(response.json()["id"])
        
    # Wait for completion
    start_time = time.time()
    pending_tasks = set(task_ids)
    
    while pending_tasks and (time.time() - start_time) < args.timeout:
        for task_id in list(pending_tasks):
            response = requests.get(f"{args.coordinator}/api/v1/tasks/{task_id}")
            response.raise_for_status()
            
            status = response.json()["status"]
            if status in ["completed", "failed"]:
                pending_tasks.remove(task_id)
                
        if pending_tasks:
            print(f"Waiting for {len(pending_tasks)} tasks to complete...")
            time.sleep(10)
            
    # Check if all tasks completed
    if pending_tasks:
        print(f"Timeout: {len(pending_tasks)} tasks did not complete in time")
        sys.exit(1)
        
    # Export the results
    os.makedirs(args.output_dir, exist_ok=True)
    
    response = requests.get(
        f"{args.coordinator}/api/v1/tasks/export",
        params={
            "ids": ",".join(task_ids),
            "format": args.format
        }
    )
    response.raise_for_status()
    
    output_file = os.path.join(args.output_dir, f"results.{args.format}")
    with open(output_file, "wb") as f:
        f.write(response.content)
        
    print(f"Results exported to {output_file}")
    
    # Check if all tests passed
    response = requests.get(
        f"{args.coordinator}/api/v1/tasks/summary",
        params={"ids": ",".join(task_ids)}
    )
    response.raise_for_status()
    
    summary = response.json()
    if summary["failed"] > 0:
        print(f"Tests failed: {summary['failed']} out of {summary['total']}")
        sys.exit(1)
        
    print(f"All tests passed: {summary['total']}")
    
if __name__ == "__main__":
    main()
```

For more examples and tutorials, see the [examples directory](../examples/).