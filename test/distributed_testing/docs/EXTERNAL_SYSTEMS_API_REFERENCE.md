# External Systems API Reference

This document provides a comprehensive reference for the External Systems API, which enables integration with various third-party systems through a standardized interface.

## Table of Contents

1. [Introduction](#introduction)
2. [Interface Architecture](#interface-architecture)
3. [Detailed API Reference](#detailed-api-reference)
   - [ExternalSystemInterface](#externalsysteminterface)
   - [ConnectorCapabilities](#connectorcapabilities)
   - [ExternalSystemResult](#externalsystemresult)
   - [ExternalSystemFactory](#externalsystemfactory)
4. [Connector Implementations](#connector-implementations)
   - [JIRA Connector](#jira-connector)
   - [Slack Connector](#slack-connector)
   - [TestRail Connector](#testrail-connector)
   - [Prometheus Connector](#prometheus-connector)
   - [Email Connector](#email-connector)
   - [MS Teams Connector](#ms-teams-connector)
5. [Common Patterns and Implementation Techniques](#common-patterns-and-implementation-techniques)
6. [Advanced Integration Guide](#advanced-integration-guide)
7. [Error Handling](#error-handling)
8. [Versioning and Compatibility](#versioning-and-compatibility)
9. [Security Considerations](#security-considerations)

## Introduction

The External Systems API provides a standardized way to integrate with various third-party systems, including issue trackers, notification systems, test management systems, and monitoring tools. This API follows a consistent pattern to ensure:

- **Uniform Interface**: All systems implement the same core interface
- **Easy Extension**: Adding support for new systems is straightforward
- **Runtime Configuration**: Systems can be configured at runtime
- **Consistent Error Handling**: Errors are handled in a consistent way
- **Type Safety**: Strong typing is used throughout
- **Comprehensive Documentation**: All functionality is documented

## Interface Architecture

The external systems integration architecture consists of four main components:

1. **ExternalSystemInterface**: The core interface that all connectors implement
2. **ConnectorCapabilities**: Represents the capabilities of a connector
3. **ExternalSystemResult**: Standardized operation result representation
4. **ExternalSystemFactory**: Factory for creating connector instances

### Architecture Diagram

```
┌─────────────────────────────┐
│                             │
│  External Systems API User  │
│                             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│                             │
│   ExternalSystemFactory     │
│                             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│                             │
│   ExternalSystemInterface   │◄────┐
│                             │     │
└─────────────────────────────┘     │
                                    │
┌─────────────────────────────┐     │
│                             │     │
│   Concrete Implementations  │─────┘
│    • JIRA                   │
│    • Slack                  │
│    • TestRail               │
│    • Prometheus             │
│    • Email                  │
│    • MS Teams               │
│                             │
└─────────────────────────────┘
```

## Detailed API Reference

### ExternalSystemInterface

`ExternalSystemInterface` is the abstract base class that all external system connectors must implement. It defines the contract for interacting with external systems.

#### Methods

| Method | Description | Parameters | Return Value |
|--------|-------------|------------|--------------|
| `initialize` | Initialize with configuration | `config: Dict[str, Any]` | `bool` |
| `connect` | Establish connection | None | `bool` |
| `is_connected` | Check connection status | None | `bool` |
| `execute_operation` | Execute operation | `operation: str, params: Dict[str, Any]` | `Dict[str, Any]` |
| `query` | Query for data | `query_params: Dict[str, Any]` | `List[Dict[str, Any]]` |
| `create_item` | Create item | `item_type: str, item_data: Dict[str, Any]` | `Dict[str, Any]` |
| `update_item` | Update item | `item_type: str, item_id: str, update_data: Dict[str, Any]` | `bool` |
| `delete_item` | Delete item | `item_type: str, item_id: str` | `bool` |
| `get_item` | Get item | `item_type: str, item_id: str` | `Dict[str, Any]` |
| `system_info` | Get system info | None | `Dict[str, Any]` |
| `close` | Close connection | None | `None` |

#### Method Details

##### `initialize(config: Dict[str, Any]) -> bool`

Initialize the connector with the provided configuration.

**Parameters:**
- `config`: Configuration dictionary containing provider-specific settings

**Returns:**
- `True` if initialization succeeded, `False` otherwise

**Example:**
```python
# Initialize JIRA connector
result = await jira_connector.initialize({
    "email": "user@example.com",
    "token": "api_token",
    "server_url": "https://jira.example.com",
    "project_key": "PROJECT"
})

if result:
    print("Initialization successful")
else:
    print("Initialization failed")
```

##### `connect() -> bool`

Establish a connection to the external system.

**Returns:**
- `True` if connection succeeded, `False` otherwise

**Example:**
```python
# Connect to external system
connected = await connector.connect()

if connected:
    print("Connected to external system")
else:
    print("Connection failed")
```

##### `is_connected() -> bool`

Check if the connector is currently connected to the external system.

**Returns:**
- `True` if connected, `False` otherwise

**Example:**
```python
# Check connection status
if await connector.is_connected():
    print("Connector is connected")
else:
    print("Connector is not connected")
```

##### `execute_operation(operation: str, params: Dict[str, Any]) -> Dict[str, Any]`

Execute an operation on the external system.

**Parameters:**
- `operation`: The operation to execute
- `params`: Parameters for the operation

**Returns:**
- Dictionary with operation result in ExternalSystemResult format

**Example:**
```python
# Execute custom operation
result = await connector.execute_operation("send_message", {
    "channel": "#general",
    "text": "Hello, world!"
})

if result["success"]:
    print(f"Operation successful: {result['result_data']}")
else:
    print(f"Operation failed: {result['error_message']}")
```

##### `query(query_params: Dict[str, Any]) -> List[Dict[str, Any]]`

Query the external system for data.

**Parameters:**
- `query_params`: Query parameters

**Returns:**
- List of query results

**Example:**
```python
# Query issues from JIRA
issues = await jira_connector.query({
    "project": "PROJECT",
    "status": ["Open", "In Progress"],
    "assignee": "current_user"
})

for issue in issues:
    print(f"Issue: {issue['key']} - {issue['summary']}")
```

##### `create_item(item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]`

Create an item in the external system.

**Parameters:**
- `item_type`: Type of item to create
- `item_data`: Item data

**Returns:**
- Dictionary with created item details

**Example:**
```python
# Create issue in JIRA
issue = await jira_connector.create_item("issue", {
    "summary": "Test Issue",
    "description": "This is a test issue",
    "issue_type": "Bug",
    "priority": "Medium"
})

print(f"Created issue: {issue['key']}")
```

##### `update_item(item_type: str, item_id: str, update_data: Dict[str, Any]) -> bool`

Update an item in the external system.

**Parameters:**
- `item_type`: Type of item to update
- `item_id`: ID of the item to update
- `update_data`: Data to update

**Returns:**
- `True` if update succeeded, `False` otherwise

**Example:**
```python
# Update issue in JIRA
success = await jira_connector.update_item("issue", "PROJECT-123", {
    "status": "In Progress",
    "assignee": "user@example.com"
})

if success:
    print("Issue updated successfully")
else:
    print("Failed to update issue")
```

##### `delete_item(item_type: str, item_id: str) -> bool`

Delete an item from the external system.

**Parameters:**
- `item_type`: Type of item to delete
- `item_id`: ID of the item to delete

**Returns:**
- `True` if deletion succeeded, `False` otherwise

**Example:**
```python
# Delete issue from JIRA
success = await jira_connector.delete_item("issue", "PROJECT-123")

if success:
    print("Issue deleted successfully")
else:
    print("Failed to delete issue")
```

##### `get_item(item_type: str, item_id: str) -> Dict[str, Any]`

Get an item from the external system.

**Parameters:**
- `item_type`: Type of item to get
- `item_id`: ID of the item to get

**Returns:**
- Dictionary with item details

**Example:**
```python
# Get issue from JIRA
issue = await jira_connector.get_item("issue", "PROJECT-123")

print(f"Issue Summary: {issue['summary']}")
print(f"Issue Status: {issue['status']}")
```

##### `system_info() -> Dict[str, Any]`

Get information about the external system.

**Returns:**
- Dictionary with system information

**Example:**
```python
# Get system info
info = await connector.system_info()

print(f"System Type: {info['system_type']}")
print(f"Connected: {info['connected']}")
print(f"API Version: {info.get('api_version', 'Unknown')}")
```

##### `close() -> None`

Close the connection to the external system and clean up resources.

**Example:**
```python
# Close connection
await connector.close()
print("Connection closed")
```

### ConnectorCapabilities

`ConnectorCapabilities` represents the capabilities of a connector, allowing runtime feature detection and graceful degradation.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `supports_create` | `bool` | Whether the connector supports creating items |
| `supports_update` | `bool` | Whether the connector supports updating items |
| `supports_delete` | `bool` | Whether the connector supports deleting items |
| `supports_query` | `bool` | Whether the connector supports querying |
| `supports_batch_operations` | `bool` | Whether the connector supports batch operations |
| `supports_attachments` | `bool` | Whether the connector supports attachments |
| `supports_comments` | `bool` | Whether the connector supports comments |
| `supports_custom_fields` | `bool` | Whether the connector supports custom fields |
| `supports_relationships` | `bool` | Whether the connector supports relationships |
| `supports_history` | `bool` | Whether the connector supports history tracking |
| `item_types` | `List[str]` | List of supported item types |
| `query_operators` | `List[str]` | List of supported query operators |
| `max_batch_size` | `int` | Maximum batch size for batch operations |
| `rate_limit` | `int` | Rate limit for API calls (requests per minute) |
| (Additional properties) | (Various) | Connector-specific capabilities |

#### Methods

| Method | Description | Parameters | Return Value |
|--------|-------------|------------|--------------|
| `to_dict` | Convert to dictionary | None | `Dict[str, Any]` |
| `from_dict` | Create from dictionary | `data: Dict[str, Any]` | `ConnectorCapabilities` |

#### Example

```python
# Create capabilities for a connector
capabilities = ConnectorCapabilities(
    supports_create=True,
    supports_update=True,
    supports_delete=False,
    supports_query=True,
    supports_batch_operations=True,
    supports_attachments=False,
    supports_comments=True,
    item_types=["issue", "comment"],
    max_batch_size=100,
    rate_limit=60,
    # Connector-specific capabilities
    supports_markdown=True
)

# Check capabilities at runtime
if capabilities.supports_attachments:
    # Implement attachment functionality
    pass
else:
    # Inform user that attachments are not supported
    pass

# Get dictionary representation
capabilities_dict = capabilities.to_dict()

# Create from dictionary
restored_capabilities = ConnectorCapabilities.from_dict(capabilities_dict)
```

### ExternalSystemResult

`ExternalSystemResult` provides a standardized representation of operation results from external systems.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `success` | `bool` | Whether the operation succeeded |
| `operation` | `str` | The operation that was performed |
| `result_data` | `Optional[Any]` | The result data from the operation |
| `error_message` | `Optional[str]` | Error message if the operation failed |
| `error_code` | `Optional[str]` | Error code if the operation failed |
| `metadata` | `Dict[str, Any]` | Additional metadata |

#### Methods

| Method | Description | Parameters | Return Value |
|--------|-------------|------------|--------------|
| `to_dict` | Convert to dictionary | None | `Dict[str, Any]` |
| `from_dict` | Create from dictionary | `data: Dict[str, Any]` | `ExternalSystemResult` |

#### Example

```python
# Create successful result
success_result = ExternalSystemResult(
    success=True,
    operation="create_issue",
    result_data={"key": "PROJECT-123", "id": "10001"}
)

# Create error result
error_result = ExternalSystemResult(
    success=False,
    operation="create_issue",
    error_message="Field 'summary' is required",
    error_code="VALIDATION_ERROR"
)

# Convert to dictionary
result_dict = success_result.to_dict()

# Create from dictionary
restored_result = ExternalSystemResult.from_dict(result_dict)
```

### ExternalSystemFactory

`ExternalSystemFactory` is a factory for creating external system connector instances.

#### Methods

| Method | Description | Parameters | Return Value |
|--------|-------------|------------|--------------|
| `register_connector` | Register a connector class | `system_type: str, connector_class: type` | `None` |
| `create_connector` | Create a connector instance | `system_type: str, config: Dict[str, Any]` | `ExternalSystemInterface` |
| `get_available_connectors` | Get available connector types | None | `List[str]` |
| `get_connector_class` | Get connector class | `system_type: str` | `Optional[type]` |

#### Example

```python
from distributed_testing.external_systems import ExternalSystemFactory

# Register a connector
ExternalSystemFactory.register_connector("custom", CustomConnector)

# Create a connector instance
connector = await ExternalSystemFactory.create_connector(
    "jira", 
    {
        "email": "user@example.com",
        "token": "api_token",
        "server_url": "https://jira.example.com",
        "project_key": "PROJECT"
    }
)

# Get available connector types
available_connectors = ExternalSystemFactory.get_available_connectors()
print(f"Available connectors: {available_connectors}")

# Get connector class
connector_class = ExternalSystemFactory.get_connector_class("jira")
```

## Connector Implementations

### JIRA Connector

The JIRA connector integrates with JIRA's REST API to provide issue tracking functionality.

#### Configuration

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `email` | JIRA account email | Yes | - |
| `token` | JIRA API token | Yes | - |
| `server_url` | JIRA server URL | Yes | - |
| `project_key` | JIRA project key | Yes | - |
| `verify_ssl` | Verify SSL certificates | No | `True` |
| `timeout` | Request timeout in seconds | No | `30` |
| `max_retries` | Maximum number of retries | No | `3` |
| `backoff_factor` | Backoff factor for retries | No | `0.3` |

#### Supported Item Types

- `issue`: JIRA issues
- `comment`: Issue comments
- `attachment`: Issue attachments
- `worklog`: Issue worklogs

#### Supported Operations

- `search_issues`: Search for issues using JQL
- `add_comment`: Add a comment to an issue
- `assign_issue`: Assign an issue to a user
- `transition_issue`: Transition an issue to a new status
- `upload_attachment`: Upload an attachment to an issue
- `add_worklog`: Add a worklog to an issue

#### Example

```python
from distributed_testing.external_systems import ExternalSystemFactory

# Create JIRA connector
jira = await ExternalSystemFactory.create_connector(
    "jira", 
    {
        "email": "user@example.com",
        "token": "api_token",
        "server_url": "https://jira.example.com",
        "project_key": "PROJECT"
    }
)

# Create an issue
issue = await jira.create_item("issue", {
    "summary": "Test Issue",
    "description": "This is a test issue",
    "issue_type": "Bug",
    "priority": "Medium",
    "labels": ["automated", "test"]
})

# Add a comment
await jira.execute_operation("add_comment", {
    "issue_key": issue["key"],
    "body": "This is a test comment"
})

# Update the issue
await jira.update_item("issue", issue["key"], {
    "assignee": "john.doe@example.com",
    "priority": "High"
})

# Get the issue
updated_issue = await jira.get_item("issue", issue["key"])

# Close the connection
await jira.close()
```

### Slack Connector

The Slack connector integrates with Slack's Web API to provide real-time notifications.

#### Configuration

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `token` | Slack API token | Yes | - |
| `default_channel` | Default channel for messages | No | `#general` |
| `username` | Username for bot messages | No | `Distributed Testing Bot` |
| `icon_emoji` | Emoji for bot messages | No | `:robot_face:` |
| `timeout` | Request timeout in seconds | No | `30` |
| `max_retries` | Maximum number of retries | No | `3` |

#### Supported Item Types

- `message`: Slack messages
- `file`: File uploads

#### Supported Operations

- `send_message`: Send a message to a channel
- `update_message`: Update an existing message
- `send_ephemeral`: Send an ephemeral message to a user
- `add_reaction`: Add a reaction to a message
- `create_channel`: Create a new channel
- `invite_to_channel`: Invite users to a channel

#### Example

```python
from distributed_testing.external_systems import ExternalSystemFactory

# Create Slack connector
slack = await ExternalSystemFactory.create_connector(
    "slack", 
    {
        "token": "xoxb-your-token",
        "default_channel": "#notifications"
    }
)

# Send a message
message = await slack.create_item("message", {
    "channel": "#notifications",
    "text": "Hello from the Distributed Testing Framework!",
    "blocks": [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Notification*\nTest run completed successfully"
            }
        }
    ]
})

# Upload a file
file = await slack.create_item("file", {
    "channel": "#notifications",
    "file_path": "/path/to/test_results.json",
    "title": "Test Results"
})

# Close the connection
await slack.close()
```

### TestRail Connector

The TestRail connector integrates with TestRail's API to manage test cases, test runs, and results.

#### Configuration

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `url` | TestRail URL | Yes | - |
| `username` | TestRail username | Yes | - |
| `api_key` | TestRail API key | Yes | - |
| `project_id` | TestRail project ID | Yes | - |
| `verify_ssl` | Verify SSL certificates | No | `True` |
| `timeout` | Request timeout in seconds | No | `30` |

#### Supported Item Types

- `test_case`: Test cases
- `test_run`: Test runs
- `test_result`: Test results
- `test_plan`: Test plans
- `milestone`: Milestones
- `section`: Test case sections

#### Supported Operations

- `add_result_for_case`: Add a result for a test case
- `add_run`: Add a test run
- `add_plan`: Add a test plan
- `add_plan_entry`: Add an entry to a plan
- `add_milestone`: Add a milestone
- `add_section`: Add a section

#### Example

```python
from distributed_testing.external_systems import ExternalSystemFactory

# Create TestRail connector
testrail = await ExternalSystemFactory.create_connector(
    "testrail", 
    {
        "url": "https://example.testrail.io",
        "username": "user@example.com",
        "api_key": "api_key",
        "project_id": "1"
    }
)

# Create a test run
test_run = await testrail.create_item("test_run", {
    "name": "Automated Test Run",
    "description": "Run created by Distributed Testing Framework",
    "suite_id": 1,
    "include_all": True
})

# Add a test result
await testrail.create_item("test_result", {
    "test_id": 100,
    "status_id": 1,  # Passed
    "comment": "Test passed successfully",
    "elapsed": "1m 30s"
})

# Get test run details
run_details = await testrail.get_item("test_run", test_run["id"])

# Close the connection
await testrail.close()
```

### Prometheus Connector

The Prometheus connector integrates with Prometheus API and Push Gateway to query metrics and submit metrics.

#### Configuration

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `prometheus_url` | Prometheus server URL | No | `http://localhost:9090` |
| `pushgateway_url` | Push gateway URL | No | `http://localhost:9091` |
| `job_prefix` | Job name prefix | No | `distributed_testing` |
| `timeout` | Request timeout in seconds | No | `30` |
| `verify_ssl` | Verify SSL certificates | No | `True` |

#### Supported Item Types

- `metric`: Metrics
- `alerting_rule`: Alerting rules
- `recording_rule`: Recording rules

#### Supported Operations

- `query`: Query metrics
- `query_range`: Query metrics over a time range
- `alerts`: Get current alerts
- `targets`: Get current targets
- `push_metric`: Push a metric to Push Gateway
- `delete_metric`: Delete a metric from Push Gateway

#### Example

```python
from distributed_testing.external_systems import ExternalSystemFactory

# Create Prometheus connector
prometheus = await ExternalSystemFactory.create_connector(
    "prometheus", 
    {
        "prometheus_url": "http://prometheus.example.com:9090",
        "pushgateway_url": "http://pushgateway.example.com:9091",
        "job_prefix": "distributed_testing"
    }
)

# Query metrics
results = await prometheus.execute_operation("query", {
    "query": 'up{job="distributed_testing"}'
})

# Push metrics
await prometheus.create_item("metric", {
    "job_name": "test_job",
    "instance": "test_instance",
    "name": "test_execution_duration_seconds",
    "value": 10.5,
    "help": "Duration of test execution in seconds",
    "type": "gauge",
    "labels": {"suite": "performance", "module": "api"}
})

# Close the connection
await prometheus.close()
```

### Email Connector

The Email connector provides email notification capabilities using SMTP.

#### Configuration

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `smtp_server` | SMTP server address | Yes | - |
| `smtp_port` | SMTP server port | Yes | - |
| `username` | SMTP username | No | - |
| `password` | SMTP password | No | - |
| `use_tls` | Use TLS for SMTP | No | `True` |
| `use_ssl` | Use SSL for SMTP | No | `False` |
| `default_sender` | Default sender email address | No | - |
| `default_recipients` | Default recipient email addresses | No | `[]` |
| `templates_dir` | Directory containing email templates | No | - |
| `rate_limit` | Maximum emails per minute | No | `60` |

#### Supported Item Types

- `email`: Email messages

#### Supported Operations

- `send_message`: Send an email message
- `send_template_email`: Send an email using a template
- `send_batch_emails`: Send multiple emails in a batch
- `validate_email`: Validate an email address

#### Example

```python
from distributed_testing.external_systems import ExternalSystemFactory

# Create Email connector
email = await ExternalSystemFactory.create_connector(
    "email", 
    {
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "username": "notifications@example.com",
        "password": "password",
        "use_tls": True,
        "default_sender": "notifications@example.com",
        "default_recipients": ["team@example.com"]
    }
)

# Send an email
await email.create_item("email", {
    "recipients": ["user@example.com"],
    "subject": "Test Execution Completed",
    "body": "Test execution has completed successfully.",
    "html_body": "<h1>Test Execution Results</h1><p>All tests passed!</p>",
    "attachments": [
        {"path": "/path/to/test_report.pdf"}
    ]
})

# Send a template-based email
await email.execute_operation("send_template_email", {
    "template_name": "test_report",
    "template_data": {
        "test_name": "API Test Suite",
        "result": "PASSED",
        "duration": "10m 15s",
        "failures": 0,
        "success_rate": "100%"
    },
    "subject": "Test Report: API Test Suite"
})

# Close the connection
await email.close()
```

### MS Teams Connector

The MS Teams connector integrates with Microsoft Teams using webhooks or the Microsoft Graph API.

#### Configuration

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `webhook_url` | Teams webhook URL (for webhook integration) | Yes (for webhook) | - |
| `use_graph_api` | Whether to use Microsoft Graph API | No | `False` |
| `graph_api_token` | Microsoft Graph API token | No | - |
| `tenant_id` | Microsoft Teams tenant ID | Yes (for Graph API) | - |
| `client_id` | Microsoft app client ID | Yes (for Graph API) | - |
| `client_secret` | Microsoft app client secret | Yes (for Graph API) | - |
| `default_team_id` | Default team ID | Yes (for Graph API) | - |
| `default_channel` | Default channel name | No | `General` |
| `templates_dir` | Directory containing message templates | No | - |
| `rate_limit` | Maximum messages per minute | No | `30` |

#### Supported Item Types

- `message`: Teams messages
- `adaptive_card`: Adaptive card messages

#### Supported Operations

- `send_message`: Send a Teams message
- `send_adaptive_card`: Send an adaptive card
- `send_batch_messages`: Send multiple messages in a batch
- `send_template_message`: Send a message using a template
- `get_channels`: Get channels in a team (Graph API only)
- `get_team_members`: Get members of a team (Graph API only)

#### Example

```python
from distributed_testing.external_systems import ExternalSystemFactory

# Create MS Teams connector with webhook integration
teams = await ExternalSystemFactory.create_connector(
    "msteams", 
    {
        "webhook_url": "https://outlook.office.com/webhook/YOUR_WEBHOOK_URL"
    }
)

# Create MS Teams connector with Microsoft Graph API integration
teams_graph = await ExternalSystemFactory.create_connector(
    "msteams", 
    {
        "use_graph_api": True,
        "tenant_id": "your_tenant_id",
        "client_id": "your_client_id",
        "client_secret": "your_client_secret",
        "default_team_id": "your_team_id",
        "default_channel": "General"
    }
)

# Send a simple message
await teams.create_item("message", {
    "title": "Test Execution Completed",
    "text": "Test execution has completed successfully.",
    "theme_color": "0078D7",  # Blue
    "facts": [
        ("Test Suite", "API Tests"),
        ("Result", "PASSED"),
        ("Duration", "10m 15s"),
        ("Failures", "0"),
        ("Success Rate", "100%")
    ]
})

# Send an adaptive card
await teams.create_item("adaptive_card", {
    "card_content": {
        "type": "AdaptiveCard",
        "version": "1.2",
        "body": [
            {
                "type": "TextBlock",
                "text": "Test Execution Results",
                "weight": "bolder",
                "size": "large"
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Test Suite", "value": "API Tests"},
                    {"title": "Result", "value": "PASSED"},
                    {"title": "Duration", "value": "10m 15s"},
                    {"title": "Failures", "value": "0"}
                ]
            }
        ],
        "actions": [
            {
                "type": "Action.OpenUrl",
                "title": "View Details",
                "url": "https://example.com/test-results"
            }
        ]
    }
})

# With Graph API, you can get channels in a team
if teams_graph.use_graph_api:
    channels = await teams_graph.execute_operation("get_channels", {
        "team_id": "your_team_id"
    })
    
    # And get team members
    team_members = await teams_graph.execute_operation("get_team_members", {
        "team_id": "your_team_id"
    })

# Close the connection
await teams.close()
```

## Common Patterns and Implementation Techniques

### Using Environment Variables for Configuration

For security and flexibility, connectors often support configuration from environment variables:

```python
# Configuration with environment variables as fallbacks
config = {
    "smtp_server": os.environ.get("SMTP_SERVER", "smtp.example.com"),
    "smtp_port": int(os.environ.get("SMTP_PORT", "587")),
    "username": os.environ.get("SMTP_USERNAME", ""),
    "password": os.environ.get("SMTP_PASSWORD", "")
}
```

### Rate Limiting Implementation

Most connectors implement rate limiting to avoid overloading external services:

```python
# Simple rate limiting implementation
self.last_request_time = time.time()
self.request_count = 0
self.rate_limit = 60  # 60 requests per minute

async def _handle_rate_limit(self):
    """Handle rate limiting to avoid exceeding API limits."""
    # Implement a simple token bucket algorithm
    self.request_count += 1
    current_time = time.time()
    time_passed = current_time - self.last_request_time
    
    if self.request_count >= self.rate_limit:
        # Reset counter and wait if needed
        wait_time = 60.0 - time_passed
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self.request_count = 0
        self.last_request_time = time.time()
```

### Async/Await Pattern

All connectors use the async/await pattern for asynchronous operations:

```python
async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an operation."""
    try:
        # Ensure connection is established
        await self._ensure_connection()
        
        # Handle rate limiting
        await self._handle_rate_limit()
        
        # Execute operation
        if operation == "send_message":
            result = await self._send_message(params)
        elif operation == "search_issues":
            result = await self._search_issues(params)
        else:
            result = ExternalSystemResult(
                success=False,
                operation=operation,
                error_message=f"Unsupported operation: {operation}",
                error_code="UNSUPPORTED_OPERATION"
            )
            
        return result.to_dict()
        
    except Exception as e:
        # Handle and report errors
        return ExternalSystemResult(
            success=False,
            operation=operation,
            error_message=str(e),
            error_code="EXCEPTION"
        ).to_dict()
```

### Connection Management

Properly managing connections to external systems is important:

```python
async def _ensure_connection(self):
    """Ensure connection is established."""
    if not self.connected:
        await self.connect()

async def close(self):
    """Close the connection and clean up resources."""
    if self.session:
        try:
            await self.session.close()
        except Exception as e:
            logger.warning(f"Error closing session: {str(e)}")
        finally:
            self.session = None
            self.connected = False
```

### Error Handling

Consistent error handling is a key aspect of the connectors:

```python
try:
    # Operation that might fail
    response = await self.session.post(endpoint, json=payload)
    
    if response.status == 200:
        data = await response.json()
        return ExternalSystemResult(
            success=True,
            operation="create_item",
            result_data=data
        )
    else:
        error_text = await response.text()
        return ExternalSystemResult(
            success=False,
            operation="create_item",
            error_message=f"API Error: {error_text}",
            error_code=f"API_ERROR_{response.status}"
        )
except Exception as e:
    return ExternalSystemResult(
        success=False,
        operation="create_item",
        error_message=f"Exception: {str(e)}",
        error_code="EXCEPTION"
    )
```

## Advanced Integration Guide

### Creating a Custom Connector

To create a custom connector, follow these steps:

1. **Create a Class**: Create a class that extends `ExternalSystemInterface`
2. **Implement Required Methods**: Implement all required methods
3. **Define Capabilities**: Define the connector's capabilities
4. **Register with Factory**: Register the connector with `ExternalSystemFactory`

Example:

```python
from distributed_testing.external_systems.api_interface import (
    ExternalSystemInterface,
    ConnectorCapabilities,
    ExternalSystemResult,
    ExternalSystemFactory
)

class CustomConnector(ExternalSystemInterface):
    """Custom connector implementation."""
    
    def __init__(self):
        """Initialize the connector."""
        self.connected = False
        self.config = {}
        self.session = None
        
        # Define capabilities
        self.capabilities = ConnectorCapabilities(
            supports_create=True,
            supports_update=True,
            supports_delete=True,
            supports_query=True,
            item_types=["custom_item"],
            rate_limit=100
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with configuration."""
        self.config = config
        # Basic validation
        if "api_key" not in config:
            return False
        return True
    
    # Implement other required methods...
    
    async def system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "system_type": "custom",
            "connected": self.connected,
            "api_version": "1.0",
            "capabilities": self.capabilities.to_dict()
        }
    
    async def close(self) -> None:
        """Close the connection."""
        if self.session:
            await self.session.close()
            self.session = None
            self.connected = False

# Register with factory
ExternalSystemFactory.register_connector("custom", CustomConnector)
```

### Advanced Configuration Handling

For more complex configuration needs:

```python
async def initialize(self, config: Dict[str, Any]) -> bool:
    """Initialize with configuration."""
    # Direct configuration
    self.api_key = config.get("api_key")
    
    # Environment variable fallback
    self.api_url = config.get("api_url", os.environ.get("API_URL"))
    
    # Type conversion
    self.timeout = int(config.get("timeout", "30"))
    
    # Boolean conversion
    self.verify_ssl = config.get("verify_ssl", "true").lower() in ("true", "yes", "1")
    
    # List conversion
    self.default_labels = config.get("default_labels", "").split(",") if config.get("default_labels") else []
    
    # Validation
    if not self.api_key:
        logger.error("API key is required")
        return False
    
    if not self.api_url:
        logger.error("API URL is required")
        return False
    
    return True
```

### Complex Operation Handling

For complex operations:

```python
async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an operation."""
    await self._ensure_connection()
    
    # Map operation to handler method
    handlers = {
        "create_issue": self._create_issue,
        "update_issue": self._update_issue,
        "search_issues": self._search_issues,
        "add_comment": self._add_comment,
        "upload_attachment": self._upload_attachment
    }
    
    # Get handler or return error
    handler = handlers.get(operation)
    if not handler:
        return ExternalSystemResult(
            success=False,
            operation=operation,
            error_message=f"Unsupported operation: {operation}",
            error_code="UNSUPPORTED_OPERATION"
        ).to_dict()
    
    # Execute handler
    try:
        result = await handler(params)
        return result.to_dict()
    except Exception as e:
        return ExternalSystemResult(
            success=False,
            operation=operation,
            error_message=str(e),
            error_code="EXCEPTION"
        ).to_dict()
```

## Error Handling

### Error Categories

The framework defines several error categories:

| Error Code | Description | Example |
|------------|-------------|---------|
| `CONFIGURATION_ERROR` | Error in connector configuration | Missing API key |
| `CONNECTION_ERROR` | Error connecting to the external system | Network error |
| `AUTHENTICATION_ERROR` | Authentication failure | Invalid credentials |
| `PERMISSION_ERROR` | Permission denied | Insufficient permissions |
| `VALIDATION_ERROR` | Input validation failure | Missing required field |
| `RATE_LIMIT_ERROR` | Rate limit exceeded | Too many requests |
| `SERVER_ERROR` | Server-side error | Internal server error |
| `CLIENT_ERROR` | Client-side error | Invalid request |
| `NOT_FOUND_ERROR` | Resource not found | Item not found |
| `TIMEOUT_ERROR` | Operation timed out | Request timeout |
| `UNSUPPORTED_OPERATION` | Operation not supported | Unsupported method |
| `EXCEPTION` | Unexpected exception | Code exception |

### Best Practices for Error Handling

1. **Be Specific**: Use specific error codes and descriptive messages
2. **Include Context**: Include context information in error messages
3. **Centralize Handling**: Use consistent error handling across all methods
4. **Log Errors**: Log errors for debugging
5. **Retry When Appropriate**: Implement retries for transient errors

### Example Error Handling

```python
try:
    response = await self.session.post(endpoint, json=payload, timeout=self.timeout)
    
    if response.status == 200:
        # Success case
        data = await response.json()
        return ExternalSystemResult(
            success=True,
            operation="create_item",
            result_data=data
        )
    elif response.status == 401:
        # Authentication error
        return ExternalSystemResult(
            success=False,
            operation="create_item",
            error_message="Authentication failed. Check credentials.",
            error_code="AUTHENTICATION_ERROR"
        )
    elif response.status == 403:
        # Permission error
        return ExternalSystemResult(
            success=False,
            operation="create_item",
            error_message="Permission denied. Check account permissions.",
            error_code="PERMISSION_ERROR"
        )
    elif response.status == 404:
        # Not found error
        return ExternalSystemResult(
            success=False,
            operation="create_item",
            error_message="Resource not found.",
            error_code="NOT_FOUND_ERROR"
        )
    elif response.status == 422:
        # Validation error
        error_text = await response.text()
        return ExternalSystemResult(
            success=False,
            operation="create_item",
            error_message=f"Validation error: {error_text}",
            error_code="VALIDATION_ERROR"
        )
    elif response.status == 429:
        # Rate limit error
        return ExternalSystemResult(
            success=False,
            operation="create_item",
            error_message="Rate limit exceeded. Try again later.",
            error_code="RATE_LIMIT_ERROR"
        )
    elif response.status >= 500:
        # Server error
        return ExternalSystemResult(
            success=False,
            operation="create_item",
            error_message=f"Server error: {response.status}",
            error_code="SERVER_ERROR"
        )
    else:
        # Other client error
        error_text = await response.text()
        return ExternalSystemResult(
            success=False,
            operation="create_item",
            error_message=f"API Error {response.status}: {error_text}",
            error_code=f"CLIENT_ERROR"
        )
except asyncio.TimeoutError:
    # Timeout error
    return ExternalSystemResult(
        success=False,
        operation="create_item",
        error_message=f"Request timed out after {self.timeout} seconds",
        error_code="TIMEOUT_ERROR"
    )
except Exception as e:
    # Unexpected exception
    logger.exception("Unexpected error in create_item")
    return ExternalSystemResult(
        success=False,
        operation="create_item",
        error_message=f"Exception: {str(e)}",
        error_code="EXCEPTION"
    )
```

## Versioning and Compatibility

### API Versioning

The External Systems API follows semantic versioning:

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible additions
- **PATCH**: Backward-compatible fixes

### Backward Compatibility

To maintain backward compatibility:

1. **Add, Don't Change**: Add new methods rather than changing existing ones
2. **Default Arguments**: Use default arguments for new parameters
3. **Optional Features**: Make new features optional
4. **Deprecation Process**: Mark deprecated features before removing them

### Compatibility Testing

To ensure compatibility:

1. **Unit Tests**: Test each connector individually
2. **Integration Tests**: Test with actual external systems
3. **Version Tests**: Test against different API versions
4. **Mock Tests**: Test against mock servers for reliable testing

## Security Considerations

### Credential Handling

Best practices for credential handling:

1. **Environment Variables**: Store credentials in environment variables
2. **Secret Management**: Use a secret management system in production
3. **Minimum Permissions**: Use credentials with minimum required permissions
4. **Token Rotation**: Regularly rotate API tokens
5. **No Hardcoding**: Never hardcode credentials in code

### Secure Communication

Ensure secure communication:

1. **TLS/SSL**: Use TLS/SSL for all communication
2. **Certificate Verification**: Verify SSL certificates
3. **Input Validation**: Validate all input before sending to external systems
4. **Output Escaping**: Escape output to prevent injection attacks

### Rate Limiting

Proper rate limiting:

1. **Respect Limits**: Respect API rate limits
2. **Exponential Backoff**: Use exponential backoff for retries
3. **Jitter**: Add jitter to backoff to prevent thundering herd
4. **Circuit Breaker**: Use circuit breaker pattern for failing services