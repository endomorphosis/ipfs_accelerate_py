# Plugin Architecture for Distributed Testing Framework

This document describes the Plugin Architecture implemented as part of Phase 8 (Integration and Extensibility) for the distributed testing framework.

## Overview

The Plugin Architecture enables extending the distributed testing framework's functionality without modifying its core code. It provides a flexible, standardized way to add new capabilities, integrate with external systems, and customize the framework's behavior.

## Key Features

1. **Modular Design**: Add, remove, or update plugins without modifying core code
2. **Lifecycle Management**: Plugins are automatically discovered, loaded, configured, and managed
3. **Event-Based Hooks**: Plugins can respond to framework events through a comprehensive hook system
4. **Configuration Options**: Plugins can be configured through command-line arguments or configuration files
5. **Type System**: Structured categorization of plugins by functionality
6. **Runtime Management**: Enable, disable, or reconfigure plugins at runtime

## Plugin Types

The architecture supports the following plugin types:

| Type | Description | Examples |
|------|-------------|----------|
| SCHEDULER | Custom task scheduling algorithms | Priority scheduler, fairness scheduler |
| TASK_EXECUTOR | Custom task execution handlers | GPU-specific executor, specialized model runners |
| REPORTER | Result reporting and visualization | CSV exporter, dashboard reporter |
| NOTIFICATION | Notification systems | Email notifier, Slack integration |
| MONITORING | System monitoring tools | Resource monitor, performance tracker |
| INTEGRATION | External system integrations | CI/CD integration, issue tracker integration |
| SECURITY | Security extensions | Access control, encryption |
| CUSTOM | Custom plugin types | Any specialized functionality |

## Hook System

Plugins register for specific hooks to respond to framework events:

### Coordinator Hooks

- `COORDINATOR_STARTUP`: Called when the coordinator starts
- `COORDINATOR_SHUTDOWN`: Called when the coordinator shuts down

### Task Hooks

- `TASK_CREATED`: Called when a task is created
- `TASK_ASSIGNED`: Called when a task is assigned to a worker
- `TASK_STARTED`: Called when a task execution begins
- `TASK_COMPLETED`: Called when a task completes successfully
- `TASK_FAILED`: Called when a task fails
- `TASK_CANCELLED`: Called when a task is cancelled

### Worker Hooks

- `WORKER_REGISTERED`: Called when a worker registers with the coordinator
- `WORKER_DISCONNECTED`: Called when a worker disconnects
- `WORKER_FAILED`: Called when a worker fails
- `WORKER_RECOVERED`: Called when a worker recovers from failure

### Recovery Hooks

- `RECOVERY_STARTED`: Called when a recovery process begins
- `RECOVERY_COMPLETED`: Called when a recovery process completes
- `RECOVERY_FAILED`: Called when a recovery process fails

### Custom Hooks

- `CUSTOM`: Used for custom, plugin-specific events

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                      Distributed Testing Framework                     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                          Plugin Manager                                │
│                                                                       │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│   │  Plugin Loading │    │  Hook System    │    │  Configuration  │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘   │
│                                                                       │
└─────────────┬─────────────┬─────────────┬─────────────┬───────────────┘
              │             │             │             │
              ▼             ▼             ▼             ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Reporter       │ │  Integration    │ │  Notification   │ │  Custom         │
│  Plugins        │ │  Plugins        │ │  Plugins        │ │  Plugins        │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Creating a Plugin

Creating a plugin involves extending the base `Plugin` class and implementing its methods:

```python
from plugin_architecture import Plugin, PluginType, HookType

class MyCustomPlugin(Plugin):
    """Custom plugin example."""
    
    def __init__(self):
        """Initialize the plugin."""
        super().__init__(
            name="MyCustomPlugin",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM
        )
        
        # Register hooks
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        self.register_hook(HookType.WORKER_FAILED, self.on_worker_failed)
    
    async def initialize(self, coordinator) -> bool:
        """Initialize the plugin with the coordinator reference."""
        self.coordinator = coordinator
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        return True
    
    async def on_task_completed(self, task_id: str, result: Any):
        """Handle task completed event."""
        print(f"Task {task_id} completed with result: {result}")
    
    async def on_worker_failed(self, worker_id: str, error: str):
        """Handle worker failed event."""
        print(f"Worker {worker_id} failed: {error}")
```

## Sample Plugins

The framework includes the following sample plugins:

### 1. Task Reporter Plugin

This plugin hooks into task lifecycle events and reports task status to an external system:

- Tracks task creation, assignment, execution, completion, and failure
- Records performance metrics for all tasks
- Provides task summary statistics
- Can report events to external systems in real-time

### 2. CI Integration Plugin

This plugin integrates with CI/CD systems to report test results:

- Supports GitHub Actions, Jenkins, GitLab CI, Azure DevOps, CircleCI, and more
- Creates test runs in the CI system
- Updates test status in real-time
- Provides final test reports with pass/fail status
- Automatically detects the CI environment

### 3. Notification Plugin

This plugin provides real-time notifications about framework events through various channels:

- Supports multiple notification channels:
  - Slack: Messages sent to Slack channels via API
  - Discord: Messages and embeds sent via webhook or Bot API
  - Telegram: Messages sent via Telegram Bot API
  - Email: Messages sent via SMTP
  - MS Teams: Messages sent via webhook or Graph API
  - JIRA: Issues created for error events
- Features include:
  - Event filtering: Configure which events trigger notifications
  - Notification grouping: Group similar notifications to reduce noise
  - Throttling: Control notification frequency
  - Customizable formatting: Format notifications for each channel
  - Metadata support: Include additional data in notifications
  - Standardized API: Consistent interface across channels

For detailed documentation, see [Notification System Guide](docs/NOTIFICATION_SYSTEM_GUIDE.md).

## Usage

### Command Line Options

The coordinator server supports the following plugin-related options:

```bash
# Enable/disable plugins
python coordinator.py --disable-plugins

# Specify plugin directories
python coordinator.py --plugin-dirs plugins,custom_plugins

# List available plugins
python coordinator.py --list-plugins

# Enable/disable specific plugins
python coordinator.py --enable-plugin my_plugin --disable-plugin other_plugin
```

### Testing the Plugin Architecture

A test script is provided to demonstrate the plugin architecture functionality:

```bash
# Run with default plugins
python run_test_plugins.py

# Test specific plugins
python run_test_plugins.py --plugins sample_reporter_plugin.py,ci_integration_plugin.py,notification_plugin.py

# Enable external reporting in the reporter plugin
python run_test_plugins.py --enable-external-reporting

# Specify CI system for the CI integration plugin
python run_test_plugins.py --ci-system jenkins

# Simulate task lifecycle events
python run_test_plugins.py --simulate-tasks 10

# Run notification system example with Discord notifications
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your-webhook-url"
python examples/notification_system_example.py --discord

# Run notification system example with multiple notification channels
python examples/notification_system_example.py --discord --telegram --slack

# Run custom scheduler example
python examples/custom_scheduler_example.py --scheduler fairness --fairness-window-hours 24
```

## Plugin Directory Structure

Plugins are organized in the `plugins` directory:

```
plugins/
├── sample_reporter_plugin.py        # Task reporter plugin
├── ci_integration_plugin.py         # CI integration plugin
├── notification_plugin.py           # Main notification plugin
├── scheduler/                       # Scheduler plugins
│   ├── base_scheduler_plugin.py     # Base scheduler implementation
│   ├── fairness_scheduler_plugin.py # Fair resource allocation scheduler
│   ├── scheduler_plugin_interface.py # Scheduler plugin interface
│   ├── scheduler_plugin_registry.py # Plugin registry for schedulers
│   └── scheduler_coordinator.py     # Scheduler coordinator
├── external_systems/                # External system connectors
│   ├── api_interface.py             # Standardized API interface
│   ├── slack_connector.py           # Slack integration
│   ├── jira_connector.py            # JIRA integration
│   ├── discord_connector.py         # Discord integration
│   ├── telegram_connector.py        # Telegram integration
│   └── ms_teams_connector.py        # MS Teams integration
└── custom_plugins/                  # Custom plugins
    └── my_custom_plugin.py

examples/
├── plugin_example.py                # Basic plugin usage example
├── notification_system_example.py   # Notification system example
└── custom_scheduler_example.py      # Custom scheduler example

docs/
├── NOTIFICATION_SYSTEM_GUIDE.md     # Notification system documentation
└── CI_CD_INTEGRATION_GUIDE.md       # CI/CD integration documentation
```

## Future Enhancements

Planned future enhancements for the plugin architecture:

1. **Plugin Repository**: Online repository for discovering and sharing plugins
2. **Dynamic Loading**: Hot-swapping plugins without restarting the framework
3. **Dependencies**: Managing plugin dependencies and conflicts
4. **UI Integration**: Plugin management through the web interface
5. **Versioning**: Compatibility checking and version management
6. **Marketplace**: Ecosystem for sharing and distributing plugins

## Implementation Status

The Plugin Architecture is part of Phase 8 (Integration and Extensibility) and is now 100% implemented with all planned features completed:

- ✅ Core plugin architecture
- ✅ Plugin discovery and loading
- ✅ Hook system
- ✅ Configuration mechanism
- ✅ Sample plugins
- ✅ Integration with coordinator
- ✅ Custom scheduler extensibility
- ✅ External systems integration
- ✅ Notification system
  - ✅ Slack integration
  - ✅ Discord integration
  - ✅ Telegram integration
  - ✅ Email integration
  - ✅ MS Teams integration
  - ✅ JIRA integration
- ✅ CI/CD system integration
  - ✅ GitHub Actions
  - ✅ GitLab CI
  - ✅ Jenkins
  - ✅ Azure DevOps
  - ✅ CircleCI
  - ✅ Bitbucket Pipelines
  - ✅ TeamCity

All planned components for Phase 8 (Integration and Extensibility) have been successfully implemented as of May 28, 2025, ahead of the originally planned completion date of June 25, 2025. The framework now provides a comprehensive plugin ecosystem with standardized interfaces for extending functionality without modifying core code.