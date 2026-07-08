# Notification System for Distributed Testing Framework

This guide provides a comprehensive overview of the notification system in the Distributed Testing Framework, including configuration, usage examples, and extending the system with custom notification channels.

## Table of Contents

1. [Overview](#overview)
2. [Supported Notification Channels](#supported-notification-channels)
3. [Configuration](#configuration)
4. [Environment Variables](#environment-variables)
5. [Usage Examples](#usage-examples)
6. [Hook Events](#hook-events)
7. [Plugin Architecture](#plugin-architecture)
8. [Adding Custom Notification Channels](#adding-custom-notification-channels)
9. [Troubleshooting](#troubleshooting)

## Overview

The notification system is implemented as a plugin for the Distributed Testing Framework. It integrates with various external notification services to provide real-time notifications about events in the distributed testing environment, such as task failures, worker disconnections, and coordinator status changes.

The notification system is designed to be:

- **Extensible**: Easily add new notification channels
- **Configurable**: Control which events trigger notifications
- **Fault-tolerant**: Handles connection failures gracefully
- **Group-aware**: Intelligently groups similar notifications to reduce noise

## Supported Notification Channels

The notification system supports the following channels:

| Channel | Description | Required Configuration |
|---------|-------------|------------------------|
| Slack | Messages sent to Slack channels | Slack API token and channel |
| Discord | Messages sent to Discord channels via webhook or bot API | Webhook URL or Bot token |
| Telegram | Messages sent to Telegram chats via Telegram Bot API | Bot token and chat ID |
| Email | Messages sent via SMTP | SMTP server details, credentials |
| MS Teams | Messages sent to Teams channels via webhook or Graph API | Webhook URL or Graph API credentials |
| JIRA | Issues created in JIRA for error events | JIRA API credentials and project key |

## Configuration

The notification system can be configured via environment variables or by directly modifying the configuration dictionary in the `NotificationPlugin` class.

Here's an example of directly configuring the plugin:

```python
from distributed_testing.plugins.notification_plugin import NotificationPlugin

# Create plugin instance
notification_plugin = NotificationPlugin()

# Configure plugin
notification_plugin.config.update({
    "enabled": True,
    "notification_throttle_seconds": 5,
    "group_similar_notifications": True,
    "group_time_window_seconds": 60,
    
    # Discord configuration
    "discord_enabled": True,
    "discord_webhook_url": "https://discord.com/api/webhooks/your-webhook-url",
    
    # Telegram configuration
    "telegram_enabled": True,
    "telegram_bot_token": "your-telegram-bot-token",
    "telegram_default_chat_id": "your-chat-id"
})

# Initialize plugin
await notification_plugin.initialize(coordinator)
```

## Environment Variables

The notification system can be configured using the following environment variables:

### General Settings

- `NOTIFICATION_ENABLED`: Set to "true" to enable notifications (default: "true")
- `NOTIFICATION_THROTTLE_SECONDS`: Minimum time between similar notifications (default: 5)
- `NOTIFICATION_GROUP_SIMILAR`: Set to "true" to group similar notifications (default: "true")
- `NOTIFICATION_GROUP_WINDOW_SECONDS`: Time window for grouping similar notifications (default: 60)

### Slack Configuration

- `SLACK_ENABLED`: Set to "true" to enable Slack notifications
- `SLACK_TOKEN`: Slack API token
- `SLACK_CHANNEL`: Default Slack channel (default: "#distributed-testing")
- `SLACK_USERNAME`: Username for Slack messages (default: "Distributed Testing Framework")
- `SLACK_ICON_EMOJI`: Emoji icon for Slack messages (default: ":robot_face:")

### Discord Configuration

- `DISCORD_ENABLED`: Set to "true" to enable Discord notifications
- `DISCORD_WEBHOOK_URL`: Discord webhook URL (required if using webhooks)
- `DISCORD_BOT_TOKEN`: Discord bot token (required if using Bot API)
- `DISCORD_USE_BOT_API`: Set to "true" to use Discord Bot API instead of webhooks
- `DISCORD_CHANNEL_ID`: Default Discord channel ID (required if using Bot API)
- `DISCORD_USERNAME`: Username for Discord messages (default: "Distributed Testing Framework")
- `DISCORD_AVATAR_URL`: Avatar URL for Discord messages

### Telegram Configuration

- `TELEGRAM_ENABLED`: Set to "true" to enable Telegram notifications
- `TELEGRAM_BOT_TOKEN`: Telegram bot token (required)
- `TELEGRAM_CHAT_ID`: Default Telegram chat ID
- `TELEGRAM_PARSE_MODE`: Parse mode for Telegram messages (default: "HTML")
- `TELEGRAM_DISABLE_WEB_PAGE_PREVIEW`: Set to "true" to disable web page previews
- `TELEGRAM_DISABLE_NOTIFICATION`: Set to "true" to disable notification sounds

### Email Configuration

- `EMAIL_ENABLED`: Set to "true" to enable email notifications
- `EMAIL_SMTP_SERVER`: SMTP server address
- `EMAIL_SMTP_PORT`: SMTP server port (default: "587")
- `EMAIL_USERNAME`: SMTP username
- `EMAIL_PASSWORD`: SMTP password
- `EMAIL_FROM`: Sender email address
- `EMAIL_TO`: Comma-separated list of recipient email addresses
- `EMAIL_USE_TLS`: Set to "true" to use TLS (default: "true")

### MS Teams Configuration

- `MSTEAMS_ENABLED`: Set to "true" to enable MS Teams notifications
- `MSTEAMS_WEBHOOK_URL`: MS Teams webhook URL (required if not using Graph API)
- `MSTEAMS_USE_GRAPH_API`: Set to "true" to use Graph API instead of webhooks
- `MSTEAMS_TENANT_ID`: Microsoft tenant ID (required if using Graph API)
- `MSTEAMS_CLIENT_ID`: Microsoft client ID (required if using Graph API)
- `MSTEAMS_CLIENT_SECRET`: Microsoft client secret (required if using Graph API)
- `MSTEAMS_TEAM_ID`: Default MS Teams team ID (required if using Graph API)
- `MSTEAMS_CHANNEL`: Default MS Teams channel name (default: "General")

### JIRA Configuration

- `JIRA_ENABLED`: Set to "true" to enable JIRA notifications
- `JIRA_EMAIL`: JIRA email address
- `JIRA_TOKEN`: JIRA API token
- `JIRA_SERVER_URL`: JIRA server URL
- `JIRA_PROJECT_KEY`: JIRA project key

## Usage Examples

Here's an example of using the notification system in your code:

```python
import anyio
import os
from distributed_testing.plugins.notification_plugin import NotificationPlugin
from distributed_testing.plugin_architecture import PluginManager, HookType

# Set environment variables
os.environ["DISCORD_ENABLED"] = "true"
os.environ["DISCORD_WEBHOOK_URL"] = "https://discord.com/api/webhooks/your-webhook-url"

# Create and initialize coordinator
coordinator = YourCoordinator()
await coordinator.initialize()

# Use hooks to emit events
await coordinator.plugin_manager.invoke_hook(HookType.TASK_FAILED, "task-123", "Out of memory error")
```

### Example Script

The distributed testing framework includes an example script at `distributed_testing/examples/notification_system_example.py` that demonstrates how to use the notification system with various channels.

Run the example script with the desired notification channels:

```bash
# Enable Discord notifications
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your-webhook-url"
python notification_system_example.py --discord

# Enable Telegram notifications
export TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"
python notification_system_example.py --telegram

# Enable multiple channels
python notification_system_example.py --discord --telegram --slack
```

## Hook Events

The notification system registers hooks for various events in the distributed testing framework. The following events are supported:

| Event | Description | Hook Type |
|-------|-------------|-----------|
| Coordinator startup | Coordinator started | `HookType.COORDINATOR_STARTUP` |
| Coordinator shutdown | Coordinator shutting down | `HookType.COORDINATOR_SHUTDOWN` |
| Task created | New task created | `HookType.TASK_CREATED` |
| Task assigned | Task assigned to a worker | `HookType.TASK_ASSIGNED` |
| Task started | Task started execution | `HookType.TASK_STARTED` |
| Task completed | Task completed successfully | `HookType.TASK_COMPLETED` |
| Task failed | Task failed with an error | `HookType.TASK_FAILED` |
| Task cancelled | Task cancelled by user or system | `HookType.TASK_CANCELLED` |
| Worker registered | New worker registered | `HookType.WORKER_REGISTERED` |
| Worker disconnected | Worker disconnected | `HookType.WORKER_DISCONNECTED` |
| Worker failed | Worker failed with an error | `HookType.WORKER_FAILED` |
| Recovery started | Recovery operation started | `HookType.RECOVERY_STARTED` |
| Recovery completed | Recovery operation completed | `HookType.RECOVERY_COMPLETED` |
| Recovery failed | Recovery operation failed | `HookType.RECOVERY_FAILED` |

By default, only important events like task failures, worker failures, and coordinator events trigger notifications. You can enable notifications for other events by changing the configuration.

## Plugin Architecture

The notification system is implemented as a plugin for the distributed testing framework, using the plugin architecture defined in `plugin_architecture.py`.

The main components are:

- **NotificationPlugin**: The plugin class that integrates with the framework
- **External System Connectors**: Classes that implement the `ExternalSystemInterface` for each notification channel
- **Plugin Manager**: Manages loading and initialization of plugins

## Adding Custom Notification Channels

You can add custom notification channels by implementing the `ExternalSystemInterface` and registering your connector with the `ExternalSystemFactory`. Here's a basic example:

```python
from distributed_testing.external_systems.api_interface import (
    ExternalSystemInterface,
    ConnectorCapabilities,
    ExternalSystemResult,
    ExternalSystemFactory
)

class MyCustomConnector(ExternalSystemInterface):
    """Custom notification connector."""
    
    # Implement all required methods from ExternalSystemInterface
    # ...
    
    async def create_item(self, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a notification in my custom system."""
        if item_type == "message":
            # Send notification to my custom system
            # ...
            return {"success": True}
        else:
            return {"success": False, "error": f"Unsupported item type: {item_type}"}
    
    # Implement other required methods
    # ...

# Register connector with the factory
ExternalSystemFactory.register_connector("mycustom", MyCustomConnector)
```

Then update the `NotificationPlugin` to support your custom channel:

```python
# Add this to the _initialize_connectors method
if self.config.get("mycustom_enabled"):
    try:
        custom_config = {
            "api_key": self.config.get("mycustom_api_key")
            # Add other config options
        }
        
        custom_connector = await ExternalSystemFactory.create_connector("mycustom", custom_config)
        
        # Test connection
        if await custom_connector.connect():
            self.connectors["mycustom"] = custom_connector
            logger.info("Custom connector initialized successfully")
        else:
            logger.error("Failed to connect to custom system")
    except Exception as e:
        logger.error(f"Error initializing custom connector: {str(e)}")

# Add this to the _send_to_connectors method
if "mycustom" in self.connectors:
    await self._send_to_mycustom(notification, message)

# Implement the send method
async def _send_to_mycustom(self, notification: Dict[str, Any], formatted_message: str):
    """Send a notification to my custom system."""
    try:
        # Get custom connector
        custom = self.connectors["mycustom"]
        
        # Format message for my custom system
        # ...
        
        # Send message
        await custom.create_item("message", {
            "content": formatted_message
            # Add other parameters
        })
        
    except Exception as e:
        logger.error(f"Error sending notification to custom system: {str(e)}")
```

## Troubleshooting

If you're having issues with the notification system, check the following:

1. **Environment Variables**: Make sure all required environment variables are set correctly
2. **API Credentials**: Verify your API credentials for each service
3. **Network Connectivity**: Ensure your system can access the notification services
4. **Logs**: Check the logs for error messages using the `logging` module
5. **Plugin Initialization**: Verify that the plugin was successfully initialized
6. **Service Status**: Check if the external services are operational

For Discord specific issues:
- Ensure the webhook URL is valid and hasn't been revoked
- If using Bot API, make sure the bot has the necessary permissions

For Telegram specific issues:
- Ensure the bot token is valid
- Make sure the bot has been added to the chat
- Verify the chat ID is correct

For detailed logs, set the logging level to DEBUG:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```