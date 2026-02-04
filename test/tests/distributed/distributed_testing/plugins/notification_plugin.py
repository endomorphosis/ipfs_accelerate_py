#!/usr/bin/env python3
"""
Notification Plugin for Distributed Testing Framework

This plugin integrates with external notification systems like Slack, email, and more
to provide real-time notifications for events in the distributed testing framework.
"""

import anyio
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

from test.tests.distributed.distributed_testing.plugin_architecture import Plugin, PluginType, HookType
from test.tests.distributed.distributed_testing.external_systems import ExternalSystemFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NotificationPlugin(Plugin):
    """
    Plugin for sending notifications to external systems.
    
    This plugin integrates with the standardized external systems interface to send
    notifications about test runs, worker status, and other events to various notification
    systems like Slack, email, etc.
    """
    
    def __init__(self):
        """Initialize the plugin."""
        super().__init__(
            name="NotificationPlugin",
            version="1.0.0",
            plugin_type=PluginType.NOTIFICATION
        )
        
        # Notification connectors
        self.connectors = {}
        
        # Notification history
        self.notifications = []
        
        # Default configuration
        self.config = {
            # General settings
            "enabled": True,
            "notification_throttle_seconds": 5,
            "group_similar_notifications": True,
            "group_time_window_seconds": 60,
            
            # Event types to notify on
            "notify_task_failed": True,
            "notify_worker_failed": True,
            "notify_test_completion": True,
            "notify_coordinator_events": True,
            
            # General notification settings
            "max_history": 1000,
            "include_timestamps": True,
            "detailed_errors": True,
            
            # Event filters
            "notify_coordinator_startup": True,
            "notify_coordinator_shutdown": True,
            "notify_task_created": False,  # High volume, off by default
            "notify_task_assigned": False,  # High volume, off by default
            "notify_task_started": False,   # High volume, off by default
            "notify_task_completed": False, # High volume, off by default
            "notify_task_failed": True,     # Important, on by default
            "notify_task_cancelled": True,  # Important, on by default
            "notify_worker_registered": True,
            "notify_worker_disconnected": True,
            "notify_worker_failed": True,
            "notify_recovery_started": True,
            "notify_recovery_completed": True,
            "notify_recovery_failed": True,
            
            # Notification grouping (to reduce volume)
            "group_similar_notifications": True,
            "group_time_window_seconds": 60,
            
            # Slack configuration
            "slack_enabled": os.environ.get("SLACK_ENABLED", "false").lower() == "true",
            "slack_token": os.environ.get("SLACK_TOKEN", ""),
            "slack_default_channel": os.environ.get("SLACK_CHANNEL", "#distributed-testing"),
            "slack_username": os.environ.get("SLACK_USERNAME", "Distributed Testing Framework"),
            "slack_icon_emoji": os.environ.get("SLACK_ICON_EMOJI", ":robot_face:"),
            
            # JIRA configuration
            "jira_enabled": os.environ.get("JIRA_ENABLED", "false").lower() == "true",
            "jira_email": os.environ.get("JIRA_EMAIL", ""),
            "jira_token": os.environ.get("JIRA_TOKEN", ""),
            "jira_server_url": os.environ.get("JIRA_SERVER_URL", ""),
            "jira_project_key": os.environ.get("JIRA_PROJECT_KEY", ""),
            
            # Email configuration
            "email_enabled": os.environ.get("EMAIL_ENABLED", "false").lower() == "true",
            "email_smtp_server": os.environ.get("EMAIL_SMTP_SERVER", ""),
            "email_smtp_port": int(os.environ.get("EMAIL_SMTP_PORT", "587")),
            "email_username": os.environ.get("EMAIL_USERNAME", ""),
            "email_password": os.environ.get("EMAIL_PASSWORD", ""),
            "email_from": os.environ.get("EMAIL_FROM", ""),
            "email_to": os.environ.get("EMAIL_TO", ""),
            "email_use_tls": os.environ.get("EMAIL_USE_TLS", "true").lower() == "true",
            
            # MS Teams configuration
            "msteams_enabled": os.environ.get("MSTEAMS_ENABLED", "false").lower() == "true",
            "msteams_webhook_url": os.environ.get("MSTEAMS_WEBHOOK_URL", ""),
            "msteams_use_graph_api": os.environ.get("MSTEAMS_USE_GRAPH_API", "false").lower() == "true",
            "msteams_tenant_id": os.environ.get("MSTEAMS_TENANT_ID", ""),
            "msteams_client_id": os.environ.get("MSTEAMS_CLIENT_ID", ""),
            "msteams_client_secret": os.environ.get("MSTEAMS_CLIENT_SECRET", ""),
            "msteams_team_id": os.environ.get("MSTEAMS_TEAM_ID", ""),
            "msteams_channel": os.environ.get("MSTEAMS_CHANNEL", "General"),
            
            # Discord configuration
            "discord_enabled": os.environ.get("DISCORD_ENABLED", "false").lower() == "true",
            "discord_webhook_url": os.environ.get("DISCORD_WEBHOOK_URL", ""),
            "discord_bot_token": os.environ.get("DISCORD_BOT_TOKEN", ""),
            "discord_use_bot_api": os.environ.get("DISCORD_USE_BOT_API", "false").lower() == "true",
            "discord_default_channel_id": os.environ.get("DISCORD_CHANNEL_ID", ""),
            "discord_username": os.environ.get("DISCORD_USERNAME", "Distributed Testing Framework"),
            "discord_avatar_url": os.environ.get("DISCORD_AVATAR_URL", ""),
            
            # Telegram configuration
            "telegram_enabled": os.environ.get("TELEGRAM_ENABLED", "false").lower() == "true",
            "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            "telegram_default_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
            "telegram_parse_mode": os.environ.get("TELEGRAM_PARSE_MODE", "HTML"),
            "telegram_disable_web_page_preview": os.environ.get("TELEGRAM_DISABLE_WEB_PAGE_PREVIEW", "false").lower() == "true",
            "telegram_disable_notification": os.environ.get("TELEGRAM_DISABLE_NOTIFICATION", "false").lower() == "true"
        }
        
        # State
        self.last_notification_time = {}
        self.notification_groups = {}
        
        # Register hooks
        self._register_hooks()
        
        logger.info("NotificationPlugin initialized")
    
    def _register_hooks(self):
        """Register all hooks based on configuration."""
        # Coordinator hooks
        if self.config["notify_coordinator_startup"]:
            self.register_hook(HookType.COORDINATOR_STARTUP, self.on_coordinator_startup)
            
        if self.config["notify_coordinator_shutdown"]:
            self.register_hook(HookType.COORDINATOR_SHUTDOWN, self.on_coordinator_shutdown)
        
        # Task hooks
        if self.config["notify_task_created"]:
            self.register_hook(HookType.TASK_CREATED, self.on_task_created)
            
        if self.config["notify_task_assigned"]:
            self.register_hook(HookType.TASK_ASSIGNED, self.on_task_assigned)
            
        if self.config["notify_task_started"]:
            self.register_hook(HookType.TASK_STARTED, self.on_task_started)
            
        if self.config["notify_task_completed"]:
            self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
            
        if self.config["notify_task_failed"]:
            self.register_hook(HookType.TASK_FAILED, self.on_task_failed)
            
        if self.config["notify_task_cancelled"]:
            self.register_hook(HookType.TASK_CANCELLED, self.on_task_cancelled)
        
        # Worker hooks
        if self.config["notify_worker_registered"]:
            self.register_hook(HookType.WORKER_REGISTERED, self.on_worker_registered)
            
        if self.config["notify_worker_disconnected"]:
            self.register_hook(HookType.WORKER_DISCONNECTED, self.on_worker_disconnected)
            
        if self.config["notify_worker_failed"]:
            self.register_hook(HookType.WORKER_FAILED, self.on_worker_failed)
        
        # Recovery hooks
        if self.config["notify_recovery_started"]:
            self.register_hook(HookType.RECOVERY_STARTED, self.on_recovery_started)
            
        if self.config["notify_recovery_completed"]:
            self.register_hook(HookType.RECOVERY_COMPLETED, self.on_recovery_completed)
            
        if self.config["notify_recovery_failed"]:
            self.register_hook(HookType.RECOVERY_FAILED, self.on_recovery_failed)
    
    async def initialize(self, coordinator) -> bool:
        """
        Initialize the plugin with reference to the coordinator.
        
        Args:
            coordinator: Reference to the coordinator instance
            
        Returns:
            True if initialization succeeded
        """
        self.coordinator = coordinator
        
        # Initialize connectors
        await self._initialize_connectors()
        
        logger.info("NotificationPlugin initialized with coordinator")
        return True
    
    async def _initialize_connectors(self):
        """Initialize all enabled notification connectors."""
        # Initialize Slack connector if enabled
        if self.config["slack_enabled"] and self.config["slack_token"]:
            try:
                slack_config = {
                    "token": self.config["slack_token"],
                    "default_channel": self.config["slack_default_channel"]
                }
                
                slack_connector = await ExternalSystemFactory.create_connector("slack", slack_config)
                
                # Test connection
                if await slack_connector.connect():
                    self.connectors["slack"] = slack_connector
                    logger.info("Slack connector initialized successfully")
                else:
                    logger.error("Failed to connect to Slack")
            except Exception as e:
                logger.error(f"Error initializing Slack connector: {str(e)}")
                
        # Initialize JIRA connector if enabled
        if self.config["jira_enabled"] and self.config["jira_email"] and self.config["jira_token"]:
            try:
                jira_config = {
                    "email": self.config["jira_email"],
                    "token": self.config["jira_token"],
                    "server_url": self.config["jira_server_url"],
                    "project_key": self.config["jira_project_key"]
                }
                
                jira_connector = await ExternalSystemFactory.create_connector("jira", jira_config)
                
                # Test connection
                if await jira_connector.connect():
                    self.connectors["jira"] = jira_connector
                    logger.info("JIRA connector initialized successfully")
                else:
                    logger.error("Failed to connect to JIRA")
            except Exception as e:
                logger.error(f"Error initializing JIRA connector: {str(e)}")
                
        # Initialize Email connector if enabled
        if self.config["email_enabled"] and self.config["email_smtp_server"]:
            try:
                email_config = {
                    "smtp_server": self.config["email_smtp_server"],
                    "smtp_port": self.config["email_smtp_port"],
                    "username": self.config["email_username"],
                    "password": self.config["email_password"],
                    "use_tls": self.config["email_use_tls"],
                    "default_sender": self.config["email_from"],
                    "default_recipients": self.config["email_to"].split(",") if self.config["email_to"] else []
                }
                
                email_connector = await ExternalSystemFactory.create_connector("email", email_config)
                
                # Test connection
                if await email_connector.connect():
                    self.connectors["email"] = email_connector
                    logger.info("Email connector initialized successfully")
                else:
                    logger.error("Failed to connect to SMTP server")
            except Exception as e:
                logger.error(f"Error initializing Email connector: {str(e)}")
                
        # Initialize MS Teams connector if enabled
        if self.config["msteams_enabled"]:
            try:
                # Setup configuration based on whether we're using Graph API or webhook
                if self.config["msteams_use_graph_api"]:
                    teams_config = {
                        "use_graph_api": True,
                        "tenant_id": self.config["msteams_tenant_id"],
                        "client_id": self.config["msteams_client_id"],
                        "client_secret": self.config["msteams_client_secret"],
                        "default_team_id": self.config["msteams_team_id"],
                        "default_channel": self.config["msteams_channel"]
                    }
                else:
                    teams_config = {
                        "webhook_url": self.config["msteams_webhook_url"]
                    }
                
                teams_connector = await ExternalSystemFactory.create_connector("msteams", teams_config)
                
                # Test connection
                if await teams_connector.connect():
                    self.connectors["msteams"] = teams_connector
                    logger.info("MS Teams connector initialized successfully")
                else:
                    logger.error("Failed to connect to MS Teams")
            except Exception as e:
                logger.error(f"Error initializing MS Teams connector: {str(e)}")
                
        # Initialize Discord connector if enabled
        if self.config["discord_enabled"]:
            try:
                # Setup configuration based on whether we're using Bot API or webhook
                if self.config["discord_use_bot_api"] and self.config["discord_bot_token"]:
                    discord_config = {
                        "use_bot_api": True,
                        "bot_token": self.config["discord_bot_token"],
                        "default_channel_id": self.config["discord_default_channel_id"],
                        "username": self.config["discord_username"],
                        "avatar_url": self.config["discord_avatar_url"]
                    }
                elif self.config["discord_webhook_url"]:
                    discord_config = {
                        "webhook_url": self.config["discord_webhook_url"],
                        "username": self.config["discord_username"],
                        "avatar_url": self.config["discord_avatar_url"]
                    }
                else:
                    logger.error("Either Discord webhook URL or bot token must be provided")
                    return
                
                discord_connector = await ExternalSystemFactory.create_connector("discord", discord_config)
                
                # Test connection
                if await discord_connector.connect():
                    self.connectors["discord"] = discord_connector
                    logger.info("Discord connector initialized successfully")
                else:
                    logger.error("Failed to connect to Discord")
            except Exception as e:
                logger.error(f"Error initializing Discord connector: {str(e)}")
                
        # Initialize Telegram connector if enabled
        if self.config["telegram_enabled"] and self.config["telegram_bot_token"]:
            try:
                telegram_config = {
                    "bot_token": self.config["telegram_bot_token"],
                    "default_chat_id": self.config["telegram_default_chat_id"]
                }
                
                telegram_connector = await ExternalSystemFactory.create_connector("telegram", telegram_config)
                
                # Test connection
                if await telegram_connector.connect():
                    self.connectors["telegram"] = telegram_connector
                    logger.info("Telegram connector initialized successfully")
                else:
                    logger.error("Failed to connect to Telegram")
            except Exception as e:
                logger.error(f"Error initializing Telegram connector: {str(e)}")
    
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin.
        
        Returns:
            True if shutdown succeeded
        """
        # Close all connectors
        for name, connector in self.connectors.items():
            try:
                await connector.close()
                logger.info(f"Closed {name} connector")
            except Exception as e:
                logger.error(f"Error closing {name} connector: {str(e)}")
        
        logger.info("NotificationPlugin shutdown complete")
        return True
    
    async def send_notification(self, event_type: str, message: str, level: str = "info", metadata: Dict[str, Any] = None):
        """
        Send a notification to all configured systems.
        
        Args:
            event_type: Type of event
            message: Notification message
            level: Notification level (info, warning, error)
            metadata: Additional metadata
        """
        if not self.config["enabled"]:
            return
            
        # Create notification object
        notification = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "level": level,
            "metadata": metadata or {}
        }
        
        # Store in history
        self.notifications.append(notification)
        
        # Limit history size
        if len(self.notifications) > self.config["max_history"]:
            self.notifications = self.notifications[-self.config["max_history"]:]
        
        # Check if we should throttle this notification type
        if self._should_throttle(event_type):
            return
            
        # Group similar notifications if enabled
        if self.config["group_similar_notifications"]:
            if self._add_to_group(event_type, message, level):
                # Added to existing group, don't send yet
                return
        
        # Send to each connector
        await self._send_to_connectors(notification)
    
    def _should_throttle(self, event_type: str) -> bool:
        """
        Check if a notification should be throttled.
        
        Args:
            event_type: Type of event
            
        Returns:
            True if notification should be throttled
        """
        now = datetime.now().timestamp()
        throttle_seconds = self.config["notification_throttle_seconds"]
        
        if event_type in self.last_notification_time:
            last_time = self.last_notification_time[event_type]
            
            if now - last_time < throttle_seconds:
                return True
        
        # Update last notification time
        self.last_notification_time[event_type] = now
        return False
    
    def _add_to_group(self, event_type: str, message: str, level: str) -> bool:
        """
        Add a notification to a group if similar ones exist.
        
        Args:
            event_type: Type of event
            message: Notification message
            level: Notification level
            
        Returns:
            True if added to a group
        """
        now = datetime.now().timestamp()
        group_window = self.config["group_time_window_seconds"]
        
        # Create group key
        # For task failures, group by error type
        if event_type == "task_failed" and level == "error":
            # Extract error type if available
            error_type = "Unknown Error"
            
            if ": " in message:
                error_type = message.split(": ")[0]
                
            group_key = f"{event_type}:{error_type}"
        else:
            group_key = event_type
        
        # Check if group exists and is still within window
        if group_key in self.notification_groups:
            group = self.notification_groups[group_key]
            
            if now - group["first_time"] <= group_window:
                # Add to existing group
                group["count"] += 1
                group["last_time"] = now
                
                # If this is the first update after creation, schedule a send
                if group["count"] == 2:
                    # TODO: Replace with task group - anyio task group for delayed notifications
                    
                return True
        
        # Create new group
        self.notification_groups[group_key] = {
            "event_type": event_type,
            "level": level,
            "first_message": message,
            "count": 1,
            "first_time": now,
            "last_time": now
        }
        
        return False
    
    async def _send_group_after_window(self, group_key: str):
        """
        Send a grouped notification after the window expires.
        
        Args:
            group_key: Group identifier
        """
        # Wait for window to expire
        await anyio.sleep(self.config["group_time_window_seconds"])
        
        # Get group data
        if group_key not in self.notification_groups:
            return
            
        group = self.notification_groups[group_key]
        
        # Create grouped notification
        notification = {
            "timestamp": datetime.now().isoformat(),
            "event_type": group["event_type"],
            "message": f"[Grouped] {group['count']} occurrences of: {group['first_message']}",
            "level": group["level"],
            "metadata": {
                "grouped": True,
                "count": group["count"],
                "first_time": datetime.fromtimestamp(group["first_time"]).isoformat(),
                "last_time": datetime.fromtimestamp(group["last_time"]).isoformat(),
                "window_seconds": self.config["group_time_window_seconds"]
            }
        }
        
        # Send notification
        await self._send_to_connectors(notification)
        
        # Remove group
        del self.notification_groups[group_key]
    
    async def _send_to_connectors(self, notification: Dict[str, Any]):
        """
        Send a notification to all configured connectors.
        
        Args:
            notification: Notification data
        """
        # Format timestamp if included
        timestamp_prefix = ""
        if self.config["include_timestamps"]:
            timestamp = datetime.fromisoformat(notification["timestamp"])
            timestamp_prefix = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
        
        # Format message
        message = f"{timestamp_prefix}{notification['message']}"
        
        # Send to Slack if configured
        if "slack" in self.connectors:
            await self._send_to_slack(notification, message)
        
        # Send to JIRA if configured and it's an error
        if "jira" in self.connectors and notification["level"] == "error":
            await self._send_to_jira(notification)
        
        # Send to Email if configured
        if "email" in self.connectors:
            await self._send_to_email(notification, message)
            
        # Send to MS Teams if configured
        if "msteams" in self.connectors:
            await self._send_to_msteams(notification, message)
            
        # Send to Discord if configured
        if "discord" in self.connectors:
            await self._send_to_discord(notification, message)
            
        # Send to Telegram if configured
        if "telegram" in self.connectors:
            await self._send_to_telegram(notification, message)
        
        # Log locally
        log_method = getattr(logger, notification["level"])
        log_method(f"NOTIFICATION: {message}")
    
    async def _send_to_slack(self, notification: Dict[str, Any], formatted_message: str):
        """
        Send a notification to Slack.
        
        Args:
            notification: Notification data
            formatted_message: Formatted message text
        """
        try:
            # Get slack connector
            slack = self.connectors["slack"]
            
            # Set emoji based on level
            emoji = {
                "info": ":information_source:",
                "warning": ":warning:",
                "error": ":rotating_light:"
            }.get(notification["level"], ":information_source:")
            
            # Set color based on level
            color = {
                "info": "#2196F3",  # Blue
                "warning": "#FFC107",  # Amber
                "error": "#F44336"  # Red
            }.get(notification["level"], "#2196F3")
            
            # Create message blocks
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{emoji} *{notification['event_type'].replace('_', ' ').title()}*\n{formatted_message}"
                    }
                }
            ]
            
            # Add metadata section if there's metadata
            if notification.get("metadata") and notification["metadata"]:
                # Format metadata
                metadata_text = ""
                
                for key, value in notification["metadata"].items():
                    if key == "grouped" and value:
                        continue  # Skip grouped flag
                        
                    metadata_text += f"*{key.replace('_', ' ').title()}*: {value}\n"
                
                if metadata_text:
                    blocks.append(
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": metadata_text
                            }
                        }
                    )
            
            # Set appropriate channel based on level
            channel = self.config["slack_default_channel"]
            
            # Send message
            await slack.execute_operation("send_message", {
                "channel": channel,
                "blocks": blocks,
                "username": self.config["slack_username"],
                "icon_emoji": self.config["slack_icon_emoji"]
            })
            
        except Exception as e:
            logger.error(f"Error sending notification to Slack: {str(e)}")
    
    async def _send_to_jira(self, notification: Dict[str, Any]):
        """
        Send an error notification to JIRA as an issue.
        
        Args:
            notification: Notification data
        """
        try:
            # Get JIRA connector
            jira = self.connectors["jira"]
            
            # Only create JIRA issues for errors
            if notification["level"] != "error":
                return
                
            # Create issue summary
            summary = f"[DTF] {notification['event_type'].replace('_', ' ').title()}: {notification['message'][:100]}"
            
            # Create issue description
            timestamp = datetime.fromisoformat(notification["timestamp"])
            description = f"*Error Notification from Distributed Testing Framework*\n\n"
            description += f"*Timestamp:* {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            description += f"*Event Type:* {notification['event_type']}\n"
            description += f"*Message:* {notification['message']}\n\n"
            
            # Add metadata if available
            if notification.get("metadata") and notification["metadata"]:
                description += "*Additional Information:*\n"
                
                for key, value in notification["metadata"].items():
                    if key == "grouped" and value:
                        continue  # Skip grouped flag
                        
                    description += f"*{key.replace('_', ' ').title()}:* {value}\n"
            
            # Create JIRA issue
            issue_data = {
                "project_key": self.config["jira_project_key"],
                "issue_type": "Bug",
                "summary": summary,
                "description": description,
                "priority": "Medium",
                "labels": ["distributed-testing", notification["event_type"], notification["level"]]
            }
            
            # Create issue
            await jira.create_item("issue", issue_data)
            
        except Exception as e:
            logger.error(f"Error sending notification to JIRA: {str(e)}")
            
    async def _send_to_email(self, notification: Dict[str, Any], formatted_message: str):
        """
        Send a notification via email.
        
        Args:
            notification: Notification data
            formatted_message: Formatted message text
        """
        try:
            # Get email connector
            email = self.connectors["email"]
            
            # Set subject based on level and event type
            level_prefix = {
                "info": "INFO",
                "warning": "WARNING",
                "error": "ERROR"
            }.get(notification["level"], "INFO")
            
            subject = f"[{level_prefix}] {notification['event_type'].replace('_', ' ').title()}"
            
            # Create plain text body
            body = f"{formatted_message}\n\n"
            
            # Add metadata if present
            if notification.get("metadata") and notification["metadata"]:
                body += "Additional Information:\n"
                
                for key, value in notification["metadata"].items():
                    if key == "grouped" and value:
                        continue  # Skip grouped flag
                        
                    body += f"* {key.replace('_', ' ').title()}: {value}\n"
            
            # Create HTML body
            html_body = f"<h3>{notification['event_type'].replace('_', ' ').title()}</h3>"
            html_body += f"<p>{formatted_message}</p>"
            
            # Add metadata if present
            if notification.get("metadata") and notification["metadata"]:
                html_body += "<h4>Additional Information:</h4>"
                html_body += "<ul>"
                
                for key, value in notification["metadata"].items():
                    if key == "grouped" and value:
                        continue  # Skip grouped flag
                        
                    html_body += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
                
                html_body += "</ul>"
            
            # Send email
            await email.create_item("email", {
                "subject": subject,
                "body": body,
                "html_body": html_body
            })
            
        except Exception as e:
            logger.error(f"Error sending notification to email: {str(e)}")
            
    async def _send_to_msteams(self, notification: Dict[str, Any], formatted_message: str):
        """
        Send a notification to Microsoft Teams.
        
        Args:
            notification: Notification data
            formatted_message: Formatted message text
        """
        try:
            # Get MS Teams connector
            teams = self.connectors["msteams"]
            
            # Set title based on level and event type
            level_emoji = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "🚨"
            }.get(notification["level"], "ℹ️")
            
            title = f"{level_emoji} {notification['event_type'].replace('_', ' ').title()}"
            
            # Set color based on level
            color = {
                "info": "0078D7",  # Blue
                "warning": "FFC107",  # Amber
                "error": "F44336"  # Red
            }.get(notification["level"], "0078D7")
            
            # Create facts from metadata
            facts = []
            if notification.get("metadata") and notification["metadata"]:
                for key, value in notification["metadata"].items():
                    if key == "grouped" and value:
                        continue  # Skip grouped flag
                        
                    facts.append({"name": key.replace('_', ' ').title(), "value": str(value)})
            
            # For Graph API
            if hasattr(teams, 'use_graph_api') and teams.use_graph_api:
                # Format message for Graph API (HTML)
                content = f"<h3>{title}</h3><p>{formatted_message}</p>"
                
                if facts:
                    content += "<ul>"
                    for fact in facts:
                        content += f"<li><b>{fact['name']}:</b> {fact['value']}</li>"
                    content += "</ul>"
                
                await teams.execute_operation("send_message", {
                    "content": content,
                    "subject": title
                })
            else:
                # For webhook, use adaptive card or message card
                sections = []
                if facts:
                    sections.append({
                        "facts": facts
                    })
                
                await teams.create_item("message", {
                    "title": title,
                    "text": formatted_message,
                    "theme_color": color,
                    "sections": sections
                })
            
        except Exception as e:
            logger.error(f"Error sending notification to MS Teams: {str(e)}")
            
    async def _send_to_discord(self, notification: Dict[str, Any], formatted_message: str):
        """
        Send a notification to Discord.
        
        Args:
            notification: Notification data
            formatted_message: Formatted message text
        """
        try:
            # Get Discord connector
            discord = self.connectors["discord"]
            
            # Set emoji based on level
            level_emoji = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "🚨"
            }.get(notification["level"], "ℹ️")
            
            # Set color based on level
            color = {
                "info": 0x3498DB,  # Blue
                "warning": 0xF1C40F,  # Yellow
                "error": 0xE74C3C   # Red
            }.get(notification["level"], 0x3498DB)
            
            # Create embed title
            title = f"{level_emoji} {notification['event_type'].replace('_', ' ').title()}"
            
            # Create embed fields from metadata
            fields = []
            if notification.get("metadata") and notification["metadata"]:
                for key, value in notification["metadata"].items():
                    if key == "grouped" and value:
                        continue  # Skip grouped flag
                    
                    fields.append({
                        "name": key.replace('_', ' ').title(),
                        "value": str(value),
                        "inline": True
                    })
            
            # Create embed footer with timestamp
            footer = {
                "text": f"Distributed Testing Framework • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
            
            # Send as embed
            await discord.create_item("embed", {
                "title": title,
                "description": formatted_message,
                "color": color,
                "fields": fields,
                "footer": footer
            })
            
        except Exception as e:
            logger.error(f"Error sending notification to Discord: {str(e)}")
            
    async def _send_to_telegram(self, notification: Dict[str, Any], formatted_message: str):
        """
        Send a notification to Telegram.
        
        Args:
            notification: Notification data
            formatted_message: Formatted message text
        """
        try:
            # Get Telegram connector
            telegram = self.connectors["telegram"]
            
            # Set emoji based on level
            level_emoji = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "🚨"
            }.get(notification["level"], "ℹ️")
            
            # Format message with HTML
            event_type = notification['event_type'].replace('_', ' ').title()
            html_message = f"<b>{level_emoji} {event_type}</b>\n\n{formatted_message}"
            
            # Add metadata if present
            if notification.get("metadata") and notification["metadata"]:
                html_message += "\n\n<b>Additional Information:</b>"
                
                for key, value in notification["metadata"].items():
                    if key == "grouped" and value:
                        continue  # Skip grouped flag
                    
                    html_message += f"\n• <b>{key.replace('_', ' ').title()}:</b> {value}"
            
            # Create message
            message_data = {
                "text": html_message,
                "parse_mode": "HTML",
                "disable_web_page_preview": self.config["telegram_disable_web_page_preview"],
                "disable_notification": self.config["telegram_disable_notification"]
            }
            
            # Send message
            await telegram.create_item("message", message_data)
            
        except Exception as e:
            logger.error(f"Error sending notification to Telegram: {str(e)}")
    
    # Hook handlers
    
    async def on_coordinator_startup(self, coordinator):
        """Handle coordinator startup event."""
        await self.send_notification(
            "coordinator_startup",
            "Distributed Testing Coordinator started",
            "info"
        )
    
    async def on_coordinator_shutdown(self, coordinator):
        """Handle coordinator shutdown event."""
        await self.send_notification(
            "coordinator_shutdown",
            "Distributed Testing Coordinator shutting down",
            "info"
        )
    
    async def on_task_created(self, task_id: str, task_data: Dict[str, Any]):
        """Handle task created event."""
        task_type = task_data.get("type", "unknown")
        await self.send_notification(
            "task_created",
            f"Task {task_id} created with type: {task_type}",
            "info",
            metadata={"task_id": task_id, "task_type": task_type}
        )
    
    async def on_task_assigned(self, task_id: str, worker_id: str):
        """Handle task assigned event."""
        await self.send_notification(
            "task_assigned",
            f"Task {task_id} assigned to worker {worker_id}",
            "info",
            metadata={"task_id": task_id, "worker_id": worker_id}
        )
    
    async def on_task_started(self, task_id: str, worker_id: str):
        """Handle task started event."""
        await self.send_notification(
            "task_started",
            f"Task {task_id} started on worker {worker_id}",
            "info",
            metadata={"task_id": task_id, "worker_id": worker_id}
        )
    
    async def on_task_completed(self, task_id: str, result: Any):
        """Handle task completed event."""
        await self.send_notification(
            "task_completed",
            f"Task {task_id} completed successfully",
            "info",
            metadata={"task_id": task_id, "result_summary": str(result)[:100] if result else "None"}
        )
    
    async def on_task_failed(self, task_id: str, error: str):
        """Handle task failed event."""
        await self.send_notification(
            "task_failed",
            f"Task {task_id} failed: {error}",
            "error",
            metadata={"task_id": task_id, "error": error}
        )
    
    async def on_task_cancelled(self, task_id: str, reason: str):
        """Handle task cancelled event."""
        await self.send_notification(
            "task_cancelled",
            f"Task {task_id} cancelled: {reason}",
            "warning",
            metadata={"task_id": task_id, "reason": reason}
        )
    
    async def on_worker_registered(self, worker_id: str, capabilities: Dict[str, Any]):
        """Handle worker registered event."""
        await self.send_notification(
            "worker_registered",
            f"Worker {worker_id} registered with {len(capabilities)} capabilities",
            "info",
            metadata={"worker_id": worker_id, "capabilities_summary": str(list(capabilities.keys()))}
        )
    
    async def on_worker_disconnected(self, worker_id: str):
        """Handle worker disconnected event."""
        await self.send_notification(
            "worker_disconnected",
            f"Worker {worker_id} disconnected",
            "warning",
            metadata={"worker_id": worker_id}
        )
    
    async def on_worker_failed(self, worker_id: str, error: str):
        """Handle worker failed event."""
        await self.send_notification(
            "worker_failed",
            f"Worker {worker_id} failed: {error}",
            "error",
            metadata={"worker_id": worker_id, "error": error}
        )
    
    async def on_recovery_started(self, recovery_id: str, recovery_data: Dict[str, Any]):
        """Handle recovery started event."""
        recovery_type = recovery_data.get("type", "unknown")
        await self.send_notification(
            "recovery_started",
            f"Recovery operation {recovery_id} started for {recovery_type} recovery",
            "warning",
            metadata={"recovery_id": recovery_id, "recovery_type": recovery_type}
        )
    
    async def on_recovery_completed(self, recovery_id: str, result: Dict[str, Any]):
        """Handle recovery completed event."""
        success_count = result.get("success_count", 0)
        failed_count = result.get("failed_count", 0)
        await self.send_notification(
            "recovery_completed",
            f"Recovery operation {recovery_id} completed with {success_count} successes and {failed_count} failures",
            "info",
            metadata={"recovery_id": recovery_id, "success_count": success_count, "failed_count": failed_count}
        )
    
    async def on_recovery_failed(self, recovery_id: str, error: str):
        """Handle recovery failed event."""
        await self.send_notification(
            "recovery_failed",
            f"Recovery operation {recovery_id} failed: {error}",
            "error",
            metadata={"recovery_id": recovery_id, "error": error}
        )
    
    def get_notifications(self, limit: int = None, level: str = None, event_type: str = None) -> List[Dict[str, Any]]:
        """
        Get notification history with optional filtering.
        
        Args:
            limit: Maximum number of notifications to return (from most recent)
            level: Filter by notification level
            event_type: Filter by event type
            
        Returns:
            List of notifications
        """
        notifications = self.notifications.copy()
        
        # Apply filters
        if level:
            notifications = [n for n in notifications if n["level"] == level]
            
        if event_type:
            notifications = [n for n in notifications if n["event_type"] == event_type]
            
        # Apply limit (from most recent)
        if limit is not None and limit > 0:
            notifications = notifications[-limit:]
            
        return notifications
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """
        Get notification statistics.
        
        Returns:
            Dictionary with notification statistics
        """
        # Count by level
        levels = {}
        for notification in self.notifications:
            level = notification["level"]
            levels[level] = levels.get(level, 0) + 1
            
        # Count by event type
        event_types = {}
        for notification in self.notifications:
            event_type = notification["event_type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
        return {
            "total": len(self.notifications),
            "by_level": levels,
            "by_event_type": event_types,
            "connector_status": {name: "connected" for name in self.connectors},
            "config": {
                "enabled": self.config["enabled"],
                "slack_enabled": self.config["slack_enabled"],
                "jira_enabled": self.config["jira_enabled"],
                "email_enabled": self.config["email_enabled"],
                "msteams_enabled": self.config["msteams_enabled"],
                "discord_enabled": self.config["discord_enabled"],
                "telegram_enabled": self.config["telegram_enabled"]
            }
        }