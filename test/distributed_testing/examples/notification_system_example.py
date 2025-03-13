#!/usr/bin/env python3
"""
Example script for using the Notification System

This script demonstrates how to set up and use the notification system with
various external systems including Discord and Telegram.
"""

import asyncio
import logging
import os
import sys
import argparse
from typing import Dict, Any

# Add parent directory to system path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plugins.notification_plugin import NotificationPlugin
from plugin_architecture import PluginManager, HookType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockCoordinator:
    """Mock coordinator for testing the notification plugin."""
    
    def __init__(self):
        """Initialize the mock coordinator."""
        self.plugin_manager = PluginManager(self)
        self.worker_ids = ["worker-1", "worker-2", "worker-3"]
        self.task_ids = []
        
    async def initialize(self):
        """Initialize the coordinator."""
        # Initialize the plugin manager
        await self.plugin_manager.initialize()
        
        # Discover and load plugins
        plugins = await self.plugin_manager.discover_plugins()
        for plugin_name in plugins:
            if "notification" in plugin_name.lower():
                logger.info(f"Loading plugin: {plugin_name}")
                await self.plugin_manager.load_plugin(plugin_name)
        
        # Print loaded plugins
        loaded_plugins = self.plugin_manager.get_all_plugins()
        for plugin_id, plugin in loaded_plugins.items():
            logger.info(f"Loaded plugin: {plugin.name} v{plugin.version} ({plugin.plugin_type.value})")
            
    async def generate_test_notifications(self):
        """Generate test notifications to demonstrate the notification system."""
        # Emit coordinator startup event
        await self.plugin_manager.invoke_hook(HookType.COORDINATOR_STARTUP, self)
        
        # Emit worker registration events
        for worker_id in self.worker_ids:
            capabilities = {
                "cpu": 8,
                "memory": 16384,
                "disk": 512,
                "gpu": True,
                "os": "Linux"
            }
            await self.plugin_manager.invoke_hook(HookType.WORKER_REGISTERED, worker_id, capabilities)
            
        # Emit task creation events
        for i in range(5):
            task_id = f"task-{i+1}"
            self.task_ids.append(task_id)
            task_data = {
                "type": "test",
                "name": f"Test Task {i+1}",
                "priority": i % 3,
                "parameters": {
                    "param1": f"value{i+1}",
                    "param2": i * 10
                }
            }
            await self.plugin_manager.invoke_hook(HookType.TASK_CREATED, task_id, task_data)
            
        # Emit task assignment events
        for i, task_id in enumerate(self.task_ids):
            worker_id = self.worker_ids[i % len(self.worker_ids)]
            await self.plugin_manager.invoke_hook(HookType.TASK_ASSIGNED, task_id, worker_id)
            
        # Emit task started events
        for i, task_id in enumerate(self.task_ids):
            worker_id = self.worker_ids[i % len(self.worker_ids)]
            await self.plugin_manager.invoke_hook(HookType.TASK_STARTED, task_id, worker_id)
            
        # Emit task completed events for some tasks
        for i in range(3):
            task_id = self.task_ids[i]
            result = {
                "status": "completed",
                "execution_time": i * 5.3,
                "output": f"Output for task {task_id}"
            }
            await self.plugin_manager.invoke_hook(HookType.TASK_COMPLETED, task_id, result)
            
        # Emit task failed events for some tasks
        task_id = self.task_ids[3]
        error = "Out of memory error occurred during task execution"
        await self.plugin_manager.invoke_hook(HookType.TASK_FAILED, task_id, error)
        
        # Emit task cancelled events for some tasks
        task_id = self.task_ids[4]
        reason = "Cancelled by user"
        await self.plugin_manager.invoke_hook(HookType.TASK_CANCELLED, task_id, reason)
        
        # Emit worker disconnected events
        worker_id = self.worker_ids[0]
        await self.plugin_manager.invoke_hook(HookType.WORKER_DISCONNECTED, worker_id)
        
        # Emit worker failed events
        worker_id = self.worker_ids[1]
        error = "Connection timeout after 30 seconds"
        await self.plugin_manager.invoke_hook(HookType.WORKER_FAILED, worker_id, error)
        
        # Emit recovery started events
        recovery_id = "recovery-1"
        recovery_data = {
            "type": "worker_failure",
            "worker_id": worker_id,
            "affected_tasks": [self.task_ids[i] for i in range(1, 3)]
        }
        await self.plugin_manager.invoke_hook(HookType.RECOVERY_STARTED, recovery_id, recovery_data)
        
        # Emit recovery completed events
        result = {
            "success_count": 2,
            "failed_count": 0,
            "recovered_tasks": [self.task_ids[i] for i in range(1, 3)]
        }
        await self.plugin_manager.invoke_hook(HookType.RECOVERY_COMPLETED, recovery_id, result)
        
        # Emit coordinator shutdown event
        await self.plugin_manager.invoke_hook(HookType.COORDINATOR_SHUTDOWN, self)
        
    async def shutdown(self):
        """Shutdown the coordinator."""
        await self.plugin_manager.shutdown()
        

async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Notification System Example")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--slack", action="store_true", help="Enable Slack notifications")
    parser.add_argument("--discord", action="store_true", help="Enable Discord notifications")
    parser.add_argument("--telegram", action="store_true", help="Enable Telegram notifications")
    parser.add_argument("--email", action="store_true", help="Enable Email notifications")
    parser.add_argument("--msteams", action="store_true", help="Enable MS Teams notifications")
    parser.add_argument("--jira", action="store_true", help="Enable JIRA notifications")
    args = parser.parse_args()
    
    # Set environment variables based on command line arguments
    if args.slack:
        os.environ["SLACK_ENABLED"] = "true"
        # Check if SLACK_TOKEN is set
        if "SLACK_TOKEN" not in os.environ:
            logger.error("SLACK_TOKEN environment variable is required for Slack notifications")
            return
    
    if args.discord:
        os.environ["DISCORD_ENABLED"] = "true"
        # Check if either DISCORD_WEBHOOK_URL or DISCORD_BOT_TOKEN is set
        if "DISCORD_WEBHOOK_URL" not in os.environ and "DISCORD_BOT_TOKEN" not in os.environ:
            logger.error("Either DISCORD_WEBHOOK_URL or DISCORD_BOT_TOKEN environment variable is required for Discord notifications")
            return
    
    if args.telegram:
        os.environ["TELEGRAM_ENABLED"] = "true"
        # Check if TELEGRAM_BOT_TOKEN is set
        if "TELEGRAM_BOT_TOKEN" not in os.environ:
            logger.error("TELEGRAM_BOT_TOKEN environment variable is required for Telegram notifications")
            return
    
    if args.email:
        os.environ["EMAIL_ENABLED"] = "true"
        # Check if required email environment variables are set
        required_vars = ["EMAIL_SMTP_SERVER", "EMAIL_USERNAME", "EMAIL_PASSWORD", "EMAIL_FROM", "EMAIL_TO"]
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            logger.error(f"The following environment variables are required for Email notifications: {', '.join(missing_vars)}")
            return
    
    if args.msteams:
        os.environ["MSTEAMS_ENABLED"] = "true"
        # Check if MSTEAMS_WEBHOOK_URL is set or Graph API credentials
        if "MSTEAMS_WEBHOOK_URL" not in os.environ and not (
            "MSTEAMS_USE_GRAPH_API" in os.environ and
            os.environ["MSTEAMS_USE_GRAPH_API"].lower() == "true" and
            "MSTEAMS_TENANT_ID" in os.environ and
            "MSTEAMS_CLIENT_ID" in os.environ and
            "MSTEAMS_CLIENT_SECRET" in os.environ
        ):
            logger.error("Either MSTEAMS_WEBHOOK_URL or Graph API credentials (MSTEAMS_USE_GRAPH_API, MSTEAMS_TENANT_ID, MSTEAMS_CLIENT_ID, MSTEAMS_CLIENT_SECRET) are required for MS Teams notifications")
            return
    
    if args.jira:
        os.environ["JIRA_ENABLED"] = "true"
        # Check if required JIRA environment variables are set
        required_vars = ["JIRA_EMAIL", "JIRA_TOKEN", "JIRA_SERVER_URL", "JIRA_PROJECT_KEY"]
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            logger.error(f"The following environment variables are required for JIRA notifications: {', '.join(missing_vars)}")
            return
    
    # Create and initialize the mock coordinator
    coordinator = MockCoordinator()
    await coordinator.initialize()
    
    try:
        # Generate test notifications
        logger.info("Generating test notifications...")
        await coordinator.generate_test_notifications()
        
        # Give some time for the notifications to be sent
        logger.info("Waiting for notifications to be sent...")
        await asyncio.sleep(5)
        
        # Get notification stats
        notification_plugin = None
        for plugin_id, plugin in coordinator.plugin_manager.get_all_plugins().items():
            if plugin.name == "NotificationPlugin":
                notification_plugin = plugin
                break
                
        if notification_plugin:
            stats = notification_plugin.get_notification_stats()
            logger.info(f"Notification stats: {stats}")
            
    finally:
        # Shutdown the coordinator
        await coordinator.shutdown()
        

if __name__ == "__main__":
    asyncio.run(main())