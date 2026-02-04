#!/usr/bin/env python3
"""
Example script demonstrating how to use the Distributed Testing Framework plugins.

This script shows how to:
1. Start a coordinator with plugin support
2. Load and configure plugins
3. Use integration plugins (WebGPU Resource Pool, CI/CD)
4. Use the custom scheduler
5. Create and register a custom notification plugin
"""

import anyio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Ensure distributed_testing is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import coordinator
from test.tests.distributed.distributed_testing.coordinator import DistributedTestingCoordinator

# Import plugin architecture
from test.tests.distributed.distributed_testing.plugin_architecture import Plugin, PluginType, HookType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a simple notification plugin for demonstration
class NotificationPlugin(Plugin):
    """Simple notification plugin for demonstration purposes."""
    
    def __init__(self):
        """Initialize the plugin."""
        super().__init__(
            name="SimpleNotification",
            version="1.0.0",
            plugin_type=PluginType.NOTIFICATION
        )
        
        # Default configuration
        self.config = {
            "notify_task_creation": True,
            "notify_task_completion": True,
            "notify_task_failure": True,
            "notify_worker_events": True
        }
        
        # Notification history
        self.notifications = []
        
        # Register hooks
        self.register_hook(HookType.TASK_CREATED, self.on_task_created)
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        self.register_hook(HookType.TASK_FAILED, self.on_task_failed)
        self.register_hook(HookType.WORKER_REGISTERED, self.on_worker_registered)
        self.register_hook(HookType.WORKER_DISCONNECTED, self.on_worker_disconnected)
        
        logger.info("NotificationPlugin initialized")
    
    async def initialize(self, coordinator) -> bool:
        """Initialize the plugin with reference to the coordinator."""
        self.coordinator = coordinator
        logger.info("NotificationPlugin initialized with coordinator")
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        logger.info("NotificationPlugin shutdown complete")
        return True
    
    async def on_task_created(self, task_id: str, task_data: Dict[str, Any]):
        """Handle task created event."""
        if not self.config["notify_task_creation"]:
            return
            
        message = f"Task {task_id} created with type: {task_data.get('type', 'unknown')}"
        self._send_notification("task_created", message)
    
    async def on_task_completed(self, task_id: str, result: Any):
        """Handle task completed event."""
        if not self.config["notify_task_completion"]:
            return
            
        message = f"Task {task_id} completed successfully"
        self._send_notification("task_completed", message)
    
    async def on_task_failed(self, task_id: str, error: str):
        """Handle task failed event."""
        if not self.config["notify_task_failure"]:
            return
            
        message = f"Task {task_id} failed: {error}"
        self._send_notification("task_failed", message, level="error")
    
    async def on_worker_registered(self, worker_id: str, capabilities: Dict[str, Any]):
        """Handle worker registered event."""
        if not self.config["notify_worker_events"]:
            return
            
        message = f"Worker {worker_id} registered with capabilities: {capabilities}"
        self._send_notification("worker_registered", message)
    
    async def on_worker_disconnected(self, worker_id: str):
        """Handle worker disconnected event."""
        if not self.config["notify_worker_events"]:
            return
            
        message = f"Worker {worker_id} disconnected"
        self._send_notification("worker_disconnected", message, level="warning")
    
    def _send_notification(self, event_type: str, message: str, level: str = "info"):
        """
        Send a notification.
        
        Args:
            event_type: Type of event
            message: Notification message
            level: Notification level (info, warning, error)
        """
        notification = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "level": level
        }
        
        self.notifications.append(notification)
        
        # In a real implementation, this would send to external systems
        # Here we just log it
        log_method = getattr(logger, level)
        log_method(f"NOTIFICATION: {message}")
    
    def get_notifications(self) -> List[Dict[str, Any]]:
        """
        Get all notifications.
        
        Returns:
            List of notification objects
        """
        return self.notifications


async def main():
    """Main function to demonstrate plugin usage."""
    try:
        # Create coordinator with plugin support
        coordinator = DistributedTestingCoordinator(
            db_path=":memory:",  # In-memory database for demo
            host="localhost",
            port=8080,
            enable_plugins=True,
            plugin_dirs=["plugins", "distributed_testing/integration"]
        )
        
        # Create directories for plugins if needed
        os.makedirs("plugins", exist_ok=True)
        os.makedirs("distributed_testing/integration", exist_ok=True)
        
        # Manually write and save notification plugin
        with open("plugins/notification_plugin.py", "w") as f:
            f.write("""
#!/usr/bin/env python3
\"\"\"
Simple Notification Plugin for Distributed Testing Framework
\"\"\"

import logging
from datetime import datetime
from typing import Dict, List, Any

from test.tests.distributed.distributed_testing.plugin_architecture import Plugin, PluginType, HookType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NotificationPlugin(Plugin):
    \"\"\"Simple notification plugin for demonstration purposes.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the plugin.\"\"\"
        super().__init__(
            name="SimpleNotification",
            version="1.0.0",
            plugin_type=PluginType.NOTIFICATION
        )
        
        # Default configuration
        self.config = {
            "notify_task_creation": True,
            "notify_task_completion": True,
            "notify_task_failure": True,
            "notify_worker_events": True
        }
        
        # Notification history
        self.notifications = []
        
        # Register hooks
        self.register_hook(HookType.TASK_CREATED, self.on_task_created)
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        self.register_hook(HookType.TASK_FAILED, self.on_task_failed)
        self.register_hook(HookType.WORKER_REGISTERED, self.on_worker_registered)
        self.register_hook(HookType.WORKER_DISCONNECTED, self.on_worker_disconnected)
        
        logger.info("NotificationPlugin initialized")
    
    async def initialize(self, coordinator) -> bool:
        \"\"\"Initialize the plugin with reference to the coordinator.\"\"\"
        self.coordinator = coordinator
        logger.info("NotificationPlugin initialized with coordinator")
        return True
    
    async def shutdown(self) -> bool:
        \"\"\"Shutdown the plugin.\"\"\"
        logger.info("NotificationPlugin shutdown complete")
        return True
    
    async def on_task_created(self, task_id: str, task_data: Dict[str, Any]):
        \"\"\"Handle task created event.\"\"\"
        if not self.config["notify_task_creation"]:
            return
            
        message = f"Task {task_id} created with type: {task_data.get('type', 'unknown')}"
        self._send_notification("task_created", message)
    
    async def on_task_completed(self, task_id: str, result: Any):
        \"\"\"Handle task completed event.\"\"\"
        if not self.config["notify_task_completion"]:
            return
            
        message = f"Task {task_id} completed successfully"
        self._send_notification("task_completed", message)
    
    async def on_task_failed(self, task_id: str, error: str):
        \"\"\"Handle task failed event.\"\"\"
        if not self.config["notify_task_failure"]:
            return
            
        message = f"Task {task_id} failed: {error}"
        self._send_notification("task_failed", message, level="error")
    
    async def on_worker_registered(self, worker_id: str, capabilities: Dict[str, Any]):
        \"\"\"Handle worker registered event.\"\"\"
        if not self.config["notify_worker_events"]:
            return
            
        message = f"Worker {worker_id} registered with capabilities: {capabilities}"
        self._send_notification("worker_registered", message)
    
    async def on_worker_disconnected(self, worker_id: str):
        \"\"\"Handle worker disconnected event.\"\"\"
        if not self.config["notify_worker_events"]:
            return
            
        message = f"Worker {worker_id} disconnected"
        self._send_notification("worker_disconnected", message, level="warning")
    
    def _send_notification(self, event_type: str, message: str, level: str = "info"):
        \"\"\"
        Send a notification.
        
        Args:
            event_type: Type of event
            message: Notification message
            level: Notification level (info, warning, error)
        \"\"\"
        notification = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "level": level
        }
        
        self.notifications.append(notification)
        
        # In a real implementation, this would send to external systems
        # Here we just log it
        log_method = getattr(logger, level)
        log_method(f"NOTIFICATION: {message}")
    
    def get_notifications(self) -> List[Dict[str, Any]]:
        \"\"\"
        Get all notifications.
        
        Returns:
            List of notification objects
        \"\"\"
        return self.notifications
""")
        
        # Start coordinator
        logger.info("Starting coordinator...")
        await coordinator.start()
        
        # Get plugin manager
        plugin_manager = coordinator.plugin_manager
        
        # Discover and load plugins
        discovered_plugins = await plugin_manager.discover_plugins()
        logger.info(f"Discovered plugins: {discovered_plugins}")
        
        # Load notification plugin
        notification_plugin_id = await plugin_manager.load_plugin("notification_plugin")
        
        if notification_plugin_id:
            logger.info(f"Loaded notification plugin: {notification_plugin_id}")
            
            # Configure notification plugin
            notification_plugin = plugin_manager.get_plugin(notification_plugin_id)
            
            if notification_plugin:
                await plugin_manager.configure_plugin(notification_plugin_id, {
                    "notify_task_creation": True,
                    "notify_task_completion": True,
                    "notify_task_failure": True,
                    "notify_worker_events": True
                })
        
        # Register a mock worker for demonstration
        mock_worker_id = "worker-001"
        await coordinator.plugin_manager.invoke_hook(
            HookType.WORKER_REGISTERED,
            mock_worker_id,
            {
                "hardware_type": "gpu",
                "cpu_cores": 8,
                "memory_gb": 16,
                "gpu_memory_gb": 8,
                "supports_cuda": True,
                "supports_webgpu": True,
                "supports_webnn": True
            }
        )
        
        # Simulate tasks
        for i in range(5):
            task_id = f"task-{i+1}"
            
            # Create task
            await coordinator.plugin_manager.invoke_hook(
                HookType.TASK_CREATED,
                task_id,
                {
                    "type": "model_test",
                    "model_name": f"model-{i+1}",
                    "priority": 5,
                    "hardware_requirements": {
                        "gpu_memory_gb": 4,
                        "supports_cuda": True
                    },
                    "deadline": (datetime.now() + timedelta(minutes=10)).isoformat()
                }
            )
            
            # Wait a bit
            await anyio.sleep(0.5)
            
            # Simulate task completion or failure
            if i % 4 != 0:  # 75% success rate
                await coordinator.plugin_manager.invoke_hook(
                    HookType.TASK_COMPLETED,
                    task_id,
                    {"status": "success", "metrics": {"accuracy": 0.95, "latency": 1.5}}
                )
            else:
                await coordinator.plugin_manager.invoke_hook(
                    HookType.TASK_FAILED,
                    task_id,
                    "Model test failed due to out of memory error"
                )
        
        # Wait a bit to allow notification processing
        await anyio.sleep(1)
        
        # Print summary
        if notification_plugin_id:
            notification_plugin = plugin_manager.get_plugin(notification_plugin_id)
            
            if notification_plugin:
                notifications = notification_plugin.get_notifications()
                
                logger.info("Notification Summary:")
                logger.info(f"Total notifications: {len(notifications)}")
                
                by_level = {}
                for n in notifications:
                    level = n["level"]
                    by_level[level] = by_level.get(level, 0) + 1
                
                for level, count in by_level.items():
                    logger.info(f"  {level}: {count}")
                
                logger.info("Last 3 notifications:")
                for n in notifications[-3:]:
                    logger.info(f"  [{n['level']}] {n['message']}")
        
        # Clean up
        logger.info("Shutting down coordinator...")
        
        # Disconnect mock worker
        await coordinator.plugin_manager.invoke_hook(
            HookType.WORKER_DISCONNECTED,
            mock_worker_id
        )
        
        # Shutdown coordinator
        await coordinator.shutdown()
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")


if __name__ == "__main__":
    anyio.run(main())