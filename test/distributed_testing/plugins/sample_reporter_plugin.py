#!/usr/bin/env python3
"""
Sample Reporter Plugin for Distributed Testing Framework

This plugin demonstrates how to create a task reporter plugin that hooks into
task lifecycle events and reports task status to an external system.
"""

import anyio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import plugin base class
from plugin_architecture import Plugin, PluginType, HookType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskReporterPlugin(Plugin):
    """
    Sample reporter plugin that logs task events and can send to external systems.
    
    This plugin hooks into task lifecycle events and reports task status to an
    external system (simulated in this example).
    """
    
    def __init__(self):
        """Initialize the plugin."""
        super().__init__(
            name="TaskReporter",
            version="1.0.0",
            plugin_type=PluginType.REPORTER
        )
        
        # Task tracking
        self.tasks = {}
        
        # Statistics
        self.stats = {
            "tasks_created": 0,
            "tasks_started": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0
        }
        
        # External system connection (would be a real API client in production)
        self.external_system = None
        
        # Default configuration
        self.config = {
            "enable_external_reporting": False,
            "external_system_url": "https://example.com/api/reports",
            "report_interval": 60,
            "log_level": "INFO"
        }
        
        # Register hooks
        self.register_hook(HookType.TASK_CREATED, self.on_task_created)
        self.register_hook(HookType.TASK_ASSIGNED, self.on_task_assigned)
        self.register_hook(HookType.TASK_STARTED, self.on_task_started)
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        self.register_hook(HookType.TASK_FAILED, self.on_task_failed)
        self.register_hook(HookType.TASK_CANCELLED, self.on_task_cancelled)
        
        logger.info("TaskReporterPlugin initialized")
    
    async def initialize(self, coordinator) -> bool:
        """
        Initialize the plugin with reference to the coordinator.
        
        Args:
            coordinator: Reference to the coordinator instance
            
        Returns:
            True if initialization succeeded
        """
        # Store coordinator reference
        self.coordinator = coordinator
        
        # Initialize external system connection if enabled
        if self.config["enable_external_reporting"]:
            await self._connect_to_external_system()
        
        # Start periodic reporting task
        self.reporting_task = # TODO: Replace with task group - asyncio.create_task(self._periodic_reporting())
        
        logger.info("TaskReporterPlugin initialized with coordinator")
        return True
    
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin.
        
        Returns:
            True if shutdown succeeded
        """
        # Cancel reporting task
        if hasattr(self, "reporting_task") and self.reporting_task:
            self.reporting_task.cancel()
            try:
                await self.reporting_task
            except anyio.get_cancelled_exc_class():
                pass
        
        # Disconnect from external system
        if self.external_system:
            await self._disconnect_from_external_system()
        
        logger.info("TaskReporterPlugin shutdown complete")
        return True
    
    async def _connect_to_external_system(self):
        """Connect to external reporting system."""
        logger.info(f"Connecting to external system at {self.config['external_system_url']}")
        
        # Simulate connection to external system
        await anyio.sleep(0.5)
        
        # In a real implementation, would create an API client
        self.external_system = {
            "connected": True,
            "url": self.config["external_system_url"],
            "connect_time": datetime.now().isoformat()
        }
        
        logger.info("Connected to external reporting system")
    
    async def _disconnect_from_external_system(self):
        """Disconnect from external reporting system."""
        if not self.external_system:
            return
            
        logger.info("Disconnecting from external reporting system")
        
        # Simulate disconnection
        await anyio.sleep(0.1)
        
        self.external_system["connected"] = False
        self.external_system = None
        
        logger.info("Disconnected from external reporting system")
    
    async def _periodic_reporting(self):
        """Periodic reporting task to send summary stats to external system."""
        while True:
            try:
                # Sleep for reporting interval
                await anyio.sleep(self.config["report_interval"])
                
                # Skip if external reporting is disabled
                if not self.config["enable_external_reporting"]:
                    continue
                    
                # Skip if not connected to external system
                if not self.external_system or not self.external_system["connected"]:
                    logger.warning("Not connected to external reporting system")
                    continue
                
                # Create summary report
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "stats": self.stats,
                    "active_tasks": len([t for t in self.tasks.values() if t["status"] in ["assigned", "started"]]),
                    "total_tasks": len(self.tasks)
                }
                
                # Send report
                await self._send_to_external_system("summary", report)
                
            except anyio.get_cancelled_exc_class():
                logger.info("Periodic reporting task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in periodic reporting: {str(e)}")
    
    async def _send_to_external_system(self, event_type: str, data: Dict[str, Any]):
        """
        Send event data to external system.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if not self.config["enable_external_reporting"]:
            return
            
        if not self.external_system or not self.external_system["connected"]:
            logger.warning("Not connected to external reporting system")
            return
        
        logger.info(f"Sending {event_type} event to external system")
        
        # Simulate sending to external system
        await anyio.sleep(0.1)
        
        # In a real implementation, would make an API call
        logger.debug(f"Sent {event_type} event: {json.dumps(data)}")
    
    # Hook handlers
    
    async def on_task_created(self, task_id: str, task_data: Dict[str, Any]):
        """
        Handle task created event.
        
        Args:
            task_id: Task ID
            task_data: Task data
        """
        logger.info(f"Task created: {task_id}")
        
        # Store task
        self.tasks[task_id] = {
            "id": task_id,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "data": task_data
        }
        
        # Update stats
        self.stats["tasks_created"] += 1
        
        # Send to external system
        if self.config["enable_external_reporting"]:
            await self._send_to_external_system("task_created", {
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
                "data": task_data
            })
    
    async def on_task_assigned(self, task_id: str, worker_id: str):
        """
        Handle task assigned event.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
        """
        logger.info(f"Task {task_id} assigned to worker {worker_id}")
        
        # Update task
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "assigned"
            self.tasks[task_id]["worker_id"] = worker_id
            self.tasks[task_id]["assigned_at"] = datetime.now().isoformat()
        
        # Send to external system
        if self.config["enable_external_reporting"]:
            await self._send_to_external_system("task_assigned", {
                "task_id": task_id,
                "worker_id": worker_id,
                "timestamp": datetime.now().isoformat()
            })
    
    async def on_task_started(self, task_id: str, worker_id: str):
        """
        Handle task started event.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
        """
        logger.info(f"Task {task_id} started on worker {worker_id}")
        
        # Update task
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "started"
            self.tasks[task_id]["worker_id"] = worker_id
            self.tasks[task_id]["started_at"] = datetime.now().isoformat()
        
        # Update stats
        self.stats["tasks_started"] += 1
        
        # Send to external system
        if self.config["enable_external_reporting"]:
            await self._send_to_external_system("task_started", {
                "task_id": task_id,
                "worker_id": worker_id,
                "timestamp": datetime.now().isoformat()
            })
    
    async def on_task_completed(self, task_id: str, result: Any):
        """
        Handle task completed event.
        
        Args:
            task_id: Task ID
            result: Task result
        """
        logger.info(f"Task {task_id} completed")
        
        # Update task
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["result"] = result
            
            # Calculate duration if started_at is available
            if "started_at" in self.tasks[task_id]:
                started = datetime.fromisoformat(self.tasks[task_id]["started_at"])
                completed = datetime.fromisoformat(self.tasks[task_id]["completed_at"])
                duration = (completed - started).total_seconds()
                self.tasks[task_id]["duration"] = duration
        
        # Update stats
        self.stats["tasks_completed"] += 1
        
        # Send to external system
        if self.config["enable_external_reporting"]:
            await self._send_to_external_system("task_completed", {
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
                "result": result
            })
    
    async def on_task_failed(self, task_id: str, error: str):
        """
        Handle task failed event.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        logger.info(f"Task {task_id} failed: {error}")
        
        # Update task
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["failed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["error"] = error
            
            # Calculate duration if started_at is available
            if "started_at" in self.tasks[task_id]:
                started = datetime.fromisoformat(self.tasks[task_id]["started_at"])
                failed = datetime.fromisoformat(self.tasks[task_id]["failed_at"])
                duration = (failed - started).total_seconds()
                self.tasks[task_id]["duration"] = duration
        
        # Update stats
        self.stats["tasks_failed"] += 1
        
        # Send to external system
        if self.config["enable_external_reporting"]:
            await self._send_to_external_system("task_failed", {
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
                "error": error
            })
    
    async def on_task_cancelled(self, task_id: str, reason: str):
        """
        Handle task cancelled event.
        
        Args:
            task_id: Task ID
            reason: Cancellation reason
        """
        logger.info(f"Task {task_id} cancelled: {reason}")
        
        # Update task
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "cancelled"
            self.tasks[task_id]["cancelled_at"] = datetime.now().isoformat()
            self.tasks[task_id]["cancel_reason"] = reason
        
        # Update stats
        self.stats["tasks_cancelled"] += 1
        
        # Send to external system
        if self.config["enable_external_reporting"]:
            await self._send_to_external_system("task_cancelled", {
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
                "reason": reason
            })
    
    def get_task_summary(self) -> Dict[str, Any]:
        """
        Get a summary of task data.
        
        Returns:
            Dictionary with task summary
        """
        # Calculate task counts by status
        status_counts = {}
        
        for task in self.tasks.values():
            status = task["status"]
            
            if status not in status_counts:
                status_counts[status] = 0
                
            status_counts[status] += 1
        
        # Calculate average duration for completed tasks
        completed_tasks = [t for t in self.tasks.values() if "duration" in t]
        avg_duration = sum(t["duration"] for t in completed_tasks) / len(completed_tasks) if completed_tasks else 0
        
        return {
            "total_tasks": len(self.tasks),
            "status_counts": status_counts,
            "stats": self.stats,
            "avg_duration": avg_duration
        }