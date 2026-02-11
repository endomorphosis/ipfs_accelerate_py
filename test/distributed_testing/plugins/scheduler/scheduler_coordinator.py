#!/usr/bin/env python3
"""
Scheduler Coordination Utility for Distributed Testing Framework

This module provides utilities for integrating scheduler plugins with the
distributed testing framework coordinator.
"""

import anyio
import logging
import importlib
from typing import Dict, List, Any, Optional, Type

from ...plugin_architecture import Plugin, PluginType, HookType
from .scheduler_plugin_interface import SchedulerPluginInterface, SchedulingStrategy
from .scheduler_plugin_registry import SchedulerPluginRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchedulerCoordinator:
    """
    Utility class for integrating scheduler plugins with the distributed testing framework.
    
    This class bridges between the coordinator, the scheduler plugin registry,
    and the TaskScheduler of the distributed testing framework.
    """
    
    def __init__(self, coordinator: Any, plugin_dirs: List[str] = None):
        """
        Initialize the scheduler coordinator.
        
        Args:
            coordinator: Reference to the coordinator instance
            plugin_dirs: List of directories to search for scheduler plugins
        """
        self.coordinator = coordinator
        self.registry = SchedulerPluginRegistry(plugin_dirs)
        
        # Store reference to the original scheduler
        self.original_scheduler = None
        if hasattr(coordinator, "task_scheduler"):
            self.original_scheduler = coordinator.task_scheduler
        
        # Integration setup flag
        self.is_integrated = False
        
        # Active plugin name
        self.active_plugin_name = None
        
        logger.info("SchedulerCoordinator initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the scheduler coordinator.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        # Discover available scheduler plugins
        discovered_plugins = await self.registry.discover_plugins()
        
        logger.info(f"Discovered {len(discovered_plugins)} scheduler plugins: {', '.join(discovered_plugins)}")
        
        return True
    
    async def activate_scheduler(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """
        Activate a scheduler plugin.
        
        Args:
            plugin_name: Name of the plugin to activate
            config: Configuration for the plugin
            
        Returns:
            bool: True if plugin was activated successfully, False otherwise
        """
        # Load the plugin if not already loaded
        if plugin_name not in self.registry.plugin_instances:
            success = await self.registry.load_plugin(plugin_name, config)
            if not success:
                logger.error(f"Failed to load scheduler plugin '{plugin_name}'")
                return False
        
        # Initialize the plugin with the coordinator
        success = await self.registry.initialize_plugin(plugin_name, self.coordinator, config)
        if not success:
            logger.error(f"Failed to initialize scheduler plugin '{plugin_name}'")
            return False
        
        # Set as active plugin
        success = self.registry.set_active_plugin(plugin_name)
        if not success:
            logger.error(f"Failed to set '{plugin_name}' as active scheduler plugin")
            return False
        
        # Store active plugin name
        self.active_plugin_name = plugin_name
        
        logger.info(f"Activated scheduler plugin '{plugin_name}'")
        
        # Integrate with coordinator if not already integrated
        if not self.is_integrated:
            await self.integrate_with_coordinator()
        
        return True
    
    async def deactivate_scheduler(self, plugin_name: str) -> bool:
        """
        Deactivate a scheduler plugin.
        
        Args:
            plugin_name: Name of the plugin to deactivate
            
        Returns:
            bool: True if plugin was deactivated successfully, False otherwise
        """
        # Check if plugin is active
        if self.active_plugin_name != plugin_name:
            logger.warning(f"Scheduler plugin '{plugin_name}' is not active")
            return False
        
        # Unload the plugin
        success = await self.registry.unload_plugin(plugin_name)
        if not success:
            logger.error(f"Failed to unload scheduler plugin '{plugin_name}'")
            return False
        
        # Clear active plugin name
        self.active_plugin_name = None
        
        # Restore original scheduler
        if self.is_integrated and self.original_scheduler:
            self.coordinator.task_scheduler = self.original_scheduler
            self.is_integrated = False
            logger.info("Restored original task scheduler")
        
        return True
    
    async def integrate_with_coordinator(self) -> bool:
        """
        Integrate the active scheduler plugin with the coordinator.
        
        Returns:
            bool: True if integration succeeded, False otherwise
        """
        # Check if we have an active plugin
        active_plugin = self.registry.get_active_plugin()
        if not active_plugin:
            logger.error("No active scheduler plugin to integrate with coordinator")
            return False
        
        # Store original scheduler reference if not already stored
        if not self.original_scheduler and hasattr(self.coordinator, "task_scheduler"):
            self.original_scheduler = self.coordinator.task_scheduler
        
        # Create a task scheduler wrapper that delegates to the active plugin
        scheduler_wrapper = SchedulerPluginWrapper(self.registry, self.original_scheduler)
        
        # Replace coordinator's task scheduler with the wrapper
        self.coordinator.task_scheduler = scheduler_wrapper
        
        # Set integration flag
        self.is_integrated = True
        
        logger.info(f"Integrated scheduler plugin '{self.active_plugin_name}' with coordinator")
        
        return True
    
    async def restore_original_scheduler(self) -> bool:
        """
        Restore the original task scheduler.
        
        Returns:
            bool: True if restoration succeeded, False otherwise
        """
        # Check if we're integrated and have the original scheduler
        if not self.is_integrated or not self.original_scheduler:
            logger.warning("Not integrated or no original scheduler to restore")
            return False
        
        # Restore original scheduler
        self.coordinator.task_scheduler = self.original_scheduler
        
        # Reset integration flag
        self.is_integrated = False
        
        logger.info("Restored original task scheduler")
        
        return True
    
    def get_available_plugins(self) -> List[str]:
        """
        Get list of available scheduler plugins.
        
        Returns:
            List[str]: List of available plugin names
        """
        return list(self.registry.plugins.keys())
    
    def get_active_plugin_name(self) -> Optional[str]:
        """
        Get the name of the active scheduler plugin.
        
        Returns:
            Optional[str]: Name of the active plugin or None if no active plugin
        """
        return self.active_plugin_name
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a scheduler plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Optional[Dict[str, Any]]: Plugin information or None if plugin not found
        """
        # Get the plugin instance
        plugin = self.registry.get_plugin(plugin_name)
        if not plugin:
            return None
        
        # Get plugin information
        info = {
            "name": plugin.get_name(),
            "version": plugin.get_version(),
            "description": plugin.get_description(),
            "strategies": [s.value for s in plugin.get_strategies()],
            "active_strategy": plugin.get_active_strategy().value,
            "configuration": plugin.get_configuration_schema(),
            "metrics": plugin.get_metrics()
        }
        
        return info
    
    async def set_strategy(self, strategy: str) -> bool:
        """
        Set the active scheduling strategy.
        
        Args:
            strategy: Name of the strategy to set
            
        Returns:
            bool: True if strategy was set successfully, False otherwise
        """
        # Check if we have an active plugin
        active_plugin = self.registry.get_active_plugin()
        if not active_plugin:
            logger.error("No active scheduler plugin to set strategy")
            return False
        
        # Convert string to enum
        try:
            strategy_enum = SchedulingStrategy(strategy)
        except ValueError:
            logger.error(f"Invalid scheduling strategy: {strategy}")
            return False
        
        # Set the strategy
        success = active_plugin.set_strategy(strategy_enum)
        
        if success:
            logger.info(f"Set active scheduling strategy to {strategy}")
        else:
            logger.error(f"Failed to set active scheduling strategy to {strategy}")
        
        return success
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the active scheduler plugin.
        
        Returns:
            Dict[str, Any]: Metrics from the active plugin or empty dict if no active plugin
        """
        # Check if we have an active plugin
        active_plugin = self.registry.get_active_plugin()
        if not active_plugin:
            return {}
        
        # Get metrics from the active plugin
        return active_plugin.get_metrics()


class SchedulerPluginWrapper:
    """
    Wrapper for scheduler plugins that implements the TaskScheduler interface.
    
    This class wraps a scheduler plugin to make it compatible with the
    TaskScheduler interface expected by the coordinator.
    """
    
    def __init__(self, registry: SchedulerPluginRegistry, original_scheduler: Any = None):
        """
        Initialize the scheduler plugin wrapper.
        
        Args:
            registry: Reference to the scheduler plugin registry
            original_scheduler: Reference to the original task scheduler
        """
        self.registry = registry
        self.original_scheduler = original_scheduler
        
        # Default configuration
        self.config = {
            "max_tasks_per_worker": 5,
            "prioritize_hardware_match": True,
            "load_balance": True,
            "consider_worker_performance": True,
            "enable_task_affinity": True,
            "enable_worker_specialization": True,
            "enable_predictive_scheduling": True
        }
        
        # Copy configuration from original scheduler if available
        if original_scheduler:
            for key in self.config:
                if hasattr(original_scheduler, key):
                    self.config[key] = getattr(original_scheduler, key)
        
        logger.info("SchedulerPluginWrapper initialized")
    
    async def schedule_pending_tasks(self) -> int:
        """
        Schedule pending tasks to available workers.
        
        This method is called by the coordinator to schedule pending tasks.
        It delegates to the active scheduler plugin.
        
        Returns:
            int: Number of tasks scheduled
        """
        # Get active plugin
        active_plugin = self.registry.get_active_plugin()
        
        # If no active plugin, delegate to original scheduler
        if not active_plugin:
            if self.original_scheduler and hasattr(self.original_scheduler, "schedule_pending_tasks"):
                return await self.original_scheduler.schedule_pending_tasks()
            return 0
        
        # Get coordinator reference from plugin
        coordinator = active_plugin.coordinator
        
        # If no coordinator, return 0
        if not coordinator:
            logger.error("No coordinator reference in active plugin")
            return 0
        
        # Get pending tasks
        pending_tasks = coordinator.pending_tasks if hasattr(coordinator, "pending_tasks") else []
        
        # If no pending tasks, return 0
        if not pending_tasks:
            return 0
        
        # Get available workers
        available_workers = {}
        for worker_id, worker in coordinator.workers.items() if hasattr(coordinator, "workers") else {}:
            if worker.get("status") == "active" and worker_id in coordinator.worker_connections:
                available_workers[worker_id] = worker
        
        # If no available workers, return 0
        if not available_workers:
            return 0
        
        # Get current worker task load
        worker_task_count = {}
        for task_id, worker_id in coordinator.running_tasks.items() if hasattr(coordinator, "running_tasks") else {}:
            worker_task_count[worker_id] = worker_task_count.get(worker_id, 0) + 1
        
        # Filter workers with capacity
        available_workers_with_capacity = {}
        for worker_id, worker in available_workers.items():
            if worker_task_count.get(worker_id, 0) < self.config["max_tasks_per_worker"]:
                available_workers_with_capacity[worker_id] = worker
        
        # If no workers with capacity, return 0
        if not available_workers_with_capacity:
            return 0
        
        # Count tasks scheduled
        tasks_scheduled = 0
        
        # Process each pending task
        for task_id in list(pending_tasks):
            # Get task data
            task_data = coordinator.tasks.get(task_id, {})
            
            # Schedule task using active plugin
            worker_id = await self.registry.schedule_task(
                task_id,
                task_data,
                available_workers_with_capacity,
                worker_task_count
            )
            
            # If a worker was selected, assign the task
            if worker_id:
                # Assign task to worker
                if hasattr(coordinator, "_assign_task_to_worker"):
                    await coordinator._assign_task_to_worker(task_id, worker_id)
                
                # Remove from pending tasks
                coordinator.pending_tasks.remove(task_id)
                
                # Add to running tasks
                coordinator.running_tasks[task_id] = worker_id
                
                # Update worker task count
                worker_task_count[worker_id] = worker_task_count.get(worker_id, 0) + 1
                
                # Check if worker is at capacity
                if worker_task_count[worker_id] >= self.config["max_tasks_per_worker"]:
                    # Remove worker from available workers
                    if worker_id in available_workers_with_capacity:
                        del available_workers_with_capacity[worker_id]
                
                # Update worker status if it reaches capacity
                if worker_task_count[worker_id] >= self.config["max_tasks_per_worker"]:
                    coordinator.workers[worker_id]["status"] = "busy"
                
                tasks_scheduled += 1
            
            # Stop if no more workers available
            if not available_workers_with_capacity:
                break
        
        return tasks_scheduled