#!/usr/bin/env python3
"""
Scheduler Plugin Registry for Distributed Testing Framework

This module provides a registry for scheduler plugins that implements
dynamic discovery, loading, and management of scheduler plugins.
"""

import importlib
import importlib.util
import inspect
import logging
import os
import pkgutil
import sys
from typing import Dict, List, Any, Optional, Type, Tuple, Set

from .scheduler_plugin_interface import SchedulerPluginInterface, SchedulingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchedulerPluginRegistry:
    """
    Registry for scheduler plugins.
    
    This class handles the discovery, registration, and management of
    scheduler plugins for the distributed testing framework.
    """
    
    def __init__(self, plugin_dirs: List[str] = None):
        """
        Initialize the scheduler plugin registry.
        
        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self.plugin_dirs = plugin_dirs or [
            "distributed_testing/plugins/scheduler",
            "plugins/scheduler",
            "scheduler"
        ]
        
        # Map of plugin name to plugin class
        self.plugins: Dict[str, Type[SchedulerPluginInterface]] = {}
        
        # Map of plugin name to plugin instance
        self.plugin_instances: Dict[str, SchedulerPluginInterface] = {}
        
        # Map of strategy to list of plugin names that implement it
        self.strategy_plugins: Dict[SchedulingStrategy, List[str]] = {
            strategy: [] for strategy in SchedulingStrategy
        }
        
        # Active plugin name
        self.active_plugin: Optional[str] = None
        
        logger.info(f"SchedulerPluginRegistry initialized with {len(self.plugin_dirs)} plugin directories")
    
    async def discover_plugins(self) -> List[str]:
        """
        Discover scheduler plugins in plugin directories.
        
        Returns:
            List[str]: List of discovered plugin names
        """
        discovered_plugins = []
        
        for plugin_dir in self.plugin_dirs:
            # Ensure plugin directory exists
            if not os.path.isdir(plugin_dir):
                logger.warning(f"Plugin directory {plugin_dir} does not exist")
                continue
                
            # Add to Python path if not already there
            if plugin_dir not in sys.path:
                sys.path.append(plugin_dir)
                
            # Discover modules in directory
            for _, name, is_pkg in pkgutil.iter_modules([plugin_dir]):
                # Skip packages and special names
                if is_pkg or name.startswith('_'):
                    continue
                    
                # Check if module is a scheduler plugin
                try:
                    module = importlib.import_module(f"{os.path.basename(plugin_dir)}.{name}")
                    
                    # Look for SchedulerPluginInterface implementations
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        if (inspect.isclass(attr) and 
                            issubclass(attr, SchedulerPluginInterface) and 
                            attr is not SchedulerPluginInterface):
                            
                            # Register the plugin
                            plugin_name = attr().get_name()
                            self.plugins[plugin_name] = attr
                            
                            # Register strategies
                            strategies = attr().get_strategies()
                            for strategy in strategies:
                                self.strategy_plugins[strategy].append(plugin_name)
                                
                            discovered_plugins.append(plugin_name)
                            logger.info(f"Discovered scheduler plugin: {plugin_name}")
                            break
                            
                except Exception as e:
                    logger.error(f"Error importing scheduler plugin module {name}: {str(e)}")
        
        return discovered_plugins
    
    async def load_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """
        Load and initialize a scheduler plugin.
        
        Args:
            plugin_name: Name of the plugin to load
            config: Configuration for the plugin
            
        Returns:
            bool: True if plugin was loaded successfully, False otherwise
        """
        if plugin_name not in self.plugins:
            logger.error(f"Scheduler plugin '{plugin_name}' not found")
            return False
            
        try:
            # Create plugin instance
            plugin_class = self.plugins[plugin_name]
            plugin = plugin_class()
            
            # Configure plugin if config provided
            if config:
                plugin.configure(config)
                
            # Store plugin instance
            self.plugin_instances[plugin_name] = plugin
            
            logger.info(f"Loaded scheduler plugin '{plugin_name}' v{plugin.get_version()}")
            
            # Set as active plugin if no active plugin
            if self.active_plugin is None:
                self.active_plugin = plugin_name
                logger.info(f"Set '{plugin_name}' as active scheduler plugin")
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading scheduler plugin '{plugin_name}': {str(e)}")
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a scheduler plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            bool: True if plugin was unloaded successfully, False otherwise
        """
        if plugin_name not in self.plugin_instances:
            logger.warning(f"Scheduler plugin '{plugin_name}' not loaded")
            return False
            
        # Get plugin instance
        plugin = self.plugin_instances[plugin_name]
        
        try:
            # Shutdown plugin
            await plugin.shutdown()
            
            # Remove from instances
            del self.plugin_instances[plugin_name]
            
            # Update active plugin if needed
            if self.active_plugin == plugin_name:
                self.active_plugin = next(iter(self.plugin_instances)) if self.plugin_instances else None
                
            logger.info(f"Unloaded scheduler plugin '{plugin_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Error unloading scheduler plugin '{plugin_name}': {str(e)}")
            return False
    
    async def initialize_plugin(self, plugin_name: str, coordinator: Any, config: Dict[str, Any] = None) -> bool:
        """
        Initialize a loaded scheduler plugin with the coordinator.
        
        Args:
            plugin_name: Name of the plugin to initialize
            coordinator: Coordinator instance
            config: Configuration for the plugin
            
        Returns:
            bool: True if plugin was initialized successfully, False otherwise
        """
        if plugin_name not in self.plugin_instances:
            logger.error(f"Scheduler plugin '{plugin_name}' not loaded")
            return False
            
        # Get plugin instance
        plugin = self.plugin_instances[plugin_name]
        
        try:
            # Initialize plugin
            success = await plugin.initialize(coordinator, config)
            
            if success:
                logger.info(f"Initialized scheduler plugin '{plugin_name}' with coordinator")
            else:
                logger.error(f"Failed to initialize scheduler plugin '{plugin_name}'")
                
            return success
            
        except Exception as e:
            logger.error(f"Error initializing scheduler plugin '{plugin_name}': {str(e)}")
            return False
    
    def set_active_plugin(self, plugin_name: str) -> bool:
        """
        Set the active scheduler plugin.
        
        Args:
            plugin_name: Name of the plugin to set as active
            
        Returns:
            bool: True if plugin was set as active, False otherwise
        """
        if plugin_name not in self.plugin_instances:
            logger.error(f"Scheduler plugin '{plugin_name}' not loaded")
            return False
            
        self.active_plugin = plugin_name
        logger.info(f"Set '{plugin_name}' as active scheduler plugin")
        
        return True
    
    def get_active_plugin(self) -> Optional[SchedulerPluginInterface]:
        """
        Get the active scheduler plugin instance.
        
        Returns:
            Optional[SchedulerPluginInterface]: Active plugin instance or None
        """
        if self.active_plugin is None:
            return None
            
        return self.plugin_instances.get(self.active_plugin)
    
    def get_plugin(self, plugin_name: str) -> Optional[SchedulerPluginInterface]:
        """
        Get a scheduler plugin instance by name.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Optional[SchedulerPluginInterface]: Plugin instance or None
        """
        return self.plugin_instances.get(plugin_name)
    
    def get_plugins_for_strategy(self, strategy: SchedulingStrategy) -> List[str]:
        """
        Get list of plugin names that implement a specific strategy.
        
        Args:
            strategy: Scheduling strategy
            
        Returns:
            List[str]: List of plugin names
        """
        return self.strategy_plugins.get(strategy, [])
    
    def get_all_plugins(self) -> Dict[str, SchedulerPluginInterface]:
        """
        Get all loaded scheduler plugin instances.
        
        Returns:
            Dict[str, SchedulerPluginInterface]: Dictionary of plugin name to instance
        """
        return self.plugin_instances.copy()
    
    def get_registered_plugins(self) -> Dict[str, Type[SchedulerPluginInterface]]:
        """
        Get all registered scheduler plugin classes.
        
        Returns:
            Dict[str, Type[SchedulerPluginInterface]]: Dictionary of plugin name to class
        """
        return self.plugins.copy()
    
    async def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """
        Configure a scheduler plugin.
        
        Args:
            plugin_name: Name of the plugin
            config: Configuration dictionary
            
        Returns:
            bool: True if configuration succeeded, False otherwise
        """
        if plugin_name not in self.plugin_instances:
            logger.error(f"Scheduler plugin '{plugin_name}' not loaded")
            return False
            
        # Get plugin instance
        plugin = self.plugin_instances[plugin_name]
        
        # Configure plugin
        success = plugin.configure(config)
        
        if success:
            logger.info(f"Configured scheduler plugin '{plugin_name}'")
        else:
            logger.error(f"Failed to configure scheduler plugin '{plugin_name}'")
            
        return success
    
    async def schedule_task(self, task_id: str, task_data: Dict[str, Any],
                           available_workers: Dict[str, Dict[str, Any]],
                           worker_load: Dict[str, int]) -> Optional[str]:
        """
        Schedule a task using the active scheduler plugin.
        
        Args:
            task_id: ID of the task to schedule
            task_data: Task data including requirements and metadata
            available_workers: Dictionary of available worker IDs to worker data
            worker_load: Dictionary of worker IDs to current task counts
            
        Returns:
            Optional[str]: Selected worker ID or None if no suitable worker found
        """
        if self.active_plugin is None or self.active_plugin not in self.plugin_instances:
            logger.error("No active scheduler plugin to schedule task")
            return None
            
        # Get active plugin
        plugin = self.plugin_instances[self.active_plugin]
        
        try:
            # Schedule task using plugin
            worker_id = await plugin.schedule_task(task_id, task_data, available_workers, worker_load)
            
            if worker_id:
                logger.debug(f"Scheduled task {task_id} to worker {worker_id} using {self.active_plugin} plugin")
            else:
                logger.debug(f"No suitable worker found for task {task_id} using {self.active_plugin} plugin")
                
            return worker_id
            
        except Exception as e:
            logger.error(f"Error scheduling task {task_id} with plugin {self.active_plugin}: {str(e)}")
            return None
    
    async def update_task_status(self, task_id: str, status: str,
                                worker_id: Optional[str],
                                execution_time: Optional[float] = None,
                                result: Any = None) -> None:
        """
        Update the status of a task in all loaded plugins.
        
        Args:
            task_id: ID of the task
            status: New status of the task
            worker_id: ID of the worker that processed the task
            execution_time: Execution time in seconds
            result: Task result or error information
        """
        for plugin_name, plugin in self.plugin_instances.items():
            try:
                await plugin.update_task_status(task_id, status, worker_id, execution_time, result)
            except Exception as e:
                logger.error(f"Error updating task status in plugin {plugin_name}: {str(e)}")
    
    async def update_worker_status(self, worker_id: str, status: str,
                                  capabilities: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the status of a worker in all loaded plugins.
        
        Args:
            worker_id: ID of the worker
            status: New status of the worker
            capabilities: Worker capabilities
        """
        for plugin_name, plugin in self.plugin_instances.items():
            try:
                await plugin.update_worker_status(worker_id, status, capabilities)
            except Exception as e:
                logger.error(f"Error updating worker status in plugin {plugin_name}: {str(e)}")