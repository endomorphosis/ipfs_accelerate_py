#!/usr/bin/env python3
"""
Plugin Architecture for Distributed Testing Framework

This module implements the plugin architecture that enables extensibility of the 
distributed testing framework without modifying its core functionality.

Usage:
    Import this module to add plugin capabilities to the distributed testing framework.
"""

import importlib
import inspect
import logging
import os
import pkgutil
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple, Type, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PluginType(Enum):
    """Types of plugins supported by the framework."""
    
    SCHEDULER = "scheduler"
    TASK_EXECUTOR = "task_executor"
    REPORTER = "reporter"
    NOTIFICATION = "notification"
    MONITORING = "monitoring"
    INTEGRATION = "integration"
    SECURITY = "security"
    CUSTOM = "custom"

class HookType(Enum):
    """Hook points where plugins can be invoked."""
    
    # Coordinator hooks
    COORDINATOR_STARTUP = "coordinator_startup"
    COORDINATOR_SHUTDOWN = "coordinator_shutdown"
    
    # Task hooks
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    
    # Worker hooks
    WORKER_REGISTERED = "worker_registered"
    WORKER_DISCONNECTED = "worker_disconnected"
    WORKER_FAILED = "worker_failed"
    WORKER_RECOVERED = "worker_recovered"
    
    # Recovery hooks
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    RECOVERY_FAILED = "recovery_failed"
    
    # Custom hooks
    CUSTOM = "custom"

class Plugin(ABC):
    """Base class for all plugins."""
    
    def __init__(self, name: str, version: str, plugin_type: PluginType):
        """
        Initialize plugin.
        
        Args:
            name: Plugin name
            version: Plugin version
            plugin_type: Type of plugin
        """
        self.name = name
        self.version = version
        self.plugin_type = plugin_type
        self.enabled = True
        self.config = {}
        self.hooks = {}
        
        logger.info(f"Plugin {name} v{version} ({plugin_type.value}) initialized")
    
    @abstractmethod
    async def initialize(self, coordinator) -> bool:
        """
        Initialize the plugin with reference to the coordinator.
        
        Args:
            coordinator: Reference to the coordinator instance
            
        Returns:
            True if initialization succeeded
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin.
        
        Returns:
            True if shutdown succeeded
        """
        pass
    
    def register_hook(self, hook_type: HookType, callback: Callable) -> bool:
        """
        Register a hook callback.
        
        Args:
            hook_type: Type of hook
            callback: Callback function to invoke
            
        Returns:
            True if registration succeeded
        """
        if hook_type not in self.hooks:
            self.hooks[hook_type] = []
            
        self.hooks[hook_type].append(callback)
        
        logger.debug(f"Plugin {self.name} registered hook {hook_type.value}")
        return True
    
    def unregister_hook(self, hook_type: HookType, callback: Callable) -> bool:
        """
        Unregister a hook callback.
        
        Args:
            hook_type: Type of hook
            callback: Callback function to remove
            
        Returns:
            True if unregistration succeeded
        """
        if hook_type not in self.hooks:
            return False
            
        if callback not in self.hooks[hook_type]:
            return False
            
        self.hooks[hook_type].remove(callback)
        
        logger.debug(f"Plugin {self.name} unregistered hook {hook_type.value}")
        return True
    
    async def invoke_hook(self, hook_type: HookType, *args, **kwargs) -> List[Any]:
        """
        Invoke all callbacks registered for a hook.
        
        Args:
            hook_type: Type of hook
            *args: Positional arguments to pass to callbacks
            **kwargs: Keyword arguments to pass to callbacks
            
        Returns:
            List of results from callbacks
        """
        if not self.enabled:
            return []
            
        if hook_type not in self.hooks:
            return []
            
        results = []
        
        for callback in self.hooks[hook_type]:
            try:
                result = callback(*args, **kwargs)
                
                # Handle coroutines
                if inspect.iscoroutine(result):
                    result = await result
                    
                results.append(result)
            except Exception as e:
                logger.error(f"Error invoking hook {hook_type.value} in plugin {self.name}: {str(e)}")
        
        return results
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the plugin.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration succeeded
        """
        self.config.update(config)
        
        logger.debug(f"Plugin {self.name} configured with {len(config)} settings")
        return True
    
    def enable(self) -> bool:
        """
        Enable the plugin.
        
        Returns:
            True if enabled
        """
        self.enabled = True
        
        logger.info(f"Plugin {self.name} enabled")
        return True
    
    def disable(self) -> bool:
        """
        Disable the plugin.
        
        Returns:
            True if disabled
        """
        self.enabled = False
        
        logger.info(f"Plugin {self.name} disabled")
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin information.
        
        Returns:
            Dictionary with plugin information
        """
        return {
            "name": self.name,
            "version": self.version,
            "type": self.plugin_type.value,
            "enabled": self.enabled,
            "hooks": [hook.value for hook in self.hooks.keys()],
            "config": self.config
        }
    
    @property
    def id(self) -> str:
        """
        Get plugin ID.
        
        Returns:
            Plugin ID as string
        """
        return f"{self.name}-{self.version}"


class PluginManager:
    """
    Manager for loading, configuring, and invoking plugins.
    
    This class manages the lifecycle of plugins, including discovery, loading,
    configuration, and invocation of plugin hooks.
    """
    
    def __init__(self, coordinator, plugin_dirs: List[str] = None):
        """
        Initialize the plugin manager.
        
        Args:
            coordinator: Reference to the coordinator instance
            plugin_dirs: List of directories to search for plugins
        """
        self.coordinator = coordinator
        self.plugin_dirs = plugin_dirs or ["plugins"]
        
        # Loaded plugins by ID
        self.plugins: Dict[str, Plugin] = {}
        
        # Plugins by type
        self.plugins_by_type: Dict[PluginType, Dict[str, Plugin]] = {
            plugin_type: {} for plugin_type in PluginType
        }
        
        # Hooks registry
        self.hooks: Dict[HookType, List[Tuple[str, Callable]]] = {
            hook_type: [] for hook_type in HookType
        }
        
        logger.info(f"PluginManager initialized with {len(self.plugin_dirs)} plugin directories")
    
    async def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in plugin directories.
        
        Returns:
            List of discovered plugin module names
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
                # Skip packages
                if is_pkg:
                    continue
                    
                # Check if module is a plugin
                try:
                    module = importlib.import_module(name)
                    
                    # Look for Plugin subclasses
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        if (inspect.isclass(attr) and 
                            issubclass(attr, Plugin) and 
                            attr is not Plugin):
                            
                            discovered_plugins.append(name)
                            logger.info(f"Discovered plugin module: {name}")
                            break
                            
                except Exception as e:
                    logger.error(f"Error importing module {name}: {str(e)}")
        
        return discovered_plugins
    
    async def load_plugin(self, module_name: str) -> Optional[str]:
        """
        Load a plugin by module name.
        
        Args:
            module_name: Name of the module containing the plugin
            
        Returns:
            Plugin ID if loaded successfully, None otherwise
        """
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Find the Plugin subclass
            plugin_class = None
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if (inspect.isclass(attr) and 
                    issubclass(attr, Plugin) and 
                    attr is not Plugin):
                    
                    plugin_class = attr
                    break
            
            if not plugin_class:
                logger.error(f"No Plugin subclass found in module {module_name}")
                return None
                
            # Create plugin instance
            plugin = plugin_class()
            
            # Initialize plugin
            success = await plugin.initialize(self.coordinator)
            
            if not success:
                logger.error(f"Failed to initialize plugin {plugin.name}")
                return None
                
            # Register plugin
            plugin_id = plugin.id
            
            self.plugins[plugin_id] = plugin
            self.plugins_by_type[plugin.plugin_type][plugin_id] = plugin
            
            # Register hooks
            for hook_type, callbacks in plugin.hooks.items():
                for callback in callbacks:
                    self.hooks[hook_type].append((plugin_id, callback))
            
            logger.info(f"Loaded plugin {plugin.name} v{plugin.version} ({plugin.plugin_type.value})")
            
            return plugin_id
            
        except Exception as e:
            logger.error(f"Error loading plugin {module_name}: {str(e)}")
            return None
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_id: ID of the plugin to unload
            
        Returns:
            True if unloaded successfully
        """
        if plugin_id not in self.plugins:
            logger.warning(f"Plugin {plugin_id} not found")
            return False
            
        plugin = self.plugins[plugin_id]
        
        try:
            # Shutdown plugin
            await plugin.shutdown()
            
            # Unregister hooks
            for hook_type in HookType:
                self.hooks[hook_type] = [(pid, cb) for pid, cb in self.hooks[hook_type] if pid != plugin_id]
            
            # Remove from registries
            del self.plugins_by_type[plugin.plugin_type][plugin_id]
            del self.plugins[plugin_id]
            
            logger.info(f"Unloaded plugin {plugin.name} v{plugin.version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {str(e)}")
            return False
    
    async def invoke_hook(self, hook_type: HookType, *args, **kwargs) -> List[Any]:
        """
        Invoke all callbacks registered for a hook.
        
        Args:
            hook_type: Type of hook
            *args: Positional arguments to pass to callbacks
            **kwargs: Keyword arguments to pass to callbacks
            
        Returns:
            List of results from callbacks
        """
        if hook_type not in self.hooks:
            return []
            
        results = []
        
        for plugin_id, callback in self.hooks[hook_type]:
            if plugin_id not in self.plugins:
                continue
                
            plugin = self.plugins[plugin_id]
            
            if not plugin.enabled:
                continue
                
            try:
                result = callback(*args, **kwargs)
                
                # Handle coroutines
                if inspect.iscoroutine(result):
                    result = await result
                    
                results.append((plugin_id, result))
            except Exception as e:
                logger.error(f"Error invoking hook {hook_type.value} in plugin {plugin.name}: {str(e)}")
        
        return results
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """
        Get a plugin by ID.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, Plugin]:
        """
        Get all plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to get
            
        Returns:
            Dictionary of plugin ID to plugin instance
        """
        return self.plugins_by_type.get(plugin_type, {})
    
    def get_all_plugins(self) -> Dict[str, Plugin]:
        """
        Get all loaded plugins.
        
        Returns:
            Dictionary of plugin ID to plugin instance
        """
        return self.plugins.copy()
    
    async def configure_plugin(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """
        Configure a plugin.
        
        Args:
            plugin_id: Plugin ID
            config: Configuration dictionary
            
        Returns:
            True if configuration succeeded
        """
        if plugin_id not in self.plugins:
            logger.warning(f"Plugin {plugin_id} not found")
            return False
            
        plugin = self.plugins[plugin_id]
        
        success = plugin.configure(config)
        
        if success:
            logger.info(f"Configured plugin {plugin.name} with {len(config)} settings")
        else:
            logger.error(f"Failed to configure plugin {plugin.name}")
            
        return success
    
    async def enable_plugin(self, plugin_id: str) -> bool:
        """
        Enable a plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            True if enabled
        """
        if plugin_id not in self.plugins:
            logger.warning(f"Plugin {plugin_id} not found")
            return False
            
        plugin = self.plugins[plugin_id]
        
        success = plugin.enable()
        
        if success:
            logger.info(f"Enabled plugin {plugin.name}")
        else:
            logger.error(f"Failed to enable plugin {plugin.name}")
            
        return success
    
    async def disable_plugin(self, plugin_id: str) -> bool:
        """
        Disable a plugin.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            True if disabled
        """
        if plugin_id not in self.plugins:
            logger.warning(f"Plugin {plugin_id} not found")
            return False
            
        plugin = self.plugins[plugin_id]
        
        success = plugin.disable()
        
        if success:
            logger.info(f"Disabled plugin {plugin.name}")
        else:
            logger.error(f"Failed to disable plugin {plugin.name}")
            
        return success
    
    async def shutdown(self):
        """Shutdown all plugins."""
        for plugin_id, plugin in list(self.plugins.items()):
            try:
                await plugin.shutdown()
                logger.info(f"Shutdown plugin {plugin.name}")
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin.name}: {str(e)}")
        
        # Clear registries
        self.plugins.clear()
        
        for plugin_type in PluginType:
            self.plugins_by_type[plugin_type].clear()
            
        for hook_type in HookType:
            self.hooks[hook_type].clear()
            
        logger.info("PluginManager shutdown complete")