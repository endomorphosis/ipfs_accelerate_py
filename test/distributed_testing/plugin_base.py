#!/usr/bin/env python3
"""
Plugin Base for Distributed Testing Framework

This module provides the base class for all plugins in the distributed testing framework.
Plugins extend the functionality of the framework without modifying its core code.
"""

import time
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PluginBase:
    """Base class for all distributed testing framework plugins"""
    
    def __init__(self, plugin_id: str):
        """
        Initialize plugin base
        
        Args:
            plugin_id: Unique identifier for this plugin
        """
        self.plugin_id = plugin_id
        self.created_at = time.time()
        self.logger = logging.getLogger(f"plugin.{plugin_id}")
        
        # Plugin capabilities
        self.capabilities = []
        
        self.logger.info(f"Plugin {plugin_id} created")
    
    async def initialize(self) -> bool:
        """
        Initialize plugin (to be implemented by subclasses)
        
        Returns:
            Success status
        """
        self.logger.info(f"Plugin {self.plugin_id} initialize called")
        return True
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task (to be implemented by subclasses)
        
        Args:
            task_data: Task definition and parameters
            
        Returns:
            Task execution results
        """
        self.logger.warning(f"Plugin {self.plugin_id} execute_task not implemented")
        return {
            "success": False,
            "error": "Not implemented",
            "plugin_id": self.plugin_id
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get plugin status
        
        Returns:
            Dictionary with status information
        """
        return {
            "plugin_id": self.plugin_id,
            "created_at": self.created_at,
            "uptime": time.time() - self.created_at,
            "capabilities": self.capabilities
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get plugin metrics (to be implemented by subclasses)
        
        Returns:
            Dictionary with metrics information
        """
        return {
            "plugin_id": self.plugin_id,
            "metrics_available": False
        }
    
    async def shutdown(self) -> Dict[str, Any]:
        """
        Shutdown plugin and clean up resources (to be implemented by subclasses)
        
        Returns:
            Shutdown status
        """
        self.logger.info(f"Plugin {self.plugin_id} shutdown called")
        return {
            "success": True,
            "plugin_id": self.plugin_id
        }