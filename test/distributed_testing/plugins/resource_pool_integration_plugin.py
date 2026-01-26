#!/usr/bin/env python3
"""
Resource Pool Integration Plugin for Distributed Testing Framework

This plugin provides integration between the WebGPU/WebNN Resource Pool
and the Distributed Testing Framework, enabling efficient management of
browser-based testing resources with fault tolerance capabilities.
"""

import anyio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

# Import plugin base class
from plugin_architecture import Plugin, PluginType, HookType
from resource_pool_bridge import ResourcePoolBridgeIntegration
from resource_pool_bridge_recovery import ResourcePoolRecoveryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResourcePoolIntegrationPlugin(Plugin):
    """
    Resource Pool Integration Plugin for the Distributed Testing Framework.
    
    This plugin integrates the WebGPU/WebNN Resource Pool with the Distributed
    Testing Framework, providing fault-tolerant management of browser-based
    testing resources with automatic recovery capabilities.
    """
    
    def __init__(self):
        """Initialize the plugin."""
        super().__init__(
            name="ResourcePoolIntegration",
            version="1.0.0",
            plugin_type=PluginType.INTEGRATION
        )
        
        # Resource pool integration
        self.resource_pool = None
        self.recovery_manager = None
        
        # Resource tracking
        self.active_resources = {}
        self.resource_metrics = {}
        self.performance_history = {}
        
        # Default configuration
        self.config = {
            "max_connections": 4,
            "browser_preferences": {
                "audio": "firefox",
                "vision": "chrome", 
                "text_embedding": "edge"
            },
            "adaptive_scaling": True,
            "enable_fault_tolerance": True,
            "recovery_strategy": "progressive",
            "state_sync_interval": 5,
            "redundancy_factor": 2,
            "metrics_collection_interval": 30,
            "auto_optimization": True
        }
        
        # Register hooks
        self.register_hook(HookType.COORDINATOR_STARTUP, self.on_coordinator_startup)
        self.register_hook(HookType.COORDINATOR_SHUTDOWN, self.on_coordinator_shutdown)
        self.register_hook(HookType.TASK_CREATED, self.on_task_created)
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        self.register_hook(HookType.TASK_FAILED, self.on_task_failed)
        self.register_hook(HookType.RECOVERY_STARTED, self.on_recovery_started)
        self.register_hook(HookType.RECOVERY_COMPLETED, self.on_recovery_completed)
        
        logger.info("ResourcePoolIntegrationPlugin initialized")
    
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
        
        # Initialize resource pool
        await self._initialize_resource_pool()
        
        # Start metrics collection task
        self.metrics_task = # TODO: Replace with task group - asyncio.create_task(self._collect_metrics())
        
        logger.info("ResourcePoolIntegrationPlugin initialized with coordinator")
        return True
    
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin.
        
        Returns:
            True if shutdown succeeded
        """
        # Cancel metrics task
        if hasattr(self, "metrics_task") and self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except anyio.get_cancelled_exc_class():
                pass
        
        # Shutdown resource pool
        if self.resource_pool:
            await self._shutdown_resource_pool()
        
        logger.info("ResourcePoolIntegrationPlugin shutdown complete")
        return True
    
    async def _initialize_resource_pool(self):
        """Initialize the resource pool integration."""
        logger.info("Initializing Resource Pool integration")
        
        # Create resource pool integration
        self.resource_pool = ResourcePoolBridgeIntegration(
            max_connections=self.config["max_connections"],
            browser_preferences=self.config["browser_preferences"],
            adaptive_scaling=self.config["adaptive_scaling"],
            enable_fault_tolerance=self.config["enable_fault_tolerance"],
            recovery_strategy=self.config["recovery_strategy"],
            state_sync_interval=self.config["state_sync_interval"],
            redundancy_factor=self.config["redundancy_factor"]
        )
        
        # Initialize resource pool
        await self.resource_pool.initialize()
        
        # Initialize recovery manager if fault tolerance is enabled
        if self.config["enable_fault_tolerance"]:
            self.recovery_manager = ResourcePoolRecoveryManager(
                resource_pool=self.resource_pool,
                recovery_strategy=self.config["recovery_strategy"],
                coordinator=self.coordinator
            )
            await self.recovery_manager.initialize()
        
        logger.info("Resource Pool integration initialized")
    
    async def _shutdown_resource_pool(self):
        """Shutdown the resource pool integration."""
        logger.info("Shutting down Resource Pool integration")
        
        # Shutdown recovery manager
        if self.recovery_manager:
            await self.recovery_manager.shutdown()
        
        # Shutdown resource pool
        await self.resource_pool.shutdown()
        
        logger.info("Resource Pool integration shutdown complete")
    
    async def _collect_metrics(self):
        """Collect metrics from the resource pool."""
        while True:
            try:
                # Sleep for collection interval
                await anyio.sleep(self.config["metrics_collection_interval"])
                
                # Skip if no resource pool
                if not self.resource_pool:
                    continue
                
                logger.debug("Collecting Resource Pool metrics")
                
                # Collect metrics
                metrics = await self.resource_pool.get_metrics()
                
                # Store metrics
                timestamp = datetime.now().isoformat()
                self.resource_metrics[timestamp] = metrics
                
                # Clean up old metrics (keep only the last 100)
                if len(self.resource_metrics) > 100:
                    oldest_key = min(self.resource_metrics.keys())
                    del self.resource_metrics[oldest_key]
                
                # Update performance history
                await self._update_performance_history()
                
                # Optimize resource allocation if auto-optimization is enabled
                if self.config["auto_optimization"]:
                    await self._optimize_resource_allocation()
                
                logger.debug("Resource Pool metrics collected")
                
            except anyio.get_cancelled_exc_class():
                logger.info("Metrics collection task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {str(e)}")
    
    async def _update_performance_history(self):
        """Update the performance history based on collected metrics."""
        # Skip if no metrics
        if not self.resource_metrics:
            return
        
        # Get latest metrics
        latest_timestamp = max(self.resource_metrics.keys())
        latest_metrics = self.resource_metrics[latest_timestamp]
        
        # Update performance history by browser type
        for browser_type, browser_metrics in latest_metrics.get("browsers", {}).items():
            if browser_type not in self.performance_history:
                self.performance_history[browser_type] = []
            
            self.performance_history[browser_type].append({
                "timestamp": latest_timestamp,
                "metrics": browser_metrics
            })
            
            # Keep only the last 20 entries
            if len(self.performance_history[browser_type]) > 20:
                self.performance_history[browser_type].pop(0)
        
        # Update performance history by model type
        for model_type, model_metrics in latest_metrics.get("models", {}).items():
            if model_type not in self.performance_history:
                self.performance_history[model_type] = []
            
            self.performance_history[model_type].append({
                "timestamp": latest_timestamp,
                "metrics": model_metrics
            })
            
            # Keep only the last 20 entries
            if len(self.performance_history[model_type]) > 20:
                self.performance_history[model_type].pop(0)
    
    async def _optimize_resource_allocation(self):
        """Optimize resource allocation based on performance history."""
        # Skip if no performance history
        if not self.performance_history:
            return
        
        logger.debug("Optimizing resource allocation")
        
        # Analyze performance history
        recommendations = await self.resource_pool.analyze_performance_trends(
            self.performance_history
        )
        
        # Apply recommendations
        if recommendations:
            await self.resource_pool.apply_performance_optimizations(recommendations)
            logger.info(f"Applied {len(recommendations)} resource allocation optimizations")
    
    async def allocate_model_for_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate a model for a task from the resource pool.
        
        Args:
            task_id: Task ID
            task_data: Task data with model requirements
            
        Returns:
            Dictionary with allocated model information
        """
        if not self.resource_pool:
            logger.warning(f"Resource pool not initialized, cannot allocate model for task {task_id}")
            return None
        
        # Extract model requirements from task data
        model_type = task_data.get("model_type", "text_embedding")
        model_name = task_data.get("model_name", "bert-base-uncased")
        hardware_preferences = task_data.get("hardware_preferences", {
            "priority_list": ["webgpu", "cpu"]
        })
        
        # Configure fault tolerance options
        fault_tolerance = {
            "recovery_timeout": 30,
            "state_persistence": True,
            "failover_strategy": "immediate"
        }
        
        # Update with task-specific fault tolerance settings if provided
        if "fault_tolerance" in task_data:
            fault_tolerance.update(task_data["fault_tolerance"])
        
        logger.info(f"Allocating {model_type} model {model_name} for task {task_id}")
        
        try:
            # Get model from resource pool
            model = await self.resource_pool.get_model(
                model_type=model_type,
                model_name=model_name,
                hardware_preferences=hardware_preferences,
                fault_tolerance=fault_tolerance
            )
            
            # Track allocated resource
            self.active_resources[task_id] = {
                "model_type": model_type,
                "model_name": model_name,
                "allocated_at": datetime.now().isoformat(),
                "status": "active",
                "model_info": model.get_info() if hasattr(model, "get_info") else {}
            }
            
            logger.info(f"Allocated model for task {task_id}")
            
            return {
                "task_id": task_id,
                "model": model,
                "model_info": model.get_info() if hasattr(model, "get_info") else {}
            }
            
        except Exception as e:
            logger.error(f"Error allocating model for task {task_id}: {str(e)}")
            return None
    
    async def release_model_for_task(self, task_id: str) -> bool:
        """
        Release a model allocated for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if released successfully
        """
        if not self.resource_pool or task_id not in self.active_resources:
            return False
        
        logger.info(f"Releasing model for task {task_id}")
        
        try:
            # Get active resource
            resource = self.active_resources[task_id]
            
            # Release model
            await self.resource_pool.release_model(
                model_type=resource["model_type"],
                model_name=resource["model_name"]
            )
            
            # Update status
            resource["status"] = "released"
            resource["released_at"] = datetime.now().isoformat()
            
            logger.info(f"Released model for task {task_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error releasing model for task {task_id}: {str(e)}")
            return False
    
    # Hook handlers
    
    async def on_coordinator_startup(self, coordinator):
        """
        Handle coordinator startup event.
        
        Args:
            coordinator: Coordinator instance
        """
        logger.info("Coordinator startup detected")
        
        # Resource pool should already be initialized in the initialize method
        pass
    
    async def on_coordinator_shutdown(self, coordinator):
        """
        Handle coordinator shutdown event.
        
        Args:
            coordinator: Coordinator instance
        """
        logger.info("Coordinator shutdown detected")
        
        # Shutdown should already handle the resource pool shutdown
        pass
    
    async def on_task_created(self, task_id: str, task_data: Dict[str, Any]):
        """
        Handle task created event.
        
        Args:
            task_id: Task ID
            task_data: Task data
        """
        # Check if this task needs a model from the resource pool
        if "resource_pool" in task_data and task_data["resource_pool"]:
            # Allocate model for task
            allocation = await self.allocate_model_for_task(task_id, task_data)
            
            # Update task data with allocation information
            if allocation:
                self.coordinator.update_task_data(task_id, {
                    "resource_pool_allocation": {
                        "allocated_at": datetime.now().isoformat(),
                        "model_info": allocation["model_info"]
                    }
                })
    
    async def on_task_completed(self, task_id: str, result: Any):
        """
        Handle task completed event.
        
        Args:
            task_id: Task ID
            result: Task result
        """
        # Release model if allocated
        if task_id in self.active_resources:
            await self.release_model_for_task(task_id)
    
    async def on_task_failed(self, task_id: str, error: str):
        """
        Handle task failed event.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        # Release model if allocated
        if task_id in self.active_resources:
            await self.release_model_for_task(task_id)
    
    async def on_recovery_started(self, component_id: str, error: str):
        """
        Handle recovery started event.
        
        Args:
            component_id: Component ID
            error: Error message
        """
        logger.info(f"Recovery started for component {component_id}: {error}")
        
        # If recovery manager exists, notify it of the recovery event
        if self.recovery_manager:
            await self.recovery_manager.handle_recovery_event(
                event_type="started",
                component_id=component_id,
                error=error
            )
    
    async def on_recovery_completed(self, component_id: str, result: Any):
        """
        Handle recovery completed event.
        
        Args:
            component_id: Component ID
            result: Recovery result
        """
        logger.info(f"Recovery completed for component {component_id}")
        
        # If recovery manager exists, notify it of the recovery event
        if self.recovery_manager:
            await self.recovery_manager.handle_recovery_event(
                event_type="completed",
                component_id=component_id,
                result=result
            )
    
    def get_resource_pool_status(self) -> Dict[str, Any]:
        """
        Get the current resource pool status.
        
        Returns:
            Dictionary with resource pool status
        """
        if not self.resource_pool:
            return {
                "status": "not_initialized",
                "resources": {}
            }
        
        # Get basic status
        status = {
            "status": "active" if self.resource_pool.is_active() else "inactive",
            "active_resources": len(self.active_resources),
            "browser_connections": self.resource_pool.get_connection_count(),
            "fault_tolerance_enabled": self.config["enable_fault_tolerance"],
            "recovery_strategy": self.config["recovery_strategy"],
            "resources": {}
        }
        
        # Add active resources
        for task_id, resource in self.active_resources.items():
            if resource["status"] == "active":
                status["resources"][task_id] = {
                    "model_type": resource["model_type"],
                    "model_name": resource["model_name"],
                    "allocated_at": resource["allocated_at"]
                }
        
        # Add performance metrics if available
        if self.resource_metrics:
            latest_timestamp = max(self.resource_metrics.keys())
            status["latest_metrics"] = self.resource_metrics[latest_timestamp]
        
        return status