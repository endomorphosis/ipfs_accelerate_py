#!/usr/bin/env python3
"""
WebGPU/WebNN Resource Pool Integration Plugin for Distributed Testing Framework

This plugin integrates the WebGPU/WebNN Resource Pool with the Distributed Testing Framework,
providing browser-based acceleration capabilities for distributed tests with fault tolerance.
"""

import anyio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

# Import plugin base class
from .plugin_architecture import Plugin, PluginType, HookType

# Import WebGPU/WebNN Resource Pool components
try:
    from test.tests.web.web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
    from test.tests.web.web_platform.model_sharding import ShardedModelExecution
    RESOURCE_POOL_AVAILABLE = True
except ImportError:
    RESOURCE_POOL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebGPUResourcePoolPlugin(Plugin):
    """
    WebGPU/WebNN Resource Pool Integration Plugin for the Distributed Testing Framework.
    
    This plugin provides integration between the Distributed Testing Framework and the
    WebGPU/WebNN Resource Pool, enabling browser-based acceleration with fault tolerance
    for distributed tests.
    """
    
    def __init__(self):
        """Initialize the plugin."""
        super().__init__(
            name="WebGPUResourcePool",
            version="1.0.0",
            plugin_type=PluginType.INTEGRATION
        )
        
        if not RESOURCE_POOL_AVAILABLE:
            logger.warning("WebGPU Resource Pool components not available, plugin features will be limited")
        
        # Resource pool integration
        self.resource_pool_integration = None
        self.sharded_executions = {}
        
        # Test tracking
        self.active_browsers = {}
        self.resource_pool_tasks = {}
        self.recovery_events = {}
        
        # Default configuration
        self.config = {
            "max_browser_connections": 4,
            "browser_preferences": {
                "audio": "firefox",     # Firefox for audio models
                "vision": "chrome",     # Chrome for vision models
                "text_embedding": "edge" # Edge for embedding models
            },
            "enable_fault_tolerance": True,
            "recovery_strategy": "progressive",
            "state_sync_interval": 5,
            "redundancy_factor": 2,
            "advanced_logging": True,
            "metric_collection": True,
            "recovery_timeout": 30
        }
        
        # Register hooks
        self.register_hook(HookType.COORDINATOR_STARTUP, self.on_coordinator_startup)
        self.register_hook(HookType.COORDINATOR_SHUTDOWN, self.on_coordinator_shutdown)
        self.register_hook(HookType.TASK_CREATED, self.on_task_created)
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        self.register_hook(HookType.TASK_FAILED, self.on_task_failed)
        self.register_hook(HookType.WORKER_REGISTERED, self.on_worker_registered)
        self.register_hook(HookType.WORKER_FAILED, self.on_worker_failed)
        self.register_hook(HookType.RECOVERY_STARTED, self.on_recovery_started)
        self.register_hook(HookType.RECOVERY_COMPLETED, self.on_recovery_completed)
        
        logger.info("WebGPUResourcePoolPlugin initialized")
    
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
        
        if not RESOURCE_POOL_AVAILABLE:
            logger.warning("Resource Pool components not available, initialization limited")
            return True
        
        try:
            # Initialize resource pool integration
            self.resource_pool_integration = ResourcePoolBridgeIntegration(
                max_connections=self.config["max_browser_connections"],
                browser_preferences=self.config["browser_preferences"],
                adaptive_scaling=True,
                enable_fault_tolerance=self.config["enable_fault_tolerance"],
                recovery_strategy=self.config["recovery_strategy"],
                state_sync_interval=self.config["state_sync_interval"],
                redundancy_factor=self.config["redundancy_factor"]
            )
            
            # Initialize the integration
            await self.resource_pool_integration.initialize()
            
            # Start metrics collection if enabled
            if self.config["metric_collection"]:
                # TODO: Replace with task group - anyio task group for metrics collection
            
            logger.info("WebGPUResourcePoolPlugin initialized with coordinator")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing WebGPUResourcePoolPlugin: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin.
        
        Returns:
            True if shutdown succeeded
        """
        if not RESOURCE_POOL_AVAILABLE or not self.resource_pool_integration:
            return True
            
        try:
            # Release all active browser resources
            for browser_id, browser_info in list(self.active_browsers.items()):
                try:
                    await self.resource_pool_integration.release_browser(browser_id)
                    logger.info(f"Released browser resource: {browser_id}")
                except Exception as e:
                    logger.error(f"Error releasing browser {browser_id}: {e}")
            
            # Shutdown all sharded executions
            for exec_id, execution in list(self.sharded_executions.items()):
                try:
                    await execution.shutdown()
                    logger.info(f"Shutdown sharded execution: {exec_id}")
                except Exception as e:
                    logger.error(f"Error shutting down sharded execution {exec_id}: {e}")
            
            # Shutdown resource pool integration
            await self.resource_pool_integration.shutdown()
            
            logger.info("WebGPUResourcePoolPlugin shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down WebGPUResourcePoolPlugin: {e}")
            return False
    
    async def _collect_metrics(self):
        """Collect metrics from resource pool periodically."""
        if not RESOURCE_POOL_AVAILABLE or not self.resource_pool_integration:
            return
            
        while True:
            try:
                # Sleep for interval
                await anyio.sleep(60)  # Collect metrics every minute
                
                # Get performance history
                history = await self.resource_pool_integration.get_performance_history(
                    time_range="10m",  # Last 10 minutes
                    metrics=["latency", "throughput", "browser_utilization", "recovery_events"]
                )
                
                # Store in coordinator database if available
                if hasattr(self.coordinator, "db") and self.coordinator.db:
                    try:
                        # Store in database
                        if history:
                            # Transform metrics for storage
                            metrics_data = json.dumps(history)
                            timestamp = datetime.now().isoformat()
                            
                            # Store in database
                            query = """
                            INSERT INTO resource_pool_metrics (timestamp, metrics_data)
                            VALUES (?, ?)
                            """
                            self.coordinator.db.execute(query, (timestamp, metrics_data))
                            
                            logger.debug(f"Stored resource pool metrics in database: {len(metrics_data)} bytes")
                    except Exception as e:
                        logger.error(f"Error storing metrics in database: {e}")
                
                # Analyze trends if enabled
                if self.config.get("analyze_performance_trends", False):
                    try:
                        recommendations = await self.resource_pool_integration.analyze_performance_trends(history)
                        
                        # Apply recommendations if configured
                        if self.config.get("auto_apply_optimizations", False) and recommendations:
                            await self.resource_pool_integration.apply_performance_optimizations(recommendations)
                            logger.info(f"Applied {len(recommendations)} performance optimizations")
                    except Exception as e:
                        logger.error(f"Error analyzing performance trends: {e}")
                
            except anyio.get_cancelled_exc_class():
                logger.info("Metrics collection task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    async def get_model(self, model_type, model_name, hardware_preferences=None, fault_tolerance=None):
        """
        Get model from resource pool with fault tolerance.
        
        Args:
            model_type: Type of model (text_embedding, vision, audio)
            model_name: Name of model
            hardware_preferences: Hardware preferences
            fault_tolerance: Fault tolerance configuration
            
        Returns:
            Model instance
        """
        if not RESOURCE_POOL_AVAILABLE or not self.resource_pool_integration:
            raise RuntimeError("Resource Pool components not available")
            
        # Set default hardware preferences if not provided
        if not hardware_preferences:
            hardware_preferences = {
                'priority_list': ['webgpu', 'webnn', 'cpu']
            }
            
        # Set default fault tolerance if not provided
        if not fault_tolerance:
            fault_tolerance = {
                'recovery_timeout': self.config["recovery_timeout"],
                'state_persistence': True,
                'failover_strategy': 'immediate'
            }
            
        # Get model from resource pool
        model = await self.resource_pool_integration.get_model(
            model_type=model_type,
            model_name=model_name,
            hardware_preferences=hardware_preferences,
            fault_tolerance=fault_tolerance
        )
        
        # Track model in active resources
        browser_id = model.browser_id if hasattr(model, 'browser_id') else str(id(model))
        self.active_browsers[browser_id] = {
            'model_type': model_type,
            'model_name': model_name,
            'created_at': datetime.now().isoformat(),
            'model': model
        }
        
        return model
    
    async def create_sharded_execution(self, model_name, num_shards=3, sharding_strategy="layer_balanced", 
                                       fault_tolerance_level="high", recovery_strategy="coordinated"):
        """
        Create sharded model execution across multiple browsers.
        
        Args:
            model_name: Name of model to shard
            num_shards: Number of shards to create
            sharding_strategy: Strategy for sharding model
            fault_tolerance_level: Level of fault tolerance
            recovery_strategy: Strategy for recovery
            
        Returns:
            ShardedModelExecution instance
        """
        if not RESOURCE_POOL_AVAILABLE or not self.resource_pool_integration:
            raise RuntimeError("Resource Pool components not available")
            
        # Create sharded execution
        sharded_execution = ShardedModelExecution(
            model_name=model_name,
            sharding_strategy=sharding_strategy,
            num_shards=num_shards,
            fault_tolerance_level=fault_tolerance_level,
            recovery_strategy=recovery_strategy,
            connection_pool=self.resource_pool_integration.connection_pool
        )
        
        # Initialize sharded execution
        await sharded_execution.initialize()
        
        # Generate unique ID for this execution
        exec_id = f"shard-{model_name}-{int(time.time())}"
        
        # Track in sharded executions
        self.sharded_executions[exec_id] = sharded_execution
        
        return sharded_execution, exec_id
    
    async def release_resources(self, browser_id=None, exec_id=None):
        """
        Release browser or sharded execution resources.
        
        Args:
            browser_id: ID of browser to release
            exec_id: ID of sharded execution to release
        """
        if not RESOURCE_POOL_AVAILABLE or not self.resource_pool_integration:
            return
            
        if browser_id and browser_id in self.active_browsers:
            await self.resource_pool_integration.release_browser(browser_id)
            del self.active_browsers[browser_id]
            logger.info(f"Released browser resource: {browser_id}")
            
        if exec_id and exec_id in self.sharded_executions:
            await self.sharded_executions[exec_id].shutdown()
            del self.sharded_executions[exec_id]
            logger.info(f"Released sharded execution: {exec_id}")
    
    # Hook handlers
    
    async def on_coordinator_startup(self, coordinator):
        """
        Handle coordinator startup event.
        
        Args:
            coordinator: Coordinator instance
        """
        logger.info("Initializing WebGPU Resource Pool on coordinator startup")
        
        if not RESOURCE_POOL_AVAILABLE:
            return
            
        # Ensure the database table exists if coordinator has a database
        if hasattr(coordinator, "db") and coordinator.db:
            try:
                # Create metrics table if it doesn't exist
                coordinator.db.execute("""
                CREATE TABLE IF NOT EXISTS resource_pool_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metrics_data TEXT NOT NULL
                )
                """)
                
                # Create recovery events table if it doesn't exist
                coordinator.db.execute("""
                CREATE TABLE IF NOT EXISTS resource_pool_recovery_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    browser_id TEXT,
                    model_name TEXT,
                    recovery_duration REAL,
                    success BOOLEAN,
                    details TEXT
                )
                """)
                
                logger.info("Created resource pool metrics tables in database")
            except Exception as e:
                logger.error(f"Error creating database tables: {e}")
    
    async def on_coordinator_shutdown(self, coordinator):
        """
        Handle coordinator shutdown event.
        
        Args:
            coordinator: Coordinator instance
        """
        logger.info("Shutting down WebGPU Resource Pool on coordinator shutdown")
        
        # Cleanup will be handled by the shutdown method
    
    async def on_task_created(self, task_id: str, task_data: Dict[str, Any]):
        """
        Handle task created event.
        
        Args:
            task_id: Task ID
            task_data: Task data
        """
        # Check if task requires browser resources
        if not RESOURCE_POOL_AVAILABLE:
            return
            
        if task_data.get("requires_browser_resources", False):
            logger.info(f"Task {task_id} requires browser resources, tracking")
            
            # Track in resource pool tasks
            self.resource_pool_tasks[task_id] = {
                "id": task_id,
                "created_at": datetime.now().isoformat(),
                "status": "created",
                "data": task_data,
                "resources": []
            }
    
    async def on_task_completed(self, task_id: str, result: Any):
        """
        Handle task completed event.
        
        Args:
            task_id: Task ID
            result: Task result
        """
        if not RESOURCE_POOL_AVAILABLE or task_id not in self.resource_pool_tasks:
            return
            
        # Update task in tracking
        self.resource_pool_tasks[task_id]["status"] = "completed"
        self.resource_pool_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        self.resource_pool_tasks[task_id]["result"] = result
        
        # Release resources associated with task
        for resource in self.resource_pool_tasks[task_id]["resources"]:
            if resource["type"] == "browser" and resource["id"] in self.active_browsers:
                await self.release_resources(browser_id=resource["id"])
            elif resource["type"] == "sharded_execution" and resource["id"] in self.sharded_executions:
                await self.release_resources(exec_id=resource["id"])
        
        logger.info(f"Released all browser resources for completed task {task_id}")
    
    async def on_task_failed(self, task_id: str, error: str):
        """
        Handle task failed event.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        if not RESOURCE_POOL_AVAILABLE or task_id not in self.resource_pool_tasks:
            return
            
        # Update task in tracking
        self.resource_pool_tasks[task_id]["status"] = "failed"
        self.resource_pool_tasks[task_id]["failed_at"] = datetime.now().isoformat()
        self.resource_pool_tasks[task_id]["error"] = error
        
        # Release resources associated with task
        for resource in self.resource_pool_tasks[task_id]["resources"]:
            if resource["type"] == "browser" and resource["id"] in self.active_browsers:
                await self.release_resources(browser_id=resource["id"])
            elif resource["type"] == "sharded_execution" and resource["id"] in self.sharded_executions:
                await self.release_resources(exec_id=resource["id"])
        
        logger.info(f"Released all browser resources for failed task {task_id}")
    
    async def on_worker_registered(self, worker_id: str, capabilities: Dict[str, Any]):
        """
        Handle worker registered event.
        
        Args:
            worker_id: Worker ID
            capabilities: Worker capabilities
        """
        # Nothing to do for this event currently
        pass
    
    async def on_worker_failed(self, worker_id: str):
        """
        Handle worker failed event.
        
        Args:
            worker_id: Worker ID
        """
        # Check if worker had any browser resources
        if not RESOURCE_POOL_AVAILABLE:
            return
            
        # Identify tasks associated with this worker
        affected_tasks = []
        for task_id, task in self.resource_pool_tasks.items():
            if task["data"].get("assigned_worker") == worker_id:
                affected_tasks.append(task_id)
        
        if affected_tasks:
            logger.warning(f"Worker {worker_id} failed with {len(affected_tasks)} browser resource tasks")
            
            # Reset resources for affected tasks (recovery will be handled by the coordinator)
            for task_id in affected_tasks:
                self.resource_pool_tasks[task_id]["resources"] = []
    
    async def on_recovery_started(self, entity_id: str, entity_type: str, details: Dict[str, Any]):
        """
        Handle recovery started event.
        
        Args:
            entity_id: ID of entity being recovered
            entity_type: Type of entity
            details: Recovery details
        """
        if not RESOURCE_POOL_AVAILABLE:
            return
            
        # Track recovery event
        recovery_id = f"{entity_type}-{entity_id}"
        self.recovery_events[recovery_id] = {
            "started_at": datetime.now().isoformat(),
            "entity_id": entity_id,
            "entity_type": entity_type,
            "details": details,
            "status": "in_progress"
        }
        
        logger.info(f"Recovery started for {entity_type} {entity_id}")
    
    async def on_recovery_completed(self, entity_id: str, entity_type: str, success: bool, details: Dict[str, Any]):
        """
        Handle recovery completed event.
        
        Args:
            entity_id: ID of entity that was recovered
            entity_type: Type of entity
            success: Whether recovery was successful
            details: Recovery details
        """
        if not RESOURCE_POOL_AVAILABLE:
            return
            
        # Update recovery event
        recovery_id = f"{entity_type}-{entity_id}"
        if recovery_id in self.recovery_events:
            self.recovery_events[recovery_id]["completed_at"] = datetime.now().isoformat()
            self.recovery_events[recovery_id]["success"] = success
            self.recovery_events[recovery_id]["details"].update(details)
            self.recovery_events[recovery_id]["status"] = "completed" if success else "failed"
            
            # Calculate duration
            started_at = datetime.fromisoformat(self.recovery_events[recovery_id]["started_at"])
            completed_at = datetime.fromisoformat(self.recovery_events[recovery_id]["completed_at"])
            duration = (completed_at - started_at).total_seconds()
            self.recovery_events[recovery_id]["duration"] = duration
            
            # Store in database if available
            if hasattr(self.coordinator, "db") and self.coordinator.db:
                try:
                    # Store in database
                    event_data = self.recovery_events[recovery_id]
                    query = """
                    INSERT INTO resource_pool_recovery_events 
                    (timestamp, event_type, browser_id, model_name, recovery_duration, success, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    self.coordinator.db.execute(
                        query, 
                        (
                            event_data["completed_at"],
                            entity_type,
                            entity_id,
                            event_data["details"].get("model_name", "unknown"),
                            duration,
                            success,
                            json.dumps(event_data["details"])
                        )
                    )
                    logger.debug(f"Stored recovery event in database: {recovery_id}")
                except Exception as e:
                    logger.error(f"Error storing recovery event in database: {e}")
            
            logger.info(f"Recovery {'completed successfully' if success else 'failed'} for {entity_type} {entity_id} (duration: {duration:.2f}s)")
        else:
            logger.warning(f"Received recovery completion for unknown event: {entity_type} {entity_id}")
    
    def get_resource_pool_status(self) -> Dict[str, Any]:
        """
        Get the current resource pool status.
        
        Returns:
            Dictionary with resource pool status
        """
        if not RESOURCE_POOL_AVAILABLE or not self.resource_pool_integration:
            return {"status": "unavailable"}
            
        # Get basic status
        status = {
            "active_browsers": len(self.active_browsers),
            "sharded_executions": len(self.sharded_executions),
            "tasks_using_resources": len(self.resource_pool_tasks),
            "recovery_events": len(self.recovery_events)
        }
        
        # Add connection pool status if available
        if hasattr(self.resource_pool_integration, "connection_pool"):
            try:
                status["connection_pool"] = {
                    "total_connections": self.resource_pool_integration.connection_pool.total_connections,
                    "active_connections": self.resource_pool_integration.connection_pool.active_connections,
                    "available_connections": self.resource_pool_integration.connection_pool.available_connections,
                    "connection_failures": self.resource_pool_integration.connection_pool.connection_failures
                }
            except Exception as e:
                logger.error(f"Error getting connection pool status: {e}")
        
        return status