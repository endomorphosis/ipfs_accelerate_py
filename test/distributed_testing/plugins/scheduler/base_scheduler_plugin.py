#!/usr/bin/env python3
"""
Base Scheduler Plugin for Distributed Testing Framework

This module provides a base implementation of the SchedulerPluginInterface that
can be extended to create custom scheduler plugins with minimal effort.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

from .scheduler_plugin_interface import SchedulerPluginInterface, SchedulingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseSchedulerPlugin(SchedulerPluginInterface):
    """
    Base implementation of the SchedulerPluginInterface.
    
    This class provides common functionality and default implementations
    for scheduler plugins, making it easier to create custom schedulers.
    """
    
    def __init__(self, name: str, version: str, description: str,
                strategies: List[SchedulingStrategy] = None):
        """
        Initialize the base scheduler plugin.
        
        Args:
            name: Plugin name
            version: Plugin version
            description: Plugin description
            strategies: List of scheduling strategies implemented by this plugin
        """
        self._name = name
        self._version = version
        self._description = description
        self._strategies = strategies or [SchedulingStrategy.ROUND_ROBIN]
        self._active_strategy = self._strategies[0] if self._strategies else SchedulingStrategy.ROUND_ROBIN
        
        # Coordinator reference
        self.coordinator = None
        
        # Task data
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_status: Dict[str, str] = {}
        self.task_worker: Dict[str, str] = {}
        self.task_execution_times: Dict[str, float] = {}
        
        # Worker data
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.worker_status: Dict[str, str] = {}
        self.worker_load: Dict[str, int] = {}
        self.worker_task_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance metrics
        self.metrics: Dict[str, Any] = {
            "tasks_scheduled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_execution_time": 0.0,
            "worker_utilization": {},
            "strategy_usage": {strategy.value: 0 for strategy in self._strategies}
        }
        
        # Configuration
        self.config: Dict[str, Any] = {
            "max_tasks_per_worker": 5,
            "history_window_size": 100,
            "detailed_logging": False,
        }
        
        logger.info(f"BaseSchedulerPlugin '{name}' v{version} initialized")
    
    def get_name(self) -> str:
        """
        Get the name of the scheduler plugin.
        
        Returns:
            str: Scheduler plugin name
        """
        return self._name
    
    def get_description(self) -> str:
        """
        Get a description of the scheduler plugin.
        
        Returns:
            str: Scheduler plugin description
        """
        return self._description
    
    def get_version(self) -> str:
        """
        Get the version of the scheduler plugin.
        
        Returns:
            str: Scheduler plugin version
        """
        return self._version
    
    def get_strategies(self) -> List[SchedulingStrategy]:
        """
        Get the scheduling strategies implemented by this plugin.
        
        Returns:
            List[SchedulingStrategy]: List of implemented scheduling strategies
        """
        return self._strategies.copy()
    
    async def initialize(self, coordinator: Any, config: Dict[str, Any] = None) -> bool:
        """
        Initialize the scheduler plugin.
        
        Args:
            coordinator: Reference to the coordinator instance
            config: Configuration dictionary for the scheduler
            
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        self.coordinator = coordinator
        
        # Update configuration if provided
        if config:
            self.configure(config)
        
        logger.info(f"BaseSchedulerPlugin '{self._name}' initialized with coordinator")
        return True
    
    async def shutdown(self) -> bool:
        """
        Shutdown the scheduler plugin.
        
        Returns:
            bool: True if shutdown succeeded, False otherwise
        """
        # Clear data structures
        self.tasks.clear()
        self.task_status.clear()
        self.task_worker.clear()
        self.task_execution_times.clear()
        self.workers.clear()
        self.worker_status.clear()
        self.worker_load.clear()
        self.worker_task_history.clear()
        
        logger.info(f"BaseSchedulerPlugin '{self._name}' shutdown complete")
        return True
    
    async def schedule_task(self, task_id: str, task_data: Dict[str, Any],
                          available_workers: Dict[str, Dict[str, Any]],
                          worker_load: Dict[str, int]) -> Optional[str]:
        """
        Schedule a task to an available worker.
        
        This implementation uses a simple round-robin scheduling strategy by default.
        
        Args:
            task_id: ID of the task to schedule
            task_data: Task data including requirements and metadata
            available_workers: Dictionary of available worker IDs to worker data
            worker_load: Dictionary of worker IDs to current task counts
            
        Returns:
            Optional[str]: Selected worker ID or None if no suitable worker found
        """
        # Store task data
        self.tasks[task_id] = task_data
        self.task_status[task_id] = "pending"
        
        # Update worker load data
        self.worker_load.update(worker_load)
        
        # Update worker data
        for worker_id, worker_data in available_workers.items():
            self.workers[worker_id] = worker_data
            self.worker_status[worker_id] = "active"
        
        # If no available workers, return None
        if not available_workers:
            logger.debug(f"No available workers for task {task_id}")
            return None
        
        # Schedule based on active strategy
        worker_id = None
        
        if self._active_strategy == SchedulingStrategy.ROUND_ROBIN:
            # Simple round-robin
            worker_ids = list(available_workers.keys())
            if worker_ids:
                # Use task ID hash for deterministic assignment
                worker_id = worker_ids[hash(task_id) % len(worker_ids)]
        
        elif self._active_strategy == SchedulingStrategy.LOAD_BALANCED:
            # Load-balanced scheduling
            worker_id = min(available_workers.keys(), key=lambda w: worker_load.get(w, 0))
        
        elif self._active_strategy == SchedulingStrategy.HARDWARE_MATCH:
            # Hardware matching (to be implemented by subclasses)
            worker_id = self._match_hardware(task_data, available_workers)
        
        elif self._active_strategy == SchedulingStrategy.PERFORMANCE_BASED:
            # Performance-based scheduling (to be implemented by subclasses)
            worker_id = self._performance_based_scheduling(task_id, task_data, available_workers)
        
        # Fallback to round-robin if no worker selected or strategy not implemented
        if worker_id is None:
            worker_ids = list(available_workers.keys())
            if worker_ids:
                worker_id = worker_ids[hash(task_id) % len(worker_ids)]
        
        # Update metrics if a worker was selected
        if worker_id:
            self.metrics["tasks_scheduled"] += 1
            self.metrics["strategy_usage"][self._active_strategy.value] += 1
            
            # Update task-worker mapping
            self.task_worker[task_id] = worker_id
            
            # Update worker load
            self.worker_load[worker_id] = self.worker_load.get(worker_id, 0) + 1
            
            # Update task status
            self.task_status[task_id] = "assigned"
            
            if self.config["detailed_logging"]:
                logger.info(f"Scheduled task {task_id} to worker {worker_id} using {self._active_strategy.value} strategy")
        
        return worker_id
    
    async def update_task_status(self, task_id: str, status: str,
                               worker_id: Optional[str],
                               execution_time: Optional[float] = None,
                               result: Any = None) -> None:
        """
        Update the status of a task.
        
        Args:
            task_id: ID of the task
            status: New status of the task
            worker_id: ID of the worker that processed the task
            execution_time: Execution time in seconds
            result: Task result or error information
        """
        # Update task status
        self.task_status[task_id] = status
        
        # Update task-worker mapping if provided
        if worker_id:
            self.task_worker[task_id] = worker_id
        
        # Update execution time if provided
        if execution_time is not None:
            self.task_execution_times[task_id] = execution_time
            
            # Update average execution time metric
            total_exec_time = sum(self.task_execution_times.values())
            count = len(self.task_execution_times)
            self.metrics["avg_execution_time"] = total_exec_time / count if count > 0 else 0.0
        
        # Update metrics based on status
        if status == "completed":
            self.metrics["tasks_completed"] += 1
            
            # Update worker task history
            if worker_id:
                # Initialize worker task history if needed
                if worker_id not in self.worker_task_history:
                    self.worker_task_history[worker_id] = []
                
                # Add task to worker history
                self.worker_task_history[worker_id].append({
                    "task_id": task_id,
                    "status": status,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Limit history size
                if len(self.worker_task_history[worker_id]) > self.config["history_window_size"]:
                    self.worker_task_history[worker_id].pop(0)
                
                # Update worker load
                if worker_id in self.worker_load:
                    self.worker_load[worker_id] = max(0, self.worker_load.get(worker_id, 0) - 1)
                
                # Update worker utilization metrics
                if worker_id not in self.metrics["worker_utilization"]:
                    self.metrics["worker_utilization"][worker_id] = {
                        "tasks_completed": 0,
                        "tasks_failed": 0,
                        "total_execution_time": 0.0
                    }
                
                self.metrics["worker_utilization"][worker_id]["tasks_completed"] += 1
                
                if execution_time is not None:
                    self.metrics["worker_utilization"][worker_id]["total_execution_time"] += execution_time
        
        elif status == "failed":
            self.metrics["tasks_failed"] += 1
            
            # Update worker metrics
            if worker_id:
                # Update worker load
                if worker_id in self.worker_load:
                    self.worker_load[worker_id] = max(0, self.worker_load.get(worker_id, 0) - 1)
                
                # Update worker utilization metrics
                if worker_id not in self.metrics["worker_utilization"]:
                    self.metrics["worker_utilization"][worker_id] = {
                        "tasks_completed": 0,
                        "tasks_failed": 0,
                        "total_execution_time": 0.0
                    }
                
                self.metrics["worker_utilization"][worker_id]["tasks_failed"] += 1
        
        if self.config["detailed_logging"]:
            logger.info(f"Updated task {task_id} status to {status}" + 
                       (f" (worker: {worker_id})" if worker_id else "") +
                       (f" (execution time: {execution_time:.2f}s)" if execution_time is not None else ""))
    
    async def update_worker_status(self, worker_id: str, status: str,
                                 capabilities: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the status of a worker.
        
        Args:
            worker_id: ID of the worker
            status: New status of the worker
            capabilities: Worker capabilities
        """
        # Update worker status
        self.worker_status[worker_id] = status
        
        # Update worker capabilities if provided
        if capabilities:
            if worker_id not in self.workers:
                self.workers[worker_id] = {}
            
            self.workers[worker_id].update(capabilities)
        
        # Handle worker disconnection
        if status == "disconnected" or status == "failed":
            # Remove from active worker load
            if worker_id in self.worker_load:
                del self.worker_load[worker_id]
        
        if self.config["detailed_logging"]:
            logger.info(f"Updated worker {worker_id} status to {status}")
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for this scheduler plugin.
        
        Returns:
            Dict[str, Any]: Configuration schema
        """
        return {
            "max_tasks_per_worker": {
                "type": "integer",
                "default": 5,
                "description": "Maximum number of concurrent tasks per worker"
            },
            "history_window_size": {
                "type": "integer",
                "default": 100,
                "description": "Number of tasks to keep in performance history"
            },
            "detailed_logging": {
                "type": "boolean",
                "default": False,
                "description": "Enable detailed scheduler logging"
            }
        }
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the scheduler plugin.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if configuration succeeded, False otherwise
        """
        # Update configuration with provided values
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
            else:
                logger.warning(f"Unknown configuration option: {key}")
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the scheduler's performance.
        
        Returns:
            Dict[str, Any]: Dictionary of scheduler metrics
        """
        # Update additional metrics that depend on current state
        self.metrics["active_tasks"] = len([status for status in self.task_status.values() 
                                         if status in ("pending", "assigned", "running")])
        self.metrics["active_workers"] = len([status for status in self.worker_status.values() 
                                          if status == "active"])
        
        # Calculate worker load
        total_workers = len(self.workers)
        self.metrics["avg_worker_load"] = sum(self.worker_load.values()) / total_workers if total_workers > 0 else 0.0
        
        # Add timestamp
        self.metrics["timestamp"] = datetime.now().isoformat()
        
        return self.metrics.copy()
    
    def set_strategy(self, strategy: SchedulingStrategy) -> bool:
        """
        Set the active scheduling strategy.
        
        Args:
            strategy: Scheduling strategy to use
            
        Returns:
            bool: True if strategy was set successfully, False otherwise
        """
        if strategy not in self._strategies:
            logger.error(f"Strategy {strategy.value} not supported by this plugin")
            return False
        
        self._active_strategy = strategy
        logger.info(f"Set active scheduling strategy to {strategy.value}")
        
        return True
    
    def get_active_strategy(self) -> SchedulingStrategy:
        """
        Get the currently active scheduling strategy.
        
        Returns:
            SchedulingStrategy: Currently active scheduling strategy
        """
        return self._active_strategy
    
    # Helper methods for subclasses to override
    
    def _match_hardware(self, task_data: Dict[str, Any], 
                       available_workers: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Match task to worker based on hardware requirements.
        
        Args:
            task_data: Task data including requirements
            available_workers: Dictionary of available workers
            
        Returns:
            Optional[str]: Selected worker ID or None if no match found
        """
        # To be implemented by subclasses
        return None
    
    def _performance_based_scheduling(self, task_id: str, task_data: Dict[str, Any], 
                                    available_workers: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Schedule task based on historical performance.
        
        Args:
            task_id: Task ID
            task_data: Task data
            available_workers: Dictionary of available workers
            
        Returns:
            Optional[str]: Selected worker ID or None if no suitable worker found
        """
        # To be implemented by subclasses
        return None