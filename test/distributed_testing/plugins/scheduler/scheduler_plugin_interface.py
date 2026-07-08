#!/usr/bin/env python3
"""
Scheduler Plugin Interface for Distributed Testing Framework

This module defines the interface that all scheduler plugins must implement
to be compatible with the distributed testing framework.
"""

import abc
import enum
from typing import Dict, List, Any, Optional, Tuple, Set

class SchedulingStrategy(enum.Enum):
    """
    Enumeration of scheduling strategies for task assignment.
    
    These strategies can be used individually or combined in custom schedulers.
    """
    
    ROUND_ROBIN = "round_robin"          # Simple round-robin scheduling
    PRIORITY_BASED = "priority_based"    # Scheduling based on task priority
    HARDWARE_MATCH = "hardware_match"    # Matching tasks to hardware capabilities
    PERFORMANCE_BASED = "performance"    # Based on historical performance
    DEADLINE_DRIVEN = "deadline_driven"  # Meeting task deadlines
    ENERGY_EFFICIENT = "energy_efficient"  # Optimizing for energy efficiency
    LOAD_BALANCED = "load_balanced"      # Balancing load across workers
    FAIR_SHARE = "fair_share"            # Fair resource allocation
    CUSTOM = "custom"                    # Custom scheduling algorithm

class SchedulerPluginInterface(abc.ABC):
    """
    Interface for scheduler plugins in the distributed testing framework.
    
    Custom schedulers must implement these methods to integrate with the
    distributed testing framework's scheduler plugin system.
    """
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the scheduler plugin.
        
        Returns:
            str: Scheduler plugin name
        """
        pass
    
    @abc.abstractmethod
    def get_description(self) -> str:
        """
        Get a description of the scheduler plugin.
        
        Returns:
            str: Scheduler plugin description
        """
        pass
    
    @abc.abstractmethod
    def get_version(self) -> str:
        """
        Get the version of the scheduler plugin.
        
        Returns:
            str: Scheduler plugin version
        """
        pass
    
    @abc.abstractmethod
    def get_strategies(self) -> List[SchedulingStrategy]:
        """
        Get the scheduling strategies implemented by this plugin.
        
        Returns:
            List[SchedulingStrategy]: List of implemented scheduling strategies
        """
        pass
    
    @abc.abstractmethod
    async def initialize(self, coordinator: Any, config: Dict[str, Any] = None) -> bool:
        """
        Initialize the scheduler plugin.
        
        Args:
            coordinator: Reference to the coordinator instance
            config: Configuration dictionary for the scheduler
            
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def shutdown(self) -> bool:
        """
        Shutdown the scheduler plugin.
        
        Returns:
            bool: True if shutdown succeeded, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def schedule_task(self, task_id: str, task_data: Dict[str, Any], 
                            available_workers: Dict[str, Dict[str, Any]],
                            worker_load: Dict[str, int]) -> Optional[str]:
        """
        Schedule a task to an available worker.
        
        This is the core scheduling method that assigns a task to the most
        appropriate worker based on the scheduler's algorithms and strategies.
        
        Args:
            task_id: ID of the task to schedule
            task_data: Task data including requirements and metadata
            available_workers: Dictionary of available worker IDs to worker data
            worker_load: Dictionary of worker IDs to current task counts
            
        Returns:
            Optional[str]: Selected worker ID or None if no suitable worker found
        """
        pass
    
    @abc.abstractmethod
    async def update_task_status(self, task_id: str, status: str, 
                                worker_id: Optional[str], 
                                execution_time: Optional[float] = None,
                                result: Any = None) -> None:
        """
        Update the status of a task.
        
        This method is called when a task status changes (e.g., starts, completes,
        fails) to allow the scheduler to update its internal state and metrics.
        
        Args:
            task_id: ID of the task
            status: New status of the task (e.g., 'running', 'completed', 'failed')
            worker_id: ID of the worker that processed the task, if any
            execution_time: Execution time in seconds, if available
            result: Task result or error information, if available
        """
        pass
    
    @abc.abstractmethod
    async def update_worker_status(self, worker_id: str, status: str, 
                                  capabilities: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the status of a worker.
        
        This method is called when a worker status changes (e.g., connects, 
        disconnects, fails) to allow the scheduler to update its internal state.
        
        Args:
            worker_id: ID of the worker
            status: New status of the worker (e.g., 'active', 'disconnected', 'failed')
            capabilities: Worker capabilities, if available
        """
        pass
    
    @abc.abstractmethod
    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for this scheduler plugin.
        
        The schema defines the configuration options that can be set for this 
        scheduler plugin, including types, default values, and descriptions.
        
        Returns:
            Dict[str, Any]: Configuration schema
        """
        pass
    
    @abc.abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the scheduler plugin.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if configuration succeeded, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the scheduler's performance.
        
        Returns:
            Dict[str, Any]: Dictionary of scheduler metrics
        """
        pass
    
    @abc.abstractmethod
    def set_strategy(self, strategy: SchedulingStrategy) -> bool:
        """
        Set the active scheduling strategy.
        
        Args:
            strategy: Scheduling strategy to use
            
        Returns:
            bool: True if strategy was set successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_active_strategy(self) -> SchedulingStrategy:
        """
        Get the currently active scheduling strategy.
        
        Returns:
            SchedulingStrategy: Currently active scheduling strategy
        """
        pass