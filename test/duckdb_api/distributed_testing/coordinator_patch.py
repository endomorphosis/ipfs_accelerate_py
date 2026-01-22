#!/usr/bin/env python3
"""
Coordinator Load Balancer Integration Patch

This module contains the patch for integrating the CoordinatorLoadBalancerIntegration
with the Coordinator. It modifies the Coordinator.__init__ method to initialize
the load balancer and modify the task assignment logic to use the load balancer.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the coordinator and load balancer integration
from duckdb_api.distributed_testing.coordinator import CoordinatorServer
from duckdb_api.distributed_testing.coordinator_load_balancer_integration import CoordinatorLoadBalancerIntegration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_patch")

# Store the original __init__ method
original_init = CoordinatorServer.__init__

def patched_init(self, host: str = "localhost", port: int = 8080,
                db_path: str = None, token_secret: str = None,
                heartbeat_timeout: int = 60, auto_recovery: bool = False,
                coordinator_id: str = None, coordinator_addresses: List[str] = None,
                performance_analyzer: bool = False, visualization_path: str = None,
                enable_load_balancer: bool = True, load_balancer_config: Dict[str, Any] = None):
    """
    Patched __init__ method that initializes the load balancer integration.
    
    Added parameters:
        enable_load_balancer: Whether to enable the load balancer integration
        load_balancer_config: Configuration for the load balancer
    """
    # Call the original __init__ method
    original_init(self, host, port, db_path, token_secret, heartbeat_timeout,
                 auto_recovery, coordinator_id, coordinator_addresses,
                 performance_analyzer, visualization_path)
    
    # Initialize load balancer integration if enabled
    self.load_balancer = None
    if enable_load_balancer:
        try:
            logger.info("Initializing load balancer integration...")
            self.load_balancer = CoordinatorLoadBalancerIntegration(
                coordinator=self,
                load_balancer_config=load_balancer_config,
                db_path=db_path
            )
            logger.info("Load balancer integration initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing load balancer integration: {e}")
            self.load_balancer = None

# Store the original start method
original_start = CoordinatorServer.start

def patched_start(self):
    """
    Patched start method that starts the load balancer integration.
    """
    # Start the load balancer if it exists
    if getattr(self, 'load_balancer', None):
        try:
            logger.info("Starting load balancer integration...")
            self.load_balancer.start()
            logger.info("Load balancer integration started successfully")
        except Exception as e:
            logger.error(f"Error starting load balancer integration: {e}")
    
    # Call the original start method
    original_start(self)

# Store the original stop method
original_stop = CoordinatorServer.stop

def patched_stop(self):
    """
    Patched stop method that stops the load balancer integration.
    """
    # Stop the load balancer if it exists
    if getattr(self, 'load_balancer', None):
        try:
            logger.info("Stopping load balancer integration...")
            self.load_balancer.stop()
            logger.info("Load balancer integration stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping load balancer integration: {e}")
    
    # Call the original stop method
    original_stop(self)

# Patch the TaskManager.get_next_task method to use the load balancer
# Store the original get_next_task method
from duckdb_api.distributed_testing.coordinator import TaskManager
original_get_next_task = TaskManager.get_next_task

def patched_get_next_task(self, worker_id: str, worker_capabilities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Patched get_next_task method that uses the load balancer to assign tasks.
    
    Args:
        worker_id: ID of the worker
        worker_capabilities: Capabilities of the worker
        
    Returns:
        Task dict if a suitable task is found, None otherwise
    """
    # Check if we have access to the coordinator's load balancer
    coordinator = getattr(self, '_coordinator', None)
    load_balancer = getattr(coordinator, 'load_balancer', None) if coordinator else None
    
    if load_balancer:
        # Try to get an assignment from the load balancer
        assignment = load_balancer.get_next_worker_assignment(worker_id)
        
        if assignment:
            task_id = assignment['task_id']
            logger.info(f"Load balancer assigned task {task_id} to worker {worker_id}")
            
            # Get the task from the task queue
            with self.task_lock:
                for i, (_, _, queue_task_id, task) in enumerate(self.task_queue):
                    if queue_task_id == task_id:
                        # Update task status
                        task['status'] = 'assigned'
                        task['worker_id'] = worker_id
                        
                        # Remove from queue
                        self.task_queue.pop(i)
                        
                        # Track in running tasks
                        self.running_tasks[task_id] = worker_id
                        
                        # Update database if available
                        if self.db_manager:
                            self.db_manager.update_task_status(task_id, 'assigned', worker_id)
                        
                        return task
    
    # Fall back to original method if load balancer is not available or didn't provide an assignment
    return original_get_next_task(self, worker_id, worker_capabilities)

def apply_patches():
    """Apply all patches to the coordinator."""
    # Patch the CoordinatorServer.__init__ method
    CoordinatorServer.__init__ = patched_init
    
    # Patch the CoordinatorServer.start method
    CoordinatorServer.start = patched_start
    
    # Patch the CoordinatorServer.stop method
    CoordinatorServer.stop = patched_stop
    
    # Patch the TaskManager.get_next_task method
    TaskManager.get_next_task = patched_get_next_task
    
    # Apply the reference to the coordinator in TaskManager
    # This enables the task manager to access the load balancer through the coordinator
    original_task_manager_init = TaskManager.__init__
    
    def patched_task_manager_init(self, db_manager=None, coordinator=None):
        original_task_manager_init(self, db_manager)
        self._coordinator = coordinator
    
    TaskManager.__init__ = patched_task_manager_init
    
    # Patch the CoordinatorServer._create_task_manager method to pass self to TaskManager
    original_create_task_manager = getattr(CoordinatorServer, '_create_task_manager', None)
    
    if original_create_task_manager:
        def patched_create_task_manager(self):
            return TaskManager(self.db_manager, self)
        
        CoordinatorServer._create_task_manager = patched_create_task_manager
    else:
        # If _create_task_manager doesn't exist, patch the task_manager initialization in __init__
        TaskManager.__init__ = patched_task_manager_init
        
        # Modify CoordinatorServer to pass self to TaskManager
        def task_manager_initializer(self):
            """Initialize the task manager."""
            return TaskManager(self.db_manager, self)
        
        # Apply the patch by replacing the task_manager property
        setattr(CoordinatorServer, 'task_manager', property(task_manager_initializer))
    
    logger.info("Applied all coordinator load balancer integration patches")

def remove_patches():
    """Remove all patches from the coordinator."""
    # Restore original methods
    CoordinatorServer.__init__ = original_init
    CoordinatorServer.start = original_start
    CoordinatorServer.stop = original_stop
    TaskManager.get_next_task = original_get_next_task
    
    logger.info("Removed all coordinator load balancer integration patches")

# Apply patches when imported
apply_patches()