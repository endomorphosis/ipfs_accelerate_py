#!/usr/bin/env python3
"""
Coordinator Error Integration Module

This module provides functions to integrate the enhanced error handling system
with the coordinator server.
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from duckdb_api.distributed_testing.distributed_error_handler import (
    DistributedErrorHandler,
    ErrorCategory
)

logger = logging.getLogger("coordinator_error_integration")

def integrate_error_handler(coordinator):
    """Integrate the error handler with the coordinator.
    
    Args:
        coordinator: The coordinator instance
    """
    # Create error handler if not already present
    if not hasattr(coordinator, "error_handler"):
        coordinator.error_handler = DistributedErrorHandler()
    
    # Store original method references
    original_handle_task_error = coordinator.handle_task_error
    original_handle_worker_error = coordinator.handle_worker_error
    
    # Override task error handling
    def enhanced_handle_task_error(task_id, error, worker_id=None):
        """Enhanced error handling for task failures.
        
        Args:
            task_id (str): The ID of the task that failed
            error (Dict[str, Any]): Error information
            worker_id (str, optional): The ID of the worker that reported the error
        
        Returns:
            Dict[str, Any]: Error handling result
        """
        logger.info(f"Enhanced task error handling for task {task_id}")
        
        # Get error handler instance
        error_handler = coordinator.error_handler
        
        # Add context for handling
        context = {
            "worker_id": worker_id,
            "task": coordinator.tasks.get(task_id),
            "hardware_type": coordinator.get_worker_hardware_type(worker_id) if worker_id else None,
            "attempt_count": coordinator.get_task_attempt_count(task_id)
        }
        
        # Handle the error
        result = error_handler.handle_error(task_id, error, context)
        
        # Process recovery actions
        if result.get("retry"):
            # Schedule task for retry
            retry_delay = result.get("retry_delay", 60)
            coordinator.reschedule_task(task_id, delay_seconds=retry_delay)
            logger.info(f"Task {task_id} scheduled for retry in {retry_delay} seconds")
        
        # Execute recovery actions
        recovery_action = result.get("recovery_action")
        if recovery_action and recovery_action.get("actions_taken"):
            for action in recovery_action["actions_taken"]:
                execute_recovery_action(coordinator, action, task_id, worker_id)
        
        # Call original method to ensure backward compatibility
        original_result = original_handle_task_error(task_id, error, worker_id)
        
        # Merge results
        return {**original_result, **result}
    
    # Override worker error handling
    def enhanced_handle_worker_error(worker_id, error):
        """Enhanced error handling for worker failures.
        
        Args:
            worker_id (str): The ID of the worker that failed
            error (Dict[str, Any]): Error information
        
        Returns:
            Dict[str, Any]: Error handling result
        """
        logger.info(f"Enhanced worker error handling for worker {worker_id}")
        
        # Get error handler instance
        error_handler = coordinator.error_handler
        
        # Add context for handling
        context = {
            "worker_id": worker_id,
            "worker": coordinator.workers.get(worker_id),
            "hardware_type": coordinator.get_worker_hardware_type(worker_id),
            "active_tasks": coordinator.get_worker_active_tasks(worker_id)
        }
        
        # Create a worker error entry
        worker_error = {
            "type": error.get("type", "WorkerError"),
            "message": error.get("message", "Worker failed"),
            "worker_id": worker_id,
            "timestamp": error.get("timestamp"),
            "traceback": error.get("traceback"),
            "hardware_context": {
                "hardware_type": context["hardware_type"]
            }
        }
        
        # Handle the error
        result = error_handler.handle_error(f"worker_{worker_id}", worker_error, context)
        
        # Process recovery actions
        recovery_action = result.get("recovery_action")
        if recovery_action and recovery_action.get("actions_taken"):
            for action in recovery_action["actions_taken"]:
                execute_recovery_action(coordinator, action, None, worker_id)
        
        # Call original method to ensure backward compatibility
        original_result = original_handle_worker_error(worker_id, error)
        
        # Merge results
        return {**original_result, **result}
    
    # Replace the original methods with enhanced versions
    coordinator.handle_task_error = enhanced_handle_task_error
    coordinator.handle_worker_error = enhanced_handle_worker_error
    
    # Add helper methods
    coordinator.get_worker_hardware_type = lambda worker_id: (
        coordinator.workers.get(worker_id, {}).get("capabilities", {}).get("hardware_types", ["unknown"])[0]
        if worker_id in coordinator.workers else "unknown"
    )
    
    coordinator.get_task_attempt_count = lambda task_id: (
        coordinator.tasks.get(task_id, {}).get("attempt_count", 1)
        if task_id in coordinator.tasks else 1
    )
    
    coordinator.get_worker_active_tasks = lambda worker_id: [
        task_id for task_id, task in coordinator.tasks.items()
        if task.get("worker_id") == worker_id and task.get("status") == "running"
    ]
    
    coordinator.reschedule_task = reschedule_task
    
    logger.info("Enhanced error handling integrated with coordinator")
    
    return coordinator


def reschedule_task(coordinator, task_id, delay_seconds=60):
    """Reschedule a task for execution after a delay.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task to reschedule
        delay_seconds (int): Delay before rescheduling in seconds
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot reschedule non-existent task {task_id}")
        return False
    
    task = coordinator.tasks[task_id]
    
    # Update task status
    task["status"] = "pending"
    
    # Increment attempt count
    task["attempt_count"] = task.get("attempt_count", 1) + 1
    
    # Clear worker assignment
    if "worker_id" in task:
        del task["worker_id"]
    
    # Set delay
    task["scheduled_time"] = coordinator.get_current_time() + delay_seconds
    
    logger.info(f"Task {task_id} rescheduled with attempt {task['attempt_count']} "
               f"and delay {delay_seconds} seconds")
    
    return True


def execute_recovery_action(coordinator, action, task_id=None, worker_id=None):
    """Execute a recovery action.
    
    Args:
        coordinator: The coordinator instance
        action (str): The recovery action to execute
        task_id (str, optional): The ID of the task associated with the action
        worker_id (str, optional): The ID of the worker associated with the action
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    logger.info(f"Executing recovery action: {action} for task={task_id}, worker={worker_id}")
    
    # Resource actions
    if action == "request_resource_cleanup" and worker_id:
        return request_resource_cleanup(coordinator, worker_id)
    
    elif action == "mark_resource_unavailable" and worker_id:
        return mark_resource_unavailable(coordinator, worker_id)
    
    elif action == "reallocate_task" and task_id:
        return reallocate_task(coordinator, task_id)
    
    # Network actions
    elif action == "increase_timeout" and task_id:
        return increase_timeout(coordinator, task_id)
    
    elif action == "reconnect" and worker_id:
        return request_worker_reconnect(coordinator, worker_id)
    
    elif action == "failover" and worker_id:
        return failover_worker(coordinator, worker_id)
    
    # Hardware actions
    elif action.startswith("mark_hardware_unavailable:") and worker_id:
        parts = action.split(":")
        if len(parts) >= 3:
            hardware_type = parts[1]
            return mark_hardware_unavailable(coordinator, worker_id, hardware_type)
    
    elif action == "reallocate_to_alternative_hardware" and task_id:
        return reallocate_to_alternative_hardware(coordinator, task_id)
    
    elif action == "reallocate_to_compatible_hardware" and task_id:
        return reallocate_to_compatible_hardware(coordinator, task_id)
    
    elif action == "update_hardware_requirements" and task_id:
        return update_hardware_requirements(coordinator, task_id)
    
    # Worker actions
    elif action.startswith("mark_worker_unavailable:"):
        parts = action.split(":")
        if len(parts) >= 2:
            target_worker_id = parts[1]
            return mark_worker_unavailable(coordinator, target_worker_id)
    
    elif action.startswith("mark_worker_slow:"):
        parts = action.split(":")
        if len(parts) >= 2:
            target_worker_id = parts[1]
            return mark_worker_slow(coordinator, target_worker_id)
    
    elif action.startswith("mark_worker_crashed:"):
        parts = action.split(":")
        if len(parts) >= 2:
            target_worker_id = parts[1]
            return mark_worker_crashed(coordinator, target_worker_id)
    
    elif action == "reassign_task" and task_id:
        return reassign_task(coordinator, task_id)
    
    elif action == "reassign_task_with_increased_timeout" and task_id:
        return reassign_task_with_increased_timeout(coordinator, task_id)
    
    elif action == "reassign_task_to_different_worker" and task_id:
        return reassign_task_to_different_worker(coordinator, task_id, worker_id)
    
    # Test execution actions
    elif action == "check_dependencies" and task_id:
        return check_dependencies(coordinator, task_id)
    
    elif action == "resolve_dependencies" and task_id:
        return resolve_dependencies(coordinator, task_id)
    
    elif action == "record_test_failure" and task_id:
        return record_test_failure(coordinator, task_id)
    
    elif action == "record_test_error" and task_id:
        return record_test_error(coordinator, task_id)
    
    logger.warning(f"Unknown recovery action: {action}")
    return False


# Resource actions

def request_resource_cleanup(coordinator, worker_id):
    """Request resource cleanup on a worker.
    
    Args:
        coordinator: The coordinator instance
        worker_id (str): The ID of the worker
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if worker_id not in coordinator.workers:
        logger.warning(f"Cannot request resource cleanup for non-existent worker {worker_id}")
        return False
    
    # Send resource cleanup command to worker
    try:
        coordinator.send_message_to_worker(worker_id, {
            "type": "command",
            "command": "cleanup_resources",
            "timestamp": coordinator.get_current_time()
        })
        logger.info(f"Resource cleanup requested for worker {worker_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to request resource cleanup for worker {worker_id}: {e}")
        return False


def mark_resource_unavailable(coordinator, worker_id):
    """Mark resources as unavailable on a worker.
    
    Args:
        coordinator: The coordinator instance
        worker_id (str): The ID of the worker
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if worker_id not in coordinator.workers:
        logger.warning(f"Cannot mark resources unavailable for non-existent worker {worker_id}")
        return False
    
    # Mark resource status in worker record
    try:
        coordinator.workers[worker_id]["resource_status"] = "limited"
        coordinator.workers[worker_id]["last_updated"] = coordinator.get_current_time()
        logger.info(f"Resources marked as limited for worker {worker_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to mark resources unavailable for worker {worker_id}: {e}")
        return False


def reallocate_task(coordinator, task_id):
    """Reallocate a task to a different worker.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot reallocate non-existent task {task_id}")
        return False
    
    # Mark task as needing reallocation
    try:
        coordinator.tasks[task_id]["status"] = "pending"
        coordinator.tasks[task_id]["needs_reallocation"] = True
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        
        # Clear worker assignment if present
        if "worker_id" in coordinator.tasks[task_id]:
            del coordinator.tasks[task_id]["worker_id"]
        
        logger.info(f"Task {task_id} marked for reallocation")
        return True
    except Exception as e:
        logger.error(f"Failed to reallocate task {task_id}: {e}")
        return False


# Network actions

def increase_timeout(coordinator, task_id):
    """Increase timeout for a task.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot increase timeout for non-existent task {task_id}")
        return False
    
    # Increase timeout in task configuration
    try:
        current_timeout = coordinator.tasks[task_id].get("timeout_seconds", 600)
        new_timeout = int(current_timeout * 1.5)  # Increase by 50%
        coordinator.tasks[task_id]["timeout_seconds"] = new_timeout
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        logger.info(f"Timeout for task {task_id} increased from {current_timeout} to {new_timeout} seconds")
        return True
    except Exception as e:
        logger.error(f"Failed to increase timeout for task {task_id}: {e}")
        return False


def request_worker_reconnect(coordinator, worker_id):
    """Request worker to reconnect.
    
    Args:
        coordinator: The coordinator instance
        worker_id (str): The ID of the worker
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if worker_id not in coordinator.workers:
        logger.warning(f"Cannot request reconnect for non-existent worker {worker_id}")
        return False
    
    # Send reconnect command to worker
    try:
        coordinator.send_message_to_worker(worker_id, {
            "type": "command",
            "command": "reconnect",
            "timestamp": coordinator.get_current_time()
        })
        logger.info(f"Reconnect requested for worker {worker_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to request reconnect for worker {worker_id}: {e}")
        return False


def failover_worker(coordinator, worker_id):
    """Failover worker to backup coordinator.
    
    Args:
        coordinator: The coordinator instance
        worker_id (str): The ID of the worker
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if worker_id not in coordinator.workers:
        logger.warning(f"Cannot failover non-existent worker {worker_id}")
        return False
    
    # Get backup coordinator if available
    backup_coordinator = coordinator.get_backup_coordinator_url()
    if not backup_coordinator:
        logger.warning(f"No backup coordinator available for failover of worker {worker_id}")
        return False
    
    # Send failover command to worker
    try:
        coordinator.send_message_to_worker(worker_id, {
            "type": "command",
            "command": "failover",
            "coordinator_url": backup_coordinator,
            "timestamp": coordinator.get_current_time()
        })
        logger.info(f"Failover to {backup_coordinator} requested for worker {worker_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to request failover for worker {worker_id}: {e}")
        return False


# Hardware actions

def mark_hardware_unavailable(coordinator, worker_id, hardware_type):
    """Mark hardware as unavailable on a worker.
    
    Args:
        coordinator: The coordinator instance
        worker_id (str): The ID of the worker
        hardware_type (str): The type of hardware
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if worker_id not in coordinator.workers:
        logger.warning(f"Cannot mark hardware unavailable for non-existent worker {worker_id}")
        return False
    
    # Mark hardware status in worker record
    try:
        if "hardware_status" not in coordinator.workers[worker_id]:
            coordinator.workers[worker_id]["hardware_status"] = {}
        
        coordinator.workers[worker_id]["hardware_status"][hardware_type] = "unavailable"
        coordinator.workers[worker_id]["last_updated"] = coordinator.get_current_time()
        
        logger.info(f"Hardware {hardware_type} marked as unavailable for worker {worker_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to mark hardware {hardware_type} unavailable for worker {worker_id}: {e}")
        return False


def reallocate_to_alternative_hardware(coordinator, task_id):
    """Reallocate task to alternative hardware.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot reallocate non-existent task {task_id}")
        return False
    
    # Update task to use alternative hardware
    try:
        # Get current hardware requirements
        requirements = coordinator.tasks[task_id].get("requirements", {})
        hardware_types = requirements.get("hardware", [])
        
        # No hardware requirements to change
        if not hardware_types:
            logger.warning(f"Task {task_id} has no hardware requirements to change")
            return False
        
        # Get current hardware type
        current_hardware = hardware_types[0] if hardware_types else None
        
        # Define fallback order for hardware types
        fallback_order = {
            "cuda": ["rocm", "mps", "webgpu", "cpu"],
            "rocm": ["cuda", "mps", "webgpu", "cpu"],
            "mps": ["cuda", "rocm", "webgpu", "cpu"],
            "webgpu": ["cuda", "rocm", "mps", "cpu"],
            "webnn": ["webgpu", "cuda", "rocm", "mps", "cpu"],
            "qnn": ["cuda", "rocm", "mps", "webgpu", "cpu"]
        }
        
        # Find alternative hardware
        alternatives = fallback_order.get(current_hardware, ["cpu"])
        
        # Update requirements to try alternative hardware first
        if current_hardware and alternatives:
            # Add current to the end of alternatives as last resort
            all_hardware = alternatives + [current_hardware]
            
            # Update hardware requirements
            coordinator.tasks[task_id]["requirements"] = {
                **requirements,
                "hardware": all_hardware
            }
            coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
            
            logger.info(f"Task {task_id} reallocated to alternative hardware: {all_hardware}")
            return True
        
        logger.warning(f"No alternative hardware found for task {task_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to reallocate task {task_id} to alternative hardware: {e}")
        return False


def reallocate_to_compatible_hardware(coordinator, task_id):
    """Reallocate task to compatible hardware.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot reallocate non-existent task {task_id}")
        return False
    
    # Similar to reallocate_to_alternative_hardware but with focus on compatibility
    try:
        # Mark task as needing hardware compatibility check
        coordinator.tasks[task_id]["needs_compatibility_check"] = True
        coordinator.tasks[task_id]["status"] = "pending"
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        
        # Clear worker assignment if present
        if "worker_id" in coordinator.tasks[task_id]:
            del coordinator.tasks[task_id]["worker_id"]
        
        logger.info(f"Task {task_id} marked for compatibility-based reallocation")
        return True
    except Exception as e:
        logger.error(f"Failed to reallocate task {task_id} to compatible hardware: {e}")
        return False


def update_hardware_requirements(coordinator, task_id):
    """Update hardware requirements for a task.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot update hardware requirements for non-existent task {task_id}")
        return False
    
    # Update hardware requirements based on task characteristics
    try:
        # Get current requirements
        requirements = coordinator.tasks[task_id].get("requirements", {})
        
        # Relax hardware requirements if they're too strict
        if "min_cuda_compute" in requirements:
            # Reduce minimum CUDA compute capability
            current_min = requirements["min_cuda_compute"]
            new_min = max(5.0, current_min - 1.0)  # Don't go lower than 5.0
            requirements["min_cuda_compute"] = new_min
            
            logger.info(f"Reduced min_cuda_compute for task {task_id} from {current_min} to {new_min}")
        
        if "min_memory_gb" in requirements:
            # Reduce minimum memory requirement by 25%
            current_min = requirements["min_memory_gb"]
            new_min = max(1.0, current_min * 0.75)  # Don't go lower than 1GB
            requirements["min_memory_gb"] = new_min
            
            logger.info(f"Reduced min_memory_gb for task {task_id} from {current_min} to {new_min}")
        
        # Update task requirements
        coordinator.tasks[task_id]["requirements"] = requirements
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        
        logger.info(f"Hardware requirements updated for task {task_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to update hardware requirements for task {task_id}: {e}")
        return False


# Worker actions

def mark_worker_unavailable(coordinator, worker_id):
    """Mark a worker as unavailable.
    
    Args:
        coordinator: The coordinator instance
        worker_id (str): The ID of the worker
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if worker_id not in coordinator.workers:
        logger.warning(f"Cannot mark non-existent worker {worker_id} as unavailable")
        return False
    
    # Mark worker as unavailable
    try:
        coordinator.workers[worker_id]["status"] = "unavailable"
        coordinator.workers[worker_id]["last_updated"] = coordinator.get_current_time()
        
        # Reassign active tasks
        active_tasks = [
            task_id for task_id, task in coordinator.tasks.items()
            if task.get("worker_id") == worker_id and task.get("status") == "running"
        ]
        
        for task_id in active_tasks:
            reassign_task(coordinator, task_id)
        
        logger.info(f"Worker {worker_id} marked as unavailable, {len(active_tasks)} tasks reassigned")
        return True
    except Exception as e:
        logger.error(f"Failed to mark worker {worker_id} as unavailable: {e}")
        return False


def mark_worker_slow(coordinator, worker_id):
    """Mark a worker as slow.
    
    Args:
        coordinator: The coordinator instance
        worker_id (str): The ID of the worker
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if worker_id not in coordinator.workers:
        logger.warning(f"Cannot mark non-existent worker {worker_id} as slow")
        return False
    
    # Mark worker as slow
    try:
        # Add slow marker with timestamp
        coordinator.workers[worker_id]["performance"] = "slow"
        coordinator.workers[worker_id]["slow_since"] = coordinator.get_current_time()
        coordinator.workers[worker_id]["last_updated"] = coordinator.get_current_time()
        
        # Adjust priority for future task assignment
        if "priority" not in coordinator.workers[worker_id]:
            coordinator.workers[worker_id]["priority"] = 0
        
        # Lower priority (higher number = lower priority)
        coordinator.workers[worker_id]["priority"] += 10
        
        logger.info(f"Worker {worker_id} marked as slow with priority {coordinator.workers[worker_id]['priority']}")
        return True
    except Exception as e:
        logger.error(f"Failed to mark worker {worker_id} as slow: {e}")
        return False


def mark_worker_crashed(coordinator, worker_id):
    """Mark a worker as crashed.
    
    Args:
        coordinator: The coordinator instance
        worker_id (str): The ID of the worker
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if worker_id not in coordinator.workers:
        logger.warning(f"Cannot mark non-existent worker {worker_id} as crashed")
        return False
    
    # Mark worker as crashed
    try:
        coordinator.workers[worker_id]["status"] = "crashed"
        coordinator.workers[worker_id]["crash_time"] = coordinator.get_current_time()
        coordinator.workers[worker_id]["last_updated"] = coordinator.get_current_time()
        
        # Reassign active tasks
        active_tasks = [
            task_id for task_id, task in coordinator.tasks.items()
            if task.get("worker_id") == worker_id and task.get("status") == "running"
        ]
        
        for task_id in active_tasks:
            reassign_task_to_different_worker(coordinator, task_id, worker_id)
        
        logger.info(f"Worker {worker_id} marked as crashed, {len(active_tasks)} tasks reassigned")
        return True
    except Exception as e:
        logger.error(f"Failed to mark worker {worker_id} as crashed: {e}")
        return False


def reassign_task(coordinator, task_id):
    """Reassign a task to any available worker.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot reassign non-existent task {task_id}")
        return False
    
    # Mark task for reassignment
    try:
        coordinator.tasks[task_id]["status"] = "pending"
        coordinator.tasks[task_id]["needs_reassignment"] = True
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        
        # Clear worker assignment if present
        if "worker_id" in coordinator.tasks[task_id]:
            del coordinator.tasks[task_id]["worker_id"]
        
        logger.info(f"Task {task_id} marked for reassignment")
        return True
    except Exception as e:
        logger.error(f"Failed to reassign task {task_id}: {e}")
        return False


def reassign_task_with_increased_timeout(coordinator, task_id):
    """Reassign a task with increased timeout.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot reassign non-existent task {task_id}")
        return False
    
    # Mark task for reassignment with increased timeout
    try:
        # Increase timeout
        current_timeout = coordinator.tasks[task_id].get("timeout_seconds", 600)
        new_timeout = int(current_timeout * 1.5)  # Increase by 50%
        coordinator.tasks[task_id]["timeout_seconds"] = new_timeout
        
        # Mark for reassignment
        coordinator.tasks[task_id]["status"] = "pending"
        coordinator.tasks[task_id]["needs_reassignment"] = True
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        
        # Clear worker assignment if present
        if "worker_id" in coordinator.tasks[task_id]:
            del coordinator.tasks[task_id]["worker_id"]
        
        logger.info(f"Task {task_id} marked for reassignment with timeout increased to {new_timeout} seconds")
        return True
    except Exception as e:
        logger.error(f"Failed to reassign task {task_id} with increased timeout: {e}")
        return False


def reassign_task_to_different_worker(coordinator, task_id, current_worker_id):
    """Reassign a task to a worker different from the current one.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
        current_worker_id (str): The ID of the current worker
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot reassign non-existent task {task_id}")
        return False
    
    # Mark task for reassignment to a different worker
    try:
        coordinator.tasks[task_id]["status"] = "pending"
        coordinator.tasks[task_id]["needs_reassignment"] = True
        coordinator.tasks[task_id]["exclude_workers"] = coordinator.tasks[task_id].get("exclude_workers", [])
        
        # Add current worker to exclusion list
        if current_worker_id and current_worker_id not in coordinator.tasks[task_id]["exclude_workers"]:
            coordinator.tasks[task_id]["exclude_workers"].append(current_worker_id)
        
        # Clear worker assignment if present
        if "worker_id" in coordinator.tasks[task_id]:
            del coordinator.tasks[task_id]["worker_id"]
        
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        
        logger.info(f"Task {task_id} marked for reassignment to a different worker (excluding {current_worker_id})")
        return True
    except Exception as e:
        logger.error(f"Failed to reassign task {task_id} to a different worker: {e}")
        return False


# Test execution actions

def check_dependencies(coordinator, task_id):
    """Check and install missing dependencies for a task.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot check dependencies for non-existent task {task_id}")
        return False
    
    # Mark task as needing dependency check
    try:
        coordinator.tasks[task_id]["check_dependencies"] = True
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        
        logger.info(f"Task {task_id} marked for dependency checking")
        return True
    except Exception as e:
        logger.error(f"Failed to mark task {task_id} for dependency checking: {e}")
        return False


def resolve_dependencies(coordinator, task_id):
    """Resolve task dependencies.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot resolve dependencies for non-existent task {task_id}")
        return False
    
    # Mark task as needing dependency resolution
    try:
        coordinator.tasks[task_id]["resolve_dependencies"] = True
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        
        logger.info(f"Task {task_id} marked for dependency resolution")
        return True
    except Exception as e:
        logger.error(f"Failed to mark task {task_id} for dependency resolution: {e}")
        return False


def record_test_failure(coordinator, task_id):
    """Record a test failure.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot record failure for non-existent task {task_id}")
        return False
    
    # Record test failure
    try:
        coordinator.tasks[task_id]["status"] = "failed"
        coordinator.tasks[task_id]["failure_type"] = "assertion"
        coordinator.tasks[task_id]["completed_time"] = coordinator.get_current_time()
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        
        logger.info(f"Test failure recorded for task {task_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to record test failure for task {task_id}: {e}")
        return False


def record_test_error(coordinator, task_id):
    """Record a test error.
    
    Args:
        coordinator: The coordinator instance
        task_id (str): The ID of the task
    
    Returns:
        bool: True if action was executed successfully, False otherwise
    """
    if task_id not in coordinator.tasks:
        logger.warning(f"Cannot record error for non-existent task {task_id}")
        return False
    
    # Record test error
    try:
        coordinator.tasks[task_id]["status"] = "error"
        coordinator.tasks[task_id]["error_type"] = "syntax"
        coordinator.tasks[task_id]["completed_time"] = coordinator.get_current_time()
        coordinator.tasks[task_id]["last_updated"] = coordinator.get_current_time()
        
        logger.info(f"Test error recorded for task {task_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to record test error for task {task_id}: {e}")
        return False