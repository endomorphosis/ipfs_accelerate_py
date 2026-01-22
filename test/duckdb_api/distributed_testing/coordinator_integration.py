#!/usr/bin/env python3
"""
Coordinator Integration for Circuit Breaker Pattern

This module updates the CoordinatorServer class to include circuit breaker integration
for enhanced fault tolerance and monitoring.
"""

import logging
import os
import sys
from typing import Dict, List, Any, Optional, Union

# Import coordinator circuit breaker integration
from duckdb_api.distributed_testing.coordinator_circuit_breaker_integration import (
    CoordinatorCircuitBreakerIntegration
)

logger = logging.getLogger(__name__)

def integrate_circuit_breaker_with_coordinator(coordinator):
    """
    Integrate the circuit breaker pattern with the coordinator server.
    
    Args:
        coordinator: CoordinatorServer instance to integrate with
    
    Returns:
        True if integration was successful, False otherwise
    """
    try:
        # Create circuit breaker integration
        circuit_breaker_integration = CoordinatorCircuitBreakerIntegration(coordinator)
        
        # Store circuit breaker integration in coordinator
        coordinator.circuit_breaker_integration = circuit_breaker_integration
        
        # Modify worker manager to use circuit breakers
        _patch_worker_manager(coordinator, circuit_breaker_integration)
        
        # Modify task manager to use circuit breakers
        _patch_task_manager(coordinator, circuit_breaker_integration)
        
        logger.info("Successfully integrated circuit breaker pattern with coordinator")
        return True
    except Exception as e:
        logger.error(f"Error integrating circuit breaker pattern with coordinator: {e}")
        return False

def _patch_worker_manager(coordinator, circuit_breaker_integration):
    """
    Patch the worker manager to use circuit breakers.
    
    Args:
        coordinator: CoordinatorServer instance
        circuit_breaker_integration: Circuit breaker integration instance
    """
    worker_manager = getattr(coordinator, 'worker_manager', None)
    if not worker_manager:
        logger.warning("Worker manager not found in coordinator, skipping patch")
        return
    
    # Store original methods
    original_assign_task_to_worker = getattr(worker_manager, 'assign_task_to_worker', None)
    original_handle_worker_failure = getattr(worker_manager, 'handle_worker_failure', None)
    
    # Define patched methods
    async def patched_assign_task_to_worker(worker_id, task_id):
        """Patched version of assign_task_to_worker that uses circuit breakers."""
        try:
            # Use circuit breaker to protect against worker failures
            result = await circuit_breaker_integration.wrap_worker_execution(
                worker_id=worker_id,
                action=lambda: original_assign_task_to_worker(worker_id, task_id),
                fallback=lambda: _fallback_assign_task(worker_manager, worker_id, task_id)
            )
            return result
        except Exception as e:
            logger.error(f"Error in patched assign_task_to_worker: {e}")
            # Call the original method as a last resort
            return await original_assign_task_to_worker(worker_id, task_id)
    
    async def patched_handle_worker_failure(worker_id, reason=None):
        """Patched version of handle_worker_failure that notifies circuit breakers."""
        try:
            # Notify circuit breaker of the failure
            circuit_breaker_integration.on_worker_failure(worker_id, reason or 'unknown')
            
            # Call original method
            result = await original_handle_worker_failure(worker_id, reason)
            return result
        except Exception as e:
            logger.error(f"Error in patched handle_worker_failure: {e}")
            # Call the original method as a last resort
            return await original_handle_worker_failure(worker_id, reason)
    
    # Replace original methods with patched versions
    if original_assign_task_to_worker:
        worker_manager.assign_task_to_worker = patched_assign_task_to_worker
    
    if original_handle_worker_failure:
        worker_manager.handle_worker_failure = patched_handle_worker_failure
    
    logger.info("Worker manager successfully patched to use circuit breakers")

def _patch_task_manager(coordinator, circuit_breaker_integration):
    """
    Patch the task manager to use circuit breakers.
    
    Args:
        coordinator: CoordinatorServer instance
        circuit_breaker_integration: Circuit breaker integration instance
    """
    task_manager = getattr(coordinator, 'task_manager', None)
    if not task_manager:
        logger.warning("Task manager not found in coordinator, skipping patch")
        return
    
    # Store original methods
    original_start_task = getattr(task_manager, 'start_task', None)
    original_handle_task_failure = getattr(task_manager, 'handle_task_failure', None)
    
    # Define patched methods
    async def patched_start_task(task_id):
        """Patched version of start_task that uses circuit breakers."""
        try:
            # Get task type from task info
            task_info = await task_manager.get_task_info(task_id)
            task_type = task_info.get('type', 'unknown')
            
            # Use circuit breaker to protect against task failures
            result = await circuit_breaker_integration.wrap_task_execution(
                task_id=task_id,
                task_type=task_type,
                action=lambda: original_start_task(task_id),
                fallback=lambda: _fallback_start_task(task_manager, task_id)
            )
            return result
        except Exception as e:
            logger.error(f"Error in patched start_task: {e}")
            # Call the original method as a last resort
            return await original_start_task(task_id)
    
    async def patched_handle_task_failure(task_id, reason=None):
        """Patched version of handle_task_failure that notifies circuit breakers."""
        try:
            # Get task type from task info
            task_info = await task_manager.get_task_info(task_id)
            task_type = task_info.get('type', 'unknown')
            
            # Notify circuit breaker of the failure
            circuit_breaker_integration.on_task_failure(task_id, task_type, reason or 'unknown')
            
            # Call original method
            result = await original_handle_task_failure(task_id, reason)
            return result
        except Exception as e:
            logger.error(f"Error in patched handle_task_failure: {e}")
            # Call the original method as a last resort
            return await original_handle_task_failure(task_id, reason)
    
    # Replace original methods with patched versions
    if original_start_task:
        task_manager.start_task = patched_start_task
    
    if original_handle_task_failure:
        task_manager.handle_task_failure = patched_handle_task_failure
    
    logger.info("Task manager successfully patched to use circuit breakers")

async def _fallback_assign_task(worker_manager, worker_id, task_id):
    """
    Fallback method for when assign_task_to_worker fails due to circuit breaker.
    
    Args:
        worker_manager: Worker manager instance
        worker_id: ID of the worker
        task_id: ID of the task
        
    Returns:
        False to indicate failure
    """
    logger.warning(f"Circuit breaker prevented assigning task {task_id} to worker {worker_id}")
    
    # Find another worker to assign the task to
    try:
        # Query available workers
        available_workers = await worker_manager.get_available_workers()
        
        # Remove the current worker from the list
        available_workers = [w for w in available_workers if w['worker_id'] != worker_id]
        
        if available_workers:
            # Choose an alternative worker
            alternative_worker = available_workers[0]
            logger.info(f"Attempting to assign task {task_id} to alternative worker {alternative_worker['worker_id']}")
            
            # Call the original method with the alternative worker
            original_assign_task = getattr(worker_manager, 'assign_task_to_worker_original', 
                                        getattr(worker_manager, 'assign_task_to_worker', None))
            
            if original_assign_task:
                return await original_assign_task(alternative_worker['worker_id'], task_id)
    except Exception as e:
        logger.error(f"Error in fallback assign task: {e}")
    
    return False

async def _fallback_start_task(task_manager, task_id):
    """
    Fallback method for when start_task fails due to circuit breaker.
    
    Args:
        task_manager: Task manager instance
        task_id: ID of the task
        
    Returns:
        False to indicate failure
    """
    logger.warning(f"Circuit breaker prevented starting task {task_id}")
    
    try:
        # Update task status to indicate circuit breaker prevented execution
        await task_manager.update_task_status(task_id, "failed", error_message="Circuit breaker prevented execution")
        
        # Requeue the task for later execution with increased priority
        task_info = await task_manager.get_task_info(task_id)
        current_priority = task_info.get('priority', 0)
        await task_manager.update_task_priority(task_id, current_priority + 1)
        
        logger.info(f"Task {task_id} has been requeued with increased priority")
    except Exception as e:
        logger.error(f"Error in fallback start task: {e}")
    
    return False