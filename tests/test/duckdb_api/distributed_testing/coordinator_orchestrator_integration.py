#!/usr/bin/env python3
"""
Coordinator-Orchestrator Integration Module

This module integrates the MultiDeviceOrchestrator with the CoordinatorServer,
enabling complex distributed task orchestration across multiple worker nodes.

The integration enables:
1. Orchestrated task submission and tracking via the coordinator API
2. Resource-aware task splitting and scheduling based on worker capabilities
3. Result aggregation from subtasks
4. Fault tolerance mechanisms for orchestrated tasks
5. Monitoring and visualization of complex multi-node tasks
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure parent directory is in the path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required modules
from duckdb_api.distributed_testing.multi_device_orchestrator import (
    MultiDeviceOrchestrator, SplitStrategy, TaskStatus
)

class CoordinatorOrchestratorIntegration:
    """
    Integrates the MultiDeviceOrchestrator with the CoordinatorServer.
    
    This class extends the coordinator with advanced orchestration capabilities,
    allowing complex tasks to be distributed and executed across multiple worker nodes
    with different hardware capabilities.
    """
    
    def __init__(self, coordinator):
        """
        Initialize the integration.
        
        Args:
            coordinator: The CoordinatorServer instance
        """
        self.coordinator = coordinator
        self.task_manager = getattr(coordinator, 'task_manager', None)
        self.worker_manager = getattr(coordinator, 'worker_manager', None)
        
        # Initialize the orchestrator with coordinator components
        self.orchestrator = MultiDeviceOrchestrator(
            coordinator=coordinator,
            task_manager=self.task_manager,
            worker_manager=self.worker_manager,
            resource_manager=getattr(coordinator, 'resource_manager', None)
        )
        
        # Track orchestrated tasks
        self.orchestrated_tasks = {}
        
        # Add API endpoints to the coordinator
        self._register_api_endpoints()
        
        # Setup result callback handling
        self._setup_result_handling()
        
        logger.info("Coordinator-Orchestrator integration initialized")
    
    def _register_api_endpoints(self):
        """Register additional API endpoints for orchestration in the coordinator."""
        if not hasattr(self.coordinator, 'api_handlers'):
            logger.warning("Coordinator does not support API handlers, skipping endpoint registration")
            return
        
        # Register API handlers
        self.coordinator.api_handlers.update({
            "/api/orchestrate": self._handle_orchestrate_request,
            "/api/orchestrated_task": self._handle_orchestrated_task_request,
            "/api/orchestrated_tasks": self._handle_list_orchestrated_tasks,
            "/api/cancel_orchestrated_task": self._handle_cancel_orchestrated_task
        })
        
        logger.info("Registered orchestration API endpoints")
    
    def _setup_result_handling(self):
        """Setup callback handling for subtask results."""
        # Register subtask result callback with task manager
        if hasattr(self.coordinator, 'register_callback_handler'):
            self.coordinator.register_callback_handler(
                "subtask_result", self._handle_subtask_result
            )
            logger.info("Registered subtask result callback handler")
        else:
            logger.warning("Coordinator does not support callback registration, subtask results will not be processed")
    
    async def _handle_orchestrate_request(self, request_data):
        """
        Handle an API request to orchestrate a task.
        
        Args:
            request_data: The request data
            
        Returns:
            Dict: Response data
        """
        logger.info("Handling orchestrate request")
        
        # Validate request
        required_fields = ["task_data", "strategy"]
        for field in required_fields:
            if field not in request_data:
                return {
                    "success": False,
                    "error": f"Missing required field: {field}"
                }
        
        # Extract request data
        task_data = request_data["task_data"]
        strategy = request_data["strategy"]
        
        # Orchestrate the task
        try:
            task_id = self.orchestrator.orchestrate_task(task_data, strategy)
            
            # Store in tracked tasks
            self.orchestrated_tasks[task_id] = {
                "request_data": request_data,
                "creation_time": datetime.now(),
                "status": "orchestrating"
            }
            
            return {
                "success": True,
                "task_id": task_id,
                "message": f"Task orchestrated with strategy: {strategy}"
            }
        except Exception as e:
            logger.exception(f"Error orchestrating task: {e}")
            return {
                "success": False,
                "error": f"Orchestration error: {str(e)}"
            }
    
    async def _handle_orchestrated_task_request(self, request_data):
        """
        Handle an API request to get orchestrated task status.
        
        Args:
            request_data: The request data
            
        Returns:
            Dict: Response data
        """
        # Validate request
        if "task_id" not in request_data:
            return {
                "success": False,
                "error": "Missing required field: task_id"
            }
        
        task_id = request_data["task_id"]
        
        # Get task status
        try:
            task_status = self.orchestrator.get_task_status(task_id)
            
            if task_status.get("status") == "not_found":
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}"
                }
            
            return {
                "success": True,
                "task_status": task_status
            }
        except Exception as e:
            logger.exception(f"Error getting task status: {e}")
            return {
                "success": False,
                "error": f"Error getting task status: {str(e)}"
            }
    
    async def _handle_list_orchestrated_tasks(self, request_data):
        """
        Handle an API request to list all orchestrated tasks.
        
        Args:
            request_data: The request data
            
        Returns:
            Dict: Response data
        """
        # Get optional filters
        filters = request_data.get("filters", {})
        limit = filters.get("limit", 100)
        offset = filters.get("offset", 0)
        status = filters.get("status")
        
        # Get tasks with status
        tasks = []
        count = 0
        skipped = 0
        
        for task_id, task_info in self.orchestrated_tasks.items():
            # Apply status filter if specified
            task_status = self.orchestrator.get_task_status(task_id)
            current_status = task_status.get("status", "unknown")
            
            if status and current_status != status:
                continue
                
            # Apply pagination
            if skipped < offset:
                skipped += 1
                continue
                
            if count >= limit:
                break
                
            count += 1
            
            # Add task to response
            tasks.append({
                "task_id": task_id,
                "status": current_status,
                "creation_time": task_info.get("creation_time", "").isoformat() if task_info.get("creation_time") else None,
                "strategy": task_status.get("strategy"),
                "completion_percentage": task_status.get("completion_percentage", 0)
            })
        
        return {
            "success": True,
            "tasks": tasks,
            "total": len(self.orchestrated_tasks),
            "returned": len(tasks),
            "offset": offset,
            "limit": limit
        }
    
    async def _handle_cancel_orchestrated_task(self, request_data):
        """
        Handle an API request to cancel an orchestrated task.
        
        Args:
            request_data: The request data
            
        Returns:
            Dict: Response data
        """
        # Validate request
        if "task_id" not in request_data:
            return {
                "success": False,
                "error": "Missing required field: task_id"
            }
        
        task_id = request_data["task_id"]
        
        # Cancel the task
        try:
            canceled = self.orchestrator.cancel_task(task_id)
            
            if not canceled:
                return {
                    "success": False,
                    "error": f"Task not found or could not be canceled: {task_id}"
                }
            
            return {
                "success": True,
                "message": f"Task canceled: {task_id}"
            }
        except Exception as e:
            logger.exception(f"Error canceling task: {e}")
            return {
                "success": False,
                "error": f"Error canceling task: {str(e)}"
            }
    
    def _handle_subtask_result(self, task_id, worker_id, result_data):
        """
        Handle a subtask result callback from a worker.
        
        Args:
            task_id: The task ID
            worker_id: The worker ID
            result_data: The result data
        """
        logger.info(f"Received subtask result for task {task_id} from worker {worker_id}")
        
        # Extract subtask ID and result from the result data
        subtask_id = result_data.get("subtask_id")
        if not subtask_id:
            logger.warning(f"Missing subtask_id in result data for task {task_id}")
            return
        
        # Process the subtask result
        success = result_data.get("success", False)
        result = result_data.get("result", {})
        
        try:
            self.orchestrator.process_subtask_result(subtask_id, result, success)
            logger.info(f"Processed result for subtask {subtask_id}, success={success}")
        except Exception as e:
            logger.exception(f"Error processing subtask result: {e}")
    
    def get_task_result(self, task_id):
        """
        Get the merged result of a completed orchestrated task.
        
        Args:
            task_id: The task ID
            
        Returns:
            Dict: The task result or None if not found/completed
        """
        return self.orchestrator.get_task_result(task_id)
    
    def get_task_status(self, task_id):
        """
        Get the status of an orchestrated task.
        
        Args:
            task_id: The task ID
            
        Returns:
            Dict: The task status
        """
        return self.orchestrator.get_task_status(task_id)
    
    def orchestrate_task(self, task_data, strategy):
        """
        Orchestrate a task for multi-device execution.
        
        Args:
            task_data: The task data
            strategy: The splitting strategy
            
        Returns:
            str: The orchestrated task ID
        """
        task_id = self.orchestrator.orchestrate_task(task_data, strategy)
        
        # Store in tracked tasks
        self.orchestrated_tasks[task_id] = {
            "task_data": task_data,
            "strategy": strategy,
            "creation_time": datetime.now(),
            "status": "orchestrating"
        }
        
        return task_id
    
    def stop(self):
        """Stop the integration and release resources."""
        logger.info("Stopping Coordinator-Orchestrator integration")
        
        # Stop the orchestrator
        if self.orchestrator:
            self.orchestrator.stop()


def integrate_orchestrator_with_coordinator(coordinator):
    """
    Integrate the MultiDeviceOrchestrator with a CoordinatorServer.
    
    Args:
        coordinator: The CoordinatorServer instance
        
    Returns:
        CoordinatorOrchestratorIntegration: The integration instance
    """
    integration = CoordinatorOrchestratorIntegration(coordinator)
    
    # Attach the integration to the coordinator for reference
    setattr(coordinator, 'orchestrator_integration', integration)
    
    return integration