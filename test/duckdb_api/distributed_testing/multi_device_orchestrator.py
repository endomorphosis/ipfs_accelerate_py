#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Device Orchestrator for the Distributed Testing Framework.

This module provides multi-device orchestration capabilities for the Distributed Testing
Framework, allowing complex tasks to be split and executed across multiple worker nodes
with different hardware capabilities.
"""

import os
import sys
import json
import uuid
import time
import asyncio
import logging
import threading
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Enumeration for task status."""
    PENDING = "pending"          # Not yet started
    SPLITTING = "splitting"      # Being split into subtasks
    IN_PROGRESS = "in_progress"  # At least one subtask is running
    MERGING = "merging"          # Combining subtask results
    COMPLETED = "completed"      # All subtasks completed successfully
    FAILED = "failed"            # At least one subtask failed
    CANCELLED = "cancelled"      # Task was cancelled


class SubtaskStatus(Enum):
    """Enumeration for subtask status."""
    PENDING = "pending"          # Not yet assigned
    ASSIGNED = "assigned"        # Assigned to a worker
    RUNNING = "running"          # Currently executing
    COMPLETED = "completed"      # Successfully completed
    FAILED = "failed"            # Failed to complete
    CANCELLED = "cancelled"      # Subtask was cancelled


class SplitStrategy(Enum):
    """Enumeration for task splitting strategies."""
    DATA_PARALLEL = "data_parallel"        # Split input data across workers
    MODEL_PARALLEL = "model_parallel"      # Split model across workers
    PIPELINE_PARALLEL = "pipeline_parallel"  # Process data in stages across workers
    ENSEMBLE = "ensemble"                  # Run multiple versions in parallel
    FUNCTION_PARALLEL = "function_parallel"  # Split different functions across workers


class MultiDeviceOrchestrator:
    """
    Orchestrates the execution of complex tasks across multiple worker nodes.
    
    This class provides capabilities for:
    1. Splitting tasks into subtasks based on various strategies
    2. Scheduling subtasks across different workers based on their capabilities
    3. Tracking and managing subtask execution
    4. Merging results from subtasks into a coherent final result
    5. Handling failures and recovery
    """

    def __init__(self, 
                 coordinator=None, 
                 task_manager=None, 
                 worker_manager=None,
                 resource_manager=None):
        """
        Initialize the orchestrator.
        
        Args:
            coordinator: The coordinator server instance
            task_manager: The task manager instance
            worker_manager: The worker manager instance
            resource_manager: The dynamic resource manager instance
        """
        self.coordinator = coordinator
        self.task_manager = task_manager
        self.worker_manager = worker_manager
        self.resource_manager = resource_manager
        
        # Internal state
        self.orchestrated_tasks = {}  # Maps task_id to orchestration metadata
        self.subtasks = {}  # Maps subtask_id to subtask metadata
        self.task_subtasks = {}  # Maps task_id to set of subtask_ids
        self.subtask_results = {}  # Maps subtask_id to result
        
        # Splitting strategies mapped to implementation functions
        self.split_strategies = {
            SplitStrategy.DATA_PARALLEL: self._split_data_parallel,
            SplitStrategy.MODEL_PARALLEL: self._split_model_parallel,
            SplitStrategy.PIPELINE_PARALLEL: self._split_pipeline_parallel,
            SplitStrategy.ENSEMBLE: self._split_ensemble,
            SplitStrategy.FUNCTION_PARALLEL: self._split_function_parallel
        }
        
        # Merging strategies mapped to implementation functions
        self.merge_strategies = {
            SplitStrategy.DATA_PARALLEL: self._merge_data_parallel,
            SplitStrategy.MODEL_PARALLEL: self._merge_model_parallel,
            SplitStrategy.PIPELINE_PARALLEL: self._merge_pipeline_parallel,
            SplitStrategy.ENSEMBLE: self._merge_ensemble,
            SplitStrategy.FUNCTION_PARALLEL: self._merge_function_parallel
        }
        
        # Locks for thread safety
        self.tasks_lock = threading.RLock()
        self.subtasks_lock = threading.RLock()
        
        # Start background threads
        self.stop_event = threading.Event()
        self.status_monitor_thread = threading.Thread(target=self._monitor_subtasks)
        self.status_monitor_thread.daemon = True
        self.status_monitor_thread.start()
        
        logger.info("Multi-Device Orchestrator initialized")

    def orchestrate_task(self, task_data: Dict[str, Any], strategy: Union[str, SplitStrategy]) -> str:
        """
        Orchestrate a task for multi-device execution.
        
        Args:
            task_data: Task data dictionary
            strategy: Splitting strategy to use
            
        Returns:
            str: Orchestrated task ID
        """
        logger.info(f"Orchestrating task with strategy: {strategy}")
        
        # Convert string strategy to enum if needed
        if isinstance(strategy, str):
            strategy = SplitStrategy(strategy)
        
        # Generate a unique ID for the orchestrated task
        task_id = task_data.get("task_id", str(uuid.uuid4()))
        
        # Record orchestration metadata
        with self.tasks_lock:
            self.orchestrated_tasks[task_id] = {
                "task_data": task_data,
                "strategy": strategy,
                "status": TaskStatus.PENDING,
                "start_time": datetime.now(),
                "end_time": None,
                "error": None
            }
            self.task_subtasks[task_id] = set()
        
        # Start orchestration in a background thread to avoid blocking
        threading.Thread(
            target=self._execute_orchestration,
            args=(task_id,),
            daemon=True
        ).start()
        
        return task_id

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of an orchestrated task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict: Task status information
        """
        with self.tasks_lock:
            if task_id not in self.orchestrated_tasks:
                return {"status": "not_found"}
            
            task_info = self.orchestrated_tasks[task_id].copy()
            
            # Add subtask information
            subtask_ids = self.task_subtasks.get(task_id, set())
            subtasks_info = []
            
            for subtask_id in subtask_ids:
                with self.subtasks_lock:
                    if subtask_id in self.subtasks:
                        subtask_info = self.subtasks[subtask_id].copy()
                        # Remove large data fields
                        if "subtask_data" in subtask_info:
                            del subtask_info["subtask_data"]
                        subtasks_info.append(subtask_info)
            
            task_info["subtasks"] = subtasks_info
            task_info["total_subtasks"] = len(subtask_ids)
            
            # Calculate completion percentage
            completed_subtasks = sum(
                1 for s in subtasks_info 
                if s.get("status") in [SubtaskStatus.COMPLETED, SubtaskStatus.FAILED, SubtaskStatus.CANCELLED]
            )
            if subtask_ids:
                task_info["completion_percentage"] = int(completed_subtasks * 100 / len(subtask_ids))
            else:
                task_info["completion_percentage"] = 0
                
            return task_info

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel an orchestrated task and all its subtasks.
        
        Args:
            task_id: Task ID
            
        Returns:
            bool: True if task was cancelled, False otherwise
        """
        logger.info(f"Cancelling orchestrated task: {task_id}")
        
        with self.tasks_lock:
            if task_id not in self.orchestrated_tasks:
                return False
            
            # Update task status
            self.orchestrated_tasks[task_id]["status"] = TaskStatus.CANCELLED
            self.orchestrated_tasks[task_id]["end_time"] = datetime.now()
            
            # Cancel all subtasks
            subtask_ids = self.task_subtasks.get(task_id, set())
            
            for subtask_id in subtask_ids:
                with self.subtasks_lock:
                    if subtask_id in self.subtasks:
                        if self.subtasks[subtask_id]["status"] in [
                            SubtaskStatus.PENDING, 
                            SubtaskStatus.ASSIGNED,
                            SubtaskStatus.RUNNING
                        ]:
                            self.subtasks[subtask_id]["status"] = SubtaskStatus.CANCELLED
                            
                            # Cancel in task manager if assigned
                            original_task_id = self.subtasks[subtask_id].get("original_task_id")
                            if original_task_id and self.task_manager:
                                self.task_manager.cancel_task(original_task_id)
            
            return True

    def process_subtask_result(self, subtask_id: str, result: Dict[str, Any], success: bool = True) -> None:
        """
        Process the result of a completed subtask.
        
        Args:
            subtask_id: Subtask ID
            result: Result data
            success: Whether the subtask completed successfully
        """
        logger.info(f"Processing result for subtask {subtask_id}, success={success}")
        
        with self.subtasks_lock:
            if subtask_id not in self.subtasks:
                logger.warning(f"Unknown subtask ID: {subtask_id}")
                return
            
            # Update subtask status
            self.subtasks[subtask_id]["status"] = (
                SubtaskStatus.COMPLETED if success else SubtaskStatus.FAILED
            )
            self.subtasks[subtask_id]["end_time"] = datetime.now()
            
            if not success and "error" in result:
                self.subtasks[subtask_id]["error"] = result["error"]
            
            # Store the result
            self.subtask_results[subtask_id] = result
            
            # Get task ID
            task_id = self.subtasks[subtask_id]["task_id"]
            
        # Check if all subtasks are completed and merge results if needed
        self._check_task_completion(task_id)

    def get_subtask_result(self, subtask_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a completed subtask.
        
        Args:
            subtask_id: Subtask ID
            
        Returns:
            Dict: Result data or None if not found
        """
        return self.subtask_results.get(subtask_id)

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the merged result of a completed task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict: Merged result data or None if not found/completed
        """
        with self.tasks_lock:
            if task_id not in self.orchestrated_tasks:
                return None
            
            task_info = self.orchestrated_tasks[task_id]
            
            if task_info["status"] != TaskStatus.COMPLETED:
                return None
            
            return task_info.get("result")

    def stop(self) -> None:
        """Stop the orchestrator and clean up resources."""
        logger.info("Stopping Multi-Device Orchestrator")
        self.stop_event.set()
        
        if self.status_monitor_thread.is_alive():
            self.status_monitor_thread.join(timeout=5)
        
        # Cancel all running orchestrated tasks
        with self.tasks_lock:
            for task_id in list(self.orchestrated_tasks.keys()):
                if self.orchestrated_tasks[task_id]["status"] in [
                    TaskStatus.PENDING,
                    TaskStatus.SPLITTING,
                    TaskStatus.IN_PROGRESS,
                    TaskStatus.MERGING
                ]:
                    self.cancel_task(task_id)

    def _execute_orchestration(self, task_id: str) -> None:
        """
        Execute the orchestration workflow for a task.
        
        Args:
            task_id: Task ID
        """
        try:
            # Update task status
            with self.tasks_lock:
                if task_id not in self.orchestrated_tasks:
                    logger.warning(f"Task {task_id} not found for orchestration")
                    return
                
                task_info = self.orchestrated_tasks[task_id]
                task_info["status"] = TaskStatus.SPLITTING
            
            # Split the task into subtasks
            subtasks = self._split_task(task_id)
            
            # Update task status
            with self.tasks_lock:
                if self.orchestrated_tasks[task_id]["status"] == TaskStatus.CANCELLED:
                    logger.info(f"Task {task_id} was cancelled during splitting")
                    return
                
                self.orchestrated_tasks[task_id]["status"] = TaskStatus.IN_PROGRESS
            
            # Schedule subtasks for execution
            self._schedule_subtasks(subtasks)
            
        except Exception as e:
            logger.exception(f"Error orchestrating task {task_id}: {e}")
            
            # Update task status
            with self.tasks_lock:
                if task_id in self.orchestrated_tasks:
                    self.orchestrated_tasks[task_id]["status"] = TaskStatus.FAILED
                    self.orchestrated_tasks[task_id]["error"] = str(e)
                    self.orchestrated_tasks[task_id]["end_time"] = datetime.now()

    def _split_task(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Split a task into subtasks based on its strategy.
        
        Args:
            task_id: Task ID
            
        Returns:
            List[Dict]: List of subtask data dictionaries
        """
        with self.tasks_lock:
            task_info = self.orchestrated_tasks[task_id]
            task_data = task_info["task_data"]
            strategy = task_info["strategy"]
        
        # Get the appropriate splitting function for the strategy
        split_func = self.split_strategies.get(strategy)
        if not split_func:
            raise ValueError(f"Unsupported split strategy: {strategy}")
        
        # Call the strategy-specific splitting function
        subtasks = split_func(task_id, task_data)
        
        # Register subtasks
        with self.subtasks_lock:
            for subtask in subtasks:
                subtask_id = subtask["subtask_id"]
                self.subtasks[subtask_id] = {
                    "subtask_id": subtask_id,
                    "task_id": task_id,
                    "strategy": strategy,
                    "status": SubtaskStatus.PENDING,
                    "worker_id": None,
                    "original_task_id": None,
                    "start_time": None,
                    "end_time": None,
                    "error": None,
                    "subtask_data": subtask
                }
                self.task_subtasks.setdefault(task_id, set()).add(subtask_id)
        
        return subtasks

    def _schedule_subtasks(self, subtasks: List[Dict[str, Any]]) -> None:
        """
        Schedule subtasks for execution.
        
        Args:
            subtasks: List of subtask data dictionaries
        """
        for subtask in subtasks:
            subtask_id = subtask["subtask_id"]
            task_id = subtask["task_id"]
            
            # Skip if task was cancelled
            with self.tasks_lock:
                if task_id in self.orchestrated_tasks and self.orchestrated_tasks[task_id]["status"] == TaskStatus.CANCELLED:
                    logger.info(f"Task {task_id} was cancelled, skipping subtask scheduling")
                    continue
            
            # Prepare a task from the subtask
            task_request = self._prepare_task_from_subtask(subtask)
            
            # Add the task to the task manager
            if self.task_manager:
                new_task_id = self.task_manager.add_task(task_request, priority=task_request.get("priority", 5))
                
                # Update subtask with the new task ID
                with self.subtasks_lock:
                    if subtask_id in self.subtasks:
                        self.subtasks[subtask_id]["original_task_id"] = new_task_id
                        self.subtasks[subtask_id]["status"] = SubtaskStatus.ASSIGNED
                        self.subtasks[subtask_id]["start_time"] = datetime.now()
            else:
                logger.warning("Task manager not available, cannot schedule subtask")

    def _monitor_subtasks(self) -> None:
        """Background thread to monitor subtask status."""
        while not self.stop_event.is_set():
            try:
                # Check for running subtasks that have timed out
                current_time = datetime.now()
                
                with self.subtasks_lock:
                    for subtask_id, subtask_info in list(self.subtasks.items()):
                        if subtask_info["status"] == SubtaskStatus.RUNNING:
                            # Get the subtask timeout
                            timeout = subtask_info.get("timeout_seconds", 3600)  # Default 1 hour
                            
                            # Check if subtask has exceeded timeout
                            if subtask_info["start_time"] and (current_time - subtask_info["start_time"]).total_seconds() > timeout:
                                logger.warning(f"Subtask {subtask_id} timed out")
                                
                                # Mark as failed
                                subtask_info["status"] = SubtaskStatus.FAILED
                                subtask_info["error"] = "Subtask timed out"
                                subtask_info["end_time"] = current_time
                                
                                # Check if this completes the parent task
                                task_id = subtask_info["task_id"]
                                self._check_task_completion(task_id)
                
                # Sleep for a short time
                time.sleep(5)
                
            except Exception as e:
                logger.exception(f"Error in subtask monitor: {e}")
                time.sleep(10)  # Sleep longer on error

    def _check_task_completion(self, task_id: str) -> None:
        """
        Check if all subtasks for a task are completed and merge results if needed.
        
        Args:
            task_id: Task ID
        """
        with self.tasks_lock:
            if task_id not in self.orchestrated_tasks:
                return
            
            # Skip if task already completed or failed
            if self.orchestrated_tasks[task_id]["status"] in [
                TaskStatus.COMPLETED, 
                TaskStatus.FAILED,
                TaskStatus.CANCELLED
            ]:
                return
            
            subtask_ids = self.task_subtasks.get(task_id, set())
            if not subtask_ids:
                return
            
            # Check status of all subtasks
            all_completed = True
            any_failed = False
            
            with self.subtasks_lock:
                for subtask_id in subtask_ids:
                    if subtask_id not in self.subtasks:
                        continue
                    
                    subtask_status = self.subtasks[subtask_id]["status"]
                    
                    if subtask_status == SubtaskStatus.FAILED:
                        any_failed = True
                    elif subtask_status != SubtaskStatus.COMPLETED:
                        all_completed = False
            
            # If any subtask failed, mark task as failed
            if any_failed and self.orchestrated_tasks[task_id].get("fail_on_first_error", True):
                self.orchestrated_tasks[task_id]["status"] = TaskStatus.FAILED
                self.orchestrated_tasks[task_id]["end_time"] = datetime.now()
                self.orchestrated_tasks[task_id]["error"] = "One or more subtasks failed"
                return
            
            # If all subtasks completed, merge results
            if all_completed:
                self.orchestrated_tasks[task_id]["status"] = TaskStatus.MERGING
                
                try:
                    # Merge results
                    result = self._merge_subtask_results(task_id)
                    
                    # Update task status
                    self.orchestrated_tasks[task_id]["status"] = TaskStatus.COMPLETED
                    self.orchestrated_tasks[task_id]["result"] = result
                    self.orchestrated_tasks[task_id]["end_time"] = datetime.now()
                    
                except Exception as e:
                    logger.exception(f"Error merging results for task {task_id}: {e}")
                    self.orchestrated_tasks[task_id]["status"] = TaskStatus.FAILED
                    self.orchestrated_tasks[task_id]["error"] = f"Error merging results: {str(e)}"
                    self.orchestrated_tasks[task_id]["end_time"] = datetime.now()

    def _merge_subtask_results(self, task_id: str) -> Dict[str, Any]:
        """
        Merge results from subtasks based on the task's strategy.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict: Merged result data
        """
        with self.tasks_lock:
            task_info = self.orchestrated_tasks[task_id]
            strategy = task_info["strategy"]
        
        # Get the appropriate merging function for the strategy
        merge_func = self.merge_strategies.get(strategy)
        if not merge_func:
            raise ValueError(f"Unsupported merge strategy: {strategy}")
        
        # Get subtask IDs
        subtask_ids = self.task_subtasks.get(task_id, set())
        
        # Get subtask results
        subtask_results = []
        for subtask_id in subtask_ids:
            if subtask_id in self.subtask_results:
                subtask_result = self.subtask_results[subtask_id]
                subtask_data = self.subtasks[subtask_id]["subtask_data"]
                subtask_results.append({
                    "subtask_id": subtask_id,
                    "subtask_data": subtask_data,
                    "result": subtask_result
                })
        
        # Call the strategy-specific merging function
        return merge_func(task_id, subtask_results)

    def _prepare_task_from_subtask(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a task request from a subtask.
        
        Args:
            subtask: Subtask data dictionary
            
        Returns:
            Dict: Task request dictionary
        """
        # Copy task fields from the subtask
        task_request = subtask.get("task_data", {}).copy()
        
        # Add subtask specific fields
        task_request["task_id"] = str(uuid.uuid4())
        task_request["subtask_id"] = subtask["subtask_id"]
        task_request["parent_task_id"] = subtask["task_id"]
        task_request["is_subtask"] = True
        
        # Add callback information
        task_request["callback"] = {
            "type": "subtask_result",
            "subtask_id": subtask["subtask_id"],
            "task_id": subtask["task_id"]
        }
        
        return task_request

    # Splitting strategy implementations
    
    def _split_data_parallel(self, task_id: str, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a task using data parallelism (divide input data across workers).
        
        Args:
            task_id: Task ID
            task_data: Task data dictionary
            
        Returns:
            List[Dict]: List of subtask data dictionaries
        """
        # Get input data to split
        input_data = task_data.get("input_data", [])
        if not input_data:
            return []
        
        # Get number of partitions from task data or determine based on available workers
        num_partitions = task_data.get("num_partitions")
        if not num_partitions and self.worker_manager:
            # Use number of available workers, with a maximum of 10
            num_partitions = min(len(self.worker_manager.workers), 10)
        
        if not num_partitions or num_partitions <= 0:
            num_partitions = 1
        
        # Split input data into partitions
        partitions = [[] for _ in range(num_partitions)]
        for i, item in enumerate(input_data):
            partitions[i % num_partitions].append(item)
        
        # Create subtasks
        subtasks = []
        for i, partition in enumerate(partitions):
            if not partition:
                continue
                
            subtask_id = f"{task_id}_{i}"
            subtask_data = task_data.copy()
            subtask_data["input_data"] = partition
            subtask_data["partition_index"] = i
            subtask_data["num_partitions"] = num_partitions
            
            subtasks.append({
                "subtask_id": subtask_id,
                "task_id": task_id,
                "task_data": subtask_data,
                "partition_index": i,
                "num_partitions": num_partitions
            })
        
        return subtasks

    def _split_model_parallel(self, task_id: str, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a task using model parallelism (divide model across workers).
        
        Args:
            task_id: Task ID
            task_data: Task data dictionary
            
        Returns:
            List[Dict]: List of subtask data dictionaries
        """
        # Get model components to split
        model_components = task_data.get("model_components", [])
        if not model_components:
            # Default components for standard models
            model_type = task_data.get("config", {}).get("model_type", "")
            
            if "transformer" in model_type.lower() or "bert" in model_type.lower():
                # For transformer models, split by encoder/decoder or by layers
                if "encoder_decoder" in model_type.lower() or "t5" in model_type.lower():
                    model_components = ["encoder", "decoder"]
                else:
                    # Split by layers
                    num_layers = task_data.get("config", {}).get("num_layers", 12)
                    model_components = [f"layer_{i}" for i in range(num_layers)]
            elif "vit" in model_type.lower() or "vision" in model_type.lower():
                # For vision models, split by stages
                model_components = ["embedding", "transformer", "classifier"]
        
        if not model_components:
            # Fallback: create a single subtask
            subtask_id = f"{task_id}_0"
            return [{
                "subtask_id": subtask_id,
                "task_id": task_id,
                "task_data": task_data,
                "component_index": 0,
                "num_components": 1
            }]
        
        # Create subtasks
        subtasks = []
        for i, component in enumerate(model_components):
            subtask_id = f"{task_id}_{i}"
            subtask_data = task_data.copy()
            subtask_data["model_component"] = component
            subtask_data["component_index"] = i
            subtask_data["num_components"] = len(model_components)
            
            if "config" not in subtask_data:
                subtask_data["config"] = {}
            subtask_data["config"]["component"] = component
            
            subtasks.append({
                "subtask_id": subtask_id,
                "task_id": task_id,
                "task_data": subtask_data,
                "component_index": i,
                "num_components": len(model_components)
            })
        
        return subtasks

    def _split_pipeline_parallel(self, task_id: str, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a task using pipeline parallelism (process data in stages across workers).
        
        Args:
            task_id: Task ID
            task_data: Task data dictionary
            
        Returns:
            List[Dict]: List of subtask data dictionaries
        """
        # Get pipeline stages
        pipeline_stages = task_data.get("pipeline_stages", [])
        if not pipeline_stages:
            # Default stages for standard tasks
            task_type = task_data.get("type", "")
            
            if "benchmark" in task_type.lower():
                pipeline_stages = ["data_preparation", "model_loading", "inference", "metrics_calculation"]
            elif "test" in task_type.lower():
                pipeline_stages = ["setup", "execution", "validation", "cleanup"]
            else:
                pipeline_stages = ["stage_1"]
        
        # Create subtasks
        subtasks = []
        for i, stage in enumerate(pipeline_stages):
            subtask_id = f"{task_id}_{i}"
            subtask_data = task_data.copy()
            
            # Add stage-specific information
            subtask_data["pipeline_stage"] = stage
            subtask_data["stage_index"] = i
            subtask_data["num_stages"] = len(pipeline_stages)
            subtask_data["is_first_stage"] = (i == 0)
            subtask_data["is_last_stage"] = (i == len(pipeline_stages) - 1)
            
            # For non-first stages, we'll need to receive data from previous stage
            if i > 0:
                subtask_data["previous_stage"] = pipeline_stages[i-1]
                subtask_data["previous_stage_index"] = i-1
            
            # For non-last stages, we'll need to send data to next stage
            if i < len(pipeline_stages) - 1:
                subtask_data["next_stage"] = pipeline_stages[i+1]
                subtask_data["next_stage_index"] = i+1
            
            subtasks.append({
                "subtask_id": subtask_id,
                "task_id": task_id,
                "task_data": subtask_data,
                "stage_index": i,
                "num_stages": len(pipeline_stages)
            })
        
        return subtasks

    def _split_ensemble(self, task_id: str, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a task using ensemble approach (run multiple versions in parallel).
        
        Args:
            task_id: Task ID
            task_data: Task data dictionary
            
        Returns:
            List[Dict]: List of subtask data dictionaries
        """
        # Get ensemble configurations
        ensemble_configs = task_data.get("ensemble_configs", [])
        if not ensemble_configs:
            # Default ensemble configurations
            model_type = task_data.get("config", {}).get("model_type", "")
            
            if "bert" in model_type.lower():
                ensemble_configs = [
                    {"model_name": "bert-base-uncased", "variant": "base"},
                    {"model_name": "bert-large-uncased", "variant": "large"},
                    {"model_name": "distilbert-base-uncased", "variant": "distil"}
                ]
            elif "vit" in model_type.lower():
                ensemble_configs = [
                    {"model_name": "vit-base-patch16-224", "variant": "base"},
                    {"model_name": "vit-large-patch16-224", "variant": "large"},
                    {"model_name": "deit-base-patch16-224", "variant": "deit"}
                ]
            elif task_data.get("config", {}).get("model_name"):
                # Single model, use different hyperparameters
                ensemble_configs = [
                    {"precision": "fp16", "variant": "fp16"},
                    {"precision": "fp32", "variant": "fp32"},
                    {"batch_size": 1, "variant": "batch1"},
                    {"batch_size": 4, "variant": "batch4"}
                ]
        
        if not ensemble_configs:
            # Fallback: create a single subtask
            subtask_id = f"{task_id}_0"
            return [{
                "subtask_id": subtask_id,
                "task_id": task_id,
                "task_data": task_data,
                "ensemble_index": 0,
                "num_ensembles": 1
            }]
        
        # Create subtasks
        subtasks = []
        for i, config in enumerate(ensemble_configs):
            subtask_id = f"{task_id}_{i}"
            subtask_data = task_data.copy()
            
            # Merge configuration
            if "config" not in subtask_data:
                subtask_data["config"] = {}
            
            for key, value in config.items():
                subtask_data["config"][key] = value
            
            # Add ensemble metadata
            subtask_data["ensemble_index"] = i
            subtask_data["num_ensembles"] = len(ensemble_configs)
            subtask_data["ensemble_variant"] = config.get("variant", f"variant_{i}")
            
            subtasks.append({
                "subtask_id": subtask_id,
                "task_id": task_id,
                "task_data": subtask_data,
                "ensemble_index": i,
                "num_ensembles": len(ensemble_configs)
            })
        
        return subtasks

    def _split_function_parallel(self, task_id: str, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a task by functions (different operations across workers).
        
        Args:
            task_id: Task ID
            task_data: Task data dictionary
            
        Returns:
            List[Dict]: List of subtask data dictionaries
        """
        # Get functions to parallelize
        functions = task_data.get("functions", [])
        if not functions:
            # Default functions for standard tasks
            task_type = task_data.get("type", "")
            
            if "benchmark" in task_type.lower():
                functions = ["latency_test", "throughput_test", "memory_usage_test"]
            elif "test" in task_type.lower():
                functions = ["correctness_test", "performance_test", "memory_test"]
            else:
                functions = ["main_function"]
        
        # Create subtasks
        subtasks = []
        for i, function in enumerate(functions):
            subtask_id = f"{task_id}_{i}"
            subtask_data = task_data.copy()
            
            # Add function-specific information
            subtask_data["function"] = function
            subtask_data["function_index"] = i
            subtask_data["num_functions"] = len(functions)
            
            if "config" not in subtask_data:
                subtask_data["config"] = {}
            subtask_data["config"]["function"] = function
            
            subtasks.append({
                "subtask_id": subtask_id,
                "task_id": task_id,
                "task_data": subtask_data,
                "function_index": i,
                "num_functions": len(functions)
            })
        
        return subtasks

    # Merging strategy implementations
    
    def _merge_data_parallel(self, task_id: str, subtask_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from data parallel subtasks.
        
        Args:
            task_id: Task ID
            subtask_results: List of subtask results
            
        Returns:
            Dict: Merged result data
        """
        # Sort subtasks by partition index
        subtask_results.sort(key=lambda x: x["subtask_data"].get("partition_index", 0))
        
        # Merge results
        merged_results = {
            "task_id": task_id,
            "strategy": "data_parallel",
            "num_partitions": len(subtask_results),
            "results": []
        }
        
        for subtask in subtask_results:
            result = subtask["result"]
            if "results" in result:
                merged_results["results"].extend(result["results"])
        
        # Aggregate metrics if present
        metrics = {}
        for subtask in subtask_results:
            result = subtask["result"]
            if "metrics" in result:
                for key, value in result["metrics"].items():
                    if key in metrics:
                        # Average the metrics
                        metrics[key] = (metrics[key] + value) / 2
                    else:
                        metrics[key] = value
        
        if metrics:
            merged_results["metrics"] = metrics
        
        return merged_results

    def _merge_model_parallel(self, task_id: str, subtask_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from model parallel subtasks.
        
        Args:
            task_id: Task ID
            subtask_results: List of subtask results
            
        Returns:
            Dict: Merged result data
        """
        # Sort subtasks by component index
        subtask_results.sort(key=lambda x: x["subtask_data"].get("component_index", 0))
        
        # Extract component results
        component_results = {}
        for subtask in subtask_results:
            component = subtask["subtask_data"].get("model_component", "")
            component_results[component] = subtask["result"]
        
        # Create merged result
        merged_result = {
            "task_id": task_id,
            "strategy": "model_parallel",
            "num_components": len(subtask_results),
            "component_results": component_results
        }
        
        # Aggregate metrics if present
        metrics = {}
        for subtask in subtask_results:
            result = subtask["result"]
            if "metrics" in result:
                for key, value in result["metrics"].items():
                    if key not in metrics:
                        metrics[key] = value
        
        if metrics:
            merged_result["metrics"] = metrics
        
        # If output data is present in the last component, add it to merged result
        last_subtask = subtask_results[-1]["result"]
        if "output" in last_subtask:
            merged_result["output"] = last_subtask["output"]
        
        return merged_result

    def _merge_pipeline_parallel(self, task_id: str, subtask_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from pipeline parallel subtasks.
        
        Args:
            task_id: Task ID
            subtask_results: List of subtask results
            
        Returns:
            Dict: Merged result data
        """
        # Sort subtasks by stage index
        subtask_results.sort(key=lambda x: x["subtask_data"].get("stage_index", 0))
        
        # Extract stage results
        stage_results = {}
        for subtask in subtask_results:
            stage = subtask["subtask_data"].get("pipeline_stage", "")
            stage_results[stage] = subtask["result"]
        
        # Create merged result
        merged_result = {
            "task_id": task_id,
            "strategy": "pipeline_parallel",
            "num_stages": len(subtask_results),
            "stage_results": stage_results
        }
        
        # For pipeline tasks, the result of the last stage is typically the overall result
        last_stage_result = subtask_results[-1]["result"]
        
        # Extract relevant fields from the last stage
        for key in ["output", "metrics", "status"]:
            if key in last_stage_result:
                merged_result[key] = last_stage_result[key]
        
        return merged_result

    def _merge_ensemble(self, task_id: str, subtask_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from ensemble subtasks.
        
        Args:
            task_id: Task ID
            subtask_results: List of subtask results
            
        Returns:
            Dict: Merged result data
        """
        # Sort subtasks by ensemble index
        subtask_results.sort(key=lambda x: x["subtask_data"].get("ensemble_index", 0))
        
        # Extract variant results
        variant_results = {}
        for subtask in subtask_results:
            variant = subtask["subtask_data"].get("ensemble_variant", "")
            variant_results[variant] = subtask["result"]
        
        # Create merged result
        merged_result = {
            "task_id": task_id,
            "strategy": "ensemble",
            "num_ensembles": len(subtask_results),
            "variant_results": variant_results
        }
        
        # For ensemble tasks, we often need to aggregate predictions
        if all("predictions" in subtask["result"] for subtask in subtask_results):
            # Combine predictions (e.g., average for regression, voting for classification)
            predictions = {}
            
            for subtask in subtask_results:
                for key, value in subtask["result"]["predictions"].items():
                    if key not in predictions:
                        predictions[key] = []
                    predictions[key].append(value)
            
            # Average the predictions
            ensemble_predictions = {}
            for key, values in predictions.items():
                if isinstance(values[0], (int, float)):
                    # Average for numerical values
                    ensemble_predictions[key] = sum(values) / len(values)
                elif isinstance(values[0], list) and all(isinstance(v, (int, float)) for v in values[0]):
                    # Average for lists of numerical values
                    ensemble_predictions[key] = [sum(x) / len(x) for x in zip(*values)]
                else:
                    # For other types, use the most common value (voting)
                    ensemble_predictions[key] = max(set(values), key=values.count)
            
            merged_result["ensemble_predictions"] = ensemble_predictions
        
        # Aggregate metrics
        ensemble_metrics = {}
        for subtask in subtask_results:
            if "metrics" in subtask["result"]:
                for key, value in subtask["result"]["metrics"].items():
                    if key not in ensemble_metrics:
                        ensemble_metrics[key] = []
                    ensemble_metrics[key].append(value)
        
        # Calculate statistics for metrics
        if ensemble_metrics:
            metrics_stats = {}
            for key, values in ensemble_metrics.items():
                metrics_stats[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5
                }
            
            merged_result["ensemble_metrics"] = metrics_stats
        
        return merged_result

    def _merge_function_parallel(self, task_id: str, subtask_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from function parallel subtasks.
        
        Args:
            task_id: Task ID
            subtask_results: List of subtask results
            
        Returns:
            Dict: Merged result data
        """
        # Sort subtasks by function index
        subtask_results.sort(key=lambda x: x["subtask_data"].get("function_index", 0))
        
        # Extract function results
        function_results = {}
        for subtask in subtask_results:
            function = subtask["subtask_data"].get("function", "")
            function_results[function] = subtask["result"]
        
        # Create merged result
        merged_result = {
            "task_id": task_id,
            "strategy": "function_parallel",
            "num_functions": len(subtask_results),
            "function_results": function_results
        }
        
        # Aggregate metrics if present
        metrics = {}
        for subtask in subtask_results:
            result = subtask["result"]
            if "metrics" in result:
                for key, value in result["metrics"].items():
                    if key not in metrics:
                        metrics[key] = value
        
        if metrics:
            merged_result["metrics"] = metrics
        
        return merged_result