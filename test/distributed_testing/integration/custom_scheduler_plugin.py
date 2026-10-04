#!/usr/bin/env python3
"""
Custom Scheduler Plugin for Distributed Testing Framework

This plugin provides a custom scheduler implementation for the Distributed Testing Framework
that optimizes task assignment based on advanced algorithms, priorities, and hardware capabilities.
"""

import anyio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
import heapq
import random

# Import plugin base class
from .plugin_architecture import Plugin, PluginType, HookType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomSchedulerPlugin(Plugin):
    """
    Custom Scheduler Plugin for the Distributed Testing Framework.
    
    This plugin extends the task scheduling capabilities with:
    - Hardware-aware task assignment
    - Priority-based scheduling
    - Deadline-driven scheduling
    - Adaptive task distribution
    - Performance history-based assignment
    - Resource contention management
    - Test dependency resolution
    """
    
    def __init__(self):
        """Initialize the plugin."""
        super().__init__(
            name="CustomScheduler",
            version="1.0.0",
            plugin_type=PluginType.SCHEDULER
        )
        
        # Task queues
        self.high_priority_queue = []  # Priority queue for high priority tasks
        self.normal_priority_queue = []  # Priority queue for normal priority tasks
        self.low_priority_queue = []  # Priority queue for low priority tasks
        
        # Dependencies
        self.task_dependencies = {}  # Map of task_id -> set of dependency task_ids
        self.dependent_tasks = {}  # Map of task_id -> set of dependent task_ids
        self.completed_tasks = set()  # Set of completed task_ids
        
        # Worker statistics
        self.worker_stats = {}  # Map of worker_id -> performance statistics
        self.worker_capabilities = {}  # Map of worker_id -> hardware capabilities
        self.worker_current_load = {}  # Map of worker_id -> current number of assigned tasks
        self.worker_task_history = {}  # Map of worker_id -> list of completed tasks with metrics
        
        # Task metadata
        self.task_metadata = {}  # Map of task_id -> task metadata
        self.task_priorities = {}  # Map of task_id -> priority (1-10, 10 being highest)
        self.task_deadlines = {}  # Map of task_id -> deadline timestamp
        
        # Performance prediction
        self.model_performance_history = {}  # Map of (model_type, hardware_type) -> performance metrics
        
        # Default configuration
        self.config = {
            "max_tasks_per_worker": 5,  # Maximum number of concurrent tasks per worker
            "priority_levels": 10,  # Number of priority levels (1-10)
            "enable_adaptive_scheduling": True,  # Enable adaptive scheduling based on performance
            "enable_deadline_scheduling": True,  # Enable deadline-driven scheduling
            "enable_hardware_matching": True,  # Enable hardware-aware task assignment
            "enable_performance_prediction": True,  # Enable performance prediction
            "prediction_confidence_threshold": 0.7,  # Minimum confidence for predictions
            "max_retry_attempts": 3,  # Maximum number of retry attempts for failed tasks
            "worker_load_threshold": 0.8,  # Worker load threshold for task assignment (0.0-1.0)
            "history_window_size": 100,  # Number of tasks to keep in performance history
            "scheduler_interval": 1.0,  # Scheduler run interval in seconds
            "detailed_logging": False,  # Enable detailed scheduler logging
        }
        
        # Scheduling algorithm
        self.scheduling_algorithm = "adaptive"  # adaptive, round-robin, performance, hardware-match
        
        # Register hooks
        self.register_hook(HookType.COORDINATOR_STARTUP, self.on_coordinator_startup)
        self.register_hook(HookType.COORDINATOR_SHUTDOWN, self.on_coordinator_shutdown)
        self.register_hook(HookType.WORKER_REGISTERED, self.on_worker_registered)
        self.register_hook(HookType.WORKER_DISCONNECTED, self.on_worker_disconnected)
        self.register_hook(HookType.TASK_CREATED, self.on_task_created)
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        self.register_hook(HookType.TASK_FAILED, self.on_task_failed)
        
        logger.info("CustomSchedulerPlugin initialized")
    
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
        
        # Store reference to task scheduler if available
        if hasattr(coordinator, "task_scheduler"):
            self.original_scheduler = coordinator.task_scheduler
            logger.info("Found existing task scheduler, will integrate with it")
        else:
            self.original_scheduler = None
            logger.warning("No existing task scheduler found, will operate independently")
        
        # Start scheduler if not already running
        self.scheduler_running = False
        self.scheduler_task = None
        
        # Create the scheduler task
        self.scheduler_task = # TODO: Replace with task group - anyio task group for scheduler
        
        logger.info("CustomSchedulerPlugin initialized with coordinator")
        return True
    
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin.
        
        Returns:
            True if shutdown succeeded
        """
        # Cancel scheduler task if running
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except anyio.get_cancelled_exc_class():
                pass
            
        logger.info("CustomSchedulerPlugin shutdown complete")
        return True
    
    async def _run_scheduler(self):
        """Run the scheduler loop."""
        self.scheduler_running = True
        
        logger.info(f"Scheduler starting with algorithm: {self.scheduling_algorithm}")
        
        try:
            while True:
                # Wait for next scheduling interval
                await anyio.sleep(self.config["scheduler_interval"])
                
                # Check for tasks that are ready to be scheduled
                await self._schedule_ready_tasks()
                
                # Check for deadline violations
                if self.config["enable_deadline_scheduling"]:
                    await self._check_deadlines()
                
                # Update worker load metrics
                await self._update_worker_load()
                
        except anyio.get_cancelled_exc_class():
            logger.info("Scheduler task cancelled")
            self.scheduler_running = False
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
            self.scheduler_running = False
    
    async def _schedule_ready_tasks(self):
        """Schedule tasks that are ready to be executed."""
        # Combine all queues based on priority
        all_tasks = []
        
        # Start with high priority tasks
        while self.high_priority_queue:
            priority, timestamp, task_id = self.high_priority_queue[0]
            
            # Check if dependencies are met
            if task_id in self.task_dependencies:
                dependencies_met = all(dep in self.completed_tasks for dep in self.task_dependencies[task_id])
                if not dependencies_met:
                    # Skip this task for now
                    heapq.heappop(self.high_priority_queue)
                    continue
            
            # Add to schedule list
            all_tasks.append((priority, timestamp, task_id))
            heapq.heappop(self.high_priority_queue)
        
        # Add normal priority tasks
        while self.normal_priority_queue:
            priority, timestamp, task_id = self.normal_priority_queue[0]
            
            # Check if dependencies are met
            if task_id in self.task_dependencies:
                dependencies_met = all(dep in self.completed_tasks for dep in self.task_dependencies[task_id])
                if not dependencies_met:
                    # Skip this task for now
                    heapq.heappop(self.normal_priority_queue)
                    continue
            
            # Add to schedule list
            all_tasks.append((priority, timestamp, task_id))
            heapq.heappop(self.normal_priority_queue)
        
        # Add low priority tasks
        while self.low_priority_queue:
            priority, timestamp, task_id = self.low_priority_queue[0]
            
            # Check if dependencies are met
            if task_id in self.task_dependencies:
                dependencies_met = all(dep in self.completed_tasks for dep in self.task_dependencies[task_id])
                if not dependencies_met:
                    # Skip this task for now
                    heapq.heappop(self.low_priority_queue)
                    continue
            
            # Add to schedule list
            all_tasks.append((priority, timestamp, task_id))
            heapq.heappop(self.low_priority_queue)
        
        # Sort tasks by priority and timestamp
        all_tasks.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Count tasks scheduled in this round
        tasks_scheduled = 0
        
        # Get list of available workers
        available_workers = self._get_available_workers()
        
        if not available_workers:
            if self.config["detailed_logging"]:
                logger.info("No available workers for scheduling")
            return
        
        # Schedule tasks based on the selected algorithm
        for _, _, task_id in all_tasks:
            if task_id not in self.task_metadata:
                logger.warning(f"Task {task_id} not found in task metadata, skipping")
                continue
            
            task_data = self.task_metadata[task_id]
            
            # Skip if already completed or assigned
            if task_id in self.completed_tasks:
                continue
            
            # Check if we should use a specific scheduling algorithm for this task
            task_scheduling_algorithm = task_data.get("scheduling_algorithm", self.scheduling_algorithm)
            
            # Select worker based on scheduling algorithm
            selected_worker = None
            
            if task_scheduling_algorithm == "round-robin":
                # Simple round-robin scheduling
                if available_workers:
                    selected_worker = available_workers[tasks_scheduled % len(available_workers)]
                    
            elif task_scheduling_algorithm == "hardware-match" and self.config["enable_hardware_matching"]:
                # Hardware matching
                selected_worker = await self._find_matching_worker(task_data, available_workers)
                
            elif task_scheduling_algorithm == "performance" and self.config["enable_performance_prediction"]:
                # Performance-based scheduling
                selected_worker = await self._find_best_performing_worker(task_data, available_workers)
                
            elif task_scheduling_algorithm == "adaptive":
                # Adaptive scheduling - combines hardware matching and performance prediction
                # First try hardware matching
                if self.config["enable_hardware_matching"]:
                    selected_worker = await self._find_matching_worker(task_data, available_workers)
                
                # If no match, try performance-based scheduling
                if not selected_worker and self.config["enable_performance_prediction"]:
                    selected_worker = await self._find_best_performing_worker(task_data, available_workers)
                
                # If still no match, use round-robin
                if not selected_worker and available_workers:
                    selected_worker = available_workers[tasks_scheduled % len(available_workers)]
            
            # Fallback to round-robin if no worker selected
            if not selected_worker and available_workers:
                selected_worker = available_workers[tasks_scheduled % len(available_workers)]
            
            # Assign task to selected worker
            if selected_worker:
                # Call the original scheduler's assign_task method or directly assign task
                if self.original_scheduler and hasattr(self.original_scheduler, "assign_task"):
                    await self.original_scheduler.assign_task(task_id, selected_worker)
                elif hasattr(self.coordinator, "assign_task"):
                    await self.coordinator.assign_task(task_id, selected_worker)
                
                # Update worker load
                self.worker_current_load[selected_worker] = self.worker_current_load.get(selected_worker, 0) + 1
                
                # Log task assignment
                if self.config["detailed_logging"]:
                    logger.info(f"Assigned task {task_id} to worker {selected_worker} using {task_scheduling_algorithm} algorithm")
                
                tasks_scheduled += 1
        
        if self.config["detailed_logging"]:
            logger.info(f"Scheduled {tasks_scheduled} tasks out of {len(all_tasks)} ready tasks")
    
    async def _find_matching_worker(self, task_data: Dict[str, Any], available_workers: List[str]) -> Optional[str]:
        """
        Find a worker with matching hardware capabilities for the task.
        
        Args:
            task_data: Task metadata
            available_workers: List of available worker IDs
            
        Returns:
            Worker ID or None if no match found
        """
        if not task_data.get("hardware_requirements"):
            return None
            
        hardware_requirements = task_data["hardware_requirements"]
        
        # Filter workers by hardware requirements
        matching_workers = []
        
        for worker_id in available_workers:
            if worker_id not in self.worker_capabilities:
                continue
                
            capabilities = self.worker_capabilities[worker_id]
            
            # Check if worker matches all hardware requirements
            match = True
            
            for req_key, req_value in hardware_requirements.items():
                if req_key not in capabilities:
                    match = False
                    break
                    
                # Handle different requirement types
                if isinstance(req_value, bool) and capabilities[req_key] != req_value:
                    match = False
                    break
                elif isinstance(req_value, (int, float)) and capabilities[req_key] < req_value:
                    match = False
                    break
                elif isinstance(req_value, str) and capabilities[req_key] != req_value:
                    match = False
                    break
                elif isinstance(req_value, list) and capabilities[req_key] not in req_value:
                    match = False
                    break
            
            if match:
                matching_workers.append(worker_id)
        
        if not matching_workers:
            return None
            
        # Select the matching worker with the lowest load
        return min(matching_workers, key=lambda w: self.worker_current_load.get(w, 0))
    
    async def _find_best_performing_worker(self, task_data: Dict[str, Any], available_workers: List[str]) -> Optional[str]:
        """
        Find the worker with the best predicted performance for the task.
        
        Args:
            task_data: Task metadata
            available_workers: List of available worker IDs
            
        Returns:
            Worker ID or None if no prediction available
        """
        if not task_data.get("model_type"):
            return None
            
        model_type = task_data["model_type"]
        
        # Calculate predicted performance for each worker
        worker_predictions = {}
        
        for worker_id in available_workers:
            if worker_id not in self.worker_capabilities:
                continue
                
            hardware_type = self.worker_capabilities.get(worker_id, {}).get("hardware_type")
            
            if not hardware_type:
                continue
                
            # Check if we have performance history for this model and hardware type
            history_key = (model_type, hardware_type)
            
            if history_key in self.model_performance_history:
                history = self.model_performance_history[history_key]
                
                # Calculate average execution time
                avg_execution_time = sum(h["execution_time"] for h in history) / len(history)
                
                # Calculate confidence based on number of samples
                confidence = min(1.0, len(history) / 10.0)  # Linear confidence up to 10 samples
                
                # Only consider predictions with sufficient confidence
                if confidence >= self.config["prediction_confidence_threshold"]:
                    # Adjust prediction based on worker's current load
                    load_factor = 1.0 + (self.worker_current_load.get(worker_id, 0) * 0.2)  # 20% penalty per task
                    
                    predicted_time = avg_execution_time * load_factor
                    
                    worker_predictions[worker_id] = {
                        "predicted_time": predicted_time,
                        "confidence": confidence
                    }
        
        if not worker_predictions:
            return None
            
        # Select the worker with the shortest predicted execution time
        return min(worker_predictions.keys(), key=lambda w: worker_predictions[w]["predicted_time"])
    
    async def _check_deadlines(self):
        """Check for tasks that are approaching their deadlines."""
        if not self.config["enable_deadline_scheduling"]:
            return
            
        now = datetime.now()
        
        # Get tasks with deadlines
        deadline_tasks = {}
        
        for task_id, deadline in self.task_deadlines.items():
            # Skip completed tasks
            if task_id in self.completed_tasks:
                continue
                
            # Calculate time until deadline
            time_until_deadline = deadline - now
            
            # Convert to seconds
            seconds_until_deadline = time_until_deadline.total_seconds()
            
            if seconds_until_deadline > 0:
                deadline_tasks[task_id] = seconds_until_deadline
        
        if not deadline_tasks:
            return
            
        # Sort tasks by time until deadline
        sorted_tasks = sorted(deadline_tasks.items(), key=lambda x: x[1])
        
        for task_id, seconds in sorted_tasks:
            # If task is approaching deadline, increase its priority
            if seconds < 60:  # Less than 1 minute
                priority = 10  # Highest priority
            elif seconds < 300:  # Less than 5 minutes
                priority = 9
            elif seconds < 900:  # Less than 15 minutes
                priority = 8
            else:
                continue  # Not approaching deadline
                
            # Update task priority
            if task_id in self.task_priorities:
                original_priority = self.task_priorities[task_id]
                
                if priority > original_priority:
                    self.task_priorities[task_id] = priority
                    
                    if self.config["detailed_logging"]:
                        logger.info(f"Increased priority of task {task_id} from {original_priority} to {priority} due to approaching deadline")
    
    async def _update_worker_load(self):
        """Update worker load metrics."""
        for worker_id in list(self.worker_current_load.keys()):
            # Check if worker is still connected
            if worker_id not in self.worker_capabilities:
                del self.worker_current_load[worker_id]
                continue
                
            # Get current load from coordinator if available
            if hasattr(self.coordinator, "get_worker_load"):
                load = await self.coordinator.get_worker_load(worker_id)
                
                if load is not None:
                    self.worker_current_load[worker_id] = load
    
    def _get_available_workers(self) -> List[str]:
        """
        Get list of available workers.
        
        Returns:
            List of worker IDs
        """
        available_workers = []
        
        for worker_id in self.worker_capabilities:
            # Check if worker is not overloaded
            max_tasks = self.config["max_tasks_per_worker"]
            current_load = self.worker_current_load.get(worker_id, 0)
            
            if current_load < max_tasks:
                available_workers.append(worker_id)
        
        return available_workers
    
    def _add_task_to_queue(self, task_id: str, priority: int):
        """
        Add a task to the appropriate priority queue.
        
        Args:
            task_id: Task ID
            priority: Task priority (1-10)
        """
        # Create a tuple of (priority, timestamp, task_id)
        # Note: We negate priority for min-heap to work as max-heap
        task_item = (-priority, time.time(), task_id)
        
        # Determine which queue to use
        if priority >= 8:  # High priority (8-10)
            heapq.heappush(self.high_priority_queue, task_item)
            
            if self.config["detailed_logging"]:
                logger.info(f"Added task {task_id} to high priority queue with priority {priority}")
                
        elif priority >= 4:  # Normal priority (4-7)
            heapq.heappush(self.normal_priority_queue, task_item)
            
            if self.config["detailed_logging"]:
                logger.info(f"Added task {task_id} to normal priority queue with priority {priority}")
                
        else:  # Low priority (1-3)
            heapq.heappush(self.low_priority_queue, task_item)
            
            if self.config["detailed_logging"]:
                logger.info(f"Added task {task_id} to low priority queue with priority {priority}")
    
    def _add_task_dependency(self, task_id: str, dependency_id: str):
        """
        Add a dependency between tasks.
        
        Args:
            task_id: Task ID
            dependency_id: Dependency task ID
        """
        # Add to task dependencies
        if task_id not in self.task_dependencies:
            self.task_dependencies[task_id] = set()
            
        self.task_dependencies[task_id].add(dependency_id)
        
        # Add to dependent tasks
        if dependency_id not in self.dependent_tasks:
            self.dependent_tasks[dependency_id] = set()
            
        self.dependent_tasks[dependency_id].add(task_id)
        
        if self.config["detailed_logging"]:
            logger.info(f"Added dependency: task {task_id} depends on {dependency_id}")
    
    def _update_performance_history(self, model_type: str, hardware_type: str, execution_time: float):
        """
        Update performance history for a model and hardware type.
        
        Args:
            model_type: Model type
            hardware_type: Hardware type
            execution_time: Execution time in seconds
        """
        history_key = (model_type, hardware_type)
        
        if history_key not in self.model_performance_history:
            self.model_performance_history[history_key] = []
            
        history = self.model_performance_history[history_key]
        
        # Add new entry
        history.append({
            "execution_time": execution_time,
            "timestamp": time.time()
        })
        
        # Limit history size
        if len(history) > self.config["history_window_size"]:
            history.pop(0)
    
    # Hook handlers
    
    async def on_coordinator_startup(self, coordinator):
        """
        Handle coordinator startup event.
        
        Args:
            coordinator: Coordinator instance
        """
        logger.info("Coordinator startup detected, CustomSchedulerPlugin ready")
    
    async def on_coordinator_shutdown(self, coordinator):
        """
        Handle coordinator shutdown event.
        
        Args:
            coordinator: Coordinator instance
        """
        logger.info("Coordinator shutdown detected")
    
    async def on_worker_registered(self, worker_id: str, capabilities: Dict[str, Any]):
        """
        Handle worker registered event.
        
        Args:
            worker_id: Worker ID
            capabilities: Worker capabilities
        """
        # Store worker capabilities
        self.worker_capabilities[worker_id] = capabilities
        
        # Initialize worker statistics
        self.worker_stats[worker_id] = {
            "registered_at": datetime.now().isoformat(),
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0
        }
        
        # Initialize worker task history
        self.worker_task_history[worker_id] = []
        
        # Initialize worker load
        self.worker_current_load[worker_id] = 0
        
        logger.info(f"Worker {worker_id} registered with capabilities: {capabilities}")
    
    async def on_worker_disconnected(self, worker_id: str):
        """
        Handle worker disconnected event.
        
        Args:
            worker_id: Worker ID
        """
        # Remove worker from capabilities
        if worker_id in self.worker_capabilities:
            del self.worker_capabilities[worker_id]
            
        # Remove worker from load tracking
        if worker_id in self.worker_current_load:
            del self.worker_current_load[worker_id]
            
        logger.info(f"Worker {worker_id} disconnected")
    
    async def on_task_created(self, task_id: str, task_data: Dict[str, Any]):
        """
        Handle task created event.
        
        Args:
            task_id: Task ID
            task_data: Task data
        """
        # Store task metadata
        self.task_metadata[task_id] = task_data
        
        # Extract task priority
        priority = task_data.get("priority", 5)  # Default to middle priority (5)
        
        # Ensure priority is within range
        priority = max(1, min(self.config["priority_levels"], priority))
        
        # Store task priority
        self.task_priorities[task_id] = priority
        
        # Extract task deadline if provided
        deadline = task_data.get("deadline")
        
        if deadline:
            if isinstance(deadline, str):
                try:
                    deadline = datetime.fromisoformat(deadline)
                    self.task_deadlines[task_id] = deadline
                except ValueError:
                    logger.warning(f"Invalid deadline format for task {task_id}: {deadline}")
            elif isinstance(deadline, (int, float)):
                # Assume seconds from now
                deadline = datetime.now() + timedelta(seconds=deadline)
                self.task_deadlines[task_id] = deadline
        
        # Extract task dependencies
        dependencies = task_data.get("dependencies", [])
        
        for dependency_id in dependencies:
            self._add_task_dependency(task_id, dependency_id)
        
        # Add task to queue
        self._add_task_to_queue(task_id, priority)
        
        if self.config["detailed_logging"]:
            logger.info(f"Task {task_id} created with priority {priority}")
    
    async def on_task_completed(self, task_id: str, result: Any):
        """
        Handle task completed event.
        
        Args:
            task_id: Task ID
            result: Task result
        """
        # Skip if task not in metadata
        if task_id not in self.task_metadata:
            return
            
        # Update completed tasks set
        self.completed_tasks.add(task_id)
        
        # Get task metadata
        task_data = self.task_metadata[task_id]
        
        # Get worker ID if available
        worker_id = task_data.get("assigned_worker")
        
        if not worker_id:
            return
            
        # Update worker statistics
        if worker_id in self.worker_stats:
            self.worker_stats[worker_id]["tasks_completed"] += 1
            
            # Calculate execution time if start time is available
            if "start_time" in task_data:
                start_time = datetime.fromisoformat(task_data["start_time"])
                end_time = datetime.now()
                
                execution_time = (end_time - start_time).total_seconds()
                
                self.worker_stats[worker_id]["total_execution_time"] += execution_time
                
                tasks_completed = self.worker_stats[worker_id]["tasks_completed"]
                total_time = self.worker_stats[worker_id]["total_execution_time"]
                
                self.worker_stats[worker_id]["avg_execution_time"] = total_time / tasks_completed
                
                # Update worker task history
                self.worker_task_history[worker_id].append({
                    "task_id": task_id,
                    "execution_time": execution_time,
                    "completed_at": end_time.isoformat()
                })
                
                # Limit history size
                if len(self.worker_task_history[worker_id]) > self.config["history_window_size"]:
                    self.worker_task_history[worker_id].pop(0)
                
                # Update performance history
                model_type = task_data.get("model_type")
                
                if model_type and worker_id in self.worker_capabilities:
                    hardware_type = self.worker_capabilities.get(worker_id, {}).get("hardware_type")
                    
                    if hardware_type:
                        self._update_performance_history(model_type, hardware_type, execution_time)
        
        # Update worker load
        if worker_id in self.worker_current_load:
            self.worker_current_load[worker_id] = max(0, self.worker_current_load[worker_id] - 1)
        
        # Check if any dependent tasks are now ready
        if task_id in self.dependent_tasks:
            for dependent_id in self.dependent_tasks[task_id]:
                if dependent_id in self.task_dependencies:
                    # Check if all dependencies are completed
                    dependencies_met = all(dep in self.completed_tasks for dep in self.task_dependencies[dependent_id])
                    
                    if dependencies_met and dependent_id in self.task_priorities:
                        # Re-add to queue with current priority
                        priority = self.task_priorities[dependent_id]
                        self._add_task_to_queue(dependent_id, priority)
        
        if self.config["detailed_logging"]:
            logger.info(f"Task {task_id} completed by worker {worker_id}")
    
    async def on_task_failed(self, task_id: str, error: str):
        """
        Handle task failed event.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        # Skip if task not in metadata
        if task_id not in self.task_metadata:
            return
            
        # Get task metadata
        task_data = self.task_metadata[task_id]
        
        # Get worker ID if available
        worker_id = task_data.get("assigned_worker")
        
        if worker_id:
            # Update worker statistics
            if worker_id in self.worker_stats:
                self.worker_stats[worker_id]["tasks_failed"] += 1
            
            # Update worker load
            if worker_id in self.worker_current_load:
                self.worker_current_load[worker_id] = max(0, self.worker_current_load[worker_id] - 1)
        
        # Check if task should be retried
        retry_count = task_data.get("retry_count", 0)
        
        if retry_count < self.config["max_retry_attempts"]:
            # Increment retry count
            task_data["retry_count"] = retry_count + 1
            
            # Add back to queue with same priority
            if task_id in self.task_priorities:
                priority = self.task_priorities[task_id]
                self._add_task_to_queue(task_id, priority)
                
                logger.info(f"Retrying failed task {task_id} (attempt {retry_count + 1}/{self.config['max_retry_attempts']})")
        else:
            logger.warning(f"Task {task_id} failed after {retry_count} retry attempts: {error}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get the current scheduler status.
        
        Returns:
            Dictionary with scheduler status
        """
        return {
            "high_priority_queue_size": len(self.high_priority_queue),
            "normal_priority_queue_size": len(self.normal_priority_queue),
            "low_priority_queue_size": len(self.low_priority_queue),
            "completed_tasks": len(self.completed_tasks),
            "dependency_count": len(self.task_dependencies),
            "worker_count": len(self.worker_capabilities),
            "scheduling_algorithm": self.scheduling_algorithm,
            "running": self.scheduler_running
        }