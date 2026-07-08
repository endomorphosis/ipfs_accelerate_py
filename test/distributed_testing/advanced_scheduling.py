"""
Advanced Scheduling Algorithms for Distributed Testing Framework.

This module implements intelligent task scheduling algorithms for
the Distributed Testing Framework, including:
1. Priority-based scheduling
2. Resource-aware scheduling
3. Predictive scheduling (using ML predictions)
4. Dynamic adaptive scheduling 
5. Fairness-preserving scheduling

These algorithms optimize resource utilization, minimize task completion time,
and ensure fair allocation of resources among different task types and users.
"""

import logging
import heapq
import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Set
from collections import defaultdict
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("advanced_scheduling")

# Type aliases for better readability
TaskId = str
WorkerId = str
UserId = str
ResourceType = str
Priority = int  # Higher number = higher priority
Timestamp = float


class Task:
    """Representation of a task to be scheduled."""
    
    def __init__(
        self,
        task_id: TaskId,
        task_type: str,
        user_id: UserId,
        priority: Priority = 0,
        estimated_duration: float = 0.0,
        required_resources: Dict[ResourceType, float] = None,
        dependencies: List[TaskId] = None,
        metadata: Dict[str, Any] = None,
        submission_time: Optional[Timestamp] = None,
        deadline: Optional[Timestamp] = None,
    ):
        """
        Initialize a task.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task (e.g., 'test', 'benchmark')
            user_id: ID of the user who submitted the task
            priority: Task priority (higher value = higher priority)
            estimated_duration: Estimated execution time in seconds
            required_resources: Dictionary mapping resource types to required amount
            dependencies: List of task IDs that must complete before this task
            metadata: Additional task metadata
            submission_time: Time when the task was submitted (default: now)
            deadline: Optional deadline for task completion
        """
        self.task_id = task_id
        self.task_type = task_type
        self.user_id = user_id
        self.priority = priority
        self.estimated_duration = estimated_duration
        self.required_resources = required_resources or {}
        self.dependencies = dependencies or []
        self.metadata = metadata or {}
        self.submission_time = submission_time or time.time()
        self.deadline = deadline
        
        # Task state
        self.assigned_worker: Optional[WorkerId] = None
        self.start_time: Optional[Timestamp] = None
        self.end_time: Optional[Timestamp] = None
        self.status = "pending"  # pending, running, completed, failed
        self.attempt_count = 0
        self.result = None
        
    def __lt__(self, other):
        """Compare tasks by priority for priority queue usage."""
        # Primary: priority (higher first)
        if self.priority != other.priority:
            return self.priority > other.priority
        
        # Secondary: deadline (earlier first)
        if self.deadline is not None and other.deadline is not None:
            if self.deadline != other.deadline:
                return self.deadline < other.deadline
        elif self.deadline is not None:
            return True
        elif other.deadline is not None:
            return False
            
        # Tertiary: submission time (earlier first)
        return self.submission_time < other.submission_time
    
    def is_ready(self, completed_tasks: Set[TaskId]) -> bool:
        """Check if task is ready to run (all dependencies satisfied)."""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def remaining_time_to_deadline(self) -> Optional[float]:
        """Calculate time remaining until deadline."""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())
    
    def urgency_score(self) -> float:
        """Calculate an urgency score based on deadline and priority."""
        # Base score from priority
        score = float(self.priority) * 10.0
        
        # Add urgency based on deadline
        if self.deadline is not None:
            remaining = self.remaining_time_to_deadline()
            if remaining is not None:
                if remaining <= 0:
                    # Past deadline
                    score += 1000.0
                else:
                    # Approaching deadline - add up to 500 points
                    # The closer to deadline, the higher the score
                    estimated_time_needed = self.estimated_duration or 60.0
                    if remaining < estimated_time_needed * 2:
                        score += 500.0
                    elif remaining < estimated_time_needed * 5:
                        score += 200.0
                    elif remaining < estimated_time_needed * 10:
                        score += 100.0
        
        # Add waiting time factor (wait time in minutes)
        wait_time = (time.time() - self.submission_time) / 60.0
        score += min(100.0, wait_time)  # Cap at 100 points
        
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "user_id": self.user_id,
            "priority": self.priority,
            "estimated_duration": self.estimated_duration,
            "required_resources": self.required_resources,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "submission_time": self.submission_time,
            "deadline": self.deadline,
            "assigned_worker": self.assigned_worker,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "attempt_count": self.attempt_count,
        }


class Worker:
    """Representation of a worker in the system."""
    
    def __init__(
        self,
        worker_id: WorkerId,
        worker_type: str,
        capabilities: Dict[ResourceType, float],
        status: str = "idle",
        current_task: Optional[TaskId] = None,
        performance_metrics: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize a worker.
        
        Args:
            worker_id: Unique identifier for the worker
            worker_type: Type of worker (e.g., 'cpu', 'gpu')
            capabilities: Dictionary mapping resource types to capacity
            status: Current worker status
            current_task: ID of the currently executing task, if any
            performance_metrics: Performance metrics for this worker
            metadata: Additional worker metadata
        """
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.capabilities = capabilities
        self.status = status
        self.current_task = current_task
        self.performance_metrics = performance_metrics or {}
        self.metadata = metadata or {}
        
        # Track historical performance
        self.completed_tasks: List[TaskId] = []
        self.task_history: Dict[str, List[float]] = defaultdict(list)  # task_type -> durations
        self.task_failures: Dict[TaskId, str] = {}  # task_id -> error
        
        # Resource utilization and health
        self.resource_utilization: Dict[ResourceType, float] = {
            k: 0.0 for k in capabilities.keys()
        }
        self.health_score = 100.0
        self.last_heartbeat = time.time()
    
    def can_execute_task(self, task: Task) -> bool:
        """Check if this worker can execute the given task."""
        # Check if worker is available
        if self.status not in ["idle", "ready"]:
            return False
            
        # Check if worker has sufficient resources
        for resource_type, required_amount in task.required_resources.items():
            if resource_type not in self.capabilities:
                return False
            if self.capabilities[resource_type] < required_amount:
                return False
                
        return True
    
    def resource_match_score(self, task: Task) -> float:
        """
        Calculate how well this worker's resources match the task requirements.
        Higher score means better match.
        """
        if not self.can_execute_task(task):
            return 0.0
            
        # Calculate match score based on resource fit
        score = 100.0
        
        # Penalize over-provisioning
        for resource_type, required_amount in task.required_resources.items():
            available = self.capabilities.get(resource_type, 0.0)
            # The closer the match, the better (don't waste resources)
            # Perfect match = 1.0, 2x oversized = 0.5, 10x oversized = 0.1
            ratio = min(1.0, required_amount / max(1e-6, available))
            score *= ratio
            
        return score
    
    def expected_task_duration(self, task: Task) -> float:
        """
        Estimate how long this task will take on this worker.
        Uses historical data when available.
        """
        # Use task's own estimate if we have no history
        if not self.task_history.get(task.task_type):
            return task.estimated_duration or 60.0
        
        # Use historical average for this task type
        durations = self.task_history[task.task_type]
        if not durations:
            return task.estimated_duration or 60.0
            
        # Use average of recent executions
        return sum(durations[-5:]) / min(5, len(durations[-5:]))
    
    def update_task_history(self, task: Task, duration: float, success: bool) -> None:
        """Update the task execution history for this worker."""
        if success:
            self.task_history[task.task_type].append(duration)
            self.completed_tasks.append(task.task_id)
        else:
            self.task_failures[task.task_id] = task.metadata.get("error", "Unknown error")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert worker to dictionary representation."""
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "capabilities": self.capabilities,
            "status": self.status,
            "current_task": self.current_task,
            "performance_metrics": self.performance_metrics,
            "metadata": self.metadata,
            "resource_utilization": self.resource_utilization,
            "health_score": self.health_score,
            "last_heartbeat": self.last_heartbeat,
        }


class AdvancedScheduler:
    """
    Advanced scheduler for Distributed Testing Framework.
    Implements multiple scheduling algorithms with adaptive strategy selection.
    """
    
    # Constants for scheduling algorithm types
    ALGORITHM_PRIORITY = "priority"
    ALGORITHM_RESOURCE_AWARE = "resource_aware"
    ALGORITHM_PREDICTIVE = "predictive"
    ALGORITHM_ADAPTIVE = "adaptive"
    ALGORITHM_FAIR = "fair"
    
    def __init__(
        self,
        algorithm: str = "adaptive",
        fairness_window: int = 100,
        prediction_confidence_threshold: float = 0.7,
        resource_match_weight: float = 0.7,
        user_fair_share_enabled: bool = True,
        adaptive_interval: int = 50,
        preemption_enabled: bool = True,
        max_task_retries: int = 3,
        performance_history_size: int = 1000,
    ):
        """
        Initialize advanced scheduler.
        
        Args:
            algorithm: Scheduling algorithm to use 
                (priority, resource_aware, predictive, adaptive, fair)
            fairness_window: Number of tasks to consider for fairness calculations
            prediction_confidence_threshold: Minimum confidence for predictions
            resource_match_weight: Weight for resource matching score (0-1)
            user_fair_share_enabled: Whether to enforce fair share among users
            adaptive_interval: Number of tasks between algorithm evaluations
            preemption_enabled: Whether to allow task preemption
            max_task_retries: Maximum number of retries for failed tasks
            performance_history_size: Size of performance history to maintain
        """
        self.algorithm = algorithm
        self.fairness_window = fairness_window
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.resource_match_weight = resource_match_weight
        self.user_fair_share_enabled = user_fair_share_enabled
        self.adaptive_interval = adaptive_interval
        self.preemption_enabled = preemption_enabled
        self.max_task_retries = max_task_retries
        self.performance_history_size = performance_history_size
        
        # Task queues and tracking
        self.tasks: Dict[TaskId, Task] = {}
        self.task_queue: List[Task] = []  # Priority queue of pending tasks
        self.running_tasks: Dict[WorkerId, Task] = {}
        self.completed_tasks: Set[TaskId] = set()
        self.failed_tasks: Dict[TaskId, str] = {}  # task_id -> error
        
        # Worker management
        self.workers: Dict[WorkerId, Worker] = {}
        self.available_workers: Set[WorkerId] = set()
        
        # Performance tracking
        self.algorithm_performance: Dict[str, List[float]] = defaultdict(list)
        self.task_completion_times: List[float] = []
        self.user_task_counts: Dict[UserId, int] = defaultdict(int)
        self.task_type_counts: Dict[str, int] = defaultdict(int)
        
        # Predictive scheduling data
        self.task_duration_predictions: Dict[str, float] = {}  # task_type -> predicted duration
        self.worker_speed_factors: Dict[WorkerId, float] = {}  # worker -> speed factor
        
        # Adaptive scheduling state
        self.tasks_since_adaptation = 0
        self.current_best_algorithm = self.ALGORITHM_ADAPTIVE
        
        logger.info(f"Advanced scheduler initialized with algorithm: {algorithm}")
    
    def add_task(self, task: Task) -> bool:
        """
        Add a task to the scheduling queue.
        
        Args:
            task: Task to add to the queue
            
        Returns:
            True if task was added successfully
        """
        if task.task_id in self.tasks:
            logger.warning(f"Task {task.task_id} already in queue, skipping")
            return False
            
        # Add to task dictionary
        self.tasks[task.task_id] = task
        
        # Add to priority queue
        heapq.heappush(self.task_queue, task)
        
        # Update tracking counters
        self.user_task_counts[task.user_id] += 1
        self.task_type_counts[task.task_type] += 1
        
        logger.info(f"Added task {task.task_id} to queue (priority: {task.priority})")
        return True
    
    def add_worker(self, worker: Worker) -> bool:
        """
        Add or update a worker in the system.
        
        Args:
            worker: Worker to add or update
            
        Returns:
            True if worker was added/updated successfully
        """
        is_new = worker.worker_id not in self.workers
        self.workers[worker.worker_id] = worker
        
        # Update available workers set if worker is idle
        if worker.status in ["idle", "ready"]:
            self.available_workers.add(worker.worker_id)
        elif worker.worker_id in self.available_workers:
            self.available_workers.remove(worker.worker_id)
            
        if is_new:
            logger.info(f"Added new worker {worker.worker_id} ({worker.worker_type})")
        else:
            logger.debug(f"Updated worker {worker.worker_id} (status: {worker.status})")
            
        return True
    
    def remove_worker(self, worker_id: WorkerId) -> bool:
        """
        Remove a worker from the system.
        
        Args:
            worker_id: ID of worker to remove
            
        Returns:
            True if worker was removed
        """
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found, cannot remove")
            return False
            
        # If worker has a running task, mark it as failed
        if worker_id in self.running_tasks:
            task = self.running_tasks[worker_id]
            self._handle_task_failure(task, f"Worker {worker_id} removed while task running")
            
        # Remove from workers and available workers
        if worker_id in self.available_workers:
            self.available_workers.remove(worker_id)
        del self.workers[worker_id]
        
        logger.info(f"Removed worker {worker_id}")
        return True
    
    def update_worker_status(self, worker_id: WorkerId, status: str) -> bool:
        """
        Update a worker's status.
        
        Args:
            worker_id: ID of worker to update
            status: New status (idle, busy, offline, etc.)
            
        Returns:
            True if status was updated
        """
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found, cannot update status")
            return False
            
        worker = self.workers[worker_id]
        old_status = worker.status
        worker.status = status
        
        # Update available workers set
        if status in ["idle", "ready"]:
            self.available_workers.add(worker_id)
        elif worker_id in self.available_workers:
            self.available_workers.remove(worker_id)
            
        logger.debug(f"Updated worker {worker_id} status: {old_status} -> {status}")
        return True
    
    def schedule_tasks(self) -> List[Tuple[TaskId, WorkerId]]:
        """
        Schedule pending tasks to available workers.
        
        Returns:
            List of (task_id, worker_id) assignments
        """
        if not self.available_workers:
            logger.debug("No available workers for scheduling")
            return []
            
        if not self.task_queue:
            logger.debug("No pending tasks for scheduling")
            return []
            
        # Check if we need to adjust our scheduling algorithm (adaptive)
        if self.algorithm == self.ALGORITHM_ADAPTIVE:
            self._maybe_adapt_scheduling_algorithm()
            
        # Schedule tasks using the selected algorithm
        if self.current_best_algorithm == self.ALGORITHM_PRIORITY:
            assignments = self._schedule_priority_based()
        elif self.current_best_algorithm == self.ALGORITHM_RESOURCE_AWARE:
            assignments = self._schedule_resource_aware()
        elif self.current_best_algorithm == self.ALGORITHM_PREDICTIVE:
            assignments = self._schedule_predictive()
        elif self.current_best_algorithm == self.ALGORITHM_FAIR:
            assignments = self._schedule_fair()
        else:
            # Default to resource-aware scheduling
            assignments = self._schedule_resource_aware()
            
        # Record algorithm performance
        if assignments:
            algorithm = self.current_best_algorithm or self.algorithm
            self.algorithm_performance[algorithm].append(len(assignments))
            
        # Process assignments
        for task_id, worker_id in assignments:
            self._assign_task_to_worker(task_id, worker_id)
            
        return assignments
    
    def _assign_task_to_worker(self, task_id: TaskId, worker_id: WorkerId) -> bool:
        """
        Assign a task to a worker.
        
        Args:
            task_id: ID of task to assign
            worker_id: ID of worker to assign task to
            
        Returns:
            True if assignment was successful
        """
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found, cannot assign")
            return False
            
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found, cannot assign task")
            return False
            
        task = self.tasks[task_id]
        worker = self.workers[worker_id]
        
        # Update task state
        task.assigned_worker = worker_id
        task.start_time = time.time()
        task.status = "running"
        task.attempt_count += 1
        
        # Update worker state
        worker.status = "busy"
        worker.current_task = task_id
        if worker_id in self.available_workers:
            self.available_workers.remove(worker_id)
            
        # Update running tasks
        self.running_tasks[worker_id] = task
        
        # Remove task from queue if it's still there
        # This is O(n) but should be rare as we typically remove from the front
        new_queue = [t for t in self.task_queue if t.task_id != task_id]
        if len(new_queue) != len(self.task_queue):
            self.task_queue = new_queue
            heapq.heapify(self.task_queue)
            
        logger.info(f"Assigned task {task_id} to worker {worker_id}")
        return True
    
    def complete_task(self, worker_id: WorkerId, success: bool, result: Any = None) -> Optional[TaskId]:
        """
        Mark a task as completed or failed.
        
        Args:
            worker_id: ID of worker that completed the task
            success: Whether the task completed successfully
            result: Optional result data
            
        Returns:
            Task ID that was completed/failed, or None if worker not found
        """
        if worker_id not in self.running_tasks:
            logger.warning(f"Worker {worker_id} has no running task to complete")
            return None
            
        task = self.running_tasks[worker_id]
        task_id = task.task_id
        
        # Calculate task duration
        duration = time.time() - (task.start_time or time.time())
        
        # Update worker status
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.status = "idle"
            worker.current_task = None
            worker.update_task_history(task, duration, success)
            self.available_workers.add(worker_id)
            
        # Update task state
        if success:
            task.status = "completed"
            task.end_time = time.time()
            task.result = result
            self.completed_tasks.add(task_id)
            self.task_completion_times.append(duration)
            logger.info(f"Task {task_id} completed successfully (duration: {duration:.2f}s)")
        else:
            self._handle_task_failure(task, result)
            
        # Remove from running tasks
        del self.running_tasks[worker_id]
        
        return task_id
    
    def _handle_task_failure(self, task: Task, error: Any = None) -> None:
        """Handle a task failure, potentially requeueing for retry."""
        task.status = "failed"
        task.end_time = time.time()
        task.result = error
        
        # Check if we should retry
        if task.attempt_count < self.max_task_retries:
            # Reset task for retry
            task.assigned_worker = None
            task.start_time = None
            task.end_time = None
            task.status = "pending"
            
            # Re-add to queue with adjusted priority
            # Failed tasks get a small priority boost to prevent starvation
            task.priority = max(task.priority, task.priority + 1)
            heapq.heappush(self.task_queue, task)
            
            logger.info(f"Task {task.task_id} failed, requeueing for retry "
                       f"(attempt {task.attempt_count}/{self.max_task_retries})")
        else:
            # Mark as permanently failed
            self.failed_tasks[task.task_id] = str(error)
            logger.warning(f"Task {task.task_id} failed permanently after "
                          f"{task.attempt_count} attempts: {error}")
    
    def preempt_task(self, task_id: TaskId) -> bool:
        """
        Preempt a running task to run a higher priority task.
        
        Args:
            task_id: ID of task to preempt
            
        Returns:
            True if task was preempted
        """
        if not self.preemption_enabled:
            return False
            
        # Find worker running this task
        worker_id = None
        for wid, task in self.running_tasks.items():
            if task.task_id == task_id:
                worker_id = wid
                break
                
        if not worker_id:
            logger.warning(f"Task {task_id} not running, cannot preempt")
            return False
            
        # Mark task as preempted and requeue
        task = self.tasks[task_id]
        task.status = "preempted"
        task.assigned_worker = None
        task.end_time = time.time()
        
        # Re-add to queue
        heapq.heappush(self.task_queue, task)
        
        # Update worker
        worker = self.workers[worker_id]
        worker.status = "idle"
        worker.current_task = None
        self.available_workers.add(worker_id)
        
        # Remove from running tasks
        del self.running_tasks[worker_id]
        
        logger.info(f"Preempted task {task_id} on worker {worker_id}")
        return True
    
    def _schedule_priority_based(self) -> List[Tuple[TaskId, WorkerId]]:
        """
        Schedule tasks based on priority only.
        Simple algorithm that assigns highest priority tasks to any available worker.
        
        Returns:
            List of (task_id, worker_id) assignments
        """
        assignments = []
        
        # Create a copy of the task queue for iteration
        # We'll use a heap to ensure we get tasks in priority order
        pending_tasks = self.task_queue.copy()
        heapq.heapify(pending_tasks)
        
        # Create a list of available workers
        available_workers = list(self.available_workers)
        
        # Simple greedy assignment - highest priority tasks first
        while pending_tasks and available_workers:
            task = heapq.heappop(pending_tasks)
            worker_id = available_workers.pop(0)
            
            # Check if this task can run on this worker
            worker = self.workers[worker_id]
            if worker.can_execute_task(task):
                assignments.append((task.task_id, worker_id))
            else:
                # Put the worker back and try next task
                available_workers.append(worker_id)
                
        return assignments
    
    def _schedule_resource_aware(self) -> List[Tuple[TaskId, WorkerId]]:
        """
        Schedule tasks based on resource requirements and worker capabilities.
        
        Returns:
            List of (task_id, worker_id) assignments
        """
        assignments = []
        
        # Create a copy of the task queue for iteration
        pending_tasks = self.task_queue.copy()
        heapq.heapify(pending_tasks)
        
        # Get available workers
        available_workers = list(self.available_workers)
        
        # Calculate resource match scores for all task-worker pairs
        scores = []
        for task in pending_tasks:
            if not task.is_ready(self.completed_tasks):
                continue
                
            for worker_id in available_workers:
                worker = self.workers[worker_id]
                
                if worker.can_execute_task(task):
                    # Calculate score combining priority and resource match
                    priority_score = task.urgency_score()
                    resource_score = worker.resource_match_score(task)
                    
                    # Weight resource score vs priority
                    combined_score = (
                        self.resource_match_weight * resource_score +
                        (1 - self.resource_match_weight) * priority_score
                    )
                    
                    scores.append((combined_score, task, worker_id))
        
        # Sort scores in descending order (highest score first)
        scores.sort(reverse=True)
        
        # Assign tasks to workers
        assigned_workers = set()
        assigned_tasks = set()
        
        for _, task, worker_id in scores:
            # Skip if this worker or task is already assigned
            if worker_id in assigned_workers or task.task_id in assigned_tasks:
                continue
                
            # Make assignment
            assignments.append((task.task_id, worker_id))
            assigned_workers.add(worker_id)
            assigned_tasks.add(task.task_id)
            
            # Break if we've assigned all available workers
            if len(assigned_workers) == len(available_workers):
                break
                
        return assignments
    
    def _schedule_predictive(self) -> List[Tuple[TaskId, WorkerId]]:
        """
        Schedule tasks using predictive metrics for optimal worker assignment.
        
        Returns:
            List of (task_id, worker_id) assignments
        """
        assignments = []
        
        # Create a copy of the task queue for iteration
        pending_tasks = [t for t in self.task_queue if t.is_ready(self.completed_tasks)]
        
        # Get available workers
        available_workers = list(self.available_workers)
        
        # Calculate expected completion times for all task-worker pairs
        completion_estimates = []
        for task in pending_tasks:
            for worker_id in available_workers:
                worker = self.workers[worker_id]
                
                if worker.can_execute_task(task):
                    # Predict how long this task will take on this worker
                    expected_duration = worker.expected_task_duration(task)
                    
                    # Adjust by confidence factor based on historical data
                    history_count = len(worker.task_history.get(task.task_type, []))
                    confidence = min(1.0, history_count / 10.0)  # 10+ samples = full confidence
                    
                    # If confidence too low, weight less heavily
                    if confidence < self.prediction_confidence_threshold:
                        # Blend with task's own estimate based on confidence
                        expected_duration = (
                            confidence * expected_duration +
                            (1 - confidence) * (task.estimated_duration or 60.0)
                        )
                    
                    # Calculate utility value (negative because we want to minimize time)
                    # but also consider task priority
                    utility = -expected_duration * (1.0 / (task.priority + 1))
                    
                    completion_estimates.append((utility, task, worker_id))
        
        # Sort by utility (highest utility first, which means lowest time considering priority)
        completion_estimates.sort()
        
        # Make assignments
        assigned_workers = set()
        assigned_tasks = set()
        
        for _, task, worker_id in completion_estimates:
            # Skip if this worker or task is already assigned
            if worker_id in assigned_workers or task.task_id in assigned_tasks:
                continue
                
            # Make assignment
            assignments.append((task.task_id, worker_id))
            assigned_workers.add(worker_id)
            assigned_tasks.add(task.task_id)
            
            # Break if we've assigned all available workers
            if len(assigned_workers) == len(available_workers):
                break
                
        return assignments
    
    def _schedule_fair(self) -> List[Tuple[TaskId, WorkerId]]:
        """
        Schedule tasks with fairness considerations across users and task types.
        
        Returns:
            List of (task_id, worker_id) assignments
        """
        assignments = []
        
        # Get pending tasks and available workers
        pending_tasks = [t for t in self.task_queue if t.is_ready(self.completed_tasks)]
        available_workers = list(self.available_workers)
        
        # Calculate fair shares for users and task types
        total_tasks = sum(self.user_task_counts.values())
        user_fair_shares = {}
        
        for user_id, count in self.user_task_counts.items():
            user_fair_shares[user_id] = count / max(1, total_tasks)
            
        # Calculate actual shares (tasks completed)
        user_actual_shares = defaultdict(float)
        for task_id in self.completed_tasks:
            if task_id in self.tasks:
                user_id = self.tasks[task_id].user_id
                user_actual_shares[user_id] += 1
                
        # Normalize actual shares
        total_completed = sum(user_actual_shares.values())
        if total_completed > 0:
            for user_id in user_actual_shares:
                user_actual_shares[user_id] /= total_completed
                
        # Calculate fairness deficits (positive means user needs more resources)
        user_deficits = {}
        for user_id in set(user_fair_shares.keys()) | set(user_actual_shares.keys()):
            fair_share = user_fair_shares.get(user_id, 0.0)
            actual_share = user_actual_shares.get(user_id, 0.0)
            user_deficits[user_id] = fair_share - actual_share
            
        # Score tasks based on fairness and priority
        task_scores = []
        for task in pending_tasks:
            # Start with task's base priority and urgency
            base_score = task.urgency_score()
            
            # Adjust score based on user's fairness deficit
            # If deficit is positive, user is under-served and gets higher priority
            user_deficit = user_deficits.get(task.user_id, 0.0)
            fairness_score = 100.0 * user_deficit if user_deficit > 0 else 0.0
            
            # Combine scores
            if self.user_fair_share_enabled:
                combined_score = base_score + fairness_score
            else:
                combined_score = base_score
                
            task_scores.append((combined_score, task))
            
        # Sort tasks by combined score (highest first)
        task_scores.sort(reverse=True)
        
        # Now find best worker for each task in order
        assigned_workers = set()
        for _, task in task_scores:
            if not available_workers:
                break
                
            # Find best worker for this task
            best_worker = None
            best_score = -1
            
            for worker_id in available_workers:
                if worker_id in assigned_workers:
                    continue
                    
                worker = self.workers[worker_id]
                if worker.can_execute_task(task):
                    # Calculate match score for this worker
                    score = worker.resource_match_score(task)
                    if score > best_score:
                        best_score = score
                        best_worker = worker_id
            
            # If we found a suitable worker, make the assignment
            if best_worker:
                assignments.append((task.task_id, best_worker))
                assigned_workers.add(best_worker)
                
        return assignments
    
    def _maybe_adapt_scheduling_algorithm(self) -> None:
        """
        Periodically evaluate and adjust scheduling algorithm based on performance.
        This is the core of the adaptive scheduling feature.
        """
        self.tasks_since_adaptation += 1
        
        # Only adapt every adaptive_interval tasks
        if self.tasks_since_adaptation < self.adaptive_interval:
            return
            
        self.tasks_since_adaptation = 0
        
        # Not enough data to evaluate algorithms
        if not all(len(perfs) > 5 for perfs in self.algorithm_performance.values()):
            # Try each algorithm at least once
            if not self.algorithm_performance.get(self.ALGORITHM_PRIORITY):
                self.current_best_algorithm = self.ALGORITHM_PRIORITY
            elif not self.algorithm_performance.get(self.ALGORITHM_RESOURCE_AWARE):
                self.current_best_algorithm = self.ALGORITHM_RESOURCE_AWARE
            elif not self.algorithm_performance.get(self.ALGORITHM_PREDICTIVE):
                self.current_best_algorithm = self.ALGORITHM_PREDICTIVE
            elif not self.algorithm_performance.get(self.ALGORITHM_FAIR):
                self.current_best_algorithm = self.ALGORITHM_FAIR
            return
            
        # Calculate average tasks assigned for each algorithm
        # and their variance (stability)
        algorithm_scores = {}
        for alg, perfs in self.algorithm_performance.items():
            # Use only recent performance data
            recent_perfs = perfs[-20:]
            if not recent_perfs:
                continue
                
            avg_tasks = sum(recent_perfs) / len(recent_perfs)
            variance = sum((x - avg_tasks) ** 2 for x in recent_perfs) / len(recent_perfs)
            
            # Score is average with a small penalty for high variance (instability)
            score = avg_tasks - (0.1 * variance)
            algorithm_scores[alg] = score
            
        # Find algorithm with best score
        if algorithm_scores:
            self.current_best_algorithm = max(
                algorithm_scores.items(), 
                key=lambda x: x[1]
            )[0]
            
            logger.info(f"Adaptive scheduler selected algorithm: {self.current_best_algorithm} "
                       f"(scores: {algorithm_scores})")
    
    def get_task_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the current task queue."""
        stats = {
            "total_tasks": len(self.tasks),
            "pending_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "avg_completion_time": (
                sum(self.task_completion_times) / max(1, len(self.task_completion_times))
                if self.task_completion_times else 0
            ),
            "users": {
                user_id: count for user_id, count in self.user_task_counts.items()
            },
            "task_types": {
                task_type: count for task_type, count in self.task_type_counts.items()
            },
            "current_algorithm": self.current_best_algorithm or self.algorithm,
        }
        return stats
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics about workers."""
        stats = {
            "total_workers": len(self.workers),
            "available_workers": len(self.available_workers),
            "worker_types": {},
            "resource_utilization": {},
        }
        
        # Count worker types
        for worker in self.workers.values():
            wtype = worker.worker_type
            stats["worker_types"][wtype] = stats["worker_types"].get(wtype, 0) + 1
            
            # Track resource utilization
            for res_type, usage in worker.resource_utilization.items():
                if res_type not in stats["resource_utilization"]:
                    stats["resource_utilization"][res_type] = {
                        "total": 0,
                        "used": 0,
                        "percent": 0,
                    }
                    
                stats["resource_utilization"][res_type]["total"] += worker.capabilities.get(res_type, 0)
                stats["resource_utilization"][res_type]["used"] += usage
                
        # Calculate utilization percentages
        for res_type, data in stats["resource_utilization"].items():
            if data["total"] > 0:
                data["percent"] = (data["used"] / data["total"]) * 100
                
        return stats
    
    def get_algorithm_performance(self) -> Dict[str, Any]:
        """Get performance statistics for each scheduling algorithm."""
        stats = {}
        
        for alg, perfs in self.algorithm_performance.items():
            if not perfs:
                continue
                
            # Calculate performance metrics
            avg = sum(perfs) / len(perfs)
            recent_avg = sum(perfs[-10:]) / min(10, len(perfs))
            max_perf = max(perfs)
            min_perf = min(perfs)
            
            # Calculate a trend (positive = improving)
            if len(perfs) >= 10:
                first_half = perfs[-20:-10]
                second_half = perfs[-10:]
                if first_half and second_half:
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)
                    trend = second_avg - first_avg
                else:
                    trend = 0
            else:
                trend = 0
                
            stats[alg] = {
                "average_tasks_assigned": avg,
                "recent_average": recent_avg,
                "max": max_perf,
                "min": min_perf,
                "trend": trend,
                "samples": len(perfs),
            }
            
        return stats


def create_advanced_scheduler(config_file: Optional[str] = None, **kwargs) -> AdvancedScheduler:
    """
    Create an instance of the AdvancedScheduler with optional config file.
    
    Args:
        config_file: Path to JSON configuration file
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured AdvancedScheduler instance
    """
    import json
    
    # Start with default config
    config = {}
    
    # Load from file if provided
    if config_file:
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger.error(f"Error loading scheduler config from {config_file}: {e}")
    
    # Override with kwargs
    config.update(kwargs)
    
    # Create and return scheduler
    return AdvancedScheduler(**config)


if __name__ == "__main__":
    # Example usage of the advanced scheduler
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Advanced Task Scheduler for DTF")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--algorithm", default="adaptive", 
                       choices=["priority", "resource_aware", "predictive", "adaptive", "fair"],
                       help="Scheduling algorithm to use")
    parser.add_argument("--workers", type=int, default=5, help="Number of test workers to create")
    parser.add_argument("--tasks", type=int, default=20, help="Number of test tasks to create")
    
    args = parser.parse_args()
    
    # Create scheduler
    scheduler = create_advanced_scheduler(
        config_file=args.config,
        algorithm=args.algorithm
    )
    
    # For testing: Create some synthetic workers and tasks
    for i in range(args.workers):
        worker_type = random.choice(["cpu", "gpu", "webgpu", "webnn"])
        capabilities = {
            "cpu": random.uniform(1, 8),
            "memory": random.uniform(1, 16),
            "gpu": random.uniform(0, 2) if worker_type in ["gpu", "webgpu"] else 0,
        }
        
        worker = Worker(
            worker_id=f"worker-{i}",
            worker_type=worker_type,
            capabilities=capabilities,
            status="idle"
        )
        scheduler.add_worker(worker)
        
    for i in range(args.tasks):
        task_type = random.choice(["test", "benchmark", "validation", "analysis"])
        priority = random.randint(0, 5)
        user_id = f"user-{random.randint(1, 3)}"
        
        # Random resource requirements
        required_resources = {
            "cpu": random.uniform(0.1, 4),
            "memory": random.uniform(0.5, 8),
        }
        
        # Some tasks need GPU
        if random.random() < 0.3:
            required_resources["gpu"] = random.uniform(0.1, 1.0)
            
        task = Task(
            task_id=f"task-{i}",
            task_type=task_type,
            user_id=user_id,
            priority=priority,
            estimated_duration=random.uniform(10, 300),
            required_resources=required_resources,
        )
        scheduler.add_task(task)
        
    # Run scheduling
    print("\nRunning scheduling with algorithm:", args.algorithm)
    assignments = scheduler.schedule_tasks()
    
    print(f"\nMade {len(assignments)} assignments:")
    for task_id, worker_id in assignments:
        task = scheduler.tasks[task_id]
        worker = scheduler.workers[worker_id]
        print(f"  {task_id} ({task.task_type}, priority={task.priority}) -> {worker_id} ({worker.worker_type})")
        
    # Complete some tasks randomly (success or failure)
    print("\nSimulating task completions:")
    for worker_id in list(scheduler.running_tasks.keys()):
        if random.random() < 0.8:  # 80% success rate
            task_id = scheduler.complete_task(worker_id, True, {"result": "success"})
            print(f"  {task_id} completed successfully on {worker_id}")
        else:
            task_id = scheduler.complete_task(worker_id, False, "Simulated failure")
            print(f"  {task_id} failed on {worker_id}")
            
    # Show statistics
    print("\nTask Queue Statistics:")
    stats = scheduler.get_task_queue_stats()
    print(json.dumps(stats, indent=2))
    
    print("\nWorker Statistics:")
    stats = scheduler.get_worker_stats()
    print(json.dumps(stats, indent=2))
    
    # If we've run multiple algorithms in adaptive mode, show performance
    if args.algorithm == "adaptive":
        print("\nAlgorithm Performance:")
        perf = scheduler.get_algorithm_performance()
        print(json.dumps(perf, indent=2))