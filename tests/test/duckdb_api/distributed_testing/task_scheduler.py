#!/usr/bin/env python3
"""
Distributed Testing Framework - Task Scheduler

This module implements the advanced task scheduling and distribution logic for the
distributed testing framework. It's responsible for:

- Hardware-aware task assignment
- Test-specific requirements matching
- Priority-based scheduling
- Workload balancing across workers
- Task dependency management
- Resource optimization

Usage:
    This module is used by the coordinator server to schedule tasks efficiently
    across available worker nodes.
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("task_scheduler")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Task status constants
TASK_STATUS_QUEUED = "queued"
TASK_STATUS_ASSIGNED = "assigned"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_TIMED_OUT = "timed_out"
TASK_STATUS_CANCELED = "canceled"

# Worker status constants
WORKER_STATUS_REGISTERED = "registered"
WORKER_STATUS_ACTIVE = "active"
WORKER_STATUS_BUSY = "busy"
WORKER_STATUS_UNAVAILABLE = "unavailable"
WORKER_STATUS_DISCONNECTED = "disconnected"

# Matching algorithm constants
MATCH_ALGORITHM_EXACT = "exact"     # Require exact match of all requirements
MATCH_ALGORITHM_WEIGHTED = "weighted"  # Use weighted scoring for requirements
MATCH_ALGORITHM_FLEXIBLE = "flexible"  # Allow some requirements to be missing

# Task type to priority defaults
DEFAULT_TYPE_PRIORITIES = {
    "benchmark": 5,
    "test": 3,
    "command": 2,
    "critical": 1,
    "maintenance": 10
}

class TaskScheduler:
    """Intelligent task scheduler for the distributed testing framework."""
    
    def __init__(self, db_manager=None):
        """Initialize the task scheduler.
        
        Args:
            db_manager: Optional database manager for task persistence
        """
        self.db_manager = db_manager
        self.task_queue = []  # [(priority, create_time, task_id, task)]
        self.running_tasks = {}  # task_id -> worker_id
        self.task_lock = threading.Lock()
        self.worker_performance = {}  # worker_id -> performance metrics
        self.task_stats = {}  # task_type -> performance statistics
        self.worker_suitability_cache = {}  # (worker_id, task_type) -> suitability score
        self.match_algorithm = MATCH_ALGORITHM_WEIGHTED
        
        # Task dependencies
        self.task_dependencies = {}  # task_id -> set of dependent task_ids
        self.reverse_dependencies = {}  # task_id -> set of tasks that depend on this
        
        # Configuration
        self.config = {
            "max_retries": 3,
            "retry_delay_seconds": 60,
            "timeout_seconds": 3600,  # 1 hour default timeout
            "resource_factor_weight": 0.7,  # Weight for resource efficiency in scoring
            "performance_weight": 0.3,  # Weight for historical performance in scoring
            "type_priorities": DEFAULT_TYPE_PRIORITIES.copy(),
            "adaptive_scheduling": True,  # Enable adaptive scheduling based on performance
            "max_concurrent_tasks_per_worker": 1,  # Default is one task per worker
            "preemption_enabled": False,  # Whether to enable task preemption
        }
        
        logger.info("Task scheduler initialized")
    
    def configure(self, config_updates: Dict[str, Any]):
        """Update the scheduler configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.config.update(config_updates)
        logger.info(f"Task scheduler configuration updated: {config_updates}")
    
    def add_task(self, task_id: str, task_type: str, priority: int, 
                config: Dict[str, Any], requirements: Dict[str, Any],
                dependencies: List[str] = None) -> str:
        """Add a task to the scheduling queue.
        
        Args:
            task_id: Unique identifier for the task (or None to generate)
            task_type: Type of task (benchmark, test, etc.)
            priority: Priority of the task (lower is higher priority)
            config: Configuration for the task
            requirements: Hardware requirements for the task
            dependencies: List of task IDs that this task depends on
            
        Returns:
            Task ID
        """
        # Apply type-based priority adjustment if no explicit priority
        if priority is None and task_type in self.config["type_priorities"]:
            priority = self.config["type_priorities"][task_type]
            
        task = {
            "task_id": task_id,
            "type": task_type,
            "priority": priority,
            "status": TASK_STATUS_QUEUED,
            "create_time": datetime.now(),
            "config": config,
            "requirements": requirements,
            "attempts": 0,
            "timeout_seconds": config.get("timeout_seconds", self.config["timeout_seconds"]),
            "max_retries": config.get("max_retries", self.config["max_retries"]),
            "retry_delay": config.get("retry_delay_seconds", self.config["retry_delay_seconds"]),
            "dependencies": dependencies or []
        }
        
        # Add to database if available
        if self.db_manager:
            self.db_manager.add_task(task_id, task_type, priority, config, requirements)
        
        # Add to queue if no dependencies or all dependencies completed
        with self.task_lock:
            # Store dependencies
            if dependencies:
                self.task_dependencies[task_id] = set(dependencies)
                
                # Update reverse dependencies
                for dep_id in dependencies:
                    if dep_id not in self.reverse_dependencies:
                        self.reverse_dependencies[dep_id] = set()
                    self.reverse_dependencies[dep_id].add(task_id)
                
                # Check if all dependencies are satisfied
                if not self._are_dependencies_satisfied(task_id):
                    logger.info(f"Task {task_id} added but waiting for dependencies")
                    return task_id
            
            # Add to queue
            create_time = task["create_time"]
            self.task_queue.append((priority, create_time, task_id, task))
            self.task_queue.sort()  # Sort by priority, then create_time
        
        logger.info(f"Task {task_id} added to queue with priority {priority}")
        return task_id
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        if task_id not in self.task_dependencies:
            return True
            
        for dep_id in self.task_dependencies[task_id]:
            if dep_id in self.running_tasks:
                return False
                
            # Check if dependency is in queue
            for _, _, queued_task_id, _ in self.task_queue:
                if queued_task_id == dep_id:
                    return False
                    
        return True
    
    def add_task_batch(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Add multiple tasks to the queue efficiently.
        
        Args:
            tasks: List of task configurations
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        with self.task_lock:
            for task_config in tasks:
                task_id = task_config.get("task_id", f"task_{len(self.task_queue)}")
                task_type = task_config.get("type", "benchmark")
                priority = task_config.get("priority", self.config["type_priorities"].get(task_type, 5))
                config = task_config.get("config", {})
                requirements = task_config.get("requirements", {})
                dependencies = task_config.get("dependencies", [])
                
                task_ids.append(self.add_task(task_id, task_type, priority, config, requirements, dependencies))
                
        logger.info(f"Added batch of {len(tasks)} tasks")
        return task_ids
    
    def get_next_task(self, worker_id: str, 
                     worker_capabilities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get the next task for a worker based on capabilities and intelligent matching.
        
        Args:
            worker_id: ID of the worker
            worker_capabilities: Capabilities of the worker
            
        Returns:
            Task dict if a suitable task is found, None otherwise
        """
        with self.task_lock:
            if not self.task_queue:
                return None
                
            # Get currently running tasks for this worker
            worker_running_tasks = sum(1 for w_id in self.running_tasks.values() if w_id == worker_id)
            
            # Check if worker is at capacity
            if worker_running_tasks >= self.config["max_concurrent_tasks_per_worker"]:
                logger.debug(f"Worker {worker_id} is at capacity with {worker_running_tasks} tasks")
                return None
            
            # Find best matching task
            best_task_index = None
            best_task = None
            best_score = -1
            
            for i, (_, _, task_id, task) in enumerate(self.task_queue):
                # Skip tasks that have dependencies
                if not self._are_dependencies_satisfied(task_id):
                    continue
                    
                # Calculate match score
                score = self._calculate_worker_task_match(worker_id, worker_capabilities, task)
                
                if score > best_score:
                    best_score = score
                    best_task_index = i
                    best_task = task
                    
            if best_task_index is not None and best_score > 0:
                # Remove from queue
                self.task_queue.pop(best_task_index)
                
                # Mark as assigned
                best_task["status"] = TASK_STATUS_ASSIGNED
                best_task["worker_id"] = worker_id
                best_task["start_time"] = datetime.now()
                best_task["attempts"] += 1
                
                # Track in running tasks
                self.running_tasks[best_task["task_id"]] = worker_id
                
                # Update in database if available
                if self.db_manager:
                    self.db_manager.update_task_status(
                        best_task["task_id"], 
                        TASK_STATUS_ASSIGNED, 
                        worker_id
                    )
                
                logger.info(f"Assigned task {best_task['task_id']} to worker {worker_id} (score: {best_score:.2f})")
                return best_task
            
            return None
    
    def _calculate_worker_task_match(self, worker_id: str, 
                                   worker_capabilities: Dict[str, Any],
                                   task: Dict[str, Any]) -> float:
        """Calculate how well a worker matches a task's requirements.
        
        Args:
            worker_id: ID of the worker
            worker_capabilities: Capabilities of the worker
            task: Task configuration
            
        Returns:
            Match score (0-1, higher is better)
        """
        # Check cache for similar task types to avoid recalculation
        task_type = task["type"]
        cache_key = (worker_id, task_type)
        
        if cache_key in self.worker_suitability_cache:
            base_score = self.worker_suitability_cache[cache_key]
        else:
            base_score = self._calculate_base_suitability(worker_capabilities, task)
            self.worker_suitability_cache[cache_key] = base_score
            
        # If hard requirements not met, return 0
        if base_score == 0:
            return 0
            
        # Additional factors for smart scheduling
        performance_score = self._calculate_performance_score(worker_id, task)
        resource_score = self._calculate_resource_efficiency(worker_id, task)
        
        # Weight the scores
        final_score = (
            base_score * 0.6 + 
            performance_score * self.config["performance_weight"] +
            resource_score * self.config["resource_factor_weight"]
        )
        
        return final_score
    
    def _calculate_base_suitability(self, worker_capabilities: Dict[str, Any],
                                  task: Dict[str, Any]) -> float:
        """Calculate base suitability score based on hardware requirements.
        
        Args:
            worker_capabilities: Capabilities of the worker
            task: Task configuration
            
        Returns:
            Base suitability score (0-1, higher is better)
        """
        requirements = task.get("requirements", {})
        
        # Different matching algorithms
        if self.match_algorithm == MATCH_ALGORITHM_EXACT:
            # Exact matching - all requirements must be met exactly
            return 1.0 if self._worker_meets_requirements(worker_capabilities, requirements) else 0.0
            
        elif self.match_algorithm == MATCH_ALGORITHM_FLEXIBLE:
            # Flexible matching - score based on how many requirements are met
            return self._flexible_match_score(worker_capabilities, requirements)
            
        else:  # Default to weighted matching
            # Weighted matching - different requirements have different weights
            return self._weighted_match_score(worker_capabilities, requirements)
    
    def _worker_meets_requirements(self, worker_capabilities: Dict[str, Any],
                                task_requirements: Dict[str, Any]) -> bool:
        """Check if a worker meets the requirements for a task.
        
        Args:
            worker_capabilities: Worker's hardware capabilities
            task_requirements: Task's hardware requirements
            
        Returns:
            True if worker meets requirements, False otherwise
        """
        # Check hardware requirements
        if "hardware" in task_requirements:
            required_hardware = task_requirements["hardware"]
            if isinstance(required_hardware, list):
                # Check if worker has any of the required hardware
                worker_hardware = worker_capabilities.get("hardware_types", [])
                if not any(hw in worker_hardware for hw in required_hardware):
                    return False
            elif isinstance(required_hardware, str):
                # Check if worker has the required hardware
                worker_hardware = worker_capabilities.get("hardware_types", [])
                if required_hardware not in worker_hardware:
                    return False
        
        # Check minimum memory
        if "min_memory_gb" in task_requirements:
            min_memory = task_requirements["min_memory_gb"]
            worker_memory = worker_capabilities.get("memory_gb", 0)
            if worker_memory < min_memory:
                return False
        
        # Check minimum CUDA compute capability
        if "min_cuda_compute" in task_requirements:
            min_cuda = task_requirements["min_cuda_compute"]
            worker_cuda = worker_capabilities.get("cuda_compute", 0)
            if worker_cuda < min_cuda:
                return False
        
        # Check for specific browser requirements
        if "browser" in task_requirements:
            required_browser = task_requirements["browser"]
            available_browsers = worker_capabilities.get("browsers", [])
            if required_browser not in available_browsers:
                return False
        
        # Check for specific device requirements (mobile, etc.)
        if "device_type" in task_requirements:
            required_device = task_requirements["device_type"]
            worker_device = worker_capabilities.get("device_type")
            if worker_device != required_device:
                return False
                
        # Check for specialized requirements (audio, vision, etc.)
        if "specialization" in task_requirements:
            required_spec = task_requirements["specialization"]
            worker_specs = worker_capabilities.get("specializations", [])
            if required_spec not in worker_specs:
                return False
        
        return True
    
    def _flexible_match_score(self, worker_capabilities: Dict[str, Any],
                            task_requirements: Dict[str, Any]) -> float:
        """Calculate a flexible match score based on how many requirements are met.
        
        Args:
            worker_capabilities: Worker's hardware capabilities
            task_requirements: Task's hardware requirements
            
        Returns:
            Match score (0-1, higher is better)
        """
        # Essential requirements that must be met
        essential_reqs = ["hardware", "min_cuda_compute"]
        for req in essential_reqs:
            if req in task_requirements:
                # Use the exact matching function for essential requirements
                if req == "hardware":
                    required_hardware = task_requirements["hardware"]
                    worker_hardware = worker_capabilities.get("hardware_types", [])
                    
                    if isinstance(required_hardware, list):
                        if not any(hw in worker_hardware for hw in required_hardware):
                            return 0.0
                    elif isinstance(required_hardware, str):
                        if required_hardware not in worker_hardware:
                            return 0.0
                            
                elif req == "min_cuda_compute":
                    min_cuda = task_requirements["min_cuda_compute"]
                    worker_cuda = worker_capabilities.get("cuda_compute", 0)
                    if worker_cuda < min_cuda:
                        return 0.0
        
        # Scoring for flexible requirements
        total_reqs = 0
        met_reqs = 0
        req_weights = {
            "min_memory_gb": 0.5,
            "browser": 0.3,
            "device_type": 0.4,
            "specialization": 0.4
        }
        
        total_weight = 0
        weighted_score = 0
        
        for req, weight in req_weights.items():
            if req in task_requirements:
                total_weight += weight
                total_reqs += 1
                
                if req == "min_memory_gb":
                    min_memory = task_requirements["min_memory_gb"]
                    worker_memory = worker_capabilities.get("memory_gb", 0)
                    
                    if worker_memory >= min_memory:
                        met_reqs += 1
                        weighted_score += weight
                    elif worker_memory >= min_memory * 0.8:
                        # Partial credit for close matches
                        met_reqs += 0.5
                        weighted_score += weight * 0.5
                        
                elif req == "browser":
                    required_browser = task_requirements["browser"]
                    available_browsers = worker_capabilities.get("browsers", [])
                    
                    if required_browser in available_browsers:
                        met_reqs += 1
                        weighted_score += weight
                        
                elif req == "device_type":
                    required_device = task_requirements["device_type"]
                    worker_device = worker_capabilities.get("device_type")
                    
                    if worker_device == required_device:
                        met_reqs += 1
                        weighted_score += weight
                        
                elif req == "specialization":
                    required_spec = task_requirements["specialization"]
                    worker_specs = worker_capabilities.get("specializations", [])
                    
                    if required_spec in worker_specs:
                        met_reqs += 1
                        weighted_score += weight
        
        # If no flexible requirements, return 1.0 since essential are met
        if total_reqs == 0:
            return 1.0
            
        # If no weights, use simple average
        if total_weight == 0:
            return met_reqs / total_reqs
            
        # Return weighted score
        return weighted_score / total_weight
    
    def _weighted_match_score(self, worker_capabilities: Dict[str, Any],
                            task_requirements: Dict[str, Any]) -> float:
        """Calculate a weighted match score based on requirement importance.
        
        Args:
            worker_capabilities: Worker's hardware capabilities
            task_requirements: Task's hardware requirements
            
        Returns:
            Match score (0-1, higher is better)
        """
        # Define weights for different requirement types
        weights = {
            "hardware": 1.0,  # Most important
            "min_memory_gb": 0.7,
            "min_cuda_compute": 0.8,
            "browser": 0.5,
            "device_type": 0.6,
            "specialization": 0.7
        }
        
        total_weight = 0
        weighted_score = 0
        
        # Check each requirement type
        for req_type, weight in weights.items():
            if req_type in task_requirements:
                total_weight += weight
                
                if req_type == "hardware":
                    required_hardware = task_requirements["hardware"]
                    worker_hardware = worker_capabilities.get("hardware_types", [])
                    
                    if isinstance(required_hardware, list):
                        # Calculate how many of the required hardware types are available
                        matches = sum(1 for hw in required_hardware if hw in worker_hardware)
                        if matches > 0:
                            req_score = matches / len(required_hardware)
                            weighted_score += weight * req_score
                            
                    elif isinstance(required_hardware, str):
                        if required_hardware in worker_hardware:
                            weighted_score += weight
                            
                elif req_type == "min_memory_gb":
                    min_memory = task_requirements["min_memory_gb"]
                    worker_memory = worker_capabilities.get("memory_gb", 0)
                    
                    if worker_memory >= min_memory:
                        weighted_score += weight
                    elif worker_memory >= min_memory * 0.8:
                        # Partial score for close match
                        weighted_score += weight * 0.7
                        
                elif req_type == "min_cuda_compute":
                    min_cuda = task_requirements["min_cuda_compute"]
                    worker_cuda = worker_capabilities.get("cuda_compute", 0)
                    
                    if worker_cuda >= min_cuda:
                        weighted_score += weight
                    elif worker_cuda >= min_cuda - 0.5:
                        # Partial score for close match
                        weighted_score += weight * 0.8
                        
                elif req_type == "browser":
                    required_browser = task_requirements["browser"]
                    available_browsers = worker_capabilities.get("browsers", [])
                    
                    if required_browser in available_browsers:
                        weighted_score += weight
                        
                elif req_type == "device_type":
                    required_device = task_requirements["device_type"]
                    worker_device = worker_capabilities.get("device_type")
                    
                    if worker_device == required_device:
                        weighted_score += weight
                        
                elif req_type == "specialization":
                    required_spec = task_requirements["specialization"]
                    worker_specs = worker_capabilities.get("specializations", [])
                    
                    if required_spec in worker_specs:
                        weighted_score += weight
        
        # If no requirements, return 1.0
        if total_weight == 0:
            return 1.0
            
        # Return normalized score
        return weighted_score / total_weight
    
    def _calculate_performance_score(self, worker_id: str, task: Dict[str, Any]) -> float:
        """Calculate a performance score based on historical performance.
        
        Args:
            worker_id: ID of the worker
            task: Task configuration
            
        Returns:
            Performance score (0-1, higher is better)
        """
        # If no historical data, return neutral score
        if worker_id not in self.worker_performance:
            return 0.5
            
        task_type = task["type"]
        worker_perf = self.worker_performance[worker_id]
        
        # Check if we have type-specific performance data
        if task_type in worker_perf.get("task_types", {}):
            type_perf = worker_perf["task_types"][task_type]
            
            # Check success rate
            success_rate = type_perf.get("success_rate", 0.5)
            
            # Check relative speed (compared to average for this task type)
            if task_type in self.task_stats:
                avg_time = self.task_stats[task_type].get("avg_execution_time", 0)
                if avg_time > 0:
                    worker_avg_time = type_perf.get("avg_execution_time", avg_time)
                    speed_factor = avg_time / worker_avg_time if worker_avg_time > 0 else 1.0
                    speed_score = min(speed_factor, 2.0) / 2.0  # Normalize to 0-1
                else:
                    speed_score = 0.5
            else:
                speed_score = 0.5
                
            # Combine success rate and speed
            return success_rate * 0.7 + speed_score * 0.3
            
        # Default to neutral score if no type-specific data
        return 0.5
    
    def _calculate_resource_efficiency(self, worker_id: str, task: Dict[str, Any]) -> float:
        """Calculate resource efficiency score for worker-task pair.
        
        Args:
            worker_id: ID of the worker
            task: Task configuration
            
        Returns:
            Resource efficiency score (0-1, higher is better)
        """
        # If no worker perf data, return neutral score
        if worker_id not in self.worker_performance:
            return 0.5
            
        worker_perf = self.worker_performance[worker_id]
        
        # Check available memory vs required
        min_memory = task.get("requirements", {}).get("min_memory_gb", 0)
        avail_memory = worker_perf.get("available_memory_gb", 0)
        
        # If no memory data available, return neutral score
        if avail_memory == 0:
            return 0.5
            
        # If worker doesn't have enough memory, return low score
        if avail_memory < min_memory:
            return 0.1
            
        # Calculate how efficiently this task uses available memory
        # Prefer workers where the task uses most of available memory
        # without being too close to the limit
        usage_ratio = min_memory / avail_memory if avail_memory > 0 else 0
        
        # Ideal ratio is around 0.7-0.8 (using most memory but not too close to limit)
        if usage_ratio <= 0.8:
            # Scale up to 1.0 as we approach ideal
            efficiency = min(usage_ratio / 0.8, 1.0)
        else:
            # Scale down as we get too close to limit
            efficiency = max(1.0 - (usage_ratio - 0.8) * 5, 0.0)
            
        return efficiency
    
    def complete_task(self, task_id: str, worker_id: str, 
                     results: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Mark a task as completed and update performance statistics.
        
        Args:
            task_id: ID of the task
            worker_id: ID of the worker that executed the task
            results: Results of the task
            metadata: Metadata about the task execution
            
        Returns:
            True if successful, False otherwise
        """
        # Verify this task is assigned to this worker
        with self.task_lock:
            if task_id not in self.running_tasks:
                logger.warning(f"Task {task_id} not found in running tasks")
                return False
                
            if self.running_tasks[task_id] != worker_id:
                logger.warning(
                    f"Task {task_id} is assigned to {self.running_tasks[task_id]}, "
                    f"not {worker_id}"
                )
                return False
                
            # Remove from running tasks
            del self.running_tasks[task_id]
        
        # Update task status in database
        if self.db_manager:
            self.db_manager.update_task_status(task_id, TASK_STATUS_COMPLETED)
            
            # Store results
            self.db_manager.add_task_result(task_id, worker_id, results, metadata)
            
            # Add execution history
            start_time = metadata.get("start_time")
            end_time = metadata.get("end_time")
            
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                
            if not start_time:
                start_time = datetime.now() - timedelta(seconds=metadata.get("execution_time", 0))
                
            if not end_time:
                end_time = datetime.now()
                
            execution_time = metadata.get("execution_time", 0)
            hardware_metrics = metadata.get("hardware_metrics", {})
            
            self.db_manager.add_execution_history(
                task_id, worker_id, metadata.get("attempt", 1),
                TASK_STATUS_COMPLETED, start_time, end_time,
                execution_time, "", hardware_metrics
            )
        
        # Update performance statistics
        self._update_performance_stats(worker_id, task_id, True, metadata)
        
        # Process task dependencies
        self._process_task_dependencies(task_id)
        
        logger.info(f"Task {task_id} completed by worker {worker_id}")
        return True
    
    def _process_task_dependencies(self, completed_task_id: str):
        """Process dependencies for a completed task.
        
        Args:
            completed_task_id: ID of the completed task
        """
        # Check if any tasks depend on this one
        if completed_task_id in self.reverse_dependencies:
            dependent_tasks = self.reverse_dependencies[completed_task_id].copy()
            
            for dep_task_id in dependent_tasks:
                # Remove completed task from dependencies
                if dep_task_id in self.task_dependencies:
                    self.task_dependencies[dep_task_id].remove(completed_task_id)
                    
                    # If all dependencies satisfied, task can be queued
                    if len(self.task_dependencies[dep_task_id]) == 0:
                        logger.info(f"All dependencies satisfied for task {dep_task_id}")
                        
                        # Clean up dependency tracking
                        del self.task_dependencies[dep_task_id]
                        
                        # Get task from database and add to queue
                        if self.db_manager:
                            task = self.db_manager.get_task(dep_task_id)
                            if task:
                                with self.task_lock:
                                    priority = task["priority"]
                                    create_time = task.get("create_time", datetime.now())
                                    self.task_queue.append((priority, create_time, dep_task_id, task))
                                    self.task_queue.sort()
                                    
                                logger.info(f"Task {dep_task_id} added to queue after dependencies satisfied")
            
            # Remove completed task from reverse dependencies
            del self.reverse_dependencies[completed_task_id]
    
    def fail_task(self, task_id: str, worker_id: str, 
                 error: str, metadata: Dict[str, Any]) -> bool:
        """Mark a task as failed and handle retry logic.
        
        Args:
            task_id: ID of the task
            worker_id: ID of the worker that executed the task
            error: Error message
            metadata: Metadata about the task execution
            
        Returns:
            True if successful, False otherwise
        """
        # Verify this task is assigned to this worker
        with self.task_lock:
            if task_id not in self.running_tasks:
                logger.warning(f"Task {task_id} not found in running tasks")
                return False
                
            if self.running_tasks[task_id] != worker_id:
                logger.warning(
                    f"Task {task_id} is assigned to {self.running_tasks[task_id]}, "
                    f"not {worker_id}"
                )
                return False
                
            # Remove from running tasks
            del self.running_tasks[task_id]
            
            # Check if we should retry
            task = self.db_manager.get_task(task_id) if self.db_manager else None
            max_retries = metadata.get("max_retries", self.config["max_retries"])
            
            if task and task["attempts"] < max_retries:
                # Requeue task with delay and increased priority to ensure it gets picked up
                current_priority = task["priority"]
                # Slightly increase priority (lower value) for retry
                retry_priority = max(1, current_priority - 1) 
                
                # Create delayed task for retry
                # Current time + delay to ensure it sorts after other tasks with same priority
                retry_time = datetime.now() + timedelta(
                    seconds=metadata.get("retry_delay", self.config["retry_delay_seconds"])
                )
                
                # Update attempts count
                task["attempts"] += 1
                
                # Add back to queue with retry delay factored into the time
                self.task_queue.append((retry_priority, retry_time, task_id, task))
                self.task_queue.sort()
                
                logger.info(f"Requeued task {task_id} after failure (attempt {task['attempts']})")
                
                # Update status in database
                if self.db_manager:
                    self.db_manager.update_task_status(task_id, TASK_STATUS_QUEUED)
            else:
                # Mark as failed
                if self.db_manager:
                    self.db_manager.update_task_status(task_id, TASK_STATUS_FAILED)
                
                logger.info(f"Task {task_id} failed by worker {worker_id}: {error}")
                
                # Process dependencies to clean up dependent tasks
                self._handle_failed_dependencies(task_id)
        
        # Add execution history
        if self.db_manager:
            start_time = metadata.get("start_time")
            end_time = metadata.get("end_time")
            
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                
            if not start_time:
                start_time = datetime.now() - timedelta(seconds=metadata.get("execution_time", 0))
                
            if not end_time:
                end_time = datetime.now()
                
            execution_time = metadata.get("execution_time", 0)
            hardware_metrics = metadata.get("hardware_metrics", {})
            
            self.db_manager.add_execution_history(
                task_id, worker_id, metadata.get("attempt", 1),
                TASK_STATUS_FAILED, start_time, end_time,
                execution_time, error, hardware_metrics
            )
        
        # Update performance statistics
        self._update_performance_stats(worker_id, task_id, False, metadata)
        
        return True
    
    def _handle_failed_dependencies(self, failed_task_id: str):
        """Handle dependent tasks when a dependency fails.
        
        Args:
            failed_task_id: ID of the failed task
        """
        # Check if any tasks depend on this one
        if failed_task_id in self.reverse_dependencies:
            dependent_tasks = self.reverse_dependencies[failed_task_id].copy()
            
            for dep_task_id in dependent_tasks:
                # Mark dependent task as failed due to dependency failure
                logger.warning(f"Task {dep_task_id} will be canceled due to dependency failure")
                
                # Update database status if available
                if self.db_manager:
                    self.db_manager.update_task_status(dep_task_id, TASK_STATUS_CANCELED)
                    
                    # Add execution history entry
                    self.db_manager.add_execution_history(
                        dep_task_id, "none", 0, TASK_STATUS_CANCELED,
                        datetime.now(), datetime.now(), 0,
                        f"Canceled due to dependency failure: {failed_task_id}", {}
                    )
                
                # Remove from queue if present
                with self.task_lock:
                    for i, (_, _, queue_task_id, _) in enumerate(self.task_queue):
                        if queue_task_id == dep_task_id:
                            self.task_queue.pop(i)
                            break
                
                # Recursively handle tasks that depend on this one
                self._handle_failed_dependencies(dep_task_id)
            
            # Remove failed task from reverse dependencies
            del self.reverse_dependencies[failed_task_id]
    
    def _update_performance_stats(self, worker_id: str, task_id: str, 
                                success: bool, metadata: Dict[str, Any]):
        """Update performance statistics for a worker based on task results.
        
        Args:
            worker_id: ID of the worker
            task_id: ID of the task
            success: Whether the task was successful
            metadata: Metadata about the task execution
        """
        # Get task details
        task_type = None
        if self.db_manager:
            task = self.db_manager.get_task(task_id)
            if task:
                task_type = task.get("type")
                
        # If no task type available, can't update stats
        if not task_type:
            return
            
        # Initialize worker performance if not exists
        if worker_id not in self.worker_performance:
            self.worker_performance[worker_id] = {
                "success_count": 0,
                "failure_count": 0,
                "total_execution_time": 0,
                "task_count": 0,
                "task_types": {},
                "last_update": datetime.now()
            }
            
        worker_perf = self.worker_performance[worker_id]
        
        # Initialize task type if not exists
        if task_type not in worker_perf["task_types"]:
            worker_perf["task_types"][task_type] = {
                "success_count": 0,
                "failure_count": 0,
                "total_execution_time": 0,
                "task_count": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0
            }
            
        type_perf = worker_perf["task_types"][task_type]
        
        # Update overall stats
        worker_perf["task_count"] += 1
        if success:
            worker_perf["success_count"] += 1
        else:
            worker_perf["failure_count"] += 1
            
        # Update type-specific stats
        type_perf["task_count"] += 1
        if success:
            type_perf["success_count"] += 1
        else:
            type_perf["failure_count"] += 1
            
        # Update execution time
        execution_time = metadata.get("execution_time", 0)
        worker_perf["total_execution_time"] += execution_time
        type_perf["total_execution_time"] += execution_time
        
        # Calculate success rate and average execution time
        type_perf["success_rate"] = type_perf["success_count"] / type_perf["task_count"]
        if type_perf["task_count"] > 0:
            type_perf["avg_execution_time"] = type_perf["total_execution_time"] / type_perf["task_count"]
            
        # Update hardware metrics if available
        if "hardware_metrics" in metadata:
            hw_metrics = metadata["hardware_metrics"]
            
            # Extract available memory
            if "memory_available_gb" in hw_metrics:
                worker_perf["available_memory_gb"] = hw_metrics["memory_available_gb"]
                
            # Extract CPU usage
            if "cpu_percent" in hw_metrics:
                worker_perf["cpu_percent"] = hw_metrics["cpu_percent"]
                
            # Extract GPU memory if available
            if "gpu_metrics" in hw_metrics and hw_metrics["gpu_metrics"]:
                gpu_metrics = hw_metrics["gpu_metrics"][0]  # Use first GPU for simplicity
                if "memory_used_mb" in gpu_metrics and "memory_total_mb" in gpu_metrics:
                    used_mb = gpu_metrics["memory_used_mb"]
                    total_mb = gpu_metrics["memory_total_mb"]
                    worker_perf["gpu_memory_available_mb"] = total_mb - used_mb
        
        # Update last update timestamp
        worker_perf["last_update"] = datetime.now()
        
        # Update global task stats
        if task_type not in self.task_stats:
            self.task_stats[task_type] = {
                "success_count": 0,
                "failure_count": 0,
                "total_execution_time": 0,
                "task_count": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0
            }
            
        global_stats = self.task_stats[task_type]
        
        # Update global stats
        global_stats["task_count"] += 1
        if success:
            global_stats["success_count"] += 1
        else:
            global_stats["failure_count"] += 1
            
        global_stats["total_execution_time"] += execution_time
        
        # Calculate global success rate and average execution time
        if global_stats["task_count"] > 0:
            global_stats["success_rate"] = global_stats["success_count"] / global_stats["task_count"]
            global_stats["avg_execution_time"] = global_stats["total_execution_time"] / global_stats["task_count"]
            
        # Invalidate suitability cache for this worker and task type
        cache_key = (worker_id, task_type)
        if cache_key in self.worker_suitability_cache:
            del self.worker_suitability_cache[cache_key]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if successful, False otherwise
        """
        with self.task_lock:
            # Check if task is queued
            for i, (_, _, tid, _) in enumerate(self.task_queue):
                if tid == task_id:
                    self.task_queue.pop(i)
                    logger.info(f"Canceled queued task {task_id}")
                    
                    # Update status in database
                    if self.db_manager:
                        self.db_manager.update_task_status(task_id, TASK_STATUS_CANCELED)
                        
                    # Handle dependent tasks
                    self._handle_failed_dependencies(task_id)
                        
                    return True
            
            # Check if task is running
            if task_id in self.running_tasks:
                # Can't actually stop a running task, just mark it as canceled
                # The worker will continue to execute it
                worker_id = self.running_tasks[task_id]
                logger.info(f"Marked running task {task_id} as canceled (worker: {worker_id})")
                
                # Update status in database
                if self.db_manager:
                    self.db_manager.update_task_status(task_id, TASK_STATUS_CANCELED)
                    
                # Handle dependent tasks
                self._handle_failed_dependencies(task_id)
                    
                return True
        
        # Task not found
        logger.warning(f"Task {task_id} not found")
        return False
    
    def get_worker_performance(self, worker_id: str = None) -> Dict[str, Any]:
        """Get worker performance statistics.
        
        Args:
            worker_id: Optional worker ID to get stats for (all workers if None)
            
        Returns:
            Dict containing performance statistics
        """
        if worker_id:
            return self.worker_performance.get(worker_id, {})
        else:
            return self.worker_performance
    
    def get_task_stats(self, task_type: str = None) -> Dict[str, Any]:
        """Get task type statistics.
        
        Args:
            task_type: Optional task type to get stats for (all types if None)
            
        Returns:
            Dict containing task statistics
        """
        if task_type:
            return self.task_stats.get(task_type, {})
        else:
            return self.task_stats
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the task queue.
        
        Returns:
            Dict with queue statistics
        """
        with self.task_lock:
            task_types = {}
            priorities = {}
            
            # Count tasks by type and priority
            for _, _, _, task in self.task_queue:
                task_type = task.get("type", "unknown")
                priority = task.get("priority", 5)
                
                if task_type not in task_types:
                    task_types[task_type] = 0
                task_types[task_type] += 1
                
                if priority not in priorities:
                    priorities[priority] = 0
                priorities[priority] += 1
            
            return {
                "queued_tasks": len(self.task_queue),
                "running_tasks": len(self.running_tasks),
                "total_tasks": len(self.task_queue) + len(self.running_tasks),
                "task_types": task_types,
                "priorities": priorities
            }
    
    def check_timeouts(self) -> List[str]:
        """Check for and handle timed out tasks.
        
        Returns:
            List of timed out task IDs
        """
        timeout_tasks = []
        current_time = datetime.now()
        
        with self.task_lock:
            # Check running tasks for timeout
            for task_id, worker_id in list(self.running_tasks.items()):
                # Get task details
                task = None
                if self.db_manager:
                    task = self.db_manager.get_task(task_id)
                
                if task and "start_time" in task:
                    start_time = task["start_time"]
                    timeout_seconds = task.get("timeout_seconds", self.config["timeout_seconds"])
                    
                    # Check if task has timed out
                    if (current_time - start_time).total_seconds() > timeout_seconds:
                        logger.warning(f"Task {task_id} timed out after {timeout_seconds}s")
                        
                        # Remove from running tasks
                        del self.running_tasks[task_id]
                        
                        # Update status in database
                        if self.db_manager:
                            self.db_manager.update_task_status(task_id, TASK_STATUS_TIMED_OUT)
                            
                            # Add execution history for timeout
                            self.db_manager.add_execution_history(
                                task_id, worker_id, task.get("attempts", 1),
                                TASK_STATUS_TIMED_OUT, start_time, current_time,
                                (current_time - start_time).total_seconds(),
                                "Task timed out", {}
                            )
                        
                        # Check if we should retry
                        max_retries = task.get("max_retries", self.config["max_retries"])
                        
                        if task["attempts"] < max_retries:
                            # Requeue task with higher priority
                            current_priority = task["priority"]
                            retry_priority = max(1, current_priority - 1)  # Increase priority
                            retry_time = current_time + timedelta(seconds=60)  # 1 minute delay
                            
                            # Update attempts count
                            task["attempts"] += 1
                            
                            # Add back to queue
                            self.task_queue.append((retry_priority, retry_time, task_id, task))
                            self.task_queue.sort()
                            
                            logger.info(f"Requeued timed out task {task_id} (attempt {task['attempts']})")
                            
                            # Update status in database
                            if self.db_manager:
                                self.db_manager.update_task_status(task_id, TASK_STATUS_QUEUED)
                        else:
                            # Mark as failed (used all retry attempts)
                            logger.warning(f"Task {task_id} failed after timing out {max_retries} times")
                            if self.db_manager:
                                self.db_manager.update_task_status(task_id, TASK_STATUS_FAILED)
                                
                            # Handle dependencies
                            self._handle_failed_dependencies(task_id)
                        
                        # Add to list of timed out tasks
                        timeout_tasks.append(task_id)
                        
                        # Update worker performance stats for timeout
                        self._update_performance_stats(
                            worker_id, task_id, False, 
                            {
                                "execution_time": (current_time - start_time).total_seconds(),
                                "error": "Task timed out"
                            }
                        )
        
        return timeout_tasks
    
    def reset_cache(self):
        """Reset the worker suitability cache."""
        self.worker_suitability_cache = {}
        logger.info("Worker suitability cache reset")