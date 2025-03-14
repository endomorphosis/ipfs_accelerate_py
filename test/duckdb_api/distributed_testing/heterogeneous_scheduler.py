"""
Heterogeneous Hardware Scheduler for Distributed Testing Framework

This module provides specialized scheduling algorithms optimized for
heterogeneous hardware environments. It leverages the hardware taxonomy
and enhanced hardware detection to make intelligent scheduling decisions
based on workload characteristics and hardware capabilities.
"""

import logging
import time
import heapq
import threading
import itertools
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import copy
import json
import random

from .hardware_taxonomy import (
    HardwareClass,
    HardwareArchitecture,
    HardwareVendor,
    SoftwareBackend,
    PrecisionType,
    AcceleratorFeature,
    HardwareCapabilityProfile,
    HardwareTaxonomy
)

from .enhanced_hardware_detector import EnhancedHardwareDetector

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class WorkloadProfile:
    """
    Profile for a specific workload type, describing its hardware requirements
    and execution characteristics.
    """
    workload_type: str  # e.g., "nlp", "vision", "audio"
    operation_types: List[str]  # e.g., ["matmul", "conv", "softmax"]
    precision_types: List[str]  # e.g., ["fp32", "fp16", "int8"]
    
    # Resource requirements
    min_memory_gb: float = 1.0
    preferred_memory_gb: float = 4.0
    min_compute_units: int = 1
    
    # Required hardware features
    required_features: List[str] = field(default_factory=list)  # e.g., ["tensor_cores", "avx2"]
    
    # Required backends
    required_backends: List[str] = field(default_factory=list)  # e.g., ["cuda", "webgpu"]
    
    # Enhanced capability requirements
    required_capabilities: Set[str] = field(default_factory=set)  # Required capabilities from Enhanced Hardware Taxonomy
    preferred_capabilities: Set[str] = field(default_factory=set)  # Preferred capabilities from Enhanced Hardware Taxonomy
    
    # Performance characteristics
    batch_size_options: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    optimal_batch_size: Optional[int] = None
    
    # Workload priority (higher is more important)
    priority: int = 1
    
    # Execution constraints
    max_execution_time_ms: Optional[int] = None
    is_latency_sensitive: bool = False
    is_throughput_sensitive: bool = False
    is_power_sensitive: bool = False
    
    # Compatibility with hardware classes (0.0 to 1.0, higher is better)
    hardware_class_affinity: Dict[str, float] = field(default_factory=dict)
    
    # History of execution performance by hardware class
    performance_history: Dict[str, List[float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default affinities if not provided."""
        if not self.hardware_class_affinity:
            # Set default affinities based on workload type
            if self.workload_type == "nlp":
                self.hardware_class_affinity = {
                    "gpu": 0.9,
                    "cpu": 0.6,
                    "tpu": 0.8,
                    "npu": 0.7,
                    "hybrid": 0.5
                }
            elif self.workload_type == "vision":
                self.hardware_class_affinity = {
                    "gpu": 0.9,
                    "cpu": 0.4,
                    "tpu": 0.8,
                    "npu": 0.8,
                    "hybrid": 0.6
                }
            elif self.workload_type == "audio":
                self.hardware_class_affinity = {
                    "gpu": 0.7,
                    "cpu": 0.8,
                    "tpu": 0.6,
                    "npu": 0.7,
                    "hybrid": 0.6
                }
            else:
                # Default balanced affinity
                self.hardware_class_affinity = {
                    "gpu": 0.7,
                    "cpu": 0.7,
                    "tpu": 0.7,
                    "npu": 0.7,
                    "hybrid": 0.5
                }
    
    def add_required_capability(self, capability_id: str) -> None:
        """
        Add a required capability for this workload.
        
        Args:
            capability_id: ID of the capability to add
        """
        self.required_capabilities.add(capability_id)
    
    def add_preferred_capability(self, capability_id: str) -> None:
        """
        Add a preferred capability for this workload.
        
        Args:
            capability_id: ID of the capability to add
        """
        self.preferred_capabilities.add(capability_id)
    
    def remove_capability(self, capability_id: str) -> bool:
        """
        Remove a capability from both required and preferred sets.
        
        Args:
            capability_id: ID of the capability to remove
            
        Returns:
            True if the capability was removed, False if it wasn't found
        """
        removed = False
        if capability_id in self.required_capabilities:
            self.required_capabilities.remove(capability_id)
            removed = True
        if capability_id in self.preferred_capabilities:
            self.preferred_capabilities.remove(capability_id)
            removed = True
        return removed
    
    def update_performance(self, hardware_class: str, execution_time_ms: float):
        """
        Update performance history for a hardware class.
        
        Args:
            hardware_class: The hardware class that executed the workload
            execution_time_ms: The execution time in milliseconds
        """
        if hardware_class not in self.performance_history:
            self.performance_history[hardware_class] = []
        
        # Keep history bounded to recent executions (last 100)
        history = self.performance_history[hardware_class]
        history.append(execution_time_ms)
        if len(history) > 100:
            history.pop(0)
    
    def get_average_performance(self, hardware_class: str) -> Optional[float]:
        """
        Get average execution time for a hardware class.
        
        Args:
            hardware_class: The hardware class to get average performance for
            
        Returns:
            Average execution time in milliseconds, or None if no history
        """
        if hardware_class not in self.performance_history:
            return None
        
        history = self.performance_history[hardware_class]
        if not history:
            return None
        
        return sum(history) / len(history)


@dataclass
class TestTask:
    """
    Represents a test task to be scheduled on a worker.
    """
    task_id: str
    workload_profile: WorkloadProfile
    
    # Inputs for the task (can be serialized for transport)
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # Execution configuration
    batch_size: int = 1
    timeout_ms: Optional[int] = None
    
    # Priority and ordering
    priority: int = 1
    submission_time: float = field(default_factory=time.time)
    
    # Scheduling state
    assigned_worker_id: Optional[str] = None
    scheduled_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Result and status
    status: str = "pending"  # pending, scheduled, running, completed, failed
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    
    # Hardware execution details
    executed_on_hardware_class: Optional[str] = None
    executed_on_hardware_model: Optional[str] = None
    
    def mark_scheduled(self, worker_id: str):
        """Mark the task as scheduled on a worker."""
        self.assigned_worker_id = worker_id
        self.scheduled_time = time.time()
        self.status = "scheduled"
    
    def mark_running(self):
        """Mark the task as running."""
        self.start_time = time.time()
        self.status = "running"
    
    def mark_completed(self, result: Any, hardware_class: str, hardware_model: str):
        """Mark the task as completed with a result."""
        self.end_time = time.time()
        self.status = "completed"
        self.result = result
        self.executed_on_hardware_class = hardware_class
        self.executed_on_hardware_model = hardware_model
        
        # Calculate execution time
        if self.start_time:
            self.execution_time_ms = (self.end_time - self.start_time) * 1000
            
            # Update workload profile performance history
            if self.execution_time_ms is not None and hardware_class:
                self.workload_profile.update_performance(hardware_class, self.execution_time_ms)
    
    def mark_failed(self, error: str):
        """Mark the task as failed with an error message."""
        self.end_time = time.time()
        self.status = "failed"
        self.error = error
        
        # Calculate execution time even for failures
        if self.start_time:
            self.execution_time_ms = (self.end_time - self.start_time) * 1000
    
    def get_waiting_time(self) -> float:
        """Get time spent waiting before scheduling."""
        if self.scheduled_time is None:
            return time.time() - self.submission_time
        return self.scheduled_time - self.submission_time
    
    def get_queue_time(self) -> float:
        """Get total time spent in queue before execution."""
        if self.start_time is None:
            if self.scheduled_time is None:
                return time.time() - self.submission_time
            return time.time() - self.submission_time
        return self.start_time - self.submission_time
    
    def is_timeout(self) -> bool:
        """Check if the task has exceeded its timeout."""
        if self.timeout_ms is None:
            return False
        
        if self.start_time is None:
            return False
        
        elapsed_ms = (time.time() - self.start_time) * 1000
        return elapsed_ms > self.timeout_ms


@dataclass
class WorkerState:
    """
    Represents the current state of a worker node.
    """
    worker_id: str
    capabilities: Dict[str, Any]  # From EnhancedHardwareDetector
    hardware_profiles: List[Dict[str, Any]]  # Serialized profiles
    
    # Current resource utilization
    current_load: Dict[str, float] = field(default_factory=dict)
    available_memory_gb: float = 0.0
    
    # Workload execution performance metrics
    performance_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Task tracking
    active_tasks: List[TestTask] = field(default_factory=list)
    completed_tasks: List[TestTask] = field(default_factory=list)
    failed_tasks: List[TestTask] = field(default_factory=list)
    
    # Worker status
    status: str = "online"  # online, busy, offline, warming, cooling
    last_heartbeat: float = field(default_factory=time.time)
    
    # Thermal management
    thermal_state: Dict[str, Any] = field(default_factory=dict)
    
    # Hardware capability summaries
    hardware_classes: Set[str] = field(default_factory=set)
    hardware_vendors: Set[str] = field(default_factory=set)
    hardware_architectures: Set[str] = field(default_factory=set)
    supported_backends: Set[str] = field(default_factory=set)
    supported_precisions: Set[str] = field(default_factory=set)
    hardware_features: Set[str] = field(default_factory=set)
    
    # Workload specializations based on hardware taxonomy
    workload_specializations: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived state from capabilities and hardware profiles."""
        # Extract hardware classes
        for profile in self.hardware_profiles:
            if "hardware_class" in profile:
                self.hardware_classes.add(profile["hardware_class"])
            
            if "vendor" in profile:
                self.hardware_vendors.add(profile["vendor"])
            
            if "architecture" in profile:
                self.hardware_architectures.add(profile["architecture"])
            
            if "supported_backends" in profile:
                self.supported_backends.update(profile["supported_backends"])
            
            if "supported_precisions" in profile:
                self.supported_precisions.update(profile["supported_precisions"])
            
            if "features" in profile:
                self.hardware_features.update(profile["features"])
        
        # Extract workload specializations if available
        if "optimal_hardware" in self.capabilities:
            for workload_type, hardware in self.capabilities["optimal_hardware"].items():
                if hardware and "effectiveness_score" in hardware:
                    self.workload_specializations[workload_type] = hardware["effectiveness_score"]
        
        # Initialize current load for each hardware class
        for hardware_class in self.hardware_classes:
            self.current_load[hardware_class] = 0.0
        
        # Initialize available memory
        for profile in self.hardware_profiles:
            # Take the largest memory amount as a simple heuristic
            if "memory_available_gb" in profile:
                self.available_memory_gb = max(self.available_memory_gb, profile["memory_available_gb"])
        
        # Initialize thermal state
        self.thermal_state = {
            "temperature": 50.0,  # Default temp in Celsius
            "warming_rate": 0.1,  # Degrees per active task
            "cooling_rate": 0.2,  # Degrees per second when idle
            "throttle_threshold": 80.0,  # Temperature at which to start throttling
            "critical_threshold": 90.0,  # Temperature at which to stop assigning tasks
            "last_update_time": time.time()
        }
    
    def update_load(self, task: TestTask = None):
        """
        Update load metrics based on active tasks.
        
        Args:
            task: Optional new task to consider in the load
        """
        # Reset load counters
        for hardware_class in self.hardware_classes:
            self.current_load[hardware_class] = 0.0
        
        # Count active tasks per hardware class
        for active_task in self.active_tasks:
            hardware_class = active_task.executed_on_hardware_class
            if hardware_class and hardware_class in self.current_load:
                self.current_load[hardware_class] += 1.0
        
        # Add the new task if provided
        if task and task.workload_profile:
            # Find the most likely hardware class for this workload
            best_hardware_class = None
            best_affinity = -1.0
            
            for hardware_class in self.hardware_classes:
                affinity = task.workload_profile.hardware_class_affinity.get(hardware_class, 0.0)
                if affinity > best_affinity:
                    best_affinity = affinity
                    best_hardware_class = hardware_class
            
            if best_hardware_class and best_hardware_class in self.current_load:
                self.current_load[best_hardware_class] += 1.0
    
    def update_thermal_state(self):
        """Update thermal state based on workload and time elapsed."""
        current_time = time.time()
        elapsed_seconds = current_time - self.thermal_state["last_update_time"]
        
        # Calculate warming from active tasks
        warming = len(self.active_tasks) * self.thermal_state["warming_rate"]
        
        # Calculate cooling when idle
        cooling = self.thermal_state["cooling_rate"] * elapsed_seconds if not self.active_tasks else 0.0
        
        # Update temperature
        self.thermal_state["temperature"] += warming - cooling
        
        # Clamp temperature to reasonable bounds
        self.thermal_state["temperature"] = max(30.0, min(self.thermal_state["temperature"], 100.0))
        
        # Update status based on temperature
        if self.thermal_state["temperature"] >= self.thermal_state["critical_threshold"]:
            self.status = "cooling"
        elif self.thermal_state["temperature"] >= self.thermal_state["throttle_threshold"]:
            # Still accept tasks but with lower priority
            self.status = "warming"
        else:
            # Normal operation
            self.status = "online" if len(self.active_tasks) < 10 else "busy"
        
        # Update last update time
        self.thermal_state["last_update_time"] = current_time
    
    def has_capacity_for(self, task: TestTask) -> bool:
        """
        Check if this worker has capacity to execute a task.
        
        Args:
            task: The task to check capacity for
            
        Returns:
            bool: True if the worker has capacity, False otherwise
        """
        # Check if worker is offline
        if self.status == "offline":
            return False
        
        # Check if worker is in cooling state
        if self.status == "cooling":
            return False
        
        # Check if worker has memory capacity
        if task.workload_profile.min_memory_gb > self.available_memory_gb:
            return False
        
        # Check if worker has required backends
        required_backends = set(task.workload_profile.required_backends)
        if required_backends and not required_backends.issubset(self.supported_backends):
            return False
        
        # Check if worker has required features
        required_features = set(task.workload_profile.required_features)
        if required_features and not required_features.issubset(self.hardware_features):
            return False
        
        # Check load threshold - this is a simple heuristic and could be more sophisticated
        total_load = sum(self.current_load.values())
        total_capacity = 10  # Default arbitrary capacity
        
        # Estimate capacity based on compute units across all hardware
        for profile in self.hardware_profiles:
            if "compute_units" in profile:
                total_capacity += profile["compute_units"] // 2  # Conservative estimate
        
        # Check if adding this task would exceed capacity
        return total_load < total_capacity
    
    def calculate_affinity_score(self, task: TestTask) -> float:
        """
        Calculate an affinity score for a task based on hardware compatibility
        and specialization.
        
        Args:
            task: The task to calculate affinity for
            
        Returns:
            float: Affinity score (0.0 to 1.0, higher is better)
        """
        workload_type = task.workload_profile.workload_type
        
        # Start with base score from workload specialization
        base_score = self.workload_specializations.get(workload_type, 0.5)
        
        # Adjust based on hardware class affinities
        hardware_affinity = 0.0
        for hardware_class in self.hardware_classes:
            class_affinity = task.workload_profile.hardware_class_affinity.get(hardware_class, 0.0)
            hardware_affinity = max(hardware_affinity, class_affinity)
        
        # Adjust based on historical performance
        performance_factor = 1.0
        for hardware_class in self.hardware_classes:
            avg_performance = task.workload_profile.get_average_performance(hardware_class)
            if avg_performance is not None:
                # Normalize performance to favor faster execution
                # This assumes lower execution times are better
                normalized_perf = 1.0 / (1.0 + avg_performance / 1000.0)
                performance_factor = max(performance_factor, normalized_perf)
        
        # Adjust based on thermal state
        thermal_factor = 1.0
        if self.status == "warming":
            thermal_factor = 0.7
        
        # Combine factors
        return base_score * hardware_affinity * performance_factor * thermal_factor
    
    def add_task(self, task: TestTask):
        """
        Add a task to this worker's active tasks.
        
        Args:
            task: The task to add
        """
        self.active_tasks.append(task)
        task.mark_scheduled(self.worker_id)
        self.update_load(task)
    
    def complete_task(self, task_id: str, result: Any, hardware_class: str, hardware_model: str) -> Optional[TestTask]:
        """
        Mark a task as completed and move it to completed tasks.
        
        Args:
            task_id: ID of the task to complete
            result: Result of the task execution
            hardware_class: Hardware class that executed the task
            hardware_model: Hardware model that executed the task
            
        Returns:
            The completed task, or None if not found
        """
        for i, task in enumerate(self.active_tasks):
            if task.task_id == task_id:
                task.mark_completed(result, hardware_class, hardware_model)
                self.completed_tasks.append(task)
                self.active_tasks.pop(i)
                self.update_load()
                return task
        return None
    
    def fail_task(self, task_id: str, error: str) -> Optional[TestTask]:
        """
        Mark a task as failed and move it to failed tasks.
        
        Args:
            task_id: ID of the task to fail
            error: Error message
            
        Returns:
            The failed task, or None if not found
        """
        for i, task in enumerate(self.active_tasks):
            if task.task_id == task_id:
                task.mark_failed(error)
                self.failed_tasks.append(task)
                self.active_tasks.pop(i)
                self.update_load()
                return task
        return None


class HeterogeneousScheduler:
    """
    Scheduler for heterogeneous hardware environments that allocates tasks
    to worker nodes based on hardware capabilities, workload requirements,
    and performance history.
    """
    
    def __init__(self, 
                strategy: str = "adaptive",
                thermal_management: bool = True,
                enable_workload_learning: bool = True,
                use_enhanced_taxonomy: bool = False):
        """
        Initialize the heterogeneous scheduler.
        
        Args:
            strategy: Scheduling strategy (adaptive, resource_aware, performance_aware, round_robin)
            thermal_management: Enable thermal management
            enable_workload_learning: Enable learning from past workload executions
            use_enhanced_taxonomy: Enable integration with enhanced hardware taxonomy
        """
        self._lock = threading.Lock()
        self.strategy = strategy
        self.thermal_management = thermal_management
        self.enable_workload_learning = enable_workload_learning
        self.use_enhanced_taxonomy = use_enhanced_taxonomy
        
        # Worker management
        self.workers: Dict[str, WorkerState] = {}
        
        # Task queues
        self.pending_tasks: List[TestTask] = []
        self.scheduled_tasks: Dict[str, TestTask] = {}  # By task_id
        self.completed_tasks: List[TestTask] = []
        self.failed_tasks: List[TestTask] = []
        
        # Workload profiles
        self.workload_profiles: Dict[str, WorkloadProfile] = {}
        
        # Performance history
        self.hardware_performance: Dict[str, Dict[str, List[float]]] = {}
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_scheduled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_queue_time_ms": 0.0,
            "avg_execution_time_ms": 0.0,
            "worker_utilization": {}
        }
        
        # Enhanced taxonomy integration
        if use_enhanced_taxonomy:
            try:
                from .hardware_taxonomy_integrator import HardwareTaxonomyIntegrator
                self.taxonomy_integrator = HardwareTaxonomyIntegrator()
                logger.info("Enhanced hardware taxonomy integration enabled")
            except ImportError as e:
                logger.warning(f"Failed to import HardwareTaxonomyIntegrator: {e}")
                logger.warning("Enhanced hardware taxonomy integration disabled")
                self.use_enhanced_taxonomy = False
    
    def register_worker(self, worker_id: str, capabilities: Dict[str, Any]) -> WorkerState:
        """
        Register a worker with the scheduler.
        
        Args:
            worker_id: Unique ID for the worker
            capabilities: Hardware and software capabilities of the worker
            
        Returns:
            WorkerState: The registered worker state
        """
        with self._lock:
            # Create worker state from capabilities
            worker = WorkerState(
                worker_id=worker_id,
                capabilities=capabilities,
                hardware_profiles=capabilities.get("hardware_profiles", [])
            )
            
            # Apply enhanced taxonomy if enabled
            if self.use_enhanced_taxonomy:
                try:
                    worker = self.taxonomy_integrator.enhance_worker_state(worker)
                    logger.info(f"Enhanced worker {worker_id} with taxonomy-based capabilities")
                except Exception as e:
                    logger.warning(f"Failed to enhance worker with taxonomy: {e}")
            
            # Store worker
            self.workers[worker_id] = worker
            
            # Initialize worker utilization stats
            self.stats["worker_utilization"][worker_id] = 0.0
            
            logger.info(f"Registered worker {worker_id} with {len(worker.hardware_profiles)} hardware profiles")
            return worker
    
    def unregister_worker(self, worker_id: str):
        """
        Unregister a worker from the scheduler.
        
        Args:
            worker_id: ID of the worker to unregister
        """
        with self._lock:
            if worker_id in self.workers:
                # Mark worker as offline
                self.workers[worker_id].status = "offline"
                
                # Reschedule any active tasks
                for task in self.workers[worker_id].active_tasks:
                    task.status = "pending"
                    task.assigned_worker_id = None
                    task.scheduled_time = None
                    task.start_time = None
                    self.pending_tasks.append(task)
                
                # Remove worker
                del self.workers[worker_id]
                
                logger.info(f"Unregistered worker {worker_id}")
    
    def update_worker_state(self, worker_id: str, state_update: Dict[str, Any]):
        """
        Update the state of a worker.
        
        Args:
            worker_id: ID of the worker to update
            state_update: Dictionary with state updates
        """
        with self._lock:
            if worker_id not in self.workers:
                logger.warning(f"Tried to update unknown worker {worker_id}")
                return
            
            worker = self.workers[worker_id]
            
            # Update load
            if "current_load" in state_update:
                worker.current_load.update(state_update["current_load"])
            
            # Update available memory
            if "available_memory_gb" in state_update:
                worker.available_memory_gb = state_update["available_memory_gb"]
            
            # Update status
            if "status" in state_update:
                worker.status = state_update["status"]
            
            # Update heartbeat
            worker.last_heartbeat = time.time()
            
            # Update thermal state if provided
            if "thermal_state" in state_update:
                worker.thermal_state.update(state_update["thermal_state"])
            elif self.thermal_management:
                # Otherwise update thermal state based on time and workload
                worker.update_thermal_state()
    
    def submit_task(self, task: TestTask) -> str:
        """
        Submit a task to be scheduled.
        
        Args:
            task: The task to submit
            
        Returns:
            str: The task ID
        """
        with self._lock:
            # Enhance workload profile with taxonomy-based capabilities if enabled
            if self.use_enhanced_taxonomy:
                try:
                    task.workload_profile = self.taxonomy_integrator.enhance_workload_profile(
                        task.workload_profile
                    )
                    logger.debug(
                        f"Enhanced workload profile for task {task.task_id} with "
                        f"{len(task.workload_profile.required_capabilities)} required capabilities and "
                        f"{len(task.workload_profile.preferred_capabilities)} preferred capabilities"
                    )
                except Exception as e:
                    logger.warning(f"Failed to enhance workload profile with taxonomy: {e}")
            
            # Register workload profile if needed
            if task.workload_profile.workload_type not in self.workload_profiles:
                self.workload_profiles[task.workload_profile.workload_type] = task.workload_profile
            
            # Add task to pending queue
            self.pending_tasks.append(task)
            
            # Update statistics
            self.stats["tasks_submitted"] += 1
            
            logger.debug(f"Submitted task {task.task_id} of type {task.workload_profile.workload_type}")
            return task.task_id
    
    def schedule_tasks(self):
        """
        Schedule pending tasks to available workers.
        """
        with self._lock:
            # Skip if no pending tasks or no workers
            if not self.pending_tasks or not self.workers:
                return
            
            # Update thermal state for all workers
            if self.thermal_management:
                for worker in self.workers.values():
                    worker.update_thermal_state()
            
            # Sort pending tasks by priority (higher first) and then submission time
            self.pending_tasks.sort(key=lambda task: (-task.priority, task.submission_time))
            
            # Make a copy of the list since we'll be modifying it
            tasks_to_schedule = self.pending_tasks.copy()
            
            # Strategy dispatch
            if self.strategy == "adaptive":
                scheduled_tasks = self._schedule_adaptive(tasks_to_schedule)
            elif self.strategy == "resource_aware":
                scheduled_tasks = self._schedule_resource_aware(tasks_to_schedule)
            elif self.strategy == "performance_aware":
                scheduled_tasks = self._schedule_performance_aware(tasks_to_schedule)
            elif self.strategy == "round_robin":
                scheduled_tasks = self._schedule_round_robin(tasks_to_schedule)
            else:
                # Default to adaptive
                scheduled_tasks = self._schedule_adaptive(tasks_to_schedule)
            
            # Remove scheduled tasks from pending queue
            for task in scheduled_tasks:
                if task in self.pending_tasks:
                    self.pending_tasks.remove(task)
                
                # Add to scheduled tasks
                self.scheduled_tasks[task.task_id] = task
                
                # Update statistics
                self.stats["tasks_scheduled"] += 1
            
            logger.debug(f"Scheduled {len(scheduled_tasks)} tasks, {len(self.pending_tasks)} pending")
    
    def _calculate_standard_affinity(self, worker: WorkerState, task: TestTask) -> float:
        """
        Calculate the standard affinity score for a worker and task.
        
        Args:
            worker: The worker to calculate affinity for
            task: The task to calculate affinity for
            
        Returns:
            float: Affinity score (0.0 to 1.0, higher is better)
        """
        workload_type = task.workload_profile.workload_type
        
        # Calculate baseline score from specialization
        base_score = worker.workload_specializations.get(workload_type, 0.5)
        
        # Adjust for current load
        load_factor = 1.0
        for hardware_class, load in worker.current_load.items():
            affinity = task.workload_profile.hardware_class_affinity.get(hardware_class, 0.0)
            if affinity > 0.0:
                # Higher affinity hardware types are more impacted by load
                load_impact = load * affinity
                load_factor = min(load_factor, 1.0 / (1.0 + load_impact / 5.0))
        
        # Adjust for thermal state
        thermal_factor = 1.0
        if worker.status == "warming":
            thermal_factor = 0.7
        
        # Combine factors
        final_score = base_score * load_factor * thermal_factor
        return final_score
    
    def _schedule_adaptive(self, tasks: List[TestTask]) -> List[TestTask]:
        """
        Adaptive scheduling that combines multiple strategies.
        
        Args:
            tasks: List of tasks to schedule
            
        Returns:
            List of scheduled tasks
        """
        scheduled_tasks = []
        available_workers = [w for w in self.workers.values() if w.status != "offline" and w.status != "cooling"]
        
        if not available_workers:
            return scheduled_tasks
        
        # Group tasks by workload type
        tasks_by_workload = {}
        for task in tasks:
            workload_type = task.workload_profile.workload_type
            if workload_type not in tasks_by_workload:
                tasks_by_workload[workload_type] = []
            tasks_by_workload[workload_type].append(task)
        
        # For each workload type, find the best workers
        for workload_type, workload_tasks in tasks_by_workload.items():
            # Sort workers by affinity for this workload type
            workers_with_scores = []
            
            for worker in available_workers:
                # Use enhanced affinity calculation if enabled
                if self.use_enhanced_taxonomy:
                    try:
                        # Use taxonomy-based affinity calculation
                        final_score = self.taxonomy_integrator.calculate_enhanced_affinity(
                            worker, workload_tasks[0]
                        )
                        
                        workers_with_scores.append((worker, final_score))
                        
                        logger.debug(
                            f"Enhanced affinity for worker {worker.worker_id} and task type "
                            f"{workload_type}: {final_score:.2f}"
                        )
                    except Exception as e:
                        # Fall back to standard affinity calculation
                        logger.warning(
                            f"Error calculating enhanced affinity for worker {worker.worker_id}: {e}. "
                            f"Falling back to standard method."
                        )
                        final_score = self._calculate_standard_affinity(worker, workload_tasks[0])
                        workers_with_scores.append((worker, final_score))
                else:
                    # Use standard affinity calculation
                    final_score = self._calculate_standard_affinity(worker, workload_tasks[0])
                    workers_with_scores.append((worker, final_score))
            
            # Sort workers by score (descending)
            workers_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Assign tasks to workers
            for task in workload_tasks:
                assigned = False
                
                # Try workers in order of score
                for worker, _ in workers_with_scores:
                    if worker.has_capacity_for(task):
                        worker.add_task(task)
                        scheduled_tasks.append(task)
                        assigned = True
                        
                        # Update worker utilization stats
                        self.stats["worker_utilization"][worker.worker_id] = len(worker.active_tasks)
                        
                        break
                
                if not assigned:
                    logger.debug(f"Could not find suitable worker for task {task.task_id}")
        
        return scheduled_tasks
    
    def _schedule_resource_aware(self, tasks: List[TestTask]) -> List[TestTask]:
        """
        Resource-aware scheduling that prioritizes even resource distribution.
        
        Args:
            tasks: List of tasks to schedule
            
        Returns:
            List of scheduled tasks
        """
        scheduled_tasks = []
        available_workers = [w for w in self.workers.values() if w.status != "offline" and w.status != "cooling"]
        
        if not available_workers:
            return scheduled_tasks
        
        # Sort workers by load (ascending)
        for task in tasks:
            # Sort workers by current total load
            sorted_workers = sorted(available_workers, key=lambda w: sum(w.current_load.values()))
            
            assigned = False
            for worker in sorted_workers:
                if worker.has_capacity_for(task):
                    worker.add_task(task)
                    scheduled_tasks.append(task)
                    assigned = True
                    
                    # Update worker utilization stats
                    self.stats["worker_utilization"][worker.worker_id] = len(worker.active_tasks)
                    
                    break
            
            if not assigned:
                logger.debug(f"Could not find suitable worker for task {task.task_id}")
        
        return scheduled_tasks
    
    def _schedule_performance_aware(self, tasks: List[TestTask]) -> List[TestTask]:
        """
        Performance-aware scheduling that prioritizes workers with best historical performance.
        
        Args:
            tasks: List of tasks to schedule
            
        Returns:
            List of scheduled tasks
        """
        scheduled_tasks = []
        available_workers = [w for w in self.workers.values() if w.status != "offline" and w.status != "cooling"]
        
        if not available_workers:
            return scheduled_tasks
        
        for task in tasks:
            # Get workload type
            workload_type = task.workload_profile.workload_type
            
            # Calculate worker scores based on historical performance
            workers_with_scores = []
            for worker in available_workers:
                if not worker.has_capacity_for(task):
                    continue
                
                # Calculate score based on affinity and historical performance
                score = worker.calculate_affinity_score(task)
                workers_with_scores.append((worker, score))
            
            # Sort workers by score (descending)
            workers_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Assign task to best worker
            if workers_with_scores:
                best_worker, _ = workers_with_scores[0]
                best_worker.add_task(task)
                scheduled_tasks.append(task)
                
                # Update worker utilization stats
                self.stats["worker_utilization"][best_worker.worker_id] = len(best_worker.active_tasks)
            else:
                logger.debug(f"No suitable worker for task {task.task_id}")
        
        return scheduled_tasks
    
    def _schedule_round_robin(self, tasks: List[TestTask]) -> List[TestTask]:
        """
        Simple round-robin scheduling.
        
        Args:
            tasks: List of tasks to schedule
            
        Returns:
            List of scheduled tasks
        """
        scheduled_tasks = []
        available_workers = [w for w in self.workers.values() if w.status != "offline" and w.status != "cooling"]
        
        if not available_workers:
            return scheduled_tasks
        
        # Circular assignment of tasks to workers
        worker_cycle = itertools.cycle(available_workers)
        
        for task in tasks:
            assigned = False
            
            # Try up to len(available_workers) workers
            for _ in range(len(available_workers)):
                worker = next(worker_cycle)
                if worker.has_capacity_for(task):
                    worker.add_task(task)
                    scheduled_tasks.append(task)
                    assigned = True
                    
                    # Update worker utilization stats
                    self.stats["worker_utilization"][worker.worker_id] = len(worker.active_tasks)
                    
                    break
            
            if not assigned:
                logger.debug(f"Could not find suitable worker for task {task.task_id}")
        
        return scheduled_tasks
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task to get status for
            
        Returns:
            Dict with task status, or None if task not found
        """
        with self._lock:
            # Check scheduled tasks
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                return {
                    "task_id": task.task_id,
                    "status": task.status,
                    "worker_id": task.assigned_worker_id,
                    "queue_time": task.get_queue_time(),
                    "execution_time": task.execution_time_ms
                }
            
            # Check pending tasks
            for task in self.pending_tasks:
                if task.task_id == task_id:
                    return {
                        "task_id": task.task_id,
                        "status": task.status,
                        "worker_id": None,
                        "queue_time": task.get_queue_time(),
                        "execution_time": None
                    }
            
            # Check completed tasks
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    return {
                        "task_id": task.task_id,
                        "status": task.status,
                        "worker_id": task.assigned_worker_id,
                        "queue_time": task.get_queue_time(),
                        "execution_time": task.execution_time_ms,
                        "hardware_class": task.executed_on_hardware_class,
                        "hardware_model": task.executed_on_hardware_model
                    }
            
            # Check failed tasks
            for task in self.failed_tasks:
                if task.task_id == task_id:
                    return {
                        "task_id": task.task_id,
                        "status": task.status,
                        "worker_id": task.assigned_worker_id,
                        "queue_time": task.get_queue_time(),
                        "execution_time": task.execution_time_ms,
                        "error": task.error
                    }
            
            return None
    
    def report_task_completion(self, worker_id: str, task_id: str, result: Any, hardware_info: Dict[str, str]):
        """
        Report completion of a task.
        
        Args:
            worker_id: ID of the worker that completed the task
            task_id: ID of the completed task
            result: Result of the task execution
            hardware_info: Information about the hardware that executed the task
        """
        with self._lock:
            if worker_id not in self.workers:
                logger.warning(f"Task completion reported by unknown worker {worker_id}")
                return
            
            # Get hardware class and model
            hardware_class = hardware_info.get("hardware_class", "unknown")
            hardware_model = hardware_info.get("hardware_model", "unknown")
            
            # Mark task as completed in worker
            task = self.workers[worker_id].complete_task(task_id, result, hardware_class, hardware_model)
            
            if task:
                # Remove from scheduled tasks
                if task_id in self.scheduled_tasks:
                    del self.scheduled_tasks[task_id]
                
                # Add to completed tasks
                self.completed_tasks.append(task)
                
                # Update statistics
                self.stats["tasks_completed"] += 1
                
                if task.execution_time_ms:
                    # Running average of execution time
                    old_avg = self.stats["avg_execution_time_ms"]
                    self.stats["avg_execution_time_ms"] = (old_avg * (self.stats["tasks_completed"] - 1) + task.execution_time_ms) / self.stats["tasks_completed"]
                
                queue_time_ms = task.get_queue_time() * 1000
                old_avg = self.stats["avg_queue_time_ms"]
                self.stats["avg_queue_time_ms"] = (old_avg * (self.stats["tasks_completed"] - 1) + queue_time_ms) / self.stats["tasks_completed"]
                
                logger.debug(f"Task {task_id} completed on {worker_id} ({hardware_class}/{hardware_model}) in {task.execution_time_ms:.2f}ms")
            else:
                logger.warning(f"Completion reported for unknown task {task_id} on worker {worker_id}")
    
    def report_task_failure(self, worker_id: str, task_id: str, error: str):
        """
        Report failure of a task.
        
        Args:
            worker_id: ID of the worker where the task failed
            task_id: ID of the failed task
            error: Error message
        """
        with self._lock:
            if worker_id not in self.workers:
                logger.warning(f"Task failure reported by unknown worker {worker_id}")
                return
            
            # Mark task as failed in worker
            task = self.workers[worker_id].fail_task(task_id, error)
            
            if task:
                # Remove from scheduled tasks
                if task_id in self.scheduled_tasks:
                    del self.scheduled_tasks[task_id]
                
                # Add to failed tasks
                self.failed_tasks.append(task)
                
                # Update statistics
                self.stats["tasks_failed"] += 1
                
                logger.warning(f"Task {task_id} failed on {worker_id}: {error}")
            else:
                logger.warning(f"Failure reported for unknown task {task_id} on worker {worker_id}")
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.
        
        Returns:
            Dict with scheduler statistics
        """
        with self._lock:
            # Copy stats
            stats = copy.deepcopy(self.stats)
            
            # Add current counts
            stats["pending_tasks"] = len(self.pending_tasks)
            stats["scheduled_tasks"] = len(self.scheduled_tasks)
            stats["completed_tasks"] = len(self.completed_tasks)
            stats["failed_tasks"] = len(self.failed_tasks)
            stats["active_workers"] = len([w for w in self.workers.values() if w.status != "offline"])
            
            # Calculate current worker utilization
            worker_utilization = {}
            for worker_id, worker in self.workers.items():
                if worker.status != "offline":
                    worker_utilization[worker_id] = {
                        "active_tasks": len(worker.active_tasks),
                        "load": worker.current_load,
                        "status": worker.status
                    }
            
            stats["current_worker_utilization"] = worker_utilization
            
            return stats
    
    def get_worker_stats(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific worker.
        
        Args:
            worker_id: ID of the worker to get statistics for
            
        Returns:
            Dict with worker statistics, or None if worker not found
        """
        with self._lock:
            if worker_id not in self.workers:
                return None
            
            worker = self.workers[worker_id]
            
            # Basic stats
            stats = {
                "worker_id": worker_id,
                "status": worker.status,
                "active_tasks": len(worker.active_tasks),
                "completed_tasks": len(worker.completed_tasks),
                "failed_tasks": len(worker.failed_tasks),
                "current_load": worker.current_load,
                "available_memory_gb": worker.available_memory_gb,
                "thermal_state": worker.thermal_state,
                "hardware_classes": list(worker.hardware_classes),
                "workload_specializations": worker.workload_specializations,
                "last_heartbeat": worker.last_heartbeat
            }
            
            # Active task details
            active_task_details = []
            for task in worker.active_tasks:
                active_task_details.append({
                    "task_id": task.task_id,
                    "workload_type": task.workload_profile.workload_type,
                    "priority": task.priority,
                    "queue_time_ms": task.get_queue_time() * 1000,
                    "running_time_ms": (time.time() - task.start_time) * 1000 if task.start_time else None
                })
            
            stats["active_task_details"] = active_task_details
            
            return stats
    
    def get_workload_stats(self, workload_type: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific workload type.
        
        Args:
            workload_type: Type of workload to get statistics for
            
        Returns:
            Dict with workload statistics, or None if workload type not found
        """
        with self._lock:
            if workload_type not in self.workload_profiles:
                return None
            
            # Tasks of this workload type
            pending = [t for t in self.pending_tasks if t.workload_profile.workload_type == workload_type]
            scheduled = [t for t in self.scheduled_tasks.values() if t.workload_profile.workload_type == workload_type]
            completed = [t for t in self.completed_tasks if t.workload_profile.workload_type == workload_type]
            failed = [t for t in self.failed_tasks if t.workload_profile.workload_type == workload_type]
            
            # Performance statistics
            execution_times = [t.execution_time_ms for t in completed if t.execution_time_ms is not None]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else None
            
            # Performance by hardware class
            performance_by_hardware = {}
            for task in completed:
                if task.executed_on_hardware_class and task.execution_time_ms:
                    hardware_class = task.executed_on_hardware_class
                    if hardware_class not in performance_by_hardware:
                        performance_by_hardware[hardware_class] = []
                    performance_by_hardware[hardware_class].append(task.execution_time_ms)
            
            # Average performance by hardware class
            avg_performance_by_hardware = {}
            for hardware_class, times in performance_by_hardware.items():
                avg_performance_by_hardware[hardware_class] = sum(times) / len(times)
            
            # Prepare stats
            stats = {
                "workload_type": workload_type,
                "pending_count": len(pending),
                "scheduled_count": len(scheduled),
                "completed_count": len(completed),
                "failed_count": len(failed),
                "avg_execution_time_ms": avg_execution_time,
                "performance_by_hardware": avg_performance_by_hardware,
                "profile": {
                    "operation_types": self.workload_profiles[workload_type].operation_types,
                    "precision_types": self.workload_profiles[workload_type].precision_types,
                    "min_memory_gb": self.workload_profiles[workload_type].min_memory_gb,
                    "required_features": self.workload_profiles[workload_type].required_features,
                    "required_backends": self.workload_profiles[workload_type].required_backends,
                    "hardware_class_affinity": self.workload_profiles[workload_type].hardware_class_affinity
                }
            }
            
            return stats
    
    def remove_completed_tasks(self, age_seconds: float = 3600.0):
        """
        Remove completed tasks older than a specified age.
        
        Args:
            age_seconds: Age in seconds beyond which to remove tasks
        """
        with self._lock:
            now = time.time()
            
            # Filter completed tasks
            self.completed_tasks = [t for t in self.completed_tasks 
                                   if t.end_time is None or (now - t.end_time) < age_seconds]
            
            # Filter failed tasks
            self.failed_tasks = [t for t in self.failed_tasks 
                                if t.end_time is None or (now - t.end_time) < age_seconds]
    
    def check_worker_heartbeats(self, timeout_seconds: float = 300.0):
        """
        Check worker heartbeats and mark workers as offline if they haven't
        reported in too long.
        
        Args:
            timeout_seconds: Time in seconds after which a worker is considered offline
        """
        with self._lock:
            now = time.time()
            
            for worker_id, worker in list(self.workers.items()):
                if now - worker.last_heartbeat > timeout_seconds and worker.status != "offline":
                    logger.warning(f"Worker {worker_id} hasn't reported in {timeout_seconds} seconds, marking as offline")
                    worker.status = "offline"
                    
                    # Reschedule active tasks
                    for task in worker.active_tasks:
                        task.status = "pending"
                        task.assigned_worker_id = None
                        task.scheduled_time = None
                        task.start_time = None
                        self.pending_tasks.append(task)
                    
                    worker.active_tasks = []
    
    def get_optimal_worker_for_workload(self, workload_type: str) -> Optional[str]:
        """
        Find the optimal worker for a specific workload type based on
        specialization and current load.
        
        Args:
            workload_type: Type of workload
            
        Returns:
            ID of the optimal worker, or None if no suitable worker found
        """
        with self._lock:
            best_worker_id = None
            best_score = -1.0
            
            for worker_id, worker in self.workers.items():
                if worker.status == "offline" or worker.status == "cooling":
                    continue
                
                # Calculate basic score from specialization
                base_score = worker.workload_specializations.get(workload_type, 0.5)
                
                # Adjust for current load - simple inverse scaling
                total_load = sum(worker.current_load.values())
                load_factor = 1.0 / (1.0 + total_load / 5.0)  # 5 is arbitrary scaling factor
                
                # Combine scores
                score = base_score * load_factor
                
                if score > best_score:
                    best_score = score
                    best_worker_id = worker_id
            
            return best_worker_id
    
    def perform_load_balancing(self):
        """
        Perform load balancing by moving tasks between workers.
        """
        with self._lock:
            # Identify overloaded and underloaded workers
            worker_loads = []
            for worker_id, worker in self.workers.items():
                if worker.status == "offline" or worker.status == "cooling":
                    continue
                
                total_load = sum(worker.current_load.values())
                worker_loads.append((worker_id, total_load, worker))
            
            if not worker_loads:
                return
            
            # Sort by load (descending)
            worker_loads.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate average load
            avg_load = sum(load for _, load, _ in worker_loads) / len(worker_loads)
            
            # Identify workers more than 50% above average load
            overloaded = []
            for worker_id, load, worker in worker_loads:
                if load > avg_load * 1.5 and load > 1:  # At least 50% above average and more than 1 task
                    overloaded.append((worker_id, load, worker))
            
            # Identify workers more than 50% below average load
            underloaded = []
            for worker_id, load, worker in worker_loads:
                if load < avg_load * 0.5:  # At least 50% below average
                    underloaded.append((worker_id, load, worker))
            
            # Balance load by moving tasks from overloaded to underloaded workers
            tasks_moved = 0
            for over_id, over_load, over_worker in overloaded:
                if not over_worker.active_tasks:
                    continue
                
                # Sort tasks by recent scheduling time (move newer tasks first)
                tasks_to_move = sorted(over_worker.active_tasks, key=lambda t: t.scheduled_time or 0, reverse=True)
                
                for task in tasks_to_move:
                    # Skip tasks that have already started execution
                    if task.start_time is not None:
                        continue
                    
                    # Try to find a suitable underloaded worker
                    for under_id, under_load, under_worker in underloaded:
                        if under_worker.has_capacity_for(task):
                            # Move task to underloaded worker
                            logger.info(f"Moving task {task.task_id} from {over_id} to {under_id} for load balancing")
                            
                            # Remove from overloaded worker
                            over_worker.active_tasks.remove(task)
                            
                            # Add to underloaded worker
                            task.assigned_worker_id = under_id
                            under_worker.add_task(task)
                            
                            # Update loads
                            tasks_moved += 1
                            break
                    
                    # Only move a limited number of tasks per balancing operation
                    if tasks_moved >= 5:
                        break
                
                # Only balance a limited number of overloaded workers per operation
                if tasks_moved >= 5:
                    break
            
            if tasks_moved > 0:
                logger.info(f"Load balancing moved {tasks_moved} tasks")
    
    def export_scheduler_state(self, file_path: str):
        """
        Export the current scheduler state to a file for analysis.
        
        Args:
            file_path: Path to export the state to
        """
        with self._lock:
            # Prepare state for serialization
            state = {
                "timestamp": time.time(),
                "stats": self.stats,
                "workers": {},
                "pending_tasks_count": len(self.pending_tasks),
                "scheduled_tasks_count": len(self.scheduled_tasks),
                "completed_tasks_count": len(self.completed_tasks),
                "failed_tasks_count": len(self.failed_tasks),
                "workload_profiles": {}
            }
            
            # Export worker state
            for worker_id, worker in self.workers.items():
                worker_state = {
                    "status": worker.status,
                    "hardware_classes": list(worker.hardware_classes),
                    "supported_backends": list(worker.supported_backends),
                    "current_load": worker.current_load,
                    "available_memory_gb": worker.available_memory_gb,
                    "workload_specializations": worker.workload_specializations,
                    "active_tasks_count": len(worker.active_tasks),
                    "completed_tasks_count": len(worker.completed_tasks),
                    "failed_tasks_count": len(worker.failed_tasks)
                }
                state["workers"][worker_id] = worker_state
            
            # Export workload profiles
            for workload_type, profile in self.workload_profiles.items():
                workload_state = {
                    "operation_types": profile.operation_types,
                    "precision_types": profile.precision_types,
                    "min_memory_gb": profile.min_memory_gb,
                    "required_features": profile.required_features,
                    "required_backends": profile.required_backends,
                    "hardware_class_affinity": profile.hardware_class_affinity
                }
                state["workload_profiles"][workload_type] = workload_state
            
            # Write to file
            with open(file_path, "w") as f:
                json.dump(state, f, indent=2)
    
    def import_scheduler_state(self, file_path: str):
        """
        Import scheduler state from a file.
        
        Args:
            file_path: Path to import the state from
        """
        with self._lock:
            try:
                with open(file_path, "r") as f:
                    state = json.load(f)
                
                # Import workload profiles
                for workload_type, profile_data in state.get("workload_profiles", {}).items():
                    # Create workload profile
                    profile = WorkloadProfile(
                        workload_type=workload_type,
                        operation_types=profile_data.get("operation_types", []),
                        precision_types=profile_data.get("precision_types", []),
                        min_memory_gb=profile_data.get("min_memory_gb", 1.0),
                        required_features=profile_data.get("required_features", []),
                        required_backends=profile_data.get("required_backends", [])
                    )
                    
                    # Set hardware class affinity
                    profile.hardware_class_affinity = profile_data.get("hardware_class_affinity", {})
                    
                    # Store profile
                    self.workload_profiles[workload_type] = profile
                
                logger.info(f"Imported {len(state.get('workload_profiles', {}))} workload profiles from {file_path}")
                return True
            except Exception as e:
                logger.error(f"Error importing scheduler state: {e}")
                return False