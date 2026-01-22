#!/usr/bin/env python3
"""
Distributed Testing Framework - Dynamic Resource Manager

This module implements the Dynamic Resource Manager for the distributed testing framework,
which optimizes resource allocation based on workload patterns.

Core responsibilities:
- Resource tracking for worker nodes
- Adaptive scaling based on workload
- Worker reassessment for capability updates
- Cloud integration for ephemeral workers
- Reservation tracking for resource allocation

Usage:
    # Import and initialize
    from duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager
    
    # Create resource manager
    resource_mgr = DynamicResourceManager(
        target_utilization=0.7,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3
    )
    
    # Register a worker with its resources
    resource_mgr.register_worker(
        worker_id="worker-1",
        resources={
            "cpu": {"cores": 8, "available_cores": 8},
            "memory": {"total_mb": 16384, "available_mb": 16384},
            "gpu": {"devices": 1, "memory_mb": 8192, "available_memory_mb": 8192}
        }
    )
    
    # Reserve resources for a task
    reservation_id = resource_mgr.reserve_resources(
        task_id="task-1",
        resource_requirements={
            "cpu_cores": 2,
            "memory_mb": 4096,
            "gpu_memory_mb": 2048
        }
    )
    
    # Release resources after task completion
    resource_mgr.release_resources(reservation_id)
    
    # Evaluate if scaling is needed
    scaling_decision = resource_mgr.evaluate_scaling()
    
    # Get worker statistics
    worker_stats = resource_mgr.get_worker_statistics()
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, NamedTuple
from pathlib import Path
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("dynamic_resource_manager")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import optional dependencies
try:
    from resource_performance_predictor import ResourcePerformancePredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    logger.warning("ResourcePerformancePredictor not available. Prediction features disabled.")
    PREDICTOR_AVAILABLE = False

try:
    from cloud_provider_manager import CloudProviderManager
    CLOUD_INTEGRATION_AVAILABLE = True
except ImportError:
    try:
        import cloud_provider_integration
        CLOUD_INTEGRATION_AVAILABLE = True
    except ImportError:
        logger.warning("Cloud provider integration not available. Cloud scaling features disabled.")
        CLOUD_INTEGRATION_AVAILABLE = False

# Resource management constants
DEFAULT_TARGET_UTILIZATION = 0.7  # 70% target utilization
DEFAULT_SCALE_UP_THRESHOLD = 0.8  # Scale up at 80% utilization
DEFAULT_SCALE_DOWN_THRESHOLD = 0.3  # Scale down at 30% utilization
DEFAULT_EVALUATION_WINDOW = 300  # 5 minutes (in seconds)
DEFAULT_SCALE_UP_COOLDOWN = 300  # 5 minutes (in seconds)
DEFAULT_SCALE_DOWN_COOLDOWN = 600  # 10 minutes (in seconds)
DEFAULT_WORKER_REASSESSMENT_INTERVAL = 3600  # 1 hour (in seconds)
DEFAULT_HISTORY_RETENTION = 86400  # 24 hours (in seconds)


@dataclass
class ScalingDecision:
    """Class representing a scaling decision."""
    action: str  # "none", "scale_up", "scale_down", "maintain"
    reason: str  # Human-readable reason for the decision
    count: int = 0  # Number of workers to scale up/down
    worker_ids: List[str] = None  # Specific worker IDs to scale down
    utilization: float = 0.0  # Current overall utilization
    worker_type: str = "default"  # Type of worker to create
    resource_requirements: Dict[str, Any] = None  # Resource requirements for new workers
    provider: str = None  # Preferred cloud provider
    timestamp: float = None  # When the decision was made
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.worker_ids is None:
            self.worker_ids = []
        if self.resource_requirements is None:
            self.resource_requirements = {}
        if self.timestamp is None:
            self.timestamp = time.time()


class DynamicResourceManager:
    """
    Dynamic Resource Manager for the distributed testing framework.
    
    Handles resource tracking, allocation, and scaling decisions based on workload patterns.
    """
    
    def __init__(self, 
                 target_utilization: float = DEFAULT_TARGET_UTILIZATION,
                 scale_up_threshold: float = DEFAULT_SCALE_UP_THRESHOLD,
                 scale_down_threshold: float = DEFAULT_SCALE_DOWN_THRESHOLD,
                 evaluation_window: int = DEFAULT_EVALUATION_WINDOW,
                 scale_up_cooldown: int = DEFAULT_SCALE_UP_COOLDOWN,
                 scale_down_cooldown: int = DEFAULT_SCALE_DOWN_COOLDOWN,
                 worker_reassessment_interval: int = DEFAULT_WORKER_REASSESSMENT_INTERVAL,
                 history_retention: int = DEFAULT_HISTORY_RETENTION):
        """
        Initialize the Dynamic Resource Manager.
        
        Args:
            target_utilization: Target resource utilization (0.0-1.0)
            scale_up_threshold: Threshold to trigger scale up (0.0-1.0)
            scale_down_threshold: Threshold to trigger scale down (0.0-1.0)
            evaluation_window: Window for utilization evaluation (seconds)
            scale_up_cooldown: Cooldown period after scaling up (seconds)
            scale_down_cooldown: Cooldown period after scaling down (seconds)
            worker_reassessment_interval: Interval for worker capability reassessment (seconds)
            history_retention: How long to retain resource history (seconds)
        """
        # Configuration
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.evaluation_window = evaluation_window
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        self.worker_reassessment_interval = worker_reassessment_interval
        self.history_retention = history_retention
        
        # Resource tracking
        self.worker_resources = {}  # worker_id -> resource data
        self.resource_reservations = {}  # reservation_id -> reservation data
        self.worker_tasks = {}  # worker_id -> set of task_ids
        self.task_reservation = {}  # task_id -> reservation_id
        
        # Performance tracking
        self.worker_performance = {}  # worker_id -> performance metrics
        self.resource_history = []  # List of historical resource usage snapshots
        
        # Scaling state
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0
        self.scaling_evaluation_lock = threading.Lock()
        
        # Initialize performance predictor if available
        self.predictor = None
        if PREDICTOR_AVAILABLE:
            self.predictor = ResourcePerformancePredictor()
        
        # Start background threads
        self.running = True
        self.resource_tracker_thread = threading.Thread(target=self._resource_tracker_loop)
        self.resource_tracker_thread.daemon = True
        self.resource_tracker_thread.start()
        
        self.worker_reassessment_thread = threading.Thread(target=self._worker_reassessment_loop)
        self.worker_reassessment_thread.daemon = True
        self.worker_reassessment_thread.start()
        
        logger.info(f"Dynamic Resource Manager initialized with target utilization: {target_utilization:.1%}")
    
    def register_worker(self, worker_id: str, resources: Dict[str, Any]) -> bool:
        """
        Register a worker with its resource capabilities.
        
        Args:
            worker_id: Unique identifier for the worker
            resources: Dictionary of worker resources
                {
                    "cpu": {"cores": int, "available_cores": int},
                    "memory": {"total_mb": int, "available_mb": int},
                    "gpu": {"devices": int, "memory_mb": int, "available_memory_mb": int}
                }
        
        Returns:
            bool: Success status
        """
        try:
            # Validate resource data
            if not self._validate_resource_data(resources):
                logger.error(f"Invalid resource data for worker {worker_id}")
                return False
            
            # Add worker resources
            self.worker_resources[worker_id] = {
                "registration_time": time.time(),
                "last_updated": time.time(),
                "resources": resources,
                "utilization": {
                    "cpu": 0.0,
                    "memory": 0.0,
                    "gpu": 0.0,
                    "overall": 0.0
                }
            }
            
            # Initialize performance metrics
            self.worker_performance[worker_id] = {
                "task_count": 0,
                "success_rate": 1.0,
                "avg_execution_time": 0.0,
                "resource_efficiency": 1.0,
                "last_updated": time.time()
            }
            
            # Initialize task tracking
            self.worker_tasks[worker_id] = set()
            
            logger.info(f"Worker {worker_id} registered with {resources['cpu']['cores']} CPU cores, " 
                       f"{resources['memory']['total_mb']} MB RAM, "
                       f"{resources.get('gpu', {}).get('devices', 0)} GPUs")
            return True
        
        except Exception as e:
            logger.error(f"Error registering worker {worker_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def update_worker_resources(self, worker_id: str, resources: Dict[str, Any]) -> bool:
        """
        Update a worker's resource capabilities.
        
        Args:
            worker_id: Unique identifier for the worker
            resources: Dictionary of worker resources
        
        Returns:
            bool: Success status
        """
        try:
            if worker_id not in self.worker_resources:
                logger.error(f"Worker {worker_id} not registered")
                return False
            
            # Validate resource data
            if not self._validate_resource_data(resources):
                logger.error(f"Invalid resource data for worker {worker_id}")
                return False
            
            # Update worker resources
            self.worker_resources[worker_id]["resources"] = resources
            self.worker_resources[worker_id]["last_updated"] = time.time()
            
            logger.info(f"Worker {worker_id} resources updated")
            return True
        
        except Exception as e:
            logger.error(f"Error updating worker {worker_id} resources: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def deregister_worker(self, worker_id: str) -> bool:
        """
        Deregister a worker from the resource manager.
        
        Args:
            worker_id: Unique identifier for the worker
        
        Returns:
            bool: Success status
        """
        try:
            if worker_id not in self.worker_resources:
                logger.warning(f"Worker {worker_id} not registered")
                return False
            
            # Check if worker has active tasks
            if worker_id in self.worker_tasks and self.worker_tasks[worker_id]:
                active_tasks = len(self.worker_tasks[worker_id])
                logger.warning(f"Worker {worker_id} has {active_tasks} active tasks")
                # Could implement forced task reassignment here
            
            # Remove worker data
            self.worker_resources.pop(worker_id, None)
            self.worker_performance.pop(worker_id, None)
            self.worker_tasks.pop(worker_id, None)
            
            logger.info(f"Worker {worker_id} deregistered")
            return True
        
        except Exception as e:
            logger.error(f"Error deregistering worker {worker_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def reserve_resources(self, task_id: str, resource_requirements: Dict[str, Any]) -> Optional[str]:
        """
        Reserve resources for a task.
        
        Args:
            task_id: Unique identifier for the task
            resource_requirements: Dictionary of resource requirements
                {
                    "cpu_cores": int,
                    "memory_mb": int,
                    "gpu_memory_mb": int,
                    "worker_id": str (optional - specific worker request)
                }
        
        Returns:
            str: Reservation ID on success, None on failure
        """
        try:
            # Validate requirements
            if not self._validate_resource_requirements(resource_requirements):
                logger.error(f"Invalid resource requirements for task {task_id}")
                return None
            
            # Extract requirements
            cpu_cores = resource_requirements.get("cpu_cores", 1)
            memory_mb = resource_requirements.get("memory_mb", 1024)
            gpu_memory_mb = resource_requirements.get("gpu_memory_mb", 0)
            requested_worker_id = resource_requirements.get("worker_id", None)
            
            # Find suitable worker
            worker_id = self._find_suitable_worker(cpu_cores, memory_mb, gpu_memory_mb, requested_worker_id)
            if not worker_id:
                logger.warning(f"No suitable worker found for task {task_id}")
                return None
            
            # Create reservation
            reservation_id = str(uuid.uuid4())
            self.resource_reservations[reservation_id] = {
                "task_id": task_id,
                "worker_id": worker_id,
                "resources": {
                    "cpu_cores": cpu_cores,
                    "memory_mb": memory_mb,
                    "gpu_memory_mb": gpu_memory_mb
                },
                "reservation_time": time.time(),
                "expiration_time": time.time() + 3600  # 1 hour timeout by default
            }
            
            # Update worker available resources
            worker_resources = self.worker_resources[worker_id]["resources"]
            worker_resources["cpu"]["available_cores"] -= cpu_cores
            worker_resources["memory"]["available_mb"] -= memory_mb
            if gpu_memory_mb > 0 and "gpu" in worker_resources:
                worker_resources["gpu"]["available_memory_mb"] -= gpu_memory_mb
            
            # Update task tracking
            self.task_reservation[task_id] = reservation_id
            self.worker_tasks[worker_id].add(task_id)
            
            logger.info(f"Reserved resources for task {task_id} on worker {worker_id} (reservation {reservation_id})")
            return reservation_id
        
        except Exception as e:
            logger.error(f"Error reserving resources for task {task_id}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def release_resources(self, reservation_id: str) -> bool:
        """
        Release resources after task completion.
        
        Args:
            reservation_id: Reservation ID to release
        
        Returns:
            bool: Success status
        """
        try:
            if reservation_id not in self.resource_reservations:
                logger.warning(f"Reservation {reservation_id} not found")
                return False
            
            # Get reservation data
            reservation = self.resource_reservations[reservation_id]
            task_id = reservation["task_id"]
            worker_id = reservation["worker_id"]
            
            if worker_id not in self.worker_resources:
                logger.warning(f"Worker {worker_id} not registered, cleaning up reservation")
                self.resource_reservations.pop(reservation_id, None)
                self.task_reservation.pop(task_id, None)
                return True
            
            # Update worker available resources
            worker_resources = self.worker_resources[worker_id]["resources"]
            worker_resources["cpu"]["available_cores"] += reservation["resources"]["cpu_cores"]
            worker_resources["memory"]["available_mb"] += reservation["resources"]["memory_mb"]
            if reservation["resources"]["gpu_memory_mb"] > 0 and "gpu" in worker_resources:
                worker_resources["gpu"]["available_memory_mb"] += reservation["resources"]["gpu_memory_mb"]
            
            # Update task tracking
            self.task_reservation.pop(task_id, None)
            if worker_id in self.worker_tasks:
                self.worker_tasks[worker_id].discard(task_id)
            
            # Remove reservation
            self.resource_reservations.pop(reservation_id, None)
            
            logger.info(f"Released resources for task {task_id} (reservation {reservation_id})")
            return True
        
        except Exception as e:
            logger.error(f"Error releasing resources for reservation {reservation_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def get_worker_utilization(self, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get utilization metrics for a worker or all workers.
        
        Args:
            worker_id: Specific worker ID or None for all workers
        
        Returns:
            dict: Utilization metrics
        """
        if worker_id:
            if worker_id not in self.worker_resources:
                return {}
            return self._calculate_worker_utilization(worker_id)
        
        # Get utilization for all workers
        result = {}
        for worker_id in self.worker_resources:
            result[worker_id] = self._calculate_worker_utilization(worker_id)
        
        # Calculate overall system utilization
        overall = {
            "cpu": 0.0,
            "memory": 0.0,
            "gpu": 0.0,
            "overall": 0.0
        }
        
        if not result:
            return {"workers": {}, "overall": overall}
        
        # Average across all workers
        worker_count = len(result)
        for worker_id, metrics in result.items():
            overall["cpu"] += metrics["utilization"].get("cpu", 0.0)
            overall["memory"] += metrics["utilization"].get("memory", 0.0)
            overall["gpu"] += metrics["utilization"].get("gpu", 0.0)
            overall["overall"] += metrics["utilization"].get("overall", 0.0)
        
        # Calculate averages
        overall["cpu"] /= worker_count
        overall["memory"] /= worker_count
        overall["gpu"] /= worker_count
        overall["overall"] /= worker_count
        
        return {"workers": result, "overall": overall}
    
    def evaluate_scaling(self) -> ScalingDecision:
        """
        Evaluate if scaling up or down is needed based on current utilization.
        
        Returns:
            ScalingDecision: Scaling decision with details
        """
        with self.scaling_evaluation_lock:
            # Get current utilization
            utilization_data = self.get_worker_utilization()
            overall_utilization = utilization_data["overall"]["overall"]
            
            # Default response (no scaling needed)
            decision = ScalingDecision(
                action="maintain",
                reason="Current utilization within target range",
                utilization=overall_utilization,
                timestamp=time.time()
            )
            
            # Check cooldown periods
            current_time = time.time()
            if current_time - self.last_scale_up_time < self.scale_up_cooldown:
                decision.reason = f"In scale-up cooldown period ({int(self.scale_up_cooldown - (current_time - self.last_scale_up_time))}s remaining)"
                return decision
            
            if current_time - self.last_scale_down_time < self.scale_down_cooldown:
                decision.reason = f"In scale-down cooldown period ({int(self.scale_down_cooldown - (current_time - self.last_scale_down_time))}s remaining)"
                return decision
            
            # No workers yet - recommend creating initial worker
            if len(self.worker_resources) == 0:
                self.last_scale_up_time = current_time
                decision.action = "scale_up"
                decision.reason = "No workers available, creating initial worker"
                decision.count = 1
                decision.worker_type = "default"
                decision.resource_requirements = {
                    "cpu_cores": 2,
                    "memory_mb": 4096
                }
                logger.info(f"Scale-up recommended: creating initial worker")
                return decision
            
            # Check if scaling is needed
            if overall_utilization >= self.scale_up_threshold:
                # Need to scale up
                self.last_scale_up_time = current_time
                
                # Determine scale-up size based on utilization
                scale_factor = min(max(1.0, overall_utilization / self.target_utilization), 2.0)
                worker_count = len(self.worker_resources)
                additional_workers = max(1, int(worker_count * (scale_factor - 1.0)))
                
                # Determine worker type based on resource requirements
                worker_type = "default"
                if overall_utilization > 0.9:  # Critical utilization, determine bottleneck
                    cpu_util = utilization_data["overall"]["cpu"]
                    memory_util = utilization_data["overall"]["memory"]
                    gpu_util = utilization_data["overall"]["gpu"]
                    
                    if gpu_util > 0.9:  # GPU is the bottleneck
                        worker_type = "gpu"
                    elif memory_util > 0.9:  # Memory is the bottleneck
                        worker_type = "memory"
                    elif cpu_util > 0.9:  # CPU is the bottleneck
                        worker_type = "cpu"
                
                # Determine resource requirements based on workload
                resource_requirements = {
                    "cpu_cores": 4,
                    "memory_mb": 8192
                }
                
                if worker_type == "gpu":
                    resource_requirements["gpu_memory_mb"] = 8192
                elif worker_type == "memory":
                    resource_requirements["memory_mb"] = 32768
                elif worker_type == "cpu":
                    resource_requirements["cpu_cores"] = 8
                
                # Create decision
                decision = ScalingDecision(
                    action="scale_up",
                    reason=f"Utilization ({overall_utilization:.1%}) exceeds scale-up threshold ({self.scale_up_threshold:.1%})",
                    count=additional_workers,
                    utilization=overall_utilization,
                    worker_type=worker_type,
                    resource_requirements=resource_requirements
                )
                
                logger.info(f"Scale-up recommended: {additional_workers} additional {worker_type} workers")
                
            elif overall_utilization <= self.scale_down_threshold and len(self.worker_resources) > 1:
                # Need to scale down (but keep at least one worker)
                self.last_scale_down_time = current_time
                
                # Determine scale-down size based on utilization
                scale_factor = max(0.5, overall_utilization / self.target_utilization)
                worker_count = len(self.worker_resources)
                target_count = max(1, int(worker_count * scale_factor))
                workers_to_remove = worker_count - target_count
                
                # Find workers to scale down
                workers_to_remove_ids = self._get_workers_to_scale_down(workers_to_remove)
                
                # Create decision
                decision = ScalingDecision(
                    action="scale_down",
                    reason=f"Utilization ({overall_utilization:.1%}) below scale-down threshold ({self.scale_down_threshold:.1%})",
                    count=len(workers_to_remove_ids),
                    worker_ids=workers_to_remove_ids,
                    utilization=overall_utilization
                )
                
                logger.info(f"Scale-down recommended: remove {len(workers_to_remove_ids)} workers")
            
            return decision
    
    def predict_resource_needs(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict resource needs for a task based on historical data.
        
        Args:
            task_data: Task metadata including model, batch size, etc.
        
        Returns:
            dict: Predicted resource requirements
        """
        if not PREDICTOR_AVAILABLE or not self.predictor:
            # Fallback to conservative estimates if predictor not available
            return {
                "cpu_cores": task_data.get("cpu_cores", 2),
                "memory_mb": task_data.get("memory_mb", 4096),
                "gpu_memory_mb": task_data.get("gpu_memory_mb", 2048),
                "predicted": False
            }
        
        try:
            # Get predictions from the performance predictor
            prediction = self.predictor.predict_resource_requirements(task_data)
            prediction["predicted"] = True
            return prediction
        
        except Exception as e:
            logger.error(f"Error predicting resource needs: {e}")
            logger.debug(traceback.format_exc())
            
            # Fallback to conservative estimates
            return {
                "cpu_cores": task_data.get("cpu_cores", 2),
                "memory_mb": task_data.get("memory_mb", 4096),
                "gpu_memory_mb": task_data.get("gpu_memory_mb", 2048),
                "predicted": False
            }
    
    def record_task_execution(self, task_id: str, execution_data: Dict[str, Any]) -> bool:
        """
        Record task execution data for future predictions.
        
        Args:
            task_id: Task ID
            execution_data: Execution metrics including duration, resource usage, etc.
        
        Returns:
            bool: Success status
        """
        if not PREDICTOR_AVAILABLE or not self.predictor:
            return False
        
        try:
            # Get reservation data
            reservation_id = self.task_reservation.get(task_id)
            if not reservation_id or reservation_id not in self.resource_reservations:
                logger.warning(f"No reservation found for task {task_id}")
                return False
            
            reservation = self.resource_reservations[reservation_id]
            worker_id = reservation["worker_id"]
            
            # Update worker performance metrics
            if worker_id in self.worker_performance:
                perf = self.worker_performance[worker_id]
                task_count = perf["task_count"]
                success = execution_data.get("success", True)
                
                # Update success rate
                if task_count > 0:
                    perf["success_rate"] = ((perf["success_rate"] * task_count) + (1.0 if success else 0.0)) / (task_count + 1)
                else:
                    perf["success_rate"] = 1.0 if success else 0.0
                
                # Update average execution time
                execution_time = execution_data.get("execution_time_ms", 0) / 1000.0  # Convert to seconds
                if task_count > 0:
                    perf["avg_execution_time"] = ((perf["avg_execution_time"] * task_count) + execution_time) / (task_count + 1)
                else:
                    perf["avg_execution_time"] = execution_time
                
                # Update resource efficiency
                reserved_resources = reservation["resources"]
                used_resources = {
                    "cpu_cores": execution_data.get("cpu_cores_used", reserved_resources["cpu_cores"]),
                    "memory_mb": execution_data.get("memory_mb_used", reserved_resources["memory_mb"]),
                    "gpu_memory_mb": execution_data.get("gpu_memory_mb_used", reserved_resources["gpu_memory_mb"])
                }
                
                efficiency = self._calculate_resource_efficiency(reserved_resources, used_resources)
                if task_count > 0:
                    perf["resource_efficiency"] = ((perf["resource_efficiency"] * task_count) + efficiency) / (task_count + 1)
                else:
                    perf["resource_efficiency"] = efficiency
                
                # Update task count and timestamp
                perf["task_count"] += 1
                perf["last_updated"] = time.time()
            
            # Record in predictor
            if self.predictor:
                self.predictor.record_task_execution(task_id, execution_data)
            
            return True
        
        except Exception as e:
            logger.error(f"Error recording task execution data: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def get_worker_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all workers.
        
        Returns:
            dict: Worker statistics
        """
        result = {
            "total_workers": len(self.worker_resources),
            "active_tasks": sum(len(tasks) for tasks in self.worker_tasks.values()),
            "resource_reservations": len(self.resource_reservations),
            "workers": {},
            "overall_utilization": {}
        }
        
        # Get overall utilization
        utilization_data = self.get_worker_utilization()
        result["overall_utilization"] = utilization_data["overall"]
        
        # Get worker-specific statistics
        for worker_id, resources in self.worker_resources.items():
            result["workers"][worker_id] = {
                "resources": resources,
                "tasks": len(self.worker_tasks.get(worker_id, set())),
                "performance": self.worker_performance.get(worker_id, {}),
                "utilization": utilization_data["workers"].get(worker_id, {}).get("utilization", {})
            }
        
        return result
    
    def cleanup(self) -> None:
        """
        Cleanup resources and stop background threads.
        """
        self.running = False
        
        # Wait for threads to terminate
        if self.resource_tracker_thread.is_alive():
            self.resource_tracker_thread.join(timeout=5.0)
        
        if self.worker_reassessment_thread.is_alive():
            self.worker_reassessment_thread.join(timeout=5.0)
        
        logger.info("Dynamic Resource Manager cleaned up")
    
    # Internal helper methods
    def _validate_resource_data(self, resources: Dict[str, Any]) -> bool:
        """Validate worker resource data."""
        # Basic validation
        if not isinstance(resources, dict):
            return False
        
        # Check required fields
        if "cpu" not in resources or "memory" not in resources:
            return False
        
        # Validate CPU data
        cpu = resources["cpu"]
        if not isinstance(cpu, dict) or "cores" not in cpu or "available_cores" not in cpu:
            return False
        
        if not isinstance(cpu["cores"], int) or not isinstance(cpu["available_cores"], int):
            return False
        
        if cpu["cores"] <= 0 or cpu["available_cores"] < 0 or cpu["available_cores"] > cpu["cores"]:
            return False
        
        # Validate memory data
        memory = resources["memory"]
        if not isinstance(memory, dict) or "total_mb" not in memory or "available_mb" not in memory:
            return False
        
        if not isinstance(memory["total_mb"], int) or not isinstance(memory["available_mb"], int):
            return False
        
        if memory["total_mb"] <= 0 or memory["available_mb"] < 0 or memory["available_mb"] > memory["total_mb"]:
            return False
        
        # Validate GPU data if present
        if "gpu" in resources:
            gpu = resources["gpu"]
            if not isinstance(gpu, dict):
                return False
            
            # Check if required GPU fields are present
            if "devices" not in gpu or "memory_mb" not in gpu or "available_memory_mb" not in gpu:
                return False
            
            if not isinstance(gpu["devices"], int) or not isinstance(gpu["memory_mb"], int) or not isinstance(gpu["available_memory_mb"], int):
                return False
            
            if gpu["devices"] < 0 or gpu["memory_mb"] < 0 or gpu["available_memory_mb"] < 0 or gpu["available_memory_mb"] > gpu["memory_mb"]:
                return False
        
        return True
    
    def _validate_resource_requirements(self, requirements: Dict[str, Any]) -> bool:
        """Validate task resource requirements."""
        # Basic validation
        if not isinstance(requirements, dict):
            return False
        
        # Check CPU cores
        if "cpu_cores" in requirements:
            if not isinstance(requirements["cpu_cores"], int) or requirements["cpu_cores"] <= 0:
                return False
        
        # Check memory
        if "memory_mb" in requirements:
            if not isinstance(requirements["memory_mb"], int) or requirements["memory_mb"] <= 0:
                return False
        
        # Check GPU memory
        if "gpu_memory_mb" in requirements:
            if not isinstance(requirements["gpu_memory_mb"], int) or requirements["gpu_memory_mb"] < 0:
                return False
        
        # If a specific worker is requested, check it exists
        if "worker_id" in requirements and requirements["worker_id"]:
            if requirements["worker_id"] not in self.worker_resources:
                return False
        
        return True
    
    def _find_suitable_worker(self, cpu_cores: int, memory_mb: int, gpu_memory_mb: int, 
                              requested_worker_id: Optional[str] = None) -> Optional[str]:
        """Find a suitable worker for the given resource requirements."""
        # If a specific worker is requested, check if it can handle the task
        if requested_worker_id:
            if requested_worker_id in self.worker_resources:
                worker = self.worker_resources[requested_worker_id]
                
                # Check if worker has enough resources
                if (worker["resources"]["cpu"]["available_cores"] >= cpu_cores and
                    worker["resources"]["memory"]["available_mb"] >= memory_mb and
                    (gpu_memory_mb == 0 or 
                     ("gpu" in worker["resources"] and 
                      worker["resources"]["gpu"]["available_memory_mb"] >= gpu_memory_mb))):
                    return requested_worker_id
            
            # Requested worker not available or doesn't have enough resources
            logger.warning(f"Requested worker {requested_worker_id} not suitable")
            return None
        
        # Find the most suitable worker
        suitable_workers = []
        
        for worker_id, worker in self.worker_resources.items():
            # Check if worker has enough resources
            if (worker["resources"]["cpu"]["available_cores"] >= cpu_cores and
                worker["resources"]["memory"]["available_mb"] >= memory_mb and
                (gpu_memory_mb == 0 or 
                 ("gpu" in worker["resources"] and 
                  worker["resources"]["gpu"]["available_memory_mb"] >= gpu_memory_mb))):
                
                # Calculate a fitness score (lower is better)
                cpu_fitness = worker["resources"]["cpu"]["available_cores"] / max(1, worker["resources"]["cpu"]["cores"])
                memory_fitness = worker["resources"]["memory"]["available_mb"] / max(1, worker["resources"]["memory"]["total_mb"])
                
                if gpu_memory_mb > 0 and "gpu" in worker["resources"]:
                    gpu_fitness = worker["resources"]["gpu"]["available_memory_mb"] / max(1, worker["resources"]["gpu"]["memory_mb"])
                    overall_fitness = (cpu_fitness + memory_fitness + gpu_fitness) / 3.0
                else:
                    overall_fitness = (cpu_fitness + memory_fitness) / 2.0
                
                # Account for worker performance
                if worker_id in self.worker_performance:
                    perf = self.worker_performance[worker_id]
                    success_rate = perf.get("success_rate", 1.0)
                    efficiency = perf.get("resource_efficiency", 1.0)
                    
                    # Adjust fitness by performance (higher performance -> lower fitness score)
                    performance_factor = 1.0 - ((success_rate + efficiency) / 2.0)
                    overall_fitness = overall_fitness * (1.0 + performance_factor)
                
                suitable_workers.append((worker_id, overall_fitness))
        
        if not suitable_workers:
            return None
        
        # Sort by fitness score (lower is better)
        suitable_workers.sort(key=lambda x: x[1])
        
        # Return the best fit worker
        return suitable_workers[0][0]
    
    def _calculate_worker_utilization(self, worker_id: str) -> Dict[str, Any]:
        """Calculate utilization metrics for a worker."""
        if worker_id not in self.worker_resources:
            return {}
        
        worker = self.worker_resources[worker_id]
        resources = worker["resources"]
        
        # Calculate CPU utilization
        cpu_total = resources["cpu"]["cores"]
        cpu_available = resources["cpu"]["available_cores"]
        cpu_utilization = (cpu_total - cpu_available) / max(1, cpu_total)
        
        # Calculate memory utilization
        memory_total = resources["memory"]["total_mb"]
        memory_available = resources["memory"]["available_mb"]
        memory_utilization = (memory_total - memory_available) / max(1, memory_total)
        
        # Calculate GPU utilization if available
        gpu_utilization = 0.0
        if "gpu" in resources and resources["gpu"]["devices"] > 0:
            gpu_memory_total = resources["gpu"]["memory_mb"]
            gpu_memory_available = resources["gpu"]["available_memory_mb"]
            gpu_utilization = (gpu_memory_total - gpu_memory_available) / max(1, gpu_memory_total)
        
        # Calculate overall utilization
        if "gpu" in resources and resources["gpu"]["devices"] > 0:
            overall_utilization = (cpu_utilization + memory_utilization + gpu_utilization) / 3.0
        else:
            overall_utilization = (cpu_utilization + memory_utilization) / 2.0
        
        # Update worker utilization data
        self.worker_resources[worker_id]["utilization"] = {
            "cpu": cpu_utilization,
            "memory": memory_utilization,
            "gpu": gpu_utilization,
            "overall": overall_utilization
        }
        
        # Return complete result
        return {
            "worker_id": worker_id,
            "resources": resources,
            "utilization": {
                "cpu": cpu_utilization,
                "memory": memory_utilization,
                "gpu": gpu_utilization,
                "overall": overall_utilization
            },
            "tasks": len(self.worker_tasks.get(worker_id, set())),
            "performance": self.worker_performance.get(worker_id, {})
        }
    
    def _get_workers_to_scale_down(self, count: int) -> List[str]:
        """
        Determine which workers to scale down based on utilization and performance.
        
        Args:
            count: Number of workers to scale down
        
        Returns:
            list: Worker IDs to scale down
        """
        # Can't scale down more workers than we have
        count = min(count, len(self.worker_resources) - 1)
        if count <= 0:
            return []
        
        # Calculate metric for each worker (lower is better candidate for scale-down)
        worker_metrics = []
        
        for worker_id, worker in self.worker_resources.items():
            # Skip workers with active tasks
            if worker_id in self.worker_tasks and self.worker_tasks[worker_id]:
                continue
            
            # Get utilization
            utilization = worker["utilization"]["overall"]
            
            # Get performance metrics
            perf = self.worker_performance.get(worker_id, {})
            success_rate = perf.get("success_rate", 1.0)
            efficiency = perf.get("resource_efficiency", 1.0)
            
            # Calculate score (lower is better candidate for scale-down)
            # Low utilization + high success rate + high efficiency = good scale-down candidate
            score = utilization + (1.0 - success_rate) + (1.0 - efficiency)
            
            worker_metrics.append((worker_id, score))
        
        # Sort by score (lower is better)
        worker_metrics.sort(key=lambda x: x[1])
        
        # Return the best candidates up to count
        return [worker_id for worker_id, _ in worker_metrics[:count]]
    
    def _calculate_resource_efficiency(self, reserved: Dict[str, Any], used: Dict[str, Any]) -> float:
        """Calculate resource efficiency (ratio of used to reserved resources)."""
        if not reserved or not used:
            return 1.0
        
        # Calculate CPU efficiency
        cpu_reserved = reserved.get("cpu_cores", 1)
        cpu_used = used.get("cpu_cores", cpu_reserved)
        cpu_efficiency = min(1.0, cpu_used / max(1, cpu_reserved))
        
        # Calculate memory efficiency
        memory_reserved = reserved.get("memory_mb", 1)
        memory_used = used.get("memory_mb", memory_reserved)
        memory_efficiency = min(1.0, memory_used / max(1, memory_reserved))
        
        # Calculate GPU efficiency if applicable
        gpu_efficiency = 1.0
        if reserved.get("gpu_memory_mb", 0) > 0:
            gpu_reserved = reserved.get("gpu_memory_mb", 1)
            gpu_used = used.get("gpu_memory_mb", gpu_reserved)
            gpu_efficiency = min(1.0, gpu_used / max(1, gpu_reserved))
            
            # Overall is average of all three
            overall_efficiency = (cpu_efficiency + memory_efficiency + gpu_efficiency) / 3.0
        else:
            # Overall is average of CPU and memory
            overall_efficiency = (cpu_efficiency + memory_efficiency) / 2.0
        
        return overall_efficiency
    
    def _resource_tracker_loop(self) -> None:
        """Background thread for tracking resource usage over time."""
        while self.running:
            try:
                # Get current resource usage
                utilization_data = self.get_worker_utilization()
                
                # Create resource snapshot
                snapshot = {
                    "timestamp": time.time(),
                    "worker_count": len(self.worker_resources),
                    "overall_utilization": utilization_data["overall"],
                    "worker_utilization": {
                        worker_id: data["utilization"] 
                        for worker_id, data in utilization_data["workers"].items()
                    }
                }
                
                # Add to resource history
                self.resource_history.append(snapshot)
                
                # Prune history to retain only recent entries
                current_time = time.time()
                self.resource_history = [
                    entry for entry in self.resource_history
                    if current_time - entry["timestamp"] <= self.history_retention
                ]
                
            except Exception as e:
                logger.error(f"Error in resource tracker: {e}")
                logger.debug(traceback.format_exc())
            
            # Sleep for evaluation window / 10 to get enough data points
            time.sleep(self.evaluation_window / 10)
    
    def _worker_reassessment_loop(self) -> None:
        """Background thread for reassessing worker capabilities."""
        while self.running:
            try:
                # Check if any workers need reassessment
                current_time = time.time()
                for worker_id, worker in list(self.worker_resources.items()):
                    last_updated = worker.get("last_updated", 0)
                    if current_time - last_updated >= self.worker_reassessment_interval:
                        # Worker needs reassessment
                        logger.info(f"Worker {worker_id} due for reassessment")
                        # Note: actual reassessment happens via API call from worker
            
            except Exception as e:
                logger.error(f"Error in worker reassessment: {e}")
                logger.debug(traceback.format_exc())
            
            # Sleep for worker reassessment interval / 10 to check more frequently
            time.sleep(self.worker_reassessment_interval / 10)


# Main function for testing
if __name__ == "__main__":
    """Run standalone test of the Dynamic Resource Manager."""
    manager = DynamicResourceManager()
    
    # Register test workers
    manager.register_worker("worker-1", {
        "cpu": {"cores": 8, "available_cores": 8},
        "memory": {"total_mb": 16384, "available_mb": 16384},
        "gpu": {"devices": 1, "memory_mb": 8192, "available_memory_mb": 8192}
    })
    
    manager.register_worker("worker-2", {
        "cpu": {"cores": 4, "available_cores": 4},
        "memory": {"total_mb": 8192, "available_mb": 8192}
    })
    
    # Reserve resources
    reservation_id = manager.reserve_resources("task-1", {
        "cpu_cores": 2,
        "memory_mb": 4096,
        "gpu_memory_mb": 2048
    })
    
    if reservation_id:
        print(f"Reserved resources: {reservation_id}")
        
        # Get utilization
        utilization = manager.get_worker_utilization()
        print(f"Utilization: {json.dumps(utilization, indent=2)}")
        
        # Release resources
        manager.release_resources(reservation_id)
        
        # Evaluate scaling
        scaling = manager.evaluate_scaling()
        print(f"Scaling decision: {json.dumps(scaling, indent=2)}")
    
    # Cleanup
    manager.cleanup()