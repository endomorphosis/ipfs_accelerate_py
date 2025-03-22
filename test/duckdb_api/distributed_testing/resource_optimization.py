#!/usr/bin/env python3
"""
Distributed Testing Framework - Resource Optimization

This module integrates the Dynamic Resource Manager and Resource Performance Predictor
to enable intelligent resource allocation and workload optimization.

Core responsibilities:
- Optimize resource allocation for tasks based on historical performance data
- Predict resource requirements for different task types and batch sizes
- Cluster similar workloads for efficient batch processing 
- Balance resources across task types for optimal throughput
- Provide scaling recommendations for different workload patterns

Usage:
    # Import and initialize
    from duckdb_api.distributed_testing.resource_optimization import ResourceOptimizer
    
    # Create optimizer
    optimizer = ResourceOptimizer(
        resource_manager=dynamic_resource_manager,
        performance_predictor=resource_predictor
    )
    
    # Get optimized resource allocation for a batch of tasks
    allocation = optimizer.allocate_resources(
        task_batch=[task1, task2, task3],
        available_workers=["worker1", "worker2"]
    )
    
    # Get worker type recommendations for scheduled tasks
    recommendations = optimizer.recommend_worker_types(pending_tasks)
    
    # Get scaling recommendations based on workload patterns
    scaling_recommendations = optimizer.get_scaling_recommendations()
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("resource_optimization")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import required components
try:
    from duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager, ScalingDecision
    DRM_AVAILABLE = True
except ImportError:
    logger.warning("Dynamic Resource Manager not available. Limited optimization capabilities.")
    DRM_AVAILABLE = False

try:
    from duckdb_api.distributed_testing.resource_performance_predictor import ResourcePerformancePredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    logger.warning("Resource Performance Predictor not available. Using default resource predictions.")
    PREDICTOR_AVAILABLE = False

try:
    from duckdb_api.distributed_testing.cloud_provider_manager import CloudProviderManager
    CLOUD_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("Cloud Provider Manager not available. Limited provider capabilities.")
    CLOUD_MANAGER_AVAILABLE = False

# Try to import optional ML dependencies for advanced optimization
try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    logger.warning("ML libraries not available. Using basic optimization methods.")
    ML_AVAILABLE = False


@dataclass
class TaskRequirements:
    """Class for task resource requirements."""
    cpu_cores: int = 2
    memory_mb: int = 4096
    gpu_memory_mb: int = 0
    disk_mb: int = 1024
    network_bandwidth_mbps: int = 100
    execution_time_ms: int = 0
    priority: int = 5
    model_type: str = "default"
    model_name: str = "unknown"
    batch_size: int = 1
    confidence: float = 0.7
    prediction_method: str = "default"


@dataclass
class WorkerTypeRecommendation:
    """Class for worker type recommendations."""
    recommended_type: str
    reason: str
    required_resources: Dict[str, Any]
    estimated_task_count: int
    estimated_utilization: float
    priority: int = 3  # Higher is more important
    provider: Optional[str] = None


@dataclass
class ResourceAllocation:
    """Class for resource allocation result."""
    task_id: str
    worker_id: str
    reservation_id: Optional[str] = None
    start_time: Optional[float] = None
    estimated_completion_time: Optional[float] = None
    requirements: Dict[str, Any] = field(default_factory=dict)
    allocated: bool = False
    reason: str = ""


class ResourceOptimizer:
    """
    Resource Optimizer for the distributed testing framework.
    
    Integrates the Dynamic Resource Manager and Resource Performance Predictor
    to enable intelligent resource allocation and workload optimization.
    """
    
    def __init__(self, resource_manager: Optional[DynamicResourceManager] = None,
                 performance_predictor: Optional[ResourcePerformancePredictor] = None,
                 cloud_manager: Optional[CloudProviderManager] = None):
        """
        Initialize the Resource Optimizer.
        
        Args:
            resource_manager: Dynamic Resource Manager instance
            performance_predictor: Resource Performance Predictor instance
            cloud_manager: Cloud Provider Manager instance
        """
        self.resource_manager = resource_manager
        self.performance_predictor = performance_predictor
        self.cloud_manager = cloud_manager
        
        # Initialize internal state
        self.workload_history = []
        self.worker_type_cache = {}  # Cache for worker type recommendations
        self.last_optimization_time = 0
        self.optimization_interval = 300  # 5 minutes
        
        # Workload clustering
        self.workload_clusters = None
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("Resource Optimizer initialized")
    
    def predict_task_requirements(self, task_data: Dict[str, Any]) -> TaskRequirements:
        """
        Predict resource requirements for a task.
        
        Args:
            task_data: Task metadata
        
        Returns:
            TaskRequirements: Predicted resource requirements
        """
        try:
            # Use performance predictor if available
            if PREDICTOR_AVAILABLE and self.performance_predictor:
                prediction = self.performance_predictor.predict_resource_requirements(task_data)
                
                # Convert to TaskRequirements
                requirements = TaskRequirements(
                    cpu_cores=prediction.get("cpu_cores", 2),
                    memory_mb=prediction.get("memory_mb", 4096),
                    gpu_memory_mb=prediction.get("gpu_memory_mb", 0),
                    model_type=task_data.get("model_type", "default"),
                    model_name=task_data.get("model_name", "unknown"),
                    batch_size=task_data.get("batch_size", 1),
                    confidence=prediction.get("confidence", 0.7),
                    prediction_method=prediction.get("prediction_method", "default"),
                    execution_time_ms=prediction.get("execution_time_ms", 0)
                )
                
                return requirements
            
            # Fallback to default prediction
            return self._default_task_requirements(task_data)
        
        except Exception as e:
            logger.error(f"Error predicting task requirements: {e}")
            logger.debug(traceback.format_exc())
            
            # Fallback to default requirements
            return self._default_task_requirements(task_data)
    
    def allocate_resources(self, task_batch: List[Dict[str, Any]], 
                          available_workers: List[str]) -> List[ResourceAllocation]:
        """
        Allocate resources for a batch of tasks across available workers.
        
        Args:
            task_batch: List of tasks to allocate
            available_workers: List of available worker IDs
        
        Returns:
            List[ResourceAllocation]: Resource allocation results
        """
        try:
            if not DRM_AVAILABLE or not self.resource_manager:
                logger.warning("Dynamic Resource Manager not available. Cannot allocate resources.")
                return [
                    ResourceAllocation(
                        task_id=task.get("task_id", "unknown"),
                        worker_id="",
                        allocated=False,
                        reason="Dynamic Resource Manager not available"
                    ) for task in task_batch
                ]
            
            # Sort tasks by priority (higher first)
            sorted_tasks = sorted(task_batch, 
                                 key=lambda t: t.get("priority", 5),
                                 reverse=True)
            
            # Get worker resources
            worker_resources = {}
            for worker_id in available_workers:
                if worker_id in self.resource_manager.worker_resources:
                    worker_resources[worker_id] = self.resource_manager.worker_resources[worker_id]
            
            # Task allocation result
            allocations = []
            
            # Track remaining resources
            remaining_resources = {
                worker_id: worker_data["resources"].copy() 
                for worker_id, worker_data in worker_resources.items()
            }
            
            # Allocate tasks
            for task in sorted_tasks:
                task_id = task.get("task_id", str(uuid.uuid4()))
                
                # Predict resource requirements
                requirements = self.predict_task_requirements({
                    "model_type": task.get("config", {}).get("model_type", "default"),
                    "model_name": task.get("config", {}).get("model", "unknown"),
                    "batch_size": task.get("config", {}).get("batch_size", 1)
                })
                
                # Find suitable worker
                best_worker_id = None
                best_fit_score = float('inf')
                
                for worker_id, resources in remaining_resources.items():
                    # Check if worker has enough resources
                    if (resources["cpu"]["available_cores"] >= requirements.cpu_cores and
                        resources["memory"]["available_mb"] >= requirements.memory_mb and
                        (requirements.gpu_memory_mb == 0 or 
                         ("gpu" in resources and 
                          resources["gpu"]["available_memory_mb"] >= requirements.gpu_memory_mb))):
                        
                        # Calculate fitness score (lower is better)
                        # This prefers workers with just enough resources
                        cpu_ratio = resources["cpu"]["available_cores"] / requirements.cpu_cores
                        memory_ratio = resources["memory"]["available_mb"] / requirements.memory_mb
                        
                        if requirements.gpu_memory_mb > 0 and "gpu" in resources:
                            gpu_ratio = resources["gpu"]["available_memory_mb"] / requirements.gpu_memory_mb
                            fitness_score = (cpu_ratio + memory_ratio + gpu_ratio) / 3
                        else:
                            fitness_score = (cpu_ratio + memory_ratio) / 2
                        
                        # Check if this is the best fit so far
                        if fitness_score < best_fit_score:
                            best_fit_score = fitness_score
                            best_worker_id = worker_id
                
                # Allocate resources if a suitable worker was found
                if best_worker_id:
                    # Create resource reservation
                    try:
                        reservation_id = self.resource_manager.reserve_resources(
                            task_id=task_id,
                            resource_requirements={
                                "cpu_cores": requirements.cpu_cores,
                                "memory_mb": requirements.memory_mb,
                                "gpu_memory_mb": requirements.gpu_memory_mb,
                                "worker_id": best_worker_id
                            }
                        )
                        
                        if reservation_id:
                            # Update remaining resources
                            remaining_resources[best_worker_id]["cpu"]["available_cores"] -= requirements.cpu_cores
                            remaining_resources[best_worker_id]["memory"]["available_mb"] -= requirements.memory_mb
                            
                            if requirements.gpu_memory_mb > 0 and "gpu" in remaining_resources[best_worker_id]:
                                remaining_resources[best_worker_id]["gpu"]["available_memory_mb"] -= requirements.gpu_memory_mb
                            
                            # Create allocation result
                            allocation = ResourceAllocation(
                                task_id=task_id,
                                worker_id=best_worker_id,
                                reservation_id=reservation_id,
                                start_time=time.time(),
                                estimated_completion_time=(time.time() + requirements.execution_time_ms / 1000) 
                                                          if requirements.execution_time_ms > 0 else None,
                                requirements=asdict(requirements),
                                allocated=True,
                                reason="Successfully allocated resources"
                            )
                            
                            logger.debug(f"Allocated task {task_id} to worker {best_worker_id} (reservation {reservation_id})")
                        else:
                            # Reservation failed
                            allocation = ResourceAllocation(
                                task_id=task_id,
                                worker_id=best_worker_id,
                                allocated=False,
                                reason="Failed to create resource reservation"
                            )
                            
                            logger.warning(f"Failed to create resource reservation for task {task_id} on worker {best_worker_id}")
                    
                    except Exception as e:
                        # Error during reservation
                        allocation = ResourceAllocation(
                            task_id=task_id,
                            worker_id=best_worker_id,
                            allocated=False,
                            reason=f"Error creating resource reservation: {str(e)}"
                        )
                        
                        logger.error(f"Error creating resource reservation for task {task_id}: {e}")
                        logger.debug(traceback.format_exc())
                else:
                    # No suitable worker found
                    allocation = ResourceAllocation(
                        task_id=task_id,
                        worker_id="",
                        allocated=False,
                        reason="No suitable worker found"
                    )
                    
                    logger.info(f"No suitable worker found for task {task_id}")
                
                allocations.append(allocation)
            
            return allocations
        
        except Exception as e:
            logger.error(f"Error allocating resources: {e}")
            logger.debug(traceback.format_exc())
            
            # Return failed allocations
            return [
                ResourceAllocation(
                    task_id=task.get("task_id", "unknown"),
                    worker_id="",
                    allocated=False,
                    reason=f"Error allocating resources: {str(e)}"
                ) for task in task_batch
            ]
    
    def recommend_worker_types(self, pending_tasks: List[Dict[str, Any]]) -> List[WorkerTypeRecommendation]:
        """
        Recommend worker types based on pending tasks.
        
        Args:
            pending_tasks: List of pending tasks
        
        Returns:
            List[WorkerTypeRecommendation]: Worker type recommendations
        """
        try:
            if not pending_tasks:
                return []
            
            # Group tasks by model type
            tasks_by_type = defaultdict(list)
            
            for task in pending_tasks:
                model_type = task.get("config", {}).get("model_type", "default")
                tasks_by_type[model_type].append(task)
            
            # Generate recommendations for each task type
            recommendations = []
            
            for model_type, tasks in tasks_by_type.items():
                # Skip if no tasks of this type
                if not tasks:
                    continue
                
                # Sample tasks for resource prediction
                sample_tasks = tasks[:5]  # Use up to 5 tasks for prediction
                
                # Predict resource requirements
                requirements_list = []
                for task in sample_tasks:
                    requirements = self.predict_task_requirements({
                        "model_type": task.get("config", {}).get("model_type", "default"),
                        "model_name": task.get("config", {}).get("model", "unknown"),
                        "batch_size": task.get("config", {}).get("batch_size", 1)
                    })
                    requirements_list.append(requirements)
                
                # Average resource requirements
                avg_cpu_cores = sum(r.cpu_cores for r in requirements_list) / len(requirements_list)
                avg_memory_mb = sum(r.memory_mb for r in requirements_list) / len(requirements_list)
                avg_gpu_memory_mb = sum(r.gpu_memory_mb for r in requirements_list) / len(requirements_list)
                
                # Determine worker type based on requirements
                worker_type, reason = self._determine_worker_type(
                    cpu_cores=avg_cpu_cores,
                    memory_mb=avg_memory_mb,
                    gpu_memory_mb=avg_gpu_memory_mb
                )
                
                # Calculate task capacity per worker
                capacity = self._calculate_worker_capacity(
                    worker_type=worker_type,
                    cpu_cores=avg_cpu_cores,
                    memory_mb=avg_memory_mb,
                    gpu_memory_mb=avg_gpu_memory_mb
                )
                
                # Determine minimum workers needed
                task_count = len(tasks)
                min_workers = max(1, task_count // capacity) if capacity > 0 else 1
                
                # Create recommendation
                recommendation = WorkerTypeRecommendation(
                    recommended_type=worker_type,
                    reason=reason,
                    required_resources={
                        "cpu_cores": avg_cpu_cores,
                        "memory_mb": avg_memory_mb,
                        "gpu_memory_mb": avg_gpu_memory_mb
                    },
                    estimated_task_count=task_count,
                    estimated_utilization=min(1.0, task_count / (capacity * min_workers)) if capacity > 0 else 1.0,
                    priority=3 if "gpu" in worker_type else (2 if "memory" in worker_type else 1),
                    provider=self._get_preferred_provider(
                        cpu_cores=avg_cpu_cores,
                        memory_mb=avg_memory_mb,
                        gpu_memory_mb=avg_gpu_memory_mb
                    )
                )
                
                recommendations.append(recommendation)
            
            # Sort recommendations by priority (higher first)
            recommendations.sort(key=lambda r: r.priority, reverse=True)
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error generating worker type recommendations: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def get_scaling_recommendations(self) -> ScalingDecision:
        """
        Get scaling recommendations based on current workload patterns.
        
        Returns:
            ScalingDecision: Enhanced scaling decision with optimized parameters
        """
        try:
            if not DRM_AVAILABLE or not self.resource_manager:
                logger.warning("Dynamic Resource Manager not available. Cannot provide scaling recommendations.")
                return ScalingDecision(
                    action="maintain",
                    reason="Dynamic Resource Manager not available",
                    utilization=0.0
                )
            
            # Get basic scaling decision from DRM
            base_decision = self.resource_manager.evaluate_scaling()
            
            # If no scaling needed, return the base decision
            if base_decision.action == "maintain":
                return base_decision
            
            # Get pending tasks from DRM
            task_queue = []  # This should be replaced with actual task queue from coordinator
            
            # Enhance scaling decision with predicted workload patterns
            if base_decision.action == "scale_up":
                # Get worker type recommendations
                worker_recommendations = self.recommend_worker_types(task_queue)
                
                # If we have recommendations, use them to enhance the decision
                if worker_recommendations:
                    # Use the highest priority recommendation
                    top_recommendation = worker_recommendations[0]
                    
                    # Update worker type and resource requirements
                    base_decision.worker_type = top_recommendation.recommended_type
                    base_decision.resource_requirements = top_recommendation.required_resources
                    base_decision.provider = top_recommendation.provider
                    
                    # Update reason with more details
                    base_decision.reason += f" - Optimized for {top_recommendation.recommended_type} workload: {top_recommendation.reason}"
            
            elif base_decision.action == "scale_down":
                # Enhance scale-down by ensuring we keep workers that match current workload patterns
                if self.resource_manager.worker_resources and task_queue:
                    # Find best workers to keep based on task requirements
                    worker_scores = self._score_workers_for_scaling(task_queue)
                    
                    # Sort workers by score (higher is better to keep)
                    sorted_workers = sorted(worker_scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # Determine which workers to remove (lowest scores first)
                    workers_to_keep = [w[0] for w in sorted_workers[:len(sorted_workers) - base_decision.count]]
                    workers_to_remove = [w for w in self.resource_manager.worker_resources if w not in workers_to_keep]
                    
                    # Update worker IDs in scaling decision
                    base_decision.worker_ids = workers_to_remove[:base_decision.count]
                    
                    # Update reason with more details
                    base_decision.reason += " - Workers selected based on task requirements"
            
            return base_decision
        
        except Exception as e:
            logger.error(f"Error generating scaling recommendations: {e}")
            logger.debug(traceback.format_exc())
            
            # Return basic decision or maintain
            if DRM_AVAILABLE and self.resource_manager:
                return self.resource_manager.evaluate_scaling()
            else:
                return ScalingDecision(
                    action="maintain",
                    reason=f"Error generating scaling recommendations: {str(e)}",
                    utilization=0.0
                )
    
    def record_task_result(self, task_id: str, worker_id: str, result: Dict[str, Any]) -> bool:
        """
        Record task execution result for optimization.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            result: Task execution result
        
        Returns:
            bool: Success status
        """
        try:
            # Record in performance predictor if available
            if PREDICTOR_AVAILABLE and self.performance_predictor:
                # Extract task data
                task_data = result.get("task_data", {})
                metrics = result.get("metrics", {})
                
                # Prepare execution data
                execution_data = {
                    "model_type": task_data.get("model_type", "unknown"),
                    "model_name": task_data.get("model_name", "unknown"),
                    "batch_size": task_data.get("batch_size", 1),
                    "cpu_cores_used": metrics.get("cpu_cores_used", 0),
                    "memory_mb_used": metrics.get("memory_mb_used", 0),
                    "gpu_memory_mb_used": metrics.get("gpu_memory_mb_used", 0),
                    "execution_time_ms": metrics.get("execution_time_ms", 0),
                    "success": result.get("success", True)
                }
                
                # Record in predictor
                self.performance_predictor.record_task_execution(task_id, execution_data)
            
            # Record in resource manager if available
            if DRM_AVAILABLE and self.resource_manager:
                # Prepare execution metrics
                execution_metrics = result.get("metrics", {})
                
                # Record in resource manager
                self.resource_manager.record_task_execution(task_id, execution_metrics)
            
            # Record in workload history
            with self.lock:
                self.workload_history.append({
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "timestamp": time.time(),
                    "result": result
                })
                
                # Limit history size
                if len(self.workload_history) > 1000:
                    self.workload_history = self.workload_history[-1000:]
            
            # Update workload clusters if ML available
            if ML_AVAILABLE and len(self.workload_history) % 20 == 0:
                self._update_workload_clusters()
            
            return True
        
        except Exception as e:
            logger.error(f"Error recording task result: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    # Internal helper methods
    def _default_task_requirements(self, task_data: Dict[str, Any]) -> TaskRequirements:
        """Generate default task requirements."""
        model_type = task_data.get("model_type", "default")
        batch_size = task_data.get("batch_size", 1)
        
        # Map model type to default requirements
        if model_type == "text_generation":
            cpu_cores = 4
            memory_mb = 8192
            gpu_memory_mb = 4096 if batch_size > 1 else 2048
        elif model_type == "text_embedding":
            cpu_cores = 2
            memory_mb = 4096
            gpu_memory_mb = 2048 if batch_size > 8 else 0
        elif model_type == "vision":
            cpu_cores = 2
            memory_mb = 4096
            gpu_memory_mb = 2048
        elif model_type == "audio":
            cpu_cores = 2
            memory_mb = 4096
            gpu_memory_mb = 2048
        elif model_type == "multimodal":
            cpu_cores = 4
            memory_mb = 8192
            gpu_memory_mb = 4096
        else:
            # Default
            cpu_cores = 2
            memory_mb = 4096
            gpu_memory_mb = 0
        
        # Scale with batch size
        batch_factor = max(1, batch_size) / 8  # Normalized to batch size 8
        
        # Apply scaling factors
        cpu_cores = max(1, int(round(cpu_cores * (1 + 0.3 * batch_factor))))
        memory_mb = max(512, int(memory_mb * (1 + 0.7 * batch_factor)))
        gpu_memory_mb = max(0, int(gpu_memory_mb * (1 + 0.9 * batch_factor)))
        
        return TaskRequirements(
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            model_type=model_type,
            model_name=task_data.get("model_name", "unknown"),
            batch_size=batch_size
        )
    
    def _determine_worker_type(self, cpu_cores: float, memory_mb: float, 
                              gpu_memory_mb: float) -> Tuple[str, str]:
        """Determine optimal worker type based on resource requirements."""
        # Define worker type thresholds
        GPU_THRESHOLD = 1024  # 1 GB
        HIGH_MEMORY_THRESHOLD = 32768  # 32 GB
        HIGH_CPU_THRESHOLD = 8  # 8 cores
        
        # Calculate resource ratios
        memory_cpu_ratio = memory_mb / max(1, cpu_cores)
        gpu_cpu_ratio = gpu_memory_mb / max(1, cpu_cores)
        
        # Determine worker type based on requirements and ratios
        if gpu_memory_mb > GPU_THRESHOLD:
            return "gpu", "Task requires GPU acceleration"
        elif memory_mb > HIGH_MEMORY_THRESHOLD or memory_cpu_ratio > 8192:
            return "memory", "Task is memory intensive"
        elif cpu_cores > HIGH_CPU_THRESHOLD:
            return "cpu", "Task is CPU intensive"
        else:
            return "default", "Task has standard resource requirements"
    
    def _calculate_worker_capacity(self, worker_type: str, cpu_cores: float, 
                                  memory_mb: float, gpu_memory_mb: float) -> int:
        """Calculate how many tasks a worker can handle."""
        # Define worker capacity based on type
        if worker_type == "gpu":
            # GPU worker
            worker_cpu = 16
            worker_memory = 32 * 1024  # 32 GB
            worker_gpu = 16 * 1024  # 16 GB
            
            # Capacity limited by the most constrained resource
            cpu_capacity = worker_cpu / max(1, cpu_cores)
            memory_capacity = worker_memory / max(1, memory_mb)
            gpu_capacity = worker_gpu / max(1, gpu_memory_mb) if gpu_memory_mb > 0 else float('inf')
            
            # Return minimum capacity (most constrained resource)
            return max(1, int(min(cpu_capacity, memory_capacity, gpu_capacity)))
        
        elif worker_type == "memory":
            # Memory optimized worker
            worker_cpu = 8
            worker_memory = 64 * 1024  # 64 GB
            
            # Capacity calculation
            cpu_capacity = worker_cpu / max(1, cpu_cores)
            memory_capacity = worker_memory / max(1, memory_mb)
            
            return max(1, int(min(cpu_capacity, memory_capacity)))
        
        elif worker_type == "cpu":
            # CPU optimized worker
            worker_cpu = 32
            worker_memory = 16 * 1024  # 16 GB
            
            # Capacity calculation
            cpu_capacity = worker_cpu / max(1, cpu_cores)
            memory_capacity = worker_memory / max(1, memory_mb)
            
            return max(1, int(min(cpu_capacity, memory_capacity)))
        
        else:
            # Default worker
            worker_cpu = 4
            worker_memory = 8 * 1024  # 8 GB
            
            # Capacity calculation
            cpu_capacity = worker_cpu / max(1, cpu_cores)
            memory_capacity = worker_memory / max(1, memory_mb)
            
            return max(1, int(min(cpu_capacity, memory_capacity)))
    
    def _get_preferred_provider(self, cpu_cores: float, memory_mb: float, 
                               gpu_memory_mb: float) -> Optional[str]:
        """Determine preferred cloud provider for given requirements."""
        if not CLOUD_MANAGER_AVAILABLE or not self.cloud_manager:
            return None
        
        # Determine requirements for provider selection
        requirements = {
            "gpu": gpu_memory_mb > 1024,
            "min_cpu_cores": int(cpu_cores),
            "min_memory_gb": memory_mb / 1024
        }
        
        # Get preferred provider
        return self.cloud_manager.get_preferred_provider(requirements)
    
    def _update_workload_clusters(self) -> None:
        """Update workload clusters using ML."""
        if not ML_AVAILABLE or len(self.workload_history) < 20:
            return
        
        try:
            # Extract features from workload history
            features = []
            for entry in self.workload_history:
                result = entry.get("result", {})
                task_data = result.get("task_data", {})
                metrics = result.get("metrics", {})
                
                # Extract numerical features
                feature_vec = [
                    metrics.get("cpu_cores_used", 0),
                    metrics.get("memory_mb_used", 0) / 1024,  # Normalize to GB
                    metrics.get("gpu_memory_mb_used", 0) / 1024,  # Normalize to GB
                    metrics.get("execution_time_ms", 0) / 1000,  # Normalize to seconds
                    task_data.get("batch_size", 1)
                ]
                
                features.append(feature_vec)
            
            if not features:
                return
            
            # Convert to numpy array
            X = np.array(features)
            
            # Determine number of clusters (max 5 clusters)
            n_clusters = min(5, len(X) // 5)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Store clusters
            self.workload_clusters = {
                "centroids": kmeans.cluster_centers_.tolist(),
                "labels": clusters.tolist(),
                "features": X.tolist(),
                "timestamp": time.time()
            }
            
            logger.debug(f"Updated workload clusters with {n_clusters} clusters")
        
        except Exception as e:
            logger.error(f"Error updating workload clusters: {e}")
            logger.debug(traceback.format_exc())
    
    def _score_workers_for_scaling(self, pending_tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Score workers for scaling decisions."""
        if not self.resource_manager or not self.resource_manager.worker_resources:
            return {}
        
        # Predict task requirements
        task_requirements = []
        for task in pending_tasks:
            requirements = self.predict_task_requirements({
                "model_type": task.get("config", {}).get("model_type", "default"),
                "model_name": task.get("config", {}).get("model", "unknown"),
                "batch_size": task.get("config", {}).get("batch_size", 1)
            })
            task_requirements.append(requirements)
        
        # Calculate average requirements
        if not task_requirements:
            # No tasks, score based on utilization
            worker_scores = {}
            for worker_id, worker_data in self.resource_manager.worker_resources.items():
                utilization = worker_data.get("utilization", {}).get("overall", 0.0)
                worker_scores[worker_id] = 1.0 - utilization  # Lower utilization = higher score (more likely to be removed)
            return worker_scores
        
        # Calculate average requirements
        avg_cpu = sum(r.cpu_cores for r in task_requirements) / len(task_requirements)
        avg_memory = sum(r.memory_mb for r in task_requirements) / len(task_requirements)
        avg_gpu = sum(r.gpu_memory_mb for r in task_requirements) / len(task_requirements)
        
        # Score each worker based on how well it matches requirements
        worker_scores = {}
        
        for worker_id, worker_data in self.resource_manager.worker_resources.items():
            resources = worker_data.get("resources", {})
            
            # Calculate match score (higher is better match)
            worker_score = 0.0
            
            # CPU match
            cpu_avail = resources.get("cpu", {}).get("cores", 0)
            cpu_match = min(cpu_avail / avg_cpu, 2.0) if avg_cpu > 0 else 1.0
            
            # Memory match
            memory_avail = resources.get("memory", {}).get("total_mb", 0)
            memory_match = min(memory_avail / avg_memory, 2.0) if avg_memory > 0 else 1.0
            
            # GPU match
            if avg_gpu > 0:
                gpu_avail = resources.get("gpu", {}).get("memory_mb", 0)
                gpu_match = min(gpu_avail / avg_gpu, 2.0) if gpu_avail > 0 else 0.0
                # GPU is important for GPU tasks
                worker_score = (cpu_match * 0.2) + (memory_match * 0.2) + (gpu_match * 0.6)
            else:
                # No GPU needed
                worker_score = (cpu_match * 0.5) + (memory_match * 0.5)
            
            # Consider utilization (prefer removing lower-utilized workers)
            utilization = worker_data.get("utilization", {}).get("overall", 0.0)
            worker_score = worker_score * (0.5 + 0.5 * utilization)
            
            # Consider active tasks (prefer keeping workers with tasks)
            task_count = len(self.resource_manager.worker_tasks.get(worker_id, set()))
            if task_count > 0:
                worker_score += 10.0  # Strong bonus for workers with active tasks
            
            worker_scores[worker_id] = worker_score
        
        return worker_scores


# Main function for testing
if __name__ == "__main__":
    """Run standalone test of the Resource Optimizer."""
    # Import DRM components
    from duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager
    from duckdb_api.distributed_testing.resource_performance_predictor import ResourcePerformancePredictor
    
    # Create components
    resource_manager = DynamicResourceManager()
    performance_predictor = ResourcePerformancePredictor()
    
    # Create optimizer
    optimizer = ResourceOptimizer(
        resource_manager=resource_manager,
        performance_predictor=performance_predictor
    )
    
    # Register mock workers
    resource_manager.register_worker(
        "worker-1", 
        {
            "cpu": {"cores": 8, "available_cores": 8},
            "memory": {"total_mb": 16384, "available_mb": 16384},
            "gpu": {"devices": 1, "memory_mb": 8192, "available_memory_mb": 8192}
        }
    )
    
    resource_manager.register_worker(
        "worker-2", 
        {
            "cpu": {"cores": 16, "available_cores": 16},
            "memory": {"total_mb": 32768, "available_mb": 32768}
        }
    )
    
    # Create mock tasks
    tasks = [
        {
            "task_id": "task-1",
            "type": "benchmark",
            "priority": 1,
            "config": {
                "model_type": "text_embedding",
                "model": "bert-base-uncased",
                "batch_size": 32
            }
        },
        {
            "task_id": "task-2",
            "type": "benchmark",
            "priority": 2,
            "config": {
                "model_type": "text_generation",
                "model": "llama-7b",
                "batch_size": 1
            }
        },
        {
            "task_id": "task-3",
            "type": "benchmark",
            "priority": 3,
            "config": {
                "model_type": "vision",
                "model": "vit-base-patch16-224",
                "batch_size": 16
            }
        }
    ]
    
    # Predict task requirements
    print("Task requirements:")
    for task in tasks:
        requirements = optimizer.predict_task_requirements({
            "model_type": task["config"]["model_type"],
            "model_name": task["config"]["model"],
            "batch_size": task["config"]["batch_size"]
        })
        print(f"  {task['task_id']} ({task['config']['model_type']}): "
              f"CPU={requirements.cpu_cores}, Memory={requirements.memory_mb}MB, "
              f"GPU={requirements.gpu_memory_mb}MB")
    
    # Allocate resources
    print("\nResource allocation:")
    allocations = optimizer.allocate_resources(tasks, ["worker-1", "worker-2"])
    for allocation in allocations:
        print(f"  {allocation.task_id}: {'Success' if allocation.allocated else 'Failed'} - "
              f"Worker: {allocation.worker_id}, Reason: {allocation.reason}")
    
    # Get worker type recommendations
    print("\nWorker type recommendations:")
    recommendations = optimizer.recommend_worker_types(tasks)
    for recommendation in recommendations:
        print(f"  {recommendation.recommended_type}: {recommendation.reason}, "
              f"Tasks: {recommendation.estimated_task_count}, "
              f"Utilization: {recommendation.estimated_utilization:.1%}")
    
    # Get scaling recommendations
    print("\nScaling recommendations:")
    scaling = optimizer.get_scaling_recommendations()
    print(f"  Action: {scaling.action}, Reason: {scaling.reason}")
    
    # Cleanup
    optimizer.cleanup()
    resource_manager.cleanup()
    performance_predictor.cleanup()