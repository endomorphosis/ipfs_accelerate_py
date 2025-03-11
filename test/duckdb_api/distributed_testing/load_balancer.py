#!/usr/bin/env python3
"""
Distributed Testing Framework - Adaptive Load Balancer

This module implements the adaptive load balancing system for the distributed
testing framework. It's responsible for:

- Dynamic worker capability reassessment
- Real-time performance monitoring
- Workload redistribution based on performance
- Automatic task migration between workers
- Optimal resource utilization
- Handling heterogeneous hardware environments

Usage:
    This module is used by the coordinator server to optimize task distribution
    and balance the workload across available worker nodes.
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
logger = logging.getLogger("load_balancer")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load balancing strategy constants
STRATEGY_ROUND_ROBIN = "round_robin"
STRATEGY_LEAST_LOADED = "least_loaded"
STRATEGY_WEIGHTED = "weighted"
STRATEGY_PERFORMANCE_AWARE = "performance_aware"
STRATEGY_RESOURCE_AWARE = "resource_aware"
STRATEGY_ADAPTIVE = "adaptive"

# Worker ranking criteria
RANK_CPU_USAGE = "cpu_usage"          # Lower is better
RANK_MEMORY_AVAILABLE = "memory_available"  # Higher is better
RANK_GPU_MEMORY = "gpu_memory"        # Higher is better
RANK_TASK_SUCCESS_RATE = "success_rate"  # Higher is better
RANK_EXECUTION_SPEED = "execution_speed"  # Higher is better
RANK_CONSISTENT_PERFORMANCE = "consistency"  # Higher is better

# Default weights for ranking criteria
DEFAULT_RANKING_WEIGHTS = {
    RANK_CPU_USAGE: 0.15,
    RANK_MEMORY_AVAILABLE: 0.2,
    RANK_GPU_MEMORY: 0.2,
    RANK_TASK_SUCCESS_RATE: 0.25,
    RANK_EXECUTION_SPEED: 0.15,
    RANK_CONSISTENT_PERFORMANCE: 0.05
}

class LoadBalancer:
    """Adaptive load balancer for the distributed testing framework."""
    
    def __init__(self, worker_manager=None, task_scheduler=None, db_manager=None):
        """Initialize the load balancer.
        
        Args:
            worker_manager: Worker manager instance
            task_scheduler: Task scheduler instance
            db_manager: Database manager instance
        """
        self.worker_manager = worker_manager
        self.task_scheduler = task_scheduler
        self.db_manager = db_manager
        
        # Load balancing metrics
        self.worker_metrics = {}  # worker_id -> metrics
        self.worker_load_history = {}  # worker_id -> list of historical load metrics
        self.worker_rankings = {}  # worker_id -> ranking score (0-100, higher is better)
        self.worker_capacity = {}  # worker_id -> estimated task capacity
        
        # Task migration tracking
        self.migration_history = {}  # task_id -> list of worker_ids
        self.worker_task_counts = {}  # worker_id -> current task count
        
        # Worker preference for task types (specialized workers)
        self.worker_type_preferences = {}  # worker_id -> {task_type: preference_score}
        
        # Configuration
        self.config = {
            "strategy": STRATEGY_ADAPTIVE,  # Default strategy
            "ranking_weights": DEFAULT_RANKING_WEIGHTS.copy(),
            "load_history_size": 10,  # How many historical data points to keep
            "ranking_update_interval": 60,  # Seconds between ranking updates
            "rebalance_threshold": 0.25,  # Load imbalance threshold to trigger rebalancing
            "task_migration_enabled": True,  # Enable task migration between workers
            "adaptive_weights_enabled": True,  # Dynamically adjust ranking weights
            "specialized_workers_enabled": True,  # Enable worker specialization
            "max_migrations_per_task": 2,  # Maximum number of migrations for a single task
            "performance_history_weight": 0.7,  # Weight for historical performance vs current
            "resource_monitoring_interval": 30,  # Seconds between resource monitoring updates
        }
        
        # Internal state
        self.last_ranking_update = datetime.now()
        self.last_rebalance_check = datetime.now()
        self.strategy_metrics = {
            "migrations_performed": 0,
            "rebalances_triggered": 0,
            "total_task_assignments": 0,
            "optimal_assignments": 0,
        }
        
        # Start monitoring thread
        self.monitoring_thread = None
        self.monitoring_stop_event = threading.Event()
        
        logger.info("Load balancer initialized")
    
    def configure(self, config_updates: Dict[str, Any]):
        """Update the load balancer configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.config.update(config_updates)
        logger.info(f"Load balancer configuration updated: {config_updates}")
    
    def start_monitoring(self):
        """Start the resource monitoring thread."""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return
            
        self.monitoring_stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Load balancer monitoring thread started")
    
    def stop_monitoring(self):
        """Stop the resource monitoring thread."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread not running")
            return
            
        self.monitoring_stop_event.set()
        self.monitoring_thread.join(timeout=5.0)
        if self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread did not stop gracefully")
            
        logger.info("Load balancer monitoring thread stopped")
    
    def _monitoring_loop(self):
        """Resource monitoring thread function."""
        while not self.monitoring_stop_event.is_set():
            try:
                # Update worker rankings
                if (datetime.now() - self.last_ranking_update).total_seconds() >= self.config["ranking_update_interval"]:
                    self._update_worker_rankings()
                    self.last_ranking_update = datetime.now()
                    
                # Check for rebalancing opportunity
                if (datetime.now() - self.last_rebalance_check).total_seconds() >= self.config["ranking_update_interval"]:
                    self._check_for_rebalancing()
                    self.last_rebalance_check = datetime.now()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            # Wait for next interval
            self.monitoring_stop_event.wait(self.config["resource_monitoring_interval"])
    
    def select_worker_for_task(self, task: Dict[str, Any], 
                              available_workers: List[Dict[str, Any]]) -> Optional[str]:
        """Select the best worker for a task based on the current strategy.
        
        Args:
            task: Task configuration
            available_workers: List of available worker information dicts
            
        Returns:
            ID of the selected worker, or None if no suitable worker found
        """
        if not available_workers:
            return None
            
        # Update load counts
        self._update_worker_task_counts()
            
        # Choose strategy
        strategy = self.config["strategy"]
        
        # Apply strategy
        if strategy == STRATEGY_ROUND_ROBIN:
            return self._select_round_robin(task, available_workers)
        elif strategy == STRATEGY_LEAST_LOADED:
            return self._select_least_loaded(task, available_workers)
        elif strategy == STRATEGY_WEIGHTED:
            return self._select_weighted(task, available_workers)
        elif strategy == STRATEGY_PERFORMANCE_AWARE:
            return self._select_performance_aware(task, available_workers)
        elif strategy == STRATEGY_RESOURCE_AWARE:
            return self._select_resource_aware(task, available_workers)
        elif strategy == STRATEGY_ADAPTIVE:
            return self._select_adaptive(task, available_workers)
        else:
            # Default to adaptive
            return self._select_adaptive(task, available_workers)
    
    def _select_round_robin(self, task: Dict[str, Any], 
                          available_workers: List[Dict[str, Any]]) -> Optional[str]:
        """Select a worker using round-robin strategy.
        
        Args:
            task: Task configuration
            available_workers: List of available worker information dicts
            
        Returns:
            ID of the selected worker, or None if no suitable worker found
        """
        # Filter workers with matching requirements
        matching_workers = []
        for worker in available_workers:
            # Skip workers that don't meet requirements
            if not self._worker_meets_requirements(worker, task.get("requirements", {})):
                continue
            matching_workers.append(worker)
            
        if not matching_workers:
            return None
            
        # Sort by current task count (ascending)
        matching_workers.sort(key=lambda w: self.worker_task_counts.get(w["worker_id"], 0))
        
        # Return the worker with the lowest task count
        return matching_workers[0]["worker_id"]
    
    def _select_least_loaded(self, task: Dict[str, Any], 
                           available_workers: List[Dict[str, Any]]) -> Optional[str]:
        """Select the least loaded worker for a task.
        
        Args:
            task: Task configuration
            available_workers: List of available worker information dicts
            
        Returns:
            ID of the selected worker, or None if no suitable worker found
        """
        # Filter workers with matching requirements
        matching_workers = []
        for worker in available_workers:
            # Skip workers that don't meet requirements
            if not self._worker_meets_requirements(worker, task.get("requirements", {})):
                continue
            matching_workers.append(worker)
            
        if not matching_workers:
            return None
            
        # Calculate load scores
        worker_loads = []
        for worker in matching_workers:
            worker_id = worker["worker_id"]
            
            # Current task count (normalized by capacity)
            task_count = self.worker_task_counts.get(worker_id, 0)
            capacity = self.worker_capacity.get(worker_id, 1)
            load_score = task_count / capacity if capacity > 0 else task_count
            
            # Add CPU usage if available
            if worker_id in self.worker_metrics and "cpu_percent" in self.worker_metrics[worker_id]:
                cpu_percent = self.worker_metrics[worker_id]["cpu_percent"]
                # Blend with task-based load (70% task count, 30% CPU)
                load_score = 0.7 * load_score + 0.3 * (cpu_percent / 100.0)
            
            worker_loads.append((worker_id, load_score))
            
        # Sort by load score (ascending)
        worker_loads.sort(key=lambda w: w[1])
        
        # Return the worker with the lowest load
        return worker_loads[0][0]
    
    def _select_weighted(self, task: Dict[str, Any], 
                        available_workers: List[Dict[str, Any]]) -> Optional[str]:
        """Select a worker using weighted scoring based on multiple factors.
        
        Args:
            task: Task configuration
            available_workers: List of available worker information dicts
            
        Returns:
            ID of the selected worker, or None if no suitable worker found
        """
        # Filter workers with matching requirements
        matching_workers = []
        for worker in available_workers:
            # Skip workers that don't meet requirements
            if not self._worker_meets_requirements(worker, task.get("requirements", {})):
                continue
            matching_workers.append(worker)
            
        if not matching_workers:
            return None
            
        # Calculate weighted scores
        worker_scores = []
        for worker in matching_workers:
            worker_id = worker["worker_id"]
            
            # Start with ranking score (if available) or default to 50
            base_score = self.worker_rankings.get(worker_id, 50.0)
            
            # Apply load factor (inversely proportional to load)
            task_count = self.worker_task_counts.get(worker_id, 0)
            capacity = self.worker_capacity.get(worker_id, 1)
            load_factor = 1.0 - (task_count / capacity if capacity > 0 else task_count)
            load_factor = max(0.1, min(1.0, load_factor))  # Clamp between 0.1 and 1.0
            
            # Task type preference factor
            task_type = task.get("type", "unknown")
            type_factor = 1.0
            if worker_id in self.worker_type_preferences and task_type in self.worker_type_preferences[worker_id]:
                type_preference = self.worker_type_preferences[worker_id][task_type]
                type_factor = 1.0 + (type_preference * 0.5)  # Up to 50% bonus for preferred task types
            
            # Combine factors
            final_score = base_score * load_factor * type_factor
            
            worker_scores.append((worker_id, final_score))
            
        # Sort by final score (descending)
        worker_scores.sort(key=lambda w: w[1], reverse=True)
        
        # Return the worker with the highest score
        return worker_scores[0][0]
    
    def _select_performance_aware(self, task: Dict[str, Any], 
                                available_workers: List[Dict[str, Any]]) -> Optional[str]:
        """Select a worker based on past performance for similar tasks.
        
        Args:
            task: Task configuration
            available_workers: List of available worker information dicts
            
        Returns:
            ID of the selected worker, or None if no suitable worker found
        """
        # Filter workers with matching requirements
        matching_workers = []
        for worker in available_workers:
            # Skip workers that don't meet requirements
            if not self._worker_meets_requirements(worker, task.get("requirements", {})):
                continue
            matching_workers.append(worker)
            
        if not matching_workers:
            return None
            
        task_type = task.get("type", "unknown")
        
        # Check which workers have performance data for this task type
        workers_with_perf_data = []
        workers_without_perf_data = []
        
        for worker in matching_workers:
            worker_id = worker["worker_id"]
            
            if worker_id in self.worker_metrics and "task_types" in self.worker_metrics[worker_id]:
                task_types = self.worker_metrics[worker_id]["task_types"]
                if task_type in task_types and task_types[task_type].get("task_count", 0) > 0:
                    # Has performance data for this task type
                    success_rate = task_types[task_type].get("success_rate", 0.5)
                    avg_time = task_types[task_type].get("avg_execution_time", 0)
                    
                    # Apply load factor
                    task_count = self.worker_task_counts.get(worker_id, 0)
                    capacity = self.worker_capacity.get(worker_id, 1)
                    load_factor = 1.0 - (task_count / capacity if capacity > 0 else task_count)
                    load_factor = max(0.1, min(1.0, load_factor))  # Clamp between 0.1 and 1.0
                    
                    # Calculate performance score (higher is better)
                    # Weight success rate more heavily than execution time
                    # Success rate range: 0-1 (higher is better)
                    # For execution time, we want lower values, so invert the factor
                    execution_factor = 1.0
                    if avg_time > 0:
                        # Normalize against other workers (if we have data)
                        all_times = []
                        for w_id in self.worker_metrics:
                            if w_id == worker_id:
                                continue
                            if "task_types" in self.worker_metrics[w_id] and task_type in self.worker_metrics[w_id]["task_types"]:
                                w_time = self.worker_metrics[w_id]["task_types"][task_type].get("avg_execution_time", 0)
                                if w_time > 0:
                                    all_times.append(w_time)
                        
                        if all_times:
                            avg_other_time = sum(all_times) / len(all_times)
                            if avg_other_time > 0:
                                execution_factor = avg_other_time / avg_time  # Higher if this worker is faster
                                execution_factor = min(2.0, execution_factor)  # Cap at 2.0
                    
                    performance_score = (0.7 * success_rate + 0.3 * execution_factor) * load_factor
                    
                    workers_with_perf_data.append((worker_id, performance_score))
                else:
                    # No performance data for this task type
                    workers_without_perf_data.append(worker_id)
            else:
                # No performance data at all
                workers_without_perf_data.append(worker_id)
        
        # If we have workers with performance data, select the best one
        if workers_with_perf_data:
            workers_with_perf_data.sort(key=lambda w: w[1], reverse=True)
            return workers_with_perf_data[0][0]
        
        # Otherwise, use round-robin for workers without data
        if workers_without_perf_data:
            # Sort by task count
            workers_without_perf_data.sort(key=lambda w_id: self.worker_task_counts.get(w_id, 0))
            return workers_without_perf_data[0]
        
        # Shouldn't reach here
        return None
    
    def _select_resource_aware(self, task: Dict[str, Any], 
                             available_workers: List[Dict[str, Any]]) -> Optional[str]:
        """Select a worker based on resource availability.
        
        Args:
            task: Task configuration
            available_workers: List of available worker information dicts
            
        Returns:
            ID of the selected worker, or None if no suitable worker found
        """
        # Filter workers with matching requirements
        matching_workers = []
        for worker in available_workers:
            # Skip workers that don't meet requirements
            if not self._worker_meets_requirements(worker, task.get("requirements", {})):
                continue
            matching_workers.append(worker)
            
        if not matching_workers:
            return None
            
        # Calculate resource scores
        worker_scores = []
        for worker in matching_workers:
            worker_id = worker["worker_id"]
            
            resource_score = 0.0
            score_count = 0
            
            # Memory score (higher is better)
            if worker_id in self.worker_metrics and "memory_available_gb" in self.worker_metrics[worker_id]:
                memory_gb = self.worker_metrics[worker_id]["memory_available_gb"]
                min_memory = task.get("requirements", {}).get("min_memory_gb", 0)
                
                if memory_gb >= min_memory:
                    # Calculate how much headroom we have (but not too much)
                    mem_ratio = min_memory / memory_gb if memory_gb > 0 else 0
                    
                    # Ideal ratio is 0.7-0.8
                    if mem_ratio <= 0.8:
                        mem_score = min(1.0, mem_ratio / 0.8)
                    else:
                        mem_score = max(0.5, 1.0 - (mem_ratio - 0.8) * 5)
                        
                    resource_score += mem_score
                    score_count += 1
            
            # CPU score (lower usage is better)
            if worker_id in self.worker_metrics and "cpu_percent" in self.worker_metrics[worker_id]:
                cpu_percent = self.worker_metrics[worker_id]["cpu_percent"]
                
                # CPU score is inverse of usage (100% -> 0, 0% -> 1)
                cpu_score = 1.0 - (cpu_percent / 100.0)
                resource_score += cpu_score
                score_count += 1
            
            # GPU memory score (if task requires GPU)
            if "hardware" in task.get("requirements", {}) and "cuda" in task.get("requirements", {}).get("hardware", []):
                if worker_id in self.worker_metrics and "gpu_memory_available_mb" in self.worker_metrics[worker_id]:
                    gpu_mem_mb = self.worker_metrics[worker_id]["gpu_memory_available_mb"]
                    min_gpu_mem = task.get("requirements", {}).get("min_gpu_memory_mb", 0)
                    
                    if gpu_mem_mb >= min_gpu_mem:
                        # Similar to memory ratio
                        gpu_ratio = min_gpu_mem / gpu_mem_mb if gpu_mem_mb > 0 else 0
                        
                        if gpu_ratio <= 0.8:
                            gpu_score = min(1.0, gpu_ratio / 0.8)
                        else:
                            gpu_score = max(0.5, 1.0 - (gpu_ratio - 0.8) * 5)
                            
                        resource_score += gpu_score
                        score_count += 1
            
            # Calculate average score if we have any
            if score_count > 0:
                resource_score = resource_score / score_count
            else:
                # No resource data, use default score
                resource_score = 0.5
                
            # Apply task count factor
            task_count = self.worker_task_counts.get(worker_id, 0)
            capacity = self.worker_capacity.get(worker_id, 1)
            load_factor = 1.0 - (task_count / capacity if capacity > 0 else task_count)
            load_factor = max(0.1, min(1.0, load_factor))  # Clamp between 0.1 and 1.0
            
            # Final score
            final_score = resource_score * load_factor
            
            worker_scores.append((worker_id, final_score))
            
        # Sort by final score (descending)
        worker_scores.sort(key=lambda w: w[1], reverse=True)
        
        # Return the worker with the highest score
        return worker_scores[0][0]
    
    def _select_adaptive(self, task: Dict[str, Any], 
                       available_workers: List[Dict[str, Any]]) -> Optional[str]:
        """Select a worker using an adaptive strategy that combines multiple approaches.
        
        This dynamically chooses the best strategy based on context.
        
        Args:
            task: Task configuration
            available_workers: List of available worker information dicts
            
        Returns:
            ID of the selected worker, or None if no suitable worker found
        """
        # Filter workers with matching requirements
        matching_workers = []
        for worker in available_workers:
            # Skip workers that don't meet requirements
            if not self._worker_meets_requirements(worker, task.get("requirements", {})):
                continue
            matching_workers.append(worker)
            
        if not matching_workers:
            return None
        
        # Get scores from different strategies
        task_type = task.get("type", "unknown")
        
        # Strategy weights will be adjusted based on context
        strategy_weights = {
            "performance": 0.4,
            "resource": 0.3,
            "load": 0.2,
            "type_preference": 0.1
        }
        
        # Adjust weights based on task type
        if task_type == "benchmark":
            # For benchmarks, resource availability is more important
            strategy_weights["resource"] = 0.5
            strategy_weights["performance"] = 0.2
            strategy_weights["load"] = 0.2
            strategy_weights["type_preference"] = 0.1
        elif task_type == "test":
            # For tests, performance history is most important
            strategy_weights["performance"] = 0.5
            strategy_weights["resource"] = 0.2
            strategy_weights["load"] = 0.2
            strategy_weights["type_preference"] = 0.1
        
        # Calculate scores for each worker
        worker_scores = {}
        for worker in matching_workers:
            worker_id = worker["worker_id"]
            
            # Initialize scores
            worker_scores[worker_id] = {
                "performance": 0.5,  # Default performance score
                "resource": 0.5,     # Default resource score
                "load": 0.5,         # Default load score
                "type_preference": 0.5  # Default type preference score
            }
            
            # Performance score
            if worker_id in self.worker_metrics and "task_types" in self.worker_metrics[worker_id]:
                task_types = self.worker_metrics[worker_id]["task_types"]
                if task_type in task_types and task_types[task_type].get("task_count", 0) > 0:
                    success_rate = task_types[task_type].get("success_rate", 0.5)
                    avg_time = task_types[task_type].get("avg_execution_time", 0)
                    
                    # Calculate execution factor similar to performance-aware strategy
                    execution_factor = 1.0
                    if avg_time > 0:
                        all_times = []
                        for w_id in self.worker_metrics:
                            if w_id == worker_id:
                                continue
                            if "task_types" in self.worker_metrics[w_id] and task_type in self.worker_metrics[w_id]["task_types"]:
                                w_time = self.worker_metrics[w_id]["task_types"][task_type].get("avg_execution_time", 0)
                                if w_time > 0:
                                    all_times.append(w_time)
                        
                        if all_times:
                            avg_other_time = sum(all_times) / len(all_times)
                            if avg_other_time > 0:
                                execution_factor = avg_other_time / avg_time
                                execution_factor = min(2.0, execution_factor)
                    
                    worker_scores[worker_id]["performance"] = 0.7 * success_rate + 0.3 * execution_factor
            
            # Resource score
            resource_score = 0.0
            score_count = 0
            
            # Memory score
            if worker_id in self.worker_metrics and "memory_available_gb" in self.worker_metrics[worker_id]:
                memory_gb = self.worker_metrics[worker_id]["memory_available_gb"]
                min_memory = task.get("requirements", {}).get("min_memory_gb", 0)
                
                if memory_gb >= min_memory:
                    mem_ratio = min_memory / memory_gb if memory_gb > 0 else 0
                    
                    if mem_ratio <= 0.8:
                        mem_score = min(1.0, mem_ratio / 0.8)
                    else:
                        mem_score = max(0.5, 1.0 - (mem_ratio - 0.8) * 5)
                        
                    resource_score += mem_score
                    score_count += 1
            
            # CPU score
            if worker_id in self.worker_metrics and "cpu_percent" in self.worker_metrics[worker_id]:
                cpu_percent = self.worker_metrics[worker_id]["cpu_percent"]
                cpu_score = 1.0 - (cpu_percent / 100.0)
                resource_score += cpu_score
                score_count += 1
            
            # Calculate average resource score
            if score_count > 0:
                worker_scores[worker_id]["resource"] = resource_score / score_count
            
            # Load score (inverse of task count normalized by capacity)
            task_count = self.worker_task_counts.get(worker_id, 0)
            capacity = self.worker_capacity.get(worker_id, 1)
            load_factor = 1.0 - (task_count / capacity if capacity > 0 else task_count)
            worker_scores[worker_id]["load"] = max(0.1, min(1.0, load_factor))
            
            # Type preference score
            if worker_id in self.worker_type_preferences and task_type in self.worker_type_preferences[worker_id]:
                type_preference = self.worker_type_preferences[worker_id][task_type]
                worker_scores[worker_id]["type_preference"] = min(1.0, 0.5 + type_preference)
        
        # Calculate final scores
        final_scores = {}
        for worker_id, scores in worker_scores.items():
            final_score = sum(score * strategy_weights[key] for key, score in scores.items())
            final_scores[worker_id] = final_score
        
        # Find worker with highest score
        best_worker_id = max(final_scores.items(), key=lambda x: x[1])[0]
        
        # Track assignment quality
        self.strategy_metrics["total_task_assignments"] += 1
        if final_scores[best_worker_id] >= 0.8:
            self.strategy_metrics["optimal_assignments"] += 1
        
        return best_worker_id
    
    def _worker_meets_requirements(self, worker: Dict[str, Any],
                                  task_requirements: Dict[str, Any]) -> bool:
        """Check if a worker meets the requirements for a task.
        
        Args:
            worker: Worker information
            task_requirements: Task hardware requirements
            
        Returns:
            True if worker meets requirements, False otherwise
        """
        # Check hardware requirements
        if "hardware" in task_requirements:
            required_hardware = task_requirements["hardware"]
            if isinstance(required_hardware, list):
                # Check if worker has any of the required hardware
                worker_hardware = worker.get("capabilities", {}).get("hardware_types", [])
                if not any(hw in worker_hardware for hw in required_hardware):
                    return False
            elif isinstance(required_hardware, str):
                # Check if worker has the required hardware
                worker_hardware = worker.get("capabilities", {}).get("hardware_types", [])
                if required_hardware not in worker_hardware:
                    return False
        
        # Check minimum memory
        if "min_memory_gb" in task_requirements:
            min_memory = task_requirements["min_memory_gb"]
            worker_memory = worker.get("capabilities", {}).get("memory_gb", 0)
            if worker_memory < min_memory:
                return False
        
        # Check minimum CUDA compute capability
        if "min_cuda_compute" in task_requirements:
            min_cuda = task_requirements["min_cuda_compute"]
            worker_cuda = worker.get("capabilities", {}).get("cuda_compute", 0)
            if worker_cuda < min_cuda:
                return False
        
        # Check for specific browser requirements
        if "browser" in task_requirements:
            required_browser = task_requirements["browser"]
            available_browsers = worker.get("capabilities", {}).get("browsers", [])
            if required_browser not in available_browsers:
                return False
        
        # Check for specific device requirements (mobile, etc.)
        if "device_type" in task_requirements:
            required_device = task_requirements["device_type"]
            worker_device = worker.get("capabilities", {}).get("device_type")
            if worker_device != required_device:
                return False
        
        return True
    
    def update_worker_metrics(self, worker_id: str, metrics: Dict[str, Any]):
        """Update metrics for a worker.
        
        Args:
            worker_id: ID of the worker
            metrics: Dictionary of metrics
        """
        # Initialize metrics if not exists
        if worker_id not in self.worker_metrics:
            self.worker_metrics[worker_id] = {}
            
        # Update with new metrics
        self.worker_metrics[worker_id].update(metrics)
        
        # Update load history
        if "cpu_percent" in metrics or "memory_available_gb" in metrics:
            if worker_id not in self.worker_load_history:
                self.worker_load_history[worker_id] = []
                
            # Create load data point
            load_data = {
                "timestamp": datetime.now(),
                "cpu_percent": metrics.get("cpu_percent"),
                "memory_available_gb": metrics.get("memory_available_gb"),
                "task_count": self.worker_task_counts.get(worker_id, 0)
            }
            
            # Add to history
            self.worker_load_history[worker_id].append(load_data)
            
            # Limit history size
            max_history = self.config["load_history_size"]
            if len(self.worker_load_history[worker_id]) > max_history:
                self.worker_load_history[worker_id] = self.worker_load_history[worker_id][-max_history:]
        
        # Update worker capacity estimate based on hardware
        self._estimate_worker_capacity(worker_id)
        
        # Clear suitability cache in the task scheduler if it exists
        if self.task_scheduler and hasattr(self.task_scheduler, "reset_cache"):
            self.task_scheduler.reset_cache()
            
        logger.debug(f"Updated metrics for worker {worker_id}")
    
    def _estimate_worker_capacity(self, worker_id: str):
        """Estimate the task capacity of a worker based on its hardware.
        
        Args:
            worker_id: ID of the worker
        """
        if worker_id not in self.worker_metrics:
            self.worker_capacity[worker_id] = 1
            return
            
        # Base capacity on hardware capabilities
        capacity = 1.0
        
        # Adjust based on CPU cores
        if "cpu" in self.worker_metrics[worker_id] and "count" in self.worker_metrics[worker_id]["cpu"]:
            cpu_count = self.worker_metrics[worker_id]["cpu"]["count"]
            # More cores = more capacity, but not linear
            core_factor = min(4.0, 1.0 + (cpu_count - 1) * 0.5)
            capacity *= core_factor
        
        # Adjust based on memory
        if "memory_gb" in self.worker_metrics[worker_id]:
            memory_gb = self.worker_metrics[worker_id]["memory_gb"]
            # More memory = more capacity for parallel tasks
            mem_factor = min(3.0, 1.0 + (memory_gb / 8.0))
            capacity *= mem_factor
        
        # Adjust for GPU (gpu enables higher capacity)
        if "gpu" in self.worker_metrics[worker_id] and self.worker_metrics[worker_id]["gpu"].get("count", 0) > 0:
            capacity *= 1.5
        
        # Cap reasonable limits based on real-world testing
        capacity = max(1, min(8, int(capacity)))
        
        self.worker_capacity[worker_id] = capacity
        logger.debug(f"Estimated capacity for worker {worker_id}: {capacity} tasks")
    
    def _update_worker_rankings(self):
        """Update rankings for all workers based on performance and resource metrics."""
        for worker_id in self.worker_metrics:
            ranking_score = 0.0
            weights = self.config["ranking_weights"]
            total_weight = sum(weights.values())
            
            # CPU usage (lower is better)
            if RANK_CPU_USAGE in weights and "cpu_percent" in self.worker_metrics[worker_id]:
                cpu_percent = self.worker_metrics[worker_id]["cpu_percent"]
                cpu_score = 100.0 - cpu_percent  # Invert (0% usage = 100 score)
                ranking_score += weights[RANK_CPU_USAGE] * cpu_score
            
            # Memory available (higher is better)
            if RANK_MEMORY_AVAILABLE in weights and "memory_available_gb" in self.worker_metrics[worker_id]:
                memory_gb = self.worker_metrics[worker_id]["memory_available_gb"]
                # Normalize to 0-100 range (assume 32GB is max)
                memory_score = min(100.0, memory_gb * (100.0 / 32.0))
                ranking_score += weights[RANK_MEMORY_AVAILABLE] * memory_score
            
            # GPU memory (higher is better)
            if RANK_GPU_MEMORY in weights and "gpu_memory_available_mb" in self.worker_metrics[worker_id]:
                gpu_mem_mb = self.worker_metrics[worker_id]["gpu_memory_available_mb"]
                # Normalize to 0-100 range (assume 24GB is max)
                gpu_score = min(100.0, gpu_mem_mb / 24576.0 * 100.0)
                ranking_score += weights[RANK_GPU_MEMORY] * gpu_score
            
            # Task success rate (higher is better)
            if RANK_TASK_SUCCESS_RATE in weights:
                if "success_count" in self.worker_metrics[worker_id] and "task_count" in self.worker_metrics[worker_id]:
                    success_count = self.worker_metrics[worker_id]["success_count"]
                    task_count = self.worker_metrics[worker_id]["task_count"]
                    if task_count > 0:
                        success_rate = success_count / task_count
                        success_score = success_rate * 100.0
                        ranking_score += weights[RANK_TASK_SUCCESS_RATE] * success_score
            
            # Execution speed (higher is better)
            if RANK_EXECUTION_SPEED in weights:
                # Calculate relative speed compared to other workers
                speed_score = 50.0  # Default to average
                
                # To calculate this properly, we need global stats
                if hasattr(self, "task_stats") and self.task_stats:
                    total_speed_ratio = 0.0
                    count = 0
                    
                    for task_type, stats in self.task_stats.items():
                        if "task_types" in self.worker_metrics[worker_id] and task_type in self.worker_metrics[worker_id]["task_types"]:
                            worker_time = self.worker_metrics[worker_id]["task_types"][task_type].get("avg_execution_time", 0)
                            global_time = stats.get("avg_execution_time", 0)
                            
                            if worker_time > 0 and global_time > 0:
                                speed_ratio = global_time / worker_time  # Higher if worker is faster
                                total_speed_ratio += speed_ratio
                                count += 1
                    
                    if count > 0:
                        avg_speed_ratio = total_speed_ratio / count
                        # Normalize to 0-100 range (1.0 = average = 50, 2.0 = twice as fast = 100)
                        speed_score = min(100.0, avg_speed_ratio * 50.0)
                
                ranking_score += weights[RANK_EXECUTION_SPEED] * speed_score
            
            # Consistency (lower variance is better)
            if RANK_CONSISTENT_PERFORMANCE in weights:
                consistency_score = 50.0  # Default to average
                
                # Calculate variance in execution times
                if "task_types" in self.worker_metrics[worker_id]:
                    total_variance_ratio = 0.0
                    count = 0
                    
                    for task_type, type_metrics in self.worker_metrics[worker_id]["task_types"].items():
                        if "execution_time_variance" in type_metrics and type_metrics["task_count"] > 1:
                            variance = type_metrics["execution_time_variance"]
                            mean = type_metrics["avg_execution_time"]
                            
                            if mean > 0:
                                # Coefficient of variation (lower is better)
                                cv = math.sqrt(variance) / mean
                                # Invert and normalize (low CV = high score)
                                cv_score = max(0.0, 100.0 - (cv * 100.0))
                                
                                total_variance_ratio += cv_score
                                count += 1
                    
                    if count > 0:
                        consistency_score = total_variance_ratio / count
                
                ranking_score += weights[RANK_CONSISTENT_PERFORMANCE] * consistency_score
            
            # Normalize to 0-100 range
            if total_weight > 0:
                ranking_score = ranking_score / total_weight
            
            # Update ranking
            self.worker_rankings[worker_id] = ranking_score
            
        logger.debug(f"Updated rankings for {len(self.worker_rankings)} workers")
    
    def _update_worker_task_counts(self):
        """Update task counts for all workers."""
        # Initialize counts
        self.worker_task_counts = {}
        
        # Count running tasks per worker
        for task_id, worker_id in self.task_scheduler.running_tasks.items():
            if worker_id not in self.worker_task_counts:
                self.worker_task_counts[worker_id] = 0
            self.worker_task_counts[worker_id] += 1
    
    def _check_for_rebalancing(self):
        """Check if task rebalancing is needed and perform it if necessary."""
        if not self.config["task_migration_enabled"]:
            return
            
        # Get current task counts
        self._update_worker_task_counts()
        
        # Check if we have any workers with tasks
        if not self.worker_task_counts:
            return
            
        # Calculate average tasks per worker
        avg_tasks = sum(self.worker_task_counts.values()) / len(self.worker_task_counts)
        
        # Find overloaded and underloaded workers
        overloaded = []
        underloaded = []
        
        for worker_id, count in self.worker_task_counts.items():
            capacity = self.worker_capacity.get(worker_id, 1)
            
            # Normalize by capacity
            load_ratio = count / capacity if capacity > 0 else count
            avg_load_ratio = avg_tasks / capacity if capacity > 0 else avg_tasks
            
            # Check if significantly overloaded or underloaded
            threshold = self.config["rebalance_threshold"]
            
            if load_ratio > avg_load_ratio * (1 + threshold):
                overloaded.append((worker_id, count, capacity, load_ratio))
            elif load_ratio < avg_load_ratio * (1 - threshold) and count < capacity:
                underloaded.append((worker_id, count, capacity, load_ratio))
        
        # Sort overloaded (most overloaded first) and underloaded (least loaded first)
        overloaded.sort(key=lambda w: w[3], reverse=True)
        underloaded.sort(key=lambda w: w[3])
        
        # Check if we have both overloaded and underloaded workers
        if not overloaded or not underloaded:
            return
            
        logger.info(f"Detected load imbalance: {len(overloaded)} overloaded, {len(underloaded)} underloaded workers")
        
        # Find tasks that can be migrated
        migrations = []
        
        for over_worker_id, _, _, _ in overloaded:
            # Find tasks assigned to this worker
            for task_id, worker_id in list(self.task_scheduler.running_tasks.items()):
                if worker_id != over_worker_id:
                    continue
                    
                # Check if this task can be migrated
                if task_id in self.migration_history and len(self.migration_history[task_id]) >= self.config["max_migrations_per_task"]:
                    # Already migrated too many times
                    continue
                
                # Get task details
                task = None
                if self.db_manager:
                    task = self.db_manager.get_task(task_id)
                
                if not task:
                    continue
                    
                # Check for suitable destination workers
                for under_worker_id, _, _, _ in underloaded:
                    # Skip self-migration
                    if under_worker_id == over_worker_id:
                        continue
                        
                    # Get worker details
                    under_worker = None
                    if self.worker_manager:
                        under_worker = self.worker_manager.get_worker(under_worker_id)
                    
                    if not under_worker:
                        continue
                        
                    # Check if worker meets requirements
                    if self._worker_meets_requirements(under_worker, task.get("requirements", {})):
                        # Found a suitable migration target
                        migrations.append((task_id, over_worker_id, under_worker_id))
                        break
        
        # Perform migrations
        if migrations:
            logger.info(f"Performing {len(migrations)} task migrations for load balancing")
            self.strategy_metrics["rebalances_triggered"] += 1
            
            for task_id, source_worker_id, dest_worker_id in migrations:
                self._migrate_task(task_id, source_worker_id, dest_worker_id)
    
    def _migrate_task(self, task_id: str, source_worker_id: str, dest_worker_id: str) -> bool:
        """Migrate a task from one worker to another.
        
        Args:
            task_id: ID of the task to migrate
            source_worker_id: ID of the source worker
            dest_worker_id: ID of the destination worker
            
        Returns:
            True if migration was successful, False otherwise
        """
        # Check if task exists and is assigned to source worker
        if task_id not in self.task_scheduler.running_tasks:
            logger.warning(f"Task {task_id} not found in running tasks")
            return False
            
        if self.task_scheduler.running_tasks[task_id] != source_worker_id:
            logger.warning(f"Task {task_id} is not assigned to worker {source_worker_id}")
            return False
        
        # Get task details
        task = None
        if self.db_manager:
            task = self.db_manager.get_task(task_id)
        
        if not task:
            logger.warning(f"Task {task_id} details not found in database")
            return False
        
        # Check migration history
        if task_id not in self.migration_history:
            self.migration_history[task_id] = []
        
        if len(self.migration_history[task_id]) >= self.config["max_migrations_per_task"]:
            logger.warning(f"Task {task_id} has already been migrated {len(self.migration_history[task_id])} times (max: {self.config['max_migrations_per_task']})")
            return False
        
        # Log migration
        logger.info(f"Migrating task {task_id} from worker {source_worker_id} to {dest_worker_id}")
        
        # Update running tasks dictionary
        self.task_scheduler.running_tasks[task_id] = dest_worker_id
        
        # Update database if available
        if self.db_manager:
            self.db_manager.update_task_worker(task_id, dest_worker_id)
        
        # Update migration history
        self.migration_history[task_id].append((datetime.now(), source_worker_id, dest_worker_id))
        
        # Update task counts
        self._update_worker_task_counts()
        
        # Update metrics
        self.strategy_metrics["migrations_performed"] += 1
        
        return True
    
    def update_type_preferences(self, worker_id: str, task_type: str, preference_score: float):
        """Update task type preferences for a worker based on performance.
        
        Args:
            worker_id: ID of the worker
            task_type: Type of task
            preference_score: Preference score (0-1, higher means more preferred)
        """
        if not self.config["specialized_workers_enabled"]:
            return
            
        if worker_id not in self.worker_type_preferences:
            self.worker_type_preferences[worker_id] = {}
            
        # Update preference score
        self.worker_type_preferences[worker_id][task_type] = preference_score
        
        logger.debug(f"Updated task type preference for worker {worker_id}, type {task_type}: {preference_score:.2f}")
    
    def _calculate_type_preferences(self, worker_id: str):
        """Calculate task type preferences for a worker based on performance history.
        
        Args:
            worker_id: ID of the worker
        """
        if worker_id not in self.worker_metrics or "task_types" not in self.worker_metrics[worker_id]:
            return
            
        task_types = self.worker_metrics[worker_id]["task_types"]
        
        # Need global stats for comparison
        if not hasattr(self, "task_stats") or not self.task_stats:
            return
            
        preferences = {}
        
        for task_type, type_metrics in task_types.items():
            if task_type not in self.task_stats:
                continue
                
            global_stats = self.task_stats[task_type]
            
            # Need at least a few tasks for meaningful comparison
            if type_metrics.get("task_count", 0) < 3:
                continue
                
            # Calculate performance relative to global average
            worker_success_rate = type_metrics.get("success_rate", 0.5)
            global_success_rate = global_stats.get("success_rate", 0.5)
            
            worker_exec_time = type_metrics.get("avg_execution_time", 0)
            global_exec_time = global_stats.get("avg_execution_time", 0)
            
            # Calculate preference score
            success_factor = worker_success_rate / global_success_rate if global_success_rate > 0 else 1.0
            time_factor = global_exec_time / worker_exec_time if worker_exec_time > 0 else 1.0
            
            # Combine factors (success is more important)
            preference = (success_factor * 0.7 + time_factor * 0.3) - 1.0
            
            # Normalize to 0-1 range
            preference = max(0.0, min(1.0, preference))
            
            preferences[task_type] = preference
        
        # Update preferences
        for task_type, preference in preferences.items():
            self.update_type_preferences(worker_id, task_type, preference)
    
    def get_worker_rankings(self) -> Dict[str, float]:
        """Get current worker rankings.
        
        Returns:
            Dict mapping worker IDs to ranking scores
        """
        return self.worker_rankings
    
    def get_worker_load(self) -> Dict[str, Dict[str, Any]]:
        """Get current worker load information.
        
        Returns:
            Dict with worker load information
        """
        load_info = {}
        
        for worker_id in self.worker_task_counts:
            task_count = self.worker_task_counts[worker_id]
            capacity = self.worker_capacity.get(worker_id, 1)
            
            load_info[worker_id] = {
                "task_count": task_count,
                "capacity": capacity,
                "load_percent": (task_count / capacity * 100) if capacity > 0 else 100,
                "ranking": self.worker_rankings.get(worker_id, 50.0),
                "cpu_percent": self.worker_metrics.get(worker_id, {}).get("cpu_percent"),
                "memory_available_gb": self.worker_metrics.get(worker_id, {}).get("memory_available_gb")
            }
        
        return load_info
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get metrics about the load balancing strategy performance.
        
        Returns:
            Dict with strategy metrics
        """
        metrics = self.strategy_metrics.copy()
        
        # Calculate additional metrics
        if metrics["total_task_assignments"] > 0:
            metrics["optimal_assignment_percent"] = (metrics["optimal_assignments"] / metrics["total_task_assignments"]) * 100
        else:
            metrics["optimal_assignment_percent"] = 0
            
        return metrics