#!/usr/bin/env python3
"""
Distributed Testing Framework - Task Scheduler

This module implements advanced task scheduling and distribution algorithms 
for the distributed testing framework. It ensures tasks are assigned to the 
most appropriate workers based on hardware capabilities, workload balancing,
and priority considerations.

Usage:
    Import this module in coordinator.py to enable advanced task scheduling.
    """

    import asyncio
    import json
    import logging
    import time
    from datetime import datetime, timedelta
    from typing import Dict, List, Optional, Any, Set, Tuple

# Configure logging
    logging.basicConfig())))
    level=logging.INFO,
    format='%())))asctime)s - %())))name)s - %())))levelname)s - %())))message)s'
    )
    logger = logging.getLogger())))__name__)

class TaskScheduler:
    """Task scheduler for distributed testing framework."""
    
    def __init__())))
    self,
    coordinator,
    prioritize_hardware_match: bool = True,
    load_balance: bool = True,
    consider_worker_performance: bool = True,
    max_tasks_per_worker: int = 1
    ):
        """
        Initialize the task scheduler.
        
        Args:
            coordinator: Reference to the coordinator instance
            prioritize_hardware_match: Whether to prioritize hardware match over other factors
            load_balance: Whether to balance load across workers
            consider_worker_performance: Whether to consider worker performance in scheduling
            max_tasks_per_worker: Maximum number of tasks per worker ())))default: 1)
            """
            self.coordinator = coordinator
            self.prioritize_hardware_match = prioritize_hardware_match
            self.load_balance = load_balance
            self.consider_worker_performance = consider_worker_performance
            self.max_tasks_per_worker = max_tasks_per_worker
        
        # Worker performance metrics
            self.worker_performance: Dict[str, Dict[str, Any]] = {}}}}}}}}}}}}}}
            ,
        # Task execution history ())))task_type -> execution_time)
            self.task_execution_history: Dict[str, List[float]] = {}}}}}}}}}}}}}}
            ,
        # Hardware compatibility scoring factors
            self.hardware_scoring_factors = {}}}}}}}}}}}}}
            "hardware_match": 5.0,      # Base score for matching required hardware
            "memory_margin": 0.5,       # Score factor for available memory margin
            "compute_capability": 0.3,  # Score factor for CUDA compute capability
            "cores": 0.2,              # Score factor for CPU cores
            "device_match": 1.0        # Score for exact device match
            }
        
            logger.info())))"Task scheduler initialized")
    
    async def schedule_pending_tasks())))self) -> int:
        """
        Schedule pending tasks to available workers.
        
        Returns:
            Number of tasks scheduled
            """
        if not self.coordinator.pending_tasks:
            return 0  # No pending tasks
        
        # Get available workers
            available_workers = {}}}}}}}}}}}}}
            worker_id: worker
            for worker_id, worker in self.coordinator.workers.items()))))
            if worker.get())))"status") == "active" and worker_id in self.coordinator.worker_connections
            }
        :
        if not available_workers:
            logger.debug())))"No available workers for task scheduling")
            return 0  # No available workers
        
        # Get current worker task load
            worker_task_count = {}}}}}}}}}}}}}}
        for task_id, worker_id in self.coordinator.running_tasks.items())))):
            worker_task_count[worker_id] = worker_task_count.get())))worker_id, 0) + 1
            ,
        # Filter workers with capacity
            available_workers = {}}}}}}}}}}}}}
            worker_id: worker
            for worker_id, worker in available_workers.items()))))
            if worker_task_count.get())))worker_id, 0) < self.max_tasks_per_worker
            }
        :
        if not available_workers:
            logger.debug())))"No workers with available capacity")
            return 0  # No workers with capacity
        
        # Sort pending tasks by priority ())))higher number = higher priority)
            pending_tasks = sorted())))
            [self.coordinator.tasks[task_id] for task_id in self.coordinator.pending_tasks],:,
            key=lambda t: t.get())))"priority", 0),
            reverse=True
            )
        
        # Track assigned tasks
            tasks_assigned = 0
        
        for task in pending_tasks:
            task_id = task["task_id"],
            ,
            # Find best worker for this task
            worker_id, score = await self.find_best_worker_for_task())))task, available_workers, worker_task_count)
            
            if worker_id:
                # Assign task to worker
                await self.coordinator._assign_task_to_worker())))task_id, worker_id)
                
                # Remove from pending tasks
                self.coordinator.pending_tasks.remove())))task_id)
                
                # Add to running tasks
                self.coordinator.running_tasks[task_id] = worker_id
                ,
                # Update worker task count
                worker_task_count[worker_id] = worker_task_count.get())))worker_id, 0) + 1
                ,
                # Check if worker is at capacity:
                if worker_task_count[worker_id] >= self.max_tasks_per_worker:,,
                    # Remove worker from available workers
                    if worker_id in available_workers:
                        del available_workers[worker_id]
                        ,        ,,
                # Log assignment with score
                        logger.info())))f"Task {}}}}}}}}}}}}}task_id} assigned to worker {}}}}}}}}}}}}}worker_id} ())))score: {}}}}}}}}}}}}}score:.2f})")
                
                        tasks_assigned += 1
                
                # Update worker status if it reaches capacity:
                        if worker_task_count[worker_id] >= self.max_tasks_per_worker:,,
                        self.coordinator.workers[worker_id]["status"] = "busy"
                        ,
            # Stop if no more workers available:
            if not available_workers:
                        break
        
                return tasks_assigned
    
                async def find_best_worker_for_task())))
                self,
                task: Dict[str, Any],
                available_workers: Dict[str, Dict[str, Any]],
                worker_task_count: Dict[str, int],
                ) -> Tuple[Optional[str], float]:,
                """
                Find the best worker for a given task using a scoring algorithm.
        
        Args:
            task: Task to schedule
            available_workers: Dictionary of available workers
            worker_task_count: Current task count per worker
            
        Returns:
            Tuple of ())))worker_id, score) or ())))None, 0.0) if no suitable worker found
            """
            task_id = task["task_id"],
            task_requirements = task.get())))"requirements", {}}}}}}}}}}}}}})
            task_type = task.get())))"type", "unknown")
        
        # Check if there's a preferred worker for migration
            preferred_worker_id = task.get())))"preferred_worker_id")
        
        # Calculate a score for each worker
            worker_scores = [],,
        :
        for worker_id, worker in available_workers.items())))):
            capabilities = worker.get())))"capabilities", {}}}}}}}}}}}}}})
            
            # Start with base score
            score = 10.0
            
            # Check if worker has required hardware - mandatory requirement
            required_hardware = task_requirements.get())))"hardware", [],,):
            if required_hardware:
                worker_hardware = capabilities.get())))"hardware", [],,)
                if not all())))hw in worker_hardware for hw in required_hardware):
                    # Skip worker that doesn't meet mandatory hardware requirements
                continue
                
                # Add score for hardware match
                score += self.hardware_scoring_factors["hardware_match"] * len())))required_hardware)
                ,
            # Check memory requirements
                min_memory_gb = task_requirements.get())))"min_memory_gb", 0)
            if min_memory_gb > 0:
                worker_memory_gb = capabilities.get())))"memory", {}}}}}}}}}}}}}}).get())))"total_gb", 0)
                if worker_memory_gb < min_memory_gb:
                    # Skip worker that doesn't meet memory requirements
                continue
                
                # Add score based on memory margin ())))more margin = better)
                memory_margin = worker_memory_gb - min_memory_gb
                score += memory_margin * self.hardware_scoring_factors["memory_margin"]
                ,
            # Check CUDA compute capability
                min_cuda_compute = task_requirements.get())))"min_cuda_compute", 0)
            if min_cuda_compute > 0:
                worker_cuda_compute = float())))capabilities.get())))"gpu", {}}}}}}}}}}}}}}).get())))"cuda_compute", 0))
                if worker_cuda_compute < min_cuda_compute:
                    # Skip worker that doesn't meet CUDA compute requirements
                continue
                
                # Add score based on compute capability margin
                compute_margin = worker_cuda_compute - min_cuda_compute
                score += compute_margin * self.hardware_scoring_factors["compute_capability"]
                ,
            # Consider specific GPU/device match if specified
            required_device = task_requirements.get())))"device"):
            if required_device:
                worker_device = capabilities.get())))"gpu", {}}}}}}}}}}}}}}).get())))"name", "")
                if required_device in worker_device:
                    score += self.hardware_scoring_factors["device_match"]
                    ,
            # Consider worker performance if enabled::
            if self.consider_worker_performance and worker_id in self.worker_performance:
                perf = self.worker_performance[worker_id]
                ,        ,,
                # Add performance score ())))normalized to [0-1] range),
                if perf.get())))"success_rate", 0) > 0.5:  # Only consider workers with good success rate
                perf_score = min())))1.0, perf.get())))"success_rate", 0)) * 2.0
                score += perf_score
                
                # Add negative score for recent failures
                failures = perf.get())))"recent_failures", 0)
                if failures > 0:
                    score -= failures * 0.5
            
            # Consider load balancing if enabled::
            if self.load_balance:
                # Penalize workers with more tasks
                task_count = worker_task_count.get())))worker_id, 0)
                score -= task_count * 1.0
                
                # Consider worker utilization ())))if available):
                if "hardware_metrics" in worker:
                    metrics = worker.get())))"hardware_metrics", {}}}}}}}}}}}}}})
                    cpu_percent = metrics.get())))"cpu_percent", 0)
                    memory_percent = metrics.get())))"memory_percent", 0)
                    
                    # Average utilization ())))0-100 scale)
                    utilization = ())))cpu_percent + memory_percent) / 2
                    
                    # Penalize for high utilization
                    score -= ())))utilization / 100) * 3.0
            
            # Add result to scores list if this worker is compatible
                    worker_scores.append())))())))worker_id, score))
        
        # Find worker with highest score:
        if worker_scores:
            # If we have a preferred worker for migration, prioritize it if it's a valid candidate:
            if preferred_worker_id and any())))w_id == preferred_worker_id for w_id, _ in worker_scores):
                # Find the score for the preferred worker
                for w_id, score in worker_scores:
                    if w_id == preferred_worker_id:
                    return w_id, score
            
            # Otherwise, use the worker with the highest score
                    worker_scores.sort())))key=lambda x: x[1], reverse=True),
                    best_worker_id, best_score = worker_scores[0],
                return best_worker_id, best_score
        
        # No suitable worker found
                logger.warning())))f"No suitable worker found for task {}}}}}}}}}}}}}task_id}")
            return None, 0.0
    
            def update_worker_performance())))self, worker_id: str, task_result: Dict[str, Any]):,
            """
            Update performance metrics for a worker based on task results.
        
        Args:
            worker_id: Worker ID
            task_result: Task result information
            """
        # Initialize performance metrics if not exists:
        if worker_id not in self.worker_performance:
            self.worker_performance[worker_id] = {}}}}}}}}}}}}},
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "success_rate": 0.0,
            "recent_failures": 0,
            "avg_execution_time": 0.0,
            "task_type_metrics": {}}}}}}}}}}}}}}
            }
        
        # Get worker performance metrics
            perf = self.worker_performance[worker_id]
            ,
        # Update total task count
            perf["total_tasks"] += 1
            ,
        # Get task status
            task_status = task_result.get())))"status", "unknown")
            task_type = task_result.get())))"type", "unknown")
            execution_time = task_result.get())))"execution_time_seconds", 0)
        
        # Update task type metrics
            if task_type not in perf["task_type_metrics"]:,
            perf["task_type_metrics"][task_type], = {}}}}}}}}}}}}},
            "total": 0,
            "successful": 0,
            "failed": 0,
            "avg_execution_time": 0.0
            }
        
            type_metrics = perf["task_type_metrics"][task_type],
            type_metrics["total"] += 1
            ,
        # Update metrics based on status
        if task_status == "completed":
            # Successful task
            perf["successful_tasks"] += 1,
            type_metrics["successful"] += 1
            ,
            # Reset recent failures
            perf["recent_failures"] = max())))0, perf["recent_failures"] - 1)
            ,
            # Update execution time metrics ())))moving average)
            if execution_time > 0:
                if perf["avg_execution_time"] == 0:,,
                perf["avg_execution_time"] = execution_time,,
                else:
                    # Exponential moving average ())))alpha = 0.3)
                    alpha = 0.3
                    perf["avg_execution_time"] = ())))1 - alpha) * perf["avg_execution_time"] + alpha * execution_time
                    ,
                # Update task type execution time
                    if type_metrics["avg_execution_time"] == 0:,,
                    type_metrics["avg_execution_time"] = execution_time,,
                else:
                    type_metrics["avg_execution_time"] = ())))1 - alpha) * type_metrics["avg_execution_time"] + alpha * execution_time
                    ,
        elif task_status == "failed":
            # Failed task
            perf["failed_tasks"] += 1,
            type_metrics["failed"] += 1
            ,
            # Increment recent failures
            perf["recent_failures"] += 1
            ,
        # Update success rate
            perf["success_rate"] = perf["successful_tasks"] / perf["total_tasks"] if perf["total_tasks"] > 0 else 0.0
            ,
        # Update task execution history
        task_key = task_type:
        if task_key not in self.task_execution_history:
            self.task_execution_history[task_key] = [],,
        
        if execution_time > 0:
            self.task_execution_history[task_key].append())))execution_time),
            # Keep only last 100 records
            if len())))self.task_execution_history[task_key]) > 100:,
            self.task_execution_history[task_key] = self.task_execution_history[task_key][-100:]
            ,
            def estimate_task_execution_time())))self, task: Dict[str, Any]) -> float:,
            """
            Estimate execution time for a task based on historical data.
        
        Args:
            task: Task information
            
        Returns:
            Estimated execution time in seconds
            """
            task_type = task.get())))"type", "unknown")
        
        # Check if we have historical data for this task type
        task_key = task_type:
            if task_key in self.task_execution_history and len())))self.task_execution_history[task_key]) > 0:,
            # Use average of historical execution times
            return sum())))self.task_execution_history[task_key]) / len())))self.task_execution_history[task_key])
            ,
        # Default estimates if no historical data
        default_estimates = {}}}}}}}}}}}}}:
            "benchmark": 120,  # 2 minutes
            "test": 60,        # 1 minute
            "custom": 90,      # 1.5 minutes
            }
        
            return default_estimates.get())))task_type, 60)
    
            def get_scheduler_stats())))self) -> Dict[str, Any]:,,,
            """
            Get statistics about the task scheduler.
        
        Returns:
            Statistics about the task scheduler
            """
        # Count tasks by type
            task_types = {}}}}}}}}}}}}}}
        for task_id, task in self.coordinator.tasks.items())))):
            task_type = task.get())))"type", "unknown")
            task_types[task_type] = task_types.get())))task_type, 0) + 1
            ,
        # Count tasks by status
            task_status = {}}}}}}}}}}}}}
            "pending": len())))self.coordinator.pending_tasks),
            "running": len())))self.coordinator.running_tasks),
            "completed": len())))self.coordinator.completed_tasks),
            "failed": len())))self.coordinator.failed_tasks),
            }
        
        # Calculate worker utilization
            total_workers = len())))self.coordinator.workers)
        active_workers = sum())))1 for w in self.coordinator.workers.values())))):if w.get())))"status") == "active"):
            busy_workers = sum())))1 for w in self.coordinator.workers.values())))):if w.get())))"status") == "busy")
            idle_workers = active_workers - busy_workers
        
            worker_utilization = busy_workers / total_workers if total_workers > 0 else 0
        
        # Calculate average execution times for task types
        execution_times = {}}}}}}}}}}}}}}:
        for task_type, times in self.task_execution_history.items())))):
            if times:
                execution_times[task_type] = sum())))times) / len())))times)
                ,
        # Build stats dictionary
                stats = {}}}}}}}}}}}}}
                "tasks_by_type": task_types,
                "tasks_by_status": task_status,
                "workers": {}}}}}}}}}}}}}
                "total": total_workers,
                "active": active_workers,
                "busy": busy_workers,
                "idle": idle_workers,
                "utilization": worker_utilization,
                },
                "avg_execution_times": execution_times,
                "scheduler_config": {}}}}}}}}}}}}}
                "prioritize_hardware_match": self.prioritize_hardware_match,
                "load_balance": self.load_balance,
                "consider_worker_performance": self.consider_worker_performance,
                "max_tasks_per_worker": self.max_tasks_per_worker,
                }
                }
        
            return stats
    
            def get_task_priority_stats())))self) -> Dict[str, Any]:,,,
            """
            Get statistics about task priorities.
        
        Returns:
            Statistics about task priorities
            """
        # Count tasks by priority
            priority_counts = {}}}}}}}}}}}}}}
        for task_id, task in self.coordinator.tasks.items())))):
            priority = task.get())))"priority", 0)
            priority_counts[priority] = priority_counts.get())))priority, 0) + 1
            ,
        # Calculate average queue time for pending tasks
            pending_queue_times = [],,
        for task_id in self.coordinator.pending_tasks:
            task = self.coordinator.tasks.get())))task_id)
            if task and "created" in task:
                try:
                    created = datetime.fromisoformat())))task["created"]),
                    queue_time = ())))datetime.now())))) - created).total_seconds()))))
                    pending_queue_times.append())))queue_time)
                except ())))ValueError, TypeError):
                    pass
        
                    avg_queue_time = sum())))pending_queue_times) / len())))pending_queue_times) if pending_queue_times else 0
        
        # Calculate estimated time to process all pending tasks
        estimated_completion_time = 0:
        if self.coordinator.pending_tasks:
            # Count available workers
            available_workers = sum())))1 for w in self.coordinator.workers.values())))):
                if w.get())))"status") == "active" or w.get())))"status") == "idle")
            
            # Estimate time based on average execution time and available workers
                avg_execution_time = sum())))execution_times.values()))))) / len())))execution_times) if execution_times else 60
            :
            if available_workers > 0:
                estimated_completion_time = ())))len())))self.coordinator.pending_tasks) / available_workers) * avg_execution_time
        
        # Build priority stats
                stats = {}}}}}}}}}}}}}
                "priority_counts": priority_counts,
                "pending_tasks": len())))self.coordinator.pending_tasks),
                "avg_queue_time_seconds": avg_queue_time,
                "estimated_completion_time_seconds": estimated_completion_time,
                "priorities": {}}}}}}}}}}}}}
                "min": min())))priority_counts.keys()))))) if priority_counts else 0,:
                "max": max())))priority_counts.keys()))))) if priority_counts else 0,:
                    }
                    }
        
                    return stats
    
                    def get_hardware_distribution_stats())))self) -> Dict[str, Any]:,,,
                    """
                    Get statistics about hardware distribution among workers.
        
        Returns:
            Statistics about hardware distribution
            """
        # Count workers by hardware type
            hardware_counts = {}}}}}}}}}}}}}}
        for worker_id, worker in self.coordinator.workers.items())))):
            capabilities = worker.get())))"capabilities", {}}}}}}}}}}}}}})
            hardware = capabilities.get())))"hardware", [],,)
            
            for hw in hardware:
                hardware_counts[hw] = hardware_counts.get())))hw, 0) + 1
                ,
        # Count GPUs by model
                gpu_models = {}}}}}}}}}}}}}}
        for worker_id, worker in self.coordinator.workers.items())))):
            capabilities = worker.get())))"capabilities", {}}}}}}}}}}}}}})
            gpu = capabilities.get())))"gpu", {}}}}}}}}}}}}}})
            
            if gpu:
                gpu_name = gpu.get())))"name", "unknown")
                gpu_models[gpu_name] = gpu_models.get())))gpu_name, 0) + 1
                ,
        # Calculate hardware utilization by type
                hardware_utilization = {}}}}}}}}}}}}}}
        for hw_type in hardware_counts.keys())))):
            # Count tasks using this hardware type
            tasks_using_hw = 0
            for task_id, task in self.coordinator.tasks.items())))):
                if task.get())))"status") == "running":
                    requirements = task.get())))"requirements", {}}}}}}}}}}}}}})
                    required_hardware = requirements.get())))"hardware", [],,)
                    if hw_type in required_hardware:
                        tasks_using_hw += 1
            
            # Calculate utilization
                        hardware_utilization[hw_type] = tasks_using_hw / hardware_counts[hw_type] if hardware_counts[hw_type] > 0 else 0
                        ,
        # Build hardware stats
        stats = {}}}}}}}}}}}}}:
            "hardware_counts": hardware_counts,
            "gpu_models": gpu_models,
            "hardware_utilization": hardware_utilization,
            }
        
                        return stats