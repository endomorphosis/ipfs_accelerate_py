#!/usr/bin/env python3
"""
Hardware Monitoring Integration for Coordinator

This module integrates the hardware utilization monitor with the coordinator
and task scheduler components of the distributed testing framework. It enables
resource-aware task scheduling based on real-time hardware utilization metrics.

Key features:
- Integration with coordinator for hardware-aware scheduling
- Real-time hardware utilization monitoring during task execution
- Database integration for utilization metrics storage
- Resource-aware task assignment based on current hardware load
- Performance history tracking for predictive scheduling
- Threshold-based alert integration with coordinator status

Usage:
    integration = CoordinatorHardwareMonitoringIntegration(
        coordinator_instance,
        db_path="./hardware_metrics.duckdb"
    )
    integration.initialize()
    # Coordinator will now use hardware utilization for task scheduling
"""

import os
import sys
import json
import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path

# Import hardware utilization monitor
from hardware_utilization_monitor import (
    HardwareUtilizationMonitor, 
    MonitoringLevel,
    ResourceUtilization,
    TaskResourceUsage,
    HardwareAlert
)

# Import hardware capability detector
from hardware_capability_detector import (
    HardwareCapabilityDetector,
    HardwareType,
    HardwareVendor,
    PrecisionType,
    CapabilityScore,
    HardwareCapability,
    WorkerHardwareCapabilities
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_hardware_monitoring")


class CoordinatorHardwareMonitoringIntegration:
    """
    Integrates hardware utilization monitoring with the coordinator component
    for resource-aware task scheduling.
    """
    
    def __init__(
        self,
        coordinator,
        db_path: Optional[str] = None,
        monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
        enable_resource_aware_scheduling: bool = True,
        worker_monitors: Optional[Dict[str, HardwareUtilizationMonitor]] = None,
        hardware_detector: Optional[HardwareCapabilityDetector] = None,
        utilization_threshold: float = 80.0,
        update_interval_seconds: float = 5.0
    ):
        """
        Initialize the hardware monitoring integration.
        
        Args:
            coordinator: Reference to the coordinator instance
            db_path: Path to DuckDB database for metrics storage
            monitoring_level: Level of detail for monitoring
            enable_resource_aware_scheduling: Whether to enable resource-aware scheduling
            worker_monitors: Dictionary of worker_id -> HardwareUtilizationMonitor
            hardware_detector: Optional hardware capability detector instance
            utilization_threshold: Threshold for considering a worker overloaded
            update_interval_seconds: Interval for updating utilization metrics
        """
        self.coordinator = coordinator
        self.db_path = db_path
        self.monitoring_level = monitoring_level
        self.enable_resource_aware_scheduling = enable_resource_aware_scheduling
        self.worker_monitors = worker_monitors or {}
        self.hardware_detector = hardware_detector
        self.utilization_threshold = utilization_threshold
        self.update_interval_seconds = update_interval_seconds
        
        # Create hardware detector if not provided
        if not self.hardware_detector and db_path:
            self.hardware_detector = HardwareCapabilityDetector(
                db_path=self.db_path
            )
        
        # Internal state
        self.initialized = False
        self.update_thread = None
        self.worker_utilization_cache = {}  # worker_id -> utilization metrics
        self.task_resource_history = {}  # task_type -> list of task resource usage
        self.update_running = False
        
        # Integration with task scheduler
        self.original_find_best_worker = None
        self.original_update_worker_performance = None
    
    def initialize(self):
        """Initialize the hardware monitoring integration."""
        if self.initialized:
            logger.warning("Hardware monitoring integration already initialized")
            return
        
        logger.info("Initializing hardware monitoring integration")
        
        # Create scheduler reference if coordinator has a task scheduler
        if hasattr(self.coordinator, 'task_scheduler'):
            self.scheduler = self.coordinator.task_scheduler
            # Save original method references for patching
            self.original_find_best_worker = self.scheduler.find_best_worker_for_task
            self.original_update_worker_performance = self.scheduler.update_worker_performance
            
            # Patch scheduler methods with our versions
            if self.enable_resource_aware_scheduling:
                self._patch_scheduler_methods()
        else:
            logger.warning("Coordinator does not have a task_scheduler attribute")
            self.scheduler = None
        
        # Register for coordinator events
        self._register_coordinator_callbacks()
        
        # Initialize worker monitors for existing workers
        self._init_worker_monitors()
        
        # Start update thread
        self.update_running = True
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
        self.initialized = True
        logger.info("Hardware monitoring integration initialized successfully")
    
    def shutdown(self):
        """Shutdown the hardware monitoring integration."""
        if not self.initialized:
            return
        
        logger.info("Shutting down hardware monitoring integration")
        
        # Stop update thread
        self.update_running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        
        # Stop all worker monitors
        for worker_id, monitor in self.worker_monitors.items():
            logger.debug(f"Stopping monitor for worker {worker_id}")
            monitor.stop_monitoring()
        
        # Restore original scheduler methods
        if self.scheduler and self.original_find_best_worker:
            self.scheduler.find_best_worker_for_task = self.original_find_best_worker
            self.scheduler.update_worker_performance = self.original_update_worker_performance
        
        self.initialized = False
        logger.info("Hardware monitoring integration shut down")
    
    def _init_worker_monitors(self):
        """Initialize monitors for existing workers."""
        if not self.coordinator.workers:
            logger.info("No existing workers to initialize monitors for")
            return
        
        logger.info(f"Initializing monitors for {len(self.coordinator.workers)} existing workers")
        
        for worker_id, worker_info in self.coordinator.workers.items():
            self._create_worker_monitor(worker_id)
    
    def _create_worker_monitor(self, worker_id: str) -> HardwareUtilizationMonitor:
        """
        Create a hardware utilization monitor for a worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            HardwareUtilizationMonitor instance
        """
        if worker_id in self.worker_monitors:
            return self.worker_monitors[worker_id]
        
        logger.info(f"Creating hardware monitor for worker {worker_id}")
        
        monitor = HardwareUtilizationMonitor(
            worker_id=worker_id,
            db_path=self.db_path,
            monitoring_level=self.monitoring_level,
            hardware_detector=self.hardware_detector
        )
        
        # Register alert callback
        monitor.register_alert_callback(lambda alert: self._handle_worker_alert(worker_id, alert))
        
        # Store in monitors dictionary
        self.worker_monitors[worker_id] = monitor
        
        # Start monitoring
        monitor.start_monitoring()
        
        return monitor
    
    def _handle_worker_alert(self, worker_id: str, alert: HardwareAlert):
        """
        Handle hardware alert from a worker.
        
        Args:
            worker_id: Worker ID
            alert: Hardware alert
        """
        logger.warning(f"Hardware alert from worker {worker_id}: {alert.message}")
        
        # Update worker status based on alert severity
        if alert.severity == "critical":
            # Mark worker as overloaded
            if hasattr(self.coordinator, 'workers') and worker_id in self.coordinator.workers:
                self.coordinator.workers[worker_id]["status"] = "overloaded"
                
                # Log to coordinator event log if available
                if hasattr(self.coordinator, 'log_event'):
                    self.coordinator.log_event(
                        event_type="worker_hardware_alert",
                        worker_id=worker_id,
                        alert=alert.message,
                        severity=alert.severity,
                        resource_type=alert.resource_type,
                        metric_value=alert.metric_value,
                        threshold=alert.threshold
                    )
    
    def _register_coordinator_callbacks(self):
        """Register callbacks with the coordinator."""
        # Check if coordinator has the necessary methods
        if hasattr(self.coordinator, 'register_worker_callback'):
            # Register for worker registration events
            self.coordinator.register_worker_callback(
                event="register",
                callback=self._on_worker_registered
            )
            
            # Register for worker deregistration events
            self.coordinator.register_worker_callback(
                event="deregister",
                callback=self._on_worker_deregistered
            )
        else:
            logger.warning("Coordinator does not support callback registration")
    
    def _on_worker_registered(self, worker_id: str, worker_info: Dict[str, Any]):
        """
        Callback for worker registration.
        
        Args:
            worker_id: Worker ID
            worker_info: Worker information
        """
        logger.info(f"Worker registered: {worker_id}")
        
        # Create monitor for new worker
        self._create_worker_monitor(worker_id)
    
    def _on_worker_deregistered(self, worker_id: str):
        """
        Callback for worker deregistration.
        
        Args:
            worker_id: Worker ID
        """
        logger.info(f"Worker deregistered: {worker_id}")
        
        # Stop and remove monitor for worker
        if worker_id in self.worker_monitors:
            logger.debug(f"Stopping monitor for worker {worker_id}")
            self.worker_monitors[worker_id].stop_monitoring()
            del self.worker_monitors[worker_id]
    
    def _update_loop(self):
        """Background thread for updating utilization metrics."""
        try:
            import time  # Import time at the function level to avoid global import issues
            while self.update_running:
                # Update utilization metrics for all workers
                self._update_worker_utilization()
                
                # Sleep for interval
                time.sleep(self.update_interval_seconds)
        except Exception as e:
            logger.error(f"Error in update loop: {str(e)}")
            self.update_running = False
    
    def _update_worker_utilization(self):
        """Update worker utilization metrics."""
        # Skip if no workers
        if not self.worker_monitors:
            return
        
        for worker_id, monitor in self.worker_monitors.items():
            try:
                # Get current metrics
                metrics = monitor.get_current_metrics()
                if not metrics:
                    continue
                
                # Update cache
                self.worker_utilization_cache[worker_id] = {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "gpu_utilization": [
                        {
                            "id": gpu.get("id", 0),
                            "load": gpu.get("load", 0.0),
                            "memory_percent": gpu.get("memory_percent", 0.0)
                        }
                        for gpu in metrics.gpu_utilization
                    ],
                    "disk_percent": metrics.disk_percent,
                    "timestamp": metrics.timestamp.isoformat(),
                    "updated": datetime.now().isoformat()
                }
                
                # Update worker information in coordinator if available
                if hasattr(self.coordinator, 'workers') and worker_id in self.coordinator.workers:
                    # Update hardware metrics in worker info
                    if "hardware_metrics" not in self.coordinator.workers[worker_id]:
                        self.coordinator.workers[worker_id]["hardware_metrics"] = {}
                    
                    self.coordinator.workers[worker_id]["hardware_metrics"] = {
                        "cpu_percent": metrics.cpu_percent,
                        "memory_percent": metrics.memory_percent,
                        "gpu_percent": max([gpu.get("load", 0.0) for gpu in metrics.gpu_utilization]) if metrics.gpu_utilization else 0.0,
                        "disk_percent": metrics.disk_percent,
                        "updated": datetime.now().isoformat()
                    }
                    
                    # Update worker status based on utilization
                    if self._is_worker_overloaded(metrics):
                        if self.coordinator.workers[worker_id].get("status") != "overloaded":
                            logger.warning(f"Worker {worker_id} is overloaded, updating status")
                            self.coordinator.workers[worker_id]["status"] = "overloaded"
                    elif self.coordinator.workers[worker_id].get("status") == "overloaded":
                        # Reset status if no longer overloaded
                        logger.info(f"Worker {worker_id} is no longer overloaded, resetting status")
                        self.coordinator.workers[worker_id]["status"] = "active"
            
            except Exception as e:
                logger.error(f"Error updating utilization for worker {worker_id}: {str(e)}")
    
    def _is_worker_overloaded(self, metrics: ResourceUtilization) -> bool:
        """
        Check if a worker is overloaded based on utilization metrics.
        
        Args:
            metrics: Resource utilization metrics
            
        Returns:
            True if worker is overloaded, False otherwise
        """
        # Check CPU utilization
        if metrics.cpu_percent >= self.utilization_threshold:
            return True
        
        # Check memory utilization
        if metrics.memory_percent >= self.utilization_threshold:
            return True
        
        # Check GPU utilization
        for gpu_info in metrics.gpu_utilization:
            if gpu_info.get("load", 0.0) >= self.utilization_threshold:
                return True
            if gpu_info.get("memory_percent", 0.0) >= self.utilization_threshold:
                return True
        
        # Check disk utilization
        if metrics.disk_percent >= 95.0:  # Higher threshold for disk
            return True
        
        return False
    
    def _patch_scheduler_methods(self):
        """Patch task scheduler methods with hardware-aware versions."""
        if not self.scheduler:
            logger.warning("No task scheduler to patch")
            return
        
        logger.info("Patching task scheduler methods for hardware-aware scheduling")
        
        # Store original methods
        if not self.original_find_best_worker:
            self.original_find_best_worker = self.scheduler.find_best_worker_for_task
        
        if not self.original_update_worker_performance:
            self.original_update_worker_performance = self.scheduler.update_worker_performance
        
        # Replace with our hardware-aware versions
        self.scheduler.find_best_worker_for_task = self._hardware_aware_find_best_worker
        self.scheduler.update_worker_performance = self._hardware_aware_update_worker_performance
    
    async def _hardware_aware_find_best_worker(self, task: Dict[str, Any], 
                                       available_workers: Dict[str, Dict[str, Any]],
                                       worker_task_count: Dict[str, int]) -> Tuple[Optional[str], float]:
        """
        Hardware-aware version of find_best_worker_for_task that considers utilization.
        
        Args:
            task: Task to schedule
            available_workers: Dictionary of available workers
            worker_task_count: Current task count per worker
            
        Returns:
            Tuple of (worker_id, score) or (None, 0.0) if no suitable worker found
        """
        # Call original method to get base worker selection
        worker_id, score = await self.original_find_best_worker(
            task, available_workers, worker_task_count
        )
        
        # If no worker found or not hardware aware, return original result
        if not worker_id or not self.enable_resource_aware_scheduling:
            return worker_id, score
        
        # Adjust score based on hardware utilization
        adjusted_score = score
        
        # Get current utilization for the selected worker
        utilization = self.worker_utilization_cache.get(worker_id, {})
        if utilization:
            # Penalize for high CPU utilization
            cpu_percent = utilization.get("cpu_percent", 0.0)
            if cpu_percent > 70.0:
                cpu_penalty = (cpu_percent - 70.0) / 10.0  # 0.0 to 3.0 penalty
                adjusted_score -= cpu_penalty
                logger.debug(f"Applied CPU penalty of {cpu_penalty:.2f} to worker {worker_id}")
            
            # Penalize for high memory utilization
            memory_percent = utilization.get("memory_percent", 0.0)
            if memory_percent > 70.0:
                memory_penalty = (memory_percent - 70.0) / 10.0  # 0.0 to 3.0 penalty
                adjusted_score -= memory_penalty
                logger.debug(f"Applied memory penalty of {memory_penalty:.2f} to worker {worker_id}")
            
            # Penalize for high GPU utilization if task requires GPU
            gpu_utilization = utilization.get("gpu_utilization", [])
            if gpu_utilization and "hardware" in task.get("requirements", {}) and "gpu" in task["requirements"]["hardware"]:
                max_gpu_load = max([gpu.get("load", 0.0) for gpu in gpu_utilization]) if gpu_utilization else 0.0
                if max_gpu_load > 60.0:  # GPU threshold lower since GPU tasks are more sensitive
                    gpu_penalty = (max_gpu_load - 60.0) / 8.0  # 0.0 to 5.0 penalty
                    adjusted_score -= gpu_penalty
                    logger.debug(f"Applied GPU penalty of {gpu_penalty:.2f} to worker {worker_id}")
            
            # Check if worker is severely overloaded (prevent assignment)
            if self._is_worker_critically_overloaded(utilization):
                logger.warning(f"Worker {worker_id} is critically overloaded, preventing assignment")
                return None, 0.0
        
        # If score adjusted significantly, log it
        if abs(adjusted_score - score) > 1.0:
            logger.info(f"Adjusted worker {worker_id} score from {score:.2f} to {adjusted_score:.2f} based on hardware utilization")
        
        # Find a better worker if this one is not optimal
        if adjusted_score < score * 0.7:  # If adjusted score is much worse
            # Try to find another worker with better utilization
            alternate_worker_id, alternate_score = self._find_alternate_worker(
                task, available_workers, worker_task_count, worker_id
            )
            
            if alternate_worker_id and alternate_score > adjusted_score:
                logger.info(f"Selected alternate worker {alternate_worker_id} with better utilization (score: {alternate_score:.2f} vs {adjusted_score:.2f})")
                return alternate_worker_id, alternate_score
        
        return worker_id, adjusted_score
    
    def _is_worker_critically_overloaded(self, utilization: Dict[str, Any]) -> bool:
        """
        Check if a worker is critically overloaded.
        
        Args:
            utilization: Worker utilization metrics
            
        Returns:
            True if worker is critically overloaded, False otherwise
        """
        # Check CPU utilization
        if utilization.get("cpu_percent", 0.0) >= 95.0:
            return True
        
        # Check memory utilization
        if utilization.get("memory_percent", 0.0) >= 95.0:
            return True
        
        # Check GPU utilization
        gpu_utilization = utilization.get("gpu_utilization", [])
        for gpu in gpu_utilization:
            if gpu.get("load", 0.0) >= 98.0:
                return True
            if gpu.get("memory_percent", 0.0) >= 98.0:
                return True
        
        return False
    
    def _find_alternate_worker(self, task: Dict[str, Any],
                             available_workers: Dict[str, Dict[str, Any]],
                             worker_task_count: Dict[str, int],
                             original_worker_id: str) -> Tuple[Optional[str], float]:
        """
        Find an alternate worker with better utilization.
        
        Args:
            task: Task to schedule
            available_workers: Dictionary of available workers
            worker_task_count: Current task count per worker
            original_worker_id: Original selected worker
            
        Returns:
            Tuple of (worker_id, score) or (None, 0.0) if no suitable worker found
        """
        best_worker_id = None
        best_score = 0.0
        
        # Check all other available workers
        for worker_id, worker in available_workers.items():
            if worker_id == original_worker_id:
                continue
            
            # Skip workers that are overloaded
            utilization = self.worker_utilization_cache.get(worker_id, {})
            if utilization and self._is_worker_critically_overloaded(utilization):
                continue
            
            # Calculate hardware match score
            match_score = self._calculate_hardware_match_score(task, worker)
            if match_score <= 0:
                continue  # Skip if not compatible
            
            # Calculate utilization score (lower utilization = higher score)
            utilization_score = self._calculate_utilization_score(worker_id)
            
            # Calculate task count score (fewer tasks = higher score)
            task_count = worker_task_count.get(worker_id, 0)
            task_count_score = max(0, 5.0 - task_count)
            
            # Calculate final score
            score = match_score + utilization_score + task_count_score
            
            # Update best worker if score is higher
            if score > best_score:
                best_worker_id = worker_id
                best_score = score
        
        return best_worker_id, best_score
    
    def _calculate_hardware_match_score(self, task: Dict[str, Any], worker: Dict[str, Any]) -> float:
        """
        Calculate hardware match score for a task and worker.
        
        Args:
            task: Task to schedule
            worker: Worker information
            
        Returns:
            Hardware match score
        """
        score = 10.0  # Base score
        
        # Check if worker has required hardware
        required_hardware = task.get("requirements", {}).get("hardware", [])
        if required_hardware:
            worker_hardware = worker.get("capabilities", {}).get("hardware", [])
            if not all(hw in worker_hardware for hw in required_hardware):
                return 0.0  # Not compatible
            
            # Add score for hardware match
            score += len(required_hardware) * 5.0
        
        # Check memory requirements
        min_memory_gb = task.get("requirements", {}).get("min_memory_gb", 0)
        if min_memory_gb > 0:
            worker_memory_gb = worker.get("capabilities", {}).get("memory", {}).get("total_gb", 0)
            if worker_memory_gb < min_memory_gb:
                return 0.0  # Not compatible
            
            # Add score based on memory margin
            memory_margin = worker_memory_gb - min_memory_gb
            score += memory_margin * 0.5
        
        return score
    
    def _calculate_utilization_score(self, worker_id: str) -> float:
        """
        Calculate utilization score for a worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Utilization score (higher is better, means less utilized)
        """
        utilization = self.worker_utilization_cache.get(worker_id, {})
        if not utilization:
            return 5.0  # Default score if no utilization data
        
        # Calculate CPU score (0-10, higher is better)
        cpu_percent = utilization.get("cpu_percent", 0.0)
        cpu_score = 10.0 - (cpu_percent / 10.0)
        
        # Calculate memory score (0-10, higher is better)
        memory_percent = utilization.get("memory_percent", 0.0)
        memory_score = 10.0 - (memory_percent / 10.0)
        
        # Calculate GPU score if present
        gpu_score = 10.0
        gpu_utilization = utilization.get("gpu_utilization", [])
        if gpu_utilization:
            max_gpu_load = max([gpu.get("load", 0.0) for gpu in gpu_utilization])
            max_gpu_memory = max([gpu.get("memory_percent", 0.0) for gpu in gpu_utilization])
            gpu_score = 10.0 - (max(max_gpu_load, max_gpu_memory) / 10.0)
        
        # Weighted average (prioritize the most constrained resource)
        return min(cpu_score, memory_score, gpu_score)
    
    def _hardware_aware_update_worker_performance(self, worker_id: str, task_result: Dict[str, Any]):
        """
        Hardware-aware version of update_worker_performance that records resource usage.
        
        Args:
            worker_id: Worker ID
            task_result: Task result information
        """
        # Call original method
        self.original_update_worker_performance(worker_id, task_result)
        
        # Get task resource usage if available
        if worker_id in self.worker_monitors:
            monitor = self.worker_monitors[worker_id]
            task_id = task_result.get("task_id")
            
            if task_id:
                # Get resource usage for this task
                task_usage = monitor.get_task_metrics(task_id)
                if task_usage:
                    # Store in task resource history
                    task_type = task_result.get("type", "unknown")
                    if task_type not in self.task_resource_history:
                        self.task_resource_history[task_type] = []
                    
                    # Limit history size
                    if len(self.task_resource_history[task_type]) >= 100:
                        self.task_resource_history[task_type] = self.task_resource_history[task_type][-99:]
                    
                    # Add to history
                    self.task_resource_history[task_type].append({
                        "task_id": task_id,
                        "worker_id": worker_id,
                        "start_time": task_usage.start_time.isoformat(),
                        "end_time": task_usage.end_time.isoformat() if task_usage.end_time else None,
                        "peak_cpu_percent": task_usage.peak_cpu_percent,
                        "peak_memory_percent": task_usage.peak_memory_percent,
                        "peak_gpu_percent": task_usage.peak_gpu_percent,
                        "avg_cpu_percent": task_usage.avg_cpu_percent,
                        "avg_memory_percent": task_usage.avg_memory_percent,
                        "avg_gpu_percent": task_usage.avg_gpu_percent,
                        "completed": task_usage.completed,
                        "success": task_usage.success
                    })
                    
                    # Update task type stats in scheduler if available
                    if hasattr(self.scheduler, 'task_type_stats'):
                        if task_type not in self.scheduler.task_type_stats:
                            self.scheduler.task_type_stats[task_type] = {
                                "count": 0,
                                "avg_cpu_percent": 0.0,
                                "avg_memory_percent": 0.0,
                                "avg_gpu_percent": 0.0,
                                "peak_cpu_percent": 0.0,
                                "peak_memory_percent": 0.0,
                                "peak_gpu_percent": 0.0
                            }
                        
                        # Update stats with exponential moving average
                        stats = self.scheduler.task_type_stats[task_type]
                        alpha = 0.3  # Smoothing factor
                        
                        stats["count"] += 1
                        stats["avg_cpu_percent"] = (1 - alpha) * stats["avg_cpu_percent"] + alpha * task_usage.avg_cpu_percent
                        stats["avg_memory_percent"] = (1 - alpha) * stats["avg_memory_percent"] + alpha * task_usage.avg_memory_percent
                        stats["avg_gpu_percent"] = (1 - alpha) * stats["avg_gpu_percent"] + alpha * task_usage.avg_gpu_percent
                        stats["peak_cpu_percent"] = max(stats["peak_cpu_percent"], task_usage.peak_cpu_percent)
                        stats["peak_memory_percent"] = max(stats["peak_memory_percent"], task_usage.peak_memory_percent)
                        stats["peak_gpu_percent"] = max(stats["peak_gpu_percent"], task_usage.peak_gpu_percent)
    
    def get_worker_utilization(self, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get worker utilization metrics.
        
        Args:
            worker_id: Optional worker ID to filter for a specific worker
            
        Returns:
            Dictionary of worker utilization metrics
        """
        if worker_id:
            return self.worker_utilization_cache.get(worker_id, {})
        
        return self.worker_utilization_cache
    
    def get_task_resource_history(self, task_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get task resource usage history.
        
        Args:
            task_type: Optional task type to filter for a specific type
            
        Returns:
            Dictionary of task type -> list of task resource usage
        """
        if task_type:
            return {task_type: self.task_resource_history.get(task_type, [])}
        
        return self.task_resource_history
    
    def start_task_monitoring(self, task_id: str, worker_id: str):
        """
        Start monitoring resources for a specific task.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
        """
        if worker_id in self.worker_monitors:
            logger.info(f"Starting resource monitoring for task {task_id} on worker {worker_id}")
            self.worker_monitors[worker_id].start_task_monitoring(task_id)
    
    def stop_task_monitoring(self, task_id: str, worker_id: str, success: bool = True, error: Optional[str] = None):
        """
        Stop monitoring resources for a specific task.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
            success: Whether the task completed successfully
            error: Error message if the task failed
            
        Returns:
            TaskResourceUsage object if available
        """
        if worker_id in self.worker_monitors:
            logger.info(f"Stopping resource monitoring for task {task_id} on worker {worker_id}")
            return self.worker_monitors[worker_id].stop_task_monitoring(task_id, success, error)
        
        return None
    
    def generate_resource_report(self, worker_id: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a resource utilization report.
        
        Args:
            worker_id: Optional worker ID to filter for a specific worker
            task_id: Optional task ID to filter for a specific task
            
        Returns:
            Dictionary with resource report data
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "workers": {},
            "tasks": {},
            "summary": {
                "worker_count": len(self.worker_monitors),
                "active_tasks": 0
            }
        }
        
        # Add worker utilization data
        for w_id, utilization in self.worker_utilization_cache.items():
            if worker_id and w_id != worker_id:
                continue
                
            report["workers"][w_id] = utilization
        
        # Add task resource data
        task_count = 0
        for task_type, tasks in self.task_resource_history.items():
            report["tasks"][task_type] = []
            
            for task in tasks:
                if task_id and task["task_id"] != task_id:
                    continue
                
                report["tasks"][task_type].append(task)
                task_count += 1
        
        # Add summary data
        report["summary"]["task_count"] = task_count
        report["summary"]["active_tasks"] = sum(1 for m in self.worker_monitors.values() if m.active_task_id)
        
        # Add scheduler stats if available
        if self.scheduler and hasattr(self.scheduler, 'get_scheduler_stats'):
            try:
                report["scheduler_stats"] = self.scheduler.get_scheduler_stats()
            except Exception as e:
                logger.error(f"Error getting scheduler stats: {str(e)}")
        
        return report
    
    def generate_html_report(self, file_path: str, worker_id: Optional[str] = None, task_id: Optional[str] = None):
        """
        Generate an HTML report of resource utilization.
        
        Args:
            file_path: Path to output HTML file
            worker_id: Optional worker ID to filter for a specific worker
            task_id: Optional task ID to filter for a specific task
        """
        # Generate report data
        report_data = self.generate_resource_report(worker_id, task_id)
        
        # Create HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Resource Utilization Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .section {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .warning {{
            color: #e67e22;
        }}
        .critical {{
            color: #e74c3c;
        }}
        .success {{
            color: #2ecc71;
        }}
        .failure {{
            color: #e74c3c;
        }}
    </style>
</head>
<body>
    <h1>Resource Utilization Report</h1>
    <p>Report Time: {report_data["timestamp"]}</p>
    <p>Worker Count: {report_data["summary"]["worker_count"]}</p>
    <p>Active Tasks: {report_data["summary"]["active_tasks"]}</p>
    
    <div class="section">
        <h2>Worker Utilization</h2>
        <table>
            <tr>
                <th>Worker ID</th>
                <th>CPU (%)</th>
                <th>Memory (%)</th>
                <th>GPU (%)</th>
                <th>Disk (%)</th>
                <th>Updated</th>
            </tr>
"""
        
        # Add worker rows
        for w_id, utilization in report_data["workers"].items():
            cpu_class = ""
            mem_class = ""
            
            cpu_percent = utilization.get("cpu_percent", 0.0)
            memory_percent = utilization.get("memory_percent", 0.0)
            
            if cpu_percent >= 90.0:
                cpu_class = "critical"
            elif cpu_percent >= 75.0:
                cpu_class = "warning"
                
            if memory_percent >= 90.0:
                mem_class = "critical"
            elif memory_percent >= 75.0:
                mem_class = "warning"
            
            # Calculate max GPU utilization
            gpu_utilization = utilization.get("gpu_utilization", [])
            max_gpu_load = max([gpu.get("load", 0.0) for gpu in gpu_utilization]) if gpu_utilization else 0.0
            gpu_class = ""
            
            if max_gpu_load >= 90.0:
                gpu_class = "critical"
            elif max_gpu_load >= 75.0:
                gpu_class = "warning"
            
            html += f"""
            <tr>
                <td>{w_id}</td>
                <td class="{cpu_class}">{cpu_percent:.1f}%</td>
                <td class="{mem_class}">{memory_percent:.1f}%</td>
                <td class="{gpu_class}">{max_gpu_load:.1f}%</td>
                <td>{utilization.get("disk_percent", 0.0):.1f}%</td>
                <td>{utilization.get("updated", "N/A")}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Task Resource Usage</h2>
"""
        
        # Check if we have any task data
        has_task_data = False
        for task_type, tasks in report_data["tasks"].items():
            if tasks:
                has_task_data = True
                break
        
        if has_task_data:
            # Add task type sections
            for task_type, tasks in report_data["tasks"].items():
                if not tasks:
                    continue
                    
                html += f"""
        <h3>Task Type: {task_type}</h3>
        <table>
            <tr>
                <th>Task ID</th>
                <th>Worker ID</th>
                <th>Status</th>
                <th>Start Time</th>
                <th>End Time</th>
                <th>Peak CPU (%)</th>
                <th>Peak Memory (%)</th>
                <th>Peak GPU (%)</th>
                <th>Avg CPU (%)</th>
                <th>Avg Memory (%)</th>
                <th>Avg GPU (%)</th>
            </tr>
"""
                
                for task in tasks:
                    status_class = "success" if task.get("success") else "failure"
                    status_text = "Completed" if task.get("completed") else "Running"
                    status_text = "Success" if task.get("completed") and task.get("success") else status_text
                    status_text = "Failed" if task.get("completed") and not task.get("success") else status_text
                    
                    html += f"""
            <tr>
                <td>{task["task_id"]}</td>
                <td>{task["worker_id"]}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{task["start_time"]}</td>
                <td>{task["end_time"] or "Running"}</td>
                <td>{task["peak_cpu_percent"]:.1f}%</td>
                <td>{task["peak_memory_percent"]:.1f}%</td>
                <td>{task["peak_gpu_percent"]:.1f}%</td>
                <td>{task["avg_cpu_percent"]:.1f}%</td>
                <td>{task["avg_memory_percent"]:.1f}%</td>
                <td>{task["avg_gpu_percent"]:.1f}%</td>
            </tr>
"""
                
                html += """
        </table>
"""
        else:
            html += """
        <p>No task resource usage data available.</p>
"""
        
        html += """
    </div>
"""
        
        # Add scheduler stats if available
        if "scheduler_stats" in report_data:
            html += """
    <div class="section">
        <h2>Scheduler Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""
            
            scheduler_stats = report_data["scheduler_stats"]
            
            # Tasks by status
            if "tasks_by_status" in scheduler_stats:
                for status, count in scheduler_stats["tasks_by_status"].items():
                    html += f"""
            <tr>
                <td>Tasks {status}</td>
                <td>{count}</td>
            </tr>
"""
            
            # Workers
            if "workers" in scheduler_stats:
                for metric, value in scheduler_stats["workers"].items():
                    html += f"""
            <tr>
                <td>Workers {metric}</td>
                <td>{value if not isinstance(value, float) else f"{value:.2f}"}</td>
            </tr>
"""
            
            html += """
        </table>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated HTML report at {file_path}")


# Example usage
def main():
    """Simple example of using the hardware monitoring integration."""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Hardware Monitoring Integration Example")
    parser.add_argument("--db-path", default="./hardware_metrics.duckdb", help="Path to DuckDB database")
    parser.add_argument("--duration", type=int, default=30, help="Example duration in seconds")
    parser.add_argument("--report", help="Generate HTML report at path")
    
    args = parser.parse_args()
    
    print("This is an example of using the hardware monitoring integration.")
    print("In a real application, you would pass your coordinator instance to the integration.")
    print(f"Running example for {args.duration} seconds...")
    
    # Create a mock coordinator for demonstration
    class MockCoordinator:
        def __init__(self):
            self.workers = {
                "worker1": {"status": "active", "capabilities": {"hardware": ["cpu", "gpu"]}},
                "worker2": {"status": "active", "capabilities": {"hardware": ["cpu"]}}
            }
            self.task_scheduler = MockTaskScheduler()
            self.pending_tasks = []
            self.running_tasks = {}
            self.callbacks = {}
        
        def register_worker_callback(self, event, callback):
            if event not in self.callbacks:
                self.callbacks[event] = []
            self.callbacks[event].append(callback)
            
        def log_event(self, **kwargs):
            print(f"Event logged: {kwargs}")
    
    class MockTaskScheduler:
        def __init__(self):
            self.task_type_stats = {}
        
        async def find_best_worker_for_task(self, task, available_workers, worker_task_count):
            # Simple mock implementation
            for worker_id in available_workers:
                return worker_id, 10.0
            return None, 0.0
        
        def update_worker_performance(self, worker_id, task_result):
            # Simple mock implementation
            print(f"Updating performance for worker {worker_id}")
        
        def get_scheduler_stats(self):
            return {
                "tasks_by_status": {"pending": 2, "running": 1, "completed": 5},
                "workers": {"total": 2, "active": 2, "busy": 0, "idle": 2, "utilization": 0.0}
            }
    
    # Create mock coordinator
    coordinator = MockCoordinator()
    
    # Create integration
    integration = CoordinatorHardwareMonitoringIntegration(
        coordinator=coordinator,
        db_path=args.db_path,
        monitoring_level=MonitoringLevel.DETAILED,
        update_interval_seconds=1.0
    )
    
    # Initialize integration
    integration.initialize()
    
    try:
        # Simulate some worker activity
        for i in range(args.duration):
            if i == 5:
                # Simulate task start
                integration.start_task_monitoring("task1", "worker1")
                print("Started task1 on worker1")
            
            if i == 15:
                # Simulate task completion
                integration.stop_task_monitoring("task1", "worker1", success=True)
                print("Completed task1 on worker1")
            
            # Print current utilization
            for worker_id, utilization in integration.get_worker_utilization().items():
                cpu = utilization.get("cpu_percent", 0.0)
                memory = utilization.get("memory_percent", 0.0)
                print(f"Worker {worker_id}: CPU {cpu:.1f}%, Memory {memory:.1f}%")
            
            # Sleep
            time.sleep(1)
    
    finally:
        # Generate report if requested
        if args.report:
            integration.generate_html_report(args.report)
            print(f"Generated report at {args.report}")
        
        # Shutdown integration
        integration.shutdown()
        print("Integration shut down")


if __name__ == "__main__":
    main()