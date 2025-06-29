#!/usr/bin/env python3
"""
Distributed Testing Framework - Advanced Adaptive Load Balancer

This module implements the advanced adaptive load balancing system for the distributed
testing framework. It's responsible for:

- Dynamic threshold adjustment based on system-wide load
- Cost-benefit analysis for task migrations
- Predictive load balancing based on historical data
- Resource efficiency optimization
- Hardware-specific balancing strategies
- Workload redistribution based on performance
- Automatic task migration between workers
- Optimal resource utilization
- Handling heterogeneous hardware environments

The Advanced Adaptive Load Balancer uses real-time performance data from workers,
historical performance metrics, and the Result Aggregation system to make
intelligent load balancing decisions.

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
    """Advanced adaptive load balancer for the distributed testing framework."""
    
    def __init__(self, worker_manager=None, task_scheduler=None, db_manager=None, 
                 result_aggregator=None, detailed_result_aggregator=None):
        """Initialize the load balancer.
        
        Args:
            worker_manager: Worker manager instance
            task_scheduler: Task scheduler instance
            db_manager: Database manager instance
            result_aggregator: High-level result aggregator instance
            detailed_result_aggregator: Detailed result aggregator instance
        """
        self.worker_manager = worker_manager
        self.task_scheduler = task_scheduler
        self.db_manager = db_manager
        self.result_aggregator = result_aggregator
        self.detailed_result_aggregator = detailed_result_aggregator
        
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
        
        # System load tracking for dynamic threshold adjustment
        self.system_load_history = []  # List of historical system load records
        self.predicted_load = {}  # worker_id -> predicted load
        self.prediction_accuracy_history = []  # List of prediction accuracy records
        
        # Task migration cost-benefit tracking
        self.migration_costs = {}  # task_type -> historical migration cost
        self.migration_benefits = {}  # worker_id -> historical migration benefit
        self.migration_success_rate = {}  # task_type -> historical success rate
        
        # Hardware-specific profiles
        self.hardware_profiles = self._initialize_hardware_profiles()
        
        # Configuration
        self.config = {
            # Base configuration
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
            
            # Advanced adaptive load balancing options
            "check_interval": 30,  # Interval for load balance checks in seconds
            "utilization_threshold_high": 0.85,  # Initial threshold for high utilization (0.0-1.0)
            "utilization_threshold_low": 0.2,  # Initial threshold for low utilization (0.0-1.0)
            "performance_window": 5,  # Window size for performance measurements in minutes
            "enable_dynamic_thresholds": True,  # Whether to dynamically adjust thresholds based on system load
            "enable_predictive_balancing": True,  # Whether to predict future load and proactively balance
            "enable_cost_benefit_analysis": True,  # Whether to analyze cost vs benefit of migrations
            "enable_hardware_specific_strategies": True,  # Whether to use hardware-specific balancing strategies
            "enable_resource_efficiency": True,  # Whether to consider resource efficiency in balancing
            "threshold_adjustment_rate": 0.05,  # Rate at which thresholds are adjusted (0.0-1.0)
            "prediction_window": 3,  # Window size for load prediction in minutes
            "max_simultaneous_migrations": 2,  # Maximum number of simultaneous task migrations
            "min_threshold_separation": 0.3,  # Minimum separation between high and low thresholds
            "prediction_confidence_threshold": 0.7,  # Minimum confidence level for predictions to trigger actions
            "imbalance_threshold": 0.4,  # Minimum imbalance level to trigger balancing (0.0-1.0)
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
        
    def _initialize_hardware_profiles(self):
        """Initialize hardware-specific profiles for balancing strategies.
        
        Returns:
            Dictionary of hardware profiles
        """
        # Base hardware profiles with performance metrics
        profiles = {
            "CPU": {
                "performance_weight": 1.0,
                "energy_efficiency": 0.7,
                "thermal_efficiency": 0.8,
                "memory_efficiency": 0.8,
                "best_task_types": ["general", "cpu_bound"],
                "scaling_factor": 1.0
            },
            "CUDA": {
                "performance_weight": 3.0,
                "energy_efficiency": 0.5,
                "thermal_efficiency": 0.4,
                "memory_efficiency": 0.7,
                "best_task_types": ["gpu_compute", "model_training", "benchmark"],
                "scaling_factor": 2.5
            },
            "ROCm": {
                "performance_weight": 2.8,
                "energy_efficiency": 0.5,
                "thermal_efficiency": 0.4,
                "memory_efficiency": 0.7,
                "best_task_types": ["gpu_compute", "model_training", "benchmark"],
                "scaling_factor": 2.3
            },
            "MPS": {
                "performance_weight": 2.5,
                "energy_efficiency": 0.6,
                "thermal_efficiency": 0.6,
                "memory_efficiency": 0.8,
                "best_task_types": ["gpu_compute", "model_training", "benchmark"],
                "scaling_factor": 2.0
            },
            "OpenVINO": {
                "performance_weight": 1.8,
                "energy_efficiency": 0.8,
                "thermal_efficiency": 0.7,
                "memory_efficiency": 0.9,
                "best_task_types": ["inference", "model_optimization"],
                "scaling_factor": 1.5
            },
            "QNN": {
                "performance_weight": 1.4,
                "energy_efficiency": 0.9,
                "thermal_efficiency": 0.9,
                "memory_efficiency": 0.8,
                "best_task_types": ["mobile", "edge_inference"],
                "scaling_factor": 1.2
            },
            "WebNN": {
                "performance_weight": 1.0,
                "energy_efficiency": 0.7,
                "thermal_efficiency": 0.8,
                "memory_efficiency": 0.7,
                "best_task_types": ["browser", "web_inference"],
                "scaling_factor": 1.0
            },
            "WebGPU": {
                "performance_weight": 1.2,
                "energy_efficiency": 0.6,
                "thermal_efficiency": 0.7,
                "memory_efficiency": 0.6,
                "best_task_types": ["browser", "web_compute"],
                "scaling_factor": 1.1
            }
        }
        
        # Add any additional profiles from result aggregator if available
        if self.result_aggregator or self.detailed_result_aggregator:
            try:
                # If detailed aggregator is available, use it for hardware-specific 
                # performance data as it has more detailed breakdowns
                if self.detailed_result_aggregator:
                    hardware_dimension = self.detailed_result_aggregator.get_dimension_analysis("hardware")
                    
                    if hardware_dimension:
                        # Update profiles with actual performance data
                        for hardware_type, stats in hardware_dimension.items():
                            if hardware_type in profiles:
                                # Extract performance metrics (normalized)
                                if "throughput" in stats:
                                    throughput = stats["throughput"].get("mean", 1.0)
                                    # Use throughput to adjust performance weight
                                    base_weight = profiles[hardware_type]["performance_weight"]
                                    profiles[hardware_type]["performance_weight"] = base_weight * (throughput / 100.0)
                                
                                # Extract energy metrics if available
                                if "energy_usage" in stats:
                                    energy = stats["energy_usage"].get("mean", 1.0)
                                    profiles[hardware_type]["energy_efficiency"] = 1.0 / (energy / 100.0) if energy > 0 else 0.5
            except Exception as e:
                logger.warning(f"Error updating hardware profiles from result aggregator: {e}")
                
        return profiles
    
    def _monitoring_loop(self):
        """Resource monitoring thread function for advanced adaptive load balancing."""
        while not self.monitoring_stop_event.is_set():
            try:
                # Update worker performance metrics
                self._update_performance_metrics()
                
                # Update worker rankings
                if (datetime.now() - self.last_ranking_update).total_seconds() >= self.config["ranking_update_interval"]:
                    self._update_worker_rankings()
                    self.last_ranking_update = datetime.now()
                
                # Apply dynamic threshold adjustment if enabled
                if self.config["enable_dynamic_thresholds"]:
                    self._update_dynamic_thresholds()
                
                # Perform predictive load analysis if enabled
                if self.config["enable_predictive_balancing"]:
                    self._predict_future_load()
                
                # Check for load imbalance and rebalance if necessary
                if (datetime.now() - self.last_rebalance_check).total_seconds() >= self.config["check_interval"]:
                    load_imbalance = self.detect_load_imbalance()
                    
                    if load_imbalance:
                        if self.config["enable_cost_benefit_analysis"]:
                            self.balance_load_with_cost_benefit()
                        else:
                            self.balance_load()
                            
                    self.last_rebalance_check = datetime.now()
                    
                # Record metrics in database
                self._record_metrics()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                logger.debug(f"Stack trace: {traceback.format_exc()}")
                
            # Wait for next interval
            self.monitoring_stop_event.wait(self.config["resource_monitoring_interval"])
    
    def _update_performance_metrics(self):
        """Update worker performance metrics from all available sources."""
        # Get current workers
        if not self.worker_manager:
            return
            
        try:
            # Get all workers
            available_workers = self.worker_manager.get_available_workers()
            
            # Calculate system-wide metrics
            total_workers = len(available_workers)
            total_utilization = 0
            min_util = 1.0
            max_util = 0.0
            
            # Process each worker
            for worker in available_workers:
                worker_id = worker["worker_id"]
                
                # Extract worker hardware metrics
                hardware_metrics = worker.get("hardware_metrics", {})
                capabilities = worker.get("capabilities", {})
                
                # Initialize worker metrics if not exists
                if worker_id not in self.worker_metrics:
                    self.worker_metrics[worker_id] = {}
                
                # Update CPU metrics
                if "cpu_percent" in hardware_metrics:
                    self.worker_metrics[worker_id]["cpu_percent"] = hardware_metrics["cpu_percent"]
                
                # Update memory metrics
                if "memory_used_gb" in hardware_metrics and "memory_total_gb" in hardware_metrics:
                    memory_used = hardware_metrics["memory_used_gb"]
                    memory_total = hardware_metrics["memory_total_gb"]
                    memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                    
                    self.worker_metrics[worker_id]["memory_used_gb"] = memory_used
                    self.worker_metrics[worker_id]["memory_total_gb"] = memory_total
                    self.worker_metrics[worker_id]["memory_percent"] = memory_percent
                
                # Update GPU metrics if available
                if "gpu_metrics" in hardware_metrics and hardware_metrics["gpu_metrics"]:
                    gpu_metrics = hardware_metrics["gpu_metrics"][0]  # Use first GPU for now
                    
                    if "load_percent" in gpu_metrics:
                        self.worker_metrics[worker_id]["gpu_load_percent"] = gpu_metrics["load_percent"]
                    
                    if "memory_used_mb" in gpu_metrics and "memory_total_mb" in gpu_metrics:
                        gpu_memory_used = gpu_metrics["memory_used_mb"]
                        gpu_memory_total = gpu_metrics["memory_total_mb"]
                        gpu_memory_percent = (gpu_memory_used / gpu_memory_total * 100) if gpu_memory_total > 0 else 0
                        
                        self.worker_metrics[worker_id]["gpu_memory_used_mb"] = gpu_memory_used
                        self.worker_metrics[worker_id]["gpu_memory_total_mb"] = gpu_memory_total
                        self.worker_metrics[worker_id]["gpu_memory_percent"] = gpu_memory_percent
                
                # Calculate overall utilization (weighted average of CPU, memory, GPU)
                # This can be customized based on hardware type
                utilization = self._calculate_overall_utilization(worker_id)
                self.worker_metrics[worker_id]["overall_utilization"] = utilization
                
                # Update system-wide metrics
                total_utilization += utilization
                min_util = min(min_util, utilization)
                max_util = max(max_util, utilization)
                
                # Update load history
                if worker_id not in self.worker_load_history:
                    self.worker_load_history[worker_id] = []
                
                # Add to history, keeping limited size
                self.worker_load_history[worker_id].append({
                    "timestamp": datetime.now(),
                    "utilization": utilization,
                    "cpu_percent": self.worker_metrics[worker_id].get("cpu_percent", 0),
                    "memory_percent": self.worker_metrics[worker_id].get("memory_percent", 0),
                    "gpu_load_percent": self.worker_metrics[worker_id].get("gpu_load_percent", 0),
                    "task_count": self.worker_task_counts.get(worker_id, 0)
                })
                
                # Trim history if needed
                if len(self.worker_load_history[worker_id]) > self.config["load_history_size"]:
                    self.worker_load_history[worker_id].pop(0)
            
            # Calculate system-wide average utilization
            avg_system_load = total_utilization / total_workers if total_workers > 0 else 0
            
            # Calculate imbalance score
            imbalance_score = max_util - min_util
            
            # Update system load history
            self.system_load_history.append({
                "timestamp": datetime.now(),
                "avg_utilization": avg_system_load,
                "min_utilization": min_util,
                "max_utilization": max_util,
                "imbalance_score": imbalance_score,
                "worker_count": total_workers
            })
            
            # Trim system load history if needed
            if len(self.system_load_history) > self.config["load_history_size"]:
                self.system_load_history.pop(0)
                
            # Get additional performance data from result aggregators if available
            self._update_from_result_aggregators()
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_overall_utilization(self, worker_id):
        """Calculate overall utilization for a worker based on all metrics.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            Overall utilization as a float between 0.0 and 1.0
        """
        metrics = self.worker_metrics.get(worker_id, {})
        
        # Define weights for different metrics based on hardware type
        weights = {"cpu": 0.3, "memory": 0.3, "gpu": 0.4}
        
        # Adjust weights based on hardware type if known
        hardware_types = self.worker_manager.get_worker(worker_id).get("capabilities", {}).get("hardware_types", [])
        if hardware_types:
            primary_type = hardware_types[0] if hardware_types else "CPU"
            if primary_type in ["CUDA", "ROCm", "MPS"]:
                weights = {"cpu": 0.2, "memory": 0.3, "gpu": 0.5}  # GPU-heavy
            elif primary_type in ["CPU"]:
                weights = {"cpu": 0.5, "memory": 0.3, "gpu": 0.2}  # CPU-heavy
        
        # Get normalized metrics (0.0-1.0)
        cpu_util = metrics.get("cpu_percent", 0) / 100.0
        memory_util = metrics.get("memory_percent", 0) / 100.0
        gpu_util = metrics.get("gpu_load_percent", 0) / 100.0 if "gpu_load_percent" in metrics else 0.0
        
        # Calculate weighted sum
        overall = (
            weights["cpu"] * cpu_util +
            weights["memory"] * memory_util +
            weights["gpu"] * gpu_util
        )
        
        # Normalize to 0.0-1.0
        return max(0.0, min(1.0, overall))
    
    def _update_from_result_aggregators(self):
        """Update performance metrics from result aggregators."""
        if not (self.result_aggregator or self.detailed_result_aggregator):
            return
            
        try:
            # First try to get data from the detailed aggregator
            if self.detailed_result_aggregator:
                # Get worker performance metrics
                worker_dimension = self.detailed_result_aggregator.get_dimension_analysis("worker")
                
                if worker_dimension:
                    for worker_id, metrics in worker_dimension.items():
                        # Skip if worker not in our current list
                        if worker_id not in self.worker_metrics:
                            continue
                            
                        # Extract performance metrics
                        if "throughput" in metrics:
                            self.worker_metrics[worker_id]["aggregated_throughput"] = metrics["throughput"].get("mean", 0)
                            
                        if "latency_ms" in metrics:
                            self.worker_metrics[worker_id]["aggregated_latency"] = metrics["latency_ms"].get("mean", 0)
                            
                        if "success_rate" in metrics:
                            self.worker_metrics[worker_id]["aggregated_success_rate"] = metrics["success_rate"].get("mean", 0)
                
                # Get task type performance metrics
                task_type_dimension = self.detailed_result_aggregator.get_dimension_analysis("task_type")
                
                if task_type_dimension:
                    for task_type, metrics in task_type_dimension.items():
                        # Process for task type preferences
                        for worker_id in self.worker_metrics:
                            # Initialize task type preferences if needed
                            if worker_id not in self.worker_type_preferences:
                                self.worker_type_preferences[worker_id] = {}
                                
                            # Check if we have worker-specific task type metrics
                            if self.task_scheduler and hasattr(self.task_scheduler, "worker_performance"):
                                worker_perf = self.task_scheduler.worker_performance.get(worker_id, {})
                                task_types = worker_perf.get("task_types", {})
                                
                                if task_type in task_types:
                                    type_perf = task_types[task_type]
                                    success_rate = type_perf.get("success_rate", 0.5)
                                    avg_execution_time = type_perf.get("avg_execution_time", 0)
                                    
                                    # Calculate preference score (higher is better)
                                    preference_score = success_rate
                                    
                                    if "avg_execution_time" in metrics and avg_execution_time > 0:
                                        # Normalize execution time (faster is better)
                                        avg_time = metrics["avg_execution_time"].get("mean", 0)
                                        if avg_time > 0:
                                            time_score = avg_time / avg_execution_time
                                            preference_score = 0.7 * success_rate + 0.3 * min(time_score, 2.0)
                                    
                                    self.worker_type_preferences[worker_id][task_type] = preference_score
            
            # Get data from the high-level aggregator if needed
            elif self.result_aggregator:
                # Get worker performance metrics
                perf_results = self.result_aggregator.aggregate_results(
                    result_type="performance",
                    aggregation_level="worker"
                )
                
                if perf_results and "results" in perf_results:
                    basic_stats = perf_results["results"].get("basic_statistics", {})
                    
                    for worker_id, metrics in basic_stats.items():
                        # Skip if worker not in our current list
                        if worker_id not in self.worker_metrics:
                            continue
                            
                        # Extract relevant metrics
                        for metric_name, metric_values in metrics.items():
                            if metric_name in ["throughput", "latency_ms", "success_rate"]:
                                self.worker_metrics[worker_id][f"aggregated_{metric_name}"] = metric_values.get("mean", 0)
        except Exception as e:
            logger.warning(f"Error updating metrics from result aggregators: {e}")
    
    def _update_dynamic_thresholds(self):
        """Dynamically adjust utilization thresholds based on system load and trends."""
        if not self.system_load_history or len(self.system_load_history) < 3:
            return  # Need at least 3 data points for trend analysis
            
        try:
            # Get recent system load data
            recent_loads = self.system_load_history[-min(5, len(self.system_load_history)):]
            avg_system_load = sum(record["avg_utilization"] for record in recent_loads) / len(recent_loads)
            
            # Calculate load trend using linear regression
            x = list(range(len(recent_loads)))
            y = [record["avg_utilization"] for record in recent_loads]
            
            # Calculate trend slope (linear regression)
            n = len(x)
            if n < 2:
                trend_slope = 0
            else:
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_xx = sum(x[i] * x[i] for i in range(n))
                
                # Simple linear regression formula
                trend_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
            
            # Determine adjustment factors
            adjustment_rate = self.config["threshold_adjustment_rate"]
            
            if trend_slope > 0.01:  # Load increasing
                adjustment_factor = adjustment_rate * 1.5
                direction = "increasing"
            elif trend_slope < -0.01:  # Load decreasing
                adjustment_factor = adjustment_rate * 0.5
                direction = "decreasing"
            else:  # Load stable
                adjustment_factor = adjustment_rate
                direction = "stable"
            
            # Adjust based on current load
            if avg_system_load > 0.75:  # High load
                high_adjust = -adjustment_factor * 1.2  # Lower high threshold
                low_adjust = adjustment_factor * 0.8    # Raise low threshold
            elif avg_system_load < 0.3:  # Low load
                high_adjust = adjustment_factor * 0.8   # Raise high threshold
                low_adjust = -adjustment_factor * 1.2   # Lower low threshold
            else:  # Normal load
                high_adjust = -adjustment_factor if avg_system_load > 0.5 else adjustment_factor
                low_adjust = adjustment_factor if avg_system_load > 0.5 else -adjustment_factor
            
            # Apply adjustments within boundaries
            new_high = max(0.6, min(0.95, self.config["utilization_threshold_high"] + high_adjust))
            new_low = max(0.1, min(0.4, self.config["utilization_threshold_low"] + low_adjust))
            
            # Ensure minimum separation
            min_separation = self.config["min_threshold_separation"]
            if new_high - new_low < min_separation:
                if high_adjust < low_adjust:
                    new_high = new_low + min_separation
                else:
                    new_low = new_high - min_separation
            
            # Update thresholds
            self.config["utilization_threshold_high"] = new_high
            self.config["utilization_threshold_low"] = new_low
            
            logger.debug(f"Dynamic thresholds adjusted: high={new_high:.2f}, low={new_low:.2f}, load={avg_system_load:.2f}, trend={direction}")
            
        except Exception as e:
            logger.error(f"Error updating dynamic thresholds: {e}")
    
    def _predict_future_load(self):
        """Predict future load for each worker and system-wide using linear regression."""
        prediction_window_seconds = self.config["prediction_window"] * 60  # Convert minutes to seconds
        
        try:
            # Predict for each worker
            for worker_id, history in self.worker_load_history.items():
                if len(history) < 3:
                    continue  # Need at least 3 data points for prediction
                
                # Extract timestamps and utilization
                timestamps = []
                utilizations = []
                
                for record in history:
                    timestamp = record["timestamp"]
                    utilization = record["utilization"]
                    
                    # Convert timestamp to seconds since start
                    seconds = (timestamp - history[0]["timestamp"]).total_seconds()
                    timestamps.append(seconds)
                    utilizations.append(utilization)
                
                # Calculate trend using linear regression
                n = len(timestamps)
                sum_x = sum(timestamps)
                sum_y = sum(utilizations)
                sum_xy = sum(timestamps[i] * utilizations[i] for i in range(n))
                sum_xx = sum(timestamps[i] * timestamps[i] for i in range(n))
                
                # Linear regression formula
                if (n * sum_xx - sum_x * sum_x) != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                    intercept = (sum_y - slope * sum_x) / n
                else:
                    slope = 0
                    intercept = sum_y / n if n > 0 else 0
                
                # Predict future utilization
                future_seconds = timestamps[-1] + prediction_window_seconds
                predicted_utilization = intercept + slope * future_seconds
                
                # Ensure prediction is within 0.0-1.0
                predicted_utilization = max(0.0, min(1.0, predicted_utilization))
                
                # Calculate prediction confidence based on variance
                if n >= 3:
                    # Calculate variance of the residuals
                    residuals = [utilizations[i] - (intercept + slope * timestamps[i]) for i in range(n)]
                    variance = sum(r**2 for r in residuals) / n
                    
                    # Higher variance = lower confidence
                    confidence = max(0.0, min(1.0, 1.0 - variance * 5))
                else:
                    confidence = 0.5  # Default confidence
                
                # Store prediction
                self.predicted_load[worker_id] = {
                    "timestamp": datetime.now(),
                    "current_utilization": utilizations[-1] if utilizations else 0,
                    "predicted_utilization": predicted_utilization,
                    "prediction_window_seconds": prediction_window_seconds,
                    "confidence": confidence,
                    "slope": slope  # Store trend direction
                }
                
            # Record prediction accuracy if previous predictions exist
            self._calculate_prediction_accuracy()
            
        except Exception as e:
            logger.error(f"Error in future load prediction: {e}")
    
    def _calculate_prediction_accuracy(self):
        """Calculate accuracy of previous load predictions."""
        try:
            # Check each worker with current metrics
            total_error = 0
            count = 0
            
            for worker_id, prediction in list(self.predicted_load.items()):
                # Skip recent predictions
                predict_age = (datetime.now() - prediction["timestamp"]).total_seconds()
                if predict_age < prediction["prediction_window_seconds"]:
                    continue
                    
                # Get actual utilization
                if worker_id in self.worker_metrics:
                    actual_util = self.worker_metrics[worker_id].get("overall_utilization", 0)
                    predicted_util = prediction["predicted_utilization"]
                    
                    # Calculate error (absolute difference)
                    error = abs(actual_util - predicted_util)
                    total_error += error
                    count += 1
                    
                    # Store accuracy for this prediction
                    accuracy = 1.0 - min(error, 1.0)  # 0.0-1.0, higher is better
                    
                    # Add to accuracy history
                    self.prediction_accuracy_history.append({
                        "timestamp": datetime.now(),
                        "worker_id": worker_id,
                        "predicted": predicted_util,
                        "actual": actual_util,
                        "error": error,
                        "accuracy": accuracy
                    })
                    
                    # Limit history size
                    if len(self.prediction_accuracy_history) > 100:
                        self.prediction_accuracy_history.pop(0)
                    
                # Remove old prediction
                del self.predicted_load[worker_id]
            
            # Calculate overall accuracy if we have data
            if count > 0:
                avg_error = total_error / count
                avg_accuracy = 1.0 - min(avg_error, 1.0)
                logger.debug(f"Prediction accuracy: {avg_accuracy:.2f} (error: {avg_error:.2f}, count: {count})")
                
        except Exception as e:
            logger.error(f"Error calculating prediction accuracy: {e}")
            
    def detect_load_imbalance(self):
        """Detect current or predicted load imbalance.
        
        Returns:
            Dict with imbalance information or None if no imbalance detected
        """
        try:
            # Get available workers
            available_workers = self.worker_manager.get_available_workers()
            if not available_workers or len(available_workers) < 2:
                return None  # Need at least 2 workers to detect imbalance
                
            # Process current imbalance
            worker_utils = {}
            max_util = 0
            min_util = 1.0
            max_worker = None
            min_worker = None
            
            for worker in available_workers:
                worker_id = worker["worker_id"]
                
                # Skip if worker metrics not available
                if worker_id not in self.worker_metrics:
                    continue
                    
                # Get current utilization
                utilization = self.worker_metrics[worker_id].get("overall_utilization", 0)
                worker_utils[worker_id] = utilization
                
                # Track min/max
                if utilization > max_util:
                    max_util = utilization
                    max_worker = worker_id
                    
                if utilization < min_util:
                    min_util = utilization
                    min_worker = worker_id
            
            # Calculate imbalance
            current_imbalance = max_util - min_util
            
            # Check against thresholds for current imbalance
            high_threshold = self.config["utilization_threshold_high"]
            low_threshold = self.config["utilization_threshold_low"]
            imbalance_threshold = self.config["imbalance_threshold"]
            
            current_imbalance_detected = (
                max_util >= high_threshold and 
                min_util <= low_threshold and
                current_imbalance >= imbalance_threshold
            )
            
            # Check predicted imbalance if enabled
            predicted_imbalance_detected = False
            predicted_max_util = 0
            predicted_min_util = 1.0
            predicted_max_worker = None
            predicted_min_worker = None
            predicted_imbalance = 0
            
            if self.config["enable_predictive_balancing"] and self.predicted_load:
                worker_predictions = {}
                
                for worker_id, prediction in self.predicted_load.items():
                    # Skip if confidence is too low
                    if prediction["confidence"] < self.config["prediction_confidence_threshold"]:
                        continue
                        
                    # Get predicted utilization
                    predicted_util = prediction["predicted_utilization"]
                    worker_predictions[worker_id] = predicted_util
                    
                    # Track min/max
                    if predicted_util > predicted_max_util:
                        predicted_max_util = predicted_util
                        predicted_max_worker = worker_id
                        
                    if predicted_util < predicted_min_util:
                        predicted_min_util = predicted_util
                        predicted_min_worker = worker_id
                
                # Calculate predicted imbalance
                if predicted_max_worker and predicted_min_worker:
                    predicted_imbalance = predicted_max_util - predicted_min_util
                    
                    # Check against thresholds with higher bar for prediction
                    predicted_imbalance_detected = (
                        predicted_max_util >= high_threshold * 1.1 and
                        predicted_min_util <= low_threshold * 0.9 and
                        predicted_imbalance >= imbalance_threshold * 1.2
                    )
            
            # Return results if imbalance detected
            if current_imbalance_detected or predicted_imbalance_detected:
                return {
                    "current_imbalance": {
                        "detected": current_imbalance_detected,
                        "imbalance_value": current_imbalance,
                        "max_worker": max_worker,
                        "max_utilization": max_util,
                        "min_worker": min_worker,
                        "min_utilization": min_util
                    },
                    "predicted_imbalance": {
                        "detected": predicted_imbalance_detected,
                        "imbalance_value": predicted_imbalance,
                        "max_worker": predicted_max_worker,
                        "max_utilization": predicted_max_util,
                        "min_worker": predicted_min_worker,
                        "min_utilization": predicted_min_util
                    },
                    "worker_utils": worker_utils,
                    "thresholds": {
                        "high": high_threshold,
                        "low": low_threshold,
                        "imbalance": imbalance_threshold
                    }
                }
            
            return None  # No imbalance detected
            
        except Exception as e:
            logger.error(f"Error detecting load imbalance: {e}")
            return None
        
    def balance_load(self):
        """Balance load by migrating tasks from overloaded to underloaded workers.
        
        This is the standard load balancing method without cost-benefit analysis.
        """
        try:
            # Get imbalance information
            imbalance = self.detect_load_imbalance()
            if not imbalance:
                return  # No imbalance to fix
                
            # Log imbalance detection
            if imbalance["current_imbalance"]["detected"]:
                logger.info(f"Current load imbalance detected: "
                           f"max worker {imbalance['current_imbalance']['max_worker']} at "
                           f"{imbalance['current_imbalance']['max_utilization']:.2f}, "
                           f"min worker {imbalance['current_imbalance']['min_worker']} at "
                           f"{imbalance['current_imbalance']['min_utilization']:.2f}")
            elif imbalance["predicted_imbalance"]["detected"]:
                logger.info(f"Predicted load imbalance detected: "
                           f"max worker {imbalance['predicted_imbalance']['max_worker']} at "
                           f"{imbalance['predicted_imbalance']['max_utilization']:.2f}, "
                           f"min worker {imbalance['predicted_imbalance']['min_worker']} at "
                           f"{imbalance['predicted_imbalance']['min_utilization']:.2f}")
            
            # Determine source and target workers
            if imbalance["current_imbalance"]["detected"]:
                source_worker_id = imbalance["current_imbalance"]["max_worker"]
                target_worker_id = imbalance["current_imbalance"]["min_worker"]
            else:
                source_worker_id = imbalance["predicted_imbalance"]["max_worker"]
                target_worker_id = imbalance["predicted_imbalance"]["min_worker"]
                
            # Find migratable tasks on source worker
            migratable_tasks = self._find_migratable_tasks(source_worker_id)
            
            if not migratable_tasks:
                logger.info(f"No migratable tasks found on worker {source_worker_id}")
                return
                
            # Check if target worker can handle any of the tasks
            for task_id, task_info in migratable_tasks.items():
                if self._can_worker_handle_task(target_worker_id, task_info):
                    # Migrate the task
                    success = self._migrate_task(task_id, source_worker_id, target_worker_id)
                    
                    if success:
                        logger.info(f"Migrated task {task_id} from {source_worker_id} to {target_worker_id}")
                        
                        # Record migration in strategy metrics
                        self.strategy_metrics["migrations_performed"] += 1
                        
                        # Only perform one migration per balancing cycle for now
                        return
                        
            logger.info(f"No suitable tasks found for migration from {source_worker_id} to {target_worker_id}")
            
        except Exception as e:
            logger.error(f"Error balancing load: {e}")
    
    def balance_load_with_cost_benefit(self):
        """Balance load using cost-benefit analysis to make optimal migration decisions."""
        try:
            # Get imbalance information
            imbalance = self.detect_load_imbalance()
            if not imbalance:
                return  # No imbalance to fix
                
            # Get available workers
            available_workers = self.worker_manager.get_available_workers()
            
            # Identify overloaded and underloaded workers
            overloaded_workers = []
            underloaded_workers = []
            
            high_threshold = self.config["utilization_threshold_high"]
            low_threshold = self.config["utilization_threshold_low"]
            
            for worker in available_workers:
                worker_id = worker["worker_id"]
                
                # Skip if worker metrics not available
                if worker_id not in self.worker_metrics:
                    continue
                    
                # Get utilization
                utilization = self.worker_metrics[worker_id].get("overall_utilization", 0)
                
                # Categorize worker
                if utilization >= high_threshold:
                    overloaded_workers.append((worker_id, utilization))
                elif utilization <= low_threshold:
                    underloaded_workers.append((worker_id, utilization))
            
            # Sort overloaded workers by utilization (highest first)
            overloaded_workers.sort(key=lambda w: w[1], reverse=True)
            
            # Sort underloaded workers by utilization (lowest first)
            underloaded_workers.sort(key=lambda w: w[1])
            
            # No migration if no overloaded or underloaded workers
            if not overloaded_workers or not underloaded_workers:
                return
                
            # Get maximum allowed migrations for this cycle
            max_migrations = self.config["max_simultaneous_migrations"]
            migrations_performed = 0
            
            # Create a list to track migration candidates
            migration_candidates = []
            
            # Process each overloaded worker
            for source_worker_id, source_util in overloaded_workers:
                # Find migratable tasks on this worker
                migratable_tasks = self._find_migratable_tasks(source_worker_id)
                
                if not migratable_tasks:
                    continue
                    
                # For each migratable task, evaluate potential target workers
                for task_id, task_info in migratable_tasks.items():
                    for target_worker_id, target_util in underloaded_workers:
                        # Skip if target worker cannot handle the task
                        if not self._can_worker_handle_task(target_worker_id, task_info):
                            continue
                            
                        # Calculate migration cost
                        migration_cost = self._analyze_migration_cost(task_id, task_info)
                        
                        # Calculate migration benefit
                        migration_benefit = self._analyze_migration_benefit(
                            source_worker_id, source_util,
                            target_worker_id, target_util,
                            task_info
                        )
                        
                        # Calculate net benefit
                        net_benefit = migration_benefit - migration_cost
                        
                        # Only consider positive net benefit
                        if net_benefit > 0:
                            migration_candidates.append({
                                "task_id": task_id,
                                "source_worker_id": source_worker_id,
                                "target_worker_id": target_worker_id,
                                "cost": migration_cost,
                                "benefit": migration_benefit,
                                "net_benefit": net_benefit,
                                "task_info": task_info
                            })
            
            # Sort migration candidates by net benefit (highest first)
            migration_candidates.sort(key=lambda m: m["net_benefit"], reverse=True)
            
            # Perform migrations up to the limit
            for candidate in migration_candidates[:max_migrations]:
                task_id = candidate["task_id"]
                source_worker_id = candidate["source_worker_id"]
                target_worker_id = candidate["target_worker_id"]
                
                # Log migration decision
                logger.info(f"Migrating task {task_id} from {source_worker_id} to {target_worker_id} "
                          f"(cost: {candidate['cost']:.2f}, benefit: {candidate['benefit']:.2f}, "
                          f"net benefit: {candidate['net_benefit']:.2f})")
                
                # Perform the migration
                success = self._migrate_task(task_id, source_worker_id, target_worker_id)
                
                if success:
                    migrations_performed += 1
                    
                    # Record migration in strategy metrics
                    self.strategy_metrics["migrations_performed"] += 1
                    
                # Stop if we've reached the limit
                if migrations_performed >= max_migrations:
                    break
            
            if migrations_performed > 0:
                logger.info(f"Performed {migrations_performed} task migrations this cycle")
            else:
                logger.info("No migrations performed this cycle (no positive net benefit)")
                
        except Exception as e:
            logger.error(f"Error in cost-benefit load balancing: {e}")
    
    def _find_migratable_tasks(self, worker_id):
        """Find tasks that can be migrated from a worker.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            Dict mapping task_id to task info for migratable tasks
        """
        migratable_tasks = {}
        
        try:
            # Get running tasks on this worker
            if not self.task_scheduler:
                return {}
                
            with self.task_scheduler.task_lock:
                # Get tasks assigned to this worker
                worker_tasks = {
                    task_id: worker
                    for task_id, worker in self.task_scheduler.running_tasks.items()
                    if worker == worker_id
                }
            
            # No tasks to migrate
            if not worker_tasks:
                return {}
                
            # Get detailed task information
            for task_id in worker_tasks:
                # Get task from database
                if self.db_manager:
                    task = self.db_manager.get_task(task_id)
                    
                    if task:
                        # Check migration eligibility
                        if self._is_task_migratable(task_id, task):
                            migratable_tasks[task_id] = task
                
        except Exception as e:
            logger.error(f"Error finding migratable tasks: {e}")
            
        return migratable_tasks
    
    def _is_task_migratable(self, task_id, task):
        """Check if a task is eligible for migration.
        
        Args:
            task_id: ID of the task
            task: Task information dict
            
        Returns:
            True if task can be migrated, False otherwise
        """
        # Don't migrate tasks that have already been migrated too many times
        if task_id in self.migration_history:
            if len(self.migration_history[task_id]) >= self.config["max_migrations_per_task"]:
                return False
        
        # Don't migrate tasks that are nearly complete
        # This would require task-specific knowledge - we'll add a placeholder here
        # In a real implementation, you would check task progress, ETA, etc.
        
        # Don't migrate tasks with specific flags or properties
        if task.get("non_migratable", False):
            return False
            
        # Task-specific migration checks could be added here
        task_type = task.get("type", "")
        
        # By default, consider tasks migratable
        return True
    
    def _can_worker_handle_task(self, worker_id, task):
        """Check if a worker can handle a specific task.
        
        Args:
            worker_id: ID of the worker
            task: Task information dict
            
        Returns:
            True if worker can handle the task, False otherwise
        """
        # Get worker capabilities
        worker = self.worker_manager.get_worker(worker_id)
        if not worker:
            return False
            
        worker_capabilities = worker.get("capabilities", {})
        
        # Check if worker meets task requirements
        requirements = task.get("requirements", {})
        
        if self.task_scheduler:
            return self.task_scheduler._worker_meets_requirements(worker_capabilities, requirements)
        
        # Basic requirement checking if task scheduler not available
        # Check hardware requirements
        if "hardware" in requirements:
            required_hardware = requirements["hardware"]
            worker_hardware = worker_capabilities.get("hardware_types", [])
            
            if isinstance(required_hardware, list):
                if not any(hw in worker_hardware for hw in required_hardware):
                    return False
            elif isinstance(required_hardware, str):
                if required_hardware not in worker_hardware:
                    return False
        
        return True
    
    def _migrate_task(self, task_id, source_worker_id, target_worker_id):
        """Migrate a task from one worker to another.
        
        Args:
            task_id: ID of the task to migrate
            source_worker_id: ID of the source worker
            target_worker_id: ID of the target worker
            
        Returns:
            True if migration was successful, False otherwise
        """
        try:
            # 1. Cancel the task on the source worker
            # This is simplified - in a real implementation, you would
            # save the task state, progress, etc.
            
            # Record migration attempt
            if task_id not in self.migration_history:
                self.migration_history[task_id] = []
                
            self.migration_history[task_id].append({
                "timestamp": datetime.now(),
                "source": source_worker_id,
                "target": target_worker_id,
                "result": "pending"
            })
            
            # TODO: This is a simplified implementation
            # In a real system, you would:
            # 1. Notify the source worker to pause and save state
            # 2. Transfer state to the coordinator
            # 3. Cancel the task on the source worker
            # 4. Create a new task with saved state on the target worker
            
            # For now, just assume it works if task is in running tasks
            if self.task_scheduler:
                with self.task_scheduler.task_lock:
                    if task_id in self.task_scheduler.running_tasks:
                        if self.task_scheduler.running_tasks[task_id] == source_worker_id:
                            # Remove from running tasks
                            del self.task_scheduler.running_tasks[task_id]
                            
                            # Add to target worker
                            self.task_scheduler.running_tasks[task_id] = target_worker_id
                            
                            # Update migration history
                            self.migration_history[task_id][-1]["result"] = "success"
                            
                            return True
            
            # Migration failed
            self.migration_history[task_id][-1]["result"] = "failed"
            return False
            
        except Exception as e:
            logger.error(f"Error migrating task {task_id}: {e}")
            
            # Update migration history if exists
            if task_id in self.migration_history and self.migration_history[task_id]:
                self.migration_history[task_id][-1]["result"] = "error"
                self.migration_history[task_id][-1]["error"] = str(e)
                
            return False
    
    def _analyze_migration_cost(self, task_id, task_info):
        """Calculate the cost of migrating a task.
        
        Args:
            task_id: ID of the task
            task_info: Task information dict
            
        Returns:
            Migration cost (higher is more costly)
        """
        cost = 1.0  # Base cost
        
        # Cost factors:
        
        # 1. Task runtime - longer running tasks are more expensive to migrate
        start_time = task_info.get("start_time")
        current_time = datetime.now()
        
        if start_time:
            # Convert to datetime if needed
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                
            # Calculate runtime in seconds
            runtime_seconds = (current_time - start_time).total_seconds()
            
            # Scale cost based on runtime (up to 3x for long-running tasks)
            # The longer a task has been running, the more expensive to migrate
            runtime_factor = min(runtime_seconds / 3600, 3.0)  # Max 3x for tasks running >1 hour
            cost *= (1.0 + runtime_factor)
        
        # 2. Task priority - higher priority tasks are more expensive to migrate
        priority = task_info.get("priority", 5)
        priority_factor = (10 - priority) / 5.0  # Convert to 0-2 scale (higher priority = higher cost)
        cost *= (1.0 + priority_factor * 0.5)  # Up to 50% more for high priority
        
        # 3. Task type - some tasks are more expensive to migrate
        task_type = task_info.get("type", "")
        
        # Get historical cost for this task type if available
        type_cost = self.migration_costs.get(task_type, 1.0)
        cost *= type_cost
        
        # 4. Migration history - tasks that have been migrated before are more expensive
        history_factor = 1.0
        if task_id in self.migration_history:
            num_migrations = len(self.migration_history[task_id])
            history_factor = 1.0 + (num_migrations * 0.5)  # Each previous migration adds 50% cost
            
        cost *= history_factor
        
        # Scale the final cost to 0-10 range
        scaled_cost = min(cost, 10.0)
        
        return scaled_cost
    
    def _analyze_migration_benefit(self, source_worker_id, source_util, target_worker_id, target_util, task_info):
        """Calculate the benefit of migrating a task.
        
        Args:
            source_worker_id: ID of the source worker
            source_util: Utilization of the source worker
            target_worker_id: ID of the target worker
            target_util: Utilization of the target worker
            task_info: Task information dict
            
        Returns:
            Migration benefit (higher is more beneficial)
        """
        benefit = 1.0  # Base benefit
        
        # Benefit factors:
        
        # 1. Utilization difference - higher difference = higher benefit
        util_diff = source_util - target_util
        util_factor = util_diff * 5.0  # Scale to 0-5 for typical differences
        benefit += util_factor
        
        # 2. Hardware capability match - better hardware match = higher benefit
        hw_match_benefit = 0.0
        
        # Get task requirements
        requirements = task_info.get("requirements", {})
        task_type = task_info.get("type", "")
        
        # Get hardware types
        source_hw = self.worker_manager.get_worker(source_worker_id).get("capabilities", {}).get("hardware_types", [])
        target_hw = self.worker_manager.get_worker(target_worker_id).get("capabilities", {}).get("hardware_types", [])
        
        source_primary = source_hw[0] if source_hw else "CPU"
        target_primary = target_hw[0] if target_hw else "CPU"
        
        # Look up profiles
        source_profile = self.hardware_profiles.get(source_primary, {})
        target_profile = self.hardware_profiles.get(target_primary, {})
        
        # Check if task type matches hardware specialization
        if task_type in target_profile.get("best_task_types", []):
            # Target hardware is well-suited for this task type
            # Check if it's better than source hardware
            if task_type not in source_profile.get("best_task_types", []):
                hw_match_benefit += 2.0
            
        # Compare performance weights
        source_perf = source_profile.get("performance_weight", 1.0)
        target_perf = target_profile.get("performance_weight", 1.0)
        
        if target_perf > source_perf:
            # Target has better general performance
            hw_match_benefit += (target_perf - source_perf)
        
        # Add hardware match benefit
        benefit += hw_match_benefit
        
        # 3. Resource efficiency improvement
        if self.config["enable_resource_efficiency"]:
            # Energy efficiency improvement
            source_energy = source_profile.get("energy_efficiency", 0.5)
            target_energy = target_profile.get("energy_efficiency", 0.5)
            
            if target_energy > source_energy:
                # Target is more energy efficient
                energy_benefit = (target_energy - source_energy) * 2.0
                benefit += energy_benefit
        
        # 4. Historical worker success with similar tasks
        if task_type in self.worker_type_preferences.get(target_worker_id, {}):
            # Target worker has good history with this task type
            preference_score = self.worker_type_preferences[target_worker_id][task_type]
            history_benefit = preference_score * 2.0  # Scale to 0-2
            benefit += history_benefit
        
        # Scale the final benefit to ensure it's positive
        scaled_benefit = max(benefit, 0.1)
        
        return scaled_benefit
    
    def _record_metrics(self):
        """Record load balancer metrics in the database."""
        if not self.db_manager:
            return
            
        try:
            # Prepare metrics for storage
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_load": 0.0,  # Will be filled in
                "threshold_high": self.config["utilization_threshold_high"],
                "threshold_low": self.config["utilization_threshold_low"],
                "imbalance_score": 0.0,  # Will be filled in
                "migrations_initiated": 0,
                "migrations_successful": 0,
                "prediction_accuracy": 0.0,  # Will be filled in
                "metrics": {}  # Detailed metrics
            }
            
            # System load metrics
            if self.system_load_history:
                latest = self.system_load_history[-1]
                metrics["system_load"] = latest["avg_utilization"]
                metrics["imbalance_score"] = latest["imbalance_score"]
            
            # Migration metrics
            migrations_initiated = self.strategy_metrics.get("migrations_performed", 0)
            metrics["migrations_initiated"] = migrations_initiated
            
            # TODO: Track successful migrations
            successful_migrations = 0
            for task_id, history in self.migration_history.items():
                successful_migrations += sum(1 for m in history if m.get("result") == "success")
            
            metrics["migrations_successful"] = successful_migrations
            
            # Prediction accuracy
            if self.prediction_accuracy_history:
                # Calculate average accuracy of recent predictions
                recent = self.prediction_accuracy_history[-min(10, len(self.prediction_accuracy_history)):]
                avg_accuracy = sum(record["accuracy"] for record in recent) / len(recent)
                metrics["prediction_accuracy"] = avg_accuracy
            
            # Additional detailed metrics
            detailed_metrics = {
                "worker_count": len(self.worker_metrics),
                "active_migrations": len([
                    m for history in self.migration_history.values()
                    for m in history if m.get("result") == "pending"
                ]),
                "thresholds": {
                    "high": self.config["utilization_threshold_high"],
                    "low": self.config["utilization_threshold_low"],
                    "initial_high": 0.85,  # Default value
                    "initial_low": 0.2   # Default value
                },
                "migrations": {
                    "initiated": migrations_initiated,
                    "successful": successful_migrations,
                    "success_rate": successful_migrations / migrations_initiated if migrations_initiated > 0 else 0
                },
                "features": {
                    "dynamic_thresholds": self.config["enable_dynamic_thresholds"],
                    "predictive_balancing": self.config["enable_predictive_balancing"],
                    "cost_benefit_analysis": self.config["enable_cost_benefit_analysis"],
                    "hardware_specific": self.config["enable_hardware_specific_strategies"],
                    "resource_efficiency": self.config["enable_resource_efficiency"]
                },
                "worker_utils": {
                    worker_id: metrics.get("overall_utilization", 0)
                    for worker_id, metrics in self.worker_metrics.items()
                }
            }
            
            # Add prediction data if available
            if self.predicted_load:
                # Get latest prediction for a random worker
                sample_worker_id = next(iter(self.predicted_load.keys()))
                prediction = self.predicted_load[sample_worker_id]
                
                detailed_metrics["prediction"] = {
                    "current_load": prediction["current_utilization"],
                    "predicted_load": prediction["predicted_utilization"],
                    "confidence": prediction["confidence"],
                    "window_minutes": self.config["prediction_window"]
                }
                
            # Add to metrics
            metrics["metrics"] = detailed_metrics
            
            # Store in database
            # In a real implementation, you would have a dedicated table for load balancer metrics
            # For now, just log the metrics
            logger.debug(f"Load balancer metrics: {json.dumps(metrics)}")
            
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")
            
    def select_worker_for_task(self, task: Dict[str, Any], 
                              available_workers: List[Dict[str, Any]]) -> Optional[str]:
    
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