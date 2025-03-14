#!/usr/bin/env python3
"""
Distributed Testing Framework - Advanced Adaptive Load Balancer

This module implements advanced adaptive load balancing for the distributed testing framework.
It monitors worker performance in real-time and redistributes tasks for optimal utilization
using dynamic thresholds, predictive analysis, and hardware-specific strategies.

Usage:
    Import this module in coordinator.py to enable advanced adaptive load balancing.
"""

import asyncio
import json
import logging
import time
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, NamedTuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkloadTrend(NamedTuple):
    """Represents a workload trend with direction and magnitude."""
    direction: str  # 'increasing', 'decreasing', 'stable'
    magnitude: float  # Rate of change (0.0-1.0)
    confidence: float  # Confidence in prediction (0.0-1.0)

class HardwareProfile(NamedTuple):
    """Represents a hardware profile for balancing strategies."""
    hardware_type: str  # 'cpu', 'cuda', 'rocm', etc.
    performance_weight: float  # Relative performance weight
    energy_efficiency: float  # Energy efficiency score (0.0-1.0)
    thermal_efficiency: float  # Thermal efficiency score (0.0-1.0)

class TaskProfile(NamedTuple):
    """Represents a task profile for migration decisions."""
    type: str
    estimated_completion_time: float
    resource_requirements: Dict[str, Any]
    migration_cost: float
    priority: int

class AdaptiveLoadBalancer:
    """Advanced adaptive load balancer for distributed testing framework."""
    
    def __init__(
        self,
        coordinator,
        check_interval: int = 30,
        utilization_threshold_high: float = 0.85,
        utilization_threshold_low: float = 0.2,
        performance_window: int = 5,
        enable_task_migration: bool = True,
        max_simultaneous_migrations: int = 2,
        enable_dynamic_thresholds: bool = True,
        enable_predictive_balancing: bool = True,
        enable_cost_benefit_analysis: bool = True,
        enable_hardware_specific_strategies: bool = True,
        enable_resource_efficiency: bool = True,
        threshold_adjustment_rate: float = 0.05,
        prediction_window: int = 3,
        db_metrics_table: str = "load_balancer_metrics"
    ):
        """
        Initialize the advanced adaptive load balancer.
        
        Args:
            coordinator: Reference to the coordinator instance
            check_interval: Interval for load balance checks in seconds
            utilization_threshold_high: Initial threshold for high utilization (0.0-1.0)
            utilization_threshold_low: Initial threshold for low utilization (0.0-1.0)
            performance_window: Window size for performance measurements in minutes
            enable_task_migration: Whether to enable task migration
            max_simultaneous_migrations: Maximum number of simultaneous task migrations
            enable_dynamic_thresholds: Whether to dynamically adjust thresholds based on system load
            enable_predictive_balancing: Whether to predict future load and proactively balance
            enable_cost_benefit_analysis: Whether to analyze cost vs benefit of migrations
            enable_hardware_specific_strategies: Whether to use hardware-specific balancing strategies
            enable_resource_efficiency: Whether to consider resource efficiency in balancing
            threshold_adjustment_rate: Rate at which thresholds are adjusted (0.0-1.0)
            prediction_window: Window size for load prediction in minutes
            db_metrics_table: Database table name for storing metrics
        """
        self.coordinator = coordinator
        self.check_interval = check_interval
        self.initial_threshold_high = utilization_threshold_high
        self.initial_threshold_low = utilization_threshold_low
        self.utilization_threshold_high = utilization_threshold_high
        self.utilization_threshold_low = utilization_threshold_low
        self.performance_window = performance_window
        self.enable_task_migration = enable_task_migration
        self.max_simultaneous_migrations = max_simultaneous_migrations
        self.enable_dynamic_thresholds = enable_dynamic_thresholds
        self.enable_predictive_balancing = enable_predictive_balancing
        self.enable_cost_benefit_analysis = enable_cost_benefit_analysis
        self.enable_hardware_specific_strategies = enable_hardware_specific_strategies
        self.enable_resource_efficiency = enable_resource_efficiency
        self.threshold_adjustment_rate = threshold_adjustment_rate
        self.prediction_window = prediction_window
        self.db_metrics_table = db_metrics_table
        
        # Performance measurements
        self.worker_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Current migrations
        self.active_migrations: Dict[str, Dict[str, Any]] = {}  # task_id -> migration info
        
        # Migration history
        self.migration_history: List[Dict[str, Any]] = []
        
        # System load history for dynamic thresholds
        self.system_load_history: List[Dict[str, Any]] = []
        
        # Migration cost metrics
        self.migration_cost_history: Dict[str, List[float]] = {}  # task_type -> [costs]
        
        # Hardware profiles for specific strategies
        self.hardware_profiles: Dict[str, HardwareProfile] = {}
        
        # Task type profiles
        self.task_profiles: Dict[str, TaskProfile] = {}
        
        # Previous workload prediction for comparison
        self.previous_workload_prediction: Optional[Dict[str, Any]] = None
        
        # Migration success rate tracking
        self.migration_success_rates: Dict[str, float] = {}  # worker_id -> success_rate
        
        # Initialize database table if needed
        self._init_database_table()
        
        logger.info("Advanced adaptive load balancer initialized")
    
    def _init_database_table(self):
        """Initialize database table for metrics if it doesn't exist."""
        try:
            self.coordinator.db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.db_metrics_table} (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                system_load FLOAT,
                threshold_high FLOAT,
                threshold_low FLOAT,
                imbalance_score FLOAT,
                migrations_initiated INTEGER,
                migrations_successful INTEGER,
                prediction_accuracy FLOAT,
                metrics JSON
            )
            """)
            logger.info(f"Initialized database table: {self.db_metrics_table}")
        except Exception as e:
            logger.error(f"Error initializing database table: {str(e)}")
    
    async def _initialize_hardware_profiles(self):
        """Initialize hardware profiles for specific balancing strategies."""
        # Create base profiles
        self.hardware_profiles = {
            "cpu": HardwareProfile(
                hardware_type="cpu",
                performance_weight=1.0,
                energy_efficiency=0.7,
                thermal_efficiency=0.8
            ),
            "cuda": HardwareProfile(
                hardware_type="cuda",
                performance_weight=3.0,
                energy_efficiency=0.5,
                thermal_efficiency=0.4
            ),
            "rocm": HardwareProfile(
                hardware_type="rocm",
                performance_weight=2.8,
                energy_efficiency=0.5,
                thermal_efficiency=0.4
            ),
            "mps": HardwareProfile(
                hardware_type="mps",
                performance_weight=2.5,
                energy_efficiency=0.6,
                thermal_efficiency=0.6
            ),
            "openvino": HardwareProfile(
                hardware_type="openvino",
                performance_weight=1.8,
                energy_efficiency=0.8,
                thermal_efficiency=0.7
            ),
            "qnn": HardwareProfile(
                hardware_type="qnn",
                performance_weight=1.4,
                energy_efficiency=0.9,
                thermal_efficiency=0.9
            ),
            "webnn": HardwareProfile(
                hardware_type="webnn",
                performance_weight=1.0,
                energy_efficiency=0.7,
                thermal_efficiency=0.8
            ),
            "webgpu": HardwareProfile(
                hardware_type="webgpu",
                performance_weight=1.2,
                energy_efficiency=0.6,
                thermal_efficiency=0.7
            )
        }
        
        # Update with any specific worker hardware profiles from current workers
        for worker_id, worker in self.coordinator.workers.items():
            capabilities = worker.get("capabilities", {})
            hardware_list = capabilities.get("hardware", [])
            
            for hw_type in hardware_list:
                if hw_type in self.hardware_profiles:
                    # Get specific metrics if available
                    gpu_info = capabilities.get("gpu", {})
                    
                    # Customize based on specific hardware
                    if hw_type == "cuda" and isinstance(gpu_info, dict):
                        cuda_compute = float(gpu_info.get("cuda_compute", 0))
                        if cuda_compute >= 8.0:
                            # High-end GPU with high performance but lower efficiency
                            self.hardware_profiles[hw_type] = HardwareProfile(
                                hardware_type=hw_type,
                                performance_weight=4.0,  # Very high performance
                                energy_efficiency=0.4,  # Lower efficiency
                                thermal_efficiency=0.3   # Lower thermal efficiency
                            )
                        elif cuda_compute >= 6.0:
                            # Mid-range GPU
                            self.hardware_profiles[hw_type] = HardwareProfile(
                                hardware_type=hw_type,
                                performance_weight=3.0,
                                energy_efficiency=0.5,
                                thermal_efficiency=0.4
                            )
        
        logger.info(f"Initialized hardware profiles for {len(self.hardware_profiles)} hardware types")
    
    async def start_balancing(self):
        """Start the load balancing loop with enhanced strategies."""
        logger.info("Starting advanced adaptive load balancing")
        
        # Initialize hardware profiles
        await self._initialize_hardware_profiles()
        
        while True:
            try:
                # Update performance metrics
                await self.update_performance_metrics()
                
                # Update dynamic thresholds if enabled
                if self.enable_dynamic_thresholds:
                    await self._update_dynamic_thresholds()
                
                # Predict future load if enabled
                future_load_prediction = None
                if self.enable_predictive_balancing:
                    future_load_prediction = await self._predict_future_load()
                
                # Check for load imbalance (considering predictions if available)
                imbalance_detected = await self.detect_load_imbalance(future_load_prediction)
                
                if imbalance_detected:
                    # Balance load with enhanced strategies
                    await self.balance_load(future_load_prediction)
                
                # Clean up completed migrations
                await self.cleanup_migrations()
                
                # Record metrics in database
                await self._record_metrics()
            except Exception as e:
                logger.error(f"Error in load balancing loop: {str(e)}")
            
            # Sleep until next check
            await asyncio.sleep(self.check_interval)
    
    async def _record_metrics(self):
        """Record load balancer metrics in database for analysis."""
        try:
            # Skip if no history
            if not self.system_load_history:
                return
            
            # Get latest metrics
            now = datetime.now()
            
            # Calculate system-wide metrics
            avg_utilization = 0
            worker_utils = []
            
            for worker_id, history in self.worker_performance_history.items():
                if history:
                    latest = history[-1]
                    worker_utils.append(latest["utilization"])
            
            if worker_utils:
                avg_utilization = sum(worker_utils) / len(worker_utils)
                max_util = max(worker_utils)
                min_util = min(worker_utils)
                imbalance_score = max_util - min_util
            else:
                avg_utilization = 0
                imbalance_score = 0
            
            # Get migration metrics
            migrations_initiated = 0
            migrations_successful = 0
            
            # Count migrations in the last interval
            cutoff_time = now - timedelta(seconds=self.check_interval * 2)
            for migration in self.migration_history:
                try:
                    end_time = datetime.fromisoformat(migration.get("end_time", "1970-01-01T00:00:00"))
                    if end_time >= cutoff_time:
                        migrations_initiated += 1
                        if migration.get("success", False):
                            migrations_successful += 1
                except (ValueError, TypeError):
                    pass
            
            # Calculate prediction accuracy if available
            prediction_accuracy = None
            if hasattr(self, "previous_workload_prediction") and self.previous_workload_prediction:
                if "previous_prediction_accuracy" in self.previous_workload_prediction:
                    prediction_accuracy = self.previous_workload_prediction["previous_prediction_accuracy"]
            
            # Create metrics record
            metrics = {
                "worker_count": len(self.worker_performance_history),
                "active_migrations": len(self.active_migrations),
                "thresholds": {
                    "high": self.utilization_threshold_high,
                    "low": self.utilization_threshold_low,
                    "initial_high": self.initial_threshold_high,
                    "initial_low": self.initial_threshold_low
                },
                "migrations": {
                    "initiated": migrations_initiated,
                    "successful": migrations_successful,
                    "success_rate": migrations_successful / migrations_initiated if migrations_initiated > 0 else None
                },
                "features": {
                    "dynamic_thresholds": self.enable_dynamic_thresholds,
                    "predictive_balancing": self.enable_predictive_balancing,
                    "cost_benefit_analysis": self.enable_cost_benefit_analysis,
                    "hardware_specific": self.enable_hardware_specific_strategies,
                    "resource_efficiency": self.enable_resource_efficiency
                }
            }
            
            # Insert into database
            self.coordinator.db.execute(
                f"""
                INSERT INTO {self.db_metrics_table} (
                    timestamp, system_load, threshold_high, threshold_low,
                    imbalance_score, migrations_initiated, migrations_successful,
                    prediction_accuracy, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    avg_utilization,
                    self.utilization_threshold_high,
                    self.utilization_threshold_low,
                    imbalance_score,
                    migrations_initiated,
                    migrations_successful,
                    prediction_accuracy,
                    json.dumps(metrics)
                )
            )
            
        except Exception as e:
            logger.error(f"Error recording metrics: {str(e)}")
    
    async def update_performance_metrics(self):
        """Update performance metrics for all workers."""
        try:
            now = datetime.now()
            
            # Collect current metrics for all active workers
            for worker_id, worker in self.coordinator.workers.items():
                # Skip offline workers
                if worker.get("status") == "offline":
                    continue
                
                # Get worker hardware metrics
                hardware_metrics = worker.get("hardware_metrics", {})
                
                # Calculate overall utilization
                cpu_percent = hardware_metrics.get("cpu_percent", 0)
                memory_percent = hardware_metrics.get("memory_percent", 0)
                
                # If GPU metrics are available, include them
                gpu_utilization = 0
                if "gpu" in hardware_metrics:
                    gpu_metrics = hardware_metrics["gpu"]
                    if isinstance(gpu_metrics, list) and len(gpu_metrics) > 0:
                        # Average utilization across GPUs
                        gpu_utils = [gpu.get("memory_utilization_percent", 0) for gpu in gpu_metrics]
                        gpu_utilization = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
                    elif isinstance(gpu_metrics, dict):
                        gpu_utilization = gpu_metrics.get("memory_utilization_percent", 0)
                
                # Calculate combined utilization (weighted average)
                # Weight CPU and memory equally, and GPU if it's used
                has_gpu = gpu_utilization > 0
                if has_gpu:
                    utilization = (cpu_percent + memory_percent + gpu_utilization) / 3
                else:
                    utilization = (cpu_percent + memory_percent) / 2
                
                # Normalize to 0.0-1.0 range
                utilization = utilization / 100
                
                # Count running tasks for this worker
                running_tasks = sum(1 for task_id, w_id in self.coordinator.running_tasks.items() if w_id == worker_id)
                
                # Create performance record
                performance = {
                    "timestamp": now.isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "gpu_utilization": gpu_utilization if has_gpu else None,
                    "utilization": utilization,
                    "running_tasks": running_tasks,
                    "has_gpu": has_gpu
                }
                
                # Add to history
                if worker_id not in self.worker_performance_history:
                    self.worker_performance_history[worker_id] = []
                
                self.worker_performance_history[worker_id].append(performance)
                
                # Limit history to performance window (e.g., last 5 minutes)
                cutoff_time = now - timedelta(minutes=self.performance_window)
                self.worker_performance_history[worker_id] = [
                    p for p in self.worker_performance_history[worker_id]
                    if datetime.fromisoformat(p["timestamp"]) >= cutoff_time
                ]
            
            # Log overall system utilization
            await self._log_system_utilization()
            
            # Update system load history for dynamic thresholds
            await self._update_system_load_history()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    async def _update_system_load_history(self):
        """Update system load history for dynamic thresholds."""
        if not self.worker_performance_history:
            return
        
        # Calculate system-wide utilization metrics
        now = datetime.now()
        worker_utils = []
        
        for worker_id, history in self.worker_performance_history.items():
            if history:
                latest = history[-1]
                worker_utils.append(latest["utilization"])
        
        if not worker_utils:
            return
        
        # Calculate statistics
        avg_utilization = sum(worker_utils) / len(worker_utils)
        min_utilization = min(worker_utils)
        max_utilization = max(worker_utils)
        std_dev = statistics.stdev(worker_utils) if len(worker_utils) > 1 else 0
        
        # Create system load record
        record = {
            "timestamp": now.isoformat(),
            "avg_utilization": avg_utilization,
            "min_utilization": min_utilization,
            "max_utilization": max_utilization,
            "std_dev": std_dev,
            "worker_count": len(worker_utils),
            "imbalance": max_utilization - min_utilization
        }
        
        # Add to history
        self.system_load_history.append(record)
        
        # Limit history size
        if len(self.system_load_history) > 100:
            self.system_load_history = self.system_load_history[-100:]
    
    async def _log_system_utilization(self):
        """Log overall system utilization."""
        if not self.worker_performance_history:
            return
        
        # Calculate average utilization across all workers
        total_utilization = 0.0
        total_workers = 0
        
        for worker_id, history in self.worker_performance_history.items():
            if history:
                # Get latest performance record
                latest = history[-1]
                total_utilization += latest["utilization"]
                total_workers += 1
        
        if total_workers > 0:
            avg_utilization = total_utilization / total_workers
            logger.debug(f"System utilization: {avg_utilization:.2%} across {total_workers} workers")
    
    async def _update_dynamic_thresholds(self):
        """
        Update dynamic thresholds based on system-wide load conditions.
        
        This method analyzes the recent system load history to determine
        appropriate threshold adjustments, making the load balancer more
        aggressive during high load periods and more conservative during
        low load periods.
        """
        if not self.system_load_history or len(self.system_load_history) < 5:
            logger.debug("Not enough system load history to update thresholds")
            return
        
        try:
            # Get recent load data (last 5 measurements)
            recent_loads = self.system_load_history[-5:]
            
            # Calculate average system load
            avg_system_load = sum(record.get("avg_utilization", 0) for record in recent_loads) / len(recent_loads)
            
            # Calculate load trend using simple linear regression
            x = list(range(len(recent_loads)))
            y = [record.get("avg_utilization", 0) for record in recent_loads]
            
            # Calculate slope for trend detection
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
            sum_xx = sum(x_i ** 2 for x_i in x)
            
            # Calculate slope (avoiding division by zero)
            if n * sum_xx - sum_x * sum_x == 0:
                trend_slope = 0
            else:
                trend_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            
            # Determine adjustment factors based on trend
            if trend_slope > 0.01:  # Load increasing
                adjustment_factor = self.threshold_adjustment_rate * 1.5
                trend_direction = "increasing"
            elif trend_slope < -0.01:  # Load decreasing
                adjustment_factor = self.threshold_adjustment_rate * 0.5
                trend_direction = "decreasing"
            else:  # Load stable
                adjustment_factor = self.threshold_adjustment_rate
                trend_direction = "stable"
            
            logger.debug(f"Load trend: {trend_direction} (slope: {trend_slope:.4f})")
            
            # Adjust based on current load conditions
            if avg_system_load > 0.75:  # High load
                high_adjust = -adjustment_factor * 1.2  # Lower high threshold (more aggressive balancing)
                low_adjust = adjustment_factor * 0.8    # Raise low threshold
            elif avg_system_load < 0.3:  # Low load
                high_adjust = adjustment_factor * 0.8   # Raise high threshold (more conservative)
                low_adjust = -adjustment_factor * 1.2   # Lower low threshold (consolidate tasks)
            else:  # Normal load
                high_adjust = -adjustment_factor if avg_system_load > 0.5 else adjustment_factor
                low_adjust = adjustment_factor if avg_system_load > 0.5 else -adjustment_factor
            
            # Apply adjustments within boundaries
            new_high = max(0.6, min(0.95, self.utilization_threshold_high + high_adjust))
            new_low = max(0.1, min(0.4, self.utilization_threshold_low + low_adjust))
            
            # Ensure minimum separation between thresholds
            min_separation = 0.3
            if new_high - new_low < min_separation:
                # Adjust to maintain minimum separation
                if high_adjust < low_adjust:  # Moving together
                    new_high = new_low + min_separation
                else:  # Moving apart
                    new_low = new_high - min_separation
            
            # Update thresholds if they changed
            if new_high != self.utilization_threshold_high or new_low != self.utilization_threshold_low:
                logger.info(f"Adjusting thresholds: {self.utilization_threshold_low:.2f}-{self.utilization_threshold_high:.2f} -> {new_low:.2f}-{new_high:.2f}")
                self.utilization_threshold_high = new_high
                self.utilization_threshold_low = new_low
        
        except Exception as e:
            logger.error(f"Error updating dynamic thresholds: {str(e)}")

    async def _predict_future_load(self):
        """
        Predict future system load for proactive load balancing.
        
        This method uses linear regression on recent load history to predict
        future load levels for each worker, enabling the system to proactively
        balance load before imbalances occur.
        
        Returns:
            Dict[str, Any]: Prediction data including predicted worker loads
                and system-wide metrics
        """
        if not self.worker_performance_history:
            return None
        
        try:
            now = datetime.now()
            prediction_window_seconds = self.prediction_window * 60
            future_time = now + timedelta(seconds=prediction_window_seconds)
            
            # Prepare prediction result
            prediction = {
                "timestamp": now.isoformat(),
                "prediction_time": future_time.isoformat(),
                "prediction_window_minutes": self.prediction_window,
                "worker_predictions": {},
                "system_prediction": {
                    "avg_utilization": 0,
                    "min_utilization": 0,
                    "max_utilization": 0,
                    "imbalance_score": 0
                },
                "confidence": 0,
                "previous_prediction_accuracy": None
            }
            
            # Calculate predictions for each worker
            worker_utils_predictions = []
            
            for worker_id, history in self.worker_performance_history.items():
                # Need at least 3 data points for prediction
                if len(history) < 3:
                    continue
                
                # Filter to relevant history (last 10 minutes)
                cutoff_time = now - timedelta(minutes=10)
                relevant_history = []
                
                for record in history:
                    try:
                        record_time = datetime.fromisoformat(record.get("timestamp"))
                        if record_time >= cutoff_time:
                            relevant_history.append(record)
                    except (ValueError, TypeError):
                        pass
                
                if len(relevant_history) < 3:
                    continue
                
                # Create x values (seconds from now)
                x = []
                y = []
                for record in relevant_history:
                    try:
                        record_time = datetime.fromisoformat(record.get("timestamp"))
                        seconds_ago = (now - record_time).total_seconds()
                        x.append(-seconds_ago)  # Negative because it's in the past
                        y.append(record.get("utilization", 0))
                    except (ValueError, TypeError):
                        pass
                
                if len(x) < 3:
                    continue
                
                # Calculate linear regression
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
                sum_xx = sum(x_i ** 2 for x_i in x)
                
                # Calculate slope and intercept (avoiding division by zero)
                if n * sum_xx - sum_x * sum_x == 0:
                    slope = 0
                    intercept = sum_y / n
                else:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                    intercept = (sum_y - slope * sum_x) / n
                
                # Calculate R-squared to measure confidence
                if len(y) <= 1 or all(val == y[0] for val in y):
                    r_squared = 1.0  # Perfect fit for constant data
                else:
                    y_mean = sum(y) / len(y)
                    ss_total = sum((y_i - y_mean) ** 2 for y_i in y)
                    y_predicted = [slope * x_i + intercept for x_i in x]
                    ss_residual = sum((y_i - yp_i) ** 2 for y_i, yp_i in zip(y, y_predicted))
                    
                    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                
                # Predict future value (prediction_window_seconds into the future)
                future_x = prediction_window_seconds
                predicted_utilization = slope * future_x + intercept
                
                # Clamp to valid range
                predicted_utilization = max(0.0, min(1.0, predicted_utilization))
                
                # Calculate prediction confidence based on R-squared and data points
                confidence = r_squared * min(1.0, len(x) / 10)  # Scale by data points up to 10
                
                # Store prediction
                prediction["worker_predictions"][worker_id] = {
                    "current_utilization": y[-1] if y else 0,
                    "predicted_utilization": predicted_utilization,
                    "confidence": confidence,
                    "trend": {
                        "slope": slope,
                        "intercept": intercept,
                        "r_squared": r_squared
                    }
                }
                
                # Add to list for system-wide calculations
                worker_utils_predictions.append(predicted_utilization)
            
            # Calculate system-wide prediction metrics
            if worker_utils_predictions:
                prediction["system_prediction"]["avg_utilization"] = sum(worker_utils_predictions) / len(worker_utils_predictions)
                prediction["system_prediction"]["min_utilization"] = min(worker_utils_predictions)
                prediction["system_prediction"]["max_utilization"] = max(worker_utils_predictions)
                prediction["system_prediction"]["imbalance_score"] = prediction["system_prediction"]["max_utilization"] - prediction["system_prediction"]["min_utilization"]
                
                # Calculate overall confidence as average of worker confidences
                confidences = [pred["confidence"] for pred in prediction["worker_predictions"].values()]
                prediction["confidence"] = sum(confidences) / len(confidences) if confidences else 0
            
            # Calculate accuracy of previous prediction if available
            if self.previous_workload_prediction:
                try:
                    # Get previous prediction details
                    prev_time = datetime.fromisoformat(self.previous_workload_prediction.get("timestamp", ""))
                    prev_window = self.previous_workload_prediction.get("prediction_window_minutes", 0) * 60
                    
                    # Check if enough time has passed to evaluate the prediction
                    if (now - prev_time).total_seconds() >= prev_window:
                        # Compare predicted vs actual values
                        actual_values = []
                        predicted_values = []
                        
                        for worker_id, pred in self.previous_workload_prediction.get("worker_predictions", {}).items():
                            if worker_id in self.worker_performance_history and self.worker_performance_history[worker_id]:
                                # Get current (actual) value
                                actual = self.worker_performance_history[worker_id][-1].get("utilization", 0)
                                predicted = pred.get("predicted_utilization", 0)
                                
                                actual_values.append(actual)
                                predicted_values.append(predicted)
                        
                        if actual_values and predicted_values:
                            # Calculate mean absolute error
                            errors = [abs(a - p) for a, p in zip(actual_values, predicted_values)]
                            mae = sum(errors) / len(errors)
                            
                            # Convert to accuracy percentage (0-1 range)
                            accuracy = max(0, 1.0 - (mae * 2))  # Scale by 2 since utilization is 0-1
                            prediction["previous_prediction_accuracy"] = accuracy
                except Exception as e:
                    logger.error(f"Error calculating prediction accuracy: {str(e)}")
            
            # Store current prediction for future accuracy evaluation
            self.previous_workload_prediction = prediction
            
            logger.info(f"Predicted future system load: {prediction['system_prediction']['avg_utilization']:.2%} (confidence: {prediction['confidence']:.2f})")
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error predicting future load: {str(e)}")
            return None
    
    async def detect_load_imbalance(self, future_load_prediction=None) -> bool:
        """
        Detect if there is a load imbalance in the system.
        
        Args:
            future_load_prediction: Optional prediction of future load
        
        Returns:
            True if imbalance detected, False otherwise
        """
        if not self.worker_performance_history:
            return False
        
        # Get current utilization for all active workers
        worker_utilization = {}
        
        for worker_id, history in self.worker_performance_history.items():
            # Skip workers with no history
            if not history:
                continue
            
            # Skip offline workers
            worker = self.coordinator.workers.get(worker_id)
            if not worker or worker.get("status") == "offline":
                continue
            
            # Get average utilization over the last few records
            recent_history = history[-min(5, len(history)):]
            avg_utilization = sum(p["utilization"] for p in recent_history) / len(recent_history)
            
            worker_utilization[worker_id] = avg_utilization
        
        # Need at least 2 workers to detect imbalance
        if len(worker_utilization) < 2:
            return False
        
        # Find highest and lowest utilization
        max_util_worker = max(worker_utilization.items(), key=lambda x: x[1])
        min_util_worker = min(worker_utilization.items(), key=lambda x: x[1])
        
        max_worker_id, max_util = max_util_worker
        min_worker_id, min_util = min_util_worker
        
        # Check if there's a significant imbalance based on current utilization
        current_imbalance_detected = (max_util > self.utilization_threshold_high and 
                                     min_util < self.utilization_threshold_low and 
                                     max_util - min_util > 0.4)  # At least 40% difference
        
        # If predictive balancing is enabled and we have a prediction, also check future imbalance
        future_imbalance_detected = False
        if self.enable_predictive_balancing and future_load_prediction:
            worker_predictions = future_load_prediction.get("worker_predictions", {})
            prediction_confidence = future_load_prediction.get("confidence", 0)
            
            # Only consider predictions with reasonable confidence
            if prediction_confidence >= 0.7 and len(worker_predictions) >= 2:
                # Get predicted utilities
                predicted_utils = {worker_id: pred.get("predicted_utilization", 0) 
                                  for worker_id, pred in worker_predictions.items()}
                
                # Find highest and lowest predicted utilization
                if predicted_utils:
                    max_pred_worker = max(predicted_utils.items(), key=lambda x: x[1])
                    min_pred_worker = min(predicted_utils.items(), key=lambda x: x[1])
                    
                    max_pred_id, max_pred_util = max_pred_worker
                    min_pred_id, min_pred_util = min_pred_worker
                    
                    # Check for predicted imbalance (use slightly higher threshold)
                    future_imbalance_detected = (max_pred_util > self.utilization_threshold_high + 0.05 and 
                                               min_pred_util < self.utilization_threshold_low - 0.05 and 
                                               max_pred_util - min_pred_util > 0.45)  # Require slightly larger imbalance
                    
                    if future_imbalance_detected:
                        logger.info(f"Predicted future load imbalance: Worker {max_pred_id} at {max_pred_util:.2%}, "
                                   f"Worker {min_pred_id} at {min_pred_util:.2%} (confidence: {prediction_confidence:.2f})")
        
        # Combine current and future detection
        imbalance_detected = current_imbalance_detected or future_imbalance_detected
        
        if current_imbalance_detected:
            logger.info(f"Current load imbalance detected: Worker {max_worker_id} at {max_util:.2%}, "
                       f"Worker {min_worker_id} at {min_util:.2%}")
        
        return imbalance_detected
    
    async def balance_load(self, future_load_prediction=None):
        """Balance load by redistributing tasks."""
        # Skip if task migration is disabled
        if not self.enable_task_migration:
            logger.info("Task migration is disabled, skipping load balancing")
            return
        
        # Skip if too many active migrations
        if len(self.active_migrations) >= self.max_simultaneous_migrations:
            logger.info(f"Too many active migrations ({len(self.active_migrations)}), skipping load balancing")
            return
        
        try:
            # Get worker utilization
            worker_utilization = {}
            
            for worker_id, history in self.worker_performance_history.items():
                # Skip workers with no history
                if not history:
                    continue
                
                # Skip offline workers
                worker = self.coordinator.workers.get(worker_id)
                if not worker or worker.get("status") == "offline":
                    continue
                
                # Get latest utilization
                latest = history[-1]
                worker_utilization[worker_id] = latest["utilization"]
            
            # Identify overloaded and underloaded workers
            overloaded_workers = [
                (worker_id, util) for worker_id, util in worker_utilization.items()
                if util > self.utilization_threshold_high
            ]
            
            underloaded_workers = [
                (worker_id, util) for worker_id, util in worker_utilization.items()
                if util < self.utilization_threshold_low
            ]
            
            # Sort overloaded workers by utilization (highest first)
            overloaded_workers.sort(key=lambda x: x[1], reverse=True)
            
            # Sort underloaded workers by utilization (lowest first)
            underloaded_workers.sort(key=lambda x: x[1])
            
            if not overloaded_workers or not underloaded_workers:
                logger.info("No workers suitable for load balancing")
                return
            
            # Attempt to migrate tasks from overloaded to underloaded workers
            migrations_initiated = 0
            
            for overloaded_id, _ in overloaded_workers:
                # Stop if we've reached maximum simultaneous migrations
                if migrations_initiated >= self.max_simultaneous_migrations:
                    break
                
                # Find tasks that can be migrated from this worker
                migratable_tasks = await self._find_migratable_tasks(overloaded_id)
                
                if not migratable_tasks:
                    logger.info(f"No migratable tasks found for overloaded worker {overloaded_id}")
                    continue
                
                for underloaded_id, _ in underloaded_workers:
                    # Skip if this would exceed max migrations
                    if migrations_initiated >= self.max_simultaneous_migrations:
                        break
                    
                    # Check if we can migrate a task to this worker
                    for task_id, task in migratable_tasks.items():
                        # Skip tasks that are already being migrated
                        if task_id in self.active_migrations:
                            continue
                        
                        # Check if worker can handle this task
                        if await self._can_worker_handle_task(underloaded_id, task):
                            # If cost-benefit analysis is enabled, calculate costs and benefits
                            if self.enable_cost_benefit_analysis:
                                # Calculate migration cost
                                cost = await self._analyze_migration_costs(task, overloaded_id, underloaded_id)
                                
                                # Calculate migration benefit
                                benefit = await self._analyze_migration_benefits(task, overloaded_id, underloaded_id, worker_utilization)
                                
                                # Skip if cost outweighs benefit
                                if cost > benefit:
                                    logger.debug(f"Migration cost ({cost:.2f}) exceeds benefit ({benefit:.2f}) for task {task_id}")
                                    continue
                                
                                logger.debug(f"Migration analysis: cost={cost:.2f}, benefit={benefit:.2f}, net={benefit-cost:.2f} for task {task_id}")
                            
                            # Initiate migration
                            success = await self._migrate_task(task_id, overloaded_id, underloaded_id)
                            
                            if success:
                                migrations_initiated += 1
                                logger.info(f"Initiated migration of task {task_id} from worker {overloaded_id} to {underloaded_id}")
                                
                                # Check if we've reached the limit
                                if migrations_initiated >= self.max_simultaneous_migrations:
                                    break
            
            if migrations_initiated > 0:
                logger.info(f"Initiated {migrations_initiated} task migrations for load balancing")
            else:
                logger.info("No suitable task migrations found for load balancing")
            
        except Exception as e:
            logger.error(f"Error balancing load: {str(e)}")
    
    async def _analyze_migration_costs(self, task, source_worker_id, target_worker_id):
        """
        Analyze the cost of migrating a task.
        
        Args:
            task: Task to be migrated
            source_worker_id: Source worker ID
            target_worker_id: Target worker ID
            
        Returns:
            float: Cost score (0-10 scale, higher is more costly)
        """
        # Base cost for any migration
        base_cost = 2.0
        
        # Task specific factors
        task_type = task.get("type", "unknown")
        task_priority = task.get("priority", 1)
        
        # Calculate running time if available
        running_time_cost = 0
        if "started" in task:
            try:
                started = datetime.fromisoformat(task["started"])
                running_time = (datetime.now() - started).total_seconds()
                
                # Higher cost for longer running tasks
                # Scale up to 3 for tasks running more than 2 minutes
                running_time_cost = min(3.0, running_time / 40)
            except (ValueError, TypeError):
                pass
        
        # Priority cost (higher priority = higher cost)
        priority_cost = min(2.0, task_priority / 2)
        
        # Historical cost for this task type if available
        historical_cost = 0
        if task_type in self.migration_cost_history and self.migration_cost_history[task_type]:
            # Use exponential moving average of past costs
            historical_cost = sum(self.migration_cost_history[task_type]) / len(self.migration_cost_history[task_type])
            # Cap at 2.0
            historical_cost = min(2.0, historical_cost)
        
        # Total cost (0-10 scale)
        total_cost = base_cost + running_time_cost + priority_cost + historical_cost
        
        # Cap at 10
        return min(10.0, total_cost)
    
    async def _analyze_migration_benefits(self, task, source_worker_id, target_worker_id, worker_utilization):
        """
        Analyze the benefit of migrating a task.
        
        Args:
            task: Task to be migrated
            source_worker_id: Source worker ID
            target_worker_id: Target worker ID
            worker_utilization: Dict of worker utilization
            
        Returns:
            float: Benefit score (0-10+ scale, higher is more beneficial)
        """
        # Base benefit for reducing load imbalance
        source_util = worker_utilization.get(source_worker_id, 0)
        target_util = worker_utilization.get(target_worker_id, 0)
        
        # Calculate utilization delta (improvement in imbalance)
        utilization_delta = source_util - target_util
        utilization_benefit = max(0, utilization_delta * 10)  # Scale to 0-10 range
        
        # Hardware match benefit (if hardware specific strategies are enabled)
        hardware_benefit = 0
        if self.enable_hardware_specific_strategies:
            # Get task requirements
            requirements = task.get("requirements", {})
            preferred_hardware = requirements.get("hardware", [])
            
            # Get worker capabilities
            source_worker = self.coordinator.workers.get(source_worker_id, {})
            target_worker = self.coordinator.workers.get(target_worker_id, {})
            
            source_hardware = source_worker.get("capabilities", {}).get("hardware", [])
            target_hardware = target_worker.get("capabilities", {}).get("hardware", [])
            
            # Check if target has better hardware match
            # - If task prefers specific hardware and target has it but source doesn't
            # - If task can use GPU and target has more capable GPU
            
            # Check if task prefers specific hardware
            if preferred_hardware:
                source_match = sum(1 for hw in preferred_hardware if hw in source_hardware)
                target_match = sum(1 for hw in preferred_hardware if hw in target_hardware)
                
                # Calculate hardware match improvement
                hardware_match_improvement = target_match - source_match
                
                # Add benefit for better hardware match
                if hardware_match_improvement > 0:
                    hardware_benefit += hardware_match_improvement * 2.0
            
            # Check for resource efficiency benefit if enabled
            if self.enable_resource_efficiency:
                # Check if power efficiency is important for this task
                if task.get("requirements", {}).get("power_efficient", False):
                    # Check if target worker has more efficient hardware
                    source_efficiency = 0
                    target_efficiency = 0
                    
                    for hw in source_hardware:
                        if hw in self.hardware_profiles:
                            source_efficiency = max(source_efficiency, self.hardware_profiles[hw].energy_efficiency)
                    
                    for hw in target_hardware:
                        if hw in self.hardware_profiles:
                            target_efficiency = max(target_efficiency, self.hardware_profiles[hw].energy_efficiency)
                    
                    # Add benefit for better energy efficiency
                    if target_efficiency > source_efficiency:
                        hardware_benefit += (target_efficiency - source_efficiency) * 3.0
        
        # Total benefit
        total_benefit = utilization_benefit + hardware_benefit
        
        return total_benefit
    
    async def _find_migratable_tasks(self, worker_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Find tasks that can be migrated from a worker.
        
        Args:
            worker_id: Worker ID to find migratable tasks for
            
        Returns:
            Dictionary of migratable tasks (task_id -> task info)
        """
        migratable_tasks = {}
        
        # Find all tasks assigned to this worker
        for task_id, assigned_worker_id in self.coordinator.running_tasks.items():
            if assigned_worker_id != worker_id:
                continue
            
            # Skip if task doesn't exist
            if task_id not in self.coordinator.tasks:
                continue
            
            task = self.coordinator.tasks[task_id]
            
            # Skip tasks that are almost complete
            # This would require task progress reporting, which we might not have
            # For now, skip tasks that have been running for a long time (assumption that they're almost done)
            if "started" in task:
                try:
                    started = datetime.fromisoformat(task["started"])
                    running_time = (datetime.now() - started).total_seconds()
                    
                    # Skip tasks that have been running for more than 5 minutes
                    # This is a simple heuristic and might need adjustment
                    if running_time > 300:  # 5 minutes
                        continue
                except (ValueError, TypeError):
                    pass
            
            # Add task to migratable tasks
            migratable_tasks[task_id] = task
        
        return migratable_tasks
    
    async def _can_worker_handle_task(self, worker_id: str, task: Dict[str, Any]) -> bool:
        """
        Check if a worker can handle a task.
        
        Args:
            worker_id: Worker ID to check
            task: Task to check
            
        Returns:
            True if the worker can handle the task, False otherwise
        """
        # Skip if worker doesn't exist
        if worker_id not in self.coordinator.workers:
            return False
        
        worker = self.coordinator.workers[worker_id]
        
        # Skip inactive workers
        if worker.get("status") != "active":
            return False
        
        # Check task requirements against worker capabilities
        task_requirements = task.get("requirements", {})
        worker_capabilities = worker.get("capabilities", {})
        
        # Check required hardware
        required_hardware = task_requirements.get("hardware", [])
        if required_hardware:
            worker_hardware = worker_capabilities.get("hardware", [])
            if not all(hw in worker_hardware for hw in required_hardware):
                return False
        
        # Check memory requirements
        min_memory_gb = task_requirements.get("min_memory_gb", 0)
        if min_memory_gb > 0:
            worker_memory_gb = worker_capabilities.get("memory", {}).get("total_gb", 0)
            if worker_memory_gb < min_memory_gb:
                return False
        
        # Check CUDA compute capability
        min_cuda_compute = task_requirements.get("min_cuda_compute", 0)
        if min_cuda_compute > 0:
            worker_cuda_compute = float(worker_capabilities.get("gpu", {}).get("cuda_compute", 0))
            if worker_cuda_compute < min_cuda_compute:
                return False
        
        return True
    
    async def _migrate_task(self, task_id: str, source_worker_id: str, target_worker_id: str) -> bool:
        """
        Migrate a task from one worker to another.
        
        Args:
            task_id: Task ID to migrate
            source_worker_id: Source worker ID
            target_worker_id: Target worker ID
            
        Returns:
            True if migration was initiated successfully, False otherwise
        """
        # Skip if either worker doesn't exist
        if source_worker_id not in self.coordinator.workers or target_worker_id not in self.coordinator.workers:
            logger.warning(f"Cannot migrate task {task_id}: Worker does not exist")
            return False
        
        # Skip if the task doesn't exist
        if task_id not in self.coordinator.tasks:
            logger.warning(f"Cannot migrate task {task_id}: Task does not exist")
            return False
        
        # Get task
        task = self.coordinator.tasks[task_id]
        
        try:
            # Step 1: Mark task as "migrating"
            task["status"] = "migrating"
            task["migration"] = {
                "source_worker_id": source_worker_id,
                "target_worker_id": target_worker_id,
                "start_time": datetime.now().isoformat()
            }
            
            # Step 2: Cancel task on source worker
            if source_worker_id in self.coordinator.worker_connections:
                try:
                    await self.coordinator.worker_connections[source_worker_id].send_json({
                        "type": "cancel_task",
                        "task_id": task_id,
                        "reason": "migration"
                    })
                    logger.info(f"Sent cancellation request for task {task_id} to worker {source_worker_id}")
                except Exception as e:
                    logger.error(f"Error sending cancellation request to worker {source_worker_id}: {str(e)}")
                    return False
            
            # Step 3: Add migration to active migrations
            self.active_migrations[task_id] = {
                "task_id": task_id,
                "source_worker_id": source_worker_id,
                "target_worker_id": target_worker_id,
                "start_time": datetime.now().isoformat(),
                "status": "cancelling"
            }
            
            # Migration initiated successfully
            return True
            
        except Exception as e:
            logger.error(f"Error initiating migration for task {task_id}: {str(e)}")
            return False
    
    async def handle_task_cancelled_for_migration(self, task_id: str, source_worker_id: str):
        """
        Handle task cancellation for migration.
        
        Args:
            task_id: Task ID
            source_worker_id: Source worker ID
        """
        # Skip if this task is not being migrated
        if task_id not in self.active_migrations:
            logger.warning(f"Task {task_id} cancellation received but not in active migrations")
            return
        
        # Get migration info
        migration = self.active_migrations[task_id]
        
        # Skip if source worker doesn't match
        if migration["source_worker_id"] != source_worker_id:
            logger.warning(f"Task {task_id} cancellation received from unexpected worker {source_worker_id}")
            return
        
        try:
            # Update migration status
            migration["status"] = "assigning"
            migration["cancel_time"] = datetime.now().isoformat()
            
            # Get task
            if task_id not in self.coordinator.tasks:
                logger.warning(f"Task {task_id} not found for migration")
                return
            
            task = self.coordinator.tasks[task_id]
            
            # Update task status to pending (so it can be assigned again)
            task["status"] = "pending"
            if "started" in task:
                del task["started"]
            if "worker_id" in task:
                del task["worker_id"]
            
            # Add to pending tasks
            self.coordinator.pending_tasks.add(task_id)
            
            # Remove from running tasks
            if task_id in self.coordinator.running_tasks:
                del self.coordinator.running_tasks[task_id]
            
            # Try to assign task to target worker
            target_worker_id = migration["target_worker_id"]
            
            # Add this back to the task so it can be used by the task scheduler
            task["preferred_worker_id"] = target_worker_id
            
            # Update database
            self.coordinator.db.execute(
                """
                UPDATE distributed_tasks
                SET status = 'pending', worker_id = NULL, start_time = NULL
                WHERE task_id = ?
                """,
                (task_id,)
            )
            
            logger.info(f"Task {task_id} cancelled on source worker, now pending reassignment to target worker")
            
            # Assign pending tasks (will include our migrated task)
            await self.coordinator._assign_pending_tasks()
            
            # Update migration status
            migration["status"] = "assigned"
            migration["assign_time"] = datetime.now().isoformat()
            
            # Check if assignment was successful
            if task_id in self.coordinator.running_tasks:
                actual_worker_id = self.coordinator.running_tasks[task_id]
                migration["actual_worker_id"] = actual_worker_id
                
                # Check if assigned to expected worker
                if actual_worker_id == target_worker_id:
                    logger.info(f"Task {task_id} successfully migrated to target worker {target_worker_id}")
                else:
                    logger.warning(f"Task {task_id} assigned to different worker {actual_worker_id} than target {target_worker_id}")
            else:
                logger.warning(f"Task {task_id} not assigned to any worker after migration")
                migration["status"] = "failed"
            
        except Exception as e:
            logger.error(f"Error handling task {task_id} cancellation for migration: {str(e)}")
            
            # Mark migration as failed
            if task_id in self.active_migrations:
                self.active_migrations[task_id]["status"] = "failed"
                self.active_migrations[task_id]["error"] = str(e)
    
    async def cleanup_migrations(self):
        """Clean up completed migrations."""
        now = datetime.now()
        
        # Identify completed migrations
        completed_migrations = []
        
        for task_id, migration in list(self.active_migrations.items()):
            # Skip recent migrations (less than 60 seconds old)
            try:
                start_time = datetime.fromisoformat(migration["start_time"])
                age = (now - start_time).total_seconds()
                
                if age < 60:
                    continue
            except (ValueError, TypeError, KeyError):
                pass
            
            # Check if migration is complete
            status = migration.get("status", "")
            
            if status in ["assigned", "failed"]:
                # Migration is complete, move to history
                migration["end_time"] = now.isoformat()
                self.migration_history.append(migration)
                completed_migrations.append(task_id)
            
            # Also clean up very old migrations (more than 10 minutes old)
            try:
                start_time = datetime.fromisoformat(migration["start_time"])
                age = (now - start_time).total_seconds()
                
                if age > 600:  # 10 minutes
                    logger.warning(f"Cleaning up stale migration for task {task_id} (age: {age:.1f}s)")
                    migration["end_time"] = now.isoformat()
                    migration["status"] = "timeout"
                    self.migration_history.append(migration)
                    completed_migrations.append(task_id)
            except (ValueError, TypeError, KeyError):
                pass
        
        # Remove completed migrations from active migrations
        for task_id in completed_migrations:
            if task_id in self.active_migrations:
                del self.active_migrations[task_id]
        
        # Limit migration history to last 100 entries
        if len(self.migration_history) > 100:
            self.migration_history = self.migration_history[-100:]
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the load balancer.
        
        Returns:
            Statistics about the load balancer
        """
        now = datetime.now()
        
        # Calculate system-wide utilization
        total_utilization = 0.0
        worker_utils = []
        
        for worker_id, history in self.worker_performance_history.items():
            if history:
                # Get latest performance record
                latest = history[-1]
                util = latest["utilization"]
                total_utilization += util
                worker_utils.append(util)
        
        # Calculate stats
        avg_utilization = total_utilization / len(worker_utils) if worker_utils else 0
        min_utilization = min(worker_utils) if worker_utils else 0
        max_utilization = max(worker_utils) if worker_utils else 0
        utilization_stdev = (sum((u - avg_utilization) ** 2 for u in worker_utils) / len(worker_utils)) ** 0.5 if worker_utils else 0
        
        # Count migrations in different time windows
        migrations_last_hour = 0
        migrations_last_day = 0
        
        for migration in self.migration_history:
            try:
                end_time = datetime.fromisoformat(migration.get("end_time", "1970-01-01T00:00:00"))
                age = (now - end_time).total_seconds()
                
                if age <= 3600:  # 1 hour
                    migrations_last_hour += 1
                
                if age <= 86400:  # 1 day
                    migrations_last_day += 1
            except (ValueError, TypeError):
                pass
        
        # Build stats
        stats = {
            "system_utilization": {
                "average": avg_utilization,
                "min": min_utilization,
                "max": max_utilization,
                "std_dev": utilization_stdev,
                "imbalance_score": max_utilization - min_utilization if worker_utils else 0,
            },
            "active_workers": len(worker_utils),
            "migrations": {
                "active": len(self.active_migrations),
                "last_hour": migrations_last_hour,
                "last_day": migrations_last_day,
                "total_history": len(self.migration_history),
            },
            "config": {
                "check_interval": self.check_interval,
                "utilization_threshold_high": self.utilization_threshold_high,
                "utilization_threshold_low": self.utilization_threshold_low,
                "enable_task_migration": self.enable_task_migration,
                "max_simultaneous_migrations": self.max_simultaneous_migrations,
            }
        }
        
        return stats