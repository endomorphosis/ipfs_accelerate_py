#!/usr/bin/env python3
"""
Distributed Testing Framework - Performance Trend Analyzer

This module implements the performance trend analysis functionality for the
distributed testing framework. It's responsible for:

- Long-term performance history tracking
- Statistical analysis of performance trends
- Anomaly detection for performance degradation
- Prediction of future performance
- Visualization of performance metrics over time

Usage:
    This module can be used to analyze and visualize performance trends for
    workers and task types, as well as detect anomalies in performance.
"""

import os
import sys
import json
import time
import logging
import threading
import statistics
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("performance_trend_analyzer")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class PerformanceTrendAnalyzer:
    """Performance trend analysis system for the distributed testing framework."""
    
    def __init__(self, db_manager=None, task_scheduler=None):
        """Initialize the performance trend analyzer.
        
        Args:
            db_manager: Database manager for data access
            task_scheduler: Task scheduler for current data
        """
        self.db_manager = db_manager
        self.task_scheduler = task_scheduler
        
        # Time series data
        self.time_series = defaultdict(dict)  # {worker_id: {metric: [(timestamp, value)]}}
        self.task_time_series = defaultdict(dict)  # {task_type: {metric: [(timestamp, value)]}}
        
        # Performance baselines
        self.worker_baselines = {}  # {worker_id: {metric: {mean, stdev, threshold}}}
        self.task_baselines = {}  # {task_type: {metric: {mean, stdev, threshold}}}
        
        # Anomaly detection
        self.anomalies = defaultdict(list)  # {worker_id: [anomaly_records]}
        self.task_anomalies = defaultdict(list)  # {task_type: [anomaly_records]}
        
        # Trend analysis results
        self.trends = defaultdict(dict)  # {worker_id: {metric: {slope, p_value, forecast}}}
        self.task_trends = defaultdict(dict)  # {task_type: {metric: {slope, p_value, forecast}}}
        
        # Configuration
        self.config = {
            "history_days": 30,  # Days of history to keep
            "anomaly_threshold": 3.0,  # Z-score threshold for anomalies
            "trend_significance_threshold": 0.05,  # P-value threshold for significant trends
            "forecast_days": 7,  # Days to forecast
            "min_data_points": 5,  # Minimum data points for trend analysis
            "update_interval": 3600,  # Seconds between automatic updates
            "visualization_enabled": True,  # Enable visualization
            "visualization_format": "png",  # Visualization format
            "visualization_path": "./performance_visualizations",  # Path for visualization files
            "database_enabled": True,  # Enable database storage
            "metrics": ["execution_time", "success_rate", "throughput", "memory_usage", "cpu_usage"],
            "aggregate_metrics": True,  # Aggregate metrics for hourly/daily views
        }
        
        # Update thread
        self.update_thread = None
        self.update_stop_event = threading.Event()
        
        # Create visualization directory if enabled
        if self.config["visualization_enabled"]:
            os.makedirs(self.config["visualization_path"], exist_ok=True)
        
        logger.info("Performance trend analyzer initialized")
        
    def configure(self, config_updates: Dict[str, Any]):
        """Update the analyzer configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.config.update(config_updates)
        logger.info(f"Performance trend analyzer configuration updated: {config_updates}")
        
        # Create visualization directory if newly enabled
        if self.config["visualization_enabled"]:
            os.makedirs(self.config["visualization_path"], exist_ok=True)
        
    def start(self):
        """Start the performance trend analyzer."""
        # Load historical data
        self._load_historical_data()
        
        # Compute baselines
        self._compute_baselines()
        
        # Start update thread
        self.update_stop_event.clear()
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
        logger.info("Performance trend analyzer started")
        
    def stop(self):
        """Stop the performance trend analyzer."""
        # Stop update thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_stop_event.set()
            self.update_thread.join(timeout=5.0)
            if self.update_thread.is_alive():
                logger.warning("Update thread did not stop gracefully")
                
        logger.info("Performance trend analyzer stopped")
        
    def _update_loop(self):
        """Update thread function."""
        while not self.update_stop_event.is_set():
            try:
                # Update time series data
                self._update_time_series()
                
                # Compute baselines
                self._compute_baselines()
                
                # Detect anomalies
                self._detect_anomalies()
                
                # Analyze trends
                self._analyze_trends()
                
                # Generate visualizations if enabled
                if self.config["visualization_enabled"]:
                    self._generate_visualizations()
                    
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                
            # Wait for next update interval
            self.update_stop_event.wait(self.config["update_interval"])
            
    def _load_historical_data(self):
        """Load historical performance data from database."""
        if not self.db_manager or not self.config["database_enabled"]:
            logger.info("No database manager available or database disabled, skipping historical data loading")
            return
            
        try:
            # Load worker performance history
            history_days = self.config["history_days"]
            start_date = datetime.now() - timedelta(days=history_days)
            
            # Get worker performance history
            worker_history = self.db_manager.get_worker_performance_history(start_date)
            
            # Process worker history
            for record in worker_history:
                worker_id = record.get("worker_id")
                timestamp = record.get("timestamp")
                metrics = record.get("metrics", {})
                
                if not worker_id or not timestamp:
                    continue
                    
                # Update time series data
                for metric, value in metrics.items():
                    if metric not in self.time_series[worker_id]:
                        self.time_series[worker_id][metric] = []
                        
                    self.time_series[worker_id][metric].append((timestamp, value))
                    
            # Get task type performance history
            task_history = self.db_manager.get_task_type_performance_history(start_date)
            
            # Process task history
            for record in task_history:
                task_type = record.get("task_type")
                timestamp = record.get("timestamp")
                metrics = record.get("metrics", {})
                
                if not task_type or not timestamp:
                    continue
                    
                # Update time series data
                for metric, value in metrics.items():
                    if metric not in self.task_time_series[task_type]:
                        self.task_time_series[task_type][metric] = []
                        
                    self.task_time_series[task_type][metric].append((timestamp, value))
                    
            logger.info(
                f"Loaded historical data: {len(worker_history)} worker records, "
                f"{len(task_history)} task type records"
            )
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            
    def _update_time_series(self):
        """Update time series data with current metrics."""
        # Get current metrics from task scheduler
        if not self.task_scheduler:
            return
            
        # Get worker performance data
        worker_performance = {}
        if hasattr(self.task_scheduler, "worker_performance"):
            worker_performance = self.task_scheduler.worker_performance
            
        # Get task statistics
        task_stats = {}
        if hasattr(self.task_scheduler, "task_stats"):
            task_stats = self.task_scheduler.task_stats
            
        # Current timestamp
        current_time = datetime.now()
        
        # Update worker time series
        for worker_id, perf in worker_performance.items():
            # Skip workers with no data
            if not perf:
                continue
                
            # Extract metrics
            metrics = {
                "success_rate": perf.get("success_count", 0) / max(1, perf.get("task_count", 1)),
                "avg_execution_time": perf.get("total_execution_time", 0) / max(1, perf.get("task_count", 1)),
                "task_count": perf.get("task_count", 0),
                "cpu_percent": perf.get("cpu_percent", 0),
                "available_memory_gb": perf.get("available_memory_gb", 0),
            }
            
            # Add GPU metrics if available
            if "gpu_memory_available_mb" in perf:
                metrics["gpu_memory_available_mb"] = perf.get("gpu_memory_available_mb", 0)
                
            # Update time series
            for metric, value in metrics.items():
                if metric not in self.time_series[worker_id]:
                    self.time_series[worker_id][metric] = []
                    
                self.time_series[worker_id][metric].append((current_time, value))
                
            # Add task type-specific metrics
            for task_type, type_perf in perf.get("task_types", {}).items():
                type_metrics = {
                    "success_rate": type_perf.get("success_rate", 0),
                    "avg_execution_time": type_perf.get("avg_execution_time", 0),
                    "task_count": type_perf.get("task_count", 0),
                }
                
                # Create worker-task type compound key
                compound_key = f"{worker_id}_{task_type}"
                
                # Update time series
                for metric, value in type_metrics.items():
                    metric_key = f"{task_type}_{metric}"
                    if metric_key not in self.time_series[worker_id]:
                        self.time_series[worker_id][metric_key] = []
                        
                    self.time_series[worker_id][metric_key].append((current_time, value))
                    
        # Update task type time series
        for task_type, stats in task_stats.items():
            # Skip task types with no data
            if not stats:
                continue
                
            # Extract metrics
            metrics = {
                "success_rate": stats.get("success_rate", 0),
                "avg_execution_time": stats.get("avg_execution_time", 0),
                "task_count": stats.get("task_count", 0),
            }
            
            # Update time series
            for metric, value in metrics.items():
                if metric not in self.task_time_series[task_type]:
                    self.task_time_series[task_type][metric] = []
                    
                self.task_time_series[task_type][metric].append((current_time, value))
                
        # Store time series in database if enabled
        if self.db_manager and self.config["database_enabled"]:
            try:
                # Store worker performance metrics
                for worker_id, metrics in worker_performance.items():
                    if not metrics:
                        continue
                        
                    # Prepare metrics for storage
                    stored_metrics = {
                        "success_rate": metrics.get("success_count", 0) / max(1, metrics.get("task_count", 1)),
                        "avg_execution_time": metrics.get("total_execution_time", 0) / max(1, metrics.get("task_count", 1)),
                        "task_count": metrics.get("task_count", 0),
                        "cpu_percent": metrics.get("cpu_percent", 0),
                        "available_memory_gb": metrics.get("available_memory_gb", 0),
                    }
                    
                    # Add GPU metrics if available
                    if "gpu_memory_available_mb" in metrics:
                        stored_metrics["gpu_memory_available_mb"] = metrics.get("gpu_memory_available_mb", 0)
                        
                    # Store metrics
                    self.db_manager.add_worker_performance_metrics(
                        worker_id, current_time, stored_metrics
                    )
                    
                # Store task type performance metrics
                for task_type, stats in task_stats.items():
                    if not stats:
                        continue
                        
                    # Prepare metrics for storage
                    stored_metrics = {
                        "success_rate": stats.get("success_rate", 0),
                        "avg_execution_time": stats.get("avg_execution_time", 0),
                        "task_count": stats.get("task_count", 0),
                    }
                    
                    # Store metrics
                    self.db_manager.add_task_type_performance_metrics(
                        task_type, current_time, stored_metrics
                    )
                    
            except Exception as e:
                logger.error(f"Error storing time series data: {e}")
                
        # Prune old data
        self._prune_time_series()
            
    def _prune_time_series(self):
        """Prune old time series data."""
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(days=self.config["history_days"])
        
        # Prune worker time series
        for worker_id in list(self.time_series.keys()):
            for metric in list(self.time_series[worker_id].keys()):
                # Filter out old data points
                self.time_series[worker_id][metric] = [
                    (timestamp, value) for timestamp, value in self.time_series[worker_id][metric]
                    if timestamp >= cutoff_time
                ]
                
                # Remove empty metrics
                if not self.time_series[worker_id][metric]:
                    del self.time_series[worker_id][metric]
                    
            # Remove empty workers
            if not self.time_series[worker_id]:
                del self.time_series[worker_id]
                
        # Prune task type time series
        for task_type in list(self.task_time_series.keys()):
            for metric in list(self.task_time_series[task_type].keys()):
                # Filter out old data points
                self.task_time_series[task_type][metric] = [
                    (timestamp, value) for timestamp, value in self.task_time_series[task_type][metric]
                    if timestamp >= cutoff_time
                ]
                
                # Remove empty metrics
                if not self.task_time_series[task_type][metric]:
                    del self.task_time_series[task_type][metric]
                    
            # Remove empty task types
            if not self.task_time_series[task_type]:
                del self.task_time_series[task_type]
                
    def _compute_baselines(self):
        """Compute performance baselines for anomaly detection."""
        # Compute worker baselines
        for worker_id, metrics in self.time_series.items():
            if worker_id not in self.worker_baselines:
                self.worker_baselines[worker_id] = {}
                
            for metric, data_points in metrics.items():
                # Skip metrics with too few data points
                if len(data_points) < self.config["min_data_points"]:
                    continue
                    
                # Extract values
                values = [value for _, value in data_points]
                
                # Compute statistics
                try:
                    mean = statistics.mean(values)
                    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
                    
                    # Store baseline
                    self.worker_baselines[worker_id][metric] = {
                        "mean": mean,
                        "stdev": stdev,
                        "threshold": self.config["anomaly_threshold"],
                        "updated_at": datetime.now(),
                        "n_samples": len(values)
                    }
                except Exception as e:
                    logger.warning(f"Error computing baseline for worker {worker_id}, metric {metric}: {e}")
                    
        # Compute task type baselines
        for task_type, metrics in self.task_time_series.items():
            if task_type not in self.task_baselines:
                self.task_baselines[task_type] = {}
                
            for metric, data_points in metrics.items():
                # Skip metrics with too few data points
                if len(data_points) < self.config["min_data_points"]:
                    continue
                    
                # Extract values
                values = [value for _, value in data_points]
                
                # Compute statistics
                try:
                    mean = statistics.mean(values)
                    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
                    
                    # Store baseline
                    self.task_baselines[task_type][metric] = {
                        "mean": mean,
                        "stdev": stdev,
                        "threshold": self.config["anomaly_threshold"],
                        "updated_at": datetime.now(),
                        "n_samples": len(values)
                    }
                except Exception as e:
                    logger.warning(f"Error computing baseline for task type {task_type}, metric {metric}: {e}")
                    
    def _detect_anomalies(self):
        """Detect anomalies in performance metrics."""
        # Detect worker anomalies
        for worker_id, metrics in self.time_series.items():
            # Skip workers with no baseline
            if worker_id not in self.worker_baselines:
                continue
                
            for metric, data_points in metrics.items():
                # Skip metrics with no baseline
                if metric not in self.worker_baselines[worker_id]:
                    continue
                    
                # Skip metrics with too few data points
                if len(data_points) < self.config["min_data_points"]:
                    continue
                    
                # Get baseline
                baseline = self.worker_baselines[worker_id][metric]
                mean = baseline["mean"]
                stdev = baseline["stdev"]
                threshold = baseline["threshold"]
                
                # Skip metrics with zero standard deviation
                if stdev <= 0:
                    continue
                    
                # Check recent data points for anomalies
                recent_points = data_points[-self.config["min_data_points"]:]
                
                for timestamp, value in recent_points:
                    # Calculate z-score
                    z_score = (value - mean) / stdev
                    
                    # Check if anomaly
                    if abs(z_score) > threshold:
                        # Create anomaly record
                        anomaly = {
                            "worker_id": worker_id,
                            "metric": metric,
                            "timestamp": timestamp,
                            "value": value,
                            "mean": mean,
                            "stdev": stdev,
                            "z_score": z_score,
                            "is_high": z_score > 0,
                            "detected_at": datetime.now()
                        }
                        
                        # Check if already detected
                        is_new = True
                        for existing in self.anomalies[worker_id]:
                            # Check if same timestamp and metric
                            if (
                                existing["timestamp"] == timestamp and 
                                existing["metric"] == metric
                            ):
                                is_new = False
                                break
                                
                        # Add if new
                        if is_new:
                            self.anomalies[worker_id].append(anomaly)
                            
                            # Log anomaly
                            logger.warning(
                                f"Detected worker anomaly: worker={worker_id}, metric={metric}, "
                                f"value={value:.2f}, z_score={z_score:.2f}"
                            )
                            
                            # Store in database if enabled
                            if self.db_manager and self.config["database_enabled"]:
                                try:
                                    self.db_manager.add_performance_anomaly(
                                        worker_id=worker_id,
                                        entity_type="worker",
                                        metric=metric,
                                        timestamp=timestamp,
                                        value=value,
                                        baseline_mean=mean,
                                        baseline_stdev=stdev,
                                        z_score=z_score,
                                        is_high=z_score > 0
                                    )
                                except Exception as e:
                                    logger.error(f"Error storing anomaly: {e}")
                                    
        # Detect task type anomalies
        for task_type, metrics in self.task_time_series.items():
            # Skip task types with no baseline
            if task_type not in self.task_baselines:
                continue
                
            for metric, data_points in metrics.items():
                # Skip metrics with no baseline
                if metric not in self.task_baselines[task_type]:
                    continue
                    
                # Skip metrics with too few data points
                if len(data_points) < self.config["min_data_points"]:
                    continue
                    
                # Get baseline
                baseline = self.task_baselines[task_type][metric]
                mean = baseline["mean"]
                stdev = baseline["stdev"]
                threshold = baseline["threshold"]
                
                # Skip metrics with zero standard deviation
                if stdev <= 0:
                    continue
                    
                # Check recent data points for anomalies
                recent_points = data_points[-self.config["min_data_points"]:]
                
                for timestamp, value in recent_points:
                    # Calculate z-score
                    z_score = (value - mean) / stdev
                    
                    # Check if anomaly
                    if abs(z_score) > threshold:
                        # Create anomaly record
                        anomaly = {
                            "task_type": task_type,
                            "metric": metric,
                            "timestamp": timestamp,
                            "value": value,
                            "mean": mean,
                            "stdev": stdev,
                            "z_score": z_score,
                            "is_high": z_score > 0,
                            "detected_at": datetime.now()
                        }
                        
                        # Check if already detected
                        is_new = True
                        for existing in self.task_anomalies[task_type]:
                            # Check if same timestamp and metric
                            if (
                                existing["timestamp"] == timestamp and 
                                existing["metric"] == metric
                            ):
                                is_new = False
                                break
                                
                        # Add if new
                        if is_new:
                            self.task_anomalies[task_type].append(anomaly)
                            
                            # Log anomaly
                            logger.warning(
                                f"Detected task type anomaly: type={task_type}, metric={metric}, "
                                f"value={value:.2f}, z_score={z_score:.2f}"
                            )
                            
                            # Store in database if enabled
                            if self.db_manager and self.config["database_enabled"]:
                                try:
                                    self.db_manager.add_performance_anomaly(
                                        worker_id=None,
                                        entity_type="task_type",
                                        entity_id=task_type,
                                        metric=metric,
                                        timestamp=timestamp,
                                        value=value,
                                        baseline_mean=mean,
                                        baseline_stdev=stdev,
                                        z_score=z_score,
                                        is_high=z_score > 0
                                    )
                                except Exception as e:
                                    logger.error(f"Error storing anomaly: {e}")
                                    
    def _analyze_trends(self):
        """Analyze trends in performance metrics."""
        # Analyze worker trends
        for worker_id, metrics in self.time_series.items():
            if worker_id not in self.trends:
                self.trends[worker_id] = {}
                
            for metric, data_points in metrics.items():
                # Skip metrics with too few data points
                if len(data_points) < self.config["min_data_points"]:
                    continue
                    
                # Extract timestamps and values
                timestamps = np.array([(ts - datetime(1970, 1, 1)).total_seconds() 
                              for ts, _ in data_points])
                values = np.array([value for _, value in data_points])
                
                # Normalize timestamps to days from earliest
                timestamps = (timestamps - timestamps.min()) / (24 * 3600)
                
                # Linear regression
                try:
                    # Check if we have enough variance in the data
                    if np.var(values) <= 0:
                        continue
                        
                    # Reshape for sklearn
                    X = timestamps.reshape(-1, 1)
                    
                    # Fit linear model
                    model = LinearRegression()
                    model.fit(X, values)
                    
                    # Get slope
                    slope = model.coef_[0]
                    
                    # Calculate p-value
                    _, _, r_value, p_value, _ = stats.linregress(timestamps, values)
                    
                    # Forecast future values
                    forecast_days = np.array(range(1, self.config["forecast_days"] + 1))
                    forecast_x = np.max(timestamps) + forecast_days
                    forecast_x = forecast_x.reshape(-1, 1)
                    forecast_values = model.predict(forecast_x)
                    
                    # Store trend analysis
                    self.trends[worker_id][metric] = {
                        "slope": slope,
                        "p_value": p_value,
                        "r_squared": r_value ** 2,
                        "is_significant": p_value < self.config["trend_significance_threshold"],
                        "forecast": list(zip(forecast_days, forecast_values)),
                        "direction": "increasing" if slope > 0 else "decreasing",
                        "updated_at": datetime.now()
                    }
                    
                    # Log significant trends
                    if p_value < self.config["trend_significance_threshold"]:
                        direction = "increasing" if slope > 0 else "decreasing"
                        logger.info(
                            f"Detected significant {direction} trend for worker={worker_id}, "
                            f"metric={metric}, slope={slope:.4f}, p={p_value:.4f}"
                        )
                        
                        # Store in database if enabled
                        if self.db_manager and self.config["database_enabled"]:
                            try:
                                self.db_manager.add_performance_trend(
                                    worker_id=worker_id,
                                    entity_type="worker",
                                    metric=metric,
                                    slope=slope,
                                    p_value=p_value,
                                    r_squared=r_value ** 2,
                                    is_significant=p_value < self.config["trend_significance_threshold"],
                                    direction=direction,
                                    forecast_values=list(forecast_values)
                                )
                            except Exception as e:
                                logger.error(f"Error storing trend: {e}")
                                
                except Exception as e:
                    logger.warning(f"Error analyzing trend for worker {worker_id}, metric {metric}: {e}")
                    
        # Analyze task type trends
        for task_type, metrics in self.task_time_series.items():
            if task_type not in self.task_trends:
                self.task_trends[task_type] = {}
                
            for metric, data_points in metrics.items():
                # Skip metrics with too few data points
                if len(data_points) < self.config["min_data_points"]:
                    continue
                    
                # Extract timestamps and values
                timestamps = np.array([(ts - datetime(1970, 1, 1)).total_seconds() 
                              for ts, _ in data_points])
                values = np.array([value for _, value in data_points])
                
                # Normalize timestamps to days from earliest
                timestamps = (timestamps - timestamps.min()) / (24 * 3600)
                
                # Linear regression
                try:
                    # Check if we have enough variance in the data
                    if np.var(values) <= 0:
                        continue
                        
                    # Reshape for sklearn
                    X = timestamps.reshape(-1, 1)
                    
                    # Fit linear model
                    model = LinearRegression()
                    model.fit(X, values)
                    
                    # Get slope
                    slope = model.coef_[0]
                    
                    # Calculate p-value
                    _, _, r_value, p_value, _ = stats.linregress(timestamps, values)
                    
                    # Forecast future values
                    forecast_days = np.array(range(1, self.config["forecast_days"] + 1))
                    forecast_x = np.max(timestamps) + forecast_days
                    forecast_x = forecast_x.reshape(-1, 1)
                    forecast_values = model.predict(forecast_x)
                    
                    # Store trend analysis
                    self.task_trends[task_type][metric] = {
                        "slope": slope,
                        "p_value": p_value,
                        "r_squared": r_value ** 2,
                        "is_significant": p_value < self.config["trend_significance_threshold"],
                        "forecast": list(zip(forecast_days, forecast_values)),
                        "direction": "increasing" if slope > 0 else "decreasing",
                        "updated_at": datetime.now()
                    }
                    
                    # Log significant trends
                    if p_value < self.config["trend_significance_threshold"]:
                        direction = "increasing" if slope > 0 else "decreasing"
                        logger.info(
                            f"Detected significant {direction} trend for task_type={task_type}, "
                            f"metric={metric}, slope={slope:.4f}, p={p_value:.4f}"
                        )
                        
                        # Store in database if enabled
                        if self.db_manager and self.config["database_enabled"]:
                            try:
                                self.db_manager.add_performance_trend(
                                    worker_id=None,
                                    entity_type="task_type",
                                    entity_id=task_type,
                                    metric=metric,
                                    slope=slope,
                                    p_value=p_value,
                                    r_squared=r_value ** 2,
                                    is_significant=p_value < self.config["trend_significance_threshold"],
                                    direction=direction,
                                    forecast_values=list(forecast_values)
                                )
                            except Exception as e:
                                logger.error(f"Error storing trend: {e}")
                                
                except Exception as e:
                    logger.warning(f"Error analyzing trend for task type {task_type}, metric {metric}: {e}")
                    
    def _generate_visualizations(self):
        """Generate visualizations for performance metrics."""
        # Skip if visualization is disabled
        if not self.config["visualization_enabled"]:
            return
            
        # Ensure directory exists
        os.makedirs(self.config["visualization_path"], exist_ok=True)
        
        # Generate worker visualizations
        for worker_id, metrics in self.time_series.items():
            # Skip workers with too few metrics
            if len(metrics) == 0:
                continue
                
            # Set up figure
            plt.figure(figsize=(12, 8))
            
            # Plot each metric
            for i, (metric, data_points) in enumerate(metrics.items()):
                # Skip metrics with too few data points
                if len(data_points) < self.config["min_data_points"]:
                    continue
                    
                # Set up subplot
                plt.subplot(len(metrics), 1, i + 1)
                
                # Extract timestamps and values
                timestamps = [ts for ts, _ in data_points]
                values = [value for _, value in data_points]
                
                # Plot data
                plt.plot(timestamps, values, 'b-')
                
                # Add trend line if available
                if worker_id in self.trends and metric in self.trends[worker_id]:
                    trend = self.trends[worker_id][metric]
                    
                    if trend["is_significant"]:
                        # Get first and last points
                        x1 = timestamps[0]
                        x2 = timestamps[-1]
                        
                        # Convert to timestamps in seconds
                        x1_sec = (x1 - datetime(1970, 1, 1)).total_seconds()
                        x2_sec = (x2 - datetime(1970, 1, 1)).total_seconds()
                        
                        # Normalize to days from earliest
                        x1_days = 0
                        x2_days = (x2_sec - x1_sec) / (24 * 3600)
                        
                        # Calculate trend line points
                        y1 = values[0]
                        y2 = y1 + trend["slope"] * x2_days
                        
                        # Plot trend line
                        plt.plot([x1, x2], [y1, y2], 'r--')
                        
                        # Add forecast points
                        forecast_timestamps = [
                            x2 + timedelta(days=days) for days, _ in trend["forecast"]
                        ]
                        forecast_values = [value for _, value in trend["forecast"]]
                        
                        # Plot forecast
                        plt.plot(forecast_timestamps, forecast_values, 'r:')
                        
                # Add anomalies if available
                if worker_id in self.anomalies:
                    for anomaly in self.anomalies[worker_id]:
                        if anomaly["metric"] == metric:
                            # Mark anomaly point
                            plt.plot(
                                anomaly["timestamp"], anomaly["value"], 
                                'ro' if anomaly["is_high"] else 'go'
                            )
                            
                # Add labels
                plt.title(f"{worker_id}: {metric}")
                plt.xlabel("Time")
                plt.ylabel(metric)
                plt.grid(True)
                
            # Save figure
            filename = f"{worker_id}_performance.{self.config['visualization_format']}"
            filepath = os.path.join(self.config["visualization_path"], filename)
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()
            
        # Generate task type visualizations
        for task_type, metrics in self.task_time_series.items():
            # Skip task types with too few metrics
            if len(metrics) == 0:
                continue
                
            # Set up figure
            plt.figure(figsize=(12, 8))
            
            # Plot each metric
            for i, (metric, data_points) in enumerate(metrics.items()):
                # Skip metrics with too few data points
                if len(data_points) < self.config["min_data_points"]:
                    continue
                    
                # Set up subplot
                plt.subplot(len(metrics), 1, i + 1)
                
                # Extract timestamps and values
                timestamps = [ts for ts, _ in data_points]
                values = [value for _, value in data_points]
                
                # Plot data
                plt.plot(timestamps, values, 'b-')
                
                # Add trend line if available
                if task_type in self.task_trends and metric in self.task_trends[task_type]:
                    trend = self.task_trends[task_type][metric]
                    
                    if trend["is_significant"]:
                        # Get first and last points
                        x1 = timestamps[0]
                        x2 = timestamps[-1]
                        
                        # Convert to timestamps in seconds
                        x1_sec = (x1 - datetime(1970, 1, 1)).total_seconds()
                        x2_sec = (x2 - datetime(1970, 1, 1)).total_seconds()
                        
                        # Normalize to days from earliest
                        x1_days = 0
                        x2_days = (x2_sec - x1_sec) / (24 * 3600)
                        
                        # Calculate trend line points
                        y1 = values[0]
                        y2 = y1 + trend["slope"] * x2_days
                        
                        # Plot trend line
                        plt.plot([x1, x2], [y1, y2], 'r--')
                        
                        # Add forecast points
                        forecast_timestamps = [
                            x2 + timedelta(days=days) for days, _ in trend["forecast"]
                        ]
                        forecast_values = [value for _, value in trend["forecast"]]
                        
                        # Plot forecast
                        plt.plot(forecast_timestamps, forecast_values, 'r:')
                        
                # Add anomalies if available
                if task_type in self.task_anomalies:
                    for anomaly in self.task_anomalies[task_type]:
                        if anomaly["metric"] == metric:
                            # Mark anomaly point
                            plt.plot(
                                anomaly["timestamp"], anomaly["value"], 
                                'ro' if anomaly["is_high"] else 'go'
                            )
                            
                # Add labels
                plt.title(f"{task_type}: {metric}")
                plt.xlabel("Time")
                plt.ylabel(metric)
                plt.grid(True)
                
            # Save figure
            filename = f"{task_type}_performance.{self.config['visualization_format']}"
            filepath = os.path.join(self.config["visualization_path"], filename)
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()
            
    def get_worker_trends(self, worker_id: str = None, metric: str = None,
                        significant_only: bool = False) -> Dict[str, Any]:
        """Get trend analysis for a worker or all workers.
        
        Args:
            worker_id: Optional worker ID to get trends for (all workers if None)
            metric: Optional metric to get trends for (all metrics if None)
            significant_only: Whether to include only significant trends
            
        Returns:
            Dict containing trend analysis
        """
        if worker_id:
            if worker_id not in self.trends:
                return {}
                
            if metric:
                if metric not in self.trends[worker_id]:
                    return {}
                    
                # Return single metric trends
                return {metric: self.trends[worker_id][metric]}
                
            else:
                # Return all metrics for this worker
                if significant_only:
                    return {
                        m: t for m, t in self.trends[worker_id].items()
                        if t.get("is_significant", False)
                    }
                else:
                    return self.trends[worker_id]
                    
        else:
            # Return all workers
            if significant_only:
                result = {}
                for w_id, metrics in self.trends.items():
                    result[w_id] = {
                        m: t for m, t in metrics.items()
                        if t.get("is_significant", False)
                    }
                    # Remove workers with no significant trends
                    if not result[w_id]:
                        del result[w_id]
                        
                return result
            else:
                return self.trends
                
    def get_task_trends(self, task_type: str = None, metric: str = None,
                       significant_only: bool = False) -> Dict[str, Any]:
        """Get trend analysis for a task type or all task types.
        
        Args:
            task_type: Optional task type to get trends for (all types if None)
            metric: Optional metric to get trends for (all metrics if None)
            significant_only: Whether to include only significant trends
            
        Returns:
            Dict containing trend analysis
        """
        if task_type:
            if task_type not in self.task_trends:
                return {}
                
            if metric:
                if metric not in self.task_trends[task_type]:
                    return {}
                    
                # Return single metric trends
                return {metric: self.task_trends[task_type][metric]}
                
            else:
                # Return all metrics for this task type
                if significant_only:
                    return {
                        m: t for m, t in self.task_trends[task_type].items()
                        if t.get("is_significant", False)
                    }
                else:
                    return self.task_trends[task_type]
                    
        else:
            # Return all task types
            if significant_only:
                result = {}
                for t_id, metrics in self.task_trends.items():
                    result[t_id] = {
                        m: t for m, t in metrics.items()
                        if t.get("is_significant", False)
                    }
                    # Remove task types with no significant trends
                    if not result[t_id]:
                        del result[t_id]
                        
                return result
            else:
                return self.task_trends
                
    def get_worker_anomalies(self, worker_id: str = None, metric: str = None,
                           limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Get anomalies for a worker or all workers.
        
        Args:
            worker_id: Optional worker ID to get anomalies for (all workers if None)
            metric: Optional metric to get anomalies for (all metrics if None)
            limit: Maximum number of anomalies to return per worker/metric
            
        Returns:
            Dict containing anomalies
        """
        if worker_id:
            if worker_id not in self.anomalies:
                return {}
                
            if metric:
                # Filter by metric
                filtered = [a for a in self.anomalies[worker_id] if a["metric"] == metric]
                
                # Sort by timestamp (most recent first) and limit
                sorted_filtered = sorted(
                    filtered, 
                    key=lambda a: a["timestamp"], 
                    reverse=True
                )[:limit]
                
                return {metric: sorted_filtered}
                
            else:
                # Group by metric
                result = {}
                for anomaly in self.anomalies[worker_id]:
                    m = anomaly["metric"]
                    if m not in result:
                        result[m] = []
                        
                    result[m].append(anomaly)
                    
                # Sort each metric by timestamp and limit
                for m in result:
                    result[m] = sorted(
                        result[m], 
                        key=lambda a: a["timestamp"], 
                        reverse=True
                    )[:limit]
                    
                return result
                
        else:
            # Return all workers
            result = {}
            
            for w_id in self.anomalies:
                if metric:
                    # Filter by metric
                    filtered = [a for a in self.anomalies[w_id] if a["metric"] == metric]
                    
                    # Sort by timestamp (most recent first) and limit
                    sorted_filtered = sorted(
                        filtered, 
                        key=lambda a: a["timestamp"], 
                        reverse=True
                    )[:limit]
                    
                    if sorted_filtered:
                        result[w_id] = {metric: sorted_filtered}
                        
                else:
                    # Group by metric
                    worker_result = {}
                    for anomaly in self.anomalies[w_id]:
                        m = anomaly["metric"]
                        if m not in worker_result:
                            worker_result[m] = []
                            
                        worker_result[m].append(anomaly)
                        
                    # Sort each metric by timestamp and limit
                    for m in worker_result:
                        worker_result[m] = sorted(
                            worker_result[m], 
                            key=lambda a: a["timestamp"], 
                            reverse=True
                        )[:limit]
                        
                    if worker_result:
                        result[w_id] = worker_result
                        
            return result
            
    def get_task_anomalies(self, task_type: str = None, metric: str = None,
                          limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Get anomalies for a task type or all task types.
        
        Args:
            task_type: Optional task type to get anomalies for (all types if None)
            metric: Optional metric to get anomalies for (all metrics if None)
            limit: Maximum number of anomalies to return per task type/metric
            
        Returns:
            Dict containing anomalies
        """
        if task_type:
            if task_type not in self.task_anomalies:
                return {}
                
            if metric:
                # Filter by metric
                filtered = [a for a in self.task_anomalies[task_type] if a["metric"] == metric]
                
                # Sort by timestamp (most recent first) and limit
                sorted_filtered = sorted(
                    filtered, 
                    key=lambda a: a["timestamp"], 
                    reverse=True
                )[:limit]
                
                return {metric: sorted_filtered}
                
            else:
                # Group by metric
                result = {}
                for anomaly in self.task_anomalies[task_type]:
                    m = anomaly["metric"]
                    if m not in result:
                        result[m] = []
                        
                    result[m].append(anomaly)
                    
                # Sort each metric by timestamp and limit
                for m in result:
                    result[m] = sorted(
                        result[m], 
                        key=lambda a: a["timestamp"], 
                        reverse=True
                    )[:limit]
                    
                return result
                
        else:
            # Return all task types
            result = {}
            
            for t_id in self.task_anomalies:
                if metric:
                    # Filter by metric
                    filtered = [a for a in self.task_anomalies[t_id] if a["metric"] == metric]
                    
                    # Sort by timestamp (most recent first) and limit
                    sorted_filtered = sorted(
                        filtered, 
                        key=lambda a: a["timestamp"], 
                        reverse=True
                    )[:limit]
                    
                    if sorted_filtered:
                        result[t_id] = {metric: sorted_filtered}
                        
                else:
                    # Group by metric
                    task_result = {}
                    for anomaly in self.task_anomalies[t_id]:
                        m = anomaly["metric"]
                        if m not in task_result:
                            task_result[m] = []
                            
                        task_result[m].append(anomaly)
                        
                    # Sort each metric by timestamp and limit
                    for m in task_result:
                        task_result[m] = sorted(
                            task_result[m], 
                            key=lambda a: a["timestamp"], 
                            reverse=True
                        )[:limit]
                        
                    if task_result:
                        result[t_id] = task_result
                        
            return result
            
    def get_time_series(self, entity_id: str, entity_type: str = "worker",
                      metric: str = None, days: int = None) -> Dict[str, List[Tuple[datetime, float]]]:
        """Get time series data for a worker or task type.
        
        Args:
            entity_id: ID of the worker or task type
            entity_type: Type of entity ("worker" or "task_type")
            metric: Optional metric to get data for (all metrics if None)
            days: Optional number of days to limit history (all days if None)
            
        Returns:
            Dict mapping metrics to lists of (timestamp, value) tuples
        """
        # Determine data source
        if entity_type == "worker":
            if entity_id not in self.time_series:
                return {}
                
            data = self.time_series[entity_id]
        else:  # task_type
            if entity_id not in self.task_time_series:
                return {}
                
            data = self.task_time_series[entity_id]
            
        # Filter by metric if specified
        if metric:
            if metric not in data:
                return {}
                
            filtered_data = {metric: data[metric]}
        else:
            filtered_data = data
            
        # Filter by days if specified
        if days:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            result = {}
            for m, points in filtered_data.items():
                result[m] = [(ts, val) for ts, val in points if ts >= cutoff_time]
                
            return result
        else:
            return filtered_data
            
    def get_performance_report(self, entity_type: str = "worker",
                             significant_only: bool = True) -> Dict[str, Any]:
        """Generate a comprehensive performance report.
        
        Args:
            entity_type: Type of entity to report on ("worker" or "task_type")
            significant_only: Whether to include only significant trends
            
        Returns:
            Dict containing the performance report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "trends": {},
            "anomalies": {},
            "baselines": {},
            "entity_type": entity_type
        }
        
        if entity_type == "worker":
            # Get trends
            if significant_only:
                trends = {}
                for worker_id, metrics in self.trends.items():
                    worker_trends = {
                        m: t for m, t in metrics.items()
                        if t.get("is_significant", False)
                    }
                    if worker_trends:
                        trends[worker_id] = worker_trends
            else:
                trends = self.trends
                
            # Get anomalies
            anomalies = {}
            for worker_id, worker_anomalies in self.anomalies.items():
                if worker_anomalies:
                    # Group by metric
                    grouped = {}
                    for anomaly in worker_anomalies:
                        metric = anomaly["metric"]
                        if metric not in grouped:
                            grouped[metric] = []
                            
                        grouped[metric].append(anomaly)
                        
                    # Sort by timestamp (most recent first)
                    for metric in grouped:
                        grouped[metric] = sorted(
                            grouped[metric],
                            key=lambda a: a["timestamp"],
                            reverse=True
                        )
                        
                    anomalies[worker_id] = grouped
                    
            # Get baselines
            baselines = self.worker_baselines
            
            # Add to report
            report["trends"] = trends
            report["anomalies"] = anomalies
            report["baselines"] = baselines
            
        else:  # task_type
            # Get trends
            if significant_only:
                trends = {}
                for task_type, metrics in self.task_trends.items():
                    task_trends = {
                        m: t for m, t in metrics.items()
                        if t.get("is_significant", False)
                    }
                    if task_trends:
                        trends[task_type] = task_trends
            else:
                trends = self.task_trends
                
            # Get anomalies
            anomalies = {}
            for task_type, task_anomalies in self.task_anomalies.items():
                if task_anomalies:
                    # Group by metric
                    grouped = {}
                    for anomaly in task_anomalies:
                        metric = anomaly["metric"]
                        if metric not in grouped:
                            grouped[metric] = []
                            
                        grouped[metric].append(anomaly)
                        
                    # Sort by timestamp (most recent first)
                    for metric in grouped:
                        grouped[metric] = sorted(
                            grouped[metric],
                            key=lambda a: a["timestamp"],
                            reverse=True
                        )
                        
                    anomalies[task_type] = grouped
                    
            # Get baselines
            baselines = self.task_baselines
            
            # Add to report
            report["trends"] = trends
            report["anomalies"] = anomalies
            report["baselines"] = baselines
            
        return report
        
    def generate_visualization(self, entity_id: str, entity_type: str = "worker",
                             metric: str = None, days: int = 30) -> Optional[str]:
        """Generate visualization for a specific entity and metric.
        
        Args:
            entity_id: ID of the worker or task type
            entity_type: Type of entity ("worker" or "task_type")
            metric: Optional metric to visualize (all metrics if None)
            days: Number of days of history to include
            
        Returns:
            Path to the generated visualization file, or None if generation failed
        """
        # Skip if visualization is disabled
        if not self.config["visualization_enabled"]:
            return None
            
        # Ensure directory exists
        os.makedirs(self.config["visualization_path"], exist_ok=True)
        
        # Get time series data
        time_series = self.get_time_series(entity_id, entity_type, metric, days)
        
        # Skip if no data
        if not time_series:
            return None
            
        # Set up figure
        plt.figure(figsize=(12, 8))
        
        # Plot each metric
        for i, (metric_name, data_points) in enumerate(time_series.items()):
            # Skip metrics with too few data points
            if len(data_points) < self.config["min_data_points"]:
                continue
                
            # Set up subplot
            plt.subplot(len(time_series), 1, i + 1)
            
            # Extract timestamps and values
            timestamps = [ts for ts, _ in data_points]
            values = [value for _, value in data_points]
            
            # Plot data
            plt.plot(timestamps, values, 'b-')
            
            # Add trend line if available
            if entity_type == "worker":
                trends = self.trends
                anomalies_dict = self.anomalies
            else:
                trends = self.task_trends
                anomalies_dict = self.task_anomalies
                
            if entity_id in trends and metric_name in trends[entity_id]:
                trend = trends[entity_id][metric_name]
                
                if trend["is_significant"]:
                    # Get first and last points
                    x1 = timestamps[0]
                    x2 = timestamps[-1]
                    
                    # Convert to timestamps in seconds
                    x1_sec = (x1 - datetime(1970, 1, 1)).total_seconds()
                    x2_sec = (x2 - datetime(1970, 1, 1)).total_seconds()
                    
                    # Normalize to days from earliest
                    x1_days = 0
                    x2_days = (x2_sec - x1_sec) / (24 * 3600)
                    
                    # Calculate trend line points
                    y1 = values[0]
                    y2 = y1 + trend["slope"] * x2_days
                    
                    # Plot trend line
                    plt.plot([x1, x2], [y1, y2], 'r--')
                    
                    # Add forecast points
                    forecast_timestamps = [
                        x2 + timedelta(days=days) for days, _ in trend["forecast"]
                    ]
                    forecast_values = [value for _, value in trend["forecast"]]
                    
                    # Plot forecast
                    plt.plot(forecast_timestamps, forecast_values, 'r:')
                    
            # Add anomalies if available
            if entity_id in anomalies_dict:
                for anomaly in anomalies_dict[entity_id]:
                    if anomaly["metric"] == metric_name:
                        # Mark anomaly point
                        plt.plot(
                            anomaly["timestamp"], anomaly["value"], 
                            'ro' if anomaly["is_high"] else 'go'
                        )
                        
            # Add labels
            plt.title(f"{entity_id}: {metric_name}")
            plt.xlabel("Time")
            plt.ylabel(metric_name)
            plt.grid(True)
            
        # Save figure
        filename = f"{entity_id}_{entity_type}_performance.{self.config['visualization_format']}"
        filepath = os.path.join(self.config["visualization_path"], filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return filepath
        
    def export_data(self, filepath: str, format: str = "json") -> bool:
        """Export performance data to a file.
        
        Args:
            filepath: Path to the output file
            format: Output format ("json" or "csv")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data for export
            export_data = {
                "time_series": {},
                "task_time_series": {},
                "trends": self.trends,
                "task_trends": self.task_trends,
                "anomalies": self.anomalies,
                "task_anomalies": self.task_anomalies,
                "export_timestamp": datetime.now().isoformat()
            }
            
            # Convert time series data (convert timestamps to strings)
            for worker_id, metrics in self.time_series.items():
                export_data["time_series"][worker_id] = {}
                for metric, data_points in metrics.items():
                    export_data["time_series"][worker_id][metric] = [
                        [ts.isoformat(), val] for ts, val in data_points
                    ]
                    
            # Convert task time series data
            for task_type, metrics in self.task_time_series.items():
                export_data["task_time_series"][task_type] = {}
                for metric, data_points in metrics.items():
                    export_data["task_time_series"][task_type][metric] = [
                        [ts.isoformat(), val] for ts, val in data_points
                    ]
                    
            # Export based on format
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif format == "csv":
                # Create directory if doesn't exist
                directory = os.path.dirname(filepath)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                    
                # Create separate CSV files for different data types
                base_name = os.path.splitext(filepath)[0]
                
                # Worker time series
                worker_ts_path = f"{base_name}_worker_timeseries.csv"
                with open(worker_ts_path, 'w') as f:
                    f.write("worker_id,metric,timestamp,value\n")
                    for worker_id, metrics in self.time_series.items():
                        for metric, data_points in metrics.items():
                            for ts, val in data_points:
                                f.write(f"{worker_id},{metric},{ts.isoformat()},{val}\n")
                                
                # Task time series
                task_ts_path = f"{base_name}_task_timeseries.csv"
                with open(task_ts_path, 'w') as f:
                    f.write("task_type,metric,timestamp,value\n")
                    for task_type, metrics in self.task_time_series.items():
                        for metric, data_points in metrics.items():
                            for ts, val in data_points:
                                f.write(f"{task_type},{metric},{ts.isoformat()},{val}\n")
                                
                # Worker trends
                worker_trends_path = f"{base_name}_worker_trends.csv"
                with open(worker_trends_path, 'w') as f:
                    f.write("worker_id,metric,slope,p_value,r_squared,is_significant,direction\n")
                    for worker_id, metrics in self.trends.items():
                        for metric, trend in metrics.items():
                            f.write(
                                f"{worker_id},{metric},{trend['slope']},{trend['p_value']},"
                                f"{trend['r_squared']},{trend['is_significant']},{trend['direction']}\n"
                            )
                            
                # Task trends
                task_trends_path = f"{base_name}_task_trends.csv"
                with open(task_trends_path, 'w') as f:
                    f.write("task_type,metric,slope,p_value,r_squared,is_significant,direction\n")
                    for task_type, metrics in self.task_trends.items():
                        for metric, trend in metrics.items():
                            f.write(
                                f"{task_type},{metric},{trend['slope']},{trend['p_value']},"
                                f"{trend['r_squared']},{trend['is_significant']},{trend['direction']}\n"
                            )
                            
                # Worker anomalies
                worker_anomalies_path = f"{base_name}_worker_anomalies.csv"
                with open(worker_anomalies_path, 'w') as f:
                    f.write("worker_id,metric,timestamp,value,mean,stdev,z_score,is_high\n")
                    for worker_id, anomalies in self.anomalies.items():
                        for anomaly in anomalies:
                            f.write(
                                f"{worker_id},{anomaly['metric']},{anomaly['timestamp'].isoformat()},"
                                f"{anomaly['value']},{anomaly['mean']},{anomaly['stdev']},"
                                f"{anomaly['z_score']},{anomaly['is_high']}\n"
                            )
                            
                # Task anomalies
                task_anomalies_path = f"{base_name}_task_anomalies.csv"
                with open(task_anomalies_path, 'w') as f:
                    f.write("task_type,metric,timestamp,value,mean,stdev,z_score,is_high\n")
                    for task_type, anomalies in self.task_anomalies.items():
                        for anomaly in anomalies:
                            f.write(
                                f"{task_type},{anomaly['metric']},{anomaly['timestamp'].isoformat()},"
                                f"{anomaly['value']},{anomaly['mean']},{anomaly['stdev']},"
                                f"{anomaly['z_score']},{anomaly['is_high']}\n"
                            )
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            logger.info(f"Exported performance data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False