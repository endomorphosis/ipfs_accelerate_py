#!/usr/bin/env python3
"""
Distributed Testing Framework - Result Aggregator

This module implements the result aggregation functionality for the distributed testing framework.
It's responsible for:

- Collecting and processing test results from distributed workers
- Aggregating results across different test executions
- Performing statistical analysis on test results
- Generating comprehensive reports
- Providing insights and recommendations based on test outcomes
- Supporting multi-dimensional analysis of test outcomes
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
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("result_aggregator")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import optional dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Visualization features will be limited.")
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("SciPy not available. Statistical analysis features will be limited.")
    SCIPY_AVAILABLE = False

class ResultAggregator:
    """Result aggregation system for the distributed testing framework."""
    
    def __init__(self, db_manager=None, task_scheduler=None):
        """Initialize the result aggregator.
        
        Args:
            db_manager: Database manager for data access
            task_scheduler: Task scheduler for current data
        """
        self.db_manager = db_manager
        self.task_scheduler = task_scheduler
        
        # Aggregated results storage
        self.test_results = defaultdict(list)  # {test_id: [result_records]}
        self.worker_results = defaultdict(list)  # {worker_id: [result_records]}
        self.task_type_results = defaultdict(list)  # {task_type: [result_records]}
        self.hardware_results = defaultdict(list)  # {hardware_id: [result_records]}
        
        # Analysis results
        self.test_analysis = {}  # {test_id: analysis_record}
        self.worker_analysis = {}  # {worker_id: analysis_record}
        self.task_type_analysis = {}  # {task_type: analysis_record}
        self.hardware_analysis = {}  # {hardware_id: analysis_record}
        
        # Multi-dimensional analysis
        self.dimension_analysis = {}  # {dimension: {value: analysis_record}}
        
        # Aggregated metrics
        self.aggregated_metrics = {}  # {metric_key: aggregated_value}
        
        # Regression comparison results
        self.regression_results = {}  # {test_id: {baseline_version: comparison_result}}
        
        # Historical performance tracking
        self.historical_performance = defaultdict(dict)  # {test_id: {date: metrics}}
        
        # Configuration
        self.config = {
            "history_days": 30,  # Days of history to keep
            "update_interval": 3600,  # Seconds between automatic updates
            "visualization_enabled": True,  # Enable visualization
            "visualization_format": "png",  # Visualization format
            "visualization_path": "./result_visualizations",  # Path for visualization files
            "database_enabled": True,  # Enable database storage
            "aggregate_dimensions": ["hardware", "task_type", "model", "batch_size", "precision"],
            "statistical_tests": ["t-test", "anova", "regression"],
            "report_formats": ["json", "html", "md"],
            "regression_threshold": 0.05,  # 5% threshold for regression detection
            "significance_threshold": 0.05,  # P-value threshold for statistical significance
            "comparison_metrics": ["throughput", "latency", "memory_usage", "success_rate"],
            "anomaly_detection": {
                "enabled": True,
                "z_score_threshold": 3.0,
                "min_samples": 5
            }
        }
        
        # Update thread
        self.update_thread = None
        self.update_stop_event = threading.Event()
        
        # Create visualization directory if enabled
        if self.config["visualization_enabled"]:
            os.makedirs(self.config["visualization_path"], exist_ok=True)
            
        logger.info("Result aggregator initialized")

    def configure(self, config_updates: Dict[str, Any]):
        """Update the aggregator configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.config.update(config_updates)
        logger.info(f"Result aggregator configuration updated: {config_updates}")
        
        # Create visualization directory if newly enabled
        if self.config["visualization_enabled"]:
            os.makedirs(self.config["visualization_path"], exist_ok=True)
    
    def start(self):
        """Start the result aggregator."""
        # Load historical data
        self._load_historical_data()
        
        # Perform initial aggregation
        self._aggregate_results()
        
        # Analyze aggregated results
        self._analyze_results()
        
        # Start update thread
        self.update_stop_event.clear()
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
        logger.info("Result aggregator started")
    
    def stop(self):
        """Stop the result aggregator."""
        # Stop update thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_stop_event.set()
            self.update_thread.join(timeout=5.0)
            if self.update_thread.is_alive():
                logger.warning("Update thread did not stop gracefully")
                
        logger.info("Result aggregator stopped")
    
    def _update_loop(self):
        """Update thread function."""
        while not self.update_stop_event.is_set():
            try:
                # Load latest results
                self._load_latest_results()
                
                # Aggregate results
                self._aggregate_results()
                
                # Analyze aggregated results
                self._analyze_results()
                
                # Generate visualizations if enabled
                if self.config["visualization_enabled"] and MATPLOTLIB_AVAILABLE:
                    self._generate_visualizations()
                    
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                
            # Wait for next update interval
            self.update_stop_event.wait(self.config["update_interval"])
    
    def _load_historical_data(self):
        """Load historical test results from database."""
        if not self.db_manager or not self.config["database_enabled"]:
            logger.info("No database manager available or database disabled, skipping historical data loading")
            return
            
        try:
            # Calculate start date
            history_days = self.config["history_days"]
            start_date = datetime.now() - timedelta(days=history_days)
            
            # Load test results
            test_results = self.db_manager.get_test_results(start_date)
            logger.info(f"Loaded {len(test_results)} historical test results")
            
            # Process test results
            for result in test_results:
                self._process_test_result(result)
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def _load_latest_results(self):
        """Load latest test results since last update."""
        if not self.db_manager or not self.config["database_enabled"]:
            return
            
        try:
            # Calculate start date (last update time)
            last_update_time = datetime.now() - timedelta(seconds=self.config["update_interval"] * 1.1)
            
            # Load latest test results
            latest_results = self.db_manager.get_test_results(last_update_time)
            logger.info(f"Loaded {len(latest_results)} latest test results")
            
            # Process latest results
            for result in latest_results:
                self._process_test_result(result)
                
        except Exception as e:
            logger.error(f"Error loading latest results: {e}")
    
    def _process_test_result(self, result):
        """Process a single test result and add to appropriate collections.
        
        Args:
            result: Test result record
        """
        # Extract key fields
        test_id = result.get("test_id")
        worker_id = result.get("worker_id")
        task_type = result.get("task_type")
        hardware_id = result.get("hardware_id")
        
        # Skip results missing key fields
        if not test_id or not worker_id:
            return
        
        # Add to test results
        self.test_results[test_id].append(result)
        
        # Add to worker results
        self.worker_results[worker_id].append(result)
        
        # Add to task type results if available
        if task_type:
            self.task_type_results[task_type].append(result)
        
        # Add to hardware results if available
        if hardware_id:
            self.hardware_results[hardware_id].append(result)
            
        # Add to historical performance tracking
        timestamp = result.get("timestamp", datetime.now())
        date_key = timestamp.strftime("%Y-%m-%d")
        
        # Extract metrics to track historically
        metrics = {}
        for key, value in result.items():
            if isinstance(value, (int, float)) and key in self.config["comparison_metrics"]:
                metrics[key] = value
                
        if metrics:
            if test_id not in self.historical_performance:
                self.historical_performance[test_id] = {}
                
            if date_key not in self.historical_performance[test_id]:
                self.historical_performance[test_id][date_key] = {
                    "metrics": defaultdict(list),
                    "count": 0
                }
                
            # Add metrics to tracking
            for key, value in metrics.items():
                self.historical_performance[test_id][date_key]["metrics"][key].append(value)
                
            # Increment count
            self.historical_performance[test_id][date_key]["count"] += 1
    
    def _aggregate_results(self):
        """Aggregate results across different dimensions."""
        # Aggregate by configured dimensions
        dimensions = self.config["aggregate_dimensions"]
        
        # Reset dimension analysis
        self.dimension_analysis = {}
        
        # Iterate through dimensions
        for dimension in dimensions:
            self.dimension_analysis[dimension] = {}
            
            # Collect all values for this dimension
            values = set()
            for results in self.test_results.values():
                for result in results:
                    if dimension in result:
                        values.add(result[dimension])
            
            # Aggregate results for each value
            for value in values:
                # Collect results with this value
                filtered_results = []
                for results in self.test_results.values():
                    for result in results:
                        if dimension in result and result[dimension] == value:
                            filtered_results.append(result)
                
                # Skip if no results
                if not filtered_results:
                    continue
                
                # Perform aggregation
                aggregated = self._calculate_aggregates(filtered_results)
                
                # Store aggregated results
                self.dimension_analysis[dimension][value] = aggregated
                
        # Calculate overall aggregated metrics
        all_results = []
        for test_results in self.test_results.values():
            all_results.extend(test_results)
            
        if all_results:
            self.aggregated_metrics = self._calculate_aggregates(all_results)
        
        # Aggregate historical performance data
        for test_id, date_data in self.historical_performance.items():
            for date_key, data in date_data.items():
                if data["count"] > 0:
                    # Calculate mean for each metric
                    for metric, values in data["metrics"].items():
                        if values:
                            data["metrics"][metric] = {
                                "mean": statistics.mean(values),
                                "count": len(values)
                            }
                            if len(values) > 1:
                                data["metrics"][metric]["std"] = statistics.stdev(values)
                            else:
                                data["metrics"][metric]["std"] = 0
        
        logger.info(f"Aggregated results across {len(dimensions)} dimensions")
    
    def _calculate_aggregates(self, results):
        """Calculate aggregate statistics for a set of results.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing aggregated statistics
        """
        # Extract numeric metrics
        metrics = {}
        
        # Identify metrics in first result
        if results:
            for key, value in results[0].items():
                if isinstance(value, (int, float)) and key not in ["timestamp", "id"]:
                    metrics[key] = []
        
        # Collect metric values from all results
        for result in results:
            for key in metrics:
                if key in result and isinstance(result[key], (int, float)):
                    metrics[key].append(result[key])
        
        # Calculate statistics
        aggregates = {
            "count": len(results),
            "start_time": min(r.get("timestamp", datetime.now()) for r in results),
            "end_time": max(r.get("timestamp", datetime.now()) for r in results),
        }
        
        # Calculate statistics for each metric
        for key, values in metrics.items():
            if not values:
                continue
                
            # Basic statistics
            aggregates[f"{key}_mean"] = statistics.mean(values)
            if len(values) > 1:
                aggregates[f"{key}_stdev"] = statistics.stdev(values)
            else:
                aggregates[f"{key}_stdev"] = 0
                
            aggregates[f"{key}_min"] = min(values)
            aggregates[f"{key}_max"] = max(values)
            
            # Percentiles
            aggregates[f"{key}_median"] = statistics.median(values)
            aggregates[f"{key}_p95"] = np.percentile(values, 95)
            aggregates[f"{key}_p99"] = np.percentile(values, 99)
            
            # Coefficient of variation (relative standard deviation)
            if aggregates[f"{key}_mean"] > 0:
                aggregates[f"{key}_cv"] = (aggregates[f"{key}_stdev"] / aggregates[f"{key}_mean"]) * 100
            else:
                aggregates[f"{key}_cv"] = 0
        
        return aggregates
    
    def _analyze_results(self):
        """Analyze aggregated results to derive insights."""
        # Analyze test results
        for test_id, results in self.test_results.items():
            # Skip tests with too few results
            if len(results) < 2:
                continue
                
            # Perform analysis
            analysis = self._analyze_test_results(test_id, results)
            
            # Store analysis
            self.test_analysis[test_id] = analysis
        
        # Analyze worker results
        for worker_id, results in self.worker_results.items():
            # Skip workers with too few results
            if len(results) < 2:
                continue
                
            # Perform analysis
            analysis = self._analyze_worker_results(worker_id, results)
            
            # Store analysis
            self.worker_analysis[worker_id] = analysis
        
        # Analyze task type results
        for task_type, results in self.task_type_results.items():
            # Skip task types with too few results
            if len(results) < 2:
                continue
                
            # Perform analysis
            analysis = self._analyze_task_type_results(task_type, results)
            
            # Store analysis
            self.task_type_analysis[task_type] = analysis
        
        # Analyze hardware results
        for hardware_id, results in self.hardware_results.items():
            # Skip hardware with too few results
            if len(results) < 2:
                continue
                
            # Perform analysis
            analysis = self._analyze_hardware_results(hardware_id, results)
            
            # Store analysis
            self.hardware_analysis[hardware_id] = analysis
            
        # Detect regressions
        self._detect_regressions()
        
        logger.info("Completed result analysis")
    
    def _analyze_test_results(self, test_id, results):
        """Analyze results for a specific test.
        
        Args:
            test_id: ID of the test
            results: List of result records
            
        Returns:
            Dict containing analysis results
        """
        # Extract test metadata
        test_name = results[0].get("test_name", test_id)
        task_type = results[0].get("task_type", "unknown")
        
        # Basic analysis
        analysis = {
            "test_id": test_id,
            "test_name": test_name,
            "task_type": task_type,
            "execution_count": len(results),
            "first_execution": min(r.get("timestamp", datetime.now()) for r in results),
            "last_execution": max(r.get("timestamp", datetime.now()) for r in results),
            "success_rate": sum(1 for r in results if r.get("status") == "success") / len(results),
            "average_duration": statistics.mean([r.get("duration", 0) for r in results]),
            "failure_reasons": self._analyze_failure_reasons(results),
            "performance_trend": self._analyze_performance_trend(results),
            "dimension_correlations": self._analyze_dimension_correlations(results),
        }
        
        return analysis
    
    def _analyze_worker_results(self, worker_id, results):
        """Analyze results for a specific worker.
        
        Args:
            worker_id: ID of the worker
            results: List of result records
            
        Returns:
            Dict containing analysis results
        """
        # Extract worker metadata
        worker_name = results[0].get("worker_name", worker_id)
        
        # Basic analysis
        analysis = {
            "worker_id": worker_id,
            "worker_name": worker_name,
            "execution_count": len(results),
            "first_execution": min(r.get("timestamp", datetime.now()) for r in results),
            "last_execution": max(r.get("timestamp", datetime.now()) for r in results),
            "success_rate": sum(1 for r in results if r.get("status") == "success") / len(results),
            "average_duration": statistics.mean([r.get("duration", 0) for r in results]),
            "task_type_distribution": self._analyze_task_type_distribution(results),
            "performance_trend": self._analyze_performance_trend(results),
            "resource_utilization": self._analyze_resource_utilization(results),
        }
        
        return analysis
    
    def _analyze_task_type_results(self, task_type, results):
        """Analyze results for a specific task type.
        
        Args:
            task_type: Type of task
            results: List of result records
            
        Returns:
            Dict containing analysis results
        """
        # Basic analysis
        analysis = {
            "task_type": task_type,
            "execution_count": len(results),
            "first_execution": min(r.get("timestamp", datetime.now()) for r in results),
            "last_execution": max(r.get("timestamp", datetime.now()) for r in results),
            "success_rate": sum(1 for r in results if r.get("status") == "success") / len(results),
            "average_duration": statistics.mean([r.get("duration", 0) for r in results]),
            "worker_distribution": self._analyze_worker_distribution(results),
            "performance_trend": self._analyze_performance_trend(results),
            "hardware_performance": self._analyze_hardware_performance(results),
            "performance_variance": self._analyze_performance_variance(results),
        }
        
        return analysis
    
    def _analyze_hardware_results(self, hardware_id, results):
        """Analyze results for a specific hardware configuration.
        
        Args:
            hardware_id: ID of the hardware
            results: List of result records
            
        Returns:
            Dict containing analysis results
        """
        # Extract hardware metadata
        hardware_name = results[0].get("hardware_name", hardware_id)
        
        # Basic analysis
        analysis = {
            "hardware_id": hardware_id,
            "hardware_name": hardware_name,
            "execution_count": len(results),
            "first_execution": min(r.get("timestamp", datetime.now()) for r in results),
            "last_execution": max(r.get("timestamp", datetime.now()) for r in results),
            "success_rate": sum(1 for r in results if r.get("status") == "success") / len(results),
            "average_duration": statistics.mean([r.get("duration", 0) for r in results]),
            "task_type_distribution": self._analyze_task_type_distribution(results),
            "performance_trend": self._analyze_performance_trend(results),
            "model_performance": self._analyze_model_performance(results),
            "optimal_configurations": self._analyze_optimal_configurations(results),
        }
        
        return analysis
    
    def _analyze_failure_reasons(self, results):
        """Analyze failure reasons in results.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing failure analysis
        """
        # Collect failure reasons
        failures = [r for r in results if r.get("status") != "success"]
        
        # Count by reason
        reason_counts = defaultdict(int)
        for failure in failures:
            reason = failure.get("failure_reason", "unknown")
            reason_counts[reason] += 1
        
        # Calculate percentages
        total_failures = len(failures)
        if total_failures > 0:
            reason_percentages = {
                reason: (count / total_failures) * 100
                for reason, count in reason_counts.items()
            }
        else:
            reason_percentages = {}
        
        return {
            "total_failures": total_failures,
            "failure_rate": (total_failures / len(results)) * 100 if results else 0,
            "reason_counts": dict(reason_counts),
            "reason_percentages": reason_percentages,
        }
    
    def _analyze_task_type_distribution(self, results):
        """Analyze task type distribution in results.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing task type distribution
        """
        # Count by task type
        type_counts = defaultdict(int)
        for result in results:
            task_type = result.get("task_type", "unknown")
            type_counts[task_type] += 1
        
        # Calculate percentages
        total = len(results)
        type_percentages = {
            task_type: (count / total) * 100
            for task_type, count in type_counts.items()
        }
        
        return {
            "type_counts": dict(type_counts),
            "type_percentages": type_percentages,
        }
    
    def _analyze_worker_distribution(self, results):
        """Analyze worker distribution in results.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing worker distribution
        """
        # Count by worker
        worker_counts = defaultdict(int)
        for result in results:
            worker_id = result.get("worker_id", "unknown")
            worker_counts[worker_id] += 1
        
        # Calculate percentages
        total = len(results)
        worker_percentages = {
            worker_id: (count / total) * 100
            for worker_id, count in worker_counts.items()
        }
        
        return {
            "worker_counts": dict(worker_counts),
            "worker_percentages": worker_percentages,
        }
    
    def _analyze_performance_trend(self, results):
        """Analyze performance trend in results.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing performance trend analysis
        """
        # Sort results by timestamp
        sorted_results = sorted(results, key=lambda r: r.get("timestamp", datetime.now()))
        
        # Extract timestamps and durations
        timestamps = np.array([(r.get("timestamp", datetime.now()) - datetime(1970, 1, 1)).total_seconds() 
                      for r in sorted_results])
        durations = np.array([r.get("duration", 0) for r in sorted_results])
        
        # Skip if not enough data points
        if len(timestamps) < 2:
            return {"trend": "insufficient_data"}
        
        # Normalize timestamps to days from earliest
        timestamps = (timestamps - timestamps.min()) / (24 * 3600)
        
        # Linear regression
        try:
            # Check if we have enough variance in the data
            if np.var(durations) <= 0:
                return {"trend": "no_variance"}
                
            # Calculate linear regression
            if SCIPY_AVAILABLE:
                slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, durations)
                
                # Determine trend
                if p_value < self.config["significance_threshold"]:  # Statistically significant
                    trend = "increasing" if slope > 0 else "decreasing"
                else:
                    trend = "stable"
                    
                return {
                    "trend": trend,
                    "slope": slope,
                    "p_value": p_value,
                    "r_squared": r_value ** 2,
                    "is_significant": p_value < self.config["significance_threshold"],
                }
            else:
                # Fallback if scipy is not available
                n = len(timestamps)
                mean_x = np.mean(timestamps)
                mean_y = np.mean(durations)
                
                # Calculate covariance and variance
                cov_xy = np.sum((timestamps - mean_x) * (durations - mean_y)) / n
                var_x = np.sum((timestamps - mean_x) ** 2) / n
                
                # Calculate slope and intercept
                slope = cov_xy / var_x if var_x > 0 else 0
                intercept = mean_y - slope * mean_x
                
                # Determine trend (without statistical significance)
                trend = "increasing" if slope > 0 else "decreasing"
                
                return {
                    "trend": trend,
                    "slope": slope,
                    "intercept": intercept,
                    "is_significant": None,  # Not available without scipy
                }
            
        except Exception as e:
            logger.warning(f"Error analyzing performance trend: {e}")
            return {"trend": "analysis_error"}
    
    def _analyze_dimension_correlations(self, results):
        """Analyze correlations between dimensions and performance.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing correlation analysis
        """
        if not SCIPY_AVAILABLE:
            return {"error": "scipy_not_available"}
            
        correlations = {}
        
        # Analyze each dimension
        for dimension in self.config["aggregate_dimensions"]:
            # Skip if not enough results
            if len(results) < 5:
                continue
                
            # Get unique values
            values = set()
            for result in results:
                if dimension in result:
                    values.add(result[dimension])
            
            # Skip if too few unique values
            if len(values) < 2:
                continue
                
            # Group by value
            grouped = defaultdict(list)
            for result in results:
                if dimension in result:
                    grouped[result[dimension]].append(result)
            
            # Skip if any group too small
            if any(len(group) < 2 for group in grouped.values()):
                continue
                
            # Extract durations
            duration_by_value = {
                value: [r.get("duration", 0) for r in group]
                for value, group in grouped.items()
            }
            
            # Calculate statistics
            stats_by_value = {
                value: {
                    "mean": statistics.mean(durations),
                    "std": statistics.stdev(durations) if len(durations) > 1 else 0,
                    "count": len(durations),
                    "min": min(durations),
                    "max": max(durations),
                }
                for value, durations in duration_by_value.items()
            }
            
            # Calculate significance
            try:
                # ANOVA for categorical dimensions with more than 2 values
                if len(values) > 2:
                    # One-way ANOVA
                    anova_groups = [durations for durations in duration_by_value.values()]
                    f_val, p_val = stats.f_oneway(*anova_groups)
                    
                    significance = {
                        "test": "anova",
                        "f_value": f_val,
                        "p_value": p_val,
                        "is_significant": p_val < self.config["significance_threshold"],
                    }
                # T-test for 2 values
                else:
                    # Two-sample t-test
                    groups = list(duration_by_value.values())
                    t_val, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    
                    significance = {
                        "test": "t-test",
                        "t_value": t_val,
                        "p_value": p_val,
                        "is_significant": p_val < self.config["significance_threshold"],
                    }
            except Exception as e:
                logger.warning(f"Error calculating significance for dimension {dimension}: {e}")
                significance = {"test": "error", "error": str(e)}
            
            # Store correlation
            correlations[dimension] = {
                "stats_by_value": stats_by_value,
                "significance": significance,
                "value_count": len(values),
            }
        
        return correlations
    
    def _analyze_resource_utilization(self, results):
        """Analyze resource utilization in results.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing resource utilization analysis
        """
        # Extract resource metrics
        cpu_utilization = [r.get("cpu_percent", 0) for r in results if "cpu_percent" in r]
        memory_utilization = [r.get("memory_percent", 0) for r in results if "memory_percent" in r]
        gpu_utilization = [r.get("gpu_percent", 0) for r in results if "gpu_percent" in r]
        
        # Calculate statistics
        analysis = {}
        
        if cpu_utilization:
            analysis["cpu"] = {
                "mean": statistics.mean(cpu_utilization),
                "max": max(cpu_utilization),
                "min": min(cpu_utilization),
                "std": statistics.stdev(cpu_utilization) if len(cpu_utilization) > 1 else 0,
            }
            
        if memory_utilization:
            analysis["memory"] = {
                "mean": statistics.mean(memory_utilization),
                "max": max(memory_utilization),
                "min": min(memory_utilization),
                "std": statistics.stdev(memory_utilization) if len(memory_utilization) > 1 else 0,
            }
            
        if gpu_utilization:
            analysis["gpu"] = {
                "mean": statistics.mean(gpu_utilization),
                "max": max(gpu_utilization),
                "min": min(gpu_utilization),
                "std": statistics.stdev(gpu_utilization) if len(gpu_utilization) > 1 else 0,
            }
        
        return analysis
    
    def _analyze_hardware_performance(self, results):
        """Analyze performance across different hardware configurations.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing hardware performance analysis
        """
        # Group by hardware
        grouped = defaultdict(list)
        for result in results:
            hardware_id = result.get("hardware_id", "unknown")
            grouped[hardware_id].append(result)
        
        # Calculate performance stats for each hardware
        hardware_stats = {}
        for hardware_id, hw_results in grouped.items():
            durations = [r.get("duration", 0) for r in hw_results]
            
            if not durations:
                continue
                
            hardware_stats[hardware_id] = {
                "count": len(hw_results),
                "mean_duration": statistics.mean(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
                "success_rate": sum(1 for r in hw_results if r.get("status") == "success") / len(hw_results),
            }
        
        # Rank hardware by performance
        if hardware_stats:
            ranked = sorted(hardware_stats.items(), key=lambda x: x[1]["mean_duration"])
            fastest = ranked[0][0]
            slowest = ranked[-1][0]
        else:
            fastest = None
            slowest = None
        
        return {
            "hardware_stats": hardware_stats,
            "fastest_hardware": fastest,
            "slowest_hardware": slowest,
            "hardware_count": len(hardware_stats),
        }
    
    def _analyze_model_performance(self, results):
        """Analyze performance across different models.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing model performance analysis
        """
        # Group by model
        grouped = defaultdict(list)
        for result in results:
            model = result.get("model", "unknown")
            grouped[model].append(result)
        
        # Calculate performance stats for each model
        model_stats = {}
        for model, model_results in grouped.items():
            durations = [r.get("duration", 0) for r in model_results]
            
            if not durations:
                continue
                
            model_stats[model] = {
                "count": len(model_results),
                "mean_duration": statistics.mean(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
                "success_rate": sum(1 for r in model_results if r.get("status") == "success") / len(model_results),
            }
        
        # Rank models by performance
        if model_stats:
            ranked = sorted(model_stats.items(), key=lambda x: x[1]["mean_duration"])
            fastest = ranked[0][0]
            slowest = ranked[-1][0]
        else:
            fastest = None
            slowest = None
        
        return {
            "model_stats": model_stats,
            "fastest_model": fastest,
            "slowest_model": slowest,
            "model_count": len(model_stats),
        }
    
    def _analyze_performance_variance(self, results):
        """Analyze performance variance across different dimensions.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing variance analysis
        """
        # Extract durations
        durations = [r.get("duration", 0) for r in results]
        
        # Skip if not enough data
        if len(durations) < 2:
            return {"variance_analysis": "insufficient_data"}
        
        # Basic statistics
        stats = {
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "std": statistics.stdev(durations),
            "min": min(durations),
            "max": max(durations),
            "p95": np.percentile(durations, 95),
            "p99": np.percentile(durations, 99),
            "coefficient_of_variation": statistics.stdev(durations) / statistics.mean(durations) if statistics.mean(durations) > 0 else 0,
        }
        
        # Analyze variance by dimension
        dimension_variance = {}
        for dimension in self.config["aggregate_dimensions"]:
            # Get unique values
            values = set()
            for result in results:
                if dimension in result:
                    values.add(result[dimension])
            
            # Skip if too few unique values
            if len(values) < 2:
                continue
                
            # Group by value
            grouped = defaultdict(list)
            for result in results:
                if dimension in result:
                    grouped[result[dimension]].append(result.get("duration", 0))
            
            # Calculate variance for each value
            value_stats = {}
            for value, value_durations in grouped.items():
                if len(value_durations) < 2:
                    continue
                    
                value_stats[value] = {
                    "mean": statistics.mean(value_durations),
                    "std": statistics.stdev(value_durations),
                    "count": len(value_durations),
                    "coefficient_of_variation": statistics.stdev(value_durations) / statistics.mean(value_durations) if statistics.mean(value_durations) > 0 else 0,
                }
            
            # Calculate within-group and between-group variance
            if len(value_stats) >= 2:
                try:
                    # Calculate group means
                    group_means = [stats["mean"] for stats in value_stats.values()]
                    group_counts = [stats["count"] for stats in value_stats.values()]
                    group_variances = [stats["std"] ** 2 for stats in value_stats.values()]
                    
                    # Calculate grand mean
                    grand_mean = statistics.mean(durations)
                    
                    # Within-group variance (weighted average of group variances)
                    within_variance = sum(count * var for count, var in zip(group_counts, group_variances)) / sum(group_counts)
                    
                    # Between-group variance (weighted variance of group means)
                    between_variance = sum(count * ((mean - grand_mean) ** 2) for count, mean in zip(group_counts, group_means)) / sum(group_counts)
                    
                    # Calculate explained variance ratio
                    total_variance = statistics.variance(durations)
                    explained_ratio = between_variance / total_variance if total_variance > 0 else 0
                    
                    dimension_variance[dimension] = {
                        "within_group_variance": within_variance,
                        "between_group_variance": between_variance,
                        "explained_variance_ratio": explained_ratio,
                        "is_significant_factor": explained_ratio > 0.3,  # 30% threshold
                    }
                except Exception as e:
                    logger.warning(f"Error calculating variance for dimension {dimension}: {e}")
                    dimension_variance[dimension] = {"error": str(e)}
        
        return {
            "overall_stats": stats,
            "dimension_variance": dimension_variance,
        }
    
    def _analyze_optimal_configurations(self, results):
        """Analyze optimal configurations for performance.
        
        Args:
            results: List of result records
            
        Returns:
            Dict containing optimal configuration analysis
        """
        # Extract dimensions of interest
        dimensions = [d for d in self.config["aggregate_dimensions"] if d != "hardware"]
        
        # Skip if not enough dimensions
        if not dimensions:
            return {"optimal_config_analysis": "no_dimensions"}
        
        # Group by configuration (combination of dimension values)
        configs = defaultdict(list)
        for result in results:
            # Create configuration key
            config = tuple(result.get(d, "unknown") for d in dimensions)
            configs[config].append(result)
        
        # Calculate performance for each configuration
        config_stats = {}
        for config, config_results in configs.items():
            # Skip configurations with too few results
            if len(config_results) < 3:
                continue
                
            # Extract durations
            durations = [r.get("duration", 0) for r in config_results]
            
            # Calculate statistics
            config_stats[config] = {
                "count": len(config_results),
                "mean_duration": statistics.mean(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
                "success_rate": sum(1 for r in config_results if r.get("status") == "success") / len(config_results),
            }
        
        # Rank configurations by performance
        if config_stats:
            ranked = sorted(config_stats.items(), key=lambda x: x[1]["mean_duration"])
            
            # Extract top configurations
            top_configs = ranked[:min(3, len(ranked))]
            top_configs_formatted = []
            
            for config, stats in top_configs:
                config_dict = {dim: val for dim, val in zip(dimensions, config)}
                top_configs_formatted.append({
                    "config": config_dict,
                    "stats": stats,
                })
            
            # Extract bottom configurations
            bottom_configs = ranked[-min(3, len(ranked)):]
            bottom_configs_formatted = []
            
            for config, stats in bottom_configs:
                config_dict = {dim: val for dim, val in zip(dimensions, config)}
                bottom_configs_formatted.append({
                    "config": config_dict,
                    "stats": stats,
                })
        else:
            top_configs_formatted = []
            bottom_configs_formatted = []
        
        return {
            "config_count": len(config_stats),
            "top_configs": top_configs_formatted,
            "bottom_configs": bottom_configs_formatted,
        }
    
    def _detect_regressions(self):
        """Detect performance regressions in historical data."""
        # Reset regression results
        self.regression_results = {}
        
        # Iterate through tests with historical data
        for test_id, history in self.historical_performance.items():
            # Sort dates
            sorted_dates = sorted(history.keys())
            
            # Skip tests with too few historical data points
            if len(sorted_dates) < 2:
                continue
                
            # Consider latest date as current and previous as baseline
            if len(sorted_dates) >= 7:
                # Use 7-day windows for comparison when enough data is available
                current_period = sorted_dates[-7:]
                baseline_period = sorted_dates[-14:-7]
            else:
                # Use latest day as current and the rest as baseline
                current_period = [sorted_dates[-1]]
                baseline_period = sorted_dates[:-1]
                
            # Skip if either period is empty
            if not current_period or not baseline_period:
                continue
                
            # Extract metrics for comparison
            regression_metrics = {}
            
            for metric in self.config["comparison_metrics"]:
                # Check if metric exists in both periods
                current_values = []
                baseline_values = []
                
                for date in current_period:
                    if date in history and metric in history[date]["metrics"]:
                        if isinstance(history[date]["metrics"][metric], dict):
                            # Already aggregated
                            if "mean" in history[date]["metrics"][metric]:
                                current_values.append(history[date]["metrics"][metric]["mean"])
                        else:
                            # Raw values
                            current_values.extend(history[date]["metrics"][metric])
                            
                for date in baseline_period:
                    if date in history and metric in history[date]["metrics"]:
                        if isinstance(history[date]["metrics"][metric], dict):
                            # Already aggregated
                            if "mean" in history[date]["metrics"][metric]:
                                baseline_values.append(history[date]["metrics"][metric]["mean"])
                        else:
                            # Raw values
                            baseline_values.extend(history[date]["metrics"][metric])
                
                # Skip if not enough data
                if len(current_values) < 2 or len(baseline_values) < 2:
                    continue
                    
                # Calculate means
                current_mean = statistics.mean(current_values)
                baseline_mean = statistics.mean(baseline_values)
                
                # Calculate percent change
                percent_change = ((current_mean - baseline_mean) / baseline_mean) * 100
                
                # Direction depends on metric (for some metrics, higher is better)
                better_when_higher = metric in ["throughput", "success_rate"]
                
                # Determine if regression
                is_regression = False
                if better_when_higher:
                    # Regression if decreasing significantly
                    is_regression = percent_change < -self.config["regression_threshold"] * 100
                else:
                    # Regression if increasing significantly
                    is_regression = percent_change > self.config["regression_threshold"] * 100
                
                # Statistical significance
                is_significant = False
                p_value = None
                
                if SCIPY_AVAILABLE:
                    try:
                        _, p_value = stats.ttest_ind(current_values, baseline_values, equal_var=False)
                        is_significant = p_value < self.config["significance_threshold"]
                    except Exception as e:
                        logger.warning(f"Error calculating significance for regression: {e}")
                
                # Store regression information
                regression_metrics[metric] = {
                    "current_mean": current_mean,
                    "baseline_mean": baseline_mean,
                    "percent_change": percent_change,
                    "is_regression": is_regression,
                    "is_significant": is_significant,
                    "p_value": p_value,
                    "better_when_higher": better_when_higher,
                    "current_sample_size": len(current_values),
                    "baseline_sample_size": len(baseline_values),
                }
            
            # Store regression results if any metrics show regression
            if any(info["is_regression"] for info in regression_metrics.values()):
                self.regression_results[test_id] = {
                    "metrics": regression_metrics,
                    "current_period": current_period,
                    "baseline_period": baseline_period,
                    "has_significant_regression": any(
                        info["is_regression"] and info["is_significant"] 
                        for info in regression_metrics.values()
                    ),
                }
    
    def _generate_visualizations(self):
        """Generate visualizations for aggregated results."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping visualizations")
            return
            
        # Ensure visualization directory exists
        os.makedirs(self.config["visualization_path"], exist_ok=True)
        
        # Generate performance trend visualizations
        for test_id, history in self.historical_performance.items():
            # Skip tests with too little data
            if len(history) < 2:
                continue
                
            # Sort dates
            sorted_dates = sorted(history.keys())
            
            # Set up figure
            plt.figure(figsize=(12, 8))
            
            # Plot each metric
            metrics_to_plot = set()
            for date in sorted_dates:
                metrics_to_plot.update(history[date]["metrics"].keys())
                
            for i, metric in enumerate(sorted(metrics_to_plot)):
                # Set up subplot
                plt.subplot(len(metrics_to_plot), 1, i + 1)
                
                # Extract dates and values
                dates = []
                values = []
                
                for date in sorted_dates:
                    if date in history and metric in history[date]["metrics"]:
                        if isinstance(history[date]["metrics"][metric], dict):
                            # Already aggregated
                            if "mean" in history[date]["metrics"][metric]:
                                dates.append(date)
                                values.append(history[date]["metrics"][metric]["mean"])
                        elif isinstance(history[date]["metrics"][metric], list):
                            # Calculate mean for raw values
                            if history[date]["metrics"][metric]:
                                dates.append(date)
                                values.append(statistics.mean(history[date]["metrics"][metric]))
                        else:
                            # Single value
                            dates.append(date)
                            values.append(history[date]["metrics"][metric])
                
                # Skip if not enough data
                if len(dates) < 2:
                    continue
                    
                # Convert dates to datetime for plotting
                plot_dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
                
                # Plot data
                plt.plot(plot_dates, values, 'b-o')
                
                # Add regression info if available
                if test_id in self.regression_results and metric in self.regression_results[test_id]["metrics"]:
                    regression_info = self.regression_results[test_id]["metrics"][metric]
                    
                    if regression_info["is_regression"]:
                        # Add regression highlighting
                        if regression_info["is_significant"]:
                            # Add red marker for significant regression
                            plt.axhspan(
                                min(regression_info["current_mean"], regression_info["baseline_mean"]),
                                max(regression_info["current_mean"], regression_info["baseline_mean"]),
                                alpha=0.2, color='red'
                            )
                        else:
                            # Add yellow marker for non-significant regression
                            plt.axhspan(
                                min(regression_info["current_mean"], regression_info["baseline_mean"]),
                                max(regression_info["current_mean"], regression_info["baseline_mean"]),
                                alpha=0.2, color='yellow'
                            )
                        
                        # Add annotation
                        plt.annotate(
                            f"{regression_info['percent_change']:.1f}%",
                            xy=(plot_dates[-1], values[-1]),
                            xytext=(10, 0),
                            textcoords="offset points",
                            color='red' if regression_info["is_significant"] else 'orange'
                        )
                
                # Add labels
                plt.title(f"{test_id}: {metric}")
                plt.xlabel("Date")
                plt.ylabel(metric)
                plt.grid(True)
                
                # Format x-axis as dates
                plt.gcf().autofmt_xdate()
            
            # Save figure
            filename = f"{test_id}_performance_trend.{self.config['visualization_format']}"
            filepath = os.path.join(self.config["visualization_path"], filename)
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()
            
        logger.info(f"Generated visualizations in {self.config['visualization_path']}")
    
    def get_overall_status(self):
        """Get overall status of test results.
        
        Returns:
            Dict containing overall status
        """
        return {
            "test_count": len(self.test_results),
            "worker_count": len(self.worker_results),
            "task_type_count": len(self.task_type_results),
            "hardware_count": len(self.hardware_results),
            "total_executions": sum(len(results) for results in self.test_results.values()),
            "aggregated_metrics": self.aggregated_metrics,
            "regression_count": len(self.regression_results),
            "significant_regression_count": sum(
                1 for result in self.regression_results.values() 
                if result.get("has_significant_regression", False)
            ),
        }
    
    def get_regressions(self, significant_only=False):
        """Get detected regressions.
        
        Args:
            significant_only: Whether to include only statistically significant regressions
            
        Returns:
            Dict containing regression information
        """
        if significant_only:
            return {
                test_id: info for test_id, info in self.regression_results.items()
                if info.get("has_significant_regression", False)
            }
        else:
            return self.regression_results
    
    def get_test_analysis(self, test_id=None):
        """Get analysis for a specific test or all tests.
        
        Args:
            test_id: Optional ID of the test to get analysis for (all tests if None)
            
        Returns:
            Dict containing test analysis
        """
        if test_id:
            return self.test_analysis.get(test_id, {})
        else:
            return self.test_analysis
    
    def get_dimension_analysis(self, dimension=None):
        """Get analysis for a specific dimension or all dimensions.
        
        Args:
            dimension: Optional dimension to get analysis for (all dimensions if None)
            
        Returns:
            Dict containing dimension analysis
        """
        if dimension:
            return self.dimension_analysis.get(dimension, {})
        else:
            return self.dimension_analysis
    
    def generate_report(self, format="json", output_file=None):
        """Generate a comprehensive report of results and analysis.
        
        Args:
            format: Report format (json, html, md)
            output_file: Optional file to write report to
            
        Returns:
            Report content as string
        """
        # Collect report data
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self.get_overall_status(),
            "regressions": self.get_regressions(),
            "test_analysis": self.get_test_analysis(),
            "dimension_analysis": self.get_dimension_analysis(),
        }
        
        # Generate report in requested format
        if format == "json":
            report_content = json.dumps(report_data, indent=2)
        elif format == "html":
            report_content = self._generate_html_report(report_data)
        elif format == "md":
            report_content = self._generate_markdown_report(report_data)
        else:
            logger.error(f"Unsupported report format: {format}")
            return None
        
        # Write to file if requested
        if output_file:
            try:
                with open(output_file, "w") as f:
                    f.write(report_content)
                logger.info(f"Report written to {output_file}")
            except Exception as e:
                logger.error(f"Error writing report to {output_file}: {e}")
        
        return report_content
    
    def _generate_html_report(self, report_data):
        """Generate HTML report.
        
        Args:
            report_data: Report data
            
        Returns:
            HTML report as string
        """
        # Simple HTML report template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Distributed Testing Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .regression {{ color: red; font-weight: bold; }}
                .warning {{ color: orange; }}
                .success {{ color: green; }}
            </style>
        </head>
        <body>
            <h1>Distributed Testing Results Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Overall Status</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Test Count</td><td>{report_data["overall_status"]["test_count"]}</td></tr>
                <tr><td>Worker Count</td><td>{report_data["overall_status"]["worker_count"]}</td></tr>
                <tr><td>Task Type Count</td><td>{report_data["overall_status"]["task_type_count"]}</td></tr>
                <tr><td>Hardware Count</td><td>{report_data["overall_status"]["hardware_count"]}</td></tr>
                <tr><td>Total Executions</td><td>{report_data["overall_status"]["total_executions"]}</td></tr>
                <tr><td>Regression Count</td><td>{report_data["overall_status"]["regression_count"]}</td></tr>
                <tr><td>Significant Regression Count</td><td>{report_data["overall_status"]["significant_regression_count"]}</td></tr>
            </table>
        """
        
        # Add regressions section if available
        if report_data["regressions"]:
            html += """
            <h2>Performance Regressions</h2>
            <table>
                <tr><th>Test ID</th><th>Metric</th><th>Change</th><th>Significance</th></tr>
            """
            
            for test_id, regression in report_data["regressions"].items():
                for metric, info in regression["metrics"].items():
                    if info["is_regression"]:
                        html += f"""
                        <tr class="{'regression' if info['is_significant'] else 'warning'}">
                            <td>{test_id}</td>
                            <td>{metric}</td>
                            <td>{info['percent_change']:.2f}%</td>
                            <td>{'Significant' if info['is_significant'] else 'Not significant'}</td>
                        </tr>
                        """
            
            html += "</table>"
        
        # Add test analysis section
        if report_data["test_analysis"]:
            html += """
            <h2>Test Analysis Summary</h2>
            <table>
                <tr><th>Test ID</th><th>Executions</th><th>Success Rate</th><th>Avg Duration</th></tr>
            """
            
            for test_id, analysis in report_data["test_analysis"].items():
                html += f"""
                <tr>
                    <td>{test_id}</td>
                    <td>{analysis.get('execution_count', 0)}</td>
                    <td>{analysis.get('success_rate', 0) * 100:.2f}%</td>
                    <td>{analysis.get('average_duration', 0):.2f}s</td>
                </tr>
                """
            
            html += "</table>"
        
        # Close HTML
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_markdown_report(self, report_data):
        """Generate Markdown report.
        
        Args:
            report_data: Report data
            
        Returns:
            Markdown report as string
        """
        # Simple Markdown report template
        md = f"""# Distributed Testing Results Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overall Status

| Metric | Value |
|--------|-------|
| Test Count | {report_data["overall_status"]["test_count"]} |
| Worker Count | {report_data["overall_status"]["worker_count"]} |
| Task Type Count | {report_data["overall_status"]["task_type_count"]} |
| Hardware Count | {report_data["overall_status"]["hardware_count"]} |
| Total Executions | {report_data["overall_status"]["total_executions"]} |
| Regression Count | {report_data["overall_status"]["regression_count"]} |
| Significant Regression Count | {report_data["overall_status"]["significant_regression_count"]} |

"""
        
        # Add regressions section if available
        if report_data["regressions"]:
            md += """## Performance Regressions

| Test ID | Metric | Change | Significance |
|---------|--------|--------|-------------|
"""
            
            for test_id, regression in report_data["regressions"].items():
                for metric, info in regression["metrics"].items():
                    if info["is_regression"]:
                        md += f"| {test_id} | {metric} | {info['percent_change']:.2f}% | {'Significant' if info['is_significant'] else 'Not significant'} |\n"
            
            md += "\n"
        
        # Add test analysis section
        if report_data["test_analysis"]:
            md += """## Test Analysis Summary

| Test ID | Executions | Success Rate | Avg Duration |
|---------|------------|--------------|--------------|
"""
            
            for test_id, analysis in report_data["test_analysis"].items():
                md += f"| {test_id} | {analysis.get('execution_count', 0)} | {analysis.get('success_rate', 0) * 100:.2f}% | {analysis.get('average_duration', 0):.2f}s |\n"
        
        return md