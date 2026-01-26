#!/usr/bin/env python3
"""
Performance Analyzer for Result Aggregator

This module provides advanced performance analysis capabilities for the Result Aggregator.
It includes statistical regression detection, comparative analysis across test runs,
hardware-specific performance analysis, and resource efficiency metrics.

Usage:
    # Create a performance analyzer with database integration
    analyzer = PerformanceAnalyzer(service)
    
    # Analyze performance regression
    regression_analysis = analyzer.detect_performance_regression(metric_name="throughput")
    
    # Compare performance across different hardware profiles
    comparison = analyzer.compare_hardware_performance(metrics=["throughput", "latency"])
    
    # Generate performance efficiency report
    efficiency_report = analyzer.analyze_resource_efficiency()
    
    # Generate comprehensive performance report
    performance_report = analyzer.generate_performance_report(format="markdown")
"""

import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("performance_analyzer.log")
    ]
)
logger = logging.getLogger(__name__)


def _is_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules)


def _log_optional_dependency(message: str) -> None:
    if _is_pytest():
        logger.info(message)
    else:
        logger.warning(message)

# Import optional dependencies if available
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    _log_optional_dependency("Visualization libraries not available. Visualization features will be disabled.")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    _log_optional_dependency("Scikit-learn not available. Advanced regression analysis will be disabled.")

class PerformanceAnalyzer:
    """Advanced performance analysis for test results."""
    
    def __init__(self, service):
        """
        Initialize the performance analyzer.
        
        Args:
            service: Reference to ResultAggregatorService
        """
        self.service = service
        self.db = service.db
        self.enable_visualization = service.enable_visualization and VISUALIZATION_AVAILABLE
        self.enable_ml = service.enable_ml and ML_AVAILABLE
        
        # Performance metrics of interest for analysis
        self.key_metrics = [
            "throughput", "latency_ms", "execution_time", "memory_usage_mb", 
            "qps", "response_time_ms", "cpu_usage_percent"
        ]
        
        # Thresholds for regression severity
        self.regression_thresholds = {
            "critical": 30.0,  # 30% regression
            "major": 15.0,     # 15% regression
            "minor": 5.0,      # 5% regression
            "improvement": -5.0  # 5% improvement (negative regression)
        }
        
        # Statistical significance level
        self.significance_level = 0.05
        
        logger.info("Performance Analyzer initialized")
    
    def detect_performance_regression(self, metric_name: str = None, 
                                     baseline_period: str = "7d",
                                     comparison_period: str = "1d",
                                     min_samples: int = 5,
                                     filter_criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect performance regression for a specific metric.
        
        Args:
            metric_name: Name of the metric to analyze (None for all key metrics)
            baseline_period: Period for baseline (e.g., "7d" for 7 days)
            comparison_period: Period for comparison (e.g., "1d" for 1 day)
            min_samples: Minimum number of samples required for analysis
            filter_criteria: Additional filter criteria
            
        Returns:
            Regression analysis results
        """
        if not self.db:
            logger.warning("No database connection available")
            return {}
        
        # Process periods
        baseline_days = int(baseline_period.rstrip("d"))
        comparison_days = int(comparison_period.rstrip("d"))
        
        # Calculate date ranges
        now = datetime.now()
        baseline_start = now - timedelta(days=baseline_days)
        comparison_start = now - timedelta(days=comparison_days)
        
        # Metrics to analyze
        metrics_to_analyze = [metric_name] if metric_name else self.key_metrics
        
        # Results container
        results = {
            "metrics": {},
            "summary": {
                "total_metrics_analyzed": 0,
                "regressions_detected": 0,
                "improvements_detected": 0,
                "stable_metrics": 0,
                "insufficient_data": 0
            }
        }
        
        # Analyze each metric
        for metric in metrics_to_analyze:
            # Query baseline data
            baseline_data = self._get_metric_data(
                metric_name=metric,
                start_time=baseline_start.isoformat(),
                end_time=comparison_start.isoformat(),
                filter_criteria=filter_criteria
            )
            
            # Query comparison data
            comparison_data = self._get_metric_data(
                metric_name=metric,
                start_time=comparison_start.isoformat(),
                end_time=now.isoformat(),
                filter_criteria=filter_criteria
            )
            
            # Check if we have enough data
            if len(baseline_data) < min_samples or len(comparison_data) < min_samples:
                results["metrics"][metric] = {
                    "status": "insufficient_data",
                    "baseline_count": len(baseline_data),
                    "comparison_count": len(comparison_data),
                    "min_samples_required": min_samples
                }
                results["summary"]["insufficient_data"] += 1
                continue
            
            # Calculate statistics
            baseline_mean = np.mean(baseline_data)
            comparison_mean = np.mean(comparison_data)
            baseline_std = np.std(baseline_data)
            comparison_std = np.std(comparison_data)
            
            # Calculate percent change
            if baseline_mean != 0:
                percent_change = ((comparison_mean - baseline_mean) / baseline_mean) * 100
            else:
                percent_change = 0.0
            
            # Perform statistical test (t-test)
            t_stat, p_value = stats.ttest_ind(baseline_data, comparison_data)
            
            # Determine if the change is statistically significant
            is_significant = p_value < self.significance_level
            
            # Determine regression severity
            if percent_change >= self.regression_thresholds["critical"]:
                severity = "critical"
            elif percent_change >= self.regression_thresholds["major"]:
                severity = "major"
            elif percent_change >= self.regression_thresholds["minor"]:
                severity = "minor"
            elif percent_change <= self.regression_thresholds["improvement"]:
                severity = "improvement"
            else:
                severity = "stable"
            
            # Determine status
            if severity in ["critical", "major", "minor"] and is_significant:
                status = "regression"
                results["summary"]["regressions_detected"] += 1
            elif severity == "improvement" and is_significant:
                status = "improvement"
                results["summary"]["improvements_detected"] += 1
            else:
                status = "stable"
                results["summary"]["stable_metrics"] += 1
            
            # Store results
            results["metrics"][metric] = {
                "status": status,
                "severity": severity,
                "percent_change": percent_change,
                "is_statistically_significant": is_significant,
                "p_value": p_value,
                "baseline": {
                    "count": len(baseline_data),
                    "mean": baseline_mean,
                    "std": baseline_std,
                    "min": min(baseline_data),
                    "max": max(baseline_data)
                },
                "comparison": {
                    "count": len(comparison_data),
                    "mean": comparison_mean,
                    "std": comparison_std,
                    "min": min(comparison_data),
                    "max": max(comparison_data)
                }
            }
            
        results["summary"]["total_metrics_analyzed"] = len(metrics_to_analyze)
        return results
    
    def _get_metric_data(self, metric_name: str, start_time: str, end_time: str, 
                        filter_criteria: Dict[str, Any] = None) -> List[float]:
        """
        Get metric data from the database.
        
        Args:
            metric_name: Name of the metric
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            filter_criteria: Additional filter criteria
            
        Returns:
            List of metric values
        """
        if not self.db:
            return []
        
        try:
            # Start building the query
            query = """
            SELECT m.metric_value
            FROM test_results t
            JOIN performance_metrics m ON t.id = m.result_id
            WHERE m.metric_name = ? AND t.timestamp >= ? AND t.timestamp <= ?
            """
            
            params = [metric_name, start_time, end_time]
            
            # Apply additional filters
            if filter_criteria:
                for key, value in filter_criteria.items():
                    if key == "test_type":
                        query += " AND t.test_type = ?"
                        params.append(value)
                    elif key == "status":
                        query += " AND t.status = ?"
                        params.append(value)
                    elif key == "worker_id":
                        query += " AND t.worker_id = ?"
                        params.append(value)
                    elif key == "task_id":
                        query += " AND t.task_id = ?"
                        params.append(value)
            
            # Execute query
            rows = self.db.execute(query, params).fetchall()
            
            # Extract values
            return [row[0] for row in rows]
            
        except Exception as e:
            logger.error(f"Error retrieving metric data: {e}")
            return []
    
    def compare_hardware_performance(self, metrics: List[str] = None,
                                    test_type: str = None,
                                    time_period: str = "30d") -> Dict[str, Any]:
        """
        Compare performance across different hardware profiles.
        
        Args:
            metrics: List of metrics to compare (None for all key metrics)
            test_type: Type of test to analyze
            time_period: Time period for analysis (e.g., "30d" for 30 days)
            
        Returns:
            Hardware performance comparison results
        """
        if not self.db:
            logger.warning("No database connection available")
            return {}
        
        # Metrics to analyze
        metrics_to_analyze = metrics if metrics else self.key_metrics
        
        # Calculate time range
        days = int(time_period.rstrip("d"))
        start_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            # Get hardware profiles from test results
            query = """
            SELECT DISTINCT json_extract(details, '$.requirements.hardware') as hardware
            FROM test_results
            WHERE timestamp >= ?
            """
            
            params = [start_time]
            
            if test_type:
                query += " AND test_type = ?"
                params.append(test_type)
            
            rows = self.db.execute(query, params).fetchall()
            
            # Process hardware profiles
            hardware_profiles = []
            for row in rows:
                if row[0]:
                    try:
                        hardware = json.loads(row[0])
                        if isinstance(hardware, list):
                            for hw in hardware:
                                if hw not in hardware_profiles:
                                    hardware_profiles.append(hw)
                        elif hardware not in hardware_profiles:
                            hardware_profiles.append(hardware)
                    except json.JSONDecodeError:
                        continue
            
            # Results container
            results = {
                "hardware_profiles": hardware_profiles,
                "metrics": {},
                "summary": {
                    "best_overall_hardware": None,
                    "best_by_metric": {}
                }
            }
            
            # Analyze each metric
            hardware_scores = {hw: 0 for hw in hardware_profiles}
            
            for metric in metrics_to_analyze:
                metric_results = {}
                best_hardware = None
                best_value = None
                is_higher_better = metric in ["throughput", "qps"]
                
                for hardware in hardware_profiles:
                    # Query data for this hardware and metric
                    query = """
                    SELECT m.metric_value
                    FROM test_results t
                    JOIN performance_metrics m ON t.id = m.result_id
                    WHERE m.metric_name = ?
                    AND t.timestamp >= ?
                    AND json_extract(t.details, '$.requirements.hardware') LIKE ?
                    """
                    
                    params = [metric, start_time, f"%{hardware}%"]
                    
                    if test_type:
                        query += " AND t.test_type = ?"
                        params.append(test_type)
                    
                    rows = self.db.execute(query, params).fetchall()
                    
                    # Extract values
                    values = [row[0] for row in rows]
                    
                    if values:
                        # Calculate statistics
                        mean = np.mean(values)
                        std = np.std(values)
                        min_val = min(values)
                        max_val = max(values)
                        
                        # Store results
                        metric_results[hardware] = {
                            "count": len(values),
                            "mean": mean,
                            "std": std,
                            "min": min_val,
                            "max": max_val
                        }
                        
                        # Update best hardware for this metric
                        if best_value is None or (is_higher_better and mean > best_value) or (not is_higher_better and mean < best_value):
                            best_hardware = hardware
                            best_value = mean
                
                if best_hardware:
                    # Award points to the best hardware
                    hardware_scores[best_hardware] += 1
                    
                    # Calculate relative performance for scoring
                    for hardware in hardware_profiles:
                        if hardware in metric_results:
                            relative_performance = 0
                            if is_higher_better and best_value > 0:
                                relative_performance = metric_results[hardware]["mean"] / best_value
                            elif not is_higher_better and metric_results[best_hardware]["mean"] > 0:
                                relative_performance = metric_results[best_hardware]["mean"] / metric_results[hardware]["mean"]
                            
                            # Add score (weighted by relative performance)
                            hardware_scores[hardware] += relative_performance * 0.5
                
                # Store in results
                results["metrics"][metric] = {
                    "hardware_results": metric_results,
                    "best_hardware": best_hardware,
                    "is_higher_better": is_higher_better
                }
                
                # Store in summary
                if best_hardware:
                    results["summary"]["best_by_metric"][metric] = best_hardware
            
            # Determine overall best hardware
            if hardware_scores:
                results["summary"]["best_overall_hardware"] = max(hardware_scores.items(), key=lambda x: x[1])[0]
                results["summary"]["hardware_scores"] = hardware_scores
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing hardware performance: {e}")
            return {}
    
    def analyze_resource_efficiency(self, test_type: str = None,
                                   time_period: str = "30d") -> Dict[str, Any]:
        """
        Analyze resource efficiency metrics.
        
        Args:
            test_type: Type of test to analyze
            time_period: Time period for analysis (e.g., "30d" for 30 days)
            
        Returns:
            Resource efficiency analysis results
        """
        if not self.db:
            logger.warning("No database connection available")
            return {}
        
        # Calculate time range
        days = int(time_period.rstrip("d"))
        start_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            # Results container
            results = {
                "efficiency_metrics": {},
                "summary": {
                    "most_efficient_setup": None,
                    "efficiency_scores": {}
                }
            }
            
            # Check if we have memory usage and throughput/execution time metrics
            throughput_memory_query = """
            SELECT t.id
            FROM test_results t
            JOIN performance_metrics m1 ON t.id = m1.result_id
            JOIN performance_metrics m2 ON t.id = m2.result_id
            WHERE t.timestamp >= ?
            AND m1.metric_name = 'memory_usage_mb'
            AND m2.metric_name IN ('throughput', 'execution_time')
            """
            
            params = [start_time]
            
            if test_type:
                throughput_memory_query += " AND t.test_type = ?"
                params.append(test_type)
            
            has_throughput_memory = len(self.db.execute(throughput_memory_query, params).fetchall()) > 0
            
            # Check if we have power consumption metrics
            power_query = """
            SELECT t.id
            FROM test_results t
            JOIN performance_metrics m ON t.id = m.result_id
            WHERE t.timestamp >= ?
            AND m.metric_name = 'power_consumption'
            """
            
            params = [start_time]
            
            if test_type:
                power_query += " AND t.test_type = ?"
                params.append(test_type)
            
            has_power_metrics = len(self.db.execute(power_query, params).fetchall()) > 0
            
            # Get distinct test configurations
            config_query = """
            SELECT DISTINCT json_extract(details, '$.requirements.hardware') as hardware, 
                            json_extract(details, '$.metadata.batch_size') as batch_size,
                            json_extract(details, '$.metadata.model') as model
            FROM test_results
            WHERE timestamp >= ?
            """
            
            params = [start_time]
            
            if test_type:
                config_query += " AND test_type = ?"
                params.append(test_type)
            
            config_rows = self.db.execute(config_query, params).fetchall()
            
            # Process configurations
            configurations = []
            for row in config_rows:
                config = {}
                
                # Process hardware
                if row[0]:
                    try:
                        hardware = json.loads(row[0])
                        config["hardware"] = hardware
                    except json.JSONDecodeError:
                        continue
                
                # Process batch size
                if row[1]:
                    try:
                        batch_size = int(row[1])
                        config["batch_size"] = batch_size
                    except (ValueError, TypeError):
                        continue
                
                # Process model
                if row[2]:
                    config["model"] = row[2]
                
                if config and "hardware" in config and "batch_size" in config and "model" in config:
                    configurations.append(config)
            
            # Calculate efficiency metrics for each configuration
            efficiency_scores = {}
            
            for config in configurations:
                config_str = self._get_config_string(config)
                
                # Get metrics for this configuration
                metrics_query = """
                SELECT m.metric_name, AVG(m.metric_value) as avg_value
                FROM test_results t
                JOIN performance_metrics m ON t.id = m.result_id
                WHERE t.timestamp >= ?
                AND json_extract(t.details, '$.requirements.hardware') LIKE ?
                AND json_extract(t.details, '$.metadata.batch_size') = ?
                AND json_extract(t.details, '$.metadata.model') = ?
                """
                
                params = [
                    start_time,
                    f"%{config['hardware'][0] if isinstance(config['hardware'], list) else config['hardware']}%",
                    str(config['batch_size']),
                    config['model']
                ]
                
                if test_type:
                    metrics_query += " AND t.test_type = ?"
                    params.append(test_type)
                
                metrics_query += " GROUP BY m.metric_name"
                
                metrics_rows = self.db.execute(metrics_query, params).fetchall()
                
                # Convert to dictionary
                metrics = {row[0]: row[1] for row in metrics_rows}
                
                # Calculate efficiency metrics
                efficiency_metrics = {}
                
                # Throughput per memory unit (if available)
                if "throughput" in metrics and "memory_usage_mb" in metrics and metrics["memory_usage_mb"] > 0:
                    throughput_per_memory = metrics["throughput"] / metrics["memory_usage_mb"]
                    efficiency_metrics["throughput_per_memory"] = throughput_per_memory
                
                # Throughput per power unit (if available)
                if "throughput" in metrics and "power_consumption" in metrics and metrics["power_consumption"] > 0:
                    throughput_per_power = metrics["throughput"] / metrics["power_consumption"]
                    efficiency_metrics["throughput_per_power"] = throughput_per_power
                
                # Execution time per memory unit (if available)
                if "execution_time" in metrics and "memory_usage_mb" in metrics and metrics["memory_usage_mb"] > 0:
                    time_memory_efficiency = 1 / (metrics["execution_time"] * metrics["memory_usage_mb"])
                    efficiency_metrics["time_memory_efficiency"] = time_memory_efficiency
                
                # Store metrics and raw data
                if efficiency_metrics:
                    results["efficiency_metrics"][config_str] = {
                        "configuration": config,
                        "raw_metrics": metrics,
                        "efficiency_metrics": efficiency_metrics
                    }
                    
                    # Calculate overall efficiency score
                    score = 0
                    if "throughput_per_memory" in efficiency_metrics:
                        score += efficiency_metrics["throughput_per_memory"]  # Weight: 1.0
                    if "throughput_per_power" in efficiency_metrics:
                        score += efficiency_metrics["throughput_per_power"] * 2.0  # Weight: 2.0
                    if "time_memory_efficiency" in efficiency_metrics:
                        score += efficiency_metrics["time_memory_efficiency"] * 0.5  # Weight: 0.5
                    
                    efficiency_scores[config_str] = score
            
            # Update summary
            if efficiency_scores:
                results["summary"]["efficiency_scores"] = efficiency_scores
                results["summary"]["most_efficient_setup"] = max(efficiency_scores.items(), key=lambda x: x[1])[0]
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing resource efficiency: {e}")
            return {}
    
    def _get_config_string(self, config: Dict[str, Any]) -> str:
        """
        Get a string representation of a configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            String representation
        """
        hardware_str = "-".join(config["hardware"]) if isinstance(config["hardware"], list) else config["hardware"]
        return f"{config['model']}_b{config['batch_size']}_{hardware_str}"
    
    def analyze_performance_over_time(self, metric_name: str,
                                    grouping: str = "day",
                                    test_type: str = None,
                                    time_period: str = "90d") -> Dict[str, Any]:
        """
        Analyze performance trends over time with advanced regression analysis.
        
        Args:
            metric_name: Metric to analyze
            grouping: Time grouping (day, week, month)
            test_type: Type of test to analyze
            time_period: Time period for analysis (e.g., "90d" for 90 days)
            
        Returns:
            Time-based performance analysis results
        """
        if not self.db:
            logger.warning("No database connection available")
            return {}
        
        # Calculate time range
        days = int(time_period.rstrip("d"))
        start_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            # Determine time grouping SQL function
            if grouping == "week":
                time_func = "DATE_TRUNC('week', t.timestamp)"
            elif grouping == "month":
                time_func = "DATE_TRUNC('month', t.timestamp)"
            else:  # day
                time_func = "DATE_TRUNC('day', t.timestamp)"
            
            # Query data
            query = f"""
            SELECT {time_func} as time_group, AVG(m.metric_value) as avg_value
            FROM test_results t
            JOIN performance_metrics m ON t.id = m.result_id
            WHERE t.timestamp >= ?
            AND m.metric_name = ?
            """
            
            params = [start_time, metric_name]
            
            if test_type:
                query += " AND t.test_type = ?"
                params.append(test_type)
            
            query += " GROUP BY time_group ORDER BY time_group ASC"
            
            rows = self.db.execute(query, params).fetchall()
            
            # Process results
            times = []
            values = []
            
            for row in rows:
                times.append(row[0])
                values.append(row[1])
            
            # Create time series
            time_series = []
            for i in range(len(times)):
                time_series.append({
                    "timestamp": times[i].isoformat() if hasattr(times[i], "isoformat") else times[i],
                    "value": values[i]
                })
            
            # Results container
            results = {
                "metric": metric_name,
                "grouping": grouping,
                "time_series": time_series,
                "analysis": {}
            }
            
            # Perform regression analysis
            if len(values) >= 3 and self.enable_ml:
                # Prepare data
                X = np.arange(len(values)).reshape(-1, 1)
                y = np.array(values)
                
                # Linear regression
                linear_model = LinearRegression()
                linear_model.fit(X, y)
                linear_pred = linear_model.predict(X)
                
                # Calculate metrics
                linear_r2 = r2_score(y, linear_pred)
                linear_mse = mean_squared_error(y, linear_pred)
                
                # Polynomial regression (degree 2)
                poly_features = PolynomialFeatures(degree=2)
                X_poly = poly_features.fit_transform(X)
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                poly_pred = poly_model.predict(X_poly)
                
                # Calculate metrics
                poly_r2 = r2_score(y, poly_pred)
                poly_mse = mean_squared_error(y, poly_pred)
                
                # Determine best model
                best_model = "linear" if linear_r2 >= poly_r2 else "polynomial"
                
                # Calculate trend
                if best_model == "linear":
                    slope = linear_model.coef_[0]
                    
                    if slope > 0.01:
                        trend = "increasing"
                    elif slope < -0.01:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                else:  # polynomial
                    # Check direction of curve at the end point
                    last_idx = len(values) - 1
                    if last_idx > 0:
                        slope_at_end = poly_pred[last_idx] - poly_pred[last_idx - 1]
                        
                        if slope_at_end > 0.01:
                            trend = "increasing"
                        elif slope_at_end < -0.01:
                            trend = "decreasing"
                        else:
                            trend = "stable"
                    else:
                        trend = "unknown"
                
                # Store results
                results["analysis"] = {
                    "linear_regression": {
                        "slope": float(linear_model.coef_[0]),
                        "intercept": float(linear_model.intercept_),
                        "r2": linear_r2,
                        "mse": linear_mse,
                        "predictions": linear_pred.tolist()
                    },
                    "polynomial_regression": {
                        "coefficients": poly_model.coef_.tolist(),
                        "intercept": float(poly_model.intercept_),
                        "r2": poly_r2,
                        "mse": poly_mse,
                        "predictions": poly_pred.tolist()
                    },
                    "best_model": best_model,
                    "trend": trend
                }
                
                # Forecast next data points
                if len(values) >= 5:
                    forecast_x = np.arange(len(values), len(values) + 3).reshape(-1, 1)
                    
                    if best_model == "linear":
                        forecast_y = linear_model.predict(forecast_x)
                    else:  # polynomial
                        forecast_x_poly = poly_features.transform(forecast_x)
                        forecast_y = poly_model.predict(forecast_x_poly)
                    
                    # Store forecast
                    results["analysis"]["forecast"] = forecast_y.tolist()
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing performance over time: {e}")
            return {}
    
    def generate_performance_report(self, report_type: str = "comprehensive",
                                  filter_criteria: Dict[str, Any] = None,
                                  format: str = "markdown",
                                  time_period: str = "30d") -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            report_type: Type of report (comprehensive, regression, hardware_comparison, efficiency)
            filter_criteria: Filter criteria for the report
            format: Report format (markdown, html, json)
            time_period: Time period for analysis (e.g., "30d" for 30 days)
            
        Returns:
            Performance report in the specified format
        """
        # Report data container
        report_data = {
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "period_analyzed": time_period
        }
        
        # Add filter criteria
        if filter_criteria:
            report_data["filter_criteria"] = filter_criteria
        
        # Generate report based on type
        if report_type == "comprehensive" or report_type == "all":
            # Regression analysis for key metrics
            regression_results = {}
            for metric in self.key_metrics:
                regression = self.detect_performance_regression(
                    metric_name=metric,
                    filter_criteria=filter_criteria,
                    baseline_period=time_period
                )
                if "metrics" in regression and metric in regression["metrics"]:
                    regression_results[metric] = regression["metrics"][metric]
            
            report_data["regression_analysis"] = regression_results
            
            # Hardware comparison
            report_data["hardware_comparison"] = self.compare_hardware_performance(
                metrics=self.key_metrics,
                time_period=time_period
            )
            
            # Resource efficiency
            report_data["resource_efficiency"] = self.analyze_resource_efficiency(
                time_period=time_period
            )
            
            # Time-based analysis for throughput
            report_data["time_analysis"] = {}
            if "throughput" in self.key_metrics:
                report_data["time_analysis"]["throughput"] = self.analyze_performance_over_time(
                    metric_name="throughput",
                    time_period=time_period
                )
            
            # Add latency analysis if available
            if "latency_ms" in self.key_metrics:
                report_data["time_analysis"]["latency"] = self.analyze_performance_over_time(
                    metric_name="latency_ms",
                    time_period=time_period
                )
            
        elif report_type == "regression":
            # Regression analysis for all key metrics
            report_data["regression_analysis"] = self.detect_performance_regression(
                filter_criteria=filter_criteria,
                baseline_period=time_period
            )
            
        elif report_type == "hardware_comparison":
            # Hardware comparison for key metrics
            report_data["hardware_comparison"] = self.compare_hardware_performance(
                metrics=self.key_metrics,
                time_period=time_period
            )
            
        elif report_type == "efficiency":
            # Resource efficiency analysis
            report_data["resource_efficiency"] = self.analyze_resource_efficiency(
                time_period=time_period
            )
            
        elif report_type == "time_analysis":
            # Time-based analysis for key metrics
            report_data["time_analysis"] = {}
            for metric in self.key_metrics:
                report_data["time_analysis"][metric] = self.analyze_performance_over_time(
                    metric_name=metric,
                    time_period=time_period
                )
        
        # Format the report
        if format == "json":
            # Return JSON string
            return json.dumps(report_data, indent=2)
            
        elif format == "markdown":
            # Generate Markdown report
            return self._generate_markdown_report(report_data)
            
        elif format == "html":
            # Generate HTML report
            return self._generate_html_report(report_data)
            
        else:
            logger.warning(f"Unknown format: {format}. Using JSON.")
            return json.dumps(report_data, indent=2)
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """
        Generate a Markdown format report.
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            Markdown report
        """
        markdown = f"# Performance Analysis Report: {report_data['report_type'].capitalize()}\n\n"
        markdown += f"*Generated at: {report_data['generated_at']}*\n\n"
        markdown += f"*Period analyzed: {report_data['period_analyzed']}*\n\n"
        
        # Add filter criteria if available
        if "filter_criteria" in report_data:
            markdown += "## Filter Criteria\n\n"
            for key, value in report_data["filter_criteria"].items():
                markdown += f"- **{key}**: {value}\n"
            markdown += "\n"
        
        # Add regression analysis if available
        if "regression_analysis" in report_data:
            markdown += "## Regression Analysis\n\n"
            
            if "summary" in report_data["regression_analysis"]:
                # Summary section
                summary = report_data["regression_analysis"]["summary"]
                markdown += "### Summary\n\n"
                markdown += f"- **Total metrics analyzed**: {summary.get('total_metrics_analyzed', 0)}\n"
                markdown += f"- **Regressions detected**: {summary.get('regressions_detected', 0)}\n"
                markdown += f"- **Improvements detected**: {summary.get('improvements_detected', 0)}\n"
                markdown += f"- **Stable metrics**: {summary.get('stable_metrics', 0)}\n"
                markdown += f"- **Insufficient data**: {summary.get('insufficient_data', 0)}\n\n"
            
            # Detailed metrics
            metric_data = report_data["regression_analysis"].get("metrics", {})
            if not metric_data and isinstance(report_data["regression_analysis"], dict):
                # Handle case where metrics are directly in regression_analysis
                metric_data = report_data["regression_analysis"]
            
            if metric_data:
                markdown += "### Metrics\n\n"
                
                for metric, data in metric_data.items():
                    status = data.get("status", "unknown")
                    severity = data.get("severity", "unknown")
                    
                    # Determine emoji based on status
                    if status == "regression":
                        emoji = "🔴" if severity == "critical" else "🟠" if severity == "major" else "🟡"
                    elif status == "improvement":
                        emoji = "🟢"
                    elif status == "stable":
                        emoji = "🔵"
                    else:
                        emoji = "⚪"
                    
                    markdown += f"#### {emoji} {metric}\n\n"
                    
                    if status == "insufficient_data":
                        markdown += f"*Insufficient data for analysis.*\n"
                        markdown += f"- Baseline count: {data.get('baseline_count', 0)}\n"
                        markdown += f"- Comparison count: {data.get('comparison_count', 0)}\n"
                        markdown += f"- Minimum samples required: {data.get('min_samples_required', 0)}\n\n"
                        continue
                    
                    # Add percent change with formatting
                    percent_change = data.get("percent_change", 0)
                    if percent_change > 0:
                        markdown += f"- **Percent change**: +{percent_change:.2f}% (worse)\n"
                    else:
                        markdown += f"- **Percent change**: {percent_change:.2f}% (better)\n"
                    
                    # Add statistical significance
                    is_significant = data.get("is_statistically_significant", False)
                    markdown += f"- **Statistically significant**: {'Yes' if is_significant else 'No'}\n"
                    
                    if "p_value" in data:
                        markdown += f"- **p-value**: {data['p_value']:.4f}\n"
                    
                    # Add baseline and comparison data
                    if "baseline" in data:
                        baseline = data["baseline"]
                        markdown += "\n**Baseline:**\n"
                        markdown += f"- Count: {baseline.get('count', 0)}\n"
                        markdown += f"- Mean: {baseline.get('mean', 0):.2f}\n"
                        markdown += f"- Std Dev: {baseline.get('std', 0):.2f}\n"
                        markdown += f"- Min: {baseline.get('min', 0):.2f}\n"
                        markdown += f"- Max: {baseline.get('max', 0):.2f}\n"
                    
                    if "comparison" in data:
                        comparison = data["comparison"]
                        markdown += "\n**Comparison:**\n"
                        markdown += f"- Count: {comparison.get('count', 0)}\n"
                        markdown += f"- Mean: {comparison.get('mean', 0):.2f}\n"
                        markdown += f"- Std Dev: {comparison.get('std', 0):.2f}\n"
                        markdown += f"- Min: {comparison.get('min', 0):.2f}\n"
                        markdown += f"- Max: {comparison.get('max', 0):.2f}\n"
                    
                    markdown += "\n"
        
        # Add hardware comparison if available
        if "hardware_comparison" in report_data:
            markdown += "## Hardware Performance Comparison\n\n"
            
            hardware_comparison = report_data["hardware_comparison"]
            hardware_profiles = hardware_comparison.get("hardware_profiles", [])
            
            if "summary" in hardware_comparison and hardware_comparison["summary"]:
                summary = hardware_comparison["summary"]
                markdown += "### Summary\n\n"
                
                if "best_overall_hardware" in summary:
                    markdown += f"- **Best overall hardware**: {summary['best_overall_hardware']}\n"
                
                if "best_by_metric" in summary:
                    markdown += "\n**Best hardware by metric:**\n"
                    for metric, hardware in summary["best_by_metric"].items():
                        markdown += f"- **{metric}**: {hardware}\n"
                
                if "hardware_scores" in summary:
                    markdown += "\n**Hardware scores:**\n"
                    for hardware, score in summary["hardware_scores"].items():
                        markdown += f"- **{hardware}**: {score:.2f}\n"
                
                markdown += "\n"
            
            # Metrics section
            metrics = hardware_comparison.get("metrics", {})
            if metrics:
                markdown += "### Metrics by Hardware\n\n"
                
                for metric, data in metrics.items():
                    is_higher_better = data.get("is_higher_better", True)
                    markdown += f"#### {metric} ({'higher is better' if is_higher_better else 'lower is better'})\n\n"
                    
                    hardware_results = data.get("hardware_results", {})
                    
                    if hardware_results:
                        # Create table
                        markdown += "| Hardware | Count | Mean | Std Dev | Min | Max |\n"
                        markdown += "|---------|-------|------|---------|-----|-----|\n"
                        
                        for hardware, results in hardware_results.items():
                            markdown += f"| {hardware} | {results.get('count', 0)} | {results.get('mean', 0):.2f} | {results.get('std', 0):.2f} | {results.get('min', 0):.2f} | {results.get('max', 0):.2f} |\n"
                    
                    markdown += "\n"
        
        # Add resource efficiency if available
        if "resource_efficiency" in report_data:
            markdown += "## Resource Efficiency Analysis\n\n"
            
            efficiency_data = report_data["resource_efficiency"]
            
            if "summary" in efficiency_data and efficiency_data["summary"]:
                summary = efficiency_data["summary"]
                markdown += "### Summary\n\n"
                
                if "most_efficient_setup" in summary:
                    markdown += f"- **Most efficient setup**: {summary['most_efficient_setup']}\n\n"
            
            # Efficiency metrics by configuration
            efficiency_metrics = efficiency_data.get("efficiency_metrics", {})
            if efficiency_metrics:
                markdown += "### Efficiency Metrics by Configuration\n\n"
                
                for config_str, data in efficiency_metrics.items():
                    config = data.get("configuration", {})
                    
                    # Format configuration for display
                    model = config.get("model", "unknown")
                    batch_size = config.get("batch_size", "unknown")
                    hardware = config.get("hardware", [])
                    hardware_str = ", ".join(hardware) if isinstance(hardware, list) else hardware
                    
                    markdown += f"#### Configuration: {model}, Batch Size {batch_size}, Hardware: {hardware_str}\n\n"
                    
                    # Add efficiency metrics
                    eff_metrics = data.get("efficiency_metrics", {})
                    if eff_metrics:
                        markdown += "**Efficiency Metrics:**\n"
                        for metric, value in eff_metrics.items():
                            markdown += f"- **{metric}**: {value:.4f}\n"
                    
                    # Add raw metrics
                    raw_metrics = data.get("raw_metrics", {})
                    if raw_metrics:
                        markdown += "\n**Raw Metrics:**\n"
                        for metric, value in raw_metrics.items():
                            markdown += f"- **{metric}**: {value:.2f}\n"
                    
                    markdown += "\n"
        
        # Add time analysis if available
        if "time_analysis" in report_data:
            markdown += "## Performance Over Time\n\n"
            
            time_analysis = report_data["time_analysis"]
            
            for metric, analysis in time_analysis.items():
                markdown += f"### {metric} Over Time\n\n"
                
                if "analysis" in analysis and analysis["analysis"]:
                    a = analysis["analysis"]
                    
                    if "trend" in a:
                        trend = a["trend"]
                        markdown += f"- **Trend**: {trend.capitalize()}\n"
                    
                    if "best_model" in a:
                        best_model = a["best_model"]
                        markdown += f"- **Best fitting model**: {best_model.capitalize()}\n"
                    
                    # Add model details
                    if "linear_regression" in a:
                        lr = a["linear_regression"]
                        markdown += f"- **Linear model**: y = {lr['slope']:.4f}x + {lr['intercept']:.4f} (R² = {lr['r2']:.4f})\n"
                    
                    if "forecast" in a:
                        forecast = a["forecast"]
                        markdown += f"- **Forecast (next 3 points)**: {', '.join([f'{v:.2f}' for v in forecast])}\n"
                
                markdown += "\n"
        
        return markdown
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """
        Generate an HTML format report.
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            HTML report
        """
        # Convert markdown to HTML
        markdown = self._generate_markdown_report(report_data)
        
        # Basic HTML template
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Analysis Report: {report_data['report_type'].capitalize()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3, h4 {{ color: #444; }}
        h1 {{ border-bottom: 2px solid #5a5a5a; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .card {{ border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 20px; }}
        .metrics-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .metric-card {{ flex: 1; min-width: 250px; border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 10px; }}
        .critical {{ border-left: 5px solid #ff4444; }}
        .major {{ border-left: 5px solid #ffbb33; }}
        .minor {{ border-left: 5px solid #ffea00; }}
        .stable {{ border-left: 5px solid #00C851; }}
        .improvement {{ border-left: 5px solid #33b5e5; }}
    </style>
</head>
<body>
    <div class="content">
        <!-- Markdown content will be converted and inserted here -->
        {markdown.replace('\n', '<br>').replace('# ', '<h1>').replace(' #', '</h1>').replace('## ', '<h2>').replace(' ##', '</h2>').replace('### ', '<h3>').replace(' ###', '</h3>').replace('#### ', '<h4>').replace(' ####', '</h4>').replace('- ', '• ').replace('**', '<strong>').replace('**', '</strong>').replace('*', '<em>').replace('*', '</em>')}
    </div>
</body>
</html>
"""
        return html