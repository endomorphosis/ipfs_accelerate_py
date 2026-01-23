#!/usr/bin/env python3
"""
Integrated Analysis System for Distributed Testing Framework

This module provides a unified interface to the result aggregation and analysis system,
integrating various components like service, analysis, coordinator integration, and
data processing pipeline. It offers a high-level API for storing, retrieving, analyzing,
and visualizing test results from the distributed testing framework.

Usage:
    from result_aggregator.integrated_analysis_system import IntegratedAnalysisSystem
    
    # Initialize the system
    analysis_system = IntegratedAnalysisSystem(db_path='./benchmark_db.duckdb')
    
    # Register with a coordinator (optional)
    analysis_system.register_with_coordinator(coordinator)
    
    # Store a test result
    result_id = analysis_system.store_result(test_result)
    
    # Analyze results
    analysis = analysis_system.analyze_results(
        filter_criteria={'test_type': 'benchmark'},
        analysis_types=['trends', 'anomalies', 'patterns']
    )
    
    # Generate comprehensive report
    report = analysis_system.generate_report(format='html', output_path='report.html')
"""

import os
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from pathlib import Path
import threading
import anyio
import concurrent.futures
import warnings
import sys

# Conditional imports for optional dependencies
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    warnings.warn("DuckDB not available. Database functionality will be disabled.")

try:
    import pandas as pd
    import numpy as np
    DATA_ANALYSIS_AVAILABLE = True
except ImportError:
    DATA_ANALYSIS_AVAILABLE = False
    warnings.warn("Pandas/NumPy not available. Data analysis capabilities will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Visualization will be disabled.")

try:
    from scipy import stats
    from scipy.signal import find_peaks
    STATISTICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    STATISTICAL_ANALYSIS_AVAILABLE = False
    warnings.warn("SciPy not available. Statistical analysis will be limited.")

# Core service import (required for most functionality)
try:
    from result_aggregator.service import ResultAggregatorService
except ImportError:
    ResultAggregatorService = None
    warnings.warn("ResultAggregatorService couldn't be imported. Functionality will be limited.")

# Optional analysis functions (require pandas/numpy via result_aggregator.analysis.analysis)
if DATA_ANALYSIS_AVAILABLE:
    try:
        from result_aggregator.analysis.analysis import (
            analyze_trend, detect_anomalies, compare_groups, calculate_efficiency_metrics,
            analyze_workload_distribution, analyze_failure_patterns, analyze_recovery_performance,
            analyze_circuit_breaker_performance, analyze_multi_dimensional_performance,
            analyze_time_series_forecasting
        )
    except ImportError:
        analyze_trend = None
        detect_anomalies = None
        compare_groups = None
        calculate_efficiency_metrics = None
        analyze_workload_distribution = None
        analyze_failure_patterns = None
        analyze_recovery_performance = None
        analyze_circuit_breaker_performance = None
        analyze_multi_dimensional_performance = None
        analyze_time_series_forecasting = None
        warnings.warn(
            "Analysis helpers couldn't be imported. Advanced analysis functionality will be limited."
        )
else:
    analyze_trend = None
    detect_anomalies = None
    compare_groups = None
    calculate_efficiency_metrics = None
    analyze_workload_distribution = None
    analyze_failure_patterns = None
    analyze_recovery_performance = None
    analyze_circuit_breaker_performance = None
    analyze_multi_dimensional_performance = None
    analyze_time_series_forecasting = None

# Coordinator integration (should not depend on pandas/numpy)
try:
    from result_aggregator.coordinator_integration import ResultAggregatorIntegration
except ImportError:
    ResultAggregatorIntegration = None
    warnings.warn(
        "Coordinator integration couldn't be imported. Coordinator registration will be disabled."
    )

try:
    from result_aggregator.pipeline.pipeline import DataSource, ProcessingPipeline
    from result_aggregator.pipeline.transforms import (
        FilterTransform, TimeWindowTransform, AggregateTransform,
        PivotTransform, CalculatedMetricTransform
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    warnings.warn("Pipeline components not available. Data processing pipeline will be disabled.")

try:
    from result_aggregator.ml_detection.ml_anomaly_detector import MLAnomalyDetector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("ML detection components not available. ML-based anomaly detection will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedAnalysisSystem:
    """
    Integrated Analysis System for Distributed Testing Framework.
    
    This class provides a unified interface to the result aggregation and analysis system,
    integrating various components like service, analysis, coordinator integration, and
    data processing pipeline.
    """
    
    def __init__(self, 
                db_path: Optional[str] = None,
                connection = None,
                enable_ml: bool = True,
                enable_visualization: bool = True,
                enable_real_time_analysis: bool = True,
                analysis_interval: timedelta = timedelta(minutes=5),
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Integrated Analysis System.
        
        Args:
            db_path: Path to the DuckDB database
            connection: Existing DuckDB connection (optional)
            enable_ml: Enable machine learning features
            enable_visualization: Enable visualization features
            enable_real_time_analysis: Enable real-time analysis
            analysis_interval: Interval for periodic analysis
            config: Additional configuration options
        
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If database configuration is invalid
        """
        self.config = config or {}
        # Keep flags reflecting user intent; optional components may still be unavailable.
        self.enable_ml = bool(enable_ml)
        self.enable_visualization = enable_visualization and VISUALIZATION_AVAILABLE
        self.enable_real_time_analysis = enable_real_time_analysis
        self.analysis_interval = analysis_interval

        # Initialize attributes early so cleanup is safe even if init fails.
        self.service = None
        self.coordinator_integration = None
        self.ml_anomaly_detector = None
        self.analysis_thread = None
        self.stop_analysis = threading.Event()
        self.notification_handlers = []
        
        # Initialize service
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is required for the IntegratedAnalysisSystem")

        # Default to an in-memory DuckDB database if none was provided.
        # This keeps the system usable for unit tests and lightweight usage.
        if connection is None and db_path is None:
            db_path = ":memory:"

        self.db_path = db_path
        
        self.service = ResultAggregatorService(
            db_path=self.db_path,
            enable_ml=self.enable_ml,
            enable_visualization=self.enable_visualization,
        )

        # Initialize additional components
        if self.enable_ml and ML_AVAILABLE:
            # The ML detector needs either a db_path or a connection.
            # Use the same DB path as the core service.
            self.ml_anomaly_detector = MLAnomalyDetector(db_path=self.db_path)
        
        logger.info("Integrated Analysis System initialized")
    
    def register_with_coordinator(self, coordinator):
        """
        Register with a coordinator for real-time result processing.
        
        Args:
            coordinator: The coordinator instance
        
        Returns:
            The ResultAggregatorIntegration instance
        
        Raises:
            ValueError: If coordinator is invalid
        """
        if coordinator is None:
            raise ValueError("Coordinator cannot be None")

        integration_cls = ResultAggregatorIntegration
        if integration_cls is None:
            try:
                from result_aggregator.coordinator_integration import ResultAggregatorIntegration as integration_cls
            except ImportError as e:
                raise ImportError(
                    "Coordinator integration is not available in this environment"
                ) from e
        
        self.coordinator_integration = integration_cls(
            coordinator=coordinator,
            db_path=self.db_path,
            enable_ml=self.enable_ml,
            enable_visualization=self.enable_visualization,
            enable_real_time_analysis=self.enable_real_time_analysis,
            analysis_interval=self.analysis_interval
        )
        
        # Register with coordinator
        self.coordinator_integration.register_with_coordinator()
        
        # Start the analysis thread if real-time analysis is enabled
        if self.enable_real_time_analysis:
            self._start_analysis_thread()
        
        logger.info(f"Registered with coordinator (ID: {coordinator.coordinator_id if hasattr(coordinator, 'coordinator_id') else 'unknown'})")
        
        return self.coordinator_integration
    
    def _start_analysis_thread(self):
        """Start the background analysis thread."""
        if self.analysis_thread is not None and self.analysis_thread.is_alive():
            logger.warning("Analysis thread is already running")
            return
        
        self.stop_analysis.clear()
        self.analysis_thread = threading.Thread(
            target=self._run_periodic_analysis,
            daemon=True
        )
        self.analysis_thread.start()
        
        logger.info(f"Started periodic analysis thread (interval: {self.analysis_interval})")
    
    def _stop_analysis_thread(self):
        """Stop the background analysis thread."""
        analysis_thread = getattr(self, "analysis_thread", None)
        if analysis_thread is not None and analysis_thread.is_alive():
            self.stop_analysis.set()
            # Ensure the thread is fully stopped before closing DB connections.
            analysis_thread.join()
            logger.info("Stopped periodic analysis thread")
        self.analysis_thread = None
    
    def _run_periodic_analysis(self):
        """Run periodic analysis in the background thread."""
        thread_service = None
        try:
            # DuckDB connections are not safe to share across threads.
            # Create a dedicated service (and DB connection) for this thread.
            if ResultAggregatorService is not None and self.db_path:
                try:
                    thread_service = ResultAggregatorService(
                        db_path=self.db_path,
                        enable_ml=self.enable_ml,
                        enable_visualization=self.enable_visualization,
                    )
                except Exception as e:
                    logger.error(f"Failed to create periodic analysis service: {e}")
                    thread_service = None

            while not self.stop_analysis.is_set():
                try:
                    # Run analysis
                    self._perform_periodic_analysis(service=thread_service)

                    # Sleep for the specified interval
                    self.stop_analysis.wait(self.analysis_interval.total_seconds())
                except Exception as e:
                    logger.error(f"Error in periodic analysis: {e}")
                    # Sleep briefly before retrying
                    time.sleep(10)
        finally:
            if thread_service is not None:
                try:
                    thread_service.close()
                except Exception:
                    pass
    
    def _perform_periodic_analysis(self, service=None):
        """Perform periodic analysis of test results."""
        try:
            analysis_service = service if service is not None else self.service

            if analysis_service is None:
                logger.warning("No service available for periodic analysis")
                return

            # Calculate the time window for analysis
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            # Define the filter criteria
            filter_criteria = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Analyze trends
            trends = analysis_service.analyze_performance_trends(filter_criteria)
            
            # Detect anomalies
            anomalies = analysis_service.detect_anomalies(filter_criteria)
            
            # Generate a summary report
            report = analysis_service.generate_analysis_report(
                filter_criteria=filter_criteria,
                report_type="summary",
                format="json"
            )
            
            # Save the report
            report_path = f"reports/periodic_analysis_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, "w") as f:
                f.write(report)
            
            # Process significant findings
            self._process_significant_findings(trends, anomalies)
            
            logger.info(f"Periodic analysis completed. Report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error in periodic analysis: {e}")
    
    def _process_significant_findings(self, trends, anomalies):
        """
        Process significant findings from analysis and send notifications.
        
        Args:
            trends: Performance trend analysis results
            anomalies: Detected anomalies
        """
        notifications = []
        
        # Process trends
        for metric_name, trend_data in trends.items():
            if "trend" in trend_data and trend_data["trend"] != "stable":
                percent_change = trend_data.get("percent_change", 0)
                
                # Only notify for significant changes
                if abs(percent_change) >= 10:
                    severity = "warning" if abs(percent_change) >= 20 else "info"
                    
                    notifications.append({
                        "type": "trend",
                        "severity": severity,
                        "metric": metric_name,
                        "trend": trend_data["trend"],
                        "percent_change": percent_change,
                        "message": f"{metric_name} {trend_data['trend']} by {percent_change:.1f}%",
                        "details": trend_data
                    })
        
        # Process anomalies
        for anomaly in anomalies:
            severity = "warning" if anomaly["score"] >= 0.8 else "info"
            
            notifications.append({
                "type": "anomaly",
                "severity": severity,
                "score": anomaly["score"],
                "anomaly_type": anomaly["type"],
                "message": f"Anomaly detected (score: {anomaly['score']:.2f}, type: {anomaly['type']})",
                "details": anomaly
            })
        
        # Send notifications
        for notification in notifications:
            self._send_notification(notification)
    
    def _send_notification(self, notification):
        """
        Send a notification to all registered handlers.
        
        Args:
            notification: Notification data
        """
        for handler in self.notification_handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
    
    def register_notification_handler(self, handler):
        """
        Register a notification handler.
        
        Args:
            handler: Function to handle notifications
        
        Returns:
            True if registration was successful
        """
        if callable(handler):
            self.notification_handlers.append(handler)
            logger.info(f"Registered notification handler {handler.__name__ if hasattr(handler, '__name__') else 'anonymous'}")
            return True
        else:
            logger.error("Notification handler must be callable")
            return False
    
    def unregister_notification_handler(self, handler):
        """
        Unregister a notification handler.
        
        Args:
            handler: Function to unregister
        
        Returns:
            True if unregistration was successful
        """
        if handler in self.notification_handlers:
            self.notification_handlers.remove(handler)
            logger.info(f"Unregistered notification handler {handler.__name__ if hasattr(handler, '__name__') else 'anonymous'}")
            return True
        else:
            logger.warning("Notification handler not found")
            return False
    
    def store_result(self, result: Dict[str, Any]) -> int:
        """
        Store a test result in the database.
        
        Args:
            result: Test result data
        
        Returns:
            Result ID
        """
        result_id = self.service.store_result(result)
        
        # Perform real-time analysis if enabled
        if self.enable_real_time_analysis and result_id > 0:
            self._analyze_result(result_id)
        
        return result_id
    
    def _analyze_result(self, result_id: int):
        """
        Perform real-time analysis on a single result.
        
        Args:
            result_id: ID of the result to analyze
        """
        try:
            # Get the result
            result = self.service.get_result(result_id)
            
            if not result:
                logger.warning(f"No result found with ID {result_id}")
                return
            
            # Detect anomalies
            anomalies = self.service._detect_anomalies_for_result(result_id)
            
            # Process anomalies
            for anomaly in anomalies:
                self._send_notification({
                    "type": "anomaly",
                    "severity": "warning" if anomaly["score"] >= 0.8 else "info",
                    "result_id": result_id,
                    "score": anomaly["score"],
                    "anomaly_type": anomaly["type"],
                    "message": f"Anomaly detected in result {result_id} (score: {anomaly['score']:.2f}, type: {anomaly['type']})",
                    "details": anomaly
                })
            
        except Exception as e:
            logger.error(f"Error analyzing result {result_id}: {e}")
    
    def get_result(self, result_id: int) -> Dict[str, Any]:
        """
        Get a test result from the database.
        
        Args:
            result_id: Result ID to retrieve
        
        Returns:
            Test result data
        """
        return self.service.get_result(result_id)
    
    def get_results(self, filter_criteria: Dict[str, Any] = None, 
                   limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get test results from the database based on filter criteria.
        
        Args:
            filter_criteria: Filter criteria for results
            limit: Maximum number of results to return
            offset: Offset for pagination
        
        Returns:
            List of test results
        """
        return self.service.get_results(filter_criteria, limit, offset)
    
    def analyze_results(self, 
                       filter_criteria: Dict[str, Any] = None,
                       analysis_types: List[str] = None,
                       metrics: List[str] = None,
                       group_by: str = None,
                       time_period_days: int = 30) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on test results.
        
        Args:
            filter_criteria: Filter criteria for results
            analysis_types: Types of analysis to perform
            metrics: Metrics to analyze
            group_by: Column to group results by
            time_period_days: Number of days to look back
        
        Returns:
            Analysis results
        """
        if not analysis_types:
            analysis_types = ["trends", "anomalies", "workload", "failures", "performance"]
        
        if not metrics:
            metrics = ["latency", "throughput", "memory_usage", "cpu_usage"]
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        
        # Update filter criteria with time range
        if filter_criteria is None:
            filter_criteria = {}
        
        if "start_time" not in filter_criteria:
            filter_criteria["start_time"] = cutoff_date.isoformat()
        
        # Get results
        results = self.get_results(filter_criteria=filter_criteria, limit=10000)
        
        # Prepare analysis results container
        analysis_results = {
            "metadata": {
                "filter_criteria": filter_criteria,
                "analysis_types": analysis_types,
                "metrics": metrics,
                "group_by": group_by,
                "time_period_days": time_period_days,
                "result_count": len(results),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Perform trend analysis
        if "trends" in analysis_types:
            trends = self.service.analyze_performance_trends(
                filter_criteria=filter_criteria,
                metrics=metrics
            )
            analysis_results["trends"] = trends
        
        # Perform anomaly detection
        if "anomalies" in analysis_types:
            anomalies = self.service.detect_anomalies(filter_criteria=filter_criteria)
            analysis_results["anomalies"] = anomalies
        
        # Convert results to pandas DataFrame for additional analysis
        if DATA_ANALYSIS_AVAILABLE and results:
            # Extract key data
            data = []
            for result in results:
                # Extract metrics
                metrics_data = result.get("metrics", {})
                
                # Create a base record with result metadata
                record = {
                    "result_id": result.get("id"),
                    "task_id": result.get("task_id"),
                    "worker_id": result.get("worker_id"),
                    "timestamp": result.get("timestamp"),
                    "type": result.get("type"),
                    "status": result.get("status"),
                    "duration": result.get("duration")
                }
                
                # Extract details
                details = result.get("details", {})
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except json.JSONDecodeError:
                        details = {}
                
                # Add relevant details
                if "hardware" in details:
                    record["hardware"] = details["hardware"]
                if "model" in details:
                    record["model"] = details["model"]
                if "batch_size" in details:
                    record["batch_size"] = details["batch_size"]
                
                # Extract metrics
                for metric_name, metric_value in metrics_data.items():
                    if isinstance(metric_value, dict):
                        record[metric_name] = metric_value.get("value", 0)
                    else:
                        record[metric_name] = metric_value
                
                data.append(record)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Analyze workload distribution
            if "workload" in analysis_types:
                workload_analysis = analyze_workload_distribution(data)
                analysis_results["workload_distribution"] = workload_analysis
            
            # Analyze failure patterns
            if "failures" in analysis_types:
                failure_analysis = analyze_failure_patterns(data)
                analysis_results["failure_patterns"] = failure_analysis
            
            # Analyze multi-dimensional performance
            if "performance" in analysis_types and group_by:
                dimensions = [group_by]
                if "hardware" in df.columns:
                    dimensions.append("hardware")
                if "model" in df.columns:
                    dimensions.append("model")
                if "batch_size" in df.columns:
                    dimensions.append("batch_size")
                
                performance_analysis = analyze_multi_dimensional_performance(data, dimensions)
                analysis_results["performance_analysis"] = performance_analysis
            
            # Analyze recovery performance if data is available
            if "recovery" in analysis_types and any("recovery" in col for col in df.columns):
                recovery_data = [r for r in data if "recovery_strategy" in r]
                if recovery_data:
                    recovery_analysis = analyze_recovery_performance(recovery_data)
                    analysis_results["recovery_performance"] = recovery_analysis
            
            # Analyze circuit breaker performance if data is available
            if "circuit_breaker" in analysis_types and any("circuit_breaker" in col for col in df.columns):
                circuit_breaker_data = [r for r in data if "circuit_breaker_state" in r]
                if circuit_breaker_data:
                    circuit_breaker_analysis = analyze_circuit_breaker_performance(circuit_breaker_data)
                    analysis_results["circuit_breaker_performance"] = circuit_breaker_analysis
            
            # Time series forecasting
            if "forecast" in analysis_types and STATISTICAL_ANALYSIS_AVAILABLE:
                forecasts = {}
                
                # Group by metric and perform forecasting
                for metric in metrics:
                    if metric in df.columns:
                        # Create time series
                        ts_data = df.sort_values("timestamp")
                        if len(ts_data) >= 5:  # Need at least 5 points for forecasting
                            ts = []
                            for _, row in ts_data.iterrows():
                                ts.append({
                                    "timestamp": row["timestamp"],
                                    "value": row[metric]
                                })
                            
                            # Perform forecasting
                            forecast = analyze_time_series_forecasting(ts)
                            forecasts[metric] = forecast
                
                if forecasts:
                    analysis_results["forecasts"] = forecasts
        
        return analysis_results
    
    def generate_report(self, 
                       analysis_results: Dict[str, Any] = None,
                       filter_criteria: Dict[str, Any] = None,
                       report_type: str = "comprehensive",
                       format: str = "markdown",
                       output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report.
        
        Args:
            analysis_results: Pre-computed analysis results (if None, perform analysis)
            filter_criteria: Filter criteria for results
            report_type: Type of report
            format: Report format (markdown, html, json)
            output_path: Path to save the report
        
        Returns:
            Generated report
        """
        # Perform analysis if not provided
        if analysis_results is None:
            analysis_results = self.analyze_results(filter_criteria=filter_criteria)
        
        # Generate appropriate report based on type and format
        if report_type == "performance":
            # Prefer the dedicated performance report API if available.
            if hasattr(self.service, "generate_performance_report"):
                report = self.service.generate_performance_report(
                    report_type="comprehensive",
                    filter_criteria=filter_criteria,
                    format=format,
                    time_period="30d"
                )
            else:
                # Fall back to basic report
                report = self.service.generate_analysis_report(
                    filter_criteria=filter_criteria,
                    report_type="performance",
                    format=format
                )
        else:
            # Use the standard report generation
            report = self.service.generate_analysis_report(
                filter_criteria=filter_criteria,
                report_type=report_type,
                format=format
            )
        
        # Save the report if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w") as f:
                f.write(report)
            
            logger.info(f"Saved report to {output_path}")
        
        return report
    
    def visualize_results(self,
                         visualization_type: str,
                         data: Any = None,
                         filter_criteria: Dict[str, Any] = None,
                         metrics: List[str] = None,
                         output_path: Optional[str] = None) -> bool:
        """
        Generate visualizations for test results.
        
        Args:
            visualization_type: Type of visualization
            data: Pre-computed data to visualize
            filter_criteria: Filter criteria for results
            metrics: Metrics to visualize
            output_path: Path to save visualization
        
        Returns:
            True if visualization was successful
        """
        if not self.enable_visualization:
            logger.warning("Visualization is disabled")
            return False
        
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available. Visualization not possible.")
            return False
        
        # Get data if not provided
        if data is None:
            if visualization_type in ["trends", "time_series"]:
                data = self.service.analyze_performance_trends(
                    filter_criteria=filter_criteria,
                    metrics=metrics
                )
            elif visualization_type == "anomalies":
                data = self.service.detect_anomalies(filter_criteria=filter_criteria)
            else:
                # Get raw results
                data = self.get_results(filter_criteria=filter_criteria, limit=10000)
        
        # Select the appropriate visualization function
        if visualization_type == "trends":
            success = self._visualize_trends(data, metrics, output_path)
        elif visualization_type == "time_series":
            success = self._visualize_time_series(data, metrics, output_path)
        elif visualization_type == "anomalies":
            success = self._visualize_anomalies(data, output_path)
        elif visualization_type == "performance_comparison":
            success = self._visualize_performance_comparison(data, metrics, output_path)
        elif visualization_type == "workload_distribution":
            success = self._visualize_workload_distribution(data, output_path)
        elif visualization_type == "failure_patterns":
            success = self._visualize_failure_patterns(data, output_path)
        elif visualization_type == "circuit_breaker":
            success = self._visualize_circuit_breaker(data, output_path)
        else:
            logger.warning(f"Unsupported visualization type: {visualization_type}")
            return False
        
        return success
    
    def _visualize_trends(self, trend_data, metrics, output_path):
        """Visualize performance trends."""
        try:
            if not metrics:
                metrics = list(trend_data.keys())
            
            # Set up the figure
            fig, axes = plt.subplots(
                nrows=len(metrics),
                figsize=(12, 5 * len(metrics)),
                squeeze=False
            )
            
            # Plot each metric
            for i, metric in enumerate(metrics):
                if metric not in trend_data:
                    continue
                
                ax = axes[i, 0]
                
                # Get time series data
                time_series = trend_data[metric].get("time_series", [])
                if not time_series:
                    continue
                
                # Extract timestamps and values
                timestamps = [datetime.fromisoformat(ts["timestamp"].replace('Z', '+00:00'))
                           if isinstance(ts["timestamp"], str) else ts["timestamp"]
                           for ts in time_series]
                values = [ts["value"] for ts in time_series]
                moving_avg = [ts.get("moving_avg") for ts in time_series]
                
                # Plot raw values
                ax.plot(timestamps, values, 'o-', label='Value', alpha=0.6)
                
                # Plot moving average if available
                valid_indices = [i for i, v in enumerate(moving_avg) if v is not None]
                if valid_indices:
                    valid_timestamps = [timestamps[i] for i in valid_indices]
                    valid_moving_avg = [moving_avg[i] for i in valid_indices]
                    ax.plot(valid_timestamps, valid_moving_avg, 'r-', label='Moving Avg', linewidth=2)
                
                # Add trend information
                trend_direction = trend_data[metric].get("trend", "unknown")
                percent_change = trend_data[metric].get("percent_change", 0)
                
                # Add labels and title
                ax.set_title(f"{metric} Trend ({trend_direction}, {percent_change:.1f}% change)")
                ax.set_xlabel('Time')
                ax.set_ylabel(metric)
                ax.legend()
                
                # Format x-axis dates
                fig.autofmt_xdate()
            
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved trend visualization to {output_path}")
                plt.close(fig)
                return True
            else:
                plt.close(fig)
                return True
                
        except Exception as e:
            logger.error(f"Error visualizing trends: {e}")
            return False
    
    def _visualize_time_series(self, time_series_data, metrics, output_path):
        """Visualize time series data with forecasting."""
        try:
            if not metrics:
                metrics = list(time_series_data.keys())
            
            # Set up the figure
            fig, axes = plt.subplots(
                nrows=len(metrics),
                figsize=(12, 5 * len(metrics)),
                squeeze=False
            )
            
            # Plot each metric
            for i, metric in enumerate(metrics):
                if metric not in time_series_data:
                    continue
                
                ax = axes[i, 0]
                
                # Get time series data
                time_series = time_series_data[metric].get("time_series", [])
                if not time_series:
                    continue
                
                # Extract timestamps and values
                timestamps = [datetime.fromisoformat(ts["timestamp"].replace('Z', '+00:00'))
                           if isinstance(ts["timestamp"], str) else ts["timestamp"]
                           for ts in time_series]
                values = [ts["value"] for ts in time_series]
                
                # Plot raw values
                ax.plot(timestamps, values, 'o-', label='Value', alpha=0.6)
                
                # Add forecast if available
                if "forecast" in time_series_data[metric]:
                    forecast = time_series_data[metric]["forecast"]
                    if forecast.get("success", False):
                        # Get the forecast values
                        forecast_values = forecast.get("forecast", [])
                        
                        # Create forecast timestamps (extending from last timestamp)
                        last_timestamp = timestamps[-1]
                        forecast_timestamps = []
                        
                        for j in range(len(forecast_values)):
                            # Assume daily frequency
                            forecast_timestamps.append(last_timestamp + timedelta(days=j+1))
                        
                        # Plot forecast
                        ax.plot(forecast_timestamps, forecast_values, 'g--', label='Forecast')
                        
                        # Add confidence intervals if available
                        if "confidence_intervals" in forecast:
                            lower_bounds = forecast["confidence_intervals"].get("lower", [])
                            upper_bounds = forecast["confidence_intervals"].get("upper", [])
                            
                            ax.fill_between(
                                forecast_timestamps,
                                lower_bounds,
                                upper_bounds,
                                color='g',
                                alpha=0.2,
                                label='95% Confidence Interval'
                            )
                
                # Add labels and title
                ax.set_title(f"{metric} Time Series")
                ax.set_xlabel('Time')
                ax.set_ylabel(metric)
                ax.legend()
                
                # Format x-axis dates
                fig.autofmt_xdate()
            
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved time series visualization to {output_path}")
                plt.close(fig)
                return True
            else:
                plt.close(fig)
                return True
                
        except Exception as e:
            logger.error(f"Error visualizing time series: {e}")
            return False
    
    def _visualize_anomalies(self, anomaly_data, output_path):
        """Visualize anomalies."""
        try:
            if not anomaly_data:
                logger.warning("No anomaly data to visualize")
                return False
            
            # Set up the figure
            fig, axes = plt.subplots(
                nrows=1,
                figsize=(12, 8)
            )
            
            # Extract anomaly scores and timestamps
            scores = [anomaly.get("score", 0) for anomaly in anomaly_data]
            timestamps = [anomaly.get("detection_time", i) for i, anomaly in enumerate(anomaly_data)]
            
            # Convert timestamps to datetime if they are strings
            if timestamps and isinstance(timestamps[0], str):
                timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00'))
                           for ts in timestamps]
            
            # Plot anomaly scores
            axes.plot(timestamps, scores, 'o-', color='blue', alpha=0.7)
            
            # Add threshold line
            axes.axhline(y=0.7, linestyle='--', color='orange', label='Warning Threshold')
            axes.axhline(y=0.8, linestyle='--', color='red', label='Critical Threshold')
            
            # Highlight critical anomalies
            critical_indices = [i for i, score in enumerate(scores) if score >= 0.8]
            warning_indices = [i for i, score in enumerate(scores) if 0.7 <= score < 0.8]
            
            if critical_indices:
                critical_x = [timestamps[i] for i in critical_indices]
                critical_y = [scores[i] for i in critical_indices]
                axes.scatter(critical_x, critical_y, color='red', s=100, label='Critical Anomalies')
            
            if warning_indices:
                warning_x = [timestamps[i] for i in warning_indices]
                warning_y = [scores[i] for i in warning_indices]
                axes.scatter(warning_x, warning_y, color='orange', s=80, label='Warning Anomalies')
            
            # Add labels and title
            axes.set_title('Anomaly Detection Results')
            axes.set_xlabel('Time')
            axes.set_ylabel('Anomaly Score')
            axes.set_ylim(0, 1.1)
            axes.legend()
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved anomaly visualization to {output_path}")
                plt.close(fig)
                return True
            else:
                plt.close(fig)
                return True
                
        except Exception as e:
            logger.error(f"Error visualizing anomalies: {e}")
            return False
    
    def _visualize_performance_comparison(self, data, metrics, output_path):
        """Visualize performance comparison across hardware platforms or models."""
        try:
            if not DATA_ANALYSIS_AVAILABLE:
                logger.warning("Pandas/NumPy not available. Performance comparison visualization not possible.")
                return False
            
            # Convert data to DataFrame if it's a list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                logger.warning("Invalid data format for performance comparison visualization")
                return False
            
            # Check if we have hardware and metrics columns
            if "hardware" not in df.columns:
                logger.warning("Hardware column not found in data")
                return False
            
            if not metrics:
                # Try to infer metrics from columns
                metrics = [col for col in df.columns if col in ["latency", "throughput", "memory_usage", "cpu_usage"]]
                
                if not metrics:
                    logger.warning("No metrics found in data")
                    return False
            
            # Set up the figure
            fig, axes = plt.subplots(
                nrows=len(metrics),
                figsize=(12, 5 * len(metrics)),
                squeeze=False
            )
            
            # Plot each metric
            for i, metric in enumerate(metrics):
                if metric not in df.columns:
                    continue
                
                ax = axes[i, 0]
                
                # Group by hardware and calculate mean, std
                grouped = df.groupby("hardware")[metric].agg(["mean", "std"]).reset_index()
                
                # Sort by mean value
                grouped = grouped.sort_values("mean", ascending=False)
                
                # Create bar plot
                x = range(len(grouped))
                ax.bar(x, grouped["mean"], yerr=grouped["std"], capsize=5, alpha=0.7)
                
                # Add labels
                ax.set_xticks(x)
                ax.set_xticklabels(grouped["hardware"], rotation=45, ha="right")
                
                # Add title and labels
                ax.set_title(f"{metric} by Hardware")
                ax.set_xlabel("Hardware")
                ax.set_ylabel(metric)
                
                # Add grid
                ax.grid(axis="y", linestyle="--", alpha=0.7)
            
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved performance comparison visualization to {output_path}")
                plt.close(fig)
                return True
            else:
                plt.close(fig)
                return True
                
        except Exception as e:
            logger.error(f"Error visualizing performance comparison: {e}")
            return False
    
    def _visualize_workload_distribution(self, workload_data, output_path):
        """Visualize workload distribution across worker nodes."""
        try:
            if not VISUALIZATION_AVAILABLE:
                logger.warning("Matplotlib not available. Workload distribution visualization not possible.")
                return False
            
            if not workload_data or "worker_stats" not in workload_data:
                logger.warning("Invalid workload data for visualization")
                return False
            
            # Extract worker statistics
            worker_stats = workload_data["worker_stats"]
            workers = list(worker_stats.keys())
            success_rates = [stats["success_rate"] for stats in worker_stats.values()]
            total_tasks = [stats["total_tasks"] for stats in worker_stats.values()]
            mean_durations = [stats["mean_duration"] for stats in worker_stats.values()]
            
            # Create a figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            
            # Plot total tasks per worker
            y_pos = range(len(workers))
            ax1.barh(y_pos, total_tasks, align='center')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(workers)
            ax1.set_xlabel('Total Tasks')
            ax1.set_title('Workload Distribution')
            
            # Plot success rates
            ax2.barh(y_pos, success_rates, align='center')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(workers)
            ax2.set_xlabel('Success Rate (%)')
            ax2.set_title('Worker Success Rates')
            
            # Plot mean durations
            ax3.barh(y_pos, mean_durations, align='center')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(workers)
            ax3.set_xlabel('Mean Duration (s)')
            ax3.set_title('Task Durations by Worker')
            
            # Add distribution statistics to the plot
            if "distribution_stats" in workload_data:
                stats = workload_data["distribution_stats"]
                stats_text = (
                    f"Total Workers: {stats.get('total_workers', 0)}\n"
                    f"Total Tasks: {stats.get('total_tasks', 0)}\n"
                    f"Mean Tasks per Worker: {stats.get('mean_tasks_per_worker', 0):.2f}\n"
                    f"Task Distribution Inequality (Gini): {stats.get('gini_coefficient', 0):.2f}\n"
                )
                
                ax1.text(0.98, 0.05, stats_text, transform=ax1.transAxes, 
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved workload distribution visualization to {output_path}")
                plt.close(fig)
                return True
            else:
                plt.close(fig)
                return True
                
        except Exception as e:
            logger.error(f"Error visualizing workload distribution: {e}")
            return False
    
    def _visualize_failure_patterns(self, failure_data, output_path):
        """Visualize failure patterns."""
        try:
            if not VISUALIZATION_AVAILABLE:
                logger.warning("Matplotlib not available. Failure patterns visualization not possible.")
                return False
            
            if not failure_data or "failure_counts" not in failure_data:
                logger.warning("Invalid failure data for visualization")
                return False
            
            # Extract failure counts
            failure_counts = failure_data["failure_counts"]
            error_types = list(failure_counts.keys())
            counts = list(failure_counts.values())
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Plot failure counts as bar chart
            y_pos = range(len(error_types))
            ax1.barh(y_pos, counts, align='center')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(error_types)
            ax1.set_xlabel('Count')
            ax1.set_title('Failure Counts by Error Type')
            
            # Plot failure counts as pie chart
            ax2.pie(counts, labels=error_types, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax2.set_title('Failure Distribution')
            
            # Add failure correlations if available
            if "failure_correlations" in failure_data:
                correlations = failure_data["failure_correlations"]
                if correlations:
                    corr_text = "Failure Correlations:\n\n"
                    
                    for i, corr in enumerate(correlations[:5]):  # Limit to top 5
                        corr_type = corr.get("type", "unknown")
                        total_failures = corr.get("total_failures", 0)
                        
                        if corr_type == "worker_issue":
                            worker_id = corr.get("worker_id", "unknown")
                            corr_text += f"{i+1}. Worker {worker_id}: {total_failures} failures\n"
                        elif corr_type == "test_type_issue":
                            test_type = corr.get("test_type", "unknown")
                            corr_text += f"{i+1}. Test type {test_type}: {total_failures} failures\n"
                        elif corr_type == "temporal_issue":
                            hour = corr.get("hour", 0)
                            corr_text += f"{i+1}. Hour {hour}: {total_failures} failures\n"
                    
                    fig.text(0.98, 0.05, corr_text, verticalalignment='bottom', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved failure patterns visualization to {output_path}")
                plt.close(fig)
                return True
            else:
                plt.close(fig)
                return True
                
        except Exception as e:
            logger.error(f"Error visualizing failure patterns: {e}")
            return False
    
    def _visualize_circuit_breaker(self, circuit_breaker_data, output_path):
        """Visualize circuit breaker performance."""
        try:
            if not VISUALIZATION_AVAILABLE:
                logger.warning("Matplotlib not available. Circuit breaker visualization not possible.")
                return False
            
            if not circuit_breaker_data or "transition_stats" not in circuit_breaker_data:
                logger.warning("Invalid circuit breaker data for visualization")
                return False
            
            # Extract circuit breaker statistics
            transition_stats = circuit_breaker_data["transition_stats"]
            
            # Create a figure with multiple subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            
            # Plot transition counts
            transition_counts = transition_stats.get("transition_counts", {})
            transitions = list(transition_counts.keys())
            counts = list(transition_counts.values())
            
            y_pos = range(len(transitions))
            ax1.barh(y_pos, counts, align='center')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(transitions)
            ax1.set_xlabel('Count')
            ax1.set_title('Circuit Breaker Transitions')
            
            # Plot recovery success rate
            recovery_success_rate = transition_stats.get("recovery_success_rate", 0)
            colors = ['green' if recovery_success_rate >= 75 else 
                    'yellow' if recovery_success_rate >= 50 else 'red']
            
            ax2.bar(['Recovery Success Rate'], [recovery_success_rate], color=colors)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Circuit Breaker Recovery Success Rate')
            
            # Add success/failure counts
            for i, v in enumerate([recovery_success_rate]):
                ax2.text(i, v + 3, f"{v:.1f}%", ha='center')
            
            # Plot parameter trends if available
            if "parameter_stats" in circuit_breaker_data:
                param_stats = circuit_breaker_data["parameter_stats"]
                
                # Add parameter trends
                threshold_trend = param_stats.get("threshold_trend", "stable")
                timeout_trend = param_stats.get("timeout_trend", "stable")
                
                threshold_color = 'green' if threshold_trend == "increasing" else 'red' if threshold_trend == "decreasing" else 'blue'
                timeout_color = 'green' if timeout_trend == "increasing" else 'red' if timeout_trend == "decreasing" else 'blue'
                
                ax3.bar([0], [1], color=threshold_color, label=f'Threshold Trend: {threshold_trend}')
                ax3.bar([1], [1], color=timeout_color, label=f'Timeout Trend: {timeout_trend}')
                
                # Remove y-axis and set custom x-ticks
                ax3.set_yticks([])
                ax3.set_xticks([0, 1])
                ax3.set_xticklabels(['Failure Threshold', 'Recovery Timeout'])
                
                ax3.set_title('Circuit Breaker Parameter Trends')
                ax3.legend()
                
                # Add parameter stats to the plot
                stats_text = (
                    f"Changes:\n"
                    f"- Threshold Changes: {param_stats.get('threshold_changes', 0)}\n"
                    f"- Timeout Changes: {param_stats.get('timeout_changes', 0)}\n\n"
                    f"Current Values:\n"
                    f"- Threshold: {param_stats.get('current_threshold', 'N/A')}\n"
                    f"- Timeout: {param_stats.get('current_timeout', 'N/A')} seconds\n"
                )
                
                ax3.text(0.98, 0.05, stats_text, transform=ax3.transAxes, 
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved circuit breaker visualization to {output_path}")
                plt.close(fig)
                return True
            else:
                plt.close(fig)
                return True
                
        except Exception as e:
            logger.error(f"Error visualizing circuit breaker: {e}")
            return False
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """
        Clean up old data from the database.
        
        Args:
            days: Number of days to keep (data older than this will be deleted)
        
        Returns:
            Number of deleted records
        """
        return self.service.cleanup_old_data(days)
    
    def close(self):
        """Close all connections and stop background processes."""
        try:
            # Stop analysis thread
            if hasattr(self, "analysis_thread"):
                self._stop_analysis_thread()
            
            # Unregister from coordinator
            if getattr(self, "coordinator_integration", None):
                self.coordinator_integration.close()
            
            # Close service
            if getattr(self, "service", None):
                self.service.close()
            
            logger.info("Integrated Analysis System closed")
            
        except Exception as e:
            logger.error(f"Error closing Integrated Analysis System: {e}")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        # Avoid complex teardown during interpreter finalization.
        try:
            if hasattr(sys, "is_finalizing") and sys.is_finalizing():
                return
        except Exception:
            return
        self.close()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Analysis System for Distributed Testing Framework')
    
    parser.add_argument('--db-path', type=str, default='./benchmark_db.duckdb',
                       help='Path to DuckDB database file')
    
    # Analysis options
    parser.add_argument('--analyze', action='store_true',
                       help='Perform analysis on test results')
    parser.add_argument('--filter', type=str,
                       help='JSON filter criteria for analysis')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to look back for analysis')
    
    # Report options
    parser.add_argument('--report', action='store_true',
                       help='Generate report')
    parser.add_argument('--report-type', type=str, default='comprehensive',
                       choices=['comprehensive', 'performance', 'summary', 'anomaly'],
                       help='Type of report to generate')
    parser.add_argument('--format', type=str, default='markdown',
                       choices=['markdown', 'html', 'json'],
                       help='Format of the report')
    parser.add_argument('--output', type=str,
                       help='Path to save the report')
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--viz-type', type=str, default='trends',
                       choices=['trends', 'time_series', 'anomalies', 'performance_comparison',
                              'workload_distribution', 'failure_patterns', 'circuit_breaker'],
                       help='Type of visualization to generate')
    parser.add_argument('--viz-output', type=str,
                       help='Path to save the visualization')
    
    # Cleanup options
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old data')
    parser.add_argument('--keep-days', type=int, default=90,
                       help='Number of days to keep when cleaning up old data')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = IntegratedAnalysisSystem(db_path=args.db_path)
        
        # Parse filter criteria
        filter_criteria = None
        if args.filter:
            try:
                filter_criteria = json.loads(args.filter)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON filter criteria: {args.filter}")
                return 1
        
        # Perform analysis
        if args.analyze:
            analysis_results = system.analyze_results(
                filter_criteria=filter_criteria,
                time_period_days=args.days
            )
            
            print(json.dumps(analysis_results, indent=2))
        
        # Generate report
        if args.report:
            report = system.generate_report(
                filter_criteria=filter_criteria,
                report_type=args.report_type,
                format=args.format,
                output_path=args.output
            )
            
            if not args.output:
                print(report)
        
        # Generate visualization
        if args.visualize:
            success = system.visualize_results(
                visualization_type=args.viz_type,
                filter_criteria=filter_criteria,
                output_path=args.viz_output
            )
            
            if not success:
                logger.error(f"Failed to generate {args.viz_type} visualization")
                return 1
        
        # Clean up old data
        if args.cleanup:
            deleted_count = system.cleanup_old_data(days=args.keep_days)
            logger.info(f"Cleaned up {deleted_count} old records")
        
        # Close system
        system.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())