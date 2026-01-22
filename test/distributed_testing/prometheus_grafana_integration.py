"""
Prometheus and Grafana integration for the Distributed Testing Framework.

This module provides the integration between the ML-based anomaly detection,
the Distributed Testing Framework metrics, and external monitoring systems
(Prometheus and Grafana).

It handles:
1. Starting and configuring the ML anomaly detection
2. Collecting metrics from the DTF
3. Exposing metrics via Prometheus HTTP endpoint
4. Managing Grafana dashboard generation and updates
"""

import os
import time
import json
import logging
import threading
import requests
from typing import Dict, List, Optional, Any, Union, Tuple
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
from prometheus_client.registry import CollectorRegistry

from distributed_testing.ml_anomaly_detection import MLAnomalyDetection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("prometheus_grafana_integration")

class PrometheusGrafanaIntegration:
    """Integration between the DTF, ML anomaly detection, Prometheus and Grafana."""
    
    def __init__(
        self,
        prometheus_port: int = 8000,
        prometheus_endpoint: str = "/metrics",
        grafana_url: Optional[str] = None,
        grafana_api_key: Optional[str] = None,
        prometheus_url: Optional[str] = None,
        metrics_collection_interval: int = 30,
        anomaly_detection_interval: int = 300,
        dashboard_update_interval: int = 3600,
        metric_patterns: Optional[List[str]] = None,
        ml_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Prometheus and Grafana integration.
        
        Args:
            prometheus_port: Port to expose Prometheus metrics on
            prometheus_endpoint: Endpoint for Prometheus metrics
            grafana_url: Base URL for Grafana API
            grafana_api_key: API key for Grafana
            prometheus_url: URL for Prometheus API (used for querying data)
            metrics_collection_interval: Interval (seconds) for collecting metrics
            anomaly_detection_interval: Interval (seconds) for running anomaly detection
            dashboard_update_interval: Interval (seconds) for updating dashboards
            metric_patterns: List of metric name patterns to monitor
            ml_config: Configuration for the ML anomaly detection module
        """
        self.prometheus_port = prometheus_port
        self.prometheus_endpoint = prometheus_endpoint
        self.grafana_url = grafana_url
        self.grafana_api_key = grafana_api_key
        self.prometheus_url = prometheus_url
        self.metrics_collection_interval = metrics_collection_interval
        self.anomaly_detection_interval = anomaly_detection_interval
        self.dashboard_update_interval = dashboard_update_interval
        
        # Default metric patterns to monitor if none provided
        self.metric_patterns = metric_patterns or [
            "dtf_worker_",
            "dtf_task_",
            "dtf_coordinator_",
            "dtf_resource_",
            "dtf_network_",
        ]
        
        # Initialize Prometheus registry and metrics
        self.registry = CollectorRegistry()
        self.metrics = self._initialize_metrics()
        
        # Initialize ML anomaly detection with default or provided config
        default_ml_config = {
            "algorithms": ["isolation_forest", "dbscan", "threshold"],
            "forecasting": ["arima", "prophet", "exponential_smoothing"],
            "visualization": True,
            "model_persistence_dir": "models/anomaly_detection",
            "confidence_threshold": 0.85,
        }
        self.ml_config = {**default_ml_config, **(ml_config or {})}
        self.ml_detector = MLAnomalyDetection(**self.ml_config)
        
        # State variables
        self.running = False
        self.threads = []
        self.metrics_data = {}  # Store recent metrics for anomaly detection
        self.anomalies = {}     # Store detected anomalies
        
        # Grafana dashboard state
        self.dashboards = {}
        
        logger.info(f"Initialized Prometheus/Grafana integration on port {prometheus_port}")
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize and return Prometheus metrics."""
        metrics = {
            # Worker metrics
            "worker_count": Gauge(
                "dtf_worker_count", "Number of active workers", 
                registry=self.registry
            ),
            "worker_task_throughput": Gauge(
                "dtf_worker_task_throughput", 
                "Tasks processed per minute by worker",
                ["worker_id", "worker_type"], 
                registry=self.registry
            ),
            "worker_resource_usage": Gauge(
                "dtf_worker_resource_usage", 
                "Resource usage percentage by worker",
                ["worker_id", "resource_type"], 
                registry=self.registry
            ),
            
            # Task metrics
            "task_execution_time": Histogram(
                "dtf_task_execution_time", 
                "Task execution time in seconds",
                ["task_type", "worker_type"],
                buckets=(1, 5, 10, 30, 60, 120, 300, 600),
                registry=self.registry
            ),
            "task_queue_length": Gauge(
                "dtf_task_queue_length", 
                "Number of tasks in queue",
                ["task_type", "priority"], 
                registry=self.registry
            ),
            "task_success_rate": Gauge(
                "dtf_task_success_rate", 
                "Percentage of tasks completed successfully",
                ["task_type", "worker_type"], 
                registry=self.registry
            ),
            
            # Coordinator metrics
            "coordinator_health": Gauge(
                "dtf_coordinator_health", 
                "Health score of coordinator (0-100)",
                ["coordinator_id"], 
                registry=self.registry
            ),
            "coordinator_leadership": Gauge(
                "dtf_coordinator_leadership", 
                "Leadership status (1=leader, 0=follower)",
                ["coordinator_id"], 
                registry=self.registry
            ),
            
            # Resource metrics
            "resource_allocation_efficiency": Gauge(
                "dtf_resource_allocation_efficiency", 
                "Efficiency of resource allocation (0-100)",
                ["resource_type"], 
                registry=self.registry
            ),
            
            # Network metrics
            "network_latency": Histogram(
                "dtf_network_latency", 
                "Network latency between components in ms",
                ["source", "destination"],
                buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
                registry=self.registry
            ),
            
            # Anomaly detection metrics
            "anomaly_count": Gauge(
                "dtf_anomaly_count", 
                "Number of anomalies detected",
                ["metric_name", "algorithm"], 
                registry=self.registry
            ),
            "anomaly_severity": Gauge(
                "dtf_anomaly_severity", 
                "Severity of detected anomalies (0-100)",
                ["metric_name", "algorithm"], 
                registry=self.registry
            ),
            
            # Forecasting metrics
            "forecast_accuracy": Gauge(
                "dtf_forecast_accuracy", 
                "Accuracy of forecast predictions (0-100)",
                ["metric_name", "algorithm"], 
                registry=self.registry
            ),
        }
        return metrics
    
    def start(self):
        """Start the integration service."""
        if self.running:
            logger.warning("Integration already running")
            return
        
        # Start Prometheus HTTP server
        start_http_server(self.prometheus_port, registry=self.registry)
        logger.info(f"Started Prometheus HTTP server on port {self.prometheus_port}")
        
        # Start background threads
        self.running = True
        
        # Thread for collecting metrics
        metrics_thread = threading.Thread(
            target=self._metrics_collection_loop,
            daemon=True
        )
        metrics_thread.start()
        self.threads.append(metrics_thread)
        
        # Thread for anomaly detection
        anomaly_thread = threading.Thread(
            target=self._anomaly_detection_loop,
            daemon=True
        )
        anomaly_thread.start()
        self.threads.append(anomaly_thread)
        
        # Thread for dashboard updates (if Grafana is configured)
        if self.grafana_url and self.grafana_api_key:
            dashboard_thread = threading.Thread(
                target=self._dashboard_update_loop,
                daemon=True
            )
            dashboard_thread.start()
            self.threads.append(dashboard_thread)
            
        logger.info("All integration threads started")
    
    def stop(self):
        """Stop the integration service."""
        self.running = False
        # Wait for threads to finish (with timeout)
        for thread in self.threads:
            thread.join(timeout=5)
        
        logger.info("Integration service stopped")
    
    def _metrics_collection_loop(self):
        """Background loop for collecting metrics from the DTF."""
        while self.running:
            try:
                # In a real implementation, this would collect metrics from
                # the Distributed Testing Framework through its API or directly
                self._collect_metrics_from_dtf()
                
                # Update Prometheus metrics based on collected data
                self._update_prometheus_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {str(e)}")
            
            # Sleep until next collection
            time.sleep(self.metrics_collection_interval)
    
    def _collect_metrics_from_dtf(self):
        """
        Collect metrics from the Distributed Testing Framework.
        In a real implementation, this would connect to the DTF's 
        internal metrics system, database, or API.
        """
        # Placeholder implementation - would be replaced with actual DTF API calls
        # This would populate self.metrics_data with the latest metrics
        
        # For now, we'll simulate some metrics for testing
        # In a real implementation, this would be removed and replaced with
        # actual data collection from the DTF
        
        # Sample metrics data structure
        from datetime import datetime
        import random
        
        timestamp = datetime.now().timestamp()
        
        # Simulate worker metrics
        worker_count = random.randint(5, 20)
        self.metrics_data["worker_count"] = worker_count
        
        worker_throughput = {}
        worker_resources = {}
        
        for i in range(worker_count):
            worker_id = f"worker-{i}"
            worker_type = random.choice(["cpu", "gpu", "webgpu", "webnn"])
            
            # Simulate throughput
            throughput = random.uniform(10, 100)
            worker_throughput[(worker_id, worker_type)] = throughput
            
            # Simulate resource usage
            for resource in ["cpu", "memory", "disk", "network"]:
                usage = random.uniform(10, 95)
                worker_resources[(worker_id, resource)] = usage
        
        self.metrics_data["worker_throughput"] = worker_throughput
        self.metrics_data["worker_resources"] = worker_resources
        
        # Simulate task metrics
        task_execution = {}
        task_queue = {}
        task_success = {}
        
        for task_type in ["test", "benchmark", "validation", "analysis"]:
            for worker_type in ["cpu", "gpu", "webgpu", "webnn"]:
                # Execution time
                task_execution[(task_type, worker_type)] = random.uniform(5, 500)
                
                # Success rate
                task_success[(task_type, worker_type)] = random.uniform(70, 100)
            
            # Queue length by priority
            for priority in ["high", "medium", "low"]:
                task_queue[(task_type, priority)] = random.randint(0, 50)
        
        self.metrics_data["task_execution"] = task_execution
        self.metrics_data["task_queue"] = task_queue
        self.metrics_data["task_success"] = task_success
        
        # Record timestamp of collection
        self.metrics_data["timestamp"] = timestamp
    
    def _update_prometheus_metrics(self):
        """Update Prometheus metrics based on collected data."""
        # Update worker metrics
        self.metrics["worker_count"].set(self.metrics_data.get("worker_count", 0))
        
        for (worker_id, worker_type), throughput in self.metrics_data.get("worker_throughput", {}).items():
            self.metrics["worker_task_throughput"].labels(
                worker_id=worker_id, 
                worker_type=worker_type
            ).set(throughput)
        
        for (worker_id, resource_type), usage in self.metrics_data.get("worker_resources", {}).items():
            self.metrics["worker_resource_usage"].labels(
                worker_id=worker_id, 
                resource_type=resource_type
            ).set(usage)
        
        # Update task metrics
        for (task_type, worker_type), execution_time in self.metrics_data.get("task_execution", {}).items():
            self.metrics["task_execution_time"].labels(
                task_type=task_type, 
                worker_type=worker_type
            ).observe(execution_time)
            
        for (task_type, priority), queue_length in self.metrics_data.get("task_queue", {}).items():
            self.metrics["task_queue_length"].labels(
                task_type=task_type, 
                priority=priority
            ).set(queue_length)
            
        for (task_type, worker_type), success_rate in self.metrics_data.get("task_success", {}).items():
            self.metrics["task_success_rate"].labels(
                task_type=task_type, 
                worker_type=worker_type
            ).set(success_rate)
        
        # Update anomaly metrics
        for (metric_name, algorithm), anomaly_info in self.anomalies.items():
            self.metrics["anomaly_count"].labels(
                metric_name=metric_name, 
                algorithm=algorithm
            ).set(anomaly_info.get("count", 0))
            
            self.metrics["anomaly_severity"].labels(
                metric_name=metric_name, 
                algorithm=algorithm
            ).set(anomaly_info.get("severity", 0))
    
    def _anomaly_detection_loop(self):
        """Background loop for running anomaly detection."""
        while self.running:
            try:
                self._run_anomaly_detection()
            except Exception as e:
                logger.error(f"Error in anomaly detection: {str(e)}")
            
            # Sleep until next detection cycle
            time.sleep(self.anomaly_detection_interval)
    
    def _run_anomaly_detection(self):
        """
        Run anomaly detection on collected metrics.
        This uses the MLAnomalyDetection module to analyze metrics and
        identify anomalies.
        """
        logger.info("Running anomaly detection cycle")
        
        # Convert metrics to time series format for ML detection
        time_series_data = self._prepare_time_series_data()
        
        # Run detection for each metric time series
        for metric_name, time_series in time_series_data.items():
            # Skip metrics with insufficient data
            if len(time_series) < 10:
                continue
                
            try:
                # Run anomaly detection
                results = self.ml_detector.detect_anomalies(
                    time_series, 
                    metric_name=metric_name
                )
                
                # Store results
                for algorithm, result in results.items():
                    anomaly_count = len(result.get("anomalies", []))
                    severity = result.get("severity", 0)
                    
                    self.anomalies[(metric_name, algorithm)] = {
                        "count": anomaly_count,
                        "severity": severity,
                        "anomalies": result.get("anomalies", []),
                        "timestamp": time.time()
                    }
                    
                    # Log significant anomalies
                    if severity > 70:
                        logger.warning(
                            f"High severity anomaly detected in {metric_name} "
                            f"using {algorithm}: severity={severity}"
                        )
                
                # Run forecasting if anomalies detected
                if any(result.get("severity", 0) > 50 for result in results.values()):
                    forecast_results = self.ml_detector.forecast_trend(
                        time_series,
                        metric_name=metric_name,
                        forecast_periods=24
                    )
                    
                    # Store forecast results
                    for algorithm, forecast in forecast_results.items():
                        if "accuracy" in forecast:
                            self.metrics["forecast_accuracy"].labels(
                                metric_name=metric_name,
                                algorithm=algorithm
                            ).set(forecast["accuracy"])
                    
                    # Generate visualization if significant anomalies
                    if any(result.get("severity", 0) > 70 for result in results.values()):
                        self.ml_detector.generate_visualization(
                            time_series,
                            results,
                            forecast_results,
                            title=f"Anomaly Detection: {metric_name}",
                            output_file=f"anomaly_{metric_name.replace(' ', '_')}.png"
                        )
            
            except Exception as e:
                logger.error(f"Error analyzing metric {metric_name}: {str(e)}")
    
    def _prepare_time_series_data(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Convert collected metrics into time series format for anomaly detection.
        Returns a dictionary mapping metric names to lists of (timestamp, value) tuples.
        """
        # In a real implementation, this would retrieve historical metrics
        # from storage or from Prometheus directly
        
        # For testing purposes, generate some synthetic time series
        # In a real implementation, this would be replaced with actual
        # historical data retrieval
        
        import numpy as np
        from datetime import datetime, timedelta
        
        time_series_data = {}
        
        # Generate 100 data points for each metric with some simulated patterns
        now = datetime.now()
        timestamps = [(now - timedelta(minutes=i)).timestamp() 
                     for i in range(100, 0, -1)]
        
        # Worker count with linear trend and seasonal pattern
        base = np.linspace(10, 15, 100)  # Linear trend
        seasonal = 2 * np.sin(np.linspace(0, 6*np.pi, 100))  # Seasonal pattern
        noise = np.random.normal(0, 0.5, 100)  # Random noise
        
        # Add an anomaly
        anomaly_idx = np.random.randint(70, 90)
        anomaly = np.zeros(100)
        anomaly[anomaly_idx] = 5  # Spike anomaly
        
        values = base + seasonal + noise + anomaly
        time_series_data["worker_count"] = list(zip(timestamps, values))
        
        # Task execution time with trend
        base = np.linspace(50, 70, 100)  # Upward trend
        noise = np.random.normal(0, 5, 100)
        
        # Add collective anomaly (sustained shift)
        anomaly = np.zeros(100)
        anomaly_start = np.random.randint(60, 75)
        anomaly[anomaly_start:anomaly_start+10] = 20
        
        values = base + noise + anomaly
        time_series_data["task_execution_time"] = list(zip(timestamps, values))
        
        # Resource usage with cyclic pattern
        base = 50 * np.ones(100)
        cyclic = 20 * np.sin(np.linspace(0, 4*np.pi, 100))
        noise = np.random.normal(0, 3, 100)
        
        values = base + cyclic + noise
        time_series_data["resource_usage"] = list(zip(timestamps, values))
        
        return time_series_data
    
    def _dashboard_update_loop(self):
        """Background loop for updating Grafana dashboards."""
        # Wait a bit before first update to ensure data is collected
        time.sleep(60)
        
        while self.running and self.grafana_url and self.grafana_api_key:
            try:
                self._update_dashboards()
            except Exception as e:
                logger.error(f"Error updating dashboards: {str(e)}")
            
            # Sleep until next update
            time.sleep(self.dashboard_update_interval)
    
    def _update_dashboards(self):
        """Update Grafana dashboards with latest metrics and anomaly information."""
        logger.info("Updating Grafana dashboards")
        
        if not self.grafana_url or not self.grafana_api_key:
            logger.warning("Grafana not configured, skipping dashboard update")
            return
        
        # Create or update main dashboard
        main_dashboard = self.ml_detector.create_grafana_dashboard(
            title="Distributed Testing Framework Overview",
            datasource="Prometheus",
            metrics=[
                "dtf_worker_count",
                "dtf_task_execution_time",
                "dtf_task_success_rate",
                "dtf_worker_resource_usage",
            ],
            refresh="30s",
            time_range="3h"
        )
        
        # Create or update anomaly dashboard
        anomaly_dashboard = self.ml_detector.create_grafana_dashboard(
            title="DTF Anomaly Detection",
            datasource="Prometheus",
            metrics=[
                "dtf_anomaly_count",
                "dtf_anomaly_severity",
                "dtf_forecast_accuracy"
            ],
            refresh="1m",
            time_range="6h",
            include_anomaly_panels=True
        )
        
        # Upload dashboards to Grafana
        for title, dashboard in [
            ("DTF Overview", main_dashboard),
            ("DTF Anomalies", anomaly_dashboard)
        ]:
            self._upload_dashboard_to_grafana(title, dashboard)
    
    def _upload_dashboard_to_grafana(self, title: str, dashboard: Dict[str, Any]):
        """Upload a dashboard to Grafana."""
        if not self.grafana_url or not self.grafana_api_key:
            return
            
        try:
            headers = {
                "Authorization": f"Bearer {self.grafana_api_key}",
                "Content-Type": "application/json",
            }
            
            # Prepare dashboard payload
            payload = {
                "dashboard": dashboard,
                "overwrite": True,
                "message": f"Updated by DTF at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            }
            
            response = requests.post(
                f"{self.grafana_url.rstrip('/')}/api/dashboards/db",
                headers=headers,
                json=payload
            )
            
            if response.status_code in (200, 201):
                logger.info(f"Successfully updated dashboard: {title}")
                result = response.json()
                dashboard_url = result.get("url", "")
                logger.info(f"Dashboard URL: {dashboard_url}")
            else:
                logger.error(f"Failed to update dashboard: {response.status_code} - {response.text}")
        
        except Exception as e:
            logger.error(f"Error uploading dashboard to Grafana: {str(e)}")
    
    def update_metrics_from_data(self, metrics_data: Dict[str, Any]):
        """
        Update metrics from external data source.
        This method can be called by external components to update metrics.
        
        Args:
            metrics_data: Dictionary of metrics data to update
        """
        # Update metrics data
        self.metrics_data.update(metrics_data)
        
        # Update Prometheus metrics
        self._update_prometheus_metrics()
    
    def get_detected_anomalies(self) -> Dict[str, Any]:
        """
        Get all detected anomalies.
        
        Returns:
            Dictionary of detected anomalies by metric and algorithm
        """
        return self.anomalies
    
    def get_forecasts(self) -> Dict[str, Any]:
        """
        Get forecasts for all metrics that have been analyzed.
        
        Returns:
            Dictionary of forecasts by metric
        """
        if not hasattr(self.ml_detector, "forecasts"):
            return {}
            
        return self.ml_detector.forecasts


# Standalone function to create and start the integration
def start_prometheus_grafana_integration(
    config_file: Optional[str] = None,
    prometheus_port: int = 8000,
    grafana_url: Optional[str] = None,
    grafana_api_key: Optional[str] = None,
    prometheus_url: Optional[str] = None,
    ml_config: Optional[Dict[str, Any]] = None,
) -> PrometheusGrafanaIntegration:
    """
    Create and start the Prometheus/Grafana integration.
    
    Args:
        config_file: Optional path to configuration file
        prometheus_port: Port to expose Prometheus metrics on
        grafana_url: Base URL for Grafana API
        grafana_api_key: API key for Grafana
        prometheus_url: URL for Prometheus API
        ml_config: Configuration for ML anomaly detection
        
    Returns:
        Running PrometheusGrafanaIntegration instance
    """
    # Load config from file if provided
    config = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
    
    # Override config with provided parameters
    if prometheus_port:
        config["prometheus_port"] = prometheus_port
    if grafana_url:
        config["grafana_url"] = grafana_url
    if grafana_api_key:
        config["grafana_api_key"] = grafana_api_key
    if prometheus_url:
        config["prometheus_url"] = prometheus_url
    if ml_config:
        config["ml_config"] = ml_config
    
    # Create integration instance
    integration = PrometheusGrafanaIntegration(**config)
    
    # Start integration
    integration.start()
    
    return integration


if __name__ == "__main__":
    # Example of running the integration as a standalone service
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Prometheus/Grafana integration")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--port", type=int, default=8000, help="Prometheus port")
    parser.add_argument("--grafana-url", help="Grafana URL")
    parser.add_argument("--grafana-key", help="Grafana API key")
    parser.add_argument("--prometheus-url", help="Prometheus URL")
    
    args = parser.parse_args()
    
    # Start integration
    integration = start_prometheus_grafana_integration(
        config_file=args.config,
        prometheus_port=args.port,
        grafana_url=args.grafana_url,
        grafana_api_key=args.grafana_key,
        prometheus_url=args.prometheus_url
    )
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping integration service")
        integration.stop()