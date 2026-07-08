#!/usr/bin/env python3
"""
ML-based Anomaly Detection for Distributed Testing Framework

This module provides machine learning-based anomaly detection for the Distributed
Testing Framework's performance metrics. It integrates with the external monitoring
systems (Prometheus/Grafana) and implements advanced anomaly detection algorithms.

Key features:
1. Automatic anomaly detection using multiple ML algorithms
2. Trend analysis with prediction capabilities
3. Integration with Prometheus metrics for real-time monitoring
4. Historical data analysis for performance regression detection
5. Visualization tools for anomaly explanation
"""

import os
import time
import logging
import threading
import numpy as np
import pandas as pd
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
import json
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("distributed_testing.ml_anomaly_detection")

# Try to import machine learning packages
try:
    import sklearn
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. Limited machine learning capabilities.")
    SKLEARN_AVAILABLE = False

# Try to import statsmodels for time series analysis
try:
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    logger.warning("statsmodels not available. Limited time series capabilities.")
    STATSMODELS_AVAILABLE = False

# Try to import Prophet for time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    logger.warning("Prophet not available. Advanced forecasting will be limited.")
    PROPHET_AVAILABLE = False

# Try to import visualization packages
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Visualization packages not available. Visualizations will be limited.")
    VISUALIZATION_AVAILABLE = False


class MLAnomalyDetection:
    """
    Machine Learning-based Anomaly Detection for the Distributed Testing Framework.
    
    This class provides advanced anomaly detection using multiple machine learning
    algorithms, with support for time series analysis, forecasting, and trend analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML anomaly detection module.
        
        Args:
            config: Configuration options for the module
        """
        self.config = config or {}
        
        # Default configuration
        default_config = {
            # Metrics to analyze
            "metrics": [
                "worker_count", "active_workers", "active_tasks", "pending_tasks",
                "cpu_utilization", "memory_utilization", "gpu_utilization",
                "task_throughput", "allocation_time", "resource_efficiency"
            ],
            
            # Anomaly detection configuration
            "anomaly_detection": {
                "enabled": True,
                "methods": ["isolation_forest", "dbscan", "threshold", "mad"],
                "isolation_forest": {
                    "contamination": 0.05,
                    "n_estimators": 100,
                    "random_state": 42
                },
                "dbscan": {
                    "eps": 0.5,
                    "min_samples": 5
                },
                "threshold": {
                    "std_multiplier": 3.0
                },
                "mad": {  # Median Absolute Deviation
                    "threshold": 3.0
                },
                "min_data_points": 10
            },
            
            # Time series analysis configuration
            "time_series_analysis": {
                "enabled": True,
                "methods": ["arima", "prophet"],
                "arima": {
                    "order": (1, 1, 1),
                    "seasonal_order": (1, 1, 1, 12) if STATSMODELS_AVAILABLE else None
                },
                "prophet": {
                    "changepoint_prior_scale": 0.05,
                    "seasonality_mode": "multiplicative"
                },
                "window_size": 24,  # Window size for rolling statistics
                "forecast_horizon": 12,  # Number of periods to forecast
                "confidence_interval": 0.9  # Confidence interval for forecasts
            },
            
            # Trend analysis configuration
            "trend_analysis": {
                "enabled": True,
                "methods": ["regression", "moving_average"],
                "regression": {
                    "poly_degree": 2
                },
                "moving_average": {
                    "window_size": 5
                },
                "min_points_for_trend": 8
            },
            
            # Model persistence
            "model_persistence": {
                "enabled": True,
                "directory": "models",
                "max_models": 5,
                "retraining_interval_hours": 24
            },
            
            # Visualization
            "visualization": {
                "enabled": True,
                "directory": "visualizations",
                "formats": ["png", "svg"],
                "dpi": 100,
                "max_visualizations": 50
            }
        }
        
        # Apply default config values if not specified
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict) and isinstance(self.config[key], dict):
                # Merge nested dictionaries
                for nested_key, nested_value in value.items():
                    if nested_key not in self.config[key]:
                        self.config[key][nested_key] = nested_value
        
        # Initialize models dictionary
        self.models = {}
        
        # Initialize data storage
        self.data = defaultdict(list)
        self.timestamps = []
        
        # Initialize detection results
        self.anomalies = defaultdict(list)
        self.forecasts = defaultdict(dict)
        self.trends = defaultdict(dict)
        
        # Create directories if needed
        if self.config["model_persistence"]["enabled"]:
            os.makedirs(self.config["model_persistence"]["directory"], exist_ok=True)
        
        if self.config["visualization"]["enabled"]:
            os.makedirs(self.config["visualization"]["directory"], exist_ok=True)
        
        # Initialize background tasks
        self.running = False
        self.analysis_thread = None
        self.stop_event = threading.Event()
        
        # Initialize last training timestamp
        self.last_training_time = None
        
        logger.info("ML Anomaly Detection initialized with configuration:")
        logger.info(f"  - Metrics: {len(self.config['metrics'])} metrics configured")
        logger.info(f"  - Anomaly Detection: {self.config['anomaly_detection']['enabled']}")
        logger.info(f"  - Time Series Analysis: {self.config['time_series_analysis']['enabled']}")
        logger.info(f"  - Trend Analysis: {self.config['trend_analysis']['enabled']}")
    
    def start(self, interval: int = 60):
        """
        Start the ML anomaly detection system.
        
        Args:
            interval: Analysis interval in seconds
        
        Returns:
            bool: Whether the system was started successfully
        """
        if self.running:
            logger.warning("ML Anomaly Detection is already running")
            return True
        
        self.running = True
        self.stop_event.clear()
        
        # Load models if available
        self._load_models()
        
        # Start background thread for periodic analysis
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop,
            args=(interval,),
            daemon=True
        )
        self.analysis_thread.start()
        
        logger.info(f"ML Anomaly Detection started with {interval}s interval")
        return True
    
    def stop(self):
        """
        Stop the ML anomaly detection system.
        
        Returns:
            bool: Whether the system was stopped successfully
        """
        if not self.running:
            logger.warning("ML Anomaly Detection is not running")
            return True
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=10.0)
        
        self.running = False
        
        # Save models if needed
        self._save_models()
        
        logger.info("ML Anomaly Detection stopped")
        return True
    
    def add_data_point(self, metrics: Dict[str, float], timestamp: Optional[datetime.datetime] = None):
        """
        Add a new data point for analysis.
        
        Args:
            metrics: Dictionary of metrics to add
            timestamp: Timestamp for the data point (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        # Add timestamp
        self.timestamps.append(timestamp)
        
        # Add metrics
        for metric_name in self.config["metrics"]:
            if metric_name in metrics:
                self.data[metric_name].append(metrics[metric_name])
            else:
                # Use NaN for missing metrics
                self.data[metric_name].append(float('nan'))
        
        # Limit data size if needed (keep last 1000 points)
        max_data_points = 1000
        if len(self.timestamps) > max_data_points:
            self.timestamps = self.timestamps[-max_data_points:]
            for metric_name in self.data:
                self.data[metric_name] = self.data[metric_name][-max_data_points:]
    
    def get_anomalies(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detected anomalies.
        
        Args:
            metric_name: Optional metric name to filter results
        
        Returns:
            Dictionary with detected anomalies
        """
        if metric_name:
            # Return anomalies for specific metric
            if metric_name in self.anomalies:
                return {metric_name: self.anomalies[metric_name]}
            else:
                return {}
        else:
            # Return all anomalies
            return dict(self.anomalies)
    
    def get_forecasts(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get forecasts for metrics.
        
        Args:
            metric_name: Optional metric name to filter results
        
        Returns:
            Dictionary with forecasts
        """
        if metric_name:
            # Return forecasts for specific metric
            if metric_name in self.forecasts:
                return {metric_name: self.forecasts[metric_name]}
            else:
                return {}
        else:
            # Return all forecasts
            return dict(self.forecasts)
    
    def get_trends(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get trend analysis for metrics.
        
        Args:
            metric_name: Optional metric name to filter results
        
        Returns:
            Dictionary with trend analysis
        """
        if metric_name:
            # Return trends for specific metric
            if metric_name in self.trends:
                return {metric_name: self.trends[metric_name]}
            else:
                return {}
        else:
            # Return all trends
            return dict(self.trends)
    
    def analyze_metrics(self, force: bool = False) -> Dict[str, Any]:
        """
        Analyze metrics for anomalies, forecasts, and trends.
        
        Args:
            force: Whether to force analysis even with insufficient data
        
        Returns:
            Dictionary with analysis results
        """
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics_analyzed": len(self.config["metrics"]),
            "anomaly_detection": {"enabled": self.config["anomaly_detection"]["enabled"]},
            "time_series_analysis": {"enabled": self.config["time_series_analysis"]["enabled"]},
            "trend_analysis": {"enabled": self.config["trend_analysis"]["enabled"]},
            "results": {
                "anomalies": {},
                "forecasts": {},
                "trends": {}
            }
        }
        
        # Check if we have enough data
        min_data_points = self.config["anomaly_detection"]["min_data_points"]
        if len(self.timestamps) < min_data_points and not force:
            logger.warning(f"Not enough data points for analysis: {len(self.timestamps)} < {min_data_points}")
            results["status"] = "insufficient_data"
            results["message"] = f"Insufficient data for analysis. Need at least {min_data_points} data points."
            return results
        
        results["data_points"] = len(self.timestamps)
        
        # Process each metric
        for metric_name in self.config["metrics"]:
            # Skip if not enough data for this metric
            if metric_name not in self.data or len(self.data[metric_name]) < min_data_points:
                continue
            
            metric_results = {}
            
            # Get data for this metric
            values = np.array(self.data[metric_name])
            timestamps = np.array(self.timestamps)
            
            # Skip metrics with all NaN values
            if np.all(np.isnan(values)):
                continue
            
            # Replace NaN values with interpolation or previous values
            values = self._handle_nan_values(values)
            
            # Detect anomalies
            if self.config["anomaly_detection"]["enabled"]:
                anomalies = self._detect_anomalies(metric_name, values, timestamps)
                if anomalies:
                    self.anomalies[metric_name] = anomalies
                    metric_results["anomalies"] = anomalies
                    results["results"]["anomalies"][metric_name] = anomalies
            
            # Perform time series analysis
            if self.config["time_series_analysis"]["enabled"]:
                forecast = self._analyze_time_series(metric_name, values, timestamps)
                if forecast:
                    self.forecasts[metric_name] = forecast
                    metric_results["forecast"] = forecast
                    results["results"]["forecasts"][metric_name] = forecast
            
            # Perform trend analysis
            if self.config["trend_analysis"]["enabled"]:
                trend = self._analyze_trend(metric_name, values, timestamps)
                if trend:
                    self.trends[metric_name] = trend
                    metric_results["trend"] = trend
                    results["results"]["trends"][metric_name] = trend
            
            # Generate visualizations for this metric
            if self.config["visualization"]["enabled"]:
                try:
                    self._generate_visualizations(metric_name, values, timestamps, metric_results)
                except Exception as e:
                    logger.warning(f"Error generating visualizations for {metric_name}: {e}")
        
        # Check if models should be retrained
        self._check_retraining()
        
        # Add summary statistics
        results["summary"] = {
            "total_anomalies": sum(len(anomalies) for anomalies in self.anomalies.values()),
            "metrics_with_anomalies": len(self.anomalies),
            "metrics_with_forecasts": len(self.forecasts),
            "metrics_with_trends": len(self.trends)
        }
        
        results["status"] = "success"
        return results
    
    def get_prometheus_metrics(self) -> Dict[str, Any]:
        """
        Get metrics formatted for Prometheus.
        
        Returns:
            Dictionary with metrics for Prometheus
        """
        prometheus_metrics = {
            "ml_anomaly_detection_anomalies_total": {
                "type": "gauge",
                "help": "Total number of anomalies detected",
                "value": sum(len(anomalies) for anomalies in self.anomalies.values())
            },
            "ml_anomaly_detection_metrics_with_anomalies": {
                "type": "gauge",
                "help": "Number of metrics with anomalies",
                "value": len(self.anomalies)
            },
            "ml_anomaly_detection_data_points": {
                "type": "gauge",
                "help": "Number of data points in analysis",
                "value": len(self.timestamps)
            }
        }
        
        # Add metric-specific anomaly counts
        for metric_name, anomalies in self.anomalies.items():
            prometheus_metrics[f"ml_anomaly_detection_{metric_name}_anomalies"] = {
                "type": "gauge",
                "help": f"Number of anomalies in {metric_name}",
                "value": len(anomalies)
            }
        
        # Add trend directions
        for metric_name, trend in self.trends.items():
            if "direction" in trend:
                # Convert trend direction to numeric value (1 = up, -1 = down, 0 = stable)
                direction_value = 0
                if trend["direction"] == "increasing":
                    direction_value = 1
                elif trend["direction"] == "decreasing":
                    direction_value = -1
                
                prometheus_metrics[f"ml_anomaly_detection_{metric_name}_trend"] = {
                    "type": "gauge",
                    "help": f"Trend direction for {metric_name} (1 = up, -1 = down, 0 = stable)",
                    "value": direction_value
                }
        
        # Add forecast values
        for metric_name, forecast in self.forecasts.items():
            if "forecast_values" in forecast and forecast["forecast_values"]:
                # Use last forecast value
                last_forecast = forecast["forecast_values"][-1]
                
                prometheus_metrics[f"ml_anomaly_detection_{metric_name}_forecast"] = {
                    "type": "gauge",
                    "help": f"Forecast value for {metric_name}",
                    "value": last_forecast
                }
        
        return prometheus_metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data_points": len(self.timestamps),
            "metrics_tracked": len(self.data),
            "anomalies": {
                "total": sum(len(anomalies) for anomalies in self.anomalies.values()),
                "metrics_with_anomalies": len(self.anomalies),
                "recent_anomalies": {}
            },
            "forecasts": {
                "metrics_with_forecasts": len(self.forecasts),
                "forecast_horizon": self.config["time_series_analysis"]["forecast_horizon"]
            },
            "trends": {
                "metrics_with_trends": len(self.trends),
                "summary": {}
            },
            "models": {
                "total": len(self.models),
                "last_training": self.last_training_time.isoformat() if self.last_training_time else None
            }
        }
        
        # Add recent anomalies
        for metric_name, anomalies in self.anomalies.items():
            if anomalies:
                # Get most recent anomaly
                recent_anomaly = max(anomalies, key=lambda x: x["timestamp"] if "timestamp" in x else "")
                summary["anomalies"]["recent_anomalies"][metric_name] = recent_anomaly
        
        # Add trend summary
        for metric_name, trend in self.trends.items():
            if "direction" in trend:
                summary["trends"]["summary"][metric_name] = {
                    "direction": trend["direction"],
                    "confidence": trend.get("confidence", None),
                    "slope": trend.get("slope", None)
                }
        
        return summary
    
    def _detect_anomalies(
        self, 
        metric_name: str, 
        values: np.ndarray, 
        timestamps: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metric values.
        
        Args:
            metric_name: Name of the metric
            values: Array of metric values
            timestamps: Array of timestamps
        
        Returns:
            List of detected anomalies
        """
        if not self.config["anomaly_detection"]["enabled"]:
            return []
        
        methods = self.config["anomaly_detection"]["methods"]
        anomalies = []
        
        # Skip if not enough data
        min_data_points = self.config["anomaly_detection"]["min_data_points"]
        if len(values) < min_data_points:
            logger.debug(f"Not enough data points for anomaly detection on {metric_name}")
            return []
        
        # Convert timestamps to datetime objects if needed
        ts_datetime = []
        for ts in timestamps:
            if isinstance(ts, str):
                ts_datetime.append(datetime.datetime.fromisoformat(ts))
            else:
                ts_datetime.append(ts)
        
        # Apply Isolation Forest if enabled and available
        if "isolation_forest" in methods and SKLEARN_AVAILABLE:
            try:
                # Get configuration
                config = self.config["anomaly_detection"]["isolation_forest"]
                
                # Initialize or get model
                model_key = f"{metric_name}_isolation_forest"
                if model_key not in self.models:
                    model = IsolationForest(
                        contamination=config["contamination"],
                        n_estimators=config["n_estimators"],
                        random_state=config["random_state"]
                    )
                    self.models[model_key] = model
                else:
                    model = self.models[model_key]
                
                # Reshape values for scikit-learn
                X = values.reshape(-1, 1)
                
                # Fit model if not fitted
                if not hasattr(model, "offset_"):
                    model.fit(X)
                
                # Predict anomalies
                y_pred = model.predict(X)
                
                # Find anomaly indices (where prediction is -1)
                anomaly_indices = np.where(y_pred == -1)[0]
                
                # Calculate anomaly scores
                scores = model.decision_function(X)
                
                # Create anomaly records
                for idx in anomaly_indices:
                    # Only include if within bounds
                    if 0 <= idx < len(values) and 0 <= idx < len(ts_datetime):
                        score = scores[idx]
                        confidence = min(1.0, max(0.0, -score * 2))  # Convert to [0, 1]
                        
                        anomaly = {
                            "method": "isolation_forest",
                            "index": int(idx),
                            "value": float(values[idx]),
                            "timestamp": ts_datetime[idx].isoformat(),
                            "confidence": float(confidence),
                            "description": f"Isolation Forest detected anomaly in {metric_name}"
                        }
                        
                        anomalies.append(anomaly)
                
                logger.debug(f"Isolation Forest found {len(anomaly_indices)} anomalies in {metric_name}")
                
            except Exception as e:
                logger.warning(f"Error in Isolation Forest anomaly detection for {metric_name}: {e}")
        
        # Apply DBSCAN if enabled and available
        if "dbscan" in methods and SKLEARN_AVAILABLE:
            try:
                # Get configuration
                config = self.config["anomaly_detection"]["dbscan"]
                
                # Prepare data
                X = values.reshape(-1, 1)
                
                # Normalize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply DBSCAN
                dbscan = DBSCAN(
                    eps=config["eps"],
                    min_samples=config["min_samples"]
                )
                labels = dbscan.fit_predict(X_scaled)
                
                # Find anomaly indices (where label is -1)
                anomaly_indices = np.where(labels == -1)[0]
                
                # Create anomaly records
                for idx in anomaly_indices:
                    # Only include if within bounds
                    if 0 <= idx < len(values) and 0 <= idx < len(ts_datetime):
                        # Calculate distance to nearest core point for confidence
                        core_samples_mask = np.zeros_like(labels, dtype=bool)
                        core_samples_mask[dbscan.core_sample_indices_] = True
                        
                        if np.any(core_samples_mask):
                            # Calculate distances to core samples
                            core_samples = X_scaled[core_samples_mask]
                            distances = np.linalg.norm(X_scaled[idx].reshape(1, -1) - core_samples, axis=1)
                            min_distance = np.min(distances)
                            
                            # Convert distance to confidence
                            confidence = min(1.0, max(0.0, min_distance / 2.0))
                        else:
                            confidence = 0.8  # Default if no core samples
                        
                        anomaly = {
                            "method": "dbscan",
                            "index": int(idx),
                            "value": float(values[idx]),
                            "timestamp": ts_datetime[idx].isoformat(),
                            "confidence": float(confidence),
                            "description": f"DBSCAN detected anomaly in {metric_name}"
                        }
                        
                        anomalies.append(anomaly)
                
                logger.debug(f"DBSCAN found {len(anomaly_indices)} anomalies in {metric_name}")
                
            except Exception as e:
                logger.warning(f"Error in DBSCAN anomaly detection for {metric_name}: {e}")
        
        # Apply threshold-based detection
        if "threshold" in methods:
            try:
                # Get configuration
                config = self.config["anomaly_detection"]["threshold"]
                
                # Calculate mean and standard deviation
                mean = np.mean(values)
                std = np.std(values)
                
                # Set thresholds
                upper_threshold = mean + config["std_multiplier"] * std
                lower_threshold = mean - config["std_multiplier"] * std
                
                # Find anomaly indices
                anomaly_indices = np.where(
                    (values > upper_threshold) | (values < lower_threshold)
                )[0]
                
                # Create anomaly records
                for idx in anomaly_indices:
                    # Only include if within bounds
                    if 0 <= idx < len(values) and 0 <= idx < len(ts_datetime):
                        # Calculate how many standard deviations from mean
                        deviations = abs(values[idx] - mean) / std
                        
                        # Convert to confidence score
                        confidence = min(1.0, max(0.0, (deviations - config["std_multiplier"]) / 2.0))
                        
                        anomaly = {
                            "method": "threshold",
                            "index": int(idx),
                            "value": float(values[idx]),
                            "timestamp": ts_datetime[idx].isoformat(),
                            "confidence": float(confidence),
                            "threshold": float(upper_threshold if values[idx] > mean else lower_threshold),
                            "description": f"Threshold-based detection found anomaly in {metric_name} "
                                         f"({deviations:.2f} std devs from mean)"
                        }
                        
                        anomalies.append(anomaly)
                
                logger.debug(f"Threshold-based detection found {len(anomaly_indices)} anomalies in {metric_name}")
                
            except Exception as e:
                logger.warning(f"Error in threshold-based anomaly detection for {metric_name}: {e}")
        
        # Apply Median Absolute Deviation (MAD) detection
        if "mad" in methods:
            try:
                # Get configuration
                config = self.config["anomaly_detection"]["mad"]
                
                # Calculate median and MAD
                median = np.median(values)
                mad = np.median(np.abs(values - median))
                
                # Handle case where MAD is 0
                if mad == 0:
                    mad = np.mean(np.abs(values - median)) or 1e-8
                
                # Set thresholds
                threshold = config["threshold"]
                upper_threshold = median + threshold * mad
                lower_threshold = median - threshold * mad
                
                # Find anomaly indices
                anomaly_indices = np.where(
                    (values > upper_threshold) | (values < lower_threshold)
                )[0]
                
                # Create anomaly records
                for idx in anomaly_indices:
                    # Only include if within bounds
                    if 0 <= idx < len(values) and 0 <= idx < len(ts_datetime):
                        # Calculate how many MADs from median
                        deviation = abs(values[idx] - median) / mad
                        
                        # Convert to confidence score
                        confidence = min(1.0, max(0.0, (deviation - threshold) / threshold))
                        
                        anomaly = {
                            "method": "mad",
                            "index": int(idx),
                            "value": float(values[idx]),
                            "timestamp": ts_datetime[idx].isoformat(),
                            "confidence": float(confidence),
                            "threshold": float(upper_threshold if values[idx] > median else lower_threshold),
                            "description": f"MAD-based detection found anomaly in {metric_name} "
                                         f"({deviation:.2f} MADs from median)"
                        }
                        
                        anomalies.append(anomaly)
                
                logger.debug(f"MAD-based detection found {len(anomaly_indices)} anomalies in {metric_name}")
                
            except Exception as e:
                logger.warning(f"Error in MAD-based anomaly detection for {metric_name}: {e}")
        
        # Remove duplicates (same index detected by multiple methods)
        unique_anomalies = {}
        for anomaly in anomalies:
            idx = anomaly["index"]
            if idx not in unique_anomalies or anomaly["confidence"] > unique_anomalies[idx]["confidence"]:
                unique_anomalies[idx] = anomaly
        
        return list(unique_anomalies.values())
    
    def _analyze_time_series(
        self, 
        metric_name: str, 
        values: np.ndarray, 
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform time series analysis for metrics.
        
        Args:
            metric_name: Name of the metric
            values: Array of metric values
            timestamps: Array of timestamps
        
        Returns:
            Dictionary with time series analysis results
        """
        if not self.config["time_series_analysis"]["enabled"]:
            return {}
        
        # Get configuration
        methods = self.config["time_series_analysis"]["methods"]
        forecast_horizon = self.config["time_series_analysis"]["forecast_horizon"]
        confidence_interval = self.config["time_series_analysis"]["confidence_interval"]
        
        # Check if we have enough data
        min_points_for_ts = max(forecast_horizon * 2, 8)
        if len(values) < min_points_for_ts:
            logger.debug(f"Not enough data points for time series analysis on {metric_name}")
            return {}
        
        results = {
            "methods_used": [],
            "forecast_values": [],
            "forecast_timestamps": [],
            "lower_bound": [],
            "upper_bound": []
        }
        
        # Convert timestamps to datetime if needed
        ts_datetime = []
        for ts in timestamps:
            if isinstance(ts, str):
                ts_datetime.append(datetime.datetime.fromisoformat(ts))
            else:
                ts_datetime.append(ts)
        
        # Create forecast timestamps
        last_timestamp = ts_datetime[-1]
        forecast_timestamps = []
        
        # Create regular intervals for forecast
        if len(ts_datetime) >= 2:
            # Calculate average interval between timestamps
            intervals = [(ts_datetime[i] - ts_datetime[i-1]).total_seconds() 
                         for i in range(1, len(ts_datetime))]
            avg_interval = sum(intervals) / len(intervals)
            
            # Create forecast timestamps
            for i in range(1, forecast_horizon + 1):
                forecast_timestamps.append(
                    last_timestamp + datetime.timedelta(seconds=i * avg_interval)
                )
        
        # Apply ARIMA if enabled and available
        if "arima" in methods and STATSMODELS_AVAILABLE:
            try:
                # Get configuration
                config = self.config["time_series_analysis"]["arima"]
                
                # Create and fit ARIMA model
                model = ARIMA(
                    values,
                    order=config["order"]
                )
                model_fit = model.fit()
                
                # Generate forecast
                forecast = model_fit.forecast(steps=forecast_horizon)
                
                # Get confidence intervals
                ci_alpha = 1 - confidence_interval
                forecast_ci = model_fit.get_forecast(steps=forecast_horizon).conf_int(alpha=ci_alpha)
                
                # Extract lower and upper bounds
                lower_bound = forecast_ci[:, 0]
                upper_bound = forecast_ci[:, 1]
                
                # Add results
                results["methods_used"].append("arima")
                results["forecast_values"] = forecast.tolist()
                results["forecast_timestamps"] = [ts.isoformat() for ts in forecast_timestamps]
                results["lower_bound"] = lower_bound.tolist()
                results["upper_bound"] = upper_bound.tolist()
                results["model_info"] = {
                    "name": "ARIMA",
                    "order": config["order"],
                    "aic": model_fit.aic
                }
                
                logger.debug(f"ARIMA forecast generated for {metric_name}")
                
            except Exception as e:
                logger.warning(f"Error in ARIMA time series analysis for {metric_name}: {e}")
        
        # Apply Prophet if enabled and available
        if "prophet" in methods and PROPHET_AVAILABLE and results["forecast_values"] == []:
            try:
                # Get configuration
                config = self.config["time_series_analysis"]["prophet"]
                
                # Prepare data for Prophet
                df = pd.DataFrame({
                    'ds': ts_datetime,
                    'y': values
                })
                
                # Create and fit Prophet model
                model = Prophet(
                    changepoint_prior_scale=config["changepoint_prior_scale"],
                    seasonality_mode=config["seasonality_mode"]
                )
                model.fit(df)
                
                # Create future dataframe
                future = pd.DataFrame({'ds': forecast_timestamps})
                
                # Make forecast
                forecast = model.predict(future)
                
                # Extract forecast values and intervals
                forecast_values = forecast['yhat'].values
                lower_bound = forecast['yhat_lower'].values
                upper_bound = forecast['yhat_upper'].values
                
                # Add results
                results["methods_used"].append("prophet")
                results["forecast_values"] = forecast_values.tolist()
                results["forecast_timestamps"] = [ts.isoformat() for ts in forecast_timestamps]
                results["lower_bound"] = lower_bound.tolist()
                results["upper_bound"] = upper_bound.tolist()
                results["model_info"] = {
                    "name": "Prophet",
                    "changepoint_prior_scale": config["changepoint_prior_scale"],
                    "seasonality_mode": config["seasonality_mode"]
                }
                
                logger.debug(f"Prophet forecast generated for {metric_name}")
                
            except Exception as e:
                logger.warning(f"Error in Prophet time series analysis for {metric_name}: {e}")
        
        # Apply simple exponential smoothing if no other method worked
        if results["forecast_values"] == []:
            try:
                # Apply simple exponential smoothing
                alpha = 0.3  # Smoothing factor
                
                # Initialize forecast with last value
                forecast_values = np.zeros(forecast_horizon)
                forecast_values[0] = values[-1]
                
                # Generate forecast
                for i in range(1, forecast_horizon):
                    forecast_values[i] = forecast_values[i-1]
                
                # Calculate standard deviation for confidence intervals
                std = np.std(values)
                z_value = 1.96  # For 95% confidence interval
                margin = z_value * std
                
                lower_bound = forecast_values - margin
                upper_bound = forecast_values + margin
                
                # Add results
                results["methods_used"].append("exponential_smoothing")
                results["forecast_values"] = forecast_values.tolist()
                results["forecast_timestamps"] = [ts.isoformat() for ts in forecast_timestamps]
                results["lower_bound"] = lower_bound.tolist()
                results["upper_bound"] = upper_bound.tolist()
                results["model_info"] = {
                    "name": "Exponential Smoothing",
                    "alpha": alpha
                }
                
                logger.debug(f"Exponential smoothing forecast generated for {metric_name}")
                
            except Exception as e:
                logger.warning(f"Error in exponential smoothing for {metric_name}: {e}")
        
        # Add latest actual values for context
        if results["forecast_values"]:
            results["latest_values"] = values[-min(10, len(values)):].tolist()
            results["latest_timestamps"] = [ts.isoformat() for ts in ts_datetime[-min(10, len(ts_datetime)):]]
        
        return results
    
    def _analyze_trend(
        self, 
        metric_name: str, 
        values: np.ndarray, 
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze trends in metric values.
        
        Args:
            metric_name: Name of the metric
            values: Array of metric values
            timestamps: Array of timestamps
        
        Returns:
            Dictionary with trend analysis results
        """
        if not self.config["trend_analysis"]["enabled"]:
            return {}
        
        # Get configuration
        methods = self.config["trend_analysis"]["methods"]
        min_points = self.config["trend_analysis"]["min_points_for_trend"]
        
        # Check if we have enough data
        if len(values) < min_points:
            logger.debug(f"Not enough data points for trend analysis on {metric_name}")
            return {}
        
        results = {
            "methods_used": [],
            "direction": "stable",
            "confidence": 0.0
        }
        
        # Convert timestamps to seconds since epoch for numerical analysis
        if isinstance(timestamps[0], datetime.datetime) or isinstance(timestamps[0], str):
            time_seconds = []
            for ts in timestamps:
                if isinstance(ts, str):
                    dt = datetime.datetime.fromisoformat(ts)
                else:
                    dt = ts
                time_seconds.append(dt.timestamp())
            time_values = np.array(time_seconds)
        else:
            time_values = timestamps.astype(float)
        
        # Normalize time to [0, 1] range for better numerical stability
        time_min = np.min(time_values)
        time_max = np.max(time_values)
        if time_max > time_min:
            time_normalized = (time_values - time_min) / (time_max - time_min)
        else:
            time_normalized = np.zeros_like(time_values)
        
        # Apply regression if enabled
        if "regression" in methods:
            try:
                # Get configuration
                poly_degree = self.config["trend_analysis"]["regression"]["poly_degree"]
                
                # Fit polynomial regression
                coeffs = np.polyfit(time_normalized, values, poly_degree)
                p = np.poly1d(coeffs)
                
                # Evaluate polynomial at normalized time points
                y_pred = p(time_normalized)
                
                # Calculate metrics
                r2 = 1 - (np.sum((values - y_pred)**2) / np.sum((values - np.mean(values))**2))
                rmse = np.sqrt(np.mean((values - y_pred)**2))
                
                # Determine trend direction based on first derivative at latest point
                p_derivative = np.polyder(p)
                slope_at_end = p_derivative(1.0)  # Derivative at latest point (normalized time = 1.0)
                
                # Determine trend direction
                slope_threshold = 0.01 * np.std(values)  # Small threshold to filter out noise
                if slope_at_end > slope_threshold:
                    direction = "increasing"
                elif slope_at_end < -slope_threshold:
                    direction = "decreasing"
                else:
                    direction = "stable"
                
                # Calculate confidence based on R² and size of slope
                confidence = min(1.0, max(0.0, r2 * min(1.0, abs(slope_at_end) / (np.std(values) / 2))))
                
                # Add results
                results["methods_used"].append("regression")
                results["direction"] = direction
                results["confidence"] = float(confidence)
                results["slope"] = float(slope_at_end)
                results["r2"] = float(r2)
                results["rmse"] = float(rmse)
                results["predicted_values"] = y_pred.tolist()
                results["coefficients"] = coeffs.tolist()
                results["poly_degree"] = poly_degree
                
                logger.debug(f"Regression trend analysis completed for {metric_name}: {direction} (conf: {confidence:.2f})")
                
            except Exception as e:
                logger.warning(f"Error in regression trend analysis for {metric_name}: {e}")
        
        # Apply moving average if enabled
        if "moving_average" in methods and results["methods_used"] == []:
            try:
                # Get configuration
                window_size = self.config["trend_analysis"]["moving_average"]["window_size"]
                
                # Ensure window size is not too large
                window_size = min(window_size, len(values) // 2)
                
                # Calculate moving averages
                ma_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                
                # We need at least 2 points in moving average to determine trend
                if len(ma_values) >= 2:
                    # Calculate slope between first and last MA points
                    slope = (ma_values[-1] - ma_values[0]) / (len(ma_values) - 1)
                    
                    # Determine trend direction
                    slope_threshold = 0.01 * np.std(values)  # Small threshold
                    if slope > slope_threshold:
                        direction = "increasing"
                    elif slope < -slope_threshold:
                        direction = "decreasing"
                    else:
                        direction = "stable"
                    
                    # Calculate confidence (simple version - based on slope magnitude)
                    confidence = min(1.0, max(0.0, abs(slope) / (np.std(values) / 2)))
                    
                    # Add results
                    results["methods_used"].append("moving_average")
                    results["direction"] = direction
                    results["confidence"] = float(confidence)
                    results["slope"] = float(slope)
                    results["ma_values"] = ma_values.tolist()
                    results["window_size"] = window_size
                    
                    logger.debug(f"Moving average trend analysis completed for {metric_name}: {direction} (conf: {confidence:.2f})")
                
            except Exception as e:
                logger.warning(f"Error in moving average trend analysis for {metric_name}: {e}")
        
        return results
    
    def _handle_nan_values(self, values: np.ndarray) -> np.ndarray:
        """
        Handle NaN values in the data.
        
        Args:
            values: Array of values potentially containing NaNs
        
        Returns:
            Array with NaN values replaced
        """
        # Check if there are any NaN values
        if not np.any(np.isnan(values)):
            return values
        
        # Create a copy to avoid modifying the original
        values_clean = values.copy()
        
        # Find indices of NaN values
        nan_indices = np.where(np.isnan(values_clean))[0]
        
        # Replace NaN values
        for idx in nan_indices:
            # Try to use previous value
            if idx > 0 and not np.isnan(values_clean[idx-1]):
                values_clean[idx] = values_clean[idx-1]
            # Otherwise use next value if available
            elif idx < len(values_clean) - 1 and not np.isnan(values_clean[idx+1]):
                values_clean[idx] = values_clean[idx+1]
            # Last resort: use mean of non-NaN values
            else:
                values_clean[idx] = np.nanmean(values_clean)
        
        return values_clean
    
    def _generate_visualizations(
        self,
        metric_name: str,
        values: np.ndarray,
        timestamps: np.ndarray,
        results: Dict[str, Any]
    ):
        """
        Generate visualizations for metric analysis.
        
        Args:
            metric_name: Name of the metric
            values: Array of metric values
            timestamps: Array of timestamps
            results: Analysis results for this metric
        """
        if not self.config["visualization"]["enabled"] or not VISUALIZATION_AVAILABLE:
            return
        
        # Create directory for metric
        metric_dir = os.path.join(
            self.config["visualization"]["directory"],
            metric_name.replace(" ", "_")
        )
        os.makedirs(metric_dir, exist_ok=True)
        
        # Limit number of visualizations
        vis_files = [f for f in os.listdir(metric_dir) if f.endswith(".png") or f.endswith(".svg")]
        if len(vis_files) >= self.config["visualization"]["max_visualizations"]:
            # Remove oldest files
            vis_files.sort()
            for file in vis_files[:-(self.config["visualization"]["max_visualizations"]-1)]:
                try:
                    os.remove(os.path.join(metric_dir, file))
                except Exception:
                    pass
        
        # Convert timestamps to datetime
        ts_datetime = []
        for ts in timestamps:
            if isinstance(ts, str):
                ts_datetime.append(datetime.datetime.fromisoformat(ts))
            else:
                ts_datetime.append(ts)
        
        # Generate timestamp for files
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up figure and style
        plt.style.use('seaborn-darkgrid')
        
        # Create overview plot with anomalies and forecast
        try:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config["visualization"]["dpi"])
            
            # Plot actual values
            ax.plot(ts_datetime, values, 'b-', label="Actual", marker='o', markersize=3)
            
            # Plot anomalies if any
            if "anomalies" in results and results["anomalies"]:
                anomaly_indices = [a["index"] for a in results["anomalies"]]
                anomaly_values = [values[i] for i in anomaly_indices if 0 <= i < len(values)]
                anomaly_times = [ts_datetime[i] for i in anomaly_indices if 0 <= i < len(ts_datetime)]
                
                ax.scatter(anomaly_times, anomaly_values, color='red', s=80, label="Anomalies", zorder=5)
            
            # Plot forecast if available
            if "forecast" in results and "forecast_values" in results["forecast"]:
                forecast = results["forecast"]
                
                # Convert forecast timestamps to datetime
                forecast_ts = []
                for ts in forecast["forecast_timestamps"]:
                    if isinstance(ts, str):
                        forecast_ts.append(datetime.datetime.fromisoformat(ts))
                    else:
                        forecast_ts.append(ts)
                
                # Plot forecast
                ax.plot(forecast_ts, forecast["forecast_values"], 'g--', label="Forecast")
                
                # Plot confidence intervals
                if "lower_bound" in forecast and "upper_bound" in forecast:
                    ax.fill_between(
                        forecast_ts,
                        forecast["lower_bound"],
                        forecast["upper_bound"],
                        color='green',
                        alpha=0.2,
                        label="Confidence Interval"
                    )
            
            # Add trend information if available
            if "trend" in results and "direction" in results["trend"]:
                trend = results["trend"]
                
                if "predicted_values" in trend:
                    ax.plot(ts_datetime, trend["predicted_values"], 'r-', label="Trend", alpha=0.7)
                
                trend_text = f"Trend: {trend['direction'].capitalize()}"
                if "confidence" in trend:
                    trend_text += f" (conf: {trend['confidence']:.2f})"
                
                ax.text(0.02, 0.02, trend_text, transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.7))
            
            # Configure plot
            ax.set_title(f"{metric_name} Analysis")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.legend(loc="upper left")
            
            # Format date labels
            fig.autofmt_xdate()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            for fmt in self.config["visualization"]["formats"]:
                filename = f"{metric_name}_overview_{timestamp_str}.{fmt}"
                plt.savefig(os.path.join(metric_dir, filename))
            
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Error generating overview visualization for {metric_name}: {e}")
        
        # Create anomaly detail plot if there are anomalies
        if "anomalies" in results and results["anomalies"]:
            try:
                fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config["visualization"]["dpi"])
                
                # Plot actual values
                ax.plot(ts_datetime, values, 'b-', label="Actual", alpha=0.7)
                
                # Get anomalies
                anomalies = results["anomalies"]
                anomaly_indices = [a["index"] for a in anomalies]
                anomaly_values = [values[i] for i in anomaly_indices if 0 <= i < len(values)]
                anomaly_times = [ts_datetime[i] for i in anomaly_indices if 0 <= i < len(ts_datetime)]
                
                # Plot anomalies
                for i, anomaly in enumerate(anomalies):
                    if 0 <= anomaly["index"] < len(values) and 0 <= anomaly["index"] < len(ts_datetime):
                        idx = anomaly["index"]
                        confidence = anomaly.get("confidence", 0.5)
                        method = anomaly.get("method", "unknown")
                        
                        # Color based on confidence
                        color = plt.cm.RdYlGn_r(confidence)
                        
                        # Plot individual anomaly
                        ax.scatter(ts_datetime[idx], values[idx], color=color, s=100, zorder=5)
                        
                        # Add annotation for method
                        ax.annotate(
                            method,
                            (ts_datetime[idx], values[idx]),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8
                        )
                
                # Configure plot
                ax.set_title(f"{metric_name} Anomalies ({len(anomalies)} detected)")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.grid(True)
                
                # Add legend for methods
                methods = set(a.get("method", "unknown") for a in anomalies)
                for method in methods:
                    ax.scatter([], [], color='red', s=50, label=f"{method.capitalize()}")
                ax.legend(loc="upper left")
                
                # Format date labels
                fig.autofmt_xdate()
                
                # Adjust layout
                plt.tight_layout()
                
                # Save the figure
                for fmt in self.config["visualization"]["formats"]:
                    filename = f"{metric_name}_anomalies_{timestamp_str}.{fmt}"
                    plt.savefig(os.path.join(metric_dir, filename))
                
                plt.close(fig)
                
            except Exception as e:
                logger.warning(f"Error generating anomaly visualization for {metric_name}: {e}")
    
    def _analysis_loop(self, interval: int):
        """
        Background thread for periodic metric analysis.
        
        Args:
            interval: Analysis interval in seconds
        """
        logger.info(f"Starting analysis loop with {interval}s interval")
        
        while not self.stop_event.is_set():
            try:
                # Check if we have data to analyze
                if len(self.timestamps) >= self.config["anomaly_detection"]["min_data_points"]:
                    # Perform analysis
                    self.analyze_metrics()
                    
                    logger.debug("Periodic analysis completed")
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
            
            # Wait for next interval or until stopped
            self.stop_event.wait(interval)
    
    def _check_retraining(self):
        """
        Check if models should be retrained based on configuration.
        """
        if not self.config["model_persistence"]["enabled"]:
            return
        
        # Check if we have a last training time
        if self.last_training_time is None:
            self._train_models()
            return
        
        # Check if retraining interval has passed
        now = datetime.datetime.now()
        interval_hours = self.config["model_persistence"]["retraining_interval_hours"]
        elapsed = (now - self.last_training_time).total_seconds() / 3600
        
        if elapsed >= interval_hours:
            self._train_models()
    
    def _train_models(self):
        """
        Train or retrain all models.
        """
        logger.info("Training machine learning models")
        
        try:
            # Check if we have enough data
            min_data_points = self.config["anomaly_detection"]["min_data_points"]
            if len(self.timestamps) < min_data_points:
                logger.warning(f"Not enough data points for training: {len(self.timestamps)} < {min_data_points}")
                return
            
            # Train models for each metric
            for metric_name in self.config["metrics"]:
                if metric_name not in self.data:
                    continue
                
                values = np.array(self.data[metric_name])
                
                # Skip if not enough data for this metric
                if len(values) < min_data_points:
                    continue
                
                # Skip metrics with all NaN values
                if np.all(np.isnan(values)):
                    continue
                
                # Replace NaN values with interpolation or previous values
                values = self._handle_nan_values(values)
                
                # Reshape for scikit-learn
                X = values.reshape(-1, 1)
                
                # Train Isolation Forest model if enabled
                if "isolation_forest" in self.config["anomaly_detection"]["methods"] and SKLEARN_AVAILABLE:
                    try:
                        # Get configuration
                        config = self.config["anomaly_detection"]["isolation_forest"]
                        
                        # Initialize model
                        model = IsolationForest(
                            contamination=config["contamination"],
                            n_estimators=config["n_estimators"],
                            random_state=config["random_state"]
                        )
                        
                        # Fit model
                        model.fit(X)
                        
                        # Store model
                        self.models[f"{metric_name}_isolation_forest"] = model
                        
                        logger.debug(f"Trained Isolation Forest model for {metric_name}")
                        
                    except Exception as e:
                        logger.warning(f"Error training Isolation Forest model for {metric_name}: {e}")
                
                # Train ARIMA model if enabled
                if "arima" in self.config["time_series_analysis"]["methods"] and STATSMODELS_AVAILABLE:
                    try:
                        # Skip if not enough data
                        min_ts_points = max(self.config["time_series_analysis"]["forecast_horizon"] * 2, 8)
                        if len(values) < min_ts_points:
                            continue
                        
                        # Get configuration
                        config = self.config["time_series_analysis"]["arima"]
                        
                        # Create and fit ARIMA model
                        model = ARIMA(
                            values,
                            order=config["order"]
                        )
                        model_fit = model.fit()
                        
                        # Store model
                        self.models[f"{metric_name}_arima"] = model_fit
                        
                        logger.debug(f"Trained ARIMA model for {metric_name}")
                        
                    except Exception as e:
                        logger.warning(f"Error training ARIMA model for {metric_name}: {e}")
            
            # Update last training time
            self.last_training_time = datetime.datetime.now()
            
            # Save models
            self._save_models()
            
            logger.info(f"Model training completed. {len(self.models)} models trained.")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
    
    def _save_models(self):
        """
        Save trained models to disk.
        """
        if not self.config["model_persistence"]["enabled"]:
            return
        
        try:
            # Create directory if needed
            model_dir = self.config["model_persistence"]["directory"]
            os.makedirs(model_dir, exist_ok=True)
            
            # Create timestamp for files
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save models
            model_file = os.path.join(model_dir, f"ml_models_{timestamp_str}.pkl")
            
            with open(model_file, 'wb') as f:
                pickle.dump(self.models, f)
            
            logger.info(f"Saved {len(self.models)} models to {model_file}")
            
            # Clean up old model files
            model_files = [f for f in os.listdir(model_dir) if f.startswith("ml_models_") and f.endswith(".pkl")]
            if len(model_files) > self.config["model_persistence"]["max_models"]:
                # Sort by name (which contains timestamp)
                model_files.sort()
                
                # Remove oldest files
                for file in model_files[:-(self.config["model_persistence"]["max_models"])]:
                    try:
                        os.remove(os.path.join(model_dir, file))
                    except Exception:
                        pass
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """
        Load trained models from disk.
        """
        if not self.config["model_persistence"]["enabled"]:
            return
        
        try:
            # Check if model directory exists
            model_dir = self.config["model_persistence"]["directory"]
            if not os.path.exists(model_dir):
                return
            
            # Find model files
            model_files = [f for f in os.listdir(model_dir) if f.startswith("ml_models_") and f.endswith(".pkl")]
            
            if not model_files:
                return
            
            # Sort by name (which contains timestamp)
            model_files.sort(reverse=True)
            
            # Load most recent models
            model_file = os.path.join(model_dir, model_files[0])
            
            with open(model_file, 'rb') as f:
                self.models = pickle.load(f)
            
            # Set last training time from filename
            try:
                timestamp_str = model_files[0].replace("ml_models_", "").replace(".pkl", "")
                self.last_training_time = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except Exception:
                self.last_training_time = datetime.datetime.now()
            
            logger.info(f"Loaded {len(self.models)} models from {model_file}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            
            # Reset models in case of error
            self.models = {}
    
    def export_prometheus_config(self, filename: str = "prometheus_ml_config.yml") -> str:
        """
        Export Prometheus configuration for ML metrics.
        
        Args:
            filename: Name of the configuration file to create
        
        Returns:
            Path to the created configuration file
        """
        try:
            config = {
                "scrape_configs": [
                    {
                        "job_name": "ml_anomaly_detection",
                        "scrape_interval": "15s",
                        "static_configs": [
                            {
                                "targets": ["localhost:9100"]
                            }
                        ],
                        "metric_relabel_configs": [
                            {
                                "source_labels": ["__name__"],
                                "regex": "ml_anomaly_detection_.*",
                                "action": "keep"
                            }
                        ]
                    }
                ]
            }
            
            with open(filename, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Exported Prometheus configuration to {filename}")
            return os.path.abspath(filename)
            
        except Exception as e:
            logger.error(f"Error exporting Prometheus configuration: {e}")
            return None


def create_grafana_dashboard(
    metrics: List[str],
    output_file: str = "grafana_ml_dashboard.json"
) -> str:
    """
    Create a Grafana dashboard for ML anomaly detection metrics.
    
    Args:
        metrics: List of metrics to include in dashboard
        output_file: Name of the output JSON file
    
    Returns:
        Path to the created dashboard file
    """
    try:
        # Base dashboard configuration
        dashboard = {
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": "-- Grafana --",
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "type": "dashboard"
                    }
                ]
            },
            "editable": True,
            "gnetId": None,
            "graphTooltip": 0,
            "id": None,
            "links": [],
            "panels": [],
            "refresh": "10s",
            "schemaVersion": 27,
            "style": "dark",
            "tags": ["ml", "anomaly-detection", "distributed-testing"],
            "templating": {
                "list": []
            },
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "timepicker": {},
            "timezone": "",
            "title": "ML Anomaly Detection Dashboard",
            "uid": "ml-anomaly-detection",
            "version": 1
        }
        
        # Add overview panels
        panel_id = 1
        
        # Add summary panel
        dashboard["panels"].append({
            "datasource": "Prometheus",
            "fieldConfig": {
                "defaults": {
                    "color": {
                        "mode": "thresholds"
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {
                                "color": "green",
                                "value": None
                            },
                            {
                                "color": "yellow",
                                "value": 1
                            },
                            {
                                "color": "red",
                                "value": 5
                            }
                        ]
                    },
                    "unit": "none"
                },
                "overrides": []
            },
            "gridPos": {
                "h": 8,
                "w": 24,
                "x": 0,
                "y": 0
            },
            "id": panel_id,
            "options": {
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": [
                        "lastNotNull"
                    ],
                    "fields": "",
                    "values": False
                },
                "text": {},
                "textMode": "auto"
            },
            "pluginVersion": "7.5.3",
            "targets": [
                {
                    "expr": "ml_anomaly_detection_anomalies_total",
                    "interval": "",
                    "legendFormat": "Total Anomalies",
                    "refId": "A"
                },
                {
                    "expr": "ml_anomaly_detection_metrics_with_anomalies",
                    "interval": "",
                    "legendFormat": "Metrics with Anomalies",
                    "refId": "B"
                },
                {
                    "expr": "ml_anomaly_detection_data_points",
                    "interval": "",
                    "legendFormat": "Data Points",
                    "refId": "C"
                }
            ],
            "title": "ML Anomaly Detection Overview",
            "type": "stat"
        })
        panel_id += 1
        
        # Add panels for each metric
        y_pos = 8
        for i, metric in enumerate(metrics):
            # Add metric panel
            dashboard["panels"].append({
                "aliasColors": {},
                "bars": False,
                "dashLength": 10,
                "dashes": False,
                "datasource": "Prometheus",
                "fieldConfig": {
                    "defaults": {},
                    "overrides": []
                },
                "fill": 1,
                "fillGradient": 0,
                "gridPos": {
                    "h": 8,
                    "w": 12,
                    "x": i % 2 * 12,
                    "y": y_pos + (i // 2) * 8
                },
                "hiddenSeries": False,
                "id": panel_id,
                "legend": {
                    "avg": False,
                    "current": True,
                    "max": False,
                    "min": False,
                    "show": True,
                    "total": False,
                    "values": True
                },
                "lines": True,
                "linewidth": 1,
                "nullPointMode": "null",
                "options": {
                    "alertThreshold": True
                },
                "percentage": False,
                "pluginVersion": "7.5.3",
                "pointradius": 2,
                "points": False,
                "renderer": "flot",
                "seriesOverrides": [],
                "spaceLength": 10,
                "stack": False,
                "steppedLine": False,
                "targets": [
                    {
                        "expr": f"drm_{metric}",
                        "interval": "",
                        "legendFormat": metric.replace("_", " ").title(),
                        "refId": "A"
                    },
                    {
                        "expr": f"ml_anomaly_detection_{metric}_forecast",
                        "interval": "",
                        "legendFormat": "Forecast",
                        "refId": "B"
                    }
                ],
                "thresholds": [],
                "timeRegions": [],
                "title": f"{metric.replace('_', ' ').title()} with Forecast",
                "tooltip": {
                    "shared": True,
                    "sort": 0,
                    "value_type": "individual"
                },
                "type": "graph",
                "xaxis": {
                    "buckets": None,
                    "mode": "time",
                    "name": None,
                    "show": True,
                    "values": []
                },
                "yaxes": [
                    {
                        "format": "short",
                        "label": None,
                        "logBase": 1,
                        "max": None,
                        "min": None,
                        "show": True
                    },
                    {
                        "format": "short",
                        "label": None,
                        "logBase": 1,
                        "max": None,
                        "min": None,
                        "show": True
                    }
                ],
                "yaxis": {
                    "align": False,
                    "alignLevel": None
                }
            })
            panel_id += 1
            
            # Add anomaly panel
            dashboard["panels"].append({
                "aliasColors": {},
                "bars": False,
                "dashLength": 10,
                "dashes": False,
                "datasource": "Prometheus",
                "fieldConfig": {
                    "defaults": {},
                    "overrides": []
                },
                "fill": 1,
                "fillGradient": 0,
                "gridPos": {
                    "h": 8,
                    "w": 12,
                    "x": (i % 2) * 12,
                    "y": y_pos + (i // 2) * 8 + 4
                },
                "hiddenSeries": False,
                "id": panel_id,
                "legend": {
                    "avg": False,
                    "current": True,
                    "max": False,
                    "min": False,
                    "show": True,
                    "total": False,
                    "values": True
                },
                "lines": False,
                "linewidth": 1,
                "nullPointMode": "null",
                "options": {
                    "alertThreshold": True
                },
                "percentage": False,
                "pluginVersion": "7.5.3",
                "pointradius": 5,
                "points": True,
                "renderer": "flot",
                "seriesOverrides": [],
                "spaceLength": 10,
                "stack": False,
                "steppedLine": False,
                "targets": [
                    {
                        "expr": f"ml_anomaly_detection_{metric}_anomalies",
                        "interval": "",
                        "legendFormat": "Anomalies",
                        "refId": "A"
                    }
                ],
                "thresholds": [],
                "timeRegions": [],
                "title": f"{metric.replace('_', ' ').title()} Anomalies",
                "tooltip": {
                    "shared": True,
                    "sort": 0,
                    "value_type": "individual"
                },
                "type": "graph",
                "xaxis": {
                    "buckets": None,
                    "mode": "time",
                    "name": None,
                    "show": True,
                    "values": []
                },
                "yaxes": [
                    {
                        "format": "short",
                        "label": "Count",
                        "logBase": 1,
                        "max": None,
                        "min": "0",
                        "show": True
                    },
                    {
                        "format": "short",
                        "label": None,
                        "logBase": 1,
                        "max": None,
                        "min": None,
                        "show": True
                    }
                ],
                "yaxis": {
                    "align": False,
                    "alignLevel": None
                }
            })
            panel_id += 1
        
        # Save dashboard to file
        with open(output_file, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        logger.info(f"Created Grafana dashboard at {output_file}")
        return os.path.abspath(output_file)
        
    except Exception as e:
        logger.error(f"Error creating Grafana dashboard: {e}")
        return None


def main():
    """Command-line entry point for ML anomaly detection module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Anomaly Detection for Distributed Testing Framework")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--metrics-port", type=int, default=9100, help="Port for Prometheus metrics")
    parser.add_argument("--interval", type=int, default=60, help="Analysis interval in seconds")
    parser.add_argument("--export-grafana", action="store_true", help="Export Grafana dashboard")
    parser.add_argument("--export-prometheus", action="store_true", help="Export Prometheus configuration")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return 1
    
    # Create ML anomaly detection instance
    detector = MLAnomalyDetection(config=config)
    
    # Export configurations if requested
    if args.export_grafana:
        metrics = detector.config["metrics"]
        create_grafana_dashboard(metrics)
    
    if args.export_prometheus:
        detector.export_prometheus_config()
    
    # Start the detector
    detector.start(interval=args.interval)
    
    try:
        # Keep running until interrupted
        print("ML Anomaly Detection running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Stop the detector
        detector.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())