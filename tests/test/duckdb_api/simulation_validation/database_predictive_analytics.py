#!/usr/bin/env python3
"""
Database Predictive Analytics for the Simulation Accuracy and Validation Framework.

This module provides predictive analytics capabilities specifically for database performance
metrics in the DuckDB database. It leverages time series forecasting, statistical methods,
and machine learning techniques to predict future database performance trends, proactively
identifying potential issues before they occur.

Features:
1. Time series forecasting for database performance metrics
2. Anomaly detection for early identification of performance issues
3. Confidence intervals for predictions to quantify uncertainty
4. Scenario-based projections (best/expected/worst case)
5. Visualization of historical trends and forecasts
6. Alert thresholds for predicted values (not just current values)
7. Integration with AutomatedOptimizationManager for proactive optimization
"""

import logging
import numpy as np
import datetime
import matplotlib.pyplot as plt
import io
import base64
import os
import warnings
import contextlib
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("database_predictive_analytics")

# Context manager for temporarily suppressing specific warnings
@contextlib.contextmanager
def suppress_warnings(warning_categories=None):
    """
    Context manager to temporarily suppress specific warnings.
    
    Args:
        warning_categories: List of warning categories to suppress.
            If None, suppresses all warnings.
    """
    with warnings.catch_warnings():
        if warning_categories:
            for category in warning_categories:
                warnings.filterwarnings("ignore", category=category)
        else:
            warnings.filterwarnings("ignore")
        yield

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available, some forecasting features will be limited")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, statistical forecasting will be limited")


class DatabasePredictiveAnalytics:
    """
    Predictive analytics for database performance metrics.
    
    This class implements various forecasting and analysis methods specifically 
    for database performance metrics, allowing proactive identification of potential
    issues before they impact system performance.
    """
    
    def __init__(
        self,
        automated_optimization_manager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the database predictive analytics.
        
        Args:
            automated_optimization_manager: AutomatedOptimizationManager instance
            config: Configuration options for predictive analytics
        """
        self.auto_manager = automated_optimization_manager
        self.config = config or {}
        self.hyperparameter_cache = {}
        
        # Default configuration
        default_config = {
            # Standard database metrics to forecast
            "metrics_to_forecast": [
                "query_time",
                "storage_size",
                "index_efficiency",
                "vacuum_status",
                "compression_ratio",
                "read_efficiency",
                "write_efficiency",
                "cache_performance"
            ],
            
            # Forecasting configuration
            "forecasting": {
                "short_term_horizon": 7,      # Days for short-term forecast
                "medium_term_horizon": 30,     # Days for medium-term forecast
                "long_term_horizon": 90,       # Days for long-term forecast
                "confidence_level": 0.95,      # Confidence level for prediction intervals
                "min_data_points": 10,         # Minimum data points needed for forecasting
                "forecast_methods": ["arima", "exponential_smoothing", "linear_regression"],
                "use_ensemble": True,          # Whether to use ensemble forecasting
                "auto_model_selection": True,  # Whether to automatically select the best model based on validation metrics
                "validation_size_percent": 25, # Percentage of historical data to use for validation (1-50)
                "hyperparameter_tuning": {     # Hyperparameter tuning configuration
                    "enabled": True,           # Whether to enable hyperparameter tuning
                    "search_method": "grid",   # "grid" for grid search, "random" for random search
                    "max_iterations": 10,      # Maximum iterations for hyperparameter tuning
                    "cv_folds": 3,             # Number of folds for cross-validation
                    "early_stopping": {        # Early stopping configuration to improve efficiency
                        "enabled": True,       # Whether to enable early stopping
                        "min_iterations": 3,   # Minimum iterations before early stopping can be triggered
                        "patience": 3,         # Number of iterations without improvement to trigger early stopping
                        "tolerance": 0.05      # Improvement threshold (fraction) below which is considered insignificant
                    },
                    "parameter_persistence": {  # Parameter persistence configuration
                        "enabled": True,        # Whether to enable parameter persistence
                        "storage_path": "./hyperparameters",  # Directory to store hyperparameters
                        "max_age_days": 30,     # Maximum age of stored parameters in days
                        "versioning": True,     # Whether to version parameters
                        "revalidate_after_days": 7,  # Days after which parameters should be revalidated
                        "force_revalidation": False, # Whether to force revalidation
                        "serialize_format": "json"   # Format to serialize parameters: "json" or "pickle"
                    },
                    "arima_params": {          # ARIMA model parameters
                        "p": [0, 1, 2],        # Autoregressive order values to try
                        "d": [0, 1],           # Difference order values to try
                        "q": [0, 1, 2]         # Moving average order values to try
                    },
                    "exp_smoothing_params": {  # Exponential smoothing parameters
                        "trend": [None, "add", "mul"], # Trend component options
                        "seasonal": [None, "add", "mul"], # Seasonal component options
                        "seasonal_periods": [7, 14],    # Seasonal periods to try
                        "damped_trend": [True, False]   # Whether to use damped trend
                    },
                    "linear_regression_params": { # Linear regression parameters
                        "fit_intercept": [True, False],    # Whether to fit intercept
                        "positive": [True, False]          # Whether to force positive coefficients
                    }
                }
            },
            
            # Anomaly detection configuration
            "anomaly_detection": {
                "enabled": True,
                "lookback_window": 30,         # Days of history to analyze
                "z_score_threshold": 3.0,      # Z-score threshold for anomaly detection
                "sensitivity": 0.95            # Sensitivity level (higher = more sensitive)
            },
            
            # Threshold configuration
            "thresholds": {
                # By default use same thresholds as AutomatedOptimizationManager
                "use_auto_manager_thresholds": True,
                
                # Additional thresholds for predicted values
                "predicted_threshold_factor": 0.8,  # Multiply normal threshold by this factor
                                                   # for predicted values (earlier warning)
                
                # Override with custom thresholds if needed
                "custom_thresholds": {}
            },
            
            # Visualization configuration
            "visualization": {
                "enabled": True,
                "theme": "light",         # light or dark
                "show_confidence_intervals": True,
                "forecast_colors": {
                    "historical": "#1f77b4",   # Blue
                    "forecast": "#ff7f0e",     # Orange
                    "lower_bound": "#2ca02c",  # Green
                    "upper_bound": "#d62728",  # Red
                    "anomalies": "#9467bd"     # Purple
                },
                "figure_size": (10, 6),
                "dpi": 100
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
    
    def forecast_database_metrics(
        self,
        horizon: str = "medium_term",
        specific_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Forecast database performance metrics for the specified horizon.
        
        Args:
            horizon: Forecast horizon ("short_term", "medium_term", or "long_term")
            specific_metrics: List of specific metrics to forecast (None for all)
            
        Returns:
            Dictionary containing forecast results for each metric
        """
        # Get metrics to forecast
        metrics_to_forecast = specific_metrics or self.config["metrics_to_forecast"]
        
        # Get forecast horizon in days
        if horizon == "short_term":
            days = self.config["forecasting"]["short_term_horizon"]
        elif horizon == "long_term":
            days = self.config["forecasting"]["long_term_horizon"]
        else:  # medium_term is default
            days = self.config["forecasting"]["medium_term_horizon"]
        
        # Get historical metrics data
        metrics_history = self.auto_manager.get_metrics_history()
        
        # Initialize results
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "horizon": horizon,
            "horizon_days": days,
            "forecasts": {},
            "anomalies": {},
            "status": "success",
            "warnings": []
        }
        
        # Check if we have metrics history data
        if not metrics_history:
            results["status"] = "error"
            results["message"] = "No metrics history data available for forecasting"
            return results
        
        # Process each metric
        for metric_name in metrics_to_forecast:
            # Check if we have history for this metric
            if metric_name not in metrics_history:
                results["warnings"].append(f"No history available for metric: {metric_name}")
                continue
            
            # Get metric history
            metric_history = metrics_history[metric_name]
            
            # Check if we have enough data points
            min_data_points = self.config["forecasting"]["min_data_points"]
            if len(metric_history) < min_data_points:
                results["warnings"].append(
                    f"Insufficient data points for metric {metric_name}. "
                    f"Required: {min_data_points}, Available: {len(metric_history)}"
                )
                continue
            
            # Extract values and timestamps
            values = []
            timestamps = []
            
            for entry in metric_history:
                if "value" in entry and entry["value"] is not None:
                    values.append(entry["value"])
                    if "timestamp" in entry:
                        try:
                            # Parse timestamp string to datetime
                            if isinstance(entry["timestamp"], str):
                                timestamp = datetime.datetime.fromisoformat(entry["timestamp"])
                            else:
                                timestamp = entry["timestamp"]
                            timestamps.append(timestamp)
                        except Exception as e:
                            logger.warning(f"Error parsing timestamp for {metric_name}: {e}")
                            # Use current time as fallback
                            timestamps.append(datetime.datetime.now())
            
            # Ensure values and timestamps are ordered by time (oldest first)
            if timestamps:
                sorted_indices = np.argsort(timestamps)
                values = [values[i] for i in sorted_indices]
                timestamps = [timestamps[i] for i in sorted_indices]
            
            # Generate forecasts
            try:
                forecast_result = self._generate_forecasts(metric_name, values, timestamps, days)
                results["forecasts"][metric_name] = forecast_result
                
                # Detect anomalies if enabled
                if self.config["anomaly_detection"]["enabled"]:
                    anomalies = self._detect_anomalies(
                        metric_name, values, timestamps, forecast_result
                    )
                    if anomalies:
                        results["anomalies"][metric_name] = anomalies
            except Exception as e:
                logger.error(f"Error forecasting metric {metric_name}: {e}")
                results["warnings"].append(f"Error forecasting metric {metric_name}: {e}")
        
        # Check if we have any successful forecasts
        if not results["forecasts"]:
            results["status"] = "warning"
            results["message"] = "No metrics could be successfully forecasted"
        
        return results
    
    def generate_forecast_visualizations(
        self,
        forecast_results: Dict[str, Any],
        output_format: str = "base64"
    ) -> Dict[str, Any]:
        """
        Generate visualizations of forecast results.
        
        Args:
            forecast_results: Results from forecast_database_metrics
            output_format: Format for visualization output ("base64", "file", or "object")
            
        Returns:
            Dictionary containing visualizations for each metric
        """
        if not self.config["visualization"]["enabled"]:
            return {"status": "skipped", "message": "Visualization is disabled in configuration"}
        
        # Check if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
        except ImportError:
            return {"status": "error", "message": "matplotlib not available for visualization"}
        
        # Initialize results
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "visualizations": {},
            "status": "success"
        }
        
        # Get visualization configuration
        fig_size = self.config["visualization"]["figure_size"]
        dpi = self.config["visualization"]["dpi"]
        show_ci = self.config["visualization"]["show_confidence_intervals"]
        
        # Set theme
        if self.config["visualization"]["theme"] == "dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        
        # Get colors
        colors = self.config["visualization"]["forecast_colors"]
        
        # Process each metric
        for metric_name, forecast in forecast_results.get("forecasts", {}).items():
            try:
                # Create a new figure
                fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
                
                # Extract historical data
                historical_dates = [datetime.datetime.fromisoformat(d) for d in forecast["historical_dates"]]
                historical_values = forecast["historical_values"]
                
                # Extract forecast data
                forecast_dates = [datetime.datetime.fromisoformat(d) for d in forecast["forecast_dates"]]
                forecast_values = forecast["forecast_values"]
                
                # Plot historical data
                ax.plot(historical_dates, historical_values, marker='o', linestyle='-', 
                        color=colors["historical"], label='Historical Data')
                
                # Plot forecast
                ax.plot(forecast_dates, forecast_values, marker='x', linestyle='--', 
                        color=colors["forecast"], label='Forecast')
                
                # Plot confidence intervals if available and enabled
                if show_ci and "lower_bound" in forecast and "upper_bound" in forecast:
                    lower_bound = forecast["lower_bound"]
                    upper_bound = forecast["upper_bound"]
                    
                    # Fill area between lower and upper bounds
                    ax.fill_between(forecast_dates, lower_bound, upper_bound, 
                                    color=colors["forecast"], alpha=0.2,
                                    label=f'{int(self.config["forecasting"]["confidence_level"]*100)}% Confidence Interval')
                
                # Plot anomalies if available
                if metric_name in forecast_results.get("anomalies", {}):
                    anomalies = forecast_results["anomalies"][metric_name]
                    
                    if "historical_anomalies" in anomalies:
                        # Get anomaly indices and values
                        anomaly_indices = anomalies["historical_anomalies"]["indices"]
                        anomaly_values = [historical_values[i] for i in anomaly_indices]
                        anomaly_dates = [historical_dates[i] for i in anomaly_indices]
                        
                        # Plot anomalies
                        ax.scatter(anomaly_dates, anomaly_values, color=colors["anomalies"],
                                   s=100, marker='*', label='Anomalies', zorder=5)
                    
                    if "forecast_anomalies" in anomalies:
                        # Get forecast anomaly indices and values
                        forecast_anomaly_indices = anomalies["forecast_anomalies"]["indices"]
                        forecast_anomaly_values = [forecast_values[i] for i in forecast_anomaly_indices]
                        forecast_anomaly_dates = [forecast_dates[i] for i in forecast_anomaly_indices]
                        
                        # Plot forecast anomalies
                        ax.scatter(forecast_anomaly_dates, forecast_anomaly_values, 
                                   color=colors["anomalies"], s=100, marker='X', 
                                   label='Predicted Anomalies', zorder=5)
                
                # Add threshold lines if available in forecast
                if "warning_threshold" in forecast:
                    ax.axhline(y=forecast["warning_threshold"], color='orange', 
                              linestyle=':', label='Warning Threshold')
                
                if "error_threshold" in forecast:
                    ax.axhline(y=forecast["error_threshold"], color='red', 
                              linestyle=':', label='Error Threshold')
                
                # Add predicted threshold lines if available
                if "predicted_warning_threshold" in forecast:
                    ax.axhline(y=forecast["predicted_warning_threshold"], color='orange', 
                              linestyle='--', label='Predicted Warning')
                
                if "predicted_error_threshold" in forecast:
                    ax.axhline(y=forecast["predicted_error_threshold"], color='red', 
                              linestyle='--', label='Predicted Error')
                
                # Format the plot
                ax.set_title(f'{metric_name} Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                
                # Format date axis
                date_formatter = DateFormatter('%Y-%m-%d')
                ax.xaxis.set_major_formatter(date_formatter)
                fig.autofmt_xdate()
                
                # Add legend
                ax.legend(loc='best')
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Tight layout
                fig.tight_layout()
                
                # Save visualization based on output format
                if output_format == "base64":
                    # Save to in-memory byte buffer as PNG
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    
                    # Convert to base64
                    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    results["visualizations"][metric_name] = {
                        "format": "base64",
                        "image": image_base64
                    }
                    
                    # Close buffer
                    buf.close()
                
                elif output_format == "file":
                    try:
                        # Create directory if it doesn't exist
                        output_dir = Path("./visualizations")
                        output_dir.mkdir(exist_ok=True)
                        
                        # Use a direct simple approach
                        filename = f"./visualizations/{metric_name}_forecast_class.png"
                        
                        # Save with a simple method
                        fig.savefig(filename)
                        
                        # Print for debugging
                        print(f"DEBUG: Saved visualization to {filename}")
                        
                        # Check if file exists
                        if os.path.exists(filename):
                            print(f"DEBUG: File {filename} exists, size: {os.path.getsize(filename)}")
                        else:
                            print(f"DEBUG: File {filename} does not exist")
                        
                        results["visualizations"][metric_name] = {
                            "format": "file",
                            "filename": filename
                        }
                    except Exception as e:
                        print(f"DEBUG ERROR: Failed to save visualization: {e}")
                        logger.error(f"Failed to save visualization: {e}")
                
                elif output_format == "object":
                    # Return the figure object directly
                    results["visualizations"][metric_name] = {
                        "format": "object",
                        "figure": fig
                    }
                
                # Close figure to free memory (unless returning the object)
                if output_format != "object":
                    plt.close(fig)
                
            except Exception as e:
                logger.error(f"Error generating visualization for {metric_name}: {e}")
                results["visualizations"][metric_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        return results
    
    def check_predicted_thresholds(
        self,
        forecast_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if forecasted values will exceed thresholds in the future.
        
        Args:
            forecast_results: Results from forecast_database_metrics
            
        Returns:
            Dictionary containing alerts for metrics predicted to exceed thresholds
        """
        # Initialize results
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "alerts": {},
            "status": "success"
        }
        
        # Get thresholds configuration
        use_auto_manager_thresholds = self.config["thresholds"]["use_auto_manager_thresholds"]
        predicted_threshold_factor = self.config["thresholds"]["predicted_threshold_factor"]
        custom_thresholds = self.config["thresholds"]["custom_thresholds"]
        
        # Get default thresholds from auto manager if enabled
        default_thresholds = {}
        if use_auto_manager_thresholds:
            default_thresholds = self.auto_manager.thresholds
        
        # Process each metric in forecast results
        for metric_name, forecast in forecast_results.get("forecasts", {}).items():
            # Get thresholds for this metric
            if metric_name in custom_thresholds:
                thresholds = custom_thresholds[metric_name]
            elif metric_name in default_thresholds:
                thresholds = default_thresholds[metric_name]
            else:
                # No thresholds for this metric
                continue
            
            # Extract warning and error thresholds
            warning_threshold = thresholds.get("warning")
            error_threshold = thresholds.get("error")
            
            if warning_threshold is None and error_threshold is None:
                # No thresholds defined
                continue
            
            # Calculate predicted thresholds (more conservative)
            predicted_warning_threshold = None
            predicted_error_threshold = None
            
            # For metrics where higher values are better
            if metric_name in ["index_efficiency", "vacuum_status", "compression_ratio", "cache_performance"]:
                # For these metrics, thresholds are minimums (higher values are better)
                if warning_threshold is not None:
                    predicted_warning_threshold = warning_threshold / predicted_threshold_factor
                if error_threshold is not None:
                    predicted_error_threshold = error_threshold / predicted_threshold_factor
            else:
                # For metrics where lower values are better
                if warning_threshold is not None:
                    predicted_warning_threshold = warning_threshold * predicted_threshold_factor
                if error_threshold is not None:
                    predicted_error_threshold = error_threshold * predicted_threshold_factor
            
            # Add thresholds to forecast for visualization
            forecast["warning_threshold"] = warning_threshold
            forecast["error_threshold"] = error_threshold
            forecast["predicted_warning_threshold"] = predicted_warning_threshold
            forecast["predicted_error_threshold"] = predicted_error_threshold
            
            # Check if forecasted values exceed thresholds
            forecast_values = forecast["forecast_values"]
            
            # Initialize alerts for this metric
            metric_alerts = []
            
            # For metrics where higher values are better
            if metric_name in ["index_efficiency", "vacuum_status", "compression_ratio", "cache_performance"]:
                # Check against predicted error threshold
                if predicted_error_threshold is not None:
                    for i, value in enumerate(forecast_values):
                        if value <= predicted_error_threshold:
                            forecast_date = forecast["forecast_dates"][i]
                            metric_alerts.append({
                                "severity": "error",
                                "threshold": predicted_error_threshold,
                                "forecasted_value": value,
                                "forecast_date": forecast_date,
                                "days_until": i + 1,
                                "message": f"{metric_name} predicted to fall below error threshold "
                                          f"({value:.2f} <= {predicted_error_threshold:.2f}) on {forecast_date}"
                            })
                            break  # Only report the first occurrence
                
                # Check against predicted warning threshold
                if predicted_warning_threshold is not None and not metric_alerts:
                    for i, value in enumerate(forecast_values):
                        if value <= predicted_warning_threshold:
                            forecast_date = forecast["forecast_dates"][i]
                            metric_alerts.append({
                                "severity": "warning",
                                "threshold": predicted_warning_threshold,
                                "forecasted_value": value,
                                "forecast_date": forecast_date,
                                "days_until": i + 1,
                                "message": f"{metric_name} predicted to fall below warning threshold "
                                          f"({value:.2f} <= {predicted_warning_threshold:.2f}) on {forecast_date}"
                            })
                            break  # Only report the first occurrence
            else:
                # For metrics where lower values are better
                
                # Check against predicted error threshold
                if predicted_error_threshold is not None:
                    for i, value in enumerate(forecast_values):
                        if value >= predicted_error_threshold:
                            forecast_date = forecast["forecast_dates"][i]
                            metric_alerts.append({
                                "severity": "error",
                                "threshold": predicted_error_threshold,
                                "forecasted_value": value,
                                "forecast_date": forecast_date,
                                "days_until": i + 1,
                                "message": f"{metric_name} predicted to exceed error threshold "
                                          f"({value:.2f} >= {predicted_error_threshold:.2f}) on {forecast_date}"
                            })
                            break  # Only report the first occurrence
                
                # Check against predicted warning threshold
                if predicted_warning_threshold is not None and not metric_alerts:
                    for i, value in enumerate(forecast_values):
                        if value >= predicted_warning_threshold:
                            forecast_date = forecast["forecast_dates"][i]
                            metric_alerts.append({
                                "severity": "warning",
                                "threshold": predicted_warning_threshold,
                                "forecasted_value": value,
                                "forecast_date": forecast_date,
                                "days_until": i + 1,
                                "message": f"{metric_name} predicted to exceed warning threshold "
                                          f"({value:.2f} >= {predicted_warning_threshold:.2f}) on {forecast_date}"
                            })
                            break  # Only report the first occurrence
            
            # Add alerts for this metric if any were found
            if metric_alerts:
                results["alerts"][metric_name] = metric_alerts
        
        # Update status if alerts found
        if results["alerts"]:
            # Check for error severity alerts
            has_error = any(
                alert[0]["severity"] == "error"
                for alerts in results["alerts"].values()
                for alert in [alerts]
            )
            
            results["status"] = "error" if has_error else "warning"
        
        return results
    
    def _get_parameter_storage_path(self) -> Path:
        """
        Get the configured storage path for hyperparameters.
        
        Returns:
            Path object for the hyperparameter storage directory
        """
        # Get storage path from config
        storage_path = self.config.get("forecasting", {}).get(
            "hyperparameter_tuning", {}).get(
            "parameter_persistence", {}).get("storage_path", "./hyperparameters")
        
        # Convert to Path object
        path = Path(storage_path)
        
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        
        return path
    
    def _get_parameter_key(self, metric_name: str, model_type: str, data_signature: Optional[str] = None) -> str:
        """
        Generate a unique key for storing/retrieving hyperparameters.
        
        Args:
            metric_name: Name of the metric
            model_type: Type of model (e.g., 'arima', 'exponential_smoothing', 'linear_regression')
            data_signature: Optional signature representing the dataset characteristics
            
        Returns:
            String key for parameter storage
        """
        if data_signature is None:
            # Create a simple signature based on metric name and model type
            return f"{metric_name}_{model_type}"
        else:
            # Use provided data signature
            return f"{metric_name}_{model_type}_{data_signature}"
    
    def _create_data_signature(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        Create a signature for the dataset to help identify if data characteristics have changed.
        
        Args:
            data: Dataset to create signature for
            
        Returns:
            String signature representing dataset characteristics
        """
        if isinstance(data, pd.DataFrame):
            # For pandas DataFrame
            # Use a combination of length, mean, std, min, max
            length = len(data)
            if length == 0:
                return "empty"
            
            # Extract numeric values for statistics
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) == 0:
                # No numeric columns
                return f"length_{length}_nonnumeric"
            
            # Use basic statistics for signature
            stats = {
                "length": length,
                "mean": round(float(numeric_data.mean().mean()), 3),
                "std": round(float(numeric_data.std().mean()), 3),
                "min": round(float(numeric_data.min().min()), 3),
                "max": round(float(numeric_data.max().max()), 3)
            }
            
            # Create a hash of the statistics
            signature_str = json.dumps(stats, sort_keys=True)
            return hashlib.md5(signature_str.encode()).hexdigest()[:10]
        
        elif isinstance(data, np.ndarray):
            # For numpy array
            if data.size == 0:
                return "empty"
            
            # Use basic statistics for signature
            stats = {
                "shape": str(data.shape),
                "mean": round(float(np.mean(data)), 3),
                "std": round(float(np.std(data)), 3),
                "min": round(float(np.min(data)), 3),
                "max": round(float(np.max(data)), 3)
            }
            
            # Create a hash of the statistics
            signature_str = json.dumps(stats, sort_keys=True)
            return hashlib.md5(signature_str.encode()).hexdigest()[:10]
        
        else:
            # For other types
            return "unknown"
    
    def _save_parameters(
        self, 
        metric_name: str, 
        model_type: str, 
        parameters: Any, 
        performance_metric: Optional[float] = None,
        data_signature: Optional[str] = None
    ) -> bool:
        """
        Save hyperparameters to persistent storage.
        
        Args:
            metric_name: Name of the metric
            model_type: Type of model
            parameters: Parameters to save
            performance_metric: Optional performance metric value (e.g., MAPE)
            data_signature: Optional signature of the dataset
            
        Returns:
            Boolean indicating success or failure
        """
        # Check if parameter persistence is enabled
        persistence_config = self.config.get("forecasting", {}).get(
            "hyperparameter_tuning", {}).get("parameter_persistence", {})
        
        if not persistence_config.get("enabled", True):
            return False
        
        try:
            # Generate parameter key
            parameter_key = self._get_parameter_key(metric_name, model_type, data_signature)
            
            # Get storage path
            storage_path = self._get_parameter_storage_path()
            
            # Prepare data to save
            save_data = {
                "metric_name": metric_name,
                "model_type": model_type,
                "parameters": parameters,
                "performance_metric": performance_metric,
                "data_signature": data_signature,
                "timestamp": datetime.datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # Determine file path and format
            serialize_format = persistence_config.get("serialize_format", "json")
            
            if serialize_format == "json":
                # For JSON serialization
                file_path = storage_path / f"{parameter_key}.json"
                
                # Convert parameters to JSON-serializable format if needed
                if hasattr(parameters, "__dict__"):
                    save_data["parameters"] = parameters.__dict__
                
                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2, default=str)
            
            elif serialize_format == "pickle":
                # For pickle serialization
                file_path = storage_path / f"{parameter_key}.pkl"
                
                # Save to file
                with open(file_path, 'wb') as f:
                    pickle.dump(save_data, f)
            
            else:
                logger.warning(f"Unsupported serialization format: {serialize_format}")
                return False
            
            # Also store in memory cache
            self.hyperparameter_cache[parameter_key] = save_data
            
            logger.debug(f"Saved hyperparameters for {metric_name} {model_type} to {file_path}")
            return True
        
        except Exception as e:
            logger.warning(f"Error saving hyperparameters: {e}")
            return False
    
    def _load_parameters(
        self, 
        metric_name: str, 
        model_type: str, 
        data_signature: Optional[str] = None,
        force_reload: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load hyperparameters from persistent storage.
        
        Args:
            metric_name: Name of the metric
            model_type: Type of model
            data_signature: Optional signature of the dataset
            force_reload: Whether to force reload from disk instead of using cache
            
        Returns:
            Dictionary with loaded parameters or None if not found or invalid
        """
        # Check if parameter persistence is enabled
        persistence_config = self.config.get("forecasting", {}).get(
            "hyperparameter_tuning", {}).get("parameter_persistence", {})
        
        if not persistence_config.get("enabled", True):
            return None
        
        try:
            # Generate parameter key
            parameter_key = self._get_parameter_key(metric_name, model_type, data_signature)
            
            # Check memory cache first if not forcing reload
            if not force_reload and parameter_key in self.hyperparameter_cache:
                logger.debug(f"Using cached hyperparameters for {metric_name} {model_type}")
                return self.hyperparameter_cache[parameter_key]
            
            # Get storage path
            storage_path = self._get_parameter_storage_path()
            
            # Determine potential file paths
            json_path = storage_path / f"{parameter_key}.json"
            pickle_path = storage_path / f"{parameter_key}.pkl"
            
            # Load parameters based on available file
            if json_path.exists():
                with open(json_path, 'r') as f:
                    parameters = json.load(f)
            elif pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    parameters = pickle.load(f)
            else:
                # No saved parameters found
                return None
            
            # Validate the loaded parameters
            if self._validate_parameters(parameters):
                # Store in memory cache
                self.hyperparameter_cache[parameter_key] = parameters
                logger.debug(f"Loaded hyperparameters for {metric_name} {model_type}")
                return parameters
            else:
                logger.debug(f"Loaded parameters failed validation for {metric_name} {model_type}")
                return None
            
        except Exception as e:
            logger.warning(f"Error loading hyperparameters: {e}")
            return None
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate loaded parameters to ensure they're still usable.
        
        Args:
            parameters: Dictionary of loaded parameters
            
        Returns:
            Boolean indicating whether parameters are valid
        """
        # Check if the necessary fields are present
        required_fields = ["metric_name", "model_type", "parameters", "timestamp"]
        if not all(field in parameters for field in required_fields):
            return False
        
        # Check if parameters are too old
        persistence_config = self.config.get("forecasting", {}).get(
            "hyperparameter_tuning", {}).get("parameter_persistence", {})
        
        max_age_days = persistence_config.get("max_age_days", 30)
        
        try:
            # Parse timestamp
            timestamp = datetime.datetime.fromisoformat(parameters["timestamp"])
            
            # Calculate age
            age = datetime.datetime.now() - timestamp
            
            # Check if too old
            if age.days > max_age_days:
                logger.debug(f"Parameters too old ({age.days} days) - max age is {max_age_days} days")
                return False
            
            # Check if needs revalidation
            revalidate_after_days = persistence_config.get("revalidate_after_days", 7)
            force_revalidation = persistence_config.get("force_revalidation", False)
            
            if force_revalidation or age.days > revalidate_after_days:
                logger.debug(f"Parameters need revalidation (age: {age.days} days)")
                # Instead of returning False, we can flag these for revalidation
                # We'll still use them but will retune soon
                parameters["needs_revalidation"] = True
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating parameters: {e}")
            return False
    
    def recommend_proactive_actions(
        self,
        forecast_results: Dict[str, Any],
        threshold_alerts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recommend proactive actions based on forecast and threshold alerts.
        
        Args:
            forecast_results: Results from forecast_database_metrics
            threshold_alerts: Results from check_predicted_thresholds
            
        Returns:
            Dictionary containing recommended proactive actions
        """
        # Initialize results
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "recommendations": [],
            "proactive_actions": {},
            "status": "success"
        }
        
        # Process alerts to determine recommendations
        for metric_name, alerts in threshold_alerts.get("alerts", {}).items():
            # Skip if no alerts
            if not alerts:
                continue
            
            # Get the first (most severe) alert
            alert = alerts[0]
            
            # Get recommended actions from auto manager for this metric
            recommended_actions = self.auto_manager.actions.get(metric_name, [])
            
            # Calculate days until threshold is exceeded
            days_until = alert.get("days_until", 0)
            
            # Determine urgency based on days until
            if days_until <= 1:
                urgency = "immediate"
            elif days_until <= 7:
                urgency = "this_week"
            elif days_until <= 30:
                urgency = "this_month"
            else:
                urgency = "future"
            
            # Add recommendation
            recommendation = {
                "metric": metric_name,
                "severity": alert["severity"],
                "urgency": urgency,
                "days_until": days_until,
                "message": alert["message"],
                "recommended_actions": recommended_actions
            }
            
            results["recommendations"].append(recommendation)
            
            # Add proactive actions for this metric
            results["proactive_actions"][metric_name] = recommended_actions
        
        # Sort recommendations by urgency and severity
        urgency_order = {
            "immediate": 0,
            "this_week": 1,
            "this_month": 2,
            "future": 3
        }
        
        severity_order = {
            "error": 0,
            "warning": 1
        }
        
        results["recommendations"].sort(
            key=lambda x: (
                urgency_order.get(x["urgency"], 99),
                severity_order.get(x["severity"], 99)
            )
        )
        
        # Generate a comprehensive recommendation summary
        if results["recommendations"]:
            summary = []
            
            # Group by urgency
            urgency_groups = defaultdict(list)
            for rec in results["recommendations"]:
                urgency_groups[rec["urgency"]].append(rec)
            
            # Generate summary for each urgency group
            for urgency, recs in sorted(urgency_groups.items(), key=lambda x: urgency_order.get(x[0], 99)):
                # Format urgency for display
                urgency_display = urgency.replace("_", " ").title()
                
                # Count error and warning severities
                error_count = sum(1 for r in recs if r["severity"] == "error")
                warning_count = sum(1 for r in recs if r["severity"] == "warning")
                
                # Generate urgency group summary
                if error_count > 0 and warning_count > 0:
                    summary.append(
                        f"{urgency_display}: {error_count} critical and {warning_count} warning issues predicted"
                    )
                elif error_count > 0:
                    summary.append(f"{urgency_display}: {error_count} critical issues predicted")
                elif warning_count > 0:
                    summary.append(f"{urgency_display}: {warning_count} warning issues predicted")
            
            # Add summary to results
            results["summary"] = summary
        
        return results
    
    def analyze_database_health_forecast(
        self,
        horizon: str = "medium_term",
        specific_metrics: Optional[List[str]] = None,
        generate_visualizations: bool = True,
        output_format: str = "base64"
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of database health forecast with recommendations.
        
        Args:
            horizon: Forecast horizon ("short_term", "medium_term", or "long_term")
            specific_metrics: List of specific metrics to forecast (None for all)
            generate_visualizations: Whether to generate visualizations
            output_format: Format for visualization output
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        # Initialize results
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "success",
            "forecasts": {},
            "threshold_alerts": {},
            "recommendations": {},
            "visualizations": {},
            "summary": {}
        }
        
        # Generate forecasts
        forecast_results = self.forecast_database_metrics(horizon, specific_metrics)
        results["forecasts"] = forecast_results
        
        # Only continue if forecasting was successful
        if forecast_results.get("status") == "error":
            results["status"] = "error"
            results["message"] = forecast_results.get("message", "Forecasting failed")
            return results
        
        # Check predicted thresholds
        threshold_alerts = self.check_predicted_thresholds(forecast_results)
        results["threshold_alerts"] = threshold_alerts
        
        # Recommend proactive actions
        recommendations = self.recommend_proactive_actions(forecast_results, threshold_alerts)
        results["recommendations"] = recommendations
        
        # Generate visualizations if requested
        if generate_visualizations:
            visualizations = self.generate_forecast_visualizations(forecast_results, output_format)
            results["visualizations"] = visualizations
        
        # Generate summary
        summary = {
            "total_metrics_analyzed": len(forecast_results.get("forecasts", {})),
            "metrics_with_alerts": len(threshold_alerts.get("alerts", {})),
            "total_recommendations": len(recommendations.get("recommendations", [])),
            "forecast_horizon": horizon,
            "forecast_horizon_days": forecast_results.get("horizon_days", 0)
        }
        
        # Add forecast summary
        forecast_summary = []
        for metric_name, forecast in forecast_results.get("forecasts", {}).items():
            if "trend_analysis" in forecast:
                trend = forecast["trend_analysis"]
                direction = trend.get("direction", "stable")
                magnitude = trend.get("magnitude", "stable")
                
                if direction != "stable":
                    trend_description = f"{metric_name} shows {magnitude} {direction} trend"
                    forecast_summary.append(trend_description)
        
        summary["forecast_trends"] = forecast_summary
        
        # Add alert summary from recommendations
        if "summary" in recommendations:
            summary["alert_summary"] = recommendations["summary"]
        
        results["summary"] = summary
        
        # Set overall status based on threshold alerts
        if threshold_alerts.get("status") in ["error", "warning"]:
            results["status"] = threshold_alerts["status"]
        
        return results
    
    def _generate_forecasts(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[datetime.datetime],
        horizon_days: int
    ) -> Dict[str, Any]:
        """
        Generate forecasts for a specific metric.
        
        Args:
            metric_name: Name of the metric
            values: Historical values
            timestamps: Corresponding timestamps
            horizon_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Initialize results
        results = {
            "metric": metric_name,
            "historical_values": values,
            "historical_dates": [ts.isoformat() for ts in timestamps],
            "forecasting_config": {
                "auto_model_selection": self.config.get("forecasting", {}).get("auto_model_selection", True),
                "use_ensemble": self.config.get("forecasting", {}).get("use_ensemble", True),
                "available_methods": self.config.get("forecasting", {}).get("forecast_methods", [])
            }
        }
        
        # Get forecasting configuration
        forecast_methods = self.config["forecasting"]["forecast_methods"]
        confidence_level = self.config["forecasting"]["confidence_level"]
        use_ensemble = self.config["forecasting"]["use_ensemble"]
        
        # Generate dates for forecast horizon
        last_date = timestamps[-1]
        forecast_dates = []
        for i in range(1, horizon_days + 1):
            forecast_date = last_date + datetime.timedelta(days=i)
            forecast_dates.append(forecast_date)
        
        results["forecast_dates"] = [date.isoformat() for date in forecast_dates]
        
        # Initialize method results
        method_results = {}
        
        # Generate forecasts using different methods
        if "arima" in forecast_methods and PANDAS_AVAILABLE and SCIPY_AVAILABLE:
            try:
                arima_forecast = self._forecast_arima(values, timestamps, forecast_dates, metric_name)
                method_results["arima"] = arima_forecast
            except Exception as e:
                logger.warning(f"Error in ARIMA forecasting for {metric_name}: {e}")
        
        if "exponential_smoothing" in forecast_methods and PANDAS_AVAILABLE:
            try:
                es_forecast = self._forecast_exponential_smoothing(values, timestamps, forecast_dates, metric_name)
                method_results["exponential_smoothing"] = es_forecast
            except Exception as e:
                logger.warning(f"Error in exponential smoothing for {metric_name}: {e}")
        
        if "linear_regression" in forecast_methods and SCIPY_AVAILABLE:
            try:
                lr_forecast = self._forecast_linear_regression(values, timestamps, forecast_dates, metric_name)
                method_results["linear_regression"] = lr_forecast
            except Exception as e:
                logger.warning(f"Error in linear regression for {metric_name}: {e}")
        
        # Determine forecast selection strategy
        auto_model_selection = self.config.get("forecasting", {}).get("auto_model_selection", True)
        
        # If automatic model selection is enabled, prioritize it
        if auto_model_selection and len(method_results) > 1:
            # Try to select the best model based on validation
            self._use_best_individual_forecast(results, method_results, values)
            
            # If model selection was successful, we're done
            if results.get("model_selection") == "auto":
                return results
                
        # If auto-selection is disabled or failed, and ensemble is enabled, use ensemble
        if use_ensemble and len(method_results) > 1:
            try:
                ensemble_forecast = self._create_ensemble_forecast(method_results, forecast_dates)
                
                # Use ensemble forecast as the primary forecast
                for key, value in ensemble_forecast.items():
                    results[key] = value
                
                # Store individual method forecasts
                results["method_forecasts"] = method_results
                results["primary_method"] = "ensemble"
                results["model_selection"] = "ensemble"
            except Exception as e:
                logger.warning(f"Error creating ensemble forecast for {metric_name}: {e}")
                # Fall back to best individual method if not already done
                if "model_selection" not in results:
                    self._use_best_individual_forecast(results, method_results, values)
        else:
            # Only use the best individual forecast if not already done
            if "model_selection" not in results:
                self._use_best_individual_forecast(results, method_results, values)
        
        # Analyze trend in forecast
        try:
            trend_analysis = self._analyze_forecast_trend(
                results["forecast_values"], forecast_dates
            )
            results["trend_analysis"] = trend_analysis
        except Exception as e:
            logger.warning(f"Error analyzing trend for {metric_name}: {e}")
        
        return results
    
    def _use_best_individual_forecast(
        self,
        results: Dict[str, Any],
        method_results: Dict[str, Dict[str, Any]],
        historical_values: List[float]
    ) -> None:
        """
        Select the best individual forecast method based on error metrics.
        
        Args:
            results: Results dictionary to update
            method_results: Dictionary of forecasts from different methods
            historical_values: Historical values for testing accuracy
        """
        if not method_results:
            # No forecasts available
            results["forecast_values"] = []
            results["lower_bound"] = []
            results["upper_bound"] = []
            results["primary_method"] = "none"
            return
        
        # If we only have one method, use it
        if len(method_results) == 1:
            method, forecast = list(method_results.items())[0]
            for key, value in forecast.items():
                results[key] = value
            results["primary_method"] = method
            return
        
        # If automatic model selection is enabled in the config, use it
        if self.config.get("forecasting", {}).get("auto_model_selection", True):
            best_method, validation_metrics = self._select_best_forecast_method(method_results, historical_values)
            if best_method:
                forecast = method_results[best_method]
                for key, value in forecast.items():
                    results[key] = value
                results["primary_method"] = best_method
                results["model_selection"] = "auto"
                
                # Include validation metrics in the results
                if validation_metrics:
                    results["validation_metrics"] = validation_metrics
                    
                    # Record the best method's metrics
                    if best_method in validation_metrics:
                        best_method_metrics = validation_metrics[best_method]
                        results["best_method_metrics"] = {
                            "method": best_method,
                            "mape": best_method_metrics.get("mape"),
                            "mae": best_method_metrics.get("mae")
                        }
                
                return
        
        # Fallback to priority-based selection if auto-selection is disabled or fails
        # For now, we'll prioritize methods in this order:
        priority_order = ["arima", "exponential_smoothing", "linear_regression"]
        
        for method in priority_order:
            if method in method_results:
                forecast = method_results[method]
                for key, value in forecast.items():
                    results[key] = value
                results["primary_method"] = method
                results["model_selection"] = "priority"
                return
        
        # If we get here, use the first available method
        method, forecast = list(method_results.items())[0]
        for key, value in forecast.items():
            results[key] = value
        results["primary_method"] = method
        results["model_selection"] = "fallback"
    
    def _tune_arima_hyperparameters(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame = None,
        cv_folds: int = 3,
        metric_name: str = "generic_metric"
    ) -> Tuple[tuple, float]:
        """
        Tune ARIMA model hyperparameters using grid search or random search with early stopping and parameter persistence.
        
        Args:
            train_data: Training data for ARIMA model
            validation_data: Validation data for model evaluation (if None, uses cross-validation)
            cv_folds: Number of folds for cross-validation
            metric_name: Name of the metric being forecasted (for parameter persistence)
            
        Returns:
            Tuple containing:
            - Optimal ARIMA order (p, d, q)
            - Validation MAPE for the optimal model
        """
        # Import necessary modules
        from statsmodels.tsa.arima.model import ARIMA
        import itertools
        import numpy as np
        
        # Get hyperparameter tuning configuration
        tuning_config = self.config["forecasting"]["hyperparameter_tuning"]
        search_method = tuning_config.get("search_method", "grid")
        max_iterations = tuning_config.get("max_iterations", 10)
        
        # Check if parameter persistence is enabled
        persistence_config = tuning_config.get("parameter_persistence", {})
        persistence_enabled = persistence_config.get("enabled", True)
        
        # Early stopping parameters - use from config if available, otherwise set defaults
        early_stopping_enabled = tuning_config.get("early_stopping", {}).get("enabled", True)
        early_stopping_min_iterations = tuning_config.get("early_stopping", {}).get("min_iterations", 3)
        early_stopping_patience = tuning_config.get("early_stopping", {}).get("patience", 3)
        early_stopping_tolerance = tuning_config.get("early_stopping", {}).get("tolerance", 0.05)  # 5% improvement threshold
        
        # Create data signature for parameter cache lookup
        data_signature = self._create_data_signature(train_data) if persistence_enabled else None
        
        # Check if we have saved parameters to reuse
        if persistence_enabled:
            saved_params = self._load_parameters(metric_name, "arima", data_signature)
            if saved_params and "parameters" in saved_params and not saved_params.get("needs_revalidation", False):
                # Parameters found and valid, use them
                saved_order = tuple(saved_params["parameters"])
                saved_mape = saved_params.get("performance_metric")
                logger.info(f"Using saved ARIMA parameters for {metric_name}: {saved_order}, MAPE: {saved_mape}")
                return saved_order, saved_mape
            elif saved_params and saved_params.get("needs_revalidation", False):
                # Parameters found but need revalidation
                logger.info(f"Saved ARIMA parameters for {metric_name} need revalidation, retuning...")
        
        # Get ARIMA parameter ranges
        p_values = tuning_config["arima_params"].get("p", [0, 1, 2])
        d_values = tuning_config["arima_params"].get("d", [0, 1])
        q_values = tuning_config["arima_params"].get("q", [0, 1, 2])
        
        # Create parameter grid
        if search_method == "grid":
            # Grid search: try all combinations
            param_combinations = list(itertools.product(p_values, d_values, q_values))
            # Limit to max_iterations if specified
            if max_iterations and len(param_combinations) > max_iterations:
                # Sample without replacement if possible
                if max_iterations <= len(param_combinations):
                    # Use random.sample instead of np.random.choice to avoid shape error
                    # This addresses the "a must be 1-dimensional" error
                    import random
                    param_combinations = random.sample(param_combinations, max_iterations)
                else:
                    # If max_iterations > number of combinations, just use all combinations
                    pass
        else:
            # Random search: randomly sample combinations
            param_combinations = []
            for _ in range(max_iterations):
                p = np.random.choice(p_values)
                d = np.random.choice(d_values)
                q = np.random.choice(q_values)
                param_combinations.append((p, d, q))
        
        # Initialize variables to track best model
        best_order = (1, 1, 1)  # Default as fallback
        best_mape = float('inf')
        best_model = None
        
        # Track all results for debugging and analysis
        results = []
        
        # Early stopping variables
        iterations_without_improvement = 0
        
        # If using validation data
        if validation_data is not None:
            # Try each parameter combination
            for idx, order in enumerate(param_combinations):
                try:
                    # Fit model with current order
                    model = ARIMA(train_data, order=order)
                    with suppress_warnings():
                        model_fit = model.fit()
                    
                    # Forecast for validation period
                    forecast_steps = len(validation_data)
                    forecast = model_fit.forecast(steps=forecast_steps)
                    
                    # Calculate MAPE on validation data
                    actual = validation_data.values.flatten()
                    predicted = forecast.values
                    
                    # Calculate error metrics
                    mape_values = []
                    for a, p in zip(actual, predicted):
                        if a != 0:
                            mape_values.append(abs((a - p) / a) * 100)
                    
                    # Calculate overall MAPE
                    mape = sum(mape_values) / len(mape_values) if mape_values else float('inf')
                    
                    # Record result
                    results.append({
                        "order": order,
                        "mape": mape
                    })
                    
                    # Update best model if this one is better
                    if mape < best_mape:
                        # Calculate improvement percentage
                        improvement = (best_mape - mape) / best_mape if best_mape != float('inf') else 1.0
                        
                        # Update best model
                        best_mape = mape
                        best_order = order
                        best_model = model_fit
                        
                        # Reset counter if significant improvement
                        if improvement > early_stopping_tolerance:
                            iterations_without_improvement = 0
                        else:
                            # Increment counter for small improvements
                            iterations_without_improvement += 1
                    else:
                        # No improvement, increment counter
                        iterations_without_improvement += 1
                    
                    # Check for early stopping conditions
                    if (early_stopping_enabled 
                        and idx >= early_stopping_min_iterations 
                        and iterations_without_improvement >= early_stopping_patience):
                        logger.info(f"Early stopping triggered after {idx+1} iterations without significant improvement")
                        break
                
                except Exception as e:
                    logger.warning(f"Error fitting ARIMA with order {order}: {e}")
                    continue
        
        # If using cross-validation or validation failed
        elif train_data is not None and len(train_data) >= cv_folds * 2:
            # Implement time series cross-validation
            # For time series, we use a rolling forecast approach
            for idx, order in enumerate(param_combinations):
                try:
                    # Initialize CV metrics
                    cv_mape_values = []
                    
                    # Time series cross-validation
                    data_length = len(train_data)
                    fold_size = data_length // cv_folds
                    
                    for i in range(cv_folds):
                        # Create training and validation splits
                        # For time series, validation always comes after training
                        if i < cv_folds - 1:
                            train_end = (i + 1) * fold_size
                            cv_train = train_data.iloc[:train_end]
                            cv_val = train_data.iloc[train_end:train_end + fold_size]
                        else:
                            # Last fold might be a different size
                            train_end = (i + 1) * fold_size
                            cv_train = train_data.iloc[:train_end]
                            cv_val = train_data.iloc[train_end:]
                        
                        # Skip fold if too small
                        if len(cv_train) < 5 or len(cv_val) < 2:
                            continue
                        
                        # Fit model with current order
                        model = ARIMA(cv_train, order=order)
                        with suppress_warnings():
                            model_fit = model.fit()
                        
                        # Forecast for validation period
                        forecast_steps = len(cv_val)
                        forecast = model_fit.forecast(steps=forecast_steps)
                        
                        # Calculate MAPE on validation data
                        actual = cv_val.values.flatten()
                        predicted = forecast.values
                        
                        # Calculate error metrics for this fold
                        fold_mape_values = []
                        for a, p in zip(actual, predicted):
                            if a != 0:
                                fold_mape_values.append(abs((a - p) / a) * 100)
                        
                        # Calculate fold MAPE
                        if fold_mape_values:
                            fold_mape = sum(fold_mape_values) / len(fold_mape_values)
                            cv_mape_values.append(fold_mape)
                    
                    # Calculate average MAPE across all folds
                    if cv_mape_values:
                        cv_mape = sum(cv_mape_values) / len(cv_mape_values)
                        
                        # Record result
                        results.append({
                            "order": order,
                            "mape": cv_mape,
                            "cv_mape_values": cv_mape_values
                        })
                        
                        # Update best model if this one is better
                        if cv_mape < best_mape:
                            # Calculate improvement percentage
                            improvement = (best_mape - cv_mape) / best_mape if best_mape != float('inf') else 1.0
                            
                            # Update best model
                            best_mape = cv_mape
                            best_order = order
                            
                            # Reset counter if significant improvement
                            if improvement > early_stopping_tolerance:
                                iterations_without_improvement = 0
                            else:
                                # Increment counter for small improvements
                                iterations_without_improvement += 1
                        else:
                            # No improvement, increment counter
                            iterations_without_improvement += 1
                        
                        # Check for early stopping conditions
                        if (early_stopping_enabled 
                            and idx >= early_stopping_min_iterations 
                            and iterations_without_improvement >= early_stopping_patience):
                            logger.info(f"Early stopping triggered after {idx+1} iterations with {iterations_without_improvement} iterations without significant improvement")
                            break
                
                except Exception as e:
                    logger.warning(f"Error in CV for ARIMA with order {order}: {e}")
                    continue
        
        # If we got this far but haven't found a best model, use default
        if best_mape == float('inf'):
            logger.warning("Hyperparameter tuning failed to find optimal ARIMA parameters. Using default (1,1,1).")
            best_order = (1, 1, 1)
            best_mape = None
        
        # Log results
        if best_mape is not None:
            logger.info(f"ARIMA hyperparameter tuning complete. Best order: {best_order}, MAPE: {best_mape:.2f}%")
        else:
            logger.info(f"ARIMA hyperparameter tuning complete. Best order: {best_order}, MAPE: N/A")
        
        # Save optimal parameters for future use
        if persistence_enabled and best_order != (1, 1, 1):
            self._save_parameters(
                metric_name=metric_name,
                model_type="arima",
                parameters=best_order,
                performance_metric=best_mape,
                data_signature=data_signature
            )
        
        return best_order, best_mape

    def _forecast_arima(
        self,
        values: List[float],
        timestamps: List[datetime.datetime],
        forecast_dates: List[datetime.datetime],
        metric_name: str = None
    ) -> Dict[str, Any]:
        """
        Generate forecast using ARIMA model.
        
        Args:
            values: Historical values
            timestamps: Corresponding timestamps
            forecast_dates: Dates to forecast
            metric_name: Optional name of the metric being forecasted (for parameter persistence)
            
        Returns:
            Dictionary with ARIMA forecast results
        """
        # If metric name is not provided, try to identify it from caller context (best effort)
        if metric_name is None:
            # We'll try to get it from _generate_forecasts method
            import inspect
            frame = inspect.currentframe().f_back
            if frame and 'metric_name' in frame.f_locals:
                metric_name = frame.f_locals['metric_name']
            else:
                # Default to "unknown_metric" if we can't identify it
                metric_name = "unknown_metric"
        
        # Create a pandas DataFrame with time series data
        df = pd.DataFrame({"value": values}, index=timestamps)
        
        # Convert index to DatetimeIndex with explicit frequency to avoid warnings
        # First, ensure the index is sorted
        df = df.sort_index()
        
        # Infer and set the frequency
        try:
            # Try to infer frequency from timestamps
            freq = pd.infer_freq(df.index)
            
            # If frequency couldn't be inferred, default to daily 'D'
            if freq is None:
                freq = 'D'
                
            # Set the frequency
            df.index = pd.DatetimeIndex(df.index).to_period(freq).to_timestamp(freq)
        except Exception as e:
            # If any errors occur during frequency inference, use daily frequency as fallback
            logger.debug(f"Error setting time series frequency: {e}. Using daily frequency.")
            freq = 'D'
            # Ensure index is a DatetimeIndex with daily frequency
            df.index = pd.DatetimeIndex(df.index).to_period('D').to_timestamp('D')
        
        # Get number of days to forecast
        horizon = len(forecast_dates)
        
        # Try to import statsmodels for ARIMA
        from statsmodels.tsa.arima.model import ARIMA
        
        # Check if hyperparameter tuning is enabled
        hyperparameter_tuning = self.config["forecasting"].get("hyperparameter_tuning", {}).get("enabled", False)
        
        if hyperparameter_tuning:
            try:
                # Determine validation size for hyperparameter tuning
                validation_size_percent = min(max(1, self.config["forecasting"].get("validation_size_percent", 20)), 50)
                validation_size = max(3, int(len(df) * validation_size_percent / 100))
                
                # Split data for hyperparameter tuning
                train_data = df.iloc[:-validation_size]
                val_data = df.iloc[-validation_size:]
                
                # Only proceed with tuning if we have enough data
                if len(train_data) >= self.config["forecasting"]["min_data_points"]:
                    # Tune hyperparameters
                    best_order, best_mape = self._tune_arima_hyperparameters(
                        train_data=train_data,
                        validation_data=val_data,
                        cv_folds=self.config["forecasting"]["hyperparameter_tuning"].get("cv_folds", 3),
                        metric_name=metric_name
                    )
                    
                    # Use the full dataset with the optimal parameters
                    model = ARIMA(df, order=best_order)
                    model_info = {
                        "type": "ARIMA",
                        "order": str(best_order),
                        "tuned": True,
                        "validation_mape": best_mape
                    }
                else:
                    # Not enough data for tuning, use default parameters
                    model = ARIMA(df, order=(1, 1, 1))
                    model_info = {
                        "type": "ARIMA",
                        "order": "(1,1,1)",
                        "tuned": False,
                        "reason": "insufficient_data"
                    }
            except Exception as e:
                # If tuning fails, fall back to default parameters
                logger.warning(f"ARIMA hyperparameter tuning failed: {e}. Using default parameters.")
                model = ARIMA(df, order=(1, 1, 1))
                model_info = {
                    "type": "ARIMA",
                    "order": "(1,1,1)",
                    "tuned": False,
                    "reason": "tuning_error"
                }
        else:
            # Use default parameters if tuning is disabled
            model = ARIMA(df, order=(1, 1, 1))
            model_info = {
                "type": "ARIMA",
                "order": "(1,1,1)",
                "tuned": False,
                "reason": "tuning_disabled"
            }
        
        # Fit the model with warnings suppressed for better user experience
        with suppress_warnings():
            model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=horizon)
        forecast_values = forecast.values.tolist()
        
        # Calculate prediction intervals
        from scipy import stats
        
        alpha = 1 - self.config["forecasting"]["confidence_level"]
        pred_intervals = model_fit.get_forecast(steps=horizon).conf_int(alpha=alpha)
        lower_bound = pred_intervals.iloc[:, 0].tolist()
        upper_bound = pred_intervals.iloc[:, 1].tolist()
        
        return {
            "forecast_values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model_info": model_info
        }
    
    def _tune_exponential_smoothing_hyperparameters(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame = None,
        cv_folds: int = 3,
        metric_name: str = "generic_metric"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune Exponential Smoothing model hyperparameters using grid search or random search with early stopping and parameter persistence.
        
        Args:
            train_data: Training data for Exponential Smoothing model
            validation_data: Validation data for model evaluation (if None, uses cross-validation)
            cv_folds: Number of folds for cross-validation
            metric_name: Name of the metric being forecasted (for parameter persistence)
            
        Returns:
            Tuple containing:
            - Dictionary of optimal parameters (trend, seasonal, seasonal_periods, damped_trend)
            - Validation MAPE for the optimal model
        """
        # Import necessary modules
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        import itertools
        import numpy as np
        
        # Get hyperparameter tuning configuration
        tuning_config = self.config["forecasting"]["hyperparameter_tuning"]
        search_method = tuning_config.get("search_method", "grid")
        max_iterations = tuning_config.get("max_iterations", 10)
        
        # Check if parameter persistence is enabled
        persistence_config = tuning_config.get("parameter_persistence", {})
        persistence_enabled = persistence_config.get("enabled", True)
        
        # Early stopping parameters - use from config if available, otherwise set defaults
        early_stopping_enabled = tuning_config.get("early_stopping", {}).get("enabled", True)
        early_stopping_min_iterations = tuning_config.get("early_stopping", {}).get("min_iterations", 3)
        early_stopping_patience = tuning_config.get("early_stopping", {}).get("patience", 3)
        early_stopping_tolerance = tuning_config.get("early_stopping", {}).get("tolerance", 0.05)  # 5% improvement threshold
        
        # Create data signature for parameter cache lookup
        data_signature = self._create_data_signature(train_data) if persistence_enabled else None
        
        # Check if we have saved parameters to reuse
        if persistence_enabled:
            saved_params = self._load_parameters(metric_name, "exponential_smoothing", data_signature)
            if saved_params and "parameters" in saved_params and not saved_params.get("needs_revalidation", False):
                # Parameters found and valid, use them
                saved_params_dict = saved_params["parameters"]
                saved_mape = saved_params.get("performance_metric")
                logger.info(f"Using saved Exponential Smoothing parameters for {metric_name}: {saved_params_dict}, MAPE: {saved_mape}")
                return saved_params_dict, saved_mape
            elif saved_params and saved_params.get("needs_revalidation", False):
                # Parameters found but need revalidation
                logger.info(f"Saved Exponential Smoothing parameters for {metric_name} need revalidation, retuning...")
        
        # Get Exponential Smoothing parameter ranges
        trends = tuning_config["exp_smoothing_params"].get("trend", [None, "add", "mul"])
        seasonals = tuning_config["exp_smoothing_params"].get("seasonal", [None, "add", "mul"])
        seasonal_periods = tuning_config["exp_smoothing_params"].get("seasonal_periods", [7, 14])
        damped_trends = tuning_config["exp_smoothing_params"].get("damped_trend", [True, False])
        
        # Create parameter grid
        params_list = []
        
        if search_method == "grid":
            # Create all valid combinations for grid search
            for trend in trends:
                for seasonal in seasonals:
                    for damped_trend in damped_trends:
                        # Damped trend only applicable when trend is not None
                        if trend is None and damped_trend:
                            continue
                            
                        # Only use seasonal periods when seasonal is not None
                        if seasonal is not None:
                            for period in seasonal_periods:
                                params_list.append({
                                    "trend": trend,
                                    "seasonal": seasonal,
                                    "seasonal_periods": period,
                                    "damped_trend": damped_trend if trend is not None else False
                                })
                        else:
                            params_list.append({
                                "trend": trend,
                                "seasonal": None,
                                "seasonal_periods": None,
                                "damped_trend": damped_trend if trend is not None else False
                            })
            
            # Limit to max_iterations if specified
            if max_iterations and len(params_list) > max_iterations:
                # Use random.sample instead of np.random.choice to avoid shape error
                # This addresses the "a must be 1-dimensional" error
                import random
                params_list = random.sample(params_list, max_iterations)
                
        else:
            # Random search
            for _ in range(max_iterations):
                # Randomly choose trend
                trend = np.random.choice(trends + [trends[0]])  # Add extra weight to first option
                
                # Randomly choose seasonal
                seasonal = np.random.choice(seasonals + [seasonals[0]])  # Add extra weight to first option
                
                # Only use seasonal_periods when seasonal is not None
                period = np.random.choice(seasonal_periods) if seasonal is not None else None
                
                # Damped trend only applicable when trend is not None
                damped_trend = np.random.choice(damped_trends) if trend is not None else False
                
                params_list.append({
                    "trend": trend,
                    "seasonal": seasonal,
                    "seasonal_periods": period,
                    "damped_trend": damped_trend
                })
        
        # Initialize variables to track best model
        best_params = {
            "trend": "add",
            "seasonal": "add" if len(train_data) >= 14 else None,
            "seasonal_periods": 7 if len(train_data) >= 14 else None,
            "damped_trend": False
        }
        best_mape = float('inf')
        
        # Track all results for debugging and analysis
        results = []
        
        # Early stopping variables
        iterations_without_improvement = 0
        
        # If using validation data
        if validation_data is not None:
            # Try each parameter combination
            for idx, params in enumerate(params_list):
                try:
                    # Skip invalid combinations
                    if params["seasonal"] is not None and params["seasonal_periods"] is None:
                        continue
                        
                    # Skip if not enough data for seasonality
                    if params["seasonal"] is not None and len(train_data) < params["seasonal_periods"] * 2:
                        continue
                    
                    # Fit model with current parameters
                    model = ExponentialSmoothing(
                        train_data,
                        trend=params["trend"],
                        seasonal=params["seasonal"],
                        seasonal_periods=params["seasonal_periods"],
                        damped_trend=params["damped_trend"]
                    )
                    
                    # Use warning suppression during model fitting
                    with suppress_warnings():
                        model_fit = model.fit()
                    
                    # Forecast for validation period
                    forecast_steps = len(validation_data)
                    forecast = model_fit.forecast(forecast_steps)
                    
                    # Calculate MAPE on validation data
                    actual = validation_data.values.flatten()
                    predicted = forecast.values
                    
                    # Calculate error metrics
                    mape_values = []
                    for a, p in zip(actual, predicted):
                        if a != 0:
                            mape_values.append(abs((a - p) / a) * 100)
                    
                    # Calculate overall MAPE
                    mape = sum(mape_values) / len(mape_values) if mape_values else float('inf')
                    
                    # Record result
                    results.append({
                        "params": params,
                        "mape": mape
                    })
                    
                    # Update best model if this one is better
                    if mape < best_mape:
                        # Calculate improvement percentage
                        improvement = (best_mape - mape) / best_mape if best_mape != float('inf') else 1.0
                        
                        # Update best parameters
                        best_mape = mape
                        best_params = params
                        
                        # Reset counter if significant improvement
                        if improvement > early_stopping_tolerance:
                            iterations_without_improvement = 0
                        else:
                            # Increment counter for small improvements
                            iterations_without_improvement += 1
                    else:
                        # No improvement, increment counter
                        iterations_without_improvement += 1
                    
                    # Check for early stopping conditions
                    if (early_stopping_enabled 
                        and idx >= early_stopping_min_iterations 
                        and iterations_without_improvement >= early_stopping_patience):
                        logger.info(f"Early stopping triggered after {idx+1} iterations without significant improvement")
                        break
                
                except Exception as e:
                    logger.warning(f"Error fitting Exponential Smoothing with params {params}: {e}")
                    continue
        
        # If using cross-validation or validation failed
        elif train_data is not None and len(train_data) >= cv_folds * 2:
            # Implement time series cross-validation
            for idx, params in enumerate(params_list):
                try:
                    # Skip invalid combinations
                    if params["seasonal"] is not None and params["seasonal_periods"] is None:
                        continue
                        
                    # Skip if not enough data for seasonality
                    if params["seasonal"] is not None and len(train_data) < params["seasonal_periods"] * 2:
                        continue
                    
                    # Initialize CV metrics
                    cv_mape_values = []
                    
                    # Time series cross-validation
                    data_length = len(train_data)
                    fold_size = data_length // cv_folds
                    
                    for i in range(cv_folds):
                        # Create training and validation splits
                        if i < cv_folds - 1:
                            train_end = (i + 1) * fold_size
                            cv_train = train_data.iloc[:train_end]
                            cv_val = train_data.iloc[train_end:train_end + fold_size]
                        else:
                            # Last fold might be a different size
                            train_end = (i + 1) * fold_size
                            cv_train = train_data.iloc[:train_end]
                            cv_val = train_data.iloc[train_end:]
                        
                        # Skip fold if too small
                        if len(cv_train) < 5 or len(cv_val) < 2:
                            continue
                            
                        # Skip if not enough data for seasonality in this fold
                        if params["seasonal"] is not None and len(cv_train) < params["seasonal_periods"] * 2:
                            continue
                        
                        # Fit model with current parameters
                        model = ExponentialSmoothing(
                            cv_train,
                            trend=params["trend"],
                            seasonal=params["seasonal"],
                            seasonal_periods=params["seasonal_periods"],
                            damped_trend=params["damped_trend"]
                        )
                        
                        # Use warning suppression during model fitting
                        with suppress_warnings():
                            model_fit = model.fit()
                        
                        # Forecast for validation period
                        forecast_steps = len(cv_val)
                        forecast = model_fit.forecast(forecast_steps)
                        
                        # Calculate MAPE on validation data
                        actual = cv_val.values.flatten()
                        predicted = forecast.values
                        
                        # Calculate error metrics for this fold
                        fold_mape_values = []
                        for a, p in zip(actual, predicted):
                            if a != 0:
                                fold_mape_values.append(abs((a - p) / a) * 100)
                        
                        # Calculate fold MAPE
                        if fold_mape_values:
                            fold_mape = sum(fold_mape_values) / len(fold_mape_values)
                            cv_mape_values.append(fold_mape)
                    
                    # Calculate average MAPE across all folds
                    if cv_mape_values:
                        cv_mape = sum(cv_mape_values) / len(cv_mape_values)
                        
                        # Record result
                        results.append({
                            "params": params,
                            "mape": cv_mape,
                            "cv_mape_values": cv_mape_values
                        })
                        
                        # Update best model if this one is better
                        if cv_mape < best_mape:
                            # Calculate improvement percentage
                            improvement = (best_mape - cv_mape) / best_mape if best_mape != float('inf') else 1.0
                            
                            # Update best parameters
                            best_mape = cv_mape
                            best_params = params
                            
                            # Reset counter if significant improvement
                            if improvement > early_stopping_tolerance:
                                iterations_without_improvement = 0
                            else:
                                # Increment counter for small improvements
                                iterations_without_improvement += 1
                        else:
                            # No improvement, increment counter
                            iterations_without_improvement += 1
                        
                        # Check for early stopping conditions
                        if (early_stopping_enabled 
                            and idx >= early_stopping_min_iterations 
                            and iterations_without_improvement >= early_stopping_patience):
                            logger.info(f"Early stopping triggered after {idx+1} iterations with {iterations_without_improvement} iterations without significant improvement")
                            break
                
                except Exception as e:
                    logger.warning(f"Error in CV for Exponential Smoothing with params {params}: {e}")
                    continue
        
        # If we got this far but haven't found a best model, use default
        if best_mape == float('inf'):
            logger.warning("Hyperparameter tuning failed to find optimal Exponential Smoothing parameters. Using defaults.")
            best_params = {
                "trend": "add",
                "seasonal": "add" if len(train_data) >= 14 else None,
                "seasonal_periods": 7 if len(train_data) >= 14 else None,
                "damped_trend": False
            }
            best_mape = None
        
        # Log results
        param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
        if best_mape is not None:
            logger.info(f"Exponential Smoothing hyperparameter tuning complete. Best params: {param_str}, MAPE: {best_mape:.2f}%")
        else:
            logger.info(f"Exponential Smoothing hyperparameter tuning complete. Best params: {param_str}, MAPE: N/A")
        
        # Save optimal parameters for future use
        if persistence_enabled and best_params is not None:
            # Don't save default parameters (which are used as a fallback when tuning fails)
            is_default = (
                best_params.get("trend") == "add" and 
                best_params.get("seasonal") == "add" and
                best_params.get("seasonal_periods") == 7 and
                best_params.get("damped_trend") == False
            )
            
            if not is_default:
                self._save_parameters(
                    metric_name=metric_name,
                    model_type="exponential_smoothing",
                    parameters=best_params,
                    performance_metric=best_mape,
                    data_signature=data_signature
                )
        
        return best_params, best_mape

    def _forecast_exponential_smoothing(
        self,
        values: List[float],
        timestamps: List[datetime.datetime],
        forecast_dates: List[datetime.datetime],
        metric_name: str = None
    ) -> Dict[str, Any]:
        """
        Generate forecast using Exponential Smoothing.
        
        Args:
            values: Historical values
            timestamps: Corresponding timestamps
            forecast_dates: Dates to forecast
            metric_name: Optional name of the metric being forecasted (for parameter persistence)
            
        Returns:
            Dictionary with Exponential Smoothing forecast results
        """
        # Create a pandas DataFrame with time series data
        df = pd.DataFrame({"value": values}, index=timestamps)
        
        # Convert index to DatetimeIndex with explicit frequency to avoid warnings
        # First, ensure the index is sorted
        df = df.sort_index()
        
        # Infer and set the frequency
        try:
            # Try to infer frequency from timestamps
            freq = pd.infer_freq(df.index)
            
            # If frequency couldn't be inferred, default to daily 'D'
            if freq is None:
                freq = 'D'
                
            # Set the frequency
            df.index = pd.DatetimeIndex(df.index).to_period(freq).to_timestamp(freq)
        except Exception as e:
            # If any errors occur during frequency inference, use daily frequency as fallback
            logger.debug(f"Error setting time series frequency: {e}. Using daily frequency.")
            freq = 'D'
            # Ensure index is a DatetimeIndex with daily frequency
            df.index = pd.DatetimeIndex(df.index).to_period('D').to_timestamp('D')
            
        # Get number of days to forecast
        horizon = len(forecast_dates)
        
        # Try to import statsmodels for Exponential Smoothing
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Check if hyperparameter tuning is enabled
        hyperparameter_tuning = self.config["forecasting"].get("hyperparameter_tuning", {}).get("enabled", False)
        
        # Default parameters
        model_params = {
            "trend": "add",
            "seasonal": "add" if len(values) >= 14 else None,
            "seasonal_periods": 7 if len(values) >= 14 else None,
            "damped_trend": False
        }
        
        # Model info will track whether tuning was used and results
        model_info = {
            "type": "ExponentialSmoothing",
            "trend": model_params["trend"],
            "seasonal": model_params["seasonal"],
            "seasonal_periods": model_params["seasonal_periods"],
            "damped_trend": model_params["damped_trend"],
            "tuned": False,
            "reason": "default_settings"
        }
        
        if hyperparameter_tuning:
            try:
                # Determine validation size for hyperparameter tuning
                validation_size_percent = min(max(1, self.config["forecasting"].get("validation_size_percent", 20)), 50)
                validation_size = max(3, int(len(df) * validation_size_percent / 100))
                
                # Split data for hyperparameter tuning
                train_data = df.iloc[:-validation_size]
                val_data = df.iloc[-validation_size:]
                
                # Only proceed with tuning if we have enough data
                if len(train_data) >= self.config["forecasting"]["min_data_points"]:
                    # If metric name is not provided, try to identify it from caller context (best effort)
                    if metric_name is None:
                        # We'll try to get it from _generate_forecasts method
                        import inspect
                        frame = inspect.currentframe().f_back
                        if frame and 'metric_name' in frame.f_locals:
                            metric_name = frame.f_locals['metric_name']
                        else:
                            # Default to "unknown_metric" if we can't identify it
                            metric_name = "unknown_metric"
                    
                    # Tune hyperparameters
                    best_params, best_mape = self._tune_exponential_smoothing_hyperparameters(
                        train_data=train_data,
                        validation_data=val_data,
                        cv_folds=self.config["forecasting"]["hyperparameter_tuning"].get("cv_folds", 3),
                        metric_name=metric_name
                    )
                    
                    # Use the tuned parameters
                    model_params = best_params
                    
                    # Update model info
                    model_info = {
                        "type": "ExponentialSmoothing",
                        "trend": model_params["trend"],
                        "seasonal": model_params["seasonal"],
                        "seasonal_periods": model_params["seasonal_periods"],
                        "damped_trend": model_params["damped_trend"],
                        "tuned": True,
                        "validation_mape": best_mape
                    }
                else:
                    # Not enough data for tuning
                    model_info["reason"] = "insufficient_data"
            except Exception as e:
                # If tuning fails, fall back to default parameters
                logger.warning(f"Exponential Smoothing hyperparameter tuning failed: {e}. Using default parameters.")
                model_info["reason"] = "tuning_error"
        else:
            # Using default parameters
            model_info["reason"] = "tuning_disabled"
        
        # Fit model with selected parameters
        model = ExponentialSmoothing(
            df,
            trend=model_params["trend"],
            seasonal=model_params["seasonal"],
            seasonal_periods=model_params["seasonal_periods"],
            damped_trend=model_params["damped_trend"]
        )
        # Fit the model with warnings suppressed for better user experience
        with suppress_warnings():
            model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(horizon)
        forecast_values = forecast.values.tolist()
        
        # Estimate prediction intervals based on the residuals
        residuals = model_fit.resid
        residual_std = np.std(residuals)
        
        # For simple confidence intervals, we use a multiple of the residual std
        from scipy import stats
        
        alpha = 1 - self.config["forecasting"]["confidence_level"]
        z = stats.norm.ppf(1 - alpha / 2)
        
        # Calculate prediction intervals
        lower_bound = [max(0, f - z * residual_std) for f in forecast_values]
        upper_bound = [f + z * residual_std for f in forecast_values]
        
        return {
            "forecast_values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model_info": model_info
        }
    
    def _tune_linear_regression_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        cv_folds: int = 3,
        metric_name: str = "generic_metric"
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune Linear Regression model hyperparameters with early stopping and parameter persistence.
        
        Args:
            X_train: Training features
            y_train: Training target values
            X_val: Validation features
            y_val: Validation target values
            cv_folds: Number of folds for cross-validation
            metric_name: Name of the metric being forecasted (for parameter persistence)
            
        Returns:
            Tuple containing:
            - Dictionary of optimal parameters
            - Validation MAPE for the optimal model
        """
        # Try to import scikit-learn for advanced linear regression
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import TimeSeriesSplit
            import numpy as np
            SKLEARN_AVAILABLE = True
        except ImportError:
            SKLEARN_AVAILABLE = False
            # Fall back to scipy.stats linregress if sklearn is not available
            from scipy import stats
        
        # Get hyperparameter tuning configuration
        tuning_config = self.config["forecasting"]["hyperparameter_tuning"]
        search_method = tuning_config.get("search_method", "grid")
        max_iterations = tuning_config.get("max_iterations", 10)
        
        # Check if parameter persistence is enabled
        persistence_config = tuning_config.get("parameter_persistence", {})
        persistence_enabled = persistence_config.get("enabled", True)
        
        # Create data signature for parameter cache lookup
        data_signature = None
        if persistence_enabled:
            # Create a signature from X_train and y_train
            combined_data = np.hstack([X_train, y_train.reshape(-1, 1)]) if len(y_train.shape) == 1 else np.hstack([X_train, y_train])
            data_signature = self._create_data_signature(combined_data)
            
        # Check if we have saved parameters to reuse
        if persistence_enabled:
            saved_params = self._load_parameters(metric_name, "linear_regression", data_signature)
            if saved_params and "parameters" in saved_params and not saved_params.get("needs_revalidation", False):
                # Parameters found and valid, use them
                saved_params_dict = saved_params["parameters"]
                saved_mape = saved_params.get("performance_metric")
                logger.info(f"Using saved Linear Regression parameters for {metric_name}: {saved_params_dict}, MAPE: {saved_mape}")
                return saved_params_dict, saved_mape
            elif saved_params and saved_params.get("needs_revalidation", False):
                # Parameters found but need revalidation
                logger.info(f"Saved Linear Regression parameters for {metric_name} need revalidation, retuning...")
        
        # Early stopping parameters - use from config if available, otherwise set defaults
        early_stopping_enabled = tuning_config.get("early_stopping", {}).get("enabled", True)
        early_stopping_min_iterations = tuning_config.get("early_stopping", {}).get("min_iterations", 2)  # Linear regression has fewer params
        early_stopping_patience = tuning_config.get("early_stopping", {}).get("patience", 2)  # Fewer combos for linear regression
        early_stopping_tolerance = tuning_config.get("early_stopping", {}).get("tolerance", 0.05)  # 5% improvement threshold
        
        # Get Linear Regression parameter ranges
        if SKLEARN_AVAILABLE:
            fit_intercept_values = tuning_config["linear_regression_params"].get("fit_intercept", [True, False])
            positive_values = tuning_config["linear_regression_params"].get("positive", [True, False])
            
            # Create parameter grid
            param_combinations = []
            
            if search_method == "grid":
                # Grid search: try all combinations
                for fit_intercept in fit_intercept_values:
                    for positive in positive_values:
                        param_combinations.append({
                            "fit_intercept": fit_intercept,
                            "positive": positive
                        })
            else:
                # Random search: randomly sample combinations
                import random
                for _ in range(max_iterations):
                    param_combinations.append({
                        "fit_intercept": random.choice(fit_intercept_values),
                        "positive": random.choice(positive_values)
                    })
            
            # Initialize variables to track best model
            best_params = {"fit_intercept": True, "positive": False}
            best_mape = float('inf')
            best_model = None
            
            # Track all results for debugging and analysis
            results = []
            
            # Early stopping variables
            iterations_without_improvement = 0
            
            # If using validation data
            if X_val is not None and y_val is not None:
                # Try each parameter combination
                for idx, params in enumerate(param_combinations):
                    try:
                        # Fit model with current parameters
                        model = LinearRegression(
                            fit_intercept=params["fit_intercept"],
                            positive=params["positive"]
                        )
                        model.fit(X_train.reshape(-1, 1), y_train)
                        
                        # Predict validation data
                        y_pred = model.predict(X_val.reshape(-1, 1))
                        
                        # Calculate MAPE
                        mape_values = []
                        for actual, predicted in zip(y_val, y_pred):
                            if actual != 0:
                                mape_values.append(abs((actual - predicted) / actual) * 100)
                        
                        # Calculate overall MAPE
                        mape = sum(mape_values) / len(mape_values) if mape_values else float('inf')
                        
                        # Record result
                        results.append({
                            "params": params,
                            "mape": mape
                        })
                        
                        # Update best model if this one is better
                        if mape < best_mape:
                            # Calculate improvement percentage
                            improvement = (best_mape - mape) / best_mape if best_mape != float('inf') else 1.0
                            
                            # Update best model
                            best_mape = mape
                            best_params = params
                            best_model = model
                            
                            # Reset counter if significant improvement
                            if improvement > early_stopping_tolerance:
                                iterations_without_improvement = 0
                            else:
                                # Increment counter for small improvements
                                iterations_without_improvement += 1
                        else:
                            # No improvement, increment counter
                            iterations_without_improvement += 1
                        
                        # Check for early stopping conditions
                        if (early_stopping_enabled 
                            and idx >= early_stopping_min_iterations 
                            and iterations_without_improvement >= early_stopping_patience):
                            logger.info(f"Early stopping triggered after {idx+1} iterations without significant improvement")
                            break
                    
                    except Exception as e:
                        logger.warning(f"Error fitting Linear Regression with params {params}: {e}")
                        continue
            
            # If using cross-validation or validation failed
            elif X_train is not None and y_train is not None and len(X_train) >= cv_folds * 2:
                # Implement time series cross-validation
                for idx, params in enumerate(param_combinations):
                    try:
                        # Initialize CV metrics
                        cv_mape_values = []
                        
                        # Time series cross-validation
                        tscv = TimeSeriesSplit(n_splits=cv_folds)
                        
                        # Reshape X_train for sklearn
                        X_reshaped = X_train.reshape(-1, 1)
                        
                        for train_idx, val_idx in tscv.split(X_reshaped):
                            # Create training and validation splits
                            cv_X_train, cv_X_val = X_reshaped[train_idx], X_reshaped[val_idx]
                            cv_y_train, cv_y_val = y_train[train_idx], y_train[val_idx]
                            
                            # Skip fold if too small
                            if len(cv_X_train) < 5 or len(cv_X_val) < 2:
                                continue
                            
                            # Fit model with current parameters
                            model = LinearRegression(
                                fit_intercept=params["fit_intercept"],
                                positive=params["positive"]
                            )
                            model.fit(cv_X_train, cv_y_train)
                            
                            # Predict validation data
                            cv_y_pred = model.predict(cv_X_val)
                            
                            # Calculate MAPE
                            fold_mape_values = []
                            for actual, predicted in zip(cv_y_val, cv_y_pred):
                                if actual != 0:
                                    fold_mape_values.append(abs((actual - predicted) / actual) * 100)
                            
                            # Calculate fold MAPE
                            if fold_mape_values:
                                fold_mape = sum(fold_mape_values) / len(fold_mape_values)
                                cv_mape_values.append(fold_mape)
                        
                        # Calculate average MAPE across all folds
                        if cv_mape_values:
                            cv_mape = sum(cv_mape_values) / len(cv_mape_values)
                            
                            # Record result
                            results.append({
                                "params": params,
                                "mape": cv_mape,
                                "cv_mape_values": cv_mape_values
                            })
                            
                            # Update best model if this one is better
                            if cv_mape < best_mape:
                                # Calculate improvement percentage
                                improvement = (best_mape - cv_mape) / best_mape if best_mape != float('inf') else 1.0
                                
                                # Update best model
                                best_mape = cv_mape
                                best_params = params
                                
                                # Reset counter if significant improvement
                                if improvement > early_stopping_tolerance:
                                    iterations_without_improvement = 0
                                else:
                                    # Increment counter for small improvements
                                    iterations_without_improvement += 1
                            else:
                                # No improvement, increment counter
                                iterations_without_improvement += 1
                            
                            # Check for early stopping conditions
                            if (early_stopping_enabled 
                                and idx >= early_stopping_min_iterations 
                                and iterations_without_improvement >= early_stopping_patience):
                                logger.info(f"Early stopping triggered after {idx+1} iterations with {iterations_without_improvement} iterations without significant improvement")
                                break
                    
                    except Exception as e:
                        logger.warning(f"Error in CV for Linear Regression with params {params}: {e}")
                        continue
            
            # If we got this far but haven't found a best model, use default
            if best_mape == float('inf'):
                logger.warning("Hyperparameter tuning failed to find optimal Linear Regression parameters. Using defaults.")
                best_params = {"fit_intercept": True, "positive": False}
                best_mape = None
        else:
            # If sklearn is not available, use default parameters
            best_params = {"fit_intercept": True, "positive": False}
            best_mape = None
            logger.warning("scikit-learn not available for advanced Linear Regression tuning. Using simple regression.")
        
        # Log results
        logger.info(f"Linear Regression hyperparameter tuning complete. Best params: {best_params}, MAPE: {best_mape if best_mape is not None else 'N/A'}")
        
        # Save optimal parameters for future use
        if persistence_enabled and best_params is not None:
            # Don't save default parameters (which are used as a fallback when tuning fails)
            is_default = (
                best_params.get("fit_intercept") == True and 
                best_params.get("positive") == False
            )
            
            if not is_default:
                self._save_parameters(
                    metric_name=metric_name,
                    model_type="linear_regression",
                    parameters=best_params,
                    performance_metric=best_mape,
                    data_signature=data_signature
                )
        
        return best_params, best_mape

    def _forecast_linear_regression(
        self,
        values: List[float],
        timestamps: List[datetime.datetime],
        forecast_dates: List[datetime.datetime],
        metric_name: str = None
    ) -> Dict[str, Any]:
        """
        Generate forecast using Linear Regression.
        
        Args:
            values: Historical values
            timestamps: Corresponding timestamps
            forecast_dates: Dates to forecast
            metric_name: Optional name of the metric being forecasted (for parameter persistence)
            
        Returns:
            Dictionary with Linear Regression forecast results
        """
        # Create a pandas DataFrame with time series data for consistency
        # Even though we don't directly use time series models here, this helps maintain consistency
        df = pd.DataFrame({"value": values}, index=timestamps)
        
        # Convert index to DatetimeIndex with explicit frequency to avoid warnings in case we use it later
        # First, ensure the index is sorted
        df = df.sort_index()
        
        # Infer and set the frequency
        try:
            # Try to infer frequency from timestamps
            freq = pd.infer_freq(df.index)
            
            # If frequency couldn't be inferred, default to daily 'D'
            if freq is None:
                freq = 'D'
                
            # Set the frequency
            df.index = pd.DatetimeIndex(df.index).to_period(freq).to_timestamp(freq)
        except Exception as e:
            # If any errors occur during frequency inference, use daily frequency as fallback
            logger.debug(f"Error setting time series frequency: {e}. Using daily frequency.")
            freq = 'D'
            # Ensure index is a DatetimeIndex with daily frequency
            df.index = pd.DatetimeIndex(df.index).to_period('D').to_timestamp('D')
        
        # Use the sorted timestamps from the dataframe
        timestamps = df.index.to_list()
        
        # Convert timestamps to numerical features (days since first timestamp)
        first_timestamp = timestamps[0]
        days_since_start = np.array([(ts - first_timestamp).total_seconds() / (24 * 3600) for ts in timestamps])
        
        # Convert forecast dates to days since start
        forecast_days = np.array([(date - first_timestamp).total_seconds() / (24 * 3600) for date in forecast_dates])
        
        # Check if hyperparameter tuning is enabled
        hyperparameter_tuning = self.config["forecasting"].get("hyperparameter_tuning", {}).get("enabled", False)
        
        # Initialize model info
        model_info = {
            "type": "LinearRegression",
            "tuned": False,
            "reason": "default_settings"
        }
        
        # Try to import scikit-learn for advanced linear regression
        try:
            from sklearn.linear_model import LinearRegression
            SKLEARN_AVAILABLE = True
        except ImportError:
            SKLEARN_AVAILABLE = False
            logger.warning("scikit-learn not available. Using scipy.stats.linregress for linear regression.")
        
        # If sklearn is available and hyperparameter tuning is enabled
        if SKLEARN_AVAILABLE and hyperparameter_tuning:
            try:
                # Determine validation size for hyperparameter tuning
                validation_size_percent = min(max(1, self.config["forecasting"].get("validation_size_percent", 20)), 50)
                validation_size = max(3, int(len(days_since_start) * validation_size_percent / 100))
                
                # Convert to numpy arrays
                X = days_since_start
                y = np.array(values)
                
                # Split data for hyperparameter tuning
                X_train = X[:-validation_size]
                y_train = y[:-validation_size]
                X_val = X[-validation_size:]
                y_val = y[-validation_size:]
                
                # Only proceed with tuning if we have enough data
                if len(X_train) >= self.config["forecasting"]["min_data_points"]:
                    # If metric name is not provided, try to identify it from caller context (best effort)
                    if metric_name is None:
                        # We'll try to get it from _generate_forecasts method
                        import inspect
                        frame = inspect.currentframe().f_back
                        if frame and 'metric_name' in frame.f_locals:
                            metric_name = frame.f_locals['metric_name']
                        else:
                            # Default to "unknown_metric" if we can't identify it
                            metric_name = "unknown_metric"
                    
                    # Tune hyperparameters
                    best_params, best_mape = self._tune_linear_regression_hyperparameters(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        cv_folds=self.config["forecasting"]["hyperparameter_tuning"].get("cv_folds", 3),
                        metric_name=metric_name
                    )
                    
                    # Use the full dataset with the optimal parameters
                    model = LinearRegression(
                        fit_intercept=best_params["fit_intercept"],
                        positive=best_params["positive"]
                    )
                    model.fit(X.reshape(-1, 1), y)
                    
                    # Generate forecast
                    forecast_values = model.predict(forecast_days.reshape(-1, 1)).tolist()
                    
                    # Store model coefficients
                    slope = model.coef_[0]
                    intercept = model.intercept_ if best_params["fit_intercept"] else 0
                    
                    # Update model info
                    model_info = {
                        "type": "LinearRegression",
                        "slope": slope,
                        "intercept": intercept,
                        "fit_intercept": best_params["fit_intercept"],
                        "positive": best_params["positive"],
                        "tuned": True,
                        "validation_mape": best_mape
                    }
                    
                    # Calculate prediction intervals using standard formulation
                    X_mean = np.mean(X)
                    fitted_values = intercept + slope * X
                    residuals = y - fitted_values
                    sum_squared_error = np.sum(residuals ** 2)
                    std_err_pred = np.sqrt(sum_squared_error / (len(X) - 2))
                    
                    # Calculate t value for prediction intervals
                    from scipy import stats
                    alpha = 1 - self.config["forecasting"]["confidence_level"]
                    t_value = stats.t.ppf(1 - alpha / 2, len(X) - 2)
                    
                    # Calculate prediction intervals for each forecast point
                    lower_bound = []
                    upper_bound = []
                    
                    for x in forecast_days:
                        # Formula for prediction interval
                        X_diff_squared = np.sum((X - X_mean) ** 2)
                        if X_diff_squared == 0:  # Avoid division by zero
                            margin = t_value * std_err_pred
                        else:
                            margin = t_value * std_err_pred * np.sqrt(1 + 1/len(X) + ((x - X_mean) ** 2) / X_diff_squared)
                        
                        pred_value = intercept + slope * x
                        lower_bound.append(max(0, pred_value - margin))
                        upper_bound.append(pred_value + margin)
                    
                else:
                    # Not enough data for tuning, use default approach
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(days_since_start, values)
                    
                    # Generate forecast
                    forecast_values = [intercept + slope * day for day in forecast_days]
                    
                    # Update model info
                    model_info = {
                        "type": "LinearRegression",
                        "slope": slope,
                        "intercept": intercept,
                        "r_squared": r_value ** 2,
                        "tuned": False,
                        "reason": "insufficient_data"
                    }
                    
                    # Calculate prediction intervals
                    X_mean = np.mean(days_since_start)
                    sum_squared_error = sum((intercept + slope * x - y) ** 2 for x, y in zip(days_since_start, values))
                    std_err_pred = np.sqrt(sum_squared_error / (len(values) - 2))
                    
                    # Calculate t value for prediction intervals
                    alpha = 1 - self.config["forecasting"]["confidence_level"]
                    t_value = stats.t.ppf(1 - alpha / 2, len(values) - 2)
                    
                    # Calculate prediction intervals for each forecast point
                    lower_bound = []
                    upper_bound = []
                    
                    for x in forecast_days:
                        # Formula for prediction interval
                        X_diff_squared = sum((xi - X_mean) ** 2 for xi in days_since_start)
                        if X_diff_squared == 0:  # Avoid division by zero
                            margin = t_value * std_err_pred
                        else:
                            margin = t_value * std_err_pred * np.sqrt(1 + 1/len(values) + ((x - X_mean) ** 2) / X_diff_squared)
                        
                        pred_value = intercept + slope * x
                        lower_bound.append(max(0, pred_value - margin))
                        upper_bound.append(pred_value + margin)
            
            except Exception as e:
                # If tuning fails, fall back to default approach
                logger.warning(f"Linear Regression hyperparameter tuning failed: {e}. Using default parameters.")
                
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(days_since_start, values)
                
                # Generate forecast
                forecast_values = [intercept + slope * day for day in forecast_days]
                
                # Update model info
                model_info = {
                    "type": "LinearRegression",
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_value ** 2,
                    "tuned": False,
                    "reason": "tuning_error"
                }
                
                # Calculate prediction intervals
                X_mean = np.mean(days_since_start)
                sum_squared_error = sum((intercept + slope * x - y) ** 2 for x, y in zip(days_since_start, values))
                std_err_pred = np.sqrt(sum_squared_error / (len(values) - 2))
                
                # Calculate t value for prediction intervals
                alpha = 1 - self.config["forecasting"]["confidence_level"]
                t_value = stats.t.ppf(1 - alpha / 2, len(values) - 2)
                
                # Calculate prediction intervals for each forecast point
                lower_bound = []
                upper_bound = []
                
                for x in forecast_days:
                    # Formula for prediction interval
                    X_diff_squared = sum((xi - X_mean) ** 2 for xi in days_since_start)
                    if X_diff_squared == 0:  # Avoid division by zero
                        margin = t_value * std_err_pred
                    else:
                        margin = t_value * std_err_pred * np.sqrt(1 + 1/len(values) + ((x - X_mean) ** 2) / X_diff_squared)
                    
                    pred_value = intercept + slope * x
                    lower_bound.append(max(0, pred_value - margin))
                    upper_bound.append(pred_value + margin)
        
        else:
            # Use default approach with scipy.stats.linregress
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(days_since_start, values)
            
            # Generate forecast
            forecast_values = [intercept + slope * day for day in forecast_days]
            
            # Update model info
            model_info = {
                "type": "LinearRegression",
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "tuned": False,
                "reason": "tuning_disabled_or_sklearn_unavailable"
            }
            
            # Calculate prediction intervals
            X_mean = np.mean(days_since_start)
            sum_squared_error = sum((intercept + slope * x - y) ** 2 for x, y in zip(days_since_start, values))
            std_err_pred = np.sqrt(sum_squared_error / (len(values) - 2))
            
            # Calculate t value for prediction intervals
            alpha = 1 - self.config["forecasting"]["confidence_level"]
            t_value = stats.t.ppf(1 - alpha / 2, len(values) - 2)
            
            # Calculate prediction intervals for each forecast point
            lower_bound = []
            upper_bound = []
            
            for x in forecast_days:
                # Formula for prediction interval
                X_diff_squared = sum((xi - X_mean) ** 2 for xi in days_since_start)
                if X_diff_squared == 0:  # Avoid division by zero
                    margin = t_value * std_err_pred
                else:
                    margin = t_value * std_err_pred * np.sqrt(1 + 1/len(values) + ((x - X_mean) ** 2) / X_diff_squared)
                
                pred_value = intercept + slope * x
                lower_bound.append(max(0, pred_value - margin))
                upper_bound.append(pred_value + margin)
        
        return {
            "forecast_values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model_info": model_info
        }
    
    def _create_ensemble_forecast(
        self,
        method_results: Dict[str, Dict[str, Any]],
        forecast_dates: List[datetime.datetime]
    ) -> Dict[str, Any]:
        """
        Create an ensemble forecast by combining multiple methods.
        
        Args:
            method_results: Dictionary of forecasts from different methods
            forecast_dates: Dates for the forecast
            
        Returns:
            Dictionary with ensemble forecast results
        """
        # Simple ensemble approach: average the forecasts
        
        # Get all available forecasts
        all_forecasts = []
        all_lower_bounds = []
        all_upper_bounds = []
        
        for method, forecast in method_results.items():
            if "forecast_values" in forecast:
                all_forecasts.append(forecast["forecast_values"])
            
            if "lower_bound" in forecast:
                all_lower_bounds.append(forecast["lower_bound"])
            
            if "upper_bound" in forecast:
                all_upper_bounds.append(forecast["upper_bound"])
        
        # Ensure all forecasts are the same length
        min_length = min(len(f) for f in all_forecasts)
        all_forecasts = [f[:min_length] for f in all_forecasts]
        
        # Calculate average forecast
        avg_forecast = []
        for i in range(min_length):
            avg_forecast.append(sum(f[i] for f in all_forecasts) / len(all_forecasts))
        
        # Calculate ensemble bounds (conservative approach: use widest intervals)
        lower_bound = []
        upper_bound = []
        
        if all_lower_bounds and all_upper_bounds:
            # Ensure all bounds are the same length
            min_bound_length = min(len(b) for b in all_lower_bounds + all_upper_bounds)
            all_lower_bounds = [b[:min_bound_length] for b in all_lower_bounds]
            all_upper_bounds = [b[:min_bound_length] for b in all_upper_bounds]
            
            for i in range(min_bound_length):
                lower_bound.append(min(b[i] for b in all_lower_bounds))
                upper_bound.append(max(b[i] for b in all_upper_bounds))
        
        # Create ensemble forecast result
        ensemble_forecast = {
            "forecast_values": avg_forecast,
            "model_info": {
                "type": "Ensemble",
                "methods": list(method_results.keys())
            }
        }
        
        # Add bounds if available
        if lower_bound and upper_bound:
            ensemble_forecast["lower_bound"] = lower_bound
            ensemble_forecast["upper_bound"] = upper_bound
        
        return ensemble_forecast
    
    def _select_best_forecast_method(
        self,
        method_results: Dict[str, Dict[str, Any]],
        historical_values: List[float]
    ) -> Tuple[Optional[str], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Select the best forecasting method based on validation metrics.
        
        This function performs validation on historical data to evaluate different
        forecasting methods and select the one with the best performance.
        
        Args:
            method_results: Dictionary of forecasts from different methods
            historical_values: Historical values for testing accuracy
            
        Returns:
            A tuple containing:
            - Name of the best forecasting method, or None if selection fails
            - Dictionary of validation metrics for each method, or None if validation fails
        """
        try:
            # We need enough historical data for validation
            min_data_points = self.config.get("forecasting", {}).get("min_data_points", 10)
            if len(historical_values) < min_data_points:
                logger.warning(f"Insufficient data points for model validation: {len(historical_values)} < {min_data_points}")
                return None, None
            
            # Determine how much historical data to use for validation
            validation_percent = self.config.get("forecasting", {}).get("validation_size_percent", 25)
            # Ensure validation percent is between 1 and 50
            validation_percent = max(1, min(50, validation_percent))
            
            validation_size = max(3, int(len(historical_values) * validation_percent / 100))
            # Ensure we don't use too many points for validation
            validation_size = min(validation_size, len(historical_values) // 3)
            
            if validation_size < 3:
                logger.warning("Validation size too small for reliable model selection")
                return None, None
            
            # Training data will be used to make predictions for the validation period
            training_data = historical_values[:-validation_size]
            validation_data = historical_values[-validation_size:]
            
            # Dictionary to store validation metrics for each method
            validation_metrics = {}
            
            # Calculate validation metrics for each method
            for method_name, method_forecast in method_results.items():
                # Skip methods without forecast values (should never happen but just in case)
                if "forecast_values" not in method_forecast:
                    continue
                
                # The predicted values we'll compare to validation data
                predicted_values = method_forecast["forecast_values"][:validation_size]
                
                # If we don't have enough predicted values, skip this method
                if len(predicted_values) < validation_size:
                    continue
                
                # Calculate error metrics
                errors = [abs(predicted - actual) for predicted, actual in zip(predicted_values, validation_data)]
                mae = sum(errors) / len(errors)  # Mean Absolute Error
                
                # Calculate relative metrics
                mape_values = []
                for actual, predicted in zip(validation_data, predicted_values):
                    if actual != 0:
                        mape_values.append(abs((actual - predicted) / actual) * 100)
                
                mape = sum(mape_values) / len(mape_values) if mape_values else float('inf')  # Mean Absolute Percentage Error
                
                # Store metrics
                validation_metrics[method_name] = {
                    "mae": mae,
                    "mape": mape,
                    "errors": errors
                }
            
            # If we couldn't calculate metrics for any method, return None
            if not validation_metrics:
                logger.warning("Could not calculate validation metrics for any forecasting method")
                return None, None
            
            # Select the best method based on MAPE (lower is better)
            best_method = min(validation_metrics.items(), key=lambda x: x[1]["mape"])[0]
            
            logger.info(f"Automated model selection chose '{best_method}' with MAPE: {validation_metrics[best_method]['mape']:.2f}%")
            
            return best_method, validation_metrics
            
        except Exception as e:
            logger.warning(f"Error in automated model selection: {e}")
            return None, None

    def clean_parameter_storage(self, older_than_days: int = None) -> Dict[str, Any]:
        """
        Clean the parameter storage by removing saved parameters.
        
        Args:
            older_than_days: If provided, only remove parameters older than this many days
            
        Returns:
            Dictionary with cleaning results
        """
        import os
        import glob
        import datetime
        
        # Get storage path
        storage_path = self._get_parameter_storage_path()
        
        # Find all parameter files
        json_files = glob.glob(str(storage_path / "*.json"))
        pickle_files = glob.glob(str(storage_path / "*.pkl"))
        all_files = json_files + pickle_files
        
        # Initialize counters
        total_files = len(all_files)
        removed_files = 0
        
        # If we need to filter by age
        if older_than_days is not None:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
            filtered_files = []
            
            for file_path in all_files:
                # Get file modification time
                file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Check if older than cutoff
                if file_mtime < cutoff_time:
                    filtered_files.append(file_path)
            
            all_files = filtered_files
        
        # Remove files
        for file_path in all_files:
            try:
                os.remove(file_path)
                removed_files += 1
            except Exception as e:
                logger.warning(f"Error removing parameter file {file_path}: {e}")
        
        # Clear memory cache
        self.hyperparameter_cache = {}
        
        # Log result
        if older_than_days is not None:
            logger.info(f"Cleaned parameter storage: removed {removed_files} files older than {older_than_days} days")
        else:
            logger.info(f"Cleaned parameter storage: removed {removed_files} files")
            
        return {
            "total_files": total_files,
            "removed_files": removed_files,
            "storage_path": str(storage_path)
        }
    
    def test_parameter_persistence(self, metric_name: str = "test_metric") -> Dict[str, Any]:
        """
        Test the parameter persistence functionality to ensure it's working correctly.
        
        Args:
            metric_name: Name of the metric to use for the test
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing parameter persistence for metric: {metric_name}")
        
        # Create some synthetic data
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Generate synthetic timestamps and values for a metric
        start_date = datetime.now() - timedelta(days=100)
        timestamps = [start_date + timedelta(days=i) for i in range(100)]
        # Create a simple time series with some trend and noise
        values = [10 + 0.1 * i + np.random.normal(0, 1) for i in range(100)]
        
        # Create dataframe for testing
        df = pd.DataFrame({"value": values}, index=timestamps)
        
        # Test ARIMA parameter persistence
        try:
            # First run should tune and save parameters
            logger.info("First ARIMA run (should tune and save parameters)")
            arima_order1, mape1 = self._tune_arima_hyperparameters(
                train_data=df, 
                metric_name=metric_name
            )
            
            # Second run should load parameters from storage
            logger.info("Second ARIMA run (should load parameters from storage)")
            arima_order2, mape2 = self._tune_arima_hyperparameters(
                train_data=df, 
                metric_name=metric_name
            )
            
            # Parameters should be the same
            arima_test_success = (arima_order1 == arima_order2)
            logger.info(f"ARIMA parameter persistence test: {'Success' if arima_test_success else 'Failed'}")
            
        except Exception as e:
            logger.error(f"Error testing ARIMA parameter persistence: {e}")
            arima_test_success = False
        
        # Test Exponential Smoothing parameter persistence
        try:
            # First run should tune and save parameters
            logger.info("First Exponential Smoothing run (should tune and save parameters)")
            es_params1, es_mape1 = self._tune_exponential_smoothing_hyperparameters(
                train_data=df, 
                metric_name=f"{metric_name}_es"
            )
            
            # Second run should load parameters from storage
            logger.info("Second Exponential Smoothing run (should load parameters from storage)")
            es_params2, es_mape2 = self._tune_exponential_smoothing_hyperparameters(
                train_data=df, 
                metric_name=f"{metric_name}_es"
            )
            
            # Parameters should be the same
            es_test_success = (
                es_params1.get("trend") == es_params2.get("trend") and
                es_params1.get("seasonal") == es_params2.get("seasonal") and
                es_params1.get("seasonal_periods") == es_params2.get("seasonal_periods") and
                es_params1.get("damped_trend") == es_params2.get("damped_trend")
            )
            logger.info(f"Exponential Smoothing parameter persistence test: {'Success' if es_test_success else 'Failed'}")
            
        except Exception as e:
            logger.error(f"Error testing Exponential Smoothing parameter persistence: {e}")
            es_test_success = False
        
        # Test Linear Regression parameter persistence
        try:
            # Prepare data for linear regression
            X = np.array(range(100)).reshape(-1, 1)
            y = np.array(values)
            
            # First run should tune and save parameters
            logger.info("First Linear Regression run (should tune and save parameters)")
            lr_params1, lr_mape1 = self._tune_linear_regression_hyperparameters(
                X_train=X, 
                y_train=y,
                metric_name=f"{metric_name}_lr"
            )
            
            # Second run should load parameters from storage
            logger.info("Second Linear Regression run (should load parameters from storage)")
            lr_params2, lr_mape2 = self._tune_linear_regression_hyperparameters(
                X_train=X, 
                y_train=y,
                metric_name=f"{metric_name}_lr"
            )
            
            # Parameters should be the same
            lr_test_success = (
                lr_params1.get("fit_intercept") == lr_params2.get("fit_intercept") and
                lr_params1.get("positive") == lr_params2.get("positive")
            )
            logger.info(f"Linear Regression parameter persistence test: {'Success' if lr_test_success else 'Failed'}")
            
        except Exception as e:
            logger.error(f"Error testing Linear Regression parameter persistence: {e}")
            lr_test_success = False
        
        # Return test results
        return {
            "arima_test": arima_test_success,
            "exponential_smoothing_test": es_test_success,
            "linear_regression_test": lr_test_success,
            "overall_success": arima_test_success and es_test_success and lr_test_success,
            "arima_params": {"order1": arima_order1, "order2": arima_order2} if arima_test_success else None,
            "es_params": {"params1": es_params1, "params2": es_params2} if es_test_success else None,
            "lr_params": {"params1": lr_params1, "params2": lr_params2} if lr_test_success else None
        }
    
    def _analyze_forecast_trend(
        self,
        forecast_values: List[float],
        forecast_dates: List[datetime.datetime]
    ) -> Dict[str, Any]:
        """
        Analyze the trend in a forecast.
        
        Args:
            forecast_values: Forecasted values
            forecast_dates: Corresponding dates
            
        Returns:
            Dictionary with trend analysis results
        """
        # Check if we have enough data
        if len(forecast_values) < 2:
            return {
                "direction": "stable",
                "magnitude": "stable",
                "percent_change": 0.0
            }
        
        # Calculate percent change from start to end
        start_value = forecast_values[0]
        end_value = forecast_values[-1]
        
        if start_value == 0:
            # Avoid division by zero
            if end_value == 0:
                percent_change = 0.0
            else:
                percent_change = 100.0  # Arbitrary large value for increase from zero
        else:
            percent_change = ((end_value - start_value) / abs(start_value)) * 100
        
        # Determine trend direction
        if percent_change > 5:
            direction = "increasing"
        elif percent_change < -5:
            direction = "decreasing"
        else:
            direction = "stable"
        
        # Determine trend magnitude
        abs_change = abs(percent_change)
        if abs_change < 5:
            magnitude = "stable"
        elif abs_change < 20:
            magnitude = "moderate"
        elif abs_change < 50:
            magnitude = "substantial"
        else:
            magnitude = "dramatic"
        
        return {
            "direction": direction,
            "magnitude": magnitude,
            "percent_change": percent_change
        }
    
    def _detect_anomalies(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[datetime.datetime],
        forecast_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect anomalies in historical and forecasted values.
        
        Args:
            metric_name: Name of the metric
            values: Historical values
            timestamps: Corresponding timestamps
            forecast_result: Forecast results
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Get anomaly detection configuration
        z_score_threshold = self.config["anomaly_detection"]["z_score_threshold"]
        
        # Initialize results
        results = {
            "metric": metric_name,
            "anomaly_detection_method": "z_score",
            "z_score_threshold": z_score_threshold
        }
        
        # Detect anomalies in historical data
        if values and len(values) >= 3:
            # Calculate mean and standard deviation
            mean = np.mean(values)
            std_dev = np.std(values)
            
            # Check for anomalies using z-score
            anomaly_indices = []
            anomaly_z_scores = []
            
            if std_dev > 0:  # Avoid division by zero
                for i, value in enumerate(values):
                    z_score = abs((value - mean) / std_dev)
                    if z_score > z_score_threshold:
                        anomaly_indices.append(i)
                        anomaly_z_scores.append(float(z_score))
            
            if anomaly_indices:
                results["historical_anomalies"] = {
                    "indices": anomaly_indices,
                    "z_scores": anomaly_z_scores,
                    "timestamps": [timestamps[i].isoformat() for i in anomaly_indices]
                }
        
        # Detect anomalies in forecast
        forecast_values = forecast_result.get("forecast_values", [])
        if forecast_values and len(forecast_values) >= 1:
            # Use historical statistics for comparison
            mean = np.mean(values)
            std_dev = np.std(values)
            
            # Check for anomalies in forecast
            anomaly_indices = []
            anomaly_z_scores = []
            
            if std_dev > 0:  # Avoid division by zero
                for i, value in enumerate(forecast_values):
                    z_score = abs((value - mean) / std_dev)
                    if z_score > z_score_threshold:
                        anomaly_indices.append(i)
                        anomaly_z_scores.append(float(z_score))
            
            if anomaly_indices:
                results["forecast_anomalies"] = {
                    "indices": anomaly_indices,
                    "z_scores": anomaly_z_scores,
                    "forecast_dates": [forecast_result["forecast_dates"][i] for i in anomaly_indices]
                }
        
        return results