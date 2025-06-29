#!/usr/bin/env python
"""
API Anomaly Detection Module

This module provides anomaly detection capabilities for the API Distributed Testing Framework,
allowing automatic identification of performance issues and unexpected API behavior patterns.

Features:
1. Statistical anomaly detection for API performance metrics
2. Time-series based pattern detection
3. Configurable sensitivity and detection algorithms
4. Integration with the API monitoring dashboard
5. Alert generation for detected anomalies

Usage:
    Import this module into the API monitoring dashboard for automatic anomaly detection.
"""

import os
import sys
import time
import logging
import numpy as np
import smtplib
import ssl
import threading
import queue
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_anomaly_detection")

class AnomalyDetectionAlgorithm(Enum):
    """Supported anomaly detection algorithms."""
    ZSCORE = "zscore"
    MOVING_AVERAGE = "moving_average"
    IQR = "iqr"  # Interquartile Range
    PATTERN_DETECTION = "pattern_detection"  # Time series pattern detection
    SEASONALITY = "seasonality"  # Seasonality-aware detection
    ENSEMBLE = "ensemble"  # Combination of multiple algorithms

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    LATENCY_SPIKE = "latency_spike"
    THROUGHPUT_DROP = "throughput_drop"
    ERROR_RATE_INCREASE = "error_rate_increase"
    COST_SPIKE = "cost_spike"
    PATTERN_CHANGE = "pattern_change"
    SEASONAL_DEVIATION = "seasonal_deviation"
    TREND_BREAK = "trend_break"
    PERSISTENT_DEGRADATION = "persistent_degradation"
    OSCILLATION = "oscillation"

class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    """Supported notification channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    CALLBACK = "callback"

class AnomalyDetector:
    """
    Anomaly detector for API performance metrics.
    
    This class provides various algorithms for detecting anomalies in API
    performance data, including latency spikes, throughput drops, error rate
    increases, and unusual behavior patterns.
    """
    
    def __init__(
        self,
        algorithm: AnomalyDetectionAlgorithm = AnomalyDetectionAlgorithm.ENSEMBLE,
        sensitivity: float = 1.0,
        window_size: int = 20,
        baseline_days: int = 3,
        notification_enabled: bool = False,
        notification_manager = None
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            algorithm: The anomaly detection algorithm to use
            sensitivity: Sensitivity multiplier (1.0 = default, higher values = more sensitive)
            window_size: Window size for moving averages and pattern detection
            baseline_days: Number of days to use for establishing baseline metrics
            notification_enabled: Whether to enable notifications
            notification_manager: Optional notification manager instance
        """
        self.algorithm = algorithm
        self.sensitivity = sensitivity
        self.window_size = window_size
        self.baseline_days = baseline_days
        self.notification_enabled = notification_enabled
        self.notification_manager = notification_manager
        
        # Store anomaly history
        self.anomaly_history: List[Dict[str, Any]] = []
        
        # Baseline metrics by API and metric type
        self.baselines: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Pattern recognition state
        self.seasonality_patterns: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.trend_models: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Historical statistics for comparison
        self.historical_stats: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        
        # Try to import notification manager if needed but not provided
        if self.notification_enabled and not self.notification_manager:
            try:
                from api_notification_manager import NotificationManager
                self.notification_manager = NotificationManager()
                self.notification_manager.start()
                logger.info("Initialized notification manager")
            except ImportError:
                logger.warning("Could not import NotificationManager, notifications disabled")
                self.notification_enabled = False
        
        logger.info(f"Anomaly detector initialized with algorithm: {algorithm.value}, sensitivity: {sensitivity}, notifications: {notification_enabled}")
    
    def update_baseline(
        self, 
        api: str, 
        metric_type: str, 
        historical_data: List[Dict[str, Any]]
    ) -> None:
        """
        Update baseline statistics for a specific API and metric type.
        
        Args:
            api: The API name (e.g., 'openai', 'claude', 'groq')
            metric_type: The metric type (e.g., 'latency', 'throughput', 'reliability')
            historical_data: List of historical data points
        """
        if not historical_data:
            logger.warning(f"No historical data provided for {api} {metric_type}")
            return
        
        # Sort data by timestamp
        sorted_data = sorted(historical_data, key=lambda x: x["timestamp"])
        
        # Calculate cutoff time for baseline (using only recent data)
        now = time.time()
        baseline_cutoff = now - (self.baseline_days * 24 * 60 * 60)
        
        # Filter data for baseline calculation
        baseline_data = [
            point for point in sorted_data 
            if point["timestamp"] >= baseline_cutoff
        ]
        
        if not baseline_data:
            logger.warning(f"No recent data for baseline calculation for {api} {metric_type}")
            return
        
        # Extract values based on metric type
        value_key = self._get_value_key_for_metric(metric_type)
        if value_key is None:
            logger.warning(f"Unknown metric type: {metric_type}")
            return
        
        values = [point.get(value_key, 0) for point in baseline_data if value_key in point]
        
        if not values:
            logger.warning(f"No values found for {value_key} in {api} {metric_type}")
            return
        
        # Calculate baseline statistics
        baseline = {
            "mean": np.mean(values),
            "std_dev": np.std(values),
            "median": np.median(values),
            "min": np.min(values),
            "max": np.max(values),
            "q1": np.percentile(values, 25),
            "q3": np.percentile(values, 75),
            "iqr": np.percentile(values, 75) - np.percentile(values, 25),
            "last_update": now,
            "sample_size": len(values)
        }
        
        # Store baseline
        if api not in self.baselines:
            self.baselines[api] = {}
        
        self.baselines[api][metric_type] = baseline
        
        logger.debug(f"Updated baseline for {api} {metric_type}: mean={baseline['mean']:.4f}, std_dev={baseline['std_dev']:.4f}")
    
    def _get_value_key_for_metric(self, metric_type: str) -> Optional[str]:
        """
        Get the key for extracting values from data points based on metric type.
        
        Args:
            metric_type: The metric type
            
        Returns:
            The key for accessing values in data points
        """
        metric_keys = {
            "latency": "avg_latency",
            "throughput": "requests_per_second",
            "reliability": "success_rate",
            "cost": "cost_per_request"
        }
        
        return metric_keys.get(metric_type)
    
    def detect_anomalies(
        self, 
        api: str, 
        metric_type: str, 
        recent_data: List[Dict[str, Any]],
        algorithm: Optional[AnomalyDetectionAlgorithm] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in recent data for a specific API and metric type.
        
        Args:
            api: The API name
            metric_type: The metric type
            recent_data: List of recent data points
            algorithm: Override the default algorithm
            
        Returns:
            List of detected anomalies
        """
        if not recent_data:
            return []
        
        # Use specified algorithm or instance default
        algo = algorithm or self.algorithm
        
        # Sort data by timestamp
        sorted_data = sorted(recent_data, key=lambda x: x["timestamp"])
        
        # Get value key for metric type
        value_key = self._get_value_key_for_metric(metric_type)
        if value_key is None:
            logger.warning(f"Unknown metric type: {metric_type}")
            return []
        
        # Extract values
        timestamps = [point["timestamp"] for point in sorted_data]
        values = [point.get(value_key, 0) for point in sorted_data]
        
        if not values:
            return []
        
        # Get baseline for this API and metric
        baseline = self.baselines.get(api, {}).get(metric_type)
        if baseline is None:
            # If no baseline exists, update it first
            self.update_baseline(api, metric_type, sorted_data)
            baseline = self.baselines.get(api, {}).get(metric_type)
            
            # If still no baseline, return empty
            if baseline is None:
                return []
        
        # Detect anomalies based on algorithm
        anomalies = []
        
        if algo == AnomalyDetectionAlgorithm.ZSCORE:
            anomalies = self._detect_zscore_anomalies(
                api, metric_type, sorted_data, timestamps, values, baseline, value_key
            )
        elif algo == AnomalyDetectionAlgorithm.MOVING_AVERAGE:
            anomalies = self._detect_moving_average_anomalies(
                api, metric_type, sorted_data, timestamps, values, baseline, value_key
            )
        elif algo == AnomalyDetectionAlgorithm.IQR:
            anomalies = self._detect_iqr_anomalies(
                api, metric_type, sorted_data, timestamps, values, baseline, value_key
            )
        elif algo == AnomalyDetectionAlgorithm.PATTERN_DETECTION:
            anomalies = self._detect_pattern_anomalies(
                api, metric_type, sorted_data, timestamps, values, baseline, value_key
            )
        elif algo == AnomalyDetectionAlgorithm.SEASONALITY:
            anomalies = self._detect_seasonality_anomalies(
                api, metric_type, sorted_data, timestamps, values, baseline, value_key
            )
        elif algo == AnomalyDetectionAlgorithm.ENSEMBLE:
            # Run all algorithms and combine results
            zscore_anomalies = self._detect_zscore_anomalies(
                api, metric_type, sorted_data, timestamps, values, baseline, value_key
            )
            ma_anomalies = self._detect_moving_average_anomalies(
                api, metric_type, sorted_data, timestamps, values, baseline, value_key
            )
            iqr_anomalies = self._detect_iqr_anomalies(
                api, metric_type, sorted_data, timestamps, values, baseline, value_key
            )
            pattern_anomalies = self._detect_pattern_anomalies(
                api, metric_type, sorted_data, timestamps, values, baseline, value_key
            )
            seasonality_anomalies = self._detect_seasonality_anomalies(
                api, metric_type, sorted_data, timestamps, values, baseline, value_key
            )
            
            # Combine and deduplicate anomalies
            all_anomalies = zscore_anomalies + ma_anomalies + iqr_anomalies + pattern_anomalies + seasonality_anomalies
            anomalies = self._deduplicate_anomalies(all_anomalies)
        
        # Add anomalies to history
        self.anomaly_history.extend(anomalies)
        
        # Trim history (keep last 100)
        if len(self.anomaly_history) > 100:
            self.anomaly_history = self.anomaly_history[-100:]
        
        # Send notifications if enabled
        if self.notification_enabled and self.notification_manager and anomalies:
            for anomaly in anomalies:
                try:
                    self.notification_manager.notify(anomaly)
                except Exception as e:
                    logger.error(f"Error sending notification: {e}")
        
        return anomalies
    
    def _detect_zscore_anomalies(
        self,
        api: str,
        metric_type: str,
        data: List[Dict[str, Any]],
        timestamps: List[float],
        values: List[float],
        baseline: Dict[str, Any],
        value_key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using Z-score method.
        
        Args:
            api: The API name
            metric_type: The metric type
            data: The original data points
            timestamps: List of timestamps
            values: List of values
            baseline: Baseline statistics
            value_key: Key for extracting values
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Z-score threshold adjusted by sensitivity
        threshold = 3.0 / self.sensitivity
        
        mean = baseline["mean"]
        std_dev = baseline["std_dev"]
        
        # Skip if standard deviation is too small
        if std_dev < 1e-6:
            return []
        
        for i, (ts, value) in enumerate(zip(timestamps, values)):
            # Calculate z-score
            z_score = abs((value - mean) / std_dev)
            
            if z_score > threshold:
                # Determine severity based on z-score
                severity = self._determine_severity(z_score, threshold)
                
                # Determine anomaly type
                anomaly_type = self._determine_anomaly_type(metric_type, value, mean)
                
                # Create anomaly entry
                anomaly = {
                    "timestamp": ts,
                    "api": api,
                    "metric_type": metric_type,
                    "value": value,
                    "expected_value": mean,
                    "detection_method": "zscore",
                    "z_score": z_score,
                    "severity": severity.value,
                    "anomaly_type": anomaly_type.value,
                    "description": f"{anomaly_type.value.replace('_', ' ').title()} detected for {api} {metric_type}"
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_moving_average_anomalies(
        self,
        api: str,
        metric_type: str,
        data: List[Dict[str, Any]],
        timestamps: List[float],
        values: List[float],
        baseline: Dict[str, Any],
        value_key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using moving average method.
        
        Args:
            api: The API name
            metric_type: The metric type
            data: The original data points
            timestamps: List of timestamps
            values: List of values
            baseline: Baseline statistics
            value_key: Key for extracting values
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Need enough data points for moving average
        if len(values) < self.window_size:
            return []
        
        # Deviation threshold adjusted by sensitivity
        threshold = 2.0 / self.sensitivity
        
        for i in range(self.window_size, len(values)):
            # Calculate moving average
            window = values[i-self.window_size:i]
            moving_avg = np.mean(window)
            
            # Current value
            value = values[i]
            
            # Calculate standard deviation of window
            window_std = np.std(window)
            
            # Skip if window standard deviation is too small
            if window_std < 1e-6:
                continue
            
            # Calculate deviation
            deviation = abs((value - moving_avg) / window_std)
            
            if deviation > threshold:
                # Determine severity
                severity = self._determine_severity(deviation, threshold)
                
                # Determine anomaly type
                anomaly_type = self._determine_anomaly_type(metric_type, value, moving_avg)
                
                # Create anomaly entry
                anomaly = {
                    "timestamp": timestamps[i],
                    "api": api,
                    "metric_type": metric_type,
                    "value": value,
                    "expected_value": moving_avg,
                    "detection_method": "moving_average",
                    "deviation": deviation,
                    "severity": severity.value,
                    "anomaly_type": anomaly_type.value,
                    "description": f"{anomaly_type.value.replace('_', ' ').title()} detected for {api} {metric_type}"
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_iqr_anomalies(
        self,
        api: str,
        metric_type: str,
        data: List[Dict[str, Any]],
        timestamps: List[float],
        values: List[float],
        baseline: Dict[str, Any],
        value_key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using interquartile range (IQR) method.
        
        Args:
            api: The API name
            metric_type: The metric type
            data: The original data points
            timestamps: List of timestamps
            values: List of values
            baseline: Baseline statistics
            value_key: Key for extracting values
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Get IQR from baseline
        q1 = baseline["q1"]
        q3 = baseline["q3"]
        iqr = baseline["iqr"]
        
        # Skip if IQR is too small
        if iqr < 1e-6:
            return []
        
        # IQR threshold adjusted by sensitivity
        threshold = 1.5 / self.sensitivity
        
        # Define bounds
        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)
        
        for i, (ts, value) in enumerate(zip(timestamps, values)):
            # Check if value is outside bounds
            if value < lower_bound or value > upper_bound:
                # Calculate how many IQRs away
                iqr_distance = min(
                    abs(value - lower_bound) / iqr if value < lower_bound else float('inf'),
                    abs(value - upper_bound) / iqr if value > upper_bound else float('inf')
                )
                
                # Determine severity
                severity = self._determine_severity(iqr_distance, threshold)
                
                # Determine anomaly type
                anomaly_type = self._determine_anomaly_type(metric_type, value, (q1 + q3) / 2)
                
                # Create anomaly entry
                anomaly = {
                    "timestamp": ts,
                    "api": api,
                    "metric_type": metric_type,
                    "value": value,
                    "expected_range": [lower_bound, upper_bound],
                    "detection_method": "iqr",
                    "iqr_distance": iqr_distance,
                    "severity": severity.value,
                    "anomaly_type": anomaly_type.value,
                    "description": f"{anomaly_type.value.replace('_', ' ').title()} detected for {api} {metric_type}"
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _determine_severity(
        self, 
        deviation: float, 
        threshold: float
    ) -> AnomalySeverity:
        """
        Determine the severity of an anomaly based on its deviation.
        
        Args:
            deviation: The deviation from expected value
            threshold: The threshold for anomaly detection
            
        Returns:
            Severity level
        """
        if deviation > threshold * 3:
            return AnomalySeverity.CRITICAL
        elif deviation > threshold * 2:
            return AnomalySeverity.HIGH
        elif deviation > threshold * 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _detect_pattern_anomalies(
        self,
        api: str,
        metric_type: str,
        data: List[Dict[str, Any]],
        timestamps: List[float],
        values: List[float],
        baseline: Dict[str, Any],
        value_key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using pattern detection methods.
        
        This method looks for trend breaks, oscillations, and persistent
        degradation patterns in the data.
        
        Args:
            api: The API name
            metric_type: The metric type
            data: The original data points
            timestamps: List of timestamps
            values: List of values
            baseline: Baseline statistics
            value_key: Key for extracting values
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Need enough data points for pattern detection
        if len(values) < self.window_size * 2:
            return []
        
        # Update trend model for this API and metric if needed
        self._update_trend_model(api, metric_type, timestamps, values)
        
        # Get trend model
        trend_model = self.trend_models.get(api, {}).get(metric_type, {})
        if not trend_model:
            return []
        
        # Check for trend breaks (sudden changes in slope)
        trend_break_anomalies = self._detect_trend_breaks(
            api, metric_type, data, timestamps, values, trend_model, value_key
        )
        anomalies.extend(trend_break_anomalies)
        
        # Check for oscillations (rapid up and down movements)
        oscillation_anomalies = self._detect_oscillations(
            api, metric_type, data, timestamps, values, trend_model, value_key
        )
        anomalies.extend(oscillation_anomalies)
        
        # Check for persistent degradation (continual decline/increase beyond expected)
        degradation_anomalies = self._detect_persistent_degradation(
            api, metric_type, data, timestamps, values, trend_model, value_key
        )
        anomalies.extend(degradation_anomalies)
        
        return anomalies
    
    def _detect_seasonality_anomalies(
        self,
        api: str,
        metric_type: str,
        data: List[Dict[str, Any]],
        timestamps: List[float],
        values: List[float],
        baseline: Dict[str, Any],
        value_key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using seasonality-aware methods.
        
        This method looks for deviations from expected seasonal patterns,
        like time-of-day, day-of-week, or other cyclical patterns.
        
        Args:
            api: The API name
            metric_type: The metric type
            data: The original data points
            timestamps: List of timestamps
            values: List of values
            baseline: Baseline statistics
            value_key: Key for extracting values
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Need enough data points for seasonality detection (at least 2 days of data)
        if len(values) < 48:  # Assuming 30 min intervals, need 48 points for 24 hours
            return []
        
        # Update seasonality patterns for this API and metric if needed
        self._update_seasonality_patterns(api, metric_type, timestamps, values)
        
        # Get seasonality patterns
        patterns = self.seasonality_patterns.get(api, {}).get(metric_type, {})
        if not patterns:
            return []
        
        # Check for deviations from hourly patterns
        if "hourly" in patterns:
            hourly_anomalies = self._detect_hourly_pattern_deviations(
                api, metric_type, data, timestamps, values, patterns["hourly"], value_key
            )
            anomalies.extend(hourly_anomalies)
        
        # Check for deviations from daily patterns
        if "daily" in patterns:
            daily_anomalies = self._detect_daily_pattern_deviations(
                api, metric_type, data, timestamps, values, patterns["daily"], value_key
            )
            anomalies.extend(daily_anomalies)
        
        return anomalies
    
    def _update_trend_model(
        self,
        api: str,
        metric_type: str,
        timestamps: List[float],
        values: List[float]
    ) -> None:
        """
        Update trend model for an API and metric type.
        
        Args:
            api: The API name
            metric_type: The metric type
            timestamps: List of timestamps
            values: List of values
        """
        if len(values) < self.window_size:
            return
        
        # Initialize trend models dictionary if needed
        if api not in self.trend_models:
            self.trend_models[api] = {}
        
        if metric_type not in self.trend_models[api]:
            self.trend_models[api][metric_type] = {}
        
        # Calculate recent slope using linear regression
        # We'll use the numpy polyfit function to fit a line to the data
        try:
            # Convert timestamps to relative time (seconds since first timestamp)
            relative_times = [t - timestamps[0] for t in timestamps]
            
            # Calculate overall trend
            coeffs = np.polyfit(relative_times, values, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Calculate short-term trends using rolling window
            short_term_slopes = []
            for i in range(len(values) - self.window_size + 1):
                window_times = relative_times[i:i + self.window_size]
                window_values = values[i:i + self.window_size]
                window_coeffs = np.polyfit(window_times, window_values, 1)
                short_term_slopes.append(window_coeffs[0])
            
            # Calculate slope variability
            slope_variability = np.std(short_term_slopes) if short_term_slopes else 0
            
            # Update trend model
            self.trend_models[api][metric_type] = {
                "slope": slope,
                "intercept": intercept,
                "short_term_slopes": short_term_slopes,
                "slope_variability": slope_variability,
                "last_update": time.time()
            }
            
            logger.debug(f"Updated trend model for {api} {metric_type}: slope={slope:.6f}")
            
        except Exception as e:
            logger.error(f"Error updating trend model: {e}")
    
    def _update_seasonality_patterns(
        self,
        api: str,
        metric_type: str,
        timestamps: List[float],
        values: List[float]
    ) -> None:
        """
        Update seasonality patterns for an API and metric type.
        
        Args:
            api: The API name
            metric_type: The metric type
            timestamps: List of timestamps
            values: List of values
        """
        if len(timestamps) < 48:  # Need at least 24 hours of data (assuming 30min intervals)
            return
        
        # Initialize seasonality patterns dictionary if needed
        if api not in self.seasonality_patterns:
            self.seasonality_patterns[api] = {}
        
        if metric_type not in self.seasonality_patterns[api]:
            self.seasonality_patterns[api][metric_type] = {}
        
        try:
            # Hourly patterns (values grouped by hour of day)
            hourly_patterns = {}
            for i, (ts, val) in enumerate(zip(timestamps, values)):
                dt = datetime.fromtimestamp(ts)
                hour = dt.hour
                
                if hour not in hourly_patterns:
                    hourly_patterns[hour] = []
                
                hourly_patterns[hour].append(val)
            
            # Calculate hourly statistics
            hourly_stats = {}
            for hour, hour_values in hourly_patterns.items():
                if len(hour_values) >= 3:  # Need at least 3 data points per hour
                    hourly_stats[hour] = {
                        "mean": np.mean(hour_values),
                        "std": np.std(hour_values),
                        "median": np.median(hour_values),
                        "count": len(hour_values)
                    }
            
            # Daily patterns (values grouped by day of week)
            daily_patterns = {}
            for i, (ts, val) in enumerate(zip(timestamps, values)):
                dt = datetime.fromtimestamp(ts)
                day = dt.weekday()  # 0 = Monday, 6 = Sunday
                
                if day not in daily_patterns:
                    daily_patterns[day] = []
                
                daily_patterns[day].append(val)
            
            # Calculate daily statistics
            daily_stats = {}
            for day, day_values in daily_patterns.items():
                if len(day_values) >= 3:  # Need at least 3 data points per day
                    daily_stats[day] = {
                        "mean": np.mean(day_values),
                        "std": np.std(day_values),
                        "median": np.median(day_values),
                        "count": len(day_values)
                    }
            
            # Update seasonality patterns
            self.seasonality_patterns[api][metric_type] = {
                "hourly": hourly_stats,
                "daily": daily_stats,
                "last_update": time.time()
            }
            
            logger.debug(f"Updated seasonality patterns for {api} {metric_type}")
            
        except Exception as e:
            logger.error(f"Error updating seasonality patterns: {e}")
    
    def _detect_trend_breaks(
        self,
        api: str,
        metric_type: str,
        data: List[Dict[str, Any]],
        timestamps: List[float],
        values: List[float],
        trend_model: Dict[str, Any],
        value_key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect trend breaks in time series data.
        
        Args:
            api: The API name
            metric_type: The metric type
            data: The original data points
            timestamps: List of timestamps
            values: List of values
            trend_model: Trend model for this API and metric
            value_key: Key for extracting values
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(values) < self.window_size * 2:
            return anomalies
        
        # Get model parameters
        overall_slope = trend_model.get("slope", 0)
        short_term_slopes = trend_model.get("short_term_slopes", [])
        slope_variability = trend_model.get("slope_variability", 0)
        
        if not short_term_slopes or slope_variability < 1e-6:
            return anomalies
        
        # Threshold for significant slope change, adjusted by sensitivity
        threshold = 3.0 / self.sensitivity
        
        # Check for trend breaks in the most recent window
        if len(short_term_slopes) >= 2:
            recent_slope = short_term_slopes[-1]
            prev_slope = short_term_slopes[-2]
            
            # Normalize the slope change by slope variability
            normalized_change = abs(recent_slope - prev_slope) / slope_variability
            
            # If the change is significant, flag as anomaly
            if normalized_change > threshold:
                # Determine severity based on how extreme the change is
                severity = self._determine_severity(normalized_change, threshold)
                
                # Create anomaly entry
                anomaly = {
                    "timestamp": timestamps[-1],
                    "api": api,
                    "metric_type": metric_type,
                    "value": values[-1],
                    "expected_value": values[-self.window_size-1] + (prev_slope * (timestamps[-1] - timestamps[-self.window_size-1])),
                    "detection_method": "trend_break",
                    "normalized_change": normalized_change,
                    "severity": severity.value,
                    "anomaly_type": AnomalyType.TREND_BREAK.value,
                    "description": f"Trend break detected for {api} {metric_type}"
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_oscillations(
        self,
        api: str,
        metric_type: str,
        data: List[Dict[str, Any]],
        timestamps: List[float],
        values: List[float],
        trend_model: Dict[str, Any],
        value_key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect unusual oscillations in time series data.
        
        Args:
            api: The API name
            metric_type: The metric type
            data: The original data points
            timestamps: List of timestamps
            values: List of values
            trend_model: Trend model for this API and metric
            value_key: Key for extracting values
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(values) < self.window_size:
            return anomalies
        
        # Calculate rate of change for consecutive points
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        
        # Count sign changes (oscillations)
        sign_changes = 0
        for i in range(len(diffs)-1):
            if diffs[i] * diffs[i+1] < 0:  # Sign change
                sign_changes += 1
        
        # Calculate expected number of sign changes for random walk
        # In a random walk, we expect sign changes about 50% of the time
        expected_changes = (len(diffs) - 1) / 2
        
        # If sign changes are significantly higher than expected, flag as anomaly
        if sign_changes > expected_changes * 1.5:  # Adjust threshold as needed
            # Create anomaly entry
            anomaly = {
                "timestamp": timestamps[-1],
                "api": api,
                "metric_type": metric_type,
                "value": sign_changes,
                "expected_value": expected_changes,
                "detection_method": "oscillation",
                "severity": AnomalySeverity.MEDIUM.value,
                "anomaly_type": AnomalyType.OSCILLATION.value,
                "description": f"Unusual oscillation pattern detected for {api} {metric_type}"
            }
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_persistent_degradation(
        self,
        api: str,
        metric_type: str,
        data: List[Dict[str, Any]],
        timestamps: List[float],
        values: List[float],
        trend_model: Dict[str, Any],
        value_key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect persistent degradation in performance metrics.
        
        Args:
            api: The API name
            metric_type: The metric type
            data: The original data points
            timestamps: List of timestamps
            values: List of values
            trend_model: Trend model for this API and metric
            value_key: Key for extracting values
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(values) < self.window_size * 2:
            return anomalies
        
        # Get model parameters
        overall_slope = trend_model.get("slope", 0)
        
        # Define what constitutes degradation based on metric type
        is_degradation = False
        
        if metric_type == "latency":
            # For latency, persistent increase is bad
            is_degradation = overall_slope > 0 and values[-1] > values[0] * 1.2
        elif metric_type == "throughput":
            # For throughput, persistent decrease is bad
            is_degradation = overall_slope < 0 and values[-1] < values[0] * 0.8
        elif metric_type == "reliability":
            # For reliability, persistent decrease is bad
            is_degradation = overall_slope < 0 and values[-1] < values[0] * 0.9
        elif metric_type == "cost":
            # For cost, persistent increase is bad
            is_degradation = overall_slope > 0 and values[-1] > values[0] * 1.2
        
        # If degradation is detected, flag as anomaly
        if is_degradation:
            # Determine severity based on magnitude of degradation
            severity_ratio = abs(values[-1] / values[0] - 1)
            severity = AnomalySeverity.LOW
            if severity_ratio > 0.5:
                severity = AnomalySeverity.CRITICAL
            elif severity_ratio > 0.3:
                severity = AnomalySeverity.HIGH
            elif severity_ratio > 0.1:
                severity = AnomalySeverity.MEDIUM
            
            # Create anomaly entry
            anomaly = {
                "timestamp": timestamps[-1],
                "api": api,
                "metric_type": metric_type,
                "value": values[-1],
                "expected_value": values[0],
                "detection_method": "persistent_degradation",
                "severity": severity.value,
                "anomaly_type": AnomalyType.PERSISTENT_DEGRADATION.value,
                "description": f"Persistent degradation detected for {api} {metric_type}"
            }
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_hourly_pattern_deviations(
        self,
        api: str,
        metric_type: str,
        data: List[Dict[str, Any]],
        timestamps: List[float],
        values: List[float],
        hourly_patterns: Dict[int, Dict[str, Any]],
        value_key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect deviations from hourly patterns.
        
        Args:
            api: The API name
            metric_type: The metric type
            data: The original data points
            timestamps: List of timestamps
            values: List of values
            hourly_patterns: Hourly pattern statistics
            value_key: Key for extracting values
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not hourly_patterns:
            return anomalies
        
        # Check the most recent values for deviations from hourly patterns
        # We'll check the last 6 hours (assuming data points are reasonably frequent)
        check_count = min(len(values), 12)  # Check up to 12 most recent points
        
        for i in range(check_count):
            idx = len(values) - 1 - i
            if idx < 0:
                continue
            
            ts = timestamps[idx]
            val = values[idx]
            
            # Get hour of day
            dt = datetime.fromtimestamp(ts)
            hour = dt.hour
            
            # Skip if we don't have pattern data for this hour
            if hour not in hourly_patterns:
                continue
            
            # Get expected value and standard deviation for this hour
            expected = hourly_patterns[hour]["mean"]
            std_dev = hourly_patterns[hour]["std"]
            
            # Skip if standard deviation is too small
            if std_dev < 1e-6:
                continue
            
            # Calculate z-score
            z_score = abs((val - expected) / std_dev)
            
            # Define threshold, adjusted by sensitivity
            threshold = 3.0 / self.sensitivity
            
            # If z-score exceeds threshold, flag as anomaly
            if z_score > threshold:
                # Determine severity
                severity = self._determine_severity(z_score, threshold)
                
                # Create anomaly entry
                anomaly = {
                    "timestamp": ts,
                    "api": api,
                    "metric_type": metric_type,
                    "value": val,
                    "expected_value": expected,
                    "detection_method": "hourly_pattern",
                    "z_score": z_score,
                    "severity": severity.value,
                    "anomaly_type": AnomalyType.SEASONAL_DEVIATION.value,
                    "description": f"Hourly pattern deviation detected for {api} {metric_type} at hour {hour}"
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_daily_pattern_deviations(
        self,
        api: str,
        metric_type: str,
        data: List[Dict[str, Any]],
        timestamps: List[float],
        values: List[float],
        daily_patterns: Dict[int, Dict[str, Any]],
        value_key: str
    ) -> List[Dict[str, Any]]:
        """
        Detect deviations from daily patterns.
        
        Args:
            api: The API name
            metric_type: The metric type
            data: The original data points
            timestamps: List of timestamps
            values: List of values
            daily_patterns: Daily pattern statistics
            value_key: Key for extracting values
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not daily_patterns:
            return anomalies
        
        # Check the most recent values for deviations from daily patterns
        check_count = min(len(values), 24)  # Check up to 24 most recent points
        
        for i in range(check_count):
            idx = len(values) - 1 - i
            if idx < 0:
                continue
            
            ts = timestamps[idx]
            val = values[idx]
            
            # Get day of week
            dt = datetime.fromtimestamp(ts)
            day = dt.weekday()  # 0 = Monday, 6 = Sunday
            
            # Skip if we don't have pattern data for this day
            if day not in daily_patterns:
                continue
            
            # Get expected value and standard deviation for this day
            expected = daily_patterns[day]["mean"]
            std_dev = daily_patterns[day]["std"]
            
            # Skip if standard deviation is too small
            if std_dev < 1e-6:
                continue
            
            # Calculate z-score
            z_score = abs((val - expected) / std_dev)
            
            # Define threshold, adjusted by sensitivity
            threshold = 3.0 / self.sensitivity
            
            # If z-score exceeds threshold, flag as anomaly
            if z_score > threshold:
                # Determine severity
                severity = self._determine_severity(z_score, threshold)
                
                # Create anomaly entry
                anomaly = {
                    "timestamp": ts,
                    "api": api,
                    "metric_type": metric_type,
                    "value": val,
                    "expected_value": expected,
                    "detection_method": "daily_pattern",
                    "z_score": z_score,
                    "severity": severity.value,
                    "anomaly_type": AnomalyType.SEASONAL_DEVIATION.value,
                    "description": f"Daily pattern deviation detected for {api} {metric_type} on {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day]}"
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _determine_anomaly_type(
        self, 
        metric_type: str, 
        value: float, 
        expected: float
    ) -> AnomalyType:
        """
        Determine the type of anomaly based on the metric and deviation.
        
        Args:
            metric_type: The metric type
            value: The actual value
            expected: The expected value
            
        Returns:
            Anomaly type
        """
        if metric_type == "latency":
            if value > expected:
                return AnomalyType.LATENCY_SPIKE
            else:
                return AnomalyType.PATTERN_CHANGE
        elif metric_type == "throughput":
            if value < expected:
                return AnomalyType.THROUGHPUT_DROP
            else:
                return AnomalyType.PATTERN_CHANGE
        elif metric_type == "reliability":
            if value < expected:
                return AnomalyType.ERROR_RATE_INCREASE
            else:
                return AnomalyType.PATTERN_CHANGE
        elif metric_type == "cost":
            if value > expected:
                return AnomalyType.COST_SPIKE
            else:
                return AnomalyType.PATTERN_CHANGE
        else:
            return AnomalyType.PATTERN_CHANGE
    
    def _deduplicate_anomalies(
        self, 
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate anomalies by timestamp and metric.
        
        Args:
            anomalies: List of anomalies
            
        Returns:
            Deduplicated list of anomalies
        """
        if not anomalies:
            return []
        
        # Sort by timestamp
        sorted_anomalies = sorted(anomalies, key=lambda x: x["timestamp"])
        
        # Group by timestamp, api, and metric_type
        grouped = {}
        for anomaly in sorted_anomalies:
            key = (anomaly["timestamp"], anomaly["api"], anomaly["metric_type"])
            
            if key not in grouped:
                grouped[key] = []
            
            grouped[key].append(anomaly)
        
        # Take the highest severity anomaly from each group
        deduped = []
        for group in grouped.values():
            if not group:
                continue
            
            # Sort by severity (critical > high > medium > low)
            severity_order = {
                AnomalySeverity.CRITICAL.value: 0,
                AnomalySeverity.HIGH.value: 1,
                AnomalySeverity.MEDIUM.value: 2,
                AnomalySeverity.LOW.value: 3
            }
            
            sorted_group = sorted(group, key=lambda x: severity_order.get(x["severity"], 999))
            
            # Add highest severity anomaly
            deduped.append(sorted_group[0])
        
        return deduped
    
    def get_recent_anomalies(
        self, 
        hours: int = 24, 
        min_severity: AnomalySeverity = AnomalySeverity.LOW
    ) -> List[Dict[str, Any]]:
        """
        Get recent anomalies filtered by minimum severity.
        
        Args:
            hours: Number of hours to look back
            min_severity: Minimum severity level to include
            
        Returns:
            List of recent anomalies
        """
        if not self.anomaly_history:
            return []
        
        # Calculate cutoff time
        cutoff = time.time() - (hours * 60 * 60)
        
        # Severity order for comparison
        severity_order = {
            AnomalySeverity.LOW.value: 0,
            AnomalySeverity.MEDIUM.value: 1,
            AnomalySeverity.HIGH.value: 2,
            AnomalySeverity.CRITICAL.value: 3
        }
        
        min_severity_level = severity_order.get(min_severity.value, 0)
        
        # Filter anomalies
        recent_anomalies = [
            anomaly for anomaly in self.anomaly_history
            if anomaly["timestamp"] >= cutoff and 
               severity_order.get(anomaly["severity"], 0) >= min_severity_level
        ]
        
        # Sort by timestamp (newest first)
        return sorted(recent_anomalies, key=lambda x: x["timestamp"], reverse=True)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent anomalies.
        
        Returns:
            Dictionary with anomaly summary
        """
        # Get recent anomalies (last 24 hours)
        recent_anomalies = self.get_recent_anomalies(hours=24)
        
        if not recent_anomalies:
            return {
                "total_anomalies": 0,
                "severity_counts": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0
                },
                "type_counts": {},
                "api_counts": {},
                "metric_counts": {},
                "latest_anomaly": None
            }
        
        # Count anomalies by severity
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for anomaly in recent_anomalies:
            severity = anomaly.get("severity")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Count anomalies by type
        type_counts = {}
        for anomaly in recent_anomalies:
            anomaly_type = anomaly.get("anomaly_type")
            if anomaly_type:
                if anomaly_type not in type_counts:
                    type_counts[anomaly_type] = 0
                type_counts[anomaly_type] += 1
        
        # Count anomalies by API
        api_counts = {}
        for anomaly in recent_anomalies:
            api = anomaly.get("api")
            if api:
                if api not in api_counts:
                    api_counts[api] = 0
                api_counts[api] += 1
        
        # Count anomalies by metric
        metric_counts = {}
        for anomaly in recent_anomalies:
            metric = anomaly.get("metric_type")
            if metric:
                if metric not in metric_counts:
                    metric_counts[metric] = 0
                metric_counts[metric] += 1
        
        # Get latest anomaly
        latest_anomaly = recent_anomalies[0] if recent_anomalies else None
        
        return {
            "total_anomalies": len(recent_anomalies),
            "severity_counts": severity_counts,
            "type_counts": type_counts,
            "api_counts": api_counts,
            "metric_counts": metric_counts,
            "latest_anomaly": latest_anomaly
        }
    
    def save_anomaly_data(self, data_dir: str) -> bool:
        """
        Save anomaly data to disk.
        
        Args:
            data_dir: Directory to save data
            
        Returns:
            True if successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Save anomaly history
            history_file = os.path.join(data_dir, "anomaly_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.anomaly_history, f, indent=2)
            
            # Save baselines
            baselines_file = os.path.join(data_dir, "anomaly_baselines.json")
            with open(baselines_file, 'w') as f:
                # Convert numpy types to regular Python types for JSON serialization
                serializable_baselines = {}
                for api, metrics in self.baselines.items():
                    serializable_baselines[api] = {}
                    for metric, baseline in metrics.items():
                        serializable_baselines[api][metric] = {
                            k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                            for k, v in baseline.items()
                        }
                
                json.dump(serializable_baselines, f, indent=2)
            
            logger.info(f"Saved anomaly data to {data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving anomaly data: {e}")
            return False
    
    def load_anomaly_data(self, data_dir: str) -> bool:
        """
        Load anomaly data from disk.
        
        Args:
            data_dir: Directory to load data from
            
        Returns:
            True if successful
        """
        # Check for history file
        history_file = os.path.join(data_dir, "anomaly_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.anomaly_history = json.load(f)
                logger.info(f"Loaded anomaly history from {history_file}")
            except Exception as e:
                logger.error(f"Error loading anomaly history: {e}")
                return False
        
        # Check for baselines file
        baselines_file = os.path.join(data_dir, "anomaly_baselines.json")
        if os.path.exists(baselines_file):
            try:
                with open(baselines_file, 'r') as f:
                    self.baselines = json.load(f)
                logger.info(f"Loaded anomaly baselines from {baselines_file}")
            except Exception as e:
                logger.error(f"Error loading anomaly baselines: {e}")
                return False
        
        return True


def generate_anomaly_alert_html(anomaly: Dict[str, Any]) -> str:
    """
    Generate HTML for an anomaly alert.
    
    Args:
        anomaly: The anomaly data
        
    Returns:
        HTML string for the alert
    """
    severity = anomaly.get("severity", "low")
    severity_class = "text-warning"
    
    if severity == "critical":
        severity_class = "text-danger"
    elif severity == "high":
        severity_class = "text-danger"
    elif severity == "medium":
        severity_class = "text-warning"
    elif severity == "low":
        severity_class = "text-info"
    
    # Format timestamp
    timestamp = datetime.fromtimestamp(anomaly["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
    
    # Format expected value
    expected = anomaly.get("expected_value")
    if expected is None and "expected_range" in anomaly:
        expected = f"{anomaly['expected_range'][0]:.4f} - {anomaly['expected_range'][1]:.4f}"
    elif expected is not None:
        expected = f"{expected:.4f}"
    else:
        expected = "N/A"
    
    # Format actual value
    value = anomaly.get("value")
    if value is not None:
        value = f"{value:.4f}"
    else:
        value = "N/A"
    
    html = f"""
    <div class="anomaly-alert {severity_class}-bg">
        <div class="anomaly-header">
            <span class="anomaly-severity {severity_class}">{severity.upper()}</span>
            <span class="anomaly-timestamp">{timestamp}</span>
        </div>
        <div class="anomaly-title">
            {anomaly.get("description", "Anomaly Detected")}
        </div>
        <div class="anomaly-details">
            <div><strong>API:</strong> {anomaly.get("api", "Unknown").upper()}</div>
            <div><strong>Metric:</strong> {anomaly.get("metric_type", "Unknown")}</div>
            <div><strong>Value:</strong> {value}</div>
            <div><strong>Expected:</strong> {expected}</div>
            <div><strong>Detection:</strong> {anomaly.get("detection_method", "Unknown")}</div>
        </div>
    </div>
    """
    
    return html


def main():
    """Main function for testing anomaly detection."""
    # Create sample data
    sample_data = {
        "latency": {
            "openai": [
                {"timestamp": time.time() - 3600 * i, "avg_latency": 1.2 + (0.1 if i != 5 else 3.0)}
                for i in range(24)
            ]
        }
    }
    
    # Create anomaly detector
    detector = AnomalyDetector()
    
    # Update baseline
    detector.update_baseline("openai", "latency", sample_data["latency"]["openai"])
    
    # Detect anomalies
    anomalies = detector.detect_anomalies("openai", "latency", sample_data["latency"]["openai"])
    
    # Print results
    print(f"Detected {len(anomalies)} anomalies:")
    for anomaly in anomalies:
        timestamp = datetime.fromtimestamp(anomaly["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {timestamp} - {anomaly['description']} (severity: {anomaly['severity']})")


if __name__ == "__main__":
    # Import numpy here to avoid dependency issues
    try:
        import numpy as np
    except ImportError:
        print("Error: This module requires numpy to be installed.")
        print("Install it with: pip install numpy")
        sys.exit(1)
    
    main()