#!/usr/bin/env python
"""
API Predictive Analytics Module

This module provides advanced predictive analytics and machine learning capabilities
for forecasting API performance metrics and detecting anomalies before they occur.

Features:
1. Machine learning based anomaly prediction
2. Trend forecasting for capacity planning
3. Anomaly root cause analysis 
4. Performance degradation prediction
5. Cost optimization recommendations
6. Advanced seasonality modeling

Usage:
    Import this module into the API monitoring dashboard for predictive analytics.
"""

import os
import sys
import time
import logging
import json
import warnings
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_predictive_analytics")

# Try importing optional ML dependencies
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    ML_AVAILABLE = True
    logger.info("Machine learning dependencies available")
except ImportError:
    ML_AVAILABLE = False
    logger.warning("Machine learning dependencies not available - install scikit-learn for full functionality")

# Try importing anomaly detection types if available
try:
    from api_anomaly_detection import AnomalySeverity, AnomalyType
    ANOMALY_TYPES_AVAILABLE = True
except ImportError:
    logger.warning("Anomaly detection types not available, using local definitions")
    ANOMALY_TYPES_AVAILABLE = False
    
    # Local definitions if imports fail
    class AnomalySeverity(Enum):
        """Severity levels for detected anomalies."""
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
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


class PredictionHorizon(Enum):
    """Time horizon for predictions."""
    SHORT_TERM = "short_term"  # Minutes to hours
    MEDIUM_TERM = "medium_term"  # Hours to days
    LONG_TERM = "long_term"  # Days to weeks


class ModelType(Enum):
    """Available prediction model types."""
    ISOLATION_FOREST = "isolation_forest"
    RANDOM_FOREST = "random_forest"
    RIDGE_REGRESSION = "ridge_regression"
    ARIMA = "arima"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class FeatureImportance:
    """Helper for tracking feature importance in models."""
    
    def __init__(self):
        """Initialize feature importance tracker."""
        self.features = {}
    
    def update(self, feature_dict: Dict[str, float]):
        """
        Update feature importance scores.
        
        Args:
            feature_dict: Dictionary of feature names to importance scores
        """
        for feature, importance in feature_dict.items():
            if feature not in self.features:
                self.features[feature] = []
            self.features[feature].append(importance)
    
    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        avg_importance = {}
        for feature, scores in self.features.items():
            avg_importance[feature] = sum(scores) / len(scores)
        
        # Sort by importance (descending)
        sorted_features = sorted(
            avg_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_features[:n]


class FeatureExtractor:
    """
    Extracts and transforms features from time series data.
    
    This class extracts various statistical, temporal, and domain-specific
    features from raw API performance metrics data.
    """
    
    def __init__(self, window_sizes: List[int] = None, use_time_features: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            window_sizes: List of window sizes for rolling features
            use_time_features: Whether to include time-based features
        """
        self.window_sizes = window_sizes or [5, 10, 20]
        self.use_time_features = use_time_features
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.fitted = False
    
    def extract_features(
        self, 
        timestamps: List[float], 
        values: List[float],
        additional_data: Optional[Dict[str, List[float]]] = None
    ) -> np.ndarray:
        """
        Extract features from time series data.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            additional_data: Optional dictionary of additional feature values
            
        Returns:
            Numpy array of features
        """
        if not ML_AVAILABLE:
            logger.warning("ML dependencies not available for feature extraction")
            return np.array([values[-1]])
        
        if len(timestamps) < max(self.window_sizes):
            # Not enough data, return simple features
            return np.array([[
                values[-1],
                np.mean(values),
                np.std(values) if len(values) > 1 else 0
            ]])
        
        features = []
        
        # Process each data point (for training) or just the latest (for prediction)
        for i in range(len(values)):
            # Skip points that don't have enough history for all windows
            if i < max(self.window_sizes):
                continue
            
            point_features = []
            
            # Add raw value
            point_features.append(values[i])
            
            # Add basic statistics for each window size
            for window_size in self.window_sizes:
                window = values[i - window_size:i]
                
                # Basic statistics
                point_features.append(np.mean(window))
                point_features.append(np.std(window))
                point_features.append(np.min(window))
                point_features.append(np.max(window))
                
                # Trend features
                point_features.append(window[-1] - window[0])  # Change over window
                point_features.append((window[-1] - window[0]) / window_size)  # Slope
                
                # Volatility
                diffs = np.diff(window)
                point_features.append(np.mean(np.abs(diffs)))  # Mean abs change
                point_features.append(np.std(diffs))  # Std of changes
                
                # Outlier measures
                z_scores = (window - np.mean(window)) / np.std(window) if np.std(window) > 0 else np.zeros_like(window)
                point_features.append(np.max(np.abs(z_scores)))  # Max z-score
                
                # Add seasonality features if available and window is large enough
                if window_size >= 24:  # Need at least 24 points for hourly patterns
                    try:
                        hour_means = {}
                        for j in range(min(24, window_size)):
                            hour = datetime.fromtimestamp(timestamps[i - j]).hour
                            if hour not in hour_means:
                                hour_means[hour] = []
                            hour_means[hour].append(window[j])
                        
                        # Calculate hourly deviation
                        current_hour = datetime.fromtimestamp(timestamps[i]).hour
                        if current_hour in hour_means and len(hour_means[current_hour]) > 0:
                            hour_mean = np.mean(hour_means[current_hour])
                            hour_dev = values[i] - hour_mean
                            point_features.append(hour_dev)
                    except Exception as e:
                        logger.debug(f"Error calculating hour features: {e}")
                        point_features.append(0)  # Placeholder
            
            # Add time features if enabled
            if self.use_time_features:
                dt = datetime.fromtimestamp(timestamps[i])
                
                # Hour of day (normalized to [0, 1])
                point_features.append(dt.hour / 24.0)
                
                # Day of week (normalized to [0, 1])
                point_features.append(dt.weekday() / 6.0)
                
                # Is weekend
                point_features.append(1.0 if dt.weekday() >= 5 else 0.0)
                
                # Is business hours (9am-5pm)
                point_features.append(1.0 if 9 <= dt.hour < 17 else 0.0)
            
            # Add additional features if provided
            if additional_data:
                for feature_name, feature_values in additional_data.items():
                    if i < len(feature_values):
                        point_features.append(feature_values[i])
            
            features.append(point_features)
        
        if not features:
            logger.warning("No features could be extracted")
            return np.array([[values[-1]]])
        
        # Convert to numpy array
        features_array = np.array(features)
        
        # Handle NaNs
        features_array = np.nan_to_num(features_array)
        
        # Normalize features
        if not self.fitted:
            self.scaler.fit(features_array)
            self.fitted = True
        
        scaled_features = self.scaler.transform(features_array)
        
        return scaled_features
    
    def extract_latest_features(
        self, 
        timestamps: List[float], 
        values: List[float],
        additional_data: Optional[Dict[str, List[float]]] = None
    ) -> np.ndarray:
        """
        Extract features for the latest data point only.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            additional_data: Optional dictionary of additional feature values
            
        Returns:
            Numpy array of features for the latest point
        """
        all_features = self.extract_features(timestamps, values, additional_data)
        if len(all_features) > 0:
            return all_features[-1:]
        return all_features


class TimeSeriesPredictor:
    """
    Time series prediction for API metrics.
    
    This class provides various forecasting models for predicting future
    values of API performance metrics.
    """
    
    def __init__(
        self, 
        model_type: ModelType = ModelType.RANDOM_FOREST,
        forecast_horizon: int = 12,  # Hours
        features_extractor: Optional[FeatureExtractor] = None,
        use_exogenous: bool = True
    ):
        """
        Initialize time series predictor.
        
        Args:
            model_type: Type of prediction model to use
            forecast_horizon: How far ahead to forecast (hours)
            features_extractor: Custom feature extractor
            use_exogenous: Whether to use exogenous variables
        """
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon
        self.use_exogenous = use_exogenous
        
        # Initialize feature extractor if not provided
        self.feature_extractor = features_extractor or FeatureExtractor()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Track feature importance
        self.feature_importance = FeatureImportance()
        
        # Training data
        self.train_features = None
        self.train_targets = None
        
        # Model metadata
        self.train_timestamp = None
        self.train_size = 0
        self.mse = None
        self.is_trained = False
    
    def _initialize_model(self):
        """
        Initialize prediction model based on model type.
        
        Returns:
            Initialized model
        """
        if not ML_AVAILABLE:
            logger.warning("ML dependencies not available for prediction")
            return None
        
        if self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == ModelType.RIDGE_REGRESSION:
            return Ridge(alpha=1.0)
        else:
            # Default to Random Forest for now
            logger.info(f"Model type {self.model_type.value} not implemented yet, using Random Forest")
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
    
    def fit(
        self,
        timestamps: List[float],
        values: List[float],
        exogenous_data: Optional[Dict[str, List[float]]] = None
    ) -> bool:
        """
        Fit the prediction model.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            exogenous_data: Optional dictionary of exogenous variables
            
        Returns:
            True if fitting was successful
        """
        if not ML_AVAILABLE or self.model is None:
            logger.warning("ML dependencies not available for model fitting")
            return False
        
        if len(timestamps) < max(self.feature_extractor.window_sizes) + self.forecast_horizon:
            logger.warning("Not enough data points for training")
            return False
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(
                timestamps, values, 
                exogenous_data if self.use_exogenous else None
            )
            
            if len(features) < self.forecast_horizon + 1:
                logger.warning("Not enough features for training")
                return False
            
            # Set up targets with a time shift (predict n steps ahead)
            targets = values[max(self.feature_extractor.window_sizes) + self.forecast_horizon:]
            
            # Trim features to match target length
            features = features[:len(targets)]
            
            # Verify dimensios
            if len(features) != len(targets):
                logger.warning(f"Feature/target dimension mismatch: {features.shape} vs {len(targets)}")
                return False
            
            # Train the model
            self.model.fit(features, targets)
            
            # Store training data
            self.train_features = features
            self.train_targets = targets
            self.train_timestamp = time.time()
            self.train_size = len(features)
            
            # Calculate MSE on training data
            predictions = self.model.predict(features)
            self.mse = mean_squared_error(targets, predictions)
            
            # Extract feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                # Create feature names
                feature_names = []
                
                # Base value feature
                feature_names.append("current_value")
                
                # Window features
                for window_size in self.feature_extractor.window_sizes:
                    feature_names.extend([
                        f"mean_{window_size}",
                        f"std_{window_size}",
                        f"min_{window_size}",
                        f"max_{window_size}",
                        f"change_{window_size}",
                        f"slope_{window_size}",
                        f"mean_abs_change_{window_size}",
                        f"std_change_{window_size}",
                        f"max_zscore_{window_size}"
                    ])
                    
                    # Add hourly deviation if window is large enough
                    if window_size >= 24:
                        feature_names.append(f"hour_dev_{window_size}")
                
                # Add time features
                if self.feature_extractor.use_time_features:
                    feature_names.extend([
                        "hour_of_day",
                        "day_of_week",
                        "is_weekend",
                        "is_business_hours"
                    ])
                
                # Add exogenous features
                if self.use_exogenous and exogenous_data:
                    feature_names.extend(exogenous_data.keys())
                
                # Trim to actual feature count
                feature_names = feature_names[:self.model.feature_importances_.shape[0]]
                
                # Update importance
                importance_dict = dict(zip(feature_names, self.model.feature_importances_))
                self.feature_importance.update(importance_dict)
            
            self.is_trained = True
            logger.info(f"Model trained successfully with {len(features)} samples, MSE: {self.mse:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training prediction model: {e}")
            return False
    
    def predict(
        self,
        timestamps: List[float],
        values: List[float],
        exogenous_data: Optional[Dict[str, List[float]]] = None,
        steps_ahead: int = 1
    ) -> List[float]:
        """
        Predict future values.
        
        Args:
            timestamps: List of timestamps
            values: List of values
            exogenous_data: Optional dictionary of exogenous variables
            steps_ahead: Number of steps ahead to predict
            
        Returns:
            List of predicted values
        """
        if not ML_AVAILABLE or self.model is None or not self.is_trained:
            logger.warning("Model not available or not trained")
            return []
        
        if len(timestamps) < max(self.feature_extractor.window_sizes):
            logger.warning("Not enough data points for prediction")
            return []
        
        try:
            # Get features for most recent data point
            latest_features = self.feature_extractor.extract_latest_features(
                timestamps, values,
                exogenous_data if self.use_exogenous else None
            )
            
            # Make prediction
            predictions = []
            current_features = latest_features.copy()
            
            # For multi-step forecasting, feed predictions back into features
            for _ in range(steps_ahead):
                # Make prediction for current features
                pred = self.model.predict(current_features)[0]
                predictions.append(pred)
                
                # Update features for next step prediction
                # (In a basic implementation, we just update the first feature which is the value)
                if len(current_features[0]) > 0:
                    current_features[0][0] = pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return []
    
    def calculate_confidence_interval(
        self,
        prediction: float,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for prediction.
        
        Args:
            prediction: Predicted value
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            (lower_bound, upper_bound) tuple
        """
        if not self.is_trained or self.mse is None:
            return (prediction * 0.9, prediction * 1.1)  # Default 10% interval
        
        # Use model MSE to calculate prediction interval
        # For approximately normal errors, 95% interval is ±1.96 standard errors
        z_value = {
            0.68: 1.0,
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }.get(confidence, 1.96)
        
        std_error = np.sqrt(self.mse)
        margin = z_value * std_error
        
        return (prediction - margin, prediction + margin)
    
    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        return self.feature_importance.get_top_features(n)
    
    def save_model(self, path: str) -> bool:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
            
        Returns:
            True if saving was successful
        """
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return False
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            model_data = {
                "model": self.model,
                "feature_extractor": self.feature_extractor,
                "feature_importance": self.feature_importance,
                "model_type": self.model_type.value,
                "forecast_horizon": self.forecast_horizon,
                "use_exogenous": self.use_exogenous,
                "train_timestamp": self.train_timestamp,
                "train_size": self.train_size,
                "mse": self.mse,
                "is_trained": self.is_trained
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load_model(cls, path: str) -> Optional['TimeSeriesPredictor']:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
            
        Returns:
            Loaded TimeSeriesPredictor or None if loading failed
        """
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return None
        
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create predictor with saved model type
            model_type = ModelType(model_data.get("model_type", "random_forest"))
            predictor = cls(
                model_type=model_type,
                forecast_horizon=model_data.get("forecast_horizon", 12),
                features_extractor=model_data.get("feature_extractor"),
                use_exogenous=model_data.get("use_exogenous", True)
            )
            
            # Load saved attributes
            predictor.model = model_data.get("model")
            predictor.feature_importance = model_data.get("feature_importance", FeatureImportance())
            predictor.train_timestamp = model_data.get("train_timestamp")
            predictor.train_size = model_data.get("train_size", 0)
            predictor.mse = model_data.get("mse")
            predictor.is_trained = model_data.get("is_trained", False)
            
            logger.info(f"Model loaded from {path}")
            return predictor
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None


class AnomalyPredictor:
    """
    Anomaly prediction for API metrics.
    
    This class uses machine learning to predict potential anomalies
    before they occur, based on patterns in the data.
    """
    
    def __init__(
        self, 
        prediction_window: int = 12,  # Hours
        contamination: float = 0.05,
        min_train_size: int = 100
    ):
        """
        Initialize anomaly predictor.
        
        Args:
            prediction_window: How far ahead to predict anomalies (hours)
            contamination: Expected ratio of anomalies in training data
            min_train_size: Minimum data points required for training
        """
        self.prediction_window = prediction_window
        self.contamination = contamination
        self.min_train_size = min_train_size
        
        # Initialize models dictionary by API and metric
        self.models = {}
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(window_sizes=[5, 10, 20, 50])
        
        # Training data for each API and metric
        self.training_data = {}
        
        # Predictive forecasting models
        self.forecasters = {}
        
        # Timestamp for last training
        self.last_training_time = {}
        
        # Root cause models
        self.root_cause_models = {}
        
        # Known anomaly patterns
        self.anomaly_patterns = defaultdict(list)
    
    def _get_or_create_model(self, api: str, metric_type: str):
        """
        Get or create model for an API and metric type.
        
        Args:
            api: API name
            metric_type: Metric type
            
        Returns:
            Model instance
        """
        if not ML_AVAILABLE:
            return None
        
        key = f"{api}:{metric_type}"
        
        if key not in self.models:
            # Create isolation forest for anomaly detection
            self.models[key] = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            
            # Create forecaster for this API/metric
            self.forecasters[key] = TimeSeriesPredictor(
                model_type=ModelType.RANDOM_FOREST,
                forecast_horizon=self.prediction_window,
                features_extractor=self.feature_extractor
            )
            
            # Initialize training data
            self.training_data[key] = {
                "timestamps": [],
                "values": [],
                "features": [],
                "known_anomalies": []
            }
            
            # Initialize last training time
            self.last_training_time[key] = 0
        
        return self.models[key]
    
    def record_data(
        self,
        api: str,
        metric_type: str,
        timestamps: List[float],
        values: List[float],
        known_anomalies: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Record data for an API and metric type.
        
        Args:
            api: API name
            metric_type: Metric type
            timestamps: List of timestamps
            values: List of values
            known_anomalies: Optional list of known anomalies
        """
        if not ML_AVAILABLE:
            return
        
        key = f"{api}:{metric_type}"
        
        # Get or create model and training data
        model = self._get_or_create_model(api, metric_type)
        
        if model is None:
            return
        
        # Add data to training set
        self.training_data[key]["timestamps"].extend(timestamps)
        self.training_data[key]["values"].extend(values)
        
        # Add known anomalies if provided
        if known_anomalies:
            self.training_data[key]["known_anomalies"].extend(known_anomalies)
            
            # Record anomaly patterns for root cause analysis
            for anomaly in known_anomalies:
                ts = anomaly.get("timestamp")
                
                # Find index of this timestamp in our data
                if ts in timestamps:
                    idx = timestamps.index(ts)
                    
                    # Get window of data around the anomaly
                    start_idx = max(0, idx - 10)
                    end_idx = min(len(values), idx + 10)
                    
                    pattern = values[start_idx:end_idx]
                    
                    # Store pattern with anomaly type
                    anomaly_type = anomaly.get("anomaly_type", "unknown")
                    self.anomaly_patterns[anomaly_type].append(pattern)
        
        # Check if we should train the model
        if len(self.training_data[key]["timestamps"]) >= self.min_train_size:
            current_time = time.time()
            
            # Retrain every 24 hours
            if current_time - self.last_training_time.get(key, 0) > 24 * 60 * 60:
                self.train(api, metric_type)
    
    def train(self, api: str, metric_type: str) -> bool:
        """
        Train models for an API and metric type.
        
        Args:
            api: API name
            metric_type: Metric type
            
        Returns:
            True if training was successful
        """
        if not ML_AVAILABLE:
            logger.warning("ML dependencies not available for anomaly prediction")
            return False
        
        key = f"{api}:{metric_type}"
        
        if key not in self.training_data:
            logger.warning(f"No training data available for {api} {metric_type}")
            return False
        
        training_data = self.training_data[key]
        
        if len(training_data["timestamps"]) < self.min_train_size:
            logger.warning(f"Not enough training data for {api} {metric_type}")
            return False
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(
                training_data["timestamps"],
                training_data["values"]
            )
            
            # Store features
            training_data["features"] = features
            
            # Train anomaly detection model
            model = self.models[key]
            model.fit(features)
            
            # Train forecasting model
            forecaster = self.forecasters[key]
            forecaster.fit(
                training_data["timestamps"],
                training_data["values"]
            )
            
            # Train root cause model if we have known anomalies
            if training_data["known_anomalies"]:
                self._train_root_cause_model(api, metric_type, training_data)
            
            # Update last training time
            self.last_training_time[key] = time.time()
            
            logger.info(f"Trained anomaly prediction models for {api} {metric_type} with {len(features)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training anomaly prediction models: {e}")
            return False
    
    def _train_root_cause_model(
        self, 
        api: str, 
        metric_type: str, 
        training_data: Dict[str, Any]
    ) -> None:
        """
        Train root cause analysis model.
        
        Args:
            api: API name
            metric_type: Metric type
            training_data: Training data
        """
        if not ML_AVAILABLE:
            return
        
        key = f"{api}:{metric_type}"
        
        try:
            # Get features for known anomalies
            anomaly_features = []
            anomaly_types = []
            
            for anomaly in training_data["known_anomalies"]:
                # Find index of this timestamp in our data
                ts = anomaly.get("timestamp")
                if ts not in training_data["timestamps"]:
                    continue
                
                idx = training_data["timestamps"].index(ts)
                if idx >= len(training_data["features"]):
                    continue
                
                # Get features for this anomaly
                features = training_data["features"][idx]
                
                # Get anomaly type
                anomaly_type = anomaly.get("anomaly_type", "unknown")
                
                anomaly_features.append(features)
                anomaly_types.append(anomaly_type)
            
            if not anomaly_features:
                return
            
            # Convert to numpy arrays
            anomaly_features = np.array(anomaly_features)
            
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=min(5, anomaly_features.shape[1], anomaly_features.shape[0]))
            reduced_features = pca.fit_transform(anomaly_features)
            
            # Use DBSCAN for clustering similar anomalies
            dbscan = DBSCAN(eps=1.0, min_samples=2)
            clusters = dbscan.fit_predict(reduced_features)
            
            # Store root cause model
            self.root_cause_models[key] = {
                "pca": pca,
                "dbscan": dbscan,
                "features": anomaly_features,
                "types": anomaly_types,
                "clusters": clusters
            }
            
            logger.info(f"Trained root cause model for {api} {metric_type} with {len(anomaly_features)} anomalies")
            
        except Exception as e:
            logger.error(f"Error training root cause model: {e}")
    
    def predict_anomalies(
        self,
        api: str,
        metric_type: str,
        timestamps: List[float],
        values: List[float],
        prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM
    ) -> List[Dict[str, Any]]:
        """
        Predict potential anomalies in the future.
        
        Args:
            api: API name
            metric_type: Metric type
            timestamps: List of timestamps
            values: List of values
            prediction_horizon: Time horizon for prediction
            
        Returns:
            List of predicted anomalies
        """
        if not ML_AVAILABLE:
            logger.warning("ML dependencies not available for anomaly prediction")
            return []
        
        key = f"{api}:{metric_type}"
        
        if key not in self.models:
            logger.warning(f"No model available for {api} {metric_type}")
            return []
        
        # Convert prediction horizon to steps
        horizon_steps = {
            PredictionHorizon.SHORT_TERM: 6,   # 6 hours
            PredictionHorizon.MEDIUM_TERM: 24, # 24 hours
            PredictionHorizon.LONG_TERM: 72    # 72 hours
        }.get(prediction_horizon, 6)
        
        try:
            # Check if models are trained
            if key not in self.last_training_time or self.last_training_time[key] == 0:
                # Train if not trained yet
                if not self.train(api, metric_type):
                    return []
            
            # Get forecaster
            forecaster = self.forecasters[key]
            
            # Predict future values
            future_values = forecaster.predict(
                timestamps, values, 
                steps_ahead=horizon_steps
            )
            
            if not future_values:
                return []
            
            # Predict confidence intervals
            predicted_anomalies = []
            
            # Create future timestamps
            future_timestamps = []
            if timestamps:
                last_ts = timestamps[-1]
                interval = 3600  # 1 hour in seconds
                
                for i in range(horizon_steps):
                    future_timestamps.append(last_ts + (i+1) * interval)
            
            # Create features for future values
            future_features = []
            
            # Extract features for the most recent point
            latest_features = self.feature_extractor.extract_latest_features(
                timestamps, values
            )
            
            # For each future value, create a feature vector
            # (In a real implementation, this would be more sophisticated)
            for i, future_val in enumerate(future_values):
                if len(latest_features) > 0 and len(latest_features[0]) > 0:
                    feature = latest_features[0].copy()
                    feature[0] = future_val  # Replace current value with prediction
                    future_features.append(feature)
            
            # Apply anomaly detection model to future features
            if future_features and len(future_features) > 0:
                model = self.models[key]
                anomaly_scores = model.decision_function(future_features)
                predictions = model.predict(future_features)
                
                # Create predicted anomaly objects
                for i, (score, pred) in enumerate(zip(anomaly_scores, predictions)):
                    # Negative score = more anomalous, -1 prediction = anomaly
                    if pred == -1 or score < -0.5:
                        # Get confidence interval
                        lower, upper = forecaster.calculate_confidence_interval(future_values[i])
                        
                        # Calculate severity based on anomaly score
                        severity = AnomalySeverity.LOW
                        if score < -1.5:
                            severity = AnomalySeverity.CRITICAL
                        elif score < -1.0:
                            severity = AnomalySeverity.HIGH
                        elif score < -0.7:
                            severity = AnomalySeverity.MEDIUM
                        
                        # Determine most likely anomaly type
                        anomaly_type = self._predict_anomaly_type(
                            api, metric_type, future_values[i], values[-1], metric_type
                        )
                        
                        # Determine likely root cause if possible
                        root_cause = self._predict_root_cause(
                            api, metric_type, 
                            latest_features[0] if len(latest_features) > 0 else None
                        )
                        
                        # Create predicted anomaly
                        anomaly = {
                            "timestamp": future_timestamps[i] if i < len(future_timestamps) else time.time() + (i+1) * 3600,
                            "api": api,
                            "metric_type": metric_type,
                            "predicted_value": future_values[i],
                            "confidence_interval": [lower, upper],
                            "anomaly_score": float(score),
                            "severity": severity.value,
                            "anomaly_type": anomaly_type.value,
                            "prediction_horizon": prediction_horizon.value,
                            "is_predicted": True,
                            "root_cause": root_cause,
                            "description": f"Predicted {anomaly_type.value.replace('_', ' ')} for {api} {metric_type} in {i+1} hours"
                        }
                        
                        predicted_anomalies.append(anomaly)
            
            return predicted_anomalies
            
        except Exception as e:
            logger.error(f"Error predicting anomalies: {e}")
            return []
    
    def _predict_anomaly_type(
        self, 
        api: str, 
        metric_type: str, 
        predicted_value: float, 
        current_value: float,
        metric: str
    ) -> AnomalyType:
        """
        Predict anomaly type for a future value.
        
        Args:
            api: API name
            metric_type: Metric type
            predicted_value: Predicted value
            current_value: Current value
            metric: Metric name
            
        Returns:
            Predicted anomaly type
        """
        # Calculate percent change
        if current_value == 0:
            percent_change = 1.0 if predicted_value > 0 else -1.0
        else:
            percent_change = (predicted_value - current_value) / current_value
        
        # Determine anomaly type based on metric and direction of change
        if metric == "latency":
            if percent_change > 0.2:  # 20% increase
                return AnomalyType.LATENCY_SPIKE
            else:
                return AnomalyType.PATTERN_CHANGE
        elif metric == "throughput":
            if percent_change < -0.2:  # 20% decrease
                return AnomalyType.THROUGHPUT_DROP
            else:
                return AnomalyType.PATTERN_CHANGE
        elif metric == "reliability":
            if percent_change < -0.1:  # 10% decrease
                return AnomalyType.ERROR_RATE_INCREASE
            else:
                return AnomalyType.PATTERN_CHANGE
        elif metric == "cost":
            if percent_change > 0.3:  # 30% increase
                return AnomalyType.COST_SPIKE
            else:
                return AnomalyType.PATTERN_CHANGE
        else:
            # Check for oscillation patterns in recent history
            key = f"{api}:{metric_type}"
            if key in self.training_data and len(self.training_data[key]["values"]) >= 10:
                recent_values = self.training_data[key]["values"][-10:]
                diffs = np.diff(recent_values)
                sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
                
                if sign_changes >= 4:  # Many sign changes = oscillation
                    return AnomalyType.OSCILLATION
            
            # Check for trend breaks
            if key in self.forecasters:
                forecaster = self.forecasters[key]
                top_features = forecaster.get_top_features()
                
                # If trend features are important, might be a trend break
                trend_features = [name for name, _ in top_features if "slope" in name or "change" in name]
                if len(trend_features) >= 2:
                    return AnomalyType.TREND_BREAK
            
            return AnomalyType.PATTERN_CHANGE
    
    def _predict_root_cause(
        self, 
        api: str, 
        metric_type: str, 
        features: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """
        Predict root cause for a potential anomaly.
        
        Args:
            api: API name
            metric_type: Metric type
            features: Feature vector for the anomaly
            
        Returns:
            Dictionary of root cause probabilities
        """
        key = f"{api}:{metric_type}"
        
        if key not in self.root_cause_models or features is None:
            return {"unknown": 1.0}
        
        try:
            root_cause_model = self.root_cause_models[key]
            
            # Transform features with PCA
            pca = root_cause_model["pca"]
            reduced_features = pca.transform([features])[0]
            
            # Find similar anomalies in our database
            model_features = root_cause_model["features"]
            model_types = root_cause_model["types"]
            clusters = root_cause_model["clusters"]
            
            # Calculate distances to all known anomalies
            distances = []
            for i, feat in enumerate(model_features):
                reduced_feat = pca.transform([feat])[0]
                distance = np.sum((reduced_features - reduced_feat) ** 2)
                distances.append((distance, i))
            
            # Sort by distance
            distances.sort()
            
            # Get top 5 closest anomalies
            closest_anomalies = []
            for dist, idx in distances[:5]:
                closest_anomalies.append((model_types[idx], dist))
            
            # Count types and calculate probabilities
            type_counts = {}
            for anomaly_type, dist in closest_anomalies:
                if anomaly_type not in type_counts:
                    type_counts[anomaly_type] = 0
                type_counts[anomaly_type] += 1 / (dist + 0.01)  # Weight by inverse distance
            
            # Normalize to get probabilities
            total = sum(type_counts.values())
            if total > 0:
                root_causes = {t: c / total for t, c in type_counts.items()}
            else:
                root_causes = {"unknown": 1.0}
            
            return root_causes
            
        except Exception as e:
            logger.error(f"Error predicting root cause: {e}")
            return {"unknown": 1.0}
    
    def analyze_patterns(
        self, 
        api: str,
        metric_type: str,
        timestamps: List[float],
        values: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in the time series data.
        
        Args:
            api: API name
            metric_type: Metric type
            timestamps: List of timestamps
            values: List of values
            
        Returns:
            Dictionary of pattern analysis results
        """
        if not ML_AVAILABLE or len(values) < 24:
            return {"status": "insufficient_data"}
        
        try:
            # Detect trends
            # Use linear regression on recent data
            recent_n = min(168, len(values))  # Last week or all data
            recent_values = values[-recent_n:]
            recent_times = list(range(len(recent_values)))
            
            coeffs = np.polyfit(recent_times, recent_values, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Detect seasonality
            # Group by hour of day
            hour_values = {}
            for i, ts in enumerate(timestamps):
                if i >= len(values):
                    continue
                    
                hour = datetime.fromtimestamp(ts).hour
                if hour not in hour_values:
                    hour_values[hour] = []
                hour_values[hour].append(values[i])
            
            # Calculate hourly stats
            hourly_stats = {}
            has_hourly_pattern = False
            
            for hour, hour_vals in hour_values.items():
                if len(hour_vals) > 2:
                    mean = np.mean(hour_vals)
                    std = np.std(hour_vals)
                    hourly_stats[hour] = {"mean": mean, "std": std}
            
            # Check if there's clear hourly pattern
            if hourly_stats:
                means = [stats["mean"] for stats in hourly_stats.values()]
                min_mean = min(means)
                max_mean = max(means)
                
                # If max is significantly higher than min, we have a pattern
                if min_mean > 0 and max_mean / min_mean > 1.3:
                    has_hourly_pattern = True
            
            # Detect weekly pattern
            day_values = {}
            for i, ts in enumerate(timestamps):
                if i >= len(values):
                    continue
                    
                day = datetime.fromtimestamp(ts).weekday()
                if day not in day_values:
                    day_values[day] = []
                day_values[day].append(values[i])
            
            # Calculate daily stats
            daily_stats = {}
            has_daily_pattern = False
            
            for day, day_vals in day_values.items():
                if len(day_vals) > 2:
                    mean = np.mean(day_vals)
                    std = np.std(day_vals)
                    daily_stats[day] = {"mean": mean, "std": std}
            
            # Check if there's clear daily pattern
            if daily_stats and len(daily_stats) >= 5:  # Need at least 5 days of data
                means = [stats["mean"] for stats in daily_stats.values()]
                min_mean = min(means)
                max_mean = max(means)
                
                # If max is significantly higher than min, we have a pattern
                if min_mean > 0 and max_mean / min_mean > 1.2:
                    has_daily_pattern = True
            
            # Detect outliers
            z_scores = np.abs((values - np.mean(values)) / (np.std(values) if np.std(values) > 0 else 1))
            outliers = [i for i, z in enumerate(z_scores) if z > 3]
            
            # Forecast expected range for next 24 hours
            key = f"{api}:{metric_type}"
            future_range = {}
            
            if key in self.forecasters and self.forecasters[key].is_trained:
                forecaster = self.forecasters[key]
                
                # Predict next 24 hours
                future_values = forecaster.predict(
                    timestamps, values, 
                    steps_ahead=24
                )
                
                if future_values:
                    # Get confidence intervals
                    intervals = [
                        forecaster.calculate_confidence_interval(val)
                        for val in future_values
                    ]
                    
                    # Format for output
                    future_range = {
                        "predictions": future_values,
                        "confidence_intervals": intervals
                    }
            
            # Return analysis
            return {
                "status": "success",
                "trend": {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                },
                "seasonality": {
                    "hourly": {
                        "detected": has_hourly_pattern,
                        "stats": {str(h): {"mean": float(s["mean"]), "std": float(s["std"])} for h, s in hourly_stats.items()}
                    },
                    "daily": {
                        "detected": has_daily_pattern,
                        "stats": {str(d): {"mean": float(s["mean"]), "std": float(s["std"])} for d, s in daily_stats.items()}
                    }
                },
                "outliers": {
                    "count": len(outliers),
                    "indices": outliers
                },
                "forecast": future_range
            }
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_cost_optimization_recommendations(
        self, 
        api: str,
        cost_data: List[Dict[str, Any]],
        usage_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate cost optimization recommendations.
        
        Args:
            api: API name
            cost_data: List of cost data points
            usage_patterns: Usage pattern data
            
        Returns:
            List of recommendations
        """
        if not cost_data:
            return []
        
        recommendations = []
        
        try:
            # Extract costs and timestamps
            timestamps = [point.get("timestamp", 0) for point in cost_data]
            costs = [point.get("cost", 0) for point in cost_data]
            
            # Analyze cost trends
            if len(costs) > 10:
                # Look for consistent increase
                recent_costs = costs[-10:]
                if all(recent_costs[i] <= recent_costs[i+1] for i in range(len(recent_costs)-1)):
                    recommendations.append({
                        "type": "cost_trend",
                        "severity": "high",
                        "description": f"Costs for {api} are consistently increasing",
                        "recommendation": "Review recent usage patterns and API configuration"
                    })
                
                # Check for sudden spike
                avg_cost = np.mean(costs[:-1])
                latest_cost = costs[-1]
                
                if latest_cost > avg_cost * 1.5:
                    recommendations.append({
                        "type": "cost_spike",
                        "severity": "high",
                        "description": f"Latest cost for {api} is {latest_cost:.2f}, which is {(latest_cost/avg_cost - 1)*100:.1f}% above average",
                        "recommendation": "Investigate recent workload changes or potential misconfigurations"
                    })
            
            # Check usage patterns
            if usage_patterns:
                # Look for opportunities to optimize based on time of day
                if "hourly" in usage_patterns and usage_patterns["hourly"].get("detected", False):
                    peak_hours = []
                    hourly_stats = usage_patterns["hourly"].get("stats", {})
                    
                    for hour, stats in hourly_stats.items():
                        mean = stats.get("mean", 0)
                        if mean > np.mean([s.get("mean", 0) for s in hourly_stats.values()]) * 1.2:
                            peak_hours.append(int(hour))
                    
                    if peak_hours:
                        peak_hours_str = ", ".join(f"{h}:00" for h in sorted(peak_hours))
                        recommendations.append({
                            "type": "usage_pattern",
                            "severity": "medium",
                            "description": f"Peak usage hours for {api}: {peak_hours_str}",
                            "recommendation": "Consider workload shifting to off-peak hours or optimizing requests during peak times"
                        })
                
                # Check if weekend usage is significantly different
                if "daily" in usage_patterns and usage_patterns["daily"].get("detected", False):
                    daily_stats = usage_patterns["daily"].get("stats", {})
                    
                    if "5" in daily_stats and "6" in daily_stats:  # Saturday and Sunday
                        weekday_means = [daily_stats.get(str(d), {}).get("mean", 0) for d in range(5)]
                        weekend_means = [daily_stats.get(str(d), {}).get("mean", 0) for d in [5, 6]]
                        
                        weekday_avg = np.mean(weekday_means) if weekday_means else 0
                        weekend_avg = np.mean(weekend_means) if weekend_means else 0
                        
                        if weekend_avg < weekday_avg * 0.5 and weekday_avg > 0:
                            recommendations.append({
                                "type": "usage_pattern",
                                "severity": "low",
                                "description": f"Weekend usage for {api} is {weekend_avg/weekday_avg*100:.1f}% of weekday usage",
                                "recommendation": "Consider batch processing or training workloads during weekends to optimize costs"
                            })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating cost recommendations: {e}")
            return []
    
    def save_models(self, directory: str) -> bool:
        """
        Save all models to directory.
        
        Args:
            directory: Directory to save models
            
        Returns:
            True if successful
        """
        if not ML_AVAILABLE:
            return False
        
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save anomaly detection models
            for key, model in self.models.items():
                model_path = os.path.join(directory, f"anomaly_model_{key.replace(':', '_')}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save forecasting models
            for key, forecaster in self.forecasters.items():
                if forecaster.is_trained:
                    forecaster_path = os.path.join(directory, f"forecaster_{key.replace(':', '_')}.pkl")
                    forecaster.save_model(forecaster_path)
            
            # Save root cause models
            if self.root_cause_models:
                rc_path = os.path.join(directory, "root_cause_models.pkl")
                with open(rc_path, 'wb') as f:
                    pickle.dump(self.root_cause_models, f)
            
            # Save anomaly patterns
            if self.anomaly_patterns:
                patterns_path = os.path.join(directory, "anomaly_patterns.pkl")
                with open(patterns_path, 'wb') as f:
                    pickle.dump(dict(self.anomaly_patterns), f)
            
            # Save training timestamps
            meta_path = os.path.join(directory, "predictor_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump({
                    "last_training_time": self.last_training_time,
                    "prediction_window": self.prediction_window,
                    "contamination": self.contamination,
                    "min_train_size": self.min_train_size
                }, f, indent=2)
            
            logger.info(f"Saved anomaly prediction models to {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    @classmethod
    def load_models(cls, directory: str) -> Optional['AnomalyPredictor']:
        """
        Load models from directory.
        
        Args:
            directory: Directory to load models from
            
        Returns:
            Loaded AnomalyPredictor or None if loading failed
        """
        if not ML_AVAILABLE or not os.path.exists(directory):
            return None
        
        try:
            # Load metadata
            meta_path = os.path.join(directory, "predictor_metadata.json")
            if not os.path.exists(meta_path):
                logger.warning(f"Metadata file not found: {meta_path}")
                return None
            
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Create predictor with saved parameters
            predictor = cls(
                prediction_window=metadata.get("prediction_window", 12),
                contamination=metadata.get("contamination", 0.05),
                min_train_size=metadata.get("min_train_size", 100)
            )
            
            # Load timestamps
            predictor.last_training_time = metadata.get("last_training_time", {})
            
            # Load anomaly detection models
            for file in os.listdir(directory):
                if file.startswith("anomaly_model_") and file.endswith(".pkl"):
                    key = file[13:-4].replace('_', ':')
                    model_path = os.path.join(directory, file)
                    
                    with open(model_path, 'rb') as f:
                        predictor.models[key] = pickle.load(f)
            
            # Load forecasting models
            for file in os.listdir(directory):
                if file.startswith("forecaster_") and file.endswith(".pkl"):
                    key = file[11:-4].replace('_', ':')
                    forecaster_path = os.path.join(directory, file)
                    
                    forecaster = TimeSeriesPredictor.load_model(forecaster_path)
                    if forecaster:
                        predictor.forecasters[key] = forecaster
            
            # Load root cause models
            rc_path = os.path.join(directory, "root_cause_models.pkl")
            if os.path.exists(rc_path):
                with open(rc_path, 'rb') as f:
                    predictor.root_cause_models = pickle.load(f)
            
            # Load anomaly patterns
            patterns_path = os.path.join(directory, "anomaly_patterns.pkl")
            if os.path.exists(patterns_path):
                with open(patterns_path, 'rb') as f:
                    predictor.anomaly_patterns = defaultdict(list, pickle.load(f))
            
            logger.info(f"Loaded anomaly prediction models from {directory}")
            return predictor
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return None


def generate_prediction_summary(
    api: str,
    metric_type: str,
    predictions: List[float],
    confidence_intervals: List[Tuple[float, float]],
    predicted_anomalies: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a summary of predictions and anomalies.
    
    Args:
        api: API name
        metric_type: Metric type
        predictions: List of predicted values
        confidence_intervals: List of confidence intervals
        predicted_anomalies: List of predicted anomalies
        
    Returns:
        Dictionary with prediction summary
    """
    # Calculate timing information
    now = time.time()
    hours_ahead = len(predictions)
    
    # Format predictions for hours
    hourly_predictions = []
    for i, (pred, interval) in enumerate(zip(predictions, confidence_intervals)):
        # Calculate timestamp for this prediction
        future_ts = now + (i + 1) * 3600  # 1 hour increments
        future_time = datetime.fromtimestamp(future_ts).strftime('%Y-%m-%d %H:%M:%S')
        
        # Find any anomalies for this hour
        hour_anomalies = [a for a in predicted_anomalies if abs(a.get("timestamp", 0) - future_ts) < 1800]
        
        # Determine status
        status = "normal"
        if hour_anomalies:
            max_severity = max([a.get("severity", "low") for a in hour_anomalies], 
                               key=lambda s: ["low", "medium", "high", "critical"].index(s))
            status = max_severity
        
        # Add prediction
        hourly_predictions.append({
            "time": future_time,
            "hour": i + 1,
            "value": pred,
            "lower_bound": interval[0],
            "upper_bound": interval[1],
            "status": status,
            "anomalies": len(hour_anomalies)
        })
    
    # Create summary
    has_anomalies = len(predicted_anomalies) > 0
    
    # Categorize anomalies by severity if any exist
    severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    if has_anomalies:
        for anomaly in predicted_anomalies:
            severity = anomaly.get("severity", "low")
            severity_counts[severity] += 1
    
    # Get time of first critical or high anomaly
    first_severe_anomaly_time = None
    if has_anomalies:
        severe_anomalies = [a for a in predicted_anomalies 
                            if a.get("severity") in ("high", "critical")]
        if severe_anomalies:
            first_severe = min(severe_anomalies, key=lambda a: a.get("timestamp", float('inf')))
            first_severe_anomaly_time = datetime.fromtimestamp(
                first_severe.get("timestamp", now)
            ).strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        "api": api,
        "metric": metric_type,
        "forecast_hours": hours_ahead,
        "generated_at": datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'),
        "predictions": hourly_predictions,
        "anomalies": {
            "detected": has_anomalies,
            "count": len(predicted_anomalies),
            "severity_counts": severity_counts,
            "first_severe": first_severe_anomaly_time
        },
        "trend": {
            "direction": "improving" if predictions and len(predictions) > 1 and 
                         ((metric_type == "latency" and predictions[-1] < predictions[0]) or
                          (metric_type == "throughput" and predictions[-1] > predictions[0]) or
                          (metric_type == "reliability" and predictions[-1] > predictions[0])) 
                      else "degrading",
            "percent_change": abs(predictions[-1] / predictions[0] - 1) * 100 if predictions and len(predictions) > 1 and predictions[0] != 0 else 0
        }
    }


def main():
    """Example usage of the predictive analytics module."""
    if not ML_AVAILABLE:
        print("Machine learning dependencies not available. Please install scikit-learn.")
        return
    
    print("API Predictive Analytics Example")
    print("================================")
    
    # Create synthetic data
    timestamps = []
    latency_values = []
    
    # Generate 7 days of hourly data with daily pattern
    now = time.time()
    day_seconds = 24 * 60 * 60
    
    for day in range(7):
        for hour in range(24):
            ts = now - (7 - day) * day_seconds + hour * 3600
            timestamps.append(ts)
            
            # Base latency
            latency = 1.0
            
            # Add time-of-day pattern (higher during business hours)
            if 9 <= hour < 17:
                latency *= 1.3
            
            # Add day-of-week pattern (higher on weekdays)
            weekday = datetime.fromtimestamp(ts).weekday()
            if weekday < 5:  # Monday-Friday
                latency *= 1.2
            
            # Add trend (gradually improving)
            latency *= (1.0 - day * 0.02)
            
            # Add noise
            latency *= (0.9 + 0.2 * np.random.random())
            
            # Add occasional spikes
            if np.random.random() < 0.05:
                latency *= 2.0
            
            latency_values.append(latency)
    
    # Create predictor
    predictor = AnomalyPredictor(
        prediction_window=12,
        contamination=0.05
    )
    
    # Record data
    api = "example_api"
    metric_type = "latency"
    predictor.record_data(api, metric_type, timestamps, latency_values)
    
    # Train models
    print("Training prediction models...")
    predictor.train(api, metric_type)
    
    # Analyze patterns
    print("\nAnalyzing patterns...")
    patterns = predictor.analyze_patterns(api, metric_type, timestamps, latency_values)
    print(f"Pattern analysis results:")
    print(f"  Trend: {patterns['trend']['direction']}")
    print(f"  Hourly pattern detected: {patterns['seasonality']['hourly']['detected']}")
    print(f"  Daily pattern detected: {patterns['seasonality']['daily']['detected']}")
    
    # Predict anomalies
    print("\nPredicting potential anomalies...")
    anomalies = predictor.predict_anomalies(
        api, metric_type, timestamps, latency_values,
        prediction_horizon=PredictionHorizon.MEDIUM_TERM
    )
    
    if anomalies:
        print(f"Predicted {len(anomalies)} potential anomalies:")
        for i, anomaly in enumerate(anomalies[:3]):  # Show top 3
            pred_time = datetime.fromtimestamp(anomaly["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {i+1}. {anomaly['description']} at {pred_time} (severity: {anomaly['severity']})")
    else:
        print("No anomalies predicted")
    
    # Generate cost recommendations
    print("\nGenerating cost optimization recommendations...")
    cost_data = [
        {"timestamp": ts, "cost": latency_values[i] * 0.02} 
        for i, ts in enumerate(timestamps)
    ]
    recommendations = predictor.get_cost_optimization_recommendations(
        api, cost_data, patterns["seasonality"]
    )
    
    if recommendations:
        print(f"Generated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec['description']} - {rec['recommendation']}")
    else:
        print("No cost optimization recommendations generated")
    
    # Save models
    print("\nSaving models...")
    predictor.save_models("example_models")
    
    print("\nDone!")


if __name__ == "__main__":
    # Import additional dependencies here to avoid module-level import errors
    try:
        import numpy as np
        import random
    except ImportError:
        print("Error: This module requires numpy to be installed.")
        print("Install it with: pip install numpy scikit-learn")
        sys.exit(1)
        
    main()