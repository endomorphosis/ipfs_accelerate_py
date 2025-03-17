#!/usr/bin/env python3
"""
Adaptive Circuit Breaker with Machine Learning Optimization

This module extends the Circuit Breaker pattern with machine learning capabilities
to dynamically optimize thresholds based on historical performance data. It provides
an adaptive system that learns from past failures and recoveries to fine-tune
circuit breaker parameters for optimal fault tolerance.

Key features:
1. ML-based threshold optimization
2. Performance metrics collection and analysis
3. Predictive circuit breaking based on early warning signals
4. Periodic model retraining based on new data
5. Hardware-specific optimization strategies
"""

import os
import sys
import time
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable, Union
from pathlib import Path
import pickle
import warnings

# Import the base CircuitBreaker
try:
    from circuit_breaker import CircuitBreaker, CircuitState
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    logging.warning("Base CircuitBreaker not available, some functionality will be limited")
    
    # Define a fallback CircuitState for type checking
    from enum import Enum
    class CircuitState(str, Enum):
        """States for the circuit breaker"""
        CLOSED = "closed"  # Normal operation, requests pass through
        OPEN = "open"      # Failure detected, requests are blocked
        HALF_OPEN = "half_open"  # Testing if system has recovered

# Conditional imports for ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("adaptive_circuit_breaker")

class AdaptiveCircuitBreaker:
    """
    Adaptive Circuit Breaker that uses machine learning to optimize thresholds
    based on historical performance data.
    """
    
    def __init__(self, 
                 name: str = "adaptive",
                 base_failure_threshold: int = 3,
                 base_recovery_timeout: float = 30,
                 base_half_open_timeout: float = 5,
                 optimization_enabled: bool = True,
                 prediction_enabled: bool = True,
                 db_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 metrics_path: Optional[str] = None,
                 learning_rate: float = 0.1,
                 retraining_interval_hours: float = 24,
                 min_data_points: int = 20,
                 hardware_specific: bool = False,
                 hardware_type: Optional[str] = None):
        """
        Initialize the Adaptive Circuit Breaker.
        
        Args:
            name: Name for this circuit breaker
            base_failure_threshold: Initial/fallback threshold for failures
            base_recovery_timeout: Initial/fallback timeout for recovery
            base_half_open_timeout: Initial/fallback timeout for half-open state
            optimization_enabled: Enable ML-based threshold optimization
            prediction_enabled: Enable predictive circuit breaking
            db_path: Path to DuckDB database file (optional)
            model_path: Path to save/load ML models
            metrics_path: Path to save performance metrics
            learning_rate: Rate at which to adjust parameters (0.0-1.0)
            retraining_interval_hours: Hours between model retraining
            min_data_points: Minimum data points needed for optimization
            hardware_specific: Enable hardware-specific optimizations
            hardware_type: Type of hardware for specific optimizations
        """
        # Store base configuration
        self.name = name
        self.base_failure_threshold = base_failure_threshold
        self.base_recovery_timeout = base_recovery_timeout
        self.base_half_open_timeout = base_half_open_timeout
        
        # Feature flags
        self.optimization_enabled = optimization_enabled and SKLEARN_AVAILABLE
        self.prediction_enabled = prediction_enabled and SKLEARN_AVAILABLE
        self.hardware_specific = hardware_specific
        self.hardware_type = hardware_type
        
        # Optimization parameters
        self.learning_rate = learning_rate
        self.retraining_interval_hours = retraining_interval_hours
        self.min_data_points = min_data_points
        self.last_model_training = datetime.min
        
        # Paths for persistence
        self.db_path = db_path
        self.model_path = model_path or "models/circuit_breaker"
        self.metrics_path = metrics_path or "metrics/circuit_breaker"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        
        # DB connection (optional)
        self.db_conn = None
        if self.db_path and DUCKDB_AVAILABLE:
            try:
                self.db_conn = duckdb.connect(self.db_path)
                self._initialize_db_schema()
                logger.info(f"Connected to database: {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to connect to database: {str(e)}")
                self.db_conn = None
        
        # Performance metrics storage
        self.metrics = []
        self.recent_failures = []
        self.recent_recoveries = []
        
        # Optimized parameters
        self.current_failure_threshold = base_failure_threshold
        self.current_recovery_timeout = base_recovery_timeout
        self.current_half_open_timeout = base_half_open_timeout
        
        # ML models
        self.threshold_model = None
        self.recovery_timeout_model = None
        self.half_open_timeout_model = None
        self.prediction_model = None
        
        # Feature importances from models
        self.feature_importances = {}
        
        # Load existing models if available
        self._load_models()
        
        # Create the underlying circuit breaker
        if CIRCUIT_BREAKER_AVAILABLE:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=self.current_failure_threshold,
                recovery_timeout=self.current_recovery_timeout,
                half_open_timeout=self.current_half_open_timeout,
                name=f"{name}_base"
            )
            logger.info(f"Created adaptive circuit breaker '{name}' with optimization={optimization_enabled}")
        else:
            logger.error("Base CircuitBreaker not available, functionality will be limited")
            self.circuit_breaker = None
    
    async def execute(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Execute function with adaptive circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if not self.circuit_breaker:
            raise RuntimeError("Base CircuitBreaker not available")
        
        start_time = time.time()
        start_state = self.circuit_breaker.state
        execution_result = None
        error = None
        
        # Check for early warning signals if prediction is enabled
        if self.prediction_enabled and start_state == CircuitState.CLOSED:
            should_preemptively_open = self._check_early_warning_signals()
            if should_preemptively_open:
                logger.warning(f"Circuit {self.name} preemptively opened based on early warning signals")
                self.circuit_breaker.state = CircuitState.OPEN
                self.circuit_breaker.last_failure_time = time.time()
                
                # Record predictive action
                self._record_metric({
                    "time": time.time(),
                    "action": "predictive_open",
                    "failure_threshold": self.current_failure_threshold,
                    "recovery_timeout": self.current_recovery_timeout,
                    "half_open_timeout": self.current_half_open_timeout,
                    "current_state": str(self.circuit_breaker.state),
                    "failure_count": self.circuit_breaker.failure_count
                })
                
                # Jump to the exception path
                raise Exception(f"Circuit {self.name} preemptively opened to prevent cascading failures")
        
        try:
            # Execute the function using the base circuit breaker
            execution_result = await self.circuit_breaker.execute(func, *args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(start_state, execution_time)
            
            return execution_result
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            error = str(e)
            self._record_failure(start_state, execution_time, error)
            
            # Re-raise the exception
            raise
        
        finally:
            # Check if it's time to retrain models
            await self._check_model_retraining()
    
    def _record_success(self, start_state: CircuitState, execution_time: float) -> None:
        """
        Record a successful operation and gather metrics.
        
        Args:
            start_state: Circuit state before execution
            execution_time: Time taken for execution
        """
        # Record the recovery if we were in half-open state
        if start_state == CircuitState.HALF_OPEN:
            recovery_data = {
                "time": time.time(),
                "recovery_time": execution_time,
                "previous_state": str(start_state),
                "failure_threshold": self.current_failure_threshold,
                "recovery_timeout": self.current_recovery_timeout,
                "half_open_timeout": self.current_half_open_timeout,
                "hardware_type": self.hardware_type
            }
            
            self.recent_recoveries.append(recovery_data)
            
            # Save recovery to database or file
            self._record_metric({
                **recovery_data,
                "event_type": "recovery",
                "success": True
            })
            
            # If we have enough data, optimize parameters
            if (len(self.recent_recoveries) > self.min_data_points and 
                self.optimization_enabled and SKLEARN_AVAILABLE):
                self._optimize_parameters()
    
    def _record_failure(self, start_state: CircuitState, execution_time: float, error: str) -> None:
        """
        Record a failed operation and gather metrics.
        
        Args:
            start_state: Circuit state before execution
            execution_time: Time taken before failure
            error: Error message from the exception
        """
        # Store failure data
        failure_data = {
            "time": time.time(),
            "failure_time": execution_time,
            "previous_state": str(start_state),
            "current_state": str(self.circuit_breaker.state if self.circuit_breaker else "unknown"),
            "error": error,
            "failure_count": getattr(self.circuit_breaker, "failure_count", 0),
            "failure_threshold": self.current_failure_threshold,
            "recovery_timeout": self.current_recovery_timeout,
            "half_open_timeout": self.current_half_open_timeout,
            "hardware_type": self.hardware_type
        }
        
        self.recent_failures.append(failure_data)
        
        # Save failure to database or file
        self._record_metric({
            **failure_data,
            "event_type": "failure" 
        })
        
        # If we have enough data, optimize parameters
        if (len(self.recent_failures) > self.min_data_points and 
            self.optimization_enabled and SKLEARN_AVAILABLE):
            self._optimize_parameters()
    
    def _record_metric(self, metric_data: Dict[str, Any]) -> None:
        """
        Record a metric to the metrics storage.
        
        Args:
            metric_data: The metric data to record
        """
        # Add to in-memory metrics
        self.metrics.append(metric_data)
        
        # Store in database if available
        if self.db_conn is not None:
            try:
                # Add metric_id and timestamp if not present
                if "metric_id" not in metric_data:
                    metric_data["metric_id"] = len(self.metrics)
                if "timestamp" not in metric_data:
                    metric_data["timestamp"] = datetime.now().isoformat()
                
                # Convert to DataFrame and write to DB
                metric_df = pd.DataFrame([metric_data])
                self.db_conn.execute("""
                    INSERT INTO circuit_breaker_metrics 
                    SELECT * FROM metric_df
                """)
            except Exception as e:
                logger.error(f"Failed to write metric to database: {str(e)}")
        
        # Store in metrics file if memory metrics exceed threshold
        if len(self.metrics) > 1000:
            self._save_metrics_to_file()
            self.metrics = []  # Clear in-memory metrics
    
    def _save_metrics_to_file(self) -> None:
        """Save accumulated metrics to a file."""
        if not self.metrics:
            return
            
        try:
            # Create metrics filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = f"{self.metrics_path}/{self.name}_metrics_{timestamp}.json"
            
            # Save metrics to file
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
                
            logger.info(f"Saved {len(self.metrics)} metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics to file: {str(e)}")
    
    def _initialize_db_schema(self) -> None:
        """Initialize the database schema for circuit breaker metrics."""
        if self.db_conn is None:
            return
            
        try:
            # Create circuit breaker metrics table if it doesn't exist
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS circuit_breaker_metrics (
                    metric_id INTEGER,
                    timestamp VARCHAR,
                    time FLOAT,
                    event_type VARCHAR,
                    action VARCHAR,
                    previous_state VARCHAR,
                    current_state VARCHAR,
                    failure_count INTEGER,
                    failure_threshold INTEGER,
                    recovery_timeout FLOAT,
                    half_open_timeout FLOAT,
                    hardware_type VARCHAR,
                    failure_time FLOAT,
                    recovery_time FLOAT,
                    error VARCHAR,
                    success BOOLEAN
                );
                
                CREATE TABLE IF NOT EXISTS circuit_breaker_models (
                    model_id INTEGER PRIMARY KEY,
                    model_name VARCHAR,
                    circuit_breaker_name VARCHAR,
                    training_time VARCHAR,
                    parameters JSON,
                    feature_importances JSON,
                    performance_metrics JSON,
                    model_blob BLOB
                );
            """)
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {str(e)}")
    
    def _load_models(self) -> None:
        """Load ML models from disk if available."""
        if not SKLEARN_AVAILABLE:
            return
            
        models_to_load = {
            "threshold": {"path": f"{self.model_path}_threshold.pkl", "destination": "threshold_model"},
            "recovery": {"path": f"{self.model_path}_recovery.pkl", "destination": "recovery_timeout_model"},
            "half_open": {"path": f"{self.model_path}_half_open.pkl", "destination": "half_open_timeout_model"},
            "prediction": {"path": f"{self.model_path}_prediction.pkl", "destination": "prediction_model"}
        }
        
        for model_name, model_info in models_to_load.items():
            path = model_info["path"]
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        model_data = pickle.load(f)
                        setattr(self, model_info["destination"], model_data.get("model"))
                        
                        # Also load feature importances if available
                        if "feature_importances" in model_data:
                            self.feature_importances[model_name] = model_data["feature_importances"]
                            
                    logger.info(f"Loaded {model_name} model from {path}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name} model: {str(e)}")
    
    def _save_models(self) -> None:
        """Save ML models to disk."""
        if not SKLEARN_AVAILABLE:
            return
            
        models_to_save = {
            "threshold": {"model": self.threshold_model, "path": f"{self.model_path}_threshold.pkl"},
            "recovery": {"model": self.recovery_timeout_model, "path": f"{self.model_path}_recovery.pkl"},
            "half_open": {"model": self.half_open_timeout_model, "path": f"{self.model_path}_half_open.pkl"},
            "prediction": {"model": self.prediction_model, "path": f"{self.model_path}_prediction.pkl"}
        }
        
        for model_name, model_info in models_to_save.items():
            model = model_info["model"]
            path = model_info["path"]
            
            if model is not None:
                try:
                    model_data = {
                        "model": model,
                        "training_time": datetime.now().isoformat(),
                        "feature_importances": self.feature_importances.get(model_name, {})
                    }
                    
                    with open(path, 'wb') as f:
                        pickle.dump(model_data, f)
                        
                    logger.info(f"Saved {model_name} model to {path}")
                except Exception as e:
                    logger.error(f"Failed to save {model_name} model: {str(e)}")
    
    def _optimize_parameters(self) -> None:
        """
        Optimize circuit breaker parameters using ML models.
        
        This method trains models to predict optimal parameter values based on
        historical failure and recovery data, then applies the learned values.
        """
        if not SKLEARN_AVAILABLE or not self.optimization_enabled:
            return
            
        # Check if we have enough data
        if len(self.recent_failures) < self.min_data_points:
            logger.info(f"Not enough failure data for optimization ({len(self.recent_failures)}/{self.min_data_points})")
            return
            
        try:
            # Prepare failure data
            failure_df = pd.DataFrame(self.recent_failures)
            
            # Prepare recovery data (might be empty if no successful recoveries yet)
            recovery_df = pd.DataFrame(self.recent_recoveries) if self.recent_recoveries else None
            
            # Optimize failure threshold
            new_threshold = self._optimize_failure_threshold(failure_df)
            
            # Optimize recovery timeout
            new_recovery_timeout = self._optimize_recovery_timeout(failure_df, recovery_df)
            
            # Optimize half-open timeout
            new_half_open_timeout = self._optimize_half_open_timeout(failure_df, recovery_df)
            
            # Apply the optimized parameters with learning rate
            # Learning rate controls how quickly we adjust parameters
            old_threshold = self.current_failure_threshold
            old_recovery = self.current_recovery_timeout
            old_half_open = self.current_half_open_timeout
            
            self.current_failure_threshold = int(round(
                old_threshold * (1 - self.learning_rate) + new_threshold * self.learning_rate
            ))
            self.current_recovery_timeout = float(
                old_recovery * (1 - self.learning_rate) + new_recovery_timeout * self.learning_rate
            )
            self.current_half_open_timeout = float(
                old_half_open * (1 - self.learning_rate) + new_half_open_timeout * self.learning_rate
            )
            
            # Apply constraints to ensure sane values
            self.current_failure_threshold = max(1, self.current_failure_threshold)
            self.current_recovery_timeout = max(1.0, self.current_recovery_timeout)
            self.current_half_open_timeout = max(0.5, min(self.current_half_open_timeout, 
                                                        self.current_recovery_timeout / 2))
            
            # Update the base circuit breaker with new parameters
            if self.circuit_breaker:
                self.circuit_breaker.failure_threshold = self.current_failure_threshold
                self.circuit_breaker.recovery_timeout = self.current_recovery_timeout
                self.circuit_breaker.half_open_timeout = self.current_half_open_timeout
            
            # Record the optimization
            self._record_metric({
                "time": time.time(),
                "event_type": "optimization",
                "previous_threshold": old_threshold,
                "new_threshold": self.current_failure_threshold,
                "previous_recovery": old_recovery,
                "new_recovery": self.current_recovery_timeout,
                "previous_half_open": old_half_open,
                "new_half_open": self.current_half_open_timeout,
                "hardware_type": self.hardware_type
            })
            
            logger.info(f"Optimized circuit breaker parameters: threshold={self.current_failure_threshold}, "
                       f"recovery_timeout={self.current_recovery_timeout:.2f}s, "
                       f"half_open_timeout={self.current_half_open_timeout:.2f}s")
                       
            # Retrain prediction model for early warnings
            if self.prediction_enabled:
                self._train_prediction_model(failure_df)
                
            # Save the models for future use
            self._save_models()
            
            # Update last model training time
            self.last_model_training = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to optimize parameters: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _optimize_failure_threshold(self, failure_df: pd.DataFrame) -> int:
        """
        Optimize the failure threshold using machine learning.
        
        Args:
            failure_df: DataFrame with failure data
            
        Returns:
            Optimized failure threshold value
        """
        # Default to base threshold if insufficient data
        if len(failure_df) < self.min_data_points:
            return self.base_failure_threshold
            
        try:
            # Extract features for training
            features = self._extract_failure_features(failure_df)
            
            # Target variable is failure count that leads to successful recovery
            # Higher values mean more resilience needed
            successful_failures = failure_df[failure_df['current_state'] == 'open'].copy()
            if len(successful_failures) < 5:  # Need enough successful state transitions
                return self.base_failure_threshold
                
            # The target is the optimal failure threshold
            target = successful_failures['failure_count'].values
            
            # Train a regression model
            model = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Split data if enough samples
            if len(features) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
                
                # Evaluate model
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                logger.info(f"Failure threshold model - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
                
                # Store feature importances
                if hasattr(model, 'feature_importances_'):
                    self.feature_importances['threshold'] = dict(
                        zip(features.columns, model.feature_importances_)
                    )
            else:
                # Not enough data for splitting, use all data
                model.fit(features, target)
            
            # Store the model
            self.threshold_model = model
            
            # Predict optimal threshold for current conditions
            current_features = self._extract_current_condition_features()
            predicted_threshold = model.predict([current_features])[0]
            
            # Round to nearest integer and apply bounds
            return max(1, min(10, int(round(predicted_threshold))))
            
        except Exception as e:
            logger.error(f"Failed to optimize failure threshold: {str(e)}")
            return self.base_failure_threshold
    
    def _optimize_recovery_timeout(self, failure_df: pd.DataFrame, 
                                  recovery_df: Optional[pd.DataFrame]) -> float:
        """
        Optimize the recovery timeout using machine learning.
        
        Args:
            failure_df: DataFrame with failure data
            recovery_df: DataFrame with recovery data (can be None)
            
        Returns:
            Optimized recovery timeout value
        """
        # Default to base timeout if insufficient data
        if recovery_df is None or len(recovery_df) < 5:
            return self.base_recovery_timeout
            
        try:
            # Extract features for training
            features = self._extract_recovery_features(failure_df, recovery_df)
            
            # Target is the optimal recovery time
            # We use the actual recovery times as targets
            target = recovery_df['recovery_time'].values
            
            # Train a regression model
            model = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Split data if enough samples
            if len(features) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
                
                # Evaluate model
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                logger.info(f"Recovery timeout model - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
                
                # Store feature importances
                if hasattr(model, 'feature_importances_'):
                    self.feature_importances['recovery'] = dict(
                        zip(features.columns, model.feature_importances_)
                    )
            else:
                # Not enough data for splitting, use all data
                model.fit(features, target)
            
            # Store the model
            self.recovery_timeout_model = model
            
            # Predict optimal timeout for current conditions
            current_features = self._extract_current_condition_features()
            predicted_timeout = model.predict([current_features])[0]
            
            # Apply a safety margin - recovery timeout should be higher than actual recovery time
            recovery_timeout = predicted_timeout * 2.0  # double the expected recovery time
            
            # Apply bounds
            return max(1.0, min(60.0, recovery_timeout))
            
        except Exception as e:
            logger.error(f"Failed to optimize recovery timeout: {str(e)}")
            return self.base_recovery_timeout
    
    def _optimize_half_open_timeout(self, failure_df: pd.DataFrame,
                                   recovery_df: Optional[pd.DataFrame]) -> float:
        """
        Optimize the half-open timeout using machine learning.
        
        Args:
            failure_df: DataFrame with failure data
            recovery_df: DataFrame with recovery data (can be None)
            
        Returns:
            Optimized half-open timeout value
        """
        # Default to base timeout if insufficient data
        if recovery_df is None or len(recovery_df) < 5:
            return self.base_half_open_timeout
            
        try:
            # Extract features for training
            features = self._extract_recovery_features(failure_df, recovery_df)
            
            # Target is the optimal half-open timeout
            # We use recovery times / 3 as a heuristic (should be less than recovery timeout)
            target = recovery_df['recovery_time'].values / 3.0
            
            # Train a regression model
            model = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Split data if enough samples
            if len(features) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
                
                # Evaluate model
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                logger.info(f"Half-open timeout model - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
                
                # Store feature importances
                if hasattr(model, 'feature_importances_'):
                    self.feature_importances['half_open'] = dict(
                        zip(features.columns, model.feature_importances_)
                    )
            else:
                # Not enough data for splitting, use all data
                model.fit(features, target)
            
            # Store the model
            self.half_open_timeout_model = model
            
            # Predict optimal timeout for current conditions
            current_features = self._extract_current_condition_features()
            predicted_timeout = model.predict([current_features])[0]
            
            # Apply bounds and ensure it's less than recovery timeout
            half_open_timeout = min(predicted_timeout, self.current_recovery_timeout / 2)
            return max(0.5, min(30.0, half_open_timeout))
            
        except Exception as e:
            logger.error(f"Failed to optimize half-open timeout: {str(e)}")
            return self.base_half_open_timeout
    
    def _train_prediction_model(self, failure_df: pd.DataFrame) -> None:
        """
        Train a model to predict circuit breaking needs based on early warning signals.
        
        Args:
            failure_df: DataFrame with failure data
        """
        if len(failure_df) < self.min_data_points:
            return
            
        try:
            # Extract features for predicting failures
            features = self._extract_failure_prediction_features(failure_df)
            
            # Target is binary: 1 if a failure led to circuit opening, 0 otherwise
            target = (failure_df['current_state'] == 'open').astype(int).values
            
            # Ensure we have both classes
            if len(np.unique(target)) < 2:
                logger.info("Not enough diverse data to train prediction model (need both success and failure cases)")
                return
                
            # Train a binary classification model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=42,
                class_weight='balanced'  # Handle potential class imbalance
            )
            
            # Split data if enough samples
            if len(features) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42, stratify=target
                )
                model.fit(X_train, y_train)
                
                # Evaluate model
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                logger.info(f"Prediction model - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                           f"Recall: {recall:.3f}, F1: {f1:.3f}")
                
                # Store feature importances
                if hasattr(model, 'feature_importances_'):
                    self.feature_importances['prediction'] = dict(
                        zip(features.columns, model.feature_importances_)
                    )
            else:
                # Not enough data for splitting, use all data
                model.fit(features, target)
            
            # Store the model
            self.prediction_model = model
            
        except Exception as e:
            logger.error(f"Failed to train prediction model: {str(e)}")
    
    def _extract_failure_features(self, failure_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from failure data for threshold optimization.
        
        Args:
            failure_df: DataFrame with failure data
            
        Returns:
            DataFrame with extracted features
        """
        # Create basic features
        features = pd.DataFrame()
        
        # Time-based features (time of day, day of week)
        if 'time' in failure_df.columns:
            failure_df['datetime'] = pd.to_datetime(failure_df['time'], unit='s')
            features['hour'] = failure_df['datetime'].dt.hour
            features['day_of_week'] = failure_df['datetime'].dt.dayofweek
            
        # Error type features (extract from error strings)
        if 'error' in failure_df.columns:
            # Count occurrences of different error types
            error_types = [
                'timeout', 'connection', 'resource', 'memory', 'gpu', 
                'crash', 'network', 'internal', 'api'
            ]
            
            for error_type in error_types:
                features[f'error_{error_type}'] = failure_df['error'].str.contains(
                    error_type, case=False).astype(int)
                
        # Failure timing features
        if 'failure_time' in failure_df.columns:
            features['failure_time'] = failure_df['failure_time']
            features['failure_time_log'] = np.log1p(failure_df['failure_time'])
            
        # Previous state features
        if 'previous_state' in failure_df.columns:
            # One-hot encode previous state
            for state in ['closed', 'open', 'half_open']:
                features[f'prev_state_{state}'] = (failure_df['previous_state'] == state).astype(int)
                
        # Hardware type features (if hardware-specific is enabled)
        if self.hardware_specific and 'hardware_type' in failure_df.columns:
            # One-hot encode hardware type
            hardware_types = failure_df['hardware_type'].unique()
            for hardware in hardware_types:
                if hardware and not pd.isna(hardware):
                    features[f'hardware_{hardware}'] = (failure_df['hardware_type'] == hardware).astype(int)
        
        # Add simplified numeric features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Drop original time features
        features = features.drop(['hour', 'day_of_week'], axis=1, errors='ignore')
        
        return features
    
    def _extract_recovery_features(self, failure_df: pd.DataFrame, 
                                 recovery_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Extract features from failure and recovery data for timeout optimization.
        
        Args:
            failure_df: DataFrame with failure data
            recovery_df: DataFrame with recovery data (can be None)
            
        Returns:
            DataFrame with extracted features
        """
        # Use failure features as base
        features = self._extract_failure_features(failure_df)
        
        # Add recovery-specific features if available
        if recovery_df is not None and len(recovery_df) > 0:
            # Recovery time statistics
            recovery_times = recovery_df['recovery_time'].values
            features['avg_recovery_time'] = np.mean(recovery_times)
            features['med_recovery_time'] = np.median(recovery_times)
            features['std_recovery_time'] = np.std(recovery_times) if len(recovery_times) > 1 else 0
            features['min_recovery_time'] = np.min(recovery_times)
            features['max_recovery_time'] = np.max(recovery_times)
            
            # Recovery time trend (are recoveries getting faster or slower?)
            if len(recovery_df) >= 3:
                try:
                    recovery_df = recovery_df.sort_values('time')
                    x = np.arange(len(recovery_df)).reshape(-1, 1)
                    y = recovery_df['recovery_time'].values
                    
                    from sklearn.linear_model import LinearRegression
                    trend_model = LinearRegression()
                    trend_model.fit(x, y)
                    
                    features['recovery_time_trend'] = trend_model.coef_[0]
                except Exception:
                    features['recovery_time_trend'] = 0
            else:
                features['recovery_time_trend'] = 0
                
        return features
    
    def _extract_failure_prediction_features(self, failure_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for predicting failures before they happen.
        
        Args:
            failure_df: DataFrame with failure data
            
        Returns:
            DataFrame with extracted features
        """
        # Start with basic failure features
        features = self._extract_failure_features(failure_df)
        
        # Add additional time-based features
        if 'time' in failure_df.columns:
            failure_df['datetime'] = pd.to_datetime(failure_df['time'], unit='s')
            
            # Group failures by hour to find patterns
            failures_by_hour = failure_df.groupby(failure_df['datetime'].dt.hour).size()
            failures_by_hour = failures_by_hour / failures_by_hour.sum()  # Normalize
            
            # Add hour probability as a feature
            hour_dict = failures_by_hour.to_dict()
            failure_df['hour_prob'] = failure_df['datetime'].dt.hour.map(
                lambda x: hour_dict.get(x, 0)
            )
            features['hour_failure_prob'] = failure_df['hour_prob']
        
        # Add failure density features (how many failures recently)
        if 'time' in failure_df.columns:
            # Sort by time
            sorted_df = failure_df.sort_values('time')
            
            # Calculate time difference between consecutive failures
            sorted_df['time_diff'] = sorted_df['time'].diff()
            
            # Calculate moving averages of time differences
            sorted_df['time_diff_ma_3'] = sorted_df['time_diff'].rolling(3).mean()
            sorted_df['time_diff_ma_5'] = sorted_df['time_diff'].rolling(5).mean()
            
            # Add to features
            features['time_since_last_failure'] = sorted_df['time_diff']
            features['avg_time_between_failures_3'] = sorted_df['time_diff_ma_3']
            features['avg_time_between_failures_5'] = sorted_df['time_diff_ma_5']
            
            # Fill NaN values
            features = features.fillna(features.mean())
        
        return features
    
    def _extract_current_condition_features(self) -> np.ndarray:
        """
        Extract features representing current conditions for prediction.
        
        Returns:
            Array of features for current conditions
        """
        # Create a feature vector based on current time
        now = datetime.now()
        current_time = time.time()
        
        # Base features
        features = {
            'hour_sin': np.sin(2 * np.pi * now.hour / 24),
            'hour_cos': np.cos(2 * np.pi * now.hour / 24),
            'day_sin': np.sin(2 * np.pi * now.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * now.weekday() / 7),
        }
        
        # Hardware type features
        if self.hardware_specific and self.hardware_type:
            features[f'hardware_{self.hardware_type}'] = 1
            
        # Get the circuit breaker state
        if self.circuit_breaker:
            # Previous state features
            state = self.circuit_breaker.state
            features[f'prev_state_{state}'] = 1
            
            # Failure count and timing
            features['failure_count'] = self.circuit_breaker.failure_count
            
            # Time since failures
            if hasattr(self.circuit_breaker, 'last_failure_time') and self.circuit_breaker.last_failure_time > 0:
                features['time_since_last_failure'] = current_time - self.circuit_breaker.last_failure_time
            
            # Circuit state duration
            features['circuit_state_duration'] = 0  # Default value
            if state == CircuitState.OPEN and hasattr(self.circuit_breaker, 'last_failure_time'):
                features['circuit_state_duration'] = current_time - self.circuit_breaker.last_failure_time
            elif hasattr(self.circuit_breaker, 'last_success_time'):
                features['circuit_state_duration'] = current_time - self.circuit_breaker.last_success_time
                
        # Error type features (from recent failures)
        if self.recent_failures:
            # Look at most recent failures
            recent_errors = [f.get('error', '') for f in self.recent_failures[-5:]]
            
            # Count occurrences of different error types
            error_types = [
                'timeout', 'connection', 'resource', 'memory', 'gpu', 
                'crash', 'network', 'internal', 'api'
            ]
            
            for error_type in error_types:
                count = sum(1 for e in recent_errors if error_type in e.lower())
                features[f'error_{error_type}'] = count
        
        # Ensure we have all required features
        if self.threshold_model:
            missing_features = set(self.threshold_model.feature_names_in_) - set(features.keys())
            for feature in missing_features:
                features[feature] = 0
                
            # Return features in the correct order for the model
            return np.array([features[f] for f in self.threshold_model.feature_names_in_])
        
        # Return a default feature vector
        return np.array(list(features.values()))
    
    def _check_early_warning_signals(self) -> bool:
        """
        Check for early warning signals that might predict a failure.
        
        Returns:
            True if circuit should be preemptively opened, False otherwise
        """
        if not self.prediction_enabled or self.prediction_model is None:
            return False
            
        try:
            # Extract current conditions
            features = self._extract_current_condition_features()
            
            # Reshape to match model expectations
            features = features.reshape(1, -1)
            
            # Make prediction
            probability = self.prediction_model.predict_proba(features)[0][1]
            
            # Check if probability exceeds threshold (70%)
            should_preemptively_open = probability > 0.7
            
            # Log decision with probability
            if should_preemptively_open:
                logger.info(f"Early warning system triggered (probability={probability:.2f})")
            
            return should_preemptively_open
            
        except Exception as e:
            logger.error(f"Error in early warning detection: {str(e)}")
            return False
    
    async def _check_model_retraining(self) -> None:
        """Check if models should be retrained based on interval."""
        # Skip if optimization disabled or insufficient data
        if not self.optimization_enabled or not SKLEARN_AVAILABLE:
            return
            
        # Check if it's time to retrain
        now = datetime.now()
        hours_since_training = (now - self.last_model_training).total_seconds() / 3600
        
        if hours_since_training >= self.retraining_interval_hours:
            # Retrain models
            logger.info(f"Retraining ML models after {hours_since_training:.1f} hours")
            
            # Create DataFrames
            failure_df = pd.DataFrame(self.recent_failures)
            recovery_df = pd.DataFrame(self.recent_recoveries) if self.recent_recoveries else None
            
            # Check if we have enough data
            if len(failure_df) >= self.min_data_points:
                # Optimize all parameters
                self._optimize_parameters()
                
                # Save metrics to file before clearing
                self._save_metrics_to_file()
                
                # Update last training time
                self.last_model_training = now
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get detailed state information from the adaptive circuit breaker.
        
        Returns:
            Dictionary with state information
        """
        base_state = {}
        if self.circuit_breaker:
            base_state = self.circuit_breaker.get_state()
        
        # Add adaptive information
        adaptive_state = {
            "name": self.name,
            "optimization_enabled": self.optimization_enabled,
            "prediction_enabled": self.prediction_enabled,
            "learning_rate": self.learning_rate,
            "base_failure_threshold": self.base_failure_threshold,
            "current_failure_threshold": self.current_failure_threshold,
            "base_recovery_timeout": self.base_recovery_timeout,
            "current_recovery_timeout": self.current_recovery_timeout,
            "base_half_open_timeout": self.base_half_open_timeout,
            "current_half_open_timeout": self.current_half_open_timeout,
            "hardware_specific": self.hardware_specific,
            "hardware_type": self.hardware_type,
            "last_model_training": self.last_model_training.isoformat() if self.last_model_training else None,
            "metrics_collected": len(self.metrics),
            "failures_collected": len(self.recent_failures),
            "recoveries_collected": len(self.recent_recoveries),
            "has_threshold_model": self.threshold_model is not None,
            "has_recovery_model": self.recovery_timeout_model is not None,
            "has_half_open_model": self.half_open_timeout_model is not None,
            "has_prediction_model": self.prediction_model is not None,
        }
        
        # Add feature importances if available
        if self.feature_importances:
            adaptive_state["feature_importances"] = self.feature_importances
            
        # Combine states
        return {**base_state, **adaptive_state}
    
    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        if self.circuit_breaker:
            self.circuit_breaker.reset()
            
        # Optionally: clear recent failure/recovery data
        # Uncomment if you want a full reset including ML data
        # self.recent_failures = []
        # self.recent_recoveries = []
        
    async def execute_with_retries(self, 
                               func: Callable[..., Awaitable[Any]], 
                               max_retries: int = 3, 
                               retry_delay: float = 1.0,
                               *args, **kwargs) -> Any:
        """
        Execute a function with automatic retries.
        
        Args:
            func: Async function to execute
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff applied)
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        retries = 0
        last_exception = None
        
        while retries <= max_retries:
            try:
                # Execute the function with circuit breaker protection
                return await self.execute(func, *args, **kwargs)
                
            except Exception as e:
                last_exception = e
                retries += 1
                
                # Check if we've exhausted retries
                if retries > max_retries:
                    break
                    
                # Log retry attempt
                logger.info(f"Retry {retries}/{max_retries} after error: {str(e)}")
                
                # Exponential backoff
                delay = retry_delay * (2 ** (retries - 1))
                await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        logger.error(f"All {max_retries} retries failed")
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"Failed after {max_retries} retries with unknown error")
    
    def close(self) -> None:
        """Clean up resources used by the adaptive circuit breaker."""
        # Save remaining metrics
        if self.metrics:
            self._save_metrics_to_file()
            
        # Close database connection
        if self.db_conn is not None:
            self.db_conn.close()
            self.db_conn = None
    
    def __del__(self) -> None:
        """Ensure resources are properly cleaned up."""
        self.close()


async def demo():
    """Demo of the Adaptive Circuit Breaker functionality."""
    # Create an adaptive circuit breaker
    circuit_breaker = AdaptiveCircuitBreaker(
        name="demo_circuit",
        base_failure_threshold=3,
        base_recovery_timeout=10,
        base_half_open_timeout=3,
        optimization_enabled=True,
        prediction_enabled=True,
        metrics_path="./metrics/demo_circuit",
        learning_rate=0.2
    )
    
    # Example operations to demonstrate functionality
    async def successful_operation():
        await asyncio.sleep(0.1)
        return "Success!"
        
    async def failing_operation():
        await asyncio.sleep(0.1)
        raise Exception("Simulated failure")
    
    # Test successful operations
    print("Running successful operations...")
    for i in range(5):
        try:
            result = await circuit_breaker.execute(successful_operation)
            print(f"  Operation {i+1} result: {result}")
        except Exception as e:
            print(f"  Operation {i+1} failed: {str(e)}")
    
    # Test failing operations
    print("\nRunning failing operations...")
    for i in range(5):
        try:
            result = await circuit_breaker.execute(failing_operation)
            print(f"  Operation {i+1} result: {result}")
        except Exception as e:
            print(f"  Operation {i+1} failed: {str(e)}")
    
    # Check circuit state
    state = circuit_breaker.get_state()
    print("\nCircuit Breaker State:")
    print(json.dumps(state, indent=2))
    
    # Cleanup
    circuit_breaker.close()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo())