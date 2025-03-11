#!/usr/bin/env python3
"""
Model Update Pipeline for the Predictive Performance System.

This module implements a streamlined pipeline for incorporating new benchmark data
into the predictive models without requiring full retraining. It tracks model
improvement, implements efficient incremental updates, and provides mechanisms
for continuous integration of new benchmark results.

Key features:
1. Incremental model updates without full retraining
2. Model improvement tracking over successive updates
3. Uncertainty calibration based on new data
4. Selective update strategies based on data characteristics
5. Update quality validation
6. Integration with Active Learning System for sequential testing strategy
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
import copy

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("scikit-learn not available, using simulation mode")
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predictive_performance.model_update_pipeline")

# Default paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDICTIVE_DIR = PROJECT_ROOT / "predictive_performance"
DATA_DIR = PREDICTIVE_DIR / "data"
MODELS_DIR = PREDICTIVE_DIR / "models" / "trained_models"

class ModelUpdatePipeline:
    """
    Pipeline for efficiently updating predictive models with new benchmark data.
    
    This class implements methods for incremental model updates, model improvement
    tracking, and integration with the Active Learning System for continuous
    model refinement.
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        update_strategy: str = "incremental",
        learning_rate_decay: float = 0.9,
        min_samples_for_update: int = 5,
        max_update_iterations: int = 100,
        update_threshold: float = 0.01,
        retrain_threshold: float = 0.3,
        verbose: bool = False
    ):
        """
        Initialize the Model Update Pipeline.
        
        Args:
            model_dir: Directory containing trained models
            data_dir: Directory containing benchmark data
            metrics: List of metrics to update models for (e.g., throughput, latency, memory)
            update_strategy: Strategy for updating models ("incremental", "window", "weighted")
            learning_rate_decay: Factor to decay learning rate with each update
            min_samples_for_update: Minimum number of samples required for an update
            max_update_iterations: Maximum number of iterations for incremental updates
            update_threshold: Minimum improvement threshold to accept an update
            retrain_threshold: Threshold for when to perform full retraining
            verbose: Whether to log detailed information
        """
        self.model_dir = model_dir or str(MODELS_DIR)
        self.data_dir = data_dir or str(DATA_DIR)
        self.metrics = metrics or ["throughput", "latency", "memory"]
        self.update_strategy = update_strategy
        self.learning_rate_decay = learning_rate_decay
        self.min_samples_for_update = min_samples_for_update
        self.max_update_iterations = max_update_iterations
        self.update_threshold = update_threshold
        self.retrain_threshold = retrain_threshold
        
        # Configure logging verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize model and data attributes
        self.models = {}
        self.original_models = {}
        self.model_info = {}
        self.data = None
        self.feature_columns = None
        self.target_columns = None
        self.update_history = []
        
        # Try to load models if available
        self._load_models()
        
        # Try to load existing data if available
        self._load_data()
    
    def _load_models(self) -> bool:
        """
        Load existing trained models.
        
        Returns:
            Success flag
        """
        try:
            from train_models import load_prediction_models
            self.models = load_prediction_models(self.model_dir)
            
            if not self.models:
                logger.warning(f"No models found in {self.model_dir}")
                return False
            
            # Create a deep copy of the original models for comparison
            self.original_models = copy.deepcopy(self.models)
            
            # Load model info if available
            model_info_file = os.path.join(self.model_dir, "model_info.json")
            if os.path.exists(model_info_file):
                with open(model_info_file, 'r') as f:
                    self.model_info = json.load(f)
                
                logger.info(f"Loaded model info from {model_info_file}")
            
            logger.info(f"Loaded {len(self.models)} prediction models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def _load_data(self) -> bool:
        """
        Load existing benchmark data.
        
        Returns:
            Success flag
        """
        try:
            data_file = os.path.join(self.data_dir, "benchmark_data.parquet")
            
            if not os.path.exists(data_file):
                logger.warning(f"No benchmark data found at {data_file}")
                return False
            
            self.data = pd.read_parquet(data_file)
            
            # Infer feature and target columns
            self.target_columns = [col for col in self.data.columns if col in self.metrics]
            self.feature_columns = [col for col in self.data.columns if col not in self.target_columns]
            
            logger.info(f"Loaded {len(self.data)} benchmark records from {data_file}")
            logger.debug(f"Feature columns: {self.feature_columns}")
            logger.debug(f"Target columns: {self.target_columns}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def _save_models(self) -> bool:
        """
        Save updated models.
        
        Returns:
            Success flag
        """
        try:
            from train_models import save_prediction_models
            
            # Save models
            save_path = save_prediction_models(self.models, self.model_dir)
            
            # Update model info with update history
            update_history_entry = {
                "timestamp": datetime.now().isoformat(),
                "update_strategy": self.update_strategy,
                "metrics_updated": self.metrics,
                "update_history": self.update_history
            }
            
            # Add to model info
            if "update_history" not in self.model_info:
                self.model_info["update_history"] = []
            
            self.model_info["update_history"].append(update_history_entry)
            self.model_info["last_updated"] = datetime.now().isoformat()
            
            # Save model info
            model_info_file = os.path.join(self.model_dir, "model_info.json")
            with open(model_info_file, 'w') as f:
                json.dump(self.model_info, f, indent=2)
            
            logger.info(f"Saved updated models to {save_path}")
            logger.info(f"Updated model info saved to {model_info_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False
    
    def update_models(
        self, 
        new_data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        update_strategy: Optional[str] = None,
        validation_split: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Update predictive models with new benchmark data.
        
        Args:
            new_data: DataFrame containing new benchmark data
            metrics: List of metrics to update models for
            update_strategy: Strategy to use for update
            validation_split: Fraction of new data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing update metrics and information
        """
        metrics = metrics or self.metrics
        update_strategy = update_strategy or self.update_strategy
        
        if len(new_data) < self.min_samples_for_update:
            logger.warning(f"Not enough samples for update. Minimum required: {self.min_samples_for_update}")
            return {"success": False, "reason": "insufficient_samples"}
        
        logger.info(f"Updating models with {len(new_data)} new benchmark records")
        logger.info(f"Update strategy: {update_strategy}")
        logger.info(f"Metrics to update: {metrics}")
        
        if not self.models:
            logger.error("No models loaded for update")
            return {"success": False, "reason": "no_models_loaded"}
        
        # Combine with existing data if available
        if self.data is not None:
            combined_data = pd.concat([self.data, new_data], ignore_index=True)
            logger.info(f"Combined with existing data. Total records: {len(combined_data)}")
        else:
            combined_data = new_data
            logger.info(f"No existing data. Using only new data: {len(combined_data)}")
        
        # Track update metrics
        update_metrics = {}
        
        # Process each metric
        for metric in metrics:
            if metric not in self.models:
                logger.warning(f"No model found for metric: {metric}")
                continue
            
            logger.info(f"Updating model for metric: {metric}")
            
            # Get the current model
            model = self.models[metric]
            
            # Extract features and target
            X = self._extract_features(combined_data)
            y = combined_data[metric].values
            
            # Split new data for validation
            new_X = self._extract_features(new_data)
            new_y = new_data[metric].values
            
            X_val, X_update, y_val, y_update = train_test_split(
                new_X, new_y, test_size=(1-validation_split), random_state=random_state
            )
            
            # Get baseline metrics before update
            baseline_predictions = model.predict(X_val)
            baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_predictions))
            baseline_r2 = r2_score(y_val, baseline_predictions)
            
            try:
                baseline_mape = mean_absolute_percentage_error(y_val, baseline_predictions)
            except:
                # Handle zero values in y_val
                baseline_mape = np.mean(np.abs((y_val - baseline_predictions) / np.maximum(y_val, 1e-10)))
            
            logger.info(f"Baseline metrics - RMSE: {baseline_rmse:.4f}, R²: {baseline_r2:.4f}, MAPE: {baseline_mape:.4f}")
            
            # Update the model based on strategy
            if update_strategy == "incremental":
                updated_model, update_info = self._incremental_update(
                    model, X_update, y_update, X_val, y_val
                )
            elif update_strategy == "window":
                updated_model, update_info = self._window_update(
                    model, X, y, X_val, y_val
                )
            elif update_strategy == "weighted":
                updated_model, update_info = self._weighted_update(
                    model, X_update, y_update, X_val, y_val
                )
            else:
                logger.error(f"Unknown update strategy: {update_strategy}")
                continue
            
            # Check if update improved the model
            if update_info["improvement_percent"] > self.update_threshold:
                # Accept the update
                self.models[metric] = updated_model
                logger.info(f"Update accepted for {metric}. Improvement: {update_info['improvement_percent']:.2f}%")
                update_metrics[metric] = update_info
            else:
                # Reject the update
                logger.info(f"Update rejected for {metric}. Improvement below threshold: {update_info['improvement_percent']:.2f}%")
                update_metrics[metric] = {
                    "improvement_percent": update_info["improvement_percent"],
                    "status": "rejected",
                    "reason": "below_threshold"
                }
        
        # Calculate overall improvement
        overall_improvement = np.mean([
            metrics["improvement_percent"] 
            for metric, metrics in update_metrics.items() 
            if isinstance(metrics, dict) and "improvement_percent" in metrics
        ])
        
        # Record update information
        update_record = {
            "timestamp": datetime.now().isoformat(),
            "update_strategy": update_strategy,
            "metrics_updated": metrics,
            "new_samples": len(new_data),
            "total_samples": len(combined_data),
            "overall_improvement": overall_improvement,
            "metric_improvements": {
                metric: info.get("improvement_percent", 0) 
                for metric, info in update_metrics.items()
            }
        }
        
        self.update_history.append(update_record)
        
        # Update internal data
        self.data = combined_data
        
        # Save updated models
        if overall_improvement > 0:
            self._save_models()
            logger.info(f"Models updated and saved. Overall improvement: {overall_improvement:.2f}%")
        else:
            logger.info("No overall improvement. Models not saved.")
        
        return {
            "success": True,
            "update_record": update_record,
            "metric_details": update_metrics
        }
    
    def _incremental_update(
        self, 
        model, 
        X_update: np.ndarray, 
        y_update: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Update model incrementally with new data.
        
        Args:
            model: Model to update
            X_update: Features for update
            y_update: Target values for update
            X_val: Validation features
            y_val: Validation target values
            
        Returns:
            Tuple of (updated model, update info)
        """
        logger.debug("Applying incremental update strategy")
        
        # Make a copy of the original model
        updated_model = copy.deepcopy(model)
        
        # Get current learning rate
        current_lr = updated_model.get_params().get('learning_rate', 0.1)
        
        # Adjust learning rate for incremental update
        incremental_lr = current_lr * self.learning_rate_decay
        updated_model.set_params(learning_rate=incremental_lr)
        
        # Predict on validation data before update
        y_pred_before = updated_model.predict(X_val)
        rmse_before = np.sqrt(mean_squared_error(y_val, y_pred_before))
        r2_before = r2_score(y_val, y_pred_before)
        
        # Perform incremental update
        # For GradientBoostingRegressor, we can use warm_start for incremental updates
        if isinstance(updated_model, GradientBoostingRegressor):
            # Store original n_estimators
            original_n_estimators = updated_model.n_estimators
            
            # Set warm_start to True for incremental learning
            updated_model.set_params(warm_start=True)
            
            # Add more estimators incrementally
            n_new_estimators = min(30, original_n_estimators // 5)  # Add 20% more or 30, whichever is less
            updated_model.set_params(n_estimators=original_n_estimators + n_new_estimators)
            
            # Fit on the new data
            updated_model.fit(X_update, y_update)
            
        elif isinstance(updated_model, RandomForestRegressor):
            # For RandomForestRegressor, we'll train new trees and merge
            # Train a separate model on new data
            new_model = RandomForestRegressor(
                n_estimators=min(30, len(updated_model.estimators_) // 5),
                random_state=42
            )
            new_model.fit(X_update, y_update)
            
            # Combine estimators from both models
            updated_model.estimators_ = updated_model.estimators_ + new_model.estimators_
            updated_model.n_estimators = len(updated_model.estimators_)
            
        else:
            # For other model types, perform a simple refit on update data
            # This is a simplified approach and might not be optimal for all model types
            updated_model.fit(X_update, y_update)
        
        # Predict on validation data after update
        y_pred_after = updated_model.predict(X_val)
        rmse_after = np.sqrt(mean_squared_error(y_val, y_pred_after))
        r2_after = r2_score(y_val, y_pred_after)
        
        # Calculate improvement
        rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
        r2_improvement = (r2_after - r2_before) / (1 - r2_before) * 100 if r2_before < 1 else 0
        
        update_info = {
            "rmse_before": float(rmse_before),
            "rmse_after": float(rmse_after),
            "r2_before": float(r2_before),
            "r2_after": float(r2_after),
            "rmse_improvement": float(rmse_improvement),
            "r2_improvement": float(r2_improvement),
            "improvement_percent": float(rmse_improvement),  # Using RMSE improvement as primary metric
            "status": "accepted" if rmse_improvement > 0 else "rejected",
            "learning_rate": float(incremental_lr),
            "strategy": "incremental"
        }
        
        logger.debug(f"Incremental update results - RMSE improvement: {rmse_improvement:.2f}%, R² improvement: {r2_improvement:.2f}%")
        
        return updated_model, update_info
    
    def _window_update(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Update model using a sliding window approach.
        
        Args:
            model: Model to update
            X: All features (including new data)
            y: All target values (including new data)
            X_val: Validation features
            y_val: Validation target values
            
        Returns:
            Tuple of (updated model, update info)
        """
        logger.debug("Applying window update strategy")
        
        # Make a copy of the original model
        updated_model = copy.deepcopy(model)
        
        # Predict on validation data before update
        y_pred_before = updated_model.predict(X_val)
        rmse_before = np.sqrt(mean_squared_error(y_val, y_pred_before))
        r2_before = r2_score(y_val, y_pred_before)
        
        # Create a new model with the same hyperparameters
        if isinstance(updated_model, GradientBoostingRegressor):
            # Get current parameters
            params = updated_model.get_params()
            
            # Create new model with same parameters
            new_model = GradientBoostingRegressor(**params)
            
            # Fit on all available data
            new_model.fit(X, y)
            
            # Replace the model
            updated_model = new_model
            
        elif isinstance(updated_model, RandomForestRegressor):
            # Get current parameters
            params = updated_model.get_params()
            
            # Create new model with same parameters
            new_model = RandomForestRegressor(**params)
            
            # Fit on all available data
            new_model.fit(X, y)
            
            # Replace the model
            updated_model = new_model
            
        else:
            # For other model types, perform a simple refit
            updated_model.fit(X, y)
        
        # Predict on validation data after update
        y_pred_after = updated_model.predict(X_val)
        rmse_after = np.sqrt(mean_squared_error(y_val, y_pred_after))
        r2_after = r2_score(y_val, y_pred_after)
        
        # Calculate improvement
        rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
        r2_improvement = (r2_after - r2_before) / (1 - r2_before) * 100 if r2_before < 1 else 0
        
        update_info = {
            "rmse_before": float(rmse_before),
            "rmse_after": float(rmse_after),
            "r2_before": float(r2_before),
            "r2_after": float(r2_after),
            "rmse_improvement": float(rmse_improvement),
            "r2_improvement": float(r2_improvement),
            "improvement_percent": float(rmse_improvement),  # Using RMSE improvement as primary metric
            "status": "accepted" if rmse_improvement > 0 else "rejected",
            "strategy": "window"
        }
        
        logger.debug(f"Window update results - RMSE improvement: {rmse_improvement:.2f}%, R² improvement: {r2_improvement:.2f}%")
        
        return updated_model, update_info
    
    def _weighted_update(
        self, 
        model, 
        X_update: np.ndarray, 
        y_update: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Update model using weighted combination of old and new models.
        
        Args:
            model: Model to update
            X_update: Features for update
            y_update: Target values for update
            X_val: Validation features
            y_val: Validation target values
            
        Returns:
            Tuple of (updated model, update info)
        """
        logger.debug("Applying weighted update strategy")
        
        # Make a copy of the original model
        original_model = copy.deepcopy(model)
        
        # Predict on validation data before update
        y_pred_before = original_model.predict(X_val)
        rmse_before = np.sqrt(mean_squared_error(y_val, y_pred_before))
        r2_before = r2_score(y_val, y_pred_before)
        
        # Train a new model on the update data
        if isinstance(original_model, GradientBoostingRegressor):
            # Get parameters from original model
            params = original_model.get_params()
            
            # Create new model with same parameters
            new_model = GradientBoostingRegressor(**params)
            
            # Fit on update data
            new_model.fit(X_update, y_update)
            
        elif isinstance(original_model, RandomForestRegressor):
            # Get parameters from original model
            params = original_model.get_params()
            
            # Create new model with same parameters
            new_model = RandomForestRegressor(**params)
            
            # Fit on update data
            new_model.fit(X_update, y_update)
            
        else:
            # For other model types, create a simple copy and refit
            new_model = copy.deepcopy(original_model)
            new_model.fit(X_update, y_update)
        
        # Try different weights for model combination to find optimal
        best_rmse = float('inf')
        best_weight = 0.5
        best_predictions = None
        
        # Search for optimal weight
        for weight in np.linspace(0.1, 0.9, 9):
            # Make weighted predictions
            pred_original = original_model.predict(X_val)
            pred_new = new_model.predict(X_val)
            
            weighted_pred = weight * pred_original + (1 - weight) * pred_new
            
            # Calculate RMSE
            current_rmse = np.sqrt(mean_squared_error(y_val, weighted_pred))
            
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_weight = weight
                best_predictions = weighted_pred
        
        # Create a weighted model combiner
        class WeightedModelCombiner:
            def __init__(self, model1, model2, weight):
                self.model1 = model1
                self.model2 = model2
                self.weight = weight
            
            def predict(self, X):
                pred1 = self.model1.predict(X)
                pred2 = self.model2.predict(X)
                return self.weight * pred1 + (1 - self.weight) * pred2
            
            def get_params(self, deep=True):
                # Return a combination of both models' params
                params1 = self.model1.get_params(deep=deep)
                params2 = self.model2.get_params(deep=deep)
                
                # Add special params for the combiner
                params = {}
                params.update({f"model1__{k}": v for k, v in params1.items()})
                params.update({f"model2__{k}": v for k, v in params2.items()})
                params["weight"] = self.weight
                
                return params
        
        # Create the combined model
        combined_model = WeightedModelCombiner(original_model, new_model, best_weight)
        
        # Calculate metrics with best weight
        rmse_after = best_rmse
        y_pred_after = best_predictions
        r2_after = r2_score(y_val, y_pred_after)
        
        # Calculate improvement
        rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
        r2_improvement = (r2_after - r2_before) / (1 - r2_before) * 100 if r2_before < 1 else 0
        
        update_info = {
            "rmse_before": float(rmse_before),
            "rmse_after": float(rmse_after),
            "r2_before": float(r2_before),
            "r2_after": float(r2_after),
            "rmse_improvement": float(rmse_improvement),
            "r2_improvement": float(r2_improvement),
            "improvement_percent": float(rmse_improvement),  # Using RMSE improvement as primary metric
            "optimal_weight": float(best_weight),
            "status": "accepted" if rmse_improvement > 0 else "rejected",
            "strategy": "weighted"
        }
        
        logger.debug(f"Weighted update results - RMSE improvement: {rmse_improvement:.2f}%, R² improvement: {r2_improvement:.2f}%, Weight: {best_weight:.2f}")
        
        return combined_model, update_info
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from data.
        
        Args:
            data: DataFrame containing benchmark data
            
        Returns:
            Feature matrix
        """
        # If feature columns are not defined, use all columns except target metrics
        if self.feature_columns is None:
            self.feature_columns = [col for col in data.columns if col not in self.metrics]
        
        # Check for categorical features
        categorical_features = []
        numerical_features = []
        
        for col in self.feature_columns:
            if col in data.columns:
                if data[col].dtype == 'object' or data[col].dtype == 'category':
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
        
        # Create dummy variables for categorical features
        if categorical_features:
            X_categorical = pd.get_dummies(data[categorical_features], drop_first=False)
        else:
            X_categorical = pd.DataFrame(index=data.index)
        
        # Add numerical features
        if numerical_features:
            X_numerical = data[numerical_features]
            X = pd.concat([X_numerical, X_categorical], axis=1)
        else:
            X = X_categorical
        
        return X.values
    
    def evaluate_model_improvement(self, metric: str) -> Dict[str, Any]:
        """
        Evaluate improvement since original model.
        
        Args:
            metric: Metric to evaluate (e.g., throughput, latency, memory)
            
        Returns:
            Dictionary containing improvement metrics
        """
        if metric not in self.models or metric not in self.original_models:
            logger.error(f"Model for metric {metric} not found")
            return {"success": False, "reason": "model_not_found"}
        
        if self.data is None:
            logger.error("No data available for evaluation")
            return {"success": False, "reason": "no_data"}
        
        # Extract features and target
        X = self._extract_features(self.data)
        y = self.data[metric].values
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get current model and original model
        current_model = self.models[metric]
        original_model = self.original_models[metric]
        
        # Evaluate original model
        y_pred_original = original_model.predict(X_test)
        rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
        r2_original = r2_score(y_test, y_pred_original)
        
        try:
            mape_original = mean_absolute_percentage_error(y_test, y_pred_original)
        except:
            # Handle zero values in y_test
            mape_original = np.mean(np.abs((y_test - y_pred_original) / np.maximum(y_test, 1e-10)))
        
        # Evaluate current model
        y_pred_current = current_model.predict(X_test)
        rmse_current = np.sqrt(mean_squared_error(y_test, y_pred_current))
        r2_current = r2_score(y_test, y_pred_current)
        
        try:
            mape_current = mean_absolute_percentage_error(y_test, y_pred_current)
        except:
            # Handle zero values in y_test
            mape_current = np.mean(np.abs((y_test - y_pred_current) / np.maximum(y_test, 1e-10)))
        
        # Calculate improvement
        rmse_improvement = (rmse_original - rmse_current) / rmse_original * 100
        r2_improvement = (r2_current - r2_original) / (1 - r2_original) * 100 if r2_original < 1 else 0
        mape_improvement = (mape_original - mape_current) / mape_original * 100
        
        # Create evaluation result
        evaluation = {
            "success": True,
            "metric": metric,
            "original_model": {
                "rmse": float(rmse_original),
                "r2": float(r2_original),
                "mape": float(mape_original)
            },
            "current_model": {
                "rmse": float(rmse_current),
                "r2": float(r2_current),
                "mape": float(mape_current)
            },
            "improvement": {
                "rmse_percent": float(rmse_improvement),
                "r2_percent": float(r2_improvement),
                "mape_percent": float(mape_improvement)
            },
            "update_count": len(self.update_history)
        }
        
        logger.info(f"Model improvement for {metric} - RMSE: {rmse_improvement:.2f}%, R²: {r2_improvement:.2f}%, MAPE: {mape_improvement:.2f}%")
        
        return evaluation
    
    def determine_update_need(self, new_data: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Determine if models need update based on prediction errors on new data.
        
        Args:
            new_data: DataFrame containing new benchmark data
            threshold: Error threshold to recommend update
            
        Returns:
            Dictionary containing update recommendations
        """
        if not self.models:
            logger.error("No models loaded")
            return {"needs_update": True, "reason": "no_models_loaded"}
        
        if len(new_data) < 5:
            logger.warning("Not enough new data to determine update need")
            return {"needs_update": False, "reason": "insufficient_data"}
        
        recommendations = {}
        overall_error_increase = 0
        
        for metric in self.metrics:
            if metric not in self.models or metric not in new_data.columns:
                continue
            
            # Extract features from new data
            X_new = self._extract_features(new_data)
            y_new = new_data[metric].values
            
            # Make predictions with current model
            model = self.models[metric]
            y_pred = model.predict(X_new)
            
            # Calculate error metrics
            rmse = np.sqrt(mean_squared_error(y_new, y_pred))
            r2 = r2_score(y_new, y_pred)
            
            try:
                mape = mean_absolute_percentage_error(y_new, y_pred)
            except:
                # Handle zero values in y_new
                mape = np.mean(np.abs((y_new - y_pred) / np.maximum(y_new, 1e-10)))
            
            # Get baseline metrics from model info
            baseline_metrics = self.model_info.get("model_metrics", {}).get(metric, {})
            baseline_rmse = baseline_metrics.get("rmse", rmse)
            
            # Calculate error increase
            error_increase = (rmse - baseline_rmse) / baseline_rmse
            
            # Determine if update is needed
            needs_update = error_increase > threshold
            
            recommendations[metric] = {
                "needs_update": needs_update,
                "error_increase": float(error_increase),
                "current_rmse": float(rmse),
                "baseline_rmse": float(baseline_rmse),
                "r2": float(r2),
                "mape": float(mape)
            }
            
            overall_error_increase += error_increase
        
        # Calculate overall recommendation
        metrics_analyzed = len(recommendations)
        overall_error_increase = overall_error_increase / metrics_analyzed if metrics_analyzed > 0 else 0
        overall_needs_update = overall_error_increase > threshold
        
        # Determine update strategy based on error increase
        update_strategy = "incremental"
        if overall_error_increase > self.retrain_threshold:
            update_strategy = "window"
        
        result = {
            "needs_update": overall_needs_update,
            "error_increase": float(overall_error_increase),
            "metric_recommendations": recommendations,
            "recommended_strategy": update_strategy,
            "threshold": threshold,
            "retrain_threshold": self.retrain_threshold
        }
        
        logger.info(f"Update need analysis - Overall error increase: {overall_error_increase:.2f}, Needs update: {overall_needs_update}")
        
        return result
    
    def integrate_with_active_learning(
        self, 
        active_learning_system, 
        new_data: pd.DataFrame,
        sequential_rounds: int = 1,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Integrate with Active Learning System for sequential test-update cycles.
        
        Args:
            active_learning_system: Instance of ActiveLearningSystem
            new_data: DataFrame containing new benchmark data
            sequential_rounds: Number of sequential test-update rounds
            batch_size: Number of configurations per batch
            
        Returns:
            Dictionary containing integration results
        """
        results = []
        
        for round_idx in range(sequential_rounds):
            logger.info(f"Sequential testing round {round_idx + 1}/{sequential_rounds}")
            
            # Update the active learning system with new data
            active_learning_system.update_with_benchmark_results(new_data)
            
            # Generate the next batch of configurations to test
            batch = active_learning_system.suggest_test_batch(
                configurations=active_learning_system.recommend_configurations(batch_size * 2),
                batch_size=batch_size,
                ensure_diversity=True
            )
            
            # Update models with new data
            update_result = self.update_models(new_data)
            
            round_result = {
                "round": round_idx + 1,
                "batch_size": len(batch),
                "update_result": update_result,
                "improvement": update_result.get("update_record", {}).get("overall_improvement", 0)
            }
            
            results.append(round_result)
            
            logger.info(f"Round {round_idx + 1} complete - Improvement: {round_result['improvement']:.2f}%")
        
        # Calculate overall improvement across all rounds
        overall_improvement = np.mean([r["improvement"] for r in results])
        
        integration_result = {
            "success": True,
            "rounds": sequential_rounds,
            "overall_improvement": float(overall_improvement),
            "round_results": results,
            "next_batch": batch.to_dict(orient="records") if isinstance(batch, pd.DataFrame) else []
        }
        
        logger.info(f"Integration complete - Overall improvement: {overall_improvement:.2f}%")
        
        return integration_result

def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Update Pipeline")
    parser.add_argument("--model-dir", help="Directory containing trained models")
    parser.add_argument("--data-dir", help="Directory containing benchmark data")
    parser.add_argument("--new-data", help="Path to new benchmark data (CSV or Parquet)")
    parser.add_argument("--update-strategy", choices=["incremental", "window", "weighted"], default="incremental", help="Update strategy")
    parser.add_argument("--metrics", help="Comma-separated list of metrics to update")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model improvement")
    parser.add_argument("--determine-need", action="store_true", help="Determine if update is needed")
    parser.add_argument("--output", help="Path to output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize the pipeline
    pipeline = ModelUpdatePipeline(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        update_strategy=args.update_strategy,
        verbose=args.verbose
    )
    
    # Parse metrics if provided
    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",")]
    
    # Load new data if provided
    new_data = None
    if args.new_data:
        if args.new_data.endswith(".csv"):
            new_data = pd.read_csv(args.new_data)
        elif args.new_data.endswith(".parquet"):
            new_data = pd.read_parquet(args.new_data)
        else:
            logger.error("Unsupported file format. Use CSV or Parquet.")
            return
    
    result = None
    
    # Determine action
    if args.evaluate:
        # Evaluate model improvement
        results = {}
        for metric in metrics or pipeline.metrics:
            results[metric] = pipeline.evaluate_model_improvement(metric)
        
        # Calculate overall improvement
        overall_improvement = np.mean([
            r.get("improvement", {}).get("rmse_percent", 0) 
            for r in results.values() 
            if isinstance(r, dict) and "improvement" in r
        ])
        
        result = {
            "action": "evaluate",
            "overall_improvement": float(overall_improvement),
            "metric_results": results
        }
        
    elif args.determine_need and new_data is not None:
        # Determine if update is needed
        result = {
            "action": "determine_need",
            "analysis": pipeline.determine_update_need(new_data)
        }
        
    elif new_data is not None:
        # Update models with new data
        result = {
            "action": "update",
            "update_result": pipeline.update_models(new_data, metrics=metrics)
        }
    
    # Save result if output file is provided
    if result and args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to {args.output}")
    
    # Print summary
    if result:
        print("\nModel Update Pipeline Summary:")
        if result["action"] == "evaluate":
            print(f"Overall improvement: {result['overall_improvement']:.2f}%")
            for metric, res in result["metric_results"].items():
                if isinstance(res, dict) and "improvement" in res:
                    imp = res["improvement"]
                    print(f"  {metric}: RMSE: {imp.get('rmse_percent', 0):.2f}%, R²: {imp.get('r2_percent', 0):.2f}%")
        
        elif result["action"] == "determine_need":
            analysis = result["analysis"]
            print(f"Update needed: {analysis.get('needs_update', False)}")
            print(f"Error increase: {analysis.get('error_increase', 0):.2f}")
            print(f"Recommended strategy: {analysis.get('recommended_strategy', 'incremental')}")
        
        elif result["action"] == "update":
            update_result = result["update_result"]
            update_record = update_result.get("update_record", {})
            print(f"Update successful: {update_result.get('success', False)}")
            print(f"Overall improvement: {update_record.get('overall_improvement', 0):.2f}%")
            print(f"New samples: {update_record.get('new_samples', 0)}")
            print(f"Metrics updated: {', '.join(update_record.get('metrics_updated', []))}")

if __name__ == "__main__":
    main()