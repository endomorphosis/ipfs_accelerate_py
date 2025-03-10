#!/usr/bin/env python3
"""
Active Learning Pipeline for the Predictive Performance System.

This module implements a sophisticated active learning pipeline that identifies
high-value benchmark configurations for testing, prioritizes them based on expected
information gain, and updates prediction models with new benchmark results.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
import warnings

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("scikit-learn not available, using simulation mode")
    SKLEARN_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    warnings.warn("joblib not available, parallel processing disabled")
    JOBLIB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predictive_performance.active_learning")

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ActiveLearningSystem:
    """
    Active Learning System for identifying high-value benchmark configurations.
    
    This system uses uncertainty estimation and expected information gain
    to identify which model-hardware configurations would be most valuable
    to benchmark next, helping to improve prediction accuracy efficiently.
    """
    
    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize the active learning system.
        
        Args:
            data_file: Path to existing benchmark data file
        """
        self.model_types = ["text_embedding", "text_generation", "vision", "audio", "multimodal"]
        self.hardware_platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        self.precision_formats = ["fp32", "fp16", "int8", "int4"]
        
        self.data_file = data_file
        self.data = self._load_data() if data_file else self._generate_synthetic_data()
        
        # Matrix of explored configurations
        self.explored_configs = set()
        if self.data is not None:
            for _, row in self.data.iterrows():
                config_key = (row["model_type"], row["hardware"], row["batch_size"])
                self.explored_configs.add(config_key)
        
        # Generate all possible configurations
        self.all_configs = []
        for model_type in self.model_types:
            for hardware in self.hardware_platforms:
                for batch_size in self.batch_sizes:
                    config = {
                        "model_name": f"example_{model_type}_model",
                        "model_type": model_type,
                        "hardware": hardware,
                        "batch_size": batch_size
                    }
                    self.all_configs.append(config)
                    
        # Initialize prediction model if scikit-learn is available
        self.prediction_model = None
        if SKLEARN_AVAILABLE and self.data is not None and len(self.data) > 10:
            self._initialize_prediction_model()
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        """Load benchmark data from file."""
        if self.data_file and os.path.exists(self.data_file):
            try:
                return pd.read_csv(self.data_file)
            except Exception as e:
                logger.warning(f"Failed to load data from {self.data_file}: {e}")
        
        return None
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic benchmark data for testing."""
        # Create a small dataset of "already benchmarked" configurations
        data = []
        
        # Add some synthetic benchmark results
        for model_type in self.model_types[:2]:  # Just use first 2 model types
            for hardware in self.hardware_platforms[:3]:  # Just use first 3 hardware platforms
                for batch_size in [1, 4]:  # Just use batch sizes 1 and 4
                    # Create a synthetic benchmark result
                    throughput_base = {
                        "text_embedding": 200,
                        "text_generation": 20,
                        "vision": 50,
                        "audio": 10,
                        "multimodal": 5
                    }.get(model_type, 100)
                    
                    latency_base = {
                        "text_embedding": 10,
                        "text_generation": 100,
                        "vision": 30,
                        "audio": 200,
                        "multimodal": 300
                    }.get(model_type, 50)
                    
                    memory_base = {
                        "text_embedding": 1024,
                        "text_generation": 4096,
                        "vision": 2048,
                        "audio": 3072,
                        "multimodal": 6144
                    }.get(model_type, 2048)
                    
                    # Hardware factors
                    hw_factor = {
                        "cpu": 1.0,
                        "cuda": 8.0,
                        "rocm": 7.5,
                        "mps": 5.0,
                        "openvino": 3.5,
                        "qnn": 2.5,
                        "webnn": 2.0,
                        "webgpu": 3.0
                    }.get(hardware, 1.0)
                    
                    # Add some randomness
                    import random
                    random.seed(hash(f"{model_type}_{hardware}_{batch_size}"))
                    
                    throughput = throughput_base * hw_factor * (batch_size ** 0.7) * random.uniform(0.85, 1.15)
                    latency = latency_base / hw_factor * (1 + 0.1 * batch_size) * random.uniform(0.85, 1.15)
                    memory = memory_base * (1 + 0.2 * (batch_size - 1)) * random.uniform(0.9, 1.1)
                    
                    data.append({
                        "model_name": f"example_{model_type}_model",
                        "model_type": model_type,
                        "hardware": hardware,
                        "batch_size": batch_size,
                        "throughput": throughput,
                        "latency": latency,
                        "memory": memory
                    })
        
        return pd.DataFrame(data)
    
    def _initialize_prediction_model(self) -> None:
        """Initialize a prediction model for uncertainty estimation."""
        if not SKLEARN_AVAILABLE or self.data is None or len(self.data) < 10:
            return
            
        try:
            # Prepare features: model_type, hardware, batch_size
            X = pd.get_dummies(self.data[["model_type", "hardware", "batch_size"]])
            
            # Target: throughput (we'll train separate models for each metric later)
            y = self.data["throughput"]
            
            # Initialize and train a model
            self.prediction_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=0
            )
            
            # Train model
            self.prediction_model.fit(X, y)
            
            logger.info("Initialized prediction model for uncertainty estimation")
        except Exception as e:
            logger.warning(f"Failed to initialize prediction model: {e}")
            self.prediction_model = None
    
    def recommend_configurations(self, budget: int = 10) -> List[Dict[str, Any]]:
        """
        Recommend high-value configurations to benchmark next.
        
        Args:
            budget: Number of configurations to recommend
            
        Returns:
            List of configurations with expected information gain
        """
        logger.info(f"Generating recommendations with budget {budget}")
        
        if not SKLEARN_AVAILABLE or self.prediction_model is None:
            # In simulation mode, just return some random unexplored configurations
            return self._simulated_recommendations(budget)
            
        # Use actual ML-based active learning algorithm to select configurations
        return self._active_learning_recommendations(budget)
    
    def _active_learning_recommendations(self, budget: int) -> List[Dict[str, Any]]:
        """
        Generate recommendations using active learning strategies.
        
        This method uses uncertainty sampling, expected model change, and
        density-weighted approaches to identify high-value configurations.
        
        Args:
            budget: Number of configurations to recommend
            
        Returns:
            List of configurations with expected information gain
        """
        # Filter to unexplored configurations
        unexplored = []
        for config in self.all_configs:
            config_key = (config["model_type"], config["hardware"], config["batch_size"])
            if config_key not in self.explored_configs:
                unexplored.append(config)
        
        # If we've explored everything, just return random configurations
        if not unexplored:
            import random
            selected = random.sample(self.all_configs, min(budget, len(self.all_configs)))
            for config in selected:
                config["expected_information_gain"] = 0.5
                config["selection_method"] = "random (all explored)"
            return selected
        
        # Create feature matrix for unexplored configurations
        unexplored_df = pd.DataFrame(unexplored)
        
        # One-hot encode features
        categorical_features = ["model_type", "hardware"]
        numerical_features = ["batch_size"]
        
        # Create dummy variables for categorical features
        X_unexplored = pd.get_dummies(unexplored_df[categorical_features + numerical_features])
        
        # Ensure columns match training data
        X_explored = pd.get_dummies(self.data[categorical_features + numerical_features])
        
        # Align column names
        missing_cols = set(X_explored.columns) - set(X_unexplored.columns)
        for col in missing_cols:
            X_unexplored[col] = 0
        X_unexplored = X_unexplored[X_explored.columns]
        
        # Method 1: Uncertainty Sampling
        # Get uncertainty estimates from the model (prediction variance)
        
        # For gradient boosting, we'll use the standard deviation of individual tree predictions
        # as a measure of uncertainty
        y_pred = np.zeros((X_unexplored.shape[0], self.prediction_model.n_estimators))
        
        # Get predictions from individual estimators
        for i, estimator in enumerate(self.prediction_model.estimators_):
            y_pred[:, i] = estimator[0].predict(X_unexplored)
        
        # Calculate uncertainty as standard deviation of predictions
        uncertainty = np.std(y_pred, axis=1)
        
        # Method 2: Density-weighted approach
        # Calculate distance to explored configurations
        scaler = StandardScaler()
        X_explored_scaled = scaler.fit_transform(X_explored)
        X_unexplored_scaled = scaler.transform(X_unexplored)
        
        # Use k-nearest neighbors to find distance to closest explored configurations
        k = min(5, len(X_explored))
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X_explored_scaled)
        
        # Get distances to k nearest neighbors
        distances, _ = knn.kneighbors(X_unexplored_scaled)
        
        # Average distance to k nearest neighbors
        avg_distances = np.mean(distances, axis=1)
        
        # Normalize to [0, 1]
        if avg_distances.max() > avg_distances.min():
            normalized_distances = (avg_distances - avg_distances.min()) / (avg_distances.max() - avg_distances.min())
        else:
            normalized_distances = np.ones_like(avg_distances) * 0.5
        
        # Combine uncertainty and diversity
        # Higher values indicate higher expected information gain
        information_gain = 0.7 * uncertainty / uncertainty.max() + 0.3 * normalized_distances
        
        # Add information gain to configurations
        for i, config in enumerate(unexplored):
            config["expected_information_gain"] = float(information_gain[i])
            config["uncertainty"] = float(uncertainty[i])
            config["diversity"] = float(normalized_distances[i])
            config["selection_method"] = "active_learning"
        
        # Sort by expected information gain (descending)
        unexplored.sort(key=lambda x: x["expected_information_gain"], reverse=True)
        
        # Return top configurations up to budget
        return unexplored[:budget]
    
    def _simulated_recommendations(self, budget: int) -> List[Dict[str, Any]]:
        """Generate simulated recommendations for high-value configurations."""
        # Filter to unexplored configurations
        unexplored = []
        for config in self.all_configs:
            config_key = (config["model_type"], config["hardware"], config["batch_size"])
            if config_key not in self.explored_configs:
                unexplored.append(config)
        
        # If we've explored everything, just return random configurations
        if not unexplored:
            import random
            selected = random.sample(self.all_configs, min(budget, len(self.all_configs)))
            for config in selected:
                config["expected_information_gain"] = 0.5
                config["selection_method"] = "random (all explored)"
            return selected
        
        # Calculate "information gain" for each configuration
        for config in unexplored:
            # Simulate expected information gain based on configuration properties
            model_factor = {
                "text_embedding": 0.4, 
                "text_generation": 0.8,
                "vision": 0.6,
                "audio": 0.7,
                "multimodal": 0.9
            }.get(config["model_type"], 0.5)
            
            hw_factor = {
                "cpu": 0.3,
                "cuda": 0.6,
                "rocm": 0.7,
                "mps": 0.8,
                "openvino": 0.7,
                "qnn": 0.9,
                "webnn": 0.8,
                "webgpu": 0.8
            }.get(config["hardware"], 0.5)
            
            # Batch size scaling factor (logarithmic)
            batch_factor = 0.5 + (0.2 * np.log2(config["batch_size"]) / np.log2(64))
            
            # Combined factor
            combined_factor = model_factor * hw_factor * batch_factor
            
            # Add some randomness
            import random
            random.seed(hash(f"{config['model_type']}_{config['hardware']}_{config['batch_size']}"))
            randomness = random.uniform(0.8, 1.2)
            
            # Calculate information gain
            info_gain = combined_factor * randomness
            
            # Add to config
            config["expected_information_gain"] = info_gain
            config["selection_method"] = "simulated"
        
        # Sort by expected information gain (descending)
        unexplored.sort(key=lambda x: x["expected_information_gain"], reverse=True)
        
        # Return top configurations up to budget
        return unexplored[:budget]
    
    def update_with_benchmark_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Update the active learning system with new benchmark results.
        
        Args:
            results: List of benchmark results
        """
        if not results:
            return
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Ensure required columns are present
        required_columns = ["model_type", "hardware", "batch_size", "throughput"]
        if not all(col in results_df.columns for col in required_columns):
            logger.error("Missing required columns in benchmark results")
            return
        
        # Append to existing data
        if self.data is None:
            self.data = results_df
        else:
            self.data = pd.concat([self.data, results_df], ignore_index=True)
        
        # Update explored configurations
        for _, row in results_df.iterrows():
            config_key = (row["model_type"], row["hardware"], row["batch_size"])
            self.explored_configs.add(config_key)
        
        # Re-initialize prediction model with updated data
        if SKLEARN_AVAILABLE and len(self.data) > 10:
            self._initialize_prediction_model()
            
        logger.info(f"Updated active learning system with {len(results)} new benchmark results")
    
    def save_state(self, output_file: str) -> bool:
        """
        Save the current state of the active learning system.
        
        Args:
            output_file: Path to output file
            
        Returns:
            Success flag
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Save data
            if self.data is not None:
                self.data.to_csv(output_file, index=False)
                
            logger.info(f"Saved active learning state to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save active learning state: {e}")
            return False