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
    
    Key strategies implemented:
    1. Uncertainty Sampling: Identifies configurations with high prediction uncertainty
    2. Expected Model Change: Estimates how much a new data point would change the model
    3. Diversity-Weighted Approach: Ensures coverage of different areas of the feature space
    4. Information Gain Calculation: Combines multiple signals for optimal selection
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
                    
        latency_base = {}}
        "text_embedding": 10,
        "text_generation": 100,
        "vision": 30,
        "audio": 200,
        "multimodal": 300
        }.get()model_type, 50)
                    
        memory_base = {}}
        "text_embedding": 1024,
        "text_generation": 4096,
        "vision": 2048,
        "audio": 3072,
        "multimodal": 6144
        }.get()model_type, 2048)
                    
                    # Hardware factors
        hw_factor = {}}
        "cpu": 1.0,
        "cuda": 8.0,
        "rocm": 7.5,
        "mps": 5.0,
        "openvino": 3.5,
        "qnn": 2.5,
        "webnn": 2.0,
        "webgpu": 3.0
        }.get()hardware, 1.0)
                    
                    # Add some randomness
        import random
        random.seed((hash()f"{}}model_type}_{}}hardware}_{}}batch_size}"(
                    
        throughput = throughput_base * hw_factor * ()batch_size ** 0.7) * random.uniform()0.85, 1.15)
        latency = latency_base / hw_factor * ()1 + 0.1 * batch_size) * random.uniform()0.85, 1.15)
        memory = memory_base * ((1 + 0.2 * ()batch_size - 1( * random.uniform()0.9, 1.1)
                    
        data.append(){}}
        "model_name": f"example_{}}model_type}_model",
        "model_type": model_type,
        "hardware": hardware,
        "batch_size": batch_size,
        "throughput": throughput,
        "latency": latency,
        "memory": memory
        })
        
            return pd.DataFrame()data)
    
    def _initialize_prediction_model()self) -> None:
        """Initialize a prediction model for uncertainty estimation."""
        if not SKLEARN_AVAILABLE or self.data is None or len()self.data) < 10:
        return
            
        try:
            # Prepare features: model_type, hardware, batch_size
            X = pd.get_dummies()self.data[["model_type", "hardware", "batch_size"]])
            ,
            # Target: throughput ()we'll train separate models for each metric later)
            y = self.data["throughput"]
            ,
            # Initialize and train a model
            self.prediction_model = GradientBoostingRegressor()
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=0
            )
            
            # Train model
            self.prediction_model.fit()X, y)
            
            logger.info()"Initialized prediction model for uncertainty estimation")
        except Exception as e:
            logger.warning()f"Failed to initialize prediction model: {}}e}")
            self.prediction_model = None
    
            def recommend_configurations()self, budget: int = 10) -> List[Dict[str, Any]]:,,,
            """
            Recommend high-value configurations to benchmark next.
        
        Args:
            budget: Number of configurations to recommend
            
        Returns:
            List of configurations with expected information gain
            """
            logger.info()f"Generating recommendations with budget {}}budget}")
        
        if not SKLEARN_AVAILABLE or self.prediction_model is None:
            # In simulation mode, just return some random unexplored configurations
            return self._simulated_recommendations()budget)
            
        # Use actual ML-based active learning algorithm to select configurations
            return self._active_learning_recommendations()budget)
    
            def _active_learning_recommendations()self, budget: int) -> List[Dict[str, Any]]:,,,
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
            unexplored = [,,
        for config in self.all_configs:
            config_key = ()config["model_type"], config["hardware"], config["batch_size"]),,
            if config_key not in self.explored_configs:
                unexplored.append()config)
        
        # If we've explored everything, just return random configurations
        if not unexplored:
            import random
            selected = random.sample((self.all_configs, min((budget, len()self.all_configs)(
            for config in selected:
                config["expected_information_gain"] = 0.5,,
                config["selection_method"] = "random ()all explored)",,
            return selected
        
        # Create feature matrix for unexplored configurations
            unexplored_df = pd.DataFrame()unexplored)
        
        # One-hot encode features
            categorical_features = ["model_type", "hardware"],
            numerical_features = ["batch_size"]
            ,
        # Create dummy variables for categorical features
            X_unexplored = pd.get_dummies()unexplored_df[categorical_features + numerical_features])
            ,,
        # Ensure columns match training data
            X_explored = pd.get_dummies()self.data[categorical_features + numerical_features])
            ,,
        # Align column names
            missing_cols = set()X_explored.columns) - set()X_unexplored.columns)
        for col in missing_cols:
            X_unexplored[col] = 0,
            X_unexplored = X_unexplored[X_explored.columns]
            ,
        # Method 1: Uncertainty Sampling
        # Get uncertainty estimates from the model ()prediction variance)
        
        # For gradient boosting, we'll use the standard deviation of individual tree predictions
        # as a measure of uncertainty
            y_pred = np.zeros((()X_unexplored.shape[0], self.prediction_model.n_estimators(
            ,
        # Get predictions from individual estimators
        for i, estimator in enumerate()self.prediction_model.estimators_):
            y_pred[:, i] = estimator[0].predict()X_unexplored)
            ,
        # Calculate uncertainty as standard deviation of predictions
            uncertainty = np.std()y_pred, axis=1)
        
        # Method 2: Density-weighted approach
        # Calculate distance to explored configurations
            scaler = StandardScaler()(
            X_explored_scaled = scaler.fit_transform()X_explored)
            X_unexplored_scaled = scaler.transform()X_unexplored)
        
        # Use k-nearest neighbors to find distance to closest explored configurations
            k = min((5, len()X_explored(
            knn = NearestNeighbors()n_neighbors=k)
            knn.fit()X_explored_scaled)
        
        # Get distances to k nearest neighbors
            distances, _ = knn.kneighbors()X_unexplored_scaled)
        
        # Average distance to k nearest neighbors
            avg_distances = np.mean()distances, axis=1)
        
        # Normalize to [0, 1],
        if avg_distances.max()( > avg_distances.min()(:
            normalized_distances = ((avg_distances - avg_distances.min()( / ((avg_distances.max()( - avg_distances.min()(
        else:
            normalized_distances = np.ones_like()avg_distances) * 0.5
        
        # Combine uncertainty and diversity
        # Higher values indicate higher expected information gain
            information_gain = 0.7 * uncertainty / uncertainty.max()( + 0.3 * normalized_distances
        
        # Add information gain to configurations
        for i, config in enumerate()unexplored):
            config["expected_information_gain"] = float()information_gain[i]),
            config["uncertainty"] = float()uncertainty[i]),
            config["diversity"] = float()normalized_distances[i]),
            config["selection_method"] = "active_learning"
            ,
        # Sort by expected information gain ()descending)
            unexplored.sort()key=lambda x: x["expected_information_gain"], reverse=True)
            ,,
        # Return top configurations up to budget
            return unexplored[:budget]
            ,,
            def _simulated_recommendations()self, budget: int) -> List[Dict[str, Any]]:,,,
            """Generate simulated recommendations for high-value configurations."""
        # Filter to unexplored configurations
            unexplored = [,,
        for config in self.all_configs:
            config_key = ()config["model_type"], config["hardware"], config["batch_size"]),,
            if config_key not in self.explored_configs:
                unexplored.append()config)
        
        # If we've explored everything, just return random configurations
        if not unexplored:
            import random
            selected = random.sample((self.all_configs, min((budget, len()self.all_configs)(
            for config in selected:
                config["expected_information_gain"] = 0.5,,
                config["selection_method"] = "random ()all explored)",,
            return selected
        
        # Calculate "information gain" for each configuration
        for config in unexplored:
            # Simulate expected information gain based on configuration properties
            model_factor = {}}
            "text_embedding": 0.4,
            "text_generation": 0.8,
            "vision": 0.6,
            "audio": 0.7,
            "multimodal": 0.9
            }.get()config["model_type"], 0.5)
            ,
            hw_factor = {}}
            "cpu": 0.3,
            "cuda": 0.6,
            "rocm": 0.7,
            "mps": 0.8,
            "openvino": 0.7,
            "qnn": 0.9,
            "webnn": 0.8,
            "webgpu": 0.8
            }.get()config["hardware"], 0.5)
            ,
            # Batch size scaling factor ()logarithmic)
            batch_factor = 0.5 + ((0.2 * np.log2()config["batch_size"]) / np.log2()64(
            ,
            # Combined factor
            combined_factor = model_factor * hw_factor * batch_factor
            
            # Add some randomness
            import random
            random.seed((hash()f"{}}config['model_type']}_{}}config['hardware']}_{}}config['batch_size']}"(,
            randomness = random.uniform()0.8, 1.2)
            
            # Calculate information gain
            info_gain = combined_factor * randomness
            
            # Add to config
            config["expected_information_gain"] = info_gain,
            config["selection_method"] = "simulated"
            ,
        # Sort by expected information gain ()descending)
            unexplored.sort()key=lambda x: x["expected_information_gain"], reverse=True)
            ,,
        # Return top configurations up to budget
            return unexplored[:budget]
            ,,
            def update_with_benchmark_results()self, results: List[Dict[str, Any]]) -> None:,
            """
            Update the active learning system with new benchmark results.
        
        Args:
            results: List of benchmark results
            """
        if not results:
            return
            
        # Convert results to DataFrame
            results_df = pd.DataFrame()results)
        
        # Ensure required columns are present
            required_columns = ["model_type", "hardware", "batch_size", "throughput"],
        if not all()col in results_df.columns for col in required_columns):
            logger.error()"Missing required columns in benchmark results")
            return
        
        # Append to existing data
        if self.data is None:
            self.data = results_df
        else:
            self.data = pd.concat()[self.data, results_df], ignore_index=True)
            ,
        # Update explored configurations
        for _, row in results_df.iterrows()(:
            config_key = ()row["model_type"], row["hardware"], row["batch_size"]),,
            self.explored_configs.add()config_key)
        
        # Re-initialize prediction model with updated data
        if SKLEARN_AVAILABLE and len()self.data) > 10:
            self._initialize_prediction_model()(
            
            logger.info((f"Updated active learning system with {}}len()results)} new benchmark results")
    
    def save_state()self, output_file: str) -> bool:
        """
        Save the current state of the active learning system.
        
        Args:
            output_file: Path to output file
            
        Returns:
            Success flag
            """
        try:
            # Create output directory if it doesn't exist
            os.makedirs((os.path.dirname((os.path.abspath()output_file), exist_ok=True)
            
            # Save data:
            if self.data is not None:
                self.data.to_csv()output_file, index=False)
                
                logger.info()f"Saved active learning state to {}}output_file}")
            return True
        except Exception as e:
            logger.error()f"Failed to save active learning state: {}}e}")
            return False
            
    def suggest_test_batch(self, configurations, batch_size=10, ensure_diversity=True, 
                         hardware_constraints=None, hardware_availability=None,
                         diversity_weight=0.5):
        """
        Generate an optimized batch of test configurations for benchmarking.
        
        This method takes a list of prioritized configurations and generates a batch
        that maximizes expected information gain while ensuring diversity and respecting
        hardware constraints. It balances exploration vs. exploitation to improve 
        prediction accuracy efficiently.
        
        Args:
            configurations: DataFrame or list of configuration dictionaries
            batch_size: Maximum number of configurations to include in the batch
            ensure_diversity: Whether to ensure diversity in the selected batch
            hardware_constraints: Dictionary mapping hardware types to maximum count in batch
            hardware_availability: Dictionary mapping hardware types to availability factor (0-1)
            diversity_weight: Weight to give diversity vs. information gain (0-1)
            
        Returns:
            DataFrame of selected configurations for the test batch
        """
        logger.info(f"Generating test batch with size {batch_size}, diversity={ensure_diversity}")
        
        # Convert to DataFrame if needed
        if isinstance(configurations, list):
            configs_df = pd.DataFrame(configurations)
        else:
            configs_df = configurations.copy()
            
        # Check if we have enough configurations
        if len(configs_df) <= batch_size:
            logger.info(f"Not enough configurations, returning all {len(configs_df)} available")
            return configs_df
            
        # Use different columns depending on which scoring system we're dealing with
        if "combined_score" in configs_df.columns:
            score_column = "combined_score"
        elif "adjusted_score" in configs_df.columns:
            score_column = "adjusted_score"
        elif "expected_information_gain" in configs_df.columns:
            score_column = "expected_information_gain"
        else:
            # If no score column exists, add a default one
            logger.warning("No score column found, using equal weights for all configurations")
            configs_df["score"] = 1.0
            score_column = "score"
            
        # Apply hardware availability constraints if provided
        if hardware_availability is not None:
            logger.info("Applying hardware availability constraints")
            configs_df = self._apply_hardware_availability(configs_df, 
                                                          hardware_availability, 
                                                          score_column)
            
        # If diversity is not required, simply return the top configurations by score
        if not ensure_diversity:
            sorted_configs = configs_df.sort_values(by=score_column, ascending=False)
            
            # Apply hardware constraints if provided
            if hardware_constraints is not None:
                batch = self._apply_hardware_constraints(sorted_configs, 
                                                       hardware_constraints, 
                                                       batch_size)
            else:
                batch = sorted_configs.head(batch_size)
                
            logger.info(f"Generated non-diverse batch with {len(batch)} configurations")
            return batch
            
        # For diversity-aware selection, we'll select configurations one by one
        logger.info("Using diversity-aware selection")
        return self._diversity_sampling(configs_df, 
                                       score_column, 
                                       batch_size, 
                                       diversity_weight, 
                                       hardware_constraints)
    
    def _apply_hardware_availability(self, configs_df, hardware_availability, score_column):
        """
        Adjust scores based on hardware availability.
        
        Args:
            configs_df: DataFrame of configurations
            hardware_availability: Dictionary mapping hardware types to availability factor (0-1)
            score_column: Name of the column containing scores
            
        Returns:
            DataFrame with adjusted scores
        """
        # Create a copy so we don't modify the original
        adjusted_df = configs_df.copy()
        
        # Hardware column might be called 'hardware' or 'hardware_platform'
        hardware_column = 'hardware' if 'hardware' in adjusted_df.columns else 'hardware_platform'
        
        # Adjust scores based on hardware availability
        for hw_type, availability in hardware_availability.items():
            # Find configurations with this hardware type
            mask = adjusted_df[hardware_column] == hw_type
            
            # Adjust scores
            adjusted_df.loc[mask, score_column] = adjusted_df.loc[mask, score_column] * availability
            
        return adjusted_df
    
    def _apply_hardware_constraints(self, configs_df, hardware_constraints, batch_size):
        """
        Apply hardware constraints to selection.
        
        Args:
            configs_df: DataFrame of configurations sorted by score
            hardware_constraints: Dictionary mapping hardware types to maximum count in batch
            batch_size: Maximum batch size
            
        Returns:
            DataFrame of selected configurations respecting hardware constraints
        """
        # Hardware column might be called 'hardware' or 'hardware_platform'
        hardware_column = 'hardware' if 'hardware' in configs_df.columns else 'hardware_platform'
        
        # Initialize empty batch and hardware counts
        batch = []
        hw_counts = {hw: 0 for hw in hardware_constraints.keys()}
        total_selected = 0
        
        # Iterate through sorted configurations
        for _, config in configs_df.iterrows():
            hw_type = config[hardware_column]
            
            # Check if we've reached the hardware constraint
            if hw_type in hardware_constraints:
                if hw_counts[hw_type] >= hardware_constraints[hw_type]:
                    continue  # Skip this configuration
                    
                # Increment the hardware count
                hw_counts[hw_type] += 1
            
            # Add configuration to batch
            batch.append(config)
            total_selected += 1
            
            # Check if we've reached the batch size limit
            if total_selected >= batch_size:
                break
                
        # Convert list back to DataFrame
        return pd.DataFrame(batch)
    
    def _diversity_sampling(self, configs_df, score_column, batch_size, diversity_weight, hardware_constraints=None):
        """
        Select diverse configurations with high scores.
        
        Args:
            configs_df: DataFrame of configurations
            score_column: Name of the column containing scores
            batch_size: Maximum number of configurations to select
            diversity_weight: Weight to give diversity vs. score (0-1)
            hardware_constraints: Dictionary mapping hardware types to maximum count in batch
            
        Returns:
            DataFrame of selected diverse configurations
        """
        # Hardware column might be called 'hardware' or 'hardware_platform'
        hardware_column = 'hardware' if 'hardware' in configs_df.columns else 'hardware_platform'
        
        # Get numerical features for diversity calculation
        numeric_columns = [col for col in configs_df.columns if configs_df[col].dtype in [np.int64, np.float64]]
        categorical_columns = [col for col in configs_df.columns if col not in numeric_columns 
                              and col != score_column 
                              and col != 'uncertainty'
                              and col != 'diversity'
                              and col != 'information_gain'
                              and col != 'selection_method']
        
        # Create feature matrix for diversity calculation
        feature_df = pd.get_dummies(configs_df[categorical_columns])
        if numeric_columns:
            # Scale numeric columns
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_numeric = scaler.fit_transform(configs_df[numeric_columns])
            numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_columns)
            feature_df = pd.concat([feature_df, numeric_df], axis=1)
        
        # Convert to numpy array for faster processing
        features = feature_df.values
        scores = configs_df[score_column].values
        
        # Initialize hardware counts if constraints are provided
        hw_counts = {hw: 0 for hw in hardware_constraints.keys()} if hardware_constraints else None
        
        # Initialize selected configurations
        selected_indices = []
        remaining_indices = list(range(len(configs_df)(
        
        # Select first configuration with highest score
        best_idx = np.argmax(scores)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # If hardware constraints are provided, update the count
        if hardware_constraints:
            hw_type = configs_df.iloc[best_idx][hardware_column]
            if hw_type in hw_counts:
                hw_counts[hw_type] += 1
        
        # Select remaining configurations
        from scipy.spatial.distance import euclidean
        
        while len(selected_indices) < batch_size and remaining_indices:
            best_score = -float('inf')
            best_idx = -1
            
            for idx in remaining_indices:
                # Calculate diversity as minimum distance to already selected points
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    distance = euclidean(features[idx], features[selected_idx])
                    min_distance = min(min_distance, distance)
                
                # Normalize min_distance to [0, 1] range
                # We'll use a simple approach here, assuming distances are roughly in [0, 10] range
                norm_distance = min(min_distance / 10.0, 1.0)
                
                # Calculate combined score as weighted combination of original score and diversity
                norm_score = scores[idx] / max(scores) if max(scores) > 0 else scores[idx]
                combined_score = (1 - diversity_weight) * norm_score + diversity_weight * norm_distance
                
                # Check hardware constraints if provided
                if hardware_constraints:
                    hw_type = configs_df.iloc[idx][hardware_column]
                    if hw_type in hw_counts and hw_counts[hw_type] >= hardware_constraints[hw_type]:
                        continue  # Skip this configuration as we've reached the hardware constraint
                
                # Update best if this is better
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            # If we couldn't find a valid configuration, break
            if best_idx == -1:
                break
                
            # Add best configuration to selected
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # Update hardware count if constraints are provided
            if hardware_constraints:
                hw_type = configs_df.iloc[best_idx][hardware_column]
                if hw_type in hw_counts:
                    hw_counts[hw_type] += 1
        
        # Extract selected configurations
        selected_configs = configs_df.iloc[selected_indices].copy()
        
        # Add a column indicating selection order
        selected_configs['selection_order'] = range(1, len(selected_configs) + 1)
        
        logger.info(f"Generated diverse batch with {len(selected_configs)} configurations")
        return selected_configs
    
    def integrate_with_hardware_recommender()self, hardware_recommender, test_budget: int = 10,
                                       optimize_for: str = "throughput") -> Dict[str, Any]:
        """
        Integrate active learning with hardware recommender to prioritize tests
        that are both informative and relevant for hardware selection.
        
        This method combines active learning's ability to identify configurations
        with high uncertainty or expected information gain with the hardware 
        recommender's knowledge of hardware capabilities and constraints.
        
        Args:
            hardware_recommender: Hardware recommender instance
            test_budget: Maximum number of test configurations to recommend
            optimize_for: Metric to optimize for (throughput, latency, memory)
            
        Returns:
            Dictionary with recommended configurations and metadata
        """
        logger.info((f"Integrating active learning with hardware recommender (budget: {}}test_budget}, metric: {}}optimize_for})")
        
        # Step 1: Get high-value configurations from active learning
        high_value_configs = self.recommend_configurations()test_budget * 2)  # Get 2x budget to allow for filtering
        
        # Step 2: For each configuration, get hardware recommendations
        enhanced_configs = []
        
        for config in high_value_configs:
            try:
                # Get hardware recommendation for the configuration
                hw_recommendation = hardware_recommender.recommend_hardware()
                    model_name=config["model_name"],
                    model_type=config["model_type"],
                    batch_size=config["batch_size"],
                    optimization_metric=optimize_for,
                    power_constrained=False,
                    include_alternatives=True
                )
                
                # Check if the currently selected hardware matches the recommendation
                current_hw = config["hardware"]
                recommended_hw = hw_recommendation["recommended_hardware"]
                
                # Enhance the configuration with hardware recommendation data
                config["hardware_match"] = (current_hw == recommended_hw)
                config["recommended_hardware"] = recommended_hw
                config["hardware_score"] = hw_recommendation.get()f"{}}optimize_for}_score", 0.5)
                config["alternatives"] = hw_recommendation.get()f"alternatives", [])
                
                # Calculate combined score: 70% info gain, 30% hardware optimality
                if config["hardware_match"]:
                    # If already using recommended hardware, just use info gain
                    config["combined_score"] = config["expected_information_gain"]
                else:
                    # If not using recommended hardware, factor in potential improvement
                    # from switching to recommended hardware
                    potential_improvement = hw_recommendation.get()f"estimated_improvement", 0.2)
                    config["combined_score"] = 0.7 * config["expected_information_gain"] + 0.3 * potential_improvement
                
                enhanced_configs.append()config)
            except Exception as e:
                logger.warning()f"Error enhancing configuration {}}config['model_name']}: {}}e}")
                # Still include the original config
                config["combined_score"] = config["expected_information_gain"]
                enhanced_configs.append()config)
        
        # Step 3: Sort by combined score
        enhanced_configs.sort((key=lambda x: x.get()f"combined_score", 0), reverse=True)
        
        # Step 4: Prepare final recommendations within budget
        final_recommendations = enhanced_configs[:test_budget]
        
        # Step 5: Prepare result with metadata
        result = {}}
            "recommendations": final_recommendations,
            "total_candidates": len(high_value_configs),
            "enhanced_candidates": len(enhanced_configs),
            "final_recommendations": len(final_recommendations),
            "optimization_metric": optimize_for,
            "strategy": "integrated_active_learning",
            "timestamp": datetime.now((.isoformat((,
        }
        
        logger.info((f"Generated {}}len(final_recommendations)} integrated recommendations")
        
        return result