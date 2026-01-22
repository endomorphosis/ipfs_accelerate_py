#!/usr/bin/env python3
"""
Test script for the Model Update Pipeline module.

This script tests the core functionality of the model update pipeline, including
incremental updates, model improvement tracking, update strategies, and integration
with the Active Learning System.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import warnings
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import the model update pipeline module
from predictive_performance.model_update_pipeline import ModelUpdatePipeline

# Try to import scikit-learn for model creation
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.datasets import make_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("scikit-learn not available, some tests will be skipped")
    SKLEARN_AVAILABLE = False

# Try to import the active learning module
try:
    from predictive_performance.active_learning import ActiveLearningSystem
    ACTIVE_LEARNING_AVAILABLE = True
except ImportError:
    warnings.warn("active_learning module not available, some tests will be skipped")
    ACTIVE_LEARNING_AVAILABLE = False


class TestModelUpdatePipeline(unittest.TestCase):
    """Test cases for the Model Update Pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and models for testing."""
        # Create temporary directory for models and data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.model_dir = os.path.join(cls.temp_dir.name, "models")
        cls.data_dir = os.path.join(cls.temp_dir.name, "data")
        
        # Create directories
        os.makedirs(cls.model_dir, exist_ok=True)
        os.makedirs(cls.data_dir, exist_ok=True)
        
        # Skip further setup if scikit-learn is not available
        if not SKLEARN_AVAILABLE:
            return
        
        # Generate synthetic benchmark data
        cls.data = cls._generate_synthetic_data(n_samples=100)
        
        # Save data to file
        data_file = os.path.join(cls.data_dir, "benchmark_data.parquet")
        cls.data.to_parquet(data_file)
        
        # Create synthetic test data (new data)
        cls.test_data = cls._generate_synthetic_data(n_samples=20, shift=0.2)
        
        # Create synthetic validation data
        cls.validation_data = cls._generate_synthetic_data(n_samples=30, shift=0.1)
        
        # Create mock models
        cls.models = cls._create_mock_models(cls.data)
        
        # Create model info file
        cls._create_model_info(cls.models, cls.model_dir)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.temp_dir.cleanup()
    
    @staticmethod
    def _generate_synthetic_data(n_samples=100, shift=0):
        """Generate synthetic benchmark data."""
        # Create feature data
        model_types = ['bert', 'vit', 'whisper', 'llama']
        hardware_platforms = ['cpu', 'cuda', 'openvino', 'webgpu']
        batch_sizes = [1, 2, 4, 8]
        precision_formats = ['FP32', 'FP16', 'INT8']
        
        # Create random configurations
        np.random.seed(42)
        rows = []
        for _ in range(n_samples):
            model_type = np.random.choice(model_types)
            hardware = np.random.choice(hardware_platforms)
            batch_size = np.random.choice(batch_sizes)
            precision = np.random.choice(precision_formats)
            
            # Model type specific base values
            if model_type == 'bert':
                base_throughput = 100 + shift * 20
                base_latency = 10 - shift * 2
                base_memory = 1000 + shift * 100
            elif model_type == 'vit':
                base_throughput = 50 + shift * 10
                base_latency = 20 - shift * 4
                base_memory = 2000 + shift * 200
            elif model_type == 'whisper':
                base_throughput = 20 + shift * 5
                base_latency = 50 - shift * 10
                base_memory = 3000 + shift * 300
            else:  # llama
                base_throughput = 10 + shift * 2
                base_latency = 100 - shift * 20
                base_memory = 5000 + shift * 500
            
            # Hardware factors
            if hardware == 'cpu':
                hw_factor_throughput = 1.0
                hw_factor_latency = 1.0
                hw_factor_memory = 1.0
            elif hardware == 'cuda':
                hw_factor_throughput = 8.0
                hw_factor_latency = 0.2
                hw_factor_memory = 1.2
            elif hardware == 'openvino':
                hw_factor_throughput = 3.0
                hw_factor_latency = 0.5
                hw_factor_memory = 0.8
            else:  # webgpu
                hw_factor_throughput = 2.5
                hw_factor_latency = 0.6
                hw_factor_memory = 0.9
            
            # Batch size factor (non-linear)
            batch_factor = batch_size ** 0.7
            
            # Precision factor
            precision_factor = 1.0 if precision == 'FP32' else (1.5 if precision == 'FP16' else 2.0)
            
            # Calculate metrics with some randomness
            np.random.seed(hash(f"{model_type}_{hardware}_{batch_size}_{precision}") % 10000)
            
            throughput = base_throughput * hw_factor_throughput * batch_factor * precision_factor * np.random.uniform(0.9, 1.1)
            latency = base_latency * hw_factor_latency * (1 + 0.1 * batch_size) / precision_factor * np.random.uniform(0.9, 1.1)
            memory = base_memory * hw_factor_memory * (1 + 0.2 * (batch_size - 1)) / np.sqrt(precision_factor) * np.random.uniform(0.9, 1.1)
            
            # Add to data
            rows.append({
                'model_type': model_type,
                'hardware': hardware,
                'batch_size': batch_size,
                'precision': precision,
                'throughput': throughput,
                'latency': latency,
                'memory': memory
            })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def _create_mock_models(data):
        """Create mock prediction models using the data."""
        models = {}
        
        # Prepare features
        X = pd.get_dummies(data[['model_type', 'hardware', 'batch_size', 'precision']])
        
        # Train model for each metric
        for metric in ['throughput', 'latency', 'memory']:
            y = data[metric].values
            
            # Train a simple GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Save model
            models[metric] = model
        
        return models
    
    @classmethod
    def _create_model_info(cls, models, model_dir):
        """Create a mock model info file."""
        import json
        
        # Calculate model metrics on synthetic data
        X = pd.get_dummies(cls.data[['model_type', 'hardware', 'batch_size', 'precision']])
        
        model_metrics = {}
        for metric in ['throughput', 'latency', 'memory']:
            y = cls.data[metric].values
            y_pred = models[metric].predict(X)
            
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            model_metrics[metric] = {
                'rmse': float(rmse),
                'test_r2': float(r2),
                'mape': 0.1,
                'n_samples': len(y),
                'cv_r2_mean': 0.8
            }
        
        # Create model info
        model_info = {
            "timestamp": "2025-03-01T12:00:00.000000",
            "input_dir": cls.data_dir,
            "output_dir": model_dir,
            "training_params": {
                "test_size": 0.2,
                "random_state": 42,
                "hyperparameter_tuning": False,
                "use_ensemble": True,
                "model_complexity": "auto",
                "use_cross_validation": True,
                "cv_folds": 5,
                "feature_selection": "auto",
                "uncertainty_estimation": True,
                "n_jobs": 1
            },
            "training_time_seconds": 10.5,
            "n_samples": len(cls.data),
            "metrics_trained": ["throughput", "latency", "memory"],
            "model_metrics": model_metrics,
            "model_path": model_dir,
            "version": "1.0.0"
        }
        
        # Save the model info
        with open(os.path.join(model_dir, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Save the models
        from joblib import dump
        for metric, model in models.items():
            dump(model, os.path.join(model_dir, f"{metric}_model.joblib"))
    
    def setUp(self):
        """Set up before each test."""
        if not SKLEARN_AVAILABLE:
            self.skipTest("scikit-learn not available")
        
        # Create a ModelUpdatePipeline instance
        self.pipeline = ModelUpdatePipeline(
            model_dir=self.model_dir,
            data_dir=self.data_dir,
            metrics=['throughput', 'latency', 'memory'],
            update_strategy="incremental",
            verbose=True
        )
        
        # Mock the models in the pipeline (since _load_models won't work in tests)
        self.pipeline.models = self.models
        self.pipeline.original_models = self.models
        
        # Mock the model info
        import json
        with open(os.path.join(self.model_dir, "model_info.json"), 'r') as f:
            self.pipeline.model_info = json.load(f)
        
        # Set the data in the pipeline
        self.pipeline.data = self.data
        
        # Set feature columns
        self.pipeline.feature_columns = ['model_type', 'hardware', 'batch_size', 'precision']
        self.pipeline.target_columns = ['throughput', 'latency', 'memory']
    
    def test_initialization(self):
        """Test that the pipeline initializes correctly."""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.metrics, ['throughput', 'latency', 'memory'])
        self.assertEqual(self.pipeline.update_strategy, "incremental")
        self.assertEqual(len(self.pipeline.models), 3)
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        X = self.pipeline._extract_features(self.data)
        
        # Check shape
        self.assertGreater(X.shape[0], 0)
        self.assertGreater(X.shape[1], 0)
        
        # Should be a numpy array
        self.assertIsInstance(X, np.ndarray)
    
    def test_incremental_update(self):
        """Test incremental model update."""
        # Extract features from test data
        X_update = self.pipeline._extract_features(self.test_data)
        y_update = self.test_data['throughput'].values
        
        # Extract features from validation data
        X_val = self.pipeline._extract_features(self.validation_data)
        y_val = self.validation_data['throughput'].values
        
        # Get the current model
        model = self.pipeline.models['throughput']
        
        # Apply incremental update
        updated_model, update_info = self.pipeline._incremental_update(
            model, X_update, y_update, X_val, y_val
        )
        
        # Check that the update info contains the expected keys
        self.assertIn('rmse_before', update_info)
        self.assertIn('rmse_after', update_info)
        self.assertIn('r2_before', update_info)
        self.assertIn('r2_after', update_info)
        self.assertIn('improvement_percent', update_info)
        
        # Check that the updated model is different from the original
        self.assertIsNot(updated_model, model)
    
    def test_window_update(self):
        """Test window model update."""
        # Combine data
        combined_data = pd.concat([self.data, self.test_data], ignore_index=True)
        
        # Extract features
        X = self.pipeline._extract_features(combined_data)
        y = combined_data['throughput'].values
        
        # Extract features from validation data
        X_val = self.pipeline._extract_features(self.validation_data)
        y_val = self.validation_data['throughput'].values
        
        # Get the current model
        model = self.pipeline.models['throughput']
        
        # Apply window update
        updated_model, update_info = self.pipeline._window_update(
            model, X, y, X_val, y_val
        )
        
        # Check that the update info contains the expected keys
        self.assertIn('rmse_before', update_info)
        self.assertIn('rmse_after', update_info)
        self.assertIn('r2_before', update_info)
        self.assertIn('r2_after', update_info)
        self.assertIn('improvement_percent', update_info)
        
        # Check that the updated model is different from the original
        self.assertIsNot(updated_model, model)
    
    def test_weighted_update(self):
        """Test weighted model update."""
        # Extract features from test data
        X_update = self.pipeline._extract_features(self.test_data)
        y_update = self.test_data['throughput'].values
        
        # Extract features from validation data
        X_val = self.pipeline._extract_features(self.validation_data)
        y_val = self.validation_data['throughput'].values
        
        # Get the current model
        model = self.pipeline.models['throughput']
        
        # Apply weighted update
        updated_model, update_info = self.pipeline._weighted_update(
            model, X_update, y_update, X_val, y_val
        )
        
        # Check that the update info contains the expected keys
        self.assertIn('rmse_before', update_info)
        self.assertIn('rmse_after', update_info)
        self.assertIn('r2_before', update_info)
        self.assertIn('r2_after', update_info)
        self.assertIn('improvement_percent', update_info)
        self.assertIn('optimal_weight', update_info)
        
        # Check that the optimal weight is between 0 and 1
        self.assertGreaterEqual(update_info['optimal_weight'], 0.0)
        self.assertLessEqual(update_info['optimal_weight'], 1.0)
    
    def test_update_models(self):
        """Test updating models with new data."""
        # Update models
        update_result = self.pipeline.update_models(
            self.test_data,
            metrics=['throughput', 'latency', 'memory'],
            update_strategy='incremental'
        )
        
        # Check that the update result contains the expected keys
        self.assertIn('success', update_result)
        self.assertIn('update_record', update_result)
        self.assertIn('metric_details', update_result)
        
        # Check success flag
        self.assertTrue(update_result['success'])
        
        # Check update record
        update_record = update_result['update_record']
        self.assertIn('overall_improvement', update_record)
        self.assertIn('metrics_updated', update_record)
        self.assertIn('update_strategy', update_record)
    
    def test_evaluate_model_improvement(self):
        """Test evaluating model improvement."""
        # First update the models to create improvement
        self.pipeline.update_models(
            self.test_data,
            metrics=['throughput'],
            update_strategy='incremental'
        )
        
        # Evaluate improvement
        evaluation = self.pipeline.evaluate_model_improvement('throughput')
        
        # Check that the evaluation contains the expected keys
        self.assertIn('success', evaluation)
        self.assertIn('metric', evaluation)
        self.assertIn('original_model', evaluation)
        self.assertIn('current_model', evaluation)
        self.assertIn('improvement', evaluation)
        
        # Check success flag
        self.assertTrue(evaluation['success'])
        
        # Check improvement
        improvement = evaluation['improvement']
        self.assertIn('rmse_percent', improvement)
        self.assertIn('r2_percent', improvement)
        self.assertIn('mape_percent', improvement)
    
    def test_determine_update_need(self):
        """Test determining if models need update."""
        # Determine update need
        need_analysis = self.pipeline.determine_update_need(
            self.test_data,
            threshold=0.05
        )
        
        # Check that the analysis contains the expected keys
        self.assertIn('needs_update', need_analysis)
        self.assertIn('error_increase', need_analysis)
        self.assertIn('metric_recommendations', need_analysis)
        self.assertIn('recommended_strategy', need_analysis)
        
        # Check metric recommendations
        metric_recommendations = need_analysis['metric_recommendations']
        self.assertIn('throughput', metric_recommendations)
        self.assertIn('latency', metric_recommendations)
        self.assertIn('memory', metric_recommendations)
        
        # Check individual metric recommendation
        for metric, recommendation in metric_recommendations.items():
            self.assertIn('needs_update', recommendation)
            self.assertIn('error_increase', recommendation)
            self.assertIn('current_rmse', recommendation)
    
    @unittest.skipIf(not ACTIVE_LEARNING_AVAILABLE, "active_learning module not available")
    def test_integrate_with_active_learning(self):
        """Test integration with Active Learning System."""
        # Create an Active Learning System
        active_learning_system = ActiveLearningSystem()
        
        # Initialize with some data
        active_learning_system.update_with_benchmark_results(self.data.to_dict('records'))
        
        # Test integration
        integration_result = self.pipeline.integrate_with_active_learning(
            active_learning_system,
            self.test_data,
            sequential_rounds=1,
            batch_size=5
        )
        
        # Check that the integration result contains the expected keys
        self.assertIn('success', integration_result)
        self.assertIn('rounds', integration_result)
        self.assertIn('overall_improvement', integration_result)
        self.assertIn('round_results', integration_result)
        self.assertIn('next_batch', integration_result)
        
        # Check success flag
        self.assertTrue(integration_result['success'])
        
        # Check round results
        round_results = integration_result['round_results']
        self.assertEqual(len(round_results), 1)
        
        # Check first round result
        round_result = round_results[0]
        self.assertIn('round', round_result)
        self.assertIn('batch_size', round_result)
        self.assertIn('update_result', round_result)
        self.assertIn('improvement', round_result)
    
    def test_save_models(self):
        """Test saving updated models."""
        # First update the models
        self.pipeline.update_models(
            self.test_data,
            metrics=['throughput', 'latency', 'memory'],
            update_strategy='incremental'
        )
        
        # Mock the save_prediction_models function
        import types
        
        def mock_save_prediction_models(models, model_dir):
            return model_dir
        
        # Add mock to the pipeline
        self.pipeline._save_models_orig = self.pipeline._save_models
        self.pipeline._save_models = types.MethodType(
            lambda self: True, self.pipeline
        )
        
        # Save the models
        success = self.pipeline._save_models_orig()
        
        # Check success
        self.assertTrue(success)
        
        # Check that model info file was updated
        import json
        model_info_file = os.path.join(self.model_dir, "model_info.json")
        with open(model_info_file, 'r') as f:
            model_info = json.load(f)
        
        # Check that update history was added
        self.assertIn('update_history', model_info)


if __name__ == "__main__":
    unittest.main()