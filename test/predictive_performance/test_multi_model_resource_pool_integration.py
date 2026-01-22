#!/usr/bin/env python3
"""
Test script for the Multi-Model Resource Pool Integration module.

This script tests the integration between the multi-model execution predictor
and the Web Resource Pool, including strategy execution, empirical validation,
and adaptive optimization.
"""

import os
import sys
import unittest
import time
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import tempfile
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
try:
    from predictive_performance.multi_model_resource_pool_integration import MultiModelResourcePoolIntegration
    from predictive_performance.multi_model_execution import MultiModelPredictor
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure the necessary modules are available")
    raise


class TestMultiModelResourcePoolIntegration(unittest.TestCase):
    """Test cases for the Multi-Model Resource Pool Integration module."""
    
    def setUp(self):
        """Set up before each test."""
        # Create mock predictor
        self.mock_predictor = MagicMock()
        self.mock_predictor.predict_multi_model_performance.return_value = {
            "total_metrics": {
                "combined_throughput": 100.0,
                "combined_latency": 50.0,
                "combined_memory": 2000.0
            },
            "execution_schedule": {
                "timeline": [{"model": "model1", "start_time": 0, "end_time": 50}],
                "total_execution_time": 50.0
            },
            "execution_strategy": "parallel"
        }
        self.mock_predictor.recommend_execution_strategy.return_value = {
            "recommended_strategy": "parallel",
            "best_prediction": {
                "total_metrics": {
                    "combined_throughput": 100.0,
                    "combined_latency": 50.0,
                    "combined_memory": 2000.0
                },
                "execution_schedule": {
                    "timeline": [{"model": "model1", "start_time": 0, "end_time": 50}],
                    "total_execution_time": 50.0
                },
                "execution_strategy": "parallel"
            }
        }
        
        # Create mock resource pool
        self.mock_resource_pool = MagicMock()
        self.mock_resource_pool.initialize.return_value = True
        self.mock_resource_pool.close.return_value = True
        self.mock_resource_pool.get_model.return_value = MagicMock()
        self.mock_resource_pool.execute_concurrent.return_value = [{"success": True, "result": [1, 2, 3]}]
        self.mock_resource_pool.get_metrics.return_value = {
            "base_metrics": {
                "peak_memory_usage": 1800.0
            }
        }
        
        # Create test model configurations
        self.model_configs = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
            {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1}
        ]
        
        # Create integration instance with mocks
        self.integration = MultiModelResourcePoolIntegration(
            predictor=self.mock_predictor,
            resource_pool=self.mock_resource_pool,
            max_connections=2,
            enable_empirical_validation=True,
            validation_interval=1,  # Use 1 for testing
            prediction_refinement=False,  # Disable for testing
            enable_adaptive_optimization=True,
            verbose=True
        )
        
        # Initialize
        self.integration.initialize()
    
    @patch('predictive_performance.multi_model_resource_pool_integration.time')
    def test_initialization(self, mock_time):
        """Test that the integration initializes correctly."""
        # Setup mock time
        mock_time.time.return_value = 1234567890
        
        # Create fresh instance
        integration = MultiModelResourcePoolIntegration(
            predictor=self.mock_predictor,
            resource_pool=self.mock_resource_pool
        )
        
        # Initialize
        success = integration.initialize()
        
        # Verify
        self.assertTrue(success)
        self.assertTrue(integration.initialized)
        self.mock_resource_pool.initialize.assert_called_once()
    
    @patch('predictive_performance.multi_model_resource_pool_integration.time')
    def test_execute_with_strategy_specified(self, mock_time):
        """Test execution with a specified strategy."""
        # Setup mock time
        mock_time.time.side_effect = [1000, 1010]  # Start time, end time
        
        # Set up resource pool response for actual execution
        self.mock_resource_pool.execute_concurrent.return_value = [
            {"success": True, "result": [1, 2, 3]},
            {"success": True, "result": [4, 5, 6]}
        ]
        
        # Execute with specified strategy
        result = self.integration.execute_with_strategy(
            model_configs=self.model_configs,
            hardware_platform="webgpu",
            execution_strategy="parallel",
            optimization_goal="latency"
        )
        
        # Verify predictor was called for prediction
        self.mock_predictor.predict_multi_model_performance.assert_called_with(
            model_configs=self.model_configs,
            hardware_platform="webgpu",
            execution_strategy="parallel"
        )
        
        # Verify execution was performed with resource pool
        self.assertTrue(self.mock_resource_pool.execute_concurrent.called)
        
        # Verify result structure
        self.assertIn("success", result)
        self.assertIn("execution_strategy", result)
        self.assertIn("predicted_throughput", result)
        self.assertIn("predicted_latency", result)
        self.assertIn("predicted_memory", result)
        self.assertIn("actual_throughput", result)
        self.assertIn("actual_latency", result)
        self.assertIn("actual_memory", result)
        
        # Check actual values
        self.assertEqual(result["execution_strategy"], "parallel")
        self.assertEqual(result["predicted_throughput"], 100.0)
        self.assertEqual(result["predicted_latency"], 50.0)
        self.assertEqual(result["predicted_memory"], 2000.0)
    
    @patch('predictive_performance.multi_model_resource_pool_integration.time')
    def test_execute_with_auto_strategy(self, mock_time):
        """Test execution with automatic strategy selection."""
        # Setup mock time
        mock_time.time.side_effect = [1000, 1010]  # Start time, end time
        
        # Execute with automatic strategy
        result = self.integration.execute_with_strategy(
            model_configs=self.model_configs,
            hardware_platform="webgpu",
            execution_strategy=None,  # Auto-select
            optimization_goal="throughput"
        )
        
        # Verify recommendation was requested
        self.mock_predictor.recommend_execution_strategy.assert_called_with(
            model_configs=self.model_configs,
            hardware_platform="webgpu",
            optimization_goal="throughput"
        )
        
        # Verify execution was performed
        self.assertTrue(self.mock_resource_pool.execute_concurrent.called)
        
        # Verify result structure
        self.assertIn("success", result)
        self.assertIn("execution_strategy", result)
        self.assertEqual(result["execution_strategy"], "parallel")  # From mock recommendation
    
    @patch('predictive_performance.multi_model_resource_pool_integration.time')
    def test_compare_strategies(self, mock_time):
        """Test comparing different execution strategies."""
        # Setup mock time
        mock_time.time.side_effect = [1000, 1010, 1020, 1030, 1040]
        
        # Configure mock for different strategies
        def get_metrics_for_strategy(model_configs, hardware_platform, execution_strategy, **kwargs):
            metrics = {
                "parallel": {"combined_throughput": 100.0, "combined_latency": 30.0, "combined_memory": 2000.0},
                "sequential": {"combined_throughput": 50.0, "combined_latency": 60.0, "combined_memory": 1000.0},
                "batched": {"combined_throughput": 80.0, "combined_latency": 40.0, "combined_memory": 1500.0}
            }
            
            return {
                "total_metrics": metrics[execution_strategy],
                "execution_schedule": {
                    "timeline": [{"model": "model1", "start_time": 0, "end_time": 50}],
                    "total_execution_time": metrics[execution_strategy]["combined_latency"]
                },
                "execution_strategy": execution_strategy
            }
        
        self.mock_predictor.predict_multi_model_performance.side_effect = get_metrics_for_strategy
        
        # Configure mock for different concurrent executions
        def execute_concurrent_per_strategy(model_inputs):
            # Start with default successful result
            result = [{"success": True, "result": [1, 2, 3]} for _ in range(len(model_inputs))]
            return result
        
        self.mock_resource_pool.execute_concurrent.side_effect = execute_concurrent_per_strategy
        
        # Compare strategies
        comparison = self.integration.compare_strategies(
            model_configs=self.model_configs,
            hardware_platform="webgpu",
            optimization_goal="throughput"
        )
        
        # Verify prediction was called for each strategy
        self.assertEqual(self.mock_predictor.predict_multi_model_performance.call_count, 3)  # 3 strategies
        
        # Verify comparison result
        self.assertIn("best_strategy", comparison)
        self.assertIn("recommended_strategy", comparison)
        self.assertIn("strategy_results", comparison)
        self.assertIn("optimization_impact", comparison)
        
        # For throughput optimization, "parallel" should be best
        self.assertEqual(comparison["best_strategy"], "parallel")
    
    def test_get_validation_metrics(self):
        """Test retrieving validation metrics."""
        # Run a few executions to generate metrics
        for _ in range(3):
            self.integration.execute_with_strategy(
                model_configs=self.model_configs,
                hardware_platform="webgpu",
                execution_strategy="parallel"
            )
        
        # Get metrics
        metrics = self.integration.get_validation_metrics()
        
        # Verify metrics structure
        self.assertIn("validation_count", metrics)
        self.assertIn("execution_count", metrics)
        self.assertIn("error_rates", metrics)
        
        # Check counts
        self.assertEqual(metrics["validation_count"], 3)  # All executions validated due to interval=1
        self.assertEqual(metrics["execution_count"], 3)
    
    def test_update_strategy_configuration(self):
        """Test updating strategy configuration."""
        # Get original configuration
        original_config = self.integration.strategy_configuration["webgpu"].copy()
        
        # Update with custom configuration
        new_config = {
            "parallel_threshold": 5,
            "sequential_threshold": 10,
            "batching_size": 6
        }
        
        updated_config = self.integration.update_strategy_configuration("webgpu", new_config)
        
        # Verify configuration was updated
        self.assertEqual(updated_config["parallel_threshold"], 5)
        self.assertEqual(updated_config["sequential_threshold"], 10)
        self.assertEqual(updated_config["batching_size"], 6)
        
        # Verify original fields are preserved
        self.assertEqual(updated_config["memory_threshold"], original_config["memory_threshold"])
    
    def test_get_adaptive_configuration(self):
        """Test getting adaptive configuration."""
        # Run executions to generate validation metrics
        for _ in range(5):
            # Mock validation data directly to avoid full execution
            validation_record = {
                "timestamp": time.time(),
                "model_count": 2,
                "hardware_platform": "webgpu",
                "execution_strategy": "parallel",
                "predicted_throughput": 100.0,
                "actual_throughput": 90.0,
                "predicted_latency": 50.0,
                "actual_latency": 55.0,
                "predicted_memory": 2000.0,
                "actual_memory": 1800.0,
                "throughput_error": 0.1,
                "latency_error": 0.1,
                "memory_error": 0.1,
                "optimization_goal": "latency"
            }
            
            self.integration.validation_metrics["predicted_vs_actual"].append(validation_record)
            self.integration.validation_metrics["error_rates"]["throughput"].append(0.1)
            self.integration.validation_metrics["error_rates"]["latency"].append(0.1)
            self.integration.validation_metrics["error_rates"]["memory"].append(0.1)
        
        self.integration.validation_metrics["validation_count"] = 5
        
        # Get adaptive configuration
        adaptive_config = self.integration.get_adaptive_configuration("webgpu")
        
        # Verify it returns a configuration
        self.assertIsInstance(adaptive_config, dict)
        self.assertIn("parallel_threshold", adaptive_config)
        self.assertIn("sequential_threshold", adaptive_config)
        self.assertIn("batching_size", adaptive_config)
        self.assertIn("memory_threshold", adaptive_config)
    
    def test_close(self):
        """Test closing the integration."""
        # Close the integration
        success = self.integration.close()
        
        # Verify resource pool was closed
        self.mock_resource_pool.close.assert_called_once()
        
        # Verify success
        self.assertTrue(success)


# Integration test with real components (using in-memory simulation)
class TestMultiModelResourcePoolRealIntegration(unittest.TestCase):
    """Integration tests with real components in simulation mode."""
    
    def setUp(self):
        """Set up before each test."""
        # Create real predictor
        self.predictor = MultiModelPredictor(verbose=True)
        
        # Create integration without resource pool (will use simulation)
        self.integration = MultiModelResourcePoolIntegration(
            predictor=self.predictor,
            resource_pool=None,  # No resource pool, will use simulation
            enable_empirical_validation=True,
            validation_interval=1,
            enable_adaptive_optimization=True,
            verbose=True
        )
        
        # Initialize
        self.integration.initialize()
        
        # Create test model configurations
        self.model_configs = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
            {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1}
        ]
    
    def test_execute_with_strategy_simulation(self):
        """Test execution with strategy in simulation mode."""
        # Execute with specified strategy
        result = self.integration.execute_with_strategy(
            model_configs=self.model_configs,
            hardware_platform="webgpu",
            execution_strategy="parallel",
            optimization_goal="latency"
        )
        
        # Verify result
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertEqual(result["execution_strategy"], "parallel")
        self.assertIn("simulated", result)
        self.assertTrue(result["simulated"])
        
        # Verify metrics were predicted
        self.assertIn("predicted_throughput", result)
        self.assertIn("predicted_latency", result)
        self.assertIn("predicted_memory", result)
        
        # Verify actual values were simulated
        self.assertGreater(result["actual_throughput"], 0)
        self.assertGreater(result["actual_latency"], 0)
        self.assertGreater(result["actual_memory"], 0)
    
    def test_compare_strategies_simulation(self):
        """Test comparing strategies in simulation mode."""
        # Compare strategies
        comparison = self.integration.compare_strategies(
            model_configs=self.model_configs,
            hardware_platform="webgpu",
            optimization_goal="throughput"
        )
        
        # Verify comparison result
        self.assertIn("best_strategy", comparison)
        self.assertIn("recommended_strategy", comparison)
        self.assertIn("strategy_results", comparison)
        
        # Verify all three strategies were compared
        self.assertEqual(len(comparison["strategy_results"]), 4)  # 3 strategies + recommended
        
        # Verify optimization impact was calculated
        self.assertIn("optimization_impact", comparison)
        self.assertIn("improvement_percent", comparison["optimization_impact"])


# Run the tests
if __name__ == "__main__":
    unittest.main()