#!/usr/bin/env python3
"""
Test script for the Multi-Model Web Integration module.

This script tests the integration between the multi-model execution predictor,
the Web Resource Pool Adapter, and the Multi-Model Resource Pool Integration.
"""

import os
import sys
import unittest
import time
from unittest.mock import MagicMock, patch
import logging
from pathlib import Path
import tempfile
import warnings
import numpy as np
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import the modules to test
try:
    from predictive_performance.multi_model_resource_pool_integration import MultiModelResourcePoolIntegration
    from predictive_performance.web_resource_pool_adapter import WebResourcePoolAdapter
    from predictive_performance.multi_model_execution import MultiModelPredictor
    from predictive_performance.multi_model_empirical_validation import MultiModelEmpiricalValidator
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure the necessary modules are available")
    raise


class TestMultiModelWebIntegration(unittest.TestCase):
    """Test cases for the Multi-Model Web Integration module."""
    
    def setUp(self):
        """Set up before each test."""
        # Create mock objects
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
        self.mock_resource_pool.get_available_browsers.return_value = ["chrome", "firefox", "edge"]
        
        # Create a mock browser instance
        mock_browser = MagicMock()
        mock_browser.check_webgpu_support.return_value = True
        mock_browser.check_webnn_support.return_value = True
        mock_browser.check_compute_shader_support.return_value = True
        mock_browser.get_memory_info.return_value = {"limit": 4000}
        self.mock_resource_pool.get_browser_instance.return_value = mock_browser
        
        # Create test model configurations
        self.model_configs = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
            {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1}
        ]
        
        # Create predictor
        self.predictor = MultiModelPredictor(verbose=False)
        
        # Create resource pool adapter
        self.adapter = WebResourcePoolAdapter(
            resource_pool=self.mock_resource_pool,
            max_connections=2,
            enable_tensor_sharing=True,
            enable_strategy_optimization=True,
            browser_capability_detection=True,
            verbose=True
        )
        
        # Create empirical validator
        self.validator = MultiModelEmpiricalValidator(
            validation_history_size=50,
            error_threshold=0.15,
            refinement_interval=5,
            enable_trend_analysis=True,
            verbose=True
        )
        
        # Create integration
        self.integration = MultiModelResourcePoolIntegration(
            predictor=self.predictor,
            resource_pool=self.mock_resource_pool,
            validator=self.validator,
            max_connections=2,
            enable_empirical_validation=True,
            validation_interval=1,  # Use 1 for testing
            prediction_refinement=True,
            enable_adaptive_optimization=True,
            verbose=True
        )
        
        # Initialize components
        self.adapter.initialize()
        self.integration.initialize()
    
    def test_integration_initialization(self):
        """Test that the integration initializes correctly with the adapter components."""
        self.assertTrue(self.integration.initialized)
        self.assertTrue(self.adapter.initialized)
        self.assertIsNotNone(self.integration.predictor)
        self.assertIsNotNone(self.integration.validator)
        self.assertIsNotNone(self.integration.resource_pool)
    
    def test_adapter_browser_capabilities(self):
        """Test the adapter's browser capability detection."""
        capabilities = self.adapter.get_browser_capabilities()
        self.assertIn("chrome", capabilities)
        self.assertTrue(capabilities["chrome"]["webgpu"])
        self.assertTrue(capabilities["chrome"]["webnn"])
    
    def test_adapter_optimal_browser_selection(self):
        """Test the adapter's optimal browser selection logic."""
        # Test for text embedding
        browser = self.adapter.get_optimal_browser("text_embedding")
        self.assertEqual(browser, "edge")
        
        # Test for vision
        browser = self.adapter.get_optimal_browser("vision")
        self.assertEqual(browser, "chrome")
        
        # Test for audio
        browser = self.adapter.get_optimal_browser("audio")
        self.assertEqual(browser, "firefox")
    
    def test_adapter_optimal_strategy_selection(self):
        """Test the adapter's optimal strategy selection logic."""
        # Test with small number of models
        strategy = self.adapter.get_optimal_strategy(self.model_configs, "chrome", "latency")
        self.assertEqual(strategy, "parallel")
        
        # Test with larger number of models
        large_configs = self.model_configs * 6  # 12 models
        strategy = self.adapter.get_optimal_strategy(large_configs, "chrome", "latency")
        self.assertEqual(strategy, "sequential")
        
        # Test with medium number and memory optimization
        medium_configs = self.model_configs * 3  # 6 models
        strategy = self.adapter.get_optimal_strategy(medium_configs, "chrome", "memory")
        self.assertEqual(strategy, "sequential")
    
    @patch('predictive_performance.web_resource_pool_adapter.time')
    def test_adapter_execute_models(self, mock_time):
        """Test the adapter's model execution with different strategies."""
        # Set up mock time
        mock_time.time.side_effect = [1000, 1010, 1020]  # Start, execution start, end
        
        # Test parallel execution
        result = self.adapter.execute_models(
            model_configs=self.model_configs,
            execution_strategy="parallel",
            optimization_goal="latency",
            browser="chrome"
        )
        
        # Verify execution results
        self.assertTrue(result["success"])
        self.assertEqual(result["execution_strategy"], "parallel")
        self.assertEqual(result["browser"], "chrome")
        self.assertIn("throughput", result)
        self.assertIn("latency", result)
        self.assertIn("memory_usage", result)
        
        # Verify resource pool was called
        self.mock_resource_pool.get_model.assert_called()
        self.mock_resource_pool.execute_concurrent.assert_called()
    
    @patch('predictive_performance.multi_model_resource_pool_integration.time')
    def test_integration_execute_with_strategy(self, mock_time):
        """Test the integration's execution with strategy and validation."""
        # Set up mock time
        mock_time.time.side_effect = [1000, 1010, 1020, 1030]
        
        # Execute with strategy
        result = self.integration.execute_with_strategy(
            model_configs=self.model_configs,
            hardware_platform="webgpu",
            execution_strategy="parallel",
            optimization_goal="latency",
            validate_predictions=True
        )
        
        # Verify execution results
        self.assertTrue(result["success"])
        self.assertEqual(result["execution_strategy"], "parallel")
        self.assertIn("predicted_throughput", result)
        self.assertIn("predicted_latency", result)
        self.assertIn("predicted_memory", result)
        self.assertIn("actual_throughput", result)
        self.assertIn("actual_latency", result)
        self.assertIn("actual_memory", result)
        
        # Verify validation was performed (validator is mocked)
        self.assertEqual(self.integration.validation_metrics["validation_count"], 1)
    
    @patch('predictive_performance.multi_model_resource_pool_integration.time')
    def test_integration_compare_strategies(self, mock_time):
        """Test the integration's strategy comparison functionality."""
        # Set up mock time
        time_values = [1000 + 10*i for i in range(20)]
        mock_time.time.side_effect = time_values
        
        # Compare strategies
        comparison = self.integration.compare_strategies(
            model_configs=self.model_configs,
            hardware_platform="webgpu",
            optimization_goal="throughput"
        )
        
        # Verify comparison results
        self.assertIn("best_strategy", comparison)
        self.assertIn("recommended_strategy", comparison)
        self.assertIn("recommendation_accuracy", comparison)
        self.assertIn("strategy_results", comparison)
        self.assertIn("optimization_impact", comparison)
        
        # Verify all strategies were compared
        self.assertIn("parallel", comparison["strategy_results"])
        self.assertIn("sequential", comparison["strategy_results"])
        self.assertIn("batched", comparison["strategy_results"])
    
    def test_integration_get_validation_metrics(self):
        """Test retrieving validation metrics from the integration."""
        # Execute to generate validation metrics
        with patch('predictive_performance.multi_model_resource_pool_integration.time', return_value=1000):
            self.integration.execute_with_strategy(
                model_configs=self.model_configs,
                hardware_platform="webgpu",
                execution_strategy="parallel"
            )
        
        # Get validation metrics
        metrics = self.integration.get_validation_metrics()
        
        # Verify metrics structure
        self.assertIn("validation_count", metrics)
        self.assertIn("execution_count", metrics)
        self.assertIn("error_rates", metrics)
        
        # Verify the validation count
        self.assertEqual(metrics["validation_count"], 1)
    
    def test_full_integration_flow(self):
        """Test the full integration flow with all components."""
        # Set up the predictor with a contention model update method
        self.predictor.update_contention_models = MagicMock()
        
        # Execute with strategy
        with patch('predictive_performance.multi_model_resource_pool_integration.time', return_value=1000):
            result = self.integration.execute_with_strategy(
                model_configs=self.model_configs,
                hardware_platform="webgpu",
                execution_strategy=None,  # Auto-select
                optimization_goal="throughput",
                validate_predictions=True
            )
        
        # Verify execution results
        self.assertTrue(result["success"])
        self.assertIn("execution_strategy", result)
        
        # Compare strategies to see if recommendation is accurate
        with patch('predictive_performance.multi_model_resource_pool_integration.time', return_value=1010):
            comparison = self.integration.compare_strategies(
                model_configs=self.model_configs,
                hardware_platform="webgpu",
                optimization_goal="throughput"
            )
        
        # Verify comparison results
        self.assertIn("best_strategy", comparison)
        self.assertIn("recommended_strategy", comparison)
        
        # Update strategy configuration adaptively
        config = self.integration.update_strategy_configuration("webgpu")
        
        # Verify configuration was updated
        self.assertIn("parallel_threshold", config)
        self.assertIn("sequential_threshold", config)
        self.assertIn("batching_size", config)
        self.assertIn("memory_threshold", config)
    
    def test_adapter_tensor_sharing(self):
        """Test the adapter's tensor sharing functionality."""
        # Set up tensor sharing method in resource pool
        self.mock_resource_pool.setup_tensor_sharing = MagicMock(return_value={"success": True, "memory_saved": 200})
        self.mock_resource_pool.cleanup_tensor_sharing = MagicMock()
        
        # Create models that can share tensors
        text_models = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 1},
            {"model_name": "bert-large-uncased", "model_type": "text_embedding", "batch_size": 1},
            {"model_name": "t5-small", "model_type": "text_embedding", "batch_size": 1}
        ]
        
        # Execute models with tensor sharing
        with patch('predictive_performance.web_resource_pool_adapter.time'):
            result = self.adapter.execute_models(
                model_configs=text_models,
                execution_strategy="parallel",
                browser="chrome"
            )
        
        # Verify tensor sharing was used
        self.mock_resource_pool.setup_tensor_sharing.assert_called()
        self.mock_resource_pool.cleanup_tensor_sharing.assert_called()
        
        # Verify execution succeeded
        self.assertTrue(result["success"])
    
    def test_empirical_validation_workflow(self):
        """Test the empirical validation workflow in the integration."""
        # Create a real validator with mock methods
        validator = MultiModelEmpiricalValidator(
            validation_history_size=10,
            error_threshold=0.15,
            refinement_interval=2,  # Set low for testing
            enable_trend_analysis=True
        )
        
        # Mock the validator methods
        validator.validate_prediction = MagicMock(return_value={
            "validation_count": 1,
            "current_errors": {"throughput": 0.12, "latency": 0.15, "memory": 0.1},
            "average_errors": {"throughput": 0.12, "latency": 0.15, "memory": 0.1},
            "needs_refinement": True
        })
        
        validator.get_refinement_recommendations = MagicMock(return_value={
            "refinement_needed": True,
            "reason": "Error rates exceed threshold",
            "recommended_method": "incremental",
            "error_rates": {"throughput": 0.12, "latency": 0.15, "memory": 0.1}
        })
        
        validator.generate_validation_dataset = MagicMock(return_value={
            "success": True,
            "records": [{"model_count": 2, "hardware_platform": "webgpu"}],
            "record_count": 1
        })
        
        validator.record_model_refinement = MagicMock()
        
        # Replace the integration's validator
        self.integration.validator = validator
        
        # Mock the predictor's update methods
        self.integration.predictor.update_models = MagicMock()
        
        # Execute to trigger validation and refinement
        with patch('predictive_performance.multi_model_resource_pool_integration.time', return_value=1000):
            for _ in range(3):  # Run multiple times to trigger refinement
                result = self.integration.execute_with_strategy(
                    model_configs=self.model_configs,
                    hardware_platform="webgpu",
                    execution_strategy="parallel",
                    validate_predictions=True
                )
        
        # Verify validation and refinement occurred
        validator.validate_prediction.assert_called()
        validator.get_refinement_recommendations.assert_called()
        validator.generate_validation_dataset.assert_called()
        validator.record_model_refinement.assert_called()
        self.integration.predictor.update_models.assert_called()
    
    def test_web_resource_pool_adapter_integration(self):
        """Test the integration between the web resource pool adapter and the resource pool integration."""
        # Create a special integration using the adapter
        adapter_integration = MultiModelResourcePoolIntegration(
            predictor=self.predictor,
            resource_pool=self.adapter,  # Use adapter as resource pool
            validator=self.validator,
            enable_empirical_validation=True,
            validation_interval=1
        )
        
        # Initialize the integration
        adapter_integration.initialize()
        
        # Execute a strategy using the adapter
        with patch('predictive_performance.multi_model_resource_pool_integration.time', return_value=1000):
            result = adapter_integration.execute_with_strategy(
                model_configs=self.model_configs,
                hardware_platform="webgpu",
                execution_strategy="parallel"
            )
        
        # Verify execution was successful through the adapter
        self.assertTrue(result["success"])
        
        # Verify execution used the adapter's execute_models method
        self.mock_resource_pool.execute_concurrent.assert_called()
    
    def tearDown(self):
        """Clean up after each test."""
        # Close the integration
        self.integration.close()
        self.adapter.close()


class TestMultiModelWebIntegrationWithTempDB(unittest.TestCase):
    """Test cases for the Multi-Model Web Integration with a real temporary database."""
    
    def setUp(self):
        """Set up before each test with a temporary database."""
        try:
            import duckdb
            self.has_duckdb = True
        except ImportError:
            self.has_duckdb = False
            self.skipTest("DuckDB not available, skipping DB tests")
            return
        
        # Create temporary file for database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False)
        self.temp_db.close()
        
        # Create mock objects with real database
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
        
        # Create components with real database
        self.predictor = MultiModelPredictor(verbose=False)
        self.validator = MultiModelEmpiricalValidator(db_path=self.temp_db.name, verbose=True)
        
        # Create integration with real database
        self.integration = MultiModelResourcePoolIntegration(
            predictor=self.predictor,
            resource_pool=self.mock_resource_pool,
            validator=self.validator,
            db_path=self.temp_db.name,
            enable_empirical_validation=True,
            validation_interval=1,
            verbose=True
        )
        
        # Initialize
        self.integration.initialize()
    
    def test_database_validation_storage(self):
        """Test that validation metrics are stored in the database."""
        if not self.has_duckdb:
            self.skipTest("DuckDB not available")
            return
        
        # Execute to generate validation metrics
        with patch('predictive_performance.multi_model_resource_pool_integration.time', return_value=1000):
            for _ in range(3):  # Generate multiple validation records
                self.integration.execute_with_strategy(
                    model_configs=self.model_configs,
                    hardware_platform="webgpu",
                    execution_strategy="parallel"
                )
        
        # Connect to database to verify records
        import duckdb
        conn = duckdb.connect(self.temp_db.name)
        
        # Check if validation metrics were stored
        result = conn.execute("SELECT COUNT(*) FROM multi_model_validation_metrics").fetchone()
        self.assertGreater(result[0], 0)
        
        # Close database connection
        conn.close()
    
    def test_database_metrics_retrieval(self):
        """Test retrieving metrics from the database."""
        if not self.has_duckdb:
            self.skipTest("DuckDB not available")
            return
        
        # Execute to generate validation metrics
        with patch('predictive_performance.multi_model_resource_pool_integration.time', return_value=1000):
            for _ in range(3):
                self.integration.execute_with_strategy(
                    model_configs=self.model_configs,
                    hardware_platform="webgpu",
                    execution_strategy="parallel"
                )
        
        # Get validation metrics with database statistics
        metrics = self.integration.get_validation_metrics()
        
        # Verify database statistics were included
        self.assertIn("database", metrics)
        self.assertIn("validation_count", metrics["database"])
        self.assertEqual(metrics["database"]["validation_count"], 3)
    
    def tearDown(self):
        """Clean up after each test."""
        # Close the integration
        self.integration.close()
        
        # Remove temporary database file
        if hasattr(self, 'temp_db'):
            try:
                os.unlink(self.temp_db.name)
            except:
                pass


if __name__ == "__main__":
    unittest.main()