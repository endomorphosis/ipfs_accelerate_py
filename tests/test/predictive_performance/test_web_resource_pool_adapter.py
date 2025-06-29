#!/usr/bin/env python3
"""
Test script for the WebNN/WebGPU Resource Pool Adapter.

This script tests the core functionality of the web resource pool adapter,
including browser capability detection, strategy optimization, tensor sharing,
and execution with different strategies.
"""

import os
import sys
import unittest
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from predictive_performance.web_resource_pool_adapter import (
    WebResourcePoolAdapter,
    BROWSER_CAPABILITIES,
    MODEL_BROWSER_PREFERENCES,
    BROWSER_STRATEGY_PREFERENCES
)


class TestWebResourcePoolAdapter(unittest.TestCase):
    """Test cases for the WebNN/WebGPU Resource Pool Adapter."""
    
    def setUp(self):
        """Set up before each test."""
        # Create a mock resource pool
        self.mock_resource_pool = MagicMock()
        self.mock_resource_pool.initialize.return_value = True
        self.mock_resource_pool.get_available_browsers.return_value = ["chrome", "firefox", "edge"]
        
        # Mock browser instance
        self.mock_browser_instance = MagicMock()
        self.mock_browser_instance.check_webgpu_support.return_value = True
        self.mock_browser_instance.check_webnn_support.return_value = True
        self.mock_browser_instance.check_compute_shader_support.return_value = True
        self.mock_browser_instance.get_memory_info.return_value = {"limit": 4000}
        
        # Configure resource pool to return mock browser instance
        self.mock_resource_pool.get_browser_instance.return_value = self.mock_browser_instance
        
        # Configure resource pool for model execution
        self.mock_model = MagicMock()
        self.mock_model.return_value = {"success": True, "result": [0.1, 0.2, 0.3]}
        self.mock_resource_pool.get_model.return_value = self.mock_model
        self.mock_resource_pool.execute_concurrent.return_value = [
            {"success": True, "result": [0.1, 0.2, 0.3]},
            {"success": True, "result": [0.4, 0.5, 0.6]}
        ]
        self.mock_resource_pool.get_metrics.return_value = {
            "base_metrics": {
                "peak_memory_usage": 1500,
                "execution_time": 0.5
            }
        }
        
        # Create adapter with mock resource pool
        self.adapter = WebResourcePoolAdapter(
            resource_pool=self.mock_resource_pool,
            max_connections=2,
            enable_tensor_sharing=True,
            enable_strategy_optimization=True,
            browser_capability_detection=True,
            verbose=True
        )
        
        # Define test model configurations
        self.model_configs = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
            {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1}
        ]
    
    def test_initialization(self):
        """Test that the adapter initializes correctly."""
        self.assertIsNotNone(self.adapter)
        self.assertEqual(self.adapter.resource_pool, self.mock_resource_pool)
        self.assertEqual(self.adapter.max_connections, 2)
        self.assertTrue(self.adapter.enable_tensor_sharing)
        self.assertTrue(self.adapter.enable_strategy_optimization)
        self.assertTrue(self.adapter.browser_capability_detection)
        self.assertFalse(self.adapter.initialized)
        
        # Initialize the adapter
        success = self.adapter.initialize()
        self.assertTrue(success)
        self.assertTrue(self.adapter.initialized)
    
    def test_browser_detection(self):
        """Test browser capability detection."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Check that browser capabilities were detected
        self.assertEqual(len(self.adapter.browser_capabilities), 3)  # chrome, firefox, edge
        self.assertIn("chrome", self.adapter.browser_capabilities)
        self.assertIn("firefox", self.adapter.browser_capabilities)
        self.assertIn("edge", self.adapter.browser_capabilities)
        
        # Check specific capabilities
        chrome_caps = self.adapter.browser_capabilities["chrome"]
        self.assertTrue(chrome_caps["webgpu"])
        self.assertTrue(chrome_caps["webnn"])
        self.assertTrue(chrome_caps["compute_shader"])
    
    def test_get_optimal_browser(self):
        """Test browser selection based on model type."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Test for text embedding (should prefer Edge with WebNN)
        browser = self.adapter.get_optimal_browser("text_embedding")
        self.assertEqual(browser, "edge")
        
        # Test for audio (should prefer Firefox with compute shaders)
        browser = self.adapter.get_optimal_browser("audio")
        self.assertEqual(browser, "firefox")
        
        # Test for vision (should prefer Chrome)
        browser = self.adapter.get_optimal_browser("vision")
        self.assertEqual(browser, "chrome")
        
        # Test with modified capabilities
        self.adapter.browser_capabilities["edge"]["webnn"] = False
        browser = self.adapter.get_optimal_browser("text_embedding")
        self.assertNotEqual(browser, "edge")  # Should not be edge if WebNN is not available
    
    def test_get_optimal_strategy(self):
        """Test strategy selection based on model count and browser."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Small number of models - should be parallel
        strategy = self.adapter.get_optimal_strategy(
            model_configs=self.model_configs,  # 2 models
            browser="chrome",
            optimization_goal="latency"
        )
        self.assertEqual(strategy, "parallel")
        
        # Large number of models - should be sequential
        large_configs = self.model_configs * 10  # 20 models
        strategy = self.adapter.get_optimal_strategy(
            model_configs=large_configs,
            browser="chrome",
            optimization_goal="latency"
        )
        self.assertEqual(strategy, "sequential")
        
        # Medium number of models with throughput goal - should be batched
        medium_configs = self.model_configs * 3  # 6 models
        strategy = self.adapter.get_optimal_strategy(
            model_configs=medium_configs,
            browser="chrome",
            optimization_goal="throughput"
        )
        self.assertEqual(strategy, "batched")
        
        # Test with high memory usage - should be batched or sequential
        with patch.object(self.adapter, '_estimate_total_memory', return_value=5000):
            strategy = self.adapter.get_optimal_strategy(
                model_configs=self.model_configs,
                browser="chrome",
                optimization_goal="latency"
            )
            self.assertEqual(strategy, "batched")
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Test with basic models
        memory = self.adapter._estimate_total_memory(self.model_configs)
        self.assertGreater(memory, 0)
        
        # Test with tensor sharing
        original_memory = memory
        # Add a duplicate text model to enable sharing
        configs_with_sharing = self.model_configs + [{"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4}]
        memory_with_sharing = self.adapter._estimate_total_memory(configs_with_sharing)
        
        # Memory with sharing should be less than just adding another model's memory
        expected_no_sharing = original_memory + self.adapter._estimate_total_memory([self.model_configs[0]])
        self.assertLess(memory_with_sharing, expected_no_sharing)
        
        # Test with tensor sharing disabled
        self.adapter.enable_tensor_sharing = False
        memory_no_sharing = self.adapter._estimate_total_memory(configs_with_sharing)
        self.assertGreater(memory_no_sharing, memory_with_sharing)
    
    def test_execute_models_parallel(self):
        """Test model execution with parallel strategy."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Execute models with parallel strategy
        result = self.adapter.execute_models(
            model_configs=self.model_configs,
            execution_strategy="parallel",
            optimization_goal="latency",
            browser="chrome"
        )
        
        # Check result structure
        self.assertTrue(result["success"])
        self.assertEqual(result["execution_strategy"], "parallel")
        self.assertEqual(result["browser"], "chrome")
        self.assertEqual(result["model_count"], 2)
        self.assertIn("throughput", result)
        self.assertIn("latency", result)
        self.assertIn("memory_usage", result)
        self.assertIn("model_results", result)
        
        # Check that execute_concurrent was called
        self.mock_resource_pool.execute_concurrent.assert_called_once()
    
    def test_execute_models_sequential(self):
        """Test model execution with sequential strategy."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Execute models with sequential strategy
        result = self.adapter.execute_models(
            model_configs=self.model_configs,
            execution_strategy="sequential",
            optimization_goal="latency",
            browser="chrome"
        )
        
        # Check result structure
        self.assertTrue(result["success"])
        self.assertEqual(result["execution_strategy"], "sequential")
        self.assertEqual(result["browser"], "chrome")
        self.assertEqual(result["model_count"], 2)
        self.assertIn("throughput", result)
        self.assertIn("latency", result)
        self.assertIn("memory_usage", result)
        self.assertIn("model_results", result)
        
        # Check that each model was called individually
        self.assertEqual(self.mock_model.call_count, 2)
    
    def test_execute_models_batched(self):
        """Test model execution with batched strategy."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Execute models with batched strategy
        result = self.adapter.execute_models(
            model_configs=self.model_configs,
            execution_strategy="batched",
            optimization_goal="latency",
            browser="chrome"
        )
        
        # Check result structure
        self.assertTrue(result["success"])
        self.assertEqual(result["execution_strategy"], "batched")
        self.assertEqual(result["browser"], "chrome")
        self.assertEqual(result["model_count"], 2)
        self.assertIn("throughput", result)
        self.assertIn("latency", result)
        self.assertIn("memory_usage", result)
        self.assertIn("model_results", result)
        
        # Check that execute_concurrent was called once (all models in one batch)
        self.mock_resource_pool.execute_concurrent.assert_called_once()
    
    def test_execute_models_auto_strategy(self):
        """Test model execution with automatic strategy selection."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Mock get_optimal_strategy to return a specific strategy
        with patch.object(self.adapter, 'get_optimal_strategy', return_value="parallel"):
            # Execute models with auto strategy
            result = self.adapter.execute_models(
                model_configs=self.model_configs,
                execution_strategy="auto",
                optimization_goal="latency",
                browser="chrome"
            )
            
            # Check that the selected strategy was used
            self.assertEqual(result["execution_strategy"], "parallel")
    
    def test_tensor_sharing(self):
        """Test tensor sharing setup and cleanup."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Add setup_tensor_sharing method to resource pool
        self.mock_resource_pool.setup_tensor_sharing = MagicMock(
            return_value={"success": True, "memory_saved": 200}
        )
        self.mock_resource_pool.cleanup_tensor_sharing = MagicMock()
        
        # Create models list
        models = [MagicMock(), MagicMock()]
        
        # Test setup_tensor_sharing
        self.adapter._setup_tensor_sharing(
            model_configs=[
                {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
                {"model_name": "bert-large-uncased", "model_type": "text_embedding", "batch_size": 2}
            ],
            models=models
        )
        
        # Check that setup_tensor_sharing was called
        self.mock_resource_pool.setup_tensor_sharing.assert_called_once()
        
        # Check that stats were updated
        stats = self.adapter.execution_stats["tensor_sharing_stats"]
        self.assertEqual(stats["models_sharing_tensors"], 2)
        self.assertEqual(stats["memory_saved_mb"], 200)
        
        # Test cleanup_tensor_sharing
        self.adapter._cleanup_tensor_sharing(models)
        
        # Check that cleanup_tensor_sharing was called
        self.mock_resource_pool.cleanup_tensor_sharing.assert_called_once_with(models)
    
    def test_compare_strategies(self):
        """Test comparison of different execution strategies."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Mock execute_models to return different metrics for different strategies
        def mock_execute_models(model_configs, execution_strategy, optimization_goal, browser, return_metrics):
            if execution_strategy == "parallel":
                return {
                    "success": True,
                    "execution_strategy": "parallel",
                    "throughput": 100,
                    "latency": 50,
                    "memory_usage": 1000
                }
            elif execution_strategy == "sequential":
                return {
                    "success": True,
                    "execution_strategy": "sequential",
                    "throughput": 50,
                    "latency": 100,
                    "memory_usage": 800
                }
            else:  # batched
                return {
                    "success": True,
                    "execution_strategy": "batched",
                    "throughput": 80,
                    "latency": 60,
                    "memory_usage": 900
                }
        
        # Apply mock
        self.adapter.execute_models = MagicMock(side_effect=mock_execute_models)
        
        # Compare strategies with throughput optimization goal
        comparison = self.adapter.compare_strategies(
            model_configs=self.model_configs,
            browser="chrome",
            optimization_goal="throughput"
        )
        
        # Check comparison result
        self.assertTrue(comparison["success"])
        self.assertEqual(comparison["best_strategy"], "parallel")  # Highest throughput
        self.assertIn("throughput_improvement_percent", comparison)
        
        # Compare strategies with latency optimization goal
        comparison = self.adapter.compare_strategies(
            model_configs=self.model_configs,
            browser="chrome",
            optimization_goal="latency"
        )
        
        # Check comparison result
        self.assertTrue(comparison["success"])
        self.assertEqual(comparison["best_strategy"], "parallel")  # Lowest latency
        self.assertIn("latency_improvement_percent", comparison)
        
        # Compare strategies with memory optimization goal
        comparison = self.adapter.compare_strategies(
            model_configs=self.model_configs,
            browser="chrome",
            optimization_goal="memory"
        )
        
        # Check comparison result
        self.assertTrue(comparison["success"])
        self.assertEqual(comparison["best_strategy"], "sequential")  # Lowest memory
        self.assertIn("memory_improvement_percent", comparison)
    
    def test_browser_auto_selection(self):
        """Test automatic browser selection."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Mock get_optimal_browser to track calls
        with patch.object(self.adapter, 'get_optimal_browser', return_value="edge") as mock_get_browser:
            # Execute models without specifying browser
            result = self.adapter.execute_models(
                model_configs=self.model_configs,
                execution_strategy="parallel"
            )
            
            # Check that get_optimal_browser was called
            mock_get_browser.assert_called_once()
            
            # Check that the selected browser was used
            self.assertEqual(result["browser"], "edge")
    
    def test_execution_statistics(self):
        """Test execution statistics tracking."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Execute models multiple times with different strategies
        self.adapter.execute_models(
            model_configs=self.model_configs,
            execution_strategy="parallel",
            browser="chrome"
        )
        
        self.adapter.execute_models(
            model_configs=self.model_configs,
            execution_strategy="sequential",
            browser="firefox"
        )
        
        self.adapter.execute_models(
            model_configs=self.model_configs,
            execution_strategy="parallel",
            browser="chrome"
        )
        
        # Get statistics
        stats = self.adapter.get_execution_statistics()
        
        # Check total executions
        self.assertEqual(stats["total_executions"], 3)
        
        # Check browser executions
        self.assertEqual(stats["browser_executions"]["chrome"], 2)
        self.assertEqual(stats["browser_executions"]["firefox"], 1)
        
        # Check strategy executions
        self.assertEqual(stats["strategy_executions"]["parallel"], 2)
        self.assertEqual(stats["strategy_executions"]["sequential"], 1)
    
    def test_close(self):
        """Test closing the adapter."""
        # Initialize the adapter
        self.adapter.initialize()
        
        # Close the adapter
        success = self.adapter.close()
        
        # Check that the resource pool was closed
        self.mock_resource_pool.close.assert_called_once()
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()