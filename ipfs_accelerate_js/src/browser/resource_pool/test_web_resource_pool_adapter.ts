/**
 * Converted from Python: test_web_resource_pool_adapter.py
 * Conversion date: 2025-03-11 04:08:53
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Test script for the WebNN/WebGPU Resource Pool Adapter.

This script tests the core functionality of the web resource pool adapter,
including browser capability detection, strategy optimization, tensor sharing,
and execution with different strategies.
"""

import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"
from unittest.mock import * as $1, patch
import ${$1} from "$1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.$1.push($2).parent.parent))

# Import the module to test
from predictive_performance.web_resource_pool_adapter import (
  WebResourcePoolAdapter,
  BROWSER_CAPABILITIES,
  MODEL_BROWSER_PREFERENCES,
  BROWSER_STRATEGY_PREFERENCES
)


class TestWebResourcePoolAdapter(unittest.TestCase):
  """Test cases for the WebNN/WebGPU Resource Pool Adapter."""
  
  $1($2) {
    """Set up before each test."""
    # Create a mock resource pool
    this.mock_resource_pool = MagicMock()
    this.mock_resource_pool.initialize.return_value = true
    this.mock_resource_pool.get_available_browsers.return_value = ["chrome", "firefox", "edge"]
    
  }
    # Mock browser instance
    this.mock_browser_instance = MagicMock()
    this.mock_browser_instance.check_webgpu_support.return_value = true
    this.mock_browser_instance.check_webnn_support.return_value = true
    this.mock_browser_instance.check_compute_shader_support.return_value = true
    this.mock_browser_instance.get_memory_info.return_value = ${$1}
    
    # Configure resource pool to return mock browser instance
    this.mock_resource_pool.get_browser_instance.return_value = this.mock_browser_instance
    
    # Configure resource pool for model execution
    this.mock_model = MagicMock()
    this.mock_model.return_value = ${$1}
    this.mock_resource_pool.get_model.return_value = this.mock_model
    this.mock_resource_pool.execute_concurrent.return_value = [
      ${$1},
      ${$1}
    ]
    this.mock_resource_pool.get_metrics.return_value = {
      "base_metrics": ${$1}
    }
    }
    
    # Create adapter with mock resource pool
    this.adapter = WebResourcePoolAdapter(
      resource_pool=this.mock_resource_pool,
      max_connections=2,
      enable_tensor_sharing=true,
      enable_strategy_optimization=true,
      browser_capability_detection=true,
      verbose=true
    )
    
    # Define test model configurations
    this.model_configs = [
      ${$1},
      ${$1}
    ]
  
  $1($2) {
    """Test that the adapter initializes correctly."""
    this.assertIsNotnull(this.adapter)
    this.assertEqual(this.adapter.resource_pool, this.mock_resource_pool)
    this.assertEqual(this.adapter.max_connections, 2)
    this.asserttrue(this.adapter.enable_tensor_sharing)
    this.asserttrue(this.adapter.enable_strategy_optimization)
    this.asserttrue(this.adapter.browser_capability_detection)
    this.assertfalse(this.adapter.initialized)
    
  }
    # Initialize the adapter
    success = this.adapter.initialize()
    this.asserttrue(success)
    this.asserttrue(this.adapter.initialized)
  
  $1($2) {
    """Test browser capability detection."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Check that browser capabilities were detected
    this.assertEqual(len(this.adapter.browser_capabilities), 3)  # chrome, firefox, edge
    this.assertIn("chrome", this.adapter.browser_capabilities)
    this.assertIn("firefox", this.adapter.browser_capabilities)
    this.assertIn("edge", this.adapter.browser_capabilities)
    
    # Check specific capabilities
    chrome_caps = this.adapter.browser_capabilities["chrome"]
    this.asserttrue(chrome_caps["webgpu"])
    this.asserttrue(chrome_caps["webnn"])
    this.asserttrue(chrome_caps["compute_shader"])
  
  $1($2) {
    """Test browser selection based on model type."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Test for text embedding (should prefer Edge with WebNN)
    browser = this.adapter.get_optimal_browser("text_embedding")
    this.assertEqual(browser, "edge")
    
    # Test for audio (should prefer Firefox with compute shaders)
    browser = this.adapter.get_optimal_browser("audio")
    this.assertEqual(browser, "firefox")
    
    # Test for vision (should prefer Chrome)
    browser = this.adapter.get_optimal_browser("vision")
    this.assertEqual(browser, "chrome")
    
    # Test with modified capabilities
    this.adapter.browser_capabilities["edge"]["webnn"] = false
    browser = this.adapter.get_optimal_browser("text_embedding")
    this.assertNotEqual(browser, "edge")  # Should !be edge if WebNN is !available
  
  $1($2) {
    """Test strategy selection based on model count && browser."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Small number of models - should be parallel
    strategy = this.adapter.get_optimal_strategy(
      model_configs=this.model_configs,  # 2 models
      browser="chrome",
      optimization_goal="latency"
    )
    this.assertEqual(strategy, "parallel")
    
    # Large number of models - should be sequential
    large_configs = this.model_configs * 10  # 20 models
    strategy = this.adapter.get_optimal_strategy(
      model_configs=large_configs,
      browser="chrome",
      optimization_goal="latency"
    )
    this.assertEqual(strategy, "sequential")
    
    # Medium number of models with throughput goal - should be batched
    medium_configs = this.model_configs * 3  # 6 models
    strategy = this.adapter.get_optimal_strategy(
      model_configs=medium_configs,
      browser="chrome",
      optimization_goal="throughput"
    )
    this.assertEqual(strategy, "batched")
    
    # Test with high memory usage - should be batched || sequential
    with patch.object(this.adapter, '_estimate_total_memory', return_value=5000):
      strategy = this.adapter.get_optimal_strategy(
        model_configs=this.model_configs,
        browser="chrome",
        optimization_goal="latency"
      )
      this.assertEqual(strategy, "batched")
  
  $1($2) {
    """Test memory usage estimation."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Test with basic models
    memory = this.adapter._estimate_total_memory(this.model_configs)
    this.assertGreater(memory, 0)
    
    # Test with tensor sharing
    original_memory = memory
    # Add a duplicate text model to enable sharing
    configs_with_sharing = this.model_configs + [${$1}]
    memory_with_sharing = this.adapter._estimate_total_memory(configs_with_sharing)
    
    # Memory with sharing should be less than just adding another model's memory
    expected_no_sharing = original_memory + this.adapter._estimate_total_memory([this.model_configs[0]])
    this.assertLess(memory_with_sharing, expected_no_sharing)
    
    # Test with tensor sharing disabled
    this.adapter.enable_tensor_sharing = false
    memory_no_sharing = this.adapter._estimate_total_memory(configs_with_sharing)
    this.assertGreater(memory_no_sharing, memory_with_sharing)
  
  $1($2) {
    """Test model execution with parallel strategy."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Execute models with parallel strategy
    result = this.adapter.execute_models(
      model_configs=this.model_configs,
      execution_strategy="parallel",
      optimization_goal="latency",
      browser="chrome"
    )
    
    # Check result structure
    this.asserttrue(result["success"])
    this.assertEqual(result["execution_strategy"], "parallel")
    this.assertEqual(result["browser"], "chrome")
    this.assertEqual(result["model_count"], 2)
    this.assertIn("throughput", result)
    this.assertIn("latency", result)
    this.assertIn("memory_usage", result)
    this.assertIn("model_results", result)
    
    # Check that execute_concurrent was called
    this.mock_resource_pool.execute_concurrent.assert_called_once()
  
  $1($2) {
    """Test model execution with sequential strategy."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Execute models with sequential strategy
    result = this.adapter.execute_models(
      model_configs=this.model_configs,
      execution_strategy="sequential",
      optimization_goal="latency",
      browser="chrome"
    )
    
    # Check result structure
    this.asserttrue(result["success"])
    this.assertEqual(result["execution_strategy"], "sequential")
    this.assertEqual(result["browser"], "chrome")
    this.assertEqual(result["model_count"], 2)
    this.assertIn("throughput", result)
    this.assertIn("latency", result)
    this.assertIn("memory_usage", result)
    this.assertIn("model_results", result)
    
    # Check that each model was called individually
    this.assertEqual(this.mock_model.call_count, 2)
  
  $1($2) {
    """Test model execution with batched strategy."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Execute models with batched strategy
    result = this.adapter.execute_models(
      model_configs=this.model_configs,
      execution_strategy="batched",
      optimization_goal="latency",
      browser="chrome"
    )
    
    # Check result structure
    this.asserttrue(result["success"])
    this.assertEqual(result["execution_strategy"], "batched")
    this.assertEqual(result["browser"], "chrome")
    this.assertEqual(result["model_count"], 2)
    this.assertIn("throughput", result)
    this.assertIn("latency", result)
    this.assertIn("memory_usage", result)
    this.assertIn("model_results", result)
    
    # Check that execute_concurrent was called once (all models in one batch)
    this.mock_resource_pool.execute_concurrent.assert_called_once()
  
  $1($2) {
    """Test model execution with automatic strategy selection."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Mock get_optimal_strategy to return a specific strategy
    with patch.object(this.adapter, 'get_optimal_strategy', return_value="parallel"):
      # Execute models with auto strategy
      result = this.adapter.execute_models(
        model_configs=this.model_configs,
        execution_strategy="auto",
        optimization_goal="latency",
        browser="chrome"
      )
      
      # Check that the selected strategy was used
      this.assertEqual(result["execution_strategy"], "parallel")
  
  $1($2) {
    """Test tensor sharing setup && cleanup."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Add setup_tensor_sharing method to resource pool
    this.mock_resource_pool.setup_tensor_sharing = MagicMock(
      return_value=${$1}
    )
    this.mock_resource_pool.cleanup_tensor_sharing = MagicMock()
    
    # Create models list
    models = [MagicMock(), MagicMock()]
    
    # Test setup_tensor_sharing
    this.adapter._setup_tensor_sharing(
      model_configs=[
        ${$1},
        ${$1}
      ],
      models=models
    )
    
    # Check that setup_tensor_sharing was called
    this.mock_resource_pool.setup_tensor_sharing.assert_called_once()
    
    # Check that stats were updated
    stats = this.adapter.execution_stats["tensor_sharing_stats"]
    this.assertEqual(stats["models_sharing_tensors"], 2)
    this.assertEqual(stats["memory_saved_mb"], 200)
    
    # Test cleanup_tensor_sharing
    this.adapter._cleanup_tensor_sharing(models)
    
    # Check that cleanup_tensor_sharing was called
    this.mock_resource_pool.cleanup_tensor_sharing.assert_called_once_with(models)
  
  $1($2) {
    """Test comparison of different execution strategies."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Mock execute_models to return different metrics for different strategies
    $1($2) {
      if ($1) {
        return ${$1}
      elif ($1) {
        return ${$1}
      } else {  # batched
      }
        return ${$1}
    
      }
    # Apply mock
    }
    this.adapter.execute_models = MagicMock(side_effect=mock_execute_models)
    
    # Compare strategies with throughput optimization goal
    comparison = this.adapter.compare_strategies(
      model_configs=this.model_configs,
      browser="chrome",
      optimization_goal="throughput"
    )
    
    # Check comparison result
    this.asserttrue(comparison["success"])
    this.assertEqual(comparison["best_strategy"], "parallel")  # Highest throughput
    this.assertIn("throughput_improvement_percent", comparison)
    
    # Compare strategies with latency optimization goal
    comparison = this.adapter.compare_strategies(
      model_configs=this.model_configs,
      browser="chrome",
      optimization_goal="latency"
    )
    
    # Check comparison result
    this.asserttrue(comparison["success"])
    this.assertEqual(comparison["best_strategy"], "parallel")  # Lowest latency
    this.assertIn("latency_improvement_percent", comparison)
    
    # Compare strategies with memory optimization goal
    comparison = this.adapter.compare_strategies(
      model_configs=this.model_configs,
      browser="chrome",
      optimization_goal="memory"
    )
    
    # Check comparison result
    this.asserttrue(comparison["success"])
    this.assertEqual(comparison["best_strategy"], "sequential")  # Lowest memory
    this.assertIn("memory_improvement_percent", comparison)
  
  $1($2) {
    """Test automatic browser selection."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Mock get_optimal_browser to track calls
    with patch.object(this.adapter, 'get_optimal_browser', return_value="edge") as mock_get_browser:
      # Execute models without specifying browser
      result = this.adapter.execute_models(
        model_configs=this.model_configs,
        execution_strategy="parallel"
      )
      
      # Check that get_optimal_browser was called
      mock_get_browser.assert_called_once()
      
      # Check that the selected browser was used
      this.assertEqual(result["browser"], "edge")
  
  $1($2) {
    """Test execution statistics tracking."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Execute models multiple times with different strategies
    this.adapter.execute_models(
      model_configs=this.model_configs,
      execution_strategy="parallel",
      browser="chrome"
    )
    
    this.adapter.execute_models(
      model_configs=this.model_configs,
      execution_strategy="sequential",
      browser="firefox"
    )
    
    this.adapter.execute_models(
      model_configs=this.model_configs,
      execution_strategy="parallel",
      browser="chrome"
    )
    
    # Get statistics
    stats = this.adapter.get_execution_statistics()
    
    # Check total executions
    this.assertEqual(stats["total_executions"], 3)
    
    # Check browser executions
    this.assertEqual(stats["browser_executions"]["chrome"], 2)
    this.assertEqual(stats["browser_executions"]["firefox"], 1)
    
    # Check strategy executions
    this.assertEqual(stats["strategy_executions"]["parallel"], 2)
    this.assertEqual(stats["strategy_executions"]["sequential"], 1)
  
  $1($2) {
    """Test closing the adapter."""
    # Initialize the adapter
    this.adapter.initialize()
    
  }
    # Close the adapter
    success = this.adapter.close()
    
    # Check that the resource pool was closed
    this.mock_resource_pool.close.assert_called_once()
    this.asserttrue(success)


if ($1) {
  unittest.main()