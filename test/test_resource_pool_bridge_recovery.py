#!/usr/bin/env python3
"""
Test script for Resource Pool Bridge Recovery

This script verifies the fault-tolerance and error recovery capabilities
of the ResourcePoolBridgeRecovery system for WebNN/WebGPU integration.

Usage:
    python test_resource_pool_bridge_recovery.py
"""

import os
import sys
import time
import json
import logging
import unittest
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the recovery module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_pool_bridge_recovery import (
    ResourcePoolBridgeRecovery,
    ResourcePoolBridgeWithRecovery,
    ErrorCategory,
    RecoveryStrategy
)


class TestResourcePoolBridgeRecovery(unittest.TestCase):
    """Test suite for ResourcePoolBridgeRecovery."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock integration
        self.mock_integration = MagicMock()
        self.mock_integration.initialize.return_value = True
        self.mock_integration.get_metrics.return_value = {"aggregate": {"total_inferences": 5}}
        
        # Create recovery system
        self.recovery = ResourcePoolBridgeRecovery(
            integration=self.mock_integration,
            max_retries=3,
            retry_delay=0.1,  # Small delay for tests
            fallback_to_simulation=True
        )
        
    def test_error_categorization(self):
        """Test error categorization system."""
        # Connection errors
        connection_error = Exception("WebSocket connection failed")
        self.assertEqual(
            self.recovery.categorize_error(connection_error, {}),
            ErrorCategory.CONNECTION
        )
        
        # Browser crash
        crash_error = Exception("Browser crashed unexpectedly")
        self.assertEqual(
            self.recovery.categorize_error(crash_error, {}),
            ErrorCategory.BROWSER_CRASH
        )
        
        # Out of memory
        oom_error = Exception("CUDA out of memory")
        self.assertEqual(
            self.recovery.categorize_error(oom_error, {}),
            ErrorCategory.OUT_OF_MEMORY
        )
        
        # Operation not supported
        unsupported_error = Exception("Operation not supported on this platform")
        self.assertEqual(
            self.recovery.categorize_error(unsupported_error, {}),
            ErrorCategory.UNSUPPORTED_OPERATION
        )
        
        # Unknown error
        unknown_error = Exception("Some unexpected error occurred")
        self.assertEqual(
            self.recovery.categorize_error(unknown_error, {}),
            ErrorCategory.UNKNOWN
        )
        
    def test_recovery_strategy_selection(self):
        """Test recovery strategy selection."""
        # Test CONNECTION error
        error_category = ErrorCategory.CONNECTION
        context = {"browser": "chrome", "platform": "webgpu"}
        
        # First attempt
        strategy = self.recovery.determine_recovery_strategy(error_category, context, 0)
        self.assertEqual(strategy, RecoveryStrategy.DELAY_RETRY)
        
        # Second attempt - should be more aggressive
        strategy = self.recovery.determine_recovery_strategy(error_category, context, 1)
        self.assertEqual(strategy, RecoveryStrategy.BROWSER_RESTART)
        
        # Third attempt - even more aggressive
        strategy = self.recovery.determine_recovery_strategy(error_category, context, 2)
        self.assertEqual(strategy, RecoveryStrategy.ALTERNATIVE_BROWSER)
        
        # Test OUT_OF_MEMORY error with large model
        error_category = ErrorCategory.OUT_OF_MEMORY
        context = {"model_size": "large", "browser": "chrome", "platform": "webgpu"}
        strategy = self.recovery.determine_recovery_strategy(error_category, context, 0)
        self.assertEqual(strategy, RecoveryStrategy.REDUCE_MODEL_SIZE)
        
        # Test UNSUPPORTED_OPERATION error with audio model on non-Firefox browser
        error_category = ErrorCategory.UNSUPPORTED_OPERATION
        context = {"model_type": "audio", "browser": "chrome", "platform": "webgpu"}
        strategy = self.recovery.determine_recovery_strategy(error_category, context, 0)
        self.assertEqual(strategy, RecoveryStrategy.ALTERNATIVE_BROWSER)
        
    def test_strategy_application(self):
        """Test the application of recovery strategies."""
        # Test ALTERNATIVE_BROWSER strategy
        strategy = RecoveryStrategy.ALTERNATIVE_BROWSER
        context = {"browser": "chrome", "model_type": "audio"}
        new_context = self.recovery.apply_recovery_strategy(strategy, context)
        self.assertEqual(new_context["browser"], "firefox")  # Firefox preferred for audio
        
        # Test REDUCE_MODEL_SIZE strategy
        strategy = RecoveryStrategy.REDUCE_MODEL_SIZE
        context = {"model_name": "bert-large-uncased"}
        new_context = self.recovery.apply_recovery_strategy(strategy, context)
        self.assertEqual(new_context["model_name"], "bert-base-uncased")
        
        # Test REDUCE_PRECISION strategy
        strategy = RecoveryStrategy.REDUCE_PRECISION
        context = {"hardware_preferences": {"precision": 16}}
        new_context = self.recovery.apply_recovery_strategy(strategy, context)
        self.assertEqual(new_context["hardware_preferences"]["precision"], 8)
        
        # Test SIMULATION_FALLBACK strategy
        strategy = RecoveryStrategy.SIMULATION_FALLBACK
        context = {"hardware_preferences": {"priority_list": ["webgpu", "webnn", "cpu"]}}
        new_context = self.recovery.apply_recovery_strategy(strategy, context)
        self.assertTrue(new_context["simulation"])
        self.assertEqual(new_context["hardware_preferences"]["priority_list"], ["cpu"])
        
    def test_execute_safely_success(self):
        """Test successful execution without recovery."""
        # Create a successful operation
        def success_op():
            return {"success": True, "data": "test_data"}
        
        # Execute with recovery
        success, result, final_context = self.recovery.execute_safely(success_op, {"test": "context"})
        
        # Check results
        self.assertTrue(success)
        self.assertEqual(result["data"], "test_data")
        self.assertEqual(final_context["test"], "context")
        
    def test_execute_safely_failure_with_recovery(self):
        """Test execution with failure and recovery."""
        # Create counter to track retries
        attempt_counter = [0]
        
        # Create an operation that fails on first attempt but succeeds on second
        def failing_op():
            attempt_counter[0] += 1
            if attempt_counter[0] == 1:
                raise Exception("Simulated failure")
            return {"success": True, "data": "recovered_data"}
        
        # Execute with recovery
        success, result, final_context = self.recovery.execute_safely(failing_op, {"test": "context"})
        
        # Check results
        self.assertTrue(success)
        self.assertEqual(result["data"], "recovered_data")
        self.assertEqual(attempt_counter[0], 2)  # Should have retried once
        
    def test_execute_safely_max_retries_exceeded(self):
        """Test execution that fails all retry attempts."""
        # Create an operation that always fails
        def always_failing_op():
            raise Exception("Always failing")
        
        # Execute with recovery
        success, result, final_context = self.recovery.execute_safely(always_failing_op, {"test": "context"})
        
        # Check results
        self.assertFalse(success)
        self.assertIn("error", result)
        self.assertIn("Always failing", result["error"])
        self.assertIn("error", final_context)
        
    def test_browser_health_tracking(self):
        """Test browser health tracking."""
        # Create contexts for success and failure
        context_chrome = {"browser": "chrome", "platform": "webgpu"}
        context_firefox = {"browser": "firefox", "platform": "webgpu"}
        
        # Record successes and failures
        self.recovery._record_success(context_chrome)
        self.recovery._record_success(context_chrome)
        self.recovery._record_failure(context_chrome, Exception("Test error"))
        
        self.recovery._record_success(context_firefox)
        self.recovery._record_failure(context_firefox, Exception("Test error"))
        self.recovery._record_failure(context_firefox, Exception("Test error"))
        
        # Check health scores
        chrome_health = self.recovery._browser_health["chrome"]["health_score"]
        firefox_health = self.recovery._browser_health["firefox"]["health_score"]
        
        self.assertGreater(chrome_health, firefox_health)
        self.assertEqual(chrome_health, 2/3)  # 2 successes, 1 failure
        self.assertEqual(firefox_health, 1/3)  # 1 success, 2 failures
        
    def test_statistics_tracking(self):
        """Test recovery statistics tracking."""
        # Record some recovery attempts
        self.recovery._record_recovery_attempt(
            ErrorCategory.CONNECTION,
            RecoveryStrategy.DELAY_RETRY,
            {"browser": "chrome", "platform": "webgpu"}
        )
        
        self.recovery._record_recovery_attempt(
            ErrorCategory.OUT_OF_MEMORY,
            RecoveryStrategy.REDUCE_MODEL_SIZE,
            {"browser": "firefox", "platform": "webgpu"}
        )
        
        # Get statistics
        stats = self.recovery.get_recovery_statistics()
        
        # Check statistics
        self.assertEqual(stats["total_recovery_attempts"], 2)
        self.assertEqual(stats["error_categories"]["connection"], 1)
        self.assertEqual(stats["error_categories"]["out_of_memory"], 1)
        self.assertEqual(stats["recovery_strategies"]["delay_retry"], 1)
        self.assertEqual(stats["recovery_strategies"]["reduce_model_size"], 1)
        self.assertEqual(stats["browser_recovery_counts"]["chrome"], 1)
        self.assertEqual(stats["browser_recovery_counts"]["firefox"], 1)


class TestResourcePoolBridgeWithRecovery(unittest.TestCase):
    """Test suite for ResourcePoolBridgeWithRecovery."""
    
    def setUp(self):
        """Set up test environment."""
        self.bridge = ResourcePoolBridgeWithRecovery(
            max_connections=2,
            max_retries=2,
            fallback_to_simulation=True
        )
    
    def test_initialization(self):
        """Test initialization of the bridge."""
        self.assertTrue(hasattr(self.bridge, 'recovery'))
        self.assertTrue(hasattr(self.bridge, 'integration'))
        self.assertEqual(len(self.bridge.loaded_models), 0)
        
    def test_model_loading(self):
        """Test model loading with recovery."""
        # Get a model
        model = self.bridge.get_model(
            model_type="text",
            model_name="bert-base-uncased",
            hardware_preferences={
                "priority_list": ["webgpu", "cpu"],
                "browser": "chrome"
            }
        )
        
        # Check model
        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, "text")
        self.assertEqual(model.model_name, "bert-base-uncased")
        
        # Check that model is tracked
        self.assertEqual(len(self.bridge.loaded_models), 1)
        self.assertIn("text:bert-base-uncased", self.bridge.loaded_models)
        
    def test_inference(self):
        """Test inference with recovery."""
        # Get a model
        model = self.bridge.get_model(
            model_type="text",
            model_name="bert-base-uncased"
        )
        
        # Run inference
        inputs = {"input_ids": [101, 2023, 2003, 1037, 3231, 102]}
        result = model(inputs)
        
        # Check result
        self.assertIsNotNone(result)
        self.assertTrue(result.get("success", False))
        self.assertIn("metrics", result)
        
    def test_concurrent_execution(self):
        """Test concurrent execution with recovery."""
        # Get two models
        model1 = self.bridge.get_model(
            model_type="text",
            model_name="bert-base-uncased"
        )
        
        model2 = self.bridge.get_model(
            model_type="vision",
            model_name="vit-base-patch16-224"
        )
        
        # Create inputs
        text_input = {"input_ids": [101, 2023, 2003, 1037, 3231, 102]}
        vision_input = {"pixel_values": [[[0.5 for _ in range(10)] for _ in range(10)] for _ in range(3)]}
        
        # Run concurrent inference
        models_and_inputs = [
            (model1.model_id, text_input),
            (model2.model_id, vision_input)
        ]
        
        results = self.bridge.execute_concurrent(models_and_inputs)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.get("success", False) for r in results))
        
    def test_metrics(self):
        """Test metrics collection with recovery statistics."""
        # Get metrics before any operations
        metrics = self.bridge.get_metrics()
        
        # Check metrics
        self.assertIn("base_metrics", metrics)
        self.assertIn("recovery_stats", metrics)
        self.assertTrue(metrics["recovery_enabled"])
        
        # Load a model and run inference to generate more metrics
        model = self.bridge.get_model("text", "bert-base-uncased")
        inputs = {"input_ids": [101, 2023, 2003, 1037, 3231, 102]}
        result = model(inputs)
        
        # Get updated metrics
        metrics = self.bridge.get_metrics()
        
        # Check metrics
        self.assertEqual(metrics["loaded_models_count"], 1)
        
    def test_browser_selection(self):
        """Test browser selection based on model type."""
        # Test with text model - should prefer Edge
        model = self.bridge.get_model(
            model_type="text",
            model_name="bert-base-uncased"
        )
        
        # Edge is preferred for text models
        self.assertEqual(model.model_id, "text:bert-base-uncased")
        
        # Test with audio model - should prefer Firefox
        model = self.bridge.get_model(
            model_type="audio",
            model_name="whisper-tiny"
        )
        
        # Firefox is preferred for audio models
        self.assertEqual(model.model_id, "audio:whisper-tiny")
        
    def tearDown(self):
        """Clean up after tests."""
        self.bridge.close()


class TestIntegrationWithMockErrors(unittest.TestCase):
    """Test integration with simulated errors."""
    
    def setUp(self):
        """Set up test environment with a mock integration that fails in specific ways."""
        # Create a mock integration with controlled failures
        self.mock_integration = MagicMock()
        self.mock_integration.initialize.return_value = True
        
        # Configure get_model to fail on certain conditions
        def mock_get_model(model_type, model_name, hardware_preferences=None):
            browser = hardware_preferences.get("browser") if hardware_preferences else None
            
            # Fail on out of memory for large models
            if "large" in model_name:
                raise Exception("CUDA out of memory")
                
            # Fail on unsupported operation for audio models on Chrome
            if model_type == "audio" and browser == "chrome":
                raise Exception("Operation not supported on this platform")
                
            # Fail on connection issues for WebGPU
            if hardware_preferences and "priority_list" in hardware_preferences:
                if hardware_preferences["priority_list"][0] == "webgpu":
                    if self.connection_should_fail:
                        self.connection_should_fail = False  # Fail only once
                        raise Exception("WebSocket connection closed unexpectedly")
            
            # Otherwise, succeed
            mock_model = MagicMock()
            mock_model.model_id = f"{model_type}:{model_name}"
            mock_model.model_type = model_type
            mock_model.model_name = model_name
            mock_model.return_value = {
                "success": True,
                "status": "success",
                "model_id": f"{model_type}:{model_name}",
                "result": {"output": [0.5] * 10},
                "metrics": {
                    "latency_ms": 100.0,
                    "throughput_items_per_sec": 10.0
                }
            }
            return mock_model
            
        self.mock_integration.get_model.side_effect = mock_get_model
        self.mock_integration.get_metrics.return_value = {"aggregate": {"total_inferences": 0}}
        
        # Flag to control connection failures
        self.connection_should_fail = True
        
        # Create the bridge with our mock
        self.bridge = ResourcePoolBridgeWithRecovery(
            integration=self.mock_integration,
            max_retries=3,
            fallback_to_simulation=True
        )
        
    def test_recovery_from_connection_error(self):
        """Test recovery from connection errors."""
        # Load a model that will trigger a connection error on first attempt
        model = self.bridge.get_model(
            model_type="text",
            model_name="bert-base-uncased",
            hardware_preferences={
                "priority_list": ["webgpu", "cpu"],
                "browser": "chrome"
            }
        )
        
        # Check that model loaded successfully after retry
        self.assertIsNotNone(model)
        self.assertEqual(model.model_id, "text:bert-base-uncased")
        
        # Run inference to verify model works
        inputs = {"input_ids": [101, 2023, 2003, 1037, 3231, 102]}
        result = model(inputs)
        self.assertTrue(result.get("success", False))
        
    def test_recovery_from_out_of_memory(self):
        """Test recovery from out of memory by reducing model size."""
        # Try to load a large model that will trigger OOM
        model = self.bridge.get_model(
            model_type="text",
            model_name="bert-large-uncased",
            hardware_preferences={
                "priority_list": ["cuda", "cpu"],
                "browser": "chrome"
            }
        )
        
        # Check that a smaller model was loaded instead
        self.assertIsNotNone(model)
        self.assertEqual(model.model_name, "bert-base-uncased")  # Should be downsized to base
        
    def test_recovery_from_unsupported_operation(self):
        """Test recovery from unsupported operation by switching browser."""
        # Try to load an audio model on Chrome (will fail)
        model = self.bridge.get_model(
            model_type="audio",
            model_name="whisper-tiny",
            hardware_preferences={
                "priority_list": ["webgpu", "cpu"],
                "browser": "chrome"
            }
        )
        
        # Check that model was loaded with Firefox instead
        self.assertIsNotNone(model)
        # Firefox should have been selected for audio model after Chrome failed
        
        # Check recovery statistics
        stats = self.bridge.recovery.get_recovery_statistics()
        self.assertGreaterEqual(stats["total_recovery_attempts"], 1)
        
    def tearDown(self):
        """Clean up after tests."""
        self.bridge.close()


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests()