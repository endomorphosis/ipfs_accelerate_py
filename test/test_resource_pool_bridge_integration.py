#!/usr/bin/env python3
"""
Test script for Resource Pool Bridge Integration with Recovery

This script tests the integration of the ResourcePoolBridgeRecovery system
with the main WebNN/WebGPU Resource Pool Bridge.

Usage:
    python test_resource_pool_bridge_integration.py
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

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fixed_web_platform.resource_pool_bridge_integration import ResourcePoolBridgeIntegrationWithRecovery

# Import recovery system for testing
try:
    from resource_pool_bridge_recovery import (
        ResourcePoolBridgeRecovery,
        ResourcePoolBridgeWithRecovery,
        ErrorCategory,
        RecoveryStrategy
    )
    RECOVERY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import recovery system: {e}")
    RECOVERY_AVAILABLE = False


# Mock base resource pool bridge
class MockResourcePoolBridge:
    """Mock ResourcePoolBridgeIntegration for testing."""
    
    def __init__(self, **kwargs):
        self.initialized = False
        self.max_connections = kwargs.get('max_connections', 4)
        self.browser_preferences = kwargs.get('browser_preferences', {})
        self.models = {}
        
    async def initialize(self):
        self.initialized = True
        return True
        
    async def get_model(self, model_type, model_name, hardware_preferences=None):
        model_id = f"{model_type}:{model_name}"
        # Create simple callable mock
        model = MagicMock()
        model.model_id = model_id
        model.model_name = model_name
        model.model_type = model_type
        model.return_value = {
            "success": True, 
            "model_id": model_id,
            "result": {"output": [0.5] * 10},
            "metrics": {
                "latency_ms": 100.0,
                "throughput_items_per_sec": 10.0
            }
        }
        self.models[model_id] = model
        return model
    
    async def execute_concurrent(self, models_and_inputs):
        results = []
        for model_id, inputs in models_and_inputs:
            if model_id in self.models:
                result = self.models[model_id](inputs)
                results.append(result)
            else:
                results.append({"success": False, "error": f"Model {model_id} not found"})
        return results
    
    def execute_concurrent_sync(self, models_and_inputs):
        import asyncio
        loop = # TODO: Remove event loop management - asyncio.new_event_loop()
        return loop.run_until_complete(self.execute_concurrent(models_and_inputs))
    
    def get_metrics(self):
        return {"aggregate": {"total_inferences": len(self.models)}}
    
    async def get_health_status(self):
        return {"status": "healthy"}
    
    def get_health_status_sync(self):
        return {"status": "healthy"}
    
    async def close(self):
        self.initialized = False
        return True
    
    def close_sync(self):
        self.initialized = False
        return True
    
    def setup_tensor_sharing(self, max_memory_mb=None):
        return {"success": True, "max_memory_mb": max_memory_mb or 1024}
    
    async def share_tensor_between_models(self, tensor_data, tensor_name, producer_model, consumer_models, **kwargs):
        return {
            "success": True,
            "tensor_name": tensor_name,
            "producer": producer_model,
            "consumers": consumer_models
        }


@unittest.skipIf(not RECOVERY_AVAILABLE, "Recovery system not available")
class TestResourcePoolBridgeIntegration(unittest.TestCase):
    """
    Test suite for ResourcePoolBridgeIntegrationWithRecovery.
    
    This test suite verifies the integration between the ResourcePoolBridgeIntegration
    and the ResourcePoolBridgeRecovery system.
    """
    
    @patch("fixed_web_platform.resource_pool_bridge_integration.ResourcePoolBridgeIntegration", MockResourcePoolBridge)
    def setUp(self):
        """Set up test environment with mocked resource pool."""
        self.integration = ResourcePoolBridgeIntegrationWithRecovery(
            max_connections=2,
            adaptive_scaling=True,
            enable_recovery=True,
            max_retries=2,
            fallback_to_simulation=True
        )
        self.integration.initialize()
    
    def test_initialization(self):
        """Test successful initialization."""
        self.assertTrue(self.integration.initialized)
        self.assertIsNotNone(self.integration.bridge)
        self.assertIsNotNone(self.integration.bridge_with_recovery)
        self.assertTrue(self.integration.enable_recovery)
    
    def test_get_model(self):
        """Test model retrieval with recovery."""
        # Get a model
        model = self.integration.get_model(
            model_type="text",
            model_name="bert-base-uncased",
            hardware_preferences={
                "priority_list": ["webgpu", "cpu"],
                "browser": "chrome"
            }
        )
        
        # Check model
        self.assertIsNotNone(model)
        self.assertEqual(model.model_id, "text:bert-base-uncased")
        
        # Run inference
        result = model({"input_ids": [1, 2, 3]})
        self.assertTrue(result.get("success", False))
    
    def test_execute_concurrent(self):
        """Test concurrent model execution with recovery."""
        # Get two models
        text_model = self.integration.get_model("text", "bert-base-uncased")
        vision_model = self.integration.get_model("vision", "vit-base-patch16-224")
        
        # Create input data
        text_input = {"input_ids": [1, 2, 3]}
        vision_input = {"pixel_values": [[[0.5]]]}
        
        # Run concurrent execution
        model_inputs = [
            (text_model.model_id, text_input),
            (vision_model.model_id, vision_input)
        ]
        
        results = self.integration.execute_concurrent(model_inputs)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.get("success", False) for r in results))
    
    def test_get_metrics(self):
        """Test metrics collection."""
        # Load a model to generate metrics
        self.integration.get_model("text", "bert-base-uncased")
        
        # Get metrics
        metrics = self.integration.get_metrics()
        
        # Check metrics
        self.assertTrue(metrics.get("recovery_enabled", False))
        self.assertTrue(metrics.get("initialized", False))
        
        # Should have recovery stats with no attempt
        if "recovery_stats" in metrics:
            self.assertEqual(metrics["recovery_stats"]["total_recovery_attempts"], 0)
    
    def test_get_health_status(self):
        """Test health status reporting."""
        health = self.integration.get_health_status()
        self.assertEqual(health.get("status"), "healthy")
    
    def test_tensor_sharing(self):
        """Test tensor sharing functionality."""
        # Setup tensor sharing
        result = self.integration.setup_tensor_sharing(max_memory_mb=1024)
        self.assertIsNotNone(result)
        
        # Get models
        text_model = self.integration.get_model("text", "bert-base-uncased")
        vision_model = self.integration.get_model("vision", "vit-base-patch16-224")
        
        # Share tensor between models
        tensor_data = [0.1, 0.2, 0.3]
        result = self.integration.share_tensor_between_models(
            tensor_data=tensor_data,
            tensor_name="test_tensor",
            producer_model=text_model,
            consumer_models=[vision_model],
            shape=[3],
            storage_type="cpu"
        )
        
        # Check result
        self.assertTrue(result.get("success", False))
        self.assertEqual(result.get("tensor_name"), "test_tensor")
    
    def tearDown(self):
        """Clean up resources."""
        self.integration.close()


@unittest.skipIf(not RECOVERY_AVAILABLE, "Recovery system not available")
class TestRecoveryIntegrationWithMockedErrors(unittest.TestCase):
    """
    Test suite for ResourcePoolBridgeIntegrationWithRecovery with error scenarios.
    
    This test suite verifies how the integration handles various error conditions
    and recovery scenarios.
    """
    
    def setUp(self):
        """Set up test environment with custom mock."""
        # Create a custom mock that simulates specific errors
        self.mock_bridge = MagicMock()
        self.mock_bridge.initialize.return_value = True
        
        # Configure get_model to fail on certain conditions
        def mock_get_model(model_type, model_name, hardware_preferences=None):
            if model_type == "text" and model_name == "bert-large-uncased":
                # Simulate out of memory error for large models
                raise Exception("CUDA out of memory")
            
            if model_type == "audio" and hardware_preferences and hardware_preferences.get("browser") == "chrome":
                # Simulate unsupported operation for audio on Chrome
                raise Exception("Operation not supported on this platform")
                
            if model_type == "vision" and self.connection_failures > 0:
                # Simulate connection errors
                self.connection_failures -= 1
                raise Exception("WebSocket connection closed unexpectedly")
            
            # Otherwise return a mock model
            model = MagicMock()
            model.model_id = f"{model_type}:{model_name}"
            model.model_type = model_type
            model.model_name = model_name
            model.return_value = {
                "success": True,
                "model_id": f"{model_type}:{model_name}",
                "result": {"output": [0.5] * 10}
            }
            return model
            
        async def async_mock_get_model(*args, **kwargs):
            return mock_get_model(*args, **kwargs)
            
        self.mock_bridge.get_model = async_mock_get_model
        self.mock_bridge.get_metrics.return_value = {"fake_metrics": True}
        
        # Create patched integration
        with patch("fixed_web_platform.resource_pool_bridge_integration.ResourcePoolBridgeIntegration", return_value=self.mock_bridge):
            self.integration = ResourcePoolBridgeIntegrationWithRecovery(
                max_connections=2,
                enable_recovery=True,
                max_retries=3,
                fallback_to_simulation=True
            )
            self.integration.bridge = self.mock_bridge
            
            # Manually create and configure the recovery bridge
            self.recovery = ResourcePoolBridgeWithRecovery(
                integration=self.mock_bridge,
                max_connections=2,
                max_retries=3,
                fallback_to_simulation=True
            )
            self.integration.bridge_with_recovery = self.recovery
            self.integration.initialized = True
            
        # Counter for connection failures
        self.connection_failures = 1
        
    def test_recovery_from_connection_error(self):
        """Test recovery from WebSocket connection errors."""
        # Set up to fail once with connection error
        self.connection_failures = 1
        
        # Get vision model (should trigger connection error and recover)
        model = self.integration.get_model(
            model_type="vision",
            model_name="vit-base-patch16-224"
        )
        
        # Should recover after one failure
        self.assertIsNotNone(model)
        self.assertEqual(model.model_id, "vision:vit-base-patch16-224")
        
        # Check recovery statistics
        metrics = self.integration.get_metrics()
        if "recovery_stats" in metrics:
            self.assertGreaterEqual(metrics["recovery_stats"]["total_recovery_attempts"], 1)
    
    def test_recovery_from_out_of_memory(self):
        """Test recovery from out-of-memory errors."""
        # Try to load a large model that will trigger OOM
        model = self.integration.get_model(
            model_type="text",
            model_name="bert-large-uncased"
        )
        
        # Should downsize to base model or provide a fallback
        self.assertIsNotNone(model)
        
        # Check recovery statistics
        metrics = self.integration.get_metrics()
        if "recovery_stats" in metrics:
            self.assertGreaterEqual(metrics["recovery_stats"]["total_recovery_attempts"], 1)
            if "error_categories" in metrics["recovery_stats"]:
                self.assertIn("out_of_memory", metrics["recovery_stats"]["error_categories"])
    
    def test_recovery_from_unsupported_operation(self):
        """Test recovery from unsupported operation errors."""
        # Try to load an audio model with Chrome (will fail)
        model = self.integration.get_model(
            model_type="audio",
            model_name="whisper-tiny",
            hardware_preferences={
                "browser": "chrome"
            }
        )
        
        # Should switch to Firefox or another browser that supports it
        self.assertIsNotNone(model)
        
        # Check recovery statistics
        metrics = self.integration.get_metrics()
        if "recovery_stats" in metrics:
            self.assertGreaterEqual(metrics["recovery_stats"]["total_recovery_attempts"], 1)
            if "error_categories" in metrics["recovery_stats"]:
                self.assertIn("unsupported_operation", metrics["recovery_stats"]["error_categories"])
    
    def tearDown(self):
        """Clean up resources."""
        self.integration.close()


@unittest.skipIf(not RECOVERY_AVAILABLE, "Recovery system not available")
class TestResourcePoolIntegrationWithRecoveryDisabled(unittest.TestCase):
    """
    Test suite for ResourcePoolBridgeIntegrationWithRecovery with recovery disabled.
    
    This test suite verifies the integration functions correctly when recovery
    capabilities are disabled.
    """
    
    @patch("fixed_web_platform.resource_pool_bridge_integration.ResourcePoolBridgeIntegration", MockResourcePoolBridge)
    def setUp(self):
        """Set up test environment with recovery disabled."""
        self.integration = ResourcePoolBridgeIntegrationWithRecovery(
            max_connections=2,
            adaptive_scaling=True,
            enable_recovery=False,  # Disable recovery
            max_retries=2,
            fallback_to_simulation=True
        )
        self.integration.initialize()
    
    def test_initialization(self):
        """Test successful initialization with recovery disabled."""
        self.assertTrue(self.integration.initialized)
        self.assertIsNotNone(self.integration.bridge)
        self.assertIsNone(self.integration.bridge_with_recovery)
        self.assertFalse(self.integration.enable_recovery)
    
    def test_get_model_without_recovery(self):
        """Test model retrieval without recovery."""
        # Get a model
        model = self.integration.get_model(
            model_type="text",
            model_name="bert-base-uncased"
        )
        
        # Check model
        self.assertIsNotNone(model)
        
        # Run inference
        result = model({"input_ids": [1, 2, 3]})
        self.assertTrue(result.get("success", False))
    
    def test_get_metrics_without_recovery(self):
        """Test metrics collection without recovery."""
        # Get metrics
        metrics = self.integration.get_metrics()
        
        # Check metrics has base metrics but no recovery stats
        self.assertFalse(metrics.get("recovery_enabled", True))
        self.assertIn("base_metrics", metrics)
        self.assertNotIn("recovery_stats", metrics)
    
    def tearDown(self):
        """Clean up resources."""
        self.integration.close()


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests()