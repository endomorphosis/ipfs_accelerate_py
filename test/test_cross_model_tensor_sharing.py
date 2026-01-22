#!/usr/bin/env python3
"""
Test for Cross-Model Tensor Sharing functionality.

This test verifies the functionality of the Cross-Model Tensor Sharing system,
which enables efficient tensor sharing across multiple models in the WebGPU/WebNN
resource pool.
"""

import os
import sys
import json
import time
import unittest
from typing import Dict, List, Any, Set

# Import modules from current directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import tensor sharing implementation
from fixed_web_platform.cross_model_tensor_sharing import (
    SharedTensor,
    SharedTensorView,
    TensorSharingManager,
    get_compatible_models_for_tensor
)

# Import resource pool implementation
from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration

class TestCrossModelTensorSharing(unittest.TestCase):
    """Test Cross-Model Tensor Sharing functionality."""
    
    def setUp(self):
        """Set up for tests."""
        self.manager = TensorSharingManager(max_memory_mb=1024)
    
    def test_tensor_creation(self):
        """Test creating shared tensors."""
        tensor = self.manager.register_shared_tensor(
            name="test_tensor",
            shape=[1, 768],
            storage_type="cpu",
            producer_model="bert",
            consumer_models=["t5"],
            dtype="float32"
        )
        
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.name, "test_tensor")
        self.assertEqual(tensor.shape, [1, 768])
        self.assertEqual(tensor.storage_type, "cpu")
        self.assertEqual(tensor.producer_model, "bert")
        self.assertEqual(tensor.reference_count, 1)  # Producer model has a reference
        self.assertIn("bert", tensor.consumer_models)
    
    def test_tensor_view(self):
        """Test creating tensor views."""
        # Create parent tensor
        parent = self.manager.register_shared_tensor(
            name="parent_tensor",
            shape=[1, 768],
            storage_type="cpu",
            producer_model="bert"
        )
        
        # Create view
        view = self.manager.create_tensor_view(
            tensor_name="parent_tensor",
            view_name="half_view",
            offset=[0, 0],
            size=[1, 384],
            model_name="t5"
        )
        
        self.assertIsNotNone(view)
        self.assertEqual(view.name, "half_view")
        self.assertEqual(view.offset, [0, 0])
        self.assertEqual(view.size, [1, 384])
        self.assertEqual(view.parent, parent)
        self.assertEqual(view.reference_count, 1)
        self.assertIn("t5", view.consumer_models)
        
        # Creating a view should also acquire the parent tensor
        self.assertIn("t5", parent.consumer_models)
    
    def test_tensor_sharing(self):
        """Test sharing tensors between models."""
        # Create shared tensor
        tensor = self.manager.register_shared_tensor(
            name="shared_tensor",
            shape=[1, 1024],
            storage_type="cpu",
            producer_model="vit",
            consumer_models=["clip"]
        )
        
        # Share with additional models
        result = self.manager.share_tensor_between_models(
            tensor_name="shared_tensor",
            from_model="vit",
            to_models=["llava", "xclip"]
        )
        
        self.assertTrue(result)
        self.assertIn("shared_tensor", self.manager.model_tensors.get("llava", set()))
        self.assertIn("shared_tensor", self.manager.model_tensors.get("xclip", set()))
    
    def test_memory_optimization(self):
        """Test memory optimization."""
        # Create tensors
        for i in range(5):
            self.manager.register_shared_tensor(
                name=f"tensor_{i}",
                shape=[1, 768],
                storage_type="cpu",
                producer_model=f"model_{i}",
                consumer_models=[]
            )
        
        # Release all models to make tensors available for freeing
        for i in range(5):
            self.manager.release_model_tensors(f"model_{i}")
        
        # Wait a bit to exceed the grace period (in a real test we'd mock time)
        # For this test, we'll temporarily modify the tensor's last_accessed
        for tensor in self.manager.tensors.values():
            tensor.last_accessed = time.time() - 31  # Just over 30 second grace period
        
        # Run optimization
        result = self.manager.optimize_memory_usage()
        
        self.assertEqual(result["freed_tensors_count"], 5)
        self.assertGreater(result["freed_memory_bytes"], 0)
        self.assertEqual(len(self.manager.tensors), 0)
    
    def test_analyzing_sharing_opportunities(self):
        """Test analyzing sharing opportunities."""
        # Register models that could potentially share tensors
        self.manager.register_shared_tensor(
            name="bert_emb",
            shape=[1, 768],
            storage_type="cpu",
            producer_model="bert"
        )
        
        self.manager.register_shared_tensor(
            name="t5_emb",
            shape=[1, 768],
            storage_type="cpu",
            producer_model="t5"
        )
        
        # Analyze opportunities
        opportunities = self.manager.analyze_sharing_opportunities()
        
        # Should identify "text_embedding" as shareable
        self.assertIn("text_embedding", opportunities)
        self.assertIn("bert", opportunities["text_embedding"])
        self.assertIn("t5", opportunities["text_embedding"])
    
    def test_get_compatible_models(self):
        """Test getting compatible models for a tensor type."""
        # Test text embedding compatibility
        text_models = get_compatible_models_for_tensor("text_embedding")
        self.assertIn("bert", text_models)
        self.assertIn("t5", text_models)
        self.assertIn("llama", text_models)
        
        # Test vision embedding compatibility
        vision_models = get_compatible_models_for_tensor("vision_embedding")
        self.assertIn("vit", vision_models)
        self.assertIn("clip", vision_models)
        
        # Test multimodal embedding compatibility
        multimodal_models = get_compatible_models_for_tensor("vision_text_joint")
        self.assertIn("clip", multimodal_models)
        self.assertIn("llava", multimodal_models)
    
    def test_release_model_tensors(self):
        """Test releasing all tensors for a model."""
        # Create multiple tensors for a model
        for i in range(3):
            self.manager.register_shared_tensor(
                name=f"model_tensor_{i}",
                shape=[1, 768],
                storage_type="cpu",
                producer_model="test_model"
            )
        
        # Release all tensors for the model
        released = self.manager.release_model_tensors("test_model")
        
        self.assertEqual(released, 3)
        self.assertNotIn("test_model", self.manager.model_tensors)
    
    def test_tensor_memory_usage(self):
        """Test calculating tensor memory usage."""
        # Create tensor with known size
        tensor = SharedTensor(
            name="memory_test",
            shape=[1, 1024],  # 1x1024 float32 = 4096 bytes
            dtype="float32",
            storage_type="cpu"
        )
        
        # Check memory calculation
        expected_bytes = 1 * 1024 * 4  # 1 * 1024 elements * 4 bytes per float32
        self.assertEqual(tensor.get_memory_usage(), expected_bytes)
        
        # Test with float16
        tensor.dtype = "float16"
        expected_bytes = 1 * 1024 * 2  # 1 * 1024 elements * 2 bytes per float16
        self.assertEqual(tensor.get_memory_usage(), expected_bytes)
    
    def test_stats_collection(self):
        """Test collecting usage statistics."""
        # Create some tensors
        self.manager.register_shared_tensor(
            name="stats_tensor_1",
            shape=[1, 768],
            storage_type="cpu",
            producer_model="model_a",
            consumer_models=["model_b", "model_c"]
        )
        
        self.manager.register_shared_tensor(
            name="stats_tensor_2",
            shape=[1, 1024],
            storage_type="webgpu",
            producer_model="model_d"
        )
        
        # Access tensor to generate stats
        self.manager.get_shared_tensor("stats_tensor_1", "model_e")
        self.manager.get_shared_tensor("stats_tensor_1", "model_e")  # Access twice
        
        # Get stats
        stats = self.manager.get_stats()
        
        # Validate results
        self.assertEqual(stats["total_tensors"], 2)
        self.assertGreaterEqual(stats["total_models"], 5)  # a, b, c, d, e
        self.assertEqual(stats["cache_hits"], 2)
        self.assertEqual(stats["cache_misses"], 0)
        self.assertEqual(stats["hit_rate"], 1.0)  # 100% hit rate
    
    def test_get_optimization_recommendations(self):
        """Test getting optimization recommendations."""
        # Create some tensors
        for i in range(5):
            self.manager.register_shared_tensor(
                name=f"recommendation_tensor_{i}",
                shape=[1, 768 * (i+1)],  # Increasing sizes
                storage_type="cpu",
                producer_model=f"model_{i}"
            )
        
        # Release some models to create low-reference tensors
        self.manager.release_model_tensors("model_0")
        self.manager.release_model_tensors("model_1")
        
        # Get recommendations
        recommendations = self.manager.get_optimization_recommendations()
        
        # Largest tensor should be recommendation_tensor_4
        self.assertEqual(recommendations["largest_tensors"][0]["name"], "recommendation_tensor_4")
        
        # Should have low-reference tensors (actual number may vary)
        self.assertGreaterEqual(len(recommendations["low_reference_tensors"]), 2)


class TestResourcePoolIntegration(unittest.TestCase):
    """Test integration with ResourcePoolBridgeIntegration."""
    
    def setUp(self):
        """Set up for tests."""
        self.tensor_manager = TensorSharingManager(max_memory_mb=1024)
        self.resource_pool = ResourcePoolBridgeIntegration(
            max_connections=2,
            browser_preferences={
                "text": "edge",
                "vision": "chrome",
                "audio": "firefox"
            }
        )
    
    def test_integration_model_sharing(self):
        """
        Test integration with ResourcePoolBridgeIntegration for model sharing.
        
        This test verifies that models can share tensors through the integration.
        """
        # Create a shared embedding tensor
        embedding = self.tensor_manager.register_shared_tensor(
            name="bert_embedding",
            shape=[1, 768],
            storage_type="cpu",
            producer_model="bert",
            consumer_models=["t5"]
        )
        
        # TODO: Once integrated, we would call resource_pool methods here
        # For now, we'll just check that the tensor is properly registered
        
        self.assertEqual(embedding.name, "bert_embedding")
        self.assertEqual(embedding.reference_count, 1)  # Producer reference
        
        # Get the tensor for a consumer model
        retrieved = self.tensor_manager.get_shared_tensor("bert_embedding", "t5")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.reference_count, 2)  # Producer + consumer
        self.assertIn("t5", retrieved.consumer_models)


if __name__ == "__main__":
    unittest.main()