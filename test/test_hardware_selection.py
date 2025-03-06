#!/usr/bin/env python
"""
Test the hardware selection system.

This module tests the hardware selector class to ensure it correctly recommends
hardware for various models and scenarios, including fallback functionality when
prediction models aren't available.
"""

import os
import sys
import unittest
import tempfile
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import hardware selector
from hardware_selector import HardwareSelector

# Configure logging
logging.basicConfig(level=logging.INFO)


class TestHardwareSelector(unittest.TestCase):
    """Test cases for the hardware selector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for benchmark data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.benchmark_path = os.path.join(self.temp_dir.name, "benchmark_results")
        os.makedirs(self.benchmark_path, exist_ok=True)
        
        # Create empty benchmark files
        os.makedirs(os.path.join(self.benchmark_path, "raw_results"), exist_ok=True)
        os.makedirs(os.path.join(self.benchmark_path, "processed_results"), exist_ok=True)
        
        # Create compatibility matrix
        self.compatibility_matrix = {
            "timestamp": "2025-03-01T00:00:00Z",
            "hardware_types": ["cpu", "cuda", "rocm", "mps", "openvino", "webnn", "webgpu"],
            "model_families": {
                "embedding": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "medium"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": True, "performance_rating": "high"},
                        "mps": {"compatible": True, "performance_rating": "high"},
                        "openvino": {"compatible": True, "performance_rating": "medium"},
                        "webnn": {"compatible": True, "performance_rating": "high"},
                        "webgpu": {"compatible": True, "performance_rating": "medium"}
                    }
                },
                "text_generation": {
                    "hardware_compatibility": {
                        "cpu": {"compatible": True, "performance_rating": "low"},
                        "cuda": {"compatible": True, "performance_rating": "high"},
                        "rocm": {"compatible": True, "performance_rating": "medium"},
                        "mps": {"compatible": True, "performance_rating": "medium"},
                        "openvino": {"compatible": True, "performance_rating": "low"},
                        "webnn": {"compatible": False, "performance_rating": "unknown"},
                        "webgpu": {"compatible": True, "performance_rating": "low"}
                    }
                }
            }
        }
        
        with open(os.path.join(self.benchmark_path, "hardware_compatibility_matrix.json"), "w") as f:
            json.dump(self.compatibility_matrix, f)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test basic initialization of the hardware selector."""
        selector = HardwareSelector(database_path=self.benchmark_path)
        self.assertIsNotNone(selector)
        self.assertEqual(selector.database_path, Path(self.benchmark_path))
        self.assertIn("embedding", selector.compatibility_matrix["model_families"])
    
    def test_hardware_selection_basic(self):
        """Test basic hardware selection without prediction models."""
        selector = HardwareSelector(database_path=self.benchmark_path)
        
        # Test with embedding model
        result = selector.select_hardware(
            model_family="embedding",
            model_name="bert-base-uncased",
            batch_size=1,
            mode="inference",
            available_hardware=["cpu", "cuda", "openvino"]
        )
        
        self.assertIn("primary_recommendation", result)
        self.assertIn("fallback_options", result)
        self.assertIn("compatible_hardware", result)
        self.assertGreater(len(result["compatible_hardware"]), 0)
        self.assertEqual(len(result["fallback_options"]), 2)
        
        # For embedding models, CUDA should be recommended for inference
        self.assertEqual(result["primary_recommendation"], "cuda")
        
        # Test with text generation model
        result = selector.select_hardware(
            model_family="text_generation",
            model_name="gpt2",
            batch_size=1,
            mode="inference",
            available_hardware=["cpu", "cuda", "openvino"]
        )
        
        # For text generation models, CUDA should also be recommended
        self.assertEqual(result["primary_recommendation"], "cuda")
    
    def test_hardware_selection_with_sklearn_unavailable(self):
        """Test hardware selection when scikit-learn is unavailable."""
        # Mock sklearn import to simulate unavailability
        with patch.dict("sys.modules", {"sklearn": None}):
            selector = HardwareSelector(database_path=self.benchmark_path)
            
            # Test with embedding model
            result = selector.select_hardware(
                model_family="embedding",
                model_name="bert-base-uncased",
                batch_size=1,
                mode="inference",
                available_hardware=["cpu", "cuda", "openvino"]
            )
            
            # Even without sklearn, we should still get recommendations
            self.assertIn("primary_recommendation", result)
            self.assertEqual(result["primary_recommendation"], "cuda")
    
    def test_hardware_selection_with_fallback_models(self):
        """Test hardware selection with fallback prediction models."""
        # Create a selector with fallback models instead of trained models
        selector = HardwareSelector(database_path=self.benchmark_path)
        
        # Directly initialize fallback models
        selector._initialize_fallback_models("inference")
        selector._initialize_fallback_models("training")
        
        # Test with embedding model
        result = selector.select_hardware(
            model_family="embedding",
            model_name="bert-base-uncased",
            batch_size=1,
            mode="inference",
            available_hardware=["cpu", "cuda", "openvino"]
        )
        
        # We should still get recommendations
        self.assertIn("primary_recommendation", result)
        self.assertEqual(result["primary_recommendation"], "cuda")
        
        # Test with different batch sizes
        result_large_batch = selector.select_hardware(
            model_family="embedding",
            model_name="bert-base-uncased",
            batch_size=64,
            mode="inference",
            available_hardware=["cpu", "cuda", "openvino"]
        )
        
        # Larger batch sizes should still recommend cuda
        self.assertEqual(result_large_batch["primary_recommendation"], "cuda")
    
    def test_distributed_training_config(self):
        """Test generation of distributed training configuration."""
        selector = HardwareSelector(database_path=self.benchmark_path)
        
        # Test with small model
        config = selector.get_distributed_training_config(
            model_family="text_generation",
            model_name="gpt2",
            gpu_count=4,
            batch_size=8
        )
        
        # Check that config has the expected fields
        self.assertEqual(config["model_family"], "text_generation")
        self.assertEqual(config["model_name"], "gpt2")
        self.assertEqual(config["gpu_count"], 4)
        self.assertEqual(config["per_gpu_batch_size"], 8)
        self.assertEqual(config["global_batch_size"], 32)
        self.assertIn("distributed_strategy", config)
        self.assertIn("estimated_memory", config)
        
        # Test with large model and memory constraints
        config = selector.get_distributed_training_config(
            model_family="text_generation",
            model_name="llama-7b",
            gpu_count=4,
            batch_size=8,
            max_memory_gb=16
        )
        
        # Should include memory optimizations
        self.assertIn("memory_optimizations", config)
        self.assertTrue(len(config["memory_optimizations"]) > 0)
    
    def test_create_selection_map(self):
        """Test creation of hardware selection map."""
        selector = HardwareSelector(database_path=self.benchmark_path)
        
        # Create selection map
        selection_map = selector.create_hardware_selection_map(["embedding"])
        
        # Check that map has the expected structure
        self.assertIn("model_families", selection_map)
        self.assertIn("embedding", selection_map["model_families"])
        self.assertIn("model_sizes", selection_map["model_families"]["embedding"])
        self.assertIn("inference", selection_map["model_families"]["embedding"])
        self.assertIn("training", selection_map["model_families"]["embedding"])


if __name__ == "__main__":
    unittest.main()