#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the model selection system in the refactored generator suite.
Tests the model registry, selector, and filters.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_selection.registry import ModelRegistry
from model_selection.selector import ModelSelector
from model_selection.filters import (
    TaskFilter,
    HardwareFilter,
    SizeFilter,
    FrameworkFilter
)


class ModelRegistryTest(unittest.TestCase):
    """Tests for the model registry component."""

    def setUp(self):
        """Set up test environment."""
        self.registry = ModelRegistry()
        
        # Register some test models
        self.registry.register_model("bert-base-uncased", {
            "name": "BERT Base Uncased",
            "id": "bert-base-uncased",
            "architecture": "encoder-only",
            "task": "fill-mask",
            "framework": "pt",
            "size_mb": 420,
            "recommended_tasks": ["fill-mask", "sequence-classification", "token-classification"],
            "frameworks": ["pt", "tf", "onnx"]
        })
        
        self.registry.register_model("gpt2", {
            "name": "GPT-2",
            "id": "gpt2",
            "architecture": "decoder-only",
            "task": "text-generation",
            "framework": "pt",
            "size_mb": 510,
            "recommended_tasks": ["text-generation", "text-completion"],
            "frameworks": ["pt", "tf"]
        })
        
        self.registry.register_model("t5-small", {
            "name": "T5 Small",
            "id": "t5-small",
            "architecture": "encoder-decoder",
            "task": "text2text-generation",
            "framework": "pt",
            "size_mb": 300,
            "recommended_tasks": ["text2text-generation", "translation", "summarization"],
            "frameworks": ["pt", "tf", "onnx"]
        })
        
        self.registry.register_model("vit-base-patch16-224", {
            "name": "ViT Base",
            "id": "vit-base-patch16-224",
            "architecture": "vision",
            "task": "image-classification",
            "framework": "pt",
            "size_mb": 340,
            "recommended_tasks": ["image-classification"],
            "frameworks": ["pt", "tf"]
        })

    def test_register_model(self):
        """Test registering a model."""
        # Register a new model
        self.registry.register_model("new-model", {
            "name": "New Model",
            "id": "new-model",
            "architecture": "encoder-only",
            "task": "fill-mask",
            "framework": "pt",
            "size_mb": 100,
            "recommended_tasks": ["fill-mask"],
            "frameworks": ["pt"]
        })
        
        # Verify the model was registered
        self.assertIn("new-model", self.registry.get_model_ids())
        self.assertEqual("New Model", self.registry.get_model("new-model")["name"])
        
        # Verify architecture mapping was updated
        encoder_only_models = self.registry.get_models_by_architecture("encoder-only")
        self.assertIn("new-model", [model["id"] for model in encoder_only_models])

    def test_get_model(self):
        """Test retrieving a model by ID."""
        # Get an existing model
        model = self.registry.get_model("bert-base-uncased")
        self.assertEqual("BERT Base Uncased", model["name"])
        self.assertEqual("encoder-only", model["architecture"])
        
        # Get a non-existent model
        model = self.registry.get_model("non-existent-model")
        self.assertIsNone(model)

    def test_get_models_by_architecture(self):
        """Test retrieving models by architecture."""
        # Get encoder-only models
        encoder_models = self.registry.get_models_by_architecture("encoder-only")
        self.assertEqual(1, len(encoder_models))
        self.assertEqual("bert-base-uncased", encoder_models[0]["id"])
        
        # Get decoder-only models
        decoder_models = self.registry.get_models_by_architecture("decoder-only")
        self.assertEqual(1, len(decoder_models))
        self.assertEqual("gpt2", decoder_models[0]["id"])
        
        # Get non-existent architecture
        non_existent_models = self.registry.get_models_by_architecture("non-existent")
        self.assertEqual(0, len(non_existent_models))

    def test_get_architectures(self):
        """Test retrieving all architectures."""
        architectures = self.registry.get_architectures()
        self.assertEqual(4, len(architectures))
        self.assertIn("encoder-only", architectures)
        self.assertIn("decoder-only", architectures)
        self.assertIn("encoder-decoder", architectures)
        self.assertIn("vision", architectures)

    def test_get_model_ids(self):
        """Test retrieving all model IDs."""
        model_ids = self.registry.get_model_ids()
        self.assertEqual(4, len(model_ids))
        self.assertIn("bert-base-uncased", model_ids)
        self.assertIn("gpt2", model_ids)
        self.assertIn("t5-small", model_ids)
        self.assertIn("vit-base-patch16-224", model_ids)


class ModelFiltersTest(unittest.TestCase):
    """Tests for the model selection filters."""

    def setUp(self):
        """Set up test models for filtering."""
        self.test_models = [
            {
                "name": "BERT Base Uncased",
                "id": "bert-base-uncased",
                "architecture": "encoder-only",
                "task": "fill-mask",
                "framework": "pt",
                "size_mb": 420,
                "recommended_tasks": ["fill-mask", "sequence-classification", "token-classification"],
                "frameworks": ["pt", "tf", "onnx"],
                "hardware": {
                    "min_cuda_compute": 3.5,
                    "min_ram_gb": 4,
                    "recommended_gpu_vram_gb": 2
                }
            },
            {
                "name": "GPT-2",
                "id": "gpt2",
                "architecture": "decoder-only",
                "task": "text-generation",
                "framework": "pt",
                "size_mb": 510,
                "recommended_tasks": ["text-generation", "text-completion"],
                "frameworks": ["pt", "tf"],
                "hardware": {
                    "min_cuda_compute": 5.0,
                    "min_ram_gb": 8,
                    "recommended_gpu_vram_gb": 4
                }
            },
            {
                "name": "T5 Small",
                "id": "t5-small",
                "architecture": "encoder-decoder",
                "task": "text2text-generation",
                "framework": "pt",
                "size_mb": 300,
                "recommended_tasks": ["text2text-generation", "translation", "summarization"],
                "frameworks": ["pt", "tf", "onnx"],
                "hardware": {
                    "min_cuda_compute": 3.5,
                    "min_ram_gb": 4,
                    "recommended_gpu_vram_gb": 2
                }
            },
            {
                "name": "ViT Base",
                "id": "vit-base-patch16-224",
                "architecture": "vision",
                "task": "image-classification",
                "framework": "pt",
                "size_mb": 340,
                "recommended_tasks": ["image-classification"],
                "frameworks": ["pt", "tf"],
                "hardware": {
                    "min_cuda_compute": 3.5,
                    "min_ram_gb": 4,
                    "recommended_gpu_vram_gb": 2
                }
            }
        ]
        
        # Initialize filters
        self.task_filter = TaskFilter()
        self.hardware_filter = HardwareFilter()
        self.size_filter = SizeFilter()
        self.framework_filter = FrameworkFilter()

    def test_task_filter(self):
        """Test filtering models by task."""
        # Filter for fill-mask task
        fill_mask_models = self.task_filter.filter(self.test_models, "fill-mask")
        self.assertEqual(1, len(fill_mask_models))
        self.assertEqual("bert-base-uncased", fill_mask_models[0]["id"])
        
        # Filter for text-generation task
        text_gen_models = self.task_filter.filter(self.test_models, "text-generation")
        self.assertEqual(1, len(text_gen_models))
        self.assertEqual("gpt2", text_gen_models[0]["id"])
        
        # Filter for non-existent task
        non_existent_task_models = self.task_filter.filter(self.test_models, "non-existent-task")
        self.assertEqual(0, len(non_existent_task_models))
        
        # No filter (should return all models)
        all_models = self.task_filter.filter(self.test_models, None)
        self.assertEqual(4, len(all_models))

    def test_hardware_filter(self):
        """Test filtering models by hardware requirements."""
        # Filter for basic hardware (should match all models)
        basic_hw_profile = {
            "cuda_compute": 6.0,
            "ram_gb": 16,
            "gpu_vram_gb": 8
        }
        basic_models = self.hardware_filter.filter(self.test_models, basic_hw_profile)
        self.assertEqual(4, len(basic_models))
        
        # Filter for limited hardware (should exclude GPT-2)
        limited_hw_profile = {
            "cuda_compute": 4.0,
            "ram_gb": 4,
            "gpu_vram_gb": 2
        }
        limited_models = self.hardware_filter.filter(self.test_models, limited_hw_profile)
        self.assertEqual(3, len(limited_models))
        self.assertNotIn("gpt2", [model["id"] for model in limited_models])
        
        # Filter for very limited hardware (should return no models)
        very_limited_hw_profile = {
            "cuda_compute": 2.0,
            "ram_gb": 2,
            "gpu_vram_gb": 1
        }
        very_limited_models = self.hardware_filter.filter(self.test_models, very_limited_hw_profile)
        self.assertEqual(0, len(very_limited_models))
        
        # No filter (should return all models)
        all_models = self.hardware_filter.filter(self.test_models, None)
        self.assertEqual(4, len(all_models))

    def test_size_filter(self):
        """Test filtering models by size."""
        # Filter for models under 400MB
        small_models = self.size_filter.filter(self.test_models, 400)
        self.assertEqual(2, len(small_models))
        self.assertIn("t5-small", [model["id"] for model in small_models])
        self.assertIn("vit-base-patch16-224", [model["id"] for model in small_models])
        
        # Filter for models under 500MB (should include BERT but exclude GPT-2)
        medium_models = self.size_filter.filter(self.test_models, 500)
        self.assertEqual(3, len(medium_models))
        self.assertIn("bert-base-uncased", [model["id"] for model in medium_models])
        self.assertNotIn("gpt2", [model["id"] for model in medium_models])
        
        # Filter for models under 600MB (should include all models)
        large_models = self.size_filter.filter(self.test_models, 600)
        self.assertEqual(4, len(large_models))
        
        # No filter (should return all models)
        all_models = self.size_filter.filter(self.test_models, None)
        self.assertEqual(4, len(all_models))

    def test_framework_filter(self):
        """Test filtering models by framework."""
        # Filter for PyTorch models
        pt_models = self.framework_filter.filter(self.test_models, "pt")
        self.assertEqual(4, len(pt_models))
        
        # Filter for TensorFlow models
        tf_models = self.framework_filter.filter(self.test_models, "tf")
        self.assertEqual(4, len(tf_models))
        
        # Filter for ONNX models
        onnx_models = self.framework_filter.filter(self.test_models, "onnx")
        self.assertEqual(2, len(onnx_models))
        self.assertIn("bert-base-uncased", [model["id"] for model in onnx_models])
        self.assertIn("t5-small", [model["id"] for model in onnx_models])
        
        # Filter for non-existent framework
        non_existent_framework_models = self.framework_filter.filter(self.test_models, "non-existent-framework")
        self.assertEqual(0, len(non_existent_framework_models))
        
        # No filter (should return all models)
        all_models = self.framework_filter.filter(self.test_models, None)
        self.assertEqual(4, len(all_models))


class ModelSelectorTest(unittest.TestCase):
    """Tests for the model selector component."""

    def setUp(self):
        """Set up test environment."""
        # Create a model registry with test models
        self.registry = ModelRegistry()
        
        # Register some test models
        self.registry.register_model("bert-base-uncased", {
            "name": "BERT Base Uncased",
            "id": "bert-base-uncased",
            "architecture": "encoder-only",
            "task": "fill-mask",
            "framework": "pt",
            "size_mb": 420,
            "recommended_tasks": ["fill-mask", "sequence-classification", "token-classification"],
            "frameworks": ["pt", "tf", "onnx"],
            "default": True
        })
        
        self.registry.register_model("bert-large-uncased", {
            "name": "BERT Large Uncased",
            "id": "bert-large-uncased",
            "architecture": "encoder-only",
            "task": "fill-mask",
            "framework": "pt",
            "size_mb": 1200,
            "recommended_tasks": ["fill-mask", "sequence-classification", "token-classification"],
            "frameworks": ["pt", "tf", "onnx"],
            "default": False
        })
        
        self.registry.register_model("gpt2", {
            "name": "GPT-2",
            "id": "gpt2",
            "architecture": "decoder-only",
            "task": "text-generation",
            "framework": "pt",
            "size_mb": 510,
            "recommended_tasks": ["text-generation", "text-completion"],
            "frameworks": ["pt", "tf"],
            "default": True
        })
        
        self.registry.register_model("gpt2-medium", {
            "name": "GPT-2 Medium",
            "id": "gpt2-medium",
            "architecture": "decoder-only",
            "task": "text-generation",
            "framework": "pt",
            "size_mb": 1500,
            "recommended_tasks": ["text-generation", "text-completion"],
            "frameworks": ["pt", "tf"],
            "default": False
        })
        
        self.registry.register_model("t5-small", {
            "name": "T5 Small",
            "id": "t5-small",
            "architecture": "encoder-decoder",
            "task": "text2text-generation",
            "framework": "pt",
            "size_mb": 300,
            "recommended_tasks": ["text2text-generation", "translation", "summarization"],
            "frameworks": ["pt", "tf", "onnx"],
            "default": True
        })
        
        # Initialize the model selector
        self.selector = ModelSelector(self.registry)

    def test_select_default_model(self):
        """Test selecting the default model for a model type."""
        # Select default BERT model
        bert_model = self.selector.select_model("bert")
        self.assertEqual("bert-base-uncased", bert_model["id"])
        
        # Select default GPT-2 model
        gpt2_model = self.selector.select_model("gpt2")
        self.assertEqual("gpt2", gpt2_model["id"])
        
        # Select default T5 model
        t5_model = self.selector.select_model("t5")
        self.assertEqual("t5-small", t5_model["id"])
        
        # Select model with unknown model type (should return None or fallback)
        unknown_model = self.selector.select_model("unknown-model-type")
        self.assertIsNotNone(unknown_model)  # Should return a fallback

    def test_select_model_with_criteria(self):
        """Test selecting models with specific criteria."""
        # Select BERT model for token-classification task
        token_bert_model = self.selector.select_model("bert", task="token-classification")
        self.assertEqual("bert-base-uncased", token_bert_model["id"])
        
        # Select GPT-2 model for text-completion task with size constraint
        small_gpt2_model = self.selector.select_model("gpt2", 
                                                     task="text-completion", 
                                                     max_size=600)
        self.assertEqual("gpt2", small_gpt2_model["id"])
        
        # Select T5 model for ONNX framework
        onnx_t5_model = self.selector.select_model("t5", framework="onnx")
        self.assertEqual("t5-small", onnx_t5_model["id"])
        
        # Select BERT model with complex criteria
        complex_bert_model = self.selector.select_model("bert", 
                                                      task="sequence-classification",
                                                      framework="pt",
                                                      max_size=500)
        self.assertEqual("bert-base-uncased", complex_bert_model["id"])
        
        # Try to select a model with criteria that cannot be satisfied
        # Should fallback to default model
        impossible_bert_model = self.selector.select_model("bert", 
                                                        task="non-existent-task",
                                                        framework="non-existent-framework",
                                                        max_size=100)
        self.assertEqual("bert-base-uncased", impossible_bert_model["id"])  # Default model

    def test_model_type_to_architecture_mapping(self):
        """Test mapping model types to architectures."""
        # Test direct mapping
        self.assertEqual("encoder-only", self.selector._map_model_to_architecture("bert"))
        self.assertEqual("decoder-only", self.selector._map_model_to_architecture("gpt2"))
        self.assertEqual("encoder-decoder", self.selector._map_model_to_architecture("t5"))
        
        # Test mapping variants
        self.assertEqual("encoder-only", self.selector._map_model_to_architecture("bert-base"))
        self.assertEqual("encoder-only", self.selector._map_model_to_architecture("bert-large"))
        self.assertEqual("decoder-only", self.selector._map_model_to_architecture("gpt2-medium"))
        
        # Test mapping for unknown model types (should return fallback)
        fallback = self.selector._map_model_to_architecture("unknown-model-type")
        self.assertIsNotNone(fallback)  # Should provide some fallback architecture


if __name__ == "__main__":
    unittest.main()