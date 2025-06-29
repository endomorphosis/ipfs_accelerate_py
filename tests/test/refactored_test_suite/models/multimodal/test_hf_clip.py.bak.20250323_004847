#!/usr/bin/env python3
"""
Test file for CLIP (Contrastive Language-Image Pre-Training) models.

This file has been migrated to the refactored test suite.
Generated: 2025-03-21 from skills/fixed_tests/test_hf_clip.py
"""

import os
import sys
import json
import logging
import unittest
from unittest.mock import patch, MagicMock
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from refactored_test_suite.model_test import ModelTest

# Try to import required packages with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False

try:
    import transformers
    from transformers import CLIPProcessor, CLIPModel, CLIPConfig
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    CLIPProcessor = MagicMock()
    CLIPModel = MagicMock()
    CLIPConfig = MagicMock()
    HAS_TRANSFORMERS = False

try:
    from PIL import Image
    import requests
    from io import BytesIO
    HAS_PIL = True
except ImportError:
    Image = MagicMock()
    requests = MagicMock()
    BytesIO = MagicMock()
    HAS_PIL = False

try:
    import openvino
    HAS_OPENVINO = True
except ImportError:
    openvino = MagicMock()
    HAS_OPENVINO = False

# CLIP models registry
CLIP_MODELS_REGISTRY = {
    "openai/clip-vit-base-patch32": {
        "description": "CLIP ViT-B/32 model (151M parameters)",
        "image_size": 224
    },
    "openai/clip-vit-base-patch16": {
        "description": "CLIP ViT-B/16 model (151M parameters)",
        "image_size": 224
    },
    "openai/clip-vit-large-patch14": {
        "description": "CLIP ViT-L/14 model (428M parameters)",
        "image_size": 224
    }
}

class TestCLIPModels(ModelTest):
    """Test class for CLIP (Contrastive Language-Image Pre-Training) models."""
    
    def setUp(self):
        """Initialize the test with model details and hardware detection."""
        super().setUp()
        self.model_id = "openai/clip-vit-base-patch32"  # Default model
        
        # Model parameters
        if self.model_id not in CLIP_MODELS_REGISTRY:
            self.logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = CLIP_MODELS_REGISTRY["openai/clip-vit-base-patch32"]
        else:
            self.model_info = CLIP_MODELS_REGISTRY[self.model_id]
        
        self.image_size = self.model_info["image_size"]
        self.description = self.model_info["description"]
        
        # Test data
        self.test_image_path = os.path.join(self.model_dir, "test_image.jpg")
        self.test_text_candidates = [
            "a photo of a cat", 
            "a photo of a dog", 
            "a photo of a person",
            "a photo of a landscape"
        ]
        
        # Create a test image
        self._create_test_image()
        
        # Hardware detection
        self.setup_hardware()
        
        # Results storage
        self.results = {}
    
    def setup_hardware(self):
        """Set up hardware detection for the test."""
        # Skip test if torch is not available
        if not HAS_TORCH:
            self.skipTest("PyTorch not available")
            
        # CUDA support
        self.has_cuda = torch.cuda.is_available()
        
        # MPS support (Apple Silicon)
        try:
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except AttributeError:
            self.has_mps = False
            
        # ROCm support (AMD)
        self.has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
        
        # OpenVINO support
        self.has_openvino = HAS_OPENVINO
        
        # WebNN/WebGPU support
        self.has_webnn = False
        self.has_webgpu = False
        
        # Set default device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
            
        self.logger.info(f"Using device: {self.device}")
    
    def _create_test_image(self):
        """Create a test image for inference."""
        if not HAS_PIL:
            self.skipTest("PIL not available")
            
        # Create a simple test image (blue gradient)
        width = height = self.image_size
        image = Image.new('RGB', (width, height), color=(73, 109, 137))
        
        # Save to file
        image.save(self.test_image_path)
        self.logger.info(f"Created test image at {self.test_image_path}")
    
    def load_model(self, model_id=None):
        """Load CLIP model from HuggingFace."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
            
        model_id = model_id or self.model_id
            
        try:
            # Load processor
            self.logger.info(f"Loading CLIP processor: {model_id}")
            processor = CLIPProcessor.from_pretrained(model_id)
            
            # Load model
            self.logger.info(f"Loading CLIP model: {model_id} on {self.device}")
            model = CLIPModel.from_pretrained(model_id).to(self.device)
            
            # Store processor for later use
            self.processor = processor
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.skipTest(f"Could not load model: {str(e)}")
    
    def test_basic_inference(self):
        """Test basic inference with the CLIP model."""
        if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_PIL:
            self.skipTest("Required dependencies not available")
        
        self.logger.info(f"Testing {self.model_id} with basic inference on {self.device}...")
        
        try:
            # Load model and processor
            model = self.load_model()
            processor = self.processor
            
            # Load image
            image = Image.open(self.test_image_path)
            
            # Process inputs
            inputs = processor(
                text=self.test_text_candidates,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to the right device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                start_time = time.time()
                outputs = model(**inputs)
                inference_time = time.time() - start_time
            
            # Verify outputs
            self.assertIsNotNone(outputs, "Model outputs should not be None")
            self.assertIn("logits_per_image", outputs, "Model should output logits_per_image")
            self.assertIn("logits_per_text", outputs, "Model should output logits_per_text")
            
            # Calculate probabilities
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()
            
            # Get top prediction
            top_prob_idx = np.argmax(probs[0])
            top_prediction = self.test_text_candidates[top_prob_idx]
            top_score = probs[0][top_prob_idx]
            
            self.logger.info(f"Top prediction: '{top_prediction}' with score {top_score:.4f}")
            self.logger.info(f"All probabilities: {probs[0]}")
            self.logger.info(f"Inference time: {inference_time:.4f} seconds")
            
            # Store results
            self.results['basic_inference'] = {
                'top_prediction': top_prediction,
                'top_score': float(top_score),
                'probabilities': probs[0].tolist(),
                'inference_time': inference_time
            }
            
            self.logger.info("Basic inference test passed")
            
        except Exception as e:
            self.logger.error(f"Error in basic inference: {e}")
            self.fail(f"Basic inference test failed: {str(e)}")
    
    def test_image_text_similarity(self):
        """Test image-text similarity calculation."""
        if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_PIL:
            self.skipTest("Required dependencies not available")
        
        self.logger.info(f"Testing {self.model_id} image-text similarity...")
        
        try:
            # Load model and processor
            model = self.load_model()
            processor = self.processor
            
            # Load image
            image = Image.open(self.test_image_path)
            
            # Create multiple images (use the same one for simplicity)
            images = [image] * 3
            
            # Process inputs
            inputs = processor(
                text=self.test_text_candidates,
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to the right device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get similarity scores
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            # Convert to probabilities
            probs_per_image = logits_per_image.softmax(dim=1).cpu().detach().numpy()
            probs_per_text = logits_per_text.softmax(dim=1).cpu().detach().numpy()
            
            # Verify output shapes
            self.assertEqual(probs_per_image.shape, (len(images), len(self.test_text_candidates)), 
                             f"Expected shape {(len(images), len(self.test_text_candidates))}, got {probs_per_image.shape}")
            
            # Log results for each image
            for i in range(len(images)):
                self.logger.info(f"Image {i+1} similarities:")
                for j, text in enumerate(self.test_text_candidates):
                    self.logger.info(f"  '{text}': {probs_per_image[i][j]:.4f}")
            
            # Store results
            self.results['image_text_similarity'] = {
                'image_to_text_probs': probs_per_image.tolist(),
                'text_to_image_probs': probs_per_text.tolist()
            }
            
            self.logger.info("Image-text similarity test passed")
            
        except Exception as e:
            self.logger.error(f"Error in image-text similarity test: {e}")
            self.fail(f"Image-text similarity test failed: {str(e)}")
    
    def test_zero_shot_classification(self):
        """Test CLIP for zero-shot image classification."""
        if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_PIL:
            self.skipTest("Required dependencies not available")
        
        self.logger.info(f"Testing {self.model_id} for zero-shot classification...")
        
        try:
            # Use the pipeline API for zero-shot classification
            pipeline = transformers.pipeline(
                "zero-shot-image-classification",
                model=self.model_id,
                device=self.device
            )
            
            # Define candidate classes
            candidate_labels = [
                "a photo of a cat",
                "a photo of a dog",
                "a photo of a building",
                "a photo of a landscape",
                "a photo of the ocean"
            ]
            
            # Run inference
            start_time = time.time()
            result = pipeline(
                self.test_image_path,
                candidate_labels=candidate_labels
            )
            inference_time = time.time() - start_time
            
            # Verify output
            self.assertIsNotNone(result, "Pipeline output should not be None")
            
            # Log results
            self.logger.info("Zero-shot classification results:")
            for item in result:
                self.logger.info(f"  {item['label']}: {item['score']:.4f}")
            
            self.logger.info(f"Inference time: {inference_time:.4f} seconds")
            
            # Store results
            self.results['zero_shot_classification'] = {
                'classifications': result,
                'inference_time': inference_time
            }
            
            self.logger.info("Zero-shot classification test passed")
            
        except Exception as e:
            self.logger.error(f"Error in zero-shot classification: {e}")
            self.fail(f"Zero-shot classification test failed: {str(e)}")
    
    def test_hardware_compatibility(self):
        """Test model compatibility with different hardware platforms."""
        if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_PIL:
            self.skipTest("Required dependencies not available")
            
        devices_to_test = []
        
        # Always test CPU
        devices_to_test.append('cpu')
        
        # Only test available hardware
        if self.has_cuda:
            devices_to_test.append('cuda')
        if self.has_mps:
            devices_to_test.append('mps')
            
        # Test each device
        for device in devices_to_test:
            original_device = self.device
            try:
                self.logger.info(f"Testing on {device}...")
                self.device = device
                
                # Load model and processor
                model = self.load_model()
                processor = self.processor
                
                # Load image
                image = Image.open(self.test_image_path)
                
                # Process inputs
                inputs = processor(
                    text=self.test_text_candidates,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move inputs to the right device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Run inference with timing
                with torch.no_grad():
                    start_time = time.time()
                    outputs = model(**inputs)
                    inference_time = time.time() - start_time
                
                # Verify outputs are valid
                self.assertIsNotNone(outputs, f"Model outputs on {device} should not be None")
                self.assertIn("logits_per_image", outputs, f"Model on {device} should output logits_per_image")
                
                # Store performance results
                self.results[f'performance_{device}'] = {
                    'inference_time': inference_time
                }
                
                self.logger.info(f"Test on {device} passed (inference time: {inference_time:.4f}s)")
            except Exception as e:
                self.logger.error(f"Error testing on {device}: {e}")
                self.fail(f"Test on {device} failed: {str(e)}")
            finally:
                # Restore original device
                self.device = original_device

if __name__ == "__main__":
    unittest.main()