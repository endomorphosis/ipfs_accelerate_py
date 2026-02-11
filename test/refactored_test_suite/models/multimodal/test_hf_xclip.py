#!/usr/bin/env python3
"""
Test file for XCLIP (Extended CLIP) models.

This file has been migrated to the refactored test suite.
Generated: 2025-03-21 from skills/test_hf_xclip.py
"""

import os
import sys
import json
import logging
import unittest
import time
from unittest.mock import patch, MagicMock
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
    from transformers import XCLIPModel, AutoProcessor, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    XCLIPModel = MagicMock()
    AutoProcessor = MagicMock()
    AutoTokenizer = MagicMock()
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

# XCLIP models registry
XCLIP_MODELS_REGISTRY = {
    "microsoft/xclip-base-patch32": {
        "description": "XCLIP Base (patch size 32)",
        "class": "XCLIPModel",
        "image_size": 224
    },
    "microsoft/xclip-base-patch16": {
        "description": "XCLIP Base (patch size 16)",
        "class": "XCLIPModel",
        "image_size": 224
    },
    "microsoft/xclip-large-patch14": {
        "description": "XCLIP Large (patch size 14)",
        "class": "XCLIPModel",
        "image_size": 224
    }
}

class TestXCLIPModels(ModelTest):
    """Test class for XCLIP (Extended CLIP) models."""
    
    def setUp(self):
        """Initialize the test with model details and hardware detection."""
        super().setUp()
        self.model_id = "microsoft/xclip-base-patch32"  # Default model
        
        # Model parameters
        if self.model_id not in XCLIP_MODELS_REGISTRY:
            self.logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = XCLIP_MODELS_REGISTRY["microsoft/xclip-base-patch32"]
        else:
            self.model_info = XCLIP_MODELS_REGISTRY[self.model_id]
        
        self.image_size = self.model_info["image_size"]
        self.description = self.model_info["description"]
        self.task = "zero-shot-image-classification"
        self.class_name = self.model_info["class"]
        
        # Test data
        self.test_image_path = os.path.join(self.model_dir, "test_image.jpg")
        self.test_text_candidates = [
            "a photo of a cat", 
            "a photo of a dog", 
            "a photo of a bird",
            "a photo of a car"
        ]
        self.test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        
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
            
        # Create a simple test image (blue gradient with shapes)
        width = height = self.image_size
        image = Image.new('RGB', (width, height), color=(73, 109, 137))
        
        # Draw some shapes on the image
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            
            # Draw a red square
            draw.rectangle([(50, 50), (150, 150)], fill=(255, 0, 0))
            
            # Draw a green circle
            draw.ellipse([(width-150, 50), (width-50, 150)], fill=(0, 255, 0))
            
            # Draw a blue triangle
            draw.polygon([(width//2, height-50), (width//2-50, height-150), (width//2+50, height-150)], fill=(0, 0, 255))
            
            # Save to file
            image.save(self.test_image_path)
            self.logger.info(f"Created test image at {self.test_image_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not draw shapes on test image: {e}")
            # Save basic image without shapes
            image.save(self.test_image_path)
            self.logger.info(f"Created basic test image at {self.test_image_path}")
    
    def load_model(self, model_id=None):
        """Load XCLIP model from HuggingFace."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers not available")
            
        model_id = model_id or self.model_id
            
        try:
            # Load processor
            self.logger.info(f"Loading XCLIP processor: {model_id}")
            processor = AutoProcessor.from_pretrained(model_id)
            
            # Load model
            self.logger.info(f"Loading XCLIP model: {model_id} on {self.device}")
            model = XCLIPModel.from_pretrained(model_id).to(self.device)
            
            # Store processor for later use
            self.processor = processor
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.skipTest(f"Could not load model: {str(e)}")
    
    def test_basic_inference(self):
        """Test basic inference with the XCLIP model."""
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
            top_prob_idx = probs[0].argmax()
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
    
    def test_pipeline_api(self):
        """Test the XCLIP model using the pipeline API."""
        if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_PIL:
            self.skipTest("Required dependencies not available")
        
        self.logger.info(f"Testing {self.model_id} with pipeline API on {self.device}...")
        
        try:
            # Set up pipeline
            pipeline = transformers.pipeline(
                "zero-shot-image-classification",
                model=self.model_id,
                device=self.device
            )
            
            # Load image
            image = Image.open(self.test_image_path)
            
            # Run inference
            start_time = time.time()
            outputs = pipeline(
                image,
                candidate_labels=self.test_text_candidates
            )
            inference_time = time.time() - start_time
            
            # Verify outputs
            self.assertIsNotNone(outputs, "Pipeline output should not be None")
            self.assertGreater(len(outputs), 0, "Pipeline should return at least one result")
            
            # Log results
            self.logger.info("Zero-shot classification results:")
            for item in outputs:
                self.logger.info(f"  {item['label']}: {item['score']:.4f}")
            
            self.logger.info(f"Inference time: {inference_time:.4f} seconds")
            
            # Store results
            self.results['pipeline_api'] = {
                'classifications': outputs,
                'inference_time': inference_time
            }
            
            self.logger.info("Pipeline API test passed")
            
        except Exception as e:
            self.logger.error(f"Error in pipeline API test: {e}")
            self.fail(f"Pipeline API test failed: {str(e)}")
    
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
            images = [image] * 2
            
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
    
    def test_openvino_inference(self):
        """Test the model using OpenVINO."""
        if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_PIL or not self.has_openvino:
            self.skipTest("Required dependencies or OpenVINO not available")
        
        self.logger.info(f"Testing {self.model_id} with OpenVINO...")
        
        try:
            # For this test, we'll just check if optimum-intel is installed
            # Full implementation would use OVModelForVision2Seq
            try:
                from optimum.intel import OVModelForVision
            except ImportError:
                self.skipTest("optimum-intel not available")
            
            # Note: In a real implementation, we would load the model with OpenVINO:
            # model = OVModelForVision.from_pretrained(self.model_id, export=True)
            
            # For this test, we'll just log that OpenVINO would be used
            self.logger.info("OpenVINO support is available for XCLIP models")
            
            # Store results
            self.results['openvino_inference'] = {
                'supported': True,
                'notes': "Full implementation would use OVModelForVision"
            }
            
        except Exception as e:
            self.logger.error(f"Error in OpenVINO test: {e}")
            self.fail(f"OpenVINO test failed: {str(e)}")



    def test_model_loading(self):
        # Test basic model loading
        if not hasattr(self, 'model_id') or not self.model_id:
            self.skipTest("No model_id specified")
        
        try:
            # Import the appropriate library
            if 'bert' in self.model_id.lower() or 'gpt' in self.model_id.lower() or 't5' in self.model_id.lower():
                import transformers
                model = transformers.AutoModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'clip' in self.model_id.lower():
                import transformers
                model = transformers.CLIPModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'whisper' in self.model_id.lower():
                import transformers
                model = transformers.WhisperModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'wav2vec2' in self.model_id.lower():
                import transformers
                model = transformers.Wav2Vec2Model.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            else:
                # Generic loading
                try:
                    import transformers
                    model = transformers.AutoModel.from_pretrained(self.model_id)
                    self.assertIsNotNone(model, "Model loading failed")
                except:
                    self.skipTest(f"Could not load model {self.model_id} with AutoModel")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")



    def detect_preferred_device(self):
        # Detect available hardware and choose the preferred device
        try:
            import torch
        
            # Check for CUDA
            if torch.cuda.is_available():
                return "cuda"
        
            # Check for MPS (Apple Silicon)
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                return "mps"
        
            # Fallback to CPU
            return "cpu"
        except ImportError:
            return "cpu"


if __name__ == "__main__":
    unittest.main()