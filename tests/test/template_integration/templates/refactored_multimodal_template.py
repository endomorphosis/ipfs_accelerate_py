#!/usr/bin/env python3
"""
Refactored test template for multimodal (vision-text) models.

This template is used to generate test files for multimodal models like:
- CLIP (Contrastive Language-Image Pre-training)
- BLIP (Bootstrapping Language-Image Pre-training)
- FLAVA (A Foundational Language And Vision Alignment model)

Template customization variables:
- model_name: Full model ID/name (e.g. "openai/clip-vit-base-patch32")
- sanitized_model_name: Python-safe model name (e.g. "ClipVitBasePatch32")
- timestamp: Generation timestamp
- architecture: Model architecture (always "multimodal")
- base_class: Base test class name (ModelTest)
"""

import os
import sys
import logging
import unittest
import tempfile
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time
import datetime
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import test base classes
from refactored_test_suite.model_test import ModelTest

# Dynamically define mocks based on environment variables
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'
MOCK_PIL = os.environ.get('MOCK_PIL', 'False').lower() == 'true'

# Import required modules with mocking support
if MOCK_TORCH:
    from unittest.mock import MagicMock
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("Using mock torch module")
else:
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        from unittest.mock import MagicMock
        torch = MagicMock()
        HAS_TORCH = False
        logger.warning("torch not available, using mock")

if MOCK_TRANSFORMERS:
    from unittest.mock import MagicMock
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("Using mock transformers module")
else:
    try:
        import transformers
        HAS_TRANSFORMERS = True
    except ImportError:
        from unittest.mock import MagicMock
        transformers = MagicMock()
        HAS_TRANSFORMERS = False
        logger.warning("transformers not available, using mock")

if MOCK_PIL:
    from unittest.mock import MagicMock
    Image = MagicMock()
    HAS_PIL = False
    logger.warning("Using mock PIL.Image module")
else:
    try:
        from PIL import Image
        HAS_PIL = True
    except ImportError:
        from unittest.mock import MagicMock
        Image = MagicMock()
        HAS_PIL = False
        logger.warning("PIL.Image not available, using mock")

# Define multimodal model registry
MULTIMODAL_MODELS_REGISTRY = {
    # CLIP models
    "openai/clip-vit-base-patch32": {
        "description": "CLIP model with ViT base patch32 encoder",
        "class": "CLIPModel",
        "type": "clip",
        "image_size": 224,
        "task": "zero-shot-image-classification",
        "processor": "CLIPProcessor"
    },
    "openai/clip-vit-base-patch16": {
        "description": "CLIP model with ViT base patch16 encoder",
        "class": "CLIPModel",
        "type": "clip",
        "image_size": 224,
        "task": "zero-shot-image-classification",
        "processor": "CLIPProcessor"
    },
    "openai/clip-vit-large-patch14": {
        "description": "CLIP model with ViT large patch14 encoder",
        "class": "CLIPModel",
        "type": "clip",
        "image_size": 224,
        "task": "zero-shot-image-classification",
        "processor": "CLIPProcessor"
    },
    
    # BLIP models
    "Salesforce/blip-image-captioning-base": {
        "description": "BLIP image captioning base model",
        "class": "BlipForConditionalGeneration",
        "type": "blip",
        "image_size": 384,
        "task": "image-to-text",
        "processor": "BlipProcessor"
    },
    "Salesforce/blip-image-captioning-large": {
        "description": "BLIP image captioning large model",
        "class": "BlipForConditionalGeneration",
        "type": "blip",
        "image_size": 384,
        "task": "image-to-text",
        "processor": "BlipProcessor"
    },
    "Salesforce/blip-vqa-base": {
        "description": "BLIP visual question answering base model",
        "class": "BlipForQuestionAnswering",
        "type": "blip",
        "image_size": 384,
        "task": "visual-question-answering",
        "processor": "BlipProcessor"
    },
    
    # FLAVA models
    "facebook/flava-full": {
        "description": "FLAVA multimodal model",
        "class": "FlavaModel",
        "type": "flava",
        "image_size": 224,
        "task": "multimodal-classification",
        "processor": "FlavaProcessor"
    }
}

class TestMultimodalModel(ModelTest):
    """Test class for {model_name} model."""
    
    def setUp(self):
        """Set up resources for each test method."""
        super().setUp()
        self.model_id = "{model_name}"
        
        # Verify model exists in registry
        if self.model_id not in MULTIMODAL_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = MULTIMODAL_MODELS_REGISTRY["openai/clip-vit-base-patch32"]
        else:
            self.model_info = MULTIMODAL_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.model_type = self.model_info.get("type", "clip")  # Default to clip if not specified
        self.task = self.model_info.get("task", "zero-shot-image-classification")
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        self.image_size = self.model_info["image_size"]
        self.processor_class = self.model_info.get("processor", "CLIPProcessor")
        
        # Define test inputs
        self.test_image_path = self.create_test_image()
        if "vqa" in self.model_id.lower():
            self.test_text = "What is shown in the image?"
            self.test_texts = ["What is shown in the image?", "What can you see in this picture?"]
        else:
            self.test_text = ["a photo of a cat", "a photo of a dog", "a photo of a person"]
        
        # Configure hardware preference
        self.preferred_device = self.detect_preferred_device()
    
    def create_test_image(self):
        """Create a test image for multimodal testing."""
        test_image_candidates = [
            "test.jpg", 
            "test.png", 
            "test_image.jpg", 
            "test_image.png"
        ]
        
        for path in test_image_candidates:
            if os.path.exists(path):
                return path
        
        # Create a dummy image if no test image is found
        if HAS_PIL:
            dummy_path = os.path.join(self.model_dir, "test_dummy.jpg")
            img = Image.new('RGB', (self.image_size, self.image_size), color = (73, 109, 137))
            img.save(dummy_path)
            return dummy_path
        
        return None
    
    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device."""
        if not HAS_TORCH:
            return "cpu"
        
        # Check for CUDA
        if torch.cuda.is_available():
            return "cuda"
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            return "mps"
        
        # Fallback to CPU
        return "cpu"
    
    def test_model_loading(self):
        """Test basic model and processor loading."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers library not available")
            
        # Test processor loading
        try:
            processor_class = getattr(transformers, self.processor_class)
            processor = processor_class.from_pretrained(self.model_id)
            self.assertIsNotNone(processor, "Processor loading failed")
        except Exception as e:
            self.fail(f"Processor loading failed: {e}")
        
        # Test model loading
        try:
            model_class = getattr(transformers, self.class_name)
            model = model_class.from_pretrained(self.model_id)
            self.assertIsNotNone(model, "Model loading failed")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")
    
    def test_pipeline(self):
        """Test using the model with the transformers pipeline API."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers library not available")
        if not HAS_PIL:
            self.skipTest("PIL library not available")
            
        # Skip if we don't have a test image
        if self.test_image_path is None:
            self.skipTest("No test image available")
        
        # Create pipeline with appropriate parameters
        try:
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": self.preferred_device
            }
            
            pipeline = transformers.pipeline(**pipeline_kwargs)
            self.assertIsNotNone(pipeline, "Pipeline creation failed")
            
            # Prepare input based on task
            if self.task == "visual-question-answering":
                # For VQA models like BLIP-VQA
                pipeline_input = {"image": self.test_image_path, "question": self.test_text}
            elif self.task == "zero-shot-image-classification":
                # For CLIP models
                pipeline_input = self.test_image_path
                pipeline_kwargs = {"candidate_labels": self.test_text}
                output = pipeline(pipeline_input, **pipeline_kwargs)
            elif self.task == "image-to-text":
                # For image captioning models like BLIP
                pipeline_input = self.test_image_path
                output = pipeline(pipeline_input)
            else:
                # Generic fallback
                pipeline_input = self.test_image_path
                output = pipeline(pipeline_input)
            
            # Verify we got output
            self.assertIsNotNone(output, "Pipeline produced no output")
            
        except Exception as e:
            self.fail(f"Pipeline test failed: {e}")
    
    def test_from_pretrained(self):
        """Test the model using direct from_pretrained loading."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers library not available")
        if not HAS_PIL:
            self.skipTest("PIL library not available")
        if not HAS_TORCH:
            self.skipTest("PyTorch not available")
        
        # Skip if we don't have a test image
        if self.test_image_path is None:
            self.skipTest("No test image available")
        
        try:
            # Load processor
            processor_class = getattr(transformers, self.processor_class)
            processor = processor_class.from_pretrained(self.model_id)
            
            # Load model
            model_class = getattr(transformers, self.class_name)
            model = model_class.from_pretrained(self.model_id)
            
            # Move model to preferred device
            if self.preferred_device != "cpu":
                model = model.to(self.preferred_device)
            
            # Prepare image
            image = Image.open(self.test_image_path)
            
            # Process inputs based on model type
            if self.model_type == "clip":
                # For CLIP models
                inputs = processor(text=self.test_text, images=image, return_tensors="pt", padding=True)
            elif self.model_type == "blip" and self.task == "visual-question-answering":
                # For BLIP VQA
                inputs = processor(image, self.test_text[0], return_tensors="pt")
            elif self.model_type == "flava":
                # For FLAVA models
                inputs = processor(text=self.test_text[0], images=image, return_tensors="pt")
            else:
                # Default (image captioning models like BLIP)
                inputs = processor(image, return_tensors="pt")
            
            # Move inputs to device
            if self.preferred_device != "cpu":
                inputs = {key: val.to(self.preferred_device) for key, val in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                if self.model_type == "clip":
                    # For CLIP, just forward pass
                    outputs = model(**inputs)
                elif self.task == "image-to-text" or self.task == "visual-question-answering":
                    # For text generation models like BLIP
                    outputs = model.generate(**inputs)
                else:
                    # Default forward pass
                    outputs = model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs, "Model produced no outputs")
            
            # Process outputs based on model type for verification
            if self.model_type == "clip" and hasattr(outputs, "logits_per_image"):
                # Process CLIP-specific outputs
                logits_per_image = outputs.logits_per_image
                self.assertIsNotNone(logits_per_image, "No logits_per_image in outputs")
                probs = torch.softmax(logits_per_image, dim=1)
                self.assertEqual(probs.shape[1], len(self.test_text), 
                                "Output probabilities don't match number of test texts")
            
        except Exception as e:
            self.fail(f"Direct from_pretrained test failed: {e}")
    
    def test_with_openvino(self):
        """Test the model using OpenVINO integration."""
        if not HAS_TRANSFORMERS:
            self.skipTest("Transformers library not available")
        if not HAS_PIL:
            self.skipTest("PIL library not available")
        
        # Check for OpenVINO
        try:
            import openvino
        except ImportError:
            self.skipTest("OpenVINO not available")
        
        # Skip if we don't have a test image
        if self.test_image_path is None:
            self.skipTest("No test image available")
        
        try:
            # Import OpenVINO optimum utilities based on model type
            if self.model_type == "clip":
                from optimum.intel import OVModelForImageClassification
                ov_model_class = OVModelForImageClassification
            elif self.model_type == "blip" and "vqa" in self.model_id.lower():
                from optimum.intel import OVModelForVision2Seq
                ov_model_class = OVModelForVision2Seq
            elif self.model_type == "blip":
                from optimum.intel import OVModelForVision2Seq
                ov_model_class = OVModelForVision2Seq
            else:
                self.skipTest(f"OpenVINO integration not implemented for {self.model_type}")
            
            # Load processor
            processor_class = getattr(transformers, self.processor_class)
            processor = processor_class.from_pretrained(self.model_id)
            
            # Load model with OpenVINO
            model = ov_model_class.from_pretrained(
                self.model_id,
                export=True,
                provider="CPU"
            )
            
            # Prepare image
            image = Image.open(self.test_image_path)
            
            # Process inputs based on model type
            if self.model_type == "clip":
                # For CLIP models
                inputs = processor(text=self.test_text, images=image, return_tensors="pt", padding=True)
            elif self.model_type == "blip" and self.task == "visual-question-answering":
                # For BLIP VQA
                inputs = processor(image, self.test_text[0], return_tensors="pt")
            else:
                # Default (image captioning models like BLIP)
                inputs = processor(image, return_tensors="pt")
            
            # Run inference
            outputs = model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs, "OpenVINO model produced no outputs")
            
        except Exception as e:
            self.fail(f"OpenVINO integration test failed: {e}")