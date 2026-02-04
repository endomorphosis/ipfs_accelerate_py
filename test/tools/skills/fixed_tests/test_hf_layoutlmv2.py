#!/usr/bin/env python3

# Import hardware detection capabilities if available
try:
    from scripts.generators.hardware.hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback

import os
import sys
import json
import time
import datetime

# ANSI color codes for terminal output
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"
import traceback
import logging
import argparse
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np

# Check if we should mock specific dependencies
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'
MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'
# Try to import torch
try:
    if MOCK_TORCH:
        raise ImportError("Mocked torch import failure")
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

    # Try to import transformers
try:
    if MOCK_TRANSFORMERS:
        raise ImportError("Mocked transformers import failure")
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Try to import tokenizers
try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

# Try to import PIL
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = MagicMock()
    HAS_PIL = False
    logger.warning("PIL not available, using mock")

# Hardware detection
def check_hardware():
    """Check available hardware and return capabilities."""
    capabilities = {
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False
    }
    
    # Check CUDA
    if HAS_TORCH:
        capabilities["cuda"] = torch.cuda.is_available()
        if capabilities["cuda"]:
            capabilities["cuda_devices"] = torch.cuda.device_count()
            capabilities["cuda_version"] = torch.version.cuda
    
    # Check MPS (Apple Silicon)
    if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        capabilities["mps"] = torch.mps.is_available()
    
    # Check OpenVINO
    try:
        import openvino
        capabilities["openvino"] = True
    except ImportError:
        pass
    
    return capabilities

# Get hardware capabilities
HW_CAPABILITIES = check_hardware()

# Models registry - Maps model IDs to their specific configurations
VISION_TEXT_MODELS_REGISTRY = {
    "layoutlmv2": {
        "description": "LAYOUTLMV2 model",
        "class": "LayoutLMv2ForSequenceClassification",
        "default_model": "microsoft/layoutlmv2-base-uncased",
        "architecture": "vision-encoder-text-decoder",
        "task": "document-question-answering"
    },
    # BLIP models
    "Salesforce/blip-image-captioning-base": {
        "description": "BLIP image captioning base model",
        "class": "BlipForConditionalGeneration",
        "type": "blip",
        "image_size": 384,
        "task": "image-to-text"
    },
    "Salesforce/blip-image-captioning-large": {
        "description": "BLIP image captioning large model",
        "class": "BlipForConditionalGeneration",
        "type": "blip",
        "image_size": 384,
        "task": "image-to-text"
    },
    "Salesforce/blip-vqa-base": {
        "description": "BLIP visual question answering base model",
        "class": "BlipForQuestionAnswering",
        "type": "blip",
        "image_size": 384,
        "task": "visual-question-answering"
    },
    # CLIP models
    "openai/clip-vit-base-patch32": {
        "description": "CLIP vision-language base model",
        "class": "CLIPModel",
        "type": "clip",
        "image_size": 224,
        "task": "zero-shot-image-classification"
    },
    "openai/clip-vit-large-patch14": {
        "description": "CLIP vision-language large model",
        "class": "CLIPModel",
        "type": "clip",
        "image_size": 224,
        "task": "zero-shot-image-classification"
    },
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {
        "description": "LAION CLIP vision-language large model",
        "class": "CLIPModel",
        "type": "clip",
        "image_size": 224,
        "task": "zero-shot-image-classification"
    }
}

class TestVisionTextModels:
    """Base test class for all vision-text models (BLIP, CLIP, etc.)."""
    
    def __init__(self, model_id=None):
        """Initialize the test class for a specific model or default."""
        self.model_id = model_id or "Salesforce/blip-image-captioning-base"
        
        # Verify model exists in registry
        if self.model_id not in VISION_TEXT_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = VISION_TEXT_MODELS_REGISTRY["Salesforce/blip-image-captioning-base"]
        else:
            self.model_info = VISION_TEXT_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.layoutlmv2 = self.model_info.get("type", "blip")  # Default to blip if not specified
        self.task = self.model_info.get("task", "image-to-text")
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        self.image_size = self.model_info["image_size"]
        
        # Define test inputs
        self.test_image_path = self._find_test_image()
        if "vqa" in self.model_id:
            self.test_text = "What is shown in the image?"
            self.test_texts = ["What is shown in the image?", "What can you see in this picture?"]
        
        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        logger.info(f"Using {self.preferred_device} as preferred device")
        
        # Results storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}
        
    def _find_test_image(self):
        """Find a test image or create a dummy one if none exists."""
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
            dummy_path = "test_dummy.jpg"
            img = Image.new('RGB', (self.image_size, self.image_size), color = (73, 109, 137))
            img.save(dummy_path)
            return dummy_path
        
        return None
        
    def test_pipeline(self, device="auto"):
        """Test the model using transformers pipeline API."""
        if device == "auto":
            device = self.preferred_device
        
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
            
        if not HAS_PIL:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_deps"] = ["Pillow"]
            results["pipeline_success"] = False
            return results
        
        try:
            # Create a dummy image for testing if needed
            if not os.path.exists(test_image_path):
                dummy_image = Image.new('RGB', (224, 224), color='white')
                dummy_image.save(test_image_path)
            
            logger.info(f"Testing {self.model_id} with pipeline() on {device}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": device
            }
            
            # Time the model loading
            load_start_time = time.time()
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Prepare test inputs
            if self.test_image_path and os.path.exists(self.test_image_path):
                test_image = self.test_image_path
            else:
                # Create a random dummy image
                dummy_path = "test_dummy_temp.jpg"
                img = Image.new('RGB', (self.image_size, self.image_size), color = (73, 109, 137))
                img.save(dummy_path)
                test_image = dummy_path
            
            # Define input based on model type and task
            if self.layoutlmv2 == "clip":
                # CLIP models expect candidate labels for classification
                pipeline_input = test_image
                pipeline_kwargs = {
                    "candidate_labels": ["a photo of a cat", "a photo of a dog", "a photo of a person"]
                }
            elif self.task == "visual-question-answering":
                pipeline_input = {"image": test_image, "question": self.test_text}
                pipeline_kwargs = {}
            else:
                pipeline_input = test_image
                pipeline_kwargs = {}
            
            # Run warmup inference if on CUDA
            if device == "cuda":
                try:
                    _ = pipeline(pipeline_input)
                except Exception:
                    pass
            
            # Run multiple inference passes
            num_runs = 3
            times = []
            outputs = []
            
            for _ in range(num_runs):
                start_time = time.time()
                output = pipeline(pipeline_input)
                end_time = time.time()
                times.append(end_time - start_time)
                outputs.append(output)
            
            # Clean up temporary dummy image if created
            if not self.test_image_path and os.path.exists(dummy_path):
                os.remove(dummy_path)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_avg_time"] = avg_time
            results["pipeline_min_time"] = min_time
            results["pipeline_max_time"] = max_time
            results["pipeline_load_time"] = load_time
            results["pipeline_error_type"] = "none"
            
            # Add to examples
            self.examples.append({
                "method": f"pipeline() on {device}",
                "input": pipeline_input if isinstance(pipeline_input, str) else str(pipeline_input),
                "output_preview": str(outputs[0])[:200] + "..." if len(str(outputs[0])) > 200 else str(outputs[0])
            })
            
            # Store in performance stats
            self.performance_stats[f"pipeline_{device}"] = {
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "load_time": load_time,
                "num_runs": num_runs
            }
            
        except Exception as e:
            # Clean up temporary dummy image if created
            if not self.test_image_path and "dummy_path" in locals() and os.path.exists(dummy_path):
                os.remove(dummy_path)
                
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_traceback"] = traceback.format_exc()
            logger.error(f"Error testing pipeline on {device}: {e}")
            
            # Classify error type
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
            
            if "cuda" in error_str or "cuda" in traceback_str:
                results["pipeline_error_type"] = "cuda_error"
            elif "memory" in error_str:
                results["pipeline_error_type"] = "out_of_memory"
            elif "no module named" in error_str:
                results["pipeline_error_type"] = "missing_dependency"
            else:
                results["pipeline_error_type"] = "other"
        
        # Add to overall results
        self.results[f"pipeline_{device}"] = results
        return results
        
    def test_from_pretrained(self, device="auto"):
        """Test the model using direct from_pretrained loading."""
        if device == "auto":
            device = self.preferred_device
        
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name,
            "layoutlmv2": self.layoutlmv2
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_core"] = ["transformers"]
            results["from_pretrained_success"] = False
            return results
            
        if not HAS_PIL:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_deps"] = ["Pillow"]
            results["from_pretrained_success"] = False
            return results
        
        try:
            logger.info(f"Testing {self.model_id} with from_pretrained() on {device}...")
            
            # Common parameters for loading
            pretrained_kwargs = {
                "local_files_only": False
            }
            
            # Load processor and model based on model type
            processor_load_start = time.time()
            
            if self.layoutlmv2 == "clip":
                # Load CLIP processor
                processor = transformers.CLIPProcessor.from_pretrained(
                    self.model_id,
                    **pretrained_kwargs
                )
            else:
                # Default to BLIP processor
                processor = transformers.BlipProcessor.from_pretrained(
                    self.model_id,
                    **pretrained_kwargs
                )
            
            processor_load_time = time.time() - processor_load_start
            
            # Time model loading - select appropriate model class based on model type
            model_load_start = time.time()
            
            if self.layoutlmv2 == "clip":
                # Load CLIP model
                model = transformers.CLIPModel.from_pretrained(
                    self.model_id, 
                    **pretrained_kwargs
                )
            elif "vqa" in self.model_id:
                # Load BLIP question answering model
                model = transformers.BlipForQuestionAnswering.from_pretrained(
                    self.model_id, 
                    **pretrained_kwargs
                )
            else:
                # Default to BLIP conditional generation model
                model = transformers.BlipForConditionalGeneration.from_pretrained(
                    self.model_id, 
                    **pretrained_kwargs
                )
                
            model_load_time = time.time() - model_load_start
            
            # Move model to device
            if device != "cpu":
                model = model.to(device)
            
            # Prepare test inputs
            if self.test_image_path and os.path.exists(self.test_image_path):
                image = Image.open(self.test_image_path)
            else:
                # Create a random dummy image
                image = Image.new('RGB', (self.image_size, self.image_size), color = (73, 109, 137))
            
            # Process inputs based on model type and task
            if self.layoutlmv2 == "clip":
                # For CLIP models, we need both images and text
                text_inputs = processor.tokenizer(
                    ["a photo of a cat", "a photo of a dog", "a photo of a person"],
                    padding=True,
                    return_tensors="pt"
                )
                image_inputs = processor.image_processor(image, return_tensors="pt")
                
                # Combine inputs
                inputs = {
                    **text_inputs,
                    **image_inputs
                }
            elif self.task == "visual-question-answering":
                # For BLIP VQA
                inputs = processor(image, self.test_text, return_tensors="pt")
            else:
                # Default BLIP image captioning
                inputs = processor(image, return_tensors="pt")
            
            # Move inputs to device
            if device != "cpu":
                inputs = {key: val.to(device) for key, val in inputs.items()}
            
            # Run warmup inference if using CUDA
            if device == "cuda":
                try:
                    with torch.no_grad():
                        if self.layoutlmv2 == "clip":
                            _ = model(**inputs)
                        else:
                            _ = model.generate(**inputs)
                except Exception:
                    pass
            
            # Run multiple inference passes
            num_runs = 3
            times = []
            outputs = []
            
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    if self.layoutlmv2 == "clip":
                        # For CLIP, just forward pass
                        output = model(**inputs)
                    else:
                        # For BLIP, generate text
                        output = model.generate(**inputs)
                end_time = time.time()
                times.append(end_time - start_time)
                outputs.append(output)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Process output based on model type
            if self.layoutlmv2 == "clip":
                # Process CLIP output (similarity scores)
                logits_per_image = outputs[0].logits_per_image.cpu()
                probs = logits_per_image.softmax(dim=1).detach().numpy()
                
                # Get top prediction
                predictions = {
                    "labels": ["a photo of a cat", "a photo of a dog", "a photo of a person"],
                    "scores": probs[0].tolist()
                }
            else:
                # Process BLIP output (decode generated text)
                if hasattr(processor, "decode"):
                    generated_text = processor.decode(outputs[0][0], skip_special_tokens=True)
                    predictions = {"generated_text": generated_text}
                else:
                    predictions = {"generated_ids": outputs[0][0].cpu().numpy().tolist()}
            
            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = (param_count * 4) / (1024 * 1024)  # Rough size in MB
            
            # Store results
            results["from_pretrained_success"] = True
            results["from_pretrained_avg_time"] = avg_time
            results["from_pretrained_min_time"] = min_time
            results["from_pretrained_max_time"] = max_time
            results["processor_load_time"] = processor_load_time
            results["model_load_time"] = model_load_time
            results["model_size_mb"] = model_size_mb
            results["from_pretrained_error_type"] = "none"
            
            # Add predictions if available
            if 'predictions' in locals():
                results["predictions"] = predictions
            
            # Add to examples
            example_data = {
                "method": f"from_pretrained() on {device}",
                "input": {
                    "image": self.test_image_path if self.test_image_path else "Generated image",
                    "text": self.test_text if hasattr(self, "test_text") else None
                }
            }
            
            if 'predictions' in locals():
                example_data["predictions"] = predictions
            
            self.examples.append(example_data)
            
            # Store in performance stats
            self.performance_stats[f"from_pretrained_{device}"] = {
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "processor_load_time": processor_load_time,
                "model_load_time": model_load_time,
                "model_size_mb": model_size_mb,
                "num_runs": num_runs
            }
            
        except Exception as e:
            # Store error information
            results["from_pretrained_success"] = False
            results["from_pretrained_error"] = str(e)
            results["from_pretrained_traceback"] = traceback.format_exc()
            logger.error(f"Error testing from_pretrained on {device}: {e}")
            
            # Classify error type
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
            
            if "cuda" in error_str or "cuda" in traceback_str:
                results["from_pretrained_error_type"] = "cuda_error"
            elif "memory" in error_str:
                results["from_pretrained_error_type"] = "out_of_memory"
            elif "no module named" in error_str:
                results["from_pretrained_error_type"] = "missing_dependency"
            else:
                results["from_pretrained_error_type"] = "other"
        
            # Add to overall results
            self.results[f"from_pretrained_{device}"] = results
            return results
            
    def test_with_openvino(self):
        """Test the model using OpenVINO integration."""
        results = {
            "model": self.model_id,
            "task": self.task,
            "class": self.class_name,
            "layoutlmv2": self.layoutlmv2
        }
        
        # Check for OpenVINO support
        if not HW_CAPABILITIES["openvino"]:
            results["openvino_error_type"] = "missing_dependency"
            results["openvino_missing_core"] = ["openvino"]
            results["openvino_success"] = False
            return results
        
        # Check for transformers
        if not HAS_TRANSFORMERS:
            results["openvino_error_type"] = "missing_dependency"
            results["openvino_missing_core"] = ["transformers"]
            results["openvino_success"] = False
            return results
        
        try:
            # Import the appropriate OpenVINO model class based on model type
            if self.layoutlmv2 == "clip":
                from optimum.intel import OVModelForImageClassification
                logger.info(f"Testing {self.model_id} (CLIP) with OpenVINO...")
            else:
                from optimum.intel import OVModelForVision2Seq
                logger.info(f"Testing {self.model_id} (BLIP) with OpenVINO...")
            
            # Time processor loading - select appropriate processor based on model type
            processor_load_start = time.time()
            
            if self.layoutlmv2 == "clip":
                processor = transformers.CLIPProcessor.from_pretrained(self.model_id)
            else:
                processor = transformers.BlipProcessor.from_pretrained(self.model_id)
                
            processor_load_time = time.time() - processor_load_start
            
            # Time model loading - use appropriate OpenVINO model class based on model type
            model_load_start = time.time()
            
            if self.layoutlmv2 == "clip":
                model = OVModelForImageClassification.from_pretrained(
                    self.model_id,
                    export=True,
                    provider="CPU"
                )
            else:
                model = OVModelForVision2Seq.from_pretrained(
                    self.model_id,
                    export=True,
                    provider="CPU"
                )
                
            model_load_time = time.time() - model_load_start
            
            # Prepare image input
            if self.test_image_path and os.path.exists(self.test_image_path):
                image = Image.open(self.test_image_path)
            else:
                # Create a random dummy image
                image = Image.new('RGB', (self.image_size, self.image_size), color = (73, 109, 137))
            
            # Process inputs based on model type and task
            if self.layoutlmv2 == "clip":
                # For CLIP models, we need both images and text
                text_inputs = processor.tokenizer(
                    ["a photo of a cat", "a photo of a dog", "a photo of a person"],
                    padding=True,
                    return_tensors="pt"
                )
                image_inputs = processor.image_processor(image, return_tensors="pt")
                
                # Combine inputs
                inputs = {
                    **text_inputs,
                    **image_inputs
                }
            elif self.task == "visual-question-answering":
                # For BLIP VQA
                inputs = processor(image, self.test_text, return_tensors="pt")
            else:
                # Default BLIP image captioning
                inputs = processor(image, return_tensors="pt")
            
            # Run inference based on model type
            start_time = time.time()
            
            if self.layoutlmv2 == "clip":
                outputs = model(**inputs)
                inference_time = time.time() - start_time
                
                # Process CLIP output
                logits_per_image = outputs.logits_per_image.cpu()
                probs = logits_per_image.softmax(dim=1).detach().numpy()
                
                # Format result
                result_text = str({"labels": ["a photo of a cat", "a photo of a dog", "a photo of a person"], 
                                   "scores": probs[0].tolist()})
            else:
                # For BLIP models, generate text
                outputs = model.generate(**inputs)
                inference_time = time.time() - start_time
                
                # Decode BLIP output
                if hasattr(processor, "decode"):
                    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
                    result_text = generated_text
                else:
                    result_text = str(outputs[0].cpu().numpy().tolist())
            
            # Store results
            results["openvino_success"] = True
            results["openvino_load_time"] = model_load_time
            results["openvino_inference_time"] = inference_time
            results["openvino_processor_load_time"] = processor_load_time
            
            # Add predictions
            results["openvino_prediction"] = result_text
            
            results["openvino_error_type"] = "none"
            
            # Add to examples
            example_data = {
                "method": "OpenVINO inference",
                "input": {
                    "image": self.test_image_path if self.test_image_path else "Generated image",
                    "text": self.test_text if hasattr(self, "test_text") else None
                },
                "prediction": result_text
            }
            
            self.examples.append(example_data)
            
            # Store in performance stats
            self.performance_stats["openvino"] = {
                "inference_time": inference_time,
                "load_time": model_load_time,
                "processor_load_time": processor_load_time
            }
            
        except Exception as e:
            # Store error information
            results["openvino_success"] = False
            results["openvino_error"] = str(e)
            results["openvino_traceback"] = traceback.format_exc()
            logger.error(f"Error testing with OpenVINO: {e}")
            
            # Classify error
            error_str = str(e).lower()
            if "no module named" in error_str:
                results["openvino_error_type"] = "missing_dependency"
            else:
                results["openvino_error_type"] = "other"
        
            # Add to overall results
            self.results["openvino"] = results
            return results
            
    def run_tests(self, all_hardware=False):
        """
        Run all tests for this model.
        
        Args:
            all_hardware: If True, tests on all available hardware (CPU, CUDA, OpenVINO)
        
        Returns:
            Dict containing test results
        """
        # Always test on default device
        self.test_pipeline()
        self.test_from_pretrained()
        
        # Test on all available hardware if requested
        if all_hardware:
            # Always test on CPU
            if self.preferred_device != "cpu":
                self.test_pipeline(device="cpu")
                self.test_from_pretrained(device="cpu")
            
            # Test on CUDA if available
            if HW_CAPABILITIES["cuda"] and self.preferred_device != "cuda":
                self.test_pipeline(device="cuda")
                self.test_from_pretrained(device="cuda")
            
            # Test on OpenVINO if available
            if HW_CAPABILITIES["openvino"]:
                self.test_with_openvino()
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
        # Build final results
        return {
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "hardware": HW_CAPABILITIES,
            "metadata": {
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "layoutlmv2": self.layoutlmv2,
                "description": self.description,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_tokenizers": HAS_TOKENIZERS,
                "has_sentencepiece": HAS_SENTENCEPIECE,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
            }
        }
        
def save_results(model_id, results, output_dir="collected_results"):
    """Save test results to a file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    
    # Determine model type (clip or blip) from results metadata
    layoutlmv2 = results.get("metadata", {}).get("layoutlmv2", "vision_text")
    filename = f"hf_{layoutlmv2}_{safe_model_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    return output_path
    
def get_available_models():
    """Get a list of all available vision-text models in the registry."""
    return list(VISION_TEXT_MODELS_REGISTRY.keys())
    
def test_all_models(output_dir="collected_results", all_hardware=False):
    """Test all registered vision-text models."""
    models = get_available_models()
    results = {}
    
    for model_id in models:
        logger.info(f"Testing model: {model_id}")
        tester = TestVisionTextModels(model_id)
        model_results = tester.run_tests(all_hardware=all_hardware)
        
        # Save individual results
        save_results(model_id, model_results, output_dir=output_dir)
        
        # Add to summary
        results[model_id] = {
            "success": any(r.get("pipeline_success", False) for r in model_results["results"].values() 
                       if r.get("pipeline_success") is not False)
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, f"hf_vision_text_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")
    return results
    
def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test Vision-Text models (BLIP, CLIP, etc.)")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Specific model to test")
    model_group.add_argument("--all-models", action="store_true", help="Test all registered models")
    model_group.add_argument("--blip-only", action="store_true", help="Test only BLIP models")
    model_group.add_argument("--clip-only", action="store_true", help="Test only CLIP models")
    
    # Hardware options
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for output files")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    # List options
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = get_available_models()
        print("\nAvailable Vision-Text models:")
        
        # Group by model type
        blip_models = []
        clip_models = []
        other_models = []
        
        for model in models:
            layoutlmv2 = VISION_TEXT_MODELS_REGISTRY[model].get("type", "unknown")
            if layoutlmv2 == "blip":
                blip_models.append(model)
            elif layoutlmv2 == "clip":
                clip_models.append(model)
            else:
                other_models.append(model)
        
        if blip_models:
            print("\nBLIP Models:")
            for model in blip_models:
                info = VISION_TEXT_MODELS_REGISTRY[model]
                print(f"  - {model} ({info['class']}): {info['description']}")
        
        if clip_models:
            print("\nCLIP Models:")
            for model in clip_models:
                info = VISION_TEXT_MODELS_REGISTRY[model]
                print(f"  - {model} ({info['class']}): {info['description']}")
        
        if other_models:
            print("\nOther Vision-Text Models:")
            for model in other_models:
                info = VISION_TEXT_MODELS_REGISTRY[model]
                print(f"  - {model} ({info['class']}): {info['description']}")
                
        return
    
    # Filter models by type if requested
    if args.blip_only or args.clip_only:
        all_models = get_available_models()
        filtered_models = []
        
        if args.blip_only:
            layoutlmv2 = "blip"

            default_model = "Salesforce/blip-image-captioning-base"
        elif args.clip_only:
            layoutlmv2 = "clip"
            default_model = "openai/clip-vit-base-patch32"
            
        for model in all_models:
            if VISION_TEXT_MODELS_REGISTRY[model].get("type") == layoutlmv2:
                filtered_models.append(model)
                
        if args.all_models:
            results = {}
            for model_id in filtered_models:
                logger.info(f"Testing model: {model_id}")
                tester = TestVisionTextModels(model_id)
                model_results = tester.run_tests(all_hardware=args.all_hardware)
                
                # Save individual results if requested
                if args.save:
                    save_results(model_id, model_results, output_dir=args.output_dir)
                
                # Add to summary
                results[model_id] = {
                    "success": any(r.get("pipeline_success", False) for r in model_results["results"].values()
                        if r.get("pipeline_success") is not False)
                }
                
            # Print summary
            print(f"\n{layoutlmv2.upper()} Models Testing Summary:")
            total = len(results)
            successful = sum(1 for r in results.values() if r["success"])
            print(f"Successfully tested {successful} of {total} models ({successful/total*100:.1f}%)")
            return
            
        # Test single model from filtered type
        model_id = args.model or default_model
            
    else:
        # Test single model (default or specified)
        model_id = args.model or "Salesforce/blip-image-captioning-base"
    
    logger.info(f"Testing model: {model_id}")
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Create tester and run tests
    tester = TestVisionTextModels(model_id)
    results = tester.run_tests(all_hardware=args.all_hardware)
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = any(r.get("pipeline_success", False) for r in results["results"].values()
        if r.get("pipeline_success") is not False)
    
    # Determine if real inference or mock objects were used
    using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
    using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
    
    print("\nTEST RESULTS SUMMARY:")
    
    # Indicate real vs mock inference clearly
    if using_real_inference and not using_mocks:
        print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")
    else:
        print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
    
    # Get model type from results
    layoutlmv2 = results.get("metadata", {}).get("layoutlmv2", "unknown")
    
    if success:
        print(f"✅ Successfully tested {model_id} ({layoutlmv2.upper()} model)")
        
        # Print performance highlights
        for device, stats in results["performance"].items():
            if "avg_time" in stats:
                print(f"  - {device}: {stats['avg_time']:.4f}s average inference time")
        
        # Print example outputs if available
        if results.get("examples") and len(results["examples"]) > 0:
            print("\nExample output:")
            example = results["examples"][0]
            if "predictions" in example:
                print(f"  Input: {example['input']}")
                print(f"  Predictions: {example['predictions']}")
            elif "output_preview" in example:
                print(f"  Input: {example['input']}")
                print(f"  Output: {example['output_preview']}")
    else:
        print(f"❌ Failed to test {model_id} ({layoutlmv2.upper()} model)")
        
        # Print error information
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {test_name}: {result.get('pipeline_error_type', 'unknown')}")
                print(f"    {result.get('pipeline_error', 'Unknown error')}")
    
    print("\nFor detailed results, use --save flag and check the JSON output file.")

if __name__ == "__main__":
    main()