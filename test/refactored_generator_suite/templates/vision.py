#!/usr/bin/env python3
"""
Vision Template

This module provides the template for vision models like ViT, ResNet, etc.
"""

import logging
from typing import Dict, Any, List

from .base import TemplateBase

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionTemplate(TemplateBase):
    """
    Template for vision models like ViT, ResNet, etc.
    
    This template provides specialized support for models that process image inputs
    for tasks like image classification, object detection, and segmentation.
    """
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this template.
        
        Returns:
            Dictionary of metadata
        """
        metadata = super().get_metadata()
        metadata.update({
            "name": "VisionTemplate",
            "description": "Template for vision models like ViT, ResNet, etc.",
            "supported_architectures": ["vision"],
            "supported_models": [
                "vit", "swin", "deit", "beit", "convnext", "detr", "segformer", 
                "mask2former", "yolos", "sam", "dinov2", "mobilenet-v2", "resnet"
            ]
        })
        return metadata
    
    def get_imports(self) -> List[str]:
        """
        Get the imports required by this template.
        
        Returns:
            List of import statements
        """
        imports = super().get_imports()
        imports.extend([
            "import numpy as np",
            "try:",
            "    import torch",
            "    HAS_TORCH = True",
            "except ImportError:",
            "    torch = MagicMock()",
            "    HAS_TORCH = False",
            "    print(\"torch not available, using mock\")",
            "",
            "try:",
            "    import transformers",
            "    from transformers import AutoImageProcessor, AutoModelForImageClassification",
            "    from transformers import pipeline",
            "    HAS_TRANSFORMERS = True",
            "except ImportError:",
            "    transformers = MagicMock()",
            "    AutoImageProcessor = MagicMock()",
            "    AutoModelForImageClassification = MagicMock()",
            "    pipeline = MagicMock()",
            "    HAS_TRANSFORMERS = False",
            "    print(\"transformers not available, using mock\")",
            "",
            "try:",
            "    from PIL import Image",
            "    HAS_PIL = True",
            "except ImportError:",
            "    Image = MagicMock()",
            "    HAS_PIL = False",
            "    print(\"PIL not available, using mock\")"
        ])
        return imports
    
    def get_template_str(self) -> str:
        """
        Get the template string for vision models.
        
        Returns:
            The template as a string
        """
        return """#!/usr/bin/env python3
"""
"""
Test file for {{ model_info.name }}

This test file was automatically generated for the {{ model_info.name }} model,
which is a vision model from the {{ model_info.type }} family.

Generated on: {{ timestamp }}
"""

# Standard library imports
import os
import sys
import json
import time
import datetime
import logging
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Third-party imports
import numpy as np

# Try to import hardware detection if available
try:
    from generators.hardware.hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback

# Check if we should mock specific dependencies
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
MOCK_PIL = os.environ.get('MOCK_PIL', 'False').lower() == 'true'

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
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    AutoImageProcessor = MagicMock()
    AutoModelForImageClassification = MagicMock()
    pipeline = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Try to import PIL
try:
    if MOCK_PIL:
        raise ImportError("Mocked PIL import failure")
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = MagicMock()
    HAS_PIL = False
    logger.warning("PIL not available, using mock")

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        cuda_version = torch.version.cuda
        logger.info(f"CUDA available: version {cuda_version}")
        num_devices = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {num_devices}")
        
        # Log CUDA device properties
        for i in range(num_devices):
            device_props = torch.cuda.get_device_properties(i)
            logger.info(f"CUDA Device {i}: {device_props.name} with {device_props.total_memory / 1024**3:.2f} GB memory")
    else:
        logger.info("CUDA not available")
else:
    HAS_CUDA = False
    logger.info("CUDA detection skipped (torch not available)")

# MPS (Apple Silicon) detection
if HAS_TORCH:
    HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if HAS_MPS:
        logger.info("MPS available for Apple Silicon acceleration")
    else:
        logger.info("MPS not available")
else:
    HAS_MPS = False
    logger.info("MPS detection skipped (torch not available)")

# ROCm detection
HAS_ROCM = False
if HAS_TORCH:
    try:
        if torch.cuda.is_available() and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
            HAS_ROCM = True
            ROCM_VERSION = torch._C._rocm_version()
            logger.info(f"ROCm available: version {ROCM_VERSION}")
        elif 'ROCM_HOME' in os.environ:
            HAS_ROCM = True
            logger.info("ROCm available (detected via ROCM_HOME)")
    except:
        HAS_ROCM = False
        logger.info("ROCm not available")

# OpenVINO detection
try:
    import openvino
    from openvino.runtime import Core
    HAS_OPENVINO = True
    logger.info(f"OpenVINO available: version {openvino.__version__}")
except ImportError:
    HAS_OPENVINO = False
    logger.info("OpenVINO not available")

# WebGPU detection
HAS_WEBGPU = False
try:
    import ctypes.util
    HAS_WEBGPU = hasattr(ctypes.util, 'find_library') and ctypes.util.find_library('webgpu') is not None
    if HAS_WEBGPU:
        logger.info("WebGPU available")
    else:
        logger.info("WebGPU not available")
except ImportError:
    HAS_WEBGPU = False
    logger.info("WebGPU not available")

# WebNN detection
HAS_WEBNN = False
try:
    import ctypes.util
    HAS_WEBNN = hasattr(ctypes.util, 'find_library') and ctypes.util.find_library('webnn') is not None
    if HAS_WEBNN:
        logger.info("WebNN available")
    else:
        logger.info("WebNN not available")
except ImportError:
    HAS_WEBNN = False
    logger.info("WebNN not available")

def select_device():
    """Select the best available device for inference."""
    if HAS_CUDA:
        return "cuda:0"
    elif HAS_ROCM:
        return "cuda:0"  # ROCm uses CUDA interface
    elif HAS_MPS:
        return "mps"
    else:
        return "cpu"

def get_sample_image_path():
    """
    Get the path to a sample image for testing.
    
    Returns:
        Path to a sample image file
    """
    # Check for test image in the project
    test_paths = [
        # Current directory
        Path("test.jpg"),
        # Test directory
        Path(__file__).parent / "test.jpg",
        # Project root
        Path(__file__).parent.parent.parent / "test.jpg",
    ]
    
    for path in test_paths:
        if path.exists():
            return str(path)
    
    # If no test image is found, create one
    if HAS_PIL:
        # Create a simple image with some colored rectangles
        img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        img_path = "temp_test_image.jpg"
        img.save(img_path)
        logger.info(f"Created temporary test image at {img_path}")
        return img_path
    else:
        logger.warning("No sample image found and PIL not available to create one")
        return None

class {{ model_info.type|capitalize }}Test:
    """
    Test class for {{ model_info.name }} model.
    
    This class provides methods to test the model's functionality
    using both the pipeline API and direct model access.
    """
    
    def __init__(self, model_name=None, output_dir=None, device=None, image_path=None):
        """
        Initialize the test class.
        
        Args:
            model_name: The name or path of the model to test (default: {{ model_info.name }})
            output_dir: Directory to save outputs (default: None)
            device: Device to run the model on (default: auto-selected)
            image_path: Path to image for testing (default: auto-selected)
        """
        self.model_name = model_name or "{{ model_info.name }}"
        self.output_dir = output_dir
        self.device = device or select_device()
        self.image_path = image_path or get_sample_image_path()
        
        # Create output directory if specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Mock detection
        self.using_real_inference = HAS_TRANSFORMERS and HAS_TORCH and HAS_PIL
        self.using_mocks = not self.using_real_inference
        
        logger.info(f"Initialized test for {self.model_name} on {self.device}")
        logger.info(f"Test type: {'🚀 REAL INFERENCE' if self.using_real_inference else '🔷 MOCK OBJECTS (CI/CD)'}")
        logger.info(f"Image path: {self.image_path or 'None'}")
    
    def run(self):
        """
        Run all tests for this model.
        
        Returns:
            Dictionary with test results
        """
        results = {
            "metadata": {
                "model_name": self.model_name,
                "device": self.device,
                "image_path": self.image_path,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_pil": HAS_PIL,
                "has_cuda": HAS_CUDA,
                "has_rocm": HAS_ROCM,
                "has_mps": HAS_MPS,
                "has_openvino": HAS_OPENVINO,
                "has_webgpu": HAS_WEBGPU,
                "has_webnn": HAS_WEBNN,
                "using_real_inference": self.using_real_inference,
                "using_mocks": self.using_mocks,
                "test_type": "REAL INFERENCE" if (self.using_real_inference and not self.using_mocks) else "MOCK OBJECTS (CI/CD)"
            },
            "tests": {}
        }
        
        # Skip image tests if no image is available
        if not self.image_path:
            results["tests"]["error"] = {
                "success": False,
                "elapsed_time": 0,
                "error": "No test image available"
            }
            return results
        
        # Run tests
        results["tests"]["pipeline"] = self.test_pipeline()
        results["tests"]["model"] = self.test_model()
        results["tests"]["processor"] = self.test_processor()
        
        {% if has_openvino %}
        # Run OpenVINO test if available
        results["tests"]["openvino"] = self.test_openvino()
        {% endif %}
        
        return results
    
    def test_pipeline(self):
        """
        Test the model using the pipeline API.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with pipeline API")
        start_time = time.time()
        
        try:
            # Create image-classification pipeline
            image_classifier = pipeline(
                "image-classification",
                model=self.model_name,
                device=self.device if self.device != "mps" else -1  # MPS not supported by pipeline
            )
            
            # Load image
            if HAS_PIL:
                image = Image.open(self.image_path)
            else:
                # Mock image for CI/CD
                image = MagicMock()
            
            # Run inference
            results = image_classifier(image)
            
            # Process results
            if isinstance(results, list) and len(results) > 0:
                predictions = []
                for result in results[:5]:  # Show top 5 predictions
                    if isinstance(result, dict) and "label" in result and "score" in result:
                        predictions.append({
                            "label": result["label"],
                            "score": result["score"]
                        })
                        
                # Log top prediction
                if predictions:
                    top_label = predictions[0]["label"]
                    top_score = predictions[0]["score"]
                    logger.info(f"Pipeline result: {top_label} ({top_score:.4f})")
            else:
                predictions = [{"label": "unknown", "score": 0.0}]
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "predictions": predictions,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in pipeline test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "predictions": None,
                "error": str(e)
            }
    
    def test_model(self):
        """
        Test the model using direct model access.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with direct model access")
        start_time = time.time()
        
        try:
            # Load processor and model
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = AutoModelForImageClassification.from_pretrained(self.model_name)
            
            # Move model to device
            if self.device != "cpu":
                model = model.to(self.device)
            
            # Load and process image
            if HAS_PIL:
                image = Image.open(self.image_path)
                inputs = processor(image, return_tensors="pt")
            else:
                # Mock inputs for CI/CD
                inputs = {"pixel_values": torch.rand(1, 3, 224, 224)}
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get predictions
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
                top_5_probs, top_5_indices = torch.topk(probabilities, 5)
                
                # Get labels if available
                if hasattr(model.config, "id2label"):
                    id2label = model.config.id2label
                    predictions = [
                        {"label": id2label.get(idx.item(), f"LABEL_{idx.item()}"), 
                         "score": prob.item()}
                        for idx, prob in zip(top_5_indices, top_5_probs)
                    ]
                else:
                    predictions = [
                        {"label": f"LABEL_{idx.item()}", "score": prob.item()}
                        for idx, prob in zip(top_5_indices, top_5_probs)
                    ]
                
                # Log top prediction
                top_label = predictions[0]["label"]
                top_score = predictions[0]["score"]
                logger.info(f"Model result: {top_label} ({top_score:.4f})")
            else:
                predictions = [{"label": "unknown", "score": 0.0}]
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "predictions": predictions,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in model test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "predictions": None,
                "error": str(e)
            }
    
    def test_processor(self):
        """
        Test the model's image processor.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} image processor")
        start_time = time.time()
        
        try:
            # Load processor
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # Load and process image
            if HAS_PIL:
                image = Image.open(self.image_path)
                processed = processor(image, return_tensors="pt")
            else:
                # Mock processed for CI/CD
                processed = {"pixel_values": torch.rand(1, 3, 224, 224)}
            
            # Get processor info
            processor_info = {
                "image_size": processor.size if hasattr(processor, "size") else None,
                "num_channels": processed["pixel_values"].shape[1] if "pixel_values" in processed else None,
                "padding": processor.do_pad if hasattr(processor, "do_pad") else None,
                "normalize": processor.do_normalize if hasattr(processor, "do_normalize") else None
            }
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "processor_info": processor_info,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in processor test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "processor_info": None,
                "error": str(e)
            }
    
    {% if has_openvino %}
    def test_openvino(self):
        """
        Test the model using OpenVINO.
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {self.model_name} with OpenVINO")
        start_time = time.time()
        
        try:
            from optimum.intel import OVModelForImageClassification
            
            # Load processor and model
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = OVModelForImageClassification.from_pretrained(self.model_name)
            
            # Load and process image
            if HAS_PIL:
                image = Image.open(self.image_path)
                inputs = processor(image, return_tensors="pt")
            else:
                # Mock inputs for CI/CD
                inputs = {"pixel_values": torch.rand(1, 3, 224, 224)}
            
            # Run inference
            outputs = model(**inputs)
            
            # Get predictions
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            top_5_probs, top_5_indices = torch.topk(probabilities, 5)
            
            # Get labels if available
            if hasattr(model.config, "id2label"):
                id2label = model.config.id2label
                predictions = [
                    {"label": id2label.get(idx.item(), f"LABEL_{idx.item()}"), 
                     "score": prob.item()}
                    for idx, prob in zip(top_5_indices, top_5_probs)
                ]
            else:
                predictions = [
                    {"label": f"LABEL_{idx.item()}", "score": prob.item()}
                    for idx, prob in zip(top_5_indices, top_5_probs)
                ]
            
            # Log top prediction
            top_label = predictions[0]["label"]
            top_score = predictions[0]["score"]
            logger.info(f"OpenVINO result: {top_label} ({top_score:.4f})")
            
            return {
                "success": True,
                "elapsed_time": time.time() - start_time,
                "predictions": predictions,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in OpenVINO test: {str(e)}")
            return {
                "success": False,
                "elapsed_time": time.time() - start_time,
                "predictions": None,
                "error": str(e)
            }
    {% endif %}
    
    def save_results(self, results, filename=None):
        """
        Save test results to a file.
        
        Args:
            results: Dictionary with test results
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to the saved file
        """
        if not self.output_dir:
            logger.warning("No output directory specified, results not saved")
            return None
            
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_safe = self.model_name.replace("/", "_")
            filename = f"{model_name_safe}_{timestamp}.json"
            
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")
        return output_path

def main():
    """
    Main function to run the test.
    """
    parser = argparse.ArgumentParser(description="Test {{ model_info.type|capitalize }} model")
    parser.add_argument("--model", default="{{ model_info.name }}", help="Model name or path")
    parser.add_argument("--output-dir", help="Directory to save outputs")
    parser.add_argument("--device", help="Device to run on (cpu, cuda:0, etc.)")
    parser.add_argument("--image", help="Path to image for testing")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    args = parser.parse_args()
    
    # Run test
    test = {{ model_info.type|capitalize }}Test(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        image_path=args.image
    )
    
    results = test.run()
    
    # Save results if requested
    if args.save or args.output_dir:
        test.save_results(results)
    
    # Print summary
    print(f"\nTest Summary for {args.model}:")
    print(f"  Device: {results['metadata']['device']}")
    print(f"  Test Type: {results['metadata']['test_type']}")
    
    for test_name, test_result in results["tests"].items():
        status = "✅ Passed" if test_result.get("success", False) else "❌ Failed"
        print(f"  {test_name}: {status}")
        if not test_result.get("success", False):
            print(f"    Error: {test_result.get('error', 'Unknown')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""