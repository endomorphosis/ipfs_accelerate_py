#!/usr/bin/env python3

import os
import sys
import json
import time
import datetime
import logging
import argparse
import traceback
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import ModelTest with fallbacks
try:
    from refactored_test_suite.model_test import ModelTest
except ImportError:
    try:
        from model_test import ModelTest
    except ImportError:
        # Create a minimal ModelTest class if not available
        class ModelTest(unittest.TestCase):
            """Minimal ModelTest class if the real one is not available."""
            def setUp(self):
                super().setUp()

# Try to import required packages with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = MagicMock()
    HAS_PIL = False
    logger.warning("PIL not available, using mock")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = MagicMock()
    HAS_REQUESTS = False
    logger.warning("requests not available, using mock")

# Models registry
VIT_MODELS_REGISTRY = {
    "google/vit-base-patch16-224": {
        "description": "vit base model",
        "class": "ViTForImageClassification",
    }
}

class TestVitModels(ModelTest):
    """Test class for vit models."""
    
    def setUp(self):
        """Initialize the test class."""
        super().setUp()
        self.model_id = "google/vit-base-patch16-224"
        
        # Use registry information
        if self.model_id not in VIT_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default")
            self.model_info = VIT_MODELS_REGISTRY["google/vit-base-patch16-224"]
        else:
            self.model_info = VIT_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "image-classification"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        
        # Configure hardware preference using detect_preferred_device
        self.preferred_device = self.detect_preferred_device()
        logger.info(f"Using {self.preferred_device} as preferred device")
        
        # Results storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}
    
    def load_model(self, model_name):
        """Load a model for testing."""
        try:
            if not HAS_TRANSFORMERS:
                raise ImportError("transformers package not available")
                
            logger.info(f"Loading model {model_name}...")
            
            # Create model and processor for ViT image classification
            model = transformers.ViTForImageClassification.from_pretrained(model_name)
            processor = transformers.ViTImageProcessor.from_pretrained(model_name)
            
            # Move model to preferred device if possible
            if self.preferred_device == "cuda" and HAS_TORCH and torch.cuda.is_available():
                model = model.to("cuda")
            elif self.preferred_device == "mps" and HAS_TORCH and hasattr(torch, "mps") and torch.mps.is_available():
                model = model.to("mps")
                
            return {"model": model, "processor": processor}
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def verify_model_output(self, model, input_data, expected_output=None):
        """Verify that model produces expected output."""
        try:
            if not isinstance(model, dict) or "model" not in model or "processor" not in model:
                raise ValueError("Model should be a dict containing 'model' and 'processor' keys")
                
            # Unpack model components
            vit_model = model["model"]
            processor = model["processor"]
            
            # Process input image
            if isinstance(input_data, str) and input_data.startswith("http"):
                # Input is a URL, download image
                if not HAS_REQUESTS or not HAS_PIL:
                    raise ImportError("requests and PIL are required for processing image URLs")
                    
                from io import BytesIO
                response = requests.get(input_data)
                image = Image.open(BytesIO(response.content))
            elif isinstance(input_data, str) and os.path.isfile(input_data):
                # Input is a file path
                if not HAS_PIL:
                    raise ImportError("PIL is required for processing image files")
                image = Image.open(input_data)
            elif hasattr(input_data, 'mode') and hasattr(input_data, 'size'):
                # Input appears to be a PIL image
                image = input_data
            else:
                # Assume input is already processed
                image = input_data
            
            # Process the image
            inputs = processor(images=image, return_tensors="pt")
            
            # Move inputs to the right device if model is on a specific device
            if hasattr(vit_model, 'device') and str(vit_model.device) != "cpu":
                inputs = {k: v.to(vit_model.device) for k, v in inputs.items() if hasattr(v, 'to')}
            
            # Perform inference
            with torch.no_grad():
                outputs = vit_model(**inputs)
            
            # Process the outputs to get predicted class
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            
            # Get the class label if available
            if hasattr(vit_model.config, 'id2label') and predicted_class_idx in vit_model.config.id2label:
                predicted_class = vit_model.config.id2label[predicted_class_idx]
            else:
                predicted_class = f"class_{predicted_class_idx}"
            
            # Verify output
            self.assertIsNotNone(outputs, "Model output should not be None")
            self.assertIsNotNone(predicted_class, "Predicted class should not be None")
            
            # If expected output is provided, compare with actual output
            if expected_output is not None:
                self.assertEqual(expected_output, predicted_class)
                
            return {"class_idx": predicted_class_idx, "class_name": predicted_class}
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            raise
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        try:
            model_components = self.load_model(self.model_id)
            self.assertIsNotNone(model_components, "Model should not be None")
            self.assertIn("model", model_components, "Model dict should contain 'model' key")
            self.assertIn("processor", model_components, "Model dict should contain 'processor' key")
            
            vit_model = model_components["model"]
            self.assertEqual(vit_model.config.model_type, "vit", "Model should be a ViT model")
            
            logger.info(f"Successfully loaded {self.model_id}")
            return model_components
        except Exception as e:
            logger.error(f"Error testing model loading: {e}")
            self.fail(f"Model loading failed: {e}")
    
    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device."""
        try:
            # Check CUDA
            if HAS_TORCH and torch.cuda.is_available():
                return "cuda"
            
            # Check MPS (Apple Silicon)
            if HAS_TORCH and hasattr(torch, "mps") and torch.mps.is_available():
                return "mps"
            
            # Fallback to CPU
            return "cpu"
        except Exception as e:
            logger.error(f"Error detecting device: {e}")
            return "cpu"
    
    def test_pipeline(self, device="auto"):
        """Test the model using pipeline API."""
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
        
        try:
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
            
            # Prepare test input
            # For vision models
            import requests
            from PIL import Image
            from io import BytesIO
            
            response = requests.get(self.test_image_url)
            pipeline_input = Image.open(BytesIO(response.content))

            # Run inference
            output = pipeline(pipeline_input)
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_load_time"] = load_time
            results["pipeline_error_type"] = "none"
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_error_type"] = "other"
            logger.error(f"Error testing pipeline: {e}")
        
        # Add to overall results
        self.results["pipeline"] = results
        return results
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        # Test on default device
        self.test_pipeline()
        
        # Build results
        return {
            "results": self.results,
            "examples": self.examples,
            "hardware": self.detect_preferred_device(),
            "metadata": {
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        }

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test vit models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--unittest", action="store_true", help="Run as unittest")
    
    args = parser.parse_args()
    
    if args.unittest or "unittest" in sys.argv:
        # Run as unittest
        unittest.main(argv=[sys.argv[0]])
    else:
        # Override preferred device if CPU only
        if args.cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("CPU-only mode enabled")
        
        # Run test
        tester = TestVitModels()
        if args.model:
            tester.model_id = args.model
            
        results = tester.run_tests()
        
        # Print summary
        success = any(r.get("pipeline_success", False) for r in results["results"].values())
        
        print("\nTEST RESULTS SUMMARY:")
        if success:
            print(f"✅ Successfully tested {tester.model_id}")
        else:
            print(f"❌ Failed to test {tester.model_id}")
            for test_name, result in results["results"].items():
                if "pipeline_error" in result:
                    print(f"  - Error in {test_name}: {result.get('pipeline_error', 'Unknown error')}")

if __name__ == "__main__":
    main()
