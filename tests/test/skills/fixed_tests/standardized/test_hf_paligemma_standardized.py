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
from typing import Dict, List, Any, Optional, Union

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

# Check if we should mock specific dependencies
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'

# Try to import required packages with fallbacks
try:
    if MOCK_TORCH:
        raise ImportError("Mocked torch import failure")
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    np = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

try:
    if MOCK_TRANSFORMERS:
        raise ImportError("Mocked transformers import failure")
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Try to import PIL
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
    logger.warning("PIL or requests not available, using mock")

# Create mock implementations if PIL is missing
if not HAS_PIL:
    class MockImage:
        @staticmethod
        def open(file):
            class MockImg:
                def __init__(self):
                    self.size = (224, 224)
                
                def convert(self, mode):
                    return self
                
                def resize(self, size):
                    return self
            return MockImg()
            
    class MockRequests:
        @staticmethod
        def get(url):
            class MockResponse:
                def __init__(self):
                    self.content = b"mock image data"
                
                def raise_for_status(self):
                    pass
            return MockResponse()

    Image.open = MockImage.open
    requests.get = MockRequests.get

# Models registry - Maps model IDs to their specific configurations
PALIGEMMA_MODELS_REGISTRY = {
    "google/paligemma-3b-pt-224": {
        "description": "Paligemma 3B model (pretraining, image size 224)",
        "class": "PaligemmaModel",
        "type": "paligemma",
        "image_size": 224,
        "task": "image-to-text",
        "variant": "paligemma-pt"
    },
    "google/paligemma-3b-mix-224": {
        "description": "Paligemma 3B model (mixed, image size 224)",
        "class": "PaligemmaForConditionalGeneration",
        "type": "paligemma",
        "image_size": 224,
        "task": "image-to-text",
        "variant": "paligemma-mix"
    },
}

class TestPaligemmaModels(ModelTest):
    """Test class for Paligemma vision-language models."""
    
    def setUp(self):
        """Initialize the test class for a specific model or default."""
        super().setUp()
        self.model_id = "google/paligemma-3b-pt-224"  # Default model for testing
        
        # Verify model exists in registry
        if self.model_id not in PALIGEMMA_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = PALIGEMMA_MODELS_REGISTRY["google/paligemma-3b-pt-224"]
        else:
            self.model_info = PALIGEMMA_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = self.model_info.get("task", "image-to-text")
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        self.image_size = self.model_info["image_size"]
        self.variant = self.model_info.get("variant", "paligemma-pt")
        
        # Define test inputs
        self.test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.test_image_path = self._find_test_image()
        self.test_prompts = [
            "What do you see in this image?", 
            "Describe this image in detail.",
            "What objects are present in this image?"
        ]
        
        # Configure hardware preference using detect_preferred_device
        self.preferred_device = self.detect_preferred_device()
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
        
        # Try to download an image if PIL and requests are available
        if HAS_PIL and 'requests' in sys.modules:
            try:
                dummy_path = "paligemma_test_image.jpg"
                response = requests.get(self.test_image_url)
                response.raise_for_status()
                with open(dummy_path, 'wb') as f:
                    f.write(response.content)
                return dummy_path
            except Exception as e:
                logger.warning(f"Failed to download test image: {e}")
        
        # Create a dummy image if download failed or PIL not available
        if HAS_PIL:
            dummy_path = "paligemma_test_dummy.jpg"
            img = Image.new('RGB', (self.image_size, self.image_size), color = (73, 109, 137))
            img.save(dummy_path)
            return dummy_path
        
        return None

    def load_model(self, model_name):
        """Load a model for testing - implements required ModelTest method."""
        try:
            if not HAS_TRANSFORMERS:
                raise ImportError("transformers package not available")
                
            logger.info(f"Loading Paligemma model {model_name}...")
            
            # Load processor and model
            if self.class_name == "PaligemmaModel":
                processor = transformers.AutoProcessor.from_pretrained(model_name)
                model = transformers.PaligemmaModel.from_pretrained(model_name)
            else:
                processor = transformers.AutoProcessor.from_pretrained(model_name)
                model = transformers.PaligemmaForConditionalGeneration.from_pretrained(model_name)
            
            # Move model to preferred device if possible
            if self.preferred_device == "cuda" and HAS_TORCH and torch.cuda.is_available():
                model = model.to("cuda")
                logger.info(f"Moved model to CUDA device")
            elif self.preferred_device == "mps" and HAS_TORCH and hasattr(torch, "mps") and torch.mps.is_available():
                model = model.to("mps")
                logger.info(f"Moved model to MPS (Apple Silicon) device")
                
            return {"model": model, "processor": processor}
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def verify_model_output(self, model, input_data, expected_output=None):
        """Verify that model produces expected output - implements required ModelTest method."""
        try:
            if not isinstance(model, dict) or "model" not in model or "processor" not in model:
                raise ValueError("Model should be a dict containing 'model' and 'processor' keys")
                
            # Unpack model components
            paligemma_model = model["model"]
            processor = model["processor"]
            
            # Process inputs based on type
            if isinstance(input_data, dict):
                if "image" in input_data:
                    image = input_data["image"]
                else:
                    image = self._get_test_image()
                    
                if "prompt" in input_data:
                    prompt = input_data["prompt"]
                else:
                    prompt = self.test_prompts[0]
            else:
                # Use default test inputs
                image = self._get_test_image()
                prompt = self.test_prompts[0]
            
            # Process image input
            if isinstance(image, str):
                # Load image from path
                if os.path.exists(image):
                    image = Image.open(image)
                else:
                    # Create a random dummy image
                    image = Image.new('RGB', (self.image_size, self.image_size), color=(73, 109, 137))
            
            # Process inputs with Paligemma processor
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device if needed
            if hasattr(paligemma_model, "device") and str(paligemma_model.device) != "cpu":
                inputs = {k: v.to(paligemma_model.device) for k, v in inputs.items()}
            
            # Run generation - different based on model type
            with torch.no_grad():
                if self.class_name == "PaligemmaModel":
                    # Encoder-only case
                    outputs = paligemma_model(**inputs)
                    # Create a summary of the outputs
                    last_hidden_state = outputs.last_hidden_state
                    result = {
                        "prompt": prompt,
                        "output_shape": list(last_hidden_state.shape),
                        "output_mean": float(last_hidden_state.mean().item()),
                        "output_std": float(last_hidden_state.std().item())
                    }
                else:
                    # PaligemmaForConditionalGeneration case - generates text
                    output_ids = paligemma_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False
                    )
                    
                    # Decode the generated text
                    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                    
                    # Create result with both input prompt and generated text
                    result = {
                        "prompt": prompt,
                        "generated_text": generated_text
                    }
            
            # Verify output
            self.assertIsNotNone(result, "Model output should not be None")
            
            if self.class_name == "PaligemmaForConditionalGeneration":
                # For generation models, verify text output
                self.assertIn("generated_text", result, "Generated text should be present")
                self.assertIsInstance(result["generated_text"], str, "Generated text should be a string")
                self.assertGreater(len(result["generated_text"]), 0, "Generated text should not be empty")
            else:
                # For encoder models, verify embedding output
                self.assertIn("output_shape", result, "Output shape should be present")
                self.assertIsInstance(result["output_shape"], list, "Output shape should be a list")
                self.assertGreater(len(result["output_shape"]), 0, "Output shape should not be empty")
            
            # If expected output is provided, compare with actual output
            if expected_output is not None:
                self.assertEqual(expected_output, result)
                
            return result
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            raise
    
    def _get_test_image(self):
        """Get test image from path or create dummy."""
        if self.test_image_path and os.path.exists(self.test_image_path):
            return Image.open(self.test_image_path)
        else:
            # Create a random dummy image
            return Image.new('RGB', (self.image_size, self.image_size), color=(73, 109, 137))
    
    def test_model_loading(self):
        """Test that the model loads correctly - implements required ModelTest method."""
        try:
            model_components = self.load_model(self.model_id)
            self.assertIsNotNone(model_components, "Model should not be None")
            self.assertIn("model", model_components, "Model dict should contain 'model' key")
            self.assertIn("processor", model_components, "Model dict should contain 'processor' key")
            
            paligemma_model = model_components["model"]
            
            # Check if we're using a mock or real model
            if isinstance(paligemma_model, MagicMock):
                logger.info("Using mocked model for testing")
            else:
                # Check for expected class name
                expected_class_name = self.class_name
                actual_class_name = paligemma_model.__class__.__name__
                
                self.assertIn(expected_class_name, actual_class_name, 
                             f"Model should be a {expected_class_name}, got {actual_class_name}")
            
            logger.info(f"Successfully loaded {self.model_id}")
            return model_components
        except Exception as e:
            logger.error(f"Error testing model loading: {e}")
            self.fail(f"Model loading failed: {e}")
    
    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device - implements required ModelTest method."""
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
    
    def test_with_openvino(self):
        """Test the model using OpenVINO integration."""
        results = {
            "model": self.model_id,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for OpenVINO support
        try:
            import openvino
            from openvino.runtime import Core
            HAS_OPENVINO = True
        except ImportError:
            HAS_OPENVINO = False
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
            from optimum.intel import OVModelForVision2Seq
            logger.info(f"Testing {self.model_id} with OpenVINO...")
            
            # Load model components
            model_components = self.load_model(self.model_id)
            processor = model_components["processor"]
            
            # Convert model to OpenVINO IR
            model_ov = OVModelForVision2Seq.from_pretrained(
                self.model_id,
                export=True,
                provider="CPU"
            )
            
            # Prepare inputs
            test_image = self._get_test_image()
            test_prompt = self.test_prompts[0]
            
            # Process inputs
            inputs = processor(
                text=test_prompt,
                images=test_image,
                return_tensors="pt"
            )
            
            # Run inference
            start_time = time.time()
            outputs = model_ov(**inputs)
            inference_time = time.time() - start_time
            
            # Process results
            if hasattr(outputs, "last_hidden_state"):
                last_hidden_state = outputs.last_hidden_state
                output_preview = {
                    "shape": list(last_hidden_state.shape),
                    "mean": float(last_hidden_state.mean().item()),
                    "std": float(last_hidden_state.std().item())
                }
            else:
                output_preview = {"output": "Processed OpenVINO output"}
            
            # Store results
            results["openvino_success"] = True
            results["openvino_inference_time"] = inference_time
            
            if "shape" in output_preview:
                results["output_shape"] = output_preview["shape"]
            
            results["openvino_error_type"] = "none"
            
            # Add to examples
            self.examples.append({
                "method": "OpenVINO inference",
                "input": f"Image and text prompt: {test_prompt}",
                "output_preview": output_preview
            })
            
            # Store in performance stats
            self.performance_stats["openvino"] = {
                "inference_time": inference_time
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
        # Run required ModelTest method
        model_components = self.test_model_loading()
        
        # Test some sample input with the model
        if model_components:
            try:
                test_image = self._get_test_image()
                test_prompt = self.test_prompts[0]
                
                test_input = {
                    "image": test_image,
                    "prompt": test_prompt
                }
                
                output = self.verify_model_output(model_components, test_input)
                self.examples.append({
                    "method": "verify_model_output",
                    "input": f"Image size: {test_image.size if hasattr(test_image, 'size') else 'unknown'}, Prompt: {test_prompt}",
                    "output": output
                })
            except Exception as e:
                logger.error(f"Error running model verification: {e}")
        
        # Test on OpenVINO if available and requested
        if all_hardware:
            try:
                import openvino
                from openvino.runtime import Core
                HAS_OPENVINO = True
                openvino_results = self.test_with_openvino()
                self.results["openvino"] = openvino_results
            except ImportError:
                logger.warning("OpenVINO not available for testing")
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_PIL
        
        # Build final results
        return {
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "metadata": {
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "description": self.description,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_pil": HAS_PIL,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if using_real_inference and not using_mocks else "MOCK OBJECTS (CI/CD)",
                "variant": self.variant
            }
        }

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test Paligemma vision-language models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
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
        model_id = args.model or "google/paligemma-3b-pt-224"
        tester = TestPaligemmaModels()
        if args.model:
            tester.model_id = args.model
            
        results = tester.run_tests(all_hardware=args.all_hardware)
        
        # Print summary
        success = len(results["examples"]) > 0
        
        # Determine if real inference or mock objects were used
        using_real_inference = results["metadata"]["has_transformers"] and results["metadata"]["has_torch"]
        using_mocks = not using_real_inference or not results["metadata"]["has_pil"]
        
        print("\nTEST RESULTS SUMMARY:")
        
        # Indicate real vs mock inference clearly
        if using_real_inference and not using_mocks:
            print("🚀 Using REAL INFERENCE with actual models")
        else:
            print("🔷 Using MOCK OBJECTS for CI/CD testing only")
            print(f"   Dependencies: transformers={results['metadata']['has_transformers']}, torch={results['metadata']['has_torch']}, PIL={results['metadata']['has_pil']}")
        
        if success:
            print(f"✅ Successfully tested {model_id}")
            
            # Print a sample output if available
            if results["examples"]:
                example = results["examples"][0]
                if isinstance(example, dict) and "output" in example:
                    print(f"\nSample output:")
                    output = example["output"]
                    if isinstance(output, dict):
                        if "generated_text" in output:
                            print(f"  Generated text: {output['generated_text'][:100]}...")
                        elif "output_shape" in output:
                            print(f"  Output shape: {output['output_shape']}")
                            print(f"  Output mean: {output['output_mean']:.4f}")
        else:
            print(f"❌ Failed to test {model_id}")
            # Print error details if available
            if "results" in results:
                for test_name, result in results["results"].items():
                    if isinstance(result, dict) and "error" in result:
                        print(f"  - Error in {test_name}: {result.get('error')}")

if __name__ == "__main__":
    main()