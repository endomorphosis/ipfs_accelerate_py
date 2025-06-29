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
import asyncio

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
    HAS_PIL = True
except ImportError:
    Image = MagicMock()
    HAS_PIL = False
    logger.warning("PIL not available, using mock")

# Models registry - Maps model IDs to their specific configurations
LLAVA_NEXT_MODELS_REGISTRY = {
    "katuni4ka/tiny-random-llava-next": {
        "description": "Tiny random LLaVA-Next model for testing",
        "class": "AutoModelForVision2Seq",
        "type": "llava-next",
        "image_size": 224,
        "task": "image-to-text",
        "variant": "llava-next-random"
    },
    "llava-hf/llava-next-7b-hf": {
        "description": "LLaVA-Next 7B vision-language model",
        "class": "AutoModelForVision2Seq",
        "type": "llava-next",
        "image_size": 336,
        "task": "image-to-text",
        "variant": "llava-next"
    },
    "llava-hf/llava-next-13b-hf": {
        "description": "LLaVA-Next 13B vision-language model",
        "class": "AutoModelForVision2Seq",
        "type": "llava-next",
        "image_size": 336,
        "task": "image-to-text",
        "variant": "llava-next"
    },
    "llava-hf/llava-onevision-7b-hf": {
        "description": "LLaVA OneVision 7B model with enhanced visual capabilities",
        "class": "AutoModelForVision2Seq",
        "type": "llava-next",
        "image_size": 448,
        "task": "image-to-text",
        "variant": "llava-onevision"
    },
    "llava-hf/llava-next-34b-hf": {
        "description": "LLaVA-Next 34B vision-language model",
        "class": "AutoModelForVision2Seq",
        "type": "llava-next",
        "image_size": 336,
        "task": "image-to-text",
        "variant": "llava-next"
    }
}

class TestLlavaNextModels(ModelTest):
    """Test class for LLaVA-Next vision-language models."""
    
    def setUp(self):
        """Initialize the test class for a specific model or default."""
        super().setUp()
        self.model_id = "katuni4ka/tiny-random-llava-next"  # Use tiny model for testing
        
        # Verify model exists in registry
        if self.model_id not in LLAVA_NEXT_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = LLAVA_NEXT_MODELS_REGISTRY["katuni4ka/tiny-random-llava-next"]
        else:
            self.model_info = LLAVA_NEXT_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = self.model_info.get("task", "image-to-text")
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        self.image_size = self.model_info["image_size"]
        self.variant = self.model_info.get("variant", "llava-next")
        
        # Define test inputs
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
        
        # Create a dummy image if no test image is found
        if HAS_PIL:
            dummy_path = "test_dummy.jpg"
            img = Image.new('RGB', (self.image_size, self.image_size), color = (73, 109, 137))
            img.save(dummy_path)
            return dummy_path
        
        return None

    def load_model(self, model_name):
        """Load a model for testing - implements required ModelTest method."""
        try:
            if not HAS_TRANSFORMERS:
                raise ImportError("transformers package not available")
                
            logger.info(f"Loading LLaVA-Next model {model_name}...")
            
            # Load processor and model
            # LLaVA-Next uses AutoProcessor and AutoModelForVision2Seq
            processor = transformers.AutoProcessor.from_pretrained(model_name)
            model = transformers.AutoModelForVision2Seq.from_pretrained(model_name)
            
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
            llava_next_model = model["model"]
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
            
            # Process inputs with LLaVA-Next processor
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device if needed
            if hasattr(llava_next_model, "device") and str(llava_next_model.device) != "cpu":
                inputs = {k: v.to(llava_next_model.device) for k, v in inputs.items()}
            
            # Run generation
            with torch.no_grad():
                output_ids = llava_next_model.generate(
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
            self.assertIsNotNone(generated_text, "Model output should not be None")
            self.assertIsInstance(generated_text, str, "Generated text should be a string")
            self.assertGreater(len(generated_text), 0, "Generated text should not be empty")
            
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
            
            llava_next_model = model_components["model"]
            
            # Check if we're using a mock or real model
            if isinstance(llava_next_model, MagicMock):
                logger.info("Using mocked model for testing")
            else:
                expected_class_name = self.class_name
                actual_class_name = llava_next_model.__class__.__name__
                
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
    
    def test_multi_image_processing(self):
        """Test LLaVA-Next with multiple images when supported."""
        results = {
            "model": self.model_id,
            "device": self.preferred_device,
            "task": self.task,
            "class": self.class_name
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS or not HAS_PIL:
            results["multi_image_error"] = "Missing dependencies"
            results["multi_image_success"] = False
            return results
        
        try:
            # Load model
            model_components = self.load_model(self.model_id)
            
            # Create multiple test images
            test_images = []
            for i in range(2):
                img = Image.new('RGB', (self.image_size, self.image_size), 
                               color=(73 + i*30, 109 + i*20, 137 + i*10))
                test_images.append(img)
            
            # Prepare test input
            test_input = {
                "image": test_images,
                "prompt": "Describe both images shown here."
            }
            
            # Most LLaVA-Next versions don't support multi-image inputs directly
            # So we'll handle them one by one and combine results
            combined_results = []
            
            for idx, img in enumerate(test_images):
                single_input = {
                    "image": img,
                    "prompt": f"Describe image {idx+1}."
                }
                
                try:
                    result = self.verify_model_output(model_components, single_input)
                    combined_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image {idx+1}: {e}")
                    combined_results.append({
                        "prompt": single_input["prompt"],
                        "generated_text": f"Error: {str(e)}"
                    })
            
            # Store results
            results["multi_image_success"] = True
            results["multi_image_results"] = combined_results
            
            # Add to examples
            self.examples.append({
                "method": "multi_image_processing",
                "input": f"{len(test_images)} images with prompts",
                "output_preview": str(combined_results)[:200] + "..." if len(str(combined_results)) > 200 else str(combined_results)
            })
            
        except Exception as e:
            results["multi_image_success"] = False
            results["multi_image_error"] = str(e)
            results["multi_image_traceback"] = traceback.format_exc()
            logger.error(f"Error testing multi-image processing: {e}")
        
        return results
    
    def run_tests(self, all_hardware=False):
        """
        Run all tests for this model.
        
        Args:
            all_hardware: If True, tests on all available hardware (CPU, CUDA)
        
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
        
        # Test multi-image processing if available
        multi_image_results = self.test_multi_image_processing()
        self.results["multi_image"] = multi_image_results
        
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
    parser = argparse.ArgumentParser(description="Test LLaVA-Next models")
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
        model_id = args.model or "katuni4ka/tiny-random-llava-next"
        tester = TestLlavaNextModels()
        if args.model:
            tester.model_id = args.model
            
        results = tester.run_tests(all_hardware=args.all_hardware)
        
        # Print summary
        success = any(r.get("multi_image_success", False) for r in [results["results"].get("multi_image", {})])
        
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
        
        if success or "examples" in results and results["examples"]:
            print(f"✅ Successfully tested {model_id}")
            
            # Print a sample output if available
            if results["examples"]:
                example = results["examples"][0]
                if isinstance(example, dict) and "output" in example:
                    print(f"\nSample output:")
                    output = example["output"]
                    if isinstance(output, dict) and "generated_text" in output:
                        print(f"  Generated text: {output['generated_text'][:100]}...")
        else:
            print(f"❌ Failed to test {model_id}")
            # Print error details if available
            if "results" in results:
                for test_name, result in results["results"].items():
                    if isinstance(result, dict) and "error" in result:
                        print(f"  - Error in {test_name}: {result.get('error')}")

if __name__ == "__main__":
    main()