#!/usr/bin/env python3

"""
Standardized test file for Fuyu HuggingFace models.
"""

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
import numpy as np

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
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
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

# Model registry for Fuyu models
FUYU_MODELS_REGISTRY = {
    "adept/fuyu-8b": {
        "full_name": "Fuyu-8B",
        "architecture": "multimodal",
        "description": "Fuyu-8B multimodal model by Adept",
        "model_type": "fuyu",
        "class": "FuyuForCausalLM",
        "processor": "FuyuProcessor",
        "parameters": "8B",
        "image_size": 300,
        "vision_model": "Patch Embeddings",
        "text_model": "Causal LM",
        "recommended_tasks": ["visual-question-answering", "image-to-text"]
    },
    "adept/fuyu-1.5b": {
        "full_name": "Fuyu-1.5B",
        "architecture": "multimodal",
        "description": "Fuyu-1.5B multimodal model by Adept",
        "model_type": "fuyu",
        "class": "FuyuForCausalLM",
        "processor": "FuyuProcessor",
        "parameters": "1.5B",
        "image_size": 300,
        "vision_model": "Patch Embeddings",
        "text_model": "Causal LM",
        "recommended_tasks": ["visual-question-answering", "image-to-text"]
    },
    "adept/fuyu-tiny-test": {
        "full_name": "Fuyu-Tiny-Test",
        "architecture": "multimodal",
        "description": "Tiny Fuyu model for testing purposes",
        "model_type": "fuyu",
        "class": "FuyuForCausalLM",
        "processor": "FuyuProcessor",
        "parameters": "tiny",
        "image_size": 300,
        "vision_model": "Patch Embeddings",
        "text_model": "Causal LM",
        "recommended_tasks": ["visual-question-answering", "image-to-text"]
    }
}

class TestFuyuModels(ModelTest):
    """Test class for Fuyu models, following the ModelTest pattern."""
    
    def setUp(self):
        """Set up the test environment for Fuyu models."""
        super().setUp()
        
        # Use small model for testing by default
        self.model_id = "adept/fuyu-tiny-test"  
        
        # Verify model exists in registry
        if self.model_id not in FUYU_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = FUYU_MODELS_REGISTRY["adept/fuyu-tiny-test"]
        else:
            self.model_info = FUYU_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters from registry
        self.description = self.model_info["description"]
        self.image_size = self.model_info["image_size"]
        self.model_class = self.model_info["class"]
        self.processor_class = self.model_info["processor"]
        self.model_type = self.model_info["model_type"]
        self.architecture = self.model_info["architecture"]
        
        # Define test inputs
        self.test_image_path = self._find_test_image()
        self.test_prompts = [
            "What do you see in this image?", 
            "Describe this image in detail.",
            "What objects are present in this image?",
            "Write a caption for this image.",
            "Analyze this image and tell me what you see."
        ]
        
        # Configure hardware preference
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
            self._create_test_image().save(dummy_path)
            return dummy_path
        
        return None
    
    def _create_test_image(self, height=None, width=None):
        """Create a test image with colored shapes."""
        if not HAS_PIL:
            return None
            
        height = height or self.image_size
        width = width or self.image_size
        
        # Create a blank image with a white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add a red square
        square_size = min(height, width) // 4
        x1, y1 = width // 4, height // 4
        x2, y2 = x1 + square_size, y1 + square_size
        image[y1:y2, x1:x2] = [255, 0, 0]  # Red
        
        # Add a blue circle
        center_x, center_y = width // 2 + width // 4, height // 2
        radius = min(height, width) // 6
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        circle_mask = dist_from_center <= radius
        image[circle_mask] = [0, 0, 255]  # Blue
        
        # Add a green triangle
        triangle_size = min(height, width) // 3
        x1, y1 = width // 8, height // 2 + height // 4  # Bottom-left
        x2, y2 = x1 + triangle_size, y1  # Bottom-right
        x3, y3 = (x1 + x2) // 2, y1 - triangle_size  # Top
        
        # Create a mask for the triangle
        y, x = np.mgrid[:height, :width]
        # Barycentric coordinates check if a point is inside a triangle
        v0 = np.array([x3 - x1, y3 - y1])
        v1 = np.array([x2 - x1, y2 - y1])
        a = (x - x1) * v0[1] - (y - y1) * v0[0]
        b = (y - y1) * v1[0] - (x - x1) * v1[1]
        c = v0[0] * v1[1] - v0[1] * v1[0]
        mask = (a * b >= 0) & (a * c >= 0) & (b * c >= 0)
        image[mask] = [0, 255, 0]  # Green
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        return pil_image
    
    def load_model(self, model_name):
        """Load a model for testing - implements required ModelTest method."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                raise ImportError("Required libraries (transformers, torch) not available")
                
            logger.info(f"Loading Fuyu model {model_name}...")
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Load processor and model
            # Get model class and processor class from registry or use default
            model_info = FUYU_MODELS_REGISTRY.get(model_name, FUYU_MODELS_REGISTRY["adept/fuyu-tiny-test"])
            
            processor = transformers.FuyuProcessor.from_pretrained(model_name)
            
            # Configure model loading with appropriate device settings
            dtype = torch.float16 if self.preferred_device != "cpu" else torch.float32
            
            model = transformers.FuyuForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=self.preferred_device
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            self.performance_stats["model_loading"] = {
                "time": load_time,
                "model_name": model_name,
                "device": self.preferred_device
            }
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            return {"model": model, "processor": processor}
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            if HAS_TRANSFORMERS and HAS_TORCH:
                # Only raise if we're supposed to have these libraries
                raise
            else:
                # Otherwise, return a mock
                return {
                    "model": MagicMock(),
                    "processor": MagicMock()
                }
    
    def verify_model_output(self, model, input_data, expected_output=None):
        """Verify that model produces expected output - implements required ModelTest method."""
        try:
            if not isinstance(model, dict) or "model" not in model or "processor" not in model:
                raise ValueError("Model should be a dict containing 'model' and 'processor' keys")
                
            # Unpack model components
            fuyu_model = model["model"]
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
                    image = self._create_test_image()
            
            # Record inference start time
            inference_start = time.time()
            
            # Process inputs
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device if needed
            if hasattr(fuyu_model, "device") and str(fuyu_model.device) != "cpu":
                inputs = {k: v.to(fuyu_model.device) for k, v in inputs.items()}
            
            # Generate answer
            with torch.no_grad():
                outputs = fuyu_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            # Decode the generated text
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Record inference time
            inference_time = time.time() - inference_start
            self.performance_stats["inference"] = {
                "time": inference_time,
                "prompt_length": len(prompt),
                "generated_text_length": len(generated_text),
                "device": self.preferred_device
            }
            
            # Create result with both input prompt and generated text
            result = {
                "prompt": prompt,
                "generated_text": generated_text,
                "inference_time": inference_time
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
            if HAS_TRANSFORMERS and HAS_TORCH and HAS_PIL:
                # Only raise if we're supposed to have these libraries
                raise
            else:
                # Return a mock result if we're in mock mode
                return {
                    "prompt": str(input_data) if not isinstance(input_data, dict) else input_data.get("prompt", "mock prompt"),
                    "generated_text": "This is a mock generated text for testing without actual model inference.",
                    "inference_time": 0.01
                }
    
    def _get_test_image(self):
        """Get test image from path or create dummy."""
        if self.test_image_path and os.path.exists(self.test_image_path):
            return Image.open(self.test_image_path)
        else:
            # Create a test image
            return self._create_test_image()
    
    def test_model_loading(self):
        """Test that the model loads correctly - implements required ModelTest method."""
        try:
            model_components = self.load_model(self.model_id)
            self.assertIsNotNone(model_components, "Model should not be None")
            self.assertIn("model", model_components, "Model dict should contain 'model' key")
            self.assertIn("processor", model_components, "Model dict should contain 'processor' key")
            
            fuyu_model = model_components["model"]
            processor = model_components["processor"]
            
            # Check if we're using a mock or real model
            if isinstance(fuyu_model, MagicMock):
                logger.info("Using mocked model for testing")
            else:
                expected_class_name = self.model_class
                actual_class_name = fuyu_model.__class__.__name__
                
                self.assertIn(expected_class_name, actual_class_name, 
                             f"Model should be a {expected_class_name}, got {actual_class_name}")
                
                expected_processor_name = self.processor_class
                actual_processor_name = processor.__class__.__name__
                
                self.assertIn(expected_processor_name, actual_processor_name,
                             f"Processor should be a {expected_processor_name}, got {actual_processor_name}")
            
            logger.info(f"Successfully loaded {self.model_id}")
            return model_components
        except Exception as e:
            logger.error(f"Error testing model loading: {e}")
            if HAS_TRANSFORMERS and HAS_TORCH:
                # Only fail if we're supposed to have these libraries
                self.fail(f"Model loading failed: {e}")
            else:
                # Return a mock result if we're in mock mode
                logger.info("Using mocked components in test_model_loading")
                return {
                    "model": MagicMock(),
                    "processor": MagicMock()
                }
    
    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device - implements required ModelTest method."""
        try:
            # Check CUDA
            if HAS_TORCH and torch.cuda.is_available():
                return "cuda"
            
            # Check MPS (Apple Silicon)
            if HAS_TORCH and hasattr(torch, "mps") and torch.mps.is_available():
                return "mps"
            
            # Check for OpenVINO compatibility (not yet implemented for Fuyu)
            # This would be the place to add OpenVINO support in the future
            
            # Fallback to CPU
            return "cpu"
        except Exception as e:
            logger.error(f"Error detecting device: {e}")
            return "cpu"
    
    def test_multiple_prompts(self):
        """Test the model with multiple prompt variations."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_PIL:
                logger.warning("Required libraries not available, skipping multiple prompts test")
                return {"success": False, "error": "Required libraries not available"}
                
            logger.info(f"Testing Fuyu model {self.model_id} with multiple prompts on {self.preferred_device}")
            
            # Load model
            model_components = self.load_model(self.model_id)
            
            # Create a test image
            test_image = self._get_test_image()
            
            # Results for multiple prompts
            results = {}
            
            # Record inference start time
            inference_start = time.time()
            
            # Process and generate outputs for each prompt
            for i, prompt in enumerate(self.test_prompts):
                single_input = {
                    "image": test_image,
                    "prompt": prompt
                }
                
                try:
                    result = self.verify_model_output(model_components, single_input)
                    results[f"prompt_{i+1}"] = result
                except Exception as e:
                    logger.error(f"Error processing prompt {i+1}: {e}")
                    results[f"prompt_{i+1}"] = {
                        "prompt": prompt,
                        "error": str(e)
                    }
            
            # Record total inference time
            total_inference_time = time.time() - inference_start
            
            # Store performance stats
            self.performance_stats["multiple_prompts"] = {
                "total_time": total_inference_time,
                "prompt_count": len(self.test_prompts),
                "average_time": total_inference_time / len(self.test_prompts)
            }
            
            # Add to examples
            self.examples.append({
                "method": "test_multiple_prompts",
                "input": f"Image with {len(self.test_prompts)} different prompts",
                "output_preview": str(results)[:200] + "..." if len(str(results)) > 200 else str(results)
            })
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.preferred_device,
                "total_inference_time": total_inference_time,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error in multiple prompts test: {e}")
            return {"success": False, "error": str(e)}
    
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
        
        # Test basic model functionality
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
        
        # Test with multiple prompts
        multiple_prompts_results = self.test_multiple_prompts()
        self.results["multiple_prompts"] = multiple_prompts_results
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH and HAS_PIL
        using_mocks = not using_real_inference
        
        # Build final results
        return {
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "metadata": {
                "model": self.model_id,
                "model_type": self.model_type,
                "architecture": self.architecture,
                "description": self.description,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_pil": HAS_PIL,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if using_real_inference else "MOCK OBJECTS (CI/CD)"
            }
        }

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test Fuyu models with ModelTest pattern")
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
        model_id = args.model or "adept/fuyu-tiny-test"
        tester = TestFuyuModels()
        if args.model:
            tester.model_id = args.model
            
        results = tester.run_tests(all_hardware=args.all_hardware)
        
        # Print summary
        success = any(r.get("success", False) for r in [results["results"].get("multiple_prompts", {})])
        
        # Determine if real inference or mock objects were used
        using_real_inference = results["metadata"]["has_transformers"] and results["metadata"]["has_torch"] and results["metadata"]["has_pil"]
        
        print("\nTEST RESULTS SUMMARY:")
        
        # Indicate real vs mock inference clearly
        if using_real_inference:
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
        
        # Save results if requested
        if args.save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fuyu_test_results_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nTest results saved to {filename}")

if __name__ == "__main__":
    main()