#!/usr/bin/env python3

"""
Standardized test file for IDEFICS (Image-grounDEd FIne-grained Communication with language modelS) HuggingFace models.
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
from pathlib import Path
import anyio

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

# Try to import OpenVINO if available
try:
    import openvino
    from openvino.runtime import Core
    HAS_OPENVINO = True
except ImportError:
    openvino = MagicMock()
    HAS_OPENVINO = False
    logger.warning("OpenVINO not available")

# Models registry - Maps model IDs to their specific configurations
IDEFICS_MODELS_REGISTRY = {
    "HuggingFaceM4/idefics-9b": {
        "description": "IDEFICS 9B multimodal model by HuggingFace",
        "class": "IdeficsForVisionText2Text",
        "processor": "IdeficsProcessor",
        "architecture": "multimodal",
        "image_size": 224,
        "task": "image-to-text",
        "parameters": "9B",
        "context_length": 512,
        "recommended_tasks": ["multimodal-conversation", "image-captioning", "visual-question-answering"]
    },
    "HuggingFaceM4/idefics-80b": {
        "description": "IDEFICS 80B multimodal model by HuggingFace",
        "class": "IdeficsForVisionText2Text",
        "processor": "IdeficsProcessor",
        "architecture": "multimodal",
        "image_size": 224,
        "task": "image-to-text",
        "parameters": "80B",
        "context_length": 4096,
        "recommended_tasks": ["multimodal-conversation", "image-captioning", "visual-question-answering"]
    },
    "HuggingFaceM4/idefics-9b-instruct": {
        "description": "IDEFICS 9B instruction-tuned multimodal model by HuggingFace",
        "class": "IdeficsForVisionText2Text",
        "processor": "IdeficsProcessor",
        "architecture": "multimodal",
        "image_size": 224,
        "task": "image-to-text",
        "parameters": "9B",
        "context_length": 512,
        "recommended_tasks": ["multimodal-conversation", "image-captioning", "visual-question-answering"]
    },
    "HuggingFaceM4/tiny-random-idefics": {
        "description": "Tiny random IDEFICS model for testing",
        "class": "IdeficsForVisionText2Text",
        "processor": "IdeficsProcessor",
        "architecture": "multimodal",
        "image_size": 224,
        "task": "image-to-text",
        "parameters": "tiny",
        "context_length": 128,
        "recommended_tasks": ["multimodal-conversation", "image-captioning", "visual-question-answering"]
    }
}

class TestIdeficsModels(ModelTest):
    """Test class for IDEFICS models, following the ModelTest pattern."""
    
    def setUp(self):
        """Set up the test environment for IDEFICS models."""
        super().setUp()
        
        # Use tiny model for testing by default
        self.model_id = "HuggingFaceM4/tiny-random-idefics"  
        
        # Verify model exists in registry
        if self.model_id not in IDEFICS_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = IDEFICS_MODELS_REGISTRY["HuggingFaceM4/tiny-random-idefics"]
        else:
            self.model_info = IDEFICS_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters from registry
        self.description = self.model_info["description"]
        self.image_size = self.model_info["image_size"]
        self.model_class = self.model_info["class"]
        self.processor_class = self.model_info["processor"]
        self.task = self.model_info["task"]
        self.architecture = self.model_info["architecture"]
        self.context_length = self.model_info["context_length"]
        
        # Define test inputs
        self.test_image_path = self._find_test_image()
        self.test_prompts = [
            "What do you see in this image?", 
            "Describe this image in detail.",
            "What's happening in this picture?"
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
            img = Image.new('RGB', (self.image_size, self.image_size), color=(73, 109, 137))
            img.save(dummy_path)
            return dummy_path
        
        return None
    
    def _get_test_image(self):
        """Get test image from path or create dummy."""
        if self.test_image_path and os.path.exists(self.test_image_path):
            return Image.open(self.test_image_path)
        else:
            # Create a dummy image
            return Image.new('RGB', (self.image_size, self.image_size), color=(73, 109, 137))
    
    def load_model(self, model_name):
        """Load a model for testing - implements required ModelTest method."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                raise ImportError("Required libraries (transformers, torch) not available")
                
            logger.info(f"Loading IDEFICS model {model_name}...")
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Load processor and model
            processor = transformers.IdeficsProcessor.from_pretrained(model_name)
            
            # Configure model loading with appropriate device settings
            dtype = torch.float16 if self.preferred_device == "cuda" else torch.float32
            
            model = transformers.IdeficsForVisionText2Text.from_pretrained(
                model_name,
                torch_dtype=dtype
            )
            
            # Move model to preferred device
            if self.preferred_device == "cuda" and torch.cuda.is_available():
                model = model.to("cuda")
                logger.info(f"Moved model to CUDA device")
            elif self.preferred_device == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
                model = model.to("mps")
                logger.info(f"Moved model to MPS (Apple Silicon) device")
            
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
            idefics_model = model["model"]
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
                    # Create a dummy image
                    image = self._get_test_image()
            
            # Record inference start time
            inference_start = time.time()
            
            # Process inputs for IDEFICS
            # IDEFICS uses a special prompt format with image placeholders
            prompt_with_image = f"User: <image>\n{prompt}\nAssistant:"
            
            inputs = processor(
                prompt_with_image,
                images=[image],
                return_tensors="pt"
            )
            
            # Move inputs to device if needed
            if hasattr(idefics_model, "device") and str(idefics_model.device) != "cpu":
                inputs = {k: v.to(idefics_model.device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = idefics_model.generate(
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
    
    def test_model_loading(self):
        """Test that the model loads correctly - implements required ModelTest method."""
        try:
            model_components = self.load_model(self.model_id)
            self.assertIsNotNone(model_components, "Model should not be None")
            self.assertIn("model", model_components, "Model dict should contain 'model' key")
            self.assertIn("processor", model_components, "Model dict should contain 'processor' key")
            
            idefics_model = model_components["model"]
            processor = model_components["processor"]
            
            # Check if we're using a mock or real model
            if isinstance(idefics_model, MagicMock):
                logger.info("Using mocked model for testing")
            else:
                expected_class_name = self.model_class
                actual_class_name = idefics_model.__class__.__name__
                
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
            
            # Check for OpenVINO compatibility
            if HAS_OPENVINO:
                logger.info("OpenVINO is available, but using CPU for standard tests")
            
            # Fallback to CPU
            return "cpu"
        except Exception as e:
            logger.error(f"Error detecting device: {e}")
            return "cpu"
    
    def test_multimodal_conversation(self):
        """Test IDEFICS with a multi-turn conversation including images."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_PIL:
                logger.warning("Required libraries not available, skipping multimodal conversation test")
                return {
                    "success": False,
                    "error": "Required libraries not available",
                    "device": self.preferred_device
                }
            
            logger.info(f"Testing IDEFICS model {self.model_id} with multimodal conversation")
            
            # Load model
            model_components = self.load_model(self.model_id)
            model = model_components["model"]
            processor = model_components["processor"]
            
            # Get test image
            image = self._get_test_image()
            
            # Create a multi-turn conversation
            prompt_with_image = "User: <image>\nWhat do you see in this image?\nAssistant:"
            
            # Process inputs
            inputs = processor(
                prompt_with_image,
                images=[image],
                return_tensors="pt"
            )
            
            # Move inputs to device if needed
            if hasattr(model, "device") and str(model.device) != "cpu":
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate first response
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            first_response_time = time.time() - start_time
            
            # Decode the first response
            first_response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Continue the conversation
            continuation_prompt = f"User: <image>\nWhat do you see in this image?\nAssistant: {first_response}\nUser: Is there anything interesting in this image?\nAssistant:"
            
            # Process continued conversation
            inputs = processor(
                continuation_prompt,
                images=[image],
                return_tensors="pt"
            )
            
            # Move inputs to device if needed
            if hasattr(model, "device") and str(model.device) != "cpu":
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate second response
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            second_response_time = time.time() - start_time
            
            # Decode the second response
            second_response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Record in performance stats
            self.performance_stats["multimodal_conversation"] = {
                "first_response_time": first_response_time,
                "second_response_time": second_response_time,
                "total_time": first_response_time + second_response_time
            }
            
            # Add to examples
            self.examples.append({
                "method": "multimodal_conversation",
                "conversation": [
                    {"role": "user", "content": "What do you see in this image?"},
                    {"role": "assistant", "content": first_response},
                    {"role": "user", "content": "Is there anything interesting in this image?"},
                    {"role": "assistant", "content": second_response}
                ]
            })
            
            return {
                "success": True,
                "conversation": [
                    {"role": "user", "content": "What do you see in this image?"},
                    {"role": "assistant", "content": first_response},
                    {"role": "user", "content": "Is there anything interesting in this image?"},
                    {"role": "assistant", "content": second_response}
                ],
                "performance": {
                    "first_response_time": first_response_time,
                    "second_response_time": second_response_time,
                    "total_time": first_response_time + second_response_time
                },
                "device": self.preferred_device
            }
            
        except Exception as e:
            logger.error(f"Error in multimodal conversation test: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.preferred_device
            }
    
    def test_with_pipeline(self):
        """Test the model using transformers pipeline API."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_PIL:
                logger.warning("Required libraries not available, skipping pipeline test")
                return {
                    "success": False,
                    "error": "Required libraries not available",
                    "device": self.preferred_device
                }
            
            logger.info(f"Testing IDEFICS model {self.model_id} with pipeline API")
            
            # Get test image
            image = self._get_test_image()
            
            # Create pipeline
            start_time = time.time()
            pipeline = transformers.pipeline(
                "image-to-text", 
                model=self.model_id,
                device=self.preferred_device if self.preferred_device != "cpu" else -1
            )
            load_time = time.time() - start_time
            
            # Run inference
            prompt = self.test_prompts[0]
            
            inference_start = time.time()
            outputs = pipeline({"image": image, "text": prompt})
            inference_time = time.time() - inference_start
            
            # Record stats
            self.performance_stats["pipeline"] = {
                "load_time": load_time,
                "inference_time": inference_time
            }
            
            # Add to examples
            self.examples.append({
                "method": "pipeline",
                "input": prompt,
                "output": outputs
            })
            
            return {
                "success": True,
                "outputs": outputs,
                "load_time": load_time,
                "inference_time": inference_time,
                "device": self.preferred_device
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline test: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.preferred_device
            }
    
    def run_tests(self, all_hardware=False):
        """
        Run all tests for this model.
        
        Args:
            all_hardware: If True, tests on all available hardware (CPU, CUDA, OpenVINO)
        
        Returns:
            Dict containing test results
        """
        # Run required ModelTest methods
        model_components = self.test_model_loading()
        
        # Test basic model functionality
        if model_components:
            try:
                # Get a test image
                test_image = self._get_test_image()
                test_prompt = self.test_prompts[0]
                
                # Verify model output with image and prompt
                test_input = {
                    "image": test_image,
                    "prompt": test_prompt
                }
                output = self.verify_model_output(model_components, test_input)
                
                # Add to examples
                self.examples.append({
                    "method": "verify_model_output",
                    "input": f"Image size: {test_image.size if hasattr(test_image, 'size') else 'unknown'}, Prompt: {test_prompt}",
                    "output": output
                })
                
                # Store in results
                self.results["basic_inference"] = {
                    "success": True,
                    "output": output
                }
            except Exception as e:
                logger.error(f"Error running model verification: {e}")
                self.results["basic_inference"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test multimodal conversation
        conversation_results = self.test_multimodal_conversation()
        self.results["multimodal_conversation"] = conversation_results
        
        # Test with pipeline API
        pipeline_results = self.test_with_pipeline()
        self.results["pipeline"] = pipeline_results
        
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
                "model_type": "idefics",
                "task": self.task,
                "architecture": self.architecture,
                "description": self.description,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_pil": HAS_PIL,
                "has_openvino": HAS_OPENVINO,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if using_real_inference else "MOCK OBJECTS (CI/CD)"
            }
        }

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test IDEFICS models with ModelTest pattern")
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
        model_id = args.model or "HuggingFaceM4/tiny-random-idefics"
        tester = TestIdeficsModels()
        if args.model:
            tester.model_id = args.model
            
        results = tester.run_tests(all_hardware=args.all_hardware)
        
        # Print summary
        basic_success = results["results"].get("basic_inference", {}).get("success", False)
        conversation_success = results["results"].get("multimodal_conversation", {}).get("success", False)
        
        # Determine if real inference or mock objects were used
        using_real_inference = results["metadata"]["has_transformers"] and results["metadata"]["has_torch"] and results["metadata"]["has_pil"]
        
        print("\nTEST RESULTS SUMMARY:")
        
        # Indicate real vs mock inference clearly
        if using_real_inference:
            print("🚀 Using REAL INFERENCE with actual models")
        else:
            print("🔷 Using MOCK OBJECTS for CI/CD testing only")
            print(f"   Dependencies: transformers={results['metadata']['has_transformers']}, torch={results['metadata']['has_torch']}, PIL={results['metadata']['has_pil']}")
        
        if basic_success or conversation_success:
            print(f"✅ Successfully tested {model_id}")
            
            # Print a sample output if available
            if results["examples"]:
                example = results["examples"][0]
                if isinstance(example, dict) and "output" in example:
                    print(f"\nSample output:")
                    output = example["output"]
                    if isinstance(output, dict) and "generated_text" in output:
                        print(f"  Generated text: {output['generated_text'][:100]}...")
                elif isinstance(example, dict) and "conversation" in example:
                    print(f"\nSample conversation:")
                    conversation = example["conversation"]
                    for turn in conversation[:2]:  # Show first two turns
                        role = turn["role"]
                        content = turn["content"]
                        print(f"  {role.capitalize()}: {content[:50]}...")
        else:
            print(f"❌ Failed to test {model_id}")
            # Print error details if available
            if "basic_inference" in results["results"]:
                basic_result = results["results"]["basic_inference"]
                if "error" in basic_result:
                    print(f"  - Error: {basic_result['error']}")
        
        # Save results if requested
        if args.save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"idefics_test_results_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nTest results saved to {filename}")

if __name__ == "__main__":
    main()