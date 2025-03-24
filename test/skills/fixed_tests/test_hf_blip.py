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
MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'

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

# Models registry - Maps model IDs to their specific configurations
BLIP_MODELS_REGISTRY = {
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
    "Salesforce/blip-vqa-capfilt-large": {
        "description": "BLIP visual question answering large model",
        "class": "BlipForQuestionAnswering",
        "type": "blip",
        "image_size": 384,
        "task": "visual-question-answering"
    }
}

class TestBlipModels(ModelTest):
    """Test class for BLIP vision-text models."""
    
    def setUp(self):
        """Initialize the test class for a specific model or default."""
        super().setUp()
        self.model_id = "Salesforce/blip-image-captioning-base"
        
        # Verify model exists in registry
        if self.model_id not in BLIP_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = BLIP_MODELS_REGISTRY["Salesforce/blip-image-captioning-base"]
        else:
            self.model_info = BLIP_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = self.model_info.get("task", "image-to-text")
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        self.image_size = self.model_info["image_size"]
        
        # Define test inputs
        self.test_image_path = self._find_test_image()
        if self.task == "visual-question-answering":
            self.test_text = "What is shown in the image?"
            self.test_texts = ["What is shown in the image?", "What can you see in this picture?"]
        
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
                
            logger.info(f"Loading model {model_name}...")
            
            # Load processor and model based on task
            processor = transformers.BlipProcessor.from_pretrained(model_name)
            
            if "vqa" in model_name or self.task == "visual-question-answering":
                model = transformers.BlipForQuestionAnswering.from_pretrained(model_name)
            else:
                model = transformers.BlipForConditionalGeneration.from_pretrained(model_name)
            
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
        """Verify that model produces expected output - implements required ModelTest method."""
        try:
            if not isinstance(model, dict) or "model" not in model or "processor" not in model:
                raise ValueError("Model should be a dict containing 'model' and 'processor' keys")
                
            # Unpack model components
            blip_model = model["model"]
            processor = model["processor"]
            
            # Process inputs based on type
            if isinstance(input_data, dict):
                if "image" in input_data:
                    image = input_data["image"]
                else:
                    image = self._get_test_image()
                    
                if "text" in input_data:
                    text = input_data["text"]
                elif self.task == "visual-question-answering" and hasattr(self, "test_text"):
                    text = self.test_text
                else:
                    text = None
            else:
                # Use default test inputs
                image = self._get_test_image()
                text = self.test_text if hasattr(self, "test_text") else None
            
            # Process inputs
            if isinstance(image, str):
                # Load image from path
                if os.path.exists(image):
                    image = Image.open(image)
                else:
                    # Create a random dummy image
                    image = Image.new('RGB', (self.image_size, self.image_size), color=(73, 109, 137))
            
            # Prepare inputs based on task
            if self.task == "visual-question-answering" and text is not None:
                inputs = processor(image, text, return_tensors="pt")
            else:
                inputs = processor(image, return_tensors="pt")
            
            # Move inputs to device if needed
            if hasattr(blip_model, "device") and str(blip_model.device) != "cpu":
                inputs = {k: v.to(blip_model.device) for k, v in inputs.items()}
            
            # Run inference based on task
            if self.task == "visual-question-answering":
                with torch.no_grad():
                    outputs = blip_model(**inputs)
                
                # Get logits and convert to most likely answer
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    predicted_answer_idx = logits.argmax(-1).item()
                    # In a real implementation, we'd map this to an answer
                    predicted_answer = f"Answer index: {predicted_answer_idx}"
                else:
                    predicted_answer = "Model output does not contain logits"
                
                result = {"answer": predicted_answer}
            else:
                # For image captioning
                with torch.no_grad():
                    generated_ids = blip_model.generate(**inputs, max_length=50)
                
                # Decode generated caption
                generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
                result = {"caption": generated_caption}
            
            # Verify output
            self.assertIsNotNone(result, "Model output should not be None")
            
            # For image captioning
            if "caption" in result:
                self.assertIsInstance(result["caption"], str, "Caption should be a string")
                self.assertGreater(len(result["caption"]), 0, "Caption should not be empty")
            
            # For VQA
            if "answer" in result:
                self.assertIsNotNone(result["answer"], "Answer should not be None")
            
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
            
            blip_model = model_components["model"]
            
            # For BLIP, check config values
            expected_class_name = self.class_name
            actual_class_name = blip_model.__class__.__name__
            
            self.assertEqual(expected_class_name, actual_class_name, 
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
            logger.info(f"Testing {self.model_id} with pipeline() on {device}...")
            
            # Create pipeline with appropriate parameters based on task
            if self.task == "visual-question-answering":
                pipeline_task = "visual-question-answering"
            else:
                pipeline_task = "image-to-text"
                
            # Time the model loading
            load_start_time = time.time()
            pipeline = transformers.pipeline(
                pipeline_task,
                model=self.model_id,
                device=device if device != "cpu" else -1
            )
            load_time = time.time() - load_start_time
            
            # Prepare test inputs
            if self.test_image_path and os.path.exists(self.test_image_path):
                test_image = self.test_image_path
            else:
                # Create a random dummy image
                dummy_path = "test_dummy_temp.jpg"
                img = Image.new('RGB', (self.image_size, self.image_size), color=(73, 109, 137))
                img.save(dummy_path)
                test_image = dummy_path
            
            # Prepare pipeline input based on task
            if self.task == "visual-question-answering":
                pipeline_input = {"image": test_image, "question": self.test_text}
            else:
                pipeline_input = test_image
            
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
            if not self.test_image_path and "dummy_path" in locals() and os.path.exists(dummy_path):
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
                "input": str(pipeline_input),
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
        
        # Always test on default device
        self.test_pipeline()
        
        # Test some sample input with the model
        if model_components:
            try:
                test_image = self._get_test_image()
                
                # Prepare test input based on task
                if self.task == "visual-question-answering":
                    test_input = {
                        "image": test_image,
                        "text": self.test_text
                    }
                else:
                    test_input = {
                        "image": test_image
                    }
                
                output = self.verify_model_output(model_components, test_input)
                self.examples.append({
                    "method": "verify_model_output",
                    "input": str(test_input),
                    "output": output
                })
            except Exception as e:
                logger.error(f"Error running model verification: {e}")
        
        # Test on all available hardware if requested
        if all_hardware:
            # Always test on CPU
            if self.preferred_device != "cpu":
                self.test_pipeline(device="cpu")
            
            # Test on CUDA if available
            if HAS_TORCH and torch.cuda.is_available() and self.preferred_device != "cuda":
                self.test_pipeline(device="cuda")
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_PIL
        
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
                "has_tokenizers": HAS_TOKENIZERS,
                "has_pil": HAS_PIL,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if using_real_inference and not using_mocks else "MOCK OBJECTS (CI/CD)"
            }
        }

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test BLIP models")
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
        model_id = args.model or "Salesforce/blip-image-captioning-base"
        tester = TestBlipModels()
        if args.model:
            tester.model_id = args.model
            
        results = tester.run_tests(all_hardware=args.all_hardware)
        
        # Print summary
        success = any(r.get("pipeline_success", False) for r in results["results"].values())
        
        # Determine if real inference or mock objects were used
        using_real_inference = results["metadata"]["has_transformers"] and results["metadata"]["has_torch"]
        using_mocks = not using_real_inference or not results["metadata"]["has_tokenizers"] or not results["metadata"]["has_pil"]
        
        print("\nTEST RESULTS SUMMARY:")
        
        # Indicate real vs mock inference clearly
        if using_real_inference and not using_mocks:
            print("🚀 Using REAL INFERENCE with actual models")
        else:
            print("🔷 Using MOCK OBJECTS for CI/CD testing only")
            print(f"   Dependencies: transformers={results['metadata']['has_transformers']}, torch={results['metadata']['has_torch']}, tokenizers={results['metadata']['has_tokenizers']}, PIL={results['metadata']['has_pil']}")
        
        if success:
            print(f"✅ Successfully tested {model_id}")
            
            # Print performance highlights
            for device, stats in results["performance"].items():
                if "avg_time" in stats:
                    print(f"  - {device}: {stats['avg_time']:.4f}s average inference time")
        else:
            print(f"❌ Failed to test {model_id}")
            for test_name, result in results["results"].items():
                if "pipeline_error" in result:
                    print(f"  - Error in {test_name}: {result.get('pipeline_error_type', 'unknown')}")
                    print(f"    {result.get('pipeline_error', 'Unknown error')}")

if __name__ == "__main__":
    main()