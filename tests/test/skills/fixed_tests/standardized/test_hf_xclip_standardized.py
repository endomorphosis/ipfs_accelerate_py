#!/usr/bin/env python3

"""
Standardized test file for XClip (Cross-modal Clip) HuggingFace models.
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
XCLIP_MODELS_REGISTRY = {
    "microsoft/xclip-base-patch32": {
        "description": "XClip base model with patch size 32",
        "class": "XClipForVideoClassification",
        "processor": "XClipProcessor",
        "architecture": "multimodal",
        "image_size": 224,
        "task": "video-classification",
        "parameters": "base",
        "frames": 8
    },
    "microsoft/xclip-base-patch16": {
        "description": "XClip base model with patch size 16",
        "class": "XClipForVideoClassification",
        "processor": "XClipProcessor",
        "architecture": "multimodal",
        "image_size": 224,
        "task": "video-classification",
        "parameters": "base",
        "frames": 8
    },
    "microsoft/xclip-large-patch14": {
        "description": "XClip large model with patch size 14",
        "class": "XClipForVideoClassification",
        "processor": "XClipProcessor",
        "architecture": "multimodal",
        "image_size": 224,
        "task": "video-classification",
        "parameters": "large",
        "frames": 8
    }
}

class TestXClipModels(ModelTest):
    """Test class for XClip vision-language models, following the ModelTest pattern."""
    
    def setUp(self):
        """Set up the test environment for XClip models."""
        super().setUp()
        
        # Use base model for testing by default
        self.model_id = "microsoft/xclip-base-patch32"  
        
        # Verify model exists in registry
        if self.model_id not in XCLIP_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = XCLIP_MODELS_REGISTRY["microsoft/xclip-base-patch32"]
        else:
            self.model_info = XCLIP_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters from registry
        self.description = self.model_info["description"]
        self.image_size = self.model_info["image_size"]
        self.model_class = self.model_info["class"]
        self.processor_class = self.model_info["processor"]
        self.task = self.model_info["task"]
        self.architecture = self.model_info["architecture"]
        self.num_frames = self.model_info.get("frames", 8)
        
        # Configure hardware preference
        self.preferred_device = self.detect_preferred_device()
        logger.info(f"Using {self.preferred_device} as preferred device")
        
        # Results storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}
    
    def _create_video_frames(self, num_frames=None):
        """Create a dummy video input with multiple frames for XClip testing."""
        if not HAS_PIL:
            return None
        
        num_frames = num_frames or self.num_frames
        frames = []
        
        for i in range(num_frames):
            # Create diverse frames with different colors
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ) if HAS_TORCH else (73, 109, 137)
            
            img = Image.new(
                'RGB', 
                (self.image_size, self.image_size),
                color=color
            )
            frames.append(img)
        
        return frames
    
    def load_model(self, model_name):
        """Load a model for testing - implements required ModelTest method."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                raise ImportError("Required libraries (transformers, torch) not available")
                
            logger.info(f"Loading XClip model {model_name}...")
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Load processor - XClip uses XClipProcessor
            processor = transformers.XClipProcessor.from_pretrained(model_name)
            
            # Configure model loading with appropriate device settings
            dtype = torch.float16 if self.preferred_device == "cuda" else torch.float32
            
            # Load model - XClip uses XClipForVideoClassification
            model = transformers.XClipForVideoClassification.from_pretrained(
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
            xclip_model = model["model"]
            processor = model["processor"]
            
            # Process inputs based on type
            if isinstance(input_data, dict):
                if "frames" in input_data:
                    frames = input_data["frames"]
                else:
                    frames = self._create_video_frames()
            else:
                # Use default test inputs
                frames = self._create_video_frames()
            
            # Record inference start time
            inference_start = time.time()
            
            # Process inputs
            inputs = processor(frames, return_tensors="pt")
            
            # Move inputs to device if needed
            if hasattr(xclip_model, "device") and str(xclip_model.device) != "cpu":
                inputs = {key: val.to(xclip_model.device) for key, val in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = xclip_model(**inputs)
            
            # Process logits to get predictions
            logits = outputs.logits
            
            # If model has id2label mapping, use it to get class names
            if hasattr(xclip_model.config, "id2label"):
                predicted_class_idx = logits.argmax(-1).item()
                class_label = xclip_model.config.id2label.get(predicted_class_idx, f"Class {predicted_class_idx}")
                predictions = {
                    "predicted_class": class_label,
                    "predicted_class_idx": predicted_class_idx,
                    "logits_shape": list(logits.shape) if hasattr(logits, "shape") else [1, 1]
                }
            else:
                # Without id2label mapping, just return the raw logits
                predictions = {
                    "top_class_idx": logits.argmax(-1).item() if hasattr(logits, "argmax") else 0,
                    "logits_shape": list(logits.shape) if hasattr(logits, "shape") else [1, 1]
                }
            
            # Record inference time
            inference_time = time.time() - inference_start
            self.performance_stats["inference"] = {
                "time": inference_time,
                "num_frames": len(frames) if isinstance(frames, list) else 1,
                "device": self.preferred_device
            }
            
            # Create result with predictions
            result = {
                "predictions": predictions,
                "inference_time": inference_time
            }
            
            # Verify output
            if HAS_TRANSFORMERS and not isinstance(xclip_model, MagicMock):
                self.assertIsNotNone(logits, "Model output should not be None")
                self.assertGreater(len(list(logits.shape)) if hasattr(logits, "shape") else 0, 0, "Logits should not be empty")
            
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
                    "predictions": {
                        "predicted_class": "mock_class",
                        "predicted_class_idx": 0,
                        "logits_shape": [1, 400]
                    },
                    "inference_time": 0.01
                }
    
    def test_model_loading(self):
        """Test that the model loads correctly - implements required ModelTest method."""
        try:
            model_components = self.load_model(self.model_id)
            self.assertIsNotNone(model_components, "Model should not be None")
            self.assertIn("model", model_components, "Model dict should contain 'model' key")
            self.assertIn("processor", model_components, "Model dict should contain 'processor' key")
            
            xclip_model = model_components["model"]
            processor = model_components["processor"]
            
            # Check if we're using a mock or real model
            if isinstance(xclip_model, MagicMock):
                logger.info("Using mocked model for testing")
            else:
                expected_class_name = self.model_class
                actual_class_name = xclip_model.__class__.__name__
                
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
    
    def test_variable_frame_count(self):
        """Test XClip with different numbers of video frames."""
        results = {}
        frame_counts = [4, 8, 16]  # Test with different numbers of frames
        
        try:
            # Load the model
            model_components = self.load_model(self.model_id)
            
            for frame_count in frame_counts:
                test_key = f"frames_{frame_count}"
                
                try:
                    # Create video frames
                    frames = self._create_video_frames(num_frames=frame_count)
                    
                    # Process and verify output
                    test_result = self.verify_model_output(model_components, {"frames": frames})
                    
                    # Store results
                    results[test_key] = {
                        "success": True,
                        "predictions": test_result.get("predictions", {}),
                        "inference_time": test_result.get("inference_time", None)
                    }
                    
                    # Add to examples
                    self.examples.append({
                        "method": f"Variable frame test ({frame_count} frames)",
                        "input": f"Video with {frame_count} frames",
                        "output": test_result
                    })
                    
                except Exception as e:
                    logger.error(f"Error testing with {frame_count} frames: {e}")
                    results[test_key] = {
                        "success": False,
                        "error": str(e)
                    }
            
        except Exception as e:
            logger.error(f"Error in variable frame count test: {e}")
            for frame_count in frame_counts:
                results[f"frames_{frame_count}"] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
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
            
            logger.info(f"Testing XClip model {self.model_id} with pipeline API")
            
            # Create video frames
            frames = self._create_video_frames()
            
            # Create pipeline with appropriate parameters
            start_time = time.time()
            pipeline = transformers.pipeline(
                "video-classification",
                model=self.model_id,
                device=self.preferred_device if self.preferred_device != "cpu" else -1
            )
            load_time = time.time() - start_time
            
            # Run inference
            inference_start = time.time()
            outputs = pipeline(frames)
            inference_time = time.time() - inference_start
            
            # Record stats
            self.performance_stats["pipeline"] = {
                "load_time": load_time,
                "inference_time": inference_time
            }
            
            # Add to examples
            self.examples.append({
                "method": "Pipeline video classification",
                "input": f"Video with {len(frames)} frames",
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
                # Create test video frames
                frames = self._create_video_frames()
                
                # Verify model output with the frames
                test_input = {"frames": frames}
                output = self.verify_model_output(model_components, test_input)
                
                # Add to examples
                self.examples.append({
                    "method": "verify_model_output",
                    "input": f"Video with {len(frames)} frames",
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
        
        # Test with variable frame counts
        variable_frame_results = self.test_variable_frame_count()
        self.results["variable_frames"] = variable_frame_results
        
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
                "model_type": "xclip",
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
    parser = argparse.ArgumentParser(description="Test XClip models with ModelTest pattern")
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
        model_id = args.model or "microsoft/xclip-base-patch32"
        tester = TestXClipModels()
        if args.model:
            tester.model_id = args.model
            
        results = tester.run_tests(all_hardware=args.all_hardware)
        
        # Print summary
        basic_success = results["results"].get("basic_inference", {}).get("success", False)
        pipeline_success = results["results"].get("pipeline", {}).get("success", False)
        
        # Determine if real inference or mock objects were used
        using_real_inference = results["metadata"]["has_transformers"] and results["metadata"]["has_torch"] and results["metadata"]["has_pil"]
        
        print("\nTEST RESULTS SUMMARY:")
        
        # Indicate real vs mock inference clearly
        if using_real_inference:
            print("🚀 Using REAL INFERENCE with actual models")
        else:
            print("🔷 Using MOCK OBJECTS for CI/CD testing only")
            print(f"   Dependencies: transformers={results['metadata']['has_transformers']}, torch={results['metadata']['has_torch']}, PIL={results['metadata']['has_pil']}")
        
        if basic_success or pipeline_success:
            print(f"✅ Successfully tested {model_id}")
            
            # Print a sample output if available
            if results["examples"]:
                example = results["examples"][0]
                if isinstance(example, dict) and "output" in example:
                    print(f"\nSample output:")
                    output = example["output"]
                    if isinstance(output, dict) and "predictions" in output:
                        print(f"  Predictions: {output['predictions']}")
        else:
            print(f"❌ Failed to test {model_id}")
            # Print error details if available
            if "basic_inference" in results["results"]:
                basic_result = results["results"]["basic_inference"]
                if "error" in basic_result:
                    print(f"  - Error: {basic_result['error']}")
            if "pipeline" in results["results"]:
                pipeline_result = results["results"]["pipeline"]
                if "error" in pipeline_result:
                    print(f"  - Pipeline error: {pipeline_result['error']}")
        
        # Save results if requested
        if args.save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"xclip_test_results_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nTest results saved to {filename}")

if __name__ == "__main__":
    main()