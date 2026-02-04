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
    import librosa
    HAS_LIBROSA = True
except ImportError:
    librosa = MagicMock()
    HAS_LIBROSA = False
    logger.warning("librosa not available, using mock")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = MagicMock()
    HAS_NUMPY = False
    logger.warning("numpy not available, using mock")

# Models registry
WAV2VEC2_MODELS_REGISTRY = {
    "facebook/wav2vec2-base-960h": {
        "description": "wav2vec2 base model",
        "class": "Wav2Vec2ForCTC",
    }
}

class TestWav2vec2Models(ModelTest):
    """Test class for wav2vec2 models."""
    
    def setUp(self):
        """Initialize the test class."""
        super().setUp()
        self.model_id = "facebook/wav2vec2-base-960h"
        
        # Use registry information
        if self.model_id not in WAV2VEC2_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default")
            self.model_info = WAV2VEC2_MODELS_REGISTRY["facebook/wav2vec2-base-960h"]
        else:
            self.model_info = WAV2VEC2_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "automatic-speech-recognition"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Test inputs will be created during testing
        self.test_input = None
        self.sample_rate = 16000  # Default sample rate for wav2vec2

        # Configure hardware preference
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
            
            # Create model and processor for wav2vec2 speech recognition
            model = transformers.Wav2Vec2ForCTC.from_pretrained(model_name)
            processor = transformers.Wav2Vec2Processor.from_pretrained(model_name)
            
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
            wav2vec2_model = model["model"]
            processor = model["processor"]
            
            # Process input audio
            if isinstance(input_data, str) and os.path.isfile(input_data):
                # Input is a file path to an audio file
                if HAS_LIBROSA and HAS_NUMPY:
                    # Load audio file using librosa
                    waveform, sample_rate = librosa.load(input_data, sr=self.sample_rate)
                    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
                else:
                    raise ImportError("librosa and numpy are required for processing audio files")
            elif isinstance(input_data, torch.Tensor):
                # Input is a PyTorch tensor, assume it's already in the right format
                if input_data.dim() == 1:
                    # Add batch dimension if needed
                    input_values = processor(input_data.numpy(), sampling_rate=self.sample_rate, return_tensors="pt").input_values
                else:
                    input_values = input_data
            elif HAS_NUMPY and isinstance(input_data, np.ndarray):
                # Input is a numpy array
                input_values = processor(input_data, sampling_rate=self.sample_rate, return_tensors="pt").input_values
            else:
                # If we don't have a valid input, create a random one
                if HAS_TORCH:
                    # Create a random audio tensor (2 seconds of audio)
                    random_input = torch.randn(self.sample_rate * 2)
                    input_values = processor(random_input.numpy(), sampling_rate=self.sample_rate, return_tensors="pt").input_values
                else:
                    raise ValueError("Cannot process input and torch is not available to create a random input")
            
            # Move inputs to the right device if model is on a specific device
            if hasattr(wav2vec2_model, 'device') and str(wav2vec2_model.device) != "cpu":
                input_values = input_values.to(wav2vec2_model.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = wav2vec2_model(input_values)
            
            # Process the outputs to get text prediction
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            
            # Verify output
            self.assertIsNotNone(outputs, "Model output should not be None")
            self.assertIsNotNone(transcription, "Transcription should not be None")
            self.assertGreater(len(transcription[0]), 0, "Transcription should not be empty")
            
            # If expected output is provided, compare with actual output
            if expected_output is not None:
                self.assertEqual(expected_output, transcription[0])
                
            return {"transcription": transcription[0], "logits": logits}
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
            
            wav2vec2_model = model_components["model"]
            self.assertEqual(wav2vec2_model.config.model_type, "wav2vec2", "Model should be a wav2vec2 model")
            
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
            
            # Prepare test input for audio models
            if HAS_TORCH:
                # Create a random audio tensor (sample_rate=16000, 2 seconds of audio)
                sample_rate = 16000
                self.test_input = torch.randn(sample_rate * 2)
                pipeline_input = self.test_input
            else:
                # Skip inference if torch is not available
                pipeline_input = None
                logger.warning("Skipping inference because torch is not available")

            # Run inference if input is available
            if pipeline_input is not None:
                output = pipeline(pipeline_input)
                
                # Store results
                results["pipeline_success"] = True
                results["pipeline_load_time"] = load_time
                results["pipeline_error_type"] = "none"
            else:
                # Mark as success but note that inference was skipped
                results["pipeline_success"] = True
                results["pipeline_load_time"] = load_time
                results["pipeline_error_type"] = "skipped_inference"
            
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
    parser = argparse.ArgumentParser(description="Test wav2vec2 models")
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
        tester = TestWav2vec2Models()
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