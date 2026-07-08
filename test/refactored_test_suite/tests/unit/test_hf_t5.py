#!/usr/bin/env python3

"""Migrated to refactored test suite on 2025-03-21

"""

import os
import sys
import json
import time
import datetime
import logging
import argparse
import traceback
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional
from refactored_test_suite.base_test import BaseTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

# Import ModelTest
from refactored_test_suite.model_test import ModelTest

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
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
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

# Models registry
T5_MODELS_REGISTRY = {
    "t5-small": {
        "description": "t5 base model",
        "class": "T5ForConditionalGeneration",
    }
}

class TestT5Models(ModelTest):
    """Test class for t5 models."""
    
    def __init__(self, model_id=None):
        """Initialize the test class."""
        self.model_id = model_id or "t5-small"
        
        # Use registry information
        if self.model_id not in T5_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default")
            self.model_info = T5_MODELS_REGISTRY["t5-small"]
        else:
            self.model_info = T5_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "translation_en_to_fr"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "translate English to French: Hello, how are you?"

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
            # For text models
            pipeline_input = self.test_text

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
            "hardware": HW_CAPABILITIES,
            "metadata": {
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        }


    def setUp(self):
        # Set up resources for each test method
        super().setUp()
        self.model_id = "t5-base"
    
        # Configure hardware preference
        self.preferred_device = self.detect_preferred_device()



    def test_model_loading(self):
        # Test basic model loading
        if not hasattr(self, 'model_id') or not self.model_id:
            self.skipTest("No model_id specified")
        
        try:
            # Import the appropriate library
            if 'bert' in self.model_id.lower() or 'gpt' in self.model_id.lower() or 't5' in self.model_id.lower():
                import transformers
                model = transformers.AutoModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'clip' in self.model_id.lower():
                import transformers
                model = transformers.CLIPModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'whisper' in self.model_id.lower():
                import transformers
                model = transformers.WhisperModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'wav2vec2' in self.model_id.lower():
                import transformers
                model = transformers.Wav2Vec2Model.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            else:
                # Generic loading
                try:
                    import transformers
                    model = transformers.AutoModel.from_pretrained(self.model_id)
                    self.assertIsNotNone(model, "Model loading failed")
                except:
                    self.skipTest(f"Could not load model {self.model_id} with AutoModel")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")



    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device."""
        try:
            import torch
        
            # Check for CUDA
            if torch.cuda.is_available():
                return "cuda"
        
            # Check for MPS (Apple Silicon)
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                return "mps"
        
            # Fallback to CPU
            return "cpu"
        except ImportError:
            return "cpu"
            
    def load_model(self, model_name=None):
        """Load a model for testing."""
        model_name = model_name or self.model_id
        
        try:
            import transformers
            
            # Load tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            if 't5' in model_name.lower():
                model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
            else:
                model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
            # Move model to appropriate device
            device = self.detect_preferred_device()
            if device != "cpu":
                model = model.to(device)
                
            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": device
            }
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
            
    def verify_model_output(self, model_components, input_data, expected_output=None):
        """Verify that model produces expected output."""
        try:
            model = model_components["model"]
            tokenizer = model_components["tokenizer"]
            device = model_components.get("device", "cpu")
            
            # Prepare input
            inputs = tokenizer(input_data, return_tensors="pt")
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
            # Generate output
            import torch
            with torch.no_grad():
                outputs = model.generate(**inputs)
                
            # Decode output
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verify output
            self.assertIsNotNone(decoded_output)
            self.assertTrue(len(decoded_output) > 0)
            
            # If expected output provided, compare
            if expected_output is not None:
                self.assertEqual(expected_output, decoded_output)
                
            return decoded_output
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            raise


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test t5 models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("CPU-only mode enabled")
    
    # Run test
    model_id = args.model or "t5-small"
    tester = TestT5Models(model_id)
    results = tester.run_tests()
    
    # Print summary
    success = any(r.get("pipeline_success", False) for r in results["results"].values())
    
    print("\nTEST RESULTS SUMMARY:")
    if success:
        print(f"✅ Successfully tested {model_id}")
    else:
        print(f"❌ Failed to test {model_id}")
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {test_name}: {result.get('pipeline_error', 'Unknown error')}")

if __name__ == "__main__":
    main()
