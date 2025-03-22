#!/usr/bin/env python3

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
FALCON_MODELS_REGISTRY = {
    "falcon": {
        "description": "falcon base model",
        "class": "FalconForCausalLM",
    }
}

class TestFalconModels:
    """Test class for falcon models."""
    
    def __init__(self, model_id=None):
        """Initialize the test class."""
        self.model_id = model_id or "falcon"
        
        # Use registry information
        if self.model_id not in FALCON_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default")
            self.model_info = FALCON_MODELS_REGISTRY["falcon"]
        else:
            self.model_info = FALCON_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "text-generation"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "Once upon a time,"

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
            logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
            
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
            
            # Set padding token for decoder-only models
            if pipeline.tokenizer.pad_token is None:
                pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
            
            # Prepare test input
            pipeline_input = self.test_text
            
            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": 20,
                "do_sample": False
            }

            # Run inference
            output = pipeline(pipeline_input, **gen_kwargs)
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_load_time"] = load_time
            results["pipeline_error_type"] = "none"
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_error_type"] = "other"
            logger.error(f"Error testing pipeline: {{e}}")
        
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

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test falcon models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("CPU-only mode enabled")
    
    # Run test
    model_id = args.model or "falcon"
    tester = TestFalconModels(model_id)
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
