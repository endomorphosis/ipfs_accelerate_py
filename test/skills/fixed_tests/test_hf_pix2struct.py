#!/usr/bin/env python3


"""
Test file for Pix2Struct models.
This file tests the Pix2Struct model type from HuggingFace Transformers.
"""

import os
import sys
import json
import time
import logging
import argparse
from unittest.mock import MagicMock
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if dependencies are available
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

# Registry for Pix2Struct models
PIX2STRUCT_MODELS_REGISTRY = {
    "google/pix2struct-base": {
        "full_name": "Pix2Struct Base",
        "architecture": "multimodal",
        "description": "Pix2Struct model for image-to-text",
        "model_type": "pix2struct",
        "parameters": "250M",
        "context_length": 512,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "recommended_tasks": ["image-to-text"]
    }
}

def select_device():
    """Select the best available device for inference."""
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

class TestPix2StructModels:
    """
    Test class for Pix2Struct models.
    """
    
    def __init__(self, model_id="google/pix2struct-base", device=None):
        """Initialize the test class for Pix2Struct models.
        
        Args:
            model_id: The model ID to test (default: "google/pix2struct-base")
            device: The device to run tests on (default: None = auto-select)
        """
        self.model_id = model_id
        self.device = device if device else select_device()
        self.performance_stats = {}
    
    def test_pipeline(self):
        """Test the model using the pipeline API."""
        try:
            if not HAS_TRANSFORMERS:
                logger.warning("Transformers library not available, skipping pipeline test")
                return {"success": False, "error": "Transformers library not available"}
                
            logger.info(f"Testing Pix2Struct model {self.model_id} with pipeline API on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Initialize the pipeline with the appropriate task
            pipe = transformers.pipeline(
                "image-to-text", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Test with a task-appropriate input
            test_input = "An image for image-to-text."
            
            # Record inference start time
            inference_start = time.time()
            
            # Run inference
            outputs = pipe(test_input)
            
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Store performance stats
            self.performance_stats["pipeline"] = {
                "load_time": load_time,
                "inference_time": inference_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "inference_time": inference_time
            }
        except Exception as e:
            logger.error(f"Error testing pipeline: {e}")
            return {"success": False, "error": str(e)}
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        results = {}
        
        # Run pipeline test
        pipeline_result = self.test_pipeline()
        results["pipeline"] = pipeline_result
        
        # Add metadata
        results["metadata"] = {
            "model_id": self.model_id,
            "device": self.device,
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH
        }
        
        return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test Pix2Struct HuggingFace models")
    parser.add_argument("--model", type=str, default="google/pix2struct-base", help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to run tests on (cuda, cpu)")
    
    args = parser.parse_args()
    
    # Initialize the test class
    pix2struct_tester = TestPix2StructModels(model_id=args.model, device=args.device)
    
    # Run the tests
    results = pix2struct_tester.run_tests()
    
    # Print a summary
    success = results["pipeline"].get("success", False)
    
    print("\nTEST RESULTS SUMMARY:")
    
    if success:
        print(f"  Successfully tested {args.model}")
        print(f"  - Device: {pix2struct_tester.device}")
        print(f"  - Inference time: {results['pipeline'].get('inference_time', 'N/A'):.4f}s")
    else:
        print(f"  Failed to test {args.model}")
        print(f"  - Error: {results['pipeline'].get('error', 'Unknown error')}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
