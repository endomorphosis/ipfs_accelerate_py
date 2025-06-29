#!/usr/bin/env python3


"""
Test file for GPTNeoX models.
This file tests the GPTNeoX model type from HuggingFace Transformers.
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

# Registry for GPTNeoX models
GPT_NEOX_MODELS_REGISTRY = {
    "EleutherAI/gpt-neox-20b": {
        "full_name": "GPTNeoX Base",
        "architecture": "decoder-only",
        "description": "GPTNeoX model for text generation",
        "model_type": "gpt-neox",
        "parameters": "20B",
        "context_length": 2048,
        "embedding_dim": 6144,
        "attention_heads": 48,
        "layers": 44,
        "recommended_tasks": ["text-generation"]
    }
}

def select_device():
    """Select the best available device for inference."""
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"

class TestGPTNeoXModels:
    """
    Test class for GPTNeoX models.
    """
    
    def __init__(self, model_id="EleutherAI/gpt-neox-20b", device=None):
        """Initialize the test class for GPTNeoX models.
        
        Args:
            model_id: The model ID to test (default: "EleutherAI/gpt-neox-20b")
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
                
            logger.info(f"Testing GPTNeoX model {self.model_id} with pipeline API on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Initialize the pipeline with the appropriate task
            pipe = transformers.pipeline("text-generation", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Test with a task-appropriate input
            test_input = "GPTNeoX is a transformer model that"
            
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
    
    def test_from_pretrained(self):
        """Test loading the model directly using from_pretrained."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning("Transformers or torch not available, skipping from_pretrained test")
                return {"success": False, "error": "Required libraries not available"}
                
            logger.info(f"Testing GPTNeoX model {self.model_id} with from_pretrained on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Load tokenizer and model
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id)
            
            if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available():
                model = model.to(self.device)
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Prepare input
            test_input = "GPTNeoX is a transformer model that"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Record inference start time
            inference_start = time.time()
            
            # Run inference
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=50)
                
            # Decode output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Store performance stats
            self.performance_stats["from_pretrained"] = {
                "load_time": load_time,
                "inference_time": inference_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "inference_time": inference_time,
                "output": output_text
            }
        except Exception as e:
            logger.error(f"Error in from_pretrained test: {e}")
            return {"success": False, "error": str(e)}
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        results = {}
        
        # Run pipeline test
        pipeline_result = self.test_pipeline()
        results["pipeline"] = pipeline_result
        
        # Run from_pretrained test
        from_pretrained_result = self.test_from_pretrained()
        results["from_pretrained"] = from_pretrained_result
        
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
    parser = argparse.ArgumentParser(description="Test GPTNeoX HuggingFace models")
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neox-20b", help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to run tests on (cuda, cpu)")
    
    args = parser.parse_args()
    
    # Initialize the test class
    gpt_neox_tester = TestGPTNeoXModels(model_id=args.model, device=args.device)
    
    # Run the tests
    results = gpt_neox_tester.run_tests()
    
    # Print a summary
    pipeline_success = results["pipeline"].get("success", False)
    pretrained_success = results["from_pretrained"].get("success", False)
    overall_success = pipeline_success and pretrained_success
    
    print("\nTEST RESULTS SUMMARY:")
    
    if overall_success:
        print(f"  Successfully tested {args.model}")
        print(f"  - Device: {gpt_neox_tester.device}")
        print(f"  - Pipeline inference time: {results['pipeline'].get('inference_time', 'N/A'):.4f}s")
        print(f"  - From_pretrained inference time: {results['from_pretrained'].get('inference_time', 'N/A'):.4f}s")
    else:
        print(f"  Failed to test {args.model}")
        if not pipeline_success:
            print(f"  - Pipeline error: {results['pipeline'].get('error', 'Unknown error')}")
        if not pretrained_success:
            print(f"  - From_pretrained error: {results['from_pretrained'].get('error', 'Unknown error')}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())
