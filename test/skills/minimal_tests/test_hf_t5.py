#!/usr/bin/env python3
"""
Minimal test file for HuggingFace model.
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

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Hardware detection
def check_hardware():
    """Check available hardware and return capabilities."""
    capabilities = {
        "cpu": True,
        "cuda": False,
        "mps": False
    }
    
    # Check CUDA
    if HAS_TORCH:
        capabilities["cuda"] = torch.cuda.is_available()
        
    # Check MPS (Apple Silicon)
    if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        capabilities["mps"] = torch.mps.is_available()
        
    return capabilities

HW_CAPABILITIES = check_hardware()

# Model registry
T5_MODELS_REGISTRY = {
    "t5-small": {
        "description": "t5 base model",
        "class": "T5ForConditionalGeneration",
    },
}

class TestT5Models:
    """Test class for t5-family models."""
    
    def __init__(self, model_id=None):
        """Initialize the test class for a specific model or default."""
        self.model_id = model_id or "t5-small"
        
        # Verify model exists in registry
        if self.model_id not in T5_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default")
            self.model_info = T5_MODELS_REGISTRY["t5-small"]
        else:
            self.model_info = T5_MODELS_REGISTRY[self.model_id]
            
        # Define model parameters
        self.task = "translation_en_to_fr"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Text input
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
            pipeline_input = self.test_text
            
            # Run inference passes
            num_runs = 1
            times = []
            outputs = []
            
            for _ in range(num_runs):
                start_time = time.time()
                output = pipeline(pipeline_input)
                end_time = time.time()
                times.append(end_time - start_time)
                outputs.append(output)
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            
            # Store results
            results["pipeline_success"] = True
            results["pipeline_avg_time"] = avg_time
            results["pipeline_load_time"] = load_time
            
            # Add to examples
            self.examples.append({
                "method": f"pipeline() on {device}",
                "input": str(pipeline_input),
                "output_preview": str(outputs[0])[:200] if len(str(outputs[0])) > 200 else str(outputs[0])
            })
            
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            logger.error(f"Error testing pipeline on {device}: {e}")
        
        # Add to overall results
        self.results[f"pipeline_{device}"] = results
        return results
    
    def run_tests(self):
        """Run all tests for this model."""
        # Run test
        self.test_pipeline()
        
        # Return results
        return {
            "results": self.results,
            "examples": self.examples,
            "metadata": {
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name
            }
        }


def save_results(model_id, results, output_dir="results"):
    """Save test results to a file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    filename = f"hf_t5_{safe_model_id}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    return output_path


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test t5-family models")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output files")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Test single model (default or specified)
    model_id = args.model or "t5-small"
    logger.info(f"Testing model: {model_id}")
    
    # Run test
    tester = TestT5Models(model_id)
    results = tester.run_tests()
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = any(r.get("pipeline_success", False) for r in results["results"].values())
    
    print("\nTEST RESULTS SUMMARY:")
    if success:
        print(f"✅ Successfully tested {model_id}")
    else:
        print(f"❌ Failed to test {model_id}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())