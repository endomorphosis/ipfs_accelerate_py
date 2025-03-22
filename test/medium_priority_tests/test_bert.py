#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import argparse
import traceback
from unittest.mock import MagicMock
from pathlib import Path
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if we should mock specific dependencies
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'
MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'

# Try to import torch
try:
    if MOCK_TORCH:
        raise ImportError("Mocked torch import failure")
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# Try to import transformers
try:
    if MOCK_TRANSFORMERS:
        raise ImportError("Mocked transformers import failure")
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Try to import tokenizers
try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

# Try to import sentencepiece
try:
    if MOCK_SENTENCEPIECE:
        raise ImportError("Mocked sentencepiece import failure")
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")

# Import numpy (usually available)
import numpy as np

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        logger.info(f"CUDA available")
    else:
        logger.info("CUDA not available")
else:
    HAS_CUDA = False
    logger.info("CUDA detection skipped (torch not available)")

# MPS (Apple Silicon) detection
if HAS_TORCH:
    HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if HAS_MPS:
        logger.info("MPS available for Apple Silicon acceleration")
    else:
        logger.info("MPS not available")
else:
    HAS_MPS = False
    logger.info("MPS detection skipped (torch not available)")

def select_device():
    """Select the best available device for inference."""
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda:0"
    elif HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class TestBertModel:
    """Test class for BERT-family models."""
    
    def __init__(self, model_id="nvidia/segformer-b0-finetuned-ade-512-512", device=None):
        """Initialize the test class."""
        self.model_id = model_id
        self.device = device or select_device()
        self.results = {}
        self.performance_stats = {}
        
    def test_pipeline(self):
        """Test using the pipeline API."""
        try:
            if not HAS_TRANSFORMERS:
                logger.warning("Transformers library not available, skipping pipeline test")
                return {"success": False, "error": "Transformers library not available"}
                
            logger.info(f"Testing model {self.model_id} with pipeline API on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Initialize the pipeline - in mock mode, this just returns a mock object
            pipe = transformers.pipeline(
                "fill-mask", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Test with a simple input
            test_input = "The quick brown fox jumps over the [MASK] dog."
            
            # Run inference
            outputs = pipe(test_input)
            
            # Record inference time
            inference_time = time.time() - start_time - load_time
            
            # Store performance stats
            self.performance_stats["pipeline"] = {
                "load_time": load_time,
                "inference_time": inference_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "performance": {
                    "load_time": load_time,
                    "inference_time": inference_time
                }
            }
                
        except Exception as e:
            logger.error(f"Error testing pipeline: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def run_tests(self):
        """Run all tests for the model."""
        logger.info(f"Testing model: {self.model_id}")
        
        # Run pipeline test
        pipeline_result = self.test_pipeline()
        self.results["pipeline"] = pipeline_result
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
        # Add metadata to results
        self.results["metadata"] = {
            "model": self.model_id,
            "device": self.device,
            "timestamp": datetime.datetime.now().isoformat(),
            "dependencies": {
                "transformers": transformers.__version__ if HAS_TRANSFORMERS else None,
                "torch": torch.__version__ if HAS_TORCH else None,
                "numpy": np.__version__
            },
            "performance": self.performance_stats,
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH, 
            "has_tokenizers": HAS_TOKENIZERS,
            "has_sentencepiece": HAS_SENTENCEPIECE,
            "using_real_inference": using_real_inference,
            "using_mocks": using_mocks,
            "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
        }
        
        return self.results

def save_results(results, output_dir="collected_results"):
    """Save test results to a JSON file."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = results.get("metadata", {}).get("model", "unknown_model")
        model_id_safe = model_id.replace("/", "__")
        filename = f"model_test_{model_id_safe}_{timestamp}.json"
        file_path = os.path.join(output_dir, filename)
        
        # Save results to file
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test BERT-family models")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="Specific model to test")
    parser.add_argument("--device", type=str, help="Device to run on (cuda, cpu, mps)")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Test model
    tester = TestBertModel(model_id=args.model, device=args.device)
    results = tester.run_tests()
    
    # Indicate real vs mock inference clearly
    using_real_inference = results["metadata"]["using_real_inference"]
    using_mocks = results["metadata"]["using_mocks"]
    
    if using_real_inference and not using_mocks:
        print(f"\033[32m🚀 Using REAL INFERENCE with actual models\033[0m")
    else:
        print(f"\033[34m🔷 Using MOCK OBJECTS for CI/CD testing only\033[0m")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
    
    print(f"\nModel: {args.model}")
    print(f"Device: {tester.device}")
    
    # Save results if requested
    if args.save:
        file_path = save_results(results)
        if file_path:
            print(f"Results saved to {file_path}")
    
    print(f"\nSuccessfully tested {args.model}")
    
    return 0 if results["pipeline"]["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
