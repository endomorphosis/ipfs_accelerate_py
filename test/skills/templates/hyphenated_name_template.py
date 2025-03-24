#!/usr/bin/env python3

"""
Template for models with hyphenated names like xlm-roberta.
This template is designed to work correctly with the test generator.
"""

import os
import sys
import json
import time
import datetime
import traceback
import logging
import argparse
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

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

try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")

# Simple registry for models
XLM_ROBERTA_MODELS_REGISTRY = {
    "xlm-roberta-base": {
        "full_name": "XLM-RoBERTa Base",
        "architecture": "encoder-only",
        "description": "XLM-RoBERTa Base model for cross-lingual understanding",
        "model_type": "xlm-roberta",
        "parameters": "270M",
        "context_length": 512,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "recommended_tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"]
    },
    "xlm-roberta-large": {
        "full_name": "XLM-RoBERTa Large",
        "architecture": "encoder-only",
        "description": "XLM-RoBERTa Large model for cross-lingual understanding",
        "model_type": "xlm-roberta",
        "parameters": "550M",
        "context_length": 512,
        "embedding_dim": 1024,
        "attention_heads": 16,
        "layers": 24,
        "recommended_tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"]
    }
}

def select_device():
    """Select the best available device for inference."""
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda:0"
    elif HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class TestXlmRobertaModels:
    """
    Test class for XLM-RoBERTa models.
    """
    
    def __init__(self, model_id="xlm-roberta-base", device=None):
        """Initialize the test class for XLM-RoBERTa models.
        
        Args:
            model_id: The model ID to test (default: "xlm-roberta-base")
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
                
            logger.info(f"Testing XLM-RoBERTa model {self.model_id} with pipeline API on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Initialize the pipeline
            pipe = transformers.pipeline(
                "fill-mask", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Test with a simple input
            test_input = "XLM-RoBERTa is a <mask> language model."
            
            # Record inference start time
            inference_start = time.time()
            
            # Run inference
            outputs = pipe(test_input)
            
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Log results
            if isinstance(outputs, list) and len(outputs) > 0:
                top_prediction = outputs[0]['token_str'] if isinstance(outputs[0], dict) and 'token_str' in outputs[0] else "N/A"
                logger.info(f"Top prediction: {top_prediction}")
                logger.info(f"Inference time: {inference_time:.2f} seconds")
                
                # Store performance stats
                self.performance_stats["pipeline"] = {
                    "load_time": load_time,
                    "inference_time": inference_time,
                    "top_prediction": top_prediction
                }
                
                return {
                    "success": True,
                    "model_id": self.model_id,
                    "device": self.device,
                    "input": test_input,
                    "top_prediction": top_prediction,
                    "performance": {
                        "load_time": load_time,
                        "inference_time": inference_time
                    }
                }
            else:
                logger.error(f"Pipeline returned unexpected output: {outputs}")
                return {"success": False, "error": "Unexpected output format"}
        except Exception as e:
            logger.error(f"Error testing pipeline: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        results = {}
        
        # Run pipeline test
        pipeline_result = self.test_pipeline()
        results["pipeline"] = pipeline_result
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
        # Add metadata
        results["metadata"] = {
            "model_id": self.model_id,
            "device": self.device,
            "timestamp": datetime.datetime.now().isoformat(),
            "success": pipeline_result.get("success", False),
            "performance": self.performance_stats,
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH,
            "has_tokenizers": HAS_TOKENIZERS,
            "has_sentencepiece": HAS_SENTENCEPIECE,
            "using_real_inference": using_real_inference,
            "using_mocks": using_mocks,
            "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
        }
        
        return results

def save_results(results, model_id=None, output_dir="collected_results"):
    """Save test results to a JSON file."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = model_id or results.get("metadata", {}).get("model_id", "unknown_model")
        model_id_safe = model_id.replace("/", "__")
        filename = f"hf_xlm_roberta_{model_id_safe}_{timestamp}.json"
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
    parser = argparse.ArgumentParser(description="Test XLM-RoBERTa HuggingFace models")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to run tests on (cuda, cpu, mps)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("\nAvailable XLM-RoBERTa models:")
        for model_id, info in XLM_ROBERTA_MODELS_REGISTRY.items():
            print(f"  - {model_id} ({info['full_name']})")
            print(f"      Parameters: {info['parameters']}, Embedding: {info['embedding_dim']}, Context: {info['context_length']}")
            print(f"      Description: {info['description']}")
        return 0
    
    # Initialize the test class
    xlm_roberta_tester = TestXlmRobertaModels(model_id=args.model, device=args.device)
    
    # Run the tests
    results = xlm_roberta_tester.run_tests(all_hardware=args.all_hardware)
    
    # Print a summary
    print("\n" + "="*50)
    print(f"TEST RESULTS SUMMARY")
    print("="*50)
    
    # Indicate real vs mock inference clearly
    using_real_inference = results["metadata"]["using_real_inference"] 
    using_mocks = results["metadata"]["using_mocks"]
    
    if using_real_inference and not using_mocks:
        print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")
    else:
        print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")
    
    print(f"\nModel: {args.model}")
    print(f"Device: {xlm_roberta_tester.device}")
    
    # Pipeline results
    pipeline_success = results["pipeline"].get("success", False)
    print(f"\nPipeline Test: {'✅ Success' if pipeline_success else '❌ Failed'}")
    if pipeline_success:
        top_prediction = results["pipeline"].get("top_prediction", "N/A")
        inference_time = results["pipeline"].get("performance", {}).get("inference_time", "N/A")
        print(f"  - Top prediction: {top_prediction}")
        print(f"  - Inference time: {inference_time:.2f} seconds" if isinstance(inference_time, (int, float)) else f"  - Inference time: {inference_time}")
    else:
        error = results["pipeline"].get("error", "Unknown error")
        print(f"  - Error: {error}")
    
    # Save results if requested
    if args.save:
        file_path = save_results(results, args.model)
        if file_path:
            print(f"\nResults saved to {file_path}")
    
    return 0 if pipeline_success else 1

if __name__ == "__main__":
    sys.exit(main())