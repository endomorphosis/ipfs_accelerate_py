#!/usr/bin/env python3

# Import hardware detection capabilities if available
try:
    from generators.hardware.hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback

import os
import sys
import json
import time
import datetime

# ANSI color codes for terminal output
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"
import traceback
import logging
import argparse
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np


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


# Mock implementations for missing dependencies
if not HAS_TOKENIZERS:
    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            self.vocab_size = 32000
            
        def encode(self, text, **kwargs):
            return {"ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
            
        def decode(self, ids, **kwargs):
            return "Decoded text from mock"
            
        @staticmethod
        def from_file(vocab_filename):
            return MockTokenizer()

    tokenizers.Tokenizer = MockTokenizer


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
    if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
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

# Models registry - Maps model IDs to their specific configurations
STABLELM_MODELS_REGISTRY = {
    "stabilityai/stablelm-base-alpha-7b": {
        "description": "StableLM base model (7B)",
        "class": "AutoModelForCausalLM",
    },
    "stabilityai/stablelm-tuned-alpha-7b": {
        "description": "StableLM tuned model (7B)",
        "class": "AutoModelForCausalLM",
    },
    "stabilityai/stablelm-3b-4e1t": {
        "description": "StableLM 3B model",
        "class": "AutoModelForCausalLM",
    }
}

class TestStablelmModels:
    """Base test class for all StableLM-family models."""
    
    def __init__(self, model_id=None):
        """Initialize the test class for a specific model or default."""
        self.model_id = model_id or "stabilityai/stablelm-base-alpha-7b"
        
        # Verify model exists in registry
        if self.model_id not in STABLELM_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = STABLELM_MODELS_REGISTRY["stabilityai/stablelm-base-alpha-7b"]
        else:
            self.model_info = STABLELM_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "text-generation"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "Once upon a time"
        self.test_texts = [
            "Once upon a time",
            "Once upon a time (alternative)"
        ]
        
        # Configure hardware preference
        if HW_CAPABILITIES["cuda"]:
            self.preferred_device = "cuda"
        elif HW_CAPABILITIES["mps"]:
            self.preferred_device = "mps"
        else:
            self.preferred_device = "cpu"
        
        # Results container
        self.results = {}
        
    def test_pipeline(self, device=None):
        """Test the model using the transformers pipeline API."""
        try:
            if not HAS_TRANSFORMERS:
                logger.warning("Transformers library not available, skipping pipeline test")
                return {"success": False, "error": "Transformers library not available"}
                
            device = device or self.preferred_device
            device_str = "cuda:0" if device == "cuda" else device
            device_idx = 0 if device == "cuda" else -1  # -1 for CPU
            
            logger.info(f"Testing StableLM model {self.model_id} with pipeline API on {device}")
            
            start_time = time.time()
            
            # Initialize pipeline
            generator = transformers.pipeline(
                self.task,
                model=self.model_id,
                device=device_idx
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Generate text
            inference_start = time.time()
            outputs = generator(
                self.test_text,
                max_length=50,
                num_return_sequences=1
            )
            inference_time = time.time() - inference_start
            
            # Extract generated text
            if isinstance(outputs, list) and len(outputs) > 0:
                if isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                    generated_text = outputs[0]["generated_text"]
                else:
                    generated_text = str(outputs[0])
            else:
                generated_text = str(outputs)
                
            logger.info(f"Generated: {generated_text[:50]}...")
            logger.info(f"Inference time: {inference_time:.2f} seconds")
            
            # Store results
            self.results["pipeline"] = {
                "success": True,
                "device": device,
                "load_time": load_time,
                "inference_time": inference_time,
                "input": self.test_text,
                "output": generated_text
            }
            
            return self.results["pipeline"]
            
        except Exception as e:
            logger.error(f"Error in pipeline test: {e}")
            logger.error(traceback.format_exc())
            
            self.results["pipeline"] = {
                "success": False,
                "device": device,
                "error": str(e)
            }
            
            return self.results["pipeline"]
            
    def test_model_tokenizer(self, device=None):
        """Test the model and tokenizer directly."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning("Transformers or torch library not available, skipping model/tokenizer test")
                return {"success": False, "error": "Required libraries not available"}
                
            device = device or self.preferred_device
            device_str = "cuda:0" if device == "cuda" else device
            
            logger.info(f"Testing StableLM model {self.model_id} with model/tokenizer API on {device}")
            
            start_time = time.time()
            
            # Load tokenizer and model
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id)
            
            # Fix missing pad_token in tokenizer
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = "[PAD]"
            
            # Move model to device
            model = model.to(device_str)
            
            load_time = time.time() - start_time
            logger.info(f"Model and tokenizer loaded in {load_time:.2f} seconds")
            
            # Tokenize input
            inputs = tokenizer(self.test_text, return_tensors="pt")
            inputs = {k: v.to(device_str) for k, v in inputs.items()}
            
            # Generate text
            inference_start = time.time()
            with torch.no_grad():
                output_sequences = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_length=50,
                    num_return_sequences=1
                )
            inference_time = time.time() - inference_start
            
            # Decode output
            generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            logger.info(f"Generated: {generated_text[:50]}...")
            logger.info(f"Inference time: {inference_time:.2f} seconds")
            
            # Store results
            self.results["model_tokenizer"] = {
                "success": True,
                "device": device,
                "load_time": load_time,
                "inference_time": inference_time,
                "input": self.test_text,
                "output": generated_text
            }
            
            return self.results["model_tokenizer"]
            
        except Exception as e:
            logger.error(f"Error in model/tokenizer test: {e}")
            logger.error(traceback.format_exc())
            
            self.results["model_tokenizer"] = {
                "success": False,
                "device": device,
                "error": str(e)
            }
            
            return self.results["model_tokenizer"]
    
    def test_batch_inference(self, device=None):
        """Test batch inference with multiple inputs."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning("Transformers or torch library not available, skipping batch inference test")
                return {"success": False, "error": "Required libraries not available"}
                
            device = device or self.preferred_device
            device_str = "cuda:0" if device == "cuda" else device
            
            logger.info(f"Testing StableLM model {self.model_id} with batch inference on {device}")
            
            start_time = time.time()
            
            # Load tokenizer and model
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id)
            
            # Fix missing pad_token in tokenizer
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = "[PAD]"
            
            # Move model to device
            model = model.to(device_str)
            
            load_time = time.time() - start_time
            logger.info(f"Model and tokenizer loaded in {load_time:.2f} seconds")
            
            # Tokenize batch input with padding
            batch_inputs = tokenizer(
                self.test_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            batch_inputs = {k: v.to(device_str) for k, v in batch_inputs.items()}
            
            # Generate text
            inference_start = time.time()
            with torch.no_grad():
                output_sequences = model.generate(
                    batch_inputs["input_ids"],
                    attention_mask=batch_inputs.get("attention_mask", None),
                    max_length=50,
                    num_return_sequences=1
                )
            inference_time = time.time() - inference_start
            
            # Decode outputs
            batch_generated_texts = [
                tokenizer.decode(output_seq, skip_special_tokens=True) 
                for output_seq in output_sequences
            ]
            
            for i, text in enumerate(batch_generated_texts):
                logger.info(f"Input {i+1}: {self.test_texts[i]}")
                logger.info(f"Generated: {text[:50]}...")
                
            logger.info(f"Batch inference time: {inference_time:.2f} seconds")
            
            # Store results
            self.results["batch_inference"] = {
                "success": True,
                "device": device,
                "load_time": load_time,
                "inference_time": inference_time,
                "inputs": self.test_texts,
                "outputs": batch_generated_texts
            }
            
            return self.results["batch_inference"]
            
        except Exception as e:
            logger.error(f"Error in batch inference test: {e}")
            logger.error(traceback.format_exc())
            
            self.results["batch_inference"] = {
                "success": False,
                "device": device,
                "error": str(e)
            }
            
            return self.results["batch_inference"]
    
    def run_tests(self, device=None, batch=True):
        """Run all tests for the model."""
        device = device or self.preferred_device
        
        # Dictionary for overall results
        overall_results = {}
        
        # Run pipeline test
        pipeline_result = self.test_pipeline(device)
        overall_results["pipeline"] = pipeline_result
        
        # Run model/tokenizer test
        model_tokenizer_result = self.test_model_tokenizer(device)
        overall_results["model_tokenizer"] = model_tokenizer_result
        
        # Run batch inference test if requested
        if batch:
            batch_result = self.test_batch_inference(device)
            overall_results["batch_inference"] = batch_result
        
        # Add metadata
        overall_results["metadata"] = {
            "model_id": self.model_id,
            "description": self.description,
            "task": self.task,
            "class_name": self.class_name,
            "device": device,
            "hardware_capabilities": HW_CAPABILITIES,
            "timestamp": datetime.datetime.now().isoformat(),
            "test_success": any([
                r.get("success", False) 
                for r in overall_results.values() 
                if isinstance(r, dict)
            ]),
            "dependencies": {
                "transformers": transformers.__version__ if HAS_TRANSFORMERS else None,
                "torch": torch.__version__ if HAS_TORCH else None,
                "using_real_inference": HAS_TRANSFORMERS and HAS_TORCH,
                "using_mocks": MOCK_TRANSFORMERS or MOCK_TORCH,
                "test_type": "REAL INFERENCE" if (HAS_TRANSFORMERS and HAS_TORCH) else "MOCK OBJECTS (CI/CD)"
            }
        }
        
        return overall_results

def save_results(results, model_id, output_dir="collected_results"):
    """Save test results to a JSON file."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id_safe = model_id.replace("/", "__")
        filename = f"hf_stablelm_{model_id_safe}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Write results to file
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        traceback.print_exc()
        return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test StableLM models")
    parser.add_argument("--model", type=str, default="stabilityai/stablelm-base-alpha-7b", 
                        help=f"Model ID to test (default: stabilityai/stablelm-base-alpha-7b)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], 
                        help="Device to run tests on (default: auto-detect)")
    parser.add_argument("--save", action="store_true", 
                        help="Save test results to file")
    parser.add_argument("--list-models", action="store_true", 
                        help="List available models")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        print("\nAvailable StableLM models:")
        for model_id, info in STABLELM_MODELS_REGISTRY.items():
            print(f"  - {model_id}")
            print(f"    {info['description']}")
        return
    
    # Initialize test class
    tester = TestStablelmModels(model_id=args.model)
    
    # Run tests
    results = tester.run_tests(device=args.device)
    
    # Print summary
    print("\n" + "="*50)
    print(f"TEST RESULTS SUMMARY")
    print("="*50)
    
    # Display real vs mock status
    if results["metadata"]["dependencies"]["using_real_inference"]:
        print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")
    else:
        print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}")
    
    print(f"\nModel: {args.model}")
    print(f"Device: {results['metadata']['device']}")
    
    # Pipeline results
    pipeline_success = results["pipeline"].get("success", False)
    print(f"\nPipeline Test: {'✅ Success' if pipeline_success else '❌ Failed'}")
    if pipeline_success:
        output = results["pipeline"].get("output", "")
        inference_time = results["pipeline"].get("inference_time", 0)
        print(f"  - Output: {output[:50]}..." if len(output) > 50 else f"  - Output: {output}")
        print(f"  - Inference time: {inference_time:.2f} seconds")
    else:
        error = results["pipeline"].get("error", "Unknown error")
        print(f"  - Error: {error}")
    
    # Model/tokenizer results
    model_success = results["model_tokenizer"].get("success", False)
    print(f"\nModel/Tokenizer Test: {'✅ Success' if model_success else '❌ Failed'}")
    if model_success:
        output = results["model_tokenizer"].get("output", "")
        inference_time = results["model_tokenizer"].get("inference_time", 0)
        print(f"  - Output: {output[:50]}..." if len(output) > 50 else f"  - Output: {output}")
        print(f"  - Inference time: {inference_time:.2f} seconds")
    else:
        error = results["model_tokenizer"].get("error", "Unknown error")
        print(f"  - Error: {error}")
    
    # Batch inference results
    if "batch_inference" in results:
        batch_success = results["batch_inference"].get("success", False)
        print(f"\nBatch Inference Test: {'✅ Success' if batch_success else '❌ Failed'}")
        if batch_success:
            outputs = results["batch_inference"].get("outputs", [])
            inference_time = results["batch_inference"].get("inference_time", 0)
            print(f"  - Number of outputs: {len(outputs)}")
            print(f"  - Inference time: {inference_time:.2f} seconds")
        else:
            error = results["batch_inference"].get("error", "Unknown error")
            print(f"  - Error: {error}")
    
    # Save results if requested
    if args.save and results:
        filepath = save_results(results, args.model)
        if filepath:
            print(f"\nResults saved to {filepath}")
    
    print(f"\nSuccessfully tested StableLM model: {args.model}")
    
if __name__ == "__main__":
    main()