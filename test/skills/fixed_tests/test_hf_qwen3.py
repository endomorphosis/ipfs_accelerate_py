#\!/usr/bin/env python3

"""
Test file for Qwen3 HuggingFace models.
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

# Model registry for Qwen3 models
QWEN3_MODELS_REGISTRY = {
    "Qwen/Qwen3-7B": {
        "full_name": "Qwen3 7B",
        "architecture": "decoder-only",
        "description": "Qwen3 7B large language model developed by Alibaba",
        "model_type": "qwen2",
        "parameters": "7B",
        "context_length": 32768,
        "embedding_dim": 4096,
        "attention_heads": 32,
        "layers": 32,
        "recommended_tasks": ["text-generation", "chat"]
    },
    "Qwen/Qwen3-7B-Instruct": {
        "full_name": "Qwen3 7B Instruct",
        "architecture": "decoder-only",
        "description": "Instruction-tuned Qwen3 7B model",
        "model_type": "qwen2",
        "parameters": "7B",
        "context_length": 32768,
        "embedding_dim": 4096,
        "attention_heads": 32,
        "layers": 32,
        "recommended_tasks": ["text-generation", "chat"]
    }
}

def select_device():
    """Select the best available device for inference."""
    if HAS_TORCH:
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"

class TestQwen3Models:
    """
    Test class for Qwen3 models.
    """
    
    def __init__(self, model_id="Qwen/Qwen3-7B", device=None):
        """Initialize the test class for Qwen3 models.
        
        Args:
            model_id: The model ID to test (default: "Qwen/Qwen3-7B")
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
                
            logger.info(f"Testing Qwen3 model {self.model_id} with pipeline API on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Initialize the pipeline
            pipe = transformers.pipeline(
                "text-generation", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Test with a simple input
            test_input = "Qwen3 is a large language model that"
            
            # Record inference start time
            inference_start = time.time()
            
            # Run inference
            outputs = pipe(test_input, max_length=50, num_return_sequences=1)
            
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
                "inference_time": inference_time,
                "output": outputs[0]["generated_text"] if outputs else None
            }
        except Exception as e:
            logger.error(f"Error testing pipeline: {e}")
            return {"success": False, "error": str(e)}
    
    def test_direct_model_inference(self):
        """Test the model using direct model API."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning("Required libraries not available, skipping direct model test")
                return {"success": False, "error": "Required libraries not available"}
                
            logger.info(f"Testing Qwen3 model {self.model_id} with direct model API on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Load tokenizer and model
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Test with a simple input
            test_input = "Qwen3 is a large language model that"
            
            # Tokenize input
            inputs = tokenizer(test_input, return_tensors="pt").to(self.device)
            
            # Record inference start time
            inference_start = time.time()
            
            # Run inference
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=50,
                    num_return_sequences=1
                )
            
            # Decode output
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Store performance stats
            self.performance_stats["direct_model"] = {
                "load_time": load_time,
                "inference_time": inference_time
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "inference_time": inference_time,
                "output": decoded_output
            }
        except Exception as e:
            logger.error(f"Error in direct model inference: {e}")
            return {"success": False, "error": str(e)}
    
    def test_multiple_prompts(self):
        """Test the model with multiple different prompts."""
        try:
            if not HAS_TRANSFORMERS:
                logger.warning("Required libraries not available, skipping multiple prompts test")
                return {"success": False, "error": "Required libraries not available"}
            
            logger.info(f"Testing Qwen3 model {self.model_id} with multiple prompts on {self.device}")
            
            # Initialize the pipeline once
            pipe = transformers.pipeline(
                "text-generation", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Test prompts
            test_prompts = [
                "What are the main features of Qwen3?",
                "Explain how Transformers work in simple terms.",
                "Write a short poem about artificial intelligence."
            ]
            
            results = {}
            
            for i, prompt in enumerate(test_prompts):
                # Record inference start time
                inference_start = time.time()
                
                # Run inference
                outputs = pipe(prompt, max_length=100, num_return_sequences=1)
                
                # Record inference time
                inference_time = time.time() - inference_start
                
                # Store result
                results[f"prompt_{i+1}"] = {
                    "prompt": prompt,
                    "inference_time": inference_time,
                    "output": outputs[0]["generated_text"] if outputs else None
                }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "prompt_results": results
            }
        except Exception as e:
            logger.error(f"Error in multiple prompts test: {e}")
            return {"success": False, "error": str(e)}
    
    def test_hardware_compatibility(self):
        """Test model compatibility with different hardware acceleration options."""
        results = {}
        
        # Test CPU
        logger.info("Testing Qwen3 model on CPU")
        cpu_tester = TestQwen3Models(model_id=self.model_id, device="cpu")
        cpu_result = cpu_tester.test_pipeline()
        results["cpu"] = {
            "success": cpu_result.get("success", False),
            "inference_time": cpu_result.get("inference_time", None)
        }
        
        # Test CUDA if available
        if HAS_TORCH and torch.cuda.is_available():
            logger.info("Testing Qwen3 model on CUDA")
            cuda_tester = TestQwen3Models(model_id=self.model_id, device="cuda:0")
            cuda_result = cuda_tester.test_pipeline()
            results["cuda"] = {
                "success": cuda_result.get("success", False),
                "inference_time": cuda_result.get("inference_time", None)
            }
        
        # Test MPS if available (Apple Silicon)
        if HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Testing Qwen3 model on MPS (Apple Silicon)")
            mps_tester = TestQwen3Models(model_id=self.model_id, device="mps")
            mps_result = mps_tester.test_pipeline()
            results["mps"] = {
                "success": mps_result.get("success", False),
                "inference_time": mps_result.get("inference_time", None)
            }
        
        # Add OpenVINO test stub for future implementation
        results["openvino"] = {
            "success": False,
            "error": "OpenVINO support not yet implemented for Qwen3 models"
        }
        
        return results
    
    def test_openvino_inference(self):
        """Test the model using OpenVINO for inference acceleration (placeholder for future implementation)."""
        logger.warning("OpenVINO support not yet implemented for Qwen3 models")
        return {
            "success": False,
            "error": "OpenVINO support not yet implemented for Qwen3 models"
        }
    
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        results = {}
        
        # Run pipeline test
        pipeline_result = self.test_pipeline()
        results["pipeline"] = pipeline_result
        
        # Run direct model test if pipeline was successful
        if pipeline_result.get("success", False):
            direct_result = self.test_direct_model_inference()
            results["direct_model"] = direct_result
            
            # Run multiple prompts test
            prompts_result = self.test_multiple_prompts()
            results["multiple_prompts"] = prompts_result
        
        # Test hardware compatibility if requested
        if all_hardware:
            hardware_results = self.test_hardware_compatibility()
            results["hardware_compatibility"] = hardware_results
        
        # Add metadata
        results["metadata"] = {
            "model_id": self.model_id,
            "device": self.device,
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test Qwen3 HuggingFace models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-7B", help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to run tests on (cuda, cpu, mps)")
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    
    args = parser.parse_args()
    
    # Initialize the test class
    qwen3_tester = TestQwen3Models(model_id=args.model, device=args.device)
    
    # Run the tests
    results = qwen3_tester.run_tests(all_hardware=args.all_hardware)
    
    # Print a summary
    success = results["pipeline"].get("success", False)
    
    print("\nTEST RESULTS SUMMARY:")
    
    if success:
        print(f"  Successfully tested {args.model}")
        print(f"  - Device: {qwen3_tester.device}")
        print(f"  - Inference time: {results['pipeline'].get('inference_time', 'N/A'):.4f}s")
    else:
        print(f"  Failed to test {args.model}")
        print(f"  - Error: {results['pipeline'].get('error', 'Unknown error')}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
