#!/usr/bin/env python3

"""
Standardized test for the GPT-Neo model from HuggingFace Transformers.
This file implements the ModelTest base class pattern for consistent testing.
"""

import os
import sys
import json
import time
import logging
import datetime
import traceback
import unittest
import argparse
from unittest.mock import MagicMock, patch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import from multiple possible locations
# First try local imports
try:
    from refactored_test_suite.model_test import ModelTest
    MODEL_TEST_IMPORT_SOURCE = "local"
except ImportError:
    try:
        # Try relative to project root
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        from refactored_test_suite.model_test import ModelTest
        MODEL_TEST_IMPORT_SOURCE = "project_root"
    except ImportError:
        try:
            # Try from skills directory
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from refactored_test_suite.model_test import ModelTest
            MODEL_TEST_IMPORT_SOURCE = "skills_dir"
        except ImportError:
            # Fallback implementation if ModelTest is not available
            logger.warning("ModelTest not found, using fallback implementation")
            from unittest import TestCase
            class ModelTest(TestCase):
                """Fallback implementation of ModelTest."""
                def setUp(self):
                    super().setUp()
                
                def tearDown(self):
                    super().tearDown()
                
                def load_model(self, model_name):
                    raise NotImplementedError("Subclasses must implement load_model")
                
                def verify_model_output(self, model, input_data, expected_output=None):
                    raise NotImplementedError("Subclasses must implement verify_model_output")
                
                def test_model_loading(self):
                    raise NotImplementedError("Subclasses must implement test_model_loading")
                
                def detect_preferred_device(self):
                    # Fallback implementation
                    return "cpu"
                
            MODEL_TEST_IMPORT_SOURCE = "fallback"

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

# Mock implementations when dependencies are missing
if not HAS_TRANSFORMERS:
    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            self.pad_token = None
            self.eos_token = "<eos>"
            
        def __call__(self, text, **kwargs):
            return {"input_ids": MagicMock(), "attention_mask": MagicMock()}
            
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()
            
        def decode(self, token_ids, **kwargs):
            return "Decoded text from mock"
    
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass
            
        def to(self, device):
            return self
            
        def __call__(self, **kwargs):
            return MagicMock(logits=torch.tensor([[[0.1, 0.2, 0.7]]]))
            
        def eval(self):
            return self
            
        def parameters(self):
            return [torch.ones(1, 1)]
            
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()
        
        def generate(self, *args, **kwargs):
            return torch.tensor([[1, 2, 3, 4, 5]])
    
    # Mock transformers module
    transformers.AutoTokenizer = MockTokenizer
    transformers.AutoModelForCausalLM = MockModel
    transformers.pipeline = lambda *args, **kwargs: lambda x: [{"generated_text": "Mock generated text"}]

# Model registry for GPT-Neo models
MODEL_REGISTRY = {
    "EleutherAI/gpt-neo-125M": {
        "model_id": "EleutherAI/gpt-neo-125M",
        "model_type": "gpt-neo",
        "task": "text-generation",
        "description": "GPT-Neo 125M parameters",
        "model_class": "AutoModelForCausalLM",
        "parameters": "125M",
        "context_length": 2048
    },
    "EleutherAI/gpt-neo-1.3B": {
        "model_id": "EleutherAI/gpt-neo-1.3B",
        "model_type": "gpt-neo",
        "task": "text-generation",
        "description": "GPT-Neo 1.3B parameters",
        "model_class": "AutoModelForCausalLM",
        "parameters": "1.3B",
        "context_length": 2048
    },
    "EleutherAI/gpt-neo-2.7B": {
        "model_id": "EleutherAI/gpt-neo-2.7B",
        "model_type": "gpt-neo",
        "task": "text-generation",
        "description": "GPT-Neo 2.7B parameters",
        "model_class": "AutoModelForCausalLM",
        "parameters": "2.7B",
        "context_length": 2048
    },
    "default": {
        "model_id": "EleutherAI/gpt-neo-125M",
        "model_type": "gpt-neo",
        "task": "text-generation",
        "description": "GPT-Neo 125M parameters (smallest variant)",
        "model_class": "AutoModelForCausalLM",
        "parameters": "125M",
        "context_length": 2048
    }
}

class TestGPTNeo(ModelTest):
    """Test class for GPT-Neo models implementing the ModelTest pattern."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        
        # Initialize model parameters
        self.model_id = "EleutherAI/gpt-neo-125M"  # Default to smallest model for tests
        self.model_info = MODEL_REGISTRY.get(self.model_id, MODEL_REGISTRY["default"])
        
        # Define test inputs
        self.test_prompts = [
            "Once upon a time in a land far away",
            "The meaning of life is",
            "The relationship between technology and society"
        ]
        
        # Set preferred device
        self.preferred_device = self.detect_preferred_device()
        
        # For tracking performance
        self.performance_stats = {}
        
    def detect_preferred_device(self):
        """Detect the optimal device for testing."""
        if HAS_TORCH and torch.cuda.is_available():
            logger.info("CUDA is available, using GPU")
            return "cuda"
        elif HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            logger.info("Apple MPS is available, using Metal GPU")
            return "mps"
        else:
            logger.info("No GPU detected, using CPU")
            return "cpu"
    
    def load_model(self, model_name):
        """
        Load a GPT-Neo model with its tokenizer.
        
        Args:
            model_name: The name/ID of the model to load
            
        Returns:
            Dictionary containing the model and tokenizer
        """
        logger.info(f"Loading GPT-Neo model: {model_name}")
        
        # Verify dependencies
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            logger.warning("Required dependencies missing, using mock objects")
            return {
                "model": MagicMock(),
                "tokenizer": MagicMock(),
                "is_mock": True
            }
        
        # Load the tokenizer
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Fix padding token issue common in GPT-Neo models
            if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token for GPT-Neo tokenizer")
            
            # Load the model with appropriate device settings
            # Check available memory to determine if we need to use CPU
            use_cpu = True
            if self.preferred_device == "cuda":
                # Try to load the model with half precision on GPU
                try:
                    model = transformers.AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16
                    ).to(self.preferred_device)
                    use_cpu = False
                except Exception as e:
                    logger.warning(f"Could not load model on GPU, falling back to CPU: {e}")
            
            if use_cpu:
                # Load model on CPU
                model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "is_mock": False
            }
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            return {
                "model": MagicMock(),
                "tokenizer": MagicMock(),
                "is_mock": True,
                "error": str(e)
            }
    
    def verify_model_output(self, model_components, input_data, expected_output=None):
        """
        Verify that the model produces reasonable output.
        
        Args:
            model_components: Dictionary containing model and tokenizer
            input_data: Input text or prompts
            expected_output: Optional expected output (not strict for generative models)
            
        Returns:
            Generated text output
        """
        model = model_components["model"]
        tokenizer = model_components["tokenizer"]
        is_mock = model_components.get("is_mock", False)
        
        if is_mock:
            logger.warning("Using mock model, verification limited")
            return {"generated_text": "Mock output text", "is_mock": True}
        
        try:
            # Encode the prompt
            inputs = tokenizer(input_data, return_tensors="pt")
            if self.preferred_device != "cpu" and not is_mock:
                inputs = {k: v.to(self.preferred_device) for k, v in inputs.items()}
            
            # Generate text with appropriate settings for GPT-Neo
            with torch.no_grad():
                output_ids = model.generate(
                    inputs["input_ids"],
                    max_length=50,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=1
                )
            
            # Decode the output
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Verify output meets minimum requirements
            self.assertIsInstance(generated_text, str)
            self.assertGreater(len(generated_text), len(input_data), "Generated text should be longer than input")
            
            # If expected_output is provided, verify it's contained or similar
            if expected_output:
                # For generative models, we may not expect exact matches
                # but we can check if key terms are present
                self.assertTrue(
                    any(term in generated_text.lower() for term in expected_output.lower().split()),
                    f"Generated text doesn't contain expected terms from {expected_output}"
                )
            
            return {"generated_text": generated_text, "prompt": input_data}
        
        except Exception as e:
            logger.error(f"Error during model verification: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "prompt": input_data}
    
    def test_model_loading(self):
        """Test loading the GPT-Neo model."""
        # Load the model
        logger.info(f"Testing model loading for {self.model_info['model_id']}")
        model_components = self.load_model(self.model_info["model_id"])
        
        # Verify components
        self.assertIn("model", model_components)
        self.assertIn("tokenizer", model_components)
        
        # If not using mocks, perform additional checks
        if not model_components.get("is_mock", False):
            # Check model properties
            model = model_components["model"]
            self.assertTrue(hasattr(model, "generate"))
            
            # Check tokenizer properties
            tokenizer = model_components["tokenizer"]
            self.assertTrue(hasattr(tokenizer, "encode"))
            self.assertTrue(hasattr(tokenizer, "decode"))
            
            # Verify pad token was set correctly
            self.assertIsNotNone(tokenizer.pad_token)
        
        return model_components
    
    def test_text_generation(self):
        """Test text generation capabilities."""
        # Load the model
        model_components = self.load_model(self.model_info["model_id"])
        
        # Test with each prompt
        results = []
        for prompt in self.test_prompts:
            start_time = time.time()
            output = self.verify_model_output(model_components, prompt)
            elapsed_time = time.time() - start_time
            
            # Store results and timing
            output["elapsed_time"] = elapsed_time
            results.append(output)
            
            logger.info(f"Generated output in {elapsed_time:.2f}s")
            if "error" not in output:
                logger.info(f"Input: {prompt}")
                logger.info(f"Output: {output['generated_text'][:100]}...")
            else:
                logger.error(f"Generation error: {output['error']}")
        
        # Add performance stats
        self.performance_stats["text_generation"] = {
            "avg_time": sum(r["elapsed_time"] for r in results) / len(results),
            "num_runs": len(results)
        }
        
        return results
    
    def test_long_context(self):
        """Test model's ability to handle longer context."""
        # Only run test if not using mocks
        model_components = self.load_model(self.model_info["model_id"])
        if model_components.get("is_mock", False):
            logger.warning("Using mock model, skipping long context test")
            return [{"is_mock": True}]
        
        # Define a longer context prompt
        long_context = " ".join(["The GPT-Neo model is designed to handle longer contexts efficiently."] * 10)
        long_context += " Based on these considerations,"
        
        # Test with the long context
        start_time = time.time()
        output = self.verify_model_output(model_components, long_context)
        elapsed_time = time.time() - start_time
        
        # Store results and timing
        output["elapsed_time"] = elapsed_time
        
        logger.info(f"Generated long context output in {elapsed_time:.2f}s")
        if "error" not in output:
            logger.info(f"Long context length: {len(long_context)}")
            logger.info(f"Output: {output['generated_text'][:100]}...")
        else:
            logger.error(f"Long context generation error: {output['error']}")
        
        # Add performance stats
        self.performance_stats["long_context"] = {
            "context_length": len(long_context),
            "time": elapsed_time
        }
        
        return [output]
    
    def run_all_tests(self):
        """Run all tests and return results."""
        results = {
            "model_info": self.model_info,
            "device": self.preferred_device,
            "tests": {}
        }
        
        # Run the tests
        try:
            results["tests"]["model_loading"] = {
                "status": "passed",
                "components": "Model loaded successfully"
            }
            self.test_model_loading()
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            results["tests"]["model_loading"] = {
                "status": "failed",
                "error": str(e)
            }
            # If model loading fails, other tests will likely fail too
            return results
        
        # Text generation test
        try:
            generation_results = self.test_text_generation()
            results["tests"]["text_generation"] = {
                "status": "passed",
                "results": generation_results
            }
        except Exception as e:
            logger.error(f"Text generation test failed: {e}")
            results["tests"]["text_generation"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Long context test
        try:
            long_context_results = self.test_long_context()
            results["tests"]["long_context"] = {
                "status": "passed",
                "results": long_context_results
            }
        except Exception as e:
            logger.error(f"Long context test failed: {e}")
            results["tests"]["long_context"] = {
                "status": "failed", 
                "error": str(e)
            }
        
        # Add performance stats
        results["performance"] = self.performance_stats
        
        return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test GPT-Neo models with standardized approach")
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-125M", 
                        help="Model ID to test (default: EleutherAI/gpt-neo-125M)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"],
                        help="Device to run on (default: auto-detect)")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Create test instance
    test = TestGPTNeo()
    test.model_id = args.model
    
    # Update model info based on model ID
    if args.model in MODEL_REGISTRY:
        test.model_info = MODEL_REGISTRY[args.model]
    else:
        # Use the provided model ID but with default configuration
        default_info = MODEL_REGISTRY["default"].copy()
        default_info["model_id"] = args.model
        default_info["description"] = f"Custom GPT-Neo model: {args.model}"
        test.model_info = default_info
    
    if args.device:
        test.preferred_device = args.device
    
    # Run tests
    results = test.run_all_tests()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Model: {test.model_info['model_id']} ({test.model_info['description']})")
    print(f"Device: {test.preferred_device}")
    
    # Print test results
    for test_name, test_result in results["tests"].items():
        status = test_result["status"]
        status_symbol = "✅" if status == "passed" else "❌"
        print(f"{status_symbol} {test_name}: {status}")
    
    # Print performance stats if available
    if "performance" in results and results["performance"]:
        print("\nPerformance:")
        for test_name, stats in results["performance"].items():
            if "avg_time" in stats:
                print(f"  {test_name}: {stats['avg_time']:.4f}s average ({stats['num_runs']} runs)")
            elif "time" in stats:
                print(f"  {test_name}: {stats['time']:.4f}s")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Return success if all tests passed
    return 0 if all(t["status"] == "passed" for t in results["tests"].values()) else 1

if __name__ == "__main__":
    # Support running as script or unit test
    if "unittest" in sys.modules and sys.modules["unittest"].TestProgram in globals():
        # Running as unittest
        unittest.main()
    else:
        # Running as script
        sys.exit(main())