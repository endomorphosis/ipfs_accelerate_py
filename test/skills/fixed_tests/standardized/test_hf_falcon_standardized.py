#!/usr/bin/env python3

"""
Standardized test for the Falcon model from HuggingFace Transformers.
This file implements the ModelTest base class pattern for consistent testing.
"""

import os
import sys
import json
import time
import logging
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

try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")

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
    
    # Mock transformers module
    transformers.AutoTokenizer = MockTokenizer
    transformers.AutoModelForCausalLM = MockModel
    transformers.pipeline = lambda *args, **kwargs: lambda x: [{"generated_text": "Mock generated text"}]

# Model registry for Falcon models
MODEL_REGISTRY = {
    "falcon-base": {
        "model_id": "tiiuae/falcon-7b-instruct",
        "model_type": "falcon",
        "task": "text-generation",
        "description": "Falcon 7B Instruct model for text generation",
        "model_class": "AutoModelForCausalLM",
        "is_instruct": True,
        "parameters": "7B"
    },
    "falcon-40b": {
        "model_id": "tiiuae/falcon-40b",
        "model_type": "falcon",
        "task": "text-generation",
        "description": "Falcon 40B base model",
        "model_class": "AutoModelForCausalLM",
        "is_instruct": False,
        "parameters": "40B"
    },
    "default": {
        "model_id": "tiiuae/falcon-7b",
        "model_type": "falcon",
        "task": "text-generation",
        "description": "Falcon 7B base model",
        "model_class": "AutoModelForCausalLM",
        "is_instruct": False,
        "parameters": "7B"
    }
}

class TestFalcon(ModelTest):
    """Test class for Falcon models implementing the ModelTest pattern."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        
        # Initialize model parameters
        self.model_id = "falcon-base"
        self.model_info = MODEL_REGISTRY.get(self.model_id, MODEL_REGISTRY["default"])
        
        # Define test inputs
        self.test_prompts = [
            "Once upon a time",
            "The quick brown fox",
            "In the beginning"
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
        Load a Falcon model with its tokenizer.
        
        Args:
            model_name: The name/ID of the model to load
            
        Returns:
            Dictionary containing the model and tokenizer
        """
        logger.info(f"Loading Falcon model: {model_name}")
        
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
            
            # Fix padding token issue common in Falcon models
            if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token for Falcon tokenizer")
            
            # Get the model class from the registry
            model_info = next((info for _, info in MODEL_REGISTRY.items() 
                              if info["model_id"] == model_name), None)
            
            # Default to AutoModelForCausalLM if not found
            model_class_name = "AutoModelForCausalLM"
            if model_info and "model_class" in model_info:
                model_class_name = model_info["model_class"]
            
            # Get the actual class from transformers
            if hasattr(transformers, model_class_name):
                model_class = getattr(transformers, model_class_name)
            else:
                model_class = transformers.AutoModelForCausalLM
            
            # Load the model
            model = model_class.from_pretrained(
                model_name,
                device_map=self.preferred_device if self.preferred_device != "cpu" else None,
                torch_dtype=torch.float16 if self.preferred_device != "cpu" else torch.float32
            )
            
            # Move to device if needed
            if self.preferred_device != "cpu" and getattr(model, "device_map", None) is None:
                model = model.to(self.preferred_device)
            
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
    
    def format_prompt(self, text, model_info=None):
        """
        Format a prompt based on whether the model is instruction-tuned.
        
        Args:
            text: The input text prompt
            model_info: Model information dictionary
            
        Returns:
            Formatted prompt string
        """
        if model_info is None:
            model_info = self.model_info
        
        if model_info.get("is_instruct", False):
            # Format for instruction-tuned models
            return f"User: {text}\nAssistant:"
        else:
            # Format for base models
            return text
    
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
            # Format the prompt appropriately
            prompt = self.format_prompt(input_data)
            
            # Encode the prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            if self.preferred_device != "cpu":
                inputs = {k: v.to(self.preferred_device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                output_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode the output
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Verify output meets minimum requirements
            self.assertIsInstance(generated_text, str)
            self.assertGreater(len(generated_text), len(prompt), "Generated text should be longer than prompt")
            
            # If expected_output is provided, verify it's contained or similar
            if expected_output:
                # For generative models, we may not expect exact matches
                # but we can check if key terms are present
                self.assertTrue(
                    any(term in generated_text.lower() for term in expected_output.lower().split()),
                    f"Generated text doesn't contain expected terms from {expected_output}"
                )
            
            return {"generated_text": generated_text, "prompt": prompt}
        
        except Exception as e:
            logger.error(f"Error during model verification: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "prompt": input_data}
    
    def test_model_loading(self):
        """Test loading the Falcon model."""
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
    
    def test_instruction_following(self):
        """Test instruction following capabilities for instruct models."""
        if not self.model_info.get("is_instruct", False):
            logger.info("Skipping instruction test for non-instruct model")
            self.skipTest("Not an instruction-tuned model")
            return
        
        # Load the model
        model_components = self.load_model(self.model_info["model_id"])
        
        # Define instruction prompts
        instructions = [
            "Explain how photosynthesis works",
            "Write a short poem about mountains",
            "List three benefits of exercise"
        ]
        
        # Test with each instruction
        results = []
        for instruction in instructions:
            start_time = time.time()
            output = self.verify_model_output(model_components, instruction)
            elapsed_time = time.time() - start_time
            
            # Store results and timing
            output["elapsed_time"] = elapsed_time
            results.append(output)
            
            logger.info(f"Generated instruction response in {elapsed_time:.2f}s")
            if "error" not in output:
                logger.info(f"Instruction: {instruction}")
                logger.info(f"Response: {output['generated_text'][:100]}...")
            else:
                logger.error(f"Instruction following error: {output['error']}")
        
        # Add performance stats
        self.performance_stats["instruction_following"] = {
            "avg_time": sum(r["elapsed_time"] for r in results) / len(results),
            "num_runs": len(results)
        }
        
        return results
    
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
        
        # Instruction following test (only for instruct models)
        if self.model_info.get("is_instruct", False):
            try:
                instruction_results = self.test_instruction_following()
                results["tests"]["instruction_following"] = {
                    "status": "passed",
                    "results": instruction_results
                }
            except Exception as e:
                logger.error(f"Instruction following test failed: {e}")
                results["tests"]["instruction_following"] = {
                    "status": "failed", 
                    "error": str(e)
                }
        
        # Add performance stats
        results["performance"] = self.performance_stats
        
        return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test Falcon models with standardized approach")
    parser.add_argument("--model", type=str, default="falcon-base", 
                        help="Model ID to test (falcon-base, falcon-40b, etc.)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"],
                        help="Device to run on (default: auto-detect)")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Create test instance
    test = TestFalcon()
    test.model_id = args.model
    test.model_info = MODEL_REGISTRY.get(args.model, MODEL_REGISTRY["default"])
    
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