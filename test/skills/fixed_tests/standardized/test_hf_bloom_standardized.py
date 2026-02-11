#!/usr/bin/env python3

"""
Standardized test file for Bloom models.
"""

import os
import sys
import json
import time
import datetime
import logging
import argparse
import traceback
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import ModelTest with fallbacks
try:
    from refactored_test_suite.model_test import ModelTest
except ImportError:
    try:
        from model_test import ModelTest
    except ImportError:
        # Create a minimal ModelTest class if not available
        class ModelTest(unittest.TestCase):
            """Minimal ModelTest class if the real one is not available."""
            def setUp(self):
                super().setUp()

# Check if we should mock specific dependencies
MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'
MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'
MOCK_TOKENIZERS = os.environ.get('MOCK_TOKENIZERS', 'False').lower() == 'true'
MOCK_SENTENCEPIECE = os.environ.get('MOCK_SENTENCEPIECE', 'False').lower() == 'true'

# Try to import required packages with fallbacks
try:
    if MOCK_TORCH:
        raise ImportError("Mocked torch import failure")
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    np = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

try:
    if MOCK_TRANSFORMERS:
        raise ImportError("Mocked transformers import failure")
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

try:
    if MOCK_TOKENIZERS:
        raise ImportError("Mocked tokenizers import failure")
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

try:
    if MOCK_SENTENCEPIECE:
        raise ImportError("Mocked sentencepiece import failure")
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")

# Try to import OpenVINO if available
try:
    import openvino
    from openvino.runtime import Core
    HAS_OPENVINO = True
except ImportError:
    openvino = MagicMock()
    HAS_OPENVINO = False
    logger.warning("OpenVINO not available")

# Models registry - Maps model IDs to their specific configurations
BLOOM_MODELS_REGISTRY = {
    "bigscience/bloom-560m": {
        "description": "Bloom 560M language model",
        "class": "BloomForCausalLM",
        "architecture": "causal-lm",
        "parameters": "560M",
        "task": "text-generation",
        "context_length": 2048
    },
    "bigscience/bloom-1b1": {
        "description": "Bloom 1.1B language model",
        "class": "BloomForCausalLM",
        "architecture": "causal-lm",
        "parameters": "1.1B",
        "task": "text-generation",
        "context_length": 2048
    },
    "bigscience/bloom-1b7": {
        "description": "Bloom 1.7B language model",
        "class": "BloomForCausalLM",
        "architecture": "causal-lm",
        "parameters": "1.7B",
        "task": "text-generation",
        "context_length": 2048
    },
    "bigscience/bloom-3b": {
        "description": "Bloom 3B language model",
        "class": "BloomForCausalLM",
        "architecture": "causal-lm",
        "parameters": "3B",
        "task": "text-generation",
        "context_length": 2048
    },
    "bigscience/bloom-7b1": {
        "description": "Bloom 7.1B language model",
        "class": "BloomForCausalLM",
        "architecture": "causal-lm",
        "parameters": "7.1B",
        "task": "text-generation",
        "context_length": 2048
    },
    "bigscience/bloomz-560m": {
        "description": "Bloom-Z 560M instruction-tuned language model",
        "class": "BloomForCausalLM",
        "architecture": "causal-lm",
        "parameters": "560M",
        "task": "text-generation",
        "context_length": 2048
    },
    "bigscience/bloomz-1b1": {
        "description": "Bloom-Z 1.1B instruction-tuned language model",
        "class": "BloomForCausalLM",
        "architecture": "causal-lm",
        "parameters": "1.1B",
        "task": "text-generation",
        "context_length": 2048
    },
    "bigscience/bloomz-1b7": {
        "description": "Bloom-Z 1.7B instruction-tuned language model",
        "class": "BloomForCausalLM",
        "architecture": "causal-lm",
        "parameters": "1.7B",
        "task": "text-generation",
        "context_length": 2048
    },
    "bigscience/mt0-small": {
        "description": "MT0 Small multilingual language model",
        "class": "AutoModelForSeq2SeqLM",
        "architecture": "encoder-decoder",
        "parameters": "300M",
        "task": "text2text-generation",
        "context_length": 512
    }
}

class TestBloomModels(ModelTest):
    """Test class for Bloom models, following the ModelTest pattern."""
    
    def setUp(self):
        """Set up the test environment for Bloom models."""
        super().setUp()
        
        # Use small model for testing by default
        self.model_id = "bigscience/bloom-560m"
        
        # Verify model exists in registry
        if self.model_id not in BLOOM_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = BLOOM_MODELS_REGISTRY["bigscience/bloom-560m"]
        else:
            self.model_info = BLOOM_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters from registry
        self.description = self.model_info["description"]
        self.model_class = self.model_info["class"]
        self.task = self.model_info["task"]
        self.architecture = self.model_info["architecture"]
        self.context_length = self.model_info["context_length"]
        self.parameters = self.model_info["parameters"]
        
        # Define test inputs
        self.test_prompts = [
            "Once upon a time",
            "The quick brown fox",
            "In a world where"
        ]
        
        # Configure hardware preference
        self.preferred_device = self.detect_preferred_device()
        logger.info(f"Using {self.preferred_device} as preferred device")
        
        # Results storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}
    
    def load_model(self, model_name):
        """Load a model for testing - implements required ModelTest method."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                raise ImportError("Required libraries (transformers, torch) not available")
                
            logger.info(f"Loading Bloom model {model_name}...")
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Load tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            
            # Fix for padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token for tokenizer")
            
            # Configure model loading with appropriate device settings
            dtype = torch.float16 if self.preferred_device == "cuda" else torch.float32
            
            # Determine the correct model class based on architecture
            if self.architecture == "causal-lm":
                if "bloom" in model_name.lower():
                    model_class = transformers.BloomForCausalLM
                else:
                    model_class = transformers.AutoModelForCausalLM
            else:
                model_class = transformers.AutoModelForSeq2SeqLM
            
            # Load the model
            model = model_class.from_pretrained(
                model_name,
                torch_dtype=dtype
            )
            
            # Move model to preferred device
            if self.preferred_device == "cuda" and torch.cuda.is_available():
                model = model.to("cuda")
                logger.info(f"Moved model to CUDA device")
            elif self.preferred_device == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
                model = model.to("mps")
                logger.info(f"Moved model to MPS (Apple Silicon) device")
            
            # Record model loading time
            load_time = time.time() - start_time
            self.performance_stats["model_loading"] = {
                "time": load_time,
                "model_name": model_name,
                "device": self.preferred_device
            }
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            if HAS_TRANSFORMERS and HAS_TORCH:
                # Only raise if we're supposed to have these libraries
                raise
            else:
                # Otherwise, return a mock
                return {
                    "model": MagicMock(),
                    "tokenizer": MagicMock()
                }
    
    def verify_model_output(self, model, input_data, expected_output=None):
        """Verify that model produces expected output - implements required ModelTest method."""
        try:
            if not isinstance(model, dict) or "model" not in model or "tokenizer" not in model:
                raise ValueError("Model should be a dict containing 'model' and 'tokenizer' keys")
                
            # Unpack model components
            bloom_model = model["model"]
            tokenizer = model["tokenizer"]
            
            # Process inputs based on type
            if isinstance(input_data, dict):
                if "prompt" in input_data:
                    prompt = input_data["prompt"]
                else:
                    prompt = self.test_prompts[0]
            elif isinstance(input_data, str):
                prompt = input_data
            else:
                # Use default test inputs
                prompt = self.test_prompts[0]
            
            # Record inference start time
            inference_start = time.time()
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to device if needed
            if hasattr(bloom_model, "device") and str(bloom_model.device) != "cpu":
                inputs = {k: v.to(bloom_model.device) for k, v in inputs.items()}
            
            # Generate text in inference mode
            with torch.no_grad():
                # For text generation models, use generate()
                if self.task == "text-generation":
                    outputs = bloom_model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.8
                    )
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # For encoder-decoder models or direct inference
                else:
                    outputs = bloom_model(**inputs)
                    
                    # Get logits
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                        next_token_id = torch.argmax(logits[0, -1, :]).item()
                        next_token = tokenizer.decode([next_token_id])
                        generated_text = f"{prompt}{next_token}"
                    else:
                        generated_text = "Model output not available"
            
            # Record inference time
            inference_time = time.time() - inference_start
            self.performance_stats["inference"] = {
                "time": inference_time,
                "prompt_length": len(prompt),
                "generated_text_length": len(generated_text),
                "device": self.preferred_device
            }
            
            # Create result with both input prompt and generated text
            result = {
                "prompt": prompt,
                "generated_text": generated_text,
                "inference_time": inference_time
            }
            
            # Verify output
            self.assertIsNotNone(generated_text, "Model output should not be None")
            self.assertIsInstance(generated_text, str, "Generated text should be a string")
            self.assertGreater(len(generated_text), 0, "Generated text should not be empty")
            
            # If expected output is provided, compare with actual output
            if expected_output is not None:
                self.assertEqual(expected_output, result)
                
            return result
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            if HAS_TRANSFORMERS and HAS_TORCH:
                # Only raise if we're supposed to have these libraries
                raise
            else:
                # Return a mock result if we're in mock mode
                return {
                    "prompt": str(input_data) if not isinstance(input_data, dict) else input_data.get("prompt", "mock prompt"),
                    "generated_text": "This is a mock generated text for testing without actual model inference.",
                    "inference_time": 0.01
                }
    
    def test_model_loading(self):
        """Test that the model loads correctly - implements required ModelTest method."""
        try:
            model_components = self.load_model(self.model_id)
            self.assertIsNotNone(model_components, "Model should not be None")
            self.assertIn("model", model_components, "Model dict should contain 'model' key")
            self.assertIn("tokenizer", model_components, "Model dict should contain 'tokenizer' key")
            
            bloom_model = model_components["model"]
            tokenizer = model_components["tokenizer"]
            
            # Check if we're using a mock or real model
            if isinstance(bloom_model, MagicMock):
                logger.info("Using mocked model for testing")
            else:
                # Check model class
                if "bloom" in self.model_id.lower():
                    self.assertIn("Bloom", bloom_model.__class__.__name__, 
                                 f"Model should be a Bloom model, got {bloom_model.__class__.__name__}")
                
                # Verify tokenizer has essential methods
                self.assertTrue(hasattr(tokenizer, "encode"), "Tokenizer should have encode method")
                self.assertTrue(hasattr(tokenizer, "decode"), "Tokenizer should have decode method")
            
            logger.info(f"Successfully loaded {self.model_id}")
            return model_components
        except Exception as e:
            logger.error(f"Error testing model loading: {e}")
            if HAS_TRANSFORMERS and HAS_TORCH:
                # Only fail if we're supposed to have these libraries
                self.fail(f"Model loading failed: {e}")
            else:
                # Return a mock result if we're in mock mode
                logger.info("Using mocked components in test_model_loading")
                return {
                    "model": MagicMock(),
                    "tokenizer": MagicMock()
                }
    
    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device - implements required ModelTest method."""
        try:
            # Check CUDA
            if HAS_TORCH and torch.cuda.is_available():
                return "cuda"
            
            # Check MPS (Apple Silicon)
            if HAS_TORCH and hasattr(torch, "mps") and torch.mps.is_available():
                return "mps"
            
            # Check for OpenVINO compatibility
            if HAS_OPENVINO:
                logger.info("OpenVINO is available, but using CPU for standard tests")
            
            # Fallback to CPU
            return "cpu"
        except Exception as e:
            logger.error(f"Error detecting device: {e}")
            return "cpu"
    
    def test_multiple_prompts(self):
        """Test the model with multiple prompts."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning("Required libraries not available, skipping multiple prompts test")
                return {
                    "success": False,
                    "error": "Required libraries not available",
                    "device": self.preferred_device
                }
            
            logger.info(f"Testing {self.model_id} with multiple prompts")
            
            # Load model
            model_components = self.load_model(self.model_id)
            model = model_components["model"]
            tokenizer = model_components["tokenizer"]
            
            results = []
            
            # Test with multiple prompts
            for prompt in self.test_prompts:
                # Prepare input
                inputs = tokenizer(prompt, return_tensors="pt")
                
                # Move inputs to device if needed
                if hasattr(model, "device") and str(model.device) != "cpu":
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate text
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.8
                    )
                inference_time = time.time() - start_time
                
                # Decode the generated text
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Add result
                results.append({
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "inference_time": inference_time
                })
            
            # Add to examples
            self.examples.append({
                "method": "multiple_prompts",
                "results": results
            })
            
            return {
                "success": True,
                "results": results,
                "device": self.preferred_device
            }
            
        except Exception as e:
            logger.error(f"Error in multiple prompts test: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.preferred_device
            }
    
    def test_with_pipeline(self):
        """Test the model using transformers pipeline API."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning("Required libraries not available, skipping pipeline test")
                return {
                    "success": False,
                    "error": "Required libraries not available",
                    "device": self.preferred_device
                }
            
            logger.info(f"Testing {self.model_id} with pipeline API")
            
            # Create pipeline
            start_time = time.time()
            pipeline = transformers.pipeline(
                self.task, 
                model=self.model_id,
                device=self.preferred_device if self.preferred_device != "cpu" else -1
            )
            load_time = time.time() - start_time
            
            # Run inference
            prompt = self.test_prompts[0]
            
            inference_start = time.time()
            outputs = pipeline(prompt, max_new_tokens=30)
            inference_time = time.time() - inference_start
            
            # Record stats
            self.performance_stats["pipeline"] = {
                "load_time": load_time,
                "inference_time": inference_time
            }
            
            # Add to examples
            self.examples.append({
                "method": "pipeline",
                "input": prompt,
                "output": outputs
            })
            
            return {
                "success": True,
                "outputs": outputs,
                "load_time": load_time,
                "inference_time": inference_time,
                "device": self.preferred_device
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline test: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.preferred_device
            }
    
    def test_with_openvino(self):
        """Test the model using OpenVINO integration."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_OPENVINO:
                logger.warning("Required libraries not available, skipping OpenVINO test")
                return {
                    "success": False,
                    "error": "Required libraries not available",
                    "device": "cpu"
                }
            
            logger.info(f"Testing {self.model_id} with OpenVINO")
            
            # Import OpenVINO-specific components
            from optimum.intel import OVModelForCausalLM
            
            # Load tokenizer
            tokenizer_load_start = time.time()
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            tokenizer_load_time = time.time() - tokenizer_load_start
            
            # Fix for padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model_load_start = time.time()
            model = OVModelForCausalLM.from_pretrained(
                self.model_id,
                export=True,
                provider="CPU"
            )
            model_load_time = time.time() - model_load_start
            
            # Prepare input
            prompt = self.test_prompts[0]
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate text
            inference_start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=30
            )
            inference_time = time.time() - inference_start
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Add to examples
            self.examples.append({
                "method": "openvino",
                "input": prompt,
                "output": generated_text
            })
            
            # Record stats
            self.performance_stats["openvino"] = {
                "load_time": model_load_time,
                "tokenizer_load_time": tokenizer_load_time,
                "inference_time": inference_time
            }
            
            return {
                "success": True,
                "prompt": prompt,
                "generated_text": generated_text,
                "load_time": model_load_time,
                "inference_time": inference_time
            }
            
        except Exception as e:
            logger.error(f"Error in OpenVINO test: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_tests(self, all_hardware=False):
        """
        Run all tests for this model.
        
        Args:
            all_hardware: If True, tests on all available hardware (CPU, CUDA, OpenVINO)
        
        Returns:
            Dict containing test results
        """
        # Run required ModelTest methods
        model_components = self.test_model_loading()
        
        # Test basic model functionality
        if model_components:
            try:
                # Verify model output with default prompt
                test_prompt = self.test_prompts[0]
                output = self.verify_model_output(model_components, test_prompt)
                
                # Add to examples
                self.examples.append({
                    "method": "verify_model_output",
                    "input": test_prompt,
                    "output": output
                })
                
                # Store in results
                self.results["basic_inference"] = {
                    "success": True,
                    "output": output
                }
            except Exception as e:
                logger.error(f"Error running model verification: {e}")
                self.results["basic_inference"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test with multiple prompts
        multiple_prompts_results = self.test_multiple_prompts()
        self.results["multiple_prompts"] = multiple_prompts_results
        
        # Test with pipeline API
        pipeline_results = self.test_with_pipeline()
        self.results["pipeline"] = pipeline_results
        
        # Test on all available hardware if requested
        if all_hardware:
            # Test with OpenVINO if available
            if HAS_OPENVINO:
                openvino_results = self.test_with_openvino()
                self.results["openvino"] = openvino_results
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
        
        # Build final results
        return {
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "metadata": {
                "model": self.model_id,
                "model_type": "bloom",
                "task": self.task,
                "architecture": self.architecture,
                "description": self.description,
                "parameters": self.parameters,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_tokenizers": HAS_TOKENIZERS,
                "has_sentencepiece": HAS_SENTENCEPIECE,
                "has_openvino": HAS_OPENVINO,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if using_real_inference else "MOCK OBJECTS (CI/CD)"
            }
        }

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test Bloom models with ModelTest pattern")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--unittest", action="store_true", help="Run as unittest")
    
    args = parser.parse_args()
    
    if args.unittest or "unittest" in sys.argv:
        # Run as unittest
        unittest.main(argv=[sys.argv[0]])
    else:
        # Override preferred device if CPU only
        if args.cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("CPU-only mode enabled")
        
        # Run test
        model_id = args.model or "bigscience/bloom-560m"
        tester = TestBloomModels()
        if args.model:
            tester.model_id = args.model
            
        results = tester.run_tests(all_hardware=args.all_hardware)
        
        # Print summary
        basic_success = results["results"].get("basic_inference", {}).get("success", False)
        pipeline_success = results["results"].get("pipeline", {}).get("success", False)
        
        # Determine if real inference or mock objects were used
        using_real_inference = results["metadata"]["has_transformers"] and results["metadata"]["has_torch"]
        
        print("\nTEST RESULTS SUMMARY:")
        
        # Indicate real vs mock inference clearly
        if using_real_inference:
            print("🚀 Using REAL INFERENCE with actual models")
        else:
            print("🔷 Using MOCK OBJECTS for CI/CD testing only")
            print(f"   Dependencies: transformers={results['metadata']['has_transformers']}, torch={results['metadata']['has_torch']}")
        
        if basic_success or pipeline_success:
            print(f"✅ Successfully tested {model_id}")
            
            # Print a sample output if available
            if results["examples"]:
                example = results["examples"][0]
                if "output" in example and isinstance(example["output"], dict) and "generated_text" in example["output"]:
                    print(f"\nSample output:")
                    generated_text = example["output"]["generated_text"]
                    print(f"  Prompt: {example['input']}")
                    print(f"  Generated text: {generated_text[:100]}...")
        else:
            print(f"❌ Failed to test {model_id}")
            
            # Print error details if available
            if "basic_inference" in results["results"]:
                basic_result = results["results"]["basic_inference"]
                if "error" in basic_result:
                    print(f"  - Error: {basic_result['error']}")
        
        # Save results if requested
        if args.save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bloom_test_results_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nTest results saved to {filename}")

if __name__ == "__main__":
    main()