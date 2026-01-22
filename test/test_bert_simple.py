#!/usr/bin/env python3

"""
Simple test script to test a BERT model using HuggingFace transformers.
This is a simplified version for testing the basic transformers functionality.
"""

import os
import sys
import time
import logging
import argparse
import json
import datetime
import unittest
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ModelTest base class
try:
    from refactored_test_suite.model_test import ModelTest
except ImportError:
    # Fallback to alternative import path
    try:
        from model_test import ModelTest
    except ImportError:
        # Create a temporary ModelTest class if not available
        class ModelTest(unittest.TestCase):
            """Temporary ModelTest class."""
            pass

# Try to import required libraries
try:
    import torch
    import transformers
    from transformers import BertModel, BertTokenizer, pipeline
    
    HAS_REQUIREMENTS = True
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    HAS_REQUIREMENTS = False

class TestBertSimple(ModelTest):
    """Test class for BERT model with simplified implementation."""
    
    def setUp(self):
        """Initialize the test with model details and hardware detection."""
        super().setUp()
        self.model_id = "bert-base-uncased"
        self.model_type = "text_embedding"
        self.device = self.detect_preferred_device()
        self.results = {"model_id": self.model_id, "timestamp": datetime.datetime.now().isoformat()}
        self.results["device"] = self.device
        logger.info(f"Using device: {self.device}")
    
    def detect_preferred_device(self):
        """Detect available hardware and choose the preferred device."""
        try:
            # Check for CUDA
            if torch.cuda.is_available():
                return "cuda"
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                return "mps"
            
            # Fallback to CPU
            return "cpu"
        except ImportError:
            return "cpu"
    
    def load_model(self, model_name):
        """Load a model for testing."""
        if not HAS_REQUIREMENTS:
            logger.error("Missing required dependencies. Please install transformers and torch.")
            return None
        
        try:
            # Load model using BertModel for direct loading
            model_start = time.time()
            model = BertModel.from_pretrained(model_name)
            if self.device == "cuda":
                model = model.to(self.device)
            model_load_time = time.time() - model_start
            
            self.results["direct_load_time"] = model_load_time
            logger.info(f"Model load time: {model_load_time:.2f}s")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def verify_model_output(self, model, input_data, expected_output=None):
        """Verify that model produces expected output for text data."""
        if not HAS_REQUIREMENTS:
            self.fail("Missing required dependencies")
        
        try:
            # For BERT models, input_data should be tokenized text
            if isinstance(input_data, dict):
                # Input is already a dict of tensors (e.g., from tokenizer)
                inputs = input_data
            elif isinstance(input_data, str):
                # Raw text, needs tokenization
                tokenizer = BertTokenizer.from_pretrained(model.config._name_or_path)
                inputs = tokenizer(input_data, return_tensors="pt")
            else:
                # Otherwise, assume input is already properly formatted
                inputs = input_data
            
            # Move inputs to the right device if model is on a specific device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run model
            with torch.no_grad():
                direct_start = time.time()
                outputs = model(**inputs)
                direct_time = time.time() - direct_start
            
            self.results["direct_inference_time"] = direct_time
            logger.info(f"Inference time: {direct_time:.2f}s")
            
            # Verify outputs
            self.assertIsNotNone(outputs, "Model output should not be None")
            self.assertTrue(hasattr(outputs, "last_hidden_state"), "Model output should have last_hidden_state")
            
            # If expected output is provided, verify it matches
            if expected_output is not None:
                self.assertEqual(outputs, expected_output)
            
            return outputs
        except Exception as e:
            logger.error(f"Error verifying model output: {e}")
            self.fail(f"Model verification failed: {e}")
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        model = self.load_model(self.model_id)
        self.assertIsNotNone(model, "Model should load successfully")
        self.results["direct_success"] = True
    
    def test_bert_pipeline(self):
        """Test BERT model using HuggingFace pipeline API."""
        if not HAS_REQUIREMENTS:
            self.skipTest("Missing required dependencies")
        
        try:
            # Test with pipeline API
            logger.info(f"Testing {self.model_id} with pipeline...")
            start_time = time.time()
            nlp = pipeline("fill-mask", model=self.model_id, device=0 if self.device == "cuda" else -1)
            load_time = time.time() - start_time
            
            test_text = "The man worked as a [MASK]."
            
            # Run inference
            inference_start = time.time()
            outputs = nlp(test_text)
            inference_time = time.time() - inference_start
            
            # Record results
            self.results["pipeline_success"] = True
            self.results["pipeline_load_time"] = load_time
            self.results["pipeline_inference_time"] = inference_time
            top_prediction = outputs[0]["token_str"] if isinstance(outputs, list) and len(outputs) > 0 else "N/A"
            self.results["top_prediction"] = top_prediction
            
            logger.info(f"Pipeline test succeeded. Top prediction: {top_prediction}")
            logger.info(f"Pipeline load time: {load_time:.2f}s, Inference time: {inference_time:.2f}s")
            
            self.assertIsNotNone(outputs, "Pipeline output should not be None")
            self.assertTrue(isinstance(outputs, list) and len(outputs) > 0, "Pipeline should return list of results")
            
        except Exception as e:
            logger.error(f"Error testing pipeline: {e}")
            self.fail(f"Pipeline test failed: {e}")
    
    def test_bert_direct(self):
        """Test BERT model using direct model loading and inference."""
        if not HAS_REQUIREMENTS:
            self.skipTest("Missing required dependencies")
        
        try:
            # Load model
            model = self.load_model(self.model_id)
            self.assertIsNotNone(model, "Model should load successfully")
            
            # Get tokenizer
            tokenizer = BertTokenizer.from_pretrained(self.model_id)
            
            # Tokenize test input
            test_text = "The man worked as a programmer."
            inputs = tokenizer(test_text, return_tensors="pt")
            
            # Run verification
            outputs = self.verify_model_output(model, inputs)
            
            # Check output shape is as expected for BERT
            self.assertTrue(hasattr(outputs, "last_hidden_state"), "Output should have last_hidden_state")
            self.assertEqual(outputs.last_hidden_state.shape[0], 1, "Batch size should be 1")
            logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
            
            self.results["direct_success"] = True
            
        except Exception as e:
            logger.error(f"Error in direct test: {e}")
            self.fail(f"Direct test failed: {e}")
    
    def print_results(self):
        """Print formatted test results."""
        if self.results.get("pipeline_success", False) or self.results.get("direct_success", False):
            print("\nTest Summary:")
            print(f"Model: {self.model_id}")
            print(f"Device: {self.results.get('device', 'unknown')}")
            
            if self.results.get("pipeline_success", False):
                print("\nPipeline API:")
                print(f"  Load time: {self.results.get('pipeline_load_time', 'N/A'):.2f}s")
                print(f"  Inference time: {self.results.get('pipeline_inference_time', 'N/A'):.2f}s")
                print(f"  Top prediction: {self.results.get('top_prediction', 'N/A')}")
            
            if self.results.get("direct_success", False):
                print("\nDirect API:")
                print(f"  Load time: {self.results.get('direct_load_time', 'N/A'):.2f}s")
                print(f"  Inference time: {self.results.get('direct_inference_time', 'N/A'):.2f}s")
        else:
            print("\nTest Failed:")
            print(f"Error: {self.results.get('error', 'Unknown error')}")
    
    def save_results(self, output_path):
        """Save test results to a JSON file."""
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(exist_ok=True, parents=True)
            with open(path, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Test BERT models with HuggingFace transformers")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model ID to test")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only mode")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    
    args = parser.parse_args()
    
    # Create test instance
    test = TestBertSimple()
    
    # Override model and device if specified
    if args.model:
        test.model_id = args.model
    
    if args.cpu_only:
        test.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Run tests
    try:
        test.test_model_loading()
        test.test_bert_pipeline()
        test.test_bert_direct()
        test.results["success"] = True
    except Exception as e:
        logger.error(f"Error during tests: {e}")
        test.results["success"] = False
        test.results["error"] = str(e)
    
    # Print and save results
    test.print_results()
    if args.output:
        test.save_results(args.output)
    
    return 0 if test.results.get("success", False) else 1

if __name__ == "__main__":
    # Run either with unittest or as a script
    if 'unittest' in sys.argv:
        unittest.main()
    else:
        sys.exit(main())