#!/usr/bin/env python3
"""
Class-based test file for encoder-only models compatible with the refactored test suite.

This template provides a unified testing interface for encoder-only models like BERT
within the refactored test suite architecture, inheriting from ModelTest.
"""

import os
import sys
import json
import time
import datetime
import logging
import numpy as np
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from unittest.mock import patch, MagicMock, Mock

# Import from the refactored test suite
from refactored_test_suite.model_test import ModelTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models registry - Maps model IDs to their specific configurations
ENCODER_MODELS_REGISTRY = {
    "bert-base-uncased": {
        "full_name": "BERT Base Uncased",
        "architecture": "encoder-only",
        "description": "BERT Base model with uncased vocabulary",
        "model_type": "bert",
        "parameters": "110M",
        "context_length": 512,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "model_class": "BertForMaskedLM",
        "tokenizer_class": "BertTokenizer",
        "recommended_tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"]
    },
    "roberta-base": {
        "full_name": "RoBERTa Base",
        "architecture": "encoder-only",
        "description": "RoBERTa Base model",
        "model_type": "roberta",
        "parameters": "125M",
        "context_length": 512,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "model_class": "RobertaForMaskedLM",
        "tokenizer_class": "RobertaTokenizer",
        "recommended_tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"]
    },
    "distilbert-base-uncased": {
        "full_name": "DistilBERT Base Uncased",
        "architecture": "encoder-only",
        "description": "DistilBERT Base model with uncased vocabulary (smaller and faster than BERT)",
        "model_type": "distilbert",
        "parameters": "66M",
        "context_length": 512,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 6,
        "model_class": "DistilBertForMaskedLM",
        "tokenizer_class": "DistilBertTokenizer",
        "recommended_tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"]
    }
}

class TestEncoderModel(ModelTest):
    """Test class for encoder-only models like BERT."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "bert-base-uncased"
        
        # Verify model exists in registry
        if self.model_id not in ENCODER_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = ENCODER_MODELS_REGISTRY["bert-base-uncased"]
        else:
            self.model_info = ENCODER_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "fill-mask"
        self.model_class = self.model_info["model_class"]
        self.tokenizer_class = self.model_info["tokenizer_class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "The quick brown fox jumps over the [MASK] dog."
        
        # Setup hardware detection
        self.setup_hardware()
    
    def setup_hardware(self):
        """Set up hardware detection."""
        try:
            # Try to import hardware detection capabilities
            from generators.hardware.hardware_detection import (
                HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
                detect_all_hardware
            )
            hardware_info = detect_all_hardware()
        except ImportError:
            # Fallback to manual detection
            import torch
            
            # Basic hardware detection
            self.has_cuda = torch.cuda.is_available()
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            self.has_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            
            # Check for OpenVINO
            try:
                import openvino
                self.has_openvino = True
            except ImportError:
                self.has_openvino = False
            
            # WebNN/WebGPU are not directly accessible in Python
            self.has_webnn = False
            self.has_webgpu = False
        
        # Configure preferred device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
        
        logger.info(f"Using device: {self.device}")
    
    def tearDown(self):
        """Clean up resources after the test."""
        # Release any resources that need cleanup
        super().tearDown()
    
    def load_model(self, model_id=None):
        """Load the model for testing."""
        model_id = model_id or self.model_id
        
        try:
            import torch
            import transformers
            
            # Get model and tokenizer classes
            model_class = getattr(transformers, self.model_class)
            tokenizer_class = getattr(transformers, self.tokenizer_class, transformers.AutoTokenizer)
            
            # Load the tokenizer
            tokenizer = tokenizer_class.from_pretrained(model_id)
            
            # Load the model
            model = model_class.from_pretrained(model_id)
            
            # Move to appropriate device
            model = model.to(self.device)
            
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_input(self):
        """Prepare input for the model."""
        return self.test_text
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        model_components = self.load_model()
        
        # Verify that model and tokenizer were loaded
        self.assertIsNotNone(model_components["model"])
        self.assertIsNotNone(model_components["tokenizer"])
        
        logger.info("Model loaded successfully")
    
    def test_basic_inference(self):
        """Test basic inference with the model."""
        import torch
        
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        tokenizer = model_components["tokenizer"]
        
        # Prepare input
        input_text = self.prepare_input()
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Move inputs to device if needed
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Check for logits in output
        self.assertTrue(hasattr(outputs, "logits"))
        self.assertGreater(outputs.logits.shape[0], 0)
        
        # For masked language modeling, check the predictions for the mask token
        if self.task == "fill-mask" and "[MASK]" in input_text:
            # Get the position of the mask token
            mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            
            # Get top predictions
            mask_token_logits = outputs.logits[0, mask_token_index, :]
            top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            top_tokens_words = [tokenizer.decode([token]).strip() for token in top_tokens]
            
            logger.info(f"Top predictions for mask: {', '.join(top_tokens_words)}")
        
        logger.info(f"Basic inference successful: {outputs.logits.shape}")
    
    def test_pipeline_inference(self):
        """Test inference using the pipeline API."""
        try:
            import transformers
            
            # Initialize the pipeline
            pipe = transformers.pipeline(
                self.task, 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Test with input text
            input_text = self.prepare_input()
            
            # Run inference
            outputs = pipe(input_text)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            
            # Log results
            if isinstance(outputs, list) and len(outputs) > 0:
                if self.task == "fill-mask":
                    top_prediction = outputs[0]['token_str'] if 'token_str' in outputs[0] else outputs[0].get('token', 'N/A')
                    logger.info(f"Top prediction: {top_prediction}")
                else:
                    logger.info(f"Pipeline output: {outputs[:2]}")
            
            logger.info("Pipeline inference successful")
            
        except Exception as e:
            logger.error(f"Error in pipeline inference: {e}")
            self.fail(f"Pipeline inference failed: {e}")
    
    def test_hardware_compatibility(self):
        """Test the model's compatibility with different hardware platforms."""
        devices_to_test = []
        
        # Add available devices
        if self.has_cuda:
            devices_to_test.append('cuda')
        if self.has_mps:
            devices_to_test.append('mps')
        
        # Always test CPU
        if 'cpu' not in devices_to_test:
            devices_to_test.append('cpu')
        
        results = {}
        
        # Test on each device
        for device in devices_to_test:
            original_device = self.device
            try:
                logger.info(f"Testing on {device}...")
                self.device = device
                
                # Load model and prepare input
                model_components = self.load_model()
                model = model_components["model"]
                tokenizer = model_components["tokenizer"]
                
                input_text = self.prepare_input()
                inputs = tokenizer(input_text, return_tensors="pt")
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference
                import torch
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Verify results
                results[device] = True
                logger.info(f"Test on {device} successful")
                
            except Exception as e:
                logger.error(f"Error testing on {device}: {e}")
                results[device] = False
            finally:
                # Restore original device
                self.device = original_device
        
        # Verify at least one device works
        self.assertTrue(any(results.values()), "Model should work on at least one device")
        
        # Log results
        for device, success in results.items():
            logger.info(f"Device {device}: {'Success' if success else 'Failed'}")
    
    def test_openvino_compatibility(self):
        """Test compatibility with OpenVINO, if available."""
        if not getattr(self, 'has_openvino', False):
            logger.info("OpenVINO not available, skipping test")
            self.skipTest("OpenVINO not available")
        
        try:
            from optimum.intel import OVModelForMaskedLM, OVModelForSequenceClassification
            
            # Determine the appropriate OV model class based on task
            if self.task == "fill-mask":
                ov_model_class = OVModelForMaskedLM
            else:
                ov_model_class = OVModelForSequenceClassification
                
            # Load tokenizer
            model_components = self.load_model()
            tokenizer = model_components["tokenizer"]
            
            # Load model with OpenVINO
            model = ov_model_class.from_pretrained(
                self.model_id,
                export=True,
                provider="CPU"
            )
            
            # Prepare input
            input_text = self.prepare_input()
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Run inference
            outputs = model(**inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            
            logger.info("OpenVINO compatibility test successful")
        except ImportError:
            logger.warning("optimum-intel not available, skipping detailed test")
            self.skipTest("optimum-intel not available")
        except Exception as e:
            logger.error(f"Error in OpenVINO test: {e}")
            raise
    
    def run_all_tests(self):
        """Run all tests for this model."""
        test_methods = [method for method in dir(self) if method.startswith('test_')]
        results = {}
        
        for method in test_methods:
            try:
                logger.info(f"Running {method}...")
                getattr(self, method)()
                results[method] = "PASS"
            except Exception as e:
                logger.error(f"Error in {method}: {e}")
                results[method] = f"FAIL: {str(e)}"
        
        return results


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test encoder-only models with refactored test suite")
    parser.add_argument("--model", type=str, default="bert-base-uncased", 
                       help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to test on (cpu, cuda, etc.)")
    parser.add_argument("--task", type=str, default="fill-mask", 
                       help="Task to test (fill-mask, text-classification, etc.)")
    parser.add_argument("--save-results", action="store_true", help="Save test results to file")
    
    args = parser.parse_args()
    
    # Create test instance
    test = TestEncoderModel()
    
    # Override model ID if specified
    if args.model:
        test.model_id = args.model
    
    # Override device if specified
    if args.device:
        test.device = args.device
        
    # Override task if specified
    if args.task:
        test.task = args.task
    
    # Run tests
    test.setUp()
    results = test.run_all_tests()
    test.tearDown()
    
    # Print results
    print("\nTest Results:")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    # Save results if requested
    if args.save_results:
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{args.model.replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "w") as f:
            json.dump({
                "model": args.model,
                "device": test.device,
                "task": test.task,
                "results": results,
                "timestamp": datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()