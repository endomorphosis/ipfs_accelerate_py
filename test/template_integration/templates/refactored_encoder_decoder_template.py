#!/usr/bin/env python3
"""
Class-based test file for encoder-decoder models compatible with the refactored test suite.

This template provides a unified testing interface for encoder-decoder models like T5 and BART
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
ENCODER_DECODER_MODELS_REGISTRY = {
    "t5-small": {
        "full_name": "T5 Small",
        "architecture": "encoder-decoder",
        "description": "T5 Small model",
        "model_type": "t5",
        "parameters": "60M",
        "context_length": 512,
        "embedding_dim": 512,
        "attention_heads": 8,
        "layers": 6,
        "model_class": "T5ForConditionalGeneration",
        "tokenizer_class": "T5Tokenizer",
        "recommended_tasks": ["translation", "summarization", "question-answering"]
    },
    "t5-base": {
        "full_name": "T5 Base",
        "architecture": "encoder-decoder",
        "description": "T5 Base model",
        "model_type": "t5",
        "parameters": "220M",
        "context_length": 512,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "model_class": "T5ForConditionalGeneration",
        "tokenizer_class": "T5Tokenizer",
        "recommended_tasks": ["translation", "summarization", "question-answering"]
    },
    "facebook/bart-base": {
        "full_name": "BART Base",
        "architecture": "encoder-decoder",
        "description": "BART Base model",
        "model_type": "bart",
        "parameters": "140M",
        "context_length": 1024,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 6,
        "model_class": "BartForConditionalGeneration",
        "tokenizer_class": "BartTokenizer",
        "recommended_tasks": ["translation", "summarization", "question-answering"]
    }
}

class TestEncoderDecoderModel(ModelTest):
    """Test class for encoder-decoder models like T5, BART, etc."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "t5-small"
        
        # Verify model exists in registry
        if self.model_id not in ENCODER_DECODER_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = ENCODER_DECODER_MODELS_REGISTRY["t5-small"]
        else:
            self.model_info = ENCODER_DECODER_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "translation_en_to_fr"  # Default task - can be changed
        self.model_class = self.model_info["model_class"]
        self.tokenizer_class = self.model_info["tokenizer_class"]
        self.description = self.model_info["description"]
        
        # Define test inputs based on task
        if "translation" in self.task:
            self.test_text = "My name is Sarah and I live in London."
        elif "summarization" in self.task:
            self.test_text = "The quick brown fox jumps over the lazy dog. " * 10
        else:
            self.test_text = "What is the capital of France?"
        
        # Setup hardware detection
        self.setup_hardware()
    
    def setup_hardware(self):
        """Set up hardware detection."""
        try:
            # Try to import hardware detection capabilities
            from scripts.generators.hardware.hardware_detection import (
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
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to device if needed
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Setup task-specific prefixes for T5
        if self.model_info["model_type"] == "t5" and not input_text.startswith(self.task):
            if "translation" in self.task:
                # T5 needs task prefix for translation
                task_prefix = self.task.replace("_", " ") + ": "
                decoder_input_ids = tokenizer(task_prefix + input_text, return_tensors="pt").input_ids.to(self.device)
                # Store for generation
                self.decoder_prefix = task_prefix
            else:
                # Other tasks like summarization need their own prefixes
                task_prefix = self.task.replace("_", " ") + ": "
                decoder_input_ids = tokenizer(task_prefix + input_text, return_tensors="pt").input_ids.to(self.device)
                # Store for generation
                self.decoder_prefix = task_prefix
        else:
            # For non-T5 or if input already has prefix
            decoder_input_ids = None
            self.decoder_prefix = ""
        
        # Run inference
        with torch.no_grad():
            if decoder_input_ids is not None:
                outputs = model(**inputs, decoder_input_ids=decoder_input_ids[:, :1])
            else:
                outputs = model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Check for logits in output
        self.assertTrue(hasattr(outputs, "logits"))
        self.assertGreater(outputs.logits.shape[0], 0)
        
        logger.info(f"Basic inference successful: {outputs.logits.shape}")
    
    def test_generation(self):
        """Test text generation with the model."""
        import torch
        
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        tokenizer = model_components["tokenizer"]
        
        # Prepare input
        input_text = self.prepare_input()
        
        # Add task prefix for T5 if needed
        if self.model_info["model_type"] == "t5" and hasattr(self, 'decoder_prefix') and self.decoder_prefix:
            if not input_text.startswith(self.decoder_prefix):
                input_text = self.decoder_prefix + input_text
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=4,
                early_stopping=True,
                num_return_sequences=1
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Verify output
        self.assertIsNotNone(generated_text)
        self.assertGreater(len(generated_text), 0)
        
        logger.info(f"Generated text: {generated_text}")
        logger.info("Generation successful")
    
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
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference (simple forward pass only for hardware test)
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
    
    def test_pipeline_inference(self):
        """Test inference using the pipeline API."""
        try:
            import transformers
            
            # Select appropriate pipeline task
            pipeline_task = self.task
            if "translation" in self.task:
                pipeline_task = "translation"
            elif "summarization" in self.task:
                pipeline_task = "summarization"
            elif "question-answering" in self.task:
                pipeline_task = "question-answering"
            else:
                pipeline_task = "text2text-generation"
            
            # Initialize the pipeline
            pipe = transformers.pipeline(
                pipeline_task, 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Test with input text
            input_text = self.prepare_input()
            
            # Special handling for T5 translation
            if self.model_info["model_type"] == "t5" and "translation" in self.task:
                if not input_text.startswith(self.task.replace("_", " ")):
                    input_text = f"{self.task.replace('_', ' ')}: {input_text}"
            
            # Special handling for question-answering
            if pipeline_task == "question-answering":
                inputs = {
                    "question": "What is the capital of France?",
                    "context": "Paris is the capital of France. It is known for the Eiffel Tower."
                }
            else:
                inputs = input_text
            
            # Run inference
            outputs = pipe(inputs)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            
            # Log results based on task
            if pipeline_task == "translation":
                if isinstance(outputs, list):
                    translated_text = outputs[0]["translation_text"]
                else:
                    translated_text = outputs["translation_text"]
                logger.info(f"Translated text: {translated_text}")
            elif pipeline_task == "summarization":
                if isinstance(outputs, list):
                    summary = outputs[0]["summary_text"]
                else:
                    summary = outputs["summary_text"]
                logger.info(f"Summary: {summary}")
            elif pipeline_task == "question-answering":
                answer = outputs["answer"]
                logger.info(f"Answer: {answer}")
            else:
                if isinstance(outputs, list):
                    generated = outputs[0]["generated_text"]
                else:
                    generated = outputs["generated_text"]
                logger.info(f"Generated text: {generated}")
            
            logger.info("Pipeline inference successful")
            
        except Exception as e:
            logger.error(f"Error in pipeline inference: {e}")
            self.fail(f"Pipeline inference failed: {e}")
    
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
    
    parser = argparse.ArgumentParser(description="Test encoder-decoder models with refactored test suite")
    parser.add_argument("--model", type=str, default="t5-small", 
                       help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to test on (cpu, cuda, etc.)")
    parser.add_argument("--task", type=str, default="translation_en_to_fr", 
                       help="Task to test (translation_en_to_fr, summarization, etc.)")
    parser.add_argument("--save-results", action="store_true", help="Save test results to file")
    
    args = parser.parse_args()
    
    # Create test instance
    test = TestEncoderDecoderModel()
    
    # Override settings if specified
    if args.model:
        test.model_id = args.model
    if args.device:
        test.device = args.device
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