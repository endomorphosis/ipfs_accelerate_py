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
    """
Class-based test file for all bert-family models.
This file provides a unified testing interface for:
    - BertModel
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")

# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")


# Try to import sentencepiece
try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")


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


if not HAS_SENTENCEPIECE:
    class MockSentencePieceProcessor:
        def __init__(self, *args, **kwargs):
            self.vocab_size = 32000
            
        def encode(self, text, out_type=str):
            return [1, 2, 3, 4, 5]
            
        def decode(self, ids):
            return "Decoded text from mock"
            
        def get_piece_size(self):
            return 32000
            
        @staticmethod
        def load(model_file):
            return MockSentencePieceProcessor()

    sentencepiece.SentencePieceProcessor = MockSentencePieceProcessor


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
        capabilities["cuda"] = torch.cuda.is_available(),
        if capabilities["cuda"]:
            capabilities["cuda_devices"] = torch.cuda.device_count(),
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
BERT_MODELS_REGISTRY = {
    "bert-base-uncased": {
        "description": "bert base model",
        "class": "BertModel",
    },
}

class TestBertModels:
    """Base test class for all bert-family models."""
    
    def __init__(self, model_id=None):
        """Initialize the test class for a specific model or default."""
        self.model_id = model_id or "bert-base-uncased"
        
        # Verify model exists in registry
        if self.model_id not in BERT_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = BERT_MODELS_REGISTRY["bert-base-uncased"]
        else:
            self.model_info = BERT_MODELS_REGISTRY[self.model_id]
            
        # Define model parameters
        self.task = "fill-mask"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
        self.test_text = "The man worked as a [MASK]."

        # Configure hardware preference
        # Check for CPU-only flag from command line args
        try:
            # Access command-line args safely (defined at module level)
            if 'args' in globals() and hasattr(args, 'cpu_only') and args.cpu_only:
                self.preferred_device = "cpu"
            elif HW_CAPABILITIES["cuda"]:
                self.preferred_device = "cuda"
            elif HW_CAPABILITIES["mps"]:
                self.preferred_device = "mps"
            else:
                self.preferred_device = "cpu"
        except:
            # Fallback if args are not defined
            if HW_CAPABILITIES["cuda"]:
                self.preferred_device = "cuda"
            elif HW_CAPABILITIES["mps"]:
                self.preferred_device = "mps"
            else:
                self.preferred_device = "cpu"
        
        logger.info(f"Using {self.preferred_device} as preferred device")
        
        # Results storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}

    def test_pipeline(self, device="auto"):
        """Test the model using transformers pipeline API."""
        if device == "auto":
            device = self.preferred_device
    
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
    
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
    
            if not HAS_TOKENIZERS:
                results["pipeline_error_type"] = "missing_dependency"
                results["pipeline_missing_deps"] = ["tokenizers>=0.11.0"]
                results["pipeline_success"] = False
                return results
    
            if not HAS_SENTENCEPIECE:
                results["pipeline_error_type"] = "missing_dependency"
                results["pipeline_missing_deps"] = ["sentencepiece>=0.1.91"]
                results["pipeline_success"] = False
                return results
    
    
        try:
            logger.info(f"Testing {self.model_id} with pipeline() on {device}...")
    
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": self.task,
                "model": self.model_id,
                "device": device
            }
    
            # Time the model loading
            load_start_time = time.time()
    
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
    
            # Prepare test input
            pipeline_input = self.test_text
    
            # Run warmup inference if on CUDA
            if device == "cuda":
                try:
                    _ = pipeline(pipeline_input)
                except Exception:
                    pass
    
            # Run multiple inference passes
            num_runs = 3
            times = []
            outputs = []
    
            for _ in range(num_runs):
                start_time = time.time()
                output = pipeline(pipeline_input)
                end_time = time.time()
                times.append(end_time - start_time)
                outputs.append(output)
    
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
    
            # Store results
            results["pipeline_success"] = True
            results["pipeline_avg_time"] = avg_time
            results["pipeline_min_time"] = min_time
            results["pipeline_max_time"] = max_time
            results["pipeline_load_time"] = load_time
            results["pipeline_error_type"] = "none"
    
            # Add to examples
            self.examples.append({
                "method": f"pipeline() on {device}",
                "input": str(pipeline_input),
                "output_preview": str(outputs[0])[:200] + "..." if len(str(outputs[0])) > 200 else str(outputs[0])
            })
    
            # Store in performance stats
            self.performance_stats[f"pipeline_{device}"] = {
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "load_time": load_time,
                "num_runs": num_runs
            }
    
        except Exception as e:
            # Store error information
            results["pipeline_success"] = False
            results["pipeline_error"] = str(e)
            results["pipeline_traceback"] = traceback.format_exc()
            logger.error(f"Error testing pipeline on {device}: {e}")
    
            # Classify error type
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
    
            if "cuda" in error_str or "cuda" in traceback_str:
                results["pipeline_error_type"] = "cuda_error"
            elif "memory" in error_str:
                results["pipeline_error_type"] = "out_of_memory"
            elif "no module named" in error_str:
                results["pipeline_error_type"] = "missing_dependency"
            else:
                results["pipeline_error_type"] = "other"
    
        # Add to overall results
        self.results[f"pipeline_{device}"] = results
        return results

    def test_from_pretrained(self, device="auto"):
        """Test the model using direct from_pretrained loading."""
        if device == "auto":
            device = self.preferred_device
    
        results = {
            "model": self.model_id,
            "device": device,
            "task": self.task,
            "class": self.class_name
        }
    
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_core"] = ["transformers"]
            results["from_pretrained_success"] = False
            return results
    
            if not HAS_TOKENIZERS:
                results["from_pretrained_error_type"] = "missing_dependency"
                results["from_pretrained_missing_deps"] = ["tokenizers>=0.11.0"]
                results["from_pretrained_success"] = False
                return results
    
            if not HAS_SENTENCEPIECE:
                results["from_pretrained_error_type"] = "missing_dependency"
                results["from_pretrained_missing_deps"] = ["sentencepiece>=0.1.91"]
                results["from_pretrained_success"] = False
                return results
    
    
        try:
            logger.info(f"Testing {self.model_id} with from_pretrained() on {device}...")
    
            # Common parameters for loading
            pretrained_kwargs = {
                "local_files_only": False
            }
    
                # Time tokenizer loading
                tokenizer_load_start = time.time()
                tokenizer = transformers.BertTokenizer.from_pretrained(
                    self.model_id,
                    **pretrained_kwargs
                )
                tokenizer_load_time = time.time() - tokenizer_load_start
    
                # Use appropriate model class based on model type
                        model_class = None
                        if self.class_name == "BertModel":
                            model_class = transformers.BertModel
                        else:
                            # Fallback to Auto class
                            model_class = transformers.AutoModelForSequenceClassification
    
                        # Time model loading
                        model_load_start = time.time()
                        model = model_class.from_pretrained(
                            self.model_id,
                            **pretrained_kwargs
                        )
                        model_load_time = time.time() - model_load_start
    
            # Move model to device
            if device != "cpu":
                model = model.to(device)
    
            # Prepare test input
                test_input = self.test_text
    
                        # Tokenize input
                        inputs = tokenizer(test_input, return_tensors="pt")
    
            # Move inputs to device
            if device != "cpu":
                inputs = {key: val.to(device) for key, val in inputs.items()}
    
            # Run warmup inference if using CUDA
            if device == "cuda":
                try:
                    with torch.no_grad():
                        _ = model(**inputs)
                except Exception:
                    pass
    
            # Run multiple inference passes
            num_runs = 3
            times = []
            outputs = []
    
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    output = model(**inputs)
                    end_time = time.time()
                    times.append(end_time - start_time)
                    outputs.append(output)
    
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
    
            # Process output
                # Process masked language modeling output
                        if hasattr(outputs[0], "logits"):
                            logits = outputs[0].logits
                            # Get predictions for masked tokens
                            predicted_token_ids = torch.argmax(logits, dim=-1)
    
                            if hasattr(tokenizer, "decode"):
                                predictions = [tokenizer.decode(token_ids) for token_ids in predicted_token_ids]
                                predictions = [{"token": pred, "score": 1.0} for pred in predictions[:5]]
                            else:
                                predictions = [{"token": "<mask>", "score": 1.0}]
                        else:
                            predictions = [{"token": "<mask>", "score": 1.0}]
    
            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = (param_count * 4) / (1024 * 1024)  # Rough size in MB
    
            # Store results
            results["from_pretrained_success"] = True
            results["from_pretrained_avg_time"] = avg_time
            results["from_pretrained_min_time"] = min_time
            results["from_pretrained_max_time"] = max_time
            results["tokenizer_load_time"] = tokenizer_load_time
            results["model_load_time"] = model_load_time
            results["model_size_mb"] = model_size_mb
            results["from_pretrained_error_type"] = "none"
    
            # Add predictions if available
            if 'predictions' in locals():
                results["predictions"] = predictions
    
            # Add to examples
            example_data = {
                "method": f"from_pretrained() on {device}",
                "input": str(test_input)
            }
    
            if 'predictions' in locals():
                example_data["predictions"] = predictions
    
            self.examples.append(example_data)
    
            # Store in performance stats
            self.performance_stats[f"from_pretrained_{device}"] = {
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "tokenizer_load_time": tokenizer_load_time,
                "model_load_time": model_load_time,
                "model_size_mb": model_size_mb,
                "num_runs": num_runs
            }
    
        except Exception as e:
            # Store error information
            results["from_pretrained_success"] = False
            results["from_pretrained_error"] = str(e)
            results["from_pretrained_traceback"] = traceback.format_exc()
            logger.error(f"Error testing from_pretrained on {device}: {e}")
    
            # Classify error type
            error_str = str(e).lower()
            traceback_str = traceback.format_exc().lower()
    
            if "cuda" in error_str or "cuda" in traceback_str:
                results["from_pretrained_error_type"] = "cuda_error"
            elif "memory" in error_str:
                results["from_pretrained_error_type"] = "out_of_memory"
            elif "no module named" in error_str:
                results["from_pretrained_error_type"] = "missing_dependency"
            else:
                results["from_pretrained_error_type"] = "other"
    
        # Add to overall results
        self.results[f"from_pretrained_{device}"] = results
        return results

    def run_tests(self, all_hardware=False):
        """
        Run all tests for this model.
    
        Args:
            all_hardware: If True, tests on all available hardware (CPU, CUDA, OpenVINO)
    
        Returns:
            Dict containing test results
        """
        # Always test on default device
        self.test_pipeline()
        self.test_from_pretrained()
    
        # Test on all available hardware if requested
        if all_hardware:
            # Always test on CPU
            if self.preferred_device != "cpu":
                self.test_pipeline(device="cpu")
                self.test_from_pretrained(device="cpu")
    
            # Test on CUDA if available
            if HW_CAPABILITIES["cuda"] and self.preferred_device != "cuda":
                self.test_pipeline(device="cuda")
                self.test_from_pretrained(device="cuda")
    
            # Test on OpenVINO if available
            if HW_CAPABILITIES["openvino"]:
                self.test_with_openvino()
    
        # Build final results
        return {
            "results": self.results,
            "examples": self.examples,
            "performance": self.performance_stats,
            "hardware": HW_CAPABILITIES,
            "metadata": {
                "model": self.model_id,
                "task": self.task,
                "class": self.class_name,
                "description": self.description,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH
            }
        }


def save_results(model_id, results, output_dir="collected_results"):
    """Save test results to a file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    filename = f"hf_bert_{safe_model_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    return output_path

def get_available_models():
    """Get a list of all available bert models in the registry."""
    return list(BERT_MODELS_REGISTRY.keys())

def test_all_models(output_dir="collected_results", all_hardware=False):
    """Test all registered bert models."""
    models = get_available_models()
    results = {}
    
    for model_id in models:
        logger.info(f"Testing model: {model_id}")
        tester = TestBertModels(model_id)
        model_results = tester.run_tests(all_hardware=all_hardware)
        
        # Save individual results
        save_results(model_id, model_results, output_dir=output_dir)
        
        # Add to summary
        results[model_id] = {
            "success": any((r.get("pipeline_success", False) for r in model_results["results"].values() 
                        if r.get("pipeline_success") is not False))
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, f"hf_bert_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")
    return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test bert-family models")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Specific model to test")
    model_group.add_argument("--all-models", action="store_true", help="Test all registered models")
    
    # Hardware options
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for output files")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    # List options
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = get_available_models()
        print("\nAvailable bert-family models:")
        for model in models:
            info = BERT_MODELS_REGISTRY[model]
            print(f"  - {model} ({info['class']}): {info['description']}")
        return
    
    # Create output directory if needed
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test all models if requested
    if args.all_models:
        results = test_all_models(output_dir=args.output_dir, all_hardware=args.all_hardware)
        
        # Print summary
        print("\nBert Models Testing Summary:")
        total = len(results)
        successful = sum(1 for r in results.values() if r["success"])
        print(f"Successfully tested {successful} of {total} models ({successful/total*100:.1f}%)")
        return
    
    # Test single model (default or specified)
    model_id = args.model or "bert-base-uncased"
    logger.info(f"Testing model: {model_id}")
    
    # Make args available globally for hardware detection in the test class
    global args
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("CPU-only mode enabled - disabled CUDA")
    
    # Run test
    tester = TestBertModels(model_id)
    results = tester.run_tests(all_hardware=args.all_hardware)
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = any((r.get("pipeline_success", False) for r in results["results"].values() 
                 if r.get("pipeline_success") is not False))
    
    print("\nTEST RESULTS SUMMARY:")
    if success:
        print(f"✅ Successfully tested {model_id}")
        
        # Print performance highlights
        for device, stats in results["performance"].items():
            if "avg_time" in stats:
                print(f"  - {device}: {stats['avg_time']:.4f}s average inference time")
        
        # Print example outputs if available
        if results.get("examples") and len(results["examples"]) > 0:
            print("\nExample output:")
            example = results["examples"][0]
            if "predictions" in example:
                print(f"  Input: {example['input']}")
                print(f"  Predictions: {example['predictions']}")
            elif "output_preview" in example:
                print(f"  Input: {example['input']}")
                print(f"  Output: {example['output_preview']}")
    else:
        print(f"❌ Failed to test {model_id}")
        
        # Print error information
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {test_name}: {result.get('pipeline_error_type', 'unknown')}")
                print(f"    {result.get('pipeline_error', 'Unknown error')}")
    
        print("\nFor detailed results, use --save flag and check the JSON output file.")

if __name__ == "__main__":
    main()
