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
import traceback
import logging
import argparse
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# ANSI color codes for terminal output
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np

# WebGPU imports and mock setup
HAS_WEBGPU = False
try:
    # Attempt to check for WebGPU availability
    import ctypes.util
    HAS_WEBGPU = hasattr(ctypes.util, 'find_library') and ctypes.util.find_library('webgpu') is not None
except ImportError:
    HAS_WEBGPU = False

# WebNN imports and mock setup
HAS_WEBNN = False
try:
    # Attempt to check for WebNN availability
    import ctypes
    HAS_WEBNN = hasattr(ctypes.util, 'find_library') and ctypes.util.find_library('webnn') is not None
except ImportError:
    HAS_WEBNN = False

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

# ROCm detection
HAS_ROCM = False
try:
    if HAS_TORCH and torch.cuda.is_available() and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
        HAS_ROCM = True
        ROCM_VERSION = torch._C._rocm_version()
        logger.info(f"ROCm available: version {ROCM_VERSION}")
    elif 'ROCM_HOME' in os.environ:
        HAS_ROCM = True
        logger.info("ROCm available (detected via ROCM_HOME)")
except Exception as e:
    HAS_ROCM = False
    logger.info(f"ROCm detection failed: {e}")

# OpenVINO detection
try:
    import openvino
    from openvino.runtime import Core
    HAS_OPENVINO = True
    logger.info(f"OpenVINO available: version {openvino.__version__}")
except ImportError:
    HAS_OPENVINO = False
    logger.info("OpenVINO not available")

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        cuda_version = torch.version.cuda
        logger.info(f"CUDA available: version {cuda_version}")
        num_devices = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {num_devices}")
        
        # Log CUDA device properties
        for i in range(num_devices):
            device_props = torch.cuda.get_device_properties(i)
            logger.info(f"CUDA Device {i}: {device_props.name} with {device_props.total_memory / 1024**3:.2f} GB memory")
    else:
        logger.info("CUDA not available")
else:
    HAS_CUDA = False
    logger.info("CUDA detection skipped (torch not available)")

# MPS (Apple Silicon) detection
if HAS_TORCH:
    HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if HAS_MPS:
        logger.info("MPS available for Apple Silicon acceleration")
    else:
        logger.info("MPS not available")
else:
    HAS_MPS = False
    logger.info("MPS detection skipped (torch not available)")

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

# LED Models registry - Maps model IDs to their specific configurations
LED_MODELS_REGISTRY = {
    "allenai/led-base-16384": {
        "description": "LED base model with 16K context length",
        "class": "LEDForConditionalGeneration",
        "context_length": 16384,
        "parameters": "220M"
    },
    "allenai/led-large-16384": {
        "description": "LED large model with 16K context length",
        "class": "LEDForConditionalGeneration",
        "context_length": 16384,
        "parameters": "440M"
    },
    "patrickvonplaten/led-large-16384-pubmed": {
        "description": "LED large model fine-tuned on PubMed",
        "class": "LEDForConditionalGeneration",
        "context_length": 16384,
        "parameters": "440M",
        "domain": "medical"
    }
}

class TestLEDModels:
    """Base test class for all LED-family models."""
    
    def __init__(self, model_id=None):
        """Initialize the test class for a specific model or default."""
        self.model_id = model_id or "allenai/led-base-16384"
        
        # Verify model exists in registry
        if self.model_id not in LED_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = LED_MODELS_REGISTRY["allenai/led-base-16384"]
        else:
            self.model_info = LED_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "summarization"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        self.context_length = self.model_info["context_length"]
        
        # Define test inputs
        self.test_text = "Summarize: LED (Longformer Encoder-Decoder) is a transformer model designed for long document summarization that combines the Longformer's efficient attention mechanism with a standard transformer decoder. LED can process documents with up to 16,384 tokens, which is much longer than traditional transformer models. The model uses a combination of local windowed attention and global attention on specific tokens to efficiently process long documents."
        self.test_texts = [
            "Summarize: LED (Longformer Encoder-Decoder) is a transformer model designed for long document summarization that combines the Longformer's efficient attention mechanism with a standard transformer decoder. LED can process documents with up to 16,384 tokens, which is much longer than traditional transformer models. The model uses a combination of local windowed attention and global attention on specific tokens to efficiently process long documents.",
            "Summarize: The Longformer Encoder-Decoder (LED) was developed by the Allen Institute for AI (AI2) and builds upon the Longformer architecture. It addresses the context length limitations of standard transformer models by using an efficient attention mechanism that scales linearly with sequence length. This makes it particularly suitable for tasks that require processing long documents like summarization, question answering, and document-level natural language understanding."
        ]
        
        # Configure hardware preference
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
            "class": self.class_name,
            "context_length": self.context_length
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_core"] = ["transformers"]
            results["pipeline_success"] = False
            return results
            
        if not HAS_TOKENIZERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_deps"] = ["tokenizers"]
            results["pipeline_success"] = False
            return results
        
        try:
            logger.info(f"Testing {self.model_id} with pipeline() on {device}...")
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "task": "summarization",
                "model": self.model_id,
                "device": device if device != "cpu" else -1
            }
            
            # Time the model loading
            load_start_time = time.time()
            
            # Create pipeline with LED-specific configuration
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            
            # Use custom pipeline instead of the default one
            pipeline_kwargs["model"] = model
            pipeline_kwargs["tokenizer"] = tokenizer
            
            pipeline = transformers.pipeline(**pipeline_kwargs)
            load_time = time.time() - load_start_time
            
            # Prepare test input
            pipeline_input = self.test_text
            
            # Generation parameters for summarization
            gen_kwargs = {
                "max_length": 150,
                "min_length": 40,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "length_penalty": 2.0
            }
            
            # Run warmup inference if on CUDA
            if device == "cuda":
                try:
                    _ = pipeline(pipeline_input, **gen_kwargs)
                except Exception:
                    pass
            
            # Run multiple inference passes
            num_runs = 3
            times = []
            outputs = []
            
            for _ in range(num_runs):
                start_time = time.time()
                output = pipeline(pipeline_input, **gen_kwargs)
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
            "class": self.class_name,
            "context_length": self.context_length
        }
        
        # Check for dependencies
        if not HAS_TRANSFORMERS:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_core"] = ["transformers"]
            results["from_pretrained_success"] = False
            return results
            
        if not HAS_TOKENIZERS:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_deps"] = ["tokenizers"]
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
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
                **pretrained_kwargs
            )
            tokenizer_load_time = time.time() - tokenizer_load_start
            
            # Use appropriate model class based on model type
            model_class = None
            if self.class_name == "LEDForConditionalGeneration":
                try:
                    model_class = transformers.LEDForConditionalGeneration
                except:
                    # Fallback to Auto class
                    model_class = transformers.AutoModelForSeq2SeqLM
            else:
                # Fallback to Auto class
                model_class = transformers.AutoModelForSeq2SeqLM
            
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
            inputs = tokenizer(test_input, return_tensors="pt", max_length=1024, truncation=True)
            
            # Create global attention mask for LED
            # Set global attention on the first token (CLS token)
            global_attention_mask = torch.zeros_like(inputs["input_ids"])
            global_attention_mask[:, 0] = 1
            inputs["global_attention_mask"] = global_attention_mask
            
            logger.info("Added global_attention_mask for LED model")
            
            # Add decoder inputs for LED models
            decoder_input_ids = tokenizer("", return_tensors="pt")["input_ids"]
            inputs["decoder_input_ids"] = decoder_input_ids
            
            logger.info("Added empty decoder_input_ids for LED model")
            
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
            
            # Process generation output
            if hasattr(outputs[0], "logits"):
                logits = outputs[0].logits
                generated_ids = torch.argmax(logits, dim=-1)
                if hasattr(tokenizer, "decode"):
                    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    predictions = [{"generated_text": decoded_output}]
                else:
                    predictions = [{"generated_text": "Mock generated text"}]
            else:
                predictions = [{"generated_text": "Mock generated text"}]
            
            # Test generation method
            try:
                generation_start_time = time.time()
                generation_output = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    global_attention_mask=inputs.get("global_attention_mask"),
                    max_length=150,
                    min_length=40,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
                generation_time = time.time() - generation_start_time
                
                # Decode generation
                if hasattr(tokenizer, "decode"):
                    generated_summary = tokenizer.decode(generation_output[0], skip_special_tokens=True)
                    generation_result = {"summary": generated_summary}
                else:
                    generation_result = {"summary": "Mock generated summary"}
                
                # Add to results
                results["generation_output"] = generation_result
                results["generation_time"] = generation_time
            except Exception as e:
                logger.warning(f"Generation method failed: {e}")
                results["generation_error"] = str(e)
            
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
            
            if 'generation_result' in locals():
                example_data["generation"] = generation_result
            
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
            
            if 'generation_time' in locals():
                self.performance_stats[f"from_pretrained_{device}"]["generation_time"] = generation_time
            
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
        
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS
        
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
                "context_length": self.context_length,
                "timestamp": datetime.datetime.now().isoformat(),
                "has_transformers": HAS_TRANSFORMERS,
                "has_torch": HAS_TORCH,
                "has_tokenizers": HAS_TOKENIZERS,
                "using_real_inference": using_real_inference,
                "using_mocks": using_mocks,
                "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
            }
        }
        
def save_results(model_id, results, output_dir="collected_results"):
    """Save test results to a file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    filename = f"hf_led_{safe_model_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    return output_path
    
def get_available_models():
    """Get a list of all available LED models in the registry."""
    return list(LED_MODELS_REGISTRY.keys())
    
def test_all_models(output_dir="collected_results", all_hardware=False):
    """Test all registered LED models."""
    models = get_available_models()
    results = {}
    
    for model_id in models:
        logger.info(f"Testing model: {model_id}")
        tester = TestLEDModels(model_id)
        model_results = tester.run_tests(all_hardware=all_hardware)
        
        # Save individual results
        save_results(model_id, model_results, output_dir=output_dir)
        
        # Add to summary
        results[model_id] = {
            "success": any(r.get("pipeline_success", False) for r in model_results["results"].values()
                if isinstance(r, dict) and "pipeline_success" in r)
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, f"hf_led_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")
    return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Test LED models")
    
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
        print(f"\nAvailable LED models:")
        for model in models:
            info = LED_MODELS_REGISTRY[model]
            print(f"  - {model} ({info.get('parameters', 'unknown parameters')}):")
            print(f"      {info['description']}")
            print(f"      Context Length: {info['context_length']}")
        return
    
    # Create output directory if needed
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test all models if requested
    if args.all_models:
        results = test_all_models(output_dir=args.output_dir, all_hardware=args.all_hardware)
        
        # Print summary
        print(f"\nLED Models Testing Summary:")
        total = len(results)
        successful = sum(1 for r in results.values() if r["success"])
        print(f"Successfully tested {successful} of {total} models ({successful/total*100:.1f}%)")
        return
    
    # Test single model (default or specified)
    model_id = args.model or "allenai/led-base-16384"
    logger.info(f"Testing model: {model_id}")
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Run test
    tester = TestLEDModels(model_id)
    results = tester.run_tests(all_hardware=args.all_hardware)
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = False
    for r in results["results"].values():
        if isinstance(r, dict) and r.get("pipeline_success", False):
            success = True
            break
    
    # Determine if real inference or mock objects were used
    using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
    using_mocks = not using_real_inference or not HAS_TOKENIZERS
    
    print("\n" + "="*50)
    print(f"TEST RESULTS SUMMARY")
    print("="*50)
    
    # Indicate real vs mock inference clearly
    if using_real_inference and not using_mocks:
        print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")
    else:
        print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}")
    
    # Print model information
    print(f"\nModel: {model_id}")
    print(f"Description: {tester.description}")
    print(f"Context Length: {tester.context_length} tokens")
    print(f"Device: {tester.preferred_device}")
    
    if success:
        print(f"\n✅ Successfully tested LED model")
        
        # Print performance highlights
        if results["performance"]:
            print("\nPerformance Stats:")
            for test_name, stats in results["performance"].items():
                if "avg_time" in stats:
                    print(f"  - {test_name}: {stats['avg_time']:.4f}s average inference time")
                if "model_load_time" in stats:
                    print(f"      Model load time: {stats['model_load_time']:.2f}s")
                if "generation_time" in stats:
                    print(f"      Generation time: {stats['generation_time']:.4f}s")
        
        # Print example outputs
        if results["examples"]:
            example = results["examples"][0]
            print(f"\nExample:")
            print(f"  Input: {example['input'][:100]}..." if len(example['input']) > 100 else f"  Input: {example['input']}")
            
            if "output_preview" in example:
                print(f"  Output: {example['output_preview']}")
            elif "predictions" in example:
                print(f"  Predictions: {example['predictions']}")
            elif "generation" in example and "summary" in example["generation"]:
                print(f"  Generated Summary: {example['generation']['summary']}")
    else:
        print(f"\n❌ Failed to test LED model")
        
        # Print error information
        for test_name, result in results["results"].items():
            if isinstance(result, dict):
                if "pipeline_error" in result:
                    print(f"  - Error in {test_name}: {result.get('pipeline_error_type', 'unknown')}")
                    print(f"    {result.get('pipeline_error', 'Unknown error')}")
                elif "from_pretrained_error" in result:
                    print(f"  - Error in {test_name}: {result.get('from_pretrained_error_type', 'unknown')}")
                    print(f"    {result.get('from_pretrained_error', 'Unknown error')}")

if __name__ == "__main__":
    main()