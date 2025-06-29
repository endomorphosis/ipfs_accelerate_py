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

import asyncio
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

# ROCm imports and detection
HAS_ROCM = False

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

    # Try ROCm detection with torch
    if torch.cuda.is_available() and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
        HAS_ROCM = True
        ROCM_VERSION = torch._C._rocm_version()
    elif 'ROCM_HOME' in os.environ:
        HAS_ROCM = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

try:
    import openvino
    from openvino.runtime import Core
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    logger.warning("OpenVINO not available")

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

# Main BERT model registry
FUYU_MODELS_REGISTRY = {
    "fuyu-base-uncased": {
        "full_name": "FUYU Base Uncased",
        "architecture": "encoder-only",
        "description": "FUYU Base model with uncased vocabulary",
        "fuyu": "fuyu",
        "parameters": "110M",
        "context_length": 512,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "recommended_tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"]
    },
    "fuyu-large-uncased": {
        "full_name": "FUYU Large Uncased",
        "architecture": "encoder-only",
        "description": "FUYU Large model with uncased vocabulary",
        "fuyu": "fuyu",
        "parameters": "336M",
        "context_length": 512,
        "embedding_dim": 1024,
        "attention_heads": 16,
        "layers": 24,
        "recommended_tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"]
    },
    "fuyu-base-cased": {
        "full_name": "FUYU Base Cased",
        "architecture": "encoder-only",
        "description": "FUYU Base model with cased vocabulary",
        "fuyu": "fuyu",
        "parameters": "110M",
        "context_length": 512,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "recommended_tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"]
    }
}

def select_device():
    """Select the best available device for inference."""
    if HAS_CUDA:
        return "cuda:0"
    elif HAS_ROCM:
        return "cuda:0"  # ROCm uses CUDA API
    elif HAS_MPS:
        return "mps"
    else:
        return "cpu"

# Create mock classes for testing without dependencies
class MockTokenizer:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 30522
        self.mask_token = "[MASK]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.all_special_tokens = [self.mask_token, self.pad_token, self.cls_token, self.sep_token, self.unk_token]
        self.mask_token_id = 103
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.unk_token_id = 100
        
    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        # Create a simple mock encoding with appropriate shape
        if isinstance(text, list):
            batch_size = len(text)
        else:
            batch_size = 1
        
        # Generate token ids for input
        input_ids = [[self.cls_token_id] + [i % 100 + 999 for i in range(10)] + [self.sep_token_id] for _ in range(batch_size)]
        
        # Find the mask token and replace with mask_token_id
        for i in range(batch_size):
            if "[MASK]" in (text[i] if isinstance(text, list) else text):
                # Replace a token with [MASK] token
                pos = 5  # Arbitrary position for mask token
                input_ids[i][pos] = self.mask_token_id
        
        attention_mask = [[1] * len(ids) for ids in input_ids]
        
        # Convert to tensors if requested
        if return_tensors == "pt" and HAS_TORCH:
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask)
            }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
    def decode(self, token_ids, skip_special_tokens=True):
        # Simple mock decode
        tokens = [f"token_{id}" for id in token_ids if id not in [self.cls_token_id, self.sep_token_id, self.pad_token_id]]
        return " ".join(tokens)
    
    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        return cls()

class MockSentencePieceProcessor:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 32000
        self.mask_token = "[MASK]"
        
    def encode(self, text, *args, **kwargs):
        # Return a list of "token ids"
        return [i for i in range(min(len(text.split()), 15))]

    def decode(self, token_ids):
        # Return a simple mock decoded string
        return " ".join([f"token_{id}" for id in token_ids])
        
    def get_piece(self, id):
        # Return a fake token
        return f"token_{id}"
        
    def id_to_piece(self, id):
        # Return a fake token
        return f"token_{id}"
        
    def piece_to_id(self, piece):
        # Return a fake id
        return hash(piece) % 32000
    
    @classmethod
    def load(cls, model_path):
        return cls()

class TestModelTypeModels:
    def __init__(self, model_id="fuyu-base-uncased", device=None):
        """Initialize the test class for FUYU models.
        
        Args:
            model_id: The model ID to test (default: "fuyu-base-uncased")
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
                
            logger.info(f"Testing FUYU model {self.model_id} with pipeline API on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Initialize the pipeline
            pipe = transformers.pipeline(
                "fill-mask", 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Test with a simple input
            test_input = "The quick brown fox jumps over the [MASK] dog."
            
            # Record inference start time
            inference_start = time.time()
            
            # Run inference
            outputs = pipe(test_input)
            
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Log results
            if isinstance(outputs, list) and len(outputs) > 0:
                top_prediction = outputs[0]['token_str'] if isinstance(outputs[0], dict) and 'token_str' in outputs[0] else "N/A"
                logger.info(f"Top prediction: {top_prediction}")
                logger.info(f"Inference time: {inference_time:.2f} seconds")
                
                # Store performance stats
                self.performance_stats["pipeline"] = {
                    "load_time": load_time,
                    "inference_time": inference_time,
                    "top_prediction": top_prediction
                }
                
                return {
                    "success": True,
                    "model_id": self.model_id,
                    "device": self.device,
                    "input": test_input,
                    "top_prediction": top_prediction,
                    "performance": {
                        "load_time": load_time,
                        "inference_time": inference_time
                    }
                }
            else:
                logger.error(f"Pipeline returned unexpected output: {outputs}")
                return {"success": False, "error": "Unexpected output format"}
                
        except Exception as e:
            logger.error(f"Error testing pipeline: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
            
    def test_from_pretrained(self):
        """Test the model using the from_pretrained API."""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning("Transformers or torch library not available, skipping from_pretrained test")
                return {"success": False, "error": "Required libraries not available"}
                
            logger.info(f"Testing FUYU model {self.model_id} with from_pretrained API on {self.device}")
            
            # Record start time
            start_time = time.time()
            
            # Load the model and tokenizer
            model = transformers.ModelTypeForMaskedLM.from_pretrained(self.model_id)
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            
            # Move the model to the appropriate device
            model = model.to(self.device)
            
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"Model loading time: {load_time:.2f} seconds")
            
            # Test with a simple input
            test_input = "The quick brown fox jumps over the [MASK] dog."
            
            # Tokenize the input
            inputs = tokenizer(test_input, return_tensors="pt")
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Record inference start time
            inference_start = time.time()
            
            # Get the position of the [MASK] token
            mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Record inference time
            inference_time = time.time() - inference_start
            
            # Get the top prediction
            logits = outputs.logits
            mask_token_logits = logits[0, mask_token_index, :]
            top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            top_tokens_words = [tokenizer.decode([token]).strip() for token in top_tokens]
            
            # Log results
            logger.info(f"Top predictions: {', '.join(top_tokens_words)}")
            logger.info(f"Inference time: {inference_time:.2f} seconds")
            
            # Store performance stats
            self.performance_stats["from_pretrained"] = {
                "load_time": load_time,
                "inference_time": inference_time,
                "top_predictions": top_tokens_words
            }
            
            return {
                "success": True,
                "model_id": self.model_id,
                "device": self.device,
                "input": test_input,
                "top_predictions": top_tokens_words,
                "performance": {
                    "load_time": load_time,
                    "inference_time": inference_time
                }
            }
                
        except Exception as e:
            logger.error(f"Error testing from_pretrained: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
            
    def test_openvino(self):
        """Test the model using OpenVINO."""
        try:
            if not HAS_OPENVINO:
                logger.warning("OpenVINO not available, skipping OpenVINO test")
                return {"success": False, "error": "OpenVINO not available"}
                
            if not HAS_TRANSFORMERS:
                logger.warning("Transformers library not available, skipping OpenVINO test")
                return {"success": False, "error": "Transformers library not available"}
                
            logger.info(f"Testing FUYU model {self.model_id} with OpenVINO")
            
            # Record start time
            start_time = time.time()
            
            # Load tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            
            # Initialize OpenVINO Core and load the model
            core = Core()
            model_path = os.path.join("openvino_model", f"{self.model_id.replace('/', '_')}.xml")
            
            # Check if optimized model exists, otherwise try to convert
            if not os.path.exists(model_path):
                logger.info(f"OpenVINO model not found at {model_path}, attempting to convert")
                
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                # Load the PyTorch model
                pt_model = transformers.ModelTypeForMaskedLM.from_pretrained(self.model_id)
                
                # Get example inputs
                example_input = "The quick brown fox jumps over the [MASK] dog."
                inputs = tokenizer(example_input, return_tensors="pt")
                
                # Convert model to ONNX
                onnx_path = model_path.replace(".xml", ".onnx")
                try:
                    import torch.onnx
                    torch.onnx.export(
                        pt_model,
                        tuple(inputs.values()),
                        onnx_path,
                        input_names=list(inputs.keys()),
                        output_names=["logits"],
                        dynamic_axes={
                            key: {0: "batch_size", 1: "sequence_length"} 
                            for key in inputs.keys()
                        }
                    )
                    logger.info(f"Converted model to ONNX format at {onnx_path}")
                    
                    # Convert ONNX to OpenVINO IR
                    from openvino.tools import mo
                    model_xml = mo.convert_model(onnx_path)
                    from openvino.runtime import serialize
                    serialize(model_xml, model_path)
                    logger.info(f"Converted ONNX to OpenVINO IR format at {model_path}")
                except Exception as e:
                    logger.error(f"Error converting model to OpenVINO format: {e}")
                    return {"success": False, "error": f"Model conversion failed: {str(e)}"}
            
            # Load the optimized model
            ov_model = core.read_model(model_path)
            compiled_model = core.compile_model(ov_model, "CPU")
    
            # Record model loading time
            load_time = time.time() - start_time
            logger.info(f"OpenVINO model loading time: {load_time:.2f} seconds")
    
            # Test with a simple input
            test_input = "The quick brown fox jumps over the [MASK] dog."
    
            # Tokenize the input
            inputs = tokenizer(test_input, return_tensors="pt")
    
            # Convert to numpy arrays
            inputs_np = {k: v.numpy() for k, v in inputs.items()}
    
            # Record inference start time
            inference_start = time.time()
    
            # Get the position of the [MASK] token
            mask_token_index = np.where(inputs_np["input_ids"][0] == tokenizer.mask_token_id)[0][0]
    
            # Forward pass
            logits = compiled_model(inputs_np)[0]
    
            # Record inference time
            inference_time = time.time() - inference_start
    
            # Get the top prediction
            mask_token_logits = logits[0, mask_token_index, :]
            top_indices = np.argsort(mask_token_logits)[-5:][::-1]
            top_tokens_words = [tokenizer.decode([token]) for token in top_indices]
    
            # Log results
            logger.info(f"OpenVINO top predictions: {', '.join(top_tokens_words)}")
            logger.info(f"OpenVINO inference time: {inference_time:.2f} seconds")
    
            # Store performance stats
            self.performance_stats["openvino"] = {
                "load_time": load_time,
                "inference_time": inference_time,
                "top_predictions": top_tokens_words
            }
    
            return {
                "success": True,
                "model_id": self.model_id,
                "device": "CPU (OpenVINO)",
                "input": test_input,
                "top_predictions": top_tokens_words,
                "performance": {
                    "load_time": load_time,
                    "inference_time": inference_time
                }
            }
                
        except Exception as e:
            logger.error(f"Error testing with OpenVINO: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
            
    def run_tests(self, all_hardware=False):
        """Run all tests for this model."""
        results = {}
    
        # Run pipeline test
        pipeline_result = self.test_pipeline()
        results["pipeline"] = pipeline_result
    
        # Run from_pretrained test
        from_pretrained_result = self.test_from_pretrained()
        results["from_pretrained"] = from_pretrained_result
    
        # Run OpenVINO test if requested and available
        if all_hardware and HAS_OPENVINO:
            openvino_result = self.test_openvino()
            results["openvino"] = openvino_result
    
        # Determine if real inference or mock objects were used
        using_real_inference = HAS_TRANSFORMERS and HAS_TORCH
        using_mocks = not using_real_inference or not HAS_TOKENIZERS or not HAS_SENTENCEPIECE
    
        # Add metadata
        results["metadata"] = {
            "model_id": self.model_id,
            "device": self.device,
            "timestamp": datetime.datetime.now().isoformat(),
            "hardware": {
                "cuda": HAS_CUDA,
                "rocm": HAS_ROCM,
                "openvino": HAS_OPENVINO,
                "mps": HAS_MPS,
                "webgpu": HAS_WEBGPU,
                "webnn": HAS_WEBNN
            },
            "dependencies": {
                "transformers": transformers.__version__ if HAS_TRANSFORMERS else None,
                "torch": torch.__version__ if HAS_TORCH else None,
                "numpy": np.__version__
            },
            "success": pipeline_result.get("success", False) or from_pretrained_result.get("success", False),
            "performance": self.performance_stats,
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH, 
            "has_tokenizers": HAS_TOKENIZERS,
            "has_sentencepiece": HAS_SENTENCEPIECE,
            "using_real_inference": using_real_inference,
            "using_mocks": using_mocks,
            "test_type": "REAL INFERENCE" if (using_real_inference and not using_mocks) else "MOCK OBJECTS (CI/CD)"
        }
    
        return results

def save_results(results, model_id=None, output_dir="collected_results"):
    """Save test results to a JSON file."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = model_id or results.get("metadata", {}).get("model_id", "unknown_model")
        model_id_safe = model_id.replace("/", "__")
        filename = f"hf_fuyu_{model_id_safe}_{timestamp}.json"
        file_path = os.path.join(output_dir, filename)
        
        # Save results to file
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test FUYU HuggingFace models")
    parser.add_argument("--model", type=str, default="fuyu-base-uncased", help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to run tests on (cuda, cpu, mps)")
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--list-models", action="store_true", help="List available FUYU models")

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print(f"\nAvailable FUYU models:")
        for model_id, info in FUYU_MODELS_REGISTRY.items():
            print(f"  - {model_id} ({info['full_name']})")
            print(f"      Parameters: {info['parameters']}, Embedding: {info['embedding_dim']}, Context: {info['context_length']}")
            print(f"      Description: {info['description']}")
            print()
        return

    # Initialize the test class
    bert_tester = TestModelTypeModels(model_id=args.model, device=args.device)

    # Run the tests
    results = bert_tester.run_tests(all_hardware=args.all_hardware)

    # Print a summary
    print("\n" + "="*50)
    print(f"TEST RESULTS SUMMARY")
    print("="*50)

    # Indicate real vs mock inference clearly
    using_real_inference = results["metadata"]["using_real_inference"]
    using_mocks = results["metadata"]["using_mocks"]

    if using_real_inference and not using_mocks:
        print(f"{GREEN}🚀 Using REAL INFERENCE with actual models{RESET}")
    else:
        print(f"{BLUE}🔷 Using MOCK OBJECTS for CI/CD testing only{RESET}")
        print(f"   Dependencies: transformers={HAS_TRANSFORMERS}, torch={HAS_TORCH}, tokenizers={HAS_TOKENIZERS}, sentencepiece={HAS_SENTENCEPIECE}")

    print(f"\nModel: {args.model}")
    print(f"Device: {bert_tester.device}")
    
    # Pipeline results
    pipeline_success = results["pipeline"].get("success", False)
    print(f"\nPipeline Test: {'✅ Success' if pipeline_success else '❌ Failed'}")
    if pipeline_success:
        top_prediction = results["pipeline"].get("top_prediction", "N/A")
        inference_time = results["pipeline"].get("performance", {}).get("inference_time", "N/A")
        print(f"  - Top prediction: {top_prediction}")
        print(f"  - Inference time: {inference_time:.2f} seconds" if isinstance(inference_time, (int, float)) else f"  - Inference time: {inference_time}")
    else:
        error = results["pipeline"].get("error", "Unknown error")
        print(f"  - Error: {error}")
    
    # from_pretrained results
    from_pretrained_success = results["from_pretrained"].get("success", False)
    print(f"\nFrom Pretrained Test: {'✅ Success' if from_pretrained_success else '❌ Failed'}")
    if from_pretrained_success:
        top_predictions = results["from_pretrained"].get("top_predictions", ["N/A"])
        inference_time = results["from_pretrained"].get("performance", {}).get("inference_time", "N/A")
        print(f"  - Top predictions: {', '.join(top_predictions[:3])}")
        print(f"  - Inference time: {inference_time:.2f} seconds" if isinstance(inference_time, (int, float)) else f"  - Inference time: {inference_time}")
    else:
        error = results["from_pretrained"].get("error", "Unknown error")
        print(f"  - Error: {error}")
    
    # OpenVINO results if available
    if "openvino" in results:
        openvino_success = results["openvino"].get("success", False)
        print(f"\nOpenVINO Test: {'✅ Success' if openvino_success else '❌ Failed'}")
        if openvino_success:
            top_predictions = results["openvino"].get("top_predictions", ["N/A"])
            inference_time = results["openvino"].get("performance", {}).get("inference_time", "N/A")
            print(f"  - Top predictions: {', '.join(top_predictions[:3])}")
            print(f"  - Inference time: {inference_time:.2f} seconds" if isinstance(inference_time, (int, float)) else f"  - Inference time: {inference_time}")
        else:
            error = results["openvino"].get("error", "Unknown error")
            print(f"  - Error: {error}")
    
    # Save results if requested
    if args.save:
        file_path = save_results(results, args.model)
        if file_path:
            print(f"\nResults saved to {file_path}")
    
    print(f"\nSuccessfully tested FUYU model: {args.model}")

if __name__ == "__main__":
    main()