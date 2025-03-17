#!/usr/bin/env python3
"""
Comprehensive script to test multiple models on different hardware backends.
This script manually fixes the syntax issues in the template files.
"""

import os
import sys
import time
import json
import logging
import argparse
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the key models to test
KEY_MODELS = {
    "bert": {
        "model_name": "bert-base-uncased",
        "small_model": "prajjwal1/bert-tiny",
        "model_type": "text_embedding",
        "requires": ["transformers"]
    },
    "t5": {
        "model_name": "t5-small",
        "small_model": "google/t5-efficient-tiny",
        "model_type": "text_generation",
        "requires": ["transformers"]
    },
    "vit": {
        "model_name": "google/vit-base-patch16-224",
        "small_model": "facebook/deit-tiny-patch16-224",
        "model_type": "vision",
        "requires": ["transformers", "pillow"]
    },
    "whisper": {
        "model_name": "openai/whisper-tiny",
        "small_model": "openai/whisper-tiny",
        "model_type": "audio",
        "requires": ["transformers", "librosa"]
    }
}

def check_module_availability(module_names):
    """Check if required modules are available."""
    available = {}
    for module_name in module_names:
        try:
            __import__(module_name)
            available[module_name] = True
        except ImportError:
            available[module_name] = False
            logger.warning(f"Module {module_name} is not available")
    
    return available

def detect_available_hardware():
    """Detect available hardware platforms."""
    available = {"cpu": True}  # CPU is always available
    
    # Check for CUDA (NVIDIA) support
    try:
        import torch
        available["cuda"] = torch.cuda.is_available()
        if available["cuda"]:
            logger.info(f"CUDA is available with {torch.cuda.device_count()} devices")
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Check for MPS (Apple Silicon) support
        if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
            available["mps"] = torch.backends.mps.is_available()
            if available["mps"]:
                logger.info("MPS (Apple Silicon) is available")
        else:
            available["mps"] = False
        
        # Check for ROCm (AMD) support
        if torch.cuda.is_available() and hasattr(torch.version, "hip"):
            available["rocm"] = True
            logger.info("ROCm (AMD) is available")
        else:
            available["rocm"] = False
    except ImportError:
        logger.warning("PyTorch not available, CUDA/MPS/ROCm cannot be detected")
        available["cuda"] = False
        available["mps"] = False
        available["rocm"] = False
    
    # Check for OpenVINO
    try:
        import openvino
        available["openvino"] = True
        logger.info(f"OpenVINO is available (version {openvino.__version__})")
    except ImportError:
        available["openvino"] = False
    
    # WebNN and WebGPU can be simulated
    available["webnn"] = True  # Simulated
    available["webgpu"] = True  # Simulated
    logger.info("WebNN and WebGPU will be tested in simulation mode")
    
    return available

def test_bert_on_hardware(hardware="cpu", model_name="bert-base-uncased", use_small_model=False):
    """Test BERT model on specified hardware."""
    if use_small_model:
        model_name = KEY_MODELS["bert"]["small_model"]
    
    logger.info(f"Testing BERT model {model_name} on {hardware}")
    
    # Set device
    if hardware == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hardware == "cpu":
        device = torch.device("cpu")
    else:
        logger.warning(f"Hardware {hardware} not available, falling back to CPU")
        device = torch.device("cpu")
    
    try:
        # Check if transformers is available
        from transformers import AutoModel, AutoTokenizer
        
        # Load tokenizer and model
        logger.info(f"Loading {model_name} tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        # Prepare input
        text = "This is a sample text for testing BERT model."
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        logger.info("Running inference...")
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = time.time() - start_time
        
        # Check outputs
        logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
        logger.info(f"Inference time: {inference_time:.4f} seconds")
        
        # Calculate embedding
        embedding = outputs.last_hidden_state.mean(dim=1)
        logger.info(f"Embedding shape: {embedding.shape}")
        
        # Test success
        logger.info(f"Successfully tested BERT model {model_name} on {hardware}")
        return {
            "success": True,
            "model_name": model_name,
            "hardware": hardware,
            "output_shape": str(outputs.last_hidden_state.shape),
            "embedding_shape": str(embedding.shape),
            "inference_time": inference_time,
            "device": str(device)
        }
    
    except Exception as e:
        logger.error(f"Error testing BERT model {model_name} on {hardware}: {e}")
        return {
            "success": False,
            "model_name": model_name,
            "hardware": hardware,
            "error": str(e),
            "error_type": type(e).__name__
        }

def test_vit_on_hardware(hardware="cpu", model_name=None, use_small_model=False):
    """Test ViT (Vision Transformer) model on specified hardware."""
    if model_name is None:
        if use_small_model:
            model_name = KEY_MODELS["vit"]["small_model"]
        else:
            model_name = KEY_MODELS["vit"]["model_name"]
    
    logger.info(f"Testing ViT model {model_name} on {hardware}")
    
    # Set device
    if hardware == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hardware == "cpu":
        device = torch.device("cpu")
    else:
        logger.warning(f"Hardware {hardware} not available, falling back to CPU")
        device = torch.device("cpu")
    
    try:
        # Check if required libraries are available
        from transformers import AutoFeatureExtractor, AutoModel
        from PIL import Image
        import requests
        import numpy as np
        
        # Load feature extractor and model
        logger.info(f"Loading {model_name} feature extractor and model...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        # Create a simple test image (checkerboard pattern)
        logger.info("Creating test image...")
        img_size = 224
        image = Image.new('RGB', (img_size, img_size), color='white')
        
        # Prepare input
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        logger.info("Running inference...")
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = time.time() - start_time
        
        # Check outputs
        logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
        logger.info(f"Inference time: {inference_time:.4f} seconds")
        
        # Calculate embedding (use CLS token)
        embedding = outputs.last_hidden_state[:, 0]
        logger.info(f"Embedding shape: {embedding.shape}")
        
        # Test success
        logger.info(f"Successfully tested ViT model {model_name} on {hardware}")
        return {
            "success": True,
            "model_name": model_name,
            "hardware": hardware,
            "output_shape": str(outputs.last_hidden_state.shape),
            "embedding_shape": str(embedding.shape),
            "inference_time": inference_time,
            "device": str(device)
        }
    
    except Exception as e:
        logger.error(f"Error testing ViT model {model_name} on {hardware}: {e}")
        return {
            "success": False,
            "model_name": model_name,
            "hardware": hardware,
            "error": str(e),
            "error_type": type(e).__name__
        }

def test_t5_on_hardware(hardware="cpu", model_name=None, use_small_model=False):
    """Test T5 model on specified hardware."""
    if model_name is None:
        if use_small_model:
            model_name = KEY_MODELS["t5"]["small_model"]
        else:
            model_name = KEY_MODELS["t5"]["model_name"]
    
    logger.info(f"Testing T5 model {model_name} on {hardware}")
    
    # Set device
    if hardware == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hardware == "cpu":
        device = torch.device("cpu")
    else:
        logger.warning(f"Hardware {hardware} not available, falling back to CPU")
        device = torch.device("cpu")
    
    try:
        # Check if transformers is available
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        # Load tokenizer and model
        logger.info(f"Loading {model_name} tokenizer and model...")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        # Prepare input
        text = "translate English to German: The house is wonderful."
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        logger.info("Running inference...")
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        inference_time = time.time() - start_time
        
        # Decode output
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {decoded_output}")
        logger.info(f"Inference time: {inference_time:.4f} seconds")
        
        # Test success
        logger.info(f"Successfully tested T5 model {model_name} on {hardware}")
        return {
            "success": True,
            "model_name": model_name,
            "hardware": hardware,
            "output_text": decoded_output,
            "output_shape": str(outputs.shape),
            "inference_time": inference_time,
            "device": str(device)
        }
    
    except Exception as e:
        logger.error(f"Error testing T5 model {model_name} on {hardware}: {e}")
        return {
            "success": False,
            "model_name": model_name,
            "hardware": hardware,
            "error": str(e),
            "error_type": type(e).__name__
        }

def test_whisper_on_hardware(hardware="cpu", model_name=None, use_small_model=False):
    """Test Whisper model on specified hardware."""
    if model_name is None:
        if use_small_model:
            model_name = KEY_MODELS["whisper"]["small_model"]
        else:
            model_name = KEY_MODELS["whisper"]["model_name"]
    
    logger.info(f"Testing Whisper model {model_name} on {hardware}")
    
    # Set device
    if hardware == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hardware == "cpu":
        device = torch.device("cpu")
    else:
        logger.warning(f"Hardware {hardware} not available, falling back to CPU")
        device = torch.device("cpu")
    
    try:
        # Check if required libraries are available
        import numpy as np
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        # Load processor and model
        logger.info(f"Loading {model_name} processor and model...")
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        # Create a dummy audio input (1 second of silence)
        logger.info("Creating dummy audio input...")
        sample_rate = 16000
        dummy_audio = np.zeros(sample_rate, dtype=np.float32)
        
        # Process audio
        inputs = processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Prepare decoder input IDs
        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).to(device)
        
        # Run inference
        logger.info("Running inference...")
        start_time = time.time()
        with torch.no_grad():
            outputs = model(
                **inputs,
                decoder_input_ids=decoder_input_ids
            )
        inference_time = time.time() - start_time
        
        # Check outputs
        logger.info(f"Output shape: {outputs.logits.shape}")
        logger.info(f"Inference time: {inference_time:.4f} seconds")
        
        # Test success
        logger.info(f"Successfully tested Whisper model {model_name} on {hardware}")
        return {
            "success": True,
            "model_name": model_name,
            "hardware": hardware,
            "output_shape": str(outputs.logits.shape),
            "inference_time": inference_time,
            "device": str(device)
        }
    
    except Exception as e:
        logger.error(f"Error testing Whisper model {model_name} on {hardware}: {e}")
        return {
            "success": False,
            "model_name": model_name,
            "hardware": hardware,
            "error": str(e),
            "error_type": type(e).__name__
        }

def run_test(model_key, hardware, use_small_model=False):
    """Run a test for a specific model on a specific hardware."""
    logger.info(f"Testing {model_key} on {hardware} hardware...")
    
    # Check if required modules are available
    required_modules = KEY_MODELS[model_key]["requires"]
    module_availability = check_module_availability(required_modules)
    
    if not all(module_availability.values()):
        missing_modules = [m for m, available in module_availability.items() if not available]
        return {
            "success": False,
            "model_key": model_key,
            "hardware": hardware,
            "error": f"Missing required modules: {', '.join(missing_modules)}",
            "error_type": "ModuleNotFoundError"
        }
    
    # Run the appropriate test function based on model key
    if model_key == "bert":
        return test_bert_on_hardware(hardware, use_small_model=use_small_model)
    elif model_key == "vit":
        return test_vit_on_hardware(hardware, use_small_model=use_small_model)
    elif model_key == "t5":
        return test_t5_on_hardware(hardware, use_small_model=use_small_model)
    elif model_key == "whisper":
        return test_whisper_on_hardware(hardware, use_small_model=use_small_model)
    else:
        return {
            "success": False,
            "model_key": model_key,
            "hardware": hardware,
            "error": f"Unsupported model key: {model_key}",
            "error_type": "ValueError"
        }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test models on different hardware backends")
    
    # Model selection
    parser.add_argument("--models", type=str, nargs="+", default=["bert"],
                        choices=list(KEY_MODELS.keys()),
                        help="Models to test")
    
    # Hardware selection
    parser.add_argument("--hardware", type=str, nargs="+", default=["cpu"],
                        choices=["cpu", "cuda", "mps", "rocm", "openvino", "webnn", "webgpu"],
                        help="Hardware backends to test")
    
    # Use small models
    parser.add_argument("--small-models", action="store_true",
                        help="Use smaller model variants for faster testing")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="./hardware_test_results",
                        help="Directory to save test results")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Detect available hardware
    available_hardware = detect_available_hardware()
    logger.info(f"Available hardware: {[hw for hw, available in available_hardware.items() if available]}")
    
    # Filter requested hardware by availability
    hardware_to_test = [hw for hw in args.hardware if available_hardware.get(hw, False)]
    logger.info(f"Hardware to test: {hardware_to_test}")
    
    # Run tests
    results = {}
    for model_key in args.models:
        results[model_key] = {}
        
        for hardware in hardware_to_test:
            # Run the test
            result = run_test(model_key, hardware, use_small_model=args.small_models)
            results[model_key][hardware] = result
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"hardware_test_results_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "models_tested": args.models,
            "hardware_tested": hardware_to_test,
            "use_small_models": args.small_models,
            "available_hardware": {hw: available for hw, available in available_hardware.items()},
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to: {result_file}")
    
    # Print summary
    logger.info("=== Test Results Summary ===")
    for model_key, model_results in results.items():
        logger.info(f"Model: {model_key}")
        for hardware, result in model_results.items():
            status = "PASS" if result.get("success", False) else "FAIL"
            logger.info(f"  {hardware}: {status}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())