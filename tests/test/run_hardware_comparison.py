#!/usr/bin/env python3
"""
Enhanced hardware comparison benchmark for testing multiple models on various hardware backends.
This version adds improved OpenVINO support, better hardware detection, and expanded model type support.
"""

import os
import sys
import time
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define model sets
MODEL_SETS = {
    "text_embedding": [
        "prajjwal1/bert-tiny",
        "bert-base-uncased",
    ],
    "text_generation": [
        "google/t5-efficient-tiny",
        "t5-small",
    ],
    "vision": [
        "google/vit-base-patch16-224",
    ],
    "audio": [
        "openai/whisper-tiny",
    ],
    "all": [
        "prajjwal1/bert-tiny",
        "google/t5-efficient-tiny",
        "google/vit-base-patch16-224",
        "openai/whisper-tiny",
    ],
    "quick": [
        "prajjwal1/bert-tiny",
        "google/t5-efficient-tiny",
    ]
}

# Define hardware sets
HARDWARE_SETS = {
    "local": ["cpu"],
    "gpu": ["cuda"],
    "intel": ["cpu", "openvino"],
    "all": ["cpu", "cuda", "rocm", "mps", "openvino"],
    "web": ["cpu", "webnn", "webgpu"],
    "quick": ["cpu", "cuda"],
}

def detect_available_hardware() -> Dict[str, bool]:
    """Detect available hardware backends."""
    available_hardware = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "rocm": torch.cuda.is_available() and hasattr(torch.version, "hip"),
    }
    
    # Try to import OpenVINO
    try:
        import openvino
        available_hardware["openvino"] = True
        logger.info(f"OpenVINO version {openvino.__version__} detected")
    except ImportError:
        available_hardware["openvino"] = False
        logger.info("OpenVINO not available")
    
    # WebNN and WebGPU (these would typically be available through browser interfaces)
    # We'll mark these as unavailable since they require browser runtime
    available_hardware["webnn"] = False
    available_hardware["webgpu"] = False
    
    # Try to detect Qualcomm AI Engine (QNN)
    try:
        import qti.aisw
        available_hardware["qualcomm"] = True
    except ImportError:
        available_hardware["qualcomm"] = False
    
    # Log available hardware
    logger.info(f"Available hardware: {[hw for hw, available in available_hardware.items() if available]}")
    
    return available_hardware

def get_device_for_hardware(hardware: str) -> torch.device:
    """Get PyTorch device for the specified hardware."""
    if hardware == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA with {torch.cuda.device_count()} devices")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    elif hardware == "rocm" and torch.cuda.is_available() and hasattr(torch.version, "hip"):
        device = torch.device("cuda")
        logger.info("Using ROCm (AMD GPU)")
    elif hardware == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU")
    else:
        logger.warning(f"Hardware {hardware} not directly supported in PyTorch, falling back to CPU")
        device = torch.device("cpu")
    
    return device

def benchmark_bert(model_name: str, hardware: str, device: torch.device, batch_sizes: List[int], 
                  warmup: int = 2, runs: int = 5) -> List[Dict[str, Any]]:
    """Benchmark BERT-type models."""
    try:
        from transformers import BertModel, BertTokenizer
        
        logger.info(f"Loading {model_name} tokenizer and model...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        if hardware == "openvino":
            # Special handling for OpenVINO
            try:
                import openvino as ov
                from transformers.utils.backbone_utils import get_backend_from_model_name
                
                logger.info("Converting BERT model to OpenVINO IR format...")
                from optimum.intel.openvino import OVModelForFeatureExtraction
                
                # Load the model with OpenVINO backend
                model = OVModelForFeatureExtraction.from_pretrained(
                    model_name, 
                    export=True
                )
                logger.info("Model loaded with OpenVINO backend")
                
                # Define function to run inference
                def run_inference(inputs):
                    return model(**inputs)
                    
            except (ImportError, Exception) as e:
                logger.error(f"Error setting up OpenVINO for {model_name}: {e}")
                logger.info("Falling back to PyTorch on CPU")
                hardware = "cpu"
                device = torch.device("cpu")
                model = BertModel.from_pretrained(model_name)
                model = model.to(device)
                model.eval()
                
                # Define function to run inference
                def run_inference(inputs):
                    return model(**inputs)
        else:
            # Standard PyTorch model
            model = BertModel.from_pretrained(model_name)
            model = model.to(device)
            model.eval()
            
            # Define function to run inference
            def run_inference(inputs):
                return model(**inputs)
        
        # Define text for benchmarking
        text = "This is a sample text for benchmarking models on different hardware backends."
        
        # Benchmark results
        results = []
        
        # Run benchmarks for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking with batch size {batch_size}")
            
            # Prepare input (replicate for batch size)
            encoded = tokenizer(text, return_tensors="pt", padding=True)
            inputs = {k: v.repeat(batch_size, 1).to(device) for k, v in encoded.items()}
            
            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = run_inference(inputs)
            
            # Measure inference time
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            timings = []
            for _ in range(runs):
                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = run_inference(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                timings.append(end_time - start_time)
            
            # Calculate statistics
            latency = np.mean(timings) * 1000  # Convert to ms
            throughput = batch_size / np.mean(timings)  # Items per second
            latency_std = np.std(timings) * 1000  # Convert to ms
            
            # Get memory usage if CUDA is used
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = run_inference(inputs)
                memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                memory_usage = 0
            
            # Record results
            result = {
                "batch_size": batch_size,
                "latency_ms": latency,
                "latency_std_ms": latency_std,
                "throughput_items_per_sec": throughput,
                "memory_usage_mb": memory_usage,
            }
            
            # Add input shape info if available
            if hasattr(inputs, "items"):
                result["input_shape"] = {k: str(v.shape) for k, v in inputs.items()}
            
            # Add output shape info if available
            if hasattr(outputs, "items"):
                result["output_shape"] = {k: str(v.shape) for k, v in outputs.items() if hasattr(v, "shape")}
            
            results.append(result)
            
            logger.info(f"Batch size {batch_size}: Latency = {latency:.2f} ms, Throughput = {throughput:.2f} items/sec")
        
        return results
        
    except Exception as e:
        logger.error(f"Error benchmarking BERT model {model_name} on {hardware}: {e}")
        raise e

def benchmark_t5(model_name: str, hardware: str, device: torch.device, batch_sizes: List[int], 
                warmup: int = 2, runs: int = 5) -> List[Dict[str, Any]]:
    """Benchmark T5-type models."""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        logger.info(f"Loading {model_name} tokenizer and model...")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        if hardware == "openvino":
            # Special handling for OpenVINO
            try:
                import openvino as ov
                from transformers.utils.backbone_utils import get_backend_from_model_name
                
                logger.info("Converting T5 model to OpenVINO IR format...")
                from optimum.intel.openvino import OVModelForSeq2SeqLM
                
                # Load the model with OpenVINO backend
                model = OVModelForSeq2SeqLM.from_pretrained(
                    model_name, 
                    export=True
                )
                logger.info("Model loaded with OpenVINO backend")
                
                # Define functions to run inference
                def run_forward(inputs, decoder_input_ids):
                    return model(**inputs, decoder_input_ids=decoder_input_ids)
                
                def run_generation(inputs):
                    return model.generate(**inputs, max_length=20, num_beams=1)
                    
            except (ImportError, Exception) as e:
                logger.error(f"Error setting up OpenVINO for {model_name}: {e}")
                logger.info("Falling back to PyTorch on CPU")
                hardware = "cpu"
                device = torch.device("cpu")
                model = T5ForConditionalGeneration.from_pretrained(model_name)
                model = model.to(device)
                model.eval()
                
                # Define functions to run inference
                def run_forward(inputs, decoder_input_ids):
                    return model(**inputs, decoder_input_ids=decoder_input_ids)
                
                def run_generation(inputs):
                    return model.generate(**inputs, max_length=20, num_beams=1)
        else:
            # Standard PyTorch model
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            model = model.to(device)
            model.eval()
            
            # Define functions to run inference
            def run_forward(inputs, decoder_input_ids):
                return model(**inputs, decoder_input_ids=decoder_input_ids)
            
            def run_generation(inputs):
                return model.generate(**inputs, max_length=20, num_beams=1)
        
        # Define text for benchmarking
        text = "translate English to German: The house is wonderful."
        
        # Benchmark results
        results = []
        
        # Run benchmarks for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking with batch size {batch_size}")
            
            # Prepare input (replicate for batch size)
            encoded = tokenizer(text, return_tensors="pt", padding=True)
            inputs = {k: v.repeat(batch_size, 1).to(device) for k, v in encoded.items()}
            
            # Create decoder_input_ids
            decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * model.config.decoder_start_token_id
            
            # Warmup for forward pass
            for _ in range(warmup):
                with torch.no_grad():
                    _ = run_forward(inputs, decoder_input_ids)
            
            # Measure forward pass inference time
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            forward_timings = []
            for _ in range(runs):
                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = run_forward(inputs, decoder_input_ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                forward_timings.append(end_time - start_time)
            
            # Warmup for generation
            for _ in range(1):  # Fewer warmup runs for generation as it's slower
                with torch.no_grad():
                    _ = run_generation(inputs)
            
            # Measure generation inference time
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            generation_timings = []
            for _ in range(2):  # Fewer measurement runs for generation
                start_time = time.perf_counter()
                with torch.no_grad():
                    generated = run_generation(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                generation_timings.append(end_time - start_time)
            
            # Calculate statistics for forward pass
            forward_latency = np.mean(forward_timings) * 1000  # Convert to ms
            forward_throughput = batch_size / np.mean(forward_timings)  # Items per second
            forward_latency_std = np.std(forward_timings) * 1000  # Convert to ms
            
            # Calculate statistics for generation
            generation_latency = np.mean(generation_timings) * 1000  # Convert to ms
            generation_throughput = batch_size / np.mean(generation_timings)  # Items per second
            generation_latency_std = np.std(generation_timings) * 1000  # Convert to ms
            
            # Get memory usage if CUDA is used
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = run_forward(inputs, decoder_input_ids)
                forward_memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = run_generation(inputs)
                generation_memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                forward_memory_usage = 0
                generation_memory_usage = 0
            
            # Record results
            result = {
                "batch_size": batch_size,
                "forward_latency_ms": forward_latency,
                "forward_latency_std_ms": forward_latency_std,
                "forward_throughput_items_per_sec": forward_throughput,
                "forward_memory_usage_mb": forward_memory_usage,
                "generation_latency_ms": generation_latency,
                "generation_latency_std_ms": generation_latency_std,
                "generation_throughput_items_per_sec": generation_throughput,
                "generation_memory_usage_mb": generation_memory_usage,
            }
            
            # Add input shape info if available
            if hasattr(inputs, "items"):
                result["input_shape"] = {k: str(v.shape) for k, v in inputs.items()}
            
            # Add output shape info if available
            if hasattr(outputs, "items"):
                result["output_shape"] = {k: str(v.shape) for k, v in outputs.items() if hasattr(v, "shape")}
            
            results.append(result)
            
            logger.info(f"Batch size {batch_size} forward pass: Latency = {forward_latency:.2f} ms, "
                      f"Throughput = {forward_throughput:.2f} items/sec")
            logger.info(f"Batch size {batch_size} generation: Latency = {generation_latency:.2f} ms, "
                      f"Throughput = {generation_throughput:.2f} items/sec")
        
        return results
        
    except Exception as e:
        logger.error(f"Error benchmarking T5 model {model_name} on {hardware}: {e}")
        raise e

def benchmark_vit(model_name: str, hardware: str, device: torch.device, batch_sizes: List[int],
                 warmup: int = 2, runs: int = 5) -> List[Dict[str, Any]]:
    """Benchmark Vision Transformer (ViT) models."""
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        from PIL import Image
        import requests
        from io import BytesIO
        
        logger.info(f"Loading {model_name} processor and model...")
        processor = ViTImageProcessor.from_pretrained(model_name)
        
        if hardware == "openvino":
            # Special handling for OpenVINO
            try:
                import openvino as ov
                
                logger.info("Converting ViT model to OpenVINO IR format...")
                from optimum.intel.openvino import OVModelForImageClassification
                
                # Load the model with OpenVINO backend
                model = OVModelForImageClassification.from_pretrained(
                    model_name, 
                    export=True
                )
                logger.info("Model loaded with OpenVINO backend")
                
                # Define function to run inference
                def run_inference(inputs):
                    return model(**inputs)
                    
            except (ImportError, Exception) as e:
                logger.error(f"Error setting up OpenVINO for {model_name}: {e}")
                logger.info("Falling back to PyTorch on CPU")
                hardware = "cpu"
                device = torch.device("cpu")
                model = ViTForImageClassification.from_pretrained(model_name)
                model = model.to(device)
                model.eval()
                
                # Define function to run inference
                def run_inference(inputs):
                    return model(**inputs)
        else:
            # Standard PyTorch model
            model = ViTForImageClassification.from_pretrained(model_name)
            model = model.to(device)
            model.eval()
            
            # Define function to run inference
            def run_inference(inputs):
                return model(**inputs)
        
        # Get sample image for benchmarking
        if os.path.exists('test.jpg'):
            logger.info("Loading sample image from test.jpg")
            image = Image.open('test.jpg')
        else:
            logger.info("Downloading sample image")
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # Sample image (cat)
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                image.save('test.jpg')
            except Exception as e:
                logger.error(f"Error downloading sample image: {e}")
                # Create a simple test image
                image = Image.new('RGB', (224, 224), color='red')
                image.save('test.jpg')
        
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Benchmark results
        results = []
        
        # Run benchmarks for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking with batch size {batch_size}")
            
            # Prepare input (replicate for batch size)
            batch_inputs = {k: v.repeat(batch_size, 1, 1, 1) if v.dim() == 4 else v.repeat(batch_size, 1) 
                            for k, v in inputs.items()}
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = run_inference(batch_inputs)
            
            # Measure inference time
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            timings = []
            for _ in range(runs):
                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = run_inference(batch_inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                timings.append(end_time - start_time)
            
            # Calculate statistics
            latency = np.mean(timings) * 1000  # Convert to ms
            throughput = batch_size / np.mean(timings)  # Items per second
            latency_std = np.std(timings) * 1000  # Convert to ms
            
            # Get memory usage if CUDA is used
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = run_inference(batch_inputs)
                memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                memory_usage = 0
            
            # Record results
            result = {
                "batch_size": batch_size,
                "latency_ms": latency,
                "latency_std_ms": latency_std,
                "throughput_items_per_sec": throughput,
                "memory_usage_mb": memory_usage,
            }
            
            # Add input shape info if available
            if hasattr(batch_inputs, "items"):
                result["input_shape"] = {k: str(v.shape) for k, v in batch_inputs.items()}
            
            # Add output shape info if available
            if hasattr(outputs, "items"):
                result["output_shape"] = {k: str(v.shape) for k, v in outputs.items() if hasattr(v, "shape")}
            
            results.append(result)
            
            logger.info(f"Batch size {batch_size}: Latency = {latency:.2f} ms, Throughput = {throughput:.2f} items/sec")
        
        return results
        
    except Exception as e:
        logger.error(f"Error benchmarking ViT model {model_name} on {hardware}: {e}")
        raise e

def benchmark_whisper(model_name: str, hardware: str, device: torch.device, batch_sizes: List[int],
                     warmup: int = 2, runs: int = 5) -> List[Dict[str, Any]]:
    """Benchmark Whisper models."""
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import torch.nn.functional as F
        
        logger.info(f"Loading {model_name} processor and model...")
        processor = WhisperProcessor.from_pretrained(model_name)
        
        if hardware == "openvino":
            # Special handling for OpenVINO
            try:
                import openvino as ov
                
                logger.info("Converting Whisper model to OpenVINO IR format...")
                from optimum.intel.openvino import OVModelForSpeechSeq2Seq
                
                # Load the model with OpenVINO backend
                model = OVModelForSpeechSeq2Seq.from_pretrained(
                    model_name, 
                    export=True
                )
                logger.info("Model loaded with OpenVINO backend")
                
                # Define function to run inference
                def run_inference(inputs):
                    return model(**inputs)
                    
            except (ImportError, Exception) as e:
                logger.error(f"Error setting up OpenVINO for {model_name}: {e}")
                logger.info("Falling back to PyTorch on CPU")
                hardware = "cpu"
                device = torch.device("cpu")
                model = WhisperForConditionalGeneration.from_pretrained(model_name)
                model = model.to(device)
                model.eval()
                
                # Define function to run inference
                def run_inference(inputs):
                    return model(**inputs)
        else:
            # Standard PyTorch model
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            model = model.to(device)
            model.eval()
            
            # Define function to run inference
            def run_inference(inputs):
                return model(**inputs)
        
        # Generate dummy audio for benchmarking
        sample_rate = 16000
        audio_length = 3  # seconds
        dummy_audio = torch.rand(sample_rate * audio_length).to(device)
        
        # Convert to mel spectrogram feature
        inputs = processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Add forced decoder IDs
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "forced_decoder_ids"):
            decoder_input_ids = torch.tensor(model.generation_config.forced_decoder_ids).unsqueeze(0).to(device)
        else:
            # Default for English
            decoder_input_ids = torch.tensor([[1, 1]]).to(device)
        
        # Benchmark results
        results = []
        
        # Run benchmarks for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking with batch size {batch_size}")
            
            # Prepare inputs for batching
            batch_inputs = {k: v.repeat(batch_size, *([1] * (v.dim() - 1))) for k, v in inputs.items()}
            
            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = run_inference(batch_inputs)
            
            # Measure inference time
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            timings = []
            for _ in range(runs):
                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = run_inference(batch_inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                timings.append(end_time - start_time)
            
            # Calculate statistics
            latency = np.mean(timings) * 1000  # Convert to ms
            throughput = batch_size / np.mean(timings)  # Items per second
            latency_std = np.std(timings) * 1000  # Convert to ms
            
            # Get memory usage if CUDA is used
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = run_inference(batch_inputs)
                memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                memory_usage = 0
            
            # Record results
            result = {
                "batch_size": batch_size,
                "latency_ms": latency,
                "latency_std_ms": latency_std,
                "throughput_items_per_sec": throughput,
                "memory_usage_mb": memory_usage,
            }
            
            # Add input shape info if available
            if hasattr(batch_inputs, "items"):
                result["input_shape"] = {k: str(v.shape) for k, v in batch_inputs.items()}
            
            # Add output shape info if available
            if hasattr(outputs, "items"):
                result["output_shape"] = {k: str(v.shape) for k, v in outputs.items() if hasattr(v, "shape")}
            
            results.append(result)
            
            logger.info(f"Batch size {batch_size}: Latency = {latency:.2f} ms, Throughput = {throughput:.2f} items/sec")
        
        return results
        
    except Exception as e:
        logger.error(f"Error benchmarking Whisper model {model_name} on {hardware}: {e}")
        raise e

def run_benchmark(model_name: str, hardware: str, batch_sizes: Optional[List[int]] = None, 
                 warmup: int = 2, runs: int = 5) -> Dict[str, Any]:
    """Run benchmarks for a model on a specific hardware."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]
    
    logger.info(f"Benchmarking {model_name} on {hardware}")
    
    # Get device for hardware
    device = get_device_for_hardware(hardware)
    
    # Determine model type and run the appropriate benchmark
    model_type = "unknown"
    try:
        if "bert" in model_name.lower():
            model_type = "bert"
            results = benchmark_bert(model_name, hardware, device, batch_sizes, warmup, runs)
        elif "t5" in model_name.lower():
            model_type = "t5"
            results = benchmark_t5(model_name, hardware, device, batch_sizes, warmup, runs)
        elif "vit" in model_name.lower():
            model_type = "vit"
            results = benchmark_vit(model_name, hardware, device, batch_sizes, warmup, runs)
        elif "whisper" in model_name.lower():
            model_type = "whisper"
            results = benchmark_whisper(model_name, hardware, device, batch_sizes, warmup, runs)
        else:
            # Default to BERT for other models
            logger.warning(f"Defaulting to BERT benchmark for unknown model type: {model_name}")
            model_type = "bert"
            results = benchmark_bert(model_name, hardware, device, batch_sizes, warmup, runs)
        
        return {
            "model_name": model_name,
            "model_type": model_type,
            "hardware": hardware,
            "device": str(device),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error benchmarking {model_name} on {hardware}: {e}")
        return {
            "model_name": model_name,
            "hardware": hardware,
            "error": str(e),
            "error_type": type(e).__name__
        }

def run_benchmark_for_model(model_name: str, hardware_backends: List[str], batch_sizes: List[int], 
                           warmup: int, runs: int, output_dir: Path) -> Dict[str, Any]:
    """Run benchmark for a specific model across hardware backends."""
    logger.info(f"Benchmarking {model_name} across {len(hardware_backends)} hardware backends")
    
    # Results container
    results = {
        "model_name": model_name,
        "hardware_results": {},
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    
    # Run benchmarks for each hardware backend
    for hardware in hardware_backends:
        try:
            logger.info(f"Running benchmark for {model_name} on {hardware}")
            
            # Run the benchmark
            result = run_benchmark(
                model_name=model_name,
                hardware=hardware,
                batch_sizes=batch_sizes,
                warmup=warmup,
                runs=runs
            )
            
            # Store the result
            results["hardware_results"][hardware] = result
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_name} on {hardware}: {e}")
            results["hardware_results"][hardware] = {
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # Save the results
    model_name_safe = model_name.replace("/", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"benchmark_{model_name_safe}_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Benchmark results for {model_name} saved to: {result_file}")
    
    return results

def generate_markdown_report(all_results: Dict[str, Dict[str, Any]], output_dir: Path) -> Path:
    """Generate a comprehensive markdown report."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"benchmark_report_{timestamp}.md"
    
    with open(report_file, "w") as f:
        # Header
        f.write("# Comprehensive Model Benchmark Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Summary](#summary)\n")
        f.write("2. [Hardware Comparison](#hardware-comparison)\n")
        for model_name in all_results.keys():
            model_name_anchor = model_name.replace("/", "_").lower()
            f.write(f"3. [{model_name}](#{model_name_anchor})\n")
        f.write("\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("| Model | Hardware | Batch Size | Latency (ms) | Throughput (items/s) |\n")
        f.write("|-------|----------|------------|--------------|---------------------|\n")
        
        for model_name, model_results in all_results.items():
            for hardware, hw_result in model_results["hardware_results"].items():
                if "error" in hw_result:
                    continue
                
                model_type = hw_result.get("model_type", "unknown")
                
                if model_type in ["bert", "vit", "whisper"]:
                    # Get the result for batch size 1 (or the smallest batch size)
                    results = hw_result.get("results", [])
                    if results:
                        batch_result = sorted(results, key=lambda x: x.get("batch_size", 0))[0]
                        batch_size = batch_result.get("batch_size", "N/A")
                        latency = batch_result.get("latency_ms", "N/A")
                        throughput = batch_result.get("throughput_items_per_sec", "N/A")
                        
                        f.write(f"| {model_name} | {hardware} | {batch_size} | {latency:.2f} | {throughput:.2f} |\n")
                
                elif model_type == "t5":
                    # Get the result for batch size 1 (or the smallest batch size)
                    results = hw_result.get("results", [])
                    if results:
                        batch_result = sorted(results, key=lambda x: x.get("batch_size", 0))[0]
                        batch_size = batch_result.get("batch_size", "N/A")
                        forward_latency = batch_result.get("forward_latency_ms", "N/A")
                        forward_throughput = batch_result.get("forward_throughput_items_per_sec", "N/A")
                        
                        f.write(f"| {model_name} | {hardware} | {batch_size} | {forward_latency:.2f} | {forward_throughput:.2f} |\n")
        
        f.write("\n")
        
        # Hardware Comparison
        f.write("## Hardware Comparison\n\n")
        f.write("### Throughput Comparison\n\n")
        
        # Get unique hardware backends and models
        hardware_backends = set()
        for model_results in all_results.values():
            for hardware in model_results["hardware_results"].keys():
                hardware_backends.add(hardware)
        
        hardware_backends = sorted(list(hardware_backends))
        
        # Create throughput comparison table
        f.write("| Model | Batch Size |")
        for hardware in hardware_backends:
            f.write(f" {hardware} (items/s) |")
        f.write("\n")
        
        f.write("|-------|------------|")
        for _ in hardware_backends:
            f.write("--------------|")
        f.write("\n")
        
        # Batch sizes to include in the comparison
        batch_sizes_to_show = [1, 4, 16]
        
        for model_name, model_results in all_results.items():
            for batch_size in batch_sizes_to_show:
                f.write(f"| {model_name} | {batch_size} |")
                
                for hardware in hardware_backends:
                    hw_result = model_results["hardware_results"].get(hardware, {"error": "Not tested"})
                    
                    if "error" in hw_result:
                        f.write(" N/A |")
                        continue
                    
                    model_type = hw_result.get("model_type", "unknown")
                    results = hw_result.get("results", [])
                    
                    # Find the result for this batch size
                    batch_result = next((r for r in results if r.get("batch_size") == batch_size), None)
                    
                    if batch_result:
                        if model_type in ["bert", "vit", "whisper"]:
                            throughput = batch_result.get("throughput_items_per_sec", "N/A")
                            f.write(f" {throughput:.2f} |")
                        elif model_type == "t5":
                            throughput = batch_result.get("forward_throughput_items_per_sec", "N/A")
                            f.write(f" {throughput:.2f} |")
                        else:
                            f.write(" N/A |")
                    else:
                        f.write(" N/A |")
                
                f.write("\n")
        
        f.write("\n")
        
        # Detailed results for each model
        for model_name, model_results in all_results.items():
            model_name_anchor = model_name.replace("/", "_").lower()
            f.write(f"## {model_name}\n\n")
            
            for hardware, hw_result in model_results["hardware_results"].items():
                f.write(f"### {hardware}\n\n")
                
                if "error" in hw_result:
                    f.write(f"Error: {hw_result['error']}\n\n")
                    continue
                
                model_type = hw_result.get("model_type", "unknown")
                
                if model_type in ["bert", "vit", "whisper"]:
                    f.write("| Batch Size | Latency (ms) | Throughput (items/s) | Memory (MB) |\n")
                    f.write("|------------|--------------|---------------------|------------|\n")
                    
                    for batch_result in hw_result.get("results", []):
                        batch_size = batch_result.get("batch_size", "N/A")
                        latency = batch_result.get("latency_ms", "N/A")
                        throughput = batch_result.get("throughput_items_per_sec", "N/A")
                        memory = batch_result.get("memory_usage_mb", "N/A")
                        
                        f.write(f"| {batch_size} | {latency:.2f} | {throughput:.2f} | {memory:.2f} |\n")
                
                elif model_type == "t5":
                    f.write("| Batch Size | Forward Latency (ms) | Forward Throughput (items/s) | Generation Latency (ms) | Generation Throughput (items/s) |\n")
                    f.write("|------------|----------------------|----------------------------|-------------------------|-------------------------------|\n")
                    
                    for batch_result in hw_result.get("results", []):
                        batch_size = batch_result.get("batch_size", "N/A")
                        forward_latency = batch_result.get("forward_latency_ms", "N/A")
                        forward_throughput = batch_result.get("forward_throughput_items_per_sec", "N/A")
                        generation_latency = batch_result.get("generation_latency_ms", "N/A")
                        generation_throughput = batch_result.get("generation_throughput_items_per_sec", "N/A")
                        
                        f.write(f"| {batch_size} | {forward_latency:.2f} | {forward_throughput:.2f} | {generation_latency:.2f} | {generation_throughput:.2f} |\n")
                
                else:
                    f.write(f"Unknown model type: {model_type}\n")
                
                f.write("\n")
            
            f.write("\n")
    
    logger.info(f"Markdown report saved to: {report_file}")
    return report_file

def generate_csv_report(all_results: Dict[str, Dict[str, Any]], output_dir: Path) -> Path:
    """Generate a CSV report with benchmark results for easier data analysis."""
    import csv
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"benchmark_report_{timestamp}.csv"
    
    # Determine model types present in results
    model_types = set()
    for model_results in all_results.values():
        for hw_result in model_results["hardware_results"].values():
            if "error" not in hw_result:
                model_types.add(hw_result.get("model_type", "unknown"))
    
    # Create CSV file
    with open(report_file, "w", newline="") as f:
        # Create appropriate CSV writer
        writer = csv.writer(f)
        
        # Write header row based on model types present
        header = ["Model", "Model Type", "Hardware", "Batch Size"]
        
        # Add metrics based on model types
        if "bert" in model_types or "vit" in model_types or "whisper" in model_types:
            header.extend(["Latency (ms)", "Latency Std (ms)", "Throughput (items/s)", "Memory (MB)"])
        
        if "t5" in model_types:
            header.extend([
                "Forward Latency (ms)", "Forward Latency Std (ms)", "Forward Throughput (items/s)", "Forward Memory (MB)",
                "Generation Latency (ms)", "Generation Latency Std (ms)", "Generation Throughput (items/s)", "Generation Memory (MB)"
            ])
        
        # Write the header
        writer.writerow(header)
        
        # Write data rows
        for model_name, model_results in all_results.items():
            for hardware, hw_result in model_results["hardware_results"].items():
                if "error" in hw_result:
                    # Write error row
                    writer.writerow([model_name, "error", hardware, "N/A", hw_result["error"]])
                    continue
                
                model_type = hw_result.get("model_type", "unknown")
                results = hw_result.get("results", [])
                
                for batch_result in results:
                    row = [model_name, model_type, hardware, batch_result.get("batch_size", "N/A")]
                    
                    # Add metrics based on model type
                    if model_type in ["bert", "vit", "whisper"]:
                        row.extend([
                            batch_result.get("latency_ms", "N/A"),
                            batch_result.get("latency_std_ms", "N/A"),
                            batch_result.get("throughput_items_per_sec", "N/A"),
                            batch_result.get("memory_usage_mb", "N/A")
                        ])
                        # Add empty values for T5 columns if they exist in the header
                        if "t5" in model_types:
                            row.extend(["N/A"] * 8)  # 8 columns for T5
                    
                    elif model_type == "t5":
                        # Add empty values for BERT/ViT/Whisper columns if they exist in the header
                        if "bert" in model_types or "vit" in model_types or "whisper" in model_types:
                            row.extend(["N/A"] * 4)  # 4 columns for BERT/ViT/Whisper
                        
                        # Add T5 specific metrics
                        row.extend([
                            batch_result.get("forward_latency_ms", "N/A"),
                            batch_result.get("forward_latency_std_ms", "N/A"),
                            batch_result.get("forward_throughput_items_per_sec", "N/A"),
                            batch_result.get("forward_memory_usage_mb", "N/A"),
                            batch_result.get("generation_latency_ms", "N/A"),
                            batch_result.get("generation_latency_std_ms", "N/A"),
                            batch_result.get("generation_throughput_items_per_sec", "N/A"),
                            batch_result.get("generation_memory_usage_mb", "N/A")
                        ])
                    
                    writer.writerow(row)
    
    logger.info(f"CSV report saved to: {report_file}")
    return report_file

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run comprehensive hardware comparison benchmarks")
    
    # Model selection
    parser.add_argument("--model-set", type=str, default="quick",
                      choices=list(MODEL_SETS.keys()),
                      help="Set of models to benchmark")
    parser.add_argument("--models", type=str, nargs="+",
                      help="Specific models to benchmark (overrides model-set)")
    
    # Hardware selection
    parser.add_argument("--hardware-set", type=str, default="quick",
                      choices=list(HARDWARE_SETS.keys()),
                      help="Set of hardware backends to test")
    parser.add_argument("--hardware", type=str, nargs="+",
                      help="Specific hardware backends to test (overrides hardware-set)")
    
    # Batch sizes
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16],
                      help="Batch sizes to test")
    
    # Benchmark parameters
    parser.add_argument("--warmup", type=int, default=2,
                      help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5,
                      help="Number of measurement runs")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                      help="Directory to save benchmark results")
    
    # Output format
    parser.add_argument("--format", type=str, default="markdown",
                      choices=["markdown", "json", "csv"],
                      help="Output format for the report")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    # OpenVINO specific parameters
    parser.add_argument("--openvino-precision", type=str, default="FP32",
                      choices=["FP32", "FP16", "INT8"],
                      help="Precision for OpenVINO models")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Also set root logger to debug
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Detect available hardware
    available_hardware = detect_available_hardware()
    logger.info(f"Available hardware platforms: {[hw for hw, available in available_hardware.items() if available]}")
    
    # Determine models to benchmark
    if args.models:
        models_to_benchmark = args.models
    else:
        models_to_benchmark = MODEL_SETS[args.model_set]
    
    # Determine hardware backends to test (filtered by availability)
    if args.hardware:
        hardware_to_test = args.hardware
    else:
        hardware_to_test = HARDWARE_SETS[args.hardware_set]
    
    # Filter requested hardware by availability
    hardware_to_test = [hw for hw in hardware_to_test if available_hardware.get(hw, False)]
    if not hardware_to_test:
        logger.warning("None of the requested hardware is available, falling back to CPU")
        hardware_to_test = ["cpu"]
    
    logger.info(f"Models to benchmark: {models_to_benchmark}")
    logger.info(f"Hardware backends to test: {hardware_to_test}")
    
    # Export OpenVINO environment variables if needed
    if "openvino" in hardware_to_test:
        os.environ["OPENVINO_PRECISION"] = args.openvino_precision
        logger.info(f"Setting OpenVINO precision to {args.openvino_precision}")
    
    # Run benchmarks for each model
    all_results = {}
    for model_name in models_to_benchmark:
        results = run_benchmark_for_model(
            model_name=model_name,
            hardware_backends=hardware_to_test,
            batch_sizes=args.batch_sizes,
            warmup=args.warmup,
            runs=args.runs,
            output_dir=output_dir
        )
        all_results[model_name] = results
    
    # Generate report based on requested format
    if args.format == "markdown":
        report_file = generate_markdown_report(all_results, output_dir)
        logger.info(f"Markdown report saved to: {report_file}")
    elif args.format == "json":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"benchmark_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"JSON report saved to: {report_file}")
    elif args.format == "csv":
        report_file = generate_csv_report(all_results, output_dir)
        logger.info(f"CSV report saved to: {report_file}")
    
    logger.info("Hardware comparison benchmark completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())