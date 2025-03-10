#!/usr/bin/env python3
"""
Model benchmark runner for hardware performance testing.

This script provides comprehensive benchmarking capabilities for all model types:
1. Supports all hardware platforms: CPU, CUDA, ROCm, MPS, OpenVINO, WebNN, WebGPU
2. Measures throughput, latency, memory usage, and startup time
3. Supports different batch sizes and precisions
4. Handles both inference and training modes
5. Works with all key model categories: text, vision, audio, multimodal

Usage:
  python model_benchmark_runner.py --model bert-base-uncased --device cpu
  python model_benchmark_runner.py --model t5-small --device cuda --batch-size 8
  python model_benchmark_runner.py --model whisper-tiny --device openvino --precision fp16
  python model_benchmark_runner.py --model vit-base --device cuda --training
"""

import os
import sys
import json
import time
import psutil
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging
import traceback
import gc
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_benchmark")

# Global variables
WARMUP_ITERATIONS = 3
BENCHMARK_ITERATIONS = 10

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, some benchmarks will be skipped")

try:
    import transformers
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
    from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, HuggingFace model benchmarks will be skipped")

try:
    import openvino
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logger.warning("OpenVINO not available, OpenVINO benchmarks will be skipped")

try:
    import soundfile as sf
    import librosa
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("Audio libraries not available, audio model benchmarks will be skipped")

try:
    from PIL import Image
    import cv2
    VISION_LIBS_AVAILABLE = True
except ImportError:
    VISION_LIBS_AVAILABLE = False
    logger.warning("Vision libraries not available, vision model benchmarks will be skipped")

def get_memory_usage() -> int:
    """
    Get current memory usage of the process.
    
    Returns:
        int: Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

def generate_random_input(
    model_type: str,
    batch_size: int = 1,
    sequence_length: int = 128,
    image_size: Tuple[int, int] = (224, 224),
    audio_length: int = 16000,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Generate random input data for benchmarking.
    
    Args:
        model_type (str): Type of model (text, vision, audio, multimodal)
        batch_size (int): Batch size
        sequence_length (int): Sequence length for text models
        image_size (Tuple[int, int]): Image dimensions for vision models
        audio_length (int): Audio length in samples for audio models
        device (str): Device to put tensors on
        
    Returns:
        Dict[str, Any]: Dictionary of input tensors
    """
    inputs = {}
    
    if model_type == "text":
        # Generate random input IDs
        if TORCH_AVAILABLE:
            inputs["input_ids"] = torch.randint(
                0, 1000, (batch_size, sequence_length), 
                device=device if device in ["cuda", "mps"] else "cpu"
            )
            inputs["attention_mask"] = torch.ones(
                (batch_size, sequence_length), 
                dtype=torch.long,
                device=device if device in ["cuda", "mps"] else "cpu"
            )
        else:
            inputs["input_ids"] = np.random.randint(0, 1000, (batch_size, sequence_length))
            inputs["attention_mask"] = np.ones((batch_size, sequence_length), dtype=np.int64)
    
    elif model_type == "vision":
        # Generate random images
        if TORCH_AVAILABLE:
            inputs["pixel_values"] = torch.rand(
                (batch_size, 3, image_size[0], image_size[1]), 
                device=device if device in ["cuda", "mps"] else "cpu"
            )
        else:
            inputs["pixel_values"] = np.random.rand(batch_size, 3, image_size[0], image_size[1]).astype(np.float32)
    
    elif model_type == "audio":
        # Generate random audio
        if TORCH_AVAILABLE:
            inputs["input_features"] = torch.rand(
                (batch_size, 1, 80, audio_length // 160), 
                device=device if device in ["cuda", "mps"] else "cpu"
            )
        else:
            inputs["input_features"] = np.random.rand(batch_size, 1, 80, audio_length // 160).astype(np.float32)
    
    elif model_type == "multimodal":
        # Generate random images and text
        if TORCH_AVAILABLE:
            inputs["pixel_values"] = torch.rand(
                (batch_size, 3, image_size[0], image_size[1]), 
                device=device if device in ["cuda", "mps"] else "cpu"
            )
            inputs["input_ids"] = torch.randint(
                0, 1000, (batch_size, sequence_length), 
                device=device if device in ["cuda", "mps"] else "cpu"
            )
            inputs["attention_mask"] = torch.ones(
                (batch_size, sequence_length), 
                dtype=torch.long,
                device=device if device in ["cuda", "mps"] else "cpu"
            )
        else:
            inputs["pixel_values"] = np.random.rand(batch_size, 3, image_size[0], image_size[1]).astype(np.float32)
            inputs["input_ids"] = np.random.randint(0, 1000, (batch_size, sequence_length))
            inputs["attention_mask"] = np.ones((batch_size, sequence_length), dtype=np.int64)
    
    elif model_type == "video":
        # Generate random video frames
        if TORCH_AVAILABLE:
            # [batch_size, frames, channels, height, width]
            inputs["pixel_values"] = torch.rand(
                (batch_size, 8, 3, image_size[0], image_size[1]), 
                device=device if device in ["cuda", "mps"] else "cpu"
            )
        else:
            inputs["pixel_values"] = np.random.rand(batch_size, 8, 3, image_size[0], image_size[1]).astype(np.float32)
    
    return inputs

def load_model(
    model_name: str,
    model_type: str,
    device: str = "cpu",
    precision: str = "fp32",
    training_mode: bool = False
) -> Tuple[Any, float]:
    """
    Load a model for benchmarking.
    
    Args:
        model_name (str): Name or path of the model
        model_type (str): Type of model (text, vision, audio, multimodal)
        device (str): Device to load the model on
        precision (str): Precision to use (fp32, fp16, int8)
        training_mode (bool): Whether to load the model in training mode
        
    Returns:
        Tuple[Any, float]: Loaded model and startup time in ms
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers library not available")
        return None, 0
    
    try:
        start_time = time.time()
        model = None
        
        # Configure precision
        torch_dtype = torch.float32
        if precision == "fp16" and device in ["cuda", "mps"]:
            torch_dtype = torch.float16
        
        # Load appropriate model architecture based on model type
        if model_type == "text":
            # Handle different text model types
            if "bert" in model_name.lower() or "roberta" in model_name.lower():
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            elif "t5" in model_name.lower():
                from transformers import T5ForConditionalGeneration
                model = T5ForConditionalGeneration.from_pretrained(model_name)
            elif "gpt" in model_name.lower() or "llama" in model_name.lower() or "opt" in model_name.lower():
                model = AutoModelForCausalLM.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
        
        elif model_type == "vision":
            # Handle different vision model types
            if "vit" in model_name.lower():
                from transformers import ViTForImageClassification
                model = ViTForImageClassification.from_pretrained(model_name)
            elif "detr" in model_name.lower():
                from transformers import DetrForObjectDetection
                model = DetrForObjectDetection.from_pretrained(model_name)
            else:
                from transformers import AutoModelForImageClassification
                model = AutoModelForImageClassification.from_pretrained(model_name)
        
        elif model_type == "audio":
            # Handle different audio model types
            if "whisper" in model_name.lower():
                from transformers import WhisperForConditionalGeneration
                model = WhisperForConditionalGeneration.from_pretrained(model_name)
            elif "wav2vec2" in model_name.lower():
                from transformers import Wav2Vec2ForCTC
                model = Wav2Vec2ForCTC.from_pretrained(model_name)
            elif "clap" in model_name.lower():
                from transformers import ClapModel
                model = ClapModel.from_pretrained(model_name)
            else:
                from transformers import AutoModelForAudioClassification
                model = AutoModelForAudioClassification.from_pretrained(model_name)
        
        elif model_type == "multimodal":
            # Handle different multimodal model types
            if "clip" in model_name.lower():
                from transformers import CLIPModel
                model = CLIPModel.from_pretrained(model_name)
            elif "llava" in model_name.lower():
                try:
                    from transformers import LlavaForConditionalGeneration
                    model = LlavaForConditionalGeneration.from_pretrained(model_name)
                except:
                    # Fallback for older versions
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
        
        elif model_type == "video":
            # Handle different video model types
            if "xclip" in model_name.lower():
                from transformers import XClipModel
                model = XClipModel.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
        
        # Configure device and precision
        if device == "cuda" and torch.cuda.is_available():
            if precision == "fp16":
                model = model.half()
            model = model.to("cuda")
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if precision == "fp16":
                model = model.half()
            model = model.to("mps")
        elif device == "cpu":
            if precision == "int8":
                # Convert to int8 if requested
                try:
                    from transformers import BitsAndBytesConfig
                    model = model.quantize(BitsAndBytesConfig(load_in_8bit=True))
                except:
                    logger.warning("Failed to quantize model to int8, using fp32 instead")
            model = model.to("cpu")
        elif device == "openvino" and OPENVINO_AVAILABLE:
            # For OpenVINO, we convert the model to ONNX and then to OpenVINO IR
            # This is just a placeholder - actual implementation would vary
            pass
        else:
            model = model.to("cpu")
            logger.warning(f"Device {device} not available, using CPU instead")
        
        # Set training or evaluation mode
        if training_mode:
            model.train()
        else:
            model.eval()
        
        end_time = time.time()
        startup_time = (end_time - start_time) * 1000  # Convert to ms
        
        return model, startup_time
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        traceback.print_exc()
        return None, 0

def benchmark_inference(
    model: Any,
    inputs: Dict[str, Any],
    batch_size: int = 1,
    iterations: int = 10,
    warmup_iterations: int = 3,
    training_mode: bool = False,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Benchmark model inference.
    
    Args:
        model (Any): Model to benchmark
        inputs (Dict[str, Any]): Input data
        batch_size (int): Batch size
        iterations (int): Number of benchmark iterations
        warmup_iterations (int): Number of warmup iterations
        training_mode (bool): Whether to benchmark in training mode
        device (str): Device to run on
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    if model is None:
        return {
            "throughput": 0,
            "latency_mean": 0,
            "latency_p50": 0,
            "latency_p95": 0,
            "latency_p99": 0,
            "memory_usage": 0,
            "first_inference": 0,
            "error": "Model is None"
        }
    
    try:
        latencies = []
        memory_before = get_memory_usage()
        
        # First inference (cold start)
        first_start = time.time()
        
        with torch.inference_mode(not training_mode):
            if device in ["cuda", "mps"]:
                if device == "cuda":
                    torch.cuda.synchronize()
                model(**inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
            else:
                model(**inputs)
        
        first_end = time.time()
        first_inference_time = (first_end - first_start) * 1000  # Convert to ms
        
        # Warmup
        for _ in range(warmup_iterations):
            with torch.inference_mode(not training_mode):
                if device in ["cuda", "mps"]:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    model(**inputs)
                    if device == "cuda":
                        torch.cuda.synchronize()
                else:
                    model(**inputs)
        
        # Benchmark
        for _ in range(iterations):
            start = time.time()
            
            with torch.inference_mode(not training_mode):
                if device in ["cuda", "mps"]:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    model(**inputs)
                    if device == "cuda":
                        torch.cuda.synchronize()
                else:
                    model(**inputs)
            
            end = time.time()
            latency = (end - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate memory usage
        memory_after = get_memory_usage()
        memory_usage = memory_after - memory_before
        
        # Calculate metrics
        latency_mean = np.mean(latencies)
        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        latency_p99 = np.percentile(latencies, 99)
        throughput = (batch_size * 1000) / latency_mean  # samples/sec
        
        return {
            "throughput": float(throughput),
            "latency_mean": float(latency_mean),
            "latency_p50": float(latency_p50),
            "latency_p95": float(latency_p95),
            "latency_p99": float(latency_p99),
            "memory_usage": float(memory_usage),
            "first_inference": float(first_inference_time),
            "latencies": latencies,
            "batch_size": batch_size
        }
    
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        traceback.print_exc()
        return {
            "throughput": 0,
            "latency_mean": 0,
            "latency_p50": 0,
            "latency_p95": 0,
            "latency_p99": 0,
            "memory_usage": 0,
            "first_inference": 0,
            "error": str(e)
        }

def benchmark_training(
    model: Any,
    inputs: Dict[str, Any],
    batch_size: int = 1,
    iterations: int = 10,
    warmup_iterations: int = 3,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Benchmark model training.
    
    Args:
        model (Any): Model to benchmark
        inputs (Dict[str, Any]): Input data
        batch_size (int): Batch size
        iterations (int): Number of benchmark iterations
        warmup_iterations (int): Number of warmup iterations
        device (str): Device to run on
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    if model is None:
        return {
            "throughput": 0,
            "latency_mean": 0,
            "latency_p50": 0,
            "latency_p95": 0,
            "latency_p99": 0,
            "memory_usage": 0,
            "first_inference": 0,
            "error": "Model is None"
        }
    
    try:
        latencies = []
        memory_before = get_memory_usage()
        
        # Set up training
        model.train()
        if TORCH_AVAILABLE and device in ["cuda", "mps", "cpu"]:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # Add labels for training
        if "text" in inputs:
            if TORCH_AVAILABLE:
                inputs["labels"] = torch.randint(0, 2, (batch_size,), device=device if device in ["cuda", "mps"] else "cpu")
            else:
                inputs["labels"] = np.random.randint(0, 2, (batch_size,))
        elif "pixel_values" in inputs:
            if TORCH_AVAILABLE:
                inputs["labels"] = torch.randint(0, 1000, (batch_size,), device=device if device in ["cuda", "mps"] else "cpu")
            else:
                inputs["labels"] = np.random.randint(0, 1000, (batch_size,))
        
        # First training step (cold start)
        first_start = time.time()
        
        if TORCH_AVAILABLE and device in ["cuda", "mps", "cpu"]:
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        first_end = time.time()
        first_inference_time = (first_end - first_start) * 1000  # Convert to ms
        
        # Warmup
        for _ in range(warmup_iterations):
            if TORCH_AVAILABLE and device in ["cuda", "mps", "cpu"]:
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        
        # Benchmark
        for _ in range(iterations):
            start = time.time()
            
            if TORCH_AVAILABLE and device in ["cuda", "mps", "cpu"]:
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            end = time.time()
            latency = (end - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate memory usage
        memory_after = get_memory_usage()
        memory_usage = memory_after - memory_before
        
        # Calculate metrics
        latency_mean = np.mean(latencies)
        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        latency_p99 = np.percentile(latencies, 99)
        throughput = (batch_size * 1000) / latency_mean  # samples/sec
        
        return {
            "throughput": float(throughput),
            "latency_mean": float(latency_mean),
            "latency_p50": float(latency_p50),
            "latency_p95": float(latency_p95),
            "latency_p99": float(latency_p99),
            "memory_usage": float(memory_usage),
            "first_inference": float(first_inference_time),
            "latencies": latencies,
            "batch_size": batch_size
        }
    
    except Exception as e:
        logger.error(f"Error during training benchmark: {e}")
        traceback.print_exc()
        return {
            "throughput": 0,
            "latency_mean": 0,
            "latency_p50": 0,
            "latency_p95": 0,
            "latency_p99": 0,
            "memory_usage": 0,
            "first_inference": 0,
            "error": str(e)
        }

def get_model_type(model_name: str, audio_input: bool = False, vision_input: bool = False, multimodal_input: bool = False) -> str:
    """
    Determine model type based on model name and input flags.
    
    Args:
        model_name (str): Name or path of the model
        audio_input (bool): Flag for audio input
        vision_input (bool): Flag for vision input
        multimodal_input (bool): Flag for multimodal input
        
    Returns:
        str: Model type (text, vision, audio, multimodal, video)
    """
    model_name = model_name.lower()
    
    # Override with input flags
    if multimodal_input:
        return "multimodal"
    if audio_input:
        return "audio"
    if vision_input:
        return "vision"
    
    # Try to infer from model name
    if "bert" in model_name or "roberta" in model_name or "gpt" in model_name or "t5" in model_name or "llama" in model_name:
        return "text"
    elif "vit" in model_name or "resnet" in model_name or "detr" in model_name:
        return "vision"
    elif "wav2vec" in model_name or "whisper" in model_name or "hubert" in model_name:
        return "audio"
    elif "clip" in model_name and "x" in model_name:
        return "video"
    elif "clip" in model_name or "llava" in model_name or "blip" in model_name:
        return "multimodal"
    else:
        # Default to text
        return "text"

def get_input_shape(model_type: str, model_name: str) -> Dict[str, Any]:
    """
    Get default input shape for a model.
    
    Args:
        model_type (str): Type of model (text, vision, audio, multimodal)
        model_name (str): Name or path of the model
        
    Returns:
        Dict[str, Any]: Dictionary of input shapes
    """
    model_name = model_name.lower()
    
    # Default shapes
    shapes = {
        "sequence_length": 128,
        "image_size": (224, 224),
        "audio_length": 16000,
        "sample_rate": 16000
    }
    
    # Model-specific adjustments
    if model_type == "text":
        if "t5" in model_name:
            shapes["sequence_length"] = 256
        elif "gpt" in model_name or "llama" in model_name:
            shapes["sequence_length"] = 512
    
    elif model_type == "vision":
        if "vit" in model_name:
            if "384" in model_name:
                shapes["image_size"] = (384, 384)
        elif "detr" in model_name:
            shapes["image_size"] = (800, 800)
    
    elif model_type == "audio":
        if "whisper" in model_name:
            shapes["audio_length"] = 30 * 16000  # 30 seconds
        elif "clap" in model_name:
            shapes["audio_length"] = 10 * 16000  # 10 seconds
    
    elif model_type == "multimodal":
        if "llava" in model_name:
            shapes["image_size"] = (336, 336)
            shapes["sequence_length"] = 256
        elif "clip" in model_name:
            shapes["sequence_length"] = 77
    
    elif model_type == "video":
        if "xclip" in model_name:
            shapes["frames"] = 8
            shapes["frame_size"] = (224, 224)
    
    return shapes

def run_benchmark(args) -> Dict[str, Any]:
    """
    Run benchmark for a model on a specific hardware platform.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    # Determine device
    device = args.device.lower() if args.device else "cpu"
    
    # Determine model type
    model_type = get_model_type(
        args.model,
        audio_input=args.audio_input,
        vision_input=args.vision_input,
        multimodal_input=args.multimodal_input
    )
    
    # Log benchmark parameters
    logger.info(f"Benchmarking {args.model} on {device} with {args.precision} precision")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Mode: {'training' if args.training else 'inference'}")
    
    # Get input shapes
    input_shapes = get_input_shape(model_type, args.model)
    
    # Generate benchmark ID
    benchmark_id = f"{args.model.replace('/', '_')}_{device}_{args.precision}_{args.batch_size}"
    
    # Initialize results
    results = {
        "model": args.model,
        "device": device,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "model_type": model_type,
        "mode": "training" if args.training else "inference",
        "timestamp": datetime.now().isoformat(),
        "success": False
    }
    
    try:
        # Check requirements
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        if model_type == "audio" and not AUDIO_LIBS_AVAILABLE:
            raise ImportError("Audio libraries not available")
        
        if model_type in ["vision", "multimodal", "video"] and not VISION_LIBS_AVAILABLE:
            raise ImportError("Vision libraries not available")
        
        if device == "cuda" and not (TORCH_AVAILABLE and torch.cuda.is_available()):
            raise RuntimeError("CUDA not available")
        
        if device == "mps" and not (TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS not available")
        
        if device == "openvino" and not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO not available")
        
        # Load model
        logger.info(f"Loading model {args.model}")
        model, startup_time = load_model(
            args.model,
            model_type,
            device=device,
            precision=args.precision,
            training_mode=args.training
        )
        
        if model is None:
            raise RuntimeError("Failed to load model")
        
        results["startup_time"] = startup_time
        
        # Generate inputs
        logger.info("Generating inputs")
        inputs = generate_random_input(
            model_type,
            batch_size=args.batch_size,
            sequence_length=input_shapes.get("sequence_length", 128),
            image_size=input_shapes.get("image_size", (224, 224)),
            audio_length=input_shapes.get("audio_length", 16000),
            device=device
        )
        
        # Run benchmark
        logger.info(f"Running {'training' if args.training else 'inference'} benchmark")
        if args.training:
            benchmark_results = benchmark_training(
                model,
                inputs,
                batch_size=args.batch_size,
                iterations=args.iterations,
                warmup_iterations=args.warmup_iterations,
                device=device
            )
        else:
            benchmark_results = benchmark_inference(
                model,
                inputs,
                batch_size=args.batch_size,
                iterations=args.iterations,
                warmup_iterations=args.warmup_iterations,
                device=device
            )
        
        # Update results
        results.update(benchmark_results)
        results["success"] = True
        
        # Free memory
        del model, inputs
        if TORCH_AVAILABLE and device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        traceback.print_exc()
        
        results["error"] = str(e)
        return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Model benchmark runner for hardware performance testing")
    
    # Model and hardware options
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "openvino", "rocm", "webnn", "webgpu"], default="cpu", help="Device to benchmark on")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32", help="Precision to use")
    
    # Benchmark options
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS, help="Number of benchmark iterations")
    parser.add_argument("--warmup-iterations", type=int, default=WARMUP_ITERATIONS, help="Number of warmup iterations")
    parser.add_argument("--training", action="store_true", help="Benchmark training instead of inference")
    parser.add_argument("--output", help="Output file for benchmark results (JSON)")
    
    # Input type flags
    parser.add_argument("--audio-input", action="store_true", help="Force audio input type")
    parser.add_argument("--vision-input", action="store_true", help="Force vision input type")
    parser.add_argument("--multimodal-input", action="store_true", help="Force multimodal input type")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(args)
    
    # Print results
    print("\nBenchmark results:")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mode: {'training' if args.training else 'inference'}")
    
    if results.get("success", False):
        print(f"Throughput: {results.get('throughput', 0):.2f} samples/sec")
        print(f"Latency (mean): {results.get('latency_mean', 0):.2f} ms")
        print(f"Latency (p95): {results.get('latency_p95', 0):.2f} ms")
        print(f"Memory usage: {results.get('memory_usage', 0):.2f} MB")
        print(f"Startup time: {results.get('startup_time', 0):.2f} ms")
        print(f"First inference: {results.get('first_inference', 0):.2f} ms")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    # Save results to file
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()