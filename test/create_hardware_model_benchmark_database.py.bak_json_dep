#!/usr/bin/env python3
"""
Create comprehensive benchmark database for all model-hardware combinations.

This script implements Phase 16 of the project plan:
1. Creates a benchmark database for all model-hardware combinations
2. Implements comparative analysis reporting system for hardware performance 
3. Supports training mode benchmarking in addition to inference
4. Generates hardware selection recommendations based on benchmarking data

Usage:
  python create_hardware_model_benchmark_database.py --all
  python create_hardware_model_benchmark_database.py --model bert --hardware all
  python create_hardware_model_benchmark_database.py --category vision --compare
  python create_hardware_model_benchmark_database.py --training-mode
  python create_hardware_model_benchmark_database.py --recommendations
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hardware_model_benchmark.log")
    ]
)
logger = logging.getLogger("hardware_model_benchmark")

# Paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
BENCHMARK_DIR = TEST_DIR / "benchmark_results"
BENCHMARK_DB_FILE = BENCHMARK_DIR / "hardware_model_benchmark_db.parquet"
BENCHMARK_MATRIX_FILE = BENCHMARK_DIR / "hardware_compatibility_matrix.json"
VISUALIZATION_DIR = BENCHMARK_DIR / "visualizations"

# Create directories
BENCHMARK_DIR.mkdir(exist_ok=True, parents=True)
VISUALIZATION_DIR.mkdir(exist_ok=True, parents=True)

# Key model classes for comprehensive testing
KEY_MODELS = {
    "bert": {
        "name": "BERT",
        "models": ["bert-base-uncased", "prajjwal1/bert-tiny"],
        "category": "text_embedding",
        "batch_sizes": [1, 8, 16, 32, 64],
        "input_shapes": {"sequence_length": 128}
    },
    "t5": {
        "name": "T5",
        "models": ["t5-small", "google/t5-efficient-tiny"],
        "category": "text_generation",
        "batch_sizes": [1, 4, 8, 16],
        "input_shapes": {"sequence_length": 128, "decoder_length": 32}
    },
    "llama": {
        "name": "LLAMA",
        "models": ["facebook/opt-125m"],
        "category": "text_generation",
        "batch_sizes": [1, 2, 4, 8],
        "input_shapes": {"sequence_length": 256, "decoder_length": 64}
    },
    "clip": {
        "name": "CLIP",
        "models": ["openai/clip-vit-base-patch32"],
        "category": "vision_text",
        "batch_sizes": [1, 8, 16, 32],
        "input_shapes": {"image_size": [224, 224], "sequence_length": 77}
    },
    "vit": {
        "name": "ViT",
        "models": ["google/vit-base-patch16-224"],
        "category": "vision",
        "batch_sizes": [1, 8, 16, 32, 64],
        "input_shapes": {"image_size": [224, 224]}
    },
    "clap": {
        "name": "CLAP",
        "models": ["laion/clap-htsat-unfused"],
        "category": "audio_text",
        "batch_sizes": [1, 4, 8, 16],
        "input_shapes": {"audio_length": 16000, "sequence_length": 77}
    },
    "whisper": {
        "name": "Whisper",
        "models": ["openai/whisper-tiny"],
        "category": "audio",
        "batch_sizes": [1, 2, 4, 8],
        "input_shapes": {"audio_length": 30, "sample_rate": 16000}
    },
    "wav2vec2": {
        "name": "Wav2Vec2",
        "models": ["facebook/wav2vec2-base"],
        "category": "audio",
        "batch_sizes": [1, 2, 4, 8],
        "input_shapes": {"audio_length": 16000, "sample_rate": 16000}
    },
    "llava": {
        "name": "LLaVA",
        "models": ["llava-hf/llava-1.5-7b-hf"],
        "category": "multimodal",
        "batch_sizes": [1, 2, 4],
        "input_shapes": {"image_size": [336, 336], "sequence_length": 128}
    },
    "llava_next": {
        "name": "LLaVA-Next",
        "models": ["llava-hf/llava-v1.6-34b-hf"],
        "category": "multimodal",
        "batch_sizes": [1, 2],
        "input_shapes": {"image_size": [336, 336], "sequence_length": 128}
    },
    "xclip": {
        "name": "XCLIP",
        "models": ["microsoft/xclip-base-patch32"],
        "category": "video",
        "batch_sizes": [1, 2, 4, 8],
        "input_shapes": {"frames": 8, "frame_size": [224, 224], "sequence_length": 77}
    },
    "qwen2": {
        "name": "Qwen2",
        "models": ["Qwen/Qwen2-7B-Instruct"],
        "category": "text_generation",
        "batch_sizes": [1, 2, 4],
        "input_shapes": {"sequence_length": 256, "decoder_length": 64}
    },
    "detr": {
        "name": "DETR",
        "models": ["facebook/detr-resnet-50"],
        "category": "vision",
        "batch_sizes": [1, 4, 8, 16],
        "input_shapes": {"image_size": [800, 800]}
    }
}

# Hardware platforms for testing
HARDWARE_PLATFORMS = {
    "cpu": {
        "name": "CPU",
        "compatibility": set(KEY_MODELS.keys()),
        "flag": "--device cpu",
        "precision": ["fp32", "int8"]
    },
    "cuda": {
        "name": "CUDA",
        "compatibility": set(KEY_MODELS.keys()),
        "flag": "--device cuda",
        "precision": ["fp32", "fp16", "int8"]
    },
    "rocm": {
        "name": "AMD ROCm",
        "compatibility": set(KEY_MODELS.keys()) - {"llava", "llava_next"},
        "flag": "--device rocm",
        "precision": ["fp32", "fp16"]
    },
    "mps": {
        "name": "Apple MPS",
        "compatibility": set(KEY_MODELS.keys()) - {"llava", "llava_next"},
        "flag": "--device mps",
        "precision": ["fp32", "fp16"]
    },
    "openvino": {
        "name": "OpenVINO",
        "compatibility": set(KEY_MODELS.keys()) - {"llava_next"},
        "flag": "--device openvino",
        "precision": ["fp32", "fp16", "int8"]
    },
    "webnn": {
        "name": "WebNN",
        "compatibility": {"bert", "t5", "clip", "vit", "whisper", "detr"},
        "flag": "--web-platform webnn",
        "precision": ["fp32", "fp16"]
    },
    "webgpu": {
        "name": "WebGPU",
        "compatibility": {"bert", "t5", "clip", "vit", "whisper", "detr"},
        "flag": "--web-platform webgpu",
        "precision": ["fp32", "fp16"]
    }
}

# Benchmark metrics to collect
METRICS = [
    "throughput",      # Samples/second
    "latency_mean",    # Average latency in ms
    "latency_p50",     # 50th percentile latency in ms
    "latency_p95",     # 95th percentile latency in ms
    "latency_p99",     # 99th percentile latency in ms
    "memory_usage",    # Memory usage in MB
    "startup_time",    # Time to load model in ms
    "first_inference", # Time for first inference in ms
]

def detect_available_hardware() -> Dict[str, bool]:
    """
    Detect which hardware platforms are available on the current system.
    
    Returns:
        Dict[str, bool]: Dictionary mapping hardware platforms to availability
    """
    available = {}
    
    # Check CPU (always available)
    available["cpu"] = True
    
    # Check CUDA
    try:
        import torch
        available["cuda"] = torch.cuda.is_available()
    except ImportError:
        available["cuda"] = False
    
    # Check ROCm (AMD)
    try:
        import torch
        available["rocm"] = torch.cuda.is_available() and hasattr(torch.version, "hip")
    except ImportError:
        available["rocm"] = False
    
    # Check MPS (Apple)
    try:
        import torch
        available["mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        available["mps"] = False
    
    # Check OpenVINO
    try:
        import openvino
        available["openvino"] = True
    except ImportError:
        available["openvino"] = False
    
    # Web platforms (always marked as unavailable for local testing)
    available["webnn"] = False
    available["webgpu"] = False
    
    logger.info(f"Available hardware platforms: {[k for k, v in available.items() if v]}")
    return available

def create_benchmark_command(
    model_key: str, 
    hardware: str, 
    batch_size: int = 1,
    precision: str = "fp32",
    mode: str = "inference"
) -> Optional[str]:
    """
    Generate a command to benchmark a specific model on specific hardware.
    
    Args:
        model_key (str): Key for the model to benchmark
        hardware (str): Hardware platform to benchmark on
        batch_size (int): Batch size to use
        precision (str): Precision to use (fp32, fp16, int8)
        mode (str): Mode to benchmark (inference, training)
        
    Returns:
        Optional[str]: Command to run the benchmark, or None if incompatible
    """
    if model_key not in KEY_MODELS:
        return None
    
    if hardware not in HARDWARE_PLATFORMS:
        return None
    
    if model_key not in HARDWARE_PLATFORMS[hardware]["compatibility"]:
        return None
    
    if precision not in HARDWARE_PLATFORMS[hardware]["precision"]:
        logger.warning(f"Precision {precision} not supported on {hardware}, defaulting to fp32")
        precision = "fp32"
    
    model_name = KEY_MODELS[model_key]["models"][0]
    hw_flag = HARDWARE_PLATFORMS[hardware]["flag"]
    category = KEY_MODELS[model_key]["category"]
    
    # Basic benchmark command
    command = f"python test/model_benchmark_runner.py --model {model_name} {hw_flag}"
    
    # Add batch size
    command += f" --batch-size {batch_size}"
    
    # Add precision
    command += f" --precision {precision}"
    
    # Add mode
    if mode == "training":
        command += " --training"
    
    # Add category-specific flags
    if category == "audio" or category == "audio_text":
        command += " --audio-input"
    elif category == "vision" or category == "video":
        command += " --vision-input"
    elif category == "multimodal":
        command += " --multimodal-input"
    
    # Add output format flag to get JSON results
    command += f" --output {BENCHMARK_DIR}/benchmark_{model_key}_{hardware}_{batch_size}_{precision}_{mode}.json"
    
    return command

def run_benchmark(
    model_key: str, 
    hardware: str, 
    batch_size: int = 1,
    precision: str = "fp32",
    mode: str = "inference",
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Run a benchmark for a specific model on specific hardware.
    
    Args:
        model_key (str): Key for the model to benchmark
        hardware (str): Hardware platform to benchmark on
        batch_size (int): Batch size to use
        precision (str): Precision to use (fp32, fp16, int8)
        mode (str): Mode to benchmark (inference, training)
        timeout (int): Timeout in seconds
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    command = create_benchmark_command(model_key, hardware, batch_size, precision, mode)
    
    if not command:
        return {
            "model": model_key,
            "hardware": hardware,
            "batch_size": batch_size,
            "precision": precision,
            "mode": mode,
            "status": "incompatible",
            "error": "Incompatible model-hardware combination"
        }
    
    logger.info(f"Running benchmark: {command}")
    
    try:
        # Create a unique output file name
        output_file = f"{BENCHMARK_DIR}/benchmark_{model_key}_{hardware}_{batch_size}_{precision}_{mode}.json"
        
        # Run the command
        start_time = time.time()
        os.system(command)
        end_time = time.time()
        
        # Check if output file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
                
                # Add metadata
                results["model"] = model_key
                results["hardware"] = hardware
                results["batch_size"] = batch_size
                results["precision"] = precision
                results["mode"] = mode
                results["status"] = "success"
                results["execution_time"] = end_time - start_time
                
                return results
        else:
            return {
                "model": model_key,
                "hardware": hardware,
                "batch_size": batch_size,
                "precision": precision,
                "mode": mode,
                "status": "failed",
                "error": "No output file generated"
            }
    
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return {
            "model": model_key,
            "hardware": hardware,
            "batch_size": batch_size,
            "precision": precision,
            "mode": mode,
            "status": "failed",
            "error": str(e)
        }

def create_benchmark_database():
    """
    Create a benchmark database with placeholder entries for all compatible
    model-hardware combinations.
    
    Returns:
        pd.DataFrame: Benchmark database
    """
    # Detect available hardware
    available_hardware = detect_available_hardware()
    
    # Create entries for all compatible combinations
    entries = []
    
    for model_key, model_info in KEY_MODELS.items():
        for hw_key, hw_info in HARDWARE_PLATFORMS.items():
            # Skip unavailable hardware platforms
            if not available_hardware.get(hw_key, False) and hw_key != "cpu":
                continue
                
            # Skip incompatible combinations
            if model_key not in hw_info["compatibility"]:
                continue
                
            # Add entries for different batch sizes
            for batch_size in model_info.get("batch_sizes", [1]):
                # Add entries for different precision options
                for precision in hw_info.get("precision", ["fp32"]):
                    # Add entry for inference mode
                    entries.append({
                        "model": model_key,
                        "model_name": model_info["name"],
                        "model_path": model_info["models"][0],
                        "category": model_info["category"],
                        "hardware": hw_key,
                        "hardware_name": hw_info["name"],
                        "batch_size": batch_size,
                        "precision": precision,
                        "mode": "inference",
                        "status": "pending",
                        "timestamp": None
                    })
                    
                    # Add entry for training mode (if applicable)
                    if model_info["category"] not in ["audio", "video"]:
                        entries.append({
                            "model": model_key,
                            "model_name": model_info["name"],
                            "model_path": model_info["models"][0],
                            "category": model_info["category"],
                            "hardware": hw_key,
                            "hardware_name": hw_info["name"],
                            "batch_size": batch_size,
                            "precision": precision,
                            "mode": "training",
                            "status": "pending",
                            "timestamp": None
                        })
    
    # Create DataFrame
    df = pd.DataFrame(entries)
    
    logger.info(f"Created benchmark database with {len(df)} entries")
    return df

def save_benchmark_database(df: pd.DataFrame):
    """
    Save the benchmark database to a file.
    
    Args:
        df (pd.DataFrame): Benchmark database
    """
    df.to_parquet(BENCHMARK_DB_FILE)
    logger.info(f"Saved benchmark database to {BENCHMARK_DB_FILE}")

def load_benchmark_database() -> pd.DataFrame:
    """
    Load the benchmark database from a file.
    
    Returns:
        pd.DataFrame: Benchmark database
    """
    if not os.path.exists(BENCHMARK_DB_FILE):
        logger.info(f"Benchmark database not found at {BENCHMARK_DB_FILE}, creating new one")
        df = create_benchmark_database()
        save_benchmark_database(df)
        return df
    
    df = pd.read_parquet(BENCHMARK_DB_FILE)
    logger.info(f"Loaded benchmark database with {len(df)} entries from {BENCHMARK_DB_FILE}")
    return df

def update_benchmark_entry(df: pd.DataFrame, results: Dict[str, Any]) -> pd.DataFrame:
    """
    Update a benchmark entry in the database with results.
    
    Args:
        df (pd.DataFrame): Benchmark database
        results (Dict[str, Any]): Benchmark results
        
    Returns:
        pd.DataFrame: Updated benchmark database
    """
    # Extract key information from results
    model = results.get("model")
    hardware = results.get("hardware")
    batch_size = results.get("batch_size")
    precision = results.get("precision")
    mode = results.get("mode")
    status = results.get("status")
    
    # Find the matching row
    mask = (
        (df["model"] == model) & 
        (df["hardware"] == hardware) & 
        (df["batch_size"] == batch_size) & 
        (df["precision"] == precision) & 
        (df["mode"] == mode)
    )
    
    if mask.sum() == 0:
        logger.warning(f"No matching entry found for {model} on {hardware} with batch size {batch_size}, precision {precision}, mode {mode}")
        return df
    
    # Update the entry
    for key, value in results.items():
        if key in df.columns:
            df.loc[mask, key] = value
    
    # Update status and timestamp
    df.loc[mask, "status"] = status
    df.loc[mask, "timestamp"] = datetime.now().isoformat()
    
    logger.info(f"Updated benchmark entry for {model} on {hardware} with batch size {batch_size}, precision {precision}, mode {mode}")
    return df

def benchmark_model(
    model_key: str, 
    hardware_platforms: List[str] = None,
    batch_sizes: List[int] = None,
    precision: str = "fp32",
    mode: str = "inference"
) -> List[Dict[str, Any]]:
    """
    Run benchmarks for a specific model across multiple hardware platforms,
    batch sizes, and precision options.
    
    Args:
        model_key (str): Key for the model to benchmark
        hardware_platforms (List[str]): Hardware platforms to benchmark on
        batch_sizes (List[int]): Batch sizes to use
        precision (str): Precision to use (fp32, fp16, int8)
        mode (str): Mode to benchmark (inference, training)
        
    Returns:
        List[Dict[str, Any]]: Benchmark results
    """
    if model_key not in KEY_MODELS:
        logger.error(f"Unknown model: {model_key}")
        return []
    
    # Load benchmark database
    df = load_benchmark_database()
    
    # Set default hardware platforms to all compatible platforms
    available_hardware = detect_available_hardware()
    if hardware_platforms is None:
        hardware_platforms = [hw for hw, available in available_hardware.items() if available]
    else:
        # Filter out unavailable hardware
        hardware_platforms = [hw for hw in hardware_platforms if available_hardware.get(hw, False) or hw == "cpu"]
    
    # Set default batch sizes to model-specific batch sizes
    if batch_sizes is None:
        batch_sizes = KEY_MODELS[model_key].get("batch_sizes", [1])
    
    # Run benchmarks
    results = []
    for hw in hardware_platforms:
        if model_key not in HARDWARE_PLATFORMS[hw]["compatibility"]:
            logger.info(f"Skipping incompatible model-hardware combination: {model_key} on {hw}")
            continue
            
        for batch_size in batch_sizes:
            # Check if precision is supported
            if precision not in HARDWARE_PLATFORMS[hw]["precision"]:
                logger.warning(f"Precision {precision} not supported on {hw}, defaulting to fp32")
                actual_precision = "fp32"
            else:
                actual_precision = precision
                
            # Run benchmark
            benchmark_result = run_benchmark(model_key, hw, batch_size, actual_precision, mode)
            results.append(benchmark_result)
            
            # Update database
            df = update_benchmark_entry(df, benchmark_result)
    
    # Save updated database
    save_benchmark_database(df)
    
    return results

def benchmark_hardware(
    hardware: str, 
    model_keys: List[str] = None,
    batch_sizes: List[int] = None,
    precision: str = "fp32",
    mode: str = "inference"
) -> List[Dict[str, Any]]:
    """
    Run benchmarks for a specific hardware platform across multiple models,
    batch sizes, and precision options.
    
    Args:
        hardware (str): Hardware platform to benchmark on
        model_keys (List[str]): Models to benchmark
        batch_sizes (List[int]): Batch sizes to use
        precision (str): Precision to use (fp32, fp16, int8)
        mode (str): Mode to benchmark (inference, training)
        
    Returns:
        List[Dict[str, Any]]: Benchmark results
    """
    if hardware not in HARDWARE_PLATFORMS:
        logger.error(f"Unknown hardware platform: {hardware}")
        return []
    
    # Check if hardware is available
    available_hardware = detect_available_hardware()
    if not available_hardware.get(hardware, False) and hardware != "cpu":
        logger.error(f"Hardware platform {hardware} is not available")
        return []
    
    # Load benchmark database
    df = load_benchmark_database()
    
    # Set default models to all compatible models
    if model_keys is None:
        model_keys = [m for m in KEY_MODELS.keys() if m in HARDWARE_PLATFORMS[hardware]["compatibility"]]
    else:
        # Filter out incompatible models
        model_keys = [m for m in model_keys if m in HARDWARE_PLATFORMS[hardware]["compatibility"]]
    
    # Run benchmarks
    results = []
    for model_key in model_keys:
        # Set default batch sizes to model-specific batch sizes
        if batch_sizes is None:
            model_batch_sizes = KEY_MODELS[model_key].get("batch_sizes", [1])
        else:
            model_batch_sizes = batch_sizes
            
        for batch_size in model_batch_sizes:
            # Check if precision is supported
            if precision not in HARDWARE_PLATFORMS[hardware]["precision"]:
                logger.warning(f"Precision {precision} not supported on {hardware}, defaulting to fp32")
                actual_precision = "fp32"
            else:
                actual_precision = precision
                
            # Run benchmark
            benchmark_result = run_benchmark(model_key, hardware, batch_size, actual_precision, mode)
            results.append(benchmark_result)
            
            # Update database
            df = update_benchmark_entry(df, benchmark_result)
    
    # Save updated database
    save_benchmark_database(df)
    
    return results

def benchmark_category(
    category: str, 
    hardware_platforms: List[str] = None,
    batch_sizes: List[int] = None,
    precision: str = "fp32",
    mode: str = "inference"
) -> List[Dict[str, Any]]:
    """
    Run benchmarks for a specific category of models across multiple hardware platforms,
    batch sizes, and precision options.
    
    Args:
        category (str): Category of models to benchmark
        hardware_platforms (List[str]): Hardware platforms to benchmark on
        batch_sizes (List[int]): Batch sizes to use
        precision (str): Precision to use (fp32, fp16, int8)
        mode (str): Mode to benchmark (inference, training)
        
    Returns:
        List[Dict[str, Any]]: Benchmark results
    """
    # Find models in the category
    model_keys = [k for k, v in KEY_MODELS.items() if v.get("category") == category]
    
    if not model_keys:
        logger.error(f"No models found in category: {category}")
        return []
    
    # Set default hardware platforms to all available platforms
    available_hardware = detect_available_hardware()
    if hardware_platforms is None:
        hardware_platforms = [hw for hw, available in available_hardware.items() if available]
    else:
        # Filter out unavailable hardware
        hardware_platforms = [hw for hw in hardware_platforms if available_hardware.get(hw, False) or hw == "cpu"]
    
    # Load benchmark database
    df = load_benchmark_database()
    
    # Run benchmarks
    results = []
    for model_key in model_keys:
        # Set default batch sizes to model-specific batch sizes
        if batch_sizes is None:
            model_batch_sizes = KEY_MODELS[model_key].get("batch_sizes", [1])
        else:
            model_batch_sizes = batch_sizes
        
        for hw in hardware_platforms:
            if model_key not in HARDWARE_PLATFORMS[hw]["compatibility"]:
                logger.info(f"Skipping incompatible model-hardware combination: {model_key} on {hw}")
                continue
                
            for batch_size in model_batch_sizes:
                # Check if precision is supported
                if precision not in HARDWARE_PLATFORMS[hw]["precision"]:
                    logger.warning(f"Precision {precision} not supported on {hw}, defaulting to fp32")
                    actual_precision = "fp32"
                else:
                    actual_precision = precision
                    
                # Run benchmark
                benchmark_result = run_benchmark(model_key, hw, batch_size, actual_precision, mode)
                results.append(benchmark_result)
                
                # Update database
                df = update_benchmark_entry(df, benchmark_result)
    
    # Save updated database
    save_benchmark_database(df)
    
    return results

def benchmark_all(
    precision: str = "fp32",
    mode: str = "inference",
    limit: int = None
) -> List[Dict[str, Any]]:
    """
    Run benchmarks for all compatible model-hardware combinations.
    
    Args:
        precision (str): Precision to use (fp32, fp16, int8)
        mode (str): Mode to benchmark (inference, training)
        limit (int): Maximum number of benchmarks to run
        
    Returns:
        List[Dict[str, Any]]: Benchmark results
    """
    # Load benchmark database
    df = load_benchmark_database()
    
    # Get entries that need benchmarking
    mask = (df["status"] == "pending") & (df["mode"] == mode)
    pending_entries = df[mask].copy()
    
    # Detect available hardware
    available_hardware = detect_available_hardware()
    
    # Filter out unavailable hardware
    pending_entries = pending_entries[
        pending_entries["hardware"].apply(lambda h: available_hardware.get(h, False) or h == "cpu")
    ]
    
    # Limit number of entries if specified
    if limit is not None and limit > 0:
        pending_entries = pending_entries.head(limit)
    
    logger.info(f"Running {len(pending_entries)} benchmarks")
    
    # Run benchmarks
    results = []
    for _, entry in tqdm(pending_entries.iterrows(), total=len(pending_entries)):
        model_key = entry["model"]
        hardware = entry["hardware"]
        batch_size = entry["batch_size"]
        
        # Check if precision is supported
        if precision not in HARDWARE_PLATFORMS[hardware]["precision"]:
            logger.warning(f"Precision {precision} not supported on {hardware}, defaulting to fp32")
            actual_precision = "fp32"
        else:
            actual_precision = precision
        
        # Run benchmark
        benchmark_result = run_benchmark(model_key, hardware, batch_size, actual_precision, mode)
        results.append(benchmark_result)
        
        # Update database
        df = update_benchmark_entry(df, benchmark_result)
    
    # Save updated database
    save_benchmark_database(df)
    
    return results

def generate_comparative_analysis(output_file: str = None) -> pd.DataFrame:
    """
    Generate a comparative analysis of benchmark results across hardware platforms.
    
    Args:
        output_file (str): Path to save the analysis
        
    Returns:
        pd.DataFrame: Comparative analysis
    """
    # Load benchmark database
    df = load_benchmark_database()
    
    # Filter to successful benchmarks for inference mode
    df_success = df[(df["status"] == "success") & (df["mode"] == "inference")]
    
    if len(df_success) == 0:
        logger.error("No successful benchmarks found for analysis")
        return None
    
    # Group by model and calculate mean metrics for each hardware platform
    analysis = df_success.groupby(["model", "hardware", "precision"]).agg({
        "throughput": "mean",
        "latency_mean": "mean",
        "memory_usage": "mean",
        "batch_size": "max"
    }).reset_index()
    
    # Generate speedup relative to CPU
    for model in analysis["model"].unique():
        model_data = analysis[analysis["model"] == model]
        
        # Get CPU baseline
        cpu_data = model_data[model_data["hardware"] == "cpu"]
        
        if len(cpu_data) == 0:
            logger.warning(f"No CPU data found for {model}, skipping speedup calculation")
            continue
        
        # Calculate speedup for each hardware platform
        for hw in model_data["hardware"].unique():
            if hw == "cpu":
                continue
                
            hw_data = model_data[model_data["hardware"] == hw]
            
            for metric in ["throughput", "latency_mean"]:
                if metric in cpu_data.columns and metric in hw_data.columns:
                    cpu_value = cpu_data[metric].values[0]
                    
                    # For throughput, higher is better, so divide hw by cpu
                    # For latency, lower is better, so divide cpu by hw
                    if metric == "throughput":
                        speedup = hw_data[metric] / cpu_value
                    else:
                        speedup = cpu_value / hw_data[metric]
                    
                    speedup_col = f"{metric}_speedup_vs_cpu"
                    analysis.loc[hw_data.index, speedup_col] = speedup
    
    # Save to file if specified
    if output_file:
        analysis.to_csv(output_file, index=False)
        logger.info(f"Saved comparative analysis to {output_file}")
    
    return analysis

def generate_hardware_recommendations() -> Dict[str, Dict[str, str]]:
    """
    Generate hardware recommendations for each model based on benchmark results.
    
    Returns:
        Dict[str, Dict[str, str]]: Hardware recommendations
    """
    # Load benchmark database
    df = load_benchmark_database()
    
    # Filter to successful benchmarks for inference mode
    df_success = df[(df["status"] == "success") & (df["mode"] == "inference")]
    
    if len(df_success) == 0:
        logger.error("No successful benchmarks found for recommendations")
        return {}
    
    # Group by model and find best hardware for different metrics
    recommendations = {}
    
    for model in df_success["model"].unique():
        model_data = df_success[df_success["model"] == model]
        
        # Initialize recommendation
        recommendations[model] = {
            "model_name": KEY_MODELS[model]["name"],
            "category": KEY_MODELS[model]["category"],
            "best_overall": None,
            "best_throughput": None,
            "best_latency": None,
            "best_memory": None,
            "best_value": None
        }
        
        # Find best hardware for throughput
        if "throughput" in model_data.columns:
            best_throughput = model_data.loc[model_data["throughput"].idxmax()]
            recommendations[model]["best_throughput"] = {
                "hardware": best_throughput["hardware"],
                "precision": best_throughput["precision"],
                "batch_size": best_throughput["batch_size"],
                "value": best_throughput["throughput"]
            }
        
        # Find best hardware for latency
        if "latency_mean" in model_data.columns:
            best_latency = model_data.loc[model_data["latency_mean"].idxmin()]
            recommendations[model]["best_latency"] = {
                "hardware": best_latency["hardware"],
                "precision": best_latency["precision"],
                "batch_size": best_latency["batch_size"],
                "value": best_latency["latency_mean"]
            }
        
        # Find best hardware for memory usage
        if "memory_usage" in model_data.columns:
            best_memory = model_data.loc[model_data["memory_usage"].idxmin()]
            recommendations[model]["best_memory"] = {
                "hardware": best_memory["hardware"],
                "precision": best_memory["precision"],
                "batch_size": best_memory["batch_size"],
                "value": best_memory["memory_usage"]
            }
        
        # Determine best overall hardware based on a weighted score
        if "throughput" in model_data.columns and "latency_mean" in model_data.columns:
            # Normalize metrics
            throughput_norm = model_data["throughput"] / model_data["throughput"].max()
            latency_norm = model_data["latency_mean"].min() / model_data["latency_mean"]
            
            # Calculate overall score (higher is better)
            model_data["overall_score"] = 0.6 * throughput_norm + 0.4 * latency_norm
            
            best_overall = model_data.loc[model_data["overall_score"].idxmax()]
            recommendations[model]["best_overall"] = {
                "hardware": best_overall["hardware"],
                "precision": best_overall["precision"],
                "batch_size": best_overall["batch_size"],
                "score": best_overall["overall_score"]
            }
        
        # Calculate best value (performance per watt or dollar)
        # This would require additional data on hardware costs/energy
        
    return recommendations

def generate_visualizations():
    """
    Generate visualizations of benchmark results.
    """
    # Load benchmark database
    df = load_benchmark_database()
    
    # Filter to successful benchmarks for inference mode
    df_success = df[(df["status"] == "success") & (df["mode"] == "inference")]
    
    if len(df_success) == 0:
        logger.error("No successful benchmarks found for visualization")
        return
    
    # 1. Throughput comparison across hardware platforms for each model
    for model in df_success["model"].unique():
        model_data = df_success[df_success["model"] == model]
        
        plt.figure(figsize=(12, 6))
        
        # Group by hardware and precision, take max throughput
        plot_data = model_data.groupby(["hardware", "precision"])["throughput"].max().unstack()
        
        plot_data.plot(kind="bar")
        plt.title(f"Throughput Comparison for {KEY_MODELS[model]['name']}")
        plt.xlabel("Hardware Platform")
        plt.ylabel("Throughput (samples/sec)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{VISUALIZATION_DIR}/{model}_throughput_comparison.png")
        plt.close()
    
    # 2. Latency comparison across hardware platforms for each model
    for model in df_success["model"].unique():
        model_data = df_success[df_success["model"] == model]
        
        plt.figure(figsize=(12, 6))
        
        # Group by hardware and precision, take min latency
        plot_data = model_data.groupby(["hardware", "precision"])["latency_mean"].min().unstack()
        
        plot_data.plot(kind="bar")
        plt.title(f"Latency Comparison for {KEY_MODELS[model]['name']}")
        plt.xlabel("Hardware Platform")
        plt.ylabel("Latency (ms)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{VISUALIZATION_DIR}/{model}_latency_comparison.png")
        plt.close()
    
    # 3. Memory usage comparison
    for model in df_success["model"].unique():
        model_data = df_success[df_success["model"] == model]
        
        plt.figure(figsize=(12, 6))
        
        # Group by hardware and precision, take min memory usage
        plot_data = model_data.groupby(["hardware", "precision"])["memory_usage"].min().unstack()
        
        plot_data.plot(kind="bar")
        plt.title(f"Memory Usage Comparison for {KEY_MODELS[model]['name']}")
        plt.xlabel("Hardware Platform")
        plt.ylabel("Memory Usage (MB)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{VISUALIZATION_DIR}/{model}_memory_comparison.png")
        plt.close()
    
    # 4. Batch size scaling for each model-hardware combination
    for model in df_success["model"].unique():
        model_data = df_success[df_success["model"] == model]
        
        for hardware in model_data["hardware"].unique():
            hw_data = model_data[model_data["hardware"] == hardware]
            
            if len(hw_data["batch_size"].unique()) <= 1:
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Plot throughput vs batch size
            for precision in hw_data["precision"].unique():
                precision_data = hw_data[hw_data["precision"] == precision]
                
                if len(precision_data) <= 1:
                    continue
                
                plt.plot(
                    precision_data["batch_size"], 
                    precision_data["throughput"], 
                    marker='o', 
                    label=f"{precision}"
                )
            
            plt.title(f"Batch Size Scaling for {KEY_MODELS[model]['name']} on {HARDWARE_PLATFORMS[hardware]['name']}")
            plt.xlabel("Batch Size")
            plt.ylabel("Throughput (samples/sec)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xscale("log", base=2)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{VISUALIZATION_DIR}/{model}_{hardware}_batch_scaling.png")
            plt.close()
    
    # 5. Training vs Inference comparison for applicable models
    training_data = df[(df["status"] == "success") & (df["mode"] == "training")]
    
    if len(training_data) > 0:
        for model in set(training_data["model"].unique()).intersection(set(df_success["model"].unique())):
            model_training = training_data[training_data["model"] == model]
            model_inference = df_success[df_success["model"] == model]
            
            # Find common hardware platforms
            common_hardware = set(model_training["hardware"].unique()).intersection(set(model_inference["hardware"].unique()))
            
            if not common_hardware:
                continue
            
            plt.figure(figsize=(12, 6))
            
            # Prepare data for plotting
            comparison_data = []
            
            for hw in common_hardware:
                hw_training = model_training[model_training["hardware"] == hw]
                hw_inference = model_inference[model_inference["hardware"] == hw]
                
                if len(hw_training) > 0 and len(hw_inference) > 0:
                    comparison_data.append({
                        "hardware": hw,
                        "training_throughput": hw_training["throughput"].max(),
                        "inference_throughput": hw_inference["throughput"].max()
                    })
            
            if not comparison_data:
                continue
                
            # Convert to DataFrame for plotting
            comparison_df = pd.DataFrame(comparison_data)
            
            # Plot
            x = range(len(comparison_df))
            width = 0.35
            
            plt.bar(
                [i - width/2 for i in x], 
                comparison_df["inference_throughput"],
                width=width,
                label="Inference"
            )
            plt.bar(
                [i + width/2 for i in x], 
                comparison_df["training_throughput"],
                width=width,
                label="Training"
            )
            
            plt.xticks(x, comparison_df["hardware"])
            plt.title(f"Training vs Inference Throughput for {KEY_MODELS[model]['name']}")
            plt.xlabel("Hardware Platform")
            plt.ylabel("Throughput (samples/sec)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{VISUALIZATION_DIR}/{model}_training_vs_inference.png")
            plt.close()
    
    logger.info(f"Generated visualizations in {VISUALIZATION_DIR}")

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Create comprehensive benchmark database for model-hardware combinations")
    
    # Main options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Benchmark all compatible model-hardware combinations")
    group.add_argument("--model", help="Benchmark a specific model across hardware platforms")
    group.add_argument("--hardware", help="Benchmark a specific hardware platform across models")
    group.add_argument("--category", help="Benchmark a specific category of models")
    group.add_argument("--recommendations", action="store_true", help="Generate hardware recommendations")
    group.add_argument("--visualize", action="store_true", help="Generate visualizations")
    group.add_argument("--analyze", action="store_true", help="Generate comparative analysis")
    group.add_argument("--init-db", action="store_true", help="Initialize the benchmark database without running benchmarks")
    
    # Additional options
    parser.add_argument("--batch-sizes", help="Comma-separated list of batch sizes")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32", help="Precision to benchmark")
    parser.add_argument("--training-mode", action="store_true", help="Benchmark in training mode instead of inference")
    parser.add_argument("--compare", action="store_true", help="Generate comparative analysis after benchmarking")
    parser.add_argument("--limit", type=int, help="Limit the number of benchmarks to run")
    parser.add_argument("--output", help="Output file for analysis or recommendations")
    
    args = parser.parse_args()
    
    # Process batch sizes
    batch_sizes = None
    if args.batch_sizes:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    
    # Set mode
    mode = "training" if args.training_mode else "inference"
    
    if args.all:
        print(f"Benchmarking all compatible model-hardware combinations in {mode} mode")
        results = benchmark_all(args.precision, mode, args.limit)
        print(f"Completed {len(results)} benchmarks")
        
        if args.compare:
            print("Generating comparative analysis")
            analysis = generate_comparative_analysis(args.output)
    
    elif args.model:
        print(f"Benchmarking model {args.model} across hardware platforms in {mode} mode")
        
        if args.model not in KEY_MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available models: {', '.join(KEY_MODELS.keys())}")
            sys.exit(1)
            
        results = benchmark_model(args.model, batch_sizes=batch_sizes, precision=args.precision, mode=mode)
        print(f"Completed {len(results)} benchmarks")
        
        if args.compare:
            print("Generating comparative analysis")
            analysis = generate_comparative_analysis(args.output)
    
    elif args.hardware:
        print(f"Benchmarking hardware {args.hardware} across models in {mode} mode")
        
        if args.hardware not in HARDWARE_PLATFORMS:
            print(f"Unknown hardware platform: {args.hardware}")
            print(f"Available platforms: {', '.join(HARDWARE_PLATFORMS.keys())}")
            sys.exit(1)
            
        results = benchmark_hardware(args.hardware, batch_sizes=batch_sizes, precision=args.precision, mode=mode)
        print(f"Completed {len(results)} benchmarks")
        
        if args.compare:
            print("Generating comparative analysis")
            analysis = generate_comparative_analysis(args.output)
    
    elif args.category:
        print(f"Benchmarking category {args.category} across hardware platforms in {mode} mode")
        
        # Check if category exists
        categories = set(info["category"] for info in KEY_MODELS.values())
        if args.category not in categories:
            print(f"Unknown category: {args.category}")
            print(f"Available categories: {', '.join(categories)}")
            sys.exit(1)
            
        results = benchmark_category(args.category, batch_sizes=batch_sizes, precision=args.precision, mode=mode)
        print(f"Completed {len(results)} benchmarks")
        
        if args.compare:
            print("Generating comparative analysis")
            analysis = generate_comparative_analysis(args.output)
    
    elif args.recommendations:
        print("Generating hardware recommendations")
        recommendations = generate_hardware_recommendations()
        
        # Save recommendations to file
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(recommendations, f, indent=2)
            print(f"Saved recommendations to {args.output}")
        else:
            # Print a summary of recommendations
            for model, rec in recommendations.items():
                print(f"\n{rec['model_name']} ({model}):")
                
                if rec["best_overall"]:
                    best = rec["best_overall"]
                    print(f"  Best overall: {HARDWARE_PLATFORMS[best['hardware']]['name']} "
                          f"(precision: {best['precision']}, batch size: {best['batch_size']})")
                    
                if rec["best_throughput"]:
                    best = rec["best_throughput"]
                    print(f"  Best throughput: {HARDWARE_PLATFORMS[best['hardware']]['name']} "
                          f"({best['value']:.2f} samples/sec)")
                    
                if rec["best_latency"]:
                    best = rec["best_latency"]
                    print(f"  Best latency: {HARDWARE_PLATFORMS[best['hardware']]['name']} "
                          f"({best['value']:.2f} ms)")
    
    elif args.visualize:
        print("Generating visualizations")
        generate_visualizations()
        print(f"Saved visualizations to {VISUALIZATION_DIR}")
    
    elif args.analyze:
        print("Generating comparative analysis")
        output_file = args.output or "benchmark_analysis.csv"
        analysis = generate_comparative_analysis(output_file)
        print(f"Saved analysis to {output_file}")
    
    elif args.init_db:
        print("Initializing benchmark database")
        df = create_benchmark_database()
        save_benchmark_database(df)
        print(f"Created benchmark database with {len(df)} entries")

if __name__ == "__main__":
    main()