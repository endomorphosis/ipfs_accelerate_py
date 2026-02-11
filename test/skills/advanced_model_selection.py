#!/usr/bin/env python3

"""
Advanced Model Selection for HuggingFace Model Lookup.

This script enhances the model lookup system with:
1. Task-specific model selection based on model capabilities
2. Size-aware model selection based on hardware constraints
3. Advanced filtering by model attributes like license, framework, etc.
4. Smart fallbacks based on model similarity
5. Performance-based model ranking using benchmark data

Usage:
    python advanced_model_selection.py --model-type TYPE --task TASK [--max-size SIZE_MB]
"""

import os
import sys
import json
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Integration with find_models.py
try:
    from find_models import get_recommended_default_model, query_huggingface_api as find_models_query
    HAS_FIND_MODELS = True
    logger.info("find_models.py integration available")
except ImportError:
    HAS_FIND_MODELS = False
    logger.warning("find_models.py not available, using built-in API query")
    find_models_query = None

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_FILE = CURRENT_DIR / "huggingface_model_types.json"
BENCHMARK_DIR = CURRENT_DIR / "benchmark_results"

# Task definitions mapping task names to model types
TASK_TO_MODEL_TYPES = {
    "text-classification": ["bert", "roberta", "distilbert", "albert", "electra", "xlm-roberta"],
    "token-classification": ["bert", "roberta", "distilbert", "electra"],
    "question-answering": ["bert", "roberta", "distilbert", "albert", "electra"],
    "text-generation": ["gpt2", "gpt-j", "gpt-neo", "bloom", "llama", "opt", "mistral", "falcon", "phi"],
    "summarization": ["t5", "bart", "pegasus", "led"],
    "translation": ["t5", "mbart", "m2m_100", "mt5"],
    "image-classification": ["vit", "resnet", "deit", "convnext", "swin"],
    "image-segmentation": ["mask2former", "segformer", "detr"],
    "object-detection": ["yolos", "detr", "mask2former"],
    "image-to-text": ["blip", "git", "pix2struct"],
    "text-to-image": ["stable-diffusion", "dall-e"],
    "automatic-speech-recognition": ["whisper", "wav2vec2", "hubert"],
    "audio-classification": ["wav2vec2", "hubert", "audio-spectrogram-transformer"],
    "visual-question-answering": ["llava", "blip", "git"],
    "document-question-answering": ["layoutlm", "donut", "pix2struct"],
    "fill-mask": ["bert", "roberta", "distilbert", "albert", "electra", "xlm-roberta"]
}

# Model size categories (approximate sizes in MB)
MODEL_SIZE_CATEGORIES = {
    "tiny": 50,      # ~50MB models like DistilBERT-tiny
    "small": 250,    # ~250MB models like BERT-base, ViT-small
    "base": 500,     # ~500MB models like BERT-large, ViT-base
    "large": 1500,   # ~1.5GB models like T5-large
    "xl": 3000,      # ~3GB models like T5-3B
    "xxl": 10000,    # ~10GB models like LLaMA-7B
    "xxxl": 30000    # ~30GB models like LLaMA-13B and larger
}

# Hardware profiles
HARDWARE_PROFILES = {
    "cpu-small": {
        "max_size_mb": 500,  # ~500MB max for CPU-only small systems
        "description": "CPU-only with limited RAM (e.g., CI runners, small VMs)"
    },
    "cpu-medium": {
        "max_size_mb": 2000,  # ~2GB max for standard CPU systems
        "description": "CPU-only with moderate RAM (e.g., laptops, desktops)"
    },
    "cpu-large": {
        "max_size_mb": 10000,  # ~10GB max for large CPU systems
        "description": "CPU-only with large RAM (e.g., servers, high-end desktops)"
    },
    "gpu-small": {
        "max_size_mb": 5000,  # ~5GB max for small GPUs
        "description": "Small GPU systems (e.g., GTX 1060 6GB, T4)"
    },
    "gpu-medium": {
        "max_size_mb": 15000,  # ~15GB max for medium GPUs
        "description": "Medium GPU systems (e.g., RTX 3080 10GB, A10)"
    },
    "gpu-large": {
        "max_size_mb": 50000,  # ~50GB max for large GPUs
        "description": "Large GPU systems (e.g., A100 40GB, multiple GPUs)"
    }
}

# Framework compatibility
FRAMEWORK_COMPATIBILITY = {
    "pytorch": ["transformers", "accelerate", "peft"],
    "tensorflow": ["tensorflow", "keras", "tensorflow-hub"],
    "jax": ["flax", "optax", "jax"],
    "onnx": ["onnxruntime", "onnx", "onnx-tf"]
}

def load_registry_data():
    """Load model registry data from JSON file."""
    try:
        if os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded model registry from {REGISTRY_FILE}")
                return data
        else:
            logger.warning(f"Registry file {REGISTRY_FILE} not found, using empty registry")
            return {}
    except Exception as e:
        logger.error(f"Error loading registry file: {e}")
        return {}

def query_huggingface_api(model_type, limit=10, task=None, size_mb=None, framework=None):
    """Query the HuggingFace API for models with advanced filtering."""
    try:
        logger.info(f"Querying HuggingFace API for {model_type} models (task: {task}, size: {size_mb}MB)")
        
        # Use find_models.py query function if available
        if HAS_FIND_MODELS and find_models_query:
            try:
                # Basic query using find_models.py
                models = find_models_query(model_type, limit=limit)
                
                # Apply additional filtering here for task, size, and framework
                filtered_models = []
                for model in models:
                    # Get model size if available, otherwise estimate
                    model_size = estimate_model_size(model)
                    
                    # Skip models that exceed the size limit
                    if size_mb and model_size and model_size > size_mb:
                        logger.info(f"Skipping {model['id']} (size: ~{model_size}MB, limit: {size_mb}MB)")
                        continue
                    
                    # Skip models that don't match the framework
                    if framework and not model_matches_framework(model, framework):
                        logger.info(f"Skipping {model['id']} (framework mismatch)")
                        continue
                    
                    # Skip models that don't match the task (if specified)
                    if task and not task_matches_model(model, task):
                        logger.info(f"Skipping {model['id']} (task mismatch)")
                        continue
                    
                    filtered_models.append(model)
                
                logger.info(f"Found {len(filtered_models)}/{len(models)} models matching criteria")
                return filtered_models
            except Exception as e:
                logger.warning(f"Error using find_models query: {e}, falling back to direct API query")
        
        # Fall back to direct API query
        import requests
        url = f"https://huggingface.co/api/models?filter={model_type}&sort=downloads&direction=-1&limit={limit}"
        
        # Add task filter if specified
        if task:
            url += f"&filter={task}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        models = response.json()
        
        # If no size limit, return all models
        if not size_mb:
            return models
        
        # Filter models by size if size_mb is specified
        filtered_models = []
        for model in models:
            # Get model size if available, otherwise estimate
            model_size = estimate_model_size(model)
            
            # Skip models that exceed the size limit
            if model_size and model_size > size_mb:
                logger.info(f"Skipping {model['id']} (size: ~{model_size}MB, limit: {size_mb}MB)")
                continue
            
            # Skip models that don't match the framework
            if framework and not model_matches_framework(model, framework):
                logger.info(f"Skipping {model['id']} (framework mismatch)")
                continue
            
            filtered_models.append(model)
        
        logger.info(f"Found {len(filtered_models)}/{len(models)} models matching criteria")
        return filtered_models
        
    except Exception as e:
        logger.error(f"Error querying HuggingFace API: {e}")
        return []

def task_matches_model(model, task):
    """Check if a model is suitable for a specific task."""
    # Extract tags and model ID
    tags = model.get("tags", [])
    model_id = model.get("id", "").lower()
    
    # Check if task is mentioned in tags
    if any(task.lower() in tag.lower() for tag in tags):
        return True
    
    # Check model type based on model_id parts
    model_parts = model_id.split("/")[-1].split("-")
    model_type = model_parts[0].lower()
    
    # See if model_type is in the list for this task
    for model_list in TASK_TO_MODEL_TYPES.get(task, []):
        if model_type in model_list:
            return True
    
    return False
def estimate_model_size(model_info):
    """Estimate model size in MB from model info."""
    # Try to get size from model info if available
    if "size" in model_info:
        return model_info["size"] / (1024 * 1024)  # Convert bytes to MB
    
    # Estimate from model tags
    tags = model_info.get("tags", [])
    
    # Look for size indicators in model ID
    model_id = model_info.get("id", "")
    
    # Check for size category in model name (tiny, small, base, large, etc.)
    size_indicators = [
        ("tiny", MODEL_SIZE_CATEGORIES["tiny"]),
        ("small", MODEL_SIZE_CATEGORIES["small"]),
        ("mini", MODEL_SIZE_CATEGORIES["tiny"]),
        ("base", MODEL_SIZE_CATEGORIES["base"]),
        ("medium", MODEL_SIZE_CATEGORIES["base"]),
        ("large", MODEL_SIZE_CATEGORIES["large"]),
        ("xl", MODEL_SIZE_CATEGORIES["xl"]),
        ("xxl", MODEL_SIZE_CATEGORIES["xxl"]),
    ]
    
    for indicator, size in size_indicators:
        if indicator in model_id.lower():
            return size
    
    # Check for parameter count in tags
    for tag in tags:
        if "parameters" in tag.lower():
            try:
                # Extract parameter count (e.g., "1b-parameters" -> 1000MB)
                param_parts = tag.lower().split("-parameters")[0].strip()
                if "b" in param_parts:
                    # Convert billions to MB (rough estimate: 1B params ~= 4GB)
                    b_params = float(param_parts.replace("b", ""))
                    return b_params * 4000  # 1B params ~= 4000MB model
                elif "m" in param_parts:
                    # Convert millions to MB (rough estimate: 1M params ~= 4MB)
                    m_params = float(param_parts.replace("m", ""))
                    return m_params * 4  # 1M params ~= 4MB model
            except (ValueError, TypeError):
                pass
    
    # Default estimate based on model type
    model_type = next((t for t in ["bert", "gpt2", "t5", "vit", "roberta", "xlm", 
                               "bart", "bloom", "llama", "whisper"] 
                      if t in model_id.lower()), None)
    
    if model_type:
        # Rough default estimates by model type
        type_estimates = {
            "bert": 400,      # ~400MB
            "gpt2": 500,      # ~500MB
            "t5": 700,        # ~700MB
            "vit": 300,       # ~300MB
            "roberta": 450,   # ~450MB
            "xlm": 550,       # ~550MB
            "bart": 600,      # ~600MB
            "bloom": 3500,    # ~3.5GB
            "llama": 7000,    # ~7GB
            "whisper": 250,   # ~250MB
        }
        return type_estimates.get(model_type)
    
    # If all else fails, return None (unknown size)
    return None

def model_matches_framework(model_info, framework):
    """Check if model is compatible with specified framework."""
    # Extract tags and library info
    tags = model_info.get("tags", [])
    model_id = model_info.get("id", "")
    
    # Look for framework indicators in tags
    framework_indicators = {
        "pytorch": ["pytorch", "torch", "peft", "transformers"],
        "tensorflow": ["tensorflow", "tf", "keras"],
        "jax": ["jax", "flax", "optax"],
        "onnx": ["onnx", "onnxruntime"]
    }
    
    # If framework is not one of our known ones, return True (no filtering)
    if framework not in framework_indicators:
        return True
    
    # Check tags for framework indicators
    for tag in tags:
        for indicator in framework_indicators[framework]:
            if indicator.lower() in tag.lower():
                return True
    
    # Most HuggingFace models support PyTorch by default
    if framework == "pytorch" and not any(
        indicator in model_id.lower() for indicator in 
        ["tensorflow", "tf", "jax", "flax", "onnx"]):
        return True
    
    return False

def get_models_for_task(task, size_mb=None, framework=None):
    """Get suitable models for a specific task with hardware constraints."""
    # Get model types suitable for this task
    model_types = TASK_TO_MODEL_TYPES.get(task, [])
    
    if not model_types:
        logger.warning(f"No model types found for task: {task}")
        return []
    
    logger.info(f"Finding models for task '{task}' using types: {', '.join(model_types)}")
    
    all_models = []
    for model_type in model_types:
        # Query models for this type
        models = query_huggingface_api(
            model_type, 
            limit=5,
            task=task,
            size_mb=size_mb,
            framework=framework
        )
        all_models.extend(models)
    
    # Sort by downloads
    all_models.sort(key=lambda m: m.get("downloads", 0), reverse=True)
    
    # Take top matches
    return all_models[:10]  # Return top 10 overall

def get_hardware_profile(profile_name=None):
    """Get hardware profile by name or detect automatically."""
    if profile_name and profile_name in HARDWARE_PROFILES:
        return HARDWARE_PROFILES[profile_name]
    
    # Auto-detect hardware
    try:
        # Check for CUDA availability if PyTorch is installed
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                gpu_count = torch.cuda.device_count()
                # Get GPU memory
                total_memory = 0
                for i in range(gpu_count):
                    total_memory += torch.cuda.get_device_properties(i).total_memory
                
                # Convert to MB
                total_memory_mb = total_memory / (1024 * 1024)
                
                if total_memory_mb > 30000:  # >30GB
                    return HARDWARE_PROFILES["gpu-large"]
                elif total_memory_mb > 8000:  # >8GB
                    return HARDWARE_PROFILES["gpu-medium"]
                else:
                    return HARDWARE_PROFILES["gpu-small"]
            else:
                # CPU only, check memory
                import psutil
                memory_mb = psutil.virtual_memory().total / (1024 * 1024)
                
                if memory_mb > 16000:  # >16GB
                    return HARDWARE_PROFILES["cpu-large"]
                elif memory_mb > 8000:  # >8GB
                    return HARDWARE_PROFILES["cpu-medium"]
                else:
                    return HARDWARE_PROFILES["cpu-small"]
        except ImportError:
            # If torch not available, estimate based on total system memory
            import psutil
            memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            
            if memory_mb > 16000:  # >16GB
                return HARDWARE_PROFILES["cpu-large"]
            elif memory_mb > 8000:  # >8GB
                return HARDWARE_PROFILES["cpu-medium"]
            else:
                return HARDWARE_PROFILES["cpu-small"]
    except Exception as e:
        logger.warning(f"Error detecting hardware profile: {e}")
        # Default to cpu-medium as a safe fallback
        return HARDWARE_PROFILES["cpu-medium"]

def get_benchmark_data(model_name):
    """Get benchmark data for a model if available."""
    # Try to find benchmark files
    benchmark_files = []
    if os.path.exists(BENCHMARK_DIR):
        for filename in os.listdir(BENCHMARK_DIR):
            # Simple name match for now
            model_id_parts = model_name.split("/")[-1].lower().split("-")
            if any(part in filename.lower() for part in model_id_parts):
                benchmark_files.append(os.path.join(BENCHMARK_DIR, filename))
    
    if not benchmark_files:
        return None
    
    # Load the first matching benchmark file
    try:
        with open(benchmark_files[0], 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.warning(f"Error loading benchmark data: {e}")
        return None

def select_model_advanced(model_type, task=None, hardware_profile=None, max_size_mb=None, framework=None):
    """Advanced model selection with task, hardware, and framework constraints."""
    logger.info(f"Advanced model selection for {model_type} (task: {task}, hardware: {hardware_profile})")
    
    # Load registry data
    registry_data = load_registry_data()
    
    # Get hardware constraints
    if hardware_profile and hardware_profile in HARDWARE_PROFILES:
        profile = HARDWARE_PROFILES[hardware_profile]
        size_limit = profile["max_size_mb"]
        logger.info(f"Using hardware profile: {hardware_profile} (max size: {size_limit}MB)")
    else:
        # Auto-detect if not specified
        profile = get_hardware_profile()
        size_limit = profile["max_size_mb"]
        logger.info(f"Auto-detected hardware profile: {size_limit}MB limit")
    
    # Override with explicit size if provided
    if max_size_mb:
        size_limit = max_size_mb
        logger.info(f"Using explicit size limit: {size_limit}MB")
    
    # Try task-specific model selection if task is provided
    if task:
        # Check if task is valid
        if task in TASK_TO_MODEL_TYPES:
            # Check if the model type is suitable for this task
            if model_type in TASK_TO_MODEL_TYPES[task]:
                logger.info(f"Model type {model_type} is suitable for task {task}")
                
                # Query specific task models with hardware constraints
                models = query_huggingface_api(
                    model_type, 
                    limit=10,
                    task=task,
                    size_mb=size_limit,
                    framework=framework
                )
                
                if models:
                    # Find models with benchmark data
                    models_with_benchmark = []
                    for model in models:
                        benchmark = get_benchmark_data(model["id"])
                        if benchmark:
                            model["benchmark"] = benchmark
                            models_with_benchmark.append(model)
                    
                    # Prioritize models with benchmark data if available
                    if models_with_benchmark:
                        logger.info(f"Found {len(models_with_benchmark)} models with benchmark data")
                        # Sort by benchmark score (lower is better)
                        models_with_benchmark.sort(
                            key=lambda m: m["benchmark"].get("inference_time", float("inf")))
                        
                        selected_model = models_with_benchmark[0]["id"]
                        logger.info(f"Selected model based on benchmark: {selected_model}")
                        return selected_model
                    
                    # Otherwise select by downloads
                    selected_model = models[0]["id"]
                    logger.info(f"Selected model based on popularity: {selected_model}")
                    return selected_model
                
                logger.warning("No suitable models found with API query")
            else:
                logger.warning(f"Model type {model_type} is not recommended for task {task}")
        else:
            logger.warning(f"Unknown task: {task}")
    
    # Fall back to registry lookup with size constraint
    if model_type in registry_data:
        reg_entry = registry_data[model_type]
        
        # Try to find a suitable model from registry considering size
        if "models" in reg_entry:
            models = reg_entry["models"]
            downloads = reg_entry.get("downloads", {})
            
            # Filter models by size if possible
            size_filtered_models = []
            for model_id in models:
                # Try to estimate size
                model_size = estimate_model_size({"id": model_id})
                if model_size is None or model_size <= size_limit:
                    size_filtered_models.append(model_id)
            
            if size_filtered_models:
                # Sort by downloads
                size_filtered_models.sort(
                    key=lambda m: downloads.get(m, 0), reverse=True)
                
                selected_model = size_filtered_models[0]
                logger.info(f"Selected model from registry by size and popularity: {selected_model}")
                return selected_model
            
            logger.warning("No models in registry match size constraint")
        
        # Fall back to default model if it fits size constraint
        default_model = reg_entry.get("default_model")
        if default_model:
            model_size = estimate_model_size({"id": default_model})
            if model_size is None or model_size <= size_limit:
                logger.info(f"Using default model from registry: {default_model}")
                return default_model
            else:
                logger.warning(f"Default model {default_model} exceeds size constraint")
    
    # Final fallback: use a minimal size variant if available
    small_variants = [
        f"{model_type}-base", 
        f"{model_type}-small", 
        f"{model_type}-mini", 
        f"{model_type}-tiny"
    ]
    
    for variant in small_variants:
        logger.info(f"Trying fallback model: {variant}")
        model_size = estimate_model_size({"id": variant})
        if model_size is None or model_size <= size_limit:
            return variant
    
    # If all else fails, just return the model type itself
    logger.warning(f"No suitable model found for {model_type} within size constraint")
    return model_type

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Advanced model selection")
    parser.add_argument("--model-type", type=str, required=True, help="Model type to select")
    parser.add_argument("--task", type=str, help="Specific task for model selection")
    parser.add_argument("--hardware", type=str, choices=HARDWARE_PROFILES.keys(), 
                      help="Hardware profile for size constraints")
    parser.add_argument("--max-size", type=int, help="Maximum model size in MB")
    parser.add_argument("--framework", type=str, choices=list(FRAMEWORK_COMPATIBILITY.keys()),
                      help="Framework compatibility")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--list-hardware", action="store_true", help="List available hardware profiles")
    parser.add_argument("--detect-hardware", action="store_true", help="Detect hardware profile")
    
    args = parser.parse_args()
    
    # List tasks if requested
    if args.list_tasks:
        print("\nAvailable Tasks:")
        for task, model_types in sorted(TASK_TO_MODEL_TYPES.items()):
            print(f"  - {task}: {', '.join(model_types[:3])}{'...' if len(model_types) > 3 else ''}")
        return 0
    
    # List hardware profiles if requested
    if args.list_hardware:
        print("\nAvailable Hardware Profiles:")
        for name, profile in sorted(HARDWARE_PROFILES.items()):
            print(f"  - {name}: {profile['description']} (max size: {profile['max_size_mb']}MB)")
        return 0
    
    # Detect hardware if requested
    if args.detect_hardware:
        profile = get_hardware_profile()
        print(f"\nDetected Hardware Profile:")
        print(f"  Max model size: {profile['max_size_mb']}MB")
        print(f"  Description: {profile['description']}")
        return 0
    
    # Select model with advanced options
    selected_model = select_model_advanced(
        args.model_type,
        task=args.task,
        hardware_profile=args.hardware,
        max_size_mb=args.max_size,
        framework=args.framework
    )
    
    print(f"\nSelected Model: {selected_model}")
    
    # If a task was provided, also show other recommended models for the task
    if args.task:
        print(f"\nOther recommended models for {args.task}:")
        task_models = get_models_for_task(
            args.task, 
            size_mb=args.max_size or (HARDWARE_PROFILES[args.hardware]["max_size_mb"] 
                                      if args.hardware else None),
            framework=args.framework
        )
        
        for i, model in enumerate(task_models[:5], 1):
            model_id = model.get("id", "")
            if model_id != selected_model:
                downloads = model.get("downloads", 0)
                downloads_str = f"{downloads:,}" if downloads else "Unknown"
                print(f"  {i}. {model_id} (Downloads: {downloads_str})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())