#!/usr/bin/env python3
"""
Fix Template Syntax in All Generator Files

This script creates clean versions of all test generator files
by removing spurious triple-quoted strings and ensuring proper syntax.
"""

import os
import sys
import datetime
from pathlib import Path

# Current directory
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Files to fix
TARGET_FILES = {
    "fixed_merged_test_generator.py": "fixed_merged_test_generator_clean.py",
    "merged_test_generator.py": "merged_test_generator_clean.py",
    "integrated_skillset_generator.py": "integrated_skillset_generator_clean.py"
}

def copy_essential_parts(src_file_path, dest_file_path):
    """
    Copy only the essential working parts of the file, skipping the template strings.
    """
    try:
        with open(src_file_path, 'r', encoding='utf-8') as src_file:
            content = src_file.read()
        
        # Get the docstring and imports
        docstring_end = content.find('"""', content.find('"""') + 3) + 3
        imports_end = content.find('\n\n', content.find('from typing import'))
        
        header = content[:imports_end + 2]
        
        # Get the main function and any supporting classes
        main_function = content.find('def main(')
        if main_function == -1:
            main_function = content.find('if __name__ ==')
        
        footer = content[main_function:]
        
        # Add essential components
        hardware_detect = """
# Hardware detection with web optimization support
import os
import sys
import importlib.util
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardware detection
try:
    from centralized_hardware_detection.hardware_detection import (
        get_capabilities,
        get_web_optimizations,
        get_browser_info,
        get_model_hardware_compatibility
    )
    HAS_CENTRALIZED_HARDWARE = True
except ImportError:
    HAS_CENTRALIZED_HARDWARE = False
    logger.warning("Centralized hardware detection not available")

# Try to import torch first (needed for CUDA/ROCm/MPS)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")

# Initialize hardware capability flags
HAS_CUDA = False
HAS_ROCM = False
HAS_MPS = False
HAS_OPENVINO = False
HAS_WEBNN = False
HAS_WEBGPU = False

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available()
    
    # ROCm detection
    if HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
        HAS_ROCM = True
    elif 'ROCM_HOME' in os.environ:
        HAS_ROCM = True
    
    # Apple MPS detection
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        HAS_MPS = torch.mps.is_available()

# OpenVINO detection
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None

# WebNN detection (browser API)
HAS_WEBNN = (
    importlib.util.find_spec("webnn") is not None or 
    importlib.util.find_spec("webnn_js") is not None or
    "WEBNN_AVAILABLE" in os.environ or
    "WEBNN_SIMULATION" in os.environ
)

# WebGPU detection (browser API)
HAS_WEBGPU = (
    importlib.util.find_spec("webgpu") is not None or
    importlib.util.find_spec("wgpu") is not None or
    "WEBGPU_AVAILABLE" in os.environ or
    "WEBGPU_SIMULATION" in os.environ
)

# Hardware detection function for comprehensive hardware info
def check_hardware():
    """Check available hardware and return capabilities."""
    if HAS_CENTRALIZED_HARDWARE:
        return get_capabilities()
        
    capabilities = {
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False,
        "rocm": False,
        "webnn": False,
        "webgpu": False
    }
    
    # CUDA capabilities
    if HAS_TORCH and HAS_CUDA:
        capabilities["cuda"] = True
        capabilities["cuda_devices"] = torch.cuda.device_count()
        capabilities["cuda_version"] = torch.version.cuda
    
    # MPS capabilities (Apple Silicon)
    capabilities["mps"] = HAS_MPS
    
    # OpenVINO capabilities
    capabilities["openvino"] = HAS_OPENVINO
    
    # ROCm capabilities
    capabilities["rocm"] = HAS_ROCM
    
    # WebNN capabilities
    capabilities["webnn"] = HAS_WEBNN
    
    # WebGPU capabilities
    capabilities["webgpu"] = HAS_WEBGPU
    
    return capabilities

# Get hardware capabilities
HW_CAPABILITIES = check_hardware()

# Web Platform Optimizations - March 2025
def apply_web_platform_optimizations(model_type, implementation_type=None):
    """
    Apply web platform optimizations based on model type and environment settings.
    
    Args:
        model_type: Type of model (audio, multimodal, etc.)
        implementation_type: Implementation type (WebNN, WebGPU)
        
    Returns:
        Dict of optimization settings
    """
    if HAS_CENTRALIZED_HARDWARE:
        return get_web_optimizations(model_type, implementation_type)
        
    optimizations = {
        "compute_shaders": False,
        "parallel_loading": False,
        "shader_precompile": False
    }
    
    # Check for optimization environment flags
    compute_shaders_enabled = (
        os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1" or
        os.environ.get("WEBGPU_COMPUTE_SHADERS", "0") == "1"
    )
    
    parallel_loading_enabled = (
        os.environ.get("WEB_PARALLEL_LOADING_ENABLED", "0") == "1" or
        os.environ.get("PARALLEL_LOADING_ENABLED", "0") == "1"
    )
    
    shader_precompile_enabled = (
        os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1" or
        os.environ.get("WEBGPU_SHADER_PRECOMPILE", "0") == "1"
    )
    
    # Enable all optimizations flag
    if os.environ.get("WEB_ALL_OPTIMIZATIONS", "0") == "1":
        compute_shaders_enabled = True
        parallel_loading_enabled = True
        shader_precompile_enabled = True
    
    # Only apply WebGPU compute shaders for audio models
    if compute_shaders_enabled and implementation_type == "WebGPU" and model_type == "audio":
        optimizations["compute_shaders"] = True
    
    # Only apply parallel loading for multimodal models
    if parallel_loading_enabled and model_type == "multimodal":
        optimizations["parallel_loading"] = True
    
    # Apply shader precompilation for most model types with WebGPU
    if shader_precompile_enabled and implementation_type == "WebGPU":
        optimizations["shader_precompile"] = True
    
    return optimizations

def detect_browser_for_optimizations():
    """
    Detect browser type for optimizations, particularly for Firefox WebGPU compute shader optimizations.
    
    Returns:
        Dict with browser information
    """
    if HAS_CENTRALIZED_HARDWARE:
        return get_browser_info()
        
    # Start with default (simulation environment)
    browser_info = {
        "is_browser": False,
        "browser_type": "unknown",
        "is_firefox": False,
        "is_chrome": False,
        "is_edge": False,
        "is_safari": False,
        "supports_compute_shaders": False,
        "workgroup_size": [128, 1, 1]  # Default workgroup size
    }
    
    # Try to detect browser environment
    try:
        import js
        if hasattr(js, 'navigator'):
            browser_info["is_browser"] = True
            user_agent = js.navigator.userAgent.lower()
            
            # Detect browser type
            if "firefox" in user_agent:
                browser_info["browser_type"] = "firefox"
                browser_info["is_firefox"] = True
                browser_info["supports_compute_shaders"] = True
                browser_info["workgroup_size"] = [256, 1, 1]  # Firefox optimized workgroup size
            elif "chrome" in user_agent:
                browser_info["browser_type"] = "chrome"
                browser_info["is_chrome"] = True
                browser_info["supports_compute_shaders"] = True
            elif "edg" in user_agent:
                browser_info["browser_type"] = "edge"
                browser_info["is_edge"] = True
                browser_info["supports_compute_shaders"] = True
            elif "safari" in user_agent:
                browser_info["browser_type"] = "safari"
                browser_info["is_safari"] = True
                browser_info["supports_compute_shaders"] = False  # Safari has limited compute shader support
    except (ImportError, AttributeError):
        # Not in a browser environment
        pass
    
    # Check environment variables for browser simulation
    if os.environ.get("SIMULATE_FIREFOX", "0") == "1":
        browser_info["browser_type"] = "firefox"
        browser_info["is_firefox"] = True
        browser_info["supports_compute_shaders"] = True
        browser_info["workgroup_size"] = [256, 1, 1]
    
    return browser_info

# Define key model hardware map
KEY_MODEL_HARDWARE_MAP = {
    "bert": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "REAL",
        "mps": "REAL",
        "openvino": "REAL",
        "webnn": "REAL",
        "webgpu": "REAL"
    },
    "t5": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "REAL",
        "mps": "REAL",
        "openvino": "REAL",
        "webnn": "REAL",
        "webgpu": "REAL"
    },
    "llama": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "REAL",
        "mps": "REAL",
        "openvino": "REAL",
        "webnn": "SIMULATION",
        "webgpu": "SIMULATION"
    },
    "clip": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "REAL",
        "mps": "REAL",
        "openvino": "REAL",
        "webnn": "REAL",
        "webgpu": "REAL"
    },
    "vit": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "REAL",
        "mps": "REAL",
        "openvino": "REAL",
        "webnn": "REAL",
        "webgpu": "REAL"
    },
    "clap": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "REAL",
        "mps": "REAL",
        "openvino": "REAL",
        "webnn": "SIMULATION",
        "webgpu": "SIMULATION"
    },
    "whisper": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "REAL",
        "mps": "REAL",
        "openvino": "REAL",
        "webnn": "SIMULATION",
        "webgpu": "SIMULATION"
    },
    "wav2vec2": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "REAL",
        "mps": "REAL",
        "openvino": "REAL",
        "webnn": "SIMULATION",
        "webgpu": "SIMULATION"
    },
    "llava": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "SIMULATION",
        "mps": "SIMULATION",
        "openvino": "SIMULATION",
        "webnn": "SIMULATION",
        "webgpu": "SIMULATION"
    },
    "llava_next": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "SIMULATION",
        "mps": "SIMULATION",
        "openvino": "SIMULATION",
        "webnn": "SIMULATION",
        "webgpu": "SIMULATION"
    },
    "xclip": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "REAL",
        "mps": "REAL",
        "openvino": "REAL",
        "webnn": "SIMULATION",
        "webgpu": "SIMULATION"
    },
    "qwen2": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "SIMULATION",
        "mps": "SIMULATION",
        "openvino": "SIMULATION",
        "webnn": "SIMULATION",
        "webgpu": "SIMULATION"
    },
    "detr": {
        "cpu": "REAL",
        "cuda": "REAL",
        "rocm": "REAL",
        "mps": "REAL",
        "openvino": "REAL",
        "webnn": "SIMULATION",
        "webgpu": "SIMULATION"
    }
}

# Modality types for better model categorization
MODALITY_TYPES = {
    "text": [
        "bert", "roberta", "gpt2", "bloom", "llama", "falcon", "mistral", "phi", "gemma", 
        "t5", "bart", "mt5", "pegasus", "albert", "electra", "distilbert", "reformer", 
        "longformer", "flaubert", "xlm", "xlnet", "deberta", "camembert", "layoutlm",
        "canine", "roformer", "led", "biogpt", "text_embed", "sentence_transformer"
    ],
    "vision": [
        "vit", "detr", "bit", "deit", "beit", "clip_vision", "convnext", "yolos", "dpt",
        "segformer", "perceiver", "owlvit", "convbert", "resnet", "sam", "dinov2", "vilt"
    ],
    "audio": [
        "wav2vec2", "hubert", "whisper", "unispeech", "wavlm", "data2vec_audio", 
        "clap", "encodec", "musicgen", "seamless", "audiolm2", "mms"
    ],
    "multimodal": [
        "clip", "blip", "blip2", "flava", "llava", "siglip", "git", "pix2struct", "owlv2",
        "donut", "paligemma", "idefics", "flamingo", "instructblip"
    ],
    "video": [
        "xclip", "videomae", "vivit", "videobert", "tvlt", "videomaev2"
    ],
    "specialized": [
        "esm", "dnabert", "gptneo", "tapas", "informer", "patchtst", "autoformer"
    ]
}

# Specialized models with specific tasks
SPECIALIZED_MODELS = {
    "esm": "protein-folding",
    "dnabert": "dna-sequence-analysis",
    "tapas": "table-question-answering",
    "patchtst": "time-series-prediction",
    "autoformer": "time-series-prediction",
    "informer": "time-series-prediction"
}

def detect_model_modality(model_type):
    """
    Detect the modality of a model based on its type name
    
    Args:
        model_type (str): The model type/family name (e.g., bert, clip, wav2vec2)
    
    Returns:
        str: One of "text", "vision", "audio", "multimodal", "video", or "specialized"
    """
    # Normalize the model type
    model_name = model_type.lower().replace('-', '_').replace('.', '_')
    
    # Check direct matches in defined modality types
    for modality, types in MODALITY_TYPES.items():
        if model_name in types or any(t in model_name for t in types):
            return modality
    
    # Check for specialized models
    for model_key, task in SPECIALIZED_MODELS.items():
        if model_name == model_key.lower().replace('-', '_').replace('.', '_'):
            # Determine modality based on task
            if 'audio' in task or 'speech' in task:
                return 'audio'
            elif 'image' in task or 'vision' in task or 'depth' in task:
                return 'vision'
            elif 'text-to-text' in task or 'text-generation' in task:
                return 'text'
            elif 'multimodal' in task or ('image' in task and 'text' in task):
                return 'multimodal'
            else:
                return 'specialized'
    
    # Check common patterns
    if any(x in model_name for x in ['text', 'bert', 'gpt', 'llama', 'mistral']):
        return 'text'
    elif any(x in model_name for x in ['vision', 'vit', 'resnet', 'dino']):
        return 'vision'
    elif any(x in model_name for x in ['audio', 'speech', 'whisper', 'wav2vec']):
        return 'audio'
    elif any(x in model_name for x in ['clip', 'llava', 'multimodal']):
        return 'multimodal'
    elif any(x in model_name for x in ['video', 'xclip']):
        return 'video'
    
    # Default to text
    return 'text'

def platform_supported_for_model(model_name, platform):
    """Check if a platform is supported for a specific model."""
    # If we have direct access to the compatibility function, use it
    if HAS_CENTRALIZED_HARDWARE:
        compat = get_model_hardware_compatibility(model_name)
        if platform in compat:
            return compat[platform]
    
    # Load compatibility matrix from key models map
    model_base = model_name.split("-")[0].lower() if "-" in model_name else model_name.lower()
    
    # Check if it's a key model
    if model_base in KEY_MODEL_HARDWARE_MAP:
        if platform in KEY_MODEL_HARDWARE_MAP[model_base]:
            return KEY_MODEL_HARDWARE_MAP[model_base][platform] != "NONE"
    
    # Use modality as a fallback
    modality = detect_model_modality(model_name)
    
    # Default compatibility based on modality
    if modality == "text" or modality == "vision":
        # Text and vision models generally work on all platforms
        return True
    elif modality == "audio":
        # Audio models have limited support on web platforms
        return platform not in ["webnn", "webgpu"] or platform == "cpu"
    elif modality == "multimodal":
        # Multimodal models have limited support outside CPU and CUDA
        return platform in ["cpu", "cuda"]
    elif modality == "video":
        # Video models have limited support on web platforms
        return platform not in ["webnn", "webgpu"] or platform == "cpu"
    else:
        # For unknown modalities, assume only CPU support
        return platform.lower() == "cpu"

# MockHandler class for function stubs
class MockHandler:
    """Mock handler for platforms that do not have real implementations."""
    
    def __init__(self, model_path, platform="cpu"):
        self.model_path = model_path
        self.platform = platform
        print(f"Created mock handler for {platform}")
    
    def __call__(self, *args, **kwargs):
        """Return mock output."""
        print(f"MockHandler for {self.platform} called with {len(args)} args and {len(kwargs)} kwargs")
        return {"output": "MOCK OUTPUT", "implementation_type": f"MOCK_{self.platform.upper()}"}
"""
        
        # Combine the parts
        with open(dest_file_path, 'w', encoding='utf-8') as dest_file:
            dest_file.write(header)
            dest_file.write(hardware_detect)
            dest_file.write(footer)
        
        print(f"Created clean version: {dest_file_path}")
        return True
    
    except Exception as e:
        print(f"Error processing {src_file_path}: {e}")
        return False

def main():
    """Main entry point."""
    success_count = 0
    
    for src_filename, dest_filename in TARGET_FILES.items():
        src_path = CURRENT_DIR / src_filename
        dest_path = CURRENT_DIR / dest_filename
        
        if copy_essential_parts(src_path, dest_path):
            success_count += 1
    
    print(f"\nCreated {success_count} of {len(TARGET_FILES)} clean generator files")
    
    if success_count == len(TARGET_FILES):
        print("\nNow you can use these clean versions:")
        for dest_filename in TARGET_FILES.values():
            print(f"  python {dest_filename} --generate bert --platform all")
    
    return 0 if success_count == len(TARGET_FILES) else 1

if __name__ == "__main__":
    sys.exit(main())