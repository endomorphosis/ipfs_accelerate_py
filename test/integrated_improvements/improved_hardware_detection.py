#!/usr/bin/env python3
"""
Improved Hardware Detection Module

This module provides consolidated hardware detection functionality to be used by
all test generators and benchmark runners in the framework. It eliminates code duplication
and ensures consistent hardware detection across the codebase.

Usage:
  from improvements.improved_hardware_detection import detect_all_hardware, HAS_CUDA, HAS_MPS, etc.
"""

import os
import logging
import importlib.util
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import torch first (needed for CUDA/ROCm/MPS)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("torch not available, hardware detection will be limited")
    # Use MagicMock to avoid failures when accessing torch attributes
    from unittest.mock import MagicMock
    torch = MagicMock()

# Initialize hardware capability flags at module level
HAS_CUDA = False
HAS_ROCM = False
HAS_MPS = False
HAS_OPENVINO = False
HAS_QUALCOMM = False
HAS_WEBNN = False
HAS_WEBGPU = False

# Hardware detection function
def detect_all_hardware() -> Dict[str, Any]:
    """
    Comprehensive hardware detection function that checks for all supported hardware platforms.
    
    Returns:
        Dict with hardware detection results for all platforms
    """
    global HAS_CUDA, HAS_ROCM, HAS_MPS, HAS_OPENVINO, HAS_QUALCOMM, HAS_WEBNN, HAS_WEBGPU
    
    # Initialize capabilities dictionary
    capabilities = {
        "cpu": {"detected": True, "version": None, "devices": 1},
        "cuda": {"detected": False, "version": None, "devices": 0},
        "rocm": {"detected": False, "version": None, "devices": 0},
        "mps": {"detected": False, "version": None, "devices": 0},
        "openvino": {"detected": False, "version": None, "devices": 0},
        "qualcomm": {"detected": False, "version": None, "devices": 0},
        "webnn": {"detected": False, "version": None, "simulation": False},
        "webgpu": {"detected": False, "version": None, "simulation": False}
    }
    
    try:
        # CUDA detection
        if HAS_TORCH:
            cuda_available = torch.cuda.is_available()
            HAS_CUDA = cuda_available
            
            if cuda_available:
                capabilities["cuda"]["detected"] = True
                capabilities["cuda"]["devices"] = torch.cuda.device_count()
                capabilities["cuda"]["version"] = torch.version.cuda
                
                # ROCm detection (shows up as CUDA in torch)
                if hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
                    HAS_ROCM = True
                    capabilities["rocm"]["detected"] = True
                    capabilities["rocm"]["version"] = getattr(torch._C, '_rocm_version', None)
                    capabilities["rocm"]["devices"] = torch.cuda.device_count()
                elif 'ROCM_HOME' in os.environ:
                    HAS_ROCM = True
                    capabilities["rocm"]["detected"] = True
                    capabilities["rocm"]["devices"] = torch.cuda.device_count()
            
            # Apple MPS detection (Metal Performance Shaders)
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
                mps_available = torch.mps.is_available()
                HAS_MPS = mps_available
                
                if mps_available:
                    capabilities["mps"]["detected"] = True
                    capabilities["mps"]["devices"] = 1  # MPS typically has 1 device
        
        # OpenVINO detection
        openvino_spec = importlib.util.find_spec("openvino")
        openvino_available = openvino_spec is not None
        HAS_OPENVINO = openvino_available
        
        if openvino_available:
            capabilities["openvino"]["detected"] = True
            try:
                import openvino
                capabilities["openvino"]["version"] = openvino.__version__
            except (ImportError, AttributeError):
                pass
        
        # WebNN detection (browser API or simulation)
        webnn_available = (
            importlib.util.find_spec("webnn") is not None or 
            importlib.util.find_spec("webnn_js") is not None or
            "WEBNN_AVAILABLE" in os.environ or
            "WEBNN_SIMULATION" in os.environ
        )
        HAS_WEBNN = webnn_available
        
        if webnn_available:
            capabilities["webnn"]["detected"] = True
            capabilities["webnn"]["simulation"] = "WEBNN_SIMULATION" in os.environ
        
        # Qualcomm AI Engine detection
        qualcomm_available = (
            importlib.util.find_spec("qnn_wrapper") is not None or
            importlib.util.find_spec("qti") is not None or
            "QUALCOMM_SDK" in os.environ
        )
        HAS_QUALCOMM = qualcomm_available
        
        if qualcomm_available:
            capabilities["qualcomm"]["detected"] = True
            # Try to get version information
            try:
                import qti
                capabilities["qualcomm"]["version"] = getattr(qti, "__version__", "unknown")
            except (ImportError, AttributeError):
                pass

        # WebGPU detection (browser API or simulation)
        webgpu_available = (
            importlib.util.find_spec("webgpu") is not None or
            importlib.util.find_spec("wgpu") is not None or
            "WEBGPU_AVAILABLE" in os.environ or
            "WEBGPU_SIMULATION" in os.environ
        )
        HAS_WEBGPU = webgpu_available
        
        if webgpu_available:
            capabilities["webgpu"]["detected"] = True
            capabilities["webgpu"]["simulation"] = "WEBGPU_SIMULATION" in os.environ
            
            # Check for web platform optimizations
            capabilities["webgpu"]["optimizations"] = {
                "compute_shaders": os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1",
                "parallel_loading": os.environ.get("WEB_PARALLEL_LOADING_ENABLED", "0") == "1",
                "shader_precompile": os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1"
            }
    
    except Exception as e:
        logger.error(f"Error during hardware detection: {e}")
        # If an error occurs, we default to only CPU being available
    
    # Store hardware capabilities as global reference
    global HARDWARE_CAPABILITIES
    HARDWARE_CAPABILITIES = capabilities
    
    return capabilities

# Web Platform Optimization Functions
def apply_web_platform_optimizations(model_type: str, platform: str = "webgpu") -> Dict[str, bool]:
    """
    Apply web platform optimizations based on model type and environment settings.
    
    Args:
        model_type: Type of model (audio, multimodal, etc.)
        platform: Platform type (WebNN or WebGPU)
        
    Returns:
        Dict of optimization settings
    """
    model_type = model_type.lower() if model_type else ""
    
    # Default optimization settings
    optimizations = {
        "compute_shaders": False,
        "parallel_loading": False,
        "shader_precompile": False
    }
    
    # Only apply optimizations for WebGPU
    if platform.lower() != "webgpu" and platform.lower() != "browser_gpu":
        return optimizations
    
    # Check for optimization environment flags
    compute_shaders_enabled = (
        os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1" or
        os.environ.get("WEBGPU_COMPUTE_SHADERS", "0") == "1"
    )
    
    parallel_loading_enabled = (
        os.environ.get("WEB_PARALLEL_LOADING_ENABLED", "0") == "1" or
        os.environ.get("WEB_PARALLEL_LOADING", "0") == "1"
    )
    
    shader_precompile_enabled = (
        os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1" or
        os.environ.get("WEBGPU_SHADER_PRECOMPILE", "0") == "1"
    )
    
    # Check --all-optimizations flag via environment variable
    all_optimizations = os.environ.get("WEB_ALL_OPTIMIZATIONS", "0") == "1"
    
    if all_optimizations:
        compute_shaders_enabled = True
        parallel_loading_enabled = True
        shader_precompile_enabled = True
    
    # Determine model category
    is_audio_model = any(x in model_type for x in ["audio", "whisper", "wav2vec", "clap", "encodec", "speech"])
    is_multimodal_model = any(x in model_type for x in ["llava", "clip", "xclip", "visual", "vision-text", "multimodal"])
    is_vision_model = any(x in model_type for x in ["vision", "image", "vit", "cnn", "convnext", "resnet"])
    
    # Apply optimization recommendations based on model type
    if compute_shaders_enabled:
        # Audio models benefit most from compute shader optimizations
        optimizations["compute_shaders"] = is_audio_model or all_optimizations
    
    if parallel_loading_enabled:
        # Multimodal models benefit most from parallel loading
        optimizations["parallel_loading"] = is_multimodal_model or all_optimizations
    
    if shader_precompile_enabled:
        # All models benefit from shader precompilation, but especially vision models
        optimizations["shader_precompile"] = True
    
    return optimizations

def get_hardware_compatibility_matrix() -> Dict[str, Dict[str, str]]:
    """
    Returns the hardware compatibility matrix for key model types.
    
    The matrix indicates whether each model type should use REAL implementation,
    SIMULATION, or is not supported on each hardware platform.
    
    Returns:
        Dict mapping model types to hardware platform compatibility
    """
    # Default compatibility - most models work on most platforms
    default_compat = {
        "cpu": "REAL",
        "cuda": "REAL", 
        "openvino": "REAL", 
        "mps": "REAL", 
        "rocm": "REAL",
        "webnn": "REAL", 
        "webgpu": "REAL"
    }
    
    # Text embedding models - good compatibility everywhere
    text_embedding_compat = {
        "cpu": "REAL",
        "cuda": "REAL", 
        "openvino": "REAL", 
        "mps": "REAL", 
        "rocm": "REAL",
        "webnn": "REAL", 
        "webgpu": "REAL"
    }
    
    # Vision models - good compatibility everywhere
    vision_compat = {
        "cpu": "REAL",
        "cuda": "REAL", 
        "openvino": "REAL", 
        "mps": "REAL", 
        "rocm": "REAL",
        "webnn": "REAL", 
        "webgpu": "REAL"
    }
    
    # Audio models - limited web support
    audio_compat = {
        "cpu": "REAL",
        "cuda": "REAL", 
        "openvino": "REAL", 
        "mps": "REAL", 
        "rocm": "REAL",
        "webnn": "SIMULATION", 
        "webgpu": "SIMULATION"
    }
    
    # Multimodal models - limited support outside CUDA
    multimodal_compat = {
        "cpu": "REAL",
        "cuda": "REAL", 
        "openvino": "SIMULATION", 
        "mps": "SIMULATION", 
        "rocm": "SIMULATION",
        "webnn": "SIMULATION", 
        "webgpu": "SIMULATION"
    }
    
    # Build the full matrix
    compatibility_matrix = {
        # Text embedding models
        "bert": text_embedding_compat,
        "roberta": text_embedding_compat,
        "distilbert": text_embedding_compat,
        "albert": text_embedding_compat,
        "bart": text_embedding_compat,
        
        # Text generation models
        "t5": text_embedding_compat,
        "gpt2": text_embedding_compat,
        "llama": text_embedding_compat,
        "opt": text_embedding_compat,
        "bloom": text_embedding_compat,
        
        # Vision models
        "vit": vision_compat,
        "resnet": vision_compat,
        "convnext": vision_compat,
        "clip": vision_compat,
        "detr": vision_compat,
        
        # Audio models
        "whisper": audio_compat,
        "wav2vec2": audio_compat,
        "clap": audio_compat,
        "encodec": audio_compat,
        
        # Multimodal models
        "llava": multimodal_compat,
        "llava-next": multimodal_compat,
        "xclip": audio_compat,
        "qwen2": multimodal_compat,
        "qwen3": multimodal_compat
    }
    
    return compatibility_matrix

# Initialize hardware capabilities by running detection at module load time
HARDWARE_CAPABILITIES = detect_all_hardware()

# Key hardware platforms supported by the framework
HARDWARE_PLATFORMS = ["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"]

# Matrix of key model compatibility with hardware platforms
KEY_MODEL_HARDWARE_MATRIX = get_hardware_compatibility_matrix()

# Export the hardware detection function and flags
__all__ = [
    'detect_all_hardware', 
    'apply_web_platform_optimizations',
    'get_hardware_compatibility_matrix',
    'HAS_CUDA', 
    'HAS_ROCM', 
    'HAS_MPS', 
    'HAS_OPENVINO', 
    'HAS_WEBNN', 
    'HAS_WEBGPU',
    'HARDWARE_CAPABILITIES',
    'HARDWARE_PLATFORMS',
    'KEY_MODEL_HARDWARE_MATRIX'
]