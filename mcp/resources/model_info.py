#!/usr/bin/env python
"""
IPFS Accelerate MCP Model Information Resource

This module provides information about available AI models as an MCP resource.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_model_info() -> Dict[str, Any]:
    """
    Get information about available models
    
    Returns:
        Dict[str, Any]: Model information
    """
    model_info = {
        "webgpu_models": get_webgpu_models(),
        "webnn_models": get_webnn_models(),
        "ipfs_models": get_ipfs_models(),
        "available_architectures": get_available_architectures(),
    }
    
    return model_info

def get_webgpu_models() -> List[Dict[str, Any]]:
    """
    Get available WebGPU models
    
    Returns:
        List[Dict[str, Any]]: List of WebGPU models
    """
    try:
        # Try to import ipfs_accelerate_py to get WebGPU models
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        
        # Check if we can import webgpu_platform
        try:
            from webgpu_platform import get_compatible_models  # type: ignore
            return get_compatible_models()
        except ImportError:
            logger.warning("Could not import webgpu_platform")
        
        # If that fails, try to directly import from ipfs_accelerate_py
        try:
            from ipfs_accelerate_py import get_compatible_webgpu_models  # type: ignore
            return get_compatible_webgpu_models()
        except ImportError:
            logger.warning("Could not import ipfs_accelerate_py")
        
        # If all else fails, return an empty list
        return []
    
    except Exception as e:
        logger.warning(f"Error getting WebGPU models: {e}")
        return []

def get_webnn_models() -> List[Dict[str, Any]]:
    """
    Get available WebNN models
    
    Returns:
        List[Dict[str, Any]]: List of WebNN models
    """
    try:
        # Try to import ipfs_accelerate_py to get WebNN models
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        
        try:
            from ipfs_accelerate_py import get_compatible_webnn_models  # type: ignore
            return get_compatible_webnn_models()
        except ImportError:
            logger.warning("Could not import ipfs_accelerate_py")
        
        # If all else fails, return an empty list
        return []
    
    except Exception as e:
        logger.warning(f"Error getting WebNN models: {e}")
        return []

def get_ipfs_models() -> List[Dict[str, Any]]:
    """
    Get available IPFS models
    
    Returns:
        List[Dict[str, Any]]: List of IPFS models
    """
    # This would typically query an IPFS node for available models
    # For now, return a sample list of models
    return [
        {
            "name": "llama2-7b-webgpu",
            "description": "Llama 2 7B model optimized for WebGPU",
            "ipfs_cid": "QmSampleCIDForLlama2WebGPU",
            "model_type": "language",
            "architecture": "llama",
            "parameters": "7B",
            "compatible_runtimes": ["webgpu"],
        },
        {
            "name": "stable-diffusion-webnn",
            "description": "Stable Diffusion model optimized for WebNN",
            "ipfs_cid": "QmSampleCIDForStableDiffusionWebNN",
            "model_type": "diffusion",
            "architecture": "stable-diffusion",
            "compatible_runtimes": ["webnn"],
        },
        {
            "name": "whisper-tiny-webgpu",
            "description": "Whisper Tiny model optimized for WebGPU",
            "ipfs_cid": "QmSampleCIDForWhisperTinyWebGPU",
            "model_type": "speech",
            "architecture": "whisper",
            "parameters": "39M",
            "compatible_runtimes": ["webgpu", "webnn"],
        },
    ]

def get_available_architectures() -> Dict[str, List[str]]:
    """
    Get available model architectures
    
    Returns:
        Dict[str, List[str]]: Available architectures by category
    """
    return {
        "language": ["llama", "llama2", "gpt-2", "gpt-neo", "bloom", "bert", "t5"],
        "vision": ["resnet", "efficientnet", "vit", "yolo", "mobilenet"],
        "audio": ["whisper", "wav2vec2", "hubert"],
        "multimodal": ["clip", "stable-diffusion", "dalle"],
    }

def get_model_details(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific model
    
    Args:
        model_name: Name of the model
    
    Returns:
        Optional[Dict[str, Any]]: Model details if found, None otherwise
    """
    # Get all models
    webgpu_models = get_webgpu_models()
    webnn_models = get_webnn_models()
    ipfs_models = get_ipfs_models()
    
    # Combine models
    all_models = webgpu_models + webnn_models + ipfs_models
    
    # Find model by name
    for model in all_models:
        if model.get("name") == model_name:
            return model
    
    return None

if __name__ == "__main__":
    # When run directly, print model info
    model_info = get_model_info()
    
    # Print as formatted JSON
    print(json.dumps(model_info, indent=2))
