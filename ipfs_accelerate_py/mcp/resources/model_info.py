"""
IPFS Accelerate MCP Model Information Resources

This module provides model information resources for the IPFS Accelerate MCP server.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List, Union

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.resources.model_info")

def register_model_info_resources(mcp: Any) -> None:
    """
    Register model information resources with the MCP server
    
    This function registers model information resources with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.debug("Registering model information resources")
    
    try:
        # Register supported models resource
        mcp.register_resource(
            uri="ipfs_accelerate/supported_models",
            function=get_supported_models,
            description="Information about supported models"
        )
        
        logger.debug("Model information resources registered")
    
    except Exception as e:
        logger.error(f"Error registering model information resources: {e}")
        raise

def get_supported_models() -> Dict[str, Any]:
    """
    Get information about supported models
    
    Returns:
        Dictionary with supported models information
    """
    logger.debug("Getting supported models information")
    
    try:
        # Try to import ipfs_accelerate_py
        try:
            import ipfs_accelerate_py
            
            # Use ipfs_accelerate_py's model information if available
            if hasattr(ipfs_accelerate_py, "get_supported_models"):
                supported_models = ipfs_accelerate_py.get_supported_models()
                
                # Return models if available
                if isinstance(supported_models, dict):
                    logger.debug("Supported models information retrieved from IPFS Accelerate")
                    return supported_models
        except ImportError:
            pass
        
        # Fallback to default model information
        supported_models = get_default_supported_models()
        
        logger.debug("Supported models information retrieved")
        
        return supported_models
    
    except Exception as e:
        logger.error(f"Error getting supported models information: {e}")
        raise

def get_default_supported_models() -> Dict[str, Any]:
    """
    Get default information about supported models
    
    This function returns default information about supported models when
    IPFS Accelerate is not available or does not provide this information.
    
    Returns:
        Dictionary with supported models information
    """
    # Default model categories
    model_categories = {
        "llm": {
            "description": "Large Language Models",
            "models": [
                {
                    "name": "llama-7b",
                    "description": "Meta's Llama 7B language model",
                    "parameters": 7,
                    "quantization": ["none", "int8", "int4"],
                    "accelerators": ["cuda", "cpu"],
                    "tags": ["language", "text-generation"],
                    "memory_requirements": {
                        "fp32": 28,
                        "fp16": 14,
                        "int8": 7,
                        "int4": 3.5
                    }
                },
                {
                    "name": "llama-13b",
                    "description": "Meta's Llama 13B language model",
                    "parameters": 13,
                    "quantization": ["none", "int8", "int4"],
                    "accelerators": ["cuda", "cpu"],
                    "tags": ["language", "text-generation"],
                    "memory_requirements": {
                        "fp32": 52,
                        "fp16": 26,
                        "int8": 13,
                        "int4": 6.5
                    }
                }
            ]
        },
        "vision": {
            "description": "Computer Vision Models",
            "models": [
                {
                    "name": "clip",
                    "description": "OpenAI's CLIP (Contrastive Language-Image Pre-Training) model",
                    "parameters": 0.4,
                    "quantization": ["none", "int8"],
                    "accelerators": ["cuda", "cpu"],
                    "tags": ["vision", "multimodal", "image-classification"],
                    "memory_requirements": {
                        "fp32": 1.6,
                        "fp16": 0.8,
                        "int8": 0.4
                    }
                }
            ]
        },
        "audio": {
            "description": "Audio Processing Models",
            "models": [
                {
                    "name": "whisper",
                    "description": "OpenAI's Whisper speech recognition model",
                    "parameters": 1.5,
                    "quantization": ["none", "int8"],
                    "accelerators": ["cuda", "cpu"],
                    "tags": ["audio", "speech-recognition"],
                    "memory_requirements": {
                        "fp32": 6,
                        "fp16": 3,
                        "int8": 1.5
                    }
                }
            ]
        },
        "multimodal": {
            "description": "Multimodal Models",
            "models": [
                {
                    "name": "llava",
                    "description": "LLaVA (Large Language and Vision Assistant)",
                    "parameters": 7,
                    "quantization": ["none", "int8", "int4"],
                    "accelerators": ["cuda", "cpu"],
                    "tags": ["multimodal", "vision-language"],
                    "memory_requirements": {
                        "fp32": 28,
                        "fp16": 14,
                        "int8": 7,
                        "int4": 3.5
                    }
                }
            ]
        }
    }
    
    # Add IPFS Accelerate optimizations
    for category in model_categories.values():
        for model in category["models"]:
            model["ipfs_accelerate"] = {
                "optimized": True,
                "webgpu_support": model["name"] in ["llama-7b", "clip", "whisper"],
                "webnn_support": model["name"] in ["llama-7b", "clip"],
                "quantization_support": model["name"] in ["llama-7b", "llama-13b", "llava"],
                "recommended_settings": {
                    "cuda": {
                        "batch_size": 1,
                        "dtype": "float16",
                        "quantization": "int4" if "int4" in model["quantization"] else "none"
                    },
                    "cpu": {
                        "batch_size": 1,
                        "dtype": "float32",
                        "quantization": "int8" if "int8" in model["quantization"] else "none"
                    }
                }
            }
    
    # Construct the response
    return {
        "count": sum(len(category["models"]) for category in model_categories.values()),
        "categories": model_categories,
        "ipfs_accelerate": {
            "optimization_available": True,
            "supported_accelerators": ["cuda", "cpu", "webgpu", "webnn"],
            "supported_quantization": ["none", "int8", "int4"],
            "model_compatibility": {
                "llm": ["cuda", "cpu", "webgpu", "webnn"],
                "vision": ["cuda", "cpu", "webgpu"],
                "audio": ["cuda", "cpu"],
                "multimodal": ["cuda", "cpu"]
            }
        }
    }
