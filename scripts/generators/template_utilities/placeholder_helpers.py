#!/usr/bin/env python3
"""
Placeholder helper functions for template rendering.

This module provides utilities for working with template placeholders, including:
- Extraction and validation of placeholders
- Default context generation with hardware detection
- Template rendering with error handling and fallbacks
- Modality detection for model types
"""

import os
import re
import json
import logging
import datetime
from typing import Dict, Any, List, Set, Optional, Tuple

logger = logging.getLogger(__name__)

# Model type modality mapping
MODALITY_TYPES = {
    "text": ["bert", "t5", "llama", "roberta", "gpt2", "qwen"],
    "vision": ["vit", "resnet", "detr"],
    "audio": ["whisper", "wav2vec2", "clap"],
    "multimodal": ["clip", "llava", "xclip"]
}

def get_standard_placeholders() -> Dict[str, Dict[str, Any]]:
    """
    Get standard placeholders and their properties
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of standard placeholders with their properties
    """
    # Standard placeholders used across all templates
    return {
        # Core placeholders
        "model_name": {"description": "Full model name", "default_value": None, "required": True},
        "normalized_name": {"description": "Normalized model name for class names", "default_value": None, "required": True},
        "generated_at": {"description": "Generation timestamp", "default_value": None, "required": True},
        
        # Hardware-related placeholders
        "best_hardware": {"description": "Best available hardware for the model", "default_value": "cpu", "required": False},
        "torch_device": {"description": "PyTorch device to use", "default_value": "cpu", "required": False},
        "has_cuda": {"description": "Boolean indicating CUDA availability", "default_value": "False", "required": False},
        "has_rocm": {"description": "Boolean indicating ROCm availability", "default_value": "False", "required": False},
        "has_mps": {"description": "Boolean indicating MPS availability", "default_value": "False", "required": False},
        "has_openvino": {"description": "Boolean indicating OpenVINO availability", "default_value": "False", "required": False},
        "has_qualcomm": {"description": "Boolean indicating Qualcomm AI Engine availability", "default_value": "False", "required": False},
        "has_samsung": {"description": "Boolean indicating Samsung NPU availability", "default_value": "False", "required": False},
        "has_webnn": {"description": "Boolean indicating WebNN availability", "default_value": "False", "required": False},
        "has_webgpu": {"description": "Boolean indicating WebGPU availability", "default_value": "False", "required": False},
        
        # Model-related placeholders
        "model_family": {"description": "Model family classification", "default_value": "default", "required": False},
        "model_subfamily": {"description": "Model subfamily classification", "default_value": None, "required": False},
        "modality": {"description": "Model modality (text, vision, audio, multimodal)", "default_value": "text", "required": False},
        
        # Template metadata
        "template_type": {"description": "Type of template (test, benchmark, skill)", "default_value": "test", "required": False},
        "template_version": {"description": "Template version", "default_value": "1.0", "required": False},
        "parent_template": {"description": "Parent template name", "default_value": None, "required": False},
    }

def extract_placeholders(template: str) -> Set[str]:
    """
    Extract all placeholders from a template
    
    Args:
        template (str): The template string to extract placeholders from
        
    Returns:
        Set[str]: Set of placeholder names found in the template
    """
    # Find all patterns like {placeholder_name}
    pattern = r'\{([a-zA-Z0-9_]+)\}'
    placeholders = set(re.findall(pattern, template))
    return placeholders

def detect_missing_placeholders(template: str, context: Dict[str, Any]) -> List[str]:
    """
    Detect missing placeholders in a template
    
    Args:
        template (str): The template string to check
        context (Dict[str, Any]): The context dictionary with placeholder values
        
    Returns:
        List[str]: List of placeholder names that are missing from the context
    """
    placeholders = extract_placeholders(template)
    
    # Find placeholders that are not in context
    missing = [p for p in placeholders if p not in context]
    return missing

def validate_placeholders(template: str, context: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate placeholders in a template against a context
    
    Args:
        template (str): The template string to validate
        context (Dict[str, Any]): The context dictionary with placeholder values
        
    Returns:
        Tuple[bool, List[str], List[str]]: Success status, missing required placeholders, and all missing placeholders
    """
    # Get all placeholders in the template
    template_placeholders = extract_placeholders(template)
    
    # Get standard placeholders and their properties
    standard_placeholders = get_standard_placeholders()
    
    # Check for missing placeholders
    all_missing = [p for p in template_placeholders if p not in context]
    
    # Check for missing required placeholders
    missing_required = [
        p for p in all_missing 
        if p in standard_placeholders and standard_placeholders[p]["required"]
    ]
    
    # Success if no required placeholders are missing
    success = len(missing_required) == 0
    
    return success, missing_required, all_missing

def detect_hardware(use_torch: bool = True) -> Dict[str, bool]:
    """
    Detect available hardware platforms
    
    Args:
        use_torch (bool): Whether to use PyTorch for hardware detection
        
    Returns:
        Dict[str, bool]: Dictionary of hardware platform availability
    """
    # Initialize hardware support status for all platforms
    hardware_support = {
        "cpu": True,  # CPU is always available
        "cuda": False,
        "rocm": False,
        "mps": False,
        "openvino": False,
        "qualcomm": False,
        "samsung": False,
        "webnn": False,
        "webgpu": False
    }
    
    # Check for PyTorch-related hardware if requested
    if use_torch:
        try:
            import torch
            
            # Check CUDA
            hardware_support["cuda"] = torch.cuda.is_available()
            
            # Check for ROCm (AMD)
            if hasattr(torch, '_C') and hasattr(torch._C, '_TORCH_ROCM_VERSION'):
                hardware_support["rocm"] = True
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                hardware_support["mps"] = True
        except ImportError:
            logger.warning("PyTorch not available, some hardware detection will be limited")
    
    # Check for OpenVINO
    try:
        import openvino
        hardware_support["openvino"] = True
    except ImportError:
        pass
    
    # Check for Qualcomm AI Engine (QNN)
    try:
        # First try with the actual QNN module
        import qnn
        hardware_support["qualcomm"] = True
    except ImportError:
        # Then check if we have the simulation helper
        try:
            import qnn_simulation_helper
            hardware_support["qualcomm"] = True
        except ImportError:
            # Finally check environment variables for QNN
            if os.environ.get("QNN_SDK_ROOT") or os.environ.get("QUALCOMM_SDK_PATH"):
                hardware_support["qualcomm"] = True
    
    # Check for Samsung NPU
    try:
        import enn
        hardware_support["samsung"] = True
    except ImportError:
        # Check environment variables for Samsung
        if os.environ.get("SAMSUNG_NPU_SDK_ROOT") or os.environ.get("EXYNOS_SDK_PATH"):
            hardware_support["samsung"] = True
    
    # Check for WebNN/WebGPU availability (simulated or browser)
    # WebNN - check for navigator.ml or transformers.js
    try:
        import transformers_js
        hardware_support["webnn"] = True
    except ImportError:
        # Check environment variable for WebNN simulation
        if os.environ.get("WEBNN_AVAILABLE") or os.environ.get("SIMULATE_WEBNN"):
            hardware_support["webnn"] = True
    
    # WebGPU - check for WebGPU or transformers.js
    try:
        import webgpu
        hardware_support["webgpu"] = True
    except ImportError:
        try:
            import transformers_js
            hardware_support["webgpu"] = True
        except ImportError:
            # Check environment variable for WebGPU simulation
            if os.environ.get("WEBGPU_AVAILABLE") or os.environ.get("SIMULATE_WEBGPU"):
                hardware_support["webgpu"] = True
    
    return hardware_support

def get_modality_for_model_type(model_type: str) -> str:
    """
    Determine modality for a given model type
    
    Args:
        model_type (str): The model type to get modality for
        
    Returns:
        str: The modality (text, vision, audio, multimodal) or "unknown"
    """
    model_type_lower = model_type.lower()
    
    for modality, model_types in MODALITY_TYPES.items():
        if model_type_lower in model_types:
            return modality
    
    # Check for partial matches (e.g., "bert-base" should match "bert")
    for modality, model_types in MODALITY_TYPES.items():
        for mt in model_types:
            if mt in model_type_lower:
                return modality
    
    return "unknown"

def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for class names
    
    Args:
        model_name (str): The model name to normalize
        
    Returns:
        str: Normalized model name suitable for class names
    """
    # Replace non-alphanumeric characters with underscores and title case
    return re.sub(r'[^a-zA-Z0-9]', '_', model_name).title()

def get_default_context(model_name: str, hardware_platform: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default context for template rendering with hardware detection
    
    Args:
        model_name (str): Model name to use in the context
        hardware_platform (Optional[str]): Specific hardware platform to prioritize
        
    Returns:
        Dict[str, Any]: Context dictionary with default values
    """
    # Normalize model name for class names
    normalized_name = normalize_model_name(model_name)
    
    # Extract model type from model name (e.g., bert-base-uncased -> bert)
    model_type = None
    for modality, model_types in MODALITY_TYPES.items():
        for mt in model_types:
            if mt in model_name.lower():
                model_type = mt
                break
        if model_type:
            break
    
    if not model_type:
        model_type = "default"
    
    # Determine modality for the model
    modality = get_modality_for_model_type(model_type)
    
    # Detect available hardware
    hardware = detect_hardware()
    
    # Prepare context with hardware information
    context = {
        "model_name": model_name,
        "normalized_name": normalized_name,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "modality": modality,
        "has_cuda": str(hardware.get("cuda", False)).lower(),
        "has_rocm": str(hardware.get("rocm", False)).lower(),
        "has_mps": str(hardware.get("mps", False)).lower(),
        "has_openvino": str(hardware.get("openvino", False)).lower(),
        "has_qualcomm": str(hardware.get("qualcomm", False)).lower(),
        "has_samsung": str(hardware.get("samsung", False)).lower(),
        "has_webnn": str(hardware.get("webnn", False)).lower(),
        "has_webgpu": str(hardware.get("webgpu", False)).lower(),
    }
    
    # Determine best hardware platform
    if hardware_platform and hardware.get(hardware_platform, False):
        context["best_hardware"] = hardware_platform
    elif hardware.get("cuda", False):
        context["best_hardware"] = "cuda"
    elif hardware.get("mps", False):
        context["best_hardware"] = "mps"
    elif hardware.get("rocm", False):
        context["best_hardware"] = "rocm"
    elif hardware.get("openvino", False):
        context["best_hardware"] = "openvino"
    elif hardware.get("qualcomm", False):
        context["best_hardware"] = "qualcomm"
    else:
        context["best_hardware"] = "cpu"
    
    # Set torch device based on best hardware
    if context["best_hardware"] == "cuda":
        context["torch_device"] = "cuda"
    elif context["best_hardware"] == "mps":
        context["torch_device"] = "mps"
    elif context["best_hardware"] == "rocm":
        context["torch_device"] = "cuda"  # ROCm uses CUDA-compatible interface
    else:
        context["torch_device"] = "cpu"
    
    return context

def render_template(template: str, context: Dict[str, Any], strict: bool = False) -> str:
    """
    Render a template with placeholder substitution and fallbacks
    
    Args:
        template (str): The template string to render
        context (Dict[str, Any]): The context dictionary with placeholder values
        strict (bool): Whether to raise an error for missing placeholders
        
    Returns:
        str: The rendered template string
        
    Raises:
        ValueError: If strict is True and required placeholders are missing
    """
    # Ensure all required placeholders are present
    success, missing_required, all_missing = validate_placeholders(template, context)
    
    if missing_required and strict:
        raise ValueError(f"Missing required placeholders: {missing_required}")
    
    if all_missing:
        # Try to fill in defaults
        standard_placeholders = get_standard_placeholders()
        for placeholder in all_missing:
            if placeholder in standard_placeholders and standard_placeholders[placeholder]["default_value"] is not None:
                context[placeholder] = standard_placeholders[placeholder]["default_value"]
                logger.debug(f"Using default value for {placeholder}: {standard_placeholders[placeholder]['default_value']}")
        
        # Check again after filling defaults
        _, missing_required, all_missing = validate_placeholders(template, context)
        
        if missing_required and strict:
            raise ValueError(f"Missing required placeholders after defaults: {missing_required}")
        
        if all_missing:
            logger.warning(f"Missing placeholders after defaults: {all_missing}")
            # For missing placeholders, use a placeholder marker
            for placeholder in all_missing:
                context[placeholder] = f"<<MISSING:{placeholder}>>"
    
    # Render template
    try:
        result = template.format(**context)
        return result
    except KeyError as e:
        placeholder = str(e).strip("'")
        if strict:
            raise ValueError(f"Missing placeholder in template: {placeholder}")
        
        logger.warning(f"Missing placeholder in template: {placeholder}")
        context[placeholder] = f"<<MISSING:{placeholder}>>"
        return template.format(**context)
    except Exception as e:
        if strict:
            raise
        
        logger.error(f"Error rendering template: {e}")
        return f"ERROR RENDERING TEMPLATE: {e}\n\n{template}"