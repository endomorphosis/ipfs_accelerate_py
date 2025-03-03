#!/usr/bin/env python
"""
Hardware-Model Integration Module for the IPFS Accelerate framework.
This module provides integration between hardware detection and model classification
systems with robust error handling for resilient operations.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model family hardware preference mapping
MODEL_FAMILY_DEVICE_PREFERENCES = {
    "embedding": ["cuda", "mps", "openvino", "rocm", "webnn", "cpu"],  # Embedding models work well on all hardware
    "text_generation": ["cuda", "rocm", "cpu", "mps", "webgpu"],        # LLMs prefer CUDA, ROCm for performance
    "vision": ["cuda", "openvino", "mps", "rocm", "webgpu", "webnn", "cpu"],  # Vision models do well with OpenVINO and WebGPU
    "audio": ["cuda", "cpu", "rocm", "mps", "webnn", "webgpu"],        # Audio models with expanded web platform support
    "multimodal": ["cuda", "rocm", "mps", "webgpu", "cpu"]            # Multimodal models with expanded platform support
}

# Memory requirements by model family (in MB)
MODEL_FAMILY_MEMORY_REQUIREMENTS = {
    "embedding": {
        "small": 250,
        "medium": 500,
        "large": 1000
    },
    "text_generation": {
        "small": 500,
        "medium": 2000,
        "large": 8000
    },
    "vision": {
        "small": 300,
        "medium": 800,
        "large": 2000
    },
    "audio": {
        "small": 400,
        "medium": 1000,
        "large": 3000
    },
    "multimodal": {
        "small": 1000,
        "medium": 4000,
        "large": 10000
    }
}

# Hardware compatibility by model family
MODEL_FAMILY_HARDWARE_COMPATIBILITY = {
    "embedding": {
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": True,
        "webnn": True,
        "webgpu": True,
        "cpu": True
    },
    "text_generation": {
        "cuda": True,
        "rocm": True,
        "mps": {"constraint": "size", "max_size": "medium"},
        "openvino": {"constraint": "size", "max_size": "small"},
        "webnn": {"constraint": "size", "max_size": "small"},
        "webgpu": {"constraint": "size", "max_size": "small"},
        "cpu": True
    },
    "vision": {
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": True,
        "webnn": {"constraint": "size", "max_size": "medium"},
        "webgpu": {"constraint": "size", "max_size": "medium"},
        "cpu": True
    },
    "audio": {
        "cuda": True,
        "rocm": True,
        "mps": {"constraint": "size", "max_size": "medium"},
        "openvino": {"constraint": "size", "max_size": "medium"},
        "webnn": {"constraint": "size", "max_size": "small"},
        "webgpu": {"constraint": "size", "max_size": "small"},
        "cpu": True
    },
    "multimodal": {
        "cuda": True,
        "rocm": {"constraint": "size", "max_size": "small"},
        "mps": {"constraint": "size", "max_size": "small"},
        "openvino": {"constraint": "size", "max_size": "small"},
        "webnn": {"constraint": "model", "models": ["clip"]},
        "webgpu": {"constraint": "model", "models": ["clip", "qwen"]},
        "cpu": True
    }
}

def get_model_size_tier(model_name: str, family: Optional[str] = None) -> str:
    """
    Determine model size tier (small, medium, large) based on model name and family

    Args:
        model_name: The name of the model
        family: Optional model family to provide additional context

    Returns:
        String indicating model size tier: "small", "medium", or "large"
    """
    # Default to medium if unknown
    size_tier = "medium"
    
    # Check for size indicators in model name
    model_name_lower = model_name.lower()
    
    # Size keywords in model name
    if any(kw in model_name_lower for kw in ["tiny", "mini", "small", "efficient"]):
        size_tier = "small"
    elif any(kw in model_name_lower for kw in ["base", "medium"]):
        size_tier = "medium"
    elif any(kw in model_name_lower for kw in ["large", "xl", "xxl", "huge"]):
        size_tier = "large"
        
    # Check for numeric size indicators
    # Common patterns like bert-large, t5-3b, llama-7b, etc.
    size_indicators = {
        "small": ["128m", "235m", "350m", "small", "tiny"],
        "medium": ["1b", "1.5b", "2b", "3b", "base"],
        "large": ["7b", "13b", "30b", "65b", "70b", "large"]
    }
    
    for size, indicators in size_indicators.items():
        if any(ind in model_name_lower for ind in indicators):
            size_tier = size
            break
    
    # Use family-specific size detection if available
    if family == "text_generation":
        # LLMs often include parameter count
        if "7b" in model_name_lower or "13b" in model_name_lower:
            size_tier = "large"
        elif "1.5b" in model_name_lower or "3b" in model_name_lower:
            size_tier = "medium"
    
    return size_tier

def check_hardware_compatibility(model_family: str, 
                                model_size: str, 
                                hardware_type: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a specific hardware type is compatible with the model family and size

    Args:
        model_family: The model family (embedding, text_generation, etc.)
        model_size: Size tier of the model (small, medium, large)
        hardware_type: Hardware type to check (cuda, mps, etc.)

    Returns:
        Tuple of (is_compatible, reason)
    """
    # Default to compatible if family not recognized
    if model_family not in MODEL_FAMILY_HARDWARE_COMPATIBILITY:
        return True, None
    
    # Get compatibility info for this family
    compatibility = MODEL_FAMILY_HARDWARE_COMPATIBILITY[model_family].get(hardware_type)
    
    # If direct boolean
    if isinstance(compatibility, bool):
        if compatibility:
            return True, None
        else:
            return False, f"{hardware_type} is not compatible with {model_family} models"
    
    # If dictionary with constraints
    elif isinstance(compatibility, dict):
        constraint_type = compatibility.get("constraint")
        
        if constraint_type == "size":
            max_size = compatibility.get("max_size", "large")
            size_order = {"small": 0, "medium": 1, "large": 2}
            
            if size_order.get(model_size, 1) <= size_order.get(max_size, 2):
                return True, None
            else:
                return False, f"{hardware_type} only supports {model_family} models up to {max_size} size"
    
    # Default to compatible
    return True, None

def estimate_memory_requirements(model_family: str, model_size: str) -> int:
    """
    Estimate memory requirements for a model based on family and size

    Args:
        model_family: The model family (embedding, text_generation, etc.)
        model_size: Size tier of the model (small, medium, large)

    Returns:
        Estimated memory requirement in MB
    """
    # Get family requirements if available
    family_reqs = MODEL_FAMILY_MEMORY_REQUIREMENTS.get(model_family, {})
    memory_mb = family_reqs.get(model_size, 500)  # Default to 500MB if unknown
    
    return memory_mb

def detect_hardware_availability(hardware_info: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """
    Detect available hardware platforms with robust error handling

    Args:
        hardware_info: Optional pre-detected hardware information

    Returns:
        Dictionary of available hardware platforms
    """
    # Start with all platforms unavailable
    available_hardware = {
        "cpu": True,  # CPU is always available
        "cuda": False,
        "rocm": False,
        "mps": False,
        "openvino": False,
        "webnn": False,
        "webgpu": False
    }
    
    # If hardware info is provided, use it
    if hardware_info:
        for hw_type in available_hardware.keys():
            if hw_type in hardware_info:
                available_hardware[hw_type] = hardware_info[hw_type]
        return available_hardware
    
    # Try to detect hardware using hardware_detection if available
    try:
        # Check if hardware_detection module exists
        module_path = os.path.join(os.path.dirname(__file__), "hardware_detection.py")
        if os.path.exists(module_path):
            # Try to import the module
            try:
                from hardware_detection import detect_hardware_with_comprehensive_checks
                detected_hw = detect_hardware_with_comprehensive_checks()
                
                # Update hardware availability
                for hw_type in available_hardware.keys():
                    if hw_type in detected_hw:
                        available_hardware[hw_type] = detected_hw[hw_type]
                
                logger.info(f"Hardware detection successful")
            except ImportError as e:
                logger.warning(f"Could not import hardware_detection module: {e}")
            except Exception as e:
                logger.warning(f"Error during hardware detection: {e}")
        else:
            logger.warning("hardware_detection.py not found, using basic detection")
    except Exception as e:
        logger.warning(f"Error checking for hardware_detection module: {e}")
    
    # If hardware detection failed or wasn't available, try basic detection
    if not any(available_hardware.values()):
        try:
            # Try to detect CUDA using PyTorch
            try:
                import torch
                available_hardware["cuda"] = torch.cuda.is_available()
                
                # Check for MPS (Apple Silicon)
                if hasattr(torch.backends, "mps"):
                    available_hardware["mps"] = torch.backends.mps.is_available()
            except ImportError:
                logger.debug("PyTorch not available for basic hardware detection")
            
            # Try to detect OpenVINO
            try:
                import openvino
                available_hardware["openvino"] = True
            except ImportError:
                pass
        except Exception as e:
            logger.warning(f"Error during basic hardware detection: {e}")
    
    return available_hardware

def classify_model_family(model_name: str, model_class: Optional[str] = None) -> Tuple[str, float]:
    """
    Classify a model into a family with robust error handling

    Args:
        model_name: The name of the model
        model_class: Optional model class name for better classification

    Returns:
        Tuple of (family, confidence)
    """
    # Default classification
    default_family = "embedding" if "bert" in model_name.lower() else "text_generation"
    default_confidence = 0.5
    
    # Try to use model_family_classifier if available
    try:
        # Check if module exists
        module_path = os.path.join(os.path.dirname(__file__), "model_family_classifier.py")
        if os.path.exists(module_path):
            # Try to import the module
            try:
                from model_family_classifier import classify_model
                
                # Classify the model
                result = classify_model(model_name=model_name, model_class=model_class)
                
                # Extract family and confidence
                family = result.get("family")
                confidence = result.get("confidence", 0.0)
                
                if family:
                    logger.info(f"Model {model_name} classified as {family} (confidence: {confidence:.2f})")
                    return family, confidence
                
                logger.warning(f"Model classification returned no family, using default")
            except ImportError as e:
                logger.warning(f"Could not import model_family_classifier module: {e}")
            except Exception as e:
                logger.warning(f"Error during model classification: {e}")
    except Exception as e:
        logger.warning(f"Error checking for model_family_classifier module: {e}")
    
    # Use simple heuristic classification as fallback
    model_name_lower = model_name.lower()
    
    # Simple keyword matching fallback
    for family, info in MODEL_FAMILY_DEVICE_PREFERENCES.items():
        # Check for family-specific keywords
        keywords = []
        if family == "embedding":
            keywords = ["bert", "roberta", "albert", "xlm", "distilbert", "sentence", "embedding"]
        elif family == "text_generation":
            keywords = ["gpt", "llama", "t5", "bart", "bloom", "opt", "falcon", "mistral", "phi"]
        elif family == "vision":
            keywords = ["vit", "swin", "resnet", "convnext", "deit", "clip", "image", "vision", "xclip", "detr"]
        elif family == "audio":
            keywords = ["whisper", "wav2vec", "wav2vec2", "hubert", "audio", "speech", "clap", "speecht5"]
        elif family == "multimodal":
            keywords = ["llava", "blip", "flava", "multimodal", "vision-text", "mm", "qwen2_vl", "qwen3_vl"]
        
        # Check for matches
        for keyword in keywords:
            if keyword in model_name_lower:
                logger.info(f"Model {model_name} classified as {family} using fallback heuristics")
                return family, 0.7  # Heuristic match with moderate confidence
    
    # If all else fails, return default
    logger.info(f"Using default classification for {model_name}: {default_family}")
    return default_family, default_confidence

def select_optimal_device(model_family: str, 
                         model_size: str,
                         available_hardware: Dict[str, bool],
                         memory_requirements: Optional[int] = None) -> Dict[str, Any]:
    """
    Select the optimal device for a model based on family, size, and available hardware

    Args:
        model_family: The model family (embedding, text_generation, etc.)
        model_size: Size tier of the model (small, medium, large)
        available_hardware: Dictionary of available hardware platforms
        memory_requirements: Optional known memory requirements in MB

    Returns:
        Dictionary with device selection information
    """
    # Get device preferences for this model family
    device_preferences = MODEL_FAMILY_DEVICE_PREFERENCES.get(model_family, ["cuda", "cpu"])
    
    # If memory requirements not provided, estimate them
    if memory_requirements is None:
        memory_requirements = estimate_memory_requirements(model_family, model_size)
    
    # Initialize result
    result = {
        "device": "cpu",  # Default to CPU
        "reason": "Default device selection",
        "memory_required_mb": memory_requirements,
        "compatible_devices": [],
        "preferred_devices": device_preferences
    }
    
    # Check compatibility for each device type
    compatible_devices = []
    compatibility_reasons = {}
    
    for device_type in device_preferences:
        # Skip if not available
        if not available_hardware.get(device_type, False):
            compatibility_reasons[device_type] = f"{device_type} not available on this system"
            continue
        
        # Check compatibility
        is_compatible, reason = check_hardware_compatibility(model_family, model_size, device_type)
        
        if is_compatible:
            compatible_devices.append(device_type)
        else:
            compatibility_reasons[device_type] = reason
    
    # Update result with compatible devices
    result["compatible_devices"] = compatible_devices
    result["compatibility_reasons"] = compatibility_reasons
    
    # Select best available device from preferences
    for device_type in device_preferences:
        if device_type in compatible_devices:
            result["device"] = device_type
            result["reason"] = f"Selected {device_type} based on model family preferences"
            break
    
    # Special handling for PyTorch device strings
    if result["device"] == "cuda":
        result["torch_device"] = "cuda"
    elif result["device"] == "mps":
        result["torch_device"] = "mps"
    elif result["device"] == "cpu":
        result["torch_device"] = "cpu"
    else:
        # Default to CPU for other device types
        result["torch_device"] = "cpu"
    
    return result

def integrate_hardware_and_model(model_name: str, 
                                model_family: Optional[str] = None,
                                model_class: Optional[str] = None,
                                hardware_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Integrate hardware detection and model classification to determine optimal device

    Args:
        model_name: The name of the model
        model_family: Optional pre-detected model family
        model_class: Optional model class name for better classification
        hardware_info: Optional pre-detected hardware information

    Returns:
        Dictionary with integrated hardware-model information
    """
    logger.info(f"Integrating hardware and model information for {model_name}")
    
    # Initialize result
    result = {
        "model_name": model_name,
        "original_family": model_family,
        "hardware_info_provided": hardware_info is not None
    }
    
    # Step 1: Detect available hardware
    available_hardware = detect_hardware_availability(hardware_info)
    result["available_hardware"] = available_hardware
    
    # Get list of available hardware types
    available_hw_types = [hw for hw, available in available_hardware.items() if available]
    logger.info(f"Available hardware: {', '.join(available_hw_types)}")
    
    # Step 2: Classify model family if not provided
    if model_family is None:
        model_family, confidence = classify_model_family(model_name, model_class)
        result["detected_family"] = model_family
        result["family_confidence"] = confidence
    
    # Use provided family or detected family
    effective_family = model_family or result.get("detected_family", "text_generation")
    result["effective_family"] = effective_family
    
    # Step 3: Determine model size
    model_size = get_model_size_tier(model_name, effective_family)
    result["model_size"] = model_size
    
    # Step 4: Estimate memory requirements
    memory_mb = estimate_memory_requirements(effective_family, model_size)
    result["estimated_memory_mb"] = memory_mb
    
    # Step 5: Select optimal device
    device_selection = select_optimal_device(
        effective_family,
        model_size,
        available_hardware,
        memory_mb
    )
    
    # Add device selection to result
    result.update(device_selection)
    
    # Step 6: Add hardware preferences for ResourcePool
    result["hardware_preferences"] = {
        "device": device_selection["torch_device"],
        "priority_list": device_selection["preferred_devices"],
        "hw_compatibility": {
            hw_type: {"compatible": hw_type in device_selection["compatible_devices"]}
            for hw_type in ["cuda", "mps", "rocm", "openvino", "cpu"]
        }
    }
    
    logger.info(f"Selected device {result['device']} for {model_name} ({effective_family}, {model_size})")
    
    return result

def get_hardware_model_compatibility_matrix() -> Dict[str, Any]:
    """
    Generate a comprehensive hardware-model compatibility matrix

    Returns:
        Dictionary with compatibility information
    """
    # Create compatibility matrix
    matrix = {
        "model_families": {},
        "hardware_types": ["cuda", "rocm", "mps", "openvino", "webnn", "webgpu", "cpu"],
        "model_sizes": ["small", "medium", "large"]
    }
    
    # Add compatibility information for each family
    for family, compatibility in MODEL_FAMILY_HARDWARE_COMPATIBILITY.items():
        matrix["model_families"][family] = {
            "hardware_compatibility": {},
            "memory_requirements": MODEL_FAMILY_MEMORY_REQUIREMENTS.get(family, {}),
            "device_preferences": MODEL_FAMILY_DEVICE_PREFERENCES.get(family, [])
        }
        
        # Add compatibility for each hardware type
        for hw_type, compat_info in compatibility.items():
            if isinstance(compat_info, bool):
                matrix["model_families"][family]["hardware_compatibility"][hw_type] = {
                    "compatible": compat_info,
                    "constraints": None
                }
            elif isinstance(compat_info, dict):
                matrix["model_families"][family]["hardware_compatibility"][hw_type] = {
                    "compatible": True,
                    "constraints": compat_info
                }
    
    return matrix

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test hardware-model integration")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name to test")
    parser.add_argument("--matrix", action="store_true", help="Print compatibility matrix")
    parser.add_argument("--detect", action="store_true", help="Detect available hardware")
    parser.add_argument("--family", type=str, help="Override model family")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Handle compatibility matrix request
    if args.matrix:
        matrix = get_hardware_model_compatibility_matrix()
        print(json.dumps(matrix, indent=2))
        sys.exit(0)
    
    # Handle hardware detection
    if args.detect:
        available_hw = detect_hardware_availability()
        print("Detected hardware:")
        for hw_type, available in available_hw.items():
            status = "✅ Available" if available else "❌ Not available"
            print(f"  - {hw_type}: {status}")
        sys.exit(0)
    
    # Test integration with the specified model
    result = integrate_hardware_and_model(
        model_name=args.model,
        model_family=args.family
    )
    
    # Print result
    print(f"\nHardware-Model Integration Results for {args.model}:")
    print(f"  Model Family: {result['effective_family']}")
    print(f"  Model Size: {result['model_size']}")
    print(f"  Estimated Memory: {result['estimated_memory_mb']} MB")
    print(f"  Selected Device: {result['device']}")
    print(f"  Compatible Devices: {', '.join(result['compatible_devices'])}")
    print(f"  Reason: {result['reason']}")
    
    # Show hardware preferences for ResourcePool
    print("\nResourcePool Hardware Preferences:")
    print(json.dumps(result["hardware_preferences"], indent=2))