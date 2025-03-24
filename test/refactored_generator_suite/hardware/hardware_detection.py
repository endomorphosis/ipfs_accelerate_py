#!/usr/bin/env python3
"""
Hardware detection module for the refactored generator suite.

This module provides functions to detect available hardware backends
and select the optimal device for a given model type.
"""

import os
import sys
import logging
import importlib
from typing import Dict, List, Optional, Tuple, Any, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of supported hardware backends
SUPPORTED_BACKENDS = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn"]

# Check if environment variables are set for mock mode
MOCK_MODE = os.environ.get("MOCK_MODE", "False").lower() == "true"

# Define compatibility matrix for architecture types and hardware backends
ARCHITECTURE_HARDWARE_COMPATIBILITY = {
    "encoder-only": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": True,
        "qnn": True
    },
    "decoder-only": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": True,
        "qnn": False  # QNN has limited support for decoder-only models
    },
    "encoder-decoder": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": True,
        "qnn": False
    },
    "vision": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": True,
        "qnn": True
    },
    "vision-encoder-text-decoder": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": True,
        "qnn": True
    },
    "speech": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": True,
        "qnn": True
    },
    "multimodal": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": True,
        "qnn": False
    },
    "diffusion": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": True,
        "qnn": False
    },
    "mixture-of-experts": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": False,  # MoE models often exceed MPS memory limits
        "openvino": False,
        "qnn": False
    },
    "state-space": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": False,  # State-space models often lack OpENVINO optimization
        "qnn": False
    },
    "rag": {
        "cpu": True,
        "cuda": True,
        "rocm": True,
        "mps": True,
        "openvino": False,
        "qnn": False
    }
}


def detect_available_hardware() -> Dict[str, bool]:
    """
    Detect which hardware backends are available on the current system.
    
    Returns:
        A dictionary mapping backend names to boolean availability.
    """
    available_hardware = {backend: False for backend in SUPPORTED_BACKENDS}
    
    # CPU is always available
    available_hardware["cpu"] = True
    
    if MOCK_MODE:
        # In mock mode, pretend all hardware is available
        for backend in SUPPORTED_BACKENDS:
            available_hardware[backend] = True
        return available_hardware
    
    # Check for PyTorch
    try:
        import torch
        
        # Check CUDA
        if torch.cuda.is_available():
            available_hardware["cuda"] = True
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        
        # Check ROCm/HIP
        if hasattr(torch, 'hip') and torch.hip.is_available():
            available_hardware["rocm"] = True
            logger.info("ROCm/HIP is available")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_hardware["mps"] = True
            logger.info("MPS (Apple Silicon) is available")
    
    except ImportError:
        logger.warning("PyTorch not available, CUDA/ROCm/MPS detection skipped")
    
    # Check for OpenVINO
    try:
        import openvino
        from openvino.runtime import Core
        available_hardware["openvino"] = True
        
        # Log available OpenVINO devices
        try:
            core = Core()
            available_devices = core.available_devices
            logger.info(f"OpenVINO is available with devices: {available_devices}")
        except Exception as e:
            logger.warning(f"OpenVINO available but device enumeration failed: {e}")
    
    except ImportError:
        logger.warning("OpenVINO not available")
    
    # Check for QNN (Qualcomm Neural Network)
    try:
        from qnnpy import PyQnnManager
        available_hardware["qnn"] = True
        logger.info("QNN is available")
    except ImportError:
        logger.warning("QNN not available")
    
    return available_hardware


def get_optimal_device(priority_list: Optional[List[str]] = None) -> str:
    """
    Get the optimal available device based on priority.
    
    Args:
        priority_list: Optional list of devices in order of priority.
                      If None, defaults to ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"].
    
    Returns:
        The name of the optimal available device.
    """
    if priority_list is None:
        priority_list = ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"]
    
    available = detect_available_hardware()
    
    # Return the first available device in the priority list
    for device in priority_list:
        if device in available and available[device]:
            return device
    
    # Fallback to CPU if nothing else is available
    return "cpu"


def get_device_settings(device: str) -> Dict[str, Any]:
    """
    Get device-specific settings for the given device.
    
    Args:
        device: The device to get settings for.
    
    Returns:
        A dictionary of device-specific settings.
    """
    settings = {
        "device_name": device,
        "precision": "float32",
        "device_map": None,
        "half_precision_supported": False,
        "dynamic_shapes_supported": False,
        "quantization_supported": False
    }
    
    if device == "cpu":
        settings["precision"] = "float32"
        settings["half_precision_supported"] = False
        settings["dynamic_shapes_supported"] = True
        settings["quantization_supported"] = True
    
    elif device == "cuda":
        settings["precision"] = "float16"
        settings["device_map"] = "cuda"
        settings["half_precision_supported"] = True
        settings["dynamic_shapes_supported"] = True
        settings["quantization_supported"] = True
    
    elif device == "rocm":
        settings["precision"] = "float16"
        settings["device_map"] = "cuda"  # ROCm uses CUDA device map
        settings["half_precision_supported"] = True
        settings["dynamic_shapes_supported"] = True
        settings["quantization_supported"] = True
    
    elif device == "mps":
        settings["precision"] = "float16"
        settings["device_map"] = "mps"
        settings["half_precision_supported"] = True
        settings["dynamic_shapes_supported"] = True
        settings["quantization_supported"] = False
    
    elif device == "openvino":
        settings["precision"] = "float32"
        settings["half_precision_supported"] = True
        settings["dynamic_shapes_supported"] = True
        settings["quantization_supported"] = True
        
        # Try to determine default OpenVINO device
        try:
            from openvino.runtime import Core
            core = Core()
            available_devices = core.available_devices
            
            # Prioritize GPUs if available
            if "GPU" in available_devices:
                settings["openvino_device"] = "GPU"
            else:
                settings["openvino_device"] = "CPU"
        except:
            # Default to CPU if detection fails
            settings["openvino_device"] = "CPU"
    
    elif device == "qnn":
        settings["precision"] = "float32"
        settings["half_precision_supported"] = True
        settings["dynamic_shapes_supported"] = False
        settings["quantization_supported"] = True
    
    return settings


def is_device_compatible_with_model(device: str, arch_type: str) -> bool:
    """
    Check if a device is compatible with a model architecture type.
    
    Args:
        device: The device to check.
        arch_type: The model architecture type.
    
    Returns:
        True if the device is compatible, False otherwise.
    """
    if device not in SUPPORTED_BACKENDS:
        logger.warning(f"Unknown device: {device}")
        return False
    
    if arch_type not in ARCHITECTURE_HARDWARE_COMPATIBILITY:
        logger.warning(f"Unknown architecture type: {arch_type}")
        return True  # Default to compatible
    
    return ARCHITECTURE_HARDWARE_COMPATIBILITY[arch_type].get(device, False)


def get_model_hardware_recommendations(arch_type: str) -> List[str]:
    """
    Get a list of recommended hardware backends for a model architecture type.
    
    Args:
        arch_type: The model architecture type.
    
    Returns:
        A list of recommended hardware backends, in priority order.
    """
    if arch_type not in ARCHITECTURE_HARDWARE_COMPATIBILITY:
        logger.warning(f"Unknown architecture type: {arch_type}")
        return ["cpu", "cuda", "rocm", "mps"]  # Default recommendations
    
    # Get all compatible hardware
    compatible_hw = []
    for device, compatible in ARCHITECTURE_HARDWARE_COMPATIBILITY[arch_type].items():
        if compatible:
            compatible_hw.append(device)
    
    # Order by general priority
    priority_order = ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"]
    return sorted(compatible_hw, key=lambda x: priority_order.index(x) if x in priority_order else 999)


def get_device_code_snippet(device: str, model_class: str, arch_type: str) -> str:
    """
    Get a code snippet for initializing a model on the specified device.
    
    Args:
        device: The device to generate code for.
        model_class: The model class name.
        arch_type: The model architecture type.
    
    Returns:
        A string containing the code snippet.
    """
    if device == "cpu":
        return f"""
# CPU initialization
self.model = {model_class}.from_pretrained(self.model_id)
"""
    
    elif device == "cuda":
        return f"""
# CUDA initialization with half precision
self.model = {model_class}.from_pretrained(
    self.model_id,
    device_map="cuda",
    torch_dtype=torch.float16
)
"""
    
    elif device == "rocm":
        return f"""
# ROCm initialization with half precision
self.model = {model_class}.from_pretrained(
    self.model_id,
    device_map="cuda",  # ROCm uses CUDA device map
    torch_dtype=torch.float16
)
"""
    
    elif device == "mps":
        return f"""
# MPS (Apple Silicon) initialization
self.model = {model_class}.from_pretrained(self.model_id)
self.model.to("mps")
"""
    
    elif device == "openvino":
        # Get the short model class name for OpenVINO
        model_class_short = model_class
        if model_class.startswith("Auto"):
            model_class_short = model_class[4:]
        
        return f"""
# OpenVINO initialization
from optimum.intel import OVModelFor{model_class_short}
# Determine OpenVINO device (CPU, GPU, VPU, etc.)
ov_device = "CPU"  # Default to CPU
if os.environ.get("OV_DEVICE") in ["GPU", "VPU"]:
    ov_device = os.environ.get("OV_DEVICE")

self.model = OVModelFor{model_class_short}.from_pretrained(
    self.model_id,
    export=True,
    provider=ov_device
)
"""
    
    elif device == "qnn":
        if not is_device_compatible_with_model(device, arch_type):
            return f"""
# QNN implementation is not fully supported for this model type
# Falling back to CPU
self.model = {model_class}.from_pretrained(self.model_id)
"""
        
        return f"""
# QNN implementation
from qnnpy import PyQnnManager

# This is a simplified QNN implementation
# In a real implementation, you would need to convert the model to QNN format
self.model = {model_class}.from_pretrained(self.model_id)
# Convert to QNN (placeholder)
# self.model = convert_to_qnn(self.model)
"""
    
    else:
        return f"""
# Unknown device: {device}, falling back to CPU
self.model = {model_class}.from_pretrained(self.model_id)
"""


# For testing
if __name__ == "__main__":
    # Print available hardware
    available_hw = detect_available_hardware()
    print("Available Hardware:")
    for hw, available in available_hw.items():
        print(f"  {hw}: {'✅' if available else '❌'}")
    
    # Get optimal device
    optimal = get_optimal_device()
    print(f"\nOptimal device: {optimal}")
    
    # Test compatibility
    print("\nHardware compatibility for architecture types:")
    for arch in ARCHITECTURE_HARDWARE_COMPATIBILITY.keys():
        compatible_devices = get_model_hardware_recommendations(arch)
        print(f"  {arch}: {', '.join(compatible_devices)}")
    
    # Test device settings
    for device in SUPPORTED_BACKENDS:
        settings = get_device_settings(device)
        print(f"\nSettings for {device}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    # Example code snippets
    print("\nExample code snippets:")
    print(get_device_code_snippet("cuda", "AutoModelForCausalLM", "decoder-only"))