#!/usr/bin/env python3
"""
Hardware detection module for the IPFS Accelerate test suite.

This module provides functions to detect available hardware backends and
select the optimal device based on availability and performance.
"""

import os
import sys
import logging
import importlib
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define supported hardware backends
SUPPORTED_BACKENDS = [
    "cpu",      # CPU (always available)
    "cuda",     # NVIDIA GPUs via PyTorch
    "rocm",     # AMD GPUs via PyTorch+ROCm
    "mps",      # Apple Metal via PyTorch MPS
    "openvino", # Intel CPUs, GPUs, VPUs via OpenVINO
    "qnn"       # Qualcomm Neural Network via QNN SDK
]

def detect_available_hardware() -> Dict[str, bool]:
    """
    Detect available hardware backends.
    
    Returns:
        Dict mapping backend names to availability (True/False)
    """
    available = {"cpu": True}  # CPU is always available
    
    # Check for CUDA (NVIDIA)
    try:
        import torch
        available["cuda"] = torch.cuda.is_available()
        if available["cuda"]:
            logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    except ImportError:
        available["cuda"] = False
        logger.info("CUDA not available: PyTorch not installed")
    except Exception as e:
        available["cuda"] = False
        logger.info(f"CUDA not available: {e}")
    
    # Check for ROCm (AMD)
    try:
        import torch
        # Check if PyTorch was built with ROCm
        available["rocm"] = hasattr(torch, 'hip') and torch.hip.is_available()
        if available["rocm"]:
            logger.info("ROCm detected")
    except (ImportError, AttributeError):
        available["rocm"] = False
        logger.info("ROCm not available")
    
    # Check for MPS (Apple Metal)
    try:
        import torch
        available["mps"] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if available["mps"]:
            logger.info("MPS (Apple Metal) detected")
    except (ImportError, AttributeError):
        available["mps"] = False
        logger.info("MPS (Apple Metal) not available")
    
    # Check for OpenVINO
    try:
        spec = importlib.util.find_spec("openvino")
        available["openvino"] = spec is not None
        if available["openvino"]:
            logger.info("OpenVINO detected")
    except (ImportError, AttributeError):
        available["openvino"] = False
        logger.info("OpenVINO not available")
    
    # Check for QNN (Qualcomm)
    try:
        spec = importlib.util.find_spec("qnn")
        available["qnn"] = spec is not None
        if available["qnn"]:
            logger.info("QNN (Qualcomm Neural Network) detected")
    except (ImportError, AttributeError):
        available["qnn"] = False
        logger.info("QNN (Qualcomm Neural Network) not available")
    
    return available

def get_optimal_device(priority_list: Optional[List[str]] = None) -> str:
    """
    Get the optimal available device.
    
    Args:
        priority_list: Optional list of device priorities (default: None, use built-in prioritization)
    
    Returns:
        Name of the optimal device
    """
    available = detect_available_hardware()
    
    # Default priority: CUDA > ROCm > MPS > OpenVINO > QNN > CPU
    default_priority = ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"]
    
    # Use provided priority list or default
    priorities = priority_list or default_priority
    
    # Find the highest priority available device
    for device in priorities:
        if available.get(device, False):
            return device
    
    # Fallback to CPU if no device found (should never happen as CPU is always available)
    return "cpu"

def get_device_settings(device: str) -> Dict[str, Any]:
    """
    Get device-specific settings for a given device.
    
    Args:
        device: The device name
    
    Returns:
        Dictionary of device-specific settings
    """
    settings = {
        "name": device,
        "description": f"{device.upper()} backend",
        "performance_tier": "low",
        "pytorch_compatible": device in ["cpu", "cuda", "rocm", "mps"],
        "specific_settings": {}
    }
    
    if device == "cpu":
        # CPU settings
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        settings.update({
            "description": f"CPU ({cpu_count} cores)",
            "performance_tier": "low",
            "specific_settings": {
                "num_threads": cpu_count,
                "inter_op_num_threads": max(1, cpu_count // 2),
                "use_mkldnn": True
            }
        })
    
    elif device == "cuda":
        # CUDA settings
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            settings.update({
                "description": f"CUDA ({gpu_name})",
                "performance_tier": "high",
                "specific_settings": {
                    "gpu_count": gpu_count,
                    "gpu_name": gpu_name,
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "cudnn_enabled": torch.backends.cudnn.enabled,
                    "use_cudnn_benchmark": True,
                    "use_tensor_cores": True
                }
            })
        except Exception as e:
            logger.error(f"Error getting CUDA settings: {e}")
    
    elif device == "rocm":
        # ROCm settings
        try:
            import torch
            settings.update({
                "description": "ROCm (AMD GPU)",
                "performance_tier": "high",
                "specific_settings": {
                    "hip_version": torch.version.hip if hasattr(torch.version, 'hip') else "Unknown"
                }
            })
        except Exception as e:
            logger.error(f"Error getting ROCm settings: {e}")
    
    elif device == "mps":
        # MPS settings
        try:
            import torch
            settings.update({
                "description": "MPS (Apple Metal)",
                "performance_tier": "medium",
                "specific_settings": {
                    "mps_backend_available": torch.backends.mps.is_available(),
                    "mps_built": torch.backends.mps.is_built()
                }
            })
        except Exception as e:
            logger.error(f"Error getting MPS settings: {e}")
    
    elif device == "openvino":
        # OpenVINO settings
        try:
            import openvino as ov
            settings.update({
                "description": "OpenVINO (Intel)",
                "performance_tier": "medium",
                "pytorch_compatible": False,
                "specific_settings": {
                    "version": ov.__version__,
                    "available_devices": ov.Core().available_devices
                }
            })
        except Exception as e:
            logger.error(f"Error getting OpenVINO settings: {e}")
    
    elif device == "qnn":
        # QNN settings
        try:
            import qnn
            settings.update({
                "description": "QNN (Qualcomm Neural Network)",
                "performance_tier": "medium",
                "pytorch_compatible": False,
                "specific_settings": {
                    "version": getattr(qnn, "__version__", "Unknown")
                }
            })
        except Exception as e:
            logger.error(f"Error getting QNN settings: {e}")
    
    return settings

def is_device_compatible_with_model(device: str, model_type: str) -> bool:
    """
    Check if a device is compatible with a given model type.
    
    Args:
        device: The device name
        model_type: The model type to check
    
    Returns:
        True if the device is compatible with the model type, False otherwise
    """
    # Default: all devices are compatible with all models
    compatibility = True
    
    # Device-specific compatibility checks
    if device == "qnn":
        # QNN has limited model support - mostly optimized for vision and audio models
        if model_type in ["speech", "vision", "multimodal"]:
            compatibility = True
        else:
            # Some text models might not work optimally or might require special handling
            compatibility = False
    
    elif device == "openvino":
        # OpenVINO supports most models, but some sophisticated architectures might need special handling
        if model_type in ["diffusion", "state-space", "mixture-of-experts"]:
            # These might require additional model optimization
            compatibility = False
    
    return compatibility

def get_model_hardware_recommendations(model_type: str) -> List[str]:
    """
    Get recommended hardware devices for a given model type.
    
    Args:
        model_type: The model type
    
    Returns:
        List of recommended devices in order of preference
    """
    # Define model-specific recommendations
    recommendations = {
        "encoder-only": ["cuda", "rocm", "mps", "openvino", "cpu"],
        "decoder-only": ["cuda", "rocm", "mps", "cpu"],
        "encoder-decoder": ["cuda", "rocm", "mps", "cpu"],
        "vision": ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"],
        "vision-encoder-text-decoder": ["cuda", "rocm", "mps", "cpu"],
        "speech": ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"],
        "multimodal": ["cuda", "rocm", "mps", "cpu"],
        "diffusion": ["cuda", "rocm", "mps", "cpu"],
        "mixture-of-experts": ["cuda", "rocm", "cpu"],
        "state-space": ["cuda", "rocm", "cpu"],
        "rag": ["cuda", "rocm", "mps", "cpu"]
    }
    
    # Return recommendations for the given model type or a default if not found
    return recommendations.get(model_type, ["cuda", "rocm", "mps", "openvino", "qnn", "cpu"])

def print_device_summary():
    """Print a summary of available devices."""
    available = detect_available_hardware()
    optimal = get_optimal_device()
    
    print("\n=== Hardware Backend Summary ===")
    print(f"Optimal device: {optimal}")
    print("\nAvailable backends:")
    
    for device in SUPPORTED_BACKENDS:
        status = "✅ Available" if available.get(device, False) else "❌ Not available"
        optimal_marker = " (SELECTED)" if device == optimal else ""
        print(f"- {device.upper():<8}: {status}{optimal_marker}")
    
    if optimal != "cpu":
        settings = get_device_settings(optimal)
        print(f"\nSelected device: {settings['description']}")
        if settings['specific_settings']:
            print("Device details:")
            for key, value in settings['specific_settings'].items():
                print(f"  - {key}: {value}")
    
    print("===============================\n")

def initialize_device(device: str) -> Dict[str, Any]:
    """
    Initialize a device for use.
    
    Args:
        device: The device name
    
    Returns:
        Dictionary with initialization status and device info
    """
    result = {
        "device": device,
        "success": False,
        "message": "",
        "settings": {}
    }
    
    try:
        # Get device settings
        settings = get_device_settings(device)
        result["settings"] = settings
        
        # Initialize based on device type
        if device == "cpu":
            # CPU initialization
            import torch
            specific = settings["specific_settings"]
            torch.set_num_threads(specific["num_threads"])
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(specific["inter_op_num_threads"])
            result["success"] = True
            result["message"] = "CPU initialized successfully"
        
        elif device == "cuda":
            # CUDA initialization
            import torch
            specific = settings["specific_settings"]
            torch.backends.cudnn.benchmark = specific["use_cudnn_benchmark"]
            torch.backends.cudnn.enabled = specific["cudnn_enabled"]
            result["success"] = torch.cuda.is_available()
            result["message"] = "CUDA initialized successfully" if result["success"] else "CUDA not available"
        
        elif device == "rocm":
            # ROCm initialization
            import torch
            result["success"] = hasattr(torch, 'hip') and torch.hip.is_available()
            result["message"] = "ROCm initialized successfully" if result["success"] else "ROCm not available"
        
        elif device == "mps":
            # MPS initialization
            import torch
            result["success"] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            result["message"] = "MPS initialized successfully" if result["success"] else "MPS not available"
        
        elif device == "openvino":
            # OpenVINO initialization
            import openvino as ov
            core = ov.Core()
            result["success"] = True
            result["settings"]["specific_settings"]["available_devices"] = core.available_devices
            result["message"] = "OpenVINO initialized successfully"
        
        elif device == "qnn":
            # QNN initialization
            import qnn
            result["success"] = True
            result["message"] = "QNN initialized successfully"
        
        else:
            result["success"] = False
            result["message"] = f"Unknown device: {device}"
        
    except Exception as e:
        result["success"] = False
        result["message"] = f"Error initializing {device}: {e}"
        logger.error(result["message"])
    
    return result

if __name__ == "__main__":
    # When run as a script, print device summary
    print_device_summary()