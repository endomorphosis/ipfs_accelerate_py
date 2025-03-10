#\!/usr/bin/env python3
"""
Hardware detection module for IPFS Accelerate.
This module provides comprehensive hardware detection functionality
for various platforms and hardware types.
"""

import os
import sys
import platform
import logging
import re
import json
from datetime import datetime
import importlib.util
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_FILE_ENV = "HARDWARE_DETECTION_CACHE"
DEFAULT_CACHE_PATH = ".hardware_detection_cache.json"

def detect_available_hardware(use_cache: bool = True, 
                             cache_file: Optional[str] = None,
                             force_refresh: bool = False) -> Dict[str, Any]:
    """
    Detects available hardware on the system
    
    Args:
        use_cache: Whether to use cached hardware information
        cache_file: Path to cache file (defaults to environment variable or default path)
        force_refresh: Whether to force hardware re-detection
        
    Returns:
        Dictionary with hardware information
    """
    # Determine cache file path
    if not cache_file:
        cache_file = os.environ.get(CACHE_FILE_ENV, DEFAULT_CACHE_PATH)
    
    # Check if cache exists and is valid
    if use_cache and not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check cache freshness (7 days)
            cache_timestamp = cached_data.get('timestamp', 0)
            cache_time = datetime.fromtimestamp(cache_timestamp)
            current_time = datetime.now()
            
            # Cache is still fresh
            if (current_time - cache_time).days < 7:
                logger.debug(f"Using hardware cache from {cache_time.isoformat()}")
                return cached_data
            else:
                logger.debug(f"Cache expired (from {cache_time.isoformat()})")
        except Exception as e:
            logger.warning(f"Error reading hardware cache: {e}")
    
    # Detect hardware
    hardware_info = detect_hardware_with_comprehensive_checks()
    
    # Add timestamp
    hardware_info["timestamp"] = datetime.now().timestamp()
    
    # Save cache if needed
    if use_cache:
        try:
            with open(cache_file, 'w') as f:
                json.dump(hardware_info, f, indent=2)
            logger.debug(f"Saved hardware cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Error saving hardware cache: {e}")
    
    return hardware_info

def detect_hardware_with_comprehensive_checks() -> Dict[str, Any]:
    """
    Performs comprehensive hardware detection with detailed checks
    
    Returns:
        Dictionary with detailed hardware information
    """
    hardware_info = {
        "cpu": True,  # CPU is always available
        "cuda": False,
        "rocm": False,
        "mps": False,
        "openvino": False,
        "qnn": False,
        "webnn": False,
        "webgpu": False,
        "system": detect_system_info(),
        "best_available": "cpu",
        "torch_device": "cpu"
    }
    
    # Check for PyTorch-based hardware
    torch_info = detect_torch_hardware()
    hardware_info.update(torch_info)
    
    # Check for other hardware frameworks
    check_openvino(hardware_info)
    check_qnn(hardware_info)
    check_webnn_webgpu(hardware_info)
    
    # Determine best available device
    if hardware_info.get("cuda", False):
        hardware_info["best_available"] = "cuda"
        hardware_info["torch_device"] = "cuda"
    elif hardware_info.get("mps", False):
        hardware_info["best_available"] = "mps"
        hardware_info["torch_device"] = "mps"
    elif hardware_info.get("rocm", False):
        hardware_info["best_available"] = "rocm"
        hardware_info["torch_device"] = "cuda"  # PyTorch uses cuda device for ROCm
    elif hardware_info.get("openvino", False):
        hardware_info["best_available"] = "openvino"
        hardware_info["torch_device"] = "cpu"  # OpenVINO typically uses CPU backend
    
    return hardware_info

def detect_system_info() -> Dict[str, Any]:
    """
    Detects basic system information
    
    Returns:
        Dictionary with system information
    """
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "architecture": platform.machine(),
        "available_memory": detect_available_memory()
    }
    
    return system_info

def detect_available_memory() -> float:
    """
    Detects available system memory in MB
    
    Returns:
        Available memory in MB
    """
    # Try using psutil if available
    try:
        import psutil
        vm = psutil.virtual_memory()
        available_mb = vm.available / (1024 * 1024)
        return available_mb
    except ImportError:
        # If psutil is not available, try platform-specific approaches
        if platform.system() == "Linux":
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                # Extract available memory
                match = re.search(r'MemAvailable:\s+(\d+)', meminfo)
                if match:
                    return int(match.group(1)) / 1024  # Convert from KB to MB
            except:
                pass
    
    # Default if we can't detect
    return 8192  # Assume 8GB as default

def detect_torch_hardware() -> Dict[str, Any]:
    """
    Detects PyTorch-compatible hardware
    
    Returns:
        Dictionary with PyTorch hardware information
    """
    torch_info = {
        "cuda": False,
        "cuda_info": {},
        "rocm": False,
        "mps": False,
    }
    
    try:
        import torch
        
        # CUDA detection
        if torch.cuda.is_available():
            torch_info["cuda"] = True
            
            # Gather detailed CUDA information
            cuda_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "devices": []
            }
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "name": props.name,
                    "total_memory": props.total_memory / (1024 * 1024),  # MB
                    "compute_capability": f"{props.major}.{props.minor}"
                }
                cuda_info["devices"].append(device_info)
            
            # Add CUDA version if available
            if hasattr(torch.version, 'cuda'):
                cuda_info["cuda_version"] = torch.version.cuda
            
            torch_info["cuda_info"] = cuda_info
        
        # ROCm (AMD) detection
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            torch_info["rocm"] = True
            torch_info["rocm_info"] = {"hip_version": torch.version.hip}
        
        # MPS (Apple Silicon) detection
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch_info["mps"] = True
            torch_info["mps_info"] = {"is_built": torch.backends.mps.is_built()}
    
    except ImportError:
        logger.debug("PyTorch not available for hardware detection")
    except Exception as e:
        logger.warning(f"Error in PyTorch hardware detection: {e}")
    
    return torch_info

def check_openvino(hardware_info: Dict[str, Any]) -> None:
    """
    Checks for OpenVINO availability
    
    Args:
        hardware_info: Hardware information dictionary to update
    """
    try:
        if importlib.util.find_spec("openvino") is not None:
            hardware_info["openvino"] = True
            
            # Try to get OpenVINO version
            try:
                import openvino
                hardware_info["openvino_info"] = {
                    "version": getattr(openvino, "__version__", "unknown")
                }
                
                # Try to detect available OpenVINO devices
                if hasattr(openvino, "runtime") and hasattr(openvino.runtime, "Core"):
                    core = openvino.runtime.Core()
                    available_devices = core.available_devices
                    hardware_info["openvino_info"]["available_devices"] = available_devices
                    
            except Exception as e:
                logger.debug(f"Error getting OpenVINO details: {e}")
                hardware_info["openvino_info"] = {"error": str(e)}
    except:
        pass

def check_qnn(hardware_info: Dict[str, Any]) -> None:
    """
    Checks for Qualcomm AI Engine (QNN) availability
    
    Args:
        hardware_info: Hardware information dictionary to update
    """
    try:
        # Check for QNN wrapper
        if importlib.util.find_spec("qnn_wrapper") is not None:
            hardware_info["qnn"] = True
            
            # Try to get QNN details
            try:
                import qnn_wrapper
                hardware_info["qnn_info"] = {
                    "version": getattr(qnn_wrapper, "__version__", "unknown")
                }
            except Exception as e:
                logger.debug(f"Error getting QNN details: {e}")
                hardware_info["qnn_info"] = {"error": str(e)}
        
        # Also check for direct Qualcomm AI engine
        elif importlib.util.find_spec("qti") is not None:
            hardware_info["qnn"] = True
            
            # Try to get QTI details
            try:
                import qti
                hardware_info["qnn_info"] = {
                    "type": "qti",
                    "version": getattr(qti, "__version__", "unknown")
                }
            except Exception as e:
                logger.debug(f"Error getting QTI details: {e}")
                hardware_info["qnn_info"] = {"type": "qti", "error": str(e)}
    except:
        pass

def check_webnn_webgpu(hardware_info: Dict[str, Any]) -> None:
    """
    Checks for WebNN and WebGPU availability
    
    Args:
        hardware_info: Hardware information dictionary to update
    """
    # WebNN and WebGPU are typically only available in browser environments
    # For Python testing, we check for simulation capabilities
    
    # Check for WebNN simulation
    try:
        if os.environ.get("WEBNN_SIMULATION_ENABLED", "").lower() in ("1", "true", "yes"):
            hardware_info["webnn"] = True
            hardware_info["webnn_info"] = {"simulation": True}
    except:
        pass
    
    # Check for WebGPU simulation
    try:
        if os.environ.get("WEBGPU_SIMULATION_ENABLED", "").lower() in ("1", "true", "yes"):
            hardware_info["webgpu"] = True
            hardware_info["webgpu_info"] = {"simulation": True}
    except:
        pass
    
    # Check for wgpu-py package for potential WebGPU support
    try:
        if importlib.util.find_spec("wgpu") is not None:
            hardware_info["webgpu"] = True
            
            # Try to get wgpu details
            try:
                import wgpu
                hardware_info["webgpu_info"] = {
                    "type": "wgpu-py",
                    "version": getattr(wgpu, "__version__", "unknown"),
                    "simulation": False
                }
            except Exception as e:
                logger.debug(f"Error getting wgpu details: {e}")
                hardware_info["webgpu_info"] = {"type": "wgpu-py", "error": str(e)}
    except:
        pass

def is_hardware_simulated(hardware_type: str, hardware_info: Optional[Dict[str, Any]] = None) -> bool:
    """
    Checks if a hardware platform is being simulated
    
    Args:
        hardware_type: Type of hardware to check
        hardware_info: Optional hardware information dictionary (will detect if not provided)
        
    Returns:
        True if the hardware is simulated, False otherwise
    """
    if hardware_info is None:
        hardware_info = detect_available_hardware()
    
    # WebNN and WebGPU are typically simulated
    if hardware_type == "webnn":
        info = hardware_info.get("webnn_info", {})
        return info.get("simulation", True)
    
    if hardware_type == "webgpu":
        info = hardware_info.get("webgpu_info", {})
        return info.get("simulation", True)
    
    # Check QNN simulation
    if hardware_type == "qnn":
        # Check simulation environment variable
        if os.environ.get("QNN_SIMULATION_ENABLED", "").lower() in ("1", "true", "yes"):
            return True
        
        # Also check info for simulation flag
        info = hardware_info.get("qnn_info", {})
        return info.get("simulation", False)
    
    # Other hardware types are typically not simulated
    return False

def get_hardware_priority_list() -> List[str]:
    """
    Gets a list of hardware platforms in priority order
    
    Returns:
        List of hardware types in priority order
    """
    return ["cuda", "mps", "rocm", "openvino", "qnn", "webnn", "webgpu", "cpu"]

def get_recommended_device(model_type: str = None, model_name: str = None) -> str:
    """
    Gets the recommended device for a model
    
    Args:
        model_type: Optional model type
        model_name: Optional model name
        
    Returns:
        String with recommended device
    """
    hardware_info = detect_available_hardware()
    
    # First check model-specific preferences
    if model_type or model_name:
        # For memory-intensive models, prioritize platforms with more memory
        memory_intensive_patterns = [
            "llama", "gpt", "t5", "opt", "falcon", "mixtral", "llava"
        ]
        
        # For audio models, prioritize CPU or CUDA
        audio_patterns = [
            "whisper", "wav2vec2", "hubert", "clap"
        ]
        
        is_memory_intensive = False
        is_audio_model = False
        
        if model_name:
            model_name_lower = model_name.lower()
            is_memory_intensive = any(pattern in model_name_lower for pattern in memory_intensive_patterns)
            is_audio_model = any(pattern in model_name_lower for pattern in audio_patterns)
        
        if model_type:
            model_type_lower = model_type.lower()
            is_memory_intensive = is_memory_intensive or "generation" in model_type_lower
            is_audio_model = is_audio_model or "audio" in model_type_lower
        
        # Memory-intensive models should avoid MPS
        if is_memory_intensive and hardware_info.get("best_available") == "mps":
            if hardware_info.get("cuda", False):
                return "cuda"
            else:
                return "cpu"
        
        # Audio models should prioritize CUDA or CPU over other platforms
        if is_audio_model:
            if hardware_info.get("cuda", False):
                return "cuda"
            elif hardware_info.get("rocm", False):
                return "rocm"
            else:
                return "cpu"
    
    # Default to best available device
    return hardware_info.get("best_available", "cpu")

if __name__ == "__main__":
    # Configure logging to show debug info when run directly
    logger.setLevel(logging.DEBUG)
    
    # When run directly, detect hardware and print information
    hardware_info = detect_hardware_with_comprehensive_checks()
    print(json.dumps(hardware_info, indent=2))
    
    # Show simulation status
    for hw_type in ["cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]:
        simulated = is_hardware_simulated(hw_type, hardware_info)
        print(f"{hw_type}: {'Simulated' if simulated else 'Real hardware'}")
    
    # Show recommended device for different model types
    print(f"\nRecommended device for general use: {get_recommended_device()}")
    print(f"Recommended device for LLMs: {get_recommended_device('text_generation', 'llama')}")
    print(f"Recommended device for audio: {get_recommended_device('audio', 'whisper')}")
    print(f"Recommended device for vision: {get_recommended_device('vision', 'vit')}")
