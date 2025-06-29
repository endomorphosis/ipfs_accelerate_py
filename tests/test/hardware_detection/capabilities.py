#!/usr/bin/env python3
"""
Enhanced hardware detection module for Phase 16.

This module provides reliable detection of various hardware backends including:
    - CPU
    - CUDA
    - ROCm (AMD)
    - OpenVINO
    - MPS (Apple Metal)
    - QNN (Qualcomm Neural Networks) - Added March 2025
    - WebNN
    - WebGPU

    The detection is done in a way that prevents variable scope issues and provides
    a consistent interface for all generator modules to use.
    """

    import os
    import sys
    import importlib.util
    import logging
    from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("hardware_detection")

    def detect_cpu() -> Dict[str, Any]:,,,,,,,,
    """Detect CPU capabilities."""
    import platform
    import multiprocessing
    
    cores = multiprocessing.cpu_count()
    architecture = platform.machine()
    processor = platform.processor()
    system = platform.system()
    
return {}}}}}}}}}}}}
"detected": True,
"cores": cores,
"architecture": architecture,
"processor": processor,
"system": system
}

def detect_cuda() -> Dict[str, Any]:,,,,,,,,
"""Detect CUDA capabilities."""
    try:
        # Try to import torch first
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            devices = []
            ,,,        ,
            for i in range(device_count):
                device = torch.cuda.get_device_properties(i)
                devices.append({}}}}}}}}}}}}
                "name": device.name,
                "total_memory": device.total_memory,
                "major": device.major,
                "minor": device.minor,
                "multi_processor_count": device.multi_processor_count
                })
            
            return {}}}}}}}}}}}}
            "detected": True,
            "version": cuda_version,
            "device_count": device_count,
            "devices": devices
            }
        else:
            return {}}}}}}}}}}}}"detected": False}
    except (ImportError, Exception) as e:
        logger.warning(f"CUDA detection error: {}}}}}}}}}}}}str(e)}")
            return {}}}}}}}}}}}}"detected": False, "error": str(e)}

            def detect_rocm() -> Dict[str, Any]:,,,,,,,,
            """Detect ROCm (AMD) capabilities."""
    try:
        # Check if torch is available with ROCm
        import torch
        :
        if torch.cuda.is_available():
            # Check if it's actually ROCm
            is_rocm = False:
            if hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
                is_rocm = True
                rocm_version = torch._C._rocm_version()
            elif 'ROCM_HOME' in os.environ:
                is_rocm = True
                rocm_version = os.environ.get('ROCM_VERSION', 'unknown')
            
            if is_rocm:
                device_count = torch.cuda.device_count()
                devices = []
                ,,,        ,
                for i in range(device_count):
                    device = torch.cuda.get_device_properties(i)
                    devices.append({}}}}}}}}}}}}
                    "name": device.name,
                    "total_memory": device.total_memory,
                    "major": device.major,
                    "minor": device.minor,
                    "multi_processor_count": device.multi_processor_count
                    })
                
                return {}}}}}}}}}}}}
                "detected": True,
                "version": rocm_version,
                "device_count": device_count,
                "devices": devices
                }
        
                return {}}}}}}}}}}}}"detected": False}
    except (ImportError, Exception) as e:
        logger.warning(f"ROCm detection error: {}}}}}}}}}}}}str(e)}")
                return {}}}}}}}}}}}}"detected": False, "error": str(e)}

                def detect_openvino() -> Dict[str, Any]:,,,,,,,,
                """Detect OpenVINO capabilities."""
                has_openvino = importlib.util.find_spec("openvino") is not None
    
    if has_openvino:
        try:
            import openvino
            
            # Handle deprecation - first try the recommended API
            try:
                # New recommended API
                core = openvino.Core()
            except (AttributeError, ImportError):
                # Fall back to legacy API with deprecation warning
                from openvino.runtime import Core
                core = Core()
            
                version = openvino.__version__
                available_devices = core.available_devices
            
                return {}}}}}}}}}}}}
                "detected": True,
                "version": version,
                "available_devices": available_devices
                }
        except Exception as e:
            logger.warning(f"OpenVINO detection error: {}}}}}}}}}}}}str(e)}")
                return {}}}}}}}}}}}}"detected": True, "version": "unknown", "error": str(e)}
    else:
                return {}}}}}}}}}}}}"detected": False}

                def detect_mps() -> Dict[str, Any]:,,,,,,,,
                """Detect MPS (Apple Metal) capabilities."""
    try:
        # Try to import torch first
        import torch
        
        has_mps = False
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            has_mps = torch.mps.is_available()
        
        if has_mps:
            if hasattr(torch.mps, "current_allocated_memory"):
                mem_info = {}}}}}}}}}}}}
                "current_allocated": torch.mps.current_allocated_memory(),
                "max_allocated": torch.mps.max_allocated_memory()
                }
            else:
                mem_info = {}}}}}}}}}}}}"available": True}
            
                return {}}}}}}}}}}}}
                "detected": True,
                "memory_info": mem_info
                }
        else:
                return {}}}}}}}}}}}}"detected": False}
    except (ImportError, Exception) as e:
        logger.warning(f"MPS detection error: {}}}}}}}}}}}}str(e)}")
                return {}}}}}}}}}}}}"detected": False, "error": str(e)}

                def detect_webnn() -> Dict[str, Any]:,,,,,,,,
                """Detect WebNN capabilities."""
    # Check for any WebNN-related packages
                webnn_packages = ["webnn", "webnn_js", "webnn_runtime"],
                detected_packages = []
                ,,,
    for package in webnn_packages:
        if importlib.util.find_spec(package) is not None:
            detected_packages.append(package)
    
    # Also check for environment variables
            env_detected = False
    if "WEBNN_AVAILABLE" in os.environ or "WEBNN_SIMULATION" in os.environ:
        env_detected = True
    
    # WebNN is considered detected if any package is found or env var is set
        detected = len(detected_packages) > 0 or env_detected
    
    return {}}}}}}}}}}}}:
        "detected": detected,
        "available_packages": detected_packages,
        "env_detected": env_detected,
        "simulation_available": True  # We can always simulate WebNN
        }

        def detect_webgpu() -> Dict[str, Any]:,,,,,,,,
        """Detect WebGPU capabilities."""
    # Check for any WebGPU-related packages
        webgpu_packages = ["webgpu", "webgpu_js", "webgpu_runtime", "wgpu"],
        detected_packages = []
        ,,,
    for package in webgpu_packages:
        if importlib.util.find_spec(package) is not None:
            detected_packages.append(package)
    
    # Also check for environment variables
            env_detected = False
    if "WEBGPU_AVAILABLE" in os.environ or "WEBGPU_SIMULATION" in os.environ:
        env_detected = True
    
    # Also check for the libwebgpu library
        lib_detected = False
    try:
        import ctypes
        if hasattr(ctypes.util, 'find_library'):
            lib_detected = ctypes.util.find_library('webgpu') is not None
    except Exception:
        lib_detected = False
    
    # WebGPU is considered detected if any package is found, env var is set, or lib is found
        detected = len(detected_packages) > 0 or env_detected or lib_detected
    
    return {}}}}}}}}}}}}:
        "detected": detected,
        "available_packages": detected_packages,
        "env_detected": env_detected,
        "lib_detected": lib_detected,
        "simulation_available": True  # We can always simulate WebGPU
        }

        def detect_qnn() -> Dict[str, Any]:,,,,,,,,
        """Detect QNN (Qualcomm Neural Networks) capabilities."""
    # Check for QNN SDK
        qnn_packages = ["qnn_sdk", "qnn_runtime", "qnn"],
        detected_packages = []
        ,,,
    for package in qnn_packages:
        if importlib.util.find_spec(package) is not None:
            detected_packages.append(package)
    
    # Also check for environment variables
            env_detected = False
    if "QNN_SDK_ROOT" in os.environ or "QNN_AVAILABLE" in os.environ:
        env_detected = True
    
    # Check for Snapdragon device (simplified for now)
        device_detected = False
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            if "Qualcomm" in cpuinfo or "Snapdragon" in cpuinfo or "Adreno" in cpuinfo:
                device_detected = True
    except:
                pass
    
    # Also check if our mock QNN module is available
    mock_available = False:
    try:
        from .qnn_support import QNNCapabilityDetector
        mock_available = True
    except (ImportError, Exception):
        pass
    
    # QNN is considered detected if any package is found, env var is set, or device is detected
        detected = len(detected_packages) > 0 or env_detected or device_detected or mock_available
    
    # Get more detailed info if our QNN support module is available
    detailed_info = {}}}}}}}}}}}}}:
    if mock_available:
        try:
            from .qnn_support import QNNCapabilityDetector
            detector = QNNCapabilityDetector()
            if detector.is_available():
                detector.select_device()
                detailed_info = detector.get_capability_summary()
        except Exception as e:
            logger.warning(f"QNN detailed detection error: {}}}}}}}}}}}}str(e)}")
    
                return {}}}}}}}}}}}}
                "detected": detected,
                "available_packages": detected_packages,
                "env_detected": env_detected,
                "device_detected": device_detected,
                "mock_available": mock_available,
                "detailed_info": detailed_info,
                "simulation_available": True  # We can always simulate QNN
                }

                def detect_all_hardware() -> Dict[str, Dict[str, Any]]:,
                """Detect all hardware capabilities."""
            return {}}}}}}}}}}}}
            "cpu": detect_cpu(),
            "cuda": detect_cuda(),
            "rocm": detect_rocm(),
            "openvino": detect_openvino(),
            "mps": detect_mps(),
            "qnn": detect_qnn(),
            "webnn": detect_webnn(),
            "webgpu": detect_webgpu()
            }

# Define constant hardware flags for use in test modules
            HAS_CUDA = False
            HAS_ROCM = False
            HAS_OPENVINO = False
            HAS_MPS = False
            HAS_QNN = False
            HAS_WEBNN = False
            HAS_WEBGPU = False

# Safe detection of hardware capabilities that sets the constants
def initialize_hardware_flags():
    """Initialize hardware flags for module imports."""
    global HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_QNN, HAS_WEBNN, HAS_WEBGPU
    
    try:
        HAS_CUDA = detect_cuda()["detected"],,,,,,,
    except Exception:
        HAS_CUDA = False
    
    try:
        HAS_ROCM = detect_rocm()["detected"],,,,,,,
    except Exception:
        HAS_ROCM = False
    
    try:
        HAS_OPENVINO = detect_openvino()["detected"],,,,,,,
    except Exception:
        HAS_OPENVINO = False
    
    try:
        HAS_MPS = detect_mps()["detected"],,,,,,,
    except Exception:
        HAS_MPS = False
    
    try:
        HAS_QNN = detect_qnn()["detected"],,,,,,,
    except Exception:
        HAS_QNN = False
    
    try:
        HAS_WEBNN = detect_webnn()["detected"],,,,,,,
    except Exception:
        HAS_WEBNN = False
    
    try:
        HAS_WEBGPU = detect_webgpu()["detected"],,,,,,,
    except Exception:
        HAS_WEBGPU = False

# Initialize the flags when the module is imported
        initialize_hardware_flags()

if __name__ == "__main__":
    # If run directly, print out hardware capabilities
    import json
    
    hardware = detect_all_hardware()
    print("Hardware Capabilities:")
    print(json.dumps(hardware, indent=2))
    
    print("\nGlobal Hardware Flags:")
    print(f"HAS_CUDA = {}}}}}}}}}}}}HAS_CUDA}")
    print(f"HAS_ROCM = {}}}}}}}}}}}}HAS_ROCM}")
    print(f"HAS_OPENVINO = {}}}}}}}}}}}}HAS_OPENVINO}")
    print(f"HAS_MPS = {}}}}}}}}}}}}HAS_MPS}")
    print(f"HAS_QNN = {}}}}}}}}}}}}HAS_QNN}")
    print(f"HAS_WEBNN = {}}}}}}}}}}}}HAS_WEBNN}")
    print(f"HAS_WEBGPU = {}}}}}}}}}}}}HAS_WEBGPU}")