#!/usr/bin/env python3
"""
Hardware Detector

This module provides hardware detection capabilities for the generator system.
"""

import os
import sys
import logging
import platform
from typing import Dict, Any, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwareDetectorBase:
    """
    Base class for hardware detectors.
    
    This class defines the interface for all hardware detectors.
    """
    
    def detect(self) -> Dict[str, Any]:
        """
        Detect hardware availability and properties.
        
        Returns:
            Dictionary with detection results
        """
        raise NotImplementedError("Subclasses must implement detect()")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this detector.
        
        Returns:
            Dictionary with metadata
        """
        return {
            "name": self.__class__.__name__,
            "version": "1.0.0",
            "description": "Base hardware detector"
        }

class CUDADetector(HardwareDetectorBase):
    """Detector for NVIDIA CUDA GPUs."""
    
    def detect(self) -> Dict[str, Any]:
        """
        Detect CUDA availability and properties.
        
        Returns:
            Dictionary with CUDA detection results
        """
        result = {
            "available": False,
            "version": None,
            "devices": [],
            "device_count": 0
        }
        
        try:
            # Try to import torch
            import torch
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            result["available"] = cuda_available
            
            if cuda_available:
                # Get CUDA version
                result["version"] = torch.version.cuda
                
                # Get device count
                device_count = torch.cuda.device_count()
                result["device_count"] = device_count
                
                # Get device properties
                devices = []
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        "index": i,
                        "name": props.name,
                        "total_memory": props.total_memory,
                        "total_memory_gb": round(props.total_memory / (1024**3), 2),
                        "major": props.major,
                        "minor": props.minor,
                        "multi_processor_count": props.multi_processor_count
                    })
                
                result["devices"] = devices
                
                # Log CUDA information
                logger.info(f"CUDA available: version {result['version']}")
                logger.info(f"Number of CUDA devices: {result['device_count']}")
                for device in devices:
                    logger.info(f"CUDA Device {device['index']}: {device['name']} with {device['total_memory_gb']} GB memory")
            else:
                logger.info("CUDA not available")
                
        except ImportError:
            logger.info("CUDA detection skipped (torch not installed)")
        except Exception as e:
            logger.warning(f"Error detecting CUDA: {str(e)}")
            
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this detector."""
        metadata = super().get_metadata()
        metadata.update({
            "name": "CUDADetector",
            "description": "Detector for NVIDIA CUDA GPUs"
        })
        return metadata

class ROCmDetector(HardwareDetectorBase):
    """Detector for AMD ROCm GPUs."""
    
    def detect(self) -> Dict[str, Any]:
        """
        Detect ROCm availability and properties.
        
        Returns:
            Dictionary with ROCm detection results
        """
        result = {
            "available": False,
            "version": None,
            "devices": [],
            "device_count": 0
        }
        
        try:
            # Try to import torch
            import torch
            
            # Check ROCm via torch._C._rocm_version()
            if hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
                result["available"] = True
                result["version"] = torch._C._rocm_version()
                
                # If CUDA interface is available, get device information
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    result["device_count"] = device_count
                    
                    # Get device properties
                    devices = []
                    for i in range(device_count):
                        props = torch.cuda.get_device_properties(i)
                        devices.append({
                            "index": i,
                            "name": props.name,
                            "total_memory": props.total_memory,
                            "total_memory_gb": round(props.total_memory / (1024**3), 2),
                            "major": props.major,
                            "minor": props.minor,
                            "multi_processor_count": props.multi_processor_count
                        })
                    
                    result["devices"] = devices
                    
                # Log ROCm information
                logger.info(f"ROCm available: version {result['version']}")
                logger.info(f"Number of ROCm devices: {result['device_count']}")
                for device in result["devices"]:
                    logger.info(f"ROCm Device {device['index']}: {device['name']} with {device['total_memory_gb']} GB memory")
            
            # Check ROCm via environment variables
            elif 'ROCM_HOME' in os.environ:
                result["available"] = True
                logger.info("ROCm available (detected via ROCM_HOME)")
            else:
                logger.info("ROCm not available")
                
        except ImportError:
            logger.info("ROCm detection skipped (torch not installed)")
        except Exception as e:
            logger.warning(f"Error detecting ROCm: {str(e)}")
            
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this detector."""
        metadata = super().get_metadata()
        metadata.update({
            "name": "ROCmDetector",
            "description": "Detector for AMD ROCm GPUs"
        })
        return metadata

class MPSDetector(HardwareDetectorBase):
    """Detector for Apple Metal Performance Shaders (MPS)."""
    
    def detect(self) -> Dict[str, Any]:
        """
        Detect MPS availability and properties.
        
        Returns:
            Dictionary with MPS detection results
        """
        result = {
            "available": False,
            "device_name": None,
            "pytorch_support": False,
            "tensorflow_support": False
        }
        
        try:
            # Check if running on macOS
            if platform.system() != "Darwin":
                logger.info("MPS detection skipped (not running on macOS)")
                return result
                
            # Try to import torch
            import torch
            
            # Check MPS availability
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                result["available"] = True
                result["pytorch_support"] = True
                
                # Get device name (Apple Silicon or AMD GPU)
                if "arm" in platform.processor().lower():
                    result["device_name"] = "Apple Silicon"
                else:
                    result["device_name"] = "AMD GPU"
                    
                logger.info(f"MPS available: {result['device_name']}")
            else:
                logger.info("MPS not available")
                
        except ImportError:
            logger.info("MPS detection skipped (torch not installed)")
        except Exception as e:
            logger.warning(f"Error detecting MPS: {str(e)}")
            
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this detector."""
        metadata = super().get_metadata()
        metadata.update({
            "name": "MPSDetector",
            "description": "Detector for Apple Metal Performance Shaders (MPS)"
        })
        return metadata

class OpenVINODetector(HardwareDetectorBase):
    """Detector for Intel OpenVINO."""
    
    def detect(self) -> Dict[str, Any]:
        """
        Detect OpenVINO availability and properties.
        
        Returns:
            Dictionary with OpenVINO detection results
        """
        result = {
            "available": False,
            "version": None,
            "devices": [],
            "supported_devices": []
        }
        
        try:
            # Try to import openvino
            import openvino
            from openvino.runtime import Core
            
            # Check OpenVINO availability
            result["available"] = True
            result["version"] = openvino.__version__
            
            # Create Core object
            core = Core()
            
            # Get available devices
            devices = core.available_devices
            result["devices"] = devices
            
            # Get supported devices
            supported_devices = ["CPU"]
            if "GPU" in devices:
                supported_devices.append("GPU")
            if "MYRIAD" in devices:
                supported_devices.append("MYRIAD")
            if "HDDL" in devices:
                supported_devices.append("HDDL")
            if "NPU" in devices:
                supported_devices.append("NPU")
                
            result["supported_devices"] = supported_devices
            
            # Log OpenVINO information
            logger.info(f"OpenVINO available: version {result['version']}")
            logger.info(f"OpenVINO devices: {', '.join(result['devices'])}")
            
        except ImportError:
            logger.info("OpenVINO detection skipped (openvino not installed)")
        except Exception as e:
            logger.warning(f"Error detecting OpenVINO: {str(e)}")
            
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this detector."""
        metadata = super().get_metadata()
        metadata.update({
            "name": "OpenVINODetector",
            "description": "Detector for Intel OpenVINO"
        })
        return metadata

class WebGPUDetector(HardwareDetectorBase):
    """Detector for WebGPU."""
    
    def detect(self) -> Dict[str, Any]:
        """
        Detect WebGPU availability and properties.
        
        Returns:
            Dictionary with WebGPU detection results
        """
        result = {
            "available": False,
            "library_found": False,
            "browser_support": False
        }
        
        try:
            # Check for shared library
            import ctypes.util
            webgpu_lib = ctypes.util.find_library("webgpu")
            result["library_found"] = webgpu_lib is not None
            
            # Check for wgpu package
            try:
                import wgpu
                result["available"] = True
                result["package_found"] = True
                result["package_version"] = getattr(wgpu, "__version__", "unknown")
                logger.info(f"WebGPU available (wgpu package version {result['package_version']})")
            except ImportError:
                result["package_found"] = False
                
            # Determine availability
            result["available"] = result["library_found"] or result["package_found"]
            
            # Browser support is not detectable in Python directly
            result["browser_support"] = "unknown"
            
            if result["available"]:
                logger.info("WebGPU available")
            else:
                logger.info("WebGPU not available")
                
        except Exception as e:
            logger.warning(f"Error detecting WebGPU: {str(e)}")
            
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this detector."""
        metadata = super().get_metadata()
        metadata.update({
            "name": "WebGPUDetector",
            "description": "Detector for WebGPU"
        })
        return metadata

class WebNNDetector(HardwareDetectorBase):
    """Detector for WebNN."""
    
    def detect(self) -> Dict[str, Any]:
        """
        Detect WebNN availability and properties.
        
        Returns:
            Dictionary with WebNN detection results
        """
        result = {
            "available": False,
            "library_found": False,
            "browser_support": False
        }
        
        try:
            # Check for shared library
            import ctypes.util
            webnn_lib = ctypes.util.find_library("webnn")
            result["library_found"] = webnn_lib is not None
            
            # Determine availability
            result["available"] = result["library_found"]
            
            # Browser support is not detectable in Python directly
            result["browser_support"] = "unknown"
            
            if result["available"]:
                logger.info("WebNN available")
            else:
                logger.info("WebNN not available")
                
        except Exception as e:
            logger.warning(f"Error detecting WebNN: {str(e)}")
            
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this detector."""
        metadata = super().get_metadata()
        metadata.update({
            "name": "WebNNDetector",
            "description": "Detector for WebNN"
        })
        return metadata

class CPUDetector(HardwareDetectorBase):
    """Detector for CPU information."""
    
    def detect(self) -> Dict[str, Any]:
        """
        Detect CPU information.
        
        Returns:
            Dictionary with CPU detection results
        """
        result = {
            "available": True,
            "name": platform.processor() or "Unknown CPU",
            "architecture": platform.machine(),
            "system": platform.system(),
            "python_version": platform.python_version(),
            "cores": self._get_cpu_cores()
        }
        
        logger.info(f"CPU: {result['name']} ({result['cores']} cores, {result['architecture']})")
        
        return result
    
    def _get_cpu_cores(self) -> int:
        """Get the number of CPU cores."""
        try:
            import os
            
            # Try to get the number of CPU cores
            try:
                # Python 3.4+
                return os.cpu_count() or 1
            except AttributeError:
                try:
                    # Linux, Unix
                    import multiprocessing
                    return multiprocessing.cpu_count()
                except (ImportError, NotImplementedError):
                    return 1
                    
        except Exception:
            return 1
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this detector."""
        metadata = super().get_metadata()
        metadata.update({
            "name": "CPUDetector",
            "description": "Detector for CPU information"
        })
        return metadata

class HardwareDetector:
    """
    Main hardware detector that combines all individual detectors.
    
    This class provides a unified interface for hardware detection.
    """
    
    def __init__(self, config=None):
        """
        Initialize the hardware detector.
        
        Args:
            config: Configuration object or dict
        """
        self.config = config or {}
        self.detectors = {}
        
        # Register default detectors
        self._register_default_detectors()
    
    def register_detector(self, hardware_type: str, detector: HardwareDetectorBase) -> None:
        """
        Register a hardware detector.
        
        Args:
            hardware_type: Hardware type
            detector: Hardware detector
        """
        self.detectors[hardware_type] = detector
        logger.debug(f"Registered {hardware_type} detector")
    
    def detect_all(self) -> Dict[str, Any]:
        """
        Detect all available hardware.
        
        Returns:
            Dictionary with all detection results
        """
        results = {}
        
        # Run all registered detectors
        for hardware_type, detector in self.detectors.items():
            # Skip if disabled in config
            if self.config.get(f"detect_{hardware_type}", True) is False:
                logger.debug(f"Skipping {hardware_type} detection (disabled in config)")
                results[hardware_type] = {"available": False, "disabled": True}
                continue
                
            try:
                # Run detector
                logger.debug(f"Running {hardware_type} detector")
                results[hardware_type] = detector.detect()
            except Exception as e:
                logger.warning(f"Error detecting {hardware_type}: {str(e)}")
                results[hardware_type] = {"available": False, "error": str(e)}
                
        return results
    
    def detect(self, hardware_type: str) -> Dict[str, Any]:
        """
        Detect specific hardware.
        
        Args:
            hardware_type: Hardware type to detect
            
        Returns:
            Dictionary with detection results for the specified hardware
        """
        # Skip if disabled in config
        if self.config.get(f"detect_{hardware_type}", True) is False:
            logger.debug(f"Skipping {hardware_type} detection (disabled in config)")
            return {"available": False, "disabled": True}
            
        # Check if detector exists
        if hardware_type not in self.detectors:
            logger.warning(f"No detector for {hardware_type}")
            return {"available": False, "error": f"No detector for {hardware_type}"}
            
        try:
            # Run detector
            logger.debug(f"Running {hardware_type} detector")
            return self.detectors[hardware_type].detect()
        except Exception as e:
            logger.warning(f"Error detecting {hardware_type}: {str(e)}")
            return {"available": False, "error": str(e)}
    
    def _register_default_detectors(self) -> None:
        """Register default hardware detectors."""
        self.register_detector("cpu", CPUDetector())
        self.register_detector("cuda", CUDADetector())
        self.register_detector("rocm", ROCmDetector())
        self.register_detector("mps", MPSDetector())
        self.register_detector("openvino", OpenVINODetector())
        self.register_detector("webgpu", WebGPUDetector())
        self.register_detector("webnn", WebNNDetector())
    
    def get_device_recommendations(self) -> Dict[str, Any]:
        """
        Get device recommendations based on detected hardware.
        
        Returns:
            Dictionary with device recommendations
        """
        # Detect all hardware
        hardware = self.detect_all()
        
        # Initialize recommendations
        recommendations = {
            "best_device": "cpu",
            "available_devices": ["cpu"],
            "reason": "Only CPU is available"
        }
        
        # Check CUDA
        if hardware.get("cuda", {}).get("available", False):
            recommendations["best_device"] = "cuda:0"
            recommendations["available_devices"].append("cuda")
            recommendations["reason"] = "CUDA is available"
            
            # Add device count
            device_count = hardware["cuda"].get("device_count", 1)
            if device_count > 1:
                recommendations["available_devices"].extend([f"cuda:{i}" for i in range(device_count)])
                
        # Check ROCm
        elif hardware.get("rocm", {}).get("available", False):
            recommendations["best_device"] = "cuda:0"  # ROCm uses CUDA interface
            recommendations["available_devices"].append("cuda")
            recommendations["reason"] = "ROCm is available"
            
        # Check MPS
        elif hardware.get("mps", {}).get("available", False):
            recommendations["best_device"] = "mps"
            recommendations["available_devices"].append("mps")
            recommendations["reason"] = "MPS is available"
            
        # Check OpenVINO
        if hardware.get("openvino", {}).get("available", False):
            recommendations["available_devices"].append("openvino")
            supported_devices = hardware["openvino"].get("supported_devices", [])
            
            # Add OpenVINO device types
            for device in supported_devices:
                recommendations["available_devices"].append(f"openvino_{device.lower()}")
                
            # Set as best device if no GPU is available
            if recommendations["best_device"] == "cpu":
                recommendations["best_device"] = "openvino_cpu"
                recommendations["reason"] = "OpenVINO is available"
                
        return recommendations