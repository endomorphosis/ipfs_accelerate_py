"""
Hardware Manager Module

This module provides hardware detection and management functionality.
"""

import logging
import platform
import os
from typing import Dict, Any, Optional, List, Type

logger = logging.getLogger(__name__)

class HardwareBackend:
    """Base class for hardware backend implementations."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hardware backend.
        
        Args:
            name: Hardware backend name
            config: Optional configuration
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"Hardware.{name}")
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get hardware information.
        
        Returns:
            Dictionary containing hardware information
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__
        }
        
    def is_available(self) -> bool:
        """
        Check if hardware is available.
        
        Returns:
            True if hardware is available, False otherwise
        """
        raise NotImplementedError("Subclasses must implement is_available()")
        
    def setup(self) -> None:
        """Set up hardware for benchmark."""
        pass
        
    def cleanup(self) -> None:
        """Clean up hardware resources."""
        pass


class CPUBackend(HardwareBackend):
    """CPU hardware backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CPU backend."""
        super().__init__("cpu", config)
        
    def is_available(self) -> bool:
        """CPU is always available."""
        return True
        
    def get_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        info = super().get_info()
        
        info.update({
            'processor': platform.processor(),
            'cores': os.cpu_count(),
            'system': platform.system(),
            'architecture': platform.machine()
        })
        
        return info


class CUDABackend(HardwareBackend):
    """CUDA hardware backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CUDA backend."""
        super().__init__("cuda", config)
        
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def get_info(self) -> Dict[str, Any]:
        """Get CUDA information."""
        info = super().get_info()
        
        if not self.is_available():
            info['available'] = False
            return info
            
        try:
            import torch
            
            info.update({
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0)
            })
            
            devices = []
            for i in range(torch.cuda.device_count()):
                devices.append({
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory': torch.cuda.get_device_properties(i).total_memory
                })
                
            info['devices'] = devices
            
        except Exception as e:
            info['error'] = str(e)
            
        return info


class ROCmBackend(HardwareBackend):
    """ROCm (AMD GPU) hardware backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ROCm backend."""
        super().__init__("rocm", config)
        
    def is_available(self) -> bool:
        """Check if ROCm is available."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
                
            # Check if this is an AMD GPU
            device_name = torch.cuda.get_device_name(0).lower()
            return any(x in device_name for x in ["amd", "radeon"])
        except ImportError:
            return False
            
    def get_info(self) -> Dict[str, Any]:
        """Get ROCm information."""
        info = super().get_info()
        
        if not self.is_available():
            info['available'] = False
            return info
            
        try:
            import torch
            
            info.update({
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0)
            })
            
            devices = []
            for i in range(torch.cuda.device_count()):
                devices.append({
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory': torch.cuda.get_device_properties(i).total_memory
                })
                
            info['devices'] = devices
            
            # Add ROCm-specific info
            environ_vars = {}
            for key in ['HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES']:
                if key in os.environ:
                    environ_vars[key] = os.environ[key]
                    
            info['environ_vars'] = environ_vars
            
        except Exception as e:
            info['error'] = str(e)
            
        return info


class MPSBackend(HardwareBackend):
    """Apple Metal Performance Shaders backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MPS backend."""
        super().__init__("mps", config)
        
    def is_available(self) -> bool:
        """Check if MPS is available."""
        try:
            import torch
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except ImportError:
            return False
            
    def get_info(self) -> Dict[str, Any]:
        """Get MPS information."""
        info = super().get_info()
        
        if not self.is_available():
            info['available'] = False
            return info
            
        try:
            import torch
            
            info.update({
                'available': True,
                'built': torch.backends.mps.is_built(),
                'system': platform.system(),
                'architecture': platform.machine()
            })
            
        except Exception as e:
            info['error'] = str(e)
            
        return info


class WebGPUBackend(HardwareBackend):
    """WebGPU hardware backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize WebGPU backend."""
        super().__init__("webgpu", config)
        
    def is_available(self) -> bool:
        """Check if WebGPU is available (mock implementation)."""
        # In a real implementation, this would check for WebGPU availability
        # through browser detection or a WebGPU runtime
        return False
        
    def get_info(self) -> Dict[str, Any]:
        """Get WebGPU information."""
        info = super().get_info()
        info['available'] = self.is_available()
        return info


class WebNNBackend(HardwareBackend):
    """WebNN hardware backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize WebNN backend."""
        super().__init__("webnn", config)
        
    def is_available(self) -> bool:
        """Check if WebNN is available (mock implementation)."""
        # In a real implementation, this would check for WebNN availability
        # through browser detection or a WebNN runtime
        return False
        
    def get_info(self) -> Dict[str, Any]:
        """Get WebNN information."""
        info = super().get_info()
        info['available'] = self.is_available()
        return info


class OpenVINOBackend(HardwareBackend):
    """OpenVINO hardware backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenVINO backend."""
        super().__init__("openvino", config)
        
    def is_available(self) -> bool:
        """Check if OpenVINO is available."""
        try:
            from openvino.runtime import Core
            return True
        except ImportError:
            try:
                # Alternative import for older versions
                from openvino import Core
                return True
            except ImportError:
                return False
        except Exception:
            return False
            
    def get_info(self) -> Dict[str, Any]:
        """Get OpenVINO information."""
        info = super().get_info()
        
        if not self.is_available():
            info['available'] = False
            return info
            
        try:
            # Try to get OpenVINO version and device info
            try:
                from openvino.runtime import Core, get_version
                core = Core()
                info.update({
                    'available': True,
                    'version': get_version(),
                    'devices': core.available_devices
                })
            except ImportError:
                # Alternative import for older versions
                from openvino import Core
                core = Core()
                info.update({
                    'available': True,
                    'devices': core.available_devices
                })
                
        except Exception as e:
            info['error'] = str(e)
            
        return info


class QualcommBackend(HardwareBackend):
    """Qualcomm Neural Processing hardware backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Qualcomm backend."""
        super().__init__("qnn", config)
        
    def is_available(self) -> bool:
        """Check if Qualcomm Neural Processing is available."""
        # Simple check for QNN SDK availability
        # In a real implementation, this would check for Qualcomm SDK installation
        return False
        
    def get_info(self) -> Dict[str, Any]:
        """Get Qualcomm information."""
        info = super().get_info()
        info['available'] = self.is_available()
        
        # Add mock device info
        if self.is_available():
            info.update({
                'device': 'Qualcomm Neural Processor',
                'sdk_version': '2.0'
            })
            
        return info


# Registry of hardware backend implementations
hardware_backends: Dict[str, Type[HardwareBackend]] = {
    "cpu": CPUBackend,
    "cuda": CUDABackend,
    "rocm": ROCmBackend,
    "mps": MPSBackend,
    "openvino": OpenVINOBackend,
    "qnn": QualcommBackend,
    "webgpu": WebGPUBackend,
    "webnn": WebNNBackend
}


class HardwareManager:
    """
    Hardware detection and management.
    
    This class provides a unified interface for detecting and managing
    hardware backends for benchmarks.
    """
    
    def __init__(self):
        """Initialize hardware manager."""
        self.logger = logging.getLogger("HardwareManager")
        self._available_hardware = None
        self.initialized_backends = {}
        
    def detect_available_hardware(self) -> Dict[str, bool]:
        """
        Detect available hardware backends.
        
        Returns:
            Dictionary mapping hardware names to availability status
        """
        if self._available_hardware is not None:
            return self._available_hardware
            
        self._available_hardware = {}
        
        for name, backend_class in hardware_backends.items():
            try:
                backend = backend_class()
                is_available = backend.is_available()
                self._available_hardware[name] = is_available
                
                if is_available:
                    self.logger.info(f"Detected available hardware: {name}")
                else:
                    self.logger.debug(f"Hardware not available: {name}")
                    
            except Exception as e:
                self.logger.warning(f"Error detecting hardware {name}: {e}")
                self._available_hardware[name] = False
                
        return self._available_hardware
        
    def get_hardware(self, hardware_name: str, config: Optional[Dict[str, Any]] = None) -> HardwareBackend:
        """
        Get hardware implementation by name.
        
        Args:
            hardware_name: Name of hardware backend
            config: Optional configuration for hardware backend
            
        Returns:
            Hardware backend instance
            
        Raises:
            ValueError: If hardware is not available
        """
        # Auto-detect if needed
        if self._available_hardware is None:
            self.detect_available_hardware()
            
        # Validate hardware name
        if hardware_name not in hardware_backends:
            raise ValueError(f"Unknown hardware backend: {hardware_name}")
            
        # Check if hardware is available
        if not self._available_hardware.get(hardware_name, False):
            raise ValueError(f"Hardware not available: {hardware_name}")
            
        # Reuse existing instance or create new one
        if hardware_name in self.initialized_backends:
            return self.initialized_backends[hardware_name]
            
        # Create and initialize hardware backend
        backend = hardware_backends[hardware_name](config)
        backend.setup()
        
        # Cache backend for reuse
        self.initialized_backends[hardware_name] = backend
        
        return backend
        
    def get_hardware_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available hardware.
        
        Returns:
            Dictionary mapping hardware names to information
        """
        # Auto-detect if needed
        if self._available_hardware is None:
            self.detect_available_hardware()
            
        info = {}
        
        for name, available in self._available_hardware.items():
            if available:
                try:
                    backend = self.get_hardware(name)
                    info[name] = backend.get_info()
                except Exception as e:
                    info[name] = {
                        'name': name,
                        'available': False,
                        'error': str(e)
                    }
            else:
                info[name] = {
                    'name': name,
                    'available': False
                }
                
        return info
        
    def cleanup(self) -> None:
        """Clean up all initialized hardware backends."""
        for name, backend in self.initialized_backends.items():
            try:
                backend.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up hardware {name}: {e}")
                
        self.initialized_backends = {}