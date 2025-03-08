"""
Hardware profile configuration for model acceleration.

This module provides a comprehensive hardware profile configuration system
that supports all hardware backends available in the IPFS acceleration system.
"""

from typing import Dict, Any, List, Optional, Union, Literal

class HardwareProfile:
    """
    Configuration profile for specific hardware backends.
    
    This class encapsulates all configuration options for specific hardware
    backends, providing a consistent interface for hardware-specific settings.
    """
    
    def __init__(
        self,
        backend: str = "auto",
        device_id: Union[int, str] = 0,
        memory_limit: Optional[Union[int, str]] = None,
        precision: str = "auto",
        optimization_level: Literal["default", "performance", "memory", "balanced"] = "default",
        quantization: Optional[Dict[str, Any]] = None,
        feature_flags: Optional[Dict[str, bool]] = None,
        compiler_options: Optional[Dict[str, Any]] = None,
        browser: Optional[str] = None,
        browser_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize a hardware profile with specific configuration.
        
        Args:
            backend: Hardware backend name (e.g., "cuda", "openvino", "webgpu")
            device_id: Specific device ID or name (e.g., 0 for cuda:0)
            memory_limit: Maximum memory usage (e.g., "4GB")
            precision: Computation precision (e.g., "fp32", "fp16", "int8", "auto")
            optimization_level: Overall optimization strategy
            quantization: Quantization-specific configuration
            feature_flags: Enable/disable specific hardware features
            compiler_options: Backend-specific compiler options
            browser: Browser name for WebNN/WebGPU backends (e.g., "chrome", "firefox")
            browser_options: Browser-specific configuration options
            **kwargs: Additional backend-specific options
        """
        self.backend = backend
        self.device_id = device_id
        self.memory_limit = memory_limit
        self.precision = precision
        self.optimization_level = optimization_level
        self.quantization = quantization or {}
        self.feature_flags = feature_flags or {}
        self.compiler_options = compiler_options or {}
        self.browser = browser
        self.browser_options = browser_options or {}
        self.extra_options = kwargs
        
        # Map legacy backend names to standardized names
        self._normalize_backend_name()
        
    def _normalize_backend_name(self):
        """Normalize backend name to standard format."""
        backend_mapping = {
            "gpu": "cuda",
            "nvidia": "cuda",
            "amd": "rocm",
            "apple": "mps",
            "intel": "openvino",
            "qnn": "qualcomm",
            "snapdragon": "qualcomm",
            "web": "webgpu"
        }
        
        self.backend = backend_mapping.get(self.backend.lower(), self.backend.lower())
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert hardware profile to dictionary format."""
        return {
            "backend": self.backend,
            "device_id": self.device_id,
            "memory_limit": self.memory_limit,
            "precision": self.precision,
            "optimization_level": self.optimization_level,
            "quantization": self.quantization,
            "feature_flags": self.feature_flags,
            "compiler_options": self.compiler_options,
            "browser": self.browser,
            "browser_options": self.browser_options,
            **self.extra_options
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HardwareProfile':
        """Create hardware profile from dictionary configuration."""
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        """String representation of hardware profile."""
        return f"HardwareProfile(backend={self.backend}, device_id={self.device_id}, precision={self.precision})"
    
    def get_worker_compatible_config(self) -> Dict[str, Any]:
        """
        Get configuration compatible with the worker architecture.
        
        This method converts the hardware profile to a format compatible
        with the existing worker implementation for backward compatibility.
        """
        worker_config = {
            "hardware_type": self.backend,
            "device_id": self.device_id,
        }
        
        # Map precision to worker format
        if self.precision != "auto":
            worker_config["precision"] = self.precision
            
        # Add memory limit if specified
        if self.memory_limit:
            worker_config["memory_limit"] = self.memory_limit
            
        # Add browser configuration for web backends
        if self.backend in ["webgpu", "webnn"] and self.browser:
            worker_config["browser"] = self.browser
            worker_config["browser_options"] = self.browser_options
            
        # Add optimization level
        if self.optimization_level != "default":
            worker_config["optimization_level"] = self.optimization_level
            
        # Add feature flags
        if self.feature_flags:
            for flag, value in self.feature_flags.items():
                worker_config[flag] = value
                
        return worker_config