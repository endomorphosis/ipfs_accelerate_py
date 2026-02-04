"""
ROCm hardware backend implementation for AMD GPUs.
"""

from typing import Dict, Any, Optional
import logging

from test.tools.skills.refactored_benchmark_suite.hardware.base import HardwareBackend

logger = logging.getLogger("benchmark.hardware.rocm")

class ROCmBackend(HardwareBackend):
    """ROCm hardware backend for AMD GPUs."""
    
    name = "rocm"
    
    def __init__(self, device_index: int = 0):
        """
        Initialize ROCm backend.
        
        Args:
            device_index: ROCm device index
        """
        super().__init__()
        self.device_index = device_index
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if ROCm is available.
        
        Returns:
            True if ROCm is available, False otherwise
        """
        try:
            import torch
            return hasattr(torch, "hip") and torch.hip.is_available()
        except (ImportError, AttributeError):
            return False
    
    @classmethod
    def get_device_count(cls) -> int:
        """
        Get number of available ROCm devices.
        
        Returns:
            Number of ROCm devices
        """
        try:
            import torch
            if hasattr(torch, "hip") and torch.hip.is_available():
                return torch.hip.device_count()
            return 0
        except (ImportError, AttributeError):
            return 0
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        Get information about the ROCm hardware.
        
        Returns:
            Dictionary with ROCm information
        """
        info = {"available": cls.is_available()}
        
        if not info["available"]:
            return info
        
        try:
            import torch
            
            device_info = []
            
            for device in range(torch.hip.device_count()):
                device_data = {
                    "index": device,
                    "name": torch.cuda.get_device_name(device)  # ROCm uses CUDA API
                }
                
                if hasattr(torch.cuda, "get_device_properties"):
                    props = torch.cuda.get_device_properties(device)
                    device_data.update({
                        "total_memory": props.total_memory,
                        "multi_processor_count": props.multi_processor_count
                    })
                
                device_info.append(device_data)
            
            info.update({
                "device_count": torch.hip.device_count(),
                "current_device": torch.cuda.current_device(),  # ROCm uses CUDA API
                "devices": device_info
            })
            
            # Try to get ROCm version
            try:
                import subprocess
                result = subprocess.run(['rocm-smi', '--showversion'], 
                                       capture_output=True, text=True, check=True)
                info["rocm_version"] = result.stdout.strip()
            except (ImportError, subprocess.SubprocessError):
                pass
            
        except (ImportError, AttributeError):
            logger.warning("PyTorch ROCm support not available")
        
        return info
    
    def initialize(self) -> Any:
        """
        Initialize ROCm for use.
        
        Returns:
            torch.device("cuda") for ROCm (ROCm uses CUDA device interface)
        """
        try:
            import torch
            if not self.is_available():
                logger.warning("ROCm not available")
                return None
            
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()  # ROCm uses CUDA API
            
            if self.device_index >= torch.hip.device_count():
                logger.warning(f"Requested ROCm device {self.device_index} not available, using device 0")
                self.device_index = 0
            
            # Set device (ROCm uses CUDA API)
            torch.cuda.set_device(self.device_index)
            
            self.initialized = True
            return torch.device(f"cuda:{self.device_index}")  # ROCm uses CUDA device interface
        except (ImportError, AttributeError):
            logger.warning("PyTorch ROCm support not available")
            return None
    
    def cleanup(self) -> None:
        """
        Cleanup ROCm resources.
        """
        try:
            import torch
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()  # ROCm uses CUDA API
        except (ImportError, AttributeError):
            pass
        
        self.initialized = False