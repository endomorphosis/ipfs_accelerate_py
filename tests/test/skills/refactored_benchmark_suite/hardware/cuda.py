"""
CUDA hardware backend implementation.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging

from .base import HardwareBackend

logger = logging.getLogger("benchmark.hardware.cuda")

class CUDABackend(HardwareBackend):
    """CUDA hardware backend."""
    
    name = "cuda"
    
    def __init__(self, device_index: int = 0):
        """
        Initialize CUDA backend.
        
        Args:
            device_index: CUDA device index
        """
        super().__init__()
        self.device_index = device_index
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if CUDA is available.
        
        Returns:
            True if CUDA is available, False otherwise
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        """
        Get CUDA capabilities.
        
        Returns:
            List of capabilities (e.g., ["cuda", "cuda_tensor_cores"])
        """
        capabilities = []
        
        try:
            import torch
            if torch.cuda.is_available():
                capabilities.append("cuda")
                
                # Check for tensor cores (Volta+ GPUs)
                if hasattr(torch.cuda, "get_device_capability"):
                    for device in range(torch.cuda.device_count()):
                        cap_major, cap_minor = torch.cuda.get_device_capability(device)
                        if cap_major >= 7:  # Tensor cores available in Volta+ (7.0+)
                            capabilities.append("cuda_tensor_cores")
                            break
        except ImportError:
            pass
        
        return capabilities
    
    @classmethod
    def get_device_count(cls) -> int:
        """
        Get number of available CUDA devices.
        
        Returns:
            Number of CUDA devices
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
            return 0
        except ImportError:
            return 0
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        Get information about the CUDA hardware.
        
        Returns:
            Dictionary with CUDA information
        """
        info = {"available": cls.is_available()}
        
        if not info["available"]:
            return info
        
        try:
            import torch
            
            device_info = []
            
            for device in range(torch.cuda.device_count()):
                device_data = {
                    "index": device,
                    "name": torch.cuda.get_device_name(device)
                }
                
                if hasattr(torch.cuda, "get_device_capability"):
                    device_data["capability"] = torch.cuda.get_device_capability(device)
                
                if hasattr(torch.cuda, "get_device_properties"):
                    props = torch.cuda.get_device_properties(device)
                    device_data.update({
                        "total_memory": props.total_memory,
                        "multi_processor_count": props.multi_processor_count,
                        "compute_capability": f"{props.major}.{props.minor}"
                    })
                    
                    # Add additional properties if available
                    for prop_name in ["max_threads_per_block", "max_threads_per_multi_processor"]:
                        if hasattr(props, prop_name):
                            device_data[prop_name] = getattr(props, prop_name)
                
                device_info.append(device_data)
            
            info.update({
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "devices": device_info,
                "arch_list": torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else None,
                "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
                "cudnn_enabled": torch.backends.cudnn.enabled if hasattr(torch.backends, "cudnn") else None
            })
            
        except ImportError:
            logger.warning("PyTorch not available for CUDA info")
        
        return info
    
    def initialize(self) -> Any:
        """
        Initialize CUDA for use.
        
        Returns:
            torch.device("cuda")
        """
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                return None
            
            # Reset CUDA for clean measurement
            torch.cuda.empty_cache()
            
            if self.device_index >= torch.cuda.device_count():
                logger.warning(f"Requested CUDA device {self.device_index} not available, using device 0")
                self.device_index = 0
            
            # Set device
            torch.cuda.set_device(self.device_index)
            
            self.initialized = True
            return torch.device(f"cuda:{self.device_index}")
        except ImportError:
            logger.warning("PyTorch not available for CUDA initialization")
            return None
    
    def cleanup(self) -> None:
        """
        Cleanup CUDA resources.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self.initialized = False