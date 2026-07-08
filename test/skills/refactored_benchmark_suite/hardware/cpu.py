"""
CPU hardware backend implementation.
"""

from typing import Dict, Any, Optional
import logging

from .base import HardwareBackend

logger = logging.getLogger("benchmark.hardware.cpu")

class CPUBackend(HardwareBackend):
    """CPU hardware backend."""
    
    name = "cpu"
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if CPU is available.
        
        Returns:
            True (CPU is always available)
        """
        return True
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        Get information about the CPU.
        
        Returns:
            Dictionary with CPU information
        """
        info = {"available": True}
        
        try:
            import psutil
            import platform
            
            info.update({
                "processor": platform.processor(),
                "architecture": platform.architecture()[0],
                "cores_physical": psutil.cpu_count(logical=False),
                "cores_logical": psutil.cpu_count(logical=True),
                "system": platform.system(),
                "python_implementation": platform.python_implementation()
            })
            
            # Try to get more detailed CPU info
            try:
                import cpuinfo
                cpu_data = cpuinfo.get_cpu_info()
                info.update({
                    "brand": cpu_data.get("brand_raw", "Unknown"),
                    "hz": cpu_data.get("hz_actual_friendly", "Unknown"),
                    "arch": cpu_data.get("arch", "Unknown"),
                    "bits": cpu_data.get("bits", "Unknown"),
                    "features": cpu_data.get("flags", [])
                })
            except ImportError:
                pass
                
        except ImportError:
            logger.warning("psutil or platform module not available")
        
        return info
    
    def initialize(self) -> Any:
        """
        Initialize CPU for use.
        
        Returns:
            torch.device("cpu")
        """
        try:
            import torch
            self.initialized = True
            return torch.device("cpu")
        except ImportError:
            logger.warning("PyTorch not available for CPU initialization")
            self.initialized = False
            return None
    
    def cleanup(self) -> None:
        """
        Cleanup CPU resources.
        """
        import gc
        gc.collect()
        self.initialized = False