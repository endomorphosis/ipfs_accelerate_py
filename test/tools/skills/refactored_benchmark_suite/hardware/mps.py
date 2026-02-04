"""
MPS (Metal Performance Shaders) hardware backend implementation.
Primarily for Apple Silicon devices.
"""

from typing import Dict, Any, Optional
import logging

from test.tools.skills.refactored_benchmark_suite.hardware.base import HardwareBackend

logger = logging.getLogger("benchmark.hardware.mps")

class MPSBackend(HardwareBackend):
    """MPS (Metal Performance Shaders) hardware backend."""
    
    name = "mps"
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if MPS is available.
        
        Returns:
            True if MPS is available, False otherwise
        """
        try:
            import torch
            return hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available()
        except (ImportError, AttributeError):
            return False
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        Get information about the MPS hardware.
        
        Returns:
            Dictionary with MPS information
        """
        info = {"available": cls.is_available()}
        
        if not info["available"]:
            return info
        
        try:
            import platform
            info.update({
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor()
            })
            
            # Check if we're on Apple Silicon
            is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
            info["is_apple_silicon"] = is_apple_silicon
            
            # Try to get more detailed Apple device info
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                        capture_output=True, text=True, check=True)
                info["cpu_brand"] = result.stdout.strip()
            except (ImportError, subprocess.SubprocessError):
                pass
                
        except ImportError:
            logger.warning("Platform module not available")
        
        return info
    
    def initialize(self) -> Any:
        """
        Initialize MPS for use.
        
        Returns:
            torch.device("mps")
        """
        try:
            import torch
            if not self.is_available():
                logger.warning("MPS not available")
                return None
            
            self.initialized = True
            return torch.device("mps")
        except (ImportError, AttributeError):
            logger.warning("PyTorch MPS support not available")
            return None
    
    def cleanup(self) -> None:
        """
        Cleanup MPS resources.
        """
        try:
            import torch
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass
        
        import gc
        gc.collect()
        self.initialized = False