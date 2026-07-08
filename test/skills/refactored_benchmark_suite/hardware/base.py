"""
Base class for hardware backends.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("benchmark.hardware")

class HardwareBackend:
    """Base class for hardware backends."""
    
    name = "base"
    
    def __init__(self):
        """Initialize hardware backend."""
        self.initialized = False
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if the hardware is available.
        
        Returns:
            True if available, False otherwise
        """
        return False
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        Get information about the hardware.
        
        Returns:
            Dictionary with hardware information
        """
        return {"available": False}
    
    def initialize(self) -> Any:
        """
        Initialize the hardware for use.
        
        Returns:
            Device object for the hardware
        """
        self.initialized = True
        return None
    
    def cleanup(self) -> None:
        """
        Cleanup hardware resources.
        """
        self.initialized = False
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} Backend"