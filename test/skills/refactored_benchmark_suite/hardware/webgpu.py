"""
WebGPU hardware backend implementation.
"""

from typing import Dict, Any, Optional, List
import logging

from .base import HardwareBackend

logger = logging.getLogger("benchmark.hardware.webgpu")

class WebGPUBackend(HardwareBackend):
    """WebGPU hardware backend."""
    
    name = "webgpu"
    
    def __init__(self, adapter_type: str = "gpu"):
        """
        Initialize WebGPU backend.
        
        Args:
            adapter_type: WebGPU adapter type ("gpu", "cpu")
        """
        super().__init__()
        self.adapter_type = adapter_type
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if WebGPU is available.
        
        Returns:
            True if WebGPU is available, False otherwise
        """
        try:
            import wgpu
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        Get information about the WebGPU runtime.
        
        Returns:
            Dictionary with WebGPU information
        """
        info = {"available": cls.is_available()}
        
        if not info["available"]:
            return info
        
        try:
            import wgpu
            
            # Get wgpu version
            version = getattr(wgpu, "__version__", "unknown")
            info["version"] = version
            
            # Try to get adapter info
            try:
                # Get preferred adapter
                adapter = wgpu.request_adapter(power_preference="high-performance")
                
                if adapter:
                    adapter_info = {
                        "name": adapter.name,
                        "backend": adapter.backend,
                        "features": [str(feature) for feature in adapter.features],
                        "limits": {k: v for k, v in adapter.limits.items()},
                        "is_fallback_adapter": adapter.is_fallback_adapter
                    }
                    
                    info["adapter"] = adapter_info
            except Exception as e:
                logger.warning(f"Error getting WebGPU adapter info: {e}")
            
            # Try to get browser information if in browser context
            try:
                import js
                from js import navigator
                
                user_agent = navigator.userAgent
                info["user_agent"] = user_agent
                
                # Try to extract browser and OS info
                if "Chrome" in user_agent:
                    info["browser"] = "Chrome"
                elif "Firefox" in user_agent:
                    info["browser"] = "Firefox"
                elif "Safari" in user_agent:
                    info["browser"] = "Safari"
                elif "Edge" in user_agent:
                    info["browser"] = "Edge"
                
                if "Windows" in user_agent:
                    info["os"] = "Windows"
                elif "Mac" in user_agent:
                    info["os"] = "macOS"
                elif "Linux" in user_agent:
                    info["os"] = "Linux"
                elif "Android" in user_agent:
                    info["os"] = "Android"
                elif "iPhone" in user_agent or "iPad" in user_agent:
                    info["os"] = "iOS"
            except ImportError:
                # Not in browser context
                pass
                
        except ImportError:
            logger.warning("WebGPU module not available")
        
        return info
    
    def initialize(self) -> Any:
        """
        Initialize WebGPU for use.
        
        Returns:
            WebGPU device
        """
        try:
            import wgpu
            
            # Request adapter
            power_preference = "high-performance" if self.adapter_type == "gpu" else "low-power"
            adapter = wgpu.request_adapter(power_preference=power_preference)
            
            if not adapter:
                logger.warning(f"No WebGPU adapter available with type '{self.adapter_type}'")
                return None
            
            # Create device
            device = adapter.request_device()
            
            self.adapter = adapter
            self.device = device
            self.initialized = True
            
            logger.info(f"Initialized WebGPU device: {adapter.name}, backend: {adapter.backend}")
            return device
            
        except ImportError:
            logger.warning("WebGPU module not available")
            return None
        except Exception as e:
            logger.warning(f"Error initializing WebGPU: {e}")
            return None
    
    def cleanup(self) -> None:
        """
        Cleanup WebGPU resources.
        """
        if hasattr(self, "device") and self.device:
            # Destroy device if there's a method for it
            if hasattr(self.device, "destroy"):
                try:
                    self.device.destroy()
                except:
                    pass
        
        self.initialized = False