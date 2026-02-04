"""
WebNN hardware backend implementation.
"""

from typing import Dict, Any, Optional, List
import logging

from test.tools.skills.refactored_benchmark_suite.hardware.base import HardwareBackend

logger = logging.getLogger("benchmark.hardware.webnn")

class WebNNBackend(HardwareBackend):
    """WebNN hardware backend."""
    
    name = "webnn"
    
    def __init__(self, device_type: str = "gpu"):
        """
        Initialize WebNN backend.
        
        Args:
            device_type: WebNN device type ("gpu", "cpu")
        """
        super().__init__()
        self.device_type = device_type
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if WebNN is available.
        
        Returns:
            True if WebNN is available, False otherwise
        """
        try:
            # Try to import the WebNN module
            import webnn
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_available_backends(cls) -> List[str]:
        """
        Get list of available WebNN backends.
        
        Returns:
            List of available backend names
        """
        backends = []
        
        try:
            import webnn
            
            # Check for GPU backend
            try:
                webnn.create_context(device_type="gpu")
                backends.append("gpu")
            except:
                pass
                
            # Check for CPU backend
            try:
                webnn.create_context(device_type="cpu")
                backends.append("cpu")
            except:
                pass
        except ImportError:
            pass
        
        return backends
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        Get information about the WebNN runtime.
        
        Returns:
            Dictionary with WebNN information
        """
        info = {"available": cls.is_available()}
        
        if not info["available"]:
            return info
        
        try:
            import webnn
            
            # Try to get WebNN version
            version = getattr(webnn, "__version__", "unknown")
            info["version"] = version
            
            # Get available backends
            backends = cls.get_available_backends()
            info["backends"] = backends
            
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
            logger.warning("WebNN module not available")
        
        return info
    
    def initialize(self) -> Any:
        """
        Initialize WebNN for use.
        
        Returns:
            WebNN context
        """
        try:
            import webnn
            
            # Create WebNN context
            try:
                context = webnn.create_context(device_type=self.device_type)
                self.context = context
                self.initialized = True
                return context
            except Exception as e:
                logger.warning(f"Failed to create WebNN context with device '{self.device_type}': {e}")
                
                # Try fallback to CPU if GPU failed
                if self.device_type == "gpu":
                    try:
                        logger.info("Falling back to WebNN CPU backend")
                        self.device_type = "cpu"
                        context = webnn.create_context(device_type="cpu")
                        self.context = context
                        self.initialized = True
                        return context
                    except Exception as e2:
                        logger.warning(f"Failed to create WebNN CPU context: {e2}")
                
                return None
                
        except ImportError:
            logger.warning("WebNN module not available")
            return None
    
    def cleanup(self) -> None:
        """
        Cleanup WebNN resources.
        """
        if hasattr(self, "context") and self.context:
            # Release context if there's a method for it
            if hasattr(self.context, "release"):
                try:
                    self.context.release()
                except:
                    pass
        
        self.initialized = False