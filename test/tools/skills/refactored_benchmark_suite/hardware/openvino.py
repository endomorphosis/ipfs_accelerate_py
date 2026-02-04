"""
OpenVINO hardware backend implementation.
"""

from typing import Dict, Any, Optional, List
import logging

from test.tools.skills.refactored_benchmark_suite.hardware.base import HardwareBackend

logger = logging.getLogger("benchmark.hardware.openvino")

class OpenVINOBackend(HardwareBackend):
    """OpenVINO hardware backend."""
    
    name = "openvino"
    
    def __init__(self, device: str = "CPU"):
        """
        Initialize OpenVINO backend.
        
        Args:
            device: OpenVINO device ("CPU", "GPU", "VPU", etc.)
        """
        super().__init__()
        self.device = device
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if OpenVINO is available.
        
        Returns:
            True if OpenVINO is available, False otherwise
        """
        try:
            import openvino
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_available_devices(cls) -> List[str]:
        """
        Get list of available OpenVINO devices.
        
        Returns:
            List of available device names
        """
        try:
            import openvino as ov
            from openvino.runtime import Core
            
            core = Core()
            return core.available_devices
        except ImportError:
            return []
        except Exception as e:
            logger.warning(f"Error getting OpenVINO devices: {e}")
            return []
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        Get information about the OpenVINO runtime.
        
        Returns:
            Dictionary with OpenVINO information
        """
        info = {"available": cls.is_available()}
        
        if not info["available"]:
            return info
        
        try:
            import openvino as ov
            from openvino.runtime import Core
            
            info["version"] = ov.__version__
            
            try:
                core = Core()
                devices = core.available_devices
                
                device_info = []
                for device_name in devices:
                    device_data = {
                        "name": device_name,
                        "version": core.get_property(device_name, "FULL_DEVICE_NAME")
                    }
                    
                    # Get device-specific properties
                    if device_name == "CPU":
                        device_data["optimization_capabilities"] = core.get_property(device_name, "OPTIMIZATION_CAPABILITIES")
                    elif device_name == "GPU":
                        device_data["optimization_capabilities"] = core.get_property(device_name, "OPTIMIZATION_CAPABILITIES")
                        device_data["device_total_mem_size"] = core.get_property(device_name, "DEVICE_TOTAL_MEM_SIZE")
                    
                    device_info.append(device_data)
                
                info.update({
                    "devices": devices,
                    "device_info": device_info,
                    "cache_dir": core.get_property("CPU", "CACHE_DIR") if "CPU" in devices else None
                })
            except Exception as e:
                logger.warning(f"Error getting OpenVINO core info: {e}")
                
        except ImportError:
            logger.warning("OpenVINO module not available")
        
        return info
    
    def initialize(self) -> Any:
        """
        Initialize OpenVINO for use.
        
        Returns:
            OpenVINO Core and device string
        """
        try:
            import openvino as ov
            from openvino.runtime import Core
            
            core = Core()
            available_devices = core.available_devices
            
            if not available_devices:
                logger.warning("No OpenVINO devices available")
                return None
            
            if self.device not in available_devices:
                logger.warning(f"Requested OpenVINO device '{self.device}' not available, using {available_devices[0]}")
                self.device = available_devices[0]
            
            logger.info(f"Initialized OpenVINO with device: {self.device}")
            
            self.core = core
            self.initialized = True
            return {"core": core, "device": self.device}
        except ImportError:
            logger.warning("OpenVINO module not available")
            return None
        except Exception as e:
            logger.warning(f"Error initializing OpenVINO: {e}")
            return None
    
    def cleanup(self) -> None:
        """
        Cleanup OpenVINO resources.
        """
        self.initialized = False
        # No specific cleanup needed for OpenVINO Core