"""
Hardware detection module for IPFS Accelerate SDK.

This module provides comprehensive hardware detection capabilities,
building on the existing HardwareDetector implementation while adding
enhanced features and a cleaner API.
"""

import os
import importlib
import logging
import platform
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ipfs_accelerate.hardware")
:
class HardwareDetector:
    """
    Enhanced hardware detection for IPFS Accelerate SDK.
    
    This class provides enhanced hardware detection capabilities,
    building on the existing implementation while adding new features
    and a cleaner API.
    """
    :
    def __init__(self, config_instance=None):
        """
        Initialize the hardware detector.
        
        Args:
            config_instance: Configuration instance (optional)
            """
            self.config = config_instance
            self._details = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            self._available_hardware = []]],,,],
            self._browser_details = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            self._simulation_status = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            self._hardware_capabilities = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Try to import the legacy implementation for compatibility
        try:
            from ipfs_accelerate_impl import HardwareDetector as LegacyDetector
            self._legacy_detector = LegacyDetector(config_instance)
            logger.info("Legacy hardware detection loaded for compatibility")
        except ImportError:
            self._legacy_detector = None
            logger.info("Legacy hardware detection not available")
            
        # Try to load hardware_detection module if available:
        self._hardware_detection_module = None:
        try:
            if importlib.util.find_spec("hardware_detection") is not None:
                self._hardware_detection_module = importlib.import_module("hardware_detection")
                logger.info("Hardware detection module loaded")
        except ImportError:
            logger.info("Hardware detection module not available, using built-in detection")
        
        # Load the fixed_web_platform module if available: (for WebNN/WebGPU)
        self._web_platform_module = None:
        try:
            if importlib.util.find_spec("fixed_web_platform.browser_capability_detection") is not None:
                self._web_platform_module = importlib.import_module("fixed_web_platform.browser_capability_detection")
                logger.info("Web platform module loaded")
        except ImportError:
            logger.info("Web platform module not available")
            
        # Detect available hardware on initialization
            self.detect_all()
        
            def detect_all(self) -> Dict[]]],,,str, Any]:,,,,,
            """
            Detect all available hardware platforms.
        
        Returns:
            Dictionary with hardware details keyed by platform name.
            """
        # If legacy detector is available, use it for compatibility
        if self._legacy_detector:
            legacy_hardware = self._legacy_detector.detect_hardware()
            legacy_details = self._legacy_detector.get_hardware_details()
            self._available_hardware = legacy_hardware
            self._details = legacy_details
            
            # Convert legacy format to enhanced format
            return self._convert_legacy_to_enhanced(legacy_details)
        
        # Otherwise, use built-in detection
            return self._detect_hardware_enhanced()
    
            def _detect_hardware_enhanced(self) -> Dict[]]],,,str, Any]:,,,,,
            """
            Enhanced hardware detection implementation.
        
        Returns:
            Dictionary with hardware details keyed by platform name.
            """
            available = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # CPU is always available
            available[]]],,,"cpu"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
            "available": True,
            "name": platform.processor() or "Unknown CPU",
            "platform": platform.platform(),
            "simulation_enabled": False,
            "performance_score": 1.0,
            "recommended_batch_size": 32,
            "recommended_models": []]],,,"bert", "t5", "vit", "clip"],
            "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 15.0, "t5-small": 25.0, "vit-base": 30.0},
            "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 67.0, "t5-small": 40.0, "vit-base": 33.0},
            "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 500, "t5-small": 750, "vit-base": 600}
            }
            }
        
        # Try to use external hardware_detection module if available:
        if self._hardware_detection_module:
            try:
                detector = self._hardware_detection_module.HardwareDetector()
                hardware_info = detector.detect_all()
                
                # Map hardware detection results to our format
                if hardware_info:
                    for hw_type, hw_data in hardware_info.items():
                        if hw_data.get("available"):
                            available[]]],,,hw_type] = hw_data
                            ,
                            logger.info(f"Detected hardware using external module: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}list(available.keys())}")
                        return available
            except Exception as e:
                logger.warning(f"Error using external hardware detection: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
                # Fall back to built-in detection
        
        # Built-in hardware detection (similar to legacy but with enhanced metrics)
        try:
            # Check CUDA
            try:
                if importlib.util.find_spec("torch") is not None:
                    import torch
                    if torch.cuda.is_available():
                        cuda_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "available": True,
                        "device_count": torch.cuda.device_count(),
                            "name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown CUDA Device",:
                                "simulation_enabled": False,
                                "performance_score": 5.0,
                                "recommended_batch_size": 64,
                                "recommended_models": []]],,,"bert", "t5", "vit", "clip", "whisper", "llama"],
                                "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                                "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 3.0, "t5-small": 5.0, "vit-base": 6.0},
                                "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 333.0, "t5-small": 200.0, "vit-base": 167.0},
                                "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 600, "t5-small": 850, "vit-base": 700}
                                }
                                }
                                available[]]],,,"cuda"] = cuda_info,
            except ImportError:
                                pass
            
            # Check ROCm (for AMD GPUs)
            try:
                if (importlib.util.find_spec("torch") is not None and 
                    hasattr(importlib.import_module("torch"), "hip") and:
                    importlib.import_module("torch").hip.is_available()):
                        rocm_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "available": True,
                        "device_count": importlib.import_module("torch").hip.device_count(),
                        "name": "AMD ROCm GPU",
                        "simulation_enabled": False,
                        "performance_score": 4.5,
                        "recommended_batch_size": 48,
                        "recommended_models": []]],,,"bert", "t5", "vit", "clip"],
                        "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 3.5, "t5-small": 5.5, "vit-base": 6.5},
                        "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 280.0, "t5-small": 180.0, "vit-base": 150.0},
                        "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 620, "t5-small": 870, "vit-base": 720}
                        }
                        }
                        available[]]],,,"rocm"] = rocm_info,
            except (ImportError, AttributeError):
                        pass
            
            # Check MPS (for Apple Silicon)
            try:
                if (importlib.util.find_spec("torch") is not None and 
                hasattr(importlib.import_module("torch"), "backends") and
                    hasattr(importlib.import_module("torch").backends, "mps") and:
                    importlib.import_module("torch").backends.mps.is_available()):
                        mps_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "available": True,
                        "name": "Apple Metal Performance Shaders",
                        "simulation_enabled": False,
                        "performance_score": 3.5,
                        "recommended_batch_size": 32,
                        "recommended_models": []]],,,"bert", "vit", "clip", "whisper"],
                        "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 5.0, "vit-base": 8.0},
                        "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 200.0, "vit-base": 120.0},
                        "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 550, "vit-base": 650}
                        }
                        }
                        available[]]],,,"mps"] = mps_info,
            except (ImportError, AttributeError):
                        pass
            
            # Check OpenVINO
            try:
                if importlib.util.find_spec("openvino") is not None:
                    openvino_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "available": True,
                    "name": "Intel OpenVINO",
                    "simulation_enabled": False,
                    "performance_score": 3.0,
                    "recommended_batch_size": 32,
                    "recommended_models": []]],,,"bert", "t5", "vit", "clip"],
                    "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 5.5, "vit-base": 9.0},
                    "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 180.0, "vit-base": 110.0},
                    "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 400, "vit-base": 500}
                    }
                    }
                    available[]]],,,"openvino"] = openvino_info,
            except ImportError:
                    pass
            
            # Check Qualcomm QNN (usually needs simulation unless on device)
                    qualcomm_simulation = True
            if os.environ.get("QNN_SDK_ROOT") is not None:
                qualcomm_simulation = False
            
                qualcomm_info = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "available": True,  # Always available through simulation
                "name": "Qualcomm Neural Network",
                "simulation_enabled": qualcomm_simulation,
                "performance_score": 2.5,
                "recommended_batch_size": 16,
                "recommended_models": []]],,,"bert", "t5", "vit", "whisper"],
                "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 6.0, "vit-base": 10.0},
                "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 160.0, "vit-base": 100.0},
                "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 300, "vit-base": 400}
                }
                }
                available[]]],,,"qualcomm"] = qualcomm_info
                ,
            # WebNN and WebGPU detection using fixed_web_platform
            if self._web_platform_module:
                try:
                    browser_detector = self._web_platform_module.BrowserCapabilityDetector()
                    browser_capabilities = browser_detector.detect_capabilities()
                    
                    # Map browser capabilities to hardware format
                    if browser_capabilities.get("webgpu_support", False):
                        available[]]],,,"webgpu"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},,,
                        "available": True,
                        "name": "Web GPU API",
                        "simulation_enabled": not browser_capabilities.get("real_webgpu", False),
                        "browsers": browser_capabilities.get("webgpu_browsers", []]],,,],),
                        "performance_score": 3.5,
                        "recommended_batch_size": 16,
                        "recommended_models": []]],,,"bert", "vit", "clip"],
                        "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 8.0, "vit-base": 12.0},
                        "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 125.0, "vit-base": 83.0},
                        "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 250, "vit-base": 350}
                        }
                        }
                    
                    if browser_capabilities.get("webnn_support", False):
                        available[]]],,,"webnn"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},,,
                        "available": True,
                        "name": "Web Neural Network API",
                        "simulation_enabled": not browser_capabilities.get("real_webnn", False),
                        "browsers": browser_capabilities.get("webnn_browsers", []]],,,],),
                        "performance_score": 3.0,
                        "recommended_batch_size": 8,
                        "recommended_models": []]],,,"bert", "vit"],
                        "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 10.0, "vit-base": 15.0},
                        "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 100.0, "vit-base": 67.0},
                        "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 200, "vit-base": 300}
                        }
                        }
                except Exception as e:
                    logger.warning(f"Error detecting browser capabilities: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
                    
                    # Fallback web platform detection
                    webgpu_simulation = not bool(os.environ.get("USE_BROWSER_AUTOMATION"))
                    available[]]],,,"webgpu"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},,,
                    "available": True,
                    "name": "Web GPU API",
                    "simulation_enabled": webgpu_simulation,
                    "browsers": []]],,,"chrome", "firefox", "edge", "safari"],
                    "performance_score": 3.5,
                    "recommended_batch_size": 16,
                    "recommended_models": []]],,,"bert", "vit", "clip"],
                    "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 8.0, "vit-base": 12.0},
                    "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 125.0, "vit-base": 83.0},
                    "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 250, "vit-base": 350}
                    }
                    }
                    
                    webnn_simulation = not bool(os.environ.get("USE_BROWSER_AUTOMATION"))
                    available[]]],,,"webnn"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},,,
                    "available": True,
                    "name": "Web Neural Network API",
                    "simulation_enabled": webnn_simulation,
                    "browsers": []]],,,"edge", "chrome", "safari"],
                    "performance_score": 3.0,
                    "recommended_batch_size": 8,
                    "recommended_models": []]],,,"bert", "vit"],
                    "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 10.0, "vit-base": 15.0},
                    "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 100.0, "vit-base": 67.0},
                    "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 200, "vit-base": 300}
                    }
                    }
            else:
                # Basic web platform detection without module
                webgpu_simulation = not bool(os.environ.get("USE_BROWSER_AUTOMATION"))
                available[]]],,,"webgpu"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},,,
                "available": True,
                "name": "Web GPU API",
                "simulation_enabled": webgpu_simulation,
                "browsers": []]],,,"chrome", "firefox", "edge", "safari"],
                "performance_score": 3.5,
                "recommended_batch_size": 16,
                "recommended_models": []]],,,"bert", "vit", "clip"],
                "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 8.0, "vit-base": 12.0},
                "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 125.0, "vit-base": 83.0},
                "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 250, "vit-base": 350}
                }
                }
                
                webnn_simulation = not bool(os.environ.get("USE_BROWSER_AUTOMATION"))
                available[]]],,,"webnn"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},,,
                "available": True,
                "name": "Web Neural Network API",
                "simulation_enabled": webnn_simulation,
                "browsers": []]],,,"edge", "chrome", "safari"],
                "performance_score": 3.0,
                "recommended_batch_size": 8,
                "recommended_models": []]],,,"bert", "vit"],
                "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 10.0, "vit-base": 15.0},
                "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 100.0, "vit-base": 67.0},
                "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 200, "vit-base": 300}
                }
                }
            
            # Store detected hardware
                self._details = available
                self._available_hardware = list(available.keys())
                logger.info(f"Detected hardware using built-in detection: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}self._available_hardware}")
                    return available
            
        except Exception as e:
            logger.error(f"Error in hardware detection: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
            # Always return CPU as fallback
                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"cpu": available[]]],,,"cpu"]}
                    ,
                    def _convert_legacy_to_enhanced(self, legacy_details: Dict[]]],,,str, Any]) -> Dict[]]],,,str, Any]:,,,,,,
                    """
                    Convert legacy hardware details to enhanced format.
        
        Args:
            legacy_details: Hardware details in legacy format.
            
        Returns:
            Hardware details in enhanced format.
            """
            enhanced = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        for hw_type, hw_data in legacy_details.items():
            enhanced_data = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "available": hw_data.get("available", False),
            "name": hw_data.get("name", f"Unknown {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}hw_type.upper()} Device"),
            "simulation_enabled": hw_data.get("simulation_enabled", False),
            }
            
            # Add enhanced metrics based on hardware type
            if hw_type == "cuda":
                enhanced_data.update({}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "performance_score": 5.0,
                "recommended_batch_size": 64,
                "recommended_models": []]],,,"bert", "t5", "vit", "clip", "whisper", "llama"],
                "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 3.0, "t5-small": 5.0, "vit-base": 6.0},
                "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 333.0, "t5-small": 200.0, "vit-base": 167.0},
                "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 600, "t5-small": 850, "vit-base": 700}
                }
                })
            elif hw_type == "cpu":
                enhanced_data.update({}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "performance_score": 1.0,
                "recommended_batch_size": 32,
                "recommended_models": []]],,,"bert", "t5", "vit", "clip"],
                "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 15.0, "t5-small": 25.0, "vit-base": 30.0},
                "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 67.0, "t5-small": 40.0, "vit-base": 33.0},
                "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 500, "t5-small": 750, "vit-base": 600}
                }
                })
            elif hw_type == "webgpu":
                enhanced_data.update({}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "performance_score": 3.5,
                "recommended_batch_size": 16,
                "recommended_models": []]],,,"bert", "vit", "clip"],
                "performance_metrics": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "latency_ms": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 8.0, "vit-base": 12.0},
                "throughput_items_per_sec": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 125.0, "vit-base": 83.0},
                "memory_usage_mb": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"bert-base-uncased": 250, "vit-base": 350}
                }
                })
            # Add other hardware types as needed
            
                enhanced[]]],,,hw_type] = enhanced_data
                ,
                return enhanced
    
                def get_hardware_details(self, hardware_type: str = None) -> Dict[]]],,,str, Any]:,,,,,
                """
                Get details about available hardware platforms.
        
        Args:
            hardware_type: Specific hardware type to get details for, or None for all.
            
        Returns:
            Dictionary with hardware details.
            """
        if not self._details:
            self.detect_all()
            
        if hardware_type:
            return self._details.get(hardware_type, {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
        else:
            return self._details
    
    def is_real_hardware(self, hardware_type: str) -> bool:
        """
        Check if real hardware is available (not simulation).
        :
        Args:
            hardware_type: Hardware type to check.
            
        Returns:
            True if real hardware is available, False if simulation.
        """:
        if not self._details:
            self.detect_all()
            
            details = self._details.get(hardware_type, {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}})
            return details.get("available", False) and not details.get("simulation_enabled", True)
    
    def get_optimal_hardware(self, model_name: str, model_type: str = None, batch_size: int = 1) -> str:
        """
        Get the optimal hardware platform for a model.
        
        Args:
            model_name: Name of the model.
            model_type: Type of model (text, vision, audio, multimodal).
            batch_size: Batch size to use.
            
        Returns:
            Hardware platform name.
            """
        if not self._details:
            self.detect_all()
            
        # If legacy detector is available, delegate to it for compatibility
        if self._legacy_detector:
            return self._legacy_detector.get_optimal_hardware(model_name, model_type)
            
        # Determine model type based on model name if not provided:
        if not model_type:
            model_type = "text"
            if any(x in model_name.lower() for x in []]],,,"whisper", "wav2vec", "clap"]):,
            model_type = "audio"
            elif any(x in model_name.lower() for x in []]],,,"vit", "clip", "detr", "image"]):,
            model_type = "vision"
            elif any(x in model_name.lower() for x in []]],,,"llava", "xclip"]):,
            model_type = "multimodal"
        
        # Hardware ranking by model type and batch size (best to worst)
            hardware_ranking = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "text": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "small": []]],,,"cuda", "rocm", "mps", "openvino", "qualcomm", "webnn", "webgpu", "cpu"],
            "medium": []]],,,"cuda", "rocm", "mps", "openvino", "qualcomm", "webgpu", "cpu", "webnn"],
            "large": []]],,,"cuda", "rocm", "mps", "openvino", "cpu", "qualcomm", "webgpu", "webnn"],
            },
            "vision": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "small": []]],,,"cuda", "rocm", "mps", "openvino", "webgpu", "qualcomm", "webnn", "cpu"],
            "medium": []]],,,"cuda", "rocm", "mps", "webgpu", "openvino", "qualcomm", "cpu", "webnn"],
            "large": []]],,,"cuda", "rocm", "mps", "openvino", "webgpu", "cpu", "qualcomm", "webnn"],
            },
            "audio": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "small": []]],,,"cuda", "qualcomm", "rocm", "mps", "webgpu", "openvino", "webnn", "cpu"],
            "medium": []]],,,"cuda", "qualcomm", "rocm", "mps", "webgpu", "openvino", "cpu", "webnn"],
            "large": []]],,,"cuda", "rocm", "qualcomm", "mps", "openvino", "cpu", "webgpu", "webnn"],
            },
            "multimodal": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "small": []]],,,"cuda", "rocm", "mps", "openvino", "webgpu", "qualcomm", "webnn", "cpu"],
            "medium": []]],,,"cuda", "rocm", "mps", "openvino", "webgpu", "cpu", "qualcomm", "webnn"],,
            "large": []]],,,"cuda", "rocm", "mps", "openvino", "cpu", "webgpu", "qualcomm", "webnn"],
            }
            }
        
        # Determine batch size category
        if batch_size <= 4:
            size_category = "small"
        elif batch_size <= 32:
            size_category = "medium"
        else:
            size_category = "large"
            
        # Special case for audio models on Firefox WebGPU
        if model_type == "audio" and self.get_browser_details().get("firefox", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get("available", False):
            # Check if Firefox has WebGPU support
            firefox_webgpu = self.get_browser_details().get("firefox", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).get("webgpu_support", False):
            if firefox_webgpu:
                # Firefox has optimized compute shaders for audio models
                current_ranking = hardware_ranking[]]],,,model_type][]]],,,size_category],
                webgpu_index = current_ranking.index("webgpu")
                # Move WebGPU to the front for small batch sizes
                if size_category == "small":
                    new_ranking = []]],,,"webgpu"] + current_ranking[]]],,,:webgpu_index] + current_ranking[]]],,,webgpu_index+1:],
                    hardware_ranking[]]],,,model_type][]]],,,size_category], = new_ranking
        
        # Get optimal hardware from ranking
                    for hw in hardware_ranking.get(model_type, hardware_ranking[]]],,,"text"]).get(size_category, []]],,,"cuda", "cpu"]):,
            if hw in self._available_hardware:
                    return hw
        
        # Fallback to CPU
                return "cpu"
    
                def get_browser_details(self, update: bool = False) -> Dict[]]],,,str, Any]:,,,,,
                """
                Get details about available browsers for WebNN/WebGPU.
        
        Args:
            update: Whether to update browser details.
            
        Returns:
            Dictionary with browser details.
            """
        if update or not self._browser_details:
            # If legacy detector is available, delegate to it for compatibility
            if self._legacy_detector:
                self._browser_details = self._legacy_detector.get_browser_details(update)
            else:
                self._detect_browsers()
                return self._browser_details
    
                def _detect_browsers(self) -> Dict[]]],,,str, Any]:,,,,,
                """
                Detect available browsers for WebNN/WebGPU.
        
        Returns:
            Dictionary with browser details.
            """
        # If web platform module is available, use it for detection
        if self._web_platform_module:
            try:
                browser_detector = self._web_platform_module.BrowserCapabilityDetector()
                browser_capabilities = browser_detector.detect_capabilities()
                
                # Convert to browser details format
                browsers = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                for browser_name, browser_data in browser_capabilities.get("browsers", {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}).items():
                    browsers[]]],,,browser_name] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
                    "available": browser_data.get("available", False),
                    "path": browser_data.get("path", ""),
                    "webgpu_support": browser_data.get("webgpu_support", False),
                    "webnn_support": browser_data.get("webnn_support", False),
                    "name": browser_data.get("name", browser_name.capitalize())
                    }
                
                    self._browser_details = browsers
                return browsers
            except Exception as e:
                logger.warning(f"Error using web platform module for browser detection: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
        
        # Fall back to basic detection similar to legacy implementation
                browsers = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Check for browsers
        try:
            # Check Chrome
            chrome_path = self._find_browser_path("chrome")
            if chrome_path:
                browsers[]]],,,"chrome"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
                "available": True,
                "path": chrome_path,
                "webgpu_support": True,
                "webnn_support": True,
                "name": "Google Chrome"
                }
            
            # Check Firefox
                firefox_path = self._find_browser_path("firefox")
            if firefox_path:
                browsers[]]],,,"firefox"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
                "available": True,
                "path": firefox_path,
                "webgpu_support": True,
                "webnn_support": False,  # Firefox support for WebNN is limited
                "name": "Mozilla Firefox"
                }
            
            # Check Edge
                edge_path = self._find_browser_path("edge")
            if edge_path:
                browsers[]]],,,"edge"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
                "available": True,
                "path": edge_path,
                "webgpu_support": True,
                "webnn_support": True,
                "name": "Microsoft Edge"
                }
            
            # Check Safari (macOS only)
            if platform.system() == "Darwin":
                safari_path = "/Applications/Safari.app/Contents/MacOS/Safari"
                if os.path.exists(safari_path):
                    browsers[]]],,,"safari"] = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},
                    "available": True,
                    "path": safari_path,
                    "webgpu_support": True,
                    "webnn_support": True,
                    "name": "Apple Safari"
                    }
        
        except Exception as e:
            logger.error(f"Error detecting browsers: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
        
            self._browser_details = browsers
                    return browsers
    
                    def _find_browser_path(self, browser_name: str) -> Optional[]]],,,str]:,
                    """Find browser executable path."""
                    common_paths = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "chrome": []]],,,
                    "/usr/bin/google-chrome",
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
                    ],
                    "firefox": []]],,,
                    "/usr/bin/firefox",
                    "/Applications/Firefox.app/Contents/MacOS/firefox",
                    "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
                    "C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe"
                    ],
                    "edge": []]],,,
                    "/usr/bin/microsoft-edge",
                    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
                    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
                    ]
                    }
        
        for path in common_paths.get(browser_name, []]],,,],):
            if os.path.exists(path):
            return path
        
                    return None