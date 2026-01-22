#!/usr/bin/env python3
"""
Centralized Hardware Detection Module

This module provides a standardized, centralized interface for hardware detection
across all test generators and test files in the project. It ensures consistency
in hardware detection and hardware-aware optimizations.

Key features:
    - Reliable detection of CPU, CUDA, ROCm, MPS, OpenVINO, Qualcomm, WebNN, and WebGPU
    - Web platform optimization management for WebGPU compute shaders, parallel loading, and shader precompilation
    - Browser detection for platform-specific optimizations ()))))))))especially Firefox WebGPU)
    - Hardware capability reporting
    - Integration with test generators

Usage:
    from generators.hardware.hardware_detection import HardwareManager
    
    # Get hardware capabilities
    hw_manager = HardwareManager())))))))))
    capabilities = hw_manager.get_capabilities())))))))))
    
    # Check specific hardware
    if hw_manager.has_cuda:
        # CUDA-specific code
        
    # Get web platform optimizations
        audio_optimizations = hw_manager.get_web_optimizations()))))))))"audio", "WebGPU")
        """

        import os
        import importlib.util
        import logging
        from typing import Dict, List, Any, Tuple, Set, Optional, Union

# Configure logging
        logging.basicConfig()))))))))level=logging.INFO, format='%()))))))))asctime)s - %()))))))))levelname)s - %()))))))))message)s')
        logger = logging.getLogger()))))))))"hardware_detection")

class HardwareManager:
    """
    Centralized manager for hardware detection and capabilities.
    Provides a unified interface for accessing hardware information.
    """
    
    def __init__()))))))))self):
        """Initialize hardware detection."""
        # Try to import torch ()))))))))for CUDA/ROCm/MPS)
        try:
            import torch
            self.torch = torch
            self.has_torch = True
        except ImportError:
            from unittest.mock import MagicMock
            self.torch = MagicMock())))))))))
            self.has_torch = False
            logger.warning()))))))))"torch not available, using mock")
            
        # Initialize hardware capability flags
            self.has_cuda = False
            self.has_rocm = False
            self.has_mps = False
            self.has_openvino = False
            self.has_qualcomm = False
            self.has_samsung_npu = False
            self.has_webnn = False
            self.has_webgpu = False
        
        # Detect hardware
            self._detect_hardware())))))))))
        
        # Store complete capabilities
            self.capabilities = self._check_hardware())))))))))
        
        # Web optimization information
            self.browser_info = self._detect_browser())))))))))
    
    def _detect_hardware()))))))))self):
        """Detect available hardware."""
        # CUDA detection
        if self.has_torch:
            self.has_cuda = self.torch.cuda.is_available())))))))))
            
            # ROCm detection
            if self.has_cuda and hasattr()))))))))self.torch, '_C') and hasattr()))))))))self.torch._C, '_rocm_version'):
                self.has_rocm = True
            elif 'ROCM_HOME' in os.environ:
                self.has_rocm = True
            
            # Apple MPS detection
            if hasattr()))))))))self.torch, "mps") and hasattr()))))))))self.torch.mps, "is_available"):
                self.has_mps = self.torch.mps.is_available())))))))))
        
        # OpenVINO detection
                self.has_openvino = importlib.util.find_spec()))))))))"openvino") is not None
        
        # Qualcomm detection ()))))))))enhanced to properly check for simulation vs real)
                self.has_qualcomm = False
                self.qnn_simulation_mode = False
        try:
            # Try to import our enhanced QNN support module
            try:
                # First try our standardized module
                from hardware_detection.qnn_support import QNN_AVAILABLE, QNN_SIMULATION_MODE
                self.has_qualcomm = QNN_AVAILABLE
                self.qnn_simulation_mode = QNN_SIMULATION_MODE
                logger.info()))))))))f"QNN detection: Available={}}}}}}}}QNN_AVAILABLE}, Simulation={}}}}}}}}QNN_SIMULATION_MODE}")
            except ImportError:
                try:
                    # Try alternative path
                    import sys
                    sys.path.append()))))))))'/home/barberb/ipfs_accelerate_py/test')
                    from hardware_detection.qnn_support import QNN_AVAILABLE, QNN_SIMULATION_MODE
                    self.has_qualcomm = QNN_AVAILABLE
                    self.qnn_simulation_mode = QNN_SIMULATION_MODE
                    logger.info()))))))))f"QNN detection ()))))))))alt path): Available={}}}}}}}}QNN_AVAILABLE}, Simulation={}}}}}}}}QNN_SIMULATION_MODE}")
                except ImportError:
                    # Fall back to basic detection
                    self.has_qualcomm = ()))))))))
                    importlib.util.find_spec()))))))))"qnn_wrapper") is not None or
                    importlib.util.find_spec()))))))))"qti") is not None or
                    "QUALCOMM_SDK" in os.environ
                    )
                    self.qnn_simulation_mode = os.environ.get()))))))))"QNN_SIMULATION_MODE", "0").lower()))))))))) in ()))))))))"1", "true", "yes")
                    logger.warning()))))))))"Using basic QNN detection ()))))))))qnn_support module not found)")
        except ()))))))))ValueError, ImportError, Exception) as e:
            # Handle errors if modules are not properly installed:
            if "QUALCOMM_SDK" in os.environ:
                self.has_qualcomm = True
                self.qnn_simulation_mode = os.environ.get()))))))))"QNN_SIMULATION_MODE", "0").lower()))))))))) in ()))))))))"1", "true", "yes")
                logger.warning()))))))))f"QNN detection error: {}}}}}}}}str()))))))))e)}")
        
        # Samsung NPU detection (for Exynos devices)
        self.has_samsung_npu = False
        self.samsung_npu_simulation_mode = False
        try:
            # Try to import our Samsung NPU support module
            try:
                # Import SamsungDetector if available
                from samsung_support import SamsungDetector
                
                # Initialize detector and detect hardware
                samsung_detector = SamsungDetector()
                samsung_chipset = samsung_detector.detect_samsung_hardware()
                
                # Set flags based on detection results
                self.has_samsung_npu = samsung_chipset is not None
                self.samsung_npu_simulation_mode = "TEST_SAMSUNG_CHIPSET" in os.environ
                
                if self.has_samsung_npu:
                    logger.info()))))))))f"Samsung NPU detected: {}}}}}}}}samsung_chipset.name}")
                    if self.samsung_npu_simulation_mode:
                        logger.warning()))))))))"Samsung NPU running in SIMULATION mode")
                    self.samsung_chipset = samsung_chipset
            except ImportError:
                # Fall back to basic environment variable detection
                self.has_samsung_npu = (
                    "SAMSUNG_SDK_PATH" in os.environ or
                    "TEST_SAMSUNG_CHIPSET" in os.environ or
                    os.environ.get()))))))))"SAMSUNG_NPU_AVAILABLE", "0") == "1"
                )
                self.samsung_npu_simulation_mode = (
                    "TEST_SAMSUNG_CHIPSET" in os.environ or
                    os.environ.get()))))))))"SAMSUNG_NPU_SIMULATION", "0") == "1"
                )
                logger.warning()))))))))"Using basic Samsung NPU detection (samsung_support module not found)")
        except Exception as e:
            # Handle errors if modules are not properly installed
            logger.warning()))))))))f"Samsung NPU detection error: {}}}}}}}}str()))))))))e)}")
            # Still allow simulation via environment variables
            if "TEST_SAMSUNG_CHIPSET" in os.environ:
                self.has_samsung_npu = True
                self.samsung_npu_simulation_mode = True
        
        # WebNN detection with proper simulation tracking
                webnn_library_available = ()))))))))
                importlib.util.find_spec()))))))))"webnn") is not None or
                importlib.util.find_spec()))))))))"webnn_js") is not None
                )
        
        # Track simulation status separately
                self.webnn_simulation_mode = os.environ.get()))))))))"WEBNN_SIMULATION", "0").lower()))))))))) in ()))))))))"1", "true", "yes")
                webnn_override = os.environ.get()))))))))"WEBNN_AVAILABLE", "0").lower()))))))))) in ()))))))))"1", "true", "yes")
        
        if webnn_library_available:
            self.has_webnn = True
            logger.info()))))))))"WebNN library detected")
        elif self.webnn_simulation_mode:
            # Allow simulation if explicitly requested
            self.has_webnn = True
            logger.warning()))))))))"WebNN SIMULATION mode enabled via environment variable"):
        elif webnn_override:
            # Allow override if explicitly requested, but mark as simulation
            self.has_webnn = True
            self.webnn_simulation_mode = True
            logger.warning()))))))))"WebNN availability forced by environment variable ()))))))))treated as simulation)"):
        else:
            self.has_webnn = False
            self.webnn_simulation_mode = False
        
        # WebGPU detection with proper simulation tracking
            webgpu_library_available = ()))))))))
            importlib.util.find_spec()))))))))"webgpu") is not None or
            importlib.util.find_spec()))))))))"wgpu") is not None
            )
        
        # Track simulation status separately
            self.webgpu_simulation_mode = os.environ.get()))))))))"WEBGPU_SIMULATION", "0").lower()))))))))) in ()))))))))"1", "true", "yes")
            webgpu_override = os.environ.get()))))))))"WEBGPU_AVAILABLE", "0").lower()))))))))) in ()))))))))"1", "true", "yes")
        
        if webgpu_library_available:
            self.has_webgpu = True
            logger.info()))))))))"WebGPU library detected")
        elif self.webgpu_simulation_mode:
            # Allow simulation if explicitly requested
            self.has_webgpu = True
            logger.warning()))))))))"WebGPU SIMULATION mode enabled via environment variable"):
        elif webgpu_override:
            # Allow override if explicitly requested, but mark as simulation
            self.has_webgpu = True
            self.webgpu_simulation_mode = True
            logger.warning()))))))))"WebGPU availability forced by environment variable ()))))))))treated as simulation)"):
        else:
            self.has_webgpu = False
            self.webgpu_simulation_mode = False
    
            def _check_hardware()))))))))self) -> Dict[str, Any]:,,,,,
            """Generate comprehensive hardware capability information."""
            capabilities = {}}}}}}}}
            "cpu": True,
            "cuda": False,
            "cuda_version": None,
            "cuda_devices": 0,
            "mps": False,
            "openvino": False,
            "qualcomm": False,
            "qualcomm_simulation": False,
            "mediatek": False,
            "mediatek_simulation": False,
            "rocm": False,
            "webnn": False,
            "webnn_simulation": False,
            "webgpu": False,
            "webgpu_simulation": False
            }
        
        # CUDA capabilities
        if self.has_torch and self.has_cuda:
            capabilities["cuda"] = True,,
            capabilities["cuda_devices"] = self.torch.cuda.device_count()))))))))),
            capabilities["cuda_version"] = self.torch.version.cuda
            ,
        # MPS capabilities ()))))))))Apple Silicon)
            capabilities["mps"] = self.has_mps
            ,
        # OpenVINO capabilities
            capabilities["openvino"] = self.has_openvino
            ,
        # Qualcomm capabilities with simulation flag
            capabilities["qualcomm"] = self.has_qualcomm,
            capabilities["qualcomm_simulation"] = self.qnn_simulation_mode if self.has_qualcomm else False
            ,
        # MediaTek capabilities with simulation flag
            capabilities["mediatek"] = self.has_mediatek if hasattr(self, 'has_mediatek') else False,
            capabilities["mediatek_simulation"] = self.mediatek_simulation_mode if hasattr(self, 'has_mediatek') and self.has_mediatek else False
            ,
        # Samsung NPU capabilities with simulation flag
            capabilities["samsung_npu"] = self.has_samsung_npu,
            capabilities["samsung_npu_simulation"] = self.samsung_npu_simulation_mode if self.has_samsung_npu else False
            ,
        # ROCm capabilities
            capabilities["rocm"] = self.has_rocm
            ,
        # WebNN capabilities with simulation flag
            capabilities["webnn"] = self.has_webnn,
            capabilities["webnn_simulation"] = self.webnn_simulation_mode if self.has_webnn else False
            ,
        # WebGPU capabilities with simulation flag
            capabilities["webgpu"] = self.has_webgpu,
            capabilities["webgpu_simulation"] = self.webgpu_simulation_mode if self.has_webgpu else False
            ,
            return capabilities
    :
        def _detect_browser()))))))))self) -> Dict[str, Any]:,,,,,
        """
        Detect browser type for optimizations, particularly for 
        Firefox WebGPU compute shader optimizations.
        """
        # Start with default ()))))))))simulation environment)
        browser_info = {}}}}}}}}
        "is_browser": False,
        "browser_type": "unknown",
        "is_firefox": False,
        "is_chrome": False,
        "is_edge": False,
        "is_safari": False,
        "supports_compute_shaders": False,
        "workgroup_size": [128, 1, 1]  # Default workgroup size,
        }
        
        # Try to detect browser environment
        try:
            import js
            if hasattr()))))))))js, 'navigator'):
                browser_info["is_browser"] = True,
                user_agent = js.navigator.userAgent.lower())))))))))
                
                # Detect browser type
                if "firefox" in user_agent:
                    browser_info["browser_type"] = "firefox",,
                    browser_info["is_firefox"], = True,,
                    browser_info["supports_compute_shaders"], = True,,,,
                    browser_info["workgroup_size"] = [256, 1, 1]  # Firefox optimized workgroup size,
                elif "chrome" in user_agent:
                    browser_info["browser_type"] = "chrome",
                    browser_info["is_chrome"], = True,
                    browser_info["supports_compute_shaders"], = True,,,,
                elif "edg" in user_agent:
                    browser_info["browser_type"] = "edge",
                    browser_info["is_edge"] = True,
                    browser_info["supports_compute_shaders"], = True,,,,
                elif "safari" in user_agent:
                    browser_info["browser_type"] = "safari",
                    browser_info["is_safari"] = True,
                    browser_info["supports_compute_shaders"], = False  # Safari has limited compute shader support,
        except ()))))))))ImportError, AttributeError):
            # Not in a browser environment
                    pass
        
        # Check environment variables for browser simulation
        if os.environ.get()))))))))"SIMULATE_FIREFOX", "0") == "1":
            browser_info["browser_type"] = "firefox",,
            browser_info["is_firefox"], = True,,
            browser_info["supports_compute_shaders"], = True,,,,
            browser_info["workgroup_size"] = [256, 1, 1]
            ,
                    return browser_info
    
                    def get_capabilities()))))))))self) -> Dict[str, Any]:,,,,,
                    """Get all hardware capabilities."""
                    return self.capabilities
    
                    def get_web_optimizations()))))))))self, model_type: str, implementation_type: Optional[str] = None) -> Dict[str, bool]:,,
                    """
                    Get optimizations for web platform based on model type and implementation.
        
        Args:
            model_type: Type of model ()))))))))audio, multimodal, etc.)
            implementation_type: Implementation type ()))))))))WebNN, WebGPU)
            
        Returns:
            Dict of optimization settings
            """
            optimizations = {}}}}}}}}
            "compute_shaders": False,
            "parallel_loading": False,
            "shader_precompile": False
            }
        
        # Check for optimization environment flags
            compute_shaders_enabled = ()))))))))
            os.environ.get()))))))))"WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1" or
            os.environ.get()))))))))"WEBGPU_COMPUTE_SHADERS", "0") == "1"
            )
        
            parallel_loading_enabled = ()))))))))
            os.environ.get()))))))))"WEB_PARALLEL_LOADING_ENABLED", "0") == "1" or
            os.environ.get()))))))))"PARALLEL_LOADING_ENABLED", "0") == "1"
            )
        
            shader_precompile_enabled = ()))))))))
            os.environ.get()))))))))"WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1" or
            os.environ.get()))))))))"WEBGPU_SHADER_PRECOMPILE", "0") == "1"
            )
        
        # Enable all optimizations flag
        if os.environ.get()))))))))"WEB_ALL_OPTIMIZATIONS", "0") == "1":
            compute_shaders_enabled = True
            parallel_loading_enabled = True
            shader_precompile_enabled = True
        
        # Only apply WebGPU compute shaders for audio models
        if compute_shaders_enabled and implementation_type == "WebGPU" and model_type == "audio":
            optimizations["compute_shaders"] = True
            ,
        # Only apply parallel loading for multimodal models
        if parallel_loading_enabled and model_type == "multimodal":
            optimizations["parallel_loading"] = True
            ,
        # Apply shader precompilation for most model types with WebGPU
        if shader_precompile_enabled and implementation_type == "WebGPU":
            optimizations["shader_precompile"] = True
            ,
            return optimizations
    
            def get_browser_info()))))))))self) -> Dict[str, Any]:,,,,,
            """Get detected browser information."""
            return self.browser_info
    
            def get_workgroup_size()))))))))self) -> List[int]:,
            """Get optimal workgroup size for the current browser."""
            return self.browser_info["workgroup_size"]
            ,
    def is_firefox()))))))))self) -> bool:
        """Check if the current browser is Firefox."""
            return self.browser_info["is_firefox"],
    :
    def is_chrome()))))))))self) -> bool:
        """Check if the current browser is Chrome."""
        return self.browser_info["is_chrome"],
    :
    def supports_compute_shaders()))))))))self) -> bool:
        """Check if the current browser supports compute shaders."""
        return self.browser_info["supports_compute_shaders"],
    :
    def get_hardware_detection_code()))))))))self) -> str:
        """
        Generate hardware detection code that can be inserted into templates.
        Returns Python code as a string.
        """
        code = """
# Hardware Detection
        import os
        import importlib.util

# Try to import torch first ()))))))))needed for CUDA/ROCm/MPS)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock())))))))))
    HAS_TORCH = False
    logger.warning()))))))))"torch not available, using mock")

# Initialize hardware capability flags
    HAS_CUDA = False
    HAS_ROCM = False
    HAS_MPS = False
    HAS_OPENVINO = False
    HAS_WEBNN = False
    HAS_WEBGPU = False

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available())))))))))
    
    # ROCm detection
    if HAS_CUDA and hasattr()))))))))torch, '_C') and hasattr()))))))))torch._C, '_rocm_version'):
        HAS_ROCM = True
    elif 'ROCM_HOME' in os.environ:
        HAS_ROCM = True
    
    # Apple MPS detection
    if hasattr()))))))))torch, "mps") and hasattr()))))))))torch.mps, "is_available"):
        HAS_MPS = torch.mps.is_available())))))))))

# OpenVINO detection
        HAS_OPENVINO = importlib.util.find_spec()))))))))"openvino") is not None

# QNN/Qualcomm detection with simulation awareness
        HAS_QUALCOMM = False
        QNN_SIMULATION_MODE = os.environ.get()))))))))"QNN_SIMULATION_MODE", "0").lower()))))))))) in ()))))))))"1", "true", "yes")
try:
    # Try standardized QNN support module if available:
    try:
        from hardware_detection.qnn_support import QNN_AVAILABLE, QNN_SIMULATION_MODE
        HAS_QUALCOMM = QNN_AVAILABLE
    except ImportError:
        try:
            # Try alternative path
            import sys
            sys.path.append()))))))))'/home/barberb/ipfs_accelerate_py/test')
            from hardware_detection.qnn_support import QNN_AVAILABLE, QNN_SIMULATION_MODE
            HAS_QUALCOMM = QNN_AVAILABLE
        except ImportError:
            # Fall back to basic detection
            HAS_QUALCOMM = ()))))))))
            importlib.util.find_spec()))))))))"qnn_wrapper") is not None or
            importlib.util.find_spec()))))))))"qti") is not None or
            "QUALCOMM_SDK" in os.environ
            )
except Exception as e:
    # Handle unexpected errors gracefully
    if "QUALCOMM_SDK" in os.environ:
        HAS_QUALCOMM = True
        logger.warning()))))))))f"QNN detection error in template: {}}}}}}}}str()))))))))e)}")

# WebNN detection with proper simulation tracking
        webnn_library_available = ()))))))))
        importlib.util.find_spec()))))))))"webnn") is not None or
        importlib.util.find_spec()))))))))"webnn_js") is not None
        )

# Track WebNN simulation status separately
        WEBNN_SIMULATION_MODE = os.environ.get()))))))))"WEBNN_SIMULATION", "0").lower()))))))))) in ()))))))))"1", "true", "yes")
        webnn_override = os.environ.get()))))))))"WEBNN_AVAILABLE", "0").lower()))))))))) in ()))))))))"1", "true", "yes")

if webnn_library_available:
    HAS_WEBNN = True
    logger.debug()))))))))"WebNN library detected")
elif WEBNN_SIMULATION_MODE:
    # Allow simulation if explicitly requested
    HAS_WEBNN = True
    logger.warning()))))))))"WebNN SIMULATION mode enabled via environment variable"):
elif webnn_override:
    # Allow override if explicitly requested, but mark as simulation
    HAS_WEBNN = True
    WEBNN_SIMULATION_MODE = True
    logger.warning()))))))))"WebNN availability forced by environment variable ()))))))))treated as simulation)"):
else:
    HAS_WEBNN = False
    WEBNN_SIMULATION_MODE = False

# WebGPU detection with proper simulation tracking
    webgpu_library_available = ()))))))))
    importlib.util.find_spec()))))))))"webgpu") is not None or
    importlib.util.find_spec()))))))))"wgpu") is not None
    )

# Track WebGPU simulation status separately
    WEBGPU_SIMULATION_MODE = os.environ.get()))))))))"WEBGPU_SIMULATION", "0").lower()))))))))) in ()))))))))"1", "true", "yes")
    webgpu_override = os.environ.get()))))))))"WEBGPU_AVAILABLE", "0").lower()))))))))) in ()))))))))"1", "true", "yes")

if webgpu_library_available:
    HAS_WEBGPU = True
    logger.debug()))))))))"WebGPU library detected")
elif WEBGPU_SIMULATION_MODE:
    # Allow simulation if explicitly requested
    HAS_WEBGPU = True
    logger.warning()))))))))"WebGPU SIMULATION mode enabled via environment variable"):
elif webgpu_override:
    # Allow override if explicitly requested, but mark as simulation
    HAS_WEBGPU = True
    WEBGPU_SIMULATION_MODE = True
    logger.warning()))))))))"WebGPU availability forced by environment variable ()))))))))treated as simulation)"):
else:
    HAS_WEBGPU = False
    WEBGPU_SIMULATION_MODE = False

# Hardware detection function for comprehensive hardware info
def check_hardware()))))))))):
    \"\"\"Check available hardware and return capabilities.\"\"\"
    capabilities = {}}}}}}}}
    "cpu": True,
    "cuda": False,
    "cuda_version": None,
    "cuda_devices": 0,
    "mps": False,
    "openvino": False,
    "qualcomm": False,
    "qualcomm_simulation": QNN_SIMULATION_MODE,
    "rocm": False,
    "webnn": False,
    "webnn_simulation": WEBNN_SIMULATION_MODE,
    "webgpu": False,
    "webgpu_simulation": WEBGPU_SIMULATION_MODE
    }
    
    # CUDA capabilities
    if HAS_TORCH and HAS_CUDA:
        capabilities["cuda"] = True,,
        capabilities["cuda_devices"] = torch.cuda.device_count()))))))))),
        capabilities["cuda_version"] = torch.version.cuda
        ,
    # MPS capabilities ()))))))))Apple Silicon)
        capabilities["mps"] = HAS_MPS
        ,
    # OpenVINO capabilities
        capabilities["openvino"] = HAS_OPENVINO
        ,
    # Qualcomm capabilities
        capabilities["qualcomm"] = HAS_QUALCOMM
        ,
    # ROCm capabilities
        capabilities["rocm"] = HAS_ROCM
        ,
    # WebNN capabilities
        capabilities["webnn"] = HAS_WEBNN
        ,
    # WebGPU capabilities
        capabilities["webgpu"] = HAS_WEBGPU
        ,
    return capabilities

# Get hardware capabilities
    HW_CAPABILITIES = check_hardware())))))))))
    """
    return code
    
    def get_model_hardware_compatibility()))))))))self, model_name: str) -> Dict[str, bool]:,
    """
    Determine hardware compatibility for a specific model.
        
        Args:
            model_name: Name/type of the model to check
            
        Returns:
            Dict indicating which hardware platforms are compatible
            """
        # Base compatibility that's always available
            compatibility = {}}}}}}}}
            "cpu": True,
            "cuda": self.has_cuda,
            "openvino": self.has_openvino,
            "mps": self.has_mps,
            "qualcomm": self.has_qualcomm,
            "mediatek": hasattr(self, 'has_mediatek') and self.has_mediatek,
            "samsung_npu": self.has_samsung_npu,
            "rocm": self.has_rocm,
            "webnn": self.has_webnn,
            "webgpu": self.has_webgpu
            }
        
        # Special cases for specific model families
            model_name = model_name.lower())))))))))
        
        # Multimodal models like LLaVA may have limited support
        if "llava" in model_name:
            compatibility["mps"] = False  # Limited MPS support for LLaVA,
            compatibility["webnn"] = False  # Limited WebNN support for LLaVA,
            compatibility["webgpu"] = False  # Limited WebGPU support for LLaVA,
            compatibility["mediatek"] = False  # Limited MediaTek NPU support for LLaVA
            compatibility["samsung_npu"] = False  # Limited Samsung NPU support for LLaVA
            ,
        # Audio models have limited web support
            if any()))))))))audio_model in model_name for audio_model in ["whisper", "wav2vec2", "clap", "hubert"]):,
            # Audio models have limited but improving web support
            compatibility["webnn"] = compatibility["webnn"] and "WEBNN_AUDIO_SUPPORT" in os.environ,
            compatibility["webgpu"] = compatibility["webgpu"] and "WEBGPU_AUDIO_SUPPORT" in os.environ
            ,
        # LLMs may have limited web support due to size
            if any()))))))))llm in model_name for llm in ["llama", "gpt", "falcon", "mixtral", "qwen"]):,
            compatibility["webnn"] = compatibility["webnn"] and "WEBNN_LLM_SUPPORT" in os.environ,
            compatibility["webgpu"] = compatibility["webgpu"] and "WEBGPU_LLM_SUPPORT" in os.environ,
            compatibility["mediatek"] = compatibility["mediatek"] and (
                "MEDIATEK_LLM_SUPPORT" in os.environ or  # Check for explicit LLM support
                (hasattr(self, 'has_mediatek') and self.has_mediatek and 
                 hasattr(self, 'mediatek_simulation_mode') and not self.mediatek_simulation_mode)  # Real hardware may support it
            ),
            compatibility["samsung_npu"] = compatibility["samsung_npu"] and (
                "SAMSUNG_LLM_SUPPORT" in os.environ or  # Check for explicit LLM support
                (self.has_samsung_npu and not self.samsung_npu_simulation_mode and  # Real hardware may support it
                 hasattr(self, 'samsung_chipset') and self.samsung_chipset.npu_tops >= 25.0)  # High-end chipsets only
            )
            ,
        # Models optimized for mobile may have better performance on mobile NPUs
            if any()))))))))mobile_model in model_name for mobile_model in ["mobilenet", "efficientnet", "mobilevit", "mobilebertx"]):,
            # These models are well-suited for mobile NPUs
            if compatibility["mediatek"]:
                # Prioritize MediaTek for mobile models when available
                logger.info()))))))))f"Mobile-optimized model {}}}}}}}}model_name} is well-suited for MediaTek NPU")
            
            if compatibility["samsung_npu"]:
                # Prioritize Samsung for mobile models when available
                logger.info()))))))))f"Mobile-optimized model {}}}}}}}}model_name} is well-suited for Samsung Exynos NPU")
            ,
            return compatibility

# Create singleton instance
            HARDWARE_MANAGER = HardwareManager())))))))))

# Convenience functions for importing 
def get_hardware_manager()))))))))):
    """Get the HardwareManager singleton instance."""
            return HARDWARE_MANAGER

def get_capabilities()))))))))):
    """Get hardware capabilities."""
            return HARDWARE_MANAGER.get_capabilities())))))))))

def get_web_optimizations()))))))))model_type, implementation_type=None):
    """Get web platform optimizations."""
            return HARDWARE_MANAGER.get_web_optimizations()))))))))model_type, implementation_type)

def get_browser_info()))))))))):
    """Get browser information."""
            return HARDWARE_MANAGER.get_browser_info())))))))))

def get_hardware_detection_code()))))))))):
    """Get hardware detection code for templates."""
            return HARDWARE_MANAGER.get_hardware_detection_code())))))))))

def get_model_hardware_compatibility()))))))))model_name):
    """Get hardware compatibility for a model."""
            return HARDWARE_MANAGER.get_model_hardware_compatibility()))))))))model_name)

            def detect_web_platform_capabilities()))))))))browser: str = "chrome", use_browser_automation: bool = False) -> Dict[str, Any]:,,,,,
            """
            Detect WebNN and WebGPU capabilities using browser automation or environment variables.
    
    Args:
        browser: Browser to use for detection ()))))))))"chrome", "firefox", "edge", "safari")
        use_browser_automation: Whether to use real browser automation ()))))))))requires Selenium)
        
    Returns:
        Dictionary with capability information
        """
        capabilities = {}}}}}}}}
        "webnn_available": HARDWARE_MANAGER.capabilities.get()))))))))"webnn", False),
        "webnn_simulated": HARDWARE_MANAGER.capabilities.get()))))))))"webnn_simulation", True),
        "webgpu_available": HARDWARE_MANAGER.capabilities.get()))))))))"webgpu", False),
        "webgpu_simulated": HARDWARE_MANAGER.capabilities.get()))))))))"webgpu_simulation", True),
        "browser": browser,
        "browser_version": "unknown",
        "webnn_backends": [],
        "webgpu_features": {}}}}}}}}},
        "detection_method": "environment",
        }
    
    # Check if we're using simulation mode:
    if HARDWARE_MANAGER.webnn_simulation_mode:
        capabilities["webnn_simulated"] = True
        ,
    if HARDWARE_MANAGER.webgpu_simulation_mode:
        capabilities["webgpu_simulated"] = True
        ,
    # If browser info is available from the hardware manager, use it
        browser_info = HARDWARE_MANAGER.get_browser_info())))))))))
    if browser_info.get()))))))))"is_browser", False):
        capabilities["browser"] = browser_info.get()))))))))"browser_type", "unknown"),
        capabilities["detection_method"] = "browser"
        ,
        # Set WebGPU features based on browser type
        if browser_info.get()))))))))"is_firefox", False):
            capabilities["webgpu_features"] = {}}}}}}}},,
            "workgroup_size": [256, 1, 1],
            "supports_compute_shaders": True,
            "browser_optimized": True
            }
        elif browser_info.get()))))))))"is_chrome", False):
            capabilities["webgpu_features"] = {}}}}}}}},,
            "workgroup_size": [128, 2, 1],
            "supports_compute_shaders": True,
            "browser_optimized": True
            }
    
            return capabilities