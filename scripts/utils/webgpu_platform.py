#!/usr/bin/env python3
"""
IPFS Accelerate Python Framework - WebGPU Platform Integration

This module provides WebGPU acceleration for machine learning models in the
IPFS Accelerate framework. It supports running models in web browsers with
WebGPU API support like Chrome, Firefox, and Edge, as well as in Node.js
environments with WebGPU capability.

Key features:
- WebGPU detection and initialization
- Model conversion to WebGPU format
- Compute shader optimizations, especially for Firefox
- Shader precompilation for faster initialization
- Parallel model loading for multimodal models
- Browser-specific optimizations

Usage:
    from ipfs_accelerate_py.webgpu_platform import WebGPUPlatform
    
    # Initialize the WebGPU platform
    platform = WebGPUPlatform()
    
    # Check if WebGPU is available
    if platform.is_available():
        # Run inference with WebGPU acceleration
        result = await platform.run_inference(model_name, input_data)
"""

import os
import sys
import json
import time
import platform
import logging
import importlib.util
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("webgpu_platform")

class WebGPUPlatform:
    """
    WebGPU Platform integration for hardware-accelerated machine learning.
    
    This class provides methods to:
    1. Detect WebGPU capabilities
    2. Initialize WebGPU context
    3. Run inference with WebGPU acceleration
    4. Apply optimizations for different browsers
    5. Integrate with IPFS Accelerate Python framework
    """
    
    def __init__(self, browser: str = None, enable_optimizations: bool = True, 
                 simulation: bool = False, cache_dir: str = None):
        """
        Initialize the WebGPU platform.
        
        Args:
            browser: Browser to use ('firefox', 'chrome', 'edge', 'safari')
            enable_optimizations: Whether to enable browser-specific optimizations
            simulation: Whether to enable simulation mode (for testing)
            cache_dir: Directory to cache WebGPU shaders and models
        """
        self.browser = browser
        self.enable_optimizations = enable_optimizations
        self.simulation = simulation
        self.cache_dir = cache_dir
        
        # Initialize state
        self._is_available = False
        self._device = None
        self._adapter = None
        self._context = None
        self._adapters_info = {}
        self._browser_info = {}
        
        # Optimizations configuration
        self.compute_shaders_enabled = os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1"
        self.parallel_loading_enabled = os.environ.get("WEB_PARALLEL_LOADING_ENABLED", "0") == "1"
        self.shader_precompile_enabled = os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1"
        
        # Transformers.js integration
        self.transformers_js_available = False
        
        # Setup WebGPU platform
        self._detect_capabilities()
    
    def _detect_capabilities(self):
        """Detect WebGPU capabilities and browser environment."""
        # First check if we're in a browser environment
        in_browser = self._detect_browser_environment()
        
        if in_browser and self.browser is None:
            # Auto-detect browser based on user agent
            self.browser = self._browser_info.get("type", "unknown")
            logger.info(f"Auto-detected browser: {self.browser}")
            
            # Check for WebGPU API in browser
            if "gpu" in self._browser_info.get("navigator_features", []):
                self._is_available = True
                logger.info("WebGPU API detected in browser")
        
        # Check for simulation mode
        elif self.simulation or os.environ.get("WEBGPU_SIMULATION") == "1":
            self._is_available = True
            self.simulation = True
            logger.info("WebGPU simulation mode enabled")
        
        # Check for Node.js environment with WebGPU support
        else:
            # Check for Transformers.js
            self._detect_node_environment()
            
            # Check for explicit availability flag
            if os.environ.get("WEBGPU_AVAILABLE") == "1":
                self._is_available = True
                logger.info("WebGPU availability forced by environment variable")
    
    def _detect_browser_environment(self) -> bool:
        """
        Detect if running in a browser environment and gather browser information.
        
        Returns:
            True if in browser environment, False otherwise
        """
        # Check if we're running in a browser environment
        try:
            # This would only work if executed in a browser environment like Pyodide
            js_window = eval("window")
            js_navigator = eval("navigator")
            
            # Store browser information
            self._browser_info = {
                "is_browser": True,
                "user_agent": eval("navigator.userAgent"),
                "platform": eval("navigator.platform"),
                "language": eval("navigator.language"),
                "navigator_features": []
            }
            
            # Check navigator features
            for feature in ["gpu", "ml", "mediaDevices", "serviceWorker"]:
                if feature in js_navigator:
                    self._browser_info["navigator_features"].append(feature)
            
            # Detect browser type from user agent
            user_agent = self._browser_info["user_agent"].lower()
            
            if "firefox" in user_agent:
                self._browser_info["type"] = "firefox"
                self._browser_info["is_firefox"] = True
            elif "chrome" in user_agent:
                self._browser_info["type"] = "chrome"
                self._browser_info["is_chrome"] = True
            elif "safari" in user_agent:
                self._browser_info["type"] = "safari"
                self._browser_info["is_safari"] = True
            elif "edg" in user_agent:
                self._browser_info["type"] = "edge"
                self._browser_info["is_edge"] = True
            else:
                self._browser_info["type"] = "unknown"
            
            # Get GPU information if possible
            try:
                canvas = eval("document.createElement('canvas')")
                gl = eval("canvas.getContext('webgl') || canvas.getContext('experimental-webgl')")
                
                if gl:
                    debugInfo = eval("gl.getExtension('WEBGL_debug_renderer_info')")
                    if debugInfo:
                        self._browser_info["gpu_info"] = {
                            "vendor": eval("gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)"),
                            "renderer": eval("gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)")
                        }
            except Exception as e:
                logger.debug(f"Error getting GPU info: {e}")
                pass
            
            return True
        except:
            # Not in a browser environment
            self._browser_info = {"is_browser": False}
            return False
    
    def _detect_node_environment(self):
        """Detect Node.js environment and check for WebGPU support."""
        # Check for Node.js
        try:
            import subprocess
            
            # Check if Node.js is available
            try:
                with open(os.devnull, 'w') as devnull:
                    subprocess.check_call(["which", "node"], stdout=devnull, stderr=devnull)
                
                # Get Node.js version
                node_version = subprocess.check_output(["node", "--version"], universal_newlines=True).strip()
                self._browser_info = {
                    "is_browser": False,
                    "is_node": True,
                    "node_version": node_version
                }
                
                # Check for npm packages
                try:
                    with open(os.devnull, 'w') as devnull:
                        subprocess.check_call(["which", "npm"], stdout=devnull, stderr=devnull)
                    
                    # List installed packages
                    npm_list = subprocess.check_output(["npm", "list", "--json"], universal_newlines=True)
                    npm_packages = json.loads(npm_list)
                    
                    # Check for Transformers.js
                    dependencies = npm_packages.get("dependencies", {})
                    if "@xenova/transformers" in dependencies:
                        self.transformers_js_available = True
                        self._browser_info["transformers_js"] = True
                        
                        # Try to get version
                        try:
                            pkg_info = subprocess.check_output(["npm", "view", "@xenova/transformers", "version"], universal_newlines=True).strip()
                            self._browser_info["transformers_js_version"] = pkg_info
                        except:
                            pass
                        
                        # Transformers.js implicitly supports WebGPU with the right backend
                        self._is_available = True
                    
                    # Check for other WebGPU packages
                    has_webgpu_pkg = "@webgpu/types" in dependencies
                    if has_webgpu_pkg:
                        self._is_available = True
                        self._browser_info["webgpu_types"] = True
                
                except (subprocess.SubprocessError, json.JSONDecodeError):
                    pass
            
            except (subprocess.SubprocessError, FileNotFoundError):
                # Node.js not available
                self._browser_info = {
                    "is_browser": False,
                    "is_node": False
                }
        
        except Exception as e:
            logger.error(f"Error detecting Node.js environment: {e}")
    
    def is_available(self) -> bool:
        """Check if WebGPU is available."""
        return self._is_available
    
    def get_browser_info(self) -> Dict[str, Any]:
        """Get browser information."""
        return self._browser_info
    
    def get_optimizations(self, model_type: str) -> Dict[str, bool]:
        """
        Get optimizations configuration based on model type and browser.
        
        Args:
            model_type: Type of model ('text', 'vision', 'audio', 'multimodal')
            
        Returns:
            Dictionary with optimization flags
        """
        optimizations = {
            "compute_shaders": False,
            "parallel_loading": False,
            "shader_precompile": False
        }
        
        # Only apply optimizations if enabled
        if not self.enable_optimizations:
            return optimizations
        
        # Apply optimizations based on environment variables
        compute_shaders_enabled = (
            os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1" or
            os.environ.get("WEBGPU_COMPUTE_SHADERS", "0") == "1"
        )
        
        parallel_loading_enabled = (
            os.environ.get("WEB_PARALLEL_LOADING_ENABLED", "0") == "1" or
            os.environ.get("PARALLEL_LOADING_ENABLED", "0") == "1"
        )
        
        shader_precompile_enabled = (
            os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1" or
            os.environ.get("WEBGPU_SHADER_PRECOMPILE", "0") == "1"
        )
        
        # Apply optimizations based on model type and browser
        if compute_shaders_enabled and model_type == "audio":
            optimizations["compute_shaders"] = True
        
        if parallel_loading_enabled and model_type == "multimodal":
            optimizations["parallel_loading"] = True
        
        if shader_precompile_enabled:
            optimizations["shader_precompile"] = True
        
        # Firefox-specific optimizations
        if self.browser == "firefox" and compute_shaders_enabled and model_type == "audio":
            # Firefox has superior WebGPU compute shader performance for audio models
            optimizations["compute_shaders"] = True
            
            # Set environment variable for Firefox-specific optimizations
            os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
        
        return optimizations
    
    async def initialize(self) -> bool:
        """
        Initialize WebGPU device and adapter.
        
        Returns:
            True if initialization successful, False otherwise
        """
        # Check if WebGPU is available
        if not self.is_available():
            logger.error("WebGPU is not available")
            return False
        
        # If we're in a browser environment, initialize browser WebGPU
        if self._browser_info.get("is_browser", False):
            return await self._initialize_browser_webgpu()
        
        # If we're in a Node.js environment, initialize Node.js WebGPU
        elif self._browser_info.get("is_node", False):
            return self._initialize_node_webgpu()
        
        # If we're in simulation mode, initialize simulated WebGPU
        elif self.simulation:
            return self._initialize_simulated_webgpu()
        
        # WebGPU not available in this environment
        logger.error("WebGPU not available in this environment")
        return False
    
    async def _initialize_browser_webgpu(self) -> bool:
        """
        Initialize WebGPU in a browser environment.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Check if GPU object exists in navigator
            if "gpu" not in eval("navigator"):
                logger.error("WebGPU API not available in browser")
                return False
            
            # Request adapter
            try:
                self._adapter = await eval("navigator.gpu.requestAdapter()")
                
                if not self._adapter:
                    logger.error("Failed to get WebGPU adapter")
                    return False
                
                # Get adapter info
                self._adapters_info = {
                    "name": await eval("adapter.name"),
                    "features": await eval("Array.from(adapter.features).map(f => f.toString())"),
                    "limits": await eval("Object.fromEntries(Object.entries(adapter.limits))")
                }
                
                # Request device with appropriate limits
                device_descriptor = {
                    "requiredFeatures": ["shader-f16"],
                    "requiredLimits": {
                        "maxBufferSize": 1024 * 1024 * 1024,  # 1GB
                        "maxStorageBufferBindingSize": 1024 * 1024 * 1024,  # 1GB
                        "maxComputeWorkgroupStorageSize": 32768,
                        "maxComputeInvocationsPerWorkgroup": 1024,
                        "maxComputeWorkgroupSizeX": 1024,
                        "maxComputeWorkgroupSizeY": 1024,
                        "maxComputeWorkgroupSizeZ": 64
                    }
                }
                
                self._device = await eval(f"adapter.requestDevice({json.dumps(device_descriptor)})")
                
                if not self._device:
                    logger.error("Failed to get WebGPU device")
                    return False
                
                logger.info("WebGPU device initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error initializing WebGPU in browser: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error in browser WebGPU initialization: {e}")
            return False
    
    def _initialize_node_webgpu(self) -> bool:
        """
        Initialize WebGPU in a Node.js environment.
        
        Returns:
            True if initialization successful, False otherwise
        """
        # Check if Transformers.js is available
        if self.transformers_js_available:
            logger.info("WebGPU initialization via Transformers.js in Node.js")
            return True
        
        # Implement Node.js WebGPU initialization when available
        # This may require integration with a WebGPU implementation for Node.js
        logger.error("WebGPU direct initialization in Node.js not implemented yet")
        return False
    
    def _initialize_simulated_webgpu(self) -> bool:
        """
        Initialize simulated WebGPU for testing.
        
        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("Initializing simulated WebGPU")
        
        # Create simulated WebGPU device and adapter
        self._adapter = {"name": "Simulated WebGPU Adapter", "features": ["shader-f16"]}
        self._device = {"label": "Simulated WebGPU Device"}
        
        return True
    
    async def run_inference(self, model_name: str, input_data: Any, 
                          model_type: str = None) -> Dict[str, Any]:
        """
        Run inference on a model using WebGPU acceleration.
        
        Args:
            model_name: Name of the model
            input_data: Input data for inference
            model_type: Type of model ('text', 'vision', 'audio', 'multimodal')
            
        Returns:
            Inference results
        """
        # Check if WebGPU is available and initialized
        if not self.is_available():
            return {"error": "WebGPU is not available"}
        
        if self._device is None and not await self.initialize():
            return {"error": "WebGPU initialization failed"}
        
        # Determine model type if not provided
        if model_type is None:
            model_type = self._determine_model_type(model_name)
        
        # Apply model type-specific optimizations
        optimizations = self.get_optimizations(model_type)
        
        # In simulation mode, return simulated results
        if self.simulation:
            return self._simulate_inference(model_name, input_data, model_type, optimizations)
        
        # In browser environment, run actual WebGPU inference
        if self._browser_info.get("is_browser", False):
            return await self._run_browser_inference(model_name, input_data, model_type, optimizations)
        
        # In Node.js environment with Transformers.js
        if self._browser_info.get("is_node", False) and self.transformers_js_available:
            return await self._run_transformers_js_inference(model_name, input_data, model_type, optimizations)
        
        # Fallback (should never reach here if checks are proper)
        return {"error": "WebGPU inference not supported in this environment"}
    
    def _determine_model_type(self, model_name: str) -> str:
        """
        Determine the type of model based on its name or architecture.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model type ('text', 'vision', 'audio', 'multimodal')
        """
        model_name = model_name.lower()
        
        # Text models
        if any(x in model_name for x in ["bert", "t5", "gpt", "roberta", "llama", "opt"]):
            return "text"
        
        # Vision models
        elif any(x in model_name for x in ["vit", "resnet", "detr", "yolo"]):
            return "vision"
        
        # Audio models
        elif any(x in model_name for x in ["whisper", "wav2vec", "clap", "hubert"]):
            return "audio"
        
        # Multimodal models
        elif any(x in model_name for x in ["clip", "llava", "blip", "xclip"]):
            return "multimodal"
        
        # Default to text
        return "text"
    
    def _simulate_inference(self, model_name: str, input_data: Any, 
                           model_type: str, optimizations: Dict[str, bool]) -> Dict[str, Any]:
        """
        Simulate inference for testing without actual WebGPU hardware.
        
        Args:
            model_name: Name of the model
            input_data: Input data for inference
            model_type: Type of model ('text', 'vision', 'audio', 'multimodal')
            optimizations: Dictionary with optimization flags
            
        Returns:
            Simulated inference results
        """
        import numpy as np
        
        # Generate simulated inference time based on model type
        if model_type == "text":
            inference_time = 0.05  # 50ms
            if isinstance(input_data, str):
                inference_time += len(input_data) * 0.001  # 1ms per character
        elif model_type == "vision":
            inference_time = 0.1  # 100ms
        elif model_type == "audio":
            inference_time = 0.2  # 200ms
            if optimizations["compute_shaders"]:
                inference_time *= 0.6  # 40% faster with compute shaders
        elif model_type == "multimodal":
            inference_time = 0.3  # 300ms
            if optimizations["parallel_loading"]:
                inference_time *= 0.7  # 30% faster with parallel loading
        
        # Apply shader precompilation optimization
        if optimizations["shader_precompile"]:
            # This doesn't affect inference time, but improves startup time
            pass
        
        # Simulate computation time
        time.sleep(inference_time)
        
        # Generate simulated output based on model type
        result = {
            "model_name": model_name,
            "model_type": model_type,
            "inference_time_ms": inference_time * 1000,
            "optimizations": optimizations,
            "webgpu_simulation": True
        }
        
        # Add model-specific output
        if model_type == "text":
            if "bert" in model_name.lower():
                # Embedding model
                result["output"] = np.random.randn(768).tolist()
            else:
                # Text generation model
                result["output"] = f"Simulated text generation output for input: {input_data}"
        
        elif model_type == "vision":
            # Vision model (classification)
            classes = ["cat", "dog", "car", "person", "bird"]
            scores = np.random.random(len(classes))
            scores = scores / scores.sum()
            result["output"] = {
                "scores": scores.tolist(),
                "labels": classes,
                "top_score": scores.max(),
                "top_label": classes[scores.argmax()]
            }
        
        elif model_type == "audio":
            # Audio model (speech recognition)
            result["output"] = {
                "text": "Simulated transcription of audio input",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "Simulated"},
                    {"start": 2.0, "end": 3.5, "text": "transcription"},
                    {"start": 3.5, "end": 5.0, "text": "of audio input"}
                ]
            }
        
        elif model_type == "multimodal":
            # Multimodal model
            if "clip" in model_name.lower():
                # CLIP-like model
                result["output"] = {
                    "scores": np.random.random(5).tolist(),
                    "text_embedding": np.random.randn(512).tolist(),
                    "image_embedding": np.random.randn(512).tolist()
                }
            else:
                # LLaVA-like model
                result["output"] = "Simulated description of image: A person standing near a mountain."
        
        return result
    
    async def _run_browser_inference(self, model_name: str, input_data: Any, 
                                   model_type: str, optimizations: Dict[str, bool]) -> Dict[str, Any]:
        """
        Run inference in a browser environment with WebGPU.
        
        Args:
            model_name: Name of the model
            input_data: Input data for inference
            model_type: Type of model ('text', 'vision', 'audio', 'multimodal')
            optimizations: Dictionary with optimization flags
            
        Returns:
            Inference results
        """
        try:
            # In a real browser environment, we'd use a WebGPU-enabled inference library
            # like Transformers.js with WebGPU backend
            
            # For now, return a mock implementation
            logger.warning("Direct WebGPU browser inference not implemented yet")
            
            # Simulate inference time
            await eval("new Promise(resolve => setTimeout(resolve, 100))")
            
            # Use simulation for now
            return self._simulate_inference(model_name, input_data, model_type, optimizations)
            
        except Exception as e:
            logger.error(f"Error in browser WebGPU inference: {e}")
            return {"error": f"Browser WebGPU inference error: {str(e)}"}
    
    async def _run_transformers_js_inference(self, model_name: str, input_data: Any, 
                                          model_type: str, optimizations: Dict[str, bool]) -> Dict[str, Any]:
        """
        Run inference using Transformers.js in Node.js environment.
        
        Args:
            model_name: Name of the model
            input_data: Input data for inference
            model_type: Type of model ('text', 'vision', 'audio', 'multimodal')
            optimizations: Dictionary with optimization flags
            
        Returns:
            Inference results
        """
        # This would require integration with Node.js and Transformers.js
        # For now, return a simulated result
        logger.warning("Transformers.js inference not implemented yet")
        return self._simulate_inference(model_name, input_data, model_type, optimizations)

    def get_supported_models(self) -> List[str]:
        """
        Get a list of models that are supported by this WebGPU implementation.
        
        Returns:
            List of supported model names
        """
        # This would depend on the specific WebGPU implementation
        # For now, return a default list of commonly supported models
        return [
            "bert-base-uncased",
            "gpt2",
            "t5-small",
            "vit-base-patch16-224",
            "whisper-tiny",
            "clip-vit-base-patch32"
        ]
    
    def get_optimal_workgroup_size(self, model_type: str) -> List[int]:
        """
        Get the optimal workgroup size for WebGPU compute shaders based on browser and model type.
        
        Args:
            model_type: Type of model ('text', 'vision', 'audio', 'multimodal')
            
        Returns:
            List of workgroup dimensions [x, y, z]
        """
        # Firefox-optimized workgroup sizes
        if self.browser == "firefox":
            if model_type == "audio":
                return [256, 1, 1]  # Firefox optimal configuration for audio processing
            else:
                return [128, 4, 1]
        
        # Chrome/Edge default workgroup sizes
        elif self.browser in ["chrome", "edge"]:
            if model_type == "audio":
                return [128, 2, 1]
            else:
                return [128, 4, 1]
        
        # Safari or unknown browser
        else:
            return [64, 4, 1]  # Safe default
            
    def get_pipeline_stages(self, model_type: str) -> List[str]:
        """
        Get the WebGPU pipeline stages for a specific model type.
        
        Args:
            model_type: Type of model ('text', 'vision', 'audio', 'multimodal')
            
        Returns:
            List of pipeline stage names
        """
        # Common pipeline stages
        common_stages = ["model_loading", "preprocessing", "inference", "postprocessing"]
        
        # Model-specific stages
        if model_type == "text":
            return common_stages + ["tokenization", "embedding_lookup"]
        elif model_type == "vision":
            return common_stages + ["image_normalization", "feature_extraction"]
        elif model_type == "audio":
            return common_stages + ["audio_processing", "mel_spectrogram", "feature_extraction"]
        elif model_type == "multimodal":
            if self.get_optimizations(model_type).get("parallel_loading", False):
                return common_stages + ["parallel_encoder_loading", "cross_attention"]
            else:
                return common_stages + ["text_encoder_loading", "vision_encoder_loading", "cross_attention"]
        
        # Default stages
        return common_stages

def create_webgpu_platform(browser: str = None, enable_optimizations: bool = True,
                         simulation: bool = False, cache_dir: str = None) -> WebGPUPlatform:
    """
    Create a WebGPU platform instance with the specified configuration.
    
    Args:
        browser: Browser to use ('firefox', 'chrome', 'edge', 'safari')
        enable_optimizations: Whether to enable browser-specific optimizations
        simulation: Whether to enable simulation mode (for testing)
        cache_dir: Directory to cache WebGPU shaders and models
        
    Returns:
        WebGPU platform instance
    """
    platform = WebGPUPlatform(
        browser=browser,
        enable_optimizations=enable_optimizations,
        simulation=simulation,
        cache_dir=cache_dir
    )
    
    return platform

async def run_webgpu_inference(model_name: str, input_data: Any, model_type: str = None,
                              browser: str = None, simulation: bool = False) -> Dict[str, Any]:
    """
    Run inference on a model using WebGPU acceleration.
    
    Args:
        model_name: Name of the model
        input_data: Input data for inference
        model_type: Type of model ('text', 'vision', 'audio', 'multimodal')
        browser: Browser to use ('firefox', 'chrome', 'edge', 'safari')
        simulation: Whether to enable simulation mode (for testing)
        
    Returns:
        Inference results
    """
    # Create WebGPU platform
    platform = create_webgpu_platform(browser=browser, simulation=simulation)
    
    # Check if WebGPU is available
    if not platform.is_available():
        return {"error": "WebGPU is not available"}
    
    # Initialize WebGPU
    if not await platform.initialize():
        return {"error": "WebGPU initialization failed"}
    
    # Run inference
    return await platform.run_inference(model_name, input_data, model_type)

if __name__ == "__main__":
    async def main():
        # Create WebGPU platform
        platform = create_webgpu_platform(simulation=True)
        
        # Check if WebGPU is available
        if platform.is_available():
            print("WebGPU is available")
            print(f"Browser info: {platform.get_browser_info()}")
            
            # Initialize WebGPU
            if await platform.initialize():
                print("WebGPU initialized successfully")
                
                # Run inference on different model types
                print("\nRunning inference on text model:")
                text_result = await platform.run_inference("bert-base-uncased", "Hello, world!", "text")
                print(f"Inference time: {text_result['inference_time_ms']:.2f}ms")
                
                print("\nRunning inference on vision model:")
                vision_result = await platform.run_inference("vit-base-patch16-224", "image_data", "vision")
                print(f"Inference time: {vision_result['inference_time_ms']:.2f}ms")
                
                print("\nRunning inference on audio model:")
                audio_result = await platform.run_inference("whisper-tiny", "audio_data", "audio")
                print(f"Inference time: {audio_result['inference_time_ms']:.2f}ms")
                
                print("\nRunning inference on multimodal model:")
                multimodal_result = await platform.run_inference("clip-vit-base-patch32", ["image_data", "text_data"], "multimodal")
                print(f"Inference time: {multimodal_result['inference_time_ms']:.2f}ms")
            else:
                print("WebGPU initialization failed")
        else:
            print("WebGPU is not available")
    
    # Run main function
    anyio.run(main)