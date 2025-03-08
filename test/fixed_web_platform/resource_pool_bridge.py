#!/usr/bin/env python3
"""
Resource Pool Bridge for Web Platform (March 2025)

This module provides a resource pool implementation for WebNN and WebGPU platforms
that enables concurrent model execution on both GPU and CPU using Selenium bridge.

Key features:
- Pool of browser connections for efficient resource utilization
- Parallel model execution across GPU (WebGPU) and CPU backends
- Model-specific hardware preferences based on model type
- Automatic load balancing and resource allocation
- Connection management with health monitoring and recovery
- Efficient resource cleanup and memory management

March 8, 2025 Update:
- Enhanced model retry capability with progressive backoff
- Improved error handling for reliable model loading
- Added model-specific optimizations based on model family
- Added adaptive connection management for large workloads
- Enhanced diagnostics and error reporting

Usage:
    from fixed_web_platform.resource_pool_bridge import ResourcePoolBridge
    
    # Create resource pool bridge
    bridge = ResourcePoolBridge(max_connections=4)
    
    # Initialize with model configurations
    bridge.initialize([
        {
            'model_id': 'vision-model',
            'model_path': 'https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.onnx',
            'backend': 'webgpu',
            'family': 'vision'
        },
        {
            'model_id': 'text-model',
            'model_path': 'https://huggingface.co/bert-base-uncased/resolve/main/model.onnx',
            'backend': 'cpu',
            'family': 'text_embedding'
        }
    ])
    
    # Run parallel inference
    vision_result, text_result = bridge.run_parallel([
        ('vision-model', {'input': image_data}),
        ('text-model', {'input_ids': text_data})
    ])
"""

import os
import sys
import json
import time
import uuid
import random
import logging
import asyncio
import threading
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Import resource pool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_pool import get_global_resource_pool

# Import locally to avoid circular imports
from fixed_web_platform.browser_automation import BrowserAutomation

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrowserConnection:
    """
    Manages a connection to a browser instance for model inference.
    
    This class handles communication with a browser via WebSocket and Selenium,
    providing an interface for loading models and running inference in the browser.
    """
    
    def __init__(self, connection_id: str, browser_name: str = 'chrome', 
                 platform: str = 'webgpu', headless: bool = True, 
                 compute_shaders: bool = False, precompile_shaders: bool = False,
                 parallel_loading: bool = False, websocket_port: int = None):
        """
        Initialize browser connection.
        
        Args:
            connection_id: Unique identifier for this connection
            browser_name: Browser name ('chrome', 'firefox', 'edge', 'safari')
            platform: 'webgpu' or 'webnn'
            headless: Whether to run browser in headless mode
            compute_shaders: Enable compute shader optimization
            precompile_shaders: Enable shader precompilation
            parallel_loading: Enable parallel model loading
            websocket_port: Port for WebSocket communication with browser (auto-assigned if None)
        """
        self.connection_id = connection_id
        self.browser_name = browser_name
        self.platform = platform
        self.headless = headless
        self.compute_shaders = compute_shaders
        self.precompile_shaders = precompile_shaders
        self.parallel_loading = parallel_loading
        self.websocket_port = websocket_port or (8765 + random.randint(0, 1000))  # Use random port if not specified
        
        # State variables
        self.initialized = False
        self.browser_automation = None
        self.loaded_models = set()
        self.busy = False
        self.creation_time = time.time()
        self.last_used_time = time.time()
        self.error_count = 0
        self.max_errors = 3
        
        # WebSocket bridge for real communication
        self.has_websocket_module = False
        try:
            import websockets
            self.has_websocket_module = True
        except ImportError:
            logger.warning("websockets module not available, will use simulation")
        
        # Concurrency control
        self._lock = threading.RLock()
        self._loop = None  # Will be set during initialization
    
    async def initialize(self):
        """
        Initialize the browser connection.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        with self._lock:
            if self.initialized:
                return True
            
            try:
                # Store event loop for later use
                self._loop = asyncio.get_event_loop()
                
                # Import modules needed for browser and WebSocket
                websocket_bridge = None
                
                # Import WebSocketBridge from fixed_web_platform module
                try:
                    # First try to import from current module's path
                    module_path = os.path.dirname(os.path.abspath(__file__))
                    sys.path.append(module_path)
                    
                    # Import WebSocketBridge
                    if self.has_websocket_module:
                        try:
                            from websocket_bridge import WebSocketBridge, create_websocket_bridge
                            
                            # Create WebSocket bridge
                            logger.info(f"Creating WebSocket bridge on port {self.websocket_port}")
                            websocket_bridge = await create_websocket_bridge(port=self.websocket_port)
                            
                            if websocket_bridge:
                                logger.info(f"WebSocket bridge created successfully for connection {self.connection_id}")
                            else:
                                logger.warning(f"Failed to create WebSocket bridge for connection {self.connection_id}")
                        except ImportError:
                            logger.warning("websocket_bridge module not found, will use simulation")
                        except Exception as e:
                            logger.warning(f"Error creating WebSocket bridge: {e}")
                except Exception as e:
                    logger.warning(f"Error importing WebSocket bridge: {e}")
                
                # Create browser automation instance
                try:
                    from browser_automation import BrowserAutomation
                    
                    # Create with WebSocket port if bridge was created
                    self.browser_automation = BrowserAutomation(
                        platform=self.platform,
                        browser_name=self.browser_name,
                        headless=self.headless,
                        compute_shaders=self.compute_shaders,
                        precompile_shaders=self.precompile_shaders,
                        parallel_loading=self.parallel_loading,
                        test_port=self.websocket_port if websocket_bridge else None
                    )
                    
                    # Add WebSocket bridge to browser automation
                    if websocket_bridge:
                        self.browser_automation.websocket_bridge = websocket_bridge
                    
                    # Launch browser
                    success = await self.browser_automation.launch()
                    if not success:
                        logger.error(f"Failed to launch browser for connection {self.connection_id}")
                        # Clean up WebSocket bridge if it was created
                        if websocket_bridge:
                            await websocket_bridge.stop()
                        return False
                    
                    # Wait for WebSocket connection if using bridge
                    if websocket_bridge:
                        logger.info("Waiting for WebSocket connection from browser...")
                        # Use retry capability for more reliable connection establishment
                        websocket_connected = await websocket_bridge.wait_for_connection(
                            timeout=15.0,
                            retry_attempts=3  # Use multiple retry attempts with our enhanced bridge
                        )
                        if websocket_connected:
                            logger.info(f"WebSocket connection established successfully for {self.browser_name}/{self.platform}")
                            
                            # Get browser capabilities to verify feature support
                            try:
                                capabilities = await websocket_bridge.get_browser_capabilities(retry_attempts=2)
                                
                                # Log hardware-specific capabilities 
                                if capabilities:
                                    # Check WebGPU support
                                    webgpu_support = capabilities.get("webgpu_supported", False)
                                    if webgpu_support and self.platform == "webgpu":
                                        adapter = capabilities.get("webgpu_adapter", {})
                                        logger.info(f"WebGPU Adapter: {adapter.get('vendor', 'Unknown')} - {adapter.get('device', 'Unknown')}")
                                    
                                    # Check WebNN support
                                    webnn_support = capabilities.get("webnn_supported", False)
                                    if webnn_support and self.platform == "webnn":
                                        backend = capabilities.get("webnn_backend", "Unknown")
                                        logger.info(f"WebNN Backend: {backend}")
                                    
                                    # Log compute shader support for audio models
                                    if self.compute_shaders:
                                        compute_shaders = capabilities.get("compute_shaders_supported", False)
                                        logger.info(f"Compute Shader Support: {'Available' if compute_shaders else 'Not Available'}")
                            except Exception as e:
                                logger.warning(f"Error checking browser capabilities: {e}")
                        else:
                            logger.warning("WebSocket connection timed out, will use simulation for inference")
                except ImportError:
                    logger.error("BrowserAutomation not available, using simulation")
                    return False
                except Exception as e:
                    logger.error(f"Error creating BrowserAutomation: {e}")
                    traceback.print_exc()
                    # Clean up WebSocket bridge if it was created
                    if websocket_bridge:
                        await websocket_bridge.stop()
                    return False
                
                self.initialized = True
                logger.info(f"Browser connection {self.connection_id} initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error initializing browser connection {self.connection_id}: {e}")
                traceback.print_exc()
                return False
    
    async def load_model(self, model_config: Dict[str, Any]) -> bool:
        """
        Load a model in the browser.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        with self._lock:
            self.busy = True
            self.last_used_time = time.time()
            
            try:
                model_id = model_config.get('model_id')
                model_path = model_config.get('model_path')
                
                if not model_id or not model_path:
                    logger.error(f"Invalid model configuration: {model_config}")
                    self.busy = False
                    return False
                
                if model_id in self.loaded_models:
                    logger.info(f"Model {model_id} already loaded in connection {self.connection_id}")
                    self.busy = False
                    return True
                
                # Execute load model operation
                # Connect to the browser via WebSocket to load the model using transformers.js
                
                if hasattr(self, 'browser_automation') and self.browser_automation:
                    try:
                        # Use WebSocket bridge if it exists on the browser_automation
                        if hasattr(self.browser_automation, 'websocket_bridge') and self.browser_automation.websocket_bridge:
                            # Create model initialization request
                            bridge = self.browser_automation.websocket_bridge
                            model_data = {
                                'model_name': model_id,
                                'model_type': model_config.get('family', 'text'),
                                'platform': self.platform,
                                'quantization': {
                                    'bits': model_config.get('bits', None),
                                    'mixed': model_config.get('mixed_precision', False),
                                    'experimental': model_config.get('experimental_precision', False)
                                }
                            }
                            
                            # Prepare enhanced options with model-specific optimizations
                            options = model_data.copy()
                            
                            # Add optimization flags based on model type
                            if model_data['model_type'] == 'audio' and self.platform == 'webgpu':
                                # Enable compute shader optimization for audio models (especially in Firefox)
                                if 'optimizations' not in options:
                                    options['optimizations'] = {}
                                options['optimizations']['compute_shaders'] = True
                                logger.info(f"Enabling compute shader optimization for audio model {model_id}")
                            
                            # For vision models, enable shader precompilation
                            if model_data['model_type'] == 'vision' and self.platform == 'webgpu':
                                if 'optimizations' not in options:
                                    options['optimizations'] = {}
                                options['optimizations']['precompile_shaders'] = True
                                logger.info(f"Enabling shader precompilation for vision model {model_id}")
                                
                            # For multimodal models, enable parallel loading
                            if model_data['model_type'] in ['multimodal', 'vision_language'] and self.platform == 'webgpu':
                                if 'optimizations' not in options:
                                    options['optimizations'] = {}
                                options['optimizations']['parallel_loading'] = True
                                logger.info(f"Enabling parallel loading for multimodal model {model_id}")
                                
                            # Send request to browser via WebSocket with retries for reliability
                            logger.info(f"Initializing model {model_id} in browser via WebSocket...")
                            start_time = time.time()
                            
                            # Use retry capability for more reliable model initialization
                            response = await bridge.initialize_model(
                                model_id, 
                                model_data['model_type'], 
                                self.platform, 
                                options,
                                retry_attempts=2  # Use retry capabilities in our enhanced bridge
                            )
                            
                            init_time = time.time() - start_time
                            
                            if response and response.get('status') == 'success':
                                logger.info(f"Model {model_id} loaded in browser via WebSocket in {init_time:.2f}s")
                                
                                # Log memory usage if available
                                if 'memory_usage' in response:
                                    logger.info(f"Initial memory usage: {response['memory_usage']} MB")
                                    
                                # Log adapter info if available
                                if 'adapter_info' in response:
                                    adapter = response['adapter_info']
                                    logger.info(f"Using adapter: {adapter.get('vendor', 'Unknown')} - {adapter.get('device', 'Unknown')}")
                            else:
                                error = response.get('error', 'Unknown error') if response else 'No response'
                                logger.warning(f"Failed to load model {model_id} in browser: {error}")
                                logger.warning("Falling back to simulation")
                                await asyncio.sleep(1.0)  # Fallback to simulation
                        else:
                            # Fallback to simulation if WebSocket bridge not available
                            logger.warning("WebSocket bridge not available, using simulation")
                            await asyncio.sleep(1.0)  # Simulate model loading time
                    except Exception as e:
                        logger.error(f"Error during real model loading: {e}")
                        await asyncio.sleep(1.0)  # Fallback to simulation
                else:
                    # Fallback to simulation if browser automation not available
                    logger.warning("Browser automation not available, using simulation")
                    await asyncio.sleep(1.0)  # Simulate model loading time
                
                # Mark model as loaded
                self.loaded_models.add(model_id)
                logger.info(f"Model {model_id} loaded in connection {self.connection_id}")
                
                self.busy = False
                return True
                
            except Exception as e:
                logger.error(f"Error loading model in connection {self.connection_id}: {e}")
                traceback.print_exc()
                self.error_count += 1
                self.busy = False
                return False
    
    async def run_inference(self, model_id: str, inputs: Dict[str, Any], retry_attempts: int = 1) -> Dict[str, Any]:
        """
        Run inference with a loaded model with enhanced reliability.
        
        Args:
            model_id: ID of the model to use
            inputs: Model inputs
            retry_attempts: Number of retry attempts if inference fails
            
        Returns:
            Dictionary with inference results
        """
        with self._lock:
            if not self.initialized:
                error_msg = "Browser connection not initialized"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'model_id': model_id}
                
            if model_id not in self.loaded_models:
                error_msg = f"Model {model_id} not loaded in this connection"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg, 'model_id': model_id}
                
            self.busy = True
            self.last_used_time = time.time()
            
            # Track retries
            attempt = 0
            last_error = None
            
            while attempt <= retry_attempts:
                try:
                    # Execute inference operation using WebSocket to communicate with real browser
                    if hasattr(self, 'browser_automation') and self.browser_automation:
                        # Use WebSocket bridge if it exists on the browser_automation
                        if hasattr(self.browser_automation, 'websocket_bridge') and self.browser_automation.websocket_bridge:
                            bridge = self.browser_automation.websocket_bridge
                            
                            # For retry attempts, log that we're retrying
                            if attempt > 0:
                                logger.info(f"Retry {attempt}/{retry_attempts} for inference with model {model_id}")
                            
                            # Determine appropriate timeout based on model and content
                            input_size = 'unknown'
                            timeout_multiplier = 2  # Default multiplier
                            
                            # Estimate input size for better diagnostics and timeout adjustment
                            if isinstance(inputs, dict):
                                # Check for common input keys and their sizes
                                if 'input_ids' in inputs and isinstance(inputs['input_ids'], list):
                                    input_size = len(inputs['input_ids'])
                                    # Increase timeout for large inputs
                                    if input_size > 512:
                                        timeout_multiplier = 3
                                elif 'pixel_values' in inputs:
                                    input_size = 'image data'
                                    timeout_multiplier = 2.5  # Vision models can take longer
                                elif 'input_features' in inputs:
                                    input_size = 'audio data'
                                    timeout_multiplier = 3  # Audio models often take longer
                            
                            logger.info(f"Running inference with model {model_id} (input size: {input_size})")
                            
                            # Send inference request to browser via WebSocket with our enhanced timeout and retry
                            start_time = time.time()
                            inference_response = await bridge.run_inference(
                                model_id, 
                                inputs, 
                                self.platform,
                                retry_attempts=1,        # Use internal retry in WebSocket bridge
                                timeout_multiplier=timeout_multiplier  # Use calculated timeout multiplier
                            )
                            inference_time = time.time() - start_time
                            
                            if inference_response and inference_response.get('status') == 'success':
                                logger.info(f"Real inference completed in {inference_time:.2f}s via WebSocket")
                                
                                # Extract response data with better error handling
                                result_data = inference_response.get('result', {})
                                performance_metrics = inference_response.get('performance_metrics', {})
                                
                                # If metrics don't exist or are incomplete, create them from our timing
                                if not performance_metrics or 'inference_time_ms' not in performance_metrics:
                                    performance_metrics['inference_time_ms'] = inference_time * 1000
                                    performance_metrics['throughput_items_per_sec'] = 1.0 / inference_time if inference_time > 0 else 0
                                
                                # Add memory metrics if available
                                if 'memory_usage' in inference_response:
                                    memory_usage = inference_response['memory_usage']
                                    performance_metrics['memory_usage_mb'] = memory_usage
                                    logger.info(f"Memory usage: {memory_usage} MB")
                                
                                # Convert inference response to comprehensive result format
                                result = {
                                    'success': True,
                                    'status': 'success',
                                    'model_id': model_id,
                                    'platform': self.platform,
                                    'backend': 'webgpu' if self.platform == 'webgpu' else 'webnn',
                                    'browser': self.browser_name,
                                    'is_real_implementation': True,
                                    'is_simulation': False,
                                    'output': result_data,
                                    'performance_metrics': performance_metrics,
                                    'inference_time': inference_time,
                                    'compute_shader_optimized': self.compute_shaders,
                                    'precompile_shaders': self.precompile_shaders,
                                    'parallel_loading': self.parallel_loading
                                }
                                
                                # Add any custom fields from the response
                                for key, value in inference_response.items():
                                    if key not in ['status', 'result', 'performance_metrics'] and key not in result:
                                        result[key] = value
                                
                                self.busy = False
                                return result
                            else:
                                # Extract error information for better debugging
                                error = "Unknown error"
                                if inference_response:
                                    error = inference_response.get('error', 'No error details')
                                    
                                last_error = f"WebSocket inference failed: {error}"
                                logger.warning(last_error)
                                
                                # If we have retries left, try again
                                if attempt < retry_attempts:
                                    attempt += 1
                                    # Brief pause before retry
                                    await asyncio.sleep(0.5 * attempt)
                                    continue
                                    
                                # No more retries, fall back to simulation
                                logger.warning(f"Falling back to simulation after {attempt} failed attempts")
                                break  # Break out to simulation fallback
                        else:
                            last_error = "WebSocket bridge not available for inference"
                            logger.warning(last_error)
                            break  # Break out to simulation fallback
                    else:
                        last_error = "Browser automation not available"
                        logger.warning(last_error)
                        break  # Break out to simulation fallback
                        
                except Exception as e:
                    # Handle exceptions during inference
                    last_error = f"Error during inference: {e}"
                    logger.error(last_error)
                    if attempt < retry_attempts:
                        attempt += 1
                        # Brief pause before retry
                        await asyncio.sleep(0.5 * attempt)
                        continue
                    else:
                        # No more retries, fall back to simulation
                        break
                
                # This should never be reached due to breaks, but just in case
                break
            
            # Fallback to simulation after all retries failed or if real implementation not available
            logger.warning(f"Using simulation for inference with model {model_id}")
            
            # Simulate realistic inference time based on model type
            model_type = model_id.split("/")[-1] if "/" in model_id else model_id
            inference_time = 0.5  # Default simulation time
            
            # Adjust simulation time based on model name hints
            if "bert" in model_type.lower():
                inference_time = 0.3  # BERT models are often faster
            elif "gpt" in model_type.lower() or "llama" in model_type.lower():
                inference_time = 1.0  # LLMs take longer
            elif "vit" in model_type.lower() or "clip" in model_type.lower():
                inference_time = 0.4  # Vision models
            elif "whisper" in model_type.lower() or "wav2vec" in model_type.lower():
                inference_time = 0.8  # Audio models take longer
                
            # Simulate execution time
            await asyncio.sleep(inference_time)
            
            # Create detailed mock result that mimics a real response
            result = {
                'success': True,
                'status': 'success',
                'model_id': model_id,
                'output_shape': [1, 768],  # Mock output shape
                'platform': self.platform,
                'backend': 'webgpu' if self.platform == 'webgpu' else 'webnn',
                'browser': self.browser_name,
                'is_real_implementation': False,
                'is_simulation': True,
                'error_info': last_error,  # Include error that caused fallback
                'performance_metrics': {
                    'inference_time_ms': inference_time * 1000,
                    'throughput_items_per_sec': 1.0 / inference_time,
                    'memory_usage_mb': 500  # Mock memory usage
                },
                'compute_shader_optimized': self.compute_shaders,
                'precompile_shaders': self.precompile_shaders,
                'parallel_loading': self.parallel_loading
            }
            
            self.busy = False
            return result
    
    async def close(self):
        """Close the browser connection and clean up resources."""
        with self._lock:
            # Shutdown browser automation and WebSocket
            if self.browser_automation:
                try:
                    # Send shutdown command to browser via WebSocket
                    if hasattr(self.browser_automation, 'websocket_bridge') and self.browser_automation.websocket_bridge:
                        try:
                            await self.browser_automation.websocket_bridge.shutdown_browser()
                            # Stop WebSocket bridge
                            await self.browser_automation.websocket_bridge.stop()
                            logger.info(f"WebSocket bridge closed for connection {self.connection_id}")
                        except Exception as e:
                            logger.error(f"Error closing WebSocket bridge: {e}")
                    
                    # Close browser automation
                    await self.browser_automation.close()
                    logger.info(f"Browser connection {self.connection_id} closed")
                except Exception as e:
                    logger.error(f"Error closing browser connection {self.connection_id}: {e}")
            
            # Reset state
            self.initialized = False
            self.loaded_models.clear()
            self.browser_automation = None
    
    def is_busy(self) -> bool:
        """Check if connection is busy."""
        with self._lock:
            return self.busy
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        with self._lock:
            return self.initialized and self.error_count < self.max_errors
    
    def get_idle_time(self) -> float:
        """Get idle time in seconds."""
        with self._lock:
            return time.time() - self.last_used_time
    
    def get_age(self) -> float:
        """Get age in seconds."""
        with self._lock:
            return time.time() - self.creation_time
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model IDs."""
        with self._lock:
            return list(self.loaded_models)
    
    def has_model(self, model_id: str) -> bool:
        """Check if model is loaded."""
        with self._lock:
            return model_id in self.loaded_models
    
    def get_platform(self) -> str:
        """Get platform (webgpu or webnn)."""
        return self.platform
    
    def get_connection_id(self) -> str:
        """Get connection ID."""
        return self.connection_id

class ResourcePoolBridge:
    """
    Resource pool for WebNN and WebGPU platforms.
    
    This class manages a pool of browser connections for running models on WebNN and WebGPU,
    allowing concurrent execution of models on different hardware backends.
    """
    
    def __init__(self, max_connections: int = 4, 
                 browser: str = 'chrome', enable_gpu: bool = True, 
                 enable_cpu: bool = True, headless: bool = True,
                 cleanup_interval: int = 300):
        """
        Initialize resource pool bridge.
        
        Args:
            max_connections: Maximum number of concurrent browser connections
            browser: Default browser name ('chrome', 'firefox', 'edge', 'safari')
            enable_gpu: Enable GPU (WebGPU) backend
            enable_cpu: Enable CPU backend
            headless: Whether to run browsers in headless mode
            cleanup_interval: Interval in seconds for connection cleanup
        """
        self.max_connections = max_connections
        self.default_browser = browser
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu
        self.headless = headless
        self.cleanup_interval = cleanup_interval
        
        # Connection pools by platform
        self.connections = {
            'webgpu': [],
            'webnn': [],
            'cpu': []
        }
        
        # Model configurations
        self.model_configs = {}
        
        # Model to connection mapping
        self.model_connections = {}
        
        # Stats and monitoring
        self.stats = {
            'created_connections': 0,
            'total_inferences': 0,
            'total_model_loads': 0,
            'errors': 0,
            'peak_connections': 0
        }
        
        # Concurrency control
        self._lock = threading.RLock()
        self._loop = None
        self._initialized = False
        self._cleanup_task = None
        
        # Initialize asyncio event loop
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if none exists
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        
        logger.info(f"ResourcePoolBridge initialized with max {max_connections} connections")
    
    async def initialize(self, model_configs: List[Dict[str, Any]] = None):
        """
        Initialize the resource pool bridge with model configurations.
        
        Args:
            model_configs: List of model configuration dictionaries
            
        Returns:
            True if initialization succeeded, False otherwise
        """
        with self._lock:
            if self._initialized:
                return True
            
            # Store model configurations
            if model_configs:
                for config in model_configs:
                    model_id = config.get('model_id')
                    if model_id:
                        self.model_configs[model_id] = config
            
            # Start cleanup task
            self._start_cleanup_task()
            
            self._initialized = True
            logger.info("ResourcePoolBridge initialized successfully")
            return True
    
    def _start_cleanup_task(self):
        """Start the connection cleanup task."""
        async def cleanup_task():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self._cleanup_connections()
                except asyncio.CancelledError:
                    # Task is being cancelled
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        # Schedule cleanup task
        self._cleanup_task = asyncio.ensure_future(cleanup_task(), loop=self._loop)
    
    async def _cleanup_connections(self):
        """Clean up idle and unhealthy connections."""
        with self._lock:
            # Collect connections to close
            to_close = []
            
            for platform, connections in self.connections.items():
                for conn in connections[:]:  # Create a copy to safely modify during iteration
                    # Close unhealthy connections
                    if not conn.is_healthy():
                        to_close.append(conn)
                        connections.remove(conn)
                        logger.info(f"Removing unhealthy connection {conn.get_connection_id()}")
                        continue
                    
                    # Close idle connections after 10 minutes
                    if conn.get_idle_time() > 600 and not conn.is_busy():
                        to_close.append(conn)
                        connections.remove(conn)
                        logger.info(f"Removing idle connection {conn.get_connection_id()} (idle for {conn.get_idle_time():.1f}s)")
                        continue
            
            # Close connections
            for conn in to_close:
                try:
                    # Remove from model_connections mapping
                    removed_models = []
                    for model_id, connection in list(self.model_connections.items()):
                        if connection == conn:
                            del self.model_connections[model_id]
                            removed_models.append(model_id)
                    
                    # Log models that were removed
                    if removed_models:
                        logger.info(f"Removed {len(removed_models)} models from connection {conn.get_connection_id()}: {', '.join(removed_models)}")
                    
                    # Close connection
                    await conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            logger.info(f"Cleaned up {len(to_close)} connections")
    
    async def get_connection(self, platform: str, browser: str = None) -> Optional[BrowserConnection]:
        """
        Get an available connection for the specified platform.
        
        Args:
            platform: 'webgpu', 'webnn', or 'cpu'
            browser: Optional browser name to prefer
            
        Returns:
            BrowserConnection or None if unable to create connection
        """
        with self._lock:
            # Validate platform
            if platform not in ['webgpu', 'webnn', 'cpu']:
                logger.error(f"Invalid platform: {platform}")
                return None
            
            # Check if we should use this platform
            if platform == 'webgpu' and not self.enable_gpu:
                logger.warning("GPU backend is disabled, using CPU instead")
                platform = 'cpu'
            elif platform == 'cpu' and not self.enable_cpu:
                logger.warning("CPU backend is disabled, using GPU instead")
                platform = 'webgpu' if self.enable_gpu else 'webnn'
            
            # Use specified browser or default
            browser = browser or self.default_browser
            
            # First, try exact match (platform AND browser)
            if browser:
                exact_matches = [
                    conn for conn in self.connections[platform]
                    if not conn.is_busy() and conn.is_healthy() and conn.browser_name == browser
                ]
                
                if exact_matches:
                    # Find the connection with the least loaded models
                    exact_matches.sort(key=lambda conn: len(conn.loaded_models))
                    connection = exact_matches[0]
                    logger.debug(f"Reusing existing connection {connection.connection_id} for {platform}/{browser} with {len(connection.loaded_models)} models")
                    return connection
            
            # Then try platform match with any browser
            available = [
                conn for conn in self.connections[platform]
                if not conn.is_busy() and conn.is_healthy()
            ]
            
            if available:
                # Balance model loading across connections based on model count and usage time
                # This prevents overloading a single connection with too many models
                
                # Use a weighted formula considering both model count and recency
                # Connections with fewer models are preferred, but recently used connections
                # are preferred to maximize cache benefit
                weighted_connections = []
                for conn in available:
                    # Factor in both model count (lower is better) and recency (higher is better)
                    model_count = len(conn.loaded_models)
                    recency = time.time() - conn.last_used_time
                    
                    # Compute weight (lower is better)
                    # Models count more heavily than recency
                    weight = model_count * 100 + min(recency, 60)
                    weighted_connections.append((conn, weight))
                
                # Sort by weight (lower is better)
                weighted_connections.sort(key=lambda x: x[1])
                connection = weighted_connections[0][0]
                
                # Log detailed connection selection
                logger.debug(f"Selected connection {connection.connection_id} for {platform} with {len(connection.loaded_models)} models")
                logger.debug(f"Available connections: {[(conn.connection_id, len(conn.loaded_models)) for conn, _ in weighted_connections]}")
                
                return connection
            
            # Check if we can create a new connection
            total_connections = sum(len(conns) for conns in self.connections.values())
            if total_connections >= self.max_connections:
                # Try to find a suitable connection from any platform
                all_available = []
                for p, platform_conns in self.connections.items():
                    # Get connections that match the preferred browser
                    matching_browser = [
                        conn for conn in platform_conns
                        if not conn.is_busy() and conn.is_healthy() and conn.browser_name == browser
                    ]
                    
                    # First prefer connections with matching browser
                    if matching_browser:
                        all_available.extend(matching_browser)
                    else:
                        # Then add any available connections
                        all_available.extend([
                            conn for conn in platform_conns
                            if not conn.is_busy() and conn.is_healthy()
                        ])
                
                if all_available:
                    # Use connection with the smallest number of models and compatible browser if possible
                    all_available.sort(key=lambda conn: (
                        # Prioritize browser match (lower is better)
                        0 if conn.browser_name == browser else 1,
                        # Then prioritize fewer models
                        len(conn.loaded_models),
                        # If tied, prefer most recently used for cache benefits
                        -conn.last_used_time
                    ))
                    
                    connection = all_available[0]
                    logger.debug(f"Reusing connection {connection.connection_id} from platform {connection.platform} with {len(connection.loaded_models)} models")
                    return connection
                
                # No connections available
                logger.warning(f"Maximum connections ({self.max_connections}) reached and all are busy")
                return None
            
            # Create a new connection
            connection_id = str(uuid.uuid4())
            
            # Configure connection based on platform
            compute_shaders = False
            precompile_shaders = False
            parallel_loading = False
            
            if platform == 'webgpu':
                # Enable WebGPU optimizations
                compute_shaders = True
                precompile_shaders = True
                parallel_loading = True
            
            # Log detailed information about connection creation
            logger.info(f"Creating new connection {connection_id} for {platform}/{browser} " +
                       f"(Current: {total_connections}/{self.max_connections})")
            
            # Create and initialize connection
            connection = BrowserConnection(
                connection_id=connection_id,
                browser_name=browser,
                platform=platform,
                headless=self.headless,
                compute_shaders=compute_shaders,
                precompile_shaders=precompile_shaders,
                parallel_loading=parallel_loading
            )
            
            # Initialize connection
            success = await connection.initialize()
            if not success:
                logger.error(f"Failed to initialize connection {connection_id}")
                return None
            
            # Add to pool
            self.connections[platform].append(connection)
            self.stats['created_connections'] += 1
            
            # Update peak connections
            total_connections = sum(len(conns) for conns in self.connections.values())
            self.stats['peak_connections'] = max(self.stats['peak_connections'], total_connections)
            
            logger.info(f"Created new connection {connection_id} for {platform}/{browser}")
            return connection
    
    async def load_model(self, model_id: str) -> Tuple[bool, Optional[BrowserConnection]]:
        """
        Load a model in a browser connection.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Tuple of (success, connection)
        """
        with self._lock:
            # Check if model is already loaded
            if model_id in self.model_connections:
                connection = self.model_connections[model_id]
                if connection.is_healthy() and connection.has_model(model_id):
                    logger.debug(f"Model {model_id} already loaded in connection {connection.get_connection_id()}")
                    return True, connection
            
            # Check if model is configured
            if model_id not in self.model_configs:
                logger.error(f"No configuration found for model {model_id}")
                return False, None
            
            # Get model configuration
            model_config = self.model_configs[model_id]
            
            # Try to import web_utils from worker module
            try:
                from fixed_web_platform.worker.web_utils import initialize_web_model, get_optimal_browser_for_model
                worker_module_available = True
            except ImportError:
                worker_module_available = False
                logger.warning("Worker module not available for model loading, using default implementation")
            
            # Determine platform based on model configuration
            platform = model_config.get('backend', 'webgpu')
            if platform not in ['webgpu', 'webnn', 'cpu']:
                logger.warning(f"Invalid platform '{platform}' for model {model_id}, using webgpu")
                platform = 'webgpu'
            
            # Get appropriate browser based on model family and platform
            if worker_module_available and model_config.get('family'):
                # Use worker module to get optimal browser
                browser = get_optimal_browser_for_model(model_config.get('family'), platform)
            else:
                # Use default browser selection
                browser = self._get_browser_for_model(model_config)
            
            # Get connection for platform
            connection = await self.get_connection(platform, browser)
            if not connection:
                logger.error(f"Failed to get connection for model {model_id}")
                return False, None
            
            # Use worker module to initialize model if available
            if worker_module_available and hasattr(connection, 'browser_automation') and \
               hasattr(connection.browser_automation, 'websocket_bridge'):
                try:
                    # Get WebSocket bridge
                    websocket_bridge = connection.browser_automation.websocket_bridge
                    
                    # Initialize model using worker module
                    model_type = model_config.get('model_type', 'text')
                    options = model_config.get('optimizations', {})
                    
                    # Initialize using worker module
                    init_result = await initialize_web_model(
                        model_id,
                        model_type,
                        platform,
                        options=options,
                        websocket_bridge=websocket_bridge
                    )
                    
                    # Check if initialization was successful
                    if init_result and init_result.get('status') == 'success':
                        # Mark model as loaded in connection
                        connection.loaded_models.add(model_id)
                        self.model_connections[model_id] = connection
                        self.stats['total_model_loads'] += 1
                        
                        logger.info(f"Model {model_id} loaded in connection {connection.get_connection_id()} using worker module")
                        return True, connection
                    else:
                        logger.warning(f"Model initialization with worker module failed, falling back to default implementation")
                except Exception as e:
                    logger.error(f"Error initializing model with worker module: {e}")
                    # Fall back to default implementation
            
            # Load model in connection using default method
            success = await connection.load_model(model_config)
            if not success:
                logger.error(f"Failed to load model {model_id}")
                return False, None
            
            # Store model-connection mapping
            self.model_connections[model_id] = connection
            self.stats['total_model_loads'] += 1
            
            logger.info(f"Model {model_id} loaded in connection {connection.get_connection_id()}")
            return True, connection
    
    async def run_inference(self, model_id: str, inputs: Dict[str, Any], retry_attempts: int = 1) -> Dict[str, Any]:
        """
        Run inference with a model with enhanced reliability.
        
        Args:
            model_id: ID of the model to use
            inputs: Model inputs
            retry_attempts: Number of retry attempts if inference or load fails
            
        Returns:
            Dictionary with inference results
        """
        with self._lock:
            # Track overall start time for performance monitoring
            overall_start_time = time.time()
            
            # Load model if not already loaded, with retries
            load_attempt = 0
            while load_attempt <= retry_attempts:
                # Try to load the model
                success, connection = await self.load_model(model_id)
                
                if success and connection:
                    # Model loaded successfully
                    break
                    
                # Model loading failed, try again if we have retries left
                load_attempt += 1
                if load_attempt <= retry_attempts:
                    logger.warning(f"Retrying model load {load_attempt}/{retry_attempts} for {model_id}")
                    await asyncio.sleep(0.5 * load_attempt)  # Progressive backoff
                else:
                    # All retries failed
                    logger.error(f"Failed to load model {model_id} for inference after {retry_attempts} attempts")
                    return {
                        'success': False, 
                        'status': 'error',
                        'error': f"Failed to load model {model_id} after {retry_attempts} attempts",
                        'model_id': model_id,
                        'total_time': time.time() - overall_start_time
                    }
            
            # Model is now loaded, attempt to use worker module for inference if available
            try:
                from fixed_web_platform.worker.web_utils import run_web_inference
                worker_module_available = True
                logger.debug("Worker module available for inference")
            except ImportError:
                worker_module_available = False
                logger.debug("Worker module not available for inference, using default implementation")
            
            # Run inference with appropriate method
            inference_attempt = 0
            last_error = None
            
            while inference_attempt <= retry_attempts:
                try:
                    # If this is a retry, log it
                    if inference_attempt > 0:
                        logger.info(f"Retry {inference_attempt}/{retry_attempts} for inference with model {model_id}")
                    
                    # Check if we should use the worker module
                    if worker_module_available and hasattr(connection, 'platform') and connection.platform in ['webgpu', 'webnn']:
                        try:
                            # Get WebSocket bridge if available
                            websocket_bridge = None
                            if hasattr(connection, 'browser_automation') and connection.browser_automation:
                                if hasattr(connection.browser_automation, 'websocket_bridge'):
                                    websocket_bridge = connection.browser_automation.websocket_bridge
                            
                            if websocket_bridge:
                                # Get model configuration for optimization options
                                model_config = self.model_configs.get(model_id, {})
                                options = model_config.get('optimizations', {})
                                
                                # Add model-specific optimizations based on model type
                                model_family = model_config.get('family', '')
                                if model_family == 'audio' and connection.platform == 'webgpu':
                                    # Enable compute shader optimization for audio models
                                    options['compute_shaders'] = True
                                elif model_family == 'vision' and connection.platform == 'webgpu':
                                    # Enable shader precompilation for vision models
                                    options['precompile_shaders'] = True
                                elif model_family in ['multimodal', 'vision_language'] and connection.platform == 'webgpu':
                                    # Enable parallel loading for multimodal models
                                    options['parallel_loading'] = True
                                
                                # Run inference using worker module with the enhanced bridge
                                platform = connection.platform
                                
                                logger.info(f"Running inference with worker module using WebSocket bridge (platform: {platform})")
                                result = await run_web_inference(
                                    model_id, 
                                    inputs, 
                                    platform,
                                    options=options,
                                    websocket_bridge=websocket_bridge
                                )
                                
                                if result and result.get('success', False):
                                    # Update stats
                                    self.stats['total_inferences'] += 1
                                    
                                    # Add additional metadata
                                    result['total_time'] = time.time() - overall_start_time
                                    result['model_family'] = model_family
                                    result['browser'] = connection.browser_name
                                    
                                    return result
                                else:
                                    # Worker module inference failed, record error and try standard method
                                    error = result.get('error', 'Unknown error in worker module') if result else 'No result from worker module'
                                    logger.warning(f"Worker module inference failed: {error}")
                                    last_error = error
                            else:
                                logger.debug("WebSocket bridge not available in worker module, using standard method")
                        except Exception as e:
                            logger.warning(f"Error running inference with worker module: {e}")
                            last_error = f"Worker module error: {e}"
                    
                    # Run inference using standard method with retry capability
                    logger.info(f"Running inference using standard method for model {model_id}")
                    result = await connection.run_inference(model_id, inputs, retry_attempts=1)
                    
                    if result and result.get('success', False):
                        # Update stats
                        self.stats['total_inferences'] += 1
                        
                        # Add total time to result
                        result['total_time'] = time.time() - overall_start_time
                        
                        # Add model family if available
                        if model_id in self.model_configs:
                            result['model_family'] = self.model_configs[model_id].get('family', '')
                        
                        return result
                    else:
                        # Standard inference failed, record error
                        error = result.get('error', 'Unknown error') if result else 'No result from inference'
                        logger.warning(f"Standard inference failed: {error}")
                        last_error = error
                        
                        # Try again if we have retries left
                        if inference_attempt < retry_attempts:
                            inference_attempt += 1
                            await asyncio.sleep(0.5 * inference_attempt)  # Progressive backoff
                            continue
                        else:
                            # Return the last result we got, even if unsuccessful
                            logger.error(f"Inference failed after {retry_attempts} attempts")
                            if result:
                                result['total_attempts'] = inference_attempt + 1
                                return result
                            else:
                                return {
                                    'success': False,
                                    'status': 'error',
                                    'error': last_error or 'Unknown error during inference',
                                    'model_id': model_id,
                                    'total_attempts': inference_attempt + 1,
                                    'total_time': time.time() - overall_start_time
                                }
                
                except Exception as e:
                    # Handle unexpected exceptions during inference
                    logger.error(f"Exception during inference execution: {e}")
                    last_error = str(e)
                    
                    # Try again if we have retries left
                    if inference_attempt < retry_attempts:
                        inference_attempt += 1
                        await asyncio.sleep(0.5 * inference_attempt)  # Progressive backoff
                    else:
                        # No more retries, return error
                        return {
                            'success': False,
                            'status': 'error',
                            'error': f"Exception during inference: {e}",
                            'model_id': model_id,
                            'total_attempts': inference_attempt + 1,
                            'total_time': time.time() - overall_start_time
                        }
                
                # This should only be reached if all ways to run inference have been tried
                inference_attempt += 1
            
            # This should not be reached due to the returns above, but just in case
            return {
                'success': False,
                'status': 'error',
                'error': last_error or 'Failed to run inference (unknown reason)',
                'model_id': model_id,
                'total_time': time.time() - overall_start_time
            }
    
    async def run_parallel(self, tasks: List[Tuple[str, Dict[str, Any]]], batch_size: int = 0, 
                    timeout: float = None) -> List[Dict[str, Any]]:
        """
        Run multiple inferences in parallel with intelligent batching.
        
        This method executes multiple inference tasks concurrently, with options for 
        controlling batch size and timeout. It includes dynamic batching based on available
        resources and model characteristics.
        
        Args:
            tasks: List of (model_id, inputs) tuples
            batch_size: Maximum number of concurrent tasks (0 for auto-sizing)
            timeout: Timeout in seconds for each batch (None for no timeout)
            
        Returns:
            List of inference results in the same order as tasks
        """
        # If no tasks, return empty list
        if not tasks:
            return []
            
        # Configure auto-batching based on available connections
        if batch_size <= 0:
            # Count available connections
            available_connections = 0
            for platform_conns in self.connections.values():
                available_connections += sum(1 for conn in platform_conns if conn.is_healthy())
            
            # Set batch size based on available connections with a minimum
            # of 1 and maximum of 8 concurrent tasks
            batch_size = max(1, min(available_connections * 2, 8))
            logger.debug(f"Auto-sized batch to {batch_size} based on {available_connections} available connections")
        
        # Initialize results list with placeholder for each task
        results = [None] * len(tasks)
        
        # Process tasks in batches
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        start_time = time.time()
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            batch_size_actual = len(batch_tasks)
            
            logger.debug(f"Processing batch {batch_idx+1}/{total_batches} with {batch_size_actual} tasks")
            
            # Create async tasks for this batch
            batch_futures = []
            batch_indices = []
            
            for i, (model_id, inputs) in enumerate(batch_tasks):
                task_idx = batch_start + i
                future = asyncio.create_task(self.run_inference(model_id, inputs))
                batch_futures.append(future)
                batch_indices.append(task_idx)
            
            # Wait for batch completion with optional timeout
            try:
                if timeout:
                    # Use wait_for with timeout
                    batch_start_time = time.time()
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*batch_futures, return_exceptions=True),
                        timeout=timeout
                    )
                    batch_duration = time.time() - batch_start_time
                else:
                    # No timeout
                    batch_start_time = time.time()
                    batch_results = await asyncio.gather(*batch_futures, return_exceptions=True)
                    batch_duration = time.time() - batch_start_time
                
                # Calculate and log batch performance metrics
                tasks_per_second = batch_size_actual / batch_duration if batch_duration > 0 else 0
                logger.debug(f"Batch {batch_idx+1}/{total_batches} completed in {batch_duration:.2f}s " +
                          f"({tasks_per_second:.2f} tasks/second)")
                
                # Process batch results
                for i, result in enumerate(batch_results):
                    task_idx = batch_indices[i]
                    model_id = tasks[task_idx][0]
                    
                    if isinstance(result, Exception):
                        # Handle exception
                        error_result = {
                            'success': False,
                            'error': str(result),
                            'model_id': model_id
                        }
                        results[task_idx] = error_result
                        logger.error(f"Error in parallel inference for model {model_id}: {result}")
                        self.stats['errors'] += 1
                    else:
                        # Normal result
                        results[task_idx] = result
                
            except asyncio.TimeoutError:
                # Handle timeout for entire batch
                logger.error(f"Timeout in batch {batch_idx+1}/{total_batches} after {timeout}s")
                
                # Mark all tasks in this batch as timed out
                for i, (model_id, _) in enumerate(batch_tasks):
                    task_idx = batch_start + i
                    results[task_idx] = {
                        'success': False,
                        'error': f"Timeout after {timeout}s",
                        'model_id': model_id
                    }
                
                # Cancel all pending futures in this batch
                for future in batch_futures:
                    if not future.done():
                        future.cancel()
                
                self.stats['errors'] += batch_size_actual
                
            except Exception as e:
                # Handle unexpected error in batch processing
                logger.error(f"Unexpected error in batch {batch_idx+1}/{total_batches}: {e}")
                
                # Mark all remaining tasks as failed
                for i, (model_id, _) in enumerate(batch_tasks):
                    task_idx = batch_start + i
                    if results[task_idx] is None:
                        results[task_idx] = {
                            'success': False,
                            'error': f"Batch processing error: {e}",
                            'model_id': model_id
                        }
                
                self.stats['errors'] += batch_size_actual
        
        # Calculate and log overall performance
        total_duration = time.time() - start_time
        tasks_per_second = len(tasks) / total_duration if total_duration > 0 else 0
        logger.info(f"Completed {len(tasks)} tasks in {total_duration:.2f}s ({tasks_per_second:.2f} tasks/second)")
        
        # Update resource utilization statistics
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        self.stats['parallel_executions'] = self.stats.get('parallel_executions', 0) + 1
        self.stats['parallel_tasks'] = self.stats.get('parallel_tasks', 0) + len(tasks)
        self.stats['parallel_success'] = self.stats.get('parallel_success', 0) + success_count
        
        # Track largest batch processed
        self.stats['largest_parallel_batch'] = max(
            self.stats.get('largest_parallel_batch', 0),
            len(tasks)
        )
        
        return results
    
    def _get_browser_for_model(self, model_config: Dict[str, Any]) -> str:
        """
        Determine the best browser for a model based on its configuration.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Browser name to use
        """
        # Check model family
        family = model_config.get('family', '')
        platform = model_config.get('backend', 'webgpu')
        
        # Default browser
        browser = self.default_browser
        
        # Optimize browser choice based on model family and platform
        if platform == 'webgpu':
            if family in ['audio', 'speech', 'audio_classification', 'asr']:
                # Firefox has excellent compute shader performance for audio models
                browser = 'firefox'
            elif family in ['vision', 'image_classification', 'object_detection']:
                # Chrome/Edge have good WebGPU support for vision models
                browser = 'chrome' if self.default_browser != 'edge' else 'edge'
        elif platform == 'webnn':
            # Edge has the best WebNN support
            browser = 'edge'
        
        return browser
    
    async def close(self):
        """Close all connections and clean up resources."""
        with self._lock:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
                self._cleanup_task = None
            
            # Close all connections
            close_tasks = []
            for platform, connections in self.connections.items():
                for conn in connections:
                    close_tasks.append(conn.close())
            
            # Wait for all connections to close
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            
            # Clear connections
            for platform in self.connections:
                self.connections[platform] = []
            
            # Clear model mappings
            self.model_connections.clear()
            
            logger.info("ResourcePoolBridge closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        with self._lock:
            # Count current connections
            connection_counts = {
                platform: len(connections)
                for platform, connections in self.connections.items()
            }
            
            total_connections = sum(connection_counts.values())
            
            # Count loaded models
            loaded_models = set()
            for platform, connections in self.connections.items():
                for conn in connections:
                    loaded_models.update(conn.get_loaded_models())
            
            # Update stats
            stats = {
                **self.stats,
                'current_connections': total_connections,
                'connection_counts': connection_counts,
                'loaded_models': len(loaded_models),
                'model_count': len(self.model_configs)
            }
            
            return stats
    
    def register_model(self, model_config: Dict[str, Any]) -> bool:
        """
        Register a model configuration.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            True if registration succeeded, False otherwise
        """
        with self._lock:
            model_id = model_config.get('model_id')
            if not model_id:
                logger.error("Model configuration missing model_id")
                return False
            
            # Store model config
            self.model_configs[model_id] = model_config
            
            # Log registration with model details
            model_name = model_config.get('model_name', 'unknown')
            model_type = model_config.get('model_type', 'unknown')
            platform = model_config.get('platform', model_config.get('backend', 'unknown'))
            
            logger.info(f"Registered model {model_id} ({model_name}, type: {model_type}, platform: {platform})")
            
            # Pre-load model if requested
            if model_config.get('preload', False):
                logger.info(f"Pre-loading model {model_id}")
                asyncio.run_coroutine_threadsafe(self.load_model(model_id), self.loop)
            
            return True
    
    def register_models(self, model_configs: List[Dict[str, Any]]) -> int:
        """
        Register multiple model configurations.
        
        Args:
            model_configs: List of model configuration dictionaries
            
        Returns:
            Number of successfully registered models
        """
        with self._lock:
            success_count = 0
            for config in model_configs:
                if self.register_model(config):
                    success_count += 1
            
            logger.info(f"Registered {success_count}/{len(model_configs)} models")
            return success_count
    
    def has_model(self, model_id: str) -> bool:
        """Check if model is registered."""
        with self._lock:
            return model_id in self.model_configs
            
    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model configuration.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model configuration dictionary or None if not found
        """
        with self._lock:
            return self.model_configs.get(model_id)

# Forward declaration of classes to fix circular reference
class EnhancedWebModel:
    pass

class ResourcePoolBridgeIntegration:
    """
    Integration with global resource pool for WebNN/WebGPU hardware acceleration.
    
    This class provides integration between the global resource pool and
    the ResourcePoolBridge for WebNN and WebGPU platforms, enabling
    efficient model execution across browser backends.
    
    Key features:
    - Support for concurrent model execution across WebGPU and WebNN backends
    - Efficient resource allocation for heterogeneous execution
    - Connection pooling for Selenium browser instances
    - Optimized browser selection based on model type (Firefox for audio, Edge for text)
    - Resource monitoring and adaptive scaling
    - IPFS acceleration integration
    - Database integration for result storage and analysis
    """
    
    def __init__(self, max_connections: int = 4, 
                 enable_gpu: bool = True, enable_cpu: bool = True,
                 headless: bool = True, browser_preferences: Dict[str, str] = None,
                 adaptive_scaling: bool = True, monitoring_interval: int = 60,
                 enable_ipfs: bool = True, db_path: str = None,
                 enable_heartbeat: bool = True, priority_families: List[str] = None):
        """
        Initialize resource pool bridge integration with WebNN/WebGPU support.
        
        Args:
            max_connections: Maximum number of concurrent browser connections
            enable_gpu: Enable GPU (WebGPU) backend
            enable_cpu: Enable CPU backend
            headless: Whether to run browsers in headless mode
            browser_preferences: Dict mapping model families to preferred browsers
            adaptive_scaling: Whether to enable adaptive connection scaling
            monitoring_interval: Interval for resource monitoring in seconds
            enable_ipfs: Whether to enable IPFS integration
            db_path: Path to database for storing results (DuckDB)
            enable_heartbeat: Whether to enable heartbeat for connection health
            priority_families: List of model families to prioritize in resource allocation
        """
        self.resource_pool = get_global_resource_pool()
        self.bridge = None
        self.max_connections = max_connections
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu
        self.headless = headless
        self.initialized = False
        
        # Set browser preferences with specialized optimizations
        self.browser_preferences = browser_preferences or {
            'audio': 'firefox',  # Firefox has better WebGPU compute shader performance for audio
            'vision': 'chrome',  # Chrome has good WebGPU support for vision models
            'text_embedding': 'edge',  # Edge has excellent WebNN support for text embeddings
            'text': 'edge',      # Edge works well for text models
            'multimodal': 'chrome',  # Chrome is good for multimodal models
            'speech': 'firefox', # Firefox optimized for speech models
            'text_generation': 'chrome' # Chrome good balance for generation
        }
        
        # Dynamic resource management settings
        self.adaptive_scaling = adaptive_scaling
        self.monitoring_interval = monitoring_interval
        self.enable_heartbeat = enable_heartbeat
        self.priority_families = priority_families or ['audio', 'text_embedding', 'vision', 'multimodal']
        
        # IPFS acceleration integration
        self.enable_ipfs = enable_ipfs
        self.db_path = db_path or os.environ.get("BENCHMARK_DB_PATH")
        self.ipfs_module = None
        
        # Initialize IPFS acceleration if enabled
        if self.enable_ipfs:
            try:
                import ipfs_accelerate_impl
                self.ipfs_module = ipfs_accelerate_impl
                logger.info("IPFS acceleration module imported successfully")
                
                # Initialize IPFS module settings for optimal performance
                if hasattr(ipfs_accelerate_impl, 'initialize_acceleration'):
                    ipfs_accelerate_impl.initialize_acceleration(
                        enable_browser_optimizations=True,
                        enable_p2p_optimization=True,
                        cache_size_mb=1024,  # 1GB cache by default
                        web_optimization=True
                    )
                    logger.info("IPFS acceleration initialized with browser optimizations")
            except ImportError:
                logger.warning("IPFS acceleration module not available")
            except Exception as e:
                logger.warning(f"Error initializing IPFS acceleration: {e}")
        
        # Initialize database connection
        self.db_connection = None
        if self.db_path:
            try:
                import duckdb
                self.db_connection = duckdb.connect(self.db_path)
                logger.info(f"Connected to database: {self.db_path}")
                self._ensure_db_schema()
            except ImportError:
                logger.warning("DuckDB not available - install with: pip install duckdb")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
        
        # Browser connection management
        self.browser_connections = {}
        self.connection_status = {}
        self.connection_health = {}
        self.available_browsers = self._detect_available_browsers()
        
        # Get list of preferred browsers based on detected browser capabilities
        self.preferred_browser_order = self._get_preferred_browser_order()
        
        # System characteristics
        self.system_info = self._collect_system_info()
        
        # Concurrent execution and resource management
        self.execution_queue = None
        self.executor_thread = None
        self.monitoring_task = None
        self.connection_task = None
        
        # Comprehensive execution statistics with detailed metrics
        self.execution_stats = {
            # Basic execution statistics
            'executed_tasks': 0,
            'total_execution_time': 0,
            'errors': 0,
            'concurrent_peak': 0,
            'queue_peak': 0,
            
            # Detailed model tracking
            'model_execution_times': {},     # Per model execution timing
            'model_memory_usage': {},        # Memory usage by model
            'model_execution_count': {},     # Count of executions by model
            'model_success_rate': {},        # Success rate by model
            
            # Family-level metrics
            'family_execution_times': {},    # Execution times by model family
            'family_execution_count': {},    # Count by model family
            
            # Browser-specific metrics
            'browser_execution_times': {},   # Execution times by browser
            'browser_success_count': {},     # Successful executions by browser
            'browser_error_count': {},       # Errors by browser
            
            # IPFS acceleration metrics
            'ipfs_acceleration_count': 0,    # Number of IPFS-accelerated requests
            'ipfs_cache_hits': 0,            # Number of IPFS cache hits
            'ipfs_latency_reduction': {},    # Latency reduction by model with IPFS
            
            # Detailed timing metrics
            'avg_startup_time': 0,           # Average model startup time
            'avg_inference_time': 0,         # Average inference time
            'avg_time_to_first_token': 0,    # Average time to first token (for generation)
            
            # Parallel execution metrics
            'parallel_executions': 0,        # Count of parallel execution batches
            'parallel_tasks': 0,             # Total tasks executed in parallel
            'parallel_success': 0,           # Successful parallel tasks
            'avg_parallel_batch_size': 0,    # Average parallel batch size
            'largest_parallel_batch': 0      # Largest parallel batch processed
        }
        
        # Detailed resource utilization metrics
        self.resource_metrics = {
            # Hardware utilization
            'webgpu_util': 0.0,
            'webnn_util': 0.0,
            'cpu_util': 0.0,
            'memory_usage': 0.0,
            
            # Connection metrics
            'connection_util': 0.0,
            'active_connections': 0,
            'active_models': 0, 
            'models_per_connection': 0.0,
            'queue_size': 0,
            
            # Browser usage tracking
            'browser_usage': {
                'chrome': 0,
                'firefox': 0,
                'edge': 0,
                'safari': 0
            },
            
            # Platform-specific metrics
            'platform_usage': {
                'webgpu': 0,
                'webnn': 0,
                'cpu': 0
            },
            
            # IPFS statistics
            'ipfs_cache_size': 0,
            'ipfs_cache_hit_rate': 0.0,
            
            # Performance metrics
            'avg_latency_ms': 0.0,
            'avg_throughput': 0.0,
            'peak_memory_mb': 0.0,
            'energy_efficiency': 0.0
        }
        
        # Shutdown flag for clean termination
        self._is_shutting_down = False
        
        # Create event loop for async operations
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        logger.info(f"ResourcePoolBridgeIntegration created with max_connections={max_connections}, " +
                   f"adaptive_scaling={'enabled' if adaptive_scaling else 'disabled'}, " +
                   f"IPFS={'enabled' if enable_ipfs else 'disabled'}")
                   
    def _get_preferred_browser_order(self) -> List[str]:
        """
        Determines the preferred order of browsers based on detected capabilities.
        
        Returns:
            List of browser names in order of preference
        """
        # Start with empty list
        browser_order = []
        
        # First include all detected browsers sorted by priority
        priority_browsers = sorted(
            self.available_browsers.items(),
            key=lambda x: x[1].get("priority", 999)
        )
        
        for browser, info in priority_browsers:
            browser_order.append(browser)
        
        # If we don't have any browsers, add default fallbacks
        if not browser_order:
            browser_order = ["chrome", "firefox", "edge", "safari"]
            
        logger.debug(f"Preferred browser order: {', '.join(browser_order)}")
        return browser_order
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Collects information about the current system.
        
        Returns:
            Dictionary with system information
        """
        from datetime import datetime
        
        system_info = {
            "platform": platform_module.platform(),
            "python_version": platform_module.python_version(),
            "processor": platform_module.processor(),
            "memory": self._get_system_memory(),
            "browsers": list(self.available_browsers.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add GPU information if available
        try:
            # Try to get GPU info on different platforms
            if platform_module.system() == "Linux":
                # Try to get GPU info from lspci
                try:
                    proc = subprocess.Popen(
                        ["lspci", "-v"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, _ = proc.communicate(timeout=2.0)
                    
                    # Extract relevant GPU information
                    gpu_info = []
                    for line in stdout.splitlines():
                        if "VGA" in line or "3D" in line or "Display" in line:
                            gpu_info.append(line.strip())
                    
                    if gpu_info:
                        system_info["gpu"] = gpu_info
                except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception):
                    pass
            
            elif platform_module.system() == "Darwin":
                # On macOS, use system_profiler
                try:
                    proc = subprocess.Popen(
                        ["system_profiler", "SPDisplaysDataType"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, _ = proc.communicate(timeout=2.0)
                    
                    # Extract GPU model
                    match = re.search(r"Chipset Model: (.+)", stdout)
                    if match:
                        system_info["gpu"] = [match.group(1).strip()]
                except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception):
                    pass
            
            elif platform_module.system() == "Windows":
                # On Windows, use wmic
                try:
                    proc = subprocess.Popen(
                        ["wmic", "path", "win32_VideoController", "get", "Name"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, _ = proc.communicate(timeout=2.0)
                    
                    # Extract GPU names
                    lines = stdout.strip().splitlines()
                    if len(lines) > 1:
                        # Skip the header line "Name"
                        gpu_info = [line.strip() for line in lines[1:] if line.strip()]
                        if gpu_info:
                            system_info["gpu"] = gpu_info
                except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception):
                    pass
        except Exception as e:
            logger.debug(f"Error collecting GPU information: {e}")
            
        return system_info
    
    def _get_system_memory(self) -> Optional[float]:
        """
        Get system total memory in GB.
        
        Returns:
            Memory in GB or None if unable to detect
        """
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            return round(memory_info.total / (1024**3), 1)  # Convert to GB and round
        except ImportError:
            # psutil not available, try platform-specific methods
            try:
                if platform_module.system() == "Linux":
                    # Read from /proc/meminfo
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if "MemTotal" in line:
                                # Extract KB value and convert to GB
                                kb = int(line.split()[1])
                                return round(kb / (1024**2), 1)
                
                elif platform_module.system() == "Darwin":
                    # Use sysctl
                    proc = subprocess.Popen(
                        ["sysctl", "-n", "hw.memsize"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, _ = proc.communicate(timeout=1.0)
                    bytes_value = int(stdout.strip())
                    return round(bytes_value / (1024**3), 1)
                
                elif platform_module.system() == "Windows":
                    # Use wmic
                    proc = subprocess.Popen(
                        ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, _ = proc.communicate(timeout=1.0)
                    lines = stdout.strip().splitlines()
                    if len(lines) > 1:
                        bytes_value = int(lines[1].strip())
                        return round(bytes_value / (1024**3), 1)
            except Exception:
                pass
                
        return None
    
    def _ensure_db_schema(self):
        """Ensure the database has the required tables for storing results."""
        if not self.db_connection:
            return
            
        try:
            # Create table for WebNN/WebGPU acceleration results
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS webnn_webgpu_results (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                model_name VARCHAR,
                model_type VARCHAR,
                platform VARCHAR,
                browser VARCHAR,
                is_real_implementation BOOLEAN,
                is_simulation BOOLEAN,
                precision INTEGER,
                mixed_precision BOOLEAN,
                ipfs_accelerated BOOLEAN,
                ipfs_cache_hit BOOLEAN,
                compute_shader_optimized BOOLEAN,
                precompile_shaders BOOLEAN,
                parallel_loading BOOLEAN,
                latency_ms FLOAT,
                throughput_items_per_sec FLOAT,
                memory_usage_mb FLOAT,
                energy_efficiency_score FLOAT,
                adapter_info JSON,
                system_info JSON,
                details JSON
            )
            """)
            
            # Create table for browser connection metrics
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS browser_connection_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                browser_name VARCHAR,
                platform VARCHAR,
                connection_id VARCHAR,
                connection_duration_sec FLOAT,
                models_executed INTEGER,
                total_inference_time_sec FLOAT,
                error_count INTEGER,
                connection_success BOOLEAN,
                heartbeat_failures INTEGER,
                browser_version VARCHAR,
                adapter_info JSON,
                backend_info JSON
            )
            """)
            
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Error ensuring database schema: {e}")
    
    def _detect_available_browsers(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect available browsers on the system.
        
        Returns:
            Dictionary mapping browser names to browser information
        """
        browsers = {}
        
        # Check for Chrome
        chrome_path = self._find_browser_path("chrome")
        if chrome_path:
            browsers["chrome"] = {
                "name": "Google Chrome",
                "path": chrome_path,
                "webgpu_support": True,
                "webnn_support": True,
                "priority": 1
            }
        
        # Check for Firefox
        firefox_path = self._find_browser_path("firefox")
        if firefox_path:
            browsers["firefox"] = {
                "name": "Mozilla Firefox",
                "path": firefox_path,
                "webgpu_support": True,
                "webnn_support": False,  # Firefox WebNN support is limited
                "priority": 2,
                "audio_optimized": True  # Firefox has special optimization for audio models
            }
        
        # Check for Edge
        edge_path = self._find_browser_path("edge")
        if edge_path:
            browsers["edge"] = {
                "name": "Microsoft Edge",
                "path": edge_path,
                "webgpu_support": True,
                "webnn_support": True,  # Edge has the best WebNN support
                "priority": 3
            }
        
        # Check for Safari (macOS only)
        if platform_module.system() == "Darwin":
            safari_path = "/Applications/Safari.app/Contents/MacOS/Safari"
            if os.path.exists(safari_path):
                browsers["safari"] = {
                    "name": "Apple Safari",
                    "path": safari_path,
                    "webgpu_support": True,
                    "webnn_support": True,
                    "priority": 4
                }
        
        logger.info(f"Detected browsers: {', '.join(browsers.keys())}")
        return browsers
    
    def _find_browser_path(self, browser_name: str) -> Optional[str]:
        """
        Find the path to a browser executable.
        
        Args:
            browser_name: Name of the browser
            
        Returns:
            Path to browser executable or None if not found
        """
        system = platform_module.system()
        
        if system == "Windows":
            if browser_name == "chrome":
                paths = [
                    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
                    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
                    os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe")
                ]
            elif browser_name == "firefox":
                paths = [
                    os.path.expandvars(r"%ProgramFiles%\Mozilla Firefox\firefox.exe"),
                    os.path.expandvars(r"%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe")
                ]
            elif browser_name == "edge":
                paths = [
                    os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
                    os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe")
                ]
            else:
                return None
        
        elif system == "Darwin":  # macOS
            if browser_name == "chrome":
                paths = [
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                ]
            elif browser_name == "firefox":
                paths = [
                    "/Applications/Firefox.app/Contents/MacOS/firefox"
                ]
            elif browser_name == "edge":
                paths = [
                    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
                ]
            else:
                return None
        
        elif system == "Linux":
            if browser_name == "chrome":
                paths = [
                    "/usr/bin/google-chrome",
                    "/usr/bin/google-chrome-stable",
                    "/usr/bin/chromium-browser",
                    "/usr/bin/chromium"
                ]
            elif browser_name == "firefox":
                paths = [
                    "/usr/bin/firefox"
                ]
            elif browser_name == "edge":
                paths = [
                    "/usr/bin/microsoft-edge",
                    "/usr/bin/microsoft-edge-stable"
                ]
            else:
                return None
        
        else:
            return None
        
        # Check each path
        for path in paths:
            if os.path.exists(path):
                return path
        
        return None
        
    def _get_browser_version(self, browser_path: str, browser_type: str) -> Optional[str]:
        """
        Attempt to get browser version.
        
        Args:
            browser_path: Path to browser executable
            browser_type: Type of browser ('chrome', 'firefox', 'edge', 'safari')
            
        Returns:
            Version string or None if unable to detect
        """
        if not browser_path or not os.path.exists(browser_path):
            return None
            
        try:
            version_arg = "--version"
            if browser_type == "safari":
                # Safari doesn't support version command line argument
                return None
                
            # Run browser with version argument
            proc = subprocess.Popen(
                [browser_path, version_arg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = proc.communicate(timeout=2.0)
            output = stdout or stderr
            
            # Extract version number
            if browser_type == "chrome":
                match = re.search(r'Chrome\s+(\d+\.\d+\.\d+\.\d+)', output)
                if match:
                    return match.group(1)
            elif browser_type == "firefox":
                match = re.search(r'Firefox\s+(\d+\.\d+)', output)
                if match:
                    return match.group(1)
            elif browser_type == "edge":
                match = re.search(r'Microsoft Edge\s+(\d+\.\d+\.\d+\.\d+)', output)
                if match:
                    return match.group(1)
                    
            # If no specific pattern matched but output contains digits, return that
            digits_match = re.search(r'(\d+\.\d+[\.\d]*)', output)
            if digits_match:
                return digits_match.group(1)
                
            return None
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception) as e:
            logger.debug(f"Failed to get browser version for {browser_type}: {e}")
            return None
    
    def initialize(self):
        """Initialize the resource pool bridge integration with enhanced concurrency support."""
        if self.initialized:
            return
        
        # Import necessary modules
        try:
            from datetime import datetime  # For timestamps
        except ImportError:
            pass
        
        # Create bridge instance with browser preferences
        browser_to_use = os.environ.get('TEST_BROWSER', None)
        
        # If we have a requested browser from os.environ, check if it's available
        if browser_to_use and browser_to_use not in self.available_browsers:
            if self.available_browsers:
                logger.warning(f"Requested browser {browser_to_use} not found. Using available browser instead.")
                browser_to_use = next(iter(self.available_browsers.keys()))
            else:
                logger.warning(f"Requested browser {browser_to_use} not found and no browsers available.")
                browser_to_use = 'chrome'  # Default fallback
        
        # If no browser specified in environment, use first available or default to chrome
        if not browser_to_use:
            if self.available_browsers:
                browser_to_use = next(iter(self.available_browsers.keys()))
            else:
                browser_to_use = 'chrome'  # Default fallback
        
        # Create the bridge with optimized settings
        self.bridge = ResourcePoolBridge(
            max_connections=self.max_connections,
            browser=browser_to_use,
            enable_gpu=self.enable_gpu,
            enable_cpu=self.enable_cpu,
            headless=self.headless,
            cleanup_interval=300  # 5 minutes cleanup interval
        )
        
        # Initialize bridge
        logger.info(f"Initializing ResourcePoolBridge with browser: {browser_to_use}")
        self.loop.run_until_complete(self.bridge.initialize())
        
        # Initialize concurrent execution queue with appropriate size
        queue_size = max(20, self.max_connections * 5)  # Size queue based on connection count
        self.execution_queue = asyncio.Queue(maxsize=queue_size)
        
        # Start executor thread for concurrent model execution
        self._start_executor()
        
        # Start resource monitoring if adaptive scaling is enabled
        if self.adaptive_scaling:
            self._start_monitoring()
        
        # Start connection management task
        self._start_connection_manager()
        
        # Pre-initialize connections based on browser availability and model type preferences
        if self.enable_gpu:
            self._pre_initialize_connections()
        
        # Update system information after initialization
        self.system_info.update({
            "initialization_time": datetime.now().isoformat(),
            "active_connections": len(self.browser_connections),
            "browser_in_use": browser_to_use
        })
        
        # Register with global resource pool for other components to use
        if self.resource_pool:
            self.resource_pool.register_resource("web_platform_integration", self)
            logger.info("Registered with global resource pool as 'web_platform_integration'")
        
        self.initialized = True
        
        # Detailed initialization log with environment information
        logger.info(f"ResourcePoolBridgeIntegration initialized with {self.max_connections} connections")
        logger.info(f"Using browser: {browser_to_use}, WebGPU: {'enabled' if self.enable_gpu else 'disabled'}, "
                   f"WebNN: {'enabled' if self.enable_cpu else 'disabled'}, "
                   f"IPFS: {'enabled' if self.enable_ipfs else 'disabled'}")
        
        # Log active browsers and capabilities
        if self.available_browsers:
            logger.info(f"Available browsers: {', '.join(self.available_browsers.keys())}")
        else:
            logger.warning("No browsers detected - simulation mode will be used")
    
    def _pre_initialize_connections(self):
        """Pre-initialize browser connections for faster model loading."""
        # Initialize connections for each available browser based on priority
        for browser_name, browser_info in sorted(
            self.available_browsers.items(), 
            key=lambda x: x[1].get('priority', 999)
        ):
            # Skip if browser doesn't support required platforms
            if not browser_info.get('webgpu_support') and not browser_info.get('webnn_support'):
                continue
                
            # Determine which platform to use
            platform = None
            if browser_info.get('webgpu_support') and self.enable_gpu:
                platform = 'webgpu'
            elif browser_info.get('webnn_support') and self.enable_cpu:
                platform = 'webnn'
                
            if platform:
                logger.info(f"Pre-initializing {platform} connection for {browser_name}")
                # Create task to initialize connection
                asyncio.run_coroutine_threadsafe(
                    self._initialize_browser_connection(browser_name, platform),
                    self.loop
                )
    
    async def _initialize_browser_connection(self, browser_name: str, platform: str) -> Optional[str]:
        """
        Initialize a browser connection for the specified browser and platform.
        
        Args:
            browser_name: Name of the browser ('chrome', 'firefox', 'edge', 'safari')
            platform: Platform to use ('webgpu' or 'webnn')
            
        Returns:
            Connection ID or None if initialization failed
        """
        if not self.bridge:
            logger.error("Bridge not initialized")
            return None
            
        try:
            # Get browser-specific options
            compute_shaders = browser_name == 'firefox' and platform == 'webgpu'
            precompile_shaders = platform == 'webgpu'
            parallel_loading = platform == 'webgpu'
            
            # Set environment variables for Firefox audio optimizations
            if browser_name == 'firefox' and compute_shaders:
                os.environ["USE_FIREFOX_WEBGPU"] = "1"
                os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
                os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
                logger.info("Enabled Firefox audio optimizations (256x1x1 workgroup size)")
            
            # Create unique connection ID
            connection_id = f"{browser_name}_{platform}_{int(time.time() * 1000)}"
            
            # Initialize connection with browser-specific settings
            connection = await self.bridge.get_connection(
                platform=platform,
                browser=browser_name
            )
            
            if connection:
                # Store connection in mapping
                self.browser_connections[connection_id] = {
                    'connection': connection,
                    'browser_name': browser_name,
                    'platform': platform,
                    'compute_shaders': compute_shaders,
                    'precompile_shaders': precompile_shaders,
                    'parallel_loading': parallel_loading,
                    'created_at': time.time(),
                    'last_used': time.time(),
                    'models_executed': 0
                }
                
                # Update metrics
                self.resource_metrics['browser_usage'][browser_name] += 1
                
                logger.info(f"Successfully initialized {platform} connection for {browser_name}")
                return connection_id
            else:
                logger.error(f"Failed to initialize {platform} connection for {browser_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error initializing {platform} connection for {browser_name}: {e}")
            return None
    
    def _start_connection_manager(self):
        """Start the connection management task."""
        async def connection_management_loop():
            """Manage browser connections to ensure availability."""
            while not self._is_shutting_down:
                try:
                    # Check connection health every 30 seconds
                    await asyncio.sleep(30)
                    
                    # Get total active connections
                    active_connections = len(self.browser_connections)
                    browser_counts = {}
                    for browser_name in self.available_browsers:
                        browser_counts[browser_name] = 0
                    
                    # Count connections per browser
                    for conn_info in self.browser_connections.values():
                        browser_name = conn_info.get('browser_name')
                        if browser_name in browser_counts:
                            browser_counts[browser_name] += 1
                    
                    # Check if we need to initialize more connections
                    for browser_name, count in browser_counts.items():
                        # If no connections for this browser and total connections < max_connections
                        if count == 0 and active_connections < self.max_connections:
                            browser_info = self.available_browsers.get(browser_name)
                            if not browser_info:
                                continue
                                
                            # Determine platform
                            platform = None
                            if browser_info.get('webgpu_support') and self.enable_gpu:
                                platform = 'webgpu'
                            elif browser_info.get('webnn_support') and self.enable_cpu:
                                platform = 'webnn'
                                
                            if platform:
                                logger.info(f"Initializing missing {platform} connection for {browser_name}")
                                await self._initialize_browser_connection(browser_name, platform)
                                active_connections += 1
                    
                    # Update connection utilization metric
                    self.resource_metrics['connection_util'] = active_connections / self.max_connections
                    
                except asyncio.CancelledError:
                    logger.info("Connection management task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in connection management loop: {e}")
        
        # Start connection management task
        self.connection_task = asyncio.run_coroutine_threadsafe(
            connection_management_loop(),
            self.loop
        )
    
    def _start_executor(self):
        """Start the executor thread for processing concurrent model executions."""
        async def executor_loop():
            """Asyncio loop for executing concurrent model inference tasks."""
            while not self._is_shutting_down:
                try:
                    # Get next task from queue with timeout
                    try:
                        task = await asyncio.wait_for(self.execution_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    
                    # Unpack task
                    model_id, inputs, future = task
                    
                    # Execute task
                    start_time = time.time()
                    try:
                        result = await self.bridge.run_inference(model_id, inputs)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                        self.execution_stats['errors'] += 1
                        logger.error(f"Error executing model {model_id}: {e}")
                    
                    # Update stats
                    execution_time = time.time() - start_time
                    self.execution_stats['executed_tasks'] += 1
                    self.execution_stats['total_execution_time'] += execution_time
                    
                    # Track model-specific execution time
                    if model_id not in self.execution_stats['model_execution_times']:
                        self.execution_stats['model_execution_times'][model_id] = []
                    self.execution_stats['model_execution_times'][model_id].append(execution_time)
                    
                    # Limit history to last 100 executions
                    self.execution_stats['model_execution_times'][model_id] = \
                        self.execution_stats['model_execution_times'][model_id][-100:]
                    
                    # Mark task as done
                    self.execution_queue.task_done()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in executor loop: {e}")
                    traceback.print_exc()
        
        # Start executor loop in the event loop
        self.executor_future = asyncio.run_coroutine_threadsafe(executor_loop(), self.loop)
    
    def _start_monitoring(self):
        """Start resource monitoring for adaptive scaling."""
        async def monitoring_loop():
            """Asyncio loop for monitoring resource usage and adapting resources."""
            while not self._is_shutting_down:
                try:
                    # Wait for monitoring interval
                    await asyncio.sleep(self.monitoring_interval)
                    
                    # Get current stats from bridge
                    bridge_stats = self.bridge.get_stats()
                    
                    # Update resource metrics
                    total_connections = bridge_stats.get('current_connections', 0)
                    max_connections = self.max_connections
                    
                    self.resource_metrics['connection_util'] = total_connections / max_connections if max_connections > 0 else 0
                    
                    # Analyze queue backlog
                    queue_size = self.execution_queue.qsize()
                    self.execution_stats['queue_peak'] = max(self.execution_stats['queue_peak'], queue_size)
                    
                    # Implement adaptive scaling based on queue and resource metrics
                    self._adapt_resources(queue_size, self.resource_metrics['connection_util'])
                    
                    # Log monitoring stats at debug level
                    logger.debug(f"Resource monitoring: conn_util={self.resource_metrics['connection_util']:.2f}, "
                                f"queue={queue_size}, tasks={self.execution_stats['executed_tasks']}")
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
        
        # Start monitoring loop in the event loop
        self.monitoring_task = asyncio.run_coroutine_threadsafe(monitoring_loop(), self.loop)
    
    def _adapt_resources(self, queue_size: int, connection_util: float):
        """
        Adapt resources based on current usage and queue backlog.
        
        This method implements smart resource scaling to efficiently manage browser
        connections based on current workload:
        1. Scales up quickly when queue is growing to handle higher demand
        2. Scales down gradually when idle to optimize resource usage
        3. Takes into account both queue size and connection utilization
        4. Optimizes for browser-specific model types
        
        Args:
            queue_size: Current size of execution queue
            connection_util: Connection utilization (0.0-1.0)
        """
        # Only adapt if we have adaptive scaling enabled
        if not self.adaptive_scaling:
            return
        
        # Don't exceed max_connections
        current_max = self.bridge.max_connections
        
        # Get current browser distribution
        browser_counts = self.resource_metrics.get('browser_usage', {})
        
        # Count active connections and models by browser
        active_connections = 0
        active_models = 0
        browser_models = {}
        
        if self.bridge:
            # Aggregate stats
            for platform, connections in self.bridge.connections.items():
                for conn in connections:
                    if conn.is_healthy():
                        active_connections += 1
                        browser = conn.browser_name
                        model_count = len(conn.loaded_models)
                        active_models += model_count
                        
                        # Track models by browser
                        if browser not in browser_models:
                            browser_models[browser] = 0
                        browser_models[browser] += model_count
        
        # Scaling decision logic based on multiple factors
        scale_up = False
        scale_down = False
        scale_reason = ""
        
        # SCALE UP CONDITIONS
        # Check for high queue backlog
        if queue_size >= 8 and connection_util > 0.7:
            scale_up = True
            scale_reason = f"high queue backlog ({queue_size}) with high utilization ({connection_util:.2f})"
        # Check for moderate queue with very high utilization
        elif queue_size >= 3 and connection_util > 0.9:
            scale_up = True
            scale_reason = f"moderate queue ({queue_size}) with very high utilization ({connection_util:.2f})"
        # Check for high active model count per connection
        elif active_connections > 0 and active_models / active_connections > 5:
            scale_up = True
            scale_reason = f"high model density ({active_models / active_connections:.1f} models/conn)"
        
        # SCALE DOWN CONDITIONS
        # Only scale down if we're above the initial max_connections
        if current_max > self.max_connections:
            # Check for empty queue and low utilization over time
            if queue_size == 0 and connection_util < 0.3:
                scale_down = True
                scale_reason = f"empty queue with low utilization ({connection_util:.2f})"
            # Check for very low model count per connection
            elif active_connections > 2 and active_models / active_connections < 1.5:
                scale_down = True
                scale_reason = f"low model density ({active_models / active_connections:.1f} models/conn)"
        
        # Check if we need more browser-specific connections
        browser_scaling_needed = False
        target_browser = None
        
        # Get current execution stats to detect browser bottlenecks
        if hasattr(self, 'execution_stats') and 'model_execution_times' in self.execution_stats:
            # Analyze model types to identify browser-specific needs
            model_families = {}
            
            # Count models by family
            for model_id in self.execution_stats.get('model_execution_times', {}):
                if ':' in model_id:
                    family = model_id.split(':', 1)[0]
                    model_families[family] = model_families.get(family, 0) + 1
            
            # Check if we have a dominant model family that needs specific browser
            if model_families:
                dominant_family = max(model_families.items(), key=lambda x: x[1])[0]
                
                # Audio models need Firefox with compute shader optimizations
                if dominant_family == 'audio' and browser_models.get('firefox', 0) < 1:
                    browser_scaling_needed = True
                    target_browser = 'firefox'
                    scale_reason = f"need Firefox for audio models ({model_families.get('audio', 0)} models)"
                
                # Text models work best with Edge for WebNN
                elif dominant_family == 'text_embedding' and browser_models.get('edge', 0) < 1:
                    browser_scaling_needed = True
                    target_browser = 'edge'
                    scale_reason = f"need Edge for text models ({model_families.get('text_embedding', 0)} models)"
        
        # Apply scaling decisions
        if scale_up and current_max < 8:
            # Scale up more aggressively when needed
            new_max = min(current_max + 2, 8)
            logger.info(f"Scaling up max connections from {current_max} to {new_max} due to {scale_reason}")
            self.bridge.max_connections = new_max
            
            # Also trigger browser-specific connection creation if needed
            if browser_scaling_needed and target_browser:
                logger.info(f"Will create specialized connection for {target_browser} browser")
                # Store browser request for next connection creation
                os.environ["NEXT_BROWSER_REQUEST"] = target_browser
            
        elif scale_down:
            # Scale down conservatively (one at a time)
            new_max = max(current_max - 1, self.max_connections)
            logger.info(f"Scaling down max connections from {current_max} to {new_max} due to {scale_reason}")
            self.bridge.max_connections = new_max
        
        # Log detailed resource metrics at debug level
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Resource metrics: conn_util={connection_util:.2f}, queue={queue_size}, " +
                       f"active_conn={active_connections}, active_models={active_models}, " +
                       f"models_per_conn={active_models/active_connections if active_connections else 0:.1f}")
            logger.debug(f"Browser distribution: {browser_models}")
            
        # Update resource metrics for monitoring
        self.resource_metrics['connection_util'] = connection_util
        self.resource_metrics['active_connections'] = active_connections
        self.resource_metrics['active_models'] = active_models
        self.resource_metrics['models_per_connection'] = active_models / active_connections if active_connections else 0
        self.resource_metrics['queue_size'] = queue_size
    
    def get_model(self, model_type: str, model_name: str, constructor=None, hardware_preferences=None):
        """
        Get a model from the global resource pool with WebNN/WebGPU support.
        
        Args:
            model_type: Type of model (e.g., 'text_embedding', 'vision', 'audio')
            model_name: Name of the model
            constructor: Function to create the model if not present
            hardware_preferences: Hardware preferences
                - priority_list: List of hardware platforms in order of preference
                - browser: Preferred browser
                - precision: Preferred precision (4, 8, 16, 32)
                - mixed_precision: Whether to use mixed precision
                - enable_ipfs: Whether to enable IPFS acceleration
                - use_firefox_optimizations: Whether to use Firefox audio optimizations
                
        Returns:
            The loaded model with WebNN/WebGPU acceleration
        """
        # Ensure bridge is initialized
        if not self.initialized:
            self.initialize()
        
        # Default hardware preferences
        if hardware_preferences is None:
            hardware_preferences = {}
        
        # Default priority list
        if 'priority_list' not in hardware_preferences:
            hardware_preferences['priority_list'] = ['webgpu', 'webnn', 'cpu']
        
        # Check if WebGPU or WebNN are requested
        web_platforms = ['webgpu', 'webnn']
        if any(platform in hardware_preferences['priority_list'] for platform in web_platforms):
            # Get optimal browser for this model type
            browser = hardware_preferences.get('browser')
            if not browser:
                browser = self.browser_preferences.get(model_type, self.browser_preferences.get('text', 'chrome'))
            
            # Special case for audio models - prefer Firefox with compute shader optimizations
            if model_type == 'audio' and not browser:
                browser = 'firefox'
                # Enable Firefox audio optimizations if not explicitly disabled
                if 'use_firefox_optimizations' not in hardware_preferences:
                    hardware_preferences['use_firefox_optimizations'] = True
            
            # Create a WebNN/WebGPU model instance with IPFS acceleration if enabled
            return self._create_web_model(
                model_type=model_type, 
                model_name=model_name, 
                hardware_preferences={
                    **hardware_preferences,
                    'browser': browser,
                    'enable_ipfs': hardware_preferences.get('enable_ipfs', self.enable_ipfs)
                }
            )
        
        # Fallback to standard resource pool
        return self.resource_pool.get_model(model_type, model_name, constructor, hardware_preferences)
    
    def get_models_concurrent(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get multiple models concurrently using the resource pool.
        
        Args:
            model_configs: List of model configurations, each with:
                - model_type: Type of model
                - model_name: Name of the model
                - hardware_preferences: Hardware preferences (optional)
                - model_id: Unique ID for the model (optional, will be generated if not provided)
                
        Returns:
            Dictionary mapping model IDs to loaded models
        """
        # Ensure bridge is initialized
        if not self.initialized:
            self.initialize()
        
        # Prepare models list with IDs
        models_with_ids = []
        for config in model_configs:
            model_id = config.get('model_id')
            if not model_id:
                model_id = f"{config['model_type']}_{config['model_name']}_{int(time.time() * 1000)}"
                config['model_id'] = model_id
            models_with_ids.append((model_id, config))
        
        # Create models concurrently
        loaded_models = {}
        futures = []
        
        for model_id, config in models_with_ids:
            # Create future for model loading
            future = asyncio.run_coroutine_threadsafe(
                self._load_model_async(
                    model_type=config['model_type'],
                    model_name=config['model_name'],
                    hardware_preferences=config.get('hardware_preferences', {})
                ),
                self.loop
            )
            futures.append((model_id, future))
        
        # Wait for all futures
        for model_id, future in futures:
            try:
                model = future.result(timeout=120)  # 2 minute timeout for model loading
                if model:
                    loaded_models[model_id] = model
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {e}")
        
        return loaded_models
    
    def execute_concurrent(self, models_and_inputs: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Execute multiple models concurrently with efficient resource allocation.
        
        Args:
            models_and_inputs: List of (model_id, inputs) tuples
            
        Returns:
            List of results in the same order as inputs
        """
        # Ensure bridge is initialized
        if not self.initialized:
            self.initialize()
        
        # Create futures for each task
        futures = []
        for model_id, inputs in models_and_inputs:
            future = self.loop.create_future()
            futures.append(future)
            
            # Schedule task for execution
            asyncio.run_coroutine_threadsafe(
                self.execution_queue.put((model_id, inputs, future)),
                self.loop
            )
        
        # Update concurrent peak stat
        self.execution_stats['concurrent_peak'] = max(
            self.execution_stats['concurrent_peak'],
            len(futures)
        )
        
        # Wait for all futures to complete
        results = []
        for future in futures:
            try:
                # Wait for future with timeout
                result = self.loop.run_until_complete(
                    asyncio.wait_for(future, timeout=60.0)
                )
                results.append(result)
            except Exception as e:
                # Handle timeout or other errors
                logger.error(f"Error in concurrent execution: {e}")
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def _load_model_async(self, model_type: str, model_name: str, hardware_preferences: Dict[str, Any]):
        """
        Asynchronously load a model with WebNN/WebGPU acceleration.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            hardware_preferences: Hardware preferences
            
        Returns:
            Loaded model or None if loading failed
        """
        try:
            # Create the web model
            model = self._create_web_model(model_type, model_name, hardware_preferences)
            
            # Return the model
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def _create_web_model(self, model_type: str, model_name: str, hardware_preferences: Dict[str, Any]):
        """
        Create a WebNN/WebGPU model instance with IPFS acceleration integration.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            hardware_preferences: Hardware preferences
            
        Returns:
            WebModel instance
        """
        # Create unique model ID
        model_id = f"{model_type}:{model_name}"
        
        # Get platform from hardware preferences
        priority_list = hardware_preferences.get('priority_list', ['webgpu', 'webnn', 'cpu'])
        platform = next((p for p in priority_list if p in ['webgpu', 'webnn', 'cpu']), 'webgpu')
        
        # Determine model family for browser selection
        model_family = hardware_preferences.get('model_family', model_type)
        
        # Get browser preference
        browser = hardware_preferences.get('browser')
        if not browser:
            browser = self.browser_preferences.get(model_family, self.browser_preferences.get('text', 'chrome'))
        
        # Determine optimal precision
        precision = hardware_preferences.get('precision', 16)
        mixed_precision = hardware_preferences.get('mixed_precision', False)
        
        # Determine if we should use Firefox optimizations for audio models
        use_firefox_optimizations = hardware_preferences.get(
            'use_firefox_optimizations', 
            browser == 'firefox' and model_family == 'audio'
        )
        
        # Configure optimizations
        optimizations = {
            'compute_shaders': use_firefox_optimizations or model_family in ['audio', 'speech'],
            'precompile_shaders': True,
            'parallel_loading': model_family in ['multimodal', 'vision_language'],
            'precision': precision,
            'mixed_precision': mixed_precision
        }
        
        # Configure IPFS acceleration if enabled
        enable_ipfs = hardware_preferences.get('enable_ipfs', self.enable_ipfs)
        ipfs_config = None
        
        if enable_ipfs and self.ipfs_module:
            ipfs_config = {
                'model_name': model_name,
                'model_type': model_type,
                'platform': platform,
                'browser': browser,
                'p2p_optimization': True,
                'precision': precision,
                'mixed_precision': mixed_precision,
                'use_firefox_optimizations': use_firefox_optimizations
            }
        
        # Determine model URL with fallback options
        model_path = f"https://huggingface.co/{model_name}/resolve/main/model.onnx"
        
        # Register model with bridge
        model_config = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'backend': platform,
            'family': model_family,
            'model_path': model_path,
            'browser': browser,
            'optimizations': optimizations,
            'ipfs_config': ipfs_config,
            'precision': precision,
            'mixed_precision': mixed_precision
        }
        
        self.bridge.register_model(model_config)
        
        # Create enhanced web model instance with IPFS acceleration
        return EnhancedWebModelWithIPFS(
            model_id=model_id,
            model_type=model_type,
            model_name=model_name,
            bridge=self.bridge,
            platform=platform,
            loop=self.loop,
            integration=self,
            family=model_family,
            ipfs_module=self.ipfs_module if enable_ipfs else None,
            browser=browser,
            precision=precision,
            mixed_precision=mixed_precision,
            use_firefox_optimizations=use_firefox_optimizations,
            db_connection=self.db_connection
        )
    
    def get_optimal_browser_connection(self, model_type: str, platform: str = 'webgpu') -> Tuple[Optional[str], Optional[Dict]]:
        """
        Get the optimal browser connection for a model type and platform.
        
        Args:
            model_type: Type of model ('text', 'vision', 'audio', etc.)
            platform: Platform to use ('webgpu' or 'webnn')
            
        Returns:
            Tuple of (connection_id, connection_info) or (None, None) if no connection available
        """
        # Determine preferred browser for this model type
        preferred_browser = self.browser_preferences.get(model_type, 'chrome')
        
        # First, try to find a connection with the preferred browser
        for conn_id, conn_info in self.browser_connections.items():
            if (conn_info['browser_name'] == preferred_browser and 
                conn_info['platform'] == platform and 
                not conn_info['connection'].is_busy()):
                return conn_id, conn_info
        
        # If not found, try any available connection with the right platform
        for conn_id, conn_info in self.browser_connections.items():
            if conn_info['platform'] == platform and not conn_info['connection'].is_busy():
                return conn_id, conn_info
        
        # No suitable connection found
        return None, None
    
    def store_acceleration_result(self, result: Dict[str, Any]) -> bool:
        """
        Store acceleration result in the database.
        
        Args:
            result: Inference result with acceleration metrics
            
        Returns:
            True if storage succeeded, False otherwise
        """
        if not self.db_connection:
            return False
            
        try:
            # Extract values from result
            timestamp = result.get('timestamp', datetime.now().isoformat())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            
            # Get metrics
            model_name = result.get('model_name', 'unknown')
            model_type = result.get('model_type', 'unknown')
            platform = result.get('platform', 'unknown')
            browser = result.get('browser', 'unknown')
            is_real = result.get('is_real_implementation', False)
            is_simulation = result.get('is_simulation', not is_real)
            precision = result.get('precision', 16)
            mixed_precision = result.get('mixed_precision', False)
            
            # Get IPFS-specific metrics
            ipfs_accelerated = result.get('ipfs_accelerated', False)
            ipfs_cache_hit = result.get('ipfs_cache_hit', False)
            
            # Get optimization flags
            compute_shader_optimized = result.get('compute_shader_optimized', False)
            precompile_shaders = result.get('precompile_shaders', False)
            parallel_loading = result.get('parallel_loading', False)
            
            # Get performance metrics
            metrics = result.get('metrics', {})
            latency_ms = metrics.get('latency_ms', 0)
            throughput = metrics.get('throughput_items_per_sec', 0)
            memory_mb = metrics.get('memory_usage_mb', 0)
            energy_score = metrics.get('energy_efficiency_score', 0)
            
            # Get hardware info
            adapter_info = result.get('adapter_info', {})
            system_info = result.get('system_info', {})
            
            # Insert into database
            self.db_connection.execute("""
            INSERT INTO webnn_webgpu_results (
                timestamp,
                model_name,
                model_type,
                platform,
                browser,
                is_real_implementation,
                is_simulation,
                precision,
                mixed_precision,
                ipfs_accelerated,
                ipfs_cache_hit,
                compute_shader_optimized,
                precompile_shaders,
                parallel_loading,
                latency_ms,
                throughput_items_per_sec,
                memory_usage_mb,
                energy_efficiency_score,
                adapter_info,
                system_info,
                details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                timestamp,
                model_name,
                model_type,
                platform,
                browser,
                is_real,
                is_simulation,
                precision,
                mixed_precision,
                ipfs_accelerated,
                ipfs_cache_hit,
                compute_shader_optimized,
                precompile_shaders,
                parallel_loading,
                latency_ms,
                throughput,
                memory_mb,
                energy_score,
                json.dumps(adapter_info),
                json.dumps(system_info),
                json.dumps(result)
            ])
            
            logger.info(f"Stored acceleration result in database: {model_name} ({platform}/{browser})")
            return True
            
        except Exception as e:
            logger.error(f"Error storing acceleration result in database: {e}")
            return False


class EnhancedWebModelWithIPFS(EnhancedWebModel):
    """
    Enhanced WebNN/WebGPU model with IPFS acceleration integration.
    
    This class extends EnhancedWebModel with integrated IPFS acceleration
    to provide optimized inference with WebNN/WebGPU and efficient content delivery.
    """
    
    def __init__(self, model_id: str, model_type: str, model_name: str,
                 bridge: ResourcePoolBridge, platform: str, loop,
                 integration=None, family: str = None, ipfs_module=None,
                 browser: str = 'chrome', precision: int = 16, 
                 mixed_precision: bool = False, use_firefox_optimizations: bool = False,
                 db_connection=None):
        """
        Initialize EnhancedWebModelWithIPFS.
        
        Args:
            model_id: Unique model ID
            model_type: Type of model
            model_name: Name of the model
            bridge: ResourcePoolBridge instance
            platform: 'webgpu', 'webnn', or 'cpu'
            loop: Asyncio event loop
            integration: ResourcePoolBridgeIntegration instance
            family: Model family
            ipfs_module: IPFS acceleration module
            browser: Browser to use
            precision: Precision to use (4, 8, 16, 32)
            mixed_precision: Whether to use mixed precision
            use_firefox_optimizations: Whether to use Firefox audio optimizations
            db_connection: Database connection for result storage
        """
        super().__init__(model_id, model_type, model_name, bridge, platform, loop, integration, family)
        self.ipfs_module = ipfs_module
        self.browser = browser
        self.precision = precision
        self.mixed_precision = mixed_precision
        self.use_firefox_optimizations = use_firefox_optimizations
        self.db_connection = db_connection
        self.ipfs_enabled = ipfs_module is not None
        self.ipfs_cache_hits = 0
        
        # Optimization flags
        self.compute_shader_optimized = use_firefox_optimizations or (browser == 'firefox' and family == 'audio')
        self.precompile_shaders = True
        self.parallel_loading = family in ['multimodal', 'vision_language']
        
        # System info
        self.system_info = {
            'platform': platform_module.platform(),
            'processor': platform_module.processor(),
            'python_version': platform_module.python_version()
        }
        
        logger.info(f"Created EnhancedWebModelWithIPFS for {model_name} ({platform}/{browser}) with IPFS {'enabled' if self.ipfs_enabled else 'disabled'}")
    
    def __call__(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run inference with IPFS acceleration and WebNN/WebGPU hardware.
        
        Args:
            inputs: Model inputs as dict or list of dicts for batched execution
            
        Returns:
            Inference results (single dict or list of dicts)
        """
        # Track execution stats
        start_time = time.time()
        self.execution_stats['total_calls'] += 1
        
        # Handle batched input
        is_batch = isinstance(inputs, list)
        batch_size = len(inputs) if is_batch else 1
        self.execution_stats['total_items'] += batch_size
        
        # Configure acceleration
        accel_config = {
            'platform': self.platform,
            'browser': self.browser,
            'precision': self.precision,
            'mixed_precision': self.mixed_precision,
            'use_firefox_optimizations': self.use_firefox_optimizations,
            'compute_shader_optimized': self.compute_shader_optimized,
            'precompile_shaders': self.precompile_shaders,
            'parallel_loading': self.parallel_loading
        }
        
        # Check if IPFS acceleration is available
        if self.ipfs_enabled and self.ipfs_module:
            try:
                # Use IPFS acceleration for the inference
                logger.info(f"Running acceleration with IPFS for {self.model_name}")
                
                if is_batch:
                    # Process batch with IPFS acceleration
                    results = []
                    for item in inputs:
                        result = self.ipfs_module.accelerate(self.model_name, item, accel_config)
                        results.append(result)
                else:
                    # Process single item with IPFS acceleration
                    results = self.ipfs_module.accelerate(self.model_name, inputs, accel_config)
                
                # Track IPFS cache hits
                if is_batch:
                    cache_hits = sum(1 for r in results if r.get('ipfs_cache_hit', False))
                    self.ipfs_cache_hits += cache_hits
                elif results.get('ipfs_cache_hit', False):
                    self.ipfs_cache_hits += 1
                
                # Store result in database if available
                if self.db_connection and hasattr(self.integration, 'store_acceleration_result'):
                    if is_batch:
                        for result in results:
                            self._prepare_and_store_result(result, batch_size)
                    else:
                        self._prepare_and_store_result(results, batch_size)
                
                # Update execution stats
                execution_time = time.time() - start_time
                self.execution_stats['total_time'] += execution_time
                self._update_performance_metrics(execution_time, batch_size)
                
                return results
                
            except Exception as e:
                logger.error(f"IPFS acceleration failed: {e}, falling back to standard inference")
                # Fall back to standard inference
        
        # If IPFS acceleration not available or failed, use standard WebNN/WebGPU inference
        try:
            # Run standard inference
            if is_batch:
                results = super()._process_batch(inputs)
            else:
                results = super().__call__(inputs)
            
            # Store result in database if available
            if self.db_connection and hasattr(self.integration, 'store_acceleration_result'):
                if is_batch:
                    for result in results:
                        self._prepare_and_store_result(result, batch_size, ipfs_accelerated=False)
                else:
                    self._prepare_and_store_result(results, batch_size, ipfs_accelerated=False)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running inference: {e}")
            # Return error result
            error_result = {
                'status': 'error',
                'error': str(e),
                'model_name': self.model_name,
                'model_type': self.model_type,
                'platform': self.platform,
                'browser': self.browser
            }
            return [error_result] * batch_size if is_batch else error_result
    
    def _prepare_and_store_result(self, result: Dict[str, Any], batch_size: int, ipfs_accelerated: bool = True):
        """
        Prepare and store result in database.
        
        Args:
            result: Inference result
            batch_size: Batch size used for inference
            ipfs_accelerated: Whether IPFS acceleration was used
        """
        # Add required fields if missing
        result.update({
            'model_name': self.model_name,
            'model_type': self.model_type or self.family,
            'platform': self.platform,
            'browser': self.browser,
            'precision': self.precision,
            'mixed_precision': self.mixed_precision,
            'ipfs_accelerated': ipfs_accelerated,
            'compute_shader_optimized': self.compute_shader_optimized,
            'precompile_shaders': self.precompile_shaders,
            'parallel_loading': self.parallel_loading,
            'system_info': self.system_info,
            'timestamp': datetime.now().isoformat(),
            'batch_size': batch_size
        })
        
        # Add metrics if missing
        if 'metrics' not in result:
            result['metrics'] = {
                'latency_ms': result.get('latency_ms', 0),
                'throughput_items_per_sec': result.get('throughput_items_per_sec', 0),
                'memory_usage_mb': result.get('memory_usage_mb', 0)
            }
        
        # Store result
        self.integration.store_acceleration_result(result)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for monitoring."""
        if not self.initialized:
            return {}
        
        # Get current queue size
        queue_size = 0
        if self.execution_queue:
            queue_size = self.execution_queue.qsize()
        
        # Compute average execution time per model
        avg_execution_times = {}
        for model_id, times in self.execution_stats.get('model_execution_times', {}).items():
            if times:
                avg_execution_times[model_id] = sum(times) / len(times)
        
        # Combine stats
        stats = {
            **self.execution_stats,
            'current_queue_size': queue_size,
            'avg_execution_times': avg_execution_times,
            'bridge_stats': self.bridge.get_stats() if self.bridge else {},
            'resource_metrics': self.resource_metrics
        }
        
        return stats
    
    def close(self):
        """Close the resource pool bridge integration and all concurrent execution resources."""
        self._is_shutting_down = True
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
        
        # Cancel executor future
        if self.executor_future:
            self.executor_future.cancel()
            self.executor_future = None
        
        # Close bridge
        if self.bridge:
            self.loop.run_until_complete(self.bridge.close())
            self.bridge = None
        
        self.initialized = False
        logger.info("ResourcePoolBridgeIntegration closed with all concurrent resources")

class WebModel:
    """
    WebNN/WebGPU model instance.
    
    This class provides a model-like interface for models running in a browser
    via WebNN or WebGPU.
    """
    
    def __init__(self, model_id: str, model_type: str, model_name: str, 
                 bridge: ResourcePoolBridge, platform: str, loop):
        """
        Initialize WebModel.
        
        Args:
            model_id: Unique model ID
            model_type: Type of model
            model_name: Name of the model
            bridge: ResourcePoolBridge instance
            platform: 'webgpu', 'webnn', or 'cpu'
            loop: Asyncio event loop
        """
        self.model_id = model_id
        self.model_type = model_type
        self.model_name = model_name
        self.bridge = bridge
        self.platform = platform
        self.loop = loop
        
        # Model metadata
        self.device = platform
        
        logger.info(f"WebModel {model_id} created for {platform}")
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference with the model.
        
        Args:
            inputs: Model inputs
            
        Returns:
            Inference results
        """
        # Run inference using bridge
        return self.loop.run_until_complete(self.bridge.run_inference(self.model_id, inputs))
    
    def to(self, device: str):
        """
        Move model to a different device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        # WebModels can't actually be moved, but we'll log the request
        logger.info(f"WebModel {self.model_id} received request to move to {device} (ignored)")
        return self


class EnhancedWebModel(WebModel):
    """
    Enhanced WebNN/WebGPU model instance with support for concurrent execution.
    
    This class extends WebModel with additional features for concurrent execution,
    batch processing, and resource-aware execution.
    """
    
    def __init__(self, model_id: str, model_type: str, model_name: str,
                 bridge: ResourcePoolBridge, platform: str, loop,
                 integration=None, family: str = None):
        """
        Initialize EnhancedWebModel.
        
        Args:
            model_id: Unique model ID
            model_type: Type of model
            model_name: Name of the model
            bridge: ResourcePoolBridge instance
            platform: 'webgpu', 'webnn', or 'cpu'
            loop: Asyncio event loop
            integration: ResourcePoolBridgeIntegration instance
            family: Model family (e.g., 'vision', 'text', 'audio')
        """
        super().__init__(model_id, model_type, model_name, bridge, platform, loop)
        self.integration = integration
        self.family = family or model_type
        self.batch_size = 1
        self.max_batch_size = 16
        self.memory_usage = {}
        self.execution_stats = {
            'total_calls': 0,
            'total_items': 0,
            'total_time': 0.0,
            'batch_sizes': {},
            'avg_latency': 0.0,
            'throughput': 0.0
        }
        
        logger.info(f"EnhancedWebModel {model_id} created for {platform} with family {family}")
    
    def __call__(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run inference with the model, supporting batched input.
        
        Args:
            inputs: Model inputs as dict or list of dicts for batched execution
            
        Returns:
            Inference results (single dict or list of dicts)
        """
        # Track execution stats
        start_time = time.time()
        self.execution_stats['total_calls'] += 1
        
        # Handle batched input
        is_batch = isinstance(inputs, list)
        if is_batch:
            batch_size = len(inputs)
            self.execution_stats['total_items'] += batch_size
            
            # Track batch size distribution
            if batch_size not in self.execution_stats['batch_sizes']:
                self.execution_stats['batch_sizes'][batch_size] = 0
            self.execution_stats['batch_sizes'][batch_size] += 1
            
            # Process batch
            if batch_size > self.max_batch_size:
                # Split into smaller batches
                results = []
                for i in range(0, batch_size, self.max_batch_size):
                    batch_inputs = inputs[i:i+self.max_batch_size]
                    batch_results = self._process_batch(batch_inputs)
                    results.extend(batch_results)
                return results
            else:
                # Process single batch
                return self._process_batch(inputs)
        else:
            # Single item
            self.execution_stats['total_items'] += 1
            
            # Track batch size = 1
            if 1 not in self.execution_stats['batch_sizes']:
                self.execution_stats['batch_sizes'][1] = 0
            self.execution_stats['batch_sizes'][1] += 1
            
            # Process single item
            result = super().__call__(inputs)
            
            # Update execution stats
            execution_time = time.time() - start_time
            self.execution_stats['total_time'] += execution_time
            self._update_performance_metrics(execution_time, 1)
            
            return result
    
    def _process_batch(self, batch_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of inputs using the model.
        
        Args:
            batch_inputs: List of input dicts
            
        Returns:
            List of result dicts
        """
        batch_size = len(batch_inputs)
        start_time = time.time()
        
        # In real implementation, would use batched inference in browser
        # For now, process sequentially but track as batch
        results = []
        for inputs in batch_inputs:
            result = super().__call__(inputs)
            results.append(result)
        
        # Update execution stats
        execution_time = time.time() - start_time
        self.execution_stats['total_time'] += execution_time
        self._update_performance_metrics(execution_time, batch_size)
        
        return results
    
    def _update_performance_metrics(self, execution_time: float, batch_size: int):
        """
        Update performance metrics based on execution.
        
        Args:
            execution_time: Execution time in seconds
            batch_size: Batch size processed
        """
        # Calculate moving averages for metrics
        if self.execution_stats['total_calls'] > 1:
            # Update average latency (exponential moving average with alpha=0.1)
            alpha = 0.1
            new_latency = execution_time / batch_size  # Latency per item
            self.execution_stats['avg_latency'] = (
                (1 - alpha) * self.execution_stats['avg_latency'] + 
                alpha * new_latency
            )
            
            # Update throughput (items per second)
            if self.execution_stats['total_time'] > 0:
                self.execution_stats['throughput'] = (
                    self.execution_stats['total_items'] / 
                    self.execution_stats['total_time']
                )
        else:
            # First call
            self.execution_stats['avg_latency'] = execution_time / batch_size
            self.execution_stats['throughput'] = batch_size / execution_time
    
    def run_batch(self, batch_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of inputs.
        
        Args:
            batch_inputs: List of input dicts
            
        Returns:
            List of result dicts
        """
        return self(batch_inputs)
    
    def run_concurrent(self, items: List[Dict[str, Any]], other_models: List['EnhancedWebModel'] = None) -> List[Dict[str, Any]]:
        """
        Run inference on multiple items concurrently, potentially with other models.
        
        This method uses the integration's concurrent execution capabilities to
        efficiently run multiple inferences across potentially multiple models.
        
        Args:
            items: List of input dicts for this model
            other_models: Optional list of other models to run concurrently
            
        Returns:
            List of result dicts for this model's inputs
        """
        if not self.integration:
            # Fallback to batch processing if integration not available
            return self.run_batch(items)
        
        # Prepare tasks for this model
        tasks = [(self.model_id, item) for item in items]
        
        # Add tasks for other models if provided
        if other_models:
            # Get one input per model (simplified demo)
            for model in other_models:
                if hasattr(model, 'sample_input') and model.sample_input:
                    tasks.append((model.model_id, model.sample_input))
        
        # Execute all tasks concurrently
        all_results = self.integration.execute_concurrent(tasks)
        
        # Return only results for this model's inputs
        return all_results[:len(items)]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the model.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'platform': self.platform,
            'family': self.family,
            'stats': self.execution_stats,
            'memory_usage': self.memory_usage
        }
    
    def set_max_batch_size(self, batch_size: int):
        """
        Set maximum batch size for the model.
        
        Args:
            batch_size: Maximum batch size
            
        Returns:
            Self for method chaining
        """
        self.max_batch_size = max(1, batch_size)
        return self
    
    @property
    def sample_input(self) -> Optional[Dict[str, Any]]:
        """
        Get a sample input for the model based on its type and family.
        
        Returns:
            Sample input dictionary or None if not available
        """
        # Create sample inputs based on model family
        if self.family == 'text_embedding':
            return {
                'input_ids': [101, 2023, 2003, 1037, 3231, 102],
                'attention_mask': [1, 1, 1, 1, 1, 1]
            }
        elif self.family == 'vision':
            # Create a simple 224x224x3 tensor with all values being 0.5
            pixel_values = []
            for _ in range(1):  # batch size 1
                img = []
                for _ in range(224):  # height
                    row = []
                    for _ in range(224):  # width
                        pixel = [0.5, 0.5, 0.5]  # RGB channels
                        row.append(pixel)
                    img.append(row)
                pixel_values.append(img)
            
            return {
                'pixel_values': pixel_values
            }
        elif self.family == 'audio':
            return {
                'input_features': [[[0.1 for _ in range(80)] for _ in range(3000)]]
            }
        else:
            # Generic input
            return {
                'inputs': [0.0 for _ in range(10)]
            }

# Initialize resource pool integration and register with global resource pool
def integrate_with_resource_pool():
    """Integrate ResourcePoolBridge with global resource pool."""
    integration = ResourcePoolBridgeIntegration()
    
    # Store in global resource pool for access
    global_pool = get_global_resource_pool()
    global_pool.get_resource("web_platform_integration", constructor=lambda: integration)
    
    return integration

# Automate integration when module is imported
if __name__ != "__main__":
    integrate_with_resource_pool()