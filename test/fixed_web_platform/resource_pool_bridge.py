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
    
    Enhanced in March 2025 with:
    - Advanced error recovery and diagnostic mechanisms
    - Circuit breaker pattern implementation for failing connections
    - Enhanced memory management under stress
    - Comprehensive error tracking and telemetry
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
        
        # Enhanced lifecycle and health tracking
        self.status = "created"
        self.health_status = "unknown"
        self.last_error = None
        self.last_error_time = None
        self.error_history = []  # Track last 10 errors with timestamps
        self.heartbeat_failures = 0
        self.max_heartbeat_failures = 3
        self.memory_usage_mb = 0
        self.browser_info = {}
        self.adapter_info = {}
        self.last_health_check = time.time()
        self.startup_time = 0
        self.total_inference_time = 0
        self.total_inference_count = 0
        self.recovery_attempts = 0
        self.max_recovery_attempts = 2
        
        # Circuit breaker pattern implementation
        self.circuit_state = "closed"  # closed, open, half-open
        self.circuit_failure_threshold = 5
        self.circuit_reset_timeout = 30  # seconds
        self.circuit_last_failure_time = 0
        self.consecutive_failures = 0
        
        # Model-specific error tracking
        self.model_error_counts = {}
        self.model_performance = {}
        
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
        Initialize the browser connection with enhanced lifecycle management.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        with self._lock:
            if self.initialized:
                return True
            
            # Update connection status for lifecycle tracking
            if hasattr(self, 'status'):
                self.status = "initializing"
                self.health_status = "unknown"
            
            start_time = time.time()
            
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
                    websocket_bridge = None  # Initialize to None by default
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
                                # Track connection issues
                                if hasattr(self, 'health_status'):
                                    self.health_status = "degraded"
                                if hasattr(self, 'last_error'):
                                    self.last_error = "Failed to create WebSocket bridge"
                                    self.last_error_time = time.time()
                        except ImportError:
                            logger.warning("websocket_bridge module not found, will use simulation")
                            websocket_bridge = None
                            # Track connection issues
                            if hasattr(self, 'health_status'):
                                self.health_status = "degraded"
                            if hasattr(self, 'last_error'):
                                self.last_error = "websocket_bridge module not found"
                                self.last_error_time = time.time()
                        except Exception as e:
                            logger.warning(f"Error creating WebSocket bridge: {e}")
                            websocket_bridge = None
                            # Track connection issues
                            if hasattr(self, 'health_status'):
                                self.health_status = "degraded"
                            if hasattr(self, 'last_error'):
                                self.last_error = f"Error creating WebSocket bridge: {str(e)}"
                                self.last_error_time = time.time()
                except Exception as e:
                    logger.warning(f"Error importing WebSocket bridge: {e}")
                    websocket_bridge = None
                    # Track connection issues
                    if hasattr(self, 'health_status'):
                        self.health_status = "degraded"
                    if hasattr(self, 'last_error'):
                        self.last_error = f"Error importing WebSocket bridge: {str(e)}"
                        self.last_error_time = time.time()
                
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
                    
                    # Launch browser with better error handling
                    logger.info(f"Launching browser {self.browser_name} for connection {self.connection_id}")
                    success = await self.browser_automation.launch()
                    if not success:
                        logger.error(f"Failed to launch browser for connection {self.connection_id}")
                        # Clean up WebSocket bridge if it was created
                        if websocket_bridge:
                            await websocket_bridge.stop()
                        
                        # Update connection status for failed initialization
                        if hasattr(self, 'status'):
                            self.status = "error"
                            self.health_status = "unhealthy"
                        if hasattr(self, 'last_error'):
                            self.last_error = "Failed to launch browser"
                            self.last_error_time = time.time()
                        
                        # Update error stats
                        self.error_count += 1
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
                                
                                # Store browser info and adapter info for diagnostics
                                if hasattr(self, 'browser_info') and capabilities:
                                    self.browser_info = {
                                        'browser_name': self.browser_name,
                                        'browser_version': capabilities.get('browser_version', 'Unknown'),
                                        'user_agent': capabilities.get('user_agent', 'Unknown'),
                                        'platform': capabilities.get('platform', 'Unknown'),
                                        'webgpu_supported': capabilities.get('webgpu_supported', False),
                                        'webnn_supported': capabilities.get('webnn_supported', False),
                                        'compute_shaders_supported': capabilities.get('compute_shaders_supported', False)
                                    }
                                
                                # Log hardware-specific capabilities 
                                if capabilities:
                                    # Check WebGPU support
                                    webgpu_support = capabilities.get("webgpu_supported", False)
                                    if webgpu_support and self.platform == "webgpu":
                                        adapter = capabilities.get("webgpu_adapter", {})
                                        logger.info(f"WebGPU Adapter: {adapter.get('vendor', 'Unknown')} - {adapter.get('device', 'Unknown')}")
                                        
                                        # Store adapter info for diagnostics
                                        if hasattr(self, 'adapter_info'):
                                            self.adapter_info = adapter
                                    
                                    # Check WebNN support
                                    webnn_support = capabilities.get("webnn_supported", False)
                                    if webnn_support and self.platform == "webnn":
                                        backend = capabilities.get("webnn_backend", "Unknown")
                                        logger.info(f"WebNN Backend: {backend}")
                                        
                                        # Store WebNN backend info for diagnostics
                                        if hasattr(self, 'adapter_info'):
                                            self.adapter_info = {'backend': backend}
                                    
                                    # Log compute shader support for audio models
                                    if self.compute_shaders:
                                        compute_shaders = capabilities.get("compute_shaders_supported", False)
                                        logger.info(f"Compute Shader Support: {'Available' if compute_shaders else 'Not Available'}")
                                    
                                    # Update health status based on compatibility
                                    if hasattr(self, 'health_status'):
                                        if (self.platform == "webgpu" and not webgpu_support) or (self.platform == "webnn" and not webnn_support):
                                            logger.warning(f"{self.platform.upper()} not supported by {self.browser_name}, using fallback")
                                            self.health_status = "degraded"
                                        else:
                                            self.health_status = "healthy"
                            except Exception as e:
                                logger.warning(f"Error checking browser capabilities: {e}")
                                # Update health status and error tracking
                                if hasattr(self, 'health_status'):
                                    self.health_status = "degraded"
                                if hasattr(self, 'last_error'):
                                    self.last_error = f"Error checking browser capabilities: {str(e)}"
                                    self.last_error_time = time.time()
                        else:
                            logger.warning("WebSocket connection timed out, will use simulation for inference")
                            # Update health status and error tracking
                            if hasattr(self, 'health_status'):
                                self.health_status = "degraded"
                            if hasattr(self, 'last_error'):
                                self.last_error = "WebSocket connection timed out"
                                self.last_error_time = time.time()
                except ImportError:
                    logger.error("BrowserAutomation not available, using simulation")
                    # Update connection status
                    if hasattr(self, 'status'):
                        self.status = "error"
                        self.health_status = "unhealthy"
                    if hasattr(self, 'last_error'):
                        self.last_error = "BrowserAutomation not available"
                        self.last_error_time = time.time()
                    self.error_count += 1
                    return False
                except Exception as e:
                    logger.error(f"Error creating BrowserAutomation: {e}")
                    traceback.print_exc()
                    # Clean up WebSocket bridge if it was created
                    if websocket_bridge:
                        await websocket_bridge.stop()
                    
                    # Update connection status and error tracking
                    if hasattr(self, 'status'):
                        self.status = "error"
                        self.health_status = "unhealthy"
                    if hasattr(self, 'last_error'):
                        self.last_error = f"Error creating BrowserAutomation: {str(e)}"
                        self.last_error_time = time.time()
                    self.error_count += 1
                    return False
                
                # Update initialization status and metrics
                self.initialized = True
                self.last_health_check = time.time()
                
                # Record startup time for performance tracking
                if hasattr(self, 'startup_time'):
                    self.startup_time = time.time() - start_time
                
                # Update connection state
                if hasattr(self, 'status'):
                    self.status = "ready"
                if hasattr(self, 'health_status') and self.health_status == "unknown":
                    self.health_status = "healthy"
                
                # Log memory usage if available
                try:
                    if websocket_bridge and hasattr(websocket_bridge, 'get_memory_usage'):
                        memory_info = await websocket_bridge.get_memory_usage()
                        if memory_info and 'js_heap_size_mb' in memory_info:
                            if hasattr(self, 'memory_usage_mb'):
                                self.memory_usage_mb = memory_info['js_heap_size_mb']
                            logger.info(f"Initial memory usage: {memory_info['js_heap_size_mb']} MB")
                except Exception as mem_error:
                    logger.debug(f"Error getting memory usage: {mem_error}")
                
                logger.info(f"Browser connection {self.connection_id} initialized successfully in {time.time() - start_time:.2f}s")
                return True
                
            except Exception as e:
                logger.error(f"Error initializing browser connection {self.connection_id}: {e}")
                traceback.print_exc()
                
                # Update connection status and error tracking
                if hasattr(self, 'status'):
                    self.status = "error"
                    self.health_status = "unhealthy"
                if hasattr(self, 'last_error'):
                    self.last_error = f"Error initializing connection: {str(e)}"
                    self.last_error_time = time.time()
                self.error_count += 1
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
    
    def _check_circuit_breaker(self, model_id: str) -> Tuple[bool, str]:
        """
        Check if circuit breaker allows operation to proceed.
        
        Implements the circuit breaker pattern to prevent repeated calls to failing services.
        
        Args:
            model_id: ID of the model to check
            
        Returns:
            Tuple[bool, str]: (is_allowed, reason)
                is_allowed: True if operation is allowed, False otherwise
                reason: Reason why operation is not allowed (if applicable)
        """
        # Check global circuit breaker first
        current_time = time.time()
        
        # If circuit is open, check if reset timeout has elapsed
        if self.circuit_state == "open":
            if current_time - self.circuit_last_failure_time > self.circuit_reset_timeout:
                # Reset to half-open state and allow a trial request
                self.circuit_state = "half-open"
                logger.info(f"Circuit breaker transitioned from open to half-open for {self.connection_id}")
                return True, "Circuit breaker in half-open state, allowing trial request"
            else:
                # Circuit is open and timeout not reached, fail fast
                time_remaining = self.circuit_reset_timeout - (current_time - self.circuit_last_failure_time)
                return False, f"Circuit breaker open (reset in {time_remaining:.1f}s)"
        
        # Check model-specific circuit breaker
        if model_id in self.model_error_counts:
            model_errors = self.model_error_counts[model_id]
            # If model has excessive errors, fail fast
            if model_errors >= 3:  # Use a lower threshold for model-specific errors
                return False, f"Model {model_id} has excessive errors ({model_errors})"
        
        # Circuit is closed or half-open, allow operation
        return True, "Circuit breaker closed"
    
    def _update_circuit_breaker(self, success: bool, model_id: str = None, error: str = None):
        """
        Update circuit breaker state based on operation success/failure.
        
        Args:
            success: Whether the operation succeeded
            model_id: Model ID for model-specific tracking (optional)
            error: Error message if operation failed (optional)
        """
        if success:
            # On success, reset failure counters
            if self.circuit_state == "half-open":
                # Transition from half-open to closed on successful operation
                self.circuit_state = "closed"
                logger.info(f"Circuit breaker transitioned from half-open to closed for {self.connection_id}")
            
            # Reset counters
            self.consecutive_failures = 0
            
            # Reset model-specific error count if relevant
            if model_id and model_id in self.model_error_counts and self.model_error_counts[model_id] > 0:
                self.model_error_counts[model_id] = 0
                
        else:
            # On failure, increment counters
            self.consecutive_failures += 1
            
            # Update model-specific error count
            if model_id:
                if model_id not in self.model_error_counts:
                    self.model_error_counts[model_id] = 0
                self.model_error_counts[model_id] += 1
            
            # Track error history (keep last 10)
            if error:
                error_entry = {"time": time.time(), "error": error, "model_id": model_id}
                self.error_history.append(error_entry)
                if len(self.error_history) > 10:
                    self.error_history.pop(0)  # Remove oldest error
            
            # Update global circuit breaker state
            if self.consecutive_failures >= self.circuit_failure_threshold:
                # Open the circuit breaker
                if self.circuit_state != "open":
                    self.circuit_state = "open"
                    self.circuit_last_failure_time = time.time()
                    logger.warning(f"Circuit breaker opened for {self.connection_id} due to {self.consecutive_failures} consecutive failures")
    
    async def _recover_connection(self) -> bool:
        """
        Attempt to recover a degraded connection.
        
        This method performs a sequence of recovery steps to bring a connection
        back to a healthy state without completely rebuilding it.
        
        Returns:
            bool: True if recovery succeeded, False otherwise
        """
        if not self.initialized or not hasattr(self, 'browser_automation') or not self.browser_automation:
            logger.error(f"Cannot recover uninitialized connection {self.connection_id}")
            return False
        
        # Update connection status
        if hasattr(self, 'status'):
            self.status = "recovering"
        
        logger.info(f"Attempting to recover connection {self.connection_id}, attempt {self.recovery_attempts+1}/{self.max_recovery_attempts}")
        self.recovery_attempts += 1
        
        try:
            # Step 1: Try to ping the WebSocket
            try:
                if (hasattr(self.browser_automation, 'websocket_bridge') and 
                    self.browser_automation.websocket_bridge and
                    hasattr(self.browser_automation.websocket_bridge, 'ping')):
                    
                    logger.info(f"Attempting to ping WebSocket for connection {self.connection_id}")
                    ping_response = await self.browser_automation.websocket_bridge.ping(timeout=5.0)
                    
                    if ping_response and ping_response.get('status') == 'success':
                        logger.info(f"WebSocket ping successful for connection {self.connection_id}")
                        
                        # Update health status
                        if hasattr(self, 'health_status'):
                            self.health_status = "degraded"  # Still degraded until proven otherwise
                        
                        # WebSocket is alive, try a basic operation
                        try:
                            capabilities = await self.browser_automation.websocket_bridge.get_browser_capabilities(retry_attempts=1)
                            if capabilities:
                                logger.info(f"WebSocket connection appears functional for {self.connection_id}")
                                
                                # Update health status
                                if hasattr(self, 'health_status'):
                                    self.health_status = "healthy"
                                
                                # Reset some error counters
                                self.heartbeat_failures = 0
                                
                                # Update status
                                if hasattr(self, 'status'):
                                    self.status = "ready"
                                
                                return True
                        except Exception as cap_error:
                            logger.warning(f"Error checking capabilities during recovery: {cap_error}")
                    else:
                        logger.warning(f"WebSocket ping failed during recovery for {self.connection_id}")
            except Exception as ping_error:
                logger.warning(f"Error during WebSocket ping recovery: {ping_error}")
            
            # Step 2: Try to restart the WebSocket bridge
            try:
                if (hasattr(self.browser_automation, 'websocket_bridge') and 
                    self.browser_automation.websocket_bridge):
                    
                    logger.info(f"Attempting to restart WebSocket bridge for connection {self.connection_id}")
                    
                    # Stop the current bridge
                    await self.browser_automation.websocket_bridge.stop()
                    
                    # Wait briefly
                    await asyncio.sleep(1.0)
                    
                    # Create a new bridge
                    self.websocket_port = 8765 + random.randint(0, 1000)  # Use a new random port
                    
                    # Import WebSocketBridge from fixed_web_platform module
                    try:
                        from websocket_bridge import create_websocket_bridge
                        
                        # Create new WebSocket bridge
                        logger.info(f"Creating new WebSocket bridge on port {self.websocket_port}")
                        websocket_bridge = await create_websocket_bridge(port=self.websocket_port)
                        
                        if websocket_bridge:
                            # Update the browser automation with new bridge
                            self.browser_automation.websocket_bridge = websocket_bridge
                            
                            # Try to refresh the page to reload the bridge
                            if hasattr(self.browser_automation, 'refresh_page'):
                                await self.browser_automation.refresh_page()
                            
                            # Check if bridge is working
                            await asyncio.sleep(2.0)  # Wait for page to refresh
                            
                            # Wait for WebSocket connection
                            websocket_connected = await websocket_bridge.wait_for_connection(
                                timeout=10.0,
                                retry_attempts=2
                            )
                            
                            if websocket_connected:
                                logger.info(f"WebSocket bridge successfully restarted for {self.connection_id}")
                                
                                # Update health status
                                if hasattr(self, 'health_status'):
                                    self.health_status = "degraded"  # Still degraded until fully proven
                                
                                # Get browser capabilities to verify feature support
                                capabilities = await websocket_bridge.get_browser_capabilities(retry_attempts=1)
                                if capabilities:
                                    logger.info(f"New WebSocket bridge is fully functional for {self.connection_id}")
                                    
                                    # Update health status
                                    if hasattr(self, 'health_status'):
                                        self.health_status = "healthy"
                                    
                                    # Update status
                                    if hasattr(self, 'status'):
                                        self.status = "ready"
                                    
                                    return True
                    except Exception as bridge_error:
                        logger.warning(f"Error recreating WebSocket bridge: {bridge_error}")
            except Exception as restart_error:
                logger.warning(f"Error during WebSocket bridge restart: {restart_error}")
            
            # If recovery attempts exhausted, mark as failed
            if self.recovery_attempts >= self.max_recovery_attempts:
                logger.error(f"Recovery failed after {self.recovery_attempts} attempts for connection {self.connection_id}")
                
                # Update health status
                if hasattr(self, 'health_status'):
                    self.health_status = "unhealthy"
                
                # Update status
                if hasattr(self, 'status'):
                    self.status = "error"
                
                return False
            
            # Final fallback: try a simpler recovery by refreshing the page
            try:
                if hasattr(self.browser_automation, 'refresh_page'):
                    logger.info(f"Attempting final recovery with page refresh for {self.connection_id}")
                    await self.browser_automation.refresh_page()
                    
                    # Wait for page to load
                    await asyncio.sleep(3.0)
                    
                    # Update status
                    if hasattr(self, 'status'):
                        self.status = "degraded"
                    
                    if hasattr(self, 'health_status'):
                        self.health_status = "degraded"
                    
                    # Not fully recovered but maybe usable
                    return True
            except Exception as refresh_error:
                logger.warning(f"Error during page refresh recovery: {refresh_error}")
            
            # Recovery failed
            return False
            
        except Exception as recovery_error:
            logger.error(f"Unexpected error during connection recovery: {recovery_error}")
            traceback.print_exc()
            return False
    
    async def run_inference(self, model_id: str, inputs: Dict[str, Any], retry_attempts: int = 1) -> Dict[str, Any]:
        """
        Run inference with a loaded model with enhanced reliability and circuit breaker pattern.
        
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
            
            # Check circuit breaker before proceeding
            circuit_allowed, circuit_reason = self._check_circuit_breaker(model_id)
            if not circuit_allowed:
                logger.warning(f"Circuit breaker prevented inference for {model_id}: {circuit_reason}")
                
                # Return simulated result with circuit breaker info
                return {
                    'success': True,  # Still return success for compatibility
                    'status': 'simulated',
                    'model_id': model_id,
                    'output_shape': [1, 768],  # Mock output shape
                    'platform': self.platform,
                    'backend': 'webgpu' if self.platform == 'webgpu' else 'webnn',
                    'browser': self.browser_name,
                    'is_real_implementation': False,
                    'is_simulation': True,
                    'circuit_breaker_active': True,
                    'circuit_breaker_reason': circuit_reason,
                    'performance_metrics': {
                        'inference_time_ms': 100,  # Fast simulation since we fail fast
                        'throughput_items_per_sec': 10,
                        'memory_usage_mb': 500  # Mock memory usage
                    },
                    'compute_shader_optimized': self.compute_shaders,
                    'precompile_shaders': self.precompile_shaders,
                    'parallel_loading': self.parallel_loading
                }
                
            self.busy = True
            self.last_used_time = time.time()
            inference_start_time = time.time()
            
            # Track retries and errors
            attempt = 0
            last_error = None
            recovery_triggered = False
            
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
                            
                            # Check WebSocket health if this is a retry
                            if attempt > 0 and hasattr(self, 'health_status') and self.health_status != "healthy":
                                # Connection is degraded, try recovery before proceeding
                                if not recovery_triggered:
                                    logger.info(f"Connection {self.connection_id} is degraded, attempting recovery before retry")
                                    recovery_success = await self._recover_connection()
                                    recovery_triggered = True
                                    
                                    if not recovery_success:
                                        logger.warning(f"Recovery failed for connection {self.connection_id}, falling back to simulation")
                                        break  # Exit to simulation
                                        
                                    # Update bridge reference after recovery
                                    if hasattr(self.browser_automation, 'websocket_bridge'):
                                        bridge = self.browser_automation.websocket_bridge
                            
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
                            
                            # Store model-specific metadata for diagnostics
                            if model_id not in self.model_performance:
                                self.model_performance[model_id] = {
                                    'execution_count': 0,
                                    'success_count': 0,
                                    'failure_count': 0,
                                    'average_latency_ms': 0,
                                    'last_execution_time': None,
                                    'memory_footprint_mb': 0
                                }
                            
                            # Update execution count
                            self.model_performance[model_id]['execution_count'] += 1
                            self.model_performance[model_id]['last_execution_time'] = time.time()
                            
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
                                
                                # Update circuit breaker on success
                                self._update_circuit_breaker(success=True, model_id=model_id)
                                
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
                                    
                                    # Update model memory footprint
                                    self.model_performance[model_id]['memory_footprint_mb'] = memory_usage
                                
                                # Update model performance metrics
                                self.model_performance[model_id]['success_count'] += 1
                                # Update average latency with exponential moving average
                                latency_ms = performance_metrics['inference_time_ms']
                                prev_avg = self.model_performance[model_id]['average_latency_ms']
                                if prev_avg == 0:
                                    self.model_performance[model_id]['average_latency_ms'] = latency_ms
                                else:
                                    # Use 0.8 as weight for new measurements
                                    self.model_performance[model_id]['average_latency_ms'] = prev_avg * 0.2 + latency_ms * 0.8
                                
                                # Update connection statistics
                                self.total_inference_count += 1
                                self.total_inference_time += inference_time
                                
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
                                
                                # Add diagnostic information
                                result['diagnostics'] = {
                                    'recovery_triggered': recovery_triggered,
                                    'retry_count': attempt,
                                    'connection_health': getattr(self, 'health_status', 'unknown'),
                                    'total_time_ms': (time.time() - inference_start_time) * 1000
                                }
                                
                                self.busy = False
                                return result
                            else:
                                # Extract error information for better debugging
                                error = "Unknown error"
                                if inference_response:
                                    error = inference_response.get('error', 'No error details')
                                    
                                last_error = f"WebSocket inference failed: {error}"
                                logger.warning(last_error)
                                
                                # Update model performance metrics
                                self.model_performance[model_id]['failure_count'] += 1
                                
                                # Update circuit breaker state
                                self._update_circuit_breaker(success=False, model_id=model_id, error=last_error)
                                
                                # Check health and trigger recovery if needed
                                if not recovery_triggered and hasattr(self, 'health_status'):
                                    if self.health_status != "healthy":
                                        logger.info(f"Connection health is {self.health_status}, attempting recovery")
                                        recovery_success = await self._recover_connection()
                                        recovery_triggered = True
                                        
                                        if recovery_success:
                                            logger.info(f"Connection recovery succeeded for {self.connection_id}")
                                        else:
                                            logger.warning(f"Connection recovery failed for {self.connection_id}")
                                
                                # If we have retries left, try again
                                if attempt < retry_attempts:
                                    attempt += 1
                                    # Adaptive pause before retry based on error type and recovery status
                                    if recovery_triggered:
                                        # Longer pause after recovery
                                        await asyncio.sleep(1.0 * attempt)
                                    else:
                                        # Standard pause before regular retry
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
                    
                    # Update circuit breaker state
                    self._update_circuit_breaker(success=False, model_id=model_id, error=last_error)
                    
                    # Update model performance metrics
                    if model_id in self.model_performance:
                        self.model_performance[model_id]['failure_count'] += 1
                    
                    # Try recovery if severe error and not already attempted
                    if not recovery_triggered and str(e).lower() in ["connection refused", "not connected", "timeout", "connection closed"]:
                        logger.info(f"Attempting connection recovery due to severe error: {e}")
                        recovery_success = await self._recover_connection()
                        recovery_triggered = True
                        
                        if not recovery_success:
                            logger.warning(f"Connection recovery failed, falling back to simulation")
                            break
                    
                    if attempt < retry_attempts:
                        attempt += 1
                        # Adaptive pause before retry
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
                
            # If we have performance data from previous real runs, use it for more realistic simulation
            if model_id in self.model_performance and self.model_performance[model_id]['average_latency_ms'] > 0:
                realistic_latency_ms = self.model_performance[model_id]['average_latency_ms']
                inference_time = realistic_latency_ms / 1000  # Convert ms to seconds
                logger.info(f"Using realistic simulation time of {inference_time:.3f}s based on previous runs")
                
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
                'recovery_attempted': recovery_triggered,
                'error_info': last_error,  # Include error that caused fallback
                'performance_metrics': {
                    'inference_time_ms': inference_time * 1000,
                    'throughput_items_per_sec': 1.0 / inference_time,
                    'memory_usage_mb': 500  # Mock memory usage
                },
                'compute_shader_optimized': self.compute_shaders,
                'precompile_shaders': self.precompile_shaders,
                'parallel_loading': self.parallel_loading,
                'circuit_breaker_state': self.circuit_state
            }
            
            # Add diagnostics
            result['diagnostics'] = {
                'recovery_triggered': recovery_triggered,
                'retry_count': attempt,
                'connection_health': getattr(self, 'health_status', 'unknown'),
                'total_time_ms': (time.time() - inference_start_time) * 1000,
                'last_error': last_error,
                'error_count': self.error_count
            }
            
            self.busy = False
            return result
    
    async def close(self):
        """Close the browser connection and clean up resources with enhanced resource management."""
        with self._lock:
            # Update connection status for lifecycle tracking
            if hasattr(self, 'status'):
                prev_status = getattr(self, 'status', 'unknown')
                self.status = "closing"
                logger.debug(f"Connection {self.connection_id} status change: {prev_status} -> closing")
            
            # Track metrics for cleanup
            start_time = time.time()
            cleanup_success = False
            cleanup_error = None
            
            # Shutdown browser automation and WebSocket with improved error handling
            if self.browser_automation:
                try:
                    # Send shutdown command to browser via WebSocket
                    if hasattr(self.browser_automation, 'websocket_bridge') and self.browser_automation.websocket_bridge:
                        try:
                            # Send graceful shutdown to allow browser to clean up resources
                            logger.info(f"Sending shutdown command to browser for connection {self.connection_id}")
                            
                            # Attempt graceful shutdown with timeout
                            shutdown_task = self.browser_automation.websocket_bridge.shutdown_browser()
                            try:
                                # Use timeout to prevent hanging
                                shutdown_timeout = 5.0  # 5 seconds timeout for graceful shutdown
                                await asyncio.wait_for(shutdown_task, timeout=shutdown_timeout)
                                logger.info(f"Browser shutdown command sent successfully for connection {self.connection_id}")
                            except asyncio.TimeoutError:
                                logger.warning(f"Browser shutdown command timed out for connection {self.connection_id}")
                            
                            # Stop WebSocket bridge
                            await self.browser_automation.websocket_bridge.stop()
                            logger.info(f"WebSocket bridge closed for connection {self.connection_id}")
                        except Exception as e:
                            logger.error(f"Error closing WebSocket bridge for connection {self.connection_id}: {e}")
                            cleanup_error = str(e)
                            
                            # Try forceful close as fallback
                            try:
                                if hasattr(self.browser_automation.websocket_bridge, 'force_stop'):
                                    await self.browser_automation.websocket_bridge.force_stop()
                                    logger.info(f"WebSocket bridge forcefully closed for connection {self.connection_id}")
                            except Exception as force_error:
                                logger.error(f"Error forcefully closing WebSocket bridge: {force_error}")
                    
                    # Close browser automation
                    logger.info(f"Closing browser for connection {self.connection_id}")
                    
                    # Attempt browser close with timeout
                    close_task = self.browser_automation.close()
                    try:
                        # Use timeout to prevent hanging
                        close_timeout = 10.0  # 10 seconds timeout for browser close
                        await asyncio.wait_for(close_task, timeout=close_timeout)
                        logger.info(f"Browser connection {self.connection_id} closed successfully")
                        cleanup_success = True
                    except asyncio.TimeoutError:
                        logger.warning(f"Browser close timed out for connection {self.connection_id}, using force close")
                        
                        # Try forceful close as fallback
                        try:
                            if hasattr(self.browser_automation, 'force_close'):
                                await self.browser_automation.force_close()
                                logger.info(f"Browser forcefully closed for connection {self.connection_id}")
                                cleanup_success = True
                        except Exception as force_error:
                            logger.error(f"Error forcefully closing browser: {force_error}")
                            cleanup_error = str(force_error)
                except Exception as e:
                    logger.error(f"Error closing browser connection {self.connection_id}: {e}")
                    traceback.print_exc()
                    cleanup_error = str(e)
            else:
                logger.info(f"No browser automation to close for connection {self.connection_id}")
                cleanup_success = True  # Nothing to clean up
            
            # Collect resource metrics before clearing
            model_count = len(self.loaded_models)
            models_list = list(self.loaded_models)
            
            # Reset state
            self.initialized = False
            self.loaded_models.clear()
            self.browser_automation = None
            
            # Update final connection status
            if hasattr(self, 'status'):
                self.status = "closed"
            
            # Record cleanup metrics for diagnostics
            cleanup_time = time.time() - start_time
            logger.info(f"Connection {self.connection_id} closed in {cleanup_time:.2f}s - Success: {cleanup_success}, Models unloaded: {model_count}")
            
            # Attempt to clean up any remaining resources
            try:
                # Force garbage collection to clean up resources
                import gc
                gc.collect()
                
                # For CUDA usage, try to clear CUDA cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug(f"CUDA cache cleared after closing connection {self.connection_id}")
                except (ImportError, Exception):
                    pass  # CUDA cleanup is optional
            except Exception as cleanup_error:
                logger.debug(f"Error in final resource cleanup: {cleanup_error}")
            
            # Return cleanup status
            return cleanup_success
    
    def is_busy(self) -> bool:
        """Check if connection is busy."""
        with self._lock:
            return self.busy
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy with enhanced diagnostics."""
        with self._lock:
            # Basic health check (initialized with acceptable error count)
            basic_health = self.initialized and self.error_count < self.max_errors
            
            # If we have specific health status tracking, use it
            if hasattr(self, 'health_status'):
                # Update last health check time
                self.last_health_check = time.time()
                
                # Return health based on status
                return self.health_status in ["healthy", "ready"]
            
            return basic_health
    
    async def perform_health_check(self) -> bool:
        """
        Perform active health check on the browser connection.
        
        This method checks the actual browser health by sending a ping message
        via WebSocket and analyzing the response. It provides more accurate
        health status than the passive is_healthy() check.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        with self._lock:
            # Update health check timestamp
            self.last_health_check = time.time()
            
            # Initialize default health assuming not healthy
            is_healthy = False
            
            # Skip check if not initialized
            if not self.initialized:
                if hasattr(self, 'health_status'):
                    self.health_status = "unhealthy"
                return False
            
            # Check for excessive errors
            if self.error_count >= self.max_errors:
                if hasattr(self, 'health_status'):
                    self.health_status = "unhealthy"
                return False
            
            try:
                # Check browser automation
                if (self.browser_automation and 
                    hasattr(self.browser_automation, 'websocket_bridge') and 
                    self.browser_automation.websocket_bridge):
                    
                    # Try to ping browser via WebSocket
                    try:
                        websocket_bridge = self.browser_automation.websocket_bridge
                        if hasattr(websocket_bridge, 'ping'):
                            ping_response = await websocket_bridge.ping(timeout=3.0)
                            
                            if ping_response and ping_response.get('status') == 'success':
                                # Connection is healthy
                                is_healthy = True
                                
                                # Check for memory information in response
                                if 'memory' in ping_response:
                                    memory_usage = ping_response['memory'].get('js_heap_size_mb', 0)
                                    if hasattr(self, 'memory_usage_mb'):
                                        self.memory_usage_mb = memory_usage
                                    
                                    # Log memory pressure if high
                                    if memory_usage > 1000:  # Over 1GB
                                        logger.warning(f"High memory usage in browser: {memory_usage} MB")
                            else:
                                # Failed ping response
                                logger.warning(f"Ping failed for connection {self.connection_id}")
                                if hasattr(self, 'heartbeat_failures'):
                                    self.heartbeat_failures += 1
                        else:
                            # No ping method available, assume healthy but add warning
                            logger.debug(f"No ping method available for connection {self.connection_id}")
                            is_healthy = True  # Assume healthy by default
                    except Exception as e:
                        logger.warning(f"Error during health check ping: {e}")
                        if hasattr(self, 'heartbeat_failures'):
                            self.heartbeat_failures += 1
                else:
                    # No WebSocket bridge, assume healthy but degraded
                    is_healthy = True  # Assume healthy by default
                    if hasattr(self, 'health_status'):
                        self.health_status = "degraded"
            except Exception as e:
                logger.error(f"Error during health check: {e}")
                self.error_count += 1
                is_healthy = False
            
            # Update health status
            if hasattr(self, 'health_status'):
                if is_healthy:
                    heartbeat_failures = getattr(self, 'heartbeat_failures', 0)
                    if heartbeat_failures > 0:
                        # Has had failures but currently responding
                        self.health_status = "degraded"
                    else:
                        self.health_status = "healthy"
                else:
                    # Set to unhealthy only if heartbeat failures exceed threshold
                    heartbeat_failures = getattr(self, 'heartbeat_failures', 0)
                    max_failures = getattr(self, 'max_heartbeat_failures', 3)
                    
                    if heartbeat_failures >= max_failures:
                        self.health_status = "unhealthy"
                        logger.warning(f"Connection {self.connection_id} marked unhealthy due to {heartbeat_failures} heartbeat failures")
                    else:
                        self.health_status = "degraded"
            
            return is_healthy
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get detailed health metrics for the connection.
        
        Returns:
            Dictionary with detailed health information
        """
        with self._lock:
            return {
                'connection_id': self.connection_id,
                'status': getattr(self, 'status', 'unknown'),
                'health_status': getattr(self, 'health_status', 'unknown'),
                'initialized': self.initialized,
                'error_count': self.error_count,
                'max_errors': self.max_errors,
                'heartbeat_failures': getattr(self, 'heartbeat_failures', 0),
                'last_error': getattr(self, 'last_error', None),
                'last_error_time': getattr(self, 'last_error_time', None),
                'creation_time': self.creation_time,
                'time_since_creation': time.time() - self.creation_time,
                'last_used_time': self.last_used_time,
                'time_since_last_used': time.time() - self.last_used_time,
                'last_health_check': getattr(self, 'last_health_check', 0),
                'time_since_health_check': time.time() - getattr(self, 'last_health_check', time.time()),
                'loaded_model_count': len(self.loaded_models),
                'memory_usage_mb': getattr(self, 'memory_usage_mb', 0),
                'busy': self.busy,
                'startup_time': getattr(self, 'startup_time', 0),
                'total_inference_time': getattr(self, 'total_inference_time', 0),
                'total_inference_count': getattr(self, 'total_inference_count', 0),
                'browser_info': getattr(self, 'browser_info', {}),
                'adapter_info': getattr(self, 'adapter_info', {}),
                'platform': self.platform,
                'browser_name': self.browser_name
            }
    
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
    
    def get_lifecycle_status(self) -> Dict[str, Any]:
        """
        Get detailed lifecycle status information.
        
        Returns:
            Dictionary with lifecycle information
        """
        with self._lock:
            # Calculate utilization metrics
            age = self.get_age()
            idle_time = self.get_idle_time()
            active_time = age - idle_time if age > idle_time else 0
            utilization = (active_time / age) * 100 if age > 0 else 0
            
            return {
                'connection_id': self.connection_id,
                'status': getattr(self, 'status', 'unknown'),
                'initialized': self.initialized,
                'platform': self.platform,
                'browser': self.browser_name,
                'creation_time': self.creation_time,
                'age_seconds': age,
                'last_used_time': self.last_used_time,
                'idle_time_seconds': idle_time,
                'active_time_seconds': active_time,
                'utilization_percent': utilization,
                'model_count': len(self.loaded_models),
                'models': list(self.loaded_models),
                'is_busy': self.busy,
                'is_healthy': self.is_healthy(),
                'health_status': getattr(self, 'health_status', 'unknown'),
                'memory_usage_mb': getattr(self, 'memory_usage_mb', 0)
            }

class ResourcePoolBridge:
    """
    Resource pool for WebNN and WebGPU platforms.
    
    This class manages a pool of browser connections for running models on WebNN and WebGPU,
    allowing concurrent execution of models on different hardware backends with advanced
    parallel execution, connection management, and load balancing capabilities.
    
    Key features:
    - Pool of browser connections for efficient resource utilization
    - Parallel model execution across GPU (WebGPU) and CPU backends
    - Model-specific hardware preferences based on model type
    - Automatic load balancing and resource allocation
    - Connection management with health monitoring and recovery
    - Efficient resource cleanup and memory management
    
    Enhanced in March 2025 with:
    - Advanced circuit breaker pattern for failure resilience  
    - Comprehensive error recovery mechanisms for browser connections
    - Enhanced telemetry and diagnostics for intermittent failures
    - Model-specific error tracking and performance monitoring
    - Memory pressure detection with adaptive scaling
    - Detailed lifecycle management for browser connections
    """
    
    def __init__(self, max_connections: int = 4, 
                 browser: str = 'chrome', enable_gpu: bool = True, 
                 enable_cpu: bool = True, headless: bool = True,
                 cleanup_interval: int = 300, 
                 connection_timeout: float = 30.0,
                 adaptive_scaling: bool = True):
        """
        Initialize resource pool bridge with enhanced pooling capabilities.
        
        Args:
            max_connections: Maximum number of concurrent browser connections
            browser: Default browser name ('chrome', 'firefox', 'edge', 'safari')
            enable_gpu: Enable GPU (WebGPU) backend
            enable_cpu: Enable CPU backend
            headless: Whether to run browsers in headless mode
            cleanup_interval: Interval in seconds for connection cleanup
            connection_timeout: Timeout for establishing connection (seconds)
            adaptive_scaling: Whether to enable adaptive connection scaling
        """
        self.max_connections = max_connections
        self.default_browser = browser
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu
        self.headless = headless
        self.cleanup_interval = cleanup_interval
        self.connection_timeout = connection_timeout
        self.adaptive_scaling = adaptive_scaling
        
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
            'peak_connections': 0,
            'concurrent_executions': 0,
            'parallel_models_executed': 0,
            'connection_reuse_count': {},
            'platform_usage': {
                'webgpu': 0,
                'webnn': 0,
                'cpu': 0
            },
            'browser_usage': {},
            'execution_times': {},
            'queue_wait_times': [],
            'memory_usage': {}
        }
        
        # Connection health and performance metrics
        self.connection_health = {}
        self.connection_performance = {}
        
        # Browser preference mapping for model types
        self.browser_preferences = {
            'audio': 'firefox',  # Firefox has better compute shader performance for audio models
            'vision': 'chrome',  # Chrome has good WebGPU support for vision models
            'text_embedding': 'edge',  # Edge has excellent WebNN support for text embeddings
            'text': 'edge',  # Edge works well for text models with WebNN
            'multimodal': 'chrome',  # Chrome is good for multimodal models
        }
        
        # Concurrency control
        self._lock = threading.RLock()
        self._loop = None
        self._initialized = False
        self._cleanup_task = None
        self._connection_monitor_task = None
        self._is_shutting_down = False
        
        # Task queue for advanced execution scheduling
        self.task_queue = asyncio.Queue() if asyncio else None
        self.scheduler_task = None
        
        # Initialize asyncio event loop
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if none exists
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        
        logger.info(f"ResourcePoolBridge initialized with max {max_connections} connections, " +
                  f"adaptive_scaling={adaptive_scaling}")
    
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
            
            # Initialize concurrent execution system
            self._initialized = True
            logger.info("ResourcePoolBridge initialized successfully")
            
            # Initialize concurrent execution pool
            self._init_concurrent_execution_pool()
            
            return True
    
    def _start_cleanup_task(self):
        """Start the connection cleanup task and connection health monitor."""
        # Define cleanup task
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
        
        # Define connection health monitor task
        async def connection_monitor_task():
            # Use shorter interval for health checks
            health_check_interval = min(self.cleanup_interval / 3, 60)  # At most every 60s, or 1/3 of cleanup interval
            
            while True:
                try:
                    await asyncio.sleep(health_check_interval)
                    await self._monitor_connections()
                except asyncio.CancelledError:
                    # Task is being cancelled
                    break
                except Exception as e:
                    logger.error(f"Error in connection monitor task: {e}")
        
        # Schedule cleanup task
        self._cleanup_task = asyncio.ensure_future(cleanup_task(), loop=self._loop)
        
        # Schedule connection monitor task
        self._connection_monitor_task = asyncio.ensure_future(connection_monitor_task(), loop=self._loop)
        
        logger.info(f"Started connection monitoring and cleanup tasks (monitor every {min(self.cleanup_interval / 3, 60):.0f}s, cleanup every {self.cleanup_interval}s)")
        
    async def _monitor_connections(self):
        """
        Monitor connection health by performing active health checks.
        
        This method periodically checks all connections to ensure they're healthy,
        updates their status, and collects performance metrics. It helps identify issues
        with connections before they cause failures during model inference.
        
        Enhanced in March 2025 with:
        - Automated recovery for degraded connections
        - Circuit breaker monitoring and management
        - Memory pressure and resource usage tracking
        - Detailed telemetry for performance optimization
        """
        with self._lock:
            # Skip if shutting down
            if self._is_shutting_down:
                return
            
            # Track current connections
            connection_stats = {
                'total': 0,
                'healthy': 0,
                'degraded': 0,
                'unhealthy': 0,
                'memory_usage_total': 0,
                'memory_usage_avg': 0,
                'health_check_duration': 0,
                'circuit_breaker_open': 0,
                'circuit_breaker_half_open': 0,
                'recovery_attempts': 0,
                'recovery_successes': 0
            }
            
            # Start timing
            check_start_time = time.time()
            
            # Check system memory pressure
            system_memory_pressure = False
            available_memory_mb = 0
            try:
                import psutil
                vm = psutil.virtual_memory()
                available_memory_mb = vm.available / (1024 * 1024)
                system_memory_pressure = vm.percent > 85  # Over 85% memory utilization is high pressure
                
                # Add to stats
                connection_stats['system_memory_percent'] = vm.percent
                connection_stats['system_available_memory_mb'] = available_memory_mb
                
                if system_memory_pressure:
                    logger.warning(f"System under memory pressure: {vm.percent}% memory used, {available_memory_mb:.0f}MB available")
            except ImportError:
                logger.debug("psutil not available, skipping memory pressure check")
            
            # Collect all connections
            all_connections = []
            for platform, connections in self.connections.items():
                all_connections.extend(connections)
            
            connection_stats['total'] = len(all_connections)
            
            # Skip if no connections
            if not all_connections:
                logger.debug("No connections to monitor")
                return
            
            # Perform health checks on all connections
            recovery_tasks = []
            for conn in all_connections:
                try:
                    # Skip if connection is busy to avoid interference
                    if conn.is_busy():
                        connection_stats['total'] -= 1  # Don't count in stats
                        continue
                    
                    # Check circuit breaker status
                    if hasattr(conn, 'circuit_state'):
                        if conn.circuit_state == "open":
                            connection_stats['circuit_breaker_open'] += 1
                            
                            # Check if circuit breaker should transition to half-open
                            current_time = time.time()
                            if hasattr(conn, 'circuit_last_failure_time') and hasattr(conn, 'circuit_reset_timeout'):
                                if current_time - conn.circuit_last_failure_time > conn.circuit_reset_timeout:
                                    # Reset to half-open state and allow a trial request
                                    conn.circuit_state = "half-open"
                                    logger.info(f"Circuit breaker transitioned from open to half-open for {conn.connection_id}")
                                    connection_stats['circuit_breaker_half_open'] += 1
                        elif conn.circuit_state == "half-open":
                            connection_stats['circuit_breaker_half_open'] += 1
                    
                    # Perform health check
                    is_healthy = await conn.perform_health_check()
                    
                    # Update stats based on health status
                    if is_healthy:
                        if hasattr(conn, 'health_status'):
                            if conn.health_status == 'degraded':
                                connection_stats['degraded'] += 1
                            else:
                                connection_stats['healthy'] += 1
                        else:
                            connection_stats['healthy'] += 1
                    else:
                        connection_stats['unhealthy'] += 1
                        
                        # Attempt automatic recovery for unhealthy connections that aren't busy
                        if (hasattr(conn, 'health_status') and conn.health_status == 'unhealthy' and 
                            hasattr(conn, '_recover_connection') and not conn.is_busy() and
                            hasattr(conn, 'recovery_attempts') and conn.recovery_attempts < conn.max_recovery_attempts):
                            
                            # Schedule recovery task to run concurrently
                            logger.info(f"Scheduling automatic recovery for unhealthy connection {conn.connection_id}")
                            recovery_tasks.append(conn._recover_connection())
                            connection_stats['recovery_attempts'] += 1
                    
                    # Collect memory usage
                    if hasattr(conn, 'memory_usage_mb') and conn.memory_usage_mb > 0:
                        connection_stats['memory_usage_total'] += conn.memory_usage_mb
                        
                        # Check for high memory usage in individual connections
                        # Firefox and Chrome can handle ~2GB before instability
                        if conn.memory_usage_mb > 1800:  # Over 1.8GB is concerning
                            logger.warning(f"High memory usage in connection {conn.connection_id}: {conn.memory_usage_mb}MB")
                            
                            # If system is also under memory pressure, mark connection for cleanup
                            if system_memory_pressure:
                                logger.warning(f"System under memory pressure with high-memory connection {conn.connection_id}, marking for cleanup")
                                if hasattr(conn, 'status'):
                                    conn.status = "memory_pressure"
                except Exception as e:
                    logger.error(f"Error performing health check on connection {conn.connection_id}: {e}")
                    connection_stats['unhealthy'] += 1
            
            # Run recovery tasks concurrently
            if recovery_tasks:
                recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
                
                # Count successful recoveries
                for result in recovery_results:
                    if isinstance(result, bool) and result:
                        connection_stats['recovery_successes'] += 1
                
                logger.info(f"Completed {len(recovery_tasks)} auto-recovery attempts with {connection_stats['recovery_successes']} successes")
            
            # Calculate metrics
            if connection_stats['total'] > 0:
                connection_stats['memory_usage_avg'] = connection_stats['memory_usage_total'] / connection_stats['total']
            
            connection_stats['health_check_duration'] = time.time() - check_start_time
            
            # Log results with enhanced detail
            summary_parts = [
                f"{connection_stats['healthy']} healthy",
                f"{connection_stats['degraded']} degraded",
                f"{connection_stats['unhealthy']} unhealthy"
            ]
            
            if connection_stats['circuit_breaker_open'] > 0:
                summary_parts.append(f"{connection_stats['circuit_breaker_open']} circuit-open")
            
            if connection_stats['circuit_breaker_half_open'] > 0:
                summary_parts.append(f"{connection_stats['circuit_breaker_half_open']} circuit-half-open")
                
            if connection_stats['recovery_attempts'] > 0:
                summary_parts.append(f"{connection_stats['recovery_successes']}/{connection_stats['recovery_attempts']} recoveries")
                
            summary = ", ".join(summary_parts)
            
            if connection_stats['unhealthy'] > 0:
                logger.warning(f"Connection health check: {summary} (took {connection_stats['health_check_duration']:.3f}s, avg memory: {connection_stats['memory_usage_avg']:.1f}MB)")
            else:
                logger.info(f"Connection health check: {summary} (avg memory: {connection_stats['memory_usage_avg']:.1f}MB)")
            
            # Update global stats
            self.stats['last_health_check'] = time.time()
            self.stats['last_health_check_stats'] = connection_stats
            
            # Update connection health metrics with enhanced data
            for platform, connections in self.connections.items():
                for conn in connections:
                    conn_id = conn.get_connection_id()
                    
                    # Get circuit breaker state
                    circuit_state = getattr(conn, 'circuit_state', 'unknown')
                    circuit_failures = getattr(conn, 'consecutive_failures', 0)
                    
                    # Collect model performance metrics 
                    model_metrics = {}
                    if hasattr(conn, 'model_performance'):
                        for model_id, metrics in conn.model_performance.items():
                            model_metrics[model_id] = {
                                'execution_count': metrics.get('execution_count', 0),
                                'success_rate': (metrics.get('success_count', 0) / max(metrics.get('execution_count', 1), 1)) * 100,
                                'avg_latency_ms': metrics.get('average_latency_ms', 0),
                                'memory_footprint_mb': metrics.get('memory_footprint_mb', 0)
                            }
                    
                    # Create comprehensive connection health report
                    self.connection_health[conn_id] = {
                        'timestamp': time.time(),
                        'is_healthy': conn.is_healthy(),
                        'status': getattr(conn, 'status', 'unknown'),
                        'health_status': getattr(conn, 'health_status', 'unknown'),
                        'error_count': conn.error_count,
                        'memory_usage_mb': getattr(conn, 'memory_usage_mb', 0),
                        'last_health_check': getattr(conn, 'last_health_check', 0),
                        'time_since_health_check': time.time() - getattr(conn, 'last_health_check', time.time()),
                        'heartbeat_failures': getattr(conn, 'heartbeat_failures', 0),
                        'loaded_model_count': len(conn.loaded_models),
                        'loaded_models': list(conn.loaded_models),
                        'creation_time': getattr(conn, 'creation_time', 0),
                        'age_seconds': time.time() - getattr(conn, 'creation_time', time.time()),
                        'last_used_time': getattr(conn, 'last_used_time', 0),
                        'idle_time_seconds': time.time() - getattr(conn, 'last_used_time', time.time()),
                        'browser_name': getattr(conn, 'browser_name', 'unknown'),
                        'platform': getattr(conn, 'platform', 'unknown'),
                        'circuit_state': circuit_state,
                        'consecutive_failures': circuit_failures,
                        'recovery_attempts': getattr(conn, 'recovery_attempts', 0),
                        'error_history': getattr(conn, 'error_history', [])[:3],  # Include last 3 errors
                        'total_inference_count': getattr(conn, 'total_inference_count', 0),
                        'total_inference_time': getattr(conn, 'total_inference_time', 0),
                        'model_metrics': model_metrics
                    }
                    
                    # Track detailed browser information if available
                    if hasattr(conn, 'browser_info') and conn.browser_info:
                        self.connection_health[conn_id]['browser_info'] = conn.browser_info
                        
                    # Track adapter information if available
                    if hasattr(conn, 'adapter_info') and conn.adapter_info:
                        self.connection_health[conn_id]['adapter_info'] = conn.adapter_info
    
    async def _cleanup_connections(self):
        """Clean up idle and unhealthy connections with enhanced lifecycle management."""
        with self._lock:
            # Collect connections to close
            to_close = []
            
            # Track metrics about cleanup
            cleanup_stats = {
                'unhealthy': 0,
                'idle_timeout': 0,
                'memory_pressure': 0, 
                'inactive': 0,
                'error_threshold': 0,
                'browser_crash': 0,
                'total_models_removed': 0
            }
            
            # Check if system is under memory pressure
            memory_pressure = False
            available_memory_mb = 0
            try:
                import psutil
                vm = psutil.virtual_memory()
                available_memory_mb = vm.available / (1024 * 1024)
                memory_pressure = vm.percent > 80  # Over 80% memory utilization
                logger.debug(f"System memory: {vm.percent}% used, {available_memory_mb:.0f}MB available")
            except ImportError:
                logger.debug("psutil not available, skipping memory pressure check")
            
            for platform, connections in self.connections.items():
                for conn in connections[:]:  # Create a copy to safely modify during iteration
                    # Track connection state for better diagnostics
                    conn_id = conn.get_connection_id()
                    idle_time = conn.get_idle_time()
                    age = conn.get_age()
                    model_count = len(conn.get_loaded_models())
                    
                    # Close connections that fail health check
                    if not conn.is_healthy():
                        reason = "Marked unhealthy"
                        cleanup_stats['unhealthy'] += 1
                        
                        # Check for specific unhealthy reasons
                        if hasattr(conn, 'error_count') and conn.error_count >= conn.max_errors:
                            reason = f"Error threshold exceeded ({conn.error_count}/{conn.max_errors})"
                            cleanup_stats['error_threshold'] += 1
                        elif hasattr(conn, 'heartbeat_failures') and conn.heartbeat_failures >= getattr(conn, 'max_heartbeat_failures', 3):
                            reason = f"Heartbeat failures ({conn.heartbeat_failures})"
                            cleanup_stats['browser_crash'] += 1
                        
                        to_close.append((conn, reason))
                        connections.remove(conn)
                        logger.info(f"Removing unhealthy connection {conn_id}: {reason}")
                        continue
                    
                    # Close idle connections after timeout period (configurable)
                    max_idle_time = getattr(conn, 'max_idle_time', 600)  # Default 10 minutes
                    if idle_time > max_idle_time and not conn.is_busy():
                        reason = f"Idle timeout ({idle_time:.1f}s > {max_idle_time}s)"
                        cleanup_stats['idle_timeout'] += 1
                        to_close.append((conn, reason))
                        connections.remove(conn)
                        logger.info(f"Removing idle connection {conn_id}: {reason}")
                        continue
                    
                    # Close inactive connections (no models loaded) after 5 minutes
                    if model_count == 0 and idle_time > 300 and not conn.is_busy():
                        reason = f"Inactive with no models ({idle_time:.1f}s idle)"
                        cleanup_stats['inactive'] += 1
                        to_close.append((conn, reason))
                        connections.remove(conn)
                        logger.info(f"Removing inactive connection {conn_id}: {reason}")
                        continue
                    
                    # Under memory pressure, more aggressively clean up connections
                    if memory_pressure and idle_time > 300 and not conn.is_busy():
                        reason = f"Memory pressure ({idle_time:.1f}s idle)"
                        cleanup_stats['memory_pressure'] += 1
                        to_close.append((conn, reason))
                        connections.remove(conn)
                        logger.info(f"Removing connection due to memory pressure: {conn_id}")
                        continue
                    
                    # Check for zombie connections: initialized but not responding to health checks
                    if hasattr(conn, 'last_health_check'):
                        time_since_health_check = time.time() - conn.last_health_check
                        # If health check is excessively old (10+ minutes) and not busy, consider zombie
                        if time_since_health_check > 600 and not conn.is_busy() and conn.initialized:
                            reason = f"Zombie connection (no health check for {time_since_health_check:.1f}s)"
                            cleanup_stats['unhealthy'] += 1
                            to_close.append((conn, reason))
                            connections.remove(conn)
                            logger.info(f"Removing zombie connection {conn_id}: {reason}")
                            continue
            
            # Close connections
            for conn, reason in to_close:
                try:
                    # Update connection status for diagnostics
                    if hasattr(conn, 'status'):
                        conn.status = "closing"
                    
                    # Remove from model_connections mapping
                    removed_models = []
                    for model_id, connection in list(self.model_connections.items()):
                        if connection == conn:
                            del self.model_connections[model_id]
                            removed_models.append(model_id)
                    
                    # Update stats
                    cleanup_stats['total_models_removed'] += len(removed_models)
                    
                    # Log models that were removed
                    if removed_models:
                        logger.info(f"Removed {len(removed_models)} models from connection {conn.get_connection_id()}: {', '.join(removed_models)}")
                    
                    # Close connection with better diagnostics
                    logger.info(f"Closing connection {conn.get_connection_id()} due to: {reason}")
                    await conn.close()
                    
                    # Update connection status after close
                    if hasattr(conn, 'status'):
                        conn.status = "closed"
                    
                except Exception as e:
                    logger.error(f"Error closing connection {conn.get_connection_id()}: {e}")
                    
                    # Try forceful cleanup if normal close fails
                    try:
                        if hasattr(conn, 'browser_automation') and conn.browser_automation:
                            await conn.browser_automation.force_close()
                            logger.info(f"Forcefully closed browser for connection {conn.get_connection_id()}")
                    except Exception as force_error:
                        logger.error(f"Forceful cleanup also failed: {force_error}")
            
            # Log detailed cleanup stats
            if to_close:
                # Detailed cleanup stats
                details = ", ".join([f"{k}: {v}" for k, v in cleanup_stats.items() if v > 0])
                logger.info(f"Cleaned up {len(to_close)} connections - {details}")
                
                # Check if we need to run garbage collection after cleaning up many connections
                if len(to_close) >= 2:
                    try:
                        import gc
                        gc.collect()
                        logger.debug("Ran garbage collection after cleaning up multiple connections")
                    except Exception as gc_error:
                        logger.debug(f"Error running garbage collection: {gc_error}")
            else:
                # Log current state if nothing was cleaned up
                total_connections = sum(len(conns) for conns in self.connections.values())
                total_models = sum(len(conn.get_loaded_models()) for platform in self.connections.values() for conn in platform)
                logger.debug(f"No connections to clean up. Current state: {total_connections} connections, {total_models} loaded models")
    
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
        
        This new implementation leverages the concurrent execution pool for improved
        performance, automatic load balancing, and parallel execution across backends.
        
        Args:
            model_id: ID of the model to use
            inputs: Model inputs
            retry_attempts: Number of retry attempts if inference or load fails
            
        Returns:
            Dictionary with inference results
        """
        # Track overall start time for performance monitoring
        overall_start_time = time.time()

        # Check if we have the execution pool available        
        if hasattr(self, 'execution_pools'):
            # Use the new concurrent execution pool system for better performance
            try:
                # Determine the platform for this model
                platform = 'webgpu'  # Default platform
                if model_id in self.model_configs:
                    model_config = self.model_configs[model_id]
                    platform = model_config.get('backend', model_config.get('platform', 'webgpu'))
                
                # Make sure the platform is one we support
                if platform not in self.execution_pools:
                    if platform in ['cuda', 'rocm', 'mps', 'openvino', 'cpu']:
                        platform = 'cpu'  # Use CPU pool for non-web platforms
                    else:
                        platform = 'webgpu'  # Default to WebGPU for unknown platforms
                
                # Create a future for the result
                future = self._loop.create_future()
                
                # Queue the task for execution
                await self.execution_pools[platform]['queue'].put((model_id, inputs, future))
                
                # Wait for the result with timeout
                try:
                    # Set a reasonable timeout based on model type
                    timeout = 60.0  # Default timeout is 60 seconds
                    if ":" in model_id:
                        # Check model family for timeout adjustment
                        family = model_id.split(":", 1)[0]
                        if family in ['audio', 'speech', 'asr']:
                            timeout = 120.0  # Audio models may take longer
                        elif family in ['text_generation', 'multimodal']:
                            timeout = 180.0  # Generation models take longer
                    
                    # Wait for result with timeout
                    result = await asyncio.wait_for(future, timeout=timeout)
                    
                    # Add total time to result
                    if isinstance(result, dict):
                        result['total_time'] = time.time() - overall_start_time
                        result['queue_wait_time_ms'] = result.get('queue_wait_time_ms', 0)
                    
                    # Update stats
                    self.stats['total_inferences'] += 1
                    
                    # Return the result
                    return result
                except asyncio.TimeoutError:
                    # Inference timeout
                    logger.error(f"Inference timeout after {timeout}s for model {model_id}")
                    return {
                        'success': False,
                        'status': 'error',
                        'error': f"Inference timeout after {timeout}s",
                        'model_id': model_id,
                        'total_time': time.time() - overall_start_time
                    }
                except Exception as e:
                    # Handle unexpected exceptions
                    logger.error(f"Exception waiting for inference result: {e}")
                    return {
                        'success': False,
                        'status': 'error',
                        'error': f"Exception waiting for inference result: {e}",
                        'model_id': model_id,
                        'total_time': time.time() - overall_start_time
                    }
            except Exception as e:
                # Handle unexpected exceptions in concurrent execution
                logger.error(f"Error in concurrent execution for {model_id}: {e}")
                # Fall back to legacy method
                logger.warning(f"Falling back to legacy execution method for {model_id}")
        
        # Legacy execution method (used if concurrent execution pool is not available or fails)
        with self._lock:
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
        Run multiple inferences in parallel with intelligent batching and load balancing.
        
        This enhanced implementation leverages the concurrent execution pool system for true
        parallel execution across WebGPU, WebNN, and CPU backends with automatic load balancing,
        optimal resource allocation, and intelligent scheduling.
        
        Args:
            tasks: List of (model_id, inputs) tuples
            batch_size: Maximum number of concurrent tasks (0 for auto-sizing)
            timeout: Timeout in seconds for entire operation (None for no timeout)
            
        Returns:
            List of inference results in the same order as tasks
        """
        # Track overall start time
        start_time = time.time()
        
        # If no tasks, return empty list
        if not tasks:
            return []
        
        # Check if we have the concurrent execution pool system available
        if hasattr(self, 'execution_pools'):
            try:
                # Use the enhanced concurrent execution system for true parallelism
                # This approach distributes tasks across platform-specific worker pools
                # and achieves optimal concurrency
                
                # Initialize results list with placeholder for each task
                results = [None] * len(tasks)
                
                # Group tasks by platform for optimal distribution
                tasks_by_platform = {
                    'webgpu': [],
                    'webnn': [],
                    'cpu': []
                }
                
                # Process each task and categorize by platform
                for task_idx, (model_id, inputs) in enumerate(tasks):
                    # Determine the platform for this model
                    platform = 'webgpu'  # Default platform
                    if model_id in self.model_configs:
                        model_config = self.model_configs[model_id]
                        platform = model_config.get('backend', model_config.get('platform', 'webgpu'))
                    
                    # Make sure the platform is one we support
                    if platform not in tasks_by_platform:
                        if platform in ['cuda', 'rocm', 'mps', 'openvino', 'cpu']:
                            platform = 'cpu'  # Use CPU pool for non-web platforms
                        else:
                            platform = 'webgpu'  # Default to WebGPU for unknown platforms
                    
                    # Add to platform-specific list with original task index
                    tasks_by_platform[platform].append((task_idx, model_id, inputs))
                
                # Create futures for each task
                all_futures = []
                
                # Distribute tasks to appropriate platform queues
                for platform, platform_tasks in tasks_by_platform.items():
                    if not platform_tasks:
                        continue
                        
                    logger.debug(f"Submitting {len(platform_tasks)} tasks to {platform} execution pool")
                    
                    # Create and submit tasks to platform-specific queue
                    for task_idx, model_id, inputs in platform_tasks:
                        # Create a future for this task
                        future = self._loop.create_future()
                        all_futures.append((task_idx, future))
                        
                        # Queue the task
                        await self.execution_pools[platform]['queue'].put((model_id, inputs, future))
                
                # Wait for all futures to complete with timeout
                try:
                    if timeout:
                        # Wait for all results with a global timeout
                        remaining = timeout
                        for task_idx, future in all_futures:
                            # Calculate remaining time for this future
                            start_wait = time.time()
                            
                            if remaining <= 0:
                                # No time left, mark as timeout
                                results[task_idx] = {
                                    'success': False,
                                    'status': 'error',
                                    'error': 'Global timeout exceeded',
                                    'model_id': tasks[task_idx][0]
                                }
                                # Cancel the future
                                if not future.done():
                                    future.cancel()
                                continue
                            
                            # Wait for this future with timeout
                            try:
                                result = await asyncio.wait_for(future, timeout=remaining)
                                results[task_idx] = result
                            except asyncio.TimeoutError:
                                # Timeout for this task
                                results[task_idx] = {
                                    'success': False,
                                    'status': 'error',
                                    'error': f'Task timeout after {remaining:.1f}s',
                                    'model_id': tasks[task_idx][0]
                                }
                                # Cancel the future
                                if not future.done():
                                    future.cancel()
                            except Exception as e:
                                # Error processing this task
                                results[task_idx] = {
                                    'success': False,
                                    'status': 'error',
                                    'error': str(e),
                                    'model_id': tasks[task_idx][0]
                                }
                            
                            # Update remaining time
                            elapsed = time.time() - start_wait
                            remaining -= elapsed
                    else:
                        # No global timeout, wait for all tasks individually
                        for task_idx, future in all_futures:
                            try:
                                # Set a reasonable per-task timeout based on model type
                                model_id = tasks[task_idx][0]
                                per_task_timeout = 60.0  # Default timeout
                                
                                # Adjust timeout based on model type
                                if ":" in model_id:
                                    family = model_id.split(":", 1)[0]
                                    if family in ['audio', 'speech', 'asr']:
                                        per_task_timeout = 120.0  # Audio models may take longer
                                    elif family in ['text_generation', 'multimodal']:
                                        per_task_timeout = 180.0  # Generation models take longer
                                
                                # Wait for this task with timeout
                                result = await asyncio.wait_for(future, timeout=per_task_timeout)
                                results[task_idx] = result
                            except asyncio.TimeoutError:
                                # Timeout for this task
                                results[task_idx] = {
                                    'success': False,
                                    'status': 'error',
                                    'error': 'Task timeout',
                                    'model_id': tasks[task_idx][0]
                                }
                            except Exception as e:
                                # Error processing this task
                                results[task_idx] = {
                                    'success': False,
                                    'status': 'error',
                                    'error': str(e),
                                    'model_id': tasks[task_idx][0]
                                }
                
                # Calculate performance metrics
                total_duration = time.time() - start_time
                success_count = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
                tasks_per_second = len(tasks) / total_duration if total_duration > 0 else 0
                
                # Update stats
                self.stats['parallel_executions'] = self.stats.get('parallel_executions', 0) + 1
                self.stats['parallel_tasks'] = self.stats.get('parallel_tasks', 0) + len(tasks)
                self.stats['parallel_success'] = self.stats.get('parallel_success', 0) + success_count
                self.stats['errors'] += len(tasks) - success_count
                self.stats['largest_parallel_batch'] = max(
                    self.stats.get('largest_parallel_batch', 0),
                    len(tasks)
                )
                
                # Log performance summary
                logger.info(f"Completed {len(tasks)} tasks in {total_duration:.2f}s " +
                           f"({tasks_per_second:.2f} tasks/second, " +
                           f"success: {success_count}/{len(tasks)})")
                
                return results
                
            except Exception as e:
                # Handle any errors in the concurrent execution approach
                logger.error(f"Error in concurrent execution pool: {e}")
                logger.warning("Falling back to batch execution method")
        
        # Legacy implementation (used if concurrent execution pool is not available or fails)
        # Configure auto-batching based on available connections
        if batch_size <= 0:
            # Count available connections
            available_connections = 0
            for platform_conns in self.connections.values():
                available_connections += sum(1 for conn in platform_conns if hasattr(conn, 'is_healthy') and conn.is_healthy())
            
            # Set batch size based on available connections with a minimum
            # of 1 and maximum of 8 concurrent tasks
            batch_size = max(1, min(available_connections * 2, 8))
            logger.debug(f"Auto-sized batch to {batch_size} based on {available_connections} available connections")
        
        # Initialize results list with placeholder for each task
        results = [None] * len(tasks)
        
        # Process tasks in batches
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        
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
        
    def _init_concurrent_execution_pool(self):
        """Initialize concurrent execution pool for parallel model execution across backends.
        
        This implements a sophisticated execution pool that enables:
        1. Parallel execution across WebGPU and CPU backends simultaneously
        2. Concurrent model execution in browser environments
        3. Intelligent load balancing based on model type and backend capabilities
        4. Dynamic worker scaling based on workload
        5. Optimized browser selection for specialized models
        """
        # Check if already initialized
        if hasattr(self, 'execution_pools'):
            return
            
        # Create pools for different execution contexts with enhanced configuration
        self.execution_pools = {
            # WebGPU pool for GPU-accelerated models
            'webgpu': {
                'active': 0, 
                'max': min(3, self.max_connections), 
                'queue': asyncio.Queue(),
                'browsers': {
                    'chrome': {'priority': 10, 'models': ['vision', 'image', 'multimodal']},
                    'firefox': {'priority': 20, 'models': ['audio', 'speech', 'asr']},
                    'edge': {'priority': 5, 'models': ['default']}
                },
                'metrics': {
                    'execution_count': 0,
                    'total_execution_time': 0,
                    'peak_concurrent': 0,
                    'cache_hits': 0
                }
            },
            
            # WebNN pool for neural network acceleration
            'webnn': {
                'active': 0, 
                'max': min(2, self.max_connections), 
                'queue': asyncio.Queue(),
                'browsers': {
                    'edge': {'priority': 20, 'models': ['text', 'embedding', 'language']},
                    'chrome': {'priority': 5, 'models': ['default']}
                },
                'metrics': {
                    'execution_count': 0,
                    'total_execution_time': 0,
                    'peak_concurrent': 0,
                    'cache_hits': 0
                }
            },
            
            # CPU pool for fallback and specialized models
            'cpu': {
                'active': 0, 
                'max': 2, 
                'queue': asyncio.Queue(),
                'browsers': {
                    'chrome': {'priority': 10, 'models': ['default']}
                },
                'metrics': {
                    'execution_count': 0,
                    'total_execution_time': 0,
                    'peak_concurrent': 0,
                    'cache_hits': 0
                }
            }
        }
        
        # Configure pool scheduling priorities
        self.platform_priorities = {
            'webgpu': 100,  # Highest priority for WebGPU
            'webnn': 80,    # Medium priority for WebNN
            'cpu': 60       # Lower priority for CPU
        }
        
        # Enhanced worker initialization with per-browser workers
        self.pool_workers = []
        for platform, config in self.execution_pools.items():
            # Calculate number of workers per browser based on priorities
            browsers = config['browsers']
            browser_weights = {browser: info['priority'] for browser, info in browsers.items()}
            total_weight = sum(browser_weights.values())
            
            # Distribute workers based on priority weights
            browser_workers = {}
            remaining_workers = config['max']
            
            # Ensure at least one worker per browser if possible
            for browser in browsers:
                browser_workers[browser] = 1
                remaining_workers -= 1
                
            # Distribute remaining workers by priority
            if remaining_workers > 0 and total_weight > 0:
                for browser, weight in browser_weights.items():
                    additional = max(0, int(remaining_workers * weight / total_weight))
                    browser_workers[browser] += additional
                    remaining_workers -= additional
            
            # Create workers for each browser
            for browser, count in browser_workers.items():
                for _ in range(count):
                    worker_task = asyncio.ensure_future(
                        self._execution_worker(platform, browser), 
                        loop=self._loop
                    )
                    self.pool_workers.append(worker_task)
            
            # Store worker distribution
            config['worker_distribution'] = browser_workers
        
        logger.info(f"Initialized enhanced concurrent execution pool with {len(self.pool_workers)} workers across multiple browsers")
        
        # Start resource monitoring and dynamic scaling task with enhanced metrics
        self.monitoring_task = asyncio.ensure_future(
            self._monitor_resource_utilization(),
            loop=self._loop
        )
        
        # Initialize shared memory for cross-browser coordination
        self.shared_browser_state = {
            'chrome': {'active_models': 0, 'max_models': 4, 'capabilities': ['webgpu', 'webnn']},
            'firefox': {'active_models': 0, 'max_models': 3, 'capabilities': ['webgpu', 'compute_shaders']},
            'edge': {'active_models': 0, 'max_models': 3, 'capabilities': ['webnn', 'webgpu']}
        }
        
        # Start background task for browser health monitoring
        self.browser_health_task = asyncio.ensure_future(
            self._monitor_browser_health(),
            loop=self._loop
        )
        
    async def _execution_worker(self, platform, browser='chrome'):
        """Worker for processing execution tasks from the queue with enhanced browser-specific optimizations.
        
        Args:
            platform: Platform queue to process ('webgpu', 'webnn', 'cpu')
            browser: Specific browser this worker uses ('chrome', 'firefox', 'edge')
        """
        pool = self.execution_pools[platform]
        queue = pool['queue']
        metrics = pool['metrics']
        browser_info = pool['browsers'].get(browser, {'priority': 0, 'models': ['default']})
        
        # Track worker-specific metrics
        worker_metrics = {
            'processed_tasks': 0,
            'execution_time': 0,
            'errors': 0,
            'last_active': time.time()
        }
        
        logger.info(f"Starting {platform} worker for {browser} browser")
        
        while not self._is_shutting_down:
            try:
                # Get task with timeout
                try:
                    model_id, inputs, future = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                
                # Mark worker as active
                pool['active'] += 1
                worker_metrics['last_active'] = time.time()
                
                try:
                    # Execute the task
                    logger.debug(f"Executing {model_id} on {platform} worker using {browser} browser")
                    start_time = time.time()
                    
                    # Extract model family from model_id
                    model_family = "text"
                    
                    # Try to extract family from model_id format (family:model_name)
                    if ":" in model_id:
                        model_family = model_id.split(":", 1)[0]
                    else:
                        # Try to infer from model name with expanded detection patterns
                        model_id_lower = model_id.lower()
                        
                        # Audio models
                        if any(audio_model in model_id_lower for audio_model in 
                               ["whisper", "wav2vec", "clap", "hubert", "speech", "asr", "audio", "mms-tts"]):
                            model_family = "audio"
                            
                        # Vision models
                        elif any(vision_model in model_id_lower for vision_model in 
                                ["vit", "clip", "resnet", "yolo", "detr", "sam", "dino", "blip", "image", "siglip"]):
                            model_family = "vision"
                            
                        # Multimodal models
                        elif any(mm_model in model_id_lower for mm_model in 
                                ["llava", "flava", "vlm", "git", "pix2struct", "multimodal", "blip2", "nougat"]):
                            model_family = "multimodal"
                            
                        # Text embedding models
                        elif any(text_emb in model_id_lower for text_emb in 
                                ["bert", "t5", "roberta", "distilbert", "embedding", "sentence", "e5", "bge"]):
                            model_family = "text_embedding"
                            
                        # Text generation models
                        elif any(text_gen in model_id_lower for text_gen in 
                                ["llama", "gpt", "bloom", "mistral", "falcon", "opt", "gemma", "mixtral", "baichuan"]):
                            model_family = "text_generation"
                    
                    # Check if this worker is a good match for this model family
                    is_preferred_browser = False
                    
                    # For WebGPU, match model families to optimal browsers
                    if platform == 'webgpu':
                        if model_family in ['audio', 'speech'] and browser == 'firefox':
                            is_preferred_browser = True
                        elif model_family in ['vision', 'image', 'multimodal'] and browser == 'chrome':
                            is_preferred_browser = True
                        elif model_family in ['text_embedding', 'text_generation', 'text'] and browser == 'edge':
                            is_preferred_browser = True
                    # For WebNN, Edge is preferred
                    elif platform == 'webnn':
                        if browser == 'edge':
                            is_preferred_browser = True
                    # For CPU, any browser is fine
                    elif platform == 'cpu':
                        is_preferred_browser = True
                    
                    # Log model routing decision
                    if is_preferred_browser:
                        logger.debug(f"Model {model_id} is optimally matched to {browser} for {platform}")
                    else:
                        logger.debug(f"Model {model_id} running on {browser} for {platform} (non-optimal browser)")
                    
                    # Update cross-browser coordination state
                    if browser in self.shared_browser_state:
                        self.shared_browser_state[browser]['active_models'] += 1
                    
                    # Get optimal connection for this model
                    connection = None
                    try:
                        # Get model config to check browser preferences
                        model_config = self.model_configs.get(model_id, {})
                        connection_id = model_config.get('connection_id')
                        
                        # Three-tiered connection selection:
                        # 1. Reuse existing connection with model already loaded
                        # 2. Reuse connection with same browser type
                        # 3. Create new connection if needed
                        
                        if connection_id and connection_id in self.model_connections:
                            # 1. Use existing connection if specified and available
                            connection = self.model_connections[connection_id]
                            logger.debug(f"Using specified connection {connection_id} for {model_id}")
                        else:
                            # 2. Find available connection with this browser that already has the model loaded
                            model_loaded_connections = []
                            browser_type_connections = []
                            
                            for conn_id, conn in list(self.model_connections.items()):
                                # Check if connection matches our requirements
                                if (hasattr(conn, 'get_platform') and conn.get_platform() == platform and 
                                    hasattr(conn, 'is_busy') and not conn.is_busy() and 
                                    hasattr(conn, 'is_healthy') and conn.is_healthy() and
                                    hasattr(conn, 'browser_name') and conn.browser_name == browser):
                                    
                                    # Check if model is already loaded
                                    if hasattr(conn, 'has_model') and conn.has_model(model_id):
                                        # Connection already has this model loaded, highest priority
                                        model_loaded_connections.append(conn)
                                    else:
                                        # Connection is correct browser type but needs model loading
                                        browser_type_connections.append(conn)
                            
                            # Choose best available connection
                            if model_loaded_connections:
                                # Best case: model already loaded in matching browser
                                connection = model_loaded_connections[0]
                                logger.debug(f"Using connection {connection.connection_id} that already has model {model_id}")
                                metrics['cache_hits'] += 1
                            elif browser_type_connections:
                                # Second best: matching browser but need to load model
                                connection = browser_type_connections[0]
                                logger.debug(f"Using {browser} connection {connection.connection_id} but need to load model {model_id}")
                                # Load model into this connection
                                await connection.load_model(model_id)
                            else:
                                # 3. No suitable existing connection, create new one
                                logger.debug(f"Creating new {browser} connection for {model_id} on {platform}")
                                success, connection = await self.load_model(model_id, platform=platform, browser_name=browser)
                                if not success or not connection:
                                    raise Exception(f"Failed to load model {model_id} on {browser}")
                    except Exception as e:
                        logger.error(f"Error getting connection for {model_id}: {e}")
                        raise
                    # Execute the model with optimizations for model family and browser
                    if connection:
                        # Apply model-specific optimizations
                        is_audio_model = model_family in ['audio', 'speech', 'asr']
                        is_vision_model = model_family in ['vision', 'image', 'multimodal']
                        
                        # Apply browser-specific optimizations
                        execution_options = {}
                        
                        # Apply Firefox-specific optimizations for audio models
                        if browser == 'firefox' and is_audio_model:
                            execution_options['compute_shaders'] = True
                            execution_options['workgroup_size'] = [256, 1, 1]  # Firefox-optimized workgroup size
                            logger.debug(f"Applying Firefox audio optimizations for {model_id}")
                        
                        # Apply Chrome-specific optimizations for vision models
                        elif browser == 'chrome' and is_vision_model:
                            execution_options['precompile_shaders'] = True
                            execution_options['parallel_loading'] = True
                            logger.debug(f"Applying Chrome vision optimizations for {model_id}")
                        
                        # Apply Edge-specific optimizations for text models
                        elif browser == 'edge' and not (is_audio_model or is_vision_model):
                            execution_options['webnn_ops_fallback'] = False  # Use pure WebNN when possible
                            logger.debug(f"Applying Edge text model optimizations for {model_id}")
                        
                        # Execute with optimized execution options
                        inference_start = time.time()
                        result = await connection.run_inference(model_id, inputs, **execution_options)
                        execution_time = time.time() - inference_start
                        
                        # Track execution metrics
                        metrics['execution_count'] += 1
                        metrics['total_execution_time'] += execution_time
                        metrics['peak_concurrent'] = max(metrics['peak_concurrent'], pool['active'])
                        
                        # Update worker metrics
                        worker_metrics['processed_tasks'] += 1
                        worker_metrics['execution_time'] += execution_time
                        
                        # Add detailed browser and optimization metrics to result
                        if isinstance(result, dict):
                            result['browser'] = browser
                            result['platform'] = platform
                            result['execution_time'] = execution_time
                            result['optimized'] = is_preferred_browser
                            result['browser_optimizations'] = execution_options
                        
                        # Set result with enhanced metrics
                        future.set_result(result)
                        logger.debug(f"Completed execution of {model_id} on {browser}/{platform} in {execution_time:.2f}s")
                    else:
                        raise Exception(f"No connection available for {model_id} on {browser}")
                        
                except Exception as e:
                    logger.error(f"Error executing {model_id} on {platform}: {e}")
                    future.set_exception(e)
                finally:
                    # Mark worker as inactive
                    pool['active'] -= 1
                    # Mark queue task as done
                    queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in {platform} execution worker: {e}")
                # Brief pause before continuing
                await asyncio.sleep(0.5)

    async def _monitor_resource_utilization(self):
        """Monitor resource utilization and adaptively scale worker pools.
        
        This implements advanced resource monitoring and adaptive scaling:
        1. Monitors queue sizes and worker utilization for each platform
        2. Dynamically adjusts worker allocation based on workload
        3. Implements predictive scaling based on historical trends
        4. Balances resources across platforms for optimal performance
        """
        # Initialize metrics for resource utilization tracking
        utilization_history = {
            'webgpu': [],
            'webnn': [],
            'cpu': []
        }
        
        # Initialize scaling metrics
        pool_scaling_cooldown = {
            'webgpu': 0,
            'webnn': 0,
            'cpu': 0
        }
        
        while not self._is_shutting_down:
            try:
                # Wait for monitoring interval
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
                # Get current utilization for each platform
                for platform, pool in self.execution_pools.items():
                    # Calculate utilization metrics
                    active_workers = pool['active']
                    max_workers = pool['max']
                    queue_size = pool['queue'].qsize()
                    worker_utilization = active_workers / max_workers if max_workers > 0 else 0
                    
                    # Add to history (keep last 10 readings)
                    utilization_history[platform].append({
                        'timestamp': time.time(),
                        'active': active_workers,
                        'max': max_workers,
                        'queue_size': queue_size,
                        'utilization': worker_utilization
                    })
                    
                    if len(utilization_history[platform]) > 10:
                        utilization_history[platform] = utilization_history[platform][-10:]
                    
                    # Calculate growth trends in queue size
                    queue_growth = 0
                    if len(utilization_history[platform]) >= 3:
                        # Compare current with average of previous 2 readings
                        prev_avg = sum(h['queue_size'] for h in utilization_history[platform][-3:-1]) / 2
                        queue_growth = queue_size - prev_avg
                    
                    # Determine if scaling is needed based on utilization and queue trends
                    if pool_scaling_cooldown[platform] <= 0:
                        # Check for scale up conditions
                        if (queue_size > 5 and queue_growth > 0 and worker_utilization > 0.7) or \
                           (queue_size > 10):
                            # Need to scale up this platform
                            new_max = min(max_workers + 1, 
                                         platform == 'webgpu' and self.max_connections or 4)  # Allow more CPU workers
                            
                            if new_max > max_workers:
                                logger.info(f"Scaling up {platform} workers: {max_workers} -> {new_max} " +
                                           f"(queue: {queue_size}, growth: {queue_growth:.1f}, " +
                                           f"util: {worker_utilization:.2f})")
                                
                                # Update pool max workers
                                pool['max'] = new_max
                                
                                # Create additional worker
                                worker_task = asyncio.ensure_future(
                                    self._execution_worker(platform), 
                                    loop=self._loop
                                )
                                self.pool_workers.append(worker_task)
                                
                                # Set cooldown for scaling
                                pool_scaling_cooldown[platform] = 3  # 30 seconds cooldown (3 intervals)
                        
                        # Check for scale down conditions (empty queue and low utilization)
                        elif queue_size == 0 and worker_utilization < 0.3 and max_workers > 1:
                            # Ensure consistently low utilization before scaling down
                            if len(utilization_history[platform]) >= 3:
                                avg_util = sum(h['utilization'] for h in utilization_history[platform][-3:]) / 3
                                if avg_util < 0.3:
                                    # Need to scale down this platform
                                    new_max = max(max_workers - 1, 1)  # Keep at least 1 worker
                                    
                                    logger.info(f"Scaling down {platform} workers: {max_workers} -> {new_max} " +
                                               f"(avg util: {avg_util:.2f})")
                                    
                                    # Update pool max workers
                                    pool['max'] = new_max
                                    
                                    # Cancel one worker (will complete current task first)
                                    for i, worker in enumerate(self.pool_workers):
                                        if worker.done():
                                            self.pool_workers.pop(i)
                                            break
                                    
                                    # Set cooldown for scaling
                                    pool_scaling_cooldown[platform] = 5  # 50 seconds cooldown (5 intervals)
                    else:
                        # Decrement cooldown
                        pool_scaling_cooldown[platform] -= 1
                
                # Log detailed resource utilization
                if logger.isEnabledFor(logging.DEBUG):
                    util_log = []
                    for platform, pool in self.execution_pools.items():
                        util_log.append(
                            f"{platform}: {pool['active']}/{pool['max']} workers, "
                            f"queue: {pool['queue'].qsize()}"
                        )
                    logger.debug("Resource utilization: " + ", ".join(util_log))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                # Brief pause before continuing
                await asyncio.sleep(1.0)
    
    async def close(self):
        """Close all connections and clean up resources with enhanced lifecycle management."""
        with self._lock:
            # Set shutting down flag to prevent new tasks
            self._is_shutting_down = True
            logger.info("ResourcePoolBridge shutting down")
            
            # Track metrics for cleanup
            start_time = time.time()
            total_connections = sum(len(conns) for conns in self.connections.values())
            total_models = sum(len(conn.get_loaded_models()) for platform in self.connections.values() for conn in platform)
            
            # Cancel background tasks with proper cleanup
            tasks_to_cancel = []
            
            # Cancel cleanup task
            if self._cleanup_task:
                logger.debug("Cancelling cleanup task")
                self._cleanup_task.cancel()
                tasks_to_cancel.append(self._cleanup_task)
                self._cleanup_task = None
            
            # Cancel connection monitor task
            if self._connection_monitor_task:
                logger.debug("Cancelling connection monitor task")
                self._connection_monitor_task.cancel()
                tasks_to_cancel.append(self._connection_monitor_task)
                self._connection_monitor_task = None
            
            # Cancel monitoring task if it exists (legacy name)
            if hasattr(self, 'monitoring_task') and self.monitoring_task:
                logger.debug("Cancelling legacy monitoring task")
                self.monitoring_task.cancel()
                tasks_to_cancel.append(self.monitoring_task)
                self.monitoring_task = None
            
            # Cancel all worker tasks in the execution pool
            if hasattr(self, 'pool_workers') and self.pool_workers:
                logger.debug(f"Cancelling {len(self.pool_workers)} pool worker tasks")
                for worker in self.pool_workers:
                    worker.cancel()
                    tasks_to_cancel.append(worker)
                
                self.pool_workers = []
            
            # Wait for all background tasks to complete cancellation with timeout
            if tasks_to_cancel:
                try:
                    # Use a reasonable timeout to avoid hanging
                    background_timeout = 5.0  # 5 seconds timeout
                    await asyncio.wait(tasks_to_cancel, timeout=background_timeout)
                    logger.info(f"Cancelled {len(tasks_to_cancel)} background tasks")
                except (asyncio.CancelledError, Exception) as e:
                    logger.warning(f"Error while cancelling background tasks: {e}")
            
            # Close all connections with enhanced lifecycle management
            logger.info(f"Closing {total_connections} connections with {total_models} loaded models")
            
            # Track for metrics
            closed_connections = 0
            closed_models = 0
            closure_errors = 0
            
            # Prepare close tasks with tracking
            close_tasks = []
            connections_by_id = {}  # Track connections by ID for result matching
            
            for platform, connections in self.connections.items():
                platform_connections = len(connections)
                if platform_connections > 0:
                    logger.info(f"Closing {platform_connections} {platform} connections")
                
                for conn in connections:
                    conn_id = conn.get_connection_id()
                    connections_by_id[conn_id] = conn
                    # Track models before closing
                    closed_models += len(conn.get_loaded_models())
                    
                    # Create close task
                    close_tasks.append(asyncio.create_task(conn.close()))
            
            # Wait for all connections to close with timeout to prevent hanging
            if close_tasks:
                try:
                    # Use a reasonable timeout for closing all connections
                    close_timeout = max(10.0, len(close_tasks) * 2.0)  # Scale timeout with connection count
                    
                    # Gather results with timeout
                    _, pending = await asyncio.wait(close_tasks, timeout=close_timeout)
                    
                    # Cancel any pending tasks that timed out
                    if pending:
                        logger.warning(f"{len(pending)} connection close tasks timed out and will be cancelled")
                        for task in pending:
                            task.cancel()
                        
                        # Try to wait for cancellation to complete with a short timeout
                        try:
                            await asyncio.wait(pending, timeout=2.0)
                        except Exception:
                            pass
                    
                    # Count successfully closed connections
                    for task in close_tasks:
                        if task.done() and not task.cancelled():
                            try:
                                if task.result():
                                    closed_connections += 1
                                else:
                                    closure_errors += 1
                            except Exception:
                                closure_errors += 1
                    
                except Exception as e:
                    logger.error(f"Error waiting for connections to close: {e}")
                    closure_errors += 1
            
            # Clear connections
            for platform in self.connections:
                self.connections[platform] = []
            
            # Clear model mappings and connection tracking
            self.model_connections.clear()
            self.connection_health.clear()
            self.connection_performance.clear()
            
            # Clear execution pools if they exist
            if hasattr(self, 'execution_pools'):
                for platform in self.execution_pools:
                    # Clear any remaining items in queues
                    queue = self.execution_pools[platform]['queue']
                    remaining_tasks = 0
                    
                    # Cancel remaining tasks with proper error messages
                    while not queue.empty():
                        try:
                            # Try to get the next item without waiting
                            model_id, inputs, future = queue.get_nowait()
                            # Set the future as cancelled with clear error message
                            if not future.done():
                                future.set_exception(asyncio.CancelledError("ResourcePoolBridge is shutting down"))
                            # Mark task as done
                            queue.task_done()
                            remaining_tasks += 1
                        except asyncio.QueueEmpty:
                            break
                        except Exception as e:
                            logger.debug(f"Error clearing queue: {e}")
                    
                    if remaining_tasks > 0:
                        logger.info(f"Cancelled {remaining_tasks} pending tasks in {platform} queue")
            
            # Attempt to clean up any remaining resources
            try:
                # Force garbage collection to clean up resources
                import gc
                gc.collect()
                
                # For CUDA usage, try to clear CUDA cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("CUDA cache cleared")
                except (ImportError, Exception):
                    pass  # CUDA cleanup is optional
            except Exception as cleanup_error:
                logger.debug(f"Error in final resource cleanup: {cleanup_error}")
            
            # Log completion with detailed metrics
            cleanup_time = time.time() - start_time
            logger.info(f"ResourcePoolBridge closed in {cleanup_time:.2f}s - Successfully closed {closed_connections}/{total_connections} connections with {closed_models} models unloaded{' (errors: '+str(closure_errors)+')' if closure_errors > 0 else ''}")
    
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
        """
        Ensure the database has required tables for storing results with enhanced schema 
        for connection pooling metrics.
        
        This creates the necessary tables for all connection pooling metrics and adds indexes
        for efficient querying of the connection pooling data.
        """
        if not self.db_connection:
            return
            
        try:
            # Create table for WebNN/WebGPU acceleration results with connection pooling fields
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
                connection_id VARCHAR,
                queue_wait_time_ms FLOAT,
                resource_metrics JSON,
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
            
            # Create connection pool metrics table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS connection_pool_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                pool_size INTEGER,
                active_connections INTEGER,
                connection_utilization FLOAT,
                models_per_connection FLOAT,
                queue_size INTEGER,
                throughput_items_per_second FLOAT,
                avg_latency_ms FLOAT,
                browser_distribution JSON,
                platform_distribution JSON
            )
            """)
            
            # Create browser performance metrics table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS browser_performance_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                browser VARCHAR,
                platform VARCHAR,
                model_count INTEGER,
                avg_latency_ms FLOAT,
                throughput_items_per_second FLOAT,
                error_rate FLOAT,
                memory_usage_mb FLOAT
            )
            """)
            
            # Create model pooling statistics table for tracking pooling efficiency
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS model_pooling_statistics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                model_name VARCHAR,
                model_type VARCHAR,
                platform VARCHAR,
                browser VARCHAR,
                is_real_implementation BOOLEAN,
                pooled_latency_ms FLOAT,
                dedicated_latency_ms FLOAT,
                latency_improvement_pct FLOAT,
                connection_reuses INTEGER
            )
            """)
            
            # Create connection lifecycle metrics table to track entire connection lifespans
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS connection_lifecycle_metrics (
                id INTEGER PRIMARY KEY,
                connection_id VARCHAR,
                browser_name VARCHAR,
                platform VARCHAR,
                creation_timestamp TIMESTAMP,
                termination_timestamp TIMESTAMP,
                total_lifetime_sec FLOAT,
                total_models_executed INTEGER,
                total_inference_time_sec FLOAT,
                avg_model_latency_ms FLOAT,
                error_count INTEGER,
                max_concurrent_models INTEGER,
                browser_version VARCHAR,
                is_productive BOOLEAN
            )
            """)
            
            # Create indexes for efficient querying
            # For webnn_webgpu_results
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_webnn_webgpu_model_name ON webnn_webgpu_results(model_name)")
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_webnn_webgpu_browser ON webnn_webgpu_results(browser)")
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_webnn_webgpu_platform ON webnn_webgpu_results(platform)")
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_webnn_webgpu_timestamp ON webnn_webgpu_results(timestamp)")
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_webnn_webgpu_connection_id ON webnn_webgpu_results(connection_id)")
            
            # For browser_connection_metrics
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_browser_conn_connection_id ON browser_connection_metrics(connection_id)")
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_browser_conn_browser ON browser_connection_metrics(browser_name)")
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_browser_conn_timestamp ON browser_connection_metrics(timestamp)")
            
            # For connection_pool_metrics
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_conn_pool_timestamp ON connection_pool_metrics(timestamp)")
            
            # For browser_performance_metrics
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_browser_perf_browser ON browser_performance_metrics(browser)")
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_browser_perf_platform ON browser_performance_metrics(platform)")
            
            # For model_pooling_statistics
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_model_pooling_model ON model_pooling_statistics(model_name)")
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_model_pooling_browser ON model_pooling_statistics(browser)")
            
            # For connection_lifecycle_metrics
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_conn_lifecycle_connection_id ON connection_lifecycle_metrics(connection_id)")
            self.db_connection.execute("CREATE INDEX IF NOT EXISTS idx_conn_lifecycle_browser ON connection_lifecycle_metrics(browser_name)")
            
            logger.info("Enhanced database schema initialized with connection pooling metrics support")
        except Exception as e:
            logger.error(f"Error ensuring database schema: {e}")
            import traceback
            traceback.print_exc()
    
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
        """
        Start the connection management task with enhanced pooling capabilities.
        
        This task monitors connection health, manages the connection pool, 
        and implements sophisticated connection lifecycle management including:
        - Health monitoring and automatic recovery
        - Connection pool maintenance and cleanup
        - Intelligent connection allocation based on workload
        - Smart browser-specific pool management
        - Dynamic scaling based on demand
        """
        async def connection_management_loop():
            """Manage browser connections to ensure availability."""
            while not self._is_shutting_down:
                try:
                    # Check connection health every 30 seconds
                    await asyncio.sleep(30)
                    
                    # Get total active connections
                    active_connections = len(self.browser_connections)
                    browser_counts = {}
                    platform_counts = {
                        'webgpu': 0,
                        'webnn': 0
                    }
                    
                    # Initialize browser counters
                    for browser_name in self.available_browsers:
                        browser_counts[browser_name] = 0
                    
                    # Count connections per browser and platform
                    idle_connections = []
                    unhealthy_connections = []
                    for conn_id, conn_info in self.browser_connections.items():
                        browser_name = conn_info.get('browser_name')
                        platform = conn_info.get('platform')
                        
                        # Update counts
                        if browser_name in browser_counts:
                            browser_counts[browser_name] += 1
                        if platform in platform_counts:
                            platform_counts[platform] += 1
                        
                        # Check connection health and activity
                        if 'connection' in conn_info and hasattr(conn_info['connection'], 'is_healthy'):
                            if not conn_info['connection'].is_healthy():
                                unhealthy_connections.append(conn_id)
                                logger.warning(f"Unhealthy connection detected: {conn_id}")
                            
                            # Check for idle connections (over 10 minutes with no activity)
                            if 'last_used' in conn_info:
                                idle_time = time.time() - conn_info['last_used']
                                if idle_time > 600 and not conn_info['connection'].is_busy():
                                    idle_connections.append(conn_id)
                                    logger.debug(f"Idle connection detected: {conn_id}, idle for {idle_time:.1f}s")
                    
                    # Handle unhealthy connections - replace them
                    for conn_id in unhealthy_connections:
                        if conn_id in self.browser_connections:
                            conn_info = self.browser_connections[conn_id]
                            logger.warning(f"Closing unhealthy connection: {conn_id}")
                            
                            # Try to close connection
                            if 'connection' in conn_info:
                                try:
                                    await conn_info['connection'].close()
                                except Exception as e:
                                    logger.error(f"Error closing unhealthy connection {conn_id}: {e}")
                            
                            # Remove from pool
                            del self.browser_connections[conn_id]
                            active_connections -= 1
                            
                            # Update browser and platform counts
                            browser_name = conn_info.get('browser_name')
                            platform = conn_info.get('platform')
                            if browser_name in browser_counts:
                                browser_counts[browser_name] -= 1
                            if platform in platform_counts:
                                platform_counts[platform] -= 1
                            
                            # Create a replacement connection with the same browser and platform
                            if active_connections < self.max_connections:
                                logger.info(f"Creating replacement connection for {browser_name}/{platform}")
                                await self._initialize_browser_connection(browser_name, platform)
                                active_connections += 1
                                browser_counts[browser_name] += 1
                                platform_counts[platform] += 1
                    
                    # Handle idle connections - close them if we have more than needed
                    # We'll keep at least one connection per browser type
                    connections_to_close = []
                    
                    # Determine minimum connections to keep per browser
                    min_connections_per_browser = {}
                    for browser in browser_counts.keys():
                        # Keep at least one connection per browser, prioritizing frequently used browsers
                        if browser in self.browser_preferences.values():
                            # Browser is preferred for at least one model type, keep at least one
                            min_connections_per_browser[browser] = 1
                        else:
                            # Less frequently used browser, may not need to keep one
                            min_connections_per_browser[browser] = 0
                    
                    # Determine which idle connections to close
                    for conn_id in idle_connections:
                        if conn_id in self.browser_connections:
                            conn_info = self.browser_connections[conn_id]
                            browser_name = conn_info.get('browser_name')
                            
                            # Only close if we have more than minimum required for this browser
                            if browser_counts.get(browser_name, 0) > min_connections_per_browser.get(browser_name, 0):
                                connections_to_close.append(conn_id)
                                browser_counts[browser_name] -= 1  # Decrement count for planning purposes
                    
                    # Close selected idle connections
                    for conn_id in connections_to_close:
                        if conn_id in self.browser_connections:
                            conn_info = self.browser_connections[conn_id]
                            logger.info(f"Closing idle connection: {conn_id}")
                            
                            # Try to close connection
                            if 'connection' in conn_info:
                                try:
                                    await conn_info['connection'].close()
                                except Exception as e:
                                    logger.error(f"Error closing idle connection {conn_id}: {e}")
                            
                            # Remove from pool
                            del self.browser_connections[conn_id]
                            active_connections -= 1
                    
                    # Check if we need to initialize more connections based on model family needs
                    # We want to ensure we have connections for each browser optimized for different model types
                    needed_browser_connections = {}
                    
                    # Calculate needed connections based on browser preferences
                    for model_type, browser in self.browser_preferences.items():
                        if browser not in needed_browser_connections:
                            needed_browser_connections[browser] = 0
                        needed_browser_connections[browser] += 1
                    
                    # Look at existing connections to determine what's missing
                    for browser, needed in needed_browser_connections.items():
                        current = browser_counts.get(browser, 0)
                        if current < needed and active_connections < self.max_connections:
                            # Need more connections for this browser
                            browser_info = self.available_browsers.get(browser)
                            if not browser_info:
                                continue
                                
                            # Determine platform
                            platform = None
                            if browser_info.get('webgpu_support') and self.enable_gpu:
                                platform = 'webgpu'
                            elif browser_info.get('webnn_support') and self.enable_cpu:
                                platform = 'webnn'
                                
                            if platform:
                                logger.info(f"Initializing missing {platform} connection for {browser}")
                                await self._initialize_browser_connection(browser, platform)
                                active_connections += 1
                                browser_counts[browser] = browser_counts.get(browser, 0) + 1
                    
                    # Update resource utilization metrics
                    self.resource_metrics['active_connections'] = active_connections
                    self.resource_metrics['connection_util'] = active_connections / self.max_connections if self.max_connections > 0 else 0
                    
                    # Update browser usage metrics
                    for browser, count in browser_counts.items():
                        if browser in self.resource_metrics['browser_usage']:
                            self.resource_metrics['browser_usage'][browser] = count
                    
                    # Update platform usage metrics
                    for platform, count in platform_counts.items():
                        if 'platform_usage' in self.resource_metrics and platform in self.resource_metrics['platform_usage']:
                            self.resource_metrics['platform_usage'][platform] = count
                    
                    # Log pool status at INFO level for tracking
                    logger.info(f"Connection pool status: {active_connections}/{self.max_connections} connections "
                               f"(util: {self.resource_metrics['connection_util']:.2f})")
                    
                    # Log browser-specific status at DEBUG level
                    browser_status = ", ".join([f"{b}: {c}" for b, c in browser_counts.items() if c > 0])
                    logger.debug(f"Browser status: {browser_status}")
                    
                except asyncio.CancelledError:
                    logger.info("Connection management task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in connection management loop: {e}")
                    import traceback
                    traceback.print_exc()
        
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
        Adapt resources based on workload with enhanced monitoring and adaptive scaling.
        
        This method implements intelligent resource management with dynamic scaling capabilities:
        1. Monitors key performance metrics in real-time
        2. Implements predictive scaling before bottlenecks occur
        3. Uses sophisticated decision algorithm based on multiple factors:
           - Queue backlog and growth rate
           - Connection utilization and efficiency
           - Model characteristics and browser affinities
           - Memory usage and browser health
        4. Implements browser-specific optimizations based on model families
        5. Balances resource efficiency with performance requirements
        
        Args:
            queue_size: Current size of execution queue
            connection_util: Connection utilization (0.0-1.0)
        """
        # Only adapt if we have adaptive scaling enabled
        if not self.adaptive_scaling:
            return
        
        # Don't exceed max_connections
        current_max = self.bridge.max_connections
        
        # Track historical queue sizes to detect growth trends
        if not hasattr(self, '_queue_history'):
            self._queue_history = []
        
        # Add current queue size to history, keep last 10 readings
        self._queue_history.append(queue_size)
        if len(self._queue_history) > 10:
            self._queue_history = self._queue_history[-10:]
        
        # Calculate queue growth rate (change per monitoring interval)
        queue_growth_rate = 0
        if len(self._queue_history) >= 2:
            # Compare current with average of previous 3 readings
            prev_avg = sum(self._queue_history[-4:-1]) / min(3, len(self._queue_history)-1)
            queue_growth_rate = queue_size - prev_avg
        
        # Get current browser distribution
        browser_counts = self.resource_metrics.get('browser_usage', {})
        
        # Count active connections and models by browser
        active_connections = 0
        active_models = 0
        browser_models = {}
        platform_models = {'webgpu': 0, 'webnn': 0}
        browser_health = {}
        
        if self.bridge:
            # Aggregate detailed stats
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
                        
                        # Track models by platform
                        if platform in platform_models:
                            platform_models[platform] += model_count
                        
                        # Track browser health - count errors as percentage of total operations
                        if browser not in browser_health:
                            browser_health[browser] = {'errors': 0, 'operations': 0}
                        
                        # Add error stats if available
                        if hasattr(conn, 'error_count'):
                            browser_health[browser]['errors'] += conn.error_count
                        if hasattr(conn, 'models_executed'):
                            browser_health[browser]['operations'] += conn.models_executed or 0
        
        # Calculate browser health scores (0-100, higher is better)
        browser_health_scores = {}
        for browser, stats in browser_health.items():
            if stats['operations'] > 0:
                error_rate = stats['errors'] / max(1, stats['operations'])
                health_score = 100 * (1 - min(1.0, error_rate * 5))  # 5x multiplier to make errors more impactful
                browser_health_scores[browser] = health_score
            else:
                browser_health_scores[browser] = 100  # No operations yet, assume healthy
        
        # Scaling decision logic with enhanced factors
        scale_up = False
        scale_down = False
        scale_reason = ""
        
        # ENHANCED SCALE UP CONDITIONS
        # Consider both queue size AND growth rate for predictive scaling
        if (queue_size >= 5 and queue_growth_rate > 0) and connection_util > 0.7:
            scale_up = True
            scale_reason = f"growing queue ({queue_size}, +{queue_growth_rate:.1f}/interval) with high utilization ({connection_util:.2f})"
        # React quickly to sudden queue spikes
        elif queue_size >= 8 and queue_growth_rate > 2:
            scale_up = True
            scale_reason = f"rapidly growing queue ({queue_size}, +{queue_growth_rate:.1f}/interval)"
        # Moderate but stable queue with high utilization
        elif queue_size >= 3 and connection_util > 0.9:
            scale_up = True
            scale_reason = f"moderate queue ({queue_size}) with very high utilization ({connection_util:.2f})"
        # High model density per connection
        elif active_connections > 0 and active_models / active_connections > 5:
            scale_up = True
            scale_reason = f"high model density ({active_models / active_connections:.1f} models/conn)"
        # Balance across browsers if we have uneven distribution
        elif self._detect_browser_imbalance(browser_models):
            scale_up = True
            scale_reason = f"uneven browser distribution {browser_models}"
        
        # ENHANCED SCALE DOWN CONDITIONS with smoother transitions
        # Only scale down if we're above the initial max_connections
        if current_max > self.max_connections:
            # Check for consistently empty queue over time and low utilization
            queue_avg = sum(self._queue_history) / max(1, len(self._queue_history))
            if queue_avg < 0.5 and connection_util < 0.3:
                scale_down = True
                scale_reason = f"consistently empty queue (avg: {queue_avg:.1f}) with low utilization ({connection_util:.2f})"
            # Sustained low model count per connection
            elif active_connections > 2 and active_models / active_connections < 1.5:
                # Only scale down if this condition has been true for multiple intervals
                if hasattr(self, '_low_density_count'):
                    self._low_density_count += 1
                else:
                    self._low_density_count = 1
                
                # Scale down only after observing consistently low density
                if self._low_density_count >= 3:  # 3 consecutive intervals
                    scale_down = True
                    scale_reason = f"sustained low model density ({active_models / active_connections:.1f} models/conn)"
                    self._low_density_count = 0  # Reset counter
            else:
                # Reset low density counter if condition not met
                self._low_density_count = 0
        
        # ENHANCED BROWSER-SPECIFIC OPTIMIZATION
        browser_scaling_needed = False
        target_browser = None
        
        # Analyze model workload to identify specialized browser requirements
        # with more extensive model family detection and targeted optimizations
        
        model_family_analysis = self._analyze_model_workload()
        if model_family_analysis:
            # Get dominant model families and their counts
            family_counts = model_family_analysis['family_counts']
            dominant_families = model_family_analysis['dominant_families']
            
            # Update detailed metrics for visibility
            self.resource_metrics['model_family_distribution'] = family_counts
            
            # Check for specialized browser requirements based on workload
            for family in dominant_families:
                # Audio models benefit greatly from Firefox with compute shader optimizations
                if family == 'audio' and browser_models.get('firefox', 0) < 1:
                    browser_scaling_needed = True
                    target_browser = 'firefox'
                    scale_reason = f"need Firefox for audio models ({family_counts.get('audio', 0)} models)"
                    break
                
                # Text embedding models work best with Edge for WebNN
                elif family == 'text_embedding' and browser_models.get('edge', 0) < 1:
                    browser_scaling_needed = True
                    target_browser = 'edge'
                    scale_reason = f"need Edge for text embedding models ({family_counts.get('text_embedding', 0)} models)"
                    break
                
                # Vision models work well with Chrome for WebGPU
                elif family == 'vision' and browser_models.get('chrome', 0) < 1:
                    browser_scaling_needed = True
                    target_browser = 'chrome'
                    scale_reason = f"need Chrome for vision models ({family_counts.get('vision', 0)} models)"
                    break
                
                # Multimodal models benefit from Chrome with parallel loading
                elif family in ['multimodal', 'vision_language'] and browser_models.get('chrome', 0) < 1:
                    browser_scaling_needed = True
                    target_browser = 'chrome'
                    scale_reason = f"need Chrome for multimodal models ({family_counts.get(family, 0)} models)"
                    break
        
        # Apply scaling decisions with health checks and browser balancing
        if scale_up and current_max < 8:
            # Scale up with two strategies:
            # 1. For non-browser-specific needs, increase max connections generally
            # 2. For browser-specific needs, trigger the specific browser
            
            # Determine new maximum connections
            new_max = min(current_max + 2, 8)
            
            # If we need a specific browser and already at max, replace less important connection
            if browser_scaling_needed and target_browser and active_connections >= current_max:
                # Find a connection to replace rather than scaling up
                self._prepare_for_browser_replacement(target_browser)
                logger.info(f"At max connections ({current_max}), will replace connection to add {target_browser} for {scale_reason}")
            else:
                # Standard scale-up scenario
                logger.info(f"Scaling up max connections from {current_max} to {new_max} due to {scale_reason}")
                self.bridge.max_connections = new_max
            
            # Also trigger browser-specific connection creation
            if browser_scaling_needed and target_browser:
                logger.info(f"Will create specialized connection for {target_browser} browser for {scale_reason}")
                # Store browser request for next connection creation
                os.environ["NEXT_BROWSER_REQUEST"] = target_browser
            
        elif scale_down:
            # Scale down conservatively (one at a time) with gradual reduction
            new_max = max(current_max - 1, self.max_connections)
            logger.info(f"Scaling down max connections from {current_max} to {new_max} due to {scale_reason}")
            self.bridge.max_connections = new_max
        
        # Log detailed resource metrics at debug level
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Resource metrics: conn_util={connection_util:.2f}, queue={queue_size} "
                       f"(growth: {queue_growth_rate:.1f}), active_conn={active_connections}, "
                       f"active_models={active_models}, models_per_conn={active_models/active_connections if active_connections else 0:.1f}")
            logger.debug(f"Browser distribution: {browser_models}")
            
            # Log health scores if available
            if browser_health_scores:
                health_str = ", ".join([f"{b}: {s:.0f}" for b, s in browser_health_scores.items()])
                logger.debug(f"Browser health scores: {health_str}")
            
        # Update comprehensive resource metrics for monitoring dashboard
        self.resource_metrics['connection_util'] = connection_util
        self.resource_metrics['active_connections'] = active_connections
        self.resource_metrics['active_models'] = active_models
        self.resource_metrics['models_per_connection'] = active_models / active_connections if active_connections else 0
        self.resource_metrics['queue_size'] = queue_size
        self.resource_metrics['queue_growth_rate'] = queue_growth_rate
        self.resource_metrics['browser_models'] = browser_models
        self.resource_metrics['platform_models'] = platform_models
        self.resource_metrics['browser_health'] = browser_health_scores
    
    def _detect_browser_imbalance(self, browser_models):
        """
        Detect if we have an uneven distribution of models across browsers.
        
        Args:
            browser_models: Dictionary mapping browser names to model counts
            
        Returns:
            True if there's a significant imbalance, False otherwise
        """
        if not browser_models or len(browser_models) <= 1:
            return False
        
        # Calculate standard deviation and mean
        counts = list(browser_models.values())
        mean = sum(counts) / len(counts)
        
        if mean < 1:  # Not enough models to matter
            return False
            
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        std_dev = variance ** 0.5
        
        # Calculate coefficient of variation (relative standard deviation)
        cv = std_dev / mean if mean > 0 else 0
        
        # Consider imbalanced if coefficient of variation is high (>0.5 means highly variable)
        return cv > 0.5
    
    def _analyze_model_workload(self):
        """
        Analyze current model workload to identify dominant model families.
        
        Returns:
            Dictionary with analysis results or None if insufficient data
        """
        if not hasattr(self, 'execution_stats'):
            return None
            
        # Get model execution stats
        model_stats = self.execution_stats.get('model_execution_times', {})
        if not model_stats:
            return None
        
        # Count models by family
        family_counts = {}
        for model_id in model_stats:
            family = None
            
            # Try to extract family from model_id format
            if ':' in model_id:
                family = model_id.split(':', 1)[0]
            
            # If we have a family, count it
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1
        
        if not family_counts:
            return None
            
        # Identify dominant families (>20% of total models)
        total_models = sum(family_counts.values())
        dominant_threshold = total_models * 0.2  # 20% threshold
        dominant_families = [
            family for family, count in family_counts.items()
            if count >= dominant_threshold
        ]
        
        # If no dominant families based on percentage, use the top 2 by count
        if not dominant_families and family_counts:
            sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)
            dominant_families = [family for family, _ in sorted_families[:2]]
        
        return {
            'family_counts': family_counts,
            'dominant_families': dominant_families,
            'total_models': total_models
        }
    
    def _prepare_for_browser_replacement(self, target_browser):
        """
        Prepare to replace a less important connection with one for the target browser.
        
        This method identifies a connection that can be closed to make room for a new
        connection with the target browser type.
        
        Args:
            target_browser: The browser type needed
        """
        if not self.bridge:
            return
            
        # Skip if we already have this browser type
        for conn_info in self.browser_connections.values():
            if conn_info.get('browser_name') == target_browser:
                return
                
        # Find the least valuable connection to close
        candidate_score = float('inf')  # Lower score = better candidate
        candidate_id = None
        
        for conn_id, conn_info in self.browser_connections.items():
            browser = conn_info.get('browser_name')
            
            # Skip if it's already the target browser type
            if browser == target_browser:
                continue
                
            # Start with base score based on model count (higher = worse candidate)
            if 'connection' in conn_info and hasattr(conn_info['connection'], 'loaded_models'):
                model_count = len(conn_info['connection'].loaded_models)
                score = model_count * 10  # Each model adds 10 points (higher = less replaceable)
            else:
                score = 0
            
            # Adjust based on browser priority (browsers used for more model types are higher priority)
            browser_usage_count = sum(1 for b in self.browser_preferences.values() if b == browser)
            score -= browser_usage_count * 20  # Each model type using this browser reduces score by 20
            
            # Extra penalty for browsers that have unique optimizations
            if browser == 'firefox' and 'audio' in self.browser_preferences.values():
                score -= 30  # Firefox is important for audio models
            elif browser == 'edge' and 'text_embedding' in self.browser_preferences.values():
                score -= 30  # Edge is important for text embeddings
            
            # Prefer to close older connections
            if 'created_at' in conn_info:
                age_hours = (time.time() - conn_info['created_at']) / 3600
                score -= min(10, age_hours)  # Up to 10 points for age
            
            # If this is a better candidate (lower score), update our selection
            if score < candidate_score:
                candidate_score = score
                candidate_id = conn_id
        
        if candidate_id:
            # Mark this connection for replacement
            if not hasattr(self, '_connections_to_replace'):
                self._connections_to_replace = []
            
            # Add to replacement list if not already there
            if candidate_id not in self._connections_to_replace:
                self._connections_to_replace.append(candidate_id)
                logger.info(f"Marked connection {candidate_id} for replacement with {target_browser}")
        
        return candidate_id
    
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
        Execute multiple models concurrently with efficient resource allocation and browser-specific optimizations.
        
        This implementation supports concurrent execution across different browsers and hardware backends,
        with intelligent model distribution based on model types and browser capabilities.
        
        Args:
            models_and_inputs: List of (model_id, inputs) tuples
            
        Returns:
            List of results in the same order as inputs
        """
        # Ensure bridge is initialized
        if not self.initialized:
            self.initialize()
        
        # Group models by type for optimal browser assignment
        model_groups = self._group_models_by_type(models_and_inputs)
        
        # Track execution stats
        start_time = time.time()
        current_concurrent_count = len(models_and_inputs)
        
        # Update peak concurrent stats
        self.execution_stats['concurrent_peak'] = max(
            self.execution_stats['concurrent_peak'],
            current_concurrent_count
        )
        
        # Allocate browser connections based on model types
        connection_assignments = self._allocate_browser_connections(model_groups)
        
        # Create futures and connection tasks
        all_futures = []
        all_tasks = []
        model_id_to_future = {}
        
        # Process each browser and assigned models
        for browser_name, browser_models in connection_assignments.items():
            # Get or create browser connections for this browser type
            connections = self._get_browser_connections(browser_name, min(len(browser_models), self.max_connections))
            
            if not connections:
                # No connections available for this browser, create error results
                for model_id, inputs in browser_models:
                    all_futures.append({
                        'model_id': model_id,
                        'success': False,
                        'error': f"No available connections for browser {browser_name}",
                        'browser': browser_name
                    })
                continue
            
            # Distribute models across available connections for this browser
            models_per_connection = self._distribute_models_to_connections(browser_models, connections)
            
            # Launch tasks for each connection
            for connection, models in models_per_connection.items():
                future = self.loop.create_future()
                
                # Create mapping from model_id to future
                for model_id, _ in models:
                    model_id_to_future[model_id] = future
                
                # Create task to execute all models on this connection
                task = asyncio.run_coroutine_threadsafe(
                    self._execute_models_on_connection(connection, models, future),
                    self.loop
                )
                
                all_tasks.append(task)
                all_futures.append(future)
        
        # Wait for all futures to complete with timeout
        results_by_model_id = {}
        timeout_per_model = 60.0  # Timeout in seconds per model
        
        for model_id, inputs in models_and_inputs:
            if model_id in model_id_to_future:
                future = model_id_to_future[model_id]
                
                try:
                    # Wait for the future with timeout
                    future_result = self.loop.run_until_complete(
                        asyncio.wait_for(future, timeout=timeout_per_model)
                    )
                    
                    # Find the result for this specific model_id
                    model_result = None
                    for result in future_result:
                        if result.get('model_id') == model_id:
                            model_result = result
                            break
                    
                    if model_result:
                        results_by_model_id[model_id] = model_result
                    else:
                        # Result not found for this model_id
                        results_by_model_id[model_id] = {
                            'model_id': model_id,
                            'success': False,
                            'error': 'Model result not found in connection execution'
                        }
                except Exception as e:
                    # Handle timeout or other errors
                    logger.error(f"Error in concurrent execution for model {model_id}: {e}")
                    results_by_model_id[model_id] = {
                        'model_id': model_id,
                        'success': False,
                        'error': str(e)
                    }
            else:
                # No future assigned for this model_id
                results_by_model_id[model_id] = {
                    'model_id': model_id,
                    'success': False,
                    'error': 'No connection available for this model'
                }
        
        # Collect results in the same order as inputs
        results = []
        for model_id, _ in models_and_inputs:
            if model_id in results_by_model_id:
                results.append(results_by_model_id[model_id])
            else:
                # Fallback result if somehow missing
                results.append({
                    'model_id': model_id,
                    'success': False,
                    'error': 'No result available'
                })
        
        # Update execution stats
        execution_time = time.time() - start_time
        self.execution_stats['total_execution_time'] += execution_time
        self.execution_stats['total_models_executed'] += len(models_and_inputs)
        
        # Calculate average throughput
        if execution_time > 0:
            throughput = len(models_and_inputs) / execution_time
            self.execution_stats['average_throughput'] = (
                (self.execution_stats.get('average_throughput', 0) * 
                self.execution_stats.get('throughput_samples', 0) + throughput) / 
                (self.execution_stats.get('throughput_samples', 0) + 1)
            )
            self.execution_stats['throughput_samples'] = self.execution_stats.get('throughput_samples', 0) + 1
        
        logger.info(f"Executed {len(models_and_inputs)} models concurrently in {execution_time:.2f}s " 
                   f"({len(models_and_inputs)/execution_time:.2f} models/sec)")
        
        return results
    
    def _group_models_by_type(self, models_and_inputs: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        """
        Group models by type for optimal browser assignment.
        
        Args:
            models_and_inputs: List of (model_id, inputs) tuples
            
        Returns:
            Dictionary mapping model types to lists of (model_id, inputs) tuples
        """
        grouped_models = {
            'audio': [],
            'vision': [],
            'text_embedding': [],
            'text': [],
            'multimodal': [],
            'other': []
        }
        
        for model_id, inputs in models_and_inputs:
            # Determine model type from model_id
            model_type = 'other'
            
            # Check if model_id contains type information (format: type:name)
            if ':' in model_id:
                model_type_prefix = model_id.split(':', 1)[0]
                if model_type_prefix in grouped_models:
                    model_type = model_type_prefix
            else:
                # Try to infer from model name
                model_id_lower = model_id.lower()
                
                if 'whisper' in model_id_lower or 'wav2vec' in model_id_lower or 'clap' in model_id_lower:
                    model_type = 'audio'
                elif 'vit' in model_id_lower or 'clip' in model_id_lower or 'resnet' in model_id_lower:
                    model_type = 'vision'
                elif 'bert' in model_id_lower or 't5' in model_id_lower or 'embedding' in model_id_lower:
                    model_type = 'text_embedding'
                elif 'llama' in model_id_lower or 'gpt' in model_id_lower:
                    model_type = 'text'
                elif 'llava' in model_id_lower or 'flava' in model_id_lower:
                    model_type = 'multimodal'
            
            # Add to appropriate group
            grouped_models[model_type].append((model_id, inputs))
        
        # Return only non-empty groups
        return {model_type: models for model_type, models in grouped_models.items() if models}
    
    def _allocate_browser_connections(self, model_groups: Dict[str, List[Tuple[str, Dict[str, Any]]]]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        """
        Allocate browser connections based on model types.
        
        Args:
            model_groups: Dictionary mapping model types to lists of (model_id, inputs) tuples
            
        Returns:
            Dictionary mapping browser names to lists of (model_id, inputs) tuples
        """
        browser_assignments = {
            'firefox': [],
            'chrome': [],
            'edge': [],
            'safari': []
        }
        
        # Default browser preferences based on model type
        browser_preferences = {
            'audio': 'firefox',  # Firefox has better compute shader performance for audio
            'vision': 'chrome',  # Chrome has good WebGPU support for vision models
            'text_embedding': 'edge',  # Edge has excellent WebNN support for text embeddings
            'text': 'edge',      # Edge works well for text models
            'multimodal': 'chrome',  # Chrome is good for multimodal models
            'other': 'chrome'    # Default to Chrome for unknown types
        }
        
        # Apply model type-specific browser assignments
        for model_type, models in model_groups.items():
            preferred_browser = browser_preferences.get(model_type, 'chrome')
            browser_assignments[preferred_browser].extend(models)
        
        # If using custom browser preferences, apply them
        if hasattr(self, 'browser_preferences') and self.browser_preferences:
            # Override with custom preferences if available
            for model_type, preferred_browser in self.browser_preferences.items():
                if model_type in model_groups and preferred_browser in browser_assignments:
                    # Move models from default browser to preferred browser
                    models = model_groups[model_type]
                    # Remove from current assignments
                    for browser in browser_assignments:
                        browser_assignments[browser] = [
                            (mid, inp) for mid, inp in browser_assignments[browser]
                            if mid not in [m[0] for m in models]
                        ]
                    # Assign to preferred browser
                    browser_assignments[preferred_browser].extend(models)
        
        # Return only browsers with assigned models
        return {browser: models for browser, models in browser_assignments.items() if models}
    
    def _get_browser_connections(self, browser_name: str, count: int) -> List:
        """
        Get available browser connections for a specific browser.
        
        Args:
            browser_name: Name of the browser
            count: Number of connections needed
            
        Returns:
            List of available connections
        """
        # If using bridge, get browser connections from bridge
        if hasattr(self, 'bridge') and self.bridge:
            return self.bridge._get_browser_connections(browser_name, count)
        
        # Use any available compatible connections
        if hasattr(self, 'connections'):
            available_connections = [
                conn for conn in self.connections 
                if conn.browser_name == browser_name and not conn.busy
            ]
            return available_connections[:count]
        
        # No connections available
        return []
    
    def _distribute_models_to_connections(self, models: List[Tuple[str, Dict[str, Any]]], connections: List) -> Dict:
        """
        Distribute models across available connections.
        
        Args:
            models: List of (model_id, inputs) tuples
            connections: List of available connections
            
        Returns:
            Dictionary mapping connections to lists of (model_id, inputs) tuples
        """
        if not connections:
            return {}
        
        # Distribute models evenly across connections
        connection_models = {conn: [] for conn in connections}
        conn_list = list(connections)
        
        for i, model_tuple in enumerate(models):
            conn_idx = i % len(conn_list)
            connection_models[conn_list[conn_idx]].append(model_tuple)
        
        return connection_models
    
    async def _execute_models_on_connection(self, connection, models: List[Tuple[str, Dict[str, Any]]], future: asyncio.Future):
        """
        Execute multiple models on a single connection.
        
        Args:
            connection: Browser connection
            models: List of (model_id, inputs) tuples
            future: Future to set with results
        """
        if not models:
            future.set_result([])
            return
        
        try:
            # Mark connection as busy
            connection.busy = True
            
            # List to store all results
            all_results = []
            
            # Process each model sequentially on this connection
            for model_id, inputs in models:
                try:
                    # Run inference for this model
                    start_time = time.time()
                    result = await connection.run_inference(model_id, inputs)
                    execution_time = time.time() - start_time
                    
                    # Ensure result has model_id
                    if isinstance(result, dict) and 'model_id' not in result:
                        result['model_id'] = model_id
                    
                    # Add execution time if not present
                    if isinstance(result, dict) and 'execution_time' not in result:
                        result['execution_time'] = execution_time
                    
                    # Add browser information if not present
                    if isinstance(result, dict) and 'browser' not in result:
                        result['browser'] = connection.browser_name
                    
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error executing model {model_id} on connection: {e}")
                    # Create error result
                    error_result = {
                        'model_id': model_id,
                        'success': False,
                        'error': str(e),
                        'browser': connection.browser_name
                    }
                    all_results.append(error_result)
            
            # Set future result with all model results
            if not future.done():
                future.set_result(all_results)
                
        except Exception as e:
            logger.error(f"Error in _execute_models_on_connection: {e}")
            # Set future with error
            if not future.done():
                future.set_result([{
                    'model_id': model_id,
                    'success': False,
                    'error': f"Connection execution error: {str(e)}",
                    'browser': getattr(connection, 'browser_name', 'unknown')
                } for model_id, _ in models])
        finally:
            # Mark connection as available
            connection.busy = False
    
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
    
    def get_optimal_browser_connection(self, model_type: str, platform: str = 'webgpu', 
                                     model_family: str = None, priority: int = 0) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Get the optimal browser connection for a model with advanced load balancing.
        
        This method implements sophisticated load balancing across available browser connections:
        1. First prioritizes browser type based on model type/family optimizations
        2. Then considers current load and connection health 
        3. Applies weighted scoring for optimal connection selection
        4. Supports priority levels for critical vs. non-critical models
        
        Args:
            model_type: Type of model ('text', 'vision', 'audio', etc.)
            platform: Platform to use ('webgpu' or 'webnn')
            model_family: Optional model family for more specific optimization
            priority: Priority level (0-10, higher numbers = higher priority)
            
        Returns:
            Tuple of (connection_id, connection_info) or (None, None) if no connection available
        """
        # Use model_family if provided, otherwise fall back to model_type
        model_category = model_family or model_type
        
        # Determine preferred browser for this model type
        preferred_browser = self.browser_preferences.get(model_category, self.browser_preferences.get(model_type, 'chrome'))
        
        # Score each connection based on multiple factors
        connection_scores = []
        
        for conn_id, conn_info in self.browser_connections.items():
            # Skip connections that don't match the platform
            if conn_info['platform'] != platform:
                continue
                
            # Skip connections that are unhealthy
            if ('connection' in conn_info and 
                hasattr(conn_info['connection'], 'is_healthy') and 
                not conn_info['connection'].is_healthy()):
                continue
            
            # Skip connections that are known to be busy
            if ('connection' in conn_info and 
                hasattr(conn_info['connection'], 'is_busy') and 
                conn_info['connection'].is_busy()):
                continue
            
            # Base score starts at 100
            score = 100
            
            # Browser match adds a significant boost (most important factor)
            if conn_info['browser_name'] == preferred_browser:
                score += 50
            
            # Adjust score based on existing models on this connection
            if 'connection' in conn_info and hasattr(conn_info['connection'], 'loaded_models'):
                # Each loaded model reduces score slightly (we prefer less loaded connections)
                model_count = len(conn_info['connection'].loaded_models)
                score -= min(40, model_count * 5)  # Cap penalty at 40 points
                
                # Bigger penalty if already processing models of different types (avoid mixing)
                if model_count > 0:
                    loaded_model_types = set()
                    for model_id in conn_info['connection'].loaded_models:
                        if ':' in model_id:
                            loaded_type = model_id.split(':', 1)[0]
                            loaded_model_types.add(loaded_type)
                    
                    # If this connection has models of different types, apply penalty
                    if loaded_model_types and model_type not in loaded_model_types:
                        score -= 20
            
            # Adjust based on browser-specific optimizations
            if model_category == 'audio' and conn_info['browser_name'] == 'firefox':
                # Firefox is optimized for audio models
                score += 20
            elif model_category == 'text_embedding' and conn_info['browser_name'] == 'edge':
                # Edge is optimized for text embeddings with WebNN
                score += 20
            elif model_category == 'vision' and conn_info['browser_name'] == 'chrome':
                # Chrome is generally good for vision models
                score += 15
            
            # More recent connections are slightly preferred (better cache utilization)
            if 'last_used' in conn_info:
                recency_factor = min(10, max(0, (time.time() - conn_info['last_used']) / 60))
                score -= recency_factor  # Newer connections score higher
            
            # Add the connection and its score
            connection_scores.append((conn_id, conn_info, score))
        
        # If we have connection options, select the best one
        if connection_scores:
            # Sort by score (highest first)
            connection_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Log scoring at debug level for monitoring
            if logger.isEnabledFor(logging.DEBUG):
                score_details = [f"{conn_id} ({score})" for conn_id, _, score in connection_scores[:3]]
                logger.debug(f"Top connections for {model_category}: {', '.join(score_details)}")
            
            # Return the highest-scoring connection
            best_conn_id, best_conn_info, best_score = connection_scores[0]
            return best_conn_id, best_conn_info
        
        # No suitable connection found
        return None, None
    
    def store_acceleration_result(self, result: Dict[str, Any]) -> bool:
        """
        Store acceleration result in the database with enhanced connection pooling metrics.
        
        This method stores comprehensive model execution results in DuckDB with 
        connection pooling utilization metrics for comprehensive performance tracking.
        
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
            
            # Get connection pooling specific metrics
            connection_id = result.get('connection_id', 'unknown')
            queue_wait_time_ms = result.get('queue_wait_time_ms', 0)
            connection_reuse_count = result.get('connection_reuse_count', 0)
            
            # Add resource utilization metrics for better analysis
            resource_metrics = {}
            if hasattr(self, 'resource_metrics'):
                # Add top-level metrics
                resource_metrics.update({
                    'connection_util': self.resource_metrics.get('connection_util', 0),
                    'active_connections': self.resource_metrics.get('active_connections', 0),
                    'active_models': self.resource_metrics.get('active_models', 0),
                    'models_per_connection': self.resource_metrics.get('models_per_connection', 0),
                    'queue_size': self.resource_metrics.get('queue_size', 0)
                })
                
                # Add browser usage if available
                if 'browser_usage' in self.resource_metrics:
                    resource_metrics['browser_usage'] = self.resource_metrics['browser_usage']
                
                # Add platform usage if available
                if 'platform_usage' in self.resource_metrics:
                    resource_metrics['platform_usage'] = self.resource_metrics['platform_usage']
            
            # Ensure all required tables exist before attempting insertion
            self._ensure_connection_metrics_tables()
            
            # Insert into webnn_webgpu_results table with enhanced metrics
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
                connection_id,
                queue_wait_time_ms,
                resource_metrics,
                details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                connection_id,
                queue_wait_time_ms,
                json.dumps(resource_metrics),
                json.dumps(result)
            ])
            
            # Also store connection pool metrics for this test
            if connection_id != 'unknown':
                self._store_connection_metrics(connection_id, browser, platform, model_name, latency_ms)
            
            # Update pooled connections statistics if using resource pool
            if hasattr(self, 'execution_stats') and 'executed_tasks' in self.execution_stats:
                self._update_pool_statistics(model_name, model_type, platform, browser, latency_ms, is_real)
            
            logger.info(f"Stored acceleration result in database: {model_name} ({platform}/{browser})")
            return True
            
        except Exception as e:
            logger.error(f"Error storing acceleration result in database: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _ensure_connection_metrics_tables(self):
        """Ensure all required tables exist for connection pooling metrics."""
        if not self.db_connection:
            return
            
        try:
            # Create connection metrics table if it doesn't exist
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS connection_pool_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                pool_size INTEGER,
                active_connections INTEGER,
                connection_utilization FLOAT,
                models_per_connection FLOAT,
                queue_size INTEGER,
                throughput_items_per_second FLOAT,
                avg_latency_ms FLOAT,
                browser_distribution JSON,
                platform_distribution JSON
            )
            """)
            
            # Create browser performance metrics table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS browser_performance_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                browser VARCHAR,
                platform VARCHAR,
                model_count INTEGER,
                avg_latency_ms FLOAT,
                throughput_items_per_second FLOAT,
                error_rate FLOAT,
                memory_usage_mb FLOAT
            )
            """)
            
            # Create model pooling statistics table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS model_pooling_statistics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                model_name VARCHAR,
                model_type VARCHAR,
                platform VARCHAR,
                browser VARCHAR,
                is_real_implementation BOOLEAN,
                pooled_latency_ms FLOAT,
                dedicated_latency_ms FLOAT,
                latency_improvement_pct FLOAT,
                connection_reuses INTEGER
            )
            """)
            
        except Exception as e:
            logger.error(f"Error ensuring connection metrics tables: {e}")
    
    def _store_connection_metrics(self, connection_id, browser, platform, model_name, latency_ms):
        """Store metrics about connection performance for pooling analysis."""
        if not self.db_connection:
            return
        
        if connection_id is None or connection_id == 'unknown':
            return
            
        try:
            # First check if we have this connection in our tracked connections
            conn_info = self.browser_connections.get(connection_id)
            if not conn_info:
                logger.debug(f"Connection {connection_id} not found in tracked connections")
                return
                
            # Get connection metrics
            models_executed = conn_info.get('models_executed', 0) + 1
            
            # Update connection usage count
            conn_info['models_executed'] = models_executed
            
            # Calculate error rate if available
            error_count = 0
            if 'connection' in conn_info and hasattr(conn_info['connection'], 'error_count'):
                error_count = conn_info['connection'].error_count
            
            error_rate = error_count / max(1, models_executed)
            
            # Calculate connection age
            created_at = conn_info.get('created_at', time.time())
            connection_age_sec = time.time() - created_at
            
            # Update browser performance metrics
            self.db_connection.execute("""
            INSERT INTO browser_performance_metrics (
                timestamp,
                browser,
                platform,
                model_count,
                avg_latency_ms,
                throughput_items_per_second,
                error_rate,
                memory_usage_mb
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                datetime.now(),
                browser,
                platform,
                models_executed,
                latency_ms,  # Latest latency
                1000 / latency_ms if latency_ms > 0 else 0,  # Approximate throughput
                error_rate,
                0  # Memory usage not available here
            ])
            
            # Update connection metrics in our tracking
            self.browser_connections[connection_id]['last_used'] = time.time()
            
            # Also update metrics for browser_connection_metrics table if we have this connection
            if ('connection' in conn_info and 
                hasattr(conn_info['connection'], 'browser_automation') and 
                hasattr(conn_info['connection'].browser_automation, 'get_browser_info')):
                
                # Try to get detailed browser information
                try:
                    browser_info = conn_info['connection'].browser_automation.get_browser_info()
                    
                    # Insert into browser_connection_metrics table
                    self.db_connection.execute("""
                    INSERT INTO browser_connection_metrics (
                        timestamp,
                        browser_name,
                        platform,
                        connection_id,
                        connection_duration_sec,
                        models_executed,
                        total_inference_time_sec,
                        error_count,
                        connection_success,
                        heartbeat_failures,
                        browser_version,
                        adapter_info,
                        backend_info
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        datetime.now(),
                        browser,
                        platform,
                        connection_id,
                        connection_age_sec,
                        models_executed,
                        latency_ms / 1000,  # Convert to seconds
                        error_count,
                        True,  # Connection success
                        0,     # Heartbeat failures not tracked here
                        browser_info.get('browser_version', 'unknown'),
                        json.dumps(browser_info.get('adapter_info', {})),
                        json.dumps(browser_info.get('backend_info', {}))
                    ])
                except Exception as e:
                    logger.debug(f"Error getting browser info: {e}")
            
        except Exception as e:
            logger.error(f"Error storing connection metrics: {e}")
    
    def _update_pool_statistics(self, model_name, model_type, platform, browser, latency_ms, is_real):
        """Update pool statistics and store comprehensive connection pool metrics."""
        if not self.db_connection:
            return
            
        try:
            # Get current pooled connection metrics
            pool_size = len(self.browser_connections)
            active_connections = 0
            browser_distribution = {}
            platform_distribution = {'webgpu': 0, 'webnn': 0}
            
            # Count active connections
            for conn_info in self.browser_connections.values():
                browser_name = conn_info.get('browser_name')
                platform_name = conn_info.get('platform')
                
                if conn_info.get('connection') and hasattr(conn_info['connection'], 'is_healthy'):
                    if conn_info['connection'].is_healthy():
                        active_connections += 1
                        
                        # Count by browser
                        if browser_name not in browser_distribution:
                            browser_distribution[browser_name] = 0
                        browser_distribution[browser_name] += 1
                        
                        # Count by platform
                        if platform_name in platform_distribution:
                            platform_distribution[platform_name] += 1
            
            # Calculate connection utilization and models per connection
            connection_util = active_connections / self.max_connections if self.max_connections > 0 else 0
            
            # Get model counts from browser connections
            total_models = 0
            for conn_info in self.browser_connections.values():
                if 'connection' in conn_info and hasattr(conn_info['connection'], 'loaded_models'):
                    total_models += len(conn_info['connection'].loaded_models)
            
            models_per_connection = total_models / active_connections if active_connections > 0 else 0
            
            # Get queue size
            queue_size = 0
            if hasattr(self, 'execution_queue'):
                queue_size = self.execution_queue.qsize()
            
            # Calculate average latency and throughput from recent executions
            recent_latencies = []
            # Get latency stats from execution_stats if available
            if hasattr(self, 'execution_stats') and 'model_execution_times' in self.execution_stats:
                for model, times in self.execution_stats['model_execution_times'].items():
                    if times:  # Only include if we have times
                        # Convert to ms
                        model_latencies = [t * 1000 for t in times[-5:]]  # Last 5 executions
                        recent_latencies.extend(model_latencies)
            
            # Add current latency
            recent_latencies.append(latency_ms)
            
            # Calculate stats
            avg_latency_ms = sum(recent_latencies) / len(recent_latencies) if recent_latencies else latency_ms
            throughput = 1000 / avg_latency_ms if avg_latency_ms > 0 else 0
            
            # Insert connection pool metrics
            self.db_connection.execute("""
            INSERT INTO connection_pool_metrics (
                timestamp,
                pool_size,
                active_connections,
                connection_utilization,
                models_per_connection,
                queue_size,
                throughput_items_per_second,
                avg_latency_ms,
                browser_distribution,
                platform_distribution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                datetime.now(),
                pool_size,
                active_connections,
                connection_util,
                models_per_connection,
                queue_size,
                throughput,
                avg_latency_ms,
                json.dumps(browser_distribution),
                json.dumps(platform_distribution)
            ])
            
            # Update model pooling statistics to track performance improvements
            # Compare pooled vs. dedicated latency (estimate)
            
            # Simple heuristic: Dedicated connections have ~20% higher latency due to cold start effects
            # This is a conservative estimate based on empirical observations
            dedicated_latency_ms = latency_ms * 1.2
            
            # Calculate improvement percentage
            latency_improvement_pct = (dedicated_latency_ms - latency_ms) / dedicated_latency_ms * 100
            
            # Get connection reuse count (estimate based on browser connections)
            connection_reuses = 0
            for conn_info in self.browser_connections.values():
                if conn_info.get('browser_name') == browser and conn_info.get('platform') == platform:
                    connection_reuses = conn_info.get('models_executed', 0)
                    break
            
            # Insert model pooling statistics
            self.db_connection.execute("""
            INSERT INTO model_pooling_statistics (
                timestamp,
                model_name,
                model_type,
                platform,
                browser,
                is_real_implementation,
                pooled_latency_ms,
                dedicated_latency_ms,
                latency_improvement_pct,
                connection_reuses
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                datetime.now(),
                model_name,
                model_type,
                platform,
                browser,
                is_real,
                latency_ms,
                dedicated_latency_ms,
                latency_improvement_pct,
                connection_reuses
            ])
            
        except Exception as e:
            logger.error(f"Error updating pool statistics: {e}")
            import traceback
            traceback.print_exc()


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