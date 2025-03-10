#!/usr/bin/env python3
"""
Resource Pool Bridge for WebNN/WebGPU acceleration.

This module provides a bridge between the resource pool and WebNN/WebGPU backends,
allowing for efficient allocation and utilization of browser-based acceleration resources.
"""

import os
import sys
import json
import time
import random
import logging
import asyncio
import platform
import traceback

# Check for psutil availability
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ResourcePool')

class MockFallbackModel:
    """Mock model class used as a fallback when all else fails."""
    
    def __init__(self, model_name, model_type, hardware_type="cpu"):
        self.model_name = model_name
        self.model_type = model_type
        self.hardware_type = hardware_type
        
    def __call__(self, inputs):
        """Simulate model inference."""
        return {
            "success": True,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "hardware": self.hardware_type,
            "platform": self.hardware_type,
            "is_real_hardware": False,
            "is_simulation": True,
            "processing_time": 0.1,
            "inference_time": 0.1,
            "total_time": 0.2,
            "latency_ms": 100,
            "throughput_items_per_sec": 10,
            "memory_usage_mb": 100,
            "result": "Mock fallback model result"
        }

class EnhancedWebModel:
    """
    Enhanced web model with browser-specific optimizations.
    
    This enhanced model implementation includes:
    - Browser-specific optimizations for different model types
    - Hardware platform selection based on model requirements
    - Simulation capabilities for testing and development
    - Performance tracking and telemetry
    - Tensor sharing for multi-model efficiency
    """
    
    def __init__(self, model_name, model_type, hardware_type, browser=None, **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.hardware_type = hardware_type
        self.browser = browser or 'chrome'  # Default to Chrome if not specified
        self.inference_count = 0
        self.total_inference_time = 0
        self.avg_inference_time = 0
        
        # Set optimization flags - will be populated from kwargs
        # Note: We convert compute_shaders to compute_shader_optimized
        self.compute_shader_optimized = kwargs.get('compute_shaders', False)
        self.precompile_shaders = kwargs.get('precompile_shaders', False)
        self.parallel_loading = kwargs.get('parallel_loading', False)
        self.mixed_precision = kwargs.get('mixed_precision', False)
        self.precision = kwargs.get('precision', 16)
        
        # Get shared tensors if available
        self.shared_tensors = kwargs.get('shared_tensors', {})
        self.uses_shared_tensors = len(self.shared_tensors) > 0
        
        # Debug init
        logger.debug(f"EnhancedWebModel initialized with optimization flags: compute_shader_optimized={self.compute_shader_optimized}, precompile_shaders={self.precompile_shaders}, parallel_loading={self.parallel_loading}")
        if self.uses_shared_tensors:
            logger.debug(f"Model using shared tensors: {list(self.shared_tensors.keys())}")
        
    def __call__(self, inputs):
        """
        Simulate model inference with browser-specific optimizations.
        
        This implementation provides detailed metrics and simulates:
        - Browser-specific performance characteristics
        - Hardware platform efficiency
        - Model type optimization effects
        - Tensor sharing acceleration
        """
        # Track inference count
        self.inference_count += 1
        
        # Log optimization flags
        optimization_status = {
            'compute_shader_optimized': self.compute_shader_optimized,
            'precompile_shaders': self.precompile_shaders,
            'parallel_loading': self.parallel_loading,
            'mixed_precision': self.mixed_precision,
            'precision': self.precision,
            'using_shared_tensors': self.uses_shared_tensors
        }
        logger.debug(f"Model {self.model_name} optimization flags: {optimization_status}")
        
        # Determine inference time based on model and browser characteristics
        base_time = 0.1  # Base inference time
        
        # Apply speedup if using shared tensors
        # This simulates the performance improvement from tensor sharing
        shared_tensor_speedup = 1.0
        if self.uses_shared_tensors:
            # Using shared tensors provides significant speedup
            # Different components provide different levels of speedup
            for tensor_type in self.shared_tensors.keys():
                if 'embedding' in tensor_type:
                    shared_tensor_speedup *= 0.7  # 30% faster with shared embeddings
                elif 'attention' in tensor_type:
                    shared_tensor_speedup *= 0.8  # 20% faster with shared attention
            logger.debug(f"Using shared tensors: speedup factor {shared_tensor_speedup}")
        
        # Adjust for model type
        if 'audio' in self.model_type.lower():
            if self.browser == 'firefox':
                # Firefox is optimized for audio models
                model_factor = 0.8
            else:
                model_factor = 1.2
        elif 'vision' in self.model_type.lower():
            if self.browser == 'chrome':
                # Chrome is optimized for vision models
                model_factor = 0.85
            else:
                model_factor = 1.1
        elif 'text_embedding' in self.model_type.lower() or 'bert' in self.model_type.lower():
            if self.browser == 'edge':
                # Edge is optimized for text embedding models
                model_factor = 0.9
            else:
                model_factor = 1.0
        else:
            model_factor = 1.0
        
        # Adjust for hardware platform
        if self.hardware_type == 'webgpu':
            hardware_factor = 0.7  # WebGPU is faster
        elif self.hardware_type == 'webnn':
            hardware_factor = 0.8  # WebNN is faster
        else:
            hardware_factor = 1.2  # CPU is slower
        
        # Calculate simulated inference time with shared tensor speedup
        inference_time = base_time * model_factor * hardware_factor * shared_tensor_speedup
        
        # Update tracking metrics
        self.total_inference_time += inference_time
        self.avg_inference_time = self.total_inference_time / self.inference_count
        
        # Calculate memory usage based on precision and shared tensors
        base_memory = 100  # Base memory usage in MB
        memory_for_precision = {
            2: 0.25,  # 2-bit uses 25% of base memory
            3: 0.30,  # 3-bit uses 30% of base memory
            4: 0.4,   # 4-bit uses 40% of base memory
            8: 0.6,   # 8-bit uses 60% of base memory
            16: 1.0   # 16-bit uses 100% of base memory
        }
        precision_factor = memory_for_precision.get(self.precision, 1.0)
        
        # Calculate memory savings from shared tensors
        memory_saving_factor = 1.0
        if self.uses_shared_tensors:
            # Shared tensors save memory
            memory_saving_factor = 0.85  # 15% memory savings
        
        memory_usage = base_memory * precision_factor * memory_saving_factor
        
        # Prepare output tensors that could be shared with other models
        output_tensors = {}
        if 'text_embedding' in self.model_type.lower() or 'bert' in self.model_type.lower():
            # For text models, we could share embeddings
            output_tensors["text_embedding"] = f"Simulated text embedding tensor for {self.model_name}"
        elif 'vision' in self.model_type.lower():
            # For vision models, we could share image features
            output_tensors["vision_embedding"] = f"Simulated vision embedding tensor for {self.model_name}"
        
        # Return comprehensive result with optimization flags and shared tensor info
        result = {
            "success": True,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "hardware": self.hardware_type,
            "platform": self.hardware_type,
            "browser": self.browser,
            "is_real_hardware": False,
            "is_simulation": True,
            "processing_time": inference_time * 0.8,
            "inference_time": inference_time,
            "total_time": inference_time * 1.2,
            "latency_ms": inference_time * 1000,  # Convert to ms
            "throughput_items_per_sec": 1.0 / inference_time,
            "memory_usage_mb": memory_usage,
            "compute_shader_optimized": self.compute_shader_optimized,
            "precompile_shaders": self.precompile_shaders,
            "parallel_loading": self.parallel_loading,
            "mixed_precision": self.mixed_precision,
            "precision": self.precision,
            "output_tensors": output_tensors,
            "result": "Enhanced web model result"
        }
        
        # Add shared tensor info if used
        if self.uses_shared_tensors:
            result["shared_tensors_used"] = list(self.shared_tensors.keys())
            result["shared_tensor_speedup"] = (1.0 / shared_tensor_speedup - 1.0) * 100.0  # Convert to percentage
        
        return result

class ResourcePoolBridgeIntegration:
    """Bridge integration between resource pool and WebNN/WebGPU backends."""
    
    def __init__(self, max_connections=4, enable_gpu=True, enable_cpu=True,
                 headless=True, browser_preferences=None, adaptive_scaling=True,
                 monitoring_interval=60, enable_ipfs=True, db_path=None):
        """Initialize the resource pool bridge integration."""
        self.max_connections = max_connections
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu
        self.headless = headless
        self.browser_preferences = browser_preferences or {}
        self.adaptive_scaling = adaptive_scaling
        self.monitoring_interval = monitoring_interval
        self.enable_ipfs = enable_ipfs
        self.db_path = db_path
        
        # Initialize logger
        logger.info(f"ResourcePoolBridgeIntegration created with max_connections={max_connections}, adaptive_scaling={'enabled' if adaptive_scaling else 'disabled'}, IPFS={'enabled' if enable_ipfs else 'disabled'}")
    
    async def initialize(self):
        """
        Initialize the resource pool bridge with real browser integration.
        
        This enhanced implementation:
        1. Sets up real browser connections using Selenium
        2. Establishes WebSocket communication channels
        3. Configures browser-specific optimizations
        4. Manages connection pool with both real and simulated resources
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Try importing WebSocket bridge and browser automation
            from fixed_web_platform.websocket_bridge import WebSocketBridge, create_websocket_bridge
            from fixed_web_platform.browser_automation import BrowserAutomation
            
            self.websocket_bridge_class = WebSocketBridge
            self.create_websocket_bridge = create_websocket_bridge
            self.browser_automation_class = BrowserAutomation
            self.real_browser_available = True
            
            logger.info("WebSocket bridge and browser automation modules loaded successfully")
            
            # Create connection pool for browsers
            self.browser_connections = {}
            self.active_connections = 0
            
            # Create browser connection pool based on max_connections
            if self.adaptive_scaling:
                # Start with fewer connections and scale up as needed
                initial_connections = max(1, self.max_connections // 2)
                logger.info(f"Adaptive scaling enabled, starting with {initial_connections} browser connections")
                
                # Initialize adaptive manager if adaptive scaling is enabled
                from fixed_web_platform.adaptive_scaling import AdaptiveConnectionManager
                self.adaptive_manager = AdaptiveConnectionManager(
                    max_connections=self.max_connections,
                    browser_preferences=self.browser_preferences,
                    monitoring_interval=self.monitoring_interval
                )
                
                # Create browser connections
                await self._setup_initial_connections(initial_connections)
                
                # Initialize circuit breaker manager for connection health monitoring
                try:
                    from fixed_web_platform.resource_pool_circuit_breaker import ResourcePoolCircuitBreakerManager
                    self.circuit_breaker_manager = ResourcePoolCircuitBreakerManager(self.browser_connections)
                    await self.circuit_breaker_manager.initialize()
                    logger.info("Circuit breaker manager initialized for connection health monitoring")
                except ImportError as e:
                    logger.warning(f"Circuit breaker not available: {e}")
                except Exception as e:
                    logger.warning(f"Error initializing circuit breaker: {e}")
            else:
                # Create all connections at once
                logger.info(f"Adaptive scaling disabled, creating {self.max_connections} browser connections")
                await self._setup_initial_connections(self.max_connections)
            
            logger.info(f"Resource pool bridge initialized with {len(self.browser_connections)} browser connections")
            return True
            
        except ImportError as e:
            logger.warning(f"Could not import WebSocket bridge or browser automation: {e}")
            logger.info("Falling back to simulation mode")
            self.real_browser_available = False
            
            # Initialize adaptive manager if adaptive scaling is enabled (simulation mode)
            if self.adaptive_scaling:
                try:
                    from fixed_web_platform.adaptive_scaling import AdaptiveConnectionManager
                    self.adaptive_manager = AdaptiveConnectionManager(
                        max_connections=self.max_connections,
                        browser_preferences=self.browser_preferences,
                        monitoring_interval=self.monitoring_interval
                    )
                    logger.info("Adaptive scaling manager initialized in simulation mode")
                except ImportError:
                    logger.warning("Could not import AdaptiveConnectionManager, adaptive scaling disabled")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing resource pool bridge: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _setup_initial_connections(self, num_connections):
        """
        Set up initial browser connections.
        
        Args:
            num_connections: Number of connections to create
        """
        # Determine browser distribution
        browser_distribution = self._calculate_browser_distribution(num_connections)
        logger.info(f"Browser distribution: {browser_distribution}")
        
        # Create browser connections
        for browser, count in browser_distribution.items():
            for i in range(count):
                # Create connection with different port for each browser
                port = 8765 + len(self.browser_connections)
                
                # Determine platform to use (WebGPU or WebNN)
                # For text embedding models, WebNN on Edge is best
                # For audio models, WebGPU on Firefox is best
                # For vision models, WebGPU on Chrome is best
                platform = "webgpu"  # Default
                compute_shaders = False
                precompile_shaders = True
                parallel_loading = False
                
                if browser == "edge":
                    platform = "webnn"  # Edge has excellent WebNN support
                elif browser == "firefox":
                    compute_shaders = True  # Firefox has great compute shader performance
                
                # Launch browser and create WebSocket bridge
                connection_id = f"{browser}_{platform}_{i+1}"
                
                try:
                    # Set up browser automation
                    automation = self.browser_automation_class(
                        platform=platform,
                        browser_name=browser,
                        headless=self.headless,
                        compute_shaders=compute_shaders,
                        precompile_shaders=precompile_shaders,
                        parallel_loading=parallel_loading,
                        test_port=port
                    )
                    
                    # Launch browser
                    success = await automation.launch(allow_simulation=True)
                    
                    if success:
                        # Create WebSocket bridge
                        bridge = await self.create_websocket_bridge(port=port)
                        
                        if bridge:
                            # Wait for connection to be established
                            connected = await bridge.wait_for_connection(timeout=10)
                            
                            if connected:
                                # Store connection
                                self.browser_connections[connection_id] = {
                                    "automation": automation,
                                    "bridge": bridge,
                                    "platform": platform,
                                    "browser": browser,
                                    "port": port,
                                    "active": False,
                                    "initialized_models": set(),
                                    "compute_shaders": compute_shaders,
                                    "precompile_shaders": precompile_shaders,
                                    "parallel_loading": parallel_loading,
                                    "is_simulation": getattr(automation, "simulation_mode", True)
                                }
                                
                                logger.info(f"Successfully created browser connection: {connection_id}")
                                
                                # Check browser capabilities
                                capabilities = await bridge.get_browser_capabilities()
                                
                                if capabilities:
                                    # Update connection info with capabilities
                                    self.browser_connections[connection_id]["capabilities"] = capabilities
                                    
                                    # Log capability summary
                                    webgpu_support = capabilities.get("webgpu_supported", False)
                                    webnn_support = capabilities.get("webnn_supported", False)
                                    
                                    logger.info(f"Connection {connection_id} supports: WebGPU={webgpu_support}, WebNN={webnn_support}")
                                    
                                    if platform == "webgpu" and not webgpu_support:
                                        logger.warning(f"Connection {connection_id} is configured for WebGPU but does not support it")
                                    elif platform == "webnn" and not webnn_support:
                                        logger.warning(f"Connection {connection_id} is configured for WebNN but does not support it")
                            else:
                                logger.warning(f"Failed to establish WebSocket connection for {connection_id}")
                                await automation.close()
                        else:
                            logger.warning(f"Failed to create WebSocket bridge for {connection_id}")
                            await automation.close()
                    else:
                        logger.warning(f"Failed to launch browser for {connection_id}")
                except Exception as e:
                    logger.error(f"Error setting up browser connection {connection_id}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # If we have no connections but real browser is available, fall back to simulation
        if not self.browser_connections and self.real_browser_available:
            logger.warning("No browser connections could be established, falling back to simulation mode")
            self.real_browser_available = False
    
    def _calculate_browser_distribution(self, num_connections):
        """
        Calculate optimal browser distribution based on preferences.
        
        Args:
            num_connections: Number of connections to distribute
            
        Returns:
            Dict with browser distribution
        """
        # Default distribution
        distribution = {
            "chrome": 0,
            "firefox": 0,
            "edge": 0
        }
        
        # Get unique browser preferences from browser_preferences dict
        preferred_browsers = set(self.browser_preferences.values())
        
        if not preferred_browsers:
            # Default distribution if no preferences
            preferred_browsers = {"chrome", "firefox", "edge"}
        
        # Ensure we have at least the browsers in preferred_browsers
        browsers_to_use = list(preferred_browsers)
        num_browsers = len(browsers_to_use)
        
        # Distribute connections evenly across browsers
        base_count = num_connections // num_browsers
        remainder = num_connections % num_browsers
        
        for i, browser in enumerate(browsers_to_use):
            if browser in distribution:
                distribution[browser] = base_count
                if i < remainder:
                    distribution[browser] += 1
        
        return distribution
    
    async def get_model(self, model_type, model_name, hardware_preferences=None):
        """
        Get a model with optimal browser and platform selection.
        
        This enhanced implementation:
        1. Uses the adaptive scaling manager for optimal browser selection
        2. Intelligently selects the best browser based on model type
        3. Applies model-specific optimizations (Firefox for audio, Edge for text)
        4. Respects user hardware preferences when provided
        5. Uses real browser connections when available
        6. Leverages tensor sharing for efficient multi-model execution
        
        Args:
            model_type: Type of model (text, vision, audio, etc.)
            model_name: Name of the model to load
            hardware_preferences: Optional dict with hardware preferences
            
        Returns:
            Model object for inference (real or simulated)
        """
        # Get user-specified hardware preferences
        hardware_priority_list = []
        if hardware_preferences and 'priority_list' in hardware_preferences:
            hardware_priority_list = hardware_preferences['priority_list']
        
        # If no user preferences, determine optimal browser based on model type
        preferred_browser = None
        if hasattr(self, 'adaptive_manager'):
            # Use adaptive manager for optimal browser selection
            preferred_browser = self.adaptive_manager.get_browser_preference(model_type)
            
            # Update model type metrics
            if len(hardware_priority_list) > 0:
                # Estimate a reasonable inference time for metrics
                self.adaptive_manager.update_model_type_metrics(model_type, 0.5)  # 500ms is a reasonable default
        else:
            # Use static browser preferences
            for key, browser in self.browser_preferences.items():
                if key in model_type.lower():
                    preferred_browser = browser
                    break
            
            # Special case handling if no match found
            if not preferred_browser:
                if 'audio' in model_type.lower() or 'whisper' in model_type.lower() or 'wav2vec' in model_type.lower():
                    preferred_browser = 'firefox'  # Firefox has better WebGPU compute shader performance for audio
                elif 'vision' in model_type.lower() or 'clip' in model_type.lower() or 'vit' in model_type.lower():
                    preferred_browser = 'chrome'  # Chrome has good WebGPU support for vision models
                elif 'embedding' in model_type.lower() or 'bert' in model_type.lower():
                    preferred_browser = 'edge'  # Edge has excellent WebNN support for text embeddings
                else:
                    # Default to Chrome for unknown types
                    preferred_browser = 'chrome'
        
        # Extract optimization settings from hardware_preferences
        kwargs = {}
        if hardware_preferences:
            # Get optimization flags
            kwargs['compute_shaders'] = hardware_preferences.get('compute_shaders', False)
            kwargs['precompile_shaders'] = hardware_preferences.get('precompile_shaders', False)
            kwargs['parallel_loading'] = hardware_preferences.get('parallel_loading', False)
            kwargs['mixed_precision'] = hardware_preferences.get('mixed_precision', False)
            kwargs['precision'] = hardware_preferences.get('precision', 16)
            
            # Debug optimization flags
            logger.debug(f"Model optimization flags: {kwargs}")
        
        # Determine preferred hardware platform
        preferred_hardware = None
        if len(hardware_priority_list) > 0:
            preferred_hardware = hardware_priority_list[0]
        else:
            # Use WebGPU by default if no preference
            preferred_hardware = 'webgpu'
        
        # Check if we have real browser connections available
        if hasattr(self, 'browser_connections') and self.browser_connections and hasattr(self, 'real_browser_available') and self.real_browser_available:
            # Try to get a connection with the preferred browser and hardware platform
            connection = await self._get_connection_for_model(model_type, model_name, preferred_browser, preferred_hardware, **kwargs)
            
            if connection:
                # Create real browser model
                return await self._create_real_browser_model(connection, model_type, model_name, **kwargs)
            else:
                # Fall back to simulation
                logger.warning(f"No suitable browser connection available for {model_name}, falling back to simulation")
        
        # Set up tensor sharing if not already initialized
        if not hasattr(self, 'tensor_sharing_manager'):
            self.setup_tensor_sharing()
        
        # Check if tensor sharing is available and if we have a shared tensor for this model
        if hasattr(self, 'tensor_sharing_manager') and self.tensor_sharing_manager:
            # Generate tensor name based on model type
            if 'text_embedding' in model_type.lower() or 'bert' in model_type.lower():
                tensor_type = "text_embedding"
            elif 'vision' in model_type.lower() or 'vit' in model_type.lower():
                tensor_type = "vision_embedding"
            elif 'audio' in model_type.lower() or 'whisper' in model_type.lower():
                tensor_type = "audio_embedding"
            else:
                tensor_type = "embedding"
            
            embedding_tensor_name = f"{model_name}_{tensor_type}"
            
            # Check if this tensor is already available
            shared_tensor = self.tensor_sharing_manager.get_shared_tensor(embedding_tensor_name, model_name)
            if shared_tensor is not None:
                logger.info(f"Found shared tensor {embedding_tensor_name} for model {model_name}")
                
                # Add tensor sharing info to kwargs for the model to use
                kwargs['shared_tensors'] = {
                    tensor_type: embedding_tensor_name
                }
        
        # Either we don't have real browser connections or we couldn't get a suitable one
        # Fall back to simulation
        logger.debug(f"Using simulated model for {model_name} ({model_type}) using {preferred_hardware} with {preferred_browser}")
        return EnhancedWebModel(model_name, model_type, preferred_hardware, preferred_browser, **kwargs)
    
    async def _get_connection_for_model(self, model_type, model_name, preferred_browser, preferred_hardware, **kwargs):
        """
        Get an optimal browser connection for the model.
        
        This method selects the best available browser connection based on:
        1. Model type (text, vision, audio)
        2. Browser preference (edge, chrome, firefox)
        3. Hardware platform preference (webnn, webgpu)
        4. Optimization flags (compute_shaders, precompile_shaders, parallel_loading)
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            preferred_browser: Preferred browser
            preferred_hardware: Preferred hardware platform
            **kwargs: Additional optimization flags
            
        Returns:
            Selected connection or None if no suitable connection is available
        """
        # Score each connection for suitability
        connection_scores = {}
        
        # Get healthy connections if circuit breaker is available
        healthy_connections = []
        if hasattr(self, 'circuit_breaker_manager'):
            try:
                healthy_connections = await self.circuit_breaker_manager.circuit_breaker.get_healthy_connections()
                logger.debug(f"Healthy connections from circuit breaker: {healthy_connections}")
            except Exception as e:
                logger.warning(f"Error getting healthy connections from circuit breaker: {e}")
        
        for connection_id, connection in self.browser_connections.items():
            # Skip active connections (already in use)
            if connection["active"]:
                continue
            
            # Skip unhealthy connections if circuit breaker is available
            if hasattr(self, 'circuit_breaker_manager') and healthy_connections and connection_id not in healthy_connections:
                logger.warning(f"Skipping unhealthy connection {connection_id} based on circuit breaker health check")
                continue
            
            # Check if circuit breaker allows this connection
            if hasattr(self, 'circuit_breaker_manager'):
                try:
                    allowed, reason = await self.circuit_breaker_manager.pre_request_check(connection_id)
                    if not allowed:
                        logger.warning(f"Circuit breaker prevented use of connection {connection_id}: {reason}")
                        continue
                except Exception as e:
                    logger.warning(f"Error checking circuit breaker for connection {connection_id}: {e}")
            
            # Start with a base score
            score = 100
            
            # Check browser match
            if connection["browser"] == preferred_browser:
                score += 50
            elif connection["browser"] in ["chrome", "edge", "firefox"]:
                score += 10  # Any supported browser is better than nothing
            
            # Check platform match
            if connection["platform"] == preferred_hardware:
                score += 30
            
            # Check for compute shader support for audio models
            if 'audio' in model_type.lower() and connection["browser"] == "firefox" and connection["compute_shaders"]:
                score += 40  # Major bonus for audio models on Firefox with compute shaders
            
            # Check for WebNN support for text embedding models
            if ('text_embedding' in model_type.lower() or 'bert' in model_type.lower()) and connection["platform"] == "webnn":
                score += 35  # Bonus for text embedding models on WebNN
            
            # Check for precompile shaders for vision models
            if 'vision' in model_type.lower() and connection["precompile_shaders"]:
                score += 25  # Bonus for vision models with shader precompilation
            
            # Check for parallel loading for multimodal models
            if 'multimodal' in model_type.lower() and connection["parallel_loading"]:
                score += 30  # Bonus for multimodal models with parallel loading
            
            # Minor penalty for simulation mode
            if connection["is_simulation"]:
                score -= 15
            
            # Apply health score bonus if available from circuit breaker
            if hasattr(self, 'circuit_breaker_manager') and connection_id in self.circuit_breaker_manager.circuit_breaker.health_metrics:
                health_score = self.circuit_breaker_manager.circuit_breaker.health_metrics[connection_id].health_score
                # Normalize health score to 0-30 range and add as bonus
                health_bonus = (health_score / 100.0) * 30.0
                score += health_bonus
                logger.debug(f"Added health bonus of {health_bonus:.1f} to connection {connection_id} (health score: {health_score:.1f})")
                
            # Store score
            connection_scores[connection_id] = score
        
        # Get the best connection (highest score)
        if connection_scores:
            best_connection_id = max(connection_scores, key=connection_scores.get)
            best_score = connection_scores[best_connection_id]
            
            logger.info(f"Selected connection {best_connection_id} with score {best_score} for {model_name} ({model_type})")
            
            # Mark the connection as active
            self.browser_connections[best_connection_id]["active"] = True
            self.active_connections += 1
            
            # Return the connection
            return self.browser_connections[best_connection_id]
        
        return None
    
    async def _create_real_browser_model(self, connection, model_type, model_name, **kwargs):
        """
        Create a real browser model using the provided connection.
        
        This method initializes a model in the browser and returns a callable
        object that can be used for inference.
        
        Args:
            connection: Browser connection to use
            model_type: Type of model
            model_name: Name of the model
            **kwargs: Additional optimization flags
            
        Returns:
            Callable model object
        """
        # Extract connection components
        bridge = connection["bridge"]
        platform = connection["platform"]
        
        # Check if model is already initialized for this connection
        model_key = f"{model_name}_{platform}"
        if model_key not in connection["initialized_models"]:
            # Initialize model in browser
            logger.info(f"Initializing model {model_name} ({model_type}) in browser using {platform}")
            
            # Prepare initialization options
            options = {
                "compute_shaders": connection["compute_shaders"],
                "precompile_shaders": connection["precompile_shaders"],
                "parallel_loading": connection["parallel_loading"],
                "model_type": model_type
            }
            
            # Add additional options from kwargs
            for key, value in kwargs.items():
                if key not in options:
                    options[key] = value
            
            # Initialize model in browser
            init_result = await bridge.initialize_model(model_name, model_type, platform, options)
            
            if not init_result or init_result.get("status") != "success":
                logger.error(f"Failed to initialize model {model_name} in browser: {init_result.get('error', 'Unknown error')}")
                # Release the connection and fall back to simulation
                connection["active"] = False
                self.active_connections -= 1
                return EnhancedWebModel(model_name, model_type, platform, connection["browser"], **kwargs)
            
            # Mark model as initialized for this connection
            connection["initialized_models"].add(model_key)
            
            logger.info(f"Successfully initialized model {model_name} in browser")
        
        # Create callable model
        class RealBrowserModel:
            def __init__(self, pool, connection, bridge, model_name, model_type, platform):
                self.pool = pool
                self.connection = connection
                self.bridge = bridge
                self.model_name = model_name
                self.model_type = model_type
                self.platform = platform
                self.inference_count = 0
                
            async def __call__(self, inputs):
                """Run inference with the model."""
                self.inference_count += 1
                connection_id = None
                start_time = time.time()
                
                # Get connection ID
                for conn_id, conn in self.pool.browser_connections.items():
                    if conn is self.connection:
                        connection_id = conn_id
                        break
                
                try:
                    # Run inference
                    result = await self.bridge.run_inference(
                        self.model_name,
                        inputs,
                        self.platform
                    )
                    
                    # Calculate inference time
                    inference_time_ms = (time.time() - start_time) * 1000
                    
                    # Check for successful inference
                    if not result or result.get("status") != "success":
                        logger.error(f"Inference failed for {self.model_name}: {result.get('error', 'Unknown error')}")
                        
                        # Record failure with circuit breaker if available
                        if hasattr(self.pool, 'circuit_breaker_manager') and connection_id:
                            try:
                                await self.pool.circuit_breaker_manager.record_request_result(
                                    connection_id, 
                                    False, 
                                    error_type="inference_failed", 
                                    response_time_ms=inference_time_ms
                                )
                                
                                # Record model performance
                                await self.pool.circuit_breaker_manager.circuit_breaker.record_model_performance(
                                    connection_id,
                                    self.model_name,
                                    inference_time_ms,
                                    False
                                )
                                
                                # Handle error with circuit breaker
                                await self.pool.circuit_breaker_manager.handle_error(
                                    connection_id,
                                    Exception(result.get("error", "Unknown error")),
                                    {"action": "inference", "model_name": self.model_name}
                                )
                            except Exception as e:
                                logger.warning(f"Error recording failure with circuit breaker: {e}")
                        
                        return {
                            "success": False,
                            "error": result.get("error", "Unknown error"),
                            "model_name": self.model_name,
                            "model_type": self.model_type,
                            "hardware": self.platform,
                            "is_simulation": connection["is_simulation"]
                        }
                    
                    # Record success with circuit breaker if available
                    if hasattr(self.pool, 'circuit_breaker_manager') and connection_id:
                        try:
                            await self.pool.circuit_breaker_manager.record_request_result(
                                connection_id, 
                                True, 
                                response_time_ms=inference_time_ms
                            )
                            
                            # Record model performance
                            await self.pool.circuit_breaker_manager.circuit_breaker.record_model_performance(
                                connection_id,
                                self.model_name,
                                inference_time_ms,
                                True
                            )
                        except Exception as e:
                            logger.warning(f"Error recording success with circuit breaker: {e}")
                    
                    # Process and return result
                    output = {
                        "success": True,
                        "model_name": self.model_name,
                        "model_type": self.model_type,
                        "hardware": self.platform,
                        "browser": connection["browser"],
                        "is_real_hardware": not connection["is_simulation"],
                        "is_simulation": connection["is_simulation"],
                        "compute_shader_optimized": connection["compute_shaders"],
                        "precompile_shaders": connection["precompile_shaders"],
                        "parallel_loading": connection["parallel_loading"]
                    }
                    
                    # Copy performance metrics if available
                    if "performance_metrics" in result:
                        for key, value in result["performance_metrics"].items():
                            output[key] = value
                    
                    # Copy memory usage if available
                    if "memory_usage" in result:
                        output["memory_usage_mb"] = result["memory_usage"]
                    
                    # Copy result if available
                    if "result" in result:
                        output["result"] = result["result"]
                    
                    # Copy output if available
                    if "output" in result:
                        output["output"] = result["output"]
                    
                    return output
                    
                except Exception as e:
                    logger.error(f"Error during inference with {self.model_name}: {e}")
                    
                    # Calculate inference time even for failures
                    inference_time_ms = (time.time() - start_time) * 1000
                    
                    # Record failure with circuit breaker if available
                    if hasattr(self.pool, 'circuit_breaker_manager') and connection_id:
                        try:
                            await self.pool.circuit_breaker_manager.record_request_result(
                                connection_id, 
                                False, 
                                error_type="inference_exception", 
                                response_time_ms=inference_time_ms
                            )
                            
                            # Record model performance
                            await self.pool.circuit_breaker_manager.circuit_breaker.record_model_performance(
                                connection_id,
                                self.model_name,
                                inference_time_ms,
                                False
                            )
                            
                            # Handle error with circuit breaker
                            await self.pool.circuit_breaker_manager.handle_error(
                                connection_id,
                                e,
                                {"action": "inference", "model_name": self.model_name, "error_type": "exception"}
                            )
                        except Exception as circuit_error:
                            logger.warning(f"Error recording failure with circuit breaker: {circuit_error}")
                    
                    return {
                        "success": False,
                        "error": str(e),
                        "model_name": self.model_name,
                        "model_type": self.model_type,
                        "hardware": self.platform,
                        "is_simulation": connection["is_simulation"]
                    }
            
            def release(self):
                """Release the connection."""
                self.connection["active"] = False
                self.pool.active_connections -= 1
                logger.debug(f"Released connection for {self.model_name}")
        
        # Create a real model instance that uses the bridge for inference
        model = RealBrowserModel(self, connection, bridge, model_name, model_type, platform)
        
        # Wrap the async call method with a sync version
        def sync_call(inputs):
            if not hasattr(self, 'loop') or self.loop.is_closed():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            return self.loop.run_until_complete(model(inputs))
        
        # Replace the __call__ method with the sync version
        model.__call__ = sync_call
        
        return model
    
    async def get_health_status(self):
        """
        Get health status for all connections using the circuit breaker.
        
        This method provides detailed health information about all connections,
        including circuit state, health scores, and recovery recommendations.
        
        Returns:
            Dict with health status information
        """
        if hasattr(self, 'circuit_breaker_manager'):
            try:
                return await self.circuit_breaker_manager.get_health_summary()
            except Exception as e:
                logger.error(f"Error getting health status: {e}")
                return {"error": str(e)}
        else:
            return {"error": "Circuit breaker not available"}
            
    def get_health_status_sync(self):
        """
        Synchronous wrapper for get_health_status.
        
        Returns:
            Dict with health status information
        """
        # Create event loop if needed
        if not hasattr(self, 'loop') or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        # Run async method in event loop
        return self.loop.run_until_complete(self.get_health_status())
            
    def get_metrics(self):
        """
        Get detailed performance and resource metrics for the integration.
        
        This method provides comprehensive metrics about:
        - Connection utilization and scaling events
        - Browser distribution and preferences
        - Model performance by type and hardware
        - Adaptive scaling statistics (when enabled)
        - System resource utilization
        - Circuit breaker health status (if available)
        
        Returns:
            Dict with detailed metrics
        """
        # Base metrics
        metrics = {
            "connections": {
                "current": 0,
                "max": self.max_connections,
                "active": 0,
                "idle": 0,
                "utilization": 0.0,
                "browser_distribution": {},
                "platform_distribution": {}
            },
            "models": {},
            "performance": {
                "inference_times": {},
                "throughput": {},
                "memory_usage": {}
            },
            "adaptive_scaling": {
                "enabled": self.adaptive_scaling,
                "scaling_events": [],
                "current_metrics": {}
            },
            "resources": {
                "system_memory_percent": 0,
                "process_memory_mb": 0
            }
        }
        
        # Add adaptive manager metrics if available
        if hasattr(self, 'adaptive_manager'):
            adaptive_stats = self.adaptive_manager.get_scaling_stats()
            metrics["adaptive_scaling"]["current_metrics"] = adaptive_stats
            
            # Copy key metrics to top-level
            if "scaling_history" in adaptive_stats:
                metrics["adaptive_scaling"]["scaling_events"] = adaptive_stats["scaling_history"]
            
            # Add browser preferences from adaptive manager
            metrics["browser_preferences"] = self.browser_preferences
            
            # Add model type patterns
            if "model_type_patterns" in adaptive_stats:
                metrics["models"] = adaptive_stats["model_type_patterns"]
        
        # Get system metrics if available
        if PSUTIL_AVAILABLE:
            try:
                # Get system memory usage
                vm = psutil.virtual_memory()
                metrics["resources"]["system_memory_percent"] = vm.percent
                metrics["resources"]["system_memory_available_mb"] = vm.available / (1024 * 1024)
                
                # Get process memory usage
                try:
                    process = psutil.Process()
                    metrics["resources"]["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as proc_err:
                    # Handle specific process-related errors
                    logger.warning(f"Error accessing process metrics: {proc_err}")
                    metrics["resources"]["process_memory_mb"] = -1
                    metrics["resources"]["process_error"] = str(proc_err)
            except AttributeError as attr_err:
                # Handle missing attributes in psutil
                logger.warning(f"Psutil attribute error: {attr_err}")
                metrics["resources"]["error"] = f"Attribute error: {str(attr_err)}"
            except OSError as os_err:
                # Handle OS-level errors
                logger.warning(f"OS error when getting system metrics: {os_err}")
                metrics["resources"]["error"] = f"OS error: {str(os_err)}"
            except Exception as e:
                # Catch any other unexpected errors
                logger.warning(f"Unexpected error getting system metrics: {e}")
                metrics["resources"]["error"] = f"Unexpected error: {str(e)}"
        
        # Add circuit breaker metrics if available
        if hasattr(self, 'circuit_breaker_manager'):
            try:
                # Get circuit breaker states for all connections
                circuit_states = {}
                for connection_id in self.browser_connections.keys():
                    state = asyncio.run(self.circuit_breaker_manager.circuit_breaker.get_connection_state(connection_id))
                    if state:
                        circuit_states[connection_id] = {
                            "state": state["state"],
                            "failures": state["failures"],
                            "successes": state["successes"],
                            "health_score": state["health_metrics"]["health_score"] if "health_metrics" in state else 0
                        }
                
                # Add to metrics
                metrics["circuit_breaker"] = {
                    "circuit_states": circuit_states,
                    "healthy_count": len(asyncio.run(self.circuit_breaker_manager.circuit_breaker.get_healthy_connections()))
                }
            except Exception as e:
                metrics["circuit_breaker"] = {"error": str(e)}
        
        # Add timestamp
        metrics["timestamp"] = time.time()
        
        return metrics
    
    async def execute_concurrent(self, model_and_inputs_list):
        """
        Execute multiple models concurrently for efficient inference.
        
        This method enables concurrent execution of multiple models,
        which dramatically improves throughput when running multiple
        models simultaneously. It leverages asyncio for efficient
        concurrent execution with shared resources.
        
        Args:
            model_and_inputs_list: List of (model, inputs) tuples to execute
            
        Returns:
            List of results in the same order as inputs
        """
        if not model_and_inputs_list:
            return []
        
        # Create tasks for concurrent execution
        tasks = []
        for model, inputs in model_and_inputs_list:
            if not model:
                # Use a dummy task for None models
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
            else:
                # Create task for each model execution
                async def call_model(model, inputs):
                    try:
                        return model(inputs)
                    except TypeError as e:
                        # Handle invalid input types
                        logger.error(f"Type error executing model {getattr(model, 'model_name', 'unknown')}: {e}")
                        return {"success": False, "error_type": "input_type_error", "error": str(e)}
                    except ValueError as e:
                        # Handle invalid input values
                        logger.error(f"Value error executing model {getattr(model, 'model_name', 'unknown')}: {e}")
                        return {"success": False, "error_type": "input_value_error", "error": str(e)}
                    except RuntimeError as e:
                        # Handle runtime execution errors
                        logger.error(f"Runtime error executing model {getattr(model, 'model_name', 'unknown')}: {e}")
                        return {"success": False, "error_type": "runtime_error", "error": str(e)}
                    except Exception as e:
                        # Catch any other unexpected errors
                        logger.error(f"Unexpected error executing model {getattr(model, 'model_name', 'unknown')}: {e}")
                        return {"success": False, "error_type": "unexpected_error", "error": str(e)}
                
                tasks.append(asyncio.create_task(call_model(model, inputs)))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create detailed error result with categorization
                model, inputs = model_and_inputs_list[i]
                model_name = getattr(model, 'model_name', 'unknown')
                model_type = getattr(model, 'model_type', 'unknown')
                
                # Categorize the exception for better error handling
                if isinstance(result, asyncio.TimeoutError):
                    error_type = "timeout"
                    error_category = "execution_timeout"
                elif isinstance(result, asyncio.CancelledError):
                    error_type = "cancelled"
                    error_category = "execution_cancelled"
                elif isinstance(result, (TypeError, ValueError)):
                    error_type = "input_error"
                    error_category = "invalid_input"
                elif isinstance(result, RuntimeError):
                    error_type = "runtime_error"
                    error_category = "execution_failed"
                else:
                    error_type = "unknown_error"
                    error_category = "unexpected_exception"
                
                # Create detailed error response
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'error_type': error_type,
                    'error_category': error_category,
                    'model_name': model_name,
                    'model_type': model_type,
                    'input_type': type(inputs).__name__,
                    'timestamp': time.time(),
                    'traceback': getattr(result, '__traceback__', None) and ''.join(
                        traceback.format_exception(type(result), result, result.__traceback__)
                    )
                })
                
                # Log error with stack trace for debugging
                logger.error(f"Error executing model {model_name}: {result}")
                
            elif result is None:
                # Handle None results explicitly
                model, _ = model_and_inputs_list[i]
                model_name = getattr(model, 'model_name', 'unknown')
                processed_results.append({
                    'success': False, 
                    'error_type': 'null_result',
                    'error': 'Model returned None',
                    'model_name': model_name,
                    'timestamp': time.time()
                })
            else:
                # Normal successful result
                processed_results.append(result)
        
        return processed_results
    
    def execute_concurrent_sync(self, model_and_inputs_list):
        """
        Synchronous wrapper for execute_concurrent.
        
        This method provides a synchronous interface to the asynchronous
        execute_concurrent method, making it easy to use in synchronous code.
        
        Args:
            model_and_inputs_list: List of (model, inputs) tuples to execute
            
        Returns:
            List of results in the same order as inputs
        """
        # Create event loop if needed
        if not hasattr(self, 'loop') or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        # Run async method in event loop
        return self.loop.run_until_complete(self.execute_concurrent(model_and_inputs_list))
    
    async def close(self):
        """
        Close all resources and connections.
        
        This method gracefully shuts down all browser connections,
        closes WebSocket bridges, and releases resources.
        """
        logger.info("Closing resource pool bridge...")
        
        # Close circuit breaker manager if available
        if hasattr(self, 'circuit_breaker_manager'):
            logger.info("Closing circuit breaker manager")
            try:
                await self.circuit_breaker_manager.close()
            except Exception as e:
                logger.warning(f"Error closing circuit breaker manager: {e}")
        
        # Close all active browser connections
        if hasattr(self, 'browser_connections'):
            for connection_id, connection in list(self.browser_connections.items()):
                try:
                    logger.info(f"Closing browser connection: {connection_id}")
                    
                    # Try to shut down the WebSocket bridge
                    if "bridge" in connection:
                        try:
                            await connection["bridge"].shutdown_browser()
                            await connection["bridge"].stop()
                        except Exception as e:
                            logger.warning(f"Error shutting down WebSocket bridge for {connection_id}: {e}")
                    
                    # Try to close the browser automation
                    if "automation" in connection:
                        try:
                            await connection["automation"].close()
                        except Exception as e:
                            logger.warning(f"Error closing browser automation for {connection_id}: {e}")
                    
                except Exception as e:
                    logger.error(f"Error closing connection {connection_id}: {e}")
        
        # Close adaptive manager if available
        if hasattr(self, 'adaptive_manager'):
            logger.info("Closing adaptive connection manager")
            
            # If adaptive manager has a close method, call it
            if hasattr(self.adaptive_manager, 'close'):
                try:
                    if asyncio.iscoroutinefunction(self.adaptive_manager.close):
                        await self.adaptive_manager.close()
                    else:
                        self.adaptive_manager.close()
                except Exception as e:
                    logger.warning(f"Error closing adaptive manager: {e}")
        
        logger.info("Resource pool bridge closed")
        return True
    
    def close_sync(self):
        """Synchronous wrapper for close."""
        # Create event loop if needed
        if not hasattr(self, 'loop') or self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        # Run async close method in event loop
        return self.loop.run_until_complete(self.close())
        
    def setup_tensor_sharing(self, max_memory_mb=None):
        """
        Set up cross-model tensor sharing for this resource pool.
        
        This enables efficient tensor sharing between models, reducing memory usage
        and improving performance for multi-model workloads.
        
        Args:
            max_memory_mb: Maximum memory to allocate for shared tensors (in MB)
            
        Returns:
            TensorSharingManager instance
        """
        try:
            from fixed_web_platform.cross_model_tensor_sharing import TensorSharingManager
            self.tensor_sharing_manager = TensorSharingManager(max_memory_mb=max_memory_mb)
            logger.info(f"Tensor sharing enabled with max memory: {max_memory_mb} MB")
            return self.tensor_sharing_manager
        except ImportError:
            logger.warning("Cross-model tensor sharing not available. The 'cross_model_tensor_sharing' module could not be imported.")
            return None
            
    async def share_tensor_between_models(self, tensor_data, tensor_name, producer_model, consumer_models, 
                                       shape=None, storage_type="cpu", dtype="float32"):
        """
        Share a tensor between models in the resource pool.
        
        Args:
            tensor_data: The tensor data to share (optional if registering external tensor)
            tensor_name: Name for the shared tensor
            producer_model: Model that produced the tensor
            consumer_models: List of models that will consume the tensor
            shape: Shape of the tensor (required if tensor_data is None)
            storage_type: Storage type (cpu, webgpu, webnn)
            dtype: Data type of the tensor
            
        Returns:
            Registration result (success boolean and tensor info)
        """
        if not hasattr(self, 'tensor_sharing_manager'):
            # Create tensor sharing manager if it doesn't exist yet
            self.setup_tensor_sharing()
            
        if self.tensor_sharing_manager is None:
            return {"success": False, "error": "Tensor sharing manager not available"}
            
        # Register the tensor
        if shape is None and tensor_data is not None:
            # Infer shape from tensor_data if not provided
            if hasattr(tensor_data, 'shape'):
                shape = list(tensor_data.shape)
            else:
                return {"success": False, "error": "Must provide shape when tensor_data doesn't have shape attribute"}
        elif shape is None:
            return {"success": False, "error": "Must provide shape when tensor_data is None"}
            
        # Register the tensor with the manager
        shared_tensor = self.tensor_sharing_manager.register_shared_tensor(
            name=tensor_name,
            shape=shape,
            storage_type=storage_type,
            producer_model=producer_model,
            consumer_models=consumer_models,
            dtype=dtype
        )
        
        # Store the actual tensor data if provided
        if tensor_data is not None:
            shared_tensor.data = tensor_data
            
        logger.info(f"Registered shared tensor {tensor_name} for models: {producer_model} -> {consumer_models}")
        
        return {
            "success": True,
            "tensor_name": tensor_name,
            "producer": producer_model,
            "consumers": consumer_models,
            "storage_type": storage_type,
            "shape": shape,
            "memory_mb": shared_tensor.get_memory_usage() / (1024*1024)
        }

# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test_resource_pool():
        # Create and initialize with the new async interface
        integration = ResourcePoolBridgeIntegration(adaptive_scaling=True)
        success = await integration.initialize()
        
        if not success:
            print("Failed to initialize resource pool bridge")
            return
        
        print("Resource pool bridge initialized successfully")
        
        try:
            # Test single model with the new async get_model
            print("\nGetting text model (BERT)...")
            model = await integration.get_model("text", "bert-base-uncased", {"priority_list": ["webgpu", "cpu"]})
            result = model("Sample text")
            print("Single model result:")
            print(json.dumps(result, indent=2))
            
            # Test concurrent execution with different model types
            print("\nGetting vision model (ViT)...")
            model2 = await integration.get_model("vision", "vit-base", {"priority_list": ["webgpu"]})
            
            print("Getting audio model (Whisper)...")
            model3 = await integration.get_model("audio", "whisper-tiny", {
                "priority_list": ["webgpu"],
                "compute_shaders": True  # Enable compute shaders for audio models
            })
            
            models_and_inputs = [
                (model, "Text input for BERT"),
                (model2, {"image": {"width": 224, "height": 224}}),
                (model3, {"audio": {"duration": 5.0}})
            ]
            
            print("\nRunning concurrent execution...")
            results = integration.execute_concurrent_sync(models_and_inputs)
            print("Concurrent execution results:")
            for i, result in enumerate(results):
                print(f"\nModel {i+1} result:")
                print(json.dumps(result, indent=2))
            
            # Get metrics
            metrics = integration.get_metrics()
            print("\nMetrics:")
            print(json.dumps(metrics, indent=2))
            
        finally:
            # Ensure clean shutdown
            print("\nClosing resource pool bridge...")
            await integration.close()
            print("Resource pool bridge closed")
    
    # Run the async test function
    asyncio.run(test_resource_pool())
