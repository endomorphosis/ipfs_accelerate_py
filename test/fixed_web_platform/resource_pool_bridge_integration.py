#!/usr/bin/env python3
"""
Resource Pool Bridge Integration with Recovery System (March 2025)

This module integrates the WebNN/WebGPU Resource Pool Bridge with the Recovery System
and advanced features like connection pooling, health monitoring with circuit breaker pattern,
cross-model tensor sharing, and ultra-low precision support.

Key features:
- Automatic error recovery for browser connection issues
- Smart fallbacks between WebNN, WebGPU, and CPU simulation
- Browser-specific optimizations and automatic selection
- Performance monitoring and degradation detection
- Comprehensive error categorization and recovery strategies
- Detailed metrics and telemetry

Usage:
    from fixed_web_platform.resource_pool_bridge_integration import ResourcePoolBridgeIntegrationWithRecovery
    
    # Create integrated pool with recovery
    pool = ResourcePoolBridgeIntegrationWithRecovery(max_connections=4)
    
    # Initialize 
    pool.initialize()
    
    # Get model with automatic recovery
    model = pool.get_model(model_type="text", model_name="bert-base-uncased")
    
    # Run inference with recovery
    result = model(inputs)
"""

import os
import sys
import time
import json
import asyncio
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set

# Import connection pooling and health monitoring
try:
    from fixed_web_platform.connection_pool_manager import ConnectionPoolManager
    from fixed_web_platform.resource_pool_circuit_breaker import ResourcePoolCircuitBreakerManager
    ADVANCED_POOLING_AVAILABLE = True
except ImportError:
    ADVANCED_POOLING_AVAILABLE = False

# Import tensor sharing
try:
    from fixed_web_platform.cross_model_tensor_sharing import TensorSharingManager
    TENSOR_SHARING_AVAILABLE = True
except ImportError:
    TENSOR_SHARING_AVAILABLE = False
    
# Import ultra-low precision support
try:
    from fixed_web_platform.webgpu_ultra_low_precision import UltraLowPrecisionManager
    ULTRA_LOW_PRECISION_AVAILABLE = True
except ImportError:
    ULTRA_LOW_PRECISION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import recovery system
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import recovery system
try:
    from resource_pool_bridge_recovery import (
        ResourcePoolBridgeRecovery,
        ResourcePoolBridgeWithRecovery,
        ErrorCategory, 
        RecoveryStrategy
    )
    RECOVERY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import resource_pool_bridge_recovery: {e}")
    logger.warning("Continuing without recovery capabilities")
    RECOVERY_AVAILABLE = False


class ResourcePoolBridgeIntegrationWithRecovery:
    """
    Enhanced WebNN/WebGPU Resource Pool with Recovery System Integration (March 2025).
    
    This class integrates the ResourcePoolBridgeIntegration with the ResourcePoolBridgeRecovery
    system to provide fault-tolerant, resilient operation for web-based AI acceleration.
    
    The March 2025 enhancements include:
    - Advanced connection pooling with browser-specific optimizations
    - Health monitoring with circuit breaker pattern for graceful degradation
    - Cross-model tensor sharing for memory efficiency
    - Ultra-low bit quantization (2-bit, 3-bit) with shared KV cache
    - Enhanced error recovery with performance-based strategies
    """
    
    def __init__(
        self,
        max_connections: int = 4,
        enable_gpu: bool = True,
        enable_cpu: bool = True,
        headless: bool = True,
        browser_preferences: Optional[Dict[str, str]] = None,
        adaptive_scaling: bool = True,
        enable_recovery: bool = True,
        max_retries: int = 3,
        fallback_to_simulation: bool = True,
        monitoring_interval: int = 60,
        enable_ipfs: bool = True,
        db_path: Optional[str] = None,
        enable_tensor_sharing: bool = True,
        enable_ultra_low_precision: bool = True,
        enable_circuit_breaker: bool = True,
        max_memory_mb: int = 2048
    ):
        """
        Initialize the integrated resource pool with recovery.
        
        Args:
            max_connections: Maximum browser connections to maintain
            enable_gpu: Whether to enable GPU acceleration
            enable_cpu: Whether to enable CPU fallback
            headless: Whether to run browsers in headless mode
            browser_preferences: Browser preferences by model type
            adaptive_scaling: Whether to dynamically scale connections based on load
            enable_recovery: Whether to enable recovery capabilities
            max_retries: Maximum number of retry attempts per operation
            fallback_to_simulation: Whether to allow fallback to simulation mode
            monitoring_interval: Interval for monitoring in seconds
            enable_ipfs: Whether to enable IPFS acceleration
            db_path: Path to database for storing results
            enable_tensor_sharing: Whether to enable cross-model tensor sharing for memory efficiency
            enable_ultra_low_precision: Whether to enable 2-bit and 3-bit quantization support
            enable_circuit_breaker: Whether to enable circuit breaker pattern for health monitoring
            max_memory_mb: Maximum memory usage in MB for tensor sharing and browser connections
        """
        self.max_connections = max_connections
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu
        self.headless = headless
        self.browser_preferences = browser_preferences or {}
        self.adaptive_scaling = adaptive_scaling
        self.enable_recovery = enable_recovery and RECOVERY_AVAILABLE
        self.max_retries = max_retries
        self.fallback_to_simulation = fallback_to_simulation
        self.monitoring_interval = monitoring_interval
        self.enable_ipfs = enable_ipfs
        self.db_path = db_path
        
        # March 2025 enhancements
        self.enable_tensor_sharing = enable_tensor_sharing and TENSOR_SHARING_AVAILABLE
        self.enable_ultra_low_precision = enable_ultra_low_precision and ULTRA_LOW_PRECISION_AVAILABLE
        self.enable_circuit_breaker = enable_circuit_breaker and ADVANCED_POOLING_AVAILABLE
        self.max_memory_mb = max_memory_mb
        
        # Initialize logger
        logger.info(f"ResourcePoolBridgeIntegrationWithRecovery created with max_connections={max_connections}, "
                   f"recovery={'enabled' if self.enable_recovery else 'disabled'}, "
                   f"adaptive_scaling={'enabled' if adaptive_scaling else 'disabled'}, "
                   f"tensor_sharing={'enabled' if self.enable_tensor_sharing else 'disabled'}, "
                   f"ultra_low_precision={'enabled' if self.enable_ultra_low_precision else 'disabled'}, "
                   f"circuit_breaker={'enabled' if self.enable_circuit_breaker else 'disabled'}")
        
        # Will be initialized in initialize()
        self.bridge = None
        self.bridge_with_recovery = None
        self.initialized = False
        
        # March 2025 enhancements
        self.connection_pool = None
        self.circuit_breaker = None
        self.tensor_sharing_manager = None
        self.ultra_low_precision_manager = None
    
    def initialize(self) -> bool:
        """
        Initialize the resource pool bridge with recovery capabilities.
        
        Returns:
            bool: Success status
        """
        try:
            # Import core bridge implementation
            from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
            
            # Create base bridge
            self.bridge = ResourcePoolBridgeIntegration(
                max_connections=self.max_connections,
                enable_gpu=self.enable_gpu,
                enable_cpu=self.enable_cpu,
                headless=self.headless,
                browser_preferences=self.browser_preferences,
                adaptive_scaling=self.adaptive_scaling,
                monitoring_interval=self.monitoring_interval,
                enable_ipfs=self.enable_ipfs,
                db_path=self.db_path
            )
            
            # Initialize March 2025 enhancements
            
            # Initialize tensor sharing if enabled
            if self.enable_tensor_sharing and TENSOR_SHARING_AVAILABLE:
                logger.info("Initializing cross-model tensor sharing")
                self.tensor_sharing_manager = TensorSharingManager(max_memory_mb=self.max_memory_mb)
            
            # Initialize ultra-low precision if enabled
            if self.enable_ultra_low_precision and ULTRA_LOW_PRECISION_AVAILABLE:
                logger.info("Initializing ultra-low precision support")
                self.ultra_low_precision_manager = UltraLowPrecisionManager()
            
            # Initialize base bridge
            if hasattr(self.bridge, 'initialize'):
                loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
                success = loop.run_until_complete(self.bridge.initialize())
                if not success:
                    logger.error("Failed to initialize base bridge")
                    return False
            
            # Create recovery wrapper if enabled
            if self.enable_recovery:
                self.bridge_with_recovery = ResourcePoolBridgeWithRecovery(
                    integration=self.bridge,
                    max_connections=self.max_connections,
                    browser_preferences=self.browser_preferences,
                    max_retries=self.max_retries,
                    fallback_to_simulation=self.fallback_to_simulation
                )
                
                # Initialize recovery bridge
                success = self.bridge_with_recovery.initialize()
                if not success:
                    logger.error("Failed to initialize recovery bridge")
                    return False
            
            # Initialize connection pool and circuit breaker if enabled
            if self.enable_circuit_breaker and ADVANCED_POOLING_AVAILABLE:
                logger.info("Initializing connection pool and circuit breaker")
                
                # Get browser connections from bridge
                browser_connections = {}
                if hasattr(self.bridge, 'browser_connections'):
                    browser_connections = self.bridge.browser_connections
                
                if browser_connections:
                    # Create connection pool manager
                    self.connection_pool = ConnectionPoolManager(
                        min_connections=1,
                        max_connections=self.max_connections,
                        browser_preferences=self.browser_preferences,
                        adaptive_scaling=self.adaptive_scaling,
                        db_path=self.db_path
                    )
                    
                    # Create circuit breaker manager
                    self.circuit_breaker = ResourcePoolCircuitBreakerManager(browser_connections)
                    
                    # Initialize them
                    loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
                    loop.run_until_complete(self.connection_pool.initialize())
                    loop.run_until_complete(self.circuit_breaker.initialize())
                    
                    logger.info("Connection pool and circuit breaker initialized successfully")
            
            self.initialized = True
            logger.info(f"ResourcePoolBridgeIntegrationWithRecovery initialized successfully "
                       f"(recovery={'enabled' if self.enable_recovery else 'disabled'}, "
                       f"tensor_sharing={'enabled' if self.tensor_sharing_manager else 'disabled'}, "
                       f"ultra_low_precision={'enabled' if self.ultra_low_precision_manager else 'disabled'}, "
                       f"circuit_breaker={'enabled' if self.circuit_breaker else 'disabled'})")
            return True
            
        except ImportError as e:
            logger.error(f"Error importing required modules: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing resource pool bridge: {e}")
            traceback.print_exc()
            return False
    
    def get_model(self, model_type: str, model_name: str, hardware_preferences: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get a model with fault-tolerant error handling and recovery.
        
        Args:
            model_type: Type of model (text, vision, audio, etc.)
            model_name: Name of the model
            hardware_preferences: Hardware preferences for model execution
            
        Returns:
            Model object or None on failure
        """
        if not self.initialized:
            logger.error("ResourcePoolBridgeIntegrationWithRecovery not initialized")
            return None
        
        # Use recovery bridge if enabled
        if self.enable_recovery and self.bridge_with_recovery:
            return self.bridge_with_recovery.get_model(
                model_type=model_type,
                model_name=model_name,
                hardware_preferences=hardware_preferences
            )
        
        # Fall back to base bridge if recovery not enabled
        if hasattr(self.bridge, 'get_model'):
            loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
            return loop.run_until_complete(
                self.bridge.get_model(
                    model_type=model_type,
                    model_name=model_name,
                    hardware_preferences=hardware_preferences
                )
            )
        
        return None
    
    def execute_concurrent(self, model_and_inputs_list: List[Tuple[Any, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple models concurrently with fault-tolerant error handling.
        
        Args:
            model_and_inputs_list: List of (model, inputs) tuples
            
        Returns:
            List of results corresponding to inputs
        """
        if not self.initialized:
            logger.error("ResourcePoolBridgeIntegrationWithRecovery not initialized")
            return [{"success": False, "error": "Not initialized"} for _ in model_and_inputs_list]
        
        # Use recovery bridge if enabled
        if self.enable_recovery and self.bridge_with_recovery:
            return self.bridge_with_recovery.execute_concurrent(model_and_inputs_list)
        
        # Fall back to base bridge if recovery not enabled
        if hasattr(self.bridge, 'execute_concurrent_sync'):
            return self.bridge.execute_concurrent_sync(model_and_inputs_list)
        elif hasattr(self.bridge, 'execute_concurrent'):
            loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
            return loop.run_until_complete(self.bridge.execute_concurrent(model_and_inputs_list))
            
        return [{"success": False, "error": "execute_concurrent not available"} for _ in model_and_inputs_list]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics including recovery statistics.
        
        Returns:
            Dict containing metrics and recovery statistics
        """
        # Start with basic metrics
        metrics = {
            "timestamp": time.time(),
            "recovery_enabled": self.enable_recovery,
            "initialized": self.initialized
        }
        
        # Add recovery metrics if enabled
        if self.enable_recovery and self.bridge_with_recovery:
            recovery_metrics = self.bridge_with_recovery.get_metrics()
            metrics.update(recovery_metrics)
        elif self.bridge and hasattr(self.bridge, 'get_metrics'):
            # Get base bridge metrics
            base_metrics = self.bridge.get_metrics()
            metrics["base_metrics"] = base_metrics
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the resource pool including all March 2025 enhancements.
        
        Returns:
            Dict with comprehensive health status information
        """
        if not self.initialized:
            return {"status": "not_initialized"}
        
        # Get base health status
        if self.enable_recovery and self.bridge_with_recovery and hasattr(self.bridge_with_recovery, 'get_health_status_sync'):
            status = self.bridge_with_recovery.get_health_status_sync()
        elif hasattr(self.bridge, 'get_health_status_sync'):
            status = self.bridge.get_health_status_sync()
        elif hasattr(self.bridge, 'get_health_status'):
            loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
            status = loop.run_until_complete(self.bridge.get_health_status())
        else:
            status = {"status": "unknown"}
        
        # Add circuit breaker health status if enabled
        if self.enable_circuit_breaker and self.circuit_breaker:
            circuit_health = {"status": "not_available"}
            try:
                loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
                circuit_health = loop.run_until_complete(self.circuit_breaker.get_health_summary())
            except Exception as e:
                logger.error(f"Error getting circuit breaker health: {e}")
            
            status["circuit_breaker"] = circuit_health
            
        # Add tensor sharing status if enabled
        if self.enable_tensor_sharing and self.tensor_sharing_manager:
            try:
                tensor_stats = self.tensor_sharing_manager.get_stats()
                status["tensor_sharing"] = tensor_stats
            except Exception as e:
                logger.error(f"Error getting tensor sharing stats: {e}")
                status["tensor_sharing"] = {"error": str(e)}
                
        # Add ultra-low precision status if enabled
        if self.enable_ultra_low_precision and self.ultra_low_precision_manager:
            try:
                ulp_stats = self.ultra_low_precision_manager.get_stats()
                status["ultra_low_precision"] = ulp_stats
            except Exception as e:
                logger.error(f"Error getting ultra-low precision stats: {e}")
                status["ultra_low_precision"] = {"error": str(e)}
        
        return status
    
    def close(self) -> bool:
        """
        Close all resources with proper cleanup, including March 2025 enhancements.
        
        Returns:
            Success status
        """
        success = True
        
        # Close March 2025 enhancements first
        
        # Close circuit breaker if enabled
        if self.enable_circuit_breaker and self.circuit_breaker:
            try:
                logger.info("Closing circuit breaker manager")
                loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
                loop.run_until_complete(self.circuit_breaker.close())
                logger.info("Circuit breaker manager closed successfully")
            except Exception as e:
                logger.error(f"Error closing circuit breaker manager: {e}")
                success = False
        
        # Close connection pool if enabled
        if self.enable_circuit_breaker and self.connection_pool:
            try:
                logger.info("Closing connection pool manager")
                loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
                loop.run_until_complete(self.connection_pool.shutdown())
                logger.info("Connection pool manager closed successfully")
            except Exception as e:
                logger.error(f"Error closing connection pool manager: {e}")
                success = False
        
        # Clean up tensor sharing if enabled
        if self.enable_tensor_sharing and self.tensor_sharing_manager:
            try:
                logger.info("Cleaning up tensor sharing manager")
                self.tensor_sharing_manager.cleanup()
                logger.info("Tensor sharing manager cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up tensor sharing manager: {e}")
                success = False
        
        # Clean up ultra-low precision if enabled
        if self.enable_ultra_low_precision and self.ultra_low_precision_manager:
            try:
                logger.info("Cleaning up ultra-low precision manager")
                self.ultra_low_precision_manager.cleanup()
                logger.info("Ultra-low precision manager cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up ultra-low precision manager: {e}")
                success = False
        
        # Close recovery bridge if enabled
        if self.enable_recovery and self.bridge_with_recovery:
            try:
                self.bridge_with_recovery.close()
                logger.info("Recovery bridge closed successfully")
            except Exception as e:
                logger.error(f"Error closing recovery bridge: {e}")
                success = False
        
        # Close base bridge
        if self.bridge:
            try:
                if hasattr(self.bridge, 'close_sync'):
                    self.bridge.close_sync()
                elif hasattr(self.bridge, 'close'):
                    loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
                    loop.run_until_complete(self.bridge.close())
                logger.info("Base bridge closed successfully")
            except Exception as e:
                logger.error(f"Error closing base bridge: {e}")
                success = False
        
        self.initialized = False
        logger.info(f"ResourcePoolBridgeIntegrationWithRecovery closed (success={'yes' if success else 'no'}, "
                   f"closed tensor_sharing={'yes' if self.tensor_sharing_manager else 'n/a'}, "
                   f"closed ultra_low_precision={'yes' if self.ultra_low_precision_manager else 'n/a'}, "
                   f"closed circuit_breaker={'yes' if self.circuit_breaker else 'n/a'})")
        return success
    
    def setup_tensor_sharing(self, max_memory_mb: Optional[int] = None) -> Any:
        """
        Set up cross-model tensor sharing for memory efficiency.
        
        This feature enables multiple models to share tensors, significantly
        improving memory efficiency and performance for multi-model workloads.
        
        Args:
            max_memory_mb: Maximum memory in MB to use for tensor sharing (overrides the initial setting)
            
        Returns:
            TensorSharingManager instance or None if not available
        """
        if not self.initialized:
            logger.error("ResourcePoolBridgeIntegrationWithRecovery not initialized")
            return None
            
        # Check if tensor sharing is enabled
        if not self.enable_tensor_sharing:
            logger.warning("Tensor sharing is not enabled")
            return None
            
        # Check if tensor sharing is available
        if not TENSOR_SHARING_AVAILABLE:
            logger.warning("Tensor sharing is not available (missing dependencies)")
            return None
        
        # Use recovery bridge if enabled
        if self.enable_recovery and self.bridge_with_recovery and hasattr(self.bridge_with_recovery, 'setup_tensor_sharing'):
            return self.bridge_with_recovery.setup_tensor_sharing(max_memory_mb=max_memory_mb)
        
        # Fall back to base bridge if recovery not enabled
        if hasattr(self.bridge, 'setup_tensor_sharing'):
            return self.bridge.setup_tensor_sharing(max_memory_mb=max_memory_mb)
            
        # Use local tensor sharing implementation if no bridge implementation available
        try:
            # Use existing manager if already created
            if self.tensor_sharing_manager:
                if max_memory_mb is not None:
                    # Update memory limit if specified
                    self.tensor_sharing_manager.set_max_memory(max_memory_mb)
                return self.tensor_sharing_manager
                
            # Create new manager if not already created
            memory_limit = max_memory_mb if max_memory_mb is not None else self.max_memory_mb
            self.tensor_sharing_manager = TensorSharingManager(max_memory_mb=memory_limit)
            logger.info(f"Tensor sharing manager created with {memory_limit} MB memory limit")
            return self.tensor_sharing_manager
            
        except Exception as e:
            logger.error(f"Error setting up tensor sharing: {e}")
            return None

    def share_tensor_between_models(
        self, 
        tensor_data: Any, 
        tensor_name: str, 
        producer_model: Any, 
        consumer_models: List[Any], 
        shape: Optional[List[int]] = None, 
        storage_type: str = "cpu", 
        dtype: str = "float32"
    ) -> Dict[str, Any]:
        """
        Share a tensor between models.
        
        Args:
            tensor_data: The tensor data to share
            tensor_name: Name for the shared tensor
            producer_model: Model that produced the tensor
            consumer_models: List of models that will consume the tensor
            shape: Shape of the tensor (required if tensor_data is None)
            storage_type: Storage type (cpu, webgpu, webnn)
            dtype: Data type of the tensor
            
        Returns:
            Registration result (success boolean and tensor info)
        """
        if not self.initialized:
            logger.error("ResourcePoolBridgeIntegrationWithRecovery not initialized")
            return {"success": False, "error": "Not initialized"}
        
        # Use recovery bridge if enabled
        if self.enable_recovery and self.bridge_with_recovery and hasattr(self.bridge_with_recovery, 'share_tensor_between_models'):
            # Wrap in try/except to handle async methods
            try:
                return self.bridge_with_recovery.share_tensor_between_models(
                    tensor_data=tensor_data,
                    tensor_name=tensor_name,
                    producer_model=producer_model,
                    consumer_models=consumer_models,
                    shape=shape,
                    storage_type=storage_type,
                    dtype=dtype
                )
            except AttributeError:
                # Might be an async method
                loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
                return loop.run_until_complete(
                    self.bridge_with_recovery.share_tensor_between_models(
                        tensor_data=tensor_data,
                        tensor_name=tensor_name,
                        producer_model=producer_model,
                        consumer_models=consumer_models,
                        shape=shape,
                        storage_type=storage_type,
                        dtype=dtype
                    )
                )
        
        # Fall back to base bridge if recovery not enabled
        if hasattr(self.bridge, 'share_tensor_between_models'):
            # Check if it's an async method
            if asyncio.iscoroutinefunction(self.bridge.share_tensor_between_models):
                loop = asyncio.get_event_loop() if hasattr(asyncio, 'get_event_loop') else asyncio.new_event_loop()
                return loop.run_until_complete(
                    self.bridge.share_tensor_between_models(
                        tensor_data=tensor_data,
                        tensor_name=tensor_name,
                        producer_model=producer_model,
                        consumer_models=consumer_models,
                        shape=shape,
                        storage_type=storage_type,
                        dtype=dtype
                    )
                )
            else:
                return self.bridge.share_tensor_between_models(
                    tensor_data=tensor_data,
                    tensor_name=tensor_name,
                    producer_model=producer_model,
                    consumer_models=consumer_models,
                    shape=shape,
                    storage_type=storage_type,
                    dtype=dtype
                )
            
        return {"success": False, "error": "share_tensor_between_models not available"}


# Example usage
def run_example():
    """Run a demonstration of the integrated resource pool with recovery."""
    logging.info("Starting ResourcePoolBridgeIntegrationWithRecovery example")
    
    # Create the integrated resource pool with recovery
    pool = ResourcePoolBridgeIntegrationWithRecovery(
        max_connections=2,
        adaptive_scaling=True,
        enable_recovery=True,
        max_retries=3,
        fallback_to_simulation=True
    )
    
    # Initialize 
    success = pool.initialize()
    if not success:
        logging.error("Failed to initialize resource pool")
        return
    
    try:
        # Load models
        logging.info("Loading text model (BERT)")
        text_model = pool.get_model(
            model_type="text",
            model_name="bert-base-uncased",
            hardware_preferences={
                "priority_list": ["webgpu", "webnn", "cpu"],
                "browser": "chrome"
            }
        )
        
        logging.info("Loading vision model (ViT)")
        vision_model = pool.get_model(
            model_type="vision",
            model_name="vit-base-patch16-224",
            hardware_preferences={
                "priority_list": ["webgpu", "cpu"],
                "browser": "chrome"
            }
        )
        
        logging.info("Loading audio model (Whisper)")
        audio_model = pool.get_model(
            model_type="audio",
            model_name="whisper-tiny",
            hardware_preferences={
                "priority_list": ["webgpu", "cpu"],
                "browser": "firefox"  # Firefox is preferred for audio
            }
        )
        
        # Generate sample inputs
        text_input = {
            "input_ids": [101, 2023, 2003, 1037, 3231, 102],
            "attention_mask": [1, 1, 1, 1, 1, 1]
        }
        
        vision_input = {
            "pixel_values": [[[0.5 for _ in range(224)] for _ in range(224)] for _ in range(3)]
        }
        
        audio_input = {
            "input_features": [[0.1 for _ in range(80)] for _ in range(3000)]
        }
        
        # Run inference with resilient error handling
        logging.info("Running inference on text model")
        text_result = text_model(text_input)
        logging.info(f"Text result status: {text_result.get('success', False)}")
        
        logging.info("Running inference on vision model")
        vision_result = vision_model(vision_input)
        logging.info(f"Vision result status: {vision_result.get('success', False)}")
        
        logging.info("Running inference on audio model")
        audio_result = audio_model(audio_input)
        logging.info(f"Audio result status: {audio_result.get('success', False)}")
        
        # Run concurrent inference
        logging.info("Running concurrent inference")
        model_inputs = [
            (text_model.model_id, text_input),
            (vision_model.model_id, vision_input),
            (audio_model.model_id, audio_input)
        ]
        
        concurrent_results = pool.execute_concurrent(model_inputs)
        logging.info(f"Concurrent results count: {len(concurrent_results)}")
        
        # Get metrics and recovery statistics
        metrics = pool.get_metrics()
        logging.info("Metrics and recovery statistics:")
        logging.info(f"  - Recovery enabled: {metrics.get('recovery_enabled', False)}")
        
        if 'recovery_stats' in metrics:
            logging.info(f"  - Recovery attempts: {metrics['recovery_stats'].get('total_recovery_attempts', 0)}")
        
        # Get health status
        health = pool.get_health_status()
        logging.info(f"Health status: {health.get('status', 'unknown')}")
        
    finally:
        # Close the pool
        pool.close()
        logging.info("ResourcePoolBridgeIntegrationWithRecovery example completed")


if __name__ == "__main__":
    # Configure detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Run the example
    run_example()