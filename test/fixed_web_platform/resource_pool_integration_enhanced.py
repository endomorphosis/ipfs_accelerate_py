#!/usr/bin/env python3
"""
Enhanced Resource Pool Integration for WebNN/WebGPU (May 2025)

This module provides an enhanced integration between IPFS acceleration and
the WebNN/WebGPU resource pool, with improved connection pooling, adaptive
scaling, and efficient cross-browser resource management.

Key features:
- Advanced connection pooling with adaptive scaling
- Efficient browser resource utilization for heterogeneous models
- Intelligent model routing based on browser capabilities
- Comprehensive health monitoring and recovery
- Performance telemetry and metrics collection
- Browser-specific optimizations for different model types
- DuckDB integration for result storage and analysis
"""

import os
import sys
import time
import json
import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple

# Import resource pool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_pool import get_global_resource_pool

# Import adaptive scaling and connection pool manager
try:
    from fixed_web_platform.adaptive_scaling import AdaptiveConnectionManager
    ADAPTIVE_SCALING_AVAILABLE = True
except ImportError:
    ADAPTIVE_SCALING_AVAILABLE = False

try:
    from fixed_web_platform.connection_pool_manager import ConnectionPoolManager
    CONNECTION_POOL_AVAILABLE = True
except ImportError:
    CONNECTION_POOL_AVAILABLE = False

# Import ResourcePoolBridgeIntegration (local import to avoid circular imports)
from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration, EnhancedWebModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedResourcePoolIntegration:
    """
    Enhanced integration between IPFS acceleration and WebNN/WebGPU resource pool.
    
    This class provides a unified interface for accessing WebNN and WebGPU
    acceleration through the resource pool, with optimized resource management,
    intelligent browser selection, and adaptive scaling.
    """
    
    def __init__(self, 
                 max_connections: int = 4,
                 min_connections: int = 1,
                 enable_gpu: bool = True, 
                 enable_cpu: bool = True,
                 headless: bool = True,
                 browser_preferences: Dict[str, str] = None,
                 adaptive_scaling: bool = True,
                 use_connection_pool: bool = True,
                 db_path: str = None):
        """
        Initialize enhanced resource pool integration.
        
        Args:
            max_connections: Maximum number of browser connections
            min_connections: Minimum number of browser connections
            enable_gpu: Whether to enable GPU acceleration
            enable_cpu: Whether to enable CPU acceleration
            headless: Whether to run browsers in headless mode
            browser_preferences: Dict mapping model families to preferred browsers
            adaptive_scaling: Whether to enable adaptive scaling
            use_connection_pool: Whether to use the enhanced connection pool
            db_path: Path to DuckDB database for storing results
        """
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu
        self.headless = headless
        self.db_path = db_path
        self.adaptive_scaling = adaptive_scaling
        self.use_connection_pool = use_connection_pool and CONNECTION_POOL_AVAILABLE
        
        # Browser preferences for routing models to appropriate browsers
        self.browser_preferences = browser_preferences or {
            'audio': 'firefox',  # Firefox has better compute shader performance for audio
            'vision': 'chrome',  # Chrome has good WebGPU support for vision models
            'text_embedding': 'edge',  # Edge has excellent WebNN support for text embeddings
            'text_generation': 'chrome',  # Chrome works well for text generation
            'multimodal': 'chrome'  # Chrome is good for multimodal models
        }
        
        # Get global resource pool
        self.resource_pool = get_global_resource_pool()
        
        # Core integration objects
        self.bridge_integration = None
        self.connection_pool = None
        
        # Loaded models tracking
        self.loaded_models = {}
        
        # Metrics collection
        self.metrics = {
            "model_load_time": {},
            "inference_time": {},
            "memory_usage": {},
            "throughput": {},
            "latency": {},
            "batch_size": {},
            "platform_distribution": {
                "webgpu": 0,
                "webnn": 0,
                "cpu": 0
            },
            "browser_distribution": {
                "chrome": 0,
                "firefox": 0,
                "edge": 0,
                "safari": 0
            }
        }
        
        # Create connection pool if available
        if self.use_connection_pool:
            try:
                self.connection_pool = ConnectionPoolManager(
                    min_connections=self.min_connections,
                    max_connections=self.max_connections,
                    browser_preferences=self.browser_preferences,
                    adaptive_scaling=self.adaptive_scaling,
                    headless=self.headless,
                    db_path=self.db_path
                )
                logger.info("Created enhanced connection pool manager")
            except Exception as e:
                logger.error(f"Error creating connection pool manager: {e}")
                self.connection_pool = None
                self.use_connection_pool = False
        
        # Create bridge integration (fallback if connection pool not available)
        self.bridge_integration = self._get_or_create_bridge_integration()
        
        logger.info("Enhanced Resource Pool Integration initialized successfully")
    
    def _get_or_create_bridge_integration(self) -> ResourcePoolBridgeIntegration:
        """
        Get or create resource pool bridge integration.
        
        Returns:
            ResourcePoolBridgeIntegration instance
        """
        # Check if integration already exists in resource pool
        integration = self.resource_pool.get_resource("web_platform_integration")
        
        if integration is None:
            # Create new integration
            integration = ResourcePoolBridgeIntegration(
                max_connections=self.max_connections,
                enable_gpu=self.enable_gpu,
                enable_cpu=self.enable_cpu,
                headless=self.headless,
                browser_preferences=self.browser_preferences,
                adaptive_scaling=self.adaptive_scaling,
                db_path=self.db_path
            )
            
            # Store in resource pool for reuse
            self.resource_pool.set_resource(
                "web_platform_integration", 
                integration
            )
        
        return integration
    
    async def initialize(self):
        """
        Initialize the resource pool integration.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            # Initialize connection pool if available
            if self.use_connection_pool and self.connection_pool:
                pool_init = await self.connection_pool.initialize()
                if not pool_init:
                    logger.warning("Failed to initialize connection pool, falling back to bridge integration")
                    self.use_connection_pool = False
            
            # Always initialize bridge integration (even as fallback)
            if hasattr(self.bridge_integration, 'initialize'):
                bridge_init = self.bridge_integration.initialize()
                if not bridge_init:
                    logger.warning("Failed to initialize bridge integration")
                    
                    # If both init failed, return failure
                    if not self.use_connection_pool:
                        return False
            
            logger.info("Enhanced Resource Pool Integration initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing Enhanced Resource Pool Integration: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def get_model(self, 
                       model_name: str, 
                       model_type: str = None,
                       platform: str = "webgpu", 
                       batch_size: int = 1,
                       quantization: Dict[str, Any] = None,
                       optimizations: Dict[str, bool] = None,
                       browser: str = None) -> Optional[EnhancedWebModel]:
        """
        Get a model with browser-based acceleration.
        
        This method provides an optimized model with the appropriate browser and
        hardware backend based on model type, with intelligent routing.
        
        Args:
            model_name: Name of the model to load
            model_type: Type of model (text, vision, audio, multimodal)
            platform: Platform to use (webgpu, webnn, or cpu)
            batch_size: Default batch size for model
            quantization: Quantization settings (bits, mixed_precision)
            optimizations: Optional optimizations to use
            browser: Specific browser to use (overrides preferences)
            
        Returns:
            EnhancedWebModel instance or None on failure
        """
        # Determine model type if not specified
        if model_type is None:
            model_type = self._infer_model_type(model_name)
        
        # Determine model family for optimal browser selection
        model_family = self._determine_model_family(model_type, model_name)
        
        # Determine browser based on model family if not specified
        if browser is None:
            browser = self.browser_preferences.get(model_family, 'chrome')
        
        # Set default optimizations based on model family
        default_optimizations = self._get_default_optimizations(model_family)
        if optimizations:
            default_optimizations.update(optimizations)
        
        # Create model key for caching
        model_key = f"{model_name}:{platform}:{batch_size}"
        if quantization:
            bits = quantization.get("bits", 16)
            mixed = quantization.get("mixed_precision", False)
            model_key += f":{bits}bit{'_mixed' if mixed else ''}"
        
        # Check if model is already loaded
        if model_key in self.loaded_models:
            logger.info(f"Reusing already loaded model: {model_key}")
            return self.loaded_models[model_key]
        
        # Create hardware preferences
        hardware_preferences = {
            'priority_list': [platform, 'cpu'],
            'model_family': model_family,
            'browser': browser,
            'quantization': quantization or {},
            'optimizations': default_optimizations
        }
        
        # Use connection pool if available
        if self.use_connection_pool and self.connection_pool:
            try:
                # Get connection from pool
                conn_id, conn_info = await self.connection_pool.get_connection(
                    model_type=model_type,
                    platform=platform,
                    browser=browser,
                    hardware_preferences=hardware_preferences
                )
                
                if conn_id is None:
                    logger.warning(f"Failed to get connection for model {model_name}, falling back to bridge integration")
                else:
                    # Add connection info to hardware preferences
                    hardware_preferences['connection_id'] = conn_id
                    hardware_preferences['connection_info'] = conn_info
                    
                    # Update metrics
                    self.metrics["browser_distribution"][browser] = self.metrics["browser_distribution"].get(browser, 0) + 1
                    self.metrics["platform_distribution"][platform] = self.metrics["platform_distribution"].get(platform, 0) + 1
            except Exception as e:
                logger.error(f"Error getting connection from pool: {e}")
                # Fall back to bridge integration
        
        # Get model from bridge integration
        start_time = time.time()
        try:
            web_model = self.bridge_integration.get_model(
                model_type=model_type,
                model_name=model_name,
                hardware_preferences=hardware_preferences
            )
            
            # Update metrics
            load_time = time.time() - start_time
            self.metrics["model_load_time"][model_key] = load_time
            
            # Cache model
            if web_model:
                self.loaded_models[model_key] = web_model
                logger.info(f"Loaded model {model_name} in {load_time:.2f}s")
                
                # Update browser and platform metrics
                actual_browser = getattr(web_model, 'browser', browser)
                actual_platform = getattr(web_model, 'platform', platform)
                
                self.metrics["browser_distribution"][actual_browser] = self.metrics["browser_distribution"].get(actual_browser, 0) + 1
                self.metrics["platform_distribution"][actual_platform] = self.metrics["platform_distribution"].get(actual_platform, 0) + 1
            
            return web_model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    async def execute_concurrent(self, models_and_inputs: List[Tuple[EnhancedWebModel, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Execute multiple models concurrently with efficient resource management.
        
        Args:
            models_and_inputs: List of (model, inputs) tuples
            
        Returns:
            List of execution results
        """
        if self.bridge_integration and hasattr(self.bridge_integration, 'execute_concurrent'):
            try:
                # Forward to bridge integration
                return await self.bridge_integration.execute_concurrent(models_and_inputs)
            except Exception as e:
                logger.error(f"Error in execute_concurrent: {e}")
                # Fall back to sequential execution
        
        # Sequential execution fallback
        results = []
        for model, inputs in models_and_inputs:
            if hasattr(model, '__call__'):
                try:
                    result = await model(inputs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error executing model: {e}")
                    results.append({"error": str(e), "success": False})
            else:
                logger.error(f"Invalid model object: {model}")
                results.append({"error": "Invalid model object", "success": False})
        
        return results
    
    def _infer_model_type(self, model_name: str) -> str:
        """
        Infer model type from model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Inferred model type
        """
        model_name = model_name.lower()
        
        # Check for common model type patterns
        if any(name in model_name for name in ['bert', 'roberta', 'distilbert', 'albert']):
            return 'text_embedding'
        elif any(name in model_name for name in ['gpt', 'llama', 'mistral', 'falcon', 't5', 'mpt']):
            return 'text_generation'
        elif any(name in model_name for name in ['vit', 'resnet', 'efficientnet', 'clip']):
            return 'vision'
        elif any(name in model_name for name in ['whisper', 'wav2vec', 'hubert', 'clap']):
            return 'audio'
        elif any(name in model_name for name in ['llava', 'blip', 'flava']):
            return 'multimodal'
        
        # Default to text_embedding as a safe fallback
        return 'text_embedding'
    
    def _determine_model_family(self, model_type: str, model_name: str) -> str:
        """
        Determine model family for optimal hardware selection.
        
        Args:
            model_type: Type of model (text_embedding, text_generation, etc.)
            model_name: Name of the model
            
        Returns:
            Model family for hardware selection
        """
        # Normalize model type
        model_type = model_type.lower()
        model_name = model_name.lower()
        
        # Standard model families
        if 'audio' in model_type or any(name in model_name for name in ['whisper', 'wav2vec', 'hubert', 'clap']):
            return 'audio'
        elif 'vision' in model_type or any(name in model_name for name in ['vit', 'resnet', 'efficientnet']):
            return 'vision'
        elif 'embedding' in model_type or any(name in model_name for name in ['bert', 'roberta', 'distilbert', 'albert']):
            return 'text_embedding'
        elif 'generation' in model_type or any(name in model_name for name in ['gpt', 'llama', 'mistral', 'falcon', 't5']):
            return 'text_generation'
        elif 'multimodal' in model_type or any(name in model_name for name in ['llava', 'blip', 'flava', 'clip']):
            return 'multimodal'
        
        # Default to text_embedding
        return 'text_embedding'
    
    def _get_default_optimizations(self, model_family: str) -> Dict[str, bool]:
        """
        Get default optimizations for a model family.
        
        Args:
            model_family: Model family (audio, vision, text_embedding, etc.)
            
        Returns:
            Dict with default optimizations
        """
        # Start with common optimizations
        optimizations = {
            'compute_shaders': False,
            'precompile_shaders': False,
            'parallel_loading': False
        }
        
        # Model-specific optimizations
        if model_family == 'audio':
            # Audio models benefit from compute shader optimization, especially in Firefox
            optimizations['compute_shaders'] = True
        elif model_family == 'vision':
            # Vision models benefit from shader precompilation
            optimizations['precompile_shaders'] = True
        elif model_family == 'multimodal':
            # Multimodal models benefit from parallel loading
            optimizations['parallel_loading'] = True
            optimizations['precompile_shaders'] = True
        
        return optimizations
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about resource pool usage.
        
        Returns:
            Dict with detailed metrics
        """
        metrics = self.metrics.copy()
        
        # Add connection pool metrics if available
        if self.use_connection_pool and self.connection_pool:
            try:
                pool_stats = self.connection_pool.get_stats()
                metrics['connection_pool'] = pool_stats
            except Exception as e:
                logger.error(f"Error getting connection pool stats: {e}")
        
        # Add bridge integration metrics
        if self.bridge_integration and hasattr(self.bridge_integration, 'get_stats'):
            try:
                bridge_stats = self.bridge_integration.get_stats()
                metrics['bridge_integration'] = bridge_stats
            except Exception as e:
                logger.error(f"Error getting bridge integration stats: {e}")
        
        # Add loaded models count
        metrics['loaded_models_count'] = len(self.loaded_models)
        
        return metrics
    
    async def close(self):
        """
        Close all connections and clean up resources.
        """
        # Close connection pool if available
        if self.use_connection_pool and self.connection_pool:
            try:
                await self.connection_pool.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down connection pool: {e}")
        
        # Close bridge integration
        if self.bridge_integration and hasattr(self.bridge_integration, 'close'):
            try:
                await self.bridge_integration.close()
            except Exception as e:
                logger.error(f"Error closing bridge integration: {e}")
        
        # Clear loaded models
        self.loaded_models.clear()
        
        logger.info("Enhanced Resource Pool Integration closed")
    
    def store_acceleration_result(self, result: Dict[str, Any]) -> bool:
        """
        Store acceleration result in database.
        
        Args:
            result: Acceleration result to store
            
        Returns:
            True if result was stored successfully, False otherwise
        """
        if self.bridge_integration and hasattr(self.bridge_integration, 'store_acceleration_result'):
            try:
                return self.bridge_integration.store_acceleration_result(result)
            except Exception as e:
                logger.error(f"Error storing acceleration result: {e}")
        
        return False

# For testing the module directly
if __name__ == "__main__":
    async def test_enhanced_integration():
        # Create enhanced integration
        integration = EnhancedResourcePoolIntegration(
            max_connections=4,
            min_connections=1,
            adaptive_scaling=True
        )
        
        # Initialize integration
        await integration.initialize()
        
        # Get model for text embedding
        bert_model = await integration.get_model(
            model_name="bert-base-uncased",
            model_type="text_embedding",
            platform="webgpu"
        )
        
        # Get model for vision
        vit_model = await integration.get_model(
            model_name="vit-base-patch16-224",
            model_type="vision",
            platform="webgpu"
        )
        
        # Get model for audio
        whisper_model = await integration.get_model(
            model_name="whisper-tiny",
            model_type="audio",
            platform="webgpu",
            browser="firefox"  # Explicitly request Firefox for audio
        )
        
        # Print metrics
        metrics = integration.get_metrics()
        print(f"Integration metrics: {json.dumps(metrics, indent=2)}")
        
        # Close integration
        await integration.close()
    
    # Run test
    asyncio.run(test_enhanced_integration())