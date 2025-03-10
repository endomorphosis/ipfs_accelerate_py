#!/usr/bin/env python3
"""
Resource Pool Integration for IPFS Acceleration with WebNN/WebGPU Integration (May 2025)

This module provides integration between the IPFS acceleration framework and
the WebNN/WebGPU resource pool bridge, enabling efficient resource sharing
and optimal utilization of browser-based hardware acceleration.

Key features:
- Seamless integration with IPFS acceleration framework
- Efficient browser resource sharing across test runs
- Automatic hardware selection for optimal model performance
- Connection pooling for browser instances
- Adaptive resource scaling based on workload
- Comprehensive monitoring and metrics collection

Usage:
    from fixed_web_platform.resource_pool_integration import IPFSAccelerateWebIntegration
    
    # Create integration instance
    integration = IPFSAccelerateWebIntegration()
    
    # Get a model with browser acceleration
    model = integration.get_model("bert-base-uncased", platform="webgpu")
    
    # Run inference
    result = model.run_inference(inputs)
    
    # Get performance metrics
    metrics = integration.get_metrics()
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Import resource pool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_pool import get_global_resource_pool

# Import resource pool bridge (local import to avoid circular imports)
from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration, EnhancedWebModel

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IPFSAccelerateWebIntegration:
    """
    Integration between IPFS acceleration and WebNN/WebGPU resource pool.
    
    This class provides a unified interface for accessing WebNN and WebGPU
    acceleration through the resource pool, with optimized resource management
    and efficient browser connection sharing.
    
    Key features added in May 2025:
    - Enhanced connection pooling with automatic scaling
    - Model-specific browser selection for optimal performance
    - Smart fallback system when resources are constrained
    - Comprehensive telemetry and performance monitoring
    - Concurrent model execution and resource sharing
    - Real-time performance optimization based on workload
    """
    
    def __init__(self, max_connections: int = 4, 
                enable_gpu: bool = True, enable_cpu: bool = True,
                browser_preferences: Dict[str, str] = None,
                adaptive_scaling: bool = True,
                enable_telemetry: bool = True,
                db_path: str = None,
                smart_fallback: bool = True):
        """
        Initialize IPFS acceleration integration with web platform.
        
        Args:
            max_connections: Maximum number of browser connections
            enable_gpu: Whether to enable GPU acceleration
            enable_cpu: Whether to enable CPU acceleration
            browser_preferences: Dict mapping model families to preferred browsers
            adaptive_scaling: Whether to enable adaptive resource scaling
            enable_telemetry: Whether to collect detailed performance telemetry
            db_path: Path to DuckDB database for result storage
            smart_fallback: Whether to enable smart fallback when resources are constrained
        """
        self.resource_pool = get_global_resource_pool()
        self.db_path = db_path
        self.enable_telemetry = enable_telemetry
        self.smart_fallback = smart_fallback
        
        # Enhanced connection management
        self.bridge_integration = self._get_or_create_bridge_integration(
            max_connections=max_connections,
            enable_gpu=enable_gpu,
            enable_cpu=enable_cpu,
            browser_preferences=browser_preferences,
            adaptive_scaling=adaptive_scaling,
            db_path=db_path
        )
        
        # Enhanced metrics collection
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
            },
            "connection_metrics": {
                "peak_connections": 0,
                "connection_utilization": 0.0,
                "connection_wait_time": 0.0,
                "connection_errors": 0
            },
            "performance_optimization": {
                "shader_precompilation_savings_ms": 0.0,
                "compute_shader_speedup_percent": 0.0,
                "parallel_loading_savings_ms": 0.0
            },
            "quantization_metrics": {
                "memory_savings_percent": 0.0,
                "accuracy_impact_percent": 0.0,
                "latency_impact_percent": 0.0
            }
        }
        
        # Track loaded models with enhanced metadata
        self.loaded_models = {}
        
        # Store connection metrics for monitoring
        self.connection_history = []
        
        # Performance optimization tracker
        self.optimization_history = []
        
        # Initialize database connection if path provided
        self.db_connection = None
        if db_path:
            self._initialize_database_connection()
        
        logger.info(f"IPFSAccelerateWebIntegration initialized successfully with {max_connections} connections and {'enabled' if adaptive_scaling else 'disabled'} adaptive scaling")
        
    def _initialize_database_connection(self):
        """Initialize database connection for storing performance metrics"""
        if not self.db_path:
            return
            
        try:
            import duckdb
            
            # Connect to database
            self.db_connection = duckdb.connect(self.db_path)
            
            # Create required tables if they don't exist
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS web_resource_pool_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                model_name VARCHAR,
                model_type VARCHAR,
                platform VARCHAR,
                browser_name VARCHAR,
                connection_id VARCHAR,
                is_real_implementation BOOLEAN,
                latency_ms FLOAT,
                throughput_items_per_sec FLOAT,
                memory_usage_mb FLOAT,
                batch_size INTEGER,
                compute_shader_optimized BOOLEAN,
                precompile_shaders BOOLEAN,
                parallel_loading BOOLEAN,
                quantization_bits INTEGER,
                mixed_precision BOOLEAN,
                ipfs_accelerated BOOLEAN,
                adapter_info JSON,
                telemetry JSON
            )
            """)
            
            # Create index for faster queries
            self.db_connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_web_resource_pool_model 
            ON web_resource_pool_metrics(model_name)
            """)
            
            # Create connection pool metrics table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS web_connection_pool_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                pool_id VARCHAR,
                max_connections INTEGER,
                active_connections INTEGER,
                connection_utilization FLOAT,
                connection_wait_time_ms FLOAT,
                connection_errors INTEGER,
                browser_distribution JSON,
                platform_distribution JSON,
                memory_usage_mb FLOAT,
                performance_metrics JSON
            )
            """)
            
            logger.info(f"Database connection initialized: {self.db_path}")
            return True
            
        except ImportError:
            logger.warning("DuckDB not installed, database functionality will be disabled")
            self.db_connection = None
            return False
            
        except Exception as e:
            logger.error(f"Error initializing database connection: {e}")
            self.db_connection = None
            return False
    
    def _get_or_create_bridge_integration(self, max_connections=4, 
                                         enable_gpu=True, enable_cpu=True,
                                         browser_preferences=None,
                                         adaptive_scaling=True,
                                         db_path=None) -> ResourcePoolBridgeIntegration:
        """
        Get or create resource pool bridge integration with enhanced features.
        
        Args:
            max_connections: Maximum number of browser connections
            enable_gpu: Whether to enable GPU acceleration
            enable_cpu: Whether to enable CPU acceleration
            browser_preferences: Dict mapping model families to preferred browsers
            adaptive_scaling: Whether to enable adaptive resource scaling
            db_path: Path to DuckDB database for result storage
            
        Returns:
            ResourcePoolBridgeIntegration instance
        """
        # Check if integration already exists in resource pool
        integration = self.resource_pool.get_resource("web_platform_integration")
        
        if integration is None:
            # Create new integration with enhanced capabilities
            integration = ResourcePoolBridgeIntegration(
                max_connections=max_connections,
                enable_gpu=enable_gpu,
                enable_cpu=enable_cpu,
                headless=True,  # Always use headless for IPFS acceleration
                browser_preferences=browser_preferences or {
                    'audio': 'firefox',  # Firefox has better compute shader performance for audio
                    'vision': 'chrome',  # Chrome has good WebGPU support for vision models
                    'text_embedding': 'edge',  # Edge has excellent WebNN support for text embeddings
                    'text_generation': 'chrome',
                    'multimodal': 'chrome'
                },
                adaptive_scaling=adaptive_scaling,
                db_path=db_path,
                enable_telemetry=self.enable_telemetry,
                smart_fallback=self.smart_fallback,
                enable_heartbeat=True,  # Enable connection health monitoring
                optimize_shader_precompilation=True,  # Always enable shader precompilation for better performance
                enable_cross_browser_fallback=True,  # Enable cross-browser fallback for optimal model placement
                connection_monitoring=True,  # Monitor connection health and performance
                resource_utilization_tracking=True,  # Track resource utilization for better scaling
                enable_distributed_inference=True  # Enable distributed inference across multiple browsers
            )
            
            # Initialize integration with enhanced error handling
            try:
                initialization_success = integration.initialize()
                if not initialization_success:
                    logger.warning("Integration initialization returned False, using degraded capabilities")
                    # Set a flag to use degraded capabilities
                    integration.using_degraded_capabilities = True
            except Exception as e:
                logger.error(f"Error initializing integration: {e}")
                # Create a fallback integration with minimal capabilities
                integration = ResourcePoolBridgeIntegration(
                    max_connections=1,  # Reduced connections
                    enable_gpu=enable_gpu,
                    enable_cpu=True,  # Always enable CPU as fallback
                    headless=True,
                    browser_preferences={'default': 'chrome'},  # Simplified browser selection
                    adaptive_scaling=False  # Disable adaptive scaling in fallback mode
                )
                integration.initialize()
                integration.using_degraded_capabilities = True
            
            # Store in resource pool for reuse, with additional metadata
            integration.created_time = time.time()
            integration.telemetry_enabled = self.enable_telemetry
            integration.connection_stats = {
                "created_connections": 0,
                "successful_connections": 0,
                "failed_connections": 0,
                "peak_memory_usage_mb": 0,
                "browser_distribution": {
                    "chrome": 0,
                    "firefox": 0,
                    "edge": 0,
                    "safari": 0
                }
            }
            
            # Store in resource pool for reuse
            self.resource_pool.get_resource(
                "web_platform_integration", 
                constructor=lambda: integration
            )
        else:
            # If integration already exists, update its configuration
            if hasattr(integration, 'update_configuration'):
                integration.update_configuration(
                    max_connections=max_connections,
                    enable_gpu=enable_gpu,
                    enable_cpu=enable_cpu,
                    browser_preferences=browser_preferences,
                    adaptive_scaling=adaptive_scaling,
                    db_path=db_path,
                    enable_telemetry=self.enable_telemetry
                )
            logger.info("Using existing ResourcePoolBridgeIntegration instance from resource pool")
        
        return integration
    
    def get_model(self, model_name: str, model_type: str = None, 
                 platform: str = "webgpu", batch_size: int = 1,
                 quantization: Dict[str, Any] = None,
                 optimizations: Dict[str, bool] = None,
                 priority: int = 0,
                 fallback_platforms: List[str] = None,
                 timeout: float = None,
                 cache_model: bool = True,
                 force_reload: bool = False) -> EnhancedWebModel:
        """
        Get a model with browser-based acceleration and enhanced resource management.
        
        Args:
            model_name: Name of the model to load
            model_type: Type of model (text, vision, audio, multimodal)
            platform: Primary platform to use (webgpu, webnn, or cpu)
            batch_size: Default batch size for model
            quantization: Quantization settings (bits, mixed_precision)
            optimizations: Optional optimizations to use
            priority: Priority of this model (0-10, higher values have higher priority)
            fallback_platforms: List of platforms to try if primary platform fails
            timeout: Timeout for model loading in seconds
            cache_model: Whether to cache the model for reuse
            force_reload: Whether to force model reload even if cached
            
        Returns:
            EnhancedWebModel instance
        """
        # Track request start time for performance monitoring
        request_start_time = time.time()
        
        # Determine model type if not specified
        if model_type is None:
            model_type = self._infer_model_type(model_name)
        
        # Determine model family for optimal browser selection
        model_family = self._determine_model_family(model_type, model_name)
        
        # Set default optimizations based on model family with enhancements
        default_optimizations = self._get_default_optimizations(model_family)
        # Apply user-specified optimizations with higher priority
        if optimizations:
            default_optimizations.update(optimizations)
        
        # Enhanced model key for more precise caching
        model_key = f"{model_name}:{platform}:{batch_size}"
        if quantization:
            bits = quantization.get("bits", 16)
            mixed = quantization.get("mixed_precision", False)
            model_key += f":{bits}bit{'_mixed' if mixed else ''}"
        
        # Add optimization flags to model key for more specific caching
        opt_flags = []
        for opt_name, opt_value in default_optimizations.items():
            if opt_value:
                opt_flags.append(opt_name)
        if opt_flags:
            model_key += f":{'_'.join(opt_flags)}"
        
        # Enhanced caching with validity checks
        if not force_reload and model_key in self.loaded_models:
            cached_model = self.loaded_models[model_key]
            
            # Verify model is still valid (hasn't been closed/destroyed)
            if hasattr(cached_model, "is_valid") and cached_model.is_valid():
                if hasattr(cached_model, "last_used_time"):
                    cached_model.last_used_time = time.time()
                logger.info(f"Reusing cached model: {model_key}")
                
                # Update cache hit metrics
                if "cache_hits" not in self.metrics:
                    self.metrics["cache_hits"] = {}
                if model_key not in self.metrics["cache_hits"]:
                    self.metrics["cache_hits"][model_key] = 0
                self.metrics["cache_hits"][model_key] += 1
                
                # Update model cache statistics
                if hasattr(cached_model, "update_cache_stats"):
                    cached_model.update_cache_stats()
                
                return cached_model
            else:
                # Cached model is invalid, remove it
                logger.warning(f"Cached model {model_key} is invalid, removing from cache")
                del self.loaded_models[model_key]
        
        # Create enhanced hardware preferences with automatic fallback
        hardware_preferences = {
            'priority_list': [platform] + (fallback_platforms or ['webnn', 'cpu']),
            'model_family': model_family,
            'quantization': quantization or {},
            'optimizations': default_optimizations,
            'priority': priority,
            'timeout': timeout,
            'enable_telemetry': self.enable_telemetry,
            'smart_fallback': self.smart_fallback
        }
        
        # Record attempt metrics
        if "model_load_attempts" not in self.metrics:
            self.metrics["model_load_attempts"] = {}
        if model_key not in self.metrics["model_load_attempts"]:
            self.metrics["model_load_attempts"][model_key] = 0
        self.metrics["model_load_attempts"][model_key] += 1
        
        # Add cross-browser fallback if smart_fallback is enabled
        if self.smart_fallback:
            if "fallback_browsers" not in hardware_preferences:
                hardware_preferences["fallback_browsers"] = {}
            
            # Smart fallback browser selection based on model family
            if model_family == "audio":
                hardware_preferences["fallback_browsers"]["preferred"] = "firefox"
                hardware_preferences["fallback_browsers"]["alternate"] = ["chrome", "edge"]
            elif model_family == "vision":
                hardware_preferences["fallback_browsers"]["preferred"] = "chrome"
                hardware_preferences["fallback_browsers"]["alternate"] = ["edge", "firefox"]
            elif model_family == "text_embedding":
                hardware_preferences["fallback_browsers"]["preferred"] = "edge"
                hardware_preferences["fallback_browsers"]["alternate"] = ["chrome", "firefox"]
            elif model_family == "text_generation":
                hardware_preferences["fallback_browsers"]["preferred"] = "chrome"
                hardware_preferences["fallback_browsers"]["alternate"] = ["edge", "firefox"]
        
        # Get model from bridge integration with enhanced error handling
        start_time = time.time()
        web_model = None
        error_occurred = False
        error_details = None
        
        try:
            web_model = self.bridge_integration.get_model(
                model_type=model_type,
                model_name=model_name,
                hardware_preferences=hardware_preferences
            )
        except Exception as e:
            error_occurred = True
            error_details = str(e)
            logger.error(f"Error getting model {model_name}: {e}")
            
            # Attempt to recover with fallback options if smart_fallback is enabled
            if self.smart_fallback:
                logger.info(f"Attempting to recover with fallback options for {model_name}")
                try:
                    # Try with CPU platform as a last resort
                    fallback_prefs = hardware_preferences.copy()
                    fallback_prefs["priority_list"] = ["cpu"]
                    web_model = self.bridge_integration.get_model(
                        model_type=model_type,
                        model_name=model_name,
                        hardware_preferences=fallback_prefs
                    )
                    if web_model:
                        logger.info(f"Successfully recovered {model_name} with CPU fallback")
                except Exception as fallback_error:
                    logger.error(f"Fallback recovery failed for {model_name}: {fallback_error}")
        
        load_time = time.time() - start_time
        
        # Store comprehensive metrics
        self.metrics["model_load_time"][model_key] = load_time
        
        # Track latency distribution for performance analysis
        if "load_time_distribution" not in self.metrics:
            self.metrics["load_time_distribution"] = {
                "0-1s": 0,
                "1-2s": 0,
                "2-5s": 0,
                "5-10s": 0,
                "10s+": 0
            }
        
        # Update latency distribution
        if load_time < 1.0:
            self.metrics["load_time_distribution"]["0-1s"] += 1
        elif load_time < 2.0:
            self.metrics["load_time_distribution"]["1-2s"] += 1
        elif load_time < 5.0:
            self.metrics["load_time_distribution"]["2-5s"] += 1
        elif load_time < 10.0:
            self.metrics["load_time_distribution"]["5-10s"] += 1
        else:
            self.metrics["load_time_distribution"]["10s+"] += 1
        
        # Check if model loading was successful
        if web_model is None:
            logger.error(f"Failed to load model {model_name}")
            
            # Record error in metrics
            if "model_load_errors" not in self.metrics:
                self.metrics["model_load_errors"] = {}
            if model_key not in self.metrics["model_load_errors"]:
                self.metrics["model_load_errors"][model_key] = []
            
            self.metrics["model_load_errors"][model_key].append({
                "timestamp": time.time(),
                "error": error_details or "Unknown error",
                "attempted_platforms": hardware_preferences.get("priority_list", []),
                "load_duration": load_time
            })
            
            # Create a mock model as fallback if all else fails and smart_fallback is enabled
            if self.smart_fallback:
                logger.warning(f"Creating mock model for {model_name} as ultimate fallback")
                from fixed_web_platform.resource_pool_bridge import MockFallbackModel
                web_model = MockFallbackModel(model_name, model_type, model_family)
            else:
                # Return None if no fallback is enabled
                return None
        
        # Update platform distribution metrics 
        if hasattr(web_model, "platform"):
            actual_platform = web_model.platform
            if actual_platform in self.metrics["platform_distribution"]:
                self.metrics["platform_distribution"][actual_platform] += 1
        else:
            # Default to requested platform if not available on model
            self.metrics["platform_distribution"][platform] += 1
        
        # Update browser distribution metrics
        if hasattr(web_model, "browser"):
            browser = web_model.browser
            if browser in self.metrics["browser_distribution"]:
                self.metrics["browser_distribution"][browser] += 1
        
        # Set batch size if specified
        if batch_size > 1 and hasattr(web_model, "set_max_batch_size"):
            web_model.set_max_batch_size(batch_size)
        
        # Add telemetry attributes to model for enhanced monitoring
        if self.enable_telemetry:
            if hasattr(web_model, "set_telemetry_attributes"):
                telemetry_data = {
                    "load_time": load_time,
                    "requested_platform": platform,
                    "actual_platform": getattr(web_model, "platform", platform),
                    "model_family": model_family,
                    "optimizations": default_optimizations,
                    "quantization": quantization,
                    "batch_size": batch_size,
                    "is_fallback": error_occurred,
                    "is_real_implementation": getattr(web_model, "is_real_implementation", False),
                    "browser": getattr(web_model, "browser", "unknown")
                }
                web_model.set_telemetry_attributes(telemetry_data)
        
        # Cache model if requested
        if cache_model:
            self.loaded_models[model_key] = web_model
            # Set additional metadata for cache management
            if hasattr(web_model, "set_cache_metadata"):
                web_model.set_cache_metadata({
                    "load_time": load_time,
                    "cached_at": time.time(),
                    "last_used_time": time.time(),
                    "model_key": model_key,
                    "usage_count": 0
                })
        
        # Store to database if enabled
        if self.db_connection and hasattr(web_model, "get_model_info"):
            self._store_model_metrics(web_model, load_time, model_key)
        
        # Report comprehensive load metrics
        if hasattr(web_model, "get_model_info"):
            model_info = web_model.get_model_info()
            platform_info = model_info.get("platform", "unknown")
            browser_info = model_info.get("browser", "unknown")
            implementation_type = "real" if model_info.get("is_real_implementation", False) else "simulated"
            logger.info(f"Model {model_name} loaded with {platform_info} acceleration on {browser_info} ({implementation_type}) in {load_time:.2f}s")
        else:
            logger.info(f"Model {model_name} loaded with {platform} acceleration in {load_time:.2f}s")
        
        return web_model
        
    def _store_model_metrics(self, model, load_time, model_key):
        """Store model metrics in database"""
        if not self.db_connection:
            return
        
        try:
            model_info = model.get_model_info()
            
            # Prepare data for storage
            now = time.time()
            model_name = model_info.get("model_name", "unknown")
            model_type = model_info.get("model_type", "unknown")
            platform = model_info.get("platform", "unknown")
            browser = model_info.get("browser", "unknown")
            connection_id = model_info.get("connection_id", "unknown")
            is_real = model_info.get("is_real_implementation", False)
            
            # Get performance metrics if available
            latency = model_info.get("latency_ms", 0.0)
            throughput = model_info.get("throughput_items_per_sec", 0.0)
            memory_usage = model_info.get("memory_usage_mb", 0.0)
            batch_size = model_info.get("batch_size", 1)
            
            # Get optimization flags
            compute_shaders = model_info.get("compute_shader_optimized", False)
            precompile_shaders = model_info.get("precompile_shaders", False)
            parallel_loading = model_info.get("parallel_loading", False)
            
            # Get quantization settings
            quantization_bits = model_info.get("quantization_bits", 16)
            mixed_precision = model_info.get("mixed_precision", False)
            ipfs_accelerated = model_info.get("ipfs_accelerated", False)
            
            # Get adapter info if available
            adapter_info = model_info.get("adapter_info", {})
            adapter_json = json.dumps(adapter_info) if adapter_info else "{}"
            
            # Get telemetry data
            telemetry = {
                "load_time_seconds": load_time,
                "model_key": model_key,
                "error_occurred": model_info.get("error_occurred", False),
                "is_fallback": model_info.get("is_fallback", False)
            }
            telemetry_json = json.dumps(telemetry)
            
            # Insert into database
            self.db_connection.execute("""
            INSERT INTO web_resource_pool_metrics (
                timestamp, model_name, model_type, platform, browser_name, connection_id,
                is_real_implementation, latency_ms, throughput_items_per_sec, memory_usage_mb,
                batch_size, compute_shader_optimized, precompile_shaders, parallel_loading,
                quantization_bits, mixed_precision, ipfs_accelerated, adapter_info, telemetry
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, 
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                model_name, model_type, platform, browser, connection_id,
                is_real, latency, throughput, memory_usage, batch_size,
                compute_shaders, precompile_shaders, parallel_loading,
                quantization_bits, mixed_precision, ipfs_accelerated,
                adapter_json, telemetry_json
            ])
            
        except Exception as e:
            logger.error(f"Error storing model metrics in database: {e}")
            # Continue execution despite database error
    
    def _infer_model_type(self, model_name: str) -> str:
        """
        Infer model type from model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Inferred model type
        """
        model_name_lower = model_name.lower()
        
        # Text embedding models
        if any(name in model_name_lower for name in ["bert", "roberta", "albert", "distilbert", "mpnet"]):
            return "text_embedding"
        
        # Text generation models
        elif any(name in model_name_lower for name in ["gpt", "t5", "llama", "opt", "bloom", "mistral", "falcon"]):
            return "text_generation"
        
        # Vision models
        elif any(name in model_name_lower for name in ["vit", "resnet", "efficientnet", "beit", "deit", "convnext"]):
            return "vision"
        
        # Audio models
        elif any(name in model_name_lower for name in ["whisper", "wav2vec", "hubert", "mms", "clap"]):
            return "audio"
        
        # Multimodal models
        elif any(name in model_name_lower for name in ["clip", "llava", "blip", "xclip", "flamingo"]):
            return "multimodal"
        
        # Default to text_embedding if unknown
        return "text_embedding"
    
    def _determine_model_family(self, model_type: str, model_name: str) -> str:
        """
        Determine model family for optimal hardware assignment.
        
        Args:
            model_type: Type of model
            model_name: Name of the model
            
        Returns:
            Model family
        """
        # Check if model_family_classifier is available for better classification
        try:
            classifier_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "model_family_classifier.py")
            
            if os.path.exists(classifier_path):
                sys.path.append(os.path.dirname(classifier_path))
                from model_family_classifier import classify_model
                
                model_info = classify_model(model_name=model_name)
                family = model_info.get("family")
                
                if family:
                    logger.debug(f"Model {model_name} classified as {family} by model_family_classifier")
                    return family
        except (ImportError, Exception) as e:
            logger.debug(f"Error using model_family_classifier: {e}")
        
        # Map model type to family if classifier not available
        type_to_family = {
            "text_embedding": "text_embedding",
            "text_generation": "text_generation",
            "vision": "vision",
            "audio": "audio",
            "multimodal": "multimodal"
        }
        
        return type_to_family.get(model_type, model_type)
    
    def _get_default_optimizations(self, model_family: str) -> Dict[str, bool]:
        """
        Get default optimizations for a model family.
        
        Args:
            model_family: Model family
            
        Returns:
            Dict of default optimizations
        """
        # Default optimizations for all families
        default = {
            "compute_shaders": False,
            "precompile_shaders": True,  # Always beneficial
            "parallel_loading": False
        }
        
        if model_family == "audio":
            # Enable compute shader optimization for audio models (especially on Firefox)
            default["compute_shaders"] = True
        
        elif model_family == "multimodal":
            # Enable parallel loading for multimodal models
            default["parallel_loading"] = True
        
        return default
    
    def run_inference(self, model, inputs, batch_size=None, 
                      timeout=None, track_metrics=True,
                      store_in_db=True, telemetry_data=None):
        """
        Run inference with a model with enhanced monitoring and error handling.
        
        Args:
            model: Model to use for inference
            inputs: Input data for inference
            batch_size: Optional batch size override
            timeout: Optional timeout in seconds for inference
            track_metrics: Whether to track performance metrics
            store_in_db: Whether to store inference result in database
            telemetry_data: Additional telemetry data to record
            
        Returns:
            Inference results
        """
        # Track inference time
        start_time = time.time()
        inference_start_timestamp = datetime.now().isoformat()
        
        # Get model key for metrics tracking
        model_key = model.model_id if hasattr(model, "model_id") else str(id(model))
        
        # Prepare execution context
        error_occurred = False
        error_message = None
        result = None
        
        # Apply batch size override if specified and supported
        original_batch_size = None
        if batch_size is not None and hasattr(model, "set_max_batch_size"):
            # Remember original batch size for restoring later
            if hasattr(model, "get_max_batch_size"):
                original_batch_size = model.get_max_batch_size()
            model.set_max_batch_size(batch_size)
        
        # Run inference with timeout protection if specified
        try:
            if timeout is not None:
                # Use asyncio for timeout protection
                import asyncio
                loop = asyncio.get_event_loop()
                
                # Create a task that runs the model inference
                async def run_with_timeout():
                    # Run inference in a thread to avoid blocking
                    return await loop.run_in_executor(None, lambda: model(inputs))
                
                # Run the task with timeout
                result = asyncio.run(asyncio.wait_for(run_with_timeout(), timeout))
            else:
                # Run inference directly
                result = model(inputs)
                
        except TimeoutError:
            error_occurred = True
            error_message = f"Inference timed out after {timeout} seconds"
            logger.error(error_message)
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            logger.error(f"Error during inference with model {model_key}: {e}")
        finally:
            # Restore original batch size if it was overridden
            if original_batch_size is not None and hasattr(model, "set_max_batch_size"):
                model.set_max_batch_size(original_batch_size)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Update metrics if tracking is enabled
        if track_metrics:
            # Initialize metrics containers if needed
            if "inference_attempts" not in self.metrics:
                self.metrics["inference_attempts"] = {}
            if "inference_success" not in self.metrics:
                self.metrics["inference_success"] = {}
            if "inference_errors" not in self.metrics:
                self.metrics["inference_errors"] = {}
                
            # Track attempt
            if model_key not in self.metrics["inference_attempts"]:
                self.metrics["inference_attempts"][model_key] = 0
            self.metrics["inference_attempts"][model_key] += 1
            
            # Track success/error
            if not error_occurred:
                if model_key not in self.metrics["inference_success"]:
                    self.metrics["inference_success"][model_key] = 0
                self.metrics["inference_success"][model_key] += 1
            else:
                if model_key not in self.metrics["inference_errors"]:
                    self.metrics["inference_errors"][model_key] = []
                self.metrics["inference_errors"][model_key].append({
                    "timestamp": time.time(),
                    "error": error_message,
                    "inference_time": inference_time
                })
            
            # Track inference time history
            if model_key not in self.metrics["inference_time"]:
                self.metrics["inference_time"][model_key] = []
            
            self.metrics["inference_time"][model_key].append(inference_time)
            
            # Calculate running average of metrics (keep last 10 measurements)
            if len(self.metrics["inference_time"][model_key]) > 10:
                self.metrics["inference_time"][model_key] = self.metrics["inference_time"][model_key][-10:]
            
            # Update performance metrics if available
            if not error_occurred and hasattr(model, "get_performance_metrics"):
                try:
                    perf_metrics = model.get_performance_metrics()
                    if "stats" in perf_metrics:
                        stats = perf_metrics["stats"]
                        self.metrics["throughput"][model_key] = stats.get("throughput", 0)
                        self.metrics["latency"][model_key] = stats.get("avg_latency", 0)
                        self.metrics["batch_size"][model_key] = stats.get("batch_sizes", {})
                    
                    if "memory_usage" in perf_metrics:
                        self.metrics["memory_usage"][model_key] = perf_metrics["memory_usage"]
                        
                    # Track optimization metrics if available
                    if "optimization_metrics" in perf_metrics:
                        opt_metrics = perf_metrics["optimization_metrics"]
                        if "shader_precompilation_savings_ms" in opt_metrics:
                            self.metrics["performance_optimization"]["shader_precompilation_savings_ms"] += opt_metrics["shader_precompilation_savings_ms"]
                        if "compute_shader_speedup_percent" in opt_metrics:
                            # Track as running average
                            current = self.metrics["performance_optimization"]["compute_shader_speedup_percent"]
                            new_value = opt_metrics["compute_shader_speedup_percent"]
                            if current == 0:
                                self.metrics["performance_optimization"]["compute_shader_speedup_percent"] = new_value
                            else:
                                self.metrics["performance_optimization"]["compute_shader_speedup_percent"] = (current * 0.7) + (new_value * 0.3)
                except Exception as metrics_error:
                    logger.warning(f"Error updating performance metrics: {metrics_error}")
        
        # Store in database if requested and available
        if store_in_db and self.db_connection and not error_occurred:
            self._store_inference_metrics(model, result, inference_time, inputs, telemetry_data)
        
        # Return the inference result (or None if error occurred)
        return result
    
    def _store_inference_metrics(self, model, result, inference_time, inputs, telemetry_data=None):
        """Store inference metrics in database"""
        if not self.db_connection:
            return
            
        try:
            # Only store if model has get_model_info method
            if not hasattr(model, "get_model_info"):
                return
                
            model_info = model.get_model_info()
            
            # Create inference record table if it doesn't exist
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS web_inference_records (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                model_name VARCHAR,
                model_type VARCHAR,
                platform VARCHAR,
                browser_name VARCHAR,
                inference_time_ms FLOAT,
                is_real_implementation BOOLEAN,
                input_shape VARCHAR,
                output_shape VARCHAR,
                batch_size INTEGER,
                memory_usage_mb FLOAT,
                compute_shader_optimized BOOLEAN,
                precompile_shaders BOOLEAN,
                parallel_loading BOOLEAN,
                telemetry JSON
            )
            """)
            
            # Prepare data
            model_name = model_info.get("model_name", "unknown")
            model_type = model_info.get("model_type", "unknown")
            platform = model_info.get("platform", "unknown")
            browser = model_info.get("browser", "unknown")
            is_real = model_info.get("is_real_implementation", False)
            
            # Infer input and output shapes
            input_shape = "unknown"
            output_shape = "unknown"
            
            if isinstance(inputs, dict):
                input_shapes = {}
                for key, value in inputs.items():
                    if hasattr(value, "shape"):
                        input_shapes[key] = list(value.shape)
                    elif isinstance(value, list):
                        input_shapes[key] = [len(value)]
                input_shape = json.dumps(input_shapes)
            
            if isinstance(result, dict):
                output_shapes = {}
                for key, value in result.items():
                    if hasattr(value, "shape"):
                        output_shapes[key] = list(value.shape)
                    elif isinstance(value, list):
                        output_shapes[key] = [len(value)]
                output_shape = json.dumps(output_shapes)
            
            # Get batch size
            batch_size = model_info.get("batch_size", 1)
            
            # Get memory usage
            memory_usage = model_info.get("memory_usage_mb", 0.0)
            
            # Get optimization flags
            compute_shaders = model_info.get("compute_shader_optimized", False)
            precompile_shaders = model_info.get("precompile_shaders", False)
            parallel_loading = model_info.get("parallel_loading", False)
            
            # Combine with additional telemetry
            combined_telemetry = {
                "inference_time_ms": inference_time * 1000.0,
                "model_key": model_info.get("model_id", "unknown"),
                "is_fallback": model_info.get("is_fallback", False)
            }
            
            # Add user-provided telemetry
            if telemetry_data:
                combined_telemetry.update(telemetry_data)
                
            telemetry_json = json.dumps(combined_telemetry)
            
            # Insert into database
            self.db_connection.execute("""
            INSERT INTO web_inference_records (
                timestamp, model_name, model_type, platform, browser_name,
                inference_time_ms, is_real_implementation, input_shape, output_shape,
                batch_size, memory_usage_mb, compute_shader_optimized,
                precompile_shaders, parallel_loading, telemetry
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                model_name, model_type, platform, browser,
                inference_time * 1000.0, is_real, input_shape, output_shape,
                batch_size, memory_usage, compute_shaders,
                precompile_shaders, parallel_loading, telemetry_json
            ])
            
        except Exception as e:
            logger.error(f"Error storing inference metrics in database: {e}")
            # Continue execution despite database error
    
    def run_parallel_inference(self, model_data_pairs, batch_size=None, 
                               timeout=None, track_metrics=True, 
                               store_in_db=True, telemetry_data=None, 
                               distributed=False, priority=0):
        """
        Run inference on multiple models in parallel with enhanced capabilities.
        
        Args:
            model_data_pairs: List of (model, input_data) pairs
            batch_size: Optional batch size override
            timeout: Optional timeout in seconds for inference operations
            track_metrics: Whether to track performance metrics
            store_in_db: Whether to store inference result in database
            telemetry_data: Additional telemetry data to record
            distributed: Whether to distribute inference across multiple browser connections
            priority: Priority for execution scheduling (0-10, higher is higher priority)
            
        Returns:
            List of inference results in the same order
        """
        if not model_data_pairs:
            return []
        
        # Track operation start time for performance monitoring
        operation_start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        # Extract EnhancedWebModels for parallel execution
        web_models = []
        other_models = []
        other_inputs = []
        
        # Classify models by type for optimal execution
        for model, inputs in model_data_pairs:
            if hasattr(model, 'is_enhanced_web_model') and model.is_enhanced_web_model:
                web_models.append((model, inputs))
            else:
                other_models.append(model)
                other_inputs.append(inputs)
        
        # Initialize results list
        results = []
        error_occurred = False
        
        # Process enhanced web models using concurrent execution
        if web_models:
            # Check if distributed execution is available and requested
            use_distributed = distributed and hasattr(self.bridge_integration, 'run_distributed_inference')
            
            if use_distributed:
                # For distributed execution, use the specialized bridge integration method
                try:
                    logger.info(f"Running distributed inference for {len(web_models)} models")
                    model_input_pairs = [(m, i) for m, i in web_models]
                    execution_config = {
                        'batch_size': batch_size,
                        'timeout': timeout,
                        'operation_id': operation_id,
                        'priority': priority
                    }
                    
                    # Run distributed inference across multiple browser connections
                    dist_results = self.bridge_integration.run_distributed_inference(
                        model_input_pairs, execution_config)
                    
                    # Track distributed execution metrics if requested
                    if track_metrics:
                        if "distributed_metrics" not in self.metrics:
                            self.metrics["distributed_metrics"] = {
                                "operations": 0,
                                "total_models": 0,
                                "successful_models": 0,
                                "total_time": 0.0,
                                "execution_history": []
                            }
                        
                        distributed_time = time.time() - operation_start_time
                        self.metrics["distributed_metrics"]["operations"] += 1
                        self.metrics["distributed_metrics"]["total_models"] += len(web_models)
                        self.metrics["distributed_metrics"]["successful_models"] += len([r for r in dist_results if r is not None])
                        self.metrics["distributed_metrics"]["total_time"] += distributed_time
                        
                        # Keep history of last 10 operations
                        execution_record = {
                            "operation_id": operation_id,
                            "timestamp": time.time(),
                            "models": len(web_models),
                            "execution_time": distributed_time,
                            "success_rate": len([r for r in dist_results if r is not None]) / len(web_models) if web_models else 0
                        }
                        
                        self.metrics["distributed_metrics"]["execution_history"].append(execution_record)
                        if len(self.metrics["distributed_metrics"]["execution_history"]) > 10:
                            self.metrics["distributed_metrics"]["execution_history"].pop(0)
                    
                    # Add distributed results to main results list
                    results.extend(dist_results)
                    
                except Exception as e:
                    logger.error(f"Error during distributed inference: {e}")
                    error_occurred = True
                    
                    # Fall back to standard execution methods
                    logger.info("Falling back to standard concurrent execution")
                    use_distributed = False
            
            # Use standard concurrent execution if distributed is not available/requested
            # or if distributed execution failed
            if not use_distributed:
                try:
                    # Check if we have multiple web models or batch processing
                    if len(web_models) > 1 or (batch_size and batch_size > 1):
                        # Get first model for execution with others
                        first_model, first_input = web_models[0]
                        other_web_models = []
                        other_inputs_list = []
                        
                        # Setup for concurrent execution
                        for i, (model, inputs) in enumerate(web_models[1:], start=1):
                            # Skip if model is invalid
                            if not hasattr(model, 'is_valid') or not model.is_valid():
                                logger.warning(f"Skipping invalid model in position {i}")
                                results.append(None)  # Add None placeholder for invalid model
                                continue
                                
                            other_web_models.append(model)
                            other_inputs_list.append(inputs)
                        
                        # Check if first model supports run_concurrent
                        if hasattr(first_model, 'run_concurrent'):
                            # Track concurrent execution start time
                            concurrent_start_time = time.time()
                            
                            # Run with batch processing if requested
                            if batch_size and batch_size > 1:
                                # Create batch input
                                batch_inputs = [first_input] * batch_size
                                
                                # Create enhanced execution config
                                execution_config = {
                                    'timeout': timeout,
                                    'operation_id': operation_id,
                                    'track_metrics': track_metrics,
                                    'store_in_db': store_in_db,
                                    'telemetry': telemetry_data,
                                    'priority': priority
                                }
                                
                                # Run batched concurrent execution
                                batch_results = first_model.run_concurrent(
                                    batch_inputs, other_web_models, execution_config)
                                
                                # Add results
                                results.extend(batch_results)
                                
                                # Track metrics if requested
                                if track_metrics:
                                    if "batched_execution" not in self.metrics:
                                        self.metrics["batched_execution"] = {
                                            "operations": 0,
                                            "total_batch_items": 0,
                                            "total_time": 0.0
                                        }
                                    
                                    batched_time = time.time() - concurrent_start_time
                                    self.metrics["batched_execution"]["operations"] += 1
                                    self.metrics["batched_execution"]["total_batch_items"] += batch_size
                                    self.metrics["batched_execution"]["total_time"] += batched_time
                            else:
                                # Single input with concurrent models
                                web_result = first_model.run_concurrent(
                                    [first_input], other_web_models, 
                                    config={
                                        'timeout': timeout,
                                        'operation_id': operation_id
                                    })
                                
                                # Add results
                                if web_result:
                                    results.append(web_result[0])
                                else:
                                    results.append(None)
                                
                                # Track metrics if requested
                                if track_metrics and web_result:
                                    if "concurrent_execution" not in self.metrics:
                                        self.metrics["concurrent_execution"] = {
                                            "operations": 0,
                                            "total_models": 0,
                                            "total_time": 0.0
                                        }
                                    
                                    concurrent_time = time.time() - concurrent_start_time
                                    self.metrics["concurrent_execution"]["operations"] += 1
                                    self.metrics["concurrent_execution"]["total_models"] += len(other_web_models) + 1
                                    self.metrics["concurrent_execution"]["total_time"] += concurrent_time
                        else:
                            # Fallback to individual inference if run_concurrent not supported
                            logger.warning("Model does not support run_concurrent, falling back to individual inference")
                            for model, inputs in web_models:
                                results.append(self.run_inference(
                                    model, inputs, batch_size, timeout, 
                                    track_metrics, store_in_db, telemetry_data))
                    else:
                        # Only one model, run simple inference
                        model, inputs = web_models[0]
                        results.append(self.run_inference(
                            model, inputs, batch_size, timeout, 
                            track_metrics, store_in_db, telemetry_data))
                        
                except Exception as e:
                    logger.error(f"Error during concurrent execution: {e}")
                    error_occurred = True
                    
                    # Add placeholders for failed results
                    results.extend([None] * (len(web_models) - len(results)))
        
        # Process other models sequentially
        for model, inputs in zip(other_models, other_inputs):
            try:
                result = self.run_inference(
                    model, inputs, batch_size, timeout, 
                    track_metrics, store_in_db, telemetry_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error during sequential inference: {e}")
                error_occurred = True
                results.append(None)
        
        # Calculate overall operation time
        operation_time = time.time() - operation_start_time
        
        # Log overall performance
        logger.info(f"Parallel inference completed for {len(model_data_pairs)} models in {operation_time:.2f}s")
        
        # Store operation metrics in database if enabled
        if store_in_db and self.db_connection:
            self._store_parallel_execution_metrics(
                model_data_pairs, results, operation_time, operation_id, 
                batch_size, distributed, error_occurred)
        
        return results
        
    def _store_parallel_execution_metrics(self, model_data_pairs, results, 
                                         operation_time, operation_id, 
                                         batch_size=None, distributed=False,
                                         error_occurred=False):
        """Store parallel execution metrics in database"""
        if not self.db_connection:
            return
            
        try:
            # Create parallel execution metrics table if it doesn't exist
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS web_parallel_execution_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                operation_id VARCHAR,
                total_models INTEGER,
                successful_models INTEGER,
                execution_time_sec FLOAT,
                batch_size INTEGER,
                distributed BOOLEAN,
                error_occurred BOOLEAN,
                model_types JSON,
                platforms_used JSON,
                browsers_used JSON,
                memory_usage_mb FLOAT,
                detailed_metrics JSON
            )
            """)
            
            # Count successful models
            successful_models = len([r for r in results if r is not None])
            
            # Collect model types
            model_types = {}
            for model, _ in model_data_pairs:
                if hasattr(model, "get_model_info"):
                    model_info = model.get_model_info()
                    model_type = model_info.get("model_type", "unknown")
                    if model_type not in model_types:
                        model_types[model_type] = 0
                    model_types[model_type] += 1
            
            # Collect platforms and browsers used
            platforms_used = {}
            browsers_used = {}
            for model, _ in model_data_pairs:
                if hasattr(model, "get_model_info"):
                    model_info = model.get_model_info()
                    platform = model_info.get("platform", "unknown")
                    browser = model_info.get("browser", "unknown")
                    
                    if platform not in platforms_used:
                        platforms_used[platform] = 0
                    platforms_used[platform] += 1
                    
                    if browser not in browsers_used:
                        browsers_used[browser] = 0
                    browsers_used[browser] += 1
            
            # Calculate total memory usage
            memory_usage = 0.0
            for model, _ in model_data_pairs:
                if hasattr(model, "get_model_info"):
                    model_info = model.get_model_info()
                    memory_usage += model_info.get("memory_usage_mb", 0.0)
            
            # Detailed metrics for analysis
            detailed_metrics = {
                "average_model_time": operation_time / len(model_data_pairs) if model_data_pairs else 0,
                "success_rate": successful_models / len(model_data_pairs) if model_data_pairs else 0,
                "total_data_processed": len(model_data_pairs) * (batch_size or 1),
                "throughput_items_per_sec": len(model_data_pairs) * (batch_size or 1) / operation_time if operation_time > 0 else 0
            }
            
            # Insert data into database
            self.db_connection.execute("""
            INSERT INTO web_parallel_execution_metrics (
                timestamp, operation_id, total_models, successful_models,
                execution_time_sec, batch_size, distributed, error_occurred,
                model_types, platforms_used, browsers_used, memory_usage_mb,
                detailed_metrics
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, [
                operation_id, len(model_data_pairs), successful_models,
                operation_time, batch_size or 1, distributed, error_occurred,
                json.dumps(model_types), json.dumps(platforms_used),
                json.dumps(browsers_used), memory_usage,
                json.dumps(detailed_metrics)
            ])
            
        except Exception as e:
            logger.error(f"Error storing parallel execution metrics: {e}")
            # Continue execution despite database error
    
    def get_metrics(self, include_history=False, format_for_dashboard=False):
        """
        Get comprehensive performance metrics for all models with enhanced reporting.
        
        Args:
            include_history: Whether to include detailed history data (can be large)
            format_for_dashboard: Whether to format metrics for dashboard visualization
            
        Returns:
            Dict of performance metrics
        """
        # Get bridge integration metrics
        if self.bridge_integration:
            # Get real-time bridge integration stats
            bridge_stats = self.bridge_integration.get_execution_stats()
            
            # Add bridge stats to metrics
            enhanced_metrics = {
                "models": self.metrics,
                "bridge": bridge_stats,
                "resource_pool": self.resource_pool.get_stats(),
                "timestamp": time.time(),
                "uptime_seconds": time.time() - getattr(self, "creation_time", time.time()),
                "version": "2.0.0"  # Enhanced version with WebGPU/WebNN Resource Pool
            }
            
            # Calculate aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics()
            enhanced_metrics["aggregate"] = aggregate_metrics
            
            # Calculate real-time resource utilization
            if hasattr(self.bridge_integration, "get_resource_utilization"):
                resource_utilization = self.bridge_integration.get_resource_utilization()
                enhanced_metrics["resource_utilization"] = resource_utilization
            
            # Add database metrics if available
            if self.db_connection:
                db_metrics = self._get_database_metrics()
                enhanced_metrics["database"] = db_metrics
            
            # Add distributed inference metrics if available
            if "distributed_metrics" in self.metrics:
                # Either include full history or summarize
                if not include_history:
                    # Exclude detailed history to reduce size
                    distributed_metrics = self.metrics["distributed_metrics"].copy()
                    if "execution_history" in distributed_metrics:
                        # Just include count instead of full history
                        history_count = len(distributed_metrics["execution_history"])
                        distributed_metrics["execution_history_count"] = history_count
                        del distributed_metrics["execution_history"]
                    enhanced_metrics["distributed_inference"] = distributed_metrics
                else:
                    enhanced_metrics["distributed_inference"] = self.metrics["distributed_metrics"]
            
            # Add optimization metrics
            if "performance_optimization" in self.metrics:
                enhanced_metrics["optimization"] = self.metrics["performance_optimization"]
            
            # Add connection pool metrics
            if hasattr(self.bridge_integration, "get_connection_pool_metrics"):
                connection_pool_metrics = self.bridge_integration.get_connection_pool_metrics()
                enhanced_metrics["connection_pool"] = connection_pool_metrics
            
            # Format for dashboard visualization if requested
            if format_for_dashboard:
                return self._format_metrics_for_dashboard(enhanced_metrics)
            
            return enhanced_metrics
        
        return self.metrics
        
    def _get_database_metrics(self):
        """Get metrics about database usage"""
        if not self.db_connection:
            return {"available": False}
            
        try:
            # Get table row counts
            model_metrics_count = self.db_connection.execute(
                "SELECT COUNT(*) FROM web_resource_pool_metrics").fetchone()[0]
                
            inference_records_count = 0
            try:
                inference_records_count = self.db_connection.execute(
                    "SELECT COUNT(*) FROM web_inference_records").fetchone()[0]
            except:
                pass  # Table might not exist yet
                
            parallel_execution_count = 0
            try:
                parallel_execution_count = self.db_connection.execute(
                    "SELECT COUNT(*) FROM web_parallel_execution_metrics").fetchone()[0]
            except:
                pass  # Table might not exist yet
                
            # Get performance statistics
            platform_distribution = {}
            try:
                platform_results = self.db_connection.execute("""
                SELECT platform, COUNT(*) as count 
                FROM web_resource_pool_metrics 
                GROUP BY platform
                """).fetchall()
                
                for platform, count in platform_results:
                    platform_distribution[platform] = count
            except:
                pass
                
            browser_distribution = {}
            try:
                browser_results = self.db_connection.execute("""
                SELECT browser_name, COUNT(*) as count 
                FROM web_resource_pool_metrics 
                GROUP BY browser_name
                """).fetchall()
                
                for browser, count in browser_results:
                    browser_distribution[browser] = count
            except:
                pass
            
            # Return database metrics
            return {
                "available": True,
                "db_path": self.db_path,
                "tables": {
                    "web_resource_pool_metrics": model_metrics_count,
                    "web_inference_records": inference_records_count,
                    "web_parallel_execution_metrics": parallel_execution_count
                },
                "platform_distribution": platform_distribution,
                "browser_distribution": browser_distribution
            }
            
        except Exception as e:
            logger.error(f"Error getting database metrics: {e}")
            return {"available": False, "error": str(e)}
            
    def _format_metrics_for_dashboard(self, metrics):
        """Format metrics for dashboard visualization"""
        dashboard_metrics = {
            "summary": {
                "total_models_loaded": len(metrics["models"].get("model_load_time", {})),
                "total_inferences": sum(metrics["models"].get("inference_attempts", {}).values()) if "inference_attempts" in metrics["models"] else 0,
                "active_connections": metrics["bridge"].get("bridge_stats", {}).get("current_connections", 0),
                "memory_usage_mb": sum(metrics["models"].get("memory_usage", {}).values()) if "memory_usage" in metrics["models"] else 0,
                "uptime_seconds": metrics.get("uptime_seconds", 0),
                "success_rate": metrics["aggregate"].get("success_rate", 0) * 100 if "success_rate" in metrics["aggregate"] else 0
            },
            "performance": {
                "avg_load_time": metrics["aggregate"].get("avg_load_time", 0),
                "avg_inference_time": metrics["aggregate"].get("avg_inference_time", 0),
                "avg_throughput": metrics["aggregate"].get("avg_throughput", 0),
                "avg_latency": metrics["aggregate"].get("avg_latency", 0)
            },
            "distributions": {
                "platforms": metrics["aggregate"].get("platform_distribution", {}),
                "browsers": metrics["models"].get("browser_distribution", {}),
                "model_types": self._calculate_model_type_distribution(metrics["models"])
            },
            "optimizations": {
                "shader_precompilation_savings_ms": metrics.get("optimization", {}).get("shader_precompilation_savings_ms", 0),
                "compute_shader_speedup_percent": metrics.get("optimization", {}).get("compute_shader_speedup_percent", 0),
                "parallel_loading_savings_ms": metrics.get("optimization", {}).get("parallel_loading_savings_ms", 0)
            },
            "charts_data": {
                "load_time_distribution": metrics["models"].get("load_time_distribution", {}),
                "memory_by_model": self._format_memory_by_model(metrics["models"].get("memory_usage", {})),
                "throughput_by_model": self._format_throughput_by_model(metrics["models"].get("throughput", {}))
            },
            "timestamp": time.time()
        }
        
        return dashboard_metrics
        
    def _calculate_model_type_distribution(self, metrics):
        """Calculate distribution of model types"""
        type_distribution = {}
        
        # Extract model types from loaded models
        for model_key in metrics.get("model_load_time", {}):
            if ':' in model_key:
                model_name = model_key.split(':')[0]
                model_type = self._infer_model_type(model_name)
                
                if model_type not in type_distribution:
                    type_distribution[model_type] = 0
                type_distribution[model_type] += 1
        
        return type_distribution
        
    def _format_memory_by_model(self, memory_usage):
        """Format memory usage by model for visualization"""
        if not memory_usage:
            return []
            
        chart_data = []
        for model_key, memory in memory_usage.items():
            model_name = model_key
            if ':' in model_key:
                model_name = model_key.split(':')[0]
            
            chart_data.append({
                "model": model_name,
                "memory_mb": memory.get("reported", 0) if isinstance(memory, dict) else memory
            })
            
        # Sort by memory usage (descending)
        chart_data.sort(key=lambda x: x["memory_mb"], reverse=True)
        
        # Limit to top 10
        return chart_data[:10]
        
    def _format_throughput_by_model(self, throughput):
        """Format throughput by model for visualization"""
        if not throughput:
            return []
            
        chart_data = []
        for model_key, throughput_value in throughput.items():
            model_name = model_key
            if ':' in model_key:
                model_name = model_key.split(':')[0]
            
            chart_data.append({
                "model": model_name,
                "throughput": throughput_value
            })
            
        # Sort by throughput (descending)
        chart_data.sort(key=lambda x: x["throughput"], reverse=True)
        
        # Limit to top 10
        return chart_data[:10]
    
    def _calculate_aggregate_metrics(self):
        """
        Calculate aggregate metrics across all models.
        
        Returns:
            Dict of aggregate metrics
        """
        aggregate = {
            "avg_load_time": 0,
            "avg_inference_time": 0,
            "avg_throughput": 0,
            "avg_latency": 0,
            "total_models": len(self.metrics["model_load_time"]),
            "platform_distribution": self.metrics["platform_distribution"],
            "total_memory_usage": 0
        }
        
        # Calculate average load time
        if self.metrics["model_load_time"]:
            aggregate["avg_load_time"] = sum(self.metrics["model_load_time"].values()) / len(self.metrics["model_load_time"])
        
        # Calculate average inference time
        inference_times = []
        for times in self.metrics["inference_time"].values():
            inference_times.extend(times)
        
        if inference_times:
            aggregate["avg_inference_time"] = sum(inference_times) / len(inference_times)
        
        # Calculate average throughput
        if self.metrics["throughput"]:
            aggregate["avg_throughput"] = sum(self.metrics["throughput"].values()) / len(self.metrics["throughput"])
        
        # Calculate average latency
        if self.metrics["latency"]:
            aggregate["avg_latency"] = sum(self.metrics["latency"].values()) / len(self.metrics["latency"])
        
        # Calculate total memory usage
        for memory_info in self.metrics["memory_usage"].values():
            if isinstance(memory_info, dict) and "reported" in memory_info:
                aggregate["total_memory_usage"] += memory_info["reported"]
        
        return aggregate
    
    def close(self, store_final_metrics=True):
        """
        Clean up resources and close connections with proper cleanup.
        
        Args:
            store_final_metrics: Whether to store final metrics in database before closing
        """
        # Store final metrics in database if enabled
        if store_final_metrics and self.db_connection:
            try:
                # Create final metrics table if it doesn't exist
                self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS web_resource_pool_session_metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    session_id VARCHAR,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_sec FLOAT,
                    total_models_loaded INTEGER,
                    total_inferences INTEGER,
                    peak_connections INTEGER,
                    models_used JSON,
                    platforms_used JSON,
                    browsers_used JSON,
                    detailed_metrics JSON
                )
                """)
                
                # Get session metrics
                session_id = getattr(self, "session_id", str(uuid.uuid4()))
                creation_time = getattr(self, "creation_time", time.time())
                end_time = time.time()
                duration = end_time - creation_time
                
                # Get enhanced metrics with history
                session_metrics = self.get_metrics(include_history=True)
                
                # Extract key metrics
                total_models_loaded = len(session_metrics.get("models", {}).get("model_load_time", {}))
                total_inferences = sum(session_metrics.get("models", {}).get("inference_attempts", {}).values()) if "inference_attempts" in session_metrics.get("models", {}) else 0
                
                # Get peak connections
                peak_connections = 0
                if "bridge" in session_metrics and "bridge_stats" in session_metrics["bridge"]:
                    peak_connections = session_metrics["bridge"]["bridge_stats"].get("peak_connections", 0)
                
                # Get models, platforms, and browsers used
                models_used = {}
                for model_key in session_metrics.get("models", {}).get("model_load_time", {}):
                    model_name = model_key
                    if ':' in model_key:
                        model_name = model_key.split(':')[0]
                    
                    if model_name not in models_used:
                        models_used[model_name] = 0
                    models_used[model_name] += 1
                
                # Insert final metrics
                self.db_connection.execute("""
                INSERT INTO web_resource_pool_session_metrics (
                    timestamp, session_id, start_time, end_time, duration_sec,
                    total_models_loaded, total_inferences, peak_connections,
                    models_used, platforms_used, browsers_used, detailed_metrics
                ) VALUES (
                    CURRENT_TIMESTAMP, ?, datetime(?), datetime(?), ?, ?, ?, ?, ?, ?, ?, ?
                )
                """, [
                    session_id, 
                    datetime.fromtimestamp(creation_time).isoformat(), 
                    datetime.fromtimestamp(end_time).isoformat(),
                    duration,
                    total_models_loaded,
                    total_inferences,
                    peak_connections,
                    json.dumps(models_used),
                    json.dumps(session_metrics.get("models", {}).get("platform_distribution", {})),
                    json.dumps(session_metrics.get("models", {}).get("browser_distribution", {})),
                    json.dumps(session_metrics)
                ])
                
                # Close database connection
                self.db_connection.close()
                logger.info("Database connection closed and final metrics stored")
                
            except Exception as e:
                logger.error(f"Error storing final metrics: {e}")
                # Try to close database connection
                try:
                    if self.db_connection:
                        self.db_connection.close()
                except:
                    pass
        
        # Close bridge integration
        if self.bridge_integration:
            try:
                self.bridge_integration.close()
                logger.info("ResourcePoolBridgeIntegration closed successfully")
            except Exception as e:
                logger.error(f"Error closing ResourcePoolBridgeIntegration: {e}")
        
        # Clean up loaded models
        for model_key, model in list(self.loaded_models.items()):
            try:
                if hasattr(model, "close"):
                    model.close()
            except Exception as e:
                logger.warning(f"Error closing model {model_key}: {e}")
        
        # Clear loaded models
        self.loaded_models.clear()
        
        # Set closed flag
        self.closed = True
        
        logger.info("IPFSAccelerateWebIntegration closed successfully")
    
    def __del__(self):
        """Destructor to clean up resources."""
        if not getattr(self, "closed", False):
            self.close()

class IPFSWebAccelerator:
    """
    Enhanced IPFS accelerator with WebNN/WebGPU integration.
    
    This class provides a high-level interface for accelerating IPFS models
    using WebNN and WebGPU hardware acceleration in browsers.
    """
    
    def __init__(self, db_path=None, max_connections=4, enable_gpu=True, 
                enable_cpu=True, browser_preferences=None):
        """
        Initialize IPFS Web Accelerator.
        
        Args:
            db_path: Optional path to database for storing results
            max_connections: Maximum number of browser connections
            enable_gpu: Whether to enable GPU acceleration
            enable_cpu: Whether to enable CPU acceleration
            browser_preferences: Dict mapping model families to preferred browsers
        """
        self.integration = IPFSAccelerateWebIntegration(
            max_connections=max_connections,
            enable_gpu=enable_gpu,
            enable_cpu=enable_cpu,
            browser_preferences=browser_preferences
        )
        self.db_path = db_path
        self.model_cache = {}
        
        # Database integration
        self.db_integration = self._setup_db_integration() if db_path else None
        
        logger.info("IPFSWebAccelerator initialized successfully")
    
    def _setup_db_integration(self):
        """
        Set up database integration.
        
        Returns:
            Database integration object or None if not available
        """
        try:
            # Try to import database API
            from benchmark_db_api import DatabaseAPI
            
            # Create DB integration
            db_api = DatabaseAPI(db_path=self.db_path)
            logger.info(f"Database integration initialized with DB: {self.db_path}")
            return db_api
        except ImportError:
            logger.warning("benchmark_db_api module not available. Running without database integration.")
            return None
        except Exception as e:
            logger.error(f"Error setting up database integration: {e}")
            return None
    
    def accelerate_model(self, model_name, model_type=None, platform="webgpu", 
                         quantization=None, optimizations=None):
        """
        Get accelerated model for inference.
        
        Args:
            model_name: Name of the model to accelerate
            model_type: Type of model (inferred if not provided)
            platform: Acceleration platform (webgpu, webnn, cpu)
            quantization: Quantization settings (bits, mixed_precision)
            optimizations: Additional optimizations to enable
            
        Returns:
            Accelerated model ready for inference
        """
        # Create cache key
        cache_key = f"{model_name}:{platform}"
        if quantization:
            bits = quantization.get("bits", 16)
            mixed = quantization.get("mixed_precision", False)
            cache_key += f":{bits}bit{'_mixed' if mixed else ''}"
        
        # Check cache
        if cache_key in self.model_cache:
            logger.debug(f"Using cached accelerated model for {model_name}")
            return self.model_cache[cache_key]
        
        # Get accelerated model from integration
        model = self.integration.get_model(
            model_name=model_name,
            model_type=model_type,
            platform=platform,
            quantization=quantization,
            optimizations=optimizations
        )
        
        # Cache for future use
        self.model_cache[cache_key] = model
        
        return model
    
    def run_inference(self, model_name, inputs, model_type=None, platform="webgpu", 
                     quantization=None, optimizations=None, store_results=True):
        """
        Run inference with accelerated model.
        
        Args:
            model_name: Name of the model to use
            inputs: Input data for inference
            model_type: Type of model (inferred if not provided)
            platform: Acceleration platform (webgpu, webnn, cpu)
            quantization: Quantization settings (bits, mixed_precision)
            optimizations: Additional optimizations to enable
            store_results: Whether to store results in database
            
        Returns:
            Inference results
        """
        # Get accelerated model
        model = self.accelerate_model(
            model_name=model_name,
            model_type=model_type,
            platform=platform,
            quantization=quantization,
            optimizations=optimizations
        )
        
        # Run inference
        start_time = time.time()
        result = self.integration.run_inference(model, inputs)
        inference_time = time.time() - start_time
        
        # Store results in database if enabled
        if store_results and self.db_integration:
            try:
                # Get performance metrics
                metrics = model.get_performance_metrics() if hasattr(model, "get_performance_metrics") else {}
                
                # Prepare metadata
                metadata = {
                    "model_name": model_name,
                    "model_type": model_type or self.integration._infer_model_type(model_name),
                    "platform": platform,
                    "timestamp": time.time(),
                    "inference_time": inference_time,
                    "hardware_type": platform,
                    "quantization": quantization,
                    "optimizations": optimizations,
                    "performance_metrics": metrics
                }
                
                # Store in database
                self.db_integration.store_inference_result(result, metadata)
                logger.debug(f"Stored inference result in database for {model_name}")
            except Exception as e:
                logger.error(f"Error storing results in database: {e}")
        
        return result
    
    def run_batch_inference(self, model_name, batch_inputs, model_type=None, 
                           platform="webgpu", batch_size=None, store_results=True):
        """
        Run batch inference with accelerated model.
        
        Args:
            model_name: Name of the model to use
            batch_inputs: List of input data for batch inference
            model_type: Type of model (inferred if not provided)
            platform: Acceleration platform (webgpu, webnn, cpu)
            batch_size: Batch size for inference (default is auto)
            store_results: Whether to store results in database
            
        Returns:
            List of inference results
        """
        # Get accelerated model
        model = self.accelerate_model(
            model_name=model_name,
            model_type=model_type,
            platform=platform
        )
        
        # Determine batch size if not specified
        if batch_size is None:
            if hasattr(model, "max_batch_size"):
                batch_size = model.max_batch_size
            else:
                batch_size = 1  # Default to 1 if unknown
        
        # Process in batches
        all_results = []
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i+batch_size]
            
            # Run inference with batch
            start_time = time.time()
            if hasattr(model, "run_batch"):
                batch_results = model.run_batch(batch)
            else:
                # Sequential processing if batch not supported
                batch_results = [model(inputs) for inputs in batch]
            
            inference_time = time.time() - start_time
            
            # Store results in database if enabled
            if store_results and self.db_integration:
                try:
                    # Get performance metrics
                    metrics = model.get_performance_metrics() if hasattr(model, "get_performance_metrics") else {}
                    
                    # Prepare metadata
                    metadata = {
                        "model_name": model_name,
                        "model_type": model_type or self.integration._infer_model_type(model_name),
                        "platform": platform,
                        "timestamp": time.time(),
                        "inference_time": inference_time,
                        "batch_size": len(batch),
                        "hardware_type": platform,
                        "performance_metrics": metrics
                    }
                    
                    # Store in database
                    self.db_integration.store_batch_inference_result(batch_results, metadata)
                except Exception as e:
                    logger.error(f"Error storing batch results in database: {e}")
            
            # Add batch results to overall results
            all_results.extend(batch_results)
        
        return all_results
    
    def get_performance_report(self, format="json"):
        """
        Get performance report for all models.
        
        Args:
            format: Output format (json, markdown, html)
            
        Returns:
            Performance report in specified format
        """
        # Get metrics from integration
        metrics = self.integration.get_metrics()
        
        if format == "json":
            return json.dumps(metrics, indent=2)
        
        elif format == "markdown":
            # Generate markdown report
            report = "# WebNN/WebGPU Acceleration Performance Report\n\n"
            
            # Add timestamp
            report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add summary
            if "aggregate" in metrics:
                agg = metrics["aggregate"]
                report += "## Summary\n\n"
                report += f"- Total Models: {agg['total_models']}\n"
                report += f"- Average Load Time: {agg['avg_load_time']:.4f}s\n"
                report += f"- Average Inference Time: {agg['avg_inference_time']:.4f}s\n"
                report += f"- Average Throughput: {agg['avg_throughput']:.2f} items/s\n"
                report += f"- Average Latency: {agg['avg_latency']:.4f}s\n"
                report += f"- Total Memory Usage: {agg['total_memory_usage']/1024/1024:.2f} MB\n\n"
                
                # Add platform distribution
                report += "### Platform Distribution\n\n"
                report += "| Platform | Count |\n"
                report += "|----------|-------|\n"
                for platform, count in agg["platform_distribution"].items():
                    report += f"| {platform} | {count} |\n"
                report += "\n"
            
            # Add model details
            report += "## Model Details\n\n"
            report += "| Model | Platform | Load Time (s) | Avg Inference Time (s) | Throughput (items/s) | Latency (s) |\n"
            report += "|-------|----------|--------------|------------------------|---------------------|------------|\n"
            
            # Process each model
            for model_id, load_time in metrics["models"]["model_load_time"].items():
                # Extract model name
                model_name = model_id.split(':')[0] if ':' in model_id else model_id
                
                # Get platform
                platform = "unknown"
                for p, count in metrics["models"]["platform_distribution"].items():
                    if count > 0 and p in model_id:
                        platform = p
                
                # Get inference time
                inference_times = metrics["models"]["inference_time"].get(model_id, [])
                avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
                
                # Get throughput and latency
                throughput = metrics["models"]["throughput"].get(model_id, 0)
                latency = metrics["models"]["latency"].get(model_id, 0)
                
                # Add row
                report += f"| {model_name} | {platform} | {load_time:.4f} | {avg_inference_time:.4f} | {throughput:.2f} | {latency:.4f} |\n"
            
            return report
            
        elif format == "html":
            # Generate HTML report (simplified version)
            html = """<!DOCTYPE html>
<html>
<head>
    <title>WebNN/WebGPU Acceleration Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>WebNN/WebGPU Acceleration Performance Report</h1>
    <p>Generated: {timestamp}</p>
""".format(timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
            
            # Add summary
            if "aggregate" in metrics:
                agg = metrics["aggregate"]
                html += """
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Models:</strong> {total_models}</p>
        <p><strong>Average Load Time:</strong> {avg_load_time:.4f}s</p>
        <p><strong>Average Inference Time:</strong> {avg_inference_time:.4f}s</p>
        <p><strong>Average Throughput:</strong> {avg_throughput:.2f} items/s</p>
        <p><strong>Average Latency:</strong> {avg_latency:.4f}s</p>
        <p><strong>Total Memory Usage:</strong> {total_memory_usage:.2f} MB</p>
    </div>
""".format(
    total_models=agg['total_models'],
    avg_load_time=agg['avg_load_time'],
    avg_inference_time=agg['avg_inference_time'],
    avg_throughput=agg['avg_throughput'],
    avg_latency=agg['avg_latency'],
    total_memory_usage=agg['total_memory_usage']/1024/1024
)
                
                # Add platform distribution
                html += """
    <h2>Platform Distribution</h2>
    <table>
        <tr>
            <th>Platform</th>
            <th>Count</th>
        </tr>
"""
                for platform, count in agg["platform_distribution"].items():
                    html += f"        <tr><td>{platform}</td><td>{count}</td></tr>\n"
                
                html += "    </table>\n"
            
            # Add model details
            html += """
    <h2>Model Details</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Platform</th>
            <th>Load Time (s)</th>
            <th>Avg Inference Time (s)</th>
            <th>Throughput (items/s)</th>
            <th>Latency (s)</th>
        </tr>
"""
            
            # Process each model
            for model_id, load_time in metrics["models"]["model_load_time"].items():
                # Extract model name
                model_name = model_id.split(':')[0] if ':' in model_id else model_id
                
                # Get platform
                platform = "unknown"
                for p, count in metrics["models"]["platform_distribution"].items():
                    if count > 0 and p in model_id:
                        platform = p
                
                # Get inference time
                inference_times = metrics["models"]["inference_time"].get(model_id, [])
                avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
                
                # Get throughput and latency
                throughput = metrics["models"]["throughput"].get(model_id, 0)
                latency = metrics["models"]["latency"].get(model_id, 0)
                
                # Add row
                html += f"""        <tr>
            <td>{model_name}</td>
            <td>{platform}</td>
            <td>{load_time:.4f}</td>
            <td>{avg_inference_time:.4f}</td>
            <td>{throughput:.2f}</td>
            <td>{latency:.4f}</td>
        </tr>
"""
            
            html += """    </table>
</body>
</html>"""
            
            return html
        
        else:
            raise ValueError(f"Unsupported format: {format}. Supported formats: json, markdown, html")
    
    def close(self):
        """Clean up resources and close connections."""
        if self.integration:
            self.integration.close()
        
        # Close database connection if open
        if self.db_integration and hasattr(self.db_integration, "close"):
            self.db_integration.close()
        
        logger.info("IPFSWebAccelerator closed and cleaned up")
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.close()


# Helper function to create accelerator
def create_ipfs_web_accelerator(db_path=None, max_connections=4, 
                               enable_gpu=True, enable_cpu=True,
                               browser_preferences=None):
    """
    Create an IPFS Web Accelerator instance.
    
    Args:
        db_path: Optional path to database for storing results
        max_connections: Maximum number of browser connections
        enable_gpu: Whether to enable GPU acceleration
        enable_cpu: Whether to enable CPU acceleration
        browser_preferences: Dict mapping model families to preferred browsers
        
    Returns:
        IPFSWebAccelerator instance
    """
    return IPFSWebAccelerator(
        db_path=db_path,
        max_connections=max_connections,
        enable_gpu=enable_gpu,
        enable_cpu=enable_cpu,
        browser_preferences=browser_preferences
    )


# Automatically integrate with resource pool when imported
def integrate_with_resource_pool():
    """Integrate IPFSAccelerateWebIntegration with resource pool."""
    integration = IPFSAccelerateWebIntegration()
    
    # Store in global resource pool for access
    resource_pool = get_global_resource_pool()
    resource_pool.get_resource("ipfs_web_integration", constructor=lambda: integration)
    
    return integration

# Auto-integration when module is imported
if __name__ != "__main__":
    integrate_with_resource_pool()


# Simple test function for the module
def test_integration():
    """Test IPFS WebNN/WebGPU integration."""
    # Create accelerator
    accelerator = create_ipfs_web_accelerator()
    
    # Test with a simple model
    model_name = "bert-base-uncased"
    model = accelerator.accelerate_model(model_name, platform="webgpu")
    
    # Create sample input
    sample_input = {
        "input_ids": [101, 2023, 2003, 1037, 3231, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1]
    }
    
    # Run inference
    result = accelerator.run_inference(model_name, sample_input)
    
    # Get performance report
    report = accelerator.get_performance_report(format="markdown")
    print(report)
    
    # Close accelerator
    accelerator.close()
    
    return result

if __name__ == "__main__":
    test_integration()