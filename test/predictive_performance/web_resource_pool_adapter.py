#!/usr/bin/env python3
"""
WebNN/WebGPU Resource Pool Adapter for Multi-Model Execution Support.

This module provides integration between the Multi-Model Execution Support system
and the WebNN/WebGPU Resource Pool, enabling browser-based concurrent model execution
with optimized resource allocation and tensor sharing.

Key features:
1. Browser-specific execution strategies
2. Shared tensor buffers between browser-based models
3. Optimized model placement based on browser capabilities
4. Adaptive strategy selection for different browser environments
5. Memory optimization for browser-based model execution
"""

import os
import sys
import time
import json
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("predictive_performance.web_resource_pool_adapter")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import resource pool components
try:
    from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
    from fixed_web_platform.resource_pool_bridge_integration import ResourcePoolBridgeIntegrationWithRecovery
    RESOURCE_POOL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing ResourcePoolBridge components: {e}")
    logger.warning("Continuing without WebNN/WebGPU Resource Pool integration (will use simulation mode)")
    RESOURCE_POOL_AVAILABLE = False

# Browser capability constants
BROWSER_CAPABILITIES = {
    "chrome": {
        "webgpu": True,
        "webnn": True,
        "compute_shader": True,
        "memory_limit": 4000,  # MB
        "concurrent_model_limit": 8,
        "tensor_sharing": True,
        "streaming_execution": True
    },
    "firefox": {
        "webgpu": True,
        "webnn": False,
        "compute_shader": True,
        "memory_limit": 3500,  # MB
        "concurrent_model_limit": 6,
        "tensor_sharing": True,
        "streaming_execution": True
    },
    "edge": {
        "webgpu": True,
        "webnn": True,
        "compute_shader": True,
        "memory_limit": 4000,  # MB
        "concurrent_model_limit": 8,
        "tensor_sharing": True,
        "streaming_execution": True
    },
    "safari": {
        "webgpu": True,
        "webnn": False,
        "compute_shader": False,
        "memory_limit": 3000,  # MB
        "concurrent_model_limit": 4,
        "tensor_sharing": False,
        "streaming_execution": False
    }
}

# Model type to browser preferences mapping
MODEL_BROWSER_PREFERENCES = {
    "text_embedding": "edge",     # Edge has best WebNN support for text models
    "text_generation": "chrome",  # Chrome has good all-around support
    "vision": "chrome",           # Chrome works well for vision models
    "audio": "firefox",           # Firefox has best audio compute shader performance
    "multimodal": "chrome"        # Chrome has best balance for multimodal
}

# Execution strategy preferences by browser
BROWSER_STRATEGY_PREFERENCES = {
    "chrome": {
        "parallel_threshold": 6,    # Chrome handles parallel execution well
        "sequential_threshold": 12, # Use sequential for > 12 models
        "batching_size": 4,         # Batch size for batched execution
        "memory_threshold": 3500    # Memory threshold for strategy decisions
    },
    "firefox": {
        "parallel_threshold": 4,    # Firefox more limited for parallel models
        "sequential_threshold": 8,  # Use sequential for > 8 models
        "batching_size": 3,         # Smaller batch size for Firefox
        "memory_threshold": 3000    # Memory threshold for Firefox
    },
    "edge": {
        "parallel_threshold": 6,    # Edge good at parallel execution
        "sequential_threshold": 10, # Use sequential for > 10 models
        "batching_size": 4,         # Batch size for Edge
        "memory_threshold": 3500    # Memory threshold for Edge
    },
    "safari": {
        "parallel_threshold": 2,    # Safari more limited
        "sequential_threshold": 6,  # Lower threshold for Safari
        "batching_size": 2,         # Smaller batches for Safari
        "memory_threshold": 2500    # Lower memory threshold
    }
}

class WebResourcePoolAdapter:
    """
    Adapter for integrating the WebNN/WebGPU Resource Pool with Multi-Model Execution Support.
    
    This class provides a specialized integration layer between the Multi-Model Execution
    predictor and the browser-based WebNN/WebGPU Resource Pool, enabling optimized execution
    of multiple models in browser environments.
    """
    
    def __init__(
        self,
        resource_pool: Optional[Any] = None,
        max_connections: int = 4,
        browser_preferences: Optional[Dict[str, str]] = None,
        enable_tensor_sharing: bool = True,
        enable_strategy_optimization: bool = True,
        browser_capability_detection: bool = True,
        db_path: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the WebNN/WebGPU Resource Pool Adapter.
        
        Args:
            resource_pool: Existing ResourcePoolBridgeIntegration instance (will create new if None)
            max_connections: Maximum browser connections for resource pool
            browser_preferences: Browser preferences by model type (will use defaults if None)
            enable_tensor_sharing: Whether to enable tensor sharing between models
            enable_strategy_optimization: Whether to optimize execution strategies for browsers
            browser_capability_detection: Whether to detect browser capabilities
            db_path: Path to database for storing results
            verbose: Whether to enable verbose logging
        """
        self.max_connections = max_connections
        self.browser_preferences = browser_preferences or MODEL_BROWSER_PREFERENCES.copy()
        self.enable_tensor_sharing = enable_tensor_sharing
        self.enable_strategy_optimization = enable_strategy_optimization
        self.browser_capability_detection = browser_capability_detection
        self.db_path = db_path
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize resource pool (create new if not provided)
        if resource_pool is not None:
            self.resource_pool = resource_pool
        elif RESOURCE_POOL_AVAILABLE:
            # Initialize with recovery capabilities for better fault tolerance
            self.resource_pool = ResourcePoolBridgeIntegrationWithRecovery(
                max_connections=max_connections,
                browser_preferences=self.browser_preferences,
                adaptive_scaling=True,
                enable_recovery=True,
                enable_tensor_sharing=enable_tensor_sharing,
                db_path=db_path
            )
        else:
            self.resource_pool = None
            logger.error("ResourcePoolBridgeIntegration not available")
        
        # Initialize browser capability cache
        self.browser_capabilities = {}
        
        # Initialize tensor sharing registry
        self.tensor_sharing_registry = {}
        
        # Initialize execution statistics
        self.execution_stats = {
            "total_executions": 0,
            "browser_executions": {},
            "strategy_executions": {},
            "tensor_sharing_stats": {
                "models_sharing_tensors": 0,
                "total_models": 0,
                "memory_saved_mb": 0
            }
        }
        
        # Initialize
        self.initialized = False
        logger.info(f"WebResourcePoolAdapter created "
                   f"(resource_pool={'available' if self.resource_pool else 'unavailable'}, "
                   f"tensor_sharing={'enabled' if enable_tensor_sharing else 'disabled'}, "
                   f"strategy_optimization={'enabled' if enable_strategy_optimization else 'disabled'})")
    
    def initialize(self) -> bool:
        """
        Initialize the adapter with resource pool and browser detection.
        
        Returns:
            bool: Success status
        """
        if self.initialized:
            logger.warning("WebResourcePoolAdapter already initialized")
            return True
        
        success = True
        
        # Initialize resource pool if available
        if self.resource_pool:
            logger.info("Initializing resource pool")
            pool_success = self.resource_pool.initialize()
            if not pool_success:
                logger.error("Failed to initialize resource pool")
                success = False
            else:
                logger.info("Resource pool initialized successfully")
        else:
            logger.warning("No resource pool available, will operate in simulation mode")
            success = False
        
        # Detect browser capabilities if enabled
        if self.browser_capability_detection and self.resource_pool:
            try:
                logger.info("Detecting browser capabilities")
                self._detect_browser_capabilities()
            except Exception as e:
                logger.error(f"Error detecting browser capabilities: {e}")
                traceback.print_exc()
        
        self.initialized = success
        logger.info(f"WebResourcePoolAdapter initialization {'successful' if success else 'failed'}")
        return success
    
    def _detect_browser_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect capabilities of available browsers.
        
        Returns:
            Dictionary with browser capabilities
        """
        if not self.resource_pool:
            logger.warning("No resource pool available for browser detection")
            return {}
        
        try:
            # Get available browsers from resource pool
            available_browsers = self.resource_pool.get_available_browsers()
            
            for browser_name in available_browsers:
                # Start with defaults
                capabilities = BROWSER_CAPABILITIES.get(browser_name, {}).copy()
                
                # Get actual capabilities from browser
                browser_instance = self.resource_pool.get_browser_instance(browser_name)
                if browser_instance:
                    # Check WebGPU support
                    webgpu_support = browser_instance.check_webgpu_support()
                    capabilities["webgpu"] = webgpu_support
                    
                    # Check WebNN support
                    webnn_support = browser_instance.check_webnn_support()
                    capabilities["webnn"] = webnn_support
                    
                    # Check compute shader support
                    if webgpu_support:
                        compute_shader = browser_instance.check_compute_shader_support()
                        capabilities["compute_shader"] = compute_shader
                    
                    # Get memory limits
                    memory_info = browser_instance.get_memory_info()
                    if memory_info and "limit" in memory_info:
                        capabilities["memory_limit"] = memory_info["limit"]
                    
                    # Store capabilities
                    self.browser_capabilities[browser_name] = capabilities
                    logger.info(f"Detected capabilities for {browser_name}: {capabilities}")
                    
                    # Return browser instance to pool
                    self.resource_pool.return_browser_instance(browser_instance)
            
            return self.browser_capabilities
            
        except Exception as e:
            logger.error(f"Error detecting browser capabilities: {e}")
            traceback.print_exc()
            return {}
    
    def get_optimal_browser(self, model_type: str) -> str:
        """
        Get the optimal browser for a specific model type based on capabilities.
        
        Args:
            model_type: Type of model to execute
            
        Returns:
            Browser name (chrome, firefox, edge, safari)
        """
        # Start with default preference
        browser = self.browser_preferences.get(model_type, "chrome")
        
        # If no capability detection or no capabilities detected, return default
        if not self.browser_capability_detection or not self.browser_capabilities:
            return browser
        
        # Check for specific optimizations based on model type
        if model_type == "audio" and "firefox" in self.browser_capabilities:
            firefox_caps = self.browser_capabilities["firefox"]
            if firefox_caps.get("compute_shader", False) and firefox_caps.get("webgpu", False):
                # Firefox is best for audio if it has compute shader and WebGPU
                return "firefox"
        
        elif model_type == "text_embedding" and "edge" in self.browser_capabilities:
            edge_caps = self.browser_capabilities["edge"]
            if edge_caps.get("webnn", False):
                # Edge is best for text embeddings if it has WebNN
                return "edge"
        
        elif model_type == "vision" and "chrome" in self.browser_capabilities:
            chrome_caps = self.browser_capabilities["chrome"]
            if chrome_caps.get("webgpu", False) and chrome_caps.get("compute_shader", False):
                # Chrome is best for vision with WebGPU and compute shaders
                return "chrome"
        
        # For other cases, look for browser with both WebNN and WebGPU
        for browser_name, capabilities in self.browser_capabilities.items():
            if capabilities.get("webnn", False) and capabilities.get("webgpu", False):
                return browser_name
        
        # Default to chrome if available, otherwise use first available browser
        if "chrome" in self.browser_capabilities:
            return "chrome"
        elif self.browser_capabilities:
            return next(iter(self.browser_capabilities.keys()))
        
        # Fall back to default
        return browser
    
    def get_optimal_strategy(
        self, 
        model_configs: List[Dict[str, Any]], 
        browser: str, 
        optimization_goal: str = "latency"
    ) -> str:
        """
        Get the optimal execution strategy for a set of models on a specific browser.
        
        Args:
            model_configs: List of model configurations to execute
            browser: Browser to use for execution
            optimization_goal: Metric to optimize ("latency", "throughput", or "memory")
            
        Returns:
            Execution strategy ("parallel", "sequential", or "batched")
        """
        if not self.enable_strategy_optimization:
            # Default to parallel for small numbers of models, sequential otherwise
            return "parallel" if len(model_configs) <= 3 else "sequential"
        
        # Get browser-specific strategy preferences
        strategy_prefs = BROWSER_STRATEGY_PREFERENCES.get(browser, BROWSER_STRATEGY_PREFERENCES["chrome"])
        
        # Get total memory requirement
        total_memory = self._estimate_total_memory(model_configs)
        memory_threshold = strategy_prefs.get("memory_threshold", 3500)
        
        # Count of models
        model_count = len(model_configs)
        
        # Strategy selection logic
        if model_count <= strategy_prefs.get("parallel_threshold", 4):
            # For small numbers of models, use parallel execution
            # Unless memory threshold is exceeded
            if total_memory > memory_threshold:
                return "batched"
            return "parallel"
        
        elif model_count >= strategy_prefs.get("sequential_threshold", 10):
            # For large numbers of models, use sequential execution
            return "sequential"
        
        else:
            # For medium numbers of models, base on optimization goal
            if optimization_goal == "throughput":
                # For throughput, prefer batched
                return "batched"
            elif optimization_goal == "latency":
                # For latency, prefer parallel if memory allows
                if total_memory <= memory_threshold:
                    return "parallel"
                else:
                    return "batched"
            else:  # memory
                # For memory optimization, prefer sequential
                return "sequential"
    
    def _estimate_total_memory(self, model_configs: List[Dict[str, Any]]) -> float:
        """
        Estimate total memory requirement for a set of models.
        
        Args:
            model_configs: List of model configurations
            
        Returns:
            Estimated memory requirement in MB
        """
        # Memory estimates by model type and size
        memory_estimates = {
            "text_embedding": {
                "small": 100,     # e.g., small BERT
                "base": 400,      # e.g., BERT base
                "large": 1200     # e.g., BERT large
            },
            "text_generation": {
                "small": 300,     # e.g., GPT-2 small
                "base": 800,      # e.g., GPT-2 medium
                "large": 3000     # e.g., GPT-2 large
            },
            "vision": {
                "small": 200,     # e.g., MobileNet
                "base": 500,      # e.g., ResNet50
                "large": 1000     # e.g., ViT-L
            },
            "audio": {
                "small": 300,     # e.g., Whisper tiny
                "base": 800,      # e.g., Whisper base
                "large": 1500     # e.g., Whisper large
            },
            "multimodal": {
                "small": 400,     # e.g., CLIP small
                "base": 900,      # e.g., CLIP base
                "large": 2000     # e.g., CLIP large
            }
        }
        
        # Size classification based on model name
        def classify_size(model_name: str) -> str:
            if any(size in model_name.lower() for size in ["tiny", "mini", "small"]):
                return "small"
            elif any(size in model_name.lower() for size in ["large", "huge", "xl"]):
                return "large"
            else:
                return "base"
        
        # Calculate total memory
        total_memory = 0
        for config in model_configs:
            model_type = config.get("model_type", "text_embedding")
            model_name = config.get("model_name", "")
            size = classify_size(model_name)
            
            # Get memory estimate
            memory = memory_estimates.get(model_type, {}).get(size, 500)  # Default to 500MB
            
            # Adjust for batch size
            batch_size = config.get("batch_size", 1)
            adjusted_memory = memory * (1 + 0.2 * (batch_size - 1))  # 20% increase per batch item
            
            total_memory += adjusted_memory
        
        # Apply sharing benefits if enabled
        if self.enable_tensor_sharing:
            # Group by model type
            type_groups = {}
            for config in model_configs:
                model_type = config.get("model_type", "text_embedding")
                if model_type not in type_groups:
                    type_groups[model_type] = []
                type_groups[model_type].append(config)
            
            # Calculate memory savings
            savings_factor = 0.0
            for model_type, configs in type_groups.items():
                if len(configs) > 1:
                    # More models of same type = more sharing
                    if model_type == "text_embedding":
                        savings_factor += 0.25 * (len(configs) - 1)
                    elif model_type == "vision":
                        savings_factor += 0.3 * (len(configs) - 1)
                    elif model_type == "audio":
                        savings_factor += 0.15 * (len(configs) - 1)
                    elif model_type == "multimodal":
                        savings_factor += 0.1 * (len(configs) - 1)
            
            # Cap savings at 50%
            savings_factor = min(0.5, savings_factor)
            total_memory *= (1 - savings_factor)
        
        return total_memory
    
    def execute_models(
        self,
        model_configs: List[Dict[str, Any]],
        execution_strategy: str = "auto",
        optimization_goal: str = "latency",
        browser: Optional[str] = None,
        return_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Execute multiple models with the specified strategy.
        
        Args:
            model_configs: List of model configurations to execute
            execution_strategy: Strategy for execution ("parallel", "sequential", "batched", or "auto")
            optimization_goal: Metric to optimize ("latency", "throughput", or "memory")
            browser: Browser to use for execution (None for automatic selection)
            return_metrics: Whether to return detailed performance metrics
            
        Returns:
            Dictionary with execution results and metrics
        """
        if not self.initialized:
            logger.error("WebResourcePoolAdapter not initialized")
            return {"success": False, "error": "Not initialized"}
        
        if not self.resource_pool:
            logger.error("Resource pool not available")
            return {"success": False, "error": "Resource pool not available"}
        
        # Start timing
        start_time = time.time()
        
        # Automatic browser selection if not specified
        if not browser:
            # Use first model's type for browser selection
            if model_configs:
                model_type = model_configs[0].get("model_type", "text_embedding")
                browser = self.get_optimal_browser(model_type)
            else:
                browser = "chrome"  # Default
        
        # Automatic strategy selection if "auto"
        if execution_strategy == "auto":
            execution_strategy = self.get_optimal_strategy(
                model_configs, 
                browser, 
                optimization_goal
            )
        
        logger.info(f"Executing {len(model_configs)} models with {execution_strategy} strategy on {browser}")
        
        # Load models from resource pool
        models = []
        model_inputs = []
        
        for config in model_configs:
            model_type = config.get("model_type", "text_embedding")
            model_name = config.get("model_name", "")
            batch_size = config.get("batch_size", 1)
            
            # Convert model_type if needed
            if model_type == "text_embedding":
                resource_pool_type = "text" 
            elif model_type == "text_generation":
                resource_pool_type = "text"
            else:
                resource_pool_type = model_type
            
            # Create hardware preferences
            hw_preferences = {
                "browser": browser,
                "priority_list": ["webgpu", "webnn", "cpu"],
            }
            
            # Override with WebNN for text if supported
            if model_type in ["text_embedding", "text_generation"] and \
               browser in self.browser_capabilities and \
               self.browser_capabilities[browser].get("webnn", False):
                hw_preferences["priority_list"] = ["webnn", "webgpu", "cpu"]
            
            try:
                # Get model from resource pool
                model = self.resource_pool.get_model(
                    model_type=resource_pool_type,
                    model_name=model_name,
                    hardware_preferences=hw_preferences
                )
                
                if model:
                    models.append(model)
                    
                    # Create placeholder input based on model type
                    # In a real implementation, these would be actual inputs
                    if model_type == "text_embedding" or model_type == "text_generation":
                        input_data = {
                            "input_ids": [101, 2023, 2003, 1037, 3231, 102] * batch_size,
                            "attention_mask": [1, 1, 1, 1, 1, 1] * batch_size
                        }
                    elif model_type == "vision":
                        input_data = {
                            "pixel_values": [[[0.5 for _ in range(224)] for _ in range(224)] for _ in range(3)]
                        }
                    elif model_type == "audio":
                        input_data = {
                            "input_features": [[0.1 for _ in range(80)] for _ in range(3000)]
                        }
                    else:
                        input_data = {"placeholder": True}
                    
                    model_inputs.append((model, input_data))
                else:
                    logger.error(f"Failed to load model: {model_name} ({resource_pool_type})")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                traceback.print_exc()
        
        # Execute based on strategy
        if execution_strategy == "parallel":
            # Parallel execution
            execution_start = time.time()
            
            # Set up tensor sharing if enabled
            if self.enable_tensor_sharing:
                self._setup_tensor_sharing(model_configs, models)
            
            model_results = self.resource_pool.execute_concurrent([
                (model, inputs) for model, inputs in model_inputs
            ])
            
            execution_time = time.time() - execution_start
            
            # Calculate metrics
            throughput = len(model_results) / (execution_time if execution_time > 0 else 0.001)
            latency = execution_time * 1000  # Convert to ms
            
            # Get memory usage from resource pool metrics
            metrics = self.resource_pool.get_metrics() if return_metrics else {}
            memory_usage = metrics.get("base_metrics", {}).get("peak_memory_usage", 0)
            
            # Clean up tensor sharing if enabled
            if self.enable_tensor_sharing:
                self._cleanup_tensor_sharing(models)
            
        elif execution_strategy == "sequential":
            # Sequential execution
            execution_start = time.time()
            model_results = []
            
            for model, inputs in model_inputs:
                model_start = time.time()
                result = model(inputs)
                model_time = time.time() - model_start
                
                # Add timing information to result
                if isinstance(result, dict):
                    result["execution_time_ms"] = model_time * 1000
                else:
                    result = {"result": result, "execution_time_ms": model_time * 1000}
                
                model_results.append(result)
            
            execution_time = time.time() - execution_start
            
            # Calculate metrics
            throughput = len(model_results) / (execution_time if execution_time > 0 else 0.001)
            latency = execution_time * 1000  # Convert to ms
            
            # Get memory usage from resource pool metrics
            metrics = self.resource_pool.get_metrics() if return_metrics else {}
            memory_usage = metrics.get("base_metrics", {}).get("peak_memory_usage", 0)
            
        else:  # batched
            # Get batch configuration
            batch_size = BROWSER_STRATEGY_PREFERENCES.get(browser, {}).get("batching_size", 4)
            
            # Set up tensor sharing if enabled
            if self.enable_tensor_sharing:
                self._setup_tensor_sharing(model_configs, models)
            
            # Create batches
            batches = []
            current_batch = []
            
            for item in model_inputs:
                current_batch.append(item)
                if len(current_batch) >= batch_size:
                    batches.append(current_batch)
                    current_batch = []
            
            # Add remaining items
            if current_batch:
                batches.append(current_batch)
            
            # Execute batches sequentially
            execution_start = time.time()
            model_results = []
            
            for batch in batches:
                # Execute batch in parallel
                batch_results = self.resource_pool.execute_concurrent([
                    (model, inputs) for model, inputs in batch
                ])
                model_results.extend(batch_results)
            
            execution_time = time.time() - execution_start
            
            # Calculate metrics
            throughput = len(model_results) / (execution_time if execution_time > 0 else 0.001)
            latency = execution_time * 1000  # Convert to ms
            
            # Get memory usage from resource pool metrics
            metrics = self.resource_pool.get_metrics() if return_metrics else {}
            memory_usage = metrics.get("base_metrics", {}).get("peak_memory_usage", 0)
            
            # Clean up tensor sharing if enabled
            if self.enable_tensor_sharing:
                self._cleanup_tensor_sharing(models)
        
        # Update execution statistics
        self.execution_stats["total_executions"] += 1
        self.execution_stats["browser_executions"][browser] = self.execution_stats["browser_executions"].get(browser, 0) + 1
        self.execution_stats["strategy_executions"][execution_strategy] = self.execution_stats["strategy_executions"].get(execution_strategy, 0) + 1
        
        # Create result
        result = {
            "success": all(result.get("success", False) for result in model_results),
            "execution_strategy": execution_strategy,
            "browser": browser,
            "model_count": len(model_configs),
            "execution_time_ms": execution_time * 1000,
            "throughput": throughput,
            "latency": latency,
            "memory_usage": memory_usage,
            "model_results": model_results,
            "total_time_ms": (time.time() - start_time) * 1000
        }
        
        # Add detailed metrics if requested
        if return_metrics:
            result["detailed_metrics"] = {
                "browser_capabilities": self.browser_capabilities.get(browser, {}),
                "tensor_sharing_enabled": self.enable_tensor_sharing,
                "strategy_optimization_enabled": self.enable_strategy_optimization,
                "resource_pool_metrics": metrics,
                "execution_stats": self.execution_stats
            }
        
        return result
    
    def _setup_tensor_sharing(self, model_configs: List[Dict[str, Any]], models: List[Any]) -> None:
        """
        Set up tensor sharing between models.
        
        Args:
            model_configs: List of model configurations
            models: List of loaded models
        """
        if not self.enable_tensor_sharing or not self.resource_pool:
            return
        
        try:
            # Group models by type
            type_groups = {}
            for i, config in enumerate(model_configs):
                model_type = config.get("model_type", "text_embedding")
                if model_type not in type_groups:
                    type_groups[model_type] = []
                type_groups[model_type].append((i, config))
            
            # Set up sharing for each group
            sharing_count = 0
            total_models = len(models)
            memory_saved = 0
            
            for model_type, configs in type_groups.items():
                if len(configs) <= 1:
                    continue  # No sharing possible with just one model
                
                # Get model indices
                model_indices = [i for i, _ in configs]
                
                # Set up sharing for compatible models
                if model_type in ["text_embedding", "text_generation"]:
                    # Share embeddings between text models
                    if hasattr(self.resource_pool, "setup_tensor_sharing"):
                        sharing_result = self.resource_pool.setup_tensor_sharing(
                            models=[models[i] for i in model_indices],
                            sharing_type="text_embedding"
                        )
                        if sharing_result.get("success", False):
                            sharing_count += len(model_indices)
                            memory_saved += sharing_result.get("memory_saved", 0)
                            logger.debug(f"Set up tensor sharing for {len(model_indices)} {model_type} models")
                
                elif model_type == "vision":
                    # Share image embeddings between vision models
                    if hasattr(self.resource_pool, "setup_tensor_sharing"):
                        sharing_result = self.resource_pool.setup_tensor_sharing(
                            models=[models[i] for i in model_indices],
                            sharing_type="vision_embedding"
                        )
                        if sharing_result.get("success", False):
                            sharing_count += len(model_indices)
                            memory_saved += sharing_result.get("memory_saved", 0)
                            logger.debug(f"Set up tensor sharing for {len(model_indices)} {model_type} models")
                
                elif model_type == "audio":
                    # Share audio embeddings between audio models
                    if hasattr(self.resource_pool, "setup_tensor_sharing"):
                        sharing_result = self.resource_pool.setup_tensor_sharing(
                            models=[models[i] for i in model_indices],
                            sharing_type="audio_embedding"
                        )
                        if sharing_result.get("success", False):
                            sharing_count += len(model_indices)
                            memory_saved += sharing_result.get("memory_saved", 0)
                            logger.debug(f"Set up tensor sharing for {len(model_indices)} {model_type} models")
            
            # Update statistics
            self.execution_stats["tensor_sharing_stats"]["models_sharing_tensors"] += sharing_count
            self.execution_stats["tensor_sharing_stats"]["total_models"] += total_models
            self.execution_stats["tensor_sharing_stats"]["memory_saved_mb"] += memory_saved
            
            if sharing_count > 0:
                logger.info(f"Set up tensor sharing for {sharing_count}/{total_models} models, saving {memory_saved:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error setting up tensor sharing: {e}")
            traceback.print_exc()
    
    def _cleanup_tensor_sharing(self, models: List[Any]) -> None:
        """
        Clean up tensor sharing between models.
        
        Args:
            models: List of models with shared tensors
        """
        if not self.enable_tensor_sharing or not self.resource_pool:
            return
        
        try:
            if hasattr(self.resource_pool, "cleanup_tensor_sharing"):
                self.resource_pool.cleanup_tensor_sharing(models)
                logger.debug(f"Cleaned up tensor sharing for {len(models)} models")
        except Exception as e:
            logger.error(f"Error cleaning up tensor sharing: {e}")
            traceback.print_exc()
    
    def compare_strategies(
        self,
        model_configs: List[Dict[str, Any]],
        browser: Optional[str] = None,
        optimization_goal: str = "latency"
    ) -> Dict[str, Any]:
        """
        Compare different execution strategies for a set of models.
        
        Args:
            model_configs: List of model configurations to execute
            browser: Browser to use for execution (None for automatic selection)
            optimization_goal: Metric to optimize ("latency", "throughput", or "memory")
            
        Returns:
            Dictionary with comparison results
        """
        if not self.initialized:
            logger.error("WebResourcePoolAdapter not initialized")
            return {"success": False, "error": "Not initialized"}
        
        logger.info(f"Comparing execution strategies for {len(model_configs)} models")
        
        # Automatic browser selection if not specified
        if not browser:
            # Use first model's type for browser selection
            if model_configs:
                model_type = model_configs[0].get("model_type", "text_embedding")
                browser = self.get_optimal_browser(model_type)
            else:
                browser = "chrome"  # Default
        
        # Define strategies to compare
        strategies = ["parallel", "sequential", "batched"]
        results = {}
        
        # Execute with each strategy
        for strategy in strategies:
            logger.info(f"Testing {strategy} strategy on {browser}")
            result = self.execute_models(
                model_configs=model_configs,
                execution_strategy=strategy,
                optimization_goal=optimization_goal,
                browser=browser,
                return_metrics=False
            )
            results[strategy] = result
        
        # Get auto-recommended strategy
        logger.info(f"Testing auto-recommended strategy on {browser}")
        recommended_strategy = self.get_optimal_strategy(model_configs, browser, optimization_goal)
        
        # Use existing result if we already tested the recommended strategy
        if recommended_strategy in results:
            recommended_result = results[recommended_strategy]
        else:
            recommended_result = self.execute_models(
                model_configs=model_configs,
                execution_strategy=recommended_strategy,
                optimization_goal=optimization_goal,
                browser=browser,
                return_metrics=False
            )
            results[recommended_strategy] = recommended_result
        
        # Identify best strategy based on optimization goal
        best_strategy = None
        best_value = None
        
        if optimization_goal == "throughput":
            # Higher throughput is better
            for strategy, result in results.items():
                value = result.get("throughput", 0)
                if best_value is None or value > best_value:
                    best_value = value
                    best_strategy = strategy
        else:  # latency or memory
            # Lower values are better
            metric_key = "latency" if optimization_goal == "latency" else "memory_usage"
            for strategy, result in results.items():
                value = result.get(metric_key, float('inf'))
                if best_value is None or value < best_value:
                    best_value = value
                    best_strategy = strategy
        
        # Check if recommendation matches empirical best
        recommendation_accuracy = recommended_strategy == best_strategy
        
        # Create comparison result
        comparison_result = {
            "success": True,
            "model_count": len(model_configs),
            "browser": browser,
            "optimization_goal": optimization_goal,
            "best_strategy": best_strategy,
            "recommended_strategy": recommended_strategy,
            "recommendation_accuracy": recommendation_accuracy,
            "strategy_results": {
                strategy: {
                    "throughput": result.get("throughput", 0),
                    "latency": result.get("latency", 0),
                    "memory_usage": result.get("memory_usage", 0),
                    "execution_time_ms": result.get("execution_time_ms", 0),
                    "success": result.get("success", False)
                }
                for strategy, result in results.items()
            }
        }
        
        # Add strategy optimization impact
        if best_strategy and optimization_goal == "throughput":
            # Find worst throughput
            worst_strategy = min(strategies, key=lambda s: results[s].get("throughput", 0))
            worst_value = results[worst_strategy].get("throughput", 0)
            
            if worst_value > 0:
                improvement_percent = (best_value - worst_value) / worst_value * 100
                comparison_result["throughput_improvement_percent"] = improvement_percent
                logger.info(f"Throughput improvement: {improvement_percent:.1f}% ({best_strategy} vs {worst_strategy})")
        
        elif best_strategy and optimization_goal == "latency":
            # Find worst latency
            worst_strategy = max(strategies, key=lambda s: results[s].get("latency", float('inf')))
            worst_value = results[worst_strategy].get("latency", float('inf'))
            
            if worst_value > 0 and best_value > 0:
                improvement_percent = (worst_value - best_value) / worst_value * 100
                comparison_result["latency_improvement_percent"] = improvement_percent
                logger.info(f"Latency improvement: {improvement_percent:.1f}% ({best_strategy} vs {worst_strategy})")
        
        elif best_strategy and optimization_goal == "memory":
            # Find worst memory usage
            worst_strategy = max(strategies, key=lambda s: results[s].get("memory_usage", float('inf')))
            worst_value = results[worst_strategy].get("memory_usage", float('inf'))
            
            if worst_value > 0 and best_value > 0:
                improvement_percent = (worst_value - best_value) / worst_value * 100
                comparison_result["memory_improvement_percent"] = improvement_percent
                logger.info(f"Memory improvement: {improvement_percent:.1f}% ({best_strategy} vs {worst_strategy})")
        
        return comparison_result
    
    def get_browser_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detected browser capabilities.
        
        Returns:
            Dictionary with browser capabilities
        """
        if not self.browser_capabilities and self.browser_capability_detection and self.resource_pool:
            self._detect_browser_capabilities()
        
        return self.browser_capabilities
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        return self.execution_stats
    
    def close(self) -> bool:
        """
        Close the adapter and release resources.
        
        Returns:
            Success status
        """
        success = True
        
        # Close resource pool
        if self.resource_pool:
            try:
                logger.info("Closing resource pool")
                pool_success = self.resource_pool.close()
                if not pool_success:
                    logger.error("Error closing resource pool")
                    success = False
            except Exception as e:
                logger.error(f"Exception closing resource pool: {e}")
                traceback.print_exc()
                success = False
        
        logger.info(f"WebResourcePoolAdapter closed (success={'yes' if success else 'no'})")
        return success


# Example usage
if __name__ == "__main__":
    # Configure detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting WebResourcePoolAdapter example")
    
    # Create the adapter
    adapter = WebResourcePoolAdapter(
        max_connections=2,
        enable_tensor_sharing=True,
        enable_strategy_optimization=True,
        browser_capability_detection=True,
        verbose=True
    )
    
    # Initialize
    success = adapter.initialize()
    if not success:
        logger.error("Failed to initialize adapter")
        sys.exit(1)
    
    try:
        # Get browser capabilities
        capabilities = adapter.get_browser_capabilities()
        logger.info(f"Detected {len(capabilities)} browsers with capabilities")
        
        for browser, caps in capabilities.items():
            logger.info(f"{browser}: WebGPU={caps.get('webgpu', False)}, WebNN={caps.get('webnn', False)}")
        
        # Define model configurations for testing
        model_configs = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
            {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1}
        ]
        
        # Get optimal browser for text embedding
        optimal_browser = adapter.get_optimal_browser("text_embedding")
        logger.info(f"Optimal browser for text embedding: {optimal_browser}")
        
        # Get optimal strategy
        optimal_strategy = adapter.get_optimal_strategy(model_configs, optimal_browser, "throughput")
        logger.info(f"Optimal strategy: {optimal_strategy}")
        
        # Execute models with automatic strategy selection
        logger.info("Executing models with automatic strategy selection")
        result = adapter.execute_models(
            model_configs=model_configs,
            execution_strategy="auto",
            optimization_goal="throughput",
            browser=optimal_browser
        )
        
        logger.info(f"Execution complete with strategy: {result['execution_strategy']}")
        logger.info(f"Throughput: {result['throughput']:.2f} items/sec")
        logger.info(f"Latency: {result['latency']:.2f} ms")
        logger.info(f"Memory usage: {result['memory_usage']:.2f} MB")
        
        # Compare execution strategies
        logger.info("Comparing execution strategies")
        comparison = adapter.compare_strategies(
            model_configs=model_configs,
            browser=optimal_browser,
            optimization_goal="throughput"
        )
        
        logger.info(f"Best strategy: {comparison['best_strategy']}")
        logger.info(f"Recommended strategy: {comparison['recommended_strategy']}")
        logger.info(f"Recommendation accuracy: {comparison['recommendation_accuracy']}")
        
        # Get execution statistics
        stats = adapter.get_execution_statistics()
        logger.info(f"Total executions: {stats['total_executions']}")
        logger.info(f"Browser executions: {stats['browser_executions']}")
        logger.info(f"Strategy executions: {stats['strategy_executions']}")
        logger.info(f"Tensor sharing stats: {stats['tensor_sharing_stats']}")
        
    finally:
        # Close the adapter
        adapter.close()
        logger.info("WebResourcePoolAdapter example completed")