#!/usr/bin/env python3
"""
Cross-Browser Model Sharding Implementation (March 2025)

This module extends the model sharding functionality to work specifically across
different browser types, enabling specialized optimizations for each browser:

    - Run large models distributed across Chrome, Firefox, Edge, and Safari
    - Leverage browser-specific optimizations for different model components
    - Utilize specialized hardware capabilities of each browser
    - Efficiently route tensor operations to the most suitable browser
    - Provide unified interface for cross-browser model execution

Usage:
    python cross_browser_model_sharding.py --model llama-70b --browsers chrome,firefox,edge --test inference
"""

from ipfs_accelerate_py.anyio_helpers import gather, wait_for
import os
import sys
import time
import json
import argparse
import anyio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Import base model sharding functionality
try:
    from fixed_web_platform.model_sharding import (
        ModelShardingManager,
        create_model_shards,
        shard_model_for_inference,
        create_sharding_config
    )

    from fixed_web_platform.unified_framework.model_sharding import (
        ModelShardingManager as UnifiedModelShardingManager,
        ShardConfiguration,
        BrowserTabShardingIntegration
    )
except ImportError:
    # Mock implementations for testing
    class ModelShardingManager:
        pass
    
    class UnifiedModelShardingManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        
        def initialize_sharding(self):
            return True
        
        def run_inference_sharded(self, inputs):
            return {"output": f"Mock output for {inputs.get('text', '')}", "browser": "mock"}
        
        def get_sharding_status(self):
            return {"status": "active"}
        
        def shutdown_sharding(self):
            return True
    
    def create_model_shards(*args, **kwargs):
        return []
    
    def shard_model_for_inference(*args, **kwargs):
        return {}
    
    def create_sharding_config(*args, **kwargs):
        return {}
    
    class ShardConfiguration:
        pass
    
    class BrowserTabShardingIntegration:
        pass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrowserCapabilities:
    """
    Defines capabilities and optimizations for different browsers.
    """
    
    # Browser capabilities for different model components
    CAPABILITIES = {
        "chrome": {
            "precision": ["float32", "float16", "int8", "int4"],
            "optimizations": ["parallel_tensor_ops", "webgpu_compute", "shader_precompile"],
            "best_for": ["vision", "multimodal"],
            "max_memory_gb": 4.0,
            "parallel_connections": 6
        },
        "firefox": {
            "precision": ["float32", "float16", "int8", "int4"],
            "optimizations": ["audio_compute_shaders", "webgpu_optimized_audio"],
            "best_for": ["audio", "speech"],
            "max_memory_gb": 3.5,
            "parallel_connections": 8
        },
        "edge": {
            "precision": ["float32", "float16", "int8", "int4"],
            "optimizations": ["webnn_integration", "text_optimized_kernels"],
            "best_for": ["text", "embedding"],
            "max_memory_gb": 4.0,
            "parallel_connections": 6
        },
        "safari": {
            "precision": ["float32", "float16", "int8"],
            "optimizations": ["metal_integration", "power_efficiency"],
            "best_for": ["vision", "mobile"],
            "max_memory_gb": 2.5,
            "parallel_connections": 4
        }
    }
    
    @classmethod
    def get_browser_for_component(cls, component_type: str) -> str:
        """
        Get the best browser for a specific model component.
        
        Args:
            component_type: Type of model component (text, vision, audio, embedding, attention, feedforward)
                           
        Returns:
            Name of the best browser for this component type
        """
        component_browser_map = {
            "text": "edge",
            "vision": "chrome",
            "audio": "firefox",
            "speech": "firefox",
            "embedding": "edge",
            "attention": "chrome",
            "feedforward": "edge",
            "lm_head": "edge",
            "multimodal": "chrome",
            "mobile": "safari"
        }
    
        return component_browser_map.get(component_type, "chrome")  # Default to Chrome
    
    @classmethod
    def get_browser_memory_limit(cls, browser: str) -> float:
        """
        Get the estimated memory limit for a browser.
        
        Args:
            browser: Name of the browser
            
        Returns:
            Estimated memory limit in GB
        """
        return cls.CAPABILITIES.get(browser, {}).get("max_memory_gb", 2.0)
    
    @classmethod
    def get_browser_capabilities(cls, browser: str) -> Dict[str, Any]:
        """Get all capabilities for a specific browser."""
        return cls.CAPABILITIES.get(browser, {})
    
    @classmethod
    def is_capability_supported(cls, browser: str, capability: str) -> bool:
        """Check if a specific capability is supported by a browser."""
        capabilities = cls.CAPABILITIES.get(browser, {})
        
        # Check in optimizations:
        if capability in capabilities.get("optimizations", []):
            return True
            
        # Check in precision
        if capability in capabilities.get("precision", []):
            return True
        
        # Check in best_for
        if capability in capabilities.get("best_for", []):
            return True
            
        return False

class CrossBrowserModelShardingManager:
    """
    Manager for sharding models across different browsers to leverage
    browser-specific optimizations for different model components.
    """
    
    def __init__(self, 
                 model_name: str,
                 browsers: List[str] = ["chrome", "firefox", "edge"],
                 shard_type: str = "optimal",
                 num_shards: Optional[int] = None,
                 model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize cross-browser model sharding manager.
        
        Args:
            model_name: Name of the model to shard
            browsers: List of browsers to use
            shard_type: Sharding strategy (optimal, layer, browser)
            num_shards: Total number of shards (if None, determined automatically)
            model_config: Additional model configuration
        """
        self.model_name = model_name
        self.browsers = browsers
        self.shard_type = shard_type
        self.model_config = model_config or {}
        
        # Determine model characteristics
        self.model_characteristics = self._detect_model_characteristics()
        
        # Determine optimal number of shards if not specified:
        if num_shards is None:
            self.num_shards = self._determine_optimal_shard_count()
        else:
            self.num_shards = max(len(browsers), num_shards)
        
        # Create browser-specific shard assignments
        self.browser_shards = self._create_browser_shards()
        
        # Store browser managers
        self.browser_managers = {}
        
        # Store shard state
        self.initialized = False
        self.active_browsers = set()
        
        logger.info(f"Initialized cross-browser sharding for {model_name} across {', '.join(browsers)}")
        logger.info(f"Using {self.num_shards} shards with {shard_type} sharding strategy")
    
    def _detect_model_characteristics(self) -> Dict[str, Any]:
        """Detect model characteristics for optimal sharding."""
        # Extract model type and size from name
        model_name_lower = self.model_name.lower()
        
        characteristics = {
            "type": "unknown",
            "size_class": "small",
            "parameter_count": 0,
            "model_size_gb": 0,
            "components": [],
            "primary_modality": "text"
        }
        
        # Detect model type
        if "llama" in model_name_lower:
            characteristics["type"] = "llm"
            characteristics["primary_modality"] = "text"
            characteristics["components"] = ["embedding", "attention", "feedforward", "lm_head"]
            
            # Detect model size
            if "70b" in model_name_lower:
                characteristics["size_class"] = "xxlarge"
                characteristics["parameter_count"] = 70
                characteristics["model_size_gb"] = 140  # Approximate FP16 size
            elif "13b" in model_name_lower:
                characteristics["size_class"] = "large"
                characteristics["parameter_count"] = 13
                characteristics["model_size_gb"] = 26
            elif "7b" in model_name_lower:
                characteristics["size_class"] = "medium"
                characteristics["parameter_count"] = 7
                characteristics["model_size_gb"] = 14
                
        elif "clip" in model_name_lower:
            characteristics["type"] = "multimodal"
            characteristics["primary_modality"] = "vision"
            characteristics["components"] = ["vision_encoder", "text_encoder", "projection"]
            
            # Default size estimate
            characteristics["size_class"] = "medium"
            characteristics["parameter_count"] = 1
            characteristics["model_size_gb"] = 2
            
        elif "whisper" in model_name_lower:
            characteristics["type"] = "speech"
            characteristics["primary_modality"] = "audio"
            characteristics["components"] = ["audio_encoder", "text_decoder", "lm_head"]
            
            # Determine size
            if "large" in model_name_lower:
                characteristics["size_class"] = "large"
                characteristics["parameter_count"] = 1.5
                characteristics["model_size_gb"] = 3
            else:
                characteristics["size_class"] = "medium"
                characteristics["parameter_count"] = 0.8
                characteristics["model_size_gb"] = 1.6
                
        elif "t5" in model_name_lower:
            characteristics["type"] = "seq2seq"
            characteristics["primary_modality"] = "text"
            characteristics["components"] = ["encoder", "decoder", "lm_head"]
            
            # Determine size
            if "xxl" in model_name_lower or "11b" in model_name_lower:
                characteristics["size_class"] = "xlarge"
                characteristics["parameter_count"] = 11
                characteristics["model_size_gb"] = 22
            elif "xl" in model_name_lower or "3b" in model_name_lower:
                characteristics["size_class"] = "large"
                characteristics["parameter_count"] = 3
                characteristics["model_size_gb"] = 6
            else:
                characteristics["size_class"] = "medium"
                characteristics["parameter_count"] = 0.8
                characteristics["model_size_gb"] = 1.6
                
        # Default for unknown models
        if characteristics["type"] == "unknown":
            characteristics["size_class"] = "medium"
            characteristics["parameter_count"] = 1
            characteristics["model_size_gb"] = 2
            characteristics["components"] = ["embedding", "layers", "output"]
                
        return characteristics
    
    def _determine_optimal_shard_count(self) -> int:
        """Determine optimal number of shards based on model and browsers."""
        model_size_gb = self.model_characteristics["model_size_gb"]
        
        # Calculate total available memory across browsers
        total_browser_memory = sum(
            BrowserCapabilities.get_browser_memory_limit(browser)
            for browser in self.browsers
        )
        
        # Calculate minimum shards needed to fit in memory
        min_shards_for_memory = max(1, int(model_size_gb / total_browser_memory * 2))
        
        # Consider browser count
        browser_count = len(self.browsers)
        
        # Final shard count: at least one per browser, and enough to fit model
        optimal_count = max(browser_count, min_shards_for_memory)
        
        # Limit to reasonable range
        return min(32, max(browser_count, optimal_count))
        
    def _create_browser_shards(self) -> Dict[str, List[int]]:
        """Create optimal shard-to-browser assignments."""
        browser_shards = {browser: [] for browser in self.browsers}
        
        if self.shard_type == "browser":
            # Simple assignment: divide shards evenly across browsers
            shards_per_browser = self.num_shards // len(self.browsers)
            remainder = self.num_shards % len(self.browsers)
            
            shard_index = 0
            for i, browser in enumerate(self.browsers):
                # Assign extra shard to some browsers if needed
                browser_shard_count = shards_per_browser + (1 if i < remainder else 0)
                
                # Assign consecutive shards to this browser:
                for j in range(browser_shard_count):
                    browser_shards[browser].append(shard_index)
                    shard_index += 1
                    
        elif self.shard_type == "optimal":
            # Component-based assignment to leverage browser strengths
            components = self.model_characteristics["components"]
            
            # Get optimal browser for each component
            component_browser_map = {}
            for component in components:
                best_browser = BrowserCapabilities.get_browser_for_component(component)
                if best_browser in self.browsers:
                    component_browser_map[component] = best_browser
                else:
                    # Fall back to first available browser if best isn't available
                    component_browser_map[component] = self.browsers[0]
                    
            # Calculate layers per shard and distribute across browsers
            total_layers = max(12, self.model_characteristics["parameter_count"] * 2)
            layers_per_browser = {}
            
            # Default distribution: equal layers per browser
            for browser in self.browsers:
                layers_per_browser[browser] = []
                
            # Assign layers based on browser capabilities
            layer_index = 0
            while layer_index < total_layers:
                for browser in self.browsers:
                    # Skip if we've assigned all layers:
                    if layer_index >= total_layers:
                        break
                        
                    # Assign more layers to browsers with higher memory limits
                    browser_memory = BrowserCapabilities.get_browser_memory_limit(browser)
                    layers_to_assign = max(1, int(browser_memory / 2))
                    
                    # Don't exceed total
                    layers_to_assign = min(layers_to_assign, total_layers - layer_index)
                    
                    # Assign layers
                    for _ in range(layers_to_assign):
                        if layer_index < total_layers:
                            layers_per_browser[browser].append(layer_index)
                            layer_index += 1
            
            # Now map layers to shards
            layers_per_shard = max(1, int(total_layers / self.num_shards))
            
            # First, determine how many shards per browser based on layer allocation
            shards_per_browser = {
                browser: max(1, len(layers) // layers_per_shard)
                for browser, layers in layers_per_browser.items()
            }
            
            # Adjust to ensure we have exactly num_shards in total
            total_shards = sum(shards_per_browser.values())
            while total_shards != self.num_shards:
                if total_shards < self.num_shards:
                    # Add shards to browser with most layers per shard
                    browser_to_add = max(
                        self.browsers,
                        key=lambda b: len(layers_per_browser[b]) / (shards_per_browser[b] + 0.001)
                    )
                    shards_per_browser[browser_to_add] += 1
                else:
                    # Remove shard from browser with fewest layers per shard
                    browser_to_remove = min(
                        self.browsers,
                        key=lambda b: len(layers_per_browser[b]) / (shards_per_browser[b] + 0.001)
                    )
                    if shards_per_browser[browser_to_remove] > 1:
                        shards_per_browser[browser_to_remove] -= 1
                    else:
                        # Can't reduce further, adjust another browser
                        browsers_with_multiple_shards = [
                            b for b in self.browsers if shards_per_browser[b] > 1
                        ]
                        if browsers_with_multiple_shards:
                            browser_to_remove = browsers_with_multiple_shards[0]
                            shards_per_browser[browser_to_remove] -= 1
                        else:
                            # Can't satisfy the constraint, break
                            break
                
                total_shards = sum(shards_per_browser.values())
            
            # Now assign shard indices to browsers
            shard_index = 0
            for browser, shard_count in shards_per_browser.items():
                for _ in range(shard_count):
                    browser_shards[browser].append(shard_index)
                    shard_index += 1
                    
        else:  # layer-based, the default
            # Even distribution, but with consideration of browser capabilities
            memory_limits = {
                browser: BrowserCapabilities.get_browser_memory_limit(browser)
                for browser in self.browsers
            }
            
            # Total available memory
            total_memory = sum(memory_limits.values())
            
            # Distribute shards proportionally to memory
            remaining_shards = self.num_shards
            assigned_shards = {browser: 0 for browser in self.browsers}
            
            for browser in self.browsers:
                # Calculate proportion of memory
                memory_proportion = memory_limits[browser] / total_memory
                
                # Calculate shards for this browser
                browser_shards_count = max(1, int(self.num_shards * memory_proportion))
                if browser_shards_count > remaining_shards:
                    browser_shards_count = remaining_shards
                    
                assigned_shards[browser] = browser_shards_count
                remaining_shards -= browser_shards_count
                
            # Distribute any remaining shards to browsers with most memory
            if remaining_shards > 0:
                browsers_by_memory = sorted(
                    self.browsers,
                    key=lambda b: memory_limits[b],
                    reverse=True
                )
                
                for browser in browsers_by_memory:
                    if remaining_shards <= 0:
                        break
                    assigned_shards[browser] += 1
                    remaining_shards -= 1
            
            # Now assign specific shard indices
            shard_index = 0
            for browser, shard_count in assigned_shards.items():
                for _ in range(shard_count):
                    browser_shards[browser].append(shard_index)
                    shard_index += 1
                    
        return browser_shards
    
    async def initialize(self) -> bool:
        """
        Initialize all browser shards.
        
        Returns:
            Whether initialization was successful
        """
        logger.info(f"Initializing {self.num_shards} shards across {len(self.browsers)} browsers")
        
        # Initialize browser managers
        init_tasks = []
        
        for browser in self.browsers:
            shard_indices = self.browser_shards[browser]
            if not shard_indices:
                logger.warning(f"No shards assigned to {browser}, skipping")
                continue
                
            # Create initialization task for this browser
            init_tasks.append(self._initialize_browser_shards(browser, shard_indices))
        
        # Wait for all initializations to complete
        results = await gather(*init_tasks, return_exceptions=True)
        
        # Check for success
        success_count = sum(1 for result in results if result is True)
        
        # Update state
        self.initialized = success_count > 0
        
        logger.info(f"Initialization complete: {success_count}/{len(init_tasks)} browsers initialized")
        
        return self.initialized
    
    async def _initialize_browser_shards(self, browser: str, shard_indices: List[int]) -> bool:
        """Initialize shards for a specific browser."""
        logger.info(f"Initializing {len(shard_indices)} shards for {browser}")
        
        # Create configuration for unified sharding manager
        browser_model_config = {
            **self.model_config,
            "browser_type": browser,
            "browser_capabilities": BrowserCapabilities.get_browser_capabilities(browser),
            "model_components": self._get_browser_components(browser)
        }
        
        # Create unified manager for this browser
        browser_manager = UnifiedModelShardingManager(
            model_name=self.model_name,
            num_shards=len(shard_indices),
            shard_type="layer",  # Always layer-based for unified manager
            model_config=browser_model_config
        )
        
        # Initialize the browser manager
        success = browser_manager.initialize_sharding()
        
        if success:
            # Store manager for future use
            self.browser_managers[browser] = browser_manager
            self.active_browsers.add(browser)
            
            logger.info(f"Successfully initialized {browser} with {len(shard_indices)} shards")
        else:
            logger.warning(f"Failed to initialize {browser}")
            
        return success
    
    def _get_browser_components(self, browser: str) -> List[str]:
        """Get model components best suited for this browser."""
        all_components = self.model_characteristics.get("components", [])
        browser_modalities = BrowserCapabilities.get_browser_capabilities(browser).get("best_for", [])
        
        # Map components to browser modalities
        component_to_modality = {
            "embedding": "text",
            "attention": "text",
            "feedforward": "text",
            "lm_head": "text",
            "vision_encoder": "vision",
            "audio_encoder": "audio",
            "text_encoder": "text",
            "text_decoder": "text",
            "projection": "multimodal",
            "encoder": "text",
            "decoder": "text"
        }
        
        # Filter components best suited for this browser
        browser_components = [
            component for component in all_components
            if component_to_modality.get(component, "text") in browser_modalities
        ]
        
        # If no components match, include all components:
        if not browser_components:
            browser_components = all_components
            
        return browser_components
    
    async def run_inference(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference using cross-browser sharded model.
        
        Args:
            inputs: Input data for inference
            
        Returns:
            Inference results
        """
        if not self.initialized:
            raise RuntimeError("Model sharding not initialized")
            
        if not self.active_browsers:
            raise RuntimeError("No active browsers available")
            
        logger.info(f"Running cross-browser inference with {len(self.active_browsers)} browsers")
        
        # Start measuring time
        start_time = time.time()
        
        # Prepare and augment inputs for each browser
        browser_inputs = self._prepare_browser_inputs(inputs)
        
        # Create inference tasks for each browser
        inference_tasks = []
        
        for browser in self.active_browsers:
            browser_manager = self.browser_managers[browser]
            
            # Create inference task
            inference_tasks.append(
                self._run_browser_inference(
                    browser=browser,
                    manager=browser_manager,
                    inputs=browser_inputs[browser]
                )
            )
            
        # Wait for all inferences to complete
        browser_results = await gather(*inference_tasks, return_exceptions=True)
        
        # Process results and handle errors
        valid_results = []
        for result in browser_results:
            if isinstance(result, Exception):
                logger.error(f"Browser inference error: {result}")
            else:
                valid_results.append(result)
                
        # Check if we have enough results:
        if not valid_results:
            raise RuntimeError("All browser inferences failed")
            
        # Combine results from different browsers
        combined_result = self._combine_browser_results(valid_results)
        
        # Calculate overall inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Add cross-browser metrics to result
        combined_result["cross_browser_metrics"] = {
            "total_browsers": len(self.active_browsers),
            "successful_browsers": len(valid_results),
            "total_inference_time_ms": inference_time,
            "browser_count": {browser: len(self.browser_shards[browser]) for browser in self.active_browsers}
        }
        
        logger.info(f"Cross-browser inference complete in {inference_time:.1f}ms")
        
        return combined_result
    
    def _prepare_browser_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Prepare browser-specific inputs."""
        browser_inputs = {}
        
        for browser in self.active_browsers:
            # Create a copy to avoid modifying original
            browser_input = inputs.copy()
            
            # Add browser-specific metadata
            browser_input["_browser"] = browser
            browser_input["_shards"] = len(self.browser_shards[browser])
            browser_input["_capabilities"] = BrowserCapabilities.get_browser_capabilities(browser)
            
            browser_inputs[browser] = browser_input
            
        return browser_inputs
    
    async def _run_browser_inference(self, browser: str, manager: UnifiedModelShardingManager,
                                    inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference for a specific browser."""
        logger.info(f"Running inference on {browser} with {len(self.browser_shards[browser])} shards")
        
        try:
            # Call the unified manager's inference method
            result = manager.run_inference_sharded(inputs)
            
            # Add browser metadata to result
            result["browser"] = browser
            result["shard_count"] = len(self.browser_shards[browser])
            
            return result
        except Exception as e:
            logger.error(f"Error running inference on {browser}: {e}")
            raise
    
    def _combine_browser_results(self, browser_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from different browsers."""
        logger.info(f"Combining results from {len(browser_results)} browsers")
        
        # The actual combination logic would depend on the model architecture
        # For simplicity, we'll concatenate text outputs and take the first value for other outputs
        
        combined_output = ""
        combined_metadata = {}
        browser_outputs = {}
        
        for result in browser_results:
            browser = result.get("browser", "unknown")
            
            # Extract output
            if "output" in result:
                browser_outputs[browser] = result["output"]
                # Very basic combination - in reality, this would be much more sophisticated
                if not combined_output:
                    combined_output = result["output"]
                    
            # Collect metadata
            for key, value in result.items():
                if key not in ["output", "browser", "shard_count", "sharding_metrics"]:
                    combined_metadata[f"{browser}_{key}"] = value
        
        # Create combined result
        return {
            "output": combined_output,
            "browser_outputs": browser_outputs,
            "metadata": combined_metadata,
            "browsers_used": len(browser_results)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of cross-browser sharding."""
        active_browsers = len(self.active_browsers)
        total_browsers = len(self.browsers)
        
        # Get browser-specific statuses
        browser_status = {}
        for browser in self.browsers:
            if browser in self.browser_managers:
                browser_manager = self.browser_managers[browser]
                browser_status[browser] = browser_manager.get_sharding_status()
            else:
                browser_status[browser] = {"status": "not_initialized"}
        
        return {
            "model_name": self.model_name,
            "active_browsers": active_browsers,
            "total_browsers": total_browsers,
            "total_shards": self.num_shards,
            "shard_type": self.shard_type,
            "initialized": self.initialized,
            "browser_status": browser_status,
            "browser_shards": self.browser_shards,
            "model_size_gb": self.model_characteristics["model_size_gb"],
            "parameter_count": self.model_characteristics["parameter_count"],
            "model_type": self.model_characteristics["type"]
        }
    
    async def shutdown(self) -> bool:
        """
        Shutdown all browser shards.
        
        Returns:
            Whether shutdown was successful
        """
        logger.info(f"Shutting down cross-browser sharding for {self.model_name}")
        
        if not self.initialized:
            return True
            
        # Shutdown each browser manager
        shutdown_tasks = []
        
        for browser in list(self.active_browsers):
            browser_manager = self.browser_managers[browser]
            
            # Create shutdown task
            shutdown_tasks.append(self._shutdown_browser(browser, browser_manager))
            
        # Wait for all shutdowns to complete
        await gather(*shutdown_tasks, return_exceptions=True)
        
        # Update state
        self.initialized = False
        self.active_browsers = set()
        
        return True
    
    async def _shutdown_browser(self, browser: str, manager: UnifiedModelShardingManager) -> bool:
        """Shutdown a specific browser."""
        logger.info(f"Shutting down {browser}")
        
        try:
            success = manager.shutdown_sharding()
            
            if success:
                if browser in self.active_browsers:
                    self.active_browsers.remove(browser)
                    
                if browser in self.browser_managers:
                    del self.browser_managers[browser]
                    
            return success
        except Exception as e:
            logger.error(f"Error shutting down {browser}: {e}")
            return False

# Utility functions for testing and benchmarking:

async def test_cross_browser_initialization(model_name: str, browsers: List[str], verbose: bool = False) -> Dict[str, Any]:
    """
    Test cross-browser initialization.
    
    Args:
        model_name: Name of the model to test
        browsers: List of browsers to use
        verbose: Whether to print verbose output
        
    Returns:
        Test results
    """
    logger.info(f"Testing cross-browser initialization for {model_name}")
    
    # Create cross-browser sharding manager
    manager = CrossBrowserModelShardingManager(
        model_name=model_name,
        browsers=browsers,
        shard_type="optimal"
    )
    
    # Print configuration if verbose:
    if verbose:
        model_characteristics = manager.model_characteristics
        print(f"Model: {model_name}")
        print(f"Size: {model_characteristics['model_size_gb']:.1f} GB")
        print(f"Parameters: {model_characteristics['parameter_count']} billion")
        print(f"Type: {model_characteristics['type']}")
        print(f"Browsers: {', '.join(browsers)}")
        print(f"Shard count: {manager.num_shards}")
        print(f"Shard distribution:")
        for browser, shards in manager.browser_shards.items():
            print(f"  {browser}: {len(shards)} shards - {shards}")
    
    # Initialize sharding
    start_time = time.time()
    success = await manager.initialize()
    init_time = (time.time() - start_time) * 1000  # ms
    
    if verbose:
        print(f"Initialization {'succeeded' if success else 'failed'}")
        print(f"Initialization time: {init_time:.1f} ms")
        print(f"Active browsers: {', '.join(manager.active_browsers)}")
    
    # Get status after initialization
    status = manager.get_status()
    
    # Create test result
    result = {
        "model_name": model_name,
        "browsers": browsers,
        "initialization_success": success,
        "initialization_time_ms": init_time,
        "active_browsers": len(manager.active_browsers),
        "total_browsers": len(browsers),
        "shard_count": manager.num_shards,
        "shard_distribution": {b: len(s) for b, s in manager.browser_shards.items()},
        "test_status": "passed" if success else "failed"
    }
    
    # Clean up
    await manager.shutdown()
    
    return result

async def test_cross_browser_inference(model_name: str, browsers: List[str], verbose: bool = False) -> Dict[str, Any]:
    """
    Test cross-browser inference.
    
    Args:
        model_name: Name of the model to test
        browsers: List of browsers to use
        verbose: Whether to print verbose output
        
    Returns:
        Test results
    """
    logger.info(f"Testing cross-browser inference for {model_name}")
    
    # Create cross-browser sharding manager
    manager = CrossBrowserModelShardingManager(
        model_name=model_name,
        browsers=browsers,
        shard_type="optimal"
    )
    
    # Initialize sharding
    init_success = await manager.initialize()
    
    if not init_success:
        return {
            "model_name": model_name,
            "browsers": browsers,
            "test_status": "failed",
            "error": "Initialization failed"
        }
    
    # Create test input
    test_input = {
        "text": "This is a test input for cross-browser inference.",
        "max_length": 50,
        "temperature": 0.7
    }
    
    # Run inference
    try:
        start_time = time.time()
        result = await manager.run_inference(test_input)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        if verbose:
            print(f"Inference result: {result.get('output', '')}")
            print(f"Inference time: {inference_time:.1f} ms")
            print(f"Browsers used: {result.get('browsers_used', 0)}")
            
            # Print browser-specific outputs
            browser_outputs = result.get("browser_outputs", {})
            if browser_outputs:
                print("Browser outputs:")
                for browser, output in browser_outputs.items():
                    print(f"  {browser}: {output}")
        
        # Create test result
        test_result = {
            "model_name": model_name,
            "browsers": browsers,
            "inference_time_ms": inference_time,
            "browsers_used": result.get("browsers_used", 0),
            "output_length": len(result.get("output", "")),
            "cross_browser_metrics": result.get("cross_browser_metrics", {}),
            "test_status": "passed"
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        test_result = {
            "model_name": model_name,
            "browsers": browsers,
            "test_status": "failed",
            "error": str(e)
        }
    
    # Clean up
    await manager.shutdown()
    
    return test_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Browser Model Sharding")
    parser.add_argument("--model", choices=["llama", "whisper", "clip", "t5"], default="llama",
                      help="Model family to test")
    parser.add_argument("--size", default="7b",
                      help="Model size (e.g., 7b, 13b, 70b)")
    parser.add_argument("--browsers", default="chrome,firefox,edge",
                      help="Comma-separated list of browsers to use")
    parser.add_argument("--test", choices=["initialization", "inference"],
                      default="inference", help="Test to run")
    parser.add_argument("--verbose", action="store_true",
                      help="Show detailed output")
    parser.add_argument("--output", type=str,
                      help="Output file for test results (JSON)")
    
    args = parser.parse_args()
    
    # Build model name
    model_name = f"{args.model}-{args.size}"
    
    # Create browsers list
    browsers = args.browsers.split(",")
    
    # Run the specified test
    async def run_test():
        if args.test == "initialization":
            result = await test_cross_browser_initialization(model_name, browsers, args.verbose)
        else:  # inference
            result = await test_cross_browser_inference(model_name, browsers, args.verbose)
            
        # Save results if requested:
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
                print(f"Results saved to {args.output}")
                
        return result
        
    # Run the test
    result = anyio.run(run_test)