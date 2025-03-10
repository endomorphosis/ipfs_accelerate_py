#!/usr/bin/env python3
"""
Cross-Browser Model Sharding for WebNN/WebGPU Resource Pool

This module implements cross-browser model sharding, allowing large models to be split
across multiple browser instances for concurrent execution and to leverage browser-specific
optimizations.

Key features:
- Distributes model components across multiple browser types
- Leverages browser-specific optimizations (Firefox for audio, Edge for text, etc.)
- Enables running models too large for a single browser instance
- Manages cross-browser communication and synchronization
- Provides a unified interface for sharded model execution

Usage:
    from fixed_web_platform.cross_browser_model_sharding import ModelShardingManager
    
    # Create model sharding manager
    manager = ModelShardingManager(
        model_name="llama-7b",
        num_shards=4,
        shard_type="layer"
    )
    
    # Initialize sharding
    manager.initialize_sharding()
    
    # Run inference across shards
    result = manager.run_inference_sharded({"input_text": "Sample text"})
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple

# Import resource pool bridge
try:
    from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
except ImportError:
    # Use relative import as fallback
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from resource_pool_bridge import ResourcePoolBridgeIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShardedModelComponent:
    """
    Represents a sharded component of a model running in a specific browser.
    
    Each ShardedModelComponent manages a piece of the model that's executed in
    a specific browser optimized for that component type.
    """
    
    def __init__(self, component_id: str, model_type: str, model_name: str,
                 shard_index: int, shard_type: str, browser: str, platform: str,
                 resource_pool_integration: ResourcePoolBridgeIntegration):
        """
        Initialize a sharded model component.
        
        Args:
            component_id: Unique identifier for this component
            model_type: Type of model (e.g., 'text_embedding', 'vision', 'audio')
            model_name: Name of the model
            shard_index: Index of this shard
            shard_type: Type of sharding ('layer', 'attention', 'feedforward', etc.)
            browser: Browser to use ('chrome', 'firefox', 'edge', etc.)
            platform: Platform to use ('webgpu' or 'webnn')
            resource_pool_integration: ResourcePoolBridgeIntegration instance
        """
        self.component_id = component_id
        self.model_type = model_type
        self.model_name = model_name
        self.shard_index = shard_index
        self.shard_type = shard_type
        self.browser = browser
        self.platform = platform
        self.resource_pool = resource_pool_integration
        self.model = None
        self.connection_id = None
        self.is_initialized = False
        self.metrics = {
            'initialization_time': 0,
            'inference_time': 0,
            'throughput': 0,
            'memory_usage': 0
        }
    
    async def initialize(self):
        """Initialize this model component in its assigned browser."""
        if self.is_initialized:
            return True
        
        start_time = time.time()
        
        try:
            # Configure hardware preferences for this component
            hardware_preferences = {
                'priority_list': [self.platform, 'cpu'],
                'browser': self.browser,
                'precision': 16,  # Default to FP16 for good balance
                'mixed_precision': False,
                'enable_ipfs': True
            }
            
            # Add optimizations based on model type and browser
            self._add_component_optimizations(hardware_preferences)
            
            # Model ID includes shard information
            model_id = f"{self.model_type}:{self.model_name}:shard{self.shard_index}:{self.shard_type}"
            
            # Get model from resource pool
            logger.info(f"Initializing component {self.component_id} in {self.browser} browser")
            
            # Get optimal connection from resource pool
            connection_id, connection_info = self.resource_pool.get_optimal_browser_connection(
                self.model_type,
                self.platform,
                model_family=self.model_type,
                priority=10 # High priority for sharded components
            )
            
            if connection_id:
                self.connection_id = connection_id
                logger.info(f"Using existing connection {connection_id} for component {self.component_id}")
            
            # Create model with resource pool
            self.model = self.resource_pool.get_model(
                model_type=self.model_type,
                model_name=self.model_name,
                hardware_preferences=hardware_preferences
            )
            
            if not self.model:
                logger.error(f"Failed to initialize component {self.component_id}")
                return False
            
            # Track initialization time
            self.metrics['initialization_time'] = time.time() - start_time
            self.is_initialized = True
            
            logger.info(f"Component {self.component_id} initialized in {self.metrics['initialization_time']:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing component {self.component_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _add_component_optimizations(self, hardware_preferences):
        """Add component-specific optimizations based on model type and browser."""
        # For audio components in Firefox, enable compute shader optimizations
        if self.model_type == 'audio' and self.browser == 'firefox':
            hardware_preferences['compute_shader_optimized'] = True
            hardware_preferences['use_firefox_optimizations'] = True
        
        # For vision components in Chrome, enable shader precompilation
        elif self.model_type == 'vision' and self.browser == 'chrome':
            hardware_preferences['precompile_shaders'] = True
        
        # For text components in Edge with WebNN, no special optimizations needed
        elif self.model_type == 'text_embedding' and self.browser == 'edge' and self.platform == 'webnn':
            pass
        
        # For attention components, use specialized optimizations
        if self.shard_type == 'attention':
            hardware_preferences['kv_cache_optimization'] = True
        
        # For feedforward components, use specialized optimizations
        elif self.shard_type == 'feedforward':
            hardware_preferences['parallel_feedforward'] = True
            
        # For multimodal shard types, enable parallel loading
        if "multimodal" in self.model_type or self.shard_type == 'multimodal':
            hardware_preferences['parallel_loading'] = True
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs through this model component.
        
        Args:
            inputs: Input data for this component
            
        Returns:
            Processing results
        """
        if not self.is_initialized or not self.model:
            logger.error(f"Component {self.component_id} not initialized")
            return {'error': 'Component not initialized'}
        
        try:
            start_time = time.time()
            
            # Run inference on this component
            logger.debug(f"Running inference on component {self.component_id}")
            result = self.model(inputs)
            
            # Track performance metrics
            inference_time = time.time() - start_time
            self.metrics['inference_time'] = inference_time
            self.metrics['throughput'] = 1.0 / inference_time if inference_time > 0 else 0
            
            # Extract and store memory usage if available
            if isinstance(result, dict) and 'metrics' in result:
                memory_usage = result['metrics'].get('memory_usage_mb', 0)
                self.metrics['memory_usage'] = memory_usage
            
            logger.debug(f"Component {self.component_id} inference completed in {inference_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing input on component {self.component_id}: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

class ModelShardingManager:
    """
    Manager for cross-browser model sharding.
    
    This class coordinates sharding a model across multiple browser instances,
    leveraging browser-specific optimizations for different model components.
    """
    
    def __init__(self, model_name: str, num_shards: int = 2, shard_type: str = "layer",
                 model_type: str = "text", enable_ipfs: bool = True,
                 max_connections: int = 4, db_path: str = None):
        """
        Initialize the model sharding manager.
        
        Args:
            model_name: Name of the model to shard
            num_shards: Number of shards to create
            shard_type: Type of sharding to use ('layer', 'attention_feedforward', etc.)
            model_type: Type of model ('text', 'vision', 'audio', 'multimodal')
            enable_ipfs: Whether to enable IPFS acceleration
            max_connections: Maximum number of browser connections to use
            db_path: Path to database for result storage
        """
        self.model_name = model_name
        self.num_shards = num_shards
        self.shard_type = shard_type
        self.model_type = model_type
        self.enable_ipfs = enable_ipfs
        self.max_connections = max_connections
        self.db_path = db_path
        
        # Use environment variable for database path if not provided
        if not self.db_path:
            self.db_path = os.environ.get("BENCHMARK_DB_PATH")
        
        # Initialize resource pool integration
        self.resource_pool = None
        
        # Initialize components and execution metrics
        self.components = []
        self.initialized = False
        self.metrics = {
            'initialization_time': 0,
            'total_inference_time': 0,
            'average_inference_time': 0,
            'inference_count': 0,
            'memory_usage': 0
        }
        
        # Determine optimal browser allocation based on model type and shard type
        self.browser_allocation = self._determine_browser_allocation()
        logger.info(f"Browser allocation for {model_name}: {self.browser_allocation}")
    
    def _determine_browser_allocation(self) -> Dict[int, Dict[str, Any]]:
        """
        Determine which browsers to use for each shard based on model type.
        
        This implements a sophisticated allocation strategy that considers:
        1. Browser-specific optimizations (Firefox for audio, Edge for text, etc.)
        2. Component-specific requirements (attention vs. feedforward)
        3. Load balancing across available browsers
        
        Returns:
            Dictionary mapping shard index to browser configuration
        """
        allocation = {}
        
        # For layer-based sharding
        if self.shard_type == "layer":
            # For large language models, use browser specialization
            if self.model_type == "text" or self.model_type == "text_generation":
                for i in range(self.num_shards):
                    # Distribute layers across browsers based on layer characteristics
                    if i % 3 == 0:
                        # Every 3rd layer (including first) uses Edge+WebNN for text processing
                        allocation[i] = {"browser": "edge", "platform": "webnn", "specialization": "text"}
                    elif i % 3 == 1:
                        # Second set of layers use Chrome+WebGPU for general computation
                        allocation[i] = {"browser": "chrome", "platform": "webgpu", "specialization": "general"}
                    else:
                        # Third set of layers use Firefox+WebGPU for attention optimization
                        allocation[i] = {"browser": "firefox", "platform": "webgpu", "specialization": "attention"}
            
            # For vision models, prioritize Chrome and Firefox
            elif "vision" in self.model_type:
                for i in range(self.num_shards):
                    if i % 2 == 0:
                        # Even layers use Chrome for vision processing
                        allocation[i] = {"browser": "chrome", "platform": "webgpu", "specialization": "vision"}
                    else:
                        # Odd layers use Firefox for specialized processing
                        allocation[i] = {"browser": "firefox", "platform": "webgpu", "specialization": "vision_detail"}
            
            # For audio models, prioritize Firefox
            elif "audio" in self.model_type:
                for i in range(self.num_shards):
                    if i % 3 == 0:
                        # Every 3rd layer (including first) uses Firefox+WebGPU with compute shaders
                        allocation[i] = {"browser": "firefox", "platform": "webgpu", "specialization": "audio_compute"}
                    elif i % 3 == 1:
                        # Second set of layers use Chrome+WebGPU for general computation
                        allocation[i] = {"browser": "chrome", "platform": "webgpu", "specialization": "general"}
                    else:
                        # Third set of layers use Firefox+WebGPU again
                        allocation[i] = {"browser": "firefox", "platform": "webgpu", "specialization": "audio_compute"}
            
            # For multimodal models, use specialized allocation
            elif "multimodal" in self.model_type:
                for i in range(self.num_shards):
                    if i % 4 == 0:
                        # Text component uses Edge+WebNN
                        allocation[i] = {"browser": "edge", "platform": "webnn", "specialization": "text"}
                    elif i % 4 == 1:
                        # Vision component uses Chrome+WebGPU
                        allocation[i] = {"browser": "chrome", "platform": "webgpu", "specialization": "vision"}
                    elif i % 4 == 2:
                        # Audio component uses Firefox+WebGPU
                        allocation[i] = {"browser": "firefox", "platform": "webgpu", "specialization": "audio"}
                    else:
                        # Fusion component uses Chrome+WebGPU
                        allocation[i] = {"browser": "chrome", "platform": "webgpu", "specialization": "fusion"}
            
            # Default allocation for unknown model types
            else:
                browsers = ["chrome", "firefox", "edge"]
                for i in range(self.num_shards):
                    allocation[i] = {"browser": browsers[i % len(browsers)], "platform": "webgpu", "specialization": "general"}
        
        # For attention-feedforward sharding
        elif self.shard_type == "attention_feedforward":
            # Always use browsers with their strengths for these components
            for i in range(self.num_shards):
                if i % 2 == 0:  # Attention blocks
                    allocation[i] = {"browser": "firefox", "platform": "webgpu", "specialization": "attention", 
                                    "shard_subtype": "attention"}
                else:  # Feed-forward blocks
                    allocation[i] = {"browser": "chrome", "platform": "webgpu", "specialization": "feedforward",
                                    "shard_subtype": "feedforward"}
        
        # For model-specific components
        elif self.shard_type == "component":
            # For multimodal models with discrete components
            if "multimodal" in self.model_type:
                component_map = {
                    0: {"browser": "chrome", "platform": "webgpu", "specialization": "vision", "shard_subtype": "vision_encoder"},
                    1: {"browser": "edge", "platform": "webnn", "specialization": "text", "shard_subtype": "text_encoder"},
                    2: {"browser": "firefox", "platform": "webgpu", "specialization": "audio", "shard_subtype": "audio_encoder"},
                    3: {"browser": "chrome", "platform": "webgpu", "specialization": "fusion", "shard_subtype": "fusion_module"}
                }
                
                # Use only the number of components requested, up to maximum available
                for i in range(min(self.num_shards, len(component_map))):
                    allocation[i] = component_map[i]
            else:
                # For other models, default to layer-based allocation
                browsers = ["chrome", "firefox", "edge"]
                for i in range(self.num_shards):
                    allocation[i] = {"browser": browsers[i % len(browsers)], "platform": "webgpu", "specialization": "general"}
        
        # Default allocation for unknown shard types
        else:
            browsers = ["chrome", "firefox", "edge"]
            for i in range(self.num_shards):
                allocation[i] = {"browser": browsers[i % len(browsers)], "platform": "webgpu", "specialization": "general"}
        
        return allocation
    
    async def initialize_sharding(self):
        """Initialize the model sharding across multiple browsers."""
        if self.initialized:
            return True
        
        start_time = time.time()
        
        try:
            # Initialize resource pool integration with advanced configurations
            browser_preferences = {
                'audio': 'firefox',  # Firefox has better compute shader performance for audio
                'vision': 'chrome',  # Chrome has good WebGPU support for vision models
                'text_embedding': 'edge',  # Edge has excellent WebNN support for text embeddings
                'text': 'edge',  # Edge works well for text models
                'multimodal': 'chrome'  # Chrome is good for multimodal models
            }
            
            self.resource_pool = ResourcePoolBridgeIntegration(
                max_connections=self.max_connections,
                enable_gpu=True,
                enable_cpu=True,
                headless=True,  # Use headless mode by default
                browser_preferences=browser_preferences,
                adaptive_scaling=True,
                enable_ipfs=self.enable_ipfs,
                db_path=self.db_path
            )
            
            # Initialize resource pool
            logger.info("Initializing resource pool integration...")
            self.resource_pool.initialize()
            
            # Create components based on browser allocation
            self.components = []
            
            for shard_index, config in self.browser_allocation.items():
                # Create component ID
                component_id = f"{self.model_name}_shard{shard_index}_{config['specialization']}"
                
                # Determine shard subtype
                shard_subtype = config.get('shard_subtype', self.shard_type)
                
                # Create component
                component = ShardedModelComponent(
                    component_id=component_id,
                    model_type=self.model_type,
                    model_name=self.model_name,
                    shard_index=shard_index,
                    shard_type=shard_subtype,
                    browser=config['browser'],
                    platform=config['platform'],
                    resource_pool_integration=self.resource_pool
                )
                
                # Add to components list
                self.components.append(component)
            
            # Initialize all components concurrently
            logger.info(f"Initializing {len(self.components)} model components concurrently...")
            init_results = await asyncio.gather(*[component.initialize() for component in self.components], 
                                              return_exceptions=True)
            
            # Check initialization results
            success_count = sum(1 for r in init_results if r is True)
            logger.info(f"Initialized {success_count}/{len(self.components)} components successfully")
            
            # Update initialization status
            self.initialized = success_count == len(self.components)
            
            # Calculate total initialization time
            self.metrics['initialization_time'] = time.time() - start_time
            
            # Calculate total memory usage
            self.metrics['memory_usage'] = sum(component.metrics['memory_usage'] for component in self.components)
            
            logger.info(f"Model sharding initialized in {self.metrics['initialization_time']:.2f}s")
            logger.info(f"Total memory usage: {self.metrics['memory_usage']:.2f} MB")
            
            return self.initialized
            
        except Exception as e:
            logger.error(f"Error initializing model sharding: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _run_components_in_order(self, inputs: Dict[str, Any], shard_type: str) -> Dict:
        """
        Run components in the appropriate order based on shard type with failure detection.
        
        Args:
            inputs: Input data for all components
            shard_type: Type of sharding ('layer', 'attention_feedforward', 'component')
            
        Returns:
            Dict containing component_results and failed_components
        """
        component_results = {}
        failed_components = []
        current_inputs = inputs
        
        if shard_type == "layer":
            # For layer-based sharding, process sequentially through layers
            for component in self.components:
                try:
                    # Process through this component
                    result = await component.process(current_inputs)
                    
                    # Check for errors
                    if isinstance(result, dict) and 'error' in result:
                        logger.warning(f"Error in component {component.component_id}: {result['error']}")
                        failed_components.append(component)
                    else:
                        # Store result and update input for next component
                        component_results[component.component_id] = result
                        current_inputs = result  # Output becomes input to next layer
                except Exception as e:
                    logger.error(f"Exception in component {component.component_id}: {e}")
                    failed_components.append(component)
        
        elif shard_type == "attention_feedforward":
            # For attention-feedforward sharding, process attention first then feedforward
            attention_components = [c for c in self.components if "attention" in c.component_id]
            feedforward_components = [c for c in self.components if "feedforward" in c.component_id]
            
            # Process attention components (in parallel)
            attention_tasks = [component.process(inputs) for component in attention_components]
            attention_results = await asyncio.gather(*attention_tasks, return_exceptions=True)
            
            # Process results and track failures
            attention_output = {}
            for i, result in enumerate(attention_results):
                component = attention_components[i]
                if isinstance(result, Exception) or (isinstance(result, dict) and 'error' in result):
                    logger.warning(f"Error in attention component {component.component_id}")
                    failed_components.append(component)
                else:
                    component_results[component.component_id] = result
                    # Merge all attention outputs
                    if isinstance(result, dict):
                        attention_output.update(result)
            
            # Process feedforward components (in parallel) with attention output
            feedforward_tasks = [component.process({**inputs, **attention_output}) 
                              for component in feedforward_components]
            feedforward_results = await asyncio.gather(*feedforward_tasks, return_exceptions=True)
            
            # Process results and track failures
            for i, result in enumerate(feedforward_results):
                component = feedforward_components[i]
                if isinstance(result, Exception) or (isinstance(result, dict) and 'error' in result):
                    logger.warning(f"Error in feedforward component {component.component_id}")
                    failed_components.append(component)
                else:
                    component_results[component.component_id] = result
        
        elif shard_type == "component":
            # For component-based sharding, process components in parallel
            component_tasks = [component.process(inputs) for component in self.components]
            component_task_results = await asyncio.gather(*component_tasks, return_exceptions=True)
            
            # Process results and track failures
            for i, result in enumerate(component_task_results):
                component = self.components[i]
                if isinstance(result, Exception) or (isinstance(result, dict) and 'error' in result):
                    logger.warning(f"Error in component {component.component_id}")
                    failed_components.append(component)
                else:
                    component_results[component.component_id] = result
        
        else:
            # Default processing (in parallel)
            component_tasks = [component.process(inputs) for component in self.components]
            component_task_results = await asyncio.gather(*component_tasks, return_exceptions=True)
            
            # Process results and track failures
            for i, result in enumerate(component_task_results):
                component = self.components[i]
                if isinstance(result, Exception) or (isinstance(result, dict) and 'error' in result):
                    logger.warning(f"Error in component {component.component_id}")
                    failed_components.append(component)
                else:
                    component_results[component.component_id] = result
        
        return {
            'component_results': component_results,
            'failed_components': failed_components
        }
    
    async def _recover_failed_components(self, failed_components, inputs, successful_results, max_retries):
        """
        Attempt to recover failed components with progressive strategies.
        
        Args:
            failed_components: List of components that failed in first attempt
            inputs: Original inputs to all components
            successful_results: Results from successful components
            max_retries: Maximum number of recovery attempts
            
        Returns:
            Dict containing recovered_results, still_failed, and metrics
        """
        recovered_results = {}
        still_failed = []
        recovery_metrics = {
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'retry_succeeded': 0,
            'reroute_succeeded': 0,
            'browser_change_succeeded': 0
        }
        
        # Try to recover each failed component
        for component in failed_components:
            # Track recovery attempts
            recovery_metrics['recovery_attempts'] += 1
            recovered = False
            
            # Strategy 1: Simple retry with existing component
            for retry in range(max_retries):
                try:
                    logger.info(f"Recovery attempt {retry+1}/{max_retries} for component {component.component_id}")
                    
                    # Try to re-process with the component
                    result = await component.process(inputs)
                    
                    # Check if successful
                    if not (isinstance(result, dict) and 'error' in result):
                        logger.info(f"Successfully recovered component {component.component_id} with retry")
                        recovered_results[component.component_id] = result
                        recovered = True
                        recovery_metrics['retry_succeeded'] += 1
                        break
                except Exception as e:
                    logger.warning(f"Recovery attempt {retry+1} failed for {component.component_id}: {e}")
            
            # Strategy 2: If retry failed, try browser change
            if not recovered:
                try:
                    logger.info(f"Attempting browser change for component {component.component_id}")
                    
                    # Create browser allocation for different browser
                    alternate_browsers = {
                        'chrome': 'firefox',
                        'firefox': 'edge',
                        'edge': 'chrome'
                    }
                    new_browser = alternate_browsers[component.browser]
                    
                    # Create a new component with different browser
                    new_component = ShardedModelComponent(
                        component_id=f"{component.component_id}_recovery",
                        model_type=component.model_type,
                        model_name=component.model_name,
                        shard_index=component.shard_index,
                        shard_type=component.shard_type,
                        browser=new_browser,
                        platform=component.platform,
                        resource_pool_integration=self.resource_pool
                    )
                    
                    # Initialize new component
                    init_success = await new_component.initialize()
                    if init_success:
                        # Try to process with new component
                        result = await new_component.process(inputs)
                        
                        # Check if successful
                        if not (isinstance(result, dict) and 'error' in result):
                            logger.info(f"Successfully recovered component {component.component_id} with browser change")
                            recovered_results[component.component_id] = result
                            recovered = True
                            recovery_metrics['browser_change_succeeded'] += 1
                except Exception as e:
                    logger.warning(f"Browser change recovery failed for {component.component_id}: {e}")
            
            # If still not recovered, mark as failed
            if not recovered:
                still_failed.append(component)
        
        # Update recovery metrics
        recovery_metrics['successful_recoveries'] = len(failed_components) - len(still_failed)
        
        return {
            'recovered_results': recovered_results,
            'still_failed': still_failed,
            'metrics': recovery_metrics
        }
    
    def _merge_component_results(self, component_results, shard_type):
        """
        Merge results from all components into a single result.
        
        Args:
            component_results: Dictionary of component results
            shard_type: Type of sharding
            
        Returns:
            Merged inference result
        """
        if not component_results:
            return {'error': 'No successful component results to merge'}
        
        # Different merge strategies based on shard type
        if shard_type == "layer":
            # For layer-based sharding, use the result from the final layer
            components_by_index = sorted(
                [(k, v) for k, v in component_results.items()],
                key=lambda x: int(x[0].split("shard")[1].split("_")[0])
            )
            
            # Return result from final layer if available
            if components_by_index:
                return components_by_index[-1][1]
        
        elif shard_type == "attention_feedforward":
            # For attention-feedforward, combine attention and feedforward results
            merged = {}
            # Add results from all components (prioritizing feedforward for overlapping keys)
            for component_id, result in component_results.items():
                if isinstance(result, dict):
                    if "feedforward" in component_id:
                        # Feedforward results take priority
                        merged.update(result)
                    else:
                        # For attention results, only add keys not already present
                        for key, value in result.items():
                            if key not in merged:
                                merged[key] = value
            return merged
        
        elif shard_type == "component":
            # For component-based sharding (e.g., multimodal), merge specialized outputs
            merged = {}
            for component_id, result in component_results.items():
                if isinstance(result, dict):
                    # Use component specialization to determine output keys
                    if "vision" in component_id:
                        merged["vision_output"] = result
                    elif "text" in component_id:
                        merged["text_output"] = result
                    elif "audio" in component_id:
                        merged["audio_output"] = result
                    elif "fusion" in component_id:
                        # Fusion outputs may have special keys to preserve
                        merged["fusion_output"] = result
                        # Also include top-level outputs from fusion
                        for key, value in result.items():
                            if key not in ('vision_output', 'text_output', 'audio_output'):
                                merged[key] = value
            return merged
        
        else:
            # Default strategy: combine all results into a dictionary
            merged = {}
            for component_id, result in component_results.items():
                if isinstance(result, dict):
                    key = component_id.replace(":", "_")
                    merged[key] = result
            return merged
    
    async def run_inference_sharded(self, inputs: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
        """
        Run inference across sharded model components with fault tolerance.
        
        This method implements fault tolerance by automatically detecting
        failed components and attempting recovery or rerouting when possible.
        
        Args:
            inputs: Input data for the model
            max_retries: Maximum number of retries for failed components
            
        Returns:
            Combined inference results
        """
        if not self.initialized:
            logger.error("Model sharding not initialized")
            return {'error': 'Model sharding not initialized'}
        
        try:
            start_time = time.time()
            
            # Process inputs through pipeline of components with fault tolerance
            # This implements a robust execution model with failure handling
            
            # 1. First attempt - run components in appropriate order based on shard type
            processing_results = await self._run_components_in_order(inputs, self.shard_type)
            
            # 2. Handle any failed components
            if processing_results['failed_components']:
                logger.warning(f"Detected {len(processing_results['failed_components'])} failed components. Attempting recovery...")
                recovery_results = await self._recover_failed_components(
                    processing_results['failed_components'],
                    inputs,
                    processing_results['component_results'],
                    max_retries
                )
                
                # Update results with recovery information
                processing_results['component_results'].update(recovery_results['recovered_results'])
                processing_results['failed_components'] = recovery_results['still_failed']
                processing_results['recovery_metrics'] = recovery_results['metrics']
            
            # 3. Merge results from all successful components
            merged_result = self._merge_component_results(
                processing_results['component_results'],
                self.shard_type
            )
            
            # Track inference time
            inference_time = time.time() - start_time
            self.metrics['total_inference_time'] += inference_time
            self.metrics['inference_count'] += 1
            self.metrics['average_inference_time'] = (
                self.metrics['total_inference_time'] / self.metrics['inference_count']
                if self.metrics['inference_count'] > 0 else 0
            )
            
            # Add detailed metrics to the result
            detailed_result = {
                'result': merged_result,
                'metrics': {
                    'inference_time_ms': inference_time * 1000,
                    'component_count': len(self.components),
                    'successful_components': len(processing_results['component_results']),
                    'failed_components': len(processing_results['failed_components']),
                    'shard_type': self.shard_type,
                }
            }
            
            # Add recovery metrics if recovery was attempted
            if 'recovery_metrics' in processing_results:
                detailed_result['metrics']['recovery'] = processing_results['recovery_metrics']
            
            logger.info(f"Sharded inference completed in {inference_time:.2f}s with "
                      f"{detailed_result['metrics']['successful_components']}/{len(self.components)} "
                      f"successful components")
            
            return detailed_result
            
        except Exception as e:
            logger.error(f"Error in sharded inference: {e}")
            traceback.print_exc()
            return {'error': f"Sharded inference failed: {e}"}
            # For attention-feedforward sharding, process in parallel then combine
            elif self.shard_type == "attention_feedforward":
                # Process components in parallel
                results = await asyncio.gather(*[component.process(inputs) for component in self.components])
                
                # Check for errors
                if any('error' in r for r in results):
                    errors = [f"{self.components[i].component_id}: {r['error']}" 
                             for i, r in enumerate(results) if 'error' in r]
                    logger.error(f"Errors in components: {', '.join(errors)}")
                    return {'error': f"Components failed: {', '.join(errors)}"}
                
                # Combine results (implementation depends on model architecture)
                current_output = self._combine_attention_feedforward_results(results)
            
            # For component-based sharding (multimodal), process in parallel then combine
            elif self.shard_type == "component":
                # Process components in parallel
                results = await asyncio.gather(*[component.process(inputs) for component in self.components])
                
                # Check for errors
                if any('error' in r for r in results):
                    errors = [f"{self.components[i].component_id}: {r['error']}" 
                             for i, r in enumerate(results) if 'error' in r]
                    logger.error(f"Errors in components: {', '.join(errors)}")
                    return {'error': f"Components failed: {', '.join(errors)}"}
                
                # Combine results from different model components
                current_output = self._combine_component_results(results)
            
            # Calculate total inference time
            inference_time = time.time() - start_time
            
            # Update metrics
            self.metrics['total_inference_time'] += inference_time
            self.metrics['inference_count'] += 1
            self.metrics['average_inference_time'] = (
                self.metrics['total_inference_time'] / self.metrics['inference_count']
            )
            
            # Add metrics to result
            result = {
                'output': current_output,
                'metrics': {
                    'inference_time': inference_time,
                    'sharded_execution': True,
                    'num_shards': self.num_shards,
                    'shard_type': self.shard_type,
                    'average_inference_time': self.metrics['average_inference_time'],
                    'memory_usage': self.metrics['memory_usage']
                }
            }
            
            logger.info(f"Sharded inference completed in {inference_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in sharded inference: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _combine_attention_feedforward_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from attention and feedforward components.
        
        This is a placeholder for the actual implementation, which would depend
        on the specific model architecture.
        
        Args:
            results: List of results from attention and feedforward components
            
        Returns:
            Combined result
        """
        # This is a simplified combination - actual implementation would be model-specific
        combined_result = {}
        
        # Combine outputs from different components
        for i, result in enumerate(results):
            if isinstance(result, dict) and 'output' in result:
                # This is where the component-specific combination logic would go
                # For now, we just add keys from each component
                component_type = self.components[i].shard_subtype
                combined_result[f"{component_type}_output"] = result['output']
        
        # For demonstration, add combined metrics
        combined_result['combined_metrics'] = {
            'component_count': len(results),
            'components': [c.component_id for c in self.components]
        }
        
        return combined_result
    
    def _combine_component_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from different model components (e.g., vision, text, audio).
        
        This is a placeholder for the actual implementation, which would depend
        on the specific model architecture.
        
        Args:
            results: List of results from different model components
            
        Returns:
            Combined result for multimodal model
        """
        # This is a simplified implementation - actual implementation would be model-specific
        combined_result = {}
        
        # Extract outputs from different components
        component_outputs = {}
        for i, result in enumerate(results):
            if isinstance(result, dict) and 'output' in result:
                component_type = self.components[i].shard_subtype
                component_outputs[component_type] = result['output']
        
        # For multimodal models, combine vision, text, and audio outputs
        if 'vision_encoder' in component_outputs and 'text_encoder' in component_outputs:
            # This is where model-specific fusion would happen
            combined_result['multimodal_embedding'] = {
                'vision_features': component_outputs.get('vision_encoder'),
                'text_features': component_outputs.get('text_encoder'),
                'audio_features': component_outputs.get('audio_encoder'),
                'is_multimodal': True
            }
            
            # If there's a fusion module, use its output as the final result
            if 'fusion_module' in component_outputs:
                combined_result['fused_output'] = component_outputs['fusion_module']
            
        # Simplified combination for other types
        else:
            combined_result['combined_output'] = component_outputs
        
        return combined_result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics about the sharded model execution."""
        if not self.initialized:
            return {'error': 'Model sharding not initialized'}
        
        # Collect metrics from all components
        component_metrics = {}
        for component in self.components:
            component_metrics[component.component_id] = component.metrics
        
        # Build comprehensive metrics report
        metrics_report = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'num_shards': self.num_shards,
            'shard_type': self.shard_type,
            'initialization_time': self.metrics['initialization_time'],
            'average_inference_time': self.metrics['average_inference_time'],
            'inference_count': self.metrics['inference_count'],
            'memory_usage': self.metrics['memory_usage'],
            'browser_allocation': self.browser_allocation,
            'component_metrics': component_metrics
        }
        
        return metrics_report
    
    async def close(self):
        """Close all resources used by the model sharding manager."""
        if self.resource_pool:
            self.resource_pool.close()
            logger.info("Model sharding manager closed")
        self.initialized = False
        self.components = []

# Example usage
async def test_model_sharding(model_name, num_shards=3, shard_type="layer", model_type="text"):
    """Test model sharding with a sample model."""
    # Create model sharding manager
    manager = ModelShardingManager(
        model_name=model_name,
        num_shards=num_shards,
        shard_type=shard_type,
        model_type=model_type,
        enable_ipfs=True
    )
    
    try:
        # Initialize sharding
        logger.info(f"Initializing sharding for {model_name} with {num_shards} shards")
        initialized = await manager.initialize_sharding()
        
        if not initialized:
            logger.error("Failed to initialize model sharding")
            return
        
        # Create sample input
        sample_input = {}
        if model_type == "text" or model_type == "text_embedding":
            sample_input = {
                'input_ids': [101, 2023, 2003, 1037, 3231, 102],
                'attention_mask': [1, 1, 1, 1, 1, 1]
            }
        elif model_type == "vision":
            sample_input = {'pixel_values': [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(1)]}
        elif model_type == "audio":
            sample_input = {'input_features': [[[0.1 for _ in range(80)] for _ in range(3000)]]}
        elif model_type == "multimodal":
            sample_input = {
                'input_ids': [101, 2023, 2003, 1037, 3231, 102],
                'attention_mask': [1, 1, 1, 1, 1, 1],
                'pixel_values': [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(1)]
            }
        
        # Run inference
        logger.info(f"Running sharded inference for {model_name}")
        result = await manager.run_inference_sharded(sample_input)
        
        # Print result summary
        if 'error' in result:
            logger.error(f"Inference error: {result['error']}")
        else:
            logger.info(f"Inference successful")
            if 'metrics' in result:
                logger.info(f"Inference time: {result['metrics']['inference_time']:.2f}s")
                logger.info(f"Memory usage: {result['metrics']['memory_usage']:.2f} MB")
        
        # Get detailed metrics
        metrics = manager.get_metrics()
        logger.info(f"Detailed metrics: {json.dumps(metrics, indent=2)}")
        
    finally:
        # Close manager
        await manager.close()
        logger.info("Test completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test cross-browser model sharding")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name")
    parser.add_argument("--shards", type=int, default=3, help="Number of shards")
    parser.add_argument("--type", type=str, default="layer", choices=["layer", "attention_feedforward", "component"],
                      help="Sharding type")
    parser.add_argument("--model-type", type=str, default="text", 
                      choices=["text", "vision", "audio", "multimodal", "text_embedding"], 
                      help="Model type")
    
    args = parser.parse_args()
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_model_sharding(args.model, args.shards, args.type, args.model_type))