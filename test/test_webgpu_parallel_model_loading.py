#!/usr/bin/env python3
"""
Test script for evaluating WebGPU parallel model loading optimizations.

This script specifically tests the parallel model loading implementation for multimodal models,
which improves initialization time and memory efficiency for models with multiple components.

Usage:
    python test_webgpu_parallel_model_loading.py --model-type multimodal
    python test_webgpu_parallel_model_loading.py --model-type vision-language
    python test_webgpu_parallel_model_loading.py --model-name "openai/clip-vit-base-patch32"
    python test_webgpu_parallel_model_loading.py --test-all --benchmark
"""

import os
import sys
import json
import time
import random
import argparse
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("parallel_model_loading_test")

# Constants
TEST_MODELS = {
    "multimodal": "openai/clip-vit-base-patch32",
    "vision-language": "llava-hf/llava-1.5-7b-hf",
    "multi-task": "facebook/bart-large-mnli",
    "multi-encoder": "microsoft/resnet-50"
}

COMPONENT_CONFIGURATIONS = {
    "openai/clip-vit-base-patch32": ["vision_encoder", "text_encoder"],
    "llava-hf/llava-1.5-7b-hf": ["vision_encoder", "text_encoder", "fusion_model", "language_model"],
    "facebook/bart-large-mnli": ["encoder", "decoder", "classification_head"],
    "microsoft/resnet-50": ["backbone", "classification_head"],
    "default": ["primary_model", "secondary_model"]
}

def setup_environment(parallel_loading=True):
    """
    Set up the environment variables for WebGPU testing with parallel model loading.
    
    Args:
        parallel_loading: Whether to enable parallel model loading
        
    Returns:
        True if successful, False otherwise
    """
    # Set WebGPU environment variables
    os.environ["WEBGPU_ENABLED"] = "1"
    os.environ["WEBGPU_SIMULATION"] = "1" 
    os.environ["WEBGPU_AVAILABLE"] = "1"
    
    # Enable parallel loading if requested
    if parallel_loading:
        os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1"
        logger.info("WebGPU parallel model loading enabled")
    else:
        if "WEB_PARALLEL_LOADING_ENABLED" in os.environ:
            del os.environ["WEB_PARALLEL_LOADING_ENABLED"]
        logger.info("WebGPU parallel model loading disabled")
    
    # Enable shader precompilation by default for all tests
    # This isn't the focus of our test but improves overall performance
    os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
    
    return True

def setup_web_platform_handler():
    """
    Set up and import the fixed web platform handler.
    
    Returns:
        The imported module or None if failed
    """
    try:
        # Try to import fixed_web_platform from the current directory
        sys.path.append('.')
        from fixed_web_platform.web_platform_handler import (
            process_for_web, init_webgpu, create_mock_processors
        )
        logger.info("Successfully imported web platform handler from fixed_web_platform")
        return {
            "process_for_web": process_for_web,
            "init_webgpu": init_webgpu,
            "create_mock_processors": create_mock_processors
        }
    except ImportError:
        # Try to import from the test directory
        try:
            sys.path.append('test')
            from fixed_web_platform.web_platform_handler import (
                process_for_web, init_webgpu, create_mock_processors
            )
            logger.info("Successfully imported web platform handler from test/fixed_web_platform")
            return {
                "process_for_web": process_for_web,
                "init_webgpu": init_webgpu,
                "create_mock_processors": create_mock_processors
            }
        except ImportError:
            logger.error("Failed to import web platform handler from fixed_web_platform")
            return None

def enhance_parallel_loading_tracker():
    """
    Update the ParallelLoadingTracker for enhanced performance monitoring.
    
    This function will modify the web_platform_handler.py file to enhance
    the ParallelLoadingTracker class with more realistic parallel loading simulation.
    """
    # Path to the handler file
    handler_path = "fixed_web_platform/web_platform_handler.py"
    
    # Check if file exists
    if not os.path.exists(handler_path):
        handler_path = "test/fixed_web_platform/web_platform_handler.py"
        if not os.path.exists(handler_path):
            logger.error(f"Cannot find web_platform_handler.py")
            return False
    
    # Create a backup
    backup_path = f"{handler_path}.parallel.bak"
    with open(handler_path, 'r') as src:
        with open(backup_path, 'w') as dst:
            dst.write(src.read())
    
    logger.info(f"Created backup at {backup_path}")
    
    # Find the ParallelLoadingTracker class and enhance it
    with open(handler_path, 'r') as f:
        content = f.read()
    
    # Replace the basic ParallelLoadingTracker with enhanced version
    basic_tracker = 'class ParallelLoadingTracker:\n'
    basic_tracker += '                def __init__(self, model_name):\n'
    basic_tracker += '                    self.model_name = model_name\n'
    basic_tracker += '                    self.parallel_load_time = None\n'
    basic_tracker += '                    \n'
    basic_tracker += '                def test_parallel_load(self, platform="webgpu"):\n'
    basic_tracker += '                    import time\n'
    basic_tracker += '                    # Simulate parallel loading\n'
    basic_tracker += '                    start_time = time.time()\n'
    basic_tracker += '                    # Simulate different loading times\n'
    basic_tracker += '                    time.sleep(0.1)  # 100ms loading time simulation\n'
    basic_tracker += '                    self.parallel_load_time = (time.time() - start_time) * 1000  # ms\n'
    basic_tracker += '                    return self.parallel_load_time'
    
    enhanced_tracker = 'class ParallelLoadingTracker:\n'
    enhanced_tracker += '                def __init__(self, model_name):\n'
    enhanced_tracker += '                    self.model_name = model_name\n'
    enhanced_tracker += '                    self.parallel_load_time = None\n'
    enhanced_tracker += '                    self.sequential_load_time = None\n'
    enhanced_tracker += '                    self.components = []\n'
    enhanced_tracker += '                    self.parallel_loading_enabled = "WEB_PARALLEL_LOADING_ENABLED" in os.environ\n'
    enhanced_tracker += '                    self.model_components = {}\n'
    enhanced_tracker += '                    self.load_stats = {\n'
    enhanced_tracker += '                        "total_loading_time_ms": 0,\n'
    enhanced_tracker += '                        "parallel_loading_time_ms": 0,\n'
    enhanced_tracker += '                        "sequential_loading_time_ms": 0,\n'
    enhanced_tracker += '                        "components_loaded": 0,\n'
    enhanced_tracker += '                        "memory_peak_mb": 0,\n'
    enhanced_tracker += '                        "loading_speedup": 0,\n'
    enhanced_tracker += '                        "component_sizes_mb": {}\n'
    enhanced_tracker += '                    }\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Get model components based on model name\n'
    enhanced_tracker += '                    model_type = getattr(self, "mode", "unknown")\n'
    enhanced_tracker += '                    self.model_name = model_name\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Determine components based on model name\n'
    enhanced_tracker += '                    if self.model_name in COMPONENT_CONFIGURATIONS:\n'
    enhanced_tracker += '                        self.components = COMPONENT_CONFIGURATIONS[self.model_name]\n'
    enhanced_tracker += '                    elif model_type == "multimodal":\n'
    enhanced_tracker += '                        self.components = ["vision_encoder", "text_encoder"]\n'
    enhanced_tracker += '                    elif model_type == "vision-language":\n'
    enhanced_tracker += '                        self.components = ["vision_encoder", "text_encoder", "fusion_model", "language_model"]\n'
    enhanced_tracker += '                    elif model_type == "multi-task":\n'
    enhanced_tracker += '                        self.components = ["encoder", "decoder", "classification_head"]\n'
    enhanced_tracker += '                    else:\n'
    enhanced_tracker += '                        self.components = ["primary_model", "secondary_model"]\n'
    enhanced_tracker += '                        \n'
    enhanced_tracker += '                    self.load_stats["components_loaded"] = len(self.components)\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Generate random component sizes (MB) - larger for language models\n'
    enhanced_tracker += '                    import random\n'
    enhanced_tracker += '                    for component in self.components:\n'
    enhanced_tracker += '                        if "language" in component or "llm" in component:\n'
    enhanced_tracker += '                            # Language models are usually larger\n'
    enhanced_tracker += '                            size_mb = random.uniform(200, 800)\n'
    enhanced_tracker += '                        elif "vision" in component or "image" in component:\n'
    enhanced_tracker += '                            # Vision models are medium-sized\n'
    enhanced_tracker += '                            size_mb = random.uniform(80, 300)\n'
    enhanced_tracker += '                        elif "text" in component or "encoder" in component:\n'
    enhanced_tracker += '                            # Text encoders are smaller\n'
    enhanced_tracker += '                            size_mb = random.uniform(40, 150)\n' 
    enhanced_tracker += '                        else:\n'
    enhanced_tracker += '                            # Other components\n'
    enhanced_tracker += '                            size_mb = random.uniform(30, 100)\n'
    enhanced_tracker += '                            \n'
    enhanced_tracker += '                        self.load_stats["component_sizes_mb"][component] = size_mb\n'
    enhanced_tracker += '                        \n'
    enhanced_tracker += '                    # Calculate total memory peak (sum of all components)\n'
    enhanced_tracker += '                    self.load_stats["memory_peak_mb"] = sum(self.load_stats["component_sizes_mb"].values())\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # If parallel loading is enabled, initialize components in parallel\n'
    enhanced_tracker += '                    if self.parallel_loading_enabled:\n'
    enhanced_tracker += '                        self.simulate_parallel_loading()\n'
    enhanced_tracker += '                    else:\n'
    enhanced_tracker += '                        self.simulate_sequential_loading()\n'
    enhanced_tracker += '                \n'
    enhanced_tracker += '                def simulate_parallel_loading(self):\n'
    enhanced_tracker += '                    """Simulate loading model components in parallel"""\n'
    enhanced_tracker += '                    import time\n'
    enhanced_tracker += '                    import random\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    logger.info(f"Simulating parallel loading for {len(self.components)} components")\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Start timing\n'
    enhanced_tracker += '                    start_time = time.time()\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # In parallel loading, we load all components concurrently\n'
    enhanced_tracker += '                    # The total time is determined by the slowest component\n'
    enhanced_tracker += '                    # We add a small coordination overhead\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Calculate load times for each component\n'
    enhanced_tracker += '                    component_load_times = {}\n'
    enhanced_tracker += '                    for component in self.components:\n'
    enhanced_tracker += '                        # Loading time is roughly proportional to component size\n'
    enhanced_tracker += '                        # We use the component sizes already calculated plus some randomness\n'
    enhanced_tracker += '                        size_mb = self.load_stats["component_sizes_mb"][component]\n'
    enhanced_tracker += '                        # Assume 20MB/sec loading rate with some variance\n'
    enhanced_tracker += '                        load_time_ms = (size_mb / 20.0) * 1000 * random.uniform(0.9, 1.1)\n'
    enhanced_tracker += '                        component_load_times[component] = load_time_ms\n'
    enhanced_tracker += '                        \n'
    enhanced_tracker += '                    # In parallel, the total time is the maximum component time plus overhead\n'
    enhanced_tracker += '                    coordination_overhead_ms = 10 * len(self.components)  # 10ms per component coordination overhead\n'
    enhanced_tracker += '                    max_component_time = max(component_load_times.values())\n'
    enhanced_tracker += '                    parallel_time = max_component_time + coordination_overhead_ms\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Simulate the loading time\n'
    enhanced_tracker += '                    time.sleep(parallel_time / 1000)\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Store loading time\n'
    enhanced_tracker += '                    self.parallel_load_time = (time.time() - start_time) * 1000  # ms\n'
    enhanced_tracker += '                    self.load_stats["parallel_loading_time_ms"] = self.parallel_load_time\n'
    enhanced_tracker += '                    self.load_stats["total_loading_time_ms"] = self.parallel_load_time\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Simulate sequential loading for comparison but don\'t actually wait\n'
    enhanced_tracker += '                    self.simulate_sequential_loading(simulate_wait=False)\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Calculate speedup\n'
    enhanced_tracker += '                    if self.sequential_load_time > 0:\n'
    enhanced_tracker += '                        self.load_stats["loading_speedup"] = self.sequential_load_time / self.parallel_load_time\n'
    enhanced_tracker += '                        \n'
    enhanced_tracker += '                    logger.info(f"Parallel loading completed in {self.parallel_load_time:.2f}ms " +\n'
    enhanced_tracker += '                              f"(vs {self.sequential_load_time:.2f}ms sequential, " +\n'
    enhanced_tracker += '                              f"{self.load_stats[\'loading_speedup\']:.2f}x speedup)")\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    return self.parallel_load_time\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                def simulate_sequential_loading(self, simulate_wait=True):\n'
    enhanced_tracker += '                    """Simulate loading model components sequentially"""\n'
    enhanced_tracker += '                    import time\n'
    enhanced_tracker += '                    import random\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    logger.info(f"Simulating sequential loading for {len(self.components)} components")\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Start timing if we\'re actually waiting\n'
    enhanced_tracker += '                    start_time = time.time() if simulate_wait else None\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # In sequential loading, we load one component at a time\n'
    enhanced_tracker += '                    total_time_ms = 0\n'
    enhanced_tracker += '                    for component in self.components:\n'
    enhanced_tracker += '                        # Loading time calculation is the same as parallel\n'
    enhanced_tracker += '                        size_mb = self.load_stats["component_sizes_mb"][component]\n'
    enhanced_tracker += '                        load_time_ms = (size_mb / 20.0) * 1000 * random.uniform(0.9, 1.1)\n'
    enhanced_tracker += '                        total_time_ms += load_time_ms\n'
    enhanced_tracker += '                        \n'
    enhanced_tracker += '                        # Simulate the wait if requested\n'
    enhanced_tracker += '                        if simulate_wait:\n'
    enhanced_tracker += '                            time.sleep(load_time_ms / 1000)\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # Sequential has less coordination overhead but initializes each component separately\n'
    enhanced_tracker += '                    initialization_overhead_ms = 5 * len(self.components)\n'
    enhanced_tracker += '                    total_time_ms += initialization_overhead_ms\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    # If we\'re simulating the wait, calculate actual time\n'
    enhanced_tracker += '                    if simulate_wait:\n'
    enhanced_tracker += '                        self.sequential_load_time = (time.time() - start_time) * 1000  # ms\n'
    enhanced_tracker += '                        self.load_stats["sequential_loading_time_ms"] = self.sequential_load_time\n'
    enhanced_tracker += '                        self.load_stats["total_loading_time_ms"] = self.sequential_load_time\n'
    enhanced_tracker += '                    else:\n'
    enhanced_tracker += '                        # Otherwise just store the calculated time\n'
    enhanced_tracker += '                        self.sequential_load_time = total_time_ms\n'
    enhanced_tracker += '                        self.load_stats["sequential_loading_time_ms"] = total_time_ms\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    if simulate_wait:\n'
    enhanced_tracker += '                        logger.info(f"Sequential loading completed in {self.sequential_load_time:.2f}ms")\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                    return self.sequential_load_time\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                def get_components(self):\n'
    enhanced_tracker += '                    """Return model components"""\n'
    enhanced_tracker += '                    return self.components\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                def get_loading_stats(self):\n'
    enhanced_tracker += '                    """Return loading statistics"""\n'
    enhanced_tracker += '                    return self.load_stats\n'
    enhanced_tracker += '                    \n'
    enhanced_tracker += '                def test_parallel_load(self, platform="webgpu"):\n'
    enhanced_tracker += '                    """Test parallel loading performance - kept for compatibility"""\n'
    enhanced_tracker += '                    # This method maintained for backward compatibility\n'
    enhanced_tracker += '                    if self.parallel_loading_enabled:\n'
    enhanced_tracker += '                        return self.parallel_load_time or self.simulate_parallel_loading()\n'
    enhanced_tracker += '                    else:\n'
    enhanced_tracker += '                        return self.sequential_load_time or self.simulate_sequential_loading()'
    
    # Add COMPONENT_CONFIGURATIONS to the file
    component_configs = '# Model component configurations\n'
    component_configs += 'COMPONENT_CONFIGURATIONS = {\n'
    component_configs += '    "openai/clip-vit-base-patch32": ["vision_encoder", "text_encoder"],\n'
    component_configs += '    "llava-hf/llava-1.5-7b-hf": ["vision_encoder", "text_encoder", "fusion_model", "language_model"],\n'
    component_configs += '    "facebook/bart-large-mnli": ["encoder", "decoder", "classification_head"],\n'
    component_configs += '    "microsoft/resnet-50": ["backbone", "classification_head"],\n'
    component_configs += '    "default": ["primary_model", "secondary_model"]\n'
    component_configs += '}\n'
    
    # Replace the implementation
    if basic_tracker in content:
        logger.info("Found ParallelLoadingTracker class, enhancing it")
        # Add COMPONENT_CONFIGURATIONS after imports
        import_section_end = content.find("# Initialize logging")
        
        if import_section_end > 0:
            logger.info("Adding component configurations")
            content = content[:import_section_end] + component_configs + content[import_section_end:]
            
            # Now replace the ParallelLoadingTracker class
            new_content = content.replace(basic_tracker, enhanced_tracker)
            
            # Write the updated content
            with open(handler_path, 'w') as f:
                f.write(new_content)
            
            logger.info("Successfully enhanced ParallelLoadingTracker")
            return True
        else:
            logger.error("Could not find appropriate location to add component configurations")
            return False
    else:
        logger.error("Could not find ParallelLoadingTracker class to enhance")
        return False

def test_webgpu_model(model_type=None, model_name=None, parallel_loading=True, iterations=5):
    """
    Test a model with WebGPU using parallel model loading.
    
    Args:
        model_type: Type of model to test ("multimodal", "vision-language", etc.)
        model_name: Specific model name to test
        parallel_loading: Whether to use parallel model loading
        iterations: Number of inference iterations
        
    Returns:
        Dictionary with test results
    """
    # Import web platform handler
    handlers = setup_web_platform_handler()
    if not handlers:
        return {
            "success": False,
            "error": "Failed to import web platform handler"
        }
    
    process_for_web = handlers["process_for_web"]
    init_webgpu = handlers["init_webgpu"]
    create_mock_processors = handlers["create_mock_processors"]
    
    # Set up environment
    setup_environment(parallel_loading=parallel_loading)
    
    # Select model based on type or direct name
    if model_name:
        selected_model_name = model_name
        # Try to infer model type if not provided
        if not model_type:
            # Default to multimodal if can't determine
            model_type = "multimodal" 
    elif model_type in TEST_MODELS:
        selected_model_name = TEST_MODELS[model_type]
    else:
        return {
            "success": False,
            "error": f"Unknown model type: {model_type} and no model name provided"
        }
    
    # Create test class
    class TestModel:
        def __init__(self):
            self.model_name = selected_model_name
            self.mode = model_type
            self.device = "webgpu"
            self.processors = create_mock_processors()
    
    # Initialize test model
    test_model = TestModel()
    
    # Track initial load time
    start_time = time.time()
    
    # Initialize WebGPU implementation
    processor_key = "multimodal_processor" if model_type == "multimodal" or model_type == "vision-language" else None
    processor_key = "image_processor" if not processor_key and model_type == "vision" else processor_key
    
    result = init_webgpu(
        test_model,
        model_name=test_model.model_name,
        model_type=test_model.mode,
        device=test_model.device,
        web_api_mode="simulation",
        create_mock_processor=test_model.processors[processor_key]() if processor_key else None,
        parallel_loading=parallel_loading
    )
    
    # Calculate initialization time
    init_time = (time.time() - start_time) * 1000  # ms
    
    if not result or not isinstance(result, dict):
        return {
            "success": False,
            "error": f"Failed to initialize WebGPU for {model_type}"
        }
    
    # Extract endpoint and check if it's valid
    endpoint = result.get("endpoint")
    if not endpoint:
        return {
            "success": False,
            "error": f"No endpoint returned for {model_type}"
        }
    
    # Create appropriate test input based on model type
    if model_type == "multimodal" or model_type == "vision-language":
        test_input = {"image_url": "test.jpg", "text": "What is in this image?"}
    elif model_type == "vision":
        test_input = "test.jpg"
    elif model_type == "text":
        test_input = "This is a test input for text models"
    else:
        test_input = {"input": "Generic test input"}
    
    # Process input for WebGPU
    processed_input = process_for_web(test_model.mode, test_input, False)
    
    # Run initial inference to warm up and track time
    try:
        warm_up_start = time.time()
        warm_up_result = endpoint(processed_input)
        first_inference_time = (time.time() - warm_up_start) * 1000  # ms
    except Exception as e:
        return {
            "success": False,
            "error": f"Error during warm-up: {str(e)}"
        }
    
    # Get implementation details and loading stats
    implementation_type = warm_up_result.get("implementation_type", "UNKNOWN")
    performance_metrics = warm_up_result.get("performance_metrics", {})
    
    # Extract loading times if available
    parallel_load_time = performance_metrics.get("parallel_load_time_ms", 0)
    
    # Run benchmark iterations
    inference_times = []
    
    for i in range(iterations):
        start_time = time.time()
        inference_result = endpoint(processed_input)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to ms
        inference_times.append(elapsed_time)
    
    # Calculate performance metrics
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    min_inference_time = min(inference_times) if inference_times else 0
    max_inference_time = max(inference_times) if inference_times else 0
    std_dev = (
        (sum((t - avg_inference_time) ** 2 for t in inference_times) / len(inference_times)) ** 0.5 
        if len(inference_times) > 1 else 0
    )
    
    # Create result
    return {
        "success": True,
        "model_type": model_type,
        "model_name": selected_model_name,
        "implementation_type": implementation_type,
        "parallel_loading_enabled": parallel_loading,
        "initialization_time_ms": init_time,
        "first_inference_time_ms": first_inference_time,
        "parallel_load_time_ms": parallel_load_time,
        "performance": {
            "iterations": iterations,
            "avg_inference_time_ms": avg_inference_time,
            "min_inference_time_ms": min_inference_time,
            "max_inference_time_ms": max_inference_time,
            "std_dev_ms": std_dev
        },
        "performance_metrics": performance_metrics
    }

def compare_parallel_loading_options(model_type=None, model_name=None, iterations=5):
    """
    Compare model performance with and without parallel loading.
    
    Args:
        model_type: Type of model to test
        model_name: Specific model name to test
        iterations: Number of inference iterations per configuration
        
    Returns:
        Dictionary with comparison results
    """
    # Run tests with parallel loading
    with_parallel = test_webgpu_model(
        model_type=model_type,
        model_name=model_name,
        parallel_loading=True,
        iterations=iterations
    )
    
    # Run tests without parallel loading
    without_parallel = test_webgpu_model(
        model_type=model_type,
        model_name=model_name,
        parallel_loading=False,
        iterations=iterations
    )
    
    # Calculate improvements
    init_improvement = 0
    first_inference_improvement = 0
    load_time_improvement = 0
    
    if (with_parallel.get("success", False) and 
        without_parallel.get("success", False)):
        
        # Calculate initialization time improvement
        with_init = with_parallel.get("initialization_time_ms", 0)
        without_init = without_parallel.get("initialization_time_ms", 0)
        
        if without_init > 0:
            init_improvement = (without_init - with_init) / without_init * 100
        
        # Calculate first inference time improvement
        with_first = with_parallel.get("first_inference_time_ms", 0)
        without_first = without_parallel.get("first_inference_time_ms", 0)
        
        if without_first > 0:
            first_inference_improvement = (without_first - with_first) / without_first * 100
        
        # Calculate model loading time improvement (from metrics)
        with_metrics = with_parallel.get("performance_metrics", {})
        without_metrics = without_parallel.get("performance_metrics", {})
        
        with_load = with_metrics.get("parallel_loading_time_ms", 0)
        if not with_load:
            with_load = with_metrics.get("loading_stats", {}).get("total_loading_time_ms", 0)
            
        without_load = without_metrics.get("sequential_loading_time_ms", 0)
        if not without_load:
            without_load = without_metrics.get("loading_stats", {}).get("total_loading_time_ms", 0)
        
        if without_load > 0:
            load_time_improvement = (without_load - with_load) / without_load * 100
    
    # Calculate model name
    model_name = with_parallel.get("model_name") if with_parallel.get("success") else model_name
    if not model_name and model_type:
        model_name = TEST_MODELS.get(model_type, "unknown_model")
    
    return {
        "model_type": model_type,
        "model_name": model_name,
        "with_parallel": with_parallel,
        "without_parallel": without_parallel,
        "improvements": {
            "initialization_time_percent": init_improvement,
            "first_inference_percent": first_inference_improvement,
            "load_time_percent": load_time_improvement
        }
    }

def run_all_model_comparisons(iterations=5, output_json=None, create_chart=False):
    """
    Run comparisons for all test model types.
    
    Args:
        iterations: Number of inference iterations per configuration
        output_json: Path to save JSON results
        create_chart: Whether to create a performance comparison chart
        
    Returns:
        Dictionary with all comparison results
    """
    results = {}
    model_types = list(TEST_MODELS.keys())
    
    for model_type in model_types:
        logger.info(f"Testing {model_type} with and without parallel loading...")
        comparison = compare_parallel_loading_options(model_type, iterations=iterations)
        results[model_type] = comparison
        
        # Print summary
        improvements = comparison.get("improvements", {})
        init_improvement = improvements.get("initialization_time_percent", 0)
        load_improvement = improvements.get("load_time_percent", 0)
        
        logger.info(f"  • {model_type}: {init_improvement:.2f}% faster initialization, {load_improvement:.2f}% faster model loading")
    
    # Save results to JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_json}")
    
    # Create chart if requested
    if create_chart:
        create_performance_chart(results, f"webgpu_parallel_loading_comparison_{int(time.time())}.png")
    
    return results

def create_performance_chart(results, output_file):
    """
    Create a performance comparison chart.
    
    Args:
        results: Dictionary with comparison results
        output_file: Path to save the chart
    """
    try:
        model_types = list(results.keys())
        with_parallel_init = []
        without_parallel_init = []
        with_parallel_load = []
        without_parallel_load = []
        init_improvements = []
        load_improvements = []
        
        for model_type in model_types:
            comparison = results[model_type]
            
            # Get initialization times
            with_init = comparison.get("with_parallel", {}).get("initialization_time_ms", 0)
            without_init = comparison.get("without_parallel", {}).get("initialization_time_ms", 0)
            
            # Get loading time metrics
            with_metrics = comparison.get("with_parallel", {}).get("performance_metrics", {})
            without_metrics = comparison.get("without_parallel", {}).get("performance_metrics", {})
            
            with_load = with_metrics.get("parallel_loading_time_ms", 0)
            if not with_load:
                with_load = with_metrics.get("loading_stats", {}).get("total_loading_time_ms", 0)
                
            without_load = without_metrics.get("sequential_loading_time_ms", 0)
            if not without_load:
                without_load = without_metrics.get("loading_stats", {}).get("total_loading_time_ms", 0)
            
            # Get improvement percentages
            improvements = comparison.get("improvements", {})
            init_improvement = improvements.get("initialization_time_percent", 0)
            load_improvement = improvements.get("load_time_percent", 0)
            
            # Add to lists for plotting
            with_parallel_init.append(with_init)
            without_parallel_init.append(without_init)
            with_parallel_load.append(with_load)
            without_parallel_load.append(without_load)
            init_improvements.append(init_improvement)
            load_improvements.append(load_improvement)
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
        
        # Bar chart for initialization times
        x = range(len(model_types))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], without_parallel_init, width, label='Without Parallel Loading')
        ax1.bar([i + width/2 for i in x], with_parallel_init, width, label='With Parallel Loading')
        
        ax1.set_xlabel('Model Types')
        ax1.set_ylabel('Initialization Time (ms)')
        ax1.set_title('WebGPU Initialization Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_types)
        ax1.legend()
        
        # Add initialization time values on bars
        for i, v in enumerate(without_parallel_init):
            ax1.text(i - width/2, v + 5, f"{v:.1f}", ha='center')
        
        for i, v in enumerate(with_parallel_init):
            ax1.text(i + width/2, v + 5, f"{v:.1f}", ha='center')
        
        # Bar chart for model loading times
        ax2.bar([i - width/2 for i in x], without_parallel_load, width, label='Without Parallel Loading')
        ax2.bar([i + width/2 for i in x], with_parallel_load, width, label='With Parallel Loading')
        
        ax2.set_xlabel('Model Types')
        ax2.set_ylabel('Model Loading Time (ms)')
        ax2.set_title('WebGPU Model Loading Time Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_types)
        ax2.legend()
        
        # Add model loading time values on bars
        for i, v in enumerate(without_parallel_load):
            ax2.text(i - width/2, v + 5, f"{v:.1f}", ha='center')
        
        for i, v in enumerate(with_parallel_load):
            ax2.text(i + width/2, v + 5, f"{v:.1f}", ha='center')
        
        # Bar chart for improvement percentages
        ax3.bar([i - width/2 for i in x], init_improvements, width, label='Initialization Improvement')
        ax3.bar([i + width/2 for i in x], load_improvements, width, label='Loading Time Improvement')
        
        ax3.set_xlabel('Model Types')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Performance Improvement with Parallel Model Loading')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_types)
        ax3.legend()
        
        # Add improvement percentages on bars
        for i, v in enumerate(init_improvements):
            ax3.text(i - width/2, v + 1, f"{v:.1f}%", ha='center')
        
        for i, v in enumerate(load_improvements):
            ax3.text(i + width/2, v + 1, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Performance chart saved to {output_file}")
    except Exception as e:
        logger.error(f"Error creating performance chart: {e}")

def main():
    """Parse arguments and run the tests."""
    parser = argparse.ArgumentParser(
        description="Test WebGPU parallel model loading optimizations"
    )
    
    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument("--model-type", choices=list(TEST_MODELS.keys()), default="multimodal",
                           help="Model type to test")
    model_group.add_argument("--model-name", type=str,
                           help="Specific model name to test")
    model_group.add_argument("--test-all", action="store_true",
                           help="Test all available model types")
    
    # Test options
    test_group = parser.add_argument_group("Test Options")
    test_group.add_argument("--iterations", type=int, default=5,
                          help="Number of inference iterations for each test")
    test_group.add_argument("--benchmark", action="store_true",
                          help="Run in benchmark mode with 10 iterations")
    test_group.add_argument("--with-parallel-only", action="store_true",
                          help="Only test with parallel loading enabled")
    test_group.add_argument("--without-parallel-only", action="store_true",
                          help="Only test without parallel loading")
    
    # Setup options
    setup_group = parser.add_argument_group("Setup Options")
    setup_group.add_argument("--update-handler", action="store_true",
                           help="Update the WebGPU handler with enhanced parallel loading")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-json", type=str,
                            help="Save results to JSON file")
    output_group.add_argument("--create-chart", action="store_true",
                            help="Create performance comparison chart")
    output_group.add_argument("--verbose", action="store_true",
                            help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Update the handler if requested
    if args.update_handler:
        logger.info("Updating WebGPU handler with enhanced parallel loading...")
        if enhance_parallel_loading_tracker():
            logger.info("Successfully updated WebGPU handler")
        else:
            logger.error("Failed to update WebGPU handler")
            return 1
    
    # Determine number of iterations
    iterations = args.iterations
    if args.benchmark:
        iterations = 10
    
    # Run tests
    if args.test_all:
        # Test all model types with comparison
        results = run_all_model_comparisons(
            iterations=iterations,
            output_json=args.output_json,
            create_chart=args.create_chart
        )
        
        # Print comparison summary
        print("\nWebGPU Parallel Model Loading Optimization Results")
        print("===================================================\n")
        
        for model_type, comparison in results.items():
            improvements = comparison.get("improvements", {})
            init_improvement = improvements.get("initialization_time_percent", 0)
            load_improvement = improvements.get("load_time_percent", 0)
            
            with_init = comparison.get("with_parallel", {}).get("initialization_time_ms", 0)
            without_init = comparison.get("without_parallel", {}).get("initialization_time_ms", 0)
            
            # Get loading time metrics from both
            with_metrics = comparison.get("with_parallel", {}).get("performance_metrics", {})
            without_metrics = comparison.get("without_parallel", {}).get("performance_metrics", {})
            
            with_load = with_metrics.get("parallel_loading_time_ms", 0)
            if not with_load:
                with_load = with_metrics.get("loading_stats", {}).get("total_loading_time_ms", 0)
                
            without_load = without_metrics.get("sequential_loading_time_ms", 0)
            if not without_load:
                without_load = without_metrics.get("loading_stats", {}).get("total_loading_time_ms", 0)
            
            print(f"{model_type.upper()} Model:")
            print(f"  • Initialization: {with_init:.2f}ms with parallel loading, {without_init:.2f}ms without")
            print(f"    - Improvement: {init_improvement:.2f}%")
            print(f"  • Model Loading: {with_load:.2f}ms with parallel loading, {without_load:.2f}ms without")
            print(f"    - Improvement: {load_improvement:.2f}%\n")
        
        return 0
    else:
        # Test specific model type or model name
        if args.with_parallel_only:
            # Only test with parallel loading
            result = test_webgpu_model(
                model_type=args.model_type,
                model_name=args.model_name,
                parallel_loading=True,
                iterations=iterations
            )
            
            if result.get("success", False):
                init_time = result.get("initialization_time_ms", 0)
                first_time = result.get("first_inference_time_ms", 0)
                load_time = result.get("parallel_load_time_ms", 0)
                
                print(f"\nWebGPU Parallel Loading Test for {result.get('model_name', args.model_name)}")
                print("=====================================================\n")
                print(f"Initialization time: {init_time:.2f} ms")
                print(f"First inference time: {first_time:.2f} ms")
                
                # Print loading details if available
                if load_time > 0:
                    print(f"Parallel model loading time: {load_time:.2f} ms")
                
                # Print component details if available
                performance_metrics = result.get("performance_metrics", {})
                loading_stats = performance_metrics.get("loading_stats", {})
                
                if loading_stats:
                    components = loading_stats.get("components_loaded", 0)
                    memory_peak = loading_stats.get("memory_peak_mb", 0)
                    
                    print(f"\nModel Components: {components}")
                    print(f"Peak Memory: {memory_peak:.2f} MB")
                    
                    # Print individual component sizes if available
                    component_sizes = loading_stats.get("component_sizes_mb", {})
                    if component_sizes:
                        print("\nComponent Sizes:")
                        for component, size in component_sizes.items():
                            print(f"  • {component}: {size:.2f} MB")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
        elif args.without_parallel_only:
            # Only test without parallel loading
            result = test_webgpu_model(
                model_type=args.model_type,
                model_name=args.model_name,
                parallel_loading=False,
                iterations=iterations
            )
            
            if result.get("success", False):
                init_time = result.get("initialization_time_ms", 0)
                first_time = result.get("first_inference_time_ms", 0)
                
                print(f"\nWebGPU Sequential Loading Test for {result.get('model_name', args.model_name)}")
                print("================================================\n")
                print(f"Initialization time: {init_time:.2f} ms")
                print(f"First inference time: {first_time:.2f} ms")
                
                # Print loading details if available from performance metrics
                performance_metrics = result.get("performance_metrics", {})
                loading_stats = performance_metrics.get("loading_stats", {})
                
                if loading_stats:
                    sequential_time = loading_stats.get("sequential_loading_time_ms", 0)
                    components = loading_stats.get("components_loaded", 0)
                    memory_peak = loading_stats.get("memory_peak_mb", 0)
                    
                    print(f"Sequential model loading time: {sequential_time:.2f} ms")
                    print(f"\nModel Components: {components}")
                    print(f"Peak Memory: {memory_peak:.2f} MB")
                    
                    # Print individual component sizes if available
                    component_sizes = loading_stats.get("component_sizes_mb", {})
                    if component_sizes:
                        print("\nComponent Sizes:")
                        for component, size in component_sizes.items():
                            print(f"  • {component}: {size:.2f} MB")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
        else:
            # Run comparison test
            comparison = compare_parallel_loading_options(
                model_type=args.model_type,
                model_name=args.model_name,
                iterations=iterations
            )
            
            # Save results if requested
            if args.output_json:
                with open(args.output_json, 'w') as f:
                    json.dump(comparison, f, indent=2)
                logger.info(f"Results saved to {args.output_json}")
            
            # Create chart if requested
            if args.create_chart:
                model_name = comparison.get("model_name", args.model_name or args.model_type)
                model_name_safe = model_name.replace("/", "_")
                chart_file = f"webgpu_{model_name_safe}_parallel_loading_comparison_{int(time.time())}.png"
                create_performance_chart({model_name: comparison}, chart_file)
            
            # Print comparison
            improvements = comparison.get("improvements", {})
            init_improvement = improvements.get("initialization_time_percent", 0)
            load_improvement = improvements.get("load_time_percent", 0)
            
            with_results = comparison.get("with_parallel", {})
            without_results = comparison.get("without_parallel", {})
            
            with_init = with_results.get("initialization_time_ms", 0)
            without_init = without_results.get("initialization_time_ms", 0)
            
            # Get loading time metrics from both
            with_metrics = with_results.get("performance_metrics", {})
            without_metrics = without_results.get("performance_metrics", {})
            
            with_load = with_metrics.get("parallel_loading_time_ms", 0)
            if not with_load:
                with_load = with_metrics.get("loading_stats", {}).get("total_loading_time_ms", 0)
                
            without_load = without_metrics.get("sequential_loading_time_ms", 0)
            if not without_load:
                without_load = without_metrics.get("loading_stats", {}).get("total_loading_time_ms", 0)
            
            model_name = comparison.get("model_name", args.model_name or args.model_type)
            
            print(f"\nWebGPU Parallel Model Loading Comparison for {model_name}")
            print("==========================================================\n")
            print(f"Initialization Time:")
            print(f"  • With parallel loading: {with_init:.2f} ms")
            print(f"  • Without parallel loading: {without_init:.2f} ms")
            print(f"  • Improvement: {init_improvement:.2f}%\n")
            
            print(f"Model Loading Time:")
            print(f"  • With parallel loading: {with_load:.2f} ms")
            print(f"  • Without parallel loading: {without_load:.2f} ms")
            print(f"  • Improvement: {load_improvement:.2f}%\n")
            
            # Print detailed component information if available
            loading_stats = with_metrics.get("loading_stats", {})
            if loading_stats:
                components = loading_stats.get("components_loaded", 0)
                memory_peak = loading_stats.get("memory_peak_mb", 0)
                
                print(f"Model Components: {components}")
                print(f"Peak Memory: {memory_peak:.2f} MB")
                
                # Print individual component sizes if available
                component_sizes = loading_stats.get("component_sizes_mb", {})
                if component_sizes:
                    print("\nComponent Sizes:")
                    for component, size in component_sizes.items():
                        print(f"  • {component}: {size:.2f} MB")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())