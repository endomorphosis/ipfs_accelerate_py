#!/usr/bin/env python3
"""
Test script for evaluating WebGPU shader precompilation optimizations.

This script specifically tests the enhanced WebGPU shader precompilation implementation,
which improves startup time and initial inference latency for all model types.

Usage:
    python test_webgpu_shader_precompilation.py --model-type text
    python test_webgpu_shader_precompilation.py --model-type vision
    python test_webgpu_shader_precompilation.py --model-type audio
    python test_webgpu_shader_precompilation.py --test-all --benchmark
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
    logging.basicConfig()))))))))))))))
    level=logging.INFO,
    format='%()))))))))))))))asctime)s - %()))))))))))))))levelname)s - %()))))))))))))))message)s'
    )
    logger = logging.getLogger()))))))))))))))"shader_precompilation_test")

# Constants
    TEST_MODELS = {}}}}}}}}}}}}}}}}
    "text": "bert-base-uncased",
    "vision": "google/vit-base-patch16-224",
    "audio": "openai/whisper-tiny",
    "multimodal": "openai/clip-vit-base-patch32"
    }

def setup_environment()))))))))))))))precompile_shaders=True, compute_shaders=False):
    """
    Set up the environment variables for WebGPU testing with shader precompilation.
    
    Args:
        precompile_shaders: Whether to enable shader precompilation
        compute_shaders: Whether to enable compute shaders
        
    Returns:
        True if successful, False otherwise
        """
    # Set WebGPU environment variables
        os.environ["WEBGPU_ENABLED"] = "1",
        os.environ["WEBGPU_SIMULATION"] = "1" ,
        os.environ["WEBGPU_AVAILABLE"] = "1"
        ,
    # Enable shader precompilation if requested:::::::
    if precompile_shaders:
        os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"], = "1",
        logger.info()))))))))))))))"WebGPU shader precompilation enabled")
    else:
        if "WEBGPU_SHADER_PRECOMPILE_ENABLED" in os.environ:
            del os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"],
            logger.info()))))))))))))))"WebGPU shader precompilation disabled")
    
    # Enable compute shaders if requested::::::
    if compute_shaders:
        os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"], = "1",
        logger.info()))))))))))))))"WebGPU compute shaders enabled")
    else:
        if "WEBGPU_COMPUTE_SHADERS_ENABLED" in os.environ:
            del os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"],
            logger.info()))))))))))))))"WebGPU compute shaders disabled")
    
    # Enable parallel loading for multimodal models
            os.environ["WEBGPU_PARALLEL_LOADING_ENABLED"] = "1"
            ,
        return True

def setup_web_platform_handler()))))))))))))))):
    """
    Set up and import the fixed web platform handler.
    
    Returns:
        The imported module or None if failed
    """:
    try:
        # Try to import fixed_web_platform from the current directory
        sys.path.append()))))))))))))))'.')
        from test.web_platform.web_platform_handler import ()))))))))))))))
        process_for_web, init_webgpu, create_mock_processors
        )
        logger.info()))))))))))))))"Successfully imported web platform handler from test.web_platform")
        return {}}}}}}}}}}}}}}}}
        "process_for_web": process_for_web,
        "init_webgpu": init_webgpu,
        "create_mock_processors": create_mock_processors
        }
    except ImportError:
        # Try to import from the test directory
        try:
            sys.path.append()))))))))))))))'test')
            from test.web_platform.web_platform_handler import ()))))))))))))))
            process_for_web, init_webgpu, create_mock_processors
            )
            logger.info()))))))))))))))"Successfully imported web platform handler from test/fixed_web_platform")
        return {}}}}}}}}}}}}}}}}
        "process_for_web": process_for_web,
        "init_webgpu": init_webgpu,
        "create_mock_processors": create_mock_processors
        }
        except ImportError:
            logger.error()))))))))))))))"Failed to import web platform handler from test.web_platform")
        return None

def enhance_shader_compilation_tracker()))))))))))))))):
    """
    Update the ShaderCompilationTracker for enhanced precompilation performance.
    
    This function will modify the web_platform_handler.py file to add enhanced
    shader precompilation capabilities to the ShaderCompilationTracker class.
    """
    # Path to the handler file
    handler_path = "fixed_web_platform/web_platform_handler.py"
    
    # Check if file exists:
    if not os.path.exists()))))))))))))))handler_path):
        handler_path = "test/fixed_web_platform/web_platform_handler.py"
        if not os.path.exists()))))))))))))))handler_path):
            logger.error()))))))))))))))f"Cannot find web_platform_handler.py")
        return False
    
    # Create a backup
        backup_path = f"{}}}}}}}}}}}}}}}}handler_path}.bak"
    with open()))))))))))))))handler_path, 'r') as src:
        with open()))))))))))))))backup_path, 'w') as dst:
            dst.write()))))))))))))))src.read()))))))))))))))))
    
            logger.info()))))))))))))))f"Created backup at {}}}}}}}}}}}}}}}}backup_path}")
    
    # Find the ShaderCompilationTracker class and enhance it
    with open()))))))))))))))handler_path, 'r') as f:
        content = f.read())))))))))))))))
    
    # Replace the basic ShaderCompilationTracker with enhanced version
    basic_tracker = """class ShaderCompilationTracker:
                def __init__()))))))))))))))self):
                    self.shader_compilation_time = None
                    # Simulate the shader compilation process
                    import time
                    start_time = time.time())))))))))))))))
                    # Simulate different compilation times for different model types
                    time.sleep()))))))))))))))0.05)  # 50ms shader compilation time simulation
                    self.shader_compilation_time = ()))))))))))))))time.time()))))))))))))))) - start_time) * 1000  # ms
                    
                def get_shader_compilation_time()))))))))))))))self):
                    return self.shader_compilation_time"""
    
    enhanced_tracker = """class ShaderCompilationTracker:
                def __init__()))))))))))))))self):
                    self.shader_compilation_time = None
                    self.shader_cache = {}}}}}}}}}}}}}}}}}
                    self.precompile_enabled = "WEBGPU_SHADER_PRECOMPILE_ENABLED" in os.environ
                    
                    # Initialize shader compilation statistics
                    self.stats = {}}}}}}}}}}}}}}}}
                    "total_compilation_time_ms": 0,
                    "cached_shaders_used": 0,
                    "new_shaders_compiled": 0,
                    "peak_memory_bytes": 0,
                    "shader_count": 0,
                    "cache_hit_rate": 0.0
                    }
                    
                    # Simulate the shader compilation process
                    import time
                    import random
                    
                    # Determine number of shaders based on model type
                    model_type = getattr()))))))))))))))self, "mode", "unknown")
                    if model_type == "text":
                        shader_count = random.randint()))))))))))))))18, 25)
                    elif model_type == "vision":
                        shader_count = random.randint()))))))))))))))30, 40)
                    elif model_type == "audio":
                        shader_count = random.randint()))))))))))))))25, 35)
                    elif model_type == "multimodal":
                        shader_count = random.randint()))))))))))))))45, 60)
                    else:
                        shader_count = random.randint()))))))))))))))20, 30)
                        
                        self.stats["shader_count"] = shader_count
                        ,
                    # Variable to store total compilation time
                        total_compilation_time = 0
                    
                    # Shader precompilation optimization
                    if self.precompile_enabled:
                        # Precompile most shaders at init time
                        start_time = time.time())))))))))))))))
                        
                        # With precompilation, we compile all shaders at once in parallel
                        # which is much faster than compiling them one by one
                        precompile_time = 0.01 * shader_count  # 10ms per shader but in parallel
                        time.sleep()))))))))))))))precompile_time)  # Simulate bulk precompilation
                        
                        # Store in cache
                        shader_ids = [f"shader_{}}}}}}}}}}}}}}}}i}" for i in range()))))))))))))))shader_count)]:,
                        for shader_id in shader_ids:
                            self.shader_cache[shader_id] = {}}}}}}}}}}}}}}}},,,
                            "compiled": True,
                            "compilation_time": 10.0,  # Average 10ms per shader
                            "size_bytes": random.randint()))))))))))))))5000, 20000)
                            }
                        
                            self.stats["new_shaders_compiled"] = shader_count,
                            self.stats["total_compilation_time_ms"] = precompile_time * 1000,
                            total_compilation_time = precompile_time * 1000
                    else:
                        # Without precompilation, we'll simulate on-demand compilation
                        # This is slower as shaders compile one at a time during inference
                        # We'll simulate this by just tracking the expected time
                        self.stats["new_shaders_compiled"] = 0,
                        self.stats["total_compilation_time_ms"] = 0
                        ,
                    # Calculate peak memory for shader storage
                        total_shader_memory = sum()))))))))))))))
                        shader["size_bytes"] for shader in self.shader_cache.values())))))))))))))))::,,
                        )
                        self.stats["peak_memory_bytes"] = total_shader_memory
                        ,
                    # Store shader compilation time
                        self.shader_compilation_time = total_compilation_time
                    
                def get_shader_compilation_time()))))))))))))))self):
                        return self.shader_compilation_time
                    
                def get_compilation_stats()))))))))))))))self):
                        return self.stats
                
                def use_shader()))))))))))))))self, shader_id):
                    \"\"\"Simulate using a shader, returning performance impact\"\"\"
                    import time
                    import random
                    
                    if not self.precompile_enabled:
                        # If precompilation is disabled, we may need to compile now
                        if shader_id not in self.shader_cache:
                            # Need to compile ()))))))))))))))slow path)
                            compile_start = time.time())))))))))))))))
                            # Simulate compilation of a single shader ()))))))))))))))25-50ms)
                            compile_time = random.uniform()))))))))))))))0.025, 0.05)
                            time.sleep()))))))))))))))compile_time)
                            
                            # Cache shader
                            self.shader_cache[shader_id] = {}}}}}}}}}}}}}}}},,,
                            "compiled": True,
                            "compilation_time": compile_time * 1000,
                            "size_bytes": random.randint()))))))))))))))5000, 20000)
                            }
                            
                            # Update stats
                            self.stats["new_shaders_compiled"] += 1,,
                            self.stats["total_compilation_time_ms"] += compile_time * 1000
                            ,,
                            # Recalculate peak memory
                            total_shader_memory = sum()))))))))))))))
                            shader["size_bytes"] for shader in self.shader_cache.values())))))))))))))))::,,
                            )
                            self.stats["peak_memory_bytes"] = max())))))))))))))),
                            self.stats["peak_memory_bytes"], total_shader_memory,
                            )
                            
                            # Check if this was first shader ()))))))))))))))initialization):
                            if self.stats["new_shaders_compiled"] == 1:,
                            self.shader_compilation_time = compile_time * 1000
                            
                            # Return the time penalty for compiling
                        return compile_time * 1000
                        else:
                            # Shader already compiled, just lookup time ()))))))))))))))no penalty)
                            self.stats["cached_shaders_used"] += 1,,
                        return 0
                    else:
                        # With precompilation, shaders are already ready
                        if shader_id in self.shader_cache:
                            self.stats["cached_shaders_used"] += 1,,
                        return 0
                        else:
                            # Even with precompilation, some shaders might be compiled just-in-time
                            # but this is rare ()))))))))))))))only 5% of shaders)
                            compile_time = random.uniform()))))))))))))))0.01, 0.02)  # 10-20ms
                            
                            # Fast path compilation ()))))))))))))))precompiled context helps)
                            self.shader_cache[shader_id] = {}}}}}}}}}}}}}}}},,,
                            "compiled": True,
                            "compilation_time": compile_time * 1000,
                            "size_bytes": random.randint()))))))))))))))5000, 20000)
                            }
                            
                            # Update stats
                            self.stats["new_shaders_compiled"] += 1,,
                            self.stats["total_compilation_time_ms"] += compile_time * 1000
                            ,,
                            # Return small time penalty
                        return compile_time * 1000
                
                def update_cache_hit_rate()))))))))))))))self):
                    \"\"\"Update the cache hit rate statistic\"\"\"
                    total_shader_uses = self.stats["cached_shaders_used"] + self.stats["new_shaders_compiled"],
                    if total_shader_uses > 0:
                        self.stats["cache_hit_rate"] = self.stats["cached_shaders_used"] / total_shader_uses,
                    else:
                        self.stats["cache_hit_rate"] = 0.0"""
                        ,
    # Replace the implementation
    if basic_tracker in content:
        logger.info()))))))))))))))"Found ShaderCompilationTracker class, enhancing it")
        new_content = content.replace()))))))))))))))basic_tracker, enhanced_tracker)
        
        # Write the updated content
        with open()))))))))))))))handler_path, 'w') as f:
            f.write()))))))))))))))new_content)
        
            logger.info()))))))))))))))"Successfully enhanced ShaderCompilationTracker")
        return True
    else:
        logger.error()))))))))))))))"Could not find ShaderCompilationTracker class to enhance")
        return False

def test_webgpu_model()))))))))))))))model_type, precompile_shaders=True, iterations=5):
    """
    Test a model with WebGPU using shader precompilation.
    
    Args:
        model_type: Type of model to test ()))))))))))))))"text", "vision", "audio", "multimodal")
        precompile_shaders: Whether to use shader precompilation
        iterations: Number of inference iterations
        
    Returns:
        Dictionary with test results
        """
    # Import web platform handler
        handlers = setup_web_platform_handler())))))))))))))))
    if not handlers:
        return {}}}}}}}}}}}}}}}}
        "success": False,
        "error": "Failed to import web platform handler"
        }
    
        process_for_web = handlers["process_for_web"],
        init_webgpu = handlers["init_webgpu"],
        create_mock_processors = handlers["create_mock_processors"]
        ,
    # Set up environment
        setup_environment()))))))))))))))precompile_shaders=precompile_shaders)
    
    # Select model
    if model_type in TEST_MODELS:
        model_name = TEST_MODELS[model_type],
    else:
        return {}}}}}}}}}}}}}}}}
        "success": False,
        "error": f"Unknown model type: {}}}}}}}}}}}}}}}}model_type}"
        }
    
    # Create test class
    class TestModel:
        def __init__()))))))))))))))self):
            self.model_name = model_name
            self.mode = model_type
            self.device = "webgpu"
            self.processors = create_mock_processors())))))))))))))))
    
    # Initialize test model
            test_model = TestModel())))))))))))))))
    
    # Track initial load time
            start_time = time.time())))))))))))))))
    
    # Initialize WebGPU implementation
            processor_key = "image_processor" if model_type == "vision" else None
            result = init_webgpu()))))))))))))))
            test_model,
            model_name=test_model.model_name,
            model_type=test_model.mode,
            device=test_model.device,
            web_api_mode="simulation",
            create_mock_processor=test_model.processors[processor_key]()))))))))))))))) if processor_key else None,
            )
    
    # Calculate initialization time
            init_time = ()))))))))))))))time.time()))))))))))))))) - start_time) * 1000  # ms
    :
    if not result or not isinstance()))))))))))))))result, dict):
        return {}}}}}}}}}}}}}}}}
        "success": False,
        "error": f"Failed to initialize WebGPU for {}}}}}}}}}}}}}}}}model_type}"
        }
    
    # Extract endpoint and check if it's valid
    endpoint = result.get()))))))))))))))"endpoint"):
    if not endpoint:
        return {}}}}}}}}}}}}}}}}
        "success": False,
        "error": f"No endpoint returned for {}}}}}}}}}}}}}}}}model_type}"
        }
    
    # Create appropriate test input based on model type
    if model_type == "text":
        test_input = "This is a test input for text models"
    elif model_type == "vision":
        test_input = "test.jpg"
    elif model_type == "audio":
        test_input = "test.mp3"
    elif model_type == "multimodal":
        test_input = {}}}}}}}}}}}}}}}}"image": "test.jpg", "text": "What is in this image?"}
    else:
        test_input = "Generic test input"
    
    # Process input for WebGPU
        processed_input = process_for_web()))))))))))))))test_model.mode, test_input, False)
    
    # Run initial inference to warm up and track time
    try:
        warm_up_start = time.time())))))))))))))))
        warm_up_result = endpoint()))))))))))))))processed_input)
        first_inference_time = ()))))))))))))))time.time()))))))))))))))) - warm_up_start) * 1000  # ms
    except Exception as e:
        return {}}}}}}}}}}}}}}}}
        "success": False,
        "error": f"Error during warm-up: {}}}}}}}}}}}}}}}}str()))))))))))))))e)}"
        }
    
    # Get implementation details and shader compilation stats
        implementation_type = warm_up_result.get()))))))))))))))"implementation_type", "UNKNOWN")
        performance_metrics = warm_up_result.get()))))))))))))))"performance_metrics", {}}}}}}}}}}}}}}}}})
    
    # Extract shader compilation time if available
        shader_compilation_time = performance_metrics.get()))))))))))))))"shader_compilation_ms", 0)
    
    # Run benchmark iterations
        inference_times = [],,,,,,
    :
    for i in range()))))))))))))))iterations):
        start_time = time.time())))))))))))))))
        inference_result = endpoint()))))))))))))))processed_input)
        end_time = time.time())))))))))))))))
        elapsed_time = ()))))))))))))))end_time - start_time) * 1000  # Convert to ms
        inference_times.append()))))))))))))))elapsed_time)
    
    # Calculate performance metrics
        avg_inference_time = sum()))))))))))))))inference_times) / len()))))))))))))))inference_times) if inference_times else 0
        min_inference_time = min()))))))))))))))inference_times) if inference_times else 0
        max_inference_time = max()))))))))))))))inference_times) if inference_times else 0
        std_dev = ()))))))))))))))
        ()))))))))))))))sum()))))))))))))))()))))))))))))))t - avg_inference_time) ** 2 for t in inference_times) / len()))))))))))))))inference_times)) ** 0.5 
        if len()))))))))))))))inference_times) > 1 else 0
        )
    
    # Create result
    return {}}}}}}}}}}}}}}}}:
        "success": True,
        "model_type": model_type,
        "model_name": model_name,
        "implementation_type": implementation_type,
        "shader_precompilation_enabled": precompile_shaders,
        "initialization_time_ms": init_time,
        "first_inference_time_ms": first_inference_time,
        "shader_compilation_time_ms": shader_compilation_time,
        "performance": {}}}}}}}}}}}}}}}}
        "iterations": iterations,
        "avg_inference_time_ms": avg_inference_time,
        "min_inference_time_ms": min_inference_time,
        "max_inference_time_ms": max_inference_time,
        "std_dev_ms": std_dev
        },
        "performance_metrics": performance_metrics
        }

def compare_precompile_options()))))))))))))))model_type, iterations=5):
    """
    Compare model performance with and without shader precompilation.
    
    Args:
        model_type: Type of model to test
        iterations: Number of inference iterations per configuration
        
    Returns:
        Dictionary with comparison results
        """
    # Run tests with shader precompilation
        with_precompilation = test_webgpu_model()))))))))))))))
        model_type=model_type,
        precompile_shaders=True,
        iterations=iterations
        )
    
    # Run tests without shader precompilation
        without_precompilation = test_webgpu_model()))))))))))))))
        model_type=model_type,
        precompile_shaders=False,
        iterations=iterations
        )
    
    # Calculate improvements
        init_improvement = 0
        first_inference_improvement = 0
        avg_inference_improvement = 0
    
    if ()))))))))))))))with_precompilation.get()))))))))))))))"success", False) and :
        without_precompilation.get()))))))))))))))"success", False)):
        
        # Calculate initialization time improvement
            with_init = with_precompilation.get()))))))))))))))"initialization_time_ms", 0)
            without_init = without_precompilation.get()))))))))))))))"initialization_time_ms", 0)
        
        if without_init > 0:
            init_improvement = ()))))))))))))))without_init - with_init) / without_init * 100
        
        # Calculate first inference time improvement
            with_first = with_precompilation.get()))))))))))))))"first_inference_time_ms", 0)
            without_first = without_precompilation.get()))))))))))))))"first_inference_time_ms", 0)
        
        if without_first > 0:
            first_inference_improvement = ()))))))))))))))without_first - with_first) / without_first * 100
        
        # Calculate average inference time improvement
            with_avg = with_precompilation.get()))))))))))))))"performance", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"avg_inference_time_ms", 0)
            without_avg = without_precompilation.get()))))))))))))))"performance", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"avg_inference_time_ms", 0)
        
        if without_avg > 0:
            avg_inference_improvement = ()))))))))))))))without_avg - with_avg) / without_avg * 100
    
            return {}}}}}}}}}}}}}}}}
            "model_type": model_type,
            "with_precompilation": with_precompilation,
            "without_precompilation": without_precompilation,
            "improvements": {}}}}}}}}}}}}}}}}
            "initialization_time_percent": init_improvement,
            "first_inference_percent": first_inference_improvement,
            "avg_inference_percent": avg_inference_improvement
            }
            }

def run_all_model_comparisons()))))))))))))))iterations=5, output_json=None, create_chart=False):
    """
    Run comparisons for all test model types.
    
    Args:
        iterations: Number of inference iterations per configuration
        output_json: Path to save JSON results
        create_chart: Whether to create a performance comparison chart
        
    Returns:
        Dictionary with all comparison results
        """
        results = {}}}}}}}}}}}}}}}}}
        model_types = list()))))))))))))))TEST_MODELS.keys()))))))))))))))))
    
    for model_type in model_types:
        logger.info()))))))))))))))f"Testing {}}}}}}}}}}}}}}}}model_type} with and without shader precompilation...")
        comparison = compare_precompile_options()))))))))))))))model_type, iterations)
        results[model_type], = comparison
        
        # Print summary
        improvements = comparison.get()))))))))))))))"improvements", {}}}}}}}}}}}}}}}}})
        init_improvement = improvements.get()))))))))))))))"initialization_time_percent", 0)
        first_improvement = improvements.get()))))))))))))))"first_inference_percent", 0)
        
        logger.info()))))))))))))))f"  • {}}}}}}}}}}}}}}}}model_type}: {}}}}}}}}}}}}}}}}init_improvement:.2f}% faster initialization, {}}}}}}}}}}}}}}}}first_improvement:.2f}% faster first inference")
    
    # Save results to JSON if requested::::::
    if output_json:
        with open()))))))))))))))output_json, 'w') as f:
            json.dump()))))))))))))))results, f, indent=2)
            logger.info()))))))))))))))f"Results saved to {}}}}}}}}}}}}}}}}output_json}")
    
    # Create chart if requested::::::
    if create_chart:
        create_performance_chart()))))))))))))))results, f"webgpu_shader_precompilation_comparison_{}}}}}}}}}}}}}}}}int()))))))))))))))time.time()))))))))))))))))}.png")
    
            return results

def create_performance_chart()))))))))))))))results, output_file):
    """
    Create a performance comparison chart.
    
    Args:
        results: Dictionary with comparison results
        output_file: Path to save the chart
        """
    try:
        model_types = list()))))))))))))))results.keys()))))))))))))))))
        with_precompile_init = [],,,,,,
        without_precompile_init = [],,,,,,
        with_precompile_first = [],,,,,,
        without_precompile_first = [],,,,,,
        init_improvements = [],,,,,,
        first_improvements = [],,,,,,
        
        for model_type in model_types:
            comparison = results[model_type],
            
            # Get initialization times
            with_init = comparison.get()))))))))))))))"with_precompilation", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"initialization_time_ms", 0)
            without_init = comparison.get()))))))))))))))"without_precompilation", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"initialization_time_ms", 0)
            
            # Get first inference times
            with_first = comparison.get()))))))))))))))"with_precompilation", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"first_inference_time_ms", 0)
            without_first = comparison.get()))))))))))))))"without_precompilation", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"first_inference_time_ms", 0)
            
            # Get improvement percentages
            improvements = comparison.get()))))))))))))))"improvements", {}}}}}}}}}}}}}}}}})
            init_improvement = improvements.get()))))))))))))))"initialization_time_percent", 0)
            first_improvement = improvements.get()))))))))))))))"first_inference_percent", 0)
            
            # Add to lists for plotting
            with_precompile_init.append()))))))))))))))with_init)
            without_precompile_init.append()))))))))))))))without_init)
            with_precompile_first.append()))))))))))))))with_first)
            without_precompile_first.append()))))))))))))))without_first)
            init_improvements.append()))))))))))))))init_improvement)
            first_improvements.append()))))))))))))))first_improvement)
        
        # Create figure with subplots
            fig, ()))))))))))))))ax1, ax2, ax3) = plt.subplots()))))))))))))))3, 1, figsize=()))))))))))))))12, 18))
        
        # Bar chart for initialization times
            x = range()))))))))))))))len()))))))))))))))model_types))
            width = 0.35
        
            ax1.bar()))))))))))))))[i - width/2 for i in x], without_precompile_init, width, label='Without Precompilation'),
            ax1.bar()))))))))))))))[i + width/2 for i in x], with_precompile_init, width, label='With Precompilation')
            ,
            ax1.set_xlabel()))))))))))))))'Model Types')
            ax1.set_ylabel()))))))))))))))'Initialization Time ()))))))))))))))ms)')
            ax1.set_title()))))))))))))))'WebGPU Initialization Time Comparison')
            ax1.set_xticks()))))))))))))))x)
            ax1.set_xticklabels()))))))))))))))model_types)
            ax1.legend())))))))))))))))
        
        # Add initialization time values on bars
        for i, v in enumerate()))))))))))))))without_precompile_init):
            ax1.text()))))))))))))))i - width/2, v + 5, f"{}}}}}}}}}}}}}}}}v:.1f}", ha='center')
        
        for i, v in enumerate()))))))))))))))with_precompile_init):
            ax1.text()))))))))))))))i + width/2, v + 5, f"{}}}}}}}}}}}}}}}}v:.1f}", ha='center')
        
        # Bar chart for first inference times
            ax2.bar()))))))))))))))[i - width/2 for i in x], without_precompile_first, width, label='Without Precompilation'),
            ax2.bar()))))))))))))))[i + width/2 for i in x], with_precompile_first, width, label='With Precompilation')
            ,
            ax2.set_xlabel()))))))))))))))'Model Types')
            ax2.set_ylabel()))))))))))))))'First Inference Time ()))))))))))))))ms)')
            ax2.set_title()))))))))))))))'WebGPU First Inference Time Comparison')
            ax2.set_xticks()))))))))))))))x)
            ax2.set_xticklabels()))))))))))))))model_types)
            ax2.legend())))))))))))))))
        
        # Add first inference time values on bars
        for i, v in enumerate()))))))))))))))without_precompile_first):
            ax2.text()))))))))))))))i - width/2, v + 5, f"{}}}}}}}}}}}}}}}}v:.1f}", ha='center')
        
        for i, v in enumerate()))))))))))))))with_precompile_first):
            ax2.text()))))))))))))))i + width/2, v + 5, f"{}}}}}}}}}}}}}}}}v:.1f}", ha='center')
        
        # Bar chart for improvement percentages
            ax3.bar()))))))))))))))[i - width/2 for i in x], init_improvements, width, label='Initialization Improvement'),
            ax3.bar()))))))))))))))[i + width/2 for i in x], first_improvements, width, label='First Inference Improvement')
            ,
            ax3.set_xlabel()))))))))))))))'Model Types')
            ax3.set_ylabel()))))))))))))))'Improvement ()))))))))))))))%)')
            ax3.set_title()))))))))))))))'Performance Improvement with Shader Precompilation')
            ax3.set_xticks()))))))))))))))x)
            ax3.set_xticklabels()))))))))))))))model_types)
            ax3.legend())))))))))))))))
        
        # Add improvement percentages on bars
        for i, v in enumerate()))))))))))))))init_improvements):
            ax3.text()))))))))))))))i - width/2, v + 1, f"{}}}}}}}}}}}}}}}}v:.1f}%", ha='center')
        
        for i, v in enumerate()))))))))))))))first_improvements):
            ax3.text()))))))))))))))i + width/2, v + 1, f"{}}}}}}}}}}}}}}}}v:.1f}%", ha='center')
        
            plt.tight_layout())))))))))))))))
            plt.savefig()))))))))))))))output_file)
            plt.close())))))))))))))))
        
            logger.info()))))))))))))))f"Performance chart saved to {}}}}}}}}}}}}}}}}output_file}")
    except Exception as e:
        logger.error()))))))))))))))f"Error creating performance chart: {}}}}}}}}}}}}}}}}e}")

def main()))))))))))))))):
    """Parse arguments and run the tests."""
    parser = argparse.ArgumentParser()))))))))))))))
    description="Test WebGPU shader precompilation optimizations"
    )
    
    # Model selection
    model_group = parser.add_argument_group()))))))))))))))"Model Selection")
    model_group.add_argument()))))))))))))))"--model-type", choices=list()))))))))))))))TEST_MODELS.keys())))))))))))))))), default="text",
    help="Model type to test")
    model_group.add_argument()))))))))))))))"--test-all", action="store_true",
    help="Test all available model types")
    
    # Test options
    test_group = parser.add_argument_group()))))))))))))))"Test Options")
    test_group.add_argument()))))))))))))))"--iterations", type=int, default=5,
    help="Number of inference iterations for each test")
    test_group.add_argument()))))))))))))))"--benchmark", action="store_true",
    help="Run in benchmark mode with 10 iterations")
    test_group.add_argument()))))))))))))))"--with-precompile-only", action="store_true",
    help="Only test with shader precompilation enabled")
    test_group.add_argument()))))))))))))))"--without-precompile-only", action="store_true",
    help="Only test without shader precompilation")
    
    # Setup options
    setup_group = parser.add_argument_group()))))))))))))))"Setup Options")
    setup_group.add_argument()))))))))))))))"--update-handler", action="store_true",
    help="Update the WebGPU handler with enhanced shader precompilation")
    
    # Output options
    output_group = parser.add_argument_group()))))))))))))))"Output Options")
    output_group.add_argument()))))))))))))))"--output-json", type=str,
    help="Save results to JSON file")
    output_group.add_argument()))))))))))))))"--create-chart", action="store_true",
    help="Create performance comparison chart")
    output_group.add_argument()))))))))))))))"--verbose", action="store_true",
    help="Enable verbose output")
    
    args = parser.parse_args())))))))))))))))
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel()))))))))))))))logging.DEBUG)
    
    # Update the handler if requested::::::
    if args.update_handler:
        logger.info()))))))))))))))"Updating WebGPU handler with enhanced shader precompilation...")
        if enhance_shader_compilation_tracker()))))))))))))))):
            logger.info()))))))))))))))"Successfully updated WebGPU handler")
        else:
            logger.error()))))))))))))))"Failed to update WebGPU handler")
            return 1
    
    # Determine number of iterations
            iterations = args.iterations
    if args.benchmark:
        iterations = 10
    
    # Run tests
    if args.test_all:
        # Test all model types with comparison
        results = run_all_model_comparisons()))))))))))))))
        iterations=iterations,
        output_json=args.output_json,
        create_chart=args.create_chart
        )
        
        # Print comparison summary
        print()))))))))))))))"\nWebGPU Shader Precompilation Optimization Results")
        print()))))))))))))))"=================================================\n")
        
        for model_type, comparison in results.items()))))))))))))))):
            improvements = comparison.get()))))))))))))))"improvements", {}}}}}}}}}}}}}}}}})
            init_improvement = improvements.get()))))))))))))))"initialization_time_percent", 0)
            first_improvement = improvements.get()))))))))))))))"first_inference_percent", 0)
            avg_improvement = improvements.get()))))))))))))))"avg_inference_percent", 0)
            
            with_init = comparison.get()))))))))))))))"with_precompilation", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"initialization_time_ms", 0)
            without_init = comparison.get()))))))))))))))"without_precompilation", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"initialization_time_ms", 0)
            
            with_first = comparison.get()))))))))))))))"with_precompilation", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"first_inference_time_ms", 0)
            without_first = comparison.get()))))))))))))))"without_precompilation", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"first_inference_time_ms", 0)
            
            with_avg = comparison.get()))))))))))))))"with_precompilation", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"performance", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"avg_inference_time_ms", 0)
            without_avg = comparison.get()))))))))))))))"without_precompilation", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"performance", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"avg_inference_time_ms", 0)
            
            print()))))))))))))))f"{}}}}}}}}}}}}}}}}model_type.upper())))))))))))))))} Model:")
            print()))))))))))))))f"  • Initialization: {}}}}}}}}}}}}}}}}with_init:.2f}ms with precompilation, {}}}}}}}}}}}}}}}}without_init:.2f}ms without")
            print()))))))))))))))f"    - Improvement: {}}}}}}}}}}}}}}}}init_improvement:.2f}%")
            print()))))))))))))))f"  • First Inference: {}}}}}}}}}}}}}}}}with_first:.2f}ms with precompilation, {}}}}}}}}}}}}}}}}without_first:.2f}ms without")
            print()))))))))))))))f"    - Improvement: {}}}}}}}}}}}}}}}}first_improvement:.2f}%")
            print()))))))))))))))f"  • Average Inference: {}}}}}}}}}}}}}}}}with_avg:.2f}ms with precompilation, {}}}}}}}}}}}}}}}}without_avg:.2f}ms without")
            print()))))))))))))))f"    - Improvement: {}}}}}}}}}}}}}}}}avg_improvement:.2f}%\n")
        
        return 0
    else:
        # Test specific model type
        if args.with_precompile_only:
            # Only test with shader precompilation
            result = test_webgpu_model()))))))))))))))
            model_type=args.model_type,
            precompile_shaders=True,
            iterations=iterations
            )
            
            if result.get()))))))))))))))"success", False):
                init_time = result.get()))))))))))))))"initialization_time_ms", 0)
                first_time = result.get()))))))))))))))"first_inference_time_ms", 0)
                avg_time = result.get()))))))))))))))"performance", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"avg_inference_time_ms", 0)
                
                print()))))))))))))))f"\nWebGPU Shader Precompilation Test for {}}}}}}}}}}}}}}}}args.model_type.upper())))))))))))))))}")
                print()))))))))))))))"=====================================================\n")
                print()))))))))))))))f"Initialization time: {}}}}}}}}}}}}}}}}init_time:.2f} ms")
                print()))))))))))))))f"First inference time: {}}}}}}}}}}}}}}}}first_time:.2f} ms")
                print()))))))))))))))f"Average inference time: {}}}}}}}}}}}}}}}}avg_time:.2f} ms")
                
                # Print shader compilation details if available
                shader_time = result.get()))))))))))))))"shader_compilation_time_ms", 0)::
                if shader_time > 0:
                    print()))))))))))))))f"Shader compilation time: {}}}}}}}}}}}}}}}}shader_time:.2f} ms")
                
                    performance_metrics = result.get()))))))))))))))"performance_metrics", {}}}}}}}}}}}}}}}}})
                if performance_metrics:
                    print()))))))))))))))"\nPerformance Metrics:")
                    for key, value in performance_metrics.items()))))))))))))))):
                        if isinstance()))))))))))))))value, dict):
                            print()))))))))))))))f"  • {}}}}}}}}}}}}}}}}key}:")
                            for subkey, subvalue in value.items()))))))))))))))):
                                print()))))))))))))))f"    - {}}}}}}}}}}}}}}}}subkey}: {}}}}}}}}}}}}}}}}subvalue}")
                        else:
                            print()))))))))))))))f"  • {}}}}}}}}}}}}}}}}key}: {}}}}}}}}}}}}}}}}value}")
            else:
                print()))))))))))))))f"Error: {}}}}}}}}}}}}}}}}result.get()))))))))))))))'error', 'Unknown error')}")
                            return 1
        elif args.without_precompile_only:
            # Only test without shader precompilation
            result = test_webgpu_model()))))))))))))))
            model_type=args.model_type,
            precompile_shaders=False,
            iterations=iterations
            )
            
            if result.get()))))))))))))))"success", False):
                init_time = result.get()))))))))))))))"initialization_time_ms", 0)
                first_time = result.get()))))))))))))))"first_inference_time_ms", 0)
                avg_time = result.get()))))))))))))))"performance", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"avg_inference_time_ms", 0)
                
                print()))))))))))))))f"\nWebGPU Standard Test for {}}}}}}}}}}}}}}}}args.model_type.upper())))))))))))))))}")
                print()))))))))))))))"========================================\n")
                print()))))))))))))))f"Initialization time: {}}}}}}}}}}}}}}}}init_time:.2f} ms")
                print()))))))))))))))f"First inference time: {}}}}}}}}}}}}}}}}first_time:.2f} ms")
                print()))))))))))))))f"Average inference time: {}}}}}}}}}}}}}}}}avg_time:.2f} ms")
                
                # Print shader compilation details if available
                shader_time = result.get()))))))))))))))"shader_compilation_time_ms", 0)::
                if shader_time > 0:
                    print()))))))))))))))f"Shader compilation time: {}}}}}}}}}}}}}}}}shader_time:.2f} ms")
            else:
                print()))))))))))))))f"Error: {}}}}}}}}}}}}}}}}result.get()))))))))))))))'error', 'Unknown error')}")
                    return 1
        else:
            # Run comparison test
            comparison = compare_precompile_options()))))))))))))))
            model_type=args.model_type,
            iterations=iterations
            )
            
            # Save results if requested::::::
            if args.output_json:
                with open()))))))))))))))args.output_json, 'w') as f:
                    json.dump()))))))))))))))comparison, f, indent=2)
                    logger.info()))))))))))))))f"Results saved to {}}}}}}}}}}}}}}}}args.output_json}")
            
            # Create chart if requested::::::
            if args.create_chart:
                chart_file = f"webgpu_{}}}}}}}}}}}}}}}}args.model_type}_precompilation_comparison_{}}}}}}}}}}}}}}}}int()))))))))))))))time.time()))))))))))))))))}.png"
                create_performance_chart())))))))))))))){}}}}}}}}}}}}}}}}args.model_type: comparison}, chart_file)
            
            # Print comparison
                improvements = comparison.get()))))))))))))))"improvements", {}}}}}}}}}}}}}}}}})
                init_improvement = improvements.get()))))))))))))))"initialization_time_percent", 0)
                first_improvement = improvements.get()))))))))))))))"first_inference_percent", 0)
                avg_improvement = improvements.get()))))))))))))))"avg_inference_percent", 0)
            
                with_results = comparison.get()))))))))))))))"with_precompilation", {}}}}}}}}}}}}}}}}})
                without_results = comparison.get()))))))))))))))"without_precompilation", {}}}}}}}}}}}}}}}}})
            
                with_init = with_results.get()))))))))))))))"initialization_time_ms", 0)
                without_init = without_results.get()))))))))))))))"initialization_time_ms", 0)
            
                with_first = with_results.get()))))))))))))))"first_inference_time_ms", 0)
                without_first = without_results.get()))))))))))))))"first_inference_time_ms", 0)
            
                with_avg = with_results.get()))))))))))))))"performance", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"avg_inference_time_ms", 0)
                without_avg = without_results.get()))))))))))))))"performance", {}}}}}}}}}}}}}}}}}).get()))))))))))))))"avg_inference_time_ms", 0)
            
                print()))))))))))))))f"\nWebGPU Shader Precompilation Comparison for {}}}}}}}}}}}}}}}}args.model_type.upper())))))))))))))))}")
                print()))))))))))))))"==================================================================\n")
                print()))))))))))))))f"Initialization Time:")
                print()))))))))))))))f"  • With precompilation: {}}}}}}}}}}}}}}}}with_init:.2f} ms")
                print()))))))))))))))f"  • Without precompilation: {}}}}}}}}}}}}}}}}without_init:.2f} ms")
                print()))))))))))))))f"  • Improvement: {}}}}}}}}}}}}}}}}init_improvement:.2f}%\n")
            
                print()))))))))))))))f"First Inference Time:")
                print()))))))))))))))f"  • With precompilation: {}}}}}}}}}}}}}}}}with_first:.2f} ms")
                print()))))))))))))))f"  • Without precompilation: {}}}}}}}}}}}}}}}}without_first:.2f} ms")
                print()))))))))))))))f"  • Improvement: {}}}}}}}}}}}}}}}}first_improvement:.2f}%\n")
            
                print()))))))))))))))f"Average Inference Time:")
                print()))))))))))))))f"  • With precompilation: {}}}}}}}}}}}}}}}}with_avg:.2f} ms")
                print()))))))))))))))f"  • Without precompilation: {}}}}}}}}}}}}}}}}without_avg:.2f} ms")
                print()))))))))))))))f"  • Improvement: {}}}}}}}}}}}}}}}}avg_improvement:.2f}%")
        
                    return 0

if __name__ == "__main__":
    sys.exit()))))))))))))))main()))))))))))))))))