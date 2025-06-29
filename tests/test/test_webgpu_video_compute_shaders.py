#!/usr/bin/env python3
"""
Test script for evaluating WebGPU compute shader optimizations for video models.

This script tests the enhanced WebGPU compute shader implementation
for video models like XCLIP, measuring performance improvements
compared to standard WebGPU implementation.

Usage:
    python test_webgpu_video_compute_shaders.py --model xclip
    python test_webgpu_video_compute_shaders.py --model video_swin
    python test_webgpu_video_compute_shaders.py --test-all --benchmark
    """

    import os
    import sys
    import json
    import time
    import argparse
    import logging
    import matplotlib.pyplot as plt
    from pathlib import Path
    from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to sys.path
    parent_dir = os.path.dirname()))))))))))))))os.path.dirname()))))))))))))))os.path.abspath()))))))))))))))__file__)))
if parent_dir not in sys.path:
    sys.path.append()))))))))))))))parent_dir)

# Configure logging
    logging.basicConfig()))))))))))))))
    level=logging.INFO,
    format='%()))))))))))))))asctime)s - %()))))))))))))))levelname)s - %()))))))))))))))message)s'
    )
    logger = logging.getLogger()))))))))))))))"webgpu_video_compute_test")

# Define test models
    TEST_MODELS = {}}}}}}}}
    "xclip": "microsoft/xclip-base-patch32",
    "video_swin": "MCG-NJU/videoswin-base-patch244-window877-kinetics400-pt",
    "vivit": "google/vivit-b-16x2-kinetics400"
    }

def setup_environment()))))))))))))))compute_shaders_enabled=True, shader_precompile=True):
    """
    Set up the environment variables for WebGPU testing with compute shaders.
    
    Args:
        compute_shaders_enabled: Whether to enable compute shaders
        shader_precompile: Whether to enable shader precompilation
        
    Returns:
        True if successful, False otherwise
        """
    # Set WebGPU environment variables
        os.environ["WEBGPU_ENABLED"] = "1",
        os.environ["WEBGPU_SIMULATION"] = "1" ,
        os.environ["WEBGPU_AVAILABLE"] = "1"
        ,
    # Enable compute shaders if requested:::::::
    if compute_shaders_enabled:
        os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"], = "1",
        logger.info()))))))))))))))"WebGPU compute shaders enabled")
    else:
        if "WEBGPU_COMPUTE_SHADERS_ENABLED" in os.environ:
            del os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"],
            logger.info()))))))))))))))"WebGPU compute shaders disabled")
    
    # Enable shader precompilation if requested::::::
    if shader_precompile:
        os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"], = "1",
        logger.info()))))))))))))))"WebGPU shader precompilation enabled")
    else:
        if "WEBGPU_SHADER_PRECOMPILE_ENABLED" in os.environ:
            del os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"],
            logger.info()))))))))))))))"WebGPU shader precompilation disabled")
    
    # Enable parallel loading for multimodal models
            os.environ["WEBGPU_PARALLEL_LOADING_ENABLED"] = "1"
            ,
        return True

def import_webgpu_video_compute_shaders()))))))))))))))):
    """
    Import the WebGPU video compute shaders module.
    
    Returns:
        The imported module or None if failed
    """:
    try:
        # Try to import from the fixed_web_platform directory
        from fixed_web_platform.webgpu_video_compute_shaders import ()))))))))))))))
        setup_video_compute_shaders, get_supported_video_models
        )
        logger.info()))))))))))))))"Successfully imported WebGPU video compute shaders module")
        return {}}}}}}}}
        "setup_video_compute_shaders": setup_video_compute_shaders,
        "get_supported_video_models": get_supported_video_models
        }
    except ImportError as e:
        logger.error()))))))))))))))f"Failed to import WebGPU video compute shaders module: {}}}}}}}}str()))))))))))))))e)}")
        return None

def test_video_model()))))))))))))))model_name, compute_shaders=True, iterations=5, frame_count=8):
    """
    Test a video model with WebGPU implementation.
    
    Args:
        model_name: Name of the model to test
        compute_shaders: Whether to use compute shaders
        iterations: Number of inference iterations
        frame_count: Number of video frames to process
        
    Returns:
        Dictionary with test results
        """
    # Import WebGPU video compute shaders
        modules = import_webgpu_video_compute_shaders())))))))))))))))
    if not modules:
        return {}}}}}}}}
        "success": False,
        "error": "Failed to import WebGPU video compute shaders module"
        }
    
        setup_video_compute_shaders = modules["setup_video_compute_shaders"]
        ,
    # Set up environment
        setup_environment()))))))))))))))compute_shaders_enabled=compute_shaders)
    
    # Select model
    if model_name in TEST_MODELS:
        model_hf_name = TEST_MODELS[model_name],
    else:
        model_hf_name = model_name
    
    # Create WebGPU compute shaders instance
        compute_shader = setup_video_compute_shaders()))))))))))))))
        model_name=model_hf_name,
        model_type=model_name,
        frame_count=frame_count
        )
    
    # Run initial inference to warm up
        compute_shader.process_video_frames())))))))))))))))
    
    # Run benchmark iterations
        processing_times = [],,,,,
        memory_usages = [],,,,,
    
    for i in range()))))))))))))))iterations):
        # Process video frames
        metrics = compute_shader.process_video_frames())))))))))))))))
        
        # Extract metrics
        processing_time = metrics.get()))))))))))))))"total_compute_time_ms", 0)
        memory_reduction = metrics.get()))))))))))))))"memory_reduction_percent", 0)
        
        processing_times.append()))))))))))))))processing_time)
        memory_usages.append()))))))))))))))memory_reduction)
    
    # Calculate performance metrics
        avg_processing_time = sum()))))))))))))))processing_times) / len()))))))))))))))processing_times) if processing_times else 0
        min_processing_time = min()))))))))))))))processing_times) if processing_times else 0
        max_processing_time = max()))))))))))))))processing_times) if processing_times else 0
        std_dev = ()))))))))))))))
        ()))))))))))))))sum()))))))))))))))()))))))))))))))t - avg_processing_time) ** 2 for t in processing_times) / len()))))))))))))))processing_times)) ** 0.5 
        if len()))))))))))))))processing_times) > 1 else 0
        )
    
    # Get compute shader configuration
        compute_config = metrics.get()))))))))))))))"compute_shader_config", {}}}}}}}}})
    
    # Create result
    return {}}}}}}}}:
        "success": True,
        "model_name": model_name,
        "model_hf_name": model_hf_name,
        "compute_shaders_enabled": compute_shaders,
        "frame_count": frame_count,
        "performance": {}}}}}}}}
        "iterations": iterations,
        "avg_processing_time_ms": avg_processing_time,
        "min_processing_time_ms": min_processing_time,
        "max_processing_time_ms": max_processing_time,
        "std_dev_ms": std_dev,
        "frame_processing_time_ms": metrics.get()))))))))))))))"frame_processing_time_ms", 0),
        "temporal_fusion_time_ms": metrics.get()))))))))))))))"temporal_fusion_time_ms", 0),
            "memory_reduction_percent": sum()))))))))))))))memory_usages) / len()))))))))))))))memory_usages) if memory_usages else 0,:
                "estimated_speedup": metrics.get()))))))))))))))"estimated_speedup", 1.0)
                },
                "compute_shader_config": compute_config
                }

def compare_with_without_compute_shaders()))))))))))))))model_name, iterations=5, frame_count=8):
    """
    Compare model performance with and without compute shaders.
    
    Args:
        model_name: Name of the model to test
        iterations: Number of inference iterations per configuration
        frame_count: Number of video frames to process
        
    Returns:
        Dictionary with comparison results
        """
        logger.info()))))))))))))))f"Testing {}}}}}}}}model_name} with {}}}}}}}}frame_count} frames")
    # Run tests with compute shaders
        with_compute_shaders = test_video_model()))))))))))))))
        model_name=model_name,
        compute_shaders=True,
        iterations=iterations,
        frame_count=frame_count
        )
    
    # Run tests without compute shaders
        without_compute_shaders = test_video_model()))))))))))))))
        model_name=model_name,
        compute_shaders=False,
        iterations=iterations,
        frame_count=frame_count
        )
    
    # Calculate improvement
        improvement = 0
    if ()))))))))))))))with_compute_shaders.get()))))))))))))))"success", False) and ::
        without_compute_shaders.get()))))))))))))))"success", False)):
        
            with_time = with_compute_shaders.get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            without_time = without_compute_shaders.get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
        
        if without_time > 0:
            improvement = ()))))))))))))))without_time - with_time) / without_time * 100
    
            return {}}}}}}}}
            "model_name": model_name,
            "frame_count": frame_count,
            "with_compute_shaders": with_compute_shaders,
            "without_compute_shaders": without_compute_shaders,
            "improvement_percentage": improvement
            }

def run_all_model_comparisons()))))))))))))))iterations=5, output_json=None, create_chart=False, frame_count=8):
    """
    Run comparisons for all test models.
    
    Args:
        iterations: Number of inference iterations per configuration
        output_json: Path to save JSON results
        create_chart: Whether to create a performance comparison chart
        frame_count: Number of video frames to process
        
    Returns:
        Dictionary with all comparison results
        """
        results = {}}}}}}}}}
        models = list()))))))))))))))TEST_MODELS.keys()))))))))))))))))
    
    for model in models:
        logger.info()))))))))))))))f"Testing {}}}}}}}}model} with and without compute shaders...")
        comparison = compare_with_without_compute_shaders()))))))))))))))model, iterations, frame_count)
        results[model], = comparison
        ,
        # Print summary
        improvement = comparison.get()))))))))))))))"improvement_percentage", 0)
        logger.info()))))))))))))))f"  • {}}}}}}}}model}: {}}}}}}}}improvement:.2f}% improvement with compute shaders")
    
    # Save results to JSON if requested::::::
    if output_json:
        with open()))))))))))))))output_json, 'w') as f:
            json.dump()))))))))))))))results, f, indent=2)
            logger.info()))))))))))))))f"Results saved to {}}}}}}}}output_json}")
    
    # Create chart if requested::::::
    if create_chart:
        create_performance_chart()))))))))))))))results, f"webgpu_video_compute_shader_comparison_{}}}}}}}}int()))))))))))))))time.time()))))))))))))))))}.png")
    
            return results

def create_performance_chart()))))))))))))))results, output_file):
    """
    Create a performance comparison chart.
    
    Args:
        results: Dictionary with comparison results
        output_file: Path to save the chart
        """
    try:
        models = list()))))))))))))))results.keys()))))))))))))))))
        with_compute = [],,,,,
        without_compute = [],,,,,
        improvements = [],,,,,
        
        for model in models:
            comparison = results[model],
            with_time = comparison.get()))))))))))))))"with_compute_shaders", {}}}}}}}}}).get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            without_time = comparison.get()))))))))))))))"without_compute_shaders", {}}}}}}}}}).get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            improvement = comparison.get()))))))))))))))"improvement_percentage", 0)
            
            with_compute.append()))))))))))))))with_time)
            without_compute.append()))))))))))))))without_time)
            improvements.append()))))))))))))))improvement)
        
        # Create figure with two subplots
            fig, ()))))))))))))))ax1, ax2) = plt.subplots()))))))))))))))1, 2, figsize=()))))))))))))))14, 6))
        
        # Bar chart for processing times
            x = range()))))))))))))))len()))))))))))))))models))
            width = 0.35
        
            ax1.bar()))))))))))))))[i - width/2 for i in x], without_compute, width, label='Without Compute Shaders'),
            ax1.bar()))))))))))))))[i + width/2 for i in x], with_compute, width, label='With Compute Shaders')
            ,
            ax1.set_xlabel()))))))))))))))'Models')
            ax1.set_ylabel()))))))))))))))'Processing Time ()))))))))))))))ms)')
            ax1.set_title()))))))))))))))'WebGPU Video Processing Time Comparison')
            ax1.set_xticks()))))))))))))))x)
            ax1.set_xticklabels()))))))))))))))models)
            ax1.legend())))))))))))))))
        
        # Add processing time values on bars
        for i, v in enumerate()))))))))))))))without_compute):
            ax1.text()))))))))))))))i - width/2, v + 1, f"{}}}}}}}}v:.1f}", ha='center')
        
        for i, v in enumerate()))))))))))))))with_compute):
            ax1.text()))))))))))))))i + width/2, v + 1, f"{}}}}}}}}v:.1f}", ha='center')
        
        # Bar chart for improvements
            ax2.bar()))))))))))))))models, improvements, color='green')
            ax2.set_xlabel()))))))))))))))'Models')
            ax2.set_ylabel()))))))))))))))'Improvement ()))))))))))))))%)')
            ax2.set_title()))))))))))))))'Performance Improvement with Compute Shaders')
        
        # Add improvement values on bars
        for i, v in enumerate()))))))))))))))improvements):
            ax2.text()))))))))))))))i, v + 0.5, f"{}}}}}}}}v:.1f}%", ha='center')
        
            plt.tight_layout())))))))))))))))
            plt.savefig()))))))))))))))output_file)
            plt.close())))))))))))))))
        
            logger.info()))))))))))))))f"Performance chart saved to {}}}}}}}}output_file}")
    except Exception as e:
        logger.error()))))))))))))))f"Error creating performance chart: {}}}}}}}}e}")

        def test_frame_count_scaling()))))))))))))))model_name, iterations=3, frame_counts=[4, 8, 16, 24, 32],):,
        """
        Test how model performance scales with different frame counts.
    
    Args:
        model_name: Name of the model to test
        iterations: Number of inference iterations per configuration
        frame_counts: List of frame counts to test
        
    Returns:
        Dictionary with scaling results
        """
        logger.info()))))))))))))))f"Testing {}}}}}}}}model_name} scaling with different frame counts")
        scaling_results = {}}}}}}}}}
    
    for frame_count in frame_counts:
        # Run tests with compute shaders
        with_compute_shaders = test_video_model()))))))))))))))
        model_name=model_name,
        compute_shaders=True,
        iterations=iterations,
        frame_count=frame_count
        )
        
        # Run tests without compute shaders
        without_compute_shaders = test_video_model()))))))))))))))
        model_name=model_name,
        compute_shaders=False,
        iterations=iterations,
        frame_count=frame_count
        )
        
        # Calculate improvement
        improvement = 0
        if ()))))))))))))))with_compute_shaders.get()))))))))))))))"success", False) and ::
            without_compute_shaders.get()))))))))))))))"success", False)):
            
                with_time = with_compute_shaders.get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
                without_time = without_compute_shaders.get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            
            if without_time > 0:
                improvement = ()))))))))))))))without_time - with_time) / without_time * 100
        
                scaling_results[frame_count] = {}}}}}}}},
                "with_compute_shaders": with_compute_shaders,
                "without_compute_shaders": without_compute_shaders,
                "improvement_percentage": improvement
                }
        
                logger.info()))))))))))))))f"  • {}}}}}}}}frame_count} frames: {}}}}}}}}improvement:.2f}% improvement with compute shaders")
    
                return {}}}}}}}}
                "model_name": model_name,
                "frame_counts": frame_counts,
                "scaling_results": scaling_results
                }

def create_scaling_chart()))))))))))))))scaling_data, output_file):
    """
    Create a chart showing performance scaling with different frame counts.
    
    Args:
        scaling_data: Scaling test results
        output_file: Path to save the chart
        """
    try:
        model_name = scaling_data.get()))))))))))))))"model_name", "Unknown")
        frame_counts = scaling_data.get()))))))))))))))"frame_counts", [],,,,,)
        scaling_results = scaling_data.get()))))))))))))))"scaling_results", {}}}}}}}}})
        
        with_compute_times = [],,,,,
        without_compute_times = [],,,,,
        improvements = [],,,,,
        
        for frame_count in frame_counts:
            result = scaling_results.get()))))))))))))))frame_count, {}}}}}}}}})
            with_time = result.get()))))))))))))))"with_compute_shaders", {}}}}}}}}}).get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            without_time = result.get()))))))))))))))"without_compute_shaders", {}}}}}}}}}).get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            improvement = result.get()))))))))))))))"improvement_percentage", 0)
            
            with_compute_times.append()))))))))))))))with_time)
            without_compute_times.append()))))))))))))))without_time)
            improvements.append()))))))))))))))improvement)
        
        # Create figure with two subplots
            fig, ()))))))))))))))ax1, ax2) = plt.subplots()))))))))))))))1, 2, figsize=()))))))))))))))14, 6))
        
        # Line chart for processing times
            ax1.plot()))))))))))))))frame_counts, without_compute_times, 'o-', label='Without Compute Shaders')
            ax1.plot()))))))))))))))frame_counts, with_compute_times, 'o-', label='With Compute Shaders')
        
            ax1.set_xlabel()))))))))))))))'Frame Count')
            ax1.set_ylabel()))))))))))))))'Processing Time ()))))))))))))))ms)')
            ax1.set_title()))))))))))))))f'{}}}}}}}}model_name} Processing Time vs. Frame Count')
            ax1.legend())))))))))))))))
            ax1.grid()))))))))))))))True)
        
        # Line chart for improvements
            ax2.plot()))))))))))))))frame_counts, improvements, 'o-', color='green')
            ax2.set_xlabel()))))))))))))))'Frame Count')
            ax2.set_ylabel()))))))))))))))'Improvement ()))))))))))))))%)')
            ax2.set_title()))))))))))))))f'{}}}}}}}}model_name} Performance Improvement vs. Frame Count')
            ax2.grid()))))))))))))))True)
        
            plt.tight_layout())))))))))))))))
            plt.savefig()))))))))))))))output_file)
            plt.close())))))))))))))))
        
            logger.info()))))))))))))))f"Scaling chart saved to {}}}}}}}}output_file}")
    except Exception as e:
        logger.error()))))))))))))))f"Error creating scaling chart: {}}}}}}}}e}")

def main()))))))))))))))):
    """Parse arguments and run the tests."""
    parser = argparse.ArgumentParser()))))))))))))))
    description="Test WebGPU compute shader optimizations for video models"
    )
    
    # Model selection
    model_group = parser.add_argument_group()))))))))))))))"Model Selection")
    model_group.add_argument()))))))))))))))"--model", choices=list()))))))))))))))TEST_MODELS.keys())))))))))))))))), default="xclip",
    help="Video model to test")
    model_group.add_argument()))))))))))))))"--test-all", action="store_true",
    help="Test all available video models")
    
    # Test options
    test_group = parser.add_argument_group()))))))))))))))"Test Options")
    test_group.add_argument()))))))))))))))"--iterations", type=int, default=5,
    help="Number of inference iterations for each test")
    test_group.add_argument()))))))))))))))"--benchmark", action="store_true",
    help="Run in benchmark mode with 20 iterations")
    test_group.add_argument()))))))))))))))"--with-compute-only", action="store_true",
    help="Only test with compute shaders enabled")
    test_group.add_argument()))))))))))))))"--without-compute-only", action="store_true",
    help="Only test without compute shaders")
    test_group.add_argument()))))))))))))))"--frame-count", type=int, default=8,
    help="Number of video frames to process")
    test_group.add_argument()))))))))))))))"--test-scaling", action="store_true",
    help="Test performance scaling with different frame counts")
    
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
    
    # Determine number of iterations
        iterations = args.iterations
    if args.benchmark:
        iterations = 20
    
    # If testing frame count scaling
    if args.test_scaling:
        scaling_data = test_frame_count_scaling()))))))))))))))
        model_name=args.model,
        iterations=max()))))))))))))))2, iterations // 3),  # Reduce iterations for scaling test
        frame_counts=[4, 8, 16, 24, 32],
        )
        
        # Save results to JSON if requested::::::
        if args.output_json:
            output_json = args.output_json
            if not output_json.endswith()))))))))))))))".json"):
                output_json = f"{}}}}}}}}output_json}_scaling.json"
            
            with open()))))))))))))))output_json, 'w') as f:
                json.dump()))))))))))))))scaling_data, f, indent=2)
                logger.info()))))))))))))))f"Scaling results saved to {}}}}}}}}output_json}")
        
        # Create chart
                create_scaling_chart()))))))))))))))
                scaling_data=scaling_data,
                output_file=f"webgpu_{}}}}}}}}args.model}_scaling_{}}}}}}}}int()))))))))))))))time.time()))))))))))))))))}.png"
                )
        
        # Print summary
                print()))))))))))))))"\nWebGPU Compute Shader Scaling Results")
                print()))))))))))))))"=====================================\n")
                print()))))))))))))))f"Model: {}}}}}}}}args.model.upper())))))))))))))))}\n")
        
                frame_counts = scaling_data.get()))))))))))))))"frame_counts", [],,,,,)
                scaling_results = scaling_data.get()))))))))))))))"scaling_results", {}}}}}}}}})
        
                print()))))))))))))))"Frame Count | Improvement | With Compute | Without Compute")
                print()))))))))))))))"-----------|-------------|-------------|----------------")
        
        for frame_count in frame_counts:
            result = scaling_results.get()))))))))))))))frame_count, {}}}}}}}}})
            improvement = result.get()))))))))))))))"improvement_percentage", 0)
            with_time = result.get()))))))))))))))"with_compute_shaders", {}}}}}}}}}).get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            without_time = result.get()))))))))))))))"without_compute_shaders", {}}}}}}}}}).get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            
            print()))))))))))))))f"{}}}}}}}}frame_count:>10} | {}}}}}}}}improvement:>10.2f}% | {}}}}}}}}with_time:>11.2f}ms | {}}}}}}}}without_time:>14.2f}ms")
        
                return 0
    
    # Run tests
    if args.test_all:
        # Test all models with comparison
        results = run_all_model_comparisons()))))))))))))))
        iterations=iterations,
        output_json=args.output_json,
        create_chart=args.create_chart,
        frame_count=args.frame_count
        )
        
        # Print comparison summary
        print()))))))))))))))"\nWebGPU Video Compute Shader Optimization Results")
        print()))))))))))))))"==============================================\n")
        
        for model, comparison in results.items()))))))))))))))):
            improvement = comparison.get()))))))))))))))"improvement_percentage", 0)
            with_time = comparison.get()))))))))))))))"with_compute_shaders", {}}}}}}}}}).get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            without_time = comparison.get()))))))))))))))"without_compute_shaders", {}}}}}}}}}).get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            
            print()))))))))))))))f"{}}}}}}}}model.upper())))))))))))))))} Model:")
            print()))))))))))))))f"  • With compute shaders: {}}}}}}}}with_time:.2f} ms")
            print()))))))))))))))f"  • Without compute shaders: {}}}}}}}}without_time:.2f} ms")
            print()))))))))))))))f"  • Improvement: {}}}}}}}}improvement:.2f}%\n")
        
        return 0
    else:
        # Test specific model
        if args.with_compute_only:
            # Only test with compute shaders
            result = test_video_model()))))))))))))))
            model_name=args.model,
            compute_shaders=True,
            iterations=iterations,
            frame_count=args.frame_count
            )
            
            if result.get()))))))))))))))"success", False):
                performance = result.get()))))))))))))))"performance", {}}}}}}}}})
                avg_time = performance.get()))))))))))))))"avg_processing_time_ms", 0)
                
                print()))))))))))))))f"\nWebGPU Compute Shader Test for {}}}}}}}}args.model.upper())))))))))))))))}")
                print()))))))))))))))"==============================================\n")
                print()))))))))))))))f"Frame count: {}}}}}}}}args.frame_count}")
                print()))))))))))))))f"Average processing time: {}}}}}}}}avg_time:.2f} ms")
                print()))))))))))))))f"Min processing time: {}}}}}}}}performance.get()))))))))))))))'min_processing_time_ms', 0):.2f} ms")
                print()))))))))))))))f"Max processing time: {}}}}}}}}performance.get()))))))))))))))'max_processing_time_ms', 0):.2f} ms")
                print()))))))))))))))f"Standard deviation: {}}}}}}}}performance.get()))))))))))))))'std_dev_ms', 0):.2f} ms")
                
                # Print compute shader configuration
                compute_config = result.get()))))))))))))))"compute_shader_config", {}}}}}}}}})
                if compute_config:
                    print()))))))))))))))"\nCompute Shader Configuration:")
                    for key, value in compute_config.items()))))))))))))))):
                        if isinstance()))))))))))))))value, dict):
                            print()))))))))))))))f"  • {}}}}}}}}key}:")
                            for subkey, subvalue in value.items()))))))))))))))):
                                print()))))))))))))))f"    - {}}}}}}}}subkey}: {}}}}}}}}subvalue}")
                        else:
                            print()))))))))))))))f"  • {}}}}}}}}key}: {}}}}}}}}value}")
            else:
                print()))))))))))))))f"Error: {}}}}}}}}result.get()))))))))))))))'error', 'Unknown error')}")
                            return 1
        elif args.without_compute_only:
            # Only test without compute shaders
            result = test_video_model()))))))))))))))
            model_name=args.model,
            compute_shaders=False,
            iterations=iterations,
            frame_count=args.frame_count
            )
            
            if result.get()))))))))))))))"success", False):
                performance = result.get()))))))))))))))"performance", {}}}}}}}}})
                avg_time = performance.get()))))))))))))))"avg_processing_time_ms", 0)
                
                print()))))))))))))))f"\nWebGPU Standard Test for {}}}}}}}}args.model.upper())))))))))))))))}")
                print()))))))))))))))"========================================\n")
                print()))))))))))))))f"Frame count: {}}}}}}}}args.frame_count}")
                print()))))))))))))))f"Average processing time: {}}}}}}}}avg_time:.2f} ms")
                print()))))))))))))))f"Min processing time: {}}}}}}}}performance.get()))))))))))))))'min_processing_time_ms', 0):.2f} ms")
                print()))))))))))))))f"Max processing time: {}}}}}}}}performance.get()))))))))))))))'max_processing_time_ms', 0):.2f} ms")
                print()))))))))))))))f"Standard deviation: {}}}}}}}}performance.get()))))))))))))))'std_dev_ms', 0):.2f} ms")
            else:
                print()))))))))))))))f"Error: {}}}}}}}}result.get()))))))))))))))'error', 'Unknown error')}")
                return 1
        else:
            # Run comparison test
            comparison = compare_with_without_compute_shaders()))))))))))))))
            model_name=args.model,
            iterations=iterations,
            frame_count=args.frame_count
            )
            
            # Save results if requested::::::
            if args.output_json:
                with open()))))))))))))))args.output_json, 'w') as f:
                    json.dump()))))))))))))))comparison, f, indent=2)
                    logger.info()))))))))))))))f"Results saved to {}}}}}}}}args.output_json}")
            
            # Create chart if requested::::::
            if args.create_chart:
                chart_file = f"webgpu_{}}}}}}}}args.model}_compute_shader_comparison_{}}}}}}}}int()))))))))))))))time.time()))))))))))))))))}.png"
                create_performance_chart())))))))))))))){}}}}}}}}args.model: comparison}, chart_file)
            
            # Print comparison
                improvement = comparison.get()))))))))))))))"improvement_percentage", 0)
                with_result = comparison.get()))))))))))))))"with_compute_shaders", {}}}}}}}}})
                without_result = comparison.get()))))))))))))))"without_compute_shaders", {}}}}}}}}})
            
                with_time = with_result.get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
                without_time = without_result.get()))))))))))))))"performance", {}}}}}}}}}).get()))))))))))))))"avg_processing_time_ms", 0)
            
                print()))))))))))))))f"\nWebGPU Compute Shader Comparison for {}}}}}}}}args.model.upper())))))))))))))))}")
                print()))))))))))))))"===================================================\n")
                print()))))))))))))))f"Frame count: {}}}}}}}}args.frame_count}")
                print()))))))))))))))f"With compute shaders: {}}}}}}}}with_time:.2f} ms")
                print()))))))))))))))f"Without compute shaders: {}}}}}}}}without_time:.2f} ms")
                print()))))))))))))))f"Improvement: {}}}}}}}}improvement:.2f}%\n")
            
            # Print detailed metrics
                with_metrics = with_result.get()))))))))))))))"performance", {}}}}}}}}})
                print()))))))))))))))"Detailed Metrics with Compute Shaders:")
                print()))))))))))))))f"  • Frame processing time: {}}}}}}}}with_metrics.get()))))))))))))))'frame_processing_time_ms', 0):.2f} ms")
                print()))))))))))))))f"  • Temporal fusion time: {}}}}}}}}with_metrics.get()))))))))))))))'temporal_fusion_time_ms', 0):.2f} ms")
                print()))))))))))))))f"  • Memory reduction: {}}}}}}}}with_metrics.get()))))))))))))))'memory_reduction_percent', 0):.2f}%")
                print()))))))))))))))f"  • Estimated speedup: {}}}}}}}}with_metrics.get()))))))))))))))'estimated_speedup', 1.0):.2f}x\n")
            
            # Print compute shader configuration
                compute_config = with_result.get()))))))))))))))"compute_shader_config", {}}}}}}}}})
            if compute_config:
                print()))))))))))))))"Compute Shader Configuration:")
                for key, value in compute_config.items()))))))))))))))):
                    if isinstance()))))))))))))))value, dict):
                        print()))))))))))))))f"  • {}}}}}}}}key}:")
                        for subkey, subvalue in value.items()))))))))))))))):
                            print()))))))))))))))f"    - {}}}}}}}}subkey}: {}}}}}}}}subvalue}")
                    else:
                        print()))))))))))))))f"  • {}}}}}}}}key}: {}}}}}}}}value}")
        
                            return 0

if __name__ == "__main__":
    sys.exit()))))))))))))))main()))))))))))))))))