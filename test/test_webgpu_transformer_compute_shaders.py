#!/usr/bin/env python3
"""
Test script for evaluating WebGPU compute shader optimizations for transformer models.

This script tests the enhanced WebGPU compute shader implementation
for transformer models, focusing on optimized attention mechanisms,
layer normalization, and MLP computations.

Usage:
    python test_webgpu_transformer_compute_shaders.py --model bert
    python test_webgpu_transformer_compute_shaders.py --model llama
    python test_webgpu_transformer_compute_shaders.py --test-all --benchmark
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
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("webgpu_transformer_compute_test")

# Define test models
TEST_MODELS = {
    "bert": "bert-base-uncased",
    "t5": "t5-small",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2": "gpt2",
    "qwen2": "Qwen/Qwen2-0.5B-Instruct"
}

# Model configurations
MODEL_CONFIGS = {
    "bert": {
        "hidden_size": 768,
        "num_heads": 12,
        "seq_length": 512
    },
    "t5": {
        "hidden_size": 512,
        "num_heads": 8,
        "seq_length": 512
    },
    "llama": {
        "hidden_size": 2048,
        "num_heads": 16,
        "seq_length": 1024
    },
    "gpt2": {
        "hidden_size": 768,
        "num_heads": 12,
        "seq_length": 1024
    },
    "qwen2": {
        "hidden_size": 1024,
        "num_heads": 16,
        "seq_length": 1024
    }
}

def setup_environment(compute_shaders_enabled=True, shader_precompile=True):
    """
    Set up the environment variables for WebGPU testing with compute shaders.
    
    Args:
        compute_shaders_enabled: Whether to enable compute shaders
        shader_precompile: Whether to enable shader precompilation
        
    Returns:
        True if successful, False otherwise
    """
    # Set WebGPU environment variables
    os.environ["WEBGPU_ENABLED"] = "1"
    os.environ["WEBGPU_SIMULATION"] = "1" 
    os.environ["WEBGPU_AVAILABLE"] = "1"
    
    # Enable compute shaders if requested
    if compute_shaders_enabled:
        os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
        logger.info("WebGPU compute shaders enabled")
    else:
        if "WEBGPU_COMPUTE_SHADERS_ENABLED" in os.environ:
            del os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"]
        logger.info("WebGPU compute shaders disabled")
    
    # Enable shader precompilation if requested
    if shader_precompile:
        os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1"
        logger.info("WebGPU shader precompilation enabled")
    else:
        if "WEBGPU_SHADER_PRECOMPILE_ENABLED" in os.environ:
            del os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"]
        logger.info("WebGPU shader precompilation disabled")
    
    return True

def import_webgpu_transformer_compute_shaders():
    """
    Import the WebGPU transformer compute shaders module.
    
    Returns:
        The imported module or None if failed
    """
    try:
        # Try to import from the fixed_web_platform directory
        from fixed_web_platform.webgpu_transformer_compute_shaders import (
            setup_transformer_compute_shaders, get_supported_transformer_models
        )
        logger.info("Successfully imported WebGPU transformer compute shaders module")
        return {
            "setup_transformer_compute_shaders": setup_transformer_compute_shaders,
            "get_supported_transformer_models": get_supported_transformer_models
        }
    except ImportError as e:
        logger.error(f"Failed to import WebGPU transformer compute shaders module: {str(e)}")
        return None

def test_transformer_model(model_name, compute_shaders=True, iterations=5, seq_length=None):
    """
    Test a transformer model with WebGPU implementation.
    
    Args:
        model_name: Name of the model to test
        compute_shaders: Whether to use compute shaders
        iterations: Number of inference iterations
        seq_length: Custom sequence length to test
        
    Returns:
        Dictionary with test results
    """
    # Import WebGPU transformer compute shaders
    modules = import_webgpu_transformer_compute_shaders()
    if not modules:
        return {
            "success": False,
            "error": "Failed to import WebGPU transformer compute shaders module"
        }
    
    setup_transformer_compute_shaders = modules["setup_transformer_compute_shaders"]
    
    # Set up environment
    setup_environment(compute_shaders_enabled=compute_shaders)
    
    # Select model
    if model_name in TEST_MODELS:
        model_hf_name = TEST_MODELS[model_name]
    else:
        model_hf_name = model_name
    
    # Get model configuration
    config = MODEL_CONFIGS.get(model_name, {})
    if seq_length is not None:
        config["seq_length"] = seq_length
    
    # Create WebGPU compute shaders instance
    compute_shader = setup_transformer_compute_shaders(
        model_name=model_hf_name,
        model_type=model_name,
        seq_length=config.get("seq_length", 512),
        config=config
    )
    
    # Run initial inference to warm up
    compute_shader.process_transformer_layer()
    
    # Run benchmark iterations
    processing_times = []
    attention_times = []
    layernorm_times = []
    mlp_times = []
    memory_usages = []
    
    for i in range(iterations):
        # Process transformer layer
        metrics = compute_shader.process_transformer_layer(layer_idx=i)
        
        # Extract metrics
        processing_time = metrics.get("total_compute_time_ms", 0)
        attention_time = metrics.get("attention_time_ms", 0)
        layernorm_time = metrics.get("layer_norm_time_ms", 0)
        mlp_time = metrics.get("mlp_time_ms", 0)
        memory_reduction = metrics.get("memory_reduction_percent", 0)
        
        processing_times.append(processing_time)
        attention_times.append(attention_time)
        layernorm_times.append(layernorm_time)
        mlp_times.append(mlp_time)
        memory_usages.append(memory_reduction)
    
    # Calculate performance metrics
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    min_processing_time = min(processing_times) if processing_times else 0
    max_processing_time = max(processing_times) if processing_times else 0
    std_dev = (
        (sum((t - avg_processing_time) ** 2 for t in processing_times) / len(processing_times)) ** 0.5 
        if len(processing_times) > 1 else 0
    )
    
    avg_attention_time = sum(attention_times) / len(attention_times) if attention_times else 0
    avg_layernorm_time = sum(layernorm_times) / len(layernorm_times) if layernorm_times else 0
    avg_mlp_time = sum(mlp_times) / len(mlp_times) if mlp_times else 0
    
    # Get compute shader configuration
    compute_config = metrics.get("compute_shader_config", {})
    
    # Create result
    return {
        "success": True,
        "model_name": model_name,
        "model_hf_name": model_hf_name,
        "compute_shaders_enabled": compute_shaders,
        "seq_length": config.get("seq_length", 512),
        "hidden_size": config.get("hidden_size", 768),
        "num_heads": config.get("num_heads", 12),
        "performance": {
            "iterations": iterations,
            "avg_processing_time_ms": avg_processing_time,
            "min_processing_time_ms": min_processing_time,
            "max_processing_time_ms": max_processing_time,
            "std_dev_ms": std_dev,
            "avg_attention_time_ms": avg_attention_time,
            "avg_layernorm_time_ms": avg_layernorm_time,
            "avg_mlp_time_ms": avg_mlp_time,
            "component_breakdown": {
                "attention": avg_attention_time / avg_processing_time if avg_processing_time > 0 else 0,
                "layernorm": avg_layernorm_time / avg_processing_time if avg_processing_time > 0 else 0,
                "mlp": avg_mlp_time / avg_processing_time if avg_processing_time > 0 else 0
            },
            "memory_reduction_percent": sum(memory_usages) / len(memory_usages) if memory_usages else 0,
            "estimated_speedup": metrics.get("estimated_speedup", 1.0)
        },
        "compute_shader_config": compute_config
    }

def compare_with_without_compute_shaders(model_name, iterations=5, seq_length=None):
    """
    Compare model performance with and without compute shaders.
    
    Args:
        model_name: Name of the model to test
        iterations: Number of inference iterations per configuration
        seq_length: Custom sequence length to test
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Testing {model_name} with seq_length={seq_length or MODEL_CONFIGS.get(model_name, {}).get('seq_length', 512)}")
    # Run tests with compute shaders
    with_compute_shaders = test_transformer_model(
        model_name=model_name,
        compute_shaders=True,
        iterations=iterations,
        seq_length=seq_length
    )
    
    # Run tests without compute shaders
    without_compute_shaders = test_transformer_model(
        model_name=model_name,
        compute_shaders=False,
        iterations=iterations,
        seq_length=seq_length
    )
    
    # Calculate improvement
    improvement = 0
    if (with_compute_shaders.get("success", False) and 
        without_compute_shaders.get("success", False)):
        
        with_time = with_compute_shaders.get("performance", {}).get("avg_processing_time_ms", 0)
        without_time = without_compute_shaders.get("performance", {}).get("avg_processing_time_ms", 0)
        
        if without_time > 0:
            improvement = (without_time - with_time) / without_time * 100
    
    return {
        "model_name": model_name,
        "seq_length": seq_length or MODEL_CONFIGS.get(model_name, {}).get("seq_length", 512),
        "with_compute_shaders": with_compute_shaders,
        "without_compute_shaders": without_compute_shaders,
        "improvement_percentage": improvement
    }

def run_all_model_comparisons(iterations=5, output_json=None, create_chart=False, seq_length=None):
    """
    Run comparisons for all test models.
    
    Args:
        iterations: Number of inference iterations per configuration
        output_json: Path to save JSON results
        create_chart: Whether to create a performance comparison chart
        seq_length: Custom sequence length to test
        
    Returns:
        Dictionary with all comparison results
    """
    results = {}
    models = list(TEST_MODELS.keys())
    
    for model in models:
        logger.info(f"Testing {model} with and without compute shaders...")
        comparison = compare_with_without_compute_shaders(model, iterations, seq_length)
        results[model] = comparison
        
        # Print summary
        improvement = comparison.get("improvement_percentage", 0)
        logger.info(f"  • {model}: {improvement:.2f}% improvement with compute shaders")
    
    # Save results to JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_json}")
    
    # Create chart if requested
    if create_chart:
        create_performance_chart(results, f"webgpu_transformer_compute_shader_comparison_{int(time.time())}.png")
        create_component_breakdown_chart(results, f"webgpu_transformer_component_breakdown_{int(time.time())}.png")
    
    return results

def create_performance_chart(results, output_file):
    """
    Create a performance comparison chart.
    
    Args:
        results: Dictionary with comparison results
        output_file: Path to save the chart
    """
    try:
        models = list(results.keys())
        with_compute = []
        without_compute = []
        improvements = []
        
        for model in models:
            comparison = results[model]
            with_time = comparison.get("with_compute_shaders", {}).get("performance", {}).get("avg_processing_time_ms", 0)
            without_time = comparison.get("without_compute_shaders", {}).get("performance", {}).get("avg_processing_time_ms", 0)
            improvement = comparison.get("improvement_percentage", 0)
            
            with_compute.append(with_time)
            without_compute.append(without_time)
            improvements.append(improvement)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart for processing times
        x = range(len(models))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], without_compute, width, label='Without Compute Shaders')
        ax1.bar([i + width/2 for i in x], with_compute, width, label='With Compute Shaders')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Processing Time (ms)')
        ax1.set_title('WebGPU Transformer Processing Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        
        # Add processing time values on bars
        for i, v in enumerate(without_compute):
            ax1.text(i - width/2, v + 1, f"{v:.1f}", ha='center')
        
        for i, v in enumerate(with_compute):
            ax1.text(i + width/2, v + 1, f"{v:.1f}", ha='center')
        
        # Bar chart for improvements
        ax2.bar(models, improvements, color='green')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Performance Improvement with Compute Shaders')
        
        # Add improvement values on bars
        for i, v in enumerate(improvements):
            ax2.text(i, v + 0.5, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Performance chart saved to {output_file}")
    except Exception as e:
        logger.error(f"Error creating performance chart: {e}")

def create_component_breakdown_chart(results, output_file):
    """
    Create a chart showing the breakdown of time spent in each transformer component.
    
    Args:
        results: Dictionary with comparison results
        output_file: Path to save the chart
    """
    try:
        models = list(results.keys())
        attention_times = []
        layernorm_times = []
        mlp_times = []
        
        for model in models:
            comparison = results[model]
            performance = comparison.get("with_compute_shaders", {}).get("performance", {})
            component_breakdown = performance.get("component_breakdown", {})
            
            attention_times.append(component_breakdown.get("attention", 0) * 100)
            layernorm_times.append(component_breakdown.get("layernorm", 0) * 100)
            mlp_times.append(component_breakdown.get("mlp", 0) * 100)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(models))
        
        ax.bar(models, attention_times, label='Attention Mechanism')
        ax.bar(models, layernorm_times, bottom=attention_times, label='Layer Normalization')
        
        # Calculate the sum of the first two components for the bottom of the third component
        bottom_for_mlp = [a + l for a, l in zip(attention_times, layernorm_times)]
        ax.bar(models, mlp_times, bottom=bottom_for_mlp, label='MLP Computation')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Percentage of Total Processing Time')
        ax.set_title('Transformer Component Breakdown (With Compute Shaders)')
        ax.legend()
        
        # Add percentage values on bars
        for i, (attn, norm, mlp) in enumerate(zip(attention_times, layernorm_times, mlp_times)):
            # Only add percentages that are significant enough to display
            if attn > 5:
                ax.text(i, attn/2, f"{attn:.1f}%", ha='center')
            if norm > 5:
                ax.text(i, attn + norm/2, f"{norm:.1f}%", ha='center')
            if mlp > 5:
                ax.text(i, attn + norm + mlp/2, f"{mlp:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Component breakdown chart saved to {output_file}")
    except Exception as e:
        logger.error(f"Error creating component breakdown chart: {e}")

def test_sequence_length_scaling(model_name, iterations=3, seq_lengths=[64, 128, 256, 512, 1024]):
    """
    Test how model performance scales with different sequence lengths.
    
    Args:
        model_name: Name of the model to test
        iterations: Number of inference iterations per configuration
        seq_lengths: List of sequence lengths to test
        
    Returns:
        Dictionary with scaling results
    """
    logger.info(f"Testing {model_name} scaling with different sequence lengths")
    scaling_results = {}
    
    for seq_length in seq_lengths:
        # Run tests with compute shaders
        with_compute_shaders = test_transformer_model(
            model_name=model_name,
            compute_shaders=True,
            iterations=iterations,
            seq_length=seq_length
        )
        
        # Run tests without compute shaders
        without_compute_shaders = test_transformer_model(
            model_name=model_name,
            compute_shaders=False,
            iterations=iterations,
            seq_length=seq_length
        )
        
        # Calculate improvement
        improvement = 0
        if (with_compute_shaders.get("success", False) and 
            without_compute_shaders.get("success", False)):
            
            with_time = with_compute_shaders.get("performance", {}).get("avg_processing_time_ms", 0)
            without_time = without_compute_shaders.get("performance", {}).get("avg_processing_time_ms", 0)
            
            if without_time > 0:
                improvement = (without_time - with_time) / without_time * 100
        
        scaling_results[seq_length] = {
            "with_compute_shaders": with_compute_shaders,
            "without_compute_shaders": without_compute_shaders,
            "improvement_percentage": improvement
        }
        
        logger.info(f"  • {seq_length} tokens: {improvement:.2f}% improvement with compute shaders")
    
    return {
        "model_name": model_name,
        "seq_lengths": seq_lengths,
        "scaling_results": scaling_results
    }

def create_scaling_chart(scaling_data, output_file):
    """
    Create a chart showing performance scaling with different sequence lengths.
    
    Args:
        scaling_data: Scaling test results
        output_file: Path to save the chart
    """
    try:
        model_name = scaling_data.get("model_name", "Unknown")
        seq_lengths = scaling_data.get("seq_lengths", [])
        scaling_results = scaling_data.get("scaling_results", {})
        
        with_compute_times = []
        without_compute_times = []
        improvements = []
        
        for seq_length in seq_lengths:
            result = scaling_results.get(seq_length, {})
            with_time = result.get("with_compute_shaders", {}).get("performance", {}).get("avg_processing_time_ms", 0)
            without_time = result.get("without_compute_shaders", {}).get("performance", {}).get("avg_processing_time_ms", 0)
            improvement = result.get("improvement_percentage", 0)
            
            with_compute_times.append(with_time)
            without_compute_times.append(without_time)
            improvements.append(improvement)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Line chart for processing times
        ax1.plot(seq_lengths, without_compute_times, 'o-', label='Without Compute Shaders')
        ax1.plot(seq_lengths, with_compute_times, 'o-', label='With Compute Shaders')
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Processing Time (ms)')
        ax1.set_title(f'{model_name} Processing Time vs. Sequence Length')
        ax1.legend()
        ax1.grid(True)
        
        # Line chart for improvements
        ax2.plot(seq_lengths, improvements, 'o-', color='green')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title(f'{model_name} Performance Improvement vs. Sequence Length')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Scaling chart saved to {output_file}")
    except Exception as e:
        logger.error(f"Error creating scaling chart: {e}")

def main():
    """Parse arguments and run the tests."""
    parser = argparse.ArgumentParser(
        description="Test WebGPU compute shader optimizations for transformer models"
    )
    
    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument("--model", choices=list(TEST_MODELS.keys()), default="bert",
                           help="Transformer model to test")
    model_group.add_argument("--test-all", action="store_true",
                           help="Test all available transformer models")
    
    # Test options
    test_group = parser.add_argument_group("Test Options")
    test_group.add_argument("--iterations", type=int, default=5,
                          help="Number of inference iterations for each test")
    test_group.add_argument("--benchmark", action="store_true",
                          help="Run in benchmark mode with 20 iterations")
    test_group.add_argument("--with-compute-only", action="store_true",
                          help="Only test with compute shaders enabled")
    test_group.add_argument("--without-compute-only", action="store_true",
                          help="Only test without compute shaders")
    test_group.add_argument("--seq-length", type=int,
                          help="Custom sequence length to test")
    test_group.add_argument("--test-scaling", action="store_true",
                          help="Test performance scaling with different sequence lengths")
    
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
    
    # Determine number of iterations
    iterations = args.iterations
    if args.benchmark:
        iterations = 20
    
    # If testing sequence length scaling
    if args.test_scaling:
        scaling_data = test_sequence_length_scaling(
            model_name=args.model,
            iterations=max(2, iterations // 3),  # Reduce iterations for scaling test
            seq_lengths=[64, 128, 256, 512, 1024, 2048]
        )
        
        # Save results to JSON if requested
        if args.output_json:
            output_json = args.output_json
            if not output_json.endswith(".json"):
                output_json = f"{output_json}_scaling.json"
            
            with open(output_json, 'w') as f:
                json.dump(scaling_data, f, indent=2)
            logger.info(f"Scaling results saved to {output_json}")
        
        # Create chart
        create_scaling_chart(
            scaling_data=scaling_data,
            output_file=f"webgpu_{args.model}_scaling_{int(time.time())}.png"
        )
        
        # Print summary
        print("\nWebGPU Compute Shader Scaling Results")
        print("=====================================\n")
        print(f"Model: {args.model.upper()}\n")
        
        seq_lengths = scaling_data.get("seq_lengths", [])
        scaling_results = scaling_data.get("scaling_results", {})
        
        print("Seq Length | Improvement | With Compute | Without Compute")
        print("-----------|-------------|-------------|----------------")
        
        for seq_length in seq_lengths:
            result = scaling_results.get(seq_length, {})
            improvement = result.get("improvement_percentage", 0)
            with_time = result.get("with_compute_shaders", {}).get("performance", {}).get("avg_processing_time_ms", 0)
            without_time = result.get("without_compute_shaders", {}).get("performance", {}).get("avg_processing_time_ms", 0)
            
            print(f"{seq_length:>10} | {improvement:>10.2f}% | {with_time:>11.2f}ms | {without_time:>14.2f}ms")
        
        return 0
    
    # Run tests
    if args.test_all:
        # Test all models with comparison
        results = run_all_model_comparisons(
            iterations=iterations,
            output_json=args.output_json,
            create_chart=args.create_chart,
            seq_length=args.seq_length
        )
        
        # Print comparison summary
        print("\nWebGPU Transformer Compute Shader Optimization Results")
        print("===================================================\n")
        
        for model, comparison in results.items():
            improvement = comparison.get("improvement_percentage", 0)
            with_time = comparison.get("with_compute_shaders", {}).get("performance", {}).get("avg_processing_time_ms", 0)
            without_time = comparison.get("without_compute_shaders", {}).get("performance", {}).get("avg_processing_time_ms", 0)
            
            print(f"{model.upper()} Model:")
            print(f"  • With compute shaders: {with_time:.2f} ms")
            print(f"  • Without compute shaders: {without_time:.2f} ms")
            print(f"  • Improvement: {improvement:.2f}%\n")
        
        return 0
    else:
        # Test specific model
        if args.with_compute_only:
            # Only test with compute shaders
            result = test_transformer_model(
                model_name=args.model,
                compute_shaders=True,
                iterations=iterations,
                seq_length=args.seq_length
            )
            
            if result.get("success", False):
                performance = result.get("performance", {})
                avg_time = performance.get("avg_processing_time_ms", 0)
                
                print(f"\nWebGPU Compute Shader Test for {args.model.upper()}")
                print("==============================================\n")
                print(f"Sequence length: {result.get('seq_length', 0)}")
                print(f"Hidden size: {result.get('hidden_size', 0)}")
                print(f"Number of heads: {result.get('num_heads', 0)}")
                print(f"Average processing time: {avg_time:.2f} ms")
                print(f"Min processing time: {performance.get('min_processing_time_ms', 0):.2f} ms")
                print(f"Max processing time: {performance.get('max_processing_time_ms', 0):.2f} ms")
                print(f"Standard deviation: {performance.get('std_dev_ms', 0):.2f} ms")
                
                # Print component breakdown
                print("\nComponent Breakdown:")
                print(f"  • Attention mechanism: {performance.get('avg_attention_time_ms', 0):.2f} ms")
                print(f"  • Layer normalization: {performance.get('avg_layernorm_time_ms', 0):.2f} ms")
                print(f"  • MLP computation: {performance.get('avg_mlp_time_ms', 0):.2f} ms")
                
                # Print compute shader configuration
                compute_config = result.get("compute_shader_config", {})
                if compute_config:
                    print("\nCompute Shader Configuration:")
                    
                    # Print attention mechanism config
                    attention_config = compute_config.get("attention_mechanism", {})
                    print("  • Attention mechanism:")
                    print(f"    - Algorithm: {attention_config.get('algorithm', 'unknown')}")
                    print(f"    - KV cache: {'enabled' if attention_config.get('kv_cache_enabled', False) else 'disabled'}")
                    
                    # Print layer norm config
                    layernorm_config = compute_config.get("layer_norm", {})
                    print("  • Layer normalization:")
                    print(f"    - Algorithm: {layernorm_config.get('algorithm', 'unknown')}")
                    
                    # Print MLP config
                    mlp_config = compute_config.get("mlp", {})
                    print("  • MLP computation:")
                    print(f"    - Algorithm: {mlp_config.get('algorithm', 'unknown')}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
        elif args.without_compute_only:
            # Only test without compute shaders
            result = test_transformer_model(
                model_name=args.model,
                compute_shaders=False,
                iterations=iterations,
                seq_length=args.seq_length
            )
            
            if result.get("success", False):
                performance = result.get("performance", {})
                avg_time = performance.get("avg_processing_time_ms", 0)
                
                print(f"\nWebGPU Standard Test for {args.model.upper()}")
                print("========================================\n")
                print(f"Sequence length: {result.get('seq_length', 0)}")
                print(f"Hidden size: {result.get('hidden_size', 0)}")
                print(f"Number of heads: {result.get('num_heads', 0)}")
                print(f"Average processing time: {avg_time:.2f} ms")
                print(f"Min processing time: {performance.get('min_processing_time_ms', 0):.2f} ms")
                print(f"Max processing time: {performance.get('max_processing_time_ms', 0):.2f} ms")
                print(f"Standard deviation: {performance.get('std_dev_ms', 0):.2f} ms")
                
                # Print component breakdown
                print("\nComponent Breakdown:")
                print(f"  • Attention mechanism: {performance.get('avg_attention_time_ms', 0):.2f} ms")
                print(f"  • Layer normalization: {performance.get('avg_layernorm_time_ms', 0):.2f} ms")
                print(f"  • MLP computation: {performance.get('avg_mlp_time_ms', 0):.2f} ms")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1
        else:
            # Run comparison test
            comparison = compare_with_without_compute_shaders(
                model_name=args.model,
                iterations=iterations,
                seq_length=args.seq_length
            )
            
            # Save results if requested
            if args.output_json:
                with open(args.output_json, 'w') as f:
                    json.dump(comparison, f, indent=2)
                logger.info(f"Results saved to {args.output_json}")
            
            # Create chart if requested
            if args.create_chart:
                chart_file = f"webgpu_{args.model}_compute_shader_comparison_{int(time.time())}.png"
                create_performance_chart({args.model: comparison}, chart_file)
                
                component_chart_file = f"webgpu_{args.model}_component_breakdown_{int(time.time())}.png"
                create_component_breakdown_chart({args.model: comparison}, component_chart_file)
            
            # Print comparison
            improvement = comparison.get("improvement_percentage", 0)
            with_result = comparison.get("with_compute_shaders", {})
            without_result = comparison.get("without_compute_shaders", {})
            
            with_time = with_result.get("performance", {}).get("avg_processing_time_ms", 0)
            without_time = without_result.get("performance", {}).get("avg_processing_time_ms", 0)
            
            print(f"\nWebGPU Compute Shader Comparison for {args.model.upper()}")
            print("===================================================\n")
            print(f"Sequence length: {comparison.get('seq_length', 0)}")
            print(f"With compute shaders: {with_time:.2f} ms")
            print(f"Without compute shaders: {without_time:.2f} ms")
            print(f"Improvement: {improvement:.2f}%\n")
            
            # Print detailed metrics for compute shaders
            with_metrics = with_result.get("performance", {})
            print("Detailed Metrics with Compute Shaders:")
            print(f"  • Attention mechanism: {with_metrics.get('avg_attention_time_ms', 0):.2f} ms")
            print(f"  • Layer normalization: {with_metrics.get('avg_layernorm_time_ms', 0):.2f} ms")
            print(f"  • MLP computation: {with_metrics.get('avg_mlp_time_ms', 0):.2f} ms")
            print(f"  • Memory reduction: {with_metrics.get('memory_reduction_percent', 0):.2f}%")
            print(f"  • Estimated speedup: {with_metrics.get('estimated_speedup', 1.0):.2f}x\n")
            
            # Print compute shader configuration
            compute_config = with_result.get("compute_shader_config", {})
            if compute_config:
                print("Compute Shader Configuration:")
                
                # Print attention mechanism config
                attention_config = compute_config.get("attention_mechanism", {})
                print("  • Attention mechanism:")
                print(f"    - Algorithm: {attention_config.get('algorithm', 'unknown')}")
                print(f"    - KV cache: {'enabled' if attention_config.get('kv_cache_enabled', False) else 'disabled'}")
                
                # Print layer norm config
                layernorm_config = compute_config.get("layer_norm", {})
                print("  • Layer normalization:")
                print(f"    - Algorithm: {layernorm_config.get('algorithm', 'unknown')}")
                
                # Print MLP config
                mlp_config = compute_config.get("mlp", {})
                print("  • MLP computation:")
                print(f"    - Algorithm: {mlp_config.get('algorithm', 'unknown')}")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())