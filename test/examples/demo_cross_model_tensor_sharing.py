#!/usr/bin/env python3
"""
Demo of Cross-Model Tensor Sharing with the WebGPU/WebNN Resource Pool.

This demo showcases how tensor sharing improves performance and memory efficiency
when running multiple models that can share intermediate representations.
"""

import os
import sys
import json
import time
import anyio
import argparse
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from test directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from test.tests.web.web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
from test.tests.web.web_platform.cross_model_tensor_sharing import TensorSharingManager

async def run_tensor_sharing_demo(
    enable_tensor_sharing: bool = True,
    model_sequence: List[str] = None,
    max_memory_mb: int = 2048,
    hardware_type: str = "webgpu"
):
    """
    Run a demonstration of tensor sharing across multiple models.
    
    Args:
        enable_tensor_sharing: Whether to enable tensor sharing
        model_sequence: Sequence of models to run
        max_memory_mb: Maximum memory for tensor sharing
        hardware_type: Hardware type to use
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 80}")
    print(f"Cross-Model Tensor Sharing Demo {'(ENABLED)' if enable_tensor_sharing else '(DISABLED)'}")
    print(f"{'=' * 80}\n")
    
    # Use default model sequence if none provided
    if not model_sequence:
        model_sequence = [
            # Text models that can share embeddings
            ("bert-base-uncased", "text_embedding"),
            ("t5-small", "text_embedding"),
            
            # Vision models that can share embeddings
            ("vit-base", "vision"),
            ("clip-vit", "vision"),
            
            # Repeated models (to demonstrate caching and sharing)
            ("bert-base-uncased", "text_embedding"),  # Should reuse from first run
            ("vit-base", "vision"),                   # Should reuse from earlier run
        ]
    
    # Create resource pool
    pool = ResourcePoolBridgeIntegration(
        max_connections=2,
        browser_preferences={
            "text": "edge",
            "vision": "chrome",
            "audio": "firefox"
        }
    )
    
    # Set up tensor sharing if enabled
    if enable_tensor_sharing:
        tensor_manager = pool.setup_tensor_sharing(max_memory_mb=max_memory_mb)
        print(f"Tensor sharing enabled with max memory: {max_memory_mb} MB")
    else:
        print("Tensor sharing disabled for comparison")
    
    # Initialize pool
    await pool.initialize()
    
    # Prepare hardware preferences
    hardware_preferences = {
        "priority_list": [hardware_type, "cpu"],
        "precision": 4,                # Use 4-bit precision
        "mixed_precision": True,       # Enable mixed precision
        "compute_shaders": True,       # Enable compute shaders for audio
        "precompile_shaders": True,    # Enable shader precompilation
        "parallel_loading": True       # Enable parallel loading
    }
    
    # Run models in sequence
    results = []
    total_start_time = time.time()
    total_memory_usage = 0
    
    for i, (model_name, model_type) in enumerate(model_sequence):
        print(f"\nRunning model {i+1}/{len(model_sequence)}: {model_name} ({model_type})")
        
        # Get model
        model = await pool.get_model(
            model_type=model_type,
            model_name=model_name,
            hardware_preferences=hardware_preferences
        )
        
        # Run inference
        start_time = time.time()
        result = model(f"Sample input for {model_name}")
        end_time = time.time()
        
        # Track memory
        memory_usage = result.get("memory_usage_mb", 0)
        total_memory_usage += memory_usage
        
        # Print result summary
        print(f"  Inference time: {result.get('inference_time', 0) * 1000:.2f} ms")
        print(f"  Memory usage: {memory_usage:.2f} MB")
        
        if "shared_tensors_used" in result:
            shared_tensors = result.get("shared_tensors_used", [])
            speedup = result.get("shared_tensor_speedup", 0)
            print(f"  Using shared tensors: {', '.join(shared_tensors)}")
            print(f"  Speedup from sharing: {speedup:.2f}%")
        
        # For models that produce shareable tensors
        if enable_tensor_sharing and "output_tensors" in result:
            for tensor_name, tensor_data in result["output_tensors"].items():
                full_tensor_name = f"{model_name}_{tensor_name}"
                # Register tensor to be shared with future models
                sharing_result = await pool.share_tensor_between_models(
                    tensor_data=tensor_data,
                    tensor_name=full_tensor_name,
                    producer_model=model_name,
                    consumer_models=[],  # Will be filled by future models
                    storage_type=hardware_type
                )
                print(f"  Registered shared tensor: {full_tensor_name}")
        
        # Store result
        results.append({
            "model_name": model_name,
            "model_type": model_type,
            "inference_time_ms": result.get("inference_time", 0) * 1000,
            "memory_usage_mb": memory_usage,
            "shared_tensors_used": result.get("shared_tensors_used", []),
            "tensor_speedup_percent": result.get("shared_tensor_speedup", 0)
        })
    
    # Calculate totals
    total_time = time.time() - total_start_time
    total_inference_time = sum(r["inference_time_ms"] for r in results)
    
    print(f"\n{'-' * 80}")
    print(f"Demo completed in {total_time:.2f} seconds")
    print(f"Total inference time: {total_inference_time:.2f} ms")
    print(f"Total memory usage: {total_memory_usage:.2f} MB")
    print(f"{'-' * 80}")
    
    # Get memory optimization results if tensor sharing enabled
    if enable_tensor_sharing and hasattr(pool, 'tensor_sharing_manager'):
        # Analyze sharing opportunities
        opportunities = pool.tensor_sharing_manager.analyze_sharing_opportunities()
        print("\nTensor Sharing Opportunities:")
        for tensor_type, models in opportunities.items():
            print(f"  {tensor_type}: {', '.join(models)}")
        
        # Get memory usage by model
        model_memory = pool.tensor_sharing_manager.get_model_memory_usage()
        print("\nMemory Usage by Model:")
        for model_name, memory_info in model_memory.items():
            print(f"  {model_name}: {memory_info['total_memory_mb']:.2f} MB ({memory_info['tensor_count']} tensors)")
        
        # Run memory optimization
        optimization = pool.tensor_sharing_manager.optimize_memory_usage()
        print(f"\nMemory Optimization:")
        print(f"  Freed {optimization['freed_tensors_count']} tensors")
        print(f"  Memory reduction: {optimization['memory_reduction_percent']:.2f}%")
        print(f"  Freed memory: {optimization['freed_memory_bytes'] / (1024 * 1024):.2f} MB")
    
    # Close resources
    await pool.close()
    
    # Return summary
    return {
        "tensor_sharing_enabled": enable_tensor_sharing,
        "model_count": len(model_sequence),
        "total_time_seconds": total_time,
        "total_inference_time_ms": total_inference_time,
        "total_memory_usage_mb": total_memory_usage,
        "models": results,
        "hardware_type": hardware_type
    }

async def compare_with_without_sharing():
    """Run comparative benchmark with and without tensor sharing."""
    # Define a sequence with models that share the same embedding space
    model_sequence = [
        # Text models
        ("bert-base-uncased", "text_embedding"),
        ("t5-small", "text_embedding"),
        ("bert-base-uncased", "text_embedding"),  # Repeated to test cache
        
        # Vision models
        ("vit-base", "vision"),
        ("vit-base", "vision"),    # Repeated to test cache
    ]
    
    # Run without tensor sharing
    print("\nRunning benchmark WITHOUT tensor sharing...")
    no_sharing_results = await run_tensor_sharing_demo(
        enable_tensor_sharing=False,
        model_sequence=model_sequence
    )
    
    # Run with tensor sharing
    print("\nRunning benchmark WITH tensor sharing...")
    sharing_results = await run_tensor_sharing_demo(
        enable_tensor_sharing=True,
        model_sequence=model_sequence
    )
    
    # Calculate improvements
    time_improvement = (
        (no_sharing_results["total_time_seconds"] - sharing_results["total_time_seconds"]) / 
        no_sharing_results["total_time_seconds"] * 100
    )
    
    memory_improvement = (
        (no_sharing_results["total_memory_usage_mb"] - sharing_results["total_memory_usage_mb"]) / 
        no_sharing_results["total_memory_usage_mb"] * 100
    )
    
    inference_improvement = (
        (no_sharing_results["total_inference_time_ms"] - sharing_results["total_inference_time_ms"]) / 
        no_sharing_results["total_inference_time_ms"] * 100
    )
    
    # Print comparison
    print(f"\n{'=' * 80}")
    print(f"Tensor Sharing Performance Comparison")
    print(f"{'=' * 80}")
    print(f"Total execution time:")
    print(f"  Without sharing: {no_sharing_results['total_time_seconds']:.2f} seconds")
    print(f"  With sharing:    {sharing_results['total_time_seconds']:.2f} seconds")
    print(f"  Improvement:     {time_improvement:.2f}%")
    print(f"\nTotal memory usage:")
    print(f"  Without sharing: {no_sharing_results['total_memory_usage_mb']:.2f} MB")
    print(f"  With sharing:    {sharing_results['total_memory_usage_mb']:.2f} MB")
    print(f"  Improvement:     {memory_improvement:.2f}%")
    print(f"\nTotal inference time:")
    print(f"  Without sharing: {no_sharing_results['total_inference_time_ms']:.2f} ms")
    print(f"  With sharing:    {sharing_results['total_inference_time_ms']:.2f} ms")
    print(f"  Improvement:     {inference_improvement:.2f}%")
    print(f"{'=' * 80}")
    
    return {
        "without_sharing": no_sharing_results,
        "with_sharing": sharing_results,
        "improvements": {
            "time_percent": time_improvement,
            "memory_percent": memory_improvement,
            "inference_percent": inference_improvement
        }
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cross-Model Tensor Sharing Demo")
    parser.add_argument("--compare", action="store_true", help="Run comparison with and without tensor sharing")
    parser.add_argument("--no-sharing", action="store_true", help="Disable tensor sharing")
    parser.add_argument("--max-memory", type=int, default=2048, help="Maximum memory for tensor sharing (MB)")
    parser.add_argument("--hardware", type=str, default="webgpu", 
                        choices=["webgpu", "webnn", "cpu"], help="Hardware type to use")
    return parser.parse_args()

async def main():
    """Main entry point."""
    args = parse_args()
    
    if args.compare:
        # Run comparative benchmark
        await compare_with_without_sharing()
    else:
        # Run single demo
        await run_tensor_sharing_demo(
            enable_tensor_sharing=not args.no_sharing,
            max_memory_mb=args.max_memory,
            hardware_type=args.hardware
        )

if __name__ == "__main__":
    anyio.run(main())