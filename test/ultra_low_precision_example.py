#!/usr/bin/env python3
"""
Ultra-Low Precision Example Script

This example demonstrates the ultra-low precision (2-bit and 3-bit) quantization features
for WebGPU-accelerated models introduced in the fixed_web_platform module.

Key features demonstrated:
- 2-bit and 3-bit quantization configuration
- Memory reduction calculations
- KV cache optimization for extended contexts
- Mixed precision across different model components
- Browser-specific optimizations

Usage:
    python ultra_low_precision_example.py --model llama --bits 2
    python ultra_low_precision_example.py --model bert --bits 3 --mixed-precision
    python ultra_low_precision_example.py --model llama --bits 2 --extended-context
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ultra_low_precision_example")

# Try to import the ultra-low precision module
try:
    from fixed_web_platform.webgpu_ultra_low_precision import (
        setup_ultra_low_precision,
        create_2bit_compute_shaders,
        create_3bit_compute_shaders,
        quantize_model_mixed_precision,
        MixedPrecisionConfig,
        analyze_accuracy_performance_tradeoff,
        optimize_kv_cache,
        extend_context_window
    )
    ULTRA_LOW_PRECISION_AVAILABLE = True
except ImportError:
    logger.warning("fixed_web_platform.webgpu_ultra_low_precision module not available")
    ULTRA_LOW_PRECISION_AVAILABLE = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ultra-Low Precision Example")
    
    parser.add_argument("--model", type=str, default="llama",
                       help="Model to use (llama, t5, bert, clip, whisper)")
    
    parser.add_argument("--bits", type=int, default=2, choices=[2, 3],
                       help="Bit width for ultra-low precision (2 or 3)")
    
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Use mixed precision across model components")
    
    parser.add_argument("--extended-context", action="store_true",
                       help="Test context extension capabilities")
    
    parser.add_argument("--browser", type=str, default="chrome",
                       choices=["chrome", "firefox", "edge", "safari"],
                       help="Target browser for WebGPU")
    
    parser.add_argument("--memory-constraint", type=int, default=None,
                       help="Test with memory constraint (MB)")
    
    parser.add_argument("--output-json", type=str, default=None,
                       help="Output file for results (JSON)")
    
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    return parser.parse_args()

def example_ultra_low_precision_setup(model_name, bits, browser, mixed_precision=False, extended_context=False):
    """Demonstrate ultra-low precision setup"""
    if not ULTRA_LOW_PRECISION_AVAILABLE:
        logger.error("Ultra-low precision module not available")
        return None
    
    # Determine model type based on model name
    model_type = "text"
    if model_name.lower() in ["clip", "vit"]:
        model_type = "vision"
    elif model_name.lower() in ["whisper", "wav2vec2"]:
        model_type = "audio"
    
    logger.info(f"Setting up ultra-low precision for {model_name}")
    logger.info(f"Configuration: {bits}-bit precision, {browser} browser")
    logger.info(f"Mixed precision: {mixed_precision}, Extended context: {extended_context}")
    
    # Set up ultra-low precision
    start_time = time.time()
    result = setup_ultra_low_precision(
        model_name=model_name,
        model_type=model_type,
        precision_bits=bits,
        mixed_precision=mixed_precision,
        enable_kv_cache=True,
        extended_context=extended_context,
        browser=browser
    )
    elapsed = time.time() - start_time
    
    if not result["success"]:
        logger.error(f"Error setting up ultra-low precision: {result.get('error', 'Unknown error')}")
        return None
    
    # Extract results
    config = result["ultra_low_precision"]
    
    logger.info(f"Setup completed in {elapsed:.3f} seconds")
    logger.info(f"Memory reduction: {config['memory_reduction_percent']:.2f}%")
    
    if extended_context:
        logger.info(f"Context extension: {config['context_extension_factor']:.2f}x")
        logger.info(f"Extended context: {config['context_extension_factor'] * 4096:.0f} tokens (from 4096)")
    
    logger.info(f"Accuracy impact: {config['accuracy_impact_percent']:.2f}%")
    
    if mixed_precision:
        # Show layer-specific bit assignments
        logger.info("Mixed precision configuration:")
        for layer, bits in result["config"]["layer_config"].items():
            logger.info(f"  {layer}: {bits}-bit")
    
    # Show memory savings
    memory_savings = result["ultra_low_precision"]["memory_savings"]
    logger.info(f"Original model size: {memory_savings['original_size_mb']:.1f} MB")
    logger.info(f"New model size: {memory_savings['new_size_mb']:.1f} MB")
    logger.info(f"Memory saved: {memory_savings['saved_mb']:.1f} MB ({memory_savings['reduction_percent']:.1f}%)")
    
    return result

def example_mixed_precision_config(model_type, default_bits, memory_mb=None):
    """Demonstrate mixed precision configuration"""
    if not ULTRA_LOW_PRECISION_AVAILABLE:
        logger.error("Ultra-low precision module not available")
        return None
    
    logger.info(f"Creating mixed precision configuration for {model_type} models")
    logger.info(f"Default precision: {default_bits}-bit")
    
    # Create configuration
    config = MixedPrecisionConfig(model_type=model_type, default_bits=default_bits)
    
    # Display layer configuration
    logger.info("Layer-specific precision configuration:")
    for layer, bits in config.precision_map.items():
        logger.info(f"  {layer}: {bits}-bit")
    
    # Get memory reduction statistics
    memory_stats = config.get_memory_reduction()
    logger.info(f"Memory reduction: {memory_stats['memory_reduction_percent']:.2f}%")
    logger.info(f"Average bits per parameter: {memory_stats['average_bits']:.2f}")
    logger.info(f"Precision distribution: {memory_stats['precision_distribution']}")
    
    # Apply memory constraint if specified
    if memory_mb is not None:
        logger.info(f"Optimizing for memory constraint: {memory_mb} MB")
        optimized_map = config.optimize_memory_usage(memory_mb)
        config.precision_map = optimized_map
        
        # Get updated statistics
        new_stats = config.get_memory_reduction()
        logger.info(f"Memory-constrained configuration:")
        logger.info(f"Memory reduction: {new_stats['memory_reduction_percent']:.2f}%")
        logger.info(f"Average bits: {new_stats['average_bits']:.2f}")
        
        # Show updated layer configuration
        logger.info("Updated layer-specific precision configuration:")
        for layer, bits in config.precision_map.items():
            logger.info(f"  {layer}: {bits}-bit")
    
    return config

def example_context_extension(model_name, bits, browser):
    """Demonstrate context window extension"""
    if not ULTRA_LOW_PRECISION_AVAILABLE:
        logger.error("Ultra-low precision module not available")
        return None
    
    logger.info(f"Demonstrating context window extension for {model_name}")
    
    # Parameters
    original_length = 4096
    target_length = 32768
    
    logger.info(f"Original context: {original_length}, Target: {target_length}")
    logger.info(f"Configuration: {bits}-bit precision, {browser} browser")
    
    # Extend context window
    result = extend_context_window(
        model_name=model_name,
        original_length=original_length,
        target_length=target_length,
        browser=browser
    )
    
    # Display results
    logger.info(f"Original context length: {result['original_context_length']}")
    logger.info(f"Target context length: {result['target_context_length']}")
    logger.info(f"Achieved context length: {result['achieved_context_length']}")
    logger.info(f"Extension factor: {result['extension_factor']:.2f}x")
    logger.info(f"Using precision: {result['precision_bits']}-bit")
    logger.info(f"Memory reduction: {result['memory_reduction_percent']:.2f}%")
    
    if result["target_achieved"]:
        logger.info(f"✅ Target context length achieved")
    else:
        logger.warning(f"⚠️ Target context length not achieved")
    
    return result

def example_shaders(bits):
    """Demonstrate compute shader generation"""
    if not ULTRA_LOW_PRECISION_AVAILABLE:
        logger.error("Ultra-low precision module not available")
        return None
    
    logger.info(f"Generating {bits}-bit compute shaders")
    
    # Generate shaders
    if bits == 2:
        shaders = create_2bit_compute_shaders()
    elif bits == 3:
        shaders = create_3bit_compute_shaders()
    else:
        logger.error(f"Unsupported bit width: {bits}")
        return None
    
    # Display shader information
    logger.info(f"Generated {len(shaders)} shader variants:")
    for shader_type, shader_info in shaders.items():
        logger.info(f"  {shader_type}: {len(shader_info['shader_code'])} bytes")
        if 'configuration' in shader_info:
            logger.info(f"  Configuration: {shader_info['configuration']}")
    
    return shaders

def main():
    """Main function"""
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not ULTRA_LOW_PRECISION_AVAILABLE:
        logger.error("Ultra-low precision module not available. Cannot run example.")
        logger.error("Please make sure the fixed_web_platform.webgpu_ultra_low_precision module is installed.")
        return 1
    
    logger.info("Starting Ultra-Low Precision Examples")
    logger.info(f"Model: {args.model}, Bits: {args.bits}, Browser: {args.browser}")
    
    results = {
        "model": args.model,
        "bits": args.bits,
        "browser": args.browser,
        "mixed_precision": args.mixed_precision,
        "extended_context": args.extended_context,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "examples": {}
    }
    
    try:
        # Example 1: Ultra-Low Precision Setup
        logger.info("\n=== Example 1: Ultra-Low Precision Setup ===")
        setup_result = example_ultra_low_precision_setup(
            model_name=args.model,
            bits=args.bits,
            browser=args.browser,
            mixed_precision=args.mixed_precision,
            extended_context=args.extended_context
        )
        
        if setup_result:
            results["examples"]["setup"] = {
                "success": setup_result["success"],
                "memory_reduction": setup_result["ultra_low_precision"]["memory_reduction_percent"],
                "accuracy_impact": setup_result["ultra_low_precision"]["accuracy_impact_percent"]
            }
            
            if args.extended_context:
                results["examples"]["setup"]["context_extension"] = setup_result["ultra_low_precision"]["context_extension_factor"]
        
        # Example 2: Mixed Precision Configuration
        if args.mixed_precision:
            logger.info("\n=== Example 2: Mixed Precision Configuration ===")
            mp_config = example_mixed_precision_config(
                model_type="text",
                default_bits=args.bits,
                memory_mb=args.memory_constraint
            )
            
            if mp_config:
                results["examples"]["mixed_precision"] = mp_config.to_dict()
        
        # Example 3: Context Window Extension
        if args.extended_context:
            logger.info("\n=== Example 3: Context Window Extension ===")
            context_result = example_context_extension(
                model_name=args.model,
                bits=args.bits,
                browser=args.browser
            )
            
            if context_result:
                results["examples"]["context_extension"] = {
                    "original_length": context_result["original_context_length"],
                    "target_length": context_result["target_context_length"],
                    "achieved_length": context_result["achieved_context_length"],
                    "extension_factor": context_result["extension_factor"],
                    "precision_bits": context_result["precision_bits"],
                    "target_achieved": context_result["target_achieved"]
                }
        
        # Example 4: Compute Shader Generation
        logger.info("\n=== Example 4: Compute Shader Generation ===")
        shader_result = example_shaders(args.bits)
        
        if shader_result:
            results["examples"]["shaders"] = {
                "count": len(shader_result),
                "types": list(shader_result.keys())
            }
        
        # Save results to JSON if output specified
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_json}")
        
        logger.info("\nAll examples completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())