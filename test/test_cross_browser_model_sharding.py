#!/usr/bin/env python3
"""
Test Cross-Browser Model Sharding for WebNN/WebGPU Resource Pool

This script tests cross-browser model sharding using the ModelShardingManager,
which enables large models to be distributed across multiple browser instances.

Usage:
    python test_cross_browser_model_sharding.py --model bert-base-uncased --shards 3 --type layer
    python test_cross_browser_model_sharding.py --model whisper-tiny --shards 3 --type layer --model-type audio
    python test_cross_browser_model_sharding.py --model clip-vit-base-patch32 --shards 4 --type component --model-type multimodal
    """

from ipfs_accelerate_py.anyio_helpers import gather, wait_for
import anyio
    import os
    import sys
    import json
    import time
    import argparse
    import anyio
    import logging
    from pathlib import Path

# Configure logging
    logging.basicConfig())))
    level=logging.INFO,
    format='%())))asctime)s - %())))levelname)s - %())))message)s'
    )
    logger = logging.getLogger())))__name__)

# Add parent directory to path
    sys.path.append())))str())))Path())))__file__).resolve())))).parent))

# Import required modules
try:
    from test.web_platform.cross_browser_model_sharding import ModelShardingManager
    SHARDING_AVAILABLE = True
except ImportError as e:
    logger.error())))f"CrossBrowserModelSharding not available: {}}}}e}")
    SHARDING_AVAILABLE = False

# Import resource pool bridge extensions
try:
    from resource_pool_bridge_extensions import extend_resource_pool_bridge
    EXTENSIONS_AVAILABLE = True
except ImportError as e:
    logger.error())))f"ResourcePoolBridgeExtensions not available: {}}}}e}")
    EXTENSIONS_AVAILABLE = False

def get_model_input())))model_type):
    """Get appropriate test input based on model type"""
    if model_type == "text" or model_type == "text_embedding":
    return {}}
    'input_ids': [101, 2023, 2003, 1037, 3231, 102],
    'attention_mask': [1, 1, 1, 1, 1, 1],
    }
    elif model_type == "vision":
    return {}}'pixel_values': [[[0.5 for _ in range())))3)] for _ in range())))224)] for _ in range())))1)]:,}:,
    elif model_type == "audio":
    return {}}'input_features': [[[0.1 for _ in range())))80)] for _ in range())))3000)]]}:,
    elif model_type == "multimodal":
    return {}}
    'input_ids': [101, 2023, 2003, 1037, 3231, 102],
    'attention_mask': [1, 1, 1, 1, 1, 1],,
    'pixel_values': [[[0.5 for _ in range())))3)] for _ in range())))224)] for _ in range())))1)]:,
    }
    else:
    return {}}'inputs': [0.0 for _ in range())))10)]}:,
async def test_model_sharding())))args):
    """Test cross-browser model sharding"""
    if not SHARDING_AVAILABLE:
        logger.error())))"Cannot test model sharding: Cross-browser model sharding not available")
    return 1
    
    # Apply extensions if available:
    if EXTENSIONS_AVAILABLE:
        extend_resource_pool_bridge()))))
        logger.info())))"Applied resource pool bridge extensions")
    
    # Set environment variables for optimizations
    if args.compute_shaders:
        os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1",
        logger.info())))"Enabled compute shader optimization")
    
    if args.shader_precompile:
        os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1",
        logger.info())))"Enabled shader precompilation")
    
    if args.parallel_loading:
        os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1",
        logger.info())))"Enabled parallel model loading")
    
    # Create model sharding manager
        manager = ModelShardingManager())))
        model_name=args.model,
        num_shards=args.shards,
        shard_type=args.type,
        model_type=args.model_type,
        enable_ipfs=not args.disable_ipfs,
        max_connections=args.max_connections,
        db_path=args.db_path
        )
    
    try:
        # Initialize sharding
        logger.info())))f"Initializing sharding for {}}}}args.model} with {}}}}args.shards} shards")
        logger.info())))f"Using shard type: {}}}}args.type}, model type: {}}}}args.model_type}")
        
        # Set a timeout for initialization
        try:
            # Use anyio.wait_for to add timeout protection
            initialized = await wait_for())))
            manager.initialize_sharding())))),
            timeout=args.timeout
            )
        except TimeoutError:
            logger.error())))f"Initialization timeout after {}}}}args.timeout}s")
            return 1
        
        if not initialized:
            logger.error())))"Failed to initialize model sharding")
            return 1
        
            logger.info())))"✅ Model sharding initialized successfully")
        
        # Get model input based on model type
            sample_input = get_model_input())))args.model_type)
        
        # Run shard inference with timeout protection
            logger.info())))f"Running sharded inference for {}}}}args.model}")
        try:
            # Use anyio.wait_for to add timeout protection
            start_time = time.time()))))
            result = await wait_for())))
            manager.run_inference_sharded())))sample_input),
            timeout=args.timeout
            )
            execution_time = time.time())))) - start_time
        except TimeoutError:
            logger.error())))f"Inference timeout after {}}}}args.timeout}s")
            return 1
        
        # Print result summary
        if 'error' in result:
            logger.error())))f"❌ Inference error: {}}}}result['error']}"),
            return 1
        else:
            logger.info())))f"✅ Inference successful in {}}}}execution_time:.2f}s")
            if 'metrics' in result:
                logger.info())))f"Inference metrics:")
                logger.info())))f"  - Inference time: {}}}}result['metrics']['inference_time']:.2f}s"),
                logger.info())))f"  - Memory usage: {}}}}result['metrics']['memory_usage']:.2f} MB"),
                logger.info())))f"  - Average inference time: {}}}}result['metrics']['average_inference_time']:.2f}s"),
                logger.info())))f"  - Number of shards: {}}}}result['metrics']['num_shards']}"),
                ,
        # Get and print detailed metrics
                metrics = manager.get_metrics()))))
        
        if args.verbose:
            # Print full metrics
            logger.info())))f"Detailed metrics: {}}}}json.dumps())))metrics, indent=2)}")
        else:
            # Print summary metrics
            logger.info())))"Sharding Summary:")
            logger.info())))f"  - Model: {}}}}metrics['model_name']}"),
            logger.info())))f"  - Model type: {}}}}metrics['model_type']}"),
            logger.info())))f"  - Shards: {}}}}metrics['num_shards']}"),
            logger.info())))f"  - Shard type: {}}}}metrics['shard_type']}"),
            logger.info())))f"  - Initialization time: {}}}}metrics['initialization_time']:.2f}s"),
            logger.info())))f"  - Inference count: {}}}}metrics['inference_count']}"),
            logger.info())))f"  - Memory usage: {}}}}metrics['memory_usage']:.2f} MB")
            ,
            # Print browser allocation
            logger.info())))"Browser Allocation:")
            for shard_idx, config in metrics['browser_allocation'].items())))):,
            logger.info())))f"  - Shard {}}}}shard_idx}: {}}}}config['browser']} ()))){}}}}config['platform']}) - {}}}}config['specialization']}")
            ,
        # Save metrics to file if specified:
        if args.output:
            with open())))args.output, 'w') as f:
                json.dump())))metrics, f, indent=2)
                logger.info())))f"Metrics saved to {}}}}args.output}")
        
            return 0
        
    except Exception as e:
        logger.error())))f"Error testing model sharding: {}}}}e}")
        import traceback
        traceback.print_exc()))))
            return 1
    finally:
        # Close manager
        await manager.close()))))
        logger.info())))"Model sharding manager closed")

def main())))):
    """Main entry point"""
    parser = argparse.ArgumentParser())))description="Test Cross-Browser Model Sharding")
    
    # Model selection options
    parser.add_argument())))"--model", type=str, default="bert-base-uncased",
    help="Model name to shard")
    parser.add_argument())))"--model-type", type=str, default="text",
    choices=["text", "vision", "audio", "multimodal", "text_embedding"],
    help="Type of model")
    
    # Sharding options
    parser.add_argument())))"--shards", type=int, default=3,
    help="Number of shards to create")
    parser.add_argument())))"--type", type=str, default="layer",
    choices=["layer", "attention_feedforward", "component"],
    help="Type of sharding to use")
    
    # Configuration options
    parser.add_argument())))"--max-connections", type=int, default=4,
    help="Maximum number of browser connections")
    parser.add_argument())))"--timeout", type=int, default=300,
    help="Timeout in seconds for initialization and inference")
    parser.add_argument())))"--db-path", type=str, default=os.environ.get())))"BENCHMARK_DB_PATH"),
    help="Path to DuckDB database for storing results")
    
    # Feature flags
    parser.add_argument())))"--compute-shaders", action="store_true",
    help="Enable compute shader optimization for audio models")
    parser.add_argument())))"--shader-precompile", action="store_true",
    help="Enable shader precompilation for faster startup")
    parser.add_argument())))"--parallel-loading", action="store_true",
    help="Enable parallel model loading for multimodal models")
    parser.add_argument())))"--disable-ipfs", action="store_true",
    help="Disable IPFS acceleration ())))enabled by default)")
    parser.add_argument())))"--all-optimizations", action="store_true",
    help="Enable all optimizations")
    
    # Output options
    parser.add_argument())))"--output", type=str,
    help="Path to output file for metrics")
    parser.add_argument())))"--verbose", action="store_true",
    help="Enable verbose output")
    
    args = parser.parse_args()))))
    
    # Handle all optimizations flag
    if args.all_optimizations:
        args.compute_shaders = True
        args.shader_precompile = True
        args.parallel_loading = True
    
    # Set browser-specific optimizations based on model type
    if args.model_type == "audio" and not args.all_optimizations:
        args.compute_shaders = True
        os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1",
        logger.info())))"Enabled Firefox compute shader optimizations for audio model")
    
    if args.model_type == "vision" and not args.all_optimizations:
        args.shader_precompile = True
        logger.info())))"Enabled shader precompilation for vision model")
    
    if args.model_type == "multimodal" and not args.all_optimizations:
        args.parallel_loading = True
        logger.info())))"Enabled parallel loading for multimodal model")
    
    try:
        return anyio.run())))test_model_sharding())))args))
    except KeyboardInterrupt:
        logger.info())))"Interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit())))main())))))