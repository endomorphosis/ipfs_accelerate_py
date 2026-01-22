#!/usr/bin/env python3
"""
Verify WebNN/WebGPU Resource Pool Integration

This script tests the resource pool integration with WebNN and WebGPU implementations,
including the enhanced connection pooling and parallel model execution capabilities.

Usage:
    python verify_web_resource_pool.py --models bert,vit,whisper
    python verify_web_resource_pool.py --concurrent-models
    python verify_web_resource_pool.py --stress-test
    """

    import os
    import sys
    import json
    import time
    import asyncio
    import argparse
    import logging
    from pathlib import Path
    from datetime import datetime

# Configure logging
    logging.basicConfig()))
    level=logging.INFO,
    format='%()))asctime)s - %()))levelname)s - %()))message)s'
    )
    logger = logging.getLogger()))__name__)

# Add parent directory to path
    sys.path.append()))str()))Path()))__file__).resolve()))).parent))

# Import required modules
try:
    from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
    RESOURCE_POOL_AVAILABLE = True
except ImportError as e:
    logger.error()))f"ResourcePoolBridge not available: {}}}e}")
    RESOURCE_POOL_AVAILABLE = False

async def verify_concurrent_models()))integration, models, platform='webgpu'):
    """Test multiple models concurrently with IPFS acceleration"""
    if not integration:
        logger.error()))"Cannot test concurrent models: integration not initialized")
    return [],
    ,    ,
    try:
        logger.info()))f"Testing {}}}len()))models)} models concurrently on {}}}platform}")
        
        # Create models and inputs
        model_inputs = [],
        ,    ,
        for model_info in models:
            model_type, model_name = model_info
            
            # Configure hardware preferences with browser-specific optimizations
            hardware_preferences = {}}}
            'priority_list': [platform, 'cpu'],
            'model_family': model_type,
            'enable_ipfs': True,      # Enable IPFS acceleration for all models
            'precision': 16,          # Use FP16 precision
            'mixed_precision': False
            }
            
            # Apply model-specific optimizations
            if model_type == 'audio':
                # Audio models work best with Firefox and compute shader optimizations
                hardware_preferences['browser'] = 'firefox',
                hardware_preferences['use_firefox_optimizations'] = True,
            elif model_type == 'text_embedding' and platform == 'webnn':
                # Text models work best with Edge for WebNN
                hardware_preferences['browser'] = 'edge',
            elif model_type == 'vision':
                # Vision models work well with Chrome
                hardware_preferences['browser'] = 'chrome',
                hardware_preferences['precompile_shaders'] = True
                ,
            # Get model from resource pool
                model = integration.get_model()))
                model_type=model_type,
                model_name=model_name,
                hardware_preferences=hardware_preferences
                )
            
            if not model:
                logger.error()))f"Failed to get model: {}}}model_name}")
                continue
            
            # Prepare test input based on model type
            if model_type == 'text_embedding':
                test_input = {}}}
                'input_ids': [101, 2023, 2003, 1037, 3231, 102],
                'attention_mask': [1, 1, 1, 1, 1, 1],
                }
            elif model_type == 'vision':
                test_input = {}}}'pixel_values': [[[0.5 for _ in range()))3)] for _ in range()))224)] for _ in range()))1)]}:,
            elif model_type == 'audio':
                test_input = {}}}'input_features': [[[0.1 for _ in range()))80)] for _ in range()))3000)]]}:,
            else:
                test_input = {}}}'inputs': [0.0 for _ in range()))10)]}:,
            # Add to model inputs list
                model_inputs.append()))()))model.model_id, test_input))
        
        # Run concurrent execution
                start_time = time.time())))
                results = integration.execute_concurrent()))model_inputs)
                execution_time = time.time()))) - start_time
        
        # Process results
                logger.info()))f"Concurrent execution summary: {}}}len()))models)} models in {}}}execution_time:.2f}s")
                logger.info()))f"Performance: {}}}len()))models)/execution_time:.2f} models/second")
        
        # Calculate detailed metrics
                success_count = sum()))1 for r in results if r.get()))'success', False))
                ipfs_accelerated = sum()))1 for r in results if r.get()))'ipfs_accelerated', False))
                real_impl = sum()))1 for r in results if r.get()))'is_real_implementation', False))
                ipfs_cache_hits = sum()))1 for r in results if r.get()))'ipfs_cache_hit', False))
        :
            logger.info()))f"Success rate: {}}}success_count}/{}}}len()))results)} ())){}}}success_count/len()))results)*100 if results else 0:.1f}%)")
            logger.info()))f"IPFS Acceleration: {}}}ipfs_accelerated}/{}}}len()))results)}, "
                   f"Cache Hits: {}}}ipfs_cache_hits}/{}}}ipfs_accelerated if ipfs_accelerated else 1}, ":
                       f"Real Implementations: {}}}real_impl}/{}}}len()))results)}")
        
        # Get detailed stats
                       stats = integration.get_execution_stats())))
        if 'resource_metrics' in stats:
            metrics = stats['resource_metrics'],,
            logger.info()))f"Resource utilization: "
            f"Connection: {}}}metrics.get()))'connection_util', 0):.2f}, "
            f"Queue: {}}}stats.get()))'current_queue_size', 0)}")
        
            # Print browser usage if available:
            if 'browser_usage' in metrics:
                logger.info()))f"Browser usage: " + 
                ", ".join()))[f"{}}}browser}: {}}}count}" for browser, count in metrics['browser_usage'].items()))) if count > 0]))
                ,
        return results:
    except Exception as e:
        logger.error()))f"Error in concurrent model execution: {}}}e}")
        import traceback
        traceback.print_exc())))
            return [],
            ,
async def test_resource_pool()))):
    # Parse arguments
    parser = argparse.ArgumentParser()))description="Verify WebNN/WebGPU Resource Pool Integration")
    
    # Model selection options
    parser.add_argument()))"--models", type=str, default="bert-base-uncased",
    help="Comma-separated list of models to test")
    
    # Platform options
    parser.add_argument()))"--platform", type=str, choices=["webnn", "webgpu"], default="webgpu",
    help="Platform to test")
    
    # Test options
    parser.add_argument()))"--concurrent-models", action="store_true",
    help="Test multiple models concurrently")
    parser.add_argument()))"--stress-test", action="store_true",
    help="Run a stress test on the resource pool")
    
    # Configuration options
    parser.add_argument()))"--max-connections", type=int, default=4,
    help="Maximum number of browser connections")
    parser.add_argument()))"--visible", action="store_true",
    help="Run browsers in visible mode ()))not headless)")
    
    # Optimization options
    parser.add_argument()))"--compute-shaders", action="store_true",
    help="Enable compute shader optimization for audio models")
    parser.add_argument()))"--shader-precompile", action="store_true",
    help="Enable shader precompilation for faster startup")
    parser.add_argument()))"--parallel-loading", action="store_true",
    help="Enable parallel model loading for multimodal models")
    
    # IPFS acceleration options
    parser.add_argument()))"--disable-ipfs", action="store_true",
    help="Disable IPFS acceleration ()))enabled by default)")
    
    # Database options
    parser.add_argument()))"--db-path", type=str, default=os.environ.get()))"BENCHMARK_DB_PATH"),
    help="Path to DuckDB database for storing test results")
    
    # Browser-specific options
    parser.add_argument()))"--firefox", action="store_true",
    help="Use Firefox for all tests ()))best for audio models)")
    parser.add_argument()))"--chrome", action="store_true",
    help="Use Chrome for all tests ()))best for vision models)")
    parser.add_argument()))"--edge", action="store_true",
    help="Use Edge for all tests ()))best for WebNN)")
    
    # Advanced options
    parser.add_argument()))"--all-optimizations", action="store_true",
    help="Enable all optimizations ()))compute shaders, shader precompilation, parallel loading)")
    parser.add_argument()))"--mixed-precision", action="store_true",
    help="Enable mixed precision inference")
    
    args = parser.parse_args())))
    
    # Handle all optimizations flag
    if args.all_optimizations:
        args.compute_shaders = True
        args.shader_precompile = True
        args.parallel_loading = True
    
    # Set environment variables based on optimization flags
    if args.compute_shaders:
        os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1",
        logger.info()))"Enabled compute shader optimization")
    
    if args.shader_precompile:
        os.environ["WEBGPU_SHADER_PRECOMPILE_ENABLED"] = "1",
        logger.info()))"Enabled shader precompilation")
    
    if args.parallel_loading:
        os.environ["WEB_PARALLEL_LOADING_ENABLED"] = "1",
        logger.info()))"Enabled parallel model loading")
    
    # Parse models
    if "," in args.models:
        model_names = args.models.split()))",")
    else:
        model_names = [args.models]
        ,
    # Map model names to types
        model_types = [],
,    ,for model in model_names:
        if "bert" in model.lower()))) or "t5" in model.lower()))):
            model_types.append()))"text_embedding")
        elif "vit" in model.lower()))) or "clip" in model.lower()))):
            model_types.append()))"vision")
        elif "whisper" in model.lower()))) or "wav2vec" in model.lower()))) or "clap" in model.lower()))):
            model_types.append()))"audio")
        else:
            model_types.append()))"text")
    
    # Create model list
            models = list()))zip()))model_types, model_names))
    
    # Check if ResourcePoolBridge is available:
    if not RESOURCE_POOL_AVAILABLE:
        logger.error()))"ResourcePoolBridge not available, cannot run test")
            return 1
    
    try:
        # Configure browser preferences with optimization settings
        browser_preferences = {}}}
        'audio': 'firefox',  # Firefox has better compute shader performance for audio
        'vision': 'chrome',  # Chrome has good WebGPU support for vision models
        'text_embedding': 'edge'  # Edge has excellent WebNN support for text embeddings
        }
        
        # Override browser preferences if specific browser is selected:
        if args.firefox:
            browser_preferences = {}}}k: 'firefox' for k in browser_preferences}::
        elif args.chrome:
            browser_preferences = {}}}k: 'chrome' for k in browser_preferences}::
        elif args.edge:
            browser_preferences = {}}}k: 'edge' for k in browser_preferences}::
        
        # Determine IPFS acceleration setting
                enable_ipfs = not args.disable_ipfs
        
        # Create ResourcePoolBridgeIntegration instance with IPFS acceleration
                integration = ResourcePoolBridgeIntegration()))
                max_connections=args.max_connections,
                enable_gpu=True,
                enable_cpu=True,
                headless=not args.visible,
                browser_preferences=browser_preferences,
                adaptive_scaling=True,
                enable_ipfs=enable_ipfs,
                db_path=args.db_path,
                enable_heartbeat=True
                )
        
        # Initialize integration
                integration.initialize())))
        
        try:
            # Test concurrent model execution
            results = await verify_concurrent_models()))integration, models, args.platform)
            
            # Check results
            if results:
                logger.info()))f"Successfully ran {}}}len()))results)} models concurrently")
                
                # Get detailed stats
                execution_stats = integration.get_execution_stats())))
                if execution_stats:
                    print()))"\nExecution Statistics:")
                    print()))f"- Total Inferences: {}}}execution_stats.get()))'total_inferences', 0)}")
                    print()))f"- Concurrent Peak: {}}}execution_stats.get()))'concurrent_peak', 0)}")
                    
                    # Resource metrics
                    if 'resource_metrics' in execution_stats:
                        metrics = execution_stats['resource_metrics'],,
                        print()))"\nResource Metrics:")
                        print()))f"- Connection Utilization: {}}}metrics.get()))'connection_util', 0):.2f}")
                        
                        # Browser usage
                        if 'browser_usage' in metrics:
                            print()))"\nBrowser Usage:")
                            for browser, count in metrics['browser_usage'].items()))):,
                                if count > 0:
                                    print()))f"- {}}}browser}: {}}}count}")
            else:
                logger.error()))"No results returned from concurrent execution")
        finally:
            # Close integration
            await integration.close())))
        
                return 0
    except Exception as e:
        logger.error()))f"Error in test: {}}}e}")
        import traceback
        traceback.print_exc())))
                return 1

def main()))):
    try:
    return asyncio.run()))test_resource_pool()))))
    except KeyboardInterrupt:
        logger.info()))"Test interrupted by user")
    return 130

if __name__ == "__main__":
    sys.exit()))main()))))