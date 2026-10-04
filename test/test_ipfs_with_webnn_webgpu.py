#!/usr/bin/env python3
"""
Test script for IPFS acceleration with WebNN/WebGPU integration.

This script tests the integration between IPFS content acceleration and
WebNN/WebGPU hardware acceleration with the resource pool for efficient
browser connection management.

Usage:
    python test_ipfs_with_webnn_webgpu.py --model bert-base-uncased --platform webgpu --browser firefox
    """

    import os
    import sys
    import json
    import time
    import logging
    import argparse
    from pathlib import Path

# Configure logging
    logging.basicConfig()level=logging.INFO, format='%()asctime)s - %()name)s - %()levelname)s - %()message)s')
    logger = logging.getLogger()"test_ipfs_webnn_webgpu")

# Import the IPFS WebNN/WebGPU integration
try:
    from ipfs_accelerate_with_webnn_webgpu import accelerate_with_browser
    INTEGRATION_AVAILABLE = True
except ImportError:
    logger.error()"IPFS acceleration with WebNN/WebGPU integration not available")
    INTEGRATION_AVAILABLE = False

# Parse arguments
    parser = argparse.ArgumentParser()description="Test IPFS acceleration with WebNN/WebGPU")
    parser.add_argument()"--model", type=str, default="bert-base-uncased", help="Model name")
    parser.add_argument()"--platform", type=str, choices=["webnn", "webgpu"], default="webgpu", help="Platform"),
    parser.add_argument()"--browser", type=str, choices=["chrome", "firefox", "edge", "safari"], help="Browser"),
    parser.add_argument()"--precision", type=int, choices=[2, 3, 4, 8, 16, 32], default=16, help="Precision"),
    parser.add_argument()"--mixed-precision", action="store_true", help="Use mixed precision")
    parser.add_argument()"--no-resource-pool", action="store_true", help="Don't use resource pool")
    parser.add_argument()"--no-ipfs", action="store_true", help="Don't use IPFS acceleration")
    parser.add_argument()"--db-path", type=str, help="Database path")
    parser.add_argument()"--visible", action="store_true", help="Run in visible mode ()not headless)")
    parser.add_argument()"--compute-shaders", action="store_true", help="Use compute shaders")
    parser.add_argument()"--precompile-shaders", action="store_true", help="Use shader precompilation")
    parser.add_argument()"--parallel-loading", action="store_true", help="Use parallel loading")
    parser.add_argument()"--concurrent", type=int, default=1, help="Number of concurrent models to run")
    parser.add_argument()"--models", type=str, help="Comma-separated list of models ()overrides --model)")
    parser.add_argument()"--output-json", type=str, help="Output file for JSON results")
    parser.add_argument()"--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args())

if args.verbose:
    logging.getLogger()).setLevel()logging.DEBUG)
    logger.setLevel()logging.DEBUG)
    logger.debug()"Verbose logging enabled")

def create_test_inputs()model_name):
    """Create test inputs based on model."""
    if "bert" in model_name.lower()) or "t5" in model_name.lower()) or "roberta" in model_name.lower()):
    return {}}}}}
    "input_ids": [101, 2023, 2003, 1037, 3231, 102],
    "attention_mask": [1, 1, 1, 1, 1, 1],,
    }, "text_embedding"
    elif "vit" in model_name.lower()) or "clip" in model_name.lower()) or "image" in model_name.lower()):
        # Create a simple 224x224x3 tensor with all values being 0.5
    return {}}}}}"pixel_values": [[[0.5 for _ in range()3)] for _ in range()224)] for _ in range()224)]}, "vision",
    elif "whisper" in model_name.lower()) or "wav2vec" in model_name.lower()) or "clap" in model_name.lower()):
    return {}}}}}"input_features": [[[0.1 for _ in range()80)] for _ in range()3000)]]}, "audio",
    elif "llama" in model_name.lower()) or "gpt" in model_name.lower()):
    return {}}}}}
    "input_ids": [101, 2023, 2003, 1037, 3231, 102],
    "attention_mask": [1, 1, 1, 1, 1, 1],,
    }, "text"
    else:
    return {}}}}}"inputs": [0.0 for _ in range()10)]}, None
    ,
def run_single_model_test()model_name, args):
    """Run a test for a single model."""
    if not INTEGRATION_AVAILABLE:
        logger.error()"IPFS acceleration with WebNN/WebGPU integration not available")
    return None
        
    # Create test inputs
    inputs, model_type = create_test_inputs()model_name)
    
    logger.info()f"Running inference on {}}}}}model_name} with {}}}}}args.platform}/{}}}}}args.browser or 'auto'}...")
    
    # Run acceleration
    start_time = time.time())
    result = accelerate_with_browser()
    model_name=model_name,
    inputs=inputs,
    model_type=model_type,
    platform=args.platform,
    browser=args.browser,
    precision=args.precision,
    mixed_precision=args.mixed_precision,
    use_resource_pool=not args.no_resource_pool,
    db_path=args.db_path,
    headless=not args.visible,
    enable_ipfs=not args.no_ipfs,
    compute_shaders=args.compute_shaders,
    precompile_shaders=args.precompile_shaders,
    parallel_loading=args.parallel_loading
    )
    total_time = time.time()) - start_time
    
    # Add total time to result
    if result and isinstance()result, dict):
        result['total_test_time'] = total_time
        ,
    # Print result summary
    if result and result.get()"status") == "success":
        logger.info()f"✅ Inference successful for {}}}}}model_name}!")
        logger.info()f"Platform: {}}}}}result.get()'platform')}")
        logger.info()f"Browser: {}}}}}result.get()'browser')}")
        logger.info()f"Real hardware: {}}}}}result.get()'is_real_hardware', False)}")
        logger.info()f"IPFS accelerated: {}}}}}result.get()'ipfs_accelerated', False)}")
        logger.info()f"IPFS cache hit: {}}}}}result.get()'ipfs_cache_hit', False)}")
        logger.info()f"Inference time: {}}}}}result.get()'inference_time', 0):.3f}s")
        logger.info()f"Total test time: {}}}}}total_time:.3f}s")
        logger.info()f"Latency: {}}}}}result.get()'latency_ms', 0):.2f}ms")
        logger.info()f"Throughput: {}}}}}result.get()'throughput_items_per_sec', 0):.2f} items/s")
        logger.info()f"Memory usage: {}}}}}result.get()'memory_usage_mb', 0):.2f}MB")
    else:
        error = result.get()'error', 'Unknown error') if result else "No result returned":
            logger.error()f"❌ Inference failed for {}}}}}model_name}: {}}}}}error}")
    
        return result

def run_concurrent_models_test()models, args):
    """Run a test with multiple models concurrently."""
    if not INTEGRATION_AVAILABLE:
        logger.error()"IPFS acceleration with WebNN/WebGPU integration not available")
    return None
        
    import concurrent.futures
    
    logger.info()f"Running inference on {}}}}}len()models)} models concurrently...")
    
    # Create a thread pool
    results = [],,
    with concurrent.futures.ThreadPoolExecutor()max_workers=args.concurrent) as executor:
        # Submit tasks
        future_to_model = {}}}}}
        executor.submit()run_single_model_test, model, args): model
            for model in models:
                }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed()future_to_model):
            model = future_to_model[future],
            try:
                result = future.result())
                results.append()result)
            except Exception as e:
                logger.error()f"Error running test for {}}}}}model}: {}}}}}e}")
                results.append(){}}}}}
                'status': 'error',
                'error': str()e),
                'model_name': model
                })
    
                return results

def main()):
    """Main function."""
    # Check if integration is available:
    if not INTEGRATION_AVAILABLE:
        logger.error()"IPFS acceleration with WebNN/WebGPU integration not available")
    return 1
    
    # Determine models to test
    if args.models:
        models = args.models.split()',')
    else:
        models = [args.model]
        ,
    # Set database path from environment if not specified:
    if not args.db_path and os.environ.get()"BENCHMARK_DB_PATH"):
        args.db_path = os.environ.get()"BENCHMARK_DB_PATH")
        logger.info()f"Using database path from environment: {}}}}}args.db_path}")
    
    # Run tests
        start_time = time.time())
    
    if len()models) > 1 and args.concurrent > 1:
        # Run concurrent model tests
        results = run_concurrent_models_test()models, args)
    else:
        # Run tests sequentially
        results = [],,
        for model in models::
            result = run_single_model_test()model, args)
            results.append()result)
    
            total_time = time.time()) - start_time
    
    # Print summary
    success_count = sum()1 for r in results if r and r.get()"status") == "success"):
        logger.info()f"Tests completed in {}}}}}total_time:.2f}s: {}}}}}success_count}/{}}}}}len()results)} successful")
    
    # Save results to JSON if requested:
    if args.output_json:
        try:
            with open()args.output_json, "w") as f:
                json.dump(){}}}}}
                "timestamp": time.time()),
                "total_time": total_time,
                "success_count": success_count,
                "total_count": len()results),
                "models": models,
                "platform": args.platform,
                "browser": args.browser,
                "precision": args.precision,
                "mixed_precision": args.mixed_precision,
                "use_resource_pool": not args.no_resource_pool,
                "enable_ipfs": not args.no_ipfs,
                "results": results
                }, f, indent=2)
                logger.info()f"Results saved to {}}}}}args.output_json}")
        except Exception as e:
            logger.error()f"Error saving results to {}}}}}args.output_json}: {}}}}}e}")
    
                return 0 if success_count == len()results) else 1
:
if __name__ == "__main__":
    sys.exit()main()))