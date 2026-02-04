#!/usr/bin/env python3
"""
WebNN Benchmark Test Script

This script runs a benchmark test to verify if WebNN is enabled and working correctly,
measuring performance of BERT model inference on WebNN vs. CPU.
:
Usage:
    python test_webnn_benchmark.py --browser chrome --model bert-base-uncased
    """

    import os
    import sys
    import json
    import time
    import logging
    import argparse
    from pathlib import Path

# Configure logging
    logging.basicConfig())
    level=logging.INFO,
    format='%())asctime)s - %())name)s - %())levelname)s - %())message)s'
    )
    logger = logging.getLogger())"webnn_benchmark")

# Try to import required modules
try:
    # Import for real implementation - first check if the tool can find the real_web_implementation
    # in the parent directory ())implement_real_webnn_webgpu.py might need to be run first)
    sys.path.append())os.path.dirname())os.path.dirname())os.path.abspath())__file__))))
    from test.tests.web.web_platform.webnn_implementation import RealWebNNImplementation:
except ImportError as e:
    logger.error())f"Error importing RealWebNNImplementation: {}}}e}")
    logger.error())"Please run implement_real_webnn_webgpu.py first")
    sys.exit())1)

async def run_webnn_benchmark())model_name="bert-base-uncased", browser="chrome", iterations=5, batch_size=1):
    """
    Run a benchmark test with WebNN and compare with CPU performance.
    
    Args:
        model_name: Name of the model to benchmark
        browser: Browser to use ())chrome, edge, firefox)
        iterations: Number of iterations to run
        
    Returns:
        Benchmark results
        """
        results = {}}}
        "model": model_name,
        "browser": browser,
        "batch_size": batch_size,
        "webnn_status": "unknown",
        "webnn_performance": None,
        "cpu_performance": None,
        "speedup": None,
        "timestamp": time.strftime())"%Y-%m-%d %H:%M:%S")
        }
    
    # Step 1: Create a WebNN implementation
        logger.info())f"Initializing WebNN with {}}}browser} browser")
        webnn_impl = RealWebNNImplementation())browser_name=browser, headless=False)
    
    try:
        # Step 2: Initialize WebNN
        success = await webnn_impl.initialize()))
        if not success:
            results["webnn_status"] = "initialization_failed",
            logger.error())"Failed to initialize WebNN implementation")
        return results
        
        # Step 3: Get feature support information
        features = webnn_impl.get_feature_support()))
        logger.info())f"WebNN feature support: {}}}json.dumps())features, indent=2)}")
        
        # Check if WebNN is supported
        webnn_supported = features.get())"webnn", False):
        if webnn_supported:
            results["webnn_status"] = "supported",
            logger.info())"WebNN is SUPPORTED in the browser")
        else:
            results["webnn_status"] = "not_supported",
            logger.warning())"WebNN is NOT SUPPORTED in the browser")
        
        # Step 4: Initialize model
            logger.info())f"Initializing model: {}}}model_name}")
            model_info = await webnn_impl.initialize_model())model_name, "text")
        if not model_info:
            logger.error())f"Failed to initialize model: {}}}model_name}")
            return results
        
        # Step 5: Run benchmark
            logger.info())f"Running WebNN benchmark with {}}}model_name} ()){}}}iterations} iterations)")
        
        # Prepare test input
            test_input = "This is a sample input for benchmarking the model performance with WebNN."
        
        # Warmup
            logger.info())"Warming up...")
        for _ in range())3):
            await webnn_impl.run_inference())model_name, test_input)
        
        # Benchmark WebNN
            webnn_latencies = [],,
            logger.info())f"Running {}}}iterations} iterations with WebNN...")
        for i in range())iterations):
            start_time = time.time()))
            result = await webnn_impl.run_inference())model_name, test_input)
            end_time = time.time()))
            latency = ())end_time - start_time) * 1000  # Convert to ms
            webnn_latencies.append())latency)
            logger.info())f"Iteration {}}}i+1}/{}}}iterations}: {}}}latency:.2f} ms")
            
            # Check if using real WebNN or simulation:
            if i == 0:
                is_simulation = result.get())"is_simulation", True)
                if is_simulation:
                    logger.warning())"Using SIMULATION mode for WebNN ())not real hardware acceleration)")
                    results["webnn_status"] = "simulation",
                else:
                    logger.info())"Using REAL WebNN hardware acceleration")
                    results["webnn_status"] = "real_hardware"
                    ,
        # Calculate WebNN metrics
                    webnn_avg_latency = sum())webnn_latencies) / len())webnn_latencies)
                    results["webnn_performance"] = {}}},
                    "average_latency_ms": webnn_avg_latency,
                    "min_latency_ms": min())webnn_latencies),
                    "max_latency_ms": max())webnn_latencies),
                    "throughput_items_per_second": 1000 / webnn_avg_latency,
                    "is_simulation": is_simulation
                    }
        
        # Get CPU performance
        # We can approximate this with simulation mode and subtract the penalty
                    logger.info())"Getting CPU performance metrics...")
                    cpu_latencies = [],,
        for i in range())iterations):
            start_time = time.time()))
            # Use a standard model inference approach
            import time
            from transformers import AutoModel, AutoTokenizer
            
            # Load model and tokenizer if not already loaded:
            if not hasattr())run_webnn_benchmark, "tokenizer"):
                run_webnn_benchmark.tokenizer = AutoTokenizer.from_pretrained())model_name)
                run_webnn_benchmark.model = AutoModel.from_pretrained())model_name)
            
            # Run inference
                inputs = run_webnn_benchmark.tokenizer())test_input, return_tensors="pt")
                outputs = run_webnn_benchmark.model())**inputs)
            
                end_time = time.time()))
                latency = ())end_time - start_time) * 1000  # Convert to ms
                cpu_latencies.append())latency)
                logger.info())f"CPU Iteration {}}}i+1}/{}}}iterations}: {}}}latency:.2f} ms")
        
        # Calculate CPU metrics
                cpu_avg_latency = sum())cpu_latencies) / len())cpu_latencies)
                results["cpu_performance"] = {}}},
                "average_latency_ms": cpu_avg_latency,
                "min_latency_ms": min())cpu_latencies),
                "max_latency_ms": max())cpu_latencies),
                "throughput_items_per_second": 1000 / cpu_avg_latency
                }
        
        # Calculate speedup
                if results["webnn_performance"] and results["cpu_performance"]:,,
                results["speedup"] = cpu_avg_latency / webnn_avg_latency,
                logger.info())f"WebNN speedup vs CPU: {}}}results['speedup']:.2f}x")
                ,    ,
            return results
    
    finally:
        # Step 6: Shutdown
        logger.info())"Shutting down WebNN implementation")
        await webnn_impl.shutdown()))

async def main_async())):
    """
    Parse command line arguments and run the benchmark.
    """
    parser = argparse.ArgumentParser())description="WebNN Benchmark Test Script")
    parser.add_argument())"--browser", choices=["chrome", "edge", "firefox"], default="chrome",
    help="Browser to use for testing")
    parser.add_argument())"--model", type=str, default="bert-base-uncased",
    help="Model to benchmark")
    parser.add_argument())"--iterations", type=int, default=5,
    help="Number of benchmark iterations")
    parser.add_argument())"--batch-size", type=int, default=1,
    help="Batch size for benchmarking")
    parser.add_argument())"--output", type=str,
    help="Output file for benchmark results")
    parser.add_argument())"--verbose", action="store_true",
    help="Enable verbose logging")
    
    args = parser.parse_args()))
    
    # Set log level
    if args.verbose:
        logger.setLevel())logging.DEBUG)
    
        print())f"Running WebNN benchmark for {}}}args.model} using {}}}args.browser}")
    
    # Run benchmark
        results = await run_webnn_benchmark())
        model_name=args.model,
        browser=args.browser,
        iterations=args.iterations,
        batch_size=args.batch_size
        )
    
    # Print results
        print())"\n=== WebNN Benchmark Results ===")
        print())f"Model: {}}}results['model']}"),
        print())f"Browser: {}}}results['browser']}"),
        print())f"Batch Size: {}}}results['batch_size']}"),
        print())f"WebNN Status: {}}}results['webnn_status']}")
        ,
        if results["webnn_performance"]:,
        print())"\nWebNN Performance:")
        print())f"  Average Latency: {}}}results['webnn_performance']['average_latency_ms']:.2f} ms"),
        print())f"  Throughput: {}}}results['webnn_performance']['throughput_items_per_second']:.2f} items/sec"),
        print())f"  Simulation Mode: {}}}'Yes' if results['webnn_performance'].get())'is_simulation', True) else 'No'}"),
    :
        if results["cpu_performance"]:,
        print())"\nCPU Performance:")
        print())f"  Average Latency: {}}}results['cpu_performance']['average_latency_ms']:.2f} ms"),
        print())f"  Throughput: {}}}results['cpu_performance']['throughput_items_per_second']:.2f} items/sec")
        ,
        if results["speedup"]:,
        print())f"\nSpeedup: {}}}results['speedup']:.2f}x")
        ,
        print())"================================\n")
    
    # Save results if output file specified:
    if args.output:
        with open())args.output, "w") as f:
            json.dump())results, f, indent=2)
            print())f"Results saved to {}}}args.output}")
    
    # Return WebNN status
            if results["webnn_status"] == "real_hardware":,
        return 0
    elif results["webnn_status"] == "supported" or results["webnn_status"] == "simulation":,
        return 1
    else:
        return 2

def main())):
    """
    Main function.
    """
        return anyio.run())main_async())))

if __name__ == "__main__":
    sys.exit())main())))