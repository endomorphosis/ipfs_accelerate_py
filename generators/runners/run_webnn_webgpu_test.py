#!/usr/bin/env python3
"""
Run WebNN and WebGPU Test

This script runs a simple test of the WebNN and WebGPU implementations,
verifying that they are correctly using real browser-based hardware acceleration.

Usage:
    python run_webnn_webgpu_test.py [--browser chrome|firefox|edge] [--platform webgpu|webnn|both]
"""

import os
import sys
import json
import time
import logging
import argparse
import anyio
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our implementations
try:
    from unified_web_implementation import UnifiedWebImplementation
    implementation_type = "unified"
except ImportError:
    try:
        from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation
        from fixed_web_platform.webnn_implementation import RealWebNNImplementation
        implementation_type = "direct"
    except ImportError:
        logger.error("Could not import WebNN/WebGPU implementations")
        logger.error("Make sure you're running this script from the correct directory")
        sys.exit(1)

async def test_webgpu(browser="chrome", headless=False, model="bert-base-uncased"):
    """Test WebGPU implementation.
    
    Args:
        browser: Browser to use
        headless: Whether to run in headless mode
        model: Model to test
        
    Returns:
        Dict with test results
    """
    logger.info(f"Testing WebGPU with {browser} browser")
    
    if implementation_type == "unified":
        return await test_webgpu_unified(browser, headless, model)
    else:
        return await test_webgpu_direct(browser, headless, model)

async def test_webgpu_unified(browser, headless, model):
    """Test WebGPU with UnifiedWebImplementation.
    
    Args:
        browser: Browser to use
        headless: Whether to run in headless mode
        model: Model to test
        
    Returns:
        Dict with test results
    """
    impl = UnifiedWebImplementation()
    
    try:
        # Check if WebGPU hardware is available
        hardware_available = impl.is_hardware_available("webgpu")
        logger.info(f"WebGPU hardware acceleration available: {hardware_available}")
        
        # Initialize model
        logger.info(f"Initializing model {model} on WebGPU")
        init_result = impl.init_model(model, model_type="text", platform="webgpu")
        
        if not init_result:
            logger.error(f"Failed to initialize model {model} on WebGPU")
            impl.shutdown()
            return {
                "success": False,
                "error": "Model initialization failed"
            }
        
        # Run inference
        logger.info(f"Running inference with model {model} on WebGPU")
        input_text = "This is a test input for WebGPU inference"
        inference_result = impl.run_inference(model, input_text, platform="webgpu")
        
        if not inference_result:
            logger.error(f"Failed to run inference with model {model} on WebGPU")
            impl.shutdown()
            return {
                "success": False,
                "error": "Inference failed"
            }
        
        # Check if using simulation
        is_simulation = inference_result.get("is_simulation", True)
        implementation_type = inference_result.get("implementation_type", "UNKNOWN")
        
        # Get performance metrics
        perf_metrics = inference_result.get("performance_metrics", {})
        
        # Shutdown
        impl.shutdown()
        
        return {
            "success": True,
            "hardware_available": hardware_available,
            "is_simulation": is_simulation,
            "implementation_type": implementation_type,
            "performance_metrics": perf_metrics
        }
        
    except Exception as e:
        logger.error(f"Error testing WebGPU: {e}")
        impl.shutdown()
        return {
            "success": False,
            "error": str(e)
        }

async def test_webgpu_direct(browser, headless, model):
    """Test WebGPU with RealWebGPUImplementation.
    
    Args:
        browser: Browser to use
        headless: Whether to run in headless mode
        model: Model to test
        
    Returns:
        Dict with test results
    """
    impl = RealWebGPUImplementation(browser_name=browser, headless=headless)
    
    try:
        # Initialize implementation
        success = await impl.initialize()
        if not success:
            logger.error("Failed to initialize WebGPU implementation")
            return {
                "success": False,
                "error": "Failed to initialize WebGPU implementation"
            }
        
        # Initialize model
        logger.info(f"Initializing model {model} on WebGPU")
        init_result = await impl.initialize_model(model, model_type="text")
        
        if not init_result:
            logger.error(f"Failed to initialize model {model} on WebGPU")
            await impl.shutdown()
            return {
                "success": False,
                "error": "Model initialization failed"
            }
        
        # Run inference
        logger.info(f"Running inference with model {model} on WebGPU")
        input_text = "This is a test input for WebGPU inference"
        inference_result = await impl.run_inference(model, input_text)
        
        if not inference_result:
            logger.error(f"Failed to run inference with model {model} on WebGPU")
            await impl.shutdown()
            return {
                "success": False,
                "error": "Inference failed"
            }
        
        # Check implementation details
        impl_details = inference_result.get("_implementation_details", {})
        is_simulation = impl_details.get("is_simulation", True)
        implementation_type = impl_details.get("implementation_type", "UNKNOWN")
        
        # Get performance metrics
        perf_metrics = inference_result.get("performance_metrics", {})
        
        # Shutdown
        await impl.shutdown()
        
        return {
            "success": True,
            "is_simulation": is_simulation,
            "implementation_type": implementation_type,
            "performance_metrics": perf_metrics
        }
        
    except Exception as e:
        logger.error(f"Error testing WebGPU: {e}")
        await impl.shutdown()
        return {
            "success": False,
            "error": str(e)
        }

async def test_webnn(browser="chrome", headless=False, model="bert-base-uncased"):
    """Test WebNN implementation.
    
    Args:
        browser: Browser to use
        headless: Whether to run in headless mode
        model: Model to test
        
    Returns:
        Dict with test results
    """
    logger.info(f"Testing WebNN with {browser} browser")
    
    if implementation_type == "unified":
        return await test_webnn_unified(browser, headless, model)
    else:
        return await test_webnn_direct(browser, headless, model)

async def test_webnn_unified(browser, headless, model):
    """Test WebNN with UnifiedWebImplementation.
    
    Args:
        browser: Browser to use
        headless: Whether to run in headless mode
        model: Model to test
        
    Returns:
        Dict with test results
    """
    impl = UnifiedWebImplementation()
    
    try:
        # Check if WebNN hardware is available
        hardware_available = impl.is_hardware_available("webnn")
        logger.info(f"WebNN hardware acceleration available: {hardware_available}")
        
        # Initialize model
        logger.info(f"Initializing model {model} on WebNN")
        init_result = impl.init_model(model, model_type="text", platform="webnn")
        
        if not init_result:
            logger.error(f"Failed to initialize model {model} on WebNN")
            impl.shutdown()
            return {
                "success": False,
                "error": "Model initialization failed"
            }
        
        # Run inference
        logger.info(f"Running inference with model {model} on WebNN")
        input_text = "This is a test input for WebNN inference"
        inference_result = impl.run_inference(model, input_text, platform="webnn")
        
        if not inference_result:
            logger.error(f"Failed to run inference with model {model} on WebNN")
            impl.shutdown()
            return {
                "success": False,
                "error": "Inference failed"
            }
        
        # Check if using simulation
        is_simulation = inference_result.get("is_simulation", True)
        implementation_type = inference_result.get("implementation_type", "UNKNOWN")
        
        # Get performance metrics
        perf_metrics = inference_result.get("performance_metrics", {})
        
        # Shutdown
        impl.shutdown()
        
        return {
            "success": True,
            "hardware_available": hardware_available,
            "is_simulation": is_simulation,
            "implementation_type": implementation_type,
            "performance_metrics": perf_metrics
        }
        
    except Exception as e:
        logger.error(f"Error testing WebNN: {e}")
        impl.shutdown()
        return {
            "success": False,
            "error": str(e)
        }

async def test_webnn_direct(browser, headless, model):
    """Test WebNN with RealWebNNImplementation.
    
    Args:
        browser: Browser to use
        headless: Whether to run in headless mode
        model: Model to test
        
    Returns:
        Dict with test results
    """
    impl = RealWebNNImplementation(browser_name=browser, headless=headless)
    
    try:
        # Initialize implementation
        success = await impl.initialize()
        if not success:
            logger.error("Failed to initialize WebNN implementation")
            return {
                "success": False,
                "error": "Failed to initialize WebNN implementation"
            }
        
        # Initialize model
        logger.info(f"Initializing model {model} on WebNN")
        init_result = await impl.initialize_model(model, model_type="text")
        
        if not init_result:
            logger.error(f"Failed to initialize model {model} on WebNN")
            await impl.shutdown()
            return {
                "success": False,
                "error": "Model initialization failed"
            }
        
        # Run inference
        logger.info(f"Running inference with model {model} on WebNN")
        input_text = "This is a test input for WebNN inference"
        inference_result = await impl.run_inference(model, input_text)
        
        if not inference_result:
            logger.error(f"Failed to run inference with model {model} on WebNN")
            await impl.shutdown()
            return {
                "success": False,
                "error": "Inference failed"
            }
        
        # Check implementation details
        impl_details = inference_result.get("_implementation_details", {})
        is_simulation = impl_details.get("is_simulation", True)
        implementation_type = impl_details.get("implementation_type", "UNKNOWN")
        
        # Get performance metrics
        perf_metrics = inference_result.get("performance_metrics", {})
        
        # Shutdown
        await impl.shutdown()
        
        return {
            "success": True,
            "is_simulation": is_simulation,
            "implementation_type": implementation_type,
            "performance_metrics": perf_metrics
        }
        
    except Exception as e:
        logger.error(f"Error testing WebNN: {e}")
        await impl.shutdown()
        return {
            "success": False,
            "error": str(e)
        }

def display_results(webgpu_results, webnn_results):
    """Display test results.
    
    Args:
        webgpu_results: Results from WebGPU test
        webnn_results: Results from WebNN test
    """
    print("\n===== WebNN and WebGPU Implementation Test Results =====\n")
    
    # WebGPU results
    print("WebGPU Implementation:")
    if not webgpu_results.get("success", False):
        print(f"  ❌ Error: {webgpu_results.get('error', 'Unknown error')}")
    else:
        is_simulation = webgpu_results.get("is_simulation", True)
        implementation_type = webgpu_results.get("implementation_type", "UNKNOWN")
        
        print(f"  Implementation Type: {implementation_type}")
        print(f"  Using Real Hardware: {'✅ Yes' if not is_simulation else '❌ No (Simulation)'}")
        
        perf_metrics = webgpu_results.get("performance_metrics", {})
        if perf_metrics:
            print("  Performance Metrics:")
            print(f"    Inference Time: {perf_metrics.get('inference_time_ms', 'N/A')} ms")
            print(f"    Throughput: {perf_metrics.get('throughput_items_per_sec', 'N/A')} items/sec")
    
    # WebNN results
    print("\nWebNN Implementation:")
    if not webnn_results.get("success", False):
        print(f"  ❌ Error: {webnn_results.get('error', 'Unknown error')}")
    else:
        is_simulation = webnn_results.get("is_simulation", True)
        implementation_type = webnn_results.get("implementation_type", "UNKNOWN")
        
        print(f"  Implementation Type: {implementation_type}")
        print(f"  Using Real Hardware: {'✅ Yes' if not is_simulation else '❌ No (Simulation)'}")
        
        perf_metrics = webnn_results.get("performance_metrics", {})
        if perf_metrics:
            print("  Performance Metrics:")
            print(f"    Inference Time: {perf_metrics.get('inference_time_ms', 'N/A')} ms")
            print(f"    Throughput: {perf_metrics.get('throughput_items_per_sec', 'N/A')} items/sec")
    
    # Overall conclusion
    print("\nConclusion:")
    webgpu_real = webgpu_results.get("success", False) and not webgpu_results.get("is_simulation", True)
    webnn_real = webnn_results.get("success", False) and not webnn_results.get("is_simulation", True)
    
    if webgpu_real and webnn_real:
        print("  ✅ BOTH implementations are using REAL hardware acceleration.")
    elif webgpu_real:
        print("  ⚠️  WebGPU is using REAL hardware acceleration, but WebNN is using simulation.")
    elif webnn_real:
        print("  ⚠️  WebNN is using REAL hardware acceleration, but WebGPU is using simulation.")
    else:
        print("  ❌ BOTH implementations are using SIMULATION, not real hardware acceleration.")
    
    print("\n====================================================\n")

async def main_async(args):
    """Run tests asynchronously."""
    results = {}
    
    # Test WebGPU if requested
    if args.platform in ["webgpu", "both"]:
        results["webgpu"] = await test_webgpu(args.browser, args.headless, args.model)
    else:
        results["webgpu"] = {"success": False, "error": "Not tested"}
    
    # Test WebNN if requested
    if args.platform in ["webnn", "both"]:
        results["webnn"] = await test_webnn(args.browser, args.headless, args.model)
    else:
        results["webnn"] = {"success": False, "error": "Not tested"}
    
    # Display results
    display_results(results["webgpu"], results["webnn"])
    
    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    # Return 0 if at least one real implementation was found
    webgpu_real = results["webgpu"].get("success", False) and not results["webgpu"].get("is_simulation", True)
    webnn_real = results["webnn"].get("success", False) and not results["webnn"].get("is_simulation", True)
    
    return 0 if (webgpu_real or webnn_real) else 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run WebNN and WebGPU implementation test")
    parser.add_argument("--browser", choices=["chrome", "firefox", "edge"], default="chrome",
                      help="Browser to use for testing")
    parser.add_argument("--platform", choices=["webgpu", "webnn", "both"], default="both",
                      help="Platform to test")
    parser.add_argument("--headless", action="store_true",
                      help="Run in headless mode")
    parser.add_argument("--model", default="bert-base-uncased",
                      help="Model to use for testing")
    parser.add_argument("--output", type=str,
                      help="Output file for test results (JSON)")
    
    args = parser.parse_args()
    
    # Run async function
    try:
        return anyio.run(main_async, args)
    except KeyboardInterrupt:
        print("Test interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())