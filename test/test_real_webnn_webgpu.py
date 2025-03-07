#!/usr/bin/env python3
"""
Test Real WebNN and WebGPU Implementation

This script tests the real browser-based implementations of WebNN and WebGPU.
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_webgpu_implementation(browser="chrome", headless=False, model="bert-base-uncased"):
    """Test WebGPU implementation with a real browser."""
    logger.info(f"Testing WebGPU implementation with {browser} browser")
    
    from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation
    
    try:
        # Create implementation
        impl = RealWebGPUImplementation(browser_name=browser, headless=headless)
        
        # Initialize
        logger.info("Initializing WebGPU implementation")
        success = await impl.initialize()
        if not success:
            logger.error("Failed to initialize WebGPU implementation")
            return False
        
        # Get feature support
        features = impl.get_feature_support()
        logger.info(f"WebGPU feature support: {json.dumps(features, indent=2)}")
        
        # Initialize model
        logger.info(f"Initializing model: {model}")
        model_info = await impl.initialize_model(model, model_type="text")
        if not model_info:
            logger.error(f"Failed to initialize model: {model}")
            await impl.shutdown()
            return False
        
        logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
        
        # Run inference
        logger.info(f"Running inference with model: {model}")
        result = await impl.run_inference(model, "This is a test input for WebGPU implementation.")
        if not result:
            logger.error("Failed to run inference")
            await impl.shutdown()
            return False
        
        # Check if real implementation was used
        impl_details = result.get("_implementation_details", {})
        is_simulation = impl_details.get("is_simulation", True)
        using_transformers = impl_details.get("using_transformers_js", False)
        
        if is_simulation:
            logger.warning("Using SIMULATION mode - not real WebGPU implementation")
        else:
            logger.info("Using REAL WebGPU hardware acceleration!")
            
        if using_transformers:
            logger.info("Using transformers.js for model inference")
        
        logger.info(f"Inference result: {json.dumps(result, indent=2)}")
        
        # Shutdown
        await impl.shutdown()
        logger.info("WebGPU implementation test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing WebGPU implementation: {e}")
        return False

async def test_webnn_implementation(browser="edge", headless=False, model="bert-base-uncased"):
    """Test WebNN implementation with a real browser."""
    logger.info(f"Testing WebNN implementation with {browser} browser")
    
    from fixed_web_platform.webnn_implementation import RealWebNNImplementation
    
    try:
        # Create implementation
        impl = RealWebNNImplementation(browser_name=browser, headless=headless)
        
        # Initialize
        logger.info("Initializing WebNN implementation")
        success = await impl.initialize()
        if not success:
            logger.error("Failed to initialize WebNN implementation")
            return False
        
        # Get feature support
        features = impl.get_feature_support()
        logger.info(f"WebNN feature support: {json.dumps(features, indent=2)}")
        
        # Initialize model
        logger.info(f"Initializing model: {model}")
        model_info = await impl.initialize_model(model, model_type="text")
        if not model_info:
            logger.error(f"Failed to initialize model: {model}")
            await impl.shutdown()
            return False
        
        logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
        
        # Run inference
        logger.info(f"Running inference with model: {model}")
        result = await impl.run_inference(model, "This is a test input for WebNN implementation.")
        if not result:
            logger.error("Failed to run inference")
            await impl.shutdown()
            return False
        
        # Check if real implementation was used
        impl_details = result.get("_implementation_details", {})
        is_simulation = impl_details.get("is_simulation", True)
        using_transformers = impl_details.get("using_transformers_js", False)
        
        if is_simulation:
            logger.warning("Using SIMULATION mode - not real WebNN implementation")
        else:
            logger.info("Using REAL WebNN hardware acceleration!")
            
        if using_transformers:
            logger.info("Using transformers.js for model inference")
        
        logger.info(f"Inference result: {json.dumps(result, indent=2)}")
        
        # Shutdown
        await impl.shutdown()
        logger.info("WebNN implementation test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing WebNN implementation: {e}")
        return False

async def test_both_implementations(webgpu_browser="chrome", webnn_browser="edge", headless=False, model="bert-base-uncased"):
    """Test both WebGPU and WebNN implementations."""
    # Test WebGPU
    webgpu_success = await test_webgpu_implementation(browser=webgpu_browser, headless=headless, model=model)
    
    # Test WebNN
    webnn_success = await test_webnn_implementation(browser=webnn_browser, headless=headless, model=model)
    
    return webgpu_success and webnn_success

async def main_async(args):
    """Run tests asynchronously."""
    if args.platform == "webgpu":
        return await test_webgpu_implementation(
            browser=args.browser, 
            headless=args.headless,
            model=args.model
        )
    elif args.platform == "webnn":
        return await test_webnn_implementation(
            browser=args.browser,
            headless=args.headless,
            model=args.model
        )
    elif args.platform == "both":
        return await test_both_implementations(
            webgpu_browser=args.webgpu_browser,
            webnn_browser=args.webnn_browser,
            headless=args.headless,
            model=args.model
        )
    else:
        logger.error(f"Invalid platform: {args.platform}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Real WebNN and WebGPU Implementation")
    parser.add_argument("--platform", choices=["webgpu", "webnn", "both"], default="both",
                      help="Platform to test")
    parser.add_argument("--browser", default="chrome",
                      help="Browser to use for testing single platform")
    parser.add_argument("--webgpu-browser", default="chrome",
                      help="Browser to use for WebGPU when testing both platforms")
    parser.add_argument("--webnn-browser", default="edge",
                      help="Browser to use for WebNN when testing both platforms")
    parser.add_argument("--headless", action="store_true",
                      help="Run in headless mode")
    parser.add_argument("--model", default="bert-base-uncased",
                      help="Model to test")
    
    args = parser.parse_args()
    
    # Inform the user about the test
    print(f"\n===== Testing {args.platform.upper()} Implementation =====")
    if args.platform == "both":
        print(f"WebGPU Browser: {args.webgpu_browser}")
        print(f"WebNN Browser: {args.webnn_browser}")
    else:
        print(f"Browser: {args.browser}")
    print(f"Model: {args.model}")
    print(f"Headless: {args.headless}")
    print("===========================================\n")
    
    # Run test
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(main_async(args))
    
    if success:
        print("\n✅ Test completed successfully")
        return 0
    else:
        print("\n❌ Test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())