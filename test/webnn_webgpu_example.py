#!/usr/bin/env python3
"""
WebNN and WebGPU Usage Example

This script demonstrates how to use the WebNN and WebGPU implementations
in your own Python code. It provides simple examples for both direct usage
of implementation classes and the unified interface.

Usage:
    python webnn_webgpu_example.py --platform webgpu --browser chrome --model bert-base-uncased
    python webnn_webgpu_example.py --platform webnn --browser firefox --model t5-small
    python webnn_webgpu_example.py --unified --platform webgpu --model bert-base-uncased
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Default settings
DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_MODEL_TYPE = "text"
DEFAULT_PLATFORM = "webgpu"
DEFAULT_BROWSER = "chrome"

async def run_webgpu_example(model_name, model_type, browser_name="chrome", headless=True):
    """Example of using WebGPU implementation directly."""
    from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation
    
    logger.info(f"Running WebGPU example with {model_name} model on {browser_name} browser")
    
    # Create implementation
    impl = RealWebGPUImplementation(browser_name=browser_name, headless=headless)
    
    try:
        # Step 1: Initialize implementation
        logger.info("Initializing WebGPU implementation")
        success = await impl.initialize()
        if not success:
            logger.error("Failed to initialize WebGPU implementation")
            return
        
        # Step 2: Get feature support information
        features = impl.get_feature_support()
        logger.info(f"WebGPU features: {json.dumps(features, indent=2)}")
        
        # Step 3: Initialize model
        logger.info(f"Initializing model: {model_name}")
        model_info = await impl.initialize_model(model_name, model_type=model_type)
        if not model_info:
            logger.error(f"Failed to initialize model: {model_name}")
            await impl.shutdown()
            return
        
        logger.info(f"Model initialized: {model_name}")
        
        # Step 4: Run inference
        logger.info("Running inference")
        
        # Simple inference example
        input_text = "This is an example input for WebGPU inference."
        
        # Run inference
        start_time = time.time()
        result = await impl.run_inference(model_name, input_text)
        inference_time = time.time() - start_time
        
        if not result:
            logger.error("Inference failed")
            await impl.shutdown()
            return
        
        # Log the result
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        
        # Check implementation type
        impl_type = result.get("implementation_type")
        if impl_type != "REAL_WEBGPU":
            logger.warning(f"Unexpected implementation type: {impl_type}, expected: REAL_WEBGPU")
        
        # Step 5: Shutdown
        await impl.shutdown()
        logger.info("WebGPU implementation shut down successfully")
        
    except Exception as e:
        logger.error(f"Error in WebGPU example: {e}")
        await impl.shutdown()

async def run_webnn_example(model_name, model_type, browser_name="chrome", headless=True):
    """Example of using WebNN implementation directly."""
    from fixed_web_platform.webnn_implementation import RealWebNNImplementation
    
    logger.info(f"Running WebNN example with {model_name} model on {browser_name} browser")
    
    # Create implementation
    impl = RealWebNNImplementation(browser_name=browser_name, headless=headless)
    
    try:
        # Step 1: Initialize implementation
        logger.info("Initializing WebNN implementation")
        success = await impl.initialize()
        if not success:
            logger.error("Failed to initialize WebNN implementation")
            return
        
        # Step 2: Get feature support information
        features = impl.get_feature_support()
        logger.info(f"WebNN features: {json.dumps(features, indent=2)}")
        
        # Get backend information
        backend_info = impl.get_backend_info()
        logger.info(f"WebNN backend: {json.dumps(backend_info, indent=2)}")
        
        # Step 3: Initialize model
        logger.info(f"Initializing model: {model_name}")
        model_info = await impl.initialize_model(model_name, model_type=model_type)
        if not model_info:
            logger.error(f"Failed to initialize model: {model_name}")
            await impl.shutdown()
            return
        
        logger.info(f"Model initialized: {model_name}")
        
        # Step 4: Run inference
        logger.info("Running inference")
        
        # Simple inference example
        input_text = "This is an example input for WebNN inference."
        
        # Run inference
        start_time = time.time()
        result = await impl.run_inference(model_name, input_text)
        inference_time = time.time() - start_time
        
        if not result:
            logger.error("Inference failed")
            await impl.shutdown()
            return
        
        # Log the result
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        
        # Check implementation type
        impl_type = result.get("implementation_type")
        if impl_type != "REAL_WEBNN":
            logger.warning(f"Unexpected implementation type: {impl_type}, expected: REAL_WEBNN")
        
        # Step 5: Shutdown
        await impl.shutdown()
        logger.info("WebNN implementation shut down successfully")
        
    except Exception as e:
        logger.error(f"Error in WebNN example: {e}")
        await impl.shutdown()

async def run_unified_example(platform, model_name, model_type, browser_name="chrome", headless=True):
    """Example of using unified platform interface."""
    from implement_real_webnn_webgpu import RealWebPlatformIntegration
    
    logger.info(f"Running unified interface example with {model_name} model on {browser_name} browser using {platform}")
    
    # Create integration
    integration = RealWebPlatformIntegration()
    
    try:
        # Step 1: Initialize platform
        logger.info(f"Initializing {platform} platform")
        success = await integration.initialize_platform(
            platform=platform,
            browser_name=browser_name,
            headless=headless
        )
        
        if not success:
            logger.error(f"Failed to initialize {platform} platform")
            return
        
        logger.info(f"{platform} platform initialized successfully")
        
        # Step 2: Initialize model
        logger.info(f"Initializing model: {model_name}")
        response = await integration.initialize_model(
            platform=platform,
            model_name=model_name,
            model_type=model_type
        )
        
        if not response or response.get("status") != "success":
            logger.error(f"Failed to initialize model: {model_name}")
            await integration.shutdown(platform)
            return
        
        logger.info(f"Model initialized: {model_name}")
        
        # Step 3: Run inference
        logger.info(f"Running inference with model: {model_name}")
        
        # Create test input
        input_text = "This is an example input for unified interface inference."
        
        # Run inference
        start_time = time.time()
        response = await integration.run_inference(
            platform=platform,
            model_name=model_name,
            input_data=input_text
        )
        inference_time = time.time() - start_time
        
        if not response or response.get("status") != "success":
            logger.error(f"Failed to run inference with model: {model_name}")
            await integration.shutdown(platform)
            return
        
        # Log the result
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        logger.info(f"Result: {json.dumps(response, indent=2)}")
        
        # Check implementation type
        impl_type = response.get("implementation_type")
        expected_type = "REAL_WEBGPU" if platform == "webgpu" else "REAL_WEBNN"
        
        if impl_type != expected_type:
            logger.warning(f"Unexpected implementation type: {impl_type}, expected: {expected_type}")
        
        # Step 4: Shutdown platform
        await integration.shutdown(platform)
        logger.info(f"{platform} platform shut down successfully")
        
    except Exception as e:
        logger.error(f"Error in unified interface example: {e}")
        await integration.shutdown(platform)

async def run_advanced_example(platform, model_name, browser_name="chrome", headless=True):
    """
    Advanced example showing how to use WebGPU/WebNN with the unified framework.
    This demonstrates working with the high-level unified_web_framework.
    """
    try:
        from fixed_web_platform.unified_web_framework import (
            WebPlatformAccelerator,
            create_web_endpoint,
            get_optimal_config
        )
        
        logger.info(f"Running advanced example with {model_name} on {platform} platform")
        
        # Step 1: Get optimal configuration for the model
        logger.info(f"Getting optimal configuration for {model_name}")
        config = get_optimal_config(
            model_path=model_name,
            model_type="text",
            browser=browser_name
        )
        
        logger.info(f"Optimal configuration: {json.dumps(config, indent=2)}")
        
        # Step 2: Create accelerator with configuration
        logger.info("Creating WebPlatformAccelerator")
        accelerator = WebPlatformAccelerator(
            model_path=model_name,
            model_type="text",
            config=config,
            auto_detect=True  # This will detect and use the best available features
        )
        
        # Step 3: Get performance metrics
        metrics = accelerator.get_performance_metrics()
        logger.info(f"Initialization metrics: {json.dumps(metrics, indent=2)}")
        
        # Step 4: Create inference endpoint
        logger.info("Creating inference endpoint")
        endpoint = accelerator.create_endpoint()
        
        # Step 5: Run inference
        logger.info("Running inference")
        input_text = "This is an example input for the unified framework."
        
        # Call the endpoint
        start_time = time.time()
        result = endpoint(input_text)
        inference_time = time.time() - start_time
        
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        
        # Step 6: Get updated performance metrics
        metrics = accelerator.get_performance_metrics()
        logger.info(f"Updated metrics: {json.dumps(metrics, indent=2)}")
        
        # Step 7: Get feature usage
        feature_usage = accelerator.get_feature_usage()
        logger.info(f"Feature usage: {json.dumps(feature_usage, indent=2)}")
        
        # Example: Alternative simplified version using create_web_endpoint
        logger.info("Alternative usage with create_web_endpoint")
        simple_endpoint = create_web_endpoint(
            model_path=model_name,
            model_type="text",
            config={"browser": browser_name, "use_webgpu": platform == "webgpu"}
        )
        
        simple_result = simple_endpoint(input_text)
        logger.info(f"Simple endpoint result: {json.dumps(simple_result, indent=2)}")
        
    except ImportError as e:
        logger.error(f"Error importing unified_web_framework: {e}")
        logger.info("This is normal if the unified framework implementation is not available")
    except Exception as e:
        logger.error(f"Error in advanced example: {e}")

async def main():
    """Main function for running examples."""
    parser = argparse.ArgumentParser(description="WebNN and WebGPU Usage Example")
    parser.add_argument("--platform", choices=["webgpu", "webnn"], default=DEFAULT_PLATFORM,
                       help="Platform to use")
    parser.add_argument("--browser", choices=["chrome", "firefox", "edge", "safari"], default=DEFAULT_BROWSER,
                       help="Browser to use")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="Model to use")
    parser.add_argument("--type", choices=["text", "vision", "audio", "multimodal"], default=DEFAULT_MODEL_TYPE,
                       help="Model type")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode")
    parser.add_argument("--unified", action="store_true",
                       help="Use the unified interface")
    parser.add_argument("--advanced", action="store_true",
                       help="Use the advanced unified framework (if available)")
    
    args = parser.parse_args()
    
    # Run the appropriate example
    if args.advanced:
        await run_advanced_example(args.platform, args.model, args.browser, args.headless)
    elif args.unified:
        await run_unified_example(args.platform, args.model, args.type, args.browser, args.headless)
    elif args.platform == "webgpu":
        await run_webgpu_example(args.model, args.type, args.browser, args.headless)
    elif args.platform == "webnn":
        await run_webnn_example(args.model, args.type, args.browser, args.headless)
    else:
        logger.error(f"Unknown platform: {args.platform}")
    
if __name__ == "__main__":
    asyncio.run(main())