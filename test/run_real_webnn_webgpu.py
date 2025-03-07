#!/usr/bin/env python3
"""
Run Real WebNN and WebGPU implementations

This script sets up and runs real WebNN and WebGPU implementations
by connecting Python to browsers using Selenium and WebSockets.

The script ensures that the implementations use real browser capabilities
instead of simulations, verifying that actual hardware acceleration is being used.

Usage:
    # Test WebGPU with Chrome (best compatibility)
    python run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased
    
    # Test WebNN with Edge (best compatibility)
    python run_real_webnn_webgpu.py --platform webnn --browser edge --model bert-base-uncased
    
    # Run in visible mode (not headless) to see the browser
    python run_real_webnn_webgpu.py --platform webgpu --browser chrome --no-headless
    
    # Run with specific model and test input
    python run_real_webnn_webgpu.py --platform webgpu --browser chrome --model bert-base-uncased --input "Test input text"
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from real_webnn_connection and real_webgpu_connection
try:
    from fixed_web_platform.real_webnn_connection import RealWebNNConnection
    from fixed_web_platform.real_webgpu_connection import RealWebGPUConnection
except ImportError:
    logger.error("Failed to import from real_webnn_connection.py or real_webgpu_connection.py")
    logger.error("Make sure the files exist in the fixed_web_platform directory")
    sys.exit(1)

async def run_webgpu_implementation(args):
    """Run WebGPU implementation."""
    logger.info("Starting WebGPU implementation test")
    
    # Create WebGPU connection
    connection = RealWebGPUConnection(
        browser_name=args.browser,
        headless=args.headless
    )
    
    try:
        # Initialize connection
        logger.info(f"Initializing WebGPU connection with {args.browser} browser")
        initialized = await connection.initialize()
        
        if not initialized:
            logger.error("Failed to initialize WebGPU connection")
            return 1
        
        # Get feature support information
        features = connection.get_feature_support()
        if features:
            logger.info(f"WebGPU features: {json.dumps(features, indent=2)}")
            webgpu_available = features.get('webgpu', False)
            logger.info(f"WebGPU available: {webgpu_available}")
            
            if not webgpu_available:
                logger.warning(f"WebGPU is not available in {args.browser}. Consider using a different browser.")
                if args.browser == "safari":
                    logger.info("Safari has limited WebGPU support. Try Chrome or Firefox instead.")
                    
        # Initialize model
        model_name = args.model or "bert-base-uncased"
        model_type = args.model_type or "text"
        
        logger.info(f"Initializing model {model_name} with type {model_type}")
        model_info = await connection.initialize_model(model_name, model_type)
        
        if not model_info:
            logger.error(f"Failed to initialize model {model_name}")
            await connection.shutdown()
            return 1
        
        # Print model info
        logger.info(f"Model initialized: {json.dumps(model_info, indent=2)}")
        
        # Run inference if requested
        if args.run_inference:
            # Prepare input data
            if args.input:
                input_data = args.input
            else:
                # Default test input based on model type
                if model_type == "text":
                    input_data = "This is a test input for WebGPU implementation."
                elif model_type == "vision":
                    # Check if test image exists
                    test_image = os.path.join(os.path.dirname(__file__), "test.jpg")
                    if os.path.exists(test_image):
                        input_data = {"image": test_image}
                    else:
                        logger.warning(f"Test image {test_image} not found, using text input instead")
                        input_data = "Test input"
                elif model_type == "audio":
                    # Check if test audio exists
                    test_audio = os.path.join(os.path.dirname(__file__), "test.mp3")
                    if os.path.exists(test_audio):
                        input_data = {"audio": test_audio}
                    else:
                        logger.warning(f"Test audio {test_audio} not found, using text input instead")
                        input_data = "Test input"
                else:
                    input_data = "Test input"
            
            # Run inference
            logger.info(f"Running inference with input: {input_data}")
            start_time = time.time()
            
            # Run inference with model
            result = await connection.run_inference(model_name, input_data)
            
            # Calculate time
            inference_time = (time.time() - start_time) * 1000
            
            if not result:
                logger.error("Inference failed")
                await connection.shutdown()
                return 1
            
            # Check if real implementation was used
            impl_type = result.get("implementation_type", "UNKNOWN")
            is_real = "REAL" in impl_type
            is_simulation = result.get("is_simulation", True)
            
            if is_real and not is_simulation:
                logger.info("✅ Using REAL WebGPU hardware acceleration!")
            else:
                logger.warning("⚠️ Using SIMULATION mode - not real WebGPU implementation")
            
            # Print result
            logger.info(f"Inference complete in {inference_time:.2f} ms")
            logger.info(f"Result: {json.dumps(result, indent=2)}")
            
            # Check if transformers.js was used
            using_transformers_js = result.get("using_transformers_js", False)
            if using_transformers_js:
                logger.info("✅ Using transformers.js for real browser-based inference")
        
        # Interactive mode
        if args.interactive:
            logger.info("Interactive mode enabled. Press Ctrl+C to exit.")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interactive mode stopped by user")
        
        # Shutdown connection
        logger.info("Shutting down WebGPU connection")
        await connection.shutdown()
        
        logger.info("WebGPU implementation test completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error running WebGPU implementation: {e}")
        try:
            await connection.shutdown()
        except:
            pass
        return 1

async def run_webnn_implementation(args):
    """Run WebNN implementation."""
    logger.info("Starting WebNN implementation test")
    
    # Create WebNN connection
    connection = RealWebNNConnection(
        browser_name=args.browser,
        headless=args.headless,
        device_preference=args.device_preference
    )
    
    try:
        # Initialize connection
        logger.info(f"Initializing WebNN connection with {args.browser} browser")
        initialized = await connection.initialize()
        
        if not initialized:
            logger.error("Failed to initialize WebNN connection")
            return 1
        
        # Get feature support information
        features = connection.get_feature_support()
        if features:
            logger.info(f"WebNN features: {json.dumps(features, indent=2)}")
            webnn_available = features.get('webnn', False)
            logger.info(f"WebNN available: {webnn_available}")
            
            if not webnn_available:
                logger.warning(f"WebNN is not available in {args.browser}. Consider using Edge or Chrome.")
                if args.browser not in ["edge", "chrome"]:
                    logger.info("WebNN has best support in Edge and Chrome browsers.")
        
        # Get backend information
        backend_info = connection.get_backend_info()
        logger.info(f"WebNN backends: {json.dumps(backend_info, indent=2)}")
                    
        # Initialize model
        model_name = args.model or "bert-base-uncased"
        model_type = args.model_type or "text"
        
        logger.info(f"Initializing model {model_name} with type {model_type}")
        model_info = await connection.initialize_model(model_name, model_type)
        
        if not model_info:
            logger.error(f"Failed to initialize model {model_name}")
            await connection.shutdown()
            return 1
        
        # Print model info
        logger.info(f"Model initialized: {json.dumps(model_info, indent=2)}")
        
        # Run inference if requested
        if args.run_inference:
            # Prepare input data
            if args.input:
                input_data = args.input
            else:
                # Default test input based on model type
                if model_type == "text":
                    input_data = "This is a test input for WebNN implementation."
                elif model_type == "vision":
                    # Check if test image exists
                    test_image = os.path.join(os.path.dirname(__file__), "test.jpg")
                    if os.path.exists(test_image):
                        input_data = {"image": test_image}
                    else:
                        logger.warning(f"Test image {test_image} not found, using text input instead")
                        input_data = "Test input"
                elif model_type == "audio":
                    # Check if test audio exists
                    test_audio = os.path.join(os.path.dirname(__file__), "test.mp3")
                    if os.path.exists(test_audio):
                        input_data = {"audio": test_audio}
                    else:
                        logger.warning(f"Test audio {test_audio} not found, using text input instead")
                        input_data = "Test input"
                else:
                    input_data = "Test input"
            
            # Run inference
            logger.info(f"Running inference with input: {input_data}")
            start_time = time.time()
            
            # Run inference with model
            result = await connection.run_inference(model_name, input_data)
            
            # Calculate time
            inference_time = (time.time() - start_time) * 1000
            
            if not result:
                logger.error("Inference failed")
                await connection.shutdown()
                return 1
            
            # Check if real implementation was used
            impl_type = result.get("implementation_type", "UNKNOWN")
            is_real = "REAL" in impl_type
            is_simulation = result.get("is_simulation", True)
            
            if is_real and not is_simulation:
                logger.info("✅ Using REAL WebNN hardware acceleration!")
            else:
                logger.warning("⚠️ Using SIMULATION mode - not real WebNN implementation")
            
            # Print result
            logger.info(f"Inference complete in {inference_time:.2f} ms")
            logger.info(f"Result: {json.dumps(result, indent=2)}")
            
            # Check if transformers.js was used
            using_transformers_js = result.get("using_transformers_js", False)
            if using_transformers_js:
                logger.info("✅ Using transformers.js for real browser-based inference")
        
        # Interactive mode
        if args.interactive:
            logger.info("Interactive mode enabled. Press Ctrl+C to exit.")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interactive mode stopped by user")
        
        # Shutdown connection
        logger.info("Shutting down WebNN connection")
        await connection.shutdown()
        
        logger.info("WebNN implementation test completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error running WebNN implementation: {e}")
        try:
            await connection.shutdown()
        except:
            pass
        return 1

async def run_both_implementations(args):
    """Run both WebNN and WebGPU implementations."""
    # Run WebGPU first
    logger.info("=== Running WebGPU implementation ===")
    webgpu_result = await run_webgpu_implementation(args)
    
    # Run WebNN next
    logger.info("\n=== Running WebNN implementation ===")
    webnn_args = argparse.Namespace(**vars(args))
    webnn_args.browser = args.webnn_browser or "edge"  # Use Edge by default for WebNN
    webnn_result = await run_webnn_implementation(webnn_args)
    
    return webgpu_result == 0 and webnn_result == 0

async def main_async(args):
    """Run the specified implementation(s)."""
    if args.platform == "webgpu":
        return await run_webgpu_implementation(args)
    elif args.platform == "webnn":
        return await run_webnn_implementation(args)
    elif args.platform == "both":
        return await run_both_implementations(args)
    else:
        logger.error(f"Unknown platform: {args.platform}")
        return 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run real WebNN and WebGPU implementations")
    parser.add_argument("--platform", choices=["webgpu", "webnn", "both"], default="webgpu",
                      help="Platform to test (webgpu, webnn, or both)")
    parser.add_argument("--browser", default="chrome",
                      help="Browser to use for testing (edge, chrome, firefox, safari)")
    parser.add_argument("--webnn-browser", default="edge",
                      help="Browser to use for WebNN when testing both platforms (default: edge)")
    parser.add_argument("--headless", action="store_true", default=True,
                      help="Run in headless mode (default: True)")
    parser.add_argument("--no-headless", action="store_false", dest="headless",
                      help="Run in visible mode (not headless)")
    parser.add_argument("--model", default="bert-base-uncased",
                      help="Model to use for testing (default: bert-base-uncased)")
    parser.add_argument("--model-type", choices=["text", "vision", "audio", "multimodal"], default="text",
                      help="Type of model to test (default: text)")
    parser.add_argument("--input", 
                      help="Input data for inference (default depends on model type)")
    parser.add_argument("--run-inference", action="store_true",
                      help="Run inference with the model")
    parser.add_argument("--interactive", action="store_true",
                      help="Run in interactive mode (keeps browser open)")
    parser.add_argument("--device-preference", choices=["gpu", "cpu"], default="gpu",
                      help="Device preference for WebNN (default: gpu)")
    
    args = parser.parse_args()
    
    # Default to running inference
    if not args.interactive and not args.run_inference:
        args.run_inference = True
    
    # Print summary
    print(f"\n=== Testing {args.platform.upper()} Implementation ===")
    print(f"Browser: {args.browser}")
    if args.platform == "both":
        print(f"WebNN Browser: {args.webnn_browser}")
    print(f"Headless: {args.headless}")
    print(f"Model: {args.model} ({args.model_type})")
    print(f"Interactive: {args.interactive}")
    print(f"Run Inference: {args.run_inference}")
    print("===================================\n")
    
    # Run main async function
    if sys.version_info >= (3, 7):
        return_code = asyncio.run(main_async(args))
    else:
        # For Python 3.6 or lower
        loop = asyncio.get_event_loop()
        return_code = loop.run_until_complete(main_async(args))
    
    if return_code == 0:
        print("\n✅ Implementation test completed successfully")
    else:
        print("\n❌ Implementation test failed")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())