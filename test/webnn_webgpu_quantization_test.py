#!/usr/bin/env python3
"""
WebNN/WebGPU Quantization Test

This script tests WebNN and WebGPU implementations with 4-bit quantization.
It integrates with the real browser-based implementation and tests memory
and performance benefits of quantization.

Usage:
    python webnn_webgpu_quantization_test.py --platform webgpu --browser chrome --model bert-base-uncased
    python webnn_webgpu_quantization_test.py --platform webnn --browser edge --model bert-base-uncased
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules with fallbacks
try:
    from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation
    WEBGPU_AVAILABLE = True
except ImportError:
    logger.warning("WebGPU implementation not available")
    WEBGPU_AVAILABLE = False

try:
    from fixed_web_platform.webnn_implementation import RealWebNNImplementation
    WEBNN_AVAILABLE = True
except ImportError:
    logger.warning("WebNN implementation not available")
    WEBNN_AVAILABLE = False

try:
    from fixed_web_platform.webgpu_quantization import (
        WebGPUQuantizer,
        setup_4bit_inference
    )
    QUANTIZATION_AVAILABLE = True
except ImportError:
    logger.warning("WebGPU quantization not available")
    QUANTIZATION_AVAILABLE = False

import asyncio

async def test_quantization(args):
    """Test WebNN/WebGPU with quantization."""
    platform = args.platform.lower()
    browser = args.browser.lower()
    model_name = args.model
    bits = args.bits
    
    logger.info(f"Testing {platform} with {bits}-bit quantization on {browser} browser")
    
    # Initialize implementation
    if platform == "webgpu" and WEBGPU_AVAILABLE:
        implementation = RealWebGPUImplementation(browser_name=browser, headless=args.headless)
    elif platform == "webnn" and WEBNN_AVAILABLE:
        implementation = RealWebNNImplementation(browser_name=browser, headless=args.headless)
    else:
        logger.error(f"Platform {platform} is not available")
        return 1
    
    try:
        # Start implementation
        await implementation.initialize()
        logger.info(f"Initialized {platform} implementation")
        
        # Check feature support
        features = implementation.get_feature_support()
        logger.info(f"Feature support: {json.dumps(features, indent=2)}")
        
        # Initialize model with quantization options
        quantization_config = {
            "use_quantization": True,
            "bits": bits,
            "group_size": 128,
            "scheme": "symmetric",
            "mixed_precision": args.mixed_precision
        }
        
        # Initialize model
        logger.info(f"Initializing model {model_name} with {bits}-bit quantization")
        start_time = time.time()
        # Note: initialize_model doesn't accept options parameter, we'll pass it during inference
        model_info = await implementation.initialize_model(model_name, model_type="text")
        end_time = time.time()
        
        if not model_info:
            logger.error(f"Failed to initialize model {model_name}")
            await implementation.shutdown()
            return 1
        
        init_time = (end_time - start_time) * 1000
        logger.info(f"Model initialized in {init_time:.2f}ms")
        logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
        
        # Run inference
        test_inputs = [
            "This is a test of quantized inference.",
            "Let's see how well 4-bit precision works.",
            "WebGPU and WebNN provide hardware acceleration in browsers."
        ]
        
        # Run inference with each test input
        for i, input_text in enumerate(test_inputs):
            logger.info(f"Running inference {i+1}/{len(test_inputs)}")
            
            # Create inference options
            inference_options = {
                "use_quantization": True,
                "bits": bits,
                "measure_performance": True,
                "mixed_precision": args.mixed_precision,
                "scheme": "symmetric"  # Default scheme
            }
            
            # Run inference
            start_time = time.time()
            # Pass the options directly as the optional options parameter
            result = await implementation.run_inference(model_name, input_text, inference_options)
            end_time = time.time()
            
            if not result:
                logger.error(f"Inference {i+1} failed")
                continue
            
            # Calculate inference time
            inference_time = (end_time - start_time) * 1000
            
            # Log performance metrics
            logger.info(f"Inference {i+1} completed in {inference_time:.2f}ms")
            if "performance_metrics" in result:
                metrics = result["performance_metrics"]
                logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")
            
            # Log summary
            is_simulation = result.get("is_simulation", True)
            if is_simulation:
                logger.warning("Inference was performed using simulation, not hardware acceleration")
            else:
                logger.info("Inference was performed using real hardware acceleration")
                
            # Show memory usage if available
            if "memory_usage" in result:
                memory = result["memory_usage"]
                logger.info(f"Memory usage: {memory.get('used_mb', 'N/A')}MB")
                logger.info(f"Memory reduction: {memory.get('reduction_percent', 'N/A')}%")
        
        # Shutdown implementation
        await implementation.shutdown()
        logger.info(f"{platform} implementation shutdown")
        return 0
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        try:
            await implementation.shutdown()
        except:
            pass
        return 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="WebNN/WebGPU Quantization Test")
    
    parser.add_argument("--platform", type=str, choices=["webgpu", "webnn"], default="webgpu",
                        help="Platform to test")
    
    parser.add_argument("--browser", type=str, default="chrome",
                        help="Browser to use (chrome, firefox, edge, safari)")
    
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Model to test")
    
    parser.add_argument("--bits", type=int, choices=[2, 4, 8, 16], default=4,
                        help="Bit precision for quantization")
    
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use mixed precision (higher bits for critical layers)")
    
    parser.add_argument("--headless", action="store_true",
                        help="Run browser in headless mode")
    
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(test_quantization(args))

if __name__ == "__main__":
    sys.exit(main())