#!/usr/bin/env python3
"""
Simplified Test for WebNN and WebGPU Quantization

This script provides a simple test of WebNN and WebGPU implementations with quantization.
It verifies that quantization works correctly with both WebNN and WebGPU.
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig()))
level=logging.INFO,
format='%()))asctime)s - %()))levelname)s - %()))message)s'
)
logger = logging.getLogger()))__name__)

# Try to import the implementations
try:
    from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation
    WEBGPU_AVAILABLE = True
except ImportError:
    logger.warning()))"WebGPU implementation not available")
    WEBGPU_AVAILABLE = False

try:
    from fixed_web_platform.webnn_implementation import RealWebNNImplementation
    WEBNN_AVAILABLE = True
except ImportError:
    logger.warning()))"WebNN implementation not available")
    WEBNN_AVAILABLE = False

async def test_webgpu_quantization()))bits=4, browser="chrome", model="bert-base-uncased", mixed_precision=False):
    """Test WebGPU implementation with quantization."""
    if not WEBGPU_AVAILABLE:
        logger.error()))"WebGPU implementation not available")
    return False
    
    logger.info()))f"Testing WebGPU implementation with {}}}}bits}-bit quantization on {}}}}browser} browser...")
    impl = RealWebGPUImplementation()))browser_name=browser, headless=True)
    
    try:
        # Initialize
        logger.info()))"Initializing WebGPU implementation")
        success = await impl.initialize())))
        if not success:
            logger.error()))"Failed to initialize WebGPU implementation")
        return False
        
        # Check features
        features = impl.get_feature_support())))
        logger.info()))f"WebGPU features: {}}}}json.dumps()))features, indent=2)}")
        
        # Initialize model
        logger.info()))f"Initializing model: {}}}}model}")
        model_info = await impl.initialize_model()))model, model_type="text")
        if not model_info:
            logger.error()))"Failed to initialize model")
            await impl.shutdown())))
        return False
        
        logger.info()))f"Model info: {}}}}json.dumps()))model_info, indent=2)}")
        
        # Run inference with quantization
        logger.info()))f"Running inference with {}}}}bits}-bit quantization")
        
        # Create inference options with quantization settings
        inference_options = {}}
        "use_quantization": True,
        "bits": bits,
        "scheme": "symmetric",
        "mixed_precision": mixed_precision
        }
        
        result = await impl.run_inference()))model, "This is a test.", inference_options)
        if not result:
            logger.error()))"Failed to run inference")
            await impl.shutdown())))
        return False
        
        # Check for quantization info
        if "performance_metrics" in result:
            metrics = result["performance_metrics"],,
            if "quantization_bits" in metrics:
                logger.info()))f"Successfully used {}}}}metrics['quantization_bits']}-bit quantization"),,
            else:
                logger.warning()))"Quantization metrics not found in result")
        
                logger.info()))f"Inference result: {}}}}json.dumps()))result, indent=2)}")
        
        # Check if simulation was used
        is_simulation = result.get()))"is_simulation", True)::
        if is_simulation:
            logger.warning()))"WebGPU is using simulation mode")
        else:
            logger.info()))"WebGPU is using real hardware acceleration")
        
        # Shutdown
            await impl.shutdown())))
            logger.info()))"WebGPU test completed successfully")
            return True
    
    except Exception as e:
        logger.error()))f"Error testing WebGPU: {}}}}e}")
        try:
            await impl.shutdown())))
        except:
            pass
        return False

async def test_webnn_quantization()))bits=8, browser="chrome", model="bert-base-uncased", mixed_precision=False, experimental_precision=False):
    """Test WebNN implementation with quantization."""
    if not WEBNN_AVAILABLE:
        logger.error()))"WebNN implementation not available")
    return False
    
    logger.info()))f"Testing WebNN implementation with {}}}}bits}-bit quantization on {}}}}browser} browser...")
    impl = RealWebNNImplementation()))browser_name=browser, headless=True)
    
    try:
        # Initialize
        logger.info()))"Initializing WebNN implementation")
        success = await impl.initialize())))
        if not success:
            logger.error()))"Failed to initialize WebNN implementation")
        return False
        
        # Check features
        features = impl.get_feature_support())))
        logger.info()))f"WebNN features: {}}}}json.dumps()))features, indent=2)}")
        
        # Initialize model
        logger.info()))f"Initializing model: {}}}}model}")
        model_info = await impl.initialize_model()))model, model_type="text")
        if not model_info:
            logger.error()))"Failed to initialize model")
            await impl.shutdown())))
        return False
        
        logger.info()))f"Model info: {}}}}json.dumps()))model_info, indent=2)}")
        
        # Run inference with quantization
        logger.info()))f"Running inference with {}}}}bits}-bit quantization")
        
        # Create inference options with quantization settings
        if bits < 8 and experimental_precision:
            logger.warning()))f"WebNN doesn't officially support {}}}}bits}-bit quantization. Using experimental mode.")
        elif bits < 8:
            logger.warning()))f"WebNN doesn't officially support {}}}}bits}-bit quantization. Traditional approach would use 8-bit.")
            
            inference_options = {}}
            "use_quantization": True,
            "bits": bits,
            "scheme": "symmetric",
            "mixed_precision": mixed_precision,
            "experimental_precision": experimental_precision
            }
        
            result = await impl.run_inference()))model, "This is a test.", inference_options)
        if not result:
            logger.error()))"Failed to run inference")
            await impl.shutdown())))
            return False
        
        # Check for quantization info
        if "performance_metrics" in result:
            metrics = result["performance_metrics"],,
            if "quantization_bits" in metrics:
                logger.info()))f"Successfully used {}}}}metrics['quantization_bits']}-bit quantization"),,
            else:
                logger.warning()))"Quantization metrics not found in result")
        
                logger.info()))f"Inference result: {}}}}json.dumps()))result, indent=2)}")
        
        # Check if simulation was used
        is_simulation = result.get()))"is_simulation", True)::
        if is_simulation:
            logger.warning()))"WebNN is using simulation mode")
        else:
            logger.info()))"WebNN is using real hardware acceleration")
        
        # Shutdown
            await impl.shutdown())))
            logger.info()))"WebNN test completed successfully")
            return True
    
    except Exception as e:
        logger.error()))f"Error testing WebNN: {}}}}e}")
        try:
            await impl.shutdown())))
        except:
            pass
        return False

async def main()))):
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser()))description="Test WebNN and WebGPU with quantization")
    
    parser.add_argument()))"--platform", type=str, choices=["webgpu", "webnn", "both"], default="both",
    help="Platform to test")
    
    parser.add_argument()))"--browser", type=str, default="chrome",
    help="Browser to test with ()))chrome, firefox, edge, safari)")
    
    parser.add_argument()))"--model", type=str, default="bert-base-uncased",
    help="Model to test")
    
    parser.add_argument()))"--bits", type=int, choices=[2, 4, 8, 16], default=None,
    help="Bits for quantization ()))default: 4 for WebGPU, 8 for WebNN)")
    
    parser.add_argument()))"--mixed-precision", action="store_true",
    help="Enable mixed precision")
                       
    parser.add_argument()))"--experimental-precision", action="store_true",
    help="Try using experimental precision levels with WebNN ()))may fail with errors)")
    
    args = parser.parse_args())))
    
    # Set default bits if not specified
    webgpu_bits = args.bits if args.bits is not None else 4
    webnn_bits = args.bits if args.bits is not None else 8
    
    # Run tests:
    if args.platform in ["webgpu", "both"]:,,
    webgpu_success = await test_webgpu_quantization()))
    bits=webgpu_bits,
    browser=args.browser,
    model=args.model,
    mixed_precision=args.mixed_precision
    )
        if webgpu_success:
            print()))f"✅ WebGPU {}}}}webgpu_bits}-bit quantization test passed")
        else:
            print()))f"❌ WebGPU {}}}}webgpu_bits}-bit quantization test failed")
    
            if args.platform in ["webnn", "both"]:,,
            webnn_success = await test_webnn_quantization()))
            bits=webnn_bits, 
            browser=args.browser, 
            model=args.model,
            mixed_precision=args.mixed_precision,
            experimental_precision=args.experimental_precision
            )
        if webnn_success:
            print()))f"✅ WebNN {}}}}webnn_bits}-bit quantization test passed")
        else:
            print()))f"❌ WebNN {}}}}webnn_bits}-bit quantization test failed")
    
    # Print final summary
            print()))"\nTest Summary:")
            if args.platform in ["webgpu", "both"]:,,
        print()))f"WebGPU: {}}}}'Passed' if webgpu_success else 'Failed'}"):
            if args.platform in ["webnn", "both"]:,,
            print()))f"WebNN: {}}}}'Passed' if webnn_success else 'Failed'}")
    
    # Return proper exit code:
    if args.platform == "both":
        return 0 if ()))webgpu_success and webnn_success) else 1:
    elif args.platform == "webgpu":
        return 0 if webgpu_success else 1:
    else:
            return 0 if webnn_success else 1
:
if __name__ == "__main__":
    sys.exit()))asyncio.run()))main())))))