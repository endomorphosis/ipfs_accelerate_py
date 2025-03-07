#!/usr/bin/env python3
"""
Simplified Test for WebNN and WebGPU Implementations

This script provides a simple test of WebNN and WebGPU implementations.
It verifies that the basic functionality works correctly.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the implementations
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

async def test_webgpu():
    """Test WebGPU implementation."""
    if not WEBGPU_AVAILABLE:
        logger.error("WebGPU implementation not available")
        return False
    
    logger.info("Testing WebGPU implementation...")
    impl = RealWebGPUImplementation(browser_name="chrome", headless=True)
    
    try:
        # Initialize
        logger.info("Initializing WebGPU implementation")
        success = await impl.initialize()
        if not success:
            logger.error("Failed to initialize WebGPU implementation")
            return False
        
        # Check features
        features = impl.get_feature_support()
        logger.info(f"WebGPU features: {json.dumps(features, indent=2)}")
        
        # Initialize model
        logger.info("Initializing model: bert-base-uncased")
        model_info = await impl.initialize_model("bert-base-uncased", model_type="text")
        if not model_info:
            logger.error("Failed to initialize model")
            await impl.shutdown()
            return False
        
        logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
        
        # Run inference
        logger.info("Running inference")
        result = await impl.run_inference("bert-base-uncased", "This is a test.")
        if not result:
            logger.error("Failed to run inference")
            await impl.shutdown()
            return False
        
        logger.info(f"Inference result: {json.dumps(result, indent=2)}")
        
        # Check if simulation was used
        is_simulation = result.get("is_simulation", True)
        if is_simulation:
            logger.warning("WebGPU is using simulation mode")
        else:
            logger.info("WebGPU is using real hardware acceleration")
        
        # Shutdown
        await impl.shutdown()
        logger.info("WebGPU test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing WebGPU: {e}")
        try:
            await impl.shutdown()
        except:
            pass
        return False

async def test_webnn():
    """Test WebNN implementation."""
    if not WEBNN_AVAILABLE:
        logger.error("WebNN implementation not available")
        return False
    
    logger.info("Testing WebNN implementation...")
    impl = RealWebNNImplementation(browser_name="chrome", headless=True)
    
    try:
        # Initialize
        logger.info("Initializing WebNN implementation")
        success = await impl.initialize()
        if not success:
            logger.error("Failed to initialize WebNN implementation")
            return False
        
        # Check features
        features = impl.get_feature_support()
        logger.info(f"WebNN features: {json.dumps(features, indent=2)}")
        
        # Initialize model
        logger.info("Initializing model: bert-base-uncased")
        model_info = await impl.initialize_model("bert-base-uncased", model_type="text")
        if not model_info:
            logger.error("Failed to initialize model")
            await impl.shutdown()
            return False
        
        logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
        
        # Run inference
        logger.info("Running inference")
        result = await impl.run_inference("bert-base-uncased", "This is a test.")
        if not result:
            logger.error("Failed to run inference")
            await impl.shutdown()
            return False
        
        logger.info(f"Inference result: {json.dumps(result, indent=2)}")
        
        # Check if simulation was used
        is_simulation = result.get("is_simulation", True)
        if is_simulation:
            logger.warning("WebNN is using simulation mode")
        else:
            logger.info("WebNN is using real hardware acceleration")
        
        # Shutdown
        await impl.shutdown()
        logger.info("WebNN test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing WebNN: {e}")
        try:
            await impl.shutdown()
        except:
            pass
        return False

async def main():
    """Run tests."""
    # Test WebGPU
    webgpu_success = await test_webgpu()
    if webgpu_success:
        logger.info("WebGPU test passed")
    else:
        logger.error("WebGPU test failed")
    
    # Test WebNN
    webnn_success = await test_webnn()
    if webnn_success:
        logger.info("WebNN test passed")
    else:
        logger.error("WebNN test failed")
    
    # Overall result
    if webgpu_success and webnn_success:
        logger.info("All tests passed")
        return 0
    else:
        logger.error("Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))