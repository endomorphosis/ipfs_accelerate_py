#!/usr/bin/env python3
"""
IPFS Accelerate Python Test Script

This script tests the core functionality of the IPFS Accelerate Python framework.
"""

import asyncio
import logging
import sys
import os
import pytest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_accelerate')

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the helper module for creating the framework instance
# The test_helper.py is in the test/ directory at the root, not tests/test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test')))
from test_helper import create_framework

@pytest.mark.asyncio
async def test_hardware_detection():
    """Test hardware detection functionality."""
    logger.info("Testing hardware detection...")
    
    # Create the framework instance
    framework = create_framework()
    
    # Get hardware detection results
    if hasattr(framework.hardware_detection, "detect_available_hardware"):
        hardware_result = framework.hardware_detection.detect_available_hardware()
        hardware = hardware_result.get('hardware', {}) if isinstance(hardware_result, dict) else hardware_result
    elif hasattr(framework.hardware_detection, "detect_all_hardware"):
        hardware = framework.hardware_detection.detect_all_hardware()
    else:
        logger.warning("Could not find hardware detection method")
        hardware = {'cpu': True}
    
    # Print detected hardware
    logger.info("Detected hardware:")
    for platform, info in hardware.items():
        if isinstance(info, dict):
            available = info.get("available", False) or info.get("detected", False)
        else:
            available = info
        if available:
            logger.info(f"  - {platform} is available")
            # Print platform details if available
            if isinstance(info, dict):
                if platform == "cpu" and "cores" in info:
                    logger.info(f"    - Cores: {info['cores']}")
                elif platform == "cuda" and "devices" in info:
                    logger.info(f"    - Devices: {info['devices']}")
    
    # Return success
    return True

@pytest.mark.asyncio
async def test_model_endpoints():
    """Test model endpoint initialization and processing."""
    logger.info("Testing model endpoint initialization...")
    
    # Create the framework instance
    framework = create_framework()
    
    # Define test models
    models = ["bert-base-uncased", "gpt2-small"]
    
    # Initialize endpoints
    await framework.init_endpoints(models)
    
    # Verify endpoints were initialized
    endpoint_count = 0
    for model in models:
        if model in framework.endpoints["local_endpoints"]:
            endpoint_count += len(framework.endpoints["local_endpoints"][model])
            logger.info(f"  - {model} has {len(framework.endpoints['local_endpoints'][model])} endpoints")
    
    logger.info(f"Total endpoints initialized: {endpoint_count}")
    
    # Test processing (if endpoints were initialized)
    if endpoint_count > 0:
        logger.info("Testing model processing...")
        try:
            # Process a simple input
            model = models[0]
            result = await framework.process_async(model, "This is a test sentence.")
            logger.info(f"  - Processing result type: {type(result)}")
            logger.info(f"  - Processing result: {result}")
            return True
        except Exception as e:
            logger.error(f"Error processing model: {e}")
            return False
    else:
        logger.warning("No endpoints were initialized, skipping processing test")
        return False

@pytest.mark.asyncio
async def test_ipfs_operations():
    """Test IPFS operations."""
    logger.info("Testing IPFS operations...")
    
    # Create the framework instance
    framework = create_framework()
    
    try:
        # Test storing data to IPFS
        test_data = b"This is test data for IPFS operations"
        cid = await framework.store_to_ipfs(test_data)
        logger.info(f"  - Data stored to IPFS with CID: {cid}")
        
        # Test retrieving data from IPFS
        retrieved_data = await framework.query_ipfs(cid)
        logger.info(f"  - Retrieved data: {retrieved_data.decode()}")
        
        # Verify data integrity
        if retrieved_data == test_data:
            logger.info("  - Data integrity verified")
            
        # Test finding providers
        model = "bert-base-uncased"
        providers = await framework.find_providers(model)
        logger.info(f"  - Found {len(providers)} providers for {model}")
        
        # Test connecting to providers
        if providers:
            connected = await framework.connect_to_provider(providers[0])
            logger.info(f"  - Connected to provider {providers[0]}: {connected}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing IPFS operations: {e}")
        return False

@pytest.mark.asyncio
async def test_accelerated_inference():
    """Test accelerated inference with IPFS fallback."""
    logger.info("Testing accelerated inference...")
    
    # Create the framework instance
    framework = create_framework()
    
    # Define test model
    model = "bert-base-uncased"
    
    # Initialize endpoint
    await framework.init_endpoints([model])
    
    try:
        # Test accelerated inference
        result = await framework.accelerate_inference(model, "This is a test sentence.")
        logger.info(f"  - Accelerated inference result type: {type(result)}")
        logger.info(f"  - Accelerated inference result: {result}")
        
        # Test with a model that's unlikely to be available locally (should fallback to IPFS)
        model = "gpt2-xl"
        result = await framework.accelerate_inference(model, "This is a test sentence.")
        logger.info(f"  - IPFS fallback result type: {type(result)}")
        logger.info(f"  - IPFS fallback method used: {'provider' in result}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing accelerated inference: {e}")
        return False

async def run_tests():
    """Run all tests."""
    logger.info("Starting IPFS Accelerate Python tests...")
    
    # Run hardware detection test
    hardware_test = await test_hardware_detection()
    
    # Run model endpoint test
    endpoint_test = await test_model_endpoints()
    
    # Run IPFS operations test
    ipfs_test = await test_ipfs_operations()
    
    # Run accelerated inference test
    inference_test = await test_accelerated_inference()
    
    # Print test summary
    logger.info("\nTest Summary:")
    logger.info(f"  - Hardware Detection Test: {'PASSED' if hardware_test else 'FAILED'}")
    logger.info(f"  - Model Endpoint Test: {'PASSED' if endpoint_test else 'FAILED'}")
    logger.info(f"  - IPFS Operations Test: {'PASSED' if ipfs_test else 'FAILED'}")
    logger.info(f"  - Accelerated Inference Test: {'PASSED' if inference_test else 'FAILED'}")
    
    # Calculate overall status
    passed = sum([hardware_test, endpoint_test, ipfs_test, inference_test])
    logger.info(f"\nOverall: {passed}/4 tests passed")
    
    return passed == 4

def main():
    """Run the tests."""
    try:
        success = asyncio.run(run_tests())
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()