#!/usr/bin/env python3
"""
IPFS Accelerate Python - Example Usage

This example demonstrates how to use the IPFS Accelerate Python framework
for hardware-accelerated machine learning inference with automatic failover
to IPFS network-based inference when local hardware is insufficient.
"""

import asyncio
import json
import logging
from ipfs_accelerate_py import ipfs_accelerate_py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('example')

async def run_example():
    """Run the example asynchronously."""
    logger.info("Initializing IPFS Accelerate Python framework")
    
    # Initialize the framework
    framework = ipfs_accelerate_py()
    
    # Create some example models and resources
    models = ["bert-base-uncased", "gpt2-small", "vit-base-patch16-224"]
    resources = {
        "endpoints": {
            "local_endpoints": {
                "bert-base-uncased": [
                    ["bert-base-uncased", "cpu:0", 32768],
                    ["bert-base-uncased", "cuda:0", 32768] if framework.hardware_detection.detect_hardware().get("cuda", {}).get("available", False) else None,
                ],
                "gpt2-small": [
                    ["gpt2-small", "cpu:0", 32768],
                    ["gpt2-small", "cuda:0", 32768] if framework.hardware_detection.detect_hardware().get("cuda", {}).get("available", False) else None,
                ],
                "vit-base-patch16-224": [
                    ["vit-base-patch16-224", "cpu:0", 32768],
                    ["vit-base-patch16-224", "cuda:0", 32768] if framework.hardware_detection.detect_hardware().get("cuda", {}).get("available", False) else None,
                ]
            }
        }
    }
    
    # Filter out None values in resources
    for model in resources["endpoints"]["local_endpoints"]:
        resources["endpoints"]["local_endpoints"][model] = [
            endpoint for endpoint in resources["endpoints"]["local_endpoints"][model] if endpoint is not None
        ]
    
    # Initialize endpoints
    await framework.init_endpoints(models, resources)
    
    # Example 1: Use local inference with automatic hardware selection
    logger.info("\nExample 1: Local inference with automatic hardware selection")
    try:
        result = await framework.process_async("bert-base-uncased", "This is a test sentence.")
        logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Error in Example 1: {e}")
    
    # Example 2: Use local inference with explicit hardware selection
    logger.info("\nExample 2: Local inference with explicit hardware selection")
    try:
        result = await framework.process_async("bert-base-uncased", "This is a test sentence.", "cpu:0")
        logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Error in Example 2: {e}")
    
    # Example 3: Use the accelerate_inference method for automatic fallback to IPFS
    logger.info("\nExample 3: Accelerate inference with automatic IPFS fallback")
    try:
        # Use a model that might not be available locally
        result = await framework.accelerate_inference("llava-1.5-7b", "Describe this image", use_ipfs=True)
        logger.info(f"Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        logger.error(f"Error in Example 3: {e}")
    
    # Example 4: Direct IPFS operations
    logger.info("\nExample 4: Direct IPFS operations")
    try:
        # Store data to IPFS
        cid = await framework.store_to_ipfs(b"This is test data")
        logger.info(f"Stored data to IPFS, CID: {cid}")
        
        # Retrieve data from IPFS
        data = await framework.query_ipfs(cid)
        logger.info(f"Retrieved data from IPFS: {data.decode()}")
    except Exception as e:
        logger.error(f"Error in Example 4: {e}")
    
    # Example 5: Find and connect to IPFS providers
    logger.info("\nExample 5: Find and connect to IPFS providers")
    try:
        # Find providers for a model
        providers = await framework.find_providers("bert-base-uncased")
        logger.info(f"Found providers: {providers}")
        
        # Connect to the first provider
        if providers:
            connected = await framework.connect_to_provider(providers[0])
            logger.info(f"Connected to provider {providers[0]}: {connected}")
    except Exception as e:
        logger.error(f"Error in Example 5: {e}")
    
    logger.info("\nExample complete")

def main():
    """Run the example."""
    asyncio.run(run_example())

if __name__ == "__main__":
    main()