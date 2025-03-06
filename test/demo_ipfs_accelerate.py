#!/usr/bin/env python
"""
Demo script for IPFS Accelerate Python

This script demonstrates the basic functionality of the IPFS Accelerate Python package.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ipfs_demo")

# Import the package
try:
    import ipfs_accelerate_py
    logger.info(f"Successfully imported ipfs_accelerate_py version {ipfs_accelerate_py.__version__}")
except ImportError as e:
    logger.error(f"Error importing ipfs_accelerate_py: {e}")
    logger.info("Make sure ipfs_accelerate_py.py and ipfs_accelerate_impl.py are in the same directory")
    sys.exit(1)

def demo_config():
    """Demonstrate configuration functionality"""
    logger.info("\n=== Configuration Demo ===")
    
    # Create a config instance
    config = ipfs_accelerate_py.config()
    logger.info("Created config instance")
    
    # Get some configuration values
    debug = config.get("general", "debug", False)
    log_level = config.get("general", "log_level", "WARNING")
    cache_enabled = config.get("cache", "enabled", False)
    cache_path = config.get("cache", "path", "./default_cache")
    
    logger.info(f"Config values:")
    logger.info(f"  debug: {debug}")
    logger.info(f"  log_level: {log_level}")
    logger.info(f"  cache_enabled: {cache_enabled}")
    logger.info(f"  cache_path: {cache_path}")
    
    # Set a configuration value
    logger.info("Setting new config value: cache.max_size_mb = 2000")
    config.set("cache", "max_size_mb", 2000)
    
    # Get the updated value
    max_size_mb = config.get("cache", "max_size_mb", 500)
    logger.info(f"Updated max_size_mb: {max_size_mb}")
    
    return "Config demo completed"

def demo_backends():
    """Demonstrate backends functionality"""
    logger.info("\n=== Backends Demo ===")
    
    # Create a backends instance
    backends = ipfs_accelerate_py.backends()
    logger.info("Created backends instance")
    
    # Start a simulated container
    logger.info("Starting a simulated container...")
    container_result = backends.start_container(
        container_name="ipfs-node",
        image="ipfs/kubo:latest",
        ports={"4001": "4001", "5001": "5001", "8080": "8080"}
    )
    logger.info(f"Container result: {container_result}")
    
    # Create a tunnel to the container
    logger.info("Creating a tunnel to the container...")
    tunnel_result = backends.docker_tunnel(
        container_name="ipfs-node",
        local_port=5001,
        container_port=5001
    )
    logger.info(f"Tunnel result: {tunnel_result}")
    
    # List containers
    logger.info("Listing containers...")
    containers = backends.list_containers()
    logger.info(f"Containers: {containers}")
    
    # Get container status
    logger.info("Getting container status...")
    status = backends.get_container_status("ipfs-node")
    logger.info(f"Container status: {status}")
    
    # List marketplace images
    logger.info("Listing marketplace images...")
    marketplace = backends.marketplace()
    logger.info(f"Marketplace images: {marketplace}")
    
    # Stop the container
    logger.info("Stopping the container...")
    stop_result = backends.stop_container("ipfs-node")
    logger.info(f"Stop result: {stop_result}")
    
    return "Backends demo completed"

def demo_ipfs_accelerate():
    """Demonstrate IPFS accelerate functionality"""
    logger.info("\n=== IPFS Accelerate Demo ===")
    
    # Create a text file to add to IPFS
    test_file = Path("./test_file.txt")
    with open(test_file, "w") as f:
        f.write("Hello, IPFS!")
    logger.info(f"Created test file: {test_file}")
    
    # Add the file to IPFS
    logger.info("Adding file to IPFS...")
    add_result = ipfs_accelerate_py.ipfs_accelerate.add_file(str(test_file))
    logger.info(f"Add result: {add_result}")
    
    # Get the CID
    cid = add_result.get("cid")
    logger.info(f"File CID: {cid}")
    
    # Get the file from IPFS
    logger.info("Getting file from IPFS...")
    output_file = Path("./test_file_from_ipfs.txt")
    get_result = ipfs_accelerate_py.ipfs_accelerate.get_file(cid, str(output_file))
    logger.info(f"Get result: {get_result}")
    
    # Read the content of the file
    with open(output_file, "r") as f:
        content = f.read()
    logger.info(f"File content: {content}")
    
    # Load a checkpoint and dispatch
    logger.info("Loading checkpoint and dispatching...")
    checkpoint_result = ipfs_accelerate_py.load_checkpoint_and_dispatch(cid)
    logger.info(f"Checkpoint result: {checkpoint_result}")
    
    # Clean up the test files
    test_file.unlink(missing_ok=True)
    output_file.unlink(missing_ok=True)
    logger.info("Cleaned up test files")
    
    return "IPFS Accelerate demo completed"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Demo for IPFS Accelerate Python")
    parser.add_argument("--config", "-c", action="store_true", help="Run config demo")
    parser.add_argument("--backends", "-b", action="store_true", help="Run backends demo")
    parser.add_argument("--ipfs", "-i", action="store_true", help="Run IPFS accelerate demo")
    parser.add_argument("--all", "-a", action="store_true", help="Run all demos")
    args = parser.parse_args()
    
    # Get system information
    sys_info = ipfs_accelerate_py.get_system_info()
    logger.info("System Information:")
    for key, value in sys_info.items():
        logger.info(f"  {key}: {value}")
    
    # Default to all if no specific demos are specified
    if not (args.config or args.backends or args.ipfs):
        args.all = True
    
    # Run the requested demos
    results = {}
    
    if args.config or args.all:
        results["config"] = demo_config()
    
    if args.backends or args.all:
        results["backends"] = demo_backends()
    
    if args.ipfs or args.all:
        results["ipfs"] = demo_ipfs_accelerate()
    
    # Print summary
    logger.info("\n=== Demo Summary ===")
    for component, result in results.items():
        logger.info(f"{component}: {result}")
    
    logger.info("Demo completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())