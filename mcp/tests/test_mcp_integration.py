#!/usr/bin/env python3
"""
Test script for the IPFS Accelerate MCP server.

This script tests the MCP server functionality by:
1. Creating an MCP server instance
2. Testing that tools and resources are registered correctly
3. Starting the server in test mode and making a simple request
"""
import sys
import logging
import os
import json
import httpx
import asyncio
from multiprocessing import Process
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_test")

# Add the parent directory to sys.path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the required modules
from ipfs_accelerate_py import ipfs_accelerate_py
from mcp.server import create_mcp_server, get_mcp_server_instance
from mcp.integration import initialize_mcp_server
from fastapi import FastAPI
import uvicorn

def start_server():
    """Start the MCP server in a separate process for testing."""
    # Create IPFS Accelerate instance
    accelerate = ipfs_accelerate_py()
    
    # Create FastAPI app for testing
    app = FastAPI(title="MCP Test App")
    
    # Initialize the MCP server with our app
    initialize_mcp_server(app, accelerate)
    
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8765)

async def test_mcp_server():
    """Test the MCP server by making requests to it."""
    # Start server in a separate process
    server_process = Process(target=start_server)
    server_process.start()
    
    try:
        # Wait for server to start
        logger.info("Waiting for server to start...")
        await asyncio.sleep(3)
        
        # Create a client
        logger.info("Creating client and testing server...")
        base_url = "http://127.0.0.1:8765/mcp"
        
        async with httpx.AsyncClient() as client:
            # Test server info endpoint
            response = await client.get(base_url)
            logger.info(f"Server info status: {response.status_code}")
            if response.status_code == 200:
                server_info = response.json()
                logger.info(f"Server name: {server_info.get('name')}")
                logger.info(f"Server description: {server_info.get('description')}")
            else:
                logger.error(f"Failed to get server info: {response.text}")
            
            # Test health endpoint
            response = await client.get(f"{base_url}/health")
            logger.info(f"Health endpoint status: {response.status_code}")
            if response.status_code == 200:
                health_info = response.json()
                logger.info(f"Health status: {json.dumps(health_info, indent=2)}")
            else:
                logger.error(f"Health endpoint error: {response.text}")
            
            # Test system info resource
            response = await client.get(f"{base_url}/resources/system://info")
            logger.info(f"System info resource status: {response.status_code}")
            if response.status_code == 200:
                system_info = response.json()
                logger.info(f"System platform: {system_info.get('platform')}")
            else:
                logger.error(f"Failed to get system info: {response.text}")
            
            # Test hardware detection tool
            response = await client.post(
                f"{base_url}/tools/detect_hardware",
                json={}
            )
            logger.info(f"Hardware detection tool status: {response.status_code}")
            if response.status_code == 200:
                hardware_info = response.json()
                logger.info(f"Hardware detection result: {json.dumps(hardware_info, indent=2)}")
            else:
                logger.error(f"Failed to call hardware detection tool: {response.text}")
                
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Terminate the server process
        logger.info("Terminating server...")
        server_process.terminate()
        server_process.join()

def run_tests():
    """Run the MCP server tests."""
    logger.info("Starting MCP server tests")
    asyncio.run(test_mcp_server())
    logger.info("MCP server tests completed")

if __name__ == "__main__":
    run_tests()
