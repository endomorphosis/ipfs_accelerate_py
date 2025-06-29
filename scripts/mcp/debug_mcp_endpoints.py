#!/usr/bin/env python3
"""
Detailed MCP Server Debug Tool

This script starts the MCP server, tests the specific endpoints that are failing,
and provides detailed information about the failures to help with debugging.
"""

import os
import sys
import json
import time
import logging
import subprocess
import requests
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gateway_url(host="localhost", port=8001):
    """Test the ipfs_gateway_url tool specifically."""
    endpoint = f"http://{host}:{port}/mcp/tool/ipfs_gateway_url"
    test_cid = "QmTest123456789"
    
    logger.info(f"Testing ipfs_gateway_url with endpoint: {endpoint}")
    
    try:
        # Test with ipfs_hash parameter
        payload1 = {"ipfs_hash": test_cid}
        logger.info(f"Test 1: Using payload {json.dumps(payload1)}")
        response1 = requests.post(endpoint, json=payload1, timeout=5)
        logger.info(f"Response 1 status: {response1.status_code}")
        logger.info(f"Response 1 body: {response1.text}")
        
        # Test with cid parameter
        payload2 = {"cid": test_cid}
        logger.info(f"Test 2: Using payload {json.dumps(payload2)}")
        response2 = requests.post(endpoint, json=payload2, timeout=5)
        logger.info(f"Response 2 status: {response2.status_code}")
        logger.info(f"Response 2 body: {response2.text}")
        
        return response1.status_code == 200 or response2.status_code == 200
    except Exception as e:
        logger.error(f"Error testing ipfs_gateway_url: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_hardware_info(host="localhost", port=8001):
    """Test the ipfs_get_hardware_info tool specifically."""
    endpoint = f"http://{host}:{port}/mcp/tool/ipfs_get_hardware_info"
    
    logger.info(f"Testing ipfs_get_hardware_info with endpoint: {endpoint}")
    
    try:
        response = requests.post(endpoint, json={}, timeout=5)
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response body: {response.text}")
        
        if response.status_code != 200:
            logger.info("Trying fallback to get_hardware_info tool...")
            fallback_endpoint = f"http://{host}:{port}/mcp/tool/get_hardware_info"
            fallback_response = requests.post(fallback_endpoint, json={}, timeout=5)
            logger.info(f"Fallback response status: {fallback_response.status_code}")
            logger.info(f"Fallback response body: {fallback_response.text}")
            
            if fallback_response.status_code == 200:
                logger.info("get_hardware_info works but ipfs_get_hardware_info doesn't")
            
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error testing ipfs_get_hardware_info: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_all_tools(host="localhost", port=8001):
    """Get a list of all registered tools."""
    endpoint = f"http://{host}:{port}/tools"
    
    logger.info(f"Fetching all registered tools from: {endpoint}")
    
    try:
        response = requests.get(endpoint, timeout=5)
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            tools = response.json()
            logger.info(f"Found {len(tools)} tools:")
            for tool_name in tools:
                logger.info(f"  - {tool_name}")
            return tools
        else:
            logger.error(f"Failed to get tools: {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Error getting tools: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def start_server():
    """Start the unified MCP server."""
    logger.info("Starting unified MCP server...")
    
    # Kill any existing servers
    subprocess.run("pkill -f 'python.*mcp_server.py'", shell=True)
    time.sleep(2)
    
    # Start the server with verbose logging
    cmd = [sys.executable, "unified_mcp_server.py", "--port", "8001", "--verbose"]
    proc = subprocess.Popen(
        cmd,
        stdout=open("unified_mcp_debug.log", "w"),
        stderr=subprocess.STDOUT
    )
    
    # Wait for server to start
    time.sleep(5)
    
    # Check if server is running
    if proc.poll() is not None:
        logger.error("Server failed to start")
        return None
    
    logger.info("Server started")
    return proc

def stop_server(proc):
    """Stop the MCP server."""
    if proc is not None:
        logger.info("Stopping MCP server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)

def main():
    """Main function."""
    logger.info("Starting detailed MCP server debug")
    
    # Start the server
    server_proc = start_server()
    if server_proc is None:
        return 1
    
    try:
        # Get all registered tools
        tools = get_all_tools()
        
        # Check if the required tools are registered
        required_tools = [
            "ipfs_gateway_url",
            "ipfs_get_hardware_info"
        ]
        
        missing_tools = [tool for tool in required_tools if tool not in tools]
        if missing_tools:
            logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            
            # Check if the bridge tools are available
            bridge_tools = ["get_hardware_info"]
            for tool in bridge_tools:
                if tool in tools:
                    logger.info(f"Bridge tool {tool} is registered")
                else:
                    logger.info(f"Bridge tool {tool} is NOT registered")
        else:
            logger.info("All required tools are registered")
        
        # Test problematic endpoints specifically
        gateway_success = test_gateway_url()
        logger.info(f"Gateway URL test result: {'SUCCESS' if gateway_success else 'FAILED'}")
        
        hardware_success = test_hardware_info()
        logger.info(f"Hardware info test result: {'SUCCESS' if hardware_success else 'FAILED'}")
        
    finally:
        # Stop the server
        stop_server(server_proc)
    
    logger.info("Detailed debug completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
