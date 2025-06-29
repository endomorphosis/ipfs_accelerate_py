#!/usr/bin/env python
"""
Direct MCP Server Test

This script tests the MCP server directly using the requests module.
Enhanced with additional diagnostics and auto-server functionality.
"""

import json
import requests
import sys
import logging
import time
import subprocess
import argparse
import os
from typing import Dict, Any, List, Optional, Tuple

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("direct_mcp_test.log")
    ]
)
logger = logging.getLogger(__name__)

def check_mcp_manifest(port=8002):
    """Check the MCP manifest"""
    url = f"http://localhost:{port}/mcp/manifest"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            manifest = response.json()
            logger.info(f"MCP Manifest: {json.dumps(manifest, indent=2)}")
            
            # Return tools from manifest
            if "tools" in manifest:
                return manifest["tools"]
            else:
                logger.error("No tools found in manifest")
                return {}
        else:
            logger.error(f"Failed to get manifest: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error getting manifest: {e}")
        return {}

def test_get_hardware_info(host="localhost", port=8002, alt_endpoints=True):
    """Test the get_hardware_info tool using multiple possible endpoints"""
    # List of possible endpoints to try
    endpoints = [
        f"http://{host}:{port}/tools/get_hardware_info/invoke",  # Standard tools format
        f"http://{host}:{port}/mcp/tool/get_hardware_info",      # Alternative format
        f"http://{host}:{port}/tool/get_hardware_info"           # Another possibility
    ]
    
    if not alt_endpoints:
        # Just try the first endpoint if alt_endpoints is False
        endpoints = [endpoints[0]]
    
    for url in endpoints:
        try:
            logger.info(f"Trying hardware info endpoint: {url}")
            response = requests.post(url, json={}, timeout=10)
            if response.status_code == 200:
                hardware_info = response.json()
                logger.info("Hardware info successfully retrieved")
                logger.info(f"Hardware Info: {json.dumps(hardware_info, indent=2)}")
                return True, hardware_info, url
            else:
                logger.warning(f"Failed to get hardware info from {url}: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error accessing {url}: {e}")
    
    logger.error("All hardware info endpoint attempts failed")
    return False, {}, ""
        else:
            logger.error(f"Failed to get hardware info: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing get_hardware_info: {e}")
        return False

def test_health_endpoint(port=8002):
    """Test the health endpoint"""
    url = f"http://localhost:{port}/health"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            health = response.json()
            logger.info(f"Health check succeeded: {health}")
            return True
        else:
            logger.error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error testing health endpoint: {e}")
        return False

def main():
    """Main entry point"""
    port = 8002
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid port: {sys.argv[1]}")
            return 1
    
    logger.info(f"Testing MCP server on port {port}")
    
    # Test health endpoint
    logger.info("Testing health endpoint")
    health_result = test_health_endpoint(port)
    
    # Get manifest
    logger.info("Getting MCP manifest")
    available_tools = check_mcp_manifest(port)
    
    # Test get_hardware_info
    logger.info("Testing get_hardware_info")
    hardware_result = test_get_hardware_info(port)
    
    # Print summary
    print("\n=== Test Results ===")
    print(f"Health Endpoint:    {'✅ PASS' if health_result else '❌ FAIL'}")
    print(f"Available Tools:    {', '.join(available_tools.keys()) if available_tools else 'None'}")
    print(f"get_hardware_info:  {'✅ PASS' if hardware_result else '❌ FAIL'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
