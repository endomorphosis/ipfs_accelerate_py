#!/usr/bin/env python3
"""
Integration Test for IPFS Accelerate MCP with FastAPI

This script tests the integration between IPFS Accelerate's MCP server and FastAPI.
"""

import os
import sys
import json
import time
import argparse
import logging
import requests
import uvicorn
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_mcp_fastapi_test")

def check_endpoint(url: str, path: str = "") -> Dict[str, Any]:
    """
    Check if an endpoint is responding
    
    Args:
        url: Base URL to check
        path: Optional path to append
        
    Returns:
        Response data or error info
    """
    full_url = f"{url}/{path}" if path else url
    logger.info(f"Checking endpoint: {full_url}")
    
    try:
        response = requests.get(full_url, timeout=5)
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "code": response.status_code, "text": response.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP FastAPI Integration Test")
    parser.add_argument("--host", default="http://localhost:9999", help="FastAPI server URL")
    parser.add_argument("--mcp-path", default="/mcp", help="MCP path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print test info
    print("\n" + "="*70)
    print("IPFS Accelerate MCP FastAPI Integration Test")
    print("="*70)
    print(f"Host: {args.host}")
    print(f"MCP Path: {args.mcp_path}")
    print("="*70 + "\n")
    
    # Test main API
    print("\n1. Testing Main API...")
    main_result = check_endpoint(args.host)
    if main_result["status"] == "success":
        print("✅ Main API is responding")
    else:
        print(f"❌ Main API error: {main_result.get('error', main_result)}")
    
    # Test MCP API
    print("\n2. Testing MCP API...")
    mcp_url = f"{args.host}{args.mcp_path}"
    mcp_result = check_endpoint(mcp_url)
    if mcp_result["status"] == "success":
        print("✅ MCP API is responding")
        server_info = mcp_result["data"]
        print(f"   Server name: {server_info.get('name')}")
        print(f"   Description: {server_info.get('description')}")
    else:
        print(f"❌ MCP API error: {mcp_result.get('error', mcp_result)}")
    
    # Test MCP health endpoint
    print("\n3. Testing MCP Health endpoint...")
    health_result = check_endpoint(mcp_url, "health")
    if health_result["status"] == "success":
        print("✅ MCP Health endpoint is responding")
        health_info = health_result["data"]
        print(f"   Status: {health_info.get('status')}")
        print(f"   Tools count: {health_info.get('tools_count')}")
        print(f"   Resources count: {health_info.get('resources_count')}")
    else:
        print(f"❌ MCP Health endpoint error: {health_result.get('error', health_result)}")
    
    # Test MCP tools listing
    print("\n4. Testing MCP Tools listing...")
    tools_result = check_endpoint(mcp_url, "tools")
    if tools_result["status"] == "success":
        print("✅ MCP Tools endpoint is responding")
        tools = tools_result["data"]
        print(f"   Found {len(tools)} tools:")
        for i, tool_name in enumerate(tools):
            print(f"   {i+1}. {tool_name}")
    else:
        print(f"❌ MCP Tools endpoint error: {tools_result.get('error', tools_result)}")
    
    # Test MCP resources listing
    print("\n5. Testing MCP Resources listing...")
    resources_result = check_endpoint(mcp_url, "resources")
    if resources_result["status"] == "success":
        print("✅ MCP Resources endpoint is responding")
        resources = resources_result["data"]
        print(f"   Found {len(resources)} resources:")
        for i, resource_name in enumerate(resources):
            print(f"   {i+1}. {resource_name}")
    else:
        print(f"❌ MCP Resources endpoint error: {resources_result.get('error', resources_result)}")
    
    # Test calling a tool (test_hardware)
    print("\n6. Testing test_hardware tool...")
    try:
        response = requests.post(
            f"{mcp_url}/tools/test_hardware",
            json={},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ test_hardware tool executed successfully")
            print(f"   Available accelerators: {result.get('available_accelerators', [])}")
            
            # Check CUDA
            cuda_available = result.get("cuda_available", False)
            print(f"   CUDA {'✅ Available' if cuda_available else '❌ Not available'}")
            
            # Check OpenVINO
            openvino_available = result.get("openvino_available", False)
            print(f"   OpenVINO {'✅ Available' if openvino_available else '❌ Not available'}")
        else:
            print(f"❌ test_hardware tool error: HTTP {response.status_code}")
            print(f"   {response.text}")
    except Exception as e:
        print(f"❌ test_hardware tool error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    if (main_result["status"] == "success" and 
        mcp_result["status"] == "success" and 
        health_result["status"] == "success"):
        print("✅ MCP is successfully integrated with FastAPI")
    else:
        print("❌ Integration test failed - Check the errors above")
    print("="*70)

if __name__ == "__main__":
    main()
