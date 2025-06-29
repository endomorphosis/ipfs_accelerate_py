#!/usr/bin/env python3
"""
Test Enhanced MCP Server Functionality

This script tests the enhanced MCP server to verify it's working properly
and implements the MCP protocol correctly.
"""

import argparse
import json
import requests
import sys
import time
from typing import Dict, Any, List, Tuple

# ANSI color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def info(message):
    """Print an info message."""
    print(f"{BLUE}ℹ{RESET} {message}")

def success(message):
    """Print a success message."""
    print(f"{GREEN}✓{RESET} {message}")

def warning(message):
    """Print a warning message."""
    print(f"{YELLOW}!{RESET} {message}")

def error(message):
    """Print an error message."""
    print(f"{RED}✗{RESET} {message}")

def section(title):
    """Print a section title."""
    print(f"\n{BLUE}=== {title} ==={RESET}\n")

def test_endpoint(base_url: str, endpoint: str, method: str = "GET", data: Dict = None) -> Tuple[bool, Dict]:
    """Test an HTTP endpoint."""
    url = f"{base_url}{endpoint}"
    info(f"Testing {method} {url}...")
    
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            error(f"Unsupported method: {method}")
            return False, {}
        
        if response.status_code == 200:
            success(f"{method} {endpoint} returned status code 200")
            try:
                return True, response.json()
            except:
                warning(f"Response is not valid JSON: {response.text}")
                return True, {"raw": response.text}
        else:
            error(f"{method} {endpoint} returned status code {response.status_code}")
            return False, {}
    except Exception as e:
        error(f"Error testing {method} {endpoint}: {str(e)}")
        return False, {}

def test_standard_endpoints(base_url: str) -> bool:
    """Test the standard MCP endpoints."""
    section("Standard MCP Endpoints")
    
    all_passed = True
    
    # Test /mcp/manifest
    manifest_success, manifest_data = test_endpoint(base_url, "/mcp/manifest")
    if manifest_success:
        if "tools" in manifest_data:
            tool_count = len(manifest_data["tools"])
            success(f"Manifest contains {tool_count} tools")
            info(f"Available tools: {', '.join(manifest_data['tools'].keys())}")
        else:
            warning("Manifest does not contain 'tools' key")
    else:
        all_passed = False
    
    # Test /tools
    tools_success, tools_data = test_endpoint(base_url, "/tools")
    if tools_success:
        if "tools" in tools_data and isinstance(tools_data["tools"], list):
            tool_count = len(tools_data["tools"])
            success(f"Tools endpoint returned {tool_count} tools")
            info(f"Available tools: {', '.join(tools_data['tools'])}")
        else:
            warning("Tools endpoint response does not match expected format")
    else:
        all_passed = False
    
    return all_passed

def test_tool_call(base_url: str, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Dict]:
    """Test calling a tool via HTTP."""
    info(f"Testing tool call: {tool_name}")
    data = {
        "tool_name": tool_name,
        "arguments": arguments
    }
    
    success_status, result = test_endpoint(base_url, "/call_tool", "POST", data)
    if success_status:
        if "result" in result:
            tool_result = result["result"]
            success(f"Tool call successful: {json.dumps(tool_result, indent=2)}")
            return True, tool_result
        else:
            error(f"Tool call response missing 'result' key: {result}")
            return False, {}
    else:
        return False, {}

def test_ipfs_functionality(base_url: str) -> bool:
    """Test IPFS functionality."""
    section("IPFS Functionality")
    all_passed = True
    
    # Test health check
    health_success, health_result = test_tool_call(base_url, "health_check", {})
    if not health_success:
        all_passed = False
    
    # Test add file
    test_content = "Hello, IPFS from MCP!"
    test_file_path = "test_file.txt"
    
    # Write test file
    with open(test_file_path, "w") as f:
        f.write(test_content)
    
    add_success, add_result = test_tool_call(base_url, "ipfs_add_file", {"path": test_file_path})
    if add_success and "cid" in add_result:
        cid = add_result["cid"]
        success(f"Successfully added file to IPFS with CID: {cid}")
        
        # Test cat
        cat_success, cat_result = test_tool_call(base_url, "ipfs_cat", {"cid": cid})
        if cat_success and "content" in cat_result:
            content = cat_result["content"]
            if content == test_content:
                success("Successfully retrieved correct content from IPFS")
            else:
                error(f"Retrieved content does not match original: {content}")
                all_passed = False
        else:
            error("Failed to retrieve content from IPFS")
            all_passed = False
    else:
        error("Failed to add file to IPFS")
        all_passed = False
    
    # Test write and read from MFS
    mfs_path = "/mcp-test/test.txt"
    write_success, write_result = test_tool_call(
        base_url, "ipfs_files_write", 
        {"path": mfs_path, "content": test_content}
    )
    
    if write_success and "success" in write_result and write_result["success"]:
        success(f"Successfully wrote to IPFS MFS at {mfs_path}")
        
        read_success, read_result = test_tool_call(
            base_url, "ipfs_files_read", 
            {"path": mfs_path}
        )
        
        if read_success and "content" in read_result:
            content = read_result["content"]
            if content == test_content:
                success("Successfully read correct content from IPFS MFS")
            else:
                error(f"Read content does not match written content: {content}")
                all_passed = False
        else:
            error("Failed to read from IPFS MFS")
            all_passed = False
    else:
        error("Failed to write to IPFS MFS")
        all_passed = False
    
    return all_passed

def test_model_functionality(base_url: str) -> bool:
    """Test model functionality."""
    section("Model Functionality")
    all_passed = True
    
    # Test list models
    list_success, list_result = test_tool_call(base_url, "list_models", {})
    if not list_success:
        return False
    
    # Test create endpoint
    model_name = None
    if "models" in list_result and isinstance(list_result["models"], dict):
        model_names = list(list_result["models"].keys())
        if model_names:
            model_name = model_names[0]
    
    if not model_name:
        warning("No models available for testing endpoint creation")
        return True
    
    create_success, create_result = test_tool_call(
        base_url, "create_endpoint", 
        {"model_name": model_name}
    )
    
    if create_success and "endpoint_id" in create_result:
        endpoint_id = create_result["endpoint_id"]
        success(f"Successfully created endpoint for model {model_name} with ID: {endpoint_id}")
        
        # Test inference
        infer_success, infer_result = test_tool_call(
            base_url, "run_inference", 
            {
                "endpoint_id": endpoint_id,
                "inputs": ["This is a test input for inference"]
            }
        )
        
        if infer_success and "success" in infer_result and infer_result["success"]:
            success("Successfully ran inference")
        else:
            error("Failed to run inference")
            all_passed = False
    else:
        error(f"Failed to create endpoint for model {model_name}")
        all_passed = False
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Test Enhanced MCP Server")
    parser.add_argument("--url", default="http://localhost:8002", help="Base URL of the enhanced MCP server")
    parser.add_argument("--skip-ipfs", action="store_true", help="Skip IPFS functionality tests")
    parser.add_argument("--skip-model", action="store_true", help="Skip model functionality tests")
    
    args = parser.parse_args()
    
    base_url = args.url
    print(f"Testing enhanced MCP server at {base_url}")
    
    results = {}
    
    # Test standard endpoints
    results["Standard Endpoints"] = test_standard_endpoints(base_url)
    
    # Test IPFS functionality
    if not args.skip_ipfs:
        results["IPFS Functionality"] = test_ipfs_functionality(base_url)
    
    # Test model functionality
    if not args.skip_model:
        results["Model Functionality"] = test_model_functionality(base_url)
    
    # Print summary
    section("Test Results Summary")
    
    all_passed = True
    for category, passed in results.items():
        if passed:
            print(f"{category}: {GREEN}PASSED{RESET}")
        else:
            print(f"{category}: {RED}FAILED{RESET}")
            all_passed = False
    
    if all_passed:
        print(f"\n{GREEN}All tests passed!{RESET}")
        print("\nThe enhanced MCP server is functioning correctly and implements the MCP protocol properly.")
        print("You should now be able to connect Claude to this server.")
    else:
        print(f"\n{RED}Some tests failed.{RESET}")
        print("\nThe enhanced MCP server may not be functioning correctly.")
        print("Please check the logs for more information.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
