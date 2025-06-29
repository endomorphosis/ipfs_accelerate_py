#!/usr/bin/env python3
"""
Test MCP Server Functionality

This script tests the direct MCP server functionality to ensure it's
properly exposing the ipfs_accelerate_py package functions as MCP tools.
"""

import argparse
import json
import requests
import sys
import time

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

def test_server_connection(base_url):
    """Test basic server connection."""
    section("Testing Server Connection")
    
    endpoints = {
        "/tools": "Tools endpoint"
    }
    
    all_passed = True
    
    for endpoint, description in endpoints.items():
        info(f"Testing {description} ({endpoint})...")
        try:
            response = requests.get(f"{base_url}{endpoint}")
            if response.status_code == 200:
                success(f"{description} is accessible")
                if endpoint == "/tools":
                    try:
                        tools = response.json().get("tools", [])
                        info(f"Found {len(tools)} tools: {', '.join(tools)}")
                    except:
                        warning("Could not parse tools response")
            else:
                error(f"{description} returned status code {response.status_code}")
                all_passed = False
        except Exception as e:
            error(f"Error accessing {description}: {str(e)}")
            all_passed = False
    
    return all_passed

def test_call_tool(base_url, tool_name, arguments=None):
    """Test calling a specific tool."""
    if arguments is None:
        arguments = {}
    
    info(f"Testing tool: {tool_name}")
    
    try:
        response = requests.post(
            f"{base_url}/call_tool",
            json={"tool_name": tool_name, "arguments": arguments}
        )
        
        if response.status_code == 200:
            try:
                json_response = response.json()
                if isinstance(json_response, dict) and "result" in json_response:
                    result = json_response.get("result", {})
                    if not isinstance(result, dict):
                        info(f"Tool returned non-dict result: {result}")
                        result = {"raw_response": result}
                else:
                    result = {"raw_response": json_response}
                success(f"Successfully called {tool_name}")
                return True, result
            except Exception as e:
                error(f"Error parsing response from {tool_name}: {str(e)}")
                return False, {"error": str(e)}
        else:
            error(f"Tool {tool_name} call failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            return False, {}
    except Exception as e:
        error(f"Error calling tool {tool_name}: {str(e)}")
        return False, {}

def test_ipfs_functionality(base_url):
    """Test IPFS-related functionality."""
    section("Testing IPFS Functionality")
    
    all_passed = True
    
    # Test health check
    success_health, result = test_call_tool(base_url, "health_check")
    if success_health:
        if result.get("ipfs_connected", False):
            success("IPFS daemon is connected")
        else:
            warning("IPFS daemon is not connected")
    
    # Create test content
    test_content = "Hello, IPFS from MCP!"
    test_file_path = "test_file.txt"
    
    # Write test file
    with open(test_file_path, "w") as f:
        f.write(test_content)
    
    # Test ipfs_add_file
    success_add, result_add = test_call_tool(base_url, "ipfs_add_file", {"path": test_file_path})
    if success_add and result_add.get("success", False):
        cid = result_add.get("cid")
        success(f"Successfully added file to IPFS with CID: {cid}")
        
        # Test ipfs_cat
        success_cat, result_cat = test_call_tool(base_url, "ipfs_cat", {"cid": cid})
        if success_cat and result_cat.get("success", False):
            content = result_cat.get("content", "")
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
    
    # Test MFS functionality
    mfs_path = "/mcp-test/test.txt"
    success_write, result_write = test_call_tool(
        base_url, "ipfs_files_write", 
        {"path": mfs_path, "content": test_content}
    )
    
    if success_write and result_write.get("success", False):
        success(f"Successfully wrote to IPFS MFS at {mfs_path}")
        
        # Test reading from MFS
        success_read, result_read = test_call_tool(
            base_url, "ipfs_files_read", 
            {"path": mfs_path}
        )
        
        if success_read and result_read.get("success", False):
            content = result_read.get("content", "")
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

def test_hardware_info(base_url):
    """Test hardware info functionality."""
    section("Testing Hardware Info Functionality")
    
    all_passed = True
    
    # Test get_hardware_info
    success_hw, result_hw = test_call_tool(base_url, "get_hardware_info")
    if success_hw:
        success("Successfully retrieved hardware information")
        info(f"Platform: {result_hw.get('platform', 'N/A')}")
        info(f"Python version: {result_hw.get('python_version', 'N/A')}")
        info(f"CPU count: {result_hw.get('cpu_count', 'N/A')}")
    else:
        error("Failed to retrieve hardware information")
        all_passed = False
    
    # Test get_hardware_capabilities
    success_cap, result_cap = test_call_tool(base_url, "get_hardware_capabilities")
    if success_cap:
        success("Successfully retrieved hardware capabilities")
        if "cpu" in result_cap:
            info(f"CPU: {result_cap.get('cpu', {}).get('name', 'N/A')}")
        if "gpu" in result_cap:
            info(f"GPU available: {result_cap.get('gpu', {}).get('available', False)}")
    else:
        error("Failed to retrieve hardware capabilities")
        all_passed = False
    
    return all_passed

def test_model_functionality(base_url):
    """Test model functionality."""
    section("Testing Model Functionality")
    
    all_passed = True
    
    # Test list_models
    success_list, result_list = test_call_tool(base_url, "list_models")
    if not success_list:
        error("Failed to list models")
        return False
    
    # Handle different response formats
    if isinstance(result_list, dict):
        models_count = result_list.get("count", 0)
        if isinstance(models_count, str):
            try:
                models_count = int(models_count)
            except ValueError:
                models_count = 0
        
        success(f"Successfully listed {models_count} models")
        
        # If models are available, try to create an endpoint
        models = result_list.get("models", {})
        if not isinstance(models, dict):
            info(f"Models not returned as a dictionary: {models}")
            models = {}
            
        model_names = list(models.keys()) if models else []
        
        if model_names:
            model_name = model_names[0]
            info(f"Testing endpoint creation with model: {model_name}")
            
            success_create, result_create = test_call_tool(
                base_url, "create_endpoint", 
                {"model_name": model_name}
            )
            
            if success_create and result_create.get("success", False):
                endpoint_id = result_create.get("endpoint_id")
                success(f"Successfully created endpoint with ID: {endpoint_id}")
                
                # Try running inference
                success_infer, result_infer = test_call_tool(
                    base_url, "run_inference", 
                    {
                        "endpoint_id": endpoint_id,
                        "inputs": ["Test input for inference"]
                    }
                )
                
                if success_infer and result_infer.get("success", False):
                    success("Successfully ran inference")
                else:
                    error("Failed to run inference")
                    all_passed = False
            else:
                error("Failed to create endpoint")
                all_passed = False
    else:
        info(f"List models returned non-dictionary result: {result_list}")
        success(f"Successfully listed models (format unknown)")
    
    return all_passed

def print_summary(results):
    """Print test results summary."""
    section("Test Results Summary")
    
    for category, passed in results.items():
        if passed:
            print(f"{category}: {GREEN}PASSED{RESET}")
        else:
            print(f"{category}: {RED}FAILED{RESET}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n{GREEN}All tests passed!{RESET}")
    else:
        print(f"\n{RED}Some tests failed.{RESET}")

def main():
    parser = argparse.ArgumentParser(description="Test MCP Server Functionality")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL of the MCP server")
    args = parser.parse_args()
    
    base_url = args.url
    print(f"Testing MCP server at {base_url}\n")
    
    # Run tests
    results = {}
    
    # Test server connection
    results["Connection"] = test_server_connection(base_url)
    
    # Only proceed with other tests if connection succeeded
    if results["Connection"]:
        # Test hardware info
        results["Hardware"] = test_hardware_info(base_url)
        
        # Test IPFS functionality
        results["IPFS"] = test_ipfs_functionality(base_url)
        
        # Test model functionality
        results["Model"] = test_model_functionality(base_url)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main()
