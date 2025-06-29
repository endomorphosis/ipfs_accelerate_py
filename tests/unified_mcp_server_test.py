#!/usr/bin/env python3
"""
Unified MCP Server Test Script

This script tests the functionality of the unified MCP server, including:
- Server connection
- Basic server endpoints
- IPFS functionality
- Model server capabilities
- API multiplexer
- Task management
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable

import requests

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Test constants
TEST_DIR = "mcp_test_files"
TEST_TEXT_FILE = f"{TEST_DIR}/test_text.txt"
TEST_JSON_FILE = f"{TEST_DIR}/test_json.json"
TEST_MULTILINE_FILE = f"{TEST_DIR}/test_multiline.txt"
TEST_TEXT_CONTENT = "Hello, IPFS Accelerate MCP!"
TEST_JSON_CONTENT = {"message": "Hello IPFS", "test": True, "value": 42}
TEST_MULTILINE_CONTENT = """Line 1: IPFS Accelerate Test
Line 2: This is a multiline test file
Line 3: Testing MCP functionality
Line 4: End of test file"""

# Global variables
server_url = "http://localhost:8001"
test_results = {}
created_resources = []

def info(message: str) -> None:
    """Print an info message."""
    print(f"{BLUE}ℹ{RESET} {message}")

def success(message: str) -> None:
    """Print a success message."""
    print(f"{GREEN}✓{RESET} {message}")

def warning(message: str) -> None:
    """Print a warning message."""
    print(f"{YELLOW}!{RESET} {message}")

def error(message: str) -> None:
    """Print an error message."""
    print(f"{RED}✗{RESET} {message}")

def section(title: str) -> None:
    """Print a section title."""
    print(f"\n{BLUE}=== {title} ==={RESET}\n")

def setup_test_environment() -> bool:
    """Set up the test environment."""
    section("Setting up test environment")
    
    try:
        # Create test directory if it doesn't exist
        if not os.path.exists(TEST_DIR):
            os.makedirs(TEST_DIR)
            info(f"Created test directory: {TEST_DIR}")
        
        # Create test files
        with open(TEST_TEXT_FILE, "w") as f:
            f.write(TEST_TEXT_CONTENT)
            info(f"Created test file: {TEST_TEXT_FILE}")
        
        with open(TEST_JSON_FILE, "w") as f:
            json.dump(TEST_JSON_CONTENT, f, indent=2)
            info(f"Created test file: {TEST_JSON_FILE}")
        
        with open(TEST_MULTILINE_FILE, "w") as f:
            f.write(TEST_MULTILINE_CONTENT)
            info(f"Created test file: {TEST_MULTILINE_FILE}")
        
        success("Test environment setup complete")
        return True
    except Exception as e:
        error(f"Failed to set up test environment: {str(e)}")
        return False

def cleanup_test_environment() -> None:
    """Clean up the test environment."""
    section("Cleaning up test environment")
    
    try:
        # Remove test directory
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
            info(f"Removed test directory: {TEST_DIR}")
        
        # Clean up any other created resources
        for resource in created_resources:
            if os.path.exists(resource):
                if os.path.isdir(resource):
                    shutil.rmtree(resource)
                else:
                    os.remove(resource)
                info(f"Removed created resource: {resource}")
        
        success("Clean-up complete")
    except Exception as e:
        error(f"Failed to clean up test environment: {str(e)}")

def test_server_connection() -> bool:
    """Test basic server connection and endpoints."""
    section("Testing Server Connection")
    
    all_passed = True
    endpoints_to_test = [
        "/",
        "/sse",
        "/tools",
        "/mcp/manifest"
    ]
    
    info("Testing basic server connectivity...")
    try:
        response = requests.get(f"{server_url}/")
        if response.status_code == 200:
            success(f"Connected to server at {server_url}")
            info(f"Server response: {response.json()}")
            test_results["server_connection"] = True
        else:
            error(f"Request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["server_connection"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Connection error: {str(e)}")
        test_results["server_connection"] = False
        all_passed = False
    
    if not test_results.get("server_connection", False):
        error("Server is not accessible")
        error("Server connection failed, cannot proceed with further tests")
        return False
    
    # Test essential endpoints
    for endpoint in endpoints_to_test:
        info(f"Testing endpoint: {endpoint}")
        try:
            response = requests.get(f"{server_url}{endpoint}")
            if response.status_code == 200:
                success(f"Endpoint {endpoint} is accessible")
                test_results[f"endpoint_{endpoint}"] = True
            else:
                error(f"Endpoint {endpoint} request failed with status code: {response.status_code}")
                info(f"Response: {response.text}")
                test_results[f"endpoint_{endpoint}"] = False
                all_passed = False
        except requests.exceptions.RequestException as e:
            error(f"Error accessing endpoint {endpoint}: {str(e)}")
            test_results[f"endpoint_{endpoint}"] = False
            all_passed = False
    
    if all_passed:
        success("All required endpoints are accessible")
    else:
        warning("Some required endpoints are not accessible")
    
    return all_passed

def test_ipfs_functionality() -> bool:
    """Test IPFS-related functionality."""
    section("Testing IPFS Functionality")
    
    all_passed = True
    
    # Test health check first to see if IPFS is available
    info("Checking if IPFS is available...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "health_check", "arguments": {}}
        )
        health_data = response.json().get("result", {})
        if health_data.get("ipfs_connected", False):
            success("IPFS daemon is connected")
        else:
            warning("IPFS daemon is not connected, some tests may fail")
    except requests.exceptions.RequestException as e:
        error(f"Error checking IPFS availability: {str(e)}")
        warning("IPFS health check failed, some tests may fail")
    
    # Test ipfs_add_file
    info("Testing ipfs_add_file...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "ipfs_add_file", "arguments": {"path": TEST_TEXT_FILE}}
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            if result.get("success", False):
                success(f"Successfully added file to IPFS with CID: {result.get('cid')}")
                cid = result.get("cid")
                test_results["ipfs_add_file"] = True
                
                # Save CID for subsequent tests
                test_results["test_file_cid"] = cid
            else:
                error(f"Failed to add file to IPFS: {result.get('error', 'Unknown error')}")
                test_results["ipfs_add_file"] = False
                all_passed = False
        else:
            error(f"ipfs_add_file request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["ipfs_add_file"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing ipfs_add_file: {str(e)}")
        test_results["ipfs_add_file"] = False
        all_passed = False
    
    # Test ipfs_cat
    if test_results.get("test_file_cid"):
        info("Testing ipfs_cat...")
        try:
            response = requests.post(
                f"{server_url}/call_tool",
                json={"tool_name": "ipfs_cat", "arguments": {"cid": test_results["test_file_cid"]}}
            )
            
            if response.status_code == 200:
                result = response.json().get("result", {})
                if result.get("success", False):
                    content = result.get("content", "")
                    if content == TEST_TEXT_CONTENT:
                        success("Successfully retrieved correct content from IPFS")
                        test_results["ipfs_cat"] = True
                    else:
                        error(f"Retrieved content does not match original: {content}")
                        test_results["ipfs_cat"] = False
                        all_passed = False
                else:
                    error(f"Failed to retrieve content from IPFS: {result.get('error', 'Unknown error')}")
                    test_results["ipfs_cat"] = False
                    all_passed = False
            else:
                error(f"ipfs_cat request failed with status code: {response.status_code}")
                info(f"Response: {response.text}")
                test_results["ipfs_cat"] = False
                all_passed = False
        except requests.exceptions.RequestException as e:
            error(f"Error testing ipfs_cat: {str(e)}")
            test_results["ipfs_cat"] = False
            all_passed = False
    else:
        warning("Skipping ipfs_cat test because ipfs_add_file failed")
    
    # Test ipfs_files_write and ipfs_files_read
    info("Testing ipfs_files_write...")
    mfs_path = "/mcp-test/test-file.txt"
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "ipfs_files_write", "arguments": {"path": mfs_path, "content": TEST_TEXT_CONTENT}}
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            if result.get("success", False):
                success(f"Successfully wrote to IPFS MFS at {mfs_path}")
                test_results["ipfs_files_write"] = True
                
                # Test ipfs_files_read
                info("Testing ipfs_files_read...")
                try:
                    response = requests.post(
                        f"{server_url}/call_tool",
                        json={"tool_name": "ipfs_files_read", "arguments": {"path": mfs_path}}
                    )
                    
                    if response.status_code == 200:
                        result = response.json().get("result", {})
                        if result.get("success", False):
                            content = result.get("content", "")
                            if content == TEST_TEXT_CONTENT:
                                success("Successfully read correct content from IPFS MFS")
                                test_results["ipfs_files_read"] = True
                            else:
                                error(f"Read content does not match written content: {content}")
                                test_results["ipfs_files_read"] = False
                                all_passed = False
                        else:
                            error(f"Failed to read from IPFS MFS: {result.get('error', 'Unknown error')}")
                            test_results["ipfs_files_read"] = False
                            all_passed = False
                    else:
                        error(f"ipfs_files_read request failed with status code: {response.status_code}")
                        info(f"Response: {response.text}")
                        test_results["ipfs_files_read"] = False
                        all_passed = False
                except requests.exceptions.RequestException as e:
                    error(f"Error testing ipfs_files_read: {str(e)}")
                    test_results["ipfs_files_read"] = False
                    all_passed = False
            else:
                error(f"Failed to write to IPFS MFS: {result.get('error', 'Unknown error')}")
                test_results["ipfs_files_write"] = False
                all_passed = False
        else:
            error(f"ipfs_files_write request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["ipfs_files_write"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing ipfs_files_write: {str(e)}")
        test_results["ipfs_files_write"] = False
        all_passed = False
    
    if all_passed:
        success("All IPFS functionality tests passed")
    else:
        warning("Some IPFS functionality tests failed")
    
    return all_passed

def test_model_functionality() -> bool:
    """Test model-related functionality."""
    section("Testing Model Server Functionality")
    
    all_passed = True
    
    # Test list_models
    info("Testing list_models...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "list_models", "arguments": {}}
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            if result.get("count", 0) > 0:
                success(f"Successfully listed {result.get('count')} models")
                test_results["list_models"] = True
                
                # Print the models
                info(f"Available models: {', '.join(result.get('models', {}).keys())}")
            else:
                warning("No models found")
                test_results["list_models"] = True
        else:
            error(f"list_models request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["list_models"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing list_models: {str(e)}")
        test_results["list_models"] = False
        all_passed = False
    
    # Test create_endpoint
    info("Testing create_endpoint...")
    model_name = "bert-base-uncased"  # Use a model that should be available in the server
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "create_endpoint", "arguments": {"model_name": model_name}}
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            if result.get("success", False):
                success(f"Successfully created endpoint for model {model_name}: {result.get('endpoint_id')}")
                test_results["create_endpoint"] = True
                
                # Save endpoint_id for subsequent tests
                test_results["test_endpoint_id"] = result.get("endpoint_id")
            else:
                error(f"Failed to create endpoint: {result.get('error', 'Unknown error')}")
                test_results["create_endpoint"] = False
                all_passed = False
        else:
            error(f"create_endpoint request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["create_endpoint"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing create_endpoint: {str(e)}")
        test_results["create_endpoint"] = False
        all_passed = False
    
    # Test run_inference
    if test_results.get("test_endpoint_id"):
        info("Testing run_inference...")
        try:
            response = requests.post(
                f"{server_url}/call_tool",
                json={
                    "tool_name": "run_inference",
                    "arguments": {
                        "endpoint_id": test_results["test_endpoint_id"],
                        "inputs": ["Hello world", "This is a test"]
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json().get("result", {})
                if result.get("success", False):
                    success("Successfully ran inference")
                    test_results["run_inference"] = True
                    
                    # Check for embeddings or outputs
                    if "embeddings" in result:
                        info(f"Generated {len(result['embeddings'])} embeddings")
                    elif "outputs" in result:
                        info(f"Generated {len(result['outputs'])} outputs")
                else:
                    error(f"Failed to run inference: {result.get('error', 'Unknown error')}")
                    test_results["run_inference"] = False
                    all_passed = False
            else:
                error(f"run_inference request failed with status code: {response.status_code}")
                info(f"Response: {response.text}")
                test_results["run_inference"] = False
                all_passed = False
        except requests.exceptions.RequestException as e:
            error(f"Error testing run_inference: {str(e)}")
            test_results["run_inference"] = False
            all_passed = False
    else:
        warning("Skipping run_inference test because create_endpoint failed")
    
    if all_passed:
        success("All model functionality tests passed")
    else:
        warning("Some model functionality tests failed")
    
    return all_passed

def test_api_multiplexer_functionality() -> bool:
    """Test API multiplexer functionality."""
    section("Testing API Multiplexer Functionality")
    
    all_passed = True
    
    # Test register_api_key
    info("Testing register_api_key...")
    test_provider = "test-provider"
    test_api_key = "test-api-key-" + str(int(time.time()))
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={
                "tool_name": "register_api_key", 
                "arguments": {
                    "provider": test_provider, 
                    "api_key": test_api_key
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            if result.get("success", False):
                success(f"Successfully registered API key for provider {test_provider}")
                test_results["register_api_key"] = True
            else:
                error(f"Failed to register API key: {result.get('error', 'Unknown error')}")
                test_results["register_api_key"] = False
                all_passed = False
        else:
            error(f"register_api_key request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["register_api_key"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing register_api_key: {str(e)}")
        test_results["register_api_key"] = False
        all_passed = False
    
    # Test get_api_keys
    info("Testing get_api_keys...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "get_api_keys", "arguments": {}}
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            success(f"Successfully retrieved API keys information")
            test_results["get_api_keys"] = True
            
            # Print providers
            providers = result.get("providers", [])
            info(f"Found {len(providers)} providers with API keys")
            for provider in providers:
                info(f"Provider: {provider.get('name')}, Keys: {provider.get('key_count')}")
        else:
            error(f"get_api_keys request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["get_api_keys"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing get_api_keys: {str(e)}")
        test_results["get_api_keys"] = False
        all_passed = False
    
    # Test get_multiplexer_stats
    info("Testing get_multiplexer_stats...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "get_multiplexer_stats", "arguments": {}}
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            success(f"Successfully retrieved multiplexer statistics")
            test_results["get_multiplexer_stats"] = True
            
            # Print stats
            info(f"Total requests: {result.get('total_requests')}")
            info(f"Successful requests: {result.get('successful_requests')}")
        else:
            error(f"get_multiplexer_stats request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["get_multiplexer_stats"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing get_multiplexer_stats: {str(e)}")
        test_results["get_multiplexer_stats"] = False
        all_passed = False
    
    # Test simulate_api_request
    info("Testing simulate_api_request...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={
                "tool_name": "simulate_api_request", 
                "arguments": {
                    "provider": "openai",  # Use a provider that should be available in the server
                    "prompt": "What is the capital of France?"
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            if result.get("success", False):
                success(f"Successfully simulated API request")
                test_results["simulate_api_request"] = True
                
                # Print response
                info(f"Response from {result.get('provider')}: {result.get('completion')[:50]}...")
            else:
                error(f"Failed to simulate API request: {result.get('error', 'Unknown error')}")
                test_results["simulate_api_request"] = False
                all_passed = False
        else:
            error(f"simulate_api_request request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["simulate_api_request"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing simulate_api_request: {str(e)}")
        test_results["simulate_api_request"] = False
        all_passed = False
    
    if all_passed:
        success("All API multiplexer functionality tests passed")
    else:
        warning("Some API multiplexer functionality tests failed")
    
    return all_passed

def test_task_management_functionality() -> bool:
    """Test task management functionality."""
    section("Testing Task Management Functionality")
    
    all_passed = True
    
    # Test start_task
    info("Testing start_task...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={
                "tool_name": "start_task", 
                "arguments": {
                    "task_type": "test",
                    "priority": "high",
                    "params": {
                        "input_data": "test data",
                        "processing_steps": ["step1", "step2"]
                    }
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            if result.get("success", False):
                success(f"Successfully started task with ID: {result.get('task_id')}")
                test_results["start_task"] = True
                
                # Save task_id for subsequent tests
                test_results["test_task_id"] = result.get("task_id")
            else:
                error(f"Failed to start task: {result.get('error', 'Unknown error')}")
                test_results["start_task"] = False
                all_passed = False
        else:
            error(f"start_task request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["start_task"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing start_task: {str(e)}")
        test_results["start_task"] = False
        all_passed = False
    
    # Wait for task to be processed
    if test_results.get("test_task_id"):
        info("Waiting for task to be processed...")
        time.sleep(3)
    
    # Test get_task_status
    if test_results.get("test_task_id"):
        info("Testing get_task_status...")
        try:
            response = requests.post(
                f"{server_url}/call_tool",
                json={
                    "tool_name": "get_task_status", 
                    "arguments": {
                        "task_id": test_results["test_task_id"]
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json().get("result", {})
                if result.get("success", False):
                    status = result.get("status", "unknown")
                    success(f"Successfully retrieved task status: {status}")
                    test_results["get_task_status"] = True
                    
                    # Print result if available
                    if "result" in result:
                        info(f"Task result: {result['result']}")
                else:
                    error(f"Failed to get task status: {result.get('error', 'Unknown error')}")
                    test_results["get_task_status"] = False
                    all_passed = False
            else:
                error(f"get_task_status request failed with status code: {response.status_code}")
                info(f"Response: {response.text}")
                test_results["get_task_status"] = False
                all_passed = False
        except requests.exceptions.RequestException as e:
            error(f"Error testing get_task_status: {str(e)}")
            test_results["get_task_status"] = False
            all_passed = False
    else:
        warning("Skipping get_task_status test because start_task failed")
    
    # Test list_tasks
    info("Testing list_tasks...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "list_tasks", "arguments": {}}
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            if result.get("success", False):
                success(f"Successfully listed {result.get('count')} tasks")
                test_results["list_tasks"] = True
            else:
                error(f"Failed to list tasks: {result.get('error', 'Unknown error')}")
                test_results["list_tasks"] = False
                all_passed = False
        else:
            error(f"list_tasks request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["list_tasks"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing list_tasks: {str(e)}")
        test_results["list_tasks"] = False
        all_passed = False
    
    if all_passed:
        success("All task management functionality tests passed")
    else:
        warning("Some task management functionality tests failed")
    
    return all_passed

def test_hardware_info_functionality() -> bool:
    """Test hardware info functionality."""
    section("Testing Hardware Info Functionality")
    
    all_passed = True
    
    # Test health_check
    info("Testing health_check...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "health_check", "arguments": {}}
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            success(f"Successfully retrieved health check information")
            test_results["health_check"] = True
            
            # Print info
            info(f"Server status: {result.get('status')}")
            info(f"Server version: {result.get('server_version')}")
            info(f"IPFS connected: {result.get('ipfs_connected')}")
        else:
            error(f"health_check request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["health_check"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing health_check: {str(e)}")
        test_results["health_check"] = False
        all_passed = False
    
    # Test get_hardware_info
    info("Testing get_hardware_info...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "get_hardware_info", "arguments": {}}
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            success(f"Successfully retrieved hardware information")
            test_results["get_hardware_info"] = True
            
            # Print info
            info(f"Platform: {result.get('platform')}")
            info(f"Python version: {result.get('python_version')}")
            if "cpu_count" in result:
                info(f"CPU count: {result.get('cpu_count')}")
        else:
            error(f"get_hardware_info request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["get_hardware_info"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing get_hardware_info: {str(e)}")
        test_results["get_hardware_info"] = False
        all_passed = False
    
    # Test get_hardware_capabilities
    info("Testing get_hardware_capabilities...")
    try:
        response = requests.post(
            f"{server_url}/call_tool",
            json={"tool_name": "get_hardware_capabilities", "arguments": {}}
        )
        
        if response.status_code == 200:
            result = response.json().get("result", {})
            success(f"Successfully retrieved hardware capabilities")
            test_results["get_hardware_capabilities"] = True
            
            # Print capabilities
            if "cpu" in result:
                info(f"CPU: {result.get('cpu', {}).get('name')}")
            if "gpu" in result and result["gpu"].get("available", False):
                info(f"GPU available: {result['gpu'].get('count')} devices")
        else:
            error(f"get_hardware_capabilities request failed with status code: {response.status_code}")
            info(f"Response: {response.text}")
            test_results["get_hardware_capabilities"] = False
            all_passed = False
    except requests.exceptions.RequestException as e:
        error(f"Error testing get_hardware_capabilities: {str(e)}")
        test_results["get_hardware_capabilities"] = False
        all_passed = False
    
    if all_passed:
        success("All hardware info functionality tests passed")
    else:
        warning("Some hardware info functionality tests failed")
    
    return all_passed

def run_tests(test_categories: List[str] = None) -> bool:
    """Run all or selected tests."""
    
    # Map test categories to test functions
    test_functions = {
        "connection": test_server_connection,
        "ipfs": test_ipfs_functionality,
        "models": test_model_functionality,
        "api": test_api_multiplexer_functionality,
        "tasks": test_task_management_functionality,
        "hardware": test_hardware_info_functionality
    }
    
    # If no categories specified, run all tests
    if not test_categories:
        test_categories = list(test_functions.keys())
    
    # Initialize test environment
    if not setup_test_environment():
        return False
    
    # Run tests
    category_results = {}
    all_passed = True
    
    # First test connection - if this fails, can't run other tests
    if "connection" in test_categories:
        conn_result = test_server_connection()
        category_results["connection"] = conn_result
        all_passed = all_passed and conn_result
        
        # If connection failed, can't run other tests
        if not conn_result:
            test_categories = ["connection"]
    
    # Run remaining tests if connection succeeded
    for category in test_categories:
        if category == "connection":
            continue  # Already tested
        
        if category in test_functions:
            category_results[category] = test_functions[category]()
            all_passed = all_passed and category_results[category]
    
    # Print test results summary
    section("Test Results Summary")
    
    for category, result in category_results.items():
        if result:
            print(f"{category.capitalize()}: {GREEN}PASSED{RESET}")
        else:
            print(f"{category.capitalize()}: {RED}FAILED{RESET}")
    
    # Print overall result
    passed_count = sum(1 for result in category_results.values() if result)
    total_count = len(category_results)
    
    print(f"\n{YELLOW}!{RESET} {passed_count}/{total_count} tests passed")
    
    # Print recommendations for missing functionality
    if not all_passed:
        print("\nRecommendations for missing functionality:")
        
        if "connection" in category_results and not category_results["connection"]:
            info(" - Ensure the server is running and accessible at the specified URL")
            info(" - Check that the server implements the required endpoints (/sse, /tools)")
        
        if "ipfs" in category_results and not category_results["ipfs"]:
            info(" - Ensure IPFS daemon is running")
            info(" - Check that ipfshttpclient is installed")
        
        if "models" in category_results and not category_results["models"]:
            info(" - Ensure the model server functionality is implemented")
            info(" - Check that the required models are available")
        
        if "api" in category_results and not category_results["api"]:
            info(" - Ensure the API multiplexer functionality is implemented")
        
        if "tasks" in category_results and not category_results["tasks"]:
            info(" - Ensure the task management functionality is implemented")
            info(" - Check that the task worker thread is running")
            
        if "hardware" in category_results and not category_results["hardware"]:
            info(" - Ensure hardware info functionality is implemented")
            info(" - Check that psutil is installed for hardware monitoring")
    
    # Clean up
    cleanup_test_environment()
    
    return all_passed

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test the unified MCP server.")
    parser.add_argument("--url", default="http://localhost:8001", help="URL of the MCP server")
    parser.add_argument("--test-category", dest="test_categories", action="append", 
                        choices=["connection", "ipfs", "models", "api", "tasks", "hardware"],
                        help="Test categories to run (can be specified multiple times)")
    
    args = parser.parse_args()
    
    global server_url
    server_url = args.url
    
    print(f"===== Unified MCP Server Test =====")
    print(f"Server URL: {server_url}")
    print()
    
    # Use empty list as default if no categories specified
    test_categories = args.test_categories if args.test_categories else []
    run_tests(test_categories)

if __name__ == "__main__":
    main()
