#!/usr/bin/env python3
"""
Direct Tool Request Tester

This script directly tests the problematic tools in the unified_mcp_server.py
to identify and help fix the issues that are causing test failures.
"""

import os
import sys
import json
import time
import logging
import subprocess
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_unified_server():
    """Start the unified MCP server."""
    logger.info("Starting unified MCP server...")
    
    # Kill any existing servers
    subprocess.run("pkill -f 'python.*mcp_server.py'", shell=True)
    time.sleep(2)
    
    process = subprocess.Popen(
        [sys.executable, "unified_mcp_server.py", "--port", "8001", "--verbose"],
        stdout=open("unified_server_test.log", "w"),
        stderr=subprocess.STDOUT
    )
    
    # Wait for server to start
    time.sleep(5)
    
    # Check if server started successfully
    if process.poll() is not None:
        logger.error("Server failed to start")
        return None
    
    logger.info("Server started successfully")
    return process

def stop_server(process):
    """Stop the server process."""
    if process:
        logger.info("Stopping server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        time.sleep(2)
        logger.info("Server stopped")

def get_available_tools():
    """Get list of available tools from the server."""
    try:
        response = requests.get("http://localhost:8001/tools", timeout=5)
        if response.status_code == 200:
            tools = response.json()
            return tools
        else:
            logger.error(f"Failed to get tools: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error getting tools: {str(e)}")
        return {}

def test_ipfs_get_hardware_info():
    """Test the ipfs_get_hardware_info tool."""
    logger.info("Testing ipfs_get_hardware_info tool...")
    
    try:
        response = requests.post("http://localhost:8001/mcp/tool/ipfs_get_hardware_info", 
                               json={}, timeout=5)
        logger.info(f"Status code: {response.status_code}")
        if response.status_code == 200:
            logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            logger.error(f"Error: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False

def test_ipfs_gateway_url():
    """Test the ipfs_gateway_url tool with different parameter names."""
    logger.info("Testing ipfs_gateway_url tool...")
    
    test_cases = [
        {"name": "With ipfs_hash parameter", "params": {"ipfs_hash": "QmTest123"}},
        {"name": "With cid parameter", "params": {"cid": "QmTest456"}},
        {"name": "With both parameters", "params": {"ipfs_hash": "QmTest789", "cid": "QmAltTest"}}
    ]
    
    all_passed = True
    
    for case in test_cases:
        logger.info(f"Test case: {case['name']}")
        try:
            response = requests.post("http://localhost:8001/mcp/tool/ipfs_gateway_url", 
                                   json=case['params'], timeout=5)
            logger.info(f"Status code: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
            else:
                logger.error(f"Error: {response.text}")
                all_passed = False
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            all_passed = False
    
    return all_passed

def test_virtual_filesystem():
    """Test virtual filesystem operations."""
    logger.info("Testing virtual filesystem operations...")
    
    test_dir = "/test_dir"
    test_file = f"{test_dir}/test_file.txt"
    test_content = "This is test content"
    
    try:
        # Create directory
        mkdir_response = requests.post(
            "http://localhost:8001/mcp/tool/ipfs_files_mkdir", 
            json={"path": test_dir, "parents": True}, 
            timeout=5
        )
        logger.info(f"mkdir status: {mkdir_response.status_code}")
        
        if mkdir_response.status_code != 200:
            logger.error(f"Failed to create directory: {mkdir_response.text}")
            return False
        
        # Write file
        write_response = requests.post(
            "http://localhost:8001/mcp/tool/ipfs_files_write", 
            json={"path": test_file, "content": test_content}, 
            timeout=5
        )
        logger.info(f"write status: {write_response.status_code}")
        
        if write_response.status_code != 200:
            logger.error(f"Failed to write file: {write_response.text}")
            return False
        
        # List directory
        ls_response = requests.post(
            "http://localhost:8001/mcp/tool/ipfs_files_ls", 
            json={"path": test_dir}, 
            timeout=5
        )
        logger.info(f"ls status: {ls_response.status_code}")
        
        if ls_response.status_code != 200:
            logger.error(f"Failed to list directory: {ls_response.text}")
            return False
        
        # Read file
        read_response = requests.post(
            "http://localhost:8001/mcp/tool/ipfs_files_read", 
            json={"path": test_file}, 
            timeout=5
        )
        logger.info(f"read status: {read_response.status_code}")
        
        if read_response.status_code != 200:
            logger.error(f"Failed to read file: {read_response.text}")
            return False
        
        # Verify content
        content = read_response.text
        if content != test_content:
            logger.error(f"Content mismatch: {content} != {test_content}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error in VFS test: {str(e)}")
        return False

def run_all_tests():
    """Run all tests on the unified MCP server."""
    results = {
        "tools_available": False,
        "hardware_info": False,
        "gateway_url": False,
        "virtual_filesystem": False
    }
    
    server_process = start_unified_server()
    
    if server_process:
        try:
            tools = get_available_tools()
            logger.info(f"Available tools: {', '.join(tools.keys()) if tools else 'None'}")
            results["tools_available"] = bool(tools)
            
            results["hardware_info"] = test_ipfs_get_hardware_info()
            results["gateway_url"] = test_ipfs_gateway_url()
            results["virtual_filesystem"] = test_virtual_filesystem()
        finally:
            stop_server(server_process)
    
    return results

def main():
    """Main function."""
    logger.info("Starting direct tool tests")
    
    results = run_all_tests()
    
    logger.info("==== Test Results ====")
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test}: {status}")
    
    # Write results to file
    with open("direct_tool_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # All tests passed?
    all_passed = all(results.values())
    logger.info(f"Overall result: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
