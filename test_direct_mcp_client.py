#!/usr/bin/env python
"""
Test Direct MCP Client Tools

This script tests the tools exposed by the Direct MCP server.
"""

import sys
import json
import logging
import traceback

from direct_mcp_client import DirectMCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TestResult:
    def __init__(self, name, success, response=None, error=None):
        self.name = name
        self.success = success
        self.response = response
        self.error = error
    
    def __str__(self):
        if self.success:
            return f"✅ {self.name}: SUCCESS"
        else:
            return f"❌ {self.name}: FAILED - {self.error}"

def test_hardware_tools(client):
    """Test hardware tools"""
    results = []
    
    try:
        # Test get_hardware_info
        logger.info("Testing get_hardware_info...")
        response = client.get_hardware_info()
        logger.info("get_hardware_info response received")
        results.append(TestResult(
            "get_hardware_info", 
            success=True,
            response=response
        ))
        
        # Test get_hardware_capabilities
        try:
            logger.info("Testing get_hardware_capabilities...")
            response = client.get_hardware_capabilities()
            logger.info("get_hardware_capabilities response received")
            results.append(TestResult(
                "get_hardware_capabilities", 
                success=True,
                response=response
            ))
        except Exception as e:
            logger.warning(f"get_hardware_capabilities test failed: {e}")
            results.append(TestResult(
                "get_hardware_capabilities", 
                success=False,
                error=str(e)
            ))
        
    except Exception as e:
        logger.error(f"Error testing hardware tools: {e}")
        traceback.print_exc()
        results.append(TestResult(
            "hardware_tools", 
            success=False,
            error=str(e)
        ))
    
    return results

def test_ipfs_tools(client):
    """Test IPFS tools"""
    results = []
    
    try:
        # Create a test file for IPFS operations
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp:
            temp.write("Test IPFS file content")
            temp_path = temp.name
        
        logger.info(f"Created test file at {temp_path}")
        
        # Test ipfs_add_file
        try:
            logger.info("Testing ipfs_add_file...")
            response = client.call_tool("ipfs_add_file", path=temp_path)
            logger.info("ipfs_add_file response received")
            results.append(TestResult(
                "ipfs_add_file", 
                success=True,
                response=response
            ))
            
            # If we got a hash, we can test other operations
            ipfs_hash = response.get("cid")
            if ipfs_hash:
                # Test ipfs_cat
                try:
                    logger.info("Testing ipfs_cat...")
                    response = client.call_tool("ipfs_cat", cid=ipfs_hash)
                    logger.info("ipfs_cat response received")
                    results.append(TestResult(
                        "ipfs_cat", 
                        success=True,
                        response=response
                    ))
                except Exception as e:
                    logger.warning(f"ipfs_cat test failed: {e}")
                    results.append(TestResult(
                        "ipfs_cat", 
                        success=False,
                        error=str(e)
                    ))
        except Exception as e:
            logger.warning(f"ipfs_add_file test failed: {e}")
            results.append(TestResult(
                "ipfs_add_file", 
                success=False,
                error=str(e)
            ))
        
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error testing IPFS tools: {e}")
        traceback.print_exc()
        results.append(TestResult(
            "ipfs_tools", 
            success=False,
            error=str(e)
        ))
    
    return results

def test_model_tools(client):
    """Test model management tools"""
    results = []
    
    try:
        # Test list_models
        try:
            logger.info("Testing list_models...")
            response = client.call_tool("list_models")
            logger.info("list_models response received")
            results.append(TestResult(
                "list_models", 
                success=True,
                response=response
            ))
        except Exception as e:
            logger.warning(f"list_models test failed: {e}")
            results.append(TestResult(
                "list_models", 
                success=False,
                error=str(e)
            ))
        
        # Test create_endpoint
        try:
            logger.info("Testing create_endpoint...")
            response = client.call_tool("create_endpoint", model_name="bert-base-uncased")
            logger.info("create_endpoint response received")
            results.append(TestResult(
                "create_endpoint", 
                success=True,
                response=response
            ))
            
            # If we got an endpoint, test running inference
            endpoint_id = response.get("endpoint_id")
            if endpoint_id:
                # Test run_inference
                try:
                    logger.info("Testing run_inference...")
                    response = client.call_tool("run_inference", 
                                              endpoint_id=endpoint_id,
                                              inputs=["Test input for inference"])
                    logger.info("run_inference response received")
                    results.append(TestResult(
                        "run_inference", 
                        success=True,
                        response=response
                    ))
                except Exception as e:
                    logger.warning(f"run_inference test failed: {e}")
                    results.append(TestResult(
                        "run_inference", 
                        success=False,
                        error=str(e)
                    ))
        except Exception as e:
            logger.warning(f"create_endpoint test failed: {e}")
            results.append(TestResult(
                "create_endpoint", 
                success=False,
                error=str(e)
            ))
    
    except Exception as e:
        logger.error(f"Error testing model tools: {e}")
        traceback.print_exc()
        results.append(TestResult(
            "model_tools", 
            success=False,
            error=str(e)
        ))
    
    return results

def main():
    """Main entry point for the script"""
    print("=" * 60)
    print("Direct MCP Client Tools Test")
    print("=" * 60)
    
    # Create the Direct MCP client
    client = DirectMCPClient(host="localhost", port=3000)
    logger.info("Created Direct MCP client")
    
    # Check if server is available
    if not client.is_server_available():
        logger.error("MCP server is not running at localhost:3000")
        return 1
    
    logger.info("Successfully connected to MCP server")
    
    # Get available tools
    try:
        available_tools = client.get_available_tools()
        print("\nAvailable Tools:")
        for tool in available_tools:
            print(f"  - {tool}")
    except Exception as e:
        logger.error(f"Error getting available tools: {e}")
        print("\nError getting available tools")
    
    # Test hardware tools
    print("\nTesting Hardware Tools:")
    hardware_results = test_hardware_tools(client)
    for result in hardware_results:
        print(result)
        if result.success and result.response:
            print("  Response preview:")
            if isinstance(result.response, dict):
                # Show only top-level keys for brevity
                print(f"  {list(result.response.keys())}")
            else:
                print(f"  {result.response}")
    
    # Test IPFS tools
    print("\nTesting IPFS Tools:")
    ipfs_results = test_ipfs_tools(client)
    for result in ipfs_results:
        print(result)
        if result.success and result.response:
            print("  Response preview:")
            if isinstance(result.response, dict):
                # Show only top-level keys for brevity
                print(f"  {list(result.response.keys())}")
            else:
                print(f"  {result.response}")
    
    # Test model tools
    print("\nTesting Model Tools:")
    model_results = test_model_tools(client)
    for result in model_results:
        print(result)
        if result.success and result.response:
            print("  Response preview:")
            if isinstance(result.response, dict):
                # Show only top-level keys for brevity
                print(f"  {list(result.response.keys())}")
            else:
                print(f"  {result.response}")
    
    # Summary
    all_results = hardware_results + ipfs_results + model_results
    success_count = sum(1 for result in all_results if result.success)
    total_count = len(all_results)
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {success_count}/{total_count} tests passed")
    print("=" * 60)
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())
