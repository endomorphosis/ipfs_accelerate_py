#!/usr/bin/env python
"""
Test MCP Client Tools

This script tests the tools exposed by the MCP server to verify they are
working correctly and accessible via the MCP client.
"""

import sys
import json
import logging
import traceback

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

def print_json(data):
    """Print JSON data in a readable format"""
    print(json.dumps(data, indent=2, sort_keys=True))

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
        
        # Test test_hardware if available
        try:
            logger.info("Testing test_hardware...")
            response = client.call_tool("test_hardware")
            logger.info("test_hardware response received")
            results.append(TestResult(
                "test_hardware", 
                success=True,
                response=response
            ))
        except Exception as e:
            logger.warning(f"test_hardware test failed: {e}")
            results.append(TestResult(
                "test_hardware", 
                success=False,
                error=str(e)
            ))
        
        # Test recommend_hardware if available
        try:
            logger.info("Testing recommend_hardware...")
            response = client.call_tool("recommend_hardware", model_name="llama-2-7b")
            logger.info("recommend_hardware response received")
            results.append(TestResult(
                "recommend_hardware", 
                success=True,
                response=response
            ))
        except Exception as e:
            logger.warning(f"recommend_hardware test failed: {e}")
            results.append(TestResult(
                "recommend_hardware", 
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
        # Test ipfs_node_info
        try:
            logger.info("Testing ipfs_node_info...")
            response = client.call_tool("ipfs_node_info")
            logger.info("ipfs_node_info response received")
            results.append(TestResult(
                "ipfs_node_info", 
                success=True,
                response=response
            ))
        except Exception as e:
            logger.warning(f"ipfs_node_info test failed: {e}")
            results.append(TestResult(
                "ipfs_node_info", 
                success=False,
                error=str(e)
            ))
        
        # Test ipfs_gateway_url
        try:
            logger.info("Testing ipfs_gateway_url...")
            response = client.call_tool("ipfs_gateway_url", ipfs_hash="QmTest123456789")
            logger.info("ipfs_gateway_url response received")
            results.append(TestResult(
                "ipfs_gateway_url", 
                success=True,
                response=response
            ))
        except Exception as e:
            logger.warning(f"ipfs_gateway_url test failed: {e}")
            results.append(TestResult(
                "ipfs_gateway_url", 
                success=False,
                error=str(e)
            ))
        
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
            response = client.call_tool("ipfs_add_file", file_path=temp_path)
            logger.info("ipfs_add_file response received")
            results.append(TestResult(
                "ipfs_add_file", 
                success=True,
                response=response
            ))
            
            # If we got a hash, we can test other operations
            if "cid" in response:
                ipfs_hash = response["cid"]
                
                # Test ipfs_cat_file
                try:
                    logger.info("Testing ipfs_cat_file...")
                    response = client.call_tool("ipfs_cat_file", ipfs_hash=ipfs_hash)
                    logger.info("ipfs_cat_file response received")
                    results.append(TestResult(
                        "ipfs_cat_file", 
                        success=True,
                        response=response
                    ))
                except Exception as e:
                    logger.warning(f"ipfs_cat_file test failed: {e}")
                    results.append(TestResult(
                        "ipfs_cat_file", 
                        success=False,
                        error=str(e)
                    ))
                
                # Test ipfs_pin
                try:
                    logger.info("Testing ipfs_pin...")
                    response = client.call_tool("ipfs_pin", ipfs_hash=ipfs_hash)
                    logger.info("ipfs_pin response received")
                    results.append(TestResult(
                        "ipfs_pin", 
                        success=True,
                        response=response
                    ))
                except Exception as e:
                    logger.warning(f"ipfs_pin test failed: {e}")
                    results.append(TestResult(
                        "ipfs_pin", 
                        success=False,
                        error=str(e)
                    ))
                
                # Test ipfs_unpin
                try:
                    logger.info("Testing ipfs_unpin...")
                    response = client.call_tool("ipfs_unpin", ipfs_hash=ipfs_hash)
                    logger.info("ipfs_unpin response received")
                    results.append(TestResult(
                        "ipfs_unpin", 
                        success=True,
                        response=response
                    ))
                except Exception as e:
                    logger.warning(f"ipfs_unpin test failed: {e}")
                    results.append(TestResult(
                        "ipfs_unpin", 
                        success=False,
                        error=str(e)
                    ))
                
                # Test ipfs_get_file
                try:
                    output_path = f"{temp_path}_retrieved"
                    logger.info("Testing ipfs_get_file...")
                    response = client.call_tool("ipfs_get_file", ipfs_hash=ipfs_hash, output_path=output_path)
                    logger.info("ipfs_get_file response received")
                    results.append(TestResult(
                        "ipfs_get_file", 
                        success=True,
                        response=response
                    ))
                    
                    # Clean up retrieved file
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                except Exception as e:
                    logger.warning(f"ipfs_get_file test failed: {e}")
                    results.append(TestResult(
                        "ipfs_get_file", 
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

def get_available_tools(client):
    """Get available tools from MCP client"""
    try:
        # Get the manifest
        manifest = client.get_manifest()
        
        # Extract tool names
        tools = []
        if "tools" in manifest:
            tools = [tool["name"] for tool in manifest["tools"]]
        
        return tools
    except Exception as e:
        logger.error(f"Error getting available tools: {e}")
        return []

def main():
    """Main entry point for the script"""
    print("=" * 60)
    print("MCP Client Tools Test")
    print("=" * 60)
    
    # Import the MCP client
    try:
        from mcp.client import MCPClient
        logger.info("Successfully imported MCP client")
    except ImportError as e:
        logger.error(f"Error importing MCP client: {e}")
        return 1
    
    # Connect to the MCP server
    try:
        client = MCPClient(host="localhost", port=3000)
        logger.info("Successfully connected to MCP server")
    except Exception as e:
        logger.error(f"Error connecting to MCP server: {e}")
        return 1
    
    # Get available tools
    available_tools = get_available_tools(client)
    print("\nAvailable Tools:")
    for tool in available_tools:
        print(f"  - {tool}")
    
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
    
    # Summary
    all_results = hardware_results + ipfs_results
    success_count = sum(1 for result in all_results if result.success)
    total_count = len(all_results)
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {success_count}/{total_count} tests passed")
    print("=" * 60)
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())
