#!/usr/bin/env python
"""
Test MCP Tool Registdef test_hardware_registration():
    """Test hardware tools registration"""
    print("\n=== Testing Hardware Tools Registration ===")
    mock_mcp = MockMCP()
    
    try:
        from ipfs_accelerate_py.mcp.tools.hardware import register_hardware_tools
        register_hardware_tools(mock_mcp)
        mock_mcp.show_registered_tools()
        return True
    except Exception as e:
        logger.error(f"Error registering hardware tools: {e}")
        return False

def test_server_tool_registration():
    """Test tool registration with the running server"""
    print("\n=== Testing Live Server Tool Registration ===")
    tester = LiveServerToolTest()
    return tester.run_test()script tests the registration of tools with the MCP server
to identify why only hardware tools are being registered.
It can test both with mock registration and against a real running server.
"""

import os
import sys
import json
import time
import logging
import importlib
import traceback
import argparse
import requests
import subprocess
from typing import Dict, Any, List, Set

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Mock MCP server for registration
class MockMCP:
    def __init__(self):
        self.registered_tools = {}
        
    def register_tool(self, name, function=None, description=None, input_schema=None):
        logger.info(f"Registering tool: {name}")
        self.registered_tools[name] = {
            "function": function.__name__ if function else None,
            "description": description,
            "input_schema": input_schema
        }
        
    def show_registered_tools(self):
        print("\nRegistered Tools:")
        for name, details in self.registered_tools.items():
            print(f"  - {name}: {details['description']}")

def test_hardware_registration():
    """Test hardware tools registration"""
    print("\n=== Testing Hardware Tools Registration ===")
    mock_mcp = MockMCP()
    
    try:
        from ipfs_accelerate_py.mcp.tools.hardware import register_hardware_tools
        register_hardware_tools(mock_mcp)
        mock_mcp.show_registered_tools()
        return True
    except Exception as e:
        logger.error(f"Error registering hardware tools: {e}")
        traceback.print_exc()
        return False

def test_ipfs_registration():
    """Test IPFS tools registration"""
    print("\n=== Testing IPFS Tools Registration ===")
    mock_mcp = MockMCP()
    
    try:
        from ipfs_accelerate_py.mcp.tools.ipfs import register_with_mcp
        register_with_mcp(mock_mcp)
        mock_mcp.show_registered_tools()
        return True
    except Exception as e:
        logger.error(f"Error registering IPFS tools: {e}")
        traceback.print_exc()
        return False

def test_inference_registration():
    """Test inference tools registration"""
    print("\n=== Testing Inference Tools Registration ===")
    mock_mcp = MockMCP()
    
    try:
        from ipfs_accelerate_py.mcp.tools.inference import register_with_mcp
        register_with_mcp(mock_mcp)
        mock_mcp.show_registered_tools()
        return True
    except Exception as e:
        logger.error(f"Error registering inference tools: {e}")
        traceback.print_exc()
        return False

def test_endpoints_registration():
    """Test endpoints tools registration"""
    print("\n=== Testing Endpoints Tools Registration ===")
    mock_mcp = MockMCP()
    
    try:
        from ipfs_accelerate_py.mcp.tools.endpoints import register_with_mcp
        register_with_mcp(mock_mcp)
        mock_mcp.show_registered_tools()
        return True
    except Exception as e:
        logger.error(f"Error registering endpoints tools: {e}")
        traceback.print_exc()
        return False

def test_status_registration():
    """Test status tools registration"""
    print("\n=== Testing Status Tools Registration ===")
    mock_mcp = MockMCP()
    
    try:
        from ipfs_accelerate_py.mcp.tools.status import register_with_mcp
        register_with_mcp(mock_mcp)
        mock_mcp.show_registered_tools()
        return True
    except Exception as e:
        logger.error(f"Error registering status tools: {e}")
        traceback.print_exc()
        return False

def test_accelerate_registration():
    """Test accelerate tools registration"""
    print("\n=== Testing Accelerate Tools Registration ===")
    mock_mcp = MockMCP()
    
    try:
        from ipfs_accelerate_py.mcp.tools.accelerate import register_with_mcp
        register_with_mcp(mock_mcp)
        mock_mcp.show_registered_tools()
        return True
    except Exception as e:
        logger.error(f"Error registering accelerate tools: {e}")
        traceback.print_exc()
        return False

def test_all_tools_registration():
    """Test registration of all tools"""
    print("\n=== Testing All Tools Registration ===")
    mock_mcp = MockMCP()
    
    try:
        from ipfs_accelerate_py.mcp.tools import register_all_tools
        register_all_tools(mock_mcp)
        mock_mcp.show_registered_tools()
        return True
    except Exception as e:
        logger.error(f"Error registering all tools: {e}")
        traceback.print_exc()
        return False

def check_ipfs_command():
    """Check if IPFS command is available"""
    print("\n=== Checking IPFS Command Availability ===")
    try:
        import subprocess
        result = subprocess.run(["ipfs", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"IPFS is available: {result.stdout.strip()}")
            return True
        else:
            print(f"IPFS command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error checking IPFS: {e}")
        return False

# Real server testing functionality
class LiveServerToolTest:
    """Test tool registration with a live MCP server."""
    
    def __init__(self, host='localhost', port=8001, protocol='http'):
        """Initialize the tool registration tester."""
        self.host = host
        self.port = port
        self.protocol = protocol
        self.base_url = f"{self.protocol}://{self.host}:{self.port}"
        self.tools_endpoint = f"{self.base_url}/tools"
        
        # Define expected tools 
        # Add any specific tools you're looking for here
        self.expected_tools = {
            "health_check",
            "ipfs_cluster_peers",
            "ipfs_cluster_pin",
            "ipfs_cluster_status",
            "ipfs_dag_get",
            "ipfs_dag_put",
            "ipfs_dht_findpeer",
            "ipfs_dht_findprovs",
            "ipfs_files_cp",
            "ipfs_files_ls",
            "ipfs_files_mkdir",
            "ipfs_files_mv",
            "ipfs_files_read",
            "ipfs_files_rm",
            "ipfs_files_stat",
            "ipfs_files_write",
            "ipfs_fs_bridge_status",
            "ipfs_fs_bridge_sync",
            "ipfs_name_publish",
            "ipfs_name_resolve",
            "ipfs_pubsub_publish",
            "ipfs_pubsub_subscribe",
            "get_hardware_info"
        }
    
    def start_server(self):
        """Start the MCP server for testing."""
        try:
            # Kill any existing server
            subprocess.run("pkill -f 'python.*unified_mcp_server.py'", shell=True)
            time.sleep(1)
            
            # Start server
            logger.info(f"Starting MCP server on port {self.port}...")
            self.server_proc = subprocess.Popen(
                [sys.executable, "unified_mcp_server.py", "--port", str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(3)  # Give server time to start
            
            # Check if server started successfully
            if self.server_proc.poll() is not None:
                stdout, stderr = self.server_proc.communicate()
                logger.error(f"Server failed to start: {stderr}")
                return False
                
            logger.info("Server started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}")
            return False
    
    def stop_server(self):
        """Stop the MCP server."""
        if hasattr(self, 'server_proc'):
            logger.info("Stopping server...")
            self.server_proc.terminate()
            time.sleep(1)
    
    def get_registered_tools(self) -> Dict[str, Any]:
        """Get the list of registered tools from the server."""
        try:
            logger.info(f"Requesting tools from {self.tools_endpoint}")
            response = requests.get(self.tools_endpoint)
            response.raise_for_status()
            tools = response.json()
            logger.info(f"Received {len(tools)} tools")
            return tools
        except Exception as e:
            logger.error(f"Error getting tools: {str(e)}")
            return {}
    
    def check_missing_tools(self, registered_tools: Dict[str, Any]) -> Set[str]:
        """Check for missing expected tools."""
        if not registered_tools:
            return self.expected_tools
            
        registered_tool_names = set(registered_tools.keys())
        missing_tools = self.expected_tools - registered_tool_names
        return missing_tools
    
    def check_unexpected_tools(self, registered_tools: Dict[str, Any]) -> Set[str]:
        """Check for unexpected tools."""
        registered_tool_names = set(registered_tools.keys())
        unexpected_tools = registered_tool_names - self.expected_tools
        return unexpected_tools
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get the schema for a specific tool."""
        try:
            response = requests.get(f"{self.tools_endpoint}/{tool_name}")
            response.raise_for_status()
            schema = response.json()
            return schema
        except Exception as e:
            logger.error(f"Error getting schema for {tool_name}: {str(e)}")
            return {}
    
    def run_test(self, start_server=True) -> bool:
        """Run the tool registration test."""
        success = True
        
        # Start server if requested
        if start_server:
            if not self.start_server():
                logger.error("Failed to start server. Test aborted.")
                return False
        
        try:
            # Get registered tools
            registered_tools = self.get_registered_tools()
            
            if not registered_tools:
                logger.error("No tools returned from server")
                return False
                
            # Check for missing tools
            missing_tools = self.check_missing_tools(registered_tools)
            if missing_tools:
                logger.warning(f"Missing expected tools: {', '.join(missing_tools)}")
                success = False
            else:
                logger.info("All expected tools are registered")
                
            # Check for unexpected tools
            unexpected_tools = self.check_unexpected_tools(registered_tools)
            if unexpected_tools:
                logger.info(f"Unexpected tools: {', '.join(unexpected_tools)}")
                
            # Print tool details for debugging
            logger.info("Tool details:")
            for tool_name, tool_info in registered_tools.items():
                if isinstance(tool_info, dict) and 'description' in tool_info:
                    logger.info(f"- {tool_name}: {tool_info.get('description', 'No description')}")
                else:
                    logger.info(f"- {tool_name}")
                    
            # Check each missing tool schema
            for tool_name in missing_tools:
                logger.info(f"Checking schema for missing tool: {tool_name}")
                schema = self.get_tool_schema(tool_name)
                if schema:
                    logger.warning(f"Tool {tool_name} has schema but is not in tools list")
        
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            success = False
        finally:
            # Stop server if we started it
            if start_server:
                self.stop_server()
        
        return success

def main():
    """Main entry point"""
    print("=" * 50)
    print("MCP Tool Registration Test")
    print("=" * 50)
    
    # Check IPFS availability (needed for IPFS tools)
    ipfs_available = check_ipfs_command()
    
    # Test hardware tools
    hardware_success = test_hardware_registration()
    
    # Test IPFS tools
    ipfs_success = test_ipfs_registration()
    
    # Test inference tools
    inference_success = test_inference_registration()
    
    # Test endpoints tools
    endpoints_success = test_endpoints_registration()
    
    # Test status tools
    status_success = test_status_registration()
    
    # Test accelerate tools
    accelerate_success = test_accelerate_registration()
    
    # Test all tools registration
    all_tools_success = test_all_tools_registration()
    
    # Summary
    print("\n" + "=" * 50)
    print("Registration Test Results")
    print("=" * 50)
    print(f"IPFS Available:       {'✅ YES' if ipfs_available else '❌ NO'}")
    print(f"Hardware Tools:       {'✅ PASS' if hardware_success else '❌ FAIL'}")
    print(f"IPFS Tools:           {'✅ PASS' if ipfs_success else '❌ FAIL'}")
    print(f"Inference Tools:      {'✅ PASS' if inference_success else '❌ FAIL'}")
    print(f"Endpoints Tools:      {'✅ PASS' if endpoints_success else '❌ FAIL'}")
    print(f"Status Tools:         {'✅ PASS' if status_success else '❌ FAIL'}")
    print(f"Accelerate Tools:     {'✅ PASS' if accelerate_success else '❌ FAIL'}")
    print(f"All Tools:            {'✅ PASS' if all_tools_success else '❌ FAIL'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
