"""
Tests for the IPFS Accelerate MCP server.

This module provides tests to verify the MCP server functionality.
"""
import unittest
import logging
import asyncio
from fastmcp import FastMCP
from ipfs_accelerate_py import ipfs_accelerate_py
from mcp.server import create_mcp_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ipfs_accelerate_mcp_tests")

class TestMCPServer(unittest.TestCase):
    """Tests for the MCP server implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create an IPFS Accelerate instance for testing
        self.accelerate = ipfs_accelerate_py()
        
        # Create an MCP server instance
        self.mcp_server = create_mcp_server(
            name="Test MCP Server",
            accelerate_instance=self.accelerate
        )
    
    def test_server_creation(self):
        """Test that the MCP server is created successfully."""
        self.assertIsNotNone(self.mcp_server)
        self.assertEqual(self.mcp_server.name, "Test MCP Server")
        
    def test_hardware_detection_tool(self):
        """Test the hardware detection tool."""
        # Get the tool directly
        detect_hardware_tool = None
        for tool in self.mcp_server.tools:
            if tool.name == "detect_hardware":
                detect_hardware_tool = tool
                break
        
        self.assertIsNotNone(detect_hardware_tool)
        
        # Execute the tool function
        result = detect_hardware_tool.function()
        
        # Basic validation of the result
        self.assertIsInstance(result, dict)
        self.assertIn("cpu", result)
    
    def test_system_info_resource(self):
        """Test the system info resource."""
        # Find the resource
        system_info_resource = None
        for resource in self.mcp_server.resources:
            if resource.path == "system://info":
                system_info_resource = resource
                break
        
        self.assertIsNotNone(system_info_resource)
        
        # Get the resource content
        result = system_info_resource.function()
        
        # Basic validation
        self.assertIsInstance(result, dict)
        self.assertIn("platform", result)
        self.assertIn("python_version", result)

if __name__ == "__main__":
    unittest.main()
