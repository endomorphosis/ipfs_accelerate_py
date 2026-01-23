#!/usr/bin/env python3
"""
Unit tests for the IPFS Accelerate MCP server components.

This script tests the individual components of the MCP server
without starting an HTTP server.
"""
import unittest
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_unit_tests")

# Add the parent directory to sys.path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the required modules
from ipfs_accelerate_py import ipfs_accelerate_py
from mcp.server import create_mcp_server

class TestMCPComponents(unittest.TestCase):
    """Test individual components of the MCP server."""
    
    def setUp(self):
        """Set up test environment."""
        # Create an IPFS Accelerate instance for testing
        try:
            self.accelerate = ipfs_accelerate_py()
        except NotImplementedError as e:
            self.skipTest(str(e))
        except Exception as e:
            self.skipTest(f"IPFS Accelerate core not available: {e}")
        
        # Create an MCP server instance
        self.mcp_server = create_mcp_server(
            name="Test MCP Server",
            description="MCP Server for unit testing",
            accelerate_instance=self.accelerate
        )
    
    def test_server_creation(self):
        """Test that the MCP server is created with correct properties."""
        self.assertEqual(self.mcp_server.name, "Test MCP Server")
        self.assertEqual(self.mcp_server.description, "MCP Server for unit testing")
        self.assertIsNotNone(self.mcp_server.state.accelerate)
    
    def test_tools_registration(self):
        """Test that tools are registered correctly."""
        # Check that we have tools registered
        self.assertGreater(len(self.mcp_server.tools), 0)
        
        # Check for specific tools
        tool_names = [tool.name for tool in self.mcp_server.tools]
        self.assertIn("detect_hardware", tool_names)
        self.assertIn("get_optimal_hardware", tool_names)
        self.assertIn("run_inference", tool_names)
    
    def test_resources_registration(self):
        """Test that resources are registered correctly."""
        # Check that we have resources registered
        self.assertGreater(len(self.mcp_server.resources), 0)
        
        # Check for specific resources
        resource_paths = [resource.path for resource in self.mcp_server.resources]
        self.assertIn("system://info", resource_paths)
        self.assertIn("system://capabilities", resource_paths)
        self.assertIn("models://available", resource_paths)
    
    def test_hardware_detection_tool(self):
        """Test the hardware detection tool function."""
        # Find the tool
        detect_hardware_tool = None
        for tool in self.mcp_server.tools:
            if tool.name == "detect_hardware":
                detect_hardware_tool = tool
                break
                
        self.assertIsNotNone(detect_hardware_tool, "Hardware detection tool not found")
        
        # Call the function
        result = detect_hardware_tool.function()
        
        # Basic validation
        self.assertIsInstance(result, dict)
        
        # The specific hardware info might vary, but CPU should always be available
        self.assertIn("cpu", result)
    
    def test_system_info_resource(self):
        """Test the system info resource function."""
        # Find the resource
        system_info_resource = None
        for resource in self.mcp_server.resources:
            if resource.path == "system://info":
                system_info_resource = resource
                break
                
        self.assertIsNotNone(system_info_resource, "System info resource not found")
        
        # Call the function
        result = system_info_resource.function()
        
        # Basic validation
        self.assertIsInstance(result, dict)
        self.assertIn("platform", result)
        self.assertIn("python_version", result)
        
    def test_model_info_resource(self):
        """Test the models info resource function."""
        # Find the resource
        models_resource = None
        for resource in self.mcp_server.resources:
            if resource.path == "models://available":
                models_resource = resource
                break
                
        self.assertIsNotNone(models_resource, "Models resource not found")
        
        # Call the function
        result = models_resource.function()
        
        # Basic validation
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
if __name__ == "__main__":
    unittest.main()
