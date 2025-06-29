#!/usr/bin/env python
"""
IPFS Accelerate MCP Server Tests

This module contains tests for the MCP server functionality.
"""

import os
import sys
import json
import logging
import unittest
import subprocess
import time
import requests
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MCPServerTest(unittest.TestCase):
    """Test case for MCP server"""
    
    @classmethod
    def setUpClass(cls):
        """Set up for all tests - start the server"""
        # Add the parent directory to sys.path
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Import necessary components
        from mcp.client import is_server_running, start_server
        
        # Set up server parameters
        cls.host = "localhost"
        cls.port = 8002
        cls.server_process = None
        
        # Check if server is already running
        if not is_server_running(port=cls.port, host=cls.host):
            # Start the server
            print(f"Starting MCP server on port {cls.port} for tests...")
            success, port = start_server(port=cls.port, wait=2)
            
            if not success:
                raise RuntimeError(f"Failed to start MCP server on port {cls.port}")
            
            # Update port if it changed
            if port != cls.port:
                cls.port = port
        else:
            print(f"MCP server is already running on port {cls.port}")
        
        # Set up the base URL
        cls.base_url = f"http://{cls.host}:{cls.port}"
        
        # Wait a bit for the server to fully initialize
        time.sleep(1)
    
    def test_health_endpoint(self):
        """Test the health endpoint"""
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("timestamp", data)
    
    def test_manifest_endpoint(self):
        """Test the manifest endpoint"""
        response = requests.get(f"{self.base_url}/mcp/manifest")
        self.assertEqual(response.status_code, 200)
        manifest = response.json()
        
        # Check basic manifest structure
        self.assertIn("server_name", manifest)
        self.assertIn("description", manifest)
        self.assertIn("version", manifest)
        self.assertIn("mcp_version", manifest)
        self.assertIn("tools", manifest)
        self.assertIn("resources", manifest)
        
        # Check that get_hardware_info tool is available
        self.assertIn("get_hardware_info", manifest["tools"])
        
        # Check that system_info resource is available
        self.assertIn("system_info", manifest["resources"])
        
        # Check that accelerator_info resource is available
        self.assertIn("accelerator_info", manifest["resources"])
    
    def test_get_hardware_info_tool(self):
        """Test the get_hardware_info tool"""
        response = requests.post(f"{self.base_url}/mcp/tools/get_hardware_info", json={})
        self.assertEqual(response.status_code, 200)
        hardware_info = response.json()
        
        # Check basic hardware info structure
        self.assertIn("system", hardware_info)
        self.assertIn("accelerators", hardware_info)
        
        # Check system info
        system = hardware_info["system"]
        self.assertIn("os", system)
        self.assertIn("architecture", system)
        self.assertIn("python_version", system)
        
        # Check accelerators
        accelerators = hardware_info["accelerators"]
        self.assertIn("cuda", accelerators)
        self.assertIn("webgpu", accelerators)
        self.assertIn("webnn", accelerators)
    
    def test_system_info_resource(self):
        """Test the system_info resource"""
        response = requests.get(f"{self.base_url}/mcp/resources/system_info")
        self.assertEqual(response.status_code, 200)
        system_info = response.json()
        
        # Check basic system info structure
        self.assertIn("os", system_info)
        self.assertIn("architecture", system_info)
        self.assertIn("python_version", system_info)
    
    def test_accelerator_info_resource(self):
        """Test the accelerator_info resource"""
        response = requests.get(f"{self.base_url}/mcp/resources/accelerator_info")
        self.assertEqual(response.status_code, 200)
        accelerator_info = response.json()
        
        # Check accelerators
        self.assertIn("cuda", accelerator_info)
        self.assertIn("webgpu", accelerator_info)
        self.assertIn("webnn", accelerator_info)
    
    def test_nonexistent_tool(self):
        """Test calling a non-existent tool"""
        response = requests.post(f"{self.base_url}/mcp/tools/nonexistent_tool", json={})
        self.assertEqual(response.status_code, 404)
    
    def test_nonexistent_resource(self):
        """Test accessing a non-existent resource"""
        response = requests.get(f"{self.base_url}/mcp/resources/nonexistent_resource")
        self.assertEqual(response.status_code, 404)

if __name__ == "__main__":
    unittest.main()
