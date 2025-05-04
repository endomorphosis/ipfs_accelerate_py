#!/usr/bin/env python3
"""
Unit tests for IPFS Accelerate MCP Server

This module contains tests for the IPFS Accelerate MCP server implementation.
"""

import os
import sys
import json
import time
import unittest
import threading
import logging
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_mcp_server")

class TestMCPServer(unittest.TestCase):
    """
    Tests for the IPFS Accelerate MCP Server
    """
    
    def setUp(self):
        """Set up test case"""
        # Default server configuration
        self.host = "localhost"
        self.port = 8888  # Use a non-standard port for testing
        self.name = "ipfs-accelerate-test"
        self.description = "Test IPFS Accelerate MCP Server"
        
        # Import MCP modules
        try:
            from ipfs_accelerate_py.mcp import (
                create_server,
                register_components,
                start_server_thread,
                stop_server,
                get_server_info
            )
            
            self.create_server = create_server
            self.register_components = register_components
            self.start_server_thread = start_server_thread
            self.stop_server = stop_server
            self.get_server_info = get_server_info
        
        except ImportError as e:
            logger.error(f"Could not import MCP modules: {e}")
            raise
    
    def tearDown(self):
        """Tear down test case"""
        # Clean up if needed
        pass
    
    def test_create_server(self):
        """Test creating an MCP server"""
        # Create server
        try:
            mcp = self.create_server(
                host=self.host,
                port=self.port,
                name=self.name,
                description=self.description,
                verbose=True
            )
            
            # Verify server properties
            self.assertEqual(mcp.host, self.host)
            self.assertEqual(mcp.port, self.port)
            self.assertEqual(mcp.name, self.name)
            self.assertEqual(mcp.description, self.description)
            
            # Verify no components registered yet
            self.assertEqual(len(mcp.tools), 0)
            self.assertEqual(len(mcp.resources), 0)
        
        except Exception as e:
            self.fail(f"Creating server failed with error: {e}")
    
    def test_register_components(self):
        """Test registering components with an MCP server"""
        # Create server
        mcp = self.create_server(
            host=self.host,
            port=self.port,
            name=self.name,
            description=self.description
        )
        
        # Register components
        try:
            self.register_components(mcp)
            
            # Verify components were registered
            self.assertGreater(len(mcp.tools), 0)
            self.assertGreater(len(mcp.resources), 0)
            
            # Verify specific components
            tool_names = list(mcp.tools.keys())
            resource_names = list(mcp.resources.keys())
            
            logger.info(f"Registered tools: {tool_names}")
            logger.info(f"Registered resources: {resource_names}")
            
            # Check for required tools
            self.assertIn("get_hardware_info", tool_names)
            self.assertIn("test_hardware", tool_names)
            self.assertIn("recommend_hardware", tool_names)
            
            # Check for required resources
            self.assertIn("ipfs_accelerate/version", resource_names)
            self.assertIn("ipfs_accelerate/system_info", resource_names)
            self.assertIn("ipfs_accelerate/config", resource_names)
            self.assertIn("ipfs_accelerate/supported_models", resource_names)
        
        except Exception as e:
            self.fail(f"Registering components failed with error: {e}")
    
    def test_get_server_info(self):
        """Test getting server info"""
        # Create server
        mcp = self.create_server(
            host=self.host,
            port=self.port,
            name=self.name,
            description=self.description
        )
        
        # Register components
        self.register_components(mcp)
        
        # Get server info
        try:
            info = self.get_server_info(mcp)
            
            # Verify info
            self.assertEqual(info["name"], self.name)
            self.assertEqual(info["description"], self.description)
            self.assertEqual(info["host"], self.host)
            self.assertEqual(info["port"], self.port)
            self.assertEqual(info["url"], f"http://{self.host}:{self.port}")
            
            # Verify tools and resources
            self.assertIn("tools", info)
            self.assertIn("resources", info)
            self.assertGreater(len(info["tools"]), 0)
            self.assertGreater(len(info["resources"]), 0)
        
        except Exception as e:
            self.fail(f"Getting server info failed with error: {e}")
    
    def test_start_stop_server(self):
        """Test starting and stopping the server"""
        # Create server
        mcp = self.create_server(
            host=self.host,
            port=self.port,
            name=self.name,
            description=self.description
        )
        
        # Register components
        self.register_components(mcp)
        
        # Start server in thread
        try:
            server_thread = self.start_server_thread(mcp)
            
            # Wait for server to start
            time.sleep(1)
            
            # Verify thread is running
            self.assertTrue(server_thread.is_alive())
            
            # Try to connect to server
            # We can add more detailed verification in a real integration test
            
            # Stop server
            self.stop_server(mcp)
            
            # Wait for thread to stop
            time.sleep(1)
            
            # Thread may or may not be alive, depending on how stop_server is implemented
            # Just log the status
            logger.info(f"Server thread alive after stop: {server_thread.is_alive()}")
        
        except Exception as e:
            self.fail(f"Starting/stopping server failed with error: {e}")
    
    def test_full_lifecycle(self):
        """Test the full server lifecycle"""
        # Import all modules
        from ipfs_accelerate_py.mcp import (
            create_and_start_server,
            stop_server,
            get_server_info,
            get_version
        )
        
        # Create and start server
        try:
            mcp = create_and_start_server(
                host=self.host,
                port=self.port,
                name=self.name,
                description=self.description,
                verbose=True,
                thread=True
            )
            
            # Wait for server to start
            time.sleep(1)
            
            # Get server info
            info = get_server_info(mcp)
            
            # Verify info
            self.assertEqual(info["name"], self.name)
            self.assertEqual(info["description"], self.description)
            
            # Get version info
            version = get_version()
            
            # Verify version info
            self.assertIn("version", version)
            self.assertIn("name", version)
            self.assertIn("description", version)
            
            # Stop server
            stop_server(mcp)
        
        except Exception as e:
            self.fail(f"Full lifecycle test failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
