#!/usr/bin/env python3
"""
Unit tests for the IPFS Accelerate MCP integration.

These tests verify that the MCP server and tools function correctly.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import MCP components
from mcp.mock_mcp import FastMCP, Context
from mcp.types import IPFSAccelerateContext
from mcp.tools.mock_ipfs import MockIPFSClient
from mcp.server import create_ipfs_mcp_server, register_tools


class TestMCPIntegration(unittest.TestCase):
    """Test cases for the MCP integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock MCP server for testing
        self.mcp_server = FastMCP(name="Test MCP Server")
        
        # Create a mock context
        self.context = Context()
    
    def test_server_creation(self):
        """Test that the MCP server can be created."""
        # Create a new server
        server = create_ipfs_mcp_server("Test Server")
        
        # Verify that the server was created with the correct name
        self.assertEqual(server.name, "Test Server")
        
        # Verify that the server has lifespan handlers
        self.assertIsNotNone(server.lifespan_start_handler)
        self.assertIsNotNone(server.lifespan_stop_handler)
    
    def test_tool_registration(self):
        """Test that tools can be registered with the server."""
        # Register tools with the server
        register_tools(self.mcp_server)
        
        # Verify that the server has tools registered
        self.assertGreater(len(self.mcp_server.tools), 0)
    
    @patch('mcp.tools.get_ipfs_client')
    async def test_ipfs_add_file(self, mock_get_client):
        """Test the ipfs_add_file tool."""
        # Set up a mock IPFS client
        mock_client = MockIPFSClient()
        mock_get_client.return_value = mock_client
        
        # Import the tool registration function
        from mcp.tools.ipfs_files import register_files_tools
        
        # Register the file tools
        register_files_tools(self.mcp_server)
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(b"Test content")
            temp_path = temp.name
        
        try:
            # Call the tool
            tool_func = self.mcp_server.tools["ipfs_add_file"]["function"]
            result = await tool_func(path=temp_path, ctx=self.context)
            
            # Verify the result
            self.assertTrue(result["success"])
            self.assertIsNotNone(result["cid"])
            self.assertEqual(result["size"], len(b"Test content"))
            self.assertEqual(result["name"], os.path.basename(temp_path))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('mcp.tools.get_ipfs_client')
    async def test_ipfs_cat(self, mock_get_client):
        """Test the ipfs_cat tool."""
        # Set up a mock IPFS client
        mock_client = MockIPFSClient()
        mock_get_client.return_value = mock_client
        
        # Import the tool registration function
        from mcp.tools.ipfs_files import register_files_tools
        
        # Register the file tools
        register_files_tools(self.mcp_server)
        
        # Mock a CID
        cid = "QmTestCid123456789"
        
        # Call the tool
        tool_func = self.mcp_server.tools["ipfs_cat"]["function"]
        result = await tool_func(cid=cid, ctx=self.context)
        
        # Verify the result is a string
        self.assertIsInstance(result, str)
    
    @patch('mcp.tools.get_ipfs_client')
    async def test_ipfs_files_write(self, mock_get_client):
        """Test the ipfs_files_write tool."""
        # Set up a mock IPFS client
        mock_client = MockIPFSClient()
        mock_get_client.return_value = mock_client
        
        # Import the tool registration function
        from mcp.tools.ipfs_files import register_files_tools
        
        # Register the file tools
        register_files_tools(self.mcp_server)
        
        # Call the tool
        tool_func = self.mcp_server.tools["ipfs_files_write"]["function"]
        result = await tool_func(
            path="/test/file.txt",
            content="Test content",
            ctx=self.context
        )
        
        # Verify the result
        self.assertTrue(result["written"])
        self.assertEqual(result["path"], "/test/file.txt")
        self.assertIsNotNone(result["cid"])
    
    async def test_server_lifecycle(self):
        """Test the server lifecycle (start/stop)."""
        # Create a server
        server = create_ipfs_mcp_server("Lifecycle Test")
        
        # Start the server
        context = await server.start()
        
        # Verify that the context was created
        self.assertIsNotNone(context)
        self.assertIsInstance(context, IPFSAccelerateContext)
        
        # Stop the server
        await server.stop()


# Run the tests
if __name__ == "__main__":
    # For async tests, we need to use asyncio
    async def run_async_tests():
        # Create a test suite
        suite = unittest.TestSuite()
        
        # Add the tests
        test_cases = [
            TestMCPIntegration("test_server_creation"),
            TestMCPIntegration("test_tool_registration")
        ]
        
        # Add async tests
        async_tests = [
            TestMCPIntegration("test_ipfs_add_file"),
            TestMCPIntegration("test_ipfs_cat"),
            TestMCPIntegration("test_ipfs_files_write"),
            TestMCPIntegration("test_server_lifecycle")
        ]
        
        # Run the standard tests
        for test in test_cases:
            suite.addTest(test)
        
        # Run the standard tests
        runner = unittest.TextTestRunner()
        runner.run(suite)
        
        # Run the async tests
        for test in async_tests:
            try:
                await getattr(test, test._testMethodName)()
                print(f"PASS: {test._testMethodName}")
            except Exception as e:
                print(f"FAIL: {test._testMethodName} - {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Run the async tests
    asyncio.run(run_async_tests())
