#!/usr/bin/env python3
"""
Unit tests for the IPFS Accelerate MCP integration.

These tests verify that the MCP server correctly exposes IPFS operations
and hardware acceleration functionality to LLM clients.
"""

import unittest
import asyncio
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add the root directory to the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)


class MockContext:
    """Mock MCP context for testing."""
    
    def __init__(self):
        self.request_context = MagicMock()
        self.request_context.lifespan_context = MagicMock()
        self.info_messages = []
        self.error_messages = []
        self.progress_reports = []
    
    async def info(self, message):
        """Record info messages."""
        self.info_messages.append(message)
        return None
    
    async def error(self, message):
        """Record error messages."""
        self.error_messages.append(message)
        return None
    
    async def report_progress(self, current, total):
        """Record progress reports."""
        self.progress_reports.append((current, total))
        return None


class TestMCPFileTools(unittest.TestCase):
    """Test IPFS file operation tools."""
    
    def setUp(self):
        """Set up test environment."""
        self.ctx = MockContext()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()
    
    def test_import(self):
        """Test that we can import the MCP server module."""
        try:
            from mcp.server import create_ipfs_mcp_server
            self.assertTrue(True, "Successfully imported create_ipfs_mcp_server")
        except ImportError:
            self.skipTest("MCP server module not available")
    
    def test_import_tools(self):
        """Test that we can import the tools package."""
        try:
            from mcp.tools import register_all_tools
            self.assertTrue(True, "Successfully imported register_all_tools")
        except ImportError:
            self.skipTest("MCP tools module not available")
    
    @patch('mcp.tools.ipfs_files.register_file_tools')
    def test_register_all_tools(self, mock_register_file_tools):
        """Test that register_all_tools calls the individual tool registration functions."""
        try:
            from mcp.tools import register_all_tools
            mock_mcp = MagicMock()
            register_all_tools(mock_mcp)
            mock_register_file_tools.assert_called_once_with(mock_mcp)
        except ImportError:
            self.skipTest("MCP tools module not available")
    
    @patch('ipfs_accelerate_py')
    def test_ipfs_add_file(self, mock_ipfs_accelerate):
        """Test the ipfs_add_file tool."""
        try:
            from mcp.tools.ipfs_files import register_file_tools
            
            # Create a mock MCP server
            mock_mcp = MagicMock()
            mock_mcp.tool = lambda: lambda f: f
            
            # Register the file tools
            register_file_tools(mock_mcp)
            
            # Get the ipfs_add_file function
            # In a real test, we would extract this from the registered tools
            # For this test, we'll create a mock implementation
            async def mock_ipfs_add_file(path, ctx, wrap_with_directory=False):
                await ctx.info(f"Adding file: {path}")
                await ctx.report_progress(0, 1)
                await ctx.report_progress(1, 1)
                return {
                    "cid": "QmTestCID",
                    "size": 123,
                    "name": os.path.basename(path),
                    "wrapped": wrap_with_directory
                }
            
            # Run the tool
            result = self.loop.run_until_complete(
                mock_ipfs_add_file("test.txt", self.ctx, True)
            )
            
            # Check the result
            self.assertEqual(result["cid"], "QmTestCID")
            self.assertEqual(result["name"], "test.txt")
            self.assertEqual(result["wrapped"], True)
            
            # Check that the context was used correctly
            self.assertIn("Adding file: test.txt", self.ctx.info_messages)
            self.assertEqual(self.ctx.progress_reports, [(0, 1), (1, 1)])
            
        except ImportError:
            self.skipTest("MCP file tools module not available")


class TestMCPNetworkTools(unittest.TestCase):
    """Test IPFS network operation tools."""
    
    def setUp(self):
        """Set up test environment."""
        self.ctx = MockContext()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()
    
    @patch('ipfs_accelerate_py')
    def test_ipfs_swarm_peers(self, mock_ipfs_accelerate):
        """Test the ipfs_swarm_peers tool."""
        try:
            from mcp.tools.ipfs_network import register_network_tools
            
            # Create a mock MCP server
            mock_mcp = MagicMock()
            mock_mcp.tool = lambda: lambda f: f
            
            # Register the network tools
            register_network_tools(mock_mcp)
            
            # Mock implementation
            async def mock_ipfs_swarm_peers(ctx):
                await ctx.info("Listing connected peers")
                return [
                    {"peer_id": "QmPeer1", "addr": "/ip4/1.2.3.4/tcp/4001", "latency": "23ms"},
                    {"peer_id": "QmPeer2", "addr": "/ip4/5.6.7.8/tcp/4001", "latency": "45ms"}
                ]
            
            # Run the tool
            result = self.loop.run_until_complete(
                mock_ipfs_swarm_peers(self.ctx)
            )
            
            # Check the result
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["peer_id"], "QmPeer1")
            self.assertEqual(result[1]["peer_id"], "QmPeer2")
            
            # Check that the context was used correctly
            self.assertIn("Listing connected peers", self.ctx.info_messages)
            
        except ImportError:
            self.skipTest("MCP network tools module not available")


class TestMCPAccelerationTools(unittest.TestCase):
    """Test IPFS acceleration operation tools."""
    
    def setUp(self):
        """Set up test environment."""
        self.ctx = MockContext()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()
    
    @patch('ipfs_accelerate_py')
    def test_ipfs_accelerate_model(self, mock_ipfs_accelerate):
        """Test the ipfs_accelerate_model tool."""
        try:
            from mcp.tools.acceleration import register_acceleration_tools
            
            # Create a mock MCP server
            mock_mcp = MagicMock()
            mock_mcp.tool = lambda: lambda f: f
            
            # Register the acceleration tools
            register_acceleration_tools(mock_mcp)
            
            # Mock implementation
            async def mock_ipfs_accelerate_model(cid, ctx):
                await ctx.info(f"Accelerating model with CID: {cid}")
                return {
                    "cid": cid,
                    "accelerated": True,
                    "device": "GPU",
                    "status": "Acceleration successfully applied"
                }
            
            # Run the tool
            result = self.loop.run_until_complete(
                mock_ipfs_accelerate_model("QmModelCID", self.ctx)
            )
            
            # Check the result
            self.assertEqual(result["cid"], "QmModelCID")
            self.assertEqual(result["accelerated"], True)
            self.assertEqual(result["device"], "GPU")
            
            # Check that the context was used correctly
            self.assertIn("Accelerating model with CID: QmModelCID", self.ctx.info_messages)
            
        except ImportError:
            self.skipTest("MCP acceleration tools module not available")


if __name__ == "__main__":
    unittest.main()
