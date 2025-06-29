#!/usr/bin/env python
"""
MCP Integration Test

This script tests the full integration of IPFS Accelerate with the MCP server.
It starts the server, connects a client, and tests various functionality.
"""

import os
import sys
import json
import logging
import time
import unittest
from contextlib import contextmanager
import multiprocessing

# Add the parent directory to the system path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

@contextmanager
def start_test_server(port=8765):
    """Start a test MCP server in a separate process"""
    from mcp.server import app, register_tool
    import uvicorn
    
    # Start the server in a separate process
    process = multiprocessing.Process(
        target=uvicorn.run,
        kwargs={
            "app": app,
            "host": "127.0.0.1",
            "port": port,
            "log_level": "error",
        },
    )
    process.daemon = True
    process.start()
    
    try:
        # Wait for the server to start
        time.sleep(2)
        yield port
    finally:
        # Terminate the server
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()

class TestMCPIntegration(unittest.TestCase):
    """Test cases for MCP integration"""
    
    def setUp(self):
        """Set up the test case"""
        self.test_port = 8765
    
    def test_server_client_integration(self):
        """Test full integration of server and client"""
        from mcp import MCPClient, is_server_running
        
        with start_test_server(self.test_port):
            # Check server availability
            is_running = is_server_running(self.test_port, host="127.0.0.1")
            self.assertTrue(is_running, "Server should be running")
            
            # Create client
            client = MCPClient(host="127.0.0.1", port=self.test_port)
            self.assertTrue(client.is_server_available(), "Client should detect server")
            
            # Get hardware info
            hardware_info = client.get_hardware_info()
            self.assertIsInstance(hardware_info, dict, "Hardware info should be a dictionary")
            self.assertIn("system", hardware_info, "Hardware info should contain system information")
            self.assertIn("accelerators", hardware_info, "Hardware info should contain accelerator information")
            
            # Print hardware info for debugging
            logger.info(f"System: {hardware_info.get('system', {}).get('os', 'Unknown')}")
            logger.info(f"CPU: {hardware_info.get('accelerators', {}).get('cpu', {}).get('available', False)}")
            
            # Access system_info resource
            try:
                system_info = client.access_resource("system_info")
                self.assertIsInstance(system_info, dict, "System info should be a dictionary")
                logger.info(f"System info: {system_info.get('os', 'Unknown')}")
            except Exception as e:
                self.fail(f"Access to system_info resource failed: {e}")
            
            # Access accelerator_info resource
            try:
                accelerator_info = client.access_resource("accelerator_info")
                self.assertIsInstance(accelerator_info, dict, "Accelerator info should be a dictionary")
                logger.info(f"Accelerators: {list(accelerator_info.keys())}")
            except Exception as e:
                self.fail(f"Access to accelerator_info resource failed: {e}")
    
    def test_server_manifest(self):
        """Test server manifest endpoint"""
        import requests
        
        with start_test_server(self.test_port):
            # Get server manifest
            response = requests.get(f"http://127.0.0.1:{self.test_port}/mcp/manifest")
            self.assertEqual(response.status_code, 200, "Manifest endpoint should return 200")
            
            manifest = response.json()
            self.assertIsInstance(manifest, dict, "Manifest should be a dictionary")
            self.assertIn("tools", manifest, "Manifest should contain tools")
            self.assertIn("resources", manifest, "Manifest should contain resources")
            
            # Check tools
            tools = manifest["tools"]
            self.assertIn("get_hardware_info", tools, "get_hardware_info tool should be available")
            
            # Check resources
            resources = manifest["resources"]
            self.assertIn("system_info", resources, "system_info resource should be available")
            self.assertIn("accelerator_info", resources, "accelerator_info resource should be available")
            
            logger.info(f"Server name: {manifest.get('server_name', 'Unknown')}")
            logger.info(f"Server version: {manifest.get('version', 'Unknown')}")

def main():
    """Run the tests"""
    unittest.main()

if __name__ == "__main__":
    main()
