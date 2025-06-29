#!/usr/bin/env python
"""
IPFS Accelerate MCP Integration Test Script

This script tests the integration of IPFS Accelerate with MCP.
It verifies that the server starts correctly, registers tools,
and handles requests via different transport protocols.
"""

import os
import sys
import json
import time
import logging
import subprocess
import unittest
import requests
import tempfile
from pathlib import Path
import threading
import asyncio
import websockets

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_mcp_test")

class IPFSAccelerateMCPTest(unittest.TestCase):
    """Test IPFS Accelerate MCP Integration"""
    # Remove duplicate class definition
    host = "localhost"
    port = None
    mount_path = "/mcp"
    base_url = None
    ws_url = None
    
    @classmethod
    def setUpClass(cls):
        """Start MCP server for testing"""
        # Define server parameters
        cls.port = int(os.environ.get('TEST_PORT', 8001))
        cls.base_url = f"http://{cls.host}:{cls.port}{cls.mount_path}"
        cls.ws_url = f"ws://{cls.host}:{cls.port}{cls.mount_path}/ws"
        
        # Start server in a separate thread for HTTP testing
        cls.server_process = None
        cls.start_server_in_thread()
        
        # Wait for server to start
        time.sleep(3)
        
        # Temporary file for IPFS testing
        cls.temp_file = tempfile.NamedTemporaryFile(delete=False)
        cls.temp_file.write(b"IPFS Accelerate Test Content")
        cls.temp_file.close()
    
    @classmethod
    def tearDownClass(cls):
        """Stop MCP server and clean up"""
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()
            logger.info("Server process terminated")
        
        # Clean up temp file
        if hasattr(cls, 'temp_file') and os.path.exists(cls.temp_file.name):
            os.unlink(cls.temp_file.name)
    
    
    @classmethod
    def start_server_in_thread(cls):
        """Start the MCP server in a separate thread"""
        def run_server():
            try:
                # Get the current script's directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Check if run_ipfs_mcp.py exists
                server_script = os.path.join(script_dir, "run_ipfs_mcp.py")
                if not os.path.exists(server_script):
                    server_script = "run_ipfs_mcp.py"
                
                cmd = [
                    sys.executable,
                    server_script,
                    "--host", cls.host,
                    "--port", str(cls.port),
                    "--mount-path", cls.mount_path,
                    "--transport", "ws",
                    "--debug"
                ]
                
                logger.info(f"Starting server with command: {' '.join(cmd)}")
                cls.server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Log server output
                for line in cls.server_process.stdout:
                    logger.info(f"Server: {line.decode().strip()}")
            
            except Exception as e:
                logger.error(f"Error starting server: {e}")
        
        # Run server in thread
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
    
    # ... (rest of the class remains the same)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run IPFS MCP integration tests")
    parser.add_argument("--port", type=int, help="Port number", default=8001)
    args = parser.parse_args()
    
    # Update the port number
    IPFSAccelerateMCPTest.port = args.port
    IPFSAccelerateMCPTest.base_url = f"http://{IPFSAccelerateMCPTest.host}:{IPFSAccelerateMCPTest.port}{IPFSAccelerateMCPTest.mount_path}"
    IPFSAccelerateMCPTest.ws_url = f"ws://{IPFSAccelerateMCPTest.host}:{IPFSAccelerateMCPTest.port}{IPFSAccelerateMCPTest.mount_path}/ws"
    
    # Run tests
    unittest.main(argv=sys.argv[:1], verbosity=2)
