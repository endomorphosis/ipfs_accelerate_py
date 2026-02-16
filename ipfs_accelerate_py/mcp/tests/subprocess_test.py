#!/usr/bin/env python3
"""
Simple MCP test server.

This script creates a simple MCP test server that runs on a test port.

Note: This is a FastMCP smoke script; execution_context metadata is not used here.
"""
import os
import sys
import logging
from subprocess import Popen, PIPE
import time
import signal
import requests
import json
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_test_server")

# Create a simple test server script
test_server_code = """
from fastmcp import FastMCP
import uvicorn
import argparse

# Create parser
parser = argparse.ArgumentParser(description="Test MCP Server")
parser.add_argument("--port", type=int, default=8765, help="Port number to run on")
args = parser.parse_args()

# Create server
mcp = FastMCP("Test Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers\"\"\"
    return a + b

@mcp.resource("test://greeting")
def get_greeting() -> str:
    \"\"\"Get a greeting\"\"\"
    return "Hello from Test MCP Server!"

# Run the server
if __name__ == "__main__":
    print(f"Starting MCP test server on port {args.port}")
    mcp.run(host="127.0.0.1", port=args.port)
"""

# Write the server code to a temporary file
with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp:
    temp_file = temp.name
    temp.write(test_server_code)

logger.info(f"Created temporary server file: {temp_file}")

try:
    # Start the server process
    logger.info("Starting MCP test server...")
    port = 8765
    
    # Start server in a subprocess
    server_process = Popen(
        [sys.executable, temp_file, "--port", str(port)],
        stdout=PIPE,
        stderr=PIPE,
        text=True
    )
    
    # Wait for the server to start
    logger.info("Waiting for server to start...")
    time.sleep(3)
    
    # Check server output
    if server_process.stdout:
        stdout_data = server_process.stdout.read()
        if stdout_data:
            logger.info(f"Server stdout: {stdout_data}")
    
    if server_process.stderr:
        stderr_data = server_process.stderr.read()
        if stderr_data:
            logger.error(f"Server stderr: {stderr_data}")
    
    # Check if server is running
    server_url = f"http://127.0.0.1:{port}"
    logger.info(f"Testing server at {server_url}")
    
    # Try to connect to the server
    try:
        response = requests.get(server_url)
        if response.status_code == 200:
            logger.info("Server is running!")
            logger.info(f"Response: {response.text}")
            
            # Try to access a tool
            tool_url = f"{server_url}/tools/add"
            logger.info(f"Testing tool at {tool_url}")
            tool_response = requests.post(tool_url, json={"a": 5, "b": 7})
            if tool_response.status_code == 200:
                result = tool_response.json()
                logger.info(f"Tool result: {result}")
            else:
                logger.error(f"Tool error: {tool_response.status_code} - {tool_response.text}")
                
            # Try to access a resource
            resource_url = f"{server_url}/resources/test://greeting"
            logger.info(f"Testing resource at {resource_url}")
            resource_response = requests.get(resource_url)
            if resource_response.status_code == 200:
                result = resource_response.json()
                logger.info(f"Resource result: {result}")
            else:
                logger.error(f"Resource error: {resource_response.status_code} - {resource_response.text}")
        else:
            logger.error(f"Server returned status code {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to server at {server_url}")
        
except Exception as e:
    logger.error(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # Cleanup
    logger.info("Cleaning up...")
    
    # Stop the server
    if 'server_process' in locals():
        logger.info("Terminating server process...")
        server_process.terminate()
        server_process.wait(5)  # Wait up to 5 seconds
        
        # Force kill if still running
        if server_process.poll() is None:
            logger.info("Server process still running, sending SIGKILL...")
            server_process.kill()
    
    # Remove temporary file
    if os.path.exists(temp_file):
        logger.info(f"Removing temporary file {temp_file}...")
        os.unlink(temp_file)
        
    logger.info("Test completed.")
