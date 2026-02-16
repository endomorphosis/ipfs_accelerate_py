#!/usr/bin/env python3
"""
Simple MCP Server Test with Non-Blocking IO

This script tests the MCP server in a separate process with improved output handling.

Note: This is a FastMCP smoke script; execution_context metadata is not used here.
"""
import os
import sys
import logging
import subprocess
import time
import tempfile
import threading
import requests
import anyio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_server_test")

# Create a simple server script
server_script = """
from fastmcp import FastMCP
import uvicorn
import sys

# Print diagnostics
print("Python version:", sys.version)
print("Creating MCP server...")
sys.stdout.flush()

# Create server
mcp = FastMCP("Test Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers\"\"\"
    print(f"Adding {a} + {b}")
    sys.stdout.flush()
    return a + b

@mcp.resource("test://greeting")
def get_greeting() -> str:
    \"\"\"Get a greeting\"\"\"
    print("Getting greeting")
    sys.stdout.flush()
    return "Hello from Test MCP Server!"

print("Server configured, starting...")
sys.stdout.flush()

# Start the server
if __name__ == "__main__":
    mcp.run(host="127.0.0.1", port=8765)
"""

# Stream reader function
def stream_reader(stream, prefix):
    """Read from stream line by line and log with prefix."""
    for line in iter(stream.readline, b''):
        logger.info(f"{prefix}: {line.decode('utf-8').strip()}")

# Write script to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    script_path = f.name
    f.write(server_script)

logger.info(f"Created temporary script at {script_path}")

# Start the server process
try:
    logger.info("Starting server process...")
    server_proc = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,  # Line buffered
    )
    
    # Set up stream readers
    stdout_thread = threading.Thread(
        target=stream_reader, 
        args=(server_proc.stdout, "SERVER STDOUT")
    )
    stderr_thread = threading.Thread(
        target=stream_reader, 
        args=(server_proc.stderr, "SERVER STDERR")
    )
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for server to start
    logger.info("Waiting for server to start...")
    time.sleep(5)
    
    # Test server connection
    base_url = "http://127.0.0.1:8765"
    logger.info(f"Testing connection to {base_url}...")
    
    try:
        response = requests.get(base_url)
        logger.info(f"Server response status: {response.status_code}")
        logger.info(f"Server response: {response.text}")
        
        # Test tool
        tool_url = f"{base_url}/tools/add"
        logger.info(f"Testing tool: {tool_url}")
        tool_response = requests.post(tool_url, json={"a": 5, "b": 3})
        logger.info(f"Tool response status: {tool_response.status_code}")
        logger.info(f"Tool response: {tool_response.text}")
        
        # Test resource
        resource_url = f"{base_url}/resources/test://greeting"
        logger.info(f"Testing resource: {resource_url}")
        resource_response = requests.get(resource_url)
        logger.info(f"Resource response status: {resource_response.status_code}")
        logger.info(f"Resource response: {resource_response.text}")
        
        logger.info("All tests completed successfully!")
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
    except Exception as e:
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    logger.error(f"Setup error: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Cleanup
    logger.info("Cleaning up...")
    
    # Terminate server
    if 'server_proc' in locals():
        logger.info("Terminating server process...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server didn't terminate, killing it...")
            server_proc.kill()
    
    # Remove temp file
    if os.path.exists(script_path):
        logger.info(f"Removing temporary file {script_path}")
        os.unlink(script_path)
