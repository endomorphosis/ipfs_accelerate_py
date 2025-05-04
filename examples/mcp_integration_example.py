#!/usr/bin/env python3
"""
IPFS Accelerate MCP Integration Example

This example demonstrates how to use the IPFS Accelerate MCP server with LLMs for 
performing IPFS operations and hardware acceleration.

Usage:
    python mcp_integration_example.py

Requirements:
    - ipfs_accelerate_py (this package)
    - fastmcp or mcp (can be installed with `pip install fastmcp` or `pip install mcp[cli]`)
    - httpx (for LLM communication example)
    - asyncio
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional

# Add the root directory to the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import MCP components
try:
    # Try direct import first
    from mcp.server import create_ipfs_mcp_server
except ImportError:
    # Fallback to a mock implementation for demonstration purposes
    def create_ipfs_mcp_server(name="IPFS Accelerate", dependencies=None):
        """Mock implementation for the example script."""
        print(f"Creating mock MCP server: {name}")
        class MockMCP:
            def run(self, transport="stdio", **kwargs):
                print(f"Running mock MCP server with {transport} transport")
                print(f"Additional args: {kwargs}")
        return MockMCP()


async def run_mcp_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the MCP server with WebSocket transport.
    
    This runs the server in a subprocess to simulate a real deployment.
    """
    import subprocess
    import time
    
    print("Starting MCP server on http://{}:{}".format(host, port))
    process = subprocess.Popen(
        [sys.executable, "-m", "mcp", "run", "--transport", "ws", "--host", host, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for the server to start
    time.sleep(2)
    
    return process


async def simulate_llm_client(host: str = "127.0.0.1", port: int = 8000):
    """Simulate an LLM client connecting to the MCP server and using tools.
    
    This demonstrates how an LLM would interact with the MCP server.
    """
    import httpx
    import uuid
    
    ws_url = f"ws://{host}:{port}/mcp"
    print(f"Connecting to MCP server at {ws_url}")
    
    async with httpx.AsyncClient() as client:
        # In a real implementation, this would use a WebSocket connection
        # This is a simplified example using HTTP requests
        
        # Step 1: Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Step 2: Example of using the ipfs_status tool
        print("\n--- Example: Using ipfs_status tool ---")
        tool_call = {
            "conversation_id": conversation_id,
            "request_id": str(uuid.uuid4()),
            "type": "tool_call",
            "tool": "ipfs_status",
            "parameters": {}
        }
        
        print(f"Tool call: {json.dumps(tool_call, indent=2)}")
        print("Sending tool call to MCP server...")
        
        # In a real implementation, this would be sent over WebSocket
        # For this example, we'll simulate the response
        response = {
            "request_id": tool_call["request_id"],
            "type": "tool_response",
            "result": {
                "status": "online",
                "peer_count": 5,
                "acceleration_enabled": True,
                "version": "0.1.0"
            }
        }
        
        print(f"Response: {json.dumps(response, indent=2)}")
        
        # Step 3: Example of using file operations
        print("\n--- Example: Using ipfs_add_file tool ---")
        sample_file = os.path.join(root_dir, "README.md")
        
        tool_call = {
            "conversation_id": conversation_id,
            "request_id": str(uuid.uuid4()),
            "type": "tool_call",
            "tool": "ipfs_add_file",
            "parameters": {
                "path": sample_file,
                "wrap_with_directory": True
            }
        }
        
        print(f"Tool call: {json.dumps(tool_call, indent=2)}")
        print("Sending tool call to MCP server...")
        
        # Simulate response
        response = {
            "request_id": tool_call["request_id"],
            "type": "tool_response",
            "result": {
                "cid": "QmExample...",
                "size": 1024,
                "name": "README.md",
                "wrapped": True
            }
        }
        
        print(f"Response: {json.dumps(response, indent=2)}")
        
        # Step 4: Example of using network operations
        print("\n--- Example: Using ipfs_swarm_peers tool ---")
        
        tool_call = {
            "conversation_id": conversation_id,
            "request_id": str(uuid.uuid4()),
            "type": "tool_call",
            "tool": "ipfs_swarm_peers",
            "parameters": {}
        }
        
        print(f"Tool call: {json.dumps(tool_call, indent=2)}")
        print("Sending tool call to MCP server...")
        
        # Simulate response
        response = {
            "request_id": tool_call["request_id"],
            "type": "tool_response",
            "result": [
                {"peer_id": "QmPeer1...", "addr": "/ip4/1.2.3.4/tcp/4001", "latency": "23ms"},
                {"peer_id": "QmPeer2...", "addr": "/ip4/5.6.7.8/tcp/4001", "latency": "45ms"}
            ]
        }
        
        print(f"Response: {json.dumps(response, indent=2)}")
        
        # Step 5: Example of using acceleration operations
        print("\n--- Example: Using ipfs_accelerate_model tool ---")
        
        tool_call = {
            "conversation_id": conversation_id,
            "request_id": str(uuid.uuid4()),
            "type": "tool_call",
            "tool": "ipfs_accelerate_model",
            "parameters": {
                "cid": "QmModelExample..."
            }
        }
        
        print(f"Tool call: {json.dumps(tool_call, indent=2)}")
        print("Sending tool call to MCP server...")
        
        # Simulate response
        response = {
            "request_id": tool_call["request_id"],
            "type": "tool_response",
            "result": {
                "cid": "QmModelExample...",
                "accelerated": True,
                "device": "GPU",
                "status": "Acceleration successfully applied"
            }
        }
        
        print(f"Response: {json.dumps(response, indent=2)}")


async def main():
    """Main function to run the example."""
    host = "127.0.0.1"
    port = 8765  # Use a different port to avoid conflicts
    
    # Start the MCP server
    server_process = await run_mcp_server(host, port)
    
    try:
        # Simulate an LLM client
        await simulate_llm_client(host, port)
    finally:
        # Terminate the server process
        server_process.terminate()
        print("\nMCP server terminated")


if __name__ == "__main__":
    print("IPFS Accelerate MCP Integration Example")
    print("=======================================")
    print("This example demonstrates how an LLM would interact with the IPFS Accelerate MCP server")
    print("to perform IPFS operations and leverage hardware acceleration.")
    
    asyncio.run(main())
