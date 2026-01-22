#!/usr/bin/env python3
"""
VS Code MCP Server for IPFS Accelerate

This script wraps the comprehensive MCP server to work properly with VS Code.
"""

import asyncio
import subprocess
import sys
import os
import signal
import logging

# Configure logging to be quiet by default
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def run_mcp_server():
    """Run the MCP server in a way that works with VS Code."""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the comprehensive MCP server
    server_path = os.path.join(script_dir, "tools", "comprehensive_mcp_server.py")
    
    # Build the command
    cmd = [
        sys.executable,
        server_path,
        "--transport", "stdio"
    ]
    
    try:
        # Start the server process
        process = subprocess.Popen(
            cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=script_dir
        )
        
        # Forward signals to the child process
        def signal_handler(signum, frame):
            process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        process.terminate()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_mcp_server()