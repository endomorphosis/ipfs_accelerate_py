#!/usr/bin/env python3
"""
Simple MCP Server Tool Test

This script starts the unified MCP server and checks what tools are registered.
"""

import os
import sys
import json
import time
import logging
import requests
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def start_server():
    """Start the MCP server for testing."""
    # Kill any existing server
    subprocess.run("pkill -f 'python.*unified_mcp_server.py'", shell=True)
    time.sleep(2)
    
    # Start server
    logger.info("Starting MCP server...")
    cmd = [sys.executable, "unified_mcp_server.py", "--port", "8001", "--verbose"]
    
    server_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(5)
    
    # Check if server is running
    if server_proc.poll() is not None:
        stdout, stderr = server_proc.communicate()
        logger.error(f"Server failed to start: {stderr}")
        return None
    
    logger.info("Server started successfully")
    return server_proc

def stop_server(server_proc):
    """Stop the server process."""
    if server_proc:
        logger.info("Stopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()

def check_tools():
    """Check which tools are registered with the server."""
    try:
        base_url = "http://localhost:8001"
        tools_url = f"{base_url}/tools"
        
        logger.info(f"Checking tools at: {tools_url}")
        response = requests.get(tools_url, timeout=5)
        
        if response.status_code == 200:
            tools = response.json()
            logger.info(f"Found {len(tools)} registered tools")
            
            print("\nRegistered Tools:")
            for name, details in tools.items():
                desc = details.get('description', 'No description') if isinstance(details, dict) else 'No description'
                print(f"  - {name}: {desc}")
                
            return tools
        else:
            logger.error(f"Error getting tools: {response.status_code}")
            logger.error(response.text)
            return {}
    except Exception as e:
        logger.error(f"Error checking tools: {str(e)}")
        return {}

def main():
    """Main function."""
    # Start server
    server_proc = start_server()
    if not server_proc:
        logger.error("Failed to start server")
        return 1
    
    try:
        # Check which tools are registered
        registered_tools = check_tools()
        
        # Check if there are any tools
        if not registered_tools:
            logger.warning("No tools are registered with the server!")
        
        # Check how the server output
        stdout, stderr = server_proc.communicate(timeout=0.1)
        if stdout:
            logger.info(f"Server output: {stdout}")
        if stderr:
            logger.error(f"Server errors: {stderr}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    finally:
        # Stop server
        stop_server(server_proc)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
