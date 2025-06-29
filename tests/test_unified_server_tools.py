#!/usr/bin/env python3
"""
Direct MCP Server Tool Registration Test

This script starts the unified MCP server and verifies that tools are properly registered.
"""

import os
import sys
import json
import time
import logging
import subprocess
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_unified_server():
    """Start the unified MCP server."""
    # Kill any existing servers
    subprocess.run("pkill -f 'python.*mcp_server.py'", shell=True)
    time.sleep(2)
    
    # Start server
    cmd = [sys.executable, "unified_mcp_server.py", "--port", "8001", "--verbose"]
    logger.info(f"Starting server with command: {' '.join(cmd)}")
    
    with open("unified_server_test.log", "w") as log_file:
        server_proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        
    # Wait for server to start
    time.sleep(5)
    
    # Check if server failed to start
    if server_proc.poll() is not None:
        logger.error("Server failed to start")
        return None
    
    logger.info("Server started successfully")
    return server_proc

def check_tools():
    """Check which tools are registered with the server."""
    tools_url = "http://localhost:8001/tools"
    
    try:
        logger.info(f"Requesting tools from {tools_url}")
        response = requests.get(tools_url, timeout=5)
        
        if response.status_code == 200:
            tools = response.json()
            logger.info(f"Found {len(tools)} registered tools")
            
            print("\nRegistered Tools:")
            for name, details in tools.items():
                desc = details.get('description', 'No description') if isinstance(details, dict) else 'No description'
                print(f"  - {name}: {desc}")
                
            # Check expected tools
            expected_tools = [
                "get_hardware_info",
                "health_check",
                "ipfs_files_ls",
                "ipfs_files_read",
                "ipfs_files_write"
            ]
            
            missing_tools = [tool for tool in expected_tools if tool not in tools]
            
            if missing_tools:
                logger.warning(f"Missing expected tools: {', '.join(missing_tools)}")
                return False
            else:
                logger.info("All expected tools are registered")
                return True
        else:
            logger.error(f"Error getting tools: {response.status_code}")
            logger.error(response.text)
            return False
    except Exception as e:
        logger.error(f"Error checking tools: {str(e)}")
        return False

def main():
    """Main function."""
    print("==================================================")
    print("Unified MCP Server Tool Registration Test")
    print("==================================================")
    
    # Start server
    server_proc = start_unified_server()
    if not server_proc:
        logger.error("Failed to start server")
        return 1
    
    try:
        # Check tools
        success = check_tools()
        
    finally:
        # Stop server
        logger.info("Stopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
    
    if success:
        print("\n✅ All tools are properly registered!")
        return 0
    else:
        print("\n❌ Some tools are missing!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
