#!/usr/bin/env python3
"""
IPFS Accelerate MCP Debug Tool

A simple script to run and test the IPFS Accelerate MCP server.
"""

import os
import sys
import time
import logging
import subprocess
import argparse
import signal
import requests
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("mcp_debug.log", mode='w')
    ]
)
logger = logging.getLogger("mcp_debug")

def check_server_running(host="localhost", port=8002, timeout=5):
    """
    Check if the MCP server is running.
    
    Args:
        host: Host where the server is running
        port: Port the server is listening on
        timeout: Connection timeout in seconds
    
    Returns:
        bool: True if server is running, False otherwise
    """
    url = f"http://{host}:{port}/mcp/manifest"
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False

def start_server(debug=False):
    """
    Start the MCP server in a subprocess.
    
    Args:
        debug: Whether to run in debug mode
    
    Returns:
        subprocess.Popen: The server process
    """
    cmd = [
        "python3", "fixed_standards_mcp_server.py",
        "--host", "0.0.0.0",
        "--port", "8002"
    ]
    
    # Redirect output to a log file
    log_file = open("mcp_server.log", "w")
    
    logger.info(f"Starting server with command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid  # To allow killing the entire process group later
    )
    
    return process

def print_log_tail(log_file="mcp_server.log", lines=10):
    """
    Print the last N lines of a log file.
    
    Args:
        log_file: Path to the log file
        lines: Number of lines to print
    """
    try:
        if not os.path.exists(log_file):
            logger.error(f"Log file not found: {log_file}")
            return
        
        with open(log_file, 'r') as f:
            content = f.readlines()
            tail = content[-lines:] if len(content) > lines else content
            print("\nServer log tail:")
            print("".join(tail))
    except Exception as e:
        logger.error(f"Error reading log file: {str(e)}")

def call_tool(tool_name, args=None, host="localhost", port=8002):
    """
    Call a tool on the MCP server.
    
    Args:
        tool_name: Name of the tool to call
        args: Arguments to pass to the tool
        host: Host where the server is running
        port: Port the server is listening on
    
    Returns:
        dict: The tool result or error
    """
    url = f"http://{host}:{port}/mcp/tools/{tool_name}"
    args = args or {}
    
    try:
        response = requests.post(url, json=args, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Debug Tool")
    parser.add_argument("--start-server", action="store_true", help="Start the MCP server")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--test-tool", type=str, help="Test a specific tool")
    parser.add_argument("--list-tools", action="store_true", help="List available tools")
    args = parser.parse_args()
    
    server_process = None
    
    try:
        # Check if server is already running
        if check_server_running():
            logger.info("MCP server is already running")
        elif args.start_server:
            # Start the server
            server_process = start_server(args.debug)
            logger.info(f"Server started with PID {server_process.pid}")
            
            # Wait for the server to start
            max_retries = 20
            for i in range(max_retries):
                logger.info(f"Waiting for server to start (attempt {i+1}/{max_retries})...")
                if check_server_running():
                    logger.info("Server is running!")
                    break
                time.sleep(1)
            else:
                logger.error("Server failed to start within the timeout period")
                print_log_tail()
                sys.exit(1)
        else:
            logger.error("MCP server is not running. Use --start-server to start it.")
            sys.exit(1)
        
        # Get the server manifest
        try:
            url = "http://localhost:8002/mcp/manifest"
            logger.info(f"Getting server manifest from {url}")
            response = requests.get(url, timeout=5)
            manifest = response.json()
            
            server_name = manifest.get("server_name", "Unknown")
            server_version = manifest.get("version", "Unknown")
            mcp_version = manifest.get("mcp_version", "Unknown")
            tools = manifest.get("tools", {})
            resources = manifest.get("resources", {})
            
            logger.info(f"Connected to {server_name} v{server_version}")
            logger.info(f"MCP version: {mcp_version}")
            logger.info(f"Available tools: {len(tools)}")
            logger.info(f"Available resources: {len(resources)}")
            
            # List tools if requested
            if args.list_tools:
                print("\nAvailable Tools:")
                for tool_name, tool_info in tools.items():
                    print(f"  - {tool_name}: {tool_info.get('description', '')}")
            
            # Test a specific tool if requested
            if args.test_tool:
                if args.test_tool in tools:
                    logger.info(f"Testing tool: {args.test_tool}")
                    result = call_tool(args.test_tool)
                    if "error" in result:
                        logger.error(f"Tool error: {result['error']}")
                    else:
                        logger.info(f"Tool result: {result}")
                else:
                    logger.error(f"Tool {args.test_tool} not found")
                    logger.info(f"Available tools: {list(tools.keys())}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to server: {str(e)}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Clean up if we started the server
        if server_process:
            logger.info("Shutting down server...")
            try:
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                server_process.wait(timeout=5)
                logger.info("Server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Server did not stop gracefully, forcing...")
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error stopping server: {str(e)}")

if __name__ == "__main__":
    main()
