#!/usr/bin/env python
"""
IPFS Accelerate MCP Client

This module provides the client for interacting with the MCP server.
"""

import os
import sys
import json
import logging
import time
import socket
import subprocess
import requests
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def is_port_in_use(port: int, host: str = "localhost") -> bool:
    """
    Check if a port is in use
    
    Args:
        port: Port to check
        host: Host to check on
    
    Returns:
        bool: True if port is in use, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((host, port))
            return True
        except (socket.error, socket.timeout):
            return False

def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """
    Find an available port
    
    Args:
        start_port: Port to start checking from
        max_attempts: Maximum number of ports to check
    
    Returns:
        int: Available port
    """
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    
    # If no port found, return a default port (which might still be in use)
    logger.warning(f"Could not find available port after {max_attempts} attempts")
    return start_port

def is_server_running(port: int = 8002, host: str = "localhost") -> bool:
    """
    Check if the MCP server is running
    
    Args:
        port: Port to check
        host: Host to check
    
    Returns:
        bool: True if server is running, False otherwise
    """
    # First check if port is in use
    if not is_port_in_use(port, host):
        return False
    
    # If port is in use, check if it's an MCP server by calling /health
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                return True
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.debug(f"Error checking server: {e}")
    
    return False

def start_server(
    port: int = 8002,
    host: str = "0.0.0.0",
    debug: bool = False,
    find_port: bool = False,
    wait: int = 0,
    python_exec: str = sys.executable,
) -> Tuple[bool, int]:
    """
    Start the MCP server as a subprocess
    
    Args:
        port: Port to bind to
        host: Host to bind to
        debug: Whether to run in debug mode
        find_port: Whether to find an available port if specified port is in use
        wait: Seconds to wait for server to start
        python_exec: Python executable to use
    
    Returns:
        Tuple[bool, int]: Success status and port used
    """
    # Check if server is already running
    if is_server_running(port=port, host="localhost"):
        logger.info(f"MCP server is already running on port {port}")
        return True, port
    
    # Find available port if requested
    if find_port and is_port_in_use(port):
        new_port = find_available_port(start_port=port)
        logger.info(f"Port {port} is in use, using port {new_port} instead")
        port = new_port
    
    # Construct module path for the server
    try:
        # Try to find the module
        from importlib import util
        module_name = "mcp.run_server"
        spec = util.find_spec(module_name)
        
        # If we found the module, use it
        if spec:
            # Use python -m for proper module resolution
            cmd = [
                python_exec, "-m", module_name,
                "--port", str(port),
                "--host", host,
            ]
            
            if debug:
                cmd.append("--debug")
        else:
            # Fallback to direct script path
            logger.warning(f"Could not find module {module_name}, falling back to script path")
            
            # Look for run_server.py in the same directory as this file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, "run_server.py")
            
            if not os.path.exists(script_path):
                logger.error(f"Could not find run_server.py at {script_path}")
                return False, port
            
            cmd = [
                python_exec, script_path,
                "--port", str(port),
                "--host", host,
            ]
            
            if debug:
                cmd.append("--debug")
        
        # Start the server as a detached process
        if os.name == "nt":  # Windows
            # Use CREATE_NEW_PROCESS_GROUP and DETACHED_PROCESS flags
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:  # Unix-like
            # Redirect output to /dev/null and use nohup
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setpgrp,  # Detach from parent process
            )
        
        logger.info(f"Started MCP server on port {port} (PID: {process.pid})")
        
        # Wait for server to start if requested
        if wait > 0:
            logger.info(f"Waiting {wait} seconds for server to start...")
            for _ in range(wait * 5):  # Check 5 times per second
                if is_server_running(port=port, host="localhost"):
                    logger.info(f"MCP server is now running on port {port}")
                    return True, port
                time.sleep(0.2)
            
            logger.warning(f"Timed out waiting for server to start on port {port}")
            return False, port
        
        return True, port
    
    except Exception as e:
        logger.error(f"Error starting MCP server: {e}")
        return False, port

class MCPClient:
    """Client for interacting with the MCP server"""
    
    def __init__(self, host: str = "localhost", port: int = 8002):
        """
        Initialize the client
        
        Args:
            host: Host of the MCP server
            port: Port of the MCP server
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
    
    def is_server_available(self) -> bool:
        """
        Check if the server is available
        
        Returns:
            bool: True if server is available, False otherwise
        """
        return is_server_running(port=self.port, host=self.host)
    
    def get_manifest(self) -> Dict[str, Any]:
        """
        Get the server manifest
        
        Returns:
            Dict[str, Any]: Server manifest
        
        Raises:
            requests.RequestException: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        response = self.session.get(f"{self.base_url}/mcp/manifest")
        response.raise_for_status()
        return response.json()
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Call a tool on the server
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments
        
        Returns:
            Any: Tool result
        
        Raises:
            requests.RequestException: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        response = self.session.post(
            f"{self.base_url}/mcp/tools/{tool_name}",
            json=kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def access_resource(self, resource: str) -> Any:
        """
        Access a resource on the server
        
        Args:
            resource: Resource to access
        
        Returns:
            Any: Resource data
        
        Raises:
            requests.RequestException: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        response = self.session.get(f"{self.base_url}/mcp/resources/{resource}")
        response.raise_for_status()
        return response.json()
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get hardware information
        
        Returns:
            Dict[str, Any]: Hardware information
        
        Raises:
            requests.RequestException: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        return self.call_tool("get_hardware_info")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information (convenience method)
        
        Returns:
            Dict[str, Any]: System information
        
        Raises:
            requests.RequestException: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        return self.access_resource("system_info")
    
    def get_accelerator_info(self) -> Dict[str, Any]:
        """
        Get accelerator information (convenience method)
        
        Returns:
            Dict[str, Any]: Accelerator information
        
        Raises:
            requests.RequestException: If request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        return self.access_resource("accelerator_info")

def get_hardware_info(host: str = "localhost", port: int = 8002) -> Dict[str, Any]:
    """
    Get hardware information (convenience function)
    
    Args:
        host: Host of the MCP server
        port: Port of the MCP server
    
    Returns:
        Dict[str, Any]: Hardware information
    
    Raises:
        requests.RequestException: If request fails
        json.JSONDecodeError: If response is not valid JSON
    """
    client = MCPClient(host=host, port=port)
    return client.get_hardware_info()

if __name__ == "__main__":
    # When run directly, print hardware info
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    parser.add_argument("--start", action="store_true", help="Start server if not running")
    
    args = parser.parse_args()
    
    # Check if server is running
    if not is_server_running(port=args.port, host=args.host):
        print(f"MCP server is not running at {args.host}:{args.port}")
        
        if args.start:
            print("Starting server...")
            success, port = start_server(port=args.port, wait=2)
            if not success:
                print("Failed to start server")
                sys.exit(1)
            
            # Update port if it changed
            if port != args.port:
                args.port = port
        else:
            print("Use --start to start the server automatically")
            sys.exit(1)
    
    # Create client
    client = MCPClient(host=args.host, port=args.port)
    
    # Get hardware info
    hardware_info = client.get_hardware_info()
    
    # Print hardware info
    print(json.dumps(hardware_info, indent=2))
