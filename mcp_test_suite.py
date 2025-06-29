#!/usr/bin/env python
"""
MCP Server Test Suite

This is a comprehensive test suite for diagnosing and testing MCP servers.
It performs connectivity tests, tool checks, and helps diagnose issues.
"""

import os
import sys
import json
import time
import socket
import requests
import logging
import argparse
import subprocess
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MCPServerTester:
    """MCP Server Test Suite"""
    
    def __init__(self, host: str = "localhost", port: int = 8002, timeout: int = 5):
        """Initialize the tester"""
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}/mcp"
        self.manifest = None
    
    def check_port_in_use(self) -> bool:
        """Check if the port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            try:
                s.connect((self.host, self.port))
                return True
            except (socket.error, socket.timeout):
                return False
    
    def test_server_connection(self) -> Dict[str, Any]:
        """Test connection to MCP server and return manifest"""
        try:
            response = requests.get(f"{self.base_url}/manifest", timeout=self.timeout)
            response.raise_for_status()
            self.manifest = response.json()
            return {
                "success": True,
                "manifest": self.manifest,
                "error": None
            }
        except requests.RequestException as e:
            return {
                "success": False,
                "manifest": None,
                "error": str(e)
            }
    
    def test_tool(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Test a specific MCP tool"""
        try:
            args = args or {}
            response = requests.post(
                f"{self.base_url}/tools/{tool_name}",
                json=args,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return {
                "success": True,
                "result": result,
                "error": None
            }
        except requests.RequestException as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    def test_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Test all tools available on the server"""
        if not self.manifest:
            self.test_server_connection()
            if not self.manifest:
                return {}
        
        tools = self.manifest.get("tools", {})
        results = {}
        
        for tool_name in tools:
            results[tool_name] = self.test_tool(tool_name)
        
        return results
    
    def test_resource(self, resource_name: str) -> Dict[str, Any]:
        """Test a specific MCP resource"""
        try:
            response = requests.get(
                f"{self.base_url}/resources/{resource_name}",
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return {
                "success": True,
                "result": result,
                "error": None
            }
        except requests.RequestException as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    def test_all_resources(self) -> Dict[str, Dict[str, Any]]:
        """Test all resources available on the server"""
        if not self.manifest:
            self.test_server_connection()
            if not self.manifest:
                return {}
        
        resources = self.manifest.get("resources", {})
        results = {}
        
        for resource_name in resources:
            results[resource_name] = self.test_resource(resource_name)
        
        return results
    
    def restart_server(self, server_script: str = "fixed_standards_mcp_server.py") -> bool:
        """Kill existing server and start a new one"""
        # Kill existing processes
        try:
            subprocess.run(f"pkill -f {server_script}", shell=True)
            time.sleep(2)
        except Exception:
            pass
        
        # Start new server
        try:
            cmd = f"python {server_script} --host {self.host} --port {self.port} > server.log 2>&1 &"
            subprocess.run(cmd, shell=True)
            time.sleep(5)  # Wait for server to start
            return True
        except Exception as e:
            logger.error(f"Failed to restart server: {e}")
            return False
    
    def run_uvicorn_directly(self, module: str = "direct_uvicorn_mcp_server:app") -> bool:
        """Run uvicorn directly"""
        # Kill existing processes
        try:
            subprocess.run("pkill -f uvicorn", shell=True)
            time.sleep(2)
        except Exception:
            pass
        
        # Start uvicorn directly
        try:
            cmd = f"python -m uvicorn {module} --host {self.host} --port {self.port} > uvicorn_server.log 2>&1 &"
            subprocess.run(cmd, shell=True)
            time.sleep(5)  # Wait for server to start
            return True
        except Exception as e:
            logger.error(f"Failed to start Uvicorn server: {e}")
            return False
    
    def check_server_processes(self) -> List[str]:
        """Check for running server processes"""
        try:
            result = subprocess.run("ps -ef | grep python | grep -v grep", shell=True, capture_output=True, text=True)
            return result.stdout.strip().split("\n")
        except Exception:
            return []
    
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run full diagnostics on the server"""
        diagnostics = {}
        
        # Check if port is in use
        port_in_use = self.check_port_in_use()
        diagnostics["port_in_use"] = port_in_use
        
        # Check for server processes
        diagnostics["processes"] = self.check_server_processes()
        
        # Test server connection
        connection_result = self.test_server_connection()
        diagnostics["connection"] = connection_result
        
        # If connection succeeds, test tools and resources
        if connection_result["success"]:
            # Test all tools
            tool_results = self.test_all_tools()
            diagnostics["tools"] = {
                "total": len(tool_results),
                "success": sum(1 for r in tool_results.values() if r["success"]),
                "detailed": tool_results
            }
            
            # Test all resources
            resource_results = self.test_all_resources()
            diagnostics["resources"] = {
                "total": len(resource_results),
                "success": sum(1 for r in resource_results.values() if r["success"]),
                "detailed": resource_results
            }
        
        return diagnostics

def print_separator():
    """Print a separator line"""
    print("=" * 80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MCP Server Test Suite")
    parser.add_argument("--host", type=str, default="localhost",
                      help="Host to connect to (default: localhost)")
    parser.add_argument("--port", type=int, default=8002,
                      help="Port to connect to (default: 8002)")
    parser.add_argument("--timeout", type=int, default=5,
                      help="Timeout for requests in seconds (default: 5)")
    parser.add_argument("--restart-server", action="store_true",
                      help="Restart the MCP server")
    parser.add_argument("--server-script", type=str, default="fixed_standards_mcp_server.py",
                      help="Server script to use when restarting")
    parser.add_argument("--use-uvicorn", action="store_true",
                      help="Use Uvicorn directly to run the server")
    parser.add_argument("--uvicorn-module", type=str, default="direct_uvicorn_mcp_server:app",
                      help="Module to use with Uvicorn")
    parser.add_argument("--diagnostics", action="store_true",
                      help="Run full diagnostics")
    args = parser.parse_args()
    
    print_separator()
    print(f"MCP Server Test Suite - Testing server at {args.host}:{args.port}")
    print_separator()
    
    tester = MCPServerTester(host=args.host, port=args.port, timeout=args.timeout)
    
    # Check if port is in use
    port_in_use = tester.check_port_in_use()
    if port_in_use:
        print(f"Port {args.port} is already in use.")
    else:
        print(f"Port {args.port} is available.")
    
    # If requested, restart the server
    if args.restart_server:
        print(f"\nRestarting server using {args.server_script}...")
        success = tester.restart_server(args.server_script)
        if success:
            print("Server restarted successfully.")
        else:
            print("Failed to restart server.")
    
    # If requested, run uvicorn directly
    if args.use_uvicorn:
        print(f"\nStarting server using Uvicorn with module {args.uvicorn_module}...")
        success = tester.run_uvicorn_directly(args.uvicorn_module)
        if success:
            print("Uvicorn server started successfully.")
        else:
            print("Failed to start Uvicorn server.")
    
    # Test server connection
    print("\nTesting server connection...")
    result = tester.test_server_connection()
    
    if not result["success"]:
        print(f"Failed to connect to MCP server: {result['error']}")
        print("\nChecking for server processes...")
        processes = tester.check_server_processes()
        if processes:
            print("Found the following Python processes:")
            for process in processes:
                print(f"- {process}")
        else:
            print("No Python processes found.")
        
        # If we're not running diagnostics, exit
        if not args.diagnostics:
            return 1
    else:
        manifest = result["manifest"]
        print(f"Connected to MCP server")
        print(f"Server: {manifest.get('server_name', 'Unknown')} v{manifest.get('version', '?')}")
        print(f"MCP Version: {manifest.get('mcp_version', '?')}")
        
        # Print available tools
        tools = manifest.get("tools", {})
        if tools:
            print(f"\nAvailable Tools ({len(tools)}):")
            for tool_name, tool_info in tools.items():
                print(f"  - {tool_name}: {tool_info.get('description', '')}")
        else:
            print("\nWarning: No tools available on the server")
        
        # Print available resources
        resources = manifest.get("resources", {})
        if resources:
            print(f"\nAvailable Resources ({len(resources)}):")
            for resource_name, resource_info in resources.items():
                print(f"  - {resource_name}: {resource_info.get('description', '')}")
        else:
            print("\nWarning: No resources available on the server")
    
    # Run full diagnostics if requested
    if args.diagnostics:
        print("\nRunning full diagnostics...")
        diagnostics = tester.run_full_diagnostics()
        
        # Print diagnostics results
        print("\nDiagnostics Results:")
        print(f"Port {args.port} in use: {diagnostics['port_in_use']}")
        
        if "connection" in diagnostics and diagnostics["connection"]["success"]:
            print("Server connection: Success")
            
            if "tools" in diagnostics:
                tools = diagnostics["tools"]
                print(f"Tools: {tools['success']}/{tools['total']} working")
            
            if "resources" in diagnostics:
                resources = diagnostics["resources"]
                print(f"Resources: {resources['success']}/{resources['total']} working")
        else:
            print("Server connection: Failed")
        
        # Print diagnostic details to a file
        with open("mcp_diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)
        print("\nDetailed diagnostics saved to mcp_diagnostics.json")
    
    print_separator()
    return 0

if __name__ == "__main__":
    sys.exit(main())
