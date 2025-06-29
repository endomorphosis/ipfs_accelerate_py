#!/usr/bin/env python3
"""
Comprehensive test script for the IPFS Accelerate MCP server.
This script tests the connectivity to the MCP server and verifies that the required tools are available.
It also provides automatic fixing capability to ensure all required tools are properly registered.
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import subprocess
import socket
import re # Import re for regex
from typing import Dict, List, Any, Tuple, Optional
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mcp_test_comprehensive.log')
    ]
)
logger = logging.getLogger("mcp_test")

class MCPServerTester:
    """Tests an MCP server for connectivity and functionality."""
    
    def __init__(self, initial_host: str = "127.0.0.1", initial_port: int = 8002, timeout: int = 5):
        """Initialize the tester with server connection details."""
        self.initial_host = initial_host
        self.initial_port = initial_port
        self.host = initial_host # Current host, might be updated
        self.port = initial_port # Current port, might be updated
        self.timeout = timeout
        self.base_url = None # Will be set after port discovery
        self.tools_url = None
        self.resources_url = None
        self.manifest_url = None
        self.direct_tool_url_template = None
        self.server_process = None
        self.manifest_data = None
        
        # Initialize URLs
        self._update_urls()

    def _update_urls(self):
        """Update URL attributes based on current host and port."""
        self.base_url = f"http://{self.host}:{self.port}"
        self.tools_url = f"{self.base_url}/tools"
        self.resources_url = f"{self.base_url}/resources"
        self.manifest_url = f"{self.base_url}/mcp/manifest"
        self.direct_tool_url_template = f"{self.base_url}/mcp/tool/{{}}"
        
    def discover_server_port(self) -> bool:
        """
        Discover the port of a running final_mcp_server.py process.
        Updates self.port and self._update_urls() if found.
        Returns True if a process is found and port is updated, False otherwise.
        """
        logger.info("Attempting to discover running server port...")
        processes = self.check_running_processes()
        
        for proc in processes:
            cmd = proc.get("command", "")
            # Look for final_mcp_server.py and extract --port argument
            match = re.search(r"final_mcp_server\.py.*?--port\s+(\d+)", cmd)
            if match:
                discovered_port = int(match.group(1))
                logger.info(f"Discovered server running on port: {discovered_port}")
                self.port = discovered_port
                self._update_urls()
                logger.info(f"Updated tester to use port: {self.port}")
                return True
                
        logger.info("No running final_mcp_server.py process found with a specified port.")
        # Fallback to initial port if no process found
        self.port = self.initial_port
        self._update_urls()
        logger.info(f"Using initial port: {self.port}")
        return False

    def check_port_availability(self) -> bool:
        """Check if the port is available (not in use)."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((self.host, self.port))
                if result == 0:
                    logger.info(f"Port {self.port} is in use (something is already listening)")
                    return False
                else:
                    logger.info(f"Port {self.port} is available")
                    return True
        except Exception as e:
            logger.error(f"Error checking port availability: {e}")
            return False
    
    def check_running_processes(self) -> List[Dict[str, Any]]:
        """Check for any MCP processes already running."""
        processes = []
        try:
            # This works on Linux/macOS
            cmd = "ps aux | grep -E 'mcp|fixed_standards_mcp_server|final_mcp_server' | grep -v grep" # Added final_mcp_server
            output = subprocess.check_output(cmd, shell=True, text=True)
            for line in output.splitlines():
                fields = line.split()
                if len(fields) >= 11:
                    pid = fields[1]
                    cmd = ' '.join(fields[10:])
                    processes.append({"pid": pid, "command": cmd})
                    logger.info(f"Found MCP process: PID {pid} - {cmd}")
        except subprocess.CalledProcessError:
            logger.info("No MCP processes found running")
        except Exception as e:
            logger.error(f"Error checking running processes: {e}")
        
        return processes
    
    def get_manifest(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Get the MCP manifest which contains information about available tools.
        Retries multiple times to account for server startup time.
        """
        max_retries = 15 # Increased retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Fetching MCP manifest from {self.manifest_url}")
                response = requests.get(self.manifest_url, timeout=self.timeout)
                
                if response.status_code == 200:
                    manifest = response.json()
                    logger.info(f"Successfully retrieved manifest")
                    self.manifest_data = manifest
                    
                    # Extract available tools from manifest
                    if "tools" in manifest:
                        tools = manifest["tools"] # Expecting a list of tool names
                        if isinstance(tools, list):
                            logger.info(f"Found {len(tools)} tools in manifest: {', '.join(tools)}")
                            return True, manifest
                        else:
                            logger.warning(f"Manifest 'tools' key is not a list (attempt {attempt + 1}): {type(tools)}")
                            # Continue retrying if the format is unexpected
                    else:
                        logger.warning(f"Manifest does not contain 'tools' information (attempt {attempt + 1})")
                        # Continue retrying if tools key is missing
                else:
                    logger.warning(f"Failed to get manifest: Status {response.status_code} (attempt {attempt + 1})")
                    # Continue retrying on non-200 status codes
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error fetching manifest (attempt {attempt + 1}): {e}")
                # Continue retrying on connection errors
            except Exception as e:
                logger.error(f"Error fetching manifest (attempt {attempt + 1}): {e}")
                # Continue retrying on other exceptions

            time.sleep(1) # Wait before retrying

        logger.error(f"Failed to get manifest after {max_retries} attempts.")
        return False, None
    
    def start_server(self) -> bool:
        """Start the MCP server using the start_final_solution.sh script."""
        if not self.check_port_availability():
            logger.warning("Port already in use. Will attempt to use the existing server.")
            return True
        
        try:
            logger.info("Starting MCP server...")
            cmd = ["bash", "run_final_solution.sh", "--host", self.host, "--port", str(self.port)]
            
            # Start in background mode with output to log file
            with open("mcp_server_test.log", "w") as log_file:
                self.server_process = subprocess.Popen(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT
                )
            
            # Wait for server to start
            logger.info("Waiting for server to start...")
            time.sleep(5)  # Adjust as needed
            
            # Check if process is still running
            if self.server_process.poll() is not None:
                logger.error(f"Server process exited with code {self.server_process.returncode}")
                return False
                
            logger.info(f"Server started with PID {self.server_process.pid}")
            return True
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            return False
    
    def check_server_connection(self) -> bool:
        """
        Check if we can connect to the MCP server.
        Retries multiple times to account for server startup time.
        """
        max_retries = 15 # Increased retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Checking server connection at {self.base_url}")
                
                # Try the manifest endpoint which should be available
                manifest_response = requests.get(self.manifest_url, timeout=self.timeout)
                if manifest_response.status_code == 200:
                    logger.info("Successfully connected to the manifest endpoint")
                    return True
                else:
                    logger.warning(f"Manifest endpoint returned status code: {manifest_response.status_code} (attempt {attempt + 1})")
                    # Continue retrying on non-200 status codes
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error checking server connection (attempt {attempt + 1}): {e}")
                # Continue retrying on connection errors
            except Exception as e:
                logger.error(f"Error checking server connection (attempt {attempt + 1}): {e}")
                # Continue retrying on other exceptions

            time.sleep(1) # Wait before retrying

        logger.error(f"Failed to connect to server after {max_retries} attempts.")
        return False
    
    def list_tools(self) -> Tuple[bool, List[str]]:
        """Get the list of tools available on the server from the manifest."""
        success, manifest = self.get_manifest()
        
        if success and manifest and "tools" in manifest:
            # The server now returns a list of tool names directly under "tools"
            tools = manifest["tools"]
            if isinstance(tools, list):
                return True, tools
            else:
                logger.error(f"Manifest 'tools' key is not a list: {type(tools)}")
                return False, []
        else:
            logger.error("Failed to get tools list from manifest")
            return False, []
    
    def test_tool(self, tool_name: str, params: Dict[str, Any] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Test a specific tool with optional parameters using the JSON-RPC endpoint.
        Retries multiple times to account for server startup time or transient errors.
        """
        max_retries = 5 # Retries for individual tool tests
        for attempt in range(max_retries):
            try:
                jsonrpc_url = f"{self.base_url}/jsonrpc"
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Testing tool: {tool_name} via JSON-RPC at {jsonrpc_url}")
                
                if params is None:
                    params = {}
                
                params: Dict[str, Any] = params # Explicit type hint for clarity and Pylance

                # Construct JSON-RPC request
                jsonrpc_request = {
                    "jsonrpc": "2.0",
                    "method": tool_name,
                    "params": params,
                    "id": 1 # Use a simple ID
                }
                
                response = requests.post(jsonrpc_url, json=jsonrpc_request, timeout=self.timeout)
                
                if response.status_code == 200:
                    jsonrpc_response = response.json()
                    
                    if "error" in jsonrpc_response:
                        error = jsonrpc_response["error"]
                        logger.warning(f"Tool {tool_name} test failed via JSON-RPC with error: {error} (attempt {attempt + 1})")
                        return False, error # Return False immediately on error
                    elif "result" in jsonrpc_response:
                        result = jsonrpc_response["result"]
                        logger.info(f"Tool {tool_name} test succeeded via JSON-RPC.")
                        return True, result # Return True on success
                    else:
                        logger.warning(f"Tool {tool_name} test failed via JSON-RPC with unexpected response format: {jsonrpc_response} (attempt {attempt + 1})")
                        return False, {"error": "Unexpected JSON-RPC response format"} # Return False for unexpected format
                else:
                    logger.warning(f"Tool {tool_name} test failed via JSON-RPC with status {response.status_code}: {response.text} (attempt {attempt + 1})")
                    return False, {"error": f"HTTP status code {response.status_code}"} # Return False on non-200 status codes
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error testing tool {tool_name} (attempt {attempt + 1}): {e}")
                return False, {"error": f"Connection error: {e}"} # Return False on connection errors
            except Exception as e:
                logger.error(f"Error testing tool {tool_name} (attempt {attempt + 1}): {e}")
                return False, {"error": f"Unexpected error: {e}"} # Return False on other exceptions

            time.sleep(1) # Wait before retrying - This line will now only be reached if the loop continues for other reasons, but the primary failure paths now return immediately.

        logger.error(f"Tool {tool_name} test failed after {max_retries} attempts.")
        return False, None # This return is for when all retries are exhausted without a definitive success or immediate failure.
    
    def check_ipfs_connectivity(self) -> bool:
        """Check if IPFS is connected and functional using the health_check tool."""
        try:
            logger.info("Checking IPFS connectivity...")
            success, health_status = self.test_tool('health_check')
            if success and health_status and 'ipfs_connected' in health_status:
                connected = health_status['ipfs_connected']
                logger.info(f"IPFS connected: {connected}")
                return connected
            return False
        except Exception as e:
            logger.error(f"Error checking IPFS connectivity: {e}")
            return False

    def check_hardware_acceleration(self) -> Dict[str, bool]:
        """Check hardware acceleration availability using the get_hardware_info tool."""
        try:
            logger.info("Checking hardware acceleration...")
            success, hardware_info = self.test_tool('get_hardware_info')
            if success and hardware_info:
                accelerators = hardware_info.get('accelerators', {})
                results = {hw: info.get('available', False) for hw, info in accelerators.items()}
                logger.info(f"Hardware acceleration status: {results}")
                return results
            return {}
        except Exception as e:
            logger.error(f"Error checking hardware acceleration: {e}")
            return {}

    def test_required_tools(self) -> Dict[str, bool]:
        """Test the required tools for IPFS Accelerate MCP integration."""
        required_tools = {
            "health_check": {},
            "get_hardware_info": {},
            "ipfs_add_file": {"path": "/tmp/test.txt"},
            "ipfs_cat": {"cid": "QmTest"},
            "ipfs_files_write": {"path": "/test.txt", "content": "Hello, world!"},
            "ipfs_files_read": {"path": "/test.txt"},
            "list_models": {},
            "create_endpoint": {"model_name": "test-model"},
            "run_inference": {"endpoint_id": "test-endpoint", "inputs": ["test input"]}
        }
        
        results = {}
        
        # First verify which tools are available from the manifest
        success, available_tools = self.list_tools()
        if not success:
            logger.warning("Failed to get list of available tools from manifest, proceeding with direct testing")
        
        # Test each required tool
        for tool_name, params in required_tools.items():
            if success and tool_name not in available_tools:
                logger.warning(f"Tool '{tool_name}' not found in manifest, will try testing directly")
            
            logger.info(f"Testing tool: {tool_name}")
            tool_success, _ = self.test_tool(tool_name, params)
            logger.info(f"test_tool for {tool_name} returned: {tool_success}") # Add this line
            results[tool_name] = tool_success
            
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run a comprehensive test suite on the MCP server."""
        results = {
            "timestamp": time.time(),
            "server": {
                "host": self.host,
                "port": self.port,
                "reachable": False
            },
            "manifest": None,
            "tools": {
                "available": [],
                "required_tools_working": {}
            },
            "hardware_info": {
                "accelerators": {}
            }
        }
        
        # Discover the server port before testing
        self.discover_server_port()

        # Test server connection
        results["server"]["reachable"] = self.check_server_connection()
        if not results["server"]["reachable"]:
            logger.error("Server is not reachable, cannot proceed with tests")
            return results
        
        # Get manifest and tools list
        manifest_success, manifest = self.get_manifest()
        if manifest_success:
            results["manifest"] = manifest
        
        # Get available tools using the corrected list_tools method
        tools_list_success, available_tools = self.list_tools()
        if tools_list_success:
            results["tools"]["available"] = available_tools
        else:
            logger.warning("Failed to get available tools list using list_tools method.")

        # Test required tools
        results["tools"]["required_tools_working"] = self.test_required_tools()
        
        # Check hardware acceleration
        results["hardware_info"]["accelerators"] = self.check_hardware_acceleration()
        
        # Output summary
        self._print_test_summary(results)
        return results
    
    def _print_test_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the test results."""
        print("\n" + "=" * 60)
        print("MCP SERVER TEST SUMMARY")
        print("=" * 60)
        print(f"Server: {self.host}:{self.port}")
        print(f"Reachable: {'Yes' if results['server']['reachable'] else 'No'}")
        
        if results["server"]["reachable"]:
            available_tools = results["tools"].get("available", [])
            print(f"Available Tools: {len(available_tools)}")
            for tool in available_tools:
                print(f"  - {tool}")
            
            print("\nRequired Tools Status:")
            for tool, status in results["tools"]["required_tools_working"].items():
                print(f"  - {tool}: {'Working' if status else 'Not working'}")
            
            print("\nHardware Acceleration:")
            for acc, available in results["hardware_info"]["accelerators"].items():
                print(f"  - {acc}: {'Available' if available else 'Not available'}")
        
        print("=" * 60)
    
    def save_results(self, results: Dict[str, Any], output_file: str = "mcp_test_results.json") -> None:
        """Save test results to a JSON file."""
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
        
    def stop_server(self) -> None:
        """Stop the MCP server if it was started by this script."""
        if self.server_process:
            try:
                logger.info(f"Stopping MCP server (PID: {self.server_process.pid})...")
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                logger.info("Server stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
                try:
                    import signal
                    os.kill(self.server_process.pid, signal.SIGKILL)
                    logger.info("Server had to be killed with SIGKILL")
                except Exception as e2:
                    logger.error(f"Failed to kill server: {e2}")
    
    def fix_mcp_server(self) -> bool:
        """
        Fix common issues with the MCP server:
        1. Add standard API endpoints for compatibility
        2. Ensure all required tools are properly registered
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Attempting to fix MCP server issues")
        
        # Define the required files
        mcp_server_path = os.path.join(os.getcwd(), "mcp/server.py")
        final_server_path = os.path.join(os.getcwd(), "final_mcp_server.py")
        
        success = True
        
        # 1. Fix MCP server endpoints
        if os.path.exists(mcp_server_path):
            try:
                logger.info(f"Adding standard API endpoints to {mcp_server_path}")
                
                # Create backup
                backup_path = f"{mcp_server_path}.bak"
                with open(mcp_server_path, 'r') as f:
                    original_content = f.read()
                    
                with open(backup_path, 'w') as f:
                    f.write(original_content)
                
                # Check if endpoints already exist
                if "@app.get(\"/tools\")" in original_content:
                    logger.info("Standard API endpoints already exist")
                else:
                    # Find insertion point
                    insertion_point = "# MCP manifest endpoint"
                    if insertion_point not in original_content:
                        insertion_point = "@app.get(\"/mcp/manifest\")"
                    
                    if insertion_point in original_content:
                        # Create new endpoints for compatibility
                        new_endpoints = """
# Standard API endpoints for compatibility
@app.get("/tools")
def get_tools_list():
    '''Return a list of all available tools'''
    return list(_tools.keys())

@app.post("/tools/{tool_name}/invoke")
async def invoke_tool_compat(tool_name: str, request: Request):
    '''Tool invocation endpoint compatible with standard clients'''
    # Reuse the existing tool endpoint logic
    return await call_tool(tool_name, request)

"""
                        modified_content = original_content.replace(insertion_point, new_endpoints + insertion_point)
                        
                        # Write the updated content
                        with open(mcp_server_path, 'w') as f:
                            f.write(modified_content)
                        
                        logger.info(f"Added standard API endpoints to {mcp_server_path}")
                    else:
                        logger.error("Could not find a suitable insertion point in the MCP server file")
                        success = False
            except Exception as e:
                logger.error(f"Failed to fix MCP server endpoints: {e}")
                success = False
        else:
            logger.warning(f"MCP server file not found: {mcp_server_path}")
            success = False
        
        # 2. Fix missing tools in the final_mcp_server.py
        if os.path.exists(final_server_path):
            try:
                logger.info(f"Checking for missing tools in {final_server_path}")
                
                # Define required tools
                required_tools = {
                    "get_hardware_info": "Get hardware information about the system",
                    "health_check": "Check the health of the IPFS Accelerate MCP server",
                    "ipfs_add_file": "Add a file to IPFS and return its CID",
                    "ipfs_cat": "Retrieve content from IPFS by its CID",
                    "ipfs_files_write": "Write content to the IPFS Mutable File System (MFS)",
                    "ipfs_files_read": "Read content from the IPFS Mutable File System (MFS)",
                    "list_models": "lambda: ipfs_accelerate.list_models()",
                    "create_endpoint": "lambda model_name, device='cpu', max_batch_size=16: ipfs_accelerate.create_endpoint(model_name, device, max_batch_size)",
                    "run_inference": "lambda endpoint_id, inputs: ipfs_accelerate.run_inference(endpoint_id, inputs)"
                }
                
                # Create backup
                backup_path = f"{final_server_path}.bak"
                with open(final_server_path, 'r') as f:
                    original_content = f.read()
                    
                with open(backup_path, 'w') as f:
                    f.write(original_content)
                
                # Find missing tools
                missing_tools = {}
                for tool_name, description in required_tools.items():
                    if f'"name": "{tool_name}"' not in original_content and f"'name': '{tool_name}'" not in original_content:
                        missing_tools[tool_name] = description
                
                if missing_tools:
                    logger.info(f"Found {len(missing_tools)} missing tools: {', '.join(missing_tools.keys())}")
                    
                    # Find the tools list
                    tools_start = original_content.find("tools = [")
                    tools_end = original_content.find("    ]", tools_start)
                    
                    if tools_start < 0 or tools_end < 0:
                        logger.error("Could not find tools list in the server file")
                        return False
                    
                    # Generate new tools
                    new_tools = []
                    
                    for tool_name, description in missing_tools.items():
                        func_map = {
                            "health_check": "lambda: ipfs_accelerate.health_check()",
                            "get_hardware_info": "lambda: ipfs_accelerate.get_hardware_info()",
                            "ipfs_add_file": "lambda path: ipfs_accelerate.add_file(path)",
                            "ipfs_cat": "lambda cid: ipfs_accelerate.cat(cid)",
                            "ipfs_files_write": "lambda path, content: ipfs_accelerate.files_write(path, content)",
                            "ipfs_files_read": "lambda path: ipfs_accelerate.files_read(path)",
                            "list_models": "lambda: ipfs_accelerate.list_models()",
                            "create_endpoint": "lambda model_name, device='cpu', max_batch_size=16: ipfs_accelerate.create_endpoint(model_name, device, max_batch_size)",
                            "run_inference": "lambda endpoint_id, inputs: ipfs_accelerate.run_inference(endpoint_id, inputs)"
                        }
                        
                        func = func_map.get(tool_name, "lambda: {}")
                        
                        new_tools.append(f"""        {{
            "name": "{tool_name}",
            "description": "{description}",
            "function": {func}
        }}""")
                    
                    # Insert new tools
                    new_tools_str = ",\n".join(new_tools)
                    modified_content = original_content[:tools_end] + ",\n" + new_tools_str + "\n    " + original_content[tools_end:]
                    
                    # Write the updated content
                    with open(final_server_path, 'w') as f:
                        f.write(modified_content)
                    
                    logger.info(f"Added {len(missing_tools)} missing tools to {final_server_path}")
                else:
                    logger.info("All required tools are already present")
            except Exception as e:
                logger.error(f"Failed to fix missing tools: {e}")
                success = False
        else:
            logger.warning(f"Final server file not found: {final_server_path}")
            success = False
        
        # 3. Create restart script if not exists
        restart_script_path = os.path.join(os.getcwd(), "restart_mcp_server.sh")
        if not os.path.exists(restart_script_path):
            try:
                logger.info(f"Creating server restart script: {restart_script_path}")
                
                restart_script_content = """#!/bin/bash
#
# IPFS Accelerate MCP - Server Restart Script
#

# Kill any running MCP server processes
echo "Stopping existing MCP server processes..."
pkill -f "final_mcp_server.py" || true
pkill -f "uvicorn.*mcp" || true

# Wait for processes to terminate
sleep 2

# Validate that the processes are really stopped
if pgrep -f "final_mcp_server.py" > /dev/null || pgrep -f "uvicorn.*mcp" > /dev/null; then
  echo "Forcing termination of remaining processes..."
  pkill -9 -f "final_mcp_server.py" || true
  pkill -9 -f "uvicorn.*mcp" || true
  sleep 1
fi

# Start the server again with run_final_solution.sh
echo "Starting MCP server..."
./run_final_solution.sh "$@" &

echo "Server restarting in background. Check logs for status."
"""
                
                with open(restart_script_path, 'w') as f:
                    f.write(restart_script_content)
                
                # Make executable
                os.chmod(restart_script_path, 0o755)
                logger.info(f"Created server restart script: {restart_script_path}")
            except Exception as e:
                logger.error(f"Failed to create server restart script: {e}")
                success = False
        
        return success

    def auto_fix_and_restart(self, results: Dict[str, Any], restart: bool = True) -> bool:
        """Automatically fix issues based on test results and restart the server if needed."""
        logger.info(f"auto_fix_and_restart received results: {results}")
        logger.info("Analyzing test results for fixable issues...")
        
        needs_fixing = False
        needs_restart = False
        
        # Check if server is reachable
        if not results["server"]["reachable"]:
            logger.warning("Server is not reachable, cannot diagnose further")
            return False
        
        # Check for missing tools
        required_tools_status = results["tools"]["required_tools_working"]
        missing_tools = [tool for tool, status in required_tools_status.items() if not status]
        
        if missing_tools:
            logger.info(f"Found {len(missing_tools)} missing or non-functional tools: {', '.join(missing_tools)}")
            needs_fixing = True
        
        # Check for standard endpoints
        if needs_fixing:
            logger.info("Applying fixes to MCP server...")
            fix_success = self.fix_mcp_server()
            
            if fix_success:
                logger.info("Successfully applied fixes to MCP server")
                needs_restart = True
            else:
                logger.error(f"Failed to apply some fixes to MCP server: {fix_success}")
        else:
            logger.info("No issues requiring auto-fixing detected")
            
        # Restart server if needed and requested
        if needs_restart and restart:
            logger.info("Restarting MCP server to apply changes...")
            
            restart_script = "./restart_mcp_server.sh"
            if os.path.exists(restart_script):
                try:
                    subprocess.run(["bash", restart_script], check=True)
                    logger.info("Server restart initiated")
                    
                    # Wait for server to become available again
                    logger.info("Waiting for server to become available...")
                    for i in range(10):
                        time.sleep(1)
                        if self.check_server_connection():
                            logger.info("Server is now available after restart")
                            return True
                    
                    logger.error("Server did not become available after restart")
                    return False
                except Exception as e:
                    logger.error(f"Error restarting server: {e}")
                    return False
            else:
                logger.error(f"Restart script not found: {restart_script}")
                return False
        
        return True


def main():
    """Main entry point for the test script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run comprehensive tests on the IPFS Accelerate MCP server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    parser.add_argument("--start-server", action="store_true", help="Start the server if not already running")
    parser.add_argument("--output", type=str, default="mcp_test_results.json", help="Output file for test results")
    parser.add_argument("--timeout", type=int, default=5, help="Request timeout in seconds")
    parser.add_argument("--auto-fix", action="store_true", help="Automatically fix discovered issues")
    parser.add_argument("--no-restart", action="store_true", help="Skip server restart after fixes")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = MCPServerTester(initial_host=args.host, initial_port=args.port, timeout=args.timeout)
    
    # Start server if requested
    if args.start_server:
        logger.info("Starting MCP server as requested")
        if not tester.start_server():
            return 1
    
    try:
        # Check if there are existing MCP processes
        processes = tester.check_running_processes()
        if not processes and not args.start_server:
            logger.warning("No MCP processes found running. Use --start-server to start one.")
        
        # Run comprehensive tests
        results = tester.run_comprehensive_test()
        
        # Save results to file
        tester.save_results(results, args.output)
        
        # Auto-fix issues if requested
        if args.auto_fix:
            logger.info("Auto-fix flag set, attempting to fix discovered issues...")
            tester.auto_fix_and_restart(results, not args.no_restart)
            
            # Run tests again after fixes
            if not args.no_restart:
                logger.info("Running tests again after applying fixes...")
                time.sleep(3)  # Give the server a moment to fully start
                post_fix_results = tester.run_comprehensive_test()
                tester.save_results(post_fix_results, f"post_fix_{args.output}")
        
        # Return success if all required tools are working
        all_required_tools_working = all(results["tools"]["required_tools_working"].values())
        
        if all_required_tools_working:
            logger.info("All required tools are working correctly.")
            return 0
        else:
            logger.error("Some required tools are NOT working correctly.")
            # Find which tools failed
            failed_tools = [tool for tool, status in results["tools"]["required_tools_working"].items() if not status]
            logger.error(f"Failed tools: {', '.join(failed_tools)}")
            return 1
    finally:
        # Stop server if we started it
        if args.start_server:
            tester.stop_server()


if __name__ == "__main__":
    sys.exit(main())
