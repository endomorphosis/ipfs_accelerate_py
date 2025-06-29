#!/usr/bin/env python3
"""
MCP Server Tools Diagnostic Script

This script helps diagnose issues with the MCP server tool registration and functionality.
It provides detailed diagnostics on why tools might not be showing up correctly.
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import subprocess
from typing import Dict, Any, List, Set, Optional
import importlib.util
import inspect

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mcp_diagnostic.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_diagnostic")

class MCPToolsDiagnostic:
    """Diagnose MCP server tool registration issues."""
    
    def __init__(self, host='localhost', port=8001, start_server=True):
        """Initialize the diagnostic tool."""
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.tools_endpoint = f"{self.base_url}/tools"
        self.server_proc = None
        self.server_log_file = "mcp_server_diagnostic.log"
        self.should_start_server = start_server
        
        # List of expected tool categories and specific tools to check
        self.expected_tools = {
            "ipfs": [
                "ipfs_add_file", "ipfs_cat", "ipfs_get_file", 
                "ipfs_files_ls", "ipfs_files_read", "ipfs_files_write",
                "ipfs_files_mkdir", "ipfs_files_rm", "ipfs_files_cp"
            ],
            "hardware": ["get_hardware_info", "get_hardware_capabilities"],
            "model": ["list_models", "model_inference", "throughput_benchmark"],
            "system": ["health_check"]
        }
        
        # MCP implementations to check
        self.mcp_implementations = [
            {"name": "unified_mcp_server", "path": "unified_mcp_server.py"},
            {"name": "simple_mcp_server", "path": "simple_mcp_server.py"},
            {"name": "direct_mcp_server", "path": "direct_mcp_server.py"},
            {"name": "enhanced_mcp_server", "path": "enhanced_mcp_server.py"},
            {"name": "robust_mcp_server", "path": "robust_mcp_server.py"}
        ]
    
    def kill_existing_servers(self):
        """Kill any existing MCP server processes."""
        try:
            logger.info("Killing any existing MCP server processes...")
            subprocess.run("pkill -f 'python.*mcp_server.py'", shell=True)
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error killing existing servers: {str(e)}")
    
    def start_server(self, server_path="unified_mcp_server.py"):
        """Start the MCP server for testing."""
        try:
            logger.info(f"Starting MCP server from {server_path}...")
            
            # Create log file
            log_file = open(self.server_log_file, "w")
            
            # Start server
            cmd = [
                sys.executable, 
                server_path, 
                "--port", str(self.port), 
                "--verbose"
            ]
            
            self.server_proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server is running
            if self.server_proc.poll() is not None:
                logger.error("Server failed to start")
                return False
            
            logger.info("Server started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}")
            return False
    
    def stop_server(self):
        """Stop the MCP server."""
        if hasattr(self, 'server_proc') and self.server_proc:
            logger.info("Stopping server...")
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
            time.sleep(2)
    
    def check_server_health(self) -> bool:
        """Check if the server is responsive."""
        try:
            health_url = f"{self.base_url}/health"
            logger.info(f"Checking server health at: {health_url}")
            
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info("Server is healthy")
                return True
            else:
                logger.error(f"Server returned non-200 status: {response.status_code}")
                logger.error(response.text)
                return False
        except requests.RequestException as e:
            logger.error(f"Server is not responsive: {str(e)}")
            return False
    
    def get_registered_tools(self) -> Dict[str, Any]:
        """Get the list of registered tools from the server."""
        try:
            logger.info(f"Requesting tools from {self.tools_endpoint}")
            response = requests.get(self.tools_endpoint, timeout=5)
            
            if response.status_code == 200:
                tools = response.json()
                logger.info(f"Received {len(tools)} registered tools")
                return tools
            else:
                logger.error(f"Error getting tools: {response.status_code}")
                logger.error(response.text)
                return {}
        except Exception as e:
            logger.error(f"Error getting tools: {str(e)}")
            return {}
    
    def categorize_found_tools(self, tools: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize the found tools by type."""
        categories = {cat: [] for cat in self.expected_tools.keys()}
        categories["unknown"] = []
        
        for tool_name in tools.keys():
            categorized = False
            for category, expected_tools in self.expected_tools.items():
                if any(expected in tool_name for expected in [category, category.replace("_", "")]):
                    categories[category].append(tool_name)
                    categorized = True
                    break
                
                # Check by specific tool name
                if tool_name in expected_tools:
                    categories[category].append(tool_name)
                    categorized = True
                    break
            
            if not categorized:
                categories["unknown"].append(tool_name)
        
        return categories
    
    def check_missing_tools(self) -> Dict[str, List[str]]:
        """Check which expected tools are missing."""
        tools = self.get_registered_tools()
        if not tools:
            return {cat: expected for cat, expected in self.expected_tools.items()}
        
        tool_names = set(tools.keys())
        missing = {}
        
        for category, expected_tools in self.expected_tools.items():
            missing_in_category = [tool for tool in expected_tools if tool not in tool_names]
            if missing_in_category:
                missing[category] = missing_in_category
        
        return missing
    
    def inspect_server_module(self, module_path: str):
        """Inspect the server module for tool registrations."""
        try:
            logger.info(f"Inspecting server module: {module_path}")
            
            # Load the module
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for register_tool decorator
            register_tool_fn = None
            for name, obj in inspect.getmembers(module):
                if name == "register_tool" and callable(obj):
                    register_tool_fn = obj
                    break
            
            if not register_tool_fn:
                logger.warning("No register_tool function found in the module")
                return
            
            # Find all functions decorated with register_tool
            registered_functions = []
            for name, obj in inspect.getmembers(module):
                if callable(obj) and hasattr(obj, "__wrapped__"):
                    registered_functions.append(name)
            
            logger.info(f"Found {len(registered_functions)} functions with decorators: {registered_functions}")
            
            # Look for bridge class
            bridge_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and "bridge" in name.lower():
                    bridge_class = obj
                    logger.info(f"Found potential bridge class: {name}")
                    
                    # Log methods in the bridge
                    bridge_methods = [m for m in dir(obj) if callable(getattr(obj, m)) and not m.startswith("_")]
                    logger.info(f"Bridge methods: {bridge_methods}")
                    break
            
        except Exception as e:
            logger.error(f"Error inspecting module {module_path}: {str(e)}")
    
    def extract_server_logs(self):
        """Extract and analyze the server logs."""
        try:
            if os.path.exists(self.server_log_file):
                with open(self.server_log_file, 'r') as f:
                    logs = f.read()
                
                logger.info("Server logs:")
                print("\n--- Server Logs ---")
                print(logs)
                print("--- End of Server Logs ---\n")
                
                # Look for specific issues in logs
                for line in logs.split("\n"):
                    if "error" in line.lower():
                        logger.warning(f"Found error in logs: {line}")
        except Exception as e:
            logger.error(f"Error reading server logs: {str(e)}")
    
    def check_bridge_functionality(self):
        """Check if the bridge to ipfs_accelerate_py is working."""
        try:
            # Try to import ipfs_accelerate_py
            logger.info("Checking for ipfs_accelerate_py module...")
            
            import importlib
            try:
                ipfs_accelerate = importlib.import_module("ipfs_accelerate_py")
                logger.info("Successfully imported ipfs_accelerate_py module")
                
                # Check what's available in the module
                logger.info(f"Available attributes: {dir(ipfs_accelerate)}")
                
            except ImportError:
                logger.warning("Could not import ipfs_accelerate_py directly")
                
                # Try different paths
                possible_paths = [
                    os.path.join(os.path.dirname(__file__), "ipfs_accelerate_py.py"),
                    os.path.join(os.path.dirname(__file__), "ipfs_accelerate_py", "ipfs_accelerate_py.py")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.info(f"Found IPFS Accelerate at {path}")
                        try:
                            spec = importlib.util.spec_from_file_location("ipfs_accelerate_py", path)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            logger.info(f"Successfully loaded from {path}")
                            logger.info(f"Available attributes: {dir(module)}")
                            break
                        except Exception as e:
                            logger.error(f"Error importing from {path}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error checking bridge functionality: {str(e)}")
    
    def test_specific_tool(self, tool_name: str) -> Optional[Dict]:
        """Test a specific tool by calling it."""
        try:
            call_url = f"{self.base_url}/mcp/tool"
            
            # Get tool schema first
            schema_url = f"{self.tools_endpoint}/{tool_name}"
            schema_resp = requests.get(schema_url)
            
            if schema_resp.status_code != 200:
                logger.error(f"Could not get schema for tool {tool_name}")
                return None
            
            schema = schema_resp.json()
            logger.info(f"Tool schema for {tool_name}: {json.dumps(schema, indent=2)}")
            
            # Prepare test parameters based on schema
            params = {}
            if "properties" in schema:
                for param_name, param_info in schema["properties"].items():
                    # Use default values where available
                    if param_name == "path":
                        params[param_name] = "/"
                    elif param_name == "cid":
                        params[param_name] = "QmTest123"
                    elif param_info.get("type") == "string":
                        params[param_name] = "test_value"
                    elif param_info.get("type") == "integer":
                        params[param_name] = 0
                    elif param_info.get("type") == "boolean":
                        params[param_name] = False
            
            # Call the tool
            payload = {
                "name": tool_name,
                "params": params
            }
            
            logger.info(f"Testing tool {tool_name} with params: {params}")
            response = requests.post(call_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Tool {tool_name} response: {json.dumps(result, indent=2)}")
                return result
            else:
                logger.error(f"Error calling tool {tool_name}: {response.status_code}")
                logger.error(response.text)
                return None
            
        except Exception as e:
            logger.error(f"Error testing tool {tool_name}: {str(e)}")
            return None
    
    def run_diagnostic(self, server_path="unified_mcp_server.py"):
        """Run the full diagnostic test."""
        success = True
        
        # Kill any existing servers
        self.kill_existing_servers()
        
        # Check server module statically
        self.inspect_server_module(server_path)
        
        # Check bridge functionality
        self.check_bridge_functionality()
        
        if self.should_start_server:
            # Start server
            if not self.start_server(server_path):
                logger.error("Failed to start server. Test aborted.")
                return False
        
        try:
            # Check server health
            if not self.check_server_health():
                logger.error("Server health check failed")
                success = False
            
            # Get registered tools
            registered_tools = self.get_registered_tools()
            
            if not registered_tools:
                logger.error("No tools returned from server")
                success = False
            else:
                # Categorize the found tools
                categorized_tools = self.categorize_found_tools(registered_tools)
                
                print("\nFound Tools by Category:")
                for category, tools in categorized_tools.items():
                    print(f"  {category.upper()}: {', '.join(tools) if tools else 'None'}")
                
                # Check for missing tools
                missing_tools = self.check_missing_tools()
                
                if missing_tools:
                    logger.warning("Missing expected tools:")
                    for category, tools in missing_tools.items():
                        logger.warning(f"  {category}: {', '.join(tools)}")
                    success = False
                else:
                    logger.info("All expected tools are registered")
                
                # Test a few key tools
                key_tools = ["health_check", "get_hardware_info"]
                if registered_tools:
                    tool_names = list(registered_tools.keys())
                    if tool_names:
                        key_tools.append(tool_names[0])  # Test first available tool
                
                for tool_name in key_tools:
                    if tool_name in registered_tools:
                        self.test_specific_tool(tool_name)
        
        except Exception as e:
            logger.error(f"Diagnostic error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            success = False
        
        finally:
            # Extract server logs
            self.extract_server_logs()
            
            # Stop server if we started it
            if self.should_start_server:
                self.stop_server()
        
        return success
    
    def test_all_implementations(self):
        """Test all MCP server implementations."""
        results = {}
        
        for impl in self.mcp_implementations:
            if os.path.exists(impl["path"]):
                logger.info(f"\n\n===== Testing {impl['name']} =====")
                print(f"\n===== Testing {impl['name']} =====")
                
                result = self.run_diagnostic(impl["path"])
                results[impl["name"]] = result
            else:
                logger.info(f"{impl['path']} not found, skipping")
        
        # Print summary of results
        print("\n===== Test Results =====")
        for impl, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{impl}: {status}")
        
        return all(results.values())

def main():
    """Run the MCP diagnostic tool."""
    parser = argparse.ArgumentParser(description='Diagnose MCP Server Tool Registration')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8001, help='Server port')
    parser.add_argument('--no-start', action='store_true', help='Do not start the server (assume it is already running)')
    parser.add_argument('--all', action='store_true', help='Test all MCP server implementations')
    parser.add_argument('--server', default='unified_mcp_server.py', help='Server implementation to test')
    args = parser.parse_args()
    
    diagnostic = MCPToolsDiagnostic(
        host=args.host,
        port=args.port,
        start_server=not args.no_start
    )
    
    print("==================================================")
    print("MCP Server Tool Registration Diagnostic")
    print("==================================================")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Server: {args.server}")
    print("==================================================\n")
    
    if args.all:
        success = diagnostic.test_all_implementations()
    else:
        success = diagnostic.run_diagnostic(args.server)
    
    print("\n==================================================")
    if success:
        print("✅ Diagnostic completed successfully")
    else:
        print("❌ Issues were detected during diagnosis")
    print("==================================================")
    
    logger.info("See mcp_diagnostic.log for full diagnostic details")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
