#!/usr/bin/env python3
"""
Enhanced MCP Server Verification Tool

This script provides comprehensive testing and verification of
the IPFS Accelerate MCP server implementation.
"""

import os
import sys
import json
import time
import requests
import logging
import argparse
import socket
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_verify.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced-verify")

# Default settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8002
BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
MCP_URL = f"{BASE_URL}/mcp"

class EnhancedMCPVerifier:
    """Enhanced verification for MCP server."""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: int = 5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        self.mcp_url = f"{self.base_url}/mcp"
        self.server_process = None
        self.started_server = False
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "server_info": {
                "host": host,
                "port": port,
                "server_reachable": False,
                "port_open": False
            },
            "api_tests": {
                "manifest_available": False,
                "tools_endpoint_available": False,
                "health_endpoint_available": False
            },
            "tool_tests": {},
            "diagnostics": {},
            "fixes_applied": []
        }
    
    def is_port_open(self) -> bool:
        """Check if the port is open on the host."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                result = s.connect_ex((self.host, self.port))
                self.results["server_info"]["port_open"] = (result == 0)
                return result == 0
        except Exception as e:
            logger.error(f"Error checking port: {e}")
            self.results["server_info"]["port_open"] = False
            return False
    
    def start_server(self, debug: bool = True) -> bool:
        """Start the MCP server if it's not already running."""
        # Check if server is already running
        if self.is_port_open():
            logger.info(f"Port {self.port} already in use. Checking if it's the MCP server...")
            if self.test_server_reachable():
                logger.info("MCP server is already running")
                return True
            else:
                logger.warning(f"Port {self.port} is in use but it's not the MCP server")
                return False
        
        # Start the server
        logger.info("Starting MCP server...")
        
        cmd = [
            sys.executable,
            "final_mcp_server.py",
            "--host", self.host,
            "--port", str(self.port)
        ]
        
        if debug:
            cmd.append("--debug")
        
        try:
            # Create log file for server output
            log_file = open("enhanced_verify_server.log", "w")
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for server to start
            logger.info("Waiting for server to start...")
            for _ in range(10):
                time.sleep(1)
                if self.is_port_open():
                    if self.test_server_reachable():
                        logger.info("MCP server started successfully")
                        self.started_server = True
                        return True
            
            logger.error("Failed to start MCP server")
            return False
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the server if we started it."""
        if self.started_server and self.server_process:
            logger.info("Stopping MCP server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                logger.info("MCP server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("MCP server didn't stop gracefully, killing it...")
                self.server_process.kill()
            self.server_process = None
    
    def test_server_reachable(self) -> bool:
        """Test if the server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            reachable = response.status_code == 200
            self.results["server_info"]["server_reachable"] = reachable
            self.results["api_tests"]["health_endpoint_available"] = reachable
            return reachable
        except Exception as e:
            logger.error(f"Error checking server reachability: {e}")
            self.results["server_info"]["server_reachable"] = False
            self.results["api_tests"]["health_endpoint_available"] = False
            return False
    
    def test_manifest(self) -> bool:
        """Test if the manifest is available."""
        try:
            response = requests.get(f"{self.mcp_url}/manifest", timeout=self.timeout)
            manifest_ok = response.status_code == 200
            self.results["api_tests"]["manifest_available"] = manifest_ok
            
            if manifest_ok:
                manifest_data = response.json()
                self.results["diagnostics"]["manifest"] = manifest_data
                
                # Extract available tools
                if "tools" in manifest_data:
                    self.results["diagnostics"]["available_tools"] = list(manifest_data["tools"].keys())
                
                return True
            else:
                logger.error(f"Manifest request failed with status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking manifest: {e}")
            self.results["api_tests"]["manifest_available"] = False
            return False
    
    def test_tools_endpoint(self) -> bool:
        """Test if the tools endpoint is available."""
        try:
            response = requests.get(f"{self.mcp_url}/tools", timeout=self.timeout)
            tools_ok = response.status_code == 200
            self.results["api_tests"]["tools_endpoint_available"] = tools_ok
            
            if tools_ok:
                self.results["diagnostics"]["tools_list"] = response.json()
            else:
                logger.error(f"Tools endpoint request failed with status {response.status_code}")
            
            return tools_ok
        except Exception as e:
            logger.error(f"Error checking tools endpoint: {e}")
            self.results["api_tests"]["tools_endpoint_available"] = False
            return False
    
    def test_tool(self, tool_name: str, parameters: Dict[str, Any] = None) -> bool:
        """Test a specific tool."""
        parameters = parameters or {}
        
        try:
            response = requests.post(
                f"{self.mcp_url}/call_tool", 
                json={"tool_name": tool_name, "parameters": parameters},
                timeout=self.timeout
            )
            
            success = response.status_code == 200
            self.results["tool_tests"][tool_name] = success
            
            if success:
                self.results["diagnostics"][f"tool_{tool_name}"] = response.json()
                logger.info(f"Tool {tool_name} test: PASSED")
            else:
                logger.error(f"Tool {tool_name} test failed with status {response.status_code}")
                try:
                    error_data = response.json()
                    logger.error(f"Error details: {error_data}")
                    self.results["diagnostics"][f"tool_{tool_name}_error"] = error_data
                except:
                    logger.error(f"Error details not available")
            
            return success
        except Exception as e:
            logger.error(f"Error testing {tool_name} tool: {e}")
            self.results["tool_tests"][tool_name] = False
            return False
    
    def test_standard_tools(self) -> Dict[str, bool]:
        """Test all standard tools."""
        standard_tools = [
            {"name": "get_hardware_info", "params": {}},
            {"name": "ipfs_node_info", "params": {}},
            {"name": "list_models", "params": {}}
        ]
        
        results = {}
        for tool in standard_tools:
            success = self.test_tool(tool["name"], tool["params"])
            results[tool["name"]] = success
        
        return results
    
    def apply_fixes(self) -> List[str]:
        """Apply fixes to common issues."""
        fixes_applied = []
        
        # Try to fix tool registration issues
        if not all(self.results["tool_tests"].values()):
            logger.info("Attempting to fix tool registration issues...")
            try:
                result = subprocess.run(
                    [sys.executable, "fix_mcp_tool_registration.py", "--autofix"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                logger.info("Tool registration fix applied")
                fixes_applied.append("tool_registration")
                
                # Save the output for debugging
                self.results["diagnostics"]["tool_registration_fix"] = {
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to apply tool registration fix: {e}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
        
        return fixes_applied
    
    def verify(self, autofix: bool = False) -> Dict[str, Any]:
        """Run verification and return results."""
        try:
            logger.info("Starting MCP server verification...")
            
            # Check if server is already running or start it
            if not self.is_port_open() or not self.test_server_reachable():
                if not self.start_server(debug=True):
                    self.results["error"] = f"Failed to start or connect to MCP server at {self.host}:{self.port}"
                    return self.results
            
            # Test API endpoints
            self.test_manifest()
            self.test_tools_endpoint()
            
            # Test standard tools
            self.test_standard_tools()
            
            # Apply fixes if needed and autofix is enabled
            if autofix and not all(self.results["tool_tests"].values()):
                fixes = self.apply_fixes()
                if fixes:
                    self.results["fixes_applied"] = fixes
                    
                    # Re-test after fixes
                    logger.info("Re-testing after applying fixes...")
                    self.test_manifest()
                    self.test_tools_endpoint()
                    self.test_standard_tools()
            
            # Calculate overall status
            self.results["all_tests_passed"] = (
                self.results["server_info"]["server_reachable"] and
                self.results["api_tests"]["manifest_available"] and
                self.results["api_tests"]["tools_endpoint_available"] and
                all(self.results["tool_tests"].values())
            )
            
            logger.info(f"Verification completed. All tests passed: {self.results['all_tests_passed']}")
            return self.results
            
        finally:
            # Stop server if we started it
            if self.started_server:
                self.stop_server()

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Enhanced MCP Server Verification Tool")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to connect to (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to connect to (default: {DEFAULT_PORT})")
    parser.add_argument("--timeout", type=int, default=5, help="Timeout for requests in seconds (default: 5)")
    parser.add_argument("--output", default="verification_results.json", help="Output file for verification results (JSON)")
    parser.add_argument("--autostart", action="store_true", help="Automatically start the server if not running")
    parser.add_argument("--autofix", action="store_true", help="Automatically apply fixes if issues are found")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("="*60)
    print("Enhanced MCP Server Verification Tool")
    print("="*60)
    
    # Run verification
    verifier = EnhancedMCPVerifier(host=args.host, port=args.port, timeout=args.timeout)
    results = verifier.verify(autofix=args.autofix)
    
    # Print summary
    print("\nVerification Results:")
    print("="*40)
    
    # Server info
    server_status = "✅" if results["server_info"]["server_reachable"] else "❌"
    print(f"{server_status} Server at {args.host}:{args.port}")
    
    # API endpoints
    print("\nAPI Endpoints:")
    for endpoint, status in results["api_tests"].items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {endpoint}")
    
    # Tool tests
    print("\nTool Tests:")
    for tool, status in results["tool_tests"].items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {tool}")
    
    # Fixes applied
    if results.get("fixes_applied"):
        print("\nFixes Applied:")
        for fix in results["fixes_applied"]:
            print(f"✅ {fix}")
    
    # Overall status
    overall_status = "✅" if results.get("all_tests_passed", False) else "❌"
    print(f"\nOverall Status: {overall_status}")
    
    # Error message
    if "error" in results:
        print(f"\n❌ Error: {results['error']}")
    
    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {args.output}")
    
    # Exit with appropriate status
    return 0 if results.get("all_tests_passed", False) else 1

if __name__ == "__main__":
    sys.exit(main())
