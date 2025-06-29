#!/usr/bin/env python3
"""
Enhanced MCP Server Test

This script provides comprehensive testing for the MCP server with focus on
the get_hardware_info tool.
"""

import json
import requests
import sys
import logging
import time
import subprocess
import argparse
import os
from typing import Dict, Any, List, Optional, Tuple

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("enhanced_mcp_test.log")
    ]
)
logger = logging.getLogger(__name__)

class MCPTester:
    """MCP Server test class with enhanced diagnostics."""
    
    def __init__(self, host: str = "localhost", port: int = 8002):
        """Initialize with server details."""
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.server_process = None
        self.discovered_endpoints = {
            "manifest": None,
            "tools_list": None,
            "hardware_info": None
        }
    
    def start_server(self, server_script: str = "final_mcp_server.py") -> bool:
        """Start the MCP server in a subprocess."""
        try:
            logger.info(f"Starting MCP server from {server_script} on {self.host}:{self.port}")
            
            cmd = [
                sys.executable, 
                server_script, 
                "--host", self.host,
                "--port", str(self.port),
                "--debug"
            ]
            
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for server to start
            logger.info(f"Waiting for server to start (PID: {self.server_process.pid})...")
            time.sleep(3)
            
            return True
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def discover_endpoints(self) -> Dict[str, Optional[str]]:
        """Discover available MCP server endpoints by trying different patterns."""
        # Try common base endpoints
        base_endpoints = [
            "",
            "/",
            "/mcp",
            "/api"
        ]
        
        # Try manifest endpoints
        manifest_patterns = [
            "/mcp/manifest",
            "/manifest",
            "/api/manifest",
            "/server/manifest"
        ]
        
        # Try tool list endpoints
        tools_list_patterns = [
            "/tools",
            "/mcp/tools",
            "/api/tools"
        ]
        
        # Discover manifest endpoint
        for base in base_endpoints:
            for pattern in manifest_patterns:
                url = f"{self.base_url}{base}{pattern}"
                try:
                    logger.info(f"Trying manifest endpoint: {url}")
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Found manifest endpoint: {url}")
                        self.discovered_endpoints["manifest"] = url
                        break
                except Exception:
                    pass
            if self.discovered_endpoints["manifest"]:
                break
        
        # Discover tools list endpoint
        for base in base_endpoints:
            for pattern in tools_list_patterns:
                url = f"{self.base_url}{base}{pattern}"
                try:
                    logger.info(f"Trying tools list endpoint: {url}")
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Found tools list endpoint: {url}")
                        self.discovered_endpoints["tools_list"] = url
                        break
                except Exception:
                    pass
            if self.discovered_endpoints["tools_list"]:
                break
        
        return self.discovered_endpoints
    
    def check_server_availability(self) -> bool:
        """Check if the server is reachable."""
        try:
            response = requests.get(self.base_url, timeout=5)
            logger.info(f"Server responded with status code: {response.status_code}")
            return True
        except Exception as e:
            logger.error(f"Server is not reachable: {e}")
            return False
    
    def list_tools(self) -> Tuple[bool, List[str]]:
        """List available tools from the server."""
        # First try the discovered endpoint if available
        if self.discovered_endpoints["tools_list"]:
            try:
                response = requests.get(self.discovered_endpoints["tools_list"], timeout=5)
                if response.status_code == 200:
                    tools = response.json()
                    if isinstance(tools, list):
                        logger.info(f"Found {len(tools)} tools: {tools}")
                        return True, tools
            except Exception as e:
                logger.warning(f"Error with discovered tools endpoint: {e}")
        
        # Try various tools endpoints
        endpoints = [
            f"{self.base_url}/tools",
            f"{self.base_url}/mcp/tools",
            f"{self.base_url}/api/tools"
        ]
        
        for url in endpoints:
            try:
                logger.info(f"Trying tools endpoint: {url}")
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    tools = response.json()
                    if isinstance(tools, list):
                        logger.info(f"Found {len(tools)} tools: {tools}")
                        self.discovered_endpoints["tools_list"] = url
                        return True, tools
                    else:
                        logger.warning(f"Unexpected tools format from {url}: {tools}")
            except Exception as e:
                logger.warning(f"Error accessing {url}: {e}")
        
        logger.error("Failed to find tools endpoint")
        return False, []
    
    def discover_hardware_info_endpoint(self) -> Optional[str]:
        """Discover the correct endpoint for get_hardware_info tool."""
        # Try different patterns
        success, tools = self.list_tools()
        
        if success:
            # If get_hardware_info is in the tools list, try the standard format
            if "get_hardware_info" in tools:
                endpoints = [
                    f"{self.base_url}/tools/get_hardware_info/invoke",
                    f"{self.base_url}/mcp/tool/get_hardware_info",
                    f"{self.base_url}/tool/get_hardware_info"
                ]
                
                for url in endpoints:
                    try:
                        logger.info(f"Trying hardware info endpoint: {url}")
                        response = requests.post(url, json={}, timeout=5)
                        if response.status_code == 200:
                            logger.info(f"Found working hardware info endpoint: {url}")
                            self.discovered_endpoints["hardware_info"] = url
                            return url
                    except Exception as e:
                        logger.warning(f"Error with endpoint {url}: {e}")
        
        # If we couldn't find a working endpoint, try some alternative names
        alternative_names = ["hardware_info", "system_info", "get_system_info"]
        
        for name in alternative_names:
            endpoints = [
                f"{self.base_url}/tools/{name}/invoke",
                f"{self.base_url}/mcp/tool/{name}",
                f"{self.base_url}/tool/{name}"
            ]
            
            for url in endpoints:
                try:
                    logger.info(f"Trying alternative hardware info endpoint: {url}")
                    response = requests.post(url, json={}, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Found working alternative hardware info endpoint: {url}")
                        self.discovered_endpoints["hardware_info"] = url
                        return url
                except Exception:
                    pass
        
        logger.error("Failed to find hardware info endpoint")
        return None
    
    def test_get_hardware_info(self) -> Tuple[bool, Dict[str, Any]]:
        """Test the get_hardware_info tool."""
        # First try to discover the endpoint if not already known
        if not self.discovered_endpoints["hardware_info"]:
            self.discover_hardware_info_endpoint()
        
        if self.discovered_endpoints["hardware_info"]:
            try:
                url = self.discovered_endpoints["hardware_info"]
                logger.info(f"Testing hardware info endpoint: {url}")
                response = requests.post(url, json={}, timeout=10)
                if response.status_code == 200:
                    hardware_info = response.json()
                    logger.info("Hardware info successfully retrieved")
                    logger.info(f"Hardware Info: {json.dumps(hardware_info, indent=2)}")
                    
                    # Verify the structure of the result
                    if isinstance(hardware_info, dict):
                        if "system" in hardware_info:
                            logger.info("Hardware info has expected 'system' field")
                        if "accelerators" in hardware_info:
                            logger.info("Hardware info has expected 'accelerators' field")
                        
                        return True, hardware_info
                    else:
                        logger.warning("Hardware info is not a dictionary")
                else:
                    logger.error(f"Hardware info request failed with status {response.status_code}")
            except Exception as e:
                logger.error(f"Error testing hardware info endpoint: {e}")
        
        return False, {}
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics on the MCP server."""
        results = {
            "timestamp": time.time(),
            "server": {
                "host": self.host,
                "port": self.port,
                "base_url": self.base_url,
                "reachable": False
            },
            "endpoints": self.discovered_endpoints.copy(),
            "tools": {
                "available": False,
                "list": []
            },
            "hardware_info": {
                "success": False,
                "data": {}
            }
        }
        
        # Check server availability
        results["server"]["reachable"] = self.check_server_availability()
        if not results["server"]["reachable"]:
            logger.error("Server is not reachable, cannot proceed with tests")
            return results
        
        # Discover endpoints
        self.discover_endpoints()
        results["endpoints"] = self.discovered_endpoints.copy()
        
        # List tools
        success, tools = self.list_tools()
        results["tools"]["available"] = success
        results["tools"]["list"] = tools
        
        # Test hardware info
        if "get_hardware_info" in tools:
            success, hardware_info = self.test_get_hardware_info()
            results["hardware_info"]["success"] = success
            results["hardware_info"]["data"] = hardware_info
        else:
            logger.warning("get_hardware_info tool not found in tools list")
        
        return results
    
    def cleanup(self) -> None:
        """Clean up resources including stopping the server."""
        if self.server_process:
            try:
                logger.info(f"Stopping server process (PID: {self.server_process.pid})")
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, forcing kill")
                self.server_process.kill()
            except Exception as e:
                logger.error(f"Error stopping server: {e}")


def main():
    """Main function to run MCP server tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test MCP server for IPFS Accelerate")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    parser.add_argument("--server", type=str, default="final_mcp_server.py", help="Server script to run")
    parser.add_argument("--start-server", action="store_true", help="Start the server as part of the test")
    parser.add_argument("--output", type=str, default="mcp_test_results.json", help="Output file for results")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for the entire test in seconds")
    args = parser.parse_args()
    
    try:
        tester = MCPTester(host=args.host, port=args.port)
        
        # Start server if requested
        if args.start_server:
            if not tester.start_server(args.server):
                logger.error("Failed to start MCP server")
                return 1
        
        # Run diagnostics
        start_time = time.time()
        results = tester.run_diagnostics()
        elapsed_time = time.time() - start_time
        results["execution_time"] = elapsed_time
        
        # Save results to file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Print results summary
        print("\n" + "=" * 60)
        print("MCP SERVER TEST RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"Server: {args.host}:{args.port}")
        print(f"Reachable: {'Yes' if results['server']['reachable'] else 'No'}")
        
        if results['server']['reachable']:
            print("\nDiscovered Endpoints:")
            for name, url in results['endpoints'].items():
                print(f"  {name}: {url if url else 'Not found'}")
            
            print(f"\nTools Available: {'Yes' if results['tools']['available'] else 'No'}")
            if results['tools']['available']:
                print(f"Tools Count: {len(results['tools']['list'])}")
                print(f"Tools: {', '.join(results['tools']['list'])}")
            
            print(f"\nget_hardware_info Test: {'Success' if results['hardware_info']['success'] else 'Failed'}")
            if results['hardware_info']['success']:
                print("Hardware Info Summary:")
                if 'system' in results['hardware_info']['data']:
                    system = results['hardware_info']['data']['system']
                    print(f"  OS: {system.get('os', 'Unknown')}")
                    print(f"  Architecture: {system.get('architecture', 'Unknown')}")
                    print(f"  Python Version: {system.get('python_version', 'Unknown')}")
                if 'accelerators' in results['hardware_info']['data']:
                    accel = results['hardware_info']['data']['accelerators']
                    print("  Accelerators:")
                    for name, info in accel.items():
                        print(f"    {name}: {'Available' if info.get('available', False) else 'Not available'}")
        
        print("\n" + "=" * 60)
        
        # Return success if hardware_info test passed
        return 0 if results["hardware_info"]["success"] else 1
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'tester' in locals() and args.start_server:
            tester.cleanup()

if __name__ == "__main__":
    sys.exit(main())
