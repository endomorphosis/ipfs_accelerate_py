#!/usr/bin/env python3
"""
Test MCP Server Tool Registration

This script tests if tools are properly registered with the MCP server
by starting the server and querying the tools endpoint.
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import subprocess
from typing import Dict, Any, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ToolRegistrationTest:
    """Test tool registration in the MCP server."""
    
    def __init__(self, host='localhost', port=8001, protocol='http'):
        """Initialize the tool registration tester."""
        self.host = host
        self.port = port
        self.protocol = protocol
        self.base_url = f"{self.protocol}://{self.host}:{self.port}"
        self.tools_endpoint = f"{self.base_url}/tools"
        self.health_endpoint = f"{self.base_url}/health"
        
        # Define expected tools
        self.expected_tools = {
            "get_hardware_info",
            "health_check",
            "ipfs_cluster_status",
            "ipfs_dag_get",
            "ipfs_dag_put",
            "ipfs_files_ls",
            "ipfs_files_read",
            "ipfs_files_write"
        }
    
    def start_server(self):
        """Start the MCP server for testing."""
        try:
            # Kill any existing server
            logger.info("Stopping any existing MCP servers...")
            subprocess.run("pkill -f 'python.*unified_mcp_server.py'", shell=True)
            time.sleep(2)
            
            # Start server
            logger.info(f"Starting MCP server on port {self.port}...")
            cmd = [sys.executable, "unified_mcp_server.py", "--port", str(self.port), "--verbose"]
            
            self.server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            start_time = time.time()
            max_wait = 10
            server_started = False
            
            while time.time() - start_time < max_wait:
                # Check if server failed to start
                if self.server_proc.poll() is not None:
                    stdout, stderr = self.server_proc.communicate()
                    logger.error(f"Server failed to start: {stderr}")
                    return False
                
                # Check if server is responding
                try:
                    response = requests.get(self.health_endpoint, timeout=0.5)
                    if response.status_code == 200:
                        logger.info("Server started successfully")
                        server_started = True
                        break
                except requests.RequestException:
                    pass
                    
                logger.debug("Waiting for server to start...")
                time.sleep(1)
                
            if not server_started:
                logger.error("Server did not start within timeout period")
                self.stop_server()
                return False
                
            return True
                
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}")
            return False
    
    def stop_server(self):
        """Stop the MCP server."""
        if hasattr(self, 'server_proc'):
            logger.info("Stopping server...")
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
            time.sleep(1)
    
    def get_registered_tools(self) -> Dict[str, Any]:
        """Get the list of registered tools from the server."""
        try:
            logger.info(f"Requesting tools from {self.tools_endpoint}")
            response = requests.get(self.tools_endpoint, timeout=5)
            response.raise_for_status()
            tools = response.json()
            logger.info(f"Received {len(tools)} tools")
            return tools
        except Exception as e:
            logger.error(f"Error getting tools: {str(e)}")
            return {}
    
    def check_missing_tools(self, registered_tools: Dict[str, Any]) -> Set[str]:
        """Check for missing expected tools."""
        if not registered_tools:
            return self.expected_tools
            
        registered_tool_names = set(registered_tools.keys())
        missing_tools = self.expected_tools - registered_tool_names
        return missing_tools
    
    def run_test(self) -> bool:
        """Run the tool registration test."""
        success = True
        
        # Start server
        if not self.start_server():
            logger.error("Failed to start server. Test aborted.")
            return False
        
        try:
            # Get registered tools
            registered_tools = self.get_registered_tools()
            
            if not registered_tools:
                logger.error("No tools returned from server")
                return False
                
            # Check for missing tools
            missing_tools = self.check_missing_tools(registered_tools)
            if missing_tools:
                logger.warning(f"Missing expected tools: {', '.join(missing_tools)}")
                success = False
            else:
                logger.info("All expected tools are registered")
                
            # Print tool details
            print("\nRegistered Tools:")
            for tool_name, tool_info in registered_tools.items():
                if isinstance(tool_info, dict) and 'description' in tool_info:
                    print(f"  - {tool_name}: {tool_info.get('description', 'No description')}")
                else:
                    print(f"  - {tool_name}")
        
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            logger.error(traceback.format_exc())
            success = False
        finally:
            # Stop server
            self.stop_server()
        
        return success

def main():
    """Run the tool registration test."""
    parser = argparse.ArgumentParser(description='Test MCP Tool Registration')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8001, help='Server port')
    parser.add_argument('--protocol', default='http', help='Protocol (http/https)')
    args = parser.parse_args()
    
    tester = ToolRegistrationTest(
        host=args.host,
        port=args.port,
        protocol=args.protocol
    )
    
    print("=======================================")
    print("MCP Server Tool Registration Test")
    print("=======================================")
    
    if tester.run_test():
        print("\n✅ Tool registration test passed")
        sys.exit(0)
    else:
        print("\n❌ Tool registration test failed")
        sys.exit(1)

if __name__ == "__main__":
    import traceback
    main()
