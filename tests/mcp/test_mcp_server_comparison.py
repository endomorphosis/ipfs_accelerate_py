#!/usr/bin/env python3
"""
MCP Server Comparison Tool (Enhanced)

This script tests both simple_mcp_server.py and unified_mcp_server.py
focusing on the tools that are failing in the MCP tests:
1. ipfs_get_hardware_info
2. ipfs_gateway_url with ipfs_hash parameter
3. Virtual filesystem operations
"""

import os
import sys
import json
import time
import logging
import requests
import argparse
import subprocess
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mcp_tools_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_tools_test")

class MCPToolTest:
    """Test MCP server tool registration."""
    
    def __init__(self, server_script, host='localhost', port=8001):
        """Initialize the test."""
        self.server_script = server_script
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.tools_endpoint = f"{self.base_url}/tools"
        self.server_proc = None
        self.server_log = f"{server_script.split('.')[0]}_output.log"
        
    def start_server(self):
        """Start the MCP server."""
        logger.info(f"Starting {self.server_script} on port {self.port}...")
        
        try:
            log_file = open(self.server_log, "w")
            cmd = [sys.executable, self.server_script, "--port", str(self.port), "--verbose"]
            
            self.server_proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server failed to start
            if self.server_proc.poll() is not None:
                logger.error(f"{self.server_script} failed to start")
                return False
                
            # Try to connect to confirm it's running
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=2)
                if resp.status_code == 200:
                    logger.info(f"{self.server_script} started successfully")
                    return True
            except requests.RequestException:
                pass
                
            # Try again with root path
            try:
                resp = requests.get(f"{self.base_url}/", timeout=2)
                if resp.status_code == 200:
                    logger.info(f"{self.server_script} started but health endpoint not available")
                    return True
            except requests.RequestException:
                pass
                
            logger.error(f"{self.server_script} started but not responding")
            return False
            
        except Exception as e:
            logger.error(f"Error starting {self.server_script}: {str(e)}")
            return False
    
    def stop_server(self):
        """Stop the server."""
        if self.server_proc:
            logger.info(f"Stopping {self.server_script}...")
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
            time.sleep(2)
    
    def get_tools(self):
        """Get registered tools from server."""
        try:
            logger.info(f"Getting tools from {self.tools_endpoint}")
            response = requests.get(self.tools_endpoint, timeout=5)
            
            if response.status_code == 200:
                tools = response.json()
                logger.info(f"Found {len(tools)} registered tools")
                return tools
            else:
                logger.error(f"Error getting tools: {response.status_code}")
                logger.error(response.text)
                return {}
                
        except Exception as e:
            logger.error(f"Error getting tools: {str(e)}")
            return {}
    
    def test_tool_registration(self):
        """Test if tools are properly registered."""
        tools = self.get_tools()
        
        if not tools:
            print(f"\n❌ No tools found for {self.server_script}")
            return False
        
        # Print found tools
        print(f"\nRegistered Tools in {self.server_script}:")
        for name, info in tools.items():
            if isinstance(info, dict) and 'description' in info:
                print(f"  - {name}: {info['description']}")
            else:
                print(f"  - {name}")
        
        # Check for expected tools
        expected_tools = [
            "get_hardware_info",
            "health_check",
            "ipfs_files_ls",
            "ipfs_files_read",
            "ipfs_files_write"
        ]
        
        missing_tools = [tool for tool in expected_tools if tool not in tools]
        
        if missing_tools:
            print(f"\n❌ Missing expected tools: {', '.join(missing_tools)}")
            return False
        else:
            print(f"\n✅ All expected tools registered in {self.server_script}")
            return True
    
    def extract_server_logs(self):
        """Extract useful information from server logs."""
        if os.path.exists(self.server_log):
            try:
                with open(self.server_log, 'r') as f:
                    logs = f.readlines()
                
                # Extract tool registration entries
                registration_logs = [line for line in logs if "Registered tool" in line or "register" in line.lower()]
                
                if registration_logs:
                    print("\nTool Registration Logs:")
                    for log in registration_logs:
                        print(f"  {log.strip()}")
                        
                # Extract errors
                error_logs = [line for line in logs if "error" in line.lower() or "exception" in line.lower()]
                
                if error_logs:
                    print("\nError Logs:")
                    for log in error_logs:
                        print(f"  {log.strip()}")
                
            except Exception as e:
                logger.error(f"Error extracting logs: {str(e)}")
    
    def run_test(self):
        """Run the complete test."""
        success = False
        
        if self.start_server():
            try:
                success = self.test_tool_registration()
            finally:
                self.stop_server()
                self.extract_server_logs()
        
        return success

def kill_existing_servers():
    """Kill any running MCP servers."""
    subprocess.run("pkill -f 'python.*mcp_server.py'", shell=True)
    time.sleep(2)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test MCP Server Tool Registration')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port-simple', type=int, default=8000, help='Port for simple MCP server')
    parser.add_argument('--port-unified', type=int, default=8001, help='Port for unified MCP server')
    args = parser.parse_args()
    
    # Kill any existing server
    kill_existing_servers()
    
    # Test server scripts
    servers = [
        {"script": "simple_mcp_server.py", "port": args.port_simple},
        {"script": "unified_mcp_server.py", "port": args.port_unified}
    ]
    
    results = {}
    
    print("==================================================")
    print("MCP Server Tool Registration Test")
    print("==================================================")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("==================================================")
    
    for server in servers:
        script = server["script"]
        port = server["port"]
        
        # Check if script exists
        if not os.path.exists(script):
            print(f"\n❌ {script} not found, skipping")
            continue
            
        print(f"\n--------------------------------------------------")
        print(f"Testing {script} on port {port}")
        print(f"--------------------------------------------------")
        
        tester = MCPToolTest(script, port=port)
        result = tester.run_test()
        results[script] = result
    
    # Print summary
    print("\n==================================================")
    print("Test Results Summary")
    print("==================================================")
    
    all_passed = True
    for script, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{script}: {status}")
        all_passed = all_passed and passed
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
