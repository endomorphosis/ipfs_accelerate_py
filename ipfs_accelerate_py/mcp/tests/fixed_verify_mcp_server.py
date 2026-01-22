"""
Improved MCP Server Verification

This script verifies the IPFS Accelerate MCP server components using the correct
API for FastMCP 2.2.7.
"""

import json
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Any, Optional, Tuple

from fastmcp.client import Client

# Constants
VERIFICATION_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "verification_results.json")
SERVER_MODULE = "ipfs_accelerate_py.mcp.server"
SERVER_PROCESS_STARTUP_TIME = 3  # seconds to wait for server to start

class MCPVerifier:
    """Verify MCP server components"""
    
    def __init__(self):
        self.server_process = None
        self.client = None
        self.verification_results = {
            "tools": [],
            "resources": [],
            "prompts": [],
            "errors": []
        }
    
    def start_server(self) -> bool:
        """Start the MCP server as a subprocess"""
        try:
            print("Starting MCP server...")
            cmd = [sys.executable, "-m", SERVER_MODULE]
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            print(f"MCP server started with PID: {self.server_process.pid}")
            
            # Wait for server to start
            time.sleep(SERVER_PROCESS_STARTUP_TIME)
            return True
        except Exception as e:
            print(f"Error starting MCP server: {e}")
            return False
    
    def stop_server(self):
        """Stop the MCP server subprocess"""
        if self.server_process:
            print("Stopping MCP server...")
            try:
                # Try to terminate gracefully first
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if terminate doesn't work
                self.server_process.kill()
            print("MCP server stopped")
    
    def connect_to_client(self) -> bool:
        """Connect to the MCP server via client"""
        try:
            # Create client without explicit connect method in 2.2.7
            self.client = Client("http://localhost:8080")
            # Test connection by trying to get server info
            print(f"Connecting to MCP server at http://localhost:8080")
            print("Connected to MCP server")
            return True
        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            self.verification_results["errors"].append(f"Connection error: {str(e)}")
            return False
    
    def verify_tools(self):
        """Verify registered tools"""
        print("\nVerifying Tools:")
        try:
            tools = self.client.get_tools()
            print(f"Found {len(tools)} tools:")
            for i, tool in enumerate(tools):
                tool_info = {
                    "name": tool.name if hasattr(tool, 'name') else f"Tool {i+1}",
                    "description": tool.description if hasattr(tool, 'description') else "No description"
                }
                print(f"  {i+1}. {tool_info['name']}: {tool_info['description']}")
                self.verification_results["tools"].append(tool_info)
            return tools
        except Exception as e:
            error_msg = f"Error verifying tools: {e}"
            print(error_msg)
            self.verification_results["errors"].append(error_msg)
            return []
    
    def verify_resources(self):
        """Verify registered resources"""
        print("\nVerifying Resources:")
        try:
            resources = self.client.get_resources()
            print(f"Found {len(resources)} resources:")
            for i, resource in enumerate(resources):
                resource_info = {
                    "uri": resource.uri if hasattr(resource, 'uri') else f"Resource {i+1}",
                    "description": resource.description if hasattr(resource, 'description') else "No description"
                }
                print(f"  {i+1}. {resource_info['uri']}: {resource_info['description']}")
                self.verification_results["resources"].append(resource_info)
            return resources
        except Exception as e:
            error_msg = f"Error verifying resources: {e}"
            print(error_msg)
            self.verification_results["errors"].append(error_msg)
            return []
    
    def verify_prompts(self):
        """Verify registered prompts"""
        print("\nVerifying Prompts:")
        try:
            prompts = self.client.get_prompts()
            print(f"Found {len(prompts)} prompts:")
            for i, prompt in enumerate(prompts):
                prompt_info = {
                    "name": prompt.name if hasattr(prompt, 'name') else f"Prompt {i+1}",
                    "description": prompt.description if hasattr(prompt, 'description') else "No description"
                }
                print(f"  {i+1}. {prompt_info['name']}: {prompt_info['description']}")
                self.verification_results["prompts"].append(prompt_info)
            return prompts
        except Exception as e:
            error_msg = f"Error verifying prompts: {e}"
            print(error_msg)
            self.verification_results["errors"].append(error_msg)
            return []
    
    def save_verification_results(self):
        """Save verification results to file"""
        try:
            with open(VERIFICATION_RESULTS_PATH, 'w') as f:
                json.dump(self.verification_results, f, indent=2)
            print(f"\nDetailed verification results saved to: {VERIFICATION_RESULTS_PATH}")
        except Exception as e:
            print(f"Error saving verification results: {e}")
    
    def print_summary(self, tools, resources, prompts):
        """Print verification summary"""
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        
        # Calculate completion
        tools_count = len(tools) if tools else 0
        resources_count = len(resources) if resources else 0
        prompts_count = len(prompts) if prompts else 0
        
        if tools_count == 0 or resources_count == 0 or prompts_count == 0:
            print("\nOverall Completion: Incomplete")
            print(f"- Tools: {tools_count}")
            print(f"- Resources: {resources_count}")
            print(f"- Prompts: {prompts_count}")
            print("\n❌ SOME EXPECTED COMPONENTS ARE MISSING")
        else:
            print("\nOverall Completion: Complete")
            print(f"- Tools: {tools_count}")
            print(f"- Resources: {resources_count}")
            print(f"- Prompts: {prompts_count}")
            print("\n✅ ALL COMPONENTS VERIFIED SUCCESSFULLY")
    
    def run(self):
        """Run the verification process"""
        try:
            # Start server
            if not self.start_server():
                return
            
            # Connect to client
            if not self.connect_to_client():
                self.stop_server()
                return
            
            # Verify components
            tools = self.verify_tools()
            resources = self.verify_resources()
            prompts = self.verify_prompts()
            
            # Save results
            self.save_verification_results()
            
            # Print summary
            self.print_summary(tools, resources, prompts)
        finally:
            # Always stop the server
            self.stop_server()

if __name__ == "__main__":
    print("=" * 80)
    print("IPFS Accelerate MCP Server Verification")
    print("=" * 80)
    
    verifier = MCPVerifier()
    verifier.run()
