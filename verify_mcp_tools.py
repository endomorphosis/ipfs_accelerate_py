#!/usr/bin/env python3
"""
Verify MCP Tool Registration and Functionality

This script verifies that all expected tools are properly registered with the MCP server
and tests their functionality.
"""

import os
import sys
import json
import time
import requests
import logging
import datetime
import tempfile
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify_mcp_tools")

# Expected tool categories and tools
EXPECTED_TOOLS = {
    "hardware": [
        "get_hardware_info"
    ],
    "ipfs": [
        "ipfs_add_file",
        "ipfs_node_info",
        "ipfs_cat",
        "ipfs_get",
        "ipfs_files_write",
        "ipfs_files_read",
        "ipfs_files_ls",
        "ipfs_pin_add",
        "ipfs_pin_rm",
        "ipfs_pin_ls"
    ],
    "model": [
        "model_inference",
        "list_models", 
        "init_endpoints"
    ],
    "vfs": [
        "vfs_list",
        "vfs_read",
        "vfs_write",
        "vfs_delete"
    ],
    "storage": [
        "create_storage",
        "list_storage",
        "get_storage_info",
        "delete_storage"
    ]
}

class MCPToolVerifier:
    """Class to verify MCP tool registration and functionality."""

    def __init__(self, host: str = "localhost", port: int = 8002):
        """Initialize the verifier."""
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.results = {
            "server_status": None,
            "manifest": None,
            "tools_found": [],
            "tools_missing": [],
            "tools_tested": {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    def log(self, message: str, level: str = "info"):
        """Log a message with the specified level."""
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
            
    def check_server_status(self) -> bool:
        """Check if the MCP server is running."""
        self.log(f"Checking MCP server status at {self.base_url}...")
        
        try:
            # Try the /status endpoint first
            try:
                response = requests.get(f"{self.base_url}/status", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    self.log(f"MCP server is running: {status}")
                    self.results["server_status"] = "running"
                    return True
            except requests.exceptions.RequestException:
                self.log("Status endpoint not available. Trying alternative endpoints.", "warning")
            
            # Try the /mcp/manifest endpoint as an alternative
            try:
                response = requests.get(f"{self.base_url}/mcp/manifest", timeout=5)
                if response.status_code == 200:
                    self.log("MCP server is running (verified through manifest)")
                    self.results["server_status"] = "running"
                    return True
            except requests.exceptions.RequestException:
                self.log("Manifest endpoint not available. Trying /tools endpoint.", "warning")
            
            # Try the /tools endpoint as a last resort
            response = requests.get(f"{self.base_url}/tools", timeout=5)
            if response.status_code == 200:
                self.log("MCP server is running (verified through tools endpoint)")
                self.results["server_status"] = "running"
                return True
            else:
                self.log(f"Failed to verify MCP server status: {response.status_code}", "warning")
                self.results["server_status"] = "error"
                return False
        except requests.exceptions.RequestException as e:
            self.log(f"Failed to connect to MCP server: {e}", "error")
            self.results["server_status"] = "not_running"
            return False
            
    def check_manifest(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check the MCP server manifest."""
        self.log("Checking MCP server manifest...")
        
        try:
            response = requests.get(f"{self.base_url}/mcp/manifest", timeout=5)
            if response.status_code == 200:
                manifest = response.json()
                self.log("Retrieved MCP server manifest")
                self.results["manifest"] = manifest
                return True, manifest
            else:
                self.log(f"Failed to retrieve MCP manifest: {response.status_code}", "warning")
                return False, None
        except requests.exceptions.RequestException as e:
            self.log(f"Error retrieving MCP manifest: {e}", "error")
            return False, None
            
    def get_available_tools(self) -> List[str]:
        """Get a list of available tools from the MCP server."""
        self.log("Getting list of available tools...")
        
        try:
            response = requests.get(f"{self.base_url}/tools")
            if response.status_code == 200:
                tools = response.json().get("tools", [])
                self.log(f"Found {len(tools)} available tools")
                self.results["tools_found"] = tools
                return tools
            else:
                self.log(f"Failed to get available tools: {response.status_code}", "warning")
                return []
        except requests.exceptions.RequestException as e:
            self.log(f"Error getting available tools: {e}", "error")
            return []
            
    def check_missing_tools(self, available_tools: List[str]) -> List[str]:
        """Check for missing tools."""
        self.log("Checking for missing tools...")
        
        # Flatten the expected tools list
        expected_tools = []
        for category, tools in EXPECTED_TOOLS.items():
            expected_tools.extend(tools)
            
        # Find missing tools
        missing_tools = [tool for tool in expected_tools if tool not in available_tools]
        self.log(f"Found {len(missing_tools)} missing tools")
        self.results["tools_missing"] = missing_tools
        return missing_tools
        
    def test_tool(self, tool_name: str) -> bool:
        """Test a specific tool."""
        self.log(f"Testing tool: {tool_name}...")
        
        # Prepare test arguments based on tool name
        args = self._get_test_args(tool_name)
        
        try:
            # Call the tool using the /call_tool endpoint
            response = requests.post(
                f"{self.base_url}/call_tool",
                json={"tool_name": tool_name, "arguments": args}
            )
            
            if response.status_code == 200:
                result = response.json().get("result")
                self.log(f"Tool {tool_name} test successful")
                self.results["tools_tested"][tool_name] = {
                    "status": "success",
                    "result": result
                }
                return True
            else:
                error = response.json().get("error", f"Status code: {response.status_code}")
                self.log(f"Tool {tool_name} test failed: {error}", "warning")
                self.results["tools_tested"][tool_name] = {
                    "status": "error",
                    "error": error
                }
                return False
        except requests.exceptions.RequestException as e:
            self.log(f"Error testing tool {tool_name}: {e}", "error")
            self.results["tools_tested"][tool_name] = {
                "status": "error",
                "error": str(e)
            }
            return False
            
    def _get_test_args(self, tool_name: str) -> Dict[str, Any]:
        """Get test arguments for a specific tool."""
        # Create test file if needed
        if tool_name in ["ipfs_add_file", "ipfs_get"]:
            fd, path = tempfile.mkstemp()
            with os.fdopen(fd, 'w') as f:
                f.write("This is a test file for MCP tool verification")
                
        # Return arguments based on tool name
        if tool_name == "get_hardware_info":
            return {}
        elif tool_name == "ipfs_add_file":
            return {"path": path}
        elif tool_name == "ipfs_cat":
            return {"cid": "QmTest123456789"}
        elif tool_name == "ipfs_get":
            return {"cid": "QmTest123456789", "output_path": path}
        elif tool_name == "ipfs_files_write":
            return {"path": "/test.txt", "content": "This is a test content"}
        elif tool_name == "ipfs_files_read":
            return {"path": "/test.txt"}
        elif tool_name == "ipfs_node_info":
            return {}
        elif tool_name == "model_inference":
            return {"model_name": "gpt2", "input_data": "Hello, world!"}
        elif tool_name == "list_models":
            return {}
        elif tool_name == "init_endpoints":
            return {"models": ["gpt2"]}
        elif tool_name == "vfs_list":
            return {"path": "/"}
        elif tool_name == "create_storage":
            return {"name": "test-storage", "size": 1}
        else:
            return {}
            
    def test_all_tools(self, available_tools: List[str]) -> Dict[str, bool]:
        """Test all available tools."""
        self.log("Testing all available tools...")
        
        results = {}
        for tool in available_tools:
            results[tool] = self.test_tool(tool)
            
        return results
        
    def generate_report(self) -> str:
        """Generate a report of the verification results."""
        self.log("Generating verification report...")
        
        # Calculate statistics
        total_expected = sum(len(tools) for tools in EXPECTED_TOOLS.values())
        total_found = len(self.results["tools_found"])
        total_missing = len(self.results["tools_missing"])
        total_tested = len(self.results["tools_tested"])
        total_success = sum(1 for tool in self.results["tools_tested"].values() if tool["status"] == "success")
        
        report = f"""# MCP Tool Verification Report

## Overview

- **Server**: {self.host}:{self.port}
- **Timestamp**: {self.results["timestamp"]}
- **Server Status**: {self.results["server_status"]}
- **Tools Expected**: {total_expected}
- **Tools Found**: {total_found}
- **Tools Missing**: {total_missing}
- **Tools Tested**: {total_tested}
- **Tests Passed**: {total_success}

## Available Tools

"""
        
        # Group tools by category
        categorized_tools = {category: [] for category in EXPECTED_TOOLS.keys()}
        uncategorized = []
        
        for tool in self.results["tools_found"]:
            found = False
            for category, tools in EXPECTED_TOOLS.items():
                if tool in tools:
                    categorized_tools[category].append(tool)
                    found = True
                    break
            if not found:
                uncategorized.append(tool)
                
        for category, tools in categorized_tools.items():
            report += f"### {category.capitalize()} Tools\n\n"
            if tools:
                for tool in tools:
                    status = self.results["tools_tested"].get(tool, {}).get("status", "not_tested")
                    status_icon = "✅" if status == "success" else "❌"
                    report += f"- {status_icon} `{tool}`\n"
            else:
                report += "No tools found in this category.\n"
            report += "\n"
            
        if uncategorized:
            report += "### Uncategorized Tools\n\n"
            for tool in uncategorized:
                status = self.results["tools_tested"].get(tool, {}).get("status", "not_tested")
                status_icon = "✅" if status == "success" else "❌"
                report += f"- {status_icon} `{tool}`\n"
            report += "\n"
            
        if self.results["tools_missing"]:
            report += "## Missing Tools\n\n"
            for tool in self.results["tools_missing"]:
                for category, tools in EXPECTED_TOOLS.items():
                    if tool in tools:
                        report += f"- `{tool}` (Category: {category})\n"
                        break
            report += "\n"
            
        report += "## Test Results\n\n"
        report += "| Tool | Status | Result |\n"
        report += "|------|--------|--------|\n"
        
        for tool, result in self.results["tools_tested"].items():
            status = result.get("status", "not_tested")
            status_icon = "✅" if status == "success" else "❌"
            result_text = str(result.get("result", result.get("error", "N/A")))
            if len(result_text) > 50:
                result_text = result_text[:47] + "..."
            report += f"| `{tool}` | {status_icon} | {result_text} |\n"
            
        report += "\n## Recommendations\n\n"
        
        if total_missing > 0:
            report += "1. Implement the missing tools\n"
        if total_found < total_expected:
            report += "2. Fix the tool registration mechanism\n"
        if total_success < total_tested:
            report += "3. Fix the failing tool implementations\n"
            
        return report
        
    def run_verification(self) -> Dict[str, Any]:
        """Run the verification process."""
        self.log("Starting MCP tool verification...")
        
        # Check server status
        server_running = self.check_server_status()
        if not server_running:
            self.log("MCP server is not running or not responding correctly.", "warning")
            # We'll continue with verification anyway and report the issues
        
        # Check manifest (only if server is running)
        manifest = None
        if server_running:
            success, manifest = self.check_manifest()
            if not success:
                self.log("Failed to retrieve MCP manifest. Continuing with limited verification.", "warning")
        
        # Get available tools (only if server is running)
        available_tools = []
        if server_running:
            available_tools = self.get_available_tools()
            if not available_tools:
                self.log("No tools available.", "warning")
        
        # Check missing tools
        missing_tools = self.check_missing_tools(available_tools)
        
        # Test all available tools (only if server is running and tools are available)
        if server_running and available_tools:
            self.test_all_tools(available_tools)
        
        # Generate report
        report = self.generate_report()
        self.results["report"] = report
        
        return self.results

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Verify MCP tool registration and functionality")
    parser.add_argument("--host", type=str, default="localhost", help="MCP server host")
    parser.add_argument("--port", type=int, default=8002, help="MCP server port")
    parser.add_argument("--output", type=str, default="mcp_tools_verification_report.md", help="Output report file")
    parser.add_argument("--force", action="store_true", help="Force verification even if server seems down")
    args = parser.parse_args()
    
    # Run verification
    verifier = MCPToolVerifier(host=args.host, port=args.port)
    results = verifier.run_verification()
    
    # Check if verification was successful
    if "report" not in results and not args.force:
        logger.error("Verification failed. No report generated.")
        
        # Generate a minimal error report
        error_report = f"""# MCP Tool Verification Error Report

## Error

The MCP server at {args.host}:{args.port} could not be verified.

Server status: {results.get("server_status", "unknown")}

## Recommendation

1. Check if the MCP server is running
2. Verify that the server is accessible at http://{args.host}:{args.port}
3. Run the verification script with the --force flag to attempt verification anyway
"""
        
        # Save error report
        with open(args.output, "w") as f:
            f.write(error_report)
        logger.info(f"Error report saved to {args.output}")
        
        return 1
    
    # Save report to specified output file
    if "report" in results:
        with open(args.output, "w") as f:
            f.write(results["report"])
        logger.info(f"Verification report saved to {args.output}")
    
    # Return success if all expected tools are available and working
    missing_tools = results.get("tools_missing", [])
    failed_tests = len([t for t, r in results.get("tools_tested", {}).items() if r.get("status") != "success"])
    
    if not missing_tools and not failed_tests:
        logger.info("All expected tools are available and working")
        return 0
    else:
        logger.warning(f"Found {len(missing_tools)} missing tools and {failed_tests} failing tests")
        return 1

if __name__ == "__main__":
    sys.exit(main())
