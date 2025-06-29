#!/usr/bin/env python3
"""
MCP Server Integration Checker

This script checks the current MCP server integration status
and provides recommendations for fixing any issues found.
"""

import os
import sys
import json
import time
import logging
import requests
import datetime
import importlib
import platform
import subprocess
import socket
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_integration_checker")

class MCPIntegrationChecker:
    """Check the MCP server integration status."""
    
    def __init__(self, host="localhost", port=8002):
        """Initialize the checker."""
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "system": self._get_system_info(),
            "mcp_server": {
                "status": "unknown",
                "available_endpoints": [],
                "tools": [],
                "manifest": None
            },
            "ipfs_accelerate": {
                "installed": False,
                "version": None,
                "functions": []
            },
            "issues": [],
            "recommendations": []
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "distribution": platform.platform(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "processor": platform.processor()
        }
    
    def check_mcp_server(self) -> bool:
        """Check if the MCP server is running."""
        logger.info(f"Checking MCP server at {self.base_url}...")
        
        # Try various endpoints
        endpoints_to_check = [
            "/status",
            "/mcp/manifest",
            "/tools",
            "/"
        ]
        
        self.results["mcp_server"]["available_endpoints"] = []
        
        for endpoint in endpoints_to_check:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=2)
                if response.status_code == 200:
                    logger.info(f"Endpoint {endpoint} is available")
                    self.results["mcp_server"]["available_endpoints"].append(endpoint)
                    
                    # Save additional information
                    if endpoint == "/mcp/manifest":
                        self.results["mcp_server"]["manifest"] = response.json()
                    elif endpoint == "/tools":
                        self.results["mcp_server"]["tools"] = response.json().get("tools", [])
            except:
                logger.info(f"Endpoint {endpoint} is not available")
        
        if self.results["mcp_server"]["available_endpoints"]:
            self.results["mcp_server"]["status"] = "running"
            return True
        else:
            self.results["mcp_server"]["status"] = "not_running"
            self.results["issues"].append("MCP server is not running or not responding")
            self.results["recommendations"].append("Start the MCP server with './restart_mcp_server.sh'")
            return False
    
    def check_mcp_server_port(self) -> bool:
        """Check if the MCP server port is open."""
        logger.info(f"Checking if port {self.port} is open...")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            
            if result == 0:
                logger.info(f"Port {self.port} is open")
                return True
            else:
                logger.info(f"Port {self.port} is closed")
                self.results["issues"].append(f"Port {self.port} is not open")
                self.results["recommendations"].append(f"Check if any process is using port {self.port}")
                return False
        except Exception as e:
            logger.error(f"Error checking port: {e}")
            return False
    
    def check_ipfs_accelerate(self) -> bool:
        """Check if the IPFS Accelerate package is installed."""
        logger.info("Checking IPFS Accelerate package...")
        
        try:
            # First try direct import
            try:
                import ipfs_accelerate_py
                self.results["ipfs_accelerate"]["installed"] = True
                if hasattr(ipfs_accelerate_py, "__version__"):
                    self.results["ipfs_accelerate"]["version"] = ipfs_accelerate_py.__version__
                logger.info("IPFS Accelerate package is installed")
            except ImportError:
                # Try to find the module in the current directory
                module_path = os.path.join(os.getcwd(), "ipfs_accelerate_py.py")
                if os.path.exists(module_path):
                    spec = importlib.util.spec_from_file_location("ipfs_accelerate_py", module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.results["ipfs_accelerate"]["installed"] = True
                    logger.info("IPFS Accelerate module found in current directory")
                else:
                    logger.warning("IPFS Accelerate package is not installed")
                    self.results["issues"].append("IPFS Accelerate package is not installed")
                    self.results["recommendations"].append("Install IPFS Accelerate package or check the module path")
                    return False
            
            # Get available functions
            if self.results["ipfs_accelerate"]["installed"]:
                # Create a list of methods that should be available
                expected_methods = [
                    "add_file",
                    "cat",
                    "get",
                    "node_info",
                    "process"
                ]
                
                # Check which methods are available
                available_methods = []
                for method in expected_methods:
                    if hasattr(module, method) or hasattr(module.ipfs_accelerate_py, method):
                        available_methods.append(method)
                
                self.results["ipfs_accelerate"]["functions"] = available_methods
                
                if len(available_methods) < len(expected_methods):
                    missing = [m for m in expected_methods if m not in available_methods]
                    self.results["issues"].append(f"IPFS Accelerate is missing methods: {', '.join(missing)}")
                    self.results["recommendations"].append("Update IPFS Accelerate package or implement missing methods")
            
            return True
        except Exception as e:
            logger.error(f"Error checking IPFS Accelerate: {e}")
            self.results["issues"].append(f"Error checking IPFS Accelerate: {str(e)}")
            return False
    
    def check_mcp_tools(self) -> bool:
        """Check which MCP tools are available."""
        logger.info("Checking available MCP tools...")
        
        if self.results["mcp_server"]["status"] != "running":
            return False
        
        # Check tools endpoint
        try:
            response = requests.get(f"{self.base_url}/tools", timeout=2)
            if response.status_code == 200:
                tools = response.json().get("tools", [])
                self.results["mcp_server"]["tools"] = tools
                logger.info(f"Found {len(tools)} tools")
                
                # Expected categories and tools
                expected_tools = {
                    "hardware": ["get_hardware_info"],
                    "ipfs": ["ipfs_add_file", "ipfs_node_info", "ipfs_cat"],
                    "model": ["model_inference", "list_models"]
                }
                
                # Flatten the expected tools list
                all_expected = []
                for category, tools_list in expected_tools.items():
                    all_expected.extend(tools_list)
                
                # Find missing tools
                missing_tools = [tool for tool in all_expected if tool not in tools]
                if missing_tools:
                    self.results["issues"].append(f"Missing tools: {', '.join(missing_tools)}")
                    self.results["recommendations"].append("Fix tool registration in the MCP server")
                
                return True
            else:
                logger.warning(f"Failed to get tools: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking tools: {e}")
            return False
    
    def test_tool(self, tool_name: str) -> Dict[str, Any]:
        """Test a specific tool."""
        logger.info(f"Testing tool: {tool_name}...")
        
        if self.results["mcp_server"]["status"] != "running":
            return {"status": "error", "error": "MCP server is not running"}
        
        # Prepare test arguments based on tool name
        args = {}
        if tool_name == "ipfs_add_file":
            # Create a temporary file
            temp_file = "test_file.txt"
            with open(temp_file, "w") as f:
                f.write("This is a test file")
            args = {"path": temp_file}
        elif tool_name == "ipfs_cat":
            args = {"cid": "QmTest123456789"}
        elif tool_name == "model_inference":
            args = {"model_name": "gpt2", "input_data": "Hello, world!"}
        
        # Call the tool
        try:
            response = requests.post(
                f"{self.base_url}/call_tool",
                json={"tool_name": tool_name, "arguments": args},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json().get("result")
                logger.info(f"Tool {tool_name} test successful")
                return {"status": "success", "result": result}
            else:
                error = response.json().get("error", f"Status code: {response.status_code}")
                logger.warning(f"Tool {tool_name} test failed: {error}")
                return {"status": "error", "error": error}
        except Exception as e:
            logger.error(f"Error testing tool {tool_name}: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_tools_functionality(self) -> bool:
        """Check the functionality of important tools."""
        logger.info("Checking tools functionality...")
        
        if self.results["mcp_server"]["status"] != "running":
            return False
        
        if not self.results["mcp_server"]["tools"]:
            self.results["issues"].append("No tools available to test")
            return False
        
        # Tools to test
        tools_to_test = [
            "get_hardware_info",
            "ipfs_node_info"
        ]
        
        # Add other tools if they are in the available tools list
        if "ipfs_add_file" in self.results["mcp_server"]["tools"]:
            tools_to_test.append("ipfs_add_file")
        
        if "model_inference" in self.results["mcp_server"]["tools"]:
            tools_to_test.append("model_inference")
        
        # Test tools
        self.results["tool_tests"] = {}
        all_success = True
        
        for tool in tools_to_test:
            if tool in self.results["mcp_server"]["tools"]:
                test_result = self.test_tool(tool)
                self.results["tool_tests"][tool] = test_result
                
                if test_result["status"] == "error":
                    all_success = False
                    self.results["issues"].append(f"Tool {tool} is not working: {test_result.get('error')}")
                    self.results["recommendations"].append(f"Fix the implementation of the {tool} tool")
        
        return all_success
    
    def fix_common_issues(self) -> List[str]:
        """Try to fix common issues and return a list of actions taken."""
        logger.info("Attempting to fix common issues...")
        
        actions_taken = []
        
        # If MCP server is not running, try to start it
        if self.results["mcp_server"]["status"] == "not_running":
            logger.info("Trying to start MCP server...")
            
            try:
                # Check if the restart script exists
                restart_script = os.path.join(os.getcwd(), "restart_mcp_server.sh")
                if os.path.exists(restart_script):
                    # Run the script
                    subprocess.run(["bash", restart_script], check=True)
                    actions_taken.append("Started MCP server with restart_mcp_server.sh")
                    
                    # Wait for the server to start
                    time.sleep(3)
                    
                    # Check if it's running now
                    if self.check_mcp_server():
                        actions_taken.append("MCP server started successfully")
                    else:
                        actions_taken.append("MCP server still not running after restart attempt")
                else:
                    actions_taken.append("Could not find restart_mcp_server.sh script")
            except Exception as e:
                logger.error(f"Error starting MCP server: {e}")
                actions_taken.append(f"Error starting MCP server: {str(e)}")
        
        # If tools are missing, try to register them
        if self.results["mcp_server"]["status"] == "running" and "Missing tools" in "\n".join(self.results["issues"]):
            logger.info("Trying to register missing tools...")
            
            try:
                # Check if the fix script exists
                fix_script = os.path.join(os.getcwd(), "fix_mcp_tool_registration.py")
                if os.path.exists(fix_script):
                    # Run the script
                    subprocess.run(["python", fix_script], check=True)
                    actions_taken.append("Ran fix_mcp_tool_registration.py to register missing tools")
                    
                    # Check if tools are registered now
                    if self.check_mcp_tools():
                        actions_taken.append("Tools registered successfully")
                    else:
                        actions_taken.append("Some tools still missing after registration attempt")
                else:
                    actions_taken.append("Could not find fix_mcp_tool_registration.py script")
            except Exception as e:
                logger.error(f"Error registering tools: {e}")
                actions_taken.append(f"Error registering tools: {str(e)}")
        
        return actions_taken
    
    def generate_report(self) -> str:
        """Generate a report of the integration status."""
        logger.info("Generating integration report...")
        
        report = f"""# MCP Integration Checker Report

## Overview

- **Timestamp**: {self.results["timestamp"]}
- **MCP Server Status**: {self.results["mcp_server"]["status"]}
- **IPFS Accelerate Installed**: {self.results["ipfs_accelerate"]["installed"]}
- **IPFS Accelerate Version**: {self.results["ipfs_accelerate"]["version"] or "unknown"}

## System Information

- **OS**: {self.results["system"]["os"]} {self.results["system"]["os_version"]}
- **Distribution**: {self.results["system"]["distribution"]}
- **Architecture**: {self.results["system"]["architecture"]}
- **Python Version**: {self.results["system"]["python_version"]}

## MCP Server

**Status**: {self.results["mcp_server"]["status"]}

**Available Endpoints**:
"""
        
        for endpoint in self.results["mcp_server"]["available_endpoints"]:
            report += f"- `{endpoint}`\n"
        
        if not self.results["mcp_server"]["available_endpoints"]:
            report += "No endpoints available\n"
        
        report += "\n**Available Tools**:\n"
        
        for tool in self.results["mcp_server"]["tools"]:
            status = "✅" if self.results.get("tool_tests", {}).get(tool, {}).get("status") == "success" else "❓"
            report += f"- {status} `{tool}`\n"
        
        if not self.results["mcp_server"]["tools"]:
            report += "No tools available\n"
        
        report += "\n## IPFS Accelerate\n\n"
        report += f"**Installed**: {self.results['ipfs_accelerate']['installed']}\n"
        report += f"**Version**: {self.results['ipfs_accelerate']['version'] or 'unknown'}\n"
        report += "\n**Available Functions**:\n"
        
        for function in self.results["ipfs_accelerate"]["functions"]:
            report += f"- `{function}`\n"
        
        if not self.results["ipfs_accelerate"]["functions"]:
            report += "No functions available\n"
        
        if "tool_tests" in self.results:
            report += "\n## Tool Tests\n\n"
            report += "| Tool | Status | Result |\n"
            report += "|------|--------|--------|\n"
            
            for tool, test in self.results["tool_tests"].items():
                status = "✅" if test["status"] == "success" else "❌"
                result = str(test.get("result", test.get("error", "N/A")))
                if len(result) > 50:
                    result = result[:47] + "..."
                report += f"| `{tool}` | {status} | {result} |\n"
        
        report += "\n## Issues\n\n"
        
        if self.results["issues"]:
            for issue in self.results["issues"]:
                report += f"- {issue}\n"
        else:
            report += "No issues found\n"
        
        report += "\n## Recommendations\n\n"
        
        if self.results["recommendations"]:
            for recommendation in self.results["recommendations"]:
                report += f"- {recommendation}\n"
        else:
            report += "No recommendations\n"
        
        if "actions" in self.results:
            report += "\n## Actions Taken\n\n"
            
            for action in self.results["actions"]:
                report += f"- {action}\n"
        
        report += "\n## Next Steps\n\n"
        
        if self.results["issues"]:
            report += "1. Address the issues listed above\n"
            report += "2. Run `fix_mcp_tool_registration.py` to register all tools\n"
            report += "3. Run `verify_mcp_tools.py` to verify the tools are working\n"
        else:
            report += "1. Run `verify_mcp_tools.py` to get a detailed verification report\n"
            report += "2. Implement any missing tools\n"
            report += "3. Add comprehensive tests for all tools\n"
        
        return report
    
    def check_integration(self, fix_issues=False) -> Dict[str, Any]:
        """Check the MCP server integration status."""
        logger.info("Starting MCP integration check...")
        
        # Check if the MCP server is running
        self.check_mcp_server_port()
        self.check_mcp_server()
        
        # Check if the IPFS Accelerate package is installed
        self.check_ipfs_accelerate()
        
        # If the server is running, check which tools are available
        if self.results["mcp_server"]["status"] == "running":
            self.check_mcp_tools()
            self.check_tools_functionality()
        
        # Try to fix common issues if requested
        if fix_issues:
            actions = self.fix_common_issues()
            self.results["actions"] = actions
        
        # Generate report
        report = self.generate_report()
        self.results["report"] = report
        
        return self.results

def main():
    """Main entry point for the script."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check MCP server integration status")
    parser.add_argument("--host", type=str, default="localhost", help="MCP server host")
    parser.add_argument("--port", type=int, default=8002, help="MCP server port")
    parser.add_argument("--fix", action="store_true", help="Try to fix common issues")
    parser.add_argument("--output", type=str, default="mcp_integration_report.md", help="Output report file")
    args = parser.parse_args()
    
    # Run integration check
    checker = MCPIntegrationChecker(host=args.host, port=args.port)
    results = checker.check_integration(fix_issues=args.fix)
    
    # Save report to file
    with open(args.output, "w") as f:
        f.write(results["report"])
    logger.info(f"Integration report saved to {args.output}")
    
    # Return success if no issues were found
    if not results["issues"]:
        logger.info("No issues found")
        return 0
    else:
        logger.warning(f"Found {len(results['issues'])} issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
