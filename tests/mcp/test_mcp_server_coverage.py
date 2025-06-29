#!/usr/bin/env python
"""
MCP Server Coverage Test

This script tests whether all expected functions from the ipfs_accelerate_py package
are exposed as MCP tools through the currently running MCP server.
"""

import json
import logging
import requests
import tempfile
import os
import sys
from typing import Dict, Any, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Expected tools based on direct_mcp_server.py implementation
EXPECTED_TOOLS = {
    # IPFS tools
    "ipfs_add_file",
    "ipfs_cat",
    "ipfs_files_write",
    "ipfs_files_read",
    
    # Hardware tools
    "get_hardware_info",
    "get_hardware_capabilities",
    
    # Model management tools
    "list_models",
    "create_endpoint",
    "run_inference",
    
    # API tools
    "register_api_key",
    "get_api_keys",
    "get_multiplexer_stats",
    "simulate_api_request",
    
    # Task management tools 
    "start_task",
    "get_task_status",
    "list_tasks",
    
    # Advanced tools
    "throughput_benchmark",
    "quantize_model",
    
    # System tools
    "health_check"
}

class MCPServerCoverageTester:
    """Tests MCP server API coverage"""
    
    def __init__(self, host: str = "localhost", port: int = 3000):
        """Initialize the tester"""
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        
        # Results storage
        self.available_tools = set()
        self.missing_tools = set()
        self.working_tools = set()
        self.failing_tools = dict()
    
    def test_server_availability(self) -> bool:
        """Test if server is available"""
        try:
            response = self.session.get(f"{self.base_url}/tools")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Server not available: {e}")
            return False
    
    def get_available_tools(self) -> Set[str]:
        """Get the available tools from the server"""
        try:
            response = self.session.get(f"{self.base_url}/tools")
            response.raise_for_status()
            tools = response.json().get("tools", [])
            self.available_tools = set(tools)
            return self.available_tools
        except Exception as e:
            logger.error(f"Error getting available tools: {e}")
            return set()
    
    def test_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test a specific tool with optional arguments"""
        if arguments is None:
            arguments = {}
            
        try:
            logger.info(f"Testing tool: {tool_name}")
            payload = {
                "tool_name": tool_name,
                "arguments": arguments
            }
            
            response = self.session.post(
                f"{self.base_url}/call_tool",
                json=payload
            )
            
            response.raise_for_status()
            result = response.json().get("result", {})
            self.working_tools.add(tool_name)
            return result
        except Exception as e:
            self.failing_tools[tool_name] = str(e)
            logger.warning(f"Tool {tool_name} failed: {e}")
            return {}
    
    def test_ipfs_tools(self) -> Dict[str, bool]:
        """Test IPFS-related tools"""
        results = {}
        
        # Create a test file
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp:
            temp.write("Test content for IPFS tools")
            temp_path = temp.name
        
        logger.info(f"Created test file at {temp_path}")
        
        # Test ipfs_add_file
        add_result = self.test_tool("ipfs_add_file", {"path": temp_path})
        results["ipfs_add_file"] = bool(add_result)
        
        # If we have a CID, test ipfs_cat
        if "cid" in add_result:
            cat_result = self.test_tool("ipfs_cat", {"cid": add_result["cid"]})
            results["ipfs_cat"] = bool(cat_result)
        
        # Test ipfs_files_write
        write_result = self.test_tool("ipfs_files_write", {
            "path": "/test/file.txt",
            "content": "Test content for MFS"
        })
        results["ipfs_files_write"] = bool(write_result)
        
        # Test ipfs_files_read
        read_result = self.test_tool("ipfs_files_read", {"path": "/test/file.txt"})
        results["ipfs_files_read"] = bool(read_result)
        
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return results
    
    def test_hardware_tools(self) -> Dict[str, bool]:
        """Test hardware-related tools"""
        results = {}
        
        # Test get_hardware_info
        info_result = self.test_tool("get_hardware_info")
        results["get_hardware_info"] = bool(info_result)
        
        # Test get_hardware_capabilities
        cap_result = self.test_tool("get_hardware_capabilities")
        results["get_hardware_capabilities"] = bool(cap_result)
        
        return results
    
    def test_model_tools(self) -> Dict[str, bool]:
        """Test model-related tools"""
        results = {}
        
        # Test list_models
        list_result = self.test_tool("list_models")
        results["list_models"] = bool(list_result)
        
        # Test create_endpoint
        endpoint_result = self.test_tool("create_endpoint", {"model_name": "bert-base-uncased"})
        results["create_endpoint"] = bool(endpoint_result)
        
        # If we have an endpoint, test inference
        if "endpoint_id" in endpoint_result:
            inference_result = self.test_tool("run_inference", {
                "endpoint_id": endpoint_result["endpoint_id"],
                "inputs": ["Test input for inference"]
            })
            results["run_inference"] = bool(inference_result)
        
        return results
    
    def test_api_tools(self) -> Dict[str, bool]:
        """Test API-related tools"""
        results = {}
        
        # Test register_api_key
        reg_result = self.test_tool("register_api_key", {
            "provider": "openai",
            "key": "test-api-key-1234"
        })
        results["register_api_key"] = bool(reg_result)
        
        # Test get_api_keys
        keys_result = self.test_tool("get_api_keys")
        results["get_api_keys"] = bool(keys_result)
        
        # Test get_multiplexer_stats
        stats_result = self.test_tool("get_multiplexer_stats")
        results["get_multiplexer_stats"] = bool(stats_result)
        
        # Test simulate_api_request
        sim_result = self.test_tool("simulate_api_request", {
            "provider": "openai", 
            "prompt": "Test prompt for API simulation"
        })
        results["simulate_api_request"] = bool(sim_result)
        
        return results
    
    def test_task_tools(self) -> Dict[str, bool]:
        """Test task management tools"""
        results = {}
        
        # Test start_task
        task_result = self.test_tool("start_task", {
            "task_type": "model_download",
            "params": {"model_name": "test-model"}
        })
        results["start_task"] = bool(task_result)
        
        # If we have a task ID, test status
        if "task_id" in task_result:
            status_result = self.test_tool("get_task_status", {
                "task_id": task_result["task_id"]
            })
            results["get_task_status"] = bool(status_result)
        
        # Test list_tasks
        list_result = self.test_tool("list_tasks")
        results["list_tasks"] = bool(list_result)
        
        return results
    
    def test_advanced_tools(self) -> Dict[str, bool]:
        """Test advanced optimization tools"""
        results = {}
        
        # Test throughput_benchmark
        bench_result = self.test_tool("throughput_benchmark", {
            "model_name": "test-model"
        })
        results["throughput_benchmark"] = bool(bench_result)
        
        # Test quantize_model
        quant_result = self.test_tool("quantize_model", {
            "model_path": "/tmp/test-model",
            "bits": 4
        })
        results["quantize_model"] = bool(quant_result)
        
        return results
    
    def test_system_tools(self) -> Dict[str, bool]:
        """Test system-related tools"""
        results = {}
        
        # Test health_check
        health_result = self.test_tool("health_check")
        results["health_check"] = bool(health_result)
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return a comprehensive report"""
        # First check if server is available
        if not self.test_server_availability():
            return {
                "status": "error",
                "message": "Server not available"
            }
        
        # Get available tools
        available_tools = self.get_available_tools()
        
        # Calculate missing tools
        self.missing_tools = EXPECTED_TOOLS - available_tools
        
        # Test all available tools
        logger.info("Testing IPFS tools...")
        self.test_ipfs_tools()
        
        logger.info("Testing hardware tools...")
        self.test_hardware_tools()
        
        logger.info("Testing model tools...")
        self.test_model_tools()
        
        logger.info("Testing API tools...")
        self.test_api_tools()
        
        logger.info("Testing task tools...")
        self.test_task_tools()
        
        logger.info("Testing advanced tools...")
        self.test_advanced_tools()
        
        logger.info("Testing system tools...")
        self.test_system_tools()
        
        # Compile report
        report = {
            "status": "success",
            "available_tools": list(self.available_tools),
            "missing_tools": list(self.missing_tools),
            "working_tools": list(self.working_tools),
            "failing_tools": self.failing_tools,
            "summary": {
                "total_expected": len(EXPECTED_TOOLS),
                "total_available": len(self.available_tools),
                "total_missing": len(self.missing_tools),
                "total_working": len(self.working_tools),
                "total_failing": len(self.failing_tools),
                "coverage_percentage": round(len(self.available_tools) / len(EXPECTED_TOOLS) * 100, 2)
            }
        }
        
        return report

def print_report(report: Dict[str, Any]) -> None:
    """Print a formatted report"""
    print("\n" + "=" * 80)
    print("MCP SERVER COVERAGE REPORT")
    print("=" * 80)
    
    if report.get("status") == "error":
        print(f"ERROR: {report.get('message')}")
        return
    
    # Print summary
    summary = report.get("summary", {})
    print(f"\nSUMMARY:")
    print(f"  Expected Tools: {summary.get('total_expected', 0)}")
    print(f"  Available Tools: {summary.get('total_available', 0)}")
    print(f"  Missing Tools: {summary.get('total_missing', 0)}")
    print(f"  Working Tools: {summary.get('total_working', 0)}")
    print(f"  Failing Tools: {summary.get('total_failing', 0)}")
    print(f"  Coverage: {summary.get('coverage_percentage', 0)}%")
    
    # Print available tools
    available = report.get("available_tools", [])
    print(f"\nAVAILABLE TOOLS ({len(available)}):")
    for tool in sorted(available):
        print(f"  - {tool}")
    
    # Print missing tools
    missing = report.get("missing_tools", [])
    print(f"\nMISSING TOOLS ({len(missing)}):")
    for tool in sorted(missing):
        print(f"  - {tool}")
    
    # Print working tools
    working = report.get("working_tools", [])
    print(f"\nWORKING TOOLS ({len(working)}):")
    for tool in sorted(working):
        print(f"  - {tool}")
    
    # Print failing tools
    failing = report.get("failing_tools", {})
    print(f"\nFAILING TOOLS ({len(failing)}):")
    for tool, error in sorted(failing.items()):
        print(f"  - {tool}: {error}")
    
    print("\n" + "=" * 80)

def main() -> int:
    """Main entry point"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test MCP server coverage")
    parser.add_argument("--host", default="localhost", help="MCP server host")
    parser.add_argument("--port", type=int, default=3000, help="MCP server port")
    args = parser.parse_args()
    
    # Run tests
    tester = MCPServerCoverageTester(host=args.host, port=args.port)
    logger.info(f"Testing MCP server at {args.host}:{args.port}...")
    report = tester.run_all_tests()
    
    # Print report
    print_report(report)
    
    # Determine exit code based on coverage
    summary = report.get("summary", {})
    coverage = summary.get("coverage_percentage", 0)
    
    if coverage == 100:
        return 0  # Success
    else:
        return 1  # Incomplete coverage

if __name__ == "__main__":
    sys.exit(main())
