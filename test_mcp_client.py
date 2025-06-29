#!/usr/bin/env python3
"""
Standalone MCP Client for testing IPFS Accelerate MCP Tools

This module provides a command-line interface for testing MCP tools
without writing unit tests, allowing for quick verification of functionality.
"""

import os
import sys
import json
import time
import uuid
import logging
import argparse
import requests
from typing import Dict, List, Any, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_mcp_client")


class MCPClient:
    """Client for interacting with the MCP server."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """Initialize the client with server URL."""
        self.server_url = server_url
        self.tools = None
    
    def verify_server_connection(self) -> bool:
        """Verify the connection to the server."""
        try:
            response = requests.get(f"{self.server_url}/tools")
            response.raise_for_status()
            logger.info(f"Connected to MCP server at {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    def get_tools_list(self) -> List[str]:
        """Get a list of available tools."""
        try:
            response = requests.get(f"{self.server_url}/tools")
            response.raise_for_status()
            tools = response.json()["tools"]
            self.tools = tools
            return tools
        except Exception as e:
            logger.error(f"Failed to get tools list: {e}")
            raise
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a tool with arguments."""
        if arguments is None:
            arguments = {}
        
        if self.tools is None:
            self.tools = self.get_tools_list()
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        try:
            response = requests.post(
                f"{self.server_url}/call_tool",
                json={"tool_name": tool_name, "arguments": arguments}
            )
            response.raise_for_status()
            result = response.json()["result"]
            return result
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            raise


class MCPToolTester:
    """Test MCP tools functionality."""
    
    def __init__(self, client: MCPClient, output_file: Optional[str] = None):
        """Initialize the tester."""
        self.client = client
        self.output_file = output_file
        self.results = {
            "timestamp": time.time(),
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        }
    
    def run_test(self, test_name: str, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a test for a tool."""
        if arguments is None:
            arguments = {}
        
        self.results["summary"]["total"] += 1
        
        try:
            logger.info(f"Running test: {test_name} (tool: {tool_name})")
            start_time = time.time()
            result = self.client.call_tool(tool_name, arguments)
            elapsed = time.time() - start_time
            
            success = result.get("success", False) if isinstance(result, dict) else (result is not None)
            
            test_result = {
                "name": test_name,
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "elapsed": elapsed,
                "success": success
            }
            
            if success:
                self.results["summary"]["passed"] += 1
                logger.info(f"✅ Test passed: {test_name} ({elapsed:.2f}s)")
            else:
                self.results["summary"]["failed"] += 1
                logger.error(f"❌ Test failed: {test_name} ({elapsed:.2f}s)")
            
            self.results["tests"][test_name] = test_result
            return test_result
            
        except Exception as e:
            self.results["summary"]["failed"] += 1
            logger.error(f"❌ Test error: {test_name} - {str(e)}")
            
            test_result = {
                "name": test_name,
                "tool": tool_name,
                "arguments": arguments,
                "error": str(e),
                "success": False
            }
            
            self.results["tests"][test_name] = test_result
            return test_result
    
    def skip_test(self, test_name: str, reason: str) -> None:
        """Skip a test."""
        self.results["summary"]["total"] += 1
        self.results["summary"]["skipped"] += 1
        
        test_result = {
            "name": test_name,
            "skipped": True,
            "reason": reason
        }
        
        self.results["tests"][test_name] = test_result
        logger.warning(f"⚠️ Test skipped: {test_name} - {reason}")
    
    def save_results(self) -> None:
        """Save test results to a file."""
        if not self.output_file:
            return
        
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def print_summary(self) -> None:
        """Print a summary of the test results."""
        summary = self.results["summary"]
        
        print("\n" + "=" * 60)
        print(f"Test Summary")
        print("-" * 60)
        print(f"Total tests:  {summary['total']}")
        print(f"Passed:       {summary['passed']}")
        print(f"Failed:       {summary['failed']}")
        print(f"Skipped:      {summary['skipped']}")
        print("=" * 60 + "\n")


class TestRunner:
    """Run tests for MCP tools."""
    
    def __init__(self, server_url: str, output_file: Optional[str] = None):
        """Initialize the test runner."""
        self.client = MCPClient(server_url)
        self.tester = MCPToolTester(self.client, output_file)
    
    def test_ipfs_operations(self) -> None:
        """Test IPFS operations."""
        logger.info("🧪 Running IPFS operations tests...")
        
        tools = self.client.get_tools_list()
        
        # Test ipfs_add_file if available
        if "ipfs_add_file" in tools:
            # Create a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
                tmp.write("Test content for IPFS")
                tmp_path = tmp.name
            
            # Add file to IPFS
            result = self.tester.run_test(
                "ipfs_add_file",
                "ipfs_add_file",
                {"path": tmp_path}
            )
            
            # Clean up
            os.unlink(tmp_path)
            
            # Test ipfs_cat if available and add_file succeeded
            if "ipfs_cat" in tools and result["success"] and "cid" in result["result"]:
                cid = result["result"]["cid"]
                self.tester.run_test(
                    "ipfs_cat",
                    "ipfs_cat",
                    {"cid": cid}
                )
        else:
            self.tester.skip_test("ipfs_add_file", "Tool not available")
        
        # Test ipfs_files operations if available
        if "ipfs_files_write" in tools and "ipfs_files_read" in tools:
            # Write to MFS
            mfs_path = f"/mcp-test-{uuid.uuid4().hex[:8]}.txt"
            content = f"Test MFS content {time.time()}"
            
            write_result = self.tester.run_test(
                "ipfs_files_write",
                "ipfs_files_write",
                {"path": mfs_path, "content": content}
            )
            
            if write_result["success"]:
                # Read from MFS
                self.tester.run_test(
                    "ipfs_files_read",
                    "ipfs_files_read",
                    {"path": mfs_path}
                )
        else:
            self.tester.skip_test("ipfs_files_operations", "Tools not available")
    
    def test_hardware_detection(self) -> None:
        """Test hardware detection functionality."""
        logger.info("🧪 Running hardware detection tests...")
        
        tools = self.client.get_tools_list()
        
        # Test get_hardware_info if available
        if "get_hardware_info" in tools:
            self.tester.run_test(
                "get_hardware_info",
                "get_hardware_info",
                {}
            )
        else:
            self.tester.skip_test("get_hardware_info", "Tool not available")
        
        # Test get_hardware_capabilities if available
        if "get_hardware_capabilities" in tools:
            self.tester.run_test(
                "get_hardware_capabilities",
                "get_hardware_capabilities",
                {}
            )
        else:
            self.tester.skip_test("get_hardware_capabilities", "Tool not available")
    
    def test_model_operations(self) -> None:
        """Test model operations."""
        logger.info("🧪 Running model operations tests...")
        
        tools = self.client.get_tools_list()
        
        # Test list_models if available
        if "list_models" in tools:
            models_result = self.tester.run_test(
                "list_models",
                "list_models",
                {}
            )
            
            # Test create_endpoint if we have models
            if (models_result["success"] and 
                "models" in models_result["result"] and 
                models_result["result"]["models"] and
                "create_endpoint" in tools):
                
                model_name = next(iter(models_result["result"]["models"].keys()))
                
                endpoint_result = self.tester.run_test(
                    "create_endpoint",
                    "create_endpoint",
                    {
                        "model_name": model_name,
                        "device": "cpu",
                        "max_batch_size": 16
                    }
                )
                
                # Test run_inference if we have an endpoint
                if (endpoint_result["success"] and 
                    "endpoint_id" in endpoint_result["result"] and
                    "run_inference" in tools):
                    
                    self.tester.run_test(
                        "run_inference",
                        "run_inference",
                        {
                            "endpoint_id": endpoint_result["result"]["endpoint_id"],
                            "inputs": ["This is a test input", "Another test input"]
                        }
                    )
                else:
                    self.tester.skip_test("run_inference", "No endpoint available")
            else:
                self.tester.skip_test("create_endpoint", "No models available or tool not available")
        else:
            self.tester.skip_test("list_models", "Tool not available")
    
    def test_api_multiplexing(self) -> None:
        """Test API multiplexing tools."""
        logger.info("🧪 Running API multiplexing tests...")
        
        tools = self.client.get_tools_list()
        
        # Test register_api_key if available
        if "register_api_key" in tools:
            provider = "openai"
            
            self.tester.run_test(
                "register_api_key",
                "register_api_key",
                {
                    "provider": provider,
                    "key": f"test-key-{provider}-{int(time.time())}",
                    "priority": 1
                }
            )
        else:
            self.tester.skip_test("register_api_key", "Tool not available")
        
        # Test get_api_keys if available
        if "get_api_keys" in tools:
            self.tester.run_test(
                "get_api_keys",
                "get_api_keys",
                {}
            )
        else:
            self.tester.skip_test("get_api_keys", "Tool not available")
    
    def test_task_management(self) -> None:
        """Test task management functionality."""
        logger.info("🧪 Running task management tests...")
        
        tools = self.client.get_tools_list()
        
        # Test start_task if available
        if "start_task" in tools:
            task_result = self.tester.run_test(
                "start_task",
                "start_task",
                {
                    "task_type": "download_model",
                    "params": {"model_name": "bert-base-uncased"}
                }
            )
            
            # Test get_task_status if start_task succeeded
            if task_result["success"] and "task_id" in task_result["result"] and "get_task_status" in tools:
                self.tester.run_test(
                    "get_task_status",
                    "get_task_status",
                    {"task_id": task_result["result"]["task_id"]}
                )
            else:
                self.tester.skip_test("get_task_status", "No task ID available or tool not available")
        else:
            self.tester.skip_test("start_task", "Tool not available")
        
        # Test list_tasks if available
        if "list_tasks" in tools:
            self.tester.run_test(
                "list_tasks",
                "list_tasks",
                {}
            )
        else:
            self.tester.skip_test("list_tasks", "Tool not available")
    
    def test_all(self) -> None:
        """Run all tests."""
        logger.info("🧪 Running all tests...")
        
        self.test_ipfs_operations()
        self.test_hardware_detection()
        self.test_model_operations()
        self.test_api_multiplexing()
        self.test_task_management()
        
        # Test health_check if available
        tools = self.client.get_tools_list()
        if "health_check" in tools:
            self.tester.run_test(
                "health_check",
                "health_check",
                {}
            )
        else:
            self.tester.skip_test("health_check", "Tool not available")
        
        self.tester.save_results()
        self.tester.print_summary()


def main():
    """Run the client."""
    parser = argparse.ArgumentParser(description="Test MCP tools.")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="MCP server URL")
    parser.add_argument("--output", type=str, help="Output file for test results")
    parser.add_argument("--test", type=str, default="all", choices=["all", "ipfs", "hardware", "model", "api", "task", "throughput"], help="Test to run")
    args = parser.parse_args()

    try:
        runner = TestRunner(args.server_url, args.output)
        
        if not runner.client.verify_server_connection():
            logger.error("Could not connect to MCP server.")
            return 1
        
        logger.info(f"Running {args.test} tests...")
        
        if args.test == "all":
            runner.test_all()
        elif args.test == "ipfs":
            runner.test_ipfs_operations()
        elif args.test == "hardware":
            runner.test_hardware_detection()
        elif args.test == "model":
            runner.test_model_operations()
        elif args.test == "api":
            runner.test_api_multiplexing()
        elif args.test == "task":
            runner.test_task_management()
        
        runner.tester.save_results()
        runner.tester.print_summary()
        
        if runner.tester.results["summary"]["failed"] > 0:
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Test interrupted.")
        return 130
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
