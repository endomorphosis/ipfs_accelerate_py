#!/usr/bin/env python3
"""
Comprehensive Test Suite for IPFS Accelerate MCP Tools

This test suite provides full coverage for all the MCP tools that are exposed
for the ipfs_accelerate_py package.
"""

import os
import sys
import json
import time
import unittest
import tempfile
import requests
from typing import Dict, Any

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_ipfs_accelerate_mcp_tools")

# Default MCP server URL
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000")


class MCPClient:
    """Client for interacting with the MCP server."""
    
    def __init__(self, server_url: str = MCP_SERVER_URL):
        """Initialize the client with server URL."""
        self.server_url = server_url
        
    def get_tools_list(self):
        """Get the list of available tools."""
        response = requests.get(f"{self.server_url}/tools")
        response.raise_for_status()
        return response.json()["tools"]
        
    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None):
        """Call a tool with arguments."""
        if arguments is None:
            arguments = {}
            
        response = requests.post(
            f"{self.server_url}/call_tool",
            json={"tool_name": tool_name, "arguments": arguments}
        )
        response.raise_for_status()
        return response.json()["result"]


class IPFSOperationsTest(unittest.TestCase):
    """Test IPFS operations via MCP tools."""
    
    @classmethod
    def setUpClass(cls):
        cls.client = MCPClient()
        cls.available_tools = cls.client.get_tools_list()
        cls.test_file = None
        cls.test_cid = None
        
        # Create a temp file for testing
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as tmp:
            tmp.write("Test content for IPFS operations")
        cls.test_file = path
    
    @classmethod
    def tearDownClass(cls):
        # Clean up the temp file
        if cls.test_file and os.path.exists(cls.test_file):
            os.unlink(cls.test_file)
    
    def setUp(self):
        # Skip tests if MCP server is not available
        try:
            self.available_tools = self.client.get_tools_list()
        except requests.exceptions.RequestException:
            self.skipTest("MCP server is not available")
    
    def test_01_ipfs_add_file(self):
        """Test adding a file to IPFS."""
        if "ipfs_add_file" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("ipfs_add_file", {"path": self.test_file})
        self.assertIn("cid", result)
        self.assertTrue(result["cid"].startswith("Qm") or result["cid"].startswith("bafy"))
        
        # Store the CID for later tests
        self.__class__.test_cid = result["cid"]
    
    def test_02_ipfs_cat(self):
        """Test retrieving file content from IPFS by CID."""
        if "ipfs_cat" not in self.available_tools:
            self.skipTest("Tool not available")
        
        if not self.__class__.test_cid:
            self.skipTest("No test CID available")
        
        result = self.client.call_tool("ipfs_cat", {"cid": self.__class__.test_cid})
        self.assertIn("content", result)
        self.assertEqual(result["content"], "Test content for IPFS operations")
    
    def test_03_ipfs_files_mkdir(self):
        """Test creating a directory in IPFS MFS."""
        if "ipfs_files_mkdir" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("ipfs_files_mkdir", {"path": "/test_mcp_dir"})
        self.assertTrue(result["success"])
    
    def test_04_ipfs_files_write(self):
        """Test writing to a file in IPFS MFS."""
        if "ipfs_files_write" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool(
            "ipfs_files_write", 
            {
                "path": "/test_mcp_dir/test.txt",
                "content": "MFS test content"
            }
        )
        self.assertTrue(result["success"])
    
    def test_05_ipfs_files_read(self):
        """Test reading a file from IPFS MFS."""
        if "ipfs_files_read" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("ipfs_files_read", {"path": "/test_mcp_dir/test.txt"})
        self.assertIn("content", result)
        self.assertEqual(result["content"], "MFS test content")
    
    def test_06_ipfs_files_ls(self):
        """Test listing files in IPFS MFS."""
        if "ipfs_files_ls" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("ipfs_files_ls", {"path": "/test_mcp_dir"})
        self.assertIn("entries", result)
        self.assertIn("test.txt", result["entries"])
    
    def test_07_ipfs_files_stat(self):
        """Test getting stats for a file in IPFS MFS."""
        if "ipfs_files_stat" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("ipfs_files_stat", {"path": "/test_mcp_dir/test.txt"})
        self.assertIn("size", result)
        self.assertIn("hash", result)
        self.assertEqual(result["size"], len("MFS test content"))
    
    def test_08_ipfs_pin_add(self):
        """Test pinning content in IPFS."""
        if "ipfs_pin_add" not in self.available_tools or not self.__class__.test_cid:
            self.skipTest("Tool not available or no test CID")
        
        result = self.client.call_tool("ipfs_pin_add", {"cid": self.__class__.test_cid})
        self.assertTrue(result["success"])
    
    def test_09_ipfs_pin_ls(self):
        """Test listing pins in IPFS."""
        if "ipfs_pin_ls" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("ipfs_pin_ls", {})
        self.assertIn("pins", result)
        if self.__class__.test_cid:
            self.assertIn(self.__class__.test_cid, result["pins"])
    
    def test_99_ipfs_pin_rm(self):
        """Test removing a pin in IPFS."""
        if "ipfs_pin_rm" not in self.available_tools or not self.__class__.test_cid:
            self.skipTest("Tool not available or no test CID")
        
        result = self.client.call_tool("ipfs_pin_rm", {"cid": self.__class__.test_cid})
        self.assertTrue(result["success"])


class HardwareDetectionTest(unittest.TestCase):
    """Test hardware detection via MCP tools."""
    
    def setUp(self):
        self.client = MCPClient()
        # Skip tests if MCP server is not available
        try:
            self.available_tools = self.client.get_tools_list()
        except requests.exceptions.RequestException:
            self.skipTest("MCP server is not available")
    
    def test_get_hardware_info(self):
        """Test getting hardware information."""
        if "get_hardware_info" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("get_hardware_info", {})
        self.assertIn("cpu", result)
        self.assertIn("ram", result)
        
    def test_get_hardware_capabilities(self):
        """Test getting hardware capabilities."""
        if "get_hardware_capabilities" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("get_hardware_capabilities", {})
        self.assertIn("has_gpu", result)
        self.assertIn("has_webgpu", result)
        self.assertIn("has_webnn", result)


class ModelOperationsTest(unittest.TestCase):
    """Test model operations via MCP tools."""
    
    @classmethod
    def setUpClass(cls):
        cls.client = MCPClient()
        cls.endpoint_id = None
        cls.available_models = []
    
    def setUp(self):
        # Skip tests if MCP server is not available
        try:
            self.available_tools = self.client.get_tools_list()
        except requests.exceptions.RequestException:
            self.skipTest("MCP server is not available")
    
    def test_01_list_models(self):
        """Test listing available models."""
        if "list_models" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("list_models", {})
        self.assertIn("models", result)
        if result["models"]:
            # Store available models for later tests
            self.__class__.available_models = list(result["models"].keys())
    
    def test_02_get_model_info(self):
        """Test getting model information."""
        if "get_model_info" not in self.available_tools or not self.__class__.available_models:
            self.skipTest("Tool not available or no models available")
        
        model_name = self.__class__.available_models[0]
        result = self.client.call_tool("get_model_info", {"model_name": model_name})
        self.assertIn("model_type", result)
        self.assertIn("size", result)
    
    def test_03_create_endpoint(self):
        """Test creating a model endpoint."""
        if "create_endpoint" not in self.available_tools or not self.__class__.available_models:
            self.skipTest("Tool not available or no models available")
        
        model_name = self.__class__.available_models[0]
        result = self.client.call_tool(
            "create_endpoint", 
            {
                "model_name": model_name,
                "device": "cpu",
                "max_batch_size": 16
            }
        )
        self.assertIn("endpoint_id", result)
        # Store the endpoint ID for later tests
        self.__class__.endpoint_id = result["endpoint_id"]
    
    def test_04_list_endpoints(self):
        """Test listing model endpoints."""
        if "list_endpoints" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("list_endpoints", {})
        self.assertIn("endpoints", result)
        if self.__class__.endpoint_id:
            self.assertIn(self.__class__.endpoint_id, [ep["id"] for ep in result["endpoints"]])
    
    def test_05_run_inference(self):
        """Test running model inference."""
        if "run_inference" not in self.available_tools or not self.__class__.endpoint_id:
            self.skipTest("Tool not available or no endpoint ID")
        
        result = self.client.call_tool(
            "run_inference", 
            {
                "endpoint_id": self.__class__.endpoint_id,
                "inputs": ["This is a test input"]
            }
        )
        self.assertIn("outputs", result)
    
    def test_99_delete_endpoint(self):
        """Test deleting a model endpoint."""
        if "delete_endpoint" not in self.available_tools or not self.__class__.endpoint_id:
            self.skipTest("Tool not available or no endpoint ID")
        
        result = self.client.call_tool(
            "delete_endpoint", 
            {"endpoint_id": self.__class__.endpoint_id}
        )
        self.assertTrue(result["success"])


class APIMultiplexingTest(unittest.TestCase):
    """Test API multiplexing via MCP tools."""
    
    def setUp(self):
        self.client = MCPClient()
        # Skip tests if MCP server is not available
        try:
            self.available_tools = self.client.get_tools_list()
        except requests.exceptions.RequestException:
            self.skipTest("MCP server is not available")
    
    def test_01_register_api_key(self):
        """Test registering an API key."""
        if "register_api_key" not in self.available_tools:
            self.skipTest("Tool not available")
        
        # Using a test key
        result = self.client.call_tool(
            "register_api_key", 
            {
                "provider": "test_provider",
                "key": f"test-key-{int(time.time())}",
                "priority": 1
            }
        )
        self.assertTrue(result["success"])
    
    def test_02_get_api_keys(self):
        """Test getting registered API keys."""
        if "get_api_keys" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("get_api_keys", {})
        self.assertIn("keys", result)
        self.assertTrue(any(key["provider"] == "test_provider" for key in result["keys"]))
    
    def test_03_delete_api_key(self):
        """Test deleting an API key."""
        if "delete_api_key" not in self.available_tools:
            self.skipTest("Tool not available")
        
        # Get the keys first
        keys = self.client.call_tool("get_api_keys", {})["keys"]
        test_keys = [key for key in keys if key["provider"] == "test_provider"]
        
        if not test_keys:
            self.skipTest("No test keys found")
        
        result = self.client.call_tool(
            "delete_api_key", 
            {"provider": "test_provider", "key_id": test_keys[0]["id"]}
        )
        self.assertTrue(result["success"])


class TaskManagementTest(unittest.TestCase):
    """Test task management via MCP tools."""
    
    @classmethod
    def setUpClass(cls):
        cls.client = MCPClient()
        cls.task_id = None
    
    def setUp(self):
        # Skip tests if MCP server is not available
        try:
            self.available_tools = self.client.get_tools_list()
        except requests.exceptions.RequestException:
            self.skipTest("MCP server is not available")
    
    def test_01_start_task(self):
        """Test starting a task."""
        if "start_task" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool(
            "start_task",
            {
                "task_type": "test_task",
                "params": {"test_param": "test_value"}
            }
        )
        self.assertIn("task_id", result)
        # Store the task ID for later tests
        self.__class__.task_id = result["task_id"]
    
    def test_02_get_task_status(self):
        """Test getting task status."""
        if "get_task_status" not in self.available_tools or not self.__class__.task_id:
            self.skipTest("Tool not available or no task ID")
        
        result = self.client.call_tool(
            "get_task_status", 
            {"task_id": self.__class__.task_id}
        )
        self.assertIn("status", result)
    
    def test_03_list_tasks(self):
        """Test listing tasks."""
        if "list_tasks" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("list_tasks", {})
        self.assertIn("tasks", result)
        if self.__class__.task_id:
            self.assertIn(self.__class__.task_id, [task["id"] for task in result["tasks"]])


class MiscToolsTest(unittest.TestCase):
    """Test miscellaneous tools that don't fit in other categories."""
    
    def setUp(self):
        self.client = MCPClient()
        # Skip tests if MCP server is not available
        try:
            self.available_tools = self.client.get_tools_list()
        except requests.exceptions.RequestException:
            self.skipTest("MCP server is not available")
    
    def test_health_check(self):
        """Test health check tool."""
        if "health_check" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("health_check", {})
        self.assertIn("status", result)
        self.assertEqual(result["status"], "ok")
    
    def test_get_version(self):
        """Test getting version information."""
        if "get_version" not in self.available_tools:
            self.skipTest("Tool not available")
        
        result = self.client.call_tool("get_version", {})
        self.assertIn("version", result)
        self.assertIn("ipfs_version", result)


def main():
    """Run the test suite."""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(IPFSOperationsTest))
    suite.addTests(loader.loadTestsFromTestCase(HardwareDetectionTest))
    suite.addTests(loader.loadTestsFromTestCase(ModelOperationsTest))
    suite.addTests(loader.loadTestsFromTestCase(APIMultiplexingTest))
    suite.addTests(loader.loadTestsFromTestCase(TaskManagementTest))
    suite.addTests(loader.loadTestsFromTestCase(MiscToolsTest))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate a report
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "api_test_results",
        f"mcp_tests_{int(time.time())}.json"
    )
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Create the report
    report = {
        "timestamp": time.time(),
        "tests_run": result.testsRun,
        "errors": len(result.errors),
        "failures": len(result.failures),
        "skipped": len(result.skipped),
        "details": {
            "errors": [{"test": str(test), "error": str(error)} for test, error in result.errors],
            "failures": [{"test": str(test), "failure": str(failure)} for test, failure in result.failures],
            "skipped": [{"test": str(test), "reason": str(reason)} for test, reason in result.skipped]
        }
    }
    
    # Save the report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTest report saved to: {report_path}")
    
    # Return success or failure
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
