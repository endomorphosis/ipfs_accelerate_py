#!/usr/bin/env python3
"""
Comprehensive Test Suite for IPFS Accelerate MCP Tools

This module contains unit tests for all MCP tools exposed by the IPFS Accelerate
Python package, providing full coverage of functionality.
"""

import os
import sys
import json
import time
import uuid
import unittest
import tempfile
import requests
import logging
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_mcp_tools")

# All expected tools by category for full coverage testing
EXPECTED_TOOLS = {
    "ipfs_core": [
        "ipfs_add_file",        # Add a file to IPFS
        "ipfs_cat",             # Retrieve content from IPFS
        "ipfs_files_write",     # Write content to the IPFS MFS
        "ipfs_files_read",      # Read content from the IPFS MFS
        "ipfs_files_mkdir",     # Create directories in the IPFS MFS
        "ipfs_files_ls",        # List files in the IPFS MFS
        "ipfs_files_rm",        # Remove files from the IPFS MFS
        "ipfs_files_cp",        # Copy files in the IPFS MFS
        "ipfs_files_mv",        # Move files in the IPFS MFS
        "ipfs_name_publish",    # Publish an IPNS name
        "ipfs_name_resolve"     # Resolve an IPNS name
    ],
    "hardware_detection": [
        "get_hardware_info",              # Get basic hardware information
        "get_hardware_capabilities",      # Get detailed hardware capabilities
        "test_hardware_compatibility",    # Test hardware compatibility
        "recommend_hardware_settings"     # Get recommended hardware settings
    ],
    "model_operations": [
        "list_models",             # List available models
        "create_endpoint",         # Create a model inference endpoint
        "run_inference",           # Run inference on a model
        "download_model",          # Download a model
        "get_model_metadata",      # Get model metadata
        "list_endpoints",          # List all available endpoints
        "delete_endpoint"          # Delete an endpoint
    ],
    "api_multiplexing": [
        "register_api_key",        # Register an API key for multiplexing
        "get_api_keys",            # Get registered API keys
        "get_multiplexer_stats",   # Get multiplexer statistics
        "simulate_api_request",    # Simulate an API request through the multiplexer
        "configure_multiplexer",   # Configure the API multiplexer
        "set_key_rate_limit"       # Set rate limit for an API key
    ],
    "task_management": [
        "start_task",              # Start a background task
        "get_task_status",         # Get task status
        "list_tasks",              # List all tasks
        "cancel_task",             # Cancel a running task
        "task_priority"            # Set task priority
    ],
    "throughput_optimization": [
        "throughput_benchmark",    # Benchmark model throughput
        "optimize_for_throughput", # Optimize a model for throughput
        "quantize_model",          # Quantize a model for better performance
        "shard_model"              # Shard a model across devices
    ],
    "configuration": [
        "config_set",              # Set a config value
        "config_get",              # Get a config value
        "config_save",             # Save configuration
        "config_list",             # List configuration values
        "config_reset"             # Reset configuration to defaults
    ],
    "backend_management": [
        "backend_list_marketplace_images", # List marketplace images
        "backend_start_container",         # Start a container
        "backend_stop_container",          # Stop a container
        "backend_list_containers",         # List containers
        "backend_docker_tunnel"            # Create a tunnel to a Docker container
    ],
    "health": [
        "health_check"             # Check server health
    ],
    "resource_management": [
        "resource_pool_add",       # Add resource to the pool
        "resource_pool_remove",    # Remove resource from the pool
        "resource_pool_status",    # Get resource pool status
        "resource_pool_assign"     # Assign resource to a task
    ]
}

# Flatten the list of expected tools for easy checking
ALL_EXPECTED_TOOLS = [tool for category in EXPECTED_TOOLS.values() for tool in category]

class MCPClient:
    """Client for interacting with the MCP server."""
    
    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 5):
        """Initialize the client with server URL."""
        self.server_url = server_url
        self.timeout = timeout
        self.tools = None
        self.server_available = False
        self.try_connect()
    
    def try_connect(self) -> bool:
        """Try to connect to the server and check if it's available."""
        try:
            response = requests.get(f"{self.server_url}/", timeout=self.timeout)
            response.raise_for_status()
            self.server_available = True
            logger.info(f"Connected to MCP server at {self.server_url}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to server: {e}")
            self.server_available = False
            return False
    
    def _verify_server_connection(self) -> None:
        """Verify the connection to the server."""
        if not self.try_connect():
            raise ConnectionError(f"Could not connect to MCP server at {self.server_url}")
    
    def get_tools_list(self) -> List[str]:
        """Get a list of available tools."""
        if not self.server_available:
            logger.warning("Server not available, returning empty tools list")
            return []
            
        try:
            response = requests.get(f"{self.server_url}/tools", timeout=self.timeout)
            response.raise_for_status()
            tools = response.json().get("tools", [])
            logger.info(f"Available tools: {', '.join(sorted(tools))}")
            self.tools = tools
            return tools
        except Exception as e:
            logger.error(f"Failed to get tools list: {e}")
            return []
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a tool with arguments."""
        if not self.server_available:
            logger.warning(f"Server not available, cannot call tool {tool_name}")
            return {"error": "Server not available", "success": False}
            
        if arguments is None:
            arguments = {}
        
        if self.tools is None:
            self.tools = self.get_tools_list()
        
        if tool_name not in self.tools:
            logger.warning(f"Tool not found: {tool_name}")
            return {"error": f"Tool not found: {tool_name}", "success": False}
        
        try:
            response = requests.post(
                f"{self.server_url}/mcp/tool/{tool_name}",
                json=arguments,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            return {"error": str(e), "success": False}

class MockResponse:
    """Mock response for requests."""
    
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code
    
    def json(self):
        """Return JSON data."""
        return self.json_data
    
    def raise_for_status(self):
        """Raise HTTPError if status code is 4XX or 5XX."""
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP Error: {self.status_code}")

class BaseMCPToolTest(unittest.TestCase):
    """Base class for MCP tool tests with setup and helper methods."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        # Try different ports in case the server is running on a non-default port
        server_urls = [
            "http://localhost:8000",
            "http://localhost:8001",
            "http://localhost:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8001"
        ]
        
        cls.client = None
        cls.tools = []
        cls.missing_tools = []
        
        # Try each URL until we find a working server
        for url in server_urls:
            try:
                client = MCPClient(url)
                if client.server_available:
                    cls.client = client
                    cls.server_url = url
                    cls.tools = client.get_tools_list()
                    cls.missing_tools = [tool for tool in ALL_EXPECTED_TOOLS if tool not in cls.tools]
                    logger.info(f"Found MCP server at {url} with {len(cls.tools)} tools")
                    break
            except Exception:
                continue
        
        if cls.client is None:
            logger.warning("Could not connect to any MCP server, tests will run in mock mode")
            cls.client = MCPClient()  # Use default URL
            cls.server_url = "http://localhost:8000"
            cls.tools = []
            cls.missing_tools = ALL_EXPECTED_TOOLS
    
    def check_tool_exists(self, tool_name: str) -> bool:
        """Check if a tool exists."""
        return tool_name in self.tools
    
    def check_server_available(self) -> None:
        """Check if the server is available and skip test if not."""
        if not self.client.server_available:
            self.skipTest("MCP server not available")
    
    def create_temp_file(self, content: str = "Test content") -> str:
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        return tmp_path
    
    def create_temp_dir(self) -> str:
        """Create a temporary directory for testing."""
        tmp_dir = tempfile.mkdtemp()
        return tmp_dir
    
    def create_test_file_in_dir(self, directory: str, filename: str, content: str = "Test content") -> str:
        """Create a test file in a directory."""
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def run_test_tool(self, tool_name: str, args: Dict[str, Any] = None, 
                      check_keys: List[str] = None, expected_values: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a test on a tool and check for expected keys and values."""
        if not self.check_tool_exists(tool_name):
            self.skipTest(f"{tool_name} tool not available")
        
        self.check_server_available()
        
        if args is None:
            args = {}
        
        if check_keys is None:
            check_keys = []
        
        if expected_values is None:
            expected_values = {}
        
        result = self.client.call_tool(tool_name, args)
        
        # Check for success unless we explicitly expect failure
        if "success" in result and "success" not in expected_values:
            self.assertTrue(result["success"], f"Tool {tool_name} failed: {result.get('error', 'Unknown error')}")
        
        # Check for expected keys
        for key in check_keys:
            self.assertIn(key, result, f"Expected key '{key}' not found in {tool_name} result")
        
        # Check for expected values
        for key, expected in expected_values.items():
            self.assertEqual(result.get(key), expected, 
                           f"Expected {key}={expected} but got {result.get(key)} in {tool_name} result")
        
        return result

class MockMCPToolTest(unittest.TestCase):
    """Test MCP tools in mock mode when no server is available."""
    
    @patch('requests.get')
    @patch('requests.post')
    def test_mock_ipfs_add_file(self, mock_post, mock_get):
        """Test mocking IPFS add file."""
        # Mock responses
        mock_get.return_value = MockResponse({"tools": ["ipfs_add_file"]}, 200)
        mock_post.return_value = MockResponse({"result": {
            "cid": "QmTest123456789",
            "size": 100,
            "name": "test.txt",
            "success": True
        }}, 200)
        
        # Create client and call tool
        client = MCPClient()
        client.server_available = True  # Force server to be available for test
        client.tools = ["ipfs_add_file"]  # Set tools manually
        
        result = client.call_tool("ipfs_add_file", {"path": "/tmp/test.txt"})
        
        self.assertEqual(result["cid"], "QmTest123456789")
        self.assertTrue(result["success"])
    
    def test_get_missing_tools(self):
        """Test getting missing tools from expected tools list."""
        expected_tools = ["ipfs_add_file", "ipfs_cat", "ipfs_files_write"]
        available_tools = ["ipfs_add_file", "ipfs_files_write"]
        
        missing = get_missing_tools(available_tools, expected_tools)
        self.assertEqual(missing, ["ipfs_cat"])
        
        # Test with empty available tools
        missing = get_missing_tools([], expected_tools)
        self.assertEqual(missing, expected_tools)
        
        # Test with empty expected tools
        missing = get_missing_tools(available_tools, [])
        self.assertEqual(missing, [])
        
        # Test with identical lists
        missing = get_missing_tools(expected_tools, expected_tools)
        self.assertEqual(missing, [])

def get_missing_tools(available_tools: List[str], expected_tools: List[str]) -> List[str]:
    """Get missing tools from an expected list."""
    return [tool for tool in expected_tools if tool not in available_tools]

class MCPTestSuite(unittest.TestSuite):
    """Test suite for MCP tools."""
    
    def __init__(self):
        """Initialize the test suite."""
        super().__init__()
        
        # Use unittest.TestLoader instead of makeSuite
        loader = unittest.TestLoader()
        
        # Add test cases using the loader
        self.addTest(loader.loadTestsFromTestCase(MCPServerConnectionTest))
        self.addTest(loader.loadTestsFromTestCase(IPFSOperationsTest))
        self.addTest(loader.loadTestsFromTestCase(HardwareDetectionTest))
        self.addTest(loader.loadTestsFromTestCase(ModelOperationsTest))
        self.addTest(loader.loadTestsFromTestCase(APIMultiplexingTest))
        self.addTest(loader.loadTestsFromTestCase(TaskManagementTest))
        self.addTest(loader.loadTestsFromTestCase(ThroughputOptimizationTest))
        self.addTest(loader.loadTestsFromTestCase(ConfigurationTest))
        self.addTest(loader.loadTestsFromTestCase(BackendManagementTest))
        self.addTest(loader.loadTestsFromTestCase(ResourceManagementTest))
        self.addTest(loader.loadTestsFromTestCase(MockMCPToolTest))

class MCPServerConnectionTest(BaseMCPToolTest):
    """Test connection to MCP server."""
    
    def test_server_connection(self):
        """Test connection to MCP server."""
        if self.client is None or not self.client.server_available:
            self.skipTest("MCP server not available or client not initialized")
        
        try:
            self.client._verify_server_connection()
        except Exception as e:
            self.fail(f"Connection to MCP server failed: {e}")
    
    def test_tools_list(self):
        """Test getting the tools list."""
        if self.client is None or not self.client.server_available:
            self.skipTest("MCP server not available or client not initialized")
        
        try:
            tools = self.client.get_tools_list()
            self.assertIsInstance(tools, list)
            self.assertTrue(len(tools) > 0)
        except Exception as e:
            self.fail(f"Getting tools list failed: {e}")
    
    def test_health_check(self):
        """Test health check."""
        result = self.run_test_tool("health_check", {}, ["status"], {"status": "healthy"})
        self.assertIn("uptime", result, "Expected uptime in health check result")
        logger.info(f"Server uptime: {result['uptime']:.2f} seconds")

class IPFSOperationsTest(BaseMCPToolTest):
    """Test IPFS operations."""
    
    def test_ipfs_add_file(self):
        """Test adding a file to IPFS with various scenarios."""
        # Test with valid file
        tmp_path = self.create_temp_file()
        try:
            result = self.run_test_tool(
                "ipfs_add_file", 
                {"path": tmp_path}, 
                ["cid", "size", "name"]
            )
            self.__class__.test_cid = result["cid"]
            logger.info(f"Added file to IPFS with CID: {result['cid']}")
        finally:
            os.unlink(tmp_path)

        # Test with non-existent file
        with self.assertRaises(Exception):
            self.run_test_tool("ipfs_add_file", {"path": "/non/existent/file.txt"})

        # Test with empty file
        empty_tmp_path = self.create_temp_file(content="")
        try:
            result = self.run_test_tool(
                "ipfs_add_file", 
                {"path": empty_tmp_path}, 
                ["cid", "size"]
            )
            self.assertEqual(result["size"], 0)
        finally:
            os.unlink(empty_tmp_path)
        
        # ... rest of the class remains the same ...
        # Skip if we don't have a CID from previous test
        if not hasattr(self.__class__, 'test_cid'):
            if self.check_tool_exists("ipfs_add_file"):
                self.test_ipfs_add_file()  # Try to get a CID first
            
            if not hasattr(self.__class__, 'test_cid'):
                self.skipTest("No CID available from previous test")
        
        result = self.run_test_tool("ipfs_cat", {"cid": self.__class__.test_cid})
        self.assertIsNotNone(result)
        logger.info(f"Retrieved content from IPFS with CID: {self.__class__.test_cid}")
    
    def test_ipfs_files_write_and_read(self):
        """Test writing and reading from IPFS MFS."""
        mfs_path = f"/mcp-test-{uuid.uuid4().hex[:8]}.txt"
        content = f"Test MFS content {time.time()}"
        
        # Write to MFS first
        write_result = self.run_test_tool(
            "ipfs_files_write", 
            {
                "path": mfs_path,
                "content": content
            }, 
            ["path", "cid"]
        )
        
        # Then read it back
        read_result = self.run_test_tool("ipfs_files_read", {"path": mfs_path})
        self.assertIn(content, str(read_result))
        
        logger.info(f"Successfully wrote and read from IPFS MFS at path: {mfs_path}")
    
    def test_ipfs_files_mkdir_and_ls(self):
        """Test creating directories and listing files in IPFS MFS."""
        mfs_dir = f"/mcp-test-dir-{uuid.uuid4().hex[:8]}"
        
        # Create directory
        mkdir_result = self.run_test_tool("ipfs_files_mkdir", {"path": mfs_dir})
        
        # Create a test file in the directory
        file_path = f"{mfs_dir}/test.txt"
        content = f"Test content {time.time()}"
        self.run_test_tool("ipfs_files_write", {"path": file_path, "content": content})
        
        # List the directory
        ls_result = self.run_test_tool("ipfs_files_ls", {"path": mfs_dir})
        
        self.assertIn("entries", ls_result)
        self.assertIsInstance(ls_result["entries"], list)
        self.assertEqual(len(ls_result["entries"]), 1)
        self.assertEqual(ls_result["entries"][0]["name"], "test.txt")
        
        logger.info(f"Successfully created and listed directory {mfs_dir}")
    
    def test_ipfs_files_cp_mv_rm(self):
        """Test copying, moving, and removing files in IPFS MFS."""
        # Create source file
        src_path = f"/mcp-test-src-{uuid.uuid4().hex[:8]}.txt"
        content = f"Test content for copy/move {time.time()}"
        self.run_test_tool("ipfs_files_write", {"path": src_path, "content": content})
        
        # Copy file
        dst_path = f"/mcp-test-dst-{uuid.uuid4().hex[:8]}.txt"
        self.run_test_tool("ipfs_files_cp", {"source": src_path, "dest": dst_path})
        
        # Read copied file
        read_result = self.run_test_tool("ipfs_files_read", {"path": dst_path})
        self.assertIn(content, str(read_result))
        
        # Move file
        move_path = f"/mcp-test-moved-{uuid.uuid4().hex[:8]}.txt"
        self.run_test_tool("ipfs_files_mv", {"source": dst_path, "dest": move_path})
        
        # Read moved file
        read_moved = self.run_test_tool("ipfs_files_read", {"path": move_path})
        self.assertIn(content, str(read_moved))
        
        # Remove files
        self.run_test_tool("ipfs_files_rm", {"path": src_path})
        self.run_test_tool("ipfs_files_rm", {"path": move_path})
        
        logger.info("Successfully tested file copy, move, and remove operations")

class HardwareDetectionTest(BaseMCPToolTest):
    """Test hardware detection functionality."""
    
    def test_get_hardware_info(self):
        """Test getting hardware information."""
        result = self.run_test_tool("get_hardware_info", {}, ["cpu"])
        self.assertTrue(result["cpu"].get("available", False))
        logger.info(f"Got hardware info with {result['cpu'].get('cores', 'unknown')} CPU cores")
    
    def test_get_hardware_capabilities(self):
        """Test getting detailed hardware capabilities."""
        result = self.run_test_tool("get_hardware_capabilities", {}, ["cpu"])
        self.assertIn("cores", result["cpu"])
        logger.info(f"Got hardware capabilities with CPU cores: {result['cpu']['cores']}")
    
    def test_test_hardware_compatibility(self):
        """Test hardware compatibility testing."""
        model_name = "gpt2"  # Use a common model name for testing
        result = self.run_test_tool(
            "test_hardware_compatibility", 
            {"model_name": model_name}
        )
        
        self.assertIn("compatible", result)
        self.assertIn("recommendations", result)
        
        logger.info(f"Hardware compatibility for {model_name}: {'Compatible' if result['compatible'] else 'Not compatible'}")
    
    def test_recommend_hardware_settings(self):
        """Test getting hardware setting recommendations."""
        model_name = "bert-base-uncased"  # Use a common model name for testing
        result = self.run_test_tool(
            "recommend_hardware_settings", 
            {"model_name": model_name}
        )
        
        self.assertIn("device", result)
        self.assertIn("batch_size", result)
        
        logger.info(f"Recommended hardware settings for {model_name}: {result['device']} with batch size {result['batch_size']}")

class ModelOperationsTest(BaseMCPToolTest):
    """Test model operations."""
    
    def setUp(self):
        """Set up the test."""
        super().setUp()
        # Try to get models
        if self.check_tool_exists("list_models"):
            try:
                models_result = self.client.call_tool("list_models", {})
                if "models" in models_result and models_result["models"]:
                    self.models = models_result["models"]
                    self.model_names = list(self.models.keys())
                    self.test_model = self.model_names[0] if self.model_names else "bert-base-uncased"
                else:
                    self.models = {}
                    self.model_names = []
                    self.test_model = "bert-base-uncased"
            except Exception:
                self.models = {}
                self.model_names = []
                self.test_model = "bert-base-uncased"
        else:
            self.models = {}
            self.model_names = []
            self.test_model = "bert-base-uncased"
    
    def test_list_models(self):
        """Test listing models."""
        result = self.run_test_tool("list_models", {}, ["models", "count"])
        self.assertIsInstance(result["models"], dict)
        
        if result["models"]:
            logger.info(f"Listed {len(result['models'])} models")
        else:
            logger.info("No models found")
    
    def test_create_endpoint(self):
        """Test creating a model endpoint."""
        result = self.run_test_tool(
            "create_endpoint", 
            {
                "model_name": self.test_model,
                "device": "cpu",
                "max_batch_size": 16
            },
            ["endpoint_id", "model", "device", "max_batch_size", "status"]
        )
        
        # Store endpoint ID for later tests
        self.__class__.test_endpoint_id = result["endpoint_id"]
        
        logger.info(f"Created endpoint {result['endpoint_id']} for model {self.test_model}")
    
    def test_run_inference(self):
        """Test running inference."""
        # Skip if we don't have an endpoint ID from previous test
        if not hasattr(self.__class__, 'test_endpoint_id'):
            if self.check_tool_exists("create_endpoint"):
                self.test_create_endpoint()  # Try to create endpoint first
            
            if not hasattr(self.__class__, 'test_endpoint_id'):
                self.skipTest("No endpoint ID available from previous test")
        
        test_inputs = ["This is a test input", "Another test input"]
        result = self.run_test_tool(
            "run_inference", 
            {
                "endpoint_id": self.__class__.test_endpoint_id,
                "inputs": test_inputs
            }
        )
        
        logger.info(f"Successfully ran inference on endpoint {self.__class__.test_endpoint_id}")
    
    def test_list_endpoints(self):
        """Test listing endpoints."""
        # Create an endpoint first if necessary
        if not hasattr(self.__class__, 'test_endpoint_id') and self.check_tool_exists("create_endpoint"):
            self.test_create_endpoint()
        
        result = self.run_test_tool("list_endpoints", {}, ["endpoints", "count"])
        self.assertIsInstance(result["endpoints"], list)
        self.assertGreaterEqual(len(result["endpoints"]), 0)
        
        logger.info(f"Listed {len(result['endpoints'])} endpoints")
    
    def test_download_model(self):
        """Test downloading a model."""
        model_name = "bert-base-uncased"
        
        result = self.run_test_tool(
            "download_model", 
            {"model_name": model_name}
        )
        
        self.assertIn("model_name", result)
        self.assertEqual(result["model_name"], model_name)
        
        logger.info(f"Downloaded model {model_name}")
    
    def test_delete_endpoint(self):
        """Test deleting an endpoint."""
        # Skip if we don't have an endpoint ID from previous tests
        if not hasattr(self.__class__, 'test_endpoint_id'):
            if self.check_tool_exists("create_endpoint"):
                self.test_create_endpoint()  # Try to create endpoint first
            
            if not hasattr(self.__class__, 'test_endpoint_id'):
                self.skipTest("No endpoint ID available from previous test")
        
        result = self.run_test_tool(
            "delete_endpoint", 
            {"endpoint_id": self.__class__.test_endpoint_id}
        )
        
        logger.info(f"Deleted endpoint {self.__class__.test_endpoint_id}")

class APIMultiplexingTest(BaseMCPToolTest):
    """Test API multiplexing tools."""
    
    def test_register_api_key(self):
        """Test registering an API key."""
        providers = ["openai", "anthropic"]
        
        for provider in providers:
            result = self.run_test_tool(
                "register_api_key", 
                {
                    "provider": provider,
                    "key": f"test-key-{provider}-{int(time.time())}",
                    "priority": 1
                }, 
                ["key_id", "provider"]
            )
            
            logger.info(f"Registered {provider} API key with ID: {result['key_id']}")
    
    def test_get_api_keys(self):
        """Test getting API keys."""
        # Register keys first if necessary
        if self.check_tool_exists("register_api_key"):
            self.test_register_api_key()
        
        result = self.run_test_tool("get_api_keys", {}, ["providers"])
        self.assertIsInstance(result["providers"], (dict, list))
        
        logger.info(f"Got API keys for providers")
    
    def test_get_multiplexer_stats(self):
        """Test getting multiplexer stats."""
        result = self.run_test_tool("get_multiplexer_stats", {}, ["providers"])
        self.assertIsInstance(result["providers"], dict)
        
        logger.info(f"Got multiplexer stats with {result.get('total_requests', 0)} total requests")
    
    def test_simulate_api_request(self):
        """Test simulating an API request."""
        # Register keys first if necessary
        if self.check_tool_exists("register_api_key"):
            self.test_register_api_key()
        
        providers = ["openai", "anthropic"]
        
        for provider in providers:
            result = self.run_test_tool(
                "simulate_api_request", 
                {
                    "provider": provider,
                    "prompt": "What is IPFS?"
                }
            )
            
            # Allow for successful or rate-limited responses
            status = "SUCCESS" if result.get("success", False) else "RATE_LIMITED"
            logger.info(f"Simulated API request to {provider}: {status}")
    
    def test_configure_multiplexer(self):
        """Test configuring the API multiplexer."""
        result = self.run_test_tool(
            "configure_multiplexer", 
            {
                "strategy": "least_loaded",
                "max_retries": 3,
                "retry_delay": 1000
            }
        )
        
        self.assertIn("strategy", result)
        self.assertEqual(result["strategy"], "least_loaded")
        
        logger.info(f"Configured multiplexer with {result['strategy']} strategy")
    
    def test_set_key_rate_limit(self):
        """Test setting a rate limit for an API key."""
        # Register keys first if necessary
        if self.check_tool_exists("register_api_key"):
            self.test_register_api_key()
            
            # Get a key ID
            keys_result = self.client.call_tool("get_api_keys", {})
            if not keys_result.get("providers"):
                self.skipTest("No API keys available")
            
            provider = list(keys_result["providers"].keys())[0]
            key_id = list(keys_result["providers"][provider].keys())[0]
            
            result = self.run_test_tool(
                "set_key_rate_limit", 
                {
                    "provider": provider,
                    "key_id": key_id,
                    "requests_per_minute": 30
                }
            )
            
            self.assertIn("requests_per_minute", result)
            self.assertEqual(result["requests_per_minute"], 30)
            
            logger.info(f"Set rate limit of {result['requests_per_minute']} requests per minute for key {key_id}")

class TaskManagementTest(BaseMCPToolTest):
    """Test task management functionality."""
    
    def test_start_task(self):
        """Test starting a background task."""
        task_configs = [
            ("download_model", {"model_name": "bert-base-uncased"}),
            ("batch_processing", {"batch_size": 32})
        ]
        
        task_ids = []
        
        for task_type, params in task_configs:
            result = self.run_test_tool(
                "start_task", 
                {
                    "task_type": task_type,
                    "params": params
                },
                ["task_id", "type", "status"]
            )
            
            task_ids.append(result["task_id"])
            logger.info(f"Started {task_type} task with ID: {result['task_id']}")
        
        # Store task IDs for later tests
        self.__class__.test_task_ids = task_ids
    
    def test_get_task_status(self):
        """Test getting task status."""
        # Skip if we don't have task IDs from previous test
        if not hasattr(self.__class__, 'test_task_ids'):
            if self.check_tool_exists("start_task"):
                self.test_start_task()  # Try to start tasks first
            
            if not hasattr(self.__class__, 'test_task_ids'):
                self.skipTest("No task IDs available from previous test")
        
        for task_id in self.__class__.test_task_ids:
            result = self.run_test_tool(
                "get_task_status", 
                {"task_id": task_id},
                ["status"]
            )
            
            self.assertIn(result["status"], ["running", "completed", "failed", "pending"])
            
            logger.info(f"Task {task_id} status: {result['status']} ({result.get('progress', 0)}% complete)")
    
    def test_list_tasks(self):
        """Test listing tasks."""
        # Start tasks first if necessary
        if not hasattr(self.__class__, 'test_task_ids') and self.check_tool_exists("start_task"):
            self.test_start_task()
        
        result = self.run_test_tool(
            "list_tasks", 
            {},
            ["active_tasks", "completed_tasks"]
        )
        
        self.assertIsInstance(result["active_tasks"], list)
        self.assertIsInstance(result["completed_tasks"], list)
        
        logger.info(f"Listed {len(result['active_tasks'])} active tasks and {len(result['completed_tasks'])} completed tasks")
    
    def test_cancel_task(self):
        """Test canceling a task."""
        # Start a new task specifically for cancellation
        if self.check_tool_exists("start_task"):
            start_result = self.run_test_tool(
                "start_task", 
                {
                    "task_type": "batch_processing",
                    "params": {"batch_size": 64, "duration": 30}  # Long-running task
                }
            )
            
            task_id = start_result["task_id"]
            
            # Wait a moment to ensure the task has started
            time.sleep(1)
            
            # Now cancel it
            cancel_result = self.run_test_tool(
                "cancel_task", 
                {"task_id": task_id}
            )
            
            self.assertIn("status", cancel_result)
            self.assertEqual(cancel_result["status"], "cancelled")
            
            logger.info(f"Successfully cancelled task {task_id}")

class ThroughputOptimizationTest(BaseMCPToolTest):
    """Test throughput optimization tools."""
    
    def test_throughput_benchmark(self):
        """Test running a throughput benchmark."""
        # Get a model for testing
        model_name = "bert-base-uncased"
        
        if self.check_tool_exists("list_models"):
            models_result = self.client.call_tool("list_models", {})
            if "models" in models_result and models_result["models"]:
                model_name = list(models_result["models"].keys())[0]
        
        result = self.run_test_tool(
            "throughput_benchmark", 
            {
                "model_name": model_name,
                "batch_sizes": [1, 2, 4, 8]
            },
            ["results"]
        )
        
        self.assertIsInstance(result["results"], list)
        self.assertGreater(len(result["results"]), 0)
        
        logger.info(f"Ran throughput benchmark for model {model_name} with {len(result['results'])} batch sizes")
    
    def test_optimize_for_throughput(self):
        """Test optimizing a model for throughput."""
        model_name = "bert-base-uncased"
        
        result = self.run_test_tool(
            "optimize_for_throughput", 
            {
                "model_name": model_name,
                "target_device": "cpu",
                "max_memory": "4GB"
            }
        )
        
        self.assertIn("optimized_model", result)
        self.assertIn("estimated_throughput", result)
        
        logger.info(f"Optimized model {model_name} for throughput, estimated: {result['estimated_throughput']} samples/sec")
    
    def test_quantize_model(self):
        """Test quantizing a model."""
        model_name = "bert-base-uncased"
        
        result = self.run_test_tool(
            "quantize_model", 
            {
                "model_name": model_name,
                "bits": 4
            }
        )
        
        self.assertIn("compression_ratio", result)
        
        logger.info(f"Quantized model {model_name} with {result['compression_ratio']}x compression ratio")
    
    def test_shard_model(self):
        """Test sharding a model across devices."""
        model_name = "bert-base-uncased"
        
        result = self.run_test_tool(
            "shard_model", 
            {
                "model_name": model_name,
                "num_shards": 2
            }
        )
        
        self.assertIn("sharded_model_id", result)
        self.assertIn("memory_per_shard", result)
        
        logger.info(f"Sharded model {model_name} into {result['num_shards']} parts")

class ConfigurationTest(BaseMCPToolTest):
    """Test configuration management tools."""
    
    def test_config_set_and_get(self):
        """Test setting and getting configuration values."""
        # Set a config value
        test_section = "test"
        test_key = f"key_{int(time.time())}"
        test_value = f"value_{uuid.uuid4().hex[:8]}"
        
        set_result = self.run_test_tool(
            "config_set", 
            {
                "section": test_section,
                "key": test_key,
                "value": test_value
            }
        )
        
        # Get the config value
        get_result = self.run_test_tool(
            "config_get", 
            {
                "section": test_section,
                "key": test_key
            },
            ["value"],
            {"value": test_value}
        )
        
        logger.info(f"Successfully set and got config value {test_section}.{test_key} = {test_value}")
    
    def test_config_save(self):
        """Test saving configuration."""
        result = self.run_test_tool("config_save", {})
        logger.info("Successfully saved configuration")
    
    def test_config_list(self):
        """Test listing configuration values."""
        result = self.run_test_tool(
            "config_list", 
            {},
            ["sections"]
        )
        
        self.assertIsInstance(result["sections"], dict)
        
        logger.info(f"Listed configuration with {len(result['sections'])} sections")
    
    def test_config_reset(self):
        """Test resetting configuration to defaults."""
        result = self.run_test_tool(
            "config_reset", 
            {"confirm": True}
        )
        
        logger.info("Successfully reset configuration to defaults")

class BackendManagementTest(BaseMCPToolTest):
    """Test backend management tools."""
    
    def test_list_marketplace_images(self):
        """Test listing marketplace images."""
        result = self.run_test_tool(
            "backend_list_marketplace_images", 
            {},
            ["images"]
        )
        
        self.assertIsInstance(result["images"], list)
        
        logger.info(f"Listed {len(result['images'])} marketplace images")
    
    def test_container_lifecycle(self):
        """Test container lifecycle operations."""
        # Skip all container tests if Docker tools are not available
        if not (self.check_tool_exists("backend_start_container") and 
                self.check_tool_exists("backend_stop_container")):
            self.skipTest("Container lifecycle tools not available")
        
        # Start a container
        container_name = f"test-container-{int(time.time())}"
        container_image = "ipfs/kubo:latest"
        
        start_result = self.run_test_tool(
            "backend_start_container", 
            {
                "name": container_name,
                "image": container_image
            }
        )
        
        # Wait a bit for the container to start
        time.sleep(1)
        
        # List containers
        if self.check_tool_exists("backend_list_containers"):
            list_result = self.run_test_tool(
                "backend_list_containers", 
                {},
                ["containers"]
            )
            
            self.assertIsInstance(list_result["containers"], list)
            
            logger.info(f"Listed {len(list_result['containers'])} containers")
        
        # Stop the container
        stop_result = self.run_test_tool(
            "backend_stop_container", 
            {"name": container_name}
        )
        
        logger.info(f"Successfully managed container lifecycle for {container_name}")
    
    def test_docker_tunnel(self):
        """Test creating a tunnel to a Docker container."""
        # Skip if backend_docker_tunnel tool is not available
        if not self.check_tool_exists("backend_docker_tunnel"):
            self.skipTest("backend_docker_tunnel tool not available")
        
        # Start a container first
        if self.check_tool_exists("backend_start_container"):
            container_name = f"test-tunnel-{int(time.time())}"
            container_image = "ipfs/kubo:latest"
            
            self.run_test_tool(
                "backend_start_container", 
                {
                    "name": container_name,
                    "image": container_image
                }
            )
            
            # Wait a bit for the container to start
            time.sleep(1)
            
            # Create a tunnel
            result = self.run_test_tool(
                "backend_docker_tunnel", 
                {
                    "container_name": container_name,
                    "container_port": 5001,
                    "host_port": 5001
                }
            )
            
            self.assertIn("tunnel_id", result)
            
            logger.info(f"Created tunnel with ID {result['tunnel_id']} for container {container_name}")
            
            # Stop the container at the end
            if self.check_tool_exists("backend_stop_container"):
                self.run_test_tool(
                    "backend_stop_container", 
                    {"name": container_name}
                )

class ResourceManagementTest(BaseMCPToolTest):
    """Test resource pool management tools."""
    
    def test_resource_pool_add_and_remove(self):
        """Test adding and removing resources from the pool."""
        # Add a resource
        resource_id = f"test-resource-{int(time.time())}"
        resource_data = {
            "type": "cpu",
            "cores": 2,
            "memory": "4GB"
        }
        
        add_result = self.run_test_tool(
            "resource_pool_add", 
            {
                "resource_id": resource_id,
                "resource_data": resource_data
            }
        )
        
        self.assertIn("resource_id", add_result)
        self.assertEqual(add_result["resource_id"], resource_id)
        
        # Check status
        status_result = self.run_test_tool(
            "resource_pool_status", 
            {},
            ["resources", "active_count"]
        )
        
        self.assertIsInstance(status_result["resources"], list)
        
        # Remove the resource
        remove_result = self.run_test_tool(
            "resource_pool_remove", 
            {"resource_id": resource_id}
        )
        
        self.assertIn("removed", remove_result)
        self.assertTrue(remove_result["removed"])
        
        logger.info(f"Successfully added and removed resource {resource_id}")
    
    def test_resource_pool_assign(self):
        """Test assigning a resource to a task."""
        # Add a resource
        resource_id = f"test-resource-{int(time.time())}"
        resource_data = {
            "type": "cpu",
            "cores": 2,
            "memory": "4GB"
        }
        
        self.run_test_tool(
            "resource_pool_add", 
            {
                "resource_id": resource_id,
                "resource_data": resource_data
            }
        )
        
        # Start a task
        if self.check_tool_exists("start_task"):
            task_result = self.run_test_tool(
                "start_task", 
                {
                    "task_type": "batch_processing",
                    "params": {"batch_size": 16}
                }
            )
            
            task_id = task_result["task_id"]
            
            # Assign the resource to the task
            assign_result = self.run_test_tool(
                "resource_pool_assign", 
                {
                    "resource_id": resource_id,
                    "task_id": task_id
                }
            )
            
            self.assertIn("assigned", assign_result)
            self.assertTrue(assign_result["assigned"])
            
            logger.info(f"Successfully assigned resource {resource_id} to task {task_id}")
            
            # Clean up
            if self.check_tool_exists("resource_pool_remove"):
                self.run_test_tool(
                    "resource_pool_remove", 
                    {"resource_id": resource_id}
                )
        else:
            self.skipTest("start_task tool not available")

def run_test_and_report(server_url: str = None, report_path: str = None) -> Tuple[int, Dict[str, Any]]:
    """
    Run tests and generate a comprehensive report.
    
    Args:
        server_url: Optional URL to the MCP server
        report_path: Optional path to save the report
        
    Returns:
        Tuple of (exit code, report dictionary)
    """
    # Override the server URL if provided
    if server_url:
        BaseMCPToolTest.server_url = server_url
    
    # Create a simple test loader
    loader = unittest.TestLoader()
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases to the suite
    suite.addTest(loader.loadTestsFromTestCase(MCPServerConnectionTest))
    suite.addTest(loader.loadTestsFromTestCase(IPFSOperationsTest))
    suite.addTest(loader.loadTestsFromTestCase(HardwareDetectionTest))
    suite.addTest(loader.loadTestsFromTestCase(ModelOperationsTest))
    suite.addTest(loader.loadTestsFromTestCase(APIMultiplexingTest))
    suite.addTest(loader.loadTestsFromTestCase(TaskManagementTest))
    suite.addTest(loader.loadTestsFromTestCase(ThroughputOptimizationTest))
    suite.addTest(loader.loadTestsFromTestCase(ConfigurationTest))
    suite.addTest(loader.loadTestsFromTestCase(BackendManagementTest))
    suite.addTest(loader.loadTestsFromTestCase(ResourceManagementTest))
    suite.addTest(loader.loadTestsFromTestCase(MockMCPToolTest))
    
    # Create a result collector
    result = unittest.TestResult()
    
    # Run the tests
    start_time = time.time()
    suite.run(result)
    end_time = time.time()
    
    # Generate the report
    report = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": result.wasSuccessful(),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "tests_run": result.testsRun,
        "total_time": end_time - start_time,
        "details": []
    }
    
    # Collect failure details
    for test, traceback in result.failures:
        report["details"].append({
            "test": str(test),
            "status": "FAIL",
            "traceback": traceback
        })
    
    # Collect error details
    for test, traceback in result.errors:
        report["details"].append({
            "test": str(test),
            "status": "ERROR",
            "traceback": traceback
        })
    
    # Add server and tool statistics if available
    try:
        if hasattr(BaseMCPToolTest, 'client') and BaseMCPToolTest.client:
            client = BaseMCPToolTest.client
            report["server_url"] = client.server_url
            report["server_available"] = client.server_available
            
            if client.server_available:
                # Get tools list
                report["available_tools"] = client.tools or []
                
                # Get missing tools
                report["missing_tools"] = [
                    tool for tool in ALL_EXPECTED_TOOLS 
                    if tool not in (client.tools or [])
                ]
                
                report["coverage"] = {
                    "total_expected_tools": len(ALL_EXPECTED_TOOLS),
                    "available_tools": len(client.tools or []),
                    "coverage_percentage": round(
                        (len(client.tools or []) / len(ALL_EXPECTED_TOOLS)) * 100, 2
                    ) if ALL_EXPECTED_TOOLS else 0,
                    "by_category": {}
                }
                
                # Calculate coverage by category
                for category, tools in EXPECTED_TOOLS.items():
                    available_count = sum(1 for tool in tools if tool in (client.tools or []))
                    report["coverage"]["by_category"][category] = {
                        "expected": len(tools),
                        "available": available_count,
                        "percentage": round((available_count / len(tools)) * 100, 2) if tools else 0
                    }
    except Exception as e:
        report["error"] = str(e)
    
    # Save the report if a path is provided
    if report_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    # Print a summary to the console
    print("\n== MCP Tool Test Summary ==")
    print(f"Date: {report['date']}")
    print(f"Tests run: {report['tests_run']}")
    print(f"Failures: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Skipped: {report['skipped']}")
    print(f"Total time: {report['total_time']:.2f} seconds")
    
    if report.get("available_tools"):
        print(f"\nAvailable tools: {len(report['available_tools'])}/{len(ALL_EXPECTED_TOOLS)} ({report['coverage']['coverage_percentage']}%)")
        
        # Print coverage by category
        print("\nCoverage by category:")
        for category, stats in report.get("coverage", {}).get("by_category", {}).items():
            print(f"  {category}: {stats['available']}/{stats['expected']} ({stats['percentage']}%)")
    
    return 0 if result.wasSuccessful() else 1, report

def main():
    """Run the tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MCP tool tests")
    parser.add_argument("--server", help="URL of the MCP server", default=None)
    parser.add_argument("--report", help="Path to save the test report", default=None)
    parser.add_argument("--list-expected", action="store_true", help="List all expected tools")
    args = parser.parse_args()
    
    if args.list_expected:
        print("Expected MCP tools by category:")
        for category, tools in EXPECTED_TOOLS.items():
            print(f"\n{category}:")
            for tool in tools:
                print(f"  - {tool}")
        return 0
    
    # Generate a default report path if not provided
    if args.report is None:
        timestamp = int(time.time())
        args.report = f"test_results/mcp_tool_test_report_{timestamp}.json"
    
    return run_test_and_report(args.server, args.report)[0]

if __name__ == "__main__":
    sys.exit(main())
