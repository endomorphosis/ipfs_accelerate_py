#!/usr/bin/env python
"""
IPFS Accelerate MCP Integration Test

This script tests the integration between the IPFS Accelerate package and the MCP server.
It verifies that the server can be started, tools are registered correctly, and API endpoints
function as expected.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import requests
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_integration_test")


class MCPIntegrationTester:
    """Tests the integration between IPFS Accelerate and MCP server."""
    
    def __init__(self, server_port: int = 8000, server_host: str = "localhost", mount_path: str = "/mcp"):
        """Initialize the tester.
        
        Args:
            server_port: Port to run the server on
            server_host: Host to bind the server to
            mount_path: Path to mount the server at
        """
        self.server_port = server_port
        self.server_host = server_host
        self.mount_path = mount_path
        self.base_url = f"http://{server_host}:{server_port}"
        self.server_process = None
        
    def start_server(self, debug: bool = False) -> bool:
        """Start the MCP server.
        
        Args:
            debug: Enable debug mode
            
        Returns:
            True if server started successfully, False otherwise
        """
        # Construct command to run the server
        cmd = [
            sys.executable, "run_ipfs_mcp.py",
            "--port", str(self.server_port),
            "--host", self.server_host
        ]
        
        if debug:
            cmd.append("--debug")
            
        logger.info(f"Starting MCP server with command: {' '.join(cmd)}")
        
        try:
            # Start the server as a subprocess
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            for _ in range(10):
                time.sleep(1)
                try:
                    response = requests.get(f"{self.base_url}/docs")
                    if response.status_code == 200:
                        logger.info("MCP server started successfully")
                        return True
                except requests.exceptions.ConnectionError:
                    pass
                    
            logger.error("Failed to start MCP server")
            return False
            
        except Exception as e:
            logger.error(f"Error starting MCP server: {str(e)}")
            return False
            
    def stop_server(self) -> None:
        """Stop the MCP server."""
        if self.server_process:
            logger.info("Stopping MCP server")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            
    def test_documentation_endpoint(self) -> bool:
        """Test the documentation endpoint.
        
        Returns:
            True if the endpoint is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/docs")
            if response.status_code == 200:
                logger.info("Documentation endpoint is accessible")
                return True
            else:
                logger.error(f"Documentation endpoint returned status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error accessing documentation endpoint: {str(e)}")
            return False
            
    def test_tool_endpoint(self, tool_name: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Test a tool endpoint.
        
        Args:
            tool_name: Name of the tool to test
            params: Parameters to pass to the tool
            
        Returns:
            Tool response as a dictionary, or None if the request failed
        """
        if params is None:
            params = {}
            
        try:
            url = f"{self.base_url}{self.mount_path}/tool/{tool_name}"
            logger.info(f"Testing tool endpoint: {url}")
            response = requests.post(url, json=params)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Tool endpoint returned successfully: {tool_name}")
                return result
            else:
                logger.error(f"Tool endpoint {tool_name} returned status code {response.status_code}")
                try:
                    error = response.json()
                    logger.error(f"Error details: {error}")
                except json.JSONDecodeError:
                    logger.error(f"Response content: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error calling tool endpoint {tool_name}: {str(e)}")
            return None
            
    def test_resource_endpoint(self, resource_uri: str) -> Optional[Dict[str, Any]]:
        """Test a resource endpoint.
        
        Args:
            resource_uri: URI of the resource to test
            
        Returns:
            Resource data as a dictionary, or None if the request failed
        """
        try:
            url = f"{self.base_url}{self.mount_path}/resource/{resource_uri}"
            logger.info(f"Testing resource endpoint: {url}")
            response = requests.get(url)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Resource endpoint returned successfully: {resource_uri}")
                return result
            else:
                logger.error(f"Resource endpoint {resource_uri} returned status code {response.status_code}")
                try:
                    error = response.json()
                    logger.error(f"Error details: {error}")
                except json.JSONDecodeError:
                    logger.error(f"Response content: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error calling resource endpoint {resource_uri}: {str(e)}")
            return None
            
    def run_all_tests(self) -> bool:
        """Run all integration tests.
        
        Returns:
            True if all tests passed, False otherwise
        """
        tests_passed = 0
        tests_failed = 0
        
        # Test documentation endpoint
        if self.test_documentation_endpoint():
            tests_passed += 1
        else:
            tests_failed += 1
            
        # Test hardware info tool
        hardware_info = self.test_tool_endpoint("get_hardware_info")
        if hardware_info is not None:
            tests_passed += 1
            logger.info(f"Hardware info: {json.dumps(hardware_info, indent=2)}")
        else:
            tests_failed += 1
            
        # Test hardware testing tool
        hardware_test = self.test_tool_endpoint("test_hardware", {"accelerator": "cpu", "test_level": "basic"})
        if hardware_test is not None:
            tests_passed += 1
            logger.info(f"Hardware test: {json.dumps(hardware_test, indent=2)}")
        else:
            tests_failed += 1
            
        # Test hardware recommendation tool
        hardware_rec = self.test_tool_endpoint("recommend_hardware", {"model_name": "llama-7b"})
        if hardware_rec is not None:
            tests_passed += 1
            logger.info(f"Hardware recommendation: {json.dumps(hardware_rec, indent=2)}")
        else:
            tests_failed += 1
            
        # Test supported models resource
        models = self.test_resource_endpoint("ipfs_accelerate/supported_models")
        if models is not None:
            tests_passed += 1
            logger.info(f"Supported models: {json.dumps(models, indent=2)}")
        else:
            tests_failed += 1
            
        # Print test results
        logger.info(f"Integration tests completed: {tests_passed} passed, {tests_failed} failed")
        
        return tests_failed == 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Integration Test")
    parser.add_argument("--port", type=int, default=8765, help="Port to run the server on")
    parser.add_argument("--host", default="localhost", help="Host to bind the server to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create tester
    tester = MCPIntegrationTester(args.port, args.host)
    
    # Start the server
    if not tester.start_server(args.debug):
        logger.error("Failed to start MCP server, tests aborted")
        sys.exit(1)
    
    try:
        # Run tests
        success = tester.run_all_tests()
        if success:
            logger.info("All integration tests passed!")
            sys.exit(0)
        else:
            logger.error("Some integration tests failed")
            sys.exit(1)
    finally:
        # Stop the server
        tester.stop_server()


if __name__ == "__main__":
    main()
