#!/usr/bin/env python3
"""
IPFS Accelerate MCP Server Test Suite

This script provides comprehensive testing of the IPFS Accelerate MCP server,
including functionality tests, connection tests, and tool registration tests.
"""

import os
import sys
import json
import time
import logging
import requests
import subprocess
import unittest
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("mcp_test_suite.log", mode='w')
    ]
)
logger = logging.getLogger("mcp_test_suite")

# Server settings
SERVER_HOST = "localhost"
SERVER_PORT = 8002
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
MCP_BASE_URL = f"{SERVER_URL}/mcp"

class MCPServerTestSuite(unittest.TestCase):
    """
    Test suite for the IPFS Accelerate MCP server.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up the test suite by starting the server if needed"""
        cls.server_process = None
        
        # Check if server is already running
        if not cls._is_server_running():
            logger.info("Starting MCP server for tests...")
            cls.server_process = cls._start_server()
            
            # Wait for the server to start
            for i in range(10):
                if cls._is_server_running():
                    logger.info("MCP server started successfully")
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Failed to start MCP server")
    
    @classmethod
    def tearDownClass(cls):
        """Stop the server if we started it"""
        if cls.server_process:
            logger.info("Stopping MCP server...")
            cls.server_process.terminate()
            cls.server_process.wait(timeout=5)
    
    @classmethod
    def _is_server_running(cls) -> bool:
        """Check if the server is running"""
        try:
            response = requests.get(f"{MCP_BASE_URL}/manifest", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    @classmethod
    def _start_server(cls) -> subprocess.Popen:
        """Start the server in a subprocess"""
        cmd = [
            sys.executable,
            "final_mcp_server.py",
            "--host", "0.0.0.0",
            "--port", str(SERVER_PORT),
            "--debug"
        ]
        
        # Create log file for debugging
        log_file = open("mcp_test_suite.log", "w")
        
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        return process
    
    def test_manifest(self):
        """Test that the server manifest is available and well-formed"""
        response = requests.get(f"{MCP_BASE_URL}/manifest")
        self.assertEqual(response.status_code, 200)
        
        manifest = response.json()
        self.assertIn("server_name", manifest)
        self.assertIn("version", manifest)
        self.assertIn("mcp_version", manifest)
        self.assertIn("tools", manifest)
        self.assertIn("resources", manifest)
        
        logger.info(f"Server name: {manifest['server_name']}")
        logger.info(f"Server version: {manifest['version']}")
        logger.info(f"MCP version: {manifest['mcp_version']}")
    
    def test_tools_registered(self):
        """Test that all expected tools are registered"""
        response = requests.get(f"{MCP_BASE_URL}/manifest")
        self.assertEqual(response.status_code, 200)
        
        manifest = response.json()
        tools = manifest["tools"]
        
        # List of expected tools
        expected_tools = [
            "ipfs_add_file",
            "ipfs_cat",
            "ipfs_files_write",
            "ipfs_files_read",
            "health_check",
            "get_hardware_info"
        ]
        
        for tool in expected_tools:
            self.assertIn(tool, tools)
            logger.info(f"Found tool: {tool}")
    
    def test_get_hardware_info(self):
        """Test the get_hardware_info tool specifically"""
        response = requests.post(f"{MCP_BASE_URL}/tools/get_hardware_info")
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertIn("system", result)
        self.assertIn("accelerators", result)
        
        system = result["system"]
        self.assertIn("os", system)
        self.assertIn("python_version", system)
        
        accelerators = result["accelerators"]
        self.assertIn("cpu", accelerators)
    
    def test_health_check(self):
        """Test the health_check tool"""
        response = requests.post(f"{MCP_BASE_URL}/tools/health_check")
        self.assertEqual(response.status_code, 200)
        
        result = response.json()
        self.assertIn("status", result)
        self.assertEqual(result["status"], "healthy")
    
    def test_ipfs_add_file(self):
        """Test the ipfs_add_file tool"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("Test content for IPFS")
        
        try:
            # Call the tool with the temporary file path
            response = requests.post(
                f"{MCP_BASE_URL}/tools/ipfs_add_file",
                json={"path": temp_file.name}
            )
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertIn("cid", result)
            self.assertIn("success", result)
            self.assertTrue(result["success"])
            
            # Also test ipfs_cat with the returned CID
            cid = result["cid"]
            response = requests.post(
                f"{MCP_BASE_URL}/tools/ipfs_cat",
                json={"cid": cid}
            )
            self.assertEqual(response.status_code, 200)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file.name)

def generate_report(test_result):
    """
    Generate a comprehensive report from the test results
    """
    print("\n" + "="*80)
    print("IPFS Accelerate MCP Test Results")
    print("="*80)
    
    total = test_result.testsRun
    failures = len(test_result.failures)
    errors = len(test_result.errors)
    skipped = len(test_result.skipped)
    passed = total - failures - errors - skipped
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    if failures > 0 or errors > 0:
        print("\nFailures and Errors:")
        for test, traceback in test_result.failures:
            print(f"\n--- Failure in {test} ---")
            print(traceback)
        
        for test, traceback in test_result.errors:
            print(f"\n--- Error in {test} ---")
            print(traceback)
    
    print("\nRecommendations:")
    if passed == total:
        print("✓ All tests passed! The MCP server is working correctly.")
    else:
        if failures > 0:
            print("✗ Some tests failed. Please check the logs for details.")
        if errors > 0:
            print("✗ Some tests had errors. Please check the server implementation.")
        
        if any(t[0]._testMethodName == 'test_manifest' for t in test_result.failures + test_result.errors):
            print("  - The server manifest is not available or malformed.")
            print("  - Check that the server is binding to the correct host and port.")
        
        if any(t[0]._testMethodName == 'test_tools_registered' for t in test_result.failures + test_result.errors):
            print("  - Some expected tools are not registered.")
            print("  - Check the tool registration in fixed_standards_mcp_server.py.")
        
        if any(t[0]._testMethodName == 'test_get_hardware_info' for t in test_result.failures + test_result.errors):
            print("  - The get_hardware_info tool is not working correctly.")
            print("  - Check the implementation of this tool in fixed_standards_mcp_server.py.")
    
    print("="*80)


def main():
    """
    Main entry point.
    """
    print("Starting IPFS Accelerate MCP Server Test Suite...")
    
    # Run the tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(MCPServerTestSuite)
    
    # Use a custom TestResult to capture all results
    result = unittest.TestResult()
    suite.run(result)
    
    # Generate and print the report
    generate_report(result)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(main())
