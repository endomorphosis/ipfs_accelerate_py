#!/usr/bin/env python3
"""
MCP Server Integration Test Script

This script tests the MCP server integration with ipfs_accelerate_py,
verifying that all tools are properly registered and functioning.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import requests
from typing import Dict, Any, List, Optional, Tuple
import unittest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MCPServerIntegrationTest:
    """Test MCP server integration with ipfs_accelerate_py."""
    
    def __init__(self, host='localhost', port=8001, protocol='http', output_dir='test_results'):
        """Initialize the integration tester."""
        self.host = host
        self.port = port
        self.protocol = protocol
        self.base_url = f"{self.protocol}://{self.host}:{self.port}"
        self.tools_endpoint = f"{self.base_url}/tools"
        self.call_endpoint = f"{self.base_url}/mcp/tool"
        self.output_dir = output_dir
        self.timestamp = int(time.time())
        self.results = {
            "timestamp": time.ctime(),
            "server": {
                "host": host,
                "port": port
            },
            "components": {},
            "tools": {},
            "integration": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0
            }
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def record_result(self, category: str, test_name: str, passed: Any, details: Any = None):
        """Record a test result."""
        if category not in self.results:
            self.results[category] = {}
        
        self.results[category][test_name] = {
            "status": "passed" if passed else "failed",
            "details": details
        }
        
        self.results["summary"]["total_tests"] += 1
        if passed:
            self.results["summary"]["passed"] += 1
            status = "PASSED"
        else:
            self.results["summary"]["failed"] += 1
            status = "FAILED"
        
        logger.info(f"Test '{category}.{test_name}': {status}")
        if details:
            logger.debug(f"Details: {details}")
    
    def check_server_connectivity(self):
        """Check if the MCP server is running and responsive."""
        logger.info("Checking server connectivity...")
        
        try:
            # Check tools endpoint
            tools_response = requests.get(self.tools_endpoint, timeout=5)
            tools_ok = tools_response.status_code == 200
            
            # Check MCP manifest
            manifest_response = requests.get(f"{self.base_url}/mcp/manifest", timeout=5)
            manifest_ok = manifest_response.status_code == 200
            
            self.record_result("components", "server_connectivity", 
                              tools_ok and manifest_ok,
                              {
                                  "tools": tools_ok,
                                  "manifest": manifest_ok
                              })
            
            if manifest_ok:
                try:
                    manifest = manifest_response.json()
                    self.results["server"]["version"] = manifest.get("version", "unknown")
                    self.results["server"]["name"] = manifest.get("name", "unknown")
                except Exception as e:
                    logger.warning(f"Failed to parse manifest JSON: {e}")
            
            return tools_ok and manifest_ok
            
        except requests.RequestException as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self.record_result("components", "server_connectivity", False, {"error": str(e)})
            return False
    
    def check_tool_registration(self):
        """Check if all expected tools are registered."""
        logger.info("Checking tool registration...")
        
        try:
            # Get available tools
            response = requests.get(self.tools_endpoint, timeout=5)
            if response.status_code != 200:
                self.record_result("components", "tool_registration", False, {"error": response.text})
                return False
            
            tools = response.json().get('tools', [])
            if not tools:
                self.record_result("components", "tool_registration", False, {"error": "No tools found"})
                return False
            
            # Check for specific tool categories
            ipfs_tools = [t for t in tools if t.startswith('ipfs_') and not t.startswith('ipfs_files_')]
            vfs_tools = [t for t in tools if t.startswith('ipfs_files_')]
            hardware_tools = [t for t in tools if t.startswith('get_hardware')]
            model_tools = [t for t in tools if t in ['model_inference', 'list_models']]
            
            self.record_result("tools", "ipfs_tools", len(ipfs_tools) >= 3, {"tools": ipfs_tools})
            self.record_result("tools", "virtual_filesystem", len(vfs_tools) >= 3, {"tools": vfs_tools})
            self.record_result("tools", "hardware_tools", len(hardware_tools) >= 1, {"tools": hardware_tools})
            self.record_result("tools", "model_tools", len(model_tools) >= 1, {"tools": model_tools})
            
            # Store all available tools in results
            self.results["tools"]["available"] = tools
            
            return len(ipfs_tools) >= 3 and len(vfs_tools) >= 3
            
        except requests.RequestException as e:
            logger.error(f"Failed to check tool registration: {e}")
            self.record_result("components", "tool_registration", False, {"error": str(e)})
            return False
    
    def test_hardware_detection(self):
        """Test hardware detection tools."""
        logger.info("Testing hardware detection...")
        
        try:
            # Call ipfs_get_hardware_info tool
            result = self.call_tool("ipfs_get_hardware_info", {})
            
            if not result:
                self.record_result("integration", "hardware_detection", False, {"error": "Failed to get hardware info"})
                return False
            
            # Check if the result has expected structure indicating ipfs_accelerate_py integration
            has_system = "system" in result
            has_accelerators = "accelerators" in result
            
            # Check for detailed system info that would come from ipfs_accelerate_py
            detailed_info = False
            if has_system:
                system_info = result.get("system", {})
                if all(k in system_info for k in ["os", "cpu", "memory_total"]):
                    detailed_info = True
            
            # Check for specific ipfs_accelerate_py markers
            ipfs_accelerate_markers = self.check_ipfs_accelerate_integration()
            
            # Log detailed hardware information for debugging
            logger.info(f"Hardware detection results:")
            logger.info(f"  - System info present: {has_system}")
            logger.info(f"  - Accelerator info present: {has_accelerators}")
            logger.info(f"  - Detailed system info: {detailed_info}")
            logger.info(f"  - IPFS Accelerate markers: {ipfs_accelerate_markers}")
            
            self.record_result("integration", "hardware_detection", 
                              has_system and has_accelerators and detailed_info,
                              {
                                  "has_system": has_system,
                                  "has_accelerators": has_accelerators,
                                  "detailed_info": detailed_info,
                                  "ipfs_accelerate_markers": ipfs_accelerate_markers,
                                  "result": result
                              })
            
            return has_system and has_accelerators and detailed_info
            
        except Exception as e:
            logger.error(f"Failed to test hardware detection: {e}")
            self.record_result("integration", "hardware_detection", False, {"error": str(e)})
            return False
    
    def test_ipfs_functionality(self):
        """Test basic IPFS functionality."""
        logger.info("Testing IPFS functionality...")
        
        try:
            # Create a test file
            test_file = os.path.join(self.output_dir, f"ipfs_test_{self.timestamp}.txt")
            with open(test_file, "w") as f:
                f.write(f"IPFS test content {time.ctime()}")
            
            # Add the file to IPFS
            add_result = self.call_tool("ipfs_add_file", {"path": test_file})
            if not add_result or not add_result.get("success", False):
                self.record_result("integration", "ipfs_add", False, {"error": "Failed to add file to IPFS"})
                return False
            
            cid = add_result.get("cid")
            if not cid:
                self.record_result("integration", "ipfs_add", False, {"error": "No CID returned from add_file"})
                return False
            
            self.record_result("integration", "ipfs_add", True, {"cid": cid})
            
            # Get the gateway URL
            gateway_result = self.call_tool("ipfs_gateway_url", {"ipfs_hash": cid})
            gateway_success = gateway_result and gateway_result.get("success", False)
            
            self.record_result("integration", "ipfs_gateway", gateway_success, gateway_result)
            
            # Clean up
            os.remove(test_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to test IPFS functionality: {e}")
            self.record_result("integration", "ipfs_functionality", False, {"error": str(e)})
            return False
    
    def test_virtual_filesystem(self):
        """Test IPFS virtual filesystem integration."""
        logger.info("Testing virtual filesystem...")
        
        # Use our dedicated VFS test script if it exists
        vfs_test_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_mcp_virtual_fs.py")
        if os.path.exists(vfs_test_script):
            try:
                output_file = os.path.join(self.output_dir, f"vfs_test_results_{self.timestamp}.json")
                cmd = [
                    sys.executable, 
                    vfs_test_script,
                    "--host", self.host,
                    "--port", str(self.port),
                    "--protocol", self.protocol,
                    "--output", output_file
                ]
                
                logger.info(f"Running VFS test script: {' '.join(cmd)}")
                process = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                # Check if the script ran successfully
                success = process.returncode == 0
                
                # Load detailed results if available
                details = {"stdout": process.stdout, "stderr": process.stderr}
                if os.path.exists(output_file):
                    try:
                        with open(output_file, "r") as f:
                            details["results"] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load VFS test results: {e}")
                
                self.record_result("integration", "virtual_filesystem", success, details)
                return success
                
            except Exception as e:
                logger.error(f"Failed to run VFS test script: {e}")
                self.record_result("integration", "virtual_filesystem", False, {"error": str(e)})
                return False
        else:
            logger.warning(f"VFS test script not found at {vfs_test_script}")
            self.record_result("integration", "virtual_filesystem", False, 
                             {"error": "VFS test script not found"})
            return False
    
    def check_ipfs_accelerate_integration(self) -> Dict[str, bool]:
        """Check for specific markers indicating ipfs_accelerate_py integration."""
        logger.info("Checking for ipfs_accelerate_py integration markers...")
        
        # Initialize result dictionary
        markers = {
            "import_success": False,
            "hardware_detection": False,
            "ipfs_vfs_tools": False,
            "accelerate_version": False
        }
        
        try:
            # Try requesting version info if available
            try:
                response = requests.get(f"{self.base_url}/version", timeout=5)
                if response.status_code == 200:
                    version_info = response.json()
                    # Check for ipfs_accelerate_py version info
                    accelerate_version = version_info.get("ipfs_accelerate_py_version")
                    markers["accelerate_version"] = accelerate_version is not None
                    if markers["accelerate_version"]:
                        logger.info(f"Found ipfs_accelerate_py version: {accelerate_version}")
            except Exception as e:
                logger.warning(f"Failed to get version info: {e}")
            
            # Check for ipfs_vfs tools
            try:
                tools_response = requests.get(self.tools_endpoint, timeout=5)
                if tools_response.status_code == 200:
                    tools = tools_response.json().get('tools', [])
                    vfs_tools = [t for t in tools if t.startswith('ipfs_files_')]
                    markers["ipfs_vfs_tools"] = len(vfs_tools) >= 3
                    if markers["ipfs_vfs_tools"]:
                        logger.info(f"Found {len(vfs_tools)} virtual filesystem tools: {', '.join(vfs_tools[:5])}")
            except Exception as e:
                logger.warning(f"Failed to check for VFS tools: {e}")
            
            # Create a test script to check if ipfs_accelerate_py is importable
            script_path = os.path.join(self.output_dir, f"check_import_{self.timestamp}.py")
            with open(script_path, "w") as f:
                f.write("""
import sys
try:
    import ipfs_accelerate_py
    print("Successfully imported ipfs_accelerate_py")
    sys.exit(0)
except ImportError as e:
    print(f"Failed to import ipfs_accelerate_py: {e}")
    sys.exit(1)
                """)
            
            try:
                # Run the script to check if the import works
                process = subprocess.run([sys.executable, script_path], 
                                        capture_output=True, text=True, timeout=10)
                markers["import_success"] = process.returncode == 0
                if markers["import_success"]:
                    logger.info("ipfs_accelerate_py module is importable")
                else:
                    logger.warning(f"ipfs_accelerate_py import failed: {process.stderr}")
            except Exception as e:
                logger.warning(f"Failed to check for importability: {e}")
            
            # Clean up
            try:
                os.remove(script_path)
            except:
                pass
            
            return markers
        except Exception as e:
            logger.error(f"Error checking ipfs_accelerate_py integration: {e}")
            return markers
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the result."""
        
        try:
            response = requests.post(
                f"{self.base_url}/mcp/tool/{tool_name}",
                json=arguments,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            return {"error": str(e), "success": False}
    
    def run_all_tests(self):
        """Run all integration tests."""
        logger.info("Starting MCP server integration tests...")
        
        # Check server connectivity
        if not self.check_server_connectivity():
            logger.error("Server connectivity test failed, skipping remaining tests")
            return False
        
        # Check tool registration
        if not self.check_tool_registration():
            logger.warning("Tool registration test failed, some tests may be skipped")
        
        # Run functional tests
        self.test_hardware_detection()
        self.test_ipfs_functionality()
        self.test_virtual_filesystem()
        
        # Write results to output file
        output_file = os.path.join(self.output_dir, f"mcp_integration_results_{self.timestamp}.json")
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test results written to: {output_file}")
        
        # Print summary
        logger.info("=== Test Summary ===")
        logger.info(f"Total tests: {self.results['summary']['total_tests']}")
        logger.info(f"Passed: {self.results['summary']['passed']}")
        logger.info(f"Failed: {self.results['summary']['failed']}")
        
        return self.results["summary"]["failed"] == 0

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test MCP server integration with ipfs_accelerate_py")
    parser.add_argument("--host", default="localhost", help="MCP server host")
    parser.add_argument("--port", type=int, default=8000, help="MCP server port")
    parser.add_argument("--protocol", default="http", help="Protocol to use (http/https)")
    parser.add_argument("--output-dir", default="test_results", help="Directory for test output")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    test = MCPServerIntegrationTest(
        host=args.host, 
        port=args.port, 
        protocol=args.protocol,
        output_dir=args.output_dir
    )
    
    success = test.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
