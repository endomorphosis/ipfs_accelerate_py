#!/usr/bin/env python3
"""
MCP Server Virtual Filesystem Test Script

This script tests the virtual filesystem functionality of the MCP server,
verifying that all VFS tools are properly registered and working correctly.
"""

import os
import sys
import json
import time
import random
import string
import logging
import argparse
import requests
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MCPVirtualFilesystemTest:
    """Test MCP server virtual filesystem functionality."""
    
    # Expected tools for testing
    REQUIRED_VFS_TOOLS = [
        "ipfs_files_mkdir",
        "ipfs_files_write",
        "ipfs_files_read",
        "ipfs_files_ls",
        "ipfs_files_rm",
        "ipfs_files_cp",
        "ipfs_files_mv",
        "ipfs_files_stat",
        "ipfs_files_flush"
    ]
    
    def __init__(self, host='localhost', port=8002, protocol='http', output_file=None):
        """Initialize the VFS tester."""
        self.host = host
        self.port = port
        self.protocol = protocol
        self.base_url = f"{self.protocol}://{self.host}:{self.port}"
        self.tools_endpoint = f"{self.base_url}/tools"
        self.call_endpoint = f"{self.base_url}/call"
        self.output_file = output_file
        self.timestamp = int(time.time())
        self.test_dir = f"/mcp-vfs-test-{self.timestamp}"
        self.results = {
            "timestamp": time.ctime(),
            "server": {
                "host": host,
                "port": port
            },
            "vfs_tools": {},
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        }
    
    def generate_random_content(self, size=100):
        """Generate random content for testing."""
        chars = string.ascii_letters + string.digits + " \n\t"
        return ''.join(random.choice(chars) for _ in range(size))
    
    def check_tools_availability(self) -> bool:
        """Check if all required VFS tools are available."""
        logger.info("Checking VFS tools availability...")
        
        try:
            # Get available tools
            response = requests.get(self.tools_endpoint, timeout=5)
            if response.status_code != 200:
                logger.error(f"Failed to get tools list: {response.text}")
                self.results["vfs_tools"]["available"] = False
                return False
            
            tools = response.json().get('tools', [])
            missing_tools = []
            
            # Check for required tools
            for tool in self.REQUIRED_VFS_TOOLS:
                if tool not in tools:
                    missing_tools.append(tool)
            
            self.results["vfs_tools"]["available_tools"] = [t for t in self.REQUIRED_VFS_TOOLS if t in tools]
            self.results["vfs_tools"]["missing_tools"] = missing_tools
            
            if missing_tools:
                logger.warning(f"Missing VFS tools: {', '.join(missing_tools)}")
                self.results["vfs_tools"]["available"] = False
                return False
            else:
                logger.info("All required VFS tools are available")
                self.results["vfs_tools"]["available"] = True
                return True
                
        except Exception as e:
            logger.error(f"Error checking tools availability: {e}")
            self.results["vfs_tools"]["available"] = False
            self.results["vfs_tools"]["error"] = str(e)
            return False
    
    def record_test_result(self, name: str, passed: bool, details: Any = None):
        """Record a test result."""
        status = "passed" if passed else "failed"
        
        self.results["tests"][name] = {
            "status": status,
            "details": details
        }
        
        self.results["summary"]["total_tests"] += 1
        if passed:
            self.results["summary"]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1
        
        logger.info(f"Test '{name}': {status.upper()}")
        if details:
            logger.debug(f"Details: {details}")
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the result."""
        payload = {
            "tool_name": tool_name,
            "arguments": arguments
        }
        
        try:
            response = requests.post(self.call_endpoint, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to call {tool_name}: {response.text}")
                return None
            
            result = response.json()
            return result.get("result", {})
        except Exception as e:
            logger.error(f"Exception when calling {tool_name}: {e}")
            return None
    
    def test_mkdir(self) -> bool:
        """Test creating directories in the virtual filesystem."""
        logger.info(f"Testing ipfs_files_mkdir: creating {self.test_dir}")
        
        # Create test directory
        result = self.call_tool("ipfs_files_mkdir", {
            "path": self.test_dir,
            "parents": True
        })
        
        success = result is not None and result.get("success", False)
        self.record_test_result("mkdir", success, result)
        
        if success:
            # Create a subdirectory for additional tests
            subdir = f"{self.test_dir}/subdir"
            subdir_result = self.call_tool("ipfs_files_mkdir", {
                "path": subdir
            })
            subdir_success = subdir_result is not None and subdir_result.get("success", False)
            self.record_test_result("mkdir_subdir", subdir_success, subdir_result)
            return success and subdir_success
        
        return success
    
    def test_write_read(self) -> bool:
        """Test writing to and reading from the virtual filesystem."""
        # Generate random content
        test_content = self.generate_random_content()
        test_file = f"{self.test_dir}/test-file-{self.timestamp}.txt"
        
        logger.info(f"Testing ipfs_files_write: writing to {test_file}")
        
        # Write content to file
        write_result = self.call_tool("ipfs_files_write", {
            "path": test_file,
            "content": test_content,
            "create": True,
            "truncate": True
        })
        
        write_success = write_result is not None and write_result.get("success", False)
        self.record_test_result("write", write_success, write_result)
        
        if not write_success:
            return False
        
        # Read content from file
        logger.info(f"Testing ipfs_files_read: reading from {test_file}")
        read_result = self.call_tool("ipfs_files_read", {
            "path": test_file
        })
        
        read_success = read_result is not None and read_result.get("success", False)
        content_match = read_success and read_result.get("content") == test_content
        
        self.record_test_result("read", read_success, read_result)
        
        if read_success and not content_match:
            logger.warning("Content read does not match content written")
            self.record_test_result("content_verification", False, {
                "written": test_content,
                "read": read_result.get("content")
            })
            return False
        
        return write_success and read_success
    
    def test_ls(self) -> bool:
        """Test listing files in the virtual filesystem."""
        logger.info(f"Testing ipfs_files_ls: listing {self.test_dir}")
        
        # List test directory
        result = self.call_tool("ipfs_files_ls", {
            "path": self.test_dir
        })
        
        success = result is not None and result.get("success", False)
        entries = result.get("entries", []) if success else []
        
        self.record_test_result("ls", success, {
            "result": result,
            "entries_count": len(entries)
        })
        
        return success
    
    def test_stat(self) -> bool:
        """Test getting file/directory stats in the virtual filesystem."""
        logger.info(f"Testing ipfs_files_stat: getting stats for {self.test_dir}")
        
        # Get stats for test directory
        result = self.call_tool("ipfs_files_stat", {
            "path": self.test_dir
        })
        
        success = result is not None and result.get("success", False)
        
        # Check if we got comprehensive stats
        has_comprehensive_stats = False
        if success:
            stats = result.get("stats", {})
            has_comprehensive_stats = all(k in stats for k in ["size", "cumulativeSize", "blocks", "type"])
        
        self.record_test_result("stat", success, {
            "result": result,
            "has_comprehensive_stats": has_comprehensive_stats
        })
        
        return success and has_comprehensive_stats
    
    def test_cp_mv_rm(self) -> bool:
        """Test copy, move, and remove operations in the virtual filesystem."""
        # Create a test file for operations
        test_file = f"{self.test_dir}/cp-mv-test-{self.timestamp}.txt"
        test_content = self.generate_random_content()
        
        # Write test file
        write_result = self.call_tool("ipfs_files_write", {
            "path": test_file,
            "content": test_content,
            "create": True
        })
        
        if not (write_result and write_result.get("success", False)):
            logger.error("Failed to create test file for cp/mv/rm operations")
            self.record_test_result("cp_mv_rm_preparation", False, write_result)
            return False
        
        # Test copy
        copy_dest = f"{self.test_dir}/copied-file-{self.timestamp}.txt"
        logger.info(f"Testing ipfs_files_cp: copying {test_file} to {copy_dest}")
        
        cp_result = self.call_tool("ipfs_files_cp", {
            "source": test_file,
            "dest": copy_dest
        })
        
        cp_success = cp_result is not None and cp_result.get("success", False)
        self.record_test_result("cp", cp_success, cp_result)
        
        # Verify copy worked by reading the file
        if cp_success:
            read_copy_result = self.call_tool("ipfs_files_read", {"path": copy_dest})
            cp_content_match = (read_copy_result and 
                                read_copy_result.get("success", False) and 
                                read_copy_result.get("content") == test_content)
            
            self.record_test_result("cp_verification", cp_content_match, read_copy_result)
        else:
            cp_content_match = False
        
        # Test move
        move_dest = f"{self.test_dir}/moved-file-{self.timestamp}.txt"
        logger.info(f"Testing ipfs_files_mv: moving {test_file} to {move_dest}")
        
        mv_result = self.call_tool("ipfs_files_mv", {
            "source": test_file,
            "dest": move_dest
        })
        
        mv_success = mv_result is not None and mv_result.get("success", False)
        self.record_test_result("mv", mv_success, mv_result)
        
        # Verify move worked by checking original is gone and destination exists
        if mv_success:
            read_src_result = self.call_tool("ipfs_files_read", {"path": test_file})
            src_gone = read_src_result is None or not read_src_result.get("success", False)
            
            read_mv_result = self.call_tool("ipfs_files_read", {"path": move_dest})
            mv_content_match = (read_mv_result and 
                               read_mv_result.get("success", False) and 
                               read_mv_result.get("content") == test_content)
            
            self.record_test_result("mv_verification", src_gone and mv_content_match, {
                "source_gone": src_gone,
                "dest_content_match": mv_content_match
            })
        else:
            mv_content_match = False
        
        # Test remove
        logger.info(f"Testing ipfs_files_rm: removing {move_dest}")
        
        rm_result = self.call_tool("ipfs_files_rm", {
            "path": move_dest
        })
        
        rm_success = rm_result is not None and rm_result.get("success", False)
        self.record_test_result("rm", rm_success, rm_result)
        
        # Verify remove worked
        if rm_success:
            read_after_rm = self.call_tool("ipfs_files_read", {"path": move_dest})
            rm_verification = read_after_rm is None or not read_after_rm.get("success", False)
            self.record_test_result("rm_verification", rm_verification, read_after_rm)
        else:
            rm_verification = False
        
        # Also remove the copied file
        if cp_success:
            self.call_tool("ipfs_files_rm", {"path": copy_dest})
        
        return cp_success and cp_content_match and mv_success and mv_content_match and rm_success and rm_verification
    
    def test_flush(self) -> bool:
        """Test flushing changes to IPFS."""
        logger.info(f"Testing ipfs_files_flush: flushing {self.test_dir}")
        
        # Flush the test directory to get its CID
        result = self.call_tool("ipfs_files_flush", {
            "path": self.test_dir
        })
        
        success = result is not None and result.get("success", False)
        cid = result.get("cid") if success else None
        
        self.record_test_result("flush", success, {
            "result": result,
            "cid": cid
        })
        
        return success and cid is not None
    
    def cleanup(self):
        """Clean up test files and directories."""
        logger.info(f"Cleaning up test directory: {self.test_dir}")
        
        try:
            # Recursively remove the test directory
            result = self.call_tool("ipfs_files_rm", {
                "path": self.test_dir,
                "recursive": True
            })
            
            if result and result.get("success", False):
                logger.info("Cleanup successful")
                return True
            else:
                logger.warning(f"Cleanup failed: {result}")
                return False
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all VFS tests in sequence."""
        logger.info("Starting MCP virtual filesystem tests...")
        
        # First check if all required tools are available
        if not self.check_tools_availability():
            logger.error("Required VFS tools are not available. Skipping tests.")
            return False
        
        # Run tests in a logical sequence
        mkdir_ok = self.test_mkdir()
        if not mkdir_ok:
            logger.error("Failed to create directories. Cannot continue with further tests.")
            return False
        
        write_read_ok = self.test_write_read()
        ls_ok = self.test_ls()
        stat_ok = self.test_stat()
        cp_mv_rm_ok = self.test_cp_mv_rm()
        flush_ok = self.test_flush()
        
        # Clean up
        self.cleanup()
        
        # Save results to output file if specified
        if self.output_file:
            try:
                with open(self.output_file, 'w') as f:
                    json.dump(self.results, f, indent=2)
                logger.info(f"Results saved to {self.output_file}")
            except Exception as e:
                logger.error(f"Failed to save results to {self.output_file}: {e}")
        
        # Print summary
        logger.info("=== Test Summary ===")
        logger.info(f"Total tests: {self.results['summary']['total_tests']}")
        logger.info(f"Passed: {self.results['summary']['passed']}")
        logger.info(f"Failed: {self.results['summary']['failed']}")
        logger.info(f"Skipped: {self.results['summary']['skipped']}")
        
        # Return success if all tests passed
        all_passed = (self.results['summary']['failed'] == 0 and 
                      self.results['summary']['passed'] == self.results['summary']['total_tests'])
        
        if all_passed:
            logger.info("All VFS tests passed successfully!")
        else:
            logger.warning("Some VFS tests failed. Check the detailed results.")
        
        return all_passed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test MCP server virtual filesystem functionality")
    parser.add_argument("--host", default="localhost", help="MCP server host")
    parser.add_argument("--port", type=int, default=8002, help="MCP server port")
    parser.add_argument("--protocol", default="http", help="Protocol to use (http/https)")
    parser.add_argument("--output", default=None, help="Output file for JSON results")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    tester = MCPVirtualFilesystemTest(
        host=args.host,
        port=args.port,
        protocol=args.protocol,
        output_file=args.output
    )
    
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
