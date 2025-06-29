#!/usr/bin/env python3
"""
MCP Server Feature Test

This script tests all features of the IPFS Accelerate MCP server.
It assumes the server is running on localhost:8000.
"""

import argparse
import json
import os
import random
import requests
import sys
import time
from pathlib import Path

# Test configuration
SERVER_URL = "http://localhost:8000"
SSE_URL = f"{SERVER_URL}/sse"
DIRECT_CALL_URL = f"{SERVER_URL}/call"

# Test files directory
TEST_DIR = Path("./mcp_test_files")

# Colors for console output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def setup_test_environment():
    """Set up the test environment."""
    print(f"{Colors.HEADER}Setting up test environment...{Colors.ENDC}")
    
    # Create test directory if it doesn't exist
    if not TEST_DIR.exists():
        TEST_DIR.mkdir(parents=True)
        print(f"Created test directory: {TEST_DIR}")
    
    # Create test files
    test_files = [
        ("test1.txt", "This is test file 1 content."),
        ("test2.txt", "Another test file with\nmultiple lines\nof content."),
        ("test3.json", json.dumps({"name": "Test JSON", "value": 42, "nested": {"key": "value"}})),
    ]
    
    for filename, content in test_files:
        file_path = TEST_DIR / filename
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Created test file: {file_path}")
    
    print(f"{Colors.GREEN}Test environment setup complete.{Colors.ENDC}\n")

def check_server_running():
    """Check if the MCP server is running."""
    print(f"{Colors.HEADER}Checking if MCP server is running...{Colors.ENDC}")
    try:
        response = requests.get(SERVER_URL, timeout=5)
        if response.status_code == 200:
            server_info = response.json()
            print(f"{Colors.GREEN}Server is running: {server_info}{Colors.ENDC}\n")
            return True
        else:
            print(f"{Colors.FAIL}Server returned non-200 status code: {response.status_code}{Colors.ENDC}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"{Colors.FAIL}Failed to connect to server: {str(e)}{Colors.ENDC}")
        print(f"{Colors.WARNING}Make sure the server is running with: ./run_ipfs_mcp.sh{Colors.ENDC}")
        return False

def test_direct_call(tool_name, arguments):
    """Test a tool using direct API call."""
    url = DIRECT_CALL_URL
    payload = {
        "tool_name": tool_name,
        "arguments": arguments
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("result")
        else:
            print(f"{Colors.FAIL}Error calling {tool_name}: {response.status_code} - {response.text}{Colors.ENDC}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"{Colors.FAIL}Request error: {str(e)}{Colors.ENDC}")
        return None

def run_test(name, tool_name, arguments, expected_success=True):
    """Run a test case and report results."""
    print(f"{Colors.HEADER}Running test: {name}{Colors.ENDC}")
    print(f"Tool: {tool_name}")
    print(f"Arguments: {json.dumps(arguments, indent=2)}")
    
    result = test_direct_call(tool_name, arguments)
    
    if result is None:
        print(f"{Colors.FAIL}Test failed: No result returned{Colors.ENDC}\n")
        return False
    
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Check if the result indicates success or failure as expected
    success = result.get("success", True) if isinstance(result, dict) else True
    
    if success == expected_success:
        print(f"{Colors.GREEN}Test passed: {'Success' if expected_success else 'Expected failure'} as expected{Colors.ENDC}\n")
        return True
    else:
        print(f"{Colors.FAIL}Test failed: Expected {'success' if expected_success else 'failure'} but got {'success' if success else 'failure'}{Colors.ENDC}\n")
        return False

def test_health_check():
    """Test the health_check tool."""
    return run_test(
        "Health Check", 
        "health_check", 
        {}
    )

def test_ipfs_add_file_success():
    """Test adding a file to IPFS (success case)."""
    test_file = str(TEST_DIR / "test1.txt")
    return run_test(
        "Add File (Success)", 
        "ipfs_add_file", 
        {"path": test_file}
    )

def test_ipfs_add_file_failure():
    """Test adding a non-existent file to IPFS (failure case)."""
    return run_test(
        "Add File (Failure)", 
        "ipfs_add_file", 
        {"path": str(TEST_DIR / "nonexistent.txt")},
        expected_success=False
    )

def test_ipfs_files_write():
    """Test writing to IPFS MFS."""
    return run_test(
        "Write to MFS", 
        "ipfs_files_write", 
        {
            "path": "/test/mfs_file.txt",
            "content": "This is MFS test content."
        }
    )

def test_ipfs_files_read_success():
    """Test reading from IPFS MFS (success case)."""
    # First write a file, then read it
    write_result = test_direct_call("ipfs_files_write", {
        "path": "/test/read_test.txt",
        "content": "Content for reading test."
    })
    
    if write_result is None:
        print(f"{Colors.FAIL}Failed to prepare test file for reading{Colors.ENDC}\n")
        return False
    
    return run_test(
        "Read from MFS (Success)", 
        "ipfs_files_read", 
        {"path": "/test/read_test.txt"}
    )

def test_ipfs_files_read_failure():
    """Test reading from IPFS MFS (failure case)."""
    return run_test(
        "Read from MFS (Failure)", 
        "ipfs_files_read", 
        {"path": "/test/nonexistent_file.txt"},
        expected_success=False
    )

def test_ipfs_cat():
    """Test retrieving content from IPFS."""
    # First add a file to get a CID
    test_file = str(TEST_DIR / "test2.txt")
    add_result = test_direct_call("ipfs_add_file", {"path": test_file})
    
    if add_result is None or "cid" not in add_result:
        print(f"{Colors.FAIL}Failed to prepare file for IPFS cat test{Colors.ENDC}\n")
        return False
    
    cid = add_result["cid"]
    
    return run_test(
        "Retrieve Content from IPFS", 
        "ipfs_cat", 
        {"cid": cid}
    )

def cleanup():
    """Clean up test resources."""
    print(f"{Colors.HEADER}Cleaning up test resources...{Colors.ENDC}")
    
    import shutil
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"Removed test directory: {TEST_DIR}")
    
    print(f"{Colors.GREEN}Cleanup complete.{Colors.ENDC}\n")

def run_all_tests():
    """Run all tests and report overall results."""
    print(f"{Colors.BOLD}{Colors.HEADER}===== IPFS Accelerate MCP Server Feature Tests ====={Colors.ENDC}\n")
    
    if not check_server_running():
        return 1
    
    setup_test_environment()
    
    tests = [
        ("Health Check", test_health_check),
        ("Add File (Success)", test_ipfs_add_file_success),
        ("Add File (Failure)", test_ipfs_add_file_failure),
        ("Write to MFS", test_ipfs_files_write),
        ("Read from MFS (Success)", test_ipfs_files_read_success),
        ("Read from MFS (Failure)", test_ipfs_files_read_failure),
        ("Retrieve Content", test_ipfs_cat),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"{Colors.FAIL}Error in test {name}: {str(e)}{Colors.ENDC}\n")
            results.append((name, False))
    
    # Print summary
    print(f"{Colors.BOLD}{Colors.HEADER}===== Test Results Summary ====={Colors.ENDC}")
    passed = 0
    for name, result in results:
        status = f"{Colors.GREEN}PASSED{Colors.ENDC}" if result else f"{Colors.FAIL}FAILED{Colors.ENDC}"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    overall = f"{Colors.GREEN}PASSED{Colors.ENDC}" if passed == len(tests) else f"{Colors.FAIL}FAILED{Colors.ENDC}"
    print(f"\nOverall: {overall} ({passed}/{len(tests)} tests passed)")
    
    cleanup()
    
    return 0 if passed == len(tests) else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MCP server features")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up test files")
    args = parser.parse_args()
    
    if args.no_cleanup:
        # Replace cleanup with a no-op
        def cleanup():
            print(f"{Colors.HEADER}Skipping cleanup as requested{Colors.ENDC}\n")
    
    sys.exit(run_all_tests())
