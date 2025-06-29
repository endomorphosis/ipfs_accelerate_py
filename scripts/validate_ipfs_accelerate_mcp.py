#!/usr/bin/env python3
"""
IPFS Accelerate MCP Server Validation Script

This script performs comprehensive validation of the IPFS Accelerate MCP server,
testing all exposed tools and functionality.
"""

import requests
import json
import sys
import os
import tempfile
import time
import subprocess
import platform
import argparse
from typing import Dict, Any, List, Optional

def print_header(message: str, width: int = 80) -> None:
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(message.center(width))
    print("=" * width + "\n")

def print_section(message: str, width: int = 80) -> None:
    """Print a formatted section header."""
    print("\n" + "-" * width)
    print(message)
    print("-" * width + "\n")

def check_server_running(port: int) -> bool:
    """Check if the MCP server is running on the specified port."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def get_manifest(base_url: str) -> Dict[str, Any]:
    """Get the MCP server manifest."""
    try:
        response = requests.get(f"{base_url}/mcp/manifest", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"Error getting manifest: {e}")
        return {}

def test_health_check(base_url: str) -> bool:
    """Test the server health check endpoint."""
    print_section("Testing Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("Health check result:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Failed health check: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error checking health: {e}")
        return False

def test_sse_endpoint(base_url: str) -> bool:
    """Test the SSE endpoint for Claude integration."""
    print_section("Testing SSE Endpoint")
    # Check both potential SSE endpoints
    endpoints = [f"{base_url}/sse", f"{base_url}/mcp/sse"]
    
    working_endpoints = []
    for endpoint in endpoints:
        try:
            # Use curl to check the SSE endpoint since requests doesn't handle SSE well
            # Just check if it's available and returns a 200 status
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", endpoint],
                capture_output=True,
                text=True,
                timeout=5
            )
            status_code = int(result.stdout)
            print(f"SSE endpoint {endpoint}: {status_code}")
            if status_code == 200:
                working_endpoints.append(endpoint)
        except Exception as e:
            print(f"Error checking SSE endpoint {endpoint}: {e}")
    
    if working_endpoints:
        print(f"\nWorking SSE endpoints: {', '.join(working_endpoints)}")
        return True
    else:
        print("No working SSE endpoints found")
        return False

def test_get_hardware_info(base_url: str) -> bool:
    """Test the get_hardware_info tool."""
    print_section("Testing get_hardware_info Tool")
    try:
        response = requests.post(
            f"{base_url}/mcp/tool/get_hardware_info",
            json={},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print("Hardware info:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Failed to call get_hardware_info: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error calling get_hardware_info: {e}")
        return False

def test_ipfs_tools(base_url: str, manifest: Dict[str, Any]) -> bool:
    """Test the IPFS-related tools."""
    print_section("Testing IPFS Tools")
    
    tools = manifest.get("tools", {})
    
    # Track test results
    results = {
        "ipfs_add_file": False,
        "ipfs_cat": False,
        "ipfs_files_write": False,
        "ipfs_files_read": False,
        "ipfs_pin_add": False,
        "ipfs_pin_rm": False
    }
    
    # Test ipfs_add_file if available
    cid = None
    if "ipfs_add_file" in tools:
        print("Testing ipfs_add_file tool...")
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp:
                tmp.write("Test content for IPFS")
                tmp_path = tmp.name
            
            response = requests.post(
                f"{base_url}/mcp/tool/ipfs_add_file",
                json={"path": tmp_path},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print("ipfs_add_file result:")
                print(json.dumps(result, indent=2))
                
                # If we got a CID, store it for later tests
                if "result" in result and "cid" in result["result"]:
                    cid = result["result"]["cid"]
                    results["ipfs_add_file"] = True
                    print("✅ ipfs_add_file test successful!")
                else:
                    print("❌ ipfs_add_file test failed: No CID returned")
            else:
                print(f"❌ Failed to call ipfs_add_file: {response.status_code}")
                print(response.text)
            
            # Clean up the temporary file
            os.unlink(tmp_path)
        except Exception as e:
            print(f"Error testing ipfs_add_file: {e}")
    else:
        print("ipfs_add_file tool not available")
    
    print()
    
    # Test ipfs_cat if available and we have a CID
    if "ipfs_cat" in tools and cid:
        print("Testing ipfs_cat tool...")
        try:
            response = requests.post(
                f"{base_url}/mcp/tool/ipfs_cat",
                json={"cid": cid},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print("ipfs_cat result:")
                print(json.dumps(result, indent=2))
                
                if "result" in result and "content" in result["result"]:
                    if result["result"]["content"] == "Test content for IPFS":
                        results["ipfs_cat"] = True
                        print("✅ ipfs_cat test successful!")
                    else:
                        print("❌ ipfs_cat test failed: Content mismatch")
                else:
                    print("❌ ipfs_cat test failed: No content returned")
            else:
                print(f"❌ Failed to call ipfs_cat: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error testing ipfs_cat: {e}")
    else:
        print("ipfs_cat tool not available or no CID to test with")
    
    print()
    
    # Test ipfs_files_write if available
    mfs_path = "/validate-mcp-test.txt"
    if "ipfs_files_write" in tools:
        print("Testing ipfs_files_write tool...")
        try:
            response = requests.post(
                f"{base_url}/mcp/tool/ipfs_files_write",
                json={"path": mfs_path, "content": "Test MFS content"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print("ipfs_files_write result:")
                print(json.dumps(result, indent=2))
                
                if "result" in result and "success" in result["result"] and result["result"]["success"]:
                    results["ipfs_files_write"] = True
                    print("✅ ipfs_files_write test successful!")
                else:
                    print("❌ ipfs_files_write test failed")
            else:
                print(f"❌ Failed to call ipfs_files_write: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error testing ipfs_files_write: {e}")
    else:
        print("ipfs_files_write tool not available")
    
    print()
    
    # Test ipfs_files_read if available
    if "ipfs_files_read" in tools and results["ipfs_files_write"]:
        print("Testing ipfs_files_read tool...")
        try:
            response = requests.post(
                f"{base_url}/mcp/tool/ipfs_files_read",
                json={"path": mfs_path},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print("ipfs_files_read result:")
                print(json.dumps(result, indent=2))
                
                if "result" in result and "content" in result["result"]:
                    if result["result"]["content"] == "Test MFS content":
                        results["ipfs_files_read"] = True
                        print("✅ ipfs_files_read test successful!")
                    else:
                        print("❌ ipfs_files_read test failed: Content mismatch")
                else:
                    print("❌ ipfs_files_read test failed: No content returned")
            else:
                print(f"❌ Failed to call ipfs_files_read: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error testing ipfs_files_read: {e}")
    else:
        print("ipfs_files_read tool not available or ipfs_files_write failed")
    
    print()
    
    # Test ipfs_pin_add if available and we have a CID
    if "ipfs_pin_add" in tools and cid:
        print("Testing ipfs_pin_add tool...")
        try:
            response = requests.post(
                f"{base_url}/mcp/tool/ipfs_pin_add",
                json={"cid": cid},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print("ipfs_pin_add result:")
                print(json.dumps(result, indent=2))
                
                if "result" in result and "success" in result["result"] and result["result"]["success"]:
                    results["ipfs_pin_add"] = True
                    print("✅ ipfs_pin_add test successful!")
                else:
                    print("❌ ipfs_pin_add test failed")
            else:
                print(f"❌ Failed to call ipfs_pin_add: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error testing ipfs_pin_add: {e}")
    else:
        print("ipfs_pin_add tool not available or no CID to test with")
    
    print()
    
    # Test ipfs_pin_rm if available, we have a CID, and pin_add was successful
    if "ipfs_pin_rm" in tools and cid and results["ipfs_pin_add"]:
        print("Testing ipfs_pin_rm tool...")
        try:
            response = requests.post(
                f"{base_url}/mcp/tool/ipfs_pin_rm",
                json={"cid": cid},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print("ipfs_pin_rm result:")
                print(json.dumps(result, indent=2))
                
                if "result" in result and "success" in result["result"] and result["result"]["success"]:
                    results["ipfs_pin_rm"] = True
                    print("✅ ipfs_pin_rm test successful!")
                else:
                    print("❌ ipfs_pin_rm test failed")
            else:
                print(f"❌ Failed to call ipfs_pin_rm: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error testing ipfs_pin_rm: {e}")
    else:
        print("ipfs_pin_rm tool not available, no CID to test with, or ipfs_pin_add failed")
    
    print("\nIPFS Tools Test Summary:")
    for tool, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL" if tool in tools else "⚠️ SKIPPED"
        print(f"{tool:<20}: {status}")
    
    return any(results.values())

def validate_server(port: int) -> None:
    """Run validation tests on the MCP server."""
    base_url = f"http://localhost:{port}"
    
    print_header(f"IPFS Accelerate MCP Server Validation - {base_url}")
    
    # Check if server is running
    if not check_server_running(port):
        print(f"❌ MCP server is not running on port {port}")
        print(f"Please start the server first with: bash restart_mcp_server.sh --port {port}")
        sys.exit(1)
    
    print(f"✅ MCP server is running on port {port}")
    
    # Get the manifest
    manifest = get_manifest(base_url)
    if not manifest:
        print("❌ Failed to get server manifest")
        sys.exit(1)
    
    print("Server manifest:")
    print(json.dumps(manifest, indent=2))
    
    print("\nAvailable tools:")
    for tool_name in manifest.get("tools", {}):
        print(f"- {tool_name}")
    
    # Run the tests
    health_result = test_health_check(base_url)
    sse_result = test_sse_endpoint(base_url)
    hardware_result = test_get_hardware_info(base_url)
    ipfs_result = test_ipfs_tools(base_url, manifest)
    
    # Print a summary of the results
    print_header("VALIDATION SUMMARY")
    print(f"Health Check: {'✅ PASS' if health_result else '❌ FAIL'}")
    print(f"SSE Endpoint: {'✅ PASS' if sse_result else '❌ FAIL'}")
    print(f"Hardware Info: {'✅ PASS' if hardware_result else '❌ FAIL'}")
    print(f"IPFS Tools: {'✅ PASS' if ipfs_result else '❌ FAIL'}")
    
    if health_result and sse_result and hardware_result and ipfs_result:
        print("\n✅ All tests PASSED!")
        print("\nThe IPFS Accelerate MCP server is working correctly and ready for use with Claude.")
    else:
        print("\n❌ Some tests FAILED!")
        print("\nPlease check the logs above for details on the failures.")
    
    # Provide recommendations for Claude integration
    print_header("CLAUDE INTEGRATION RECOMMENDATIONS")
    
    if sse_result:
        print("To use this MCP server with Claude, ensure the following settings are in Claude's MCP configuration:\n")
        
        claude_config = {
            "mcpServers": {
                "ipfs-accelerate-mcp": {
                    "disabled": False,
                    "timeout": 60,
                    "url": f"http://localhost:{port}/sse",
                    "transportType": "sse"
                }
            }
        }
        
        print(json.dumps(claude_config, indent=2))
        
        print("\nThe configuration file is typically located at:")
        if platform.system() == "Windows":
            print(r"%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json")
        else:
            print("~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json")
        
        print("\nAfter updating the configuration, you may need to restart the VSCode Claude extension.")
    else:
        print("❌ SSE endpoint is not working correctly, so Claude integration may not work.")
        print("Please fix the SSE endpoint issues before attempting to use this server with Claude.")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Server Validation")
    parser.add_argument("--port", type=int, default=8002, help="Port the MCP server is running on")
    args = parser.parse_args()
    
    validate_server(args.port)

if __name__ == "__main__":
    main()
