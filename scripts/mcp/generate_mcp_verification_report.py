#!/usr/bin/env python3
"""
IPFS Accelerate MCP Verification Report Generator

This script generates a detailed verification report in Markdown format
about the current state of the IPFS Accelerate MCP server and its tools.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import requests
from datetime import datetime

# Default values
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8002
DEFAULT_OUTPUT = "ipfs_accelerate_mcp_verification_report.md"

def run_test_and_get_results(host, port, output_file):
    """Run the comprehensive test and return the results."""
    print(f"Running comprehensive MCP server test on {host}:{port}...")
    
    # Execute the test script
    test_script = "test_mcp_server_comprehensive.py"
    if not os.path.exists(test_script):
        print(f"Error: Test script not found: {test_script}")
        return None
    
    try:
        subprocess.run(
            [sys.executable, test_script, "--host", host, "--port", str(port), "--output", output_file],
            check=True
        )
        
        # Read the test results
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        return results
    except Exception as e:
        print(f"Error running test script: {e}")
        return None

def check_hardware_info(host, port):
    """Get hardware info directly from the server."""
    try:
        tool_url = f"http://{host}:{port}/mcp/tool/get_hardware_info"
        response = requests.post(tool_url, json={}, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting hardware info: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to get_hardware_info tool: {e}")
        return None

def check_health(host, port):
    """Get health check directly from the server."""
    try:
        tool_url = f"http://{host}:{port}/mcp/tool/health_check"
        response = requests.post(tool_url, json={}, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting health check: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to health_check tool: {e}")
        return None

def get_server_info(host, port):
    """Get server process information."""
    try:
        # Get server process info
        cmd = f"ps aux | grep -E 'final_mcp_server\\.py.*{port}' | grep -v grep"
        process_info = subprocess.check_output(cmd, shell=True, text=True).strip()
        
        # Get server logs
        cmd = "tail -10 final_mcp_server.log"
        server_logs = subprocess.check_output(cmd, shell=True, text=True).strip()
        
        return {
            "process": process_info,
            "logs": server_logs
        }
    except subprocess.CalledProcessError:
        return {"process": "Not found", "logs": "No logs available"}
    except Exception as e:
        print(f"Error getting server info: {e}")
        return {"process": "Error", "logs": str(e)}

def generate_report(results, hardware_info, health_info, server_info, host, port, output_file):
    """Generate a detailed verification report in Markdown format."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the report
    report = f"""# IPFS Accelerate MCP Verification Report

## Overview

- **Date:** {timestamp}
- **Server:** {host}:{port}
- **Status:** {"Operational" if results and results["server"]["reachable"] else "Not Available"}

## Server Information

- **Host:** {host}
- **Port:** {port}
- **Reachable:** {"Yes" if results and results["server"]["reachable"] else "No"}
"""

    # Add server process info
    if server_info:
        report += f"""
### Server Process
```
{server_info["process"]}
```

### Recent Server Logs
```
{server_info["logs"]}
```
"""

    # Add health check results
    if health_info:
        report += f"""
## Health Status

- **Status:** {health_info.get("status", "N/A")}
- **Version:** {health_info.get("version", "N/A")}
- **Uptime:** {health_info.get("uptime", 0):.2f} seconds
- **IPFS Connected:** {"Yes" if health_info.get("ipfs_connected", False) else "No"}
- **Registered Tools:** {health_info.get("registered_tools_count", 0)}
"""

    # Add tools information
    if results and "tools" in results:
        available_tools = results["tools"].get("available", [])
        required_tools_status = results["tools"].get("required_tools_working", {})
        
        report += f"""
## Tools Status

Available Tools: {len(available_tools)}

| Tool Name | Registered | Functional |
|-----------|-----------|------------|
"""
        
        # List all required tools and their status
        all_tools = set(list(required_tools_status.keys()) + available_tools)
        for tool in sorted(all_tools):
            is_registered = tool in available_tools
            is_functional = required_tools_status.get(tool, False)
            
            report += f"| {tool} | {'✅' if is_registered else '❌'} | {'✅' if is_functional else '❌'} |\n"
    
    # Add hardware info
    if hardware_info:
        system_info = hardware_info.get("system", {})
        accelerators = hardware_info.get("accelerators", {})
        
        report += f"""
## Hardware Information

### System
- **OS:** {system_info.get("os", "N/A")}
- **Distribution:** {system_info.get("distribution", "N/A")}
- **Architecture:** {system_info.get("architecture", "N/A")}
- **Python Version:** {system_info.get("python_version", "N/A")}
- **CPU Count:** {system_info.get("cpu_count", "N/A")}

### Accelerators
"""
        
        for acc_name, acc_info in accelerators.items():
            report += f"- **{acc_name.upper()}:** {'Available' if acc_info.get('available', False) else 'Not Available'}\n"
            
            # Add additional details for available accelerators
            if acc_info.get('available', False):
                if acc_name == "cuda" and "device_names" in acc_info:
                    report += f"  - Devices: {', '.join(acc_info['device_names'])}\n"
    
    # Add verification summary
    if results:
        required_tools_status = results["tools"].get("required_tools_working", {})
        all_tools_working = all(required_tools_status.values())
        get_hardware_info_working = required_tools_status.get("get_hardware_info", False)
        
        report += f"""
## Verification Summary

- **All Required Tools Working:** {'Yes' if all_tools_working else 'No'}
- **get_hardware_info Tool Working:** {'Yes' if get_hardware_info_working else 'No'}
- **IPFS Connection:** {'Yes' if health_info and health_info.get("ipfs_connected", False) else 'No'}
"""
        
        # Add recommendations if there are issues
        if not all_tools_working:
            report += """
## Recommendations

To fix issues with missing or non-functional tools:

1. Run the comprehensive auto-fix script:
   ```
   ./fix_and_verify_mcp_server.sh --auto-fix
   ```

2. If issues persist, check:
   - Server implementation in `final_mcp_server.py`
   - API endpoints in `mcp/server.py`
   - Tool registration in the server startup code

3. Restart the server after making changes:
   ```
   ./restart_mcp_server.sh
   ```
"""
    
    # Write the report to file
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Report generated successfully: {output_file}")
    return report

def main():
    """Main function to generate the report."""
    parser = argparse.ArgumentParser(description="Generate a verification report for the IPFS Accelerate MCP server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output report file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--test-output", default="report_test_results.json", help="Test results output file")
    
    args = parser.parse_args()
    
    # Run comprehensive test
    results = run_test_and_get_results(args.host, args.port, args.test_output)
    
    # Get additional info directly from the server
    hardware_info = check_hardware_info(args.host, args.port)
    health_info = check_health(args.host, args.port)
    server_info = get_server_info(args.host, args.port)
    
    # Generate and save the report
    generate_report(results, hardware_info, health_info, server_info, args.host, args.port, args.output)
    
    # Print summary to console
    if results:
        required_tools_status = results["tools"].get("required_tools_working", {})
        all_tools_working = all(required_tools_status.values())
        get_hardware_info_working = required_tools_status.get("get_hardware_info", False)
        
        if all_tools_working:
            print("✅ Verification successful! All tools are working properly.")
            return 0
        elif get_hardware_info_working:
            print("⚠️ get_hardware_info tool is working, but some other tools have issues.")
            return 0
        else:
            print("❌ Verification failed. get_hardware_info tool is not working.")
            return 1
    else:
        print("❌ Verification failed. Could not connect to MCP server.")
        return 1

if __name__ == "__main__":
    sys.exit(main())