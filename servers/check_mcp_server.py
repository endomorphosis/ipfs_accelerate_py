#!/usr/bin/env python3
"""
Simple MCP Server Checker
"""

import sys
import os
import subprocess
import time
import json
import socket
import requests

def check_port(port=8002):
    """Check if port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def main():
    print("Simple MCP Server Checker")
    print("========================")
    
    # Check if port is already in use
    if check_port(8002):
        print("Port 8002 is already in use!")
        return 1
    
    # Check if the server file exists
    server_file = "final_mcp_server.py"
    if not os.path.exists(server_file):
        print(f"Error: {server_file} does not exist!")
        return 1
    
    # Make the server file executable
    os.chmod(server_file, os.stat(server_file).st_mode | 0o111)
    
    # Start the server
    print(f"Starting {server_file}...")
    process = subprocess.Popen(
        [sys.executable, server_file, "--debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Give the server time to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Check if the server started
    if process.poll() is not None:
        # Server exited early
        stdout, stderr = process.communicate()
        print("Server failed to start! Output:")
        print("-" * 40)
        print(stdout)
        print("-" * 40)
        print(stderr)
        return 1
    
    # Test server connection
    print("Testing server connection...")
    try:
        response = requests.get("http://localhost:8002/mcp/manifest", timeout=5)
        print(f"Server response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Server manifest: {json.dumps(data, indent=2)}")
            print("Server is working correctly!")
        else:
            print(f"Error: Server returned status {response.status_code}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
    
    # Stop the server
    print("Stopping server...")
    process.terminate()
    process.wait(timeout=5)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
