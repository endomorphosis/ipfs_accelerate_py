#!/usr/bin/env python3
"""
Test Claude MCP Connection

This script simulates a Claude client connecting to the enhanced MCP server
and making tool calls. Use this to verify the MCP server is properly
configured for Claude to connect to it.
"""

import json
import requests
import uuid
import time
import sys
import argparse
import os
from sseclient import SSEClient
import threading
import queue
import pprint

# ANSI colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'

def print_info(msg):
    print(f"{BLUE}[INFO]{RESET} {msg}")

def print_success(msg):
    print(f"{GREEN}[SUCCESS]{RESET} {msg}")

def print_error(msg):
    print(f"{RED}[ERROR]{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}[WARNING]{RESET} {msg}")

def print_header(msg):
    print(f"\n{CYAN}=== {msg} ==={RESET}\n")

class MCPClientSimulator:
    """
    Simulates a Claude client connecting to an MCP server.
    """
    
    def __init__(self, sse_url, tool_call_url=None):
        self.sse_url = sse_url
        self.tool_call_url = tool_call_url or sse_url.replace("/sse", "/sse/request")
        self.client_id = None
        self.sse_client = None
        self.message_queue = queue.Queue()
        self.sse_thread = None
        self.running = False
        self.server_info = None
    
    def start(self):
        """Start the SSE connection and listening thread."""
        print_info(f"Connecting to SSE endpoint: {self.sse_url}")
        try:
            self.sse_client = SSEClient(self.sse_url)
            self.running = True
            
            # Start SSE listening thread
            self.sse_thread = threading.Thread(target=self._sse_listener)
            self.sse_thread.daemon = True
            self.sse_thread.start()
            
            # Wait for init message
            print_info("Waiting for init message...")
            init_timeout = 5
            start_time = time.time()
            
            while time.time() - start_time < init_timeout:
                try:
                    msg = self.message_queue.get(timeout=1)
                    if msg.get("event") == "init":
                        self.client_id = msg.get("data", {}).get("client_id")
                        self.server_info = msg.get("data", {}).get("server_info")
                        print_success(f"Connected! Client ID: {self.client_id}")
                        print_info(f"Server Info: {json.dumps(self.server_info, indent=2)}")
                        return True
                except queue.Empty:
                    continue
            
            print_error("Timed out waiting for init message")
            return False
            
        except Exception as e:
            print_error(f"Error connecting to SSE endpoint: {str(e)}")
            return False
    
    def _sse_listener(self):
        """Listen for SSE events and put them in the message queue."""
        try:
            # Instead of iterating directly, use the events() method
            for event in self.sse_client.events():
                if not self.running:
                    break
                
                try:
                    # Handle both string and bytes data formats
                    if isinstance(event.data, bytes):
                        data_str = event.data.decode('utf-8')
                    else:
                        data_str = event.data
                    
                    data = json.loads(data_str) if data_str else {}
                except Exception as e:
                    print_warning(f"Failed to parse event data: {str(e)}")
                    data = event.data
                
                message = {
                    "id": event.id,
                    "event": event.event,
                    "data": data
                }
                
                # Only log non-heartbeat messages
                if event.event != "heartbeat":
                    print_info(f"Received event: {event.event}")
                    print_info(f"Data: {json.dumps(data, indent=2) if isinstance(data, dict) else str(data)}")
                
                self.message_queue.put(message)
        
        except Exception as e:
            if self.running:
                print_error(f"SSE listener error: {str(e)}")
    
    def call_tool(self, tool_name, arguments):
        """Call a tool on the MCP server."""
        if not self.client_id:
            print_error("Not connected to MCP server")
            return None
        
        print_header(f"Calling tool: {tool_name}")
        print_info(f"Arguments: {json.dumps(arguments, indent=2)}")
        
        request_id = str(uuid.uuid4())
        
        payload = {
            "client_id": self.client_id,
            "request_id": request_id,
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments
        }
        
        try:
            # Send tool call request
            response = requests.post(self.tool_call_url, json=payload)
            
            if response.status_code != 200:
                print_error(f"Tool call request failed: {response.status_code}")
                print_error(f"Response: {response.text}")
                return None
            
            # Wait for tool response
            print_info(f"Tool call request sent, waiting for response...")
            
            # Wait for response with timeout
            timeout = 10
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    msg = self.message_queue.get(timeout=0.5)
                    
                    if msg.get("event") == "tool_response" and msg.get("data", {}).get("request_id") == request_id:
                        result = msg.get("data", {}).get("result")
                        print_success("Tool call successful!")
                        print_info("Result:")
                        
                        if isinstance(result, dict):
                            print(json.dumps(result, indent=2))
                        else:
                            print(result)
                        
                        return result
                    
                    # Put other messages back in the queue
                    self.message_queue.put(msg)
                
                except queue.Empty:
                    continue
            
            print_error("Timed out waiting for tool response")
            return None
            
        except Exception as e:
            print_error(f"Error calling tool: {str(e)}")
            return None
    
    def stop(self):
        """Stop the SSE connection and listening thread."""
        self.running = False
        if self.sse_thread and self.sse_thread.is_alive():
            self.sse_thread.join(2)
        
        print_info("Connection closed")

def test_tool_calls(client):
    """Test various tool calls."""
    # Test health check
    health_result = client.call_tool("health_check", {})
    if not health_result:
        return False
    
    # Test IPFS functionality with a simple files_write and files_read
    test_path = "/mcp-test/claude-test.txt"
    test_content = f"Test content from Claude simulator at {time.time()}"
    
    write_result = client.call_tool("ipfs_files_write", {
        "path": test_path,
        "content": test_content
    })
    
    if not write_result or not write_result.get("success"):
        print_error("IPFS files_write failed")
        return False
    
    read_result = client.call_tool("ipfs_files_read", {
        "path": test_path
    })
    
    if not read_result:
        print_error("IPFS files_read failed")
        return False
    
    if read_result.get("content") != test_content:
        print_error("IPFS read content doesn't match written content")
        print_error(f"Expected: {test_content}")
        print_error(f"Actual: {read_result.get('content')}")
        return False
    
    print_success("IPFS functionality tests passed!")
    
    # Test model functionality
    list_models_result = client.call_tool("list_models", {})
    if not list_models_result:
        print_error("list_models failed")
        return False
    
    # If we have models, try creating an endpoint
    if list_models_result.get("models") and len(list_models_result.get("models")) > 0:
        model_name = next(iter(list_models_result.get("models").keys()))
        
        endpoint_result = client.call_tool("create_endpoint", {
            "model_name": model_name
        })
        
        if not endpoint_result or not endpoint_result.get("success"):
            print_error(f"create_endpoint failed for model {model_name}")
            return False
        
        print_success("Model functionality tests passed!")
    
    return True

def check_mcp_settings(settings_path=None):
    """Check the Claude MCP settings file."""
    if not settings_path:
        # Try to find the default settings path
        home = os.path.expanduser("~")
        possible_paths = [
            os.path.join(home, ".config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"),
            os.path.join(home, "Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"),
            os.path.join(home, "AppData/Roaming/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                settings_path = path
                break
    
    if not settings_path or not os.path.exists(settings_path):
        print_error("Could not find Claude MCP settings file")
        return None
    
    print_info(f"Checking MCP settings file: {settings_path}")
    
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        if "mcpServers" not in settings:
            print_error("No MCP servers configured in settings file")
            return None
        
        print_success(f"Found {len(settings['mcpServers'])} MCP servers in settings")
        
        for server_name, server_config in settings["mcpServers"].items():
            print_info(f"Server: {server_name}")
            print_info(f"  URL: {server_config.get('url')}")
            print_info(f"  Transport: {server_config.get('transportType')}")
            print_info(f"  Disabled: {server_config.get('disabled')}")
            
            if server_config.get("url") == "http://localhost:8002/sse" and not server_config.get("disabled"):
                print_success(f"Found our MCP server: {server_name}")
        
        return settings
            
    except Exception as e:
        print_error(f"Error reading settings file: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test Claude MCP Connection")
    parser.add_argument("--url", default="http://localhost:8002/sse", help="SSE URL of the MCP server")
    parser.add_argument("--check-settings", action="store_true", help="Check the Claude MCP settings file")
    parser.add_argument("--settings-path", help="Path to the Claude MCP settings file")
    
    args = parser.parse_args()
    
    if args.check_settings:
        settings = check_mcp_settings(args.settings_path)
        print_header("MCP Settings Check")
        if settings:
            print_success("MCP settings file found and parsed successfully")
        else:
            print_error("Failed to find or parse MCP settings file")
    
    print_header("MCP Connection Test")
    
    client = MCPClientSimulator(args.url)
    if not client.start():
        print_error("Failed to connect to MCP server")
        return 1
    
    print_header("MCP Tool Tests")
    
    try:
        success = test_tool_calls(client)
        
        print_header("Test Results")
        
        if success:
            print_success("All tests passed! The MCP server is working correctly.")
            print_success("Claude should be able to connect to this server.")
            print_info("If Claude still can't see the tools, try:")
            print_info("1. Restart VSCode completely")
            print_info("2. Reload the Claude extension (Ctrl+Shift+P, then 'Developer: Reload Window')")
            print_info("3. Start a new conversation with Claude")
        else:
            print_error("Some tests failed. Check the logs above for details.")
        
        return 0 if success else 1
    
    finally:
        client.stop()

if __name__ == "__main__":
    sys.exit(main())
