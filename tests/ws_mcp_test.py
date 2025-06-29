#!/usr/bin/env python3
"""
WebSocket MCP Test Client

This client tests the MCP server using a WebSocket connection.
"""

import json
import time
import sys
import threading
import queue
import websocket
import argparse
import uuid

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

class WSClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self.client_id = None
        self.server_info = None
        self.message_queue = queue.Queue()
        self.connected = threading.Event()
        self.closing = False
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            event_type = data.get("event")
            event_data = data.get("data", {})
            
            if event_type == "init":
                self.client_id = event_data.get("client_id")
                self.server_info = event_data.get("server_info")
                print_success(f"Connected with client ID: {self.client_id}")
                print_info(f"Server info: {json.dumps(self.server_info, indent=2)}")
                self.connected.set()
            else:
                if event_type != "heartbeat":
                    print_info(f"Received event: {event_type}")
                    print_info(f"Data: {json.dumps(event_data, indent=2)}")
                self.message_queue.put(data)
        except json.JSONDecodeError:
            print_warning(f"Received invalid JSON: {message}")
        except Exception as e:
            print_error(f"Error processing message: {str(e)}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        if not self.closing:
            print_error(f"WebSocket error: {str(error)}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        if not self.closing:
            print_info(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def on_open(self, ws):
        """Handle WebSocket connection open."""
        print_info(f"WebSocket connection established to {self.ws_url}")
        
    def connect(self, timeout=10):
        """Connect to the WebSocket server."""
        self.closing = False
        try:
            # Connect with WebSocket
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket connection in a thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection to be established
            if not self.connected.wait(timeout):
                print_error(f"Connection timeout after {timeout} seconds")
                self.close()
                return False
            
            return True
        except Exception as e:
            print_error(f"Connection error: {str(e)}")
            return False
    
    def close(self):
        """Close the WebSocket connection."""
        self.closing = True
        if self.ws:
            self.ws.close()
    
    def call_tool(self, tool_name, arguments, timeout=10):
        """Call a tool on the MCP server."""
        if not self.client_id:
            print_error("Not connected to the server")
            return None
        
        print_header(f"Calling tool: {tool_name}")
        print_info(f"Arguments: {json.dumps(arguments, indent=2)}")
        
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Create the tool call message
        message = {
            "type": "tool_call",
            "request_id": request_id,
            "tool_name": tool_name,
            "arguments": arguments
        }
        
        try:
            # Send the message
            self.ws.send(json.dumps(message))
            print_info(f"Tool call request sent with ID: {request_id}")
            
            # Wait for the response
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    msg = self.message_queue.get(timeout=0.5)
                    event_type = msg.get("event")
                    data = msg.get("data", {})
                    
                    # Check if this is our response
                    if event_type == "tool_response" and data.get("request_id") == request_id:
                        result = data.get("result")
                        print_success("Tool call successful!")
                        
                        if isinstance(result, dict):
                            print(json.dumps(result, indent=2))
                        else:
                            print(result)
                            
                        return result
                    
                    # Not our response, put it back in the queue
                    self.message_queue.put(msg)
                    
                except queue.Empty:
                    continue
            
            print_error(f"Timed out waiting for response after {timeout} seconds")
            return None
            
        except Exception as e:
            print_error(f"Error calling tool: {str(e)}")
            return None

def test_connection(ws_url):
    """Test the connection to the MCP server."""
    print_header("WebSocket MCP Connection Test")
    
    client = WSClient(ws_url)
    if not client.connect():
        print_error("Failed to connect to the MCP server")
        return False
    
    try:
        # Test health check
        health_result = client.call_tool("health_check", {})
        if not health_result:
            print_error("Health check failed")
            return False
        
        # Test IPFS functionality
        test_path = f"/mcp-test/ws-client-test-{int(time.time())}.txt"
        test_content = f"Test content from WebSocket client at {time.time()}"
        
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
        
        result_content = read_result.get("content", "")
        if result_content != test_content:
            print_error("Content mismatch")
            print_error(f"Expected: {test_content}")
            print_error(f"Actual: {result_content}")
            return False
        
        print_success("IPFS functionality tests passed!")
        
        # Test model functionalities
        list_models_result = client.call_tool("list_models", {})
        if not list_models_result:
            print_error("list_models failed")
            return False
        
        print_info(f"Found {list_models_result.get('count', 0)} models")
        
        if list_models_result.get("models") and len(list_models_result.get("models")) > 0:
            model_name = next(iter(list_models_result.get("models").keys()))
            print_info(f"Testing endpoint creation for model: {model_name}")
            
            endpoint_result = client.call_tool("create_endpoint", {
                "model_name": model_name
            })
            
            if not endpoint_result or not endpoint_result.get("success"):
                print_error(f"create_endpoint failed for model {model_name}")
                return False
            
            print_success("Model functionality tests passed!")
        
        print_header("Test Results")
        print_success("All tests passed! The MCP server is working correctly via WebSockets.")
        return True
        
    finally:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MCP server via WebSockets")
    parser.add_argument("--url", default="ws://localhost:8003/ws", help="WebSocket URL of the MCP server")
    
    args = parser.parse_args()
    success = test_connection(args.url)
    sys.exit(0 if success else 1)
