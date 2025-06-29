#!/usr/bin/env python3
"""
MCP Server Diagnostics

This script tests and diagnoses issues with the MCP server implementation.
"""

import os
import sys
import time
import logging
import importlib
import traceback
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("mcp_diagnostics.log")
    ]
)
logger = logging.getLogger("mcp_diagnostics")

def check_imports():
    """Check if required modules can be imported"""
    modules_to_check = [
        "mcp",
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests"
    ]
    
    print("Checking required imports:")
    for module_name in modules_to_check:
        try:
            importlib.import_module(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {str(e)}")
    print()

def check_platform():
    """Check platform and environment"""
    print("Platform information:")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()}")
    print(f"Machine: {platform.machine()}")
    print()
    
    print("Environment variables:")
    for key, value in sorted(os.environ.items()):
        if key.startswith("MCP_") or key.startswith("IPFS_"):
            print(f"{key}: {value}")
    print()

def test_mcp_server_start():
    """Test starting the MCP server in this process"""
    print("Attempting to start MCP server in this process...")
    try:
        from mcp import start_server
        print("MCP start_server function found. Trying to start...")
        
        # Set environment variables for testing
        os.environ["MCP_PORT"] = "8099"  # Use a different port for this test
        os.environ["MCP_HOST"] = "127.0.0.1"
        
        # Define a simple tool for testing
        def hello_world():
            """Simple test tool"""
            return {"message": "Hello, world!"}
        
        # Register the tool
        print("Registering test tool...")
        from mcp import register_tool
        register_tool("hello_world", "Simple test tool", hello_world)
        print("Tool registered")
        
        # Start the server with a timeout
        def timeout_start_server():
            print("Starting server with 5-second timeout...")
            import threading
            
            def server_thread():
                try:
                    start_server()
                except Exception as e:
                    print(f"Error in server thread: {str(e)}")
                    traceback.print_exc()
            
            thread = threading.Thread(target=server_thread)
            thread.daemon = True
            thread.start()
            thread.join(5)  # Wait 5 seconds
            
            if thread.is_alive():
                print("Server started successfully (still running)")
                print("This is normal - the server runs in a blocking loop")
                return True
            else:
                print("Server thread exited unexpectedly")
                return False
        
        result = timeout_start_server()
        return result
    
    except ImportError as e:
        print(f"ImportError: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        return False

def test_http_endpoint(url="http://localhost:8002/mcp/manifest"):
    """Test if the HTTP endpoint is responding"""
    print(f"Testing HTTP endpoint: {url}")
    try:
        import requests
        response = requests.get(url, timeout=3)
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            print("Endpoint is responding correctly!")
            try:
                data = response.json()
                print(f"Server name: {data.get('server_name')}")
                print(f"MCP version: {data.get('mcp_version')}")
                
                tools = data.get('tools', {})
                print(f"Available tools: {list(tools.keys())}")
                
                resources = data.get('resources', {})
                print(f"Available resources: {list(resources.keys())}")
            except Exception as e:
                print(f"Error parsing JSON: {str(e)}")
        else:
            print(f"Unexpected status code: {response.status_code}")
            print(f"Response text: {response.text}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("Connection error: Could not connect to the server")
        return False
    except Exception as e:
        print(f"Error testing endpoint: {str(e)}")
        return False

def main():
    """Main function"""
    print("=" * 80)
    print("MCP Server Diagnostics")
    print("=" * 80)
    print()
    
    # Check imports
    check_imports()
    
    # Check platform
    check_platform()
    
    # Test HTTP endpoint
    endpoint_ok = test_http_endpoint()
    if not endpoint_ok:
        print("\nEndpoint test failed. Trying to start server...")
        server_ok = test_mcp_server_start()
        if server_ok:
            print("\nServer appears to start correctly in this process.")
            print("There might be an issue with the previous server instance.")
        else:
            print("\nServer failed to start in this process.")
            print("This indicates a more fundamental issue with the MCP implementation.")
    else:
        print("\nEndpoint test passed. Server is running correctly.")
    
    print("\nDiagnostic complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
