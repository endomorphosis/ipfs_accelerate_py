#!/usr/bin/env python3
"""
Debug MCP Server

This is a modified version of the unified_mcp_server.py with additional
debugging output to help diagnose tool registration issues.
"""

import os
import sys
import time
import logging
import requests

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_registered_tools():
    """Check which tools are registered with the server."""
    try:
        time.sleep(2)  # Wait for server to start
        base_url = "http://localhost:8001"
        tools_url = f"{base_url}/tools"
        
        logger.info(f"Checking tools at: {tools_url}")
        response = requests.get(tools_url, timeout=5)
        
        if response.status_code == 200:
            tools = response.json()
            logger.info(f"Found {len(tools)} registered tools")
            
            print("\nRegistered Tools:")
            for name, details in tools.items():
                desc = details.get('description', 'No description') if isinstance(details, dict) else 'No description'
                print(f"  - {name}: {desc}")
                
            return tools
        else:
            logger.error(f"Error getting tools: {response.status_code}")
            logger.error(response.text)
            return {}
    except Exception as e:
        logger.error(f"Error checking tools: {str(e)}")
        return {}

def main():
    """Main function to start server and check tools."""
    logger.info("Starting unified MCP server with additional debugging...")
    
    # Add extra tracing to track tool registration
    import types
    import unified_mcp_server
    
    # Save original decorator
    original_register = unified_mcp_server.register_tool
    
    # Create a debug version of the decorator
    def debug_register_tool(name=None):
        """
        Debug version of register_tool that logs extra information.
        """
        logger.debug(f"Debug register_tool called with name: {name}")
        
        def decorator(func):
            tool_name = name if name else func.__name__
            logger.info(f"Registering tool: {tool_name} from function {func.__name__}")
            
            # Call original decorator
            decorated = original_register(name)(func)
            return decorated
            
        return decorator
    
    # Replace the decorator
    unified_mcp_server.register_tool = debug_register_tool
    
    # Track registration of tools from the bridge
    original_bridge_init = unified_mcp_server.IPFSAccelerateBridge.__init__
    
    def debug_bridge_init(self, real_instance=None, ipfs_client=None):
        logger.info(f"IPFSAccelerateBridge initialized with real_instance: {real_instance}")
        if real_instance:
            # Log available methods
            methods = [method for method in dir(real_instance) if callable(getattr(real_instance, method)) and not method.startswith('_')]
            logger.info(f"Available methods in real_instance: {methods}")
        
        # Call original init
        original_bridge_init(self, real_instance, ipfs_client)
    
    # Replace the bridge init
    unified_mcp_server.IPFSAccelerateBridge.__init__ = debug_bridge_init
    
    # Now start the server
    try:
        # Import all the modules we need to start the server
        from unified_mcp_server import create_app, app, start_server
        
        # Start the server in a separate thread
        import threading
        server_thread = threading.Thread(
            target=start_server, 
            args=("127.0.0.1", 8001, False)
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Check which tools are registered
        check_registered_tools()
        
        # Keep the main thread running
        logger.info("Server is running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Received interrupt. Shutting down.")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
