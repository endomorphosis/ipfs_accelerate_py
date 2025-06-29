#!/usr/bin/env python
"""
IPFS Accelerate MCP Runner

This script provides a convenient way to run the IPFS Accelerate MCP server
and perform basic operations using the client.
"""

import os
import sys
import json
import logging
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def print_separator():
    """Print a separator line"""
    print("=" * 80)

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Runner")
    parser.add_argument("--port", type=int, default=8002,
                      help="Port to bind to (default: 8002)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--client-only", action="store_true",
                      help="Run only the client, connecting to an existing server")
    parser.add_argument("--server-only", action="store_true",
                      help="Run only the server, without starting a client")
    parser.add_argument("--debug", action="store_true",
                      help="Run in debug mode")
    parser.add_argument("--find-port", action="store_true",
                      help="Find an available port if the specified port is in use")
    parser.add_argument("--info", action="store_true",
                      help="Show hardware information (default action for client)")
    parser.add_argument("--list-tools", action="store_true",
                      help="List available tools")
    parser.add_argument("--list-resources", action="store_true",
                      help="List available resources")
    parser.add_argument("--tool", type=str, default=None,
                      help="Call a specific tool on the server")
    parser.add_argument("--resource", type=str, default=None,
                      help="Access a specific resource on the server")
    parser.add_argument("--args", type=str, default="{}",
                      help="JSON-encoded arguments for the tool")
    parser.add_argument("--output", type=str, default=None,
                      help="Save output to a JSON file")
    parser.add_argument("--no-register-tools", action="store_true",
                      help="Don't register tools automatically")
    args = parser.parse_args()
    
    # Import MCP components
    try:
        from mcp import MCPClient, is_server_running, start_server
    except ImportError as e:
        print(f"Error importing MCP components: {e}")
        print("Make sure you've installed the required packages")
        return 1
    
    print_separator()
    print("IPFS Accelerate MCP")
    print_separator()
    
    # Start the server if needed
    if not args.client_only:
        if is_server_running(port=args.port, host="localhost"):
            print(f"MCP server is already running on port {args.port}")
        else:
            print(f"Starting MCP server on {args.host}:{args.port}...")
            if args.server_only:
                # Run the server directly in this process
                from mcp.server import start_server as run_server
                run_server(host=args.host, port=args.port, find_port=args.find_port, debug=args.debug)
                
                # Register tools unless explicitly disabled
                if not args.no_register_tools:
                    print("Registering MCP tools...")
                    try:
                        # Import the register_mcp_tools module and call its main function
                        import register_mcp_tools
                        register_mcp_tools.main()
                        print("Tools registered successfully")
                    except Exception as e:
                        print(f"Error registering tools: {e}")
                
                return 0
            else:
                # Start server in a separate process
                success, port = start_server(port=args.port, host=args.host, find_port=args.find_port, debug=args.debug, wait=2)
                if success:
                    print(f"MCP server started on port {port}")
                    if port != args.port:
                        args.port = port  # Update port for client
                        
                    # Register tools unless explicitly disabled
                    if not args.no_register_tools:
                        print("Registering MCP tools...")
                        try:
                            # Import the register_mcp_tools module and call its main function
                            import register_mcp_tools
                            register_mcp_tools.main()
                            print("Tools registered successfully")
                        except Exception as e:
                            print(f"Error registering tools: {e}")
                else:
                    print("Failed to start MCP server")
                    return 1
    
    # Return if server only
    if args.server_only:
        return 0
    
    # Otherwise create and run client
    client = MCPClient(host="localhost", port=args.port)
    
    # Check if server is available
    if not client.is_server_available():
        print(f"MCP server is not available at localhost:{args.port}")
        return 1
    
    print(f"Connected to MCP server at localhost:{args.port}")
    
    # Get manifest
    manifest = client.get_manifest()
    print(f"\nServer: {manifest.get('server_name', 'Unknown')} v{manifest.get('version', '?')}")
    print(f"MCP Version: {manifest.get('mcp_version', '?')}")
    
    # List tools if requested
    if args.list_tools:
        print("\nAvailable Tools:")
        for tool_name, tool_info in manifest.get("tools", {}).items():
            print(f"  - {tool_name}: {tool_info.get('description', '')}")
    
    # List resources if requested
    if args.list_resources:
        print("\nAvailable Resources:")
        for resource_name, resource_info in manifest.get("resources", {}).items():
            print(f"  - {resource_name}: {resource_info.get('description', '')}")
    
    # Call tool if specified
    result = None
    if args.tool:
        print(f"\nCalling tool: {args.tool}")
        try:
            kwargs = json.loads(args.args)
        except json.JSONDecodeError as e:
            print(f"Error parsing tool arguments: {e}")
            return 1
        
        result = client.call_tool(args.tool, **kwargs)
        print(json.dumps(result, indent=2))
    
    # Access resource if specified
    if args.resource:
        print(f"\nAccessing resource: {args.resource}")
        result = client.access_resource(args.resource)
        print(json.dumps(result, indent=2))
    
    # Get hardware info if no specific operation was requested
    if not args.tool and not args.resource and (args.info or not (args.list_tools or args.list_resources)):
        print("\nRetrieving hardware information...")
        result = client.get_hardware_info()
        
        # Display simplified hardware info
        if "system" in result:
            system = result["system"]
            print(f"\nSystem Information:")
            print(f"  OS: {system.get('os', 'Unknown')} {system.get('os_version', '')}")
            print(f"  Architecture: {system.get('architecture', 'Unknown')}")
            print(f"  Python Version: {system.get('python_version', 'Unknown')}")
            print(f"  Hostname: {system.get('hostname', 'Unknown')}")
        
        if "accelerators" in result:
            accelerators = result["accelerators"]
            print(f"\nHardware Accelerators:")
            
            for name, info in accelerators.items():
                available = info.get("available", False)
                print(f"\n  {name.upper()}: {'Available' if available else 'Not Available'}")
                
                if available:
                    # Show details for available accelerators
                    for key, value in info.items():
                        if key != "available" and not key.startswith("_"):
                            print(f"    {key}: {value}")
    
    # Save output to file if requested
    if args.output and result:
        try:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nOutput saved to {args.output}")
        except Exception as e:
            print(f"Error saving output: {e}")
    
    print_separator()
    return 0

if __name__ == "__main__":
    sys.exit(main())
