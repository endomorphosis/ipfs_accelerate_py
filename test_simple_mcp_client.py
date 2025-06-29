#!/usr/bin/env python
"""
IPFS Accelerate MCP Client Test

This script tests the MCP client by connecting to a server and retrieving hardware information.
"""

import os
import sys
import json
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Test MCP Client")
    parser.add_argument("--port", type=int, default=8002,
                      help="Port to connect to (default: 8002)")
    parser.add_argument("--host", type=str, default="localhost",
                      help="Host to connect to (default: localhost)")
    parser.add_argument("--start", action="store_true",
                      help="Start the MCP server if it's not already running")
    parser.add_argument("--output", type=str, default=None,
                      help="Save output to JSON file")
    args = parser.parse_args()
    
    # Import the MCP client
    try:
        from mcp import MCPClient, is_server_running, start_server
    except ImportError as e:
        print(f"Error importing MCP client: {e}")
        print("Make sure you've installed the required packages")
        return 1
    
    print("\n==== IPFS Accelerate MCP Client Test ====\n")
    
    # Create the client
    client = MCPClient(host=args.host, port=args.port)
    
    # Check if the server is running
    if not client.is_server_available():
        print(f"MCP server is not available at {args.host}:{args.port}")
        
        if args.start:
            print("\nAttempting to start the MCP server...")
            success, port = start_server(port=args.port, wait=2)
            if success:
                print(f"Started MCP server on port {port}")
                # Update client port if needed
                if port != args.port:
                    client = MCPClient(host=args.host, port=port)
                    print(f"Updated client to use port {port}")
            else:
                print(f"Failed to start MCP server")
                print("\nTo start the server manually, try:")
                print(f"python -m mcp.run_server --port {args.port}")
                return 1
    
    # Check again if the server is running
    if not client.is_server_available():
        print("MCP server is still not available")
        return 1
    
    print(f"MCP server is available at {args.host}:{client.port}")
    
    # Get server manifest
    manifest = client.get_manifest()
    print(f"\nServer: {manifest.get('server_name', 'Unknown')} v{manifest.get('version', '?')}")
    
    # Get available tools and resources
    print("\nAvailable Tools:")
    for tool_name, tool_info in manifest.get("tools", {}).items():
        print(f"  - {tool_name}: {tool_info.get('description', '')}")
    
    print("\nAvailable Resources:")
    for resource_name, resource_info in manifest.get("resources", {}).items():
        print(f"  - {resource_name}: {resource_info.get('description', '')}")
    
    # Get hardware information
    print("\nRetrieving hardware information...")
    hardware_info = client.get_hardware_info()
    
    # Process and display hardware info
    if "error" in hardware_info:
        print(f"Error: {hardware_info['error']}")
        return 1
    
    # Display system info
    if "system" in hardware_info:
        system = hardware_info["system"]
        print("\nSystem Information:")
        print(f"  OS: {system.get('os', 'Unknown')} {system.get('os_version', '')}")
        print(f"  Architecture: {system.get('architecture', 'Unknown')}")
        print(f"  Python Version: {system.get('python_version', 'Unknown')}")
        print(f"  Hostname: {system.get('hostname', 'Unknown')}")
    
    # Display accelerators
    if "accelerators" in hardware_info:
        accelerators = hardware_info["accelerators"]
        print("\nHardware Accelerators:")
        
        for name, info in accelerators.items():
            available = info.get("available", False)
            print(f"\n  {name.upper()}: {'Available' if available else 'Not Available'}")
            
            if available:
                # Show details for available accelerators
                for key, value in info.items():
                    if key != "available" and not key.startswith("_"):
                        print(f"    {key}: {value}")
    
    # Save output to file if requested
    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(hardware_info, f, indent=2)
            print(f"\nHardware information saved to {args.output}")
        except Exception as e:
            print(f"Error saving output: {e}")
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
