#!/usr/bin/env python3
"""
Simple script to run the IPFS Accelerate MCP server.

This script serves as a simple entry point to run the MCP server without
having to deal with complex import paths.

Example usage:
    python run_mcp.py
    python run_mcp.py --transport ws --port 8080
"""

import os
import sys
import argparse
import importlib.util
from typing import Any

# Banner to print when starting
BANNER = """
╭───────────────────────────────────────────╮
│                                           │
│  IPFS Accelerate MCP Server               │
│  Model Context Protocol Integration       │
│                                           │
│  Provide IPFS operations to LLMs          │
│  Accelerate AI models with IPFS           │
│                                           │
╰───────────────────────────────────────────╯
"""


def import_module_from_path(module_path: str, module_name: str = "") -> Any:
    """Import a module from a specific file path.
    
    Args:
        module_path: Path to the module file
        module_name: Name to assign to the module (defaults to file basename)
        
    Returns:
        The imported module
    """
    # Verify the file exists
    if not os.path.exists(module_path):
        raise ImportError(f"Module file not found: {module_path}")
    
    # Generate module name if not provided
    if not module_name:
        module_name = os.path.basename(module_path).split('.')[0]
    
    # Load the module spec
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")
    
    # Create the module from the spec
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    # Execute the module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing module {module_path}: {e}")
    
    return module


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the IPFS Accelerate MCP server")
    
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse", "ws"],
        default="stdio",
        help="Transport protocol to use (default: stdio)"
    )
    parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to bind to for SSE/WS transport"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to listen on for SSE/WS transport"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main():
    """Run the MCP server."""
    # Print banner
    print(BANNER)
    
    # Parse arguments
    args = parse_args()
    
    # Import the server module directly from its path
    server_path = os.path.join('mcp', 'server.py')
    try:
        server_module = import_module_from_path(server_path, 'mcp_server')
        print("Successfully imported MCP server module")
    except ImportError as e:
        print(f"Error importing MCP server: {e}")
        sys.exit(1)
    
    # Get the server instance
    mcp_server = server_module.default_server
    
    # Run the server
    print(f"Starting IPFS Accelerate MCP server with {args.transport} transport")
    
    try:
        if args.transport in ["sse", "ws"]:
            print(f"Listening on {args.host}:{args.port}")
            kwargs = {"host": args.host, "port": args.port}
        else:
            kwargs = {}
        
        # Run the server with the specified transport
        mcp_server.run(transport=args.transport, **kwargs)
    except Exception as e:
        print(f"Error running server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
