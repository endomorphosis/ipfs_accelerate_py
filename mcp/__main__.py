"""
Main entry point for the IPFS Accelerate MCP server.

This module allows the package to be run directly with 'python -m mcp'
and provides a command-line interface for the MCP server.
"""

import argparse
import asyncio
import sys
import os
import importlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp")

# Ensure package directory is in the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the server components - try multiple import approaches
server_module = None

# Try different import approaches
try:
    # Approach 1: Direct import 
    import server as server_module
    logger.info("Successfully imported server module directly")
except ImportError:
    try:
        # Approach 2: Absolute import
        from mcp import server as server_module
        logger.info("Successfully imported server module with absolute import")
    except ImportError:
        try:
            # Approach 3: Relative import
            from . import server as server_module
            logger.info("Successfully imported server module with relative import")
        except ImportError:
            try:
                # Approach 4: Dynamic import
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, current_dir)
                server_module = importlib.import_module("server")
                logger.info("Successfully imported server module with dynamic import")
            except ImportError as e:
                logger.error(f"Error importing IPFS Accelerate MCP server: {e}")
                print(f"Error: {e}")
                sys.exit(1)

# Get the server components
create_ipfs_mcp_server = getattr(server_module, "create_ipfs_mcp_server")
default_server = getattr(server_module, "default_server")


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog="mcp",
        description="IPFS Accelerate MCP Server for LLM integration",
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the MCP server")
    run_parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse", "ws"],
        default="stdio",
        help="Transport protocol to use (default: stdio)"
    )
    run_parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to bind to for SSE/WS transport"
    )
    run_parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to listen on for SSE/WS transport"
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    # Import command
    import_parser = subparsers.add_parser("import-model", help="Import an AI model from IPFS")
    import_parser.add_argument(
        "cid",
        help="IPFS Content Identifier (CID) of the model to import"
    )
    import_parser.add_argument(
        "--name",
        help="Name to assign to the imported model"
    )
    
    return parser.parse_args()


def print_banner():
    """Print the IPFS Accelerate MCP banner."""
    banner = """
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
    print(banner)


def main():
    """Run the IPFS Accelerate MCP server."""
    print_banner()
    
    args = parse_args()
    
    # Set debug logging if requested
    if getattr(args, 'debug', False):
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if args.command == "run" or args.command is None:
        # Run the server
        print(f"Starting IPFS Accelerate MCP server with {args.transport} transport")
        
        try:
            if args.transport in ["sse", "ws"]:
                print(f"Listening on {args.host}:{args.port}")
                kwargs = {"host": args.host, "port": args.port}
            else:
                kwargs = {}
            
            # Run the server with the specified transport
            default_server.run(transport=args.transport, **kwargs)
        except Exception as e:
            logger.error(f"Error running server: {e}")
            print(f"Error: {e}")
            sys.exit(1)
        
    elif args.command == "import-model":
        # Import a model (example command - would be implemented in real code)
        print(f"Importing model with CID: {args.cid}")
        print(f"Model name: {args.name or 'auto-generated'}")
        print("This is a placeholder for the model import functionality.")
        print("It would be implemented in a future version.")


if __name__ == "__main__":
    main()
