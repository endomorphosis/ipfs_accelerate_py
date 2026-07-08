"""
Command-line entry point for the IPFS Accelerate MCP server.

This module allows running the MCP server directly with:
    python -m mcp
"""

import argparse
import logging
import sys
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_mcp_cli")

def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="IPFS Accelerate MCP Server",
        prog="python -m mcp"
    )
    
    # Define commands as subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the MCP server")
    run_parser.add_argument("-t", "--transport", choices=["stdio", "ws", "sse"], 
                         default="stdio", help="Transport to use")
    run_parser.add_argument("--host", default="127.0.0.1", 
                         help="Host to listen on (for ws and sse transports)")
    run_parser.add_argument("--port", type=int, default=8000, 
                         help="Port to listen on (for ws and sse transports)")
    run_parser.add_argument("--debug", action="store_true", 
                         help="Enable debug logging")
    
    # Version command
    subparsers.add_parser("version", help="Show version information")
    
    # Parse the arguments
    return parser.parse_args(args)

def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        Exit code
    """
    args = parse_args(sys.argv[1:])
    
    # Print help if no command is specified
    if not args.command:
        print("Error: command is required", file=sys.stderr)
        print("Run 'python -m mcp --help' for usage", file=sys.stderr)
        return 1
    
    # Handle commands
    if args.command == "run":
        # Import the run_server function
        try:
            from mcp import run_server
            
            # Set log level
            if args.debug:
                logging.getLogger().setLevel(logging.DEBUG)
                logger.setLevel(logging.DEBUG)
            
            # Run the server
            logger.info(f"Starting MCP server with {args.transport} transport")
            if args.transport in ["ws", "sse"]:
                logger.info(f"Listening on {args.host}:{args.port}")
            
            try:
                run_server(
                    transport=args.transport,
                    host=args.host,
                    port=args.port,
                    debug=args.debug
                )
                return 0
            except KeyboardInterrupt:
                logger.info("Server stopped by user")
                return 0
            except Exception as e:
                logger.error(f"Error running server: {str(e)}")
                return 1
        except ImportError as e:
            logger.error(f"Error importing run_server: {str(e)}")
            return 1
    
    elif args.command == "version":
        # Show version information
        try:
            from mcp import __version__
            print(f"IPFS Accelerate MCP version: {__version__}")
            
            # Try to get IPFS version
            try:
                import ipfs_kit_py
                ipfs = ipfs_kit_py.IPFSApi()
                version = ipfs.version()
                print(f"IPFS version: {version.get('Version', 'unknown')}")
            except (ImportError, Exception):
                print("IPFS version: Not available")
            
            # Check if FastMCP is available
            try:
                import fastmcp
                print(f"FastMCP version: {getattr(fastmcp, '__version__', 'unknown')}")
            except ImportError:
                print("FastMCP: Not available (using mock implementation)")
            
            return 0
        except ImportError as e:
            logger.error(f"Error getting version information: {str(e)}")
            return 1
    
    # Unknown command
    logger.error(f"Unknown command: {args.command}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
