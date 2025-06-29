#!/usr/bin/env python
"""
IPFS Accelerate MCP Server Installer

This script installs the IPFS Accelerate MCP server for use with Claude or other
AI assistants that support the Model Context Protocol.
"""

import os
import sys
import argparse
import subprocess
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_installer")

def check_dependencies():
    """
    Check if required dependencies are installed
    
    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    required_packages = ["fastmcp", "uvicorn", "fastapi", "sse-starlette"]
    
    try:
        import pkg_resources
        
        for package in required_packages:
            try:
                pkg_resources.get_distribution(package)
                logger.info(f"✅ {package} is installed")
            except pkg_resources.DistributionNotFound:
                logger.error(f"❌ {package} is not installed")
                return False
        
        return True
    except ImportError:
        logger.error("Could not import pkg_resources, cannot check dependencies")
        return False

def install_dependencies():
    """
    Install required dependencies
    
    Returns:
        bool: True if all dependencies were installed successfully, False otherwise
    """
    try:
        requirements_path = os.path.join("ipfs_accelerate_py", "mcp", "requirements.txt")
        
        logger.info(f"Installing dependencies from {requirements_path}")
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_path
        ])
        
        logger.info("Dependencies installed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

def install_claude_mcp_server(name="IPFS Accelerate", port=None, debug=True):
    """
    Install the MCP server for Claude
    
    Args:
        name: Name of the server in Claude
        port: Port to run the server on (None for auto-detection)
        debug: Enable debug mode
        
    Returns:
        bool: True if the server was installed successfully, False otherwise
    """
    try:
        # Check if fastmcp is installed
        try:
            import fastmcp
            logger.info("FastMCP is installed, registering with Claude")
        except ImportError:
            logger.error("FastMCP is not installed, cannot register with Claude")
            return False
        
        # Determine the command to run the server
        run_script = os.path.abspath("run_ipfs_mcp.py")
        
        cmd = [sys.executable, run_script, "--transport", "sse"]
        
        if debug:
            cmd.append("--debug")
        
        if port:
            cmd.extend(["--port", str(port)])
        else:
            cmd.append("--find-port")
        
        # Create the config file for Claude
        config = {
            "name": name,
            "command": cmd,
            "autostart": True
        }
        
        # Get the Claude MCP settings directory
        home_dir = Path.home()
        claude_settings_dir = home_dir / ".config" / "Claude" / "mcp"
        claude_settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the config file
        config_path = claude_settings_dir / "ipfs_accelerate.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"MCP server installed for Claude at {config_path}")
        logger.info(f"Server name: {name}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error installing MCP server for Claude: {e}")
        return False

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Install IPFS Accelerate MCP Server")
    
    parser.add_argument("--name", default="IPFS Accelerate", help="Name of the server in Claude")
    parser.add_argument("--port", type=int, help="Port to run the server on (default: auto-detect)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print(" IPFS Accelerate MCP Server Installer ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            logger.info("Installing dependencies...")
            if not install_dependencies():
                logger.error("Failed to install dependencies")
                sys.exit(1)
    
    # Install the MCP server for Claude
    if install_claude_mcp_server(
        name=args.name,
        port=args.port,
        debug=not args.no_debug
    ):
        print("\n" + "=" * 80)
        print(" IPFS Accelerate MCP Server Installed Successfully ".center(80, "="))
        print("=" * 80 + "\n")
        
        print("You can now use the IPFS Accelerate MCP server with Claude.")
        print("Open Claude and enable the IPFS Accelerate server in the Claude settings.")
        print("\nFor more information, see the integration guide:")
        print("IPFS_ACCELERATE_MCP_INTEGRATION_GUIDE.md\n")
    else:
        logger.error("Failed to install MCP server for Claude")
        sys.exit(1)

if __name__ == "__main__":
    main()
