#!/usr/bin/env python3
"""
Start MCP Server with IPFS Tools

This script starts an MCP server with IPFS tools registered using the fixed registration mechanism.
"""

import argparse
import logging
import os
import sys
import tempfile
import hashlib
from typing import Dict, Any, Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_ipfs_server")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Start MCP server with IPFS tools")
    parser.add_argument("--port", type=int, default=8002, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Import MCP server functions
    try:
        # Try direct import first
        try:
            from mcp.server import register_tool, register_resource, start_server
        except ImportError:
            # Try relative import if direct import fails
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from mcp.server import register_tool, register_resource, start_server
    except ImportError as e:
        logger.error(f"Could not import MCP server: {e}")
        logger.error("Make sure the MCP server is installed or in your PYTHONPATH.")
        sys.exit(1)
    
    # Register resources
    logger.info("Registering resources...")
    
    # Define resource getter functions
    def get_ipfs_nodes():
        """Get information about connected IPFS nodes."""
        return {
            "count": 1,
            "nodes": [
                {
                    "id": "QmMockIPFSNode",
                    "addresses": ["/ip4/127.0.0.1/tcp/4001"],
                    "agent_version": "mock-ipfs/0.1.0"
                }
            ]
        }
    
    def get_ipfs_files():
        """Get information about files stored in IPFS."""
        return {
            "count": len(mock_ipfs_data),
            "files": [
                {"cid": cid, "size": len(content)}
                for cid, content in mock_ipfs_data.items()
            ]
        }
    
    def get_ipfs_pins():
        """Get information about pinned content in IPFS."""
        return {
            "count": len(mock_ipfs_pins),
            "pins": list(mock_ipfs_pins)
        }
    
    # Register resources
    register_resource("ipfs_nodes", "Information about connected IPFS nodes", get_ipfs_nodes)
    register_resource("ipfs_files", "Information about files stored in IPFS", get_ipfs_files)
    register_resource("ipfs_pins", "Information about pinned content in IPFS", get_ipfs_pins)
    
    # Create mock IPFS implementation
    logger.info("Creating mock IPFS implementation...")
    
    # We'll track CIDs and file contents in memory for the mock implementation
    mock_ipfs_data = {}
    mock_ipfs_pins = set()
    mock_ipfs_mfs = {}
    
    # Register IPFS tools
    logger.info("Registering IPFS tools...")
    
    def ipfs_add_file(path: str) -> Dict[str, Any]:
        """Add a file to IPFS."""
        if not path:
            return {"error": "Path is required", "success": False}
        
        if not os.path.exists(path):
            return {"error": "File not found", "success": False}
        
        try:
            with open(path, "rb") as f:
                content = f.read()
                # Create a mock CID using a short hash of the content
                hash_obj = hashlib.sha256(content)
                cid = f"QmPy{hash_obj.hexdigest()[:16]}"
                
                # Store the file content
                mock_ipfs_data[cid] = content
                
                return {
                    "cid": cid,
                    "size": len(content),
                    "path": path,
                    "success": True
                }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def ipfs_cat(cid: str) -> Dict[str, Any]:
        """Read the contents of a file from IPFS."""
        if not cid:
            return {"error": "CID is required", "success": False}
        
        if cid not in mock_ipfs_data:
            return {"error": "CID not found", "success": False}
        
        content = mock_ipfs_data[cid]
        return {
            "content": content.decode("utf-8", errors="replace"),
            "size": len(content),
            "success": True
        }
    
    def ipfs_get_file(cid: str, output_path: str) -> Dict[str, Any]:
        """Download a file from IPFS."""
        if not cid:
            return {"error": "CID is required", "success": False}
        if not output_path:
            return {"error": "Output path is required", "success": False}
        
        if cid not in mock_ipfs_data:
            return {"error": "CID not found", "success": False}
        
        try:
            content = mock_ipfs_data[cid]
            with open(output_path, "wb") as f:
                f.write(content)
            
            return {
                "path": output_path,
                "size": len(content),
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def ipfs_files_write(path: str, content: str) -> Dict[str, Any]:
        """Write to a file in IPFS MFS (Mutable File System)."""
        if not path:
            return {"error": "Path is required", "success": False}
        if content is None:
            return {"error": "Content is required", "success": False}
        
        try:
            # Normalize path
            path = path.strip()
            if not path.startswith("/"):
                path = "/" + path
            
            # Write content
            mock_ipfs_mfs[path] = content.encode("utf-8")
            
            return {
                "path": path,
                "size": len(content),
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def ipfs_files_read(path: str) -> Dict[str, Any]:
        """Read from a file in IPFS MFS (Mutable File System)."""
        if not path:
            return {"error": "Path is required", "success": False}
        
        # Normalize path
        path = path.strip()
        if not path.startswith("/"):
            path = "/" + path
        
        if path not in mock_ipfs_mfs:
            return {"error": "Path not found", "success": False}
        
        try:
            content = mock_ipfs_mfs[path]
            return {
                "content": content.decode("utf-8", errors="replace"),
                "size": len(content),
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def ipfs_files_ls(path: str = "/") -> Dict[str, Any]:
        """List files and directories in IPFS MFS (Mutable File System)."""
        # Normalize path
        path = path.strip()
        if not path.startswith("/"):
            path = "/" + path
        
        try:
            entries = []
            for mfs_path in mock_ipfs_mfs:
                if mfs_path.startswith(path) and mfs_path != path:
                    # Get the relative path
                    rel_path = mfs_path[len(path):].lstrip("/")
                    if "/" in rel_path:
                        # This is a subdirectory
                        dir_name = rel_path.split("/")[0]
                        if not any(e["name"] == dir_name for e in entries):
                            entries.append({"name": dir_name, "type": "directory"})
                    else:
                        # This is a file
                        entries.append({"name": rel_path, "type": "file", "size": len(mock_ipfs_mfs[mfs_path])})
            
            return {
                "entries": entries,
                "path": path,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def ipfs_pin_add(cid: str) -> Dict[str, Any]:
        """Pin content in IPFS to prevent garbage collection."""
        if not cid:
            return {"error": "CID is required", "success": False}
        
        if cid not in mock_ipfs_data:
            return {"error": "CID not found", "success": False}
        
        mock_ipfs_pins.add(cid)
        
        return {
            "cid": cid,
            "success": True
        }
    
    def ipfs_pin_rm(cid: str) -> Dict[str, Any]:
        """Unpin content in IPFS."""
        if not cid:
            return {"error": "CID is required", "success": False}
        
        if cid not in mock_ipfs_pins:
            return {"error": "CID not pinned", "success": False}
        
        mock_ipfs_pins.remove(cid)
        
        return {
            "cid": cid,
            "success": True
        }
    
    def ipfs_pin_ls() -> Dict[str, Any]:
        """List pinned content in IPFS."""
        return {
            "pins": list(mock_ipfs_pins),
            "count": len(mock_ipfs_pins),
            "success": True
        }
    
    def ipfs_node_info() -> Dict[str, Any]:
        """Get information about the IPFS node."""
        return {
            "id": "QmMockIPFSNode",
            "version": "0.1.0",
            "protocol_version": "ipfs/0.1.0",
            "agent_version": "mock-ipfs/0.1.0",
            "success": True
        }
    
    # Register all the tools
    register_tool(
        name="ipfs_add_file",
        description="Add a file to IPFS",
        function=ipfs_add_file,
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["path"]
        }
    )
    
    register_tool(
        name="ipfs_cat",
        description="Read the contents of a file from IPFS",
        function=ipfs_cat,
        schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "CID of the file to read"}
            },
            "required": ["cid"]
        }
    )
    
    register_tool(
        name="ipfs_get_file",
        description="Download a file from IPFS",
        function=ipfs_get_file,
        schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "CID of the file to download"},
                "output_path": {"type": "string", "description": "Path where to save the file"}
            },
            "required": ["cid", "output_path"]
        }
    )
    
    register_tool(
        name="ipfs_files_write",
        description="Write to a file in IPFS MFS (Mutable File System)",
        function=ipfs_files_write,
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path in MFS to write to"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        }
    )
    
    register_tool(
        name="ipfs_files_read",
        description="Read from a file in IPFS MFS (Mutable File System)",
        function=ipfs_files_read,
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path in MFS to read from"}
            },
            "required": ["path"]
        }
    )
    
    register_tool(
        name="ipfs_files_ls",
        description="List files and directories in IPFS MFS (Mutable File System)",
        function=ipfs_files_ls,
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path in MFS to list"}
            },
            "required": []
        }
    )
    
    register_tool(
        name="ipfs_pin_add",
        description="Pin content in IPFS to prevent garbage collection",
        function=ipfs_pin_add,
        schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "CID to pin"}
            },
            "required": ["cid"]
        }
    )
    
    register_tool(
        name="ipfs_pin_rm",
        description="Unpin content in IPFS",
        function=ipfs_pin_rm,
        schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "CID to unpin"}
            },
            "required": ["cid"]
        }
    )
    
    register_tool(
        name="ipfs_pin_ls",
        description="List pinned content in IPFS",
        function=ipfs_pin_ls,
        schema={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
    
    register_tool(
        name="ipfs_node_info",
        description="Get information about the IPFS node",
        function=ipfs_node_info,
        schema={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
    
    # Start the server
    logger.info(f"Starting MCP server on {args.host}:{args.port}...")
    start_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
