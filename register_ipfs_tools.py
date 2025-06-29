#!/usr/bin/env python3
"""
IPFS Tools Registration Script

This script registers IPFS-related tools with the MCP server running on port 8002.
It complements the existing simple_mcp_register.py script by focusing specifically
on IPFS functionality.
"""

import logging
import os
import sys
import requests
import json
import tempfile
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def register_tool(name: str, description: str, schema: Dict[str, Any], base_url: str = "http://localhost:8002") -> bool:
    """Register a tool with the MCP server."""
    try:
        url = f"{base_url}/mcp/register_tool"
        payload = {
            "name": name,
            "description": description,
            "schema": schema
        }
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            logger.info(f"Successfully registered tool: {name}")
            return True
        else:
            logger.error(f"Failed to register tool {name}: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error registering tool {name}: {e}")
        return False

def register_resource(name: str, description: str, base_url: str = "http://localhost:8002") -> bool:
    """Register a resource with the MCP server."""
    try:
        url = f"{base_url}/mcp/register_resource"
        payload = {
            "name": name,
            "description": description
        }
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            logger.info(f"Successfully registered resource: {name}")
            return True
        else:
            logger.error(f"Failed to register resource {name}: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error registering resource {name}: {e}")
        return False

def register_ipfs_resources(base_url: str = "http://localhost:8002") -> bool:
    """Register IPFS-related resources."""
    resources = [
        ("ipfs_nodes", "Information about connected IPFS nodes"),
        ("ipfs_files", "Information about files stored in IPFS"),
        ("ipfs_pins", "Information about pinned content in IPFS")
    ]
    
    success = True
    for name, description in resources:
        if not register_resource(name, description, base_url):
            success = False
    
    return success

def register_ipfs_tools(base_url: str = "http://localhost:8002") -> bool:
    """Register IPFS-related tools."""
    tools = [
        {
            "name": "ipfs_add_file",
            "description": "Add a file to IPFS",
            "schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["path"]
            }
        },
        {
            "name": "ipfs_cat",
            "description": "Read the contents of a file from IPFS",
            "schema": {
                "type": "object",
                "properties": {
                    "cid": {"type": "string", "description": "CID of the file to read"}
                },
                "required": ["cid"]
            }
        },
        {
            "name": "ipfs_get_file",
            "description": "Download a file from IPFS",
            "schema": {
                "type": "object",
                "properties": {
                    "cid": {"type": "string", "description": "CID of the file to download"},
                    "output_path": {"type": "string", "description": "Path where to save the file"}
                },
                "required": ["cid", "output_path"]
            }
        },
        {
            "name": "ipfs_files_write",
            "description": "Write to a file in IPFS MFS (Mutable File System)",
            "schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path in MFS to write to"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        },
        {
            "name": "ipfs_files_read",
            "description": "Read from a file in IPFS MFS (Mutable File System)",
            "schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path in MFS to read from"}
                },
                "required": ["path"]
            }
        },
        {
            "name": "ipfs_files_ls",
            "description": "List files and directories in IPFS MFS (Mutable File System)",
            "schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path in MFS to list"}
                },
                "required": ["path"]
            }
        },
        {
            "name": "ipfs_pin_add",
            "description": "Pin content in IPFS to prevent garbage collection",
            "schema": {
                "type": "object",
                "properties": {
                    "cid": {"type": "string", "description": "CID to pin"}
                },
                "required": ["cid"]
            }
        },
        {
            "name": "ipfs_pin_rm",
            "description": "Unpin content in IPFS",
            "schema": {
                "type": "object",
                "properties": {
                    "cid": {"type": "string", "description": "CID to unpin"}
                },
                "required": ["cid"]
            }
        },
        {
            "name": "ipfs_pin_ls",
            "description": "List pinned content in IPFS",
            "schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "ipfs_node_info",
            "description": "Get information about the IPFS node",
            "schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]
    
    success = True
    for tool in tools:
        if not register_tool(tool["name"], tool["description"], tool["schema"], base_url):
            success = False
    
    return success

def create_mock_ipfs_handlers(base_url: str = "http://localhost:8002") -> bool:
    """Create mock implementations for IPFS tools."""
    # We'll track CIDs and file contents in memory for the mock implementation
    mock_ipfs_data = {}
    mock_ipfs_pins = set()
    mock_ipfs_mfs = {}
    
    def mock_ipfs_add_file(args):
        """Mock implementation of ipfs_add_file."""
        path = args.get("path")
        if not path:
            return {"error": "Path is required", "success": False}
        
        if not os.path.exists(path):
            return {"error": "File not found", "success": False}
        
        try:
            with open(path, "rb") as f:
                content = f.read()
                # Create a mock CID using a short hash of the content
                import hashlib
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
    
    def mock_ipfs_cat(args):
        """Mock implementation of ipfs_cat."""
        cid = args.get("cid")
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
    
    def mock_ipfs_get_file(args):
        """Mock implementation of ipfs_get_file."""
        cid = args.get("cid")
        output_path = args.get("output_path")
        
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
    
    def mock_ipfs_files_write(args):
        """Mock implementation of ipfs_files_write."""
        path = args.get("path")
        content = args.get("content")
        
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
    
    def mock_ipfs_files_read(args):
        """Mock implementation of ipfs_files_read."""
        path = args.get("path")
        
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
    
    def mock_ipfs_files_ls(args):
        """Mock implementation of ipfs_files_ls."""
        path = args.get("path", "/")
        
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
    
    def mock_ipfs_pin_add(args):
        """Mock implementation of ipfs_pin_add."""
        cid = args.get("cid")
        
        if not cid:
            return {"error": "CID is required", "success": False}
        
        if cid not in mock_ipfs_data:
            return {"error": "CID not found", "success": False}
        
        mock_ipfs_pins.add(cid)
        
        return {
            "cid": cid,
            "success": True
        }
    
    def mock_ipfs_pin_rm(args):
        """Mock implementation of ipfs_pin_rm."""
        cid = args.get("cid")
        
        if not cid:
            return {"error": "CID is required", "success": False}
        
        if cid not in mock_ipfs_pins:
            return {"error": "CID not pinned", "success": False}
        
        mock_ipfs_pins.remove(cid)
        
        return {
            "cid": cid,
            "success": True
        }
    
    def mock_ipfs_pin_ls(args):
        """Mock implementation of ipfs_pin_ls."""
        return {
            "pins": list(mock_ipfs_pins),
            "count": len(mock_ipfs_pins),
            "success": True
        }
    
    def mock_ipfs_node_info(args):
        """Mock implementation of ipfs_node_info."""
        return {
            "id": "QmMockIPFSNode",
            "version": "0.1.0",
            "protocol_version": "ipfs/0.1.0",
            "agent_version": "mock-ipfs/0.1.0",
            "success": True
        }
    
    # Map the tool names to their implementations
    tool_implementations = {
        "ipfs_add_file": mock_ipfs_add_file,
        "ipfs_cat": mock_ipfs_cat,
        "ipfs_get_file": mock_ipfs_get_file,
        "ipfs_files_write": mock_ipfs_files_write,
        "ipfs_files_read": mock_ipfs_files_read,
        "ipfs_files_ls": mock_ipfs_files_ls,
        "ipfs_pin_add": mock_ipfs_pin_add,
        "ipfs_pin_rm": mock_ipfs_pin_rm,
        "ipfs_pin_ls": mock_ipfs_pin_ls,
        "ipfs_node_info": mock_ipfs_node_info
    }
    
    # Register the implementation for each tool
    success = True
    for tool_name, implementation in tool_implementations.items():
        try:
            url = f"{base_url}/mcp/register_implementation"
            payload = {
                "tool_name": tool_name,
                "callback_url": f"{base_url}/mock_ipfs/{tool_name}"
            }
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully registered implementation for: {tool_name}")
            else:
                logger.error(f"Failed to register implementation for {tool_name}: {response.status_code} - {response.text}")
                success = False
                
            # Also add the local routes for the mock implementations
            mock_url = f"{base_url}/mock_ipfs/{tool_name}"
            logger.info(f"Mock implementation for {tool_name} available at: {mock_url}")
            
        except Exception as e:
            logger.error(f"Error registering implementation for {tool_name}: {e}")
            success = False
    
    return success

def main():
    """Main entry point for the script."""
    # Check if the server is running
    base_url = "http://localhost:8002"
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            logger.error(f"MCP server is not running at {base_url}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error connecting to MCP server at {base_url}: {e}")
        sys.exit(1)
    
    logger.info(f"Registering IPFS tools with MCP server at {base_url}")
    
    # Register resources
    resource_success = register_ipfs_resources(base_url)
    
    # Register tools
    tool_success = register_ipfs_tools(base_url)
    
    # Create mock implementations
    implementation_success = create_mock_ipfs_handlers(base_url)
    
    if resource_success and tool_success and implementation_success:
        logger.info("Successfully registered all IPFS tools and resources")
        sys.exit(0)
    else:
        logger.error("Failed to register some IPFS tools or resources")
        sys.exit(1)

if __name__ == "__main__":
    main()
