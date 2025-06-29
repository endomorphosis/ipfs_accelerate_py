"""
IPFS Operations Tool for MCP

This module provides IPFS-specific operations tools for the MCP server.
"""

import os
import sys
import json
import logging
import subprocess
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ipfs_accelerate_mcp.tools.ipfs")

def add_file_to_ipfs(file_path: str, wrap_with_directory: bool = False) -> Dict[str, Any]:
    """
    Add a file to IPFS
    
    Args:
        file_path: Path to the file to add
        wrap_with_directory: Whether to wrap the file in a directory
        
    Returns:
        Dictionary containing the hash and other metadata
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Build the command
        cmd = ["ipfs", "add", "-Q"]
        
        if wrap_with_directory:
            cmd.append("--wrap-with-directory")
        
        cmd.append(file_path)
        
        # Run the command
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get the hash
        ipfs_hash = result.stdout.strip()
        
        # Get file stats
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        return {
            "hash": ipfs_hash,
            "size": file_size,
            "name": file_name,
            "path": file_path
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error adding file to IPFS: {e}")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Error adding file to IPFS: {e.stderr}")
    
    except Exception as e:
        logger.error(f"Error adding file to IPFS: {e}")
        raise

def get_file_from_ipfs(ipfs_hash: str, output_path: str) -> Dict[str, Any]:
    """
    Get a file from IPFS
    
    Args:
        ipfs_hash: IPFS hash of the file
        output_path: Path to save the file to
        
    Returns:
        Dictionary containing the save path and other metadata
    """
    try:
        # Build the command
        cmd = ["ipfs", "get", "-o", output_path, ipfs_hash]
        
        # Run the command
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get file stats
        file_size = os.path.getsize(output_path)
        file_name = os.path.basename(output_path)
        
        return {
            "hash": ipfs_hash,
            "size": file_size,
            "name": file_name,
            "path": output_path
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting file from IPFS: {e}")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Error getting file from IPFS: {e.stderr}")
    
    except Exception as e:
        logger.error(f"Error getting file from IPFS: {e}")
        raise

def cat_ipfs_file(ipfs_hash: str, offset: int = 0, length: int = -1) -> str:
    """
    Cat a file from IPFS
    
    Args:
        ipfs_hash: IPFS hash of the file
        offset: Offset to start reading from
        length: Maximum number of bytes to read (-1 for all)
        
    Returns:
        File contents as a string
    """
    try:
        # Build the command
        cmd = ["ipfs", "cat"]
        
        if offset > 0 or length > 0:
            # Add offset and length arguments if specified
            range_arg = f"{offset}:{length if length > 0 else ''}"
            cmd.extend(["--range", range_arg])
        
        cmd.append(ipfs_hash)
        
        # Run the command
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the contents
        return result.stdout
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error catting file from IPFS: {e}")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Error catting file from IPFS: {e.stderr}")
    
    except Exception as e:
        logger.error(f"Error catting file from IPFS: {e}")
        raise

def pin_ipfs(ipfs_hash: str) -> Dict[str, Any]:
    """
    Pin a file on IPFS
    
    Args:
        ipfs_hash: IPFS hash of the file to pin
        
    Returns:
        Dictionary containing the pinned hash and status
    """
    try:
        # Build the command
        cmd = ["ipfs", "pin", "add", ipfs_hash]
        
        # Run the command
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        output = result.stdout.strip()
        
        return {
            "hash": ipfs_hash,
            "pinned": True,
            "output": output
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pinning file on IPFS: {e}")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Error pinning file on IPFS: {e.stderr}")
    
    except Exception as e:
        logger.error(f"Error pinning file on IPFS: {e}")
        raise

def unpin_ipfs(ipfs_hash: str) -> Dict[str, Any]:
    """
    Unpin a file from IPFS
    
    Args:
        ipfs_hash: IPFS hash of the file to unpin
        
    Returns:
        Dictionary containing the unpinned hash and status
    """
    try:
        # Build the command
        cmd = ["ipfs", "pin", "rm", ipfs_hash]
        
        # Run the command
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        output = result.stdout.strip()
        
        return {
            "hash": ipfs_hash,
            "unpinned": True,
            "output": output
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error unpinning file from IPFS: {e}")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Error unpinning file from IPFS: {e.stderr}")
    
    except Exception as e:
        logger.error(f"Error unpinning file from IPFS: {e}")
        raise

def get_ipfs_node_info() -> Dict[str, Any]:
    """
    Get information about the IPFS node
    
    Returns:
        Dictionary containing node information
    """
    try:
        # Build the command
        cmd = ["ipfs", "id", "--format=json"]
        
        # Run the command
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output as JSON
        node_info = json.loads(result.stdout.strip())
        
        return node_info
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting IPFS node info: {e}")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Error getting IPFS node info: {e.stderr}")
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing IPFS node info: {e}")
        raise RuntimeError(f"Error parsing IPFS node info: {e}")
    
    except Exception as e:
        logger.error(f"Error getting IPFS node info: {e}")
        raise

def ipfs_gateway_url(ipfs_hash: str, gateway: str = "https://ipfs.io") -> str:
    """
    Get an HTTP URL for an IPFS hash
    
    Args:
        ipfs_hash: IPFS hash
        gateway: IPFS gateway to use
        
    Returns:
        HTTP URL for the IPFS hash
    """
    # Remove any 'ipfs://' prefix
    if ipfs_hash.startswith("ipfs://"):
        ipfs_hash = ipfs_hash[7:]
    
    # Construct the URL
    return f"{gateway}/ipfs/{ipfs_hash}"

def register_with_mcp(mcp):
    """
    Register IPFS tools with the MCP server
    
    Args:
        mcp: MCP server instance
    """
    logger.info("Registering IPFS tools with MCP server")
    
    # Register add_file_to_ipfs
    mcp.register_tool(
        name="ipfs_add_file",
        function=add_file_to_ipfs,
        description="Add a file to IPFS",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to add"
                },
                "wrap_with_directory": {
                    "type": "boolean",
                    "description": "Whether to wrap the file in a directory",
                    "default": False
                }
            },
            "required": ["file_path"]
        }
    )
    
    # Register get_file_from_ipfs
    mcp.register_tool(
        name="ipfs_get_file",
        function=get_file_from_ipfs,
        description="Get a file from IPFS",
        input_schema={
            "type": "object",
            "properties": {
                "ipfs_hash": {
                    "type": "string",
                    "description": "IPFS hash of the file"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save the file to"
                }
            },
            "required": ["ipfs_hash", "output_path"]
        }
    )
    
    # Register cat_ipfs_file
    mcp.register_tool(
        name="ipfs_cat_file",
        function=cat_ipfs_file,
        description="Cat a file from IPFS",
        input_schema={
            "type": "object",
            "properties": {
                "ipfs_hash": {
                    "type": "string",
                    "description": "IPFS hash of the file"
                },
                "offset": {
                    "type": "integer",
                    "description": "Offset to start reading from",
                    "default": 0
                },
                "length": {
                    "type": "integer",
                    "description": "Maximum number of bytes to read (-1 for all)",
                    "default": -1
                }
            },
            "required": ["ipfs_hash"]
        }
    )
    
    # Register pin_ipfs
    mcp.register_tool(
        name="ipfs_pin",
        function=pin_ipfs,
        description="Pin a file on IPFS",
        input_schema={
            "type": "object",
            "properties": {
                "ipfs_hash": {
                    "type": "string",
                    "description": "IPFS hash of the file to pin"
                }
            },
            "required": ["ipfs_hash"]
        }
    )
    
    # Register unpin_ipfs
    mcp.register_tool(
        name="ipfs_unpin",
        function=unpin_ipfs,
        description="Unpin a file from IPFS",
        input_schema={
            "type": "object",
            "properties": {
                "ipfs_hash": {
                    "type": "string",
                    "description": "IPFS hash of the file to unpin"
                }
            },
            "required": ["ipfs_hash"]
        }
    )
    
    # Register get_ipfs_node_info
    mcp.register_tool(
        name="ipfs_node_info",
        function=get_ipfs_node_info,
        description="Get information about the IPFS node",
        input_schema={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
    
    # Register ipfs_gateway_url
    mcp.register_tool(
        name="ipfs_gateway_url",
        function=ipfs_gateway_url,
        description="Get an HTTP URL for an IPFS hash",
        input_schema={
            "type": "object",
            "properties": {
                "ipfs_hash": {
                    "type": "string",
                    "description": "IPFS hash"
                },
                "gateway": {
                    "type": "string",
                    "description": "IPFS gateway to use",
                    "default": "https://ipfs.io"
                }
            },
            "required": ["ipfs_hash"]
        }
    )
    
    logger.info("IPFS tools registered with MCP server")
