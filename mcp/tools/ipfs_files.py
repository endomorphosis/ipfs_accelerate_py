"""
IPFS file operations tools for the MCP server.

This module provides tools that expose IPFS file operations to LLM clients,
including adding, reading, listing, and pinning files.
"""

import os
import json
import anyio
import base64
import logging
from typing import Dict, Any, List, Optional, Union, cast

# Configure logging
logger = logging.getLogger(__name__)


def _is_pytest() -> bool:
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


def _log_optional_dependency(message: str) -> None:
    if _is_pytest():
        logger.info(message)
    else:
        logger.warning(message)

# Try imports with fallbacks
try:
    if _is_pytest():
        raise ImportError("Using mock MCP under pytest")
    from fastmcp import FastMCP, Context
except ImportError:
    try:
        from mcp.mock_mcp import FastMCP, Context
    except ImportError:
        from mock_mcp import FastMCP, Context

# Import shared operations
try:
    from shared import SharedCore, FileOperations
    shared_core = SharedCore()
    file_ops = FileOperations(shared_core)
    HAVE_SHARED = True
except ImportError:
    try:
        from ...shared import SharedCore, FileOperations
        shared_core = SharedCore()
        file_ops = FileOperations(shared_core)
        HAVE_SHARED = True
    except ImportError as e:
        _log_optional_dependency(f"Shared operations not available: {e}")
        HAVE_SHARED = False
        shared_core = None
        file_ops = None

# Import the get_ipfs_client function from tools module for fallback
try:
    from . import get_ipfs_client
except ImportError:
    from tools import get_ipfs_client


async def get_ipfs_client_async(ctx: Context) -> Any:
    """Get IPFS client from context or create a new one if not available."""
    try:
        # Try to get from lifespan context
        if hasattr(ctx.request_context, 'lifespan_context') and \
           hasattr(ctx.request_context.lifespan_context, 'ipfs_context') and \
           hasattr(ctx.request_context.lifespan_context.ipfs_context, 'ipfs_client') and \
           ctx.request_context.lifespan_context.ipfs_context.ipfs_client is not None:
            return ctx.request_context.lifespan_context.ipfs_context.ipfs_client
    except (AttributeError, TypeError):
        await ctx.info("IPFS client not found in context, creating new client")
    
    # Create a new client if not available
    return get_ipfs_client()


def register_files_tools(mcp: FastMCP) -> None:
    """Register IPFS file operation tools with the MCP server.
    
    Args:
        mcp: The FastMCP server instance to register tools with
    """
    
    @mcp.tool()
    async def add_file_shared(path: str, ctx: Context, pin: bool = True) -> Dict[str, Any]:
        """Add a file to IPFS using shared operations.
        
        Args:
            path: Path to the file to add
            ctx: MCP context
            pin: Whether to pin the file
            
        Returns:
            Dictionary with result information
        """
        await ctx.info(f"Adding file using shared operations: {path}")
        
        try:
            # Use shared operations if available
            if HAVE_SHARED and file_ops:
                result = file_ops.add_file(path, pin=pin)
                await ctx.info(f"File added via shared operations: {result.get('cid', 'no CID')}")
                return result
            else:
                await ctx.error("Shared operations not available")
                return {
                    "error": "Shared operations not available",
                    "path": path,
                    "success": False,
                    "fallback_needed": True
                }
                
        except Exception as e:
            logger.error(f"Error in add_file_shared: {str(e)}")
            await ctx.error(f"Failed to add file: {str(e)}")
            return {
                "error": str(e),
                "path": path,
                "success": False
            }
    
    @mcp.tool()
    async def ipfs_add_file(path: str, ctx: Context, wrap_with_directory: bool = False) -> Dict[str, Any]:
        """Add a file to IPFS.
        
        Args:
            path: Path to the file to add
            ctx: MCP context
            wrap_with_directory: Whether to wrap the file in a directory
            
        Returns:
            Dictionary with CID and size information
        """
        await ctx.info(f"Adding file: {path}")
        await ctx.report_progress(0, 1)
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client_async(ctx)
            
            # Check if file exists
            if not os.path.exists(path):
                await ctx.error(f"File not found: {path}")
                return {
                    "error": "File not found",
                    "path": path,
                    "success": False
                }
            
            # Get file size for progress reporting
            file_size = os.path.getsize(path)
            file_name = os.path.basename(path)
            
            # Add file to IPFS
            await ctx.info(f"Adding file ({file_size} bytes): {path}")
            await ctx.report_progress(0.2, 1)
            
            # Use ipfs client to add the file
            result = await anyio.to_thread.run_sync(
                ipfs.add_file,
                path,
                wrap_with_directory=wrap_with_directory
            )
            
            await ctx.report_progress(1, 1)
            await ctx.info(f"File added: {result['Hash']}")
            
            return {
                "cid": result["Hash"],
                "size": file_size,
                "name": file_name,
                "wrapped": wrap_with_directory,
                "success": True
            }
        except Exception as e:
            await ctx.error(f"Error adding file: {str(e)}")
            return {
                "error": str(e),
                "path": path,
                "success": False
            }
    
    @mcp.tool()
    async def ipfs_cat(cid: str, ctx: Context, offset: int = 0, length: int = -1) -> str:
        """Read a file from IPFS.
        
        Args:
            cid: Content identifier of the file to read
            ctx: MCP context
            offset: Byte offset to start reading from
            length: Maximum number of bytes to read (-1 for all)
            
        Returns:
            File content as a string
        """
        await ctx.info(f"Reading content with CID: {cid}")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client_async(ctx)
            
            # Use ipfs client to read the file
            content = await anyio.to_thread.run_sync(
                ipfs.cat,
                cid,
                offset=offset,
                length=length
            )
            
            if isinstance(content, bytes):
                try:
                    # Try to decode as UTF-8
                    result = content.decode('utf-8')
                except UnicodeDecodeError:
                    # If it's not a valid UTF-8 string, return a base64 representation
                    result = f"[Binary data, base64 encoded]: {base64.b64encode(content).decode('ascii')}"
            else:
                result = str(content)
                
            return result
        except Exception as e:
            await ctx.error(f"Error reading content: {str(e)}")
            return f"Error reading {cid}: {str(e)}"
    
    @mcp.tool()
    async def ipfs_ls(cid: str, ctx: Context) -> List[Dict[str, Any]]:
        """List the contents of an IPFS directory.
        
        Args:
            cid: Content identifier of the directory to list
            ctx: MCP context
            
        Returns:
            List of files and directories within the specified path
        """
        await ctx.info(f"Listing contents of: {cid}")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client_async(ctx)
            
            # Use ipfs client to list the directory
            result = await anyio.to_thread.run_sync(ipfs.ls, cid)
            
            # Process the result into a standardized format
            entries = []
            for entry in result["Objects"][0]["Links"]:
                entry_type = "directory" if entry["Type"] == 1 else "file"
                entries.append({
                    "name": entry["Name"],
                    "type": entry_type,
                    "size": entry["Size"],
                    "cid": entry["Hash"]
                })
            
            return entries
        except Exception as e:
            await ctx.error(f"Error listing directory: {str(e)}")
            return [{"error": str(e)}]
    
    @mcp.tool()
    async def ipfs_mkdir(path: str, ctx: Context) -> Dict[str, Any]:
        """Create a directory in the IPFS MFS (Mutable File System).
        
        Args:
            path: MFS path to create
            ctx: MCP context
            
        Returns:
            Information about the created directory
        """
        await ctx.info(f"Creating directory: {path}")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client_async(ctx)
            
            # Use ipfs client to create a directory in MFS
            await anyio.to_thread.run_sync(ipfs.files_mkdir, path, parents=True)
            
            # Get info about the created directory
            stat_result = await anyio.to_thread.run_sync(ipfs.files_stat, path)
            
            return {
                "path": path,
                "created": True,
                "cid": stat_result["Hash"],
                "size": stat_result["Size"]
            }
        except Exception as e:
            await ctx.error(f"Error creating directory: {str(e)}")
            return {
                "path": path,
                "created": False,
                "error": str(e)
            }
    
    @mcp.tool()
    async def ipfs_pin_add(cid: str, ctx: Context, recursive: bool = True) -> Dict[str, Any]:
        """Pin content in the local IPFS repository.
        
        Pinning ensures content is not garbage collected and remains available locally.
        
        Args:
            cid: Content identifier to pin
            ctx: MCP context
            recursive: Whether to recursively pin the entire DAG
            
        Returns:
            Information about the pinning operation
        """
        await ctx.info(f"Pinning content: {cid} (recursive={recursive})")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client_async(ctx)
            
            # Use ipfs client to pin the content
            await anyio.to_thread.run_sync(ipfs.pin_add, cid, recursive=recursive)
            
            # Get info about the pinned content
            pin_ls = await anyio.to_thread.run_sync(ipfs.pin_ls, cid)
            
            return {
                "cid": cid,
                "pinned": True,
                "recursive": recursive,
                "type": pin_ls["Keys"][cid]["Type"] if cid in pin_ls.get("Keys", {}) else "unknown"
            }
        except Exception as e:
            await ctx.error(f"Error pinning content: {str(e)}")
            return {
                "cid": cid,
                "pinned": False,
                "error": str(e)
            }
    
    @mcp.tool()
    async def ipfs_pin_rm(cid: str, ctx: Context, recursive: bool = True) -> Dict[str, Any]:
        """Unpin content from the local IPFS repository.
        
        This allows the content to be garbage collected if no longer needed.
        
        Args:
            cid: Content identifier to unpin
            ctx: MCP context
            recursive: Whether to recursively unpin
            
        Returns:
            Information about the unpinning operation
        """
        await ctx.info(f"Unpinning content: {cid} (recursive={recursive})")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client_async(ctx)
            
            # Check if the content is pinned
            try:
                pin_ls = await anyio.to_thread.run_sync(ipfs.pin_ls, cid)
                if cid not in pin_ls.get("Keys", {}):
                    return {
                        "cid": cid,
                        "unpinned": False,
                        "error": "Content is not pinned"
                    }
            except Exception:
                # If pin_ls fails, content is likely not pinned
                return {
                    "cid": cid,
                    "unpinned": False,
                    "error": "Content is not pinned or does not exist"
                }
            
            # Use ipfs client to unpin the content
            await anyio.to_thread.run_sync(ipfs.pin_rm, cid, recursive=recursive)
            
            return {
                "cid": cid,
                "unpinned": True,
                "recursive": recursive
            }
        except Exception as e:
            await ctx.error(f"Error unpinning content: {str(e)}")
            return {
                "cid": cid,
                "unpinned": False,
                "error": str(e)
            }
            
    @mcp.tool()
    async def ipfs_files_write(path: str, content: str, ctx: Context, create: bool = True, truncate: bool = True) -> Dict[str, Any]:
        """Write content to a file in the IPFS MFS.
        
        Args:
            path: MFS path to write to
            content: Content to write
            ctx: MCP context
            create: Whether to create the file if it doesn't exist
            truncate: Whether to truncate the file if it exists
            
        Returns:
            Information about the write operation
        """
        await ctx.info(f"Writing to file: {path}")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client_async(ctx)
            
            # Convert content to bytes
            content_bytes = content.encode('utf-8')
            
            # Use ipfs client to write to the file
            await anyio.to_thread.run_sync(
                ipfs.files_write,
                path,
                content_bytes,
                create=create,
                truncate=truncate
            )
            
            # Get info about the written file
            stat_result = await anyio.to_thread.run_sync(ipfs.files_stat, path)
            
            return {
                "path": path,
                "written": True,
                "size": stat_result["Size"],
                "cid": stat_result["Hash"]
            }
        except Exception as e:
            await ctx.error(f"Error writing to file: {str(e)}")
            return {
                "path": path,
                "written": False,
                "error": str(e)
            }
            
    @mcp.tool()
    async def ipfs_files_read(path: str, ctx: Context, offset: int = 0, count: int = -1) -> str:
        """Read a file from the IPFS MFS.
        
        Args:
            path: MFS path to read
            ctx: MCP context
            offset: Byte offset to start reading from
            count: Maximum number of bytes to read (-1 for all)
            
        Returns:
            File content as a string
        """
        await ctx.info(f"Reading file: {path}")
        
        try:
            # Get IPFS client
            ipfs = await get_ipfs_client_async(ctx)
            
            # Use ipfs client to read the file
            content = await anyio.to_thread.run_sync(
                ipfs.files_read,
                path,
                offset=offset,
                count=count
            )
            
            if isinstance(content, bytes):
                try:
                    # Try to decode as UTF-8
                    result = content.decode('utf-8')
                except UnicodeDecodeError:
                    # If it's not a valid UTF-8 string, return a base64 representation
                    result = f"[Binary data, base64 encoded]: {base64.b64encode(content).decode('ascii')}"
            else:
                result = str(content)
                
            return result
        except Exception as e:
            await ctx.error(f"Error reading file: {str(e)}")
            return f"Error reading {path}: {str(e)}"


if __name__ == "__main__":
    # This can be used for standalone testing
    import os
    import sys
    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from server import create_ipfs_mcp_server
    
    async def test_tools():
        mcp = create_ipfs_mcp_server("IPFS File Tools Test")
        register_files_tools(mcp)
        # Implement test code here if needed
    
    anyio.run(test_tools)
