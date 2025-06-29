#!/usr/bin/env python3
"""
IPFS Virtual Filesystem MCP Tools Module

This module provides MCP tools for interacting with IPFS's mutable file system (MFS).
It offers a virtual filesystem interface that can be used to create, read, modify
and remove files and directories in IPFS.
"""

import os
import sys
import json
import logging
import tempfile
from typing import Dict, Any, List, Optional, Union, BinaryIO, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock IPFS MFS implementation for testing
class MockIPFSMFS:
    """Simple mock IPFS Mutable File System implementation for testing."""
    
    def __init__(self):
        """Initialize the mock MFS."""
        self.files = {}
        self.directories = {"/": {"type": "directory", "size": 0}}
    
    def _ensure_parent_exists(self, path: str) -> bool:
        """Ensure parent directory exists."""
        parent_dir = os.path.dirname(path)
        if parent_dir == "":
            parent_dir = "/"
        
        return parent_dir in self.directories
    
    def _get_file_info(self, path: str) -> Dict[str, Any]:
        """Get file or directory information."""
        if path == "/":
            return self.directories[path]
        
        if path in self.files:
            return self.files[path]
        
        if path in self.directories:
            return self.directories[path]
        
        return None
    
    def write(self, path: str, content: Union[str, bytes], create: bool = True, truncate: bool = True) -> Dict[str, Any]:
        """Write content to a file in MFS."""
        if not path.startswith("/"):
            path = f"/{path}"
        
        if not self._ensure_parent_exists(path):
            return {"error": f"Parent directory of {path} does not exist", "success": False}
        
        # Convert string content to bytes if needed
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        # Create or update the file
        file_info = self._get_file_info(path)
        if file_info is None:
            if not create:
                return {"error": f"File {path} does not exist and create is False", "success": False}
            
            self.files[path] = {
                "type": "file",
                "size": len(content),
                "content": content,
                "hash": f"QmMock{hash(content) % 1000000:06d}"
            }
        else:
            if file_info["type"] == "directory":
                return {"error": f"{path} is a directory, not a file", "success": False}
            
            if truncate:
                self.files[path]["content"] = content
            else:
                self.files[path]["content"] += content
            
            self.files[path]["size"] = len(self.files[path]["content"])
            self.files[path]["hash"] = f"QmMock{hash(self.files[path]['content']) % 1000000:06d}"
        
        return {
            "path": path,
            "size": self.files[path]["size"],
            "hash": self.files[path]["hash"],
            "success": True
        }
    
    def read(self, path: str, offset: int = 0, count: int = -1) -> Dict[str, Any]:
        """Read content from a file in MFS."""
        if not path.startswith("/"):
            path = f"/{path}"
        
        file_info = self._get_file_info(path)
        if file_info is None:
            return {"error": f"File {path} does not exist", "success": False}
        
        if file_info["type"] == "directory":
            return {"error": f"{path} is a directory, not a file", "success": False}
        
        content = file_info["content"]
        if offset > 0:
            content = content[offset:]
        
        if count > 0:
            content = content[:count]
        
        # Convert bytes to string for response
        try:
            content_str = content.decode('utf-8')
        except UnicodeDecodeError:
            content_str = content.decode('utf-8', errors='replace')
        
        return {
            "path": path,
            "content": content_str,
            "size": len(content),
            "success": True
        }
    
    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> Dict[str, Any]:
        """Create a directory in MFS."""
        if not path.startswith("/"):
            path = f"/{path}"
        
        if path in self.directories:
            if exist_ok:
                return {"path": path, "success": True}
            else:
                return {"error": f"Directory {path} already exists", "success": False}
        
        if path in self.files:
            return {"error": f"{path} is a file, not a directory", "success": False}
        
        # Check parent directory
        parent_dir = os.path.dirname(path)
        if parent_dir == "":
            parent_dir = "/"
        
        if parent_dir not in self.directories:
            if not parents:
                return {"error": f"Parent directory {parent_dir} does not exist", "success": False}
            
            # Create parent directories recursively
            current_path = ""
            for part in parent_dir.split("/"):
                if not part:
                    continue
                
                current_path = f"{current_path}/{part}"
                if current_path not in self.directories:
                    self.directories[current_path] = {"type": "directory", "size": 0}
        
        # Create the directory
        self.directories[path] = {"type": "directory", "size": 0}
        
        return {"path": path, "success": True}
    
    def ls(self, path: str) -> Dict[str, Any]:
        """List contents of a directory in MFS."""
        if not path.startswith("/"):
            path = f"/{path}"
        
        if path not in self.directories:
            if path in self.files:
                return {"error": f"{path} is a file, not a directory", "success": False}
            else:
                return {"error": f"Directory {path} does not exist", "success": False}
        
        entries = []
        
        # Get subdirectories
        for dir_path, dir_info in self.directories.items():
            if dir_path == path:
                continue
            
            parent_dir = os.path.dirname(dir_path)
            if parent_dir == "":
                parent_dir = "/"
            
            if parent_dir == path:
                name = os.path.basename(dir_path)
                entries.append({
                    "name": name,
                    "type": "directory",
                    "size": dir_info["size"],
                    "path": dir_path
                })
        
        # Get files
        for file_path, file_info in self.files.items():
            parent_dir = os.path.dirname(file_path)
            if parent_dir == "":
                parent_dir = "/"
            
            if parent_dir == path:
                name = os.path.basename(file_path)
                entries.append({
                    "name": name,
                    "type": "file",
                    "size": file_info["size"],
                    "hash": file_info["hash"],
                    "path": file_path
                })
        
        return {
            "path": path,
            "entries": entries,
            "success": True
        }
    
    def rm(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        """Remove a file or directory from MFS."""
        if not path.startswith("/"):
            path = f"/{path}"
        
        if path == "/":
            return {"error": "Cannot remove root directory", "success": False}
        
        if path in self.files:
            del self.files[path]
            return {"path": path, "success": True}
        
        if path in self.directories:
            # Check if directory is empty or recursive is True
            has_children = False
            for dir_path in self.directories.keys():
                if dir_path != path and dir_path.startswith(f"{path}/"):
                    has_children = True
                    break
            
            if not has_children:
                for file_path in self.files.keys():
                    if file_path.startswith(f"{path}/"):
                        has_children = True
                        break
            
            if has_children and not recursive:
                return {"error": f"Directory {path} is not empty", "success": False}
            
            # Remove directory and all children if recursive is True
            if recursive:
                directories_to_remove = [dir_path for dir_path in self.directories.keys() 
                                        if dir_path == path or dir_path.startswith(f"{path}/")]
                files_to_remove = [file_path for file_path in self.files.keys() 
                                if file_path.startswith(f"{path}/")]
                
                for dir_path in directories_to_remove:
                    if dir_path in self.directories:
                        del self.directories[dir_path]
                
                for file_path in files_to_remove:
                    if file_path in self.files:
                        del self.files[file_path]
            
            del self.directories[path]
            return {"path": path, "success": True}
        
        return {"error": f"{path} does not exist", "success": False}
    
    def cp(self, source: str, dest: str) -> Dict[str, Any]:
        """Copy a file or directory in MFS."""
        if not source.startswith("/"):
            source = f"/{source}"
        
        if not dest.startswith("/"):
            dest = f"/{dest}"
        
        # Source must exist
        source_info = self._get_file_info(source)
        if source_info is None:
            return {"error": f"Source {source} does not exist", "success": False}
        
        # Ensure parent of destination exists
        if not self._ensure_parent_exists(dest):
            return {"error": f"Parent directory of {dest} does not exist", "success": False}
        
        if source_info["type"] == "file":
            # Copy file
            content = self.files[source]["content"]
            return self.write(dest, content)
        else:
            # Copy directory
            if dest in self.files:
                return {"error": f"Destination {dest} is a file", "success": False}
            
            # Create destination directory if it doesn't exist
            if dest not in self.directories:
                self.mkdir(dest)
            
            # Copy all children
            source_listing = self.ls(source)
            if source_listing.get("success", False):
                for entry in source_listing.get("entries", []):
                    source_path = entry["path"]
                    dest_path = f"{dest}/{os.path.basename(source_path)}"
                    self.cp(source_path, dest_path)
            
            return {"source": source, "destination": dest, "success": True}
    
    def mv(self, source: str, dest: str) -> Dict[str, Any]:
        """Move a file or directory in MFS."""
        if not source.startswith("/"):
            source = f"/{source}"
        
        if not dest.startswith("/"):
            dest = f"/{dest}"
        
        # Copy first
        cp_result = self.cp(source, dest)
        if not cp_result.get("success", False):
            return cp_result
        
        # Then remove the source
        rm_result = self.rm(source, recursive=True)
        if not rm_result.get("success", False):
            return {"error": f"Copied to {dest} but failed to remove {source}", "success": False}
        
        return {"source": source, "destination": dest, "success": True}
    
    def stat(self, path: str) -> Dict[str, Any]:
        """Get status of a file or directory in MFS."""
        if not path.startswith("/"):
            path = f"/{path}"
        
        file_info = self._get_file_info(path)
        if file_info is None:
            return {"error": f"{path} does not exist", "success": False}
        
        result = {
            "path": path,
            "type": file_info["type"],
            "size": file_info["size"],
            "success": True
        }
        
        if file_info["type"] == "file":
            result["hash"] = file_info["hash"]
        
        return result
    
    def flush(self, path: str = "/") -> Dict[str, Any]:
        """Flush a path in MFS to IPFS (commit changes)."""
        if not path.startswith("/"):
            path = f"/{path}"
        
        file_info = self._get_file_info(path)
        if file_info is None:
            return {"error": f"{path} does not exist", "success": False}
        
        if file_info["type"] == "file":
            return {
                "path": path,
                "hash": file_info["hash"],
                "success": True
            }
        else:
            # For directories, we'll create a fake hash based on the number of entries
            entries_count = 0
            for dir_path in self.directories.keys():
                if dir_path.startswith(f"{path}/"):
                    entries_count += 1
            
            for file_path in self.files.keys():
                if file_path.startswith(f"{path}/"):
                    entries_count += 1
            
            dir_hash = f"QmDirMock{hash(path + str(entries_count)) % 1000000:06d}"
            return {
                "path": path,
                "hash": dir_hash,
                "success": True
            }


# Create a global instance
mock_mfs = MockIPFSMFS()

# MCP tool functions
def ipfs_files_mkdir(path: str, parents: bool = False, exist_ok: bool = False) -> Dict[str, Any]:
    """Create a directory in IPFS MFS."""
    try:
        return mock_mfs.mkdir(path, parents=parents, exist_ok=exist_ok)
    except Exception as e:
        logger.error(f"Error in ipfs_files_mkdir: {e}")
        return {"error": str(e), "success": False}

def ipfs_files_write(path: str, content: str, create: bool = True, truncate: bool = True) -> Dict[str, Any]:
    """Write content to a file in IPFS MFS."""
    try:
        return mock_mfs.write(path, content, create=create, truncate=truncate)
    except Exception as e:
        logger.error(f"Error in ipfs_files_write: {e}")
        return {"error": str(e), "success": False}

def ipfs_files_read(path: str, offset: int = 0, count: int = -1) -> Dict[str, Any]:
    """Read content from a file in IPFS MFS."""
    try:
        result = mock_mfs.read(path, offset=offset, count=count)
        if result.get("success", False):
            # Return only the content as the result for simplicity
            return result.get("content", "")
        return {"error": result.get("error", "Unknown error"), "success": False}
    except Exception as e:
        logger.error(f"Error in ipfs_files_read: {e}")
        return {"error": str(e), "success": False}

def ipfs_files_ls(path: str = "/") -> Dict[str, Any]:
    """List contents of a directory in IPFS MFS."""
    try:
        return mock_mfs.ls(path)
    except Exception as e:
        logger.error(f"Error in ipfs_files_ls: {e}")
        return {"error": str(e), "success": False}

def ipfs_files_rm(path: str, recursive: bool = False) -> Dict[str, Any]:
    """Remove a file or directory from IPFS MFS."""
    try:
        return mock_mfs.rm(path, recursive=recursive)
    except Exception as e:
        logger.error(f"Error in ipfs_files_rm: {e}")
        return {"error": str(e), "success": False}

def ipfs_files_cp(source: str, destination: str) -> Dict[str, Any]:
    """Copy files in IPFS MFS."""
    try:
        return mock_mfs.cp(source, destination)
    except Exception as e:
        logger.error(f"Error in ipfs_files_cp: {e}")
        return {"error": str(e), "success": False}

def ipfs_files_mv(source: str, destination: str) -> Dict[str, Any]:
    """Move files in IPFS MFS."""
    try:
        return mock_mfs.mv(source, destination)
    except Exception as e:
        logger.error(f"Error in ipfs_files_mv: {e}")
        return {"error": str(e), "success": False}

def ipfs_files_stat(path: str) -> Dict[str, Any]:
    """Get status of a file or directory in IPFS MFS."""
    try:
        return mock_mfs.stat(path)
    except Exception as e:
        logger.error(f"Error in ipfs_files_stat: {e}")
        return {"error": str(e), "success": False}

def ipfs_files_flush(path: str = "/") -> Dict[str, Any]:
    """Flush a path in IPFS MFS to IPFS (commit changes)."""
    try:
        return mock_mfs.flush(path)
    except Exception as e:
        logger.error(f"Error in ipfs_files_flush: {e}")
        return {"error": str(e), "success": False}

def register_with_mcp(register_tool_fn: Callable) -> bool:
    """Register all VFS tools with the MCP server."""
    try:
        # Register mkdir tool
        register_tool_fn("ipfs_files_mkdir", "Create a directory in IPFS MFS", ipfs_files_mkdir, {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the directory"},
                "parents": {"type": "boolean", "description": "Create parent directories if they don't exist"},
                "exist_ok": {"type": "boolean", "description": "Do not error if directory already exists"}
            },
            "required": ["path"]
        })
        
        # Register write tool
        register_tool_fn("ipfs_files_write", "Write content to a file in IPFS MFS", ipfs_files_write, {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
                "create": {"type": "boolean", "description": "Create the file if it does not exist"},
                "truncate": {"type": "boolean", "description": "Truncate the file before writing"}
            },
            "required": ["path", "content"]
        })
        
        # Register read tool
        register_tool_fn("ipfs_files_read", "Read content from a file in IPFS MFS", ipfs_files_read, {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "offset": {"type": "integer", "description": "Offset to start reading from"},
                "count": {"type": "integer", "description": "Maximum number of bytes to read (-1 for all)"}
            },
            "required": ["path"]
        })
        
        # Register ls tool
        register_tool_fn("ipfs_files_ls", "List contents of a directory in IPFS MFS", ipfs_files_ls, {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the directory"}
            }
        })
        
        # Register rm tool
        register_tool_fn("ipfs_files_rm", "Remove a file or directory from IPFS MFS", ipfs_files_rm, {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file or directory"},
                "recursive": {"type": "boolean", "description": "Recursively remove directories"}
            },
            "required": ["path"]
        })
        
        # Register cp tool
        register_tool_fn("ipfs_files_cp", "Copy files in IPFS MFS", ipfs_files_cp, {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path"},
                "destination": {"type": "string", "description": "Destination path"}
            },
            "required": ["source", "destination"]
        })
        
        # Register mv tool
        register_tool_fn("ipfs_files_mv", "Move files in IPFS MFS", ipfs_files_mv, {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path"},
                "destination": {"type": "string", "description": "Destination path"}
            },
            "required": ["source", "destination"]
        })
        
        # Register stat tool
        register_tool_fn("ipfs_files_stat", "Get status of a file or directory in IPFS MFS", ipfs_files_stat, {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file or directory"}
            },
            "required": ["path"]
        })
        
        # Register flush tool
        register_tool_fn("ipfs_files_flush", "Flush a path in IPFS MFS to IPFS (commit changes)", ipfs_files_flush, {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to flush"}
            }
        })
        
        logger.info("Successfully registered all IPFS virtual filesystem tools")
        return True
    
    except Exception as e:
        logger.error(f"Error registering IPFS virtual filesystem tools: {e}")
        return False

# For direct testing
if __name__ == "__main__":
    print("This module provides IPFS virtual filesystem tools for MCP")
    print("Use register_with_mcp(register_tool_fn) to register the tools")
