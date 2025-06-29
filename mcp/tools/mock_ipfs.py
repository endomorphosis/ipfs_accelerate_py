"""
Mock implementation of the IPFS client.

This module provides a mock implementation of the IPFS client interface
for use when the actual ipfs-kit-py package is not available. This enables
testing and development without the full IPFS infrastructure.
"""

import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)


class MockIPFSClient:
    """Mock implementation of the IPFS client.
    
    This class simulates the behavior of an IPFS client for testing and
    development when the actual IPFS client is not available.
    """
    
    def __init__(self):
        """Initialize a new mock IPFS client."""
        self.files = {}  # Simulated MFS
        self.blocks = {}  # Simulated blockstore
        self.pins = set()  # Simulated pinset
        self.peers = []  # Simulated peers
        logger.info("Initialized mock IPFS client")
    
    def version(self) -> Dict[str, str]:
        """Get the IPFS version.
        
        Returns:
            Dictionary with version information
        """
        return {
            "Version": "0.1.0-mock",
            "Commit": "mock",
            "Repo": "0",
            "System": "mock",
            "Golang": "mock"
        }
    
    def add(self, path: str) -> Dict[str, Any]:
        """Add a file to IPFS.
        
        Args:
            path: Path to the file to add
            
        Returns:
            Dictionary with information about the added file
            
        Raises:
            FileNotFoundError: If the file is not found
        """
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Get file size
        size = os.path.getsize(path)
        
        # Generate a simulated CID
        cid = f"QmMock{random.randint(1000000, 9999999)}"
        
        # Store in simulated blockstore
        with open(path, "rb") as f:
            content = f.read()
            self.blocks[cid] = content
        
        # Return information
        return {
            "Name": os.path.basename(path),
            "Hash": cid,
            "Size": size,
            "success": True
        }
    
    def cat(self, cid: str) -> bytes:
        """Get the content of a file from IPFS.
        
        Args:
            cid: CID of the file to get
            
        Returns:
            File content
            
        Raises:
            KeyError: If the CID is not found
        """
        # Check if CID exists in simulated blockstore
        if cid in self.blocks:
            return self.blocks[cid]
        
        # Generate simulated content for testing
        content = f"Simulated content for CID: {cid}".encode("utf-8")
        self.blocks[cid] = content
        return content
    
    def ls(self, cid: str) -> List[Dict[str, Any]]:
        """List links from an IPFS object.
        
        Args:
            cid: CID of the object to list
            
        Returns:
            List of links
        """
        # Generate some simulated links for testing
        return [
            {"Hash": f"QmMock{random.randint(1000000, 9999999)}", "Name": f"file{i}.txt", "Size": random.randint(100, 10000), "Type": 2}
            for i in range(5)
        ]
    
    def pin_add(self, cid: str) -> List[str]:
        """Pin an IPFS object.
        
        Args:
            cid: CID of the object to pin
            
        Returns:
            List of pinned CIDs
        """
        # Add to simulated pinset
        self.pins.add(cid)
        return [cid]
    
    def pin_rm(self, cid: str) -> List[str]:
        """Unpin an IPFS object.
        
        Args:
            cid: CID of the object to unpin
            
        Returns:
            List of unpinned CIDs
        """
        # Remove from simulated pinset
        if cid in self.pins:
            self.pins.remove(cid)
        return [cid]
    
    def pin_ls(self, cid: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """List pinned objects.
        
        Args:
            cid: CID to filter by
            
        Returns:
            Dictionary of pinned objects
        """
        # Return simulated pinset
        if cid:
            return {cid: {"Type": "recursive"}} if cid in self.pins else {}
        
        return {pin: {"Type": "recursive"} for pin in self.pins}
    
    # MFS (Mutable File System) operations
    
    def files_mkdir(self, path: str, parents: bool = False) -> None:
        """Create a directory in the MFS.
        
        Args:
            path: Path to create
            parents: Whether to create parent directories
            
        Raises:
            FileExistsError: If the directory already exists
            FileNotFoundError: If a parent directory doesn't exist
        """
        # Normalize path
        path = path.rstrip("/")
        
        # Check if directory already exists
        if path in self.files:
            raise FileExistsError(f"Directory already exists: {path}")
        
        # Create parent directories if needed
        if parents:
            parts = path.split("/")
            for i in range(1, len(parts)):
                parent = "/".join(parts[:i])
                if parent and parent not in self.files:
                    self.files[parent] = {"type": "directory", "children": {}}
        
        # Create directory
        self.files[path] = {"type": "directory", "children": {}}
    
    def files_write(self, path: str, content: Union[str, bytes], create: bool = False) -> Dict[str, Any]:
        """Write to a file in the MFS.
        
        Args:
            path: Path to write to
            content: Content to write
            create: Whether to create the file if it doesn't exist
            
        Returns:
            Dictionary with information about the write operation
            
        Raises:
            FileNotFoundError: If the parent directory doesn't exist
        """
        # Convert content to bytes if it's a string
        if isinstance(content, str):
            content = content.encode("utf-8")
        
        # Generate a simulated CID
        cid = f"QmMock{random.randint(1000000, 9999999)}"
        
        # Write to simulated MFS
        self.files[path] = {"type": "file", "content": content, "cid": cid}
        
        # Return information
        return {
            "path": path,
            "written": True,
            "size": len(content),
            "cid": cid
        }
    
    def files_read(self, path: str) -> bytes:
        """Read a file from the MFS.
        
        Args:
            path: Path to read from
            
        Returns:
            File content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IsADirectoryError: If the path is a directory
        """
        # Check if file exists
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        
        # Check if path is a directory
        if self.files[path]["type"] == "directory":
            raise IsADirectoryError(f"Cannot read a directory: {path}")
        
        # Return content
        return self.files[path]["content"]
    
    def files_ls(self, path: str) -> Dict[str, Any]:
        """List files in the MFS.
        
        Args:
            path: Path to list
            
        Returns:
            Dictionary with directory contents
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
        """
        # Default to root
        if not path or path == "/":
            path = "/"
            
            # Create root if it doesn't exist
            if path not in self.files:
                self.files[path] = {"type": "directory", "children": {}}
                
            # Generate simulated root listing
            entries = []
            for p in self.files:
                if p != "/" and not p.startswith("//") and "/" not in p[1:]:
                    name = p[1:] if p.startswith("/") else p
                    entries.append({
                        "Name": name,
                        "Type": 1 if self.files[p]["type"] == "directory" else 0,
                        "Size": 0 if self.files[p]["type"] == "directory" else len(self.files[p]["content"]),
                        "Hash": "QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn" if self.files[p]["type"] == "directory" else self.files[p]["cid"]
                    })
            
            return {"Entries": entries}
        
        # Check if directory exists
        if path not in self.files:
            raise FileNotFoundError(f"Directory not found: {path}")
        
        # Check if path is a directory
        if self.files[path]["type"] != "directory":
            raise NotADirectoryError(f"Not a directory: {path}")
        
        # Generate simulated listing
        entries = []
        for p in self.files:
            if p.startswith(path + "/") and "/" not in p[len(path) + 1:]:
                name = p[len(path) + 1:]
                entries.append({
                    "Name": name,
                    "Type": 1 if self.files[p]["type"] == "directory" else 0,
                    "Size": 0 if self.files[p]["type"] == "directory" else len(self.files[p]["content"]),
                    "Hash": "QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn" if self.files[p]["type"] == "directory" else self.files[p]["cid"]
                })
        
        return {"Entries": entries}
    
    def files_rm(self, path: str, recursive: bool = False) -> None:
        """Remove a file or directory from the MFS.
        
        Args:
            path: Path to remove
            recursive: Whether to remove recursively
            
        Raises:
            FileNotFoundError: If the path doesn't exist
            OSError: If trying to remove a non-empty directory without recursive
        """
        # Check if path exists
        if path not in self.files:
            raise FileNotFoundError(f"Path not found: {path}")
        
        # Check if removing a non-empty directory without recursive
        if self.files[path]["type"] == "directory" and not recursive:
            for p in list(self.files.keys()):
                if p.startswith(path + "/"):
                    raise OSError(f"Directory not empty: {path}")
        
        # Remove path and children
        to_remove = [p for p in self.files if p == path or (recursive and p.startswith(path + "/"))]
        for p in to_remove:
            del self.files[p]
    
    def files_stat(self, path: str) -> Dict[str, Any]:
        """Get information about a file or directory in the MFS.
        
        Args:
            path: Path to get information for
            
        Returns:
            Dictionary with information
            
        Raises:
            FileNotFoundError: If the path doesn't exist
        """
        # Check if path exists
        if path not in self.files:
            raise FileNotFoundError(f"Path not found: {path}")
        
        # Return information
        file_info = self.files[path]
        return {
            "Hash": file_info.get("cid", "QmUNLLsPACCz1vLxQVkXqqLX5R1X345qqfHbsf67hvA3Nn"),
            "Size": len(file_info.get("content", b"")) if file_info["type"] == "file" else 0,
            "Type": "directory" if file_info["type"] == "directory" else "file",
            "Blocks": 1 if file_info["type"] == "file" else 0
        }


def get_mock_ipfs_client() -> MockIPFSClient:
    """Get a mock IPFS client instance.
    
    Returns:
        Mock IPFS client
    """
    return MockIPFSClient()
