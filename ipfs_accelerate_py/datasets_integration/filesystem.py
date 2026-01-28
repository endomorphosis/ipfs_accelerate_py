"""
Filesystem Handler - IPFS-based filesystem operations

This module provides a unified interface for filesystem operations using
ipfs_datasets_py's UnixFS handler, with fallbacks to standard filesystem.
"""

import os
import shutil
from typing import Optional, List, Dict, Any, Union
from pathlib import Path


class FilesystemHandler:
    """
    Handler for filesystem operations with IPFS integration.
    
    Provides a unified interface that uses ipfs_datasets_py's UnixFSHandler
    when available, falling back to standard filesystem operations otherwise.
    
    This enables local-first, decentralized file operations while maintaining
    compatibility with CI/CD environments where IPFS may not be available.
    
    Attributes:
        enabled (bool): Whether IPFS integration is active
        unixfs_handler: UnixFSHandler instance (if available)
        cache_dir (Path): Local cache directory
    
    Example:
        >>> fs = FilesystemHandler()
        >>> # Works with or without IPFS
        >>> cid = fs.add_file("/path/to/model.bin")
        >>> fs.get_file(cid, "/path/to/output.bin")
    """
    
    def __init__(self, cache_dir: Optional[str] = None, ipfs_api: Optional[str] = None):
        """
        Initialize the filesystem handler.
        
        Args:
            cache_dir: Directory for local caching (default: ~/.cache/ipfs_accelerate/files)
            ipfs_api: IPFS API endpoint (default: /ip4/127.0.0.1/tcp/5001)
        """
        self.enabled = False
        self.unixfs_handler = None
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.cache' / 'ipfs_accelerate' / 'files'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to initialize UnixFS handler
        self._initialize(ipfs_api)
    
    def _initialize(self, ipfs_api: Optional[str] = None):
        """Initialize UnixFS handler if ipfs_datasets_py is available."""
        try:
            from ipfs_datasets_py.unixfs_integration import UnixFSHandler
            
            self.unixfs_handler = UnixFSHandler(
                api_endpoint=ipfs_api or '/ip4/127.0.0.1/tcp/5001'
            )
            self.enabled = True
            
        except (ImportError, Exception):
            # IPFS not available - will use local filesystem fallback
            self.enabled = False
    
    def add_file(self, file_path: str, pin: bool = True) -> Optional[str]:
        """
        Add a file to IPFS and return its CID.
        
        Args:
            file_path: Path to file to add
            pin: Whether to pin the file in IPFS
        
        Returns:
            Optional[str]: CID of added file, or None if IPFS unavailable
        
        Example:
            >>> cid = fs.add_file("/path/to/model.bin")
            >>> print(f"Model stored at: {cid}")
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.enabled and self.unixfs_handler:
            try:
                result = self.unixfs_handler.add_file(file_path)
                if pin:
                    self.unixfs_handler.pin(result['Hash'])
                return result['Hash']
            except Exception:
                pass
        
        # Fallback: Copy to cache and return local path
        file_name = os.path.basename(file_path)
        cache_path = self.cache_dir / file_name
        shutil.copy2(file_path, cache_path)
        return None
    
    def add_directory(self, dir_path: str, recursive: bool = True, 
                     pin: bool = True) -> Optional[str]:
        """
        Add a directory to IPFS and return its CID.
        
        Args:
            dir_path: Path to directory to add
            recursive: Whether to add recursively
            pin: Whether to pin the directory in IPFS
        
        Returns:
            Optional[str]: CID of added directory, or None if IPFS unavailable
        
        Example:
            >>> cid = fs.add_directory("/path/to/dataset/")
            >>> print(f"Dataset stored at: {cid}")
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")
        
        if self.enabled and self.unixfs_handler:
            try:
                result = self.unixfs_handler.add_directory(dir_path, recursive=recursive)
                if pin:
                    self.unixfs_handler.pin(result['Hash'])
                return result['Hash']
            except Exception:
                pass
        
        # Fallback: Copy to cache
        dir_name = os.path.basename(dir_path.rstrip('/'))
        cache_path = self.cache_dir / dir_name
        if cache_path.exists():
            shutil.rmtree(cache_path)
        shutil.copytree(dir_path, cache_path)
        return None
    
    def get_file(self, cid: str, output_path: str) -> bool:
        """
        Retrieve a file from IPFS by CID.
        
        Args:
            cid: Content identifier of the file
            output_path: Path where file should be saved
        
        Returns:
            bool: True if retrieved successfully, False otherwise
        
        Example:
            >>> success = fs.get_file("Qm...", "/path/to/output.bin")
        """
        if self.enabled and self.unixfs_handler and cid:
            try:
                self.unixfs_handler.get_file(cid, output_path)
                return True
            except Exception:
                pass
        
        # Fallback: Try to find in cache
        if not cid:
            return False
        
        cache_files = list(self.cache_dir.glob('*'))
        # In fallback mode, we don't have CIDs, so we can't retrieve by CID
        return False
    
    def get_directory(self, cid: str, output_path: str) -> bool:
        """
        Retrieve a directory from IPFS by CID.
        
        Args:
            cid: Content identifier of the directory
            output_path: Path where directory should be saved
        
        Returns:
            bool: True if retrieved successfully, False otherwise
        
        Example:
            >>> success = fs.get_directory("Qm...", "/path/to/output/")
        """
        if self.enabled and self.unixfs_handler and cid:
            try:
                self.unixfs_handler.get_directory(cid, output_path)
                return True
            except Exception:
                pass
        
        return False
    
    def list_directory(self, cid: str) -> List[Dict[str, Any]]:
        """
        List contents of an IPFS directory.
        
        Args:
            cid: Content identifier of the directory
        
        Returns:
            List of entries with 'name', 'type', 'cid', and 'size'
        
        Example:
            >>> entries = fs.list_directory("Qm...")
            >>> for entry in entries:
            ...     print(f"{entry['name']}: {entry['cid']}")
        """
        if self.enabled and self.unixfs_handler and cid:
            try:
                return self.unixfs_handler.list_directory(cid)
            except Exception:
                pass
        
        return []
    
    def pin(self, cid: str) -> bool:
        """
        Pin content in IPFS to prevent garbage collection.
        
        Args:
            cid: Content identifier to pin
        
        Returns:
            bool: True if pinned successfully, False otherwise
        
        Example:
            >>> fs.pin("Qm...")
        """
        if self.enabled and self.unixfs_handler and cid:
            try:
                self.unixfs_handler.pin(cid)
                return True
            except Exception:
                pass
        
        return False
    
    def unpin(self, cid: str) -> bool:
        """
        Unpin content in IPFS to allow garbage collection.
        
        Args:
            cid: Content identifier to unpin
        
        Returns:
            bool: True if unpinned successfully, False otherwise
        
        Example:
            >>> fs.unpin("Qm...")
        """
        if self.enabled and self.unixfs_handler and cid:
            try:
                self.unixfs_handler.unpin(cid)
                return True
            except Exception:
                pass
        
        return False
    
    def cat(self, cid: str) -> Optional[bytes]:
        """
        Read file contents from IPFS.
        
        Args:
            cid: Content identifier of the file
        
        Returns:
            Optional[bytes]: File contents, or None if unavailable
        
        Example:
            >>> content = fs.cat("Qm...")
            >>> print(content.decode('utf-8'))
        """
        if self.enabled and self.unixfs_handler and cid:
            try:
                return self.unixfs_handler.cat(cid)
            except Exception:
                pass
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of filesystem handler.
        
        Returns:
            Dict with status information
        
        Example:
            >>> status = fs.get_status()
            >>> print(f"IPFS enabled: {status['ipfs_enabled']}")
        """
        return {
            'ipfs_enabled': self.enabled,
            'cache_dir': str(self.cache_dir),
            'unixfs_handler': self.unixfs_handler is not None,
        }
