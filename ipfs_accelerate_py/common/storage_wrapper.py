"""
Storage Wrapper for AI Inference Filesystem Operations

This module provides a unified wrapper around the ipfs_kit_integration storage
that can be easily integrated into existing code patterns. It provides:

1. Drop-in replacement for common filesystem operations
2. Automatic gating based on environment variables
3. Fallback to standard filesystem operations
4. Context-aware behavior (CI/CD detection)

Usage:
    from ipfs_accelerate_py.common.storage_wrapper import StorageWrapper
    
    # Initialize (auto-detects CI/CD)
    storage = StorageWrapper()
    
    # Use like normal filesystem, but content-addressed
    cid = storage.write_file(data, "model_weights.bin")
    data = storage.read_file(cid)
    
    # Or use Path-like interface
    path = storage.get_path("model_weights.bin")  # Returns Path or CID
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Union, Optional, Any, Dict, List

logger = logging.getLogger(__name__)


class StorageWrapper:
    """
    Wrapper around ipfs_kit_integration that provides easy integration
    into existing filesystem operations.
    
    Features:
    - Auto-detects CI/CD environment
    - Configurable via environment variables
    - Transparent fallback to filesystem
    - Minimal code changes required
    """
    
    # Environment variables for control
    ENV_DISABLE = "IPFS_KIT_DISABLE"
    ENV_FORCE_LOCAL = "STORAGE_FORCE_LOCAL"
    ENV_CI = "CI"  # Standard CI/CD indicator
    
    def __init__(
        self,
        enable_distributed: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        auto_detect_ci: bool = True
    ):
        """
        Initialize storage wrapper.
        
        Args:
            enable_distributed: Whether to use distributed storage (None = auto-detect)
            cache_dir: Cache directory (None = use default)
            auto_detect_ci: Automatically disable in CI/CD environments
        """
        self._storage = None
        self._use_distributed = False
        self._cache_dir = cache_dir
        
        # Determine if we should use distributed storage
        if enable_distributed is None:
            # Auto-detect based on environment
            self._use_distributed = self._should_use_distributed(auto_detect_ci)
        else:
            self._use_distributed = enable_distributed
        
        # Try to initialize ipfs_kit_integration if enabled
        if self._use_distributed:
            try:
                from ..ipfs_kit_integration import get_storage
                self._storage = get_storage(
                    enable_ipfs_kit=True,
                    cache_dir=cache_dir,
                    force_fallback=False
                )
                
                # Check if it actually loaded
                if self._storage.using_fallback:
                    logger.info("Storage initialized in fallback mode (local filesystem)")
                    self._use_distributed = False
                else:
                    logger.info("Storage initialized with distributed backend")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed storage: {e}")
                logger.info("Falling back to standard filesystem operations")
                self._use_distributed = False
        
        if not self._use_distributed:
            logger.info("Using standard filesystem operations (distributed storage disabled)")
    
    def _should_use_distributed(self, auto_detect_ci: bool) -> bool:
        """
        Determine if distributed storage should be used based on environment.
        
        Args:
            auto_detect_ci: Whether to auto-detect CI/CD environment
        
        Returns:
            True if distributed storage should be used
        """
        # Check explicit disable flags
        if os.environ.get(self.ENV_DISABLE, '').lower() in ('1', 'true', 'yes'):
            logger.debug(f"{self.ENV_DISABLE}=1, disabling distributed storage")
            return False
        
        if os.environ.get(self.ENV_FORCE_LOCAL, '').lower() in ('1', 'true', 'yes'):
            logger.debug(f"{self.ENV_FORCE_LOCAL}=1, using local filesystem")
            return False
        
        # Auto-detect CI/CD environment
        if auto_detect_ci and os.environ.get(self.ENV_CI, ''):
            logger.debug("CI environment detected, using local filesystem")
            return False
        
        # Default to enabled
        return True
    
    @property
    def is_distributed(self) -> bool:
        """Check if using distributed storage."""
        return self._use_distributed and self._storage is not None
    
    @property
    def backend_status(self) -> Dict[str, Any]:
        """Get backend status information."""
        if self._storage:
            return self._storage.get_backend_status()
        return {
            'ipfs_kit_available': False,
            'using_fallback': True,
            'mode': 'filesystem',
            'cache_dir': self._cache_dir or 'default'
        }
    
    def write_file(
        self,
        data: Union[bytes, str],
        filename: Optional[str] = None,
        pin: bool = False
    ) -> str:
        """
        Write data to storage (distributed or local).
        
        Args:
            data: Data to write
            filename: Optional filename hint
            pin: Whether to pin content (for distributed storage)
        
        Returns:
            CID if using distributed storage, or file path if local
        """
        if self._storage and self._use_distributed:
            # Use distributed storage
            return self._storage.store(data, filename=filename, pin=pin)
        else:
            # Fallback to local filesystem
            return self._write_file_local(data, filename)
    
    def _write_file_local(
        self,
        data: Union[bytes, str],
        filename: Optional[str]
    ) -> str:
        """Write to local filesystem as fallback."""
        if filename is None:
            # Generate a temporary filename
            import hashlib
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            hash_val = hashlib.sha256(data_bytes).hexdigest()[:16]
            filename = f"storage_{hash_val}.dat"
        
        # Determine storage location
        if self._cache_dir:
            cache_path = Path(self._cache_dir).expanduser()
        else:
            cache_path = Path.home() / ".cache" / "ipfs_accelerate"
        
        cache_path.mkdir(parents=True, exist_ok=True)
        file_path = cache_path / filename
        
        # Write data
        if isinstance(data, str):
            file_path.write_text(data)
        else:
            file_path.write_bytes(data)
        
        return str(file_path)
    
    def read_file(self, identifier: str) -> Optional[bytes]:
        """
        Read data from storage (distributed or local).
        
        Args:
            identifier: CID (if distributed) or file path (if local)
        
        Returns:
            Data as bytes, or None if not found
        """
        if self._storage and self._use_distributed:
            # Try distributed storage first
            data = self._storage.retrieve(identifier)
            if data is not None:
                return data
        
        # Fallback to reading as file path
        return self._read_file_local(identifier)
    
    def _read_file_local(self, path: str) -> Optional[bytes]:
        """Read from local filesystem as fallback."""
        try:
            path_obj = Path(path)
            if path_obj.exists():
                return path_obj.read_bytes()
        except Exception as e:
            logger.debug(f"Failed to read local file {path}: {e}")
        
        return None
    
    def exists(self, identifier: str) -> bool:
        """
        Check if content exists.
        
        Args:
            identifier: CID or file path
        
        Returns:
            True if exists
        """
        if self._storage and self._use_distributed:
            if self._storage.exists(identifier):
                return True
        
        # Check local filesystem
        return Path(identifier).exists()
    
    def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """
        List files in storage.
        
        Args:
            path: Path to list (for distributed storage)
        
        Returns:
            List of file information
        """
        if self._storage and self._use_distributed:
            return self._storage.list_files(path)
        
        # For local filesystem, return empty list (not implemented)
        return []
    
    def delete(self, identifier: str) -> bool:
        """
        Delete content.
        
        Args:
            identifier: CID or file path
        
        Returns:
            True if deleted successfully
        """
        if self._storage and self._use_distributed:
            if self._storage.delete(identifier):
                return True
        
        # Try local filesystem
        try:
            path = Path(identifier)
            if path.exists():
                path.unlink()
                return True
        except Exception as e:
            logger.debug(f"Failed to delete {identifier}: {e}")
        
        return False
    
    def get_cache_dir(self) -> Path:
        """
        Get the cache directory being used.
        
        Returns:
            Path to cache directory
        """
        if self._storage and self._use_distributed:
            return self._storage.cache_dir
        
        if self._cache_dir:
            return Path(self._cache_dir).expanduser()
        
        return Path.home() / ".cache" / "ipfs_accelerate"
    
    def ensure_cache_dir(self) -> Path:
        """
        Ensure cache directory exists and return it.
        
        Returns:
            Path to cache directory
        """
        cache_dir = self.get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


# Singleton instance for easy access
_storage_wrapper: Optional[StorageWrapper] = None


def get_storage_wrapper(
    enable_distributed: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    auto_detect_ci: bool = True
) -> StorageWrapper:
    """
    Get or create the singleton StorageWrapper instance.
    
    Args:
        enable_distributed: Whether to use distributed storage (None = auto-detect)
        cache_dir: Cache directory (None = use default)
        auto_detect_ci: Automatically disable in CI/CD environments
    
    Returns:
        StorageWrapper instance
    """
    global _storage_wrapper
    
    if _storage_wrapper is None:
        _storage_wrapper = StorageWrapper(
            enable_distributed=enable_distributed,
            cache_dir=cache_dir,
            auto_detect_ci=auto_detect_ci
        )
    
    return _storage_wrapper


def reset_storage_wrapper():
    """Reset the singleton storage wrapper (useful for testing)."""
    global _storage_wrapper
    _storage_wrapper = None


# Constant to indicate storage_wrapper is available
# This is used by files that import with try/except fallback
HAVE_STORAGE_WRAPPER = True
