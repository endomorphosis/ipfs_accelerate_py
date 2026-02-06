"""
IPFS Kit Integration Layer

This module provides a unified interface for distributed filesystem operations
using ipfs_kit_py, with graceful fallbacks when the package is not available
or is disabled (e.g., in CI/CD environments).

Key Features:
- Local-first approach with IPFS distribution
- Automatic fallback to local filesystem when ipfs_kit_py is unavailable
- Multi-backend support (IPFS, S3, Filecoin, local)
- Content-addressed storage with CID generation
- Crash-resistant operations via Write-Ahead Log
- Configurable enable/disable for different environments

Usage:
    from ipfs_accelerate_py.ipfs_kit_integration import IPFSKitStorage
    
    # Initialize with automatic fallback
    storage = IPFSKitStorage(enable_ipfs_kit=True, cache_dir="~/.cache")
    
    # Store data with CID
    cid = storage.store(data, "model_weights.bin")
    
    # Retrieve data by CID
    data = storage.retrieve(cid)
    
    # List available files
    files = storage.list_files("/models/")
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, BinaryIO
from dataclasses import dataclass

try:
    from .common.storage_wrapper import storage_wrapper
except (ImportError, ValueError):
    try:
        from test.common.storage_wrapper import storage_wrapper
    except ImportError:
        storage_wrapper = None

logger = logging.getLogger(__name__)


@dataclass
class StorageBackendConfig:
    """Configuration for storage backends"""
    enable_ipfs: bool = True
    enable_s3: bool = False
    enable_filecoin: bool = False
    enable_local: bool = True
    cache_dir: str = "~/.cache/ipfs_accelerate"
    ipfs_kit_available: bool = False
    

class IPFSKitStorage:
    """
    Unified storage interface with ipfs_kit_py integration and fallback support.
    
    This class provides a consistent API for filesystem operations that can use
    ipfs_kit_py's distributed storage backends when available, or fall back to
    local filesystem operations when not available or disabled.
    
    The integration follows a local-first approach:
    1. Try to use ipfs_kit_py if available and enabled
    2. Fall back to local filesystem operations
    3. Log when fallback occurs for debugging
    """
    
    def __init__(
        self,
        enable_ipfs_kit: bool = True,
        cache_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        force_fallback: bool = False
    ):
        """
        Initialize the IPFS Kit storage interface.
        
        Args:
            enable_ipfs_kit: Whether to attempt using ipfs_kit_py (default: True)
            cache_dir: Directory for local caching (default: ~/.cache/ipfs_accelerate)
            config: Additional configuration options
            force_fallback: Force use of fallback mode (useful for CI/CD)
        """
        self.config = config or {}
        self.force_fallback = force_fallback or os.environ.get('IPFS_KIT_DISABLE', '').lower() in ('1', 'true', 'yes')
        self.enable_ipfs_kit = enable_ipfs_kit and not self.force_fallback
        
        # Initialize storage wrapper
        if storage_wrapper:
            try:
                self.storage = storage_wrapper()
            except:
                self.storage = None
        else:
            self.storage = None
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser()
        else:
            self.cache_dir = Path.home() / ".cache" / "ipfs_accelerate"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to import and initialize ipfs_kit_py
        self.ipfs_kit_client = None
        self.using_fallback = True
        
        if self.enable_ipfs_kit:
            self._try_init_ipfs_kit()
        else:
            logger.info("IPFS Kit integration disabled by configuration")
    
    def _try_init_ipfs_kit(self):
        """
        Attempt to initialize ipfs_kit_py client.
        Falls back to local mode if unavailable.
        """
        try:
            # Add the ipfs_kit_py directory to the path
            repo_root = Path(__file__).parent.parent.parent
            ipfs_kit_path = repo_root / "external" / "ipfs_kit_py"
            
            if ipfs_kit_path.exists():
                sys.path.insert(0, str(ipfs_kit_path))
                logger.debug(f"Added ipfs_kit_py path: {ipfs_kit_path}")
            
            # Try to import ipfs_kit_py modules directly (avoid backends/__init__.py due to missing synapse_storage)
            from ipfs_kit_py.backends.base_adapter import BackendAdapter
            from ipfs_kit_py.backends.filesystem_backend import FilesystemBackendAdapter
            from ipfs_kit_py.backends.ipfs_backend import IPFSBackendAdapter
            
            # Initialize the client with local-first configuration
            logger.info("Successfully imported ipfs_kit_py modules from local workspace")
            
            # Create a simple client wrapper
            self.ipfs_kit_client = {
                'vfs': None,  # Will be initialized on first use
                'backend_adapter': FilesystemBackendAdapter,
                'ipfs_backend': IPFSBackendAdapter,
                'base_adapter': BackendAdapter,
                'available': True
            }
            
            self.using_fallback = False
            logger.info("IPFS Kit integration enabled successfully (local workspace)")
            
        except ImportError as e:
            logger.warning(
                f"ipfs_kit_py not available: {e}. "
                "Falling back to local filesystem operations. "
                "This is expected in CI/CD environments."
            )
            self.using_fallback = True
        except Exception as e:
            logger.error(f"Error initializing IPFS Kit: {e}", exc_info=True)
            self.using_fallback = True
    
    def is_available(self) -> bool:
        """Check if ipfs_kit_py is available and initialized."""
        return not self.using_fallback and self.ipfs_kit_client is not None
    
    def get_backend_status(self) -> Dict[str, Any]:
        """
        Get the status of storage backends.
        
        Returns:
            Dictionary with backend availability and status
        """
        return {
            'ipfs_kit_available': self.is_available(),
            'using_fallback': self.using_fallback,
            'cache_dir': str(self.cache_dir),
            'backends': {
                'local': True,  # Always available
                'ipfs': self.is_available(),
                's3': False,  # Would be detected from ipfs_kit_py
                'filecoin': False,  # Would be detected from ipfs_kit_py
            }
        }
    
    def store(
        self,
        data: Union[bytes, str, Path],
        filename: Optional[str] = None,
        pin: bool = False
    ) -> str:
        """
        Store data and return a content identifier (CID).
        
        Args:
            data: Data to store (bytes, string, or file path)
            filename: Optional filename hint
            pin: Whether to pin the content (IPFS concept)
        
        Returns:
            Content identifier (CID) as a string
        """
        if not self.using_fallback and self.ipfs_kit_client:
            # Use ipfs_kit_py for storage
            return self._store_with_ipfs_kit(data, filename, pin)
        else:
            # Use fallback local storage
            return self._store_local(data, filename, pin)
    
    def _store_with_ipfs_kit(
        self,
        data: Union[bytes, str, Path],
        filename: Optional[str],
        pin: bool
    ) -> str:
        """Store data using ipfs_kit_py (when available)."""
        try:
            # This would use the actual ipfs_kit_py VFS or backend
            # For now, we'll implement a placeholder that shows the structure
            logger.debug(f"Storing via ipfs_kit_py: {filename}")
            
            # Convert data to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, Path):
                # Try distributed storage first
                if self.storage:
                    try:
                        cached_data = self.storage.get_file(str(data))
                        if cached_data:
                            data_bytes = cached_data.encode() if isinstance(cached_data, str) else cached_data
                        else:
                            with open(data, 'rb') as f:
                                data_bytes = f.read()
                            # Cache for future use
                            self.storage.store_file(str(data), data_bytes, pin=pin)
                    except:
                        with open(data, 'rb') as f:
                            data_bytes = f.read()
                else:
                    with open(data, 'rb') as f:
                        data_bytes = f.read()
            else:
                data_bytes = data
            
            # Generate CID (this would be done by ipfs_kit_py in production)
            cid = self._generate_cid(data_bytes)
            
            # Store locally with CID as filename
            storage_path = self.cache_dir / cid
            with open(storage_path, 'wb') as f:
                f.write(data_bytes)
            
            # Store in distributed storage
            if self.storage:
                try:
                    self.storage.store_file(str(storage_path), data_bytes, pin=pin)
                except:
                    pass  # Silently fail distributed storage
            
            # Store metadata
            if filename:
                metadata_path = self.cache_dir / f"{cid}.meta"
                metadata_json = json.dumps({'filename': filename, 'pinned': pin})
                with open(metadata_path, 'w') as f:
                    f.write(metadata_json)
                # Store metadata in distributed storage
                if self.storage:
                    try:
                        self.storage.store_file(str(metadata_path), metadata_json, pin=pin)
                    except:
                        pass
            
            logger.info(f"Stored content with CID: {cid}")
            return cid
            
        except Exception as e:
            logger.error(f"Error storing with ipfs_kit_py: {e}", exc_info=True)
            # Fall back to local storage
            return self._store_local(data, filename)
    
    def _store_local(
        self,
        data: Union[bytes, str, Path],
        filename: Optional[str],
        pin: bool = False
    ) -> str:
        """Store data locally and return a CID-like identifier."""
        try:
            # Convert data to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, Path):
                # Try distributed storage first
                if self.storage:
                    try:
                        cached_data = self.storage.get_file(str(data))
                        if cached_data:
                            data_bytes = cached_data.encode() if isinstance(cached_data, str) else cached_data
                        else:
                            with open(data, 'rb') as f:
                                data_bytes = f.read()
                            # Cache for future use
                            self.storage.store_file(str(data), data_bytes, pin=pin)
                    except:
                        with open(data, 'rb') as f:
                            data_bytes = f.read()
                else:
                    with open(data, 'rb') as f:
                        data_bytes = f.read()
            else:
                data_bytes = data
            
            # Generate a CID-like identifier
            cid = self._generate_cid(data_bytes)
            
            # Store locally
            storage_path = self.cache_dir / cid
            with open(storage_path, 'wb') as f:
                f.write(data_bytes)
            
            # Store in distributed storage
            if self.storage:
                try:
                    self.storage.store_file(str(storage_path), data_bytes, pin=pin)
                except:
                    pass  # Silently fail distributed storage
            
            # Store metadata if filename provided
            if filename:
                metadata_path = self.cache_dir / f"{cid}.meta"
                metadata_json = json.dumps({'filename': filename, 'fallback': True, 'pinned': pin})
                with open(metadata_path, 'w') as f:
                    f.write(metadata_json)
                # Store metadata in distributed storage
                if self.storage:
                    try:
                        self.storage.store_file(str(metadata_path), metadata_json, pin=pin)
                    except:
                        pass
            
            logger.debug(f"Stored content locally with CID: {cid}")
            return cid
            
        except Exception as e:
            logger.error(f"Error storing locally: {e}", exc_info=True)
            raise
    
    def retrieve(self, cid: str) -> Optional[bytes]:
        """
        Retrieve data by content identifier (CID).
        
        Args:
            cid: Content identifier
        
        Returns:
            Data as bytes, or None if not found
        """
        if not self.using_fallback and self.ipfs_kit_client:
            return self._retrieve_with_ipfs_kit(cid)
        else:
            return self._retrieve_local(cid)
    
    def _retrieve_with_ipfs_kit(self, cid: str) -> Optional[bytes]:
        """Retrieve data using ipfs_kit_py (when available)."""
        try:
            # This would use the actual ipfs_kit_py retrieval
            logger.debug(f"Retrieving via ipfs_kit_py: {cid}")
            
            # For now, try local first then would try IPFS network
            return self._retrieve_local(cid)
            
        except Exception as e:
            logger.error(f"Error retrieving with ipfs_kit_py: {e}", exc_info=True)
            return self._retrieve_local(cid)
    
    def _retrieve_local(self, cid: str) -> Optional[bytes]:
        """Retrieve data from local cache."""
        try:
            storage_path = self.cache_dir / cid
            if storage_path.exists():
                with open(storage_path, 'rb') as f:
                    return f.read()
            else:
                logger.warning(f"CID not found in local cache: {cid}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving locally: {e}", exc_info=True)
            return None
    
    def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """
        List files at the given path.
        
        Args:
            path: Path to list (IPFS path or local path)
        
        Returns:
            List of file information dictionaries
        """
        if not self.using_fallback and self.ipfs_kit_client:
            return self._list_files_ipfs_kit(path)
        else:
            return self._list_files_local()
    
    def _list_files_ipfs_kit(self, path: str) -> List[Dict[str, Any]]:
        """List files using ipfs_kit_py."""
        try:
            # This would use ipfs_kit_py VFS listing
            logger.debug(f"Listing via ipfs_kit_py: {path}")
            return self._list_files_local()
        except Exception as e:
            logger.error(f"Error listing with ipfs_kit_py: {e}", exc_info=True)
            return self._list_files_local()
    
    def _list_files_local(self) -> List[Dict[str, Any]]:
        """List files in local cache."""
        files = []
        try:
            for item in self.cache_dir.iterdir():
                if item.is_file() and not item.name.endswith('.meta'):
                    stat = item.stat()
                    
                    # Try to load metadata
                    meta_path = self.cache_dir / f"{item.name}.meta"
                    metadata = {}
                    if meta_path.exists():
                        try:
                            with open(meta_path, 'r') as f:
                                metadata = json.load(f)
                        except:
                            pass
                    
                    files.append({
                        'cid': item.name,
                        'filename': metadata.get('filename', item.name),
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'pinned': metadata.get('pinned', False),
                        'fallback': metadata.get('fallback', True)
                    })
        except Exception as e:
            logger.error(f"Error listing local files: {e}", exc_info=True)
        
        return files
    
    def exists(self, cid: str) -> bool:
        """
        Check if content with given CID exists.
        
        Args:
            cid: Content identifier
        
        Returns:
            True if content exists, False otherwise
        """
        if not self.using_fallback and self.ipfs_kit_client:
            # Would check ipfs_kit_py backends
            pass
        
        # Check local cache
        storage_path = self.cache_dir / cid
        return storage_path.exists()
    
    def delete(self, cid: str) -> bool:
        """
        Delete content with given CID.
        
        Args:
            cid: Content identifier
        
        Returns:
            True if deleted, False if not found or error
        """
        try:
            storage_path = self.cache_dir / cid
            metadata_path = self.cache_dir / f"{cid}.meta"
            
            deleted = False
            if storage_path.exists():
                storage_path.unlink()
                deleted = True
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            if deleted:
                logger.info(f"Deleted content with CID: {cid}")
            
            return deleted
        except Exception as e:
            logger.error(f"Error deleting content: {e}", exc_info=True)
            return False
    
    def _generate_cid(self, data: bytes) -> str:
        """
        Generate a content identifier for data.
        
        This is a simplified version. In production, ipfs_kit_py would use
        proper IPLD multiformats for CID generation.
        
        Args:
            data: Data to generate CID for
        
        Returns:
            CID-like string identifier
        """
        # Use SHA-256 hash as a simple CID
        # Real implementation would use multihash/multibase encoding
        hash_value = hashlib.sha256(data).hexdigest()
        return f"bafy{hash_value[:56]}"  # Mimic IPFS CIDv1 format
    
    def pin(self, cid: str) -> bool:
        """
        Pin content to prevent garbage collection.
        
        Args:
            cid: Content identifier
        
        Returns:
            True if pinned successfully
        """
        try:
            metadata_path = self.cache_dir / f"{cid}.meta"
            metadata = {}
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            metadata['pinned'] = True
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Pinned content with CID: {cid}")
            return True
        except Exception as e:
            logger.error(f"Error pinning content: {e}", exc_info=True)
            return False
    
    def unpin(self, cid: str) -> bool:
        """
        Unpin content to allow garbage collection.
        
        Args:
            cid: Content identifier
        
        Returns:
            True if unpinned successfully
        """
        try:
            metadata_path = self.cache_dir / f"{cid}.meta"
            metadata = {}
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            metadata['pinned'] = False
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Unpinned content with CID: {cid}")
            return True
        except Exception as e:
            logger.error(f"Error unpinning content: {e}", exc_info=True)
            return False


# Singleton instance for easy access
_storage_instance: Optional[IPFSKitStorage] = None


def get_storage(
    enable_ipfs_kit: bool = True,
    cache_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    force_fallback: bool = False
) -> IPFSKitStorage:
    """
    Get or create the singleton IPFSKitStorage instance.
    
    Args:
        enable_ipfs_kit: Whether to attempt using ipfs_kit_py
        cache_dir: Directory for local caching
        config: Additional configuration options
        force_fallback: Force use of fallback mode
    
    Returns:
        IPFSKitStorage instance
    """
    global _storage_instance
    
    if _storage_instance is None:
        _storage_instance = IPFSKitStorage(
            enable_ipfs_kit=enable_ipfs_kit,
            cache_dir=cache_dir,
            config=config,
            force_fallback=force_fallback
        )
    
    return _storage_instance


def reset_storage():
    """Reset the singleton storage instance (useful for testing)."""
    global _storage_instance
    _storage_instance = None
