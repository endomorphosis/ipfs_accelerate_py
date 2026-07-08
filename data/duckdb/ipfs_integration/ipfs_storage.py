"""
IPFS Storage Backend

This module provides IPFS storage capabilities for the DuckDB benchmark database.
It integrates with ipfs_datasets_py and ipfs_kit_py to enable decentralized storage
of database files, benchmark results, and model embeddings.
"""

import os
import sys
import json
import logging
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, BinaryIO
from datetime import datetime

logger = logging.getLogger(__name__)

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Try to import IPFS integration modules
try:
    from ipfs_datasets_py.ipfs_datasets_py.dataset_manager import DatasetManager
    HAVE_IPFS_DATASETS = True
except ImportError:
    HAVE_IPFS_DATASETS = False
    DatasetManager = None
    logger.warning("ipfs_datasets_py not available - dataset operations will be limited")

try:
    from ipfs_kit_py.ipfs_kit_py.ipfs_kit import ipfs_kit
    HAVE_IPFS_KIT = True
except ImportError:
    HAVE_IPFS_KIT = False
    ipfs_kit = None
    logger.warning("ipfs_kit_py not available - IPFS operations will be limited")

from .ipfs_config import IPFSConfig, get_ipfs_config


class IPFSStorageBackend:
    """
    IPFS storage backend for benchmark database files.
    
    This class provides methods to store and retrieve database files, benchmark results,
    and other data using IPFS, with local caching for performance.
    """
    
    def __init__(self, config: Optional[IPFSConfig] = None):
        """Initialize IPFS storage backend.
        
        Args:
            config: IPFS configuration (uses global config if not provided)
        """
        self.config = config or get_ipfs_config()
        self.cache_dir = Path(self.config.local_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize IPFS kit if available
        self.ipfs_kit = None
        if HAVE_IPFS_KIT and self.config.use_ipfs_kit:
            try:
                self.ipfs_kit = ipfs_kit(**self.config.ipfs_kit_config)
                logger.info("IPFS Kit initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize IPFS Kit: {e}")
        
        # Initialize dataset manager if available
        self.dataset_manager = None
        if HAVE_IPFS_DATASETS and self.config.use_ipfs_datasets:
            try:
                self.dataset_manager = DatasetManager(**self.config.ipfs_datasets_config)
                logger.info("Dataset Manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Dataset Manager: {e}")
    
    def is_available(self) -> bool:
        """Check if IPFS storage is available.
        
        Returns:
            True if IPFS storage backend is available and enabled
        """
        return self.config.is_ipfs_enabled() and (self.ipfs_kit is not None or self.dataset_manager is not None)
    
    def _get_cache_path(self, content_id: str) -> Path:
        """Get local cache path for content.
        
        Args:
            content_id: Content identifier (hash or CID)
            
        Returns:
            Path to cached file
        """
        # Use first 2 chars for subdirectory to avoid too many files in one dir
        subdir = content_id[:2] if len(content_id) >= 2 else "00"
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / content_id
    
    def _calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA256 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of file hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def store_file(self, file_path: Union[str, Path], pin: Optional[bool] = None) -> Dict[str, Any]:
        """Store a file on IPFS.
        
        Args:
            file_path: Path to file to store
            pin: Whether to pin the content (uses config default if None)
            
        Returns:
            Dictionary with storage metadata including CID
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        pin = pin if pin is not None else self.config.auto_pin
        file_hash = self._calculate_file_hash(file_path)
        
        result = {
            'success': False,
            'file_path': str(file_path),
            'file_hash': file_hash,
            'file_size': file_path.stat().st_size,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        # Try IPFS Kit first
        if self.ipfs_kit:
            try:
                # Add file to IPFS
                add_result = self.ipfs_kit.ipfs_py.add(str(file_path))
                if add_result.get('success'):
                    cid = add_result.get('Hash') or add_result.get('cid')
                    result['cid'] = cid
                    result['backend'] = 'ipfs_kit'
                    result['success'] = True
                    
                    # Pin if requested
                    if pin and cid:
                        pin_result = self.ipfs_kit.ipfs_py.pin_add(cid)
                        result['pinned'] = pin_result.get('success', False)
                    
                    # Cache locally
                    cache_path = self._get_cache_path(cid)
                    shutil.copy2(file_path, cache_path)
                    result['cached'] = True
                    
                    logger.info(f"Stored file on IPFS: {file_path} -> {cid}")
                    return result
            except Exception as e:
                logger.error(f"IPFS Kit storage failed: {e}")
                result['error'] = str(e)
        
        # Fallback: just cache locally
        if not result['success']:
            cache_path = self._get_cache_path(file_hash)
            shutil.copy2(file_path, cache_path)
            result['success'] = True
            result['cached'] = True
            result['backend'] = 'local_cache'
            result['cache_id'] = file_hash
            logger.warning(f"IPFS not available, file cached locally: {file_path}")
        
        return result
    
    def retrieve_file(self, content_id: str, destination: Union[str, Path]) -> Dict[str, Any]:
        """Retrieve a file from IPFS.
        
        Args:
            content_id: Content identifier (CID or hash)
            destination: Path to save the file
            
        Returns:
            Dictionary with retrieval metadata
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            'success': False,
            'content_id': content_id,
            'destination': str(destination),
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        # Check local cache first
        cache_path = self._get_cache_path(content_id)
        if cache_path.exists():
            shutil.copy2(cache_path, destination)
            result['success'] = True
            result['source'] = 'local_cache'
            result['file_size'] = destination.stat().st_size
            logger.info(f"Retrieved from cache: {content_id} -> {destination}")
            return result
        
        # Try to get from IPFS
        if self.ipfs_kit:
            try:
                get_result = self.ipfs_kit.ipfs_py.get(content_id, str(destination))
                if get_result.get('success'):
                    result['success'] = True
                    result['source'] = 'ipfs'
                    result['file_size'] = destination.stat().st_size
                    
                    # Cache for future use
                    shutil.copy2(destination, cache_path)
                    result['cached'] = True
                    
                    logger.info(f"Retrieved from IPFS: {content_id} -> {destination}")
                    return result
            except Exception as e:
                logger.error(f"IPFS retrieval failed: {e}")
                result['error'] = str(e)
        
        if not result['success']:
            logger.error(f"Failed to retrieve content: {content_id}")
        
        return result
    
    def store_benchmark_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store benchmark result data on IPFS.
        
        Args:
            result_data: Benchmark result dictionary
            
        Returns:
            Storage metadata including CID
        """
        # Create temporary file with JSON data
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(result_data, f, indent=2)
            temp_path = f.name
        
        try:
            storage_result = self.store_file(temp_path)
            storage_result['data_type'] = 'benchmark_result'
            storage_result['result_id'] = result_data.get('run_id') or result_data.get('id')
            return storage_result
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def retrieve_benchmark_result(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve benchmark result from IPFS.
        
        Args:
            content_id: Content identifier (CID)
            
        Returns:
            Benchmark result dictionary or None if failed
        """
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            retrieval_result = self.retrieve_file(content_id, temp_path)
            if retrieval_result['success']:
                with open(temp_path, 'r') as f:
                    return json.load(f)
            return None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def list_cached_files(self) -> List[Dict[str, Any]]:
        """List all files in local cache.
        
        Returns:
            List of cached file metadata
        """
        cached_files = []
        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file():
                cached_files.append({
                    'content_id': cache_file.name,
                    'path': str(cache_file),
                    'size': cache_file.stat().st_size,
                    'modified': datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat(),
                })
        return cached_files
    
    def clear_cache(self, max_age_days: Optional[int] = None) -> Dict[str, Any]:
        """Clear local cache.
        
        Args:
            max_age_days: Only clear files older than this many days (None = all)
            
        Returns:
            Dictionary with cleanup statistics
        """
        import time
        
        removed_count = 0
        removed_size = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file():
                if max_age_days is None:
                    # Remove all files
                    removed_size += cache_file.stat().st_size
                    cache_file.unlink()
                    removed_count += 1
                else:
                    # Check age
                    file_age_days = (current_time - cache_file.stat().st_mtime) / 86400
                    if file_age_days > max_age_days:
                        removed_size += cache_file.stat().st_size
                        cache_file.unlink()
                        removed_count += 1
        
        return {
            'success': True,
            'removed_files': removed_count,
            'removed_bytes': removed_size,
            'removed_mb': round(removed_size / (1024 * 1024), 2),
        }


class IPFSStorage:
    """
    High-level IPFS storage interface.
    
    This class provides a simplified interface for storing and retrieving
    benchmark data, database files, and model embeddings using IPFS.
    """
    
    def __init__(self, config: Optional[IPFSConfig] = None):
        """Initialize IPFS storage.
        
        Args:
            config: IPFS configuration
        """
        self.backend = IPFSStorageBackend(config)
    
    def is_available(self) -> bool:
        """Check if IPFS storage is available."""
        return self.backend.is_available()
    
    def store(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Store a file on IPFS.
        
        Args:
            file_path: Path to file
            **kwargs: Additional storage options
            
        Returns:
            Storage result metadata
        """
        return self.backend.store_file(file_path, **kwargs)
    
    def retrieve(self, content_id: str, destination: Union[str, Path]) -> Dict[str, Any]:
        """Retrieve a file from IPFS.
        
        Args:
            content_id: Content identifier
            destination: Where to save the file
            
        Returns:
            Retrieval result metadata
        """
        return self.backend.retrieve_file(content_id, destination)
    
    def store_benchmark(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store benchmark result."""
        return self.backend.store_benchmark_result(result_data)
    
    def retrieve_benchmark(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve benchmark result."""
        return self.backend.retrieve_benchmark_result(content_id)
    
    def list_cache(self) -> List[Dict[str, Any]]:
        """List cached files."""
        return self.backend.list_cached_files()
    
    def clear_cache(self, max_age_days: Optional[int] = None) -> Dict[str, Any]:
        """Clear cache."""
        return self.backend.clear_cache(max_age_days)
