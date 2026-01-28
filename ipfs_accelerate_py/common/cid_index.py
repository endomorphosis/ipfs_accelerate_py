"""
CID-based Cache Index

Provides fast lookups for content-addressed cache entries using CID-based indexing.
Maintains an in-memory index of CIDs to cache entries for O(1) lookup performance.
"""

import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import json

try:
    from .storage_wrapper import storage_wrapper
except (ImportError, ValueError):
    try:
        from common.storage_wrapper import storage_wrapper
    except ImportError:
        storage_wrapper = None

logger = logging.getLogger(__name__)


class CIDCacheIndex:
    """
    Index for fast CID-based cache lookups.
    
    Features:
    - O(1) CID lookup
    - Prefix-based search (find all entries matching a CID prefix)
    - Operation-based filtering (find all entries for a given operation)
    - Thread-safe operations
    """
    
    def __init__(self):
        """Initialize the CID cache index."""
        if storage_wrapper:
            try:
                self.storage = storage_wrapper()
            except:
                self.storage = None
        else:
            self.storage = None
        self._cid_to_metadata: Dict[str, Dict[str, Any]] = {}
        self._operation_to_cids: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()
    
    def add(self, cid: str, operation: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a CID to the index.
        
        Args:
            cid: Content identifier
            operation: Operation name
            metadata: Optional metadata about the cache entry
        """
        with self._lock:
            self._cid_to_metadata[cid] = {
                "operation": operation,
                "metadata": metadata or {}
            }
            self._operation_to_cids[operation].add(cid)
    
    def remove(self, cid: str) -> bool:
        """
        Remove a CID from the index.
        
        Args:
            cid: Content identifier to remove
            
        Returns:
            True if CID was found and removed, False otherwise
        """
        with self._lock:
            if cid in self._cid_to_metadata:
                entry = self._cid_to_metadata[cid]
                operation = entry["operation"]
                
                # Remove from operation index
                if operation in self._operation_to_cids:
                    self._operation_to_cids[operation].discard(cid)
                    if not self._operation_to_cids[operation]:
                        del self._operation_to_cids[operation]
                
                # Remove from main index
                del self._cid_to_metadata[cid]
                return True
            return False
    
    def get(self, cid: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a CID.
        
        Args:
            cid: Content identifier
            
        Returns:
            Metadata dict or None if not found
        """
        with self._lock:
            return self._cid_to_metadata.get(cid)
    
    def has(self, cid: str) -> bool:
        """
        Check if a CID exists in the index.
        
        Args:
            cid: Content identifier
            
        Returns:
            True if CID exists, False otherwise
        """
        with self._lock:
            return cid in self._cid_to_metadata
    
    def find_by_prefix(self, prefix: str, max_results: int = 100) -> List[str]:
        """
        Find all CIDs matching a prefix.
        
        Args:
            prefix: CID prefix to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of matching CIDs
        """
        with self._lock:
            matches = [
                cid for cid in self._cid_to_metadata.keys()
                if cid.startswith(prefix)
            ]
            return matches[:max_results]
    
    def find_by_operation(self, operation: str) -> List[str]:
        """
        Find all CIDs for a given operation.
        
        Args:
            operation: Operation name
            
        Returns:
            List of CIDs for this operation
        """
        with self._lock:
            return list(self._operation_to_cids.get(operation, set()))
    
    def clear(self) -> None:
        """Clear the entire index."""
        with self._lock:
            self._cid_to_metadata.clear()
            self._operation_to_cids.clear()
    
    def size(self) -> int:
        """
        Get the number of CIDs in the index.
        
        Returns:
            Number of indexed CIDs
        """
        with self._lock:
            return len(self._cid_to_metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                "total_cids": len(self._cid_to_metadata),
                "operations": len(self._operation_to_cids),
                "operation_counts": {
                    op: len(cids) 
                    for op, cids in self._operation_to_cids.items()
                }
            }
    
    def save_to_file(self, filepath: Path) -> None:
        """
        Save the index to a file.
        
        Args:
            filepath: Path to save the index
        """
        with self._lock:
            data = {
                "cid_to_metadata": dict(self._cid_to_metadata),
                "operation_to_cids": {
                    op: list(cids) 
                    for op, cids in self._operation_to_cids.items()
                }
            }
            
            try:
                if self.storage:
                    json_data = json.dumps(data, indent=2)
                    self.storage.write_file(str(filepath), json_data.encode('utf-8'), pin=True)
                else:
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                logger.info(f"✓ Saved CID index to {filepath}")
            except Exception as e:
                try:
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                    logger.info(f"✓ Saved CID index to {filepath}")
                except Exception as e2:
                    logger.warning(f"Failed to save CID index: {e2}")
    
    def load_from_file(self, filepath: Path) -> None:
        """
        Load the index from a file.
        
        Args:
            filepath: Path to load the index from
        """
        if not filepath.exists():
            return
        
        try:
            if self.storage:
                json_data = self.storage.read_file(str(filepath), pin=True)
                data = json.loads(json_data.decode('utf-8'))
            else:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            
            with self._lock:
                self._cid_to_metadata = data.get("cid_to_metadata", {})
                self._operation_to_cids = defaultdict(set)
                
                for op, cids in data.get("operation_to_cids", {}).items():
                    self._operation_to_cids[op] = set(cids)
            
            logger.info(f"✓ Loaded {len(self._cid_to_metadata)} CIDs from {filepath}")
        except Exception as e:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                with self._lock:
                    self._cid_to_metadata = data.get("cid_to_metadata", {})
                    self._operation_to_cids = defaultdict(set)
                    
                    for op, cids in data.get("operation_to_cids", {}).items():
                        self._operation_to_cids[op] = set(cids)
                
                logger.info(f"✓ Loaded {len(self._cid_to_metadata)} CIDs from {filepath}")
            except Exception as e2:
                logger.warning(f"Failed to load CID index: {e2}")


# Global CID index instance
_global_cid_index: Optional[CIDCacheIndex] = None
_index_lock = threading.Lock()


def get_global_cid_index() -> CIDCacheIndex:
    """Get or create the global CID index."""
    global _global_cid_index
    
    with _index_lock:
        if _global_cid_index is None:
            _global_cid_index = CIDCacheIndex()
        
        return _global_cid_index


def reset_global_cid_index() -> None:
    """Reset the global CID index (mainly for testing)."""
    global _global_cid_index
    
    with _index_lock:
        if _global_cid_index is not None:
            _global_cid_index.clear()
        _global_cid_index = None
