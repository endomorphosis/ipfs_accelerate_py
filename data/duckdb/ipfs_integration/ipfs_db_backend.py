#!/usr/bin/env python
"""
IPFS-backed database backend for BenchmarkDBAPI.

This module extends the database operations with IPFS storage capabilities,
enabling content-addressable storage, distributed synchronization, and
decentralized benchmark result management.

Usage:
    from data.duckdb.ipfs_integration import IPFSDBBackend
    
    backend = IPFSDBBackend(db_path="benchmarks.db")
    backend.sync_to_ipfs()  # Upload to IPFS
    cid = backend.get_current_cid()  # Get IPFS content ID
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

try:
    import duckdb
except ImportError:
    duckdb = None

from .ipfs_config import IPFSConfig, get_ipfs_config
from .ipfs_storage import IPFSStorage
from .cache_manager import IPFSCacheManager

logger = logging.getLogger(__name__)


class IPFSDBBackend:
    """
    IPFS-backed storage backend for DuckDB databases.
    
    Provides:
    - Content-addressable database storage
    - Automatic IPFS synchronization
    - Local caching for performance
    - Version history with CID tracking
    - Distributed database access
    """
    
    def __init__(
        self,
        db_path: str,
        config: Optional[IPFSConfig] = None,
        auto_sync: bool = False
    ):
        """
        Initialize IPFS database backend.
        
        Args:
            db_path: Path to local DuckDB database file
            config: IPFS configuration (uses global if None)
            auto_sync: Automatically sync changes to IPFS
        """
        self.db_path = Path(db_path)
        self.config = config or get_ipfs_config()
        self.auto_sync = auto_sync
        
        # Initialize IPFS components
        self.storage = IPFSStorage(self.config)
        self.cache = IPFSCacheManager(self.config)
        
        # Track current IPFS state
        self.current_cid: Optional[str] = None
        self.last_sync: Optional[datetime] = None
        self.metadata_path = self.db_path.parent / f".{self.db_path.name}.ipfs.json"
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info(f"Initialized IPFS DB backend for {db_path}")
    
    def _load_metadata(self):
        """Load IPFS metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.current_cid = metadata.get('cid')
                    last_sync_str = metadata.get('last_sync')
                    if last_sync_str:
                        self.last_sync = datetime.fromisoformat(last_sync_str)
                    logger.debug(f"Loaded metadata: CID={self.current_cid}")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
    
    def _save_metadata(self):
        """Save IPFS metadata to disk."""
        try:
            metadata = {
                'cid': self.current_cid,
                'last_sync': self.last_sync.isoformat() if self.last_sync else None,
                'db_path': str(self.db_path),
                'size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata: CID={self.current_cid}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def is_ipfs_enabled(self) -> bool:
        """Check if IPFS storage is enabled and available."""
        return (
            self.config.enable_ipfs_storage and 
            self.storage.is_available()
        )
    
    def sync_to_ipfs(self, pin: bool = True) -> Optional[str]:
        """
        Sync database file to IPFS.
        
        Args:
            pin: Whether to pin the content on IPFS
            
        Returns:
            IPFS CID of the uploaded file, or None if failed
        """
        if not self.is_ipfs_enabled():
            logger.warning("IPFS storage not enabled or unavailable")
            return None
        
        if not self.db_path.exists():
            logger.error(f"Database file not found: {self.db_path}")
            return None
        
        try:
            logger.info(f"Syncing database to IPFS: {self.db_path}")
            result = self.storage.store(
                str(self.db_path),
                metadata={
                    'type': 'duckdb_database',
                    'filename': self.db_path.name,
                    'timestamp': datetime.now().isoformat(),
                    'size_bytes': self.db_path.stat().st_size
                }
            )
            
            if result and 'cid' in result:
                self.current_cid = result['cid']
                self.last_sync = datetime.now()
                self._save_metadata()
                
                logger.info(f"Database synced to IPFS: {self.current_cid}")
                return self.current_cid
            else:
                logger.error("Failed to sync database to IPFS")
                return None
                
        except Exception as e:
            logger.error(f"Error syncing to IPFS: {e}")
            return None
    
    def restore_from_ipfs(self, cid: Optional[str] = None) -> bool:
        """
        Restore database from IPFS.
        
        Args:
            cid: IPFS CID to restore (uses current_cid if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_ipfs_enabled():
            logger.warning("IPFS storage not enabled or unavailable")
            return False
        
        restore_cid = cid or self.current_cid
        if not restore_cid:
            logger.error("No CID specified for restore")
            return False
        
        try:
            logger.info(f"Restoring database from IPFS: {restore_cid}")
            
            # Create backup of existing file
            if self.db_path.exists():
                backup_path = self.db_path.with_suffix('.backup')
                shutil.copy2(self.db_path, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            
            # Restore from IPFS
            success = self.storage.retrieve(restore_cid, str(self.db_path))
            
            if success:
                logger.info(f"Database restored from IPFS: {restore_cid}")
                self.current_cid = restore_cid
                self.last_sync = datetime.now()
                self._save_metadata()
                return True
            else:
                logger.error("Failed to restore database from IPFS")
                return False
                
        except Exception as e:
            logger.error(f"Error restoring from IPFS: {e}")
            return False
    
    def get_current_cid(self) -> Optional[str]:
        """Get current IPFS CID of the database."""
        return self.current_cid
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get version history from cache.
        
        Returns:
            List of version entries with CID and timestamp
        """
        # This is a placeholder - would need to track versions in metadata
        versions = []
        if self.current_cid and self.last_sync:
            versions.append({
                'cid': self.current_cid,
                'timestamp': self.last_sync.isoformat(),
                'size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0
            })
        return versions
    
    def store_result_to_ipfs(self, result_data: Dict[str, Any]) -> Optional[str]:
        """
        Store a single benchmark result to IPFS.
        
        Args:
            result_data: Benchmark result dictionary
            
        Returns:
            IPFS CID of stored result, or None if failed
        """
        if not self.is_ipfs_enabled():
            return None
        
        try:
            # Create temporary file with result
            result_id = result_data.get('result_id', 'unknown')
            temp_file = self.db_path.parent / f"result_{result_id}.json"
            
            with open(temp_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            # Store to IPFS
            result = self.storage.store(
                str(temp_file),
                metadata={
                    'type': 'benchmark_result',
                    'result_id': result_id,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Clean up temp file
            temp_file.unlink()
            
            if result and 'cid' in result:
                logger.debug(f"Stored result {result_id} to IPFS: {result['cid']}")
                return result['cid']
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error storing result to IPFS: {e}")
            return None
    
    def retrieve_result_from_ipfs(self, cid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a benchmark result from IPFS.
        
        Args:
            cid: IPFS CID of the result
            
        Returns:
            Result dictionary, or None if failed
        """
        if not self.is_ipfs_enabled():
            return None
        
        try:
            # Create temporary file for retrieval
            temp_file = self.db_path.parent / f"temp_result_{cid[:8]}.json"
            
            success = self.storage.retrieve(cid, str(temp_file))
            
            if success and temp_file.exists():
                with open(temp_file, 'r') as f:
                    result_data = json.load(f)
                
                # Clean up temp file
                temp_file.unlink()
                
                logger.debug(f"Retrieved result from IPFS: {cid}")
                return result_data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving result from IPFS: {e}")
            return None
    
    def get_ipfs_stats(self) -> Dict[str, Any]:
        """
        Get statistics about IPFS usage.
        
        Returns:
            Dictionary with IPFS statistics
        """
        stats = {
            'enabled': self.is_ipfs_enabled(),
            'current_cid': self.current_cid,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'db_size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0,
            'cache_stats': self.cache.get_cache_stats() if self.config.enable_cache else {}
        }
        return stats


class IPFSDBMigration:
    """
    Utilities for migrating existing databases to IPFS.
    """
    
    def __init__(self, config: Optional[IPFSConfig] = None):
        """
        Initialize migration utilities.
        
        Args:
            config: IPFS configuration
        """
        self.config = config or get_ipfs_config()
        self.storage = IPFSStorage(self.config)
    
    def migrate_database(
        self,
        db_path: str,
        create_backup: bool = True
    ) -> Optional[str]:
        """
        Migrate a database to IPFS storage.
        
        Args:
            db_path: Path to database file
            create_backup: Whether to create a local backup
            
        Returns:
            IPFS CID of migrated database, or None if failed
        """
        db_file = Path(db_path)
        
        if not db_file.exists():
            logger.error(f"Database file not found: {db_path}")
            return None
        
        try:
            # Create backup if requested
            if create_backup:
                backup_path = db_file.with_suffix('.pre-ipfs-backup')
                shutil.copy2(db_file, backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Create backend and sync to IPFS
            backend = IPFSDBBackend(str(db_file), config=self.config)
            cid = backend.sync_to_ipfs()
            
            if cid:
                logger.info(f"Successfully migrated {db_path} to IPFS: {cid}")
                return cid
            else:
                logger.error(f"Failed to migrate {db_path} to IPFS")
                return None
                
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            return None
    
    def batch_migrate(
        self,
        db_paths: List[str],
        create_backups: bool = True
    ) -> Dict[str, Optional[str]]:
        """
        Migrate multiple databases to IPFS.
        
        Args:
            db_paths: List of database file paths
            create_backups: Whether to create local backups
            
        Returns:
            Dictionary mapping paths to CIDs (or None if failed)
        """
        results = {}
        
        for db_path in db_paths:
            logger.info(f"Migrating database: {db_path}")
            cid = self.migrate_database(db_path, create_backup=create_backups)
            results[db_path] = cid
        
        # Summary
        successful = sum(1 for cid in results.values() if cid is not None)
        total = len(results)
        logger.info(f"Migration complete: {successful}/{total} successful")
        
        return results


# Convenience function
def create_ipfs_backend(
    db_path: str,
    config: Optional[IPFSConfig] = None
) -> IPFSDBBackend:
    """
    Create an IPFS database backend.
    
    Args:
        db_path: Path to database file
        config: IPFS configuration (optional)
        
    Returns:
        IPFSDBBackend instance
    """
    return IPFSDBBackend(db_path, config)
