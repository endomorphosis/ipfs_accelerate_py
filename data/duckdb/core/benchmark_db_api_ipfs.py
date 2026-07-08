#!/usr/bin/env python
"""
IPFS-Enhanced Benchmark Database API.

Extends the base BenchmarkDBAPI with IPFS capabilities:
- Content-addressable storage
- Distributed database synchronization  
- Decentralized result persistence
- Version tracking with IPFS CIDs

Usage:
    from data.duckdb.core.benchmark_db_api_ipfs import BenchmarkDBAPIIPFS
    
    api = BenchmarkDBAPIIPFS(enable_ipfs=True)
    api.store_performance_result(...)
    
    # Sync to IPFS
    cid = api.sync_to_ipfs()
    print(f"Database on IPFS: {cid}")
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from .benchmark_db_api import BenchmarkDBAPI
    from ..ipfs_integration.ipfs_config import IPFSConfig, get_ipfs_config
    from ..ipfs_integration.ipfs_db_backend import IPFSDBBackend
    from ..ipfs_integration.knowledge_graph import BenchmarkKnowledgeGraph
    from ..ipfs_integration.distributed_ops import DistributedOperations
except ImportError as e:
    logging.warning(f"IPFS integration modules not available: {e}")
    BenchmarkDBAPI = None
    IPFSConfig = None
    IPFSDBBackend = None
    BenchmarkKnowledgeGraph = None
    DistributedOperations = None

logger = logging.getLogger(__name__)


class BenchmarkDBAPIIPFS:
    """
    IPFS-enhanced benchmark database API.
    
    Wraps BenchmarkDBAPI with IPFS capabilities while maintaining
    backward compatibility. All IPFS features are optional and can
    be enabled via configuration.
    """
    
    def __init__(
        self,
        db_path: str = "./benchmark_db.duckdb",
        debug: bool = False,
        enable_ipfs: bool = False,
        ipfs_config: Optional[IPFSConfig] = None,
        enable_distributed: bool = False,
        enable_knowledge_graph: bool = False
    ):
        """
        Initialize IPFS-enhanced benchmark database API.
        
        Args:
            db_path: Path to DuckDB database
            debug: Enable debug logging
            enable_ipfs: Enable IPFS storage features
            ipfs_config: Custom IPFS configuration (optional)
            enable_distributed: Enable distributed query features
            enable_knowledge_graph: Enable knowledge graph features
        """
        # Initialize base API
        if BenchmarkDBAPI is None:
            raise ImportError("BenchmarkDBAPI not available")
        
        self.base_api = BenchmarkDBAPI(db_path=db_path, debug=debug)
        self.db_path = db_path
        
        # Configure IPFS integration
        if ipfs_config is None and (enable_ipfs or enable_distributed or enable_knowledge_graph):
            ipfs_config = get_ipfs_config()
            # Override with explicit flags
            ipfs_config.enable_ipfs_storage = enable_ipfs
            ipfs_config.enable_distributed = enable_distributed
            ipfs_config.enable_knowledge_graph = enable_knowledge_graph
        
        self.config = ipfs_config
        self.ipfs_enabled = enable_ipfs and self.config is not None
        
        # Initialize IPFS components
        self.ipfs_backend: Optional[IPFSDBBackend] = None
        self.knowledge_graph: Optional[BenchmarkKnowledgeGraph] = None
        self.distributed_ops: Optional[DistributedOperations] = None
        
        if self.ipfs_enabled and IPFSDBBackend is not None:
            try:
                self.ipfs_backend = IPFSDBBackend(db_path, config=self.config)
                logger.info("IPFS backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize IPFS backend: {e}")
                self.ipfs_enabled = False
        
        if enable_knowledge_graph and BenchmarkKnowledgeGraph is not None:
            try:
                self.knowledge_graph = BenchmarkKnowledgeGraph(config=self.config)
                logger.info("Knowledge graph initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize knowledge graph: {e}")
        
        if enable_distributed and DistributedOperations is not None:
            try:
                self.distributed_ops = DistributedOperations(config=self.config)
                logger.info("Distributed operations initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed operations: {e}")
        
        logger.info(f"Initialized IPFS-enhanced API (IPFS: {self.ipfs_enabled})")
    
    # Delegate base API methods
    def store_performance_result(self, **kwargs) -> Dict[str, Any]:
        """Store performance result. Delegates to base API."""
        result = self.base_api.store_performance_result(**kwargs)
        
        # Optionally store to IPFS
        if self.ipfs_enabled and self.ipfs_backend:
            try:
                result_cid = self.ipfs_backend.store_result_to_ipfs(result)
                if result_cid:
                    result['ipfs_cid'] = result_cid
                    logger.debug(f"Stored result to IPFS: {result_cid}")
            except Exception as e:
                logger.warning(f"Failed to store result to IPFS: {e}")
        
        # Optionally add to knowledge graph
        if self.knowledge_graph:
            try:
                model_name = kwargs.get('model_name')
                hardware_type = kwargs.get('hardware_type')
                if model_name and hardware_type:
                    self.knowledge_graph.add_node(
                        model_name,
                        node_type='model',
                        properties={'hardware': hardware_type}
                    )
            except Exception as e:
                logger.warning(f"Failed to add to knowledge graph: {e}")
        
        return result
    
    def store_hardware_compatibility(self, **kwargs) -> Dict[str, Any]:
        """Store hardware compatibility result. Delegates to base API."""
        return self.base_api.store_hardware_compatibility(**kwargs)
    
    def store_integration_test_result(self, **kwargs) -> Dict[str, Any]:
        """Store integration test result. Delegates to base API."""
        return self.base_api.store_integration_test_result(**kwargs)
    
    def query_performance_results(self, **kwargs) -> List[Dict[str, Any]]:
        """Query performance results. Delegates to base API."""
        if self.distributed_ops and self.config.enable_distributed:
            try:
                # Use distributed query if enabled
                return self.distributed_ops.execute_distributed_query(
                    "SELECT * FROM performance_results WHERE 1=1",
                    filters=kwargs
                )
            except Exception as e:
                logger.warning(f"Distributed query failed, using local: {e}")
        
        return self.base_api.query_performance_results(**kwargs)
    
    def query_hardware_compatibility(self, **kwargs) -> List[Dict[str, Any]]:
        """Query hardware compatibility. Delegates to base API."""
        return self.base_api.query_hardware_compatibility(**kwargs)
    
    def query_integration_test_results(self, **kwargs) -> List[Dict[str, Any]]:
        """Query integration test results. Delegates to base API."""
        return self.base_api.query_integration_test_results(**kwargs)
    
    def get_model_performance_summary(self, model_name: str) -> Dict[str, Any]:
        """Get performance summary for a model. Delegates to base API."""
        return self.base_api.get_model_performance_summary(model_name)
    
    def get_hardware_benchmark_summary(self, hardware_type: str) -> Dict[str, Any]:
        """Get benchmark summary for hardware. Delegates to base API."""
        return self.base_api.get_hardware_benchmark_summary(hardware_type)
    
    # IPFS-specific methods
    def sync_to_ipfs(self, pin: bool = True) -> Optional[str]:
        """
        Synchronize database to IPFS.
        
        Args:
            pin: Whether to pin content on IPFS
            
        Returns:
            IPFS CID of the database, or None if failed/disabled
        """
        if not self.ipfs_enabled or not self.ipfs_backend:
            logger.warning("IPFS not enabled or backend not available")
            return None
        
        try:
            cid = self.ipfs_backend.sync_to_ipfs(pin=pin)
            if cid:
                logger.info(f"Database synced to IPFS: {cid}")
            return cid
        except Exception as e:
            logger.error(f"Error syncing to IPFS: {e}")
            return None
    
    def restore_from_ipfs(self, cid: Optional[str] = None) -> bool:
        """
        Restore database from IPFS.
        
        Args:
            cid: IPFS CID to restore (uses current if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.ipfs_enabled or not self.ipfs_backend:
            logger.warning("IPFS not enabled or backend not available")
            return False
        
        try:
            success = self.ipfs_backend.restore_from_ipfs(cid)
            if success:
                logger.info(f"Database restored from IPFS: {cid or 'current'}")
            return success
        except Exception as e:
            logger.error(f"Error restoring from IPFS: {e}")
            return False
    
    def get_ipfs_cid(self) -> Optional[str]:
        """
        Get current IPFS CID of the database.
        
        Returns:
            IPFS CID or None if not synced
        """
        if not self.ipfs_enabled or not self.ipfs_backend:
            return None
        
        return self.ipfs_backend.get_current_cid()
    
    def get_ipfs_stats(self) -> Dict[str, Any]:
        """
        Get IPFS usage statistics.
        
        Returns:
            Dictionary with IPFS stats
        """
        if not self.ipfs_enabled or not self.ipfs_backend:
            return {'enabled': False}
        
        return self.ipfs_backend.get_ipfs_stats()
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get database version history from IPFS.
        
        Returns:
            List of versions with CIDs and timestamps
        """
        if not self.ipfs_enabled or not self.ipfs_backend:
            return []
        
        return self.ipfs_backend.get_version_history()
    
    # Knowledge graph methods
    def search_similar_benchmarks(
        self,
        model_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar benchmark results using knowledge graph.
        
        Args:
            model_name: Model to find similar results for
            top_k: Number of results to return
            
        Returns:
            List of similar benchmark results
        """
        if not self.knowledge_graph:
            logger.warning("Knowledge graph not enabled")
            return []
        
        try:
            similar = self.knowledge_graph.find_similar(model_name, top_k=top_k)
            return similar
        except Exception as e:
            logger.error(f"Error searching similar benchmarks: {e}")
            return []
    
    def get_benchmark_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a benchmark node in the knowledge graph.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of relationships
        """
        if not self.knowledge_graph:
            logger.warning("Knowledge graph not enabled")
            return []
        
        try:
            return self.knowledge_graph.get_relationships(node_id)
        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            return []
    
    # Distributed operations methods
    def execute_distributed_query(
        self,
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute a distributed query across multiple nodes.
        
        Args:
            query: SQL query to execute
            **kwargs: Additional query parameters
            
        Returns:
            Query results
        """
        if not self.distributed_ops:
            logger.warning("Distributed operations not enabled")
            return []
        
        try:
            return self.distributed_ops.execute_distributed_query(query, **kwargs)
        except Exception as e:
            logger.error(f"Error executing distributed query: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the API and all integrations.
        
        Returns:
            Status dictionary with all component states
        """
        status = {
            'database': {
                'path': self.db_path,
                'exists': Path(self.db_path).exists(),
                'size_bytes': Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
            },
            'ipfs': {
                'enabled': self.ipfs_enabled,
                'backend_available': self.ipfs_backend is not None,
                'current_cid': self.get_ipfs_cid() if self.ipfs_enabled else None,
                'stats': self.get_ipfs_stats() if self.ipfs_enabled else {}
            },
            'knowledge_graph': {
                'enabled': self.knowledge_graph is not None,
                'available': self.knowledge_graph is not None
            },
            'distributed': {
                'enabled': self.distributed_ops is not None,
                'available': self.distributed_ops is not None
            }
        }
        
        return status


# Convenience function for backward compatibility
def create_benchmark_api(
    db_path: str = "./benchmark_db.duckdb",
    enable_ipfs: bool = False,
    **kwargs
) -> BenchmarkDBAPIIPFS:
    """
    Create a benchmark database API instance.
    
    Args:
        db_path: Path to database
        enable_ipfs: Enable IPFS features
        **kwargs: Additional configuration options
        
    Returns:
        BenchmarkDBAPIIPFS instance
    """
    return BenchmarkDBAPIIPFS(
        db_path=db_path,
        enable_ipfs=enable_ipfs,
        **kwargs
    )
