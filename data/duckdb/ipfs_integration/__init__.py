"""
IPFS Integration for DuckDB API

This module provides IPFS integration for the DuckDB-based benchmark database,
enabling decentralized storage, distributed operations, and knowledge graph capabilities.

Modules:
    ipfs_storage: IPFS storage backend for database files and benchmark data
    ipfs_config: Configuration management for IPFS integration
    distributed_ops: Distributed compute and P2P operations
    knowledge_graph: Knowledge graph for benchmark relationships
    cache_manager: IPFS-based caching mechanisms
"""

# Import with error handling to allow graceful degradation
try:
    from .ipfs_config import IPFSConfig, get_ipfs_config, set_ipfs_config, reset_ipfs_config
except ImportError as e:
    print(f"Warning: Could not import ipfs_config: {e}")
    IPFSConfig = None

try:
    from .ipfs_storage import IPFSStorage, IPFSStorageBackend
except ImportError as e:
    print(f"Warning: Could not import ipfs_storage: {e}")
    IPFSStorage = None
    IPFSStorageBackend = None

try:
    from .distributed_ops import DistributedOperations
except ImportError as e:
    print(f"Warning: Could not import distributed_ops: {e}")
    DistributedOperations = None

try:
    from .knowledge_graph import BenchmarkKnowledgeGraph
except ImportError as e:
    print(f"Warning: Could not import knowledge_graph: {e}")
    BenchmarkKnowledgeGraph = None

try:
    from .cache_manager import IPFSCacheManager
except ImportError as e:
    print(f"Warning: Could not import cache_manager: {e}")
    IPFSCacheManager = None

try:
    from .ipfs_db_backend import IPFSDBBackend, IPFSDBMigration, create_ipfs_backend
except ImportError as e:
    print(f"Warning: Could not import ipfs_db_backend: {e}")
    IPFSDBBackend = None
    IPFSDBMigration = None
    create_ipfs_backend = None

__all__ = [
    'IPFSStorage',
    'IPFSStorageBackend',
    'IPFSConfig',
    'get_ipfs_config',
    'set_ipfs_config',
    'reset_ipfs_config',
    'DistributedOperations',
    'BenchmarkKnowledgeGraph',
    'IPFSCacheManager',
    'IPFSDBBackend',
    'IPFSDBMigration',
    'create_ipfs_backend',
]

__version__ = '2.0.0'
