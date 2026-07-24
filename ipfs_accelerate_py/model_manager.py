#!/usr/bin/env python3
"""
IPFS Accelerate Python - Model Manager

This module provides a comprehensive model manager for storing and managing
metadata about different types of models, including:
- Input/output type mappings
- HuggingFace configuration data
- Inference code locations
- Model architecture information
- Performance characteristics
"""

import os
import sys
import json
import logging
import urllib.error
import urllib.request
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum

# Try to import IPFS multiformats for content addressing
try:
    from .ipfs_multiformats import ipfs_multiformats_py
    HAVE_IPFS_MULTIFORMATS = True
except ImportError:
    try:
        from ipfs_multiformats import ipfs_multiformats_py  
        HAVE_IPFS_MULTIFORMATS = True
    except ImportError:
        # Attempt auto-install of missing optional deps when allowed
        try:
            from .utils.auto_install import ensure_packages
            ensure_packages({
                "multiformats": "multiformats",
            })
            from ipfs_multiformats import ipfs_multiformats_py  # retry after install
            HAVE_IPFS_MULTIFORMATS = True
        except Exception:
            HAVE_IPFS_MULTIFORMATS = False

# Try to import storage wrapper for distributed filesystem
try:
    from .common.storage_wrapper import get_storage_wrapper
    HAVE_STORAGE_WRAPPER = True
except ImportError:
    HAVE_STORAGE_WRAPPER = False
    get_storage_wrapper = None
    logger.debug("Storage wrapper not available for model manager")

# Try to import the unified IPFS kit storage helper for artifact persistence
try:
    from .ipfs_kit_integration import IPFSKitStorage
    HAVE_IPFS_KIT_STORAGE = True
except ImportError:
    HAVE_IPFS_KIT_STORAGE = False
    IPFSKitStorage = None
    logger.debug("IPFS Kit storage helper not available for model manager")

# Try to import datasets integration for provenance tracking and IPFS storage
try:
    from .datasets_integration import (
        is_datasets_available,
        DatasetsManager,
        FilesystemHandler,
        ProvenanceLogger
    )
    HAVE_DATASETS_INTEGRATION = True
except ImportError:
    HAVE_DATASETS_INTEGRATION = False
    is_datasets_available = lambda: False
    DatasetsManager = None
    FilesystemHandler = None
    ProvenanceLogger = None
    logger.debug("Datasets integration not available for model manager")

# Try to import ModelKnowledgeGraph for GraphRAG support
try:
    from .model_manager_graphrag import ModelKnowledgeGraph
    HAVE_GRAPHRAG = True
except ImportError:
    HAVE_GRAPHRAG = False
    ModelKnowledgeGraph = None
    logger.debug("ModelKnowledgeGraph not available for model manager")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ipfs_accelerate_model_manager")

# Try to import DuckDB for database storage
try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    # Attempt auto-install
    try:
        from .utils.auto_install import ensure_packages
        ensure_packages(["duckdb"])
        import duckdb  # retry
        HAVE_DUCKDB = True
    except Exception:
        HAVE_DUCKDB = False
        logger.warning("DuckDB not available. Using JSON storage backend.")

# Try to import requests for HuggingFace API access
try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    try:
        from .utils.auto_install import ensure_packages
        ensure_packages(["requests"])  # retry
        import requests
        HAVE_REQUESTS = True
    except Exception:
        HAVE_REQUESTS = False
        logger.warning("Requests not available. HuggingFace repository fetching disabled.")

# Initialize IPFS multiformats instance if available
ipfs_multiformats = None
if HAVE_IPFS_MULTIFORMATS:
    try:
        ipfs_multiformats = ipfs_multiformats_py(None, None)
        logger.info("IPFS content addressing enabled")
    except Exception as e:
        logger.warning(f"Failed to initialize IPFS multiformats: {e}")
        HAVE_IPFS_MULTIFORMATS = False
else:
    logger.warning("IPFS multiformats not available. Content addressing disabled.")


def fetch_huggingface_repo_structure(model_id: str, branch: str = "main", include_ipfs_cids: bool = True) -> Optional[Dict[str, Any]]:
    """
    Fetch file structure and hashes from a HuggingFace repository with IPFS content addressing.
    
    Args:
        model_id: HuggingFace model ID (e.g., "bert-base-uncased")
        branch: Git branch to fetch from (default: "main")
        include_ipfs_cids: Whether to generate IPFS CIDs for files (default: True)
        
    Returns:
        Dictionary containing repository structure with file paths, hashes, and IPFS CIDs,
        or None if fetching fails
    """
    if not HAVE_REQUESTS:
        logger.warning("Cannot fetch HuggingFace repo structure: requests library not available")
        return None
    
    try:
        # HuggingFace API endpoint for repository files
        api_url = f"https://huggingface.co/api/models/{model_id}/tree/{branch}"
        
        logger.info(f"Fetching repository structure for {model_id} from branch {branch}")
        
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        files_data = response.json()
        
        # Process the file structure
        repo_structure = {
            "model_id": model_id,
            "branch": branch,
            "fetched_at": datetime.now().isoformat(),
            "files": {},
            "total_files": 0,
            "total_size": 0,
            "ipfs_enabled": include_ipfs_cids and HAVE_IPFS_MULTIFORMATS
        }
        
        for file_info in files_data:
            if file_info.get("type") == "file":
                file_path = file_info.get("path", "")
                file_data = {
                    "size": file_info.get("size", 0),
                    "lfs": file_info.get("lfs", {}),
                    "oid": file_info.get("oid", ""),
                    "download_url": f"https://huggingface.co/{model_id}/resolve/{branch}/{file_path}"
                }
                
                # Generate IPFS CID if requested and available
                if include_ipfs_cids and HAVE_IPFS_MULTIFORMATS and ipfs_multiformats:
                    try:
                        # For LFS files, use the SHA256 hash to generate CID
                        if file_info.get("lfs") and "sha256" in file_info["lfs"]:
                            sha256_hash = file_info["lfs"]["sha256"]
                            # Convert hex string to bytes
                            hash_bytes = bytes.fromhex(sha256_hash)
                            # Generate CID from the hash
                            cid = ipfs_multiformats.get_cid(hash_bytes)
                            file_data["ipfs_cid"] = cid
                        # For regular files, we'd need to download to generate CID
                        # For now, we'll generate a CID from the OID if available  
                        elif file_info.get("oid"):
                            try:
                                # Use the git OID to generate a CID
                                cid = ipfs_multiformats.get_cid(file_info["oid"])
                                file_data["ipfs_cid"] = cid
                            except Exception:
                                # If CID generation fails, continue without it
                                pass
                    except Exception as e:
                        logger.debug(f"Could not generate IPFS CID for {file_path}: {e}")
                
                repo_structure["files"][file_path] = file_data
                repo_structure["total_files"] += 1
                repo_structure["total_size"] += file_info.get("size", 0)
        
        logger.info(f"Fetched {repo_structure['total_files']} files for {model_id}")
        if repo_structure.get("ipfs_enabled"):
            cid_count = sum(1 for f in repo_structure["files"].values() if "ipfs_cid" in f)
            logger.info(f"Generated IPFS CIDs for {cid_count} files")
        
        return repo_structure
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching HuggingFace repo structure for {model_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching HuggingFace repo structure for {model_id}: {e}")
        return None


def get_file_hash_from_structure(repo_structure: Dict[str, Any], file_path: str) -> Optional[str]:
    """
    Get the hash/OID for a specific file from repository structure.
    
    Args:
        repo_structure: Repository structure from fetch_huggingface_repo_structure
        file_path: Path to the file
        
    Returns:
        File hash/OID or None if not found
    """
    if not repo_structure or "files" not in repo_structure:
        return None
    
    file_info = repo_structure["files"].get(file_path)
    if file_info:
        return file_info.get("oid")
    
    return None


def list_files_by_extension(repo_structure: Dict[str, Any], extension: str) -> List[str]:
    """
    List all files with a specific extension from repository structure.
    
    Args:
        repo_structure: Repository structure from fetch_huggingface_repo_structure
        extension: File extension to search for (e.g., ".py", ".json")
        
    Returns:
        List of file paths with the specified extension
    """
    if not repo_structure or "files" not in repo_structure:
        return []
    
    matching_files = []
    for file_path in repo_structure["files"].keys():
        if file_path.endswith(extension):
            matching_files.append(file_path)
    
    return sorted(matching_files)


class ModelType(Enum):
    """Enumeration of supported model types."""
    LANGUAGE_MODEL = "language_model"
    VISION_MODEL = "vision_model"
    MULTIMODAL = "multimodal"
    AUDIO_MODEL = "audio_model"
    EMBEDDING_MODEL = "embedding_model"
    ENCODER_DECODER = "encoder_decoder"
    ENCODER_ONLY = "encoder_only"
    DECODER_ONLY = "decoder_only"


class DataType(Enum):
    """Enumeration of supported data types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    EMBEDDINGS = "embeddings"
    TOKENS = "tokens"
    LOGITS = "logits"
    FEATURES = "features"


@dataclass
class ServingConfig:
    """
    Per-model configuration for hosting and traffic routing.

    This dataclass captures everything a serving layer needs to launch,
    configure, and route traffic to a model.  It is stored as a JSON blob
    alongside ``ModelMetadata`` so that any cluster node can discover the
    correct launch args without manual configuration.

    Fields
    ------
    engine:
        Serving backend identifier.  One of: ``"vllm"``, ``"tgi"``,
        ``"triton"``, ``"onnxruntime"``, ``"llama.cpp"``, ``"hf_pipeline"``.
    launch_args:
        Free-form engine-specific CLI flags, e.g.
        ``{"--tensor-parallel-size": 2, "--quantization": "awq"}``.
    default_generation_params:
        Default parameters passed to the inference API, e.g.
        ``{"temperature": 0.7, "max_new_tokens": 512}``.
    endpoint_schema:
        JSON schema describing the model's HTTP input / output contract,
        used by the traffic router for validation.
    routing_weight:
        Relative weight for weighted load balancing across replicas
        (default: 1.0).
    min_replicas:
        Minimum number of replicas for autoscaling (default: 1).
    max_replicas:
        Maximum number of replicas for autoscaling (default: 1).
    hardware_affinity:
        Ordered list of hardware types this model is optimised for,
        e.g. ``["cuda", "rocm", "cpu"]``.
    """
    engine: str = "hf_pipeline"
    launch_args: Optional[Dict[str, Any]] = None
    default_generation_params: Optional[Dict[str, Any]] = None
    endpoint_schema: Optional[Dict[str, Any]] = None
    routing_weight: float = 1.0
    min_replicas: int = 1
    max_replicas: int = 1
    hardware_affinity: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.launch_args is None:
            self.launch_args = {}
        if self.default_generation_params is None:
            self.default_generation_params = {}
        if self.hardware_affinity is None:
            self.hardware_affinity = []

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-compatible)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServingConfig":
        """Deserialise from a plain dict."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
@dataclass
@dataclass
class IOSpec:
    """Specification for model input or output."""
    name: str
    data_type: DataType
    shape: Optional[Tuple[int, ...]] = None
    dtype: str = "float32"
    description: str = ""
    optional: bool = False


@dataclass
class ModelMetadata:
    """Comprehensive metadata for a model."""
    model_id: str
    model_name: str
    model_type: ModelType
    architecture: str
    inputs: List[IOSpec]
    outputs: List[IOSpec]
    huggingface_config: Optional[Dict[str, Any]] = None
    inference_code_location: Optional[str] = None
    supported_backends: List[str] = None
    hardware_requirements: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    tags: List[str] = None
    source_url: Optional[str] = None
    license: Optional[str] = None
    description: str = ""
    model_card: Optional[str] = None
    repository_structure: Optional[Dict[str, Any]] = None
    model_cid: Optional[str] = None
    config_cid: Optional[str] = None
    tokenizer_cid: Optional[str] = None
    artifact_cid: Optional[str] = None
    model_revision: Optional[str] = None
    revision_id: Optional[str] = None
    revision_created_at: Optional[datetime] = None
    parent_model_id: Optional[str] = None
    parent_model_cid: Optional[str] = None
    last_used_at: Optional[datetime] = None
    last_inference_cid: Optional[str] = None
    last_run_id: Optional[str] = None
    inference_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Serving configuration: engine, launch args, routing, autoscaling
    serving_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize defaults after object creation."""
        if self.supported_backends is None:
            self.supported_backends = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.revision_id is None and self.model_revision is not None:
            self.revision_id = self.model_revision
        if self.revision_created_at is None and self.created_at is not None:
            self.revision_created_at = self.created_at


class ModelManager:
    """
    Comprehensive model manager for storing and retrieving model metadata.
    
    This class provides functionality to:
    - Store and retrieve model metadata
    - Map input/output types for different models
    - Manage HuggingFace configuration data
    - Track inference code locations
    - Query models by various criteria
    - Cache model weights via ipfs_kit_py multi-tier cache
    - Build and query a knowledge graph over registered models (GraphRAG)
    - Store and resolve per-model serving configurations (launch args, engine, etc.)
    """
    
    def __init__(
        self,
        storage_path: str = None,
        use_database: bool = None,
        enable_ipfs: bool = None,
        # --- Gap 1: tiered cache settings ---
        cache_memory_mb: int = 100,
        cache_disk_mb: int = 1024,
        cache_eviction_policy: str = "lru",
    ):
        """
        Initialize the model manager.
        
        Args:
            storage_path: Path for storage backend (JSON file or DuckDB database)
            use_database: Whether to use DuckDB backend. If None, auto-detect.
            enable_ipfs: Whether to enable IPFS backend for model storage. If None, auto-detect.
            cache_memory_mb: In-memory cache size in MB for ipfs_kit_py tiered cache (default: 100).
            cache_disk_mb: Disk cache quota in MB for ipfs_kit_py tiered cache (default: 1024).
            cache_eviction_policy: Cache eviction policy – "lru", "lfu", or "arc" (default: "lru").
        """
        # Determine storage backend
        if use_database is None:
            use_database = HAVE_DUCKDB
        
        self.use_database = use_database and HAVE_DUCKDB
        
        # Setup storage
        if storage_path is None:
            if self.use_database:
                storage_path = os.environ.get("MODEL_MANAGER_DB_PATH", "./model_manager.duckdb")
            else:
                storage_path = os.environ.get("MODEL_MANAGER_JSON_PATH", "./model_metadata.json")
        
        self.storage_path = storage_path
        self.models: Dict[str, ModelMetadata] = {}

        # Derive a per-instance cache directory co-located with storage when not
        # explicitly set, so different ModelManager instances don't share a cache.
        _default_cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(storage_path)),
            ".ipfs_kit_cache",
        )
        
        # Initialize IPFS backend router for distributed model storage
        self._ipfs_backend = None
        self.enable_ipfs = enable_ipfs if enable_ipfs is not None else \
                          os.getenv("ENABLE_IPFS_MODEL_STORAGE", "").lower() in ("1", "true", "yes")
        if self.enable_ipfs:
            self._init_ipfs_backend()
        
        # Initialize datasets integration for provenance tracking and IPFS storage
        self._datasets_manager = None
        self._filesystem_handler = None
        self._provenance_logger = None
        if HAVE_DATASETS_INTEGRATION and is_datasets_available():
            try:
                self._datasets_manager = DatasetsManager({
                    'enable_audit': True,
                    'enable_provenance': True,
                    'enable_p2p': False
                })
                self._filesystem_handler = FilesystemHandler()
                self._provenance_logger = ProvenanceLogger()
                logger.info("Model manager using datasets integration for provenance tracking")
            except Exception as e:
                logger.debug(f"Datasets integration initialization skipped: {e}")

        self._artifact_storage = None
        if HAVE_IPFS_KIT_STORAGE:
            try:
                self._artifact_storage = IPFSKitStorage(
                    enable_ipfs_kit=self.enable_ipfs,
                    cache_dir=_default_cache_dir,
                    deps=self._datasets_manager,
                )
                # Apply tiered cache configuration
                self._artifact_storage.configure_cache(
                    memory_mb=cache_memory_mb,
                    disk_mb=cache_disk_mb,
                    eviction_policy=cache_eviction_policy,
                )
            except Exception as e:
                logger.debug(f"IPFS Kit storage initialization skipped: {e}")
        
        # Initialize storage wrapper for distributed filesystem (with gating)
        self._storage_wrapper = None
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage_wrapper = get_storage_wrapper(auto_detect_ci=True)
                if self._storage_wrapper.is_distributed:
                    logger.info("Model manager using distributed storage backend")
                else:
                    logger.debug("Model manager using local filesystem (distributed storage disabled)")
            except Exception as e:
                logger.debug(f"Storage wrapper initialization skipped: {e}")

        # Initialize knowledge graph for GraphRAG (Gap 2)
        self._knowledge_graph: Optional["ModelKnowledgeGraph"] = None
        if HAVE_GRAPHRAG:
            try:
                self._knowledge_graph = ModelKnowledgeGraph(
                    datasets_manager=self._datasets_manager,
                    storage=self._artifact_storage,
                )
                logger.info("Model knowledge graph initialized")
            except Exception as e:
                logger.debug(f"ModelKnowledgeGraph initialization skipped: {e}")
        
        # Initialize storage backend
        if self.use_database:
            self._init_database()
        else:
            self._init_json_storage()
        
        # Load existing data
        self._load_data()
    
    def _init_ipfs_backend(self):
        """Initialize IPFS backend router for model storage."""
        try:
            from . import ipfs_backend_router
            self._ipfs_backend = ipfs_backend_router
            logger.info("✓ IPFS backend router initialized for model storage")
        except Exception as e:
            logger.warning(f"Failed to initialize IPFS backend router: {e}")
            self._ipfs_backend = None
        
    def _init_database(self):
        """Initialize DuckDB database backend."""
        try:
            self.con = duckdb.connect(self.storage_path)
            logger.info(f"Connected to model manager database at: {self.storage_path}")
            self._create_database_tables()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            logger.info("Falling back to JSON storage")
            self.use_database = False
            self._init_json_storage()
    
    def _init_json_storage(self):
        """Initialize JSON file storage backend."""
        self.json_path = self.storage_path
        if not self.json_path.endswith('.json'):
            self.json_path = self.storage_path + '.json'
        logger.info(f"Using JSON storage at: {self.json_path}")
    
    def _create_database_tables(self):
        """Create database tables for model metadata storage."""
        if not self.use_database:
            return
            
        try:
            # Create models table
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id VARCHAR PRIMARY KEY,
                    model_name VARCHAR NOT NULL,
                    model_type VARCHAR NOT NULL,
                    architecture VARCHAR NOT NULL,
                    inputs JSON,
                    outputs JSON,
                    huggingface_config JSON,
                    inference_code_location VARCHAR,
                    supported_backends JSON,
                    hardware_requirements JSON,
                    performance_metrics JSON,
                    tags JSON,
                    source_url VARCHAR,
                    license VARCHAR,
                    description TEXT,
                    model_card TEXT,
                    repository_structure JSON,
                    model_cid VARCHAR,
                    config_cid VARCHAR,
                    tokenizer_cid VARCHAR,
                    artifact_cid VARCHAR,
                    model_revision VARCHAR,
                    revision_id VARCHAR,
                    revision_created_at TIMESTAMP,
                    parent_model_id VARCHAR,
                    parent_model_cid VARCHAR,
                    last_used_at TIMESTAMP,
                    last_inference_cid VARCHAR,
                    last_run_id VARCHAR,
                    inference_count BIGINT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)

            for column_name in ["model_cid", "config_cid", "tokenizer_cid", "artifact_cid", "model_revision", "revision_id", "revision_created_at", "parent_model_id", "parent_model_cid", "last_used_at", "last_inference_cid", "last_run_id", "inference_count", "serving_config"]:
                try:
                    col_type = (
                        'TIMESTAMP' if column_name in {'last_used_at', 'revision_created_at'} else
                        'BIGINT' if column_name == 'inference_count' else
                        'JSON' if column_name == 'serving_config' else
                        'VARCHAR'
                    )
                    self.con.execute(
                        f"ALTER TABLE model_metadata ADD COLUMN IF NOT EXISTS {column_name} {col_type}"
                    )
                except Exception:
                    pass
            
            # Create indexes for efficient querying
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_model_type ON model_metadata(model_type)")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_architecture ON model_metadata(architecture)")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON model_metadata(model_name)")
            
            logger.info("Model manager database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    def _load_data(self):
        """Load existing model metadata from storage."""
        try:
            if self.use_database:
                self._load_from_database()
            else:
                self._load_from_json()
        except Exception as e:
            logger.error(f"Error loading model data: {e}")
    
    def _load_from_database(self):
        """Load model metadata from DuckDB database."""
        if not self.use_database:
            return
            
        try:
            result = self.con.execute("SELECT * FROM model_metadata").fetchall()
            columns = [desc[0] for desc in self.con.description]
            
            for row in result:
                row_dict = dict(zip(columns, row))
                model_id = row_dict['model_id']
                
                # Parse JSON fields
                for json_field in ['inputs', 'outputs', 'huggingface_config', 'supported_backends', 
                                 'hardware_requirements', 'performance_metrics', 'tags',
                                 'repository_structure', 'serving_config']:
                    if row_dict.get(json_field):
                        row_dict[json_field] = json.loads(row_dict[json_field])

                for datetime_field in ['created_at', 'updated_at', 'last_used_at', 'revision_created_at']:
                    if row_dict.get(datetime_field) and isinstance(row_dict[datetime_field], str):
                        row_dict[datetime_field] = datetime.fromisoformat(row_dict[datetime_field])
                
                # Convert to IOSpec objects
                if row_dict['inputs']:
                    row_dict['inputs'] = [IOSpec(**spec) for spec in row_dict['inputs']]
                if row_dict['outputs']:
                    row_dict['outputs'] = [IOSpec(**spec) for spec in row_dict['outputs']]
                
                # Convert enum fields
                row_dict['model_type'] = ModelType(row_dict['model_type'])

                # Strip unknown keys that may come from older schema versions
                known_fields = {f.name for f in ModelMetadata.__dataclass_fields__.values()}  # type: ignore[attr-defined]
                row_dict = {k: v for k, v in row_dict.items() if k in known_fields}
                
                self.models[model_id] = ModelMetadata(**row_dict)
                
            logger.info(f"Loaded {len(self.models)} models from database")
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
    
    def _load_from_json(self):
        """Load model metadata from JSON file."""
        if not os.path.exists(self.json_path):
            logger.info(f"JSON file {self.json_path} does not exist, starting with empty model registry")
            return
            
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            for model_id, model_data in data.items():
                # Convert ISO datetime strings back to datetime objects
                if 'created_at' in model_data and model_data['created_at']:
                    model_data['created_at'] = datetime.fromisoformat(model_data['created_at'])
                if 'updated_at' in model_data and model_data['updated_at']:
                    model_data['updated_at'] = datetime.fromisoformat(model_data['updated_at'])
                if 'last_used_at' in model_data and model_data['last_used_at']:
                    model_data['last_used_at'] = datetime.fromisoformat(model_data['last_used_at'])
                if 'revision_created_at' in model_data and model_data['revision_created_at']:
                    model_data['revision_created_at'] = datetime.fromisoformat(model_data['revision_created_at'])
                
                # Convert to IOSpec objects
                if 'inputs' in model_data and model_data['inputs']:
                    model_data['inputs'] = [IOSpec(**spec) for spec in model_data['inputs']]
                if 'outputs' in model_data and model_data['outputs']:
                    model_data['outputs'] = [IOSpec(**spec) for spec in model_data['outputs']]
                
                # Convert enum fields
                if 'model_type' in model_data:
                    model_data['model_type'] = ModelType(model_data['model_type'])
                
                self.models[model_id] = ModelMetadata(**model_data)
            
            logger.info(f"Loaded {len(self.models)} models from JSON file")
        except Exception as e:
            logger.error(f"Error loading from JSON: {e}")
    
    def _save_data(self):
        """Save model metadata to storage."""
        try:
            if self.use_database:
                self._save_to_database()
            else:
                self._save_to_json()
        except Exception as e:
            logger.error(f"Error saving model data: {e}")
    
    def _save_to_database(self):
        """Save model metadata to DuckDB database."""
        if not self.use_database:
            return
            
        try:
            columns = [
                "model_id",
                "model_name",
                "model_type",
                "architecture",
                "inputs",
                "outputs",
                "huggingface_config",
                "inference_code_location",
                "supported_backends",
                "hardware_requirements",
                "performance_metrics",
                "tags",
                "source_url",
                "license",
                "description",
                "model_card",
                "repository_structure",
                "model_cid",
                "config_cid",
                "tokenizer_cid",
                "artifact_cid",
                "model_revision",
                "revision_id",
                "revision_created_at",
                "parent_model_id",
                "parent_model_cid",
                "last_used_at",
                "last_inference_cid",
                "last_run_id",
                "inference_count",
                "created_at",
                "updated_at",
                "serving_config",
            ]

            for model_id, metadata in self.models.items():
                # Convert to dictionary and handle special fields
                data = asdict(metadata)
                
                # Convert inputs/outputs DataType enums to strings before JSON serialization
                for io_list in [data['inputs'], data['outputs']]:
                    for io_spec in io_list:
                        if 'data_type' in io_spec and hasattr(io_spec['data_type'], 'value'):
                            io_spec['data_type'] = io_spec['data_type'].value
                
                # inputs and outputs are already converted to dicts by asdict()
                # Just need to convert them to JSON strings
                data['inputs'] = json.dumps(data['inputs'])
                data['outputs'] = json.dumps(data['outputs'])
                
                # Convert other complex fields to JSON
                for json_field in ['huggingface_config', 'supported_backends', 
                                 'hardware_requirements', 'performance_metrics', 'tags',
                                 'repository_structure', 'serving_config']:
                    if data.get(json_field) is not None:
                        data[json_field] = json.dumps(data[json_field])
                
                # Convert enum to string
                data['model_type'] = data['model_type'].value
                
                # Update timestamp
                data['updated_at'] = datetime.now()
                
                # Insert or update
                placeholders = ", ".join(["?"] * len(columns))
                col_list = ", ".join(columns)
                self.con.execute(
                    f"INSERT OR REPLACE INTO model_metadata ({col_list}) VALUES ({placeholders})",
                    tuple(data.get(column) for column in columns),
                )
            
            logger.info(f"Saved {len(self.models)} models to database")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def _save_to_json(self):
        """Save model metadata to JSON file (with distributed storage when available)."""
        try:
            # Convert models to JSON-serializable format
            data = {}
            for model_id, metadata in self.models.items():
                model_dict = asdict(metadata)
                
                # Convert datetime objects to ISO strings
                if model_dict['created_at']:
                    model_dict['created_at'] = model_dict['created_at'].isoformat()
                if model_dict['updated_at']:
                    model_dict['updated_at'] = model_dict['updated_at'].isoformat()
                if model_dict['revision_created_at']:
                    model_dict['revision_created_at'] = model_dict['revision_created_at'].isoformat()
                
                # Convert enum to string
                model_dict['model_type'] = model_dict['model_type'].value
                
                data[model_id] = model_dict
            
            json_str = json.dumps(data, indent=2, default=str)
            
            # Try distributed storage first if available
            if self._storage_wrapper and self._storage_wrapper.is_distributed:
                try:
                    filename = os.path.basename(self.json_path)
                    cid = self._storage_wrapper.write_file(
                        json_str,
                        filename=filename,
                        pin=True  # Pin model metadata
                    )
                    logger.info(f"Saved {len(self.models)} models to distributed storage (CID: {cid[:16]}...)")
                    
                    # Also save locally for backward compatibility
                    os.makedirs(os.path.dirname(self.json_path) or '.', exist_ok=True)
                    with open(self.json_path, 'w') as f:
                        f.write(json_str)
                    return
                except Exception as e:
                    logger.debug(f"Failed to save to distributed storage, using local: {e}")
            
            # Fallback to local filesystem
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.json_path) or '.', exist_ok=True)
            
            # Write to file
            with open(self.json_path, 'w') as f:
                f.write(json_str)
            
            logger.info(f"Saved {len(self.models)} models to JSON file")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
    
    def add_model(self, metadata: ModelMetadata) -> bool:
        """
        Add or update a model in the registry.
        
        Args:
            metadata: ModelMetadata object
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            metadata.updated_at = datetime.now()
            if not metadata.model_revision:
                metadata.model_revision = metadata.updated_at.isoformat()
            if not metadata.revision_id:
                metadata.revision_id = metadata.model_revision
            if metadata.revision_created_at is None:
                metadata.revision_created_at = metadata.updated_at
            self.models[metadata.model_id] = metadata
            self._save_data()
            logger.info(f"Added/updated model: {metadata.model_id}")

            self._record_model_registration(metadata)

            # Update knowledge graph node (Gap 2)
            self._update_knowledge_graph_for_model(metadata)

            # Register serving config in service registry (Gap 3)
            if metadata.serving_config and self._artifact_storage:
                try:
                    self._artifact_storage.register_model_service(
                        metadata.model_id, metadata.serving_config
                    )
                except Exception as e:
                    logger.debug("Service registry update skipped: %s", e)
            
            return True
        except Exception as e:
            logger.error(f"Error adding model {metadata.model_id}: {e}")
            return False
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelMetadata object or None if not found
        """
        result = self.models.get(model_id)
        
        # Log model access for audit trail
        if result:
            self._record_model_access(result)
        
        return result

    def _record_model_registration(self, metadata: ModelMetadata) -> None:
        """Emit registration audit/provenance events for a model."""
        event_data = {
            "model_id": metadata.model_id,
            "model_type": metadata.model_type.value if hasattr(metadata.model_type, 'value') else str(metadata.model_type),
            "model_revision": metadata.model_revision,
            "revision_id": metadata.revision_id,
            "revision_created_at": metadata.revision_created_at.isoformat() if metadata.revision_created_at else None,
            "parent_model_id": metadata.parent_model_id,
            "model_cid": metadata.model_cid,
            "config_cid": metadata.config_cid,
            "tokenizer_cid": metadata.tokenizer_cid,
            "artifact_cid": metadata.artifact_cid,
            "timestamp": metadata.updated_at.isoformat() if metadata.updated_at else None,
            "last_run_id": metadata.last_run_id,
            "inference_count": metadata.inference_count,
        }

        if self._datasets_manager:
            try:
                self._datasets_manager.log_event("model_registered", event_data, level="INFO", category="GENERAL")
                self._datasets_manager.track_provenance("model_registration", event_data)
            except Exception as e:
                logger.debug(f"Model registration audit logging failed: {e}")

        if self._provenance_logger:
            try:
                self._provenance_logger.log_transformation(
                    operation="model_registered",
                    data={
                        "model_id": metadata.model_id,
                        "model_type": metadata.model_type.value if hasattr(metadata.model_type, 'value') else str(metadata.model_type),
                        "input_types": [inp.data_type.value if hasattr(inp.data_type, 'value') else str(inp.data_type) for inp in metadata.inputs] if metadata.inputs else [],
                        "output_types": [out.data_type.value if hasattr(out.data_type, 'value') else str(out.data_type) for out in metadata.outputs] if metadata.outputs else [],
                        "model_revision": metadata.model_revision,
                        "revision_id": metadata.revision_id,
                        "revision_created_at": metadata.revision_created_at.isoformat() if metadata.revision_created_at else None,
                        "parent_model_id": metadata.parent_model_id,
                        "model_cid": metadata.model_cid,
                        "config_cid": metadata.config_cid,
                        "tokenizer_cid": metadata.tokenizer_cid,
                        "artifact_cid": metadata.artifact_cid,
                        "timestamp": metadata.updated_at.isoformat() if metadata.updated_at else None,
                    }
                )
            except Exception as e:
                logger.debug(f"Provenance logging failed: {e}")

    def _record_model_access(self, metadata: ModelMetadata) -> None:
        """Emit model access audit/provenance events without mutating state."""
        event_data = {
            "model_id": metadata.model_id,
            "model_type": metadata.model_type.value if hasattr(metadata.model_type, 'value') else str(metadata.model_type),
            "model_revision": metadata.model_revision,
            "revision_id": metadata.revision_id,
            "revision_created_at": metadata.revision_created_at.isoformat() if metadata.revision_created_at else None,
            "parent_model_id": metadata.parent_model_id,
            "model_cid": metadata.model_cid,
            "artifact_cid": metadata.artifact_cid,
            "last_inference_cid": metadata.last_inference_cid,
            "last_run_id": metadata.last_run_id,
            "inference_count": metadata.inference_count,
        }

        if self._datasets_manager:
            try:
                self._datasets_manager.log_event("model_accessed", event_data, level="INFO", category="GENERAL")
                self._datasets_manager.track_provenance("model_access", event_data)
            except Exception as e:
                logger.debug(f"Event logging failed: {e}")

        if self._provenance_logger:
            try:
                self._provenance_logger.log_transformation(
                    operation="model_accessed",
                    data=event_data,
                )
            except Exception as e:
                logger.debug(f"Model access provenance logging failed: {e}")

    def _record_model_usage(self, metadata: ModelMetadata) -> None:
        """Emit normalized model usage linkage events after inference."""
        event_data = {
            "model_id": metadata.model_id,
            "model_type": metadata.model_type.value if hasattr(metadata.model_type, 'value') else str(metadata.model_type),
            "model_revision": metadata.model_revision,
            "revision_id": metadata.revision_id,
            "parent_model_id": metadata.parent_model_id,
            "last_used_at": metadata.last_used_at.isoformat() if metadata.last_used_at else None,
            "last_inference_cid": metadata.last_inference_cid,
            "last_run_id": metadata.last_run_id,
            "inference_count": metadata.inference_count,
            "status": "usage_linked",
        }

        if self._datasets_manager:
            try:
                self._datasets_manager.log_event("model_inference_linked", event_data, level="INFO", category="GENERAL")
                self._datasets_manager.track_provenance("model_usage", event_data)
            except Exception as e:
                logger.debug(f"Model usage audit logging failed: {e}")

        if self._provenance_logger:
            try:
                self._provenance_logger.log_transformation(
                    operation="model_inference_linked",
                    data=event_data,
                )
            except Exception as e:
                logger.debug(f"Model usage provenance logging failed: {e}")

    def mark_model_used(
        self,
        model_id: str,
        inference_cid: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> bool:
        """Update usage linkage metadata for a model after inference or evaluation."""
        metadata = self.models.get(model_id)
        if not metadata:
            return False

        metadata.last_used_at = datetime.now()
        if inference_cid:
            metadata.last_inference_cid = inference_cid
        if run_id:
            metadata.last_run_id = run_id
        metadata.inference_count = int(metadata.inference_count or 0) + 1
        metadata.updated_at = datetime.now()
        self._save_data()
        self._record_model_usage(metadata)
        return True
    
    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if model_id in self.models:
                del self.models[model_id]
                self._save_data()
                
                # Also remove from database if using database backend
                if self.use_database:
                    self.con.execute("DELETE FROM model_metadata WHERE model_id = ?", (model_id,))

                # Remove from knowledge graph (Gap 2)
                if self._knowledge_graph:
                    try:
                        self._knowledge_graph.remove_model_node(model_id)
                    except Exception as e:
                        logger.debug("Knowledge graph node removal skipped: %s", e)
                
                logger.info(f"Removed model: {model_id}")
                return True
            else:
                logger.warning(f"Model not found: {model_id}")
                return False
        except Exception as e:
            logger.error(f"Error removing model {model_id}: {e}")
            return False
    
    def store_model_to_ipfs(self, model_path: str, model_id: str = None) -> Optional[str]:
        """
        Store model weights to IPFS and return CID.
        
        This uses the IPFS backend router with preference for:
        1. ipfs_kit_py (distributed storage)
        2. HuggingFace cache (local storage with IPFS-like addressing)
        3. Kubo CLI (fallback)
        
        Args:
            model_path: Path to model weights/directory
            model_id: Optional model identifier for logging
            
        Returns:
            CID string if successful, None otherwise
        """
        if not self._ipfs_backend and not self._artifact_storage:
            logger.warning("IPFS backend not available for model storage")
            return None
        
        try:
            path = Path(model_path)
            if self._artifact_storage and path.is_file():
                cid = self._artifact_storage.store(path, filename=path.name, pin=True)
            else:
                cid = self._ipfs_backend.add_path(
                    model_path,
                    recursive=True,
                    pin=True
                )
            logger.info(f"Stored model {model_id or model_path} to IPFS: {cid}")
            return cid
        except Exception as e:
            logger.error(f"Failed to store model to IPFS: {e}")
            return None

    def _store_artifact_file(self, artifact_path: Optional[str], artifact_name: str) -> Optional[str]:
        """Store a single artifact file through IPFSKitStorage when available."""
        if not artifact_path:
            return None

        path = Path(artifact_path)
        if not path.exists():
            logger.warning("Artifact path not found: %s", artifact_path)
            return None

        if self._artifact_storage and path.is_file():
            try:
                return self._artifact_storage.store(path, filename=artifact_name, pin=True)
            except Exception as e:
                logger.debug("IPFS Kit storage failed for %s: %s", artifact_name, e)

        if self._ipfs_backend and path.is_dir():
            try:
                return self._ipfs_backend.add_path(str(path), recursive=True, pin=True)
            except Exception as e:
                logger.debug("IPFS backend failed for %s: %s", artifact_name, e)

        if self._artifact_storage:
            try:
                return self._artifact_storage.store(path.read_bytes(), filename=artifact_name, pin=True)
            except Exception as e:
                logger.debug("Byte-storage fallback failed for %s: %s", artifact_name, e)

        return None

    def _store_artifact_manifest(self, metadata: ModelMetadata, artifact_map: Dict[str, Optional[str]]) -> Optional[str]:
        """Persist a small manifest that ties the model assets together."""
        if not self._artifact_storage:
            return None

        manifest = {
            "model_id": metadata.model_id,
            "model_name": metadata.model_name,
            "model_cid": artifact_map.get("model_cid"),
            "config_cid": artifact_map.get("config_cid"),
            "tokenizer_cid": artifact_map.get("tokenizer_cid"),
            "created_at": datetime.now().isoformat(),
        }

        try:
            return self._artifact_storage.store(
                json.dumps(manifest, sort_keys=True),
                filename=f"{metadata.model_id.replace('/', '_')}.artifact-manifest.json",
                pin=True,
            )
        except Exception as e:
            logger.debug("Artifact manifest storage failed: %s", e)
            return None

    def restore_model_artifacts_from_cids(self, model_id: str, output_dir: str) -> Dict[str, str]:
        """Restore stored model artifacts into a local directory."""
        metadata = self.get_model(model_id)
        if not metadata or not self._artifact_storage:
            return {}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        restored_paths: Dict[str, str] = {}
        artifact_specs = [
            ("model_cid", f"{metadata.model_id.replace('/', '_')}.model"),
            ("config_cid", f"{metadata.model_id.replace('/', '_')}.config"),
            ("tokenizer_cid", f"{metadata.model_id.replace('/', '_')}.tokenizer"),
        ]

        for field_name, filename in artifact_specs:
            cid = getattr(metadata, field_name, None)
            if not cid:
                continue

            try:
                payload = self._artifact_storage.retrieve(cid)
            except Exception as e:
                logger.debug("Artifact restore failed for %s: %s", field_name, e)
                continue

            if payload is None:
                continue

            if isinstance(payload, str):
                payload = payload.encode("utf-8")

            target_path = output_path / filename
            with open(target_path, "wb") as f:
                f.write(payload)
            restored_paths[field_name] = str(target_path)

        return restored_paths
    
    def retrieve_model_from_ipfs(self, cid: str, output_path: str, model_id: str = None) -> bool:
        """
        Retrieve model weights from IPFS by CID.
        
        Args:
            cid: Content identifier
            output_path: Where to save the model
            model_id: Optional model identifier for logging
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ipfs_backend:
            logger.warning("IPFS backend not available for model retrieval")
            return False
        
        try:
            self._ipfs_backend.get_to_path(cid, output_path=output_path)
            logger.info(f"Retrieved model {model_id or cid} from IPFS to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to retrieve model from IPFS: {e}")
            return False
    
    def add_model_with_ipfs_storage(
        self,
        metadata: ModelMetadata,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        store_to_ipfs: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Add model metadata and optionally store model artifacts to IPFS.
        
        Args:
            metadata: ModelMetadata object
            model_path: Optional path to model weights
            config_path: Optional path to model config
            tokenizer_path: Optional path to tokenizer files
            store_to_ipfs: Whether to store weights to IPFS
            
        Returns:
            Tuple of (success: bool, cid: Optional[str])
        """
        artifact_cids: Dict[str, Optional[str]] = {
            "model_cid": None,
            "config_cid": None,
            "tokenizer_cid": None,
        }

        if store_to_ipfs:
            safe_model_id = metadata.model_id.replace("/", "_")
            artifact_cids["model_cid"] = self._store_artifact_file(model_path, f"{safe_model_id}.model") if model_path else None
            artifact_cids["config_cid"] = self._store_artifact_file(config_path, f"{safe_model_id}.config") if config_path else None
            artifact_cids["tokenizer_cid"] = self._store_artifact_file(tokenizer_path, f"{safe_model_id}.tokenizer") if tokenizer_path else None

        metadata.model_cid = artifact_cids["model_cid"]
        metadata.config_cid = artifact_cids["config_cid"]
        metadata.tokenizer_cid = artifact_cids["tokenizer_cid"]
        metadata.artifact_cid = self._store_artifact_manifest(metadata, artifact_cids)

        if metadata.artifact_cid:
            if metadata.repository_structure is None:
                metadata.repository_structure = {}
            metadata.repository_structure["artifact_cid"] = metadata.artifact_cid

        if not self.add_model(metadata):
            return False, None

        return True, metadata.artifact_cid or metadata.model_cid
    
    def list_models(self, model_type: Optional[ModelType] = None,
                   architecture: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """
        List models with optional filtering.
        
        Args:
            model_type: Filter by model type
            architecture: Filter by architecture
            tags: Filter by tags (model must have all specified tags)
            
        Returns:
            List of ModelMetadata objects
        """
        results = []
        
        for metadata in self.models.values():
            # Apply filters
            if model_type and metadata.model_type != model_type:
                continue
            if architecture and metadata.architecture != architecture:
                continue
            if tags and not all(tag in metadata.tags for tag in tags):
                continue
                
            results.append(metadata)
        
        return results

    @staticmethod
    def _served_model_endpoints(endpoint_url: Optional[str] = None) -> List[str]:
        """Return configured OpenAI-compatible inference API base URLs."""
        if endpoint_url:
            candidates = [endpoint_url]
        else:
            configured = os.getenv("IPFS_ACCELERATE_SERVED_MODEL_ENDPOINTS", "")
            candidates = [value.strip() for value in configured.split(",") if value.strip()]
            if not candidates:
                candidates = [
                    os.getenv("IPFS_ACCELERATE_LLAMA_CPP_BASE_URL", "").strip()
                    or os.getenv("IPFS_ACCELERATE_PY_LLAMA_CPP_BASE_URL", "").strip()
                    or "http://127.0.0.1:8080/v1"
                ]
        return [candidate.rstrip("/") for candidate in candidates]

    def list_served_models(
        self,
        endpoint_url: Optional[str] = None,
        timeout: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Discover models currently exposed by OpenAI-compatible servers.

        Unreachable endpoints are skipped so model discovery remains a safe MCP
        health operation. Configure multiple endpoints with
        ``IPFS_ACCELERATE_SERVED_MODEL_ENDPOINTS``.
        """
        served: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()
        for base_url in self._served_model_endpoints(endpoint_url):
            models_url = (
                f"{base_url}/models"
                if base_url.endswith("/v1")
                else f"{base_url}/v1/models"
            )
            try:
                request = urllib.request.Request(
                    models_url,
                    headers={"Accept": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (OSError, ValueError, urllib.error.URLError) as exc:
                logger.debug("Served-model discovery failed for %s: %s", base_url, exc)
                continue

            records = payload.get("data") or payload.get("models") or []
            if isinstance(records, dict):
                records = list(records.values())
            for raw in records:
                if isinstance(raw, str):
                    raw = {"id": raw}
                if not isinstance(raw, dict):
                    continue
                model_id = str(raw.get("id") or raw.get("model") or raw.get("name") or "")
                if not model_id or (base_url, model_id) in seen:
                    continue
                seen.add((base_url, model_id))
                served.append({
                    "id": model_id,
                    "model_id": model_id,
                    "name": str(raw.get("name") or model_id),
                    "provider": str(raw.get("owned_by") or "llama_cpp"),
                    "endpoint": base_url,
                    "status": "available",
                    "served": True,
                    "capabilities": raw.get("capabilities") or ["text-generation"],
                    "metadata": raw.get("meta") or {},
                })
        return served

    def get_served_model(
        self,
        model_id: str,
        endpoint_url: Optional[str] = None,
        timeout: float = 2.0,
    ) -> Optional[Dict[str, Any]]:
        """Return a currently served model by ID or configured Leanstral alias."""
        aliases = {"leanstral", "leanstral_local", "labs-leanstral-1-5"}
        models = self.list_served_models(endpoint_url=endpoint_url, timeout=timeout)
        for model in models:
            if model["id"] == model_id:
                return model
        if model_id.lower() in aliases and len(models) == 1:
            model = dict(models[0])
            model["requested_alias"] = model_id
            return model
        return None
    
    def search_models(self, query: str) -> List[ModelMetadata]:
        """
        Search models by name, description, or tags.
        
        Args:
            query: Search query string
            
        Returns:
            List of ModelMetadata objects matching the query
        """
        query_lower = query.lower()
        results = []
        
        for metadata in self.models.values():
            # Search in name, description, and tags
            if (query_lower in metadata.model_name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                results.append(metadata)
        
        return results
    
    def get_models_by_input_type(self, input_type: DataType) -> List[ModelMetadata]:
        """
        Get models that accept a specific input type.
        
        Args:
            input_type: Input data type to search for
            
        Returns:
            List of ModelMetadata objects
        """
        results = []
        
        for metadata in self.models.values():
            if any(inp.data_type == input_type for inp in metadata.inputs):
                results.append(metadata)
        
        return results
    
    def get_models_by_output_type(self, output_type: DataType) -> List[ModelMetadata]:
        """
        Get models that produce a specific output type.
        
        Args:
            output_type: Output data type to search for
            
        Returns:
            List of ModelMetadata objects
        """
        results = []
        
        for metadata in self.models.values():
            if any(out.data_type == output_type for out in metadata.outputs):
                results.append(metadata)
        
        return results
    
    def get_compatible_models(self, input_type: DataType, output_type: DataType) -> List[ModelMetadata]:
        """
        Get models compatible with specific input and output types.
        
        Args:
            input_type: Required input data type
            output_type: Required output data type
            
        Returns:
            List of ModelMetadata objects
        """
        results = []
        
        for metadata in self.models.values():
            has_input = any(inp.data_type == input_type for inp in metadata.inputs)
            has_output = any(out.data_type == output_type for out in metadata.outputs)
            
            if has_input and has_output:
                results.append(metadata)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model registry.
        
        Returns:
            Dictionary containing various statistics
        """
        if not self.models:
            return {"total_models": 0}
        
        stats = {
            "total_models": len(self.models),
            "models_by_type": {},
            "models_by_architecture": {},
            "common_input_types": {},
            "common_output_types": {},
            "models_with_hf_config": 0,
            "models_with_inference_code": 0,
            "models_with_repo_structure": 0,
            "total_tracked_files": 0,
            "models_with_serving_config": 0,
            "knowledge_graph_cid": self._knowledge_graph.graph_cid if self._knowledge_graph else None,
        }
        
        # Count by type and architecture
        for metadata in self.models.values():
            # Model type stats
            type_name = metadata.model_type.value if hasattr(metadata.model_type, 'value') else str(metadata.model_type)
            stats["models_by_type"][type_name] = stats["models_by_type"].get(type_name, 0) + 1
            
            # Architecture stats
            arch = metadata.architecture
            stats["models_by_architecture"][arch] = stats["models_by_architecture"].get(arch, 0) + 1
            
            # Input/Output type stats
            for inp in metadata.inputs:
                input_type = inp.data_type.value if hasattr(inp.data_type, 'value') else str(inp.data_type)
                stats["common_input_types"][input_type] = stats["common_input_types"].get(input_type, 0) + 1
            
            for out in metadata.outputs:
                output_type = out.data_type.value if hasattr(out.data_type, 'value') else str(out.data_type)
                stats["common_output_types"][output_type] = stats["common_output_types"].get(output_type, 0) + 1
            
            # Config and inference code stats
            if metadata.huggingface_config:
                stats["models_with_hf_config"] += 1
            if metadata.inference_code_location:
                stats["models_with_inference_code"] += 1
            if metadata.repository_structure:
                stats["models_with_repo_structure"] += 1
                stats["total_tracked_files"] += metadata.repository_structure.get("total_files", 0)
            if metadata.serving_config:
                stats["models_with_serving_config"] += 1
        
        return stats

    # ------------------------------------------------------------------
    # Gap 1 – Model Weights Caching
    # ------------------------------------------------------------------

    def warm_cache(self, model_ids: List[str]) -> Dict[str, bool]:
        """
        Proactively warm the ipfs_kit_py disk-tier cache for a set of models.

        For each model in ``model_ids`` that has stored artifact CIDs, this
        method retrieves the weights, config, and tokenizer from IPFS into the
        local disk cache so that subsequent calls to ``get_cached_weight_path``
        can be served without a network round-trip.

        Parameters
        ----------
        model_ids:
            List of model identifiers to warm.

        Returns
        -------
        Dict mapping model_id → True (warmed) / False (failed or skipped).
        """
        results: Dict[str, bool] = {}
        if not self._artifact_storage:
            logger.warning("warm_cache: no artifact storage available")
            return {mid: False for mid in model_ids}

        for model_id in model_ids:
            metadata = self.models.get(model_id)
            if not metadata:
                logger.warning("warm_cache: model not found: %s", model_id)
                results[model_id] = False
                continue

            warmed = False
            for cid_attr in ("model_cid", "config_cid", "tokenizer_cid"):
                cid = getattr(metadata, cid_attr, None)
                if not cid:
                    continue
                # Check if already cached
                if self._artifact_storage.exists(cid):
                    warmed = True
                    continue
                # Fetch from IPFS into disk cache
                try:
                    data = self._artifact_storage.retrieve(cid)
                    if data is not None:
                        warmed = True
                        logger.debug("warm_cache: fetched %s for %s", cid_attr, model_id)
                except Exception as e:
                    logger.debug("warm_cache: failed for %s/%s: %s", model_id, cid_attr, e)

            results[model_id] = warmed
            logger.info("warm_cache: model=%s warmed=%s", model_id, warmed)

        return results

    def get_cached_weight_path(
        self,
        model_id: str,
        file: str = "model.safetensors",
    ) -> Optional[str]:
        """
        Return a local filesystem path for a model weight file.

        If the weight file is already in the local disk cache it is returned
        immediately.  Otherwise, the method attempts to pull it from IPFS
        (using the CID stored on the model's ``ModelMetadata``) and saves it
        to the cache directory before returning the path.

        Parameters
        ----------
        model_id:
            Unique model identifier.
        file:
            Weight filename to look up (default: ``"model.safetensors"``).

        Returns
        -------
        Absolute path to the cached file, or None if unavailable.
        """
        if not self._artifact_storage:
            return None

        metadata = self.models.get(model_id)
        if not metadata:
            return None

        # Map canonical filenames to the CID attributes stored on ModelMetadata
        cid_map = {
            "model.safetensors": metadata.model_cid,
            "pytorch_model.bin": metadata.model_cid,
            "model.bin": metadata.model_cid,
            "config.json": metadata.config_cid,
            "tokenizer.json": metadata.tokenizer_cid,
            "tokenizer_config.json": metadata.tokenizer_cid,
        }

        # Determine which CID to use
        cid = cid_map.get(file)
        if not cid:
            # Fall back to the per-file CIDs stored in repository_structure
            if metadata.repository_structure:
                file_info = metadata.repository_structure.get("files", {}).get(file, {})
                cid = file_info.get("ipfs_cid") or file_info.get("oid")

        if not cid:
            logger.warning("get_cached_weight_path: no CID for %s/%s", model_id, file)
            return None

        # Check local cache first
        cache_path = self._artifact_storage.cache_dir / cid
        if cache_path.exists():
            return str(cache_path)

        # Fetch from IPFS / fallback storage
        try:
            data = self._artifact_storage.retrieve(cid)
            if data is None:
                return None
            cache_path.write_bytes(data)
            logger.info("get_cached_weight_path: cached %s for %s", file, model_id)
            return str(cache_path)
        except Exception as e:
            logger.warning("get_cached_weight_path: retrieve failed for %s: %s", cid, e)
            return None

    # ------------------------------------------------------------------
    # Gap 2 – Knowledge Graph / GraphRAG
    # ------------------------------------------------------------------

    def _update_knowledge_graph_for_model(self, metadata: "ModelMetadata") -> None:
        """Internal: sync a model into the knowledge graph."""
        if not self._knowledge_graph:
            return
        try:
            model_data = {
                "model_name": metadata.model_name,
                "model_type": metadata.model_type.value if hasattr(metadata.model_type, "value") else str(metadata.model_type),
                "architecture": metadata.architecture,
                "description": metadata.description,
                "model_card": metadata.model_card or "",
                "tags": metadata.tags or [],
                "source_url": metadata.source_url or "",
            }
            self._knowledge_graph.add_model_node(metadata.model_id, model_data)

            # Lineage edge
            if metadata.parent_model_id:
                self._knowledge_graph.add_lineage_edge(metadata.model_id, metadata.parent_model_id)

            # Backend compatibility edges
            if metadata.supported_backends:
                self._knowledge_graph.add_compatibility_edges(
                    metadata.model_id, metadata.supported_backends
                )

            # Hardware requirement edges
            if metadata.hardware_requirements:
                self._knowledge_graph.add_hardware_requirement_edges(
                    metadata.model_id, metadata.hardware_requirements
                )

            # Pipeline / task edges derived from serving_config or HF config
            pipeline_types: List[str] = []
            if metadata.serving_config:
                pt = metadata.serving_config.get("pipeline_types")
                if pt and isinstance(pt, list):
                    pipeline_types = pt
            if not pipeline_types and metadata.huggingface_config:
                pt = metadata.huggingface_config.get("pipeline_tag")
                if pt:
                    pipeline_types = [pt]
            if pipeline_types:
                self._knowledge_graph.add_pipeline_edges(metadata.model_id, pipeline_types)

        except Exception as e:
            logger.debug("Knowledge graph update failed for %s: %s", metadata.model_id, e)

    def build_model_graph(self) -> bool:
        """
        Build (or rebuild) the knowledge graph from all registered models.

        Iterates over every model in the registry, adds its node and
        relationships to the graph, and optionally persists the result to
        IPFS.

        Returns
        -------
        True on success, False if the knowledge graph is not available.
        """
        if not self._knowledge_graph:
            logger.warning("build_model_graph: knowledge graph not available")
            return False

        for metadata in self.models.values():
            self._update_knowledge_graph_for_model(metadata)

        # Persist to IPFS if storage is available
        cid = self._knowledge_graph.persist_to_ipfs()
        if cid:
            logger.info("build_model_graph: graph persisted to IPFS CID: %s", cid)

        logger.info("build_model_graph: built graph with %d models", len(self.models))
        return True

    def query_model_graph(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph for models and related entities.

        Uses ipfs_datasets_py's graph query layer when available, otherwise
        performs keyword search across entity IDs and properties.

        Example queries::

            "models fine-tuned from llama"
            "cuda compatible"
            "text-generation"

        Parameters
        ----------
        query:
            Natural-language or keyword query string.

        Returns
        -------
        List of result dicts, each containing at minimum ``entity_id`` and
        ``type`` fields.
        """
        if not self._knowledge_graph:
            logger.warning("query_model_graph: knowledge graph not available")
            return []
        return self._knowledge_graph.query(query)

    # ------------------------------------------------------------------
    # Gap 3 – Serving Configuration
    # ------------------------------------------------------------------

    def get_serving_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the serving configuration for a model.

        Checks the in-memory registry first, then falls back to the
        ipfs_kit_py service registry.

        Parameters
        ----------
        model_id:
            Unique model identifier.

        Returns
        -------
        Serving config dict, or None if not set.
        """
        metadata = self.models.get(model_id)
        if metadata and metadata.serving_config:
            return metadata.serving_config

        # Fallback: service registry
        if self._artifact_storage:
            try:
                svc = self._artifact_storage.get_model_service_config(model_id)
                if svc:
                    return svc.get("serving_config")
            except Exception as e:
                logger.debug("Service registry lookup failed: %s", e)

        return None

    def update_serving_config(
        self,
        model_id: str,
        config: Union[Dict[str, Any], "ServingConfig"],
    ) -> bool:
        """
        Update the serving configuration for a registered model.

        Persists the config to the model metadata store and the
        ipfs_kit_py service registry.

        Parameters
        ----------
        model_id:
            Unique model identifier.
        config:
            Either a ``ServingConfig`` instance or a plain dict.

        Returns
        -------
        True on success, False if the model is not registered.
        """
        metadata = self.models.get(model_id)
        if not metadata:
            logger.warning("update_serving_config: model not found: %s", model_id)
            return False

        cfg_dict = config.to_dict() if isinstance(config, ServingConfig) else config
        metadata.serving_config = cfg_dict
        metadata.updated_at = datetime.now()
        self._save_data()

        # Propagate to service registry
        if self._artifact_storage:
            try:
                self._artifact_storage.register_model_service(model_id, cfg_dict)
            except Exception as e:
                logger.debug("Service registry update skipped: %s", e)

        logger.info("Updated serving config for model: %s", model_id)
        return True

    def resolve_launch_command(self, model_id: str) -> Optional[List[str]]:
        """
        Build a ready-to-execute subprocess command from a model's serving config.

        The command is constructed for the engine specified in ``ServingConfig``
        (``engine`` field).  Supported engines and their command templates:

        - ``vllm``:         ``python -m vllm.entrypoints.openai.api_server --model <id> <flags>``
        - ``tgi``:          ``text-generation-launcher --model-id <id> <flags>``
        - ``triton``:       ``tritonserver --model-repository <path> <flags>``
        - ``onnxruntime``:  ``python -m onnxruntime_server --model_path <path> <flags>``
        - ``llama.cpp``:    ``./server -m <model_path> <flags>``
        - ``hf_pipeline``:  ``python -m ipfs_accelerate_py.hf_model_server.server --model-id <id> <flags>``

        ``launch_args`` keys prefixed with ``--`` are forwarded directly;
        plain keys are prefixed automatically.

        Parameters
        ----------
        model_id:
            Unique model identifier.

        Returns
        -------
        List of command tokens suitable for ``subprocess.Popen``, or None if
        the model has no serving config.
        """
        cfg = self.get_serving_config(model_id)
        if not cfg:
            logger.warning("resolve_launch_command: no serving config for %s", model_id)
            return None
        resolved_command = cfg.get("resolved_command")
        if isinstance(resolved_command, list) and resolved_command:
            return [str(part) for part in resolved_command if str(part).strip()]

        engine = cfg.get("engine", "hf_pipeline")
        launch_args: Dict[str, Any] = cfg.get("launch_args") or {}

        def _flatten_args(args: Dict[str, Any]) -> List[str]:
            tokens: List[str] = []
            for k, v in args.items():
                # Accept keys already starting with - or --; otherwise prefix with --
                if k.startswith("-"):
                    flag = k
                else:
                    flag = f"--{k}"
                if isinstance(v, bool):
                    if v:
                        tokens.append(flag)
                elif v is not None:
                    tokens.extend([flag, str(v)])
            return tokens

        extra = _flatten_args(launch_args)

        ENGINE_TEMPLATES: Dict[str, List[str]] = {
            "vllm": ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", model_id],
            "tgi": ["text-generation-launcher", "--model-id", model_id],
            "triton": ["tritonserver"],
            "onnxruntime": ["python", "-m", "onnxruntime_server", "--model_id", model_id],
            "llama.cpp": ["./server", "-m", model_id],
            "hf_pipeline": [
                "python", "-m", "ipfs_accelerate_py.hf_model_server.server",
                "--model-id", model_id,
            ],
        }

        base = ENGINE_TEMPLATES.get(engine, ENGINE_TEMPLATES["hf_pipeline"])
        cmd = base + extra
        logger.info("resolve_launch_command: %s -> %s", model_id, " ".join(cmd))
        return cmd


    
    def get_models_with_file(self, file_pattern: str) -> List[ModelMetadata]:
        """
        Get models that contain files matching a pattern in their repository structure.
        
        Args:
            file_pattern: Pattern to match against file paths (case-insensitive)
            
        Returns:
            List of ModelMetadata objects
        """
        results = []
        pattern_lower = file_pattern.lower()
        
        for metadata in self.models.values():
            if not metadata.repository_structure or "files" not in metadata.repository_structure:
                continue
            
            for file_path in metadata.repository_structure["files"].keys():
                if pattern_lower in file_path.lower():
                    results.append(metadata)
                    break  # Found at least one matching file
        
        return results
    
    def get_model_file_hash(self, model_id: str, file_path: str) -> Optional[str]:
        """
        Get the hash for a specific file in a model's repository.
        
        Args:
            model_id: Model identifier
            file_path: Path to the file
            
        Returns:
            File hash/OID or None if not found
        """
        metadata = self.get_model(model_id)
        if not metadata or not metadata.repository_structure:
            return None
        
        return get_file_hash_from_structure(metadata.repository_structure, file_path)
    
    def refresh_repository_structure(self, model_id: str, branch: str = "main", include_ipfs_cids: bool = True) -> bool:
        """
        Refresh the repository structure for a specific model with IPFS support.
        
        Args:
            model_id: Model identifier
            branch: Git branch to fetch from
            include_ipfs_cids: Whether to generate IPFS CIDs for files
            
        Returns:
            True if successful, False otherwise
        """
        metadata = self.get_model(model_id)
        if not metadata:
            logger.error(f"Model not found: {model_id}")
            return False
        
        logger.info(f"Refreshing repository structure for {model_id}")
        new_structure = fetch_huggingface_repo_structure(model_id, branch, include_ipfs_cids)
        
        if new_structure:
            metadata.repository_structure = new_structure
            metadata.updated_at = datetime.now()
            self._save_data()
            logger.info(f"Successfully refreshed repository structure for {model_id}")
            if new_structure.get("ipfs_enabled"):
                cid_count = sum(1 for f in new_structure["files"].values() if "ipfs_cid" in f)
                logger.info(f"Generated IPFS CIDs for {cid_count} files")
            return True
        else:
            logger.error(f"Failed to refresh repository structure for {model_id}")
            return False
    
    def get_model_file_ipfs_cid(self, model_id: str, file_path: str) -> Optional[str]:
        """
        Get the IPFS CID for a specific file in a model's repository.
        
        Args:
            model_id: Model identifier
            file_path: Path to the file within the repository
            
        Returns:
            IPFS CID string or None if not found
        """
        metadata = self.get_model(model_id)
        if not metadata or not metadata.repository_structure:
            return None
        
        files = metadata.repository_structure.get("files", {})
        file_info = files.get(file_path)
        if file_info:
            return file_info.get("ipfs_cid")
        
        return None
    
    def get_models_with_ipfs_cids(self) -> List[ModelMetadata]:
        """
        Get all models that have IPFS CIDs in their repository structure.
        
        Returns:
            List of models with IPFS content addressing
        """
        models_with_cids = []
        
        for model in self.models.values():
            if (model.repository_structure and 
                model.repository_structure.get("ipfs_enabled") and
                any("ipfs_cid" in f for f in model.repository_structure.get("files", {}).values())):
                models_with_cids.append(model)
        
        return models_with_cids
    
    def get_ipfs_gateway_urls(self, model_id: str, gateway_base: str = "https://ipfs.io/ipfs/") -> Dict[str, str]:
        """
        Get IPFS gateway URLs for all files in a model's repository.
        
        Args:
            model_id: Model identifier
            gateway_base: Base URL for the IPFS gateway
            
        Returns:
            Dictionary mapping file paths to IPFS gateway URLs
        """
        metadata = self.get_model(model_id)
        if not metadata or not metadata.repository_structure:
            return {}
        
        gateway_urls = {}
        files = metadata.repository_structure.get("files", {})
        
        for file_path, file_info in files.items():
            if "ipfs_cid" in file_info:
                gateway_urls[file_path] = f"{gateway_base}{file_info['ipfs_cid']}"
        
        return gateway_urls
    
    def export_metadata(self, output_path: str, format: str = "json") -> bool:
        """
        Export model metadata to a file.
        
        Args:
            output_path: Output file path
            format: Export format ("json" or "yaml")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if format.lower() == "json":
                self._save_to_json()
                # Copy to the specified path
                import shutil
                shutil.copy2(self.json_path, output_path)
            elif format.lower() == "yaml":
                try:
                    import yaml
                    data = {}
                    for model_id, metadata in self.models.items():
                        model_dict = asdict(metadata)
                        # Convert datetime objects to ISO strings
                        if model_dict['created_at']:
                            model_dict['created_at'] = model_dict['created_at'].isoformat()
                        if model_dict['updated_at']:
                            model_dict['updated_at'] = model_dict['updated_at'].isoformat()
                        # Convert enum to string
                        model_dict['model_type'] = model_dict['model_type'].value
                        data[model_id] = model_dict
                    
                    with open(output_path, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False, indent=2)
                except ImportError:
                    logger.error("PyYAML not installed. Cannot export to YAML format.")
                    return False
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported model metadata to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting metadata: {e}")
            return False
    
    def get_models_by_pipeline_type(self, pipeline_type: str, 
                                     include_api: bool = True,
                                     include_self_hosted: bool = True,
                                     provider_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get all models (self-hosted + API) that support a specific pipeline type.
        
        This method provides unified access to both self-hosted models tracked in the
        model manager and API-based models from the API model registry.
        
        Args:
            pipeline_type: HuggingFace pipeline type (e.g., "text-generation", "text-classification")
            include_api: Whether to include API models (default: True)
            include_self_hosted: Whether to include self-hosted models (default: True)
            provider_filter: Optional list of API providers to include (e.g., ["openai", "anthropic"])
            
        Returns:
            List of model dictionaries with unified schema containing:
                - model_id: Unique identifier
                - model_name: Display name
                - source_type: "self-hosted" or "api"
                - provider: API provider name (for API models) or "self-hosted"
                - pipeline_types: List of supported pipeline types
                - context_length: Maximum context length (if available)
                - architecture: Model architecture
                - additional metadata specific to source type
        """
        results = []
        
        # Add self-hosted models
        if include_self_hosted:
            # Try to import pipeline types for mapping
            try:
                from .common.pipeline_types import PipelineTypeMapper
                have_mapper = True
            except ImportError:
                have_mapper = False
                logger.warning("Pipeline type mapper not available for self-hosted model filtering")
            
            for model_id, metadata in self.models.items():
                # Check if this model supports the requested pipeline type
                supports_pipeline = False
                
                if have_mapper:
                    # Use architecture-based mapping
                    supports_pipeline = PipelineTypeMapper.supports_pipeline(
                        pipeline_type, 
                        architecture=metadata.architecture
                    )
                else:
                    # Fallback: basic heuristic matching
                    arch_lower = metadata.architecture.lower()
                    pipe_lower = pipeline_type.lower()
                    
                    # Simple heuristics for common cases
                    if "causal" in pipe_lower or "text-generation" in pipe_lower:
                        supports_pipeline = any(x in arch_lower for x in ['gpt', 'llama', 'mistral', 'falcon'])
                    elif "classification" in pipe_lower:
                        supports_pipeline = any(x in arch_lower for x in ['bert', 'roberta', 'distilbert'])
                    elif "summarization" in pipe_lower or "translation" in pipe_lower:
                        supports_pipeline = any(x in arch_lower for x in ['t5', 'bart', 'pegasus'])
                
                if supports_pipeline:
                    results.append({
                        'model_id': model_id,
                        'model_name': metadata.model_name,
                        'source_type': 'self-hosted',
                        'provider': 'self-hosted',
                        'pipeline_types': [pipeline_type],  # Could be enhanced with full type detection
                        'context_length': metadata.huggingface_config.get('max_position_embeddings') if metadata.huggingface_config else None,
                        'architecture': metadata.architecture,
                        'model_type': metadata.model_type.value,
                        'supported_backends': metadata.supported_backends,
                        'source_url': metadata.source_url,
                        'serving_config': metadata.serving_config,
                        'metadata': metadata
                    })
        
        # Add API models
        if include_api:
            try:
                from .api_integrations.model_registry import get_api_models_for_pipeline, APIProviderType
                
                api_models = get_api_models_for_pipeline(pipeline_type)
                
                for api_model in api_models:
                    # Apply provider filter if specified
                    if provider_filter and api_model.provider.value not in provider_filter:
                        continue
                    
                    results.append({
                        'model_id': api_model.model_id,
                        'model_name': api_model.model_name,
                        'source_type': 'api',
                        'provider': api_model.provider.value,
                        'pipeline_types': api_model.pipeline_types,
                        'context_length': api_model.context_length,
                        'architecture': 'api',  # API models don't expose architecture
                        'cost_per_1k_input': api_model.cost_per_1k_tokens.get('input', 0) if api_model.cost_per_1k_tokens else 0,
                        'cost_per_1k_output': api_model.cost_per_1k_tokens.get('output', 0) if api_model.cost_per_1k_tokens else 0,
                        'supports_streaming': api_model.supports_streaming,
                        'supports_function_calling': api_model.function_calling,
                        'metadata': api_model
                    })
            except ImportError:
                logger.warning("API model registry not available. Skipping API models.")
        
        # Sort results by source type (self-hosted first) and then by model name
        results.sort(key=lambda x: (0 if x['source_type'] == 'self-hosted' else 1, x['model_name']))
        
        return results
    
    def get_all_pipeline_types(self, include_api: bool = True, include_self_hosted: bool = True) -> Set[str]:
        """
        Get all available pipeline types across self-hosted and API models.
        
        Args:
            include_api: Whether to check API models
            include_self_hosted: Whether to check self-hosted models
            
        Returns:
            Set of pipeline type strings
        """
        pipeline_types = set()
        
        # Get pipeline types from self-hosted models
        if include_self_hosted:
            try:
                from .common.pipeline_types import PipelineTypeMapper
                
                for metadata in self.models.values():
                    # Get pipeline types supported by this architecture
                    types = PipelineTypeMapper.get_pipeline_types_for_architecture(metadata.architecture)
                    pipeline_types.update(types)
            except ImportError:
                logger.warning("Pipeline type mapper not available")
        
        # Get pipeline types from API models
        if include_api:
            try:
                from .api_integrations.model_registry import get_all_pipeline_types as get_api_pipeline_types
                api_types = get_api_pipeline_types()
                pipeline_types.update(api_types)
            except ImportError:
                logger.warning("API model registry not available")
        
        return pipeline_types
    
    def get_model_recommendations(self, pipeline_type: str, 
                                  max_cost_per_1k: Optional[float] = None,
                                  min_context_length: Optional[int] = None,
                                  prefer_self_hosted: bool = False) -> List[Dict[str, Any]]:
        """
        Get recommended models for a pipeline type with filtering and ranking.
        
        Args:
            pipeline_type: HuggingFace pipeline type
            max_cost_per_1k: Maximum cost per 1K tokens (for API models)
            min_context_length: Minimum context length required
            prefer_self_hosted: Rank self-hosted models higher
            
        Returns:
            List of recommended models sorted by relevance
        """
        # Get all models for this pipeline type
        models = self.get_models_by_pipeline_type(pipeline_type)
        
        # Apply filters
        filtered = []
        for model in models:
            # Cost filter (API models only)
            if model['source_type'] == 'api' and max_cost_per_1k is not None:
                avg_cost = (model.get('cost_per_1k_input', 0) + model.get('cost_per_1k_output', 0)) / 2
                if avg_cost > max_cost_per_1k:
                    continue
            
            # Context length filter
            if min_context_length is not None:
                context_len = model.get('context_length')
                if context_len is None or context_len < min_context_length:
                    continue
            
            filtered.append(model)
        
        # Score and sort
        for model in filtered:
            score = 0
            
            # Preference for self-hosted
            if model['source_type'] == 'self-hosted':
                score += 10 if prefer_self_hosted else 0
            
            # Context length bonus
            context_len = model.get('context_length', 0)
            if context_len:
                score += min(context_len / 1000, 50)  # Cap at 50 points
            
            # Cost penalty (lower cost is better)
            if model['source_type'] == 'api':
                avg_cost = (model.get('cost_per_1k_input', 0) + model.get('cost_per_1k_output', 0)) / 2
                if avg_cost > 0:
                    score -= avg_cost * 10  # Penalty proportional to cost
            
            model['recommendation_score'] = score
        
        # Sort by score (descending)
        filtered.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return filtered
    
    # ------------------------------------------------------------------
    # Well-known model seeding
    # ------------------------------------------------------------------

    def seed_well_known_models(self, overwrite: bool = False) -> int:
        """Register a curated set of well-known API-hosted models.

        Seeds the registry with commonly used models from xAI Grok, Meta AI
        (Llama / Spark), OpenAI, Anthropic, and Google.  Each entry includes
        a :class:`ServingConfig` with ``engine="api"`` so that serving-layer
        code can route requests correctly without manual configuration.

        Args:
            overwrite: When *True*, existing entries are replaced with the
                       canonical defaults.  Defaults to *False* (skip if
                       already present).

        Returns:
            Number of models that were newly registered.
        """
        added = 0

        def _reg(meta: ModelMetadata) -> None:
            nonlocal added
            exists = self.get_model(meta.model_id) is not None
            if exists and not overwrite:
                return
            if self.add_model(meta):
                added += 1

        text_in = [IOSpec("prompt", DataType.TEXT)]
        text_out = [IOSpec("response", DataType.TEXT)]
        embed_in = [IOSpec("text", DataType.TEXT)]
        embed_out = [IOSpec("embedding", DataType.EMBEDDINGS)]
        image_in = [IOSpec("image", DataType.IMAGE), IOSpec("prompt", DataType.TEXT)]

        # ----------------------------------------------------------------
        # xAI Grok
        # ----------------------------------------------------------------
        _xai_api_sc = ServingConfig(
            engine="api",
            launch_args={"provider": "xai"},
            default_generation_params={"temperature": 0.7, "max_tokens": 4096},
            routing_weight=1.0,
            hardware_affinity=["api"],
        ).to_dict()

        for mid, arch, ctx, tags, desc in [
            ("xai/grok-3", "GrokForCausalLM", 131072,
             ["xai", "grok", "chat", "reasoning"],
             "xAI Grok 3 — flagship reasoning model"),
            ("xai/grok-3-fast", "GrokForCausalLM", 131072,
             ["xai", "grok", "chat", "fast"],
             "xAI Grok 3 Fast — low-latency variant"),
            ("xai/grok-3-mini", "GrokForCausalLM", 131072,
             ["xai", "grok", "chat", "mini"],
             "xAI Grok 3 Mini — efficient compact model"),
            ("xai/grok-2-1212", "GrokForCausalLM", 131072,
             ["xai", "grok", "chat"],
             "xAI Grok 2 (Dec 2024)"),
        ]:
            sc = ServingConfig(
                engine="api",
                launch_args={"provider": "xai"},
                default_generation_params={"temperature": 0.7, "max_tokens": 4096},
                hardware_affinity=["api"],
            ).to_dict()
            sc["context_window"] = ctx
            _reg(ModelMetadata(
                model_id=mid, model_name=mid.split("/")[-1],
                model_type=ModelType.LANGUAGE_MODEL, architecture=arch,
                inputs=text_in, outputs=text_out,
                supported_backends=["api"],
                hardware_requirements={"api": True},
                tags=tags, description=desc,
                source_url="https://console.x.ai",
                license="Proprietary",
                serving_config=sc,
            ))

        # grok-2-vision (multimodal)
        _reg(ModelMetadata(
            model_id="xai/grok-2-vision-1212",
            model_name="grok-2-vision-1212",
            model_type=ModelType.MULTIMODAL,
            architecture="GrokForVisionCausalLM",
            inputs=image_in, outputs=text_out,
            supported_backends=["api"],
            hardware_requirements={"api": True},
            tags=["xai", "grok", "vision", "multimodal"],
            description="xAI Grok 2 Vision — multimodal image+text model",
            source_url="https://console.x.ai",
            license="Proprietary",
            serving_config=ServingConfig(
                engine="api",
                launch_args={"provider": "xai"},
                default_generation_params={"temperature": 0.7, "max_tokens": 2048},
                hardware_affinity=["api"],
            ).to_dict(),
        ))

        # ----------------------------------------------------------------
        # Meta AI — Llama 3.x / Spark 1.1
        # ----------------------------------------------------------------
        _meta_api_base = {"provider": "meta_ai"}

        for mid, arch, ctx, mtype, ins, outs, tags, desc in [
            ("meta-llama/Llama-3.3-70B-Instruct",
             "LlamaForCausalLM", 128000, ModelType.LANGUAGE_MODEL,
             text_in, text_out,
             ["meta", "llama", "llama-3", "chat", "instruct"],
             "Meta Llama 3.3 70B Instruct"),
            ("meta-llama/Llama-3.1-405B-Instruct",
             "LlamaForCausalLM", 128000, ModelType.LANGUAGE_MODEL,
             text_in, text_out,
             ["meta", "llama", "llama-3", "chat", "instruct", "flagship"],
             "Meta Llama 3.1 405B Instruct — largest open model"),
            ("meta-llama/Llama-3.1-8B-Instruct",
             "LlamaForCausalLM", 128000, ModelType.LANGUAGE_MODEL,
             text_in, text_out,
             ["meta", "llama", "llama-3", "chat", "instruct", "small"],
             "Meta Llama 3.1 8B Instruct — efficient small model"),
            ("meta-llama/Llama-3.2-90B-Vision-Instruct",
             "MllamaForConditionalGeneration", 128000, ModelType.MULTIMODAL,
             image_in, text_out,
             ["meta", "llama", "llama-3", "vision", "multimodal"],
             "Meta Llama 3.2 90B Vision Instruct — multimodal"),
            ("meta-spark/Spark-1.1",
             "SparkForCausalLM", 32768, ModelType.LANGUAGE_MODEL,
             text_in, text_out,
             ["meta", "spark", "meta-spark", "creative"],
             "Meta Spark 1.1 — creative writing and storytelling model"),
        ]:
            sc = ServingConfig(
                engine="api",
                launch_args=dict(_meta_api_base),
                default_generation_params={"temperature": 0.7, "max_tokens": 2048},
                hardware_affinity=["api"],
            ).to_dict()
            sc["context_window"] = ctx
            _reg(ModelMetadata(
                model_id=mid, model_name=mid.split("/")[-1],
                model_type=mtype, architecture=arch,
                inputs=ins, outputs=outs,
                supported_backends=["api"],
                hardware_requirements={"api": True},
                tags=tags, description=desc,
                source_url="https://developer.meta.com/ai",
                license="Meta Llama Community License",
                serving_config=sc,
            ))

        # ----------------------------------------------------------------
        # OpenAI
        # ----------------------------------------------------------------
        for mid, arch, ctx, mtype, ins, outs, tags, desc in [
            ("openai/gpt-4o", "GPT4ForCausalLM", 128000,
             ModelType.MULTIMODAL, image_in, text_out,
             ["openai", "gpt", "vision", "multimodal", "flagship"],
             "OpenAI GPT-4o — multimodal flagship"),
            ("openai/gpt-4o-mini", "GPT4ForCausalLM", 128000,
             ModelType.LANGUAGE_MODEL, text_in, text_out,
             ["openai", "gpt", "fast", "efficient"],
             "OpenAI GPT-4o Mini — fast and affordable"),
            ("openai/o1", "O1ForCausalLM", 200000,
             ModelType.LANGUAGE_MODEL, text_in, text_out,
             ["openai", "reasoning", "o1"],
             "OpenAI o1 — advanced reasoning"),
            ("openai/text-embedding-3-large", "OpenAIEmbeddingModel", 8192,
             ModelType.EMBEDDING_MODEL, embed_in, embed_out,
             ["openai", "embeddings", "3-large"],
             "OpenAI text-embedding-3-large — high-quality embeddings"),
        ]:
            sc = ServingConfig(
                engine="api",
                launch_args={"provider": "openai"},
                default_generation_params={"temperature": 0.7, "max_tokens": 2048},
                hardware_affinity=["api"],
            ).to_dict()
            sc["context_window"] = ctx
            _reg(ModelMetadata(
                model_id=mid, model_name=mid.split("/")[-1],
                model_type=mtype, architecture=arch,
                inputs=ins, outputs=outs,
                supported_backends=["api"],
                hardware_requirements={"api": True},
                tags=tags, description=desc,
                source_url="https://platform.openai.com",
                license="Proprietary",
                serving_config=sc,
            ))

        # ----------------------------------------------------------------
        # Anthropic Claude
        # ----------------------------------------------------------------
        for mid, arch, ctx, tags, desc in [
            ("anthropic/claude-opus-4-5", "ClaudeForCausalLM", 200000,
             ["anthropic", "claude", "flagship", "reasoning"],
             "Anthropic Claude Opus 4.5 — most capable Claude model"),
            ("anthropic/claude-sonnet-4-5", "ClaudeForCausalLM", 200000,
             ["anthropic", "claude", "balanced"],
             "Anthropic Claude Sonnet 4.5 — balanced performance"),
            ("anthropic/claude-haiku-4-5", "ClaudeForCausalLM", 200000,
             ["anthropic", "claude", "fast"],
             "Anthropic Claude Haiku 4.5 — fastest Claude model"),
        ]:
            sc = ServingConfig(
                engine="api",
                launch_args={"provider": "claude"},
                default_generation_params={"temperature": 0.7, "max_tokens": 2048},
                hardware_affinity=["api"],
            ).to_dict()
            sc["context_window"] = ctx
            _reg(ModelMetadata(
                model_id=mid, model_name=mid.split("/")[-1],
                model_type=ModelType.LANGUAGE_MODEL,
                architecture=arch, inputs=text_in, outputs=text_out,
                supported_backends=["api"],
                hardware_requirements={"api": True},
                tags=tags, description=desc,
                source_url="https://console.anthropic.com",
                license="Proprietary",
                serving_config=sc,
            ))

        # ----------------------------------------------------------------
        # Google Gemini
        # ----------------------------------------------------------------
        for mid, arch, ctx, mtype, ins, outs, tags, desc in [
            ("google/gemini-2.5-pro", "GeminiForCausalLM", 1000000,
             ModelType.MULTIMODAL, image_in, text_out,
             ["google", "gemini", "flagship", "vision"],
             "Google Gemini 2.5 Pro — long-context multimodal flagship"),
            ("google/gemini-2.5-flash", "GeminiForCausalLM", 1000000,
             ModelType.MULTIMODAL, image_in, text_out,
             ["google", "gemini", "fast", "vision"],
             "Google Gemini 2.5 Flash — fast multimodal model"),
            ("google/gemini-embedding-exp-03-07", "GeminiEmbeddingModel", 2048,
             ModelType.EMBEDDING_MODEL, embed_in, embed_out,
             ["google", "gemini", "embeddings"],
             "Google Gemini Embedding — high-quality embeddings"),
        ]:
            sc = ServingConfig(
                engine="api",
                launch_args={"provider": "gemini"},
                default_generation_params={"temperature": 0.7, "max_output_tokens": 2048},
                hardware_affinity=["api"],
            ).to_dict()
            sc["context_window"] = ctx
            _reg(ModelMetadata(
                model_id=mid, model_name=mid.split("/")[-1],
                model_type=mtype, architecture=arch,
                inputs=ins, outputs=outs,
                supported_backends=["api"],
                hardware_requirements={"api": True},
                tags=tags, description=desc,
                source_url="https://ai.google.dev",
                license="Proprietary",
                serving_config=sc,
            ))

        logger.info("seed_well_known_models: added %d models", added)
        return added

    def close(self):
        """Close the model manager and save any pending changes."""
        try:
            self._save_data()
            if self.use_database and hasattr(self, 'con'):
                self.con.close()
            logger.info("Model manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing model manager: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions for common operations
def create_model_from_huggingface(model_id: str, hf_config: Dict[str, Any], 
                                architecture: str = None,
                                inference_code_location: str = None,
                                fetch_repo_structure: bool = True,
                                include_ipfs_cids: bool = True,
                                branch: str = "main") -> ModelMetadata:
    """
    Create a ModelMetadata object from HuggingFace configuration with IPFS support.
    
    Args:
        model_id: Model identifier
        hf_config: HuggingFace model configuration
        architecture: Model architecture (if not in config)
        inference_code_location: Path to inference code
        fetch_repo_structure: Whether to fetch repository structure and hashes
        include_ipfs_cids: Whether to generate IPFS CIDs for files
        branch: Git branch to fetch from (default: "main")
        
    Returns:
        ModelMetadata object
    """
    # Extract information from HuggingFace config
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
    architecture = architecture or hf_config.get('architectures', ['unknown'])[0]
    
    # Determine model type from architecture
    arch_lower = architecture.lower()
    if 'vision' in arch_lower or 'vit' in arch_lower or 'clip' in arch_lower:
        model_type = ModelType.VISION_MODEL
    elif 'audio' in arch_lower or 'wav2vec' in arch_lower or 'whisper' in arch_lower:
        model_type = ModelType.AUDIO_MODEL
    elif any(x in arch_lower for x in ['gpt', 'llama', 'bert', 't5', 'roberta']):
        model_type = ModelType.LANGUAGE_MODEL
    else:
        model_type = ModelType.LANGUAGE_MODEL  # Default fallback
    
    # Create basic input/output specs (these would need to be customized)
    inputs = [IOSpec(name="input_ids", data_type=DataType.TOKENS, description="Tokenized input")]
    outputs = [IOSpec(name="logits", data_type=DataType.LOGITS, description="Model output logits")]
    
    # Fetch repository structure if requested
    repository_structure = None
    if fetch_repo_structure:
        logger.info(f"Fetching repository structure for {model_id}")
        repository_structure = fetch_huggingface_repo_structure(model_id, branch, include_ipfs_cids)
        if repository_structure:
            logger.info(f"Successfully fetched repository structure with {repository_structure.get('total_files', 0)} files")
            if repository_structure.get("ipfs_enabled"):
                cid_count = sum(1 for f in repository_structure["files"].values() if "ipfs_cid" in f)
                logger.info(f"Generated IPFS CIDs for {cid_count} files")
        else:
            logger.warning(f"Failed to fetch repository structure for {model_id}")
    
    # Build source URL
    source_url = f"https://huggingface.co/{model_id}"
    
    return ModelMetadata(
        model_id=model_id,
        model_name=model_name,
        model_type=model_type,
        architecture=architecture,
        inputs=inputs,
        outputs=outputs,
        huggingface_config=hf_config,
        inference_code_location=inference_code_location,
        repository_structure=repository_structure,
        source_url=source_url
    )


@lru_cache(maxsize=1)
def get_default_model_manager() -> ModelManager:
    """Get a default model manager instance."""
    return ModelManager()


# Try to import sentence transformers for vector embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    try:
        from .utils.auto_install import ensure_packages
        ensure_packages({"sentence_transformers": "sentence-transformers"})
        from sentence_transformers import SentenceTransformer  # retry
        HAVE_SENTENCE_TRANSFORMERS = True
    except Exception:
        HAVE_SENTENCE_TRANSFORMERS = False
        logger.warning("SentenceTransformers not available. Vector search disabled.")

# Try to import numpy for vector operations
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    try:
        from .utils.auto_install import ensure_packages
        ensure_packages(["numpy"])  # retry
        import numpy as np
        HAVE_NUMPY = True
    except Exception:
        HAVE_NUMPY = False
        logger.warning("NumPy not available. Vector operations disabled.")


@dataclass
class DocumentEntry:
    """Represents a document entry in the vector index."""
    file_path: str
    content: str
    embedding: Optional[List[float]] = None
    title: str = ""
    section: str = ""
    created_at: Optional[datetime] = None


@dataclass
class SearchResult:
    """Represents a search result from the vector index."""
    document: DocumentEntry
    similarity_score: float
    matched_section: str = ""


class VectorDocumentationIndex:
    """
    Vector index for searching through README and documentation files.
    
    This class creates embeddings of all README.md files in the repository
    and provides semantic search capabilities.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", storage_path: Optional[str] = None):
        """
        Initialize the vector documentation index.
        
        Args:
            model_name: Name of the sentence transformer model to use
            storage_path: Path to store the index (JSON file)
        """
        self.model_name = model_name
        self.storage_path = storage_path or "documentation_index.json"
        self.documents: List[DocumentEntry] = []
        self.model = None
        
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded SentenceTransformer model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
        else:
            logger.warning("SentenceTransformers not available. Vector search will not work.")
    
    def index_all_readmes(self, root_path: Optional[str] = None) -> int:
        """
        Index all README.md files in the repository.
        
        Args:
            root_path: Root path to search for README files
            
        Returns:
            Number of documents indexed
        """
        if not self.model or not HAVE_NUMPY:
            logger.error("Cannot index documents: missing dependencies")
            return 0
        
        root_path = root_path or os.getcwd()
        readme_files = []
        
        # Find all README.md files
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.lower() in ['readme.md', 'readme.txt', 'readme.rst']:
                    readme_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(readme_files)} README files")
        
        # Process each README file
        indexed_count = 0
        for file_path in readme_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split content into sections for better granular search
                sections = self._split_content_into_sections(content)
                
                for section_title, section_content in sections:
                    if len(section_content.strip()) < 50:  # Skip very short sections
                        continue
                    
                    # Create embedding
                    embedding = self.model.encode(section_content).tolist()
                    
                    # Create document entry
                    doc_entry = DocumentEntry(
                        file_path=os.path.relpath(file_path, root_path),
                        content=section_content,
                        embedding=embedding,
                        title=os.path.basename(file_path),
                        section=section_title,
                        created_at=datetime.now()
                    )
                    
                    self.documents.append(doc_entry)
                    indexed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")
        
        logger.info(f"Successfully indexed {indexed_count} document sections")
        
        # Save index to disk
        self.save_index()
        
        return indexed_count
    
    def _split_content_into_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split markdown content into logical sections."""
        sections = []
        lines = content.split('\n')
        current_section = ""
        current_title = "Introduction"
        
        for line in lines:
            if line.startswith('#'):
                # Save previous section
                if current_section.strip():
                    sections.append((current_title, current_section.strip()))
                
                # Start new section
                current_title = line.strip('#').strip()
                current_section = ""
            else:
                current_section += line + '\n'
        
        # Add final section
        if current_section.strip():
            sections.append((current_title, current_section.strip()))
        
        return sections
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[SearchResult]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity score threshold
            
        Returns:
            List of search results sorted by similarity
        """
        if not self.model or not HAVE_NUMPY:
            logger.error("Cannot search: missing dependencies")
            return []
        
        if not self.documents:
            logger.warning("No documents indexed. Run index_all_readmes() first.")
            return []
        
        # Create query embedding
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        results = []
        for doc in self.documents:
            if doc.embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
            )
            
            if similarity >= min_similarity:
                results.append(SearchResult(
                    document=doc,
                    similarity_score=float(similarity),
                    matched_section=doc.section
                ))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def save_index(self):
        """Save the index to disk."""
        try:
            # Convert to JSON-serializable format
            data = {
                'model_name': self.model_name,
                'documents': []
            }
            
            for doc in self.documents:
                doc_data = asdict(doc)
                if doc_data['created_at']:
                    doc_data['created_at'] = doc_data['created_at'].isoformat()
                data['documents'].append(doc_data)
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved index with {len(self.documents)} documents to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self) -> bool:
        """Load the index from disk."""
        try:
            if not os.path.exists(self.storage_path):
                logger.info("No existing index found")
                return False
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.model_name = data.get('model_name', self.model_name)
            self.documents = []
            
            for doc_data in data.get('documents', []):
                if doc_data.get('created_at'):
                    doc_data['created_at'] = datetime.fromisoformat(doc_data['created_at'])
                
                self.documents.append(DocumentEntry(**doc_data))
            
            logger.info(f"Loaded index with {len(self.documents)} documents from {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False


@dataclass
class BanditArm:
    """Represents an arm in the multi-armed bandit (a model option)."""
    model_id: str
    total_reward: float = 0.0
    num_trials: int = 0
    alpha: float = 1.0  # Beta distribution parameter
    beta: float = 1.0   # Beta distribution parameter
    
    @property
    def average_reward(self) -> float:
        """Calculate average reward for this arm."""
        return self.total_reward / max(self.num_trials, 1)
    
    @property
    def confidence_bound(self) -> float:
        """Calculate upper confidence bound."""
        if self.num_trials == 0:
            return float('inf')
        
        import math
        confidence = math.sqrt(2 * math.log(self.num_trials + 1) / self.num_trials)
        return self.average_reward + confidence


@dataclass
@dataclass
class RecommendationContext:
    """Context information for model recommendations."""
    task_type: Optional[str] = None
    hardware: Optional[str] = None
    input_type: Optional[DataType] = None
    output_type: Optional[DataType] = None
    performance_requirements: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    
    def to_key(self) -> str:
        """Convert context to a string key for grouping."""
        return f"{self.task_type}_{self.hardware}_{self.input_type}_{self.output_type}"


@dataclass
class ModelRecommendation:
    """Represents a model recommendation with confidence."""
    model_id: str
    confidence_score: float
    predicted_performance: Optional[Dict[str, float]] = None
    reasoning: Optional[str] = None


class BanditModelRecommender:
    """
    Multi-armed bandit algorithm for model recommendation.
    
    This class uses bandit algorithms to learn which models work best
    for different contexts based on user feedback.
    """
    
    def __init__(self, 
                 algorithm: str = "thompson_sampling",
                 model_manager: Optional[ModelManager] = None,
                 storage_path: Optional[str] = None):
        """
        Initialize the bandit model recommender.
        
        Args:
            algorithm: Bandit algorithm to use ('ucb', 'thompson_sampling', 'epsilon_greedy')
            model_manager: Model manager instance
            storage_path: Path to store bandit data
        """
        self.algorithm = algorithm
        self.model_manager = model_manager or ModelManager()
        self.storage_path = storage_path or "bandit_data.json"
        
        # Context-specific bandit arms
        self.bandit_arms: Dict[str, Dict[str, BanditArm]] = {}
        self.global_trial_count = 0
        
        # Algorithm parameters
        self.epsilon = 0.1  # For epsilon-greedy
        
        # Load existing data
        self.load_bandit_data()
    
    def recommend_model(self, context: RecommendationContext) -> Optional[ModelRecommendation]:
        """
        Recommend a model based on the given context using bandit algorithm.
        
        Args:
            context: Context for the recommendation
            
        Returns:
            Model recommendation or None if no suitable models found
        """
        # Get compatible models from model manager
        compatible_models = self._get_compatible_models(context)
        
        if not compatible_models:
            logger.warning("No compatible models found for context")
            return None
        
        context_key = context.to_key()
        
        # Initialize arms for this context if needed
        if context_key not in self.bandit_arms:
            self.bandit_arms[context_key] = {}
        
        for model_id in compatible_models:
            if model_id not in self.bandit_arms[context_key]:
                self.bandit_arms[context_key][model_id] = BanditArm(model_id=model_id)
        
        # Select model using bandit algorithm
        selected_model_id = self._select_arm(context_key)
        
        if not selected_model_id:
            return None
        
        # Get confidence score
        arm = self.bandit_arms[context_key][selected_model_id]
        confidence_score = min(arm.average_reward * 1.2, 1.0)  # Cap at 1.0
        
        return ModelRecommendation(
            model_id=selected_model_id,
            confidence_score=confidence_score,
            reasoning=f"Selected using {self.algorithm} algorithm with {arm.num_trials} trials"
        )
    
    def provide_feedback(self, 
                        model_id: str, 
                        feedback_score: float, 
                        context: RecommendationContext):
        """
        Provide feedback on a model recommendation.
        
        Args:
            model_id: ID of the recommended model
            feedback_score: Feedback score (0.0 to 1.0, where 1.0 is best)
            context: Context in which the model was used
        """
        context_key = context.to_key()
        
        if context_key not in self.bandit_arms:
            self.bandit_arms[context_key] = {}
        
        if model_id not in self.bandit_arms[context_key]:
            self.bandit_arms[context_key][model_id] = BanditArm(model_id=model_id)
        
        arm = self.bandit_arms[context_key][model_id]
        
        # Update arm statistics
        arm.total_reward += feedback_score
        arm.num_trials += 1
        
        # Update beta distribution parameters for Thompson sampling
        if feedback_score > 0.5:
            arm.alpha += 1
        else:
            arm.beta += 1
        
        self.global_trial_count += 1
        
        # Save updated data
        self.save_bandit_data()
        
        logger.info(f"Updated feedback for {model_id}: avg_reward={arm.average_reward:.3f}, trials={arm.num_trials}")
    
    def _get_compatible_models(self, context: RecommendationContext) -> List[str]:
        """Get list of models compatible with the given context."""
        try:
            # Start with all models
            all_models = self.model_manager.list_models()
            compatible = []
            
            for model in all_models:
                # More flexible compatibility check
                is_compatible = True
                
                # Check input/output compatibility if specified and models have detailed specs
                if context.input_type and hasattr(context.input_type, 'value'):
                    input_type_str = context.input_type.value if hasattr(context.input_type, 'value') else str(context.input_type)
                    if model.inputs and input_type_str not in ['tokens', 'text']:
                        if not any(hasattr(inp.data_type, 'value') and inp.data_type.value == input_type_str for inp in model.inputs):
                            # If no exact match but it's a language model and we want tokens/text, allow it
                            if model.model_type == ModelType.LANGUAGE_MODEL and input_type_str in ['tokens', 'text']:
                                pass  # Allow language models for text/token inputs
                            else:
                                is_compatible = False
                
                # For basic compatibility, just check task type compatibility
                if context.task_type:
                    task_lower = context.task_type.lower()
                    if task_lower == 'generation':
                        # GPT-style models good for generation
                        if 'gpt' in model.model_id.lower() or 'llama' in model.model_id.lower():
                            is_compatible = True
                        elif model.model_type == ModelType.LANGUAGE_MODEL:
                            is_compatible = True
                    elif task_lower == 'classification':
                        # BERT-style models good for classification  
                        if 'bert' in model.model_id.lower() or 'roberta' in model.model_id.lower():
                            is_compatible = True
                        elif model.model_type == ModelType.LANGUAGE_MODEL:
                            is_compatible = True
                    elif task_lower == 'embedding':
                        # Any language model can provide embeddings
                        if model.model_type == ModelType.LANGUAGE_MODEL:
                            is_compatible = True
                
                # If no specific constraints, include all models
                if not context.task_type and not context.input_type and not context.output_type:
                    is_compatible = True
                
                if is_compatible:
                    compatible.append(model.model_id)
            
            # If no models found with strict matching, return all models as fallback
            if not compatible:
                logger.info("No strictly compatible models found, returning all available models")
                compatible = [model.model_id for model in all_models]
            
            return compatible
            
        except Exception as e:
            logger.error(f"Error getting compatible models: {e}")
            # Return all models as fallback
            try:
                all_models = self.model_manager.list_models()
                return [model.model_id for model in all_models]
            except:
                return []
    
    def _select_arm(self, context_key: str) -> Optional[str]:
        """Select an arm using the configured bandit algorithm."""
        arms = self.bandit_arms.get(context_key, {})
        
        if not arms:
            return None
        
        if self.algorithm == "ucb":
            return self._select_ucb(arms)
        elif self.algorithm == "thompson_sampling":
            return self._select_thompson_sampling(arms)
        elif self.algorithm == "epsilon_greedy":
            return self._select_epsilon_greedy(arms)
        else:
            logger.error(f"Unknown bandit algorithm: {self.algorithm}")
            return None
    
    def _select_ucb(self, arms: Dict[str, BanditArm]) -> str:
        """Select arm using Upper Confidence Bound algorithm."""
        best_arm = None
        best_ucb = -float('inf')
        
        for model_id, arm in arms.items():
            ucb = arm.confidence_bound
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = model_id
        
        return best_arm
    
    def _select_thompson_sampling(self, arms: Dict[str, BanditArm]) -> str:
        """Select arm using Thompson Sampling algorithm."""
        if not HAVE_NUMPY:
            # Fallback to UCB if numpy not available
            return self._select_ucb(arms)
        
        best_arm = None
        best_sample = -1
        
        for model_id, arm in arms.items():
            # Sample from beta distribution
            sample = np.random.beta(arm.alpha, arm.beta)
            
            if sample > best_sample:
                best_sample = sample
                best_arm = model_id
        
        return best_arm
    
    def _select_epsilon_greedy(self, arms: Dict[str, BanditArm]) -> str:
        """Select arm using Epsilon-Greedy algorithm."""
        import random
        
        # Explore with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(list(arms.keys()))
        
        # Exploit: select arm with highest average reward
        best_arm = None
        best_reward = -1
        
        for model_id, arm in arms.items():
            if arm.average_reward > best_reward:
                best_reward = arm.average_reward
                best_arm = model_id
        
        return best_arm
    
    def save_bandit_data(self):
        """Save bandit data to disk."""
        try:
            data = {
                'algorithm': self.algorithm,
                'global_trial_count': self.global_trial_count,
                'epsilon': self.epsilon,
                'bandit_arms': {}
            }
            
            for context_key, arms in self.bandit_arms.items():
                data['bandit_arms'][context_key] = {}
                for model_id, arm in arms.items():
                    data['bandit_arms'][context_key][model_id] = asdict(arm)
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved bandit data to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save bandit data: {e}")
    
    def load_bandit_data(self) -> bool:
        """Load bandit data from disk."""
        try:
            if not os.path.exists(self.storage_path):
                logger.info("No existing bandit data found")
                return False
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.algorithm = data.get('algorithm', self.algorithm)
            self.global_trial_count = data.get('global_trial_count', 0)
            self.epsilon = data.get('epsilon', self.epsilon)
            
            self.bandit_arms = {}
            for context_key, arms_data in data.get('bandit_arms', {}).items():
                self.bandit_arms[context_key] = {}
                for model_id, arm_data in arms_data.items():
                    self.bandit_arms[context_key][model_id] = BanditArm(**arm_data)
            
            logger.info(f"Loaded bandit data from {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load bandit data: {e}")
            return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get a performance report of the bandit algorithm."""
        report = {
            'algorithm': self.algorithm,
            'total_trials': self.global_trial_count,
            'contexts': {}
        }
        
        for context_key, arms in self.bandit_arms.items():
            context_report = {
                'total_arms': len(arms),
                'total_trials': sum(arm.num_trials for arm in arms.values()),
                'best_model': None,
                'best_average_reward': -1,
                'arms': {}
            }
            
            for model_id, arm in arms.items():
                context_report['arms'][model_id] = {
                    'average_reward': arm.average_reward,
                    'num_trials': arm.num_trials,
                    'confidence_bound': arm.confidence_bound
                }
                
                if arm.average_reward > context_report['best_average_reward']:
                    context_report['best_average_reward'] = arm.average_reward
                    context_report['best_model'] = model_id
            
            report['contexts'][context_key] = context_report
        
        return report
