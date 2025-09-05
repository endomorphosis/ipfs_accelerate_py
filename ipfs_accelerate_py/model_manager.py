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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Try to import IPFS multiformats for content addressing
try:
    from .ipfs_multiformats import ipfs_multiformats_py
    HAVE_IPFS_MULTIFORMATS = True
except ImportError:
    try:
        from ipfs_multiformats import ipfs_multiformats_py  
    except ImportError:
        HAVE_IPFS_MULTIFORMATS = False

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ipfs_accelerate_model_manager")

# Try to import DuckDB for database storage
try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False
    logger.warning("DuckDB not available. Using JSON storage backend.")

# Try to import requests for HuggingFace API access
try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
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
    repository_structure: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
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


class ModelManager:
    """
    Comprehensive model manager for storing and retrieving model metadata.
    
    This class provides functionality to:
    - Store and retrieve model metadata
    - Map input/output types for different models
    - Manage HuggingFace configuration data
    - Track inference code locations
    - Query models by various criteria
    """
    
    def __init__(self, storage_path: str = None, use_database: bool = None):
        """
        Initialize the model manager.
        
        Args:
            storage_path: Path for storage backend (JSON file or DuckDB database)
            use_database: Whether to use DuckDB backend. If None, auto-detect.
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
        
        # Initialize storage backend
        if self.use_database:
            self._init_database()
        else:
            self._init_json_storage()
        
        # Load existing data
        self._load_data()
        
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
                    repository_structure JSON,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
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
                                 'hardware_requirements', 'performance_metrics', 'tags', 'repository_structure']:
                    if row_dict[json_field]:
                        row_dict[json_field] = json.loads(row_dict[json_field])
                
                # Convert to IOSpec objects
                if row_dict['inputs']:
                    row_dict['inputs'] = [IOSpec(**spec) for spec in row_dict['inputs']]
                if row_dict['outputs']:
                    row_dict['outputs'] = [IOSpec(**spec) for spec in row_dict['outputs']]
                
                # Convert enum fields
                row_dict['model_type'] = ModelType(row_dict['model_type'])
                
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
            for model_id, metadata in self.models.items():
                # Convert to dictionary and handle special fields
                data = asdict(metadata)
                
                # Convert IOSpec objects to dictionaries
                data['inputs'] = json.dumps([asdict(spec) for spec in data['inputs']])
                data['outputs'] = json.dumps([asdict(spec) for spec in data['outputs']])
                
                # Convert other complex fields to JSON
                for json_field in ['huggingface_config', 'supported_backends', 
                                 'hardware_requirements', 'performance_metrics', 'tags', 'repository_structure']:
                    if data[json_field] is not None:
                        data[json_field] = json.dumps(data[json_field])
                
                # Convert enum to string
                data['model_type'] = data['model_type'].value
                
                # Update timestamp
                data['updated_at'] = datetime.now()
                
                # Insert or update
                self.con.execute("""
                    INSERT OR REPLACE INTO model_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(data.values()))
            
            logger.info(f"Saved {len(self.models)} models to database")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def _save_to_json(self):
        """Save model metadata to JSON file."""
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
                
                # Convert enum to string
                model_dict['model_type'] = model_dict['model_type'].value
                
                data[model_id] = model_dict
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.json_path) or '.', exist_ok=True)
            
            # Write to file
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
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
            self.models[metadata.model_id] = metadata
            self._save_data()
            logger.info(f"Added/updated model: {metadata.model_id}")
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
        return self.models.get(model_id)
    
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
                
                logger.info(f"Removed model: {model_id}")
                return True
            else:
                logger.warning(f"Model not found: {model_id}")
                return False
        except Exception as e:
            logger.error(f"Error removing model {model_id}: {e}")
            return False
    
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
            "total_tracked_files": 0
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
        
        return stats
    
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


def get_default_model_manager() -> ModelManager:
    """Get a default model manager instance."""
    return ModelManager()


# Try to import sentence transformers for vector embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    logger.warning("SentenceTransformers not available. Vector search disabled.")

# Try to import numpy for vector operations
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
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