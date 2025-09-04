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
                                 'hardware_requirements', 'performance_metrics', 'tags']:
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
                                 'hardware_requirements', 'performance_metrics', 'tags']:
                    if data[json_field] is not None:
                        data[json_field] = json.dumps(data[json_field])
                
                # Convert enum to string
                data['model_type'] = data['model_type'].value
                
                # Update timestamp
                data['updated_at'] = datetime.now()
                
                # Insert or update
                self.con.execute("""
                    INSERT OR REPLACE INTO model_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            "models_with_inference_code": 0
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
        
        return stats
    
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
                                inference_code_location: str = None) -> ModelMetadata:
    """
    Create a ModelMetadata object from HuggingFace configuration.
    
    Args:
        model_id: Model identifier
        hf_config: HuggingFace model configuration
        architecture: Model architecture (if not in config)
        inference_code_location: Path to inference code
        
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
    
    return ModelMetadata(
        model_id=model_id,
        model_name=model_name,
        model_type=model_type,
        architecture=architecture,
        inputs=inputs,
        outputs=outputs,
        huggingface_config=hf_config,
        inference_code_location=inference_code_location
    )


def get_default_model_manager() -> ModelManager:
    """Get a default model manager instance."""
    return ModelManager()