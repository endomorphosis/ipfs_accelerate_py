#!/usr/bin/env python3
"""
Model Registry

This module provides the model registry for storing and accessing model metadata.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Registry for model metadata.
    
    This class provides a central registry for model metadata, allowing models to be looked up
    by ID, type, architecture, or other attributes.
    """
    
    def __init__(self, config=None):
        """
        Initialize the model registry.
        
        Args:
            config: Configuration object or dict
        """
        self.config = config or {}
        self.models = {}
        self.architectures = {}
        self.types = {}
        self.tasks = {}
        
        # Load default models
        self._load_defaults()
        
        # Load models from config
        self._load_from_config()
    
    def register_model(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """
        Register a model with its metadata.
        
        Args:
            model_id: The model ID
            metadata: Model metadata
        """
        # Add model to registry
        self.models[model_id] = metadata
        
        # Update indices
        self._update_indices(model_id, metadata)
        
        logger.debug(f"Registered model: {model_id}")
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata by ID.
        
        Args:
            model_id: The model ID
            
        Returns:
            Model metadata or None if not found
        """
        return self.models.get(model_id)
    
    def get_models_by_architecture(self, architecture: str) -> List[Dict[str, Any]]:
        """
        Get all models for a specific architecture.
        
        Args:
            architecture: The architecture name
            
        Returns:
            List of model metadata dictionaries
        """
        model_ids = self.architectures.get(architecture, [])
        return [self.models[model_id] for model_id in model_ids if model_id in self.models]
    
    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get all models for a specific type.
        
        Args:
            model_type: The model type
            
        Returns:
            List of model metadata dictionaries
        """
        model_ids = self.types.get(model_type, [])
        return [self.models[model_id] for model_id in model_ids if model_id in self.models]
    
    def get_models_by_task(self, task: str) -> List[Dict[str, Any]]:
        """
        Get all models that support a specific task.
        
        Args:
            task: The task name
            
        Returns:
            List of model metadata dictionaries
        """
        model_ids = self.tasks.get(task, [])
        return [self.models[model_id] for model_id in model_ids if model_id in self.models]
    
    def get_architectures(self) -> List[str]:
        """
        Get all registered architectures.
        
        Returns:
            List of architecture names
        """
        return list(self.architectures.keys())
    
    def get_types(self) -> List[str]:
        """
        Get all registered model types.
        
        Returns:
            List of model types
        """
        return list(self.types.keys())
    
    def get_tasks(self) -> List[str]:
        """
        Get all registered tasks.
        
        Returns:
            List of task names
        """
        return list(self.tasks.keys())
    
    def get_model_ids(self) -> List[str]:
        """
        Get all registered model IDs.
        
        Returns:
            List of model IDs
        """
        return list(self.models.keys())
    
    def _update_indices(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update indices for a model.
        
        Args:
            model_id: The model ID
            metadata: Model metadata
        """
        # Update architecture index
        if "architecture" in metadata:
            architecture = metadata["architecture"]
            if architecture not in self.architectures:
                self.architectures[architecture] = []
            if model_id not in self.architectures[architecture]:
                self.architectures[architecture].append(model_id)
                
        # Update type index
        if "type" in metadata:
            model_type = metadata["type"]
            if model_type not in self.types:
                self.types[model_type] = []
            if model_id not in self.types[model_type]:
                self.types[model_type].append(model_id)
                
        # Update task index
        for task_field in ["tasks", "supported_tasks", "recommended_tasks"]:
            if task_field in metadata and isinstance(metadata[task_field], (list, tuple)):
                for task in metadata[task_field]:
                    if task not in self.tasks:
                        self.tasks[task] = []
                    if model_id not in self.tasks[task]:
                        self.tasks[task].append(model_id)
    
    def _load_defaults(self) -> None:
        """Load default models."""
        # Default models for different architectures
        default_models = {
            "bert-base-uncased": {
                "architecture": "encoder-only",
                "type": "bert",
                "tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"],
                "parameters": "110M",
                "size_mb": 440,
                "frameworks": ["pytorch", "tensorflow", "onnx"],
                "description": "BERT base model (uncased)",
                "default": True
            },
            "bert-large-uncased": {
                "architecture": "encoder-only",
                "type": "bert",
                "tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"],
                "parameters": "340M",
                "size_mb": 1360,
                "frameworks": ["pytorch", "tensorflow", "onnx"],
                "description": "BERT large model (uncased)"
            },
            "roberta-base": {
                "architecture": "encoder-only",
                "type": "roberta",
                "tasks": ["fill-mask", "text-classification", "token-classification", "question-answering"],
                "parameters": "125M",
                "size_mb": 500,
                "frameworks": ["pytorch", "tensorflow", "onnx"],
                "description": "RoBERTa base model",
                "default": True
            },
            "gpt2": {
                "architecture": "decoder-only",
                "type": "gpt2",
                "tasks": ["text-generation", "text-classification"],
                "parameters": "124M",
                "size_mb": 496,
                "frameworks": ["pytorch", "tensorflow", "onnx"],
                "description": "GPT-2 small model",
                "default": True
            },
            "gpt2-medium": {
                "architecture": "decoder-only",
                "type": "gpt2",
                "tasks": ["text-generation", "text-classification"],
                "parameters": "355M",
                "size_mb": 1420,
                "frameworks": ["pytorch", "tensorflow", "onnx"],
                "description": "GPT-2 medium model"
            },
            "EleutherAI/gpt-j-6b": {
                "architecture": "decoder-only",
                "type": "gpt-j",
                "tasks": ["text-generation", "text-classification"],
                "parameters": "6B",
                "size_mb": 24000,
                "frameworks": ["pytorch", "onnx"],
                "description": "GPT-J 6B model",
                "default": True
            },
            "t5-small": {
                "architecture": "encoder-decoder",
                "type": "t5",
                "tasks": ["text2text-generation", "translation", "summarization"],
                "parameters": "60M",
                "size_mb": 240,
                "frameworks": ["pytorch", "tensorflow", "onnx"],
                "description": "T5 small model"
            },
            "t5-base": {
                "architecture": "encoder-decoder",
                "type": "t5",
                "tasks": ["text2text-generation", "translation", "summarization"],
                "parameters": "220M",
                "size_mb": 880,
                "frameworks": ["pytorch", "tensorflow", "onnx"],
                "description": "T5 base model",
                "default": True
            },
            "google/vit-base-patch16-224": {
                "architecture": "vision",
                "type": "vit",
                "tasks": ["image-classification", "image-to-text"],
                "parameters": "86M",
                "size_mb": 344,
                "frameworks": ["pytorch", "tensorflow", "onnx"],
                "description": "Vision Transformer base model",
                "default": True
            },
            "openai/clip-vit-base-patch32": {
                "architecture": "vision-text",
                "type": "clip",
                "tasks": ["zero-shot-image-classification", "image-to-text"],
                "parameters": "150M",
                "size_mb": 600,
                "frameworks": ["pytorch", "onnx"],
                "description": "CLIP Vision-Text model",
                "default": True
            },
            "openai/whisper-base": {
                "architecture": "speech",
                "type": "whisper",
                "tasks": ["automatic-speech-recognition", "audio-to-text"],
                "parameters": "74M",
                "size_mb": 296,
                "frameworks": ["pytorch", "onnx"],
                "description": "Whisper base model",
                "default": True
            }
        }
        
        # Register default models
        for model_id, metadata in default_models.items():
            self.register_model(model_id, metadata)
            
        logger.info(f"Loaded {len(default_models)} default models")
    
    def _load_from_config(self) -> None:
        """Load models from configuration."""
        if not self.config:
            return
            
        # Load from model_registry key
        if "model_registry" in self.config and isinstance(self.config["model_registry"], dict):
            for model_id, metadata in self.config["model_registry"].items():
                self.register_model(model_id, metadata)
                
            logger.info(f"Loaded {len(self.config['model_registry'])} models from config")
            
        # Load from registry_files key
        if "registry_files" in self.config and isinstance(self.config["registry_files"], list):
            for file_path in self.config["registry_files"]:
                self._load_from_file(file_path)
                
    def _load_from_file(self, file_path: str) -> bool:
        """
        Load models from a file.
        
        Args:
            file_path: Path to the model registry file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.warning(f"Model registry file not found: {file_path}")
                return False
                
            with open(path, "r") as f:
                # Determine format based on extension
                if path.suffix == ".json":
                    data = json.load(f)
                elif path.suffix in [".yaml", ".yml"]:
                    try:
                        import yaml
                        data = yaml.safe_load(f)
                    except ImportError:
                        logger.error(f"Cannot load YAML file {path} (PyYAML not installed)")
                        return False
                else:
                    logger.warning(f"Unsupported file format: {path}")
                    return False
                    
            # Register models
            if isinstance(data, dict):
                for model_id, metadata in data.items():
                    self.register_model(model_id, metadata)
                    
                logger.info(f"Loaded {len(data)} models from {file_path}")
                return True
            else:
                logger.warning(f"Invalid model registry format in {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models from {file_path}: {e}")
            return False
    
    def save(self, file_path: str) -> bool:
        """
        Save the model registry to a file.
        
        Args:
            file_path: Path to save the model registry
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(file_path)
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            with open(path, "w") as f:
                # Determine format based on extension
                if path.suffix == ".json":
                    json.dump(self.models, f, indent=2)
                elif path.suffix in [".yaml", ".yml"]:
                    try:
                        import yaml
                        yaml.dump(self.models, f, default_flow_style=False)
                    except ImportError:
                        logger.error(f"Cannot save YAML file {path} (PyYAML not installed)")
                        return False
                else:
                    logger.warning(f"Unsupported file format: {path}")
                    return False
                    
            logger.info(f"Saved model registry to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model registry to {file_path}: {e}")
            return False