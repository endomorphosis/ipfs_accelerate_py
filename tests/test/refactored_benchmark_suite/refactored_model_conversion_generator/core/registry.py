"""
Model Converter Registry Module

This module provides a registry for model converter implementations.
"""

import logging
from typing import Dict, Any, Type, List, Optional, Callable

from .converter import ModelConverter

logger = logging.getLogger(__name__)

class ModelConverterRegistry:
    """
    Registry for model converter implementations.
    
    This class maintains a registry of model converter implementations,
    organized by source format, target format, and model type.
    """
    
    _registry: Dict[str, Dict[str, Dict[str, Type[ModelConverter]]]] = {}
    
    @classmethod
    def register(cls, source_format: str, target_format: str, model_type: Optional[str] = None):
        """
        Register a model converter implementation.
        
        Args:
            source_format: Source model format (e.g., 'pytorch', 'onnx')
            target_format: Target model format (e.g., 'onnx', 'webnn')
            model_type: Optional model type (e.g., 'bert', 'vit')
            
        Returns:
            Decorator function for registering the converter class
        """
        def decorator(converter_class: Type[ModelConverter]):
            # Initialize registry structure if needed
            if source_format not in cls._registry:
                cls._registry[source_format] = {}
                
            if target_format not in cls._registry[source_format]:
                cls._registry[source_format][target_format] = {}
                
            # Register converter class
            key = model_type or '*'
            cls._registry[source_format][target_format][key] = converter_class
            
            logger.debug(f"Registered converter: {source_format} -> {target_format} for model type {key}")
            
            return converter_class
            
        return decorator
        
    @classmethod
    def get_converter(cls, source_format: str, target_format: str, model_type: Optional[str] = None) -> Optional[Type[ModelConverter]]:
        """
        Get a converter class for the specified formats and model type.
        
        Args:
            source_format: Source model format (e.g., 'pytorch', 'onnx')
            target_format: Target model format (e.g., 'onnx', 'webnn')
            model_type: Optional model type (e.g., 'bert', 'vit')
            
        Returns:
            Converter class or None if no matching converter is found
        """
        # Check if we have converters for source format
        if source_format not in cls._registry:
            logger.warning(f"No converters registered for source format: {source_format}")
            return None
            
        # Check if we have converters for target format
        if target_format not in cls._registry[source_format]:
            logger.warning(f"No converters registered for {source_format} -> {target_format}")
            return None
            
        # Try to find exact match for model type
        if model_type and model_type in cls._registry[source_format][target_format]:
            return cls._registry[source_format][target_format][model_type]
            
        # Fall back to generic converter if available
        if '*' in cls._registry[source_format][target_format]:
            return cls._registry[source_format][target_format]['*']
            
        logger.warning(f"No converters found for {source_format} -> {target_format} for model type {model_type}")
        return None
        
    @classmethod
    def list_converters(cls) -> List[Dict[str, str]]:
        """
        List all registered converters.
        
        Returns:
            List of converter information dictionaries
        """
        converters = []
        
        for source_format, targets in cls._registry.items():
            for target_format, model_types in targets.items():
                for model_type in model_types:
                    converter_class = cls._registry[source_format][target_format][model_type]
                    converters.append({
                        'source_format': source_format,
                        'target_format': target_format,
                        'model_type': model_type,
                        'converter_class': converter_class.__name__
                    })
                    
        return converters
        
    @classmethod
    def clear(cls):
        """Clear the registry."""
        cls._registry = {}


# Convenience function for registration
def register_converter(source_format: str, target_format: str, model_type: Optional[str] = None):
    """
    Decorator for registering a model converter implementation.
    
    Args:
        source_format: Source model format (e.g., 'pytorch', 'onnx')
        target_format: Target model format (e.g., 'onnx', 'webnn')
        model_type: Optional model type (e.g., 'bert', 'vit')
        
    Returns:
        Decorator function for registering the converter class
    """
    return ModelConverterRegistry.register(source_format, target_format, model_type)