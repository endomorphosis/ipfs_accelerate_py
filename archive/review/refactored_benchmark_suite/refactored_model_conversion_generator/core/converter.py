"""
Model Converter Base Module

This module provides the base class for all model converter implementations.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ConversionResult:
    """
    Result of a model conversion operation.
    
    Attributes:
        success: Whether the conversion was successful
        output_path: Path to the converted model
        format: Format of the converted model
        metadata: Additional metadata about the conversion
        error: Error message if conversion failed
        conversion_time: Time taken for conversion in seconds
    """
    success: bool
    output_path: Optional[str] = None
    format: Optional[str] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    conversion_time: Optional[float] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


class ModelConverter(ABC):
    """
    Base class for all model converter implementations.
    
    This class defines the common interface and behavior for all model converter
    implementations. Subclasses should implement the convert() method.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model converter.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @property
    def source_format(self) -> str:
        """
        Get the source format for this converter.
        
        Returns:
            Source model format
        """
        return self._get_source_format()
        
    @property
    def target_format(self) -> str:
        """
        Get the target format for this converter.
        
        Returns:
            Target model format
        """
        return self._get_target_format()
        
    @property
    def supported_model_types(self) -> List[str]:
        """
        Get the model types supported by this converter.
        
        Returns:
            List of supported model types
        """
        return self._get_supported_model_types()
        
    @abstractmethod
    def _get_source_format(self) -> str:
        """
        Get the source format for this converter.
        
        Returns:
            Source model format
        """
        raise NotImplementedError("Subclasses must implement _get_source_format()")
        
    @abstractmethod
    def _get_target_format(self) -> str:
        """
        Get the target format for this converter.
        
        Returns:
            Target model format
        """
        raise NotImplementedError("Subclasses must implement _get_target_format()")
        
    @abstractmethod
    def _get_supported_model_types(self) -> List[str]:
        """
        Get the model types supported by this converter.
        
        Returns:
            List of supported model types
        """
        raise NotImplementedError("Subclasses must implement _get_supported_model_types()")
        
    def convert(self, model_path: str, output_dir: Optional[str] = None, 
                model_type: Optional[str] = None, **kwargs) -> ConversionResult:
        """
        Convert a model from source format to target format.
        
        Args:
            model_path: Path to the source model
            output_dir: Directory to save the converted model
            model_type: Type of model (e.g., 'bert', 'vit')
            **kwargs: Additional conversion parameters
            
        Returns:
            ConversionResult with conversion details
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not os.path.exists(model_path):
                return ConversionResult(
                    success=False,
                    error=f"Source model not found: {model_path}",
                    conversion_time=time.time() - start_time
                )
                
            # Create output directory if needed
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Determine output path
            if output_dir:
                output_path = self._get_output_path(model_path, output_dir, model_type)
            else:
                output_path = self._get_default_output_path(model_path, model_type)
                
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
            # Check if converted model already exists
            if os.path.exists(output_path) and not kwargs.get('force', False):
                self.logger.info(f"Converted model already exists at {output_path}")
                return ConversionResult(
                    success=True,
                    output_path=output_path,
                    format=self.target_format,
                    metadata={'cached': True},
                    conversion_time=0.0
                )
                
            # Execute the conversion
            self.logger.info(f"Converting {self.source_format} model to {self.target_format}")
            result = self._execute_conversion(model_path, output_path, model_type, **kwargs)
            
            # Add conversion time to result
            result.conversion_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error converting model: {e}", exc_info=True)
            return ConversionResult(
                success=False,
                error=str(e),
                conversion_time=time.time() - start_time
            )
    
    @abstractmethod        
    def _execute_conversion(self, model_path: str, output_path: str, 
                          model_type: Optional[str] = None, **kwargs) -> ConversionResult:
        """
        Execute the model conversion.
        
        Args:
            model_path: Path to the source model
            output_path: Path to save the converted model
            model_type: Type of model (e.g., 'bert', 'vit')
            **kwargs: Additional conversion parameters
            
        Returns:
            ConversionResult with conversion details
        """
        raise NotImplementedError("Subclasses must implement _execute_conversion()")
        
    def _get_output_path(self, model_path: str, output_dir: str, 
                        model_type: Optional[str] = None) -> str:
        """
        Get the output path for the converted model.
        
        Args:
            model_path: Path to the source model
            output_dir: Directory to save the converted model
            model_type: Type of model (e.g., 'bert', 'vit')
            
        Returns:
            Output path for the converted model
        """
        # Extract model name from path
        model_name = os.path.basename(model_path)
        model_name = os.path.splitext(model_name)[0]  # Remove extension
        
        # Add target format extension
        extension = self._get_target_extension()
        
        # Create a unique name based on source, target, and model type
        if model_type:
            output_filename = f"{model_name}_{model_type}_{self.target_format}{extension}"
        else:
            output_filename = f"{model_name}_{self.target_format}{extension}"
            
        return os.path.join(output_dir, output_filename)
        
    def _get_default_output_path(self, model_path: str, model_type: Optional[str] = None) -> str:
        """
        Get the default output path for the converted model.
        
        Args:
            model_path: Path to the source model
            model_type: Type of model (e.g., 'bert', 'vit')
            
        Returns:
            Default output path for the converted model
        """
        # Create output path in same directory as source
        output_dir = os.path.dirname(model_path)
        return self._get_output_path(model_path, output_dir, model_type)
        
    def _get_target_extension(self) -> str:
        """
        Get the file extension for the target format.
        
        Returns:
            File extension for the target format
        """
        # Default extensions for common formats
        extensions = {
            'onnx': '.onnx',
            'pytorch': '.pt',
            'tensorflow': '.pb',
            'tflite': '.tflite',
            'webnn': '.webnn',
            'webgpu': '.webgpu',
            'openvino': '.xml',
        }
        
        return extensions.get(self.target_format, f".{self.target_format}")
        
    def validate_model(self, model_path: str, model_type: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate a model before conversion.
        
        Args:
            model_path: Path to the model
            model_type: Type of model (e.g., 'bert', 'vit')
            
        Returns:
            Tuple of (valid, error_message)
        """
        # Check if model file exists
        if not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}"
            
        try:
            # Default implementation just checks file existence
            # Subclasses should override this method to perform format-specific validation
            return True, None
        except Exception as e:
            return False, str(e)
            
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Dictionary containing model information
        """
        # Basic model information
        model_info = {
            'path': model_path,
            'format': self.source_format,
            'size': os.path.getsize(model_path) if os.path.exists(model_path) else 0,
        }
        
        # Compute hash for model file
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_info['md5'] = hashlib.md5(f.read()).hexdigest()
            except Exception as e:
                self.logger.warning(f"Error computing model hash: {e}")
                
        return model_info