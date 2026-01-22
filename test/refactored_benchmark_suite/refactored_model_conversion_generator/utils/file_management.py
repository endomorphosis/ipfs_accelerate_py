"""
File management utilities for model conversion.
"""

import os
import glob
import shutil
import logging
import tempfile
import hashlib
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ModelFileManager:
    """
    Utilities for managing model files.
    
    This class provides methods for finding, copying, and organizing model files.
    """
    
    @staticmethod
    def find_models(directory: str, formats: Optional[List[str]] = None, recursive: bool = True) -> Dict[str, List[str]]:
        """
        Find model files in directory.
        
        Args:
            directory: Directory to search in
            formats: List of model formats to look for (e.g., ['onnx', 'pytorch', 'tensorflow'])
            recursive: Whether to search recursively
            
        Returns:
            Dictionary mapping format to list of found model files
        """
        # Define extensions for each format
        format_extensions = {
            'onnx': ['.onnx'],
            'pytorch': ['.pt', '.pth', '.bin'],
            'tensorflow': ['.pb', '.savedmodel', '.h5', '.keras'],
            'tflite': ['.tflite'],
            'openvino': ['.xml'],
            'webnn': ['.webnn', '.webnn.js'],
            'webgpu': ['.webgpu', '.webgpu.js']
        }
        
        # Use all formats if none specified
        formats = formats or list(format_extensions.keys())
        
        # Find all models
        found_models = {}
        
        for fmt in formats:
            extensions = format_extensions.get(fmt, [f'.{fmt}'])
            found_models[fmt] = []
            
            for ext in extensions:
                if recursive:
                    pattern = os.path.join(directory, '**', f'*{ext}')
                    matches = glob.glob(pattern, recursive=True)
                else:
                    pattern = os.path.join(directory, f'*{ext}')
                    matches = glob.glob(pattern)
                    
                found_models[fmt].extend(matches)
                
        return found_models
        
    @staticmethod
    def organize_models(source_dir: str, target_dir: str, 
                      structure: str = 'format/model', 
                      copy: bool = True) -> Dict[str, str]:
        """
        Organize model files into a structured directory.
        
        Args:
            source_dir: Directory containing model files
            target_dir: Directory to organize models in
            structure: Directory structure to use ('format/model' or 'model/format')
            copy: Whether to copy files (True) or move them (False)
            
        Returns:
            Dictionary mapping source path to target path
        """
        # Find all models
        all_models = ModelFileManager.find_models(source_dir)
        
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Track mappings
        path_mapping = {}
        
        # Organize models
        for fmt, model_paths in all_models.items():
            for model_path in model_paths:
                # Get model name
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                
                # Create target path based on structure
                if structure == 'format/model':
                    target_path = os.path.join(target_dir, fmt, model_name, os.path.basename(model_path))
                else:  # model/format
                    target_path = os.path.join(target_dir, model_name, fmt, os.path.basename(model_path))
                    
                # Create target directory
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Copy or move file
                if copy:
                    shutil.copy2(model_path, target_path)
                else:
                    shutil.move(model_path, target_path)
                    
                path_mapping[model_path] = target_path
                
        return path_mapping
        
    @staticmethod
    def get_cache_path(model_path: str, source_format: str, target_format: str, 
                     cache_dir: Optional[str] = None) -> str:
        """
        Get cache path for converted model.
        
        Args:
            model_path: Path to source model
            source_format: Source model format
            target_format: Target model format
            cache_dir: Directory to store cached models
            
        Returns:
            Path to cached model
        """
        # Get model name
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Compute hash of model file
        with open(model_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            
        # Create cache directory if needed
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), 'model_converter_cache')
            
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache path
        cache_path = os.path.join(cache_dir, f"{model_name}_{source_format}_to_{target_format}_{file_hash[:8]}")
        
        # Add extension
        if target_format == 'onnx':
            cache_path += '.onnx'
        elif target_format == 'pytorch':
            cache_path += '.pt'
        elif target_format == 'openvino':
            cache_path += '.xml'
        elif target_format == 'webnn':
            cache_path += '.js'
        elif target_format == 'webgpu':
            cache_path += '.js'
        else:
            cache_path += f'.{target_format}'
            
        return cache_path
        
    @staticmethod
    def check_cached_model(model_path: str, source_format: str, target_format: str, 
                         cache_dir: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if converted model is already cached.
        
        Args:
            model_path: Path to source model
            source_format: Source model format
            target_format: Target model format
            cache_dir: Directory to store cached models
            
        Returns:
            Tuple of (cached, cache_path)
        """
        cache_path = ModelFileManager.get_cache_path(model_path, source_format, target_format, cache_dir)
        
        if os.path.exists(cache_path):
            # Verify cache is valid
            try:
                # Check if metadata exists
                metadata_path = cache_path + '.json'
                if os.path.exists(metadata_path):
                    # Check if source model has been modified since cache was created
                    source_modified = os.path.getmtime(model_path)
                    cache_modified = os.path.getmtime(cache_path)
                    
                    if source_modified > cache_modified:
                        # Source model is newer, cache is invalid
                        return False, cache_path
                        
                return True, cache_path
            except Exception as e:
                logger.warning(f"Error checking cached model: {e}")
                return False, cache_path
        else:
            return False, cache_path