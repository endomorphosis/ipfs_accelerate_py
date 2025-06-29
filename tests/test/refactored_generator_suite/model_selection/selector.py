#!/usr/bin/env python3
"""
Model Selector

This module provides the model selector for finding the best model based on criteria.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Union, Tuple

from .registry import ModelRegistry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Selector for finding the best model based on criteria.
    
    This class provides methods to filter and select models based on various criteria
    such as architecture, task, hardware profile, and more.
    """
    
    def __init__(self, registry=None):
        """
        Initialize the model selector.
        
        Args:
            registry: ModelRegistry instance (will create one if not provided)
        """
        self.registry = registry or ModelRegistry()
    
    def select_model(self, 
                    architecture: Optional[str] = None,
                    model_type: Optional[str] = None,
                    task: Optional[str] = None,
                    hardware_profile: Optional[str] = None,
                    max_size_mb: Optional[int] = None,
                    framework: Optional[str] = None,
                    default_only: bool = False) -> Optional[str]:
        """
        Select the best model based on criteria.
        
        Args:
            architecture: Model architecture (e.g., "encoder-only")
            model_type: Model type (e.g., "bert")
            task: Task to perform (e.g., "fill-mask")
            hardware_profile: Hardware profile for the model
            max_size_mb: Maximum model size in MB
            framework: Framework to use (e.g., "pytorch")
            default_only: Whether to only consider default models
            
        Returns:
            Model ID or None if no matching model is found
        """
        # Start with all models
        if model_type:
            # If model_type is provided, start with models of that type
            models = self.registry.get_models_by_type(model_type)
        elif architecture:
            # If architecture is provided, start with models of that architecture
            models = self.registry.get_models_by_architecture(architecture)
        elif task:
            # If task is provided, start with models that support that task
            models = self.registry.get_models_by_task(task)
        else:
            # Otherwise, start with all models
            models = [self.registry.get_model(model_id) for model_id in self.registry.get_model_ids()]
        
        # Filter models based on criteria
        filtered_models = self._filter_models(
            models,
            architecture=architecture,
            model_type=model_type,
            task=task,
            hardware_profile=hardware_profile,
            max_size_mb=max_size_mb,
            framework=framework,
            default_only=default_only
        )
        
        if not filtered_models:
            return None
        
        # Sort models (prioritize default, then by size)
        sorted_models = sorted(
            filtered_models,
            key=lambda m: (
                0 if m.get("default", False) else 1,
                m.get("size_mb", 1000000)
            )
        )
        
        # Return the best model ID
        return sorted_models[0]["id"] if "id" in sorted_models[0] else list(self.registry.models.keys())[0]
    
    def _filter_models(self, 
                      models: List[Dict[str, Any]],
                      architecture: Optional[str] = None,
                      model_type: Optional[str] = None,
                      task: Optional[str] = None,
                      hardware_profile: Optional[str] = None,
                      max_size_mb: Optional[int] = None,
                      framework: Optional[str] = None,
                      default_only: bool = False) -> List[Dict[str, Any]]:
        """
        Filter models based on criteria.
        
        Args:
            models: List of model metadata dictionaries
            architecture: Model architecture (e.g., "encoder-only")
            model_type: Model type (e.g., "bert")
            task: Task to perform (e.g., "fill-mask")
            hardware_profile: Hardware profile for the model
            max_size_mb: Maximum model size in MB
            framework: Framework to use (e.g., "pytorch")
            default_only: Whether to only consider default models
            
        Returns:
            Filtered list of model metadata dictionaries
        """
        filtered = []
        
        for model in models:
            # Skip models that don't match the criteria
            if architecture and model.get("architecture") != architecture:
                continue
                
            if model_type and model.get("type") != model_type:
                continue
                
            if task:
                tasks = model.get("tasks", [])
                if isinstance(tasks, str):
                    tasks = [tasks]
                if task not in tasks:
                    continue
            
            if max_size_mb and model.get("size_mb", 0) > max_size_mb:
                continue
                
            if framework:
                frameworks = model.get("frameworks", [])
                if isinstance(frameworks, str):
                    frameworks = [frameworks]
                if framework not in frameworks:
                    continue
            
            if default_only and not model.get("default", False):
                continue
                
            # Add the model to the filtered list
            filtered.append(model)
        
        return filtered
    
    def get_hardware_profile(self, 
                           has_cuda: bool = False,
                           has_rocm: bool = False,
                           has_mps: bool = False,
                           memory_gb: Optional[float] = None,
                           cuda_cores: Optional[int] = None) -> str:
        """
        Get the hardware profile based on available hardware.
        
        Args:
            has_cuda: Whether CUDA is available
            has_rocm: Whether ROCm is available
            has_mps: Whether MPS (Apple Silicon) is available
            memory_gb: Available GPU memory in GB
            cuda_cores: Number of CUDA cores
            
        Returns:
            Hardware profile name
        """
        if has_cuda or has_rocm:
            if memory_gb:
                if memory_gb >= 32:
                    return "gpu-large"
                elif memory_gb >= 16:
                    return "gpu-medium"
                elif memory_gb >= 8:
                    return "gpu-small"
                else:
                    return "gpu-nano"
            else:
                return "gpu-medium"
        elif has_mps:
            return "mps"
        else:
            return "cpu"