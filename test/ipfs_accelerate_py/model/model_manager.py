"""
Model manager for IPFS Accelerate SDK.

This module provides a high-level model management interface
that simplifies working with models across different hardware backends.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from ipfs_accelerate_py.worker.worker import Worker
from ipfs_accelerate_py.hardware.hardware_profile import HardwareProfile
from ipfs_accelerate_py.hardware.hardware_detector import HardwareDetector

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ipfs_accelerate.model")

class ModelWrapper:
    """
    Wrapper for model that provides a simplified interface.
    
    This class wraps a model endpoint and provides a simplified
    interface for running inference and accessing model metadata.
    """
    
    def __init__(self, 
                model_name: str,
                worker: Worker,
                hardware_profile: Optional[HardwareProfile] = None):
        """
        Initialize model wrapper.
        
        Args:
            model_name: Name of the model.
            worker: Worker instance.
            hardware_profile: Hardware profile to use (optional).
        """
        self.model_name = model_name
        self.worker = worker
        self.current_hardware = None
        
        # Initialize with specified hardware or find optimal
        if hardware_profile:
            self.hardware_profile = hardware_profile
            self.current_hardware = hardware_profile.backend
        else:
            # Get optimal hardware
            self.current_hardware = worker.get_optimal_hardware(model_name)
            self.hardware_profile = HardwareProfile(backend=self.current_hardware)
        
        # Ensure worker is initialized
        if not worker.worker_status:
            worker.init_hardware()
            
        # Ensure model is loaded
        if model_name not in worker.endpoint_handler:
            worker.init_worker([model_name])
            
        # Cache endpoint handler for quick access
        self._endpoint_handler = worker.endpoint_handler.get(model_name, {}).get(self.current_hardware)
    
    def __call__(self, content: Any, **kwargs) -> Any:
        """
        Run inference on the model.
        
        Args:
            content: Input content for inference.
            **kwargs: Additional arguments for inference.
            
        Returns:
            Inference results.
        """
        return self.run_inference(content, **kwargs)
    
    def run_inference(self, content: Any, **kwargs) -> Any:
        """
        Run inference on the model.
        
        Args:
            content: Input content for inference.
            **kwargs: Additional arguments for inference.
            
        Returns:
            Inference results.
        """
        # Check if endpoint handler is available
        if not self._endpoint_handler:
            logger.warning(f"Endpoint handler for {self.model_name} on {self.current_hardware} not available")
            # Try to initialize
            self.worker.init_worker([self.model_name])
            self._endpoint_handler = self.worker.endpoint_handler.get(self.model_name, {}).get(self.current_hardware)
            
            if not self._endpoint_handler:
                raise ValueError(f"Endpoint handler for {self.model_name} on {self.current_hardware} not available")
        
        # Run inference using endpoint handler
        return self._endpoint_handler(content, **kwargs)
    
    def get_embeddings(self, content: Any, **kwargs) -> Any:
        """
        Get embeddings from the model.
        
        Args:
            content: Input content for embedding.
            **kwargs: Additional arguments for embedding.
            
        Returns:
            Embedding results.
        """
        return self.run_inference(content, embedding=True, **kwargs)
    
    def switch_hardware(self, hardware_backend: str) -> None:
        """
        Switch to a different hardware backend.
        
        Args:
            hardware_backend: Hardware backend to switch to.
        """
        # Check if hardware is available
        if hardware_backend not in self.worker.worker_status.get("hwtest", {}):
            raise ValueError(f"Hardware backend {hardware_backend} not available")
        
        # Update hardware profile and current hardware
        self.hardware_profile = HardwareProfile(backend=hardware_backend)
        self.current_hardware = hardware_backend
        
        # Update endpoint handler
        self._endpoint_handler = self.worker.endpoint_handler.get(self.model_name, {}).get(self.current_hardware)
    
    def get_current_hardware(self) -> str:
        """
        Get current hardware backend.
        
        Returns:
            Current hardware backend name.
        """
        return self.current_hardware
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information.
        """
        # This would typically query model registry or database
        # Simplified for brevity
        return {
            "name": self.model_name,
            "hardware": self.current_hardware,
            "hardware_profile": self.hardware_profile.to_dict()
        }

class ModelManager:
    """
    Model manager for IPFS Accelerate SDK.
    
    This class provides a high-level interface for managing models
    across different hardware backends.
    """
    
    def __init__(self, worker: Optional[Worker] = None, config=None):
        """
        Initialize model manager.
        
        Args:
            worker: Worker instance (optional, will create if not provided).
            config: Configuration instance (optional).
        """
        self.config = config
        self.worker = worker or Worker(config)
        self.loaded_models = {}
        
        # Initialize hardware
        if not self.worker.worker_status:
            self.worker.init_hardware()
    
    def load_model(self, 
                  model_name: str, 
                  hardware_profile: Optional[HardwareProfile] = None) -> ModelWrapper:
        """
        Load a model.
        
        Args:
            model_name: Name of the model.
            hardware_profile: Hardware profile to use (optional).
            
        Returns:
            ModelWrapper instance.
        """
        # Create model wrapper
        model = ModelWrapper(model_name, self.worker, hardware_profile)
        
        # Store in loaded models
        self.loaded_models[model_name] = model
        
        return model
    
    def get_model(self, model_name: str) -> Optional[ModelWrapper]:
        """
        Get a loaded model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            ModelWrapper instance or None if not loaded.
        """
        return self.loaded_models.get(model_name)
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            True if unloaded, False if not loaded.
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            return True
        return False
    
    def get_optimal_hardware(self, model_name: str, task_type: str = None, batch_size: int = 1) -> str:
        """
        Get optimal hardware for a model.
        
        Args:
            model_name: Name of the model.
            task_type: Type of task or model.
            batch_size: Batch size to use.
            
        Returns:
            Name of optimal hardware backend.
        """
        return self.worker.get_optimal_hardware(model_name, task_type, batch_size)
    
    def get_available_hardware(self) -> List[str]:
        """
        Get available hardware backends.
        
        Returns:
            List of available hardware backend names.
        """
        return list(self.worker.worker_status.get("hwtest", {}).keys())