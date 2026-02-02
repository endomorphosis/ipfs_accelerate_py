"""
Model loader with skill integration using anyio.
"""

import anyio
import inspect
import logging
import importlib
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from .types import LoadedModel, ModelStatus
from .cache import ModelCache

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and caches HuggingFace models using discovered skills."""
    
    def __init__(self, registry, hardware_selector, config):
        """
        Initialize model loader.
        
        Args:
            registry: SkillRegistry instance
            hardware_selector: HardwareSelector instance
            config: ServerConfig instance
        """
        self.registry = registry
        self.hardware_selector = hardware_selector
        self.config = config
        self.cache = ModelCache(
            max_size=config.max_loaded_models,
            max_memory_mb=16384  # 16GB default
        )
        self._loading = {}  # Track in-progress loads
        self._lock = anyio.Lock()
    
    async def load_model(self, model_id: str) -> LoadedModel:
        """
        Load a model, using cache if available.
        
        Args:
            model_id: Model identifier
            
        Returns:
            LoadedModel instance
            
        Raises:
            ValueError: If model not found in registry
            RuntimeError: If model loading fails
        """
        # Check cache first
        cached = await self.cache.get(model_id)
        if cached:
            logger.info(f"Using cached model: {model_id}")
            return cached
        
        # Check if already loading
        async with self._lock:
            if model_id in self._loading:
                # Wait for ongoing load
                logger.debug(f"Waiting for ongoing load: {model_id}")
                return await self._loading[model_id]
            
            # Start loading - create a future-like object for anyio
            # Since anyio doesn't have create_task, we'll use a different approach
            # We'll just call the implementation directly
            pass  # Removed task creation, will call directly
        
        try:
            model = await self._load_model_impl(model_id)
            await self.cache.put(model_id, model)
            return model
        finally:
            async with self._lock:
                if model_id in self._loading:
                    del self._loading[model_id]
    
    async def _load_model_impl(self, model_id: str) -> LoadedModel:
        """Internal implementation of model loading."""
        logger.info(f"Loading model: {model_id}")
        
        # Get skill from registry
        skill_info = self.registry.get_skill_for_model(model_id)
        if not skill_info:
            raise ValueError(f"Model not found in registry: {model_id}")
        
        # Select hardware
        hardware, reason = self.hardware_selector.select_hardware({
            "supported_hardware": skill_info.supported_hardware
        })
        logger.info(f"Selected hardware '{hardware}' for {model_id}: {reason}")
        
        # Load skill module and instantiate
        try:
            # Import skill module
            module = importlib.import_module(skill_info.module_path)
            
            # Get skill class (assume it's named after the file)
            skill_class_name = Path(skill_info.file_path).stem
            if not hasattr(module, skill_class_name):
                raise RuntimeError(f"Skill class '{skill_class_name}' not found in module")
            
            skill_class = getattr(module, skill_class_name)
            
            # Instantiate skill
            skill_instance = skill_class()
            
            # Initialize with selected hardware
            init_method = f"init_{hardware}"
            if hasattr(skill_instance, init_method):
                init_fn = getattr(skill_instance, init_method)
                # Call init method (may be sync or async)
                if inspect.iscoroutinefunction(init_fn):
                    await init_fn()
                else:
                    init_fn()
            else:
                logger.warning(f"No {init_method} method found for {model_id}")
            
            # Create LoadedModel
            loaded_model = LoadedModel(
                model_id=model_id,
                skill_instance=skill_instance,
                hardware=hardware,
                status=ModelStatus.LOADED,
                metadata={
                    "skill_name": skill_info.skill_name,
                    "architecture": skill_info.architecture,
                    "task_type": skill_info.task_type,
                }
            )
            
            # Cache the model
            await self.cache.put(model_id, loaded_model)
            
            logger.info(f"Successfully loaded model: {model_id}")
            return loaded_model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
            # Create failed model entry
            failed_model = LoadedModel(
                model_id=model_id,
                skill_instance=None,
                hardware=hardware,
                status=ModelStatus.FAILED,
                error=str(e)
            )
            raise RuntimeError(f"Failed to load model {model_id}: {e}")
    
    async def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from cache.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model was unloaded, False if not found
        """
        model = await self.cache.remove(model_id)
        if model:
            logger.info(f"Unloaded model: {model_id}")
            return True
        return False
    
    async def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models."""
        models = {}
        for model_id in self.cache._cache:
            model = await self.cache.get(model_id)
            if model:
                models[model_id] = model.to_dict()
        return models
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
