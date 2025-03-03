import os
import threading
import logging
from datetime import datetime

class ResourcePool:
    """
    Centralized resource management to avoid duplicate loading of models and resources.
    
    This class provides efficient resource sharing across test execution and implementation
    validation, avoiding duplicate model loading and optimizing memory usage.
    
    Attributes:
        resources (dict): Dictionary of shared resources
        models (dict): Dictionary of loaded models
        tokenizers (dict): Dictionary of loaded tokenizers
        _lock (threading.RLock): Lock for thread safety
        _stats (dict): Usage statistics
    """
    
    def __init__(self):
        self.resources = {}
        self.models = {}
        self.tokenizers = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0, 
            "misses": 0, 
            "memory_usage": 0,
            "creation_timestamps": {},
            "last_accessed": {}
        }
        
        # Setup logging
        self.logger = logging.getLogger("ResourcePool")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("ResourcePool initialized")
    
    def get_resource(self, resource_type, resource_id=None, constructor=None):
        """
        Get or create a resource from the pool
        
        Args:
            resource_type (str): The type of resource (e.g., 'torch', 'transformers')
            resource_id (str, optional): Optional identifier for the resource
            constructor (callable, optional): Function to create the resource if not present
            
        Returns:
            The requested resource, or None if it couldn't be created
        """
        with self._lock:
            key = f"{resource_type}:{resource_id}" if resource_id else resource_type
            
            # Check if resource exists
            if key in self.resources:
                # Resource hit - reusing existing
                self._stats["hits"] += 1
                self._stats["last_accessed"][key] = datetime.now().isoformat()
                self.logger.debug(f"Resource hit: {key}")
                return self.resources[key]
            
            # Resource miss - need to create it
            if constructor:
                self._stats["misses"] += 1
                try:
                    self.logger.info(f"Creating resource: {key}")
                    self.resources[key] = constructor()
                    self._stats["creation_timestamps"][key] = datetime.now().isoformat()
                    self._stats["last_accessed"][key] = datetime.now().isoformat()
                    
                    # Optionally track memory usage if it's a PyTorch model
                    if hasattr(self.resources[key], "get_memory_footprint"):
                        memory_usage = self.resources[key].get_memory_footprint()
                        self._stats["memory_usage"] += memory_usage
                        self.logger.info(f"Resource {key} uses {memory_usage} bytes")
                    
                    return self.resources[key]
                except Exception as e:
                    self.logger.error(f"Error creating resource {key}: {str(e)}")
                    return None
            else:
                self.logger.warning(f"Resource not found and no constructor provided: {key}")
                return None
    
    def get_model(self, model_type, model_name, constructor=None):
        """
        Get or create a model from the pool
        
        Args:
            model_type (str): The type of model (e.g., 'bert', 't5')
            model_name (str): The specific model name (e.g., 'bert-base-uncased')
            constructor (callable, optional): Function to create the model if not present
            
        Returns:
            The requested model, or None if it couldn't be created
        """
        with self._lock:
            key = f"{model_type}:{model_name}"
            
            # Check if model exists
            if key in self.models:
                # Model hit - reusing existing
                self._stats["hits"] += 1
                self._stats["last_accessed"][key] = datetime.now().isoformat()
                self.logger.debug(f"Model hit: {key}")
                return self.models[key]
            
            # Model miss - need to create it
            if constructor:
                self._stats["misses"] += 1
                try:
                    self.logger.info(f"Loading model: {key}")
                    start_time = datetime.now()
                    self.models[key] = constructor()
                    load_time = (datetime.now() - start_time).total_seconds()
                    
                    self._stats["creation_timestamps"][key] = datetime.now().isoformat()
                    self._stats["last_accessed"][key] = datetime.now().isoformat()
                    self.logger.info(f"Model {key} loaded in {load_time:.2f} seconds")
                    
                    # Track memory usage if possible
                    try:
                        import torch
                        if hasattr(self.models[key], "get_memory_footprint"):
                            memory_usage = self.models[key].get_memory_footprint()
                        elif torch.is_tensor(self.models[key]) or hasattr(self.models[key], "parameters"):
                            # For PyTorch models
                            memory_usage = sum(p.nelement() * p.element_size() for p in self.models[key].parameters())
                        else:
                            memory_usage = 0
                            
                        self._stats["memory_usage"] += memory_usage
                        self.logger.info(f"Model {key} uses approximately {memory_usage/1024/1024:.2f} MB")
                    except (ImportError, AttributeError, Exception) as e:
                        self.logger.debug(f"Could not calculate memory usage for {key}: {str(e)}")
                    
                    return self.models[key]
                except Exception as e:
                    self.logger.error(f"Error loading model {key}: {str(e)}")
                    return None
            else:
                self.logger.warning(f"Model not found and no constructor provided: {key}")
                return None
    
    def get_tokenizer(self, model_type, model_name, constructor=None):
        """
        Get or create a tokenizer from the pool
        
        Args:
            model_type (str): The type of model (e.g., 'bert', 't5')
            model_name (str): The specific model name (e.g., 'bert-base-uncased')
            constructor (callable, optional): Function to create the tokenizer if not present
            
        Returns:
            The requested tokenizer, or None if it couldn't be created
        """
        with self._lock:
            key = f"tokenizer:{model_type}:{model_name}"
            
            # Check if tokenizer exists
            if key in self.tokenizers:
                # Tokenizer hit - reusing existing
                self._stats["hits"] += 1
                self._stats["last_accessed"][key] = datetime.now().isoformat()
                self.logger.debug(f"Tokenizer hit: {key}")
                return self.tokenizers[key]
            
            # Tokenizer miss - need to create it
            if constructor:
                self._stats["misses"] += 1
                try:
                    self.logger.info(f"Loading tokenizer: {key}")
                    self.tokenizers[key] = constructor()
                    self._stats["creation_timestamps"][key] = datetime.now().isoformat()
                    self._stats["last_accessed"][key] = datetime.now().isoformat()
                    
                    return self.tokenizers[key]
                except Exception as e:
                    self.logger.error(f"Error loading tokenizer {key}: {str(e)}")
                    return None
            else:
                self.logger.warning(f"Tokenizer not found and no constructor provided: {key}")
                return None
    
    def cleanup_unused_resources(self, max_age_minutes=30):
        """
        Clean up resources that haven't been used in a while
        
        Args:
            max_age_minutes (int): Maximum time in minutes since last access before cleaning up
        """
        with self._lock:
            current_time = datetime.now()
            resources_to_remove = []
            models_to_remove = []
            tokenizers_to_remove = []
            
            # Check resources
            for key, resource in self.resources.items():
                if key in self._stats["last_accessed"]:
                    last_accessed = datetime.fromisoformat(self._stats["last_accessed"][key])
                    age_minutes = (current_time - last_accessed).total_seconds() / 60
                    
                    if age_minutes > max_age_minutes:
                        resources_to_remove.append(key)
            
            # Check models
            for key, model in self.models.items():
                if key in self._stats["last_accessed"]:
                    last_accessed = datetime.fromisoformat(self._stats["last_accessed"][key])
                    age_minutes = (current_time - last_accessed).total_seconds() / 60
                    
                    if age_minutes > max_age_minutes:
                        models_to_remove.append(key)
            
            # Check tokenizers
            for key, tokenizer in self.tokenizers.items():
                if key in self._stats["last_accessed"]:
                    last_accessed = datetime.fromisoformat(self._stats["last_accessed"][key])
                    age_minutes = (current_time - last_accessed).total_seconds() / 60
                    
                    if age_minutes > max_age_minutes:
                        tokenizers_to_remove.append(key)
            
            # Remove resources
            for key in resources_to_remove:
                self.logger.info(f"Cleaning up unused resource: {key}")
                del self.resources[key]
            
            # Remove models - with special handling for CUDA models
            for key in models_to_remove:
                self.logger.info(f"Cleaning up unused model: {key}")
                try:
                    # Try to move model to CPU before deletion if it's a PyTorch model
                    if hasattr(self.models[key], "to") and hasattr(self.models[key], "cpu"):
                        self.models[key].to("cpu")
                except Exception:
                    pass
                
                del self.models[key]
            
            # Remove tokenizers
            for key in tokenizers_to_remove:
                self.logger.info(f"Cleaning up unused tokenizer: {key}")
                del self.tokenizers[key]
            
            # Force garbage collection
            try:
                import gc
                gc.collect()
                
                # Try to clear CUDA cache if available
                try:
                    import torch
                    if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            except Exception as e:
                self.logger.debug(f"Error during garbage collection: {str(e)}")
            
            removed_count = len(resources_to_remove) + len(models_to_remove) + len(tokenizers_to_remove)
            self.logger.info(f"Cleaned up {removed_count} unused resources")
            return removed_count
    
    def get_stats(self):
        """
        Get resource pool usage statistics
        
        Returns:
            dict: Statistics about resource usage
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_ratio = self._stats["hits"] / max(1, total_requests)
            
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "total_requests": total_requests,
                "hit_ratio": hit_ratio,
                "memory_usage": self._stats["memory_usage"],
                "memory_usage_mb": self._stats["memory_usage"] / (1024 * 1024),
                "cached_resources": len(self.resources),
                "cached_models": len(self.models),
                "cached_tokenizers": len(self.tokenizers),
                "timestamp": datetime.now().isoformat()
            }
    
    def clear(self):
        """Clear all cached resources"""
        with self._lock:
            # First try to clean up PyTorch resources properly
            try:
                # Move models to CPU before deletion if possible
                for key, model in self.models.items():
                    if hasattr(model, "to") and hasattr(model, "cpu"):
                        try:
                            model.to("cpu")
                        except Exception as e:
                            self.logger.debug(f"Error moving model {key} to CPU: {str(e)}")
                
                # Try to clear CUDA cache if available
                try:
                    import torch
                    if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            except Exception as e:
                self.logger.debug(f"Error during torch cleanup: {str(e)}")
            
            # Clear all dictionaries
            count = len(self.resources) + len(self.models) + len(self.tokenizers)
            self.resources.clear()
            self.models.clear()
            self.tokenizers.clear()
            
            # Reset stats but keep structure
            self._stats = {
                "hits": 0, 
                "misses": 0, 
                "memory_usage": 0,
                "creation_timestamps": {},
                "last_accessed": {}
            }
            
            # Force garbage collection
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            
            self.logger.info(f"ResourcePool cleared - removed {count} cached objects")

# Create a global instance for shared use
global_resource_pool = ResourcePool()

def get_global_resource_pool():
    """Get the global resource pool instance"""
    return global_resource_pool