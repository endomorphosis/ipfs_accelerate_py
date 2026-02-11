"""
HuggingFace Hub API Cache

Cache adapter for HuggingFace Hub API responses.
Caches model metadata, file listings, dataset info, and other Hub queries.
"""

import logging
import threading
from typing import Any, Dict, Optional

from .base_cache import BaseAPICache
from .provider_secrets import get_provider_cache_secret

logger = logging.getLogger(__name__)


class HuggingFaceHubCache(BaseAPICache):
    """
    Cache for HuggingFace Hub API responses.
    
    Caches:
    - Model info (metadata, configuration, README)
    - Model file listings
    - Dataset info
    - Space info
    - Search results
    """
    
    # Default TTLs for different operations (in seconds)
    DEFAULT_TTLS = {
        "model_info": 3600,  # 1 hour (model metadata changes infrequently)
        "model_files": 1800,  # 30 minutes (files can be added/updated)
        "dataset_info": 3600,  # 1 hour
        "space_info": 1800,  # 30 minutes
        "search_models": 600,  # 10 minutes (search results can change)
        "search_datasets": 600,  # 10 minutes
        "repo_commits": 300,  # 5 minutes (commits change frequently)
        "download_url": 86400,  # 24 hours (download URLs are stable)
    }
    
    def get_cache_namespace(self) -> str:
        """Get cache namespace."""
        return "huggingface_hub"
    
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        """
        Extract validation fields from HuggingFace Hub API response.
        
        Args:
            operation: Operation name
            data: API response data
            
        Returns:
            Validation fields dictionary or None
        """
        if not data or not isinstance(data, dict):
            return None
        
        validation = {}
        
        # Model info - use sha (commit hash) and lastModified
        if operation == "model_info":
            validation["sha"] = data.get("sha")
            validation["lastModified"] = data.get("lastModified")
            validation["downloads"] = data.get("downloads")
            validation["likes"] = data.get("likes")
        
        # Dataset info - similar to model info
        elif operation == "dataset_info":
            validation["sha"] = data.get("sha")
            validation["lastModified"] = data.get("lastModified")
            validation["downloads"] = data.get("downloads")
        
        # File listings - use last commit sha
        elif operation == "model_files":
            if isinstance(data, list):
                # Hash file names and sizes
                validation["files"] = {
                    f.get("path"): f.get("size") 
                    for f in data 
                    if isinstance(f, dict)
                }
            else:
                validation["sha"] = data.get("sha")
        
        # Repo commits - use latest commit
        elif operation == "repo_commits":
            if isinstance(data, list) and data:
                latest = data[0]
                if isinstance(latest, dict):
                    validation["latest_commit"] = latest.get("oid")
                    validation["commit_date"] = latest.get("date")
        
        return validation if validation else None
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        """Get operation-specific TTL."""
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)


# Global HuggingFace Hub cache instance
_global_hf_cache: Optional[HuggingFaceHubCache] = None
_hf_cache_lock = threading.Lock()


def get_global_hf_hub_cache() -> HuggingFaceHubCache:
    """Get or create the global HuggingFace Hub cache instance."""
    global _global_hf_cache
    
    with _hf_cache_lock:
        if _global_hf_cache is None:
            secret = get_provider_cache_secret("huggingface")
            _global_hf_cache = HuggingFaceHubCache(
                enable_p2p=bool(secret),
                p2p_shared_secret=secret,
                p2p_secret_salt=b"huggingface-hub-task-p2p-cache",
                enable_pubsub=bool(secret),
            )
            from .base_cache import register_cache
            register_cache("huggingface_hub", _global_hf_cache)
        
        return _global_hf_cache


def configure_hf_hub_cache(**kwargs) -> HuggingFaceHubCache:
    """
    Configure the global HuggingFace Hub cache.
    
    Args:
        **kwargs: Arguments to pass to HuggingFaceHubCache constructor
        
    Returns:
        Configured HuggingFace Hub cache instance
    """
    global _global_hf_cache
    
    with _hf_cache_lock:
        if _global_hf_cache is not None:
            _global_hf_cache.shutdown()
        
        _global_hf_cache = HuggingFaceHubCache(**kwargs)
        from .base_cache import register_cache
        register_cache("huggingface_hub", _global_hf_cache)
        
        return _global_hf_cache
