"""
HuggingFace Hugs Cache

Cache adapter for HuggingFace Hugs API responses.
Caches model/dataset information from the Hugs API.
"""

import logging
import threading
from typing import Any, Dict, Optional

from .base_cache import BaseAPICache

logger = logging.getLogger(__name__)


class HuggingFaceHugsCache(BaseAPICache):
    """
    Cache for HuggingFace Hugs API responses.
    
    Caches:
    - Model metadata and cards
    - Dataset information
    - User profiles and organizations
    - Space information
    - Discussion threads
    - Model/dataset file listings
    """
    
    # Default TTLs for different operations (in seconds)
    DEFAULT_TTLS = {
        "model_info": 3600,  # 1 hour (model info relatively stable)
        "model_list": 1800,  # 30 minutes
        "model_card": 3600,  # 1 hour
        "model_files": 1800,  # 30 minutes
        "model_tags": 3600,  # 1 hour
        "dataset_info": 3600,  # 1 hour
        "dataset_list": 1800,  # 30 minutes
        "dataset_card": 3600,  # 1 hour
        "dataset_files": 1800,  # 30 minutes
        "user_profile": 7200,  # 2 hours (user info rarely changes)
        "org_info": 7200,  # 2 hours
        "space_info": 1800,  # 30 minutes
        "space_list": 1800,  # 30 minutes
        "discussion_thread": 300,  # 5 minutes (discussions can be active)
        "trending_models": 600,  # 10 minutes
        "trending_datasets": 600,  # 10 minutes
        "model_downloads": 1800,  # 30 minutes (download counts update periodically)
        "model_likes": 600,  # 10 minutes (likes can change more frequently)
    }
    
    def get_cache_namespace(self) -> str:
        """Get cache namespace."""
        return "huggingface_hugs"
    
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        """
        Extract validation fields from HuggingFace Hugs API response.
        
        Args:
            operation: Operation name
            data: API response data
            
        Returns:
            Validation fields dictionary or None
        """
        if not data or not isinstance(data, dict):
            return None
        
        validation = {}
        
        # Model operations
        if operation.startswith("model_"):
            validation["modelId"] = data.get("modelId") or data.get("id")
            validation["sha"] = data.get("sha")
            validation["lastModified"] = data.get("lastModified")
            validation["downloads"] = data.get("downloads")
            validation["likes"] = data.get("likes")
            validation["private"] = data.get("private")
            
            # Pipeline tag for model type
            validation["pipeline_tag"] = data.get("pipeline_tag")
            
            # Library info
            validation["library_name"] = data.get("library_name")
            
            # Tags
            tags = data.get("tags", [])
            if isinstance(tags, list):
                validation["tag_count"] = len(tags)
            
            # Siblings (files)
            siblings = data.get("siblings", [])
            if isinstance(siblings, list):
                validation["file_count"] = len(siblings)
                # Total size if available
                total_size = sum(
                    s.get("size", 0) for s in siblings 
                    if isinstance(s, dict) and isinstance(s.get("size"), (int, float))
                )
                if total_size > 0:
                    validation["total_size"] = total_size
        
        # Dataset operations
        elif operation.startswith("dataset_"):
            validation["datasetId"] = data.get("id")
            validation["sha"] = data.get("sha")
            validation["lastModified"] = data.get("lastModified")
            validation["downloads"] = data.get("downloads")
            validation["likes"] = data.get("likes")
            validation["private"] = data.get("private")
            
            # Tags
            tags = data.get("tags", [])
            if isinstance(tags, list):
                validation["tag_count"] = len(tags)
            
            # Siblings (files)
            siblings = data.get("siblings", [])
            if isinstance(siblings, list):
                validation["file_count"] = len(siblings)
        
        # User/Org operations
        elif operation in ("user_profile", "org_info"):
            validation["name"] = data.get("name")
            validation["fullname"] = data.get("fullname")
            validation["avatarUrl"] = data.get("avatarUrl")
            validation["numModels"] = data.get("numModels")
            validation["numDatasets"] = data.get("numDatasets")
            validation["numSpaces"] = data.get("numSpaces")
        
        # Space operations
        elif operation.startswith("space_"):
            validation["id"] = data.get("id")
            validation["author"] = data.get("author")
            validation["sha"] = data.get("sha")
            validation["lastModified"] = data.get("lastModified")
            validation["likes"] = data.get("likes")
            validation["sdk"] = data.get("sdk")  # gradio, streamlit, etc.
            validation["private"] = data.get("private")
        
        # Discussion operations
        elif operation.startswith("discussion_"):
            validation["id"] = data.get("id")
            validation["title"] = data.get("title")
            validation["status"] = data.get("status")
            validation["num_comments"] = data.get("num_comments")
            validation["created_at"] = data.get("created_at")
            validation["updated_at"] = data.get("updated_at")
        
        # List operations - include count
        if operation.endswith("_list"):
            if isinstance(data, list):
                validation["item_count"] = len(data)
            elif isinstance(data, dict) and "items" in data:
                items = data.get("items", [])
                if isinstance(items, list):
                    validation["item_count"] = len(items)
        
        return validation if validation else None
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        """Get operation-specific TTL."""
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)


# Global HuggingFace Hugs cache instance
_global_hugs_cache: Optional[HuggingFaceHugsCache] = None
_hugs_cache_lock = threading.Lock()


def get_global_hugs_cache() -> HuggingFaceHugsCache:
    """Get or create the global HuggingFace Hugs cache instance."""
    global _global_hugs_cache
    
    with _hugs_cache_lock:
        if _global_hugs_cache is None:
            _global_hugs_cache = HuggingFaceHugsCache()
            from .base_cache import register_cache
            register_cache("huggingface_hugs", _global_hugs_cache)
        
        return _global_hugs_cache


def configure_hugs_cache(**kwargs) -> HuggingFaceHugsCache:
    """
    Configure the global HuggingFace Hugs cache.
    
    Args:
        **kwargs: Arguments to pass to HuggingFaceHugsCache constructor
        
    Returns:
        Configured HuggingFace Hugs cache instance
    """
    global _global_hugs_cache
    
    with _hugs_cache_lock:
        if _global_hugs_cache is not None:
            _global_hugs_cache.shutdown()
        
        _global_hugs_cache = HuggingFaceHugsCache(**kwargs)
        from .base_cache import register_cache
        register_cache("huggingface_hugs", _global_hugs_cache)
        
        return _global_hugs_cache
