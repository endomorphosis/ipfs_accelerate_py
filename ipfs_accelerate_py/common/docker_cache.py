"""
Docker API Cache

Cache adapter for Docker API responses.
Caches image metadata, container status, and other Docker queries.
"""

import logging
import threading
from typing import Any, Dict, Optional

from .base_cache import BaseAPICache

logger = logging.getLogger(__name__)


class DockerAPICache(BaseAPICache):
    """
    Cache for Docker API responses.
    
    Caches:
    - Image metadata (inspection, history)
    - Container status
    - Volume info
    - Network info
    - Registry queries
    """
    
    # Default TTLs for different operations (in seconds)
    DEFAULT_TTLS = {
        "image_inspect": 1800,  # 30 minutes (image metadata is relatively stable)
        "image_history": 3600,  # 1 hour (history doesn't change)
        "image_search": 600,  # 10 minutes
        "container_inspect": 30,  # 30 seconds (container state changes frequently)
        "container_list": 30,  # 30 seconds
        "volume_inspect": 300,  # 5 minutes
        "network_inspect": 300,  # 5 minutes
        "registry_tags": 1800,  # 30 minutes (tags can be added)
    }
    
    def get_cache_namespace(self) -> str:
        """Get cache namespace."""
        return "docker_api"
    
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        """
        Extract validation fields from Docker API response.
        
        Args:
            operation: Operation name
            data: API response data
            
        Returns:
            Validation fields dictionary or None
        """
        if not data or not isinstance(data, dict):
            return None
        
        validation = {}
        
        # Image operations - use Id and Created timestamp
        if operation.startswith("image_"):
            validation["Id"] = data.get("Id")
            validation["Created"] = data.get("Created")
            validation["Size"] = data.get("Size")
            # For image history, also include layers
            if operation == "image_history":
                if isinstance(data.get("History"), list):
                    validation["layers"] = len(data["History"])
        
        # Container operations - use State and timestamps
        elif operation.startswith("container_"):
            validation["Id"] = data.get("Id")
            validation["State"] = data.get("State", {}).get("Status") if isinstance(data.get("State"), dict) else None
            validation["Created"] = data.get("Created")
            # Container status changes frequently, so also include:
            validation["Running"] = data.get("State", {}).get("Running") if isinstance(data.get("State"), dict) else None
            validation["Paused"] = data.get("State", {}).get("Paused") if isinstance(data.get("State"), dict) else None
            validation["Restarting"] = data.get("State", {}).get("Restarting") if isinstance(data.get("State"), dict) else None
        
        # Volume operations
        elif operation.startswith("volume_"):
            validation["Name"] = data.get("Name")
            validation["CreatedAt"] = data.get("CreatedAt")
            validation["Mountpoint"] = data.get("Mountpoint")
        
        # Network operations
        elif operation.startswith("network_"):
            validation["Id"] = data.get("Id")
            validation["Created"] = data.get("Created")
            # Network topology can change
            containers = data.get("Containers", {})
            validation["connected_containers"] = len(containers) if isinstance(containers, dict) else 0
        
        # Registry tags
        elif operation == "registry_tags":
            if isinstance(data, list):
                validation["tags"] = sorted(data)  # List of tags
        
        return validation if validation else None
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        """Get operation-specific TTL."""
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)


# Global Docker API cache instance
_global_docker_cache: Optional[DockerAPICache] = None
_docker_cache_lock = threading.Lock()


def get_global_docker_cache() -> DockerAPICache:
    """Get or create the global Docker API cache instance."""
    global _global_docker_cache
    
    with _docker_cache_lock:
        if _global_docker_cache is None:
            _global_docker_cache = DockerAPICache()
            from .base_cache import register_cache
            register_cache("docker_api", _global_docker_cache)
        
        return _global_docker_cache


def configure_docker_cache(**kwargs) -> DockerAPICache:
    """
    Configure the global Docker API cache.
    
    Args:
        **kwargs: Arguments to pass to DockerAPICache constructor
        
    Returns:
        Configured Docker API cache instance
    """
    global _global_docker_cache
    
    with _docker_cache_lock:
        if _global_docker_cache is not None:
            _global_docker_cache.shutdown()
        
        _global_docker_cache = DockerAPICache(**kwargs)
        from .base_cache import register_cache
        register_cache("docker_api", _global_docker_cache)
        
        return _global_docker_cache
