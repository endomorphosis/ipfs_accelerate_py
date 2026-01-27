"""
Kubernetes API Cache

Cache adapter for Kubernetes API responses.
Caches pod status, deployment info, and other Kubernetes queries.
"""

import logging
import threading
from typing import Any, Dict, Optional

from .base_cache import BaseAPICache

logger = logging.getLogger(__name__)


class KubernetesAPICache(BaseAPICache):
    """
    Cache for Kubernetes API responses.
    
    Caches:
    - Pod status and metadata
    - Deployment status
    - Service endpoints
    - ConfigMap and Secret metadata (not content for security)
    - Node information
    - Resource quotas
    """
    
    # Default TTLs for different operations (in seconds)
    DEFAULT_TTLS = {
        "pod_status": 30,  # 30 seconds (pod state changes frequently)
        "pod_list": 30,  # 30 seconds
        "pod_logs": 60,  # 1 minute (logs grow over time)
        "deployment_status": 60,  # 1 minute
        "deployment_list": 60,  # 1 minute
        "service_endpoints": 120,  # 2 minutes
        "service_list": 120,  # 2 minutes
        "configmap_list": 300,  # 5 minutes (metadata only)
        "secret_list": 300,  # 5 minutes (metadata only, not content)
        "node_info": 300,  # 5 minutes (node info is relatively stable)
        "node_list": 300,  # 5 minutes
        "namespace_list": 600,  # 10 minutes (namespaces rarely change)
        "resource_quota": 300,  # 5 minutes
        "statefulset_status": 60,  # 1 minute
        "daemonset_status": 60,  # 1 minute
    }
    
    def get_cache_namespace(self) -> str:
        """Get cache namespace."""
        return "kubernetes_api"
    
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        """
        Extract validation fields from Kubernetes API response.
        
        Args:
            operation: Operation name
            data: API response data
            
        Returns:
            Validation fields dictionary or None
        """
        if not data or not isinstance(data, dict):
            return None
        
        validation = {}
        
        # Extract metadata (common to most K8s resources)
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            validation["name"] = metadata.get("name")
            validation["namespace"] = metadata.get("namespace")
            validation["uid"] = metadata.get("uid")
            validation["resourceVersion"] = metadata.get("resourceVersion")
            validation["generation"] = metadata.get("generation")
        
        # Extract status (common to most K8s resources)
        status = data.get("status", {})
        if isinstance(status, dict):
            # Pod-specific
            if operation.startswith("pod_"):
                validation["phase"] = status.get("phase")
                validation["podIP"] = status.get("podIP")
                
                # Container statuses
                container_statuses = status.get("containerStatuses", [])
                if isinstance(container_statuses, list):
                    validation["ready_containers"] = sum(
                        1 for cs in container_statuses 
                        if isinstance(cs, dict) and cs.get("ready", False)
                    )
                    validation["total_containers"] = len(container_statuses)
                    validation["restart_count"] = sum(
                        cs.get("restartCount", 0) for cs in container_statuses 
                        if isinstance(cs, dict)
                    )
            
            # Deployment-specific
            elif operation.startswith("deployment_"):
                validation["replicas"] = status.get("replicas")
                validation["readyReplicas"] = status.get("readyReplicas")
                validation["updatedReplicas"] = status.get("updatedReplicas")
                validation["availableReplicas"] = status.get("availableReplicas")
                validation["observedGeneration"] = status.get("observedGeneration")
            
            # Service-specific
            elif operation.startswith("service_"):
                validation["clusterIP"] = status.get("loadBalancer", {}).get("ingress", [{}])[0].get("ip") if isinstance(status.get("loadBalancer"), dict) else None
            
            # Node-specific
            elif operation.startswith("node_"):
                conditions = status.get("conditions", [])
                if isinstance(conditions, list):
                    # Check if node is ready
                    for condition in conditions:
                        if isinstance(condition, dict) and condition.get("type") == "Ready":
                            validation["ready"] = condition.get("status") == "True"
                            break
                
                # Node capacity
                capacity = status.get("capacity", {})
                if isinstance(capacity, dict):
                    validation["cpu"] = capacity.get("cpu")
                    validation["memory"] = capacity.get("memory")
                    validation["pods"] = capacity.get("pods")
        
        # For list operations, include count
        if operation.endswith("_list"):
            items = data.get("items", [])
            if isinstance(items, list):
                validation["item_count"] = len(items)
        
        return validation if validation else None
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        """Get operation-specific TTL."""
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)


# Global Kubernetes API cache instance
_global_k8s_cache: Optional[KubernetesAPICache] = None
_k8s_cache_lock = threading.Lock()


def get_global_kubernetes_cache() -> KubernetesAPICache:
    """Get or create the global Kubernetes API cache instance."""
    global _global_k8s_cache
    
    with _k8s_cache_lock:
        if _global_k8s_cache is None:
            _global_k8s_cache = KubernetesAPICache()
            from .base_cache import register_cache
            register_cache("kubernetes_api", _global_k8s_cache)
        
        return _global_k8s_cache


def configure_kubernetes_cache(**kwargs) -> KubernetesAPICache:
    """
    Configure the global Kubernetes API cache.
    
    Args:
        **kwargs: Arguments to pass to KubernetesAPICache constructor
        
    Returns:
        Configured Kubernetes API cache instance
    """
    global _global_k8s_cache
    
    with _k8s_cache_lock:
        if _global_k8s_cache is not None:
            _global_k8s_cache.shutdown()
        
        _global_k8s_cache = KubernetesAPICache(**kwargs)
        from .base_cache import register_cache
        register_cache("kubernetes_api", _global_k8s_cache)
        
        return _global_k8s_cache
