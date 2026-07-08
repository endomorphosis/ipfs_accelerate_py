"""
Storage and IPFS API Cache Integrations

Cache-enabled wrappers for:
- S3/Object Storage API
- IPFS API operations
"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from ..common.base_cache import BaseAPICache, register_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("Cache infrastructure not available")


class S3Cache(BaseAPICache):
    """Cache adapter for S3/Object Storage operations."""
    
    DEFAULT_TTLS = {
        "list_objects": 300,  # 5 minutes
        "head_object": 600,  # 10 minutes
        "get_object_url": 3600,  # 1 hour
    }
    
    def get_cache_namespace(self) -> str:
        return "s3_storage"
    
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        """Extract validation fields from S3 responses."""
        if not isinstance(data, dict):
            return None
        
        # Use ETag for validation if available
        if 'ETag' in data:
            return {'etag': data['ETag']}
        
        return None
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)


class IPFSCache(BaseAPICache):
    """Cache adapter for IPFS operations."""
    
    DEFAULT_TTLS = {
        "get_metadata": 3600,  # 1 hour
        "dht_query": 600,  # 10 minutes
        "pin_status": 300,  # 5 minutes
    }
    
    def get_cache_namespace(self) -> str:
        return "ipfs_api"
    
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        return None  # Content is already content-addressed
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)


class CachedS3API:
    """
    Cache-enabled wrapper for S3/Object Storage API.
    
    Caches metadata operations only (not downloads).
    """
    
    def __init__(self, api_instance, cache: Optional[BaseAPICache] = None):
        self._api = api_instance
        if cache is None and CACHE_AVAILABLE:
            cache = S3Cache()
            register_cache("s3_storage", cache)
        self._cache = cache
    
    def __getattr__(self, name):
        return getattr(self._api, name)
    
    def list_objects(self, bucket: str, prefix: str = "",
                    use_cache: bool = True, **kwargs) -> Any:
        """
        List objects with caching.
        
        Args:
            bucket: Bucket name
            prefix: Object prefix filter
            use_cache: Whether to use cache
            **kwargs: Additional arguments
            
        Returns:
            List of objects
        """
        if not use_cache or not self._cache:
            return self._api.list_objects(bucket=bucket, prefix=prefix, **kwargs)
        
        # Check cache
        cached = self._cache.get("list_objects", bucket=bucket, prefix=prefix, **kwargs)
        
        if cached:
            logger.debug(f"S3 Cache HIT for list_objects (bucket={bucket}, prefix={prefix})")
            return cached
        
        # Call API
        logger.debug(f"S3 Cache MISS for list_objects (bucket={bucket}, prefix={prefix})")
        response = self._api.list_objects(bucket=bucket, prefix=prefix, **kwargs)
        
        # Cache response
        self._cache.put("list_objects", response, bucket=bucket, prefix=prefix, **kwargs)
        
        return response
    
    def head_object(self, bucket: str, key: str,
                   use_cache: bool = True, **kwargs) -> Any:
        """
        Get object metadata with caching.
        
        Args:
            bucket: Bucket name
            key: Object key
            use_cache: Whether to use cache
            **kwargs: Additional arguments
            
        Returns:
            Object metadata
        """
        if not use_cache or not self._cache:
            return self._api.head_object(bucket=bucket, key=key, **kwargs)
        
        # Check cache
        cached = self._cache.get("head_object", bucket=bucket, key=key, **kwargs)
        
        if cached:
            logger.debug(f"S3 Cache HIT for head_object (bucket={bucket}, key={key})")
            return cached
        
        # Call API
        logger.debug(f"S3 Cache MISS for head_object (bucket={bucket}, key={key})")
        response = self._api.head_object(bucket=bucket, key=key, **kwargs)
        
        # Cache response
        self._cache.put("head_object", response, bucket=bucket, key=key, **kwargs)
        
        return response
    
    def get_object_url(self, bucket: str, key: str, expires_in: int = 3600,
                      use_cache: bool = True, **kwargs) -> str:
        """
        Get presigned URL with caching.
        
        Args:
            bucket: Bucket name
            key: Object key
            expires_in: URL expiration time (seconds)
            use_cache: Whether to use cache
            **kwargs: Additional arguments
            
        Returns:
            Presigned URL
        """
        if not use_cache or not self._cache:
            return self._api.get_object_url(bucket=bucket, key=key, expires_in=expires_in, **kwargs)
        
        # Check cache (cache for less than expires_in)
        cached = self._cache.get("get_object_url", bucket=bucket, key=key, expires_in=expires_in, **kwargs)
        
        if cached:
            logger.debug(f"S3 Cache HIT for get_object_url (bucket={bucket}, key={key})")
            return cached
        
        # Call API
        logger.debug(f"S3 Cache MISS for get_object_url (bucket={bucket}, key={key})")
        url = self._api.get_object_url(bucket=bucket, key=key, expires_in=expires_in, **kwargs)
        
        # Cache URL for shorter time than expiration
        cache_ttl = min(expires_in - 60, 3600)  # Cache for expiration - 60s or 1 hour, whichever is less
        self._cache.put("get_object_url", url, ttl=cache_ttl, bucket=bucket, key=key, expires_in=expires_in, **kwargs)
        
        return url


class CachedIPFSAPI:
    """
    Cache-enabled wrapper for IPFS API.
    
    Caches metadata and DHT queries (content is already content-addressed).
    """
    
    def __init__(self, api_instance, cache: Optional[BaseAPICache] = None):
        self._api = api_instance
        if cache is None and CACHE_AVAILABLE:
            cache = IPFSCache()
            register_cache("ipfs_api", cache)
        self._cache = cache
    
    def __getattr__(self, name):
        return getattr(self._api, name)
    
    def get_metadata(self, cid: str, use_cache: bool = True, **kwargs) -> Any:
        """
        Get IPFS object metadata with caching.
        
        Args:
            cid: Content identifier
            use_cache: Whether to use cache
            **kwargs: Additional arguments
            
        Returns:
            Object metadata
        """
        if not use_cache or not self._cache:
            return self._api.get_metadata(cid=cid, **kwargs)
        
        # Check cache
        cached = self._cache.get("get_metadata", cid=cid, **kwargs)
        
        if cached:
            logger.debug(f"IPFS Cache HIT for get_metadata (cid={cid})")
            return cached
        
        # Call API
        logger.debug(f"IPFS Cache MISS for get_metadata (cid={cid})")
        response = self._api.get_metadata(cid=cid, **kwargs)
        
        # Cache response
        self._cache.put("get_metadata", response, cid=cid, **kwargs)
        
        return response
    
    def dht_findprovs(self, cid: str, use_cache: bool = True, **kwargs) -> Any:
        """
        Find providers for CID with caching.
        
        Args:
            cid: Content identifier
            use_cache: Whether to use cache
            **kwargs: Additional arguments
            
        Returns:
            List of providers
        """
        if not use_cache or not self._cache:
            return self._api.dht_findprovs(cid=cid, **kwargs)
        
        # Check cache
        cached = self._cache.get("dht_query", cid=cid, **kwargs)
        
        if cached:
            logger.debug(f"IPFS Cache HIT for dht_findprovs (cid={cid})")
            return cached
        
        # Call API
        logger.debug(f"IPFS Cache MISS for dht_findprovs (cid={cid})")
        response = self._api.dht_findprovs(cid=cid, **kwargs)
        
        # Cache response
        self._cache.put("dht_query", response, cid=cid, **kwargs)
        
        return response
    
    def pin_ls(self, cid: Optional[str] = None, use_cache: bool = True, **kwargs) -> Any:
        """
        List pinned objects with caching.
        
        Args:
            cid: Optional specific CID to check
            use_cache: Whether to use cache
            **kwargs: Additional arguments
            
        Returns:
            Pin status/list
        """
        if not use_cache or not self._cache:
            return self._api.pin_ls(cid=cid, **kwargs)
        
        # Check cache
        cached = self._cache.get("pin_status", cid=cid or "all", **kwargs)
        
        if cached:
            logger.debug(f"IPFS Cache HIT for pin_ls (cid={cid or 'all'})")
            return cached
        
        # Call API
        logger.debug(f"IPFS Cache MISS for pin_ls (cid={cid or 'all'})")
        response = self._api.pin_ls(cid=cid, **kwargs)
        
        # Cache response
        self._cache.put("pin_status", response, cid=cid or "all", **kwargs)
        
        return response


# Factory functions

def get_cached_s3_api(**kwargs):
    """Get cache-enabled S3 API instance."""
    from ..api_backends.s3_kit import s3_kit
    api = s3_kit(**kwargs)
    return CachedS3API(api)


def get_cached_ipfs_api(**kwargs):
    """Get cache-enabled IPFS API instance."""
    # IPFS API might be in different location, adjust as needed
    try:
        from ..ipfs_kit import ipfs_kit
        api = ipfs_kit(**kwargs)
        return CachedIPFSAPI(api)
    except ImportError:
        logger.warning("IPFS API not found, cannot create cached wrapper")
        return None


__all__ = [
    'S3Cache',
    'IPFSCache',
    'CachedS3API',
    'CachedIPFSAPI',
    'get_cached_s3_api',
    'get_cached_ipfs_api',
]
