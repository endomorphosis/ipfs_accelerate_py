"""
IPFS Kit Fallback Store

Integrates with endomorphosis/ipfs_kit_py@known_good as a fallback CID store.
Allows cache to retrieve data from decentralized storage infrastructure when local cache misses.
"""

import json
import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class IPFSKitFallbackStore:
    """
    Fallback store that uses ipfs_kit_py for retrieving cached data from IPFS.
    
    When a cache miss occurs locally, this store attempts to retrieve the data
    from the decentralized IPFS network using the CID.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        ipfs_gateway: Optional[str] = None,
        timeout: int = 10,
        use_local_node: bool = True,
    ):
        """
        Initialize IPFS Kit fallback store.
        
        Args:
            enabled: Whether fallback is enabled
            ipfs_gateway: IPFS gateway URL (default: None, uses ipfs_kit_py default)
            timeout: Timeout for IPFS operations in seconds
            use_local_node: Whether to use local IPFS node if available
        """
        self.enabled = enabled
        self.ipfs_gateway = ipfs_gateway
        self.timeout = timeout
        self.use_local_node = use_local_node
        self._lock = threading.Lock()
        
        # Try to import ipfs_kit_py
        self.ipfs_client = None
        self.ipfs_available = False
        
        if enabled:
            self._init_ipfs_client()
    
    def _init_ipfs_client(self):
        """Initialize IPFS client from ipfs_kit_py."""
        try:
            # Try importing from ipfs_kit_py@known_good
            import ipfs_kit_py
            
            # Try different import patterns
            try:
                from ipfs_kit_py import IPFSApi
                self.ipfs_client = IPFSApi()
                self.ipfs_available = True
                logger.info("IPFS Kit fallback store initialized with IPFSApi")
            except (ImportError, AttributeError):
                try:
                    from ipfs_kit_py import IPFSSimpleAPI
                    self.ipfs_client = IPFSSimpleAPI()
                    self.ipfs_available = True
                    logger.info("IPFS Kit fallback store initialized with IPFSSimpleAPI")
                except (ImportError, AttributeError):
                    try:
                        # Try getting high-level API
                        get_high_level_api = getattr(ipfs_kit_py, "get_high_level_api", None)
                        if get_high_level_api:
                            self.ipfs_client = get_high_level_api()
                            self.ipfs_available = True
                            logger.info("IPFS Kit fallback store initialized with high-level API")
                    except Exception as e:
                        logger.warning(f"Could not initialize IPFS Kit high-level API: {e}")
            
            # If still not available, try ipfs_client module
            if not self.ipfs_available:
                try:
                    from ipfs_kit_py.ipfs_client import ipfs_py
                    self.ipfs_client = ipfs_py()
                    self.ipfs_available = True
                    logger.info("IPFS Kit fallback store initialized with ipfs_py")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not initialize IPFS Kit ipfs_py: {e}")
        
        except ImportError as e:
            logger.info(f"ipfs_kit_py not available ({e}). Fallback store disabled.")
            self.enabled = False
            self.ipfs_available = False
        except Exception as e:
            logger.warning(f"Error initializing IPFS Kit client: {e}")
            self.enabled = False
            self.ipfs_available = False
    
    def is_available(self) -> bool:
        """Check if IPFS fallback is available."""
        return self.enabled and self.ipfs_available and self.ipfs_client is not None
    
    def get(self, cid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data from IPFS using CID.
        
        Args:
            cid: Content identifier to retrieve
            
        Returns:
            Cached data dictionary or None if not found
        """
        if not self.is_available():
            return None
        
        try:
            with self._lock:
                # Try to get data from IPFS
                if hasattr(self.ipfs_client, 'cat'):
                    # Use cat method to retrieve content
                    content = self.ipfs_client.cat(cid, timeout=self.timeout)
                elif hasattr(self.ipfs_client, 'get'):
                    # Use get method
                    content = self.ipfs_client.get(cid, timeout=self.timeout)
                else:
                    logger.warning("IPFS client doesn't have cat or get method")
                    return None
                
                # Parse JSON content
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                
                data = json.loads(content)
                logger.info(f"Successfully retrieved cache data from IPFS for CID: {cid}")
                return data
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from IPFS CID {cid}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to retrieve from IPFS fallback for CID {cid}: {e}")
            return None
    
    def put(self, cid: str, data: Dict[str, Any]) -> bool:
        """
        Store cached data to IPFS.
        
        Args:
            cid: Content identifier (expected CID for verification)
            data: Data to store
            
        Returns:
            True if successfully stored, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            with self._lock:
                # Serialize data to JSON
                content = json.dumps(data, sort_keys=True)
                
                # Store to IPFS
                if hasattr(self.ipfs_client, 'add'):
                    # Use add method
                    result = self.ipfs_client.add(content.encode('utf-8'))
                    
                    # Extract CID from result
                    stored_cid = None
                    if isinstance(result, dict):
                        stored_cid = result.get('Hash') or result.get('cid')
                    elif isinstance(result, str):
                        stored_cid = result
                    
                    # Verify CID matches (if possible)
                    if stored_cid:
                        logger.info(f"Stored cache data to IPFS. Expected CID: {cid}, Got: {stored_cid}")
                        return True
                    else:
                        logger.warning("Failed to get CID from IPFS add result")
                        return False
                else:
                    logger.warning("IPFS client doesn't have add method")
                    return False
        
        except Exception as e:
            logger.debug(f"Failed to store to IPFS fallback for CID {cid}: {e}")
            return False
    
    def pin(self, cid: str) -> bool:
        """
        Pin content in IPFS to prevent garbage collection.
        
        Args:
            cid: Content identifier to pin
            
        Returns:
            True if successfully pinned, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            with self._lock:
                if hasattr(self.ipfs_client, 'pin_add') or hasattr(self.ipfs_client, 'pin'):
                    pin_method = getattr(self.ipfs_client, 'pin_add', None) or getattr(self.ipfs_client, 'pin', None)
                    result = pin_method(cid)
                    logger.info(f"Pinned CID in IPFS: {cid}")
                    return True
                else:
                    logger.debug("IPFS client doesn't have pin method")
                    return False
        
        except Exception as e:
            logger.debug(f"Failed to pin CID {cid} in IPFS: {e}")
            return False
    
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about IPFS fallback store.
        
        Returns:
            Statistics dictionary
        """
        return {
            "enabled": self.enabled,
            "available": self.is_available(),
            "ipfs_gateway": self.ipfs_gateway,
            "timeout": self.timeout,
            "use_local_node": self.use_local_node,
        }


# Global IPFS fallback store
_global_ipfs_fallback: Optional[IPFSKitFallbackStore] = None
_ipfs_fallback_lock = threading.Lock()


def get_global_ipfs_fallback() -> IPFSKitFallbackStore:
    """Get or create global IPFS fallback store."""
    global _global_ipfs_fallback
    
    with _ipfs_fallback_lock:
        if _global_ipfs_fallback is None:
            # Check if explicitly disabled via environment variable
            enabled = os.getenv("IPFS_FALLBACK_ENABLED", "true").lower() in ("true", "1", "yes")
            _global_ipfs_fallback = IPFSKitFallbackStore(enabled=enabled)
        
        return _global_ipfs_fallback


def configure_ipfs_fallback(**kwargs) -> IPFSKitFallbackStore:
    """
    Configure global IPFS fallback store.
    
    Args:
        **kwargs: Arguments to pass to IPFSKitFallbackStore constructor
        
    Returns:
        Configured IPFS fallback store
    """
    global _global_ipfs_fallback
    
    with _ipfs_fallback_lock:
        _global_ipfs_fallback = IPFSKitFallbackStore(**kwargs)
        return _global_ipfs_fallback
