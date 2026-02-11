"""
Vast AI CLI Integration with Common Cache

Wraps Vast AI CLI to use the common cache infrastructure.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_cli_wrapper import BaseCLIWrapper
from ..common.base_cache import BaseAPICache, register_cache
from ..common.provider_secrets import get_provider_cache_secret

logger = logging.getLogger(__name__)


class VastAICLICache(BaseAPICache):
    """Cache adapter for Vast AI CLI operations."""
    
    DEFAULT_TTLS = {
        "search_offers": 300,  # 5 minutes
        "show_instances": 30,  # 30 seconds
        "show_instance": 30,  # 30 seconds
    }
    
    def get_cache_namespace(self) -> str:
        return "vastai_cli"
    
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        return None  # Simple TTL-based expiration
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)


class VastAICLIIntegration(BaseCLIWrapper):
    """
    Vast AI CLI integration with common cache infrastructure.
    
    Provides caching for vastai CLI commands.
    """
    
    def __init__(
        self,
        vastai_path: str = "vastai",
        enable_cache: bool = True,
        cache: Optional[BaseAPICache] = None,
        **kwargs
    ):
        """
        Initialize Vast AI CLI integration.
        
        Args:
            vastai_path: Path to vastai executable
            enable_cache: Whether to enable caching
            cache: Custom cache instance (creates new if None)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            secret = get_provider_cache_secret("vastai")
            cache = VastAICLICache(
                enable_p2p=bool(secret),
                p2p_shared_secret=secret,
                p2p_secret_salt=b"vastai-cli-task-p2p-cache",
                enable_pubsub=bool(secret),
            )
            register_cache("vastai_cli", cache)
        
        super().__init__(
            cli_path=vastai_path,
            cache=cache,
            enable_cache=enable_cache,
            **kwargs
        )
    
    def get_tool_name(self) -> str:
        return "Vast AI CLI"
    
    def search_offers(
        self,
        query: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search for GPU offers.
        
        Args:
            query: Search query/filter
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with offers
        """
        args = ["search", "offers"]
        if query:
            args.append(query)
        
        return self._run_command_with_retry(
            args,
            "search_offers",
            query=query or "all"
        )
    
    def show_instances(self, **kwargs) -> Dict[str, Any]:
        """
        Show all instances.
        
        Returns:
            Command result dict with instances
        """
        args = ["show", "instances"]
        
        return self._run_command_with_retry(
            args,
            "show_instances"
        )
    
    def show_instance(
        self,
        instance_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Show specific instance details.
        
        Args:
            instance_id: Instance ID
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with instance details
        """
        args = ["show", "instance", instance_id]
        
        return self._run_command_with_retry(
            args,
            "show_instance",
            instance_id=instance_id
        )


# Global instance
_global_vastai_cli: Optional[VastAICLIIntegration] = None


def get_vastai_cli_integration() -> VastAICLIIntegration:
    """Get or create the global Vast AI CLI integration instance."""
    global _global_vastai_cli
    
    if _global_vastai_cli is None:
        _global_vastai_cli = VastAICLIIntegration()
    
    return _global_vastai_cli
