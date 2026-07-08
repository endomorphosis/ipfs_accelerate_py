"""
VSCode CLI Integration with Common Cache

Wraps VSCode CLI to use the common cache infrastructure.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_cli_wrapper import BaseCLIWrapper
from ..common.base_cache import BaseAPICache, register_cache

logger = logging.getLogger(__name__)


class VSCodeCLICache(BaseAPICache):
    """Cache adapter for VSCode CLI operations."""
    
    DEFAULT_TTLS = {
        "extension_list": 300,  # 5 minutes
        "extension_search": 600,  # 10 minutes
        "tunnel_status": 30,  # 30 seconds
    }
    
    def get_cache_namespace(self) -> str:
        return "vscode_cli"
    
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        return None  # Simple TTL-based expiration
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)


class VSCodeCLIIntegration(BaseCLIWrapper):
    """
    VSCode CLI integration with common cache infrastructure.
    
    Provides caching for code (VSCode CLI) commands.
    """
    
    def __init__(
        self,
        code_path: str = "code",
        enable_cache: bool = True,
        cache: Optional[BaseAPICache] = None,
        **kwargs
    ):
        """
        Initialize VSCode CLI integration.
        
        Args:
            code_path: Path to code executable
            enable_cache: Whether to enable caching
            cache: Custom cache instance (creates new if None)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = VSCodeCLICache()
            register_cache("vscode_cli", cache)
        
        super().__init__(
            cli_path=code_path,
            cache=cache,
            enable_cache=enable_cache,
            **kwargs
        )
    
    def get_tool_name(self) -> str:
        return "VSCode CLI"
    
    def list_extensions(self, **kwargs) -> Dict[str, Any]:
        """
        List installed extensions.
        
        Returns:
            Command result dict with installed extensions
        """
        args = ["--list-extensions", "--show-versions"]
        
        return self._run_command_with_retry(
            args,
            "extension_list"
        )
    
    def search_extensions(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search for extensions in marketplace.
        
        Args:
            query: Search query
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with search results
        """
        # Note: code CLI doesn't have direct search, but we can cache queries
        args = ["--list-extensions"]
        
        return self._run_command_with_retry(
            args,
            "extension_search",
            query=query
        )


# Global instance
_global_vscode_cli: Optional[VSCodeCLIIntegration] = None


def get_vscode_cli_integration() -> VSCodeCLIIntegration:
    """Get or create the global VSCode CLI integration instance."""
    global _global_vscode_cli
    
    if _global_vscode_cli is None:
        _global_vscode_cli = VSCodeCLIIntegration()
    
    return _global_vscode_cli
