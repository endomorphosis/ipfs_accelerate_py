"""
HuggingFace CLI Integration with Common Cache

Wraps HuggingFace CLI to use the common cache infrastructure.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_cli_wrapper import BaseCLIWrapper
from ..common.hf_hub_cache import HuggingFaceHubCache, get_global_hf_hub_cache

logger = logging.getLogger(__name__)


class HuggingFaceCLIIntegration(BaseCLIWrapper):
    """
    HuggingFace CLI integration with common cache infrastructure.
    
    Uses HuggingFace Hub cache for model/dataset operations.
    """
    
    def __init__(
        self,
        hf_path: str = "huggingface-cli",
        enable_cache: bool = True,
        cache: Optional[HuggingFaceHubCache] = None,
        **kwargs
    ):
        """
        Initialize HuggingFace CLI integration.
        
        Args:
            hf_path: Path to huggingface-cli executable
            enable_cache: Whether to enable caching
            cache: Custom cache instance (uses HF Hub cache if None)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = get_global_hf_hub_cache()
        
        super().__init__(
            cli_path=hf_path,
            cache=cache,
            enable_cache=enable_cache,
            **kwargs
        )
    
    def get_tool_name(self) -> str:
        return "HuggingFace CLI"
    
    def list_models(
        self,
        search: Optional[str] = None,
        limit: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        List models from HuggingFace Hub.
        
        Args:
            search: Search query
            limit: Maximum number of models to return
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with models
        """
        args = ["models", "list"]
        if search:
            args.extend(["--search", search])
        args.extend(["--limit", str(limit)])
        
        return self._run_command_with_retry(
            args,
            "search_models",
            search=search or "",
            limit=limit
        )
    
    def model_info(
        self,
        model_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b-hf")
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with model info
        """
        args = ["repo", "info", model_id]
        
        return self._run_command_with_retry(
            args,
            "model_info",
            model=model_id
        )
    
    def list_datasets(
        self,
        search: Optional[str] = None,
        limit: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        List datasets from HuggingFace Hub.
        
        Args:
            search: Search query
            limit: Maximum number of datasets to return
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with datasets
        """
        args = ["datasets", "list"]
        if search:
            args.extend(["--search", search])
        args.extend(["--limit", str(limit)])
        
        return self._run_command_with_retry(
            args,
            "search_datasets",
            search=search or "",
            limit=limit
        )
    
    def download_model(
        self,
        model_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Download a model.
        
        Args:
            model_id: Model identifier
            **kwargs: Additional arguments
            
        Returns:
            Command result dict
        """
        args = ["download", model_id]
        
        # Don't cache download operations
        return self._run_command_with_retry(
            args,
            "download_model",
            model=model_id
        )


# Global instance
_global_hf_cli: Optional[HuggingFaceCLIIntegration] = None


def get_huggingface_cli_integration() -> HuggingFaceCLIIntegration:
    """Get or create the global HuggingFace CLI integration instance."""
    global _global_hf_cli
    
    if _global_hf_cli is None:
        _global_hf_cli = HuggingFaceCLIIntegration()
    
    return _global_hf_cli
