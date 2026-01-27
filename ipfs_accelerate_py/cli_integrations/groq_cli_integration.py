"""
Groq CLI Integration with Common Cache

Wraps Groq CLI to use the common cache infrastructure.
"""

import logging
from typing import Any, Dict, Optional

from .base_cli_wrapper import BaseCLIWrapper
from ..common.llm_cache import LLMAPICache, get_global_llm_cache

logger = logging.getLogger(__name__)


class GroqCLIIntegration(BaseCLIWrapper):
    """
    Groq CLI integration with common cache infrastructure.
    
    Uses LLM cache for fast inference requests.
    """
    
    def __init__(
        self,
        groq_path: str = "groq",
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        **kwargs
    ):
        """
        Initialize Groq CLI integration.
        
        Args:
            groq_path: Path to groq CLI executable
            enable_cache: Whether to enable caching
            cache: Custom cache instance (uses LLM cache if None)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = get_global_llm_cache()
        
        super().__init__(
            cli_path=groq_path,
            cache=cache,
            enable_cache=enable_cache,
            **kwargs
        )
    
    def get_tool_name(self) -> str:
        return "Groq CLI"
    
    def chat(
        self,
        message: str,
        model: str = "llama3-70b-8192",
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to Groq.
        
        Args:
            message: Chat message
            model: Model to use
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with response
        """
        args = ["chat", "--model", model, "--temperature", str(temperature), message]
        
        return self._run_command_with_retry(
            args,
            "chat_completion",
            messages=[{"role": "user", "content": message}],
            model=model,
            temperature=temperature
        )
    
    def complete(
        self,
        prompt: str,
        model: str = "llama3-70b-8192",
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete a prompt with Groq.
        
        Args:
            prompt: Prompt to complete
            model: Model to use
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with completion
        """
        args = ["complete", "--model", model, "--temperature", str(temperature), prompt]
        
        return self._run_command_with_retry(
            args,
            "completion",
            prompt=prompt,
            model=model,
            temperature=temperature
        )


# Global instance
_global_groq_cli: Optional[GroqCLIIntegration] = None


def get_groq_cli_integration() -> GroqCLIIntegration:
    """Get or create the global Groq CLI integration instance."""
    global _global_groq_cli
    
    if _global_groq_cli is None:
        _global_groq_cli = GroqCLIIntegration()
    
    return _global_groq_cli
