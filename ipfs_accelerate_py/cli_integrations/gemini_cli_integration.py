"""
Gemini CLI Integration with Common Cache

Wraps Google Gemini CLI to use the common cache infrastructure.
"""

import logging
from typing import Any, Dict, Optional

from .base_cli_wrapper import BaseCLIWrapper
from ..common.llm_cache import LLMAPICache, get_global_llm_cache

logger = logging.getLogger(__name__)


class GeminiCLIIntegration(BaseCLIWrapper):
    """
    Gemini CLI integration with common cache infrastructure.
    
    Uses LLM cache for text generation.
    """
    
    def __init__(
        self,
        gemini_path: str = "gemini",
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        **kwargs
    ):
        """
        Initialize Gemini CLI integration.
        
        Args:
            gemini_path: Path to gemini CLI executable
            enable_cache: Whether to enable caching
            cache: Custom cache instance (uses LLM cache if None)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = get_global_llm_cache()
        
        super().__init__(
            cli_path=gemini_path,
            cache=cache,
            enable_cache=enable_cache,
            **kwargs
        )
    
    def get_tool_name(self) -> str:
        return "Gemini CLI"
    
    def generate_text(
        self,
        prompt: str,
        model: str = "gemini-pro",
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Text generation prompt
            model: Model to use
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with generated text
        """
        args = ["generate", "--model", model, "--temperature", str(temperature), prompt]
        
        return self._run_command_with_retry(
            args,
            "completion",
            prompt=prompt,
            model=model,
            temperature=temperature
        )
    
    def chat(
        self,
        message: str,
        model: str = "gemini-pro",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to Gemini.
        
        Args:
            message: Chat message
            model: Model to use
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with response
        """
        args = ["chat", "--model", model, message]
        
        return self._run_command_with_retry(
            args,
            "chat_completion",
            messages=[{"role": "user", "content": message}],
            model=model
        )


# Global instance
_global_gemini_cli: Optional[GeminiCLIIntegration] = None


def get_gemini_cli_integration() -> GeminiCLIIntegration:
    """Get or create the global Gemini CLI integration instance."""
    global _global_gemini_cli
    
    if _global_gemini_cli is None:
        _global_gemini_cli = GeminiCLIIntegration()
    
    return _global_gemini_cli
