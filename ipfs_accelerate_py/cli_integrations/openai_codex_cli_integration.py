"""
OpenAI Codex CLI Integration with Common Cache

Wraps OpenAI Codex CLI to use the common cache infrastructure.
Note: OpenAI Codex CLI is deprecated, but we provide this for compatibility.
Uses the LLM cache for code generation requests.
"""

import logging
from typing import Any, Dict, Optional

from .base_cli_wrapper import BaseCLIWrapper
from ..common.llm_cache import LLMAPICache, get_global_llm_cache

logger = logging.getLogger(__name__)


class OpenAICodexCLIIntegration(BaseCLIWrapper):
    """
    OpenAI Codex CLI integration with common cache infrastructure.
    
    Uses LLM cache for code generation.
    """
    
    def __init__(
        self,
        codex_path: str = "openai",
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        **kwargs
    ):
        """
        Initialize OpenAI Codex CLI integration.
        
        Args:
            codex_path: Path to openai CLI executable
            enable_cache: Whether to enable caching
            cache: Custom cache instance (uses LLM cache if None)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = get_global_llm_cache()
        
        super().__init__(
            cli_path=codex_path,
            cache=cache,
            enable_cache=enable_cache,
            **kwargs
        )
    
    def get_tool_name(self) -> str:
        return "OpenAI Codex CLI"
    
    def generate_code(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate code from prompt.
        
        Args:
            prompt: Code generation prompt
            model: Model to use
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with generated code
        """
        args = ["api", "completions.create", "-m", model, "-p", prompt, "-t", str(temperature)]
        
        return self._run_command_with_retry(
            args,
            "completion",
            prompt=prompt,
            model=model,
            temperature=temperature
        )


# Global instance
_global_openai_codex_cli: Optional[OpenAICodexCLIIntegration] = None


def get_openai_codex_cli_integration() -> OpenAICodexCLIIntegration:
    """Get or create the global OpenAI Codex CLI integration instance."""
    global _global_openai_codex_cli
    
    if _global_openai_codex_cli is None:
        _global_openai_codex_cli = OpenAICodexCLIIntegration()
    
    return _global_openai_codex_cli
