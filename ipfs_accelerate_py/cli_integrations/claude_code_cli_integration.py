"""
Claude Code CLI Integration with Common Cache

Wraps Claude API (via Anthropic Python SDK) to use the common cache infrastructure.
Note: Claude does not have an official CLI tool - this uses the Python SDK directly.
"""

import logging
from typing import Any, Dict, Optional

from ..common.llm_cache import LLMAPICache, get_global_llm_cache

logger = logging.getLogger(__name__)


class ClaudeCodeCLIIntegration:
    """
    Claude Code integration with common cache infrastructure.
    
    Uses Python SDK (anthropic) with LLM cache for code generation and chat.
    Note: This is NOT a CLI wrapper - Claude has no official CLI.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        **kwargs
    ):
        """
        Initialize Claude Code integration.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            enable_cache: Whether to enable caching
            cache: Custom cache instance (uses LLM cache if None)
            **kwargs: Additional arguments
        """
        self.api_key = api_key
        self.enable_cache = enable_cache
        
        if cache is None:
            cache = get_global_llm_cache()
        self.cache = cache
        
        # Lazy import anthropic SDK
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic SDK not installed. Install with: pip install anthropic"
                )
        return self._client
    
    def get_tool_name(self) -> str:
        return "Claude (Anthropic SDK)"
    
    def chat(
        self,
        message: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to Claude.
        
        Args:
            message: Chat message
            model: Model to use
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Dict with response
        """
        messages = [{"role": "user", "content": message}]
        
        # Check cache first
        if self.enable_cache:
            cached = self.cache.get_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature
            )
            if cached:
                logger.info("Cache hit for Claude chat")
                return {"response": cached, "cached": True}
        
        # Call API
        client = self._get_client()
        response = client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", 4096)
        )
        
        result = response.content[0].text
        
        # Cache response
        if self.enable_cache:
            self.cache.cache_chat_completion(
                messages=messages,
                response=result,
                model=model,
                temperature=temperature
            )
        
        return {"response": result, "cached": False}
    
    def generate_code(
        self,
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate code from prompt.
        
        Args:
            prompt: Code generation prompt
            model: Model to use
            **kwargs: Additional arguments
            
        Returns:
            Dict with generated code
        """
        return self.chat(prompt, model=model, temperature=0.0)


# Global instance
_global_claude_code_cli: Optional[ClaudeCodeCLIIntegration] = None


def get_claude_code_cli_integration() -> ClaudeCodeCLIIntegration:
    """Get or create the global Claude Code CLI integration instance."""
    global _global_claude_code_cli
    
    if _global_claude_code_cli is None:
        _global_claude_code_cli = ClaudeCodeCLIIntegration()
    
    return _global_claude_code_cli
