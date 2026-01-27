"""
Claude Code CLI Integration with Common Cache

Wraps Claude Code CLI (via Anthropic CLI) to use the common cache infrastructure.
"""

import logging
from typing import Any, Dict, Optional

from .base_cli_wrapper import BaseCLIWrapper
from ..common.llm_cache import LLMAPICache, get_global_llm_cache

logger = logging.getLogger(__name__)


class ClaudeCodeCLIIntegration(BaseCLIWrapper):
    """
    Claude Code CLI integration with common cache infrastructure.
    
    Uses LLM cache for code generation and chat.
    """
    
    def __init__(
        self,
        claude_path: str = "claude",
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        **kwargs
    ):
        """
        Initialize Claude Code CLI integration.
        
        Args:
            claude_path: Path to claude CLI executable
            enable_cache: Whether to enable caching
            cache: Custom cache instance (uses LLM cache if None)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = get_global_llm_cache()
        
        super().__init__(
            cli_path=claude_path,
            cache=cache,
            enable_cache=enable_cache,
            **kwargs
        )
    
    def get_tool_name(self) -> str:
        return "Claude Code CLI"
    
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
            Command result dict with response
        """
        args = ["chat", "--model", model, "--temperature", str(temperature), "--", message]
        
        return self._run_command_with_retry(
            args,
            "chat_completion",
            messages=[{"role": "user", "content": message}],
            model=model,
            temperature=temperature
        )
    
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
            Command result dict with generated code
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
