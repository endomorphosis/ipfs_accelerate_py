"""
Claude Code CLI Integration with Common Cache

Wraps Claude API (via Anthropic Python SDK) to use the common cache infrastructure.
Supports dual-mode operation with CLI fallback and secrets manager integration.
Note: Claude does not have an official CLI tool - this primarily uses the Python SDK.

Multi-user / parallel execution
--------------------------------
Pass ``api_keys`` to distribute load across multiple Anthropic keys::

    claude = ClaudeCodeCLIIntegration(api_keys=["sk-ant-1", "sk-ant-2"])
    result = await claude.achat("Hello", user_id="alice")

Async support (Trio / Hypercorn)
---------------------------------
``achat()`` and ``agenerate_code()`` offload the blocking Anthropic SDK call
to a thread via ``anyio.to_thread.run_sync`` so the event loop stays free.
"""

import logging
from typing import Any, Dict, List, Optional

from .dual_mode_wrapper import DualModeWrapper, detect_cli_tool
from ..common.llm_cache import LLMAPICache, get_global_llm_cache, get_llm_cache

logger = logging.getLogger(__name__)


class ClaudeCodeCLIIntegration(DualModeWrapper):
    """
    Claude Code integration with common cache infrastructure.
    
    Supports dual-mode operation:
    - SDK mode: Uses Anthropic Python SDK (primary mode)
    - CLI mode: Falls back to CLI if available (experimental)
    
    Features secrets manager integration for secure API key storage.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        prefer_cli: bool = False,  # Default to SDK since no official CLI
        **kwargs
    ):
        """
        Initialize Claude Code integration.
        
        Args:
            api_key: Anthropic API key (from secrets manager if None)
            api_keys: List of Anthropic API keys for multi-user round-robin pool.
            enable_cache: Whether to enable caching
            cache: Custom cache instance (uses LLM cache if None)
            prefer_cli: Whether to prefer CLI over SDK (default: False)
            **kwargs: Additional arguments
        """
        cache_was_none = cache is None
        if cache_was_none:
            cache = get_global_llm_cache()
        
        super().__init__(
            cli_path=None,  # Will be auto-detected
            api_key=api_key,
            api_keys=api_keys,
            cache=cache,
            enable_cache=enable_cache,
            prefer_cli=prefer_cli,
            **kwargs
        )

        # If caller didn't provide a custom cache, use per-provider cache
        # keyed/encrypted by the resolved API key (env or secrets manager).
        if cache_was_none:
            self.cache = get_llm_cache("anthropic", api_key=self.api_key)
    
    def get_tool_name(self) -> str:
        return "Claude (Anthropic)"
    
    def _detect_cli_path(self) -> Optional[str]:
        """Try to detect Claude CLI (experimental/unofficial)."""
        return detect_cli_tool(["claude", "claude-cli"])
    
    def _get_api_key_from_secrets(self) -> Optional[str]:
        """Get Anthropic API key from secrets manager."""
        return self.secrets_manager.get_credential("anthropic_api_key")
    
    def _create_sdk_client(self):
        """Create Anthropic SDK client using the default API key."""
        return self._build_anthropic_client(self.api_key)

    @staticmethod
    def _build_anthropic_client(api_key: Optional[str]):
        """Create a fresh Anthropic client for the given *api_key*."""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic SDK not installed. Install with: pip install anthropic"
            )
    
    def _chat_sdk(
        self,
        message: str,
        model: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute chat via SDK.

        Parameters
        ----------
        api_key:
            Optional per-request key override.  When provided a fresh
            Anthropic client is created for this key, enabling multi-user
            isolation without sharing a single client.
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
        
        # Build a per-request client when a key override is supplied;
        # otherwise fall back to the lazily-cached default client.
        effective_key = api_key or self.api_key
        if api_key and api_key != self.api_key:
            client = self._build_anthropic_client(effective_key)
        else:
            client = self._get_sdk_client()

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
    
    def chat(
        self,
        message: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.0,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to Claude (synchronous).
        
        Args:
            message: Chat message
            model: Model to use
            temperature: Sampling temperature
            user_id: Optional user identifier for per-user key pinning.
            **kwargs: Additional arguments
            
        Returns:
            Dict with response
        """
        return self._execute_with_fallback(
            sdk_func=self._chat_sdk,
            operation="chat",
            message=message,
            model=model,
            temperature=temperature,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs
        )

    async def achat(
        self,
        message: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.0,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async version of :meth:`chat` – safe for Trio / Hypercorn.

        The blocking Anthropic SDK call is offloaded to a worker thread via
        ``anyio.to_thread.run_sync`` so the event loop is never stalled.

        Parameters
        ----------
        message:
            Chat message.
        model:
            Model to use.
        temperature:
            Sampling temperature.
        user_id:
            Optional user identifier for per-user API key pinning.
        """
        return await self._aexecute_with_fallback(
            sdk_func=self._chat_sdk,
            operation="chat",
            message=message,
            model=model,
            temperature=temperature,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs,
        )
    
    def generate_code(
        self,
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate code from prompt (synchronous).
        
        Args:
            prompt: Code generation prompt
            model: Model to use
            user_id: Optional user identifier for per-user key pinning.
            **kwargs: Additional arguments
            
        Returns:
            Dict with generated code
        """
        return self.chat(prompt, model=model, temperature=0.0, user_id=user_id)

    async def agenerate_code(
        self,
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async version of :meth:`generate_code` – safe for Trio / Hypercorn.
        """
        return await self.achat(prompt, model=model, temperature=0.0, user_id=user_id, **kwargs)


# Global instance
_global_claude_code_cli: Optional[ClaudeCodeCLIIntegration] = None


def get_claude_code_cli_integration() -> ClaudeCodeCLIIntegration:
    """Get or create the global Claude Code CLI integration instance."""
    global _global_claude_code_cli
    
    if _global_claude_code_cli is None:
        _global_claude_code_cli = ClaudeCodeCLIIntegration()
    
    return _global_claude_code_cli
