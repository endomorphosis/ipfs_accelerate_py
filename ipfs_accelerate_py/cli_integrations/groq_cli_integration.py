"""
Groq CLI Integration with Common Cache

Wraps Groq API (via Groq Python SDK) to use the common cache infrastructure.
Supports dual-mode operation with CLI fallback and secrets manager integration.
Note: Groq does not have an official CLI tool - this primarily uses the Python SDK.

Multi-user / parallel execution
--------------------------------
Pass ``api_keys`` to distribute load across multiple Groq keys::

    groq = GroqCLIIntegration(api_keys=["gsk_1", "gsk_2"])
    result = await groq.achat("Hello", user_id="carol")

Async support (Trio / Hypercorn)
---------------------------------
``achat()`` and ``acomplete()`` offload the blocking Groq SDK call to a
worker thread via ``anyio.to_thread.run_sync``.
"""

import logging
from typing import Any, Dict, List, Optional

from .dual_mode_wrapper import DualModeWrapper, detect_cli_tool
from ..common.llm_cache import LLMAPICache, get_global_llm_cache, get_llm_cache

logger = logging.getLogger(__name__)


class GroqCLIIntegration(DualModeWrapper):
    """
    Groq integration with common cache infrastructure.
    
    Supports dual-mode operation:
    - SDK mode: Uses Groq Python SDK (primary mode)
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
        Initialize Groq integration.
        
        Args:
            api_key: Groq API key (from secrets manager if None)
            api_keys: List of Groq API keys for multi-user round-robin pool.
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

        if cache_was_none:
            self.cache = get_llm_cache("groq", api_key=self.api_key)
    
    def get_tool_name(self) -> str:
        return "Groq"
    
    def _detect_cli_path(self) -> Optional[str]:
        """Try to detect Groq CLI (experimental/unofficial)."""
        return detect_cli_tool(["groq", "groq-cli"])
    
    def _get_api_key_from_secrets(self) -> Optional[str]:
        """Get Groq API key from secrets manager."""
        return self.secrets_manager.get_credential("groq_api_key")
    
    def _create_sdk_client(self):
        """Create Groq SDK client using the default API key."""
        return self._build_groq_client(self.api_key)

    @staticmethod
    def _build_groq_client(api_key: Optional[str]):
        """Create a fresh Groq client for the given *api_key*."""
        try:
            import groq
            return groq.Groq(api_key=api_key)
        except ImportError:
            raise ImportError(
                "groq SDK not installed. Install with: pip install groq"
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
            Optional per-request key override for multi-user scenarios.
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
                logger.info("Cache hit for Groq chat")
                return {"response": cached, "cached": True}
        
        effective_key = api_key or self.api_key
        if api_key and api_key != self.api_key:
            client = self._build_groq_client(effective_key)
        else:
            client = self._get_sdk_client()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        result = response.choices[0].message.content
        
        # Cache response
        if self.enable_cache:
            self.cache.cache_chat_completion(
                messages=messages,
                response=result,
                model=model,
                temperature=temperature
            )
        
        return {"response": result, "cached": False}
    
    def _complete_sdk(
        self,
        prompt: str,
        model: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute completion via SDK.

        Parameters
        ----------
        api_key:
            Optional per-request key override for multi-user scenarios.
        """
        # Check cache first
        if self.enable_cache:
            cached = self.cache.get_completion(
                prompt=prompt,
                model=model,
                temperature=temperature
            )
            if cached:
                logger.info("Cache hit for Groq completion")
                return {"response": cached, "cached": True}
        
        effective_key = api_key or self.api_key
        if api_key and api_key != self.api_key:
            client = self._build_groq_client(effective_key)
        else:
            client = self._get_sdk_client()

        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature
        )
        
        result = response.choices[0].text
        
        # Cache response
        if self.enable_cache:
            self.cache.cache_completion(
                prompt=prompt,
                response=result,
                model=model,
                temperature=temperature
            )
        
        return {"response": result, "cached": False}
    
    def chat(
        self,
        message: str,
        model: str = "llama3-70b-8192",
        temperature: float = 0.0,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to Groq (synchronous).
        
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
        model: str = "llama3-70b-8192",
        temperature: float = 0.0,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async version of :meth:`chat` – safe for Trio / Hypercorn.

        The blocking Groq SDK call is offloaded to a worker thread via
        ``anyio.to_thread.run_sync``.
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
    
    def complete(
        self,
        prompt: str,
        model: str = "llama3-70b-8192",
        temperature: float = 0.0,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete a prompt with Groq (synchronous).
        
        Args:
            prompt: Prompt to complete
            model: Model to use
            temperature: Sampling temperature
            user_id: Optional user identifier for per-user key pinning.
            **kwargs: Additional arguments
            
        Returns:
            Dict with completion
        """
        return self._execute_with_fallback(
            sdk_func=self._complete_sdk,
            operation="complete",
            prompt=prompt,
            model=model,
            temperature=temperature,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs
        )

    async def acomplete(
        self,
        prompt: str,
        model: str = "llama3-70b-8192",
        temperature: float = 0.0,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async version of :meth:`complete` – safe for Trio / Hypercorn.
        """
        return await self._aexecute_with_fallback(
            sdk_func=self._complete_sdk,
            operation="complete",
            prompt=prompt,
            model=model,
            temperature=temperature,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs,
        )


# Global instance
_global_groq_cli: Optional[GroqCLIIntegration] = None


def get_groq_cli_integration() -> GroqCLIIntegration:
    """Get or create the global Groq CLI integration instance."""
    global _global_groq_cli
    
    if _global_groq_cli is None:
        _global_groq_cli = GroqCLIIntegration()
    
    return _global_groq_cli
