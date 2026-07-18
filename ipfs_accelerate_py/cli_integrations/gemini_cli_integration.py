"""
Gemini CLI Integration with Common Cache

Wraps Google Gemini API (via google-generativeai Python SDK) to use the common cache infrastructure.
Supports dual-mode operation with CLI fallback and secrets manager integration.
Note: Gemini does not have an official CLI tool - this primarily uses the Python SDK.

Multi-user / parallel execution
--------------------------------
Pass ``api_keys`` to distribute load across multiple Google API keys::

    gemini = GeminiCLIIntegration(api_keys=["AIza-1", "AIza-2"])
    result = await gemini.agenerate_text("Hello", user_id="bob")

Async support (Trio / Hypercorn)
---------------------------------
``agenerate_text()`` and ``achat()`` offload the blocking google-generativeai
SDK call to a thread via ``anyio.to_thread.run_sync``.

Note on multi-key Gemini usage
--------------------------------
``google.generativeai.configure()`` sets a **global** API key.  To avoid
race conditions when using multiple keys concurrently, each call that
specifies a non-default key acquires ``_GEMINI_CONFIG_LOCK`` before calling
``configure()``, then creates the model object, and releases the lock before
the actual network call.  This minimises contention while remaining correct.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from .dual_mode_wrapper import DualModeWrapper, detect_cli_tool
from ..common.llm_cache import LLMAPICache, get_global_llm_cache, get_llm_cache

logger = logging.getLogger(__name__)

# Lock used when configuring the global google.generativeai API key
_GEMINI_CONFIG_LOCK = threading.Lock()


class GeminiCLIIntegration(DualModeWrapper):
    """
    Gemini integration with common cache infrastructure.
    
    Supports dual-mode operation:
    - SDK mode: Uses google-generativeai Python SDK (primary mode)
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
        Initialize Gemini integration.
        
        Args:
            api_key: Google API key (from secrets manager if None)
            api_keys: List of Google API keys for multi-user round-robin pool.
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
            # Prefer GEMINI_API_KEY/GOOGLE_API_KEY, else secrets manager resolved key.
            self.cache = get_llm_cache("gemini", api_key=self.api_key)
        
        # Lazy import and configure google-generativeai
        self._configured = False
    
    def get_tool_name(self) -> str:
        return "Gemini (Google)"
    
    def _detect_cli_path(self) -> Optional[str]:
        """Try to detect Gemini CLI (experimental/unofficial)."""
        return detect_cli_tool(["gemini", "gemini-cli"])
    
    def _get_api_key_from_secrets(self) -> Optional[str]:
        """Get Google API key from secrets manager."""
        return self.secrets_manager.get_credential("google_api_key")
    
    def _configure(self, api_key: Optional[str] = None):
        """Configure google-generativeai with the given (or default) key.

        Acquires ``_GEMINI_CONFIG_LOCK`` only when a non-default key is
        requested, to serialise the global ``configure()`` call while keeping
        the network request outside the lock.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai SDK not installed. Install with: pip install google-generativeai"
            )

        effective_key = api_key or self.api_key
        if effective_key and effective_key != self.api_key:
            # Different key – must lock to avoid clobbering the global state
            with _GEMINI_CONFIG_LOCK:
                genai.configure(api_key=effective_key)
        elif not self._configured:
            genai.configure(api_key=effective_key)
            self._configured = True

        self._genai = genai
        return genai
    
    def _create_sdk_client(self):
        """Configure and return genai module."""
        self._configure()
        return self._genai
    
    def _generate_text_sdk(
        self,
        prompt: str,
        model: str,
        temperature: float,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute text generation via SDK.

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
                logger.info("Cache hit for Gemini generation")
                return {"response": cached, "cached": True}
        
        # Configure with the effective key then release the global lock
        genai = self._configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        
        result = response.text
        
        # Cache response
        if self.enable_cache:
            self.cache.cache_completion(
                prompt=prompt,
                response=result,
                model=model,
                temperature=temperature
            )
        
        return {"response": result, "cached": False}
    
    def generate_text(
        self,
        prompt: str,
        model: str = "gemini-pro",
        temperature: float = 0.0,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from prompt (synchronous).
        
        Args:
            prompt: Text generation prompt
            model: Model to use
            temperature: Sampling temperature
            user_id: Optional user identifier for per-user key pinning.
            **kwargs: Additional arguments
            
        Returns:
            Dict with generated text
        """
        return self._execute_with_fallback(
            sdk_func=self._generate_text_sdk,
            operation="generate_text",
            prompt=prompt,
            model=model,
            temperature=temperature,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs
        )

    async def agenerate_text(
        self,
        prompt: str,
        model: str = "gemini-pro",
        temperature: float = 0.0,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async version of :meth:`generate_text` – safe for Trio / Hypercorn.

        The blocking google-generativeai SDK call is offloaded to a worker
        thread via ``anyio.to_thread.run_sync``.
        """
        return await self._aexecute_with_fallback(
            sdk_func=self._generate_text_sdk,
            operation="generate_text",
            prompt=prompt,
            model=model,
            temperature=temperature,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs,
        )
    
    def chat(
        self,
        message: str,
        model: str = "gemini-pro",
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat message to Gemini (synchronous).
        
        Args:
            message: Chat message
            model: Model to use
            user_id: Optional user identifier for per-user key pinning.
            **kwargs: Additional arguments
            
        Returns:
            Dict with response
        """
        return self.generate_text(message, model=model, temperature=0.0, user_id=user_id)

    async def achat(
        self,
        message: str,
        model: str = "gemini-pro",
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async version of :meth:`chat` – safe for Trio / Hypercorn.
        """
        return await self.agenerate_text(message, model=model, temperature=0.0, user_id=user_id, **kwargs)


# Global instance
_global_gemini_cli: Optional[GeminiCLIIntegration] = None


def get_gemini_cli_integration() -> GeminiCLIIntegration:
    """Get or create the global Gemini CLI integration instance."""
    global _global_gemini_cli
    
    if _global_gemini_cli is None:
        _global_gemini_cli = GeminiCLIIntegration()
    
    return _global_gemini_cli
