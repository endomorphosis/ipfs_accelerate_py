"""
Gemini CLI Integration with Common Cache

Wraps Google Gemini API (via google-generativeai Python SDK) to use the common cache infrastructure.
Supports dual-mode operation with CLI fallback and secrets manager integration.
Note: Gemini does not have an official CLI tool - this primarily uses the Python SDK.
"""

import logging
from typing import Any, Dict, Optional

from .dual_mode_wrapper import DualModeWrapper, detect_cli_tool
from ..common.llm_cache import LLMAPICache, get_global_llm_cache, get_llm_cache

logger = logging.getLogger(__name__)


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
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        prefer_cli: bool = False,  # Default to SDK since no official CLI
        **kwargs
    ):
        """
        Initialize Gemini integration.
        
        Args:
            api_key: Google API key (from secrets manager if None)
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
    
    def _configure(self):
        """Lazy configuration of Google Generative AI."""
        if not self._configured:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._configured = True
                self._genai = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai SDK not installed. Install with: pip install google-generativeai"
                )
    
    def _create_sdk_client(self):
        """Configure and return genai module."""
        self._configure()
        return self._genai
    
    def _generate_text_sdk(
        self,
        prompt: str,
        model: str,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute text generation via SDK."""
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
        
        # Call API
        self._configure()
        model_obj = self._genai.GenerativeModel(model)
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
            Dict with generated text
        """
        # Use dual-mode execution (SDK primary, CLI fallback)
        return self._execute_with_fallback(
            sdk_func=self._generate_text_sdk,
            operation="generate_text",
            prompt=prompt,
            model=model,
            temperature=temperature,
            **kwargs
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
            Dict with response
        """
        return self.generate_text(message, model=model, temperature=0.0)


# Global instance
_global_gemini_cli: Optional[GeminiCLIIntegration] = None


def get_gemini_cli_integration() -> GeminiCLIIntegration:
    """Get or create the global Gemini CLI integration instance."""
    global _global_gemini_cli
    
    if _global_gemini_cli is None:
        _global_gemini_cli = GeminiCLIIntegration()
    
    return _global_gemini_cli
