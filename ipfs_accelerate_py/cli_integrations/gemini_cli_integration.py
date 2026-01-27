"""
Gemini CLI Integration with Common Cache

Wraps Google Gemini API (via google-generativeai Python SDK) to use the common cache infrastructure.
Note: Gemini does not have an official CLI tool - this uses the Python SDK directly.
"""

import logging
from typing import Any, Dict, Optional

from ..common.llm_cache import LLMAPICache, get_global_llm_cache

logger = logging.getLogger(__name__)


class GeminiCLIIntegration:
    """
    Gemini integration with common cache infrastructure.
    
    Uses Python SDK (google-generativeai) with LLM cache for text generation.
    Note: This is NOT a CLI wrapper - Gemini has no official CLI.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        **kwargs
    ):
        """
        Initialize Gemini integration.
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            enable_cache: Whether to enable caching
            cache: Custom cache instance (uses LLM cache if None)
            **kwargs: Additional arguments
        """
        self.api_key = api_key
        self.enable_cache = enable_cache
        
        if cache is None:
            cache = get_global_llm_cache()
        self.cache = cache
        
        # Lazy import and configure google-generativeai
        self._configured = False
    
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
    
    def get_tool_name(self) -> str:
        return "Gemini (Google SDK)"
    
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
            generation_config=self._genai.types.GenerationConfig(
                temperature=temperature
            )
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
