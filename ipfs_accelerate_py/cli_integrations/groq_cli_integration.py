"""
Groq CLI Integration with Common Cache

Wraps Groq API (via Groq Python SDK) to use the common cache infrastructure.
Note: Groq does not have an official CLI tool - this uses the Python SDK directly.
"""

import logging
from typing import Any, Dict, Optional

from ..common.llm_cache import LLMAPICache, get_global_llm_cache

logger = logging.getLogger(__name__)


class GroqCLIIntegration:
    """
    Groq integration with common cache infrastructure.
    
    Uses Python SDK (groq) with LLM cache for fast inference requests.
    Note: This is NOT a CLI wrapper - Groq has no official CLI.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        **kwargs
    ):
        """
        Initialize Groq integration.
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            enable_cache: Whether to enable caching
            cache: Custom cache instance (uses LLM cache if None)
            **kwargs: Additional arguments
        """
        self.api_key = api_key
        self.enable_cache = enable_cache
        
        if cache is None:
            cache = get_global_llm_cache()
        self.cache = cache
        
        # Lazy import groq SDK
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Groq client."""
        if self._client is None:
            try:
                import groq
                self._client = groq.Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "groq SDK not installed. Install with: pip install groq"
                )
        return self._client
    
    def get_tool_name(self) -> str:
        return "Groq (Python SDK)"
    
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
                logger.info("Cache hit for Groq chat")
                return {"response": cached, "cached": True}
        
        # Call API
        client = self._get_client()
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
            Dict with completion
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
        
        # Call API
        client = self._get_client()
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


# Global instance
_global_groq_cli: Optional[GroqCLIIntegration] = None


def get_groq_cli_integration() -> GroqCLIIntegration:
    """Get or create the global Groq CLI integration instance."""
    global _global_groq_cli
    
    if _global_groq_cli is None:
        _global_groq_cli = GroqCLIIntegration()
    
    return _global_groq_cli
