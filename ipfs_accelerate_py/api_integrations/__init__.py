"""
API Backend Cache Integrations

This module provides cache-enabled wrappers for all API backends,
transparently adding CID-based caching without modifying the original code.

Supports:
- LLM APIs: OpenAI, Claude, Gemini, Groq, Ollama
- Inference Engines: vLLM, HF TGI, HF TEI, OVMS, OPEA
- HuggingFace Hub API
- Docker API
- S3/Storage API
- IPFS API

Usage:
    from ipfs_accelerate_py.api_integrations import get_cached_openai_api
    
    # Get cache-enabled OpenAI API
    api = get_cached_openai_api(api_key="your-key")
    
    # Use normally - caching happens automatically
    response = api.chat(messages=[{"role": "user", "content": "Hello"}])
"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import cache infrastructure
try:
    from ..common.llm_cache import LLMAPICache, get_global_llm_cache
    from ..common.hf_hub_cache import HuggingFaceHubCache, get_global_hf_hub_cache
    from ..common.docker_cache import DockerAPICache, get_global_docker_cache
    from ..common.base_cache import BaseAPICache, register_cache
    CACHE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cache infrastructure not available: {e}")
    CACHE_AVAILABLE = False


class CachedLLMAPI:
    """
    Generic cache-enabled wrapper for LLM APIs.
    
    Works with OpenAI, Claude, Gemini, Groq, and similar APIs.
    """
    
    def __init__(self, api_instance, api_name: str, cache: Optional[LLMAPICache] = None):
        """
        Initialize cached LLM API.
        
        Args:
            api_instance: Instance of API class
            api_name: Name of the API (for logging)
            cache: Optional cache instance (uses global if None)
        """
        self._api = api_instance
        self._api_name = api_name
        self._cache = cache or get_global_llm_cache() if CACHE_AVAILABLE else None
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped API."""
        return getattr(self._api, name)
    
    def _chat_cached(self, messages: List[Dict[str, str]], model: str,
                    temperature: float, use_cache: bool, method_name: str, **kwargs) -> Any:
        """Generic cached chat method."""
        if not use_cache or not self._cache:
            return getattr(self._api, method_name)(messages=messages, model=model, temperature=temperature, **kwargs)
        
        # Check cache
        cached = self._cache.get_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'stream']}
        )
        
        if cached:
            logger.debug(f"{self._api_name} Cache HIT for chat (model={model})")
            return cached
        
        # Call API
        logger.debug(f"{self._api_name} Cache MISS for chat (model={model})")
        response = getattr(self._api, method_name)(messages=messages, model=model, temperature=temperature, **kwargs)
        
        # Cache response (don't cache streams)
        if not kwargs.get('stream', False):
            self._cache.cache_chat_completion(
                messages=messages,
                response=response,
                model=model,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'stream']}
            )
        
        return response
    
    def _completion_cached(self, prompt: str, model: str,
                          temperature: float, use_cache: bool, method_name: str, **kwargs) -> Any:
        """Generic cached completion method."""
        if not use_cache or not self._cache:
            return getattr(self._api, method_name)(prompt=prompt, model=model, temperature=temperature, **kwargs)
        
        # Check cache
        cached = self._cache.get_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'stream']}
        )
        
        if cached:
            logger.debug(f"{self._api_name} Cache HIT for completion (model={model})")
            return cached
        
        # Call API
        logger.debug(f"{self._api_name} Cache MISS for completion (model={model})")
        response = getattr(self._api, method_name)(prompt=prompt, model=model, temperature=temperature, **kwargs)
        
        # Cache response (don't cache streams)
        if not kwargs.get('stream', False):
            self._cache.cache_completion(
                prompt=prompt,
                response=response,
                model=model,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'stream']}
            )
        
        return response


class CachedOpenAIAPI(CachedLLMAPI):
    """Cache-enabled wrapper for OpenAI API."""
    
    def __init__(self, api_instance, cache: Optional[LLMAPICache] = None):
        super().__init__(api_instance, "OpenAI", cache)
    
    def chat(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo",
             temperature: float = 0.7, use_cache: bool = True, **kwargs) -> Any:
        """Chat completion with caching."""
        return self._chat_cached(messages, model, temperature, use_cache, 'chat', **kwargs)
    
    def complete(self, prompt: str, model: str = "gpt-3.5-turbo",
                temperature: float = 0.7, use_cache: bool = True, **kwargs) -> Any:
        """Completion with caching."""
        return self._completion_cached(prompt, model, temperature, use_cache, 'complete', **kwargs)


class CachedClaudeAPI(CachedLLMAPI):
    """Cache-enabled wrapper for Claude API."""
    
    def __init__(self, api_instance, cache: Optional[LLMAPICache] = None):
        super().__init__(api_instance, "Claude", cache)
    
    def chat(self, messages: List[Dict[str, str]], model: str = "claude-3-haiku-20240307",
             temperature: float = 0.7, use_cache: bool = True, **kwargs) -> Any:
        """Chat completion with caching."""
        return self._chat_cached(messages, model, temperature, use_cache, 'chat', **kwargs)


class CachedGeminiAPI(CachedLLMAPI):
    """Cache-enabled wrapper for Gemini API."""
    
    def __init__(self, api_instance, cache: Optional[LLMAPICache] = None):
        super().__init__(api_instance, "Gemini", cache)
    
    def generate_text(self, prompt: str, model: str = "gemini-pro",
                     temperature: float = 0.7, use_cache: bool = True, **kwargs) -> Any:
        """Text generation with caching."""
        return self._completion_cached(prompt, model, temperature, use_cache, 'generate_text', **kwargs)


class CachedGroqAPI(CachedLLMAPI):
    """Cache-enabled wrapper for Groq API."""
    
    def __init__(self, api_instance, cache: Optional[LLMAPICache] = None):
        super().__init__(api_instance, "Groq", cache)
    
    def chat(self, messages: List[Dict[str, str]], model: str = "llama3-70b-8192",
             temperature: float = 0.7, use_cache: bool = True, **kwargs) -> Any:
        """Chat completion with caching."""
        return self._chat_cached(messages, model, temperature, use_cache, 'chat', **kwargs)


class CachedOllamaAPI(CachedLLMAPI):
    """Cache-enabled wrapper for Ollama API."""
    
    def __init__(self, api_instance, cache: Optional[LLMAPICache] = None):
        super().__init__(api_instance, "Ollama", cache)
    
    def generate(self, prompt: str, model: str = "llama2",
                temperature: float = 0.7, use_cache: bool = True, **kwargs) -> Any:
        """Generation with caching."""
        return self._completion_cached(prompt, model, temperature, use_cache, 'generate', **kwargs)


class CachedVLLMAPI(CachedLLMAPI):
    """Cache-enabled wrapper for vLLM API."""
    
    def __init__(self, api_instance, cache: Optional[LLMAPICache] = None):
        super().__init__(api_instance, "vLLM", cache)
    
    def generate(self, prompt: str, model: str,
                temperature: float = 0.7, use_cache: bool = True, **kwargs) -> Any:
        """Generation with caching."""
        return self._completion_cached(prompt, model, temperature, use_cache, 'generate', **kwargs)


# Factory functions for easy access

def get_cached_openai_api(api_key: Optional[str] = None, **kwargs):
    """
    Get cache-enabled OpenAI API instance.
    
    Args:
        api_key: OpenAI API key
        **kwargs: Additional arguments
        
    Returns:
        Cached OpenAI API instance
    """
    from ..api_backends.openai_api import openai_api
    api = openai_api(api_key=api_key, **kwargs)
    return CachedOpenAIAPI(api)


def get_cached_claude_api(api_key: Optional[str] = None, **kwargs):
    """
    Get cache-enabled Claude API instance.
    
    Args:
        api_key: Claude API key
        **kwargs: Additional arguments
        
    Returns:
        Cached Claude API instance
    """
    from ..api_backends.claude import claude
    metadata = kwargs.get('metadata', {})
    if api_key:
        metadata['api_key'] = api_key
    kwargs['metadata'] = metadata
    api = claude(**kwargs)
    return CachedClaudeAPI(api)


def get_cached_gemini_api(**kwargs):
    """
    Get cache-enabled Gemini API instance.
    
    Args:
        **kwargs: Additional arguments
        
    Returns:
        Cached Gemini API instance
    """
    from ..api_backends.gemini import gemini
    api = gemini(**kwargs)
    return CachedGeminiAPI(api)


def get_cached_groq_api(api_key: Optional[str] = None, **kwargs):
    """
    Get cache-enabled Groq API instance.
    
    Args:
        api_key: Groq API key
        **kwargs: Additional arguments
        
    Returns:
        Cached Groq API instance
    """
    from ..api_backends.groq import groq
    metadata = kwargs.get('metadata', {})
    if api_key:
        metadata['api_key'] = api_key
    kwargs['metadata'] = metadata
    api = groq(**kwargs)
    return CachedGroqAPI(api)


def get_cached_ollama_api(**kwargs):
    """
    Get cache-enabled Ollama API instance.
    
    Args:
        **kwargs: Additional arguments
        
    Returns:
        Cached Ollama API instance
    """
    from ..api_backends.ollama import ollama
    api = ollama(**kwargs)
    return CachedOllamaAPI(api)


def get_cached_vllm_api(**kwargs):
    """
    Get cache-enabled vLLM API instance.
    
    Args:
        **kwargs: Additional arguments
        
    Returns:
        Cached vLLM API instance
    """
    from ..api_backends.vllm import vllm
    api = vllm(**kwargs)
    return CachedVLLMAPI(api)


# Import additional modules
from .inference_engines import (
    get_cached_hf_tgi_api,
    get_cached_hf_tei_api,
    get_cached_ovms_api,
    get_cached_opea_api,
)

from .storage import (
    get_cached_s3_api,
    get_cached_ipfs_api,
)


# Export all public APIs
__all__ = [
    # Wrapper classes
    'CachedLLMAPI',
    'CachedOpenAIAPI',
    'CachedClaudeAPI',
    'CachedGeminiAPI',
    'CachedGroqAPI',
    'CachedOllamaAPI',
    'CachedVLLMAPI',
    
    # Factory functions - LLM APIs
    'get_cached_openai_api',
    'get_cached_claude_api',
    'get_cached_gemini_api',
    'get_cached_groq_api',
    'get_cached_ollama_api',
    'get_cached_vllm_api',
    
    # Factory functions - Inference Engines
    'get_cached_hf_tgi_api',
    'get_cached_hf_tei_api',
    'get_cached_ovms_api',
    'get_cached_opea_api',
    
    # Factory functions - Storage
    'get_cached_s3_api',
    'get_cached_ipfs_api',
]

