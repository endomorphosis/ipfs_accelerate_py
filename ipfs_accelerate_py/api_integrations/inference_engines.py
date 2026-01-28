"""
Inference Engine Cache Integrations

Cache-enabled wrappers for inference engine APIs:
- HuggingFace TGI (Text Generation Inference)
- HuggingFace TEI (Text Embeddings Inference)
- OpenVINO Model Server (OVMS)
- OPEA (Open Platform for Enterprise AI)
"""

from typing import Any, Dict, List, Optional
import logging

try:
    from ...common.storage_wrapper import StorageWrapper
    DISTRIBUTED_STORAGE_AVAILABLE = True
except ImportError:
    try:
        from ..common.storage_wrapper import StorageWrapper
        DISTRIBUTED_STORAGE_AVAILABLE = True
    except ImportError:
        DISTRIBUTED_STORAGE_AVAILABLE = False
        StorageWrapper = None

if DISTRIBUTED_STORAGE_AVAILABLE:
    try:
        storage = StorageWrapper()
    except:
        storage = None
else:
    storage = None

logger = logging.getLogger(__name__)

try:
    from ..common.llm_cache import LLMAPICache, get_global_llm_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("Cache infrastructure not available")


class CachedHFTGIAPI:
    """
    Cache-enabled wrapper for HuggingFace TGI (Text Generation Inference).
    
    Caches text generation requests to reduce GPU compute.
    """
    
    def __init__(self, api_instance, cache: Optional[LLMAPICache] = None):
        self._api = api_instance
        self._cache = cache or get_global_llm_cache() if CACHE_AVAILABLE else None
    
    def __getattr__(self, name):
        return getattr(self._api, name)
    
    def generate(self, prompt: str, model: Optional[str] = None,
                temperature: float = 0.7, use_cache: bool = True, **kwargs) -> Any:
        """
        Generate text with caching.
        
        Args:
            prompt: Input prompt
            model: Model identifier
            temperature: Sampling temperature
            use_cache: Whether to use cache
            **kwargs: Additional arguments
            
        Returns:
            Generation response
        """
        if not use_cache or not self._cache:
            return self._api.generate(prompt=prompt, temperature=temperature, **kwargs)
        
        # Use model from kwargs or instance
        model_name = model or getattr(self._api, 'model_name', 'tgi-model')
        
        # Check cache
        cached = self._cache.get_completion(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'stream']}
        )
        
        if cached:
            logger.debug(f"HF TGI Cache HIT (model={model_name})")
            return cached
        
        # Call API
        logger.debug(f"HF TGI Cache MISS (model={model_name})")
        response = self._api.generate(prompt=prompt, temperature=temperature, **kwargs)
        
        # Cache response
        if not kwargs.get('stream', False):
            self._cache.cache_completion(
                prompt=prompt,
                response=response,
                model=model_name,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'stream']}
            )
        
        return response


class CachedHFTEIAPI:
    """
    Cache-enabled wrapper for HuggingFace TEI (Text Embeddings Inference).
    
    Caches embedding requests - embeddings are deterministic so cache for 24 hours.
    """
    
    def __init__(self, api_instance, cache: Optional[LLMAPICache] = None):
        self._api = api_instance
        self._cache = cache or get_global_llm_cache() if CACHE_AVAILABLE else None
    
    def __getattr__(self, name):
        return getattr(self._api, name)
    
    def embed(self, texts: List[str], model: Optional[str] = None,
             use_cache: bool = True, **kwargs) -> Any:
        """
        Generate embeddings with caching.
        
        Args:
            texts: Input texts
            model: Model identifier
            use_cache: Whether to use cache
            **kwargs: Additional arguments
            
        Returns:
            Embeddings response
        """
        if not use_cache or not self._cache:
            return self._api.embed(texts=texts, **kwargs)
        
        # Use model from kwargs or instance
        model_name = model or getattr(self._api, 'model_name', 'tei-model')
        
        # For embeddings, cache each text separately for better reuse
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            # Check cache
            cached = self._cache.get_completion(
                prompt=text,
                model=model_name,
                temperature=0.0,  # Embeddings are deterministic
                operation='embedding'
            )
            
            if cached:
                logger.debug(f"HF TEI Cache HIT for text {i}")
                results.append(cached)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append(None)  # Placeholder
        
        # Call API for uncached texts
        if uncached_texts:
            logger.debug(f"HF TEI Cache MISS for {len(uncached_texts)} texts")
            response = self._api.embed(texts=uncached_texts, **kwargs)
            
            # Cache individual embeddings
            embeddings = response if isinstance(response, list) else response.get('embeddings', [])
            for idx, embedding in zip(uncached_indices, embeddings):
                results[idx] = embedding
                self._cache.cache_completion(
                    prompt=texts[idx],
                    response=embedding,
                    model=model_name,
                    temperature=0.0,
                    operation='embedding',
                    ttl=86400  # 24 hours - embeddings are deterministic
                )
        
        return results


class CachedOVMSAPI:
    """
    Cache-enabled wrapper for OpenVINO Model Server (OVMS).
    
    Caches inference requests.
    """
    
    def __init__(self, api_instance, cache: Optional[LLMAPICache] = None):
        self._api = api_instance
        self._cache = cache or get_global_llm_cache() if CACHE_AVAILABLE else None
    
    def __getattr__(self, name):
        return getattr(self._api, name)
    
    def infer(self, inputs: Any, model: Optional[str] = None,
             use_cache: bool = True, **kwargs) -> Any:
        """
        Run inference with caching.
        
        Args:
            inputs: Input data
            model: Model name
            use_cache: Whether to use cache
            **kwargs: Additional arguments
            
        Returns:
            Inference response
        """
        if not use_cache or not self._cache:
            return self._api.infer(inputs=inputs, **kwargs)
        
        model_name = model or getattr(self._api, 'model_name', 'ovms-model')
        
        # Convert inputs to string for cache key
        import json
        try:
            input_str = json.dumps(inputs, sort_keys=True)
        except (TypeError, ValueError):
            input_str = str(inputs)
        
        # Check cache
        cached = self._cache.get_completion(
            prompt=input_str,
            model=model_name,
            temperature=0.0,  # Inference is deterministic
            **kwargs
        )
        
        if cached:
            logger.debug(f"OVMS Cache HIT (model={model_name})")
            return cached
        
        # Call API
        logger.debug(f"OVMS Cache MISS (model={model_name})")
        response = self._api.infer(inputs=inputs, **kwargs)
        
        # Cache response
        self._cache.cache_completion(
            prompt=input_str,
            response=response,
            model=model_name,
            temperature=0.0,
            **kwargs
        )
        
        return response


class CachedOPEAAPI:
    """
    Cache-enabled wrapper for OPEA (Open Platform for Enterprise AI).
    
    Caches workflow/pipeline results.
    """
    
    def __init__(self, api_instance, cache: Optional[LLMAPICache] = None):
        self._api = api_instance
        self._cache = cache or get_global_llm_cache() if CACHE_AVAILABLE else None
    
    def __getattr__(self, name):
        return getattr(self._api, name)
    
    def run_pipeline(self, inputs: Any, pipeline: str,
                    use_cache: bool = True, **kwargs) -> Any:
        """
        Run OPEA pipeline with caching.
        
        Args:
            inputs: Input data
            pipeline: Pipeline name
            use_cache: Whether to use cache
            **kwargs: Additional arguments
            
        Returns:
            Pipeline response
        """
        if not use_cache or not self._cache:
            return self._api.run_pipeline(inputs=inputs, pipeline=pipeline, **kwargs)
        
        # Convert inputs to string for cache key
        import json
        try:
            input_str = json.dumps(inputs, sort_keys=True)
        except (TypeError, ValueError):
            input_str = str(inputs)
        
        # Check cache
        cached = self._cache.get_completion(
            prompt=input_str,
            model=pipeline,
            temperature=0.0,
            **kwargs
        )
        
        if cached:
            logger.debug(f"OPEA Cache HIT (pipeline={pipeline})")
            return cached
        
        # Call API
        logger.debug(f"OPEA Cache MISS (pipeline={pipeline})")
        response = self._api.run_pipeline(inputs=inputs, pipeline=pipeline, **kwargs)
        
        # Cache response
        self._cache.cache_completion(
            prompt=input_str,
            response=response,
            model=pipeline,
            temperature=0.0,
            **kwargs
        )
        
        return response


# Factory functions

def get_cached_hf_tgi_api(**kwargs):
    """Get cache-enabled HF TGI API instance."""
    from ..api_backends.hf_tgi import hf_tgi
    api = hf_tgi(**kwargs)
    return CachedHFTGIAPI(api)


def get_cached_hf_tei_api(**kwargs):
    """Get cache-enabled HF TEI API instance."""
    from ..api_backends.hf_tei import hf_tei
    api = hf_tei(**kwargs)
    return CachedHFTEIAPI(api)


def get_cached_ovms_api(**kwargs):
    """Get cache-enabled OVMS API instance."""
    from ..api_backends.ovms import ovms
    api = ovms(**kwargs)
    return CachedOVMSAPI(api)


def get_cached_opea_api(**kwargs):
    """Get cache-enabled OPEA API instance."""
    from ..api_backends.opea import opea
    api = opea(**kwargs)
    return CachedOPEAAPI(api)


__all__ = [
    'CachedHFTGIAPI',
    'CachedHFTEIAPI',
    'CachedOVMSAPI',
    'CachedOPEAAPI',
    'get_cached_hf_tgi_api',
    'get_cached_hf_tei_api',
    'get_cached_ovms_api',
    'get_cached_opea_api',
]
