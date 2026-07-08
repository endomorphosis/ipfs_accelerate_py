# API Caching Opportunities

This document identifies all APIs in ipfs_accelerate_py that would benefit from the common cache infrastructure and provides recommendations for implementation.

## Overview

The new content-addressed cache infrastructure with CID-based lookups can significantly reduce API calls, costs, and latency across many services. This document analyzes each API and recommends caching strategies.

## Already Cached ✅

### GitHub API
- **Location**: `ipfs_accelerate_py/github_cli/cache.py`
- **Status**: Already implemented with P2P support
- **Recommendation**: Consider migrating to common base class for consistency
- **Current TTLs**:
  - `list_repos`: 300s (5 min)
  - `get_workflow_runs`: 60s
  - `list_runners`: 30s

### GitHub Copilot CLI
- **Location**: Integrated with GitHub CLI cache
- **Status**: Already implemented
- **Operations**: Command suggestions, explanations, git suggestions

### CodeQL Security Scans
- **Location**: `ipfs_accelerate_py/github_cli/codeql_cache.py`
- **Status**: Already implemented
- **TTL**: 86400s (24 hours) - scans are expensive
- **Benefit**: ~5 minutes saved per cached scan

## Newly Cacheable with Common Infrastructure 🆕

### 1. LLM APIs (High Priority) ⭐⭐⭐

All LLM APIs would benefit significantly from caching:

#### OpenAI API
- **Location**: `ipfs_accelerate_py/api_backends/openai_api.py`
- **Operations to cache**:
  - `chat.completions.create()` - TTL: 1800s (30 min for temp > 0, 3600s for temp=0)
  - `completions.create()` - TTL: 3600s (1 hour)
  - `embeddings.create()` - TTL: 86400s (24 hours - deterministic)
- **Cost savings**: $0.002-0.10 per 1K tokens
- **Implementation**: Use `LLMAPICache` from common infrastructure

#### Claude (Anthropic)
- **Location**: `ipfs_accelerate_py/api_backends/claude.py`
- **Operations**: Same as OpenAI
- **Cost savings**: Similar to OpenAI ($0.008-0.024 per 1K tokens)

#### Gemini (Google)
- **Location**: `ipfs_accelerate_py/api_backends/gemini.py`
- **Operations**: Text generation, embeddings
- **Cost savings**: Rate limit relief (60 requests/min)

#### Groq
- **Location**: `ipfs_accelerate_py/api_backends/groq.py`
- **Operations**: Fast inference
- **Benefit**: Rate limit relief, response caching

#### Ollama (Local)
- **Location**: `ipfs_accelerate_py/api_backends/ollama.py`
- **Operations**: Local model inference
- **Benefit**: Reduced compute for repeated queries

### 2. HuggingFace Hub API (High Priority) ⭐⭐⭐

- **Location**: Throughout codebase (model discovery, scraping)
- **Operations to cache**:
  - `model_info()` - TTL: 3600s (1 hour)
  - `list_repo_files()` - TTL: 1800s (30 min)
  - `dataset_info()` - TTL: 3600s (1 hour)
  - `list_models()` (search) - TTL: 600s (10 min)
  - `get_model_tags()` - TTL: 1800s (30 min)
- **Rate limit**: 1000 requests/hour
- **Implementation**: Use `HuggingFaceHubCache` from common infrastructure

### 3. Inference Engine APIs (Medium Priority) ⭐⭐

#### vLLM
- **Location**: `ipfs_accelerate_py/api_backends/vllm.py`
- **Operations**: Inference requests
- **TTL**: 1800s (30 min for temp > 0, 3600s for temp=0)
- **Benefit**: Reduced GPU compute for repeated queries

#### HuggingFace TGI (Text Generation Inference)
- **Location**: `ipfs_accelerate_py/api_backends/hf_tgi.py`
- **Operations**: Text generation
- **Recommended TTL**: 1800s (30 min)

#### HuggingFace TEI (Text Embeddings Inference)
- **Location**: `ipfs_accelerate_py/api_backends/hf_tei.py`
- **Operations**: Embedding generation
- **Recommended TTL**: 86400s (24 hours - deterministic)

#### OpenVINO Model Server (OVMS)
- **Location**: `ipfs_accelerate_py/api_backends/ovms.py`
- **Operations**: Model inference
- **Recommended TTL**: 1800s (30 min)

#### OPEA (Open Platform for Enterprise AI)
- **Location**: `ipfs_accelerate_py/api_backends/opea.py`
- **Operations**: Enterprise AI workflows
- **Recommended TTL**: 1800s (30 min)

### 4. Docker API (Medium Priority) ⭐⭐

- **Location**: Used throughout for containerized runners
- **Operations to cache**:
  - `images.inspect()` - TTL: 1800s (30 min)
  - `images.history()` - TTL: 3600s (1 hour)
  - `containers.inspect()` - TTL: 30s (short - state changes frequently)
  - `containers.list()` - TTL: 30s
  - `volumes.inspect()` - TTL: 300s (5 min)
  - Registry tag queries - TTL: 1800s (30 min)
- **Implementation**: Use `DockerAPICache` from common infrastructure

### 5. S3/Object Storage (Low Priority) ⭐

- **Location**: `ipfs_accelerate_py/api_backends/s3_kit.py`
- **Operations to cache**:
  - `list_objects()` - TTL: 300s (5 min)
  - `head_object()` (metadata) - TTL: 600s (10 min)
  - Object URLs - TTL: 3600s (1 hour)
- **Note**: Downloads themselves shouldn't be cached, only metadata

### 6. IPFS API (Low Priority) ⭐

- **Location**: Various IPFS operations
- **Operations to cache**:
  - `ipfs.get()` metadata - TTL: 3600s (1 hour)
  - DHT queries - TTL: 600s (10 min)
  - Pin status - TTL: 300s (5 min)
- **Note**: Content itself is already content-addressed

## Implementation Priority

### Phase 1: High-Value, Easy Wins (Week 1)
1. **LLM APIs** - Wrap OpenAI, Claude, Gemini, Groq with `LLMAPICache`
2. **HuggingFace Hub** - Add `HuggingFaceHubCache` to model discovery code

### Phase 2: Infrastructure (Week 2)
3. **Docker API** - Add `DockerAPICache` to container management
4. **Inference Engines** - Cache vLLM, TGI, TEI responses

### Phase 3: Nice-to-Have (Week 3+)
5. **S3/Storage** - Cache metadata queries
6. **IPFS** - Cache DHT/pin queries

## Implementation Pattern

For each API, follow this pattern:

```python
from ipfs_accelerate_py.common.llm_cache import get_global_llm_cache

# Get cache instance
cache = get_global_llm_cache()

# Before API call
cached_response = cache.get_completion(
    prompt=prompt,
    model=model,
    temperature=temperature
)

if cached_response:
    return cached_response

# Make API call
response = openai.ChatCompletion.create(...)

# Cache response
cache.cache_completion(
    prompt=prompt,
    response=response,
    model=model,
    temperature=temperature
)

return response
```

## Expected Impact

### Cost Savings (with 70% cache hit rate)

| API | Cost per 1K | Monthly Requests | Savings |
|-----|-------------|------------------|---------|
| OpenAI GPT-4 | $0.03-0.06 | 1M | $21-42K/month |
| Claude | $0.008-0.024 | 1M | $5.6-16.8K/month |
| HF Hub | Rate limit | 100K | Effective 3x capacity |
| Docker | Compute | Variable | Reduced API load |

### Performance Improvements

| Operation | Without Cache | With Cache | Speedup |
|-----------|---------------|------------|---------|
| LLM Completion | 1-5s | <0.01s | **100-500x** |
| HF Model Info | 0.2-1s | <0.01s | **20-100x** |
| Docker Inspect | 0.05-0.2s | <0.01s | **5-20x** |
| Embeddings | 0.5-2s | <0.01s | **50-200x** |

### Rate Limit Relief

- **OpenAI**: 3500 RPM → effectively 10,000+ RPM
- **HuggingFace Hub**: 1000 req/hour → effectively 3000+ req/hour
- **Groq**: 60 req/min → effectively 180+ req/min

## Monitoring

Track cache effectiveness using built-in statistics:

```python
from ipfs_accelerate_py.common.base_cache import get_all_caches

for cache_name, cache in get_all_caches().items():
    stats = cache.get_stats()
    print(f"{cache_name}:")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  API calls saved: {stats['api_calls_saved']}")
    print(f"  CID index size: {stats['cid_index']['total_cids']}")
```

## Best Practices

### 1. Cache Key Design
- Include all relevant parameters in cache key
- Use temperature=0 for deterministic responses (longer TTL)
- Don't cache user-specific or time-sensitive data

### 2. TTL Selection
- **Deterministic operations** (embeddings, temp=0): Long TTL (hours/days)
- **Semi-stable data** (model metadata): Medium TTL (minutes/hour)
- **Dynamic data** (container status): Short TTL (seconds/minutes)

### 3. Cache Invalidation
- Invalidate on mutations (e.g., after updating a model)
- Use pattern-based invalidation for related entries
- Monitor stale data via content validation

### 4. P2P Distribution (Future)
- Enable P2P for distributed teams
- Share cache across multiple runners
- Reduce combined API costs

## Migration Guide

### Migrating Existing Code

Before:
```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
)
```

After:
```python
from ipfs_accelerate_py.common.llm_cache import get_global_llm_cache

cache = get_global_llm_cache()

# Check cache
cached = cache.get_chat_completion(messages=messages, model="gpt-4")
if cached:
    response = cached
else:
    # Make API call
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    # Cache result
    cache.cache_chat_completion(messages=messages, response=response, model="gpt-4")
```

## Conclusion

The content-addressed cache infrastructure with CID-based lookups provides:
- ✅ **Fast O(1) lookups** - Hash the query, get the CID, lookup the cache
- ✅ **Significant cost savings** - 70%+ reduction in API calls
- ✅ **Performance improvements** - 100-500x faster for cached responses
- ✅ **Rate limit relief** - Effectively 3-5x capacity increase
- ✅ **Unified pattern** - Same caching approach across all APIs
- ✅ **P2P ready** - Infrastructure supports distributed caching

Implementing caching for LLM APIs and HuggingFace Hub alone would provide immediate, substantial benefits.
