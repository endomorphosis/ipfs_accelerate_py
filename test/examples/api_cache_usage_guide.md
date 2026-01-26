# API Semantic Cache Usage Guide

This guide provides detailed instructions for integrating and using the semantic caching system with various LLM API providers in the IPFS Accelerate Python framework.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Supported APIs](#supported-apis)
4. [Basic Usage](#basic-usage)
5. [Advanced Configuration](#advanced-configuration)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring and Metrics](#monitoring-and-metrics)
8. [Troubleshooting](#troubleshooting)
9. [API-Specific Features](#api-specific-features)
10. [Examples](#examples)

## Introduction

Semantic caching improves the efficiency, speed, and cost-effectiveness of LLM API usage by:

- Caching responses based on semantic similarity rather than exact string matching
- Automatically detecting when queries have the same meaning despite different wording
- Providing instant responses for semantically equivalent queries
- Reducing API costs by minimizing redundant API calls
- Improving application performance with lower latency

## Getting Started

### Installation

The semantic caching system is included in the IPFS Accelerate Python framework. To install:

```bash
# Clone the repository
git clone https://github.com/ipfs/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

Here's a minimal example to get started with semantic caching for OpenAI:

```python
import anyio
from openai import AsyncClient
from examples.openai_semantic_cache import SemanticCacheOpenAIClient

async def main():
    # Create a standard OpenAI client
    client = AsyncClient()
    
    # Wrap it with the semantic cache
    cached_client = SemanticCacheOpenAIClient(
        base_client=client,
        similarity_threshold=0.85,
        max_cache_size=1000,
        ttl=3600  # Cache entries expire after 1 hour
    )
    
    # First query (cache miss)
    response1 = await cached_client.create_chat_completion(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        model="gpt-3.5-turbo",
        temperature=0.0  # Use 0 temperature for deterministic responses
    )
    
    # Semantically similar query (likely cache hit)
    response2 = await cached_client.create_chat_completion(
        messages=[{"role": "user", "content": "Tell me the capital city of France."}],
        model="gpt-3.5-turbo",
        temperature=0.0
    )
    
    # Different query (cache miss)
    response3 = await cached_client.create_chat_completion(
        messages=[{"role": "user", "content": "What is the largest city in France?"}],
        model="gpt-3.5-turbo",
        temperature=0.0
    )
    
    # Print cache statistics
    print(cached_client.get_cache_stats())

anyio.run(main)
```

## Supported APIs

The framework provides semantic caching for these LLM API providers:

| API Provider | File Location | Models Supported |
|--------------|---------------|------------------|
| OpenAI | `examples/openai_semantic_cache.py` | All OpenAI models |
| Claude (Anthropic) | `examples/claude_cache/claude_semantic_cache.py` | All Claude models |
| Gemini (Google) | `examples/semantic_cache.py` | All Gemini models |
| Groq | `examples/groq_semantic_cache.py` | All Groq models |

## Basic Usage

### Creating a Cached Client

To add semantic caching to any supported API client:

1. Import the base client and semantic cache wrapper
2. Create the base client with your API key
3. Wrap the base client with the semantic cache wrapper
4. Configure cache parameters (threshold, size, TTL, etc.)
5. Use the cached client as you would use the normal client

Example with Claude API:

```python
import anthropic
from examples.claude_cache.claude_semantic_cache import SemanticCacheClaudeClient

# Create base client
base_client = anthropic.Anthropic(api_key="your_api_key")

# Create cached client
cached_client = SemanticCacheClaudeClient(
    base_client=base_client,
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=3600
)

# Use the client
async def example():
    response = await cached_client.chat(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        model="claude-3-opus-20240229",
        temperature=0.0
    )
    print(response)
```

### Disabling Cache for Specific Requests

For non-deterministic outputs or where fresh data is needed:

```python
# Setting temperature > 0 automatically bypasses cache
response = await cached_client.create_chat_completion(
    messages=[{"role": "user", "content": "Write a creative story."}],
    temperature=0.7  # Non-zero temperature bypasses cache
)

# Explicitly disable cache for a specific client
cached_client.set_cache_enabled(False)

# Re-enable later
cached_client.set_cache_enabled(True)
```

### Cache Management

Control the cache with these methods:

```python
# Clear the cache
cached_client.clear_cache()

# Get cache statistics
stats = cached_client.get_cache_stats()
print(f"Cache hit rate: {stats['cache_hits'] / stats['total_requests']:.1%}")
print(f"Active cache entries: {stats['active_entries']}")

# Estimate cost savings
token_savings = stats['token_savings']
cost_savings = token_savings * 0.00002  # Approximate cost per token
print(f"Estimated cost savings: ${cost_savings:.4f}")
```

## Advanced Configuration

### Tuning Similarity Threshold

The similarity threshold controls how similar queries must be to trigger a cache hit:

| Threshold | Cache Behavior | Use Case |
|-----------|----------------|----------|
| 0.90 - 0.95 | Very strict matching | High-precision requirements |
| 0.85 - 0.90 | Balanced matching | General purpose (recommended) |
| 0.80 - 0.85 | More lenient matching | Higher hit rate priority |
| 0.75 - 0.80 | Loose matching | Testing/development |

Example:

```python
# Strict matching for critical applications
cached_client = SemanticCacheOpenAIClient(
    base_client=client,
    similarity_threshold=0.92,
    max_cache_size=1000,
    ttl=3600
)

# Lenient matching for higher hit rate
cached_client = SemanticCacheOpenAIClient(
    base_client=client,
    similarity_threshold=0.82,
    max_cache_size=2000,
    ttl=7200
)
```

### Embedding Model Selection

You can provide a custom embedding model for better semantic similarity detection:

```python
from sentence_transformers import SentenceTransformer

# Load a sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create client with custom embeddings
cached_client = SemanticCacheOpenAIClient(
    base_client=client,
    embedding_model=embedding_model,
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=3600
)
```

### Time-to-Live Configuration

Configure cache expiration based on your use case:

```python
# Short-lived cache for rapidly changing information (5 minutes)
cached_client = SemanticCacheOpenAIClient(
    base_client=client,
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=300
)

# Long-lived cache for stable information (24 hours)
cached_client = SemanticCacheOpenAIClient(
    base_client=client, 
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=86400
)
```

## Performance Optimization

### Memory Usage Optimization

Control memory usage with these approaches:

```python
# Smaller cache for memory-constrained environments
cached_client = SemanticCacheOpenAIClient(
    base_client=client,
    similarity_threshold=0.85,
    max_cache_size=500,  # Reduced cache size
    ttl=3600
)

# Shorter TTL for more frequent cleanup
cached_client = SemanticCacheOpenAIClient(
    base_client=client,
    similarity_threshold=0.85,
    max_cache_size=1000,
    ttl=1800  # 30 minutes
)
```

### Concurrent Usage

All cache implementations are thread-safe for concurrent usage:

```python
import anyio

async def process_query(query, cached_client):
    response = await cached_client.create_chat_completion(
        messages=[{"role": "user", "content": query}],
        model="gpt-3.5-turbo",
        temperature=0.0
    )
    return response

async def main():
    cached_client = SemanticCacheOpenAIClient(
        base_client=AsyncClient(),
        similarity_threshold=0.85,
        max_cache_size=1000,
        ttl=3600
    )
    
    # Process multiple queries concurrently
    queries = [
        "What is the capital of France?",
        "What is the population of Paris?",
        "What's the capital city of France?",
        "Tell me about the Eiffel Tower."
    ]
    
    # Run concurrently
    tasks = [process_query(query, cached_client) for query in queries]
    results = await anyio.gather(*tasks)
    
    # Print cache statistics
    print(cached_client.get_cache_stats())

anyio.run(main)
```

## Monitoring and Metrics

### Available Metrics

The cache provides these metrics:

| Metric | Description |
|--------|-------------|
| `total_requests` | Total number of requests processed |
| `cache_hits` | Number of requests served from cache |
| `cache_misses` | Number of requests that required API calls |
| `avg_similarity` | Average similarity score for all queries |
| `token_savings` | Estimated number of tokens saved |
| `total_entries` | Total entries in the cache |
| `active_entries` | Non-expired entries in the cache |
| `expired_entries` | Expired entries not yet removed |

### Tracking Performance

Monitor cache effectiveness:

```python
# After running some queries
stats = cached_client.get_cache_stats()

# Calculate key metrics
hit_rate = stats['cache_hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
cost_savings = stats['token_savings'] * 0.00002  # $0.02 per 1K tokens

print(f"Cache hit rate: {hit_rate:.1%}")
print(f"API cost savings: ${cost_savings:.4f}")
print(f"Average similarity score: {stats['avg_similarity']:.4f}")
print(f"Cache utilization: {stats['active_entries']}/{stats['max_size']}")
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Low cache hit rate | Similarity threshold too high | Lower similarity threshold (e.g., from 0.90 to 0.85) |
| | Varied query formats | Standardize query formats before caching |
| | Non-deterministic responses | Ensure temperature=0.0 for cacheable queries |
| High memory usage | Cache size too large | Reduce max_cache_size |
| | TTL too long | Reduce TTL to expire entries sooner |
| | Too many entries | Call clear_cache() periodically |
| Thread safety issues | Concurrent modification | Ensure all client methods are called through the cache wrapper |
| Stale responses | TTL too long | Reduce TTL or clear cache when data changes |
| Poor similarity detection | Basic embedding model | Provide a better embedding_model |

### Debugging Techniques

Enable debug logging for more detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("semantic_cache")
```

Inspect cache behavior:

```python
# Before query
stats_before = cached_client.get_cache_stats()

# Run query
response = await cached_client.create_chat_completion(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    model="gpt-3.5-turbo",
    temperature=0.0
)

# After query
stats_after = cached_client.get_cache_stats()

# Check if it was a cache hit
cache_hit = stats_after['cache_hits'] > stats_before['cache_hits']
print(f"Cache hit: {cache_hit}")
```

## API-Specific Features

### OpenAI

The OpenAI implementation supports:

- All chat completion models
- Streaming completions (bypasses cache)
- Function calling/tool use
- Embedding API integration with text-embedding-3-small

Example with function calling:

```python
from openai import AsyncClient
from examples.openai_semantic_cache import SemanticCacheOpenAIClient

async def main():
    cached_client = SemanticCacheOpenAIClient(base_client=AsyncClient())
    
    # Function calling bypasses cache by default
    response = await cached_client.create_chat_completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
    )
```

### Claude (Anthropic)

The Claude implementation supports:

- All Claude models
- Messages API format
- Multi-part messages
- Streaming responses (bypasses cache)
- Claude embedding API integration

Example with multi-part messages:

```python
import anthropic
from examples.claude_cache.claude_semantic_cache import SemanticCacheClaudeClient

async def main():
    client = anthropic.Anthropic()
    cached_client = SemanticCacheClaudeClient(base_client=client)
    
    # Multi-part messages
    response = await cached_client.chat(
        model="claude-3-opus-20240229",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this data:"},
                    {"type": "text", "text": "Temperature: 72F, Humidity: 65%, Wind: 5mph"}
                ]
            }
        ],
        temperature=0.0
    )
```

### Gemini (Google)

The Gemini implementation supports:

- All Gemini models
- Content generation API
- Streaming responses (bypasses cache)
- Multi-modal prompts

Example with Gemini:

```python
from examples.semantic_cache import SemanticCacheGeminiClient
import google.generativeai as genai

async def main():
    genai.configure(api_key="your_api_key")
    model = genai.GenerativeModel('gemini-pro')
    
    cached_client = SemanticCacheGeminiClient(
        base_client=model,
        similarity_threshold=0.85
    )
    
    response = await cached_client.generate_content(
        "What is the capital of France?",
        temperature=0.0
    )
```

### Groq

The Groq implementation supports:

- All Groq models (Llama, Mixtral, etc.)
- OpenAI-compatible interface
- Streaming responses (bypasses cache)

Example with Groq:

```python
import groq
from examples.groq_semantic_cache import SemanticCacheGroqClient

async def main():
    client = groq.AsyncClient(api_key="your_api_key")
    cached_client = SemanticCacheGroqClient(
        base_client=client,
        similarity_threshold=0.85
    )
    
    response = await cached_client.create_chat_completion(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0.0
    )
```

## Examples

### Complex Conversation Caching

Caching multi-turn conversations:

```python
async def main():
    cached_client = SemanticCacheOpenAIClient(base_client=AsyncClient())
    
    # First conversation
    conversation1 = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What's the population?"}
    ]
    
    response1 = await cached_client.create_chat_completion(
        model="gpt-3.5-turbo",
        messages=conversation1,
        temperature=0.0
    )
    
    # Semantically similar conversation (should hit cache)
    conversation2 = [
        {"role": "user", "content": "What's the capital city of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."},
        {"role": "user", "content": "How many people live there?"}
    ]
    
    response2 = await cached_client.create_chat_completion(
        model="gpt-3.5-turbo",
        messages=conversation2,
        temperature=0.0
    )
```

### Combining with API Key Multiplexing

Use semantic caching with API key multiplexing:

```python
from ipfs_accelerate_py.api_key_multiplexer import ApiKeyMultiplexer
from examples.openai_semantic_cache import SemanticCacheOpenAIClient
from openai import AsyncClient

async def main():
    # Set up multiplexer
    multiplexer = ApiKeyMultiplexer()
    multiplexer.add_openai_key("key1", "sk-key1...")
    multiplexer.add_openai_key("key2", "sk-key2...")
    
    # Get a client using the multiplexer
    base_client = multiplexer.get_openai_client(strategy="least-loaded")
    
    # Wrap with semantic cache
    cached_client = SemanticCacheOpenAIClient(
        base_client=base_client,
        similarity_threshold=0.85
    )
    
    # Use the cached client
    response = await cached_client.create_chat_completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0.0
    )
```

### Application Integration Example

Complete example of integrating with a web application:

```python
from fastapi import FastAPI, BackgroundTasks
from examples.openai_semantic_cache import SemanticCacheOpenAIClient
from openai import AsyncClient
import anyio
import time

app = FastAPI()

# Create cached client
cached_client = SemanticCacheOpenAIClient(
    base_client=AsyncClient(),
    similarity_threshold=0.85,
    max_cache_size=10000,
    ttl=3600
)

# Cache maintenance task
async def clean_cache():
    while True:
        # Log cache statistics every hour
        stats = cached_client.get_cache_stats()
        print(f"Cache stats: {stats}")
        
        # Wait for an hour
        await anyio.sleep(3600)

@app.on_event("startup")
async def startup_event():
    # Start cache maintenance task
    anyio.create_task(clean_cache())

@app.post("/chat")
async def chat(query: str):
    start_time = time.time()
    
    # Get response from cached client
    response = await cached_client.create_chat_completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}],
        temperature=0.0
    )
    
    # Extract text from response
    if 'choices' in response and len(response['choices']) > 0:
        text = response['choices'][0]['message']['content']
    else:
        text = str(response)
    
    # Calculate time taken
    time_taken = time.time() - start_time
    
    # Get updated stats
    stats = cached_client.get_cache_stats()
    hit_rate = stats['cache_hits'] / stats['total_requests'] if stats['total_requests'] > 0 else 0
    
    return {
        "response": text,
        "time_taken": time_taken,
        "cache_hit_rate": hit_rate
    }
```

This comprehensive guide should provide all the information needed to effectively integrate and use semantic caching with various LLM API providers in the IPFS Accelerate Python framework.