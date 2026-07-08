# Embeddings Router for IPFS Accelerate

The Embeddings Router provides a unified interface for generating text embeddings across multiple providers, with built-in caching, automatic fallback, and integration with the existing IPFS Accelerate endpoint multiplexing infrastructure.

## Features

- **Unified API**: Single `embed_texts()` function works with all providers
- **Multiple Providers**: OpenRouter, Gemini CLI, HuggingFace (sentence-transformers/transformers), Backend Manager
- **Automatic Fallback**: Tries multiple providers in order of availability
- **Response Caching**: CID-based or SHA256-based caching for deterministic results
- **Batch Processing**: Efficient batched embedding generation with per-item caching
- **Dependency Injection**: Share resources (caches, managers) across calls
- **Integration**: Works seamlessly with existing CLI wrappers and backend manager
- **No Duplication**: Reuses existing infrastructure

## Quick Start

### Basic Usage

```python
from ipfs_accelerate_py import embed_texts, embed_text

# Auto-select best available provider with fallback
embeddings = embed_texts(["Hello world", "IPFS accelerates ML"])
print(f"Generated {len(embeddings)} embeddings of {len(embeddings[0])} dimensions")

# Single text embedding
embedding = embed_text("Distributed machine learning")
print(f"Embedding vector: {len(embedding)} dimensions")
```

### Using a Specific Provider

```python
from ipfs_accelerate_py import embed_texts

# Use OpenRouter
embeddings = embed_texts(
    ["Text 1", "Text 2", "Text 3"],
    provider="openrouter",
    model_name="text-embedding-3-small"
)

# Use HuggingFace locally
embeddings = embed_texts(
    ["Text 1", "Text 2"],
    provider="huggingface",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)
```

### With Caching

```python
import os
from ipfs_accelerate_py import embed_texts

# Enable response cache (enabled by default)
os.environ["IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE"] = "1"

# First call - cache miss
embeddings1 = embed_texts(["Machine learning", "Deep learning"])

# Second call - cache hit (much faster)
embeddings2 = embed_texts(["Machine learning", "Deep learning"])
```

### Custom Provider Registration

```python
from ipfs_accelerate_py import register_embeddings_provider, embed_texts

# Define a custom provider
class MyEmbeddingsProvider:
    def embed_texts(self, texts, *, model_name=None, device=None, **kwargs):
        # Your embedding logic here
        return [[1.0, 2.0, 3.0] for _ in texts]

# Register it
register_embeddings_provider("my_provider", lambda: MyEmbeddingsProvider())

# Use it
embeddings = embed_texts(["test"], provider="my_provider")
```

### Dependency Injection

```python
from ipfs_accelerate_py import RouterDeps, embed_texts

# Create shared deps container
deps = RouterDeps()

# You can inject pre-configured components
# deps.backend_manager = my_backend_manager
# deps.remote_cache = my_remote_cache

# All calls will share the same resources
embeddings1 = embed_texts(["First batch"], deps=deps)
embeddings2 = embed_texts(["Second batch"], deps=deps)

print(f"Shared cache has {len(deps.router_cache)} items")
```

## Available Providers

### Built-in Providers

#### 1. OpenRouter (`openrouter`)
API-based access to multiple embedding models.

**Configuration:**
```bash
export OPENROUTER_API_KEY="your-api-key"
export IPFS_ACCELERATE_PY_OPENROUTER_EMBEDDINGS_MODEL="text-embedding-3-small"  # optional
```

**Usage:**
```python
embeddings = embed_texts(
    ["Your texts"],
    provider="openrouter",
    model_name="text-embedding-3-small"
)
```

#### 2. Gemini CLI (`gemini_cli`)
Uses existing Gemini CLI integration (SDK-based).

**Configuration:**
```bash
export GOOGLE_API_KEY="your-api-key"
export IPFS_ACCELERATE_PY_GEMINI_EMBEDDINGS_MODEL="embedding-001"  # optional
```

**Usage:**
```python
embeddings = embed_texts(
    ["Your texts"],
    provider="gemini_cli"
)
```

#### 3. HuggingFace (`huggingface` or `local_hf`)
Local embeddings using sentence-transformers or transformers.

**Configuration:**
```bash
export IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export IPFS_ACCELERATE_PY_EMBEDDINGS_DEVICE="cpu"  # or "cuda"
```

**Usage:**
```python
embeddings = embed_texts(
    ["Your texts"],
    provider="huggingface",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)
```

**Models:**
- sentence-transformers models (recommended): Fast, optimized for similarity
- Any HuggingFace transformers model: More flexible but requires manual pooling

#### 4. Backend Manager (`backend_manager`)
Uses InferenceBackendManager for distributed/multiplexed inference.

**Configuration:**
```bash
export IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER="1"
export IPFS_ACCELERATE_PY_EMBEDDINGS_LOAD_BALANCING="round_robin"  # or least_loaded, best_performance
```

**Usage:**
```python
embeddings = embed_texts(
    ["Your texts"],
    provider="backend_manager"
)
```

## Environment Variables

### Provider Selection
- `IPFS_ACCELERATE_PY_EMBEDDINGS_PROVIDER`: Force a specific provider (bypasses auto-detection)

### Caching
- `IPFS_ACCELERATE_PY_ROUTER_CACHE`: Enable/disable provider caching (default: "1")
- `IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE`: Enable/disable response caching (default: "1")
- `IPFS_ACCELERATE_PY_ROUTER_CACHE_KEY`: Cache key strategy ("sha256" or "cid", default: "sha256")
- `IPFS_ACCELERATE_PY_ROUTER_CACHE_CID_BASE`: CID encoding base (default: "base32")

### Backend Manager
- `IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER`: Enable backend manager provider (default: "0")
- `IPFS_ACCELERATE_PY_EMBEDDINGS_LOAD_BALANCING`: Load balancing strategy (default: "round_robin")

### Default Model & Device
- `IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL`: Default model for providers that support it
- `IPFS_ACCELERATE_PY_EMBEDDINGS_DEVICE`: Default device (cpu/cuda)
- `IPFS_ACCELERATE_PY_EMBEDDINGS_BACKEND`: Force backend for local adapter

## Provider Resolution Order

When no provider is specified, the router tries providers in this order:

1. Backend Manager (if enabled via env var)
2. OpenRouter (if API key configured)
3. Gemini CLI (if available)
4. Local HuggingFace (fallback)

## Integration with Existing Infrastructure

### CLI Wrappers
The router integrates with existing CLI wrappers without duplication:

```python
# These are already available and used by the router
from ipfs_accelerate_py.cli_integrations import (
    GeminiCLIIntegration
)
```

### Backend Manager
The router can use the InferenceBackendManager for distributed inference:

```python
from ipfs_accelerate_py import embed_texts
import os

# Enable backend manager
os.environ["IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER"] = "1"

# This will route through the backend manager
embeddings = embed_texts(["Your texts"], provider="backend_manager")
```

### Caching
Response caching integrates with the existing CID-based caching:

```python
from ipfs_accelerate_py import embed_texts
import os

# Use CID-based caching
os.environ["IPFS_ACCELERATE_PY_ROUTER_CACHE_KEY"] = "cid"

# Embeddings are cached by content-addressed CID
embeddings = embed_texts(["Your texts"])
```

## Advanced Usage

### Provider Instance Management

```python
from ipfs_accelerate_py import get_embeddings_provider, embed_texts

# Get a provider instance
provider = get_embeddings_provider("openrouter")

# Reuse it for multiple requests
embeddings1 = embed_texts(["Batch 1"], provider_instance=provider)
embeddings2 = embed_texts(["Batch 2"], provider_instance=provider)
```

### Batch Processing with Partial Caching

```python
from ipfs_accelerate_py import embed_texts

# First batch
texts1 = ["Text A", "Text B", "Text C"]
embeddings1 = embed_texts(texts1)  # All cache misses

# Second batch (partial overlap)
texts2 = ["Text B", "Text C", "Text D"]
embeddings2 = embed_texts(texts2)  # Text B and C are cache hits!
```

### Similarity Search Example

```python
from ipfs_accelerate_py import embed_texts, embed_text
import numpy as np

# Generate embeddings for corpus
corpus = [
    "Machine learning with IPFS",
    "Distributed computing networks",
    "Neural network architectures"
]
corpus_embeddings = embed_texts(corpus)

# Generate query embedding
query = "AI and distributed systems"
query_embedding = embed_text(query)

# Calculate cosine similarity
corpus_array = np.array(corpus_embeddings)
query_array = np.array(query_embedding)

similarities = np.dot(corpus_array, query_array) / (
    np.linalg.norm(corpus_array, axis=1) * np.linalg.norm(query_array)
)

# Find most similar
max_idx = np.argmax(similarities)
print(f"Most similar: {corpus[max_idx]}")
print(f"Similarity: {similarities[max_idx]:.4f}")
```

### Custom Dependency Container

```python
from ipfs_accelerate_py import RouterDeps, embed_texts
from ipfs_accelerate_py.inference_backend_manager import get_backend_manager

# Create custom deps with pre-configured components
deps = RouterDeps()
deps.backend_manager = get_backend_manager()

# All calls use the same backend manager
embeddings1 = embed_texts(["First"], deps=deps)
embeddings2 = embed_texts(["Second"], deps=deps)
```

### Clear Caches

```python
from ipfs_accelerate_py import clear_embeddings_router_caches

# Clear internal provider caches
clear_embeddings_router_caches()
```

## Examples

See `examples/embeddings_router_example.py` for comprehensive usage examples.

Run the example:
```bash
python examples/embeddings_router_example.py
```

## Testing

Run the integration tests:
```bash
python test/test_embeddings_router_integration.py
```

## Architecture

The Embeddings Router follows these design principles:

1. **No Import-Time Side Effects**: All heavy imports are lazy
2. **Reuse Existing Infrastructure**: No duplication of CLI wrappers or backend managers
3. **Dependency Injection**: Optional `RouterDeps` for sharing resources
4. **Provider Registry**: Extensible via `register_embeddings_provider()`
5. **Automatic Fallback**: Tries multiple providers in order
6. **CID-Based Caching**: Content-addressed caching for determinism
7. **Batch Processing**: Efficient per-item caching for batches
8. **Integration Ready**: Works with existing endpoint multiplexing

## Benefits Over ipfs_datasets_py Implementation

1. **Full Integration**: Seamlessly works with existing InferenceBackendManager
2. **No Duplication**: Reuses all existing CLI/SDK wrappers
3. **Distributed Ready**: Supports distributed/P2P inference via backend manager
4. **CID Caching**: Built-in CID-based caching support
5. **Existing Patterns**: Follows patterns from llm_router
6. **Endpoint Multiplexing**: Can multiplex across peers via backend manager
7. **Batch Optimization**: Per-item caching for efficient batch processing

## Use Cases

### 1. Semantic Search
```python
# Index documents with embeddings
documents = ["Doc 1", "Doc 2", "Doc 3"]
doc_embeddings = embed_texts(documents)

# Search with query
query_embedding = embed_text("search query")
# Calculate similarities and rank
```

### 2. Clustering
```python
# Generate embeddings for clustering
texts = ["Text 1", "Text 2", "..."]
embeddings = embed_texts(texts)
# Use with scikit-learn clustering algorithms
```

### 3. Classification
```python
# Generate embeddings as features
train_texts = ["Label A text", "Label B text"]
train_embeddings = embed_texts(train_texts)
# Train classifier on embeddings
```

### 4. Recommendation
```python
# User and item embeddings
user_prefs = ["preference 1", "preference 2"]
items = ["item 1", "item 2", "item 3"]
user_emb = embed_texts(user_prefs)
item_emb = embed_texts(items)
# Calculate similarities for recommendations
```

## Future Enhancements

- [ ] Add more provider implementations (Cohere, Voyage AI, etc.)
- [ ] Add pooling strategy options for transformers
- [ ] Add normalization options
- [ ] Add metrics and monitoring integration
- [ ] Add distributed caching via libp2p
- [ ] Add provider health checks

## License

See the main project LICENSE file.
