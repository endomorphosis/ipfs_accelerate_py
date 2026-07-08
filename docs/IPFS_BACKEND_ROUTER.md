# IPFS Backend Router Documentation

## Overview

The IPFS Backend Router provides a flexible, pluggable backend system for IPFS operations within `ipfs_accelerate_py`. It implements a preference-based fallback strategy:

1. **ipfs_kit_py** (Preferred) - Full distributed storage capabilities
2. **HuggingFace Cache** (Fallback 1) - Local storage with IPFS-like addressing
3. **Kubo CLI** (Fallback 2) - Standard IPFS daemon via CLI

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              IPFS Backend Router                        │
├─────────────────────────────────────────────────────────┤
│  Convenience Functions (add_bytes, cat, pin, etc.)     │
├─────────────────────────────────────────────────────────┤
│              Backend Selection Layer                    │
│  • Environment-based configuration                      │
│  • Automatic fallback on failure                        │
│  • Backend registry for custom providers               │
├─────────────────────────────────────────────────────────┤
│              Backend Implementations                    │
├──────────────┬──────────────┬──────────────────────────┤
│ ipfs_kit_py  │  HF Cache    │  Kubo CLI                │
│ (Preferred)  │  (Fallback1) │  (Fallback2)             │
└──────────────┴──────────────┴──────────────────────────┘
```

## Installation

The router is included in `ipfs_accelerate_py`. For full functionality:

```bash
# Basic installation (includes HF Cache and Kubo backends)
pip install ipfs_accelerate_py

# With ipfs_kit_py for distributed storage (recommended)
pip install ipfs_accelerate_py[ipfs_kit]

# Ensure Kubo (go-ipfs) is installed for CLI backend
# https://docs.ipfs.tech/install/command-line/
```

## Quick Start

### Basic Usage

```python
from ipfs_accelerate_py import ipfs_backend_router

# Store data and get CID
data = b"Hello, IPFS!"
cid = ipfs_backend_router.add_bytes(data, pin=True)
print(f"Stored with CID: {cid}")

# Retrieve data
retrieved = ipfs_backend_router.cat(cid)
assert retrieved == data

# Store a file
cid = ipfs_backend_router.add_path("/path/to/model.bin", pin=True)

# Retrieve to specific path
ipfs_backend_router.get_to_path(cid, output_path="/path/to/output.bin")
```

### With Model Manager

```python
from ipfs_accelerate_py.model_manager import ModelManager

# Enable IPFS storage for models
manager = ModelManager(enable_ipfs=True)

# Store model to IPFS
cid = manager.store_model_to_ipfs(
    model_path="/path/to/model",
    model_id="bert-base-uncased"
)

# Retrieve model from IPFS
success = manager.retrieve_model_from_ipfs(
    cid=cid,
    output_path="/path/to/cache",
    model_id="bert-base-uncased"
)
```

### With HF Model Server

```python
from ipfs_accelerate_py.hf_model_server.loader.cache import ModelCache

# Enable IPFS for model cache
cache = ModelCache(max_size=10, enable_ipfs=True)

# Store model weights to IPFS
cid = await cache.store_model_to_ipfs(
    model_id="gpt2",
    model_path="/path/to/gpt2"
)

# Retrieve model from IPFS
success = await cache.retrieve_model_from_ipfs(
    cid=cid,
    output_path="/path/to/cache/gpt2"
)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IPFS_BACKEND` | (auto) | Force specific backend (e.g., "ipfs_kit", "hf_cache", "kubo") |
| `ENABLE_IPFS_KIT` | true | Enable ipfs_kit_py backend |
| `IPFS_KIT_DISABLE` | false | Explicitly disable ipfs_kit_py |
| `ENABLE_HF_CACHE` | true | Enable HuggingFace cache backend |
| `KUBO_CMD` | ipfs | IPFS CLI command path |
| `HF_HOME` | ~/.cache/huggingface | HuggingFace cache directory |
| `IPFS_ROUTER_CACHE` | 1 | Enable backend instance caching |
| `ENABLE_IPFS_MODEL_CACHE` | false | Enable IPFS for model server cache |
| `ENABLE_IPFS_MODEL_STORAGE` | false | Enable IPFS for model manager |

### Configuration Examples

#### Prefer ipfs_kit_py with fallback

```bash
# Default behavior - prefers ipfs_kit_py, falls back to HF cache, then Kubo
export ENABLE_IPFS_KIT=true
python your_script.py
```

#### Use HuggingFace Cache only

```bash
# Disable ipfs_kit_py and Kubo, use HF cache
export IPFS_KIT_DISABLE=1
export IPFS_BACKEND=hf_cache
python your_script.py
```

#### Force Kubo CLI backend

```bash
export IPFS_BACKEND=kubo
export KUBO_CMD=/usr/local/bin/ipfs
python your_script.py
```

#### Configure for CI/CD

```bash
# Minimal configuration for testing
export IPFS_KIT_DISABLE=1
export IPFS_BACKEND=hf_cache
export HF_HOME=/tmp/test_cache
python -m pytest
```

## Backend Details

### 1. ipfs_kit_py Backend (Preferred)

**Features:**
- Full distributed storage
- P2P content distribution
- Multi-backend support (IPFS, S3, Filecoin)
- Automatic local caching

**Requirements:**
- `ipfs_kit_py` package installed
- Network connectivity for distributed operations

**Use Cases:**
- Production deployments
- Distributed model sharing
- Content-addressed model registry

### 2. HuggingFace Cache Backend (Fallback 1)

**Features:**
- Local filesystem storage
- IPFS-like CID generation
- Integrates with HF model cache
- No external dependencies

**Requirements:**
- Write access to `HF_HOME` directory

**Use Cases:**
- Development environments
- CI/CD pipelines
- Air-gapped deployments
- Local model caching

### 3. Kubo CLI Backend (Fallback 2)

**Features:**
- Standard IPFS daemon integration
- Full IPFS feature set
- Block-level operations
- IPNS support

**Requirements:**
- Kubo (go-ipfs) installed and in PATH
- IPFS daemon running (for some operations)

**Use Cases:**
- Existing IPFS infrastructure
- Full IPFS protocol support
- Gateway operations

## API Reference

### Core Functions

#### `add_bytes(data: bytes, *, pin: bool = True) -> str`
Store bytes and return CID.

#### `cat(cid: str) -> bytes`
Retrieve data by CID.

#### `pin(cid: str) -> None`
Pin content to prevent garbage collection.

#### `unpin(cid: str) -> None`
Unpin content.

#### `block_put(data: bytes, *, codec: str = "raw") -> str`
Store raw block and return CID.

#### `block_get(cid: str) -> bytes`
Get raw block by CID.

#### `add_path(path: str, *, recursive: bool = True, pin: bool = True) -> str`
Add file or directory to IPFS.

#### `get_to_path(cid: str, *, output_path: str) -> None`
Retrieve content and save to path.

### Backend Management

#### `get_backend(*, deps=None, backend=None) -> IPFSBackend`
Get the current backend instance.

#### `set_default_ipfs_backend(backend: IPFSBackend | None) -> None`
Set global default backend.

#### `register_ipfs_backend(name: str, factory: Callable) -> None`
Register custom backend provider.

### Custom Backend Example

```python
from ipfs_accelerate_py import ipfs_backend_router

class CustomBackend:
    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        # Custom implementation
        pass
    
    def cat(self, cid: str) -> bytes:
        # Custom implementation
        pass
    
    # Implement other required methods...

# Register
ipfs_backend_router.register_ipfs_backend(
    "custom",
    lambda: CustomBackend()
)

# Use
import os
os.environ["IPFS_BACKEND"] = "custom"
cid = ipfs_backend_router.add_bytes(b"data")
```

## Integration Examples

### With Datasets

```python
from ipfs_accelerate_py import ipfs_backend_router
import datasets

# Store dataset to IPFS
dataset = datasets.load_dataset("squad", split="train[:100]")
dataset.save_to_disk("/tmp/squad_sample")
cid = ipfs_backend_router.add_path("/tmp/squad_sample", recursive=True)

# Share CID, others can retrieve
ipfs_backend_router.get_to_path(cid, output_path="/tmp/squad_retrieved")
retrieved = datasets.load_from_disk("/tmp/squad_retrieved")
```

### With Transformers

```python
from transformers import AutoModel, AutoTokenizer
from ipfs_accelerate_py import ipfs_backend_router

# Save model locally
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model.save_pretrained("/tmp/bert")
tokenizer.save_pretrained("/tmp/bert")

# Store to IPFS
cid = ipfs_backend_router.add_path("/tmp/bert", recursive=True)
print(f"Model CID: {cid}")

# Retrieve from IPFS
ipfs_backend_router.get_to_path(cid, output_path="/tmp/bert_from_ipfs")
model_retrieved = AutoModel.from_pretrained("/tmp/bert_from_ipfs")
```

## Performance Considerations

### Backend Selection Impact

| Backend | Latency | Throughput | Network | Use Case |
|---------|---------|------------|---------|----------|
| ipfs_kit_py | Low-Med | High | Required | Production |
| HF Cache | Very Low | Very High | None | Development |
| Kubo CLI | Medium | Medium | Optional | Full IPFS |

### Optimization Tips

1. **Enable Caching**: Keep `IPFS_ROUTER_CACHE=1` for backend reuse
2. **Local-First**: Use HF Cache backend for local development
3. **Pin Important Data**: Use `pin=True` for models you'll reuse
4. **Batch Operations**: Use `add_path` for directories instead of individual files
5. **Choose Right Backend**: Use ipfs_kit_py for distributed, HF cache for local

## Troubleshooting

### Common Issues

#### "ipfs_kit_py not available"
**Solution**: Install ipfs_kit_py or set `IPFS_KIT_DISABLE=1`

#### "CID not found in HF cache"
**Solution**: Data not stored locally. Check if CID is correct or use different backend.

#### "ipfs command failed"
**Solution**: 
- Ensure Kubo is installed and in PATH
- Check if IPFS daemon is running: `ipfs daemon`
- Verify `KUBO_CMD` environment variable

#### Backend selection not working
**Solution**:
- Clear cache: `ipfs_backend_router._get_default_backend_cached.cache_clear()`
- Check environment variables
- Verify backend registration

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from ipfs_accelerate_py import ipfs_backend_router

# This will show backend selection and operations
backend = ipfs_backend_router.get_backend()
print(f"Using backend: {type(backend).__name__}")
```

## Testing

Run the test suite:

```bash
# All router tests
pytest test/test_ipfs_backend_router.py -v

# Specific backend tests
pytest test/test_ipfs_backend_router.py::TestHuggingFaceCacheBackend -v

# With coverage
pytest test/test_ipfs_backend_router.py --cov=ipfs_accelerate_py.ipfs_backend_router
```

## Security Considerations

1. **Data Privacy**: IPFS content is content-addressed and potentially public
2. **Pinning**: Pin sensitive data carefully; it's harder to remove
3. **Network Access**: ipfs_kit_py and Kubo may expose data to network
4. **Local Storage**: HF Cache keeps everything local by default

## Migration Guide

### From ipfs_datasets_py

The router is API-compatible with `ipfs_datasets_py.ipfs_backend_router`:

```python
# Old code
from ipfs_datasets_py import ipfs_backend_router
cid = ipfs_backend_router.block_put(data)

# New code (drop-in replacement)
from ipfs_accelerate_py import ipfs_backend_router
cid = ipfs_backend_router.block_put(data)
```

Key differences:
- Backend preference: ipfs_kit_py → HF cache → Kubo (vs. just Kubo)
- Additional backends available
- Better fallback handling

## Contributing

To add a new backend:

1. Implement the `IPFSBackend` protocol
2. Register with `register_ipfs_backend()`
3. Add tests in `test_ipfs_backend_router.py`
4. Document in this guide

See `ipfs_backend_router.py` for protocol definition.

## License

See main repository LICENSE file.

## Support

- Issues: https://github.com/endomorphosis/ipfs_accelerate_py/issues
- Documentation: https://github.com/endomorphosis/ipfs_accelerate_py/tree/main/docs
