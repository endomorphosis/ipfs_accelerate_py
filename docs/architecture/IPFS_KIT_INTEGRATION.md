# IPFS Kit Integration Guide

## Overview

The `ipfs_kit_py` integration provides distributed filesystem capabilities to the IPFS Accelerate Python framework. This integration follows a **local-first** approach with automatic fallback support, making it suitable for production environments, CI/CD pipelines, and offline scenarios.

## Key Features

- ✅ **Local-First Architecture**: Works offline with local filesystem fallback
- ✅ **Distributed Storage**: Leverages IPFS, Filecoin, and other backends when available
- ✅ **Content-Addressed**: All data stored with cryptographic CIDs
- ✅ **Automatic Fallback**: Gracefully degrades when ipfs_kit_py is unavailable
- ✅ **CI/CD Friendly**: Can be disabled via environment variables
- ✅ **Multi-Backend Support**: IPFS, S3, Filecoin, local filesystem
- ✅ **Crash-Resistant**: Uses Write-Ahead Log for durability

## Architecture

```
┌────────────────────────────────────────────────────────┐
│           IPFS Accelerate Application                  │
└─────────────────┬──────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│          IPFSKitStorage (Integration Layer)             │
│  - Unified API                                          │
│  - Automatic fallback detection                         │
│  - Content addressing (CID)                             │
└───────────┬─────────────────────────────┬───────────────┘
            │                             │
            │ ipfs_kit_py available?      │
            ▼                             ▼
    ┌───────────────┐           ┌────────────────┐
    │   IPFS Kit    │           │ Local Fallback │
    │   (External)  │           │   (Built-in)   │
    │               │           │                │
    │ • IPFS        │           │ • Filesystem   │
    │ • Filecoin    │           │ • CID-like IDs │
    │ • S3          │           │ • Metadata     │
    │ • VFS         │           │                │
    └───────────────┘           └────────────────┘
```

## Installation

### Basic Installation (with fallback)

The integration is included in the main package and will work with local fallback even if ipfs_kit_py is not installed:

```bash
pip install ipfs-accelerate-py
```

### Full Installation (with ipfs_kit_py)

To enable distributed storage capabilities, initialize the ipfs_kit_py submodule:

```bash
# Clone the repository
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Initialize the ipfs_kit_py submodule
git submodule update --init external/ipfs_kit_py

# Install with all dependencies
pip install -e .
```

## Usage

### Basic Usage with Automatic Fallback

```python
from ipfs_accelerate_py import get_storage

# Get storage instance (automatically detects ipfs_kit_py)
storage = get_storage()

# Check backend status
status = storage.get_backend_status()
print(f"Using IPFS Kit: {status['ipfs_kit_available']}")
print(f"Using fallback: {status['using_fallback']}")

# Store data (works with or without ipfs_kit_py)
data = b"Hello, distributed world!"
cid = storage.store(data, filename="greeting.txt")
print(f"Stored with CID: {cid}")

# Retrieve data
retrieved = storage.retrieve(cid)
print(f"Retrieved: {retrieved.decode('utf-8')}")

# List stored files
files = storage.list_files()
for file_info in files:
    print(f"{file_info['filename']}: {file_info['cid']}")
```

### Custom Configuration

```python
from ipfs_accelerate_py import IPFSKitStorage

# Custom configuration
config = {
    'enable_ipfs': True,
    'enable_s3': False,
    'enable_filecoin': True,
}

storage = IPFSKitStorage(
    enable_ipfs_kit=True,
    cache_dir="~/.custom_cache",
    config=config
)
```

### CI/CD Environment (Force Fallback)

```python
# Explicitly disable ipfs_kit_py for CI/CD
storage = IPFSKitStorage(force_fallback=True)

# Or use environment variable
# export IPFS_KIT_DISABLE=1
storage = get_storage()  # Will automatically use fallback
```

### Content Pinning

```python
# Store with pinning
cid = storage.store(data, filename="important.bin", pin=True)

# Pin existing content
storage.pin(cid)

# Unpin content
storage.unpin(cid)

# Check if content exists
if storage.exists(cid):
    print("Content is available")
```

### File Operations

```python
from pathlib import Path

# Store from file
file_path = Path("model_weights.bin")
cid = storage.store(file_path, filename="model_weights.bin")

# Store string data
text = "Configuration data"
cid = storage.store(text, filename="config.txt")

# Delete content
storage.delete(cid)
```

## Environment Variables

Control the integration behavior via environment variables:

```bash
# Disable ipfs_kit_py (use fallback)
export IPFS_KIT_DISABLE=1

# Custom cache directory
export IPFS_ACCELERATE_CACHE_DIR="~/.custom_cache"
```

## Integration Points

The IPFS Kit integration is designed to be used throughout the accelerate framework:

### 1. Model Storage

```python
from ipfs_accelerate_py import get_storage
from ipfs_accelerate_py import ModelManager

storage = get_storage()
model_manager = ModelManager()

# Store model with content addressing
model_data = model_manager.export_model("bert-base-uncased")
cid = storage.store(model_data, filename="bert-base-uncased.bin", pin=True)

# Retrieve model by CID
model_data = storage.retrieve(cid)
```

### 2. Inference Caching

```python
# Cache inference results
input_data = {"text": "Hello, world!"}
input_cid = storage.store(json.dumps(input_data).encode())

# ... perform inference ...

output_data = {"result": "processed"}
output_cid = storage.store(json.dumps(output_data).encode())

# Associate input → output
metadata = {
    'input_cid': input_cid,
    'output_cid': output_cid,
    'model': 'bert-base-uncased',
    'timestamp': time.time()
}
```

### 3. Distributed Dataset Sharing

```python
# Store dataset chunks
for i, chunk in enumerate(dataset_chunks):
    cid = storage.store(chunk, filename=f"dataset_chunk_{i}.bin", pin=True)
    chunk_cids.append(cid)

# Create dataset manifest
manifest = {
    'chunks': chunk_cids,
    'total_size': total_size,
    'chunk_size': chunk_size
}
manifest_cid = storage.store(json.dumps(manifest).encode())
```

## Backend Status

Check which backends are available:

```python
status = storage.get_backend_status()

print(status)
# Output:
# {
#     'ipfs_kit_available': False,  # True if ipfs_kit_py is loaded
#     'using_fallback': True,        # True if using local fallback
#     'cache_dir': '/home/user/.cache/ipfs_accelerate',
#     'backends': {
#         'local': True,             # Always available
#         'ipfs': False,             # Available when ipfs_kit_py loaded
#         's3': False,               # Detected from ipfs_kit_py config
#         'filecoin': False          # Detected from ipfs_kit_py config
#     }
# }
```

## Content Addressing

The integration uses content-addressed storage with CID (Content Identifier) generation:

```python
# Same content = same CID (deduplication)
cid1 = storage.store(b"Hello")
cid2 = storage.store(b"Hello")
assert cid1 == cid2  # True

# Different content = different CID
cid3 = storage.store(b"World")
assert cid1 != cid3  # True

# CID format: bafyXXXXX (mimics IPFS CIDv1)
print(cid1)  # bafy2bzacedxxxx...
```

## Testing

Run the integration tests:

```bash
# Run all integration tests
pytest test/test_ipfs_kit_integration.py -v

# Run specific test class
pytest test/test_ipfs_kit_integration.py::TestStorageOperations -v

# Run with coverage
pytest test/test_ipfs_kit_integration.py --cov=ipfs_accelerate_py.ipfs_kit_integration
```

## Migration Guide

### From Direct Filesystem Operations

**Before:**
```python
import json
from pathlib import Path

# Direct filesystem operations
cache_dir = Path("~/.cache/myapp").expanduser()
cache_dir.mkdir(parents=True, exist_ok=True)

# Store data
data_path = cache_dir / "data.json"
with open(data_path, 'w') as f:
    json.dump(data, f)

# Retrieve data
with open(data_path, 'r') as f:
    data = json.load(f)
```

**After:**
```python
import json
from ipfs_accelerate_py import get_storage

# Use content-addressed storage
storage = get_storage()

# Store data (gets CID)
data_bytes = json.dumps(data).encode()
cid = storage.store(data_bytes, filename="data.json")

# Retrieve data by CID
retrieved = storage.retrieve(cid)
data = json.loads(retrieved.decode())
```

### From Mock IPFS Client

**Before:**
```python
from ipfs_accelerate_py.mcp.tools.mock_ipfs import MockIPFSClient

client = MockIPFSClient()
result = client.add_file(data, "file.txt")
cid = result['Hash']
```

**After:**
```python
from ipfs_accelerate_py import get_storage

storage = get_storage()  # Uses real IPFS when available
cid = storage.store(data, filename="file.txt")
```

## Advanced Features

### Multi-Backend Routing

When ipfs_kit_py is available, the integration can route storage operations across multiple backends:

```python
# This will be implemented when ipfs_kit_py is fully integrated
# For now, it provides a consistent API that's ready for enhancement

storage = get_storage(config={
    'primary_backend': 'ipfs',
    'fallback_backend': 'local',
    'archive_backend': 'filecoin'
})
```

### Write-Ahead Log (WAL)

When ipfs_kit_py is available, operations are logged for crash recovery:

```python
# Operations are automatically logged and can be recovered
# after crashes or interruptions
```

### Health Monitoring

```python
# Check backend health
status = storage.get_backend_status()

if not status['ipfs_kit_available']:
    print("Running in local mode - this is normal for CI/CD")
elif status['using_fallback']:
    print("ipfs_kit_py import failed - check installation")
```

## Troubleshooting

### ipfs_kit_py Not Found

**Symptom**: `using_fallback: True` even when ipfs_kit_py should be available

**Solutions**:
1. Check submodule initialization:
   ```bash
   git submodule update --init external/ipfs_kit_py
   ```

2. Verify ipfs_kit_py is in Python path:
   ```python
   import sys
   from pathlib import Path
   ipfs_kit_path = Path("external/ipfs_kit_py")
   if ipfs_kit_path.exists():
       sys.path.insert(0, str(ipfs_kit_path))
   ```

3. Check for import errors:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   # Will show detailed import errors
   ```

### CI/CD Integration

For CI/CD pipelines, explicitly disable ipfs_kit_py:

```yaml
# .github/workflows/test.yml
env:
  IPFS_KIT_DISABLE: 1

steps:
  - name: Run tests
    run: pytest
```

### Storage Location

Default storage location: `~/.cache/ipfs_accelerate`

To customize:
```python
storage = get_storage(cache_dir="/custom/path")
```

Or via environment:
```bash
export IPFS_ACCELERATE_CACHE_DIR="/custom/path"
```

## API Reference

### IPFSKitStorage

Main storage interface class.

#### `__init__(enable_ipfs_kit=True, cache_dir=None, config=None, force_fallback=False)`

Initialize storage interface.

**Parameters:**
- `enable_ipfs_kit` (bool): Try to use ipfs_kit_py
- `cache_dir` (str): Local cache directory
- `config` (dict): Additional configuration
- `force_fallback` (bool): Force local-only mode

#### `store(data, filename=None, pin=False) -> str`

Store data and return CID.

**Parameters:**
- `data` (bytes|str|Path): Data to store
- `filename` (str): Optional filename hint
- `pin` (bool): Pin content (prevent GC)

**Returns:** Content ID (CID) string

#### `retrieve(cid) -> Optional[bytes]`

Retrieve data by CID.

**Parameters:**
- `cid` (str): Content identifier

**Returns:** Data as bytes, or None if not found

#### `list_files(path="/") -> List[Dict]`

List stored files.

**Returns:** List of file information dictionaries

#### `exists(cid) -> bool`

Check if content exists.

#### `delete(cid) -> bool`

Delete content by CID.

#### `pin(cid) -> bool`

Pin content (prevent garbage collection).

#### `unpin(cid) -> bool`

Unpin content (allow garbage collection).

#### `get_backend_status() -> Dict`

Get status of storage backends.

#### `is_available() -> bool`

Check if ipfs_kit_py is available.

### Helper Functions

#### `get_storage(enable_ipfs_kit=True, cache_dir=None, config=None, force_fallback=False) -> IPFSKitStorage`

Get or create singleton storage instance.

#### `reset_storage()`

Reset singleton storage instance (useful for testing).

## Best Practices

1. **Use Content Addressing**: Leverage CIDs for deduplication and verification
2. **Pin Important Data**: Use `pin=True` for critical content
3. **Check Availability**: Use `is_available()` to detect backend status
4. **Handle None Gracefully**: `retrieve()` returns None if content not found
5. **Use Singleton**: Use `get_storage()` for consistent instance access
6. **Test Both Modes**: Test with and without ipfs_kit_py available
7. **Environment-Aware**: Use `force_fallback` in CI/CD environments

## Future Enhancements

Planned enhancements when ipfs_kit_py is fully integrated:

- [ ] Multi-backend routing and redundancy
- [ ] Automatic tier migration (hot → warm → cold)
- [ ] Provider discovery and selection
- [ ] P2P content distribution
- [ ] Real-time synchronization
- [ ] Content seeding and replication
- [ ] Advanced caching strategies
- [ ] Quota management and GC policies

## See Also

- [ipfs_kit_py Documentation](../external/ipfs_kit_py/README.md)
- [IPFS Documentation](https://docs.ipfs.tech/)
- [Content Addressing Guide](https://proto.school/content-addressing)
- [Model Manager Integration](./model_manager_integration.md)

## Support

For issues or questions:
- GitHub Issues: [ipfs_accelerate_py/issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
- Email: starworks5@gmail.com
