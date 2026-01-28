# Auto-Patching System for Transformers

## Overview

The `auto_patch_transformers` module provides automatic monkey-patching of HuggingFace transformers to integrate distributed filesystem support via `storage_wrapper` and `ipfs_kit_py`. This is inspired by the `ipfs_transformers_py` pattern but customized for `ipfs_accelerate_py`.

## Key Features

- **Automatic Patching**: No manual code changes required in worker skillsets
- **Distributed Storage Integration**: Automatically uses distributed cache when available
- **CI/CD Gating**: Automatically disables in CI/CD environments
- **Graceful Fallback**: Falls back to standard transformers if needed
- **Zero Breaking Changes**: Existing code continues to work unchanged

## How It Works

The auto-patching system monkey-patches the `from_pretrained` methods of HuggingFace transformers classes to automatically use distributed storage for model caching:

1. **Import Time**: When `ipfs_accelerate_py` is imported, the auto-patcher evaluates the environment
2. **Conditional Patching**: If patching is enabled (not in CI, not explicitly disabled), it patches transformers
3. **Cache Directory Override**: Patched methods use `storage_wrapper.get_cache_dir()` for `cache_dir` parameter
4. **Transparent Operation**: Worker skillsets use distributed storage without any code changes

## Usage

### Basic Usage

```python
# Simply import ipfs_accelerate_py
from ipfs_accelerate_py import auto_patch_transformers

# Patches are automatically applied on import (if environment allows)
# Now all transformers imports will use distributed storage

from transformers import AutoModel, AutoTokenizer

# These will automatically use distributed cache
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### Manual Control

```python
from ipfs_accelerate_py import auto_patch_transformers

# Check if patching is enabled
status = auto_patch_transformers.get_status()
print(status)  # {'enabled': True, 'applied': True, 'patched_classes': [...], ...}

# Manually apply patches (usually not needed)
auto_patch_transformers.apply()

# Restore original transformers behavior
auto_patch_transformers.restore()

# Or use disable() which is equivalent to restore()
auto_patch_transformers.disable()
```

### Worker Skillset Usage

Worker skillsets automatically benefit from patching with **zero code changes**:

```python
# worker/skillset/my_model.py

class MyModel:
    def init(self):
        import transformers
        self.transformers = transformers
    
    def load_model(self, model_name):
        # This automatically uses distributed storage!
        model = self.transformers.AutoModel.from_pretrained(model_name)
        return model
```

## Environment Variables

Control patching behavior via environment variables:

| Variable | Effect | Default |
|----------|--------|---------|
| `TRANSFORMERS_PATCH_DISABLE` | Explicitly disable patching | `0` (enabled) |
| `IPFS_KIT_DISABLE` | Disable IPFS features (disables patching) | `0` (enabled) |
| `STORAGE_FORCE_LOCAL` | Force local filesystem mode | `0` (distributed) |
| `CI` | Auto-detected CI environment | Auto-detect |

### Examples

```bash
# Disable patching explicitly
export TRANSFORMERS_PATCH_DISABLE=1
python my_script.py

# Disable in CI/CD (auto-detected)
# CI=1 is set by GitHub Actions, GitLab CI, etc.
python my_script.py  # Patching disabled automatically

# Force local filesystem mode
export STORAGE_FORCE_LOCAL=1
python my_script.py
```

## Patched Classes

The following HuggingFace classes are automatically patched:

- **Language Models**: `AutoModel`, `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`, `AutoModelForMaskedLM`
- **Sequence Tasks**: `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering`
- **Vision Models**: `AutoModelForImageClassification`, `AutoModelForObjectDetection`, `AutoModelForImageSegmentation`, etc.
- **Audio Models**: `AutoModelForAudioClassification`, `AutoModelForSpeechSeq2Seq`, `AutoModelForCTC`
- **Multimodal**: `AutoModelForVision2Seq`, `AutoModelForVisualQuestionAnswering`
- **Utilities**: `AutoTokenizer`, `AutoProcessor`, `AutoConfig`, `AutoFeatureExtractor`, `AutoImageProcessor`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Worker Skillset Code                      │
│  transformers.AutoModel.from_pretrained("model-name")       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Auto-Patched from_pretrained()                  │
│  1. Check if storage_wrapper available                      │
│  2. Get distributed cache_dir if enabled                    │
│  3. Override cache_dir parameter                            │
│  4. Call original from_pretrained()                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Storage Wrapper                            │
│  - Uses distributed storage if available                    │
│  - Falls back to local filesystem                           │
│  - Respects CI/CD gating                                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Distributed Cache / Local Filesystem            │
│  ~/.cache/ipfs_accelerate/ or distributed storage           │
└─────────────────────────────────────────────────────────────┘
```

## Benefits

### 1. No Code Changes Required

Previously, to use distributed storage, you would need to modify each skillset:

```python
# OLD WAY: Manual modification required
from ipfs_transformers_py import AutoModel  # Change import
model = AutoModel.from_ipfs("QmXXX...")     # Change method
```

Now, **no changes needed**:

```python
# NEW WAY: Automatic patching
from transformers import AutoModel  # Same import
model = AutoModel.from_pretrained("bert-base-uncased")  # Same method
# Distributed storage used automatically!
```

### 2. Centralized Control

- Single point of configuration in `auto_patch_transformers.py`
- Changes benefit all 27 worker skillsets automatically
- Easy to enable/disable globally

### 3. Backward Compatible

- Existing code works without modification
- Graceful fallback to standard transformers
- No breaking changes

### 4. CI/CD Friendly

- Automatically disables in CI environments
- No manual configuration needed for testing
- Respects all existing environment variables

## Comparison with ipfs_transformers_py

### ipfs_transformers_py (Original Pattern)

**Pros:**
- Mature package with IPFS support
- Well-tested

**Cons:**
- ❌ Requires changing imports in every file
- ❌ Different API (`from_ipfs` instead of `from_pretrained`)
- ❌ External dependency
- ❌ Not integrated with our storage_wrapper

### auto_patch_transformers (Our Solution)

**Pros:**
- ✅ No code changes required (automatic patching)
- ✅ Same API as standard transformers
- ✅ Integrated with storage_wrapper
- ✅ Customized for our needs
- ✅ CI/CD gating built-in

**Cons:**
- Monkey-patching has some risks
- Must maintain compatibility with transformers updates

## Troubleshooting

### Patching Not Applied

```python
from ipfs_accelerate_py import auto_patch_transformers

status = auto_patch_transformers.get_status()
print(status)

# Check if disabled by environment
print(f"Should patch: {auto_patch_transformers.should_patch()}")
```

**Common causes:**
- CI environment detected (`CI=1`)
- Explicitly disabled (`TRANSFORMERS_PATCH_DISABLE=1`)
- `transformers` not installed

### Verify Distributed Storage Usage

```python
from ipfs_accelerate_py.common.storage_wrapper import get_storage_wrapper

storage = get_storage_wrapper()
print(f"Distributed storage: {storage.is_distributed}")
print(f"Cache dir: {storage.get_cache_dir()}")
```

### Disable Patching for Testing

```bash
# Option 1: Environment variable
export TRANSFORMERS_PATCH_DISABLE=1

# Option 2: In code
from ipfs_accelerate_py import auto_patch_transformers
auto_patch_transformers.disable()
```

## Implementation Details

### Monkey-Patching Approach

The module uses Python's dynamic nature to replace methods at runtime:

```python
# Store original method
original_from_pretrained = transformers.AutoModel.from_pretrained

# Create wrapper
def patched_from_pretrained(model_name, *args, **kwargs):
    # Add distributed cache_dir
    if 'cache_dir' not in kwargs:
        kwargs['cache_dir'] = get_distributed_cache_dir()
    return original_from_pretrained(model_name, *args, **kwargs)

# Apply patch
transformers.AutoModel.from_pretrained = classmethod(patched_from_pretrained)
```

### Safety Measures

1. **Original Method Preservation**: Original methods are stored and can be restored
2. **Exception Handling**: Errors in patching don't crash the application
3. **Idempotent**: Multiple `apply()` calls are safe
4. **Reversible**: `restore()` undoes all patches

## Testing

Run the test suite:

```bash
python -m pytest test/test_auto_patch_transformers.py -v
```

Test manually:

```python
from ipfs_accelerate_py import auto_patch_transformers

# Test basic functionality
status = auto_patch_transformers.get_status()
print(f"Status: {status}")

# Test patching
if auto_patch_transformers.should_patch():
    auto_patch_transformers.apply()
    print("Patching applied")
else:
    print("Patching disabled by environment")
```

## Future Enhancements

Potential improvements:

1. **Performance Monitoring**: Track cache hits/misses
2. **Selective Patching**: Patch only specific model types
3. **Async Support**: Handle async model loading
4. **Model Registry**: Track which models are cached
5. **Cache Warming**: Pre-download popular models

## Related Documentation

- [FILESYSTEM_OPERATIONS_AUDIT.md](../FILESYSTEM_OPERATIONS_AUDIT.md) - Comprehensive audit of filesystem operations
- [INTEGRATION_STATUS.md](../INTEGRATION_STATUS.md) - Overall integration status
- [docs/IPFS_KIT_INTEGRATION.md](../docs/IPFS_KIT_INTEGRATION.md) - IPFS Kit integration guide
- [common/storage_wrapper.py](../ipfs_accelerate_py/common/storage_wrapper.py) - Storage wrapper implementation

## Conclusion

The auto-patching system provides a clean, centralized way to integrate distributed storage into all worker skillsets without requiring code changes in 27+ files. It's inspired by `ipfs_transformers_py` but customized for our specific needs with better CI/CD integration and automatic gating.
