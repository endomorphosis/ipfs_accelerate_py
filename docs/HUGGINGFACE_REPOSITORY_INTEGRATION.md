# HuggingFace Repository Structure Integration

The Model Manager now includes comprehensive support for storing and managing HuggingFace repository file structure and hash information. This feature allows you to track the exact files, their sizes, and Git hashes for any HuggingFace model repository.

## 🚀 Key Features

### 📁 Repository Structure Storage
- **Complete file listing**: Store all files in a HuggingFace repository
- **File metadata**: Track file sizes, Git hashes (OIDs), and download URLs
- **LFS information**: Handle Git LFS files with their SHA256 hashes
- **Branch tracking**: Support for different Git branches

### 🔍 Repository Queries
- **File hash lookup**: Get Git hash for specific files
- **File filtering**: Find models containing specific files or file types
- **Repository statistics**: Track total files and repository sizes
- **Cross-model search**: Find all models with similar file structures

## 📖 Usage Examples

### Basic Repository Structure Fetching

```python
from ipfs_accelerate_py.model_manager import (
    ModelManager, 
    create_model_from_huggingface,
    fetch_huggingface_repo_structure
)

# Create a model with repository structure
hf_config = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "vocab_size": 50257
}

# Automatically fetch repository structure when creating model
model = create_model_from_huggingface(
    model_id="gpt2",
    hf_config=hf_config,
    fetch_repo_structure=True,  # Enable repository fetching
    branch="main"  # Optional: specify branch (default: "main")
)

# Add to model manager
manager = ModelManager()
manager.add_model(model)

# Access repository information
if model.repository_structure:
    print(f"Total files: {model.repository_structure['total_files']}")
    print(f"Total size: {model.repository_structure['total_size']:,} bytes")
    print(f"Branch: {model.repository_structure['branch']}")
```

### Repository Queries and Analysis

```python
# Get file hash for specific file
config_hash = manager.get_model_file_hash("gpt2", "config.json")
print(f"config.json hash: {config_hash}")

# Find all models with config.json
models_with_config = manager.get_models_with_file("config.json")
print(f"Models with config.json: {len(models_with_config)}")

# Find all models with Python files
models_with_python = manager.get_models_with_file(".py")
print(f"Models with Python files: {len(models_with_python)}")

# Find all models with specific file pattern
models_with_pytorch = manager.get_models_with_file("pytorch_model")
print(f"Models with PyTorch files: {len(models_with_pytorch)}")
```

### Repository Structure Utilities

```python
from ipfs_accelerate_py.model_manager import (
    get_file_hash_from_structure,
    list_files_by_extension
)

# Get repository structure
repo_structure = fetch_huggingface_repo_structure("bert-base-uncased")

if repo_structure:
    # Get hash for specific file
    config_hash = get_file_hash_from_structure(repo_structure, "config.json")
    print(f"Config hash: {config_hash}")
    
    # List all JSON files
    json_files = list_files_by_extension(repo_structure, ".json")
    print(f"JSON files: {json_files}")
    
    # List all model files
    bin_files = list_files_by_extension(repo_structure, ".bin")
    safetensors_files = list_files_by_extension(repo_structure, ".safetensors")
    print(f"Model files: {bin_files + safetensors_files}")
```

### Repository Refresh and Updates

```python
# Refresh repository structure for a specific model
success = manager.refresh_repository_structure("gpt2", branch="main")
print(f"Refresh successful: {success}")

# The repository structure is automatically updated with latest file information
```

### Statistics and Analytics

```python
# Get comprehensive statistics
stats = manager.get_stats()

print(f"Models with repository structure: {stats['models_with_repo_structure']}")
print(f"Total tracked files across all models: {stats['total_tracked_files']:,}")

# Repository-specific analytics
if stats['models_with_repo_structure'] > 0:
    models_with_readme = manager.get_models_with_file("README")
    models_with_license = manager.get_models_with_file("LICENSE")
    models_with_requirements = manager.get_models_with_file("requirements")
    
    print(f"Models with README: {len(models_with_readme)}")
    print(f"Models with LICENSE: {len(models_with_license)}")
    print(f"Models with requirements: {len(models_with_requirements)}")
```

## 🔧 Configuration Options

### Automatic vs Manual Fetching

```python
# Automatic fetching (default enabled)
model = create_model_from_huggingface(
    model_id="distilbert-base-uncased",
    hf_config=config,
    fetch_repo_structure=True  # Default: True
)

# Manual fetching disabled (for offline use or performance)
model = create_model_from_huggingface(
    model_id="distilbert-base-uncased",
    hf_config=config,
    fetch_repo_structure=False
)

# Manual fetching with custom branch
model = create_model_from_huggingface(
    model_id="distilbert-base-uncased",
    hf_config=config,
    fetch_repo_structure=True,
    branch="development"  # Custom branch
)
```

### Direct Repository Structure Fetching

```python
# Fetch repository structure independently
repo_structure = fetch_huggingface_repo_structure("microsoft/DialoGPT-medium")

if repo_structure:
    print(f"Repository: {repo_structure['model_id']}")
    print(f"Branch: {repo_structure['branch']}")
    print(f"Files: {repo_structure['total_files']}")
    
    # Manually add to existing model
    existing_model = manager.get_model("my-existing-model")
    if existing_model:
        existing_model.repository_structure = repo_structure
        manager.add_model(existing_model)  # Update with new structure
```

## 📊 Repository Structure Format

The repository structure is stored as a JSON object with the following format:

```json
{
  "model_id": "gpt2",
  "branch": "main",
  "fetched_at": "2023-10-15T10:30:00.000Z",
  "files": {
    "config.json": {
      "size": 665,
      "lfs": {},
      "oid": "6e3c55a11b8e2e30a4fdbee5b1fb8e28c2c4b8f0",
      "download_url": "https://huggingface.co/gpt2/resolve/main/config.json"
    },
    "pytorch_model.bin": {
      "size": 503382240,
      "lfs": {
        "size": 503382240,
        "sha256": "7cb18dc9bafbfcf74629a4b760af1b160957a83e",
        "pointer_size": 135
      },
      "oid": "7cb18dc9bafbfcf74629a4b760af1b160957a83e",
      "download_url": "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin"
    }
  },
  "total_files": 8,
  "total_size": 505758687
}
```

## 🔒 Security and Privacy

- **No credentials required**: Uses public HuggingFace API endpoints
- **Read-only access**: Only fetches metadata, never modifies repositories
- **Graceful degradation**: Works offline when repository fetching is disabled
- **Error handling**: Handles network errors and API limitations gracefully

## 🚀 Performance Considerations

- **Caching**: Repository structures are cached and only refreshed when requested
- **Lazy loading**: Repository structure is only fetched when explicitly enabled
- **Batch processing**: Multiple models can be processed efficiently
- **Storage efficiency**: Compressed JSON storage for large repository structures

## 🛠️ Dependencies

### Required (Always Available)
- Core Python standard library
- `requests` library for HTTP API calls

### Optional Dependencies
- **Network access**: Required for fetching repository structures from HuggingFace
- **Internet connectivity**: Repository refresh requires network access

### Graceful Degradation
- Works fully offline when `fetch_repo_structure=False`
- Falls back gracefully when network is unavailable
- Preserves existing repository structures in storage

## 🧪 Testing

The repository structure functionality includes comprehensive tests:

```python
# Run repository structure tests
python -m unittest test_model_manager.TestRepositoryStructure -v

# Test specific functionality
python test_repo_structure_offline.py  # Offline tests with mock data
python test_repo_structure.py         # Online tests (requires internet)
```

## 🔄 Migration and Compatibility

### Existing Models
- Existing models without repository structure continue to work unchanged
- Repository structure can be added to existing models via `refresh_repository_structure()`
- Backward compatibility maintained for all existing functionality

### Storage Format
- Repository structure is stored in the `repository_structure` field of `ModelMetadata`
- JSON and DuckDB backends both support the new field
- Automatic schema migration for existing storage files

## 📝 Error Handling

The system includes comprehensive error handling:

```python
# Network errors are logged and handled gracefully
try:
    repo_structure = fetch_huggingface_repo_structure("invalid-model")
except Exception as e:
    print(f"Repository fetch failed: {e}")
    # System continues without repository structure

# Refresh failures don't affect existing functionality
success = manager.refresh_repository_structure("model-id")
if not success:
    print("Refresh failed, using cached repository structure")
```

This integration provides a complete solution for tracking and managing HuggingFace repository metadata alongside your model information, enabling more sophisticated model discovery and management workflows.