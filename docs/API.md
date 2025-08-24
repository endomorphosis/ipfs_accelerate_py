# IPFS Accelerate Python - API Reference

Complete API reference for the IPFS Accelerate Python framework.

## Table of Contents

- [Main Classes](#main-classes)
- [Core Methods](#core-methods)
- [Configuration Options](#configuration-options)
- [Data Types](#data-types)
- [Error Handling](#error-handling)
- [Utility Functions](#utility-functions)

## Main Classes

### ipfs_accelerate_py

The main class for the IPFS Accelerate Python framework.

```python
class ipfs_accelerate_py:
    def __init__(self, resources=None, metadata=None)
```

**Parameters:**
- `resources` (dict, optional): Dictionary containing model configurations, endpoints, and other resources
- `metadata` (dict, optional): Dictionary containing project metadata and configuration

**Example:**
```python
from ipfs_accelerate_py import ipfs_accelerate_py

# Basic initialization
accelerator = ipfs_accelerate_py({}, {})

# With configuration
config = {
    "ipfs": {"gateway": "http://localhost:8080/ipfs/"},
    "hardware": {"prefer_cuda": True}
}
accelerator = ipfs_accelerate_py(config, {})
```

## Core Methods

### process

Synchronous model inference with automatic hardware selection.

```python
def process(self, model: str, input_data: Any, endpoint_type: str = None) -> Any
```

**Parameters:**
- `model` (str): Model identifier (e.g., "bert-base-uncased", "gpt2")
- `input_data` (Any): Input data for the model (dict, tensor, etc.)
- `endpoint_type` (str, optional): Endpoint type hint for optimization

**Returns:**
- `Any`: Model inference result

**Example:**
```python
result = accelerator.process(
    model="bert-base-uncased",
    input_data={"input_ids": [101, 2054, 2003, 102]},
    endpoint_type="text_embedding"
)
```

### process_async

Asynchronous model inference with automatic hardware selection.

```python
async def process_async(self, model: str, input_data: Any, endpoint_type: str = None) -> Any
```

**Parameters:**
- `model` (str): Model identifier
- `input_data` (Any): Input data for the model
- `endpoint_type` (str, optional): Endpoint type hint for optimization

**Returns:**
- `Any`: Model inference result

**Example:**
```python
import asyncio

async def main():
    result = await accelerator.process_async(
        model="bert-base-uncased",
        input_data={"input_ids": [101, 2054, 2003, 102]},
        endpoint_type="text_embedding"
    )
    return result

result = asyncio.run(main())
```

### accelerate_inference

IPFS-accelerated inference with distributed processing.

```python
async def accelerate_inference(self, model: str, input_data: Any, use_ipfs: bool = True) -> Any
```

**Parameters:**
- `model` (str): Model identifier
- `input_data` (Any): Input data for the model
- `use_ipfs` (bool): Enable IPFS acceleration (default: True)

**Returns:**
- `Any`: Model inference result

**Example:**
```python
result = await accelerator.accelerate_inference(
    model="bert-base-uncased",
    input_data={"input_ids": [101, 2054, 2003, 102]},
    use_ipfs=True
)
```

### init_endpoints

Initialize model endpoints for specific models.

```python
async def init_endpoints(self, models: List[str], resources: Dict[str, Any] = None) -> Dict[str, Any]
```

**Parameters:**
- `models` (List[str]): List of model identifiers to initialize
- `resources` (Dict[str, Any], optional): Additional resources for endpoint initialization

**Returns:**
- `Dict[str, Any]`: Dictionary mapping models to their initialized endpoints

**Example:**
```python
models = ["bert-base-uncased", "gpt2", "vit-base-patch16-224"]
endpoints = await accelerator.init_endpoints(models)
```

### find_providers

Find IPFS providers for a specific model.

```python
async def find_providers(self, model: str) -> List[str]
```

**Parameters:**
- `model` (str): Model identifier

**Returns:**
- `List[str]`: List of provider IDs

**Example:**
```python
providers = await accelerator.find_providers("bert-base-uncased")
print(f"Found {len(providers)} providers")
```

### connect_to_provider

Connect to a specific IPFS provider.

```python
async def connect_to_provider(self, provider_id: str) -> bool
```

**Parameters:**
- `provider_id` (str): Provider identifier

**Returns:**
- `bool`: True if connection successful, False otherwise

**Example:**
```python
success = await accelerator.connect_to_provider("QmExampleProvider123")
```

### query_ipfs

Query data from IPFS using a Content Identifier (CID).

```python
async def query_ipfs(self, cid: str) -> bytes
```

**Parameters:**
- `cid` (str): IPFS Content Identifier

**Returns:**
- `bytes`: Raw data from IPFS

**Example:**
```python
data = await accelerator.query_ipfs("QmExampleCID123")
```

### store_to_ipfs

Store data to IPFS and get a Content Identifier (CID).

```python
async def store_to_ipfs(self, data: bytes) -> str
```

**Parameters:**
- `data` (bytes): Data to store in IPFS

**Returns:**
- `str`: IPFS Content Identifier (CID)

**Example:**
```python
import json
data = json.dumps({"model": "bert", "result": [0.1, 0.2, 0.3]}).encode()
cid = await accelerator.store_to_ipfs(data)
```

## Configuration Options

### IPFS Configuration

```python
ipfs_config = {
    "gateway": "http://localhost:8080/ipfs/",  # IPFS gateway URL
    "local_node": "http://localhost:5001",    # Local IPFS node API
    "timeout": 30,                            # Request timeout in seconds
    "retry_count": 3,                         # Number of retries for failed requests
    "enable_local_gateway": True              # Use local IPFS gateway if available
}
```

### Hardware Configuration

```python
hardware_config = {
    "prefer_cuda": True,          # Prefer CUDA acceleration
    "allow_openvino": True,       # Allow Intel OpenVINO
    "allow_mps": True,            # Allow Apple Metal Performance Shaders
    "allow_rocm": True,           # Allow AMD ROCm
    "allow_qualcomm": False,      # Allow Qualcomm acceleration
    "precision": "fp16",          # Model precision ("fp32", "fp16", "int8")
    "mixed_precision": True,      # Enable mixed precision
    "batch_size": 1,              # Default batch size
    "max_memory": "8GB"           # Maximum memory usage
}
```

### Performance Configuration

```python
performance_config = {
    "enable_caching": True,       # Enable result caching
    "cache_size": "1GB",          # Maximum cache size
    "enable_prefetch": True,      # Enable model prefetching
    "parallel_requests": 4        # Number of parallel requests
}
```

### Logging Configuration

```python
logging_config = {
    "level": "INFO",                        # Logging level
    "enable_performance_logging": True,     # Log performance metrics
    "log_file": "ipfs_accelerate.log"      # Log file path
}
```

## Data Types

### Input Data Types

The framework accepts various input data formats depending on the model type:

#### Text Models
```python
# Token IDs
text_input = {"input_ids": [101, 2054, 2003, 102]}

# With attention mask
text_input_full = {
    "input_ids": [101, 2054, 2003, 102],
    "attention_mask": [1, 1, 1, 1]
}

# For text generation
generation_input = {"prompt": "The future of AI is"}
```

#### Vision Models
```python
import torch

# Image tensor (batch_size, channels, height, width)
vision_input = {"pixel_values": torch.randn(1, 3, 224, 224)}

# With additional metadata
vision_input_full = {
    "pixel_values": torch.randn(1, 3, 224, 224),
    "image_size": (224, 224)
}
```

#### Audio Models
```python
# Audio tensor (batch_size, sequence_length)
audio_input = {"input_values": torch.randn(1, 16000)}

# With sampling rate
audio_input_full = {
    "input_values": torch.randn(1, 16000),
    "sampling_rate": 16000
}
```

#### Multimodal Models
```python
# Combined vision and text
multimodal_input = {
    "pixel_values": torch.randn(1, 3, 224, 224),
    "input_ids": [101, 2054, 2003, 1999, 2023, 3746, 102]
}
```

### Return Types

The framework returns different data structures based on the model and task:

#### Text Embedding
```python
{
    "embedding": [0.1, 0.2, -0.3, ...],  # List of floats
    "model": "bert-base-uncased",
    "inference_time": 0.045
}
```

#### Text Generation
```python
{
    "text": "The future of AI is bright and promising...",
    "model": "gpt2",
    "inference_time": 0.123
}
```

#### Image Classification
```python
{
    "label": "Egyptian cat",
    "score": 0.97,
    "logits": [2.1, -0.5, 3.2, ...],
    "model": "vit-base-patch16-224",
    "inference_time": 0.078
}
```

#### Audio Transcription
```python
{
    "text": "Hello world, this is a test recording",
    "model": "openai/whisper-small",
    "confidence": 0.94,
    "inference_time": 0.234
}
```

## Error Handling

### Exception Types

#### ValueError
Raised for invalid input parameters or model configurations.

```python
try:
    result = accelerator.process(
        model="invalid-model-name",
        input_data={},
        endpoint_type="text_embedding"
    )
except ValueError as e:
    print(f"Invalid input: {e}")
```

#### ConnectionError
Raised for IPFS connection failures.

```python
try:
    result = await accelerator.accelerate_inference(
        model="bert-base-uncased",
        input_data={"input_ids": [101, 102]},
        use_ipfs=True
    )
except ConnectionError as e:
    print(f"IPFS connection failed: {e}")
```

#### RuntimeError
Raised for hardware or model loading errors.

```python
try:
    accelerator = ipfs_accelerate_py({
        "hardware": {"prefer_cuda": True}
    }, {})
except RuntimeError as e:
    print(f"Hardware initialization failed: {e}")
```

### Error Recovery

```python
async def robust_inference(model, input_data):
    accelerator = ipfs_accelerate_py({}, {})
    
    try:
        # Try IPFS acceleration first
        return await accelerator.accelerate_inference(
            model=model,
            input_data=input_data,
            use_ipfs=True
        )
    except (ConnectionError, TimeoutError):
        # Fallback to local processing
        return accelerator.process(
            model=model,
            input_data=input_data,
            endpoint_type="auto"
        )
```

## Utility Functions

### initialize()

Create a new framework instance with default settings.

```python
def initialize() -> ipfs_accelerate_py
```

**Returns:**
- `ipfs_accelerate_py`: New framework instance

**Example:**
```python
from ipfs_accelerate_py import initialize

accelerator = initialize()
```

### get_instance()

Get or create the global framework instance (singleton pattern).

```python
def get_instance() -> ipfs_accelerate_py
```

**Returns:**
- `ipfs_accelerate_py`: Global framework instance

**Example:**
```python
from ipfs_accelerate_py import get_instance

# Multiple calls return the same instance
accelerator1 = get_instance()
accelerator2 = get_instance()
assert accelerator1 is accelerator2
```

## Hardware Detection API

The framework includes automatic hardware detection capabilities:

```python
# Access hardware detection directly
hardware_info = accelerator.hardware_detection.detect_all_hardware()

# Example output:
{
    "cpu": {"available": True, "cores": 8},
    "cuda": {"available": True, "devices": 1, "memory": "8GB"},
    "openvino": {"available": True, "version": "2023.1"},
    "mps": {"available": False},
    "rocm": {"available": False},
    "qualcomm": {"available": False},
    "webnn": {"available": True},
    "webgpu": {"available": True}
}
```

## Model Support

### Supported Model Families

The framework supports 300+ HuggingFace model types, including:

- **Text Models**: BERT, GPT, T5, RoBERTa, DistilBERT, ALBERT, etc.
- **Vision Models**: ViT, ResNet, EfficientNet, CLIP, DETR, etc.
- **Audio Models**: Whisper, Wav2Vec2, WavLM, etc.
- **Multimodal Models**: CLIP, BLIP, LLaVA, etc.

### Model Identifier Formats

```python
# HuggingFace Hub models
"bert-base-uncased"
"openai/whisper-small"
"microsoft/vit-base-patch16-224"

# Local models (if supported)
"./local_models/my_bert_model"

# Custom model identifiers
"custom:my_model_v1"
```

## Advanced Usage

### Custom Endpoint Configuration

```python
resources = {
    "endpoints": {
        "local": {
            "host": "localhost",
            "port": 8000,
            "protocol": "http"
        },
        "remote": {
            "host": "my-inference-server.com",
            "port": 443,
            "protocol": "https",
            "api_key": "your-api-key"
        }
    }
}

accelerator = ipfs_accelerate_py(resources, {})
```

### Model-Specific Configuration

```python
resources = {
    "models": {
        "bert-base-uncased": {
            "provider": "huggingface",
            "cache_dir": "./models/bert-base-uncased",
            "precision": "fp16",
            "batch_size": 32
        },
        "gpt2": {
            "provider": "huggingface",
            "cache_dir": "./models/gpt2",
            "precision": "fp32",
            "max_length": 100
        }
    }
}

accelerator = ipfs_accelerate_py(resources, {})
```

### Performance Monitoring

```python
import time

# Track inference performance
start_time = time.time()
result = accelerator.process(
    model="bert-base-uncased",
    input_data={"input_ids": [101, 2054, 2003, 102]},
    endpoint_type="text_embedding"
)
end_time = time.time()

print(f"Inference time: {end_time - start_time:.3f} seconds")
print(f"Result: {result}")
```

For more detailed examples and advanced usage patterns, see the [Usage Guide](USAGE.md) and [examples directory](../examples/).

## Related Documentation

- [Usage Guide](USAGE.md) - Comprehensive usage examples
- [Hardware Optimization](HARDWARE.md) - Hardware-specific optimization
- [IPFS Integration](IPFS.md) - Advanced IPFS features
- [WebNN/WebGPU README](../WEBNN_WEBGPU_README.md) - Browser acceleration