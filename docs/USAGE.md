# IPFS Accelerate Python - Usage Guide

This guide covers the basic and advanced usage of the IPFS Accelerate Python framework for hardware-accelerated machine learning inference.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Configuration](#configuration)
- [Hardware Selection](#hardware-selection)
- [Model Types](#model-types)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Examples](#examples)

## Installation

### Basic Installation

```bash
pip install ipfs_accelerate_py
```

### Installation with Optional Features

```bash
# With WebNN/WebGPU support
pip install ipfs_accelerate_py[webnn]

# With visualization tools
pip install ipfs_accelerate_py[viz]

# Full installation with all dependencies
pip install ipfs_accelerate_py[all]
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/endomorphosis/ipfs_accelerate_py
cd ipfs_accelerate_py

# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"
```

## Quick Start

### Basic Usage

```python
from ipfs_accelerate_py import ipfs_accelerate_py

# Initialize the framework
accelerator = ipfs_accelerate_py({}, {})

# Run inference with automatic hardware selection
result = accelerator.process(
    model="bert-base-uncased",
    input_data={"input_ids": [101, 2054, 2003, 2026, 2171, 102]},
    endpoint_type="text_embedding"
)

print(result)
```

### Asynchronous Usage

```python
import anyio
from ipfs_accelerate_py import ipfs_accelerate_py

async def main():
    # Initialize the framework
    accelerator = ipfs_accelerate_py({}, {})
    
    # Run asynchronous inference
    result = await accelerator.process_async(
        model="bert-base-uncased",
        input_data={"input_ids": [101, 2054, 2003, 2026, 2171, 102]},
        endpoint_type="text_embedding"
    )
    
    print(result)

# Run the async example
anyio.run(main)
```

## Basic Usage

### Model Inference

The framework supports various model types with automatic hardware detection and optimization:

```python
from ipfs_accelerate_py import ipfs_accelerate_py

# Initialize
accelerator = ipfs_accelerate_py({}, {})

# Text embedding model
embedding_result = accelerator.process(
    model="bert-base-uncased",
    input_data={"input_ids": [101, 2023, 2003, 1037, 3231, 102]},
    endpoint_type="text_embedding"
)

# Text generation model
generation_result = accelerator.process(
    model="gpt2",
    input_data={"prompt": "The future of AI is"},
    endpoint_type="text_generation"
)

# Vision model
vision_result = accelerator.process(
    model="vit-base-patch16-224",
    input_data={"pixel_values": image_tensor},
    endpoint_type="vision"
)
```

### IPFS Accelerated Inference

Enable IPFS acceleration for distributed inference:

```python
import anyio

async def ipfs_inference():
    accelerator = ipfs_accelerate_py({}, {})
    
    # Enable IPFS acceleration
    result = await accelerator.accelerate_inference(
        model="bert-base-uncased",
        input_data={"input_ids": [101, 2054, 2003, 2026, 2171, 102]},
        use_ipfs=True
    )
    
    return result

result = anyio.run(ipfs_inference)
```

## Configuration

### Basic Configuration

```python
from ipfs_accelerate_py import ipfs_accelerate_py

# Configuration dictionary
config = {
    "ipfs": {
        "gateway": "http://localhost:8080/ipfs/",
        "local_node": "http://localhost:5001",
        "timeout": 30
    },
    "hardware": {
        "prefer_cuda": True,
        "allow_openvino": True,
        "precision": "fp16",
        "mixed_precision": True
    },
    "db_path": "benchmark_results.duckdb"
}

# Initialize with configuration
accelerator = ipfs_accelerate_py(config, {})
```

### Advanced Configuration

```python
config = {
    "ipfs": {
        "gateway": "https://ipfs.io/ipfs/",
        "local_node": "http://localhost:5001",
        "timeout": 60,
        "retry_count": 3,
        "enable_local_gateway": True
    },
    "hardware": {
        "prefer_cuda": True,
        "allow_openvino": True,
        "allow_mps": True,
        "allow_rocm": True,
        "allow_qualcomm": False,
        "precision": "fp16",
        "mixed_precision": True,
        "batch_size": 1,
        "max_memory": "8GB"
    },
    "performance": {
        "enable_caching": True,
        "cache_size": "1GB",
        "enable_prefetch": True,
        "parallel_requests": 4
    },
    "logging": {
        "level": "INFO",
        "enable_performance_logging": True,
        "log_file": "ipfs_accelerate.log"
    }
}
```

### Resources and Metadata

```python
# Define resources (model configurations, endpoints, etc.)
resources = {
    "models": {
        "bert-base-uncased": {
            "provider": "huggingface",
            "cache_dir": "./models/bert-base-uncased",
            "precision": "fp16"
        }
    },
    "endpoints": {
        "local": {
            "host": "localhost",
            "port": 8000
        }
    }
}

# Define metadata
metadata = {
    "project": "my-ml-project",
    "version": "1.0.0",
    "environment": "production"
}

# Initialize with resources and metadata
accelerator = ipfs_accelerate_py(resources, metadata)
```

## Hardware Selection

### Automatic Hardware Detection

The framework automatically detects and selects optimal hardware:

```python
accelerator = ipfs_accelerate_py({}, {})

# Hardware detection happens automatically during initialization
# Check what hardware was detected
print(accelerator.hardware_detection.detect_all_hardware())
```

### Manual Hardware Preference

```python
config = {
    "hardware": {
        "prefer_cuda": True,      # Prefer CUDA if available
        "allow_openvino": True,   # Allow Intel OpenVINO
        "allow_mps": True,        # Allow Apple Metal Performance Shaders
        "allow_rocm": False,      # Disable AMD ROCm
        "allow_qualcomm": False   # Disable Qualcomm acceleration
    }
}

accelerator = ipfs_accelerate_py(config, {})
```

## Model Types

### Text Models

```python
# Text embedding
embedding = accelerator.process(
    model="sentence-transformers/all-MiniLM-L6-v2",
    input_data={"input_ids": [101, 2023, 2003, 1037, 3231, 102]},
    endpoint_type="text_embedding"
)

# Text generation
generation = accelerator.process(
    model="gpt2",
    input_data={"prompt": "The benefits of renewable energy include"},
    endpoint_type="text_generation"
)

# Question answering
qa_result = accelerator.process(
    model="distilbert-base-uncased-distilled-squad",
    input_data={
        "question": "What is machine learning?",
        "context": "Machine learning is a subset of AI..."
    },
    endpoint_type="question_answering"
)
```

### Vision Models

```python
import torch

# Image classification
image_tensor = torch.randn(1, 3, 224, 224)  # Example image tensor
classification = accelerator.process(
    model="vit-base-patch16-224",
    input_data={"pixel_values": image_tensor},
    endpoint_type="vision"
)

# Object detection
detection = accelerator.process(
    model="detr-resnet-50",
    input_data={"pixel_values": image_tensor},
    endpoint_type="object_detection"
)
```

### Audio Models

```python
# Speech recognition
audio_data = torch.randn(1, 16000)  # Example audio tensor
transcription = accelerator.process(
    model="openai/whisper-small",
    input_data={"input_values": audio_data},
    endpoint_type="audio"
)
```

### Multimodal Models

```python
# Vision-language models
multimodal_result = accelerator.process(
    model="llava-1.5-7b-hf",
    input_data={
        "pixel_values": image_tensor,
        "input_ids": [101, 2054, 2003, 1999, 2023, 3746, 102]
    },
    endpoint_type="multimodal"
)
```

## Error Handling

### Basic Error Handling

```python
from ipfs_accelerate_py import ipfs_accelerate_py

try:
    accelerator = ipfs_accelerate_py({}, {})
    result = accelerator.process(
        model="invalid-model-name",
        input_data={"input_ids": [101, 102]},
        endpoint_type="text_embedding"
    )
except ValueError as e:
    print(f"Value error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Advanced Error Handling

```python
import anyio
import logging

async def robust_inference():
    accelerator = ipfs_accelerate_py({}, {})
    
    try:
        # Try IPFS acceleration first
        result = await accelerator.accelerate_inference(
            model="bert-base-uncased",
            input_data={"input_ids": [101, 102]},
            use_ipfs=True
        )
        return result
    except Exception as ipfs_error:
        logging.warning(f"IPFS acceleration failed: {ipfs_error}")
        
        try:
            # Fallback to local processing
            result = accelerator.process(
                model="bert-base-uncased",
                input_data={"input_ids": [101, 102]},
                endpoint_type="text_embedding"
            )
            return result
        except Exception as local_error:
            logging.error(f"Local processing also failed: {local_error}")
            raise
```

## Performance Optimization

### Batch Processing

```python
# Process multiple inputs efficiently
batch_inputs = [
    {"input_ids": [101, 2023, 2003, 1037, 3231, 102]},
    {"input_ids": [101, 2054, 2003, 2026, 2171, 102]},
    {"input_ids": [101, 7592, 2003, 2307, 102]}
]

results = []
for input_data in batch_inputs:
    result = accelerator.process(
        model="bert-base-uncased",
        input_data=input_data,
        endpoint_type="text_embedding"
    )
    results.append(result)
```

### Async Batch Processing

```python
import anyio

async def process_batch():
    accelerator = ipfs_accelerate_py({}, {})
    results = []

    async def run_one(input_data):
        result = await accelerator.process_async(
            model="bert-base-uncased",
            input_data=input_data,
            endpoint_type="text_embedding"
        )
        results.append(result)

    async with anyio.create_task_group() as tg:
        for input_data in batch_inputs:
            tg.start_soon(run_one, input_data)

    return results

batch_results = anyio.run(process_batch)
```

### Memory Optimization

```python
config = {
    "hardware": {
        "precision": "fp16",        # Use half precision
        "mixed_precision": True,    # Enable mixed precision
        "max_memory": "4GB"         # Limit memory usage
    },
    "performance": {
        "enable_caching": False,    # Disable caching to save memory
        "batch_size": 1             # Use smaller batch sizes
    }
}

accelerator = ipfs_accelerate_py(config, {})
```

## Examples

### Example 1: Text Similarity

```python
from ipfs_accelerate_py import ipfs_accelerate_py

def compute_similarity():
    accelerator = ipfs_accelerate_py({}, {})
    
    sentences = [
        "The cat sits on the mat",
        "A feline rests on a rug",
        "Dogs are loyal pets"
    ]
    
    embeddings = []
    for sentence in sentences:
        # Convert sentence to token IDs (simplified)
        tokens = [101] + [hash(word) % 30522 for word in sentence.split()] + [102]
        
        result = accelerator.process(
            model="bert-base-uncased",
            input_data={"input_ids": tokens[:10]},  # Limit to 10 tokens
            endpoint_type="text_embedding"
        )
        embeddings.append(result.get("embedding", []))
    
    return embeddings

similarities = compute_similarity()
```

### Example 2: Multi-Hardware Benchmarking

```python
import time

def benchmark_hardware():
    # Test different hardware configurations
    configs = [
        {"hardware": {"prefer_cuda": True}},
        {"hardware": {"allow_openvino": True, "prefer_cuda": False}},
        {"hardware": {"allow_mps": True, "prefer_cuda": False}}
    ]
    
    results = {}
    input_data = {"input_ids": [101, 2023, 2003, 1037, 3231, 102]}
    
    for i, config in enumerate(configs):
        accelerator = ipfs_accelerate_py(config, {})
        
        start_time = time.time()
        result = accelerator.process(
            model="bert-base-uncased",
            input_data=input_data,
            endpoint_type="text_embedding"
        )
        end_time = time.time()
        
        results[f"config_{i}"] = {
            "inference_time": end_time - start_time,
            "config": config,
            "success": result is not None
        }
    
    return results

benchmark_results = benchmark_hardware()
```

### Example 3: IPFS Content Distribution

```python
import anyio

async def distributed_inference():
    accelerator = ipfs_accelerate_py({
        "ipfs": {
            "gateway": "http://localhost:8080/ipfs/",
            "local_node": "http://localhost:5001"
        }
    }, {})
    
    # Find available providers for the model
    providers = await accelerator.find_providers("bert-base-uncased")
    print(f"Found {len(providers)} providers")
    
    # Run distributed inference
    result = await accelerator.accelerate_inference(
        model="bert-base-uncased",
        input_data={"input_ids": [101, 2054, 2003, 2026, 2171, 102]},
        use_ipfs=True
    )
    
    return result

distributed_result = anyio.run(distributed_inference)
```

For more examples, see the [examples directory](../examples/) and the [WebNN/WebGPU README](../WEBNN_WEBGPU_README.md).

## Next Steps

- [API Reference](API.md) - Detailed API documentation
- [Hardware Optimization](HARDWARE.md) - Hardware-specific optimization guide
- [IPFS Integration](IPFS.md) - Advanced IPFS usage patterns
- [Examples](../examples/README.md) - More practical examples