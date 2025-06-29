# Hardware-Aware Model Selection API Guide

**Date: March 7, 2025**  
**Version: 1.0.0**  
**Status: Active**

This guide provides detailed information about the Hardware-Aware Model Selection API, which exposes the hardware selection and performance prediction capabilities of the IPFS Accelerate framework through a RESTful API.

## Overview

The Hardware-Aware Model Selection API allows applications to:

1. **Select optimal hardware** for running models based on model characteristics, hardware availability, and benchmark data
2. **Predict performance metrics** for models on different hardware platforms
3. **Generate distributed training configurations** optimized for specific model-hardware combinations
4. **Analyze model performance** across hardware platforms with different batch sizes
5. **Create hardware selection maps** for multiple model families

The API leverages machine learning-based prediction models trained on benchmark data to provide accurate recommendations and performance predictions. It integrates with the existing hardware selection and prediction systems in the IPFS Accelerate framework.

## Installation

### API Server Requirements

To run the API server, the following dependencies are required:

```bash
pip install fastapi uvicorn duckdb pandas scikit-learn
```

### Client Library Requirements

To use the client library, only the `requests` package is required:

```bash
pip install requests
```

## Starting the API Server

### Using Command Line

The API server can be started using the command-line interface:

```bash
python hardware_selection_api.py --host 0.0.0.0 --port 8000 --database ./benchmark_db.duckdb
```

Command line options:
- `--host`: Host address to bind the server (default: 127.0.0.1)
- `--port`: Port to run the server on (default: 8000)
- `--database`: Path to the benchmark database
- `--benchmark-dir`: Directory containing benchmark results
- `--config`: Path to configuration file
- `--reload`: Enable auto-reload for development
- `--workers`: Number of worker processes
- `--debug`: Enable debug mode

### Using Docker

A Docker image is provided for easy deployment:

```bash
docker pull ipfs-accelerate/hardware-selection-api:latest
docker run -p 8000:8000 -v /path/to/benchmark_db.duckdb:/app/benchmark_db.duckdb ipfs-accelerate/hardware-selection-api:latest
```

### Environment Variables

The following environment variables can be used to configure the API:

- `BENCHMARK_DB_PATH`: Path to the benchmark database
- `BENCHMARK_DIR`: Directory containing benchmark results
- `CONFIG_PATH`: Path to configuration file
- `DEBUG`: Enable debug mode (1 or 0)

## API Endpoints

### System Endpoints

#### `GET /health`

Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "available_components": {
    "hardware_selection": true,
    "hardware_prediction": true,
    "duckdb": true,
    "fastapi": true
  },
  "database_connected": true
}
```

### Hardware Endpoints

#### `GET /hardware/detect`

Detect available hardware platforms.

**Response:**
```json
{
  "hardware": {
    "cpu": true,
    "cuda": true,
    "rocm": false,
    "mps": false,
    "openvino": false,
    "webnn": false,
    "webgpu": false
  }
}
```

#### `POST /hardware/select`

Select optimal hardware for a model.

**Request:**
```json
{
  "model_name": "bert-base-uncased",
  "model_family": "embedding",
  "batch_size": 16,
  "sequence_length": 128,
  "mode": "inference",
  "precision": "fp32",
  "available_hardware": ["cpu", "cuda"],
  "task_type": null,
  "distributed": false,
  "gpu_count": 1
}
```

**Response:**
```json
{
  "model_family": "embedding",
  "model_name": "bert-base-uncased",
  "model_size": 110000000,
  "model_size_category": "medium",
  "batch_size": 16,
  "sequence_length": 128,
  "precision": "fp32",
  "mode": "inference",
  "primary_recommendation": "cuda",
  "fallback_options": ["cpu"],
  "compatible_hardware": ["cuda", "cpu"],
  "explanation": "Selected based on benchmark data for embedding models",
  "prediction_source": "hardware_selector"
}
```

#### `POST /hardware/batch`

Select hardware for multiple models in a batch.

**Request:**
```json
{
  "models": [
    {"name": "bert-base-uncased", "family": "embedding"},
    {"name": "gpt2", "family": "text_generation"},
    {"name": "t5-small", "family": "text_generation"}
  ],
  "batch_size": 1,
  "mode": "inference"
}
```

**Response:**
```json
{
  "bert-base-uncased": {
    "primary": "cuda",
    "fallbacks": ["cpu"],
    "explanation": "Selected based on benchmark data for embedding models"
  },
  "gpt2": {
    "primary": "cuda",
    "fallbacks": ["cpu"],
    "explanation": "Selected based on benchmark data for text_generation models"
  },
  "t5-small": {
    "primary": "cuda",
    "fallbacks": ["cpu", "mps"],
    "explanation": "Selected based on benchmark data for text_generation models"
  }
}
```

#### `POST /hardware/map`

Create a hardware selection map for multiple model families.

**Request:**
```json
{
  "model_families": ["embedding", "text_generation", "vision"],
  "batch_sizes": [1, 8, 32],
  "hardware_platforms": ["cpu", "cuda", "mps"]
}
```

**Response:**
```json
{
  "timestamp": "2025-03-07T12:34:56.789Z",
  "model_families": {
    "embedding": {
      "model_sizes": {
        "small": {
          "inference": {
            "primary": "cuda",
            "fallbacks": ["cpu"]
          },
          "training": {
            "primary": "cuda",
            "fallbacks": ["cpu"]
          }
        },
        // ... more model sizes
      },
      "inference": {
        "batch_sizes": {
          "1": {
            "primary": "cuda",
            "fallbacks": ["cpu"]
          },
          // ... more batch sizes
        }
      },
      // ... more model families
    }
  }
}
```

### Performance Endpoints

#### `POST /performance/predict`

Predict performance for a model on specified hardware.

**Request:**
```json
{
  "model_name": "bert-base-uncased",
  "model_family": "embedding",
  "hardware": ["cuda", "cpu"],
  "batch_size": 16,
  "sequence_length": 128,
  "mode": "inference",
  "precision": "fp32"
}
```

**Response:**
```json
{
  "model_name": "bert-base-uncased",
  "model_family": "embedding",
  "batch_size": 16,
  "sequence_length": 128,
  "mode": "inference",
  "precision": "fp32",
  "predictions": {
    "cuda": {
      "throughput": 124.56,
      "latency": 8.25,
      "memory_usage": 1254.32,
      "source": "hardware_model_predictor"
    },
    "cpu": {
      "throughput": 32.45,
      "latency": 48.67,
      "memory_usage": 987.65,
      "source": "hardware_model_predictor"
    }
  }
}
```

### Training Endpoints

#### `POST /training/distributed`

Generate a distributed training configuration for a model.

**Request:**
```json
{
  "model_name": "gpt2",
  "model_family": "text_generation",
  "gpu_count": 8,
  "batch_size": 16,
  "max_memory_gb": 16
}
```

**Response:**
```json
{
  "model_family": "text_generation",
  "model_name": "gpt2",
  "distributed_strategy": "DDP",
  "gpu_count": 8,
  "per_gpu_batch_size": 16,
  "global_batch_size": 128,
  "mixed_precision": true,
  "gradient_accumulation_steps": 1,
  "estimated_memory": {
    "parameters_gb": 1.5,
    "activations_gb": 6.0,
    "optimizer_gb": 3.0,
    "total_gb": 10.5,
    "per_gpu_gb": 1.3125
  }
}
```

### Analysis Endpoints

#### `POST /analysis/model`

Analyze model performance across hardware platforms.

**Request:**
```json
{
  "model_name": "bert-base-uncased",
  "model_family": "embedding",
  "batch_sizes": [1, 8, 32]
}
```

**Response:**
```json
{
  "model_name": "bert-base-uncased",
  "model_family": "embedding",
  "hardware_platforms": ["cpu", "cuda", "mps"],
  "batch_sizes": [1, 8, 32],
  "timestamp": "2025-03-07T12:34:56.789Z",
  "inference": {
    "performance": {
      "1": {
        "cpu": {
          "throughput": 15.67,
          "latency": 63.81,
          "memory_usage": 456.78
        },
        // ... more hardware platforms
      },
      // ... more batch sizes
    },
    "recommendations": {
      "1": {
        "primary": "cuda",
        "fallbacks": ["cpu", "mps"]
      },
      // ... more batch sizes
    }
  },
  "training": {
    // ... similar structure to inference
  }
}
```

## Client Library Usage

The API includes a Python client library that makes it easy to integrate with applications.

### Basic Usage

```python
from hardware_selection_client import HardwareSelectionClient

# Initialize client
client = HardwareSelectionClient(base_url="http://localhost:8000")

# Check health
health = client.health()
print(f"API Status: {health['status']}")

# Detect hardware
hardware = client.detect_hardware()
print(f"Available Hardware: {hardware}")

# Select hardware for a model
recommendation = client.select_hardware(
    model_name="bert-base-uncased",
    batch_size=16,
    mode="inference"
)
print(f"Recommended Hardware: {recommendation['primary_recommendation']}")
```

### Getting PyTorch Device

The client library includes a convenience method to get PyTorch device strings:

```python
# Get the optimal PyTorch device
device = client.get_optimal_device("bert-base-uncased")
print(f"PyTorch Device: {device}")

# Use the device in PyTorch
import torch
model = torch.load("model.pt", map_location=device)
```

### Performance Prediction

```python
# Predict performance
performance = client.predict_performance(
    model_name="bert-base-uncased",
    hardware=["cuda", "cpu"],
    batch_size=16
)

# Print predictions
for hw, pred in performance["predictions"].items():
    print(f"{hw.upper()}:")
    print(f"  Throughput: {pred['throughput']:.2f} items/sec")
    print(f"  Latency: {pred['latency']:.2f} ms")
    print(f"  Memory Usage: {pred['memory_usage']:.2f} MB")
```

### Distributed Training Configuration

```python
# Get distributed training configuration
config = client.get_distributed_training_config(
    model_name="gpt2",
    gpu_count=8,
    batch_size=16,
    max_memory_gb=16
)

# Print configuration
print(f"Distributed Strategy: {config['distributed_strategy']}")
print(f"Total Batch Size: {config['global_batch_size']}")
```

### Batch Hardware Selection

```python
# Select hardware for multiple models
models = [
    {"name": "bert-base-uncased", "family": "embedding"},
    {"name": "gpt2", "family": "text_generation"},
    {"name": "t5-small", "family": "text_generation"}
]

results = client.select_hardware_for_models(models)

# Print results
for model_name, result in results.items():
    print(f"{model_name}: {result['primary']} (fallbacks: {', '.join(result['fallbacks'])})")
```

## CLI Mode

The Hardware-Aware Model Selection API can also be used in CLI mode without starting a server:

```bash
# Select hardware for a model
python hardware_selection_api.py --cli --action select --model bert-base-uncased --batch-size 16 --mode inference

# Predict performance
python hardware_selection_api.py --cli --action predict --model bert-base-uncased --hardware cpu,cuda --batch-size 16

# Analyze model performance
python hardware_selection_api.py --cli --action analyze --model bert-base-uncased

# Generate distributed training configuration
python hardware_selection_api.py --cli --action distributed --model gpt2 --batch-size 16

# Create hardware selection map
python hardware_selection_api.py --cli --action map --hardware cpu,cuda,mps

# Detect available hardware
python hardware_selection_api.py --cli --action detect
```

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Operation successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include a JSON body with error details:

```json
{
  "status": "error",
  "message": "Hardware selection failed for model bert-base-uncased"
}
```

The client library converts these errors into Python exceptions:

```python
try:
    result = client.select_hardware(model_name="invalid-model")
except ValueError as e:
    print(f"Error: {e}")
```

## Data Model

The API uses the following data models:

### Model Families

- `embedding`: BERT, RoBERTa, etc.
- `text_generation`: GPT, T5, LLAMA, etc.
- `vision`: ViT, ResNet, etc.
- `audio`: Whisper, Wav2Vec2, etc.
- `multimodal`: CLIP, LLaVA, etc.

### Hardware Types

- `cpu`: CPU execution
- `cuda`: NVIDIA GPU execution
- `rocm`: AMD GPU execution
- `mps`: Apple Silicon GPU execution
- `openvino`: Intel hardware acceleration
- `webnn`: Web browser neural network acceleration API
- `webgpu`: Web browser GPU acceleration API

### Modes

- `inference`: Model inference/prediction
- `training`: Model training/fine-tuning

### Precision Types

- `fp32`: 32-bit floating point
- `fp16`: 16-bit floating point
- `int8`: 8-bit integer quantization

## Integration Examples

### PyTorch Integration

```python
import torch
from hardware_selection_client import HardwareSelectionClient

# Initialize client
client = HardwareSelectionClient()

# Load model with optimal device
model_name = "bert-base-uncased"
device = client.get_optimal_device(model_name)

# Create model
from transformers import AutoModel
model = AutoModel.from_pretrained(model_name).to(device)

# Get performance prediction
perf = client.predict_performance(model_name=model_name, hardware=device)
print(f"Expected latency: {perf['predictions'][device]['latency']:.2f} ms")
```

### Distributed Training

```python
import torch
from hardware_selection_client import HardwareSelectionClient

# Initialize client
client = HardwareSelectionClient()

# Get distributed training configuration
model_name = "gpt2"
config = client.get_distributed_training_config(
    model_name=model_name,
    gpu_count=torch.cuda.device_count(),
    batch_size=8
)

# Setup distributed training based on configuration
if config["distributed_strategy"] == "DDP":
    # Initialize DDP
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel
    
    # ... DDP setup code
    
elif config["distributed_strategy"] == "FSDP":
    # Initialize FSDP
    from torch.distributed.fsdp import FullyShardedDataParallel
    
    # ... FSDP setup code
    
# Apply memory optimizations
if "gradient_checkpointing" in config and config["gradient_checkpointing"]:
    model.gradient_checkpointing_enable()
    
if "gradient_accumulation_steps" in config:
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    # Use in training loop
```

### Web Application

```javascript
// Frontend JavaScript
async function selectHardware() {
    const modelName = document.getElementById('modelName').value;
    const batchSize = parseInt(document.getElementById('batchSize').value);
    
    const response = await fetch('/api/hardware/select', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_name: modelName,
            batch_size: batchSize,
            mode: 'inference'
        })
    });
    
    const result = await response.json();
    
    document.getElementById('recommendation').textContent = 
        `Recommended Hardware: ${result.primary_recommendation}`;
    document.getElementById('fallbacks').textContent = 
        `Fallback Options: ${result.fallback_options.join(', ')}`;
}
```

## Conclusion

The Hardware-Aware Model Selection API provides a powerful way to integrate hardware selection and performance prediction capabilities into applications. By leveraging the IPFS Accelerate framework's machine learning-based prediction models, it can provide accurate recommendations for optimal hardware platforms based on model characteristics, hardware availability, and benchmark data.

Whether used through the RESTful API, client library, or CLI mode, the Hardware-Aware Model Selection API empowers developers to make informed hardware choices for model deployment and training.

## Additional Resources

- [HARDWARE_SELECTION_GUIDE.md](HARDWARE_SELECTION_GUIDE.md): Comprehensive guide to hardware selection
- [MODEL_FAMILY_GUIDE.md](MODEL_FAMILY_GUIDE.md): Detailed information about model families
- [HARDWARE_BENCHMARKING_GUIDE.md](HARDWARE_BENCHMARKING_GUIDE.md): Guide to hardware benchmarking
- [HARDWARE_MODEL_PREDICTOR_GUIDE.md](HARDWARE_MODEL_PREDICTOR_GUIDE.md): Guide to the hardware model predictor

--

*Last updated: March 7, 2025*