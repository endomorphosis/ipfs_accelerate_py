# IPFS Accelerate Python - Usage Guide

This guide covers the basic and advanced usage of the IPFS Accelerate Python framework for hardware-accelerated machine learning inference.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [MCP Server Usage](#mcp-server-usage)
- [P2P Workflow Scheduling](#p2p-workflow-scheduling)
- [CLI Tools](#cli-tools)
- [Configuration](#configuration)
- [Hardware Selection](#hardware-selection)
- [Model Types](#model-types)
- [API Backend Selection](#api-backend-selection)
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

## MCP Server Usage

The Model Context Protocol (MCP) server provides a standardized interface for AI model interaction.

### Starting the MCP Server

```bash
# Basic server start
ipfs-accelerate mcp start

# Start with dashboard
ipfs-accelerate mcp dashboard

# Start with custom port
ipfs-accelerate mcp start --port 8080

# Check server status
ipfs-accelerate mcp status
```

### Using MCP Tools from Python

```python
from mcp.server import create_mcp_server
import anyio

async def use_mcp_server():
    # Create MCP server
    mcp = create_mcp_server(
        resources={
            "models": ["bert-base-uncased", "gpt2"],
            "hardware": {"preferred": "cuda"}
        }
    )
    
    # Call MCP tool for inference
    result = await mcp.call_tool(
        "run_enhanced_inference",
        {
            "model": "bert-base-uncased",
            "input": "Hello, world!",
            "mode": "auto"
        }
    )
    
    print(result)

anyio.run(use_mcp_server)
```

### Available MCP Tools

The MCP server provides 14+ tools:

**Inference Tools:**
- `run_enhanced_inference` - Multi-backend inference with automatic routing
- `run_inference` - Standard inference

**Model Tools:**
- `search_models` - Search HuggingFace models
- `get_model_recommendations` - Get model recommendations for task/hardware
- `check_model_compatibility` - Check if model works on hardware

**Hardware Tools:**
- `detect_hardware` - Detect available hardware
- `test_hardware_capability` - Test hardware capabilities
- `get_optimal_configuration` - Get optimal config for model

**Workflow Tools:**
- `list_workflows` - List all workflows
- `get_workflow_status` - Get workflow status
- `submit_p2p_workflow` - Submit P2P workflow

**GitHub Tools:**
- `gh_list_repos` - List GitHub repositories
- `gh_workflow_runs` - List workflow runs
- `gh_provision_runner` - Provision self-hosted runner

For complete documentation, see [P2P_AND_MCP.md](P2P_AND_MCP.md#mcp-tools-reference).

## P2P Workflow Scheduling

Distribute tasks across peer-to-peer networks for scalable execution.

### Basic P2P Workflow

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import (
    P2PWorkflowScheduler, 
    WorkflowTag
)
import anyio

async def p2p_inference():
    # Create scheduler
    scheduler = P2PWorkflowScheduler(
        node_id="worker-01",
        config={
            "max_concurrent_tasks": 4,
            "peer_discovery_interval": 30
        }
    )
    
    # Start scheduler
    await scheduler.start()
    
    # Submit workflow
    workflow_id = await scheduler.submit_workflow({
        "name": "batch-inference",
        "tag": WorkflowTag.P2P_ELIGIBLE,
        "priority": 1,
        "tasks": [
            {"model": "bert-base", "input": "text1"},
            {"model": "bert-base", "input": "text2"},
            {"model": "bert-base", "input": "text3"}
        ]
    })
    
    # Monitor progress
    while True:
        status = await scheduler.get_workflow_status(workflow_id)
        print(f"Progress: {status['completed']}/{status['total']}")
        
        if status['completed'] == status['total']:
            break
        
        await anyio.sleep(1)
    
    print("Workflow completed!")

anyio.run(p2p_inference)
```

### Workflow Tags

Workflows can be tagged for different execution modes:

```python
from ipfs_accelerate_py.p2p_workflow_scheduler import WorkflowTag

# Standard GitHub API workflows
WorkflowTag.GITHUB_API

# Can execute via P2P or GitHub
WorkflowTag.P2P_ELIGIBLE

# Must execute via P2P (bypasses GitHub)
WorkflowTag.P2P_ONLY

# Specific task types
WorkflowTag.CODE_GENERATION
WorkflowTag.WEB_SCRAPING
WorkflowTag.DATA_PROCESSING
```

### Advanced P2P Features

```python
# Configure peer selection
scheduler = P2PWorkflowScheduler(
    node_id="worker-01",
    config={
        "enable_task_stealing": True,  # Allow tasks to move between peers
        "priority_boost_factor": 0.9,  # Boost starved tasks
        "heartbeat_timeout": 120       # Peer timeout (seconds)
    }
)

# Get network status
network_status = await scheduler.get_network_status()
print(f"Active peers: {network_status['peer_count']}")
print(f"Active tasks: {network_status['active_tasks']}")

# Rebalance tasks across network
await scheduler.rebalance_tasks(strategy="load")
```

For complete documentation, see [P2P_AND_MCP.md](P2P_AND_MCP.md#p2p-workflow-scheduler).

## CLI Tools

Comprehensive command-line interface for all operations.

### Inference Commands

```bash
# Run inference via CLI
ipfs-accelerate inference generate \
  --model bert-base-uncased \
  --input "Hello, world!"

# Run with specific backend
ipfs-accelerate inference generate \
  --model llama-2-7b \
  --backend vllm \
  --input "Once upon a time"

# Batch inference from file
ipfs-accelerate inference batch \
  --model gpt2 \
  --input-file inputs.txt \
  --output-file results.json
```

### Model Management

```bash
# List available models
ipfs-accelerate models list

# Search for models
ipfs-accelerate models search "sentiment analysis"

# Get model info
ipfs-accelerate models info bert-base-uncased

# Check compatibility
ipfs-accelerate models check-compat \
  --model llama-2-7b \
  --hardware cuda
```

### Hardware Detection

```bash
# Detect available hardware
ipfs-accelerate hardware detect

# Test hardware capabilities
ipfs-accelerate hardware test --type cuda

# Get optimal configuration
ipfs-accelerate hardware optimize \
  --model bert-base-uncased
```

### GitHub Integration

```bash
# Start GitHub autoscaler
ipfs-accelerate github autoscaler

# List repositories
ipfs-accelerate github repos --owner myorg

# Provision self-hosted runner
ipfs-accelerate github provision-runner \
  --repo myorg/myrepo \
  --labels self-hosted,gpu

# Check cache stats
ipfs-accelerate github cache-stats
```

### Network Operations

```bash
# Check IPFS network status
ipfs-accelerate network status

# Add file to IPFS
ipfs-accelerate files add myfile.txt

# Get file from IPFS
ipfs-accelerate files get QmHash... output.txt
```

For all CLI commands, run:
```bash
ipfs-accelerate --help
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

## API Backend Selection

IPFS Accelerate supports 14+ API backends for flexible inference routing.

### Supported Backends

```python
from ipfs_accelerate_py import ipfs_accelerate_py

accelerator = ipfs_accelerate_py({}, {})

# Available backends:
backends = [
    "local",       # Local model inference
    "vllm",        # vLLM (optimized for LLMs)
    "ollama",      # Ollama local models
    "openai",      # OpenAI API
    "anthropic",   # Claude API
    "groq",        # Groq API
    "gemini",      # Google Gemini
    "hf_tgi",      # HuggingFace Text Generation Inference
    "hf_tei",      # HuggingFace Text Embeddings Inference
    "opea",        # OPEA (Open Platform for Enterprise AI)
    "ovms",        # OpenVINO Model Server
    "s3",          # S3-compatible storage
    "llvm",        # LLVM-based compilation
    "akash",       # Akash Network
]
```

### Using Specific Backend

```python
# Use vLLM for optimized LLM inference
result = accelerator.process(
    model="meta-llama/Llama-2-7b-hf",
    input_data={"prompt": "Once upon a time"},
    endpoint_type="text_generation",
    backend="vllm",
    backend_config={
        "tensor_parallel_size": 2,
        "max_num_seqs": 256
    }
)

# Use OpenAI API
result = accelerator.process(
    model="gpt-4",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    endpoint_type="chat",
    backend="openai",
    backend_config={
        "api_key": "sk-...",
        "temperature": 0.7
    }
)

# Use Ollama for local models
result = accelerator.process(
    model="llama2:7b",
    input_data={"prompt": "Explain AI"},
    endpoint_type="text_generation",
    backend="ollama",
    backend_config={
        "num_ctx": 4096,
        "temperature": 0.8
    }
)
```

### Automatic Backend Selection

```python
# Let the framework choose the best backend
result = accelerator.process(
    model="bert-base-uncased",
    input_data={"input_ids": [101, 2054, 2003, 102]},
    endpoint_type="text_embedding",
    backend="auto"  # Automatically selects optimal backend
)
```

### Backend Configuration

```python
# Configure backends globally
config = {
    "backends": {
        "vllm": {
            "host": "localhost",
            "port": 8000,
            "timeout": 60
        },
        "openai": {
            "api_key": "sk-...",
            "org_id": "org-...",
            "base_url": "https://api.openai.com/v1"
        },
        "anthropic": {
            "api_key": "sk-ant-...",
            "version": "2023-06-01"
        },
        "ollama": {
            "host": "http://localhost:11434",
            "timeout": 120
        }
    }
}

accelerator = ipfs_accelerate_py(config, {})
```

### Container Backends

For cloud deployment:

```python
# Kubernetes backend
result = accelerator.process(
    model="bert-base",
    input_data={"text": "Hello"},
    backend="kubernetes",
    backend_config={
        "namespace": "ml-inference",
        "deployment": "bert-service",
        "replicas": 3
    }
)

# Akash Network (decentralized cloud)
result = accelerator.process(
    model="llama-2-7b",
    input_data={"prompt": "AI ethics"},
    backend="akash",
    backend_config={
        "provider": "akash1...",
        "region": "us-west"
    }
)

# HuggingFace Spaces
result = accelerator.process(
    model="my-custom-model",
    input_data={"text": "test"},
    backend="hf_spaces",
    backend_config={
        "space_id": "myorg/my-space",
        "hf_token": "hf_..."
    }
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