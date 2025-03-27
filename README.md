# IPFS Accelerate Python

A comprehensive framework for hardware-accelerated machine learning inference with IPFS network-based distribution and acceleration.

Author - Benjamin Barber  
QA - Kevin De Haan

## Overview

IPFS Accelerate Python provides a unified interface for running machine learning inference across various hardware platforms and leveraging the IPFS network for distributed inference when local resources are insufficient.

This is meant to be an extension of the Huggingface accelerate library, acting as a model server that can contain lists of other endpoints to call, call a local instance, and respond to external calls for inference. It includes modular backends such as Libp2p, Akash, Lilypad, Huggingface Zero, and Vast AI for autoscaling. If the model is already listed in the ipfs_model_manager, there should be an associated hw_requirements key in the manifest. For libp2p requests, inference will go to peers in the same trusted zone; if no peers are available and local resources are sufficient, it will run locally, otherwise a docker container will be launched with one of the providers.

## Directory Structure (Updated March 2025)

The codebase has been reorganized for better maintainability with the following top-level structure:

- **`generators/`**: Generation tools for tests, models, and benchmarks
  - `generators/benchmark_generators/`: Benchmark generation tools
  - `generators/models/`: Model implementations and skills
  - `generators/runners/`: Test runner scripts
  - `generators/skill_generators/`: Skill generation tools
  - `generators/template_generators/`: Template generation utilities
  - `generators/templates/`: Template files for model generation
  - `generators/test_generators/`: Test generation tools
  - `generators/utils/`: Utility functions
  - `generators/hardware/`: Hardware-specific generator tools

- **`duckdb_api/`**: Database functionality for storing and analyzing benchmark results
  - `duckdb_api/core/`: Core database functionality
  - `duckdb_api/migration/`: Migration tools for JSON to database
  - `duckdb_api/schema/`: Database schema definitions 
  - `duckdb_api/utils/`: Utility functions for database operations
  - `duckdb_api/visualization/`: Result visualization tools
  - `duckdb_api/distributed_testing/`: Distributed testing framework components

- **`fixed_web_platform/`**: Web platform implementations
  - `fixed_web_platform/unified_framework/`: Unified API for cross-browser WebNN/WebGPU
  - `fixed_web_platform/wgsl_shaders/`: WebGPU Shading Language shader implementations

- **`predictive_performance/`**: ML-based performance prediction system

Key features:

- **Hardware-Accelerated Inference**: Support for multiple hardware platforms (CPU, CUDA, ROCm, MPS, OpenVINO, Qualcomm, WebNN, WebGPU)
- **IPFS Network Acceleration**: Distribute inference workloads across the IPFS network
- **Automatic Hardware Selection**: Intelligently select the optimal hardware for each model
- **Model Family Classification**: Identify and optimize for different model families
- **Resource Management**: Efficient handling of model loading and memory usage
- **Cross-Platform Support**: Works across Linux, macOS, and Web Platforms
- **Template-Based Generation**: Generate optimized code for 300+ HuggingFace model types

## IPFS Huggingface Bridge

The IPFS Accelerate Python framework is part of a larger ecosystem of tools:

- Huggingface transformers python library: [ipfs_transformers](https://github.com/endomorphosis/ipfs_transformers/)
- Huggingface datasets python library: [ipfs_datasets](https://github.com/endomorphosis/ipfs_datasets/)
- Faiss KNN index python library: [ipfs_faiss](https://github.com/endomorphosis/ipfs_faiss)
- Transformers.js: [ipfs_transformers_js](https://github.com/endomorphosis/ipfs_transformers_js)
- Orbitdb_kit nodejs library: [orbitdb_kit](https://github.com/endomorphosis/orbitdb_kit/)
- Fireproof_kit nodejs library: [fireproof_kit](https://github.com/endomorphosis/fireproof_kit/)
- IPFS_kit nodejs library: [ipfs_kit](https://github.com/endomorphosis/ipfs_kit/)
- Python model manager library: [ipfs_model_manager](https://github.com/endomorphosis/ipfs_model_manager/)
- Node.js model manager library: [ipfs_model_manager_js](https://github.com/endomorphosis/ipfs_model_manager_js/)
- Node.js IPFS huggingface scraper: [ipfs_huggingface_scraper](https://github.com/endomorphosis/ipfs_huggingface_scraper/)
- IPFS agents: [ipfs_agents](https://github.com/endomorphosis/ipfs_agents/)
- IPFS accelerate: [ipfs_accelerate](https://github.com/endomorphosis/ipfs_accelerate/)

## Installation

```bash
# Clone the repository
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install requirements
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
from ipfs_accelerate_py import ipfs_accelerate_py

async def main():
    # Initialize the framework
    framework = ipfs_accelerate_py()
    
    # Initialize a model
    models = ["bert-base-uncased"]
    await framework.init_endpoints(models)
    
    # Run inference with automatic hardware selection
    result = await framework.process_async("bert-base-uncased", "This is a test sentence.")
    print(f"Result: {result}")
    
    # Use IPFS acceleration with automatic fallback
    result = await framework.accelerate_inference("bert-base-uncased", "This is a test sentence.")
    print(f"IPFS Accelerated Result: {result}")

# Run the example
asyncio.run(main())
```

For more examples, see `example.py`.

## Hardware Support

The framework supports multiple hardware platforms, with automatic detection and selection:

| Hardware Platform | Status | Notes |
|-------------------|--------|-------|
| CPU | ✅ | Always available |
| CUDA (NVIDIA) | ✅ | Automatically detected |
| AMD ROCm | ✅ | For AMD GPUs |
| Apple MPS | ✅ | For M1/M2/M3 Macs |
| OpenVINO | ✅ | For Intel hardware |
| Qualcomm AI Engine | ✅ | For Snapdragon devices |
| WebNN | ✅ | For web browsers |
| WebGPU | ✅ | For web browsers |

## IPFS Network Acceleration

When local hardware resources are insufficient, the framework can distribute inference workloads across the IPFS network:

```python
# Store a model weight to IPFS
cid = await framework.store_to_ipfs(model_weights)

# Query data from IPFS
data = await framework.query_ipfs(cid)

# Find providers for a specific model
providers = await framework.find_providers("gpt2")

# Connect to a provider
connected = await framework.connect_to_provider(providers[0])

# Run accelerated inference with automatic IPFS fallback
result = await framework.accelerate_inference(
    "gpt2-xl", 
    "This is a test prompt",
    use_ipfs=True
)
```

## API Documentation

### Core Classes

#### `ipfs_accelerate_py`

The main framework class that provides the unified interface for hardware-accelerated inference.

```python
# Initialize the framework
framework = ipfs_accelerate_py(resources=None, metadata=None)

# Initialize endpoints for models
await framework.init_endpoints(models, resources=None)

# Process input with automatic hardware selection
result = await framework.process_async(model, input_data, endpoint_type=None)

# Process input synchronously
result = framework.process(model, input_data, endpoint_type=None)

# Accelerated inference with IPFS fallback
result = await framework.accelerate_inference(model, input_data, use_ipfs=True)

# IPFS operations
cid = await framework.store_to_ipfs(data)
data = await framework.query_ipfs(cid)
providers = await framework.find_providers(model)
connected = await framework.connect_to_provider(provider_id)
```

## API Backends

### OpenVINO Model Server (OVMS) Backend
The OVMS backend provides integration with OpenVINO Model Server deployments. Features:
- Any OpenVINO-supported model type (classification, NLP, vision, speech)
- Both sync and async inference modes 
- Automatic input handling and tokenization
- Custom pre/post processing pipelines
- Batched inference support
- Multiple precision support (FP32, FP16, INT8)

Example usage:
```python
from ipfs_accelerate_py.api_backends import ovms

# Initialize backend
ovms_backend = ovms()

# For text/NLP models
endpoint_url, api_key, handler, queue, batch_size = ovms_backend.init(
    endpoint_url="http://localhost:9000",
    model_name="gpt2",
    context_length=1024
)

response = handler("What is quantum computing?")

# For vision models with custom preprocessing
def preprocess_image(image_data):
    # Convert image to model input format
    return processed_data

handler = ovms_backend.create_remote_ovms_endpoint_handler(
    endpoint_url="http://localhost:9000",
    model_name="resnet50",
    preprocessing=preprocess_image
)

result = handler(image_data, parameters={"raw": True})

# For async high-throughput inference
async_handler = await ovms_backend.create_async_ovms_endpoint_handler(
    endpoint_url="http://localhost:9000",
    model_name="bert-base"
)

results = await asyncio.gather(
    async_handler(batch1),
    async_handler(batch2)
)
```

### Other Backends

The framework supports multiple API backends including:

- OpenAI API
- Claude API
- Groq API
- Ollama
- Hugging Face TGI
- Hugging Face TEI
- Gemini API
- VLLM
- OVMS
- OPEA
- S3 Kit

## Advanced Features

### Hardware Detection

The framework automatically detects available hardware platforms and selects the optimal one for each model:

```python
# Get hardware detection results
hardware = framework.hardware_detection.detect_hardware()

# Check if CUDA is available
if hardware.get("cuda", {}).get("available", False):
    print("CUDA is available")
```

### Model Classification

Models are automatically classified by family to apply appropriate optimizations:

```python
# Classify a model by family
if hasattr(framework, "model_classifier"):
    family = framework.model_classifier.classify_model("bert-base-uncased")
    print(f"Model family: {family}")
```

### Resource Management

Efficient management of model loading and memory usage:

```python
# Get the resource pool
if framework.resource_pool is not None:
    # Use the resource pool for efficient model loading
    model = framework.resource_pool.get_model("bert-base-uncased", device="cuda")
```

## Architecture

The framework consists of several main components:

1. **Hardware Detection**: Identifies available hardware platforms
2. **Resource Pool**: Manages model loading and memory usage
3. **Model Family Classifier**: Identifies model types and families
4. **Endpoint Management**: Sets up and manages inference endpoints
5. **IPFS Integration**: Provides interaction with the IPFS network
6. **Template System**: Generates optimized code for models
7. **API Backends**: Integrations with various API providers

## Requirements

- Python 3.8+
- IPFS node (for distributed inference)
- Hardware-specific requirements:
  - CUDA: NVIDIA GPU + CUDA toolkit
  - ROCm: AMD GPU + ROCm toolkit
  - MPS: Apple M1/M2/M3 Mac
  - OpenVINO: Intel CPU/GPU/VPU
  - Qualcomm: Snapdragon device + AI SDK
  - WebNN/WebGPU: Modern web browser

## License

This project is licensed under the GNU Affero General Public License v3 (AGPL-3.0) - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
