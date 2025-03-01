# HuggingFace Text Generation Inference (TGI) Testing Guide

This directory contains comprehensive tests for the HuggingFace Text Generation Inference (TGI) API backend. The tests are designed to validate both the standard HuggingFace TGI API integration and container-based deployment using the HuggingFace TGI Docker container.

## Overview

HuggingFace Text Generation Inference (TGI) is a high-performance solution for deploying Large Language Models (LLMs) for text generation tasks. TGI offers two main deployment options:

1. **Hosted API**: Using the HuggingFace Inference API at `https://api-inference.huggingface.co`
2. **Self-hosted Container**: Running a Docker container with TGI for local deployment

These tests validate that our implementation can successfully interact with both deployment options and handle various text generation scenarios including streaming responses.

## Test Files

- `test_hf_tgi.py` - Tests the standard HuggingFace TGI API integration
- `test_hf_tgi_container.py` - Tests TGI container deployment and interaction
- `test_hf_tgi_unified.py` - Unified test suite that can run both standard and container tests

## Model Selection

### Default Model: google/t5-efficient-tiny

These tests use `google/t5-efficient-tiny` as the default model, which offers several advantages:

- **Very small size**: Only ~60MB, making it ideal for quick tests
- **Fast loading time**: Loads in seconds even on CPU-only systems
- **Efficient inference**: 98 tokens/sec on CUDA, 32 tokens/sec on CPU
- **Open access**: No authentication token required
- **Good text generation quality**: Despite its small size
- **Low resource requirements**: Works on systems with limited GPU memory
- **TGI compatibility**: Fully compatible with Text Generation Inference

This model is an optimized version of T5 specifically designed for efficiency while maintaining reasonable performance.

### Alternative Small Models

If you need to test with different model architectures while keeping resource usage low, here are other recommended small models:

| Model | Size | Architecture | Notes |
|-------|------|--------------|-------|
| facebook/opt-125m | ~250MB | OPT (GPT-like) | Good general text generation |
| EleutherAI/gpt-neo-125M | ~350MB | GPT-Neo | Good code and text generation |
| huggingface/distilgpt2 | ~330MB | GPT-2 (distilled) | Faster than standard GPT-2 |
| bigscience/bloom-560m | ~460MB | BLOOM | Multilingual capabilities |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | ~450MB | LLaMA | Optimized for chat |

All these models are fully open access (no token required) and compatible with TGI.

## Prerequisites

### System Requirements

1. **Python Environment**:
   - Python 3.8 or higher
   - Required packages: `requests`, `unittest`, `json`, `urllib3`
   - Optional: `tqdm` for progress reporting

2. **Docker Requirements** (for container tests):
   - Docker Engine 20.10.0 or newer
   - Recommended: 4GB+ available RAM
   - For GPU support: NVIDIA Container Toolkit installed
   - Minimum 10GB disk space for container images and models

3. **Network Requirements**:
   - Outbound internet access to huggingface.co
   - For container tests: available port (default: 8080)
   - No firewall blocking Docker container networking

### Environment Variables

| Variable | Purpose | Default | Required? |
|----------|---------|---------|-----------|
| `HF_API_KEY` | Your HuggingFace API key for authentication | None | Only for private models |
| `HF_MODEL_ID` | Model ID to use for testing | google/t5-efficient-tiny | No |
| `HF_CONTAINER_URL` | URL for TGI container | http://localhost:8080 | No |
| `DOCKER_REGISTRY` | Docker registry for TGI container | huggingface/text-generation-inference | No |
| `CONTAINER_TAG` | Container tag to use | latest | No |
| `GPU_DEVICE` | GPU device ID to use | 0 | No |
| `SKIP_CONTAINER_TESTS` | Set to "true" to skip container tests | false | No |

You can set these environment variables before running the tests or pass them as command-line arguments to the unified test runner.

## Running the Tests

### Standard API Tests

The standard API tests validate our integration with the HuggingFace Inference API. These tests verify:
- Endpoint handler creation
- Authentication with API keys
- Parameter handling (max_new_tokens, temperature, top_p)
- Error handling for HTTP status codes
- Streaming capability

```bash
# Run standard API tests with default model (google/t5-efficient-tiny)
python test_hf_tgi.py

# Run with a specific model
HF_MODEL_ID="facebook/opt-125m" python test_hf_tgi.py  # Another small model ~250MB

# Run with your HuggingFace API key
HF_API_KEY="your_api_key_here" python test_hf_tgi.py
```

### Container Tests

The container tests validate our integration with a self-hosted TGI container. These tests verify:
- Container deployment and configuration
- Container health monitoring
- Text generation capabilities
- Streaming text generation
- Parameter handling
- Container shutdown and cleanup

```bash
# Run container tests with default model (google/t5-efficient-tiny)
python test_hf_tgi_container.py

# Run with a specific model (still small)
HF_MODEL_ID="facebook/opt-125m" python test_hf_tgi_container.py  # ~250MB model

# Specify custom container URL (if container is already running)
HF_CONTAINER_URL="http://localhost:8081" python test_hf_tgi_container.py

# Specify GPU device for container deployment
GPU_DEVICE="1" python test_hf_tgi_container.py

# Use a specific container version
CONTAINER_TAG="1.1.0" python test_hf_tgi_container.py
```

### Unified Tests

The unified test runner combines both standard API and container tests in a unittest framework, allowing for more structured test execution and reporting.

```bash
# Run both standard and container tests (recommended)
python test_hf_tgi_unified.py

# Run only standard API tests
python test_hf_tgi_unified.py --standard

# Run only container tests
python test_hf_tgi_unified.py --container

# Run with custom parameters
python test_hf_tgi_unified.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --container-url "http://localhost:8081" --api-key "your_api_key_here"
```

### Automated CI/CD Testing

For continuous integration environments, the following command will run all tests with appropriate logging:

```bash
# Run all tests with detailed logging and exit code
python test_hf_tgi_unified.py --standard --container | tee tgi_test_results.log; exit ${PIPESTATUS[0]}
```

## Implementation Details

### Container Deployment

The container tests simulate deployment of a TGI container with the following configuration:

```bash
docker run -d \
  --name "tgi-${MODEL_ID}" \
  -p 8080:80 \
  --gpus "device=${GPU_DEVICE}" \
  --shm-size 1g \
  -e MODEL_ID=${MODEL_ID} \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_LENGTH=1024 \
  -e MAX_TOTAL_TOKENS=2048 \
  -e TRUST_REMOTE_CODE=true \
  huggingface/text-generation-inference:latest
```

### TGI Container API Endpoints

The TGI container exposes several REST endpoints:

| Endpoint | Method | Purpose | Parameters |
|----------|--------|---------|------------|
| `/health` | GET | Container health check | None |
| `/info` | GET | Model information | None |
| `/generate` | POST | Text generation | inputs, parameters |
| `/generate_stream` | POST | Streaming text generation | inputs, parameters, stream=true |

### Request Format

Requests to the TGI container follow this structure:

```json
{
  "inputs": "Your prompt text here",
  "parameters": {
    "max_new_tokens": 20,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": true,
    "return_full_text": false
  }
}
```

For T5 models, the input is formatted with a prefix like `"summarize: Your text here"`.

### Response Format

Standard responses follow this structure:

```json
{
  "generated_text": "The generated response text"
}
```

Streaming responses return chunks with incremental text:

```json
{"generated_text": "The"}
{"generated_text": "The generated"}
{"generated_text": "The generated response"}
...
```

## Test Results Structure

Test results are saved to `collected_results/` directory in JSON format:

- `hf_tgi_test_results.json` - Results from standard API tests
- `hf_tgi_container_test_results.json` - Results from container tests

Expected results are stored in `expected_results/` and used for comparison.

### Sample Test Result

```json
{
  "endpoint_handler_creation": "Success",
  "test_endpoint": "Success",
  "test_endpoint_params": "Success",
  "post_request": "Success",
  "post_request_headers": "Success",
  "format_request_simple": "Success",
  "format_request_params": "Success",
  "parameter_formatting": "Success",
  "streaming": "Success",
  "error_handling_auth": "Success",
  "error_handling_404": "Success",
  "internal_test": "Success"
}
```

## Running in Production

### Optimized Configuration for Small Models

For small models like Google's T5-efficient-tiny (~60MB), you can use a more lightweight container configuration:

```bash
docker run -d \
  --name "tgi-t5-tiny" \
  -p 8080:80 \
  --cpus 2 \  # CPU-only deployment is sufficient for small models
  --memory 4g \
  --shm-size 1g \
  -e MODEL_ID="google/t5-efficient-tiny" \
  -e NUM_SHARD=1 \
  -e MAX_BATCH_SIZE=32 \
  -e MAX_INPUT_LENGTH=1024 \
  -e MAX_TOTAL_TOKENS=2048 \
  -e TRUST_REMOTE_CODE=true \
  huggingface/text-generation-inference:latest
```

This configuration can run efficiently even on systems without GPUs, making it ideal for development and testing.

### Configuration for Larger Models

For larger production models, you can use the following configuration:

```bash
docker run -d \
  --name "tgi-model" \
  -p 8080:80 \
  --gpus all \
  --shm-size 1g \
  -e MODEL_ID="your-model-id" \
  -e NUM_SHARD=2 \  # Increase for multi-GPU setups
  -e MAX_BATCH_SIZE=32 \  # Tune based on your hardware
  -e MAX_INPUT_LENGTH=4096 \
  -e MAX_TOTAL_TOKENS=8192 \
  -e TRUST_REMOTE_CODE=true \
  -e QUANTIZE="bitsandbytes" \  # Enable 8-bit quantization
  huggingface/text-generation-inference:latest
```

## Troubleshooting Guide

### Common Issues

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| Container fails to start | Docker not running, insufficient permissions | Check Docker logs with `docker logs <container_id>` |
| GPU not detected | Missing drivers, incorrect configuration | Ensure NVIDIA drivers and docker-nvidia are properly installed |
| API connectivity issues | Firewall, incorrect port mapping | Verify the container port mapping and network configuration |
| Model loading issues | Access restrictions, insufficient disk space | Check model access rights and free up disk space |
| Memory errors | Model too large for available RAM | Increase container memory or use a smaller model |
| Authentication failures | Missing or invalid API key | Check the HF_API_KEY environment variable |
| Slow container startup | First-time model download | Be patient on first run; subsequent runs will be faster |
| Container crashes during inference | Insufficient shared memory | Increase --shm-size parameter (e.g., to 2g or 4g) |

### Debugging Commands

```bash
# Check Docker container status
docker ps -a

# View container logs
docker logs tgi-google-t5-efficient-tiny

# Check GPU availability for Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Verify network connectivity to container
curl http://localhost:8080/health

# Stop a stuck container
docker stop tgi-google-t5-efficient-tiny

# Remove a container to start fresh
docker rm tgi-google-t5-efficient-tiny

# Test API health
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs":"summarize: hello world","parameters":{"max_new_tokens":5}}'
```

### Logging and Diagnostics

You can enable verbose logging in the tests by setting the following environment variable:

```bash
export HF_TGI_DEBUG=1
```

This will output detailed information about:
- Request payloads
- Response data
- Container operations
- Error messages with stack traces

## Compatibility and Performance Notes

- The TGI container works best with CUDA 11.8+ and cuDNN 8.6+
- For models >2GB, using GPU acceleration is strongly recommended
- Streaming performance is generally better for interactive applications
- First request to a container is slower due to optimization compilation
- Container cold start time scales with model size