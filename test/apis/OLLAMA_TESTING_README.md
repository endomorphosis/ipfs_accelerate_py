# Ollama API Testing Guide

This document outlines how to test the Ollama API backend implementation in the IPFS Accelerate framework.

## Overview

Ollama is a local API for running large language models on your own hardware. The test suite validates three main aspects:

1. **Standard API Implementation**: Verifies the core API functionality
2. **Performance Benchmarks**: Measures throughput and latency
3. **Real Connection**: Tests actual connection to a running Ollama server

## Test Files

- `test_ollama.py` - Core API implementation tests
- `test_ollama_unified.py` - Unified test runner with performance tests
- `OLLAMA_TESTING_README.md` - This documentation file

## Requirements

- Python 3.8+
- Ollama server (for real connection tests)
- Python packages: requests, numpy, unittest

## Running Tests

### Standard API Tests

Tests the implementation of the Ollama API client:

```bash
python scripts/generators/models/test_ollama_unified.py --standard
```

This validates:
- Endpoint handler creation
- API request formatting
- Response parsing
- Error handling
- Chat/completion methods
- Streaming support
- Model list retrieval

### Performance Tests

Benchmarks the performance of the API:

```bash
python scripts/generators/models/test_ollama_unified.py --performance
```

This measures:
- Single generation time
- Chat completion time
- Embedding generation time
- Batch processing efficiency
- Sentences per second throughput
- Batch speedup factor

### Real Connection Tests

Tests connection to an actual Ollama server:

```bash
python scripts/generators/models/test_ollama_unified.py --real
```

This verifies:
- Server availability
- Model availability
- Generation functionality

### All Tests

Run all test suites:

```bash
python scripts/generators/models/test_ollama_unified.py --all
```

## Configuration Options

You can customize the tests with these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model to use for testing | llama2 |
| `--api-url` | URL for Ollama API | http://localhost:11434/api |
| `--timeout` | Timeout in seconds for API requests | 30 |

Example with custom configuration:
```bash
python scripts/generators/models/test_ollama_unified.py --all --model mistral --api-url http://localhost:11434/api --timeout 60
```

## Environment Variables

You can also set configuration via environment variables:

- `OLLAMA_API_URL` - URL for the Ollama API
- `OLLAMA_MODEL` - Model to use for testing
- `OLLAMA_TIMEOUT` - Timeout in seconds
- `SKIP_PERFORMANCE_TESTS` - Set to "true" to skip performance tests
- `SKIP_REAL_TESTS` - Set to "true" to skip real connection tests

## Test Results

Test results are saved in the `collected_results/` directory:

- `ollama_test_results.json` - Standard API test results
- `ollama_performance_*.json` - Performance test results
- `ollama_connection_*.json` - Real connection test results
- `ollama_summary_*.json` - Test summary reports

## Extended API Features

The test suite checks for these advanced features if implemented:

### Embedding Support

Newer versions of Ollama support embeddings:

```python
# Generate embeddings for a single text
embeddings = ollama.generate_embeddings("llama2", "This is a test sentence.")

# Generate embeddings for multiple texts
batch_embeddings = ollama.batch_embeddings("llama2", ["Text 1", "Text 2", "Text 3"])
```

### Model Management

Methods for interacting with available models:

```python
# List all available models
models = ollama.list_models()

# Get information about a specific model
model_info = ollama.get_model_info("llama2")

# Pull a model from the registry
ollama.pull_model("llama2")
```

### Advanced Chat Options

Enhanced chat functionality:

```python
# Chat with system message
response = ollama.chat_with_system("llama2", messages, "You are a helpful assistant")

# Stream chat with custom parameters
for chunk in ollama.stream_chat_with_options("llama2", messages, {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 100
}):
    print(chunk)
```

## Running a Local Ollama Server

For real connection tests, you need an Ollama server:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Start the Ollama server
3. Pull a model: `ollama pull llama2`
4. The server should now be available at http://localhost:11434

## Troubleshooting

### Common Issues

- **Connection errors**: Verify Ollama is running with `curl http://localhost:11434/api/tags`
- **Model not found**: Pull the model with `ollama pull llama2`
- **Timeout errors**: Increase timeout with `--timeout` parameter
- **Memory issues**: Use a smaller model like `llama2:7b` or `mistral:7b`

### Debugging Commands

```bash
# Check if Ollama server is running
curl http://localhost:11434/api/tags

# Get information about a model
curl http://localhost:11434/api/show -d '{"name":"llama2"}'

# Test simple generation
curl http://localhost:11434/api/generate -d '{"model":"llama2","prompt":"Hello world"}'
```

## Development Notes

When implementing new features in the Ollama backend:

1. Add corresponding tests in `test_ollama.py`
2. Update the `test_ollama_unified.py` for performance testing
3. Document new features in this README

The test suite is designed to gracefully handle missing features, so it will not fail if a feature is not yet implemented.