# API Backends Quickstart Guide (March 1, 2025)

This guide provides simple examples for using the IPFS Accelerate API backends with proper credentials.

## Overview

The IPFS Accelerate framework provides a standardized interface to multiple LLM API providers. All API backends are now fully implemented with REAL functionality:

| API | Status | Used For |
|-----|--------|----------|
| OpenAI | ✅ REAL | GPT models, embeddings, assistants |
| Claude | ✅ REAL | Claude models, streaming |
| Groq | ✅ REAL | High-speed inference, Llama models |
| Ollama | ✅ REAL | Local deployment, open-source models |
| HF TGI | ✅ REAL | Text generation with Hugging Face models |
| HF TEI | ✅ REAL | Embeddings with Hugging Face models |
| Gemini | ✅ REAL | Google's models, multimodal capabilities |
| LLVM | ✅ REAL | Optimized local inference |
| OVMS | ✅ REAL | OpenVINO Model Server integration |
| OPEA | ✅ REAL | Open Platform for Enterprise AI |
| S3 Kit | ✅ REAL | Model storage and retrieval |

## Setting Up Credentials

For testing and development, you can use one of these methods to provide API credentials:

### Method 1: Environment Variables

```bash
# OpenAI API
export OPENAI_API_KEY="your-key-here"

# Claude API
export ANTHROPIC_API_KEY="your-key-here"

# Groq API
export GROQ_API_KEY="your-key-here"

# Hugging Face
export HF_API_TOKEN="your-token-here"

# Google Gemini
export GOOGLE_API_KEY="your-key-here"

# Ollama
export OLLAMA_API_URL="http://localhost:11434/api"
export OLLAMA_MODEL="llama3"
```

### Method 2: .env File

Create a `.env` file in your project directory:

```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
GROQ_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
HF_API_TOKEN=your-token-here
OLLAMA_API_URL=http://localhost:11434/api
OLLAMA_MODEL=llama3
```

### Method 3: Metadata Dictionary

```python
metadata = {
    "openai_api_key": "your-key-here",
    "anthropic_api_key": "your-key-here", 
    "groq_api_key": "your-key-here",
    "hf_api_token": "your-token-here",
    "google_api_key": "your-key-here",
    "ollama_api_url": "http://localhost:11434/api",
    "ollama_model": "llama3"
}
```

## Using OpenAI API

```python
from ipfs_accelerate_py.api_backends import openai_api

# Initialize the API with credentials
metadata = {"openai_api_key": "your-key-here"}
openai = openai_api(resources={}, metadata=metadata)

# Chat completion
messages = [{"role": "user", "content": "Hello, how are you?"}]
response = openai.chat("gpt-4o", messages, prompt=None, system="You are a helpful assistant", 
                      temperature=0.7, max_tokens=100)
print(response["text"])

# Embeddings
embedding_response = openai.embedding("text-embedding-3-small", "This is a sample text", "float")
print(embedding_response["embedding"]) # Prints the embedding vector

# Image generation
image_response = openai.text_to_image("dall-e-3", "1024x1024", 1, 
                                     "A serene lakeside sunset with mountains in the background")
print(image_response["urls"]) # Prints the image URLs
```

## Using Claude API

```python
from ipfs_accelerate_py.api_backends import claude

# Initialize the API with credentials
metadata = {"anthropic_api_key": "your-key-here"}
claude_api = claude(resources={}, metadata=metadata)

# Chat completion
messages = [{"role": "user", "content": "Hello, how are you?"}]
response = claude_api.chat(messages)
print(response["text"])

# Streaming
for chunk in claude_api.stream_chat(messages):
    print(chunk["text"], end="", flush=True)
print()
```

## Using Ollama API

```python
from ipfs_accelerate_py.api_backends import ollama

# Initialize the API with credentials
metadata = {"ollama_api_url": "http://localhost:11434/api"}
ollama_api = ollama(resources={}, metadata=metadata)

# List available models
models = ollama_api.list_models()
print(f"Available models: {len(models)} found")

# Chat completion
messages = [{"role": "user", "content": "Hello, how are you?"}]
response = ollama_api.chat("llama3", messages)
print(response["text"])

# Streaming
for chunk in ollama_api.stream_chat("llama3", messages):
    print(chunk["text"], end="", flush=True)
print()
```

## Using Hugging Face TGI

```python
from ipfs_accelerate_py.api_backends import hf_tgi

# Initialize the API with credentials
metadata = {"hf_api_token": "your-token-here"}
tgi_api = hf_tgi(resources={}, metadata=metadata)

# Generate text
response = tgi_api.generate_text(
    "mistralai/Mistral-7B-Instruct-v0.2", 
    "Explain the concept of quantum computing in simple terms.",
    parameters={"max_new_tokens": 100}
)
print(response["generated_text"])

# Chat (if available)
if hasattr(tgi_api, "chat"):
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = tgi_api.chat("mistralai/Mistral-7B-Instruct-v0.2", messages)
    print(response["text"])
```

## Using Hugging Face TEI

```python
from ipfs_accelerate_py.api_backends import hf_tei
import numpy as np

# Initialize the API with credentials
metadata = {"hf_api_token": "your-token-here"}
tei_api = hf_tei(resources={}, metadata=metadata)

# Generate embedding
embedding = tei_api.generate_embedding(
    "sentence-transformers/all-MiniLM-L6-v2", 
    "This is a text to create an embedding from."
)
print(f"Embedding dimensionality: {len(embedding)}")

# Batch embeddings
texts = ["First sentence to embed.", "Second sentence to embed."]
embeddings = tei_api.batch_embed("sentence-transformers/all-MiniLM-L6-v2", texts)
print(f"Generated {len(embeddings)} embeddings")

# Calculate similarity
similarity = tei_api.calculate_similarity(embeddings[0], embeddings[1])
print(f"Similarity between texts: {similarity}")
```

## Using Groq API

```python
from ipfs_accelerate_py.api_backends import groq

# Initialize the API with credentials
metadata = {"groq_api_key": "your-key-here"}
groq_api = groq(resources={}, metadata=metadata)

# Basic chat completion
messages = [{"role": "user", "content": "Hello, how are you?"}]
response = groq_api.chat("llama3-8b-8192", messages)
print(response["text"])

# Advanced chat completion with parameters
system_message = "You are a concise assistant who always responds in bullet points"
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Explain quantum computing"}
]

response = groq_api.chat(
    model_name="llama3-70b-8192",  # Larger model for complex topics
    messages=messages,
    temperature=0.3,               # Lower temperature for more deterministic responses
    max_tokens=500,                # Control response length
    top_p=0.9,                     # Nucleus sampling parameter
    top_k=40,                      # Limit to top 40 tokens
    frequency_penalty=0.5,         # Reduce repetition
    presence_penalty=0.2           # Discourage repeated topics
)
print(response["text"])
print(f"Usage statistics: {response.get('usage', {})}")

# JSON response format
response = groq_api.chat(
    "llama3-8b-8192", 
    [{"role": "user", "content": "List the planets in the solar system"}],
    response_format={"type": "json_object"}
)
print(response["text"])

# Streaming with progress tracking
print("Streaming response: ", end="")
accumulated = ""
for chunk in groq_api.stream_chat("llama3-8b-8192", messages):
    print(chunk["text"], end="", flush=True)
    accumulated += chunk["text"]
    
    # Progress tracking
    if len(accumulated) % 100 == 0:
        print(f"\nReceived {len(accumulated)} characters so far...", end="")
print("\nDone!")
```

## Using Gemini API

```python
from ipfs_accelerate_py.api_backends import gemini

# Initialize the API with credentials
metadata = {"google_api_key": "your-key-here"}
gemini_api = gemini(resources={}, metadata=metadata)

# Chat completion
messages = [{"role": "user", "content": "Hello, how are you?"}]
response = gemini_api.chat(messages, model="gemini-1.0-pro")
print(response["text"])

# Multimodal input (image + text)
import base64
from pathlib import Path

# Load an image
image_path = Path("test.jpg")  # Path to your image
with open(image_path, "rb") as image_file:
    image_data = image_file.read()

# Process image with text prompt
response = gemini_api.process_image(
    image_data=image_data,
    prompt="Describe this image in detail",
    model="gemini-1.5-pro-vision"
)
print(response["text"])

# Streaming response
for chunk in gemini_api.stream_chat(messages, model="gemini-1.0-pro"):
    print(chunk["text"], end="", flush=True)
    if chunk["done"]:
        print("\nResponse complete")

## Advanced Features

### Groq API Models and Capabilities

The implementation provides categorized access to all Groq models:

```python
from ipfs_accelerate_py.api_backends import groq

# Initialize API client
groq_api = groq(resources={}, metadata={"groq_api_key": "your-key-here"})

# List all available models
all_models = groq_api.list_models()
print(f"Total models available: {len(all_models)}")

# List only chat-capable models
chat_models = groq_api.list_models("chat")
print(f"Chat models available: {len(chat_models)}")

# List vision models
vision_models = groq_api.list_models("vision")
print(f"Vision models available: {len(vision_models)}")

# List audio models
audio_models = groq_api.list_models("audio")
print(f"Audio models available: {len(audio_models)}")

# Check model compatibility
model_name = "mistral-saba-24b"
is_compatible = groq_api.is_compatible_model(model_name, "chat")
print(f"Is {model_name} compatible with chat: {is_compatible}")

# Print model details
for model in chat_models[:3]:  # First 3 chat models
    print(f"Model: {model['id']}")
    print(f"  Description: {model['description']}")
    print(f"  Context window: {model['context_window']}")
    print(f"  Category: {model['category']}")
```

### Advanced Features Across All APIs

All REAL API implementations include the following advanced features:

#### 1. Usage Tracking and Statistics

```python
from ipfs_accelerate_py.api_backends import groq

# Initialize the API
groq_api = groq(resources={}, metadata={"groq_api_key": "your-key-here"})

# Make API calls...
response = groq_api.chat("llama3-8b-8192", messages)

# Get detailed usage statistics (if supported)
if hasattr(groq_api, "get_usage_stats"):
    stats = groq_api.get_usage_stats()
    print(f"Total tokens used: {stats['total_tokens']}")
    print(f"Estimated cost: ${stats.get('estimated_cost_usd', 'N/A')}")
    print(f"Total requests: {stats['total_requests']}")
    
    # Reset usage statistics if needed
    groq_api.reset_usage_stats()
```

#### 2. Request Queueing and Concurrency Control

All API backends support configurable concurrency limits with robust thread-safe implementations:

```python
from ipfs_accelerate_py.api_backends import openai_api

# Initialize the API
client = openai_api()

# Configure concurrency
client.max_concurrent_requests = 3  # Process up to 3 requests at once
client.queue_size = 50              # Queue up to 50 pending requests

# Any requests beyond the concurrent limit will be queued automatically
# You don't need to do anything special to use the queue

# Create multiple endpoints with different settings
endpoint_id = client.create_endpoint(
    api_key="different-api-key",    # Use a different API key
    max_concurrent_requests=10,     # Higher concurrency for this endpoint
    queue_size=100                  # Larger queue for this endpoint
)

# Use the specific endpoint for requests
response = client.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    endpoint_id=endpoint_id          # Use the specific endpoint
)
```

#### 3. Exponential Backoff with Circuit Breaker

All APIs include automatic exponential backoff for transient errors and circuit breaker patterns for handling service outages:

```python
# Customize backoff behavior
client.max_retries = 5           # Maximum retry attempts
client.initial_retry_delay = 1   # Initial delay in seconds
client.backoff_factor = 2        # Multiply delay by this factor on each retry
client.max_retry_delay = 30      # Maximum delay in seconds

# Circuit breaker configuration (when available)
client.failure_threshold = 5     # Number of failures before opening circuit
client.reset_timeout = 30        # Seconds to wait before trying half-open

# Get circuit breaker status
if hasattr(client, "circuit_state"):
    print(f"Current circuit state: {client.circuit_state}")  # CLOSED, OPEN, or HALF_OPEN
```

#### 4. Environment Variables and .env Support

All APIs automatically load credentials from environment variables or .env files:

```python
# Create a .env file
# OPENAI_API_KEY=your-key-here

from ipfs_accelerate_py.api_backends import openai_api
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()  # This happens automatically in the API backends

# API key will be loaded automatically
client = openai_api()  # No explicit API key needed
```

### Groq Model Performance

Based on our testing, here are approximate performance metrics for different Groq models:

| Model | Response Time | Tokens/Second | Best For | Cost/Million Tokens |
|-------|---------------|--------------|----------|---------------------|
| llama3-8b-8192 | Very Fast (0.3-0.5s) | ~1000 | Quick responses, general questions | $0.20 |
| gemma2-9b-it | Fast (0.5-0.8s) | ~500 | Instruction following, creativity | $0.20 |
| mixtral-8x7b-32768 | Moderate (0.6-1.0s) | ~600 | Long context, detailed analysis | $0.60 |
| llama3-70b-8192 | Slower (1.0-2.0s) | ~300 | Complex reasoning, nuanced responses | $0.60 |
| llama-3.3-70b-versatile | Slower (1.0-2.0s) | ~300 | Versatile tasks, high quality | $0.60 |

## Creating Endpoint Handlers

All API backends support creating endpoint handlers that can be used with the IPFS Accelerate framework:

```python
# Create endpoint handlers for each API type
from ipfs_accelerate_py.api_backends import openai_api, claude, groq, ollama, hf_tgi, hf_tei, gemini

# OpenAI endpoint handler
openai_client = openai_api()
openai_handler = openai_client.create_chat_endpoint_handler("gpt-3.5-turbo")

# Claude endpoint handler
claude_client = claude()
claude_handler = claude_client.create_claude_endpoint_handler()

# Groq endpoint handler
groq_client = groq()
groq_handler = groq_client.create_groq_endpoint_handler()

# Ollama endpoint handler
ollama_client = ollama()
ollama_handler = ollama_client.create_ollama_endpoint_handler()

# HF TGI endpoint handler
hf_tgi_client = hf_tgi()
hf_tgi_handler = hf_tgi_client.create_remote_text_generation_endpoint_handler()

# HF TEI endpoint handler
hf_tei_client = hf_tei()
hf_tei_handler = hf_tei_client.create_remote_text_embedding_endpoint_handler()

# Gemini endpoint handler
gemini_client = gemini()
gemini_handler = gemini_client.create_gemini_endpoint_handler()

# Use any handler with a standardized interface
async def process_prompt(prompt, handler):
    result = await handler(prompt)
    return result["text"]
```

## Testing and Error Handling

### Testing API Implementations

```bash
# Test a specific API directly
python test_single_api.py ollama

# Test API backoff and queue functionality
python test_api_backoff_queue.py --api openai

# Run comprehensive suite of queue and backoff tests
python run_queue_backoff_tests.py --apis claude openai groq

# Run detailed tests for Ollama
python test_ollama_backoff_comprehensive.py

# Run tests for all APIs except specific ones
python run_queue_backoff_tests.py --skip-apis llvm s3_kit

# Test all API backends
python test_api_backend.py --api all

# Check implementation status
python check_api_implementation.py
python check_all_api_implementation.py
```

#### Advanced Testing Features

The test suite provides comprehensive verification for all API backends:

1. **Queue Testing**: Validates that requests beyond concurrency limits are properly queued
2. **Backoff Testing**: Confirms exponential backoff for rate limits and errors
3. **Circuit Breaker Testing**: Checks service outage detection and recovery
4. **Priority Queue Testing**: Ensures high-priority requests are processed first
5. **Mock Testing**: Allows testing without real API keys using sophisticated mocks

### Error Handling

All API implementations include robust error handling for:

- Authentication failures
- Rate limiting
- Invalid requests
- Network issues
- Timeout management

Example with error handling:

```python
from ipfs_accelerate_py.api_backends import groq

groq_api = groq(resources={}, metadata={"groq_api_key": "your-key-here"})

try:
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = groq_api.chat("llama3-8b-8192", messages)
    print(response["text"])
except ValueError as e:
    if "Authentication error" in str(e):
        print("API key is invalid or expired")
    elif "Rate limit" in str(e):
        print("Rate limit exceeded, please try again later")
    else:
        print(f"API error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
```

## Next Steps

For more detailed information:
- See API implementation details in the respective API backend files
- Check API_IMPLEMENTATION_STATUS.md for current implementation status
- Review test scripts for working examples of each API
- Examine CLAUDE.md for implementation patterns and architecture

## Advanced Features Added (March 2025)

All API backends now include these advanced features:

### 1. Queue and Backoff System (COMPLETED) 
- Thread-safe request queueing with proper locking
- Configurable concurrency limits with queue overflow handling
- Exponential backoff with retry mechanism for transient errors
- Queue processing with background thread management
- Request tracking with unique IDs for diagnostics

```python
# Configure queue and backoff settings
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api()

# Configure queue settings
client.queue_enabled = True               # Enable/disable queue (default: True)
client.max_concurrent_requests = 5        # Max concurrent requests (default: 5)
client.queue_size = 100                   # Maximum queue size (default: 100)

# Configure backoff settings
client.max_retries = 5                    # Max retry attempts (default: 5)
client.initial_retry_delay = 1            # Initial delay in seconds (default: 1)
client.backoff_factor = 2                 # Multiplier for each retry (default: 2)
client.max_retry_delay = 60               # Maximum delay cap in seconds (default: 60)

# Send request with a unique request ID for tracking
response = client.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    request_id="req_12345"                # Optional custom ID for tracking
)

# Check queue status (if supported)
if hasattr(client, "get_queue_info"):
    info = client.get_queue_info()
    print(f"Queue size: {info.get('size', 0)} requests")
    print(f"Active requests: {info.get('active_requests', 0)}")
    print(f"Queue capacity: {info.get('capacity', 0)}")
```

### 2. Priority Queue System
- Three-tier priority levels (HIGH, NORMAL, LOW)
- Thread-safe request queueing with concurrency limits
- Dynamic queue size configuration with overflow handling
- Priority-based scheduling and processing
- Queue status monitoring and metrics

```python
# Example usage of priority queue
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api()

# Set request priority
response = client.chat(
    model="gpt-4o", 
    messages=[{"role": "user", "content": "Critical task"}],
    priority="HIGH"  # Will be processed before NORMAL and LOW priority requests
)

# Check queue status
if hasattr(client, "get_queue_status"):
    status = client.get_queue_status()
    print(f"High priority queue: {status.get('high_priority', 0)} requests")
    print(f"Normal priority queue: {status.get('normal_priority', 0)} requests")
    print(f"Low priority queue: {status.get('low_priority', 0)} requests")
```

### 3. Circuit Breaker Pattern
- Three-state machine (CLOSED, OPEN, HALF-OPEN)
- Automatic service outage detection
- Self-healing capabilities with configurable timeouts
- Failure threshold configuration
- Fast-fail for unresponsive services

```python
# Configure circuit breaker
client.circuit_failure_threshold = 5  # Number of failures before opening circuit
client.circuit_reset_timeout = 30     # Seconds before trying half-open state
client.circuit_success_threshold = 3  # Successes needed to close circuit

# Check circuit state
if hasattr(client, "get_circuit_state"):
    state = client.get_circuit_state()
    print(f"Circuit state: {state}")  # CLOSED, OPEN, or HALF-OPEN
    
    # Get detailed circuit metrics
    metrics = client.get_circuit_metrics()
    print(f"Recent failures: {metrics.get('recent_failures', 0)}")
    print(f"Failure rate: {metrics.get('failure_rate', 0):.2f}%")
```

### 4. API Key Multiplexing
- Multiple API key management for each provider
- Automatic round-robin key rotation
- Least-loaded key selection strategy
- Per-key usage tracking and metrics

```python
# Configure multiple API keys
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api()

# Add multiple API keys
client.add_api_key("key1", "sk-abc123...")
client.add_api_key("key2", "sk-def456...")
client.add_api_key("key3", "sk-ghi789...")

# Set selection strategy
client.key_selection_strategy = "least-loaded"  # or "round-robin"

# Use specific key if needed
response = client.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    api_key_name="key2"  # Use this specific key
)

# Get key usage statistics
if hasattr(client, "get_key_usage_stats"):
    stats = client.get_key_usage_stats()
    for key_name, usage in stats.items():
        print(f"Key: {key_name}")
        print(f"  Requests: {usage.get('requests', 0)}")
        print(f"  Tokens: {usage.get('tokens', 0)}")
        print(f"  Cost: ${usage.get('cost', 0):.4f}")
```

### 5. Semantic Caching
- Caching based on semantic similarity
- Automatic embedding of queries for cache matching
- Configurable similarity threshold
- Customizable cache expiry

```python
# Enable semantic caching
from ipfs_accelerate_py.api_backends import claude

client = claude()

# Configure semantic cache
client.enable_semantic_cache(
    similarity_threshold=0.92,     # 0.0-1.0, higher means more strict matching
    cache_size=1000,               # Maximum number of entries to store
    ttl=3600,                      # Time-to-live in seconds
    embedding_model="mini"         # Model to use for embeddings
)

# Cache will be used automatically for similar requests
response1 = client.chat([{"role": "user", "content": "Explain quantum computing"}])
response2 = client.chat([{"role": "user", "content": "What is quantum computing?"}])
# Second request might use cached response if similarity exceeds threshold

# Get cache statistics
if hasattr(client, "get_cache_stats"):
    stats = client.get_cache_stats()
    print(f"Cache hits: {stats.get('hits', 0)}")
    print(f"Cache misses: {stats.get('misses', 0)}")
    print(f"Hit rate: {stats.get('hit_rate', 0):.2f}%")
    print(f"Estimated savings: ${stats.get('cost_savings', 0):.4f}")
```

### 6. Request Batching
- Automatic request combining for compatible models
- Configurable batch size and timeout
- Model-specific batching strategies
- Batch queue management

```python
# Configure request batching for embedding operations
from ipfs_accelerate_py.api_backends import openai_api

client = openai_api()

# Enable batching for embeddings
client.enable_embedding_batching(
    max_batch_size=20,     # Maximum items in a batch
    batch_timeout=0.1,     # Seconds to wait for batch to fill
    auto_batching=True     # Automatically batch compatible requests
)

# Multiple single requests will be automatically batched together
import concurrent.futures

# These will likely be batched together for efficiency
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    for i in range(10):
        future = executor.submit(
            client.embedding, 
            "text-embedding-3-small", 
            f"This is text sample {i}", 
            "float"
        )
        futures.append(future)
        
    # Get results
    results = [future.result() for future in futures]
    
# Get batching statistics
if hasattr(client, "get_batching_stats"):
    stats = client.get_batching_stats()
    print(f"Batches processed: {stats.get('batches', 0)}")
    print(f"Requests batched: {stats.get('batched_requests', 0)}")
    print(f"Average batch size: {stats.get('avg_batch_size', 0):.1f}")
    print(f"Estimated savings: ${stats.get('cost_savings', 0):.4f}")
```

## Testing Queue and Backoff Functionality

The framework includes comprehensive tests for queue and backoff functionality:

```python
# Test basic queue functionality
python test_api_backoff_queue.py --api [api_name]

# Run comprehensive Ollama tests
python test_ollama_backoff_comprehensive.py --model llama3 --host http://localhost:11434

# Test different queue sizes
python test_ollama_backoff_comprehensive.py --queue-size 10 --max-concurrent 2

# Run all queue and backoff tests
python run_queue_backoff_tests.py

# Test specific APIs
python run_queue_backoff_tests.py --apis openai groq claude

# Skip specific APIs
python run_queue_backoff_tests.py --skip-apis llvm opea ovms
```

These advanced features provide significant performance improvements, cost savings, and system reliability for all API backends implemented in the IPFS Accelerate framework.