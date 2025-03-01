# API Backends Quickstart Guide (March 1, 2025)

This guide provides simple examples for using the IPFS Accelerate API backends with proper credentials.

## Overview

The IPFS Accelerate framework provides a standardized interface to multiple LLM API providers. All 7 high-priority API backends are now fully implemented with REAL functionality:

| API | Status | Used For |
|-----|--------|----------|
| OpenAI | ✅ REAL | GPT models, embeddings, assistants |
| Claude | ✅ REAL | Claude models, streaming |
| Groq | ✅ REAL | High-speed inference, Llama models |
| Ollama | ✅ REAL | Local deployment, open-source models |
| HF TGI | ✅ REAL | Text generation with Hugging Face models |
| HF TEI | ✅ REAL | Embeddings with Hugging Face models |
| Gemini | ✅ REAL | Google's models, multimodal capabilities |

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

All API backends support configurable concurrency limits:

```python
from ipfs_accelerate_py.api_backends import openai_api

# Initialize the API
client = openai_api()

# Configure concurrency
client.max_concurrent_requests = 3  # Process up to 3 requests at once
client.queue_size = 50              # Queue up to 50 pending requests

# Any requests beyond the concurrent limit will be queued automatically
# You don't need to do anything special to use the queue
```

#### 3. Exponential Backoff

All APIs include automatic exponential backoff for transient errors:

```python
# Customize backoff behavior
client.max_retries = 5           # Maximum retry attempts
client.initial_retry_delay = 1   # Initial delay in seconds
client.backoff_factor = 2        # Multiply delay by this factor on each retry
client.max_retry_delay = 30      # Maximum delay in seconds
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

# Test all API backends
python test_api_backend.py --api all

# Check implementation status
python check_api_implementation.py
```

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