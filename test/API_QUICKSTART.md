# API Backends Quickstart Guide

This guide provides simple examples for using the IPFS Accelerate API backends with proper credentials.

## Overview

The IPFS Accelerate framework provides a standardized interface to multiple LLM API providers. This guide will help you get started with the fully implemented backends.

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
```

### Method 2: Metadata Dictionary

```python
metadata = {
    "openai_api_key": "your-key-here",
    "claude_api_key": "your-key-here", 
    "groq_api_key": "your-key-here",
    "hf_api_token": "your-token-here",
    "gemini_api_key": "your-key-here"
}
```

## Using OpenAI API

```python
from api_backends import openai_api

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
print(embedding_response["text"])

# Image generation
image_response = openai.text_to_image("dall-e-3", "1024x1024", 1, 
                                     "A serene lakeside sunset with mountains in the background")
print(image_response["text"])
```

## Using Claude API

```python
from api_backends import claude

# Initialize the API with credentials
metadata = {"claude_api_key": "your-key-here"}
claude_api = claude.claude(resources={}, metadata=metadata)

# Chat completion
messages = [{"role": "user", "content": "Hello, how are you?"}]
response = claude_api.chat(messages)
print(response["text"])

# Streaming
for chunk in claude_api.stream_chat(messages):
    print(chunk["text"], end="", flush=True)
print()
```

## Using Groq API

```python
from api_backends import groq

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

### Groq API Models and Capabilities

The implementation provides categorized access to all Groq models:

```python
from api_backends import groq

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

### Groq API Advanced Features

The Groq API implementation includes several advanced features:

#### 1. Usage Tracking and Cost Estimation

```python
# Initialize the API
groq_api = groq(resources={}, metadata={"groq_api_key": "your-key-here"})

# Make API calls...
response = groq_api.chat("llama3-8b-8192", messages)

# Get detailed usage statistics
stats = groq_api.get_usage_stats()
print(f"Total tokens used: {stats['total_tokens']}")
print(f"Estimated cost: ${stats['estimated_cost_usd']}")
print(f"Total requests: {stats['total_requests']}")

# Reset usage statistics if needed
groq_api.reset_usage_stats()
```

#### 2. Client-side Token Counting

```python
# Estimate tokens before sending to the API
text = "This is a sample text to count tokens for"
count_result = groq_api.count_tokens(text, "llama3-8b-8192")
print(f"Estimated tokens: {count_result['estimated_token_count']}")
```

#### 3. Advanced Generation Parameters

```python
# Use seed for deterministic outputs
response1 = groq_api.chat("llama3-8b-8192", messages, seed=42)
response2 = groq_api.chat("llama3-8b-8192", messages, seed=42)  # Same result as response1

# Control token likelihoods with logit_bias
# Increase likelihood of specific tokens
logit_bias = {12: 2.0, 456: 1.5}  # Increase likelihood of tokens 12 and 456
response = groq_api.chat("llama3-8b-8192", messages, logit_bias=logit_bias)
```

#### 4. Request Tracking

```python
# Add custom request IDs for tracking
custom_id = f"request_{int(time.time())}"
response = groq_api.chat("llama3-8b-8192", messages, request_id=custom_id)
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

## Testing Implementation Types

To test whether an API implementation is using real API calls or mock objects, use the provided test utilities:

```python
# From the command line
cd /path/to/ipfs_accelerate_py/test
python check_api_implementation.py

# For a specific API
python test_single_api.py openai
python test_single_api.py claude
python test_single_api.py groq
```

## Credential Management for Testing

For secure credential storage during testing:

```python
from test_api_real_implementation import CredentialManager

# Store and retrieve credentials
cred_manager = CredentialManager()
cred_manager.set("openai", "your-api-key")

# Later retrieve the credential
api_key = cred_manager.get("openai")
```

## Advanced Usage

### Creating Endpoint Handlers

```python
# Create a standardized endpoint handler
from api_backends import groq

groq_api = groq.groq(resources={}, metadata={"groq_api_key": "your-key-here"})
endpoint_handler = groq_api.create_groq_endpoint_handler()

# Use the handler directly
response = endpoint_handler("What is the capital of France?")
print(response)
```

### Error Handling

The API implementations include robust error handling for:

- Authentication failures
- Rate limiting
- Invalid requests
- Network issues
- Timeout management

Example with error handling:

```python
from api_backends import groq

groq_api = groq.groq(resources={}, metadata={"groq_api_key": "your-key-here"})

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