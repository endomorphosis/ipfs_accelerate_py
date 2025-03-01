# Google Gemini API Implementation

## Overview

This is a complete implementation of the Google Gemini API for the IPFS Accelerate Python Framework. The implementation provides a full-featured Python interface to Google's Generative AI models, including Gemini Pro for text generation, Gemini Pro Vision for image understanding, and Embedding models for text embeddings.

## Features

### Core Features
- **Text Generation**: Generate text responses from prompts using Gemini Pro
- **Chat Functionality**: Support for multi-turn conversations with role-based messaging
- **Streaming Responses**: Real-time streaming of model outputs for improved user experience
- **Multimodal Support**: Process and analyze images with text prompts using Gemini Pro Vision
- **Embeddings Generation**: Create text embeddings for semantic search and retrieval

### Advanced Features
- **Batch Processing**: Process multiple prompts efficiently in batches
- **Response Caching**: LRU caching with hash-based keys for improved performance
- **Token Usage Tracking**: Monitor token consumption across all requests
- **Multiplexed API Keys**: Support for multiple API keys with automatic rotation

### Request Management
- **Request IDs**: Unique tracking IDs for all requests with monitoring
- **Request History**: Comprehensive request history with filtering and analysis
- **Active Request Tracking**: Real-time tracking of in-progress requests
- **Performance Metrics**: Detailed metrics for all requests including latency and token usage

### Endpoint Management
- **Dedicated Endpoints**: Create model-specific endpoints with unique IDs
- **Per-Endpoint Queues**: Each endpoint gets its own request queue
- **Smart Backoff**: Intelligent rate limiting with per-endpoint exponential backoff
- **Status Monitoring**: Comprehensive endpoint monitoring and metrics
- **Endpoint Metrics**: Request rate, success/error rates, and performance tracking

### Reliability Features
- **Safety Settings**: Customizable content filtering with safety thresholds
- **Automatic Retries**: Smart exponential backoff for transient errors
- **Error Classification**: Intelligent handling of different error types
- **Timeout Configuration**: Configurable timeouts for different request types
- **Load Balancing**: Distribute requests across multiple endpoints
- **Concurrent Processing**: Handle multiple requests efficiently in parallel

### Customization
- **Parameter Control**: Fine-tune generation with temperature, top-p, top-k, and more
- **Stop Sequences**: Support for custom stop sequences to control generation
- **Model Versioning**: Support for different API versions and model variants
- **Custom Prompt Formats**: Flexible prompt formatting with structure preservation

## Usage Examples

### Basic Chat

```python
from api_backends import gemini

# Initialize with API key from .env file or environment variable
gemini_api = gemini.gemini(metadata={"gemini_api_key": "YOUR_API_KEY"})

# Simple chat with Gemini
messages = [{"role": "user", "content": "What is the capital of France?"}]
response = gemini_api.chat(messages)

print(response["choices"][0]["message"]["content"])
```

### Multi-turn Conversation

```python
# Multi-turn conversation with system message
messages = [
    {"role": "system", "content": "You are a helpful travel assistant that provides concise information."},
    {"role": "user", "content": "Tell me about Paris."},
    {"role": "assistant", "content": "Paris is the capital of France and known for the Eiffel Tower, Louvre museum, and rich history."},
    {"role": "user", "content": "What's the best time to visit?"}
]

response = gemini_api.chat(messages)
print(response["choices"][0]["message"]["content"])
```

### Streaming Responses

```python
# Stream response chunks for real-time output
messages = [{"role": "user", "content": "Write a short story about a robot."}]
for chunk in gemini_api.stream_chat(messages):
    content = chunk["choices"][0]["delta"].get("content", "")
    print(content, end="", flush=True)
```

### Image Analysis

```python
# Process an image with descriptive prompt
with open("image.jpg", "rb") as f:
    image_data = f.read()

result = gemini_api.process_image(
    image_data,
    "Describe what you see in this image in detail.",
    mime_type="image/jpeg"
)

print(result["analysis"])
```

### Streaming Image Analysis

```python
# Process an image with streaming response
with open("image.jpg", "rb") as f:
    image_data = f.read()

for chunk in gemini_api.process_image(
    image_data,
    "Describe this image in detail, focusing on all visible elements.",
    stream=True
):
    print(chunk["analysis_chunk"], end="", flush=True)
```

### Custom Generation Parameters

```python
# Control generation with custom parameters
messages = [{"role": "user", "content": "Generate a creative slogan for a coffee shop."}]
response = gemini_api.chat(
    messages,
    temperature=0.9,  # Higher temperature for more creative output
    top_p=0.95,
    top_k=40,
    max_tokens=100,
    stop_sequences=[".", "!"]  # Stop at end of sentence
)

print(response["choices"][0]["message"]["content"])
```

### Direct Content Generation

```python
# Generate content directly with a prompt
response = gemini_api.generate_content(
    "Explain quantum computing in simple terms",
    temperature=0.3  # Lower temperature for more focused, factual response
)

# Extract text from response
text = ""
for candidate in response.get("candidates", []):
    content = candidate.get("content", {})
    for part in content.get("parts", []):
        if "text" in part:
            text += part["text"]

print(text)
```

### Generating Embeddings

```python
# Generate embeddings for semantic search
texts = [
    "What is machine learning?",
    "How does quantum computing work?",
    "Explain neural networks"
]

embeddings = gemini_api.generate_embeddings(texts)

# Access the embedding vectors
for item in embeddings["data"]:
    vector = item["embedding"]
    print(f"Vector dimension: {len(vector)}")
```

### Batch Processing Multiple Prompts

```python
# Process multiple prompts efficiently
prompts = [
    "Write a haiku about mountains",
    "Describe a sunset in one sentence",
    "List three facts about dolphins"
]

batch_results = gemini_api.batch_generate_content(
    prompts,
    temperature=0.7,
    batch_size=3  # Process in batches of 3
)

for i, result in enumerate(batch_results):
    print(f"Prompt {i+1} result:")
    candidates = result.get("candidates", [])
    if candidates:
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text = " ".join([part.get("text", "") for part in parts if "text" in part])
        print(text)
```

### Safety Settings Customization 

```python
# Customize safety settings
custom_safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_LOW_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

response = gemini_api.generate_content(
    "Write a story about conflict resolution",
    safety_settings=custom_safety_settings
)
```

### Caching for Performance

```python
# Enable response caching for repeated requests
gemini_api = gemini.gemini(metadata={
    "gemini_api_key": "YOUR_API_KEY",
    "use_cache": True,
    "cache_max_size": 200  # Cache up to 200 responses
})

# Repeated identical requests will use the cache
response1 = gemini_api.generate_content("What is the speed of light?")
response2 = gemini_api.generate_content("What is the speed of light?")  # Uses cache

# Get token usage statistics
print(gemini_api.get_token_usage())

# Clear cache when needed
gemini_api.clear_cache()
```

## Implementation Details

### API Endpoint Structure

The implementation uses Google's Generative Language API endpoints:

- **Text Generation**: `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent`
- **Multimodal**: `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent`
- **Embeddings**: `https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent`
- **Streaming**: Same endpoints with SSE (Server-Sent Events) streaming via `alt=sse` parameter

### Endpoint Management System

The implementation includes a sophisticated endpoint management system:

```python
# Create a dedicated endpoint for a specific model
endpoint_id = gemini_api.create_endpoint(
    model="gemini-pro", 
    api_key="YOUR_API_KEY",
    endpoint_name="my-custom-endpoint"  # Optional
)

# Use the endpoint for requests
response = gemini_api.chat(
    messages=[{"role": "user", "content": "Hello"}],
    endpoint_id=endpoint_id
)

# Get endpoint status
status = gemini_api.get_endpoint_status(endpoint_id)
print(f"Requests: {status['request_count']}, Error rate: {status['error_rate']}")

# List all endpoints
endpoints = gemini_api.list_endpoints()
for endpoint in endpoints:
    print(f"ID: {endpoint['id']}, Model: {endpoint['model']}, Active: {endpoint['active']}")

# Get endpoint metrics
metrics = gemini_api.get_endpoint_metrics()
for model, stats in metrics.items():
    print(f"Model: {model}, Success rate: {stats['success_rate']:.2%}")
```

### Request Tracking System

```python
# Generate with explicit request ID
request_id = "my-request-123"
response = gemini_api.generate_content(
    "Explain quantum computing",
    request_id=request_id
)

# Get active requests
active = gemini_api.get_active_requests()
for req_id, info in active.items():
    print(f"ID: {req_id}, Status: {info['status']}, Duration: {info['duration']:.2f}s")

# Get request history
history = gemini_api.get_request_history(limit=5, filter_status="success")
for req_id, info in history.items():
    print(f"ID: {req_id}, Model: {info['model']}, Duration: {info['duration']:.2f}s")
```

### Models Supported

- **gemini-pro**: The primary text generation model optimized for text-only inputs
- **gemini-pro-vision**: Multimodal model capable of understanding images alongside text
- **embedding-001**: Embedding model for creating vector representations of text

Additional models can be easily supported by updating the model parameters in the gemini class configuration.

### Authentication

The implementation provides multiple authentication options for flexibility:

1. **Metadata Dictionary**: Pass API key directly in the constructor:
   ```python
   gemini_api = gemini.gemini(metadata={"gemini_api_key": "YOUR_API_KEY"})
   ```

2. **Environment Variables**: Set any of these environment variables:
   - Primary: `GEMINI_API_KEY`
   - Alternative: `GOOGLE_API_KEY`
   
3. **.env File Support**: Add your API key to `.env` file in this format:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Multiplexed Keys**: For high-volume applications, use numbered keys that will be rotated:
   ```
   GEMINI_API_KEY_1=your_first_api_key
   GEMINI_API_KEY_2=your_second_api_key
   # ... more keys as needed
   ```

### Safety Settings

The implementation supports Google's comprehensive safety settings for content filtering:

```python
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]
```

Available thresholds:
- `BLOCK_NONE`: Don't block any content
- `BLOCK_ONLY_HIGH`: Block only high-risk content
- `BLOCK_MEDIUM_AND_ABOVE`: Block medium and high-risk content (default)
- `BLOCK_LOW_AND_ABOVE`: Block low, medium, and high-risk content
- `BLOCK_ALL`: Block all potentially harmful content

### Caching System

The implementation includes a sophisticated caching system:

1. **LRU Cache**: Least Recently Used cache eviction policy
2. **Configurable Size**: Adjustable cache size limit
3. **Hash-based Keys**: MD5 hashing of request data for efficient lookup
4. **Memory Efficient**: Only caches responses, not full request objects
5. **Streaming Exclusion**: Automatically skips caching for streaming responses

Enable caching with:
```python
gemini_api = gemini.gemini(metadata={"use_cache": True, "cache_max_size": 200})
```

### Token Usage Tracking

The implementation automatically tracks token usage across requests:

```python
# Get current token usage
token_stats = gemini_api.get_token_usage()
print(f"Prompt tokens: {token_stats['prompt_tokens']}")
print(f"Completion tokens: {token_stats['completion_tokens']}")
print(f"Total tokens: {token_stats['total_tokens']}")

# Reset counters when needed
gemini_api.reset_token_usage()
```

This tracking works across both streaming and non-streaming responses.

### Reliability Features

The implementation includes several reliability features:

1. **Automatic Retries**: Exponential backoff for transient errors
2. **Error Classification**: Different retry strategies based on error type
3. **Timeout Configuration**: Separate timeouts for different request types
4. **Server Error Handling**: Special handling for 5xx server errors
5. **Network Error Recovery**: Graceful handling of connection issues

### Response Formatting

For compatibility with other APIs in the framework, responses are formatted to match OpenAI's standardized structure:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Response text"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

For streaming responses, each chunk follows this format:

```json
{
  "choices": [
    {
      "delta": {
        "role": "assistant",
        "content": "partial text chunk"
      },
      "finish_reason": null
    }
  ]
}
```

### Message Format Conversion

The implementation automatically converts between different message formats:

1. **OpenAI Format**:
   ```json
   [
     {"role": "system", "content": "You are a helpful assistant"},
     {"role": "user", "content": "Hello, who are you?"}
   ]
   ```

2. **Gemini Format**:
   ```json
   [
     {"role": "user", "parts": [{"text": "System instruction: You are a helpful assistant"}]},
     {"role": "user", "parts": [{"text": "Hello, who are you?"}]}
   ]
   ```

3. **Multimodal Format**:
   ```json
   [
     {
       "role": "user", 
       "content": [
         {"type": "text", "text": "What's in this image?"},
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
       ]
     }
   ]
   ```

### Smart Backoff System

The implementation includes a sophisticated backoff system to handle rate limits and errors:

```python
# Each endpoint has its own backoff configuration
endpoint_config = {
    "backoff": {
        "current_delay": 0,          # Current backoff delay in seconds
        "base_delay": 1,             # Starting delay
        "max_delay": 60,             # Maximum delay
        "consecutive_errors": 0,     # Count of consecutive errors
        "last_request_time": time()  # Timestamp of last request
    }
}

# The backoff system automatically:
# 1. Tracks errors per endpoint
# 2. Applies exponential backoff (2^consecutive_errors * base_delay)
# 3. Respects different error types (rate limits vs. other errors)
# 4. Resets backoff after successful requests
# 5. Shares backoff status across all requests to an endpoint
```

### Load Balancing Example

```python
# Create multiple endpoints for the same model but different API keys
endpoints = []
for i, api_key in enumerate(["KEY1", "KEY2", "KEY3"]):
    endpoint_id = gemini_api.create_endpoint(
        model="gemini-pro",
        api_key=api_key,
        endpoint_name=f"gemini-pro-endpoint-{i}"
    )
    endpoints.append(endpoint_id)

# Create a simple load balancer
def get_best_endpoint():
    # Get status for all endpoints
    statuses = [gemini_api.get_endpoint_status(eid) for eid in endpoints]
    
    # Find endpoint with lowest backoff and queue size
    return min(
        statuses, 
        key=lambda s: (s['backoff']['current_delay'], s['queue_size'])
    )['id']

# Use the best endpoint for each request
for i in range(100):
    best_endpoint = get_best_endpoint()
    response = gemini_api.generate_content(
        f"Generate text {i}",
        endpoint_id=best_endpoint
    )
```

## Completed Features and Improvements

All the following features have been successfully implemented:

- ✅ **Dedicated Endpoint System**: Complete endpoint management with per-endpoint queues and metrics
- ✅ **Request ID Tracking**: Full request lifecycle tracking with history and performance metrics
- ✅ **Smart Backoff**: Per-endpoint exponential backoff with error classification
- ✅ **Embedding Generation**: Support for creating text embeddings using Google's embedding models
- ✅ **Batch Processing**: Efficient processing of multiple prompts in batches with controllable batch size
- ✅ **Response Caching**: LRU caching system with hash-based keys for improved performance
- ✅ **Token Usage Tracking**: Automatic tracking of token consumption across all request types
- ✅ **Safety Settings**: Configurable content filtering with all safety thresholds
- ✅ **Request Retry Logic**: Exponential backoff with intelligent error categorization
- ✅ **Streaming Image Analysis**: Support for streaming responses during image processing
- ✅ **Multi-key Support**: Ability to use multiple API keys with automatic rotation
- ✅ **Stop Sequences**: Support for custom stop sequences to control generation
- ✅ **Timeout Configuration**: Configurable timeouts for different request types
- ✅ **Error Handling**: Comprehensive error handling for all API error types
- ✅ **Model Versioning**: Support for different API versions and model variants

## Integration with IPFS Accelerate Framework

The Gemini API implementation integrates seamlessly with the IPFS Accelerate Python Framework:

1. **Unified Endpoint Handler**: Creates a flexible callable handler supporting all Gemini features
2. **Resource Sharing**: Shares queues and endpoints with the framework's resource system
3. **Configuration**: Uses metadata for flexible configuration of all features
4. **Standardized Interface**: Follows the same patterns as other API implementations
5. **Error Propagation**: Properly handles and propagates errors to the framework
6. **Request Multiplexing**: Supports multiplexing requests across multiple endpoints and API keys

## Future Development Potential

While all core and advanced features have been implemented, future improvements could include:

- **Function Calling**: Implement function calling capabilities when Google adds this to the Gemini API
- **Semantic Caching**: Enhanced caching based on semantic similarity rather than exact matches
- **Feedback API Integration**: Support for Google's feedback API when available
- **Custom Tuning**: Support for custom-tuned models when Google makes this available
- **Multi-GPU Dispatching**: Intelligent distribution of requests across multiple GPU resources
- **Advanced Load Balancing**: Implement weighted load balancing based on endpoint health metrics