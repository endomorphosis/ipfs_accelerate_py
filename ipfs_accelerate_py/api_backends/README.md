
## API Backends

The package provides flexible API backends for interfacing with different model serving solutions:

### LLVM Backend
The LLVM backend provides integration with LLVM-based model endpoints. Features:
- Standard text completion generation
- Chat completions with structured messages
- Streaming responses for real-time generation
- Model management and listing

Example usage:
```python
from ipfs_accelerate_py.api_backends import llvm

# Initialize backend
llvm_backend = llvm()

# Get endpoint handler
endpoint_url, api_key, handler, queue, batch_size = llvm_backend.init(
    endpoint_url="http://localhost:8080",
    model_name="my-model",
    endpoint_type="chat"  # or "completion", "streaming"
)

# Use handler for generation
messages = [
    {"role": "user", "content": "Hello! How are you?"}
]
response = handler(messages)
```

### Ollama Backend
The Ollama backend integrates with Ollama's API for local model serving. Features:
- Text completion generation
- Chat completions with history
- Streaming responses
- Text embeddings
- Model management and tags

Example usage:
```python
from ipfs_accelerate_py.api_backends import ollama

# Initialize backend
ollama_backend = ollama()

# Get endpoint handler
endpoint_url, api_key, handler, queue, batch_size = ollama_backend.init(
    endpoint_url="http://localhost:11434",
    model_name="llama2",
    endpoint_type="chat"  # or "completion", "streaming", "embedding"
)

# Use handler for generation
response = handler("What is machine learning?", 
    parameters={
        "temperature": 0.7,
        "num_predict": 100
    }
)
```

### Gemini Backend
The Gemini backend provides integration with Google's Gemini API. Features:
- Text completion generation
- Chat completions
- Multimodal inputs (text + images) with Gemini Pro Vision
- Embedding generation
- Built-in retries and error handling

Example usage:
```python
from ipfs_accelerate_py.api_backends import gemini

# Initialize backend
gemini_backend = gemini()

# Get endpoint handler for text/chat
endpoint_url, api_key, handler, queue, batch_size = gemini_backend.init(
    api_key="your-google-api-key",
    model_name="gemini-pro",
    endpoint_type="chat"  # or "completion", "vision", "embedding"
)

# For text/chat
response = handler("What is quantum computing?", 
    parameters={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_output_tokens": 1024
    }
)

# For vision tasks
from PIL import Image
image = Image.open("image.jpg")
vision_response = handler("Describe this image", image_data=image)
```

### Claude Backend
The Claude backend integrates with Anthropic's Claude API. Features:
- Advanced text completion
- Chat with system prompts
- Tool/function calling capabilities
- Support for Claude 3 models (Opus, Sonnet, Haiku)
- Long context handling

Example usage:
```python
from ipfs_accelerate_py.api_backends import claude

# Initialize backend
claude_backend = claude()

# Get endpoint handler
endpoint_url, api_key, handler, queue, batch_size = claude_backend.init(
    api_key="your-anthropic-api-key",
    model_name="claude-3-opus",
    endpoint_type="chat"  # or "completion"
)

# Using system prompts
messages = [
    {"role": "system", "content": "You are a helpful science teacher."},
    {"role": "user", "content": "Explain black holes"}
]

response = handler(messages, 
    parameters={
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.9
    }
)
```

### Common Features
Both backends support:
- Async and sync request handling
- Request queueing and batching
- Multiple response formats
- Error handling and retries
- Model endpoint management

The handlers are designed to be interchangeable, following similar patterns for:
- Text completion
- Chat completion 
- Streaming responses
- Model management

Refer to the API documentation of each backend class for detailed method signatures and parameters.
