# IPFS ACCELERATE

This is meant to be an extension of the Huggingface accelerate library, is to act as a model server, which can contain lists of other endpoints to call, or can call a local instance, and can respond to external calls for inference. I have some modular back ends, such as Libp2p, Akash, Lilypad, Huggingface Zero, Vast AI, which I use for autoscaling. If the model is already listed in the ipfs_model_manager there should be an associated hw_requirements key of the manifest. In the case of libp2p the request to do inference will go out to peers in the same trusted zone, if there are no peers in the network available to fulfill the task, and it has the resources to run the model localliy it will do so, otherwise a docker container will need to be launched with one of the providers here. 

# IPFS Huggingface Bridge:

for huggingface transformers python library visit:
https://github.com/endomorphosis/ipfs_transformers/

for huggingface datasets python library visit:
https://github.com/endomorphosis/ipfs_datasets/

for faiss KNN index python library visit:
https://github.com/endomorphosis/ipfs_faiss

for transformers.js visit:                          
https://github.com/endomorphosis/ipfs_transformers_js

for orbitdb_kit nodejs library visit:
https://github.com/endomorphosis/orbitdb_kit/

for fireproof_kit nodejs library visit:
https://github.com/endomorphosis/fireproof_kit

for ipfs_kit nodejs library visit:
https://github.com/endomorphosis/ipfs_kit/

for python model manager library visit: 
https://github.com/endomorphosis/ipfs_model_manager/

for nodejs model manager library visit: 
https://github.com/endomorphosis/ipfs_model_manager_js/

for nodejs ipfs huggingface scraper with pinning services visit:
https://github.com/endomorphosis/ipfs_huggingface_scraper/

for ipfs agents visit:
https://github.com/endomorphosis/ipfs_agents/

for ipfs accelerate visit:
https://github.com/endomorphosis/ipfs_accelerate/

# IPFS Accelerate Python

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

Author - Benjamin Barber
QA - Kevin De Haan
