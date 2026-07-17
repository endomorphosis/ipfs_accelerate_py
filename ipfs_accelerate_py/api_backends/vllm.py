import os
import requests
import json
import logging
from typing import Dict, List, Optional, Union, Any, Callable

logger = logging.getLogger(__name__)

try:
    from .base import BaseAPIBackend
except ImportError:
    try:
        from base import BaseAPIBackend
    except ImportError:
        BaseAPIBackend = object

# Try to import storage wrapper
try:
    from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

# Try to import datasets integration for API tracking
try:
    from ..datasets_integration import (
        is_datasets_available,
        ProvenanceLogger,
        DatasetsManager
    )
    HAVE_DATASETS_INTEGRATION = True
except ImportError:
    try:
        from datasets_integration import (
            is_datasets_available,
            ProvenanceLogger,
            DatasetsManager
        )
        HAVE_DATASETS_INTEGRATION = True
    except ImportError:
        HAVE_DATASETS_INTEGRATION = False
        is_datasets_available = lambda: False
        ProvenanceLogger = None
        DatasetsManager = None

class vllm(BaseAPIBackend):
    """vLLM API Backend Integration

    Provides a standard endpoint-handler interface for vLLM inference servers,
    sharing the same code path as all other API backends (ollama, hf_tgi, …).

    Supports:
    - Standard text completion  (/v1/completions)
    - Chat completions          (/v1/chat/completions)
    - Streaming responses       (SSE / server-sent events)
    - Model listing             (/v1/models)

    Handler methods follow the shared pattern:
    - completion: (prompt, parameters=None) → str
    - chat:       (messages, parameters=None) → str
    - streaming:  (prompt_or_messages, parameters=None) → generator[dict]
    """

    def __init__(self, resources=None, metadata=None):
        """Initialize vLLM backend interface.

        Args:
            resources: Resources configuration dictionary
            metadata: Additional metadata dictionary; recognised keys:
                vllm_api_key, vllm_api_url, queue_size, max_concurrent_requests
        """
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}

        # Endpoint registry
        self.endpoints: Dict[str, List[str]] = {}
        self.endpoint_status: Dict[str, int] = {}
        self.registered_models: Dict[str, Any] = {}

        # Base URL / API key (can be overridden per-call)
        self.default_api_url = self.metadata.get(
            "vllm_api_url",
            os.environ.get("VLLM_API_URL", "http://localhost:8000"),
        )
        self.default_api_key = self.metadata.get(
            "vllm_api_key",
            os.environ.get("VLLM_API_KEY", ""),
        )

        # Shared queue + circuit-breaker from BaseAPIBackend
        self._init_queue(
            queue_size=int(self.metadata.get("queue_size", 100)),
            max_concurrent_requests=int(self.metadata.get("max_concurrent_requests", 5)),
        )
        self._init_circuit_breaker()

        # Distributed storage (optional)
        self._storage = None
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper()
                if self._storage:
                    logger.info("vLLM: Distributed storage initialized")
            except Exception as exc:
                logger.debug("vLLM: Could not initialize storage: %s", exc)

    def init(self, endpoint_url=None, api_key=None, model_name=None, endpoint_type="completion"):
        """Initialize a connection to a remote vLLM endpoint
        
        Supported endpoint_types:
        - "completion": Standard text completion
        - "chat": Structured chat completion 
        - "streaming": Stream responses chunk by chunk
        
        Args:
            endpoint_url: The URL of the remote endpoint
            api_key: API key for authentication, if required  
            model_name: Name of the model to use
            endpoint_type: Type of endpoint to initialize
            
        Returns:
            tuple: (endpoint_url, api_key, handler, queue, batch_size)
        """
        # Create the appropriate endpoint handler based on type
        if endpoint_type == "chat":
            endpoint_handler = self.create_remote_vllm_chat_endpoint_handler(endpoint_url, api_key, model_name)
        elif endpoint_type == "streaming":
            endpoint_handler = self.create_remote_vllm_streaming_endpoint_handler(endpoint_url, api_key, model_name)
        else:
            endpoint_handler = self.create_remote_vllm_endpoint_handler(endpoint_url, api_key, model_name)
        
        # Register the endpoint
        if model_name not in self.endpoints:
            self.endpoints[model_name] = []
        
        if endpoint_url not in self.endpoints[model_name]:
            self.endpoints[model_name].append(endpoint_url)
            self.endpoint_status[endpoint_url] = 32  # Default batch size
            
            # Register model in the registered_models dictionary
            if model_name not in self.registered_models:
                self.registered_models[model_name] = {
                    "endpoints": [endpoint_url],
                    "types": [endpoint_type]
                }
            else:
                if endpoint_url not in self.registered_models[model_name]["endpoints"]:
                    self.registered_models[model_name]["endpoints"].append(endpoint_url)
                if endpoint_type not in self.registered_models[model_name]["types"]:
                    self.registered_models[model_name]["types"].append(endpoint_type)
        
        return endpoint_url, api_key, endpoint_handler, self.request_queue, self.endpoint_status[endpoint_url]
    
    def list_available_vllm_models(self, endpoint_url=None, api_key=None):
        """List available models from an vLLM endpoint
        
        The endpoint should implement the /models API endpoint that returns
        available models in one of these formats:
        - {"models": [...]}
        - {"data": [...]}
        
        Args:
            endpoint_url: URL of the vLLM endpoint
            api_key: API key for authentication, if required
            
        Returns:
            list: List of available models or None if request fails
        """
        if not endpoint_url:
            if self.endpoints:
                # Use the first available endpoint
                model_name = next(iter(self.endpoints))
                endpoint_url = self.endpoints[model_name][0]
            else:
                return None
                
        try:
            models_endpoint = f"{endpoint_url.rstrip('/')}/models"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.get(models_endpoint, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Different LLVM implementations might structure the response differently
            if "models" in result:
                return result["models"]
            elif "data" in result:
                return result["data"]
            else:
                return result
        except Exception as e:
            print(f"Failed to list vLLM models: {e}")
            return None
    
    def __test__(self, endpoint_url, endpoint_handler, endpoint_label, api_key=None, endpoint_type="completion"):
        """Test the remote vLLM endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            endpoint_handler: The handler function
            endpoint_label: Label for the endpoint
            api_key: API key for authentication
            endpoint_type: Type of endpoint to test ("completion", "chat", "streaming")
            
        Returns:
            bool: True if test passes, False otherwise
        """
        if endpoint_type == "chat":
            test_messages = [{"role": "user", "content": "Complete this sentence: The quick brown fox"}]
            try:
                result = endpoint_handler(test_messages)
                if result is not None:
                    print(f"Remote vLLM chat test passed for {endpoint_label}")
                    return True
                else:
                    print(f"Remote vLLM chat test failed for {endpoint_label}: No result")
                    return False
            except Exception as e:
                print(f"Remote vLLM chat test failed for {endpoint_label}: {e}")
                return False
        elif endpoint_type == "streaming":
            test_prompt = "Complete this sentence: The quick brown fox"
            try:
                response_stream = endpoint_handler(test_prompt)
                # Check if we can get at least one chunk
                first_chunk = next(response_stream, None)
                if first_chunk is not None:
                    print(f"Remote vLLM streaming test passed for {endpoint_label}")
                    # Consume the rest of the stream to clean up
                    for _ in response_stream:
                        pass
                    return True
                else:
                    print(f"Remote vLLM streaming test failed for {endpoint_label}: No response chunks")
                    return False
            except Exception as e:
                print(f"Remote vLLM streaming test failed for {endpoint_label}: {e}")
                return False
        else:
            # Standard completion test
            test_prompt = "Complete this sentence: The quick brown fox"
            try:
                result = endpoint_handler(test_prompt)
                if result is not None:
                    print(f"Remote vLLM completion test passed for {endpoint_label}")
                    return True
                else:
                    print(f"Remote vLLM completion test failed for {endpoint_label}: No result")
                    return False
            except Exception as e:
                print(f"Remote vLLM completion test failed for {endpoint_label}: {e}")
                return False
    
    def make_post_request_vllm(self, endpoint_url, data, api_key=None):
        """Make a POST request to a remote vLLM endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication, if required
            
        Returns:
            dict: Response from the endpoint
            
        Raises:
            ConnectionError: If connection to endpoint fails
            ValueError: If endpoint returns an error code
            RuntimeError: If response cannot be parsed as JSON
        """
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            try:
                response = requests.post(endpoint_url, headers=headers, json=data)
            except requests.ConnectionError as e:
                print(f"Connection error to vLLM endpoint: {e}")
                raise ConnectionError(f"Failed to connect to vLLM endpoint: {e}")
            
            # Handle specific error status codes
            if response.status_code >= 400:
                error_msg = f"vLLM API error: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg} - {error_data['error']}"
                except:
                    pass
                print(error_msg)
                raise ValueError(error_msg)
            
            try:
                return response.json()
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                raise RuntimeError(f"Invalid JSON response from LLVM: {e}")
        
        except (ConnectionError, ValueError, RuntimeError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            print(f"Error making request to vLLM endpoint: {e}")
            raise ValueError(f"vLLM API request failed: {str(e)}")
    
    async def make_async_post_request_vllm(self, endpoint_url, data, api_key=None):
        """Make an asynchronous POST request to a remote vLLM endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication, if required
            
        Returns:
            dict: Response from the endpoint
        """
        import aiohttp
        from aiohttp import ClientSession, ClientTimeout
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        timeout = ClientTimeout(total=300)
        
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.post(endpoint_url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Error {response.status}: {error_text}")
                    return await response.json()
                    
        except Exception as e:
            print(f"Error in async request to vLLM endpoint: {e}")
            raise e
    
    def make_streaming_request_vllm(self, endpoint_url, data, api_key=None):
        """Make a streaming request to a remote vLLM endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication, if required
            
        Returns:
            generator: A generator yielding response chunks
        """
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Add streaming parameter
            data["stream"] = True
            
            response = requests.post(endpoint_url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            
            # Handle different streaming formats
            for line in response.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        # Server-sent events format
                        line = line[6:]  # Remove "data: " prefix
                    if line == b"[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(line)
                        yield chunk
                    except json.JSONDecodeError:
                        # Plain text format
                        yield {"text": line.decode("utf-8")}
        
        except Exception as e:
            print(f"Error in streaming request to vLLM endpoint: {e}")
            yield {"error": str(e)}
    
    def test_vllm_endpoint(self, endpoint_url=None, api_key=None, model_name=None, endpoint_type="completion"):
        """Test an vLLM endpoint
        
        Args:
            endpoint_url: URL of the endpoint to test
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            endpoint_type: Type of endpoint to test ("completion", "chat", "streaming")
            
        Returns:
            bool: True if test passes, False otherwise
        """
        if endpoint_type == "chat":
            try:
                data = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "Complete this sentence: The quick brown fox"}
                    ],
                    "max_tokens": 20,
                    "temperature": 0.7
                }
                    
                result = self.make_post_request_vllm(endpoint_url, data, api_key)
                
                if result and ("choices" in result or "message" in result or "response" in result):
                    return True
                else:
                    print(f"Chat test failed for endpoint {endpoint_url}: Invalid response format")
                    return False
                    
            except Exception as e:
                print(f"Chat test failed for endpoint {endpoint_url}: {e}")
                return False
        elif endpoint_type == "streaming":
            try:
                data = {
                    "model": model_name,
                    "prompt": "Complete this sentence: The quick brown fox",
                    "max_tokens": 20,
                    "temperature": 0.7,
                    "stream": True
                }
                
                # Just try to get the first chunk to validate
                first_chunk = next(self.make_streaming_request_vllm(endpoint_url, data, api_key), None)
                if first_chunk is not None and not "error" in first_chunk:
                    return True
                else:
                    print(f"Streaming test failed for endpoint {endpoint_url}")
                    return False
            except Exception as e:
                print(f"Streaming test failed for endpoint {endpoint_url}: {e}")
                return False
        else:
            # Standard completion test
            try:
                data = {
                    "model": model_name,
                    "prompt": "Complete this sentence: The quick brown fox",
                    "max_tokens": 20,
                    "temperature": 0.7
                }
                    
                result = self.make_post_request_vllm(endpoint_url, data, api_key)
                
                if result and ("text" in result or "choices" in result or "response" in result):
                    return True
                else:
                    print(f"Completion test failed for endpoint {endpoint_url}: Invalid response format")
                    return False
                    
            except Exception as e:
                print(f"Completion test failed for endpoint {endpoint_url}: {e}")
                return False
    
    def create_vllm_endpoint_handler(self, model=None, endpoint=None, endpoint_type="completion", batch=None):
        """Create an endpoint handler for vLLM
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            endpoint_type: Type of endpoint ("completion", "chat", "streaming")
            batch: Batch size
            
        Returns:
            function: Handler for the endpoint
        """
        if endpoint_type == "chat":
            return self.create_remote_vllm_chat_endpoint_handler(endpoint, None, model)
        elif endpoint_type == "streaming":
            return self.create_remote_vllm_streaming_endpoint_handler(endpoint, None, model)
        else:
            return self.create_remote_vllm_endpoint_handler(endpoint, None, model)
    
    def create_vllm_chat_endpoint_handler(self, model=None, endpoint=None, batch=None):
        """Create a chat endpoint handler for vLLM
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            batch: Batch size
            
        Returns:
            function: Handler for the chat endpoint
        """
        return self.create_remote_vllm_chat_endpoint_handler(endpoint, None, model)
    
    def create_vllm_streaming_endpoint_handler(self, model=None, endpoint=None, batch=None):
        """Create a streaming endpoint handler for vLLM
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            batch: Batch size
            
        Returns:
            function: Handler for the streaming endpoint
        """
        return self.create_remote_vllm_streaming_endpoint_handler(endpoint, None, model)
    
    def request_vllm_endpoint(self, model, endpoint=None, endpoint_type="completion", batch=None):
        """Request an vLLM endpoint
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL (optional)
            endpoint_type: Type of endpoint (optional)
            batch: Batch size (optional)
            
        Returns:
            str: URL of the selected endpoint
        """
        incoming_batch_size = len(batch) if batch else 1
        
        # If endpoint is specified and has sufficient capacity, use it
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
            if incoming_batch_size <= endpoint_batch_size:
                return endpoint
        
        # Check in endpoints dictionary
        if model in self.endpoints:
            for e in self.endpoints[model]:
                # Check if endpoint has the requested type
                if model in self.registered_models:
                    endpoint_idx = self.registered_models[model]["endpoints"].index(e)
                    if endpoint_idx >= 0 and endpoint_idx < len(self.registered_models[model]["types"]):
                        if endpoint_type == self.registered_models[model]["types"][endpoint_idx]:
                            if e in self.endpoint_status and self.endpoint_status[e] >= incoming_batch_size:
                                return e
                # If no type match or type info not available, just check batch size
                elif e in self.endpoint_status and self.endpoint_status[e] >= incoming_batch_size:
                    return e
        
        # No suitable endpoint found
        return None
    
    def create_remote_vllm_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for a remote vLLM completion endpoint
        
        Example:
            ```python
            handler = vllm.create_remote_vllm_endpoint_handler(
                endpoint_url="http://localhost:8080",
                model_name="my-model"
            )
            
            # Basic usage
            response = handler("What is machine learning?")
            
            # With parameters
            response = handler(
                "What is machine learning?",
                parameters={
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["\n", "END"]
                }
            )
            ```
        
        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            
        Returns:
            function: Handler for the endpoint that takes (prompt, parameters=None)
        """
        def handler(prompt, parameters=None, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                default_params = {
                    "max_tokens": 128,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stop": None
                }
                
                # Update with any provided parameters
                if parameters:
                    default_params.update(parameters)
                
                # Prepare request data
                data = {
                    "model": model_name,
                    "prompt": prompt
                }
                
                # Add parameters to request
                data.update(default_params)
                
                # Make the request
                response = self.make_post_request_vllm(endpoint_url, data, api_key)
                
                if response:
                    # Handle different possible response formats
                    if "text" in response:
                        return response["text"]
                    elif "choices" in response and len(response["choices"]) > 0:
                        if "text" in response["choices"][0]:
                            return response["choices"][0]["text"]
                        elif "message" in response["choices"][0]:
                            return response["choices"][0]["message"]["content"]
                    elif "response" in response:
                        return response["response"]
                    else:
                        return response
                
                return None
            
            except Exception as e:
                print(f"Error in vLLM completion handler: {e}")
                return None
        
        return handler
    
    def create_remote_vllm_chat_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for chat completion with structured messages
        
        Example:
            ```python
            handler = vllm.create_remote_vllm_chat_endpoint_handler(
                endpoint_url="http://localhost:8080", 
                model_name="my-chat-model"
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "What about Germany?"}
            ]
            
            response = handler(messages)
            
            # With parameters
            response = handler(
                messages,
                parameters={
                    "temperature": 0.8,
                    "max_tokens": 200
                }
            )
            ```
        
        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            
        Returns:
            function: Handler that takes (messages, parameters=None)
        """
        def handler(messages, parameters=None, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                default_params = {
                    "max_tokens": 128,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stop": None
                }
                
                # Update with any provided parameters
                if parameters:
                    default_params.update(parameters)
                
                # Prepare request data
                data = {
                    "model": model_name,
                    "messages": messages
                }
                
                # Add parameters to request
                for key, value in default_params.items():
                    data[key] = value
                
                # Make the request
                response = self.make_post_request_vllm(endpoint_url, data, api_key)
                
                if response:
                    # Handle different possible response formats
                    if "choices" in response and len(response["choices"]) > 0:
                        if "message" in response["choices"][0]:
                            return response["choices"][0]["message"]["content"]
                        elif "text" in response["choices"][0]:
                            return response["choices"][0]["text"]
                    elif "message" in response:
                        return response["message"]["content"]
                    elif "response" in response:
                        return response["response"]
                    else:
                        return response
                
                return None
            
            except Exception as e:
                print(f"Error in vLLM chat handler: {e}")
                return None
        
        return handler
    
    def format_request(self, handler, input_data, **kwargs):
        """Format a request for the LLVM inference API
        
        Args:
            handler: The endpoint handler function
            input_data: The input data to process (string, list, or dict)
            **kwargs: Additional parameters for inference
            
        Returns:
            Any: The response from the handler
        """
        try:
            # Prepare the parameters
            parameters = {}
            
            # Extract inference parameters from kwargs
            for param in ['max_tokens', 'temperature', 'top_p', 'top_k', 
                         'repetition_penalty', 'stop', 'batch_size',
                         'precision', 'seed']:
                if param in kwargs:
                    parameters[param] = kwargs[param]
            
            # Format the request data based on input type
            if isinstance(input_data, str):
                # Simple text input
                data = {
                    "input": input_data
                }
                if parameters:
                    data["parameters"] = parameters
                    
            elif isinstance(input_data, list):
                # Batch input
                if all(isinstance(item, str) for item in input_data):
                    data = {
                        "inputs": input_data
                    }
                    if parameters:
                        data["parameters"] = parameters
                else:
                    # List of complex objects, pass as is
                    data = input_data
                    
            elif isinstance(input_data, dict):
                # Dictionary input, use as is
                data = input_data
                
                # Add parameters if they don't exist
                if parameters and "parameters" not in data:
                    data["parameters"] = parameters
                elif parameters and "parameters" in data:
                    # Merge parameters
                    data["parameters"].update(parameters)
            else:
                # Unknown input type
                raise ValueError(f"Unsupported input type: {type(input_data)}")
                
            # Call the handler with the formatted data
            return handler(data)
            
        except Exception as e:
            print(f"Error formatting LLVM request: {e}")
            return None
            
    def process_batch(self, inputs, model=None, endpoint=None, parameters=None):
        """Process a batch of inputs
        
        Args:
            inputs: List of input strings or objects
            model: Name of the model to use (optional)
            endpoint: URL of the endpoint (optional)
            parameters: Additional parameters (optional)
            
        Returns:
            list: List of outputs corresponding to inputs
        """
        try:
            # Get the endpoint URL
            endpoint_url = self.request_vllm_endpoint(model, endpoint, "completion", inputs)
            if not endpoint_url:
                raise ValueError(f"No vLLM endpoint available for batch processing")
                
            # Prepare the request data
            data = {
                "inputs": inputs
            }
            
            if model:
                data["model"] = model
                
            if parameters:
                data["parameters"] = parameters
                
            # Make the batch request
            response = self.make_post_request_vllm(endpoint_url, data)
            
            # Extract results from the response
            if isinstance(response, dict):
                if "results" in response:
                    return response["results"]
                elif "outputs" in response:
                    return response["outputs"]
                elif "choices" in response and isinstance(response["choices"], list):
                    return [choice.get("text") for choice in response["choices"]]
                    
            # Fallback: return the raw response
            return response
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return None
            
    def get_model_info(self, endpoint_url=None, model_name=None, api_key=None):
        """Get information about a specific model
        
        Args:
            endpoint_url: URL of the endpoint (optional)
            model_name: Name of the model (optional)
            api_key: API key for authentication (optional)
            
        Returns:
            dict: Model information or None if not available
        """
        try:
            # Determine the endpoint URL
            if not endpoint_url:
                if model_name and model_name in self.endpoints and self.endpoints[model_name]:
                    endpoint_url = self.endpoints[model_name][0]
                elif self.endpoints:
                    # Use the first available endpoint
                    first_model = next(iter(self.endpoints))
                    endpoint_url = self.endpoints[first_model][0]
                else:
                    return None
                    
            # Create the model info endpoint URL
            if model_name:
                model_info_url = f"{endpoint_url.rstrip('/')}/models/{model_name}"
            else:
                model_info_url = f"{endpoint_url.rstrip('/')}/models"
                
            # Set up headers
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                
            # Make the request
            response = requests.get(model_info_url, headers=headers)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None
            
    def create_remote_vllm_streaming_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for streaming responses
        
        Example:
            ```python
            handler = vllm.create_remote_vllm_streaming_endpoint_handler(
                endpoint_url="http://localhost:8080",
                model_name="my-model"
            )
            
            # Stream completion
            for chunk in handler("Write a story about:"):
                if "text" in chunk:
                    print(chunk["text"], end="")
                    
            # Stream chat
            messages = [
                {"role": "user", "content": "Write a poem about:"}
            ]
            
            for chunk in handler(messages):
                if "delta" in chunk:
                    print(chunk["delta"], end="")
            ```
        
        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            
        Returns:
            function: Handler that takes (prompt_or_messages, parameters=None)
            and yields response chunks
        """
        def handler(prompt_or_messages, parameters=None, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                default_params = {
                    "max_tokens": 128,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stream": True
                }
                
                # Update with any provided parameters
                if parameters:
                    default_params.update(parameters)
                    
                # Prepare request data based on input type
                data = {
                    "model": model_name,
                    "stream": True
                }
                
                # Check if input is for a chat model
                if isinstance(prompt_or_messages, list) and all(isinstance(msg, dict) for msg in prompt_or_messages):
                    data["messages"] = prompt_or_messages
                else:
                    data["prompt"] = prompt_or_messages
                
                # Add parameters to request
                for key, value in default_params.items():
                    if key != "stream":  # Stream is already set
                        data[key] = value
                
                # Make the streaming request
                return self.make_streaming_request_vllm(endpoint_url, data, api_key)
            
            except Exception as e:
                print(f"Error in vLLM streaming handler: {e}")
                yield {"error": str(e)}
        
        return handler
    # ------------------------------------------------------------------
    # Standard inference interface (shared code path with other backends)
    # ------------------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Optional[str]:
        """Generate text from a prompt via vLLM /v1/completions.

        Args:
            prompt: Input text prompt.
            model: Model name served by the vLLM server.
            endpoint_url: Base URL of the vLLM server (defaults to VLLM_API_URL).
            api_key: Optional bearer token.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional parameters forwarded to the endpoint.

        Returns:
            Generated text string, or None on failure.
        """
        url = (endpoint_url or self.default_api_url).rstrip("/") + "/v1/completions"
        key = api_key or self.default_api_key
        data: Dict[str, Any] = {
            "model": model or "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data.update(kwargs)
        try:
            response = self.make_post_request_vllm(url, data, key)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                return choice.get("text") or (choice.get("message") or {}).get("content")
            if response and "text" in response:
                return response["text"]
        except Exception as exc:
            logger.error("vLLM generate_text failed: %s", exc)
        return None

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Optional[str]:
        """Chat completion via vLLM /v1/chat/completions.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model name served by the vLLM server.
            endpoint_url: Base URL of the vLLM server.
            api_key: Optional bearer token.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional parameters forwarded to the endpoint.

        Returns:
            Assistant reply text, or None on failure.
        """
        url = (endpoint_url or self.default_api_url).rstrip("/") + "/v1/chat/completions"
        key = api_key or self.default_api_key
        data: Dict[str, Any] = {
            "model": model or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data.update(kwargs)
        try:
            response = self.make_post_request_vllm(url, data, key)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                msg = choice.get("message") or {}
                return msg.get("content") or choice.get("text")
        except Exception as exc:
            logger.error("vLLM chat_completion failed: %s", exc)
        return None

    def list_models(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List models available on the vLLM server via /v1/models.

        Returns:
            List of model info dicts, or empty list on failure.
        """
        url = (endpoint_url or self.default_api_url).rstrip("/") + "/v1/models"
        key = api_key or self.default_api_key
        try:
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"******"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if "data" in data:
                return data["data"]
            if "models" in data:
                return data["models"]
            return [data]
        except Exception as exc:
            logger.error("vLLM list_models failed: %s", exc)
            return []
