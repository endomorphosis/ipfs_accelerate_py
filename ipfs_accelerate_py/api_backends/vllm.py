import asyncio
import os
import requests
import json
from typing import Dict, List, Optional, Union, Any, Callable

class llvm:
    """LLVM API Backend Integration
    
    This class provides a comprehensive interface for interacting with LLVM-based model endpoints.
    It supports multiple types of interactions including:
    - Standard completion (text generation)
    - Chat completions
    - Streaming responses
    - Model management
    
    Features:
    - Dynamic endpoint management and status tracking
    - Batched request handling
    - Multiple response formats (completion, chat, streaming)
    - Async and sync request options
    - Request queueing
    
    The handler methods follow these patterns:
    - completion: Takes a prompt string, returns generated text
    - chat: Takes a list of message dictionaries, returns response
    - streaming: Takes a prompt or messages, yields response chunks
    """

    def __init__(self, resources=None, metadata=None):
        """Initialize LLVM backend interface
        
        Args:
            resources: Resources configuration dictionary
            metadata: Additional metadata dictionary
        """
        self.resources = resources
        self.metadata = metadata
        # Register method references
        self.create_remote_llvm_endpoint_handler = self.create_remote_llvm_endpoint_handler
        self.create_remote_llvm_chat_endpoint_handler = self.create_remote_llvm_chat_endpoint_handler
        self.create_remote_llvm_streaming_endpoint_handler = self.create_remote_llvm_streaming_endpoint_handler
        self.request_llvm_endpoint = self.request_llvm_endpoint
        self.test_llvm_endpoint = self.test_llvm_endpoint
        self.make_post_request_llvm = self.make_post_request_llvm
        self.make_async_post_request_llvm = self.make_async_post_request_llvm
        self.create_llvm_endpoint_handler = self.create_llvm_endpoint_handler
        self.create_llvm_chat_endpoint_handler = self.create_llvm_chat_endpoint_handler
        self.create_llvm_streaming_endpoint_handler = self.create_llvm_streaming_endpoint_handler
        self.list_available_llvm_models = self.list_available_llvm_models
        self.init = self.init
        self.__test__ = self.__test__
        # Add endpoints tracking
        self.endpoints = {}
        self.endpoint_status = {}
        self.registered_models = {}
        # Add queue for managing requests
        self.request_queue = asyncio.Queue(64)
        return None
    
    def init(self, endpoint_url=None, api_key=None, model_name=None, endpoint_type="completion"):
        """Initialize a connection to a remote LLVM endpoint
        
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
            endpoint_handler = self.create_remote_llvm_chat_endpoint_handler(endpoint_url, api_key, model_name)
        elif endpoint_type == "streaming":
            endpoint_handler = self.create_remote_llvm_streaming_endpoint_handler(endpoint_url, api_key, model_name)
        else:
            endpoint_handler = self.create_remote_llvm_endpoint_handler(endpoint_url, api_key, model_name)
        
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
    
    def list_available_llvm_models(self, endpoint_url=None, api_key=None):
        """List available models from an LLVM endpoint
        
        The endpoint should implement the /models API endpoint that returns
        available models in one of these formats:
        - {"models": [...]}
        - {"data": [...]}
        
        Args:
            endpoint_url: URL of the LLVM endpoint
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
            print(f"Failed to list LLVM models: {e}")
            return None
    
    def __test__(self, endpoint_url, endpoint_handler, endpoint_label, api_key=None, endpoint_type="completion"):
        """Test the remote LLVM endpoint
        
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
                    print(f"Remote LLVM chat test passed for {endpoint_label}")
                    return True
                else:
                    print(f"Remote LLVM chat test failed for {endpoint_label}: No result")
                    return False
            except Exception as e:
                print(f"Remote LLVM chat test failed for {endpoint_label}: {e}")
                return False
        elif endpoint_type == "streaming":
            test_prompt = "Complete this sentence: The quick brown fox"
            try:
                response_stream = endpoint_handler(test_prompt)
                # Check if we can get at least one chunk
                first_chunk = next(response_stream, None)
                if first_chunk is not None:
                    print(f"Remote LLVM streaming test passed for {endpoint_label}")
                    # Consume the rest of the stream to clean up
                    for _ in response_stream:
                        pass
                    return True
                else:
                    print(f"Remote LLVM streaming test failed for {endpoint_label}: No response chunks")
                    return False
            except Exception as e:
                print(f"Remote LLVM streaming test failed for {endpoint_label}: {e}")
                return False
        else:
            # Standard completion test
            test_prompt = "Complete this sentence: The quick brown fox"
            try:
                result = endpoint_handler(test_prompt)
                if result is not None:
                    print(f"Remote LLVM completion test passed for {endpoint_label}")
                    return True
                else:
                    print(f"Remote LLVM completion test failed for {endpoint_label}: No result")
                    return False
            except Exception as e:
                print(f"Remote LLVM completion test failed for {endpoint_label}: {e}")
                return False
    
    def make_post_request_llvm(self, endpoint_url, data, api_key=None):
        """Make a POST request to a remote LLVM endpoint
        
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
                print(f"Connection error to LLVM endpoint: {e}")
                raise ConnectionError(f"Failed to connect to LLVM endpoint: {e}")
            
            # Handle specific error status codes
            if response.status_code >= 400:
                error_msg = f"LLVM API error: HTTP {response.status_code}"
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
            print(f"Error making request to LLVM endpoint: {e}")
            raise ValueError(f"LLVM API request failed: {str(e)}")
    
    async def make_async_post_request_llvm(self, endpoint_url, data, api_key=None):
        """Make an asynchronous POST request to a remote LLVM endpoint
        
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
            print(f"Error in async request to LLVM endpoint: {e}")
            raise e
    
    def make_streaming_request_llvm(self, endpoint_url, data, api_key=None):
        """Make a streaming request to a remote LLVM endpoint
        
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
            print(f"Error in streaming request to LLVM endpoint: {e}")
            yield {"error": str(e)}
    
    def test_llvm_endpoint(self, endpoint_url=None, api_key=None, model_name=None, endpoint_type="completion"):
        """Test an LLVM endpoint
        
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
                    
                result = self.make_post_request_llvm(endpoint_url, data, api_key)
                
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
                first_chunk = next(self.make_streaming_request_llvm(endpoint_url, data, api_key), None)
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
                    
                result = self.make_post_request_llvm(endpoint_url, data, api_key)
                
                if result and ("text" in result or "choices" in result or "response" in result):
                    return True
                else:
                    print(f"Completion test failed for endpoint {endpoint_url}: Invalid response format")
                    return False
                    
            except Exception as e:
                print(f"Completion test failed for endpoint {endpoint_url}: {e}")
                return False
    
    def create_llvm_endpoint_handler(self, model=None, endpoint=None, endpoint_type="completion", batch=None):
        """Create an endpoint handler for LLVM
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            endpoint_type: Type of endpoint ("completion", "chat", "streaming")
            batch: Batch size
            
        Returns:
            function: Handler for the endpoint
        """
        if endpoint_type == "chat":
            return self.create_remote_llvm_chat_endpoint_handler(endpoint, None, model)
        elif endpoint_type == "streaming":
            return self.create_remote_llvm_streaming_endpoint_handler(endpoint, None, model)
        else:
            return self.create_remote_llvm_endpoint_handler(endpoint, None, model)
    
    def create_llvm_chat_endpoint_handler(self, model=None, endpoint=None, batch=None):
        """Create a chat endpoint handler for LLVM
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            batch: Batch size
            
        Returns:
            function: Handler for the chat endpoint
        """
        return self.create_remote_llvm_chat_endpoint_handler(endpoint, None, model)
    
    def create_llvm_streaming_endpoint_handler(self, model=None, endpoint=None, batch=None):
        """Create a streaming endpoint handler for LLVM
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            batch: Batch size
            
        Returns:
            function: Handler for the streaming endpoint
        """
        return self.create_remote_llvm_streaming_endpoint_handler(endpoint, None, model)
    
    def request_llvm_endpoint(self, model, endpoint=None, endpoint_type="completion", batch=None):
        """Request an LLVM endpoint
        
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
    
    def create_remote_llvm_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for a remote LLVM completion endpoint
        
        Example:
            ```python
            handler = llvm.create_remote_llvm_endpoint_handler(
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
                response = self.make_post_request_llvm(endpoint_url, data, api_key)
                
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
                print(f"Error in LLVM completion handler: {e}")
                return None
        
        return handler
    
    def create_remote_llvm_chat_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for chat completion with structured messages
        
        Example:
            ```python
            handler = llvm.create_remote_llvm_chat_endpoint_handler(
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
                response = self.make_post_request_llvm(endpoint_url, data, api_key)
                
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
                print(f"Error in LLVM chat handler: {e}")
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
            endpoint_url = self.request_llvm_endpoint(model, endpoint, "completion", inputs)
            if not endpoint_url:
                raise ValueError(f"No LLVM endpoint available for batch processing")
                
            # Prepare the request data
            data = {
                "inputs": inputs
            }
            
            if model:
                data["model"] = model
                
            if parameters:
                data["parameters"] = parameters
                
            # Make the batch request
            response = self.make_post_request_llvm(endpoint_url, data)
            
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
            
    def create_remote_llvm_streaming_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for streaming responses
        
        Example:
            ```python
            handler = llvm.create_remote_llvm_streaming_endpoint_handler(
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
                return self.make_streaming_request_llvm(endpoint_url, data, api_key)
            
            except Exception as e:
                print(f"Error in LLVM streaming handler: {e}")
                yield {"error": str(e)}
        
        return handler