import asyncio
import json
import requests
import os
from typing import Dict, List, Optional, Union, Any, Callable

class ollama:
    """Ollama API Backend Integration
    
    This class provides a comprehensive interface for interacting with Ollama model endpoints.
    It supports all major Ollama API features including:
    - Text completion generation
    - Chat completions
    - Streaming responses
    - Text embeddings
    - Model management
    
    Features:
    - Dynamic endpoint management
    - Multiple response types (completion, chat, streaming, embeddings)
    - Async and sync request options
    - Request queueing and batch processing
    - Model listing and status tracking
    
    The handler methods follow these patterns:
    - completion: Takes a prompt, returns generated text
    - chat: Takes message list, returns assistant response
    - streaming: Takes prompt/messages, yields response chunks
    - embedding: Takes text, returns vector embedding
    """

    def __init__(self, resources=None, metadata=None):
        """Initialize Ollama backend interface
        
        Args:
            resources: Resources configuration dictionary
            metadata: Additional metadata dictionary
        """
        self.resources = resources
        self.metadata = metadata
        # Register method references
        self.create_remote_ollama_endpoint_handler = self.create_remote_ollama_endpoint_handler
        self.create_remote_ollama_chat_endpoint_handler = self.create_remote_ollama_chat_endpoint_handler
        self.create_remote_ollama_streaming_endpoint_handler = self.create_remote_ollama_streaming_endpoint_handler
        self.create_remote_ollama_embedding_endpoint_handler = self.create_remote_ollama_embedding_endpoint_handler
        self.request_ollama_endpoint = self.request_ollama_endpoint
        self.test_ollama_endpoint = self.test_ollama_endpoint
        self.make_post_request_ollama = self.make_post_request_ollama
        self.make_async_post_request_ollama = self.make_async_post_request_ollama
        self.create_ollama_endpoint_handler = self.create_ollama_endpoint_handler
        self.create_ollama_chat_endpoint_handler = self.create_ollama_chat_endpoint_handler
        self.create_ollama_streaming_endpoint_handler = self.create_ollama_streaming_endpoint_handler
        self.create_ollama_embedding_endpoint_handler = self.create_ollama_embedding_endpoint_handler
        self.list_available_ollama_models = self.list_available_ollama_models
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
        """Initialize a connection to an Ollama endpoint
        
        Supported endpoint_types:
        - "completion": Standard text completion
        - "chat": Structured chat completion
        - "streaming": Stream responses chunk by chunk
        - "embedding": Text embedding generation
        
        Args:
            endpoint_url: The URL of the Ollama endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            endpoint_type: Type of endpoint to initialize
            
        Returns:
            tuple: (endpoint_url, api_key, handler, queue, batch_size)
        """
        # Create the appropriate endpoint handler based on type
        if endpoint_type == "chat":
            endpoint_handler = self.create_remote_ollama_chat_endpoint_handler(endpoint_url, api_key, model_name)
        elif endpoint_type == "streaming":
            endpoint_handler = self.create_remote_ollama_streaming_endpoint_handler(endpoint_url, api_key, model_name)
        elif endpoint_type == "embedding":
            endpoint_handler = self.create_remote_ollama_embedding_endpoint_handler(endpoint_url, api_key, model_name)
        else:
            endpoint_handler = self.create_remote_ollama_endpoint_handler(endpoint_url, api_key, model_name)
        
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
                if endpoint_type not in self.registered_models[model_name].get("types", []):
                    if "types" not in self.registered_models[model_name]:
                        self.registered_models[model_name]["types"] = []
                    self.registered_models[model_name]["types"].append(endpoint_type)
        
        return endpoint_url, api_key, endpoint_handler, self.request_queue, self.endpoint_status[endpoint_url]
    
    def list_available_ollama_models(self, endpoint_url=None, api_key=None):
        """List available models from an Ollama endpoint
        
        Connects to the Ollama /api/tags endpoint to get available models.
        
        Args:
            endpoint_url: URL of the Ollama endpoint
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
            # Ollama uses /api/tags endpoint to list models
            models_endpoint = f"{endpoint_url.rstrip('/')}/api/tags"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.get(models_endpoint, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Ollama returns models under the "models" key
            if "models" in result:
                return result["models"]
            else:
                return result
        except Exception as e:
            print(f"Failed to list Ollama models: {e}")
            return None
    
    def __test__(self, endpoint_url, endpoint_handler, endpoint_label, api_key=None, endpoint_type="completion"):
        """Test the remote Ollama endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            endpoint_handler: The handler function
            endpoint_label: Label for the endpoint
            api_key: API key for authentication
            endpoint_type: Type of endpoint to test
            
        Returns:
            bool: True if test passes, False otherwise
        """
        if endpoint_type == "chat":
            test_messages = [{"role": "user", "content": "Complete this sentence: The quick brown fox"}]
            try:
                result = endpoint_handler(test_messages)
                if result is not None:
                    print(f"Remote Ollama chat test passed for {endpoint_label}")
                    return True
                else:
                    print(f"Remote Ollama chat test failed for {endpoint_label}: No result")
                    return False
            except Exception as e:
                print(f"Remote Ollama chat test failed for {endpoint_label}: {e}")
                return False
        elif endpoint_type == "streaming":
            test_prompt = "Complete this sentence: The quick brown fox"
            try:
                response_stream = endpoint_handler(test_prompt)
                # Check if we can get at least one chunk
                first_chunk = next(response_stream, None)
                if first_chunk is not None:
                    print(f"Remote Ollama streaming test passed for {endpoint_label}")
                    # Consume the rest of the stream to clean up
                    for _ in response_stream:
                        pass
                    return True
                else:
                    print(f"Remote Ollama streaming test failed for {endpoint_label}: No response chunks")
                    return False
            except Exception as e:
                print(f"Remote Ollama streaming test failed for {endpoint_label}: {e}")
                return False
        elif endpoint_type == "embedding":
            test_text = "The quick brown fox jumps over the lazy dog"
            try:
                result = endpoint_handler(test_text)
                if result is not None and isinstance(result, list):
                    print(f"Remote Ollama embedding test passed for {endpoint_label}")
                    return True
                else:
                    print(f"Remote Ollama embedding test failed for {endpoint_label}: Invalid result format")
                    return False
            except Exception as e:
                print(f"Remote Ollama embedding test failed for {endpoint_label}: {e}")
                return False
        else:
            # Standard completion test
            test_prompt = "Complete this sentence: The quick brown fox"
            try:
                result = endpoint_handler(test_prompt)
                if result is not None:
                    print(f"Remote Ollama completion test passed for {endpoint_label}")
                    return True
                else:
                    print(f"Remote Ollama completion test failed for {endpoint_label}: No result")
                    return False
            except Exception as e:
                print(f"Remote Ollama completion test failed for {endpoint_label}: {e}")
                return False
    
    def make_post_request_ollama(self, endpoint_url, data, api_key=None):
        """Make a POST request to a remote Ollama endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication, if required
            
        Returns:
            dict: Response from the endpoint
        """
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(endpoint_url, headers=headers, json=data)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            print(f"Error making request to Ollama endpoint: {e}")
            return None
    
    async def make_async_post_request_ollama(self, endpoint_url, data, api_key=None):
        """Make an asynchronous POST request to a remote Ollama endpoint
        
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
            print(f"Error in async request to Ollama endpoint: {e}")
            raise e
    
    def make_streaming_request_ollama(self, endpoint_url, data, api_key=None):
        """Make a streaming request to a remote Ollama endpoint
        
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
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        yield chunk
                        # Check if this is the final response
                        if "done" in chunk and chunk["done"]:
                            break
                    except json.JSONDecodeError:
                        yield {"text": line.decode("utf-8")}
        
        except Exception as e:
            print(f"Error in streaming request to Ollama endpoint: {e}")
            yield {"error": str(e)}
    
    def test_ollama_endpoint(self, endpoint_url=None, api_key=None, model_name=None, endpoint_type="completion"):
        """Test an Ollama endpoint
        
        Args:
            endpoint_url: URL of the endpoint to test
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            endpoint_type: Type of endpoint to test ("completion", "chat", "streaming", "embedding")
            
        Returns:
            bool: True if test passes, False otherwise
        """
        if endpoint_type == "chat":
            try:
                chat_endpoint = f"{endpoint_url.rstrip('/')}/api/chat"
                data = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "Complete this sentence: The quick brown fox"}
                    ],
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 20
                    }
                }
                    
                result = self.make_post_request_ollama(chat_endpoint, data, api_key)
                
                if result and "message" in result:
                    return True
                else:
                    print(f"Chat test failed for endpoint {endpoint_url}: Invalid response format")
                    return False
                    
            except Exception as e:
                print(f"Chat test failed for endpoint {endpoint_url}: {e}")
                return False
        elif endpoint_type == "streaming":
            try:
                generate_endpoint = f"{endpoint_url.rstrip('/')}/api/generate"
                data = {
                    "model": model_name,
                    "prompt": "Complete this sentence: The quick brown fox",
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 20
                    },
                    "stream": True
                }
                
                # Just try to get the first chunk to validate
                first_chunk = next(self.make_streaming_request_ollama(generate_endpoint, data, api_key), None)
                if first_chunk is not None and "error" not in first_chunk:
                    return True
                else:
                    print(f"Streaming test failed for endpoint {endpoint_url}")
                    return False
            except Exception as e:
                print(f"Streaming test failed for endpoint {endpoint_url}: {e}")
                return False
        elif endpoint_type == "embedding":
            try:
                embedding_endpoint = f"{endpoint_url.rstrip('/')}/api/embeddings"
                data = {
                    "model": model_name,
                    "prompt": "The quick brown fox jumps over the lazy dog"
                }
                
                result = self.make_post_request_ollama(embedding_endpoint, data, api_key)
                
                if result and "embedding" in result:
                    return True
                else:
                    print(f"Embedding test failed for endpoint {endpoint_url}: Invalid response format")
                    return False
            except Exception as e:
                print(f"Embedding test failed for endpoint {endpoint_url}: {e}")
                return False
        else:
            # Standard completion test
            try:
                generate_endpoint = f"{endpoint_url.rstrip('/')}/api/generate"
                data = {
                    "model": model_name,
                    "prompt": "Complete this sentence: The quick brown fox",
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 20
                    }
                }
                    
                result = self.make_post_request_ollama(generate_endpoint, data, api_key)
                
                if result and "response" in result:
                    return True
                else:
                    print(f"Completion test failed for endpoint {endpoint_url}: Invalid response format")
                    return False
                    
            except Exception as e:
                print(f"Completion test failed for endpoint {endpoint_url}: {e}")
                return False
    
    def create_ollama_endpoint_handler(self, model=None, endpoint=None, endpoint_type="completion", batch=None):
        """Create an endpoint handler for Ollama
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            endpoint_type: Type of endpoint ("completion", "chat", "streaming", "embedding")
            batch: Batch size
            
        Returns:
            function: Handler for the endpoint
        """
        if endpoint_type == "chat":
            return self.create_remote_ollama_chat_endpoint_handler(endpoint, None, model)
        elif endpoint_type == "streaming":
            return self.create_remote_ollama_streaming_endpoint_handler(endpoint, None, model)
        elif endpoint_type == "embedding":
            return self.create_remote_ollama_embedding_endpoint_handler(endpoint, None, model)
        else:
            return self.create_remote_ollama_endpoint_handler(endpoint, None, model)
    
    def create_ollama_chat_endpoint_handler(self, model=None, endpoint=None, batch=None):
        """Create a chat endpoint handler for Ollama
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            batch: Batch size
            
        Returns:
            function: Handler for the chat endpoint
        """
        return self.create_remote_ollama_chat_endpoint_handler(endpoint, None, model)
    
    def create_ollama_streaming_endpoint_handler(self, model=None, endpoint=None, batch=None):
        """Create a streaming endpoint handler for Ollama
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            batch: Batch size
            
        Returns:
            function: Handler for the streaming endpoint
        """
        return self.create_remote_ollama_streaming_endpoint_handler(endpoint, None, model)
    
    def create_ollama_embedding_endpoint_handler(self, model=None, endpoint=None, batch=None):
        """Create an embedding endpoint handler for Ollama
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            batch: Batch size
            
        Returns:
            function: Handler for the embedding endpoint
        """
        return self.create_remote_ollama_embedding_endpoint_handler(endpoint, None, model)
    
    def request_ollama_endpoint(self, model, endpoint=None, endpoint_type="completion", batch=None):
        """Request an Ollama endpoint
        
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
                    if endpoint_idx >= 0 and "types" in self.registered_models[model]:
                        types_list = self.registered_models[model]["types"]
                        if endpoint_idx < len(types_list) and endpoint_type == types_list[endpoint_idx]:
                            if e in self.endpoint_status and self.endpoint_status[e] >= incoming_batch_size:
                                return e
                # If no type match or type info not available, just check batch size
                elif e in self.endpoint_status and self.endpoint_status[e] >= incoming_batch_size:
                    return e
        
        # No suitable endpoint found
        return None
    
    def create_remote_ollama_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for text completion with Ollama
        
        Example:
            ```python
            handler = ollama.create_remote_ollama_endpoint_handler(
                endpoint_url="http://localhost:11434",
                model_name="llama2"
            )
            
            # Basic usage
            response = handler("Explain quantum computing")
            
            # With parameters
            response = handler(
                "Explain quantum computing",
                parameters={
                    "num_predict": 100,
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9,
                    "stop": ["END"],
                    "seed": 42
                }
            )
            ```
        
        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            
        Returns:
            function: Handler for completions that takes (prompt, parameters=None)
        """
        def handler(prompt, parameters=None, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                generate_endpoint = f"{endpoint_url.rstrip('/')}/api/generate"
                
                default_options = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "num_predict": 128,
                    "seed": 0
                }
                
                # Update with any provided parameters
                options = default_options.copy()
                if parameters:
                    if "options" in parameters:
                        options.update(parameters["options"])
                    
                    # Map OpenAI-style parameters to Ollama options
                    if "max_tokens" in parameters:
                        options["num_predict"] = parameters["max_tokens"]
                    if "temperature" in parameters:
                        options["temperature"] = parameters["temperature"]
                    if "top_p" in parameters:
                        options["top_p"] = parameters["top_p"]
                    if "frequency_penalty" in parameters:
                        options["frequency_penalty"] = parameters["frequency_penalty"]
                    if "presence_penalty" in parameters:
                        options["presence_penalty"] = parameters["presence_penalty"]
                    if "stop" in parameters and parameters["stop"]:
                        options["stop"] = parameters["stop"]
                
                # Prepare request data
                data = {
                    "model": model_name,
                    "prompt": prompt,
                    "options": options
                }
                
                # Make the request
                response = self.make_post_request_ollama(generate_endpoint, data, api_key)
                
                if response and "response" in response:
                    return response["response"]
                
                return response
            
            except Exception as e:
                print(f"Error in Ollama completion handler: {e}")
                return None
        
        return handler
    
    def create_remote_ollama_chat_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for chat completion with Ollama
        
        Example:
            ```python
            handler = ollama.create_remote_ollama_chat_endpoint_handler(
                endpoint_url="http://localhost:11434",
                model_name="llama2"
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is fusion energy?"},
                {"role": "assistant", "content": "Fusion energy is..."},
                {"role": "user", "content": "What are its advantages?"}
            ]
            
            response = handler(messages)
            
            # With custom parameters
            response = handler(
                messages,
                parameters={
                    "temperature": 0.8,
                    "num_predict": 200,
                    "top_p": 0.95
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
                chat_endpoint = f"{endpoint_url.rstrip('/')}/api/chat"
                
                default_options = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "num_predict": 128,
                    "seed": 0
                }
                
                # Update with any provided parameters
                options = default_options.copy()
                if parameters:
                    if "options" in parameters:
                        options.update(parameters["options"])
                    
                    # Map OpenAI-style parameters to Ollama options
                    if "max_tokens" in parameters:
                        options["num_predict"] = parameters["max_tokens"]
                    if "temperature" in parameters:
                        options["temperature"] = parameters["temperature"]
                    if "top_p" in parameters:
                        options["top_p"] = parameters["top_p"]
                    if "frequency_penalty" in parameters:
                        options["frequency_penalty"] = parameters["frequency_penalty"]
                    if "presence_penalty" in parameters:
                        options["presence_penalty"] = parameters["presence_penalty"]
                    if "stop" in parameters and parameters["stop"]:
                        options["stop"] = parameters["stop"]
                
                # Prepare request data
                data = {
                    "model": model_name,
                    "messages": messages,
                    "options": options
                }
                
                # Make the request
                response = self.make_post_request_ollama(chat_endpoint, data, api_key)
                
                if response and "message" in response:
                    return response["message"]["content"]
                
                return response
            
            except Exception as e:
                print(f"Error in Ollama chat handler: {e}")
                return None
        
        return handler
    
    def create_remote_ollama_streaming_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for streaming responses from Ollama
        
        Example:
            ```python
            handler = ollama.create_remote_ollama_streaming_endpoint_handler(
                endpoint_url="http://localhost:11434",
                model_name="llama2"
            )
            
            # Stream completion
            for chunk in handler("Write a story about space exploration:"):
                print(chunk.get("response", ""), end="")
                
            # Stream chat
            messages = [
                {"role": "user", "content": "Write a poem about nature:"}
            ]
            
            for chunk in handler(messages):
                print(chunk.get("message", {}).get("content", ""), end="")
                
            # With parameters
            for chunk in handler(
                "Explain how rockets work:",
                parameters={
                    "temperature": 0.9,
                    "num_predict": 300
                }
            ):
                print(chunk.get("response", ""), end="")
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
                default_options = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "num_predict": 128,
                    "seed": 0
                }
                
                # Update with any provided parameters
                options = default_options.copy()
                if parameters:
                    if "options" in parameters:
                        options.update(parameters["options"])
                    
                    # Map OpenAI-style parameters to Ollama options
                    if "max_tokens" in parameters:
                        options["num_predict"] = parameters["max_tokens"]
                    if "temperature" in parameters:
                        options["temperature"] = parameters["temperature"]
                    if "top_p" in parameters:
                        options["top_p"] = parameters["top_p"]
                    if "frequency_penalty" in parameters:
                        options["frequency_penalty"] = parameters["frequency_penalty"]
                    if "presence_penalty" in parameters:
                        options["presence_penalty"] = parameters["presence_penalty"]
                    if "stop" in parameters and parameters["stop"]:
                        options["stop"] = parameters["stop"]
                
                # Determine endpoint and structure request based on input type
                if isinstance(prompt_or_messages, list) and all(isinstance(msg, dict) for msg in prompt_or_messages):
                    # This is a chat request
                    endpoint_path = f"{endpoint_url.rstrip('/')}/api/chat"
                    data = {
                        "model": model_name,
                        "messages": prompt_or_messages,
                        "options": options,
                        "stream": True
                    }
                else:
                    # This is a completion request
                    endpoint_path = f"{endpoint_url.rstrip('/')}/api/generate"
                    data = {
                        "model": model_name,
                        "prompt": prompt_or_messages,
                        "options": options,
                        "stream": True
                    }
                
                # Make the streaming request
                return self.make_streaming_request_ollama(endpoint_path, data, api_key)
            
            except Exception as e:
                print(f"Error in Ollama streaming handler: {e}")
                yield {"error": str(e)}
        
        return handler
    
    def create_remote_ollama_embedding_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for generating embeddings with Ollama
        
        Example:
            ```python
            handler = ollama.create_remote_ollama_embedding_endpoint_handler(
                endpoint_url="http://localhost:11434",
                model_name="llama2"
            )
            
            # Get embeddings for a single text
            embedding = handler("This is a sample text")
            
            # Get embeddings for multiple texts
            texts = [
                "First sample text",
                "Second sample text",
                "Third sample text"
            ]
            embeddings = [handler(text) for text in texts]
            
            # Use embeddings for similarity, clustering, etc.
            import numpy as np
            
            # Convert to numpy arrays
            vectors = np.array(embeddings)
            
            # Calculate similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(vectors)
            ```
        
        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            
        Returns:
            function: Handler that takes (text) and returns embedding vector
        """
        def handler(text, parameters=None, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                embedding_endpoint = f"{endpoint_url.rstrip('/')}/api/embeddings"
                
                default_options = {}
                
                # Update with any provided parameters
                options = default_options.copy()
                if parameters and "options" in parameters:
                    options.update(parameters["options"])
                
                # Prepare request data
                data = {
                    "model": model_name,
                    "prompt": text
                }
                
                if options:
                    data["options"] = options
                
                # Make the request
                response = self.make_post_request_ollama(embedding_endpoint, data, api_key)
                
                if response and "embedding" in response:
                    return response["embedding"]
                
                return response
            
            except Exception as e:
                print(f"Error in Ollama embedding handler: {e}")
                return None
        
        return handler

