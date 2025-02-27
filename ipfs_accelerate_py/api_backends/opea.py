import requests
import json
from typing import Dict, List, Optional, Union, Any, Callable

class opea:
    """Open Proxy for Embeddings and AI (OPEA) API Backend
    
    This class provides integration with OpenAI-compatible proxy servers like OpenAI Proxy.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize OPEA backend interface
        
        Args:
            resources: Resources configuration dictionary
            metadata: Additional metadata dictionary
        """
        self.resources = resources
        self.metadata = metadata
        self.create_opea_endpoint_handler = self.create_opea_endpoint_handler
        self.request_opea_endpoint = self.request_opea_endpoint
        self.test_opea_endpoint = self.test_opea_endpoint
        self.make_post_request_opea = self.make_post_request_opea
        self.make_stream_request_opea = self.make_stream_request_opea
        self.chat = self.chat
        self.stream_chat = self.stream_chat
        
        # Add endpoints tracking
        self.endpoints = {}
        self.endpoint_status = {}
        return None
    
    def make_post_request_opea(self, endpoint, data, endpoint_type=None, batch=None, api_key=None):
        """Make a POST request to an OPEA endpoint
        
        Args:
            endpoint: URL of the endpoint
            data: Data to send in the request
            endpoint_type: Type of endpoint (e.g. "chat", "embeddings")
            batch: Batch information (optional)
            api_key: API key for authentication (optional)
            
        Returns:
            dict: Response from the endpoint
            
        Raises:
            ConnectionError: If connection to endpoint fails
            ValueError: If endpoint returns an error code
        """
        try:
            headers = {"Content-Type": "application/json"}
            
            # Add API key if provided
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            elif self.metadata and "api_key" in self.metadata:
                headers["Authorization"] = f"Bearer {self.metadata['api_key']}"
                
            try:
                response = requests.post(endpoint, headers=headers, json=data)
            except requests.ConnectionError as e:
                raise ConnectionError(f"Failed to connect to OPEA endpoint: {e}")
            
            # Handle error status codes
            if response.status_code >= 400:
                error_message = f"OPEA API error: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_message = f"{error_message} - {error_data['error']}"
                except:
                    pass
                raise ValueError(error_message)
                
            return response.json()
            
        except (ConnectionError, ValueError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            raise ValueError(f"Error in OPEA request: {str(e)}")
    
    def make_stream_request_opea(self, endpoint, data, endpoint_type=None, batch=None, api_key=None):
        """Make a streaming request to an OPEA endpoint
        
        Args:
            endpoint: URL of the endpoint
            data: Data to send in the request
            endpoint_type: Type of endpoint (e.g. "chat")
            batch: Batch information (optional)
            api_key: API key for authentication (optional)
            
        Returns:
            iterator: An iterator over response chunks
            
        Raises:
            ConnectionError: If connection to endpoint fails
            ValueError: If endpoint returns an error code
        """
        try:
            headers = {"Content-Type": "application/json"}
            
            # Add API key if provided
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            elif self.metadata and "api_key" in self.metadata:
                headers["Authorization"] = f"Bearer {self.metadata['api_key']}"
                
            # Ensure streaming is enabled in the request
            data["stream"] = True
                
            try:
                response = requests.post(endpoint, headers=headers, json=data, stream=True)
            except requests.ConnectionError as e:
                raise ConnectionError(f"Failed to connect to OPEA streaming endpoint: {e}")
            
            # Handle error status codes
            if response.status_code >= 400:
                error_message = f"OPEA API error: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_message = f"{error_message} - {error_data['error']}"
                except:
                    pass
                raise ValueError(error_message)
                
            # Process the streaming response (follows OpenAI streaming format)
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    # Skip empty lines and "data: [DONE]" messages
                    if not line.strip() or line == "data: [DONE]":
                        continue
                    
                    # Strip "data: " prefix if present
                    if line.startswith("data: "):
                        line = line[6:]
                        
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"error": f"Invalid JSON in streaming response: {line}"}
            
        except (ConnectionError, ValueError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            yield {"error": f"Error in OPEA streaming request: {str(e)}"}
    
    def test_opea_endpoint(self, endpoint, model=None, api_key=None):
        """Test an OPEA endpoint
        
        Args:
            endpoint: URL of the endpoint to test
            model: Name of the model to use (optional)
            api_key: API key for authentication (optional)
            
        Returns:
            bool: True if test passes, False otherwise
        """
        try:
            # Create a simple test request for chat completions
            data = {
                "model": model or "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "Hello, this is a test message."}
                ],
                "max_tokens": 10
            }
            
            # Make the request
            response = self.make_post_request_opea(endpoint, data, "chat", None, api_key)
            
            # Check if we got a valid response
            valid_response = (
                isinstance(response, dict) and
                "choices" in response and
                len(response["choices"]) > 0 and
                "message" in response["choices"][0]
            )
            
            return valid_response
            
        except Exception as e:
            print(f"Failed to test OPEA endpoint: {e}")
            return False
    
    def request_opea_endpoint(self, model, endpoint=None, endpoint_type=None, batch=None):
        """Request an OPEA endpoint
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL (optional)
            endpoint_type: Type of endpoint (optional)
            batch: Batch size (optional)
            
        Returns:
            str: URL of the selected endpoint
        """
        # If endpoint is specified, use it
        if endpoint:
            return endpoint
            
        # If a default endpoint is in metadata, use it
        if self.metadata and "default_endpoint" in self.metadata:
            return self.metadata["default_endpoint"]
            
        # Check if we have an endpoint for this model
        if model in self.endpoints and self.endpoints[model]:
            return self.endpoints[model][0]
            
        # No suitable endpoint found
        return None
    
    def create_opea_endpoint_handler(self, model=None, endpoint=None, endpoint_type="chat"):
        """Create a handler for an OPEA endpoint
        
        Args:
            model: Name of the model (optional)
            endpoint: URL of the endpoint (optional)
            endpoint_type: Type of endpoint (default: "chat")
            
        Returns:
            function: Handler for the endpoint
        """
        def handler(request, model=model, endpoint=endpoint):
            try:
                # Use the model specified in the request if provided
                if isinstance(request, dict) and "model" in request:
                    model_to_use = request["model"]
                elif model:
                    model_to_use = model
                else:
                    model_to_use = "gpt-3.5-turbo"  # Default model
                    
                # Get the endpoint URL
                endpoint_url = self.request_opea_endpoint(model_to_use, endpoint, endpoint_type)
                if not endpoint_url:
                    raise ValueError(f"No OPEA endpoint available for model {model_to_use}")
                    
                # If the request is a string, convert it to a messages format
                if isinstance(request, str):
                    data = {
                        "model": model_to_use,
                        "messages": [{"role": "user", "content": request}]
                    }
                elif isinstance(request, list) and all(isinstance(msg, dict) for msg in request):
                    # This is already a messages array
                    data = {
                        "model": model_to_use,
                        "messages": request
                    }
                else:
                    # Use the request as-is
                    data = request
                    if isinstance(data, dict) and "model" not in data:
                        data["model"] = model_to_use
                    
                # Make the request
                response = self.make_post_request_opea(endpoint_url, data, endpoint_type)
                
                return response
                
            except Exception as e:
                print(f"Error in OPEA endpoint handler: {e}")
                return None
                
        return handler
    
    def chat(self, messages, model=None, parameters=None, endpoint=None, api_key=None):
        """Send a chat request to an OPEA endpoint
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            model: Name of the model to use (optional)
            parameters: Additional parameters for the request (optional)
            endpoint: URL of the endpoint (optional)
            api_key: API key for authentication (optional)
            
        Returns:
            dict: Response from the OPEA API
        """
        try:
            # Get the model to use
            model_to_use = model or "gpt-3.5-turbo"
            
            # Get the endpoint URL
            endpoint_url = self.request_opea_endpoint(model_to_use, endpoint, "chat")
            if not endpoint_url:
                raise ValueError(f"No OPEA endpoint available for model {model_to_use}")
                
            # Prepare the request data
            data = {
                "model": model_to_use,
                "messages": messages
            }
            
            # Add any additional parameters
            if parameters:
                data.update(parameters)
                
            # Make the request
            response = self.make_post_request_opea(endpoint_url, data, "chat", None, api_key)
            
            return response
            
        except Exception as e:
            print(f"Error in OPEA chat: {e}")
            return None
            
    def stream_chat(self, messages, model=None, parameters=None, endpoint=None, api_key=None):
        """Send a streaming chat request to an OPEA endpoint
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            model: Name of the model to use (optional)
            parameters: Additional parameters for the request (optional)
            endpoint: URL of the endpoint (optional)
            api_key: API key for authentication (optional)
            
        Returns:
            iterator: An iterator over response chunks
        """
        try:
            # Get the model to use
            model_to_use = model or "gpt-3.5-turbo"
            
            # Get the endpoint URL
            endpoint_url = self.request_opea_endpoint(model_to_use, endpoint, "chat")
            if not endpoint_url:
                yield {"error": f"No OPEA endpoint available for model {model_to_use}"}
                return
                
            # Prepare the request data
            data = {
                "model": model_to_use,
                "messages": messages,
                "stream": True
            }
            
            # Add any additional parameters
            if parameters:
                for key, value in parameters.items():
                    if key != "stream":  # Ensure we don't override the stream parameter
                        data[key] = value
                        
            # Make the streaming request
            for chunk in self.make_stream_request_opea(endpoint_url, data, "chat", None, api_key):
                yield chunk
                
        except Exception as e:
            print(f"Error in OPEA streaming chat: {e}")
            yield {"error": str(e)}
    