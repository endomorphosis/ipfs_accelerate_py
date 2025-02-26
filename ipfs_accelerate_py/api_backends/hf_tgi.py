import asyncio
import os
import requests
import json

class hf_tgi:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        # Register method references
        self.create_remote_text_generation_endpoint_handler = self.create_remote_text_generation_endpoint_handler
        self.request_tgi_endpoint = self.request_tgi_endpoint
        self.test_tgi_endpoint = self.test_tgi_endpoint
        self.make_post_request_hf_tgi = self.make_post_request_hf_tgi
        self.create_hf_tgi_endpoint_handler = self.create_hf_tgi_endpoint_handler
        self.init = self.init
        self.__test__ = self.__test__
        # Add endpoints tracking
        self.endpoints = {}
        self.endpoint_status = {}
        self.endpoint_types = ["remote", "local"]
        # Add queue for managing requests
        self.request_queue = asyncio.Queue(64)
        return None
    
    def init(self, endpoint_url=None, api_key=None, model_name=None):
        """Initialize a connection to a remote text generation interface endpoint
        
        Args:
            endpoint_url: The URL of the remote endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            
        Returns:
            tuple: (endpoint_url, api_key, handler, queue, batch_size)
        """
        # Create the endpoint handler
        endpoint_handler = self.create_remote_text_generation_endpoint_handler(endpoint_url, api_key, model_name)
        
        # Register the endpoint
        if model_name not in self.endpoints:
            self.endpoints[model_name] = []
        
        if endpoint_url not in self.endpoints[model_name]:
            self.endpoints[model_name].append(endpoint_url)
            self.endpoint_status[endpoint_url] = 32  # Default batch size
        
        return endpoint_url, api_key, endpoint_handler, self.request_queue, self.endpoint_status[endpoint_url]
    
    def __test__(self, endpoint_url, endpoint_handler, endpoint_label, api_key=None):
        """Test the remote text generation endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            endpoint_handler: The handler function
            endpoint_label: Label for the endpoint
            api_key: API key for authentication
            
        Returns:
            bool: True if test passes, False otherwise
        """
        test_prompt = "Complete this sentence: The quick brown fox"
        try:
            result = endpoint_handler(test_prompt)
            if result is not None:
                print(f"Remote text generation test passed for {endpoint_label}")
                return True
            else:
                print(f"Remote text generation test failed for {endpoint_label}: No result")
                return False
        except Exception as e:
            print(f"Remote text generation test failed for {endpoint_label}: {e}")
            return False
    
    def make_post_request_hf_tgi(self, endpoint_url, data, api_key=None):
        """Make a POST request to a remote text generation interface
        
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
            print(f"Error making request to text generation endpoint: {e}")
            return None
    
    def test_tgi_endpoint(self, endpoint_url=None, api_key=None, model_name=None):
        """Test a text generation endpoint
        
        Args:
            endpoint_url: URL of the endpoint to test
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            
        Returns:
            bool: True if test passes, False otherwise
        """
        test_prompt = "Complete this sentence: The quick brown fox"
        try:
            data = {
                "inputs": test_prompt,
                "parameters": {
                    "max_new_tokens": 20,
                    "do_sample": True,
                    "temperature": 0.7
                }
            }
            
            if model_name:
                data["model"] = model_name
                
            result = self.make_post_request_hf_tgi(endpoint_url, data, api_key)
            
            if result and not isinstance(result, dict) or "generated_text" in result or "outputs" in result:
                return True
            else:
                print(f"Test failed for endpoint {endpoint_url}: Invalid response format")
                return False
                
        except Exception as e:
            print(f"Test failed for endpoint {endpoint_url}: {e}")
            return False
    
    def create_hf_tgi_endpoint_handler(self, model=None, endpoint=None, endpoint_type=None, batch=None):
        """Create an endpoint handler for text generation
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL
            endpoint_type: Type of endpoint
            batch: Batch size
            
        Returns:
            function: Handler for the endpoint
        """
        return self.create_remote_text_generation_endpoint_handler(endpoint, None, model)
    
    def request_tgi_endpoint(self, model, endpoint=None, endpoint_type=None, batch=None):
        """Request a text generation endpoint
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL (optional)
            endpoint_type: Type of endpoint (optional)
            batch: Batch size (optional)
            
        Returns:
            str: URL of the selected endpoint
        """
        incoming_batch_size = len(batch) if batch else 1
        endpoint_batch_size = 0
        
        if endpoint in self.endpoint_status:
            endpoint_batch_size = self.endpoint_status[endpoint]
            if incoming_batch_size <= endpoint_batch_size:
                return endpoint
        
        elif endpoint_type == None:
            # Try to find any suitable endpoint
            for e_type in self.endpoint_types:
                if e_type in self.__dict__ and model in self.__dict__[e_type]:
                    for e in self.__dict__[e_type][model]:
                        if self.endpoint_status[e] >= incoming_batch_size:
                            return e
        
        else:
            # Check only within the specified endpoint type
            if endpoint_type in self.__dict__ and model in self.__dict__[endpoint_type]:
                for e in self.__dict__[endpoint_type][model]:
                    if self.endpoint_status[e] >= incoming_batch_size:
                        return e
        
        # Direct check in endpoints dictionary as fallback
        if model in self.endpoints:
            for e in self.endpoints[model]:
                if self.endpoint_status[e] >= incoming_batch_size:
                    return e
        
        # No suitable endpoint found
        if incoming_batch_size > max(self.endpoint_status.values(), default=0):
            raise ValueError("Batch size too large for all available endpoints")
        
        return None
    
    def create_remote_text_generation_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for a remote text generation endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication, if required
            model_name: Name of the model to use
            
        Returns:
            function: Handler for the endpoint
        """
        def handler(prompt, parameters=None, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                default_params = {
                    "max_new_tokens": 128,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.95
                }
                
                # Update with any provided parameters
                if parameters:
                    default_params.update(parameters)
                
                # Prepare request data
                data = {
                    "inputs": prompt,
                    "parameters": default_params
                }
                
                if model_name:
                    data["model"] = model_name
                
                # Make the request
                response = self.make_post_request_hf_tgi(endpoint_url, data, api_key)
                
                if response:
                    # Different TGI servers might have different response formats
                    if isinstance(response, list):
                        return response[0] if response else None
                    elif "generated_text" in response:
                        return response["generated_text"]
                    elif "outputs" in response:
                        return response["outputs"]
                    else:
                        return response
                
                return None
            
            except Exception as e:
                print(f"Error in text generation handler: {e}")
                return None
        
        return handler