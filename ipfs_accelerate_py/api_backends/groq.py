import os
import asyncio
import requests
import json
from pydantic import BaseModel

# https://console.groq.com/docs/models
PRODUCTION_MODELS = {
    "distil-whisper-large-v3-en": {
        "context_window": None
    },
    "gemma2-9b-it": {
        "context_window": 8192
    },
    "llama-3.3-70b-versatile": {
        "context_window": 128000
    },
    "llama-3.1-8b-instant": {
        "context_window": 128000
    },
    "llama-guard-3-8b": {
        "context_window": 8192
    },
    "llama3-70b-8192": {
        "context_window": 8192
    },
    "llama3-8b-8192": {
        "context_window": 8192
    },
    "mixtral-8x7b-32768": {
        "context_window": 32768
    },
    "whisper-large-v3": {
        "context_window": None
    },
    "whisper-large-v3-turbo": {
        "context_window": None
    },
}


PREVIEW_MODELS = {
    "qwen-2.5-32b": {
        "context_window": 128000
    },
    "deepseek-r1-distill-qwen-32b": {
        "context_window": 128000
    },
    "deepseek-r1-distill-llama-70b-specdec": {
        "context_window": 128000
    },
    "deepseek-r1-distill-llama-70b": {
        "context_window": 128000
    },
    "llama-3.3-70b-specdec": {
        "context_window": 8192
    },
    "llama-3.2-1b-preview": {
        "context_window": 128000
    },
    "llama-3.2-3b-preview": {
        "context_window": 128000
    },
    "llama-3.2-11b-vision-preview": {
        "context_window": 128000
    },
    "llama-3.2-90b-vision-preview": {
        "context_window": 128000
    },
}


class groq:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        # Register method references
        self.create_remote_groq_endpoint_handler = self.create_remote_groq_endpoint_handler
        self.request_groq_endpoint = self.request_groq_endpoint
        self.test_groq_endpoint = self.test_groq_endpoint
        self.make_post_request_groq = self.make_post_request_groq
        self.create_groq_endpoint_handler = self.create_groq_endpoint_handler
        self.init = self.init
        self.__test__ = self.__test__
        # Add endpoints tracking
        self.endpoints = {}
        self.endpoint_status = {}
        # Add queue for managing requests
        self.request_queue = asyncio.Queue(64)
        return None
    
    def init(self, endpoint_url=None, api_key=None, model_name=None):
        """Initialize a connection to a Groq API endpoint
        
        Args:
            endpoint_url: The URL of the API endpoint (defaults to Groq's API)
            api_key: API key for authentication
            model_name: Name of the model to use
            
        Returns:
            tuple: (endpoint_url, api_key, handler, queue, batch_size)
        """
        if not endpoint_url:
            endpoint_url = "https://api.groq.com/openai/v1/chat/completions"
            
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not provided and not found in environment variables")
        
        # Create the endpoint handler
        endpoint_handler = self.create_remote_groq_endpoint_handler(endpoint_url, api_key, model_name)
        
        # Register the endpoint
        if model_name not in self.endpoints:
            self.endpoints[model_name] = []
        
        if endpoint_url not in self.endpoints[model_name]:
            self.endpoints[model_name].append(endpoint_url)
            # Get context window from model definitions
            context_window = 0
            if model_name in PRODUCTION_MODELS:
                context_window = PRODUCTION_MODELS[model_name]["context_window"] or 8192
            elif model_name in PREVIEW_MODELS:
                context_window = PREVIEW_MODELS[model_name]["context_window"] or 8192
            else:
                context_window = 8192  # Default
                
            self.endpoint_status[endpoint_url] = context_window // 256  # Rough estimate of batch size
        
        return endpoint_url, api_key, endpoint_handler, self.request_queue, self.endpoint_status[endpoint_url]
    
    def __test__(self, endpoint_url, endpoint_handler, endpoint_label, api_key=None):
        """Test the Groq API endpoint
        
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
                print(f"Groq API test passed for {endpoint_label}")
                return True
            else:
                print(f"Groq API test failed for {endpoint_label}: No result")
                return False
        except Exception as e:
            print(f"Groq API test failed for {endpoint_label}: {e}")
            return False
    
    def make_post_request_groq(self, endpoint_url, data, api_key=None):
        """Make a POST request to the Groq API
        
        Args:
            endpoint_url: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication
            
        Returns:
            dict: Response from the endpoint
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(endpoint_url, headers=headers, json=data)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            print(f"Error making request to Groq API: {e}")
            return None
    
    def test_groq_endpoint(self, endpoint_url=None, api_key=None, model_name=None):
        """Test a Groq API endpoint
        
        Args:
            endpoint_url: URL of the endpoint to test
            api_key: API key for authentication
            model_name: Name of the model to use
            
        Returns:
            bool: True if test passes, False otherwise
        """
        if not endpoint_url:
            endpoint_url = "https://api.groq.com/openai/v1/chat/completions"
            
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not provided and not found in environment variables")
                
        if not model_name:
            model_name = "llama3-8b-8192"  # Default test model
            
        try:
            data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Complete this sentence: The quick brown fox"}
                ],
                "max_tokens": 20,
                "temperature": 0.7
            }
                
            result = self.make_post_request_groq(endpoint_url, data, api_key)
            
            if result and "choices" in result and len(result["choices"]) > 0:
                return True
            else:
                print(f"Test failed for endpoint {endpoint_url}: Invalid response format")
                return False
                
        except Exception as e:
            print(f"Test failed for endpoint {endpoint_url}: {e}")
            return False
    
    def request_groq_endpoint(self, model, endpoint=None, endpoint_type=None, batch=None):
        """Request a Groq endpoint
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL (optional)
            endpoint_type: Type of endpoint (optional)
            batch: Batch size (optional)
            
        Returns:
            str: URL of the selected endpoint
        """
        if endpoint:
            return endpoint
            
        # Default Groq API endpoint
        default_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        
        # Check if the model is in the endpoints dictionary
        if model in self.endpoints and self.endpoints[model]:
            return self.endpoints[model][0]  # Return the first registered endpoint
        
        # Return the default endpoint
        return default_endpoint
    
    def create_groq_endpoint_handler(self):
        """Create a default endpoint handler for Groq
        
        Returns:
            function: A handler for the default endpoint
        """
        api_key = os.getenv("GROQ_API_KEY")
        endpoint_url = "https://api.groq.com/openai/v1/chat/completions"
        model_name = "llama3-8b-8192"  # Default model
        
        return self.create_remote_groq_endpoint_handler(endpoint_url, api_key, model_name)
        
    def create_remote_groq_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for a remote Groq endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication
            model_name: Name of the model to use
            
        Returns:
            function: Handler for the endpoint
        """
        def handler(prompt, parameters=None, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                default_params = {
                    "max_tokens": 128,
                    "temperature": 0.7,
                    "top_p": 0.95
                }
                
                # Update with any provided parameters
                if parameters:
                    default_params.update(parameters)
                
                # Prepare request data
                data = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                
                # Add parameters to request
                data.update(default_params)
                
                # Make the request
                response = self.make_post_request_groq(endpoint_url, data, api_key)
                
                if response and "choices" in response and len(response["choices"]) > 0:
                    return response["choices"][0]["message"]["content"]
                
                return response
                
            except Exception as e:
                print(f"Error in Groq handler: {e}")
                return None
        
        return handler