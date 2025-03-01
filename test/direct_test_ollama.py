#!/usr/bin/env python
"""
Direct test for Ollama API implementation.
No dependencies on other modules.
"""

import os
import json
import time
import threading
import requests
import uuid
from concurrent.futures import Future
from queue import Queue
from pathlib import Path

# Create a dotenv implementation if needed
def load_dotenv(dotenv_path=None):
    """Load variables from .env file if it exists"""
    if dotenv_path is None:
        dotenv_path = os.path.join(os.getcwd(), '.env')
    
    if not os.path.isfile(dotenv_path):
        return False
    
    with open(dotenv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            os.environ[key] = value
    
    return True

# Import our Ollama implementation
class ollama:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get Ollama API endpoint from metadata or environment
        self.ollama_api_url = self._get_ollama_api_url()
        
        # Initialize queue and backoff systems
        self.max_concurrent_requests = 5
        self.queue_size = 100
        self.request_queue = Queue(maxsize=self.queue_size)
        self.active_requests = 0
        self.queue_lock = threading.RLock()
        
        # Start queue processor
        self.queue_processor = threading.Thread(target=self._process_queue)
        self.queue_processor.daemon = True
        self.queue_processor.start()
        
        # Initialize backoff configuration
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 16
        
        # Default model
        self.default_model = os.environ.get("OLLAMA_MODEL", "llama3")
        
        return None

    def _get_ollama_api_url(self):
        """Get Ollama API URL from metadata or environment"""
        # Try to get from metadata
        api_url = self.metadata.get("ollama_api_url")
        if api_url:
            return api_url
        
        # Try to get from environment
        env_url = os.environ.get("OLLAMA_API_URL")
        if env_url:
            return env_url
        
        # Try to load from dotenv
        try:
            load_dotenv()
            env_url = os.environ.get("OLLAMA_API_URL")
            if env_url:
                return env_url
        except ImportError:
            pass
        
        # Return default if no URL found
        return "http://localhost:11434/api"
        
    def _process_queue(self):
        """Process queued requests with proper concurrency management"""
        while True:
            try:
                future, endpoint_url, data, stream, request_id = self.request_queue.get()
                
                with self.queue_lock:
                    self.active_requests += 1
                
                # Process with retry logic
                retry_count = 0
                while retry_count <= self.max_retries:
                    try:
                        # Construct headers
                        headers = {"Content-Type": "application/json"}
                        
                        # Make request with proper error handling
                        if stream:
                            response = requests.post(
                                endpoint_url,
                                json=data,
                                headers=headers,
                                stream=True,
                                timeout=self.metadata.get("timeout", 30)
                            )
                        else:
                            response = requests.post(
                                endpoint_url,
                                json=data,
                                headers=headers,
                                timeout=self.metadata.get("timeout", 30)
                            )
                        
                        # Check for HTTP errors
                        response.raise_for_status()
                        
                        # Return response based on stream mode
                        if stream:
                            # Create a streaming generator
                            def response_generator():
                                for line in response.iter_lines():
                                    if line:
                                        yield json.loads(line)
                            
                            result = response_generator()
                        else:
                            # Parse JSON response
                            result = response.json()
                        
                        future.set_result(result)
                        break
                        
                    except requests.RequestException as e:
                        retry_count += 1
                        
                        if retry_count > self.max_retries:
                            future.set_exception(e)
                            break
                        
                        # Calculate backoff delay
                        delay = min(
                            self.initial_retry_delay * (self.backoff_factor ** (retry_count - 1)),
                            self.max_retry_delay
                        )
                        
                        # Sleep with backoff
                        time.sleep(delay)
                    
                    except Exception as e:
                        future.set_exception(e)
                        break
                
                with self.queue_lock:
                    self.active_requests -= 1
                
                self.request_queue.task_done()
                
            except Exception as e:
                print(f"Error in queue processor: {e}")
    
    def make_post_request_ollama(self, endpoint_url, data, stream=False):
        """Make a request to Ollama API with queue and backoff"""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Queue system with proper concurrency management
        future = Future()
        
        # Add to queue
        self.request_queue.put((future, endpoint_url, data, stream, request_id))
        
        # Get result (blocks until request is processed)
        return future.result()
        
    def chat(self, model, messages, options=None):
        """Send a chat request to Ollama API"""
        # Construct the proper endpoint URL
        endpoint_url = f"{self.ollama_api_url}/chat"
        
        # Format messages for Ollama API
        formatted_messages = self._format_messages(messages)
        
        # Prepare request data
        data = {
            "model": model,
            "messages": formatted_messages,
            "stream": False
        }
        
        # Add options if provided
        if options:
            data["options"] = options
        
        # Make request with queue and backoff
        response = self.make_post_request_ollama(endpoint_url, data)
        
        # Process and normalize response
        return {
            "text": response.get("message", {}).get("content", ""),
            "model": model,
            "usage": self._extract_usage(response),
            "implementation_type": "(REAL)"
        }

    def stream_chat(self, model, messages, options=None):
        """Stream a chat request from Ollama API"""
        # Construct the proper endpoint URL
        endpoint_url = f"{self.ollama_api_url}/chat"
        
        # Format messages for Ollama API
        formatted_messages = self._format_messages(messages)
        
        # Prepare request data
        data = {
            "model": model,
            "messages": formatted_messages,
            "stream": True
        }
        
        # Add options if provided
        if options:
            data["options"] = options
        
        # Make streaming request
        response_stream = self.make_post_request_ollama(endpoint_url, data, stream=True)
        
        # Process streaming response
        for chunk in response_stream:
            yield {
                "text": chunk.get("message", {}).get("content", ""),
                "done": chunk.get("done", False),
                "model": model
            }
            
    def _format_messages(self, messages):
        """Format messages for Ollama API"""
        formatted_messages = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Map standard roles to Ollama roles
            if role == "assistant":
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                role = "user"
            
            formatted_messages.append({
                "role": role,
                "content": content
            })
        
        return formatted_messages

    def _extract_usage(self, response):
        """Extract usage information from response"""
        return {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
        }

    def list_models(self):
        """List available models in Ollama"""
        endpoint_url = f"{self.ollama_api_url}/tags"
        
        try:
            response = requests.get(
                endpoint_url,
                timeout=self.metadata.get("timeout", 30)
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
            
    def create_ollama_endpoint_handler(self, endpoint_url=None):
        """Create an endpoint handler for Ollama"""
        if not endpoint_url:
            endpoint_url = self.ollama_api_url
        
        async def endpoint_handler(prompt, **kwargs):
            """Handle requests to Ollama endpoint"""
            # Get model from kwargs or default
            model = kwargs.get("model", self.metadata.get("ollama_model", self.default_model))
            
            # Create messages from prompt
            messages = [{"role": "user", "content": prompt}]
            
            # Extract options
            options = {}
            for key in ["temperature", "top_p", "top_k", "repeat_penalty"]:
                if key in kwargs:
                    options[key] = kwargs[key]
            
            # Make request
            try:
                response = self.chat(model, messages, options)
                return response
            except Exception as e:
                print(f"Error calling Ollama endpoint: {e}")
                return {"text": f"Error: {str(e)}", "implementation_type": "(ERROR)"}
        
        return endpoint_handler
        
    def test_ollama_endpoint(self, endpoint_url=None):
        """Test the Ollama endpoint"""
        if not endpoint_url:
            endpoint_url = f"{self.ollama_api_url}/chat"
            
        model = self.metadata.get("ollama_model", self.default_model)
        messages = [{"role": "user", "content": "Testing the Ollama API. Please respond with a short message."}]
        
        try:
            response = self.chat(model, messages)
            return "text" in response and response.get("implementation_type") == "(REAL)"
        except Exception as e:
            print(f"Error testing Ollama endpoint: {e}")
            return False

def test_ollama_api():
    """Test Ollama API implementation"""
    # Load environment variables if any
    load_dotenv()
    
    # Get Ollama API URL from environment or use default
    api_url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api")
    model = os.environ.get("OLLAMA_MODEL", "llama3")
    
    metadata = {
        "ollama_api_url": api_url,
        "ollama_model": model
    }
    
    client = ollama(resources={}, metadata=metadata)
    
    # Test basic functionality
    print(f"Testing Ollama API with URL: {api_url}")
    print(f"Testing with model: {model}")
    
    # Try to list models
    try:
        models = client.list_models()
        print(f"Available models: {len(models)} models found")
        if models:
            print("First few models:")
            for model_info in models[:3]:
                print(f" - {model_info.get('name', 'Unknown')}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # Test chat method
    try:
        # Keep this example very simple to avoid potential harmful outputs
        messages = [{"role": "user", "content": "Hello, respond with a single word greeting."}]
        response = client.chat(model, messages)
        
        print("\nChat response:")
        print(f"Text: {response.get('text', '')}")
        print(f"Model: {response.get('model', '')}")
        print(f"Implementation Type: {response.get('implementation_type', '')}")
        
        # Test usage statistics
        if "usage" in response:
            usage = response["usage"]
            print(f"Usage - Prompt tokens: {usage.get('prompt_tokens', 0)}")
            print(f"Usage - Completion tokens: {usage.get('completion_tokens', 0)}")
            print(f"Usage - Total tokens: {usage.get('total_tokens', 0)}")
            
        return True
    except Exception as e:
        print(f"Error testing chat: {e}")
        return False

if __name__ == "__main__":
    test_ollama_api()