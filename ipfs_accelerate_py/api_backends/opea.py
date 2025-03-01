import os
import json
import time
import threading
import hashlib
import uuid
import requests
from concurrent.futures import Future
from queue import Queue
from dotenv import load_dotenv

class opea:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get OPEA API endpoint from metadata or environment
        self.api_endpoint = self._get_api_endpoint()
        
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
        
        # Request tracking
        self.request_tracking = True
        self.recent_requests = {}
        
        
        # Retry and backoff settings
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 60  # Maximum delay in seconds
        
        # Request queue settings
        self.queue_enabled = True
        self.queue_size = 100
        self.queue_processing = False
        self.current_requests = 0
        self.max_concurrent_requests = 5
        self.request_queue = []
        self.queue_lock = threading.RLock()
        return None

    def _get_api_endpoint(self):
        """Get OPEA API endpoint from metadata or environment"""
        # Try to get from metadata
        api_endpoint = self.metadata.get("opea_endpoint")
        if api_endpoint:
            return api_endpoint
        
        # Try to get from environment
        env_endpoint = os.environ.get("OPEA_API_ENDPOINT")
        if env_endpoint:
            return env_endpoint
        
        # Try to load from dotenv
        try:
            load_dotenv()
            env_endpoint = os.environ.get("OPEA_API_ENDPOINT")
            if env_endpoint:
                return env_endpoint
        except ImportError:
            pass
        
        # Return default if no endpoint found
        return "http://localhost:8000/v1"
        
    def _process_queue(self):
        """Process queued requests with proper concurrency management"""
        while True:
            try:
                future, endpoint_url, data, request_id = self.request_queue.get()
                
                with self.queue_lock:
                    self.active_requests += 1
                
                # Process with retry logic
                retry_count = 0
                while retry_count <= self.max_retries:
                    try:
                        # Construct headers
                        headers = {"Content-Type": "application/json"}
                        
                        # Make request with proper error handling
                        response = requests.post(
                            endpoint_url,
                            json=data,
                            headers=headers,
                            timeout=self.metadata.get("timeout", 30)
                        )
                        
                        # Check for HTTP errors
                        response.raise_for_status()
                        
                        # Parse JSON response
                        result = response.json()
                        
                        # Update tracking with response
                        if self.request_tracking:
                            self.recent_requests[request_id] = {
                                "timestamp": time.time(),
                                "endpoint": endpoint_url,
                                "status": "success",
                                "response_code": response.status_code
                            }
                        
                        future.set_result(result)
                        break
                        
                    except requests.RequestException as e:
                        retry_count += 1
                        
                        if retry_count > self.max_retries:
                            # Update tracking with error
                            if self.request_tracking:
                                self.recent_requests[request_id] = {
                                    "timestamp": time.time(),
                                    "endpoint": endpoint_url,
                                    "status": "error",
                                    "error": str(e)
                                }
                            
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
                        # Update tracking with error
                        if self.request_tracking:
                            self.recent_requests[request_id] = {
                                "timestamp": time.time(),
                                "endpoint": endpoint_url,
                                "status": "error",
                                "error": str(e)
                            }
                        
                        future.set_exception(e)
                        break
                
                with self.queue_lock:
                    self.active_requests -= 1
                
                self.request_queue.task_done()
                
            except Exception as e:
                print(f"Error in queue processor: {e}")
    
    def make_post_request_opea(self, endpoint_url, data, request_id=None):
        """Make a request to OPEA API with queue and backoff"""
        # Generate unique request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Queue system with proper concurrency management
        future = Future()
        
        # Add to queue
        self.request_queue.put((future, endpoint_url, data, request_id))
        
        # Get result (blocks until request is processed)
        return future.result()
        
    def chat(self, messages, model=None, **kwargs):
        """Send a chat request to OPEA API"""
        # Construct the proper endpoint URL
        endpoint_url = f"{self.api_endpoint}/chat/completions"
        
        # Use provided model or default
        model = model or kwargs.get("model", "gpt-3.5-turbo")
        
        # Prepare request data
        data = {
            "model": model,
            "messages": messages
        }
        
        # Add optional parameters
        for key in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "stream"]:
            if key in kwargs:
                data[key] = kwargs[key]
        
        # Make request with queue and backoff
        response = self.make_post_request_opea(endpoint_url, data)
        
        # Extract text from response
        if "choices" in response and len(response["choices"]) > 0:
            text = response["choices"][0].get("message", {}).get("content", "")
        else:
            text = ""
        
        # Process and normalize response
        return {
            "text": text,
            "model": model,
            "usage": response.get("usage", {}),
            "implementation_type": "(REAL)",
            "raw_response": response  # Include raw response for advanced use
        }

    def stream_chat(self, messages, model=None, **kwargs):
        """Stream a chat request from OPEA API"""
        # Not implemented in this version - would need SSE streaming support
        raise NotImplementedError("Streaming not yet implemented for OPEA")
    
    def make_stream_request_opea(self, endpoint_url, data, request_id=None):
        """Make a streaming request to OPEA API"""
        # Not implemented in this version - would need SSE streaming support
        raise NotImplementedError("Streaming not yet implemented for OPEA")
            
    def create_opea_endpoint_handler(self):
        """Create an endpoint handler for OPEA"""
        async def endpoint_handler(prompt, **kwargs):
            """Handle requests to OPEA endpoint"""
            try:
                # Create messages from prompt
                if isinstance(prompt, list):
                    # Already formatted as messages
                    messages = prompt
                else:
                    # Create a simple user message
                    messages = [{"role": "user", "content": prompt}]
                
                # Extract model from kwargs or use default
                model = kwargs.get("model", "gpt-3.5-turbo")
                
                # Extract other parameters
                params = {k: v for k, v in kwargs.items() if k not in ["model"]}
                
                # Use streaming if requested
                if kwargs.get("stream", False):
                    raise NotImplementedError("Streaming not yet implemented for OPEA")
                else:
                    # Standard synchronous response
                    response = self.chat(messages, model, **params)
                    return response
            except Exception as e:
                print(f"Error calling OPEA endpoint: {e}")
                return {"text": f"Error: {str(e)}", "implementation_type": "(ERROR)"}
        
        return endpoint_handler
        
    def test_opea_endpoint(self, endpoint_url=None):
        """Test the OPEA endpoint"""
        if not endpoint_url:
            endpoint_url = f"{self.api_endpoint}/chat/completions"
            
        try:
            # Create a simple message
            messages = [{"role": "user", "content": "Testing the OPEA API. Please respond with a short message."}]
            
            # Make the request
            response = self.chat(messages)
            
            # Check if the response contains text
            return "text" in response and response.get("implementation_type") == "(REAL)"
        except Exception as e:
            print(f"Error testing OPEA endpoint: {e}")
            return False