import os
import json
import time
import uuid
import threading
import requests
import base64
from concurrent.futures import Future
from queue import Queue
from dotenv import load_dotenv

import hashlib
import queue
from concurrent.futures import ThreadPoolExecutor

class gemini:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get API key from metadata or environment
        self.api_key = self._get_api_key()
        
        # Set API base URL
        self.api_base = "https://generativelanguage.googleapis.com/v1"
        
        # Default model
        self.default_model = "gemini-1.5-pro"
        
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
        
        return

        # Initialize counters, queues, and settings for each endpoint 
        self.endpoints = {}  # Dictionary to store per-endpoint data
        
        # Retry and backoff settings (global defaults)
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 60  # Maximum delay in seconds
        
        # Global request queue settings
        self.queue_enabled = True
        self.queue_size = 100
        self.queue_processing = False
        self.current_requests = 0
        self.max_concurrent_requests = 5
        self.request_queue = []
        self.queue_lock = threading.RLock()
        
        # Initialize thread pool for async processing
        self.thread_pool = ThreadPoolExecutor(max_workers=10)

    def _get_api_key(self):
        """Get Gemini API key from metadata or environment"""
        # Try to get from metadata
        api_key = self.metadata.get("gemini_api_key") or self.metadata.get("google_api_key")
        if api_key:
            return api_key
        
        # Try to get from environment
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key:
            return env_key
        
        # Try to load from dotenv
        try:
            load_dotenv()
            env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if env_key:
                return env_key
        except ImportError:
            pass
        
        # Raise error if no key found
        raise ValueError("No Gemini API key found in metadata or environment")
        
    
    def _process_queue(self, endpoint_id=None):
        """Process requests in the queue for a specific endpoint or global queue"""
        # Get the endpoint or use global settings
        if endpoint_id and endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            with endpoint["queue_lock"]:
                if endpoint["queue_processing"]:
                    return  # Another thread is already processing this endpoint's queue
                endpoint["queue_processing"] = True
                
            queue_to_process = endpoint["request_queue"]
            is_global_queue = False
        else:
            # Use global queue if no endpoint specified or endpoint doesn't exist
            with self.queue_lock:
                if self.queue_processing:
                    return  # Another thread is already processing the global queue
                self.queue_processing = True
                
            queue_to_process = self.request_queue
            is_global_queue = True
        
        try:
            while True:
                # Get the next request from the queue
                request_info = None
                
                if is_global_queue:
                    with self.queue_lock:
                        if not queue_to_process:
                            self.queue_processing = False
                            break
                            
                        # Check if we're at the concurrent request limit
                        if self.current_requests >= self.max_concurrent_requests:
                            # Sleep briefly then check again
                            time.sleep(0.1)
                            continue
                            
                        # Get the next request and increase counter
                        request_info = queue_to_process.pop(0)
                        self.current_requests += 1
                else:
                    with endpoint["queue_lock"]:
                        if not queue_to_process:
                            endpoint["queue_processing"] = False
                            break
                            
                        # Check if we're at the concurrent request limit
                        if endpoint["current_requests"] >= endpoint["max_concurrent_requests"]:
                            # Sleep briefly then check again
                            time.sleep(0.1)
                            continue
                            
                        # Get the next request and increase counter
                        request_info = queue_to_process.pop(0)
                        endpoint["current_requests"] += 1
                
                # Process the request outside the lock
                if request_info:
                    try:
                        # Extract request details
                        endpoint_url = request_info.get("endpoint_url")
                        data = request_info.get("data")
                        api_key = request_info.get("api_key")
                        request_id = request_info.get("request_id")
                        endpoint_id = request_info.get("endpoint_id")
                        future = request_info.get("future")
                        method_name = request_info.get("method", "make_request")
                        method_args = request_info.get("args", [])
                        method_kwargs = request_info.get("kwargs", {})
                        
                        # Make the request (without queueing again)
                        # Save original queue_enabled value to prevent recursion
                        if is_global_queue:
                            original_queue_enabled = self.queue_enabled
                            self.queue_enabled = False  # Disable queueing to prevent recursion
                        else:
                            original_queue_enabled = endpoint["queue_enabled"]
                            endpoint["queue_enabled"] = False  # Disable queueing to prevent recursion
                        
                        try:
                            # Make the request based on method name
                            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                                method = getattr(self, method_name)
                                
                                # Call the method with the provided arguments
                                if method_name.startswith("make_"):
                                    # Direct API request methods
                                    result = method(
                                        endpoint_url=endpoint_url,
                                        data=data,
                                        api_key=api_key,
                                        request_id=request_id,
                                        endpoint_id=endpoint_id
                                    )
                                else:
                                    # Higher-level methods
                                    method_kwargs.update({
                                        "request_id": request_id,
                                        "endpoint_id": endpoint_id,
                                        "api_key": api_key
                                    })
                                    result = method(*method_args, **method_kwargs)
                            else:
                                # Fallback to make_request or similar method
                                make_method = getattr(self, "make_request", None)
                                if not make_method:
                                    make_method = getattr(self, f"make_post_request_{self.__class__.__name__.lower()}", None)
                                    
                                if make_method and callable(make_method):
                                    result = make_method(
                                        endpoint_url=endpoint_url,
                                        data=data,
                                        api_key=api_key,
                                        request_id=request_id,
                                        endpoint_id=endpoint_id
                                    )
                                else:
                                    raise AttributeError(f"Method {method_name} not found")
                            
                            # Store result in future
                            future["result"] = result
                            future["completed"] = True
                            
                            # Update counters
                            if not is_global_queue:
                                with endpoint["queue_lock"]:
                                    endpoint["successful_requests"] += 1
                                    endpoint["last_request_at"] = time.time()
                                    
                                    # Update token counts if present in result
                                    if isinstance(result, dict) and "usage" in result:
                                        usage = result["usage"]
                                        endpoint["total_tokens"] += usage.get("total_tokens", 0)
                                        endpoint["input_tokens"] += usage.get("prompt_tokens", 0)
                                        endpoint["output_tokens"] += usage.get("completion_tokens", 0)
                            
                        except Exception as e:
                            # Store error in future
                            future["error"] = e
                            future["completed"] = True
                            print(f"Error processing queued request: {str(e)}")
                            
                            # Update counters
                            if not is_global_queue:
                                with endpoint["queue_lock"]:
                                    endpoint["failed_requests"] += 1
                                    endpoint["last_request_at"] = time.time()
                        
                        finally:
                            # Restore original queue_enabled value
                            if is_global_queue:
                                self.queue_enabled = original_queue_enabled
                            else:
                                endpoint["queue_enabled"] = original_queue_enabled
                    
                    finally:
                        # Decrement counter
                        if is_global_queue:
                            with self.queue_lock:
                                self.current_requests = max(0, self.current_requests - 1)
                        else:
                            with endpoint["queue_lock"]:
                                endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
                    
                    # Brief pause to prevent CPU hogging
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"Error in queue processing thread: {str(e)}")
            
        finally:
            # Reset queue processing flag
            if is_global_queue:
                with self.queue_lock:
                    self.queue_processing = False
            else:
                with endpoint["queue_lock"]:
                    endpoint["queue_processing"] = False

    def _cleanup_old_requests(self):
        """Clean up old request tracking data"""
        current_time = time.time()
        # Keep requests from last 30 minutes
        cutoff_time = current_time - 1800
        
        keys_to_remove = []
        for request_id, request_data in self.recent_requests.items():
            if request_data.get("timestamp", 0) < cutoff_time:
                keys_to_remove.append(request_id)
        
        for key in keys_to_remove:
            del self.recent_requests[key]
    
    
    def make_post_request_gemini(self, endpoint_url, data, api_key=None, request_id=None, endpoint_id=None):
        """Make a request with endpoint-specific settings for queue, backoff, and API key"""
        # Get endpoint settings or use global defaults
        if endpoint_id and endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            is_endpoint_request = True
        else:
            endpoint = {
                "api_key": self.api_key,
                "max_retries": self.max_retries,
                "initial_retry_delay": self.initial_retry_delay,
                "backoff_factor": self.backoff_factor,
                "max_retry_delay": self.max_retry_delay,
                "queue_enabled": self.queue_enabled,
                "max_concurrent_requests": self.max_concurrent_requests,
                "current_requests": self.current_requests,
                "queue_processing": self.queue_processing,
                "request_queue": self.request_queue,
                "queue_lock": self.queue_lock,
                "queue_size": self.queue_size
            }
            is_endpoint_request = False
            
        # Use endpoint's API key if not explicitly provided
        if not api_key:
            api_key = endpoint["api_key"]
            
        if not api_key:
            raise ValueError("No API key provided for authentication")
            
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{int(time.time())}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"
            
        # If queue is enabled and we're at capacity, add to queue
        if endpoint["queue_enabled"]:
            with endpoint["queue_lock"]:
                if endpoint["current_requests"] >= endpoint["max_concurrent_requests"]:
                    # Create a future to store the result
                    result_future = {"result": None, "error": None, "completed": False}
                    
                    # Add to queue with all necessary info to process later
                    request_info = {
                        "endpoint_url": endpoint_url,
                        "data": data,
                        "api_key": api_key,
                        "request_id": request_id,
                        "endpoint_id": endpoint_id if is_endpoint_request else None,
                        "future": result_future
                    }
                    
                    # Check if queue is full
                    if len(endpoint["request_queue"]) >= endpoint["queue_size"]:
                        raise ValueError(f"Request queue is full ({endpoint['queue_size']} items). Try again later.")
                    
                    # Add to queue
                    endpoint["request_queue"].append(request_info)
                    print(f"Request queued. Queue size: {len(endpoint['request_queue'])}. Request ID: {request_id}")
                    
                    # Start queue processing if not already running
                    if not endpoint["queue_processing"]:
                        threading.Thread(target=self._process_queue, 
                                         args=(endpoint_id,) if is_endpoint_request else ()).start()
                    
                    # Wait for result with timeout
                    wait_start = time.time()
                    max_wait = 300  # 5 minutes
                    
                    while not result_future["completed"] and (time.time() - wait_start) < max_wait:
                        time.sleep(0.1)
                    
                    # Check if completed or timed out
                    if not result_future["completed"]:
                        raise TimeoutError(f"Request timed out after {max_wait} seconds in queue")
                    
                    # Propagate error if any
                    if result_future["error"]:
                        raise result_future["error"]
                    
                    return result_future["result"]
                
                # If we're not at capacity, increment counter
                endpoint["current_requests"] += 1
                
                # Update total request counter for endpoint-specific requests
                if is_endpoint_request:
                    endpoint["total_requests"] += 1
                    endpoint["last_request_at"] = time.time()
            
        # Use exponential backoff retry mechanism
        retries = 0
        retry_delay = endpoint["initial_retry_delay"]
        max_retries = endpoint["max_retries"]
        backoff_factor = endpoint["backoff_factor"]
        max_retry_delay = endpoint["max_retry_delay"]
        
        while retries < max_retries:
            try:
                # Add request_id to headers if possible
                headers = {
                    "Content-Type": "application/json",
                    "X-Request-ID": request_id
                }
                
                # Add API key to headers based on API type
                api_type = self.__class__.__name__.lower()
                if api_type == "claude":
                    headers["x-api-key"] = api_key
                    headers["anthropic-version"] = "2023-06-01"
                elif api_type == "groq":
                    headers["Authorization"] = f"Bearer {api_key}"
                elif api_type in ["openai", "openai_api"]:
                    headers["Authorization"] = f"Bearer {api_key}"
                elif api_type == "gemini":
                    # Gemini API key is typically passed as a URL parameter, but we'll set a header too
                    headers["x-goog-api-key"] = api_key
                else:
                    # Default to Bearer auth
                    headers["Authorization"] = f"Bearer {api_key}"
                
                # Make the actual request
                import requests
                response = requests.post(
                    endpoint_url,
                    json=data,
                    headers=headers,
                    timeout=60
                )
                
                # Check response status
                if response.status_code != 200:
                    error_message = f"Request failed with status code {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_message = f"{error_message}: {error_data['error'].get('message', '')}"
                    except:
                        error_message = f"{error_message}: {response.text[:100]}"
                    
                    # For specific error codes, handle differently
                    if response.status_code == 401:
                        # Decrement counter
                        if is_endpoint_request:
                            with endpoint["queue_lock"]:
                                endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
                                endpoint["failed_requests"] += 1
                                endpoint["last_request_at"] = time.time()
                        else:
                            with self.queue_lock:
                                self.current_requests = max(0, self.current_requests - 1)
                        
                        raise ValueError(f"Authentication error: {error_message}")
                        
                    elif response.status_code == 429:
                        # Rate limit error - check for Retry-After header
                        retry_after = response.headers.get("retry-after", None)
                        if retry_after:
                            try:
                                retry_delay = float(retry_after)
                            except:
                                # Use exponential backoff if Retry-After isn't a number
                                retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                        else:
                            # Use exponential backoff if Retry-After isn't present
                            retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                            
                        print(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Request ID: {request_id})")
                        time.sleep(retry_delay)
                        retries += 1
                        continue
                        
                    # For other error codes, just raise an exception
                    raise ValueError(error_message)
                
                # Parse and return successful response
                result = response.json()
                
                # Update token usage for endpoint-specific requests
                if is_endpoint_request:
                    with endpoint["queue_lock"]:
                        endpoint["successful_requests"] += 1
                        
                        # Update token counts if present in result
                        if "usage" in result:
                            usage = result["usage"]
                            endpoint["total_tokens"] += usage.get("total_tokens", 0)
                            endpoint["input_tokens"] += usage.get("prompt_tokens", 0) 
                            endpoint["output_tokens"] += usage.get("completion_tokens", 0)
                
                return result
                
            except requests.exceptions.RequestException as e:
                if retries < max_retries - 1:
                    print(f"Request failed: {str(e)}. Retrying in {retry_delay} seconds (attempt {retries+1}/{max_retries}, Request ID: {request_id})...")
                    time.sleep(retry_delay)
                    retries += 1
                    retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                else:
                    print(f"Request failed after {max_retries} attempts: {str(e)} (Request ID: {request_id})")
                    
                    # Decrement counter and update stats
                    if is_endpoint_request:
                        with endpoint["queue_lock"]:
                            endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
                            endpoint["failed_requests"] += 1
                            endpoint["last_request_at"] = time.time()
                    else:
                        with self.queue_lock:
                            self.current_requests = max(0, self.current_requests - 1)
                    
                    raise
            
            except Exception as e:
                # Decrement counter for any other exceptions
                if is_endpoint_request:
                    with endpoint["queue_lock"]:
                        endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
                        endpoint["failed_requests"] += 1
                        endpoint["last_request_at"] = time.time()
                else:
                    with self.queue_lock:
                        self.current_requests = max(0, self.current_requests - 1)
                raise
                
        # Decrement counter if we somehow exit the loop without returning or raising
        if is_endpoint_request:
            with endpoint["queue_lock"]:
                endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
        else:
            with self.queue_lock:
                self.current_requests = max(0, self.current_requests - 1)
                
        # This should never be reached due to the raise in the exception handler
        return None

    def chat(self, messages, model=None, request_id=None, endpoint_id=None, **kwargs):
        """Send a chat request to Gemini API"""
        # Check if we should use queue system
        endpoint = None
        if endpoint_id and endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            
            # Handle queueing if enabled and at capacity
            if endpoint["queue_enabled"] and endpoint["current_requests"] >= endpoint["max_concurrent_requests"]:
                try:
                    # Create a future to store the result
                    result_future = {"result": None, "error": None, "completed": False}
                    
                    # Add to queue with all necessary info to process later
                    method_kwargs = locals().copy()
                    # Remove 'self' from kwargs
                    if 'self' in method_kwargs:
                        del method_kwargs['self']
                        
                    request_info = {
                        "method": "chat",
                        "args": [],
                        "kwargs": method_kwargs,
                        "endpoint_id": endpoint_id,
                        "request_id": request_id,
                        "api_key": None,
                        "future": result_future
                    }
                    
                    # Check if queue is full
                    with endpoint["queue_lock"]:
                        if len(endpoint["request_queue"]) >= endpoint["queue_size"]:
                            raise ValueError(f"Request queue is full ({endpoint['queue_size']} items). Try again later.")
                        
                        # Add to queue
                        endpoint["request_queue"].append(request_info)
                        print(f"Request queued for chat. Queue size: {len(endpoint['request_queue'])}. Request ID: {request_id}")
                        
                        # Start queue processing if not already running
                        if not endpoint["queue_processing"]:
                            threading.Thread(target=self._process_queue, args=(endpoint_id,)).start()
                    
                    # Wait for result with timeout
                    wait_start = time.time()
                    max_wait = 300  # 5 minutes
                    
                    while not result_future["completed"] and (time.time() - wait_start) < max_wait:
                        time.sleep(0.1)
                    
                    # Check if completed or timed out
                    if not result_future["completed"]:
                        raise TimeoutError(f"Request timed out after {max_wait} seconds in queue")
                    
                    # Propagate error if any
                    if result_future["error"]:
                        raise result_future["error"]
                    
                    return result_future["result"]
                except Exception as e:
                    print(f"Error in chat queue handling: {e}")
                    # Fall through to regular processing
        
        # Update stats if using an endpoint
        if endpoint:
            with self.endpoints[endpoint_id]["queue_lock"]:
                self.endpoints[endpoint_id]["current_requests"] += 1
                self.endpoints[endpoint_id]["total_requests"] += 1
                self.endpoints[endpoint_id]["last_request_at"] = time.time()

        # Use specified model or default
        model = model or self.default_model
        
        # Format messages for Gemini API
        formatted_messages = self._format_messages(messages)
        
        # Prepare request data
        data = {
            "model": model,
            "contents": formatted_messages
        }
        
        # Add generation config if provided
        generation_config = {}
        for key in ["temperature", "topP", "topK", "maxOutputTokens"]:
            if key in kwargs:
                generation_config[key] = kwargs[key]
            elif key.lower() in kwargs:  # Handle snake_case keys too
                # Convert snake_case to camelCase
                snake_key = key.lower()
                generation_config[key] = kwargs[snake_key]
        
        if generation_config:
            data["generationConfig"] = generation_config
        
        # Make request with queue and backoff
        response = self.make_post_request_gemini(data, request_id=request_id, endpoint_id=endpoint_id)
        
        # Process and normalize response to match other APIs
       
        except Exception as e:
            # Update stats on error if using an endpoint
            if endpoint_id and endpoint_id in self.endpoints:
                with self.endpoints[endpoint_id]["queue_lock"]:
                    self.endpoints[endpoint_id]["current_requests"] = max(0, self.endpoints[endpoint_id]["current_requests"] - 1)
                    self.endpoints[endpoint_id]["failed_requests"] += 1
            raise
 
        # Update stats if using an endpoint
        if endpoint_id and endpoint_id in self.endpoints:
            with self.endpoints[endpoint_id]["queue_lock"]:
                self.endpoints[endpoint_id]["current_requests"] = max(0, self.endpoints[endpoint_id]["current_requests"] - 1)
                self.endpoints[endpoint_id]["successful_requests"] += 1
                # Update token counts if present in result (assuming result variable is named 'result')
                if 'result' in locals() and isinstance(result, dict) and "usage" in result:
                    usage = result["usage"]
                    self.endpoints[endpoint_id]["total_tokens"] += usage.get("total_tokens", 0)
                    self.endpoints[endpoint_id]["input_tokens"] += usage.get("prompt_tokens", 0)
                    self.endpoints[endpoint_id]["output_tokens"] += usage.get("completion_tokens", 0)
return {
            "text": self._extract_text(response),
            "model": model,
            "usage": self._extract_usage(response),
            "implementation_type": "(REAL)",
            "raw_response": response  # Include raw response for advanced use
        }

    def stream_chat(self, messages, model=None, **kwargs, request_id=None, endpoint_id=None):
        """Stream a chat request from Gemini API"""
        try:
        # Handle queueing if enabled and at capacity
        if endpoint_id and endpoint_id in self.endpoints:
                endpoint = self.endpoints[endpoint_id]
            if endpoint["queue_enabled"] and endpoint["current_requests"] >= endpoint["max_concurrent_requests"]:
                # Create a future to store the result
                result_future = {"result": None, "error": None, "completed": False}
                
                # Add to queue with all necessary info to process later
                method_kwargs = locals().copy()
                # Remove 'self' from kwargs
                if 'self' in method_kwargs:
                    del method_kwargs['self']
                    
                request_info = {
                    "method": "stream_chat",
                    "args": [],
                    "kwargs": method_kwargs,
                    "endpoint_id": endpoint_id,
                    "request_id": request_id,
                    "api_key": api_key if 'api_key' in locals() else None,
                    "future": result_future
                }
                
                # Check if queue is full
                with endpoint["queue_lock"]:
                    if len(endpoint["request_queue"]) >= endpoint["queue_size"]:
                        raise ValueError(f"Request queue is full ({endpoint['queue_size']} items). Try again later.")
                    
                    # Add to queue
                    endpoint["request_queue"].append(request_info)
                    print(f"Request queued for stream_chat. Queue size: {len(endpoint['request_queue'])}. Request ID: {request_id}")
                    
                    # Start queue processing if not already running
                    if not endpoint["queue_processing"]:
                        threading.Thread(target=self._process_queue, args=(endpoint_id,)).start()
                
                # Wait for result with timeout
                wait_start = time.time()
                max_wait = 300  # 5 minutes
                
                while not result_future["completed"] and (time.time() - wait_start) < max_wait:
                    time.sleep(0.1)
                
                # Check if completed or timed out
                if not result_future["completed"]:
                    raise TimeoutError(f"Request timed out after {max_wait} seconds in queue")
                
                # Propagate error if any
                if result_future["error"]:
                    raise result_future["error"]
                
               
        except Exception as e:
            # Update stats on error if using an endpoint
            if endpoint_id and endpoint_id in self.endpoints:
                with self.endpoints[endpoint_id]["queue_lock"]:
                    self.endpoints[endpoint_id]["current_requests"] = max(0, self.endpoints[endpoint_id]["current_requests"] - 1)
                    self.endpoints[endpoint_id]["failed_requests"] += 1
            raise
 
        # Update stats if using an endpoint
        if endpoint_id and endpoint_id in self.endpoints:
            with self.endpoints[endpoint_id]["queue_lock"]:
                self.endpoints[endpoint_id]["current_requests"] = max(0, self.endpoints[endpoint_id]["current_requests"] - 1)
                self.endpoints[endpoint_id]["successful_requests"] += 1
                # Update token counts if present in result (assuming result variable is named 'result')
                if 'result' in locals() and isinstance(result, dict) and "usage" in result:
                    usage = result["usage"]
                    self.endpoints[endpoint_id]["total_tokens"] += usage.get("total_tokens", 0)
                    self.endpoints[endpoint_id]["input_tokens"] += usage.get("prompt_tokens", 0)
                    self.endpoints[endpoint_id]["output_tokens"] += usage.get("completion_tokens", 0)
return result_future["result"]
                
        # Update stats if using an endpoint
        if endpoint_id and endpoint_id in self.endpoints:
            with self.endpoints[endpoint_id]["queue_lock"]:
                self.endpoints[endpoint_id]["current_requests"] += 1
                self.endpoints[endpoint_id]["total_requests"] += 1
                self.endpoints[endpoint_id]["last_request_at"] = time.time()

        # Use specified model or default
        model = model or self.default_model
        
        # Format messages for Gemini API
        formatted_messages = self._format_messages(messages)
        
        # Prepare request data
        data = {
            "model": model,
            "contents": formatted_messages
        }
        
        # Add generation config if provided
        generation_config = {}
        for key in ["temperature", "topP", "topK", "maxOutputTokens"]:
            if key in kwargs:
                generation_config[key] = kwargs[key]
            elif key.lower() in kwargs:  # Handle snake_case keys too
                # Convert snake_case to camelCase
                snake_key = key.lower()
                generation_config[key] = kwargs[snake_key]
        
        if generation_config:
            data["generationConfig"] = generation_config
        
        # Make streaming request
        response_stream = self.make_post_request_gemini(data, stream=True, request_id=request_id, endpoint_id=endpoint_id)
        
        # Process streaming response
        for chunk in response_stream:
            yield {
                "text": self._extract_text(chunk),
                "done": self._is_done(chunk),
                "model": model,
                "raw_chunk": chunk  # Include raw chunk for advanced use
            }

    def process_image(self, image_data, prompt, model=None, **kwargs, request_id=None, endpoint_id=None):
        """Process an image with Gemini API"""
        try:
        # Handle queueing if enabled and at capacity
        if endpoint_id and endpoint_id in self.endpoints:
                endpoint = self.endpoints[endpoint_id]
            if endpoint["queue_enabled"] and endpoint["current_requests"] >= endpoint["max_concurrent_requests"]:
                # Create a future to store the result
                result_future = {"result": None, "error": None, "completed": False}
                
                # Add to queue with all necessary info to process later
                method_kwargs = locals().copy()
                # Remove 'self' from kwargs
                if 'self' in method_kwargs:
                    del method_kwargs['self']
                    
                request_info = {
                    "method": "process_image",
                    "args": [],
                    "kwargs": method_kwargs,
                    "endpoint_id": endpoint_id,
                    "request_id": request_id,
                    "api_key": api_key if 'api_key' in locals() else None,
                    "future": result_future
                }
                
                # Check if queue is full
                with endpoint["queue_lock"]:
                    if len(endpoint["request_queue"]) >= endpoint["queue_size"]:
                        raise ValueError(f"Request queue is full ({endpoint['queue_size']} items). Try again later.")
                    
                    # Add to queue
                    endpoint["request_queue"].append(request_info)
                    print(f"Request queued for process_image. Queue size: {len(endpoint['request_queue'])}. Request ID: {request_id}")
                    
                    # Start queue processing if not already running
                    if not endpoint["queue_processing"]:
                        threading.Thread(target=self._process_queue, args=(endpoint_id,)).start()
                
                # Wait for result with timeout
                wait_start = time.time()
                max_wait = 300  # 5 minutes
                
                while not result_future["completed"] and (time.time() - wait_start) < max_wait:
                    time.sleep(0.1)
                
                # Check if completed or timed out
                if not result_future["completed"]:
                    raise TimeoutError(f"Request timed out after {max_wait} seconds in queue")
                
                # Propagate error if any
                if result_future["error"]:
                    raise result_future["error"]
                
                return result_future["result"]
                
        # Update stats if using an endpoint
        if endpoint_id and endpoint_id in self.endpoints:
            with self.endpoints[endpoint_id]["queue_lock"]:
                self.endpoints[endpoint_id]["current_requests"] += 1
                self.endpoints[endpoint_id]["total_requests"] += 1
                self.endpoints[endpoint_id]["last_request_at"] = time.time()

        # Use specified model or multimodal default
        model = model or "gemini-1.5-pro-vision"
        
        # Encode image data to base64
        if isinstance(image_data, bytes):
            encoded_image = base64.b64encode(image_data).decode('utf-8')
        else:
            # Assume it's already encoded
            encoded_image = image_data
        
        # Prepare content with text and image
        content = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": kwargs.get("mime_type", "image/jpeg"),
                            "data": encoded_image
                        }
                    }
                ]
            }
        ]
        
        # Prepare request data
        data = {
            "model": model,
            "contents": content
        }
        
        # Add generation config if provided
        generation_config = {}
        for key in ["temperature", "topP", "topK", "maxOutputTokens"]:
            if key in kwargs:
                generation_config[key] = kwargs[key]
            elif key.lower() in kwargs:  # Handle snake_case keys too
                # Convert snake_case to camelCase
                snake_key = key.lower()
                generation_config[key] = kwargs[snake_key]
        
        if generation_config:
            data["generationConfig"] = generation_config
        
        # Make request with queue and backoff
        response = self.make_post_request_gemini(data, request_id=request_id, endpoint_id=endpoint_id)
        
        # Process and normalize response to match other APIs
       
        except Exception as e:
            # Update stats on error if using an endpoint
            if endpoint_id and endpoint_id in self.endpoints:
                with self.endpoints[endpoint_id]["queue_lock"]:
                    self.endpoints[endpoint_id]["current_requests"] = max(0, self.endpoints[endpoint_id]["current_requests"] - 1)
                    self.endpoints[endpoint_id]["failed_requests"] += 1
            raise
 
        # Update stats if using an endpoint
        if endpoint_id and endpoint_id in self.endpoints:
            with self.endpoints[endpoint_id]["queue_lock"]:
                self.endpoints[endpoint_id]["current_requests"] = max(0, self.endpoints[endpoint_id]["current_requests"] - 1)
                self.endpoints[endpoint_id]["successful_requests"] += 1
                # Update token counts if present in result (assuming result variable is named 'result')
                if 'result' in locals() and isinstance(result, dict) and "usage" in result:
                    usage = result["usage"]
                    self.endpoints[endpoint_id]["total_tokens"] += usage.get("total_tokens", 0)
                    self.endpoints[endpoint_id]["input_tokens"] += usage.get("prompt_tokens", 0)
                    self.endpoints[endpoint_id]["output_tokens"] += usage.get("completion_tokens", 0)
return {
            "text": self._extract_text(response),
            "model": model,
            "usage": self._extract_usage(response),
            "implementation_type": "(REAL)",
            "raw_response": response  # Include raw response for advanced use
        }
            
    def _format_messages(self, messages):
        """Format messages for Gemini API"""
        formatted_messages = []
        current_role = None
        current_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Map standard roles to Gemini roles
            if role == "assistant":
                gemini_role = "model"
            elif role == "system":
                # For system messages, we add to user context
                gemini_role = "user"
            else:
                gemini_role = "user"
            
            # If role changes, add previous message
            if current_role and current_role != gemini_role and current_parts:
                formatted_messages.append({
                    "role": current_role,
                    "parts": current_parts
                })
                current_parts = []
            
            # Add content to parts
            current_role = gemini_role
            current_parts.append({"text": content})
        
        # Add final message
        if current_role and current_parts:
            formatted_messages.append({
                "role": current_role,
                "parts": current_parts
            })
        
        return formatted_messages

    def _extract_text(self, response):
        """Extract text from Gemini API response"""
        try:
            # Get candidates from response
            candidates = response.get("candidates", [])
            if not candidates:
                return ""
            
            # Get content from first candidate
            content = candidates[0].get("content", {})
            
            # Extract text from parts
            parts = content.get("parts", [])
            texts = [part.get("text", "") for part in parts if "text" in part]
            
            # Join all text parts
            return "".join(texts)
        except Exception as e:
            print(f"Error extracting text from response: {e}")
            return ""

    def _extract_usage(self, response):
        """Extract usage information from response"""
        try:
            # Get usage information from response
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            # Get candidates from response
            candidates = response.get("candidates", [])
            if not candidates:
                return usage
            
            # Get token count from first candidate
            token_count = candidates[0].get("tokenCount", {})
            
            # Extract token counts
            usage["prompt_tokens"] = token_count.get("inputTokens", 0)
            usage["completion_tokens"] = token_count.get("outputTokens", 0)
            usage["total_tokens"] = token_count.get("totalTokens", 0)
            
            return usage
        except Exception as e:
            print(f"Error extracting usage from response: {e}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _is_done(self, chunk):
        """Check if a streaming chunk indicates completion"""
        try:
            # Get candidates from chunk
            candidates = chunk.get("candidates", [])
            if not candidates:
                return False
            
            # Check finish reason from first candidate
            finish_reason = candidates[0].get("finishReason", None)
            
            # If finish reason is set, generation is done
            return finish_reason is not None
        except Exception:
            return False
            
    def create_gemini_endpoint_handler(self):
        """Create an endpoint handler for Gemini"""
        async def endpoint_handler(prompt, **kwargs):
            """Handle requests to Gemini endpoint"""
            try:
                # Extract model from kwargs or use default
                model = kwargs.get("model", self.default_model)
                
                # Check if prompt contains an image
                if isinstance(prompt, dict) and "image" in prompt:
                    # Process as image request
                    image_data = prompt["image"]
                    text_prompt = prompt.get("text", "Describe this image")
                    
                    response = self.process_image(image_data, text_prompt, model, **kwargs)
                    return response
                else:
                    # Create messages from prompt
                    if isinstance(prompt, list):
                        # Already formatted as messages
                        messages = prompt
                    else:
                        # Create a simple user message
                        messages = [{"role": "user", "content": prompt}]
                    
                    # Use streaming if requested
                    if kwargs.get("stream", False):
                        # For async streaming, need special handling
                        stream_response = self.stream_chat(messages, model, **kwargs)
                        
                        # Convert generator to async generator if needed
                        async def async_generator():
                            for chunk in stream_response:
                                yield chunk
                        
                        return async_generator()
                    else:
                        # Standard synchronous response
                        response = self.chat(messages, model, **kwargs)
                        return response
            except Exception as e:
                print(f"Error calling Gemini endpoint: {e}")
                return {"text": f"Error: {str(e)}", "implementation_type": "(ERROR)"}
        
        return endpoint_handler
        
    def test_gemini_endpoint(self, model=None):
        """Test the Gemini endpoint"""
        try:
            # Use specified model or default
            model = model or self.default_model
            
            # Create a simple message
            messages = [{"role": "user", "content": "Testing the Gemini API. Please respond with a short message."}]
            
            # Make the request
            response = self.chat(messages, model)
            
            # Check if the response contains text
            return "text" in response and response.get("implementation_type") == "(REAL)"
        except Exception as e:
            print(f"Error testing Gemini endpoint: {e}")
            return False

    def create_endpoint(self, endpoint_id=None, api_key=None, max_retries=None, initial_retry_delay=None, 
                       backoff_factor=None, max_retry_delay=None, queue_enabled=None, 
                       max_concurrent_requests=None, queue_size=None):
        """Create a new endpoint with its own settings and counters"""
        # Generate a unique endpoint ID if not provided
        if endpoint_id is None:
            endpoint_id = f"endpoint_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            
        # Use provided values or defaults
        endpoint_settings = {
            "api_key": api_key if api_key is not None else self.api_key,
            "max_retries": max_retries if max_retries is not None else self.max_retries,
            "initial_retry_delay": initial_retry_delay if initial_retry_delay is not None else self.initial_retry_delay,
            "backoff_factor": backoff_factor if backoff_factor is not None else self.backoff_factor,
            "max_retry_delay": max_retry_delay if max_retry_delay is not None else self.max_retry_delay,
            "queue_enabled": queue_enabled if queue_enabled is not None else self.queue_enabled,
            "max_concurrent_requests": max_concurrent_requests if max_concurrent_requests is not None else self.max_concurrent_requests,
            "queue_size": queue_size if queue_size is not None else self.queue_size,
            
            # Initialize endpoint-specific counters and state
            "current_requests": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "queue_processing": False,
            "request_queue": [],
            "queue_lock": threading.RLock(),
            "created_at": time.time(),
            "last_request_at": None
        }
        
        # Store the endpoint settings
        self.endpoints[endpoint_id] = endpoint_settings
        
        return endpoint_id


    def get_endpoint(self, endpoint_id=None):
        """Get an endpoint's settings or create a default one if not found"""
        # If no endpoint_id provided, use the first one or create a default
        if endpoint_id is None:
            if not self.endpoints:
                endpoint_id = self.create_endpoint()
            else:
                endpoint_id = next(iter(self.endpoints))
                
        # If endpoint doesn't exist, create it
        if endpoint_id not in self.endpoints:
            endpoint_id = self.create_endpoint(endpoint_id=endpoint_id)
            
        return self.endpoints[endpoint_id]


    def update_endpoint(self, endpoint_id, **kwargs):
        """Update an endpoint's settings"""
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_id} not found")
            
        # Update only the provided settings
        for key, value in kwargs.items():
            if key in self.endpoints[endpoint_id]:
                self.endpoints[endpoint_id][key] = value
                
        return self.endpoints[endpoint_id]


    def get_stats(self, endpoint_id=None):
        """Get usage statistics for an endpoint or global stats"""
        if endpoint_id and endpoint_id in self.endpoints:
                endpoint = self.endpoints[endpoint_id]
            stats = {
                "endpoint_id": endpoint_id,
                "total_requests": endpoint["total_requests"],
                "successful_requests": endpoint["successful_requests"],
                "failed_requests": endpoint["failed_requests"],
                "total_tokens": endpoint["total_tokens"],
                "input_tokens": endpoint["input_tokens"],
                "output_tokens": endpoint["output_tokens"],
                "created_at": endpoint["created_at"],
                "last_request_at": endpoint["last_request_at"],
                "current_queue_size": len(endpoint["request_queue"]),
                "current_requests": endpoint["current_requests"]
            }
            return stats
        else:
            # Aggregate stats across all endpoints
            total_requests = sum(e["total_requests"] for e in self.endpoints.values()) if self.endpoints else 0
            successful_requests = sum(e["successful_requests"] for e in self.endpoints.values()) if self.endpoints else 0
            failed_requests = sum(e["failed_requests"] for e in self.endpoints.values()) if self.endpoints else 0
            total_tokens = sum(e["total_tokens"] for e in self.endpoints.values()) if self.endpoints else 0
            input_tokens = sum(e["input_tokens"] for e in self.endpoints.values()) if self.endpoints else 0
            output_tokens = sum(e["output_tokens"] for e in self.endpoints.values()) if self.endpoints else 0
            
            stats = {
                "endpoints_count": len(self.endpoints),
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "global_queue_size": len(self.request_queue),
                "global_current_requests": self.current_requests
            }
            return stats


    def reset_stats(self, endpoint_id=None):
        """Reset usage statistics for an endpoint or globally"""
        if endpoint_id and endpoint_id in self.endpoints:
            # Reset stats just for this endpoint
                endpoint = self.endpoints[endpoint_id]
            endpoint["total_requests"] = 0
            endpoint["successful_requests"] = 0
            endpoint["failed_requests"] = 0
            endpoint["total_tokens"] = 0
            endpoint["input_tokens"] = 0
            endpoint["output_tokens"] = 0
        elif endpoint_id is None:
            # Reset stats for all endpoints
            for endpoint in self.endpoints.values():
                endpoint["total_requests"] = 0
                endpoint["successful_requests"] = 0
                endpoint["failed_requests"] = 0
                endpoint["total_tokens"] = 0
                endpoint["input_tokens"] = 0
                endpoint["output_tokens"] = 0
        else:
            raise ValueError(f"Endpoint {endpoint_id} not found")
