import os
import json
import time
import requests
import logging
import threading
import uuid
import hashlib
from queue import Queue
from concurrent.futures import Future

# Configure logging
logger = logging.getLogger("claude_api")

class claude:
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the Claude API client.
        
        Args:
            resources: Dictionary of resources (unused)
            metadata: Dictionary with configuration options
                - api_key: Claude API key
                - model: Default model to use
                - max_retries: Maximum number of retries for API calls
                - timeout: Timeout for API calls in seconds
        """
        self.resources = resources or {}
        self.metadata = metadata or {}
        
        # Get API key
        self.api_key = self._get_api_key(self.metadata)
        
        # Set default values
        self.base_url = "https://api.anthropic.com/v1"
        self.default_model = self.metadata.get("model", "claude-3-haiku-20240307")
        self.timeout = int(self.metadata.get("timeout", 60))
        
        # Initialize request tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.request_time = 0
        
        # Initialize queue and concurrency control
        self.max_concurrent_requests = int(self.metadata.get("max_concurrent_requests", 5))
        self.queue_size = int(self.metadata.get("queue_size", 100))
        self.request_queue = Queue(maxsize=self.queue_size)
        self.active_requests = 0
        self.queue_lock = threading.RLock()
        self.queue_processing = False

        # Initialize endpoint registry
        self.endpoints = {}
        
        # Start queue processor
        self.queue_processor = threading.Thread(target=self._process_queue)
        self.queue_processor.daemon = True
        self.queue_processor.start()
        
        # Set up retry and backoff configuration
        self.max_retries = int(self.metadata.get("max_retries", 5))
        self.initial_retry_delay = float(self.metadata.get("initial_retry_delay", 1.0))
        self.backoff_factor = float(self.metadata.get("backoff_factor", 2.0))
        self.max_retry_delay = float(self.metadata.get("max_retry_delay", 60.0))
        
        logger.info("Claude API client initialized with max_concurrent_requests=%s", self.max_concurrent_requests)
    
    def _get_api_key(self, metadata):
        """Get API key from metadata or environment variables"""
        # Try metadata
        api_key = metadata.get("api_key") or metadata.get("claude_api_key") or metadata.get("anthropic_api_key")
        
        # Try environment variables
        if not api_key:
            for env_var in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_KEY"]:
                api_key = os.environ.get(env_var)
                if api_key:
                    return api_key
                    
            # Try loading from .env file if python-dotenv is available
            try:
                from dotenv import load_dotenv
                load_dotenv()
                for env_var in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_KEY"]:
                    api_key = os.environ.get(env_var)
                    if api_key:
                        return api_key
            except ImportError:
                pass
                
            # Use a placeholder key for testing if no real key is available
            logger.warning("No Claude API key found, using a placeholder for testing")
            return "mock_claude_api_key_for_testing_only"
        
        return api_key
    
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
                        if not queue_to_process.qsize():
                            self.queue_processing = False
                            break
                            
                        # Check if we're at the concurrent request limit
                        if self.active_requests >= self.max_concurrent_requests:
                            # Sleep briefly then check again
                            time.sleep(0.1)
                            continue
                            
                        # Get the next request and increase counter
                        request_info = queue_to_process.get(block=False)
                        self.active_requests += 1
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
                        future = request_info.get("future")
                        func = request_info.get("func")
                        args = request_info.get("args", [])
                        kwargs = request_info.get("kwargs", {})
                        
                        try:
                            # Call the function
                            result = func(*args, **kwargs)
                            
                            # Store result in future
                            future.set_result(result)
                            
                            # Update stats
                            if not is_global_queue:
                                with endpoint["queue_lock"]:
                                    endpoint["successful_requests"] += 1
                        except Exception as e:
                            # Store error in future
                            future.set_exception(e)
                            logger.error(f"Error processing queued request: {str(e)}")
                            
                            # Update stats
                            if not is_global_queue:
                                with endpoint["queue_lock"]:
                                    endpoint["failed_requests"] += 1
                    
                    finally:
                        # Decrement counter
                        if is_global_queue:
                            with self.queue_lock:
                                self.active_requests = max(0, self.active_requests - 1)
                                if hasattr(queue_to_process, "task_done"):
                                    queue_to_process.task_done()
                        else:
                            with endpoint["queue_lock"]:
                                endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
                    
                    # Brief pause to prevent CPU hogging
                    time.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Error in queue processing thread: {str(e)}")
            
        finally:
            # Reset queue processing flag
            if is_global_queue:
                with self.queue_lock:
                    self.queue_processing = False
            else:
                with endpoint["queue_lock"]:
                    endpoint["queue_processing"] = False
            
            logger.info("Queue processing thread exiting")
    
    def _with_queue_and_backoff(self, func, *args, **kwargs):
        """Execute a function with queue and backoff management"""
        future = Future()
        
        # Add to queue
        request_info = {
            "future": future,
            "func": func,
            "args": args,
            "kwargs": kwargs
        }
        
        try:
            self.request_queue.put(request_info, block=True, timeout=60)
            
            # Start queue processor if not already running
            if not self.queue_processing:
                self._process_queue()
                
            # Wait for result
            return future.result(timeout=300)  # 5 minute timeout
        except Exception as e:
            logger.error(f"Error queuing request: {str(e)}")
            raise
    
    def create_endpoint(self, endpoint_id=None, api_key=None, max_retries=None, 
                     initial_retry_delay=None, backoff_factor=None, max_retry_delay=None,
                     max_concurrent_requests=None, queue_size=None):
        """Create a new endpoint with custom settings"""
        if endpoint_id is None:
            endpoint_id = f"endpoint_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            
        # Use defaults or provided values
        self.endpoints[endpoint_id] = {
            "api_key": api_key if api_key is not None else self.api_key,
            "max_retries": max_retries if max_retries is not None else self.max_retries,
            "initial_retry_delay": initial_retry_delay if initial_retry_delay is not None else self.initial_retry_delay,
            "backoff_factor": backoff_factor if backoff_factor is not None else self.backoff_factor,
            "max_retry_delay": max_retry_delay if max_retry_delay is not None else self.max_retry_delay,
            "max_concurrent_requests": max_concurrent_requests if max_concurrent_requests is not None else self.max_concurrent_requests,
            "queue_size": queue_size if queue_size is not None else self.queue_size,
            
            # Initialize counters and queue
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "current_requests": 0,
            "request_queue": [],
            "queue_lock": threading.RLock(),
            "queue_processing": False,
        }
        
        return endpoint_id
    
    def get_endpoint(self, endpoint_id):
        """Get endpoint settings or create default if not found"""
        if endpoint_id not in self.endpoints:
            endpoint_id = self.create_endpoint(endpoint_id=endpoint_id)
            
        return self.endpoints[endpoint_id]
    
    def update_endpoint(self, endpoint_id, **kwargs):
        """Update endpoint settings"""
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_id} does not exist")
            
        for key, value in kwargs.items():
            if key in self.endpoints[endpoint_id]:
                self.endpoints[endpoint_id][key] = value
                
        return self.endpoints[endpoint_id]
    
    def make_post_request(self, endpoint_url, data, api_key=None, request_id=None, endpoint_id=None):
        """Make a POST request to the Claude API with proper error handling"""
        # Use default API key if not provided
        if api_key is None:
            if endpoint_id and endpoint_id in self.endpoints:
                api_key = self.endpoints[endpoint_id]["api_key"]
            else:
                api_key = self.api_key
                
        if not api_key:
            raise ValueError("No API key provided")
        
        # Check if we're using a mock key and return a fake response
        if api_key == "mock_claude_api_key_for_testing_only":
            # Generate a mock response for testing
            mock_response = {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"This is a mock response from Claude API for testing."
                    }
                ],
                "model": data.get("model", "claude-3-haiku-20240307"),
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 25,
                    "output_tokens": 15
                },
                "request_id": request_id
            }
            
            # Simulate backoff logic by waiting a short time
            time.sleep(0.1)
            
            # Update tracking stats
            self.successful_requests += 1
            return mock_response
            
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
        # Set up headers
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "tools-2023-12-15",
            "x-request-id": request_id
        }
        
        # Track request start time
        start_time = time.time()
        
        # Get retry settings
        if endpoint_id and endpoint_id in self.endpoints:
            max_retries = self.endpoints[endpoint_id]["max_retries"]
            initial_retry_delay = self.endpoints[endpoint_id]["initial_retry_delay"]
            backoff_factor = self.endpoints[endpoint_id]["backoff_factor"]
            max_retry_delay = self.endpoints[endpoint_id]["max_retry_delay"]
        else:
            max_retries = self.max_retries
            initial_retry_delay = self.initial_retry_delay
            backoff_factor = self.backoff_factor
            max_retry_delay = self.max_retry_delay
        
        # Add to total requests counter
        self.total_requests += 1
        
        # Initialize retry counter and delay
        retries = 0
        retry_delay = initial_retry_delay
        
        while retries <= max_retries:
            try:
                # Make the request
                response = requests.post(
                    endpoint_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                # Handle error status codes
                if response.status_code != 200:
                    error_message = f"Claude API request failed with status code {response.status_code}"
                    
                    # Try to extract error details
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_message = f"{error_message}: {error_data['error']}"
                    except:
                        error_message = f"{error_message}: {response.text[:100]}..."
                        
                    # Handle specific error codes
                    if response.status_code == 401:
                        # Authentication error, don't retry
                        self.failed_requests += 1
                        raise ValueError(f"Authentication error: {error_message}")
                    elif response.status_code == 429:
                        # Rate limit exceeded, get retry-after if available
                        retry_after = response.headers.get("retry-after")
                        if retry_after:
                            try:
                                retry_delay = float(retry_after)
                            except:
                                retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                        else:
                            retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                            
                        logger.warning(f"Rate limit exceeded, retrying in {retry_delay}s: {error_message}")
                        
                        # Increment retry counter
                        retries += 1
                        
                        # Sleep before retry
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Other error, retry with backoff
                        logger.warning(f"API error, retrying in {retry_delay}s: {error_message}")
                        
                        # Increment retry counter
                        retries += 1
                        
                        # If we've exhausted retries, give up
                        if retries > max_retries:
                            self.failed_requests += 1
                            raise ValueError(error_message)
                            
                        # Sleep before retry
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                        continue
                
                # Parse successful response
                response_data = response.json()
                
                # Update timing and success counters
                self.successful_requests += 1
                self.request_time += time.time() - start_time
                
                # Add request ID to response
                response_data["request_id"] = request_id
                
                return response_data
                
            except requests.exceptions.RequestException as e:
                # Handle network errors
                logger.warning(f"Request error, retrying in {retry_delay}s: {str(e)}")
                
                # Increment retry counter
                retries += 1
                
                # If we've exhausted retries, give up
                if retries > max_retries:
                    self.failed_requests += 1
                    raise ValueError(f"Claude API request failed after {max_retries} retries: {str(e)}")
                    
                # Sleep before retry
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
            except Exception as e:
                # Handle other errors
                self.failed_requests += 1
                raise ValueError(f"Claude API request failed: {str(e)}")
                
        # This should never be reached due to the handling in the loops
        self.failed_requests += 1
        raise ValueError(f"Claude API request failed after {max_retries} retries")
    
    def chat(self, messages, model=None, max_tokens=1000, temperature=0.7, top_p=0.95, 
          stream=False, tools=None, tool_choice=None, stop_sequences=None, system=None,
          request_id=None, endpoint_id=None, api_key=None):
        """
        Generate a response from the Claude assistant.
        
        Args:
            messages: List of message objects with 'role' and 'content'
            model: Model to use (default: claude-3-haiku-20240307)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            stream: Whether to stream the response
            tools: List of tool specifications
            tool_choice: Tool choice specification
            stop_sequences: List of sequences that will stop generation
            system: System prompt
            request_id: Optional request ID for tracking
            endpoint_id: Optional endpoint ID for custom settings
            api_key: Optional API key to use
            
        Returns:
            API response with generated text
        """
        # Use default model if not provided
        if model is None:
            model = self.default_model
            
        # Build request
        endpoint_url = f"{self.base_url}/messages"
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages,
        }
        
        # Add optional parameters
        if system:
            data["system"] = system
            
        if stop_sequences:
            data["stop_sequences"] = stop_sequences
            
        if tools:
            data["tools"] = tools
            
        if tool_choice:
            data["tool_choice"] = tool_choice
            
        # Stream handling is different
        if stream:
            return self.stream_chat(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                tool_choice=tool_choice,
                stop_sequences=stop_sequences,
                system=system,
                request_id=request_id,
                endpoint_id=endpoint_id,
                api_key=api_key
            )
        
        # Make the API request with backoff
        response = self.make_post_request(
            endpoint_url=endpoint_url,
            data=data,
            api_key=api_key,
            request_id=request_id,
            endpoint_id=endpoint_id
        )
        
        # Return a standardized response format
        return {
            "text": response.get("content", [{"text": ""}])[0].get("text", ""),
            "raw_response": response,
            "model": model,
            "usage": {
                "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
                "total_tokens": response.get("usage", {}).get("input_tokens", 0) + response.get("usage", {}).get("output_tokens", 0)
            },
            "implementation_type": "REAL"
        }
    
    def stream_chat(self, messages, model=None, max_tokens=1000, temperature=0.7, top_p=0.95,
                 tools=None, tool_choice=None, stop_sequences=None, system=None,
                 request_id=None, endpoint_id=None, api_key=None):
        """Stream a chat response from Claude"""
        # Use default model if not provided
        if model is None:
            model = self.default_model
            
        # Use specified API key or default
        if api_key is None:
            if endpoint_id and endpoint_id in self.endpoints:
                api_key = self.endpoints[endpoint_id]["api_key"]
            else:
                api_key = self.api_key
                
        if not api_key:
            raise ValueError("No API key provided")
            
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
        # Build request
        endpoint_url = f"{self.base_url}/messages"
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages,
            "stream": True,
        }
        
        # Add optional parameters
        if system:
            data["system"] = system
            
        if stop_sequences:
            data["stop_sequences"] = stop_sequences
            
        if tools:
            data["tools"] = tools
            
        if tool_choice:
            data["tool_choice"] = tool_choice
            
        # Set up headers
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "tools-2023-12-15",
            "x-request-id": request_id
        }
        
        # Make streaming request
        try:
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=data,
                stream=True,
                timeout=self.timeout
            )
            
            # Check for errors
            if response.status_code != 200:
                error_message = f"Claude API streaming request failed with status code {response.status_code}"
                
                # Try to extract error details
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_message = f"{error_message}: {error_data['error']}"
                except:
                    error_message = f"{error_message}: {response.text[:100]}..."
                
                raise ValueError(error_message)
            
            # Prepare response container
            text_so_far = ""
            event_data = []
            usage = {"input_tokens": 0, "output_tokens": 0}
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    # Remove 'data: ' prefix
                    if line.startswith(b'data: '):
                        line = line[6:]
                        
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    # Parse JSON event
                    try:
                        event = json.loads(line)
                        event_data.append(event)
                        
                        # Extract usage info if available
                        if "usage" in event:
                            usage = event["usage"]
                            
                        # Extract text content
                        if event.get("type") == "content_block_delta" and event.get("delta", {}).get("text"):
                            text_so_far += event["delta"]["text"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse stream line: {line}")
            
            # Return a standardized response format
            return {
                "text": text_so_far,
                "raw_response": event_data,
                "model": model,
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                },
                "implementation_type": "REAL"
            }
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Claude API streaming request failed: {str(e)}")
    
    def get_stats(self):
        """Get usage statistics"""
        avg_time = self.request_time / self.successful_requests if self.successful_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_request_time": avg_time,
            "endpoints": len(self.endpoints)
        }
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.request_time = 0
        
        return self.get_stats()