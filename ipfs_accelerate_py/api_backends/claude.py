import os
import json
import time
import threading
import logging
import uuid
import base64
import requests
from queue import Queue
from concurrent.futures import Future

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("claude_api")

class claude:
    """
    Claude API client implementation
    
    This implements the Anthropic Claude API with support for:
    - Environment variable/metadata API key handling
    - Request queueing with concurrency control
    - Exponential backoff for rate limits and errors
    - Stream and non-stream modes
    - System prompts and conversation history
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the Claude API client with resources and metadata"""
        # Store resources and metadata
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Set up API key from environment or metadata
        self.api_key = self._get_api_key()
        self.api_url = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1")
        
        # Initialize queuing and concurrency control
        self.max_concurrent_requests = int(self.metadata.get("max_concurrent_requests", 5))
        self.queue_size = int(self.metadata.get("queue_size", 100))
        self.request_queue = []
        self.queue_lock = threading.RLock()
        self.current_requests = 0
        self.queue_enabled = True
        self.queue_processing = False
        
        # Start queue processor in a daemon thread
        self.queue_processor = threading.Thread(target=self._process_queue)
        self.queue_processor.daemon = True
        self.queue_processor.start()
        
        # Set up retry and backoff configuration
        self.max_retries = int(self.metadata.get("max_retries", 5))
        self.initial_retry_delay = float(self.metadata.get("initial_retry_delay", 1.0))
        self.backoff_factor = float(self.metadata.get("backoff_factor", 2.0))
        self.max_retry_delay = float(self.metadata.get("max_retry_delay", 32.0))
        
        # For endpoint multiplexing support
        self.endpoints = {}
        
        # Default model
        self.default_model = os.environ.get("CLAUDE_MODEL", "claude-3-opus-20240229")
        
        # Model mappings
        self.model_mappings = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-2.1": "claude-2.1",
            "claude-2.0": "claude-2.0",
            "claude-instant-1.2": "claude-instant-1.2"
        }
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.input_tokens = 0 
        self.output_tokens = 0
        
        # Record initialization
        logger.info(f"Claude API client initialized with max_concurrent_requests={self.max_concurrent_requests}")
        return None
        
    def _get_api_key(self):
        """Get API key from metadata or environment variables"""
        # Try metadata first
        api_key = self.metadata.get("claude_api_key", None)
        if api_key:
            return api_key
            
        # Try various environment variable names
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
        if not api_key and self.metadata.get("allow_mock", False):
            logger.warning("No Claude API key found, using a placeholder for testing")
            return "mock_claude_api_key_for_testing_only"
            
        raise ValueError("No Claude API key found in metadata or environment variables")
        
    def _process_queue(self):
        """Process requests in the queue in FIFO order"""
        with self.queue_lock:
            if self.queue_processing:
                return  # Another thread is already processing the queue
            self.queue_processing = True
        
        logger.info("Starting Claude API queue processing thread")
        
        try:
            while True:
                # Get the next request from the queue
                with self.queue_lock:
                    if not self.request_queue:
                        self.queue_processing = False
                        break
                        
                    if self.current_requests >= self.max_concurrent_requests:
                        time.sleep(0.1)  # Wait a bit before checking again
                        continue
                        
                    # Get the next request
                    request = self.request_queue.pop(0)
                    self.current_requests += 1
                
                # Process the request outside the lock
                try:
                    # Extract request information
                    future = request["future"]
                    endpoint_id = request.get("endpoint_id")
                    api_key = request.get("api_key")
                    model = request.get("model")
                    messages = request.get("messages")
                    system = request.get("system")
                    max_tokens = request.get("max_tokens")
                    temperature = request.get("temperature")
                    request_id = request.get("request_id", str(uuid.uuid4()))
                    stream = request.get("stream", False)
                    
                    # Process the request with retry logic
                    retry_count = 0
                    last_exception = None
                    
                    while retry_count <= self.max_retries:
                        try:
                            # Make the API request
                            result = self._make_chat_request(
                                model=model,
                                messages=messages,
                                system=system,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                api_key=api_key,
                                request_id=request_id,
                                stream=stream
                            )
                            
                            # Set the result in the future
                            future["result"] = result
                            future["completed"] = True
                            
                            # Update statistics for this endpoint
                            with self.queue_lock:
                                if endpoint_id and endpoint_id in self.endpoints:
                                    self.endpoints[endpoint_id]["successful_requests"] += 1
                                
                            break
                        
                        except Exception as e:
                            last_exception = e
                            retry_count += 1
                            
                            # Check if we should retry
                            if retry_count > self.max_retries:
                                logger.error(f"Max retries ({self.max_retries}) exceeded for request {request_id}: {str(e)}")
                                future["error"] = str(e)
                                future["completed"] = True
                                
                                # Update statistics for this endpoint
                                with self.queue_lock:
                                    if endpoint_id and endpoint_id in self.endpoints:
                                        self.endpoints[endpoint_id]["failed_requests"] += 1
                                        
                                break
                                
                            # Calculate delay with exponential backoff
                            delay = min(
                                self.initial_retry_delay * (self.backoff_factor ** (retry_count - 1)),
                                self.max_retry_delay
                            )
                            
                            # Check if this is a rate limit error with retry-after header
                            if hasattr(e, "headers") and "retry-after" in e.headers:
                                try:
                                    retry_after = float(e.headers["retry-after"])
                                    delay = max(delay, retry_after)
                                except (ValueError, TypeError):
                                    pass
                                    
                            logger.warning(f"Request {request_id} failed (attempt {retry_count}/{self.max_retries}), retrying in {delay:.2f}s: {str(e)}")
                            time.sleep(delay)
                
                except Exception as e:
                    logger.error(f"Error processing request from queue: {str(e)}")
                    
                    # Set the error in the future
                    if "future" in request:
                        request["future"]["error"] = str(e)
                        request["future"]["completed"] = True
                
                finally:
                    # Update request count
                    with self.queue_lock:
                        self.current_requests -= 1
        
        except Exception as e:
            logger.error(f"Error in Claude API queue processor: {str(e)}")
        finally:
            with self.queue_lock:
                self.queue_processing = False
                
    def _add_to_queue(self, **request_params):
        """Add a request to the queue and return a future for the result"""
        # Create a future to track the request
        result_future = {"result": None, "error": None, "completed": False}
        
        # Add request information to queue
        with self.queue_lock:
            # Check if queue is full
            if len(self.request_queue) >= self.queue_size:
                raise RuntimeError(f"Claude API request queue is full (max {self.queue_size} requests)")
                
            # Add to queue
            self.request_queue.append({
                "future": result_future,
                **request_params
            })
            
            # Make sure queue processor is running
            if not self.queue_processing:
                thread = threading.Thread(target=self._process_queue)
                thread.daemon = True
                thread.start()
                
        return result_future
        
    def _wait_for_future(self, result_future, timeout=60):
        """Wait for the future to complete and return the result or raise the error"""
        # Wait for result with timeout
        start_time = time.time()
        while not result_future["completed"]:
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for Claude API request (timeout: {timeout}s)")
                
        # Check for errors
        if result_future["error"]:
            raise RuntimeError(f"Claude API request failed: {result_future['error']}")
            
        # Return the result
        return result_future["result"]
        
    def _make_chat_request(self, model, messages, system=None, max_tokens=None, temperature=None, api_key=None, request_id=None, stream=False):
        """Make a direct chat request to the Claude API"""
        # Use provided API key or default
        api_key = api_key or self.api_key
        
        # Prepare headers
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Add request ID if provided
        if request_id:
            headers["x-request-id"] = request_id
            
        # Format messages
        formatted_messages = self._format_messages(messages)
        
        # Prepare request body
        body = {
            "model": model,
            "messages": formatted_messages,
        }
        
        # Add optional parameters
        if system:
            body["system"] = system
            
        if max_tokens:
            body["max_tokens"] = max_tokens
            
        if temperature is not None:
            body["temperature"] = temperature
            
        if stream:
            body["stream"] = True
            
        # Make the request
        try:
            url = f"{self.api_url}/messages"
            
            if stream:
                # Handle streaming response
                response = requests.post(url, headers=headers, json=body, stream=True)
                response.raise_for_status()
                
                # Return a generator for streaming responses
                def generate_stream():
                    for line in response.iter_lines():
                        if line:
                            line = line.decode("utf-8")
                            
                            # Skip keep-alives
                            if line.startswith("data: "):
                                data = line[6:]  # Remove "data: " prefix
                                
                                # Check for [DONE] marker
                                if data == "[DONE]":
                                    break
                                    
                                try:
                                    chunk = json.loads(data)
                                    yield chunk
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse Claude API streaming response: {line}")
                
                return generate_stream()
            else:
                # Regular non-streaming response
                response = requests.post(url, headers=headers, json=body)
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                
                # Update token statistics
                if "usage" in result:
                    usage = result["usage"]
                    self.total_tokens += usage.get("total_tokens", 0)
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
                    
                return result
                
        except requests.RequestException as e:
            logger.error(f"Claude API request failed: {str(e)}")
            raise
            
    def _format_messages(self, messages):
        """Format messages for the Claude API"""
        formatted_messages = []
        
        # Check if messages is a string (single user message)
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
            
        # Iterate through messages
        for msg in messages:
            # Check if message is a string (assume user role)
            if isinstance(msg, str):
                formatted_messages.append({"role": "user", "content": msg})
                continue
                
            # Process dictionary message
            if isinstance(msg, dict):
                role = msg.get("role", "user").lower()
                content = msg.get("content", "")
                
                # Map OpenAI roles to Claude roles
                if role == "assistant":
                    formatted_role = "assistant"
                elif role in ["system", "user"]:
                    formatted_role = "user"
                else:
                    formatted_role = "user"
                    
                formatted_messages.append({"role": formatted_role, "content": content})
                
        return formatted_messages
        
    def chat(self, model=None, messages=None, system=None, max_tokens=None, temperature=None, request_id=None, stream=False, endpoint_id=None, api_key=None):
        """
        Send a chat request to the Claude API with queueing and retry support
        
        Args:
            model (str): Claude model to use
            messages (list): List of message dictionaries with role and content
            system (str, optional): System prompt
            max_tokens (int, optional): Maximum tokens to generate
            temperature (float, optional): Sampling temperature
            request_id (str, optional): Custom request ID for tracking
            stream (bool, optional): Whether to stream the response
            endpoint_id (str, optional): Custom endpoint ID for multiplexing
            api_key (str, optional): Custom API key for this request
            
        Returns:
            dict: Claude API response
        """
        # Get the API key to use (endpoint, custom, or default)
        if endpoint_id and endpoint_id in self.endpoints:
            api_key = api_key or self.endpoints[endpoint_id].get("api_key") or self.api_key
        else:
            api_key = api_key or self.api_key
            
        # Map model name if needed
        if model and model in self.model_mappings:
            model = self.model_mappings[model]
        elif not model:
            model = self.default_model
            
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
            
        # Check if streaming is requested (direct call, no queue)
        if stream:
            return self._make_chat_request(
                model=model,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
                request_id=request_id,
                stream=True
            )
            
        # Update statistics
        self.total_requests += 1
        
        # Add request to queue
        result_future = self._add_to_queue(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            request_id=request_id,
            endpoint_id=endpoint_id,
            stream=False
        )
        
        # Wait for result
        try:
            result = self._wait_for_future(result_future)
            self.successful_requests += 1
            
            # Update endpoint statistics
            if endpoint_id and endpoint_id in self.endpoints:
                with self.endpoints[endpoint_id]["queue_lock"]:
                    self.endpoints[endpoint_id]["current_requests"] -= 1
                    
                    # Update token statistics if available
                    if "usage" in result:
                        usage = result["usage"]
                        self.endpoints[endpoint_id]["total_tokens"] += usage.get("total_tokens", 0)
                        self.endpoints[endpoint_id]["input_tokens"] += usage.get("input_tokens", 0)
                        self.endpoints[endpoint_id]["output_tokens"] += usage.get("completion_tokens", 0)
            return result_future["result"]
        except Exception as e:
            # Update stats if using an endpoint in case of error
            if endpoint_id and endpoint_id in self.endpoints:
                with self.endpoints[endpoint_id]["queue_lock"]:
                    self.endpoints[endpoint_id]["current_requests"] += 1
                    self.endpoints[endpoint_id]["total_requests"] += 1
                    self.endpoints[endpoint_id]["last_request_at"] = time.time()
            raise e

        try:
            if not model_name:
                model_name = self.default_model
            elif model_name in self.model_mappings:
                model_name = self.model_mappings[model_name]
                
            # Convert messages to Claude format
            formatted_messages = self._format_messages(messages)
            
            # Prepare request body
            body = {
                "model": model_name,
                "messages": formatted_messages,
            }
            
            # Add optional parameters
            if system:
                body["system"] = system
                
            if max_tokens:
                body["max_tokens"] = max_tokens
                
            if temperature is not None:
                body["temperature"] = temperature
                
            # Prepare headers
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            
            # Add request ID if provided
            if request_id:
                headers["x-request-id"] = request_id
                
            # Make the request
            response = requests.post(f"{self.api_url}/messages", headers=headers, json=body)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Update endpoint statistics
            if endpoint_id and endpoint_id in self.endpoints:
                with self.endpoints[endpoint_id]["queue_lock"]:
                    self.endpoints[endpoint_id]["successful_requests"] += 1
                    
                    # Update token statistics if available
                    if "usage" in result:
                        usage = result["usage"]
                        self.endpoints[endpoint_id]["total_tokens"] += usage.get("total_tokens", 0)
                        self.endpoints[endpoint_id]["input_tokens"] += usage.get("input_tokens", 0)
                        self.endpoints[endpoint_id]["output_tokens"] += usage.get("output_tokens", 0)
                        
            # Update global token statistics
            if "usage" in result:
                usage = result["usage"]
                self.total_tokens += usage.get("total_tokens", 0)
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                
            self.successful_requests += 1
            return result
            
        except requests.RequestException as e:
            self.failed_requests += 1
            
            # Update endpoint statistics
            if endpoint_id and endpoint_id in self.endpoints:
                with self.endpoints[endpoint_id]["queue_lock"]:
                    self.endpoints[endpoint_id]["failed_requests"] += 1
                    
            # Check if this is a rate limit error
            if hasattr(e, "response") and e.response is not None and e.response.status_code == 429:
                retry_after = e.response.headers.get("retry-after", "1")
                try:
                    retry_seconds = int(retry_after)
                except ValueError:
                    retry_seconds = 1
                    
                raise RuntimeError(f"Claude API rate limit exceeded, retry after {retry_seconds}s")
                
            raise RuntimeError(f"Claude API request failed: {str(e)}")
            
    def stream_chat(self, model=None, messages=None, system=None, max_tokens=None, temperature=None, request_id=None, api_key=None):
        """Stream a chat completion from Claude API"""
        return self.chat(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            request_id=request_id,
            stream=True,
            api_key=api_key
        )
        
    def create_endpoint(self, endpoint_id=None, api_key=None, max_retries=None, initial_retry_delay=None, 
                       backoff_factor=None, max_retry_delay=None, queue_enabled=None, 
                       max_concurrent_requests=None, queue_size=None):
        """Create a new endpoint with its own settings and counters"""
        # Generate a unique endpoint ID if not provided
        if endpoint_id is None:
            endpoint_id = str(uuid.uuid4())
            
        with self.queue_lock:
            # Check if endpoint already exists
            if endpoint_id in self.endpoints:
                raise ValueError(f"Endpoint '{endpoint_id}' already exists")
                
            # Create the endpoint
            self.endpoints[endpoint_id] = {
                "api_key": api_key or self.api_key,
                "max_retries": max_retries or self.max_retries,
                "initial_retry_delay": initial_retry_delay or self.initial_retry_delay,
                "backoff_factor": backoff_factor or self.backoff_factor,
                "max_retry_delay": max_retry_delay or self.max_retry_delay,
                "queue_enabled": queue_enabled if queue_enabled is not None else self.queue_enabled,
                "max_concurrent_requests": max_concurrent_requests or self.max_concurrent_requests,
                "queue_size": queue_size or self.queue_size,
                "queue_lock": threading.RLock(),
                "current_requests": 0,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "created_at": time.time(),
                "last_request_at": None
            }
            
        logger.info(f"Created Claude API endpoint '{endpoint_id}'")
        return endpoint_id
        
    def get_endpoint_stats(self, endpoint_id):
        """Get statistics for a specific endpoint"""
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_id}' does not exist")
            
        # Copy stats (excluding mutex)
        with self.endpoints[endpoint_id]["queue_lock"]:
            stats = {k: v for k, v in self.endpoints[endpoint_id].items() if k != "queue_lock"}
            
        return stats
        
    def get_stats(self):
        """Get global API client statistics"""
        with self.queue_lock:
            stats = {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "current_requests": self.current_requests,
                "queue_size": len(self.request_queue),
                "total_tokens": self.total_tokens,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "endpoints": len(self.endpoints),
                "queue_enabled": self.queue_enabled,
                "max_concurrent_requests": self.max_concurrent_requests
            }
            
        return stats
        
    def request_complete(self, model=None, messages=None, system=None, max_tokens=None, temperature=None):
        """Compatibility method for the unified API interface"""
        result = self.chat(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract text from response
        if "content" in result:
            text = result["content"][0]["text"]
        else:
            text = ""
            
        # Format response for compatibility
        return {
            "text": text,
            "raw": result,
            "implementation_type": "REAL"
        }