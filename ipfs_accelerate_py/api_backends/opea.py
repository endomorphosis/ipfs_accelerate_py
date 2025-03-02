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
        # Batching settings
        self.batching_enabled = True
        self.max_batch_size = 10
        self.batch_timeout = 0.5  # Max seconds to wait for more requests
        self.batch_queue = {}  # Keyed by model name
        self.batch_timers = {}  # Timers for each batch
        self.batch_lock = threading.RLock()
        
        # Models that support batching
        self.embedding_models = []  # Models supporting batched embeddings
        self.completion_models = []  # Models supporting batched completions
        self.supported_batch_models = []  # All models supporting batching

        # Circuit breaker settings
        self.circuit_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_threshold = 5  # Number of failures before opening circuit
        self.reset_timeout = 30  # Seconds to wait before trying half-open
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_lock = threading.RLock()

        # Priority levels
        self.PRIORITY_HIGH = 0
        self.PRIORITY_NORMAL = 1
        self.PRIORITY_LOW = 2
        
        # Change request queue to priority-based
        self.request_queue = []  # Will store (priority, request_info) tuples

        
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
        """Process requests in the queue with standard pattern"""
        with self.queue_lock:
            if self.queue_processing:
                return  # Another thread is already processing
            self.queue_processing = True
        
        try:
            while True:
                # Get the next request from the queue
                request_info = None
                
                with self.queue_lock:
                    if not self.request_queue:
                        self.queue_processing = False
                        break
                        
                    # Check if we're at capacity
                    if self.active_requests >= self.max_concurrent_requests:
                        time.sleep(0.1)  # Brief pause
                        continue
                        
                    # Get next request and increment counter
                    request_info = self.request_queue.pop(0)
                    self.active_requests += 1
                
                # Process the request outside the lock
                if request_info:
                    try:
                        # Extract request details
                        future = request_info.get("future")
                        func = request_info.get("func")
                        args = request_info.get("args", [])
                        kwargs = request_info.get("kwargs", {})
                        
                        # Special handling for different request formats
                        if func and callable(func):
                            # Function-based request
                            try:
                                result = func(*args, **kwargs)
                                if future:
                                    future["result"] = result
                                    future["completed"] = True
                            except Exception as e:
                                if future:
                                    future["error"] = e
                                    future["completed"] = True
                                logger.error(f"Error executing queued function: {e}")
                        else:
                            # Direct API request format
                            endpoint_url = request_info.get("endpoint_url")
                            data = request_info.get("data")
                            api_key = request_info.get("api_key")
                            request_id = request_info.get("request_id")
                            
                            if hasattr(self, "make_request"):
                                method = self.make_request
                            elif hasattr(self, "make_post_request"):
                                method = self.make_post_request
                            else:
                                raise AttributeError("No request method found")
                            
                            # Temporarily disable queueing to prevent recursion
                            original_queue_enabled = getattr(self, "queue_enabled", True)
                            setattr(self, "queue_enabled", False)
                            
                            try:
                                result = method(
                                    endpoint_url=endpoint_url,
                                    data=data,
                                    api_key=api_key,
                                    request_id=request_id
                                )
                                
                                if future:
                                    future["result"] = result
                                    future["completed"] = True
                            except Exception as e:
                                if future:
                                    future["error"] = e
                                    future["completed"] = True
                                logger.error(f"Error processing queued request: {e}")
                            finally:
                                # Restore original queue_enabled
                                setattr(self, "queue_enabled", original_queue_enabled)
                    
                    finally:
                        # Decrement counter
                        with self.queue_lock:
                            self.active_requests = max(0, self.active_requests - 1)
                
                # Brief pause to prevent CPU hogging
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in queue processing thread: {e}")
            
        finally:
            # Reset queue processing flag
            with self.queue_lock:
                self.queue_processing = False

    def queue_with_priority(self, request_info, priority=None):
        # Queue a request with a specific priority level
        if priority is None:
            priority = self.PRIORITY_NORMAL
            
        with self.queue_lock:
            # Check if queue is full
            if len(self.request_queue) >= self.queue_size:
                raise ValueError(f"Request queue is full ({self.queue_size} items). Try again later.")
            
            # Record queue entry time for metrics
            request_info["queue_entry_time"] = time.time()
            
            # Add to queue with priority
            self.request_queue.append((priority, request_info))
            
            # Sort queue by priority (lower numbers = higher priority)
            self.request_queue.sort(key=lambda x: x[0])
            
            logger.info(f"Request queued with priority {priority}. Queue size: {len(self.request_queue)}")
            
            # Start queue processing if not already running
            if not self.queue_processing:
                threading.Thread(target=self._process_queue).start()
                
            # Create future to track result
            future = {"result": None, "error": None, "completed": False}
            request_info["future"] = future
            return future
    
    def make_post_request_opea(self, endpoint_url, data, request_id=None):
        """Make a request to OPEA API with queue and backoff"""
        # Generate unique request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Queue system with proper concurrency management
        future = Future()
        
        # Add to queue
        self.request_queue.append((future, endpoint_url, data, request_id))
        
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
    def check_circuit_breaker(self):
        # Check if circuit breaker allows requests to proceed
        with self.circuit_lock:
            now = time.time()
            
            if self.circuit_state == "OPEN":
                # Check if enough time has passed to try again
                if now - self.last_failure_time > self.reset_timeout:
                    logger.info("Circuit breaker transitioning from OPEN to HALF-OPEN")
                    self.circuit_state = "HALF_OPEN"
                    return True
                else:
                    # Circuit is open, fail fast
                    return False
                    
            elif self.circuit_state == "HALF_OPEN":
                # In half-open state, we allow a single request to test the service
                return True
                
            else:  # CLOSED
                # Normal operation, allow requests
                return True

    def track_request_result(self, success, error_type=None):
        # Track the result of a request for circuit breaker logic tracking
        with self.circuit_lock:
            if success:
                # Successful request
                if self.circuit_state == "HALF_OPEN":
                    # Service is working again, close the circuit
                    logger.info("Circuit breaker transitioning from HALF-OPEN to CLOSED")
                    self.circuit_state = "CLOSED"
                    self.failure_count = 0
                elif self.circuit_state == "CLOSED":
                    # Reset failure count on success
                    self.failure_count = 0
            else:
                # Failed request
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Update error statistics
                if error_type and hasattr(self, "collect_metrics") and self.collect_metrics:
                    with self.stats_lock:
                        if error_type not in self.request_stats["errors_by_type"]:
                            self.request_stats["errors_by_type"][error_type] = 0
                        self.request_stats["errors_by_type"][error_type] += 1
                
                if self.circuit_state == "CLOSED" and self.failure_count >= self.failure_threshold:
                    # Too many failures, open the circuit
                    logger.warning(f"Circuit breaker transitioning from CLOSED to OPEN after {self.failure_count} failures")
                    self.circuit_state = "OPEN"
                    
                    # Update circuit breaker statistics
                    if hasattr(self, "stats_lock") and hasattr(self, "request_stats"):
                        with self.stats_lock:
                            if "circuit_breaker_trips" not in self.request_stats:
                                self.request_stats["circuit_breaker_trips"] = 0
                            self.request_stats["circuit_breaker_trips"] += 1
                    
                elif self.circuit_state == "HALF_OPEN":
                    # Failed during test request, back to open
                    logger.warning("Circuit breaker transitioning from HALF-OPEN to OPEN after test request failure")
                    self.circuit_state = "OPEN"
    
    def add_to_batch(self, model, request_info):
        # Add a request to the batch queue for the specified model
        if not hasattr(self, "batching_enabled") or not self.batching_enabled or model not in self.supported_batch_models:
            # Either batching is disabled or model doesn't support it
            return False
            
        with self.batch_lock:
            # Initialize batch queue for this model if needed
            if model not in self.batch_queue:
                self.batch_queue[model] = []
                
            # Add request to batch
            self.batch_queue[model].append(request_info)
            
            # Check if we need to start a timer for this batch
            if len(self.batch_queue[model]) == 1:
                # First item in batch, start timer
                if model in self.batch_timers and self.batch_timers[model] is not None:
                    self.batch_timers[model].cancel()
                
                self.batch_timers[model] = threading.Timer(
                    self.batch_timeout, 
                    self._process_batch,
                    args=[model]
                )
                self.batch_timers[model].daemon = True
                self.batch_timers[model].start()
                
            # Check if batch is full and should be processed immediately
            if len(self.batch_queue[model]) >= self.max_batch_size:
                # Cancel timer since we're processing now
                if model in self.batch_timers and self.batch_timers[model] is not None:
                    self.batch_timers[model].cancel()
                    self.batch_timers[model] = None
                    
                # Process batch immediately
                threading.Thread(target=self._process_batch, args=[model]).start()
                return True
                
            return True
    
    def _process_batch(self, model):
        # Process a batch of requests for the specified model
        with self.batch_lock:
            # Get all requests for this model
            if model not in self.batch_queue:
                return
                
            batch_requests = self.batch_queue[model]
            self.batch_queue[model] = []
            
            # Clear timer reference
            if model in self.batch_timers:
                self.batch_timers[model] = None
        
        if not batch_requests:
            return
            
        # Update batch statistics
        if hasattr(self, "collect_metrics") and self.collect_metrics and hasattr(self, "update_stats"):
            self.update_stats({"batched_requests": len(batch_requests)})
        
        try:
            # Check which type of batch processing to use
            if model in self.embedding_models:
                self._process_embedding_batch(model, batch_requests)
            elif model in self.completion_models:
                self._process_completion_batch(model, batch_requests)
            else:
                logger.warning(f"Unknown batch processing type for model {model}")
                # Fail all requests in the batch
                for req in batch_requests:
                    future = req.get("future")
                    if future:
                        future["error"] = Exception(f"No batch processing available for model {model}")
                        future["completed"] = True
                
        except Exception as e:
            logger.error(f"Error processing batch for model {model}: {e}")
            
            # Set error for all futures in the batch
            for req in batch_requests:
                future = req.get("future")
                if future:
                    future["error"] = e
                    future["completed"] = True
    
    def _process_embedding_batch(self, model, batch_requests):
        # Process a batch of embedding requests for improved throughput
        try:
            # Extract texts from requests
            texts = []
            for req in batch_requests:
                data = req.get("data", {})
                text = data.get("text", data.get("input", ""))
                texts.append(text)
            
            # This is a placeholder - subclasses should implement this
            # with the actual batched embedding API call
            batch_result = {"embeddings": [[0.1, 0.2] * 50] * len(texts)}
            
            # Distribute results to individual futures
            for i, req in enumerate(batch_requests):
                future = req.get("future")
                if future and i < len(batch_result.get("embeddings", [])):
                    future["result"] = {
                        "embedding": batch_result["embeddings"][i],
                        "model": model,
                        "implementation_type": "MOCK-BATCHED"
                    }
                    future["completed"] = True
                elif future:
                    future["error"] = Exception("Batch embedding result index out of range")
                    future["completed"] = True
                    
        except Exception as e:
            # Propagate error to all futures
            for req in batch_requests:
                future = req.get("future")
                if future:
                    future["error"] = e
                    future["completed"] = True
    
    def _process_completion_batch(self, model, batch_requests):
        # Process a batch of completion requests in one API call
        try:
            # Extract prompts from requests
            prompts = []
            for req in batch_requests:
                data = req.get("data", {})
                prompt = data.get("prompt", data.get("input", ""))
                prompts.append(prompt)
            
            # This is a placeholder - subclasses should implement this
            # with the actual batched completion API call
            batch_result = {"completions": [f"Mock response for prompt {i}" for i in range(len(prompts))]}
            
            # Distribute results to individual futures
            for i, req in enumerate(batch_requests):
                future = req.get("future")
                if future and i < len(batch_result.get("completions", [])):
                    future["result"] = {
                        "text": batch_result["completions"][i],
                        "model": model,
                        "implementation_type": "MOCK-BATCHED"
                    }
                    future["completed"] = True
                elif future:
                    future["error"] = Exception("Batch completion result index out of range")
                    future["completed"] = True
                    
        except Exception as e:
            # Propagate error to all futures
            for req in batch_requests:
                future = req.get("future")
                if future:
                    future["error"] = e
                    future["completed"] = True
    