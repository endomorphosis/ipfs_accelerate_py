# Fixed openai_api implementation with API improvements
# Added endpoint management, request tracking, and queuing mechanisms
import time
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()
import requests
import os
import threading
import hashlib

# Configure logging
logger = logging.getLogger(__name__)


class openai_api:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources or {}
        self.metadata = metadata or {}
        self.api_key = metadata.get("openai_api_key", "") if metadata else ""
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


    def _process_queue(self):
        '''Process requests in the queue in FIFO order'''
        with self.queue_lock:
            if self.queue_processing:
                return  # Another thread is already processing the queue
            self.queue_processing = True
        
        logger.info("Starting queue processing thread")
        
        try:
            while True:
                # Get the next request from the queue
                with self.queue_lock:
                    if not self.request_queue:
                        self.queue_processing = False
                        break
                        
                    # Check if we're at the concurrent request limit
                    if self.current_requests >= self.max_concurrent_requests:
                        # Sleep briefly then check again
                        time.sleep(0.1)
                        continue
                        
                    # Get the next request and increase counter
                    request_info = self.request_queue.pop(0)
                    self.current_requests += 1
                
                # Process the request outside the lock
                try:
                    # Extract request details
                    request_function = request_info["function"]
                    args = request_info["args"]
                    kwargs = request_info["kwargs"]
                    future = request_info["future"]
                    request_id = request_info.get("request_id")
                    
                    # Make the request (without queueing again)
                    # Save original queue_enabled value
                    original_queue_enabled = self.queue_enabled
                    self.queue_enabled = False  # Disable queueing to prevent recursion
                    
                    try:
                        # Make the request
                        result = request_function(*args, **kwargs)
                        
                        # Store result in future
                        future["result"] = result
                        future["completed"] = True
                        
                    except Exception as e:
                        # Store error in future
                        future["error"] = e
                        future["completed"] = True
                        logger.error(f"Error processing queued request: {str(e)}")
                    
                    finally:
                        # Restore original queue_enabled value
                        self.queue_enabled = original_queue_enabled
                
                finally:
                    # Decrement counter
                    with self.queue_lock:
                        self.current_requests = max(0, self.current_requests - 1)
                
                # Brief pause to prevent CPU hogging
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in queue processing thread: {str(e)}")
            
        finally:
            with self.queue_lock:
                self.queue_processing = False
                
            logger.info("Queue processing thread finished")

    def _with_queue_and_backoff(self, func):
        '''Decorator to handle queue and backoff for API requests'''
        def wrapper(*args, **kwargs):
            # Generate request ID if not provided
            request_id = kwargs.get("request_id")
            if request_id is None:
                request_id = f"req_{int(time.time())}_{hashlib.md5(str(args).encode()).hexdigest()[:8]}"
                kwargs["request_id"] = request_id
            
            # If queue is enabled and we're at capacity, add to queue
            if hasattr(self, "queue_enabled") and self.queue_enabled:
                with self.queue_lock:
                    if self.current_requests >= self.max_concurrent_requests:
                        # Create a future to store the result
                        result_future = {"result": None, "error": None, "completed": False}
                        
                        # Add to queue with all necessary info to process later
                        request_info = {
                            "function": func,
                            "args": args,
                            "kwargs": kwargs,
                            "future": result_future,
                            "request_id": request_id
                        }
                        
                        # Check if queue is full
                        if len(self.request_queue) >= self.queue_size:
                            raise ValueError(f"Request queue is full ({self.queue_size} items). Try again later.")
                        
                        # Add to queue
                        self.request_queue.append(request_info)
                        logger.info(f"Request queued. Queue size: {len(self.request_queue)}")
                        
                        # Start queue processing if not already running
                        if not self.queue_processing:
                            threading.Thread(target=self._process_queue).start()
                        
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
                    self.current_requests += 1
            
            # Use exponential backoff retry mechanism
            retries = 0
            retry_delay = self.initial_retry_delay if hasattr(self, "initial_retry_delay") else 1
            max_retries = self.max_retries if hasattr(self, "max_retries") else 3
            backoff_factor = self.backoff_factor if hasattr(self, "backoff_factor") else 2
            max_retry_delay = self.max_retry_delay if hasattr(self, "max_retry_delay") else 60
            
            while True:
                try:
                    # Make the actual API call
                    result = func(*args, **kwargs)
                    
                    # Decrement counter if queue enabled
                    if hasattr(self, "queue_enabled") and self.queue_enabled:
                        with self.queue_lock:
                            self.current_requests = max(0, self.current_requests - 1)
                    
                    return result
                    
                except openai.RateLimitError as e:
                    # Handle rate limit errors with backoff
                    if retries < max_retries:
                        # Check if the API returned a retry-after header
                        retry_after = e.headers.get("retry-after") if hasattr(e, "headers") else None
                        if retry_after and retry_after.isdigit():
                            retry_delay = int(retry_after)
                        
                        logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds (attempt {retries+1}/{max_retries})...")
                        time.sleep(retry_delay)
                        retries += 1
                        retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} attempts: {str(e)}")
                        
                        # Decrement counter if queue enabled
                        if hasattr(self, "queue_enabled") and self.queue_enabled:
                            with self.queue_lock:
                                self.current_requests = max(0, self.current_requests - 1)
                        
                        raise
                
                except openai.APIError as e:
                    # Handle transient API errors with backoff
                    if retries < max_retries:
                        logger.warning(f"API error: {str(e)}. Retrying in {retry_delay} seconds (attempt {retries+1}/{max_retries})...")
                        time.sleep(retry_delay)
                        retries += 1
                        retry_delay = min(retry_delay * backoff_factor, max_retry_delay)
                    else:
                        logger.error(f"API error after {max_retries} attempts: {str(e)}")
                        
                        # Decrement counter if queue enabled
                        if hasattr(self, "queue_enabled") and self.queue_enabled:
                            with self.queue_lock:
                                self.current_requests = max(0, self.current_requests - 1)
                        
                        raise
                
                except Exception as e:
                    # For other exceptions, don't retry
                    logger.error(f"Request error: {str(e)}")
                    
                    # Decrement counter if queue enabled
                    if hasattr(self, "queue_enabled") and self.queue_enabled:
                        with self.queue_lock:
                            self.current_requests = max(0, self.current_requests - 1)
                    
                    raise
        
        return wrapper
