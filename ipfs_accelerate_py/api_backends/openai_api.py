"""OpenAI API Client with streaming, rate limiting, and retry support."""

import os
import time
import json
import logging
import threading
import asyncio
import aiohttp
import hashlib
import requests
from queue import Queue, PriorityQueue
from uuid import uuid4
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai_api")

class openai_api:
    """
    OpenAI API client with retry, backoff, and streaming support.
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the OpenAI API client with retry logic and queuing."""
        # Initialize with credentials from environment or metadata
        self.api_key = self._get_api_key(metadata)
        
        # Initialize queue and backoff systems
        self.max_concurrent_requests = 5
        self.queue_size = 100
        self.request_queue = []
        self.active_requests = 0
        self.queue_lock = threading.RLock()
        self.queue_processing = False
        
        # Priority levels
        self.PRIORITY_HIGH = 0
        self.PRIORITY_NORMAL = 1
        self.PRIORITY_LOW = 2
        
        # Circuit breaker settings
        self.circuit_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_threshold = 5  # Number of failures before opening circuit
        self.reset_timeout = 30  # Seconds to wait before trying half-open
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_lock = threading.RLock()
        
        # Request monitoring
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retried_requests": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "requests_by_model": {},
            "errors_by_type": {},
            "queue_wait_times": [],
            "backoff_delays": []
        }
        self.stats_lock = threading.RLock()
        
        # Enable metrics collection
        self.collect_metrics = True
        
        # Batching settings
        self.batching_enabled = True
        self.max_batch_size = 10
        self.batch_timeout = 0.5  # Max seconds to wait for more requests
        self.batch_queue = {}  # Keyed by model name
        self.batch_timers = {}  # Timers for each batch
        self.batch_lock = threading.RLock()
        
        # Models that support batching
        self.embedding_models = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
        self.completion_models = []  # OpenAI doesn't support batched completions
        self.supported_batch_models = self.embedding_models + self.completion_models
        
        # Initialize backoff configuration
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 16
        
        # Start queue processor
        self.queue_processor = threading.Thread(target=self._process_queue)
        self.queue_processor.daemon = True
        self.queue_processor.start()

    def _get_api_key(self, metadata):
        """Get API key from metadata or environment variables."""
        # Check if provided in metadata
        if metadata and "openai_api_key" in metadata:
            return metadata["openai_api_key"]
        
        # Try environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return api_key
        
        # Try to load from dotenv file
        try:
            load_dotenv()
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                return api_key
        except Exception:
            pass
        
        logger.warning("No API key found for OpenAI. Functionality will be limited.")
        return None

    def chat(self, model_name, messages, max_tokens=None, temperature=0.7, stream=False, system_prompt=None, **kwargs):
        """Send a chat request to the OpenAI Chat API."""
        # Build request body
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        # Add optional parameters if provided
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        
        if system_prompt:
            # Add system message at the beginning if not already present
            if not any(msg.get("role") == "system" for msg in messages):
                data["messages"] = [{"role": "system", "content": system_prompt}] + messages
        
        # Add streaming parameter
        data["stream"] = stream
        
        # Add any additional parameters
        for key, value in kwargs.items():
            data[key] = value
        
        endpoint = "chat/completions"
        
        # Make the request
        request_id = kwargs.get("request_id", str(uuid4()))
        
        if stream:
            return self._stream_request(endpoint, data, request_id=request_id)
        else:
            response = self.make_request(endpoint, data, request_id=request_id)
            
            # Process the response
            if "error" in response:
                logger.error(f"Error from OpenAI API: {response['error']}")
                return {
                    "text": f"Error: {response.get('error', {}).get('message', 'Unknown error')}",
                    "error": response["error"]
                }
            
            # Extract message content
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0].get("message", {}).get("content", "")
            else:
                content = ""
            
            return {
                "text": content,
                "full_response": response
            }

    def _stream_request(self, endpoint, data, request_id=None):
        """Handle streaming requests to OpenAI API."""
        if not self.api_key:
            return {"error": "API key is required"}
        
        url = f"https://api.openai.com/v1/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Stream the response
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                stream=True
            )
            
            # Check for errors
            if response.status_code != 200:
                error = response.json().get("error", {"message": f"HTTP {response.status_code}"})
                return {"error": error}
            
            # Process the streaming response
            collected_content = ""
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    
                    # Skip "data: " prefix and empty lines
                    if line_str.startswith("data: "):
                        line_data = line_str[6:]
                        
                        # Handle [DONE] message
                        if line_data == "[DONE]":
                            break
                        
                        try:
                            json_data = json.loads(line_data)
                            
                            # Extract delta content
                            choices = json_data.get("choices", [])
                            if choices and "delta" in choices[0]:
                                delta = choices[0]["delta"]
                                if "content" in delta:
                                    content = delta["content"]
                                    collected_content += content
                                    yield {
                                        "text": content,
                                        "collected_text": collected_content,
                                        "chunk": json_data,
                                        "finished": False
                                    }
                        except json.JSONDecodeError:
                            pass
            
            # Return the complete response
            return {
                "text": collected_content,
                "finished": True
            }
            
        except Exception as e:
            logger.error(f"Error in streaming request: {str(e)}")
            return {"error": str(e)}

    def make_request(self, endpoint, data, request_id=None):
        """Make a request to the OpenAI API with retries and backoff."""
        # Check circuit breaker first
        if not self.check_circuit_breaker():
            raise Exception(f"Circuit breaker is OPEN. Service appears to be unavailable. Try again in {self.reset_timeout} seconds.")
        
        # Add to queue and use backoff for retries
        future = self.queue_with_priority(
            {
                "endpoint": endpoint,
                "data": data,
                "api_key": self.api_key,
                "request_id": request_id or str(uuid4()),
                "queue_entry_time": time.time()
            },
            self.PRIORITY_NORMAL
        )
        
        # Wait for result
        wait_start = time.time()
        max_wait = 300  # 5 minutes
        
        while not future.get("completed", False) and (time.time() - wait_start) < max_wait:
            time.sleep(0.1)
        
        if not future.get("completed", False):
            raise TimeoutError(f"Request timed out after {max_wait} seconds")
        
        # Check for and propagate errors
        if future.get("error"):
            raise future["error"]
        
        return future["result"]

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

    def _execute_api_request(self, endpoint, data, api_key, request_id=None):
        """Execute the actual HTTP request to the OpenAI API."""
        start_time = time.time()
        model = data.get('model', '')
        
        # Update request count
        if hasattr(self, 'update_stats'):
            self.update_stats({
                'total_requests': 1,
                'requests_by_model': {model: 1} if model else {}
            })
            
        url = f"https://api.openai.com/v1/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Request-ID": request_id or str(uuid4())
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=60
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                # Get retry-after header if available
                retry_after = response.headers.get("Retry-After", None)
                
                if retry_after:
                    retry_delay = int(retry_after)
                else:
                    retry_delay = 2  # Default
                
                raise Exception(f"Rate limit exceeded. Retry after {retry_delay} seconds.")
            
            # Check for other error responses
            if response.status_code != 200:
                error_data = response.json() if response.text else {"message": f"HTTP {response.status_code}"}
                error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
                
                raise Exception(f"API error: {error_msg}")
            
            # Parse JSON response
            result = response.json()
            
            # Update success stats
            if hasattr(self, 'update_stats'):
                end_time = time.time()
                self.update_stats({
                    'successful_requests': 1,
                    'total_response_time': end_time - start_time,
                    'response_time': end_time - start_time
                })
                
            # Track successful request for circuit breaker
            if hasattr(self, "track_request_result"):
                self.track_request_result(True)
                
            return result
            
        except Exception as e:
            # Update failure stats
            if hasattr(self, 'update_stats'):
                end_time = time.time()
                error_type = type(e).__name__
                self.update_stats({
                    'failed_requests': 1,
                    'errors_by_type': {error_type: 1}
                })
                
            # Track failed request for circuit breaker
            if hasattr(self, "track_request_result"):
                error_type = type(e).__name__
                self.track_request_result(False, error_type)
                
            raise

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

    def _process_request(self, request_info):
        """Process a single queued request with retries and backoff."""
        try:
            # Extract request details
            endpoint = request_info["endpoint"]
            data = request_info["data"]
            api_key = request_info["api_key"]
            request_id = request_info["request_id"]
            future = request_info["future"]
            
            # Calculate queue wait time for metrics
            if "queue_entry_time" in request_info:
                queue_wait_time = time.time() - request_info["queue_entry_time"]
                if hasattr(self, "update_stats"):
                    self.update_stats({"queue_wait_times": queue_wait_time})
            
            # Process with retries and backoff
            retries = 0
            retry_delay = self.initial_retry_delay
            max_retries = self.max_retries
            
            while retries <= max_retries:
                try:
                    # Execute the API request
                    result = self._execute_api_request(endpoint, data, api_key, request_id)
                    
                    # Set result in future
                    future["result"] = result
                    future["completed"] = True
                    break
                
                except Exception as e:
                    retries += 1
                    
                    # Track retry in stats
                    if hasattr(self, 'update_stats'):
                        self.update_stats({
                            'retried_requests': 1,
                            'backoff_delays': retry_delay
                        })
                    
                    # Check if we've exhausted retries
                    if retries > max_retries:
                        # Set error in future
                        future["error"] = e
                        future["completed"] = True
                        logger.error(f"Request failed after {max_retries} retries: {str(e)}")
                        break
                    
                    # Calculate backoff delay
                    backoff_delay = min(
                        retry_delay * (self.backoff_factor ** (retries - 1)),
                        self.max_retry_delay
                    )
                    
                    logger.warning(f"Request failed: {str(e)}. Retrying in {backoff_delay}s (attempt {retries}/{max_retries})")
                    
                    # Sleep with backoff
                    time.sleep(backoff_delay)
        
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error processing request: {str(e)}")
            
            if "future" in request_info:
                request_info["future"]["error"] = e
                request_info["future"]["completed"] = True
        
        finally:
            # Decrement active requests counter
            with self.queue_lock:
                self.active_requests = max(0, self.active_requests - 1)

    def update_stats(self, stats_update):
        # Update request statistics in a thread-safe manner
        if not hasattr(self, "collect_metrics") or not self.collect_metrics:
            return
            
        with self.stats_lock:
            for key, value in stats_update.items():
                if key in self.request_stats:
                    if isinstance(self.request_stats[key], dict) and isinstance(value, dict):
                        # Update nested dictionary
                        for k, v in value.items():
                            if k in self.request_stats[key]:
                                self.request_stats[key][k] += v
                            else:
                                self.request_stats[key][k] = v
                    elif isinstance(self.request_stats[key], list) and not isinstance(value, dict):
                        # Append to list
                        self.request_stats[key].append(value)
                    elif key == "average_response_time":
                        # Special handling for average calculation
                        total = self.request_stats["total_response_time"] + stats_update.get("response_time", 0)
                        count = self.request_stats["total_requests"]
                        if count > 0:
                            self.request_stats["average_response_time"] = total / count
                    else:
                        # Simple addition for counters
                        self.request_stats[key] += value

    def get_stats(self):
        # Get a copy of the current request statistics
        if not hasattr(self, "stats_lock") or not hasattr(self, "request_stats"):
            return {}
            
        with self.stats_lock:
            # Return a copy to avoid thread safety issues
            return dict(self.request_stats)

    def reset_stats(self):
        # Reset all statistics
        if not hasattr(self, "stats_lock") or not hasattr(self, "request_stats"):
            return
            
        with self.stats_lock:
            self.request_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "retried_requests": 0,
                "average_response_time": 0,
                "total_response_time": 0,
                "requests_by_model": {},
                "errors_by_type": {},
                "queue_wait_times": [],
                "backoff_delays": []
            }

    def generate_report(self, include_details=False):
        # Generate a report of API usage and performance
        if not hasattr(self, "get_stats") or not callable(self.get_stats):
            return {"error": "Statistics not available"}
            
        stats = self.get_stats()
        
        # Build report
        report = {
            "summary": {
                "total_requests": stats.get("total_requests", 0),
                "success_rate": (stats.get("successful_requests", 0) / stats.get("total_requests", 1)) * 100 if stats.get("total_requests", 0) > 0 else 0,
                "average_response_time": stats.get("average_response_time", 0),
                "retry_rate": (stats.get("retried_requests", 0) / stats.get("total_requests", 1)) * 100 if stats.get("total_requests", 0) > 0 else 0,
            },
            "models": stats.get("requests_by_model", {}),
            "errors": stats.get("errors_by_type", {})
        }
        
        # Add circuit breaker info if available
        if hasattr(self, "circuit_state"):
            report["circuit_breaker"] = {
                "state": self.circuit_state,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "trips": stats.get("circuit_breaker_trips", 0)
            }
        
        if include_details:
            # Add detailed metrics
            queue_wait_times = stats.get("queue_wait_times", [])
            backoff_delays = stats.get("backoff_delays", [])
            
            report["details"] = {
                "queue_wait_times": {
                    "min": min(queue_wait_times) if queue_wait_times else 0,
                    "max": max(queue_wait_times) if queue_wait_times else 0,
                    "avg": sum(queue_wait_times) / len(queue_wait_times) if queue_wait_times else 0,
                    "count": len(queue_wait_times)
                },
                "backoff_delays": {
                    "min": min(backoff_delays) if backoff_delays else 0,
                    "max": max(backoff_delays) if backoff_delays else 0,
                    "avg": sum(backoff_delays) / len(backoff_delays) if backoff_delays else 0,
                    "count": len(backoff_delays)
                }
            }
        
        return report

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
                text = data.get("input", data.get("text", ""))
                if isinstance(text, list):
                    # If already a list, use the first item
                    text = text[0] if text else ""
                texts.append(text)
            
            # Prepare batch request
            api_key = self.api_key
            batch_data = {
                "model": model,
                "input": texts
            }
            
            # Make the batch API call
            result = self._execute_api_request("embeddings", batch_data, api_key)
            
            # Process results
            if "data" in result and len(result["data"]) == len(batch_requests):
                # Distribute results to individual futures
                for i, req in enumerate(batch_requests):
                    future = req.get("future")
                    if future:
                        future["result"] = {
                            "embedding": result["data"][i]["embedding"],
                            "model": model,
                            "implementation_type": "BATCHED"
                        }
                        future["completed"] = True
            else:
                # Handle unexpected response format
                raise Exception(f"Unexpected batch response format: {result}")
                    
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
            # OpenAI doesn't support batched completions, so we'll make individual requests
            # This is a placeholder for future improvements if OpenAI adds batch support
            for req in batch_requests:
                future = req.get("future")
                if future:
                    future["error"] = Exception("Batch completions not supported by OpenAI API")
                    future["completed"] = True
                    
        except Exception as e:
            # Propagate error to all futures
            for req in batch_requests:
                future = req.get("future")
                if future:
                    future["error"] = e
                    future["completed"] = True

    # API wrappers for backward compatibility
    def completions(self, model, prompt, max_tokens=16, temperature=0.7, stream=False, **kwargs):
        """Create a text completion."""
        return self.chat(
            model_name=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs
        )
        
    def __call__(self, endpoint, data, **kwargs):
        """Directly call the API with an endpoint and data."""
        return self.make_request(endpoint, data, **kwargs)