import os
import json
import time
import threading
import hashlib
import uuid
import requests
import numpy as np
from concurrent.futures import Future
from queue import Queue
from dotenv import load_dotenv

# Try to import storage wrapper
try:
    from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

# Try to import datasets integration for API tracking
try:
    from ..datasets_integration import (
        is_datasets_available,
        ProvenanceLogger,
        DatasetsManager
    )
    HAVE_DATASETS_INTEGRATION = True
except ImportError:
    try:
        from datasets_integration import (
            is_datasets_available,
            ProvenanceLogger,
            DatasetsManager
        )
        HAVE_DATASETS_INTEGRATION = True
    except ImportError:
        HAVE_DATASETS_INTEGRATION = False
        is_datasets_available = lambda: False
        ProvenanceLogger = None
        DatasetsManager = None

class ovms:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get OVMS configuration from metadata or environment
        self.api_url = self.metadata.get("ovms_api_url", os.environ.get("OVMS_API_URL", "http://localhost:9000"))
        self.default_model = self.metadata.get("ovms_model", os.environ.get("OVMS_MODEL", "model"))
        self.default_version = self.metadata.get("ovms_version", os.environ.get("OVMS_VERSION", "latest"))
        self.default_precision = self.metadata.get("ovms_precision", os.environ.get("OVMS_PRECISION", "FP32"))
        
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
        self.queue_processing = False
        
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
        
        # Initialize per-endpoint settings
        self.queue_enabled = True
        self.endpoints = {}
        
        # API authentication - OVMS may require authentication in some deployments
        self.api_key = self.metadata.get("ovms_api_key", os.environ.get("OVMS_API_KEY", None))
        
        # Initialize thread pool for async processing
        try:
            from concurrent.futures import ThreadPoolExecutor
            self.thread_pool = ThreadPoolExecutor(max_workers=10)
        except ImportError:
            self.thread_pool = None
        
        # Initialize distributed storage
        self._storage = None
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper()
                if self._storage:
                    import logging
                    logging.getLogger(__name__).info("OVMS: Distributed storage initialized")
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"OVMS: Could not initialize storage: {e}")
        
        return None
        
    
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

    def make_post_request_ovms(self, endpoint_url, data, request_id=None, endpoint_id=None, api_key=None):
        """Make a request to OVMS API with queue and backoff and endpoint-specific settings"""
        # Use endpoint-specific settings if available
        if endpoint_id and endpoint_id in self.endpoints:
            # Get endpoint settings
            endpoint = self.endpoints[endpoint_id]
            
            # Generate unique request ID if not provided
            if request_id is None:
                request_id = f"req_{int(time.time())}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"
            
            # Queue system with proper concurrency management
            future = Future()
            
            # Create request info
            request_info = {
                "endpoint_url": endpoint_url,
                "data": data,
                "api_key": api_key or endpoint.get("api_key"),
                "request_id": request_id,
                "endpoint_id": endpoint_id,
                "future": future
            }
            
            # Check if endpoint queue is enabled and at capacity
            if endpoint.get("queue_enabled", True) and endpoint.get("current_requests", 0) >= endpoint.get("max_concurrent_requests", 5):
                with endpoint["queue_lock"]:
                    # Check if queue is full
                    if len(endpoint["request_queue"]) >= endpoint.get("queue_size", 100):
                        raise ValueError(f"Request queue is full ({endpoint['queue_size']} items). Try again later.")
                    
                    # Add to endpoint queue
                    endpoint["request_queue"].append(request_info)
                    print(f"Request queued. Queue size: {len(endpoint['request_queue'])}. Request ID: {request_id}")
                    
                    # Start queue processing if not already running
                    if not endpoint.get("queue_processing", False):
                        threading.Thread(target=self._process_queue, args=(endpoint_id,)).start()
                
                # Wait for result with timeout
                wait_start = time.time()
                max_wait = 300  # 5 minutes
                
                while not request_info["future"].get("completed", False) and (time.time() - wait_start) < max_wait:
                    time.sleep(0.1)
                
                # Check if completed or timed out
                if not request_info["future"].get("completed", False):
                    raise TimeoutError(f"Request timed out after {max_wait} seconds in queue")
                
                # Propagate error if any
                if request_info["future"].get("error"):
                    raise request_info["future"]["error"]
                
                return request_info["future"]["result"]
            else:
                # Process request directly without queueing
                with endpoint["queue_lock"]:
                    endpoint["current_requests"] = endpoint.get("current_requests", 0) + 1
                
                # Add to regular queue
                self.request_queue.append((future, endpoint_url, data, request_id))
                
                # Get result and update endpoint stats
                try:
                    result = future.result()
                    
                    # Update endpoint statistics
                    with endpoint["queue_lock"]:
                        endpoint["current_requests"] = max(0, endpoint.get("current_requests", 0) - 1)
                        endpoint["successful_requests"] = endpoint.get("successful_requests", 0) + 1
                        endpoint["last_request_at"] = time.time()
                    
                    return result
                except Exception as e:
                    # Update endpoint statistics
                    with endpoint["queue_lock"]:
                        endpoint["current_requests"] = max(0, endpoint.get("current_requests", 0) - 1)
                        endpoint["failed_requests"] = endpoint.get("failed_requests", 0) + 1
                        endpoint["last_request_at"] = time.time()
                    
                    raise
        else:
            # Use global settings
            # Generate unique request ID if not provided
            if request_id is None:
                request_id = f"req_{int(time.time())}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"
            
            # Queue system with proper concurrency management
            future = Future()
            
            # Add to queue
            self.request_queue.append((future, endpoint_url, data, request_id))
            
            # Get result (blocks until request is processed)
            return future.result()
    
    def format_request(self, handler, data, model=None, version=None):
        """Format a request for OVMS"""
        # Use default model and version if not provided
        model = model or self.default_model
        version = version or self.default_version
        
        # Format data based on its type
        if isinstance(data, dict):
            # If data is already formatted as expected by OVMS, use it directly
            if "instances" in data:
                formatted_data = data
            # If data has a 'data' field, wrap it in the OVMS format
            elif "data" in data:
                formatted_data = {"instances": [data]}
            # Otherwise, create a standard format
            else:
                formatted_data = {"instances": [{"data": data}]}
        elif isinstance(data, list):
            # If it's a list of objects with 'data' field, format as instances
            if len(data) > 0 and isinstance(data[0], dict) and "data" in data[0]:
                formatted_data = {"instances": data}
            # If it's a nested list (e.g., a batch of inputs)
            elif len(data) > 0 and isinstance(data[0], list):
                formatted_data = {"instances": [{"data": item} for item in data]}
            # Otherwise, treat as a single data array
            else:
                formatted_data = {"instances": [{"data": data}]}
        else:
            # For other data types, convert to list and wrap in standard format
            formatted_data = {"instances": [{"data": [data]}]}
        
        # Add model version if specified
        if version and version != "latest":
            formatted_data["version"] = version
        
        # Make the request
        return handler(formatted_data)
    
    def infer(self, model=None, data=None, version=None, precision=None):
        """Run inference on a model"""
        # Use defaults if not provided
        model = model or self.default_model
        version = version or self.default_version
        precision = precision or self.default_precision
        
        # Construct endpoint URL
        endpoint_url = f"{self.api_url}/v1/models/{model}:predict"
        
        # Create a handler function for this request
        def handler(formatted_data):
            return self.make_post_request_ovms(endpoint_url, formatted_data)
        
        # Format and send the request
        response = self.format_request(handler, data, model, version)
        
        # Process response
        if "predictions" in response:
            # Return just the predictions for simplicity
            return response["predictions"]
        else:
            # Return the full response if not in expected format
            return response
    
    def batch_infer(self, model=None, data_batch=None, version=None, precision=None):
        """Run batch inference on a model"""
        # Use defaults if not provided
        model = model or self.default_model
        version = version or self.default_version
        precision = precision or self.default_precision
        
        # Construct endpoint URL
        endpoint_url = f"{self.api_url}/v1/models/{model}:predict"
        
        # Format batch data
        if isinstance(data_batch, list):
            # Format each item in the batch
            formatted_data = {"instances": []}
            
            for item in data_batch:
                if isinstance(item, dict) and "data" in item:
                    # Already in the right format
                    formatted_data["instances"].append(item)
                else:
                    # Convert to standard format
                    formatted_data["instances"].append({"data": item})
        else:
            # Not a batch, treat as single instance
            formatted_data = {"instances": [{"data": data_batch}]}
        
        # Add version if specified
        if version and version != "latest":
            formatted_data["version"] = version
        
        # Make the request
        response = self.make_post_request_ovms(endpoint_url, formatted_data)
        
        # Process response
        if "predictions" in response:
            # Return just the predictions for simplicity
            return response["predictions"]
        else:
            # Return the full response if not in expected format
            return response
    
    def get_model_metadata(self, model=None, version=None):
        """Get model metadata"""
        # Use defaults if not provided
        model = model or self.default_model
        version = version or self.default_version
        
        # Construct endpoint URL
        endpoint_url = f"{self.api_url}/v1/models/{model}"
        if version and version != "latest":
            endpoint_url += f"/versions/{version}"
        endpoint_url += "/metadata"
        
        # Make the request
        try:
            response = requests.get(
                endpoint_url,
                headers={"Content-Type": "application/json"},
                timeout=self.metadata.get("timeout", 30)
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting model metadata: {e}")
            return {"error": str(e)}
    
    def get_model_status(self, model=None, version=None):
        """Get model status"""
        # Use defaults if not provided
        model = model or self.default_model
        version = version or self.default_version
        
        # Construct endpoint URL
        endpoint_url = f"{self.api_url}/v1/models/{model}"
        if version and version != "latest":
            endpoint_url += f"/versions/{version}"
        
        # Make the request
        try:
            response = requests.get(
                endpoint_url,
                headers={"Content-Type": "application/json"},
                timeout=self.metadata.get("timeout", 30)
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting model status: {e}")
            return {"error": str(e)}
    
    def get_server_statistics(self):
        """Get server statistics"""
        # Construct endpoint URL
        endpoint_url = f"{self.api_url}/v1/statistics"
        
        # Make the request
        try:
            response = requests.get(
                endpoint_url,
                headers={"Content-Type": "application/json"},
                timeout=self.metadata.get("timeout", 30)
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting server statistics: {e}")
            return {"error": str(e)}
            
    def create_ovms_endpoint_handler(self, endpoint_url=None):
        """Create an endpoint handler for OVMS"""
        async def endpoint_handler(data, **kwargs):
            """Handle requests to OVMS endpoint"""
            try:
                # Use specified model or default
                model = kwargs.get("model", self.default_model)
                
                # Use specified version or default
                version = kwargs.get("version", self.default_version)
                
                # Use specified endpoint or construct from model
                if not endpoint_url:
                    # Construct endpoint URL for the model
                    actual_endpoint = f"{self.api_url}/v1/models/{model}:predict"
                else:
                    actual_endpoint = endpoint_url
                
                # Check if this is a batch request
                is_batch = kwargs.get("batch", False)
                
                if is_batch:
                    # Process as batch
                    return self.batch_infer(model, data, version)
                else:
                    # Process as single inference
                    return self.infer(model, data, version)
            except Exception as e:
                print(f"Error calling OVMS endpoint: {e}")
                return {"error": str(e), "implementation_type": "(ERROR)"}
        
        return endpoint_handler
        
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
            "total_processing_time": 0,
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
                "total_processing_time": endpoint["total_processing_time"],
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
            total_processing_time = sum(e["total_processing_time"] for e in self.endpoints.values()) if self.endpoints else 0
            
            stats = {
                "endpoints_count": len(self.endpoints),
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "total_processing_time": total_processing_time,
                "global_queue_size": self.request_queue.qsize() if hasattr(self.request_queue, 'qsize') else len(self.request_queue) if isinstance(self.request_queue, list) else 0,
                "global_current_requests": self.active_requests
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
            endpoint["total_processing_time"] = 0
        elif endpoint_id is None:
            # Reset stats for all endpoints
            for endpoint in self.endpoints.values():
                endpoint["total_requests"] = 0
                endpoint["successful_requests"] = 0
                endpoint["failed_requests"] = 0
                endpoint["total_processing_time"] = 0
        else:
            raise ValueError(f"Endpoint {endpoint_id} not found")
            
    def make_request_with_endpoint(self, endpoint_id, data, model=None, version=None, endpoint_url=None):
        """Make a request using a specific endpoint"""
        # Get the endpoint
        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_id} not found")
        
        endpoint = self.endpoints[endpoint_id]
        
        # Use default model and version if not provided
        model = model or self.default_model
        version = version or self.default_version
        
        # Construct endpoint URL if not provided
        if not endpoint_url:
            endpoint_url = f"{self.api_url}/v1/models/{model}:predict"
        
        # Generate a unique request ID
        request_id = f"req_{int(time.time())}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"
        
        # Track start time for performance metrics
        start_time = time.time()
        
        try:
            # Make the request using this endpoint's settings
            response = self.make_post_request_ovms(
                endpoint_url=endpoint_url,
                data=data,
                request_id=request_id,
                endpoint_id=endpoint_id,
                api_key=endpoint.get("api_key")
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update endpoint statistics
            with endpoint["queue_lock"]:
                endpoint["total_processing_time"] += processing_time
            
            return response
        except Exception as e:
            # Update stats in case of error
            with endpoint["queue_lock"]:
                endpoint["last_request_at"] = time.time()
            
            # Re-raise the exception
            raise
    
    def test_ovms_endpoint(self, endpoint_url=None, model_name=None, endpoint_id=None, request_id=None):
        """Test the OVMS endpoint"""
        # Use provided values or defaults
        model_name = model_name or self.default_model
        
        if not endpoint_url:
            endpoint_url = f"{self.api_url}/v1/models/{model_name}:predict"
            
        # Use endpoint-specific settings if provided
        if endpoint_id and endpoint_id in self.endpoints:
            # Generate unique request ID for the test if not provided
            if request_id is None:
                request_id = f"test_{int(time.time())}_{hashlib.md5(model_name.encode()).hexdigest()[:8]}"
                
            # Simple test data
            test_data = {"instances": [{"data": [1.0, 2.0, 3.0, 4.0]}]}
            
            try:
                # Make the request using the specified endpoint
                response = self.make_request_with_endpoint(
                    endpoint_id=endpoint_id,
                    data=test_data,
                    model=model_name,
                    endpoint_url=endpoint_url
                )
                
                # Check if the response contains predictions
                success = "predictions" in response
                
                # Create detailed test result
                result = {
                    "success": success,
                    "model": model_name,
                    "implementation_type": "(REAL)" if success else "(ERROR)",
                    "request_id": request_id,
                    "endpoint_id": endpoint_id
                }
                
                if not success:
                    result["message"] = "Response did not contain 'predictions' field"
                    
                return result
            except Exception as e:
                error_message = str(e)
                print(f"Error testing OVMS endpoint: {error_message}")
                
                # Create error result
                return {
                    "success": False,
                    "message": f"Error: {error_message}",
                    "model": model_name,
                    "implementation_type": "(ERROR)",
                    "request_id": request_id,
                    "endpoint_id": endpoint_id
                }
        else:
            # Use global settings
            try:
                # Simple test data
                test_data = {"instances": [{"data": [1.0, 2.0, 3.0, 4.0]}]}
                
                # Make the request
                response = self.make_post_request_ovms(endpoint_url, test_data)
                
                # Check if the response contains predictions
                return "predictions" in response
            except Exception as e:
                print(f"Error testing OVMS endpoint: {e}")
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
    