import os
import time
import threading
import hashlib
import uuid
import logging
from concurrent.futures import Future
from dotenv import load_dotenv

try:
    import boto3
    HAVE_BOTO3 = True
except ImportError:
    boto3 = None
    HAVE_BOTO3 = False

# Try to import storage wrapper
try:
    from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

# Configure logger
logger = logging.getLogger("s3_kit")

try:
    from .base import BaseAPIBackend
except ImportError:
    try:
        from base import BaseAPIBackend
    except ImportError:
        BaseAPIBackend = object

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class s3_kit(BaseAPIBackend):
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get S3 configuration from metadata or environment
        self.s3cfg = self._get_s3_config()
        
        # Initialize queue and backoff systems
        self._init_queue(queue_size=100, max_concurrent_requests=5)

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

        self._init_circuit_breaker()
        
        # Retry and backoff settings
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 60  # Maximum delay in seconds
        
        # Request tracking and metrics
        self.request_tracking = True
        self.recent_requests = {}
        self.queue_enabled = True
        self.collect_metrics = True
        self.stats_lock = threading.RLock()
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retried_requests": 0,
            "circuit_breaker_trips": 0,
            "average_latency": 0,
            "errors_by_type": {},
            "operations": {}
        }
        
        # Initialize distributed storage
        self._storage = None
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper()
                if self._storage:
                    logger.info("S3 Kit: Distributed storage initialized")
            except Exception as e:
                logger.debug(f"S3 Kit: Could not initialize storage: {e}")
        
        return None

    def _get_s3_config(self):
        """Get S3 configuration from metadata or environment"""
        # Try to get from metadata
        if "s3cfg" in self.metadata:
            return self.metadata["s3cfg"]
        
        # Try to get from environment
        s3cfg = {
            "accessKey": os.environ.get("S3_ACCESS_KEY"),
            "secretKey": os.environ.get("S3_SECRET_KEY"),
            "endpoint": os.environ.get("S3_ENDPOINT", "https://s3.amazonaws.com")
        }
        
        # Check if we have essential keys
        if s3cfg["accessKey"] and s3cfg["secretKey"]:
            return s3cfg
        
        # Try to load from dotenv
        try:
            load_dotenv()
            s3cfg = {
                "accessKey": os.environ.get("S3_ACCESS_KEY"),
                "secretKey": os.environ.get("S3_SECRET_KEY"),
                "endpoint": os.environ.get("S3_ENDPOINT", "https://s3.amazonaws.com")
            }
            
            if s3cfg["accessKey"] and s3cfg["secretKey"]:
                return s3cfg
        except ImportError:
            pass
        
        # Return default with placeholders if no config found
        return {
            "accessKey": "default_access_key",
            "secretKey": "default_secret_key",
            "endpoint": "https://s3.amazonaws.com"
        }
        
    
    def _process_queue(self):
        """Delegate to the shared BaseAPIBackend implementation."""
        return super()._process_queue()

    def _get_s3_client(self):
        """Create an S3 client"""
        return boto3.client(
            's3',
            aws_access_key_id=self.s3cfg['accessKey'],
            aws_secret_access_key=self.s3cfg['secretKey'],
            endpoint_url=self.s3cfg['endpoint']
        )
    
    def _queue_operation(self, operation, **kwargs):
        """Queue an S3 operation with backoff retry."""
        request_id = str(uuid.uuid4())

        def _execute_operation():
            if not self.check_circuit_breaker():
                raise RuntimeError("Circuit breaker is OPEN. Service unavailable.")

            retry_count = 0
            while True:
                start_time = time.time()
                try:
                    s3_client = self._get_s3_client()
                    method = getattr(s3_client, operation)
                    result = method(**kwargs)
                    latency = time.time() - start_time
                    self.track_request_result(True, operation=operation, latency=latency)
                    return result
                except Exception as e:
                    latency = time.time() - start_time
                    self.track_request_result(False, error_type=type(e).__name__, operation=operation, latency=latency)
                    if retry_count >= self.max_retries:
                        logger.error(f"Max retries exceeded for request {request_id}")
                        raise
                    delay = min(self.initial_retry_delay * (self.backoff_factor ** retry_count), self.max_retry_delay)
                    retry_count += 1
                    logger.info(f"Retrying request {request_id} after {delay}s delay (retry {retry_count}/{self.max_retries})")
                    time.sleep(delay)

        if not self.queue_enabled:
            return _execute_operation()

        with self.queue_lock:
            at_capacity = self.active_requests >= self.max_concurrent_requests
            if not at_capacity:
                self.active_requests += 1

        if at_capacity:
            result_future = self.queue_with_priority({"func": _execute_operation, "request_id": request_id}, self.PRIORITY_NORMAL)
            wait_start = time.time()
            while not result_future["completed"] and (time.time() - wait_start) < 300:
                time.sleep(0.05)
            if not result_future["completed"]:
                raise TimeoutError("Request timed out waiting in queue")
            if result_future["error"]:
                raise result_future["error"]
            return result_future["result"]

        try:
            return _execute_operation()
        finally:
            with self.queue_lock:
                self.active_requests = max(0, self.active_requests - 1)

    def upload_file(self, file_path, bucket, key):
        """Upload a file to S3"""
        return self._queue_operation(
            "upload_file",
            Filename=file_path,
            Bucket=bucket,
            Key=key
        )
    
    def download_file(self, bucket, key, file_path):
        """Download a file from S3"""
        return self._queue_operation(
            "download_file",
            Bucket=bucket,
            Key=key,
            Filename=file_path
        )
    
    def list_objects(self, bucket, prefix=None):
        """List objects in a bucket"""
        kwargs = {"Bucket": bucket}
        if prefix:
            kwargs["Prefix"] = prefix
            
        return self._queue_operation("list_objects_v2", **kwargs)
    
    def delete_object(self, bucket, key):
        """Delete an object from S3"""
        return self._queue_operation(
            "delete_object",
            Bucket=bucket,
            Key=key
        )
    
    def head_object(self, bucket, key):
        """Get object metadata"""
        return self._queue_operation(
            "head_object",
            Bucket=bucket,
            Key=key
        )
    
    def create_presigned_url(self, bucket, key, expiration=3600):
        """Create a presigned URL for an object"""
        s3_client = self._get_s3_client()
        return s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        
    def create_s3_endpoint_handler(self, endpoint_url=None, access_key=None, secret_key=None, max_concurrent=None, circuit_breaker_threshold=None, retries=None):
        """
        Create an endpoint handler for S3 operations.
        
        Each endpoint handler gets its own configuration including:
        - Endpoint URL
        - Access and secret keys
        - Queue and backoff settings
        - Circuit breaker configuration
        
        This allows for multiple endpoint handlers with different configs.
        """
        # Create a separate configuration for this endpoint handler
        endpoint_config = {
            "endpoint": endpoint_url or self.s3cfg["endpoint"],
            "accessKey": access_key or self.s3cfg["accessKey"],
            "secretKey": secret_key or self.s3cfg["secretKey"],
            "max_concurrent_requests": max_concurrent or self.max_concurrent_requests,
            "circuit_breaker_threshold": circuit_breaker_threshold or self.failure_threshold,
            "max_retries": retries or self.max_retries,
            "active_requests": 0,
            "failure_count": 0,
            "circuit_state": "CLOSED",
            "last_failure_time": 0
        }
        
        # Create an S3 client for this specific endpoint
        def get_endpoint_s3_client():
            return boto3.client(
                's3',
                aws_access_key_id=endpoint_config["accessKey"],
                aws_secret_access_key=endpoint_config["secretKey"],
                endpoint_url=endpoint_config["endpoint"]
            )
        
        # Create thread synchronization locks specific to this endpoint
        endpoint_lock = threading.RLock()
        circuit_lock = threading.RLock()
        
        # Check circuit breaker state for this endpoint
        def check_endpoint_circuit():
            with circuit_lock:
                now = time.time()
                
                if endpoint_config["circuit_state"] == "OPEN":
                    # Check if enough time has passed to try again
                    if now - endpoint_config["last_failure_time"] > self.reset_timeout:
                        logger.info(f"Endpoint {endpoint_config['endpoint']}: Circuit breaker transitioning from OPEN to HALF-OPEN")
                        endpoint_config["circuit_state"] = "HALF-OPEN"
                        return True
                    else:
                        # Circuit is open, fail fast
                        return False
                        
                elif endpoint_config["circuit_state"] == "HALF-OPEN":
                    # In half-open state, we allow a single request to test the service
                    return True
                    
                else:  # CLOSED
                    # Normal operation, allow requests
                    return True
        
        # Track request results for this endpoint
        def track_endpoint_result(success, error_type=None, operation=None, latency=None):
            with circuit_lock:
                if success:
                    # Successful request
                    if endpoint_config["circuit_state"] == "HALF-OPEN":
                        # Service is working again, close the circuit
                        logger.info(f"Endpoint {endpoint_config['endpoint']}: Circuit breaker transitioning from HALF-OPEN to CLOSED")
                        endpoint_config["circuit_state"] = "CLOSED"
                        endpoint_config["failure_count"] = 0
                    elif endpoint_config["circuit_state"] == "CLOSED":
                        # Reset failure count on success
                        endpoint_config["failure_count"] = 0
                else:
                    # Failed request
                    endpoint_config["failure_count"] += 1
                    endpoint_config["last_failure_time"] = time.time()
                    
                    if endpoint_config["circuit_state"] == "CLOSED" and endpoint_config["failure_count"] >= endpoint_config["circuit_breaker_threshold"]:
                        # Too many failures, open the circuit
                        logger.warning(f"Endpoint {endpoint_config['endpoint']}: Circuit breaker transitioning from CLOSED to OPEN after {endpoint_config['failure_count']} failures")
                        endpoint_config["circuit_state"] = "OPEN"
                        
                    elif endpoint_config["circuit_state"] == "HALF-OPEN":
                        # Failed during test request, back to open
                        logger.warning(f"Endpoint {endpoint_config['endpoint']}: Circuit breaker transitioning from HALF-OPEN to OPEN after test request failure")
                        endpoint_config["circuit_state"] = "OPEN"
            
            # Also update global metrics
            if self.collect_metrics:
                self.update_metrics(
                    operation=operation,
                    success=success,
                    latency=latency,
                    error_type=error_type
                )
                
        # Create both sync and async handlers
        def sync_endpoint_handler(operation, **kwargs):
            """Handle synchronous requests to S3 endpoint"""
            # Check if we can make a request based on circuit breaker
            if not check_endpoint_circuit():
                error_msg = f"Circuit breaker is open for endpoint {endpoint_config['endpoint']} - service unavailable"
                logger.warning(error_msg)
                return {"error": error_msg, "circuit_breaker": "OPEN"}
                
            # Check if we're at capacity for this endpoint
            with endpoint_lock:
                if endpoint_config["active_requests"] >= endpoint_config["max_concurrent_requests"]:
                    error_msg = f"Too many concurrent requests for endpoint {endpoint_config['endpoint']}"
                    logger.warning(error_msg)
                    return {"error": error_msg, "retry_after": 1}
                    
                # Increment active requests counter
                endpoint_config["active_requests"] += 1
                
            # Execute operation with retry logic
            start_time = time.time()
            retry_count = 0
            last_error = None
            
            try:
                while retry_count <= endpoint_config["max_retries"]:
                    try:
                        s3_client = get_endpoint_s3_client()
                        
                        # Execute the appropriate operation
                        result = None
                        if operation == "upload_file":
                            result = s3_client.upload_file(
                                kwargs.get("file_path"),
                                kwargs.get("bucket"),
                                kwargs.get("key")
                            )
                        elif operation == "download_file":
                            result = s3_client.download_file(
                                kwargs.get("bucket"),
                                kwargs.get("key"),
                                kwargs.get("file_path")
                            )
                        elif operation == "list_objects":
                            bucket = kwargs.get("bucket")
                            prefix = kwargs.get("prefix")
                            list_args = {"Bucket": bucket}
                            if prefix:
                                list_args["Prefix"] = prefix
                            result = s3_client.list_objects_v2(**list_args)
                        elif operation == "delete_object":
                            result = s3_client.delete_object(
                                Bucket=kwargs.get("bucket"),
                                Key=kwargs.get("key")
                            )
                        elif operation == "head_object":
                            result = s3_client.head_object(
                                Bucket=kwargs.get("bucket"),
                                Key=kwargs.get("key")
                            )
                        elif operation == "create_presigned_url":
                            result = s3_client.generate_presigned_url(
                                'get_object',
                                Params={
                                    'Bucket': kwargs.get("bucket"),
                                    'Key': kwargs.get("key")
                                },
                                ExpiresIn=kwargs.get("expiration", 3600)
                            )
                        else:
                            error_msg = f"Unsupported operation: {operation}"
                            logger.error(error_msg)
                            return {"error": error_msg}
                            
                        # Calculate latency
                        latency = time.time() - start_time
                        
                        # Track successful request
                        track_endpoint_result(
                            success=True,
                            operation=operation,
                            latency=latency
                        )
                        
                        return result
                        
                    except Exception as e:
                        last_error = e
                        # Calculate latency for error
                        latency = time.time() - start_time
                        
                        # Track failed request
                        track_endpoint_result(
                            success=False,
                            error_type=type(e).__name__,
                            operation=operation,
                            latency=latency
                        )
                        
                        # Check if we should retry
                        if retry_count < endpoint_config["max_retries"]:
                            # Calculate backoff delay
                            delay = min(
                                self.initial_retry_delay * (self.backoff_factor ** retry_count),
                                self.max_retry_delay
                            )
                            
                            logger.info(f"Retrying {operation} for endpoint {endpoint_config['endpoint']} after {delay}s delay (retry {retry_count + 1}/{endpoint_config['max_retries']})")
                            
                            # Wait for backoff delay
                            time.sleep(delay)
                            retry_count += 1
                        else:
                            # Max retries exceeded
                            logger.error(f"Max retries exceeded for {operation} on endpoint {endpoint_config['endpoint']}")
                            return {"error": str(e), "max_retries_exceeded": True}
                
                # Should not reach here, but just in case
                return {"error": str(last_error) if last_error else "Unknown error"}
                
            finally:
                # Decrement active requests counter
                with endpoint_lock:
                    endpoint_config["active_requests"] = max(0, endpoint_config["active_requests"] - 1)
            
        async def async_endpoint_handler(operation, **kwargs):
            """Handle asynchronous requests to S3 endpoint"""
            try:
                return sync_endpoint_handler(operation, **kwargs)
            except Exception as e:
                logger.error(f"Error in async S3 operation {operation} for endpoint {endpoint_config['endpoint']}: {e}")
                return {"error": str(e)}
        
        # Create a callable endpoint handler function
        def endpoint_handler(*args, **kwargs):
            if kwargs.get("async_mode", False):
                return async_endpoint_handler(*args, **kwargs)
            else:
                return sync_endpoint_handler(*args, **kwargs)
                
        # Add metadata to the handler
        endpoint_handler.metadata = {
            "endpoint_url": endpoint_config["endpoint"],
            "implementation_type": "REAL",
            "queue_processing": True,
            "backoff_enabled": True,
            "circuit_breaker_enabled": True,
            "max_concurrent": endpoint_config["max_concurrent_requests"],
            "max_retries": endpoint_config["max_retries"]
        }
        
        return endpoint_handler
    
    def test_s3_endpoint(self, endpoint_url=None, access_key=None, secret_key=None):
        """
        Test a specific S3 endpoint configuration.
        
        This can be used to test different endpoint configurations without modifying
        the main S3 Kit instance's configuration.
        
        Args:
            endpoint_url: The S3 endpoint URL to test
            access_key: The access key to use for authentication
            secret_key: The secret key to use for authentication
            
        Returns:
            dict: Result of the test with success status and bucket information if successful
        """
        # Use provided values or defaults from config
        endpoint = endpoint_url or self.s3cfg["endpoint"]
        access = access_key or self.s3cfg["accessKey"]
        secret = secret_key or self.s3cfg["secretKey"]
            
        try:
            # Create a specific client for this test
            s3_client = boto3.client(
                's3',
                aws_access_key_id=access,
                aws_secret_access_key=secret,
                endpoint_url=endpoint
            )
            
            # Try to list buckets as a basic connectivity test
            response = s3_client.list_buckets()
            
            # Return success with bucket information
            return {
                "success": True,
                "endpoint": endpoint,
                "bucket_count": len(response.get("Buckets", [])),
                "buckets": [b["Name"] for b in response.get("Buckets", [])]
            }
        except Exception as e:
            error_msg = f"Error testing S3 endpoint {endpoint}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "endpoint": endpoint,
                "error": str(e)
            }
    
    def get_metrics(self):
        """Get current metrics for monitoring and reporting"""
        if not self.collect_metrics:
            return {"metrics_collection": "disabled"}
            
        with self.stats_lock:
            # Return a deep copy to avoid thread safety issues
            import copy
            return copy.deepcopy(self.request_stats)
    
    def reset_metrics(self):
        """Reset all metrics counters"""
        if not self.collect_metrics:
            return
            
        with self.stats_lock:
            self.request_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "retried_requests": 0,
                "circuit_breaker_trips": 0,
                "average_latency": 0,
                "errors_by_type": {},
                "operations": {}
            }
            
    def get_status(self):
        """Get current status of the S3 Kit API backend"""
        return {
            "endpoint": self.s3cfg["endpoint"],
            "queue_enabled": self.queue_enabled,
            "queue_size": len(self.request_queue),
            "active_requests": self.active_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
            "circuit_state": self.circuit_state,
            "failure_count": self.failure_count,
            "metrics_collection": self.collect_metrics,
            "implementation_type": "REAL"
        }
    def update_metrics(self, operation=None, success=True, latency=None, error_type=None, retried=False):
        """Update metrics for monitoring and reporting"""
        if not self.collect_metrics:
            return
            
        with self.stats_lock:
            # Update basic counters
            self.request_stats["total_requests"] += 1
            
            if success:
                self.request_stats["successful_requests"] += 1
            else:
                self.request_stats["failed_requests"] += 1
                
            if retried:
                self.request_stats["retried_requests"] += 1
                
            # Track operation-specific metrics
            if operation:
                if operation not in self.request_stats["operations"]:
                    self.request_stats["operations"][operation] = {
                        "count": 0,
                        "failures": 0,
                        "latency_sum": 0,
                        "average_latency": 0
                    }
                    
                self.request_stats["operations"][operation]["count"] += 1
                
                if not success:
                    self.request_stats["operations"][operation]["failures"] += 1
                    
                if latency:
                    self.request_stats["operations"][operation]["latency_sum"] += latency
                    self.request_stats["operations"][operation]["average_latency"] = (
                        self.request_stats["operations"][operation]["latency_sum"] / 
                        self.request_stats["operations"][operation]["count"]
                    )
            
            # Track errors by type
            if error_type and not success:
                if error_type not in self.request_stats["errors_by_type"]:
                    self.request_stats["errors_by_type"][error_type] = 0
                self.request_stats["errors_by_type"][error_type] += 1
                
            # Update average latency
            if latency:
                current_total = self.request_stats["average_latency"] * (self.request_stats["total_requests"] - 1)
                self.request_stats["average_latency"] = (current_total + latency) / self.request_stats["total_requests"]

    def track_request_result(self, success, error_type=None, operation=None, latency=None):
        """Track the result of a request for circuit breaker logic and metrics"""
        # Update metrics first
        if self.collect_metrics:
            self.update_metrics(
                operation=operation,
                success=success,
                latency=latency,
                error_type=error_type
            )
        
        # Handle circuit breaker state
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
                
                if self.circuit_state == "CLOSED" and self.failure_count >= self.failure_threshold:
                    # Too many failures, open the circuit
                    logger.warning(f"Circuit breaker transitioning from CLOSED to OPEN after {self.failure_count} failures")
                    self.circuit_state = "OPEN"
                    
                    # Update circuit breaker statistics
                    if self.collect_metrics:
                        with self.stats_lock:
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
    