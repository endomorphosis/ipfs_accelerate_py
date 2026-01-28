#!/usr/bin/env python
"""
LLVM API Backend Implementation

This module provides a complete implementation of the LLVM API backend,
including:
- Connection to LLVM server
- Request formatting and handling
- Response processing
- Queue and backoff systems
- Circuit breaker pattern
- Request batching
- Performance monitoring
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
import requests
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Union, Tuple, Callable

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlvmClient:
    """
    Client for the LLVM API with full feature parity to other APIs.
    
    Features:
    - Request queue with priority levels
    - Exponential backoff with configurable retries
    - Circuit breaker pattern for fault tolerance
    - Request batching for improved throughput
    - Comprehensive performance monitoring
    - Thread-safe operations
    """
    
    def __init__(self, **kwargs):
        """
        Initialize LLVM API client with configuration options.
        
        Args:
            api_key (str, optional): API key for authentication
            base_url (str, optional): Base URL for the LLVM API server
            max_retries (int, optional): Maximum number of retry attempts
            initial_retry_delay (float, optional): Initial delay between retries
            backoff_factor (float, optional): Factor to increase delay between retries
            max_concurrent_requests (int, optional): Maximum number of concurrent requests
            failure_threshold (int, optional): Number of failures before circuit opens
            reset_timeout (float, optional): Time to wait before closing the circuit
            batch_size (int, optional): Maximum batch size for batch processing
            batch_timeout (float, optional): Maximum time to wait for a batch to fill
        """
        # Authentication
        self.api_key = kwargs.get("api_key", os.environ.get("LLVM_API_KEY", ""))
        
        # Server configuration
        self.base_url = kwargs.get("base_url", os.environ.get("LLVM_BASE_URL", "http://localhost:8090"))
        self.timeout = kwargs.get("timeout", int(os.environ.get("LLVM_TIMEOUT", "30")))
        
        # Retry configuration
        self.max_retries = kwargs.get("max_retries", int(os.environ.get("LLVM_MAX_RETRIES", "3")))
        self.initial_retry_delay = kwargs.get("initial_retry_delay", float(os.environ.get("LLVM_RETRY_DELAY", "1.0")))
        self.backoff_factor = kwargs.get("backoff_factor", float(os.environ.get("LLVM_BACKOFF_FACTOR", "2.0")))
        
        # Queue configuration
        self.queue_enabled = kwargs.get("queue_enabled", True)
        self.request_queue = []  # Format: [(priority, future, func, args, kwargs)]
        self.queue_lock = threading.RLock()
        self.queue_processing = False  # Flag to track if queue is being processed
        self.max_concurrent_requests = kwargs.get("max_concurrent_requests", 
                                                int(os.environ.get("LLVM_MAX_CONCURRENT", "10")))
        self.current_requests = 0
        
        # Circuit breaker configuration
        self.circuit_state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.circuit_lock = threading.RLock()
        self.failure_count = 0
        self.failure_threshold = kwargs.get("failure_threshold", 
                                           int(os.environ.get("LLVM_FAILURE_THRESHOLD", "5")))
        self.last_failure_time = 0
        self.reset_timeout = kwargs.get("reset_timeout", 
                                       float(os.environ.get("LLVM_RESET_TIMEOUT", "30.0")))
        
        # Request batching
        self.batch_enabled = kwargs.get("batch_enabled", True)
        self.batch_lock = threading.RLock()
        self.max_batch_size = kwargs.get("batch_size", 
                                        int(os.environ.get("LLVM_BATCH_SIZE", "8")))
        self.batch_timeout = kwargs.get("batch_timeout", 
                                       float(os.environ.get("LLVM_BATCH_TIMEOUT", "0.1")))
        self.current_batch = {
            "requests": [],
            "created_at": None
        }
        
        # Monitoring and metrics
        self.metrics_lock = threading.RLock()
        self.metrics = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "latency": [],
            "retries": 0,
            "batches": 0,
            "batch_sizes": [],
            "error_types": {},
            "models": {}
        }
        
        # Request tracking
        self.request_tracking = True
        self.recent_requests = {}
        self.recent_requests_lock = threading.RLock()
        
        # Initialize distributed storage
        self._storage = None
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper()
                if self._storage:
                    logger.info("LLVM: Distributed storage initialized")
            except Exception as e:
                logger.debug(f"LLVM: Could not initialize storage: {e}")
        
        # Start the queue processing thread if queue is enabled
        if self.queue_enabled:
            self._start_queue_processing()
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set or update the API key.
        
        Args:
            api_key (str): New API key to use for requests
        """
        self.api_key = api_key
    
    def _start_queue_processing(self) -> None:
        """Start the queue processing thread."""
        if not self.queue_processing:
            self.queue_processing = True
            threading.Thread(target=self._process_queue, daemon=True).start()
    
    def _process_queue(self) -> None:
        """Process queued requests continuously in a separate thread."""
        while self.queue_processing:
            try:
                # Process all queued requests respecting max_concurrent_requests
                request_info = None
                
                with self.queue_lock:
                    if self.request_queue and self.current_requests < self.max_concurrent_requests:
                        # Sort queue by priority (lower number = higher priority)
                        self.request_queue.sort(key=lambda x: x[0])
                        
                        # Get next request from queue
                        priority, future, func, args, kwargs = self.request_queue.pop(0)
                        request_info = (future, func, args, kwargs)
                        self.current_requests += 1
                
                if request_info:
                    future, func, args, kwargs = request_info
                    
                    # Execute the request in a separate thread
                    threading.Thread(
                        target=self._execute_queued_request,
                        args=(future, func, args, kwargs),
                        daemon=True
                    ).start()
                else:
                    # No requests to process, sleep briefly
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
                time.sleep(0.1)  # Sleep briefly on error
    
    def _execute_queued_request(self, future: Future, func: Callable, 
                                args: tuple, kwargs: dict) -> None:
        """
        Execute a queued request and set its result in the future.
        
        Args:
            future (Future): Future to store the result
            func (Callable): Function to call
            args (tuple): Function arguments
            kwargs (dict): Function keyword arguments
        """
        try:
            # Execute the function with circuit breaker pattern
            if self._check_circuit():
                try:
                    result = func(*args, **kwargs)
                    future.set_result(result)
                    self._on_success()
                except Exception as e:
                    self._on_failure()
                    future.set_exception(e)
            else:
                # Circuit is open, fail fast
                future.set_exception(Exception("Circuit breaker is open"))
        except Exception as e:
            # Unhandled exception
            future.set_exception(e)
        finally:
            # Decrement current requests counter
            with self.queue_lock:
                self.current_requests -= 1
    
    def _with_queue(self, func: Callable, priority: int = 1) -> Any:
        """
        Execute a function with request queue.
        
        Args:
            func (Callable): Function to execute
            priority (int): Priority level (0=HIGH, 1=NORMAL, 2=LOW)
            
        Returns:
            Any: Result of the function call
        """
        if not self.queue_enabled:
            # Queue disabled, execute immediately
            return func()
        
        # Create future for async result
        future = Future()
        
        # Add request to queue
        with self.queue_lock:
            self.request_queue.append((
                priority,  # Priority level
                future,    # Future for result
                func,      # Function to call
                (),        # Args (empty as we're wrapping a lambda)
                {}         # Kwargs (empty as we're wrapping a lambda)
            ))
        
        # Return future result (blocks until completed)
        return future.result()
    
    def _check_circuit(self) -> bool:
        """
        Check the circuit state before making a request.
        
        Returns:
            bool: True if request should proceed, False if circuit is open
        """
        with self.circuit_lock:
            current_time = time.time()
            
            # If OPEN, check if we should try HALF-OPEN
            if self.circuit_state == "OPEN":
                if current_time - self.last_failure_time > self.reset_timeout:
                    self.circuit_state = "HALF-OPEN"
                    return True
                return False
                
            # If HALF-OPEN or CLOSED, allow the request
            return True
    
    def _on_success(self) -> None:
        """Handle successful request for circuit breaker."""
        with self.circuit_lock:
            if self.circuit_state == "HALF-OPEN":
                # Reset on successful request in HALF-OPEN state
                self.circuit_state = "CLOSED"
                self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed request for circuit breaker."""
        with self.circuit_lock:
            self.last_failure_time = time.time()
            
            if self.circuit_state == "HALF-OPEN":
                # Return to OPEN on failure in HALF-OPEN
                self.circuit_state = "OPEN"
            elif self.circuit_state == "CLOSED":
                # Increment failure count
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.circuit_state = "OPEN"
    
    def _with_backoff(self, func: Callable) -> Any:
        """
        Execute a function with exponential backoff for retries.
        
        Args:
            func (Callable): Function to execute with retries
            
        Returns:
            Any: Result of the function call
            
        Raises:
            Exception: Last exception if all retries fail
        """
        last_exception = None
        retry_delay = self.initial_retry_delay
        
        # Try the function with backoff
        for attempt in range(self.max_retries + 1):  # +1 for the initial attempt
            try:
                # Execute the function
                return func()
            except Exception as e:
                last_exception = e
                
                # Track retry in metrics
                with self.metrics_lock:
                    self.metrics["retries"] += 1
                
                # Last attempt, reraise the exception
                if attempt >= self.max_retries:
                    raise
                
                # Wait with exponential backoff before retrying
                time.sleep(retry_delay)
                retry_delay *= self.backoff_factor
        
        # Should never reach here, but just in case
        raise last_exception if last_exception else Exception("Unknown error in backoff")
    
    def _update_metrics(self, success: bool = True, latency: float = None, 
                       error: Exception = None, retried: bool = False, 
                       model: str = None, batch_size: int = None) -> None:
        """
        Update metrics after a request completes.
        
        Args:
            success (bool): Whether the request succeeded
            latency (float, optional): Request latency in seconds
            error (Exception, optional): Exception if request failed
            retried (bool): Whether the request was retried
            model (str, optional): Model used for the request
            batch_size (int, optional): Size of the batch for batch requests
        """
        with self.metrics_lock:
            # Basic counters
            self.metrics["requests"] += 1
            if success:
                self.metrics["successes"] += 1
            else:
                self.metrics["failures"] += 1
                
            # Latency tracking
            if latency is not None:
                self.metrics["latency"].append(latency)
                
            # Batch tracking
            if batch_size is not None:
                self.metrics["batches"] += 1
                self.metrics["batch_sizes"].append(batch_size)
                
            # Error tracking
            if error is not None:
                error_type = type(error).__name__
                if error_type not in self.metrics["error_types"]:
                    self.metrics["error_types"][error_type] = 0
                self.metrics["error_types"][error_type] += 1
                
            # Per-model tracking
            if model:
                if model not in self.metrics["models"]:
                    self.metrics["models"][model] = {
                        "requests": 0,
                        "successes": 0,
                        "failures": 0,
                        "latency": []
                    }
                self.metrics["models"][model]["requests"] += 1
                if success:
                    self.metrics["models"][model]["successes"] += 1
                else:
                    self.metrics["models"][model]["failures"] += 1
                if latency is not None:
                    self.metrics["models"][model]["latency"].append(latency)
    
    def _track_request(self, request_id: str, data: dict) -> None:
        """
        Track a request for monitoring purposes.
        
        Args:
            request_id (str): Unique ID for the request
            data (dict): Request data to track
        """
        if not self.request_tracking:
            return
            
        with self.recent_requests_lock:
            # Add request to tracking with timestamp
            self.recent_requests[request_id] = {
                "timestamp": time.time(),
                "data": data
            }
            
            # Clean up old requests (keep only last 100)
            if len(self.recent_requests) > 100:
                # Sort by timestamp and remove oldest
                sorted_keys = sorted(self.recent_requests.keys(), 
                                   key=lambda k: self.recent_requests[k]["timestamp"])
                # Remove oldest requests
                for key in sorted_keys[:-100]:
                    self.recent_requests.pop(key, None)
    
    def _add_to_batch(self, request_input: Any, model_id: str, future: Future, 
                     parameters: dict = None) -> dict:
        """
        Add a request to the current batch or create a new one.
        
        Args:
            request_input (Any): Input for the request
            model_id (str): ID of the model to use
            future (Future): Future for the request result
            parameters (dict, optional): Additional parameters for the request
            
        Returns:
            dict: Batch to process if ready, None otherwise
        """
        if not self.batch_enabled:
            return None
            
        with self.batch_lock:
            # If batch is empty, create a new one
            if not self.current_batch["requests"]:
                self.current_batch = {
                    "requests": [],
                    "created_at": time.time(),
                    "model_id": model_id
                }
                
            # Check if this request can be added to the current batch
            if self.current_batch["model_id"] != model_id:
                # Different model, can't batch
                return None
                
            # Add request to batch
            self.current_batch["requests"].append({
                "input": request_input,
                "parameters": parameters,
                "future": future
            })
            
            # Check if we should process the batch
            should_process = (
                len(self.current_batch["requests"]) >= self.max_batch_size or
                (time.time() - self.current_batch["created_at"] >= self.batch_timeout and
                 len(self.current_batch["requests"]) > 0)
            )
            
            if should_process:
                batch_to_process = self.current_batch
                self.current_batch = {
                    "requests": [],
                    "created_at": None,
                    "model_id": None
                }
                return batch_to_process
                
            return None
    
    def _process_batch(self, batch: dict) -> None:
        """
        Process a batch of requests.
        
        Args:
            batch (dict): Batch of requests to process
        """
        if not batch or not batch["requests"]:
            return
            
        # Extract batch information
        model_id = batch["model_id"]
        requests = batch["requests"]
        inputs = [req["input"] for req in requests]
        
        # Combine parameters, using the first request's parameters
        parameters = requests[0].get("parameters", {})
        
        try:
            # Make batch request
            start_time = time.time()
            batch_result = self._make_batch_request(model_id, inputs, parameters)
            latency = time.time() - start_time
            
            # Update metrics
            self._update_metrics(
                success=True,
                latency=latency,
                model=model_id,
                batch_size=len(requests)
            )
            
            # Set results for individual futures
            for i, req in enumerate(requests):
                if i < len(batch_result):
                    req["future"].set_result(batch_result[i])
                else:
                    # Handle case where less results than requests
                    req["future"].set_exception(
                        Exception(f"Batch result missing for request {i}")
                    )
        except Exception as e:
            # Handle batch failure
            self._update_metrics(
                success=False,
                error=e,
                model=model_id,
                batch_size=len(requests)
            )
            
            # Set exception for all futures
            for req in requests:
                req["future"].set_exception(e)
    
    def _make_batch_request(self, model_id: str, inputs: List[Any], 
                           parameters: dict = None) -> List[Any]:
        """
        Make a batch request to the LLVM API.
        
        Args:
            model_id (str): ID of the model to use
            inputs (List[Any]): List of inputs for the batch
            parameters (dict, optional): Additional parameters for the request
            
        Returns:
            List[Any]: List of results for each input
        """
        # Endpoint for batch requests
        endpoint = f"{self.base_url}/models/{model_id}/batch_infer"
        
        # Create batch request payload
        payload = {
            "inputs": inputs
        }
        
        # Add parameters if provided
        if parameters:
            payload["parameters"] = parameters
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Make the request
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=self._get_headers(request_id),
                timeout=self.timeout
            )
            
            # Track request
            self._track_request(request_id, {
                "endpoint": endpoint,
                "payload": payload,
                "response_code": response.status_code
            })
            
            # Check for success
            if response.status_code == 200:
                result = response.json()
                return result.get("results", [])
            else:
                # Handle error response
                error_message = f"API error: {response.status_code}"
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_message = f"API error: {error_json['error']}"
                except Exception:
                    pass
                    
                raise Exception(error_message)
        except requests.RequestException as e:
            # Handle network or timeout errors
            raise Exception(f"Request failed: {e}")
    
    def _get_headers(self, request_id: str = None) -> dict:
        """
        Get headers for API requests.
        
        Args:
            request_id (str, optional): Unique ID for the request
            
        Returns:
            dict: Headers for the request
        """
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add authorization if API key is available
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Add request ID if provided
        if request_id:
            headers["X-Request-ID"] = request_id
        
        return headers
    
    def _make_request(self, method: str, endpoint: str, json_data: dict = None, 
                     params: dict = None, request_id: str = None) -> dict:
        """
        Make an HTTP request to the LLVM API.
        
        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint
            json_data (dict, optional): JSON data for the request
            params (dict, optional): Query parameters for the request
            request_id (str, optional): Unique ID for the request
            
        Returns:
            dict: JSON response from the API
        """
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Track the start time for latency measurement
        start_time = time.time()
        
        # Make the request
        try:
            response = requests.request(
                method=method,
                url=endpoint,
                json=json_data,
                params=params,
                headers=self._get_headers(request_id),
                timeout=self.timeout
            )
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Track request
            self._track_request(request_id, {
                "endpoint": endpoint,
                "method": method,
                "json_data": json_data,
                "params": params,
                "response_code": response.status_code,
                "latency": latency
            })
            
            # Check for success
            if response.status_code < 400:
                # Update metrics for successful request
                self._update_metrics(
                    success=True,
                    latency=latency
                )
                return response.json()
            else:
                # Handle error response
                error_message = f"API error: {response.status_code}"
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_message = f"API error: {error_json['error']}"
                except Exception:
                    # If response can't be parsed as JSON, use text if available
                    if response.text:
                        error_message = f"API error: {response.status_code} - {response.text}"
                
                error = Exception(error_message)
                
                # Update metrics for failed request
                self._update_metrics(
                    success=False,
                    latency=latency,
                    error=error
                )
                
                raise error
        except requests.RequestException as e:
            # Calculate latency for failed request
            latency = time.time() - start_time
            
            # Update metrics for failed request
            self._update_metrics(
                success=False,
                latency=latency,
                error=e
            )
            
            raise Exception(f"Request failed: {e}")
    
    # API Methods
    def list_models(self) -> dict:
        """
        List available models on the LLVM server.
        
        Returns:
            dict: List of available models
        """
        endpoint = f"{self.base_url}/models"
        
        # Use queue and backoff
        return self._with_queue(
            lambda: self._with_backoff(
                lambda: self._make_request("GET", endpoint)
            )
        )
    
    def get_model_info(self, model_id: str) -> dict:
        """
        Get information about a specific model.
        
        Args:
            model_id (str): ID of the model
            
        Returns:
            dict: Model information
        """
        endpoint = f"{self.base_url}/models/{model_id}"
        
        # Use queue and backoff
        return self._with_queue(
            lambda: self._with_backoff(
                lambda: self._make_request("GET", endpoint)
            )
        )
    
    def run_inference(self, model_id: str, inputs: Any, 
                     parameters: dict = None, priority: int = 1) -> dict:
        """
        Run inference with a specific model.
        
        Args:
            model_id (str): ID of the model
            inputs (Any): Input data for inference
            parameters (dict, optional): Additional parameters for inference
            priority (int): Priority level (0=HIGH, 1=NORMAL, 2=LOW)
            
        Returns:
            dict: Inference result
        """
        endpoint = f"{self.base_url}/models/{model_id}/infer"
        
        # Create payload
        payload = {
            "input": inputs
        }
        
        # Add parameters if provided
        if parameters:
            payload["parameters"] = parameters
        
        # Check if batching is enabled and can be used
        if self.batch_enabled:
            # Create future for async result
            future = Future()
            
            # Try to add to batch
            batch = self._add_to_batch(inputs, model_id, future, parameters)
            
            # If batch is ready to process, process it
            if batch:
                self._process_batch(batch)
                
            # Return future result (blocks until completed)
            return future.result()
        
        # Use queue and backoff
        return self._with_queue(
            lambda: self._with_backoff(
                lambda: self._make_request("POST", endpoint, json_data=payload)
            ),
            priority=priority
        )
    
    def process_batch(self, endpoint_url: str, inputs: List[Any], 
                     model_id: str, parameters: dict = None) -> List[Any]:
        """
        Process a batch of inputs in a single request.
        
        Args:
            endpoint_url (str): URL for the API endpoint
            inputs (List[Any]): List of inputs to process
            model_id (str): ID of the model to use
            parameters (dict, optional): Additional parameters for the request
            
        Returns:
            List[Any]: List of results for each input
        """
        # Create batch request payload
        payload = {
            "inputs": inputs
        }
        
        # Add parameters if provided
        if parameters:
            payload["parameters"] = parameters
        
        # Use queue and backoff
        return self._with_queue(
            lambda: self._with_backoff(
                lambda: self._make_request("POST", endpoint_url, json_data=payload)
            )
        ).get("results", [])
    
    def process_batch_with_metrics(self, endpoint_url: str, inputs: List[Any], 
                                  model_id: str, parameters: dict = None) -> Tuple[List[Any], dict]:
        """
        Process a batch of inputs and return metrics.
        
        Args:
            endpoint_url (str): URL for the API endpoint
            inputs (List[Any]): List of inputs to process
            model_id (str): ID of the model to use
            parameters (dict, optional): Additional parameters for the request
            
        Returns:
            Tuple[List[Any], dict]: Tuple of results and metrics
        """
        start_time = time.time()
        
        # Use batch processing
        results = self.process_batch(endpoint_url, inputs, model_id, parameters)
        
        # Calculate metrics
        latency = time.time() - start_time
        metrics = {
            "batch_size": len(inputs),
            "total_time": latency,
            "average_time_per_item": latency / len(inputs) if inputs else 0,
            "items_per_second": len(inputs) / latency if latency > 0 else 0
        }
        
        return results, metrics
    
    def get_model_statistics(self, endpoint_url: str, model_id: str) -> dict:
        """
        Get performance statistics for a model.
        
        Args:
            endpoint_url (str): URL for the API endpoint
            model_id (str): ID of the model
            
        Returns:
            dict: Performance statistics
        """
        # Use queue and backoff
        return self._with_queue(
            lambda: self._with_backoff(
                lambda: self._make_request("GET", f"{endpoint_url}/models/{model_id}/statistics")
            )
        )
    
    def optimize_model(self, endpoint_url: str, model_id: str, 
                      optimization_type: str = "balanced") -> dict:
        """
        Optimize a model for specific performance characteristics.
        
        Args:
            endpoint_url (str): URL for the API endpoint
            model_id (str): ID of the model to optimize
            optimization_type (str): Type of optimization (speed, memory, balanced)
            
        Returns:
            dict: Optimization result
        """
        # Valid optimization types
        valid_types = ["speed", "memory", "balanced"]
        if optimization_type not in valid_types:
            raise ValueError(f"Invalid optimization type. Must be one of: {valid_types}")
        
        # Create payload
        payload = {
            "optimization_type": optimization_type
        }
        
        # Use queue and backoff
        return self._with_queue(
            lambda: self._with_backoff(
                lambda: self._make_request(
                    "POST", 
                    f"{endpoint_url}/models/{model_id}/optimize",
                    json_data=payload
                )
            )
        )
    
    def format_request_with_params(self, handler: Callable, input_data: Any, 
                                 parameters: dict) -> Any:
        """
        Format a request with specific parameters.
        
        Args:
            handler (Callable): Function to handle the request
            input_data (Any): Input data for the request
            parameters (dict): Parameters for the request
            
        Returns:
            Any: Result of the request
        """
        # Create payload with parameters
        payload = {
            "input": input_data,
            "parameters": parameters
        }
        
        # Call handler with the formatted payload
        return handler(payload)
    
    def format_structured_request(self, handler: Callable, structured_input: dict) -> Any:
        """
        Format a structured request.
        
        Args:
            handler (Callable): Function to handle the request
            structured_input (dict): Structured input for the request
            
        Returns:
            Any: Result of the request
        """
        # Call handler with the structured input
        return handler(structured_input)
    
    def create_llvm_endpoint_handler(self, endpoint_url: str, model: str = None) -> Callable:
        """
        Create a callable endpoint handler for the specified endpoint.
        
        Args:
            endpoint_url (str): URL for the API endpoint
            model (str, optional): Default model to use
            
        Returns:
            Callable: Endpoint handler function
        """
        # Return a function that handles requests to this endpoint
        def endpoint_handler(data: Any, **kwargs):
            # Extract model from kwargs or use default
            model_id = kwargs.get("model", model)
            if not model_id:
                raise ValueError("Model must be specified")
            
            # Extract other parameters
            parameters = kwargs.get("parameters", {})
            priority = kwargs.get("priority", 1)
            
            # Make the actual request
            return self.run_inference(model_id, data, parameters, priority)
        
        return endpoint_handler
    
    def create_llvm_endpoint_handler_with_params(self, endpoint_url: str, model: str, 
                                              parameters: dict) -> Callable:
        """
        Create a handler with specific parameters.
        
        Args:
            endpoint_url (str): URL for the API endpoint
            model (str): Model to use
            parameters (dict): Parameters to use with this handler
            
        Returns:
            Callable: Endpoint handler function
        """
        # Return a function that includes these parameters
        def handler_with_params(data: Any, **kwargs):
            # Merge provided parameters with default parameters
            merged_params = {**parameters, **kwargs.get("parameters", {})}
            
            # Make the actual request
            return self.run_inference(model, data, merged_params)
        
        return handler_with_params
    
    # Advanced multiplexing capabilities
    def create_endpoint(self, api_key: str = None, max_concurrent_requests: int = None,
                      queue_size: int = None, max_retries: int = None) -> str:
        """
        Create a new endpoint configuration with specific settings.
        
        Args:
            api_key (str, optional): API key for this endpoint
            max_concurrent_requests (int, optional): Maximum concurrent requests
            queue_size (int, optional): Maximum queue size
            max_retries (int, optional): Maximum retry attempts
            
        Returns:
            str: Endpoint ID for reference
        """
        # Generate endpoint ID
        endpoint_id = str(uuid.uuid4())
        
        # Store endpoint configuration
        with self.metrics_lock:  # Reuse existing lock for simplicity
            if not hasattr(self, "endpoints"):
                self.endpoints = {}
                
            self.endpoints[endpoint_id] = {
                "api_key": api_key or self.api_key,
                "max_concurrent_requests": max_concurrent_requests or self.max_concurrent_requests,
                "queue_size": queue_size,
                "max_retries": max_retries or self.max_retries,
                "created_at": time.time(),
                "stats": {
                    "requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "latency": []
                }
            }
        
        return endpoint_id
    
    def make_request_with_endpoint(self, endpoint_id: str, data: Any, 
                                 model: str, **kwargs) -> Any:
        """
        Make a request using a specific endpoint configuration.
        
        Args:
            endpoint_id (str): ID of the endpoint to use
            data (Any): Data for the request
            model (str): Model to use
            **kwargs: Additional parameters for the request
            
        Returns:
            Any: Result of the request
        """
        # Get endpoint configuration
        with self.metrics_lock:
            if not hasattr(self, "endpoints") or endpoint_id not in self.endpoints:
                raise ValueError(f"Endpoint {endpoint_id} not found")
                
            endpoint = self.endpoints[endpoint_id]
        
        # Store original values
        original_api_key = self.api_key
        original_max_concurrent = self.max_concurrent_requests
        original_max_retries = self.max_retries
        
        try:
            # Apply endpoint configuration
            self.api_key = endpoint["api_key"]
            self.max_concurrent_requests = endpoint["max_concurrent_requests"]
            self.max_retries = endpoint["max_retries"]
            
            # Make the request
            start_time = time.time()
            result = self.run_inference(model, data, kwargs.get("parameters"), 
                                       kwargs.get("priority", 1))
            latency = time.time() - start_time
            
            # Update endpoint statistics
            with self.metrics_lock:
                endpoint["stats"]["requests"] += 1
                endpoint["stats"]["successes"] += 1
                endpoint["stats"]["latency"].append(latency)
            
            return result
        except Exception as e:
            # Update endpoint statistics for failure
            with self.metrics_lock:
                endpoint["stats"]["requests"] += 1
                endpoint["stats"]["failures"] += 1
            
            # Re-raise the exception
            raise
        finally:
            # Restore original values
            self.api_key = original_api_key
            self.max_concurrent_requests = original_max_concurrent
            self.max_retries = original_max_retries
    
    def get_stats(self, endpoint_id: str = None) -> dict:
        """
        Get statistics for a specific endpoint or all endpoints.
        
        Args:
            endpoint_id (str, optional): ID of the endpoint to get stats for
            
        Returns:
            dict: Statistics for the endpoint or all endpoints
        """
        with self.metrics_lock:
            if not hasattr(self, "endpoints"):
                return {"error": "No endpoints available"}
                
            if endpoint_id:
                # Get stats for specific endpoint
                if endpoint_id not in self.endpoints:
                    return {"error": f"Endpoint {endpoint_id} not found"}
                    
                return {
                    "endpoint_id": endpoint_id,
                    "stats": self.endpoints[endpoint_id]["stats"]
                }
            else:
                # Get stats for all endpoints
                all_stats = {
                    "endpoints_count": len(self.endpoints),
                    "endpoints": {}
                }
                
                for ep_id, endpoint in self.endpoints.items():
                    all_stats["endpoints"][ep_id] = endpoint["stats"]
                
                return all_stats
    
    def get_metrics(self, reset: bool = False) -> dict:
        """
        Get current metrics for monitoring.
        
        Args:
            reset (bool): Whether to reset metrics after retrieving them
            
        Returns:
            dict: Current metrics
        """
        with self.metrics_lock:
            # Create a copy of current metrics
            metrics_copy = {
                "requests": self.metrics["requests"],
                "successes": self.metrics["successes"],
                "failures": self.metrics["failures"],
                "retries": self.metrics["retries"],
                "batches": self.metrics["batches"],
                "error_types": dict(self.metrics["error_types"]),
                "models": {}
            }
            
            # Calculate latency statistics if available
            if self.metrics["latency"]:
                latency_values = sorted(self.metrics["latency"])
                metrics_copy["latency"] = {
                    "min": min(latency_values),
                    "max": max(latency_values),
                    "avg": sum(latency_values) / len(latency_values),
                    "p50": latency_values[len(latency_values) // 2],
                    "p90": latency_values[int(len(latency_values) * 0.9)],
                    "p95": latency_values[int(len(latency_values) * 0.95)],
                    "p99": latency_values[int(len(latency_values) * 0.99)] if len(latency_values) >= 100 else None
                }
            else:
                metrics_copy["latency"] = {
                    "min": None,
                    "max": None,
                    "avg": None,
                    "p50": None,
                    "p90": None,
                    "p95": None,
                    "p99": None
                }
            
            # Calculate batch statistics if available
            if self.metrics["batch_sizes"]:
                batch_sizes = self.metrics["batch_sizes"]
                metrics_copy["batch_sizes"] = {
                    "min": min(batch_sizes),
                    "max": max(batch_sizes),
                    "avg": sum(batch_sizes) / len(batch_sizes)
                }
            else:
                metrics_copy["batch_sizes"] = {
                    "min": None,
                    "max": None,
                    "avg": None
                }
            
            # Copy model-specific metrics
            for model, model_metrics in self.metrics["models"].items():
                metrics_copy["models"][model] = {
                    "requests": model_metrics["requests"],
                    "successes": model_metrics["successes"],
                    "failures": model_metrics["failures"]
                }
                
                # Calculate model-specific latency statistics if available
                if model_metrics["latency"]:
                    latency_values = sorted(model_metrics["latency"])
                    metrics_copy["models"][model]["latency"] = {
                        "min": min(latency_values),
                        "max": max(latency_values),
                        "avg": sum(latency_values) / len(latency_values),
                        "p50": latency_values[len(latency_values) // 2],
                        "p90": latency_values[int(len(latency_values) * 0.9)],
                        "p95": latency_values[int(len(latency_values) * 0.95)]
                    }
                
            # Reset metrics if requested
            if reset:
                self.metrics = {
                    "requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "latency": [],
                    "retries": 0,
                    "batches": 0,
                    "batch_sizes": [],
                    "error_types": {},
                    "models": {}
                }
            
            return metrics_copy

class llvm:
    """
    LLVM API backend implementation with complete feature parity including
    queue, circuit breaker, batching, and monitoring systems.
    
    This class serves as a wrapper around the LlvmClient for compatibility with
    the existing API structure.
    """
    
    def __init__(self, resources=None, metadata=None):
        """
        Initialize LLVM API backend.
        
        Args:
            resources (dict, optional): Resources for the backend
            metadata (dict, optional): Configuration metadata
        """
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Initialize client with metadata
        api_key = self.metadata.get("llvm_api_key", os.environ.get("LLVM_API_KEY", ""))
        base_url = self.metadata.get("llvm_api_url", os.environ.get("LLVM_API_URL", "http://localhost:8090"))
        
        self.client = LlvmClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=int(self.metadata.get("llvm_max_retries", "3")),
            max_concurrent_requests=int(self.metadata.get("llvm_max_concurrent", "10")),
            failure_threshold=int(self.metadata.get("llvm_failure_threshold", "5")),
            reset_timeout=float(self.metadata.get("llvm_reset_timeout", "30.0"))
        )
        
        # Set up queue system variables for compatibility
        self.queue_enabled = True
        self.request_queue = []
        self.queue_lock = threading.RLock()
        self.queue_processing = True
        self.current_requests = 0
        self.max_concurrent_requests = int(self.metadata.get("llvm_max_concurrent", "10"))
        
        # Batch settings
        self.batch_enabled = bool(int(self.metadata.get("llvm_batch_enabled", "1")))
        self.max_batch_size = int(self.metadata.get("llvm_max_batch_size", "8"))
        self.batch_timeout = float(self.metadata.get("llvm_batch_timeout", "0.1"))
        
        # Model lists for batching
        self.embedding_models = []
        self.completion_models = []
        self.supported_batch_models = []
        
        # Circuit breaker configuration for compatibility
        self.circuit_state = "CLOSED"
        self.circuit_lock = threading.RLock()
        self.failure_count = 0
        self.failure_threshold = int(self.metadata.get("llvm_failure_threshold", "5"))
        self.last_failure_time = 0
        self.reset_timeout = float(self.metadata.get("llvm_reset_timeout", "30.0"))
        
        # Priority levels
        self.PRIORITY_HIGH = 0
        self.PRIORITY_NORMAL = 1
        self.PRIORITY_LOW = 2
        
        # Request tracking
        self.request_tracking = True
        self.recent_requests = {}
        
        # Initialize model lists
        self._initialize_model_lists()
        
        return None
    
    def _initialize_model_lists(self):
        """Initialize lists of models that support batching."""
        try:
            # Get model list from file
            model_list_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'model_list',
                'llvm.json'
            )
            
            # If file doesn't exist, create parent directory
            model_list_dir = os.path.dirname(model_list_path)
            if not os.path.exists(model_list_dir):
                os.makedirs(model_list_dir, exist_ok=True)
            
            # Load model list if it exists
            if os.path.exists(model_list_path):
                # Try distributed storage first
                models = None
                if self._storage and hasattr(self._storage, 'is_distributed') and self._storage.is_distributed:
                    try:
                        cache_key = "llvm_model_list"
                        data = self._storage.read_file(cache_key)
                        if data:
                            models = json.loads(data)
                            logger.debug("Loaded LLVM model list from distributed storage")
                    except Exception as e:
                        logger.debug(f"Failed to read from distributed storage: {e}")
                
                # Fall back to local filesystem
                if models is None:
                    with open(model_list_path, 'r') as f:
                        models = json.load(f)
                
                # Process models for batching support
                for model in models:
                    if model.get("batch_support", False):
                        self.supported_batch_models.append(model["name"])
                        
                        # Categorize by type
                        if model.get("type") == "nlp":
                            self.completion_models.append(model["name"])
                        elif model.get("type") in ["vision", "audio"]:
                            self.embedding_models.append(model["name"])
            else:
                # Default models if file doesn't exist
                self.supported_batch_models = ["resnet50", "bert-base", "mobilenet", "t5-small", "yolov5"]
                self.embedding_models = ["resnet50", "mobilenet", "yolov5"]
                self.completion_models = ["bert-base", "t5-small"]
                
                # Create default model list file
                default_models = [
                    {
                        "name": "resnet50",
                        "type": "vision",
                        "description": "ResNet-50 image classification model",
                        "batch_support": True,
                        "precision_modes": ["fp32", "fp16", "int8"],
                        "optimization_profiles": ["speed", "memory", "balanced"]
                    },
                    {
                        "name": "bert-base",
                        "type": "nlp",
                        "description": "BERT base language model",
                        "batch_support": True,
                        "precision_modes": ["fp32", "fp16", "int8"],
                        "optimization_profiles": ["speed", "memory", "balanced"]
                    }
                ]
                
                # Try distributed storage first
                if self._storage and hasattr(self._storage, 'is_distributed') and self._storage.is_distributed:
                    try:
                        cache_key = "llvm_model_list"
                        data = json.dumps(default_models, indent=2)
                        self._storage.write_file(data, cache_key, pin=True)
                        logger.debug("Saved LLVM model list to distributed storage")
                    except Exception as e:
                        logger.debug(f"Failed to write to distributed storage: {e}")
                
                # Always maintain local filesystem
                with open(model_list_path, 'w') as f:
                    json.dump(default_models, f, indent=2)
        except Exception as e:
            logger.error(f"Error initializing model lists: {e}")
            # Fallback to default models
            self.supported_batch_models = ["resnet50", "bert-base"]
            self.embedding_models = ["resnet50"]
            self.completion_models = ["bert-base"]
    
    def create_llvm_endpoint_handler(self, endpoint_url=None, model=None):
        """
        Create a callable endpoint handler for LLVM.
        
        Args:
            endpoint_url (str, optional): URL for the API endpoint
            model (str, optional): Default model to use
            
        Returns:
            Callable: Endpoint handler function
        """
        # Use client to create handler
        return self.client.create_llvm_endpoint_handler(endpoint_url, model)
    
    def create_llvm_endpoint_handler_with_params(self, endpoint_url, model, parameters):
        """
        Create a handler with specific parameters.
        
        Args:
            endpoint_url (str): URL for the API endpoint
            model (str): Model to use
            parameters (dict): Parameters to use with this handler
            
        Returns:
            Callable: Endpoint handler function
        """
        return self.client.create_llvm_endpoint_handler_with_params(endpoint_url, model, parameters)
    
    def make_post_request_llvm(self, endpoint_url, data, request_id=None):
        """
        Make a request to LLVM API with queue and backoff.
        
        Args:
            endpoint_url (str): URL for the API endpoint
            data (dict): Request data
            request_id (str, optional): Unique ID for the request
            
        Returns:
            dict: Response from the API
        """
        # Extract model ID from URL if possible
        model_id = None
        if "/models/" in endpoint_url:
            url_parts = endpoint_url.split("/models/")
            if len(url_parts) > 1:
                model_parts = url_parts[1].split("/")
                if model_parts:
                    model_id = model_parts[0]
        
        # Create payload based on data format
        if isinstance(data, dict) and "input" in data:
            inputs = data["input"]
            parameters = data.get("parameters", {})
        else:
            inputs = data
            parameters = {}
        
        # Make request using client
        return self.client.run_inference(
            model_id=model_id or "default",
            inputs=inputs,
            parameters=parameters
        )
    
    def check_circuit_breaker(self):
        """
        Check if circuit breaker allows requests to proceed.
        
        Returns:
            bool: True if request should proceed, False if circuit is open
        """
        return self.client._check_circuit()
    
    def track_request_result(self, success, error_type=None):
        """
        Track the result of a request for circuit breaker logic tracking.
        
        Args:
            success (bool): Whether the request succeeded
            error_type (str, optional): Type of error if request failed
        """
        if success:
            self.client._on_success()
        else:
            self.client._on_failure()
    
    def queue_with_priority(self, request_info, priority=None):
        """
        Queue a request with a specific priority level.
        
        Args:
            request_info (dict): Request information
            priority (int, optional): Priority level (0=HIGH, 1=NORMAL, 2=LOW)
            
        Returns:
            dict: Future to track result
        """
        if priority is None:
            priority = self.PRIORITY_NORMAL
        
        # Create future for result tracking
        future = {"result": None, "error": None, "completed": False}
        request_info["future"] = future
        
        # Extract request details
        endpoint_url = request_info.get("endpoint_url")
        data = request_info.get("data")
        api_key = request_info.get("api_key")
        request_id = request_info.get("request_id")
        
        # Create function to execute request
        def execute_request():
            try:
                result = self.make_post_request_llvm(
                    endpoint_url=endpoint_url,
                    data=data,
                    request_id=request_id
                )
                future["result"] = result
                future["completed"] = True
                return result
            except Exception as e:
                future["error"] = e
                future["completed"] = True
                raise
        
        # Queue the request using client
        self.client._with_queue(execute_request, priority=priority)
        
        return future
    
    def add_to_batch(self, model, request_info):
        """
        Add a request to the batch queue for the specified model.
        
        Args:
            model (str): Model ID
            request_info (dict): Request information
            
        Returns:
            bool: True if added to batch, False otherwise
        """
        if not self.batch_enabled or model not in self.supported_batch_models:
            return False
        
        # Extract request details
        data = request_info.get("data", {})
        future = request_info.get("future")
        
        # Get input data
        if isinstance(data, dict) and "input" in data:
            inputs = data["input"]
            parameters = data.get("parameters", {})
        else:
            inputs = data
            parameters = {}
        
        # Create future if not provided
        if not future:
            future = Future()
            request_info["future"] = future
        
        # Add to batch using client
        batch = self.client._add_to_batch(inputs, model, future, parameters)
        
        # Process batch if ready
        if batch:
            self.client._process_batch(batch)
            return True
        
        return True
    
    def execute_code(self, code, options=None):
        """
        Execute code using LLVM JIT compiler.
        
        Args:
            code (str): LLVM IR code to execute
            options (dict, optional): Execution options
            
        Returns:
            dict: Execution result
        """
        # Construct the proper endpoint URL
        endpoint_url = f"{self.client.base_url}/execute"
        
        # Prepare request data
        data = {
            "input": code,
            "parameters": options or {}
        }
        
        # Make request
        response = self.make_post_request_llvm(endpoint_url, data)
        
        # Format response for compatibility
        return {
            "result": response.get("result", response.get("outputs", "")),
            "output": response.get("output", ""),
            "errors": response.get("errors", []),
            "execution_time": response.get("execution_time", 0),
            "implementation_type": "(REAL)"
        }
    
    def optimize_code(self, code, optimization_level=None, options=None):
        """
        Optimize code using LLVM optimizer.
        
        Args:
            code (str): LLVM IR code to optimize
            optimization_level (str, optional): Optimization level (O0, O1, O2, O3)
            options (dict, optional): Optimization options
            
        Returns:
            dict: Optimization result
        """
        # Construct the proper endpoint URL
        endpoint_url = f"{self.client.base_url}/optimize"
        
        # Prepare request data
        data = {
            "input": code,
            "parameters": {
                "optimization_level": optimization_level or "O2",
                **(options or {})
            }
        }
        
        # Make request
        response = self.make_post_request_llvm(endpoint_url, data)
        
        # Format response for compatibility
        return {
            "optimized_code": response.get("optimized_code", ""),
            "optimization_passes": response.get("optimization_passes", []),
            "errors": response.get("errors", []),
            "implementation_type": "(REAL)"
        }
    
    def batch_execute(self, code_batch, options=None):
        """
        Execute multiple code snippets in batch.
        
        Args:
            code_batch (list): List of code snippets to execute
            options (dict, optional): Execution options
            
        Returns:
            dict: Batch execution results
        """
        # Get batch processing endpoint
        endpoint_url = f"{self.client.base_url}/batch_execute"
        
        # Process batch
        results, metrics = self.client.process_batch_with_metrics(
            endpoint_url,
            code_batch,
            "batch_execute",
            options
        )
        
        # Format response for compatibility
        return {
            "results": results,
            "errors": [],
            "execution_times": [metrics.get("average_time_per_item", 0)] * len(code_batch),
            "implementation_type": "(REAL)"
        }
    
    def test_llvm_endpoint(self, endpoint_url=None, model_name=None):
        """
        Test the LLVM endpoint.
        
        Args:
            endpoint_url (str, optional): URL for the API endpoint
            model_name (str, optional): Name of the model to test
            
        Returns:
            bool: True if test succeeds, False otherwise
        """
        if not endpoint_url:
            endpoint_url = f"{self.client.base_url}/execute"
        
        if not model_name:
            model_name = "test-model"
        
        # Simple C code to test execution
        test_code = """
        #include <stdio.h>
        int main() {
            printf("Hello from LLVM\\n");
            return 0;
        }
        """
        
        try:
            response = self.execute_code(test_code)
            return "result" in response and response.get("implementation_type") == "(REAL)"
        except Exception as e:
            logger.error(f"Error testing LLVM endpoint: {e}")
            return False
    
    def is_model_compatible(self, model_name):
        """
        Check if a model is compatible with LLVM.
        
        Args:
            model_name (str): Name of the model to check
            
        Returns:
            bool: True if model is compatible, False otherwise
        """
        # Check if model is in the supported model list
        try:
            # Get model list from file
            model_list_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'model_list',
                'llvm.json'
            )
            
            if os.path.exists(model_list_path):
                # Try distributed storage first
                models = None
                if self._storage and hasattr(self._storage, 'is_distributed') and self._storage.is_distributed:
                    try:
                        cache_key = "llvm_model_list"
                        data = self._storage.read_file(cache_key)
                        if data:
                            models = json.loads(data)
                            logger.debug("Loaded LLVM model list from distributed storage for compatibility check")
                    except Exception as e:
                        logger.debug(f"Failed to read from distributed storage: {e}")
                
                # Fall back to local filesystem
                if models is None:
                    with open(model_list_path, 'r') as f:
                        models = json.load(f)
                
                # Check if model is in list
                for model in models:
                    if model["name"] == model_name:
                        return True
            
            # Default to checking known compatible models if not in file
            return model_name in self.supported_batch_models
        except Exception as e:
            logger.error(f"Error checking model compatibility: {e}")
            # Default to True for known models
            return model_name in ["resnet50", "bert-base", "mobilenet", "t5-small", "yolov5"]
    
    def get_model_info(self, model_name):
        """
        Get information about a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model information
        """
        return self.client.get_model_info(model_name)
    
    def get_server_statistics(self):
        """
        Get statistics about the LLVM server.
        
        Returns:
            dict: Server statistics
        """
        try:
            response = requests.get(
                f"{self.client.base_url}/status",
                headers=self.client._get_headers(),
                timeout=self.client.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Failed to get server statistics: {response.status_code}"
                }
        except Exception as e:
            return {
                "error": f"Exception getting server statistics: {e}"
            }
    
    def process_batch_with_metrics(self, endpoint_url, inputs, model_id, parameters=None):
        """
        Process a batch of inputs and return metrics.
        
        Args:
            endpoint_url (str): URL for the API endpoint
            inputs (list): List of inputs to process
            model_id (str): ID of the model to use
            parameters (dict, optional): Additional parameters for the request
            
        Returns:
            tuple: (results, metrics)
        """
        return self.client.process_batch_with_metrics(endpoint_url, inputs, model_id, parameters)
    
    def reload_model(self, model_name):
        """
        Reload a model on the LLVM server.
        
        Args:
            model_name (str): Name of the model to reload
            
        Returns:
            dict: Result of the reload operation
        """
        try:
            response = requests.post(
                f"{self.client.base_url}/models/{model_name}/reload",
                headers=self.client._get_headers(),
                timeout=self.client.timeout
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": f"Model {model_name} reloaded successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to reload model: {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Exception reloading model: {e}"
            }

# Module-level functions for backward compatibility
def create_llvm_endpoint_handler(endpoint_url, model=None, **kwargs):
    """
    Create a callable endpoint handler for the specified endpoint.
    
    Args:
        endpoint_url (str): URL for the API endpoint
        model (str, optional): Default model to use
        **kwargs: Additional parameters for the client
        
    Returns:
        Callable: Endpoint handler function
    """
    client = LlvmClient(**kwargs)
    return client.create_llvm_endpoint_handler(endpoint_url, model)

def create_llvm_endpoint_handler_with_params(endpoint_url, model, parameters, **kwargs):
    """
    Create a handler with specific parameters.
    
    Args:
        endpoint_url (str): URL for the API endpoint
        model (str): Model to use
        parameters (dict): Parameters to use with this handler
        **kwargs: Additional parameters for the client
        
    Returns:
        Callable: Endpoint handler function
    """
    client = LlvmClient(**kwargs)
    return client.create_llvm_endpoint_handler_with_params(endpoint_url, model, parameters)

def process_batch(endpoint_url, inputs, model_id, **kwargs):
    """
    Process a batch of inputs in a single request.
    
    Args:
        endpoint_url (str): URL for the API endpoint
        inputs (List[Any]): List of inputs to process
        model_id (str): ID of the model to use
        **kwargs: Additional parameters for the request and client
        
    Returns:
        List[Any]: List of results for each input
    """
    # Extract parameters for the request
    parameters = kwargs.pop("parameters", None)
    
    # Create client and process batch
    client = LlvmClient(**kwargs)
    return client.process_batch(endpoint_url, inputs, model_id, parameters)

def process_batch_with_params(endpoint_url, inputs, model_id, parameters, **kwargs):
    """
    Process a batch of inputs with specific parameters.
    
    Args:
        endpoint_url (str): URL for the API endpoint
        inputs (List[Any]): List of inputs to process
        model_id (str): ID of the model to use
        parameters (dict): Parameters for the request
        **kwargs: Additional parameters for the client
        
    Returns:
        List[Any]: List of results for each input
    """
    client = LlvmClient(**kwargs)
    return client.process_batch(endpoint_url, inputs, model_id, parameters)

def process_batch_with_metrics(endpoint_url, inputs, model_id, **kwargs):
    """
    Process a batch of inputs and return metrics.
    
    Args:
        endpoint_url (str): URL for the API endpoint
        inputs (List[Any]): List of inputs to process
        model_id (str): ID of the model to use
        **kwargs: Additional parameters for the request and client
        
    Returns:
        Tuple[List[Any], dict]: Tuple of results and metrics
    """
    # Extract parameters for the request
    parameters = kwargs.pop("parameters", None)
    
    # Create client and process batch with metrics
    client = LlvmClient(**kwargs)
    return client.process_batch_with_metrics(endpoint_url, inputs, model_id, parameters)

def get_model_info(endpoint_url, model_id, **kwargs):
    """
    Get information about a specific model.
    
    Args:
        endpoint_url (str): Base URL for the API
        model_id (str): ID of the model
        **kwargs: Additional parameters for the client
        
    Returns:
        dict: Model information
    """
    # Create client with base URL from endpoint_url
    base_url = endpoint_url.split("/models")[0] if "/models" in endpoint_url else endpoint_url
    client = LlvmClient(base_url=base_url, **kwargs)
    return client.get_model_info(model_id)

def get_model_statistics(endpoint_url, model_id, **kwargs):
    """
    Get performance statistics for a model.
    
    Args:
        endpoint_url (str): Base URL for the API
        model_id (str): ID of the model
        **kwargs: Additional parameters for the client
        
    Returns:
        dict: Performance statistics
    """
    # Create client with base URL from endpoint_url
    base_url = endpoint_url.split("/models")[0] if "/models" in endpoint_url else endpoint_url
    client = LlvmClient(base_url=base_url, **kwargs)
    return client.get_model_statistics(endpoint_url, model_id)

def optimize_model(endpoint_url, model_id, optimization_type="balanced", **kwargs):
    """
    Optimize a model for specific performance characteristics.
    
    Args:
        endpoint_url (str): Base URL for the API
        model_id (str): ID of the model to optimize
        optimization_type (str): Type of optimization (speed, memory, balanced)
        **kwargs: Additional parameters for the client
        
    Returns:
        dict: Optimization result
    """
    # Create client with base URL from endpoint_url
    base_url = endpoint_url.split("/models")[0] if "/models" in endpoint_url else endpoint_url
    client = LlvmClient(base_url=base_url, **kwargs)
    return client.optimize_model(endpoint_url, model_id, optimization_type)

def format_request_with_params(handler, input_data, parameters):
    """
    Format a request with specific parameters.
    
    Args:
        handler (Callable): Function to handle the request
        input_data (Any): Input data for the request
        parameters (dict): Parameters for the request
        
    Returns:
        Any: Result of the request
    """
    # Create payload with parameters
    payload = {
        "input": input_data,
        "parameters": parameters
    }
    
    # Call handler with the formatted payload
    return handler(payload)

def format_structured_request(handler, structured_input):
    """
    Format a structured request.
    
    Args:
        handler (Callable): Function to handle the request
        structured_input (dict): Structured input for the request
        
    Returns:
        Any: Result of the request
    """
    # Call handler with the structured input
    return handler(structured_input)