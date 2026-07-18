import os
import json
import time
import threading
import hashlib
import uuid
import requests
from concurrent.futures import Future
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)

try:
    from .base import BaseAPIBackend
except ImportError:
    try:
        from base import BaseAPIBackend
    except ImportError:
        BaseAPIBackend = object



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

class opea(BaseAPIBackend):
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get OPEA API endpoint from metadata or environment
        self.api_endpoint = self._get_api_endpoint()
        
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
        """Delegate to the shared BaseAPIBackend implementation."""
        return super()._process_queue()

    def make_post_request_opea(self, endpoint_url, data, request_id=None):
        """Make a request to OPEA API with queue and backoff."""
        if not self.check_circuit_breaker():
            raise RuntimeError("Circuit breaker is OPEN. Service unavailable.")

        if request_id is None:
            request_id = str(uuid.uuid4())

        def _execute_request():
            headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
            retries = 0
            while True:
                try:
                    response = requests.post(endpoint_url, json=data, headers=headers, timeout=60)
                    if response.status_code != 200:
                        message = f"OPEA request failed with status {response.status_code}"
                        try:
                            payload = response.json()
                            if isinstance(payload, dict) and payload.get("error"):
                                message = f"{message}: {payload['error']}"
                        except Exception:
                            if response.text:
                                message = f"{message}: {response.text[:200]}"
                        raise ValueError(message)
                    self.track_request_result(True)
                    return response.json()
                except Exception as e:
                    if retries >= self.max_retries:
                        self.track_request_result(False, type(e).__name__)
                        raise
                    retries += 1
                    time.sleep(min(self.initial_retry_delay * (self.backoff_factor ** (retries - 1)), self.max_retry_delay))

        with self.queue_lock:
            at_capacity = self.active_requests >= self.max_concurrent_requests
            if not at_capacity:
                self.active_requests += 1

        if at_capacity and getattr(self, "queue_enabled", True):
            result_future = self.queue_with_priority({"func": _execute_request, "request_id": request_id})
            max_wait = 300
            wait_start = time.time()
            while not result_future["completed"] and (time.time() - wait_start) < max_wait:
                time.sleep(0.05)
            if not result_future["completed"]:
                raise TimeoutError("Request timed out waiting in queue")
            if result_future["error"]:
                raise result_future["error"]
            return result_future["result"]

        try:
            return _execute_request()
        finally:
            with self.queue_lock:
                self.active_requests = max(0, self.active_requests - 1)

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
    