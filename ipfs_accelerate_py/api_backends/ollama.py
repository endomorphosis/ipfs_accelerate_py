#!/usr/bin/env python
"""
IPFS Accelerate Python - Ollama API Backend

This module provides integration with Ollama API for local LLM deployments.
Features:
- Thread-safe request queue with concurrency limits
- Exponential backoff for error handling
- Request tracking with unique IDs
- Streaming support for chat completions
"""

import os
import json
import time
import threading
import requests
import uuid
import hashlib
import logging
from concurrent.futures import Future
from pathlib import Path

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

# Configure logging
logger = logging.getLogger("ollama_api")

try:
    from .base import BaseAPIBackend
except ImportError:
    try:
        from base import BaseAPIBackend
    except ImportError:
        BaseAPIBackend = object


class ollama(BaseAPIBackend):
    def __init__(self, resources=None, metadata=None):
        """
        Initialize the Ollama API client with resources and metadata.
        
        Args:
            resources: Optional resources dictionary
            metadata: Optional metadata dictionary with configuration
        """
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get Ollama API endpoint from metadata or environment
        self.ollama_api_url = self._get_ollama_api_url()
        
        # Initialize counters and metrics
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0
        }
        
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
        
        
        # Initialize distributed storage
        self._storage = None
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper()
                if self._storage:
                    logger.info("Ollama: Distributed storage initialized")
            except Exception as e:
                logger.debug(f"Ollama: Could not initialize storage: {e}")
        
        # Initialize backoff configuration
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 16
        
        self._init_circuit_breaker()
        self.circuit_timeout = self.reset_timeout
        
        # Default model
        self.default_model = os.environ.get("OLLAMA_MODEL", "llama3")
        
        return None

    def _get_ollama_api_url(self):
        """Get Ollama API URL from metadata or environment"""
        # Try to get from metadata
        api_url = self.metadata.get("ollama_api_url")
        if api_url:
            return api_url
        
        # Try to get from environment
        env_url = os.environ.get("OLLAMA_API_URL")
        if env_url:
            return env_url
        
        # Try to load from dotenv
        try:
            from dotenv import load_dotenv
            load_dotenv()
            env_url = os.environ.get("OLLAMA_API_URL")
            if env_url:
                return env_url
        except ImportError:
            pass
        
        # Return default if no URL found
        return "http://localhost:11434/api"
    
    def _process_queue(self):
        """Delegate to the shared BaseAPIBackend implementation."""
        return super()._process_queue()

    def make_post_request_ollama(self, endpoint_url, data, stream=False, request_id=None, priority=None):
        """Make a request to Ollama API with queue and backoff."""
        if not self.check_circuit_breaker():
            raise RuntimeError("Circuit breaker is OPEN. Service unavailable.")

        if request_id is None:
            request_id = str(uuid.uuid4())
        if priority is None:
            priority = self.PRIORITY_NORMAL

        def _execute_request():
            headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
            response = requests.post(endpoint_url, json=data, headers=headers, timeout=60, stream=stream)
            if response.status_code != 200:
                message = f"Ollama request failed with status {response.status_code}"
                try:
                    payload = response.json()
                    if isinstance(payload, dict) and payload.get("error"):
                        message = f"{message}: {payload['error']}"
                except Exception:
                    if response.text:
                        message = f"{message}: {response.text[:200]}"
                self.track_request_result(False, "HTTPError")
                raise ValueError(message)
            self.track_request_result(True)
            if stream:
                def _iter_stream():
                    try:
                        for line in response.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                yield {"message": {"content": line}, "done": False}
                    finally:
                        response.close()
                return _iter_stream()
            return response.json()

        if stream:
            return _execute_request()

        with self.queue_lock:
            at_capacity = self.active_requests >= self.max_concurrent_requests
            if not at_capacity:
                self.active_requests += 1

        if at_capacity and getattr(self, "queue_enabled", True):
            result_future = self.queue_with_priority({"func": _execute_request, "request_id": request_id}, priority)
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

    def chat(self, model_name=None, model=None, messages=None, max_tokens=None, temperature=None, request_id=None, options=None, **kwargs):
        """
        Send a chat request to Ollama API.
        
        Args:
            model_name: Name of the model to use (for compatibility with other APIs)
            model: Name of the model to use (alternative parameter name)
            messages: List of message objects with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (0.0 to 2.0)
            request_id: Optional unique ID for request tracking
            options: Additional options to pass to the API
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict containing generated response text and metadata
        """
        # Use model_name if provided, otherwise use model parameter
        model_to_use = model_name or model or self.default_model
        
        # Construct the proper endpoint URL
        endpoint_url = f"{self.ollama_api_url}/chat"
        
        # Format messages for Ollama API
        formatted_messages = self._format_messages(messages)
        
        # Prepare options dictionary if provided
        options_dict = options.copy() if options else {}
        
        # Add max_tokens and temperature to options if provided
        if max_tokens is not None:
            options_dict["num_predict"] = max_tokens
        
        if temperature is not None:
            options_dict["temperature"] = temperature
        
        # Handle additional kwargs
        for key, value in kwargs.items():
            if key not in ["stream", "messages", "model"]:
                options_dict[key] = value
        
        # Prepare request data
        data = {
            "model": model_to_use,
            "messages": formatted_messages,
            "stream": False
        }
        
        # Add options if any exist
        if options_dict:
            data["options"] = options_dict
        
        # Make request with queue and backoff
        try:
            response = self.make_post_request_ollama(endpoint_url, data, request_id=request_id)
            
            # Update token usage stats
            if "prompt_eval_count" in response and "eval_count" in response:
                with self.queue_lock:
                    self.usage_stats["total_prompt_tokens"] += response.get("prompt_eval_count", 0)
                    self.usage_stats["total_completion_tokens"] += response.get("eval_count", 0)
                    self.usage_stats["total_tokens"] += response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            
            # Process and normalize response
            return {
                "text": response.get("message", {}).get("content", ""),
                "model": model_to_use,
                "usage": self._extract_usage(response),
                "implementation_type": "(REAL)"
            }
        except Exception as e:
            return {
                "text": f"Error: {str(e)}",
                "model": model_to_use,
                "error": str(e),
                "implementation_type": "(ERROR)"
            }

    def generate(self, model=None, prompt=None, max_tokens=None, temperature=None, request_id=None, **kwargs):
        """
        Generate text using the Ollama API (compatibility with other frameworks).
        
        Args:
            model: Name of the model to use
            prompt: Text prompt to send
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (0.0 to 2.0)
            request_id: Optional unique ID for request tracking
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict containing generated response text and metadata
        """
        # Construct a message from the prompt
        messages = [{"role": "user", "content": prompt}]
        
        # Call chat method
        return self.chat(
            model_name=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_id=request_id,
            **kwargs
        )

    def completions(self, model=None, prompt=None, max_tokens=None, temperature=None, request_id=None, **kwargs):
        """
        Generate completions using the Ollama API (compatibility with other frameworks).
        
        Args:
            model: Name of the model to use
            prompt: Text prompt to send
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (0.0 to 2.0)
            request_id: Optional unique ID for request tracking
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict containing generated response text and metadata
        """
        # Construct a message from the prompt
        messages = [{"role": "user", "content": prompt}]
        
        # Call chat method
        return self.chat(
            model_name=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_id=request_id,
            **kwargs
        )

    def stream_chat(self, model_name=None, model=None, messages=None, max_tokens=None, temperature=None, request_id=None, options=None, **kwargs):
        """
        Stream a chat request from Ollama API.
        
        Args:
            model_name: Name of the model to use (for compatibility with other APIs)
            model: Name of the model to use (alternative parameter name)
            messages: List of message objects with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (0.0 to 2.0)
            request_id: Optional unique ID for request tracking
            options: Additional options to pass to the API
            **kwargs: Additional keyword arguments
            
        Returns:
            Generator yielding response chunks
        """
        # Use model_name if provided, otherwise use model parameter
        model_to_use = model_name or model or self.default_model
        
        # Construct the proper endpoint URL
        endpoint_url = f"{self.ollama_api_url}/chat"
        
        # Format messages for Ollama API
        formatted_messages = self._format_messages(messages)
        
        # Prepare options dictionary if provided
        options_dict = options.copy() if options else {}
        
        # Add max_tokens and temperature to options if provided
        if max_tokens is not None:
            options_dict["num_predict"] = max_tokens
        
        if temperature is not None:
            options_dict["temperature"] = temperature
        
        # Handle additional kwargs
        for key, value in kwargs.items():
            if key not in ["stream", "messages", "model"]:
                options_dict[key] = value
        
        # Prepare request data
        data = {
            "model": model_to_use,
            "messages": formatted_messages,
            "stream": True
        }
        
        # Add options if any exist
        if options_dict:
            data["options"] = options_dict
        
        # Make streaming request
        try:
            response_stream = self.make_post_request_ollama(endpoint_url, data, stream=True, request_id=request_id)
            
            # Process streaming response
            completion_tokens = 0
            for chunk in response_stream:
                completion_tokens += 1
                yield {
                    "text": chunk.get("message", {}).get("content", ""),
                    "done": chunk.get("done", False),
                    "model": model_to_use
                }
                
                # If this is the final chunk, update token usage
                if chunk.get("done", False):
                    with self.queue_lock:
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        self.usage_stats["total_prompt_tokens"] += prompt_tokens
                        self.usage_stats["total_completion_tokens"] += completion_tokens
                        self.usage_stats["total_tokens"] += prompt_tokens + completion_tokens
        
        except Exception as e:
            yield {
                "text": f"Error: {str(e)}",
                "error": str(e),
                "done": True,
                "model": model_to_use
            }
            
    def _format_messages(self, messages):
        """Format messages for Ollama API"""
        formatted_messages = []
        
        if not messages:
            return [{"role": "user", "content": "Hello"}]
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Map standard roles to Ollama roles
            if role == "assistant":
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                role = "user"
            
            formatted_messages.append({
                "role": role,
                "content": content
            })
        
        return formatted_messages

    def _extract_usage(self, response):
        """Extract usage information from response"""
        return {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
        }

    def list_models(self):
        """List available models in Ollama"""
        endpoint_url = f"{self.ollama_api_url}/tags"
        
        try:
            response = requests.get(
                endpoint_url,
                timeout=self.metadata.get("timeout", 30)
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
            
    def create_ollama_endpoint_handler(self, endpoint_url=None):
        """Create an endpoint handler for Ollama"""
        if not endpoint_url:
            endpoint_url = self.ollama_api_url
        
        async def endpoint_handler(prompt, **kwargs):
            """Handle requests to Ollama endpoint"""
            # Get model from kwargs or default
            model = kwargs.get("model", self.default_model)
            
            # Create messages from prompt
            messages = [{"role": "user", "content": prompt}]
            
            # Extract options
            options = {}
            for key in ["temperature", "top_p", "top_k", "repeat_penalty"]:
                if key in kwargs:
                    options[key] = kwargs[key]
            
            # Make request
            try:
                response = self.chat(model=model, messages=messages, options=options)
                return response
            except Exception as e:
                print(f"Error calling Ollama endpoint: {e}")
                return {"text": f"Error: {str(e)}", "implementation_type": "(ERROR)"}
        
        return endpoint_handler
    
    def __call__(self, endpoint_type, **kwargs):
        """Make client callable with endpoint type and parameters"""
        if endpoint_type == "chat":
            return self.chat(**kwargs)
        elif endpoint_type == "stream_chat":
            return self.stream_chat(**kwargs)
        elif endpoint_type == "completions":
            return self.completions(**kwargs)
        elif endpoint_type == "generate":
            return self.generate(**kwargs)
        else:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")
    
    def reset_usage_stats(self):
        """Reset usage statistics to zero"""
        with self.queue_lock:
            self.usage_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
        return None
        
    def test_ollama_endpoint(self, endpoint_url=None):
        """Test the Ollama endpoint"""
        if not endpoint_url:
            endpoint_url = f"{self.ollama_api_url}/chat"
            
        model = self.metadata.get("ollama_model", self.default_model)
        messages = [{"role": "user", "content": "Testing the Ollama API. Please respond with a short message."}]
        
        try:
            response = self.chat(model=model, messages=messages)
            return "text" in response and response.get("implementation_type") == "(REAL)"
        except Exception as e:
            print(f"Error testing Ollama endpoint: {e}")
            return False

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
    
# For testing as standalone module
if __name__ == "__main__":
    # Create client
    client = ollama()
    
    # Check attributes
    print(f"Max retries: {client.max_retries}")
    print(f"Backoff factor: {client.backoff_factor}")
    print(f"Max concurrent requests: {client.max_concurrent_requests}")
    
    print("Ollama API implementation has queue and backoff functionality.")