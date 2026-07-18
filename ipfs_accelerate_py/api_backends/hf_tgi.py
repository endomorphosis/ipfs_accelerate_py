import os
import json
import time
import uuid
import threading
import requests
from concurrent.futures import Future
from dotenv import load_dotenv

import hashlib
import queue
from concurrent.futures import ThreadPoolExecutor

import logging
logger = logging.getLogger(__name__)

try:
    from .base import BaseAPIBackend
except ImportError:
    try:
        from base import BaseAPIBackend
    except ImportError:
        BaseAPIBackend = object



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

class hf_tgi(BaseAPIBackend):
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get HF API token from metadata or environment
        self.api_token = self._get_api_token()
        
        # Initialize queue and backoff systems
        self._init_queue(queue_size=100, max_concurrent_requests=5)
        self._init_circuit_breaker()
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

        
        # Initialize backoff configuration
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 16
        
        # Default model
        self.default_model = "mistralai/Mistral-7B-Instruct-v0.2"
        
        self.endpoints = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=10)

        # Initialize distributed storage
        self._storage = None
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper()
                if self._storage:
                    logging.getLogger(__name__).info("HF TGI: Distributed storage initialized")
            except Exception as e:
                logging.getLogger(__name__).debug(f"HF TGI: Could not initialize storage: {e}")

        return

    def _get_api_token(self):
        """Get Hugging Face API token from metadata or environment"""
        # Try to get from metadata
        api_token = self.metadata.get("hf_api_key") or self.metadata.get("hf_api_token")
        if api_token:
            return api_token
        
        # Try to get from environment
        env_token = os.environ.get("HF_API_KEY") or os.environ.get("HF_API_TOKEN")
        if env_token:
            return env_token
        
        # Try to load from dotenv
        try:
            load_dotenv()
            env_token = os.environ.get("HF_API_KEY") or os.environ.get("HF_API_TOKEN")
            if env_token:
                return env_token
        except ImportError:
            pass
        
        # Return None if no token found (will allow unauthenticated requests)
        return None
        
    
    
    
    def _process_queue(self):
        """Delegate to the shared BaseAPIBackend implementation."""
        return super()._process_queue()

    def make_post_request_hf_tgi(self, endpoint_url, data, api_token=None, stream=False, request_id=None, endpoint_id=None):
        """Make a request to HF TGI API with queue and backoff."""
        if not self.check_circuit_breaker():
            raise RuntimeError("Circuit breaker is OPEN. Service unavailable.")

        if endpoint_id and endpoint_id in self.endpoints and api_token is None:
            api_token = self.endpoints[endpoint_id].get("api_key", self.api_token)
        elif api_token is None:
            api_token = self.api_token

        if request_id is None:
            request_id = f"req_{int(time.time())}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"

        def _execute_request():
            headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
            if api_token:
                headers["Authorization"] = f"******"

            retries = 0
            while True:
                try:
                    response = requests.post(endpoint_url, json=data, headers=headers, timeout=60, stream=stream)
                    if response.status_code != 200:
                        message = f"HF TGI request failed with status {response.status_code}"
                        try:
                            payload = response.json()
                            if isinstance(payload, dict) and payload.get("error"):
                                message = f"{message}: {payload['error']}"
                        except Exception:
                            if response.text:
                                message = f"{message}: {response.text[:200]}"
                        raise ValueError(message)
                    self.track_request_result(True)
                    if stream:
                        def _iter_stream():
                            try:
                                for line in response.iter_lines(decode_unicode=True):
                                    if not line:
                                        continue
                                    if line.startswith("data:"):
                                        line = line[5:].strip()
                                    if line == "[DONE]":
                                        break
                                    try:
                                        yield json.loads(line)
                                    except json.JSONDecodeError:
                                        yield {"text": line}
                            finally:
                                response.close()
                        return _iter_stream()
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

    def generate_text(self, model_id, inputs, parameters=None, api_token=None, request_id=None, endpoint_id=None):
        """Generate text using HF TGI API"""
        # Format endpoint URL for the model
        endpoint_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        # Prepare data
        data = {"inputs": inputs}
        if parameters:
            data["parameters"] = parameters
        
        # Make request with queue and backoff
        response = self.make_post_request_hf_tgi(
            endpoint_url=endpoint_url,
            data=data,
            api_token=api_token,
            request_id=request_id,
            endpoint_id=endpoint_id
        )
        
        # Process response based on format
        if isinstance(response, list):
            # Some models return list of generated texts
            result = response[0] if response else {"generated_text": ""}
        elif isinstance(response, dict):
            # Some models return dict with generated_text key
            result = response
        else:
            # Default fallback
            result = {"generated_text": str(response)}
            
        # Add request ID to response if provided
        if request_id and isinstance(result, dict):
            result["request_id"] = request_id
        
        return result

    def run_inference(self, model_id, inputs, parameters=None, priority=None, request_id=None, endpoint_id=None):
        """Canonical backend-manager execution entrypoint.

        Returns a normalized result envelope for text-generation tasks so higher
        layers do not need to understand raw HF TGI response shapes.
        """
        _ = priority
        response = self.generate_text(
            model_id=model_id,
            inputs=inputs,
            parameters=parameters,
            request_id=request_id,
            endpoint_id=endpoint_id,
        )

        if isinstance(response, dict):
            generated_text = response.get("generated_text")
            if generated_text is None and "text" in response:
                generated_text = response.get("text")
        else:
            generated_text = str(response)
            response = {"generated_text": generated_text}

        outputs = [generated_text] if generated_text is not None else []
        result = {
            "model": model_id,
            "task": "text-generation",
            "outputs": outputs,
            "implementation_type": "(REAL)",
            "backend": "hf_tgi",
            "raw_response": response,
        }
        if request_id:
            result["request_id"] = request_id
        if endpoint_id:
            result["endpoint_id"] = endpoint_id
        return result

    def stream_generate(self, model_id, inputs, parameters=None, api_token=None, request_id=None, endpoint_id=None):
        """Stream text generation from HF TGI API"""
        # Format endpoint URL for the model
        endpoint_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        # Prepare data
        data = {
            "inputs": inputs,
            "stream": True
        }
        if parameters:
            data["parameters"] = parameters
        
        # Make streaming request
        response_stream = self.make_post_request_hf_tgi(
            endpoint_url=endpoint_url,
            data=data,
            api_token=api_token,
            stream=True,
            request_id=request_id,
            endpoint_id=endpoint_id
        )
        
        # Yield each chunk with request ID if provided
        for chunk in response_stream:
            # Add request ID to chunk if provided and chunk is a dict
            if request_id and isinstance(chunk, dict):
                chunk["request_id"] = request_id
            yield chunk
            
    def chat(self, model, messages, options=None, request_id=None, endpoint_id=None):
        """Format chat messages and generate text"""
        # Format messages into a prompt
        prompt = self._format_chat_messages(messages)
        
        # Extract parameters from options
        parameters = {}
        if options:
            # Map common parameters
            if "temperature" in options:
                parameters["temperature"] = options["temperature"]
            if "max_tokens" in options:
                parameters["max_new_tokens"] = options["max_tokens"]
            if "top_p" in options:
                parameters["top_p"] = options["top_p"]
            if "top_k" in options:
                parameters["top_k"] = options["top_k"]
        
        # Generate request ID if not provided
        if request_id is None and options and "request_id" in options:
            request_id = options["request_id"]
        
        # Use endpoint ID from options if provided
        if endpoint_id is None and options and "endpoint_id" in options:
            endpoint_id = options["endpoint_id"]
            
        # Generate text
        response = self.generate_text(
            model_id=model, 
            inputs=prompt, 
            parameters=parameters,
            request_id=request_id,
            endpoint_id=endpoint_id
        )
        
        # Extract text from response
        if isinstance(response, dict):
            text = response.get("generated_text", "")
        else:
            text = str(response)
            
        # Return standardized response
        result = {
            "text": text,
            "model": model,
            "usage": self._estimate_usage(prompt, text),
            "implementation_type": "(REAL)"
        }
        
        # Include request ID if provided
        if request_id:
            result["request_id"] = request_id
            
        return result
        
    def stream_chat(self, model, messages, options=None, request_id=None, endpoint_id=None):
        """Stream chat responses"""
        # Format messages into a prompt
        prompt = self._format_chat_messages(messages)
        
        # Extract parameters from options
        parameters = {}
        if options:
            # Map common parameters
            if "temperature" in options:
                parameters["temperature"] = options["temperature"]
            if "max_tokens" in options:
                parameters["max_new_tokens"] = options["max_tokens"]
            if "top_p" in options:
                parameters["top_p"] = options["top_p"]
            if "top_k" in options:
                parameters["top_k"] = options["top_k"]
        
        # Generate request ID if not provided
        if request_id is None and options and "request_id" in options:
            request_id = options["request_id"]
        
        # Use endpoint ID from options if provided
        if endpoint_id is None and options and "endpoint_id" in options:
            endpoint_id = options["endpoint_id"]
            
        # Create stream
        response_stream = self.stream_generate(
            model_id=model, 
            inputs=prompt, 
            parameters=parameters,
            request_id=request_id,
            endpoint_id=endpoint_id
        )
        
        # Process each chunk
        for chunk in response_stream:
            if isinstance(chunk, dict) and "token" in chunk:
                text = chunk.get("token", {}).get("text", "")
                done = chunk.get("generated_text", False) is not False
                
                result = {
                    "text": text,
                    "done": done,
                    "model": model
                }
                
                # Include request ID if provided
                if request_id:
                    result["request_id"] = request_id
                    
                yield result
            else:
                # Handle other response formats
                text = str(chunk)
                result = {
                    "text": text,
                    "done": False,
                    "model": model
                }
                
                # Include request ID if provided
                if request_id:
                    result["request_id"] = request_id
                    
                yield result
                
    def _format_chat_messages(self, messages):
        """Format chat messages into a prompt for HF models"""
        formatted_prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_prompt += f"<|system|>\n{content}\n"
            elif role == "assistant":
                formatted_prompt += f"<|assistant|>\n{content}\n"
            else:  # user or default
                formatted_prompt += f"<|user|>\n{content}\n"
                
        # Add final assistant marker for completion
        formatted_prompt += "<|assistant|>\n"
        
        return formatted_prompt
        
    def _estimate_usage(self, prompt, response):
        """Estimate token usage (rough approximation)"""
        # Very rough approximation: 4 chars ~= 1 token
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response) // 4
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
        
    def create_remote_text_generation_endpoint_handler(self, endpoint_url=None, api_key=None, endpoint_id=None):
        """Create an endpoint handler for HF TGI remote inference"""
        async def endpoint_handler(prompt, **kwargs):
            """Handle requests to HF TGI endpoint"""
            # Extract request ID if provided
            request_id = kwargs.get("request_id")
            
            # Use specific endpoint ID from kwargs or from constructor
            current_endpoint_id = kwargs.get("endpoint_id", endpoint_id)
            
            # If no specific model endpoint provided, use standard API
            if not endpoint_url:
                # Extract model from kwargs or use default
                model = kwargs.get("model", self.default_model)
                
                # Create endpoint URL
                model_endpoint = f"https://api-inference.huggingface.co/models/{model}"
            else:
                model_endpoint = endpoint_url
                model = kwargs.get("model", self.default_model)
                
            # Extract parameters from kwargs
            max_new_tokens = kwargs.get("max_new_tokens", 1024)
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.9)
            do_sample = kwargs.get("do_sample", True)
            
            # Prepare parameters
            parameters = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "return_full_text": kwargs.get("return_full_text", False)
            }
            
            # Check if prompt is a list of messages
            if isinstance(prompt, list):
                # Format as chat messages
                prompt_text = self._format_chat_messages(prompt)
            else:
                prompt_text = prompt
            
            # Use streaming if requested
            if kwargs.get("stream", False):
                async def process_stream():
                    streaming_response = self.stream_generate(
                        model_id=model, 
                        inputs=prompt_text, 
                        parameters=parameters, 
                        api_token=api_key,
                        request_id=request_id,
                        endpoint_id=current_endpoint_id
                    )
                    
                    for chunk in streaming_response:
                        # Create a chunk with proper metadata
                        if "token" in chunk:
                            result_chunk = {
                                "text": chunk["token"]["text"], 
                                "done": False,
                                "model": model,
                                "implementation_type": "(REAL)"
                            }
                        elif "generated_text" in chunk:
                            result_chunk = {
                                "text": chunk["generated_text"], 
                                "done": True,
                                "model": model,
                                "implementation_type": "(REAL)"
                            }
                        else:
                            result_chunk = {
                                "text": str(chunk), 
                                "done": False,
                                "model": model,
                                "implementation_type": "(REAL)"
                            }
                        
                        # Add request ID if provided
                        if request_id:
                            result_chunk["request_id"] = request_id
                            
                        yield result_chunk
                
                return process_stream()
            else:
                try:
                    # Make the request
                    response = self.generate_text(
                        model_id=model, 
                        inputs=prompt_text, 
                        parameters=parameters, 
                        api_token=api_key,
                        request_id=request_id,
                        endpoint_id=current_endpoint_id
                    )
                    
                    # Extract generated text
                    if isinstance(response, list):
                        generated_text = response[0].get("generated_text", "") if response else ""
                    elif isinstance(response, dict):
                        generated_text = response.get("generated_text", "")
                    else:
                        generated_text = str(response)
                    
                    usage = self._estimate_usage(prompt_text, generated_text)
                    
                    # Create response with metadata
                    result = {
                        "text": generated_text, 
                        "model": model,
                        "usage": usage,
                        "implementation_type": "(REAL)"
                    }
                    
                    # Add request ID if provided
                    if request_id:
                        result["request_id"] = request_id
                        
                    return result
                except Exception as e:
                    error_message = str(e)
                    print(f"Error calling HF TGI endpoint: {error_message}")
                    
                    # Create error response with request ID
                    error_response = {
                        "text": f"Error: {error_message}", 
                        "implementation_type": "(ERROR)",
                        "model": model
                    }
                    
                    # Add request ID if provided
                    if request_id:
                        error_response["request_id"] = request_id
                        
                    return error_response
        
        return endpoint_handler
        
    def test_hf_tgi_endpoint(self, endpoint_url=None, api_token=None, model_id=None, endpoint_id=None, request_id=None):
        """Test the HF TGI endpoint"""
        if not model_id:
            model_id = self.default_model
            
        if not endpoint_url:
            endpoint_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        # Use endpoint-specific API token if available
        if endpoint_id and endpoint_id in self.endpoints:
            if api_token is None:
                api_token = self.endpoints[endpoint_id].get("api_key", self.api_token)
        elif api_token is None:
            api_token = self.api_token
        
        # Generate unique request ID for the test if not provided
        if request_id is None:
            request_id = f"test_{int(time.time())}_{hashlib.md5(model_id.encode()).hexdigest()[:8]}"
            
        try:
            # Test text generation with proper request tracking
            response = self.generate_text(
                model_id=model_id, 
                inputs="Testing the Hugging Face TGI API. Please respond with a short message.",
                parameters={"max_new_tokens": 50},
                api_token=api_token,
                request_id=request_id,
                endpoint_id=endpoint_id
            )
            
            # Verify we got a valid response
            if isinstance(response, dict) and ("generated_text" in response or "text" in response):
                result = {
                    "success": True,
                    "message": "TGI API test successful",
                    "model": model_id,
                    "implementation_type": "(REAL)",
                    "request_id": request_id
                }
                return result
            elif isinstance(response, list) and len(response) > 0:
                result = {
                    "success": True,
                    "message": "TGI API test successful (list response)",
                    "model": model_id,
                    "implementation_type": "(REAL)",
                    "request_id": request_id
                }
                return result
            else:
                result = {
                    "success": False,
                    "message": "TGI API test failed: unexpected response format",
                    "implementation_type": "(ERROR)",
                    "request_id": request_id
                }
                return result
        except Exception as e:
            error_message = str(e)
            print(f"Error testing HF TGI endpoint: {error_message}")
            result = {
                "success": False,
                "message": f"TGI API test failed: {error_message}",
                "implementation_type": "(ERROR)",
                "request_id": request_id
            }
            return result

    
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
    