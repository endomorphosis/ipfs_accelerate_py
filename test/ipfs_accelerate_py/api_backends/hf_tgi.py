import os
import json
import time
import uuid
import threading
import requests
from concurrent.futures import Future
from queue import Queue
from dotenv import load_dotenv

import hashlib
import queue
from concurrent.futures import ThreadPoolExecutor

class hf_tgi:
    def __init__())))))))))self, resources=None, metadata=None):
        self.resources = resources if resources else {}}}}}}}}}}}}}}}}}}}}}}}
        self.metadata = metadata if metadata else {}}}}}}}}}}}}}}}}}}}}}}}
        
        # Get HF API token from metadata or environment
        self.api_token = self._get_api_token()))))))))))
        
        # Initialize queue and backoff systems
        self.max_concurrent_requests = 5
        self.queue_size = 100
        self.request_queue = Queue())))))))))maxsize=self.queue_size)
        self.active_requests = 0
        self.queue_lock = threading.RLock()))))))))))
        # Batching settings
        self.batching_enabled = True
        self.max_batch_size = 10
        self.batch_timeout = 0.5  # Max seconds to wait for more requests
        self.batch_queue = {}}}}}}}}}}}}}}}}}}}}}}}  # Keyed by model name
        self.batch_timers = {}}}}}}}}}}}}}}}}}}}}}}}  # Timers for each batch
        self.batch_lock = threading.RLock()))))))))))
        
        # Models that support batching
        self.embedding_models = [],,,,  # Models supporting batched embeddings,
        self.completion_models = [],,,,  # Models supporting batched completions,
        self.supported_batch_models = [],,,,  # All models supporting batching
        ,
        # Circuit breaker settings
        self.circuit_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_threshold = 5  # Number of failures before opening circuit
        self.reset_timeout = 30  # Seconds to wait before trying half-open
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_lock = threading.RLock()))))))))))

        # Priority levels
        self.PRIORITY_HIGH = 0
        self.PRIORITY_NORMAL = 1
        self.PRIORITY_LOW = 2
        
        # Change request queue to priority-based
        self.request_queue = [],,,,  # Will store ())))))))))priority, request_info) tuples

        ,
        # Start queue processor
        self.queue_processor = threading.Thread())))))))))target=self._process_queue)
        self.queue_processor.daemon = True
        self.queue_processor.start()))))))))))
        
        # Initialize backoff configuration
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 16
        
        # Default model
        self.default_model = "mistralai/Mistral-7B-Instruct-v0.2"
        
    return

        # Initialize counters, queues, and settings for each endpoint 
    self.endpoints = {}}}}}}}}}}}}}}}}}}}}}}}  # Dictionary to store per-endpoint data
        
        # Retry and backoff settings ())))))))))global defaults)
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
    self.request_queue = [],,,,
    self.queue_lock = threading.RLock()))))))))))
        
        # Initialize thread pool for async processing
    self.thread_pool = ThreadPoolExecutor())))))))))max_workers=10)


        # Initialize counters, queues, and settings for each endpoint 
    self.endpoints = {}}}}}}}}}}}}}}}}}}}}}}}  # Dictionary to store per-endpoint data
        
        # Retry and backoff settings ())))))))))global defaults)
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
    self.request_queue = [],,,,
    self.queue_lock = threading.RLock()))))))))))
        
        # Initialize thread pool for async processing
    self.thread_pool = ThreadPoolExecutor())))))))))max_workers=10)
:
    def _get_api_token())))))))))self):
        """Get Hugging Face API token from metadata or environment"""
        # Try to get from metadata
        api_token = self.metadata.get())))))))))"hf_api_key") or self.metadata.get())))))))))"hf_api_token")
        if api_token:
        return api_token
        
        # Try to get from environment
        env_token = os.environ.get())))))))))"HF_API_KEY") or os.environ.get())))))))))"HF_API_TOKEN")
        if env_token:
        return env_token
        
        # Try to load from dotenv
        try:
            load_dotenv()))))))))))
            env_token = os.environ.get())))))))))"HF_API_KEY") or os.environ.get())))))))))"HF_API_TOKEN")
            if env_token:
            return env_token
        except ImportError:
            pass
        
        # Return None if no token found ())))))))))will allow unauthenticated requests)
        return None
        
    
    :
    def _process_queue())))))))))self, endpoint_id=None):
        """Process requests in the queue for a specific endpoint or global queue"""
        # Get the endpoint or use global settings
        if endpoint_id and endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id],,,,
            with endpoint["queue_lock"]:,,,,,,
            if endpoint["queue_processing"]:,
        return  # Another thread is already processing this endpoint's queue
        endpoint["queue_processing"] = True
        ,
        queue_to_process = endpoint["request_queue"],
        is_global_queue = False
        else:
            # Use global queue if no endpoint specified or endpoint doesn't exist:
            with self.queue_lock:
                if self.queue_processing:
                return  # Another thread is already processing the global queue
                self.queue_processing = True
                
                queue_to_process = self.request_queue
                is_global_queue = True
        
        try:
            while True:
                # Get the next request from the queue
                request_info = None
                
                if is_global_queue:
                    with self.queue_lock:
                        if not queue_to_process:
                            self.queue_processing = False
                        break
                            
                        # Check if we're at the concurrent request limit::
                        if self.current_requests >= self.max_concurrent_requests:
                            # Sleep briefly then check again
                            time.sleep())))))))))0.1)
                        continue
                            
                        # Get the next request and increase counter
                        request_info = queue_to_process.pop())))))))))0)
                        self.current_requests += 1
                else:
                    with endpoint["queue_lock"]:,,,,,,
                        if not queue_to_process:
                            endpoint["queue_processing"] = False,
                    break
                            
                        # Check if we're at the concurrent request limit::
                    if endpoint["current_requests"], >= endpoint["max_concurrent_requests"]:,
                            # Sleep briefly then check again
                    time.sleep())))))))))0.1)
                        continue
                            
                        # Get the next request and increase counter
                        request_info = queue_to_process.pop())))))))))0)
                        endpoint["current_requests"], += 1
                        ,
                # Process the request outside the lock
                if request_info:
                    try:
                        # Extract request details
                        endpoint_url = request_info.get())))))))))"endpoint_url")
                        data = request_info.get())))))))))"data")
                        api_key = request_info.get())))))))))"api_key")
                        request_id = request_info.get())))))))))"request_id")
                        endpoint_id = request_info.get())))))))))"endpoint_id")
                        future = request_info.get())))))))))"future")
                        method_name = request_info.get())))))))))"method", "make_request")
                        method_args = request_info.get())))))))))"args", [],,,,)
                        method_kwargs = request_info.get())))))))))"kwargs", {}}}}}}}}}}}}}}}}}}}}}}})
                        
                        # Make the request ())))))))))without queueing again)
                        # Save original queue_enabled value to prevent recursion
                        if is_global_queue:
                            original_queue_enabled = self.queue_enabled
                            self.queue_enabled = False  # Disable queueing to prevent recursion
                        else:
                            original_queue_enabled = endpoint["queue_enabled"],
                            endpoint["queue_enabled"], = False  # Disable queueing to prevent recursion
                        
                        try:
                            # Make the request based on method name
                            if hasattr())))))))))self, method_name) and callable())))))))))getattr())))))))))self, method_name)):
                                method = getattr())))))))))self, method_name)
                                
                                # Call the method with the provided arguments
                                if method_name.startswith())))))))))"make_"):
                                    # Direct API request methods
                                    result = method())))))))))
                                    endpoint_url=endpoint_url,
                                    data=data,
                                    api_key=api_key,
                                    request_id=request_id,
                                    endpoint_id=endpoint_id
                                    )
                                else:
                                    # Higher-level methods
                                    method_kwargs.update()))))))))){}}}}}}}}}}}}}}}}}}}}}}
                                    "request_id": request_id,
                                    "endpoint_id": endpoint_id,
                                    "api_key": api_key
                                    })
                                    result = method())))))))))*method_args, **method_kwargs)
                            else:
                                # Fallback to make_request or similar method
                                make_method = getattr())))))))))self, "make_request", None)
                                if not make_method:
                                    make_method = getattr())))))))))self, f"make_post_request_{}}}}}}}}}}}}}}}}}}}}}}self.__class__.__name__.lower()))))))))))}", None)
                                    
                                if make_method and callable())))))))))make_method):
                                    result = make_method())))))))))
                                    endpoint_url=endpoint_url,
                                    data=data,
                                    api_key=api_key,
                                    request_id=request_id,
                                    endpoint_id=endpoint_id
                                    )
                                else:
                                    raise AttributeError())))))))))f"Method {}}}}}}}}}}}}}}}}}}}}}}method_name} not found")
                            
                            # Store result in future
                                    future["result"] = result,
                                    future["completed"] = True,,,
                                    ,
                            # Update counters
                            if not is_global_queue:
                                with endpoint["queue_lock"]:,,,,,,
                                endpoint["successful_requests"] += 1,
                                endpoint["last_request_at"] = time.time()))))))))))
                                ,            ,
                                    # Update token counts if present in result:
                                    if isinstance())))))))))result, dict) and "usage" in result:
                                        usage = result["usage"],
                                        endpoint["total_tokens"] += usage.get())))))))))"total_tokens", 0),
                                        endpoint["input_tokens"] += usage.get())))))))))"prompt_tokens", 0),
                                        endpoint["output_tokens"] += usage.get())))))))))"completion_tokens", 0)
                                        ,
                        except Exception as e:
                            # Store error in future
                            future["error"] = e,,,,
                            future["completed"] = True,,,
                            ,print())))))))))f"Error processing queued request: {}}}}}}}}}}}}}}}}}}}}}}str())))))))))e)}")
                            
                            # Update counters
                            if not is_global_queue:
                                with endpoint["queue_lock"]:,,,,,,
                                endpoint["failed_requests"] += 1,
                                endpoint["last_request_at"] = time.time()))))))))))
                                ,
                        finally:
                            # Restore original queue_enabled value
                            if is_global_queue:
                                self.queue_enabled = original_queue_enabled
                            else:
                                endpoint["queue_enabled"], = original_queue_enabled
                    
                    finally:
                        # Decrement counter
                        if is_global_queue:
                            with self.queue_lock:
                                self.current_requests = max())))))))))0, self.current_requests - 1)
                        else:
                            with endpoint["queue_lock"]:,,,,,,
                            endpoint["current_requests"], = max())))))))))0, endpoint["current_requests"], - 1)
                            ,
                    # Brief pause to prevent CPU hogging
                            time.sleep())))))))))0.01)
                    
        except Exception as e:
            print())))))))))f"Error in queue processing thread: {}}}}}}}}}}}}}}}}}}}}}}str())))))))))e)}")
            
        finally:
            # Reset queue processing flag
            if is_global_queue:
                with self.queue_lock:
                    self.queue_processing = False
            else:
                with endpoint["queue_lock"]:,,,,,,
                endpoint["queue_processing"] = False,

    def make_post_request_hf_tgi())))))))))self, endpoint_url, data, api_token=None, stream=False, request_id=None, endpoint_id=None):
        """Make a request to HF API with queue and backoff"""
        # Use endpoint-specific API token if available:, fall back to default:
        if endpoint_id and endpoint_id in self.endpoints:
            # Get API token from endpoint settings
            endpoint = self.endpoints[endpoint_id],,,,
            if api_token is None:
                api_token = endpoint.get())))))))))"api_key", self.api_token)
        else:
            # Use provided token or default
            if api_token is None:
                api_token = self.api_token
        
        # Generate unique request ID if not provided:::::
        if request_id is None:
            request_id = f"req_{}}}}}}}}}}}}}}}}}}}}}}int())))))))))time.time())))))))))))}_{}}}}}}}}}}}}}}}}}}}}}}hashlib.md5())))))))))str())))))))))data).encode()))))))))))).hexdigest()))))))))))[:8]}"
            ,
        # Queue system with proper concurrency management
            future = Future()))))))))))
        
        # Create request info
            request_info = {}}}}}}}}}}}}}}}}}}}}}}
            "endpoint_url": endpoint_url,
            "data": data,
            "api_key": api_token,
            "stream": stream,
            "request_id": request_id,
            "endpoint_id": endpoint_id,
            "future": future
            }
        
        # Add to appropriate queue
        if endpoint_id and endpoint_id in self.endpoints:
            with self.endpoints[endpoint_id],,,,["queue_lock"]:,,,,,,
                # Check if queue is full:
                if len())))))))))self.endpoints[endpoint_id],,,,["request_queue"],) >= self.endpoints[endpoint_id],,,,["queue_size"]:
            raise ValueError())))))))))f"Request queue is full ()))))))))){}}}}}}}}}}}}}}}}}}}}}}self.endpoints[endpoint_id],,,,['queue_size']} items). Try again later.")
                
                # Add to endpoint queue
            self.endpoints[endpoint_id],,,,["request_queue"],.append())))))))))request_info)
                
                # Start queue processing if not already running:
            if not self.endpoints[endpoint_id],,,,["queue_processing"]:,
            threading.Thread())))))))))target=self._process_queue, args=())))))))))endpoint_id,)).start()))))))))))
        else:
            # Add to global queue
            self.request_queue.put())))))))))())))))))))future, endpoint_url, data, api_token, stream, request_id))
        
        # Get result ())))))))))blocks until request is processed)
            return future.result()))))))))))
        
    def generate_text())))))))))self, model_id, inputs, parameters=None, api_token=None, request_id=None, endpoint_id=None):
        """Generate text using HF TGI API"""
        # Format endpoint URL for the model
        endpoint_url = f"https://api-inference.huggingface.co/models/{}}}}}}}}}}}}}}}}}}}}}}model_id}"
        
        # Prepare data
        data = {}}}}}}}}}}}}}}}}}}}}}}"inputs": inputs}
        if parameters:
            data["parameters"] = parameters
            ,,
        # Make request with queue and backoff
            response = self.make_post_request_hf_tgi())))))))))
            endpoint_url=endpoint_url,
            data=data,
            api_token=api_token,
            request_id=request_id,
            endpoint_id=endpoint_id
            )
        
        # Process response based on format
        if isinstance())))))))))response, list):
            # Some models return list of generated texts
            result = response[0] if response else {}}}}}}}}}}}}}}}}}}}}}}"generated_text": ""},
        elif isinstance())))))))))response, dict):
            # Some models return dict with generated_text key
            result = response
        else:
            # Default fallback
            result = {}}}}}}}}}}}}}}}}}}}}}}"generated_text": str())))))))))response)}
            
        # Add request ID to response if provided::::::::::
        if request_id and isinstance())))))))))result, dict):
            result["request_id"] = request_id,
            ,
            return result

    def stream_generate())))))))))self, model_id, inputs, parameters=None, api_token=None, request_id=None, endpoint_id=None):
        """Stream text generation from HF TGI API"""
        # Format endpoint URL for the model
        endpoint_url = f"https://api-inference.huggingface.co/models/{}}}}}}}}}}}}}}}}}}}}}}model_id}"
        
        # Prepare data
        data = {}}}}}}}}}}}}}}}}}}}}}}
        "inputs": inputs,
        "stream": True
        }
        if parameters:
            data["parameters"] = parameters
            ,,
        # Make streaming request
            response_stream = self.make_post_request_hf_tgi())))))))))
            endpoint_url=endpoint_url,
            data=data,
            api_token=api_token,
            stream=True,
            request_id=request_id,
            endpoint_id=endpoint_id
            )
        
        # Yield each chunk with request ID if provided::::::::::
        for chunk in response_stream:
            # Add request ID to chunk if provided:::::::::: and chunk is a dict
            if request_id and isinstance())))))))))chunk, dict):
                chunk["request_id"] = request_id,
                ,    yield chunk
            
    def chat())))))))))self, model, messages, options=None, request_id=None, endpoint_id=None):
        """Format chat messages and generate text"""
        # Format messages into a prompt
        prompt = self._format_chat_messages())))))))))messages)
        
        # Extract parameters from options
        parameters = {}}}}}}}}}}}}}}}}}}}}}}}
        if options:
            # Map common parameters
            if "temperature" in options:
                parameters["temperature"] = options["temperature"],,
            if "max_tokens" in options:
                parameters["max_new_tokens"] = options["max_tokens"],,
            if "top_p" in options:
                parameters["top_p"] = options["top_p"],,
            if "top_k" in options:
                parameters["top_k"] = options["top_k"]
                ,,
        # Generate request ID if not provided:::::
        if request_id is None and options and "request_id" in options:
            request_id = options["request_id"]
            ,,
        # Use endpoint ID from options if provided::::::::::
        if endpoint_id is None and options and "endpoint_id" in options:
            endpoint_id = options["endpoint_id"]
            ,,
        # Generate text
            response = self.generate_text())))))))))
            model_id=model, 
            inputs=prompt, 
            parameters=parameters,
            request_id=request_id,
            endpoint_id=endpoint_id
            )
        
        # Extract text from response
        if isinstance())))))))))response, dict):
            text = response.get())))))))))"generated_text", "")
        else:
            text = str())))))))))response)
            
        # Return standardized response
            result = {}}}}}}}}}}}}}}}}}}}}}}
            "text": text,
            "model": model,
            "usage": self._estimate_usage())))))))))prompt, text),
            "implementation_type": "())))))))))REAL)"
            }
        
        # Include request ID if provided::::::::::
        if request_id:
            result["request_id"] = request_id,
            ,
            return result
        
    def stream_chat())))))))))self, model, messages, options=None, request_id=None, endpoint_id=None):
        """Stream chat responses"""
        # Format messages into a prompt
        prompt = self._format_chat_messages())))))))))messages)
        
        # Extract parameters from options
        parameters = {}}}}}}}}}}}}}}}}}}}}}}}
        if options:
            # Map common parameters
            if "temperature" in options:
                parameters["temperature"] = options["temperature"],,
            if "max_tokens" in options:
                parameters["max_new_tokens"] = options["max_tokens"],,
            if "top_p" in options:
                parameters["top_p"] = options["top_p"],,
            if "top_k" in options:
                parameters["top_k"] = options["top_k"]
                ,,
        # Generate request ID if not provided:::::
        if request_id is None and options and "request_id" in options:
            request_id = options["request_id"]
            ,,
        # Use endpoint ID from options if provided::::::::::
        if endpoint_id is None and options and "endpoint_id" in options:
            endpoint_id = options["endpoint_id"]
            ,,
        # Create stream
            response_stream = self.stream_generate())))))))))
            model_id=model, 
            inputs=prompt, 
            parameters=parameters,
            request_id=request_id,
            endpoint_id=endpoint_id
            )
        
        # Process each chunk
        for chunk in response_stream:
            if isinstance())))))))))chunk, dict) and "token" in chunk:
                text = chunk.get())))))))))"token", {}}}}}}}}}}}}}}}}}}}}}}}).get())))))))))"text", "")
                done = chunk.get())))))))))"generated_text", False) is not False
                
                result = {}}}}}}}}}}}}}}}}}}}}}}
                "text": text,
                "done": done,
                "model": model
                }
                
                # Include request ID if provided::::::::::
                if request_id:
                    result["request_id"] = request_id,
                    ,
                    yield result
            else:
                # Handle other response formats
                text = str())))))))))chunk)
                result = {}}}}}}}}}}}}}}}}}}}}}}
                "text": text,
                "done": False,
                "model": model
                }
                
                # Include request ID if provided::::::::::
                if request_id:
                    result["request_id"] = request_id,
                    ,
                    yield result
                
    def _format_chat_messages())))))))))self, messages):
        """Format chat messages into a prompt for HF models"""
        formatted_prompt = ""
        
        for message in messages:
            role = message.get())))))))))"role", "user")
            content = message.get())))))))))"content", "")
            
            if role == "system":
                formatted_prompt += f"<|system|>\n{}}}}}}}}}}}}}}}}}}}}}}content}\n"
            elif role == "assistant":
                formatted_prompt += f"<|assistant|>\n{}}}}}}}}}}}}}}}}}}}}}}content}\n"
            else:  # user or default
                formatted_prompt += f"<|user|>\n{}}}}}}}}}}}}}}}}}}}}}}content}\n"
                
        # Add final assistant marker for completion
                formatted_prompt += "<|assistant|>\n"
        
            return formatted_prompt
        
    def _estimate_usage())))))))))self, prompt, response):
        """Estimate token usage ())))))))))rough approximation)"""
        # Very rough approximation: 4 chars ~= 1 token
        prompt_tokens = len())))))))))prompt) // 4
        completion_tokens = len())))))))))response) // 4
        
            return {}}}}}}}}}}}}}}}}}}}}}}
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
            }
        
    def create_remote_text_generation_endpoint_handler())))))))))self, endpoint_url=None, api_key=None, endpoint_id=None):
        """Create an endpoint handler for HF TGI remote inference"""
        async def endpoint_handler())))))))))prompt, **kwargs):
            """Handle requests to HF TGI endpoint"""
            # Extract request ID if provided::::::::::
            request_id = kwargs.get())))))))))"request_id")
            
            # Use specific endpoint ID from kwargs or from constructor
            current_endpoint_id = kwargs.get())))))))))"endpoint_id", endpoint_id)
            
            # If no specific model endpoint provided, use standard API
            if not endpoint_url:
                # Extract model from kwargs or use default
                model = kwargs.get())))))))))"model", self.default_model)
                
                # Create endpoint URL
                model_endpoint = f"https://api-inference.huggingface.co/models/{}}}}}}}}}}}}}}}}}}}}}}model}"
            else:
                model_endpoint = endpoint_url
                model = kwargs.get())))))))))"model", self.default_model)
                
            # Extract parameters from kwargs
                max_new_tokens = kwargs.get())))))))))"max_new_tokens", 1024)
                temperature = kwargs.get())))))))))"temperature", 0.7)
                top_p = kwargs.get())))))))))"top_p", 0.9)
                do_sample = kwargs.get())))))))))"do_sample", True)
            
            # Prepare parameters
                parameters = {}}}}}}}}}}}}}}}}}}}}}}
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "return_full_text": kwargs.get())))))))))"return_full_text", False)
                }
            
            # Check if prompt is a list of messages:
            if isinstance())))))))))prompt, list):
                # Format as chat messages
                prompt_text = self._format_chat_messages())))))))))prompt)
            else:
                prompt_text = prompt
            
            # Use streaming if requested:
            if kwargs.get())))))))))"stream", False):
                async def process_stream())))))))))):
                    streaming_response = self.stream_generate())))))))))
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
                            result_chunk = {}}}}}}}}}}}}}}}}}}}}}}
                            "text": chunk["token"]["text"],
                            "done": False,
                            "model": model,
                            "implementation_type": "())))))))))REAL)"
                            }
                        elif "generated_text" in chunk:
                            result_chunk = {}}}}}}}}}}}}}}}}}}}}}}
                            "text": chunk["generated_text"],
                            "done": True,
                            "model": model,
                            "implementation_type": "())))))))))REAL)"
                            }
                        else:
                            result_chunk = {}}}}}}}}}}}}}}}}}}}}}}
                            "text": str())))))))))chunk),
                            "done": False,
                            "model": model,
                            "implementation_type": "())))))))))REAL)"
                            }
                        
                        # Add request ID if provided::::::::::
                        if request_id:
                            result_chunk["request_id"] = request_id,
                            ,
                            yield result_chunk
                
                            return process_stream()))))))))))
            else:
                try:
                    # Make the request
                    response = self.generate_text())))))))))
                    model_id=model,
                    inputs=prompt_text,
                    parameters=parameters,
                    api_token=api_key,
                    request_id=request_id,
                    endpoint_id=current_endpoint_id
                    )
                    
                    # Extract generated text
                    if isinstance())))))))))response, list):
                        generated_text = response[0].get())))))))))"generated_text", "") if response else "":,
                    elif isinstance())))))))))response, dict):
                        generated_text = response.get())))))))))"generated_text", "")
                    else:
                        generated_text = str())))))))))response)
                    
                        usage = self._estimate_usage())))))))))prompt_text, generated_text)
                    
                    # Create response with metadata
                        result = {}}}}}}}}}}}}}}}}}}}}}}
                        "text": generated_text, 
                        "model": model,
                        "usage": usage,
                        "implementation_type": "())))))))))REAL)"
                        }
                    
                    # Add request ID if provided::::::::::
                    if request_id:
                        result["request_id"] = request_id,
                        ,
                        return result
                except Exception as e:
                    error_message = str())))))))))e)
                    print())))))))))f"Error calling HF TGI endpoint: {}}}}}}}}}}}}}}}}}}}}}}error_message}")
                    
                    # Create error response with request ID
                    error_response = {}}}}}}}}}}}}}}}}}}}}}}
                    "text": f"Error: {}}}}}}}}}}}}}}}}}}}}}}error_message}",
                    "implementation_type": "())))))))))ERROR)",
                    "model": model
                    }
                    
                    # Add request ID if provided::::::::::
                    if request_id:
                        error_response["request_id"] = request_id,
                        ,
                    return error_response
        
                        return endpoint_handler
        
    def test_hf_tgi_endpoint())))))))))self, endpoint_url=None, api_token=None, model_id=None, endpoint_id=None, request_id=None):
        """Test the HF TGI endpoint"""
        if not model_id:
            model_id = self.default_model
            
        if not endpoint_url:
            endpoint_url = f"https://api-inference.huggingface.co/models/{}}}}}}}}}}}}}}}}}}}}}}model_id}"
        
        # Use endpoint-specific API token if available:
        if endpoint_id and endpoint_id in self.endpoints:
            if api_token is None:
                api_token = self.endpoints[endpoint_id],,,,.get())))))))))"api_key", self.api_token)
        elif api_token is None:
            api_token = self.api_token
        
        # Generate unique request ID for the test if not provided:::::
        if request_id is None:
            request_id = f"test_{}}}}}}}}}}}}}}}}}}}}}}int())))))))))time.time())))))))))))}_{}}}}}}}}}}}}}}}}}}}}}}hashlib.md5())))))))))model_id.encode()))))))))))).hexdigest()))))))))))[:8]}"
            ,
        try:
            # Test text generation with proper request tracking
            response = self.generate_text())))))))))
            model_id=model_id,
            inputs="Testing the Hugging Face TGI API. Please respond with a short message.",
            parameters={}}}}}}}}}}}}}}}}}}}}}}"max_new_tokens": 50},
            api_token=api_token,
            request_id=request_id,
            endpoint_id=endpoint_id
            )
            
            # Verify we got a valid response
            if isinstance())))))))))response, dict) and ())))))))))"generated_text" in response or "text" in response):
                result = {}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "message": "TGI API test successful",
                "model": model_id,
                "implementation_type": "())))))))))REAL)",
                "request_id": request_id
                }
            return result
            elif isinstance())))))))))response, list) and len())))))))))response) > 0:
                result = {}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "message": "TGI API test successful ())))))))))list response)",
                "model": model_id,
                "implementation_type": "())))))))))REAL)",
                "request_id": request_id
                }
            return result
            else:
                result = {}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "message": "TGI API test failed: unexpected response format",
                "implementation_type": "())))))))))ERROR)",
                "request_id": request_id
                }
            return result
        except Exception as e:
            error_message = str())))))))))e)
            print())))))))))f"Error testing HF TGI endpoint: {}}}}}}}}}}}}}}}}}}}}}}error_message}")
            result = {}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "message": f"TGI API test failed: {}}}}}}}}}}}}}}}}}}}}}}error_message}",
            "implementation_type": "())))))))))ERROR)",
            "request_id": request_id
            }
            return result

    
            def create_endpoint())))))))))self, endpoint_id=None, api_key=None, max_retries=None, initial_retry_delay=None,
            backoff_factor=None, max_retry_delay=None, queue_enabled=None,
                       max_concurrent_requests=None, queue_size=None):
                           """Create a new endpoint with its own settings and counters"""
        # Generate a unique endpoint ID if not provided:::::
        if endpoint_id is None:
            endpoint_id = f"endpoint_{}}}}}}}}}}}}}}}}}}}}}}int())))))))))time.time())))))))))))}_{}}}}}}}}}}}}}}}}}}}}}}hashlib.md5())))))))))str())))))))))time.time()))))))))))).encode()))))))))))).hexdigest()))))))))))[:8]}"
            ,
        # Use provided values or defaults
            endpoint_settings = {}}}}}}}}}}}}}}}}}}}}}}
            "api_key": api_key if api_key is not None else self.api_key,:
            "max_retries": max_retries if max_retries is not None else self.max_retries,:
            "initial_retry_delay": initial_retry_delay if initial_retry_delay is not None else self.initial_retry_delay,:
            "backoff_factor": backoff_factor if backoff_factor is not None else self.backoff_factor,:
            "max_retry_delay": max_retry_delay if max_retry_delay is not None else self.max_retry_delay,:
            "queue_enabled": queue_enabled if queue_enabled is not None else self.queue_enabled,:
            "max_concurrent_requests": max_concurrent_requests if max_concurrent_requests is not None else self.max_concurrent_requests,:
                "queue_size": queue_size if queue_size is not None else self.queue_size,
            
            # Initialize endpoint-specific counters and state:
                "current_requests": 0,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "queue_processing": False,
                "request_queue": [],,,,,
                "queue_lock": threading.RLock())))))))))),
                "created_at": time.time())))))))))),
                "last_request_at": None
                }
        
        # Store the endpoint settings
                self.endpoints[endpoint_id],,,, = endpoint_settings
        
                return endpoint_id

    
    def get_endpoint())))))))))self, endpoint_id=None):
        """Get an endpoint's settings or create a default one if not found"""
        # If no endpoint_id provided, use the first one or create a default:
        if endpoint_id is None:
            if not self.endpoints:
                endpoint_id = self.create_endpoint()))))))))))
            else:
                endpoint_id = next())))))))))iter())))))))))self.endpoints))
                
        # If endpoint doesn't exist, create it
        if endpoint_id not in self.endpoints:
            endpoint_id = self.create_endpoint())))))))))endpoint_id=endpoint_id)
            
                return self.endpoints[endpoint_id],,,,

    
    def update_endpoint())))))))))self, endpoint_id, **kwargs):
        """Update an endpoint's settings"""
        if endpoint_id not in self.endpoints:
        raise ValueError())))))))))f"Endpoint {}}}}}}}}}}}}}}}}}}}}}}endpoint_id} not found")
            
        # Update only the provided settings
        for key, value in kwargs.items())))))))))):
            if key in self.endpoints[endpoint_id],,,,:
                self.endpoints[endpoint_id],,,,[key] = value
                
            return self.endpoints[endpoint_id],,,,

    
    def get_stats())))))))))self, endpoint_id=None):
        """Get usage statistics for an endpoint or global stats"""
        if endpoint_id and endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id],,,,
            stats = {}}}}}}}}}}}}}}}}}}}}}}
            "endpoint_id": endpoint_id,
            "total_requests": endpoint["total_requests"],
            "successful_requests": endpoint["successful_requests"],
            "failed_requests": endpoint["failed_requests"],
            "total_tokens": endpoint["total_tokens"],
            "input_tokens": endpoint["input_tokens"],
            "output_tokens": endpoint["output_tokens"],
            "created_at": endpoint["created_at"],
            "last_request_at": endpoint["last_request_at"],
            "current_queue_size": len())))))))))endpoint["request_queue"],),
            "current_requests": endpoint["current_requests"],
            }
        return stats
        else:
            # Aggregate stats across all endpoints
            total_requests = sum())))))))))e["total_requests"] for e in self.endpoints.values()))))))))))) if self.endpoints else 0,
            successful_requests = sum())))))))))e["successful_requests"] for e in self.endpoints.values()))))))))))) if self.endpoints else 0,
            failed_requests = sum())))))))))e["failed_requests"] for e in self.endpoints.values()))))))))))) if self.endpoints else 0,
            total_tokens = sum())))))))))e["total_tokens"] for e in self.endpoints.values()))))))))))) if self.endpoints else 0,
            input_tokens = sum())))))))))e["input_tokens"] for e in self.endpoints.values()))))))))))) if self.endpoints else 0,
            output_tokens = sum())))))))))e["output_tokens"] for e in self.endpoints.values()))))))))))) if self.endpoints else 0
            ,
            stats = {}}}}}}}}}}}}}}}}}}}}}}:
                "endpoints_count": len())))))))))self.endpoints),
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "global_queue_size": len())))))))))self.request_queue),
                "global_current_requests": self.current_requests
                }
            return stats

    
    def reset_stats())))))))))self, endpoint_id=None):
        """Reset usage statistics for an endpoint or globally"""
        if endpoint_id and endpoint_id in self.endpoints:
            # Reset stats just for this endpoint
            endpoint = self.endpoints[endpoint_id],,,,
            endpoint["total_requests"] = 0,,
            endpoint["successful_requests"] = 0,,
            endpoint["failed_requests"] = 0,,
            endpoint["total_tokens"] = 0,,
            endpoint["input_tokens"] = 0,,
            endpoint["output_tokens"] = 0,,
        elif endpoint_id is None:
            # Reset stats for all endpoints
            for endpoint in self.endpoints.values())))))))))):
                endpoint["total_requests"] = 0,,
                endpoint["successful_requests"] = 0,,
                endpoint["failed_requests"] = 0,,
                endpoint["total_tokens"] = 0,,
                endpoint["input_tokens"] = 0,,
                endpoint["output_tokens"] = 0,,
        else:
                raise ValueError())))))))))f"Endpoint {}}}}}}}}}}}}}}}}}}}}}}endpoint_id} not found")

    def check_circuit_breaker())))))))))self):
        # Check if circuit breaker allows requests to proceed:
        with self.circuit_lock:
            now = time.time()))))))))))
            
            if self.circuit_state == "OPEN":
                # Check if enough time has passed to try again:
                if now - self.last_failure_time > self.reset_timeout:
                    logger.info())))))))))"Circuit breaker transitioning from OPEN to HALF-OPEN")
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

    def track_request_result())))))))))self, success, error_type=None):
        # Track the result of a request for circuit breaker logic tracking
        with self.circuit_lock:
            if success:
                # Successful request
                if self.circuit_state == "HALF_OPEN":
                    # Service is working again, close the circuit
                    logger.info())))))))))"Circuit breaker transitioning from HALF-OPEN to CLOSED")
                    self.circuit_state = "CLOSED"
                    self.failure_count = 0
                elif self.circuit_state == "CLOSED":
                    # Reset failure count on success
                    self.failure_count = 0
            else:
                # Failed request
                self.failure_count += 1
                self.last_failure_time = time.time()))))))))))
                
                # Update error statistics
                if error_type and hasattr())))))))))self, "collect_metrics") and self.collect_metrics:
                    with self.stats_lock:
                        if error_type not in self.request_stats["errors_by_type"]:,
                        self.request_stats["errors_by_type"][error_type] = 0,
                        self.request_stats["errors_by_type"][error_type] += 1
                        ,
                if self.circuit_state == "CLOSED" and self.failure_count >= self.failure_threshold:
                    # Too many failures, open the circuit
                    logger.warning())))))))))f"Circuit breaker transitioning from CLOSED to OPEN after {}}}}}}}}}}}}}}}}}}}}}}self.failure_count} failures")
                    self.circuit_state = "OPEN"
                    
                    # Update circuit breaker statistics
                    if hasattr())))))))))self, "stats_lock") and hasattr())))))))))self, "request_stats"):
                        with self.stats_lock:
                            if "circuit_breaker_trips" not in self.request_stats:
                                self.request_stats["circuit_breaker_trips"] = 0,
                                self.request_stats["circuit_breaker_trips"] += 1
                                ,
                elif self.circuit_state == "HALF_OPEN":
                    # Failed during test request, back to open
                    logger.warning())))))))))"Circuit breaker transitioning from HALF-OPEN to OPEN after test request failure")
                    self.circuit_state = "OPEN"
    
    def add_to_batch())))))))))self, model, request_info):
        # Add a request to the batch queue for the specified model
        if not hasattr())))))))))self, "batching_enabled") or not self.batching_enabled or model not in self.supported_batch_models:
            # Either batching is disabled or model doesn't support it
        return False
            
        with self.batch_lock:
            # Initialize batch queue for this model if needed:
            if model not in self.batch_queue:
                self.batch_queue[model],, = [],,,,
                
            # Add request to batch
                self.batch_queue[model],,.append())))))))))request_info)
                ,
            # Check if we need to start a timer for this batch:
                if len())))))))))self.batch_queue[model],,) == 1:,
                # First item in batch, start timer
                if model in self.batch_timers and self.batch_timers[model],, is not None:,
                self.batch_timers[model],,.cancel()))))))))))
                ,
                self.batch_timers[model],, = threading.Timer()))))))))),
                self.batch_timeout,
                self._process_batch,
                args=[model],,
                )
                self.batch_timers[model],,.daemon = True
                self.batch_timers[model],,.start()))))))))))
                
            # Check if batch is full and should be processed immediately:
            if len())))))))))self.batch_queue[model],,) >= self.max_batch_size:
                # Cancel timer since we're processing now
                if model in self.batch_timers and self.batch_timers[model],, is not None:,
                self.batch_timers[model],,.cancel()))))))))))
                ,    self.batch_timers[model],, = None
                    
                # Process batch immediately
                threading.Thread())))))))))target=self._process_batch, args=[model],,).start()))))))))))
                return True
                
            return True
    
    def _process_batch())))))))))self, model):
        # Process a batch of requests for the specified model
        with self.batch_lock:
            # Get all requests for this model
            if model not in self.batch_queue:
            return
                
            batch_requests = self.batch_queue[model],,
            self.batch_queue[model],, = [],,,,
            
            # Clear timer reference
            if model in self.batch_timers:
                self.batch_timers[model],, = None
        
        if not batch_requests:
                return
            
        # Update batch statistics
        if hasattr())))))))))self, "collect_metrics") and self.collect_metrics and hasattr())))))))))self, "update_stats"):
            self.update_stats()))))))))){}}}}}}}}}}}}}}}}}}}}}}"batched_requests": len())))))))))batch_requests)})
        
        try:
            # Check which type of batch processing to use
            if model in self.embedding_models:
                self._process_embedding_batch())))))))))model, batch_requests)
            elif model in self.completion_models:
                self._process_completion_batch())))))))))model, batch_requests)
            else:
                logger.warning())))))))))f"Unknown batch processing type for model {}}}}}}}}}}}}}}}}}}}}}}model}")
                # Fail all requests in the batch
                for req in batch_requests:
                    future = req.get())))))))))"future")
                    if future:
                        future["error"] = Exception())))))))))f"No batch processing available for model {}}}}}}}}}}}}}}}}}}}}}}model}"),
                        future["completed"] = True,,,
                
        except Exception as e:
            logger.error())))))))))f"Error processing batch for model {}}}}}}}}}}}}}}}}}}}}}}model}: {}}}}}}}}}}}}}}}}}}}}}}e}")
            
            # Set error for all futures in the batch
            for req in batch_requests:
                future = req.get())))))))))"future")
                if future:
                    future["error"] = e,,,,
                    future["completed"] = True,,,
    
    def _process_embedding_batch())))))))))self, model, batch_requests):
        # Process a batch of embedding requests for improved throughput
        try:
            # Extract texts from requests
            texts = [],,,,
            for req in batch_requests:
                data = req.get())))))))))"data", {}}}}}}}}}}}}}}}}}}}}}}})
                text = data.get())))))))))"text", data.get())))))))))"input", ""))
                texts.append())))))))))text)
            
            # This is a placeholder - subclasses should implement this
            # with the actual batched embedding API call
                batch_result = {}}}}}}}}}}}}}}}}}}}}}}"embeddings": [[0.1, 0.2] * 50] * len())))))))))texts)}
                ,
            # Distribute results to individual futures
            for i, req in enumerate())))))))))batch_requests):
                future = req.get())))))))))"future")
                if future and i < len())))))))))batch_result.get())))))))))"embeddings", [],,,,)):
                    future["result"] = {}}}}}}}}}}}}}}}}}}}}}},,
                    "embedding": batch_result["embeddings"][i],
                    "model": model,
                    "implementation_type": "MOCK-BATCHED"
                    }
                    future["completed"] = True,,,
                elif future:
                    future["error"] = Exception())))))))))"Batch embedding result index out of range"),
                    future["completed"] = True,,,
                    
        except Exception as e:
            # Propagate error to all futures
            for req in batch_requests:
                future = req.get())))))))))"future")
                if future:
                    future["error"] = e,,,,
                    future["completed"] = True,,,
    
    def _process_completion_batch())))))))))self, model, batch_requests):
        # Process a batch of completion requests in one API call
        try:
            # Extract prompts from requests
            prompts = [],,,,
            for req in batch_requests:
                data = req.get())))))))))"data", {}}}}}}}}}}}}}}}}}}}}}}})
                prompt = data.get())))))))))"prompt", data.get())))))))))"input", ""))
                prompts.append())))))))))prompt)
            
            # This is a placeholder - subclasses should implement this
            # with the actual batched completion API call
                batch_result = {}}}}}}}}}}}}}}}}}}}}}}"completions": [f"Mock response for prompt {}}}}}}}}}}}}}}}}}}}}}}i}" for i in range())))))))))len())))))))))prompts))]}:,
            # Distribute results to individual futures
            for i, req in enumerate())))))))))batch_requests):
                future = req.get())))))))))"future")
                if future and i < len())))))))))batch_result.get())))))))))"completions", [],,,,)):
                    future["result"] = {}}}}}}}}}}}}}}}}}}}}}},,
                    "text": batch_result["completions"][i],
                    "model": model,
                    "implementation_type": "MOCK-BATCHED"
                    }
                    future["completed"] = True,,,
                elif future:
                    future["error"] = Exception())))))))))"Batch completion result index out of range"),
                    future["completed"] = True,,,
                    
        except Exception as e:
            # Propagate error to all futures
            for req in batch_requests:
                future = req.get())))))))))"future")
                if future:
                    future["error"] = e,,,,
                    future["completed"] = True,,,
    