import os
import json
import time
import uuid
import threading
import requests
import numpy as np
from concurrent.futures import Future
from queue import Queue
from dotenv import load_dotenv

import hashlib
import queue
from concurrent.futures import ThreadPoolExecutor

class hf_tei:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get HF API token from metadata or environment
        self.api_token = self._get_api_token()
        
        # Initialize queue and backoff systems
        self.max_concurrent_requests = 5
        self.queue_size = 100
        self.request_queue = Queue(maxsize=self.queue_size)
        self.active_requests = 0
        self.queue_lock = threading.RLock()
        
        # Start queue processor
        self.queue_processor = threading.Thread(target=self._process_queue)
        self.queue_processor.daemon = True
        self.queue_processor.start()
        
        # Initialize backoff configuration
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 16
        
        # Default model
        self.default_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        return

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
        
        # Initialize thread pool for async processing
        self.thread_pool = ThreadPoolExecutor(max_workers=10)

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
        
    
    def _process_queue(self, endpoint_id=None):
        """Process requests in the queue for a specific endpoint or global queue"""
        # Get the endpoint or use global settings
        if endpoint_id and endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            with endpoint["queue_lock"]:
                if endpoint["queue_processing"]:
                    return  # Another thread is already processing this endpoint's queue
                endpoint["queue_processing"] = True
                
            queue_to_process = endpoint["request_queue"]
            is_global_queue = False
        else:
            # Use global queue if no endpoint specified or endpoint doesn't exist
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
                            
                        # Check if we're at the concurrent request limit
                        if self.current_requests >= self.max_concurrent_requests:
                            # Sleep briefly then check again
                            time.sleep(0.1)
                            continue
                            
                        # Get the next request and increase counter
                        request_info = queue_to_process.pop(0)
                        self.current_requests += 1
                else:
                    with endpoint["queue_lock"]:
                        if not queue_to_process:
                            endpoint["queue_processing"] = False
                            break
                            
                        # Check if we're at the concurrent request limit
                        if endpoint["current_requests"] >= endpoint["max_concurrent_requests"]:
                            # Sleep briefly then check again
                            time.sleep(0.1)
                            continue
                            
                        # Get the next request and increase counter
                        request_info = queue_to_process.pop(0)
                        endpoint["current_requests"] += 1
                
                # Process the request outside the lock
                if request_info:
                    try:
                        # Extract request details
                        endpoint_url = request_info.get("endpoint_url")
                        data = request_info.get("data")
                        api_key = request_info.get("api_key")
                        request_id = request_info.get("request_id")
                        endpoint_id = request_info.get("endpoint_id")
                        future = request_info.get("future")
                        method_name = request_info.get("method", "make_request")
                        method_args = request_info.get("args", [])
                        method_kwargs = request_info.get("kwargs", {})
                        
                        # Make the request (without queueing again)
                        # Save original queue_enabled value to prevent recursion
                        if is_global_queue:
                            original_queue_enabled = self.queue_enabled
                            self.queue_enabled = False  # Disable queueing to prevent recursion
                        else:
                            original_queue_enabled = endpoint["queue_enabled"]
                            endpoint["queue_enabled"] = False  # Disable queueing to prevent recursion
                        
                        try:
                            # Make the request based on method name
                            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                                method = getattr(self, method_name)
                                
                                # Call the method with the provided arguments
                                if method_name.startswith("make_"):
                                    # Direct API request methods
                                    result = method(
                                        endpoint_url=endpoint_url,
                                        data=data,
                                        api_key=api_key,
                                        request_id=request_id,
                                        endpoint_id=endpoint_id
                                    )
                                else:
                                    # Higher-level methods
                                    method_kwargs.update({
                                        "request_id": request_id,
                                        "endpoint_id": endpoint_id,
                                        "api_key": api_key
                                    })
                                    result = method(*method_args, **method_kwargs)
                            else:
                                # Fallback to make_request or similar method
                                make_method = getattr(self, "make_request", None)
                                if not make_method:
                                    make_method = getattr(self, f"make_post_request_{self.__class__.__name__.lower()}", None)
                                    
                                if make_method and callable(make_method):
                                    result = make_method(
                                        endpoint_url=endpoint_url,
                                        data=data,
                                        api_key=api_key,
                                        request_id=request_id,
                                        endpoint_id=endpoint_id
                                    )
                                else:
                                    raise AttributeError(f"Method {method_name} not found")
                            
                            # Store result in future
                            future["result"] = result
                            future["completed"] = True
                            
                            # Update counters
                            if not is_global_queue:
                                with endpoint["queue_lock"]:
                                    endpoint["successful_requests"] += 1
                                    endpoint["last_request_at"] = time.time()
                                    
                                    # Update token counts if present in result
                                    if isinstance(result, dict) and "usage" in result:
                                        usage = result["usage"]
                                        endpoint["total_tokens"] += usage.get("total_tokens", 0)
                                        endpoint["input_tokens"] += usage.get("prompt_tokens", 0)
                                        endpoint["output_tokens"] += usage.get("completion_tokens", 0)
                            
                        except Exception as e:
                            # Store error in future
                            future["error"] = e
                            future["completed"] = True
                            print(f"Error processing queued request: {str(e)}")
                            
                            # Update counters
                            if not is_global_queue:
                                with endpoint["queue_lock"]:
                                    endpoint["failed_requests"] += 1
                                    endpoint["last_request_at"] = time.time()
                        
                        finally:
                            # Restore original queue_enabled value
                            if is_global_queue:
                                self.queue_enabled = original_queue_enabled
                            else:
                                endpoint["queue_enabled"] = original_queue_enabled
                    
                    finally:
                        # Decrement counter
                        if is_global_queue:
                            with self.queue_lock:
                                self.current_requests = max(0, self.current_requests - 1)
                        else:
                            with endpoint["queue_lock"]:
                                endpoint["current_requests"] = max(0, endpoint["current_requests"] - 1)
                    
                    # Brief pause to prevent CPU hogging
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"Error in queue processing thread: {str(e)}")
            
        finally:
            # Reset queue processing flag
            if is_global_queue:
                with self.queue_lock:
                    self.queue_processing = False
            else:
                with endpoint["queue_lock"]:
                    endpoint["queue_processing"] = False

    def make_post_request_hf_tei(self, endpoint_url, data, api_token=None, request_id=None, endpoint_id=None):
        """Make a request to HF TEI API with queue and backoff"""
        # Use endpoint-specific API token if available, fall back to default
        if endpoint_id and endpoint_id in self.endpoints:
            # Get API token from endpoint settings
            endpoint = self.endpoints[endpoint_id]
            if api_token is None:
                api_token = endpoint.get("api_key", self.api_token)
        else:
            # Use provided token or default
            if api_token is None:
                api_token = self.api_token
        
        # Generate unique request ID if not provided
        if request_id is None:
            request_id = f"req_{int(time.time())}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"
        
        # Queue system with proper concurrency management
        future = Future()
        
        # Create request info
        request_info = {
            "endpoint_url": endpoint_url,
            "data": data,
            "api_key": api_token,
            "request_id": request_id,
            "endpoint_id": endpoint_id,
            "future": future
        }
        
        # Add to appropriate queue
        if endpoint_id and endpoint_id in self.endpoints:
            with self.endpoints[endpoint_id]["queue_lock"]:
                # Check if queue is full
                if len(self.endpoints[endpoint_id]["request_queue"]) >= self.endpoints[endpoint_id]["queue_size"]:
                    raise ValueError(f"Request queue is full ({self.endpoints[endpoint_id]['queue_size']} items). Try again later.")
                
                # Add to endpoint queue
                self.endpoints[endpoint_id]["request_queue"].append(request_info)
                
                # Start queue processing if not already running
                if not self.endpoints[endpoint_id]["queue_processing"]:
                    threading.Thread(target=self._process_queue, args=(endpoint_id,)).start()
        else:
            # Add to global queue
            self.request_queue.put((future, endpoint_url, data, api_token, request_id))
            
        # Get result (blocks until request is processed)
        return future.result()
        
    def generate_embedding(self, model_id, text, api_token=None, request_id=None, endpoint_id=None):
        """Generate embeddings for a single text using HF TEI API"""
        # Format endpoint URL for the model
        endpoint_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        
        # Prepare data
        data = {"inputs": text}
        
        # Make request with queue and backoff
        response = self.make_post_request_hf_tei(
            endpoint_url=endpoint_url, 
            data=data, 
            api_token=api_token, 
            request_id=request_id, 
            endpoint_id=endpoint_id
        )
        
        # Process response - normalize if needed
        return self.normalize_embedding(response)

    def batch_embed(self, model_id, texts, api_token=None, request_id=None, endpoint_id=None):
        """Generate embeddings for multiple texts using HF TEI API"""
        # Format endpoint URL for the model
        endpoint_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        
        # Prepare data
        data = {"inputs": texts}
        
        # Make request with queue and backoff
        response = self.make_post_request_hf_tei(
            endpoint_url=endpoint_url, 
            data=data, 
            api_token=api_token, 
            request_id=request_id, 
            endpoint_id=endpoint_id
        )
        
        # Process response - normalize if needed
        return [self.normalize_embedding(emb) for emb in response]

    def normalize_embedding(self, embedding):
        """Normalize embedding to unit length"""
        # Convert to numpy array if not already
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Compute L2 norm
        norm = np.linalg.norm(embedding)
        
        # Normalize to unit length
        if norm > 0:
            normalized = embedding / norm
        else:
            normalized = embedding
        
        # Convert back to list for JSON serialization
        return normalized.tolist()

    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        # Convert to numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Normalize if not already normalized
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 > 0:
            emb1 = emb1 / norm1
        if norm2 > 0:
            emb2 = emb2 / norm2
        
        # Calculate cosine similarity
        return np.dot(emb1, emb2)
        
    def create_remote_text_embedding_endpoint_handler(self, endpoint_url=None, api_key=None, endpoint_id=None):
        """Create an endpoint handler for HF TEI remote inference"""
        async def endpoint_handler(text, **kwargs):
            """Handle requests to HF TEI endpoint"""
            try:
                # Extract request ID if provided
                request_id = kwargs.get("request_id")
                
                # Use specific endpoint ID from kwargs or from constructor
                current_endpoint_id = kwargs.get("endpoint_id", endpoint_id)
                
                # If no specific model endpoint provided, use standard API
                if not endpoint_url:
                    # Extract model from kwargs or use default
                    model = kwargs.get("model", self.default_model)
                    
                    # Create endpoint URL
                    model_endpoint = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
                else:
                    model_endpoint = endpoint_url
                    model = kwargs.get("model", self.default_model)
                
                # Handle batch or single input
                if isinstance(text, list):
                    # Batch mode
                    embeddings = self.batch_embed(
                        model_id=model, 
                        texts=text, 
                        api_token=api_key,
                        request_id=request_id,
                        endpoint_id=current_endpoint_id
                    )
                    
                    # Normalize if requested
                    if kwargs.get("normalize", True):
                        embeddings = [self.normalize_embedding(emb) for emb in embeddings]
                    
                    # Create response with metadata
                    response = {
                        "embeddings": embeddings, 
                        "implementation_type": "(REAL)",
                        "model": model
                    }
                    
                    # Add request ID if available
                    if request_id:
                        response["request_id"] = request_id
                        
                    return response
                else:
                    # Single text mode
                    embedding = self.generate_embedding(
                        model_id=model, 
                        text=text, 
                        api_token=api_key,
                        request_id=request_id,
                        endpoint_id=current_endpoint_id
                    )
                    
                    # Normalize if requested
                    if kwargs.get("normalize", True):
                        embedding = self.normalize_embedding(embedding)
                    
                    # Create response with metadata
                    response = {
                        "embedding": embedding, 
                        "implementation_type": "(REAL)",
                        "model": model
                    }
                    
                    # Add request ID if available
                    if request_id:
                        response["request_id"] = request_id
                        
                    return response
            except Exception as e:
                print(f"Error calling HF TEI endpoint: {e}")
                return {"error": str(e), "implementation_type": "(ERROR)"}
        
        return endpoint_handler
        
    def test_hf_tei_endpoint(self, endpoint_url=None, api_token=None, model_id=None, endpoint_id=None, request_id=None):
        """Test the HF TEI endpoint"""
        if not model_id:
            model_id = self.default_model
            
        if not endpoint_url:
            endpoint_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        
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
            # Test embedding generation with proper request tracking
            response = self.generate_embedding(
                model_id=model_id, 
                text="Testing the Hugging Face TEI API.",
                api_token=api_token,
                request_id=request_id,
                endpoint_id=endpoint_id
            )
            
            # Verify we got a valid response
            if isinstance(response, list) and len(response) > 0:
                result = {
                    "success": True,
                    "message": "TEI API test successful",
                    "model": model_id,
                    "implementation_type": "(REAL)",
                    "request_id": request_id
                }
                return result
            else:
                result = {
                    "success": False,
                    "message": "TEI API test failed: unexpected response format",
                    "implementation_type": "(ERROR)",
                    "request_id": request_id
                }
                return result
        except Exception as e:
            error_message = str(e)
            print(f"Error testing HF TEI endpoint: {error_message}")
            result = {
                "success": False,
                "message": f"TEI API test failed: {error_message}",
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
