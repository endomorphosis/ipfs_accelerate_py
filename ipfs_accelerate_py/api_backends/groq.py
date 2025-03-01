import os
import asyncio
import requests
import json
import logging
import time
import datetime
import hashlib
import functools
import threading
from typing import Dict, List, Any, Optional, Iterator, Union, Callable
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("groq_api")

# Global thread-local storage for API usage metrics
_thread_local = threading.local()

# Try to import sseclient for streaming, but don't fail if not available
try:
    import sseclient
    SSECLIENT_AVAILABLE = True
except ImportError:
    SSECLIENT_AVAILABLE = False
    logger.warning("sseclient package not found. Streaming will not be available. Install with: pip install sseclient-py")
    
# Try to import tiktoken for token counting if available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.debug("tiktoken package not found. Client-side token counting will not be available.")

# https://console.groq.com/docs/models
# Models categorized by type/capability

# Chat/Text completion models
CHAT_MODELS = {
    "gemma2-9b-it": {
        "context_window": 8192,
        "description": "Google's Gemma 2 9B instruction-tuned model"
    },
    "llama-3.3-70b-versatile": {
        "context_window": 128000,
        "description": "Meta's LLaMA 3.3 70B versatile model with long context"
    },
    "llama-3.1-8b-instant": {
        "context_window": 128000,
        "description": "Meta's LLaMA 3.1 8B optimized for fast responses"
    },
    "llama-guard-3-8b": {
        "context_window": 8192,
        "description": "Meta's LLaMA Guard 3 8B for content moderation"
    },
    "llama3-70b-8192": {
        "context_window": 8192,
        "description": "Meta's LLaMA 3 70B model"
    },
    "llama3-8b-8192": {
        "context_window": 8192,
        "description": "Meta's LLaMA 3 8B model"
    },
    "mixtral-8x7b-32768": {
        "context_window": 32768,
        "description": "Mixtral 8x7B with 32K context window"
    },
    "qwen-2.5-32b": {
        "context_window": 128000,
        "description": "Qwen 2.5 32B model with long context"
    },
    "qwen-2.5-coder-32b": {
        "context_window": 128000,
        "description": "Qwen 2.5 32B coder-specialized model"
    },
    "mistral-saba-24b": {
        "context_window": 32768,
        "description": "Mistral's Saba 24B model"
    },
    "deepseek-r1-distill-qwen-32b": {
        "context_window": 128000,
        "description": "DeepSeek R1 distilled Qwen 32B model"
    },
    "deepseek-r1-distill-llama-70b-specdec": {
        "context_window": 128000,
        "description": "DeepSeek R1 distilled LLaMA 70B model with speculative decoding"
    },
    "deepseek-r1-distill-llama-70b": {
        "context_window": 128000,
        "description": "DeepSeek R1 distilled LLaMA 70B model"
    },
    "llama-3.3-70b-specdec": {
        "context_window": 8192,
        "description": "Meta's LLaMA 3.3 70B with speculative decoding"
    },
    "llama-3.2-1b-preview": {
        "context_window": 128000,
        "description": "Meta's LLaMA 3.2 1B preview model"
    },
    "llama-3.2-3b-preview": {
        "context_window": 128000,
        "description": "Meta's LLaMA 3.2 3B preview model"
    },
}

# Vision-capable models
VISION_MODELS = {
    "llama-3.2-11b-vision-preview": {
        "context_window": 128000,
        "description": "Meta's LLaMA 3.2 11B vision model"
    },
    "llama-3.2-90b-vision-preview": {
        "context_window": 128000,
        "description": "Meta's LLaMA 3.2 90B vision model"
    },
}

# Audio/Speech models (not usable with chat completions endpoint)
AUDIO_MODELS = {
    "distil-whisper-large-v3-en": {
        "context_window": None,
        "description": "Distilled Whisper Large V3 model for English",
        "chat_compatible": False
    },
    "whisper-large-v3": {
        "context_window": None,
        "description": "OpenAI's Whisper Large V3 model",
        "chat_compatible": False
    },
    "whisper-large-v3-turbo": {
        "context_window": None,
        "description": "OpenAI's Whisper Large V3 Turbo model",
        "chat_compatible": False
    },
}

# Combined dictionary of all models for easy lookup
ALL_MODELS = {}
ALL_MODELS.update(CHAT_MODELS)
ALL_MODELS.update(VISION_MODELS)
ALL_MODELS.update(AUDIO_MODELS)

# For backward compatibility
PRODUCTION_MODELS = ALL_MODELS.copy()
PREVIEW_MODELS = {}


# API usage metrics tracking class
class APIUsageTracker:
    """Track API usage across requests"""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_requests = 0
        self.total_cost = 0.0
        self.request_history = []
        self.started_at = datetime.datetime.now()
        
    def add_request(self, model: str, prompt_tokens: int, completion_tokens: int, 
                    duration: float, status: str = "success", error: str = None):
        """Add a request to the tracker"""
        timestamp = datetime.datetime.now()
        
        # Calculate approximate cost based on model and tokens
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        
        request_data = {
            "timestamp": timestamp.isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_seconds": duration,
            "status": status,
            "cost_estimate": cost
        }
        
        if error:
            request_data["error"] = error
            
        # Update totals
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_requests += 1
        self.total_cost += cost
        
        # Add to history (keep last 100 requests)
        self.request_history.append(request_data)
        if len(self.request_history) > 100:
            self.request_history.pop(0)
            
        return request_data
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate approximate cost based on model and token counts"""
        # Approximate rates from Groq pricing (subject to change)
        rates = {
            "llama3-8b-8192": {"prompt": 0.0000002, "completion": 0.0000002},  # $0.20 / M tokens
            "llama3-70b-8192": {"prompt": 0.0000006, "completion": 0.0000006},  # $0.60 / M tokens
            "mixtral-8x7b-32768": {"prompt": 0.0000006, "completion": 0.0000006},  # $0.60 / M tokens
            "gemma2-9b-it": {"prompt": 0.0000002, "completion": 0.0000002},  # $0.20 / M tokens
        }
        
        # Use default rate if model not found
        model_rates = rates.get(model, {"prompt": 0.0000003, "completion": 0.0000003})
        
        # Calculate cost
        cost = (prompt_tokens * model_rates["prompt"]) + (completion_tokens * model_rates["completion"])
        return cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of API usage"""
        now = datetime.datetime.now()
        runtime = (now - self.started_at).total_seconds()
        
        return {
            "started_at": self.started_at.isoformat(),
            "current_time": now.isoformat(),
            "runtime_seconds": runtime,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.total_cost, 6),
            "tokens_per_second": round(self.total_tokens / max(1, runtime), 2),
            "recent_requests": self.request_history[-10:] if self.request_history else []
        }
    
    def reset(self):
        """Reset the tracker"""
        self.total_tokens = 0
        self.total_requests = 0
        self.total_cost = 0.0
        self.request_history = []
        self.started_at = datetime.datetime.now()

# Function decorator for monitoring API calls
def monitor_api_call(func):
    """Decorator to monitor API calls"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            end_time = time.time()
            
            # Extract tokens info if available
            model = kwargs.get("model_name", "unknown")
            if isinstance(result, dict) and "usage" in result:
                usage = result["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                # Record usage
                if hasattr(self, "usage_tracker"):
                    self.usage_tracker.add_request(
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        duration=end_time - start_time
                    )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            
            # Record error
            if hasattr(self, "usage_tracker"):
                self.usage_tracker.add_request(
                    model=kwargs.get("model_name", "unknown"),
                    prompt_tokens=0,
                    completion_tokens=0,
                    duration=end_time - start_time,
                    status="error",
                    error=str(e)
                )
            
            raise
            
    return wrapper

# Client-side token estimator
def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Estimate the token count for a given text and model"""
    if not TIKTOKEN_AVAILABLE:
        # Rough character-based estimate if tiktoken not available
        return len(text) // 4  # Approximately 4 characters per token
    
    try:
        # Map Groq models to OpenAI encoding
        model_to_encoding = {
            "llama3-8b-8192": "cl100k_base",
            "llama3-70b-8192": "cl100k_base",
            "mixtral-8x7b-32768": "cl100k_base",
            "gemma2-9b-it": "cl100k_base",
            "default": "cl100k_base"
        }
        
        encoding_name = model_to_encoding.get(model, model_to_encoding["default"])
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fall back to character-based estimate
        return len(text) // 4

class groq:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources or {}
        self.metadata = metadata or {}
        # Register method references
        self.create_remote_groq_endpoint_handler = self.create_remote_groq_endpoint_handler
        self.request_groq_endpoint = self.request_groq_endpoint
        self.test_groq_endpoint = self.test_groq_endpoint
        self.make_post_request_groq = self.make_post_request_groq
        self.make_stream_request_groq = self.make_stream_request_groq
        self.create_groq_endpoint_handler = self.create_groq_endpoint_handler
        self.chat = self.chat
        self.stream_chat = self.stream_chat
        self.is_compatible_model = self.is_compatible_model
        self.count_tokens = self.count_tokens
        self.get_usage_stats = self.get_usage_stats
        self.reset_usage_stats = self.reset_usage_stats
        self.init = self.init
        self.__test__ = self.__test__
        
        # Add endpoints tracking
        self.endpoints = {}
        self.endpoint_status = {}
        
        # Add queue for managing requests
        self.request_queue = asyncio.Queue(64)
        
        # Extract API key from metadata or environment
        self.api_key = self.metadata.get("groq_api_key", os.environ.get("GROQ_API_KEY", ""))
        if not self.api_key:
            logger.warning("No Groq API key provided. Set GROQ_API_KEY environment variable or provide in metadata.")
            
        # Set default endpoint URL
        self.default_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        
        # Retry and backoff settings
        self.max_retries = 5
        self.initial_retry_delay = 1
        self.backoff_factor = 2
        self.max_retry_delay = 60  # Maximum delay in seconds
        
        # Request queue settings
        self.queue_enabled = True
        self.queue_size = 100
        self.queue_processing = False
        self.current_requests = 0
        self.max_concurrent_requests = 5
        self.request_queue = []
        self.queue_lock = threading.RLock()
        
        # API usage tracking
        self.usage_tracker = APIUsageTracker()
        
        # API version and user agent
        self.api_version = self.metadata.get("groq_api_version", "2023-03-01-preview")
        self.user_agent = self.metadata.get("user_agent", "IPFS-Accelerate/1.0")
        
        return None
        
    def get_usage_stats(self):
        """Get API usage statistics"""
        return self.usage_tracker.get_summary()
        
    def reset_usage_stats(self):
        """Reset API usage statistics"""
        self.usage_tracker.reset()
        return {"status": "reset", "message": "Usage statistics have been reset"}
        
    def _process_queue(self):
        """Process requests in the queue in FIFO order"""
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
                    endpoint_url = request_info["endpoint_url"]
                    data = request_info["data"]
                    api_key = request_info["api_key"]
                    request_id = request_info["request_id"]
                    future = request_info["future"]
                    
                    # Make the request (without queueing again)
                    # Save original queue_enabled value
                    original_queue_enabled = self.queue_enabled
                    self.queue_enabled = False  # Disable queueing to prevent recursion
                    
                    try:
                        # Make the request
                        result = self.make_post_request_groq(
                            endpoint_url=endpoint_url,
                            data=data,
                            api_key=api_key,
                            request_id=request_id
                        )
                        
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
        
    def count_tokens(self, text, model=None):
        """Count tokens for the given text using the specified model"""
        return {
            "text": text,
            "estimated_token_count": estimate_tokens(text, model),
            "estimation_method": "tiktoken" if TIKTOKEN_AVAILABLE else "character_based"
        }
    
    def init(self, endpoint_url=None, api_key=None, model_name=None):
        """Initialize a connection to a Groq API endpoint
        
        Args:
            endpoint_url: The URL of the API endpoint (defaults to Groq's API)
            api_key: API key for authentication
            model_name: Name of the model to use
            
        Returns:
            tuple: (endpoint_url, api_key, handler, queue, batch_size)
        """
        if not endpoint_url:
            endpoint_url = "https://api.groq.com/openai/v1/chat/completions"
            
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not provided and not found in environment variables")
        
        # Create the endpoint handler
        endpoint_handler = self.create_remote_groq_endpoint_handler(endpoint_url, api_key, model_name)
        
        # Register the endpoint
        if model_name not in self.endpoints:
            self.endpoints[model_name] = []
        
        if endpoint_url not in self.endpoints[model_name]:
            self.endpoints[model_name].append(endpoint_url)
            # Get context window from model definitions
            context_window = 0
            if model_name in PRODUCTION_MODELS:
                context_window = PRODUCTION_MODELS[model_name]["context_window"] or 8192
            elif model_name in PREVIEW_MODELS:
                context_window = PREVIEW_MODELS[model_name]["context_window"] or 8192
            else:
                context_window = 8192  # Default
                
            self.endpoint_status[endpoint_url] = context_window // 256  # Rough estimate of batch size
        
        return endpoint_url, api_key, endpoint_handler, self.request_queue, self.endpoint_status[endpoint_url]
    
    def __test__(self, endpoint_url, endpoint_handler, endpoint_label, api_key=None):
        """Test the Groq API endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            endpoint_handler: The handler function
            endpoint_label: Label for the endpoint
            api_key: API key for authentication
            
        Returns:
            bool: True if test passes, False otherwise
        """
        test_prompt = "Complete this sentence: The quick brown fox"
        try:
            result = endpoint_handler(test_prompt)
            if result is not None:
                print(f"Groq API test passed for {endpoint_label}")
                return True
            else:
                print(f"Groq API test failed for {endpoint_label}: No result")
                return False
        except Exception as e:
            print(f"Groq API test failed for {endpoint_label}: {e}")
            return False
    
    def make_post_request_groq(self, endpoint_url, data, api_key=None, request_id=None):
        """Make a POST request to the Groq API with exponential backoff and queue
        
        Args:
            endpoint_url: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication
            request_id: Optional unique ID for the request (for tracking)
            
        Returns:
            dict: Response from the endpoint
        
        Raises:
            ValueError: If API key is not provided
            requests.exceptions.RequestException: If request fails
        """
        if not api_key:
            api_key = self.api_key
            
        if not api_key:
            raise ValueError("No Groq API key provided for authentication")
        
        # If queue is enabled and we're at capacity, add to queue
        if self.queue_enabled:
            with self.queue_lock:
                if self.current_requests >= self.max_concurrent_requests:
                    # Create a future to store the result
                    result_future = {"result": None, "error": None, "completed": False}
                    
                    # Add to queue with all necessary info to process later
                    request_info = {
                        "endpoint_url": endpoint_url,
                        "data": data,
                        "api_key": api_key,
                        "request_id": request_id,
                        "future": result_future
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
            
        # Generate request ID if not provided
        if not request_id:
            request_id = f"req_{int(time.time())}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"
            
        # Prepare headers with enhanced tracking
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Request-ID": request_id,
            "User-Agent": self.user_agent
        }
        
        # Add API version if available
        if hasattr(self, "api_version") and self.api_version:
            headers["Groq-Version"] = self.api_version
        
        # Use exponential backoff retry mechanism
        retries = 0
        retry_delay = self.initial_retry_delay
        
        while retries < self.max_retries:
            try:
                response = requests.post(endpoint_url, headers=headers, json=data, timeout=60)
                
                # Handle HTTP errors
                if response.status_code == 401:
                    raise ValueError(f"Authentication error (401): {response.json().get('error', {}).get('message', 'Invalid API key')}")
                elif response.status_code == 429:
                    # Get retry-after header if available
                    retry_after = int(response.headers.get("retry-after", retry_delay))
                    
                    # Use the larger of the suggested retry time or our calculated backoff
                    wait_time = max(retry_after, retry_delay)
                    logger.warning(f"Rate limit exceeded (429). Waiting {wait_time} seconds before retry (attempt {retries+1}/{self.max_retries}).")
                    
                    time.sleep(wait_time)
                    retries += 1
                    
                    # Calculate next backoff duration with exponential increase
                    retry_delay = min(retry_delay * self.backoff_factor, self.max_retry_delay)
                    continue
                    
                elif response.status_code == 404:
                    raise ValueError(f"Resource not found (404): {response.json().get('error', {}).get('message', 'Model or endpoint not found')}")
                elif response.status_code >= 400:
                    response_json = response.json() if response.headers.get("content-type") == "application/json" else {"error": {"message": response.text}}
                    error_msg = response_json.get("error", {}).get("message", f"HTTP error {response.status_code}")
                    
                    # For server errors (5xx), we should retry
                    if response.status_code >= 500 and retries < self.max_retries - 1:
                        logger.warning(f"Server error ({response.status_code}): {error_msg}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retries += 1
                        retry_delay = min(retry_delay * self.backoff_factor, self.max_retry_delay)
                        continue
                    
                    raise ValueError(f"API error ({response.status_code}): {error_msg}")
                
                # Parse JSON response
                result = response.json()
                
                # Decrement counter if queue enabled
                if self.queue_enabled:
                    with self.queue_lock:
                        self.current_requests = max(0, self.current_requests - 1)
                
                return result
                
            except requests.exceptions.RequestException as e:
                if retries < self.max_retries - 1:
                    logger.warning(f"Request failed: {str(e)}. Retrying in {retry_delay} seconds (attempt {retries+1}/{self.max_retries})...")
                    time.sleep(retry_delay)
                    retries += 1
                    retry_delay = min(retry_delay * self.backoff_factor, self.max_retry_delay)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {str(e)}")
                    
                    # Decrement counter if queue enabled
                    if self.queue_enabled:
                        with self.queue_lock:
                            self.current_requests = max(0, self.current_requests - 1)
                    
                    raise
            
            except Exception as e:
                # Decrement counter if queue enabled for any other exceptions
                if self.queue_enabled:
                    with self.queue_lock:
                        self.current_requests = max(0, self.current_requests - 1)
                raise
                    
        # Decrement counter if we somehow exit the loop without returning or raising
        if self.queue_enabled:
            with self.queue_lock:
                self.current_requests = max(0, self.current_requests - 1)
                
        # This should never be reached due to the raise in the exception handler
        return None
        
    def make_stream_request_groq(self, endpoint_url, data, api_key=None, request_id=None):
        """Make a streaming request to the Groq API with exponential backoff
        
        Args:
            endpoint_url: URL of the endpoint
            data: Data to send in the request
            api_key: API key for authentication
            request_id: Optional unique ID for the request (for tracking)
            
        Returns:
            Iterator: Stream of response chunks
            
        Raises:
            ValueError: If API key is not provided or sseclient is not available
            requests.exceptions.RequestException: If request fails
        """
        if not SSECLIENT_AVAILABLE:
            raise ValueError("sseclient package is required for streaming. Install with: pip install sseclient-py")
            
        if not api_key:
            api_key = self.api_key
            
        if not api_key:
            raise ValueError("No Groq API key provided for authentication")
        
        # Generate request ID if not provided
        if not request_id:
            request_id = f"stream_{int(time.time())}_{hashlib.md5(str(data).encode()).hexdigest()[:8]}"
            
        # Prepare headers with enhanced tracking
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
            "X-Request-ID": request_id,
            "User-Agent": self.user_agent
        }
        
        # Add API version if available
        if hasattr(self, "api_version") and self.api_version:
            headers["Groq-Version"] = self.api_version
        
        # Ensure streaming is enabled in the request
        data["stream"] = True
        
        # Use exponential backoff retry mechanism
        retries = 0
        retry_delay = self.initial_retry_delay
        
        while retries < self.max_retries:
            try:
                response = requests.post(
                    endpoint_url,
                    headers=headers,
                    json=data,
                    stream=True,
                    timeout=90
                )
                
                # Handle HTTP errors
                if response.status_code == 401:
                    raise ValueError(f"Authentication error (401): Invalid API key")
                elif response.status_code == 429:
                    # Get retry-after header if available
                    retry_after = int(response.headers.get("retry-after", retry_delay))
                    
                    # Use the larger of the suggested retry time or our calculated backoff
                    wait_time = max(retry_after, retry_delay)
                    logger.warning(f"Rate limit exceeded (429). Waiting {wait_time} seconds before retry (attempt {retries+1}/{self.max_retries}).")
                    
                    time.sleep(wait_time)
                    retries += 1
                    
                    # Calculate next backoff duration with exponential increase
                    retry_delay = min(retry_delay * self.backoff_factor, self.max_retry_delay)
                    continue
                    
                elif response.status_code == 404:
                    # Try to read error message
                    try:
                        error_json = response.json()
                        error_msg = error_json.get("error", {}).get("message", "Model or endpoint not found")
                    except Exception:
                        error_msg = "Model or endpoint not found"
                    
                    raise ValueError(f"Resource not found (404): {error_msg}")
                    
                elif response.status_code != 200:
                    # Try to get error details
                    try:
                        if response.headers.get("content-type") == "application/json":
                            error_json = response.json()
                            error_msg = error_json.get("error", {}).get("message", f"HTTP error {response.status_code}")
                        else:
                            error_msg = response.text
                    except Exception:
                        error_msg = f"HTTP error {response.status_code}"
                    
                    # For server errors (5xx), we should retry
                    if response.status_code >= 500 and retries < self.max_retries - 1:
                        logger.warning(f"Server error ({response.status_code}): {error_msg}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retries += 1
                        retry_delay = min(retry_delay * self.backoff_factor, self.max_retry_delay)
                        continue
                    
                    raise ValueError(f"API error ({response.status_code}): {error_msg}")
                
                # Create SSE client for parsing the stream
                client = sseclient.SSEClient(response)
                
                # Yield parsed events
                for event in client.events():
                    if event.data != "[DONE]":
                        try:
                            yield json.loads(event.data)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON from stream: {event.data}")
                
                # If we get here, the streaming completed successfully
                return
            
            except requests.exceptions.RequestException as e:
                if retries < self.max_retries - 1:
                    logger.warning(f"Streaming request failed: {str(e)}. Retrying in {retry_delay} seconds (attempt {retries+1}/{self.max_retries})...")
                    time.sleep(retry_delay)
                    retries += 1
                    retry_delay = min(retry_delay * self.backoff_factor, self.max_retry_delay)
                else:
                    logger.error(f"Streaming request failed after {self.max_retries} attempts: {str(e)}")
                    raise
        
        # This should never be reached due to the raise in the exception handler
        raise ValueError(f"Stream request failed after {self.max_retries} attempts with exponential backoff")
    
    def test_groq_endpoint(self, endpoint_url=None, api_key=None, model_name=None):
        """Test a Groq API endpoint
        
        Args:
            endpoint_url: URL of the endpoint to test
            api_key: API key for authentication
            model_name: Name of the model to use
            
        Returns:
            bool: True if test passes, False otherwise
        """
        if not endpoint_url:
            endpoint_url = "https://api.groq.com/openai/v1/chat/completions"
            
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not provided and not found in environment variables")
                
        if not model_name:
            model_name = "llama3-8b-8192"  # Default test model
            
        try:
            data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Complete this sentence: The quick brown fox"}
                ],
                "max_tokens": 20,
                "temperature": 0.7
            }
                
            result = self.make_post_request_groq(endpoint_url, data, api_key)
            
            if result and "choices" in result and len(result["choices"]) > 0:
                return True
            else:
                print(f"Test failed for endpoint {endpoint_url}: Invalid response format")
                return False
                
        except Exception as e:
            print(f"Test failed for endpoint {endpoint_url}: {e}")
            return False
    
    def request_groq_endpoint(self, model, endpoint=None, endpoint_type=None, batch=None):
        """Request a Groq endpoint
        
        Args:
            model: Name of the model
            endpoint: Specific endpoint URL (optional)
            endpoint_type: Type of endpoint (optional)
            batch: Batch size (optional)
            
        Returns:
            str: URL of the selected endpoint
        """
        if endpoint:
            return endpoint
            
        # Default Groq API endpoint
        default_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        
        # Check if the model is in the endpoints dictionary
        if model in self.endpoints and self.endpoints[model]:
            return self.endpoints[model][0]  # Return the first registered endpoint
        
        # Return the default endpoint
        return default_endpoint
    
    def create_groq_endpoint_handler(self):
        """Create a default endpoint handler for Groq
        
        Returns:
            function: A handler for the default endpoint
        """
        api_key = os.getenv("GROQ_API_KEY")
        endpoint_url = "https://api.groq.com/openai/v1/chat/completions"
        model_name = "llama3-8b-8192"  # Default model
        
        return self.create_remote_groq_endpoint_handler(endpoint_url, api_key, model_name)
        
    def is_compatible_model(self, model_name, endpoint_type="chat"):
        """Check if a model is compatible with Groq
        
        Args:
            model_name: Name of the model to check
            endpoint_type: Type of endpoint to check compatibility with (chat, audio, vision)
            
        Returns:
            bool: True if compatible, False otherwise
        """
        # Remove any provider prefix
        if "/" in model_name:
            _, model_name = model_name.split("/", 1)
        
        # Check if model exists at all
        if model_name not in ALL_MODELS:
            return False
            
        # For chat completions, ensure the model is chat compatible
        if endpoint_type == "chat":
            # Audio models aren't chat compatible
            if model_name in AUDIO_MODELS:
                return False
            # All other models are chat compatible
            return True
            
        # For vision endpoint, only vision models are compatible
        if endpoint_type == "vision":
            return model_name in VISION_MODELS
            
        # For audio endpoint, only audio models are compatible
        if endpoint_type == "audio":
            return model_name in AUDIO_MODELS
            
        # Default to checking if model exists in any category
        return model_name in ALL_MODELS
        
    def list_models(self, category=None):
        """List available models, optionally filtered by category
        
        Args:
            category: Optional category to filter by (chat, vision, audio, all)
            
        Returns:
            list: List of models with their details
        """
        if category == "chat":
            models_dict = CHAT_MODELS
        elif category == "vision":
            models_dict = VISION_MODELS
        elif category == "audio":
            models_dict = AUDIO_MODELS
        else:
            models_dict = ALL_MODELS
            
        return [
            {
                "id": model_id,
                "context_window": info.get("context_window"),
                "description": info.get("description", ""),
                "category": "chat" if model_id in CHAT_MODELS else 
                            "vision" if model_id in VISION_MODELS else
                            "audio" if model_id in AUDIO_MODELS else "unknown"
            }
            for model_id, info in models_dict.items()
        ]
    
    def chat(self, model_name=None, messages=None, temperature=0.7, max_tokens=1024, top_p=0.95, top_k=None, 
              frequency_penalty=0.0, presence_penalty=0.0, response_format=None, request_id=None,
              logit_bias=None, seed=None, tools=None):
        """Generate a response using the Groq chat API
        
        Args:
            model_name: Name of the model to use
            messages: List of message objects with role and content
            temperature: Sampling temperature (0.0 to 1.0). Higher values make output more random, lower values more deterministic.
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0). Controls diversity via nucleus sampling.
            top_k: Integer that controls diversity by limiting to top k tokens (optional)
            frequency_penalty: Float (-2.0 to 2.0) that penalizes frequent tokens
            presence_penalty: Float (-2.0 to 2.0) that penalizes repeated tokens
            response_format: Dictionary specifying response format options (e.g., {"type": "json_object"})
            request_id: Optional request ID for tracking
            logit_bias: Dictionary mapping token IDs to bias values
            seed: Integer seed for deterministic generation
            tools: List of tool definitions for function calling
            
        Returns:
            dict: Response with generated text
            
        Raises:
            ValueError: If messages are not provided or model is not found
        """
        if not messages:
            raise ValueError("Messages are required for chat completion")
            
        # Default model if not specified
        if not model_name:
            model_name = "llama3-8b-8192"
            
        # Verify model compatibility with chat endpoint
        if not self.is_compatible_model(model_name, "chat"):
            # If model is in our database but not chat compatible
            if model_name in ALL_MODELS:
                model_info = ALL_MODELS[model_name]
                if model_info.get("chat_compatible") is False:
                    raise ValueError(f"Model '{model_name}' is not compatible with chat completions. " +
                                    f"It appears to be a {next((cat for cat, models in [('audio', AUDIO_MODELS), ('vision', VISION_MODELS)] if model_name in models), 'specialized')} model.")
            else:
                # Model not found at all
                similar_models = [m for m in CHAT_MODELS.keys() if model_name.lower() in m.lower()]
                suggestion = f" Did you mean {similar_models[0]}?" if similar_models else ""
                raise ValueError(f"Model '{model_name}' not found.{suggestion} Available chat models: {', '.join(list(CHAT_MODELS.keys())[:3])}...")
            
        # Get endpoint URL
        endpoint_url = self.request_groq_endpoint(model_name)
        
        # Prepare request data
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        # Add optional parameters if provided
        if top_k is not None:
            data["top_k"] = top_k
        
        if frequency_penalty != 0.0:
            data["frequency_penalty"] = frequency_penalty
            
        if presence_penalty != 0.0:
            data["presence_penalty"] = presence_penalty
            
        if response_format is not None:
            data["response_format"] = response_format
        
        # Add advanced parameters if provided
        if logit_bias is not None:
            data["logit_bias"] = logit_bias
            
        if seed is not None:
            data["seed"] = seed
            
        if tools is not None:
            data["tools"] = tools
        
        # Make the request with request ID
        try:
            response = self.make_post_request_groq(endpoint_url, data, self.api_key, request_id)
            
            # Format the response
            if response and "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                return {
                    "text": content,
                    "model": response.get("model", model_name),
                    "usage": response.get("usage", {}),
                    "id": response.get("id", ""),
                    "metadata": {
                        "finish_reason": response["choices"][0].get("finish_reason", ""),
                        "created": response.get("created", int(time.time()))
                    }
                }
            else:
                logger.error(f"Invalid response format from Groq API: {response}")
                raise ValueError("Invalid response format from Groq API")
                
        except Exception as e:
            logger.error(f"Error in Groq chat: {str(e)}")
            raise
    
    def stream_chat(self, model_name=None, messages=None, temperature=0.7, max_tokens=1024, top_p=0.95, top_k=None,
                   frequency_penalty=0.0, presence_penalty=0.0, response_format=None, request_id=None,
                   logit_bias=None, seed=None, tools=None):
        """Generate a streaming response using the Groq chat API
        
        Args:
            model_name: Name of the model to use
            messages: List of message objects with role and content
            temperature: Sampling temperature (0.0 to 1.0). Higher values make output more random, lower values more deterministic.
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0). Controls diversity via nucleus sampling.
            top_k: Integer that controls diversity by limiting to top k tokens (optional)
            frequency_penalty: Float (-2.0 to 2.0) that penalizes frequent tokens
            presence_penalty: Float (-2.0 to 2.0) that penalizes repeated tokens
            response_format: Dictionary specifying response format options (e.g., {"type": "json_object"})
            request_id: Optional request ID for tracking
            logit_bias: Dictionary mapping token IDs to bias values
            seed: Integer seed for deterministic generation
            tools: List of tool definitions for function calling
            
        Returns:
            Iterator: Stream of response chunks
            
        Raises:
            ValueError: If messages are not provided, model is not found, or sseclient is not available
        """
        if not SSECLIENT_AVAILABLE:
            logger.warning("Streaming not available (sseclient not installed). Falling back to regular chat.")
            # Fallback to regular chat with all parameters
            result = self.chat(
                model_name, messages, temperature, max_tokens, top_p, 
                top_k, frequency_penalty, presence_penalty, response_format,
                request_id, logit_bias, seed, tools
            )
            content = result.get("text", "")
            yield {
                "text": content,
                "accumulated_text": content,
                "metadata": {
                    "finish_reason": result.get("metadata", {}).get("finish_reason", "fallback"),
                    "model": result.get("model", model_name),
                    "request_id": request_id if request_id else "fallback_stream"
                }
            }
            return
            
        if not messages:
            raise ValueError("Messages are required for chat completion")
            
        # Default model if not specified
        if not model_name:
            model_name = "llama3-8b-8192"
            
        # Verify model compatibility with chat endpoint
        if not self.is_compatible_model(model_name, "chat"):
            # If model is in our database but not chat compatible
            if model_name in ALL_MODELS:
                model_info = ALL_MODELS[model_name]
                if model_info.get("chat_compatible") is False:
                    raise ValueError(f"Model '{model_name}' is not compatible with chat completions. " +
                                    f"It appears to be a {next((cat for cat, models in [('audio', AUDIO_MODELS), ('vision', VISION_MODELS)] if model_name in models), 'specialized')} model.")
            else:
                # Model not found at all
                similar_models = [m for m in CHAT_MODELS.keys() if model_name.lower() in m.lower()]
                suggestion = f" Did you mean {similar_models[0]}?" if similar_models else ""
                raise ValueError(f"Model '{model_name}' not found.{suggestion} Available chat models: {', '.join(list(CHAT_MODELS.keys())[:3])}...")
            
        # Get endpoint URL
        endpoint_url = self.request_groq_endpoint(model_name)
        
        # Prepare request data
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True
        }
        
        # Add optional parameters if provided
        if top_k is not None:
            data["top_k"] = top_k
        
        if frequency_penalty != 0.0:
            data["frequency_penalty"] = frequency_penalty
            
        if presence_penalty != 0.0:
            data["presence_penalty"] = presence_penalty
            
        if response_format is not None:
            data["response_format"] = response_format
            
        # Add advanced parameters if provided
        if logit_bias is not None:
            data["logit_bias"] = logit_bias
            
        if seed is not None:
            data["seed"] = seed
            
        if tools is not None:
            data["tools"] = tools
            
        # Add request ID for tracking
        if request_id is None:
            request_id = f"stream_{int(time.time())}_{hashlib.md5(str(messages).encode()).hexdigest()[:8]}"
        
        # Make the streaming request
        try:
            # Process the stream and yield formatted chunks
            accumulated_text = ""
            for chunk in self.make_stream_request_groq(endpoint_url, data, self.api_key, request_id):
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        content = delta["content"]
                        accumulated_text += content
                        yield {
                            "text": content,
                            "accumulated_text": accumulated_text,
                            "metadata": {
                                "finish_reason": chunk["choices"][0].get("finish_reason", None),
                                "model": model_name,
                                "request_id": request_id
                            }
                        }
                    elif "finish_reason" in chunk["choices"][0] and chunk["choices"][0]["finish_reason"]:
                        # Final chunk with finish reason
                        yield {
                            "text": "",
                            "accumulated_text": accumulated_text,
                            "metadata": {
                                "finish_reason": chunk["choices"][0]["finish_reason"],
                                "model": model_name,
                                "request_id": request_id
                            }
                        }
        
        except Exception as e:
            logger.error(f"Error in Groq stream chat: {str(e)}")
            raise
            
    def create_remote_groq_endpoint_handler(self, endpoint_url, api_key=None, model_name=None):
        """Create a handler for a remote Groq endpoint
        
        Args:
            endpoint_url: URL of the endpoint
            api_key: API key for authentication
            model_name: Name of the model to use
            
        Returns:
            function: Handler for the endpoint
        """
        if not api_key:
            api_key = self.api_key
            
        if not model_name:
            model_name = "llama3-8b-8192"
            
        def handler(prompt, parameters=None, endpoint_url=endpoint_url, api_key=api_key, model_name=model_name):
            try:
                default_params = {
                    "max_tokens": 128,
                    "temperature": 0.7,
                    "top_p": 0.95
                }
                
                # Update with any provided parameters
                if parameters:
                    default_params.update(parameters)
                
                # Prepare request data
                data = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                
                # Add parameters to request
                for key, value in default_params.items():
                    if key not in data:
                        data[key] = value
                
                # Make the request
                response = self.make_post_request_groq(endpoint_url, data, api_key)
                
                if response and "choices" in response and len(response["choices"]) > 0:
                    return response["choices"][0]["message"]["content"]
                
                return response
                
            except Exception as e:
                logger.error(f"Error in Groq handler: {e}")
                return None
        
        return handler