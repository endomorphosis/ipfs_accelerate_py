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
import logging
from queue import Queue
from typing import Dict, List, Any, Optional, Iterator, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ollama_api")

class ollama_clean:
    """Ollama API backend implementation for local LLM deployments"""
    
    def __init__(self, resources: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """Initialize the API backend"""
        self.resources = resources or {}
        self.metadata = metadata or {}
        
        # Base URL for Ollama API (default: http://localhost:11434)
        self.api_base = self.metadata.get("ollama_api_base", os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        self.api_endpoint = f"{self.api_base}/api/chat"
        
        # Default model to use
        self.default_model = "llama2"
        
        # Request queue and concurrency control
        self.max_queue_size = 100
        self.queue = Queue(maxsize=self.max_queue_size)
        self.max_concurrent = 3
        self.active_count = 0
        self.queue_lock = threading.Lock()
        
        # Circuit breaker pattern
        self.circuit_open = False
        self.error_threshold = 5
        self.error_count = 0
        self.circuit_reset_time = 30  # seconds
        
        # Start queue processor thread
        self.queue_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_thread.start()
    
    def get_api_key(self, metadata: Dict[str, Any]) -> str:
        """Get API key from metadata or environment"""
        return metadata.get("ollama_api_key") or os.environ.get("OLLAMA_API_KEY", "")
    
    def get_default_model(self) -> str:
        """Get the default model name"""
        return self.default_model
    
    def is_compatible_model(self, model: str) -> bool:
        """Check if the model is compatible with this backend"""
        # Ollama supports many model types, but we'll check for some common ones
        return (
            model.startswith("llama") or 
            model.startswith("mistral") or
            model.startswith("gemma") or
            "ollama:" in model.lower()
        )
    
    def create_endpoint_handler(self):
        """Create an endpoint handler function"""
        def handler(data: Dict[str, Any]) -> Dict[str, Any]:
            return self.make_post_request(data)
        return handler
    
    def test_endpoint(self) -> bool:
        """Test the API endpoint"""
        try:
            # Make a simple test request
            test_request = {
                "model": self.get_default_model(),
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                self.api_endpoint,
                json=test_request,
                timeout=5
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Endpoint test failed: {e}")
            return False
    
    def make_post_request(self, data: Dict[str, Any], api_key: Optional[str] = None, 
                          timeout: int = 30) -> Dict[str, Any]:
        """Make a POST request to the Ollama API"""
        # Ollama doesn't actually use API keys, but we keep this parameter for interface consistency
        
        # Prepare the URL based on endpoint type
        url = self.api_endpoint
        
        # Add the request to the queue and wait for result
        future = Future()
        request_data = {
            "data": data,
            "future": future,
            "timeout": timeout,
            "is_stream": False
        }
        
        # Add to queue
        try:
            self.queue.put(request_data, block=True, timeout=timeout)
        except Queue.Full:
            raise Exception("Request queue is full")
        
        # Wait for result
        try:
            result = future.result(timeout=timeout * 1.5)  # Allow extra time for processing
            return result
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def make_stream_request(self, data: Dict[str, Any], api_key: Optional[str] = None,
                           timeout: int = 30) -> Iterator[Dict[str, Any]]:
        """Make a streaming request to the Ollama API"""
        # Ensure stream parameter is set
        data = {**data, "stream": True}
        
        # Make direct request for streaming (not using queue)
        try:
            response = requests.post(
                self.api_endpoint,
                json=data,
                stream=True,
                timeout=timeout
            )
            
            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                raise Exception(f"API stream request failed: {error_data}")
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        yield chunk
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse stream data: {line}")
                        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Stream request failed: {str(e)}")
    
    def _process_queue(self):
        """Process the request queue"""
        while True:
            try:
                # Get next request from queue
                request = self.queue.get(block=True)
                
                # Check if circuit is open
                if self.circuit_open:
                    request["future"].set_exception(Exception("Circuit breaker open")
                    self.queue.task_done()
                    continue
                
                # Process the request
                with self.queue_lock:
                    self.active_count += 1
                
                try:
                    # Make the actual request
                    url = self.api_endpoint
                    
                    response = requests.post(
                        url,
                        json=request["data"],
                        timeout=request["timeout"]
                    )
                    
                    # Check response status
                    if response.status_code != 200:
                        error_data = response.json() if response.content else {}
                        raise Exception(f"API request failed: {error_data}")
                    
                    # Parse the response
                    result = response.json()
                    
                    # Reset error count on success
                    self.error_count = 0
                    
                    # Set the result
                    request["future"].set_result(result)
                    
                except Exception as e:
                    # Update error count
                    self.error_count += 1
                    
                    # Check circuit breaker
                    if self.error_count >= self.error_threshold:
                        self.circuit_open = True
                        logger.warning(f"Circuit breaker opened after {self.error_count} errors")
                        
                        # Schedule circuit reset
                        threading.Timer(self.circuit_reset_time, self._reset_circuit).start()
                    
                    # Set the exception
                    request["future"].set_exception(e)
                
                finally:
                    # Update counters
                    with self.queue_lock:
                        self.active_count -= 1
                    
                    # Mark task as done
                    self.queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                time.sleep(1)  # Prevent tight loop in case of errors
    
    def _reset_circuit(self):
        """Reset the circuit breaker"""
        self.circuit_open = False
        self.error_count = 0
        logger.info("Circuit breaker reset")
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             max_tokens: Optional[int] = None, temperature: Optional[float] = None,
             top_p: Optional[float] = None) -> Dict[str, Any]:
        """Generate a chat response using the Ollama API"""
        model = model or self.get_default_model()
        
        request_data = {
            "model": model,
            "messages": messages,
        }
        
        # Add optional parameters if provided
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        if temperature is not None:
            request_data["temperature"] = temperature
        if top_p is not None:
            request_data["top_p"] = top_p
        
        response = self.make_post_request(request_data)
        
        # Format response to match the expected structure
        return {
            "id": response.get("id", str(uuid.uuid4(),
            "model": model,
            "object": "chat.completion",
            "created": int(time.time(),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.get("message", {}).get("content", "")
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": response.get("usage", {})
        }
    
    def stream_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                    max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                    top_p: Optional[float] = None) -> Iterator[Dict[str, Any]]:
        """Generate a streaming chat response using the Ollama API"""
        model = model or self.get_default_model()
        
        request_data = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        # Add optional parameters if provided
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        if temperature is not None:
            request_data["temperature"] = temperature
        if top_p is not None:
            request_data["top_p"] = top_p
        
        # Ollama stream format is different from OpenAI
        # We'll convert it to match the expected format
        for chunk in self.make_stream_request(request_data):
            delta_content = chunk.get("message", {}).get("content", "")
            
            yield {
                "id": chunk.get("id", str(uuid.uuid4(),
                "model": model,
                "object": "chat.completion.chunk",
                "created": int(time.time(),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": delta_content
                        },
                        "finish_reason": chunk.get("done", False) and "stop" or None
                    }
                ]
            }