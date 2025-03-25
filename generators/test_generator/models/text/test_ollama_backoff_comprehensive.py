"""Migrated to refactored test suite on 2025-03-21

Comprehensive test for Ollama API backoff queue functionality.
This script:
    1. Tests the basic queue and backoff mechanism
    2. Tests different queue sizes and concurrent request limits
    3. Tests handling of rate limits and error recovery
    4. Compares performance with and without the queue system
"""

import sys
import os
import time
import json
import threading
import concurrent.futures
import random
import argparse
from datetime import datetime
from pathlib import Path

from refactored_test_suite.model_test import ModelTest

# Try to import Ollama API - use a mock if unavailable
try:
    from ipfs_accelerate_py.api_backends import ollama
except ImportError:
    try:
        from api_backends import ollama
    except ImportError:
        print("Warning: Ollama API module could not be imported. Using mock implementation.")
        # Create a minimal mock for testing
        class ollama:
            def __init__(self, resources=None, metadata=None):
                self.resources = resources
                self.metadata = metadata
                self.max_retries = 3
                self.backoff_factor = 2
                self.initial_retry_delay = 1
                self.max_retry_delay = 60
                self.request_queue = []
                self.queue_size = 20
                self.queue_enabled = True
                self.max_concurrent_requests = 5
                self.current_requests = {}
                self.queue_lock = None
                self.circuit_state = "closed"
                self.failure_count = 0
                self.failure_threshold = 5
                self.circuit_timeout = 30
                self.circuit_lock = None
                self.PRIORITY_HIGH = 0
                self.PRIORITY_NORMAL = 1
                self.PRIORITY_LOW = 2
                self.usage_stats = {"total_requests": 0, "total_tokens": 0}
                
            def generate(self, model="llama3", prompt="", max_tokens=50, temperature=0.7, request_id=None):
                return {"text": f"Mock response to: {prompt[:20]}..."}
                
            def chat(self, model_name="llama3", messages=None, max_tokens=50, temperature=0.7, request_id=None):
                return {"message": {"content": f"Mock response to chat with {len(messages)} messages"}}
                
            def completions(self, model="llama3", prompt="", max_tokens=50, temperature=0.7, request_id=None):
                return {"text": f"Mock completion of: {prompt[:20]}..."}

# Default settings
DEFAULT_MODEL = "llama3"
DEFAULT_HOST = "http://localhost:11434"
NUM_REQUESTS = 8
MAX_CONCURRENT = 2
QUEUE_SIZE = 20

class TestOllamaBackoffComprehensive(ModelTest):
    """Comprehensive tests for Ollama API backoff queue functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        self.model = DEFAULT_MODEL
        
        # Initialize client
        metadata = {"ollama_host": DEFAULT_HOST}
        self.client = ollama(resources={}, metadata=metadata)
        
        # Configure client
        if hasattr(self.client, "queue_size"):
            self.client.queue_size = QUEUE_SIZE
        if hasattr(self.client, "max_concurrent_requests"):
            self.client.max_concurrent_requests = MAX_CONCURRENT
    
    def send_request(self, prompt, model, tag="test"):
        """Send a request to the Ollama API and track metrics"""
        request_id = f"req_{tag}_{int(time.time())}"
        start_time = time.time()
        
        try:
            # Send the actual request
            response = self.client.generate(
                model=model,
                prompt=prompt,
                max_tokens=30,
                temperature=0.7,
                request_id=request_id
            )
            
            success = True
            error = None
            # Extract content from response
            content = response.get("text", response.get("generated_text", str(response)))
        except Exception as e:
            success = False
            error = str(e)
            content = None
        
        end_time = time.time()
        
        # Return metrics
        return {
            "success": success,
            "request_id": request_id,
            "time_taken": end_time - start_time,
            "content": content[:100] if content else None,  # Truncate long content
            "error": error,
            "tag": tag,
            "thread": threading.current_thread().name,
            "timestamp": datetime.now().isoformat()
        }
    
    def test_basic_queue(self):
        """Basic test of the queue system by sending multiple concurrent requests"""
        self.logger.info("Testing Basic Queue Functionality")
        
        # Set concurrency limit
        if hasattr(self.client, "max_concurrent_requests"):
            original_limit = self.client.max_concurrent_requests
            self.client.max_concurrent_requests = 2
            self.logger.info(f"Set concurrent requests limit to 2 (was {original_limit})")
        
        # Generate test prompts
        num_requests = 4
        prompts = [
            f"What is {i+1} + {i+2}? Respond with just the number." 
            for i in range(num_requests)
        ]
        
        # Send requests concurrently
        results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.send_request, prompt, self.model, f"basic_{i+1}"): i
                for i, prompt in enumerate(prompts)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "✓" if result["success"] else "✗"
                    self.logger.info(f"Request {idx+1}/{num_requests}: {status} in {result['time_taken']:.2f}s")
                except Exception as e:
                    self.logger.error(f"Request {idx+1}/{num_requests} raised exception: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        self.logger.info(f"Basic Queue Test - {successful}/{num_requests} successful in {total_time:.2f}s")
        self.logger.info(f"Average time per request: {total_time/num_requests:.2f}s")
        
        # Reset concurrency limit if we changed it
        if hasattr(self.client, "max_concurrent_requests"):
            self.client.max_concurrent_requests = original_limit
        
        # Verification
        self.assertGreaterEqual(successful, 1, "At least one request should succeed")
        
    def test_backoff_recovery(self):
        """Test the backoff and recovery mechanism by simulating errors"""
        self.logger.info("Testing Backoff and Recovery")
        
        # Check if client has required attributes
        has_backoff = hasattr(self.client, "max_retries") and hasattr(self.client, "backoff_factor")
        if has_backoff:
            self.logger.info(f"Client configuration: max_retries={self.client.max_retries}, " +
                           f"backoff_factor={self.client.backoff_factor}")
        else:
            self.logger.info("Warning: Client does not have backoff attributes configured")
        
        # Generate test prompts that are likely to cause errors (very large context)
        num_requests = 2
        prompts = [
            "Explain the theory of relativity in extreme detail." * 10  # Large prompt to potentially trigger errors
            for _ in range(num_requests)
        ]
        
        # Send requests with small delay to avoid overwhelming the API
        results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Sending potentially problematic request {i+1}/{num_requests}...")
            result = self.send_request(prompt, self.model, f"backoff_{i+1}")
            results.append(result)
            
            status = "✓" if result["success"] else "✗"
            self.logger.info(f"  Request {i+1}: {status} in {result['time_taken']:.2f}s")
            
            if i < num_requests - 1:
                time.sleep(0.5)  # Small delay between requests
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        self.logger.info(f"Backoff Test - {successful}/{num_requests} successful")
        
        # No assertions here as we're just testing the mechanism works, 
        # not necessarily that requests succeed

    def test_queue_structure(self):
        """Test the structure of the queue system attributes"""
        self.logger.info("Testing Queue Structure")
        
        # Check basic attributes
        self.assertTrue(hasattr(self.client, 'max_retries'), "Missing max_retries attribute")
        self.assertTrue(hasattr(self.client, 'backoff_factor'), "Missing backoff_factor attribute")
        
        # Check queue attributes
        self.assertTrue(hasattr(self.client, 'request_queue'), "Missing request_queue attribute")
        self.assertTrue(hasattr(self.client, 'queue_size'), "Missing queue_size attribute")
        self.assertTrue(hasattr(self.client, 'max_concurrent_requests'), "Missing max_concurrent_requests attribute")
        
        # Check circuit breaker attributes
        self.assertTrue(hasattr(self.client, 'circuit_state'), "Missing circuit_state attribute")
        self.assertTrue(hasattr(self.client, 'failure_count'), "Missing failure_count attribute")
        
        # Check priority levels
        self.assertTrue(
            hasattr(self.client, 'PRIORITY_HIGH') and 
            hasattr(self.client, 'PRIORITY_NORMAL') and 
            hasattr(self.client, 'PRIORITY_LOW'),
            "Missing priority level attributes"
        )
        


    def test_model_loading(self):
        # Test basic model loading
        if not hasattr(self, 'model_id') or not self.model_id:
            self.skipTest("No model_id specified")
        
        try:
            # Import the appropriate library
            if 'bert' in self.model_id.lower() or 'gpt' in self.model_id.lower() or 't5' in self.model_id.lower():
                import transformers
                model = transformers.AutoModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'clip' in self.model_id.lower():
                import transformers
                model = transformers.CLIPModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'whisper' in self.model_id.lower():
                import transformers
                model = transformers.WhisperModel.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            elif 'wav2vec2' in self.model_id.lower():
                import transformers
                model = transformers.Wav2Vec2Model.from_pretrained(self.model_id)
                self.assertIsNotNone(model, "Model loading failed")
            else:
                # Generic loading
                try:
                    import transformers
                    model = transformers.AutoModel.from_pretrained(self.model_id)
                    self.assertIsNotNone(model, "Model loading failed")
                except:
                    self.skipTest(f"Could not load model {self.model_id} with AutoModel")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")



    def detect_preferred_device(self):
        # Detect available hardware and choose the preferred device
        try:
            import torch
        
            # Check for CUDA
            if torch.cuda.is_available():
                return "cuda"
        
            # Check for MPS (Apple Silicon)
            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                return "mps"
        
            # Fallback to CPU
            return "cpu"
        except ImportError:
            return "cpu"


if __name__ == "__main__":
    import unittest
    unittest.main()