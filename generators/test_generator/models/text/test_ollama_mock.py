"""Migrated to refactored test suite on 2025-03-21

Mock test for the Ollama API implementation to verify parameter compatibility.
"""

import sys
import os
import time
import json
from datetime import datetime

from refactored_test_suite.model_test import ModelTest

# Import the Ollama API
try:
    from ipfs_accelerate_py.api_backends.ollama import ollama
except ImportError:
    try:
        from api_backends.ollama import ollama
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
                self.queue_size = 100
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

class MockOllama(ollama):
    """Mock Ollama API client that doesn't make actual HTTP requests"""
    
    def __init__(self, resources=None, metadata=None):
        """Initialize with mocked request handler"""
        super().__init__(resources, metadata)
        self.requests = []
        
    def make_post_request_ollama(self, endpoint_url, data, stream=False, request_id=None, priority=None):
        """Mock request handler that just records the request"""
        self.requests.append({
            "endpoint_url": endpoint_url,
            "data": data,
            "stream": stream,
            "request_id": request_id,
            "priority": priority,
            "timestamp": time.time()
        })
        
        # Return a mock response
        if stream:
            def mock_stream():
                for i in range(3):
                    yield {
                        "message": {"content": f"Mock response chunk {i+1}"},
                        "done": i == 2
                    }
            return mock_stream()
        else:
            return {
                "message": {"content": "This is a mock response from Ollama API"},
                "prompt_eval_count": 10,
                "eval_count": 20
            }

class TestOllamaMock(ModelTest):
    """Test class for Ollama API mock implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        self.client = MockOllama()
        
    def test_parameter_compatibility(self):
        """Test parameter compatibility with the test_api_backoff_queue.py script"""
        self.logger.info("Testing Parameter Compatibility")
        
        # Reset requests list
        self.client.requests = []
        
        # Test the chat method with model_name parameter
        test_message = "Test message for parameter compatibility"
        response = self.client.chat(
            model_name="llama3",
            messages=[{"role": "user", "content": test_message}],
            max_tokens=30,
            temperature=0.7,
            request_id="test_123"
        )
        
        # Verify the request
        self.assertTrue(len(self.client.requests) > 0, "No requests were recorded")
        request = self.client.requests[-1]
        
        self.assertIn('chat', request['endpoint_url'], "Endpoint URL should contain 'chat'")
        self.assertEqual("llama3", request['data'].get('model'), "Model name not correctly passed")
        self.assertEqual(test_message, request['data']['messages'][0]['content'], "Message content not correctly passed")
        self.assertEqual("test_123", request['request_id'], "Request ID not correctly passed")

    def test_generate_method(self):
        """Test the generate method converts properly to chat request."""
        self.logger.info("Testing Generate Method")
        
        # Reset requests list
        self.client.requests = []
        
        # Test the generate method
        response = self.client.generate(
            model="llama3",
            prompt="Generate some text",
            max_tokens=20,
            temperature=0.8,
            request_id="generate_123"
        )
        
        # Verify the generate request converts to a chat request
        self.assertTrue(len(self.client.requests) > 0, "No requests were recorded")
        request = self.client.requests[-1]
        
        self.assertIn('chat', request['endpoint_url'], "Generate should convert to chat endpoint")
        self.assertEqual("llama3", request['data'].get('model'), "Model name not correctly passed")
        self.assertEqual("Generate some text", request['data']['messages'][0]['content'], 
                        "Prompt not correctly converted to message")
        self.assertEqual("generate_123", request['request_id'], "Request ID not correctly passed")

    def test_completions_method(self):
        """Test the completions method converts properly to chat request."""
        self.logger.info("Testing Completions Method")
        
        # Reset requests list
        self.client.requests = []
        
        # Test the completions method
        response = self.client.completions(
            model="llama3",
            prompt="Complete this text",
            max_tokens=20,
            temperature=0.8,
            request_id="complete_123"
        )
        
        # Verify the completions request converts to a chat request
        self.assertTrue(len(self.client.requests) > 0, "No requests were recorded")
        request = self.client.requests[-1]
        
        self.assertIn('chat', request['endpoint_url'], "Completions should convert to chat endpoint")
        self.assertEqual("llama3", request['data'].get('model'), "Model name not correctly passed")
        self.assertEqual("Complete this text", request['data']['messages'][0]['content'], 
                        "Prompt not correctly converted to message")
        self.assertEqual("complete_123", request['request_id'], "Request ID not correctly passed")

    def test_backoff_queue_structure(self):
        """Test the structure of the backoff and queue implementation"""
        self.logger.info("Testing Backoff and Queue Structure")
        
        # Create a real client for structure testing
        client = ollama()
        
        # Check basic attributes
        self.assertTrue(hasattr(client, 'max_retries'), "Missing max_retries attribute")
        self.assertTrue(hasattr(client, 'backoff_factor'), "Missing backoff_factor attribute")
        self.assertTrue(hasattr(client, 'initial_retry_delay'), "Missing initial_retry_delay attribute")
        self.assertTrue(hasattr(client, 'max_retry_delay'), "Missing max_retry_delay attribute")
        
        # Check queue attributes
        self.assertTrue(hasattr(client, 'request_queue'), "Missing request_queue attribute")
        self.assertTrue(hasattr(client, 'queue_size'), "Missing queue_size attribute")
        self.assertTrue(hasattr(client, 'max_concurrent_requests'), "Missing max_concurrent_requests attribute")
        
        # Check circuit breaker attributes
        self.assertTrue(hasattr(client, 'circuit_state'), "Missing circuit_state attribute")
        self.assertTrue(hasattr(client, 'failure_count'), "Missing failure_count attribute")
        self.assertTrue(hasattr(client, 'failure_threshold'), "Missing failure_threshold attribute")
        
        # Check priority queue structure
        self.assertTrue(
            hasattr(client, 'PRIORITY_HIGH') and 
            hasattr(client, 'PRIORITY_NORMAL') and 
            hasattr(client, 'PRIORITY_LOW'),
            "Missing priority level attributes"
        )

    def test_callable_interface(self):
        """Test the callable interface"""
        self.logger.info("Testing Callable Interface")
        
        # Reset requests list
        self.client.requests = []
        
        # Test calling the client directly
        response = self.client("chat", 
                              model="llama3",
                              messages=[{"role": "user", "content": "Test callable interface"}])
        
        # Verify the request
        self.assertTrue(len(self.client.requests) > 0, "No requests were recorded")
        request = self.client.requests[-1]
        
        self.assertIn('chat', request['endpoint_url'], "Endpoint URL should contain 'chat'")
        self.assertEqual("llama3", request['data'].get('model'), "Model name not correctly passed")
        self.assertEqual("Test callable interface", request['data']['messages'][0]['content'], 
                        "Message content not correctly passed")



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