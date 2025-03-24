#!/usr/bin/env python3
"""
Test for Ollama API backoff and queue functionality.

This tests the backoff and queue functionality of the Ollama API client.
"""

import sys
import os
import concurrent.futures
import unittest
from refactored_test_suite.model_test import ModelTest

# Try importing the Ollama module
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "ipfs_accelerate_py"))
    from ipfs_accelerate_py.api_backends import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

class TestOllamaBackoff(ModelTest):
    """Test the Ollama API backoff and queue functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the Ollama client for the test."""
        super().setUpClass()
        if not HAS_OLLAMA:
            cls.logger.error("Ollama module not available, tests will be skipped")
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        if not HAS_OLLAMA:
            self.skipTest("Ollama module not available")
        
        # Initialize client
        self.client = ollama()
        self.logger.info("Initialized Ollama client")
        
        # Store original values
        self.original_max_retries = self.client.max_retries
        self.original_backoff_factor = self.client.backoff_factor
        self.original_max_concurrent_requests = self.client.max_concurrent_requests
    
    def tearDown(self):
        """Restore original client settings."""
        if HAS_OLLAMA:
            self.client.max_retries = self.original_max_retries
            self.client.backoff_factor = self.original_backoff_factor
            self.client.max_concurrent_requests = self.original_max_concurrent_requests
        super().tearDown()
    
    def test_client_attributes(self):
        """Test that client has required attributes."""
        self.assertIsNotNone(self.client.max_retries)
        self.assertIsNotNone(self.client.backoff_factor)
        self.assertIsNotNone(self.client.max_concurrent_requests)
        
        self.logger.info(f"Max retries: {self.client.max_retries}")
        self.logger.info(f"Backoff factor: {self.client.backoff_factor}")
        self.logger.info(f"Max concurrent requests: {self.client.max_concurrent_requests}")
    
    def test_concurrent_request_limit(self):
        """Test that concurrent request limit works."""
        # Set very low limit to force queue
        self.client.max_concurrent_requests = 1
        results = []
        
        # Send 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    lambda i=i: self.client.generate(
                        model="llama3",
                        prompt=f"Request {i}",
                        request_id=f"test-{i}"
                    )
                )
                for i in range(3)
            ]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Request completed successfully: {result.get('text', '')[:50]}...")
                except Exception as e:
                    self.logger.error(f"Request failed: {str(e)}")
                    self.fail(f"Request failed: {str(e)}")
        
        # Verify results
        self.assertEqual(len(results), 3, "Expected 3 successful results")
        self.logger.info(f"Successfully completed {len(results)} requests with concurrency limit of 1")
    
    def test_retry_mechanism(self):
        """Test retry mechanism with artificial failure."""
        # Create a test function that fails a few times then succeeds
        test_attempts = []
        
        def test_function():
            test_attempts.append(1)
            if len(test_attempts) < 3:
                raise Exception("Simulated failure")
            return {"text": "Success after retrying"}
        
        # Patch the client's generate method
        original_generate = self.client.generate
        self.client.generate = test_function
        
        # Set retry parameters
        self.client.max_retries = 5
        self.client.backoff_factor = 0.1
        
        try:
            # Should succeed after retries
            result = self.client.generate()
            self.assertEqual(result["text"], "Success after retrying")
            self.assertEqual(len(test_attempts), 3, "Expected 3 attempts")
        finally:
            # Restore original method
            self.client.generate = original_generate
    
    def load_model(self, model_name):
        """Implement the required load_model method from ModelTest."""
        # This is just a placeholder since we're not actually loading models in this test
        return lambda x: {"text": f"Mock response for {x}"}



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
    unittest.main()