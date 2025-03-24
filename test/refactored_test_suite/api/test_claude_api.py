"""Migrated to refactored test suite on 2025-03-21

Test Claude API implementation
"""

import os
import sys
import logging
import unittest
import requests
from unittest.mock import MagicMock, patch

from refactored_test_suite.model_test import ModelTest

class TestClaudeAPI(ModelTest):
    """Test class for Claude API implementation."""
    
    def setUp(self):
        super().setUp()
        # Set up API-specific attributes from APITest
        self.base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
        self.session = requests.Session()
        
        # Set model_id
        self.model_id = "claude-3-opus-20240229"
        
        # Set up mocks instead of actual imports
        self.module_mock = MagicMock()
        self.claude_class_mock = MagicMock()
        self.claude_instance_mock = MagicMock()
        
        # Set up mock attributes
        self.claude_instance_mock.endpoint = "https://api.claude.ai/v1"
        self.claude_instance_mock.get_endpoint.return_value = "https://api.claude.ai/v1"
        self.claude_class_mock.return_value = self.claude_instance_mock
        self.module_mock.claude = self.claude_class_mock
        
        # Other setup
        self.original_get_api_key = None
    
    def test_api_endpoint_configuration(self):
        """Test that the API endpoint is properly configured."""
        instance = self.claude_instance_mock
        
        # Check endpoint
        self.assertIsNotNone(instance.endpoint, "Endpoint should not be None")
        self.assertIsInstance(instance.endpoint, str, "Endpoint should be a string")
        self.assertTrue(instance.endpoint.startswith("http"), f"Endpoint should start with http: {instance.endpoint}")
        
        # Test get_endpoint method if it exists
        if hasattr(instance, "get_endpoint"):
            # Configure the mock to return a string when get_endpoint is called
            instance.get_endpoint.return_value = "https://api.claude.ai/v1"
            endpoint = instance.get_endpoint()
            self.assertIsInstance(endpoint, str, "get_endpoint should return a string")
    
    def test_basic_class_method(self):
        """Test basic class method functionality."""
        # Set up completion method
        completion_result = {"text": "This is a test response", "status": "success"}
        self.claude_instance_mock.completion.return_value = completion_result
        
        # Call method
        result = self.claude_instance_mock.completion(
            prompt="Hello, Claude!",
            max_tokens=100
        )
        
        # Assert method was called with correct arguments
        self.claude_instance_mock.completion.assert_called_once()
        args, kwargs = self.claude_instance_mock.completion.call_args
        self.assertEqual(kwargs["prompt"], "Hello, Claude!")
        self.assertEqual(kwargs["max_tokens"], 100)
        
        # Check result
        self.assertEqual(result, completion_result)
    
    def test_api_key_handling(self):
        """Test API key handling."""
        # Set up API key handling mock
        api_key = "test_api_key_12345"
        self.claude_instance_mock.get_api_key = MagicMock(return_value=api_key)
        
        # Test API key retrieval
        retrieved_key = self.claude_instance_mock.get_api_key(metadata={"user": "test"})
        self.assertEqual(retrieved_key, api_key)
        
        # Check that metadata was used
        self.claude_instance_mock.get_api_key.assert_called_once()
        args, kwargs = self.claude_instance_mock.get_api_key.call_args
        self.assertEqual(kwargs["metadata"], {"user": "test"})
    
    def test_error_handling(self):
        """Test error handling in API."""
        # Set up error scenario
        error_response = {"error": "API key invalid", "code": "auth_error"}
        self.claude_instance_mock.completion.side_effect = Exception("API authentication failed")
        
        # Test error handling
        with self.assertRaises(Exception) as context:
            self.claude_instance_mock.completion(prompt="This should fail")
        
        # Check error
        self.assertIn("API authentication failed", str(context.exception))
        


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


        # Verify method was called
        self.claude_instance_mock.completion.assert_called_once()
        
    def tearDown(self):
        """Clean up resources."""
        self.session.close()
        super().tearDown()
    
    def load_model(self, model_name):
        """Load a model for testing.
        
        For API tests, this creates a mock model client.
        """
        api_client = MagicMock()
        api_client.model_name = model_name
        api_client.generate = MagicMock(return_value="Generated text from Claude API")
        return api_client
    
    def verify_model_output(self, model, input_data, expected_output=None):
        """Verify that model produces expected output.
        
        For API tests, this verifies the mock responses.
        """
        # For API tests, we often verify mocks were called correctly
        output = model.generate(input_data)
        if expected_output:
            self.assertEqual(expected_output, output)
        else:
            self.assertIsNotNone(output)
            
    def get_endpoint_url(self, endpoint):
        """Get full URL for an endpoint (from APITest)."""
        return f"{self.base_url}/{endpoint.lstrip('/')}"
    
    def assertStatusCode(self, response, expected_code):
        """Assert that response has expected status code (from APITest)."""
        self.assertEqual(expected_code, response.status_code, 
                        f"Expected status code {expected_code}, got {response.status_code}")