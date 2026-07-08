"""
Test for model API endpoints.

This demonstrates the new test structure using the standardized ModelTest base class.
"""

import unittest
import json
import requests
import os
from unittest.mock import MagicMock, patch
from refactored_test_suite.model_test import ModelTest

class TestModelAPI(ModelTest):
    """Test suite for model API endpoints."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        # Set up API-specific attributes
        self.base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
        # Set up mock session for testing
        self.session = MagicMock()
        self.model_id = "bert-base-uncased"
    
    def test_should_get_model_info(self):
        """Test that the model info endpoint returns correct data."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": self.model_id,
            "name": "BERT Base Uncased",
            "type": "text",
            "parameters": "110M",
            "supported_hardware": ["cpu", "webgpu", "webnn"]
        }
        self.session.get.return_value = mock_response
        
        # Call API
        url = self.get_endpoint_url(f"models/{self.model_id}")
        response = self.session.get(url)
        
        # Verify response
        self.assertStatusCode(response, 200)
        data = response.json()
        self.assertEqual(data["id"], self.model_id)
        self.assertEqual(data["type"], "text")
        self.assertIn("webgpu", data["supported_hardware"])
    
    def test_should_return_404_for_nonexistent_model(self):
        """Test that the API returns 404 for nonexistent models."""
        # Set up mock response for nonexistent model
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": "Model not found",
            "code": "model_not_found"
        }
        self.session.get.return_value = mock_response
        
        # Call API with nonexistent model
        url = self.get_endpoint_url("models/nonexistent-model")
        response = self.session.get(url)
        
        # Verify response
        self.assertStatusCode(response, 404)
        data = response.json()
        self.assertEqual(data["code"], "model_not_found")
    
    def test_should_submit_inference_request(self):
        """Test that inference requests can be submitted."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "request-123",
            "model_id": self.model_id,
            "status": "processing",
            "created_at": "2025-03-21T12:34:56Z"
        }
        self.session.post.return_value = mock_response
        
        # Call API to submit inference request
        url = self.get_endpoint_url("inference")
        request_data = {
            "model_id": self.model_id,
            "inputs": {"text": "Hello, world!"},
            "options": {"use_webgpu": True}
        }
        response = self.session.post(url, json=request_data)
        
        # Verify response
        self.assertStatusCode(response, 200)
        data = response.json()
        self.assertEqual(data["model_id"], self.model_id)
        self.assertEqual(data["status"], "processing")
        
        # Verify request was made correctly
        self.session.post.assert_called_once()
        args, kwargs = self.session.post.call_args
        self.assertEqual(args[0], url)
        self.assertEqual(kwargs["json"], request_data)


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




    def load_model(self, model_name):
        """Load a model for testing.
        
        For API tests, this creates a mock model client.
        """
        api_client = MagicMock()
        api_client.model_name = model_name
        api_client.generate = MagicMock(return_value="Generated text from API")
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


if __name__ == "__main__":
    unittest.main()