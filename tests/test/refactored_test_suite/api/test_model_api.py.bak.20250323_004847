"""
Test for model API endpoints.

This demonstrates the new test structure using the standardized APITest base class.
"""

import unittest
import json
from unittest.mock import MagicMock, patch
from refactored_test_suite.api_test import APITest

class TestModelAPI(APITest):
    """Test suite for model API endpoints."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
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


if __name__ == "__main__":
    unittest.main()