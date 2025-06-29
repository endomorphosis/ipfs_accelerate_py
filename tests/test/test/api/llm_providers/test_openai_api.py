"""
Test for OpenAI API integration.

This test verifies connectivity and functionality of the OpenAI API
including chat completions, embeddings, and error handling.
"""

import pytest
import os
import time
import json
import requests
from unittest import mock
import sys
from pathlib import Path

# Add the root directory to the Python path
test_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(test_dir) not in sys.path:
    sys.path.insert(0, str(test_dir))

# Conditionally import OpenAI client
try:
    import openai
    has_openai = True
except ImportError:
    has_openai = False

from common.fixtures import mock_api_response, api_key


@pytest.fixture
def openai_client():
    """Create an OpenAI client for testing."""
    if not has_openai:
        pytest.skip("OpenAI package not installed")
        
    api_key_env = os.environ.get("OPENAI_API_KEY")
    if not api_key_env:
        pytest.skip("OPENAI_API_KEY environment variable not set")
        
    return openai.OpenAI(api_key=api_key_env)


@pytest.mark.api
@pytest.mark.openai
class TestOpenAIAPI:
    """Test suite for OpenAI API integration."""
    
    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    def test_client_initialization(self, openai_client):
        """Test that the OpenAI client initializes properly."""
        assert openai_client is not None
        assert hasattr(openai_client, "chat")
        assert hasattr(openai_client, "embeddings")

    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    @pytest.mark.integration
    def test_chat_completion(self, openai_client):
        """Test chat completion API."""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, who are you?"}
                ],
                max_tokens=50
            )
            
            assert response is not None
            assert hasattr(response, "choices")
            assert len(response.choices) > 0
            assert hasattr(response.choices[0], "message")
            assert response.choices[0].message.content != ""
            
        except openai.APIError as e:
            pytest.skip(f"OpenAI API error: {str(e)}")

    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    @pytest.mark.integration
    def test_embeddings(self, openai_client):
        """Test embeddings API."""
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input="Hello world"
            )
            
            assert response is not None
            assert hasattr(response, "data")
            assert len(response.data) > 0
            assert hasattr(response.data[0], "embedding")
            assert len(response.data[0].embedding) > 0
            
        except openai.APIError as e:
            pytest.skip(f"OpenAI API error: {str(e)}")

    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    def test_api_error_handling(self):
        """Test API error handling."""
        # Invalid API key should raise an error
        client = openai.OpenAI(api_key="invalid_key")
        
        with pytest.raises((openai.AuthenticationError, openai.APIError)):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}]
            )

    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    @pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4"])
    def test_different_models(self, openai_client, model):
        """Test different models if available."""
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            assert response is not None
            assert hasattr(response, "model")
            assert model in response.model
            
        except openai.APIError as e:
            if "model not found" in str(e).lower():
                pytest.skip(f"Model {model} not available")
            else:
                pytest.skip(f"OpenAI API error: {str(e)}")

    @pytest.mark.skipif(not has_openai, reason="OpenAI package not installed")
    def test_mock_response(self):
        """Test with mocked API response."""
        mock_data = {
            "choices": [
                {
                    "message": {
                        "content": "This is a mock response",
                        "role": "assistant"
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "created": int(time.time()),
            "id": "mock-id",
            "model": "gpt-3.5-turbo",
            "object": "chat.completion"
        }
        
        with mock.patch('openai.resources.chat.Completions.create', return_value=mock_data):
            client = openai.OpenAI(api_key="mock_key")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            assert response is not None
            assert response["choices"][0]["message"]["content"] == "This is a mock response"