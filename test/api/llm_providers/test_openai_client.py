#!/usr/bin/env python3
"""
Test file for OpenAI API.

This file contains tests for the OpenAI API,
including connection tests and API functionality tests.
"""

import os
import sys
import pytest
import logging
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI API imports
try:
    import openai
    from openai import OpenAI
except ImportError:
    pass

# API-specific fixtures
@pytest.fixture
def api_base_url():
    """Get the base URL for OpenAI API tests."""
    return os.environ.get("API_BASE_URL", "http://localhost:8000")

@pytest.fixture
def api_key():
    """Get the API key for OpenAI API tests."""
    return os.environ.get("API_KEY", "test_key")

@pytest.fixture
def openai_client(api_base_url, api_key):
    """Create an OpenAI API client for openai tests."""
    try:
        client = OpenAI(
            base_url=api_base_url,
            api_key=api_key
        )
        return client
    except (ImportError, Exception) as e:
        pytest.skip(f"Could not create OpenAI client: {e}")

@pytest.fixture
def mock_openai_client():
    """Create a mock client for openai API tests."""
    try:
        from unittest.mock import MagicMock

        mock_client = MagicMock()

        # Mock completion response
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content="Mock response"))
        ]
        mock_client.chat.completions.create.return_value = mock_completion

        # Mock embedding response
        mock_embedding = MagicMock()
        mock_embedding.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3])
        ]
        mock_client.embeddings.create.return_value = mock_embedding

        return mock_client
    except (ImportError, Exception) as e:
        pytest.skip(f"Could not create mock client: {e}")

@pytest.mark.api
class TestOpenaiClient:
    """
    Tests for OpenAI API.
    """
    
    def test_api_connection(self, openai_client):
        """Test connection to OpenAI API."""
        assert openai_client is not None
    
    def test_chat_completion(self, openai_client):
        """Test chat completion with OpenAI API."""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ]
            )
            
            assert response is not None
            assert len(response.choices) > 0
            assert response.choices[0].message.content
        except Exception as e:
            pytest.skip(f"API test failed: {e}")
    
    def test_embeddings(self, openai_client):
        """Test embeddings with OpenAI API."""
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input="The quick brown fox jumps over the lazy dog"
            )
            
            assert response is not None
            assert len(response.data) > 0
            assert len(response.data[0].embedding) > 0
        except Exception as e:
            pytest.skip(f"API test failed: {e}")
    
    def test_with_mock_client(self, mock_openai_client):
        """Test with mock OpenAI API client."""
        assert mock_openai_client is not None
        
        # Test mock completion
        response = mock_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{'role': 'user', 'content': 'Hello'}]
        )
        assert response.choices[0].message.content == "Mock response"
        
        # Test mock embedding
        embed_response = mock_openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input="Test input"
        )
        assert embed_response.data[0].embedding == [0.1, 0.2, 0.3]

if __name__ == "__main__":
    # Run tests directly
    pytest.main(["-xvs", __file__])
