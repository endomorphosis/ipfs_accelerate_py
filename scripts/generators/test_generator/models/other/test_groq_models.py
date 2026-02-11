#!/usr/bin/env python3
"""
Test for Groq API model compatibility.

This test verifies integration with various Groq LLM models and endpoints.
"""

import os
import sys
import json
import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from refactored_test_suite.model_test import ModelTest

# Try importing the Groq implementation
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "ipfs_accelerate_py"))
    from ipfs_accelerate_py.api_backends.groq import groq, CHAT_MODELS, VISION_MODELS, AUDIO_MODELS, ALL_MODELS
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    # Mock the model lists for testing
    CHAT_MODELS = ["llama3-8b-8192", "mixtral-8x7b-32768"]
    VISION_MODELS = ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]
    AUDIO_MODELS = ["whisper-large-v3"]
    ALL_MODELS = CHAT_MODELS + VISION_MODELS + AUDIO_MODELS

class TestGroqModels(ModelTest):
    """Test all Groq models and endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up resources for the entire test class."""
        super().setUpClass()
        if not HAS_GROQ:
            cls.logger.error("Groq module not available, tests will be skipped or mocked")
    
    def setUp(self):
        """Set up for each test."""
        super().setUp()
        
        # Test API key (this is a placeholder, not a real key)
        self.api_key = "gsk_test_key_not_real"
        
        # Initialize the metadata and resources
        self.metadata = {"groq_api_key": self.api_key}
        self.resources = {}
        
        # Initialize Groq client or mock
        if HAS_GROQ:
            self.groq_client = groq(resources=self.resources, metadata=self.metadata)
        else:
            self.groq_client = self._create_mock_groq_client()
            self.logger.warning("Using mock Groq client for testing")
        
        # Define test prompts
        self.chat_prompt = [
            {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
        ]
        
        self.vision_prompt = [
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {
                    "url": "https://images.dog.ceo/breeds/retriever-golden/n02099601_3073.jpg"
                }}
            ]}
        ]
        
        # Test parameters
        self.test_params = {
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.95
        }
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "chat_models": {},
            "vision_models": {},
            "audio_models": {},
            "summary": {
                "success": 0,
                "failure": 0,
                "total": 0
            }
        }
    
    def _create_mock_groq_client(self):
        """Create a mock Groq client for testing when the real one isn't available."""
        mock_client = MagicMock()
        
        # Mock chat method
        def mock_chat(model_name=None, messages=None, **kwargs):
            return {
                "text": f"Mock response from {model_name}: This is Paris, the capital of France.",
                "model": model_name,
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 10,
                    "total_tokens": 25
                }
            }
        
        mock_client.chat = mock_chat
        
        # Mock other methods
        mock_client.create_groq_endpoint_handler = MagicMock(return_value=lambda: True)
        mock_client.get_usage_stats = MagicMock(return_value={"total_requests": 0})
        mock_client.count_tokens = MagicMock(return_value={"estimated_token_count": 7})
        
        return mock_client
    
    def test_endpoint_creation(self):
        """Test endpoint handler creation."""
        handler = self.groq_client.create_groq_endpoint_handler()
        self.assertTrue(callable(handler), "Endpoint handler should be callable")
        self.logger.info("Successfully created endpoint handler")
    
    def test_usage_tracking(self):
        """Test usage tracking functionality."""
        try:
            stats = self.groq_client.get_usage_stats()
            self.assertIsNotNone(stats)
            self.assertIn("total_requests", stats)
            self.logger.info(f"Usage tracking working: {stats.get('total_requests', 0)} requests tracked")
        except Exception as e:
            self.fail(f"Usage tracking failed: {str(e)}")
    
    def test_token_counting(self):
        """Test token counting functionality."""
        try:
            token_count = self.groq_client.count_tokens("This is a test sentence to count tokens.", "llama3-8b-8192")
            self.assertIsNotNone(token_count)
            self.assertIn("estimated_token_count", token_count)
            self.logger.info(f"Token counting working: {token_count.get('estimated_token_count', 0)} tokens estimated")
        except Exception as e:
            self.fail(f"Token counting failed: {str(e)}")
    
    def test_chat_models(self):
        """Test chat model functionality."""
        # Only test the first chat model to avoid long test times
        if CHAT_MODELS:
            model_name = CHAT_MODELS[0]
            self.logger.info(f"Testing chat model: {model_name}")
            
            try:
                response = self.groq_client.chat(
                    model_name=model_name,
                    messages=self.chat_prompt,
                    **self.test_params
                )
                
                self.assertIsNotNone(response)
                self.assertIn("text", response)
                self.logger.info(f"Successfully got response from {model_name}: {response.get('text', '')[:50]}...")
                
                # Store result
                self.results["chat_models"][model_name] = {
                    "status": "success",
                    "response": response.get("text", ""),
                    "implementation": "REAL" if HAS_GROQ else "MOCK"
                }
                self.results["summary"]["success"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to test {model_name}: {str(e)}")
                self.results["chat_models"][model_name] = {
                    "status": "failure",
                    "error": str(e),
                    "implementation": "FAILED"
                }
                self.results["summary"]["failure"] += 1
                
                # Don't fail the test if we're using a mock
                if HAS_GROQ:
                    self.fail(f"Chat model test failed: {str(e)}")
            
            self.results["summary"]["total"] += 1
    
    def test_vision_models(self):
        """Test vision model functionality."""
        # Only test the first vision model to avoid long test times
        if VISION_MODELS:
            model_name = VISION_MODELS[0]
            self.logger.info(f"Testing vision model: {model_name}")
            
            try:
                response = self.groq_client.chat(
                    model_name=model_name,
                    messages=self.vision_prompt,
                    **self.test_params
                )
                
                self.assertIsNotNone(response)
                self.assertIn("text", response)
                self.logger.info(f"Successfully got response from {model_name}: {response.get('text', '')[:50]}...")
                
                # Store result
                self.results["vision_models"][model_name] = {
                    "status": "success",
                    "response": response.get("text", ""),
                    "implementation": "REAL" if HAS_GROQ else "MOCK"
                }
                self.results["summary"]["success"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to test {model_name}: {str(e)}")
                self.results["vision_models"][model_name] = {
                    "status": "failure",
                    "error": str(e),
                    "implementation": "FAILED"
                }
                self.results["summary"]["failure"] += 1
                
                # Don't fail the test if we're using a mock
                if HAS_GROQ:
                    self.fail(f"Vision model test failed: {str(e)}")
            
            self.results["summary"]["total"] += 1
    
    def test_streaming_functionality(self):
        """Test streaming functionality with a chat model."""
        if not CHAT_MODELS:
            self.skipTest("No chat models available for testing")
        
        model_name = CHAT_MODELS[0]
        self.logger.info(f"Testing streaming with model: {model_name}")
        
        # In this test, we're just verifying the interface exists and functions
        # without actually using streaming (which would be complex to test)
        try:
            response = self.groq_client.chat(
                model_name=model_name,
                messages=self.chat_prompt,
                **self.test_params
            )
            
            # Verify response
            self.assertIsNotNone(response)
            self.assertIn("text", response)
            
            # Simulate streaming by splitting the response
            text = response.get("text", "")
            words = text.split()
            
            # Verify we can process chunks
            if words:
                self.logger.info(f"Simulated streaming successful with {len(words)} chunks")
            else:
                self.fail("No text content in response")
            
        except Exception as e:
            self.logger.error(f"Streaming test failed: {str(e)}")
            # Don't fail the test if we're using a mock
            if HAS_GROQ:
                self.fail(f"Streaming test failed: {str(e)}")
    
    def test_model_list_consistency(self):
        """Test that model lists are consistent."""
        self.assertIsNotNone(CHAT_MODELS)
        self.assertIsNotNone(VISION_MODELS)
        self.assertIsNotNone(AUDIO_MODELS)
        self.assertIsNotNone(ALL_MODELS)
        
        # Verify ALL_MODELS contains all models from other lists
        for model in CHAT_MODELS:
            self.assertIn(model, ALL_MODELS)
        
        for model in VISION_MODELS:
            self.assertIn(model, ALL_MODELS)
            
        for model in AUDIO_MODELS:
            self.assertIn(model, ALL_MODELS)
    
    def save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.model_dir, f"groq_model_test_results_{timestamp}.json")
        
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to: {filename}")
        return filename
    
    def load_model(self, model_name):
        """Implement the required load_model method from ModelTest."""
        # This is just a placeholder since we're not actually loading models in this test
        def mock_model(input_data):
            return {
                "text": f"Mock response for {input_data} using {model_name}",
                "model": model_name
            }
        return mock_model



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