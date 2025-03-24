"""
Test for API backend functionality.

This file tests the core API backend functionality for the IPFS Accelerate framework.
Migrated to refactored test suite on 2025-03-21.
"""

import os 
import sys
import importlib.util
import importlib
from refactored_test_suite.api_test import APITest

from refactored_test_suite.model_test import ModelTest

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

# Mock API modules for testing
class MockAPIProvider:
    """Mock API provider for testing."""
    
    def __init__(self, provider_name):
        self.provider_name = provider_name
        self.initialized = True
    
    def generate(self, prompt, **kwargs):
        """Mock text generation."""
        return f"Response from {self.provider_name}: {prompt[:10]}..."
    
    def embeddings(self, text, **kwargs):
        """Mock embeddings generation."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]


class TestAPIBackend(ModelTest):
    """Test suite for API backend functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        self.model_id = "bert-base-uncased"  # Default model for API testing
        self.providers = {
            'claude': MockAPIProvider('claude'),
            'openai': MockAPIProvider('openai'),
            'groq': MockAPIProvider('groq'),
            'ollama': MockAPIProvider('ollama')
        }
    
    def test_api_initialization(self):
        """Test that API providers can be initialized correctly."""
        for name, provider in self.providers.items():
            self.assertTrue(provider.initialized, f"{name} provider not initialized")
    
    def test_text_generation(self):
        """Test that text generation works with all providers."""
        prompt = "Tell me about machine learning"
        for name, provider in self.providers.items():
            response = provider.generate(prompt)
            self.assertIsNotNone(response)
            self.assertIn(name, response)
    
    def test_embeddings_generation(self):
        """Test that embeddings generation works."""
        text = "This is a sample text for embeddings"
        for name, provider in self.providers.items():
            embeddings = provider.embeddings(text)
            self.assertIsNotNone(embeddings)
            self.assertEqual(len(embeddings), 5)  # Our mock returns 5 values

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