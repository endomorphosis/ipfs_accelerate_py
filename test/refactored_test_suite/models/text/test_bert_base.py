"""
Test for BERT base model.

This demonstrates the new test structure using the standardized ModelTest base class.
"""

import os
import unittest
from refactored_test_suite.model_test import ModelTest

class TestBertBaseModel(ModelTest):
    """Test suite for BERT base model."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        self.model_name = "bert-base-uncased"
        self.model = self.load_model(self.model_name)
        self.input_text = "Hello, world!"
    
    def load_model(self, model_name):
        """Load a model for testing."""
        # In a real implementation, this would load the actual model
        # For this example, we'll return a mock model
        return MockBertModel(model_name)
    
    def test_should_initialize_correctly(self):
        """Test that the model initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.name, self.model_name)
    
    def test_should_process_simple_input(self):
        """Test that the model can process a simple input."""
        result = self.model.process(self.input_text)
        self.assertIsNotNone(result)
        self.assertIn("embeddings", result)
        self.assertIn("logits", result)
    
    def test_should_handle_batch_input(self):
        """Test that the model can handle batch input."""
        batch_input = [self.input_text, "Another test input"]
        result = self.model.process_batch(batch_input)
        self.assertEqual(len(result), len(batch_input))
    
    def test_should_raise_error_for_invalid_input(self):
        """Test that the model raises an error for invalid input."""
        with self.assertRaises(ValueError):
            self.model.process(None)




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


class MockBertModel:
    """Mock BERT model for testing."""
    
    def __init__(self, name):
        """Initialize the model."""
        self.name = name
    
    def process(self, text):
        """Process a single text input."""
        if text is None:
            raise ValueError("Input text cannot be None")
        
        # Return mock results
        return {
            "embeddings": [0.1, 0.2, 0.3],
            "logits": [0.4, 0.5, 0.6]
        }
    
    def process_batch(self, texts):
        """Process a batch of text inputs."""
        return [self.process(text) for text in texts]


if __name__ == "__main__":
    unittest.main()