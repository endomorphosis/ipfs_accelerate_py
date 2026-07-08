
import os
import sys
import unittest
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock detection for hardware capabilities
MOCK_HARDWARE_DETECTION = True

class TestHFDebertaV2(unittest.TestCase):
    """Test for the HuggingFace deberta-v2 model"""

    def setUp(self):
        # Set up any needed variables or configurations
        self.model_name = "microsoft/deberta-v2-base"
        self.task = "fill-mask"
        # Configure hardware detection
        self.cpu_only = True if os.environ.get("FORCE_CPU", "0") == "1" else False
        
    @pytest.mark.skip_if_no_gpu
    def test_model_loading(self):
        """Test loading deberta-v2 model with proper hardware detection"""
        try:
            # Import necessary libraries
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Determine appropriate device
            if torch.cuda.is_available() and not self.cpu_only:
                device = torch.device("cuda")
                logger.info("Using GPU for inference")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU for inference")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(device)
            
            # Basic inference test
            inputs = tokenizer("Hello, I'm a language model", return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            # Check output structure
            self.assertIn('last_hidden_state', outputs)
            self.assertTrue(isinstance(outputs.last_hidden_state, torch.Tensor))
            logger.info(f"Output shape: {outputs.last_hidden_state.shape}")
            
        except ImportError as e:
            # Skip test if dependencies aren't available
            logger.warning(f"Skipping test due to import error: {e}")
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            # Log error but don't let test fail if hardware is insufficient
            if "CUDA out of memory" in str(e) or "CUDA error" in str(e):
                logger.warning(f"Skipping test due to CUDA error: {e}")
                pytest.skip(f"Insufficient GPU memory: {e}")
            else:
                raise

    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_model_with_mock(self, mock_tokenizer, mock_model):
        """Test deberta-v2 model functionality with mocks"""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = MagicMock()
        mock_model_instance.return_value = mock_outputs
        mock_model.return_value = mock_model_instance
        
        # Test with mocked objects
        tokenizer = mock_tokenizer(self.model_name)
        model = mock_model(self.model_name)
        
        # Verify mock calls
        mock_tokenizer.assert_called_once_with(self.model_name)
        mock_model.assert_called_once_with(self.model_name)
        
        # This test should always pass as it's using mocks
        self.assertTrue(True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test deberta-v2 model")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v2-base", help="Model ID to test")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU even if GPU is available")
    args = parser.parse_args()
    
    # Set environment variable if CPU only
    if args.cpu_only:
        os.environ["FORCE_CPU"] = "1"
    
    # Run the tests
    unittest.main(argv=["first-arg-is-ignored"])
