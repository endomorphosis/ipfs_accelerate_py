#!/usr/bin/env python3
"""
BERT model test template for {model_name}.
Generated on {generated_at}
"""

import os
import logging
import unittest
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Test{normalized_name}(unittest.TestCase):
    """Test class for {model_name}."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        try:
            # Import dependencies
            import torch
            import transformers
            cls.torch = torch
            cls.transformers = transformers
            
            # Set up device
            cls.device = "cpu"
            if torch.cuda.is_available():
                cls.device = "cuda"
                logger.info("Using CUDA")
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                cls.device = "mps"
                logger.info("Using MPS")
            
            # Load model
            cls.tokenizer = transformers.AutoTokenizer.from_pretrained("{model_name}")
            cls.model = transformers.AutoModel.from_pretrained("{model_name}")
            
            # Move to device
            if cls.device != "cpu":
                cls.model = cls.model.to(cls.device)
                
            logger.info(f"Model loaded: {model_name} on {cls.device}")
        except Exception as e:
            logger.error(f"Error setting up test: {e}")
            raise
    
    def test_model_loaded(self):
        """Test that model was loaded successfully."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
    
    def test_inference(self):
        """Test basic inference."""
        # Prepare input
        text = "This is a test sentence for BERT model."
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move inputs to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        self.assertIn("last_hidden_state", outputs)
        
        # Check shapes
        self.assertEqual(len(outputs.last_hidden_state.shape), 3)  # [batch, seq_len, hidden]
        
        logger.info("Inference successful")

if __name__ == "__main__":
    unittest.main()