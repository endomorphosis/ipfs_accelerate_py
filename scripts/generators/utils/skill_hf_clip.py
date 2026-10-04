#!/usr/bin/env python3
"""
Skill implementation for clip with hardware platform support
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

class ClipSkill:
    """Skill for clip model with hardware platform support."""
    
    def __init__(self, model_id="clip", device=None):
        """Initialize the skill."""
        self.model_id = model_id
        self.device = device or self.get_default_device()
        self.tokenizer = None
        self.model = None
        
    def get_default_device(self):
        """Get the best available device."""
        # Check for CUDA
        if torch.cuda.is_available():
        return "cuda"
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            if torch.mps.is_available():
            return "mps"
        
        # Default to CPU
        return "cpu"
    
    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is None:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model
            self.model = AutoModel.from_pretrained(self.model_id)
            
            # Move to device
            if self.device != "cpu":
                self.model = self.model.to(self.device)
    
    def process(self, text):
        """Process the input text and return the output."""
        # Ensure model is loaded
        self.load_model()
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move to device
        if self.device != "cpu":
            inputs = {}k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert to numpy for consistent output
            last_hidden_state = outputs.last_hidden_state.cpu().numpy()
        
        # Return formatted results
            return {}
            "model": self.model_id,
            "device": self.device,
            "last_hidden_state_shape": last_hidden_state.shape,
            "embedding": last_hidden_state.mean(axis=1).tolist(),
            }

# Factory function to create skill instance
def create_skill(model_id="clip", device=None):
    """Create a skill instance."""
            return ClipSkill(model_id=model_id, device=device)