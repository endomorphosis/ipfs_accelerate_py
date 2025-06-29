#!/usr/bin/env python3
"""
Skill implementation for qwen2-7b with hardware platform support
"""

import os
import sys
from .mojo_max_support import MojoMaxTargetMixin
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoFeatureExtractor, AutoProcessor, AutoImageProcessor, AutoModelForImageClassification, AutoModelForAudioClassification, AutoModelForVideoClassification

class Qwen27BSkill:
    """Skill for qwen2-7b model with hardware platform support."""
    
    def __init__(self, model_id="Qwen/Qwen2-7B-Instruct", device=None):
        """Initialize the skill."""
        super().__init__()
        self.model_id = model_id
        self.device = device or self.get_default_device()
        self.tokenizer = None
        self.model = None
        
    def get_default_device(self):
        """Get the best available device (legacy method, use get_default_device_with_mojo_max)."""
        return self.get_default_device_with_mojo_max()
    
    def load_model(self):
        """Load the model and tokenizer based on modality."""
        if self.model is None:
            # Determine model modality
            modality = "text"
            
            # Load appropriate tokenizer/processor and model based on modality
            if modality == "audio":
                self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            elif modality == "vision":
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
            elif modality == "multimodal":
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
            elif modality == "video":
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForVideoClassification.from_pretrained(self.model_id)
            else:
                # Default to text
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
            
            # Move to device
            if self.device not in ["cpu", "mojo_max", "max", "mojo"]:
                self.model = self.model.to(self.device)
    
    def process(self, text):
        """Process the input text and return the output."""
        # Check for Mojo/MAX target
        if self.device in ["mojo_max", "max", "mojo"]:
            return self.process_with_mojo_max(text, self.model_id)
        
        # Ensure model is loaded
        self.load_model()
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move to device
        if self.device not in ["cpu", "mojo_max", "max", "mojo"]:
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
def create_skill(model_id="qwen2-7b", device=None):
    """Create a skill instance."""
            return Qwen27BSkill(model_id=model_id, device=device)