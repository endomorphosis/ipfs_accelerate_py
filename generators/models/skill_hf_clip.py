#!/usr/bin/env python3
"""
Skill implementation for clip with hardware platform support
"""

import os
import sys
from .mojo_max_support import MojoMaxTargetMixin
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

class ClipSkill(MojoMaxTargetMixin):
    """Skill for clip model with hardware platform support."""
    
    def __init__(self, model_id="clip", device=None):
        """Initialize the skill."""
        super().__init__(model_id=model_id, device=device) # Pass kwargs to mixin
        self.model_id = model_id
        self.device = device or self.get_default_device_with_mojo_max()
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is None:
            if self.device in ["mojo_max", "max", "mojo"]:
                # Conceptual Mojo/MAX model loading
                print(f"Conceptual: Converting and loading {self.model_id} for Mojo/MAX on device {self.device}")
                conceptual_input_shapes = {"input_ids": (1, 77), "pixel_values": (1, 3, 224, 224)} # Example shapes for CLIP
                max_ir = self._mojo_max_ir_converter.convert_from_pytorch(
                    pytorch_model=self.model_id,
                    input_shapes=conceptual_input_shapes
                )
                optimized_ir = self._mojo_max_ir_converter.optimize_max_ir(max_ir)
                compiled_model_path = self._mojo_max_ir_converter.compile_to_mojomodel(
                    optimized_ir, f"generated_models/{self.model_id.replace('/', '_')}"
                )
                self.model = f"Mojo/MAX_model_loaded_from_{compiled_model_path}"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id) # Tokenizer is still needed
            else:
                # Standard PyTorch model loading
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                
                # Load model
                self.model = AutoModel.from_pretrained(self.model_id)
                
                # Move to device
                if self.device not in ["cpu", "mojo_max", "max", "mojo"]:
                    self.model = self.model.to(self.device)
    
    def process(self, text):
        """Process the input text and return the output."""
        # Check for Mojo/MAX target
        if self.device in ["mojo_max", "max", "mojo"]:
            return self.process_with_mojo_max(text, self.model_id)
        else:
            # Ensure model is loaded
            self.load_model()
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move to device
        if self.device not in ["cpu", "mojo_max", "max", "mojo"]:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert to numpy for consistent output
        last_hidden_state = outputs.last_hidden_state.cpu().numpy()
        
        # Return formatted results
        return {
            "model": self.model_id,
            "device": self.device,
            "last_hidden_state_shape": last_hidden_state.shape,
            "embedding": last_hidden_state.mean(axis=1).tolist(),
        }

# Factory function to create skill instance
def create_skill(model_id="clip", device=None):
    """Create a skill instance."""
    return ClipSkill(model_id=model_id, device=device)
