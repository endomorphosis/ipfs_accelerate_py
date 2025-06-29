#!/usr/bin/env python3
"""
CPU hardware template for IPFS Accelerate Python.

This module provides the CPU-specific hardware template.
"""

from typing import Dict, Any, List
from .base_hardware import BaseHardwareTemplate


class CPUHardwareTemplate(BaseHardwareTemplate):
    """Template for CPU hardware."""
    
    def __init__(self):
        """Initialize the CPU hardware template."""
        super().__init__()
        self.hardware_type = "cpu"
        self.hardware_name = "CPU"
        self.requires_specialized_code = False
    
    def get_hardware_detection_code(self) -> str:
        """Get CPU hardware detection code."""
        return """
# CPU is always available
def is_available():
    return True
"""
    
    def get_hardware_init_code(self, model_class: str, task_type: str) -> str:
        """Get CPU hardware initialization code."""
        return f"""
# Initialize model on CPU
model = {model_class}.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    cache_dir=cache_dir
)
model.eval()
"""
    
    def get_handler_creation_code(self, model_class: str, task_type: str) -> str:
        """Get code for creating handler functions for CPU."""
        return f"""
# Create handler function for CPU {task_type}
def create_handler(model, tokenizer, device):
    def handler(input_data):
        # Process input
        inputs = tokenizer(input_data, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Process output
        result = {{"success": True, "outputs": outputs, "device": device}}
        return result
        
    return handler
    
handler = create_handler(model, tokenizer, device)
"""
    
    def get_inference_code(self, task_type: str) -> str:
        """Get CPU inference code."""
        if task_type in ["text_embedding", "text_generation", "text2text_generation", "masked_lm"]:
            return """
# CPU inference for text tasks
with torch.no_grad():
    outputs = model(**inputs)
"""
        elif task_type in ["image_classification", "object_detection", "image_segmentation"]:
            return """
# CPU inference for image tasks
with torch.no_grad():
    outputs = model(**inputs)
"""
        elif task_type in ["image_text_matching", "visual_question_answering", "image_captioning"]:
            return """
# CPU inference for vision-text tasks
with torch.no_grad():
    outputs = model(**inputs)
"""
        elif task_type in ["speech_recognition", "audio_classification", "text_to_speech"]:
            return """
# CPU inference for speech tasks
with torch.no_grad():
    outputs = model(**inputs)
"""
        else:
            return f"""
# CPU inference for {task_type}
with torch.no_grad():
    outputs = model(**inputs)
"""
    
    def get_cleanup_code(self) -> str:
        """Get cleanup code for CPU."""
        return """
# CPU cleanup
del model
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
    
    def get_mock_code(self, model_class: str, task_type: str) -> str:
        """Get mock implementation code for CPU."""
        return f"""
# Create mock implementation for CPU {task_type}
class MockModel:
    def __init__(self):
        self.config = type('obj', (object,), {{
            'hidden_size': 768,
            'vocab_size': 30000,
            'num_hidden_layers': 12,
            'num_attention_heads': 12
        }})
        
    def __call__(self, **kwargs):
        # Create mock outputs based on input shape
        batch_size = kwargs.get('input_ids', [[0]]).shape[0]
        if 'input_ids' in kwargs:
            seq_len = kwargs['input_ids'].shape[1]
        else:
            seq_len = 10
            
        # Create appropriate mock outputs based on task
        mock_outputs = {{"last_hidden_state": torch.randn(batch_size, seq_len, self.config.hidden_size)}}
        if '{task_type}' == 'text_generation':
            mock_outputs['logits'] = torch.randn(batch_size, seq_len, self.config.vocab_size)
        elif '{task_type}' == 'image_classification':
            mock_outputs['logits'] = torch.randn(batch_size, 1000)  # 1000 classes
            
        return type('obj', (object,), mock_outputs)
        
model = MockModel()
model.eval = lambda: None
model.to = lambda device: model
model.generate = lambda **kwargs: torch.randint(0, 1000, (kwargs.get('input_ids', [[0]]).shape[0], 20))

class MockTokenizer:
    def __init__(self):
        pass
        
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            input_ids = torch.ones((1, 10), dtype=torch.long)
        else:
            input_ids = torch.ones((len(text), 10), dtype=torch.long)
        return {{"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}}
        
    def decode(self, token_ids, **kwargs):
        return "This is a mock output for {task_type}"
        
    def batch_decode(self, token_ids, **kwargs):
        return ["This is a mock output for {task_type}"] * token_ids.shape[0]
        
tokenizer = MockTokenizer()
"""
    
    def get_fallback_hardware(self) -> str:
        """Get fallback hardware if this one is not available."""
        return "cpu"  # CPU is always the fallback
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check if CPU hardware is compatible with the architecture type."""
        # CPU is compatible with all architecture types
        return True
    
    def get_import_statements(self) -> str:
        """Get CPU-specific import statements."""
        return """
# CPU-specific imports
import os
import torch
import numpy as np
"""