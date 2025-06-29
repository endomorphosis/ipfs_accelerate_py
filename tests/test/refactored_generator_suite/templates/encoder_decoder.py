#!/usr/bin/env python3
"""
Encoder-Decoder Architecture Template

This module provides the architecture template for encoder-decoder models like T5, BART, etc.
"""

from typing import Dict, Any, List
from .base_architecture import BaseArchitectureTemplate

class EncoderDecoderArchitectureTemplate(BaseArchitectureTemplate):
    """Template for encoder-decoder architecture models like T5, BART, etc."""
    
    def __init__(self):
        """Initialize the encoder-decoder architecture template."""
        super().__init__()
        self.architecture_type = "encoder-decoder"
        self.architecture_name = "Encoder-Decoder Architecture"
        self.supported_task_types = ["text2text_generation", "translation", "summarization"]
        self.default_task_type = "text2text_generation"
        self.model_description = "This is a transformer-based sequence-to-sequence model with separate encoder and decoder components."
        self.hidden_size = 768  # Default hidden size, varies by model
        self.test_input = "Translate to French: Hello, how are you?"
    
    def get_model_class(self, task_type: str) -> str:
        """Get encoder-decoder model class for task type."""
        if task_type == "translation":
            return "self.transformers.AutoModelForSeq2SeqLM"
        elif task_type == "summarization":
            return "self.transformers.AutoModelForSeq2SeqLM"
        else:  # text2text_generation
            return "self.transformers.AutoModelForSeq2SeqLM"
    
    def get_processor_class(self, task_type: str) -> str:
        """Get encoder-decoder processor class for task type."""
        return "self.transformers.AutoTokenizer"
    
    def get_input_processing_code(self, task_type: str) -> str:
        """Get encoder-decoder input processing code."""
        return """
# Process input for encoder-decoder model
inputs = tokenizer(
    text, 
    return_tensors="pt", 
    padding=True, 
    truncation=True,
    max_length=512
)

# Move inputs to the correct device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
    
    def get_output_processing_code(self, task_type: str) -> str:
        """Get encoder-decoder output processing code."""
        if task_type == "translation":
            return """
# Process output for translation
with self.torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )

# Decode the generated tokens
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
"""
        elif task_type == "summarization":
            return """
# Process output for summarization
with self.torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=150,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )

# Decode the generated tokens
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
"""
        else:  # text2text_generation
            return """
# Process output for text2text generation
with self.torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

# Decode the generated tokens
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
"""
    
    def get_mock_processor_code(self) -> str:
        """Get encoder-decoder mock processor code."""
        return """
def mock_tokenize(text, return_tensors="pt", padding=None, truncation=None, max_length=None):
    if isinstance(text, str):
        batch_size = 1
    else:
        batch_size = len(text)
    
    if hasattr(self, 'torch'):
        torch = self.torch
    else:
        import torch
    
    # Model-specific mock input format
    return {
        "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, 10), dtype=torch.long)
    }
"""
    
    def get_mock_output_code(self) -> str:
        """Get encoder-decoder mock output code."""
        return """
result = MagicMock()
result.sequences = torch.randint(0, 50000, (batch_size, sequence_length))
result.decoder_hidden_states = None
result.decoder_attentions = None
return result
"""
    
    def get_compatibility_matrix(self) -> Dict[str, bool]:
        """Get encoder-decoder hardware compatibility matrix."""
        return {
            "cpu": True,
            "cuda": True,
            "rocm": True,
            "mps": True,
            "openvino": True,
            "qnn": False  # Limited support for encoder-decoder models in Qualcomm QNN
        }