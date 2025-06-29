#!/usr/bin/env python3
"""
Decoder-Only Architecture Template for IPFS Accelerate Python.

This module implements the architecture template for decoder-only models like GPT-2, LLaMA, etc.
"""

from typing import Dict, Any, Optional, List
from templates.base_architecture import BaseArchitectureTemplate


class DecoderOnlyArchitectureTemplate(BaseArchitectureTemplate):
    """Decoder-only architecture template implementation for models like GPT-2, LLaMA, etc."""
    
    def __init__(self):
        """Initialize the decoder-only architecture template."""
        super().__init__()
        self.architecture_type = "decoder-only"
        self.architecture_name = "Decoder-Only"
        self.model_description = "This model uses a unidirectional (causal) Transformer decoder architecture."
        self.supported_task_types = ["text_generation", "causal_lm"]
        self.default_task_type = "text_generation"
        self.hidden_size = 768  # Default hidden size (varies by model)
        self.test_input = "Once upon a time in a land far away,"
    
    def get_model_class(self, task_type: str) -> str:
        """Get the model class for this architecture and task type."""
        if task_type == "text_generation" or task_type == "causal_lm":
            return "AutoModelForCausalLM"
        else:
            return "AutoModelForCausalLM"
    
    def get_processor_class(self, task_type: str) -> str:
        """Get the processor class for this architecture and task type."""
        return "AutoTokenizer"
    
    def get_input_processing_code(self, task_type: str) -> str:
        """Get the input processing code for this architecture and task type."""
        return """
# Process the input text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Move inputs to the correct device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
    
    def get_output_processing_code(self, task_type: str) -> str:
        """Get the output processing code for this architecture and task type."""
        if task_type == "text_generation" or task_type == "causal_lm":
            return """
# For text generation, we use the generate method rather than the forward pass
max_new_tokens = kwargs.get("max_new_tokens", 100)
temperature = kwargs.get("temperature", 0.8)
do_sample = kwargs.get("do_sample", True)
top_p = kwargs.get("top_p", 0.9)
top_k = kwargs.get("top_k", 50)

# Generate text
output_ids = endpoint.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    do_sample=do_sample,
    top_p=top_p,
    top_k=top_k
)

# Decode the generated text
generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
"""
        else:
            return """
# Generic output processing
result = outputs
"""
    
    def get_mock_processor_code(self) -> str:
        """Get code for creating a mock tokenizer."""
        return """
def mock_tokenize(text, return_tensors=None, padding=None, truncation=None, max_length=None):
    # Create a mock tokenizer output
    import torch
    
    if isinstance(text, str):
        batch_size = 1
        text_batch = [text]
    else:
        batch_size = len(text)
        text_batch = text
    
    # Create mock input IDs (just use token positions as IDs)
    input_ids = torch.tensor([[i for i in range(min(len(t.split()), 32))] for t in text_batch])
    attention_mask = torch.ones_like(input_ids)
    
    # Add a batch dimension if necessary
    if return_tensors == "pt":
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    else:
        return {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy()
        }
"""
    
    def get_mock_output_code(self) -> str:
        """Get code for creating mock outputs."""
        return """
# Create mock outputs for decoder-only models
if isinstance(self, torch.nn):
    hidden_size = kwargs.get("hidden_size", 768)
else:
    hidden_size = 768

# For text generation, mock generated token IDs
if hasattr(endpoint, 'generate'):
    # Mock generate method that returns token IDs
    def mock_generate(**kwargs):
        # Generate some simple increasing sequence of tokens
        input_ids = kwargs.get("input_ids", torch.ones((batch_size, 10), dtype=torch.long))
        max_new_tokens = kwargs.get("max_new_tokens", 20)
        input_length = input_ids.shape[1]
        
        # Create sequence with input followed by new tokens
        generated_ids = torch.cat([
            input_ids,
            torch.arange(
                input_length, 
                input_length + max_new_tokens
            ).unsqueeze(0).repeat(batch_size, 1)
        ], dim=1)
        
        return generated_ids
    
    endpoint.generate = mock_generate
    return mock_generate(**kwargs)
else:
    # Mock standard forward pass output
    mock_outputs = type('obj', (object,), {
        'logits': torch.randn(batch_size, sequence_length, 50257)  # GPT-2 vocab size
    })
    
    return mock_outputs
"""
    
    def get_model_config(self, model_name: str) -> str:
        """Get model-specific configuration code."""
        return f"""
def get_model_config(self):
    \"\"\"Get the model configuration.\"\"\"
    return {{
        "model_name": "{model_name}",
        "architecture": "decoder-only",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "primary_task": "text_generation",
        "supported_tasks": [
            "text_generation",
            "causal_lm"
        ]
    }}
"""