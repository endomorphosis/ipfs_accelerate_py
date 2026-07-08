#!/usr/bin/env python3
"""
Text pipeline template for IPFS Accelerate Python.

This module implements the pipeline template for text-based models.
"""

from typing import Dict, Any, Callable, Tuple, Optional, List, Union
from templates.base_pipeline import BasePipelineTemplate


class TextPipelineTemplate(BasePipelineTemplate):
    """Text pipeline template implementation."""
    
    def __init__(self):
        """Initialize the text pipeline template."""
        super().__init__()
        self.pipeline_type = "text"
        self.input_type = "text"
        self.output_type = "text"
        self.requires_preprocessing = True
        self.requires_postprocessing = True
        self.supports_batching = True
        self.max_batch_size = 32
    
    def get_import_statements(self) -> str:
        """Get text-specific import statements."""
        return """
# Text-specific imports
import os
import json
import numpy as np
import re
from typing import List, Dict, Union, Any
"""
    
    def get_preprocessing_code(self, task_type: str) -> str:
        """Get text-specific preprocessing code."""
        if task_type == "text_embedding":
            return """
# Convert single string to list for batch processing
if isinstance(text, str):
    batch = [text]
else:
    batch = text

# Tokenize input
inputs = tokenizer(
    batch, 
    return_tensors="pt", 
    padding=True, 
    truncation=True,
    max_length=512
)

# Move inputs to the correct device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type in ["text_generation", "text2text_generation", "causal_lm", "seq2seq_lm"]:
            return """
# Convert single string to list for batch processing
if isinstance(text, str):
    batch = [text]
else:
    batch = text

# Tokenize input
inputs = tokenizer(
    batch, 
    return_tensors="pt", 
    padding=True, 
    truncation=True,
    max_length=kwargs.get("max_input_length", 512)
)

# Move inputs to the correct device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        elif task_type == "masked_lm":
            return """
# Convert single string to list for batch processing
if isinstance(text, str):
    batch = [text]
else:
    batch = text

# Tokenize input with masked tokens
inputs = tokenizer(
    batch, 
    return_tensors="pt", 
    padding=True, 
    truncation=True,
    max_length=512
)

# Add random masking if needed
if kwargs.get("add_masks", False):
    import random
    for i in range(len(inputs["input_ids"])):
        tokens = inputs["input_ids"][i]
        mask_token_id = tokenizer.mask_token_id
        # Randomly mask 15% of tokens
        for j in range(len(tokens)):
            if random.random() < 0.15:
                tokens[j] = mask_token_id

# Move inputs to the correct device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
        else:
            return f"""
# Default preprocessing for {task_type}
if isinstance(text, str):
    batch = [text]
else:
    batch = text

# Tokenize input
inputs = tokenizer(
    batch, 
    return_tensors="pt", 
    padding=True, 
    truncation=True
)

# Move inputs to the correct device
inputs = {k: v.to(device) for k, v in inputs.items()}
"""
    
    def get_postprocessing_code(self, task_type: str) -> str:
        """Get text-specific postprocessing code."""
        if task_type == "text_embedding":
            return """
# Postprocess text embeddings
embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
"""
        elif task_type in ["text_generation", "causal_lm"]:
            return """
# Postprocess generated text
generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
"""
        elif task_type in ["text2text_generation", "seq2seq_lm"]:
            return """
# Postprocess generated text from text-to-text model
generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
"""
        elif task_type == "masked_lm":
            return """
# Postprocess masked token predictions
predicted_token_ids = outputs.logits.argmax(dim=-1)
predicted_texts = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
"""
        else:
            return f"""
# Default postprocessing for {task_type}
result = outputs
"""
    
    def get_result_formatting_code(self, task_type: str) -> str:
        """Get text-specific result formatting code."""
        if task_type == "text_embedding":
            return """
return {"success": True,
    "embeddings": embeddings,
    "dimensions": len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0,
    "device": device,
    "hardware": hardware_label}
"""
        elif task_type in ["text_generation", "causal_lm"]:
            return """
return {"success": True,
    "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
    "all_texts": generated_texts,
    "device": device,
    "hardware": hardware_label}
"""
        elif task_type in ["text2text_generation", "seq2seq_lm"]:
            return """
return {"success": True,
    "translated_text": generated_texts[0] if len(generated_texts) > 0 else "",
    "all_translations": generated_texts,
    "device": device,
    "hardware": hardware_label}
"""
        elif task_type == "masked_lm":
            return """
return {"success": True,
    "predicted_text": predicted_texts[0] if len(predicted_texts) > 0 else "",
    "all_predictions": predicted_texts,
    "device": device,
    "hardware": hardware_label}
"""
        else:
            return f"""
return {{"success": True,
    "result": result,
    "device": device,
    "hardware": hardware_label}}
"""
    
    def get_mock_input_code(self) -> str:
        """Get text-specific mock input code."""
        return """
# Mock text input
mock_input = "This is a mock text input for testing purposes."
"""
    
    def get_mock_output_code(self) -> str:
        """Get text-specific mock output code."""
        return """
# Mock text output
mock_output = "This is a mock text output generated for testing."
"""

    def get_pipeline_utilities(self) -> str:
        """Get text-specific utility functions."""
        return """
# Text pipeline utilities
def clean_text(text):
    # Basic text cleaning
    return text.strip()

def truncate_text(text, max_length=100):
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_generated_text(text, prefix="", suffix=""):
    return f"{prefix}{text}{suffix}"
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check text pipeline compatibility with architecture type."""
        # Text pipeline is compatible with text-based architectures
        return arch_type in [
            "encoder-only",
            "decoder-only",
            "encoder-decoder"
        ]
    
    def is_compatible_with_task(self, task_type: str) -> bool:
        """Check text pipeline compatibility with task type."""
        # Text pipeline is compatible with text-based tasks
        return task_type in [
            "text_embedding",
            "text_generation",
            "text2text_generation",
            "masked_lm",
            "causal_lm",
            "seq2seq_lm"
        ]