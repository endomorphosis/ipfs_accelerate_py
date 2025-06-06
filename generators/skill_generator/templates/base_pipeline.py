#!/usr/bin/env python3
"""
Base pipeline template for IPFS Accelerate Python.

This module defines the base interface for pipeline-specific templates.
Each input/output type (text, image, audio, etc.) should implement these methods
to provide a consistent interface for processing model inputs and outputs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Tuple, Optional, List, Union


class BasePipelineTemplate(ABC):
    """Base class for pipeline-specific templates."""
    
    def __init__(self):
        """Initialize the pipeline template."""
        self.pipeline_type = "base"
        self.input_type = "unknown"
        self.output_type = "unknown"
        self.requires_preprocessing = False
        self.requires_postprocessing = False
        self.supports_batching = False
        self.max_batch_size = 1
    
    @abstractmethod
    def get_import_statements(self) -> str:
        """
        Get the import statements required for this pipeline.
        
        Returns:
            String containing import statements
        """
        return """
# Base imports for pipeline
import os
import json
import numpy as np
"""
    
    @abstractmethod
    def get_preprocessing_code(self, task_type: str) -> str:
        """
        Get the preprocessing code for this pipeline.
        
        Args:
            task_type: The task type (text_embedding, text_generation, etc.)
            
        Returns:
            String containing preprocessing code
        """
        pass
    
    @abstractmethod
    def get_postprocessing_code(self, task_type: str) -> str:
        """
        Get the postprocessing code for this pipeline.
        
        Args:
            task_type: The task type (text_embedding, text_generation, etc.)
            
        Returns:
            String containing postprocessing code
        """
        pass
    
    @abstractmethod
    def get_result_formatting_code(self, task_type: str) -> str:
        """
        Get the result formatting code for this pipeline.
        
        Args:
            task_type: The task type (text_embedding, text_generation, etc.)
            
        Returns:
            String containing result formatting code
        """
        pass
    
    @abstractmethod
    def get_mock_input_code(self) -> str:
        """
        Get code for creating mock inputs for this pipeline.
        
        Returns:
            String containing mock input code
        """
        pass
    
    @abstractmethod
    def get_mock_output_code(self) -> str:
        """
        Get code for creating mock outputs for this pipeline.
        
        Returns:
            String containing mock output code
        """
        pass

    def get_pipeline_utilities(self) -> str:
        """
        Get utility functions for this pipeline.
        
        Returns:
            String containing utility functions
        """
        return """
# Base pipeline utilities
def format_result(result):
    return result
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """
        Check if this pipeline is compatible with the given architecture type.
        
        Args:
            arch_type: The architecture type
            
        Returns:
            True if compatible, False otherwise
        """
        return True
    
    def is_compatible_with_task(self, task_type: str) -> bool:
        """
        Check if this pipeline is compatible with the given task type.
        
        Args:
            task_type: The task type
            
        Returns:
            True if compatible, False otherwise
        """
        return True


# Example implementation outline for text pipeline
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
    
    def get_postprocessing_code(self, task_type: str) -> str:
        """Get text-specific postprocessing code."""
        if task_type == "text_embedding":
            return """
# Postprocess text embeddings
embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
"""
        elif task_type == "text_generation" or task_type == "text2text_generation":
            return """
# Postprocess generated text
generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
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
    "device": device,
    "hardware": hardware_label}
"""
        elif task_type == "text_generation" or task_type == "text2text_generation":
            return """
return {"success": True,
    "generated_text": generated_texts[0] if len(generated_texts) > 0 else "",
    "all_texts": generated_texts,
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
mock_input = "This is a mock text input"
"""
    
    def get_mock_output_code(self) -> str:
        """Get text-specific mock output code."""
        return """
# Mock text output
mock_output = "This is a mock text output"
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