#!/usr/bin/env python3
"""
MPS (Metal Performance Shaders) hardware template for IPFS Accelerate Python.

This module implements the hardware template for Apple Silicon GPUs using MPS.
"""

from typing import Dict, Any, Callable, Tuple, Optional, List, Union
from templates.base_hardware import BaseHardwareTemplate


class MPSHardwareTemplate(BaseHardwareTemplate):
    """MPS hardware template implementation for Apple Silicon GPUs."""
    
    def __init__(self):
        """Initialize the MPS hardware template."""
        super().__init__()
        self.hardware_type = "mps"
        self.hardware_name = "Apple MPS"
        self.supports_half_precision = True
        self.supports_quantization = False  # Limited quantization support
        self.supports_dynamic_shapes = True
        self.resource_requirements = {
            "vram_minimum": 2048,  # 2GB minimum VRAM
            "recommended_batch_size": 4
        }
    
    def get_import_statements(self) -> str:
        """Get MPS-specific import statements."""
        return """
# MPS-specific imports
import os
import torch
import numpy as np
"""
    
    def get_hardware_init_code(self, model_class_name: str, task_type: str) -> str:
        """Get MPS-specific initialization code."""
        return f"""
# Initialize model on MPS (Apple Silicon GPU)
# Check if MPS is available
if not torch.backends.mps.is_available():
    print("MPS is not available. Using CPU instead.")
    device = "cpu"
    use_half = False
else:
    device = "mps"
    
    # Determine if we should use half precision
    # Some models may not support half precision on MPS
    try:
        # Try to create a small tensor in half precision as a test
        test_tensor = torch.ones((10, 10), dtype=torch.float16, device=device)
        del test_tensor
        use_half = True
        print("Half precision is supported on this Apple Silicon device")
    except Exception as e:
        use_half = False
        print(f"Half precision not supported on this Apple Silicon device: {{e}}")

# Log the device being used
print(f"Using device: {{device}}")

# Load model with appropriate settings for MPS
model = {model_class_name}.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if use_half else torch.float32,
    device_map=device,
    cache_dir=cache_dir
)
model.eval()
"""
    
    def get_handler_creation_code(self, model_class_name: str, task_type: str) -> str:
        """Get MPS-specific handler creation code."""
        return f"""
# Create MPS handler function
handler = self.create_mps_{task_type}_endpoint_handler(
    endpoint_model=model_name,
    device=device,
    hardware_label=mps_label,
    endpoint=model,
    tokenizer=tokenizer
)
"""
    
    def get_inference_code(self, task_type: str) -> str:
        """Get MPS-specific inference code."""
        if task_type == "text_embedding":
            return """
# MPS inference for text embedding
# MPS benefits from running in no_grad mode
with torch.no_grad():
    outputs = endpoint(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
"""
        elif task_type == "text_generation" or task_type == "text2text_generation":
            return """
# MPS inference for text generation
# Create generation config
generate_kwargs = {
    "max_new_tokens": kwargs.get("max_new_tokens", 100),
    "do_sample": kwargs.get("do_sample", False),
    "temperature": kwargs.get("temperature", 1.0),
    "top_p": kwargs.get("top_p", 0.9),
    "top_k": kwargs.get("top_k", 0)
}

# Run generation in no_grad mode
with torch.no_grad():
    output_ids = endpoint.generate(
        **inputs,
        **generate_kwargs
    )
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
"""
        elif task_type == "image_classification":
            return """
# MPS inference for image classification
with torch.no_grad():
    outputs = endpoint(**inputs)
    logits = outputs.logits
    scores = torch.nn.functional.softmax(logits, dim=-1)
    predictions = scores.cpu().numpy().tolist()
"""
        elif task_type == "image_to_text":
            return """
# MPS inference for image to text
with torch.no_grad():
    output_ids = endpoint.generate(**inputs, max_length=kwargs.get("max_length", 100))
    predicted_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
"""
        elif task_type == "audio_classification":
            return """
# MPS inference for audio classification
with torch.no_grad():
    outputs = endpoint(**inputs)
    logits = outputs.logits
    scores = torch.nn.functional.softmax(logits, dim=-1)
    predictions = scores.cpu().numpy().tolist()
"""
        else:
            return f"""
# MPS inference for {task_type}
with torch.no_grad():
    outputs = endpoint(**inputs)
"""
    
    def get_cleanup_code(self) -> str:
        """Get MPS-specific cleanup code."""
        return """
# MPS cleanup - clear memory
import gc
gc.collect()
torch.mps.empty_cache()  # MPS-specific cache clearing
"""
    
    def get_mock_code(self, model_class_name: str, task_type: str) -> str:
        """Get MPS-specific mock implementation code."""
        return """
# MPS mock implementation
from unittest.mock import MagicMock
mock_model = MagicMock()
mock_model.to.return_value = mock_model  # Mock the to() method
mock_model.eval.return_value = mock_model  # Mock the eval() method
mock_model.device = "mps"  # Pretend we're on Apple Silicon
"""
    
    def get_hardware_detection_code(self) -> str:
        """Get MPS-specific hardware detection code."""
        return """
# MPS availability check
def is_available():
    try:
        import torch
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        return False
    except ImportError:
        return False
    except Exception as e:
        print(f"Error checking MPS availability: {e}")
        return False
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check MPS compatibility with architecture type."""
        # MPS is compatible with most architectures except very large ones
        # that would exceed memory limitations
        incompatible_archs = [
            "mixture-of-experts",  # MoE models might be too large for MPS devices
            "diffusion"  # Some diffusion models require too much memory for MPS
        ]
        return arch_type not in incompatible_archs
    
    def get_fallback_hardware(self) -> str:
        """Get the fallback hardware type if MPS is not available."""
        return "cpu"