#!/usr/bin/env python3
"""
Apple MPS hardware template for IPFS Accelerate Python.

This module implements the hardware template for Apple Silicon devices
using Metal Performance Shaders (MPS).
"""

from typing import Dict, Any, Callable, Tuple, Optional, List, Union
from templates.base_hardware import BaseHardwareTemplate


class AppleHardwareTemplate(BaseHardwareTemplate):
    """Apple MPS hardware template implementation for Apple Silicon."""
    
    def __init__(self):
        """Initialize the Apple MPS hardware template."""
        super().__init__()
        self.hardware_type = "mps"
        self.hardware_name = "Apple MPS"
        self.supports_half_precision = True
        self.supports_quantization = False  # MPS has limited quantization support as of now
        self.supports_dynamic_shapes = True
        self.resource_requirements = {
            "ram_shared": True,  # Apple Silicon shares RAM with GPU
            "recommended_batch_size": 2
        }
    
    def get_import_statements(self) -> str:
        """Get Apple MPS-specific import statements."""
        return """
# Apple MPS-specific imports
import os
import torch
import numpy as np
import platform
"""
    
    def get_hardware_init_code(self, model_class_name: str, task_type: str) -> str:
        """Get Apple MPS-specific initialization code."""
        return f"""
# Initialize model on Apple Silicon with MPS
if platform.system() != "Darwin" or not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
    print("Apple MPS not available, falling back to CPU")
    return self.init_cpu(model_name, "cpu", mps_label.replace("mps", "cpu"))

print(f"Platform: {{platform.system()}} - {{platform.machine()}}")
print(f"PyTorch MPS is available: {{torch.backends.mps.is_available()}}")

# MPS requires a specific approach to loading models - some models may need special handling
device = torch.device("mps")
print(f"Using device: {{device}}")

try:
    # Load model with half precision for better performance on MPS
    use_half = True  # Half precision works well on most Apple Silicon devices
    model = {model_class_name}.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_half else torch.float32,
        cache_dir=cache_dir
    )
    
    # Move model to MPS device
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {{device}}")
except Exception as e:
    print(f"Error loading model on MPS: {{e}}")
    print("Trying again with full precision...")
    
    try:
        # Try again with full precision
        model = {model_class_name}.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            cache_dir=cache_dir
        )
        model = model.to(device)
        model.eval()
        print("Model loaded successfully with full precision")
    except Exception as e2:
        print(f"Error loading model with full precision: {{e2}}")
        print("Falling back to CPU")
        return self.init_cpu(model_name, "cpu", mps_label.replace("mps", "cpu"))
"""
    
    def get_handler_creation_code(self, model_class_name: str, task_type: str) -> str:
        """Get Apple MPS-specific handler creation code."""
        return f"""
# Create Apple MPS handler function
handler = self.create_apple_{task_type}_endpoint_handler(
    endpoint_model=model_name,
    device="mps",
    hardware_label=mps_label,
    endpoint=model,
    tokenizer=tokenizer
)
"""
    
    def get_inference_code(self, task_type: str) -> str:
        """Get Apple MPS-specific inference code."""
        if task_type == "text_embedding":
            return """
# Apple MPS inference for text embedding
# Ensure inputs are on the right device
for key, value in inputs.items():
    if hasattr(value, "to") and not isinstance(value, (str, int, float, bool)):
        inputs[key] = value.to("mps")

outputs = endpoint(**inputs)
# Move results back to CPU for post-processing
embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
"""
        elif task_type == "text_generation" or task_type == "text2text_generation":
            return """
# Apple MPS inference for text generation
# Ensure inputs are on the right device
for key, value in inputs.items():
    if hasattr(value, "to") and not isinstance(value, (str, int, float, bool)):
        inputs[key] = value.to("mps")

generate_kwargs = {
    "max_new_tokens": kwargs.get("max_new_tokens", 100),
    "do_sample": kwargs.get("do_sample", False),
    "temperature": kwargs.get("temperature", 1.0),
    "top_p": kwargs.get("top_p", 0.9),
    "top_k": kwargs.get("top_k", 0)
}

output_ids = endpoint.generate(
    **inputs,
    **generate_kwargs
)
# Move results back to CPU for decoding
output_ids = output_ids.cpu()
generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
"""
        elif task_type == "image_classification":
            return """
# Apple MPS inference for image classification
# Ensure inputs are on the right device
for key, value in inputs.items():
    if hasattr(value, "to") and not isinstance(value, (str, int, float, bool)):
        inputs[key] = value.to("mps")

outputs = endpoint(**inputs)
logits = outputs.logits
# Move results back to CPU for post-processing
scores = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy().tolist()
"""
        else:
            return f"""
# Apple MPS inference for {task_type}
# Ensure inputs are on the right device
for key, value in inputs.items():
    if hasattr(value, "to") and not isinstance(value, (str, int, float, bool)):
        inputs[key] = value.to("mps")

outputs = endpoint(**inputs)
# Move results back to CPU if needed
if hasattr(outputs, "cpu"):
    outputs = outputs.cpu()
"""
    
    def get_cleanup_code(self) -> str:
        """Get Apple MPS-specific cleanup code."""
        return """
# Apple MPS cleanup
# MPS doesn't have an explicit cache clearing mechanism like CUDA,
# but we can help the garbage collector
import gc
gc.collect()

# For PyTorch, try to empty any caches we can
if hasattr(torch.cuda, "empty_cache"):
    torch.cuda.empty_cache()  # No-op on MPS but doesn't hurt
"""
    
    def get_mock_code(self, model_class_name: str, task_type: str) -> str:
        """Get Apple MPS-specific mock implementation code."""
        return """
# Apple MPS mock implementation
from unittest.mock import MagicMock
mock_model = MagicMock()
mock_model.to.return_value = mock_model  # Mock the to() method
mock_model.eval.return_value = mock_model  # Mock the eval() method

# Simulate that we're on an Apple device
mock_model.device = "mps"
"""
    
    def get_hardware_detection_code(self) -> str:
        """Get Apple MPS-specific hardware detection code."""
        return """
# Apple MPS availability check
def is_available():
    try:
        import platform
        import torch
        
        # Check if we're on macOS
        is_mac = platform.system() == "Darwin"
        
        # Check if we're on Apple Silicon (M1/M2/etc.)
        is_apple_silicon = platform.machine() == "arm64"
        
        # Check if PyTorch has MPS support
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        
        if is_mac and is_apple_silicon and has_mps:
            print("Apple MPS is available")
            return True
        
        if is_mac and is_apple_silicon and not has_mps:
            print("Running on Apple Silicon but MPS is not available in PyTorch")
        elif is_mac and not is_apple_silicon:
            print("Running on Mac but not on Apple Silicon")
        elif not is_mac:
            print(f"Not running on Mac (detected {platform.system()})")
        
        return False
    except ImportError:
        return False
    except Exception as e:
        print(f"Error checking Apple MPS availability: {e}")
        return False
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check Apple MPS compatibility with architecture type."""
        # Apple MPS has some limitations with larger models
        incompatible_archs = ["mixture-of-experts"]  # MoE models typically exceed Apple device memory
        return arch_type not in incompatible_archs
    
    def get_fallback_hardware(self) -> str:
        """Get the fallback hardware type if Apple MPS is not available."""
        return "cpu"