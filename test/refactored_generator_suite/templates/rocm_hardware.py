#!/usr/bin/env python3
"""
ROCm hardware template for IPFS Accelerate Python.

This module implements the hardware template for AMD GPUs using ROCm.
"""

from typing import Dict, Any, Callable, Tuple, Optional, List, Union
from templates.base_hardware import BaseHardwareTemplate


class RocmHardwareTemplate(BaseHardwareTemplate):
    """ROCm hardware template implementation for AMD GPUs."""
    
    def __init__(self):
        """Initialize the ROCm hardware template."""
        super().__init__()
        self.hardware_type = "rocm"
        self.hardware_name = "ROCm"
        self.supports_half_precision = True
        self.supports_quantization = True
        self.supports_dynamic_shapes = True
        self.resource_requirements = {
            "vram_minimum": 2048,  # 2GB minimum VRAM
            "recommended_batch_size": 4
        }
    
    def get_import_statements(self) -> str:
        """Get ROCm-specific import statements."""
        return """
# ROCm-specific imports
import os
import torch
import numpy as np
"""
    
    def get_hardware_init_code(self, model_class_name: str, task_type: str) -> str:
        """Get ROCm-specific initialization code."""
        return f"""
# Initialize model on ROCm (AMD GPU)
# Check for environment variables 
visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", None) or os.environ.get("CUDA_VISIBLE_DEVICES", None)
if visible_devices is not None:
    print(f"Using ROCm visible devices: {{visible_devices}}")

# Get the total GPU memory for logging purposes
total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
print(f"AMD GPU memory: {{total_mem:.2f}} GB")

# Determine if we should use half precision based on GPU capabilities
try:
    # Try to create a small tensor in half precision as a test
    test_tensor = torch.ones((10, 10), dtype=torch.float16, device="cuda")
    del test_tensor
    use_half = True
    print("Half precision is supported on this AMD GPU")
except Exception as e:
    use_half = False
    print(f"Half precision not supported on this AMD GPU: {{e}}")

model = {model_class_name}.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if use_half else torch.float32,
    device_map="auto",  # ROCm uses the same device map mechanism as CUDA
    cache_dir=cache_dir
)
model.eval()

# Log device mapping
if hasattr(model, "hf_device_map"):
    print(f"Device map: {{model.hf_device_map}}")
"""
    
    def get_handler_creation_code(self, model_class_name: str, task_type: str) -> str:
        """Get ROCm-specific handler creation code."""
        return f"""
# Create ROCm handler function
handler = self.create_rocm_{task_type}_endpoint_handler(
    endpoint_model=model_name,
    device="cuda",  # ROCm uses "cuda" as the device name
    hardware_label=rocm_label,
    endpoint=model,
    tokenizer=tokenizer
)
"""
    
    def get_inference_code(self, task_type: str) -> str:
        """Get ROCm-specific inference code."""
        if task_type == "text_embedding":
            return """
# ROCm inference for text embedding
outputs = endpoint(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
"""
        elif task_type == "text_generation" or task_type == "text2text_generation":
            return """
# ROCm inference for text generation
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
generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
"""
        elif task_type == "image_classification":
            return """
# ROCm inference for image classification
outputs = endpoint(**inputs)
logits = outputs.logits
scores = torch.nn.functional.softmax(logits, dim=-1)
predictions = scores.cpu().numpy().tolist()
"""
        elif task_type == "image_to_text":
            return """
# ROCm inference for image to text
output_ids = endpoint.generate(**inputs, max_length=kwargs.get("max_length", 100))
predicted_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
"""
        elif task_type == "audio_classification":
            return """
# ROCm inference for audio classification
outputs = endpoint(**inputs)
logits = outputs.logits
scores = torch.nn.functional.softmax(logits, dim=-1)
predictions = scores.cpu().numpy().tolist()
"""
        else:
            return f"""
# ROCm inference for {task_type}
outputs = endpoint(**inputs)
"""
    
    def get_cleanup_code(self) -> str:
        """Get ROCm-specific cleanup code."""
        return """
# ROCm cleanup
torch.cuda.empty_cache()  # Works the same as CUDA for ROCm
"""
    
    def get_mock_code(self, model_class_name: str, task_type: str) -> str:
        """Get ROCm-specific mock implementation code."""
        return """
# ROCm mock implementation
from unittest.mock import MagicMock
mock_model = MagicMock()
mock_model.to.return_value = mock_model  # Mock the to() method
mock_model.eval.return_value = mock_model  # Mock the eval() method
mock_model.device = "cuda"  # Pretend we're on an AMD GPU
"""
    
    def get_hardware_detection_code(self) -> str:
        """Get ROCm-specific hardware detection code."""
        return """
# ROCm availability check
def is_available():
    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            # Get ROCM device count
            device_count = torch.cuda.device_count()
            print(f"Found {device_count} ROCm devices")
            return device_count > 0
        elif torch.cuda.is_available():
            # Check if it's actually ROCm
            device_props = torch.cuda.get_device_properties(0)
            device_name = device_props.name
            
            # ROCm devices typically have AMD in the name
            if "AMD" in device_name or "Radeon" in device_name:
                print(f"Found ROCm device: {device_name}")
                return True
                
        return False
    except ImportError:
        return False
    except Exception as e:
        print(f"Error checking ROCm availability: {e}")
        return False
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check ROCm compatibility with architecture type."""
        # ROCm is compatible with most architectures
        # except very specific ones that might need special attention
        incompatible_archs = ["mixture-of-experts"]  # MoE models might be too large for ROCm devices
        return arch_type not in incompatible_archs
    
    def get_fallback_hardware(self) -> str:
        """Get the fallback hardware type if ROCm is not available."""
        return "cpu"