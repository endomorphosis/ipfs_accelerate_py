#!/usr/bin/env python3
"""
Qualcomm hardware template for IPFS Accelerate Python.

This module implements the hardware template for Qualcomm Neural Processing Units (NPUs).
"""

from typing import Dict, Any, Callable, Tuple, Optional, List, Union
from templates.base_hardware import BaseHardwareTemplate


class QualcommHardwareTemplate(BaseHardwareTemplate):
    """Qualcomm hardware template implementation for Qualcomm NPUs."""
    
    def __init__(self):
        """Initialize the Qualcomm hardware template."""
        super().__init__()
        self.hardware_type = "qnn"
        self.hardware_name = "Qualcomm"
        self.supports_half_precision = True
        self.supports_quantization = True
        self.supports_dynamic_shapes = False  # QNN often prefers fixed shapes
        self.resource_requirements = {
            "quantization_aware": True,  # Qualcomm devices benefit significantly from quantization
            "fixed_batch_size": True,    # QNN works best with fixed batch sizes
            "fixed_shapes": True,        # QNN works best with fixed input shapes
            "memory_efficient": True     # QNN is optimized for memory efficiency
        }
    
    def get_import_statements(self) -> str:
        """Get Qualcomm-specific import statements."""
        return """
# Qualcomm-specific imports
import os
import sys
import json
import torch
import numpy as np
import time
from pathlib import Path

try:
    from qnnpy import PyQnnManager
    from qnnpy.models import QnnModel
    from qnnpy.utils import ModelConverter
    HAS_QNN = True
except ImportError:
    HAS_QNN = False
    print("Qualcomm QNN imports failed, falling back to PyTorch")
"""
    
    def get_hardware_init_code(self, model_class_name: str, task_type: str) -> str:
        """Get Qualcomm-specific initialization code."""
        return f"""
# Initialize model for Qualcomm NPU
if not HAS_QNN:
    print("Qualcomm QNN SDK not available, falling back to CPU")
    return self.init_cpu(model_name, "cpu", qualcomm_label.replace("qualcomm", "cpu"))

print("Initializing Qualcomm NPU backend...")

try:
    # Initialize QNN manager
    self.qnn_manager = PyQnnManager()
    
    # Get Qualcomm device info
    qnn_devices = self.qnn_manager.available_devices()
    if len(qnn_devices) == 0:
        print("No Qualcomm NPU devices found, falling back to CPU")
        return self.init_cpu(model_name, "cpu", qualcomm_label.replace("qualcomm", "cpu"))
    
    print(f"Found {{len(qnn_devices)}} Qualcomm NPU devices: {{qnn_devices}}")
    
    # Select the first available device or use specified device
    qnn_device = os.environ.get("QNN_DEVICE", qnn_devices[0])
    if qnn_device not in qnn_devices:
        print(f"Warning: Specified QNN_DEVICE {{qnn_device}} not found, using {{qnn_devices[0]}} instead")
        qnn_device = qnn_devices[0]
    
    print(f"Using Qualcomm NPU device: {{qnn_device}}")
    
    # Load tokenizer using standard HuggingFace methods
    tokenizer = self.transformers.AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    # For Qualcomm, we need to convert the model to QNN format
    # Create model cache directory with model name hash to avoid path length issues
    import hashlib
    model_hash = hashlib.md5(model_name.encode()).hexdigest()
    model_path = os.path.join(cache_dir, f"qnn_{{model_hash}}")
    os.makedirs(model_path, exist_ok=True)
    
    # Check if model is already converted
    qnn_model_path = os.path.join(model_path, "model.qnn")
    if os.path.exists(qnn_model_path):
        print(f"Loading pre-converted QNN model from {{qnn_model_path}}")
        model = QnnModel.load(qnn_model_path, device=qnn_device)
    else:
        print(f"Converting {{model_name}} to QNN format...")
        
        # First load the PyTorch model
        pt_model = {model_class_name}.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            cache_dir=cache_dir
        )
        
        # Convert model to QNN format
        print("Converting PyTorch model to QNN format...")
        
        # Detect input shapes based on task_type
        if '{task_type}' == 'text_embedding' or '{task_type}' == 'text_generation' or '{task_type}' == 'text2text_generation':
            # Text models usually have sequence input
            input_shapes = {{"input_ids": [1, 64], "attention_mask": [1, 64]}}
        elif '{task_type}' == 'image_classification':
            # Image models usually have image input
            input_shapes = {{"pixel_values": [1, 3, 224, 224]}}
        else:
            # Default input shapes
            input_shapes = {{"input_ids": [1, 64]}}
        
        try:
            # Use ModelConverter to convert PyTorch model to QNN
            converter = ModelConverter(pt_model, input_shapes=input_shapes)
            model = converter.convert(target_device=qnn_device)
            
            # Save the converted model for future use
            model.save(qnn_model_path)
            print(f"Saved converted QNN model to {{qnn_model_path}}")
        except Exception as e:
            print(f"Error converting model to QNN format: {{e}}")
            print("Using PyTorch model with QNN wrappers instead")
            
            # Create a wrapped PyTorch model that mimics QNN interface
            model = self._create_pytorch_to_qnn_wrapper(pt_model, task_type)
    
    # Extract model metadata for QNN
    model_metadata = {{
        "device": qnn_device,
        "task_type": '{task_type}',
        "is_native_qnn": hasattr(model, "qnn_backend") and model.qnn_backend is not None
    }}
    
    # Log QNN model info
    print(f"QNN model initialized on {{qnn_device}}")
    if hasattr(model, "qnn_backend") and model.qnn_backend is not None:
        print(f"Using native QNN backend: {{model.qnn_backend}}")
    else:
        print("Using wrapped PyTorch model with QNN interface")
except Exception as e:
    print(f"Error initializing Qualcomm QNN: {{e}}")
    print("Falling back to CPU")
    return self.init_cpu(model_name, "cpu", qualcomm_label.replace("qualcomm", "cpu"))
"""
    
    def get_handler_creation_code(self, model_class_name: str, task_type: str) -> str:
        """Get Qualcomm-specific handler creation code."""
        return f"""
# Create Qualcomm handler function
handler = self.create_qualcomm_{task_type}_endpoint_handler(
    endpoint_model=model_name,
    device="qnn",
    hardware_label=qualcomm_label,
    endpoint=model,
    tokenizer=tokenizer,
    model_metadata=model_metadata
)
"""
    
    def get_inference_code(self, task_type: str) -> str:
        """Get Qualcomm-specific inference code."""
        if task_type == "text_embedding":
            return """
# Qualcomm inference for text embedding
is_native_qnn = model_metadata.get("is_native_qnn", False)

try:
    # Preprocess according to QNN requirements
    if is_native_qnn:
        # QNN models may need fixed shapes
        if isinstance(inputs["input_ids"], torch.Tensor):
            max_length = 64  # Fixed length for QNN
            if inputs["input_ids"].shape[1] > max_length:
                # Truncate
                inputs["input_ids"] = inputs["input_ids"][:, :max_length]
                inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
            elif inputs["input_ids"].shape[1] < max_length:
                # Pad
                pad_length = max_length - inputs["input_ids"].shape[1]
                inputs["input_ids"] = torch.nn.functional.pad(
                    inputs["input_ids"], (0, pad_length), value=tokenizer.pad_token_id
                )
                inputs["attention_mask"] = torch.nn.functional.pad(
                    inputs["attention_mask"], (0, pad_length), value=0
                )
        
        # Convert tensors to numpy arrays for QNN
        np_inputs = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                    for k, v in inputs.items()}
        
        # Run QNN inference
        outputs = endpoint.qnn_forward(**np_inputs)
        
        # Process outputs - typically the last hidden state from embedding models
        if isinstance(outputs, dict) and "last_hidden_state" in outputs:
            # If output is a dictionary with expected keys
            embeddings = outputs["last_hidden_state"].mean(axis=1).tolist()
        elif isinstance(outputs, np.ndarray) and len(outputs.shape) >= 2:
            # If output is a numpy array, assume it's the hidden states
            embeddings = outputs.mean(axis=1).tolist()
        else:
            # Fallback - assume outputs are already the embeddings
            embeddings = outputs.tolist() if isinstance(outputs, np.ndarray) else outputs
    else:
        # Fallback to PyTorch model with standard processing
        outputs = endpoint(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
except Exception as e:
    print(f"Error during QNN embedding inference: {e}")
    # Fallback to PyTorch
    outputs = endpoint(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
"""
        elif task_type == "text_generation" or task_type == "text2text_generation":
            return """
# Qualcomm inference for text generation
is_native_qnn = model_metadata.get("is_native_qnn", False)

try:
    # Prepare generation parameters
    generate_kwargs = {
        "max_new_tokens": kwargs.get("max_new_tokens", 100),
        "do_sample": kwargs.get("do_sample", False),
        "temperature": kwargs.get("temperature", 1.0),
        "top_p": kwargs.get("top_p", 0.9),
        "top_k": kwargs.get("top_k", 0)
    }
    
    if is_native_qnn:
        # QNN models may need fixed shapes and specialized handling
        if isinstance(inputs["input_ids"], torch.Tensor):
            # Convert tensors to numpy arrays for QNN
            np_inputs = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                        for k, v in inputs.items()}
        else:
            np_inputs = inputs
        
        # QNN generation
        if hasattr(endpoint, 'qnn_generate'):
            output_ids = endpoint.qnn_generate(**np_inputs, **generate_kwargs)
            
            # Convert output IDs to tokens
            if isinstance(output_ids, np.ndarray):
                # If output is already a numpy array, use it directly
                generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            else:
                # Otherwise, try to adapt the output
                generated_texts = tokenizer.batch_decode(
                    torch.tensor(output_ids), 
                    skip_special_tokens=True
                )
        else:
            # Fallback to standard forward with sequential generation
            output_texts = []
            for i in range(generate_kwargs.get("max_new_tokens", 100)):
                # QNN forward step
                step_outputs = endpoint.qnn_forward(**np_inputs)
                
                # Extract logits
                if isinstance(step_outputs, dict) and "logits" in step_outputs:
                    logits = step_outputs["logits"]
                else:
                    logits = step_outputs
                
                # Get next token
                next_token_id = np.argmax(logits[:, -1, :], axis=-1)
                
                # Append to outputs
                if isinstance(np_inputs["input_ids"], np.ndarray):
                    np_inputs["input_ids"] = np.concatenate(
                        [np_inputs["input_ids"], next_token_id.reshape(-1, 1)], 
                        axis=1
                    )
                    # Also update attention mask
                    np_inputs["attention_mask"] = np.concatenate(
                        [np_inputs["attention_mask"], np.ones((np_inputs["attention_mask"].shape[0], 1))],
                        axis=1
                    )
            
            # Decode the full sequence
            generated_texts = tokenizer.batch_decode(
                np_inputs["input_ids"], 
                skip_special_tokens=True
            )
    else:
        # Standard PyTorch generation
        output_ids = endpoint.generate(**inputs, **generate_kwargs)
        generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
except Exception as e:
    print(f"Error during QNN text generation: {e}")
    # Fallback to PyTorch
    output_ids = endpoint.generate(**inputs, **{
        "max_new_tokens": kwargs.get("max_new_tokens", 100),
        "do_sample": kwargs.get("do_sample", False),
        "temperature": kwargs.get("temperature", 1.0),
        "top_p": kwargs.get("top_p", 0.9),
        "top_k": kwargs.get("top_k", 0)
    })
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
"""
        elif task_type == "image_classification":
            return """
# Qualcomm inference for image classification
is_native_qnn = model_metadata.get("is_native_qnn", False)

try:
    if is_native_qnn:
        # QNN models need specialized handling
        if isinstance(inputs["pixel_values"], torch.Tensor):
            # QNN often requires fixed input shapes
            # Resize if necessary
            if hasattr(endpoint, "input_shape") and endpoint.input_shape:
                required_shape = endpoint.input_shape.get("pixel_values")
                if required_shape and inputs["pixel_values"].shape != required_shape:
                    from torchvision.transforms import functional as F
                    if len(required_shape) == 4:  # [batch, channels, height, width]
                        inputs["pixel_values"] = F.resize(
                            inputs["pixel_values"],
                            [required_shape[2], required_shape[3]]
                        )
            
            # Convert to numpy for QNN
            np_inputs = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                        for k, v in inputs.items()}
        else:
            np_inputs = inputs
        
        # Run QNN inference
        outputs = endpoint.qnn_forward(**np_inputs)
        
        # Process outputs
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            logits = outputs
        
        # Apply softmax to get probabilities
        from scipy.special import softmax
        scores = softmax(logits, axis=-1).tolist()
        
        # Get class IDs
        predicted_class_ids = np.argmax(logits, axis=-1).tolist()
    else:
        # Standard PyTorch
        outputs = endpoint(**inputs)
        logits = outputs.logits
        scores = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy().tolist()
        predicted_class_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        
    # If model has id2label mapping, get class names
    if hasattr(endpoint, 'config') and hasattr(endpoint.config, 'id2label'):
        predicted_labels = [endpoint.config.id2label.get(idx, f"CLASS_{idx}") 
                           for idx in predicted_class_ids]
    else:
        predicted_labels = [f"CLASS_{idx}" for idx in predicted_class_ids]
except Exception as e:
    print(f"Error during QNN image classification: {e}")
    # Fallback to PyTorch
    outputs = endpoint(**inputs)
    logits = outputs.logits
    scores = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy().tolist()
    predicted_class_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    
    # Get class names if available
    if hasattr(endpoint, 'config') and hasattr(endpoint.config, 'id2label'):
        predicted_labels = [endpoint.config.id2label.get(idx, f"CLASS_{idx}") 
                           for idx in predicted_class_ids]
    else:
        predicted_labels = [f"CLASS_{idx}" for idx in predicted_class_ids]
"""
        else:
            return f"""
# Qualcomm inference for {task_type}
is_native_qnn = model_metadata.get("is_native_qnn", False)

try:
    if is_native_qnn:
        # Convert PyTorch tensors to numpy arrays for QNN
        np_inputs = {{}}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                np_inputs[k] = v.cpu().numpy()
            else:
                np_inputs[k] = v
        
        # Run QNN inference
        if hasattr(endpoint, 'qnn_forward'):
            outputs = endpoint.qnn_forward(**np_inputs)
        else:
            # If no specific QNN forward method, use standard
            outputs = endpoint(**inputs)
    else:
        # Standard PyTorch inference
        outputs = endpoint(**inputs)
    
    # Process outputs based on structure
    if isinstance(outputs, dict):
        # Dict output - process each value
        processed_outputs = {{}}
        for k, v in outputs.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                # Convert tensors/arrays to lists
                processed_outputs[k] = v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v.tolist()
            else:
                processed_outputs[k] = v
        result = processed_outputs
    elif isinstance(outputs, (torch.Tensor, np.ndarray)):
        # Tensor output - convert to list
        result = outputs.cpu().numpy().tolist() if isinstance(outputs, torch.Tensor) else outputs.tolist()
    else:
        # Unknown output type
        result = {{"raw_output": str(outputs)}}
except Exception as e:
    print(f"Error during QNN inference: {{e}}")
    # Fallback to PyTorch
    outputs = endpoint(**inputs)
    
    # Process outputs for the fallback case
    if isinstance(outputs, dict):
        # Dict output - process each value
        processed_outputs = {{}}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                # Convert tensors to lists
                processed_outputs[k] = v.cpu().numpy().tolist()
            else:
                processed_outputs[k] = v
        result = processed_outputs
    elif isinstance(outputs, torch.Tensor):
        # Tensor output - convert to list
        result = outputs.cpu().numpy().tolist()
    else:
        # Unknown output type
        result = {{"raw_output": str(outputs)}}
"""
    
    def get_cleanup_code(self) -> str:
        """Get Qualcomm-specific cleanup code."""
        return """
# Qualcomm cleanup
# Release QNN resources

# If we have a QNN model, explicitly delete it
if 'endpoint' in locals() and hasattr(endpoint, 'qnn_backend') and endpoint.qnn_backend is not None:
    try:
        del endpoint.qnn_backend
    except:
        pass

# If we have a QNN manager instance, release its resources
if hasattr(self, 'qnn_manager'):
    try:
        del self.qnn_manager
    except:
        pass

# Run garbage collection to ensure resources are freed
import gc
gc.collect()

# Unset any CUDA tensors
if 'torch' in globals() or 'torch' in locals():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
"""
    
    def get_mock_code(self, model_class_name: str, task_type: str) -> str:
        """Get Qualcomm-specific mock implementation code."""
        return """
# Qualcomm mock implementation
from unittest.mock import MagicMock
mock_model = MagicMock()

# Add Qualcomm-specific attributes and methods
mock_model.qnn_forward = MagicMock()
mock_model.qnn_generate = MagicMock()
mock_model.qnn_backend = "mock_qnn_backend"
mock_model.input_shape = {"pixel_values": [1, 3, 224, 224]} if "image" in task_type else {"input_ids": [1, 64]}

# Configure the qnn_forward method to return appropriate mock outputs
def mock_qnn_forward(**kwargs):
    batch_size = 1
    if "input_ids" in kwargs:
        batch_size = kwargs["input_ids"].shape[0] if hasattr(kwargs["input_ids"], "shape") else 1
        seq_length = kwargs["input_ids"].shape[1] if hasattr(kwargs["input_ids"], "shape") else 10
    elif "pixel_values" in kwargs:
        batch_size = kwargs["pixel_values"].shape[0] if hasattr(kwargs["pixel_values"], "shape") else 1
    
    # Create dummy outputs based on task type
    import numpy as np
    if task_type == "text_embedding":
        # Return embedding tensor
        return np.random.rand(batch_size, seq_length, 768).astype(np.float32)
    elif task_type == "text_generation" or task_type == "text2text_generation":
        # Return logits tensor for next token prediction
        vocab_size = 50000
        return {"logits": np.random.rand(batch_size, seq_length, vocab_size).astype(np.float32)}
    elif task_type == "image_classification":
        # Return classification logits
        num_classes = 1000
        return {"logits": np.random.rand(batch_size, num_classes).astype(np.float32)}
    else:
        # Generic output
        return {"outputs": np.random.rand(batch_size, 768).astype(np.float32)}

mock_model.qnn_forward.side_effect = mock_qnn_forward

# Configure the qnn_generate method for text generation
def mock_qnn_generate(**kwargs):
    import numpy as np
    batch_size = 1
    if "input_ids" in kwargs:
        batch_size = kwargs["input_ids"].shape[0] if hasattr(kwargs["input_ids"], "shape") else 1
    
    # Get max length from generation parameters
    max_length = kwargs.get("max_new_tokens", 20) + 10  # Assume input is 10 tokens
    
    # Create dummy token ids - simple increasing sequence
    token_ids = np.array([[i + j * 100 for i in range(max_length)] for j in range(batch_size)])
    return token_ids

mock_model.qnn_generate.side_effect = mock_qnn_generate

# Add config with id2label for classification tasks
if task_type == "image_classification":
    class MockConfig:
        def __init__(self):
            self.id2label = {i: f"CLASS_{i}" for i in range(1000)}
            
    mock_model.config = MockConfig()
"""
    
    def get_hardware_detection_code(self) -> str:
        """Get Qualcomm-specific hardware detection code."""
        return """
# Qualcomm availability check
def is_available():
    try:
        from qnnpy import PyQnnManager
        qnn_manager = PyQnnManager()
        qnn_devices = qnn_manager.available_devices()
        
        if len(qnn_devices) > 0:
            # Get device info for each device
            device_info = []
            for device in qnn_devices:
                try:
                    # Try to get device details
                    device_props = qnn_manager.get_device_properties(device)
                    device_info.append({
                        "name": device,
                        "properties": device_props
                    })
                except:
                    device_info.append({"name": device})
            
            print(f"Qualcomm NPU devices available: {qnn_devices}")
            print(f"Device details: {device_info}")
            return True
        
        print("No Qualcomm NPU devices found")
        return False
    except ImportError:
        print("Qualcomm QNN SDK not installed")
        return False
    except Exception as e:
        print(f"Error checking Qualcomm NPU availability: {e}")
        return False
"""
    
    def get_utility_methods(self) -> str:
        """Get Qualcomm-specific utility methods."""
        return """
def _create_pytorch_to_qnn_wrapper(self, pt_model, task_type):
    \"\"\"Create a wrapper around PyTorch model that mimics QNN interface.
    
    Args:
        pt_model: PyTorch model
        task_type: Task type (text_embedding, etc.)
        
    Returns:
        Wrapped model with QNN interface
    \"\"\"
    # Keep original PyTorch model
    pt_model.orig_forward = pt_model.forward
    
    # Add QNN-like interface methods
    def qnn_forward(**kwargs):
        # Ensure inputs are tensors
        processed_inputs = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                processed_inputs[k] = torch.tensor(v, device=pt_model.device if hasattr(pt_model, 'device') else 'cpu')
            else:
                processed_inputs[k] = v
        
        # Call original forward
        with torch.no_grad():
            outputs = pt_model(**processed_inputs)
        
        # Convert outputs to numpy
        if isinstance(outputs, torch.Tensor):
            return outputs.cpu().numpy()
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state.cpu().numpy()
        elif hasattr(outputs, 'logits'):
            return {'logits': outputs.logits.cpu().numpy()}
        else:
            # Try to handle various output types
            processed_outputs = {}
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    processed_outputs[k] = v.cpu().numpy()
                else:
                    processed_outputs[k] = v
            return processed_outputs
    
    # Add generate method for text generation tasks
    def qnn_generate(**kwargs):
        # Ensure inputs are tensors
        processed_inputs = {}
        for k, v in kwargs.items():
            if k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'top_k']:
                # These are generation parameters, not inputs
                continue
            if isinstance(v, np.ndarray):
                processed_inputs[k] = torch.tensor(v, device=pt_model.device if hasattr(pt_model, 'device') else 'cpu')
            else:
                processed_inputs[k] = v
        
        # Get generation parameters
        gen_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', 100),
            'do_sample': kwargs.get('do_sample', False),
            'temperature': kwargs.get('temperature', 1.0),
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 0)
        }
        
        # Call generate method
        with torch.no_grad():
            output_ids = pt_model.generate(**processed_inputs, **gen_kwargs)
        
        # Return output IDs as numpy array
        return output_ids.cpu().numpy()
    
    # Add methods to the model
    pt_model.qnn_forward = qnn_forward
    if hasattr(pt_model, 'generate'):
        pt_model.qnn_generate = qnn_generate
    
    # Add QNN-like attributes
    pt_model.qnn_backend = None  # Indicate this is not a native QNN model
    
    # Add input shape information if needed
    if task_type == "image_classification":
        pt_model.input_shape = {"pixel_values": [1, 3, 224, 224]}
    elif task_type in ["text_embedding", "text_generation", "text2text_generation"]:
        pt_model.input_shape = {"input_ids": [1, 64], "attention_mask": [1, 64]}
    
    return pt_model
"""
    
    def is_compatible_with_architecture(self, arch_type: str) -> bool:
        """Check Qualcomm compatibility with architecture type."""
        # Qualcomm NPUs have limitations with some architectures
        compatible_archs = [
            "encoder-only", 
            "vision", 
            "vision-encoder-text-decoder",
            "speech"
        ]
        return arch_type in compatible_archs
    
    def get_fallback_hardware(self) -> str:
        """Get the fallback hardware type if Qualcomm is not available."""
        return "cpu"