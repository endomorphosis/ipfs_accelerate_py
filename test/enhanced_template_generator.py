#!/usr/bin/env python3
"""
Template Test Generator

This is a reference implementation for generating test files that are compatible
with the ipfs_accelerate_py worker/skillset module structure.

This file can serve as a guide for improving the merged_test_generator.py or
for creating new test generation tools.
"""

import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
SKILLS_DIR = TEST_DIR / "skills"
WORKER_SKILLSET = PROJECT_ROOT / "ipfs_accelerate_py" / "worker" / "skillset"

# Template for generating tests
def generate_test_file(model_type, output_dir=SKILLS_DIR, force=False):
    """Generate a test file for the specified model type."""
    
    # Normalize the model name
    normalized_name = model_type.replace('-', '_').replace('.', '_').lower()
    
    # Create the output file path
    output_file = output_dir / f"test_hf_{normalized_name}.py"
    
    # Check if the file already exists and we're not forcing overwrite
    if output_file.exists() and not force:
        print(f"Test file already exists for {model_type}, use --force to overwrite")
        return False
    
    # Check if we have a reference implementation in worker/skillset
    reference_file = WORKER_SKILLSET / f"hf_{normalized_name}.py"
    reference_exists = reference_file.exists()
    
    if reference_exists:
        print(f"Found reference implementation at {reference_file}")
    else:
        print(f"No reference implementation found, using template")
    
    # Define template variables
    template_vars = {
        "model_type": model_type,
        "normalized_name": normalized_name,
        "camel_case_name": ''.join(word.capitalize() for word in normalized_name.split('_')),
        "class_name": f"hf_{normalized_name}",
        "timestamp": datetime.datetime.now().isoformat(),
        "year": datetime.datetime.now().year,
        "primary_task": "text-generation",  # Default task
        "tasks": ["text-generation"],  # Default tasks
        "hardware_backends": ["cpu", "cuda", "openvino", "apple", "qualcomm", "amd", "webnn", "webgpu"],
        
        # Default tensor types
        "input_tensor_type": "int64",  # Default for token IDs
        "output_tensor_type": "float32",  # Default for embeddings/logits
        "input_format": "text",  # Default input format (text, image, audio, etc.)
        "output_format": "text",  # Default output format
        "uses_attention_mask": True,
        "uses_position_ids": False,
        "token_sequence_length": 512,  # Default sequence length
        "embedding_dim": 768,  # Default embedding dimension
        "model_precision": "float32",  # Default model precision
        
        # Precision support
        "supported_precisions": {
            "fp32": True,    # Full precision (float32)
            "fp16": True,    # Half precision (float16)
            "bf16": True,    # Brain floating point (bfloat16)
            "fp8": False,    # 8-bit floating point
            "fp4": False,    # 4-bit floating point
            "int8": True,    # 8-bit integer quantization
            "int4": False,   # 4-bit integer quantization
            "uint4": False   # Unsigned 4-bit integer quantization
        },
        "supports_half_precision": True,  # Legacy field for backward compatibility
        "batch_processing": True,  # Whether model supports batched inputs
        
        # Helper functions and dependencies
        "helper_functions": {
            "tokenization": {
                "description": "Tokenizes input text",
                "args": ["text", "max_length"],
                "returns": "Dictionary with input_ids and attention_mask"
            },
            "device_management": {
                "description": "Manages device selection and memory",
                "args": ["device_type", "memory_limit"],
                "returns": "Device object or identifier"
            },
            "web_export": {
                "description": "Exports model for web deployment",
                "args": ["format", "quantize", "optimize"],
                "returns": "Path to exported model files"
            }
        },
        
        # Endpoint handler parameters
        "handler_params": {
            "text": {
                "description": "Main textual input for the model",
                "type": "str or List[str]",
                "required": True
            },
            "max_length": {
                "description": "Maximum sequence length",
                "type": "int",
                "required": False,
                "default": 512
            },
            "truncation": {
                "description": "Whether to truncate sequences",
                "type": "bool",
                "required": False,
                "default": True
            }
        },
        
        # Dependencies
        "dependencies": {
            "python": ">=3.8,<3.11",
            "pip": [
                "torch>=1.12.0",
                "transformers>=4.26.0",
                "numpy>=1.20.0"
            ],
            "system": [],
            "optional": {
                "cuda": ["nvidia-cuda-toolkit>=11.6", "nvidia-cudnn>=8.3"],
                "openvino": ["openvino>=2022.1.0"],
                "apple": ["torch>=1.12.0"],
                "qualcomm": ["qti-aisw>=1.8.0"],
                "amd": ["rocm-smi>=5.0.0", "rccl>=2.0.0", "torch-rocm>=2.0.0"],
                "webnn": ["onnx>=1.14.0", "onnxruntime>=1.15.0", "webnn-polyfill>=1.0.0"],
                "webgpu": ["transformers.js>=2.6.0", "webgpu>=0.1.24"]
            },
            "precision": {
                "fp16": [],
                "bf16": ["torch>=1.12.0"],
                "int8": ["bitsandbytes>=0.41.0", "optimum>=1.12.0"],
                "int4": ["bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
                "uint4": ["bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
                "fp8": ["transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"],
                "fp4": ["transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"]
            },
            "web": {
                "webnn": ["onnx>=1.14.0", "onnxruntime-web>=1.16.0", "webnn-polyfill>=1.0.0"],
                "webgpu": ["@xenova/transformers>=2.6.0", "webgpu>=0.1.24"]
            }
        }
    }
    
    # Update with model-specific task info if available
    model_types_file = TEST_DIR / "huggingface_model_pipeline_map.json"
    if model_types_file.exists():
        try:
            with open(model_types_file, 'r') as f:
                pipeline_map = json.load(f)
                if model_type in pipeline_map:
                    template_vars["tasks"] = pipeline_map[model_type]
                    template_vars["primary_task"] = pipeline_map[model_type][0] if pipeline_map[model_type] else "text-generation"
                    
                    # Update template variables based on task type
                    primary_task = template_vars["primary_task"]
                    
                    # Image models
                    if primary_task in ["image-classification", "object-detection", "image-segmentation", 
                                      "depth-estimation", "feature-extraction"] and "image" in primary_task:
                        template_vars["input_tensor_type"] = "float32"  # Image pixels as float32
                        template_vars["input_format"] = "image"
                        template_vars["output_format"] = "classification" if "classification" in primary_task else "image"
                        template_vars["uses_attention_mask"] = False
                        template_vars["uses_position_ids"] = False
                        template_vars["embedding_dim"] = 1024  # Typical for vision models
                        
                        # Add image-specific helper functions
                        template_vars["helper_functions"]["image_loading"] = {
                            "description": "Loads and preprocesses images for the model",
                            "args": ["image_path", "resize", "center_crop", "normalize"],
                            "returns": "Tensor of preprocessed image"
                        }
                        template_vars["helper_functions"]["image_processing"] = {
                            "description": "Processes images to model-ready format",
                            "args": ["image", "mean", "std", "size"],
                            "returns": "Dictionary with pixel_values"
                        }
                        
                        # Add image-specific handler parameters
                        template_vars["handler_params"] = {
                            "image": {
                                "description": "Input image (path, URL, or tensor)",
                                "type": "str, PIL.Image, or torch.Tensor",
                                "required": True
                            },
                            "return_tensors": {
                                "description": "Format of returned tensors",
                                "type": "str",
                                "required": False,
                                "default": "pt"
                            }
                        }
                        
                        # Add image-specific dependencies
                        template_vars["dependencies"]["pip"].extend([
                            "pillow>=8.0.0",
                            "torchvision>=0.10.0"
                        ])
                        
                    # Text generation models
                    elif primary_task in ["text-generation", "summarization", "translation_XX_to_YY"]:
                        template_vars["input_tensor_type"] = "int64"  # Token IDs
                        template_vars["input_format"] = "text"
                        template_vars["output_format"] = "text"
                        template_vars["uses_attention_mask"] = True
                        
                        # Add text-specific helper functions
                        template_vars["helper_functions"]["tokenization"] = {
                            "description": "Tokenizes input text",
                            "args": ["text", "max_length", "padding", "truncation"],
                            "returns": "Dictionary with input_ids and attention_mask"
                        }
                        template_vars["helper_functions"]["generation_config"] = {
                            "description": "Configures text generation parameters",
                            "args": ["max_length", "temperature", "top_p", "top_k", "num_beams"],
                            "returns": "Generation configuration object"
                        }
                        
                        # Add text-specific handler parameters
                        template_vars["handler_params"] = {
                            "text": {
                                "description": "Input text to process",
                                "type": "str or List[str]",
                                "required": True
                            },
                            "max_length": {
                                "description": "Maximum token length to generate",
                                "type": "int",
                                "required": False,
                                "default": 20
                            },
                            "temperature": {
                                "description": "Sampling temperature",
                                "type": "float",
                                "required": False,
                                "default": 1.0
                            }
                        }
                        
                    # Multimodal models
                    elif primary_task in ["image-to-text", "visual-question-answering"]:
                        template_vars["input_tensor_type"] = "mixed"  # Both image and text inputs
                        template_vars["input_format"] = "multimodal"
                        template_vars["output_format"] = "text"
                        template_vars["uses_attention_mask"] = True
                        
                        # Add multimodal-specific helper functions
                        template_vars["helper_functions"]["image_loading"] = {
                            "description": "Loads and preprocesses images for the model",
                            "args": ["image_path", "resize", "center_crop", "normalize"],
                            "returns": "Tensor of preprocessed image"
                        }
                        template_vars["helper_functions"]["tokenization"] = {
                            "description": "Tokenizes input text",
                            "args": ["text", "max_length", "padding", "truncation"],
                            "returns": "Dictionary with input_ids and attention_mask"
                        }
                        template_vars["helper_functions"]["multimodal_inputs"] = {
                            "description": "Combines image and text inputs for multimodal models",
                            "args": ["image", "text", "processor"],
                            "returns": "Combined input dictionary"
                        }
                        
                        # Add multimodal-specific handler parameters
                        template_vars["handler_params"] = {
                            "image": {
                                "description": "Input image (path, URL, or tensor)",
                                "type": "str, PIL.Image, or torch.Tensor",
                                "required": True
                            },
                            "text": {
                                "description": "Text prompt or question about the image",
                                "type": "str or List[str]",
                                "required": True
                            },
                            "max_length": {
                                "description": "Maximum token length to generate",
                                "type": "int",
                                "required": False,
                                "default": 20
                            }
                        }
                        
                        # Add multimodal-specific dependencies
                        template_vars["dependencies"]["pip"].extend([
                            "pillow>=8.0.0",
                            "torchvision>=0.10.0"
                        ])
                        
                    # Audio models
                    elif primary_task in ["automatic-speech-recognition", "audio-classification", "text-to-audio"]:
                        template_vars["input_tensor_type"] = "float32"  # Audio features
                        template_vars["input_format"] = "audio"
                        template_vars["output_format"] = "text" if "speech-recognition" in primary_task else "audio"
                        template_vars["uses_attention_mask"] = True
                        template_vars["token_sequence_length"] = 16000  # 1s of audio at 16kHz
                        
                        # Add audio-specific helper functions
                        template_vars["helper_functions"]["audio_loading"] = {
                            "description": "Loads and preprocesses audio files",
                            "args": ["audio_path", "sampling_rate", "target_sampling_rate", "mono"],
                            "returns": "Audio tensor and sampling rate"
                        }
                        template_vars["helper_functions"]["audio_processing"] = {
                            "description": "Processes audio to model-ready format",
                            "args": ["audio", "sampling_rate", "max_length"],
                            "returns": "Dictionary with input_features"
                        }
                        
                        # Add audio-specific handler parameters
                        template_vars["handler_params"] = {
                            "audio": {
                                "description": "Input audio (path, URL, or tensor)",
                                "type": "str, array, or torch.Tensor",
                                "required": True
                            },
                            "sampling_rate": {
                                "description": "Audio sampling rate",
                                "type": "int",
                                "required": True,
                                "default": 16000
                            }
                        }
                        
                        # Add audio-specific dependencies
                        template_vars["dependencies"]["pip"].extend([
                            "librosa>=0.9.0",
                            "soundfile>=0.10.0"
                        ])
                        
                    # Embedding models
                    elif primary_task in ["feature-extraction"]:
                        if "image" in " ".join(template_vars["tasks"]):
                            template_vars["input_tensor_type"] = "float32"  # Image pixels
                            template_vars["input_format"] = "image"
                            template_vars["output_format"] = "embedding"
                            
                            # Add image embedding helper functions
                            template_vars["helper_functions"]["image_loading"] = {
                                "description": "Loads and preprocesses images for the model",
                                "args": ["image_path", "resize", "center_crop", "normalize"],
                                "returns": "Tensor of preprocessed image"
                            }
                            template_vars["helper_functions"]["image_processing"] = {
                                "description": "Processes images to model-ready format",
                                "args": ["image", "mean", "std", "size"],
                                "returns": "Dictionary with pixel_values"
                            }
                            
                            # Add image embedding dependencies
                            template_vars["dependencies"]["pip"].extend([
                                "pillow>=8.0.0",
                                "torchvision>=0.10.0"
                            ])
                        else:
                            template_vars["input_tensor_type"] = "int64"  # Token IDs
                            template_vars["input_format"] = "text"
                            template_vars["output_format"] = "embedding"
                            
                            # Add text embedding helper functions
                            template_vars["helper_functions"]["tokenization"] = {
                                "description": "Tokenizes input text",
                                "args": ["text", "max_length", "padding", "truncation"],
                                "returns": "Dictionary with input_ids and attention_mask"
                            }
                    
                    # Ensure all dependencies are unique
                    template_vars["dependencies"]["pip"] = list(set(template_vars["dependencies"]["pip"]))
                    
        except Exception as e:
            print(f"Error loading pipeline map: {e}")
    
    # Generate the test file content
    # This template is based on the current structure of ipfs_accelerate_py worker/skillset modules
    template = f"""#!/usr/bin/env python3
\"\"\"
Test implementation for {model_type}

This file provides a standardized test interface for {model_type} models
across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).

Generated by template_test_generator.py - {template_vars['timestamp']}
\"\"\"

import os
import sys
import json
import time
import datetime
import traceback
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np

# Try/except pattern for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = MagicMock()
    TORCH_AVAILABLE = False
    print("Warning: torch not available, using mock implementation")

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    transformers = MagicMock()
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, using mock implementation")

# Model Information:
# Model type: {model_type}
# Primary task: {template_vars['primary_task']}
# All tasks: {', '.join(template_vars['tasks'])}

# Input/Output:
# Input format: {template_vars['input_format']}
# Input tensor type: {template_vars['input_tensor_type']}
# Output format: {template_vars['output_format']}
# Output tensor type: {template_vars['output_tensor_type']}
# Uses attention mask: {template_vars['uses_attention_mask']}

# Required Helper Functions:
{',\\n# '.join([f"{name}: {helper['description']} (args: {', '.join(helper['args'])})" for name, helper in template_vars['helper_functions'].items()])}

# Primary Dependencies:
# Python: {template_vars['dependencies']['python']}
# Required pip packages: {', '.join(template_vars['dependencies']['pip'])}
# Optional platform-specific dependencies available in MODEL_REGISTRY

# Model Registry - Contains metadata about available models for this type
MODEL_REGISTRY = {{
    # Default/small model configuration
    "{model_type}": {{
        "description": "Default {model_type} model",
        
        # Model dimensions and capabilities
        "embedding_dim": {template_vars['embedding_dim']},
        "sequence_length": {template_vars['token_sequence_length']},
        "model_precision": "{template_vars['model_precision']}", 
        "supports_half_precision": {template_vars['supports_half_precision']},
        "default_batch_size": 1,
        
        # Hardware compatibility
        "hardware_compatibility": {{
            "cpu": True,
            "cuda": True,
            "openvino": True,
            "apple": True,
            "qualcomm": False,  # Usually false for complex models
            "amd": True,  # AMD ROCm support
            "webnn": True,  # WebNN support
            "webgpu": True   # WebGPU/transformers.js support
        }},
        
        # Precision support by hardware
        "precision_compatibility": {{
            "cpu": {{
                "fp32": True,
                "fp16": False,
                "bf16": True,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "cuda": {{
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": True,
                "uint4": True,
                "fp8": False,
                "fp4": False
            }},
            "openvino": {{
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "apple": {{
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": False,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "amd": {{
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "qualcomm": {{
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "webnn": {{
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "webgpu": {{
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }}
        }},
        
        # Input/Output specifications
        "input": {{
            "format": "{template_vars['input_format']}",
            "tensor_type": "{template_vars['input_tensor_type']}",
            "uses_attention_mask": {template_vars['uses_attention_mask']},
            "uses_position_ids": {template_vars['uses_position_ids']},
            "typical_shapes": ["batch_size, {template_vars['token_sequence_length']}"] if "{template_vars['input_tensor_type']}" == "int64" else 
                             ["batch_size, 3, 224, 224"] if "{template_vars['input_format']}" == "image" else
                             ["batch_size, {template_vars['token_sequence_length']}"]
        }},
        "output": {{
            "format": "{template_vars['output_format']}",
            "tensor_type": "{template_vars['output_tensor_type']}",
            "typical_shapes": ["batch_size, {template_vars['embedding_dim']}"] if "{template_vars['output_format']}" == "embedding" else
                             ["batch_size, sequence_length, {template_vars['embedding_dim']}"]
        }},
        
        # Required helper functions
        "helper_functions": {json.dumps(template_vars['helper_functions'], indent=4)},
        
        # Handler parameters
        "handler_params": {json.dumps(template_vars['handler_params'], indent=4)},
        
        # Dependencies
        "dependencies": {json.dumps(template_vars['dependencies'], indent=4)}
    }},
    
    # Small variant for low-resource environments
    "{model_type}-small": {{
        "description": "Smaller {model_type} variant for resource-constrained environments",
        "embedding_dim": {template_vars['embedding_dim'] // 2},
        "sequence_length": {template_vars['token_sequence_length'] // 2},
        "model_precision": "{template_vars['model_precision']}",
        "supports_half_precision": {template_vars['supports_half_precision']},
        "default_batch_size": 2,
        
        # Hardware compatibility - small variants often work everywhere
        "hardware_compatibility": {{
            "cpu": True,
            "cuda": True,
            "openvino": True,
            "apple": True,
            "qualcomm": True,
            "amd": True,
            "webnn": True,
            "webgpu": True
        }},
        
        # Precision support by hardware - small models usually have broader precision support
        "precision_compatibility": {{
            "cpu": {{
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": True,
                "uint4": True,
                "fp8": False,
                "fp4": False
            }},
            "cuda": {{
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": True,
                "uint4": True,
                "fp8": True,
                "fp4": False
            }},
            "openvino": {{
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": True,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "apple": {{
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "amd": {{
                "fp32": True,
                "fp16": True,
                "bf16": True,
                "int8": True,
                "int4": True,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "qualcomm": {{
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": True,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "webnn": {{
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }},
            "webgpu": {{
                "fp32": True,
                "fp16": True,
                "bf16": False,
                "int8": True,
                "int4": False,
                "uint4": False,
                "fp8": False,
                "fp4": False
            }}
        }},
        
        # Input/Output specifications - same as main model
        "input": {{
            "format": "{template_vars['input_format']}",
            "tensor_type": "{template_vars['input_tensor_type']}",
            "uses_attention_mask": {template_vars['uses_attention_mask']},
            "uses_position_ids": {template_vars['uses_position_ids']},
            "typical_shapes": ["batch_size, {template_vars['token_sequence_length'] // 2}"] if "{template_vars['input_tensor_type']}" == "int64" else 
                             ["batch_size, 3, 224, 224"] if "{template_vars['input_format']}" == "image" else
                             ["batch_size, {template_vars['token_sequence_length'] // 2}"]
        }},
        "output": {{
            "format": "{template_vars['output_format']}",
            "tensor_type": "{template_vars['output_tensor_type']}",
            "typical_shapes": ["batch_size, {template_vars['embedding_dim'] // 2}"] if "{template_vars['output_format']}" == "embedding" else
                             ["batch_size, sequence_length, {template_vars['embedding_dim'] // 2}"]
        }},
        
        # Helper functions and dependencies are same as main model
        "helper_functions": {json.dumps(template_vars['helper_functions'], indent=4)},
        "handler_params": {json.dumps(template_vars['handler_params'], indent=4)},
        "dependencies": {json.dumps(template_vars['dependencies'], indent=4)}
    }}
}}

class {template_vars['class_name']}:
    \"\"\"
    {model_type.capitalize()} implementation.
    
    This class provides standardized interfaces for working with {model_type} models
    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).
    \"\"\"
    
    def __init__(self, resources=None, metadata=None):
        \"\"\"Initialize the {model_type} model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        \"\"\"
        self.resources = resources or {{
            "torch": torch,
            "numpy": np,
            "transformers": transformers
        }}
        self.metadata = metadata or {{}}
        
        # Handler creation methods
        self.create_cpu_text_embedding_endpoint_handler = self.create_cpu_text_embedding_endpoint_handler
        self.create_cuda_text_embedding_endpoint_handler = self.create_cuda_text_embedding_endpoint_handler
        self.create_openvino_text_embedding_endpoint_handler = self.create_openvino_text_embedding_endpoint_handler
        self.create_apple_text_embedding_endpoint_handler = self.create_apple_text_embedding_endpoint_handler
        self.create_amd_text_embedding_endpoint_handler = self.create_amd_text_embedding_endpoint_handler
        self.create_qualcomm_text_embedding_endpoint_handler = self.create_qualcomm_text_embedding_endpoint_handler
        self.create_webnn_text_embedding_endpoint_handler = self.create_webnn_text_embedding_endpoint_handler
        self.create_webgpu_text_embedding_endpoint_handler = self.create_webgpu_text_embedding_endpoint_handler
        
        # Initialization methods
        self.init = self.init_cpu  # Default to CPU
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init_amd = self.init_amd
        self.init_qualcomm = self.init_qualcomm
        self.init_webnn = self.init_webnn
        self.init_webgpu = self.init_webgpu
        
        # Test methods
        self.__test__ = self.__test__
        
        # Hardware-specific utilities
        self.snpe_utils = None  # Qualcomm SNPE utils
        
        # Set up model registry and hardware detection
        self.model_registry = MODEL_REGISTRY
        self.hardware_capabilities = self._detect_hardware()
        
        # Set up detailed model information - this provides access to all registry properties
        self.model_info = {{
            "input": {{
                "format": "{template_vars['input_format']}",
                "tensor_type": "{template_vars['input_tensor_type']}",
                "uses_attention_mask": {template_vars['uses_attention_mask']},
                "uses_position_ids": {template_vars['uses_position_ids']},
                "default_sequence_length": {template_vars['token_sequence_length']}
            }},
            "output": {{
                "format": "{template_vars['output_format']}",
                "tensor_type": "{template_vars['output_tensor_type']}",
                "embedding_dim": {template_vars['embedding_dim']}
            }},
            "helper_functions": {json.dumps(template_vars['helper_functions'], indent=2)},
            "endpoint_params": {json.dumps(template_vars['handler_params'], indent=2)},
            "dependencies": {json.dumps(template_vars['dependencies'], indent=2)}
        }}
        
        # Maintain backward compatibility with old tensor_types structure
        self.tensor_types = {{
            "input": "{template_vars['input_tensor_type']}",
            "output": "{template_vars['output_tensor_type']}",
            "uses_attention_mask": {template_vars['uses_attention_mask']},
            "uses_position_ids": {template_vars['uses_position_ids']},
            "embedding_dim": {template_vars['embedding_dim']},
            "default_sequence_length": {template_vars['token_sequence_length']}
        }}
        return None
    
    def _detect_hardware(self):
        """Detect available hardware and return capabilities dictionary."""
        capabilities = {
            "cpu": True,
            "cuda": False,
            "cuda_version": None,
            "cuda_devices": 0,
            "mps": False,
            "openvino": False,
            "qualcomm": False,
            "amd": False,
            "amd_version": None,
            "amd_devices": 0,
            "webnn": False,
            "webgpu": False
        }
        
        # Check CUDA
        if TORCH_AVAILABLE:
            capabilities["cuda"] = torch.cuda.is_available()
            if capabilities["cuda"]:
                capabilities["cuda_devices"] = torch.cuda.device_count()
                if hasattr(torch.version, "cuda"):
                    capabilities["cuda_version"] = torch.version.cuda
        
        # Check MPS (Apple Silicon)
        if TORCH_AVAILABLE and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            capabilities["mps"] = torch.mps.is_available()
        
        # Check AMD ROCm support
        try:
            # Check for the presence of ROCm by importing rocm-specific modules or checking for devices
            import subprocess
            
            # Try to run rocm-smi to detect ROCm installation
            result = subprocess.run(['rocm-smi', '--showproductname'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  universal_newlines=True, check=False)
            
            if result.returncode == 0:
                capabilities["amd"] = True
                
                # Try to get version information
                version_result = subprocess.run(['rocm-smi', '--showversion'], 
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                             universal_newlines=True, check=False)
                
                if version_result.returncode == 0:
                    # Extract version from output
                    import re
                    match = re.search(r'ROCm-SMI version:\s+(\d+\.\d+\.\d+)', version_result.stdout)
                    if match:
                        capabilities["amd_version"] = match.group(1)
                
                # Try to count devices
                devices_result = subprocess.run(['rocm-smi', '--showalldevices'], 
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                             universal_newlines=True, check=False)
                
                if devices_result.returncode == 0:
                    # Count device entries in output
                    device_lines = [line for line in devices_result.stdout.split('\n') if 'GPU[' in line]
                    capabilities["amd_devices"] = len(device_lines)
        except (ImportError, FileNotFoundError):
            pass
            
        # Alternate check for AMD ROCm using torch hip if available
        if TORCH_AVAILABLE and not capabilities["amd"]:
            try:
                import torch.utils.hip as hip
                if hasattr(hip, "is_available") and hip.is_available():
                    capabilities["amd"] = True
                    capabilities["amd_devices"] = hip.device_count()
            except (ImportError, AttributeError):
                pass
        
        # Check OpenVINO
        try:
            import openvino
            capabilities["openvino"] = True
        except ImportError:
            pass
            
        # Check for Qualcomm AI Engine Direct SDK
        try:
            import qti.aisw.dlc_utils
            capabilities["qualcomm"] = True
        except ImportError:
            pass
            
        # Check for WebNN support
        try:
            import onnx
            import onnxruntime
            capabilities["webnn"] = True
        except ImportError:
            pass
            
        # Check for WebGPU/transformers.js support
        try:
            # In Node.js environment, we can check for transformers.js package
            import importlib.util
            if importlib.util.find_spec("transformers.js") or importlib.util.find_spec("@xenova/transformers"):
                capabilities["webgpu"] = True
        except ImportError:
            pass
            
        return capabilities
    
    def _get_model_info(self, model_id=None):
        """Get comprehensive model information for a specific model variant.
        
        This function returns a detailed information structure about the model including:
        - Input/output specifications
        - Hardware compatibility
        - Required helper functions
        - Endpoint handler parameters
        - Dependencies
        
        Args:
            model_id: Specific model identifier (e.g., "{model_type}", "{model_type}-small")
            
        Returns:
            Dictionary with complete model information
        """
        model_id = model_id or "{model_type}"
        
        if model_id in self.model_registry:
            # Return complete model configuration from registry
            return self.model_registry[model_id]
        
        # Return default info if model not in registry
        return {{
            "input": self.model_info["input"],
            "output": self.model_info["output"],
            "helper_functions": self.model_info["helper_functions"],
            "handler_params": self.model_info["endpoint_params"],
            "dependencies": self.model_info["dependencies"],
            "embedding_dim": {template_vars['embedding_dim']},
            "sequence_length": {template_vars['token_sequence_length']},
            "model_precision": "{template_vars['model_precision']}",
            "supports_half_precision": {template_vars['supports_half_precision']}
        }}
        
    def _get_model_tensor_types(self, model_id=None):
        """Get tensor type information for a specific model.
        
        Legacy method maintained for backward compatibility.
        For new code, use _get_model_info() instead.
        """
        model_id = model_id or "{model_type}"
        if model_id in self.model_registry:
            config = self.model_registry[model_id]
            return {{
                "embedding_dim": config.get("embedding_dim", {template_vars['embedding_dim']}),
                "sequence_length": config.get("sequence_length", {template_vars['token_sequence_length']}),
                "precision": config.get("model_precision", "{template_vars['model_precision']}"),
                "supports_half": config.get("supports_half_precision", {template_vars['supports_half_precision']}),
                "input_format": config.get("input", {{}}).get("format", "{template_vars['input_format']}"),
                "output_format": config.get("output", {{}}).get("format", "{template_vars['output_format']}")
            }}
        return self.tensor_types
    
    # Model-specific processing helpers based on task type
    
    def _process_text_input(self, text, tokenizer=None, max_length=None):
        """Process text input for text-based models."""
        if tokenizer is None:
            tokenizer = self._create_mock_processor()
            
        max_length = max_length or self.tensor_types["default_sequence_length"]
        
        # Tokenize input
        if isinstance(text, str):
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", 
                             truncation=True, max_length=max_length)
        else:
            inputs = tokenizer(list(text), return_tensors="pt", padding="max_length", 
                             truncation=True, max_length=max_length)
            
        return inputs
        
    def _process_image_input(self, image_input, processor=None):
        """Process image input for vision-based models."""
        if processor is None:
            # Create a simple mock image processor
            from unittest.mock import MagicMock
            processor = MagicMock()
            processor.return_value = {{"pixel_values": torch.rand((1, 3, 224, 224))}}
            
        # Handle file paths, URLs, PIL images, etc.
        if isinstance(image_input, str):
            # Mock image processing
            return {{"pixel_values": torch.rand((1, 3, 224, 224))}}
        elif isinstance(image_input, list):
            # Batch of images
            batch_size = len(image_input)
            return {{"pixel_values": torch.rand((batch_size, 3, 224, 224))}}
        else:
            # Assume direct tensor input
            return {{"pixel_values": image_input}}
            
    def _process_audio_input(self, audio_input, processor=None, sampling_rate=16000):
        """Process audio input for audio-based models."""
        if processor is None:
            # Create a simple mock audio processor
            from unittest.mock import MagicMock
            processor = MagicMock()
            
        # Mock audio processing
        if isinstance(audio_input, str):
            # Assuming audio_input is a file path
            return {{"input_features": torch.rand((1, 80, 3000))}}
        elif isinstance(audio_input, list):
            # Batch of audio inputs
            batch_size = len(audio_input)
            return {{"input_features": torch.rand((batch_size, 80, 3000))}}
        else:
            # Assume direct tensor input
            return {{"input_features": audio_input}}
    
    def _create_mock_processor(self):
        \"\"\"Create a mock processor/tokenizer for testing.\"\"\"
        class MockProcessor:
            def __init__(self):
                self.vocab_size = 30000
                
            def __call__(self, text, **kwargs):
                # Handle both single strings and batches
                if isinstance(text, str):
                    batch_size = 1
                else:
                    batch_size = len(text)
                    
                return {{
                    "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, 10), dtype=torch.long)
                }}
                
            def decode(self, token_ids, **kwargs):
                return "Decoded text from mock processor"
        
        return MockProcessor()

    def _create_mock_endpoint(self):
        \"\"\"Create a mock endpoint/model for testing.\"\"\"
        class MockEndpoint:
            def __init__(self):
                self.config = type('obj', (object,), {{
                    'hidden_size': 768,
                    'max_position_embeddings': 512
                }})
                
            def eval(self):
                return self
                
            def to(self, device):
                return self
                
            def __call__(self, **kwargs):
                # Handle inputs
                batch_size = kwargs.get("input_ids").shape[0]
                seq_len = kwargs.get("input_ids").shape[1]
                
                # Create mock output
                output = type('obj', (object,), {{}})
                output.last_hidden_state = torch.rand((batch_size, seq_len, 768))
                
                return output
        
        return MockEndpoint()

    def init_cpu(self, model_name, model_type, device="cpu", **kwargs):
        \"\"\"Initialize model for CPU inference.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model ('{template_vars['primary_task']}', etc.)
            device (str): CPU identifier ('cpu')
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            
            # Create processor and endpoint
            processor = self._create_mock_processor()
            endpoint = self._create_mock_endpoint()
            
            # Create handler
            handler = self.create_cpu_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label="cpu",
                endpoint=endpoint,
                tokenizer=processor
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 1
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing CPU model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock CPU output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 1

    def init_cuda(self, model_name, model_type, device_label="cuda:0", **kwargs):
        \"\"\"Initialize model for CUDA inference.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model ('{template_vars['primary_task']}', etc.)
            device_label (str): GPU device ('cuda:0', 'cuda:1', etc.)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            
            # Create processor and endpoint
            processor = self._create_mock_processor()
            endpoint = self._create_mock_endpoint()
            
            # Move to CUDA
            endpoint = endpoint.to(device_label)
            
            # Create handler
            handler = self.create_cuda_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device_label,
                hardware_label=device_label,
                endpoint=endpoint,
                tokenizer=processor,
                is_real_impl=True,
                batch_size=4
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 4  # Default to larger batch size for CUDA
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing CUDA model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock CUDA output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 2

    def init_openvino(self, model_name, model_type, device="CPU", **kwargs):
        \"\"\"Initialize model for OpenVINO inference.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model ('{template_vars['primary_task']}', etc.)
            device (str): OpenVINO device ('CPU', 'GPU', etc.)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            import numpy as np
            
            # Create processor and endpoint (OpenVINO-specific)
            processor = self._create_mock_processor()
            
            # Create OpenVINO-style endpoint
            class MockOpenVINOModel:
                def infer(self, inputs):
                    batch_size = 1
                    seq_len = 10
                    if isinstance(inputs, dict) and 'input_ids' in inputs:
                        if hasattr(inputs['input_ids'], 'shape'):
                            batch_size = inputs['input_ids'].shape[0]
                            if len(inputs['input_ids'].shape) > 1:
                                seq_len = inputs['input_ids'].shape[1]
                    
                    # Return OpenVINO-style output
                    return {{"last_hidden_state": np.random.rand(batch_size, seq_len, 768).astype(np.float32)}}
            
            endpoint = MockOpenVINOModel()
            
            # Create handler
            handler = self.create_openvino_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                tokenizer=processor,
                openvino_label=device,
                endpoint=endpoint
            )
            
            # Create queue
            queue = asyncio.Queue(64)
            batch_size = 1
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing OpenVINO model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock OpenVINO output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(64), 1

    def init_apple(self, model_name, model_type, device="mps", **kwargs):
        \"\"\"Initialize model for Apple Silicon (M1/M2/M3) inference.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model ('{template_vars['primary_task']}', etc.)
            device (str): Device identifier ('mps')
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            
            # Create processor and endpoint
            processor = self._create_mock_processor()
            endpoint = self._create_mock_endpoint()
            
            # Move to MPS
            if TORCH_AVAILABLE and hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
                endpoint = endpoint.to('mps')
            
            # Create handler
            handler = self.create_apple_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                apple_label=device,
                endpoint=endpoint,
                tokenizer=processor
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 2
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing Apple Silicon model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock Apple Silicon output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 2

    def init_amd(self, model_name, model_type, device="rocm:0", **kwargs):
        \"\"\"Initialize model for AMD ROCm inference.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model ('{template_vars['primary_task']}', etc.)
            device_label (str): ROCm device identifier ('rocm:0', 'rocm:1', etc.)
            precision (str, optional): Precision to use (fp32, fp16, bf16, int8, int4)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor and endpoint
            processor = self._create_mock_processor()
            endpoint = self._create_mock_endpoint()
            
            # Move to AMD ROCm device
            endpoint = endpoint.to(device)
            
            # Apply quantization if needed
            if precision in ["int8", "int4", "uint4"]:
                # In real implementation, would apply quantization here
                print(f"Applying {precision} quantization for AMD ROCm device")
            
            # Create handler
            handler = self.create_amd_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=f"amd_{device}",
                endpoint=endpoint,
                tokenizer=processor,
                is_real_impl=True,
                batch_size=4,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 4  # Default to larger batch size for AMD GPUs
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing AMD ROCm model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock AMD ROCm output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 2
            
    def init_qualcomm(self, model_name, model_type, device="qualcomm", **kwargs):
        \"\"\"Initialize model for Qualcomm AI inference.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model ('{template_vars['primary_task']}', etc.)
            device (str): Device identifier ('qualcomm')
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            import numpy as np
            
            # Create processor
            processor = self._create_mock_processor()
            
            # Create Qualcomm-style endpoint
            class MockQualcommModel:
                def execute(self, inputs):
                    batch_size = 1
                    seq_len = 10
                    if isinstance(inputs, dict) and 'input_ids' in inputs:
                        if hasattr(inputs['input_ids'], 'shape'):
                            batch_size = inputs['input_ids'].shape[0]
                            if len(inputs['input_ids'].shape) > 1:
                                seq_len = inputs['input_ids'].shape[1]
                    
                    # Return Qualcomm-style output
                    return {{"output": np.random.rand(batch_size, seq_len, 768).astype(np.float32)}}
            
            endpoint = MockQualcommModel()
            
            # Create handler
            handler = self.create_qualcomm_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                qualcomm_label=device,
                endpoint=endpoint,
                tokenizer=processor
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 1
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing Qualcomm model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock Qualcomm output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 1
            
    def init_webnn(self, model_name, model_type, device="webnn", **kwargs):
        \"\"\"Initialize model for WebNN inference.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model ('{template_vars['primary_task']}', etc.)
            device (str): Device identifier ('webnn')
            precision (str, optional): Precision to use (fp32, fp16, int8)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            import numpy as np
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor and endpoint
            processor = self._create_mock_processor()
            
            # Create WebNN-style endpoint
            class MockWebNNModel:
                def __init__(self):
                    self.input_names = ["input_ids", "attention_mask"]
                    self.output_names = ["last_hidden_state"]
                    
                def run(self, inputs):
                    batch_size = 1
                    seq_len = 10
                    if isinstance(inputs, dict) and 'input_ids' in inputs:
                        if hasattr(inputs['input_ids'], 'shape'):
                            batch_size = inputs['input_ids'].shape[0]
                            if len(inputs['input_ids'].shape) > 1:
                                seq_len = inputs['input_ids'].shape[1]
                    
                    # Return WebNN-style output
                    return [np.random.rand(batch_size, seq_len, 768).astype(np.float32)]
            
            endpoint = MockWebNNModel()
            
            # Create handler
            handler = self.create_webnn_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                webnn_label=device,
                endpoint=endpoint,
                tokenizer=processor,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 1
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing WebNN model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock WebNN output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 1
            
    def init_webgpu(self, model_name, model_type, device="webgpu", **kwargs):
        \"\"\"Initialize model for WebGPU (transformers.js) inference.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model ('{template_vars['primary_task']}', etc.)
            device (str): Device identifier ('webgpu')
            precision (str, optional): Precision to use (fp32, fp16, int8)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        \"\"\"
        try:
            import asyncio
            import numpy as np
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor and endpoint
            processor = self._create_mock_processor()
            
            # Create transformers.js-style endpoint
            class MockTransformersJSModel:
                def __init__(self):
                    self.model_type = model_type
                    
                async def generate(self, inputs):
                    # This simulates transformers.js API which is promise-based
                    batch_size = 1
                    if isinstance(inputs, dict) and 'input_ids' in inputs:
                        if hasattr(inputs['input_ids'], 'shape'):
                            batch_size = inputs['input_ids'].shape[0]
                    
                    return {{"generated_text": ["This is mock output from transformers.js"] * batch_size}}
                    
                async def encode(self, inputs):
                    # This simulates transformers.js embedding API
                    batch_size = 1
                    if isinstance(inputs, list):
                        batch_size = len(inputs)
                    
                    return np.random.rand(batch_size, 768).astype(np.float32)
            
            endpoint = MockTransformersJSModel()
            
            # Create handler
            handler = self.create_webgpu_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                webgpu_label=device,
                endpoint=endpoint,
                tokenizer=processor,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 1
            
            return endpoint, processor, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing WebGPU/transformers.js model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock WebGPU/transformers.js output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 1

    # Handler creation methods
    def create_cpu_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None):
        \"\"\"Create a handler function for CPU inference.
        
        Args:
            endpoint_model: Model name
            device: Device to run on ('cpu')
            hardware_label: Label for the endpoint
            endpoint: Model endpoint
            tokenizer: Tokenizer for the model
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                
                # Create mock output with appropriate structure
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                tensor_output = torch.rand((batch_size, 768))  # Standard embedding size
                
                # Return dictionary with tensor and metadata instead of adding attributes to tensor
                return {{
                    "tensor": tensor_output,
                    "implementation_type": "MOCK",
                    "device": "cpu",
                    "model": endpoint_model
                }}
            except Exception as e:
                print(f"Error in CPU handler: {{e}}")
                # Return a simple dict on error
                return {{"output": "Error in CPU handler", "implementation_type": "MOCK"}}
                
        return handler

    def create_cuda_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None, is_real_impl=False, batch_size=1):
        \"\"\"Create a handler function for CUDA inference.
        
        Args:
            endpoint_model: Model name
            device: Device to run on ('cuda:0', etc.)
            hardware_label: Label for the endpoint
            endpoint: Model endpoint
            tokenizer: Tokenizer for the model
            is_real_impl: Whether this is a real implementation
            batch_size: Batch size for processing
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                
                # Create mock output with appropriate structure
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                tensor_output = torch.rand((batch_size, 768))  # Standard embedding size
                
                # Return dictionary with tensor and metadata instead of adding attributes to tensor
                return {{
                    "tensor": tensor_output,
                    "implementation_type": "MOCK",
                    "device": device,
                    "model": endpoint_model,
                    "is_cuda": True
                }}
            except Exception as e:
                print(f"Error in CUDA handler: {{e}}")
                # Return a simple dict on error
                return {{"output": "Error in CUDA handler", "implementation_type": "MOCK"}}
                
        return handler

    def create_openvino_text_embedding_endpoint_handler(self, endpoint_model, tokenizer, openvino_label, endpoint=None):
        \"\"\"Create a handler function for OpenVINO inference.
        
        Args:
            endpoint_model: Model name
            tokenizer: Tokenizer for the model
            openvino_label: Label for the endpoint
            endpoint: OpenVINO model endpoint
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                
                # Create mock output with appropriate structure
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                tensor_output = torch.rand((batch_size, 768))  # Standard embedding size
                
                # Return dictionary with tensor and metadata instead of adding attributes to tensor
                return {{
                    "tensor": tensor_output,
                    "implementation_type": "MOCK",
                    "device": "OpenVINO",
                    "model": endpoint_model,
                    "is_openvino": True
                }}
            except Exception as e:
                print(f"Error in OpenVINO handler: {{e}}")
                # Return a simple dict on error
                return {{"output": "Error in OpenVINO handler", "implementation_type": "MOCK"}}
                
        return handler

    def create_apple_text_embedding_endpoint_handler(self, endpoint_model, apple_label, endpoint=None, tokenizer=None):
        \"\"\"Create a handler function for Apple Silicon inference.
        
        Args:
            endpoint_model: Model name
            apple_label: Label for the endpoint
            endpoint: Model endpoint
            tokenizer: Tokenizer for the model
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                
                # Create mock output with appropriate structure
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                tensor_output = torch.rand((batch_size, 768))  # Standard embedding size
                
                # Return dictionary with tensor and metadata instead of adding attributes to tensor
                return {{
                    "tensor": tensor_output,
                    "implementation_type": "MOCK",
                    "device": "MPS",
                    "model": endpoint_model,
                    "is_mps": True
                }}
            except Exception as e:
                print(f"Error in Apple Silicon handler: {{e}}")
                # Return a simple dict on error
                return {{"output": "Error in Apple Silicon handler", "implementation_type": "MOCK"}}
                
        return handler

    def create_amd_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None, is_real_impl=False, batch_size=1, precision="fp32"):
        \"\"\"Create a handler function for AMD ROCm inference.
        
        Args:
            endpoint_model: Model name
            device: Device to run on ('rocm:0', etc.)
            hardware_label: Label for the endpoint
            endpoint: Model endpoint
            tokenizer: Tokenizer for the model
            is_real_impl: Whether this is a real implementation
            batch_size: Batch size for processing
            precision: Precision to use (fp32, fp16, bf16, int8, int4, uint4)
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                
                # Create mock output with appropriate structure
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                tensor_output = torch.rand((batch_size, 768))  # Standard embedding size
                
                # Return dictionary with tensor and metadata instead of adding attributes to tensor
                return {{
                    "tensor": tensor_output,
                    "implementation_type": "MOCK",
                    "device": device,
                    "model": endpoint_model,
                    "is_amd": True,
                    "precision": precision
                }}
            except Exception as e:
                print(f"Error in AMD ROCm handler: {{e}}")
                # Return a simple dict on error
                return {{"output": "Error in AMD ROCm handler", "implementation_type": "MOCK"}}
                
        return handler
        
    def create_qualcomm_text_embedding_endpoint_handler(self, endpoint_model, qualcomm_label, endpoint=None, tokenizer=None):
        \"\"\"Create a handler function for Qualcomm AI inference.
        
        Args:
            endpoint_model: Model name
            qualcomm_label: Label for the endpoint
            endpoint: Model endpoint
            tokenizer: Tokenizer for the model
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                
                # Create mock output with appropriate structure
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                tensor_output = torch.rand((batch_size, 768))  # Standard embedding size
                
                # Return dictionary with tensor and metadata instead of adding attributes to tensor
                return {{
                    "tensor": tensor_output,
                    "implementation_type": "MOCK",
                    "device": "Qualcomm",
                    "model": endpoint_model,
                    "is_qualcomm": True
                }}
            except Exception as e:
                print(f"Error in Qualcomm handler: {{e}}")
                # Return a simple dict on error
                return {{"output": "Error in Qualcomm handler", "implementation_type": "MOCK"}}
                
        return handler
        
    def create_webnn_text_embedding_endpoint_handler(self, endpoint_model, webnn_label, endpoint=None, tokenizer=None, precision="fp32"):
        \"\"\"Create a handler function for WebNN inference.
        
        Args:
            endpoint_model: Model name
            webnn_label: Label for the endpoint
            endpoint: Model endpoint
            tokenizer: Tokenizer for the model
            precision: Precision type (fp32, fp16, int8)
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                import numpy as np
                
                # Create mock output with appropriate structure
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                
                # For WebNN, we'd typically convert from numpy arrays to torch tensors
                # after executing the ONNX model with WebNN
                np_output = np.random.rand(batch_size, 768).astype(np.float32)
                tensor_output = torch.from_numpy(np_output)
                
                # Return dictionary with tensor and metadata
                return {{
                    "tensor": tensor_output,
                    "implementation_type": "MOCK",
                    "device": "WebNN",
                    "model": endpoint_model,
                    "is_webnn": True,
                    "precision": precision,
                    "web_backend": "webnn"
                }}
            except Exception as e:
                print(f"Error in WebNN handler: {{e}}")
                # Return a simple dict on error
                return {{"output": "Error in WebNN handler", "implementation_type": "MOCK"}}
                
        return handler
        
    def create_webgpu_text_embedding_endpoint_handler(self, endpoint_model, webgpu_label, endpoint=None, tokenizer=None, precision="fp32"):
        \"\"\"Create a handler function for WebGPU (transformers.js) inference.
        
        Args:
            endpoint_model: Model name
            webgpu_label: Label for the endpoint
            endpoint: Model endpoint
            tokenizer: Tokenizer for the model
            precision: Precision type (fp32, fp16, int8)
            
        Returns:
            A handler function that accepts text input and returns embeddings
        \"\"\"
        # Create a handler that works with the endpoint and tokenizer
        def handler(text_input):
            try:
                # This should match how the actual handler would process data
                import torch
                import numpy as np
                
                # Create mock output with appropriate structure
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                
                # For transformers.js, we'd typically be working with JavaScript 
                # tensors that get converted to numpy arrays and then to torch tensors
                np_output = np.random.rand(batch_size, 768).astype(np.float32)
                tensor_output = torch.from_numpy(np_output)
                
                # Return dictionary with tensor and metadata
                return {{
                    "tensor": tensor_output,
                    "implementation_type": "MOCK",
                    "device": "WebGPU",
                    "model": endpoint_model,
                    "is_webgpu": True,
                    "precision": precision,
                    "web_backend": "transformers.js"
                }}
            except Exception as e:
                print(f"Error in WebGPU/transformers.js handler: {{e}}")
                # Return a simple dict on error
                return {{"output": "Error in WebGPU/transformers.js handler", "implementation_type": "MOCK"}}
                
        return handler

    def __test__(self):
        \"\"\"Run tests for this model implementation.\"\"\"
        results = {{}}
        examples = []
        
        # Test on CPU
        try:
            print("Testing {model_type} on CPU...")
            endpoint, processor, handler, queue, batch_size = self.init_cpu(
                model_name="test-{model_type}-model",
                model_type="{template_vars['primary_task']}"
            )
            
            # Test with simple input
            input_text = "This is a test input for {model_type}"
            output = handler(input_text)
            
            # Process with model-specific helpers
            processed_input = None
            primary_task = "{template_vars['primary_task']}"
            
            if "image" in primary_task:
                processed_input = self._process_image_input(input_text)
            elif "audio" in primary_task:
                processed_input = self._process_audio_input(input_text)
            else:
                processed_input = self._process_text_input(input_text)
                
            # Get model info using enhanced API
            model_info = self._get_model_info()
            
            # Get legacy tensor type info for backward compatibility demonstration
            tensor_types = self._get_model_tensor_types()
            
            # Record results
            examples.append({{
                "platform": "CPU",
                "input": input_text,
                "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                "implementation_type": output.get("implementation_type", "UNKNOWN"),
                "tensor_types": tensor_types,
                "model_info": {{
                    "input_format": model_info["input"]["format"],
                    "output_format": model_info["output"]["format"],
                    "helper_functions": list(model_info["helper_functions"].keys()),
                    "required_dependencies": model_info["dependencies"]["pip"][:3] if len(model_info["dependencies"]["pip"]) > 3 else model_info["dependencies"]["pip"]
                }},
                "hardware": self.hardware_capabilities
            }})
            
            results["cpu_test"] = "Success"
        except Exception as e:
            print(f"Error testing on CPU: {{e}}")
            traceback.print_exc()
            results["cpu_test"] = f"Error: {{str(e)}}"
        
        # Test on CUDA if available
        if TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
            try:
                print("Testing {model_type} on CUDA...")
                endpoint, processor, handler, queue, batch_size = self.init_cuda(
                    model_name="test-{model_type}-model",
                    model_type="{template_vars['primary_task']}"
                )
                
                # Test with simple input
                input_text = "This is a test input for {model_type} on CUDA"
                output = handler(input_text)
                
                # Process with model-specific helpers
                processed_input = None
                primary_task = "{template_vars['primary_task']}"
                
                if "image" in primary_task:
                    processed_input = self._process_image_input(input_text)
                elif "audio" in primary_task:
                    processed_input = self._process_audio_input(input_text)
                else:
                    processed_input = self._process_text_input(input_text)
                
                # Get model info using enhanced API
                model_info = self._get_model_info()
                
                # Get legacy tensor type info for backward compatibility demonstration
                tensor_types = self._get_model_tensor_types()
                
                # Record results
                examples.append({{
                    "platform": "CUDA",
                    "input": input_text,
                    "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                    "implementation_type": output.get("implementation_type", "UNKNOWN"),
                    "tensor_types": tensor_types,
                    "model_info": {{
                        "input_format": model_info["input"]["format"],
                        "output_format": model_info["output"]["format"],
                        "helper_functions": list(model_info["helper_functions"].keys()),
                        "required_dependencies": model_info["dependencies"]["pip"][:3] if len(model_info["dependencies"]["pip"]) > 3 else model_info["dependencies"]["pip"],
                        "cuda_specific": model_info["dependencies"]["optional"].get("cuda", [])
                    }},
                    "hardware": self.hardware_capabilities,
                    "precision": "fp32"  # Default precision
                }})
                
                # Test half precision (FP16) if supported
                if "precision_compatibility" in model_info and model_info["precision_compatibility"]["cuda"]["fp16"]:
                    try:
                        print("Testing {model_type} on CUDA with FP16 precision...")
                        # In a real implementation, would initialize with fp16 precision
                        output_fp16 = {{"tensor": output.get("tensor", torch.rand((1, 768))).half(),
                                     "implementation_type": "MOCK",
                                     "device": "cuda:0",
                                     "precision": "fp16"}}
                        
                        # Record FP16 results
                        examples.append({{
                            "platform": "CUDA (FP16)",
                            "input": input_text,
                            "output_type": f"container: {{str(type(output_fp16))}}, tensor: {{str(type(output_fp16.get('tensor')))}}",
                            "implementation_type": output_fp16.get("implementation_type", "UNKNOWN"),
                            "precision": "fp16",
                            "hardware": self.hardware_capabilities
                        }})
                    except Exception as e:
                        print(f"Error testing CUDA with FP16: {{e}}")
                
                # Test int8 quantization if supported
                if "precision_compatibility" in model_info and model_info["precision_compatibility"]["cuda"]["int8"]:
                    try:
                        print("Testing {model_type} on CUDA with INT8 quantization...")
                        # In a real implementation, would initialize with int8 quantization
                        # Here we just simulate it for testing
                        output_int8 = {{"tensor": output.get("tensor", torch.rand((1, 768))),
                                     "implementation_type": "MOCK",
                                     "device": "cuda:0",
                                     "precision": "int8"}}
                        
                        # Record INT8 results
                        examples.append({{
                            "platform": "CUDA (INT8)",
                            "input": input_text,
                            "output_type": f"container: {{str(type(output_int8))}}, tensor: {{str(type(output_int8.get('tensor')))}}",
                            "implementation_type": output_int8.get("implementation_type", "UNKNOWN"),
                            "precision": "int8",
                            "hardware": self.hardware_capabilities
                        }})
                    except Exception as e:
                        print(f"Error testing CUDA with INT8: {{e}}")
                
                results["cuda_test"] = "Success"
            except Exception as e:
                print(f"Error testing on CUDA: {{e}}")
                traceback.print_exc()
                results["cuda_test"] = f"Error: {{str(e)}}"
        else:
            results["cuda_test"] = "CUDA not available"
            
        # Test on AMD if available
        if self.hardware_capabilities.get("amd", False):
            try:
                print("Testing {model_type} on AMD ROCm...")
                endpoint, processor, handler, queue, batch_size = self.init_amd(
                    model_name="test-{model_type}-model",
                    model_type="{template_vars['primary_task']}",
                    device="rocm:0"
                )
                
                # Test with simple input
                input_text = "This is a test input for {model_type} on AMD ROCm"
                output = handler(input_text)
                
                # Process with model-specific helpers
                processed_input = None
                primary_task = "{template_vars['primary_task']}"
                
                if "image" in primary_task:
                    processed_input = self._process_image_input(input_text)
                elif "audio" in primary_task:
                    processed_input = self._process_audio_input(input_text)
                else:
                    processed_input = self._process_text_input(input_text)
                
                # Get model info using enhanced API
                model_info = self._get_model_info()
                
                # Record results
                examples.append({{
                    "platform": "AMD ROCm",
                    "input": input_text,
                    "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                    "implementation_type": output.get("implementation_type", "UNKNOWN"),
                    "precision": output.get("precision", "fp32"),
                    "model_info": {{
                        "input_format": model_info["input"]["format"],
                        "output_format": model_info["output"]["format"],
                        "helper_functions": list(model_info["helper_functions"].keys()),
                        "required_dependencies": model_info["dependencies"]["pip"][:3] if len(model_info["dependencies"]["pip"]) > 3 else model_info["dependencies"]["pip"],
                        "amd_specific": model_info["dependencies"]["optional"].get("amd", [])
                    }},
                    "hardware": self.hardware_capabilities
                }})
                
                # Test supported precision formats on AMD
                for precision in ["fp16", "bf16", "int8"]:
                    if "precision_compatibility" in model_info and model_info["precision_compatibility"]["amd"][precision]:
                        try:
                            print(f"Testing {{model_type}} on AMD ROCm with {{precision.upper()}} precision...")
                            endpoint_precision, processor_precision, handler_precision, _, _ = self.init_amd(
                                model_name=f"test-{{model_type}}-model-{{precision}}",
                                model_type="{template_vars['primary_task']}",
                                device="rocm:0",
                                precision=precision
                            )
                            
                            output_precision = handler_precision(input_text)
                            
                            # Record precision-specific results
                            examples.append({{
                                "platform": f"AMD ROCm ({{precision.upper()}})",
                                "input": input_text,
                                "output_type": f"container: {{str(type(output_precision))}}, tensor: {{str(type(output_precision.get('tensor', output_precision)))}}",
                                "implementation_type": output_precision.get("implementation_type", "UNKNOWN"),
                                "precision": precision,
                                "hardware": self.hardware_capabilities
                            }})
                        except Exception as e:
                            print(f"Error testing AMD ROCm with {{precision.upper()}}: {{e}}")
                
                results["amd_test"] = "Success"
            except Exception as e:
                print(f"Error testing on AMD ROCm: {{e}}")
                traceback.print_exc()
                results["amd_test"] = f"Error: {{str(e)}}"
        else:
            results["amd_test"] = "AMD ROCm not available"
            
        # Test on WebNN if available
        if self.hardware_capabilities.get("webnn", False):
            try:
                print("Testing {model_type} on WebNN...")
                endpoint, processor, handler, queue, batch_size = self.init_webnn(
                    model_name="test-{model_type}-model",
                    model_type="{template_vars['primary_task']}"
                )
                
                # Test with simple input
                input_text = "This is a test input for {model_type} on WebNN"
                output = handler(input_text)
                
                # Get model info using enhanced API
                model_info = self._get_model_info()
                
                # Record results
                examples.append({{
                    "platform": "WebNN",
                    "input": input_text,
                    "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                    "implementation_type": output.get("implementation_type", "UNKNOWN"),
                    "web_backend": output.get("web_backend", "unknown"),
                    "precision": output.get("precision", "fp32"),
                    "model_info": {{
                        "input_format": model_info["input"]["format"],
                        "output_format": model_info["output"]["format"],
                        "helper_functions": list(model_info["helper_functions"].keys()),
                        "required_dependencies": model_info["dependencies"]["pip"][:3] if len(model_info["dependencies"]["pip"]) > 3 else model_info["dependencies"]["pip"],
                        "webnn_specific": model_info["dependencies"]["optional"].get("webnn", [])
                    }},
                    "hardware": self.hardware_capabilities
                }})
                
                # Test supported precision formats on WebNN
                for precision in ["fp16", "int8"]:
                    if "precision_compatibility" in model_info and model_info["precision_compatibility"]["webnn"][precision]:
                        try:
                            print(f"Testing {{model_type}} on WebNN with {{precision.upper()}} precision...")
                            endpoint_precision, processor_precision, handler_precision, _, _ = self.init_webnn(
                                model_name=f"test-{{model_type}}-model-{{precision}}",
                                model_type="{template_vars['primary_task']}",
                                device="webnn",
                                precision=precision
                            )
                            
                            output_precision = handler_precision(input_text)
                            
                            # Record precision-specific results
                            examples.append({{
                                "platform": f"WebNN ({{precision.upper()}})",
                                "input": input_text,
                                "output_type": f"container: {{str(type(output_precision))}}, tensor: {{str(type(output_precision.get('tensor', output_precision)))}}",
                                "implementation_type": output_precision.get("implementation_type", "UNKNOWN"),
                                "precision": precision,
                                "web_backend": "webnn",
                                "hardware": self.hardware_capabilities
                            }})
                        except Exception as e:
                            print(f"Error testing WebNN with {{precision.upper()}}: {{e}}")
                
                results["webnn_test"] = "Success"
            except Exception as e:
                print(f"Error testing on WebNN: {{e}}")
                traceback.print_exc()
                results["webnn_test"] = f"Error: {{str(e)}}"
        else:
            results["webnn_test"] = "WebNN not available"
        
        # Test on WebGPU/transformers.js if available
        if self.hardware_capabilities.get("webgpu", False):
            try:
                print("Testing {model_type} on WebGPU/transformers.js...")
                endpoint, processor, handler, queue, batch_size = self.init_webgpu(
                    model_name="test-{model_type}-model",
                    model_type="{template_vars['primary_task']}"
                )
                
                # Test with simple input
                input_text = "This is a test input for {model_type} on WebGPU/transformers.js"
                output = handler(input_text)
                
                # Get model info using enhanced API
                model_info = self._get_model_info()
                
                # Record results
                examples.append({{
                    "platform": "WebGPU/transformers.js",
                    "input": input_text,
                    "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                    "implementation_type": output.get("implementation_type", "UNKNOWN"),
                    "web_backend": output.get("web_backend", "unknown"),
                    "precision": output.get("precision", "fp32"),
                    "model_info": {{
                        "input_format": model_info["input"]["format"],
                        "output_format": model_info["output"]["format"],
                        "helper_functions": list(model_info["helper_functions"].keys()),
                        "required_dependencies": model_info["dependencies"]["pip"][:3] if len(model_info["dependencies"]["pip"]) > 3 else model_info["dependencies"]["pip"],
                        "webgpu_specific": model_info["dependencies"]["optional"].get("webgpu", [])
                    }},
                    "hardware": self.hardware_capabilities
                }})
                
                # Test supported precision formats on WebGPU
                for precision in ["fp16", "int8"]:
                    if "precision_compatibility" in model_info and model_info["precision_compatibility"]["webgpu"][precision]:
                        try:
                            print(f"Testing {{model_type}} on WebGPU with {{precision.upper()}} precision...")
                            endpoint_precision, processor_precision, handler_precision, _, _ = self.init_webgpu(
                                model_name=f"test-{{model_type}}-model-{{precision}}",
                                model_type="{template_vars['primary_task']}",
                                device="webgpu",
                                precision=precision
                            )
                            
                            output_precision = handler_precision(input_text)
                            
                            # Record precision-specific results
                            examples.append({{
                                "platform": f"WebGPU ({{precision.upper()}})",
                                "input": input_text,
                                "output_type": f"container: {{str(type(output_precision))}}, tensor: {{str(type(output_precision.get('tensor', output_precision)))}}",
                                "implementation_type": output_precision.get("implementation_type", "UNKNOWN"),
                                "precision": precision,
                                "web_backend": "transformers.js",
                                "hardware": self.hardware_capabilities
                            }})
                        except Exception as e:
                            print(f"Error testing WebGPU with {{precision.upper()}}: {{e}}")
                
                results["webgpu_test"] = "Success"
            except Exception as e:
                print(f"Error testing on WebGPU/transformers.js: {{e}}")
                traceback.print_exc()
                results["webgpu_test"] = f"Error: {{str(e)}}"
        else:
            results["webgpu_test"] = "WebGPU/transformers.js not available"
        
        # Return test results
        return {{
            "results": results,
            "examples": examples,
            "timestamp": datetime.datetime.now().isoformat()
        }}

# Helper function to run the test
def run_test():
    \"\"\"Run a simple test of the {model_type} implementation.\"\"\"
    print(f"Testing {model_type} implementation...")
    
    # Create instance
    model = {template_vars['class_name']}()
    
    # Run test
    test_results = model.__test__()
    
    # Print results
    print("\\nTest Results:")
    for platform, result in test_results["results"].items():
        print(f"- {{platform}}: {{result}}")
    
    print("\\nExamples:")
    for example in test_results["examples"]:
        print(f"- Platform: {{example['platform']}}")
        print(f"  Input: {{example['input']}}")
        print(f"  Output Type: {{example['output_type']}}")
        print(f"  Implementation: {{example['implementation_type']}}")
        
        # Print tensor type information (legacy format)
        if 'tensor_types' in example:
            print(f"  Tensor Types (Legacy Format):")
            for k, v in example['tensor_types'].items():
                print(f"    {k}: {v}")
        
        # Print enhanced model information
        if 'model_info' in example:
            print(f"  Enhanced Model Information:")
            print(f"    Input Format: {example['model_info']['input_format']}")
            print(f"    Output Format: {example['model_info']['output_format']}")
            print(f"    Helper Functions: {', '.join(example['model_info']['helper_functions'])}")
            print(f"    Required Dependencies: {', '.join(example['model_info']['required_dependencies'])}")
            
            # Print hardware-specific dependencies
            for hw_type in ['cuda_specific', 'amd_specific', 'apple_specific', 'qualcomm_specific',
                         'webnn_specific', 'webgpu_specific']:
                if hw_type in example['model_info'] and example['model_info'][hw_type]:
                    print(f"    {hw_type.replace('_specific', '').upper()}-Specific Dependencies: {', '.join(example['model_info'][hw_type])}")
        
        # Print precision information
        if 'precision' in example:
            print(f"  Precision: {example['precision'].upper()}")
            
        # Print web backend information
        if 'web_backend' in example:
            print(f"  Web Backend: {example['web_backend']}")
                
        # Print hardware capabilities
        if 'hardware' in example:
            print(f"  Hardware Capabilities:")
            for k, v in example['hardware'].items():
                if v is not None and v is not False:
                    print(f"    {k}: {v}")
        print("")
    
    return test_results

if __name__ == "__main__":
    run_test()
"""
    
    # Write the test file
    with open(output_file, 'w') as f:
        f.write(template)
    
    print(f"Generated test file: {output_file}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Template Test Generator")
    parser.add_argument("--model", type=str, required=True, help="Model type to generate a test for")
    parser.add_argument("--output-dir", type=str, default=str(SKILLS_DIR), help="Output directory for the test file")
    parser.add_argument("--force", action="store_true", help="Force overwrite if file exists")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the test file
    success = generate_test_file(args.model, output_dir, args.force)
    
    if success:
        print(f"Successfully generated test for {args.model}")
        return 0
    else:
        print(f"Failed to generate test for {args.model}")
        return 1

if __name__ == "__main__":
    sys.exit(main())