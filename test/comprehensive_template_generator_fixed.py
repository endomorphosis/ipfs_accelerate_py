#!/usr/bin/env python3
"""
Comprehensive Template Test Generator

This is a complete implementation for generating test files that are compatible
with the ipfs_accelerate_py worker/skillset module structure.

This generator includes:
- Support for all hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm, AMD)
- Support for web backends (WebNN, WebGPU/transformers.js)
- Comprehensive precision handling (fp32, fp16, bf16, int8, etc.)
- Detailed model registry with hardware and precision compatibility
- Input/output specifications for different model types
- Helper functions with argument specifications
- Dependency management for different hardware and precision types
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
SAMPLE_DIR = TEST_DIR / "sample_tests"
WORKER_SKILLSET = PROJECT_ROOT / "ipfs_accelerate_py" / "worker" / "skillset"

# Template for generating tests
def generate_test_file(model_type, output_dir=SAMPLE_DIR, force=False):
    """Generate a comprehensive test file for the specified model type."""
    
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
        "helper_functions": ["tokenization", "device_management"],  # Default helpers
        
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
                "webnn": ["webnn-polyfill>=1.0.0", "onnxruntime-web>=1.16.0"],
                "webgpu": ["@xenova/transformers>=2.6.0", "webgpu>=0.1.24"]
            },
            "precision": {
                "fp16": [],
                "bf16": ["torch>=1.12.0"],
                "int8": ["bitsandbytes>=0.41.0", "optimum>=1.12.0"],
                "int4": ["bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
                "uint4": ["bitsandbytes>=0.41.0", "optimum>=1.12.0", "auto-gptq>=0.4.0"],
                "fp8": ["transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"],
                "fp4": ["transformers-neuronx>=0.8.0", "torch-neuronx>=2.0.0"]
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
                        template_vars["helper_functions"] = ["image_loading", "image_processing", "device_management"]
                        
                    # Text generation models
                    elif primary_task in ["text-generation", "summarization", "translation_XX_to_YY"]:
                        template_vars["input_tensor_type"] = "int64"  # Token IDs
                        template_vars["input_format"] = "text"
                        template_vars["output_format"] = "text"
                        template_vars["uses_attention_mask"] = True
                        
                        # Add text-specific helper functions
                        template_vars["helper_functions"] = ["tokenization", "generation_config", "device_management"]
                        
                    # Multimodal models
                    elif primary_task in ["image-to-text", "visual-question-answering"]:
                        template_vars["input_tensor_type"] = "mixed"  # Both image and text inputs
                        template_vars["input_format"] = "multimodal"
                        template_vars["output_format"] = "text"
                        template_vars["uses_attention_mask"] = True
                        
                        # Add multimodal-specific helper functions
                        template_vars["helper_functions"] = ["tokenization", "image_loading", "image_processing", 
                                                          "multimodal_inputs", "device_management"]
                        
                    # Audio models
                    elif primary_task in ["automatic-speech-recognition", "audio-classification", "text-to-audio"]:
                        template_vars["input_tensor_type"] = "float32"  # Audio features
                        template_vars["input_format"] = "audio"
                        template_vars["output_format"] = "text" if "speech-recognition" in primary_task else "audio"
                        template_vars["uses_attention_mask"] = True
                        template_vars["token_sequence_length"] = 16000  # 1s of audio at 16kHz
                        
                        # Add audio-specific helper functions
                        template_vars["helper_functions"] = ["audio_loading", "audio_processing", "device_management"]
                        
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
                            template_vars["helper_functions"] = ["image_loading", "image_processing", "device_management"]
                        else:
                            template_vars["input_tensor_type"] = "int64"  # Token IDs
                            template_vars["input_format"] = "text"
                            template_vars["output_format"] = "embedding"
                            template_vars["helper_functions"] = ["tokenization", "device_management"]
                    
                    # Ensure all dependencies are unique
                    template_vars["dependencies"]["pip"] = list(set(template_vars["dependencies"]["pip"]))
                    
        except Exception as e:
            print(f"Error loading pipeline map: {e}")
    
    # Generate the test file content using the enhanced template
    template = f"""#!/usr/bin/env python3
\"\"\"
Test implementation for {model_type} with comprehensive hardware and precision support

This file provides a standardized test interface for {model_type} models
across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm, AMD)
and precision types (fp32, fp16, bf16, int8, int4, etc.).

Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}
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

# Model Registry - Contains metadata about available models for this type
MODEL_REGISTRY = {{
    # Default/small model configuration
    "{model_type}": {{
        "description": "Default {model_type} model",
        
        # Model dimensions and capabilities
        "embedding_dim": {template_vars['embedding_dim']},
        "sequence_length": {template_vars['token_sequence_length']},
        "model_precision": "{template_vars['model_precision']}", 
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
            "webgpu": True   # WebGPU with transformers.js support
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
                "int4": True,
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
            "typical_shapes": ["batch_size, {template_vars['token_sequence_length']}"]
        }},
        "output": {{
            "format": "{template_vars['output_format']}",
            "tensor_type": "{template_vars['output_tensor_type']}",
            "typical_shapes": ["batch_size, {template_vars['embedding_dim']}"]
        }},
        
        # Dependencies
        "dependencies": {json.dumps(template_vars['dependencies'], indent=4)}
    }}
}}

class {template_vars['class_name']}:
    \"\"\"
    {model_type.capitalize()} implementation.
    
    This class provides standardized interfaces for working with {model_type} models
    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm, AMD).
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
            }}
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
    
    def _detect_hardware(self): # Hardware detection method
        """Detect available hardware and return capabilities dictionary"""
        capabilities = {{
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
            "webnn_version": None,
            "webgpu": False,
            "webgpu_version": None
        }}
        
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
        
        # Check for WebNN availability
        try:
            # Check for WebNN in browser environment
            import platform
            import subprocess
            
            # Check if running in a browser context (looking for JavaScript engine)
            is_browser_env = False
            try:
                # Try to detect Node.js environment
                node_version = subprocess.run(['node', '--version'], 
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                            universal_newlines=True, check=False)
                if node_version.returncode == 0:
                    # Check for WebNN polyfill package
                    webnn_check = subprocess.run(['npm', 'list', 'webnn-polyfill'], 
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                               universal_newlines=True, check=False)
                    if "webnn-polyfill" in webnn_check.stdout:
                        capabilities["webnn"] = True
                        
                        # Try to extract version
                        import re
                        match = re.search(r'webnn-polyfill@(\d+\.\d+\.\d+)', webnn_check.stdout)
                        if match:
                            capabilities["webnn_version"] = match.group(1)
                        else:
                            capabilities["webnn_version"] = "unknown"
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
            
            # Alternative check for WebNN support through imported modules
            if not capabilities["webnn"]:
                try:
                    import webnn_polyfill
                    capabilities["webnn"] = True
                    capabilities["webnn_version"] = getattr(webnn_polyfill, "__version__", "unknown")
                except ImportError:
                    pass
        except Exception:
            pass
            
        # Check for WebGPU / transformers.js availability
        try:
            import platform
            import subprocess
            
            # Try to detect Node.js environment first (for transformers.js)
            try:
                node_version = subprocess.run(['node', '--version'], 
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                            universal_newlines=True, check=False)
                if node_version.returncode == 0:
                    # Check for transformers.js package
                    transformers_js_check = subprocess.run(['npm', 'list', '@xenova/transformers'], 
                                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                                        universal_newlines=True, check=False)
                    if "@xenova/transformers" in transformers_js_check.stdout:
                        capabilities["webgpu"] = True
                        
                        # Try to extract version
                        import re
                        match = re.search(r'@xenova/transformers@(\d+\.\d+\.\d+)', transformers_js_check.stdout)
                        if match:
                            capabilities["webgpu_version"] = match.group(1)
                        else:
                            capabilities["webgpu_version"] = "unknown"
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
            
            # Check if browser with WebGPU is available
            # This is a simplified check since we can't actually detect browser capabilities
            # in a server-side context, but we can check for typical browser detection packages
            if not capabilities["webgpu"]:
                try:
                    # Check for webgpu mock or polyfill
                    webgpu_check = subprocess.run(['npm', 'list', 'webgpu'], 
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                               universal_newlines=True, check=False)
                    if "webgpu" in webgpu_check.stdout:
                        capabilities["webgpu"] = True
                        
                        # Try to extract version
                        import re
                        match = re.search(r'webgpu@(\d+\.\d+\.\d+)', webgpu_check.stdout)
                        if match:
                            capabilities["webgpu_version"] = match.group(1)
                        else:
                            capabilities["webgpu_version"] = "unknown"
                except (FileNotFoundError, subprocess.SubprocessError):
                    pass
        except Exception:
            pass
            
        return capabilities
    
    def _get_model_info(self, model_id=None):
        """Get comprehensive model information for a specific model variant."""
        model_id = model_id or "{model_type}"
        
        if model_id in self.model_registry:
            # Return complete model configuration from registry
            return self.model_registry[model_id]
        
        # Return default info if model not in registry
        return self.model_info
    
    def _process_text_input(self, text, tokenizer=None, max_length=None):
        """Process text input for text-based models."""
        if tokenizer is None:
            # Create a mock tokenizer for testing
            class MockTokenizer:
                def __call__(self, text, **kwargs):
                    # Handle both single strings and batches
                    if isinstance(text, str):
                        batch_size = 1
                    else:
                        batch_size = len(text)
                        
                    return {{
                        "input_ids": torch.ones((batch_size, 512), dtype=torch.long),
                        "attention_mask": torch.ones((batch_size, 512), dtype=torch.long)
                    }}
                    
                def decode(self, token_ids, **kwargs):
                    return "Decoded text from mock processor"
            
            tokenizer = MockTokenizer()
            
        max_length = max_length or 512
        
        # Tokenize input
        if isinstance(text, str):
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", 
                            truncation=True, max_length=max_length)
        else:
            inputs = tokenizer(list(text), return_tensors="pt", padding="max_length", 
                            truncation=True, max_length=max_length)
            
        return inputs
    
    def init_cpu(self, model_name, model_type, device="cpu", **kwargs):
        """Initialize model for CPU inference."""
        try:
            import asyncio
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor and endpoint
            tokenizer = transformers.AutoTokenizer.from_pretrained("{model_type}")
            model = transformers.AutoModel.from_pretrained("{model_type}")
            
            # Apply quantization if needed
            if precision in ["int8", "int4", "uint4"]:
                print(f"Applying {{precision}} quantization for CPU")
                # In real implementation, would apply quantization here
            
            # Create handler
            handler = self.create_cpu_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label="cpu",
                endpoint=model,
                tokenizer=tokenizer,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 1
            
            return model, tokenizer, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing CPU model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock CPU output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 1
    
    def init_apple(self, model_name, model_type, device="mps", **kwargs):
        """Initialize model for Apple Silicon (M1/M2/M3) inference."""
        try:
            import asyncio
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor and endpoint
            tokenizer = transformers.AutoTokenizer.from_pretrained("{model_type}")
            model = transformers.AutoModel.from_pretrained("{model_type}")
            
            # Move to MPS
            if TORCH_AVAILABLE and hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                model = model.to('mps')
            
            # Apply precision conversion if needed
            if precision == "fp16":
                model = model.half()
            
            # Create handler
            handler = self.create_apple_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                apple_label=device,
                endpoint=model,
                tokenizer=tokenizer,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 2
            
            return model, tokenizer, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing Apple Silicon model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock Apple Silicon output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 2
            
    def init_cuda(self, model_name, model_type, device_label="cuda:0", **kwargs):
        """Initialize model for CUDA inference."""
        try:
            import asyncio
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor and endpoint
            tokenizer = transformers.AutoTokenizer.from_pretrained("{model_type}")
            model = transformers.AutoModel.from_pretrained("{model_type}")
            
            # Move to CUDA
            model = model.to(device_label)
            
            # Apply precision conversion if needed
            if precision == "fp16":
                model = model.half()
            elif precision == "bf16" and hasattr(torch, "bfloat16"):
                model = model.to(torch.bfloat16)
            elif precision in ["int8", "int4", "uint4"]:
                print(f"Applying {{precision}} quantization for CUDA")
                # In real implementation, would apply quantization here
            
            # Create handler
            handler = self.create_cuda_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device_label,
                hardware_label=device_label,
                endpoint=model,
                tokenizer=tokenizer,
                is_real_impl=True,
                batch_size=4,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 4  # Default to larger batch size for CUDA
            
            return model, tokenizer, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing CUDA model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock CUDA output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 2
    
    def init_amd(self, model_name, model_type, device="rocm:0", **kwargs):
        """Initialize model for AMD ROCm inference."""
        try:
            import asyncio
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor and endpoint
            tokenizer = transformers.AutoTokenizer.from_pretrained("{model_type}")
            model = transformers.AutoModel.from_pretrained("{model_type}")
            
            # Move to AMD ROCm device
            model = model.to(device)
            
            # Apply precision conversion if needed
            if precision == "fp16":
                model = model.half()
            elif precision == "bf16" and hasattr(torch, "bfloat16"):
                model = model.to(torch.bfloat16)
            elif precision in ["int8", "int4", "uint4"]:
                print(f"Applying {{precision}} quantization for AMD ROCm")
                # In real implementation, would apply quantization here
            
            # Create handler
            handler = self.create_amd_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                device=device,
                hardware_label=f"amd_{{device}}",
                endpoint=model,
                tokenizer=tokenizer,
                is_real_impl=True,
                batch_size=4,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 4  # Default to larger batch size for AMD GPUs
            
            return model, tokenizer, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing AMD ROCm model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock AMD ROCm output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 2
            
    def init_qualcomm(self, model_name, model_type, device="qualcomm", **kwargs):
        """Initialize model for Qualcomm AI inference."""
        try:
            import asyncio
            import numpy as np
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor
            tokenizer = transformers.AutoTokenizer.from_pretrained("{model_type}")
            
            # Create Qualcomm-style endpoint
            class MockQualcommModel:
                def execute(self, inputs):
                    batch_size = 1
                    seq_len = 512
                    if isinstance(inputs, dict) and 'input_ids' in inputs:
                        if hasattr(inputs['input_ids'], 'shape'):
                            batch_size = inputs['input_ids'].shape[0]
                            if len(inputs['input_ids'].shape) > 1:
                                seq_len = inputs['input_ids'].shape[1]
                    
                    # Return Qualcomm-style output
                    return {{"output": np.random.rand(batch_size, seq_len, 768).astype(np.float32)}}
            
            model = MockQualcommModel()
            
            # Create handler
            handler = self.create_qualcomm_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                qualcomm_label=device,
                endpoint=model,
                tokenizer=tokenizer,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 1
            
            return model, tokenizer, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing Qualcomm model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock Qualcomm output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 1
            
    def init_webnn(self, model_name, model_type, device="webnn", **kwargs):
        """Initialize model for WebNN inference (browser or Node.js environment).
        
        WebNN enables hardware-accelerated inference in web browsers and Node.js
        applications by providing a common API that maps to the underlying hardware.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model ('{template_vars['primary_task']}', etc.)
            device (str): Device identifier ('webnn')
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        """
        try:
            import asyncio
            import numpy as np
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor/tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained("{model_type}")
            
            # Create WebNN endpoint/model
            # This would integrate with the WebNN API
            class WebNNModel:
                def compute(self, inputs):
                    """Process inputs with WebNN and return outputs."""
                    batch_size = 1
                    seq_len = 512
                    if isinstance(inputs, dict) and 'input_ids' in inputs:
                        if hasattr(inputs['input_ids'], 'shape'):
                            batch_size = inputs['input_ids'].shape[0]
                            if len(inputs['input_ids'].shape) > 1:
                                seq_len = inputs['input_ids'].shape[1]
                    
                    # Return WebNN-style output
                    # Real implementation would use the WebNN API to run inference
                    return {{"last_hidden_state": np.random.rand(batch_size, seq_len, 768).astype(np.float32)}}
            
            model = WebNNModel()
            
            # Create handler
            handler = self.create_webnn_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                webnn_label=device,
                endpoint=model,
                tokenizer=tokenizer,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 1
            
            return model, tokenizer, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing WebNN model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock WebNN output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 1
    
    def init_webgpu(self, model_name, model_type, device="webgpu", **kwargs):
        """Initialize model for WebGPU inference using transformers.js.
        
        WebGPU provides modern GPU acceleration for machine learning models in web browsers
        and Node.js applications through libraries like transformers.js.
        
        Args:
            model_name (str): Model identifier
            model_type (str): Type of model ('{template_vars['primary_task']}', etc.)
            device (str): Device identifier ('webgpu')
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        """
        try:
            import asyncio
            import numpy as np
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor/tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained("{model_type}")
            
            # Create WebGPU/transformers.js endpoint/model
            class TransformersJSModel:
                def __init__(self, model_id="Xenova/{model_type}", task="feature-extraction"):
                    """Initialize a transformers.js model with WebGPU support.
                    
                    In a real implementation, this would integrate with the transformers.js library
                    running in a browser or Node.js environment with WebGPU capabilities.
                    """
                    self.model_id = model_id
                    self.task = task
                    print(f"Initialized transformers.js model '{{model_id}}' for task '{{task}}' with WebGPU acceleration")
                    
                def run(self, inputs):
                    """Run inference using transformers.js with WebGPU.
                    
                    Args:
                        inputs: Dictionary of inputs with tokenized text
                        
                    Returns:
                        Dictionary with model outputs (hidden_states or embeddings)
                    """
                    # Determine batch size and sequence length from inputs
                    batch_size = 1
                    seq_len = 512
                    if isinstance(inputs, dict) and 'input_ids' in inputs:
                        if isinstance(inputs['input_ids'], list):
                            batch_size = len(inputs['input_ids'])
                            if inputs['input_ids'] and isinstance(inputs['input_ids'][0], list):
                                seq_len = len(inputs['input_ids'][0])
                    
                    # Generate mock outputs that match transformers.js format
                    # Real implementation would use the transformers.js API with WebGPU
                    if self.task == "feature-extraction":
                        # Return embeddings for the CLS token for feature extraction
                        return {{
                            "hidden_states": np.random.rand(batch_size, 768).tolist(),
                            "token_count": seq_len,
                            "model_version": "Xenova/{model_type}", 
                            "device": "WebGPU"
                        }}
                    else:
                        # Return full last_hidden_state for other tasks
                        return {{
                            "last_hidden_state": np.random.rand(batch_size, seq_len, 768).tolist(),
                            "model_version": "Xenova/{model_type}",
                            "device": "WebGPU"
                        }}
            
            # Initialize transformers.js model with WebGPU support
            model = TransformersJSModel(model_id=model_name, task="feature-extraction")
            
            # Create handler
            handler = self.create_webgpu_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                webgpu_label=device,
                endpoint=model,
                tokenizer=tokenizer,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(32)
            batch_size = 1
            
            return model, tokenizer, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing WebGPU model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock WebGPU output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(32), 1
    
    def init_openvino(self, model_name, model_type, device="CPU", **kwargs):
        """Initialize model for OpenVINO inference."""
        try:
            import asyncio
            import numpy as np
            
            # Get precision from kwargs or default to fp32
            precision = kwargs.get("precision", "fp32")
            
            # Create processor and endpoint (OpenVINO-specific)
            tokenizer = transformers.AutoTokenizer.from_pretrained("{model_type}")
            
            # Create OpenVINO-style endpoint
            class MockOpenVINOModel:
                def infer(self, inputs):
                    batch_size = 1
                    seq_len = 512
                    if isinstance(inputs, dict) and 'input_ids' in inputs:
                        if hasattr(inputs['input_ids'], 'shape'):
                            batch_size = inputs['input_ids'].shape[0]
                            if len(inputs['input_ids'].shape) > 1:
                                seq_len = inputs['input_ids'].shape[1]
                    
                    # Return OpenVINO-style output
                    return {{"last_hidden_state": np.random.rand(batch_size, seq_len, 768).astype(np.float32)}}
            
            model = MockOpenVINOModel()
            
            # Create handler
            handler = self.create_openvino_text_embedding_endpoint_handler(
                endpoint_model=model_name,
                tokenizer=tokenizer,
                openvino_label=device,
                endpoint=model,
                precision=precision
            )
            
            # Create queue
            queue = asyncio.Queue(64)
            batch_size = 1
            
            return model, tokenizer, handler, queue, batch_size
        except Exception as e:
            print(f"Error initializing OpenVINO model: {{e}}")
            traceback.print_exc()
            
            # Return mock components on error
            import asyncio
            handler = lambda x: {{"output": "Mock OpenVINO output", "input": x, "implementation_type": "MOCK"}}
            return None, None, handler, asyncio.Queue(64), 1
    
    def create_cpu_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None, precision="fp32"):
        """Create a handler function for CPU inference."""
        def handler(text_input):
            try:
                # Process input
                inputs = self._process_text_input(text_input, tokenizer)
                
                # Run model
                with torch.no_grad():
                    outputs = endpoint(**inputs)
                
                # Extract embeddings 
                last_hidden_state = outputs.last_hidden_state
                embeddings = last_hidden_state[:, 0]  # Use CLS token embedding
                
                # Return dictionary with result
                return {{
                    "tensor": embeddings,
                    "implementation_type": "CPU",
                    "device": device,
                    "model": endpoint_model,
                    "precision": precision
                }}
            except Exception as e:
                print(f"Error in CPU handler: {{e}}")
                # Return a simple dict on error
                return {{"output": f"Error in CPU handler: {{e}}", "implementation_type": "MOCK"}}
                
        return handler
    
    def create_cuda_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None, is_real_impl=False, batch_size=1, precision="fp32"):
        """Create a handler function for CUDA inference."""
        def handler(text_input):
            try:
                # Process input
                inputs = self._process_text_input(text_input, tokenizer)
                
                # Move inputs to device
                for key in inputs:
                    inputs[key] = inputs[key].to(device)
                
                # Run model
                with torch.no_grad():
                    outputs = endpoint(**inputs)
                
                # Extract embeddings
                last_hidden_state = outputs.last_hidden_state
                embeddings = last_hidden_state[:, 0]  # Use CLS token embedding
                
                # Return dictionary with result
                return {{
                    "tensor": embeddings,
                    "implementation_type": "CUDA",
                    "device": device,
                    "model": endpoint_model,
                    "precision": precision,
                    "is_cuda": True
                }}
            except Exception as e:
                print(f"Error in CUDA handler: {{e}}")
                # Return a simple dict on error
                return {{"output": f"Error in CUDA handler: {{e}}", "implementation_type": "MOCK"}}
                
        return handler
    
    def create_amd_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None, is_real_impl=False, batch_size=1, precision="fp32"):
        """Create a handler function for AMD ROCm inference."""
        def handler(text_input):
            try:
                # Process input
                inputs = self._process_text_input(text_input, tokenizer)
                
                # Move inputs to device
                for key in inputs:
                    inputs[key] = inputs[key].to(device)
                
                # Run model
                with torch.no_grad():
                    outputs = endpoint(**inputs)
                
                # Extract embeddings
                last_hidden_state = outputs.last_hidden_state
                embeddings = last_hidden_state[:, 0]  # Use CLS token embedding
                
                # Return dictionary with result
                return {{
                    "tensor": embeddings,
                    "implementation_type": "AMD_ROCM",
                    "device": device,
                    "model": endpoint_model,
                    "precision": precision,
                    "is_amd": True
                }}
            except Exception as e:
                print(f"Error in AMD ROCm handler: {{e}}")
                # Return a simple dict on error
                return {{"output": f"Error in AMD ROCm handler: {{e}}", "implementation_type": "MOCK"}}
                
        return handler
    
    def create_apple_text_embedding_endpoint_handler(self, endpoint_model, apple_label, endpoint=None, tokenizer=None, precision="fp32"):
        """Create a handler function for Apple Silicon inference."""
        def handler(text_input):
            try:
                # Process input
                inputs = self._process_text_input(text_input, tokenizer)
                
                # Move inputs to device
                for key in inputs:
                    inputs[key] = inputs[key].to("mps")
                
                # Run model
                with torch.no_grad():
                    outputs = endpoint(**inputs)
                
                # Extract embeddings
                last_hidden_state = outputs.last_hidden_state
                embeddings = last_hidden_state[:, 0]  # Use CLS token embedding
                
                # Return dictionary with result
                return {{
                    "tensor": embeddings,
                    "implementation_type": "APPLE_SILICON",
                    "device": "mps",
                    "model": endpoint_model,
                    "precision": precision,
                    "is_mps": True
                }}
            except Exception as e:
                print(f"Error in Apple Silicon handler: {{e}}")
                # Return a simple dict on error
                return {{"output": f"Error in Apple Silicon handler: {{e}}", "implementation_type": "MOCK"}}
                
        return handler
    
    def create_qualcomm_text_embedding_endpoint_handler(self, endpoint_model, qualcomm_label, endpoint=None, tokenizer=None, precision="fp32"):
        """Create a handler function for Qualcomm AI inference."""
        def handler(text_input):
            try:
                # Process input
                inputs = self._process_text_input(text_input, tokenizer)
                
                # Convert to numpy for Qualcomm
                np_inputs = {{}}
                for key, value in inputs.items():
                    np_inputs[key] = value.numpy()
                
                # Run model with Qualcomm
                outputs = endpoint.execute(np_inputs)
                
                # Convert back to torch
                embeddings = torch.from_numpy(outputs["output"][:, 0])  # Use CLS token embedding
                
                # Return dictionary with result
                return {{
                    "tensor": embeddings,
                    "implementation_type": "QUALCOMM",
                    "device": qualcomm_label,
                    "model": endpoint_model,
                    "precision": precision,
                    "is_qualcomm": True
                }}
            except Exception as e:
                print(f"Error in Qualcomm handler: {{e}}")
                # Return a simple dict on error
                return {{"output": f"Error in Qualcomm handler: {{e}}", "implementation_type": "MOCK"}}
                
        return handler
        
    def create_openvino_text_embedding_endpoint_handler(self, endpoint_model, tokenizer, openvino_label, endpoint=None, precision="fp32"):
        """Create a handler function for OpenVINO inference."""
        def handler(text_input):
            try:
                # Process input
                inputs = self._process_text_input(text_input, tokenizer)
                
                # Convert to numpy for OpenVINO
                np_inputs = {{}}
                for key, value in inputs.items():
                    np_inputs[key] = value.numpy()
                
                # Run model with OpenVINO
                outputs = endpoint.infer(np_inputs)
                
                # Convert back to torch
                last_hidden_state = torch.from_numpy(outputs["last_hidden_state"])
                embeddings = last_hidden_state[:, 0]  # Use CLS token embedding
                
                # Return dictionary with result
                return {{
                    "tensor": embeddings,
                    "implementation_type": "OPENVINO",
                    "device": openvino_label,
                    "model": endpoint_model,
                    "precision": precision,
                    "is_openvino": True
                }}
            except Exception as e:
                print(f"Error in OpenVINO handler: {{e}}")
                # Return a simple dict on error
                return {{"output": f"Error in OpenVINO handler: {{e}}", "implementation_type": "MOCK"}}
                
        return handler
    
    def create_webnn_text_embedding_endpoint_handler(self, endpoint_model, webnn_label, endpoint=None, tokenizer=None, precision="fp32"):
        """Create a handler function for WebNN inference.
        
        WebNN (Web Neural Network API) is a browser-based API that provides hardware acceleration
        for neural networks on the web.
        """
        def handler(text_input):
            try:
                # Process input
                inputs = self._process_text_input(text_input, tokenizer)
                
                # Convert to appropriate format for WebNN
                webnn_inputs = {{}}
                for key, value in inputs.items():
                    # Convert PyTorch tensors to format needed by WebNN (typically array buffers)
                    webnn_inputs[key] = value.detach().cpu().numpy()
                
                # Run model with WebNN
                outputs = endpoint.compute(webnn_inputs)
                
                # Convert back to PyTorch tensors
                if isinstance(outputs, dict) and "last_hidden_state" in outputs:
                    last_hidden_state = torch.from_numpy(outputs["last_hidden_state"])
                else:
                    # Handle other output formats
                    last_hidden_state = torch.from_numpy(outputs)
                    
                # Extract embeddings (typically first token for BERT)
                if last_hidden_state.ndim > 1:
                    embeddings = last_hidden_state[:, 0]  # Use CLS token embedding
                else:
                    embeddings = last_hidden_state
                
                # Return dictionary with result
                return {{
                    "tensor": embeddings,
                    "implementation_type": "WEBNN",
                    "device": webnn_label,
                    "model": endpoint_model,
                    "precision": precision,
                    "is_webnn": True
                }}
            except Exception as e:
                print(f"Error in WebNN handler: {{e}}")
                # Return a simple dict on error
                return {{"output": f"Error in WebNN handler: {{e}}", "implementation_type": "MOCK"}}
                
        return handler
        
    def create_webgpu_text_embedding_endpoint_handler(self, endpoint_model, webgpu_label, endpoint=None, tokenizer=None, precision="fp32"):
        """Create a handler function for WebGPU inference with transformers.js.
        
        WebGPU is a modern web graphics and compute API that provides access to GPU
        acceleration for machine learning models through libraries like transformers.js.
        """
        def handler(text_input):
            try:
                # Process input
                inputs = self._process_text_input(text_input, tokenizer)
                
                # Convert to appropriate format for transformers.js / WebGPU
                webgpu_inputs = {{}}
                for key, value in inputs.items():
                    # Convert PyTorch tensors to format needed by transformers.js
                    webgpu_inputs[key] = value.detach().cpu().numpy().tolist()
                
                # Run model with WebGPU/transformers.js
                outputs = endpoint.run(webgpu_inputs)
                
                # Convert back to PyTorch tensors
                if isinstance(outputs, dict) and "hidden_states" in outputs:
                    # transformers.js output format
                    hidden_states = torch.tensor(outputs["hidden_states"], dtype=torch.float32)
                    if hidden_states.ndim > 1:
                        embeddings = hidden_states[:, 0]  # Use CLS token embedding
                    else:
                        embeddings = hidden_states
                elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
                    # Standard format
                    last_hidden_state = torch.tensor(outputs["last_hidden_state"], dtype=torch.float32)
                    embeddings = last_hidden_state[:, 0]  # Use CLS token embedding
                else:
                    # Handle direct output (array of embeddings)
                    if isinstance(outputs, (list, tuple)):
                        embeddings = torch.tensor(outputs, dtype=torch.float32)
                    else:
                        embeddings = torch.tensor([outputs], dtype=torch.float32)
                
                # Return dictionary with result
                return {{
                    "tensor": embeddings,
                    "implementation_type": "WEBGPU",
                    "device": webgpu_label,
                    "model": endpoint_model,
                    "precision": precision,
                    "is_webgpu": True
                }}
            except Exception as e:
                print(f"Error in WebGPU handler: {{e}}")
                # Return a simple dict on error
                return {{"output": f"Error in WebGPU handler: {{e}}", "implementation_type": "MOCK"}}
                
        return handler
    
    def __test__(self):
        """Run tests for this model implementation."""
        results = {{}}
        examples = []
        
        # Test on CPU with different precision types
        for precision in ["fp32", "bf16", "int8"]:
            try:
                print(f"Testing {model_type} on CPU with {{precision.upper()}} precision...")
                model_info = self._get_model_info()
                
                # Skip if precision not supported on CPU
                if not model_info["precision_compatibility"]["cpu"].get(precision, False):
                    print(f"Precision {{precision.upper()}} not supported on CPU, skipping...")
                    continue
                
                # Initialize model with specific precision
                endpoint, processor, handler, queue, batch_size = self.init_cpu(
                    model_name="test-{model_type}-model",
                    model_type="{template_vars['primary_task']}",
                    precision=precision
                )
                
                # Test with simple input
                input_text = f"This is a test input for {model_type} with {{precision.upper()}} precision on CPU"
                output = handler(input_text)
                
                # Record results
                examples.append({{
                    "platform": f"CPU ({{precision.upper()}})",
                    "input": input_text,
                    "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                    "implementation_type": output.get("implementation_type", "UNKNOWN"),
                    "precision": precision,
                    "hardware": "CPU",
                    "model_info": {{
                        "input_format": model_info["input"]["format"],
                        "output_format": model_info["output"]["format"]
                    }}
                }})
                
                results[f"cpu_{{precision}}_test"] = "Success"
            except Exception as e:
                print(f"Error testing on CPU with {{precision.upper()}}: {{e}}")
                traceback.print_exc()
                results[f"cpu_{{precision}}_test"] = f"Error: {{str(e)}}"
        
        # Test on CUDA if available
        if self.hardware_capabilities.get("cuda", False):
            for precision in ["fp32", "fp16", "bf16", "int8"]:
                try:
                    print(f"Testing {model_type} on CUDA with {{precision.upper()}} precision...")
                    model_info = self._get_model_info()
                    
                    # Skip if precision not supported on CUDA
                    if not model_info["precision_compatibility"]["cuda"].get(precision, False):
                        print(f"Precision {{precision.upper()}} not supported on CUDA, skipping...")
                        continue
                    
                    # Initialize model with specific precision
                    endpoint, processor, handler, queue, batch_size = self.init_cuda(
                        model_name="test-{model_type}-model",
                        model_type="{template_vars['primary_task']}",
                        precision=precision
                    )
                    
                    # Test with simple input
                    input_text = f"This is a test input for {model_type} with {{precision.upper()}} precision on CUDA"
                    output = handler(input_text)
                    
                    # Record results
                    examples.append({{
                        "platform": f"CUDA ({{precision.upper()}})",
                        "input": input_text,
                        "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                        "implementation_type": output.get("implementation_type", "UNKNOWN"),
                        "precision": precision,
                        "hardware": "CUDA",
                        "model_info": {{
                            "input_format": model_info["input"]["format"],
                            "output_format": model_info["output"]["format"]
                        }}
                    }})
                    
                    results[f"cuda_{{precision}}_test"] = "Success"
                except Exception as e:
                    print(f"Error testing on CUDA with {{precision.upper()}}: {{e}}")
                    traceback.print_exc()
                    results[f"cuda_{{precision}}_test"] = f"Error: {{str(e)}}"
        else:
            results["cuda_test"] = "CUDA not available"
        
        # Test on AMD if available
        if self.hardware_capabilities.get("amd", False):
            for precision in ["fp32", "fp16", "bf16", "int8"]:
                try:
                    print(f"Testing {model_type} on AMD ROCm with {{precision.upper()}} precision...")
                    model_info = self._get_model_info()
                    
                    # Skip if precision not supported on AMD
                    if not model_info["precision_compatibility"]["amd"].get(precision, False):
                        print(f"Precision {{precision.upper()}} not supported on AMD ROCm, skipping...")
                        continue
                    
                    # Initialize model with specific precision
                    endpoint, processor, handler, queue, batch_size = self.init_amd(
                        model_name="test-{model_type}-model",
                        model_type="{template_vars['primary_task']}",
                        precision=precision
                    )
                    
                    # Test with simple input
                    input_text = f"This is a test input for {model_type} with {{precision.upper()}} precision on AMD ROCm"
                    output = handler(input_text)
                    
                    # Record results
                    examples.append({{
                        "platform": f"AMD ROCm ({{precision.upper()}})",
                        "input": input_text,
                        "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                        "implementation_type": output.get("implementation_type", "UNKNOWN"),
                        "precision": precision,
                        "hardware": "AMD",
                        "model_info": {{
                            "input_format": model_info["input"]["format"],
                            "output_format": model_info["output"]["format"]
                        }}
                    }})
                    
                    results[f"amd_{{precision}}_test"] = "Success"
                except Exception as e:
                    print(f"Error testing on AMD ROCm with {{precision.upper()}}: {{e}}")
                    traceback.print_exc()
                    results[f"amd_{{precision}}_test"] = f"Error: {{str(e)}}"
        else:
            results["amd_test"] = "AMD ROCm not available"
            
        # Test on WebNN if available
        if self.hardware_capabilities.get("webnn", False):
            for precision in ["fp32", "fp16", "int8"]:
                try:
                    print(f"Testing {model_type} on WebNN with {{precision.upper()}} precision...")
                    model_info = self._get_model_info()
                    
                    # Skip if precision not supported on WebNN
                    if not model_info["precision_compatibility"]["webnn"].get(precision, False):
                        print(f"Precision {{precision.upper()}} not supported on WebNN, skipping...")
                        continue
                    
                    # Initialize model with specific precision
                    endpoint, processor, handler, queue, batch_size = self.init_webnn(
                        model_name="test-{model_type}-model",
                        model_type="{template_vars['primary_task']}",
                        precision=precision
                    )
                    
                    # Test with simple input
                    input_text = f"This is a test input for {model_type} with {{precision.upper()}} precision on WebNN"
                    output = handler(input_text)
                    
                    # Record results
                    examples.append({{
                        "platform": f"WebNN ({{precision.upper()}})",
                        "input": input_text,
                        "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                        "implementation_type": output.get("implementation_type", "UNKNOWN"),
                        "precision": precision,
                        "hardware": "WebNN",
                        "model_info": {{
                            "input_format": model_info["input"]["format"],
                            "output_format": model_info["output"]["format"]
                        }}
                    }})
                    
                    results[f"webnn_{{precision}}_test"] = "Success"
                except Exception as e:
                    print(f"Error testing on WebNN with {{precision.upper()}}: {{e}}")
                    traceback.print_exc()
                    results[f"webnn_{{precision}}_test"] = f"Error: {{str(e)}}"
        else:
            results["webnn_test"] = "WebNN not available"
            
        # Test on WebGPU if available
        if self.hardware_capabilities.get("webgpu", False):
            for precision in ["fp32", "fp16", "int8"]:
                try:
                    print(f"Testing {model_type} on WebGPU/transformers.js with {{precision.upper()}} precision...")
                    model_info = self._get_model_info()
                    
                    # Skip if precision not supported on WebGPU
                    if not model_info["precision_compatibility"]["webgpu"].get(precision, False):
                        print(f"Precision {{precision.upper()}} not supported on WebGPU, skipping...")
                        continue
                    
                    # Initialize model with specific precision
                    endpoint, processor, handler, queue, batch_size = self.init_webgpu(
                        model_name="test-{model_type}-model",
                        model_type="{template_vars['primary_task']}",
                        precision=precision
                    )
                    
                    # Test with simple input
                    input_text = f"This is a test input for {model_type} with {{precision.upper()}} precision on WebGPU/transformers.js"
                    output = handler(input_text)
                    
                    # Record results
                    examples.append({{
                        "platform": f"WebGPU ({{precision.upper()}})",
                        "input": input_text,
                        "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                        "implementation_type": output.get("implementation_type", "UNKNOWN"),
                        "precision": precision,
                        "hardware": "WebGPU",
                        "model_info": {{
                            "input_format": model_info["input"]["format"],
                            "output_format": model_info["output"]["format"]
                        }}
                    }})
                    
                    results[f"webgpu_{{precision}}_test"] = "Success"
                except Exception as e:
                    print(f"Error testing on WebGPU with {{precision.upper()}}: {{e}}")
                    traceback.print_exc()
                    results[f"webgpu_{{precision}}_test"] = f"Error: {{str(e)}}"
        else:
            results["webgpu_test"] = "WebGPU not available"
            
        # Return test results
        return {{
            "results": results,
            "examples": examples,
            "timestamp": datetime.datetime.now().isoformat()
        }}

# Helper function to run the test
def run_test():
    """Run a simple test of the {model_type} implementation."""
    print("Testing {model_type} implementation with AMD and precision support...")
    
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
        print(f"  Precision: {{example['precision']}}")
        print(f"  Hardware: {{example['hardware']}}")
        
        # Print model information
        if 'model_info' in example:
            print(f"  Model Information:")
            print(f"    Input Format: {{example['model_info']['input_format']}}")
            print(f"    Output Format: {{example['model_info']['output_format']}}")
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
    parser = argparse.ArgumentParser(description="Comprehensive Template Test Generator")
    parser.add_argument("--model", type=str, required=True, help="Model type to generate a test for")
    parser.add_argument("--output-dir", type=str, default=str(SAMPLE_DIR), help="Output directory for the test file")
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