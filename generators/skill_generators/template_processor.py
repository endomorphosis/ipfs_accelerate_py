#!/usr/bin/env python3
"""
Improved Template Test Generator

This is an enhanced implementation for generating test files that are compatible
with the ipfs_accelerate_py worker/skillset module structure.

This generator adds comprehensive model information including:
- Detailed input/output data types
- Endpoint handler parameters
- Helper functions with argument specifications
- Required dependencies
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
    """Generate a test file for the specified model type with enhanced model registry."""
    
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
        "hardware_backends": ["cpu", "cuda", "openvino", "apple", "qualcomm"],
        # Default tensor types
        "input_tensor_type": "int64",  # Default for token IDs
        "output_tensor_type": "float32",  # Default for embeddings/logits
        "uses_attention_mask": True,
        "uses_position_ids": False,
        "token_sequence_length": 512,  # Default sequence length
        "embedding_dim": 768,  # Default embedding dimension
        "model_precision": "float32",  # Default model precision
        "supports_half_precision": True,  # Default support for FP16
        "batch_processing": True,  # Whether model supports batched inputs
        "helper_functions": ["tokenization", "device_management"]  # Default helpers
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
                        template_vars["uses_attention_mask"] = False
                        template_vars["uses_position_ids"] = False
                        template_vars["helper_functions"].append("image_processing")
                        template_vars["embedding_dim"] = 1024  # Typical for vision models
                        
                    # Text generation models
                    elif primary_task in ["text-generation", "summarization", "translation_XX_to_YY"]:
                        template_vars["input_tensor_type"] = "int64"  # Token IDs
                        template_vars["uses_attention_mask"] = True
                        template_vars["helper_functions"].append("tokenization")
                        template_vars["helper_functions"].append("generation_config")
                        
                    # Multimodal models
                    elif primary_task in ["image-to-text", "visual-question-answering"]:
                        template_vars["input_tensor_type"] = "mixed"  # Both image and text inputs
                        template_vars["uses_attention_mask"] = True
                        template_vars["helper_functions"].append("image_processing")
                        template_vars["helper_functions"].append("tokenization")
                        template_vars["helper_functions"].append("multimodal_inputs")
                        
                    # Audio models
                    elif primary_task in ["automatic-speech-recognition", "audio-classification", "text-to-audio"]:
                        template_vars["input_tensor_type"] = "float32"  # Audio features
                        template_vars["uses_attention_mask"] = True
                        template_vars["helper_functions"].append("audio_processing")
                        template_vars["token_sequence_length"] = 16000  # 1s of audio at 16kHz
                        
                    # Embedding models
                    elif primary_task in ["feature-extraction"]:
                        if "image" in " ".join(template_vars["tasks"]):
                            template_vars["input_tensor_type"] = "float32"  # Image pixels
                            template_vars["helper_functions"].append("image_processing")
                        else:
                            template_vars["input_tensor_type"] = "int64"  # Token IDs
                            template_vars["helper_functions"].append("tokenization")
                        
                    # Ensure unique helper functions
                    template_vars["helper_functions"] = list(set(template_vars["helper_functions"]))
                    
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

# Model type: {model_type}
# Primary task: {template_vars['primary_task']}
# All tasks: {', '.join(template_vars['tasks'])}
# Input tensor type: {template_vars['input_tensor_type']}
# Output tensor type: {template_vars['output_tensor_type']}
# Uses attention mask: {template_vars['uses_attention_mask']}
# Helper functions: {', '.join(template_vars['helper_functions'])}

# Model Registry - Contains metadata about available models for this type
MODEL_REGISTRY = {{
    # Default/small model configuration
    "{model_type}": {{
        "description": "Default {model_type} model",
        "embedding_dim": {template_vars['embedding_dim']},
        "sequence_length": {template_vars['token_sequence_length']},
        "model_precision": "{template_vars['model_precision']}", 
        "supports_half_precision": {template_vars['supports_half_precision']},
        "supports_cpu": True,
        "supports_cuda": True,
        "supports_openvino": True,
        "default_batch_size": 1
    }},
    # Add more model variants as needed
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
        self.create_qualcomm_text_embedding_endpoint_handler = self.create_qualcomm_text_embedding_endpoint_handler
        
        # Initialization methods
        self.init = self.init_cpu  # Default to CPU
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init_qualcomm = self.init_qualcomm
        
        # Test methods
        self.__test__ = self.__test__
        
        # Hardware-specific utilities
        self.snpe_utils = None  # Qualcomm SNPE utils
        
        # Set up model registry and hardware detection
        self.model_registry = MODEL_REGISTRY
        self.hardware_capabilities = self._detect_hardware()
        
        # Set up tensor type information
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
        capabilities = {{
            "cpu": True,
            "cuda": False,
            "cuda_version": None,
            "cuda_devices": 0,
            "mps": False,
            "openvino": False,
            "qualcomm": False
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
            
        return capabilities
    
    def _get_model_tensor_types(self, model_id=None):
        """Get tensor type information for a specific model."""
        model_id = model_id or "{model_type}"
        if model_id in self.model_registry:
            config = self.model_registry[model_id]
            return {{
                "embedding_dim": config.get("embedding_dim", {template_vars['embedding_dim']}),
                "sequence_length": config.get("sequence_length", {template_vars['token_sequence_length']}),
                "precision": config.get("model_precision", "{template_vars['model_precision']}"),
                "supports_half": config.get("supports_half_precision", {template_vars['supports_half_precision']})
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
                
            # Get model tensor type info
            tensor_types = self._get_model_tensor_types()
            
            # Record results
            examples.append({{
                "platform": "CPU",
                "input": input_text,
                "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                "implementation_type": output.get("implementation_type", "UNKNOWN"),
                "tensor_types": tensor_types,
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
                
                # Get model tensor type info
                tensor_types = self._get_model_tensor_types()
                
                # Record results
                examples.append({{
                    "platform": "CUDA",
                    "input": input_text,
                    "output_type": f"container: {{str(type(output))}}, tensor: {{str(type(output.get('tensor', output)))}}",
                    "implementation_type": output.get("implementation_type", "UNKNOWN"),
                    "tensor_types": tensor_types,
                    "hardware": self.hardware_capabilities
                }})
                
                results["cuda_test"] = "Success"
            except Exception as e:
                print(f"Error testing on CUDA: {{e}}")
                traceback.print_exc()
                results["cuda_test"] = f"Error: {{str(e)}}"
        else:
            results["cuda_test"] = "CUDA not available"
        
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
        
        # Print tensor type information
        if 'tensor_types' in example:
            print(f"  Tensor Types:")
            for k, v in example['tensor_types'].items():
                print(f"    {k}: {v}")
                
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
    parser = argparse.ArgumentParser(description="Improved Template Test Generator")
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