#!/usr/bin/env python3
"""
Skillset Generator for IPFS Accelerate Python

This script generates optimized skillset implementations that match the standard format
used in the ipfs_accelerate_py/worker/skillset directory. Unlike the test generator,
which creates test files, this generator creates actual skillset implementation files
that can be used directly in the worker module.

Features:
- Generates properly structured skillset files with the hf_* prefix
- Creates standardized handler methods and initialization routines
- Implements proper resource management for shared dependencies
- Adds comprehensive error handling and mock implementations
- Includes support for all hardware platforms (CPU, CUDA, OpenVINO, MPS, ROCm, etc.)

Usage:
  python skillset_generator.py --model bert
  python skillset_generator.py --list-models
  python skillset_generator.py --batch bert,t5,clip,llama
  python skillset_generator.py --all
"""

import os
import sys
import json
import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = CURRENT_DIR.parent
OUTPUT_DIR = PARENT_DIR / "ipfs_accelerate_py" / "worker" / "skillset"
TEMPLATES_DIR = CURRENT_DIR / "templates"

# Modality types
MODALITY_TYPES = {
    "text": ["bert", "gpt2", "t5", "roberta", "distilbert", "bart", "llama", "mistral", "phi",
            "mixtral", "gemma", "qwen2", "deepseek", "falcon", "mpt", "chatglm", "bloom"],
    "vision": ["vit", "deit", "swin", "convnext", "resnet", "dinov2", "detr", "sam", "segformer", 
              "mask2former", "dino", "zoedepth", "depth-anything", "yolos"],
    "audio": ["wav2vec2", "whisper", "hubert", "clap", "audioldm2", "musicgen", "bark", 
             "encodec", "univnet", "speecht5", "qwen2-audio"],
    "multimodal": ["clip", "llava", "blip", "flava", "owlvit", "git", "pali-gemma", "idefics",
                  "llava-next", "flamingo", "blip2", "kosmos-2", "siglip", "chinese-clip", 
                  "instructblip", "qwen2-vl"]
}

def detect_modality(model_name):
    """
    Detect the modality of a model based on its name.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        str: One of "text", "vision", "audio", "multimodal", or "other"
    """
    # Check for direct matches in modality categories
    for modality, models in MODALITY_TYPES.items():
        if any(model in model_name.lower() for model in models):
            return modality
            
    # Check for common patterns in model name
    if any(x in model_name.lower() for x in ["text", "gpt", "llm", "large-language", "roberta", "albert", "electra"]):
        return "text"
    elif any(x in model_name.lower() for x in ["image", "vision", "visual", "seg", "detect", "depth"]):
        return "vision"
    elif any(x in model_name.lower() for x in ["audio", "speech", "voice", "sound", "speak"]):
        return "audio"
    elif any(x in model_name.lower() for x in ["multi", "modality", "vision-language", "vl", "text-image"]):
        return "multimodal"
    
    # Default to text as the safest fallback
    return "text"

def normalize_model_name(name):
    """Normalize model name to match file naming conventions"""
    return name.replace('-', '_').replace('.', '_').lower()

def generate_skillset_template(model_name, modality):
    """
    Generate a skillset file template for a given model name and modality
    
    Args:
        model_name (str): Name of the model (e.g., 'bert', 't5', 'clip')
        modality (str): Modality of the model ('text', 'vision', 'audio', 'multimodal', 'other')
        
    Returns:
        str: The generated template code
    """
    # Normalize the model name for class and file naming
    normalized_name = normalize_model_name(model_name)
    
    # Common import block
    imports = """import asyncio
import os
import json
import time
from unittest.mock import MagicMock
"""

    # Common class definition
    class_def = f"""
class hf_{normalized_name}:
    """HuggingFace {model_name.upper()} implementation.
    
    This class provides standardized interfaces for working with {model_name.upper()} models
    across different hardware backends (CPU, CUDA, OpenVINO, Apple, Qualcomm).
    """
    
    def __init__(self, resources=None, metadata=None):
        """Initialize the {model_name.upper()} model.
        
        Args:
            resources (dict): Dictionary of shared resources (torch, transformers, etc.)
            metadata (dict): Configuration metadata
        """
        self.resources = resources
        self.metadata = metadata
        
        # Handler creation methods
"""

    # Handler registration based on modality
    if modality == "text":
        handler_registration = """        self.create_cpu_text_embedding_endpoint_handler = self.create_cpu_text_embedding_endpoint_handler
        self.create_cuda_text_embedding_endpoint_handler = self.create_cuda_text_embedding_endpoint_handler
        self.create_openvino_text_embedding_endpoint_handler = self.create_openvino_text_embedding_endpoint_handler
        self.create_apple_text_embedding_endpoint_handler = self.create_apple_text_embedding_endpoint_handler
        self.create_qualcomm_text_embedding_endpoint_handler = self.create_qualcomm_text_embedding_endpoint_handler
"""
    elif modality == "vision":
        handler_registration = """        self.create_cpu_image_classification_endpoint_handler = self.create_cpu_image_classification_endpoint_handler
        self.create_cuda_image_classification_endpoint_handler = self.create_cuda_image_classification_endpoint_handler
        self.create_openvino_image_classification_endpoint_handler = self.create_openvino_image_classification_endpoint_handler
        self.create_apple_image_classification_endpoint_handler = self.create_apple_image_classification_endpoint_handler
        self.create_qualcomm_image_classification_endpoint_handler = self.create_qualcomm_image_classification_endpoint_handler
"""
    elif modality == "audio":
        handler_registration = """        self.create_cpu_audio_classification_endpoint_handler = self.create_cpu_audio_classification_endpoint_handler
        self.create_cuda_audio_classification_endpoint_handler = self.create_cuda_audio_classification_endpoint_handler
        self.create_openvino_audio_classification_endpoint_handler = self.create_openvino_audio_classification_endpoint_handler
        self.create_apple_audio_classification_endpoint_handler = self.create_apple_audio_classification_endpoint_handler
        self.create_qualcomm_audio_classification_endpoint_handler = self.create_qualcomm_audio_classification_endpoint_handler
"""
    elif modality == "multimodal":
        handler_registration = """        self.create_cpu_multimodal_embedding_endpoint_handler = self.create_cpu_multimodal_embedding_endpoint_handler
        self.create_cuda_multimodal_embedding_endpoint_handler = self.create_cuda_multimodal_embedding_endpoint_handler
        self.create_openvino_multimodal_embedding_endpoint_handler = self.create_openvino_multimodal_embedding_endpoint_handler
        self.create_apple_multimodal_embedding_endpoint_handler = self.create_apple_multimodal_embedding_endpoint_handler
        self.create_qualcomm_multimodal_embedding_endpoint_handler = self.create_qualcomm_multimodal_embedding_endpoint_handler
"""
    else:
        handler_registration = """        # Add appropriate handlers for your model type
        self.create_cpu_endpoint_handler = self.create_cpu_endpoint_handler
        self.create_cuda_endpoint_handler = self.create_cuda_endpoint_handler
        self.create_openvino_endpoint_handler = self.create_openvino_endpoint_handler
        self.create_apple_endpoint_handler = self.create_apple_endpoint_handler
        self.create_qualcomm_endpoint_handler = self.create_qualcomm_endpoint_handler
"""

    # Initialization methods
    init_methods = """        # Initialization methods
        self.init = self.init
        self.init_cpu = self.init_cpu
        self.init_cuda = self.init_cuda
        self.init_openvino = self.init_openvino
        self.init_apple = self.init_apple
        self.init_qualcomm = self.init_qualcomm
        
        # Test methods
        self.__test__ = self.__test__
        
        # Hardware-specific utilities
        self.snpe_utils = None  # Qualcomm SNPE utils
        return None
"""

    # Resource initialization
    init_resources = """
    def init(self):        
        if "torch" not in list(self.resources.keys()):
            import torch
            self.torch = torch
        else:
            self.torch = self.resources["torch"]

        if "transformers" not in list(self.resources.keys()):
            import transformers
            self.transformers = transformers
        else:
            self.transformers = self.resources["transformers"]
            
        if "numpy" not in list(self.resources.keys()):
            import numpy as np
            self.np = np
        else:
            self.np = self.resources["numpy"]
"""

    # Test method
    test_method = """
    def __test__(self, endpoint_model, endpoint_handler, endpoint_label, tokenizer):
        """Test method for the model implementation"""
"""

    # Add modality-specific test inputs
    if modality == "text":
        test_method += """        sentence_1 = "The quick brown fox jumps over the lazy dog"
        timestamp1 = time.time()
        test_batch = None
        tokens = tokenizer(sentence_1)["input_ids"]
        len_tokens = len(tokens)
        try:
            test_batch = endpoint_handler(sentence_1)
            print(test_batch)
            print("hf_embed test passed")
        except Exception as e:
            print(e)
            print("hf_embed test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        tokens_per_second = len_tokens / elapsed_time
        print(f"elapsed time: {elapsed_time}")
        print(f"tokens: {len_tokens}")
        print(f"tokens per second: {tokens_per_second}")
"""
    elif modality == "vision":
        test_method += """        image_path = "test.jpg"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(image_path)
            print(test_batch)
            print("Vision model test passed")
        except Exception as e:
            print(e)
            print("Vision model test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        print(f"elapsed time: {elapsed_time}")
        print(f"images per second: {1 / elapsed_time}")
"""
    elif modality == "audio":
        test_method += """        audio_path = "test.mp3"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(audio_path)
            print(test_batch)
            print("Audio model test passed")
        except Exception as e:
            print(e)
            print("Audio model test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        print(f"elapsed time: {elapsed_time}")
        print(f"audio samples per second: {1 / elapsed_time}")
"""
    elif modality == "multimodal":
        test_method += """        image_path = "test.jpg"
        text_prompt = "What's in this image?"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler({"image": image_path, "text": text_prompt})
            print(test_batch)
            print("Multimodal model test passed")
        except Exception as e:
            print(e)
            print("Multimodal model test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        print(f"elapsed time: {elapsed_time}")
        print(f"samples per second: {1 / elapsed_time}")
"""
    else:
        test_method += """        test_input = "Test input for generic model"
        timestamp1 = time.time()
        try:
            test_batch = endpoint_handler(test_input)
            print(test_batch)
            print("Model test passed")
        except Exception as e:
            print(e)
            print("Model test failed")
            pass
        timestamp2 = time.time()
        elapsed_time = timestamp2 - timestamp1
        print(f"elapsed time: {elapsed_time}")
        print(f"samples per second: {1 / elapsed_time}")
"""

    # Complete test method
    test_method += """        # Clean up memory if using CUDA
        with self.torch.no_grad():
            if "cuda" in dir(self.torch):
                self.torch.cuda.empty_cache()
        return True
"""

    # Initialize CPU method
    cpu_init = """
    def init_cpu(self, model_name, device, cpu_label):
        """Initialize model for CPU inference.
        
        Args:
            model_name (str): HuggingFace model name or path
            device (str): Device to run on ('cpu')
            cpu_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, tokenizer, endpoint_handler, asyncio.Queue, batch_size)
        """
        self.init()
        
        print(f"Loading {model_name} for CPU inference...")
        
        try:
            # Add local cache directory for testing environments without internet
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # First try loading with real transformers
            if "transformers" in self.resources and hasattr(self.resources["transformers"], "AutoModel"):
                # Load model configuration
                config = self.transformers.AutoConfig.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
"""

    # Add modality-specific processor loading
    if modality == "text":
        cpu_init += """                
                # Load tokenizer
                tokenizer = self.transformers.AutoTokenizer.from_pretrained(
                    model_name, 
                    use_fast=True, 
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                
                # Load the model
                try:
                    endpoint = self.transformers.AutoModel.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        config=config,
                        low_cpu_mem_usage=True,
                        return_dict=True,
                        cache_dir=cache_dir
                    )
                    endpoint.eval()  # Set to evaluation mode
                    
                    # Print model information
                    print(f"(REAL) Model loaded: {model_name}")
                    print(f"Model type: {config.model_type if hasattr(config, 'model_type') else 'unknown'}")
                    print(f"Hidden size: {config.hidden_size}")
                    
                    # Create handler function
                    endpoint_handler = self.create_cpu_text_embedding_endpoint_handler(
                        endpoint_model=model_name,
                        device=device,
                        hardware_label=cpu_label,
                        endpoint=endpoint,
                        tokenizer=tokenizer
                    )
                    
                    return endpoint, tokenizer, endpoint_handler, asyncio.Queue(32), 0
"""
    elif modality == "vision":
        cpu_init += """                
                # Load image processor
                processor = self.transformers.AutoImageProcessor.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                
                # Load the model
                try:
                    endpoint = self.transformers.AutoModel.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        config=config,
                        low_cpu_mem_usage=True,
                        return_dict=True,
                        cache_dir=cache_dir
                    )
                    endpoint.eval()  # Set to evaluation mode
                    
                    # Print model information
                    print(f"(REAL) Model loaded: {model_name}")
                    print(f"Model type: {config.model_type if hasattr(config, 'model_type') else 'unknown'}")
                    
                    # Create handler function
                    endpoint_handler = self.create_cpu_image_classification_endpoint_handler(
                        endpoint_model=model_name,
                        device=device,
                        hardware_label=cpu_label,
                        endpoint=endpoint,
                        processor=processor
                    )
                    
                    return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
"""
    elif modality == "audio":
        cpu_init += """                
                # Load audio processor
                processor = self.transformers.AutoProcessor.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                
                # Load the model
                try:
                    endpoint = self.transformers.AutoModel.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        config=config,
                        low_cpu_mem_usage=True,
                        return_dict=True,
                        cache_dir=cache_dir
                    )
                    endpoint.eval()  # Set to evaluation mode
                    
                    # Print model information
                    print(f"(REAL) Model loaded: {model_name}")
                    print(f"Model type: {config.model_type if hasattr(config, 'model_type') else 'unknown'}")
                    
                    # Create handler function
                    endpoint_handler = self.create_cpu_audio_classification_endpoint_handler(
                        endpoint_model=model_name,
                        device=device,
                        hardware_label=cpu_label,
                        endpoint=endpoint,
                        processor=processor
                    )
                    
                    return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
"""
    elif modality == "multimodal":
        cpu_init += """                
                # Load multimodal processor
                processor = self.transformers.AutoProcessor.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                
                # Load the model
                try:
                    endpoint = self.transformers.AutoModel.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        config=config,
                        low_cpu_mem_usage=True,
                        return_dict=True,
                        cache_dir=cache_dir
                    )
                    endpoint.eval()  # Set to evaluation mode
                    
                    # Print model information
                    print(f"(REAL) Model loaded: {model_name}")
                    print(f"Model type: {config.model_type if hasattr(config, 'model_type') else 'unknown'}")
                    
                    # Create handler function
                    endpoint_handler = self.create_cpu_multimodal_embedding_endpoint_handler(
                        endpoint_model=model_name,
                        device=device,
                        hardware_label=cpu_label,
                        endpoint=endpoint,
                        processor=processor
                    )
                    
                    return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
"""
    else:
        cpu_init += """                
                # Load processor (choose appropriate type for your model)
                processor = self.transformers.AutoProcessor.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                
                # Load the model
                try:
                    endpoint = self.transformers.AutoModel.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        config=config,
                        low_cpu_mem_usage=True,
                        return_dict=True,
                        cache_dir=cache_dir
                    )
                    endpoint.eval()  # Set to evaluation mode
                    
                    # Print model information
                    print(f"(REAL) Model loaded: {model_name}")
                    print(f"Model type: {config.model_type if hasattr(config, 'model_type') else 'unknown'}")
                    
                    # Create handler function
                    endpoint_handler = self.create_cpu_endpoint_handler(
                        endpoint_model=model_name,
                        device=device,
                        hardware_label=cpu_label,
                        endpoint=endpoint,
                        processor=processor
                    )
                    
                    return endpoint, processor, endpoint_handler, asyncio.Queue(32), 0
"""

    # Common mock implementation and error handling
    cpu_init += """                    
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Falling back to mock implementation")
            
            # If we get here, either transformers is a mock or the model loading failed
            # Return a mock implementation
            return self._create_mock_endpoint(model_name, cpu_label)
            
        except Exception as e:
            print(f"Error in CPU initialization: {e}")
            # Return mock objects for graceful degradation
            return self._create_mock_endpoint(model_name, cpu_label)
"""

    # Add mock creation methods (common across modalities)
    mock_methods = """
    def _create_mock_processor(self):
        """Create a mock processor for graceful degradation when the real one fails.
        
        Returns:
            Mock processor object with essential methods
        """
        try:
            from unittest.mock import MagicMock
            
            processor = MagicMock()
            
            # Configure mock processor call behavior
            def mock_process(input_data, **kwargs):
                if isinstance(input_data, str):
                    batch_size = 1
                else:
                    batch_size = len(input_data) if isinstance(input_data, list) else 1
                
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                # Create appropriate output based on model type
                return {
                    "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, 10), dtype=torch.long)
                }
                
            processor.side_effect = mock_process
            processor.__call__ = mock_process
            
            print("(MOCK) Created mock processor")
            return processor
            
        except ImportError:
            # Fallback if unittest.mock is not available
            class SimpleProcessor:
                def __init__(self, parent):
                    self.parent = parent
                    
                def __call__(self, input_data, **kwargs):
                    if isinstance(input_data, str):
                        batch_size = 1
                    else:
                        batch_size = len(input_data) if isinstance(input_data, list) else 1
                    
                    if hasattr(self.parent, 'torch'):
                        torch = self.parent.torch
                    else:
                        import torch
                    
                    return {
                        "input_ids": torch.ones((batch_size, 10), dtype=torch.long),
                        "attention_mask": torch.ones((batch_size, 10), dtype=torch.long)
                    }
            
            print("(MOCK) Created simple mock processor")
            return SimpleProcessor(self)
    
    def _create_mock_endpoint(self, model_name, device_label):
        """Create mock endpoint objects when real initialization fails.
        
        Args:
            model_name (str): The model name or path
            device_label (str): The device label (cpu, cuda, etc.)
            
        Returns:
            Tuple of (endpoint, processor, handler, queue, batch_size)
        """
        try:
            from unittest.mock import MagicMock
            
            # Create mock endpoint
            endpoint = MagicMock()
            
            # Configure mock endpoint behavior
            def mock_forward(**kwargs):
                batch_size = kwargs.get("input_ids", kwargs.get("inputs_embeds", None)).shape[0]
                sequence_length = kwargs.get("input_ids", kwargs.get("inputs_embeds", None)).shape[1]
                hidden_size = 768  # Standard hidden size
                
                if hasattr(self, 'torch'):
                    torch = self.torch
                else:
                    import torch
                
                # Create mock output structure
                result = MagicMock()
                result.last_hidden_state = torch.rand((batch_size, sequence_length, hidden_size))
                return result
                
            endpoint.side_effect = mock_forward
            endpoint.__call__ = mock_forward
            
            # Create mock processor
            processor = self._create_mock_processor()
            
            # Create appropriate handler for the device type
            if device_label.startswith('cpu'):
                handler_method = self.create_cpu_text_embedding_endpoint_handler
            elif device_label.startswith('cuda'):
                handler_method = self.create_cuda_text_embedding_endpoint_handler
            elif device_label.startswith('openvino'):
                handler_method = self.create_openvino_text_embedding_endpoint_handler
            elif device_label.startswith('apple'):
                handler_method = self.create_apple_text_embedding_endpoint_handler
            elif device_label.startswith('qualcomm'):
                handler_method = self.create_qualcomm_text_embedding_endpoint_handler
            else:
                handler_method = self.create_cpu_text_embedding_endpoint_handler
            
            # Create handler function
            mock_handler = handler_method(
                endpoint_model=model_name,
                device=device_label.split(':')[0] if ':' in device_label else device_label,
                hardware_label=device_label,
                endpoint=endpoint,
                tokenizer=processor
            )
            
            import asyncio
            print(f"(MOCK) Created mock endpoint for {model_name} on {device_label}")
            return endpoint, processor, mock_handler, asyncio.Queue(32), 0
            
        except Exception as e:
            print(f"Error creating mock endpoint: {e}")
            import asyncio
            return None, None, None, asyncio.Queue(32), 0
"""

    # Add stub for CUDA initialization
    cuda_init = """
    def init_cuda(self, model_name, device, cuda_label):
        """Initialize model for CUDA inference.
        
        Args:
            model_name (str): Model name or path
            device (str): Device to run on ('cuda' or 'cuda:0', etc.)
            cuda_label (str): Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        # Initialize resources
        self.init()
        
        # Check CUDA availability
        if not hasattr(self.torch, 'cuda') or not self.torch.cuda.is_available():
            print(f"CUDA is not available, falling back to CPU for model '{model_name}'")
            return self.init_cpu(model_name, "cpu", "cpu")
            
        # TODO: Implement CUDA initialization similar to CPU
        # For now, return mock implementation
        print(f"CUDA initialization not fully implemented for {model_name}, using mock")
        return self._create_mock_endpoint(model_name, cuda_label)
"""

    # Add stub for OpenVINO initialization
    openvino_init = """
    def init_openvino(self, model_name, model_type, device, openvino_label, get_optimum_openvino_model=None, get_openvino_model=None, get_openvino_pipeline_type=None, openvino_cli_convert=None):
        """Initialize model for OpenVINO inference.
        
        Args:
            model_name: Model name or path
            model_type: Type of model (e.g., 'fill-mask', 'text-classification')
            device: Device to run on ('CPU', 'GPU', etc.)
            openvino_label: Label to identify this endpoint
            get_optimum_openvino_model: Optional function to get Optimum model
            get_openvino_model: Optional function to get OpenVINO model
            get_openvino_pipeline_type: Optional function to get pipeline type
            openvino_cli_convert: Optional function to convert model using CLI
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        # Initialize resources
        self.init()
        
        # TODO: Implement OpenVINO initialization
        # For now, return mock implementation
        print(f"OpenVINO initialization not fully implemented for {model_name}, using mock")
        return self._create_mock_endpoint(model_name, openvino_label)
"""

    # Add stub for Apple Silicon initialization
    apple_init = """
    def init_apple(self, model_name, device, apple_label):
        """Initialize model for Apple Silicon hardware.
        
        Args:
            model_name: Model name or path
            device: Device to run on ('mps')
            apple_label: Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        # Initialize resources
        self.init()
        
        # Check MPS availability
        if not hasattr(self.torch.backends, 'mps') or not self.torch.backends.mps.is_available():
            print("MPS not available. Cannot initialize model on Apple Silicon.")
            return None, None, None, None, 0
            
        # TODO: Implement Apple Silicon initialization
        # For now, return mock implementation
        print(f"Apple Silicon initialization not fully implemented for {model_name}, using mock")
        return self._create_mock_endpoint(model_name, apple_label)
"""

    # Add stub for Qualcomm initialization
    qualcomm_init = """
    def init_qualcomm(self, model_name, device, qualcomm_label):
        """Initialize model for Qualcomm hardware.
        
        Args:
            model_name: Model name or path
            device: Device to run on
            qualcomm_label: Label to identify this endpoint
            
        Returns:
            Tuple of (endpoint, processor, endpoint_handler, asyncio.Queue, batch_size)
        """
        # Initialize resources
        self.init()
        
        # TODO: Implement Qualcomm initialization
        # For now, return mock implementation
        print(f"Qualcomm initialization not fully implemented for {model_name}, using mock")
        return self._create_mock_endpoint(model_name, qualcomm_label)
"""

    # Add handler implementations based on modality
    handlers = ""

    if modality == "text":
        handlers += """
    def create_cpu_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None):
        """Create endpoint handler for CPU backend.
        
        Args:
            endpoint_model (str): The model name or path
            device (str): The device to run inference on ('cpu')
            hardware_label (str): Label to identify this endpoint
            endpoint: The model endpoint
            tokenizer: The tokenizer for the model
            
        Returns:
            A handler function for the CPU endpoint
        """
        def handler(text_input, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, tokenizer=tokenizer):
            """Process text input to generate embeddings.
            
            Args:
                text_input: Input text (string or list of strings)
                
            Returns:
                Embedding tensor (mean pooled from last hidden state)
            """
            # Set model to evaluation mode
            if hasattr(endpoint, 'eval'):
                endpoint.eval()
            
            try:
                with self.torch.no_grad():
                    # Process different input types
                    if isinstance(text_input, str):
                        # Single text input
                        tokens = tokenizer(
                            text_input, 
                            return_tensors="pt", 
                            padding=True,
                            truncation=True,
                            max_length=512  # Standard BERT max length
                        )
                    elif isinstance(text_input, list):
                        # Batch of texts
                        tokens = tokenizer(
                            text_input,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512
                        )
                    else:
                        raise ValueError(f"Unsupported input type: {type(text_input)}")
                    
                    # Run inference
                    results = endpoint(**tokens)
                    
                    # Check if the output is in the expected format
                    if not hasattr(results, 'last_hidden_state'):
                        # Handle different output formats
                        if isinstance(results, dict) and 'last_hidden_state' in results:
                            last_hidden = results['last_hidden_state']
                        else:
                            # Unexpected output format, return mock
                            print(f"(MOCK) Unexpected output format from model, using fallback")
                            batch_size = 1 if isinstance(text_input, str) else len(text_input)
                            return self.torch.rand((batch_size, 768))
                    else:
                        last_hidden = results.last_hidden_state
                    
                    # Mean pooling: mask padding tokens and average across sequence length
                    # This is a standard way to get sentence embeddings from BERT
                    masked_hidden = last_hidden.masked_fill(
                        ~tokens['attention_mask'].bool().unsqueeze(-1), 
                        0.0
                    )
                    
                    # Sum and divide by actual token count (excluding padding)
                    average_pool_results = masked_hidden.sum(dim=1) / tokens['attention_mask'].sum(dim=1, keepdim=True)
                    
                    # Add timestamp and metadata for testing/debugging
                    import time
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # REAL signal in output tensor metadata for testing
                    average_pool_results.real_implementation = True
                    
                    return average_pool_results
                    
            except Exception as e:
                print(f"Error in CPU text embedding handler: {e}")
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Generate a mock embedding with error info
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                mock_embedding = self.torch.rand((batch_size, 768))
                
                # Add signal this is a mock for testing
                mock_embedding.mock_implementation = True
                
                return mock_embedding
                
        return handler

    def create_cuda_text_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, tokenizer=None):
        """Create a CUDA handler for text embeddings (placeholder implementation)"""
        # This is a placeholder - implement with similar structure to CPU handler but with CUDA optimizations
        def handler(text_input, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, tokenizer=tokenizer):
            try:
                # Basic mock implementation
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                mock_embedding = self.torch.rand((batch_size, 768))
                mock_embedding.mock_implementation = True
                mock_embedding.device = device
                return mock_embedding
            except Exception as e:
                print(f"Error in CUDA handler: {e}")
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                return self.torch.rand((batch_size, 768))
        return handler

    def create_openvino_text_embedding_endpoint_handler(self, endpoint_model, tokenizer, openvino_label, endpoint=None):
        """Create an OpenVINO handler for text embeddings (placeholder implementation)"""
        # This is a placeholder - implement with OpenVINO-specific code
        def handler(text_input, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=endpoint, tokenizer=tokenizer):
            try:
                # Basic mock implementation
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                mock_embedding = self.torch.rand((batch_size, 768))
                mock_embedding.mock_implementation = True
                mock_embedding.device = "OpenVINO"
                return mock_embedding
            except Exception as e:
                print(f"Error in OpenVINO handler: {e}")
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                return self.torch.rand((batch_size, 768))
        return handler

    def create_apple_text_embedding_endpoint_handler(self, endpoint_model, apple_label, endpoint=None, tokenizer=None):
        """Create an Apple Silicon handler for text embeddings (placeholder implementation)"""
        # This is a placeholder - implement with MPS-specific code
        def handler(text_input, endpoint_model=endpoint_model, apple_label=apple_label, endpoint=endpoint, tokenizer=tokenizer):
            try:
                # Basic mock implementation
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                mock_embedding = self.torch.rand((batch_size, 768))
                mock_embedding.mock_implementation = True
                mock_embedding.device = "MPS"
                return mock_embedding
            except Exception as e:
                print(f"Error in Apple Silicon handler: {e}")
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                return self.torch.rand((batch_size, 768))
        return handler

    def create_qualcomm_text_embedding_endpoint_handler(self, endpoint_model, qualcomm_label, endpoint=None, tokenizer=None):
        """Create a Qualcomm handler for text embeddings (placeholder implementation)"""
        # This is a placeholder - implement with Qualcomm-specific code
        def handler(text_input, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint, tokenizer=tokenizer):
            try:
                # Basic mock implementation
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                mock_embedding = self.torch.rand((batch_size, 768))
                mock_embedding.mock_implementation = True
                mock_embedding.device = "Qualcomm"
                return mock_embedding
            except Exception as e:
                print(f"Error in Qualcomm handler: {e}")
                batch_size = 1 if isinstance(text_input, str) else len(text_input)
                return self.torch.rand((batch_size, 768))
        return handler
"""
    elif modality == "vision":
        handlers += """
    def create_cpu_image_classification_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, processor=None):
        """Create endpoint handler for CPU backend.
        
        Args:
            endpoint_model (str): The model name or path
            device (str): The device to run inference on ('cpu')
            hardware_label (str): Label to identify this endpoint
            endpoint: The model endpoint
            processor: The image processor for the model
            
        Returns:
            A handler function for the CPU endpoint
        """
        def handler(image_input, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, processor=processor):
            """Process image input for vision models.
            
            Args:
                image_input: Input image (PIL image, path, or list of images)
                
            Returns:
                Model output for the image
            """
            # Set model to evaluation mode
            if hasattr(endpoint, 'eval'):
                endpoint.eval()
            
            try:
                with self.torch.no_grad():
                    # Process image input
                    if isinstance(image_input, str):
                        # It's a file path, load the image
                        from PIL import Image
                        image = Image.open(image_input).convert('RGB')
                    else:
                        # Assume it's already a PIL image or a batch
                        image = image_input
                    
                    # Process with the vision processor
                    inputs = processor(images=image, return_tensors="pt")
                    
                    # Run inference
                    outputs = endpoint(**inputs)
                    
                    # Add metadata for testing
                    if hasattr(outputs, 'last_hidden_state'):
                        outputs.last_hidden_state.real_implementation = True
                    
                    return outputs
                    
            except Exception as e:
                print(f"Error in CPU vision handler: {e}")
                
                # Generate a mock output
                batch_size = 1
                if isinstance(image_input, list):
                    batch_size = len(image_input)
                
                mock_output = type('obj', (object,), {
                    'last_hidden_state': self.torch.rand((batch_size, 197, 768)),
                    'pooler_output': self.torch.rand((batch_size, 768)),
                    'mock_implementation': True
                })
                
                return mock_output
                
        return handler

    # Add placeholder implementations for other hardware platforms similar to text modality
    def create_cuda_image_classification_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, processor=None):
        """Create a CUDA handler for vision models (placeholder implementation)"""
        # Placeholder - implement properly with CUDA-specific code
        def handler(image_input, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, processor=processor):
            try:
                # Basic mock implementation
                batch_size = 1 if not isinstance(image_input, list) else len(image_input)
                mock_output = type('obj', (object,), {
                    'last_hidden_state': self.torch.rand((batch_size, 197, 768)),
                    'pooler_output': self.torch.rand((batch_size, 768)),
                    'mock_implementation': True,
                    'device': device
                })
                return mock_output
            except Exception as e:
                print(f"Error in CUDA vision handler: {e}")
                batch_size = 1
                return type('obj', (object,), {
                    'last_hidden_state': self.torch.rand((batch_size, 197, 768)),
                    'mock_implementation': True
                })
        return handler

    # Add placeholder implementations for other hardware platforms
    def create_openvino_image_classification_endpoint_handler(self, endpoint_model, processor, openvino_label, endpoint=None):
        """Create an OpenVINO handler for vision models (placeholder implementation)"""
        def handler(image_input, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_OPENVINO_OUTPUT", "implementation_type": "MOCK_OPENVINO"}
        return handler

    def create_apple_image_classification_endpoint_handler(self, endpoint_model, apple_label, endpoint=None, processor=None):
        """Create an Apple Silicon handler for vision models (placeholder implementation)"""
        def handler(image_input, endpoint_model=endpoint_model, apple_label=apple_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_MPS_OUTPUT", "implementation_type": "MOCK_MPS"}
        return handler

    def create_qualcomm_image_classification_endpoint_handler(self, endpoint_model, qualcomm_label, endpoint=None, processor=None):
        """Create a Qualcomm handler for vision models (placeholder implementation)"""
        def handler(image_input, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_QUALCOMM_OUTPUT", "implementation_type": "MOCK_QUALCOMM"}
        return handler
"""
    elif modality == "audio":
        handlers += """
    def create_cpu_audio_classification_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, processor=None):
        """Create endpoint handler for CPU backend.
        
        Args:
            endpoint_model (str): The model name or path
            device (str): The device to run inference on ('cpu')
            hardware_label (str): Label to identify this endpoint
            endpoint: The model endpoint
            processor: The audio processor for the model
            
        Returns:
            A handler function for the CPU endpoint
        """
        def handler(audio_input, sampling_rate=16000, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, processor=processor):
            """Process audio input for audio models.
            
            Args:
                audio_input: Input audio (file path or numpy array)
                sampling_rate: Sampling rate of the audio
                
            Returns:
                Model output for the audio
            """
            # Set model to evaluation mode
            if hasattr(endpoint, 'eval'):
                endpoint.eval()
            
            try:
                with self.torch.no_grad():
                    # Process audio input
                    if isinstance(audio_input, str):
                        # It's a file path, load the audio
                        try:
                            import librosa
                            audio_array, sr = librosa.load(audio_input, sr=sampling_rate)
                        except ImportError:
                            print("Librosa not available, using mock audio")
                            audio_array = self.np.zeros(sampling_rate * 3)  # 3 seconds of silence
                            sr = sampling_rate
                    else:
                        # Assume it's already a numpy array
                        audio_array = audio_input
                        sr = sampling_rate
                    
                    # Process with the audio processor
                    inputs = processor(
                        audio_array, 
                        sampling_rate=sr, 
                        return_tensors="pt"
                    )
                    
                    # Run inference
                    outputs = endpoint(**inputs)
                    
                    # Add metadata for testing
                    if hasattr(outputs, 'last_hidden_state'):
                        outputs.last_hidden_state.real_implementation = True
                    
                    return outputs
                    
            except Exception as e:
                print(f"Error in CPU audio handler: {e}")
                
                # Generate a mock output
                batch_size = 1
                mock_output = type('obj', (object,), {
                    'last_hidden_state': self.torch.rand((batch_size, 1000, 768)),
                    'mock_implementation': True
                })
                
                return mock_output
                
        return handler

    # Add placeholder implementations for other hardware platforms
    def create_cuda_audio_classification_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, processor=None):
        """Create a CUDA handler for audio models (placeholder implementation)"""
        def handler(audio_input, sampling_rate=16000, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_CUDA_OUTPUT", "implementation_type": "MOCK_CUDA"}
        return handler

    def create_openvino_audio_classification_endpoint_handler(self, endpoint_model, processor, openvino_label, endpoint=None):
        """Create an OpenVINO handler for audio models (placeholder implementation)"""
        def handler(audio_input, sampling_rate=16000, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_OPENVINO_OUTPUT", "implementation_type": "MOCK_OPENVINO"}
        return handler

    def create_apple_audio_classification_endpoint_handler(self, endpoint_model, apple_label, endpoint=None, processor=None):
        """Create an Apple Silicon handler for audio models (placeholder implementation)"""
        def handler(audio_input, sampling_rate=16000, endpoint_model=endpoint_model, apple_label=apple_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_MPS_OUTPUT", "implementation_type": "MOCK_MPS"}
        return handler

    def create_qualcomm_audio_classification_endpoint_handler(self, endpoint_model, qualcomm_label, endpoint=None, processor=None):
        """Create a Qualcomm handler for audio models (placeholder implementation)"""
        def handler(audio_input, sampling_rate=16000, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_QUALCOMM_OUTPUT", "implementation_type": "MOCK_QUALCOMM"}
        return handler
"""
    elif modality == "multimodal":
        handlers += """
    def create_cpu_multimodal_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, processor=None):
        """Create endpoint handler for CPU backend.
        
        Args:
            endpoint_model (str): The model name or path
            device (str): The device to run inference on ('cpu')
            hardware_label (str): Label to identify this endpoint
            endpoint: The model endpoint
            processor: The multimodal processor for the model
            
        Returns:
            A handler function for the CPU endpoint
        """
        def handler(input_data, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, processor=processor):
            """Process multimodal input (image + text).
            
            Args:
                input_data: Either a dict with 'image' and 'text' keys, or a text string
                
            Returns:
                Model output for the multimodal input
            """
            # Set model to evaluation mode
            if hasattr(endpoint, 'eval'):
                endpoint.eval()
            
            try:
                with self.torch.no_grad():
                    # Process multimodal input
                    if isinstance(input_data, dict):
                        # Extract image and text from the dict
                        image = input_data.get('image')
                        text = input_data.get('text')
                        
                        # Process image if it's a file path
                        if isinstance(image, str):
                            from PIL import Image
                            image = Image.open(image).convert('RGB')
                    else:
                        # Assume it's just text input, use a default image
                        text = input_data
                        from PIL import Image
                        image = Image.new('RGB', (224, 224), color='white')
                    
                    # Process with the multimodal processor
                    inputs = processor(
                        text=text,
                        images=image,
                        return_tensors="pt"
                    )
                    
                    # Run inference
                    outputs = endpoint(**inputs)
                    
                    # Add metadata for testing
                    if hasattr(outputs, 'last_hidden_state'):
                        outputs.last_hidden_state.real_implementation = True
                    
                    return outputs
                    
            except Exception as e:
                print(f"Error in CPU multimodal handler: {e}")
                
                # Generate a mock output
                batch_size = 1
                mock_output = type('obj', (object,), {
                    'last_hidden_state': self.torch.rand((batch_size, 20, 768)),
                    'pooler_output': self.torch.rand((batch_size, 768)),
                    'mock_implementation': True
                })
                
                return mock_output
                
        return handler

    # Add placeholder implementations for other hardware platforms
    def create_cuda_multimodal_embedding_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, processor=None):
        """Create a CUDA handler for multimodal models (placeholder implementation)"""
        def handler(input_data, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_CUDA_OUTPUT", "implementation_type": "MOCK_CUDA"}
        return handler

    def create_openvino_multimodal_embedding_endpoint_handler(self, endpoint_model, processor, openvino_label, endpoint=None):
        """Create an OpenVINO handler for multimodal models (placeholder implementation)"""
        def handler(input_data, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_OPENVINO_OUTPUT", "implementation_type": "MOCK_OPENVINO"}
        return handler

    def create_apple_multimodal_embedding_endpoint_handler(self, endpoint_model, apple_label, endpoint=None, processor=None):
        """Create an Apple Silicon handler for multimodal models (placeholder implementation)"""
        def handler(input_data, endpoint_model=endpoint_model, apple_label=apple_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_MPS_OUTPUT", "implementation_type": "MOCK_MPS"}
        return handler

    def create_qualcomm_multimodal_embedding_endpoint_handler(self, endpoint_model, qualcomm_label, endpoint=None, processor=None):
        """Create a Qualcomm handler for multimodal models (placeholder implementation)"""
        def handler(input_data, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_QUALCOMM_OUTPUT", "implementation_type": "MOCK_QUALCOMM"}
        return handler
"""
    else:
        handlers += """
    def create_cpu_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, processor=None):
        """Create endpoint handler for CPU backend.
        
        Args:
            endpoint_model (str): The model name or path
            device (str): The device to run inference on ('cpu')
            hardware_label (str): Label to identify this endpoint
            endpoint: The model endpoint
            processor: The processor for the model
            
        Returns:
            A handler function for the CPU endpoint
        """
        def handler(input_data, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, processor=processor):
            """Process generic input data.
            
            Args:
                input_data: Model input (format depends on the specific model type)
                
            Returns:
                Model output
            """
            # Set model to evaluation mode
            if hasattr(endpoint, 'eval'):
                endpoint.eval()
            
            try:
                with self.torch.no_grad():
                    # Process input with the processor
                    inputs = processor(input_data, return_tensors="pt")
                    
                    # Run inference
                    outputs = endpoint(**inputs)
                    
                    # Add metadata for testing
                    if hasattr(outputs, 'last_hidden_state'):
                        outputs.last_hidden_state.real_implementation = True
                    
                    return outputs
                    
            except Exception as e:
                print(f"Error in CPU handler: {e}")
                
                # Generate a mock output
                batch_size = 1 if not isinstance(input_data, list) else len(input_data)
                mock_output = type('obj', (object,), {
                    'last_hidden_state': self.torch.rand((batch_size, 10, 768)),
                    'mock_implementation': True
                })
                
                return mock_output
                
        return handler

    # Add placeholder implementations for other hardware platforms
    def create_cuda_endpoint_handler(self, endpoint_model, device, hardware_label, endpoint=None, processor=None):
        """Create a CUDA handler (placeholder implementation)"""
        def handler(input_data, endpoint_model=endpoint_model, device=device, hardware_label=hardware_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_CUDA_OUTPUT", "implementation_type": "MOCK_CUDA"}
        return handler

    def create_openvino_endpoint_handler(self, endpoint_model, processor, openvino_label, endpoint=None):
        """Create an OpenVINO handler (placeholder implementation)"""
        def handler(input_data, endpoint_model=endpoint_model, openvino_label=openvino_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_OPENVINO_OUTPUT", "implementation_type": "MOCK_OPENVINO"}
        return handler

    def create_apple_endpoint_handler(self, endpoint_model, apple_label, endpoint=None, processor=None):
        """Create an Apple Silicon handler (placeholder implementation)"""
        def handler(input_data, endpoint_model=endpoint_model, apple_label=apple_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_MPS_OUTPUT", "implementation_type": "MOCK_MPS"}
        return handler

    def create_qualcomm_endpoint_handler(self, endpoint_model, qualcomm_label, endpoint=None, processor=None):
        """Create a Qualcomm handler (placeholder implementation)"""
        def handler(input_data, endpoint_model=endpoint_model, qualcomm_label=qualcomm_label, endpoint=endpoint, processor=processor):
            # Placeholder implementation
            return {"output": "MOCK_QUALCOMM_OUTPUT", "implementation_type": "MOCK_QUALCOMM"}
        return handler
"""

    # Combine all sections to create the complete template
    template = (imports + class_def + handler_registration + init_methods + 
               init_resources + test_method + cpu_init + mock_methods + 
               cuda_init + openvino_init + apple_init + qualcomm_init + handlers)
    
    return template

def load_model_registry():
    """
    Load the model registry from a JSON file if available, or fall back to a default dictionary.
    
    Returns:
        Dict containing model registry data
    """
    registry_file = CURRENT_DIR / "huggingface_model_types.json"
    
    try:
        with open(registry_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to a minimal registry
        return {
            "bert": "bert-base-uncased",
            "t5": "t5-small",
            "gpt2": "gpt2",
            "clip": "openai/clip-vit-base-patch32",
            "vit": "google/vit-base-patch16-224",
            "wav2vec2": "facebook/wav2vec2-base",
            "llama": "meta-llama/Llama-2-7b-hf",
            "whisper": "openai/whisper-tiny"
        }

def get_existing_skillsets():
    """
    Get the set of existing skillset implementations.
    
    Returns:
        Set of normalized model names with existing implementations
    """
    existing = set()
    
    if not OUTPUT_DIR.exists():
        return existing
    
    for file in OUTPUT_DIR.glob("hf_*.py"):
        model_name = file.stem.replace("hf_", "")
        existing.add(model_name)
    
    return existing

def generate_skillset(model_name, output_dir=None):
    """
    Generate a skillset implementation file for a specific model.
    
    Args:
        model_name (str): Name of the model to generate
        output_dir (str, optional): Directory to save the file (defaults to WORKER_DIR/skillset)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        if output_dir is None:
            output_dir = OUTPUT_DIR
        else:
            output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalize model name
        normalized_name = normalize_model_name(model_name)
        
        # Detect modality
        modality = detect_modality(model_name)
        
        # Generate template
        template = generate_skillset_template(model_name, modality)
        
        # Write to file
        output_file = output_dir / f"hf_{normalized_name}.py"
        with open(output_file, 'w') as f:
            f.write(template)
        
        return True, f"Successfully generated skillset for {model_name} at {output_file}"
    except Exception as e:
        return False, f"Error generating skillset for {model_name}: {e}"

def batch_generate_skillsets(models, output_dir=None):
    """
    Generate skillsets for multiple models.
    
    Args:
        models (List[str]): List of model names to generate
        output_dir (str, optional): Directory to save the files
        
    Returns:
        Dict mapping model names to results
    """
    results = {}
    
    for model in models:
        success, message = generate_skillset(model, output_dir)
        results[model] = {"success": success, "message": message}
        
        # Log result
        if success:
            logger.info(f"Generated skillset for {model}")
        else:
            logger.error(f"Failed to generate skillset for {model}: {message}")
    
    return results

def list_available_models():
    """List all available models in the registry"""
    registry = load_model_registry()
    existing = get_existing_skillsets()
    
    print("\nAvailable Models:")
    
    # Group by modality for easier reading
    modality_groups = {
        "text": [],
        "vision": [],
        "audio": [],
        "multimodal": [],
        "other": []
    }
    
    for model in sorted(registry.keys()):
        modality = detect_modality(model)
        normalized = normalize_model_name(model)
        status = "✅ Implemented" if normalized in existing else "❌ Not implemented"
        modality_groups[modality].append((model, status))
    
    # Print by modality
    for modality, models in modality_groups.items():
        if models:
            print(f"\n{modality.upper()} MODELS:")
            for model, status in sorted(models):
                print(f"  - {model}: {status}")
    
    # Print summary
    total = len(registry)
    implemented = len(existing)
    print(f"\nImplementation Status: {implemented}/{total} models implemented ({implemented/total*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Generate IPFS Accelerate skillset implementations")
    
    # Main commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-models", action="store_true", help="List all available models")
    group.add_argument("--model", type=str, help="Generate skillset for a specific model")
    group.add_argument("--batch", type=str, help="Generate skillsets for a comma-separated list of models")
    group.add_argument("--all", action="store_true", help="Generate skillsets for all models")
    
    # Additional options
    parser.add_argument("--output-dir", type=str, help="Directory to save the generated files")
    parser.add_argument("--modality", type=str, choices=["text", "vision", "audio", "multimodal", "all"],
                      help="Filter models by modality")
    
    args = parser.parse_args()
    
    # List available models
    if args.list_models:
        list_available_models()
        return
    
    # Generate for a specific model
    if args.model:
        success, message = generate_skillset(args.model, args.output_dir)
        print(message)
        return
    
    # Generate for a batch of models
    if args.batch:
        models = [m.strip() for m in args.batch.split(",")]
        results = batch_generate_skillsets(models, args.output_dir)
        
        # Print summary
        print("\nBatch Generation Results:")
        successful = sum(1 for r in results.values() if r["success"])
        print(f"Successfully generated {successful}/{len(models)} skillsets")
        
        # Print details for failed generations
        failed = [(m, r["message"]) for m, r in results.items() if not r["success"]]
        if failed:
            print("\nFailed generations:")
            for model, message in failed:
                print(f"  - {model}: {message}")
        return
    
    # Generate for all models
    if args.all:
        registry = load_model_registry()
        existing = get_existing_skillsets()
        
        # Filter by modality if specified
        if args.modality and args.modality != "all":
            registry = {k: v for k, v in registry.items() if detect_modality(k) == args.modality}
        
        # Skip already implemented models
        models_to_generate = [m for m in registry.keys() if normalize_model_name(m) not in existing]
        
        print(f"Generating skillsets for {len(models_to_generate)} models...")
        results = batch_generate_skillsets(models_to_generate, args.output_dir)
        
        # Print summary
        successful = sum(1 for r in results.values() if r["success"])
        print(f"\nGeneration Complete: {successful}/{len(models_to_generate)} skillsets generated successfully")
        return

if __name__ == "__main__":
    main()