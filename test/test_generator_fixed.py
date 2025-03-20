#!/usr/bin/env python3
"""
Hugging Face model test generator script.
This script generates test files for various HF model families.
"""

import os
import sys
import json
import argparse
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Define model families and their configurations
MODEL_FAMILIES = {
    "bert": {
        "class_name": "BertModel",
        "model_class": "BertForMaskedLM",
        "tokenizer_class": "BertTokenizer",
        "task": "fill-mask",
        "model_id": "bert-base-uncased",
        "test_text": "The man worked as a [MASK].",
        "model_type": "text",
        "architecture_type": "encoder_only",
        "inputs": {"text": True},
        "base_model": "bert-base-uncased",
        "dependencies": ["tokenizers", "sentencepiece", "torch"],
    },
    "gpt2": {
        "class_name": "GPT2LMHeadModel",
        "model_class": "GPT2LMHeadModel",
        "tokenizer_class": "GPT2Tokenizer",
        "task": "text-generation",
        "model_id": "gpt2",
        "test_text": "Once upon a time",
        "model_type": "text",
        "architecture_type": "decoder_only",
        "inputs": {"text": True},
        "base_model": "gpt2",
        "dependencies": ["tokenizers", "torch"],
    },
    "t5": {
        "class_name": "T5ForConditionalGeneration",
        "model_class": "T5ForConditionalGeneration",
        "tokenizer_class": "T5Tokenizer",
        "task": "translation_en_to_fr",
        "model_id": "t5-small",
        "test_text": "translate English to French: Hello, how are you?",
        "model_type": "text",
        "architecture_type": "encoder_decoder",
        "inputs": {"text": True},
        "base_model": "t5-small",
        "dependencies": ["sentencepiece", "torch"],
    },
    "vit": {
        "class_name": "ViTForImageClassification",
        "model_class": "ViTForImageClassification",
        "processor_class": "ViTImageProcessor",
        "task": "image-classification",
        "model_id": "google/vit-base-patch16-224",
        "test_image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "model_type": "vision",
        "architecture_type": "encoder_only",
        "inputs": {"image_url": True},
        "base_model": "google/vit-base-patch16-224",
        "dependencies": ["pillow", "requests", "torch"],
    },
}

def generate_imports():
    """Generate common import statements."""
    return """#!/usr/bin/env python3

# Import hardware detection capabilities if available
try:
    from generators.hardware.hardware_detection import (
        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
        detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback
    """

def generate_docstring(family_info):
    """Generate file docstring."""
    class_name = family_info.get("class_name", "AutoModel")
    return f'''"""
Class-based test file for all {family_info["name"]}-family models.
This file provides a unified testing interface for:
    - {class_name}
"""'''

def generate_basic_imports():
    """Generate basic import statements."""
    return """
import os
import sys
import json
import time
import datetime
import traceback
import logging
import argparse
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np
"""

def generate_torch_imports():
    """Generate torch import statements with fallbacks."""
    return """
# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = MagicMock()
    HAS_TORCH = False
    logger.warning("torch not available, using mock")
"""

def generate_transformers_imports():
    """Generate transformers import statements with fallbacks."""
    return """
# Try to import transformers
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = MagicMock()
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available, using mock")
"""

def generate_tokenizers_imports():
    """Generate tokenizers import statements with fallbacks."""
    return """
# Try to import tokenizers
try:
    import tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    tokenizers = MagicMock()
    HAS_TOKENIZERS = False
    logger.warning("tokenizers not available, using mock")

"""

def generate_sentencepiece_imports():
    """Generate sentencepiece import statements with fallbacks."""
    return """
# Try to import sentencepiece
try:
    import sentencepiece
    HAS_SENTENCEPIECE = True
except ImportError:
    sentencepiece = MagicMock()
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not available, using mock")

"""

def generate_pil_imports():
    """Generate PIL import statements with fallbacks."""
    return """
# Try to import PIL
try:
    from PIL import Image
    import requests
    from io import BytesIO
    HAS_PIL = True
except ImportError:
    Image = MagicMock()
    requests = MagicMock()
    BytesIO = MagicMock()
    HAS_PIL = False
    logger.warning("PIL or requests not available, using mock")

"""

def generate_mock_implementations(family_info):
    """Generate mock implementations for missing dependencies."""
    model_type = family_info.get("model_type", "text")
    mock_code = ""
    
    if "tokenizers" in family_info.get("dependencies", []):
        mock_code += """
if not HAS_TOKENIZERS:
    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            self.vocab_size = 32000
            
        def encode(self, text, **kwargs):
            return {"ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
            
        def decode(self, ids, **kwargs):
            return "Decoded text from mock"
            
        @staticmethod
        def from_file(vocab_filename):
            return MockTokenizer()

        tokenizers.Tokenizer = MockTokenizer

"""
    
    if "sentencepiece" in family_info.get("dependencies", []):
        mock_code += """
if not HAS_SENTENCEPIECE:
    class MockSentencePieceProcessor:
        def __init__(self, *args, **kwargs):
            self.vocab_size = 32000
            
        def encode(self, text, out_type=str):
            return [1, 2, 3, 4, 5]
            
        def decode(self, ids):
            return "Decoded text from mock"
            
        def get_piece_size(self):
            return 32000
            
        @staticmethod
        def load(model_file):
            return MockSentencePieceProcessor()

        sentencepiece.SentencePieceProcessor = MockSentencePieceProcessor

"""
    
    if "pillow" in family_info.get("dependencies", []):
        mock_code += """
if not HAS_PIL:
    class MockImage:
        @staticmethod
        def open(file):
            class MockImg:
                def __init__(self):
                    self.size = (224, 224)
                def convert(self, mode):
                    return self
                def resize(self, size):
                    return self
            return MockImg()
            
    class MockRequests:
        @staticmethod
        def get(url):
            class MockResponse:
                def __init__(self):
                    self.content = b"mock image data"
                def raise_for_status(self):
                    pass
            return MockResponse()

        Image.open = MockImage.open
        requests.get = MockRequests.get

"""
    
    return mock_code

def generate_hardware_check():
    """Generate hardware detection function."""
    return """
# Hardware detection
def check_hardware():
    \"\"\"Check available hardware and return capabilities.\"\"\"
    capabilities = {
        "cpu": True,
        "cuda": False,
        "cuda_version": None,
        "cuda_devices": 0,
        "mps": False,
        "openvino": False
    }
    
    # Check CUDA
    if HAS_TORCH:
        capabilities["cuda"] = torch.cuda.is_available(),
        if capabilities["cuda"]:
            capabilities["cuda_devices"] = torch.cuda.device_count(),
            capabilities["cuda_version"] = torch.version.cuda
        
    # Check MPS (Apple Silicon)
    if HAS_TORCH and hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        capabilities["mps"] = torch.mps.is_available()
        
    # Check OpenVINO
    try:
        import openvino
        capabilities["openvino"] = True
    except ImportError:
        pass
    
    return capabilities

# Get hardware capabilities
HW_CAPABILITIES = check_hardware()
"""

def generate_model_registry(family_info):
    """Generate model registry for specific family."""
    registry_name = f"{family_info['name'].upper()}_MODELS_REGISTRY"
    
    registry = f"""
# Models registry - Maps model IDs to their specific configurations
{registry_name} = {{
    "{family_info['base_model']}": {{
        "description": "{family_info.get('description', family_info['name'] + ' base model')}",
        "class": "{family_info['class_name']}",
    }},
"""
    
    # Add additional models if specified
    for model_id, model_info in family_info.get("additional_models", {}).items():
        registry += f"""    "{model_id}": {{
        "description": "{model_info.get('description', model_id)}",
        "class": "{model_info.get('class', family_info['class_name'])}",
    }},
"""
    
    registry += "}\n"
    return registry

def generate_test_class(family_info):
    """Generate test class definition."""
    class_name = f"Test{family_info['name'].capitalize()}Models"
    registry_name = f"{family_info['name'].upper()}_MODELS_REGISTRY"
    base_model = family_info.get("base_model", family_info.get("model_id", ""))
    model_type = family_info.get("model_type", "text")
    
    init_method = f"""
class {class_name}:
    \"\"\"Base test class for all {family_info['name']}-family models.\"\"\"
    
    def __init__(self, model_id=None):
        \"\"\"Initialize the test class for a specific model or default.\"\"\"
        self.model_id = model_id or "{base_model}"
        
        # Verify model exists in registry
        if self.model_id not in {registry_name}:
            logger.warning(f"Model {{self.model_id}} not in registry, using default configuration")
            self.model_info = {registry_name}["{base_model}"]
        else:
            self.model_info = {registry_name}[self.model_id]
            
        # Define model parameters
        self.task = "{family_info.get('task', 'text-classification')}"
        self.class_name = self.model_info["class"]
        self.description = self.model_info["description"]
        
        # Define test inputs
"""
    
    # Add appropriate test inputs based on model type
    if model_type == "text":
        init_method += f"""        self.test_text = "{family_info.get('test_text', 'Test input text.')}"
"""
    elif model_type == "vision":
        init_method += f"""        self.test_image_url = "{family_info.get('test_image_url', 'http://images.cocodataset.org/val2017/000000039769.jpg')}"
"""
    elif model_type == "audio":
        init_method += f"""        self.test_audio = "{family_info.get('test_audio', 'test_audio.wav')}"
"""
    
    # Add hardware preference detection
    init_method += """
        # Configure hardware preference
        # Check for CPU-only flag from command line args
        try:
            # Access command-line args safely (defined at module level)
            if 'args' in globals() and hasattr(args, 'cpu_only') and args.cpu_only:
                self.preferred_device = "cpu"
            elif HW_CAPABILITIES["cuda"]:
                self.preferred_device = "cuda"
            elif HW_CAPABILITIES["mps"]:
                self.preferred_device = "mps"
            else:
                self.preferred_device = "cpu"
        except:
            # Fallback if args are not defined
            if HW_CAPABILITIES["cuda"]:
                self.preferred_device = "cuda"
            elif HW_CAPABILITIES["mps"]:
                self.preferred_device = "mps"
            else:
                self.preferred_device = "cpu"
        
        logger.info(f"Using {self.preferred_device} as preferred device")
        
        # Results storage
        self.results = {}
        self.examples = []
        self.performance_stats = {}
"""
    
    return init_method

def generate_pipeline_input_preparation(family_info):
    """Generate code to prepare input for pipeline."""
    task = family_info.get("task", "text-classification")
    model_type = family_info.get("model_type", "text")
    architecture_type = family_info.get("architecture_type", "encoder_only")
    
    if model_type == "text":
        return "pipeline_input = self.test_text"
    elif model_type == "vision":
        return """# Create a mock RGB image (3 channels, 224x224 pixels)
        import numpy as np
        from PIL import Image
        
        # Create a mock RGB image (3 channels, 224x224 pixels)
        mock_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        pipeline_input = mock_image
        
        logger.info("Created mock image for vision pipeline testing - Avoiding URL download")"""
    elif model_type == "audio":
        return """if os.path.exists(self.test_audio):
            pipeline_input = self.test_audio
        else:
            # Use a sample array if file not found
            pipeline_input = np.zeros(16000)"""
    else:
        return "pipeline_input = None  # Default empty input"

def generate_tokenizer_initialization(family_info):
    """Generate tokenizer initialization code specific to each model type."""
    model_type = family_info.get("model_type", "text")
    architecture_type = family_info.get("architecture_type", "encoder_only")
    
    if model_type != "text":
        return ""
    
    if architecture_type == "decoder_only":
        # GPT-2 style models need pad_token fix
        return """
        # First load the tokenizer to fix the padding token
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token for tokenizer")
            
        # Create pipeline with fixed tokenizer
        pipeline_kwargs["tokenizer"] = tokenizer"""
    else:
        return ""

def apply_indentation(code, base_indent=0):
    """
    Apply consistent indentation to code blocks.
    
    Args:
        code: The code string to indent
        base_indent: The base indentation level (number of spaces)
        
    Returns:
        Properly indented code string
    """
    # Split the code into lines
    lines = code.strip().split('\n')
    
    # Determine the minimum indentation of non-empty lines
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # Skip empty lines
            leading_spaces = len(line) - len(line.lstrip())
            min_indent = min(min_indent, leading_spaces)
    
    # If no indentation found, set to 0
    if min_indent == float('inf'):
        min_indent = 0
    
    # Remove the minimum indentation from all lines and add the base indentation
    indented_lines = []
    indent_spaces = ' ' * base_indent
    
    for line in lines:
        if line.strip():  # If not an empty line
            # Remove original indentation and add new base indentation
            indented_line = indent_spaces + line[min_indent:]
            indented_lines.append(indented_line)
        else:
            # For empty lines, just add base indentation
            indented_lines.append(indent_spaces)
    
    # Join the lines back into a single string
    return '\n'.join(indented_lines)

def generate_test_pipeline(family_info):
    """Generate pipeline test method."""
    model_type = family_info.get("model_type", "text")
    architecture_type = family_info.get("architecture_type", "encoder_only")
    dependencies = family_info.get("dependencies", [])
    
    # Create dependency checks with proper indentation - careful with spacing
    dependency_checks = ""
    if "tokenizers" in dependencies:
        dependency_checks += """
        if not HAS_TOKENIZERS:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_deps"] = ["tokenizers>=0.11.0"]
            results["pipeline_success"] = False
            return results
"""
    
    if "sentencepiece" in dependencies:
        dependency_checks += """
        if not HAS_SENTENCEPIECE:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_deps"] = ["sentencepiece>=0.1.91"]
            results["pipeline_success"] = False
            return results
"""
    
    if "pillow" in dependencies:
        dependency_checks += """
        if not HAS_PIL:
            results["pipeline_error_type"] = "missing_dependency"
            results["pipeline_missing_deps"] = ["pillow>=8.0.0", "requests>=2.25.0"]
            results["pipeline_success"] = False
            return results
"""
    
    tokenizer_init = generate_tokenizer_initialization(family_info)
    pipeline_input_prep = generate_pipeline_input_preparation(family_info)
    
    # The method content should already have proper indentation (4 spaces for method content)
    method_content = f"""def test_pipeline(self, device="auto"):
    \"\"\"Test the model using transformers pipeline API.\"\"\"
    if device == "auto":
        device = self.preferred_device
    
    results = {{
        "model": self.model_id,
        "device": device,
        "task": self.task,
        "class": self.class_name
    }}
    
    # Check for dependencies
    if not HAS_TRANSFORMERS:
        results["pipeline_error_type"] = "missing_dependency"
        results["pipeline_missing_core"] = ["transformers"]
        results["pipeline_success"] = False
        return results
    {dependency_checks}
    
    try:
        logger.info(f"Testing {{self.model_id}} with pipeline() on {{device}}...")
        
        # Create pipeline with appropriate parameters
        pipeline_kwargs = {{
            "task": self.task,
            "model": self.model_id,
            "device": device
        }}
        
        # Time the model loading
        load_start_time = time.time()
        {tokenizer_init}
        pipeline = transformers.pipeline(**pipeline_kwargs)
        load_time = time.time() - load_start_time
        
        # Prepare test input
        {pipeline_input_prep}
        
        # Run warmup inference if on CUDA
        if device == "cuda":
            try:
                _ = pipeline(pipeline_input)
            except Exception:
                pass
        
        # Run multiple inference passes
        num_runs = 3
        times = []
        outputs = []
        
        for _ in range(num_runs):
            start_time = time.time()
            output = pipeline(pipeline_input)
            end_time = time.time()
            times.append(end_time - start_time)
            outputs.append(output)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Store results
        results["pipeline_success"] = True
        results["pipeline_avg_time"] = avg_time
        results["pipeline_min_time"] = min_time
        results["pipeline_max_time"] = max_time
        results["pipeline_load_time"] = load_time
        results["pipeline_error_type"] = "none"
        
        # Add to examples
        self.examples.append({{
            "method": f"pipeline() on {{device}}",
            "input": str(pipeline_input),
            "output_preview": str(outputs[0])[:200] + "..." if len(str(outputs[0])) > 200 else str(outputs[0])
        }})
        
        # Store in performance stats
        self.performance_stats[f"pipeline_{{device}}"] = {{
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "load_time": load_time,
            "num_runs": num_runs
        }}
        
    except Exception as e:
        # Store error information
        results["pipeline_success"] = False
        results["pipeline_error"] = str(e)
        results["pipeline_traceback"] = traceback.format_exc()
        logger.error(f"Error testing pipeline on {{device}}: {{e}}")
        
        # Classify error type
        error_str = str(e).lower()
        traceback_str = traceback.format_exc().lower()
        
        if "cuda" in error_str or "cuda" in traceback_str:
            results["pipeline_error_type"] = "cuda_error"
        elif "memory" in error_str:
            results["pipeline_error_type"] = "out_of_memory"
        elif "no module named" in error_str:
            results["pipeline_error_type"] = "missing_dependency"
        else:
            results["pipeline_error_type"] = "other"
        
    # Add to overall results
    self.results[f"pipeline_{{device}}"] = results
    return results"""
    
    # Apply the base indentation for a class method (4 spaces)
    method = apply_indentation(method_content, 4)
    
    return method

def generate_from_pretrained_input_preparation(family_info):
    """Generate code for input preparation with from_pretrained."""
    model_type = family_info.get("model_type", "text")
    architecture_type = family_info.get("architecture_type", "encoder_only")
    
    if model_type == "text":
        base_code = """test_input = self.test_text
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt")"""
        
        if architecture_type == "encoder_decoder":
            # For T5-like models, add empty decoder inputs
            base_code += """
            
        # Add decoder inputs for encoder-decoder models
        decoder_input_ids = tokenizer("", return_tensors="pt")["input_ids"]
        inputs["decoder_input_ids"] = decoder_input_ids
        
        logger.info("Added empty decoder_input_ids for encoder-decoder model")"""
        elif architecture_type == "decoder_only":
            # For GPT2-like models, ensure we've set padding token
            base_code += """
            
        # For decoder-only models like GPT-2, ensure padding token is set
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token for tokenizer")"""
            
        return base_code
    
    elif model_type == "vision":
        return """# Prepare test input - Vision models require proper image input
        # Create a mock image tensor of the right shape
        import numpy as np
        
        # Create a mock image of the right shape (batch_size, channels, height, width)
        # Default vision input size is 224x224 pixels with 3 color channels
        batch_size = 1
        num_channels = 3  # RGB
        height = 224
        width = 224
        
        # Create random image tensor
        random_pixel_values = torch.rand((batch_size, num_channels, height, width))
        
        # Properly structure the inputs for vision model
        test_input = "Image input for testing"
        inputs = {"pixel_values": random_pixel_values}
        
        logger.info("Created proper image input tensor for vision model")"""
    
    elif model_type == "audio":
        return """# Prepare audio input
        if os.path.exists(self.test_audio):
            # Use real audio file
            audio_input = self.test_audio
        else:
            # Create mock audio input - 1-second of silence at 16kHz
            audio_input = np.zeros(16000, dtype=np.float32)
            
        test_input = str(audio_input)
        
        # Process with feature extractor
        inputs = processor(audio_input, return_tensors="pt")"""
    
    else:
        return """# Default input preparation
        test_input = "Default test input"
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]))}"""

def generate_from_pretrained_model_loading(family_info):
    """Generate model loading code."""
    class_name = family_info.get("class_name", "AutoModel")
    model_type = family_info.get("model_type", "text")
    
    if model_type == "text":
        return f"""# Use appropriate model class based on model type
        model_class = None
        if self.class_name == "{class_name}":
            model_class = transformers.{class_name}
        else:
            # Fallback to Auto class
            model_class = transformers.AutoModelFor{family_info.get('auto_class', 'SequenceClassification')}
        
        # Time model loading
        model_load_start = time.time()
        model = model_class.from_pretrained(
            self.model_id,
            **pretrained_kwargs
        )
        model_load_time = time.time() - model_load_start"""
    
    elif model_type == "vision":
        return f"""# Use appropriate model class for vision
        model_class = None
        if self.class_name == "{class_name}":
            model_class = transformers.{class_name}
        else:
            # Fallback to Auto class
            model_class = transformers.AutoModelForImageClassification
        
        # Time model loading
        model_load_start = time.time()
        model = model_class.from_pretrained(
            self.model_id,
            **pretrained_kwargs
        )
        model_load_time = time.time() - model_load_start"""
    
    elif model_type == "audio":
        return f"""# Use appropriate model class for audio
        model_class = None
        if self.class_name == "{class_name}":
            model_class = transformers.{class_name}
        else:
            # Fallback to Auto class
            model_class = transformers.AutoModelForAudioClassification
        
        # Time model loading
        model_load_start = time.time()
        model = model_class.from_pretrained(
            self.model_id,
            **pretrained_kwargs
        )
        model_load_time = time.time() - model_load_start"""
    
    else:
        return """# Generic model loading
        model_class = transformers.AutoModel
        
        # Time model loading
        model_load_start = time.time()
        model = model_class.from_pretrained(
            self.model_id,
            **pretrained_kwargs
        )
        model_load_time = time.time() - model_load_start"""

def generate_from_pretrained_output_processing(family_info):
    """Generate output processing code based on family info."""
    model_type = family_info.get("model_type", "text")
    architecture_type = family_info.get("architecture_type", "encoder_only")
    
    if model_type == "text":
        if architecture_type == "encoder_only":
            return """# Process masked language modeling output
        if hasattr(outputs[0], "logits"):
            logits = outputs[0].logits
            # Get predictions for masked tokens
            predicted_token_ids = torch.argmax(logits, dim=-1)
            
            if hasattr(tokenizer, "decode"):
                predictions = [tokenizer.decode(token_ids) for token_ids in predicted_token_ids]
                predictions = [{"token": pred, "score": 1.0} for pred in predictions[:5]]
            else:
                predictions = [{"token": "<mask>", "score": 1.0}]
        else:
            predictions = [{"token": "<mask>", "score": 1.0}]"""
        
        elif architecture_type == "decoder_only":
            return """# Process generation output
        if hasattr(outputs[0], "logits"):
            logits = outputs[0].logits
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            if hasattr(tokenizer, "decode"):
                next_token = tokenizer.decode([next_token_id])
                predictions = [{"token": next_token, "score": 1.0}]
            else:
                predictions = [{"generated_text": "Mock generated text"}]
        else:
            predictions = [{"generated_text": "Mock generated text"}]"""
        
        elif architecture_type == "encoder_decoder":
            return """# Process encoder-decoder output
        if hasattr(outputs[0], "logits"):
            logits = outputs[0].logits
            generated_ids = torch.argmax(logits, dim=-1)
            
            if hasattr(tokenizer, "decode"):
                decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                predictions = [{"generated_text": decoded_output}]
            else:
                predictions = [{"generated_text": "Mock generated text"}]
        else:
            predictions = [{"generated_text": "Mock generated text"}]"""
    
    elif model_type == "vision":
        return """# Process vision model output
        if hasattr(outputs[0], "logits"):
            logits = outputs[0].logits
            # Get the predicted class
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            
            # In a real scenario, you would map this to class labels
            predictions = [{"label": f"class_{predicted_class_idx}", "score": 1.0}]
        else:
            predictions = [{"label": "predicted_class", "score": 1.0}]"""
    
    elif model_type == "audio":
        return """# Process audio model output
        if hasattr(outputs[0], "logits"):
            logits = outputs[0].logits
            # Get the predicted class
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            
            # In a real scenario, you would map this to class labels
            predictions = [{"label": f"class_{predicted_class_idx}", "score": 1.0}]
        else:
            predictions = [{"label": "predicted_class", "score": 1.0}]"""
    
    else:
        return """# Generic output processing
        predictions = [{"output": "Processed model output"}]"""

def generate_test_from_pretrained(family_info):
    """Generate from_pretrained test method."""
    model_type = family_info.get("model_type", "text")
    tokenizer_class = family_info.get("tokenizer_class", "AutoTokenizer")
    processor_class = family_info.get("processor_class", "")
    dependencies = family_info.get("dependencies", [])
    
    # Create dependency checks with proper indentation - hardcoded spacing
    dependency_checks = ""
    if "tokenizers" in dependencies:
        dependency_checks += """
        if not HAS_TOKENIZERS:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_deps"] = ["tokenizers>=0.11.0"]
            results["from_pretrained_success"] = False
            return results
"""
    
    if "sentencepiece" in dependencies:
        dependency_checks += """
        if not HAS_SENTENCEPIECE:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_deps"] = ["sentencepiece>=0.1.91"]
            results["from_pretrained_success"] = False
            return results
"""
    
    if "pillow" in dependencies:
        dependency_checks += """
        if not HAS_PIL:
            results["from_pretrained_error_type"] = "missing_dependency"
            results["from_pretrained_missing_deps"] = ["pillow>=8.0.0", "requests>=2.25.0"]
            results["from_pretrained_success"] = False
            return results
"""
    
    # Process tokenizer/processor loading code with proper indentation
    tokenizer_loading_code = ""
    if model_type == "text":
        tokenizer_block = f"""# Time tokenizer loading
tokenizer_load_start = time.time()
tokenizer = transformers.{tokenizer_class}.from_pretrained(
    self.model_id,
    **pretrained_kwargs
)
tokenizer_load_time = time.time() - tokenizer_load_start"""
        tokenizer_loading_code = apply_indentation(tokenizer_block, 12)  # 12 spaces for nested code
    elif model_type == "vision" and processor_class:
        processor_block = f"""# Time processor loading
tokenizer_load_start = time.time()
processor = transformers.{processor_class}.from_pretrained(
    self.model_id,
    **pretrained_kwargs
)
tokenizer = processor  # For code compatibility
tokenizer_load_time = time.time() - tokenizer_load_start"""
        tokenizer_loading_code = apply_indentation(processor_block, 12)
    elif model_type == "vision":
        vision_block = """# Time processor loading for vision
tokenizer_load_start = time.time()
processor = transformers.AutoImageProcessor.from_pretrained(
    self.model_id,
    **pretrained_kwargs
)
tokenizer = processor  # For code compatibility
tokenizer_load_time = time.time() - tokenizer_load_start"""
        tokenizer_loading_code = apply_indentation(vision_block, 12)
    elif model_type == "audio":
        audio_block = """# Time processor loading for audio
tokenizer_load_start = time.time()
processor = transformers.AutoFeatureExtractor.from_pretrained(
    self.model_id,
    **pretrained_kwargs
)
tokenizer = processor  # For code compatibility
tokenizer_load_time = time.time() - tokenizer_load_start"""
        tokenizer_loading_code = apply_indentation(audio_block, 12)
    else:
        generic_block = """# Time tokenizer loading generic
tokenizer_load_start = time.time()
tokenizer = transformers.AutoTokenizer.from_pretrained(
    self.model_id,
    **pretrained_kwargs
)
tokenizer_load_time = time.time() - tokenizer_load_start"""
        tokenizer_loading_code = apply_indentation(generic_block, 12)
    
    # Get model loading code and properly indent
    model_loading_raw = generate_from_pretrained_model_loading(family_info)
    model_loading_code = apply_indentation(model_loading_raw, 12)  # 12 spaces for nested block
    
    # Get input preparation code and properly indent
    input_preparation_raw = generate_from_pretrained_input_preparation(family_info)
    input_preparation_code = apply_indentation(input_preparation_raw, 12)
    
    # Get output processing code and properly indent
    output_processing_raw = generate_from_pretrained_output_processing(family_info)
    output_processing_code = apply_indentation(output_processing_raw, 12)
    
    # Create the main method content with base indentation for class methods
    method_content = f"""def test_from_pretrained(self, device="auto"):
    \"\"\"Test the model using direct from_pretrained loading.\"\"\"
    if device == "auto":
        device = self.preferred_device
    
    results = {{
        "model": self.model_id,
        "device": device,
        "task": self.task,
        "class": self.class_name
    }}
    
    # Check for dependencies
    if not HAS_TRANSFORMERS:
        results["from_pretrained_error_type"] = "missing_dependency"
        results["from_pretrained_missing_core"] = ["transformers"]
        results["from_pretrained_success"] = False
        return results
{dependency_checks}
    
    try:
        logger.info(f"Testing {{self.model_id}} with from_pretrained() on {{device}}...")
        
        # Common parameters for loading
        pretrained_kwargs = {{
            "local_files_only": False
        }}
        
{tokenizer_loading_code}
        
{model_loading_code}
        
        # Move model to device
        if device != "cpu":
            model = model.to(device)
        
        # Prepare test input
{input_preparation_code}
        
        # Move inputs to device
        if device != "cpu":
            inputs = {{key: val.to(device) for key, val in inputs.items()}}
        
        # Run warmup inference if using CUDA
        if device == "cuda":
            try:
                with torch.no_grad():
                    _ = model(**inputs)
            except Exception:
                pass
        
        # Run multiple inference passes
        num_runs = 3
        times = []
        outputs = []
        
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                output = model(**inputs)
                end_time = time.time()
                times.append(end_time - start_time)
                outputs.append(output)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Process output
{output_processing_code}
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = (param_count * 4) / (1024 * 1024)  # Rough size in MB
        
        # Store results
        results["from_pretrained_success"] = True
        results["from_pretrained_avg_time"] = avg_time
        results["from_pretrained_min_time"] = min_time
        results["from_pretrained_max_time"] = max_time
        results["tokenizer_load_time"] = tokenizer_load_time
        results["model_load_time"] = model_load_time
        results["model_size_mb"] = model_size_mb
        results["from_pretrained_error_type"] = "none"
        
        # Add predictions if available
        if 'predictions' in locals():
            results["predictions"] = predictions
        
        # Add to examples
        example_data = {{
            "method": f"from_pretrained() on {{device}}",
            "input": str(test_input)
        }}
        
        if 'predictions' in locals():
            example_data["predictions"] = predictions
        
        self.examples.append(example_data)
        
        # Store in performance stats
        self.performance_stats[f"from_pretrained_{{device}}"] = {{
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "tokenizer_load_time": tokenizer_load_time,
            "model_load_time": model_load_time,
            "model_size_mb": model_size_mb,
            "num_runs": num_runs
        }}
        
    except Exception as e:
        # Store error information
        results["from_pretrained_success"] = False
        results["from_pretrained_error"] = str(e)
        results["from_pretrained_traceback"] = traceback.format_exc()
        logger.error(f"Error testing from_pretrained on {{device}}: {{e}}")
        
        # Classify error type
        error_str = str(e).lower()
        traceback_str = traceback.format_exc().lower()
        
        if "cuda" in error_str or "cuda" in traceback_str:
            results["from_pretrained_error_type"] = "cuda_error"
        elif "memory" in error_str:
            results["from_pretrained_error_type"] = "out_of_memory"
        elif "no module named" in error_str:
            results["from_pretrained_error_type"] = "missing_dependency"
        else:
            results["from_pretrained_error_type"] = "other"
        
    # Add to overall results
    self.results[f"from_pretrained_{{device}}"] = results
    return results"""
    
    # Apply the base indentation for a class method (4 spaces)
    method = apply_indentation(method_content, 4)
    
    return method

def generate_run_tests():
    """Generate run_tests method."""
    # Create the method content with minimal indentation
    method_content = """def run_tests(self, all_hardware=False):
    \"\"\"
    Run all tests for this model.
    
    Args:
        all_hardware: If True, tests on all available hardware (CPU, CUDA, OpenVINO)
    
    Returns:
        Dict containing test results
    \"\"\"
    # Always test on default device
    self.test_pipeline()
    self.test_from_pretrained()
    
    # Test on all available hardware if requested
    if all_hardware:
        # Always test on CPU
        if self.preferred_device != "cpu":
            self.test_pipeline(device="cpu")
            self.test_from_pretrained(device="cpu")
        
        # Test on CUDA if available
        if HW_CAPABILITIES["cuda"] and self.preferred_device != "cuda":
            self.test_pipeline(device="cuda")
            self.test_from_pretrained(device="cuda")
        
        # Test on OpenVINO if available
        if HW_CAPABILITIES["openvino"]:
            self.test_with_openvino()
    
    # Build final results
    return {
        "results": self.results,
        "examples": self.examples,
        "performance": self.performance_stats,
        "hardware": HW_CAPABILITIES,
        "metadata": {
            "model": self.model_id,
            "task": self.task,
            "class": self.class_name,
            "description": self.description,
            "timestamp": datetime.datetime.now().isoformat(),
            "has_transformers": HAS_TRANSFORMERS,
            "has_torch": HAS_TORCH
        }
    }"""
    
    # Apply the base indentation for a class method (4 spaces)
    method = apply_indentation(method_content, 4)
    
    return method

def generate_save_utils(family_info):
    """Generate utilities for saving results and model listings."""
    family_name = family_info["name"]
    registry_name = f"{family_name.upper()}_MODELS_REGISTRY"
    
    return f"""
def save_results(model_id, results, output_dir="collected_results"):
    \"\"\"Save test results to a file.\"\"\"
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model ID
    safe_model_id = model_id.replace("/", "__")
    filename = f"hf_{family_name}_{{safe_model_id}}_{{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {{output_path}}")
    return output_path

def get_available_models():
    \"\"\"Get a list of all available {family_name} models in the registry.\"\"\"
    return list({registry_name}.keys())

def test_all_models(output_dir="collected_results", all_hardware=False):
    \"\"\"Test all registered {family_name} models.\"\"\"
    models = get_available_models()
    results = {{}}
    
    for model_id in models:
        logger.info(f"Testing model: {{model_id}}")
        tester = Test{family_name.capitalize()}Models(model_id)
        model_results = tester.run_tests(all_hardware=all_hardware)
        
        # Save individual results
        save_results(model_id, model_results, output_dir=output_dir)
        
        # Add to summary
        results[model_id] = {{
            "success": any((r.get("pipeline_success", False) for r in model_results["results"].values() 
                        if r.get("pipeline_success") is not False))
        }}
    
    # Save summary
    summary_path = os.path.join(output_dir, f"hf_{family_name}_summary_{{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved summary to {{summary_path}}")
    return results
"""

def generate_main_function(family_info):
    """Generate main function for the test script."""
    family_name = family_info["name"]
    base_model = family_info.get("base_model", family_info.get("model_id", ""))
    registry_name = f"{family_name.upper()}_MODELS_REGISTRY"
    
    return f"""
def main():
    \"\"\"Command-line entry point.\"\"\"
    parser = argparse.ArgumentParser(description="Test {family_name}-family models")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Specific model to test")
    model_group.add_argument("--all-models", action="store_true", help="Test all registered models")
    
    # Hardware options
    parser.add_argument("--all-hardware", action="store_true", help="Test on all available hardware")
    parser.add_argument("--cpu-only", action="store_true", help="Test only on CPU")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="collected_results", help="Directory for output files")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    # List options
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models = get_available_models()
        print("\\nAvailable {family_name}-family models:")
        for model in models:
            info = {registry_name}[model]
            print(f"  - {{model}} ({{info['class']}}): {{info['description']}}")
        return
    
    # Create output directory if needed
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test all models if requested
    if args.all_models:
        results = test_all_models(output_dir=args.output_dir, all_hardware=args.all_hardware)
        
        # Print summary
        print("\\n{family_name.capitalize()} Models Testing Summary:")
        total = len(results)
        successful = sum(1 for r in results.values() if r["success"])
        print(f"Successfully tested {{successful}} of {{total}} models ({{successful/total*100:.1f}}%)")
        return
    
    # Test single model (default or specified)
    model_id = args.model or "{base_model}"
    logger.info(f"Testing model: {{model_id}}")
    
    # Make args available globally for hardware detection in the test class
    global args
    
    # Override preferred device if CPU only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("CPU-only mode enabled - disabled CUDA")
    
    # Run test
    tester = Test{family_name.capitalize()}Models(model_id)
    results = tester.run_tests(all_hardware=args.all_hardware)
    
    # Save results if requested
    if args.save:
        save_results(model_id, results, output_dir=args.output_dir)
    
    # Print summary
    success = any((r.get("pipeline_success", False) for r in results["results"].values() 
                 if r.get("pipeline_success") is not False))
    
    print("\\nTEST RESULTS SUMMARY:")
    if success:
        print(f"✅ Successfully tested {{model_id}}")
        
        # Print performance highlights
        for device, stats in results["performance"].items():
            if "avg_time" in stats:
                print(f"  - {{device}}: {{stats['avg_time']:.4f}}s average inference time")
        
        # Print example outputs if available
        if results.get("examples") and len(results["examples"]) > 0:
            print("\\nExample output:")
            example = results["examples"][0]
            if "predictions" in example:
                print(f"  Input: {{example['input']}}")
                print(f"  Predictions: {{example['predictions']}}")
            elif "output_preview" in example:
                print(f"  Input: {{example['input']}}")
                print(f"  Output: {{example['output_preview']}}")
    else:
        print(f"❌ Failed to test {{model_id}}")
        
        # Print error information
        for test_name, result in results["results"].items():
            if "pipeline_error" in result:
                print(f"  - Error in {{test_name}}: {{result.get('pipeline_error_type', 'unknown')}}")
                print(f"    {{result.get('pipeline_error', 'Unknown error')}}")
    
        print("\\nFor detailed results, use --save flag and check the JSON output file.")

if __name__ == "__main__":
    main()
"""

def generate_test_file(family_info):
    """Generate a complete test file for a model family."""
    file_content = ""
    
    # Add imports
    file_content += generate_imports()
    file_content += generate_docstring(family_info)
    file_content += generate_basic_imports()
    
    # Add specific imports based on dependencies
    if "torch" in family_info.get("dependencies", []):
        file_content += generate_torch_imports()
    if "transformers" in family_info.get("dependencies", []) or True:  # Always include transformers
        file_content += generate_transformers_imports()
    if "tokenizers" in family_info.get("dependencies", []):
        file_content += generate_tokenizers_imports()
    if "sentencepiece" in family_info.get("dependencies", []):
        file_content += generate_sentencepiece_imports()
    if "pillow" in family_info.get("dependencies", []):
        file_content += generate_pil_imports()
    
    # Add mock implementations
    file_content += generate_mock_implementations(family_info)
    
    # Add hardware detection
    file_content += generate_hardware_check()
    
    # Add model registry
    file_content += generate_model_registry(family_info)
    
    # Add test class and init method
    file_content += generate_test_class(family_info)
    
    # Add test methods - ensure proper spacing
    file_content += "\n"
    file_content += generate_test_pipeline(family_info)
    file_content += "\n"
    file_content += generate_test_from_pretrained(family_info)
    file_content += "\n"
    file_content += generate_run_tests()
    file_content += "\n\n"  # Extra space after the class definition
    
    # Add utility functions
    file_content += generate_save_utils(family_info)
    
    # Add main function
    file_content += generate_main_function(family_info)
    
    return file_content

def fix_method_boundaries(file_content):
    """Fix method boundaries to ensure proper spacing and indentation."""
    # First add proper spacing between methods
    file_content = file_content.replace("        return results\n    def ", "        return results\n\n    def ")
    
    # Make sure __init__ has correct spacing after it
    file_content = file_content.replace("        self.performance_stats = {}\n    def ", "        self.performance_stats = {}\n\n    def ")
        
    # Place all method declarations at the right indentation level
    file_content = re.sub(r'(\s+)def test_pipeline\(', r'    def test_pipeline(', file_content)
    file_content = re.sub(r'(\s+)def test_from_pretrained\(', r'    def test_from_pretrained(', file_content)
    file_content = re.sub(r'(\s+)def run_tests\(', r'    def run_tests(', file_content)
    
    return file_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model test files")
    parser.add_argument("--family", type=str, required=True, help="Model family (bert, gpt2, t5, vit)")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    if args.family not in MODEL_FAMILIES:
        print(f"Error: Unknown model family '{args.family}'. Choose from: {', '.join(MODEL_FAMILIES.keys())}")
        sys.exit(1)
    
    family_info = MODEL_FAMILIES[args.family]
    family_info["name"] = args.family
    
    # Generate test file
    print(f"Generating test file for {args.family}...")
    file_content = generate_test_file(family_info)
    
    # Fix indentation issues at method boundaries
    file_content = fix_method_boundaries(file_content)
    
    # Write to file
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, f"test_hf_{args.family}.py")
    else:
        output_path = f"test_hf_{args.family}.py"
    
    with open(output_path, "w") as f:
        f.write(file_content)
    
    print(f"Test file generated: {output_path}")