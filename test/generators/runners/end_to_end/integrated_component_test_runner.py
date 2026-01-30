#!/usr/bin/env python3
"""
Integrated Component Test Runner for IPFS Accelerate

This script implements an improved end-to-end testing framework that focuses on generating
and testing skill, test, and benchmark components together for every model. It:

1. Generates all three components together using template-driven approach
2. Validates components work together as a cohesive unit
3. Creates "expected_results" and "collected_results" folders for verification
4. Generates markdown documentation of HuggingFace class skills
5. Focuses on fixing generators rather than individual files
6. Implements template-driven approach for maintenance efficiency

Usage:
    python integrated_component_test_runner.py --model bert-base-uncased --hardware cuda
    python integrated_component_test_runner.py --model-family text-embedding --hardware all
    python integrated_component_test_runner.py --all-models --priority-hardware
"""

import os
import sys
import json
import time
import uuid
import argparse
import logging
import datetime
import tempfile
import shutil
import concurrent.futures
import subprocess
import numpy as np
import inspect
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import existing utilities
from simple_utils import ensure_dir_exists
from template_validation import ModelValidator, ResultComparer
from model_documentation_generator import ModelDocGenerator, generate_model_documentation

# Enhanced model documentation generator specifically for HuggingFace models
class EnhancedModelDocGenerator(ModelDocGenerator):
    """
    Enhanced documentation generator that includes HuggingFace-specific information.
    
    This class extends the standard ModelDocGenerator to provide richer documentation
    for HuggingFace models, including model architecture, API details, and usage examples.
    """
    
    def generate_markdown(self, test_results=None, benchmark_results=None, git_hash=None):
        """
        Generate markdown documentation for the model implementation.
        
        Args:
            test_results: Test results dictionary (optional)
            benchmark_results: Benchmark results dictionary (optional)
            git_hash: Git hash of the current commit (optional)
            
        Returns:
            Generated markdown documentation as string
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if git_hash is None:
            git_hash = "unknown"
            
        # Extract docstrings and metadata from files
        skill_docstrings = self.extract_docstrings(self.skill_path)
        test_docstrings = self.extract_docstrings(self.test_path)
        benchmark_docstrings = self.extract_docstrings(self.benchmark_path)
        
        # Extract API details from skill file (methods, parameters, etc.)
        skill_api = self.extract_api_details(self.skill_path)
        
        # Extract HuggingFace-specific information
        hf_info = self.extract_huggingface_info(self.model_name)
        
        # Get hardware-specific information
        hardware_info = self.get_hardware_info(self.hardware)
        
        # Format benchmark results if available
        benchmark_section = self._format_benchmark_results(benchmark_results)
        
        # Format test results if available
        test_section = self._format_test_results(test_results)
        
        # Create model architecture section
        model_architecture = self._create_model_architecture_section(hf_info)
        
        # Generate markdown
        md = f"""# {self.model_name} Implementation for {self.hardware}

## Overview

This document describes the implementation of [{self.model_name}]({hf_info.get('model_url', '')}) for {self.hardware} hardware.

- **Model**: {self.model_name}
- **Model Type**: {hf_info.get('model_type', 'Unknown')}
- **Hardware**: {self.hardware}
- **Generation Date**: {timestamp}
- **Git Hash**: {git_hash}

{hf_info.get('model_description', '')}

## HuggingFace Model Information

{self._format_huggingface_info(hf_info)}

## Model Architecture

{model_architecture}

## Implementation Details

### Skill Class

```python
{self._extract_class_definition(self.skill_path)}
```

### API Documentation

{self._format_api_documentation(skill_api)}

### Hardware-Specific Optimizations

{self._format_hardware_specific_info(hardware_info)}

## Usage Example

```python
# Import the skill
from {os.path.basename(self.skill_path).replace('.py', '')} import {skill_api.get('class_name', 'ModelSkill')}

# Create an instance
skill = {skill_api.get('class_name', 'ModelSkill')}()

# Set up the model
setup_success = skill.setup()

# Run inference
result = skill.run("This is a test input")
print(result)

# Clean up resources
skill.cleanup()
```

## Test Coverage

{self._format_test_coverage(test_docstrings)}

## Test Results

{test_section}

## Benchmark Results

{benchmark_section}

## Known Limitations

{hardware_info.get('limitations', 'No specific limitations documented.')}

## Additional Resources

- [HuggingFace Model Card]({hf_info.get('model_url', '')})
- [Hardware Documentation]({hardware_info.get('docs_url', '')})
- [Related Implementations](#)
"""

        return md
        
    def _format_test_results(self, test_results):
        """Format test results for markdown documentation."""
        if not test_results:
            return "No test results available."
            
        # Extract relevant information
        success = test_results.get('success', False)
        test_count = test_results.get('test_count', 0)
        execution_time = test_results.get('execution_time')
        
        status = "✅ Passed" if success else "❌ Failed"
        
        # Format test results
        result = f"""### Test Summary

- **Status**: {status}
- **Tests Run**: {test_count}"""

        # Only format execution time if it's a number
        if execution_time is not None:
            result += f"\n- **Execution Time**: {execution_time:.2f} seconds"
        else:
            result += "\n- **Execution Time**: Not available"
        
        result += "\n"
        
        # Add details if available
        if 'stdout' in test_results:
            result += "\n### Test Output\n\n```\n"
            result += test_results['stdout']
            result += "\n```\n"
            
        return result
        
    def _format_benchmark_results(self, benchmark_results):
        """Format benchmark results for markdown documentation."""
        if not benchmark_results or not isinstance(benchmark_results, dict) or 'results_by_batch' not in benchmark_results:
            return "No benchmark results available."
            
        # Extract batch results
        batch_results = benchmark_results.get('results_by_batch', {})
        
        if not batch_results:
            return "No benchmark data available."
            
        # Create a table for benchmark results
        table = "### Benchmark Summary\n\n"
        table += "| Batch Size | Average Latency (ms) | Average Throughput (items/s) |\n"
        table += "|------------|----------------------|-----------------------------|\n"
        
        for batch_size, results in batch_results.items():
            avg_latency = results.get('average_latency_ms', 0)
            avg_throughput = results.get('average_throughput_items_per_second', 0)
            table += f"| {batch_size} | {avg_latency:.2f} | {avg_throughput:.2f} |\n"
            
        # Add latency and throughput charts if data is available
        charts = "\n### Performance Charts\n\n"
        charts += "Latency and throughput charts would be added here.\n"
        
        return table + charts
        
    def _format_huggingface_info(self, hf_info):
        """Format HuggingFace model information for markdown."""
        if not hf_info:
            return "No HuggingFace information available."
            
        # Create markdown for HuggingFace info
        md = ""
        
        if 'model_card' in hf_info:
            md += f"**Model Card**: [{self.model_name}]({hf_info['model_url']})\n\n"
            
        if 'tags' in hf_info:
            md += "**Tags**: " + ", ".join([f"`{tag}`" for tag in hf_info['tags']]) + "\n\n"
            
        if 'downloads' in hf_info:
            md += f"**Downloads**: {hf_info['downloads']:,}\n\n"
            
        if 'likes' in hf_info:
            md += f"**Likes**: {hf_info['likes']:,}\n\n"
            
        if 'language' in hf_info:
            md += f"**Language**: {hf_info['language']}\n\n"
            
        if 'library' in hf_info:
            md += f"**Library**: {hf_info['library']}\n\n"
            
        if 'pipeline_tag' in hf_info:
            md += f"**Pipeline**: {hf_info['pipeline_tag']}\n\n"
            
        return md
        
    def _format_hardware_specific_info(self, hardware_info):
        """Format hardware-specific information for markdown."""
        if not hardware_info:
            return "No hardware-specific information available."
            
        # Create markdown for hardware info
        md = ""
        
        if 'description' in hardware_info:
            md += f"{hardware_info['description']}\n\n"
            
        if 'optimizations' in hardware_info:
            md += "**Optimizations**:\n\n"
            for opt in hardware_info['optimizations']:
                md += f"- {opt}\n"
            md += "\n"
            
        if 'requirements' in hardware_info:
            md += "**Requirements**:\n\n"
            for req in hardware_info['requirements']:
                md += f"- {req}\n"
            md += "\n"
            
        return md
        
    def _format_api_documentation(self, api_info):
        """Format API documentation for markdown."""
        if not api_info or 'methods' not in api_info:
            return "No API documentation available."
            
        # Create markdown for API docs
        md = ""
        
        for method_name, method_info in api_info.get('methods', {}).items():
            md += f"### `{method_name}`\n\n"
            
            # Add docstring
            if 'docstring' in method_info:
                md += f"{method_info['docstring']}\n\n"
                
            # Add parameters
            if 'parameters' in method_info and method_info['parameters']:
                md += "**Parameters**:\n\n"
                for param, param_info in method_info['parameters'].items():
                    param_type = param_info.get('type', 'Any')
                    param_desc = param_info.get('description', 'No description available.')
                    md += f"- `{param}` (`{param_type}`): {param_desc}\n"
                md += "\n"
                
            # Add return value
            if 'returns' in method_info:
                returns_type = method_info['returns'].get('type', 'Any')
                returns_desc = method_info['returns'].get('description', 'No description available.')
                md += f"**Returns**: `{returns_type}`: {returns_desc}\n\n"
                
        return md
        
    def _format_test_coverage(self, test_docstrings):
        """Format test coverage information for markdown."""
        if not test_docstrings or 'class' not in test_docstrings:
            return "No test coverage information available."
            
        # Extract test methods
        test_methods = {}
        for key, value in test_docstrings.items():
            if key.startswith('test_'):
                test_methods[key] = value
                
        if not test_methods:
            return "No test methods found."
            
        # Create markdown for test coverage
        md = ""
        
        for method_name, docstring in test_methods.items():
            md += f"### {method_name.replace('_', ' ').title()}\n\n"
            md += f"{docstring}\n\n"
            
        return md
        
    def _extract_class_definition(self, file_path):
        """Extract class definition from a Python file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Find the class definition for the model skill
            import re
            class_match = re.search(r'class\s+[\w_]+.*?:.*?(?=\n\s*?(?:class|def|\n|$))', content, re.DOTALL)
            if class_match:
                return class_match.group(0)
            else:
                return "# Class definition not found"
        except Exception as e:
            return f"# Error extracting class definition: {e}"
            
    def _create_model_architecture_section(self, hf_info):
        """Create a model architecture section based on HuggingFace model type."""
        model_type = hf_info.get('model_type', '').lower()
        
        if 'bert' in model_type:
            return """The BERT (Bidirectional Encoder Representations from Transformers) model consists of:

1. **Embedding Layer**: Token, position, and segment embeddings
2. **Transformer Encoder**: Multiple layers of self-attention and feed-forward networks
3. **Pooling Layer**: Special [CLS] token representation for sequence-level tasks

The implementation uses the HuggingFace `AutoModel` and `AutoTokenizer` classes for loading and running the model."""
        elif 'gpt' in model_type or 't5' in model_type or 'llama' in model_type:
            return """This text generation model architecture consists of:

1. **Embedding Layer**: Token and position embeddings
2. **Transformer Layers**: Self-attention and feed-forward networks
3. **Language Modeling Head**: Output projection layer for next token prediction

The implementation uses the HuggingFace `AutoModelForCausalLM` and `AutoTokenizer` classes."""
        elif 'vit' in model_type or 'clip' in model_type:
            return """This vision model architecture consists of:

1. **Patch Embedding**: Image patches are embedded into token representations
2. **Transformer Encoder**: Self-attention and feed-forward layers process patch embeddings
3. **Classification Head**: For image classification tasks (if applicable)

The implementation uses the HuggingFace `AutoModel` and `AutoFeatureExtractor` classes."""
        elif 'whisper' in model_type or 'wav2vec' in model_type:
            return """This audio model architecture consists of:

1. **Feature Extraction**: Converting audio waveforms to spectrograms or features
2. **Encoder**: Processing audio features with convolutional and/or transformer layers
3. **Decoder**: Generating text output (for speech-to-text models)

The implementation uses the HuggingFace `AutoModel` and appropriate processor classes."""
        else:
            return """This model's architecture details are not specifically documented. 

The implementation uses HuggingFace's `AutoModel` and appropriate tokenizer/processor classes for loading and running the model."""
            
    def extract_huggingface_info(self, model_name):
        """
        Extract information about a HuggingFace model.
        
        In a real implementation, this would query the HuggingFace API or use the huggingface_hub library.
        For this demonstration, we'll return some placeholder information.
        
        Args:
            model_name: Name of the HuggingFace model
            
        Returns:
            Dictionary with model information
        """
        model_url = f"https://huggingface.co/{model_name}"
        
        # Determine model type based on name (a simplistic approach)
        model_type = "text_embedding"  # Default
        if "bert" in model_name.lower():
            model_type = "text_embedding"
            model_description = "This is a text embedding model that produces vector representations of text."
            tags = ["transformers", "text-embedding", "bert"]
            pipeline_tag = "feature-extraction"
        elif "vit" in model_name.lower():
            model_type = "vision"
            model_description = "This is a vision transformer model that processes images."
            tags = ["transformers", "vision", "image-classification"]
            pipeline_tag = "image-classification"
        elif "whisper" in model_name.lower():
            model_type = "audio"
            model_description = "This is an audio processing model for speech recognition."
            tags = ["transformers", "audio", "automatic-speech-recognition"]
            pipeline_tag = "automatic-speech-recognition"
        elif "opt" in model_name.lower() or "t5" in model_name.lower():
            model_type = "text_generation"
            model_description = "This is a text generation model for natural language generation tasks."
            tags = ["transformers", "text-generation", "causal-lm"]
            pipeline_tag = "text-generation"
        elif "clip" in model_name.lower():
            model_type = "multimodal"
            model_description = "This is a multimodal model that processes both images and text."
            tags = ["transformers", "multimodal", "zero-shot-classification"]
            pipeline_tag = "zero-shot-image-classification"
        else:
            model_description = "This model's specific details are not available."
            tags = ["transformers"]
            pipeline_tag = "feature-extraction"
            
        # Create a mock HuggingFace model info dictionary
        model_info = {
            "model_name": model_name,
            "model_url": model_url,
            "model_type": model_type,
            "model_description": model_description,
            "tags": tags,
            "downloads": 500000,
            "likes": 2500,
            "language": "en",
            "library": "transformers",
            "pipeline_tag": pipeline_tag,
            "model_card": True
        }
        
        return model_info
        
    def extract_api_details(self, file_path):
        """
        Extract API details from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary with API details
        """
        # Find the class name and methods in the file
        class_name = None
        methods = {}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract class name
            import re
            class_match = re.search(r'class\s+([\w_]+)', content)
            if class_match:
                class_name = class_match.group(1)
                
            # Extract methods
            method_matches = re.finditer(r'def\s+([\w_]+)\s*\((self(?:,\s*[^)]+)*)\)(?:\s*->\s*([^:]+))?\s*:(.*?)(?=\n\s*def|\n\s*$|\n\s*class|$)', content, re.DOTALL)
            for match in method_matches:
                method_name = match.group(1)
                params_str = match.group(2)
                return_type = match.group(3)
                method_body = match.group(4).strip()
                
                # Skip private methods
                if method_name.startswith('_'):
                    continue
                    
                # Extract docstring
                docstring_match = re.search(r"'''(.*?)'''", method_body, re.DOTALL)
                docstring = docstring_match.group(1).strip() if docstring_match else ""
                
                # Parse parameters
                params = {}
                if params_str:
                    params_list = [p.strip() for p in params_str.split(',')]
                    for param in params_list[1:]:  # Skip 'self'
                        if ':' in param:
                            param_name, param_type = [p.strip() for p in param.split(':', 1)]
                            if '=' in param_type:
                                param_type, default = [p.strip() for p in param_type.split('=', 1)]
                            params[param_name] = {
                                "type": param_type,
                                "description": f"Parameter '{param_name}' of type '{param_type}'"
                            }
                        else:
                            param_name = param
                            if '=' in param_name:
                                param_name = param_name.split('=', 1)[0].strip()
                            params[param_name] = {
                                "type": "Any",
                                "description": f"Parameter '{param_name}'"
                            }
                
                # Parse return type
                returns = {
                    "type": return_type.strip() if return_type else "Any",
                    "description": "Return value"
                }
                
                # Store method info
                methods[method_name] = {
                    "docstring": docstring,
                    "parameters": params,
                    "returns": returns
                }
                
            return {
                "class_name": class_name,
                "methods": methods
            }
        except Exception as e:
            return {
                "class_name": "Unknown",
                "methods": {},
                "error": str(e)
            }
            
    def get_hardware_info(self, hardware):
        """
        Get hardware-specific information.
        
        Args:
            hardware: Hardware platform
            
        Returns:
            Dictionary with hardware information
        """
        # Hardware platform specific information
        if hardware == "cpu":
            return {
                "description": "CPU implementation focuses on compatibility and ease of use. It uses standard PyTorch operations without hardware-specific optimizations.",
                "optimizations": [
                    "Batched inference for improved throughput",
                    "Caching of model parameters for reduced memory usage",
                    "Lazy loading of model components"
                ],
                "requirements": [
                    "PyTorch 2.0 or later",
                    "Transformers 4.30 or later"
                ],
                "limitations": "CPU implementations may have lower performance compared to GPU or specialized hardware accelerators.",
                "docs_url": "https://pytorch.org/docs/stable/cpu.html"
            }
        elif hardware == "cuda":
            return {
                "description": "CUDA implementation leverages NVIDIA GPUs for accelerated inference. It uses CUDA-specific optimizations for maximum performance.",
                "optimizations": [
                    "Half-precision (FP16) inference",
                    "CUDA kernel fusion",
                    "Tensor core acceleration where available",
                    "Optimized memory management"
                ],
                "requirements": [
                    "CUDA 11.7 or later",
                    "PyTorch 2.0 or later with CUDA support",
                    "NVIDIA GPU with compute capability 7.0 or higher recommended"
                ],
                "limitations": "Requires NVIDIA GPU hardware and appropriate CUDA drivers.",
                "docs_url": "https://pytorch.org/docs/stable/cuda.html"
            }
        elif hardware == "openvino":
            return {
                "description": "OpenVINO implementation uses Intel's OpenVINO toolkit for optimized inference on Intel hardware (CPU, GPU, VPU).",
                "optimizations": [
                    "Model quantization (INT8)",
                    "Layer fusion",
                    "Hardware-specific optimizations",
                    "Heterogeneous execution"
                ],
                "requirements": [
                    "OpenVINO 2023.0 or later",
                    "PyTorch 2.0 or later",
                    "optimum-intel package"
                ],
                "limitations": "Best performance on Intel hardware; may not offer significant benefits on other platforms.",
                "docs_url": "https://docs.openvino.ai/"
            }
        elif hardware == "webgpu":
            return {
                "description": "WebGPU implementation enables GPU-accelerated inference in web browsers using the WebGPU API.",
                "optimizations": [
                    "Optimized shader programs",
                    "Workgroup-aware computation",
                    "Minimized CPU-GPU synchronization",
                    "Compute shader optimizations (March 2025)"
                ],
                "requirements": [
                    "Modern browser with WebGPU support (Chrome 113+, Firefox 116+, Edge 113+)",
                    "GPU with appropriate driver support"
                ],
                "limitations": "WebGPU support varies across browsers and devices. Falls back to CPU if WebGPU is unavailable.",
                "docs_url": "https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API"
            }
        else:
            return {
                "description": f"Implementation for {hardware} hardware platform.",
                "optimizations": [
                    "Standard optimizations for this hardware platform"
                ],
                "requirements": [
                    "Appropriate hardware and software support"
                ],
                "limitations": "Specific limitations not documented.",
                "docs_url": "#"
            }

# Try to import DB integration
try:
    sys.path.append(os.path.join(test_dir, "../duckdb_api"))
    from data.duckdb.core.benchmark_db_updater import store_test_result, initialize_db
    HAS_DB_API = True
except ImportError:
    HAS_DB_API = False
    logger.warning("DuckDB API modules not available. Using basic file storage only.")

# Constants
RESULTS_ROOT = os.path.abspath(os.path.join(script_dir, "../../"))
EXPECTED_RESULTS_DIR = os.path.join(RESULTS_ROOT, "expected_results")
COLLECTED_RESULTS_DIR = os.path.join(RESULTS_ROOT, "collected_results")
DOCS_DIR = os.path.join(RESULTS_ROOT, "model_documentation")
TEMPLATE_DB_PATH = os.path.join(test_dir, "../duckdb_api/template_db.duckdb")
TEST_TIMEOUT = 300  # seconds
DEFAULT_DB_PATH = os.environ.get("BENCHMARK_DB_PATH", os.path.join(test_dir, "benchmark_db.duckdb"))

# Ensure directories exist
for directory in [EXPECTED_RESULTS_DIR, COLLECTED_RESULTS_DIR, DOCS_DIR]:
    ensure_dir_exists(directory)

# Define model families and hardware platforms
MODEL_FAMILIES = {
    "text-embedding": ["bert-base-uncased", "bert-large-uncased", "sentence-transformers/all-MiniLM-L6-v2"],
    "text-generation": ["facebook/opt-125m", "google/flan-t5-small", "tiiuae/falcon-7b"],
    "vision": ["google/vit-base-patch16-224", "facebook/detr-resnet-50", "openai/clip-vit-base-patch32"],
    "audio": ["openai/whisper-tiny", "facebook/wav2vec2-base", "laion/clap-htsat-unfused"],
    "multimodal": ["openai/clip-vit-base-patch32", "llava-hf/llava-1.5-7b-hf", "facebook/flava-full"]
}

SUPPORTED_HARDWARE = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
PRIORITY_HARDWARE = ["cpu", "cuda", "openvino", "webgpu"]  # Hardware platforms to prioritize in testing

class IntegratedComponentTester:
    """
    Integrated Component Tester for generating and testing skill, test, and benchmark components together.
    """
    
    def __init__(self, 
                 model_name: str,
                 hardware: str,
                 db_path: Optional[str] = None,
                 template_db_path: Optional[str] = None,
                 update_expected: bool = False,
                 generate_docs: bool = False,
                 quick_test: bool = False,
                 keep_temp: bool = False,
                 verbose: bool = False,
                 tolerance: float = 0.01,
                 git_hash: Optional[str] = None):
        """
        Initialize the integrated component tester.
        
        Args:
            model_name: Name of the model to test
            hardware: Hardware platform to test on
            db_path: Path to DuckDB database for storing results
            template_db_path: Path to template database
            update_expected: Whether to update expected results with current test results
            generate_docs: Whether to generate documentation for the model
            quick_test: Whether to run a quick test with minimal validation
            keep_temp: Whether to keep temporary directories after tests
            verbose: Whether to output verbose logs
            tolerance: Tolerance for numeric comparisons (e.g., 0.01 for 1%)
            git_hash: Current git commit hash (for versioning)
        """
        self.model_name = model_name
        self.hardware = hardware
        self.db_path = db_path or DEFAULT_DB_PATH
        self.template_db_path = template_db_path
        self.update_expected = update_expected
        self.generate_docs = generate_docs
        self.quick_test = quick_test
        self.keep_temp = keep_temp
        self.verbose = verbose
        self.tolerance = tolerance
        self.git_hash = git_hash or self._get_git_hash()
        
        # Set up logging
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            
        # Model validator for checking templates
        self.model_validator = ModelValidator(model_name, hardware, verbose=verbose)
        
    def _get_git_hash(self) -> str:
        """Get the current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Failed to get git hash. Using timestamp instead.")
            return f"unknown-{int(time.time())}"
    
    def generate_components(self, temp_dir: str) -> Tuple[str, str, str]:
        """
        Generate skill, test, and benchmark components together.
        
        Args:
            temp_dir: Directory to store generated files
            
        Returns:
            Tuple of paths to generated (skill_file, test_file, benchmark_file)
        """
        logger.info(f"Generating components for {self.model_name} on {self.hardware}")
        
        # Create paths for generated files
        model_name_safe = self.model_name.replace('/', '_')
        skill_file = os.path.join(temp_dir, f"{model_name_safe}_{self.hardware}_skill.py")
        test_file = os.path.join(temp_dir, f"test_{model_name_safe}_{self.hardware}.py")
        benchmark_file = os.path.join(temp_dir, f"benchmark_{model_name_safe}_{self.hardware}.py")
        
        try:
            # Try to use the template database and renderer
            from template_database import TemplateDatabase, add_default_templates
            from template_renderer import TemplateRenderer
            
            # Initialize template database if it doesn't exist
            if not os.path.exists(self.template_db_path):
                logger.info(f"Initializing template database at {self.template_db_path}")
                add_default_templates(self.template_db_path)
            
            # Create renderer
            renderer = TemplateRenderer(db_path=self.template_db_path, verbose=self.verbose)
            
            # Basic batch size settings
            batch_sizes = [1] if self.quick_test else [1, 2, 4, 8]
            
            # Create custom variables
            variables = {
                "batch_size": batch_sizes[0],
                "batch_sizes": batch_sizes,
                "git_hash": self.git_hash,
                "test_id": str(uuid.uuid4()),
                "test_timestamp": datetime.datetime.now().isoformat()
            }
            
            # Generate all components at once
            logger.debug("Generating components using template renderer")
            generated_files = renderer.render_component_set(
                model_name=self.model_name,
                hardware_platform=self.hardware,
                variables=variables,
                output_dir=temp_dir
            )
            
            # Verify the generated files exist
            if "skill" in generated_files and os.path.exists(skill_file):
                logger.info(f"Generated skill file: {skill_file}")
            else:
                # Fall back to legacy template method
                logger.warning("Template renderer didn't generate skill file, falling back to legacy method")
                skill_template = self._get_template("skill", self.model_name, self.hardware)
                skill_content = self._render_template(skill_template, self.model_name, self.hardware)
                with open(skill_file, 'w') as f:
                    f.write(skill_content)
                
            if "test" in generated_files and os.path.exists(test_file):
                logger.info(f"Generated test file: {test_file}")
            else:
                # Fall back to legacy template method
                logger.warning("Template renderer didn't generate test file, falling back to legacy method")
                test_template = self._get_template("test", self.model_name, self.hardware)
                test_content = self._render_template(test_template, self.model_name, self.hardware)
                with open(test_file, 'w') as f:
                    f.write(test_content)
                
            if "benchmark" in generated_files and os.path.exists(benchmark_file):
                logger.info(f"Generated benchmark file: {benchmark_file}")
            else:
                # Fall back to legacy template method
                logger.warning("Template renderer didn't generate benchmark file, falling back to legacy method")
                benchmark_template = self._get_template("benchmark", self.model_name, self.hardware)
                benchmark_content = self._render_template(benchmark_template, self.model_name, self.hardware)
                with open(benchmark_file, 'w') as f:
                    f.write(benchmark_content)
            
        except Exception as e:
            # Fall back to legacy template generation if the new method fails
            logger.warning(f"Error using template renderer: {e}. Falling back to legacy template method.")
            
            # Legacy template method
            skill_template = self._get_template("skill", self.model_name, self.hardware)
            test_template = self._get_template("test", self.model_name, self.hardware)
            benchmark_template = self._get_template("benchmark", self.model_name, self.hardware)
            
            # Render templates with model and hardware information
            skill_content = self._render_template(skill_template, self.model_name, self.hardware)
            test_content = self._render_template(test_template, self.model_name, self.hardware)
            benchmark_content = self._render_template(benchmark_template, self.model_name, self.hardware)
            
            # Write rendered templates to files
            with open(skill_file, 'w') as f:
                f.write(skill_content)
                
            with open(test_file, 'w') as f:
                f.write(test_content)
                
            with open(benchmark_file, 'w') as f:
                f.write(benchmark_content)
        
        # Validate generated files
        logger.debug("Validating generated files")
        skill_validation = self.model_validator.validate_skill(skill_file)
        test_validation = self.model_validator.validate_test(test_file)
        benchmark_validation = self.model_validator.validate_benchmark(benchmark_file)
        
        if not all([skill_validation.get("valid", False),
                   test_validation.get("valid", False),
                   benchmark_validation.get("valid", False)]):
            logger.warning("Validation failed for some components:")
            if not skill_validation.get("valid", False):
                logger.warning(f"Skill validation: {skill_validation.get('error', 'Unknown error')}")
            if not test_validation.get("valid", False):
                logger.warning(f"Test validation: {test_validation.get('error', 'Unknown error')}")
            if not benchmark_validation.get("valid", False):
                logger.warning(f"Benchmark validation: {benchmark_validation.get('error', 'Unknown error')}")
        
        return skill_file, test_file, benchmark_file
    
    def _render_template(self, template_content: str, model_name: str, hardware: str) -> str:
        """
        Render a template with model and hardware information.
        
        Args:
            template_content: Template content with placeholders
            model_name: Model name to substitute
            hardware: Hardware platform to substitute
            
        Returns:
            Rendered template content
        """
        # Get timestamp for the template
        timestamp = datetime.datetime.now().isoformat()
        
        # Replace placeholders
        rendered = template_content
        rendered = rendered.replace("{model_name}", model_name)
        rendered = rendered.replace("{hardware}", hardware)
        rendered = rendered.replace("{timestamp}", timestamp)
        
        return rendered
    
    def _get_template(self, template_type: str, model_name: str, hardware: str) -> str:
        """
        Get a template from the database or template files.
        
        Args:
            template_type: Type of template (skill, test, benchmark)
            model_name: Model name
            hardware: Hardware platform
            
        Returns:
            Template content as string
        """
        # In a real implementation, this would query the template database
        # For this example, we'll return placeholder templates
        
        # Determine model type based on name
        model_type = "text_embedding"  # Default
        if "vit" in model_name.lower() or "clip" in model_name.lower():
            model_type = "vision"
        elif "whisper" in model_name.lower() or "wav2vec" in model_name.lower():
            model_type = "audio"
        elif "opt" in model_name.lower() or "flan" in model_name.lower():
            model_type = "text_generation"
            
        # Basic placeholder templates
        if template_type == "skill":
            return f"""#!/usr/bin/env python3
'''
Skill implementation for {model_name} on {hardware} hardware.
Generated by integrated component test runner.
'''

import torch
import numpy as np
from typing import Dict, Any, List, Union

class {model_name.replace('-', '_').replace('/', '_').title()}Skill:
    '''
    Model skill for {model_name} on {hardware} hardware.
    Model type: {model_type}
    '''
    
    def __init__(self):
        self.model_name = "{model_name}"
        self.hardware = "{hardware}"
        self.model_type = "{model_type}"
        self.model = None
        self.tokenizer = None
        
    def setup(self, **kwargs) -> bool:
        '''Set up the model and tokenizer.'''
        try:
            if self.hardware == "cpu":
                # CPU setup logic
                device = "cpu"
            elif self.hardware == "cuda":
                # CUDA setup logic
                device = "cuda" if torch.cuda.is_available() else "cpu"
            # Additional hardware platforms...
            
            # Common setup logic
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(device)
            self.model.eval()
            
            return True
        except Exception as e:
            print(f"Error setting up model: {{e}}")
            return False
    
    def run(self, inputs: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        '''Run the model on inputs.'''
        try:
            if isinstance(inputs, str):
                inputs = [inputs]
                
            # Tokenize inputs
            encoded_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
            encoded_inputs = {{k: v.to(self.model.device) for k, v in encoded_inputs.items()}}
            
            # Run model
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                
            # Process outputs
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return {{
                "embeddings": embeddings,
                "shape": embeddings.shape,
                "norm": np.linalg.norm(embeddings, axis=1).tolist()
            }}
        except Exception as e:
            print(f"Error running model: {{e}}")
            return {{"error": str(e)}}
            
    def cleanup(self) -> bool:
        '''Clean up resources.'''
        try:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"Error cleaning up: {{e}}")
            return False
"""
        elif template_type == "test":
            return f"""#!/usr/bin/env python3
'''
Test for {model_name} on {hardware} hardware.
Generated by integrated component test runner.
'''

import os
import sys
import unittest
import numpy as np
from typing import Dict, Any, List

# Add parent directory to path so we can import the skill
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the skill module (using the safe model name)
from {model_name.replace('-', '_').replace('/', '_').lower()}_{hardware}_skill import {model_name.replace('-', '_').replace('/', '_').title()}Skill

class Test{model_name.replace('-', '_').replace('/', '_').title()}(unittest.TestCase):
    '''
    Test case for {model_name} on {hardware} hardware.
    '''
    
    @classmethod
    def setUpClass(cls):
        '''Set up the skill and test environment.'''
        cls.skill = {model_name.replace('-', '_').replace('/', '_').title()}Skill()
        cls.setup_success = cls.skill.setup()
        
    @classmethod
    def tearDownClass(cls):
        '''Clean up after tests.'''
        cls.skill.cleanup()
    
    def test_setup(self):
        '''Test that setup was successful.'''
        self.assertTrue(self.setup_success, "Model setup failed")
        
    def test_run_with_single_input(self):
        '''Test running the model with a single input.'''
        # Skip if setup failed
        if not self.setup_success:
            self.skipTest("Setup failed, skipping test_run_with_single_input")
            
        # Run the model
        test_input = "This is a test input for {model_name}."
        result = self.skill.run(test_input)
        
        # Check result structure
        self.assertIn("embeddings", result, "Result missing 'embeddings' key")
        self.assertIn("shape", result, "Result missing 'shape' key")
        self.assertIn("norm", result, "Result missing 'norm' key")
        
        # Check embeddings
        self.assertIsInstance(result["embeddings"], np.ndarray, "Embeddings should be a numpy array")
        self.assertEqual(len(result["shape"]), 2, "Embeddings should be 2D")
        self.assertEqual(result["shape"][0], 1, "Batch size should be 1 for single input")
        
        # Check norms are reasonable (non-zero)
        self.assertGreater(result["norm"][0], 0, "Embedding norm should be positive")
        
    def test_run_with_batch_input(self):
        '''Test running the model with a batch of inputs.'''
        # Skip if setup failed
        if not self.setup_success:
            self.skipTest("Setup failed, skipping test_run_with_batch_input")
            
        # Run the model
        test_inputs = [
            "This is the first test input for {model_name}.",
            "This is the second test input for {model_name}."
        ]
        result = self.skill.run(test_inputs)
        
        # Check result structure
        self.assertIn("embeddings", result, "Result missing 'embeddings' key")
        self.assertIn("shape", result, "Result missing 'shape' key")
        self.assertIn("norm", result, "Result missing 'norm' key")
        
        # Check embeddings
        self.assertIsInstance(result["embeddings"], np.ndarray, "Embeddings should be a numpy array")
        self.assertEqual(len(result["shape"]), 2, "Embeddings should be 2D")
        self.assertEqual(result["shape"][0], 2, "Batch size should be 2 for two inputs")
        
        # Check norms are reasonable (non-zero)
        self.assertGreater(result["norm"][0], 0, "First embedding norm should be positive")
        self.assertGreater(result["norm"][1], 0, "Second embedding norm should be positive")

if __name__ == "__main__":
    unittest.main()
"""
        elif template_type == "benchmark":
            return f"""#!/usr/bin/env python3
'''
Benchmark for {model_name} on {hardware} hardware.
Generated by integrated component test runner.
'''

import os
import sys
import time
import json
import argparse
import numpy as np
from typing import Dict, Any, List

# Add parent directory to path so we can import the skill
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the skill module (using the safe model name)
from {model_name.replace('-', '_').replace('/', '_').lower()}_{hardware}_skill import {model_name.replace('-', '_').replace('/', '_').title()}Skill

def run_benchmark(batch_sizes=[1, 2, 4, 8], iterations=10, warmup=3, save_results=True, output_dir=None):
    '''
    Run benchmark for {model_name} on {hardware} hardware.
    
    Args:
        batch_sizes: List of batch sizes to test
        iterations: Number of iterations per batch size
        warmup: Number of warmup iterations
        save_results: Whether to save results to file
        output_dir: Directory to save results (defaults to script directory)
    
    Returns:
        Dictionary with benchmark results
    '''
    print(f"Benchmarking {model_name} on {hardware} hardware")
    print(f"Batch sizes: {{batch_sizes}}")
    print(f"Iterations: {{iterations}} (with {{warmup}} warmup iterations)")
    
    # Create and set up the skill
    skill = {model_name.replace('-', '_').replace('/', '_').title()}Skill()
    setup_success = skill.setup()
    
    if not setup_success:
        print("Failed to set up model. Aborting benchmark.")
        return {{"error": "Failed to set up model"}}
    
    # Prepare benchmark results
    results = {{
        "model_name": "{model_name}",
        "hardware": "{hardware}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch_sizes": batch_sizes,
        "iterations": iterations,
        "warmup": warmup,
        "results_by_batch": {{}}
    }}
    
    try:
        # Generate sample inputs for largest batch size
        base_input = "This is a test input for {model_name}."
        max_batch_size = max(batch_sizes)
        sample_inputs = [f"{{base_input}} Sample {{i}}." for i in range(max_batch_size)]
        
        # Run benchmarks for each batch size
        for batch_size in batch_sizes:
            batch_inputs = sample_inputs[:batch_size]
            batch_results = {{
                "latency_ms": [],
                "throughput_items_per_second": [],
            }}
            
            print(f"\\nBenchmarking batch size {{batch_size}}...")
            
            # Warmup
            for _ in range(warmup):
                _ = skill.run(batch_inputs)
            
            # Benchmark iterations
            for i in range(iterations):
                start_time = time.time()
                _ = skill.run(batch_inputs)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                throughput = batch_size / (end_time - start_time)
                
                batch_results["latency_ms"].append(latency_ms)
                batch_results["throughput_items_per_second"].append(throughput)
                
                print(f"  Iteration {{i+1}}: Latency {{latency_ms:.2f}} ms, Throughput {{throughput:.2f}} items/s")
            
            # Calculate statistics
            batch_results["average_latency_ms"] = np.mean(batch_results["latency_ms"])
            batch_results["min_latency_ms"] = np.min(batch_results["latency_ms"])
            batch_results["max_latency_ms"] = np.max(batch_results["latency_ms"])
            batch_results["stddev_latency_ms"] = np.std(batch_results["latency_ms"])
            
            batch_results["average_throughput_items_per_second"] = np.mean(batch_results["throughput_items_per_second"])
            batch_results["min_throughput_items_per_second"] = np.min(batch_results["throughput_items_per_second"])
            batch_results["max_throughput_items_per_second"] = np.max(batch_results["throughput_items_per_second"])
            batch_results["stddev_throughput_items_per_second"] = np.std(batch_results["throughput_items_per_second"])
            
            # Store results for this batch size
            results["results_by_batch"][str(batch_size)] = batch_results
            
            print(f"  Average latency: {{batch_results['average_latency_ms']:.2f}} ms")
            print(f"  Average throughput: {{batch_results['average_throughput_items_per_second']:.2f}} items/s")
        
        # Clean up
        skill.cleanup()
        
        # Save results if requested
        if save_results:
            output_dir = output_dir or script_dir
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"benchmark_{model_name.replace('/', '_')}_{hardware}_{timestamp}.json")
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"\\nResults saved to {{output_file}}")
        
        return results
        
    except Exception as e:
        print(f"Error during benchmark: {{e}}")
        # Clean up in case of error
        skill.cleanup()
        return {{"error": str(e)}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Benchmark for {model_name} on {hardware} hardware")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8", help="Comma-separated list of batch sizes")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per batch size")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save results")
    
    args = parser.parse_args()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    
    run_benchmark(
        batch_sizes=batch_sizes,
        iterations=args.iterations,
        warmup=args.warmup,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
"""
        else:
            return f"# Template not found for {template_type}, {model_name}, {hardware}"
    
    def _render_template(self, template: str, model_name: str, hardware: str) -> str:
        """
        Render a template with model and hardware information.
        
        Args:
            template: Template content
            model_name: Model name
            hardware: Hardware platform
            
        Returns:
            Rendered template as string
        """
        # In a real implementation, this would use a proper templating engine
        # For this example, we'll assume the template already has placeholders replaced
        return template
    
    def run_test(self, skill_file: str, test_file: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Run the test for the model on the given hardware.
        
        Args:
            skill_file: Path to the skill file
            test_file: Path to the test file
            
        Returns:
            Tuple of (success, results)
        """
        logger.info(f"Running test for {self.model_name} on {self.hardware}")
        
        # In a real implementation, this would execute the test file as a subprocess
        # For this example, we'll simulate the process
        
        # Run the test and capture output
        try:
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                check=False,
                timeout=TEST_TIMEOUT
            )
            
            logger.debug(f"Test stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"Test stderr: {result.stderr}")
                
            success = result.returncode == 0
            
            # Parse test results
            test_results = {
                "success": success,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": None,  # Would normally parse from output
            }
            
            # Extract execution time from output (simplified)
            import re
            time_match = re.search(r'Ran (\d+) tests? in ([\d.]+)s', result.stdout)
            if time_match:
                test_count = int(time_match.group(1))
                execution_time = float(time_match.group(2))
                test_results["test_count"] = test_count
                test_results["execution_time"] = execution_time
            
            if success:
                logger.info(f"Test passed for {self.model_name} on {self.hardware}")
            else:
                logger.warning(f"Test failed for {self.model_name} on {self.hardware}")
                
            return success, test_results
            
        except subprocess.TimeoutExpired:
            logger.error(f"Test timed out after {TEST_TIMEOUT} seconds")
            return False, {
                "success": False,
                "return_code": None,
                "stdout": None,
                "stderr": f"Test timed out after {TEST_TIMEOUT} seconds",
                "execution_time": TEST_TIMEOUT
            }
        except Exception as e:
            logger.error(f"Error running test: {e}")
            return False, {
                "success": False,
                "return_code": None,
                "stdout": None,
                "stderr": str(e),
                "execution_time": None
            }
    
    def run_benchmark(self, benchmark_file: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Run the benchmark for the model on the given hardware.
        
        Args:
            benchmark_file: Path to the benchmark file
            
        Returns:
            Tuple of (success, results)
        """
        logger.info(f"Running benchmark for {self.model_name} on {self.hardware}")
        
        # Define batch sizes based on quick_test flag
        batch_sizes = "1,2" if self.quick_test else "1,2,4,8"
        iterations = 3 if self.quick_test else 10
        warmup = 1 if self.quick_test else 3
        
        # Run the benchmark and capture output
        try:
            result = subprocess.run(
                [sys.executable, benchmark_file, 
                 "--batch-sizes", batch_sizes,
                 "--iterations", str(iterations),
                 "--warmup", str(warmup),
                 "--output-dir", os.path.dirname(benchmark_file)],
                capture_output=True,
                text=True,
                check=False,
                timeout=TEST_TIMEOUT * 2  # Longer timeout for benchmarks
            )
            
            logger.debug(f"Benchmark stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"Benchmark stderr: {result.stderr}")
                
            success = result.returncode == 0
            
            # Find and load the output JSON file
            benchmark_results = None
            if success:
                # Parse benchmark results from output file
                output_dir = os.path.dirname(benchmark_file)
                json_files = [f for f in os.listdir(output_dir) if f.startswith(f"benchmark_{self.model_name.replace('/', '_')}_{self.hardware}") and f.endswith(".json")]
                if json_files:
                    # Sort by timestamp (newest first)
                    json_files.sort(reverse=True)
                    latest_file = os.path.join(output_dir, json_files[0])
                    
                    with open(latest_file, 'r') as f:
                        benchmark_results = json.load(f)
                        
                        # Add metadata
                        benchmark_results["git_hash"] = self.git_hash
                        benchmark_results["timestamp"] = datetime.datetime.now().isoformat()
                        benchmark_results["run_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
            
            if success and benchmark_results:
                logger.info(f"Benchmark successful for {self.model_name} on {self.hardware}")
            else:
                logger.warning(f"Benchmark failed for {self.model_name} on {self.hardware}")
                
            return success, benchmark_results or {
                "success": False,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out after {TEST_TIMEOUT * 2} seconds")
            return False, {
                "success": False,
                "error": f"Benchmark timed out after {TEST_TIMEOUT * 2} seconds"
            }
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            return False, {
                "success": False,
                "error": str(e)
            }
    
    def save_results(self, 
                    test_success: bool, 
                    test_results: Dict[str, Any],
                    benchmark_success: bool, 
                    benchmark_results: Dict[str, Any],
                    components: Tuple[str, str, str]) -> str:
        """
        Save test and benchmark results to files and database.
        
        Args:
            test_success: Whether the test was successful
            test_results: Test results dictionary
            benchmark_success: Whether the benchmark was successful
            benchmark_results: Benchmark results dictionary
            components: Tuple of paths to (skill_file, test_file, benchmark_file)
            
        Returns:
            Path to the collected results directory
        """
        logger.info(f"Saving results for {self.model_name} on {self.hardware}")
        
        # Create timestamp for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories for results
        model_dir = os.path.join(COLLECTED_RESULTS_DIR, self.model_name.replace('/', '_'))
        hardware_dir = os.path.join(model_dir, self.hardware)
        results_dir = os.path.join(hardware_dir, timestamp)
        
        ensure_dir_exists(model_dir)
        ensure_dir_exists(hardware_dir)
        ensure_dir_exists(results_dir)
        
        # Copy the component files to the results directory
        skill_file, test_file, benchmark_file = components
        shutil.copy(skill_file, os.path.join(results_dir, os.path.basename(skill_file)))
        shutil.copy(test_file, os.path.join(results_dir, os.path.basename(test_file)))
        shutil.copy(benchmark_file, os.path.join(results_dir, os.path.basename(benchmark_file)))
        
        # Save test results
        test_output_file = os.path.join(results_dir, "test_results.json")
        with open(test_output_file, 'w') as f:
            json.dump({
                "model_name": self.model_name,
                "hardware": self.hardware,
                "success": test_success,
                "results": test_results,
                "timestamp": datetime.datetime.now().isoformat(),
                "git_hash": self.git_hash
            }, f, indent=2)
            
        # Save benchmark results (if available)
        if benchmark_success and isinstance(benchmark_results, dict) and "results_by_batch" in benchmark_results:
            benchmark_output_file = os.path.join(results_dir, "benchmark_results.json")
            with open(benchmark_output_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
                
        # Save combined results
        combined_output_file = os.path.join(results_dir, "combined_results.json")
        with open(combined_output_file, 'w') as f:
            json.dump({
                "model_name": self.model_name,
                "hardware": self.hardware,
                "timestamp": datetime.datetime.now().isoformat(),
                "git_hash": self.git_hash,
                "test": {
                    "success": test_success,
                    "results": test_results
                },
                "benchmark": {
                    "success": benchmark_success,
                    "results": benchmark_results if isinstance(benchmark_results, dict) else {}
                }
            }, f, indent=2)
            
        # Update expected results if requested
        if self.update_expected:
            expected_model_dir = os.path.join(EXPECTED_RESULTS_DIR, self.model_name.replace('/', '_'))
            expected_hardware_dir = os.path.join(expected_model_dir, self.hardware)
            
            ensure_dir_exists(expected_model_dir)
            ensure_dir_exists(expected_hardware_dir)
            
            expected_output_file = os.path.join(expected_hardware_dir, "expected_results.json")
            shutil.copy(combined_output_file, expected_output_file)
            logger.info(f"Updated expected results for {self.model_name} on {self.hardware}")
            
        # Store results in database if available
        if HAS_DB_API and HAS_DUCKDB and self.db_path:
            try:
                # Initialize the database if it doesn't exist
                initialize_db(self.db_path)
                
                # Store test results
                store_test_result(
                    self.db_path,
                    model_name=self.model_name,
                    hardware=self.hardware,
                    test_success=test_success,
                    test_results=test_results,
                    benchmark_success=benchmark_success,
                    benchmark_results=benchmark_results,
                    git_hash=self.git_hash
                )
                logger.info(f"Stored results in database: {self.db_path}")
            except Exception as e:
                logger.error(f"Error storing results in database: {e}")
                
        return results_dir
    
    def compare_with_expected(self, 
                             test_results: Dict[str, Any],
                             benchmark_results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare test and benchmark results with expected results.
        
        Args:
            test_results: Test results dictionary
            benchmark_results: Benchmark results dictionary
            
        Returns:
            Tuple of (success, comparison_results)
        """
        logger.info(f"Comparing results with expected for {self.model_name} on {self.hardware}")
        
        # Check if expected results exist
        expected_model_dir = os.path.join(EXPECTED_RESULTS_DIR, self.model_name.replace('/', '_'))
        expected_hardware_dir = os.path.join(expected_model_dir, self.hardware)
        expected_output_file = os.path.join(expected_hardware_dir, "expected_results.json")
        
        if not os.path.exists(expected_output_file):
            logger.warning(f"No expected results found for {self.model_name} on {self.hardware}")
            return False, {
                "success": False,
                "reason": "No expected results found",
                "expected_file": expected_output_file
            }
            
        # Load expected results
        try:
            with open(expected_output_file, 'r') as f:
                expected_results = json.load(f)
        except Exception as e:
            logger.error(f"Error loading expected results: {e}")
            return False, {
                "success": False,
                "reason": f"Error loading expected results: {e}",
                "expected_file": expected_output_file
            }
            
        # Use ResultComparer for detailed comparison
        result_comparer = ResultComparer(
            model_name=self.model_name,
            hardware=self.hardware,
            tolerance=self.tolerance,
            verbose=self.verbose
        )
        
        comparison_results = result_comparer.compare(
            actual={
                "test": {
                    "success": test_results.get("success", False),
                    "results": test_results
                },
                "benchmark": {
                    "success": "results_by_batch" in benchmark_results,
                    "results": benchmark_results
                }
            },
            expected=expected_results
        )
        
        # Return comparison success and results
        return comparison_results.get("success", False), comparison_results
    
    def generate_documentation(self, 
                              skill_file: str, 
                              test_file: str, 
                              benchmark_file: str,
                              test_results: Dict[str, Any],
                              benchmark_results: Dict[str, Any]) -> str:
        """
        Generate documentation for the model implementation.
        
        Args:
            skill_file: Path to the skill file
            test_file: Path to the test file
            benchmark_file: Path to the benchmark file
            test_results: Test results dictionary
            benchmark_results: Benchmark results dictionary
            
        Returns:
            Path to the generated documentation file
        """
        logger.info(f"Generating documentation for {self.model_name} on {self.hardware}")
        
        # Create model documentation directory
        model_docs_dir = os.path.join(DOCS_DIR, self.model_name.replace('/', '_'))
        ensure_dir_exists(model_docs_dir)
        
        # Generate documentation
        doc_file = os.path.join(model_docs_dir, f"{self.hardware}_implementation.md")
        
        # Try to use template system for documentation if template_db_path is available
        if hasattr(self, 'template_db_path') and self.template_db_path and os.path.exists(self.template_db_path):
            try:
                # Check if we have the updated model_documentation_generator that supports templates
                sig = inspect.signature(generate_model_documentation)
                
                # Check if the function supports template_db_path parameter (new version)
                if 'template_db_path' in sig.parameters:
                    # Use the enhanced documentation generator with template support
                    logger.info(f"Using template system for documentation from {self.template_db_path}")
                    expected_results_path = None
                    
                    # Try to find expected results if available
                    model_dir = self.model_name.replace('/', '_')
                    for ext in ['.json', '.jsonl', '.npz', '.npy']:
                        expected_path = os.path.join(EXPECTED_RESULTS_DIR, model_dir, f"{self.hardware}{ext}")
                        if os.path.exists(expected_path):
                            expected_results_path = expected_path
                            break
                    
                    # Generate documentation with template system
                    doc_path = generate_model_documentation(
                        model_name=self.model_name,
                        hardware=self.hardware,
                        skill_path=skill_file,
                        test_path=test_file,
                        benchmark_path=benchmark_file,
                        expected_results_path=expected_results_path,
                        output_dir=model_docs_dir,
                        template_db_path=self.template_db_path
                    )
                    logger.info(f"Documentation generated with template system: {doc_path}")
                    return doc_path
                else:
                    # The function exists but doesn't support template_db_path (old version)
                    logger.warning("Documentation generator doesn't support template system parameters")
            except Exception as e:
                logger.warning(f"Error using template system for documentation: {e}")
                logger.info("Falling back to enhanced documentation generator")
        
        # Create an enhanced doc generator that includes HuggingFace documentation
        doc_generator = EnhancedModelDocGenerator(
            model_name=self.model_name,
            hardware=self.hardware,
            skill_path=skill_file,
            test_path=test_file,
            benchmark_path=benchmark_file,
            output_dir=model_docs_dir,
            verbose=self.verbose
        )
        
        # Generate the documentation
        doc_content = doc_generator.generate_markdown(
            test_results=test_results,
            benchmark_results=benchmark_results,
            git_hash=self.git_hash
        )
        
        # Write to file
        with open(doc_file, 'w') as f:
            f.write(doc_content)
            
        logger.info(f"Documentation generated with EnhancedModelDocGenerator: {doc_file}")
        return doc_file
    
    def run(self) -> Dict[str, Any]:
        """
        Run the integrated component test.
        
        Returns:
            Dictionary with test results
        """
        start_time = time.time()
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temporary directory: {temp_dir}")
        
        try:
            # Step 1: Generate components
            components = self.generate_components(temp_dir)
            skill_file, test_file, benchmark_file = components
            
            # Step 2: Run test
            test_success, test_results = self.run_test(skill_file, test_file)
            
            # Step 3: Run benchmark
            benchmark_success, benchmark_results = self.run_benchmark(benchmark_file)
            
            # Step 4: Save results
            results_dir = self.save_results(
                test_success, test_results,
                benchmark_success, benchmark_results,
                components
            )
            
            # Step 5: Compare with expected results (if not updating)
            comparison_success, comparison_results = False, {}
            if not self.update_expected:
                comparison_success, comparison_results = self.compare_with_expected(
                    test_results, benchmark_results
                )
                
            # Step 6: Generate documentation (if requested)
            doc_file = None
            if self.generate_docs:
                doc_file = self.generate_documentation(
                    skill_file, test_file, benchmark_file,
                    test_results, benchmark_results
                )
                
            # Calculate total execution time
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Prepare final results
            final_results = {
                "model_name": self.model_name,
                "hardware": self.hardware,
                "test_success": test_success,
                "benchmark_success": benchmark_success,
                "comparison_success": comparison_success,
                "results_dir": results_dir,
                "documentation": doc_file,
                "execution_time": execution_time,
                "timestamp": datetime.datetime.now().isoformat(),
                "git_hash": self.git_hash,
                "update_expected": self.update_expected,
                "components": {
                    "skill_file": skill_file,
                    "test_file": test_file,
                    "benchmark_file": benchmark_file
                }
            }
            
            logger.info(f"Test completed in {execution_time:.2f} seconds")
            
            return final_results
            
        finally:
            # Clean up temporary directory
            if not self.keep_temp:
                logger.debug(f"Removing temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Integrated Component Test Runner for IPFS Accelerate")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="Specific model to test")
    model_group.add_argument("--model-family", type=str, 
                            help="Model family to test (e.g., text-embedding, vision)")
    model_group.add_argument("--all-models", action="store_true", help="Test all supported models")
    
    # Hardware selection
    hardware_group = parser.add_mutually_exclusive_group(required=True)
    hardware_group.add_argument("--hardware", type=str, 
                               help="Hardware platforms to test, comma-separated (e.g., cpu,cuda,webgpu)")
    hardware_group.add_argument("--priority-hardware", action="store_true", 
                               help="Test on priority hardware platforms (cpu, cuda, openvino, webgpu)")
    hardware_group.add_argument("--all-hardware", action="store_true", 
                               help="Test on all supported hardware platforms")
    
    # Test options
    parser.add_argument("--quick-test", action="store_true", 
                        help="Run a quick test with minimal validation")
    parser.add_argument("--update-expected", action="store_true", 
                        help="Update expected results with current test results")
    parser.add_argument("--generate-docs", action="store_true", 
                        help="Generate markdown documentation for models")
    parser.add_argument("--keep-temp", action="store_true", 
                        help="Keep temporary directories after tests")
    
    # Database options
    parser.add_argument("--db-path", type=str, default=None, 
                        help="Path to the DuckDB database for storing results")
    parser.add_argument("--no-db", action="store_true", 
                        help="Disable database storage of results")
    
    # Template system options
    parser.add_argument("--template-db-path", type=str, default=None,
                        help="Path to the template database")
    parser.add_argument("--no-templates", action="store_true",
                        help="Disable template system usage")
    
    # Misc options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--tolerance", type=float, default=0.01, 
                        help="Set custom tolerance for numeric comparisons (e.g., 0.01 for 1%)")
    parser.add_argument("--workers", type=int, default=None, 
                        help="Number of worker threads for parallel testing")
    parser.add_argument("--clean-old-results", action="store_true", 
                        help="Clean up old collected results")
    parser.add_argument("--days", type=int, default=14, 
                        help="Number of days to keep results when cleaning (default: 14)")
    
    return parser.parse_args()

def get_model_list(args) -> List[str]:
    """Get the list of models to test based on command line arguments."""
    if args.model:
        return [args.model]
    elif args.model_family:
        if args.model_family in MODEL_FAMILIES:
            return MODEL_FAMILIES[args.model_family]
        else:
            logger.error(f"Unknown model family: {args.model_family}")
            logger.info(f"Available families: {', '.join(MODEL_FAMILIES.keys())}")
            return []
    elif args.all_models:
        # Flatten all model families
        return [model for family in MODEL_FAMILIES.values() for model in family]
    else:
        logger.error("No model specified")
        return []

def get_hardware_list(args) -> List[str]:
    """Get the list of hardware platforms to test based on command line arguments."""
    if args.hardware:
        return [h.strip() for h in args.hardware.split(",")]
    elif args.priority_hardware:
        return PRIORITY_HARDWARE
    elif args.all_hardware:
        return SUPPORTED_HARDWARE
    else:
        logger.error("No hardware specified")
        return []

def clean_old_results(days: int, collected_dir: str = COLLECTED_RESULTS_DIR):
    """
    Clean up old collected results.
    
    Args:
        days: Number of days to keep results
        collected_dir: Directory containing collected results
    """
    logger.info(f"Cleaning up results older than {days} days")
    
    # Calculate cutoff date
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Track cleaned directories
    cleaned_count = 0
    
    # Traverse the collected results directory
    for model_dir in os.listdir(collected_dir):
        model_path = os.path.join(collected_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        for hardware_dir in os.listdir(model_path):
            hardware_path = os.path.join(model_path, hardware_dir)
            if not os.path.isdir(hardware_path):
                continue
                
            for timestamp_dir in os.listdir(hardware_path):
                timestamp_path = os.path.join(hardware_path, timestamp_dir)
                if not os.path.isdir(timestamp_path):
                    continue
                    
                # Try to parse the timestamp from the directory name
                try:
                    # Format: YYYYMMDD_HHMMSS
                    timestamp = datetime.datetime.strptime(timestamp_dir, "%Y%m%d_%H%M%S")
                    
                    # Check if older than cutoff
                    if timestamp < cutoff_date:
                        logger.debug(f"Removing old result: {timestamp_path}")
                        shutil.rmtree(timestamp_path)
                        cleaned_count += 1
                except ValueError:
                    # Not a timestamp directory, skip
                    logger.debug(f"Skipping non-timestamp directory: {timestamp_path}")
    
    logger.info(f"Cleaned {cleaned_count} old result directories")

def run_parallel_tests(models: List[str], hardware_platforms: List[str], args, max_workers: int = None) -> Dict[str, Dict[str, Any]]:
    """
    Run tests in parallel using a thread pool.
    
    Args:
        models: List of models to test
        hardware_platforms: List of hardware platforms to test on
        args: Command line arguments
        max_workers: Maximum number of worker threads (None for default)
        
    Returns:
        Dictionary mapping {model: {hardware: results}}
    """
    if not max_workers:
        # Default to min(32, os.cpu_count() + 4)
        max_workers = min(32, (os.cpu_count() or 4) + 4)
        
    logger.info(f"Running parallel tests with {max_workers} workers")
    
    # Create a list of all (model, hardware) combinations
    combinations = [(model, hw) for model in models for hw in hardware_platforms]
    results = {}
    
    # Function to run a single test
    def run_test(model, hardware):
        logger.info(f"Starting test: {model} on {hardware}")
        tester = IntegratedComponentTester(
            model_name=model,
            hardware=hardware,
            db_path=None if args.no_db else args.db_path,
            template_db_path=None if args.no_templates else args.template_db_path,
            update_expected=args.update_expected,
            generate_docs=args.generate_docs,
            quick_test=args.quick_test,
            keep_temp=args.keep_temp,
            verbose=args.verbose,
            tolerance=args.tolerance
        )
        test_result = tester.run()
        return model, hardware, test_result
    
    # Run tests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_combo = {
            executor.submit(run_test, model, hw): (model, hw) 
            for model, hw in combinations
        }
        
        for future in concurrent.futures.as_completed(future_to_combo):
            model, hardware, test_result = future.result()
            logger.info(f"Completed test: {model} on {hardware}")
            
            # Store results
            if model not in results:
                results[model] = {}
            results[model][hardware] = test_result
    
    return results

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set up logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Clean old results if requested
    if args.clean_old_results:
        clean_old_results(args.days)
        return
    
    # Get models and hardware platforms to test
    models = get_model_list(args)
    hardware_platforms = get_hardware_list(args)
    
    logger.info(f"Testing {len(models)} models on {len(hardware_platforms)} hardware platforms")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Hardware: {', '.join(hardware_platforms)}")
    
    # Run tests in parallel if multiple combinations
    if len(models) * len(hardware_platforms) > 1 and args.workers != 1:
        results = run_parallel_tests(
            models=models,
            hardware_platforms=hardware_platforms,
            args=args,
            max_workers=args.workers
        )
    else:
        # Run tests sequentially
        results = {}
        for model in models:
            results[model] = {}
            for hardware in hardware_platforms:
                logger.info(f"Testing {model} on {hardware}")
                tester = IntegratedComponentTester(
                    model_name=model,
                    hardware=hardware,
                    db_path=None if args.no_db else args.db_path,
                    template_db_path=None if args.no_templates else args.template_db_path,
                    update_expected=args.update_expected,
                    generate_docs=args.generate_docs,
                    quick_test=args.quick_test,
                    keep_temp=args.keep_temp,
                    verbose=args.verbose,
                    tolerance=args.tolerance
                )
                results[model][hardware] = tester.run()
    
    # Print summary
    print("\nTest Summary:")
    print("=" * 80)
    
    success_count = 0
    failure_count = 0
    
    for model, hw_results in results.items():
        print(f"\nModel: {model}")
        for hardware, result in hw_results.items():
            if result.get("test_success", False) and result.get("benchmark_success", False):
                status = "✅ Success"
                success_count += 1
            else:
                status = "❌ Failure"
                failure_count += 1
                
            print(f"  {hardware}: {status} (Test: {'✅' if result.get('test_success', False) else '❌'}, "
                  f"Benchmark: {'✅' if result.get('benchmark_success', False) else '❌'})")
    
    print("\nOverall Summary:")
    print(f"  Total: {success_count + failure_count}")
    print(f"  Success: {success_count}")
    print(f"  Failure: {failure_count}")
    print("=" * 80)
    
    # Return exit code based on results
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())